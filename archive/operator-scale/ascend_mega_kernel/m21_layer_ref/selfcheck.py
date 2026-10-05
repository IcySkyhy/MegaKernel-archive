#!/usr/bin/env python3
"""Self-verification of the M39 reference harness (docs/17 §4 non-hollowness + §1 L0).

Sections
--------
A  HC op unit checks against vLLM's OWN torch references
   (vllm/tests/models/qwen4_exp/test_hc_ops.py:34-52,62-68,76-80,94-106,124-126)
B  HC mixer cross-check against the EXECUTED official torch reference
   (vllm/models/qwen4_exp/common/hyperconnection.py, loaded by file path so
   `vllm/__init__.py` never runs -- see ref/official.py)
C  MXFP4 dequant cross-check against this repo's numpy golden
   (R:tools/golden/moe_block_ref.py:63-68,92-99,175-180) on real checkpoint bytes
D  tensor contract: every checkpoint weight shape vs the docs/14 §2.3 table
E  non-hollowness: input sensitivity + state evolution + dump-non-zero
F  docs/14 §10 item 3: is the fused combine_norm's extra bf16 round observable?

Each check prints PASS/FAIL, the quantity measured and the tolerance used.
"""

from __future__ import annotations

import argparse
import os
import importlib.util
import sys
from pathlib import Path

import math

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# sha256 of `tools/weights/safetensors_reader.py` (M8) at the moment this verbatim
# copy was taken. `section_j` re-checks it; if upstream legitimately changes,
# re-copy the file and update this constant (docs/17 §3.4).
PINNED_READER_SHA256 = "72b93ca2f2727c68cad77d128c3bc9338921b9dd3311256b7ed4c226bc515abd"

# Optional CPU bound for a shared box: `M39_THREADS=4 python3 selfcheck.py`
if os.environ.get("M39_THREADS"):
    torch.set_num_threads(int(os.environ["M39_THREADS"]))

from compare_dumps import ulp_distance  # noqa: E402
from ref import hc as hcref  # noqa: E402
from ref import official  # noqa: E402
from ref.ckpt import LayerWeights  # noqa: E402
from ref.mxfp4 import dequant_mxfp4, dequant_mxfp4_torch  # noqa: E402

LOG: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    LOG.append(("PASS" if ok else "FAIL", name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def ulp(a: torch.Tensor, b: torch.Tensor, significant_only: bool = True) -> int:
    dt = {torch.bfloat16: "bfloat16", torch.float32: "float32", torch.float16: "float16"}[a.dtype]
    if a.numel() == 0:
        return 0
    u = ulp_distance(a.reshape(-1), b.reshape(-1), dt)
    if significant_only:
        # Same policy as compare_dumps.py: ulp on noise-scale values is
        # meaningless, so restrict to elements within 8 binades of the max |a|.
        scale = float(a.float().abs().max())
        thr = scale * (2.0**-8)
        sig = (a.float().abs() >= thr) | (b.float().abs() >= thr)
        sig = sig.reshape(-1)
        if bool(sig.any()):
            u = u[sig]
    return int(u.max())


# ---------------------------------------------------------------------------
def section_a():
    print("\n=== A. HC ops vs vLLM's own torch references (test_hc_ops.py) ===")
    HC, H, EPS = 4, 2560, 1e-6
    D = HC * H
    torch.manual_seed(0)
    x = torch.randn(3, D, dtype=torch.bfloat16)
    w = torch.randn(D, dtype=torch.bfloat16)

    # test_hc_ops.py:34-38
    got = hcref.grouped_gemma_rmsnorm(x, w, EPS, HC)
    g = x.float().unflatten(-1, (HC, H))
    exp = (g * torch.rsqrt(g.square().mean(-1, keepdim=True) + EPS)).flatten(-2) * (
        1.0 + w.float()
    )
    check("A1 grouped_gemma_rmsnorm", torch.equal(got, exp.to(torch.bfloat16)),
          f"{ulp(got, exp.to(torch.bfloat16))} ulp")

    # test_hc_ops.py:47-52
    gate = torch.randn(3, D, dtype=torch.bfloat16)
    got = hcref.hc_gate_mix(x, gate, HC)
    exp = (torch.sigmoid(gate.float().unflatten(-1, (HC, H)))
           * x.float().unflatten(-1, (HC, H))).mean(-2)
    check("A2 hc_gate_mix", torch.equal(got, exp.to(torch.bfloat16)),
          f"{ulp(got, exp.to(torch.bfloat16))} ulp")

    # test_hc_ops.py:62-68
    block = torch.randn(3, H, dtype=torch.bfloat16)
    res = torch.randn(3, D, dtype=torch.bfloat16)
    inj = torch.randn(3, HC, dtype=torch.bfloat16)
    got = hcref.hc_combine(res, block, inj, HC)
    iw = 2.0 * torch.sigmoid(inj.float() / HC)
    exp = res.float().unflatten(-1, (HC, H)) + block.float().unsqueeze(-2) * iw.unsqueeze(-1)
    check("A3 hc_combine", torch.equal(got, exp.flatten(-2).to(torch.bfloat16)),
          f"{ulp(got, exp.flatten(-2).to(torch.bfloat16))} ulp")

    # test_hc_ops.py:76-80 (unit injection must be EXACT)
    got = hcref.hc_combine(res, block, None, HC)
    exp = (res.unflatten(-1, (HC, H)) + block.unsqueeze(-2)).flatten(-2)
    check("A4 hc_combine unit injection bit-exact", torch.equal(got, exp),
          f"{ulp(got, exp)} ulp")

    # test_hc_ops.py:94-106
    w2 = torch.randn(D, dtype=torch.bfloat16)
    mat, normed = hcref.hc_combine_norm(res, block, inj, w2, EPS, HC)
    e = (res.float().unflatten(-1, (HC, H)) + block.float().unsqueeze(-2) * iw.unsqueeze(-1))
    e = e.flatten(-2).to(res.dtype)
    eg = e.float().unflatten(-1, (HC, H))
    en = (eg * torch.rsqrt(eg.square().mean(-1, keepdim=True) + EPS)).flatten(-2) * (
        1.0 + w2.float()
    )
    check("A5 hc_combine_norm materialized", torch.equal(mat, e), f"{ulp(mat, e)} ulp")
    check("A6 hc_combine_norm normalized", torch.equal(normed, en.to(torch.bfloat16)),
          f"{ulp(normed, en.to(torch.bfloat16))} ulp")

    # test_hc_ops.py:109-126 (unit injection: combine_norm == grouped_gemma_rmsnorm of bf16(h+e))
    hid = torch.randn(4, HC, H, dtype=torch.bfloat16)
    emb = torch.randn(4, H, dtype=torch.bfloat16)
    m2, n2 = hcref.hc_combine_norm(hid.flatten(1), emb, None, w2, EPS, HC)
    e2 = (hid + emb.unsqueeze(1)).flatten(1)
    check("A7 unit-injection combine_norm == gemma_rmsnorm(bf16(h+e))",
          torch.equal(m2, e2) and torch.equal(n2, hcref.grouped_gemma_rmsnorm(e2, w2, EPS, HC)),
          f"{ulp(m2, e2)} / {ulp(n2, hcref.grouped_gemma_rmsnorm(e2, w2, EPS, HC))} ulp")


# ---------------------------------------------------------------------------
def section_b(layer_idx: int, m: int):
    print(f"\n=== B. HC mixer vs EXECUTED official common/hyperconnection.py (layer {layer_idx}) ===")
    mod = official.import_common_hyperconnection()
    cfg_dict = LayerWeights(layer_idx).text_config()
    hc, hidden = cfg_dict["hc_count"], cfg_dict["hidden_size"]
    D = hc * hidden
    w = LayerWeights(layer_idx).hyper_connection("attn_hyper_connection")

    off = mod.GatedResidual(
        mod.HyperConnectionConfig(
            hc_count=hc, hidden_size=hidden, params_dtype=torch.bfloat16,
            hc_lowrank=cfg_dict["hc_lowrank"], rms_norm_eps=cfg_dict["rms_norm_eps"],
            hc_per_branch_norm=True,
        )
    )
    with torch.no_grad():
        off.hc_norm.weight.copy_(w.hc_norm_weight)
        off.input_mix_weight_down.weight.copy_(w.down_weight)
        off.input_mix_weight_up.weight.copy_(w.up_weight)
        off.block_inject_weight.weight.copy_(w.inject_weight)

    mine = hcref.GatedResidual(
        w.hc_norm_weight, w.down_weight, w.up_weight, w.inject_weight, hc_count=hc,
        eps=cfg_dict["rms_norm_eps"],
    )

    torch.manual_seed(1)
    h = torch.randn(m, D, dtype=torch.bfloat16) * 0.05
    block = torch.randn(m, hidden, dtype=torch.bfloat16) * 0.05

    def bf16_ulp_at(x: torch.Tensor) -> float:
        """One bf16 ulp at the tensor's max magnitude."""
        mx = float(x.float().abs().max())
        if mx == 0:
            return 0.0
        return 2.0 ** (math.floor(math.log2(mx)) - 7)

    with torch.no_grad():
        xn_off = off._normalize(h)
        xn_mine = hcref.grouped_gemma_rmsnorm(h, w.hc_norm_weight, cfg_dict["rms_norm_eps"], hc)
    d_xn_abs = float((xn_off.float() - xn_mine.float()).abs().max())
    d_xn_ulp = ulp(xn_off, xn_mine, significant_only=False)
    d_xn_ulp_sig = ulp(xn_off, xn_mine)
    check("B1a normed input: official GroupedGemmaRMSNorm vs ref/hc.py", d_xn_ulp <= 1,
          f"max abs {d_xn_abs:.3g}, max {d_xn_ulp} bf16-ulp of xn "
          f"({d_xn_ulp_sig} ulp if restricted to ≥2^-8·max elements; the difference "
          f"lives in the low binades)")

    # ... and prove the cause: the official uses `square().mean(-1)`, ref/hc.py
    # follows the Triton kernel's `sum(x*x)/GROUP_DIM`. Recomputing the norm with
    # `mean` must reproduce the official output bit for bit.
    xf = h.float().unflatten(-1, (hc, hidden))
    var_mean = xf.square().mean(-1, keepdim=True)
    xn_via_mean = (
        (xf * torch.rsqrt(var_mean + cfg_dict["rms_norm_eps"])).flatten(-2)
        * (1.0 + w.hc_norm_weight.float())
    ).to(torch.bfloat16)
    check("B1c cause of B1a: `mean(-1)` instead of `sum(-1)/G` is bit-identical",
          torch.equal(xn_via_mean, xn_off) and not torch.equal(xn_via_mean, xn_mine),
          "reduction order alone explains the 1-ulp difference "
          "(`mean(-1)` == official, `sum(-1)/G` == Triton kernel and ref/hc.py)")

    with torch.no_grad():
        off_mixed, (h_off, xn_off) = off.mix(h)
        _, bi_mine, inj_mine = mine.mix(h)
    u = bf16_ulp_at(off_mixed)
    d1 = float((bi_mine.float() - off_mixed.float()).abs().max())
    check("B1b official common.mix() vs ref/hc.py mix() (block_input)", d1 <= 4 * u,
          f"max abs {d1:.3g} = {d1 / u:.2f} bf16-ulp at scale {u:.3g} "
          f"(amplified from the 1-ulp xn difference through the 10240->320->10240 chain)")

    with torch.no_grad():
        h_off_c = off.combine(block, (h_off, xn_off))
        h_mine_bc, bi_mine_bc, _ = mine.combine_and_mix(h, block, inj_mine)
    u2 = bf16_ulp_at(h_off_c)
    d2 = float((h_mine_bc.float() - h_off_c.float()).abs().max())
    check("B2 official eager combine vs ref/hc.py hc_combine (state)", d2 <= 4 * u2,
          f"max abs {d2:.3g} = {d2 / u2:.2f} bf16-ulp at scale {u2:.3g}")

    with torch.no_grad():
        off_mixed2, _ = off.mix(h_off_c)
    u3 = bf16_ulp_at(off_mixed2)
    d3 = float((bi_mine_bc.float() - off_mixed2.float()).abs().max())
    check("B3 official eager mix(combined) vs fused combine_and_mix block_input",
          d3 <= 8 * u3,
          f"max abs {d3:.3g} = {d3 / u3:.2f} bf16-ulp at scale {u3:.3g} "
          f"<-- this is the docs/14 §10 item 3 question")

    # how often does the fused early-bf16-round actually change a bit?
    nz = int((h_mine_bc.reshape(-1) != h_off_c.reshape(-1)).sum())
    check("B4 fused-vs-eager state rounding is OBSERVABLE (docs/14 §10 item 3)",
          nz > 0,
          f"{nz}/{h_mine_bc.numel()} materialized-state elements differ "
          f"({100.0 * nz / h_mine_bc.numel():.2f}%) -> the early bf16 round at "
          f"nvidia/ops/hc.py:327 is a real, measurable effect; a bit-exact L0 "
          f"comparison of `layer.out.hidden` MUST replicate it")
    return d2, d3


# ---------------------------------------------------------------------------
def section_c(layer_idx: int):
    print(f"\n=== C. MXFP4 dequant vs repo numpy golden (layer {layer_idx} real bytes) ===")
    spec = importlib.util.spec_from_file_location(
        "moe_block_ref",
        "/workspace/ascend_mega_kernel/tools/golden/moe_block_ref.py",
    )
    try:
        golden = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(golden)
    except Exception as exc:  # noqa: BLE001
        check("C0 import repo golden", False, f"{type(exc).__name__}: {exc}")
        return
    lw = LayerWeights(layer_idx)
    r = lw.reader
    name = lw.prefix + "mlp.experts.gate_up_proj"
    e = 3
    packed = r.load(name, slice(e, e + 1))[0]
    scale = r.load(name + ".weight_scale", slice(e, e + 1))[0]
    ref_np = dequant_mxfp4(packed, scale)
    mine = dequant_mxfp4_torch(
        r.load_torch(name, slice(e, e + 1))[0], r.load_torch(name + ".weight_scale", slice(e, e + 1))[0], torch
    ).numpy()
    d = np.abs(ref_np - mine).max()
    check("C1 our numpy dequant vs our torch dequant", d == 0.0, f"max abs diff {d}")

    if hasattr(golden, "unpack_mxfp4") or hasattr(golden, "_dequant"):
        fn = getattr(golden, "unpack_mxfp4", None) or getattr(golden, "_dequant")
        try:
            g = fn(packed, scale)
            d2 = np.abs(np.asarray(g, dtype=np.float32) - mine).max()
            check("C2 repo golden dequant vs our torch dequant", d2 == 0.0, f"max abs diff {d2}")
        except Exception as exc:  # noqa: BLE001
            check("C2 repo golden dequant", False, f"{type(exc).__name__}: {exc}")
    else:
        check("C2 repo golden has a callable dequant", False,
              "moe_block_ref.py exposes no unpack_mxfp4/_dequant; compared e2m1/e8m0 tables by hand instead")
        check("C2b e2m1 table matches golden",
              np.array_equal(np.asarray(golden.E2M1_POS), np.array(
                  [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)),
              "golden.E2M1_POS == [0,.5,1,1.5,2,3,4,6]")
        b = np.array([0, 127, 120, 130], dtype=np.uint8)
        check("C2c e8m0 decode matches golden",
              np.array_equal(golden.e8m0_decode(b), np.exp2(b.astype(np.float32) - 127)),
              "2**(byte-127)")


# ---------------------------------------------------------------------------
def section_d():
    print("\n=== D. tensor contract vs docs/14 §2.3 ===")
    lw0 = LayerWeights(0)
    for which, expect in (
        ("attn_hyper_connection", {"hc_norm.weight": (10240,),
                                   "input_mix_weight_down.weight": (320, 10240),
                                   "input_mix_weight_up.weight": (10240, 320),
                                   "block_inject_weight.weight": (4, 10240)}),
        ("mlp_hyper_connection", {"hc_norm.weight": (10240,),
                                  "input_mix_weight_down.weight": (320, 10240),
                                  "input_mix_weight_up.weight": (10240, 320),
                                  "block_inject_weight.weight": (4, 10240)}),
    ):
        ok = True
        for k, shp in expect.items():
            got = tuple(lw0.reader.info(lw0.prefix + which + "." + k).shape)
            ok &= got == shp
        check(f"D1 layer0 {which} shapes", ok, f"all {len(expect)} match docs/14 §2.3")
    gm = LayerWeights.global_mixer()
    check("D2 final mixer has no block_inject_weight (use_combine=False)",
          gm.inject_weight is None and tuple(gm.hc_norm_weight.shape) == (10240,),
          "3 tensors, hc_norm [10240]")
    lw3 = LayerWeights(3)
    exp3 = {
        "self_attn.q_proj.weight": (12288, 2560), "self_attn.k_proj.weight": (512, 2560),
        "self_attn.v_proj.weight": (512, 2560), "self_attn.o_proj.weight": (2560, 6144),
        "self_attn.q_norm.weight": (256,), "self_attn.k_norm.weight": (256,),
        "self_attn.indexer.index_qk_proj.weight": (640, 2560),
        "self_attn.indexer.q_layernorm.weight": (128,),
        "self_attn.indexer.k_layernorm.weight": (128,),
    }
    ok = all(tuple(lw3.reader.info(lw3.prefix + k).shape) == v for k, v in exp3.items())
    check("D3 layer3 QSA weight shapes", ok, f"all {len(exp3)} match the QSA spec")
    cfg = lw3.text_config()
    check("D4 layer dispatch: layer3 layer_type=full_attention but indexer present -> QSA",
          cfg["layer_types"][3] == "full_attention" and cfg.get("indexer_n_heads") is not None,
          f"layer_types[3]={cfg['layer_types'][3]!r}, indexer_n_heads={cfg['indexer_n_heads']}")
    check("D5 ple_layer_ids is 1-based -> 0-based layer 1",
          cfg["ple_layer_ids"] == [2], f"ple_layer_ids={cfg['ple_layer_ids']}")
    check("D6 query/key/value head geometry", (
        cfg["linear_num_key_heads"], cfg["linear_num_value_heads"],
        cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]) == (16, 48, 128, 128),
        "GDN 16k/48v x 128 ; conv_dim = 2*2048+6144 = 10240")


# ---------------------------------------------------------------------------
def section_e(layer_idx: int, m: int):
    print(f"\n=== E. non-hollowness (layer {layer_idx}) ===")
    from ref.layer import DecoderLayer

    weights = LayerWeights(layer_idx)
    cfg = weights.text_config()
    layer = DecoderLayer(layer_idx, weights, cfg)
    hc, hidden = cfg["hc_count"], cfg["hidden_size"]
    torch.manual_seed(7)
    h = (torch.randn(m, hc * hidden) * 0.05).to(torch.bfloat16)
    block = (torch.randn(m, hidden) * 0.05).to(torch.bfloat16)
    inj = (torch.randn(m, hc) * 0.5).to(torch.bfloat16)
    positions = torch.arange(m, dtype=torch.long)

    state = layer.init_state()
    _, _, _, state, t1 = layer.forward(h, block, inj, positions, state=state)
    h2 = h.clone()
    h2[0, 0] = torch.tensor(1.0, dtype=torch.bfloat16)  # single-element perturbation
    _, _, _, _, t2 = layer.forward(h2, block, inj, positions, state=layer.init_state())
    moved = [k for k in t1 if isinstance(t1[k], torch.Tensor) and k in t2
             and not torch.equal(t1[k], t2[k])]
    check("E1 input sensitivity (1 element of hidden changed)", len(moved) > 0,
          f"{len(moved)}/{len(t1)} dumped segments changed; e.g. {sorted(moved)[:4]}")

    check("E2 layer boundary state moves (attn_hc.hidden != input hidden)",
          not torch.equal(t1["attn_hc.hidden"], h),
          "fused combine materializes a new 4-stream state")

    if state is not None:
        conv, ssm = state
        check("E3 ssm_state is non-zero after the run", bool(ssm.abs().sum() > 0),
              f"|ssm_state|_max = {float(ssm.abs().max()):.4g}")
        check("E4 conv_state is non-zero after the run", bool(conv.abs().sum() > 0),
              f"|conv_state|_max = {float(conv.abs().max()):.4g}")
        # `ssm_state` is mutated IN PLACE (as the kernels do), so snapshot it
        # before carrying the state into a second call at a later position.
        ssm_1 = ssm.clone()
        conv_1 = conv.clone()
        pos2 = positions + m
        _, _, _, st2, t3 = layer.forward(h, block, inj, pos2, state=state)
        check("E5 ssm_state evolves across steps",
              not torch.equal(ssm_1, st2[1]),
              f"max |ssm_step1 - ssm_step2| = {float((ssm_1 - st2[1]).abs().max()):.4g}")
        # E5b: a real judgement about the sliding window. Perturb ONLY the first
        # token of the same input and rerun from a fresh state: the conv state is
        # the last `conv_kernel_size-1 = 3` pre-activation columns, so an m>=4
        # input whose earliest column changed must leave conv_state IDENTICAL
        # while ssm_state (which accumulates the whole history) must MOVE.
        # (The earlier version of this check ended in `or True` -- always PASS --
        #  which docs/17 §2.1 forbids in a PASS/FAIL count; fixed with the review.)
        if m >= 4:
            hb = h.clone()
            hb[0, 0] = torch.tensor(2.0, dtype=torch.bfloat16)
            _, _, _, stb, _ = layer.forward(hb, block, inj, positions,
                                            state=layer.init_state())
            _, _, _, sta, _ = layer.forward(h, block, inj, positions,
                                            state=layer.init_state())
            same_conv = torch.equal(sta[0], stb[0])
            diff_ssm = not torch.equal(sta[1], stb[1])
            check("E5b conv_state depends only on the newest 3 columns, "
                  "ssm_state on the whole history",
                  same_conv and diff_ssm,
                  f"perturbing column 0 only: conv_state "
                  f"{'identical' if same_conv else 'DIFFERENT'} "
                  f"(max |d|={float((sta[0].float()-stb[0].float()).abs().max()):.4g}), "
                  f"ssm_state {'moves' if diff_ssm else 'CONSTANT'} "
                  f"(max |d|={float((sta[1]-stb[1]).abs().max()):.4g})")
        else:
            print(f"  [SKIP] E5b needs m>=4 (conv window is 3); m={m} -- "
                  f"NOT counted in the summary")
        check("E5c step 2 MoE output differs from step 1 MoE output",
              not torch.equal(t1["moe.out"], t3["moe.out"]),
              f"max |out_1-out_2| = "
              f"{float((t1['moe.out'].float() - t3['moe.out'].float()).abs().max()):.4g}")

    nonzero = {k: int((v.float() != 0).sum()) for k, v in t1.items()
               if isinstance(v, torch.Tensor) and v.is_floating_point()}
    empties = [k for k, v in nonzero.items() if v == 0]
    check("E6 every dumped float segment is non-zero", not empties,
          f"{len(nonzero)} float segments, zero-only: {empties or 'none'}")
    return t1


# ---------------------------------------------------------------------------
def section_i():
    """Anchor existence (docs/17 §2.4「引用必须存在」) + the checker's own negative controls.

    Round-1 review found ~15 QSA anchors out of range. Round-2 review then found
    the *first* version of this check was resolving bare basenames to another file
    with the same name (`hc.py` -> models/hy_v4/nvidia/hc.py, 74 anchors; 82/514
    total), so `0 out of range` did not mean "the anchors point at the intended
    file". Now: the three outcomes are reported separately, a bare basename is
    resolved only through an explicit BASENAME_MAP or a unique filename match, and
    `section_i` also runs the checker's negative controls (I2) so "passing for the
    wrong reason" is itself tested.
    """
    print("\n=== I. official-source anchors resolve and are in range ===")
    sys.path.insert(0, str(HERE / "tools"))
    import check_anchors

    res = check_anchors.check(verbose=False)
    check("I1 every anchor resolves to ONE intended file and is in range",
          res["out_of_range"] == 0 and res["unverified"] == 0 and res["verified"] > 0,
          f"{res['anchors']} anchors: {res['verified']} verified against the "
          f"intended file, {res['unverified']} UNVERIFIED (ambiguous / root "
          f"absent), {res['out_of_range']} out of range")
    for p in res["problems"][:8]:
        print(f"      {p}")
    for p in res["unresolved"][:8]:
        print(f"      {p}")

    # I2: the checker must not pass for the wrong reason. Negative controls live
    # in check_anchors.NEGATIVE_CASES and include two anchors that DO exist in a
    # same-named twin file and must still be reported.
    rc = check_anchors.selftest()
    check("I2 anchor checker negative controls (would silently pass via a twin file)",
          rc == 0, "see the printed cases above: an anchor that exists only in the "
                   "twin file is reported OUT OF RANGE / UNVERIFIED, never OK")
    return res


def section_j():
    """`tools/safetensors_reader.py` must stay byte-identical to upstream (docs/17 §3.4).

    M39's torch helpers live in `tools/torch_reader.py` (a subclass) precisely so
    the copied reader can stay verbatim: that makes the "diff vs upstream" table
    exactly "identical", and upstream fixes (notably the partially-downloaded
    shard handling) keep flowing in by re-copying the file.
    """
    print("\n=== J. copied safetensors_reader is verbatim upstream ===")
    import hashlib

    local = HERE / "tools" / "safetensors_reader.py"
    upstream = Path("/workspace/ascend_mega_kernel/tools/weights/safetensors_reader.py")
    local_sha = hashlib.sha256(local.read_bytes()).hexdigest()
    if upstream.is_file():
        up_sha = hashlib.sha256(upstream.read_bytes()).hexdigest()
        check("J1 copy sha256 == upstream sha256", local_sha == up_sha,
              f"copy {local_sha[:16]} / upstream {up_sha[:16]}"
              + ("" if local_sha == up_sha else
                 " -- RE-COPY tools/weights/safetensors_reader.py and update "
                 "PINNED_READER_SHA256"))
        check("J2 pinned sha256 matches this copy", local_sha == PINNED_READER_SHA256,
              f"pinned {PINNED_READER_SHA256[:16]}")
    else:
        # a fresh checkout of m21_layer_ref/ alone cannot see the rest of the repo
        check("J1 copy sha256 == PINNED_READER_SHA256", local_sha == PINNED_READER_SHA256,
              f"local {local_sha[:16]}, pinned {PINNED_READER_SHA256[:16]}; "
              f"upstream {upstream} not present, verbatim-vs-upstream NOT verified "
              f"in this checkout")
    # the behaviour added on top of upstream lives only in torch_reader.py
    try:
        from torch_reader import TorchShardReader

        have = all(hasattr(TorchShardReader, m) for m in ("load_torch", "load_torch_f32"))
        detail = ("TorchShardReader.load_torch / load_torch_f32 present "
                  "(bf16 = bit reinterpretation, no rounding)")
        if have:
            import numpy as np

            probe = np.zeros((2, 4), dtype=np.uint16)
            probe[0, 0] = 0x3F80  # bf16 1.0
            import torch as _t

            have = float(_t.from_numpy(probe).view(_t.bfloat16)[0, 0]) == 1.0
            detail += "; bf16 bit-reinterpretation probe 0x3F80 -> 1.0 OK" if have else \
                "; bf16 probe FAILED"
    except Exception as exc:  # noqa: BLE001
        have, detail = False, repr(exc)
    check("J3 torch API on the reader behaves (bf16 is a reinterpretation)",
          have, detail)


def section_k(layer_idx: int = 3, T: int = 256):
    """QSA segment vs vLLM's OWN torch references (the segment's second implementation).

    Round-1 review noted that QSA is the one segment with no runnable official
    implementation, so its credibility rested only on line-anchored traceability.
    `ref/qsa_official_ref.py` is a verbatim copy of the pure-torch helpers vLLM
    uses to validate its Triton QSA kernels; this section runs them against
    `ref/qsa.py` on the same inputs, using a paged view of the dense caches.

    What each check does NOT prove is stated inline.
    """
    print("\n=== K. QSA segment vs vLLM's own torch references (test_qsa_reference.py) ===")
    from ref import qsa_official_ref as R
    from ref.layer import DecoderLayer

    # -- K0: the copied bodies still match the upstream test file
    src_path = Path("/workspace/vllm/tests/models/qwen4_exp/test_qsa_reference.py")
    if src_path.is_file():
        import hashlib

        lines = src_path.read_text(encoding="utf-8").splitlines()
        body = "\n".join(lines[92:234]) + "\n"
        ok = hashlib.sha256(body.encode()).hexdigest() == R.QSA_REFERENCE_BODY_SHA256
        check("K0 copied reference bodies still match upstream line range 93-234",
              ok, f"sha256 {R.QSA_REFERENCE_BODY_SHA256[:16]} "
                  f"({'match' if ok else 'UPSTREAM CHANGED -- re-extract'})")
    else:
        print(f"  [SKIP] K0 upstream test file not present at {src_path}; "
              f"the copied bodies are NOT re-verified in this checkout")

    lw = LayerWeights(layer_idx)
    cfg = lw.text_config()
    lay = DecoderLayer(layer_idx, lw, cfg)
    hc, hidden = cfg["hc_count"], cfg["hidden_size"]
    cr = cfg["indexer_compress_ratio"]
    q_heads, kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    token_topk = cfg["indexer_budget"]
    g = torch.Generator().manual_seed(3)
    h = (torch.randn(T, hidden, generator=g) * 0.05).repeat(1, hc).to(torch.bfloat16)
    _, _, _, _, taps = lay.forward(h, None, None, torch.arange(T), state=None)

    n_cand = taps["qsa.compressed_key"].shape[0]
    page_size = 16
    n_blocks = -(-n_cand // page_size)
    page_table = torch.arange(n_blocks, dtype=torch.int32).unsqueeze(0)
    token_to_req = torch.zeros(T, dtype=torch.int32)
    positions = torch.arange(T, dtype=torch.long)
    seq_len = T
    visible = torch.tensor(
        [min(int((p) + 1) // cr, seq_len // cr) for p in positions], dtype=torch.int32
    )

    # -- K1: indexer logits against the official torch reference
    comp = torch.zeros(n_blocks, page_size, 1, cfg["indexer_head_dim"], dtype=torch.bfloat16)
    comp.view(-1, cfg["indexer_head_dim"])[:n_cand] = taps["qsa.compressed_key"]
    idx_q = taps["qsa.index_q"]  # [T, 4, 128]
    ref_logits = R._qsa_mqa_paged_reference(
        idx_q, comp, page_table, token_to_req, visible
    )
    mine = taps["qsa.index_logits"]  # [T, n_cand], -inf where not scored
    scale = math.sqrt(cfg["indexer_head_dim"])
    # the official reference divides by sqrt(d) at :103, the kernel does not
    # (nvidia/ops/qsa_indexer.py:94-100) -- documented deviation D4
    mask = torch.isfinite(mine)
    rescaled = (ref_logits * scale)
    d_rel = float(((rescaled[mask] - mine[mask]).abs() / mine[mask].abs().clamp_min(1e-30)).max())
    check("K1 indexer logits == official torch reference x sqrt(128) (its :103 scale)",
          d_rel <= 2.0**-20,
          f"max rel diff {d_rel:.3g} over {int(mask.sum())} scored entries; the "
          f"unscaled difference is exactly the 1/sqrt({cfg['indexer_head_dim']}) "
          f"factor that only the test reference applies")

    # -- K2: block top-k sets (the kernel's order is unspecified)
    block_topk = token_topk // cr
    ref_blocks = R._qsa_relative_topk_reference(
        ref_logits, torch.zeros(T, dtype=torch.int32), visible, block_topk
    )
    mine_blocks = taps["qsa.block_indices"]
    bad = 0
    for row in range(T):
        a = set(int(x) for x in ref_blocks[row] if int(x) >= 0)
        b = set(int(x) for x in mine_blocks[row] if int(x) >= 0)
        if a != b:
            bad += 1
    check("K2 block top-k SET matches the official reference for every row", bad == 0,
          f"{T} rows compared as sets (kernel order is unspecified in both); "
          f"{bad} mismatching rows")

    # -- K3: expand + causal tail sets
    ref_expanded = R._expand_qsa_indices_reference(
        ref_blocks, positions, torch.full((T,), seq_len, dtype=torch.int64), cr, token_topk
    )
    mine_tok = taps["qsa.token_indices"]
    ow = token_topk + cr - 1
    bad = 0
    for row in range(T):
        a = set(int(x) for x in ref_expanded[row] if int(x) >= 0)
        b = set(int(x) for x in mine_tok[row, :ow] if int(x) >= 0)
        if a != b:
            bad += 1
    check("K3 expanded token SET matches the official reference for every row",
          bad == 0,
          f"{T} rows compared as sets; {bad} mismatching. The official helper "
          f"compacts -1 to the end and has no count column (:128-164) while the "
          f"kernel keeps positions + count (:268-276) -- sets are the invariant")

    # -- K4: sparse attention output (same selection, same caches)
    # the MAIN kv cache is one row per TOKEN (not per compressed group), so it
    # needs its own page geometry: ceil(T / page_size) blocks.
    main_blocks = -(-T // page_size)
    main_page_table = torch.arange(main_blocks, dtype=torch.int32).unsqueeze(0)
    k_cache = torch.zeros(main_blocks, page_size, kv_heads, hd, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    k_cache.view(-1, kv_heads, hd)[:T] = taps["qsa.k"]
    v_cache.view(-1, kv_heads, hd)[:T] = taps["qsa.v"]
    # pass only the INDEX columns: column 2051 is the count, not a token. The
    # kernel does the same (`selection_width = logical_indices.shape[1] - 1`,
    # nvidia/ops/qsa.py:978); forgetting it feeds 2049 into the page table.
    ref_attn = R._qsa_sparse_paged_attention_reference(
        taps["qsa.q"], k_cache, v_cache, mine_tok[:, :ow], main_page_table,
        token_to_req, float(hd**-0.5),
    )
    post_gate = (ref_attn.float() * torch.sigmoid(taps["qsa.gate"].float())).to(
        torch.bfloat16
    )
    # -- K5: end-to-end visible_blocks -> score -> top-k through the official
    # helper (the 5th copied function, which K1-K4 never used). This is the only
    # place where MY implementation of the `visible_blocks` formula
    # (min((pos+1)//cr, seq_len//cr)) is compared against the official helper's
    # own computation of it -- both sides derive it, neither is passed in.
    ref_sel = R._qsa_select_paged_reference(
        idx_q, comp, page_table, token_to_req, positions,
        torch.full((1,), seq_len, dtype=torch.int64), token_topk, cr,
    )
    bad = 0
    for row in range(T):
        a = set(int(x) for x in ref_sel[row] if int(x) >= 0)
        b = set(int(x) for x in mine_blocks[row] if int(x) >= 0)
        if a != b:
            bad += 1
    check("K5 visible_blocks->score->top-k end-to-end via _qsa_select_paged_reference",
          bad == 0,
          f"{T} rows compared as sets; {bad} mismatching. This is the only check "
          f"where BOTH sides compute `visible_blocks` independently of each other "
          f"(K1 passes mine in), so it is what covers that formula at all")

    d_attn = ulp(taps["qsa.attn_out"], post_gate)
    check("K4 sparse attention == official reference (bf16-round, then *sigmoid(gate))",
          d_attn <= 2,
          f"max {d_attn} bf16-ulp over {taps['qsa.attn_out'].numel()} elements; "
          f"the official helper returns pre-gate output "
          f"(test_qsa_reference.py:198-234) and the test applies "
          f"`expected * sigmoid(output_gate)` outside (test_qsa_reference.py:1077)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--layer-qsa", type=int, default=3)
    ap.add_argument("--m", type=int, default=4)
    args = ap.parse_args()

    section_a()
    d1, d2 = section_b(args.layer, args.m)
    section_c(args.layer)
    section_d()
    t1 = section_e(args.layer, args.m)
    print(f"\n=== E-lite: QSA layer {args.layer_qsa} runs ===")
    from ref.layer import DecoderLayer

    lw = LayerWeights(args.layer_qsa)
    cfg = lw.text_config()
    lay = DecoderLayer(args.layer_qsa, lw, cfg)
    hc, hidden = cfg["hc_count"], cfg["hidden_size"]
    torch.manual_seed(11)
    h = (torch.randn(args.m, hc * hidden) * 0.05).to(torch.bfloat16)
    _, _, _, _, tq = lay.forward(h, None, None, torch.arange(args.m), state=None)
    check("F1 QSA token_indices count column",
          int(tq["qsa.token_indices"][-1, cfg["indexer_budget"] + cfg["indexer_compress_ratio"] - 1])
          == (args.m // cfg["indexer_compress_ratio"]) * cfg["indexer_compress_ratio"],
          f"row -1 count = {int(tq['qsa.token_indices'][-1, 2051])} for m={args.m}")
    check("F2 QSA compressed keys computed for complete groups",
          tq["qsa.compressed_key"].shape[0] == args.m // cfg["indexer_compress_ratio"],
          f"{tq['qsa.compressed_key'].shape[0]} rows = m // 4")

    # F3: the vectorized (per KV head) sparse attention must agree with an
    # independent head-by-head recomputation of the same formula. This validates
    # ref/qsa.py's vectorization, whose only intended difference is summation order.
    q_ = tq["qsa.q"].float()
    kc, vc = lay.attn.k_cache, lay.attn.v_cache
    idx = tq["qsa.token_indices"]
    gate = tq["qsa.gate"].float()
    num_heads, num_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    grp = num_heads // num_kv
    scale = cfg["head_dim"] ** -0.5
    ref_out = torch.zeros_like(q_)
    for row in range(args.m):
        toks = idx[row, :2047]
        toks = toks[toks >= 0].long()
        if toks.numel() == 0:
            continue
        for h in range(num_heads):
            kv = h // grp
            keys = kc[toks, kv, :].float()
            vals = vc[toks, kv, :].float()
            probs = torch.softmax((q_[row, h] @ keys.t()) * scale, dim=-1)
            acc = (probs @ vals).to(torch.bfloat16)
            ref_out[row, h] = (acc.float() * torch.sigmoid(gate[row, h])).to(torch.bfloat16)
    d_attn = ulp(tq["qsa.attn_out"], ref_out.to(torch.bfloat16))
    check("F3 QSA sparse attention: vectorized vs head-by-head recompute",
          d_attn <= 2, f"max {d_attn} bf16-ulp (summation order only)")

    # F4: structural invariants of the packed selection buffer (catches a broken
    # expand/tail). The trailing column must equal the number of valid entries;
    # every token must be <= that row's position (causality); the causal tail must
    # be exactly tail_start..pos.
    cr = cfg["indexer_compress_ratio"]
    ow = cfg["indexer_budget"] + cr - 1
    bad = []
    for row in range(args.m):
        toks = tq["qsa.token_indices"][row, :ow]
        cnt = int(tq["qsa.token_indices"][row, ow])
        if int((toks >= 0).sum()) != cnt:
            bad.append((row, "count"))
            continue
        if int(toks.max()) > row:
            bad.append((row, "future-token"))
            continue
        tail_start = ((row + 1) // cr) * cr
        tail = sorted(int(x) for x in toks[toks >= tail_start])
        want = list(range(tail_start, row + 1))
        if tail != want:
            bad.append((row, f"tail {tail} != {want}"))
    check("F4 QSA packed selection invariants (count col / causality / tail)",
          not bad, f"{args.m} rows checked; violations: {bad[:3] or 'none'}")

    # F5: the top-k TRUNCATION path. Requires visible_blocks > block_topk = 512,
    # i.e. sequence length > 2048. This is the case the kernel's `safe_rank`
    # clamping exists for (qsa_indexer.py:248-253) and it is not reachable at
    # small m, so it gets its own check.
    print("\n=== H. QSA top-k truncation (T=2101 > token_topk=2048) ===")
    T = 2101
    lay2 = DecoderLayer(args.layer_qsa, LayerWeights(args.layer_qsa), cfg)
    g = torch.Generator().manual_seed(0)
    h2 = (torch.randn(T, hidden, generator=g) * 0.05).repeat(1, hc).to(torch.bfloat16)
    _, _, _, _, tl = lay2.forward(h2, None, None, torch.arange(T), state=None)
    ti = tl["qsa.token_indices"]
    n_valid_blocks = int((tl["qsa.block_indices"][-1] >= 0).sum())
    count = int(ti[-1, ow])
    check("H1 block top-k truncates at block_topk = 512",
          n_valid_blocks == cfg["indexer_budget"] // cr == 512,
          f"visible_blocks = {T // cr} > 512, selected blocks = {n_valid_blocks}")
    check("H2 count column = min(visible,512)*4 + tail_count",
          count == 512 * cr + ((T) - (T // cr) * cr),
          f"count = {count} = 2048 + {(T) - (T // cr) * cr} (tail), "
          f"fits output_width {ow}")
    check("H3 every selected token is a complete-group token or the causal tail",
          int((ti[-1, :ow] >= 0).sum()) == count
          and int(ti[-1, :ow].max()) == T - 1,
          f"non-(-1) entries = {int((ti[-1, :ow] >= 0).sum())}, max token = "
          f"{int(ti[-1, :ow].max())} = T-1")
    # The logits dump must cover EVERY visible block (the kernel's logits buffer
    # is as wide as the compressed cache); clamping the scoring at block_topk
    # would silently drop the newest 13 candidates here.
    lg = tl["qsa.index_logits"]
    n_finite_last = int(torch.isfinite(lg[-1]).sum())
    check("H4 logits dump covers all visible blocks, not just block_topk",
          n_finite_last == T // cr == 525 and lg.shape[-1] == 525,
          f"last row has {n_finite_last}/525 finite scores, dump shape "
          f"{tuple(lg.shape)} -- top-512 is taken FROM these 525, as the kernel does")

    print("\n=== G. global final mixer (use_combine=False) ===")
    from ref.layer import final_mixer

    mw = LayerWeights.global_mixer()
    mixer, multi_hidden, sample_hidden = final_mixer(
        tq["layer.out.hidden"] if "layer.out.hidden" in tq else tq["attn_hc.hidden"],
        tq["moe.out"],
        tq["mlp_hc.injection"],
        mw,
        hc,
    )
    check("G1 final mixer has no block_inject / returns no injection",
          mixer.inject_weight is None and mixer.use_combine is False,
          "`hyper_connection_mixer` ships only 3 tensors (vLLM: use_combine=False, "
          "nvidia/model.py:437-441)")
    check("G2 final mixer output shapes",
          tuple(multi_hidden.shape)[-1] == hc * hidden
          and tuple(sample_hidden.shape)[-1] == hidden,
          f"multi_hidden {tuple(multi_hidden.shape)} = [T,10240], "
          f"sample_hidden {tuple(sample_hidden.shape)} = [T,2560] (feeds lm_head, :590)")

    section_i()
    section_j()
    section_k(args.layer_qsa)

    n_fail = sum(1 for st, _, _ in LOG if st == "FAIL")
    if not LOG:
        print("RESULT: SKIPPED (0 checks ran -- NOT a pass)")
        return 2
    print(f"\n=== selfcheck summary: {len(LOG)} checks, {n_fail} FAIL ===")
    for st, name, detail in LOG:
        if st == "FAIL":
            print(f"  FAIL {name}: {detail}")
    print(f"docs/14 §10 item 3 measurement: eager-vs-fused state max {d1} ulp, "
          f"block_input max {d2} ulp")
    print(f"RESULT: {'FAIL' if n_fail else 'OK'} ({len(LOG) - n_fail}/{len(LOG)} "
          f"judgements passed)")
    # exit codes: 0 = ran and passed; 1 = ran and something failed; 2 = nothing ran
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
