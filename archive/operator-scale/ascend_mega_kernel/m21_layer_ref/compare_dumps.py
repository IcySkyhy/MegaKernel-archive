#!/usr/bin/env python3
"""One-click graded comparison of a kernel dump against the official reference.

Implements the reporting rules of `docs/17-verification-standard.md`:

  §1  判定项 (PASS/FAIL) and 报告项 (bit-exact rate, <=1ulp rate, ulp histogram)
      are printed in SEPARATE tables; guards never enter the PASS/FAIL count.
  §2  every error number is labelled max (never "first mismatch"), absolute or
      relative, and the input tag it was measured on.
  §3  self-consistency: the report prints the exact command and the sha256 of
      both manifests, so a reviewer can re-run it offline.

Exit codes (project rule 2026-09-26: "审校脚本在无输入可比时必须报 SKIPPED 且非零退出")
----------
  # reference dump vs our kernel dump (both carry manifest.json)
  python3 compare_dumps.py --ref reference/layer0_decode_m1 --test /path/to/kernel_dump

  # or hand it raw .bin blobs exported from the device
  python3 compare_dumps.py --ref reference/layer0_decode_m1 \
      --test-bin attn_hc.block_input=/tmp/hc_out.bin:bf16:1,2560 \
      --test-bin moe.topk_ids=/tmp/ids.bin:int32:1,10

  # L1 profile (tolerances derived from the MXFP4 budget) instead of the
  # default L0 profile (bit-exact / <=1 ulp)
  python3 compare_dumps.py --ref ... --test ... --profile mx-budget

  # non-hollowness: prove the reference reacts to its input
  python3 compare_dumps.py --ref reference/layer0_decode_m1 \
      --ref2 reference/layer0_decode_m1_perturbed --nonhollow-only

Exit codes
----------
  0  compared something and everything passed
  1  compared something and found a difference (a real judgement, not a skip)
  2  NOTHING COMPARABLE (no shared segments / all rows skipped / no pairs) --
     this is a SKIP, never an OK, and the message says what was compared
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------
_MANTISSA_BITS = {"bfloat16": 8, "float16": 11, "float32": 24}
_BIT_WIDTH = {"bfloat16": 16, "float16": 16, "float32": 32}


def load_dump(directory: Path, name: str, entry: dict) -> torch.Tensor:
    arr = np.load(directory / entry["file"])
    if entry["dtype"] == "bfloat16":
        t = torch.from_numpy(np.ascontiguousarray(arr)).view(torch.bfloat16)
    elif entry["dtype"] == "float32":
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)
    elif entry["dtype"] == "int32":
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int32)
    elif entry["dtype"] == "int64":
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int64)
    elif entry["dtype"] == "uint8":
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.uint8)
    else:
        raise ValueError(f"unknown dtype {entry['dtype']} for {name}")
    return t.reshape(entry["shape"])


def load_bin(path: Path, dtype: str, shape: list[int]) -> torch.Tensor:
    dtype = dtype.lower()
    np_dt = {
        "bf16": np.uint16,
        "bfloat16": np.uint16,
        "fp32": np.float32,
        "float32": np.float32,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "fp16": np.float16,
        "float16": np.float16,
    }[dtype]
    arr = np.fromfile(path, dtype=np_dt)
    n = int(np.prod(shape)) if shape else arr.size
    if arr.size != n:
        raise ValueError(f"{path}: expected {n} elements, found {arr.size}")
    t = torch.from_numpy(arr).reshape(shape)
    if np_dt is np.uint16:
        t = t.view(torch.bfloat16)
    return t


def monotonic_key(t: torch.Tensor, dtype: str) -> torch.Tensor:
    """Map IEEE sign-magnitude bits to a monotonically ordered integer key."""
    width = _BIT_WIDTH[dtype]
    if dtype == "bfloat16":
        bits = t.view(torch.bfloat16).view(torch.uint16).to(torch.int32)
        maxbits = 0xFFFF
    elif dtype == "float16":
        bits = t.view(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
        maxbits = 0xFFFF
    else:
        bits = t.view(torch.float32).view(torch.int32).to(torch.int64)
        maxbits = 0xFFFFFFFF
    # normalise -0 to +0 so they are not 1 ulp apart
    bits = torch.where(bits == (1 << (width - 1)), torch.zeros_like(bits), bits)
    sign = (bits >> (width - 1)) & 1
    key = torch.where(sign == 1, maxbits - bits, bits + (maxbits + 1) // 2)
    return key


def ulp_distance(a: torch.Tensor, b: torch.Tensor, dtype: str) -> torch.Tensor:
    return (monotonic_key(a, dtype) - monotonic_key(b, dtype)).abs()


# ---------------------------------------------------------------------------
# tolerance policy
# ---------------------------------------------------------------------------
# L0 profile: docs/17 §1 -- bit-exact, or <=1 ulp where reduction order makes
# bit-exactness impossible. `max_ulp` is the judgement knob for float segments.
L0_PROFILE = {
    "bfloat16": {"rtol": 2.0**-8, "atol": 0.0, "max_ulp": 1},
    "float16": {"rtol": 2.0**-11, "atol": 0.0, "max_ulp": 1},
    "float32": {"rtol": 2.0**-20, "atol": 1e-7, "max_ulp": 8},
    "int32": {"rtol": 0.0, "atol": 0.0, "max_ulp": 0},
    "int64": {"rtol": 0.0, "atol": 0.0, "max_ulp": 0},
    "uint8": {"rtol": 0.0, "atol": 0.0, "max_ulp": 0},
}

# L1 profile: docs/17 §1 -- tolerance derived from the segment's quantization
# budget. `mx_act` is the per-element bound from round-to-nearest on a 3-bit
# mantissa (half ulp = 2**-4 relative) applied to BOTH GEMM operands, plus the
# group-scale bound |delta| <= max|x|/8 (see README "quantization budget").
# These numbers are deliberately generous: their purpose is to separate
# "broken" from "inside the quantization budget", not to certify accuracy.
MX_ACT_RTOL = 0.25  # 2 * (1/8) per-operand half-ulp bound, two operands
MX_ACT_ATOL = 0.0

# The ULP criterion is evaluated only within this many binades of the tensor's
# max |reference| value; below that, values are noise-scale and are judged by
# the absolute budget alone. See `tensor_error`.
SIG_BINADES = 8

# T3 profile (docs/17 §1.1). Triggered by transcendentals (exp/rsqrt), cross-tile
# or online re-scaling, and mmad/cube accumulation -- i.e. every GEMM- or
# softmax-bearing segment of this model. The criterion is
#     |out - ref| <= eps * sum|terms| + 0.5*ulp(out),  checked per element
# and `<=1ulp rate` / `maxRel` become REPORT items.
#
# `eps` has two halves and they have different owners:
#   * REFERENCE side (this repository's contribution, derived here):
#     the reference computes softmax / normalisation in single-shot fp32, so it
#     differs from an online/exp2 relabelling by a handful of fp32 roundings:
#     4 * 2**-24. Documented so a receiving kernel can add it instead of guessing.
#   * RECEIVING side (the kernel author must supply): n * 2**-24 for an n-term
#     fp32 accumulation (one rounding per add) plus its own transcendental
#     approximation error. The default below is sized for n = 2048 (the QSA
#     token_topk) : 2 * 2048 * 2**-24 = 2.44e-4, plus the reference's 4 * 2**-24,
#     rounded up -> 2**-12. **Any other n must pass its own --rtol via
#     `--tolerance`**; this default is not a claim about any specific kernel.
T3_RTOL = 2.0**-12
# The `+ 0.5 * ulp(out)` half is applied per element inside `tensor_error`; it
# exists because a single relative bound on sum|terms| cannot cover elements that
# cancelled down (docs/17 §1.1, M24 measured 1504/6144 such elements).
T3_DEFAULT_N = 2048

# Segments for which the report ALSO prints argmax / top-k agreement
# (docs/17 §1 L2 metric ②, applied per segment). Report-only.
AGREEMENT_SEGMENTS_DEFAULT = "moe.router_logits,qsa.index_logits"
AGREEMENT_TOPK = 10

# Segments where the official kernel leaves part of its output buffer
# uninitialised, so the non-finite-mask criterion has nothing to say about those
# columns. QSA writes logits only for `column < visible_blocks`
# (vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py:101-107), and the reference
# dump marks the rest with -inf. Use `--mask-policy ignore-unscored` for these.
UNSCORED_SEGMENTS_DEFAULT = "qsa.index_logits"

# Segments that docs/17 §1.1 puts in T3 (transcendentals / cross-tile or online
# re-scaling / mmad accumulation). Judging these under the default `l0` profile
# (max_ulp=1) can FAIL for reasons that are NOT defects -- e.g. F3/K4 measure a
# genuine 2 bf16-ulp difference between two summation orders. We do NOT silently
# switch profiles; we warn, so the choice stays visible in the report.
T3_SEGMENTS = {
    "qsa.attn_out", "attn.out", "qsa.q", "qsa.k", "qsa.v", "qsa.gate",
    "qsa.index_q", "qsa.index_logits", "qsa.compressed_key",
    "gdn.conv_out", "gdn.q", "gdn.k", "gdn.v", "gdn.core_out", "gdn.normed",
    "gdn.g", "gdn.beta",
    "moe.router_logits", "moe.routed_out", "moe.shared_out", "moe.out",
    "moe.block_input",
    "attn_hc.block_input", "attn_hc.hidden", "attn_hc.injection",
    "mlp_hc.block_input", "mlp_hc.hidden", "mlp_hc.injection",
    "layer.out.hidden", "layer.out.block_output", "layer.state.ssm_state",
}


def tolerance_for(name: str, dtype: str, profile: str, overrides: dict) -> dict:
    if overrides and name in overrides:
        return overrides[name]
    if profile == "l0":
        return dict(L0_PROFILE[dtype], profile="l0")
    seg_class = overrides.get("__class__", {}).get(name) if overrides else None
    if seg_class == "mx":
        return {"rtol": MX_ACT_RTOL, "atol": MX_ACT_ATOL,
                "max_ulp": 1 << 24, "profile": "mx"}
    if profile == "mx-budget":
        return {"rtol": MX_ACT_RTOL, "atol": MX_ACT_ATOL,
                "max_ulp": 1 << 24, "profile": "mx-budget"}
    if profile == "t3":
        if dtype in ("int32", "int64", "uint8"):
            return dict(L0_PROFILE[dtype], profile="t3-int")  # T1: no downgrade
        return {"rtol": T3_RTOL, "atol": 0.0, "max_ulp": 1 << 24,
                "per_element_half_ulp": True, "profile": "t3"}
    return dict(L0_PROFILE[dtype], profile="l0")


# ---------------------------------------------------------------------------
def tensor_error(ref: torch.Tensor, test: torch.Tensor, tol: dict, dtype: str,
                 name: str = "", agreement_segments: set[str] | None = None,
                 ignore_unscored: bool = False) -> dict:
    """Every number here is a MAX over the whole tensor, labelled as such.

    ULP distance is only meaningful where the reference is not noise-scale, so
    the ulp criterion is evaluated on elements whose |value| is within
    `2**-SIG_BINADES` of the tensor's max |reference| value. Everything below
    that is judged purely by the absolute budget, and its count is reported so
    the reviewer can see how much of the tensor was excluded. This is stated
    explicitly because "ulp" on a near-zero value is a meaningless number.

    `ignore_unscored=True` (opt-in, `--mask-policy ignore-unscored`) drops the
    positions that are non-finite in the REFERENCE from every criterion. Use it
    for segments where the official kernel leaves part of its output buffer
    uninitialised -- QSA writes logits only for `column < visible_blocks`
    (`ops/qsa_indexer.py:101-107`), so the rest is officially undefined and a
    kernel that dumps 0 there is not wrong. Without this, the strict mask
    criterion would FAIL such a dump with no documented relaxation.
    """
    a = ref.detach().float().reshape(-1)
    b = test.detach().float().reshape(-1)
    exact = int((ref.detach().reshape(-1) == test.detach().reshape(-1)).sum())
    finite_dtype = dtype in ("bfloat16", "float16", "float32")
    if finite_dtype:
        # Non-finite values (e.g. the -inf fill that QSA uses for "column not
        # scored") carry no magnitude information, so they cannot be compared by
        # subtraction. They are compared by MASK EQUALITY instead, and the mask
        # mismatch count is a term in the PASS/FAIL criterion.
        fa = torch.isfinite(a)
        fb = torch.isfinite(b)
        out_finite = int((~fa).sum())
        test_finite = int((~fb).sum())
        mask_mismatch = int((fa != fb).sum())
        if ignore_unscored:
            # reference-side non-finite = "not computed upstream" -> skip entirely
            mask_mismatch = int((fa & ~fb).sum())
            both = fa & fb
        else:
            both = fa & fb
    else:
        out_finite = test_finite = mask_mismatch = 0
        both = torch.ones_like(a, dtype=torch.bool)
    diff = (a - b).abs()
    diff = torch.where(both, diff, torch.zeros_like(diff))
    a_abs = torch.where(both, a.abs(), torch.zeros_like(a))
    denom = a_abs.clamp_min(1e-30)
    scale = float(a_abs.max()) if a.numel() else 0.0
    out = {
        "n": int(a.numel()),
        "ref_max_abs": scale,
        "test_max_abs": float(torch.where(both, b.abs(), torch.zeros_like(b)).max())
        if b.numel() else 0.0,
        "max_abs_err": float(diff.max()) if a.numel() else 0.0,
        "max_rel_err": float((diff / denom).max()) if a.numel() else 0.0,
        "bit_exact": exact,
        "n_nonfinite_ref": out_finite,
        "n_nonfinite_test": test_finite,
        "nonfinite_mask_mismatch": mask_mismatch,
        "n_compared": int(both.sum()),
        "ignored_unscored": (out_finite if ignore_unscored else 0),
    }
    if finite_dtype and a.numel():
        thresh = scale * (2.0**-SIG_BINADES)
        sig = both & ((a.abs() >= thresh) | (b.abs() >= thresh))
        u = ulp_distance(ref.detach().reshape(-1), test.detach().reshape(-1), dtype)
        out["n_significant"] = int(sig.sum())
        out["n_subthreshold"] = int((~sig).sum())
        out["max_ulp"] = int(u[sig].max()) if bool(sig.any()) else 0
        out["le1ulp"] = int((u <= 1).sum())
        out["zero_ulp"] = int((u == 0).sum())
        out["max_ulp_any"] = int(u.max())
    else:
        out["n_significant"] = int(a.numel())
        out["n_subthreshold"] = 0
        out["max_ulp"] = 0
        out["max_ulp_any"] = 0
        out["le1ulp"] = exact
        out["zero_ulp"] = exact
    budget = tol["atol"] + tol["rtol"] * scale
    out["budget_abs"] = budget
    out["usage"] = (out["max_abs_err"] / budget) if budget > 0 else (
        0.0 if out["max_abs_err"] == 0 else math.inf
    )
    half_ulp_used = 0
    if tol.get("per_element_half_ulp") and finite_dtype and a.numel():
        # docs/17 §1.1 T3: |out - ref| <= eps*sum|terms| + 0.5*ulp(out), tested
        # PER ELEMENT. We only have |out| here, not sum|terms|, so the eps term
        # is charged against |ref| (valid when no catastrophic cancellation) and
        # the elements that DO cancel are counted separately so a reader can see
        # whether the failures are all boundary elements.
        # `0.5 * ulp(out)` needs the spacing at each ELEMENT, not at the tensor
        # max: spacing = 2^(floor(log2|x|) - mantissa_bits).
        mag = torch.where(both, b.abs(), torch.ones_like(b))
        expo = torch.floor(torch.log2(mag.clamp_min(torch.finfo(torch.float32).tiny)))
        ulp_out = torch.pow(2.0, expo - {"bfloat16": 8, "float16": 11, "float32": 24}[dtype])
        allowed = tol["rtol"] * a_abs + 0.5 * ulp_out
        over = both & (diff > allowed)
        out["t3_over_count"] = int(over.sum())
        out["t3_cancel_count"] = int((both & (a_abs < 0.01 * scale) & over).sum())
        out["t3_allowed_used"] = float((diff / allowed.clamp_min(1e-30)).max()) if a.numel() else 0.0
        half_ulp_used = out["t3_over_count"]
    within_tol = (
        out["max_abs_err"] <= budget
        and out["max_ulp"] <= tol["max_ulp"]
        and mask_mismatch == 0
        and half_ulp_used == 0
    )
    out["verdict"] = "PASS" if within_tol else "FAIL"

    # Top-k / argmax agreement (docs/17 §1 L2 metric ②, applied per segment).
    # REPORT ONLY: it never enters the PASS/FAIL count.
    if agreement_segments and name in agreement_segments and ref.dim() == 2:
        a2 = ref.float().reshape(ref.shape[0], -1)
        b2 = test.float().reshape(test.shape[0], -1)
        rows = a2.shape[0]
        k = min(AGREEMENT_TOPK, a2.shape[1])
        am = ka = ks = 0
        for r in range(rows):
            ar, br = a2[r], b2[r]
            if not torch.isfinite(ar).any() or not torch.isfinite(br).any():
                continue
            am += int(torch.argmax(ar).item() == torch.argmax(br).item())
            kk = min(k, int(torch.isfinite(ar).sum()), int(torch.isfinite(br).sum()))
            if kk > 0:
                sa = set(torch.topk(ar, kk).indices.tolist())
                sb = set(torch.topk(br, kk).indices.tolist())
                ka += int(len(sa & sb) == kk)
                ks += len(sa & sb)
        out["argmax_agree_rows"] = am
        out["argmax_agree_total"] = rows
        out["topk_set_rows"] = ka
        out["topk_set_total"] = rows
        out["topk_overlap"] = ks
        out["topk_k"] = k
    return out


def manifest_digest(directory: Path) -> str:
    p = directory / "manifest.json"
    if not p.exists():
        return "no-manifest"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
def build_pairs(args, ref_dir: Path, ref_man: dict):
    pairs: list[tuple[str, torch.Tensor, torch.Tensor, str]] = []
    sort_rows = {s.strip() for s in (args.sort_rows or "").split(",") if s.strip()}
    test_manifest = None
    test_dir = Path(args.test) if args.test else None
    if test_dir is not None:
        test_manifest = json.loads((test_dir / "manifest.json").read_text())["segments"]
    bins = {}
    for spec in args.test_bin or []:
        name, rest = spec.split("=", 1)
        path, dtype, shape = rest.rsplit(":", 2)
        bins[name] = (Path(path), dtype, [int(x) for x in shape.split(",") if x])

    mapping = {}
    if args.map:
        mapping = json.loads(Path(args.map).read_text())

    for name, entry in ref_man.items():
        if args.segments and name not in args.segments:
            continue
        ref = load_dump(ref_dir, name, entry)
        tname = mapping.get(name, name)
        if tname in bins:
            path, dtype, shape = bins[tname]
            test = load_bin(path, dtype, shape)
        elif test_manifest is not None and tname in test_manifest:
            test = load_dump(test_dir, tname, test_manifest[tname])
        else:
            continue
        pairs.append((name, _maybe_sort(name, ref, sort_rows),
                      _maybe_sort(name, test, sort_rows), entry["dtype"]))
    return pairs, test_manifest


def _maybe_sort(name: str, t: torch.Tensor, sort_rows: set[str]) -> torch.Tensor:
    """Sort each row of an index-valued segment before comparing.

    The top-k the kernel performs has an UNSPECIFIED output order
    (`tests/kernels/test_top_k_per_row.py` compares sorted sets only), so
    `qsa.block_indices` / `qsa.token_indices` can only be compared as multisets.
    Pad values (-1) sort to the front, so the multiset is preserved either way.
    """
    if name not in sort_rows or t.dim() < 2:
        return t
    return torch.sort(t.to(torch.int64), dim=-1).values.to(t.dtype)


def report(args) -> int:
    ref_dir = Path(args.ref)
    ref_man = json.loads((ref_dir / "manifest.json").read_text())["segments"]
    overrides = json.loads(Path(args.tolerance).read_text()) if args.tolerance else {}

    if args.nonhollow_only:
        return nonhollow(args, ref_dir, ref_man)

    pairs, test_manifest = build_pairs(args, ref_dir, ref_man)
    if not pairs:
        print("no comparable segments found "
              "(check --test/--test-bin manifests and segment names)", file=sys.stderr)
        return 2

    agreement = {s.strip() for s in (args.agreement_rows or "").split(",") if s.strip()}
    unscored = {s.strip() for s in (args.unscored_segments or "").split(",") if s.strip()}
    rows = []
    for name, ref, test, dtype in pairs:
        tol = tolerance_for(name, dtype, args.profile, overrides)
        rows.append((name, dtype,
                     tensor_error(ref, test, tol, dtype, name, agreement,
                                  ignore_unscored=(
                                      args.mask_policy == "ignore-unscored"
                                      and name in unscored)), tol))

    out: list[str] = []

    def w(line: str = "") -> None:
        out.append(line)

    w(f"# Graded comparison — {args.profile} profile")
    w()
    w(f"- reference : `{ref_dir}` (manifest sha256[:16] = `{manifest_digest(ref_dir)}` — "
      f"this covers the run PARAMETERS only; the stable tensor hashes are in "
      f"`evidence/reference_sha256.txt`)")
    if args.test:
        td = Path(args.test)
        w(f"- test      : `{td}` (manifest sha256[:16] = `{manifest_digest(td)}`)")
    for spec in args.test_bin or []:
        w(f"- test-bin  : `{spec}`")
    w(f"- command   : `compare_dumps.py --ref {args.ref}"
      + (f" --test {args.test}" if args.test else "")
      + (f" --profile {args.profile}" if args.profile != 'l0' else "")
      + (f" --mask-policy {args.mask_policy}" if args.mask_policy != "strict" else "")
      + "`")
    w()
    num_judgements = len(rows)
    n_pass = sum(1 for _, _, r, _ in rows if r["verdict"] == "PASS")
    w("## 1 判定项 (judgement: PASS/FAIL)")
    w()
    w("| segment | dtype | max abs err | ref max abs | budget abs | usage | "
      "max ulp (≥2^-%d·max) | 非有限掩码不匹配 | verdict |" % SIG_BINADES)
    w("|---|---|---|---|---|---|---|---|---|")
    for name, dtype, r, tol in rows:
        # Under the t3 profile, `usage` and `max_ulp` are NOT the judgement (the
        # per-element count is); mark them so nobody reads them as PASS/FAIL terms.
        # Round-3 review: they showed numbers like 445616% / 32699 next to a PASS.
        t3 = ""
        if "t3_over_count" in r:
            t3 = (f" 逐元素超界 {r['t3_over_count']}/{r['n_compared']}"
                  f"（其中相消区 {r['t3_cancel_count']}）")
        usage = (f"_(非判据)_ {r['usage']*100:.2f}%"
                 if "t3_over_count" in r else f"{r['usage']*100:.2f}%")
        maxulp = (f"_(非判据)_ {r['max_ulp']}"
                  if "t3_over_count" in r else f"{r['max_ulp']}")
        w(f"| `{name}` | {dtype} | {r['max_abs_err']:.6g} | {r['ref_max_abs']:.6g} | "
          f"{r['budget_abs']:.6g} | {usage} | {maxulp} | "
          f"{r['nonfinite_mask_mismatch']} | **{r['verdict']}**{t3} |")
    w()
    w(f"**判定项总数 {num_judgements}；PASS {n_pass}；FAIL {num_judgements - n_pass}**")
    w()
    ulp_knobs = sorted({r[3]["max_ulp"] for r in rows})
    ulp_clause = (
        f"；并按 profile 另加 `max_ulp <= {ulp_knobs[0]}`" if ulp_knobs and ulp_knobs[0] < (1 << 24)
        else "（本 profile 不设 ulp 上限：量化预算本身就是绝对量级判据）"
    )
    if args.profile == "t3":
        w(f"**本例用 T3 档**（`docs/17` §1.1，触发条件：超越函数 / 跨 tile·online 重标定 / "
          f"mmad 累加）：判据是 **逐元素** `|out-ref| <= rtol*|ref| + 0.5*ulp(out)` "
          f"（rtol = {T3_RTOL:.3g} = 2^-12，来源见 `compare_dumps.py` 的 `T3_RTOL` 注释："
          f"参考侧 4·2^-24 + 接收侧 2n·2^-24（n={T3_DEFAULT_N}））。"
          f"**接收侧的 ε 必须由该 kernel 作者按自己的 exp2/mad 特性给出**，"
          f"本默认值不是对任何具体 kernel 的声明。`≤1ulp 比例`/`maxRel` 在 T3 档下是**报告项**。")
        w(f"逐元素超界数在判定项表的 `verdict` 列后附带；格式为 "
          f"`逐元素超界 K/N（其中相消区 M）`。相消区定义为 `|ref| < 0.01·ref_max_abs` —— "
          f"`docs/17` §1.1 已说明单一相对累加界**必然**不覆盖相消元素。")
        w(f"⚠ 本档下表里的 `usage` 与 `max ulp` 两列**不是判据**（已标 `_(非判据)_`）："
          f"T3 的判据是逐元素的那两个数。这两列只在 `l0` / `mx-budget` 档下才参与 PASS/FAIL。")
    else:
        w(f"判据：`max_abs_err <= atol + rtol*ref_max_abs` 且 `非有限掩码不匹配 == 0`"
          f"{ulp_clause}。ulp 只在 ≥2^-{SIG_BINADES}·ref_max_abs 的元素上计算。")
    w("误差数字口径：全部为 **max**（不是首次不匹配），绝对/相对已分列，输入档位见 manifest "
      "的 `extra` 字段。")
    if args.mask_policy == "strict":
        w("非有限值（如 QSA 用 `-inf` 表示「该列未参与打分」）不参与减法比较，只比较"
          "「是否有限」的掩码是否一致。**默认 `--mask-policy strict`**："
          "参考侧非有限而测试侧有限也算不匹配。")
        w(f"若测试侧在这些列 dump 0（官方 kernel 只写 `column < visible_blocks`，"
          f"其余列是未初始化 buffer，`ops/qsa_indexer.py:101-107`），本报告会判 FAIL。"
          f"用 `--mask-policy ignore-unscored`（作用于 `--unscored-segments`，默认 "
          f"`{UNSCORED_SEGMENTS_DEFAULT}`）可把这些位置从**所有**判据里剔除，"
          f"剔除个数在报告项 `ignored_unscored` 单列。")
    else:
        w(f"`--mask-policy ignore-unscored`：对 `{args.unscored_segments or '（空）'}` 的"
          f"**参考侧非有限位置**（= 官方 kernel 未写过的列）已从所有判据中剔除，"
          f"剔除个数见报告项 `ignored_unscored`。其余段的掩码仍按 strict 比较。")
    w(f"行内排序后再比较的段（上游 top-k 顺序未定义）："
      f"`{args.sort_rows or '（无）'}`。")
    if args.profile == "l0":
        t3_here = [n for n, _, _, _ in rows if n in T3_SEGMENTS]
        if t3_here:
            w()
            w(f"⚠️ **档位提示**：被你比对的段里有 {len(t3_here)} 个按 `docs/17` §1.1 属 **T3**"
              f"（超越函数 / 跨 tile·online 重标定 / mmad 累加），而 `l0` 的判据是 `max_ulp=1`。"
              f"这些段若 FAIL，先确认不是档位问题：`{', '.join(sorted(t3_here))}`。"
              f"用 `--profile t3` 走逐元素 `rtol·|ref| + 0.5·ulp(out)` 口径。"
              f"（本工具**不会**自动换档 —— 换档必须显式，见 README §6 的档位声明表。）")
    w()

    w("## 2 报告项 (report only — NOT counted in PASS/FAIL)")
    w()
    w("| segment | bit-exact | bit-exact rate | ≤1 ulp rate | max rel err (max, relative) | "
      "显著元素/总元素 | ignored_unscored |")
    w("|---|---|---|---|---|---|---|")
    for name, dtype, r, _ in rows:
        ber = r["bit_exact"] / r["n"] if r["n"] else 0.0
        le1 = r["le1ulp"] / r["n"] if r["n"] else 0.0
        w(f"| `{name}` | {r['bit_exact']}/{r['n']} | {ber*100:.2f}% | {le1*100:.2f}% | "
          f"{r['max_rel_err']:.6g} | {r['n_significant']}/{r['n']} | "
          f"{r['ignored_unscored']} |")
    w()
    agree_rows = [x for x in rows if x[2].get("argmax_agree_total")]
    if agree_rows:
        w(f"### 2b argmax / top-k 一致率（`docs/17` §1 L2 指标②，**报告项**）")
        w()
        w(f"| segment | argmax 一致行 | top-{AGREEMENT_TOPK} 集合完全一致行 | "
          f"top-{AGREEMENT_TOPK} 元素重合率 |")
        w("|---|---|---|---|")
        for name, dtype, r, _ in agree_rows:
            tot = r["argmax_agree_total"]
            w(f"| `{name}` | {r['argmax_agree_rows']}/{tot} "
              f"({100.0*r['argmax_agree_rows']/tot:.2f}%) | "
              f"{r['topk_set_rows']}/{r['topk_set_total']} "
              f"({100.0*r['topk_set_rows']/r['topk_set_total']:.2f}%) | "
              f"{100.0*r['topk_overlap']/(r['topk_set_total']*r['topk_k']):.2f}% |")
        w()
    w(f"行内一致性口径：argmax 越界行（全 `-inf`）已跳过；top-k 的 k = "
      f"min({AGREEMENT_TOPK}, 该行有限元素数)。")
    w()

    text = "\n".join(out)
    print(text)
    if args.md_out:
        Path(args.md_out).write_text(text + "\n")
        print(f"[compare_dumps] report written to {args.md_out}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"ref": str(ref_dir), "test": args.test, "profile": args.profile,
             "mask_policy": args.mask_policy,
             "judgements": num_judgements, "pass": n_pass,
             "rows": {n: r for n, _, r, _ in rows}}, indent=1))
    if num_judgements == 0:
        print("RESULT: SKIPPED (0 judgement items compared; NOT a pass)")
        return 2
    return 0 if n_pass == num_judgements else 1


def nonhollow(args, ref_dir: Path, ref_man: dict) -> int:
    """Input sensitivity / state evolution: reference vs a second reference run."""
    other = Path(args.ref2)
    other_man = json.loads((other / "manifest.json").read_text())["segments"]
    meta_a = json.loads((ref_dir / "manifest.json").read_text()).get("extra", {})
    meta_b = json.loads((other / "manifest.json").read_text()).get("extra", {})
    print("# Non-hollowness check (reference vs reference, different input)\n")
    print(f"- A: `{ref_dir}` extra={meta_a}")
    print(f"- B: `{other}` extra={meta_b}\n")
    print("| segment | changed elements | max abs delta | verdict |")
    print("|---|---|---|---|")
    changed = 0
    total = 0
    for name, entry in sorted(ref_man.items()):
        if name not in other_man:
            continue
        if not name.startswith(("attn_hc.", "mlp_hc.", "gdn.", "qsa.", "moe.", "layer.out")):
            continue
        a = load_dump(ref_dir, name, entry)
        b = load_dump(other, name, other_man[name])
        if a.shape != b.shape:
            print(f"| `{name}` | shape mismatch {tuple(a.shape)} vs {tuple(b.shape)} | - | **CHECK** |")
            continue
        # Only compare where BOTH sides are finite. `qsa.index_logits` uses -inf
        # for "column not scored" (ops/qsa_indexer.py:101-107); (-inf) - (-inf)
        # is nan, which must never reach the report (docs/17 §2.3).
        af, bf = a.float(), b.float()
        both = torch.isfinite(af) & torch.isfinite(bf)
        n_nonfinite = int((~both).sum())
        d = torch.where(both, (af - bf).abs(), torch.zeros_like(af))
        n_changed = int((d > 0).sum())
        delta = f"{float(d.max()):.6g}" if bool(both.any()) else "n/a (no finite pair)"
        if n_nonfinite:
            delta += f" [non-finite pairs skipped: {n_nonfinite}/{d.numel()}]"
        total += 1
        if n_changed:
            changed += 1
        verdict = "moves" if n_changed else "**CONSTANT**"
        print(f"| `{name}` | {n_changed}/{d.numel()} | {delta} | {verdict} |")
    if total == 0:
        print("RESULT: SKIPPED (0 segments compared -- the two directories share "
              "no comparable segment names; NOT a pass)")
        return 2
    print(f"\n**{changed}/{total} output segments react to the input change** "
          f"(0 would mean the dump is hollow)")
    print(f"RESULT: {'OK' if changed == total else 'FAIL'} "
          f"({total}/{total} segments compared, {changed} moved)")
    return 0 if changed == total else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference dump directory (has manifest.json)")
    ap.add_argument("--test", help="kernel dump directory (has manifest.json)")
    ap.add_argument("--test-bin", action="append", default=[],
                    help="NAME=PATH:DTYPE:shape,shape  raw device blob")
    ap.add_argument("--map", help="JSON mapping reference segment -> test segment")
    ap.add_argument("--tolerance", help="JSON tolerance overrides")
    ap.add_argument("--profile", default="l0", choices=["l0", "mx-budget", "t3"],
                    help="l0 = T1/T2 (bit-exact / <=1ulp); mx-budget = L1 MXFP4 budget; "
                         "t3 = docs/17 §1.1 T3 (per-element eps*|ref| + 0.5*ulp(out), "
                         "<=1ulp rate demoted to a report item)")
    ap.add_argument("--segments", help="comma-separated segment subset")
    ap.add_argument("--sort-rows", default="qsa.block_indices,qsa.token_indices",
                    help="index-valued segments whose rows are sorted before comparing "
                         "(their top-k order is unspecified upstream); pass an empty "
                         "string to disable")
    ap.add_argument("--agreement-rows", default=AGREEMENT_SEGMENTS_DEFAULT,
                    help="segments for which argmax / top-k agreement is reported "
                         "(report-only); pass an empty string to disable")
    ap.add_argument("--mask-policy", default="strict", choices=["strict", "ignore-unscored"],
                    help="strict (default): a reference-side non-finite element must "
                         "match a test-side non-finite element. ignore-unscored: for "
                         "--unscored-segments, reference-side non-finite positions "
                         "(= columns the official kernel never wrote) are dropped from "
                         "every criterion and counted in the report")
    ap.add_argument("--unscored-segments", default=UNSCORED_SEGMENTS_DEFAULT,
                    help="segments the ignore-unscored policy applies to")
    ap.add_argument("--json-out")
    ap.add_argument("--md-out")
    ap.add_argument("--ref2", help="second reference dump for --nonhollow-only")
    ap.add_argument("--nonhollow-only", action="store_true")
    args = ap.parse_args()
    if args.segments:
        args.segments = {s.strip() for s in args.segments.split(",") if s.strip()}
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
