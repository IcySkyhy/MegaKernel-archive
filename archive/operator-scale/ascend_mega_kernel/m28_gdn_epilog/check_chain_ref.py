#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Chain cross-check for m28_epilog_chain: NPU device dumps vs an independent host chain.

被测对象：`m28_gdn_epilog/m28_epilog_chain.asc` 的 epilog 链段
（转位/落位 → S5 RMSNormGated → S6 out_proj → hcAttnOut）。

参考（本脚本，端到端链；S5 段直接调用 **m21_layer_ref/ref/gdn.py** 的
`GDN._rms_norm_gated`（`m21_layer_ref/ref/gdn.py:183-196`，norm_before_gate=True /
act=sigmoid / eps=1e-6），S6 段用它 `out_proj`（同文件 `:94`，[2560,6144]）的语义）：

    o[t, h*128+d] = wsO[h, t, d]                 # 转位
    z[t, c]       = qkvzba[t, 10240 + c]         # z 落位
    y             = bf16( GDN._rms_norm_gated(o[m,48,128], z[m,48,128]) )   # S5
    hcAttnOut     = y_bf16 @ Wout^T              # S6

判据分档（`docs/22-prefill-prolog-epilog-wiring.md:333-344`，并注明各段用哪一档）：
    T1（容差 0）：转位 o、z 落位——纯数据搬移。
    S5（bf16 网格 rel ≤ 1e-2）：走 `m12_rmsnorm_gated/check_ref.py` 的既有口径
        （RMSNormGated 出口是 bf16，bf16 网格量化 ~4e-3 相对，逐元素 1e-2 是该段既定判据）。
    T3（`docs/22:341` / m18 口径）：S6 out_proj 含 fp32 累加链，取
        `|got - ref| ≤ ε·Σ|terms| + 0.5·ulp_bf16(ref)`，ε = 5e-5。
        说明：段出口仍是 bf16，**逐元素 `1e-5·|exp|+1e-6` 不适用于 bf16 网格**
        （bf16 量化本身 ~4e-3 相对，正确核也会「超界」）；脚本仍报告该严格式的
        逐元素超界计数（信息量，不作为 PASS 条件；m18/m11 的 T3 亦用 Σ|terms| 形式）。

模式：
    pos（默认）：设备输出必须与链参考一致（T1 位级 + S5/S6 在容差内）。
    mut        ：段①仍须位级一致；S5 出口必须**超出容差**（负向对照变红）。

用法：
    <baseline-venv python> check_chain_ref.py <dump_dir> [--mode pos|mut]
退出码：0 = 全部满足预期；1 = 有反例；2 = 没得比（无 dump）。
"""

import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

sys.path.insert(0, os.path.join(REPO, "tools", "golden"))
from moe_block_ref import bf16_bits_to_f32, f32_to_bf16_bits  # noqa: E402

HEADS, HEAD, HIDDEN = 48, 128, 6144
IN_N, Z_OFF, Z_DIM = 16480, 10240, 6144
OUT_N = 2560
EPS = 1e-6
S5_TOL = 1e-2     # m12 check_ref 口径（bf16 网格相对）
S6_EPS = 5e-5     # docs/22:341 / m18 T3 口径（ε·Σ|terms| + 0.5·ulp）

TINY = 2.0 ** -100


def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                meta[p[0]] = int(p[1])
    return meta


def load_gdn_ref():
    spec = importlib.util.spec_from_file_location(
        "gdn_ref", os.path.join(REPO, "m21_layer_ref", "ref", "gdn.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _W:
    pass


def build_gdn(mod, gamma_bits, wout_bits):
    """Construct m21_layer_ref GDN with only the fields _rms_norm_gated/out_proj need."""
    w = _W()
    w.norm_weight = torch.from_numpy(bf16_bits_to_f32(gamma_bits.astype(np.uint16)))
    w.out_proj_weight = torch.from_numpy(bf16_bits_to_f32(wout_bits.astype(np.uint16)))
    # shapes/index only — not touched by _rms_norm_gated / out_proj
    w.in_proj_qkvz = torch.zeros(16384, 2560, dtype=torch.bfloat16)
    w.in_proj_ba = torch.zeros(96, 2560, dtype=torch.bfloat16)
    w.conv1d_weight = torch.zeros(10240, 4, dtype=torch.bfloat16)
    w.A_log = torch.zeros(48)
    w.dt_bias = torch.zeros(48, dtype=torch.bfloat16)
    cfg = dict(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        hidden_size=2560,
        rms_norm_eps=EPS,
        output_gate_type="sigmoid",
    )
    return mod.GDN(w, cfg)


def rel_err_bf16(got_bits, exp_bits):
    got = bf16_bits_to_f32(got_bits.astype(np.uint16)).astype(np.float32)
    exp = bf16_bits_to_f32(exp_bits.astype(np.uint16)).astype(np.float32)
    den = np.abs(exp)
    num = np.abs(got - exp)
    return np.where(den > 0, num / np.maximum(den, np.float32(1e-38)),
                    np.where(got == 0, 0.0, 1.0))


def strict_t3_over(got_f32, exp_f64, eps=1e-5, floor=1e-6):
    """逐元素 1e-5·|exp|+1e-6 超界计数（信息量；bf16 出口不以此判定）。"""
    return int((np.abs(got_f32 - exp_f64) > (eps * np.abs(exp_f64) + floor)).sum())


def check_case(mod, prefix, expect_equal):
    meta = read_meta(prefix + "_meta.txt")
    m = meta["m"]
    assert meta["heads"] == HEADS and meta["hidden"] == HIDDEN and meta["out_n"] == OUT_N

    wsO = np.fromfile(prefix + "_wsO.bin", dtype=np.float32).reshape(HEADS, m, HEAD)
    qkv = np.fromfile(prefix + "_qkvzba.bin", dtype=np.uint16).reshape(m, IN_N)
    gamma = np.fromfile(prefix + "_gamma.bin", dtype=np.uint16)
    wout = np.fromfile(prefix + "_wout.bin", dtype=np.uint16).reshape(OUT_N, HIDDEN)
    o_dev = np.fromfile(prefix + "_o.bin", dtype=np.float32).reshape(m, HIDDEN)
    z_dev = np.fromfile(prefix + "_z.bin", dtype=np.uint16).reshape(m, Z_DIM)
    y_dev = np.fromfile(prefix + "_y_device.bin", dtype=np.uint16).reshape(m, HIDDEN)
    out_dev = np.fromfile(prefix + "_hcattnout_device.bin", dtype=np.uint16).reshape(m, OUT_N)

    # ---- T1（容差 0）：转位 o / z 落位 ----
    r_o = np.transpose(wsO, (1, 0, 2)).reshape(m, HIDDEN)
    r_z = qkv[:, Z_OFF:Z_OFF + Z_DIM]
    o_eq = bool(np.array_equal(o_dev.view(np.uint32), r_o.view(np.uint32)))
    z_eq = bool(np.array_equal(z_dev, r_z))

    # ---- S5：m21_layer_ref/ref/gdn.py 的 _rms_norm_gated ----
    g = build_gdn(mod, gamma, wout)
    x = torch.from_numpy(r_o).reshape(m, HEADS, HEAD)  # fp32（vllm core_attn_out 为 fp32）
    zb = torch.from_numpy(r_z).view(torch.bfloat16).reshape(m, HEADS, HEAD)
    y_ref = g._rms_norm_gated(x, zb)  # fp32 [m,48,128]
    y_ref_b16 = y_ref.reshape(m, HIDDEN).to(torch.bfloat16)
    y_ref_bits = y_ref_b16.view(torch.uint16).numpy().astype(np.uint16)
    y_err = rel_err_bf16(y_dev, y_ref_bits)
    y_ok = bool(np.all(y_err <= S5_TOL))

    # ---- S6（隔离档）：参考 GEMM 用 **device 的 S5 输出**作 A，T3（ε·Σ|terms| + 0.5·ulp）----
    wout_f32 = bf16_bits_to_f32(wout).astype(np.float32)
    wout64 = wout_f32.astype(np.float64)
    y_dev_f32 = bf16_bits_to_f32(y_dev).astype(np.float32)
    got_f32 = bf16_bits_to_f32(out_dev).astype(np.float32)

    def gemm_t3(a_f32, b64, got):
        ref = a_f32.astype(np.float64) @ b64.T
        scale = np.abs(a_f32.astype(np.float64)) @ np.abs(b64).T
        ulp = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(ref), TINY))) - 7)
        over = np.abs(got.astype(np.float64) - ref) > (S6_EPS * scale + 0.5 * ulp)
        worst = float(np.max(np.abs(got.astype(np.float64) - ref) / scale))
        return ref, over, worst

    _, s6_over, s6_worst = gemm_t3(y_dev_f32, wout64, got_f32)
    s6_ok = bool(not s6_over.any())

    # ---- 链端到端档：参考 GEMM 用 **m21 参考的 S5 输出**作 A（含 S5 差值的传播）----
    y_ref_f32 = bf16_bits_to_f32(y_ref_bits).astype(np.float32)
    _, chain_over, chain_worst = gemm_t3(y_ref_f32, wout64, got_f32)
    chain_frac = float(chain_over.mean())

    # 交叉一致性（信息量）：m21 的 torch bf16 out_proj 路径 vs bf16(ref64)
    ref64 = y_ref_f32.astype(np.float64) @ wout64.T
    out_m21 = (y_ref_b16 @ g.out_proj.t()).view(torch.uint16).numpy().astype(np.uint16)
    m21_diff = int((out_m21 != f32_to_bf16_bits(ref64.astype(np.float32))).sum())

    s5_strict = strict_t3_over(bf16_bits_to_f32(y_dev).astype(np.float32),
                               bf16_bits_to_f32(y_ref_bits).astype(np.float64))
    s6_strict = strict_t3_over(got_f32, ref64)

    if expect_equal:
        ok = o_eq and z_eq and y_ok and s6_ok and (chain_frac <= 1e-4)
        verdict = "PASS" if ok else "FAIL"
        print(f"[{prefix}] m={m}: o(T1)={o_eq} z(T1)={z_eq} | S5 bf16 rel<= {S5_TOL}: {y_ok} "
              f"(max {y_err.max():.3g}) | S6 T3-isolated over: {int(s6_over.sum())}/{out_dev.size} "
              f"(worst {s6_worst:.3g} of Σ|terms|) | chain T3 over-frac {chain_frac:.3g} "
              f"(worst {chain_worst:.3g}) | m21 out_proj vs bf16(fp64): {m21_diff} diff -> {verdict}")
    else:
        red = not y_ok            # 负向对照以 S5 出口变红为准
        chain_red = chain_frac > 0.5   # 且必须传播到 hcAttnOut
        ok = o_eq and z_eq and red and chain_red
        verdict = "PASS" if ok else "FAIL"
        print(f"[{prefix}] m={m}: o(T1)={o_eq} z(T1)={z_eq} | S5 red(want True): {red} (max rel {y_err.max():.3g}) "
              f"| chain T3 over-frac {chain_frac:.3g} (want > 0.5): {chain_red} -> {verdict}")
    print(f"    (info) strict 1e-5|exp|+1e-6 count — S5 {s5_strict}/{y_dev.size}, S6 {s6_strict}/{out_dev.size}")
    if not o_eq:
        bad = np.argwhere(o_dev.view(np.uint32) != r_o.view(np.uint32))
        print(f"    o first mismatch {tuple(bad[0])} (total {len(bad)})")
    if not z_eq:
        bad = np.argwhere(z_dev != r_z)
        print(f"    z first mismatch {tuple(bad[0])} (total {len(bad)})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", nargs="?", default=".")
    ap.add_argument("--mode", choices=["pos", "mut"], default="pos")
    args = ap.parse_args()

    metas = sorted(glob.glob(os.path.join(args.dump_dir, "m*_s0_meta.txt")))
    if not metas:
        print(f"RESULT: SKIPPED (no m*_s0_meta.txt under {args.dump_dir})")
        return 2

    mod = load_gdn_ref()
    ok = True
    for mp in metas:
        prefix = mp[: -len("_meta.txt")]
        ok &= check_case(mod, prefix, expect_equal=(args.mode == "pos"))

    print(f"===== chain check mode={args.mode}: {'PASS' if ok else 'FAIL'} =====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
