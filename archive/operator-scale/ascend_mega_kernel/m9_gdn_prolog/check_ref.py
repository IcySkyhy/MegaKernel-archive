# SPDX-License-Identifier: Apache-2.0
"""M9 GDN 预处理核数值校验：device 输出 vs numpy float32 参考。

两条路：
  (1) m=1 路（`m9_case_b*`，m9 现有 m=1 段体，q ×1/√128）—— 口径与历史一字未改；
  (2) m>1 路（`m9mt_*`，MTP/prefill prolog 段体，跨 token conv + conv_state 交接）：
      **默认（qscale=0）= 相位 A 契约**：q/k 均不乘 1/√128，scale 由相位 A 消费者施加
      （`m15_gdn_prefill.h:967`/`:1051`、`m15_layer_kernel.h:722`）；
      带 `qscale=1` 的 m=1 档用于与 m9 独立模块做交叉复现（q ×1/√128，`m9_gdn_prolog.asc:75`）。

核语义（docs/10-gdn-analysis.md §1、m21_layer_ref/ref/gdn.py:115-155）：
    conv1d depthwise K=4 over 10240ch（q|k|v 拼接）+ bias + SiLU（fp32），
    conv_state bf16 planar [3][10240]：m 个 token 的窗口 = concat=[state;x] 的滑动 4 行，
    conv_state_out = concat 的最后 3 行；q/k per-key-head l2norm(eps=1e-6)；
    gating per value head：g = -exp(A_log)*softplus(a+dt_bias)（beta=1, thr=20），
    beta = sigmoid(b)；a/b 从 in_proj 输出 bf16 段解码。

输出几何（相位 A 输入契约，docs/22-prefill-prolog-epilog-wiring.md §5.2/§5.3）：
    q,k [16, m+64, 128] fp32（每 head 尾部 64 行必须为 0）；v [48,m,128] fp32；
    g,beta [48, align8(m)] fp32。

校验项（判据分类口径见 docs/22:333-344）：
    T1 逐位（容差 0）：conv_state_out 的 bf16 位型、q/k 尾部 pad 行、g/beta 行内 pad 列；
    T3（组合容差 |got-exp| <= 1e-5*|exp| + 1e-6）：q/k/v/g/beta 有效区。

用法（在 m9_gdn_prolog 可执行文件 dump 出的目录运行，即 build/ 或等价目录）：
    /usr/local/python3.12.13/bin/python3 check_ref.py
    /usr/local/python3.12.13/bin/python3 check_ref.py --mutant 1   # 反向对照：必须变红
    /usr/local/python3.12.13/bin/python3 check_ref.py --mutant 2   # 反向对照：必须变红
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "golden"))
from moe_block_ref import bf16_bits_to_f32, f32_to_bf16_bits  # pure-numpy bf16 helpers

BLK = [56, 28, 16, 8, 4]  # 与 host main() 保持一致
RTOL = 1e-5
ATOL = 1e-6

HD = 128
CH = 10240
VH = 48
NQB = 16
INW = 16480
XB_OFF = 16384
XA_OFF = 16432
MT_PAD = 64
EPS = np.float32(1e-6)
QSCALE = np.float32(0.08838834764831845)  # 1/sqrt(128)
SPTH = np.float32(20.0)


# ============================================================
# m=1 路（历史口径，一字未改）
# ============================================================
def load_case(prefix):
    f32 = np.float32
    x = np.fromfile(f"{prefix}x.bin", dtype=np.uint16)
    s0 = np.fromfile(f"{prefix}conv_state_init.bin", dtype=np.uint16).reshape(3, CH)  # planar [3][10240]
    s_dev = np.fromfile(f"{prefix}conv_state_out.bin", dtype=np.uint16).reshape(3, CH)
    w = np.fromfile(f"{prefix}w.bin", dtype=np.uint16).reshape(4, CH)
    bias = np.fromfile(f"{prefix}bias.bin", dtype=np.uint16)
    al = np.fromfile(f"{prefix}a_log.bin", dtype=f32)
    dt = np.fromfile(f"{prefix}dt_bias.bin", dtype=f32)
    q = np.fromfile(f"{prefix}q.bin", dtype=f32).reshape(16, HD)
    k = np.fromfile(f"{prefix}k.bin", dtype=f32).reshape(16, HD)
    v = np.fromfile(f"{prefix}v.bin", dtype=f32).reshape(VH, HD)
    g = np.fromfile(f"{prefix}g.bin", dtype=f32)
    beta = np.fromfile(f"{prefix}beta.bin", dtype=f32)
    return x, s0, s_dev, w, bias, al, dt, q, k, v, g, beta


def reference(x, s0, w, bias, al, dt):
    """numpy float32 逐公式实现（与 kernel 同序：bias; += w_j*win_j; silu）。
    conv_state planar [3][CH]：行 j = 全通道第 j 个历史样本。"""
    f32 = np.float32
    xf = bf16_bits_to_f32(x)  # [16480] fp32
    s0f = bf16_bits_to_f32(s0)  # [3,CH]
    xq = xf[:CH]
    win = np.stack([s0f[0], s0f[1], s0f[2], xq], axis=0)  # [4,CH] = [s0,s1,s2,x]
    acc = bf16_bits_to_f32(bias).astype(f32)
    for j in range(4):
        acc = (acc + bf16_bits_to_f32(w[j]) * win[j]).astype(f32)
    y = (acc / (f32(1.0) + np.exp(-acc))).astype(f32)  # SiLU
    s_ref = np.stack([s0f[1], s0f[2], xq], axis=0).astype(f32)  # new[0]=old[1], new[1]=old[2], new[2]=x
    # q/k l2norm（eps 在根号内；仅 q 乘 1/sqrt(128)）
    qk = y[: 2 * 2048].reshape(32, HD)
    n = np.sqrt((qk * qk).sum(axis=1, keepdims=True) + EPS).astype(f32)
    q_ref = (qk / n * QSCALE).astype(f32)   # [32,128]，前 16 行是 q
    k_ref = (qk / n).astype(f32)
    v_ref = y[4096:].reshape(VH, HD)
    # gating（a/b 从 x bf16 段解码）
    a = xf[XA_OFF : XA_OFF + VH]
    b = xf[XB_OFF : XB_OFF + VH]
    xa = (a + dt[:VH]).astype(f32)
    sp = np.where(xa <= SPTH, np.log1p(np.exp(xa)).astype(f32), xa).astype(f32)
    g_ref = (-np.exp(al[:VH]) * sp).astype(f32)
    beta_ref = (f32(1.0) / (f32(1.0) + np.exp(-b))).astype(f32)
    return q_ref[:16], k_ref[16:], v_ref, g_ref, beta_ref, s_ref


def compare(got, expect):
    adiff = np.abs(got.astype(np.float32) - expect.astype(np.float32))
    ok = np.all(adiff <= RTOL * np.abs(expect) + ATOL)
    rdiff = adiff / (np.abs(expect) + ATOL)
    idx = np.unravel_index(np.argmax(rdiff), rdiff.shape)
    return ok, float(rdiff.max()), float(adiff.max()), idx, float(got[idx]), float(expect[idx])


def check_m1():
    all_ok = True
    for blk in BLK:
        prefix = f"m9_case_b{blk:02d}_"
        try:
            x, s0, s_dev, w, bias, al, dt, q, k, v, g, beta = load_case(prefix)
        except FileNotFoundError as e:
            print(f"[{prefix}] MISSING FILE: {e}（请先运行 m9_gdn_prolog 生成 dump）")
            all_ok = False
            continue
        qn, kn, v_ref, g_ref, beta_ref, s_ref = reference(x, s0, w, bias, al, dt)

        # conv_state：位型精确一致（纯 bf16 搬移 [s1,s2,x]）
        s_ref_bits = f32_to_bf16_bits(s_ref)
        state_ok = np.array_equal(s_dev, s_ref_bits)

        checks = [
            ("q", q, qn), ("k", k, kn), ("v", v, v_ref),
            ("g", g, g_ref), ("beta", beta, beta_ref),
        ]
        case_ok = state_ok
        print(f"[{prefix}] state: {'PASS(bit-exact)' if state_ok else 'FAIL'}")
        if not state_ok:
            bad = np.argwhere(s_dev != s_ref_bits)
            i = tuple(bad[0])
            print(f"    state worst at {i}: got {s_dev[i]:04x} expect {s_ref_bits[i]:04x} ({bad.shape[0]} mismatches)")
        for name, got, exp in checks:
            ok, rmax, amax, idx, gv, ev = compare(got, exp)
            case_ok &= ok
            print(f"[{prefix}] {name:4s}: {'PASS' if ok else 'FAIL'} maxRelDiff={rmax:.3e} maxAbsDiff={amax:.3e}")
            if not ok:
                print(f"    worst at {idx}: got {gv:.8e} expect {ev:.8e}")
        all_ok &= case_ok
    return all_ok


# ============================================================
# m>1 路（MTP / prefill prolog）
# ============================================================
def read_meta(path):
    meta = {}
    with open(path, "r") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                meta[p[0]] = p[1]
    return meta


def reference_mt(x, s0, w, bias, al, dt, m, qscale):
    """m 个 token 的 fp32 逐公式参考；conv_state_out = [state;x] 的最后 3 行。
    qscale=1 时 q 乘 1/√128（m9 独立模块语义，m=1 交叉复现用）；qscale=0 时 q/k 均不乘
    （相位 A 契约，见文件头）。"""
    f32 = np.float32
    xf = bf16_bits_to_f32(x).astype(f32)          # [m, INW]
    s0f = bf16_bits_to_f32(s0).astype(f32)        # [3, CH]
    wf = bf16_bits_to_f32(w).astype(f32)          # [4, CH]
    bf = bf16_bits_to_f32(bias).astype(f32)       # [CH]
    concat = np.concatenate([s0f, xf[:, :CH]], axis=0)  # [m+3, CH] = [state;x]
    y = np.empty((m, CH), dtype=f32)
    for t0 in range(0, m, 64):                    # 分块控内存
        t1 = min(m, t0 + 64)
        idx = np.arange(t0, t1)[:, None] + np.arange(4)[None, :]  # [ct,4]
        win = concat[idx]                         # [ct,4,CH] = [concat[t..t+3]]
        acc = np.broadcast_to(bf, (t1 - t0, CH)).copy()
        for j in range(4):
            acc = (acc + wf[j][None, :] * win[:, j, :]).astype(f32)
        y[t0:t1] = (acc / (f32(1.0) + np.exp(-acc))).astype(f32)  # SiLU
    s_ref = concat[m : m + 3].astype(f32)         # conv_state_out = 最后 3 行
    # q/k per-head l2norm
    qk = y[:, : 2 * 2048].reshape(m, 32, HD)
    n = np.sqrt((qk * qk).sum(axis=2, keepdims=True) + EPS).astype(f32)
    q_ref = (qk[:, :16, :] / n[:, :16, :]).astype(f32)
    if qscale:
        q_ref = (q_ref * QSCALE).astype(f32)
    k_ref = (qk[:, 16:, :] / n[:, 16:, :]).astype(f32)
    v_ref = y[:, 4096:].reshape(m, VH, HD)
    # gating per value head per token
    a = xf[:, XA_OFF : XA_OFF + VH]
    b = xf[:, XB_OFF : XB_OFF + VH]
    xa = (a + dt[:VH][None, :]).astype(f32)
    sp = np.where(xa <= SPTH, np.log1p(np.exp(xa)).astype(f32), xa).astype(f32)
    g_ref = (-np.exp(al[:VH])[None, :] * sp).astype(f32)          # [m, VH]
    beta_ref = (f32(1.0) / (f32(1.0) + np.exp(-b))).astype(f32)   # [m, VH]
    return q_ref, k_ref, v_ref, g_ref, beta_ref, s_ref


def check_mt_case(prefix):
    meta = read_meta(prefix + "meta.txt")
    m = int(meta["m"])
    tp = int(meta["tp"])
    qkStride = int(meta["qkstride"])
    qscale = int(meta["qscale"])
    mut = int(meta["mut"])
    f32 = np.float32
    x = np.fromfile(prefix + "x.bin", dtype=np.uint16).reshape(m, INW)
    s0 = np.fromfile(prefix + "conv_state_init.bin", dtype=np.uint16).reshape(3, CH)
    s_dev = np.fromfile(prefix + "conv_state_out.bin", dtype=np.uint16).reshape(3, CH)
    w = np.fromfile(prefix + "w.bin", dtype=np.uint16).reshape(4, CH)
    bias = np.fromfile(prefix + "bias.bin", dtype=np.uint16)
    al = np.fromfile(prefix + "a_log.bin", dtype=f32)
    dt = np.fromfile(prefix + "dt_bias.bin", dtype=f32)
    q_dev = np.fromfile(prefix + "q.bin", dtype=f32).reshape(NQB, qkStride, HD)
    k_dev = np.fromfile(prefix + "k.bin", dtype=f32).reshape(NQB, qkStride, HD)
    v_dev = np.fromfile(prefix + "v.bin", dtype=f32).reshape(VH, m, HD)
    g_dev = np.fromfile(prefix + "g.bin", dtype=f32).reshape(VH, tp)
    beta_dev = np.fromfile(prefix + "beta.bin", dtype=f32).reshape(VH, tp)

    q_ref, k_ref, v_ref, g_ref, beta_ref, s_ref = reference_mt(x, s0, w, bias, al, dt, m, qscale)

    # T1：conv_state_out 位型精确一致
    state_bits = f32_to_bf16_bits(s_ref)
    state_ok = np.array_equal(s_dev, state_bits)
    # T1：q/k 尾部 pad 行 [m, m+64) 必须为 0；g/beta 行内 pad 列 [m, tp) 必须为 0
    q_pad = bool(np.all(q_dev[:, m:, :] == f32(0.0)))
    k_pad = bool(np.all(k_dev[:, m:, :] == f32(0.0)))
    g_pad = bool(np.all(g_dev[:, m:] == f32(0.0)))
    b_pad = bool(np.all(beta_dev[:, m:] == f32(0.0)))

    checks = [
        ("q", q_dev[:, :m, :], np.transpose(q_ref, (1, 0, 2))),
        ("k", k_dev[:, :m, :], np.transpose(k_ref, (1, 0, 2))),
        ("v", v_dev, np.transpose(v_ref, (1, 0, 2))),
        ("g", g_dev[:, :m], g_ref.T),
        ("beta", beta_dev[:, :m], beta_ref.T),
    ]
    case_ok = state_ok and q_pad and k_pad and g_pad and b_pad
    flags = f"m={m} blk={meta['blk']} mut={mut} qs={qscale}"
    print(f"[{prefix}] {flags} state={'PASS' if state_ok else 'FAIL'} "
          f"qpad={'PASS' if q_pad else 'FAIL'} kpad={'PASS' if k_pad else 'FAIL'} "
          f"gpad={'PASS' if g_pad else 'FAIL'} bpad={'PASS' if b_pad else 'FAIL'}")
    for name, got, exp in checks:
        ok, rmax, amax, idx, gv, ev = compare(got, exp)
        case_ok &= ok
        print(f"[{prefix}] {name:4s}: {'PASS' if ok else 'FAIL'} maxRelDiff={rmax:.3e} maxAbsDiff={amax:.3e}")
        if not ok:
            print(f"    worst at {idx}: got {gv:.8e} expect {ev:.8e}")
    return case_ok, mut


def check_mt(mutant):
    metas = sorted(glob.glob("m9mt_*_meta.txt"))
    if not metas:
        print("[M9MT] 没找到 dump（先跑 ./m9_gdn_prolog 生成 m9mt_*_meta.txt）")
        return False if mutant == 0 else None
    expected = []
    results = []
    for mt in metas:
        prefix = mt[: -len("meta.txt")]
        mut = int(read_meta(mt)["mut"])
        if mutant == 0:
            if mut != 0:
                continue
        else:
            if mut != mutant:
                continue
        expected.append(prefix)
        case_ok, _ = check_mt_case(prefix)
        results.append((prefix, case_ok))
    if not expected:
        print(f"[M9MT] 没有 mut={mutant} 的档（反向对照需要 host 跑出对应 mut 档）")
        return False if mutant == 0 else None
    if mutant == 0:
        ok = all(r for _, r in results)
        print(f"[M9MT] {'ALL PASS' if ok else 'FAILURES PRESENT'}（{len(results)} 档）")
        return ok
    # 反向对照：期望全红
    all_red = all(not r for _, r in results)
    print(f"[M9MT] 反向对照（mut{mutant}）：{'如预期变红 ✓' if all_red else '判据没咬住 ✗'}（{len(results)} 档）")
    return all_red


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", type=int, default=0, help="只判 mut=k 的反向对照档（期望 FAIL）")
    args = ap.parse_args()

    if args.mutant != 0:
        res = check_mt(args.mutant)
        if res is None:
            sys.exit(2)
        sys.exit(0 if res else 1)

    all_ok = check_m1()
    mt_ok = check_mt(0)
    all_ok &= (mt_ok if mt_ok is not None else True)
    print(f"[M9] numpy fp32 reference check (rtol {RTOL:g} + atol {ATOL:g}; state/pad bit-exact): "
          f"{'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
