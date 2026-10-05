#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""seq=4096 残差定位：把 out 的单元素偏差归因到**一个 P̃ 的 bf16 舍入翻转**。

背景（README §6.6）：`check_ref.py` 在 seq=4096 上有 1/6144 个元素越界（absErr 4.21e-7
> 界 3.71e-7，M61 改后）。本脚本用 kernel dump 的**每 split 部分和**（wsacc/wsm/wss）把该偏差逐层定位：

  ① 逐 split 比 acc_i（设备 ws vs numpy 参考）→ 找出唯一异常的那个 split；
  ② 在该 split 内把 Δacc 对每个 token 的 V 行做最小二乘 ⇒ 若 Δacc = α·V_t 且 |α| = 1 个
     bf16 ulp（P̃ 的 ulp），说明是**单个 P̃ 元素的 bf16 舍入翻转**（不是累加精度问题）；
  ③ 把该翻转注入参考重算 ⇒ 若整档变成 100% ≤1ulp、且该元素的位型与设备**逐位相同**，
     则归因成立（因果闭链）。

机制：P̃ 在中途被量化到 bf16（donor 同款设计），而 S 在设备侧是 fp32 mmad、exp 是 **Reg `Exp`**
实现（**官方规格 ≤1 ulp**，见 `check_ref.py` 文件头与 README §6.6）⇒ 当 exp 结果距某个 bf16 量化边界
小于该误差时，P̃ 的舍入方向就与"独立 numpy 参考"不同。该差异对输出的影响 = 2^-8·|P̃_t·V_t|·w/den——
在相消 400× 的输出元素上可达数个输出 ulp，是**设计固有的量化事件**，
不在 fp32 累加误差界 ε·Σ|acc·w| 的量纲内（故判据 A 的绝对界不覆盖它；该事件由 `check_ref.py` 的
**判据 C** 在 P̃ 元素级覆盖：`gp[unit10][slot0][row11][col252]` = 1 个 bf16 格点、界占用 0.9965）。
⇒ 本脚本的"注入翻转 ⇒ 整档 100% ≤1ulp"也正说明：**主判据若改成吃设备 P̃，这一处 FAIL 会被吃掉**
（README §7.1 因此否掉该提议）。

用法（在 dump 所在目录运行；需要 q/k/v/out + wsacc/wsm/wss）：
    /usr/local/python3.12.13/bin/python3 ../check_partials.py [seq] [n2] [g]
默认 seq=4096、取 |Δacc| 最大的那一行。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "golden"))
from check_ref import G, M_PAD, S2T, SCALE, bf16_ulp, load_case, split_ranges  # noqa: E402
from moe_block_ref import bf16_bits_to_f32, f32_to_bf16_bits  # noqa: E402

SPLITS = 14
U = 2.0 ** -24


def split_partials(q, k, v, seq, dtype=np.float64):
    """参考的每 split 部分和（acc[m][16][256], m[m][16], sum[m][16]），逐 tile 运行 max 重标定。"""
    qf = bf16_bits_to_f32(q)[:, :G, :].astype(dtype)
    kf = bf16_bits_to_f32(k).astype(dtype)
    vf = bf16_bits_to_f32(v).astype(dtype)
    s_all = (np.einsum("ngd,nkd->ngk", qf, kf) * dtype(SCALE)).astype(dtype)
    n2n = q.shape[0]
    acc_o, m_o, s_o, ab_o = [], [], [], []
    for (lo, hi) in split_ranges(seq):
        m = np.full((n2n, G, 1), -np.inf, dtype=dtype)
        acc = np.zeros((n2n, G, 256), dtype=dtype)
        ss = np.zeros((n2n, G, 1), dtype=dtype)
        ab = np.zeros((n2n, G, 256), dtype=dtype)     # Σ_tile|P̃·V|：mmad k 项保守界的底
        for a in range(lo, hi, S2T):
            b = min(a + S2T, hi)
            s = s_all[:, :, a:b]
            mn = np.maximum(m, s.max(axis=2, keepdims=True))
            e = np.exp((s - mn).astype(dtype))
            p = bf16_bits_to_f32(f32_to_bf16_bits(e.astype(np.float32))).astype(dtype)
            pv = np.einsum("ngk,nkd->ngd", p, vf[:, a:b, :])
            acc = acc * np.exp((m - mn).astype(dtype)) + pv
            ab = ab * np.exp((m - mn).astype(dtype)) + np.abs(pv)
            ss = ss * np.exp((m - mn).astype(dtype)) + p.sum(axis=2, keepdims=True)
            m = mn
        acc_o.append(acc)
        m_o.append(m[:, :, 0])
        s_o.append(ss[:, :, 0])
        ab_o.append(ab)
    return np.array(acc_o), np.array(m_o), np.array(s_o), np.array(ab_o)


def ref_out_with_flip(q, k, v, seq, flip):
    """参考的最终 out（fp64）。flip = (split_idx, row, token, value) 或 None：把该位置 P̃ 换成 value。"""
    qf = bf16_bits_to_f32(q)[:, :G, :].astype(np.float64)
    kf = bf16_bits_to_f32(k).astype(np.float64)
    vf = bf16_bits_to_f32(v).astype(np.float64)
    s_all = (np.einsum("ngd,nkd->ngk", qf, kf) * np.float64(SCALE)).astype(np.float64)
    num = np.zeros((2, G, 256))
    den = np.zeros((2, G, 1))
    mg = np.full((2, G, 1), -np.inf)
    for si, (lo, hi) in enumerate(split_ranges(seq)):
        m = np.full((2, G, 1), -np.inf)
        acc = np.zeros((2, G, 256))
        ss = np.zeros((2, G, 1))
        for a in range(lo, hi, S2T):
            b = min(a + S2T, hi)
            s = s_all[:, :, a:b]
            mn = np.maximum(m, s.max(axis=2, keepdims=True))
            p = bf16_bits_to_f32(f32_to_bf16_bits(np.exp(s - mn).astype(np.float32))).astype(np.float64)
            if flip is not None and flip[0] == si and a <= flip[2] < b:
                p[0, flip[1], flip[2] - a] = flip[3]
            pv = np.einsum("ngk,nkd->ngd", p, vf[:, a:b, :])
            acc = acc * np.exp(m - mn) + pv
            ss = ss * np.exp(m - mn) + p.sum(axis=2, keepdims=True)
            m = mn
        g2 = np.maximum(mg, m)
        num = num * np.exp(mg - g2) + acc * np.exp(m - g2)
        den = den * np.exp(mg - g2) + ss * np.exp(m - g2)
        mg = g2
    return num / den


def main():
    args = sys.argv[1:]
    seq = int(args[0]) if len(args) > 0 else 4096
    q, k, v, out, _ = load_case(seq)
    ns = len(split_ranges(seq))
    accR, mR, sR, abR = split_partials(q, k, v, seq)
    accR, mR, sR, abR = (np.transpose(x, (1, 0) + tuple(range(2, x.ndim))) for x in (accR, mR, sR, abR))
    wsA = np.fromfile(f"m10_case_s{seq}_wsacc.bin", dtype=np.float32).reshape(2, SPLITS, M_PAD, S2T)[:, :ns, :G, :]
    wsM = np.fromfile(f"m10_case_s{seq}_wsm.bin", dtype=np.float32).reshape(2, SPLITS, M_PAD)[:, :ns, :G]
    wsS = np.fromfile(f"m10_case_s{seq}_wss.bin", dtype=np.float32).reshape(2, SPLITS, M_PAD)[:, :ns, :G]
    dacc_s = wsA.astype(np.float64) - accR                         # (2, ns, G, 256) 有符号差
    dacc = np.abs(dacc_s)
    # ① 逐 split 的异常度：Δacc 相对该 split 的保守 mmad 界 k·u·Σ|P̃V|
    bnd = S2T * U * abR.sum(axis=3)                                   # (2, ns, G)
    ratio = dacc.max(axis=3) / np.maximum(bnd, 1e-30)
    n2w, sw, gw = (int(x) for x in np.unravel_index(int(np.argmax(ratio)), ratio.shape))
    print(f"[partials] seq={seq} nsplit={ns}")
    print(f"  ① 逐 split |Δacc|/保守界(k·u·Σ|P̃V|)：max {ratio.max():.4f} @ (n2={n2w},split={sw},g={gw})，"
          f"其余 split 的最大值 {np.sort(ratio.ravel())[-2]:.6f}")
    print(f"     设备 m/s 与参考：max |Δm| {np.abs(wsM.astype(np.float64) - mR).max():.3e}，"
          f"max |Δsum| {np.abs(wsS.astype(np.float64) - sR).max():.3e}")
    d = dacc_s[n2w, sw, gw]
    print(f"  ② (n2={n2w},split={sw},g={gw})：|Δacc|={np.linalg.norm(d):.3e}，max {d.max():.3e}；"
          f"该 split 的 mmad 保守界 {S2T * U * abR[n2w, sw, gw].sum():.3e}")
    vf = bf16_bits_to_f32(v).astype(np.float64)
    lo, hi = split_ranges(seq)[sw]
    fits = []
    for t in range(lo, hi):
        vt = vf[n2w, t]
        al = float(d @ vt) / float(vt @ vt)
        fits.append((float(np.linalg.norm(d - al * vt)), abs(al), t, al))
    fits.sort()
    # V 是 bf16（8 个尾数位的一部分）⇒ 用 bf16 的 V 行做基底；t 的有效性要求 P̃ 落在网格上
    qf = bf16_bits_to_f32(q)[:, :G, :].astype(np.float64)
    kf = bf16_bits_to_f32(k).astype(np.float64)
    s_all = (np.einsum("ngd,nkd->ngk", qf, kf) * np.float64(SCALE)).astype(np.float64)
    ta = (fits[0][2] // S2T) * S2T
    tb = min(ta + S2T, hi)
    e = np.exp(s_all[n2w, gw, fits[0][2]] - s_all[n2w, :G, ta:tb].max(axis=1)[gw])
    nb = int(f32_to_bf16_bits(np.float32(e))[0])
    p_ref = float(bf16_bits_to_f32(np.uint16(nb)))
    p_nbr = float(bf16_bits_to_f32(np.uint16(nb + 1)))
    ulp_bf16 = abs(p_nbr - p_ref)
    print(f"     Δacc ≈ α·V_t 的最佳拟合：token {fits[0][2]}（残差比 {fits[0][0]:.3e}，α={fits[0][3]:+.6e}）")
    print(f"     Pearson(Δacc, α·V_t) = {np.corrcoef(d, fits[0][3] * vf[n2w, fits[0][2]])[0, 1]:.9f}  ⇒ 秩一结构")
    print(f"     该位置 P̃ 参考值 {p_ref:.9e}（bf16 值），相邻 bf16 {p_nbr:.9e}，ulp = {ulp_bf16:.6e}")
    print(f"     α / ulp(P̃) = {fits[0][3] / ulp_bf16:+.4f}  ⇒ {'= ±1 个 bf16 ulp（P̃ 舍入翻转）' if abs(abs(fits[0][3] / ulp_bf16) - 1) < 0.05 else '非 1 ulp，需另找原因'}")
    tok, p_flip = fits[0][2], p_nbr if fits[0][3] > 0 else float(bf16_bits_to_f32(np.uint16(nb - 1)))
    # ③ 注入该翻转，看目标元素与整档
    eb0 = f32_to_bf16_bits(ref_out_with_flip(q, k, v, seq, None).astype(np.float32)).astype(np.uint16)
    eb1 = f32_to_bf16_bits(ref_out_with_flip(q, k, v, seq, (sw, gw, tok, p_flip)).astype(np.float32)).astype(np.uint16)
    u0, u1 = bf16_ulp(out, eb0), bf16_ulp(out, eb1)
    print(f"  ③ 注入 1 个 P̃ 翻转（split{sw}, g={gw}, token={tok}: {p_ref:.9e} → {p_flip:.9e}）：")
    print(f"     整档 位级 {float((u0 == 0).mean()):.4%} → {float((u1 == 0).mean()):.4%}；"
          f"≤1ulp {float((u0 <= 1).mean()):.4%} → {float((u1 <= 1).mean()):.4%}；"
          f">1ulp 元素 {int((u0 > 1).sum())} → {int((u1 > 1).sum())}")
    bad = np.argwhere(u0 > 1)
    for b in bad[:4]:
        i, j, l = (int(x) for x in b)
        print(f"     out[{i}][{j}][{l}]：设备 {int(out[i, j, l]):#06x}"
              f" | 参考(翻转前) {int(eb0[i, j, l]):#06x} ({int(u0[i, j, l])} ulp)"
              f" | 参考(翻转后) {int(eb1[i, j, l]):#06x}"
              f" ⇒ {'逐位相同' if int(eb1[i, j, l]) == int(out[i, j, l]) else '仍不同'}")
    ok = (int((u1 > 1).sum()) == 0) and all(int(eb1[tuple(int(x) for x in b)]) == int(out[tuple(int(x) for x in b)]) for b in bad)
    print(f"  归因结论：{'成立（单一 P̃ bf16 翻转解释全部 >1ulp 元素）' if ok else '不成立'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
