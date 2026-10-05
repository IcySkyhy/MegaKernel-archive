#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_cube_fp32/check_ref.py —— 对 prec_*.bin 的独立复核（numpy fp64，与核内 host 参考不同实现）

用法（在放有 prec_a.bin / prec_b.bin / prec_c.bin 的目录下）：
    /usr/local/python3.12.13/bin/python3 check_ref.py [dump_dir]

判据（**全部 7 条都进 bad[] ⇒ 都受退出码保护**；每条给读数 + 阈值，不给"精度够"这种无读数断言）：
  d0 ones     dev 逐元素 == K                              （恒真自检：布局/搬运/Mmad/Fixpipe）
  d1 mant13   A=1+2^-12：dev == 256.0625 精确 ⇒ 操作数保留 13 位有效位 ⇒ 未降到 hf32/bf16
  d2 mant10   A=1+2^-9 ：dev == 256.5    精确 ⇒ 操作数保留 10 位有效位 ⇒ 未降到 bf16
  d3 absorb   B[0]=2^24：dev == 2^24            ⇒ 累加器在 2^24 量级上吞掉 +1 ⇒ fp32 宽度 / cube_k=1 签名
  d4 rand     dev 最贴合的一档必须是 fp64/fp32 操作数，且 maxRelErr < 1e-4
  d5 cancel   +1/-1 交替：dev 逐元素 == 0
  d6 mant23s  单项 A=1+2^-23：dev 逐元素 == 0x3F800001 ⇒ 操作数保留 23 位尾数 = 全 fp32

退出码：0 = 上列 7 条全部成立；1 = 有任一条不成立（失败的条名会打印在 FAIL 行里）。
        脚本**不**只把自检项计入 rc —— 载重判别项（d1/d2/d3/d4/d6）同样 gate（这是 M106 r1 的 P2-2 修复）。
"""
import os
import sys

import numpy as np

M, N, K, NDIST = 16, 16, 256, 7

DIST_NAME = ["d0 ones", "d1 mant13", "d2 mant10", "d3 absorb", "d4 rand", "d5 cancel", "d6 mant23s"]


def round_mant(x, mant):
    """把 fp32 的尾数按 RNE 舍到 mant 位（保留 1s + 8e + mant 位尾数）。"""
    x32 = np.asarray(x, dtype=np.float32)
    drop = 23 - mant
    if drop <= 0:
        return x32.copy()
    b = x32.view(np.uint32)
    mask = np.uint32((1 << drop) - 1)
    half = np.uint32(1 << (drop - 1))
    keep = (b & np.uint32(~mask)).astype(np.uint32)
    rem = (b & mask).astype(np.uint32)
    up = (rem > half) | ((rem == half) & (((keep >> np.uint32(drop)) & np.uint32(1)) == 1))
    out = keep.copy()
    out[up] = (keep[up] + np.uint32(1 << drop)).astype(np.uint32)
    return out.view(np.float32).astype(np.float32)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    A = np.fromfile(os.path.join(d, "prec_a.bin"), dtype="<f4").reshape(NDIST, M, K)
    B = np.fromfile(os.path.join(d, "prec_b.bin"), dtype="<f4").reshape(NDIST, N, K)
    C = np.fromfile(os.path.join(d, "prec_c.bin"), dtype="<f4").reshape(NDIST, M, N)
    print("read A%s B%s C%s from %s" % (A.shape, B.shape, C.shape, d))

    Ad = A.astype(np.float64)
    Bd = B.astype(np.float64)
    ref = np.einsum("dmk,dnk->dmn", Ad, Bd)          # fp64 精确和

    print("\n%-10s %14s %14s %12s %12s" % ("dist", "dev C[0][0]", "ref_fp64", "maxAbsErr", "maxRelErr"))
    for i in range(NDIST):
        dev = C[i].astype(np.float64)
        r = ref[i]
        ae = np.abs(dev - r)
        denom = np.where(np.abs(r) > 0, np.abs(r), 1.0)
        re = ae / denom
        print("%-10s %14.9g %14.9g %12.3e %12.3e"
              % (DIST_NAME[i], C[i][0, 0], r[0, 0], ae.max(), re.max()))

    # 逐条判据：**每一条都进 bad[] ⇒ 影响退出码**（载重判别项 d1/d2/d3/d4/d6 与自检项 d0/d5 同等受 rc 保护）
    bad = []

    print("\n---- 逐条判据（读数 -> 结论）；任一条不成立 ⇒ rc=1 ----")
    ok0 = bool(np.all(C[0] == np.float32(K)))
    print("d0 ones   : dev C[0] 全部 == %d ? %s  (max|dev-%d| = %.3g)"
          % (K, "YES" if ok0 else "NO", K, np.abs(C[0].astype(np.float64) - K).max()))
    if not ok0:
        bad.append("d0（自检：布局/搬运/Mmad/Fixpipe）")

    d1 = C[1][0, 0].astype(np.float64)
    d2 = C[2][0, 0].astype(np.float64)
    ref1 = float(ref[1][0, 0])
    ref2 = float(ref[2][0, 0])
    ok1 = abs(d1 - ref1) == 0.0
    print("d1 mant13 : A=1+2^-12 => ref=%.9g, dev=%.9g, |diff|=%.3g" % (ref1, d1, abs(d1 - ref1)))
    print("            => 操作数%s 13 位有效位（%s）"
          % ("保留" if ok1 else "未保留",
             "未降到 hf32(10 位)/bf16(8 位)" if ok1 else "已被降到 <=12 位"))
    if not ok1:
        bad.append("d1（操作数 ≥13 位有效位）")

    ok2 = abs(d2 - ref2) == 0.0
    print("d2 mant10 : A=1+2^-9  => ref=%.9g, dev=%.9g, |diff|=%.3g%s"
          % (ref2, d2, abs(d2 - ref2), "" if ok2 else "   <== FAIL"))
    if not ok2:
        bad.append("d2（操作数 ≥10 位有效位）")

    d3 = C[3][0, 0].astype(np.float64)
    ok3 = (d3 == float(2 ** 24))
    print("d3 absorb : ref=%.9g dev=%.9g ; 2^24=%.9g ; 2^24+255 的 fp32 舍入=%.9g"
          % (float(ref[3][0, 0]), d3, float(2 ** 24), float(np.float32(2 ** 24 + 255))))
    if ok3:
        print("            => 2^24 量级上 +1 被吞掉 ⇒ 部分和保持 fp32 宽度（非宽累加器，也非“按 8 分块精确和”）")
    else:
        print("            => dev != 2^24 ⇒ 累加未在 fp32 宽度上丢 +1（>= 按 8 分块精确和 / 宽累加器 / 操作数被舍）")
        bad.append("d3（默认模式累加器 = fp32 宽度 / cube_k=1 签名）")

    # d4：dev 落在哪一档操作数参考上
    Ab, Bb = round_mant(A[4], 8), round_mant(B[4], 8)
    Ah, Bh = round_mant(A[4], 10), round_mant(B[4], 10)
    r_fp64 = ref[4]
    r_bf16 = np.einsum("mk,nk->mn", Ab.astype(np.float64), Bb.astype(np.float64))
    r_hf32 = np.einsum("mk,nk->mn", Ah.astype(np.float64), Bh.astype(np.float64))
    dev4 = C[4].astype(np.float64)
    e64 = np.abs(dev4 - r_fp64).max()
    ebf = np.abs(dev4 - r_bf16).max()
    ehf = np.abs(dev4 - r_hf32).max()
    rel4 = (np.abs(dev4 - r_fp64) / np.where(np.abs(r_fp64) > 0, np.abs(r_fp64), 1.0)).max()
    print("\nd4 rand   : dev vs fp64  maxAbsErr = %.6e" % e64)
    print("            dev vs bf16  maxAbsErr = %.6e" % ebf)
    print("            dev vs hf32  maxAbsErr = %.6e" % ehf)
    best = min([("fp64/fp32 操作数", e64), ("hf32/tf32 操作数(10 位尾数)", ehf), ("bf16 操作数", ebf)],
               key=lambda t: t[1])
    print("            => dev 最贴合【%s】（残差 %.3e）" % (best[0], best[1]))
    ok4 = (best[0] == "fp64/fp32 操作数") and (rel4 < 1e-4)
    if not ok4:
        print("            => FAIL：最贴合档不是 fp64/fp32 操作数，或 maxRelErr=%.3e >= 1e-4" % rel4)
        bad.append("d4（操作数精度档 = fp32）")

    ok5 = bool(np.all(C[5] == 0.0))
    print("\nd5 cancel : dev C[0] 全 == 0 ? %s (max|dev| = %.3g)"
          % ("YES" if ok5 else "NO", np.abs(C[5].astype(np.float64)).max()))
    if not ok5:
        bad.append("d5（完全相消 = 0）")

    # d6：单项 1+2^-23，无累加 ⇒ 直接读操作数保留了多少位尾数
    want23 = np.float32(1.0 + 2.0 ** -23)
    d6 = C[6].astype(np.float64)
    ok6 = bool(np.all(C[6] == want23))
    print("d6 mant23s: 单项 C = 1+2^-23 = %.9g (0x%08x) ; dev = %.9g (0x%08x) ; 全元素相等 ? %s"
          % (float(want23), np.float32(want23).view(np.uint32), d6[0, 0],
             C[6][0, 0].view(np.uint32), "YES" if ok6 else "NO"))
    print("            => 操作数尾数保留 %s（23 位 = 全 fp32；10 位 = hf32/tf32；8 位 = bf16）"
          % ("23 位" if ok6 else "<23 位"))
    if not ok6:
        bad.append("d6（操作数保留 23 位尾数 = 全 fp32）")

    if bad:
        print("\n[check_ref] FAIL: 下列判据不成立 -> %s" % "; ".join(bad))
        return 1
    print("\n[check_ref] PASS: d0/d5 自检 + d1/d2/d3/d4/d6 载重判别项**全部**成立（rc=0 等价于这条判据清单）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
