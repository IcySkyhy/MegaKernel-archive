#!/usr/bin/env python3
"""probe_ub2l1/check_ref.py —— M121 的独立 host 侧判据（不依赖设备侧打印的任何结论）

输入：`probe_ub2l1 <capi-mmad|basic-mmad> <kill> <dumpdir>` 落盘的三个文件
      a.bin [M=32, K=64] float16（ND 行主序）
      b.bin [K=64, N=48] float16（ND 行主序）
      c.bin [M=32, N=48] float32（ND 行主序，设备 Fixpipe/copy_l0c2gm 的 ND 输出）
以及可选的 rt_in.bin / rt_out.bin（往返档的逐字节对照）。

判据（每条独立，rc 汇总）：
  J1 三个 dump 的字节数与本探针的编译期几何一致（32*64*2 / 64*48*2 / 32*48*4）；
  J2 c 全部有限（无 NaN/Inf）；
  J3 逐元素 |c - ref_fp64| 的分布给出：max / mean / p50 / p90 / p99、与 fp32 舍入参考的
     "逐位相等"元素数（ref 用 fp64 精确内积后按 fp32 舍入，理想情况下设备输出应落在
     最后一步舍入的 ±1ulp 邻域内）；J3 的通过线是 max_abs < 1e-3（与设备侧同一口径，
     但这里是**独立实现**：不同语言、不同求和顺序，任何一方的 bug 都能被对方暴露）；
  J4 若存在 rt_in.bin/rt_out.bin，则逐字节全等（往返档的"数据原样落到 L1 再读回"）。

用法：
  python3 check_ref.py <dumpsdir>
退出码：0 = 全部通过；1 = 有判据不成立；2 = 输入缺失 / 没得比。
"""
import os
import sys

import numpy as np

M, K, N = 32, 64, 48
AB = M * K * 2
BB = K * N * 2
CB = M * N * 4
RT = 8192


def err_dist(e):
    return {k: float(np.percentile(e, p)) for k, p in
            (("max", 100.0), ("p99", 99.0), ("p90", 90.0), ("p50", 50.0), ("mean", 0.0))}


def check_rt(path, problems):
    ri = open(path("rt_in.bin"), "rb").read()
    ro = open(path("rt_out.bin"), "rb").read()
    if len(ri) != RT or len(ro) != RT:
        problems.append("[J4] 往返 dump 尺寸不符：in=%d out=%d 期望 %d" % (len(ri), len(ro), RT))
    else:
        nd = sum(1 for x, y in zip(ri, ro) if x != y)
        print("[J4] 往返逐字节：same=%d diff=%d" % (RT - nd, nd))
        if nd:
            problems.append("[J4] 往返有 %d 字节不同" % nd)
    if problems:
        print("\n[FAIL] 不成立的判据：")
        for p in problems:
            print("   " + p)
        return 1
    print("\n[PASS] 全部判据成立（仅往返档）")
    return 0


def main(dumps):
    path = lambda n: os.path.join(dumps, n)  # noqa: E731
    problems = []
    have = {n: os.path.exists(path(n)) for n in ("a.bin", "b.bin", "c.bin")}
    has_rt = os.path.exists(path("rt_in.bin")) and os.path.exists(path("rt_out.bin"))
    if not all(have.values()):
        if has_rt:
            return check_rt(path, problems)
        print("[J0] 缺 dump：%s，也没有往返 dump ⇒ 没得比" % {k: v for k, v in have.items() if not v})
        return 2

    a_raw = open(path("a.bin"), "rb").read()
    b_raw = open(path("b.bin"), "rb").read()
    c_raw = open(path("c.bin"), "rb").read()
    ok = True
    if (len(a_raw), len(b_raw), len(c_raw)) != (AB, BB, CB):
        problems.append("[J1] 尺寸不符：a=%d/%d b=%d/%d c=%d/%d" %
                        (len(a_raw), AB, len(b_raw), BB, len(c_raw), CB))
        ok = False
    else:
        print("[J1] OK  a=%d B  b=%d B  c=%d B" % (AB, BB, CB))

    a = np.frombuffer(a_raw, dtype=np.float16).reshape(M, K)
    b = np.frombuffer(b_raw, dtype=np.float16).reshape(K, N)
    c = np.frombuffer(c_raw, dtype=np.float32).reshape(M, N)

    if not np.all(np.isfinite(c)):
        problems.append("[J2] c 含非有限值：%d 个" % int((~np.isfinite(c)).sum()))
        ok = False
    else:
        print("[J2] OK  c 全部有限（max=%.6g min=%.6g）" % (c.max(), c.min()))

    # ref：fp64 精确内积（操作数先转 fp64，产品与和都精确到远超 fp32）
    ref64 = a.astype(np.float64) @ b.astype(np.float64)
    err = np.abs(c.astype(np.float64) - ref64)
    d = err_dist(err)
    eq_fp32 = int(np.sum(c == ref64.astype(np.float32)))
    denom = np.abs(ref64)
    rel = np.where(denom > 1e-3, err / np.maximum(denom, 1e-300), 0.0)
    print("[J3] 逐元素 |c-ref_fp64|： max=%.6e mean=%.6e p50=%.6e p90=%.6e p99=%.6e" %
          (d["max"], d["mean"], d["p50"], d["p90"], d["p99"]))
    print("[J3] 逐元素相对误差（仅 |ref|>1e-3 的元素参与）： max=%.6e p50=%.6e p90=%.6e" %
          (float(rel.max()), float(np.percentile(rel, 50)), float(np.percentile(rel, 90))))
    print("[J3] 与 fp64->fp32 舍入参考逐位相等元素：%d/%d" % (eq_fp32, M * N))
    print("[J3] 绝对误差分桶： " + "  ".join(
        "[%.0e,%.0e)=%d" % (lo, hi, int(((err >= lo) & (err < hi)).sum()))
        for lo, hi in ((0.0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e30))))
    if d["max"] >= 1e-3:
        problems.append("[J3] max_abs=%.6e >= 1e-3" % d["max"])
        ok = False
    else:
        print("[J3] OK  max_abs < 1e-3")

    if os.path.exists(path("rt_in.bin")) and os.path.exists(path("rt_out.bin")):
        ri = open(path("rt_in.bin"), "rb").read()
        ro = open(path("rt_out.bin"), "rb").read()
        if len(ri) != RT or len(ro) != RT:
            problems.append("[J4] 往返 dump 尺寸不符：in=%d out=%d 期望 %d" % (len(ri), len(ro), RT))
            ok = False
        else:
            nd = sum(1 for x, y in zip(ri, ro) if x != y)
            print("[J4] 往返逐字节：same=%d diff=%d" % (RT - nd, nd))
            if nd:
                problems.append("[J4] 往返有 %d 字节不同" % nd)
                ok = False
    else:
        print("[J4] 跳过（没有往返 dump）")

    if problems:
        print("\n[FAIL] 不成立的判据：")
        for p in problems:
            print("   " + p)
        return 1
    print("\n[PASS] 全部判据成立")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
