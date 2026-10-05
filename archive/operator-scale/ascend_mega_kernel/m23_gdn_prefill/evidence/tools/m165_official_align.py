#!/usr/bin/env python3
"""M165 —— 把 m23 的 chunk 参考与**官方 m21 torch 参考**的 `gdn.core_out` 对齐（只读 m21 dump）。

背景：M160 survey 认定 m23 参考的 Γ 写法（`eg=exp(ĝ)`、`ig=exp(−ĝ)` 分开物化再相乘）在
chunk 内 `|ĝ|` 超 `exp` 上溢阈值时自造 `0·inf=NaN`；官方 FLA 用**指数差**。本脚本取官方
`m21_layer_ref/run_reference.py --layer 0 --input embed` 的真实权重 dump（`.npy`），用
**同一份** `gdn.q/k/v/g/beta` 面分别跑 m23 的 旧式 / 指数差 参考，与官方 `gdn.core_out` 比。

三层比对（把「chunk 因子分解是否正确」与「输入 bf16 舍入地板」分开）：
  * `seq` = 官方递推**逐 token** 形式（m21 `ref/gdn.py::_recurrence` 的逐句等价）在**同样的**输入上重算；
    `|指数差 chunk − seq|` 应到 fp64 舍入量级（证明 chunk 分解 == 官方递推）。
  * `|seq − core_out|` = 输入地板：官方 `core_out` 用的是 **fp32** 的 `qn/kn`，而 dump 出来的
    `gdn.q/gdn.k` 是 **bf16**（`gdn.v` 本就 bf16）⇒ 这个差主要来自 q/k 的 bf16 输入舍入。
  * `|指数差 chunk − core_out|` = 上面两条之和（任务书要求的那条）。

用法：
  /workspace/venvs/baseline/bin/python3 m165_official_align.py <refdir> [...]
判据（可传播失败；rc=1）：
  ① 官方 `core_out` 出现 NaN ⇒ FAIL；② **指数差**参考出现 NaN ⇒ FAIL；
  ③ `|指数差 chunk − seq|` > `SEQ_REL_BOUND·max(1,|core|)` ⇒ FAIL（chunk 分解错）；
  ④ `|指数差 chunk − core_out|` > `REL_BOUND·max|core_out|` ⇒ FAIL（超出 bf16 输入舍入量级）。
`REL_BOUND = 2^-7` = **2 个 bf16 ulp**（bf16 相对 ulp = 2^-8）；实测该比在 2.5e-3–4.6e-3，
即 0.65–1.2 个 bf16 ulp。rc=2 = 没有可比的 dump。
"""
from __future__ import annotations

import os
import sys

import numpy as np

BT, DK, DV = 64, 128, 128
SEQ_REL_BOUND = 1e-9          # chunk-vs-seq：只容忍 fp64 舍入
REL_BOUND = 2.0 ** -7         # chunk-vs-core：2 个 bf16 ulp（相对 max|core|）

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import check_ref as CR  # noqa: E402


def load(d, n):
    a = np.load(os.path.join(d, n + ".npy"))
    if a.dtype == np.uint16:                       # bf16 存为 uint16 位型
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a


def ulp_bf16(x):
    """bf16 的 1 ulp（按 |x| 的二进制指数；bf16 尾数 8 位）。"""
    ax = np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.frexp(ax)[1]
    return np.ldexp(np.float64(1.0), e - 8)


def seq_ref(q, k, v, g, beta, T):
    """官方递推逐 token 形式（m21 ref/gdn.py::_recurrence）：h[V,K] *= e^g → delta → outer → o=h@q。"""
    h = np.zeros((DV, DK))                          # [V,K]
    o = np.zeros((T, DV))
    for t in range(T):
        h *= np.exp(g[t])                           # decay ← exp(g)（g≤0 ⇒ ≤1，官方稳定式）
        delta = (v[t] - h @ k[t]) * beta[t]
        h += np.outer(delta, k[t])
        o[t] = h @ q[t]
    return o


def maxabs_fin(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a[m] - b[m]).max()) if m.any() else float("nan")


def align(refdir):
    q = load(refdir, "gdn.q")       # [T,16,128] bf16
    k = load(refdir, "gdn.k")       # [T,16,128] bf16
    v = load(refdir, "gdn.v")       # [T,48,128] bf16
    g = load(refdir, "gdn.g")       # [T,48] fp32
    beta = load(refdir, "gdn.beta")  # [T,48] fp32
    core = load(refdir, "gdn.core_out")  # [T,48,128] bf16（官方递推输出）
    T, H, _ = v.shape
    coreH = np.ascontiguousarray(core.transpose(1, 0, 2))   # -> [H,T,DV]
    cmax = float(np.abs(coreH).max())
    print("[ALIGN] %s | T=%d H=%d max|core|=%.4g | 官方 core_out NaN = %d（shape %s）"
          % (os.path.basename(refdir), T, H, cmax, int(np.isnan(coreH).sum()), core.shape))
    rc = 1 if np.isnan(coreH).sum() else 0

    for label, stable in (("旧式", False), ("指数差", True)):
        for dt in (np.float64, np.float32):
            # 官方递推在同输入、同 dtype 上的重算（用于把「chunk 分解误差」与「输入 bf16 地板」分开）
            seq = np.zeros((H, T, DV), dtype=dt)
            for hv in range(H):
                hk = hv // 3
                seq[hv] = seq_ref(q[:, hk, :].astype(dt), k[:, hk, :].astype(dt),
                                  v[:, hv, :].astype(dt), g[:, hv].astype(dt),
                                  beta[:, hv].astype(dt), T)
            o = np.zeros((H, T, DV), dtype=dt)
            for hv in range(H):
                hk = hv // 3
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    o[hv] = CR.ref_head(q[:, hk, :].astype(dt), k[:, hk, :].astype(dt),
                                        v[:, hv, :].astype(dt), g[:, hv].astype(dt),
                                        beta[:, hv].astype(dt), np.zeros((DK, DV), dtype=dt),
                                        dt(1.0), T, stable=stable, dtype=dt)[0]
            nan = int(np.isnan(o).sum())
            d_seq = maxabs_fin(o, seq)
            d_core = maxabs_fin(o, coreH)
            rel = d_core / cmax if cmax else 0.0
            sig = np.isfinite(o) & np.isfinite(coreH) & (np.abs(coreH) >= (2.0 ** -8) * cmax)
            ud = (np.abs(o[sig] - coreH[sig]) / ulp_bf16(coreH[sig])) if sig.any() else np.array([0.0])
            print("[ALIGN] %-14s %-8s %-7s | NaN=%d | |chunk−seq|=%.3e | |chunk−官方core|=%.3e "
                  "(=%.2f bf16ulp @max, 逐元素 max ulp=%.2f) | |seq−core|=%.3e"
                  % (os.path.basename(refdir), label, np.dtype(dt).name, nan, d_seq, d_core,
                     rel / (2.0 ** -8), float(np.nanmax(ud)), maxabs_fin(seq, coreH)))
            if stable and nan:
                rc = 1
            # ③ 只在 fp64 行判：chunk 分解 == 官方递推（fp32 行的 chunk-vs-seq 差是 fp32 舍入，
            #    不是分解误差 —— 只作报告）
            if stable and dt is np.float64 and d_seq > SEQ_REL_BOUND * max(1.0, cmax):
                print("[ALIGN] ✗ 指数差 chunk 与官方递推 seq 差 %.3e 超 fp64 舍入" % d_seq)
                rc = 1
            if stable and rel > REL_BOUND:
                print("[ALIGN] ✗ 指数差参考与官方 core_out 相对差 %.3e 超 %.1e（%.2f bf16 ulp）"
                      % (rel, REL_BOUND, rel / (2.0 ** -8)))
                rc = 1
    return rc


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        return 2
    rc = 0
    for d in dirs:
        if not os.path.exists(os.path.join(d, "manifest.json")):
            print("[ALIGN] SKIP %s：无 manifest.json" % d)
            rc = max(rc, 2)
            continue
        rc = max(rc, align(d))
    print("[ALIGN] rc=%d（0=官方与指数差参考均有限、chunk==官方递推、且差在 bf16 舍入量级；"
          "1=有判据不符；2=无输入可比）" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
