#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""check_stab_nan.py —— M163 设备侧数值稳定化的 **NaN-aware** 判据（独立于 m15 实现）

修的是哪一处：`m15_layer_loop/m15_gdn_prefill.h` 的 chunk 扫描把 `eg=exp(ĝ)` 与 `ig=exp(−ĝ)`
分开物化再相乘，chunk 内 `|ĝ|>88.7` 时 `0·inf=NaN`；改成指数差 `exp(ĝ_i−ĝ_j)` / `exp(ĝL−ĝ[t])`。

参考用的是哪一份（写清）：
  * **稳定化参考 = 本文件的 `ref_head_stable`**（fp64，逐句同 m23 `ref_head` 的公式，但 Γ/KT′ 用指数差）。
    它是**权威地面真值**：`|ĝ|` 再大也恒有限。
  * **naive 参考 = 本文件的 `ref_head_naive`**（参数化 fp64/fp32，逐句复刻修前的 `eg·ig` 写法）。
    它只用来**复现溢出机理**与给出"修前应有的 NaN 数"，**不是判据**。
  * **本分支 base `0cd7fe5` 的** m23 `check_ref.py` 判据（`m23_gdn_prefill/check_ref.py:147-149`）仍是
    naive 写法且对 NaN 静默（`|dev−ref|>lim` 在 NaN 处为 False）⇒ 本文件与它的差异就在这一层：
    **NaN 单独计数**。（注：当前 main 已含 M157 的 NaN-aware 判据与 M165 的指数差参考；本句只对
    base `0cd7fe5` 成立——本 mission 的 `m23judge_*.log` 用的就是该 base 版。）

判据（可传播失败，退出码 0 过 / 1 不过 / 2 没有可判 dump）：
  1. 设备 o 与 ht 的**非有限计数必须为 0**（这是本次修复的目标量）；
  2. 稳定化参考本身非有限计数为 0（参考自洽）；
  3. 设备 o[0,cv) 行与 ht 全量对 `ref_head_stable` 逐元素 `|Δ| ≤ rtol·max(1,|ref|)`（口径同 m23）。

用法：`<python> check_stab_nan.py <dumpdir> [<dumpdir> ...] [--rtol 2e-3]`
 dumpdir 内有 `m23_<tag>_{q,k,v,g,beta,h0,o,ht,meta}`。`tag` 不含 `mut`（与 m23 的 mutant 档区分）。
"""
import argparse
import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
BT, DK, DV = 64, 128, 128


def _load_m23():
    p = ROOT / "m23_gdn_prefill" / "check_ref.py"
    spec = importlib.util.spec_from_file_location("m23_check_ref", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_head_stable(q, k, v, g, beta, h0, scale, T, dt):
    """逐句同 m23 `ref_head`，但 Γs/Γi/KT′ 用**指数差**：`exp(ĝ_i−ĝ_j)`、`exp(ĝL−ĝ)`。"""
    S = h0.astype(dt).copy()
    out = np.zeros((T, DV), dtype=dt)
    nc = (T + BT - 1) // BT
    for c in range(nc):
        t0 = c * BT
        cv = min(BT, T - t0)
        sl = slice(t0, t0 + cv)
        kc = np.zeros((BT, DK), dtype=dt)
        qc = np.zeros((BT, DK), dtype=dt)
        vc = np.zeros((BT, DV), dtype=dt)
        gc = np.zeros(BT, dtype=dt)
        bc = np.zeros(BT, dtype=dt)
        kc[:cv] = k[sl]; qc[:cv] = q[sl]; vc[:cv] = v[sl]
        gc[:cv] = g[sl]; bc[:cv] = beta[sl]
        gcum = np.cumsum(gc)
        dgc = gcum[:, None] - gcum[None, :]
        ii = np.arange(BT)[:, None]; jj = np.arange(BT)[None, :]
        # 只用 i>=j（dgc<=0）的项；i<j 处钳到 0 再 where 丢弃，避免上三角 exp 溢出噪声。
        dgc = np.minimum(dgc, 0.0)
        Gs = np.where(ii > jj, np.exp(dgc), 0.0)
        Gi = np.where(ii >= jj, np.exp(dgc), 0.0)
        egL = np.exp(gcum[cv - 1])
        kk = kc @ kc.T
        A = np.tril(bc[:, None] * Gs * kk, -1)
        M = np.eye(BT, dtype=dt) + A
        u = np.linalg.solve(M, bc[:, None] * vc)
        w = np.linalg.solve(M, bc[:, None] * np.exp(gcum)[:, None] * kc)
        d = u - w @ S
        qs = scale * qc
        o_chunk = (qs @ S) * np.exp(gcum)[:, None] + (Gi * (qs @ kc.T)) @ d
        out[sl] = o_chunk[:cv]
        S = egL * S + kc.T @ (d * np.exp(gcum[cv - 1] - gcum)[:, None])
    return out, S


def ref_head_naive(q, k, v, g, beta, h0, scale, T, dt):
    """逐句复刻**修前**写法（eg·ig），只为复现溢出机理。"""
    S = h0.astype(dt).copy()
    out = np.zeros((T, DV), dtype=dt)
    nc = (T + BT - 1) // BT
    for c in range(nc):
        t0 = c * BT
        cv = min(BT, T - t0)
        sl = slice(t0, t0 + cv)
        kc = np.zeros((BT, DK), dtype=dt); qc = np.zeros((BT, DK), dtype=dt)
        vc = np.zeros((BT, DV), dtype=dt); gc = np.zeros(BT, dtype=dt); bc = np.zeros(BT, dtype=dt)
        kc[:cv] = k[sl]; qc[:cv] = q[sl]; vc[:cv] = v[sl]; gc[:cv] = g[sl]; bc[:cv] = beta[sl]
        gcum = np.cumsum(gc)
        with np.errstate(over="ignore", invalid="ignore"):
            eg = np.exp(gcum); ig = np.exp(-gcum); egL = eg[cv - 1]
            ii = np.arange(BT)[:, None]; jj = np.arange(BT)[None, :]
            Gs = np.where(ii > jj, eg[:, None] * ig[None, :], 0.0)
            Gi = np.where(ii >= jj, eg[:, None] * ig[None, :], 0.0)
            kk = kc @ kc.T
            A = np.tril(bc[:, None] * Gs * kk, -1)
            M = np.eye(BT, dtype=dt) + A
            try:
                u = np.linalg.solve(M, (bc[:, None] * vc).astype(dt))
                w = np.linalg.solve(M, (bc[:, None] * eg[:, None] * kc).astype(dt))
            except np.linalg.LinAlgError:
                u = np.full((BT, DV), np.nan, dtype=dt); w = np.full((BT, DK), np.nan, dtype=dt)
            d = u - w @ S
            qs = scale * qc
            o_chunk = (qs @ S) * eg[:, None] + (Gi * (qs @ kc.T)) @ d
            out[sl] = o_chunk[:cv]
            S = egL * (S + kc.T @ (d * ig[:, None]))
    return out, S


def _load_case(d, m23, tag):
    mt = m23.read_meta(os.path.join(d, "m23_%s_meta.txt" % tag))
    H = int(mt["H"]); T = int(mt["T"]); nk = int(mt["NK"]); scale = float(mt["scale"])
    Tpq = int(mt["TP_qk"]); Tpg = int(mt["TP_gb"])
    pref = os.path.join(d, "m23_%s_" % tag)
    q = m23.load_bin(pref + "q.bin").reshape(nk, Tpq, DK)
    k = m23.load_bin(pref + "k.bin").reshape(nk, Tpq, DK)
    v = m23.load_bin(pref + "v.bin").reshape(H, T, DV)
    g = m23.load_bin(pref + "g.bin").reshape(H, T)
    beta = m23.load_bin(pref + "beta.bin").reshape(H, T)
    h0 = m23.load_bin(pref + "h0.bin").reshape(H, DK, DV)
    o = m23.load_bin(pref + "o.bin").reshape(H, T, DV)
    ht = m23.load_bin(pref + "ht.bin").reshape(H, DK, DV)
    return H, T, nk, scale, q, k, v, g, beta, h0, o, ht


def judge_dir(d, m23, rtol, verbose):
    rc = 0
    n_cases = 0
    for meta in sorted(glob.glob(os.path.join(d, "m23_*_meta.txt"))):
        tag = os.path.basename(meta)[len("m23_"):-len("_meta.txt")]
        if "_mut" in tag:
            continue
        n_cases += 1
        H, T, nk, scale, q, k, v, g, beta, h0, o, ht = _load_case(d, m23, tag)
        nan_dev_o = int(np.isnan(o).sum()); nan_dev_h = int(np.isnan(ht).sum())
        nan_dev_o_inf = int((~np.isfinite(o)).sum()); nan_dev_h_inf = int((~np.isfinite(ht)).sum())
        nan64_o = np.zeros((H, T, DV), dtype=bool); nan64_h = np.zeros((H, DK, DV), dtype=bool)
        nan_nv_o = np.zeros((H, T, DV), dtype=bool)
        bad_o = 0; bad_h = 0; mx_o = 0.0; mx_h = 0.0; ref_stab_nan = 0
        for hv in range(H):
            hk = hv // 3
            o64, S64 = ref_head_stable(q[hk], k[hk], v[hv], g[hv], beta[hv], h0[hv], scale, T, np.float64)
            o_nv, _ = ref_head_naive(q[hk], k[hk], v[hv], g[hv], beta[hv], h0[hv], scale, T, np.float32)
            nan64_o[hv] = np.isnan(o64); nan64_h[hv] = np.isnan(S64)
            nan_nv_o[hv] = np.isnan(o_nv)
            ref_stab_nan += int(np.isnan(o64).sum()) + int(np.isnan(S64).sum())
            do = np.abs(o[hv] - o64)
            dh = np.abs(ht[hv] - S64)
            lo = rtol * np.maximum(1.0, np.abs(o64))
            lh = rtol * np.maximum(1.0, np.abs(S64))
            fo = np.isfinite(do); fh = np.isfinite(dh)
            bad_o += int((do[fo] > lo[fo]).sum()); bad_h += int((dh[fh] > lh[fh]).sum())
            if fo.any():
                mx_o = max(mx_o, float(do[fo].max()))
            if fh.any():
                mx_h = max(mx_h, float(dh[fh].max()))
        xor_o = int((np.isnan(o) ^ nan64_o).sum())
        xor_nv = int((np.isnan(o) ^ nan_nv_o).sum())
        tot_o = o.size; tot_h = ht.size
        print("[%s] T=%d o元素 %d ht元素 %d" % (tag, T, tot_o, tot_h))
        print("  dev  o 非有限 %d / ht 非有限 %d（其中 NaN：o=%d ht=%d）"
              % (nan_dev_o_inf, nan_dev_h_inf, nan_dev_o, nan_dev_h))
        print("  ref_stab(fp64) 非有限 %d（自洽要求 0）" % ref_stab_nan)
        print("  ref_naive(fp32) o NaN = %d；XOR(NaN(dev),NaN(naive_fp32)) = %d" % (int(nan_nv_o.sum()), xor_nv))
        print("  XOR(NaN(dev),NaN(ref_stab_fp64)) = %d" % xor_o)
        print("  有限元素 |Δ|(dev,ref_stab): o 超界 %d/%d max=%.3e | ht 超界 %d/%d max=%.3e"
              % (bad_o, tot_o, mx_o, bad_h, tot_h, mx_h))
        ok = (nan_dev_o_inf == 0) and (nan_dev_h_inf == 0) and (ref_stab_nan == 0) and \
             (bad_o == 0) and (bad_h == 0)
        print("  ⇒ %s" % ("PASS" if ok else "FAIL"))
        if not ok:
            rc = 1
    if n_cases == 0:
        print("[M163 NaN-aware] 没有可判的 dump（未找到 m23_*_meta.txt）—— 路径给错了？")
        return 2
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--rtol", type=float, default=2e-3)
    args = ap.parse_args()
    m23 = _load_m23()
    rc = 0
    for d in args.dirs:
        if not os.path.isdir(d):
            print("[M163] 目录不存在：%s" % d); rc = 2; continue
        r = judge_dir(d, m23, args.rtol, True)
        if r != 0:
            rc = r
    return rc


if __name__ == "__main__":
    sys.exit(main())
