# SPDX-License-Identifier: Apache-2.0
"""M151（**M162 订正版**）：相位 A 在 m>1 的**非有限（NaN/Inf）分布**核对 —— 独立于 m15 实现。

订正缘由（M160 survey / M165 + M162）：本脚本原先自带一份**旧式**（naive）参考
`eg=exp(ĝ); ig=exp(−ĝ); Γ=eg·ig`，并据此把设备 NaN 归因为"fp32 与 fp64 的 exp 溢出范围之差（精度产物）"。
**这个归因已被推翻**：真因是**数值写法**——chunk 内 `|ĝ|>88.72` 时 `0·inf=NaN`，而正确因子
`exp(ĝ_i−ĝ_j)≤1` 恒有限（官方 FLA 用指数差：`chunk_o.py:119-120`、`chunk_delta_h.py:216-221`）。
M162 已把设备侧 `m15_layer_loop/m15_gdn_prefill.h` 改成指数差；本脚本随之把**默认参考改成指数差**，
并把旧式保留为**可切换对照**（`--naive-ref`）。

  · 默认参考 = **指数差**（fp64 与 fp32 两口径）：`exp(ĝ_i−ĝ_j)`、状态 `S = egL·S + kᵀ(d⊙exp(ĝL−ĝ))`。
    它自身**非有限计数应为 0**（自洽要求）。
  · `--naive-ref` = 旧式 `eg·ig`（保留为对照；会溢出）。
  · 无论用哪种，都额外打印旧式 fp32 的 NaN 数作为对照（证明"旧式溢出、指数差不溢出"）。

判据（可传播失败）：
  · 参考自身非有限计数必须为 0；
  · T=1：设备必须零 NaN/Inf；
  · T>1：`XOR(NaN(dev), NaN(参考 fp32)) ≤ tot/1000`（设备 NaN 掩码须与所选参考几乎重合）。
阈值 `tot/1000` 是 **smoke bound**，不是误差模型紧界。
退出码：0 = 判据全过；1 = 有判据不符；2 = 没有可判的 dump（空/缺失目录）。

用法：`<python> check_phaseA_nan.py <dumpdir> [<dumpdir> ...] [--naive-ref]`
（dumpdir 内有 `m23_<tag>_{q,k,v,g,beta,h0,o,ht,meta}`）。
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


def ref_head_dt(q, k, v, g, beta, h0, scale, T, dt, naive=False):
    """m23 `ref_head` 的 dtype 参数化版（逐句同式；fp64 与 fp32 两口径）。

    `naive=False`（默认）= **指数差**：`Γs/Γi = exp(ĝ_i−ĝ_j)`、`S = egL·S + kᵀ(d⊙exp(ĝL−ĝ))` ⇒ 恒有限。
    `naive=True` = 旧式 `eg=exp(ĝ)`、`ig=exp(−ĝ)`、`Γ=eg·ig` ⇒ chunk 内 `|ĝ|>88.72` 时 `0·inf=NaN`。
    """
    S = h0.astype(dt).copy()
    out = np.zeros((T, DV), dtype=dt)
    nc = (T + BT - 1) // BT
    ii = np.arange(BT)[:, None]; jj = np.arange(BT)[None, :]
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
        eg = np.exp(gcum)
        egL = eg[cv - 1]
        with np.errstate(over="ignore", invalid="ignore"):
            if naive:
                ig = np.exp(-gcum)
                Gs = np.where(ii > jj, eg[:, None] * ig[None, :], 0.0)
                Gi = np.where(ii >= jj, eg[:, None] * ig[None, :], 0.0)
            else:
                dgc = np.minimum(gcum[:, None] - gcum[None, :], 0.0)   # 只用 i>=j；i<j 钳 0 再丢弃
                Gs = np.where(ii > jj, np.exp(dgc), 0.0)
                Gi = np.where(ii >= jj, np.exp(dgc), 0.0)
            kk = kc @ kc.T
            A = np.tril(bc[:, None] * Gs * kk, -1)
            M = np.eye(BT, dtype=dt) + A
            # naive 下 `M` 可能含 inf/nan ⇒ solve 报奇异；该 chunk 结果按 nan 记（这正是 0·inf 的机制）。
            try:
                u = np.linalg.solve(M, (bc[:, None] * vc).astype(dt))
                w = np.linalg.solve(M, (bc[:, None] * eg[:, None] * kc).astype(dt))
            except np.linalg.LinAlgError:
                u = np.full((BT, DV), np.nan, dtype=dt)
                w = np.full((BT, DK), np.nan, dtype=dt)
            d = u - w @ S
            qs = scale * qc
            o_chunk = (qs @ S) * eg[:, None] + (Gi * (qs @ kc.T)) @ d
            out[sl] = o_chunk[:cv]
            if naive:
                S = egL * (S + kc.T @ (d * ig[:, None]))
            else:
                S = egL * S + kc.T @ (d * np.exp(gcum[cv - 1] - gcum)[:, None])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=["."])
    ap.add_argument("--naive-ref", action="store_true",
                    help="用旧式 eg·ig 作参考（对照；默认用指数差）")
    args = ap.parse_args()
    dirs = args.dirs or ["."]
    naive = args.naive_ref
    m23 = _load_m23()
    rc = 0
    n_cases = 0
    for d in dirs:
        for meta in sorted(glob.glob(os.path.join(d, "m23_*_meta.txt"))):
            tag = os.path.basename(meta)[len("m23_"):-len("_meta.txt")]
            if "_mut" in tag:
                continue
            n_cases += 1
            mt = m23.read_meta(meta)
            H = int(mt["H"]); T = int(mt["T"]); nk = int(mt["NK"]); scale = float(mt["scale"])
            Tp = T + 64
            pref = os.path.join(d, "m23_%s_" % tag)
            q = m23.load_bin(pref + "q.bin").reshape(nk, Tp, DK)
            k = m23.load_bin(pref + "k.bin").reshape(nk, Tp, DK)
            v = m23.load_bin(pref + "v.bin").reshape(H, T, DV)
            g = m23.load_bin(pref + "g.bin").reshape(H, T)
            beta = m23.load_bin(pref + "beta.bin").reshape(H, T)
            h0 = m23.load_bin(pref + "h0.bin").reshape(H, DK, DV)
            o_dev = m23.load_bin(pref + "o.bin").reshape(H, T, DV)
            nan_dev = np.isnan(o_dev)
            nf_dev = int((~np.isfinite(o_dev)).sum())
            nan_ref64 = np.zeros((H, T, DV), dtype=bool)
            nan_ref32 = np.zeros((H, T, DV), dtype=bool)
            nan_old32 = np.zeros((H, T, DV), dtype=bool)
            for hv in range(H):
                hk = hv // 3
                o64 = ref_head_dt(q[hk], k[hk], v[hv], g[hv], beta[hv], h0[hv], scale, T, np.float64, naive=naive)
                o32 = ref_head_dt(q[hk], k[hk], v[hv], g[hv], beta[hv], h0[hv], scale, T, np.float32, naive=naive)
                nan_ref64[hv] = np.isnan(o64)
                nan_ref32[hv] = np.isnan(o32)
                if not naive:
                    o_old = ref_head_dt(q[hk], k[hk], v[hv], g[hv], beta[hv], h0[hv], scale, T, np.float32,
                                        naive=True)
                    nan_old32[hv] = np.isnan(o_old)
            tot = o_dev.size
            xor64 = int((nan_dev ^ nan_ref64).sum())
            xor32 = int((nan_dev ^ nan_ref32).sum())
            ref_name = "naive(eg·ig)" if naive else "指数差"
            print(f"[{tag}] T={T} o 元素 {tot}  参考={ref_name}")
            print(f"  dev 非有限={nf_dev}（NaN={int(nan_dev.sum())}）")
            print(f"  NaN(ref_fp64)={int(nan_ref64.sum())}  NaN(ref_fp32)={int(nan_ref32.sum())}  "
                  f"（参考自洽要求 0）")
            print(f"  XOR(dev,fp64)={xor64}  XOR(dev,fp32)={xor32}")
            if not naive:
                print(f"  对照：旧式(eg·ig) fp32 的 NaN = {int(nan_old32.sum())}"
                      f"（m>1 时应为大数 ⇒ 旧式溢出；默认参考为指数差、NaN 应 0）")
            # 判据：参考自洽 + T=1 零非有限 + T>1 设备 NaN 掩码与所选参考几乎重合。
            if int(nan_ref64.sum()) != 0:
                print("  ⇒ FAIL（参考自身含 NaN ⇒ 参考口径有问题）"); rc = 1
            elif T == 1:
                if nf_dev != 0:
                    print("  ⇒ FAIL（T=1 不允许非有限）"); rc = 1
                else:
                    print("  ⇒ PASS（T=1 无非有限）")
            else:
                if xor32 > tot // 1000:
                    print(f"  ⇒ FAIL（设备 NaN 掩码与参考不符：XOR={xor32} > {tot // 1000}）"); rc = 1
                else:
                    print(f"  ⇒ PASS（设备 NaN 掩码 ≈ 参考；XOR={xor32} ≤ {tot // 1000}）")
    if n_cases == 0:
        print("[M151 NaN] 没有可判的 dump（未找到 m23_*_meta.txt）—— 路径给错了？")
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
