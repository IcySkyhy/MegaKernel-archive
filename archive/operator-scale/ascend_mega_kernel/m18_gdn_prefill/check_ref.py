#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m18_gdn_prefill/check_ref.py —— GDN prefill chunk scan 的 numpy float64 逐句参考与逐段判据。

口径遵循 docs/17-verification-standard.md（§1.1 分档 / §2 写法规范 / §4 非空洞性）。

## 档位（README §4 有理由与 ε 推导表）
* 输入六张量（q/k/v/g/β/h0）：**T1 逐位**（kernel 不碰它们，必须 bit-exact，容差 0）。
* 其余全部：**T3**（≥64 深的 fp32 累加链 + VF `Reg::Exp`（官方最大精度误差 1 ulp）
  + 65 个 chunk 的状态链）⇒ 判据
      |got − exp| ≤ ε · Σ|terms| + 0.5·ulp_fp32(exp)
  其中 ε = 5e-5（各项来源与数值见 README §4.2 推导表），**逐元素检查**；
  Σ|terms| 取该元素在其公式链上所有被加/被乘项的量值之和（子项若是算出来的量，则代入它自己的 Σ|terms|）。

## 参考实现
严格按 m18_gdn_prefill.asc 文件头 / README §1 的公式「逐句」写成 numpy float64，
per (value head hv, key head hk = hv//3) 独立做一次 BT=64 的 chunk 扫描：
  chunk 内：ĝ = cumsum(g); eg = exp(ĝ); egL = eg[cv−1]
           Γs[i,j] = exp(ĝ_i−ĝ_j), j<i（严格下三角）;  A[i,j] = β_i·Γs[i,j]·(k_i·k_j)
           u = (I+A)^{-1}(β⊙v);  w = (I+A)^{-1}(β⊙eg⊙k)     ← 行前代，不物化 T
  chunk 间：d = u − w·S₀
           o = ((q·scale)·S₀)⊙_row eg + (Γi ⊙ ((q·scale)·kᵀ))·d      Γi 含对角
           S₁ = egL·(S₀ + kᵀ·(d⊙ig))            （旧式：ig = exp(−ĝ)）
           S₁ = egL·S₀ + kᵀ·(d⊙exp(ĝ_last−ĝ_j)) （默认：指数差，恒有限）

**数值稳定化（M166）**：跨 chunk 衰减**默认**用**指数差** `exp(ĝ_last−ĝ_j)`（`≤1`，有限输入下恒有限），
与官方 FLA（`vllm/.../flash_linear_attention/ops/chunk_o.py:119-120`、`chunk_delta_h.py:216-221`）
及 m23 的 M165 修法**同形**（`m23_gdn_prefill/check_ref.py` commit `600b05e` 的指数差分支）。
旧式「`ig=exp(−ĝ)` 物化后 `d⊙ig`」在 chunk 内 `|ĝ|` 超 `exp` 上溢阈值（fp64 709 / fp32 88.7）时
`0·inf=NaN`；`--legacy-exp` 切回该式仅作对照。Γ（`Gs`/`Gi`）本就用 `exp(ĝ_i−ĝ_j)`，无需改。
`eg=exp(ĝ)` 仍用于 `w` 的右端与 `o_part`（`ĝ≤0 ⇒ eg≤1`，本就不上溢）。

## 判定项 / 报告项 / guard（§2.1）
* 判定项（决定 PASS/FAIL）：每个比较对象的「超界元素数」（T3 逐元素）+ 「非有限不匹配数」，
  两者之和必须为 0。非有限面口径与 M157（m23 判据）/M165 一致：
  **NaN↔NaN、同号 Inf↔同号 Inf 视为相等**（不计错，但也**不**由此断言该处数值正确）；
  **仅一侧 NaN、仅一侧 Inf、NaN↔Inf、异号 Inf 一律计为不匹配**。
* 报告项（不参与判定）：max|Δ|（绝对，**只在两侧皆有限处取**）、maxRel、≤1ulp 比例、
  最差容差占用、max Σ|terms|；另单列每行 `devNaN/refNaN/仅一侧NaN/双侧NaN/devInf/refInf/仅一侧Inf/异号Inf/非有限不匹配`。
* guard（不参与判定）：非空洞性（输入非常数、状态演化、结构性零、dump 位置/越界区为 0）。

用法（在 build 目录内，dump 文件在 cwd）：
  /usr/local/python3.12.13/bin/python3 ../check_ref.py [--only CASE] [--eps 5e-5]
  --legacy-exp   参考改用旧式 `exp(ĝ)·exp(−ĝ)`（仅对照，上溢档会自造 NaN）
  --selftest     零设备自检：合成小 dump 走同一条判据路（干净档 PASS + 上溢/设备NaN/chunk边界三个必红对照）
  --ref-regress  零设备：对 4 档**确定性重生成的输入**跑 旧式/指数差 参考并逐档报差异
  加 --sha256 时把本轮所有 dump 的 sha256 写进 stdout（供 evidence/dump_sha256.txt）
退出码 0 = 全部 PASS，1 = 有 FAIL，2 = 自检/回归与期望不符。
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

BT = 64
DK = 128
DV = 128
MAX_CHUNKS = 128
HE = DK * DV
PH = 2

# ---- T3 ε 的组成常数（来源与依赖见 README §4 推导表）----
U_F32 = 2.0 ** -24        # fp32 单位舍入 u（IEEE754 单精度，binary32 尾数 24 位）
DELTA_EXP = 2.0 ** -23    # Reg::Exp 官方规格「最大精度误差 1 ulp」⇒ 相对 1 ulp（归一化结果）
# ε = 5e-5 = κ∞·(γ₁₂₈ + γ₆₄) + 65·δ_exp + 3.0e-7 向上取整。
#   κ∞ = ‖(I+A)^{-1}‖∞ 在测试数据上 ≤ 3.212（矩阵的精确性质，可由 audit_readme_numbers.py
#   从归档 A 复算，不是误差观测）；**不用** Neumann 界 1/(1−‖A‖∞) —— 实测 ‖A‖∞ = 2.31 > 1，
#   该界不成立（M34 round-3 数字对账发现并修正）。
DEFAULT_EPS = 5e-5

# 非有限分类编号（cat_of 的取值）——口径对齐 M157（m23 判据）/ M165
CAT_FIN, CAT_NAN, CAT_PINF, CAT_NINF = 0, 1, 2, 3


def cat_of(x):
    """按 有限 / NaN / +Inf / −Inf 归类（M157 口径）。

    「NaN↔NaN、同号 Inf↔同号 Inf」视为同类（不计错）；「仅一侧非有限 / NaN↔Inf / 异号 Inf」
    视为不同类（计为**非有限不匹配**，进判定）。
    """
    c = np.full(x.shape, CAT_FIN, dtype=np.int8)
    c[np.isnan(x)] = CAT_NAN
    c[np.isposinf(x)] = CAT_PINF
    c[np.isneginf(x)] = CAT_NINF
    return c


def nf_stats(got, exp):
    """返回一对张量的非有限分类计数（供判定与报告）。"""
    cd, ce = cat_of(got), cat_of(exp)
    dn, rn = cd == CAT_NAN, ce == CAT_NAN
    di = (cd == CAT_PINF) | (cd == CAT_NINF)
    ri = (ce == CAT_PINF) | (ce == CAT_NINF)
    return dict(
        nfd=int((cd != ce).sum()),
        dnan=int(dn.sum()), rnan=int(rn.sum()),
        onlynan=int((dn ^ rn).sum()), bothnan=int((dn & rn).sum()),
        dinf=int(di.sum()), rinf=int(ri.sum()),
        onlyinf=int((di ^ ri).sum()),
        signinf=int((di & ri & (np.sign(got) != np.sign(exp))).sum()),
        fin=(cd == CAT_FIN) & (ce == CAT_FIN))


def gamma_n(n):
    """n 步 fp32 累加的标准界系数 γ_n = n·u/(1−n·u)。"""
    return n * U_F32 / (1.0 - n * U_F32)


CASES = [
    ("gqa3", 3, 257, 0x1101),
    ("one", 8, 64, 0x1102),
    ("all48", 48, 129, 0x1103),
    ("target", 48, 4097, 0x1104),
]

SLOT_GC, SLOT_EG, SLOT_A, SLOT_U, SLOT_W, SLOT_D, SLOT_AB, SLOT_O, SLOT_S = 0, 1, 2, 3, 4, 5, 6, 7, 8
SLOT_ELEMS = {SLOT_GC: BT, SLOT_EG: BT, SLOT_A: BT * BT, SLOT_U: BT * DV, SLOT_W: BT * DK,
              SLOT_D: BT * DV, SLOT_AB: BT * BT, SLOT_O: BT * DV}
SLOT_NAME = {SLOT_GC: "gcum", SLOT_EG: "eg", SLOT_A: "A", SLOT_U: "u", SLOT_W: "w",
             SLOT_D: "d(v_new)", SLOT_AB: "AB", SLOT_O: "o"}


# ================================================================ 输入生成（与 .asc host 侧同公式）
def hash3(i, j, salt):
    h = (i.astype(np.uint64) * np.uint64(2654435761) + j.astype(np.uint64) * np.uint64(40503)
         + np.uint64(salt) * np.uint64(97) + np.uint64(0x9E3779B9)) & np.uint64(0xFFFFFFFF)
    h = h ^ (h >> np.uint64(16))
    h = (h * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    h = h ^ (h >> np.uint64(13))
    return h.astype(np.uint32)


def hash_u11(i, j, salt):
    return (hash3(i, j, salt) % np.uint32(1000000)).astype(np.float32) / np.float32(500000.0) - np.float32(1.0)


def gen_qk(nk, T, salt, dk=DK):
    """与 .asc host 侧 GenQk **逐位一致**：float64 顺序累加 + 46 轮二分求 1/sqrt + float32 乘法。
    （不能用 np.sqrt：那是 0.5 ulp 的精确开方，与 46 轮二分的 ~45 ult 差会让 8.4M 元素里的
     几个元素在 float32 网格上翻边 → T1 逐位判据会误报。）"""
    out = np.empty((nk, T, dk), dtype=np.float32)
    ii = np.arange(dk, dtype=np.uint32)
    for kk in range(nk):
        for t in range(T):
            x = hash_u11(ii, np.uint32(t * 7 + kk), np.uint32(salt)).astype(np.float64)
            s = float(np.cumsum(x * x)[-1])  # 顺序累加（与 C++ 的 `sum += ...` 同序）
            lo, hi = 1e-4, 1.0
            for _ in range(46):
                mid = 0.5 * (lo + hi)
                if mid * mid * s < 1.0:
                    lo = mid
                else:
                    hi = mid
            y = 0.5 * (lo + hi)
            out[kk, t] = (x * y).astype(np.float32)
    return out


def gen_case(H, T, salt, g_override=None):
    """确定性输入（与 .asc host 侧同公式）。

    `g_override`：`{(h,t): value}` —— 生成后覆盖 `g`（供自检合成 chunk 内上溢档；
    覆盖同时作用于 dump 与 T1 重生成，故 T1 判定项不受影响）。
    """
    nk = (H + 2) // 3
    q = gen_qk(nk, T, salt + 0x51)
    k = gen_qk(nk, T, salt + 0x52)
    ti = np.arange(T, dtype=np.uint32)
    hi = np.arange(H, dtype=np.uint32)
    hh, tt = np.meshgrid(hi, ti, indexing="ij")
    ii = np.arange(DK, dtype=np.uint32)
    v = np.empty((H, T, DV), dtype=np.float32)
    for h in range(H):
        for t in range(T):
            v[h, t] = hash_u11(ii, np.uint32(t * 31 + h), np.uint32(salt + 0x53))
    g = (-np.float32(0.001) - (hash3(tt, hh, np.uint32(salt + 0x54)) % np.uint32(50000)).astype(np.float32)
         / np.float32(1000000.0)).astype(np.float32)
    beta = (np.float32(0.05) + (hash3(tt, hh, np.uint32(salt + 0x55)) % np.uint32(900000)).astype(np.float32)
            / np.float32(1000000.0)).astype(np.float32)
    h0 = np.empty((H, DK, DV), dtype=np.float32)
    flat = (np.arange(DK, dtype=np.uint32)[:, None] * np.uint32(128) + np.arange(DV, dtype=np.uint32)[None, :])
    for h in range(H):
        h0[h] = np.float32(0.1) * hash_u11(flat, np.uint32(h), np.uint32(salt + 0x56))
    if g_override:
        for (h, t), val in g_override.items():
            g[h, t] = np.float32(val)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "h0": h0, "nk": nk}


# ================================================================ 参考（值 + Σ|terms| 同时算）
def tril_solve(A, R, tA, tR):
    """(I+A)X = R 的行前代（A 严格下三角）。同时返回 Σ|terms| 张量 tX。

    Σ|terms| 取**一层**直接项：X_i 的直接项是 RHS_i 与 A_ij·X_j(j<i)，
    即 tX_i = |RHS_i| + Σ_j |A_ij|·|X_j|（不把 X_j/A_ij 自身的 terms 递归代入——
    递归代会让状态链形成正反馈、容差指数膨胀成空判据，见 README §4 口径说明）。
    """
    X = R.copy()
    tX = np.abs(R).copy()
    for i in range(1, BT):
        acc = np.zeros((A.shape[0], R.shape[2]), dtype=np.float64)
        for j in range(i):
            acc += A[:, i, j][:, None] * X[:, j, :]
        tX[:, i, :] = np.abs(R[:, i, :]) + np.einsum("hj,hjk->hk", np.abs(A[:, i, :i]), np.abs(X[:, :i, :]))
        X[:, i, :] = X[:, i, :] - acc
    return X, tX


def tril_solve_rec(A, R, tA, X):
    """**递归**口径的 Σ|terms|（把 X_j 自身的 terms 也展开）——只用于示范该口径会正反馈，
    不参与任何判定（见 README §4.1 的口径说明）。"""
    BTn = A.shape[0]
    t = np.abs(R).copy()
    for i in range(1, BTn):
        t[:, i, :] = (np.abs(R[:, i, :])
                      + np.einsum("hj,hjk->hk", np.abs(A[:, i, :i]), t[:, :i, :])
                      + np.einsum("hj,hjk->hk", tA[:, i, :i], np.abs(X[:, :i, :])))
    return t


def reference(inp, scale, need_probe_heads, probe_chunks, stable=True):
    """GDN prefill 的 fp64 逐句参考。

    `stable=True`（默认）：跨 chunk 衰减用**指数差** `exp(ĝ_last−ĝ_j)`（与官方 FLA / m23 的 M165 修法同形）。
    `stable=False`：旧式 —— 物化 `ig=exp(−ĝ)` 再 `d⊙ig`（chunk 内 `|ĝ|` 超 exp 上溢阈值时 `0·inf=NaN`）。
    """
    q, k, v = inp["q"], inp["k"], inp["v"]
    g, beta, h0 = inp["g"], inp["beta"], inp["h0"]
    H, T = v.shape[0], v.shape[1]
    nk = inp["nk"]
    nc = (T + BT - 1) // BT
    hk_idx = np.arange(H) // 3

    o = np.zeros((H, T, DV), dtype=np.float64)
    to = np.zeros((H, T, DV), dtype=np.float64)
    S = h0.astype(np.float64).copy()
    tS = np.abs(S).copy()
    tS_rec = np.abs(S).copy()          # 递归口径（示范用）
    state_hist, stage, stage_terms = {}, {}, {}

    for c in range(nc):
        t0 = c * BT
        cv = min(BT, T - t0)
        kk = np.zeros((H, BT, DK)); qq = np.zeros((H, BT, DK))
        vv = np.zeros((H, BT, DV)); bb = np.zeros((H, BT)); gg = np.zeros((H, BT))
        kk[:, :cv, :] = k[hk_idx, t0:t0 + cv, :]
        qq[:, :cv, :] = q[hk_idx, t0:t0 + cv, :]
        vv[:, :cv, :] = v[:, t0:t0 + cv, :]
        bb[:, :cv] = beta[:, t0:t0 + cv]
        gg[:, :cv] = g[:, t0:t0 + cv]

        gc = np.cumsum(gg, axis=1)
        t_gc = np.cumsum(np.abs(gg), axis=1)  # Σ|terms|（γ_64 由 ε 覆盖，不重复计入）
        eg = np.exp(gc)
        t_eg = np.abs(eg)
        egL = eg[:, cv - 1]
        # 状态更新因子：稳定式取指数差 exp(ĝ_last−ĝ_j)（≤1，有限输入下恒有限）；
        # 旧式物化 ig=exp(−ĝ)（chunk 内 |ĝ| 超 exp 上溢阈值 ⇒ inf ⇒ 与 0 相乘得 NaN）。
        if stable:
            decay = np.exp(gc[:, cv - 1:cv] - gc)          # [H,BT]，跨 chunk 衰减因子
        else:
            decay = np.exp(-gc)
        t_decay = np.abs(decay)
        dgc = gc[:, :, None] - gc[:, None, :]
        Gs = np.tril(np.exp(dgc), -1)
        t_Gs = np.abs(Gs)
        KKT = np.einsum("hid,hjd->hij", kk, kk)
        t_KKT = np.einsum("hid,hjd->hij", np.abs(kk), np.abs(kk))
        A = bb[:, :, None] * Gs * KKT
        t_A = np.abs(bb)[:, :, None] * t_Gs * t_KKT
        Rv = vv * bb[:, :, None]
        t_Rv = np.abs(Rv)
        Rk = kk * (bb * eg)[:, :, None]
        t_Rk = np.abs(Rk)
        u, t_u = tril_solve(A, Rv, t_A, t_Rv)
        w, t_w = tril_solve(A, Rk, t_A, t_Rk)
        wh = np.einsum("hik,hkv->hiv", w, S)
        t_wh = np.einsum("hik,hkv->hiv", np.abs(w), tS) + np.einsum("hik,hkv->hiv", t_w, np.abs(S))
        d = u - wh
        t_d = np.abs(u) + t_wh                     # 自身舍入 + 来自 S/w 的传递
        qs = qq * scale
        o_part_m = np.einsum("hik,hkv->hiv", qs, S)
        o_part = o_part_m * eg[:, :, None]
        t_o_part = np.abs(o_part) + np.einsum("hik,hkv->hiv", np.abs(qs), tS) * t_eg[:, :, None]
        Gi = np.tril(np.exp(dgc), 0)
        t_Gi = np.abs(Gi)
        M = np.einsum("hik,hjk->hij", qs, kk)
        t_M = np.einsum("hik,hjk->hij", np.abs(qs), np.abs(kk))
        AB = Gi * M
        t_AB = t_Gi * t_M + np.abs(AB)
        ABd = np.einsum("hij,hjv->hiv", AB, d)
        t_ABd = np.einsum("hij,hjv->hiv", np.abs(AB), t_d) + np.einsum("hij,hjv->hiv", t_AB, np.abs(d))
        oo = o_part + ABd
        t_oo = np.abs(o_part) + np.einsum("hik,hkv->hiv", np.abs(qs), tS) * t_eg[:, :, None] + t_ABd
        igd = d * decay[:, :, None]
        ktd = np.einsum("hjk,hjv->hkv", kk, igd)
        t_ktd = np.einsum("hjk,hjv->hkv", np.abs(kk), np.abs(igd))
        # 递归口径（示范正反馈；不进判定）
        t_u_rec = tril_solve_rec(A, Rv, t_A, u)
        t_w_rec = tril_solve_rec(A, Rk, t_A, w)
        t_wh_rec = (np.einsum("hik,hkv->hiv", np.abs(w), tS_rec)
                    + np.einsum("hik,hkv->hiv", t_w_rec, np.abs(S)))
        t_d_rec = np.abs(u) + t_wh_rec
        t_igd_rec = t_d_rec * t_decay[:, :, None]
        t_ktd_rec = np.einsum("hjk,hjv->hkv", np.abs(kk), t_igd_rec)
        if stable:
            # S₁ = egL·S₀ + kᵀ(d⊙exp(ĝ_last−ĝ_j))：衰减并进指数差，不物化 ig
            tS_rec = np.abs(egL)[:, None, None] * tS_rec + t_ktd_rec
            S = egL[:, None, None] * S + ktd
            tS = np.abs(egL)[:, None, None] * tS + t_ktd
        else:
            # 旧式：S₁ = egL·(S₀ + kᵀ(d⊙ig))
            tS_rec = np.abs(egL)[:, None, None] * (tS_rec + t_ktd_rec)
            S = egL[:, None, None] * (S + ktd)
            tS = np.abs(egL)[:, None, None] * (tS + t_ktd)

        o[:, t0:t0 + cv, :] = oo[:, :cv, :]
        to[:, t0:t0 + cv, :] = t_oo[:, :cv, :]
        for hx in need_probe_heads:
            state_hist[(hx, c)] = (S[hx].copy(), tS[hx].copy())
        if c in probe_chunks:
            ch = 0 if c == 0 else (1 if c == 1 else 2)
            for hx in need_probe_heads:
                if hx != 0:
                    continue
                for slot, val, tr in ((SLOT_GC, gc, t_gc), (SLOT_EG, eg, t_eg), (SLOT_A, A, t_A),
                                      (SLOT_U, u, t_u), (SLOT_W, w, t_w), (SLOT_D, d, t_d),
                                      (SLOT_AB, AB, t_AB), (SLOT_O, oo, t_oo)):
                    stage[(ch, slot)] = val[hx].copy()
                    stage_terms[(ch, slot)] = tr[hx].copy()
    return o, S, to, tS, state_hist, stage, stage_terms, (float(tS.max()), float(tS_rec.max()))


# ================================================================ 判据工具（判定 / 报告 / guard 分栏）
def ulp_f32(x):
    """fp32 的 1 ulp（按 |x| 所在二进制指数），输入为 float64 值域。"""
    return np.spacing(np.abs(x).astype(np.float32)).astype(np.float64)


class Judge:
    def __init__(self, eps):
        self.eps = eps
        self.verdict = []   # 判定项：(name, 超界/总数, verdict)
        self.report = []    # 报告项
        self.nf = []        # 非有限面逐行（M157 口径：devNaN/refNaN/仅一侧NaN/…/nfd）
        self.guard = []     # guard
        self.bad = 0

    def chk(self, name, got, exp, terms):
        got = np.asarray(got, dtype=np.float64)
        exp = np.asarray(exp, dtype=np.float64)
        terms = np.asarray(terms, dtype=np.float64)
        if got.shape != exp.shape:
            self.verdict.append((name, "SHAPE", "FAIL got %s exp %s" % (got.shape, exp.shape)))
            self.bad += 1
            return
        st = nf_stats(got, exp)
        fin = st["fin"]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            dif = np.abs(got - exp)
            ulp = ulp_f32(exp)
            tol = self.eps * terms + 0.5 * ulp
            # 容差只在**两侧皆有限**的元素上比（非有限面由 st["nfd"] 单列计入判定）
            bad_fin = int((fin & (dif > tol)).sum())
            if fin.any():
                dif_f = dif[fin]
                mx = float(dif_f.max())
                rel = float((dif_f / np.maximum(np.abs(exp)[fin], 1e-300)).max())
                ulp1 = float((dif_f <= ulp[fin]).mean())
                occ = float((dif_f / np.maximum(tol[fin], 1e-300)).max())
                occ_rel = float((dif_f / (self.eps * np.abs(exp)[fin] + 0.5 * ulp[fin])).max())
                mxt = float(terms[fin].max())
                mxe = float(np.abs(exp)[fin].max())
            else:
                mx = rel = ulp1 = occ = occ_rel = mxt = mxe = 0.0
        nbad = bad_fin + st["nfd"]
        verdict = "PASS" if nbad == 0 else "FAIL(%d/%d)" % (nbad, dif.size)
        if nbad:
            self.bad += 1
        self.verdict.append((name, "0/%d" % dif.size if nbad == 0 else "%d/%d" % (nbad, dif.size), verdict))
        self.report.append((name, "%.3e" % mx, "%.3e" % rel, "%.1f%%" % (100 * ulp1),
                            "%.3e" % occ, "%.3e" % mxt, "%.3e" % mxe, "%.3e" % occ_rel))
        self.nf.append((name, st))

    def chk_exact(self, name, got, exp):
        got = np.asarray(got, dtype=np.float64)
        exp = np.asarray(exp, dtype=np.float64)
        st = nf_stats(got, exp)
        fin = st["fin"]
        neq = int((fin & (got != exp)).sum()) + st["nfd"]
        if neq:
            self.bad += 1
        self.verdict.append((name, "0/%d" % got.size if neq == 0 else "%d/%d" % (neq, got.size),
                             "PASS" if neq == 0 else "FAIL"))
        with np.errstate(invalid="ignore"):
            mx = float(np.abs(got - exp)[fin].max()) if fin.any() else 0.0
            mxe = float(np.abs(exp)[fin].max()) if fin.any() else 0.0
            eq = float((fin & (got == exp)).sum()) / got.size if got.size else 1.0
        self.report.append((name, "0.000e+00" if neq == 0 else "%.3e" % mx, "0.000e+00",
                            "%.1f%%" % (100 * eq), "0.000e+00", "%.3e" % mxe, "%.3e" % mxe, "0.000e+00"))
        self.nf.append((name, st))

    def gd(self, name, ok, detail=""):
        self.guard.append((name, "PASS" if ok else "FAIL", detail))

    def gdna(self, name, detail=""):
        self.guard.append((name, "N/A", detail))

    def dump(self):
        print("  [判定项]（必须 0 超界）")
        print("    %-26s %-13s %s" % ("stage", "超界/总数", "verdict"))
        for r in self.verdict:
            print("    %-26s %-13s %s" % r)
        print("  [报告项]（不参与 PASS/FAIL）")
        print("    %-26s %-11s %-11s %-8s %-11s %-11s %-11s %s" % ("stage", "max|Δ|(绝对)", "maxRel(相对)",
                                                                "≤1ulp比例", "T3最差占用", "max Σ|terms|",
                                                                "max|exp|", "相对|out|占用"))
        for r in self.report:
            print("    %-26s %-11s %-11s %-8s %-11s %-11s %-11s %s" % r)
        print("  [非有限面]（M157 口径；nfd = 非有限不匹配，计入判定）")
        print("    %-26s %7s %7s %8s %8s %7s %7s %8s %8s %8s"
              % ("stage", "devNaN", "refNaN", "仅一侧NaN", "双侧NaN", "devInf", "refInf",
                 "仅一侧Inf", "异号Inf", "nfd"))
        for name, st in self.nf:
            print("    %-26s %7d %7d %8d %8d %7d %7d %8d %8d %8d"
                  % (name, st["dnan"], st["rnan"], st["onlynan"], st["bothnan"], st["dinf"],
                     st["rinf"], st["onlyinf"], st["signinf"], st["nfd"]))
        print("  [guard]（不参与 PASS/FAIL）")
        for r in self.guard:
            print("    %-34s %-5s %s" % r)


# ================================================================ 单 case
def run_case(tag, H, T, salt, args, g_override=None):
    pre = "m18_h%02u_t%05u_" % (H, T)
    meta = {}
    with open(pre + "meta.txt") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                meta[p[0]] = p[1]
    nk = int(meta["NK"]); nc = int(meta["nc"])
    scale = float(meta["scale"]); probe_heads = int(meta["probeHeads"])

    def rd(name):
        return np.fromfile(pre + name + ".bin", dtype=np.float32)

    q = rd("q").reshape(nk, T, DK).astype(np.float64)
    k = rd("k").reshape(nk, T, DK).astype(np.float64)
    v = rd("v").reshape(H, T, DV).astype(np.float64)
    g = rd("g").reshape(H, T).astype(np.float64)
    beta = rd("beta").reshape(H, T).astype(np.float64)
    h0 = rd("h0").reshape(H, DK, DV).astype(np.float64)
    o_k = rd("o").reshape(H, T, DV).astype(np.float64)
    ht_k = rd("ht").reshape(H, DK, DV).astype(np.float64)
    probe = rd("probe").astype(np.float64)

    print("[%s] H=%u T=%u nk=%u nc=%u scale=%.9g probeHeads=%u  (T3 ε=%.1e)"
          % (tag, H, T, nk, nc, scale, probe_heads, args.eps))
    j = Judge(args.eps)

    # ---- (0) T1：输入六张量逐位 ----
    inp = gen_case(H, T, salt, g_override=g_override)
    for nm, got in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("h0", h0)):
        j.chk_exact("T1:input:" + nm, got, inp[nm].astype(np.float64))

    # ---- (1) 参考 ----
    probe_chunks = {0, 1, nc - 1}
    need_ph = list(range(min(probe_heads, H)))
    ref_inp = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "h0": h0, "nk": nk}
    # 旧式参考在溢出档会触发 exp 上溢（正是要量的现象）；非有限计数下面显式报出，故此处静音。
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        o_r, ht_r, to_r, tht_r, state_hist, stage, stage_terms, terms_pair = reference(
            ref_inp, scale, need_ph, probe_chunks, stable=args.stable)
    print("  [terms-口径对照] 一层+传递 max Σ|terms|(S)=%.3e ; 递归展开 max=%.3e（比值 %.3g；"
          "递归口径在状态链上正反馈，故口径定为「一层+传递」，见 README §4.1）"
          % (terms_pair[0], terms_pair[1], terms_pair[1] / max(terms_pair[0], 1e-300)))

    # ---- (2) 端到端（T3）----
    j.chk("o[all_heads,tok]", o_k, o_r, to_r)
    j.chk("ht[final_state]", ht_k, ht_r, tht_r)

    # ---- (3) 逐 chunk 状态（probe slot 8，T3）----
    base = probe[: PH * MAX_CHUNKS * HE].reshape(PH, MAX_CHUNKS, DK, DV)
    for hx in need_ph:
        for c in range(nc):
            val, tr = state_hist[(hx, c)]
            j.chk("S[head%u,chunk%u]" % (hx, c), base[hx, c], val, tr)

    # ---- (4) 分段抽点（chunk 0 / 1 / 末块，T3）----
    for ch, c in ((0, 0), (1, 1), (2, nc - 1)):
        for slot in (SLOT_GC, SLOT_EG, SLOT_A, SLOT_U, SLOT_W, SLOT_D, SLOT_AB, SLOT_O):
            if (ch, slot) not in stage:
                continue
            off = (PH * MAX_CHUNKS + ch * 9 + slot) * HE
            n = SLOT_ELEMS[slot]
            j.chk("%s[chunk%u]" % (SLOT_NAME[slot], c), probe[off:off + n], stage[(ch, slot)].reshape(-1),
                  stage_terms[(ch, slot)].reshape(-1))

    # ---- (5) guard：非空洞性 / 结构性 / dump 位置（§4）----
    for nm, arr in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("h0", h0)):
        j.gd("G1 输入非空洞:" + nm, np.abs(arr).max() > 0 and arr.std() > 0,
             "max|·|=%.3e std=%.3e" % (np.abs(arr).max(), arr.std()))
    evo = 0
    for hx in need_ph:
        for c in range(1, nc):
            if not np.array_equal(state_hist[(hx, c)][0], state_hist[(hx, c - 1)][0]):
                evo += 1
    if nc > 1:
        j.gd("G2 状态跨 chunk 演化", evo == len(need_ph) * (nc - 1),
             "演化 %d/%d 个 (head,chunk) 边界" % (evo, len(need_ph) * (nc - 1)))
    else:
        j.gd("G2 状态跨 chunk 演化", True, "N/A（nc=1，无跨 chunk 边界）")
    moved = int(sum(1 for hx in range(H) if not np.array_equal(ht_k[hx], h0[hx])))
    j.gd("G2 终态≠初态(all heads)", moved == H, "%d/%d 个 head" % (moved, H))
    upper = 0
    for ch in (0, 1, 2):
        if (ch, SLOT_A) in stage:
            upper = max(upper, float(np.abs(np.triu(stage[(ch, SLOT_A)], 1)).max()))
        if (ch, SLOT_AB) in stage:
            upper = max(upper, float(np.abs(np.triu(stage[(ch, SLOT_AB)], 1)).max()))
    j.gd("G3 A/AB 严格上三角恒 0", upper == 0.0, "max|上三角|=%.1e" % upper)
    mono = 0
    for ch in (0, 1, 2):
        if (ch, SLOT_GC) in stage:
            gg = stage[(ch, SLOT_GC)]
            mono = max(mono, int((np.diff(gg) > 0).sum()))
    j.gd("G3 ĝ 单调不增(g≤0)", mono == 0, "递增位置数=%d" % mono)
    zch = 0
    if T % BT != 0:
        ch_last = 0 if nc == 1 else (1 if nc == 2 else 2)
        if (ch_last, SLOT_O) in stage:
            cv_last = T - (nc - 1) * BT
            tail = stage[(ch_last, SLOT_O)][cv_last:]  # 尾块 cv_last 行之后应为 0
            zch = int(np.abs(tail).max() == 0.0 if tail.size else 1)
            j.gd("G3 ragged 尾块 padding 行=0", zch == 1,
                 "末块(cv=%d) slot7 第 %d 行起 max|·|=%.1e" % (cv_last, cv_last,
                                                              np.abs(tail).max() if tail.size else 0.0))
        else:
            j.gdna("G3 ragged 尾块 padding 行=0", "未抽到末块 slot7")
    else:
        j.gdna("G3 ragged 尾块 padding 行=0", "N/A（T 是 BT 整数倍，无 ragged 尾块）")
    beyond = float(np.abs(base[:, nc:, :, :]).max()) if nc < MAX_CHUNKS else 0.0
    j.gd("G3 probe 未用 chunk 槽位=0", beyond == 0.0, "S[head,chunk≥%d] max|·|=%.1e" % (nc, beyond))
    distinct = all(not np.array_equal(base[0, c], base[0, c + 1]) for c in range(min(nc - 1, 8)))
    j.gd("G3 probe 槽位可区分", distinct, "前 9 个 chunk 状态互不相同")
    sha_path = None
    for cand in ("evidence/dump_sha256.txt", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "evidence", "dump_sha256.txt")):
        if os.path.exists(cand):
            sha_path = cand
            break
    if sha_path is not None:
        want = {}
        for line in open(sha_path):
            p = line.split()
            if len(p) == 2:
                want[p[1]] = p[0]
        nm = "m18_h%02u_t%05u_ht.bin" % (H, T)
        got = hashlib.sha256(open(nm, "rb").read()).hexdigest()
        j.gd("G4 确定性(dump sha256 对照)", want.get(nm, "") == got, "ht sha256=%s" % got[:16])
    else:
        j.gdna("G4 确定性(dump sha256 对照)", "无 manifest（--sha256 可生成）")

    j.dump()
    print("[%s] %s" % (tag, "PASS" if j.bad == 0 else "FAIL"))
    return j.bad == 0


# ================================================================ 零设备：合成自检 / 参考侧回归
def _assemble_probe(nc, need_ph, state_hist, stage):
    """按 run_case 的 probe 布局，把参考的 state_hist/stage 组装成 probe 向量。"""
    probe = np.zeros((PH * MAX_CHUNKS + 3 * 9) * HE, dtype=np.float64)
    base = probe[:PH * MAX_CHUNKS * HE].reshape(PH, MAX_CHUNKS, DK, DV)
    for hx in need_ph:
        for c in range(nc):
            base[hx, c] = state_hist[(hx, c)][0]
    for ch, c in ((0, 0), (1, 1), (2, nc - 1)):
        for slot, elems in SLOT_ELEMS.items():
            if (ch, slot) in stage:
                off = (PH * MAX_CHUNKS + ch * 9 + slot) * HE
                probe[off:off + elems] = stage[(ch, slot)].reshape(-1)
    return probe


def _ref_outputs(inp, scale, H, T, stable, probe_heads=2):
    """跑一次参考，返回 (o, ht, probe)；probe 按 run_case 布局组装。"""
    nc = (T + BT - 1) // BT
    need_ph = list(range(min(probe_heads, H)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        o, S, to, tS, state_hist, stage, stage_terms, _ = reference(
            inp, scale, need_ph, {0, 1, nc - 1}, stable=stable)
    return o, S, _assemble_probe(nc, need_ph, state_hist, stage)


def synth_dump(outdir, H, T, salt, ref_stable=True, g_override=None,
               dev_stable=None, dev_nan=False, dev_from_T=None):
    """零设备合成一份 m18 格式 dump（同 run_case 的文件布局）。返回参考用的输入。

    * 设备侧默认 = **同一参考**（float32 舍入）⇒ 对该参考判据应 PASS（非空洞对照）。
    * `dev_stable`：设备侧另用一套参考回填（供「另一式参考 vs 本设备档」对照）。
    * `dev_nan`：把设备侧 `o[0,0,0:16]` 置 NaN（参考仍有限）⇒ 判据必须红（仅一侧 NaN）。
    * `dev_from_T`：设备侧 o/ht/probe 取自 `dev_from_T` 个 token 的参考（< T，错一个 chunk 边界）。
    """
    inp = gen_case(H, T, salt, g_override=g_override)
    scale = float(np.float32(1.0) / np.sqrt(np.float32(DK)))
    nk = inp["nk"]
    o_r, ht_r, _ = _ref_outputs(inp, scale, H, T, ref_stable)
    ref_dev = ref_stable if dev_stable is None else dev_stable
    if dev_from_T is not None and dev_from_T < T:
        inp2 = {k2: (v2[:, :dev_from_T] if k2 in ("q", "k", "v", "g", "beta") else v2)
                for k2, v2 in inp.items() if k2 != "nk"}
        inp2["nk"] = nk
        o_d, ht_d, probe_d = _ref_outputs(inp2, scale, H, dev_from_T, ref_dev)
        o_pad = np.zeros((H, T, DV), dtype=np.float64)      # 设备侧只覆盖前 dev_from_T 行
        o_pad[:, :dev_from_T, :] = o_d
        o_d = o_pad
    else:
        o_d, ht_d, probe_d = _ref_outputs(inp, scale, H, T, ref_dev)
    if dev_nan:
        o_d = o_d.copy(); o_d[0, 0, 0:16] = np.nan
    os.makedirs(outdir, exist_ok=True)

    def w(name, arr):
        np.asarray(arr, dtype=np.float32).tofile(
            os.path.join(outdir, "m18_h%02u_t%05u_%s.bin" % (H, T, name)))

    w("q", inp["q"]); w("k", inp["k"]); w("v", inp["v"]); w("g", inp["g"])
    w("beta", inp["beta"]); w("h0", inp["h0"]); w("o", o_d); w("ht", ht_d); w("probe", probe_d)
    with open(os.path.join(outdir, "m18_h%02u_t%05u_meta.txt" % (H, T)), "w") as f:
        f.write("NK %d\nnc %d\nscale %.17g\nprobeHeads 2\n" % (nk, (T + BT - 1) // BT, scale))
    return inp, o_r, ht_r


def selftest():
    """零设备自检：合成小 dump 走**与真判据同一条** run_case 路。必须的三档：

    (1) 干净档（设备=稳定参考）⇒ 稳定判据 PASS（非空洞对照）；
    (2) 上溢档（g[0,0]=−900）⇒ **旧式判据 FAIL**（参考侧 0·inf=NaN；复现病态）；
    (3) 设备侧 NaN 注入 ⇒ FAIL（仅一侧 NaN）；
    (4) 错一个 chunk 边界（设备取自 T−1 的参考）⇒ FAIL。
    """
    base = tempfile.mkdtemp(prefix="m18_nan_selftest_")
    print("[SELFTEST] 合成目录：%s（零设备、零探针）" % base)
    rc = 0
    cwd = os.getcwd()

    class _A:
        def __init__(self, stable):
            self.eps = DEFAULT_EPS
            self.stable = stable

    def judge(dirpath, tag, H, T, salt, stable, g_override=None):
        os.chdir(dirpath)
        try:
            return run_case(tag, H, T, salt, _A(stable), g_override=g_override)
        finally:
            os.chdir(cwd)

    checks = []

    # (1) 干净档
    d = os.path.join(base, "syn")
    synth_dump(d, H=3, T=8, salt=0x1171)
    ok = judge(d, "syn", 3, 8, 0x1171, True)
    checks.append(("syn 干净（稳定判据应 PASS）", ok, True))

    # (2) 上溢档：设备=稳定参考（有限）；旧式判据必须红
    og = {(0, 0): -900.0}
    d = os.path.join(base, "syn_overflow")
    synth_dump(d, H=3, T=8, salt=0x1171, g_override=og)
    ok_leg = judge(d, "syn_overflow", 3, 8, 0x1171, False, g_override=og)
    checks.append(("syn_overflow 旧式判据（0·inf ⇒ 应 FAIL）", ok_leg, False))
    ok_stab = judge(d, "syn_overflow", 3, 8, 0x1171, True, g_override=og)
    checks.append(("syn_overflow 指数差判据（应 PASS）", ok_stab, True))

    # (3) 设备侧 NaN 注入
    d = os.path.join(base, "syn_devmut")
    synth_dump(d, H=3, T=8, salt=0x1171, dev_nan=True)
    ok = judge(d, "syn_devmut", 3, 8, 0x1171, True)
    checks.append(("syn_devmut 设备侧 NaN（应 FAIL）", ok, False))

    # (4) 错一个 chunk 边界：T=65，设备取自 T=64 的参考
    d = os.path.join(base, "syn_chunkmut")
    synth_dump(d, H=3, T=65, salt=0x1172, dev_from_T=64)
    ok = judge(d, "syn_chunkmut", 3, 65, 0x1172, True)
    checks.append(("syn_chunkmut 错一个 chunk 边界（应 FAIL）", ok, False))

    print("[SELFTEST] ===== 汇总 =====")
    for name, got, expect in checks:
        mark = "✓" if got == expect else "✗"
        print("[SELFTEST] %-46s 期望 %s / 实得 %s  %s"
              % (name, "PASS" if expect else "FAIL", "PASS" if got else "FAIL", mark))
        if got != expect:
            rc = 1
    print("[SELFTEST] %s（rc=%d，合成目录 %s）"
          % ("各档符合期望" if rc == 0 else "有档与期望不符", rc, base))
    shutil.rmtree(base, ignore_errors=True)   # 自清合成档：/tmp 下不留 m18_h*（免与"归档 dump"混淆）
    return rc


def _maxrel(a, b):
    """联合有限且 b≠0 的元素上，|a−b|/|b| 的最大值。"""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    m = np.isfinite(d) & (np.abs(b) > 0)
    return float((d[m] / np.abs(b)[m]).max()) if m.any() else 0.0


def ref_regress(only=""):
    """零设备：对 4 档**确定性重生成的输入**跑 旧式/指数差 参考，逐档报参考侧差异。

    只覆盖参考侧：m18 的设备 dump **未归档**（`.gitignore` 的 `build/`、`*.bin`；
    `git log --all --diff-filter=A -- 'm18_gdn_prefill/**/*.bin'` = 0），零设备下无法重跑设备档。
    输出：逐档 旧式 refNaN / 指数差 refNaN / 两式在联合有限元素上的 max|Δ|（o 与 ht）。
    """
    print("[REF-REGRESS] 参考侧逐档（输入由 gen_case 逐位重生成；m18 设备 dump 未归档，设备档不可重跑）")
    hdr = "  %-7s %4s %5s %4s %11s %11s %12s %12s %12s"
    print(hdr % ("case", "H", "T", "nc", "旧式refNaN", "指数差refNaN",
                 "max|Δ|o", "max|Δ|ht", "max|Δ|hg"))
    hdr2 = "  %-7s %-34s %12s %12s"
    print(hdr2 % ("case", "Σ|terms| 相对差（to=o项 / tS=终态项）", "max rel to", "max rel tS"))
    rc = 0
    scale = float(np.float32(1.0) / np.sqrt(np.float32(DK)))
    for tag, H, T, salt in CASES:
        if only and only != tag:
            continue
        inp = gen_case(H, T, salt)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            o_l, ht_l, to_l, tS_l, _, _, _, _ = reference(inp, scale, [], set(), stable=False)
            o_s, ht_s, to_s, tS_s, _, _, _, _ = reference(inp, scale, [], set(), stable=True)
        nf_l = int(np.isnan(o_l).sum()) + int(np.isnan(ht_l).sum())
        nf_s = int(np.isnan(o_s).sum()) + int(np.isnan(ht_s).sum())
        do = np.abs(o_s - o_l); mo = np.isfinite(do)
        dh = np.abs(ht_s - ht_l); mh = np.isfinite(dh)
        # Σ|terms| 口径同步改写后两式的相对差（证明归档 §4.4 的 terms/占用数字不需重定基）
        rel_to = _maxrel(to_s, to_l)
        rel_ts = _maxrel(tS_s, tS_l)
        print(hdr % (tag, H, T, (T + BT - 1) // BT, nf_l, nf_s,
                     "%.3e" % (do[mo].max() if mo.any() else 0.0),
                     "%.3e" % (dh[mh].max() if mh.any() else 0.0),
                     "%.3e" % (dh[mh].max() if mh.any() else 0.0)))
        print(hdr2 % (tag, "（exp(ĝ_L−ĝ_j) ≡ egL·ig，仅差 fp 舍入）", "%.3e" % rel_to, "%.3e" % rel_ts))
        if nf_s:
            print("[REF-REGRESS] %s：指数差参考仍有非有限 ⇒ rc=1" % tag)
            rc = 1
    print("[REF-REGRESS] 说明：以上为**参考侧**两式差异；设备侧判定差异需设备 dump（本 mission 零设备、无归档）")
    print("[REF-REGRESS] rc=%d（0=各档指数差参考恒有限且两式差异记录在案；1=指数差参考出现非有限）" % rc)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS, help="T3 ε（默认 %.1e，推导见 README §4.2）" % DEFAULT_EPS)
    ap.add_argument("--only", default="")
    ap.add_argument("--sha256", action="store_true", help="打印本轮 dump 的 sha256（供 evidence/）")
    ap.add_argument("--legacy-exp", action="store_true",
                    help="参考改用旧式 `exp(ĝ)·exp(−ĝ)` 物化相乘（仅对照；默认指数差）")
    ap.add_argument("--selftest", action="store_true",
                    help="零设备自检：合成 dump 走同一条判据路（干净 PASS + 上溢/设备 NaN/chunk 边界三个必红对照）")
    ap.add_argument("--ref-regress", action="store_true",
                    help="零设备：对 4 档确定性重生成输入跑 旧式/指数差 参考并逐档报差异")
    args = ap.parse_args()
    args.stable = not args.legacy_exp

    if args.selftest:
        return 2 if selftest() else 0
    if args.ref_regress:
        return 2 if ref_regress(args.only) else 0

    if args.sha256:
        print("# m18_gdn_prefill dump sha256（本文件由 check_ref.py --sha256 生成）")
        for tag, H, T, _ in CASES:
            for suf in ("q", "k", "v", "g", "beta", "h0", "o", "ht", "probe", "meta"):
                nm = "m18_h%02u_t%05u_%s.%s" % (H, T, suf, "txt" if suf == "meta" else "bin")
                if os.path.exists(nm):
                    h = hashlib.sha256(open(nm, "rb").read()).hexdigest()
                    print("%s  %s" % (h, nm))
        return 0

    if not args.stable:
        print("[REF] 注意：--legacy-exp —— 参考用旧式 exp(ĝ)·exp(−ĝ)（仅对照，溢出档会自造 NaN）")
    allok = True
    for tag, H, T, salt in CASES:
        if args.only and args.only != tag:
            continue
        try:
            allok &= run_case(tag, H, T, salt, args)
        except FileNotFoundError as e:
            print("[%s] SKIP: %s" % (tag, e))
            allok = False
    print("===== %s =====" % ("ALL CASES PASS" if allok else "FAILURES PRESENT"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
