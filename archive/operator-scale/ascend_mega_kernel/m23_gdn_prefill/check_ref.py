#!/usr/bin/env python3
"""check_ref.py —— M115 / Wave B1 的独立判据（numpy float64 逐句参考）

与被测对象的关系：
  * 被测 = `m15_layer_loop/m15_gdn_prefill.h`（mmad 版 GDN prefill chunk 扫描段），
    其顶层驱动 = `m23_gdn_prefill/m23_gdn_prefill.asc`（真实 shape：m=4097 与 m=1，H=48）。
  * 参考 = 本文件的 numpy **float64** 逐句实现，公式**逐句**取自 m18_gdn_prefill（算法已证）
    与 docs/15 §3.1 的 golden 口径（含 m49 的三处订正：`L = +(…)`、`T=(I+L)^{-1}`、
    `decay_mask` 两处对角口径、`e^ĝ` 只乘一次）。参考**不含** conv1d / l2norm / gating
    —— 本段的输入契约就是「已经过 prolog 的 q/k/v/g/β」（见 README §边界与未完成项）。
  * **M165 数值稳定化**：Γ（`Gs`/`Gi`）与跨 chunk 衰减改用**指数差** `exp(ĝ_i−ĝ_j)`
    （`i≥j` 时 ≤1；有限输入下恒有限），与官方 FLA 一致（`vllm/.../chunk_o.py:119-120`、
    `chunk_delta_h.py:216-221`）。旧式「`eg=exp(ĝ)`、`ig=exp(−ĝ)` 分开物化再相乘」在 chunk 内
    `|ĝ|>709`（fp64 `exp` 上溢阈值；fp32 是 88.7）时 `0·inf=NaN` —— 那是 **golden 自身的数值病态**，
    不再是默认路径；`--legacy-exp` 可切回旧式做对照，`--stab-demo` 在同一份 dump 上并排打印
    两式的 NaN 计数（修前 / 修后）。

判据（非空洞纪律，docs/17 §4；M157 订正见下）：
  * **有限子集容差**：o 的 [0, cv) 行与 ht 全量，在**两侧皆有限**的元素上逐元素
    `|dev-ref| <= rtol*max(1,|ref|)`；任一元素超界即 FAIL。
  * **非有限面（M157 新增，`cat_of()`）**：NaN/Inf 显式纳入判定 ——
      - NaN↔NaN 视为相等（不计错，但**不由此断言该处数值正确**）；
      - 同号 Inf↔同号 Inf 视为相等；
      - 仅一侧 NaN、仅一侧 Inf、NaN↔Inf、异号 Inf ⇒ **一律计为不匹配（FAIL）**。
    报告单列「设备 NaN 数 / 参考 NaN 数 / 仅一侧 NaN 数 / 双侧 NaN 数」。
    * 订正理由：旧判据数的是 `|dev-ref|>lim`（base commit `ac46c32` 的 `check_ref.py:147-149`），
      该比较**对 NaN 静默为 False** ⇒ 设备与参考都含 NaN 的那一大片会落进盲区，「0 超界」不能
      用来说明那一片。（M151 归档档实测：设备 NaN 8,374,400 / 参考 NaN 3,670,912 / 仅一侧 4,703,488，
      旧判据在同一份数据上报 0 超界 —— 复现见 README §4n 与 `--selftest`。）
  * **适用范围与残余空档**：两侧都为 NaN 的位置**不计错**，也**不**由此断言"数值正确" ——
    它只是"两侧同样非有限"，判据不声称那片数值可信。有限子集上的 max|Δ| **只在两侧皆有限的元素上取**
    （旧写法 `max(finite, nan)` / `max(finite, inf)` 会把非有限项折掉或抬成 inf，掩盖读数）。
  * 反向对照：`--mutant` 读 `m23_<tag>_mut<k>_*.bin`（host 侧按 k 搅动输入）⇒ 同一判据**必须变红**。
    这条是「判据能咬住」的证据：若反向对照也 PASS，说明判据是空的。

用法：
  <python> check_ref.py                  # 判 m23_m1_* 与 m23_p4097_*（默认：指数差参考）
  <python> check_ref.py --mutant 1       # 判反向对照档（必须 FAIL）
  <python> check_ref.py --dir <dump目录>
  <python> check_ref.py --legacy-exp     # 判据改用旧式 exp(ĝ)·exp(−ĝ)（仅对照）
  <python> check_ref.py --stab-demo --dir <dump目录>   # 并排打印 旧式/指数差 × fp64/fp32 的 NaN 计数
  <python> check_ref.py --selftest       # 零设备自检：旧盲区 + 上溢档（指数差 PASS/旧式 FAIL）
"""

import argparse
import glob
import os
import sys
import tempfile

import numpy as np

BT = 64
DK = 128
DV = 128

# 非有限分类编号（cat_of 的取值）
CAT_FIN = 0     # 有限
CAT_NAN = 1     # NaN
CAT_PINF = 2    # +Inf
CAT_NINF = 3    # -Inf


def load_bin(path, dtype=np.float32):
    return np.fromfile(path, dtype=dtype)


def read_meta(path):
    meta = {}
    with open(path, "r") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                meta[p[0]] = float(p[1]) if ("." in p[1] or "e" in p[1]) else int(p[1])
    return meta


def cat_of(x):
    """按 有限 / NaN / +Inf / -Inf 归类。

    判据用它把「NaN↔NaN、同号 Inf↔同号 Inf」视为同类（不计错），
    「仅一侧非有限 / NaN↔Inf / 异号 Inf」视为不同类（计错）。
    """
    c = np.full(x.shape, CAT_FIN, dtype=np.int8)
    c[np.isnan(x)] = CAT_NAN
    c[np.isposinf(x)] = CAT_PINF
    c[np.isneginf(x)] = CAT_NINF
    return c


def ref_head(q, k, v, g, beta, h0, scale, T, stable=True, dtype=np.float64):
    """单 value head 的逐句参考；q/k 为 [T,DK]，v 为 [T,DV]，g/beta 为 [T]。

    `stable=True`（默认）：Γ 用**指数差** `exp(ĝ_i−ĝ_j)`（i≥j 时 ≤1，有限输入下恒有限）；
    跨 chunk 衰减同样并进指数差 `exp(ĝ_last−ĝ_j)`，不再物化 `ig=exp(−ĝ)`。
    `stable=False`：旧式 —— `eg=exp(ĝ)`、`ig=exp(−ĝ)` 分开物化再相乘（仅用于修前/修后对照）；
    chunk 内 `|ĝ|>709`（fp64 上溢阈值）时 `0·inf=NaN`。`dtype` 供 fp32/fp64 对照实验用。
    """
    S = h0.astype(dtype).copy()               # [DK,DV]
    o = np.zeros((T, DV), dtype=dtype)
    nc = (T + BT - 1) // BT
    ii = np.arange(BT)[:, None]; jj = np.arange(BT)[None, :]
    for c in range(nc):
        t0 = c * BT
        cv = min(BT, T - t0)
        sl = slice(t0, t0 + cv)
        kc = np.zeros((BT, DK), dtype=dtype); qc = np.zeros((BT, DK), dtype=dtype)
        vc = np.zeros((BT, DV), dtype=dtype)
        gc = np.zeros(BT, dtype=dtype); bc = np.zeros(BT, dtype=dtype)
        kc[:cv] = k[sl]; qc[:cv] = q[sl]; vc[:cv] = v[sl]
        gc[:cv] = g[sl]; bc[:cv] = beta[sl]
        gcum = np.cumsum(gc)                                       # ĝ（chunk 内）
        eg = np.exp(gcum); egL = eg[cv - 1]
        if stable:
            # 只在下三角求 exp（上三角的 ĝ_i−ĝ_j>0 直接置 0，避开无关上溢）
            Gs = np.where(ii > jj, np.exp(np.where(ii > jj, gcum[:, None] - gcum[None, :], dtype(0.0))), dtype(0.0))
            Gi = np.where(ii >= jj, np.exp(np.where(ii >= jj, gcum[:, None] - gcum[None, :], dtype(0.0))), dtype(0.0))
        else:
            ig = np.exp(-gcum)
            Gs = np.where(ii > jj, eg[:, None] * ig[None, :], dtype(0.0))
            Gi = np.where(ii >= jj, eg[:, None] * ig[None, :], dtype(0.0))
        kk = kc @ kc.T                                             # 64×64×128
        A = np.tril(bc[:, None] * Gs * kk, -1)                     # 严格下三角
        M = np.eye(BT, dtype=dtype) + A
        try:
            u = np.linalg.solve(M, bc[:, None] * vc)
            w = np.linalg.solve(M, bc[:, None] * eg[:, None] * kc)
        except np.linalg.LinAlgError:
            # 旧式溢出时 M 含 inf/nan ⇒ 该 chunk 记非有限（稳定式不会走到这里）
            u = np.full((BT, DV), np.nan, dtype=dtype)
            w = np.full((BT, DK), np.nan, dtype=dtype)
        d = u - w @ S
        qs = scale * qc
        o_chunk = (qs @ S) * eg[:, None] + (Gi * (qs @ kc.T)) @ d
        o[sl] = o_chunk[:cv]
        if stable:
            decay_col = np.exp(gcum[cv - 1] - gcum)                # ĝ_last−ĝ_j ≤ 0 ⇒ ≤1
            S = egL * S + kc.T @ (d * decay_col[:, None])
        else:
            S = egL * (S + kc.T @ (d * ig[:, None]))
    return o, S


def new_acc():
    return dict(bad=0, tot=0, nfd=0, dnan=0, rnan=0, onlynan=0, bothnan=0,
                dinf=0, rinf=0, onlyinf=0, signinf=0, mx=0.0, mxref=0.0)


def acc_pair(dev, ref, rtol, acc):
    """把一对 (device, ref) 张量的判定量累加进 acc（有限子集容差 + 非有限面）。"""
    dev = dev.astype(np.float64)
    cd = cat_of(dev); cr = cat_of(ref)
    dn = cd == CAT_NAN; rn = cr == CAT_NAN
    di = (cd == CAT_PINF) | (cd == CAT_NINF)
    ri = (cr == CAT_PINF) | (cr == CAT_NINF)
    fin = (cd == CAT_FIN) & (cr == CAT_FIN)
    do = np.abs(dev - ref)
    lim = rtol * np.maximum(1.0, np.abs(ref))
    acc["bad"] += int((fin & (do > lim)).sum())     # 只在两侧皆有限处比容差
    acc["tot"] += int(dev.size)
    acc["nfd"] += int((cd != cr).sum())             # 非有限分类不一致 ⇒ 计错（含 NaN↔Inf）
    acc["dnan"] += int(dn.sum()); acc["rnan"] += int(rn.sum())
    acc["onlynan"] += int((dn ^ rn).sum()); acc["bothnan"] += int((dn & rn).sum())
    acc["dinf"] += int(di.sum()); acc["rinf"] += int(ri.sum())
    acc["onlyinf"] += int((di ^ ri).sum())
    acc["signinf"] += int((di & ri & (np.sign(dev) != np.sign(ref))).sum())
    dfin = do[np.isfinite(do)]                      # max|Δ| 只在两侧皆有限的元素上取
    if dfin.size:
        acc["mx"] = max(acc["mx"], float(dfin.max()))
    rfin = np.abs(ref)[np.isfinite(ref)]
    if rfin.size:
        acc["mxref"] = max(acc["mxref"], float(rfin.max()))


def judge_one(dirpath, tag, rtol, stable=True):
    """判一份 dump，返回判定量的字典（不打印）。`stable=False` 用旧式 Γ（对照用）。"""
    meta = read_meta(os.path.join(dirpath, "m23_%s_meta.txt" % tag))
    H = int(meta["H"]); T = int(meta["T"]); nk = int(meta["NK"])
    nAic = int(meta.get("nAic", 0)); scale = float(meta["scale"])
    Tp = T + 64                                   # 尾部补齐 64 行（Nd2Nz 固读 64 行）
    pref = os.path.join(dirpath, "m23_%s_" % tag)

    q = load_bin(pref + "q.bin").reshape(nk, Tp, DK)
    k = load_bin(pref + "k.bin").reshape(nk, Tp, DK)
    v = load_bin(pref + "v.bin").reshape(H, T, DV)
    g = load_bin(pref + "g.bin").reshape(H, T)
    beta = load_bin(pref + "beta.bin").reshape(H, T)
    h0 = load_bin(pref + "h0.bin").reshape(H, DK, DV)
    o_dev = load_bin(pref + "o.bin").reshape(H, T, DV)
    ht_dev = load_bin(pref + "ht.bin").reshape(H, DK, DV)

    acc = dict(o=new_acc(), h=new_acc())
    for hv in range(H):
        hk = hv // 3
        # 旧式 Γ 会触发 exp 上溢告警（正是本档要量的现象）；非有限计数下面显式报出，故此处静音。
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            o_ref, S_ref = ref_head(q[hk], k[hk], v[hv], g[hv], beta[hv], h0[hv], scale, T,
                                    stable=stable)
        acc_pair(o_dev[hv], o_ref, rtol, acc["o"])
        acc_pair(ht_dev[hv], S_ref, rtol, acc["h"])
    return dict(tag=tag, H=H, T=T, nAic=nAic, o=acc["o"], h=acc["h"])


def verdict(rec):
    """有限子集容差与非有限面**都必须**干净才算 PASS。"""
    return (rec["o"]["bad"] == 0 and rec["o"]["nfd"] == 0
            and rec["h"]["bad"] == 0 and rec["h"]["nfd"] == 0)


def render(rec, rtol):
    o = rec["o"]; h = rec["h"]
    return [
        ("[REF] %-14s H=%2d T=%-4d nAic=%u rtol=%.1e | o 超界 %d/%d max|Δ|=%.3e max|ref|=%.3e | "
         "ht 超界 %d/%d max|Δ|=%.3e max|ref|=%.3e | %s"
         % (rec["tag"], rec["H"], rec["T"], rec["nAic"], rtol,
            o["bad"], o["tot"], o["mx"], o["mxref"], h["bad"], h["tot"], h["mx"], h["mxref"],
            "PASS" if verdict(rec) else "FAIL")),
        ("[REF] %-14s 非有限 | o: devNaN=%d refNaN=%d 仅一侧NaN=%d 双侧NaN=%d devInf=%d refInf=%d "
         "仅一侧Inf=%d 异号Inf=%d 非有限不匹配=%d | ht: devNaN=%d refNaN=%d 仅一侧NaN=%d "
         "双侧NaN=%d 非有限不匹配=%d"
         % (rec["tag"], o["dnan"], o["rnan"], o["onlynan"], o["bothnan"], o["dinf"], o["rinf"],
            o["onlyinf"], o["signinf"], o["nfd"], h["dnan"], h["rnan"], h["onlynan"], h["bothnan"],
            h["nfd"])),
    ]


def _load_dump(dirpath, tag):
    """读一份 m23 dump 的输入面与设备输出（返回字典）。"""
    meta = read_meta(os.path.join(dirpath, "m23_%s_meta.txt" % tag))
    H = int(meta["H"]); T = int(meta["T"]); nk = int(meta["NK"]); scale = float(meta["scale"])
    Tp = T + 64
    pref = os.path.join(dirpath, "m23_%s_" % tag)
    return dict(
        meta=meta, H=H, T=T, nk=nk, scale=scale,
        q=load_bin(pref + "q.bin").reshape(nk, Tp, DK),
        k=load_bin(pref + "k.bin").reshape(nk, Tp, DK),
        v=load_bin(pref + "v.bin").reshape(H, T, DV),
        g=load_bin(pref + "g.bin").reshape(H, T),
        beta=load_bin(pref + "beta.bin").reshape(H, T),
        h0=load_bin(pref + "h0.bin").reshape(H, DK, DV),
        o_dev=load_bin(pref + "o.bin").reshape(H, T, DV),
        ht_dev=load_bin(pref + "ht.bin").reshape(H, DK, DV))


def _ref_all(d, stable, dtype):
    """按 dtype 跑全 H 头的参考，返回 (o_ref[H,T,DV], ht_ref[H,DK,DV])。"""
    H, T = d["H"], d["T"]
    o = np.zeros((H, T, DV), dtype=dtype)
    ht = np.zeros((H, DK, DV), dtype=dtype)
    for hv in range(H):
        hk = hv // 3
        o[hv], ht[hv] = ref_head(d["q"][hk].astype(dtype), d["k"][hk].astype(dtype),
                                 d["v"][hv].astype(dtype), d["g"][hv].astype(dtype),
                                 d["beta"][hv].astype(dtype), d["h0"][hv].astype(dtype),
                                 dtype(d["scale"]), T, stable=stable, dtype=dtype)
    return o, ht


def stab_demo(dirpath):
    """在同一份 dump 上并排跑 旧式/指数差 × fp64/fp32，打印 o/ht 的 NaN 计数。

    修前（`stable=False`）与修后（`stable=True`）用**同一份输入面**复算 ⇒ 「修前 N 个 NaN /
    修后 0 个」在同一脚本、同一数据上可复现（M165 的「修前 8,374,272 / 修后 0」即此行）。
    判据（可传播失败）：任一「指数差」档出现 NaN ⇒ 退出码 1；另打印两式在**联合有限**元素上的
    max|Δ|，用来证明指数差没有改掉计算结果（只是不再物化 `ig`）。
    """
    tags = []
    for meta in sorted(glob.glob(os.path.join(dirpath, "m23_*_meta.txt"))):
        tags.append(os.path.basename(meta)[len("m23_"):-len("_meta.txt")])
    if not tags:
        print("[STAB-DEMO] 没找到 dump（%s 下无 m23_*_meta.txt）" % dirpath)
        return 2
    rc = 0
    for tag in tags:
        try:
            d = _load_dump(dirpath, tag)
        except (FileNotFoundError, ValueError) as e:
            print("[STAB-DEMO] %-14s SKIP: %s" % (tag, e))
            rc = max(rc, 2)
            continue
        dev_nan = int(np.isnan(d["o_dev"]).sum()) + int(np.isnan(d["ht_dev"]).sum())
        print("[STAB-DEMO] %-14s H=%d T=%d | 设备 o/ht NaN = %d" % (tag, d["H"], d["T"], dev_nan))
        res = {}
        for label, stable, dt in (("旧式", False, np.float64), ("旧式", False, np.float32),
                                  ("指数差", True, np.float64), ("指数差", True, np.float32)):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                o, ht = _ref_all(d, stable, dt)
            onan = int(np.isnan(o).sum()); hnan = int(np.isnan(ht).sum())
            res[(label, np.dtype(dt).name)] = (o, ht, onan, hnan)
            print("[STAB-DEMO] %-14s %-8s %-7s | o NaN = %d / %d | ht NaN = %d / %d%s"
                  % (tag, label, np.dtype(dt).name, onan, o.size, hnan, ht.size,
                     "  ← 修后仍非有限（FAIL）" if (stable and (onan or hnan)) else ""))
            if stable and (onan or hnan):
                rc = 1
        for dt in (np.float64, np.float32):
            nm = np.dtype(dt).name
            lo, lh = res[("旧式", nm)][0], res[("旧式", nm)][1]
            so, sh = res[("指数差", nm)][0], res[("指数差", nm)][1]
            do = np.abs(so - lo); m = np.isfinite(do)
            hd = np.abs(sh - lh); hm = np.isfinite(hd)
            print("[STAB-DEMO] %-14s %-7s | 联合有限 max|指数差−旧式|: o=%.3e ht=%.3e"
                  % (tag, nm, float(do[m].max()) if m.any() else 0.0,
                     float(hd[hm].max()) if hm.any() else 0.0))
    print("[STAB-DEMO] rc=%d（0=指数差各档均有限；1=指数差仍有非有限；2=无 dump 或读取失败）" % rc)
    return rc


def synth_dump(outdir, tag, H=3, T=8, DK_=DK, DV_=DV, NK=1, scale=0.0883883476,
               seed=7, ref_nan=False, dev_nan=False, overflow=False):
    """零设备合成一份 m23 格式 dump（**指数差**参考回填 o/ht）；返回 (o_dev, o_ref) 供自检展示。

    ref_nan=True：只把**喂参考的输入** `g[0, T-1]` 置 NaN ⇒ 参考侧一片 NaN、设备档仍有限。
    dev_nan=True：只把**设备侧 dump** `o[0, 1, 0:16]` 置 NaN ⇒ 设备侧多一片 NaN、参考仍有限。
    overflow=True：把 `g[0,0]` 置 −900 ⇒ chunk 内 `ĝ` 越 fp64/fp32 `exp` 上溢阈值；设备档用
      **指数差**参考回填（有限），故「指数差判据」应 PASS、旧式判据应因参考侧 NaN 变红。
    """
    rng = np.random.default_rng(seed)
    Tp = T + 64
    q = rng.standard_normal((NK, Tp, DK_)).astype(np.float32)
    k = (0.05 * rng.standard_normal((NK, Tp, DK_))).astype(np.float32)   # 小 k ⇒ M=I+A 良态
    v = rng.standard_normal((H, T, DV_)).astype(np.float32)
    g = (-0.05 * rng.random((H, T))).astype(np.float32)     # 小负值 ⇒ 默认档参考不溢出
    if overflow:
        g[0, 0] = np.float32(-900.0)                        # chunk 内 |ĝ| 远超上溢阈值
    beta = rng.random((H, T)).astype(np.float32)
    h0 = (0.1 * rng.standard_normal((H, DK_, DV_))).astype(np.float32)
    o_ref = np.zeros((H, T, DV_), dtype=np.float64)
    S_ref = np.zeros((H, DK_, DV_), dtype=np.float64)
    for hv in range(H):
        hk = hv // 3
        o_ref[hv], S_ref[hv] = ref_head(q[hk], k[hk], v[hv], g[hv], beta[hv], h0[hv], scale, T)
    o_dev = o_ref.astype(np.float32).copy()
    ht_dev = S_ref.astype(np.float32).copy()
    g_in = g.copy()
    if dev_nan:
        o_dev[0, 1, 0:16] = np.nan
    if ref_nan:
        g_in[0, T - 1] = np.nan
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "m23_%s_meta.txt" % tag), "w") as f:
        f.write("H %d\nT %d\nNK %d\nDK %d\nDV %d\nBT %d\nscale %r\nnAic 1\n"
                % (H, T, NK, DK_, DV_, BT, scale))

    def w(name, arr):
        arr.astype(np.float32).tofile(os.path.join(outdir, "m23_%s_%s.bin" % (tag, name)))

    w("q", q); w("k", k); w("v", v); w("g", g_in); w("beta", beta); w("h0", h0)
    w("o", o_dev); w("ht", ht_dev)
    return o_dev.astype(np.float64), o_ref


def selftest():
    """零设备自检：合成小 dump 走**与真判据同一条** judge_one/render/verdict 路。

    四个档：(1) 干净档必须 PASS；(2a) 参考侧 NaN 必须红且报「仅一侧 NaN」；
    (2b) 设备侧 NaN 必须红且报「仅一侧 NaN」；(3) 合成 chunk 内 `|ĝ|>709` 的上溢档：
    **指数差**判据必须 PASS、**旧式**判据必须因参考侧 NaN 变红 —— 这是 M165 修法的零设备见证。
    并对 (1)(2) 同时打印**旧式** `|dev-ref|>lim` 计数，以复现「NaN 被静默放行」。
    """
    rtol = 2e-3
    base = tempfile.mkdtemp(prefix="m23_nan_selftest_")
    print("[SELFTEST] 合成目录：%s（零设备、零探针）" % base)
    rc = 0
    for tag, ref_nan, dev_nan, overflow in (("syn", False, False, False),
                                            ("syn_refmut", True, False, False),
                                            ("syn_devmut", False, True, False),
                                            ("syn_overflow", False, False, True)):
        d = os.path.join(base, tag)
        o_dev, o_ref = synth_dump(d, tag, ref_nan=ref_nan, dev_nan=dev_nan, overflow=overflow)
        rec = judge_one(d, tag, rtol)
        print("[SELFTEST] tag=%s（参考侧注入 NaN=%s，设备侧注入 NaN=%s，chunk 内上溢=%s）"
              % (tag, ref_nan, dev_nan, overflow))
        for ln in render(rec, rtol):
            print("           " + ln)
        expect_pass = not (ref_nan or dev_nan)
        got_pass = verdict(rec)
        mark = "✓" if got_pass == expect_pass else "✗"
        print("[SELFTEST] 期望 %s / 实得 %s  %s"
              % ("PASS" if expect_pass else "FAIL", "PASS" if got_pass else "FAIL", mark))
        if got_pass != expect_pass:
            rc = 1
        do = np.abs(o_dev - o_ref)
        lim = rtol * np.maximum(1.0, np.abs(o_ref))
        old_bad = int((do > lim).sum())
        nf = int((~np.isfinite(o_dev) | ~np.isfinite(o_ref)).sum())
        print("[SELFTEST] 旧式 `|dev-ref|>lim` 计数 = %d（该数据非有限元素 %d 个）" % (old_bad, nf))
        if not expect_pass and old_bad == 0 and nf > 0:
            print("[SELFTEST] ⇒ 复现「NaN 被静默放行」：旧判据报 0 超界，新判据报 非有限不匹配 > 0")
        if overflow:
            # 同档换旧式 Γ：参考侧应出现 NaN ⇒ 判据必须红（证明修法有效、且旧式确有此病）
            rec_leg = judge_one(d, tag, rtol, stable=False)
            rnan = rec_leg["o"]["rnan"] + rec_leg["h"]["rnan"]
            print("[SELFTEST] 同档旧式 Γ：参考 NaN=%d，判定=%s"
                  % (rnan, "FAIL" if not verdict(rec_leg) else "PASS"))
            if rnan == 0 or verdict(rec_leg):
                print("[SELFTEST] ✗ 旧式 Γ 未复现非有限 ⇒ 本档不能作为修法见证")
                rc = 1
            else:
                print("[SELFTEST] ✓ 指数差 PASS / 旧式 FAIL —— 修法见证（同一份上溢输入）")
        print("")
    print("[SELFTEST] %s（rc=%d）" % ("各档符合期望" if rc == 0 else "有档与期望不符", rc))
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--mutant", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=2e-3)
    ap.add_argument("--allow-stale", action="store_true", help="跳过新鲜度守卫（默认拒绝判定陈旧 dump）")
    ap.add_argument("--selftest", action="store_true", help="零设备自检：合成小 dump 演示 NaN 盲区、上溢档与负向对照")
    ap.add_argument("--legacy-exp", action="store_true",
                    help="判据改用旧式 `exp(ĝ)·exp(−ĝ)` 物化相乘（仅对照；默认指数差）")
    ap.add_argument("--stab-demo", action="store_true",
                    help="零设备：对 --dir 的 dump 并排跑 旧式/指数差 × fp64/fp32，打印 o/ht 的 NaN 计数")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.stab_demo:
        return stab_demo(args.dir)
    stable = not args.legacy_exp

    tags = []
    for meta in sorted(glob.glob(os.path.join(args.dir, "m23_*_meta.txt"))):
        base = os.path.basename(meta)
        tag = base[len("m23_"):-len("_meta.txt")]
        if args.mutant:
            if tag.endswith("_mut%d" % args.mutant):
                tags.append(tag)
        elif "_mut" not in tag:
            tags.append(tag)
    if not tags:
        print("[REF] 没找到 dump（先跑 ./m23_gdn_prefill 生成 m23_*_meta.txt）")
        return 2

    # ---- 新鲜度守卫（r2 复审 p2-b）：dump 若早于当前二进制，就拒绝判定 ----
    binp = None
    for cand in (os.path.join(args.dir, "build", "m23_gdn_prefill"),
                 os.path.join(args.dir, "..", "build", "m23_gdn_prefill"),
                 os.path.join(args.dir, "m23_gdn_prefill")):
        if os.path.exists(cand):
            binp = cand
            break
    if binp is not None:
        bmt = os.path.getmtime(binp)
        fresh, stale = [], []
        for t in tags:
            m = os.path.getmtime(os.path.join(args.dir, "m23_%s_meta.txt" % t))
            (fresh if m >= bmt else stale).append(t)
        if stale:
            print("[REF][陈旧] 下列 dump 早于当前二进制（%s，mtime %.0f）⇒ 拒绝判定：%s"
                  % (binp, bmt, ", ".join(stale)))
            print("[REF][陈旧] 若要强行判定请加 --allow-stale；先跑 ./m23_gdn_prefill 生成新鲜 dump。")
            if not args.allow_stale:
                tags = fresh
        if not tags:
            return 3

    if not stable:
        print("[REF] 注意：--legacy-exp —— 参考用旧式 exp(ĝ)·exp(−ĝ)（仅对照，可能自造 NaN）")
    all_pass = True
    for tag in tags:
        rec = judge_one(args.dir, tag, args.rtol, stable=stable)
        for ln in render(rec, args.rtol):
            print(ln)
        all_pass &= verdict(rec)

    if args.mutant:
        # 反向对照的**期望**是 FAIL；若这里 PASS，说明判据咬不住，必须当成失败报出。
        print("[REF] 反向对照（mut%d）：%s" % (args.mutant, "如预期变红 ✓" if not all_pass else "判据没咬住 ✗"))
        return 0 if not all_pass else 1
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
