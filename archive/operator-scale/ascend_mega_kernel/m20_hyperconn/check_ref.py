#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m20_hyperconn/check_ref.py —— M36 hyper-connection 融合 kernel 的**独立 numpy 交叉校验**

用法（在 dump 目录里跑；先用 M20_DUMP=1 M20_CASE=<case> 落盘）：
    mkdir -p /tmp/m20_dump && cd /tmp/m20_dump
    M20_DUMP=1 M20_CASE=real_m1 <repo>/m20_hyperconn/build/m20_hyperconn
    /usr/local/python3.12.13/bin/python3 <repo>/m20_hyperconn/check_ref.py [case]

## 退出码（三态，tower 规则；随 `4fe9ac8` 落地）

    0 = **比过且通过**（OK，文案里带实际比较条数）
    1 = **比过且有差异**（FAILED，列出判定项名）
    2 = **没得比 / 输入缺失**（SKIPPED）——"我没比"与"我比过通过"必须能区分

**负向对照**（必须做：喂空输入，确认它不发合格证）：
    mkdir -p /tmp/m20_empty && cd /tmp/m20_empty
    /usr/local/python3.12.13/bin/python3 <repo>/m20_hyperconn/check_ref.py real_m1; echo "rc=$?"
    # 期望：打印 RESULT: SKIPPED（缺 m20_real_m1_*.bin）且 rc=2
    实测读数归档 `evidence/check_ref_negative_control.log`。

## 三段的分工与「覆盖范围」交代

    A. 端到端      设备 vs 独立 numpy float64 参考          → 判定项（计入 JUDGE 计数）
    B. 分段链      以**设备自己的上游 dump** 为输入逐段复算  → 判定项（计入 JUDGE 计数）
    C. 非空洞性    6 个输入整张量 ×1.5 的敏感度 + 取值多样性 → **guard**（单列计数，不计入判定项）

**判据口径的自我交代（随 `4fe9ac8` 落地；`4fe9ac8` 时 = 良态 ulpMax ≤2）**：本脚本的 A/B 判定项用的是
**独立门限**：张量尺度归一化绝对误差 ≤1e-2 + 良态逐位一致率 ≥99% + **良态元素 ulp>2 的占比 ≤1e-3**
（2026-10-04 M174 按 docs/17 §9.11 规则⑪ 登记的口径修正；原为「良态 ulpMax ≤2」，旧 ulpMax 现仍作**报告项**逐张量打印）。
它**不是** docs/17 §1.1 意义上的分档判定项——docs/17 的 T1/T2′/T3 分档判据在 **kernel 内**实现
（`m20_hyperconn.asc` 的 `JudgeT1/T3` + 三类计数器）。两者互为独立见证：本脚本门限的方向与 docs/17 一致，
且**不以「实测最大 X ulp」当界**（该 max 只作报告）；改动前后对当前全部读数的唯一判定差异见
`m20_hyperconn/README.md` §7.8。⚠ **本门限不代表实现无缺陷** —— 它只覆盖「良态元素的细小 ulp 尾部」这一形态。
「跳过/无法判定」一律单列成 `SKIPPED` 计数并打印（例如 mode 0 档的 `injw`、输入恒 0 的探针），
**不静默吞掉**。

计数与比较**由同一份代码产出**（每个 `judge()`/`read_bin()` 调用点同时自增计数），
不存在"一份正则匹配、另一份手写数字"的脱节。
"""


import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "golden"))
from moe_block_ref import bf16_bits_to_f32, f32_to_bf16_bits  # noqa: E402

# ---- 与 m20_resources.h 同源的形状常量（不许出现第二套定义） ----
HC = 4
HID = 2560
HYPER = HC * HID
LOWRANK = 320
M_MAX = 64
INJ_N = 4
IJ_STRIDE = 16
INJW_SLOT = 8
EPS = 1e-6
OH_W = 336
DN_N = LOWRANK + INJ_N
UP_N = HYPER
UP_K = LOWRANK
MODE_MIX, MODE_COMBINE_MIX, MODE_FINAL_MIX = 0, 1, 2

# ---- 判定门限（M174 按 docs/17 §9.11 规则⑪ 登记的口径修正；见 README §7.8 的举证/登记/复审）----
# 原：良态元素 ulpMax ≤2（`maxulp <= 2`）。2026-10-04 改为分位口径：
#   良态元素里 ulp>2 的**占比** ≤ SIG_ULP_OVER2_MAX。`ulpMax` 仍逐张量打印为**报告项**。
SIG_ULP_OVER2_MAX = 1e-3
SFRAC_MIN = 0.99
NORM_ABS_MAX = 1e-2

FAILS = []
JUDGE = [0]      # 判定项（本脚本的独立门限：A 端到端 + B 分段链）
GUARD = [0]      # guard（结构性前提：C 非空洞性探针 + 取值多样性；不计入判定项）
SKIPPED = [0]    # 没得比/不适用（三态里的 unverified，必须可见、不静默）


class MissingInput(Exception):
    """缺 dump 文件 —— 对应退出码 2（SKIPPED），不是"比对失败"。"""


def read_bin(path, dtype):
    if not os.path.exists(path):
        raise MissingInput(path)   # → main 报 SKIPPED + rc=2
    meta = path + ".meta"
    shape = None
    if os.path.exists(meta):
        with open(meta) as f:
            parts = f.read().split()
        if len(parts) >= 3:
            shape = tuple(int(x) for x in parts[1:-1])   # meta 行格式: name <dims...> dtype
    a = np.fromfile(path, dtype=dtype)
    if shape is not None:
        a = a.reshape(shape)
    return a


def f32(u16):
    return bf16_bits_to_f32(u16.astype(np.uint16))


def b16(x):
    """舍入到 bf16 网格，返回 uint16 位模式（保持输入形状）。"""
    return f32_to_bf16_bits(np.asarray(x, dtype=np.float32).reshape(-1)).reshape(np.shape(x))


def q(x):
    """舍入到 bf16 网格，返回值（float64）——参考里每次「落盘 bf16」都必须过这一步。"""
    return bf16_bits_to_f32(b16(x)).astype(np.float64)


def ulp_key(u16):
    u = u16.astype(np.int32)
    return np.where(u & 0x8000, -(u & 0x7FFF), u & 0x7FFF)


def judge(name, got_u16, exp_f64, results):
    """逐张量判据：张量尺度归一化绝对误差 ≤1e-2 + 良态逐位率 ≥0.99 + 良态 ulp>2 占比 ≤1e-3。

    `ulpMax` 仍每次打印（报告项），但**不再**是判定子条件（M174 口径修正，见 README §7.8）。
    """
    got = f32(got_u16).astype(np.float64)
    scale = float(np.max(np.abs(exp_f64))) if exp_f64.size else 0.0
    norm_abs = float(np.max(np.abs(got - exp_f64))) / scale if scale > 0 else 0.0
    exp_bits = b16(exp_f64)
    d = np.abs(ulp_key(got_u16) - ulp_key(exp_bits))
    sig = np.abs(exp_f64) > 0.05 * scale
    sfrac = float(np.mean(d[sig] == 0)) if np.any(sig) else 1.0
    over2 = float(np.mean(d[sig] > 2)) if np.any(sig) else 0.0
    maxulp = int(np.max(d[sig])) if np.any(sig) else 0     # 报告项（保留；不作判定）
    frac = float(np.mean(d == 0))
    ok = norm_abs <= NORM_ABS_MAX and over2 <= SIG_ULP_OVER2_MAX and sfrac >= SFRAC_MIN
    JUDGE[0] += 1
    if not ok:
        FAILS.append(name)
    print("[chk] 判定 %-24s n=%-7d 逐位=%.4f 良态逐位=%.4f 良态ulp>2占比=%.3e ulpMax=%-3d 归一maxAbs=%.3e  %s"
          % (name, got_u16.size, frac, sfrac, over2, maxulp, norm_abs, "PASS" if ok else "FAIL"))
    results[name] = (got, exp_f64, ok)
    return ok


# ============================================================
# 独立 double 参考（逐句实现 README §1.3）
# ============================================================
def reference(m, mode, hin, bo, ij, wdown, winj, wup, norm):
    r = {}
    # W0: injW = 2·sigmoid(IJ/HC)
    ijv = f32(ij[:m, :HC]).astype(np.float64)
    injw = 2.0 / (1.0 + np.exp(-ijv / HC))
    r["injw"] = injw
    # S1: H' = bf16(H + BO·injW[s])
    H = f32(hin[:m]).astype(np.float64).reshape(m, HC, HID)
    BO = f32(bo[:m]).astype(np.float64)
    if mode == MODE_MIX:
        hc = H.copy()
    else:
        hc = q(H + BO[:, None, :] * injw[:, :, None]).reshape(m, HYPER)
    r["hc"] = hc.reshape(m, HYPER)
    # S2: per (token,stream) grouped GemmaRMSNorm
    x = hc.reshape(m, HC, HID)
    var = np.sum(x * x, axis=2) / HID
    rrms = 1.0 / np.sqrt(var + EPS)
    W = f32(norm).astype(np.float64).reshape(HC, HID)
    y = x * rrms[:, :, None]
    y = y + y * W[None, :, :]
    xn = q(y).reshape(m, HYPER)
    r["xn"] = xn
    r["rstd"] = rrms
    # S3: down+inject 合并 GEMM（fp64 累加 → bf16）
    lora = np.empty((m, DN_N), dtype=np.float64)
    xn32 = xn.astype(np.float64)
    wd = f32(wdown).astype(np.float64)          # [LOWRANK, HYPER]
    wi = f32(winj[:INJ_N]).astype(np.float64)   # [INJ_N, HYPER]
    acc_dn = xn32 @ wd.T
    acc_in = xn32 @ wi.T
    lora[:, :LOWRANK] = q(acc_dn)
    lora[:, LOWRANK:] = q(acc_in)
    r["lora"] = lora
    # S4: silu(lora/HC)
    u = lora[:, :LOWRANK] / HC
    ls = q(u / (1.0 + np.exp(-u)))
    r["ls"] = ls
    # S5: up GEMM
    wu = f32(wup).astype(np.float64)            # [UP_N, UP_K]
    gate = q(ls @ wu.T)
    r["gate"] = gate
    # S6: gate mix
    g = gate.reshape(m, HC, HID)
    xx = xn.reshape(m, HC, HID)
    sacc = np.zeros((m, HID), dtype=np.float64)
    for s in range(HC):
        sacc += (1.0 / (1.0 + np.exp(-g[:, s, :]))) * xx[:, s, :]
    r["blk"] = q(sacc / HC)
    return r


def main():
    try:
        return _main()
    except MissingInput as e:
        print("[chk] RESULT: SKIPPED (缺输入文件：%s)"
              "\n[chk] 先用 M20_DUMP=1 M20_CASE=<case> 在空目录跑一次 kernel 再复核。" % e)
        return 2


def _main():
    case = sys.argv[1] if len(sys.argv) > 1 else None
    if case is None:
        if os.path.exists("m20_case.txt"):
            info = dict(l.split(None, 1) for l in open("m20_case.txt").read().splitlines() if l.strip())
            case = info["name"]
        else:
            raise SystemExit("用法: check_ref.py <case>（或先跑一次 M20_DUMP=1 生成 m20_case.txt）")
    info = {}
    if os.path.exists("m20_case.txt"):
        info = dict(l.split(None, 1) for l in open("m20_case.txt").read().splitlines() if l.strip())
    if info.get("name") != case:
        print("[chk] 警告：dump 目录里的 case 是 %r，命令行给的是 %r —— 以命令行/文件名 m20_%s_* 为准"
              % (info.get("name"), case, case))
    m = int(info.get("m", 1))
    mode = int(info.get("mode", MODE_COMBINE_MIX))
    print("[chk] case=%s m=%u mode=%u（数学口径见 README §1.3）" % (case, m, mode))

    def rd(suffix, dtype):
        return read_bin("m20_%s_%s.bin" % (case, suffix), dtype)

    hin = rd("hin", np.uint16)
    bo = rd("bo", np.uint16)
    ij = rd("ij", np.uint16)
    wdown = rd("wdown", np.uint16)
    winj = rd("winj", np.uint16)
    wup = rd("wup", np.uint16)
    norm = rd("hc_norm", np.uint16)
    dev = {
        "hcp": rd("hcp", np.uint16),
        "xn": rd("xn", np.uint16),
        "rstd": rd("rstd", np.float32),
        "injw": rd("injw", np.float32),
        "oh": rd("oh", np.uint16),
        "ls": rd("ls", np.uint16),
        "gate": rd("gate", np.uint16),
        "blk": rd("blk", np.uint16),
    }

    # ---------- A. 端到端：设备 vs 独立 double 参考 ----------
    print("\n== A. 端到端（设备 vs 独立 numpy double 参考）==")
    ref = reference(m, mode, hin, bo, ij, wdown, winj, wup, norm)
    res = {}
    if mode != MODE_MIX:
        judge("h' (combine 输出)", dev["hcp"][:m], ref["hc"], res)
    judge("xn (norm 输出)", dev["xn"][:m], ref["xn"], res)
    judge("lora (OH[:, :320])", dev["oh"][:m, 0:LOWRANK], ref["lora"][:, :LOWRANK], res)
    if mode != MODE_FINAL_MIX:
        judge("inject (OH[:,320:324])", dev["oh"][:m, LOWRANK:LOWRANK + INJ_N],
              ref["lora"][:, LOWRANK:], res)
    judge("ls (silu 输出)", dev["ls"][:m], ref["ls"], res)
    judge("gate (up GEMM 输出)", dev["gate"][:m], ref["gate"], res)
    judge("blk (gate mix 输出)", dev["blk"][:m], ref["blk"], res)
    if mode != MODE_MIX:
        ijt = dev["injw"].reshape(-1, INJW_SLOT)
        got = np.array([ijt[mi * HC + s, 0] for mi in range(m) for s in range(HC)])
        rel = float(np.max(np.abs(got.astype(np.float64) - ref["injw"].reshape(-1)) /
                           np.maximum(np.abs(ref["injw"].reshape(-1)), 1e-30)))
        JUDGE[0] += 1
        ok = rel <= 1e-6
        if not ok:
            FAILS.append("injw")
        print("[chk] 判定 %-20s maxRel=%.3e  %s" % ("injw (fp32)", rel, "PASS" if ok else "FAIL"))
    # RSTD 在 GM 里是「按 (token,stream) 展平」的 fp32 序列（槽 stride = 1，即 (M_MAX,HC) 行主序）
    rel = float(np.max(np.abs(dev["rstd"][:m].reshape(-1).astype(np.float64) - ref["rstd"].reshape(-1)) /
                       np.abs(ref["rstd"].reshape(-1))))
    JUDGE[0] += 1
    ok = rel <= 1e-4
    if not ok:
        FAILS.append("rstd")
    print("[chk] 判定 %-20s maxRel=%.3e  %s" % ("rstd (fp32)", rel, "PASS" if ok else "FAIL"))

    # ---------- B. 分段链：以设备自己的上游 dump 为输入逐段复算 ----------
    # 目的：把端到端残差按段归因（若 A 段全过而这里某段失败，说明该段本身有问题而非误差传递）。
    # 判据与 A 段同口径（judge 内做显著性过滤），输入换成设备的中间张量。
    print("\n== B. 分段链（输入 = 设备自己的上游 dump；考察设备内部自洽）==")
    if mode != MODE_MIX:
        ijv = f32(ij[:m, :HC]).astype(np.float64)
        ijw = 2.0 / (1.0 + np.exp(-ijv / HC))
        H = f32(hin[:m]).astype(np.float64).reshape(m, HC, HID)
        BO = f32(bo[:m]).astype(np.float64)
        judge("B1 H'<-ij,hin,bo", dev["hcp"][:m],
              (H + BO[:, None, :] * ijw[:, :, None]).reshape(m, HYPER), res)
        g = dev["injw"].reshape(-1, INJW_SLOT)[: m * HC, 0].astype(np.float64)
        # injW 本身是 fp32 张量（不做 bf16 舍入），单列 maxRel 判据
        injw_ref = 2.0 / (1.0 + np.exp(-f32(ij[:m, :HC]).astype(np.float64) / HC))
        relw = float(np.max(np.abs(g - injw_ref.reshape(-1)) / np.abs(injw_ref.reshape(-1))))
        JUDGE[0] += 1
        ok = relw <= 1e-6
        if not ok:
            FAILS.append("B/injw")
        print("[chk] 判定 %-20s maxRel=%.3e  %s" % ("B2 injW<-dev vs ref", relw, "PASS" if ok else "FAIL"))

    # 以 device 的 H'/H 为输入复算 norm
    src = f32(dev["hcp"][:m]) if mode != MODE_MIX else f32(hin[:m])
    x = src.astype(np.float64).reshape(m, HC, HID)
    var = np.sum(x * x, axis=2) / HID
    rrms = 1.0 / np.sqrt(var + EPS)
    W = f32(norm).astype(np.float64).reshape(HC, HID)
    y = x * rrms[:, :, None]
    judge("B3 xn<-dev H',hc_norm", dev["xn"][:m], (y + y * W[None, :, :]).reshape(m, HYPER), res)
    relr = float(np.max(np.abs(dev["rstd"][:m].reshape(-1).astype(np.float64) - rrms.reshape(-1)) /
                        np.abs(rrms.reshape(-1))))
    JUDGE[0] += 1
    ok = relr <= 1e-4
    if not ok:
        FAILS.append("B/rstd")
    print("[chk] 判定 %-20s maxRel=%.3e  %s" % ("B3b rstd<-dev H'", relr, "PASS" if ok else "FAIL"))

    xnD = f32(dev["xn"][:m]).astype(np.float64)
    wd = f32(wdown).astype(np.float64)
    wi = f32(winj[:INJ_N]).astype(np.float64)
    judge("B4 lora<-dev xn,wdown", dev["oh"][:m, :LOWRANK], xnD @ wd.T, res)
    if mode != MODE_FINAL_MIX:
        judge("B5 inject<-dev xn,winj", dev["oh"][:m, LOWRANK:LOWRANK + INJ_N], xnD @ wi.T, res)

    loraD = f32(dev["oh"][:m, :LOWRANK]).astype(np.float64)
    u = loraD / HC
    judge("B6 ls<-dev lora", dev["ls"][:m], u / (1.0 + np.exp(-u)), res)

    lsD = f32(dev["ls"][:m]).astype(np.float64)
    wu = f32(wup).astype(np.float64)
    judge("B7 gate<-dev ls,wup", dev["gate"][:m], lsD @ wu.T, res)

    gg = f32(dev["gate"][:m]).astype(np.float64).reshape(m, HC, HID)
    xx = f32(dev["xn"][:m]).astype(np.float64).reshape(m, HC, HID)
    sacc = np.zeros((m, HID), dtype=np.float64)
    for s in range(HC):
        sacc += (1.0 / (1.0 + np.exp(-gg[:, s, :]))) * xx[:, s, :]
    judge("B8 blk<-dev gate,xn", dev["blk"][:m], sacc / HC, res)

    # ---------- C. 非空洞性 ----------
    print("\n== C. 非空洞性（输入整张量 ×1.5 → 参考任一输出必须显著变化）==")
    probes = [("hc_norm", "norm"), ("wDown", "wdown"), ("wUp", "wup"), ("wInj", "winj")]
    if mode != MODE_MIX:
        probes += [("BO", "bo"), ("IJ", "ij")]
    else:
        # mode 0 不消费 pending combine ⇒ BO/IJ 探针不适用（显式计入 SKIPPED，不静默）
        print("[chk] guard C BO / C IJ          mode 0 不消费 pending combine → SKIPPED（unverified）×2")
        SKIPPED[0] += 2
    srcs = {"norm": norm, "wdown": wdown, "wup": wup, "winj": winj, "bo": bo, "ij": ij}
    base = ref
    for label, key in probes:
        arrs = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in srcs.items()}
        a = arrs[key]
        # 输入恒为 0 的档（exact0 的 IJ、zerow 的三个权重）——×1.5 是恒等变换，探针不适用
        if not np.any(f32(a[:min(len(a), 4096)])):
            print("[chk] guard %-19s 输入恒 0 → 探针不适用 → SKIPPED（unverified）" % ("C " + label))
            SKIPPED[0] += 1
            continue
        b16v = f32(a).astype(np.float64) * 1.5
        arrs[key] = b16(b16v).astype(np.uint16)
        alt = reference(m, mode, hin, arrs["bo"], arrs["ij"], arrs["wdown"], arrs["winj"], arrs["wup"], arrs["norm"])
        maxd = 0.0
        for k in ("hc", "xn", "lora", "ls", "gate", "blk"):
            if mode == MODE_MIX and k == "hc":
                continue
            maxd = max(maxd, float(np.max(np.abs(base[k] - alt[k]))))
        GUARD[0] += 1
        ok = maxd > 1e-3
        if not ok:
            FAILS.append("C/" + label)
        print("[chk] guard %-19s Δout(maxAbs over all outputs)=%.6e  %s"
              % ("C " + label, maxd, "sensitive" if ok else "INSENSITIVE"))

    # 设备侧取值多样性（参考非常量时设备也不许常量）
    refmap = {"hcp": ref["hc"], "xn": ref["xn"], "gate": ref["gate"], "blk": ref["blk"]}
    for nm, key in (("h'", "hcp"), ("xn", "xn"), ("gate", "gate"), ("blk", "blk")):
        if mode == MODE_MIX and key == "hcp":
            continue
        devn = int(np.unique(dev[key][:m]).size)
        refn = int(np.unique(b16(refmap[key])).size)
        GUARD[0] += 1
        ok = (devn > 1) or (refn == 1)
        if not ok:
            FAILS.append("C/const-" + nm)
        print("[chk] guard %-19s dev 取值数=%-8d ref 取值数=%-8d  %s"
              % ("C " + nm + " 取值多样性", devn, refn, "OK" if ok else "DEGENERATE"))

    # tower 规则：三态分栏 + 覆盖计数与匹配器同源 + OK 文案带实际比较条数
    compared = JUDGE[0] + GUARD[0]
    print("\n[chk] 覆盖：判定项 %d 条 + guard %d 条（共比较 %d 条）；"
          "unverified/SKIPPED %d 条（见上，逐条打印）"
          % (JUDGE[0], GUARD[0], compared, SKIPPED[0]))
    if FAILS:
        print("[chk] RESULT: FAILED (%d 条判定项不符：%s)" % (len(FAILS), FAILS))
        return 1
    if compared == 0:
        print("[chk] RESULT: SKIPPED (没有可比对的输入)")
        return 2
    print("[chk] RESULT: OK (%d/%d 条比较过：判定项 %d + guard %d；SKIPPED %d)"
          % (compared, compared, JUDGE[0], GUARD[0], SKIPPED[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
