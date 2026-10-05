#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m27_hc_prefill/tools/ulp_envelope.py —— P2-3 的**举证工具**：`ulp ≤ 2` 在这条链深上可达吗？

## 问题（复审 r1 的 P2-3）
未被 6 条判定项**全部是 `blk`**，且三个子门限里只有「良态 bf16 ulp ≤2」越界
（ulpMax 4–9、良态逐位率 0.9883–0.9905、归一化误差 ≤6.6e-03 仍在门限内）。
塔要求**靠证据裁**、不许直接改松门限 ⇒ 需要回答：**在这条链深上，`ulp ≤ 2` 是不是本来就不该期待？**

## 做法（**零设备**，只用参考侧做数值实验）
拿设备档的**同一批输入**，算两条参考链：

    A = `m20_hyperconn/check_ref.py::reference`  —— 全部在 **fp64** 里算，只在文档写明的落盘点
        取 bf16（= 被复用门限所依据的那条参考）；
    B = **同一批落盘点**，但把两处 GEMM 的累加换成 **fp32 的 K-分块累加**（每 64 元素一块，
        块内 fp32、块间 fp32 相加 —— 对应设备的 `Mmad(k=64)` 打进 fp32 L0C 的形态），
        其余逐元素段（norm / silu / sigmoid / mix）也用 fp32 —— 即"**任何**在这批落盘点
        取整的实现"能给出的下界。

然后按**与判据同口径**统计 `A` vs `B` 的（逐位率 / 良态 ulpMax / 归一化误差），并逐段给出。
若 `B` 相对 `A` 就已经越"ulp ≤2"，则说明**该门限在这条链深 + 这批输入分布上不是"实现质量"的判据**
（任何按同一批落盘点取整的实现都会撞到它）⇒ 支持"门限口径需登记变更"；
若 `B` 相对 `A` 在门限内，则说明该门限**可达**，余差不能归因于链深 ⇒ 不支持口径变更。

⚠ 口径（不越界）：`B` 是**我方的 fp32 累加仿真**，不是设备的真实累加序（设备的 L1/L0 分块、
Fixpipe 舍入细节未仿真）⇒ 它给的是**量级与形态**的证据（是不是"这条链深必然发生的量级"），
不是"设备本可以做到 X"的断言。设备侧仍受"错误报告未归档 ⇒ 不能排除设备异常"这条限度约束。

用法：
    python3.12 m27_hc_prefill/tools/ulp_envelope.py /tmp/m27_out m33 a1
    python3.12 m27_hc_prefill/tools/ulp_envelope.py /tmp/m27_out m33 a2
退出码：0 = 出读数；2 = 缺输入 / 档不合适（m 太大或 mode 不适用）。
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "m20_hyperconn"))
import check_ref as m20   # noqa: E402

HC, HID, HYPER, LOWRANK, INJ_N = 4, 2560, 10240, 320, 4
EPS = 1e-6
BASE_K = 64          # = donor 的 BASE_K（m15_hc_resources.h）
MODE_MIX, MODE_COMBINE_MIX, MODE_COMBINE_ONLY = 0, 1, 3


def load_checker():
    path = os.path.join(REPO, "m27_hc_prefill", "check_ref.py")
    spec = importlib.util.spec_from_file_location("c27env", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["c27env"] = mod
    spec.loader.exec_module(mod)
    return mod


def f32b(x):
    """fp32 里算完再取 bf16（返回 bf16 位模式）—— 仿真"落盘点"。"""
    return m20.b16(np.asarray(x, dtype=np.float32))


def gemm_f32kblock(a_bf16, w_bf16):
    """C = A·Wᵀ，**fp32 的 K 分块累加**（块长 BASE_K，块内 fp32、块间 fp32 相加）。

    A: (m, K) bf16 位模式；W: (N, K) bf16 位模式；返回 (m, N) fp32。
    形态对准设备的 `Mmad(k=BASE_K)` 打进 fp32 L0C：每个 k 块的乘加在 fp32 里完成，
    跨块在 fp32 里累加（不升到 fp64）。
    """
    a = m20.f32(a_bf16).astype(np.float32)
    w = m20.f32(w_bf16).astype(np.float32)
    k = a.shape[1]
    acc = np.zeros((a.shape[0], w.shape[0]), dtype=np.float32)
    for kb in range(0, k, BASE_K):
        acc += np.matmul(a[:, kb:kb + BASE_K], w[:, kb:kb + BASE_K].T, dtype=np.float32)
    return acc


def chain_f32_emulated(hin, bo, ij, wdown, winj, wup, norm, mode):
    """仿真链 B：落盘点与 `m20.reference` 相同，但累加/逐元素全在 fp32。"""
    ijv = m20.f32(ij[:, :HC]).astype(np.float32)
    injw = (2.0 / (1.0 + np.exp(-(ijv / float(HC))))).astype(np.float32)
    h = m20.f32(hin).astype(np.float32).reshape(hin.shape[0], HC, HID)
    bo_f = m20.f32(bo).astype(np.float32)
    if mode == MODE_MIX:
        hc = h.reshape(hin.shape[0], HYPER)
    else:
        hc = f32b((h + bo_f[:, None, :] * injw[:, :, None]).astype(np.float32)).reshape(hin.shape[0], HYPER)
    # S2：组内 GemmaRMSNorm（两遍；pass A 先把 bf16 舍入后的值平方，与参考/设备同口径）
    x = m20.f32(hc).astype(np.float32).reshape(hin.shape[0], HC, HID)
    var = np.sum(x * x, axis=2, dtype=np.float32) / np.float32(HID)
    rrms = (np.float32(1.0) / np.sqrt(var + np.float32(EPS))).astype(np.float32)
    wnorm = m20.f32(norm).astype(np.float32).reshape(HC, HID)
    xn = f32b(((x * rrms[:, :, None]) * (np.float32(1.0) + wnorm[None, :, :])).astype(np.float32))
    xn = xn.reshape(hin.shape[0], HYPER)
    # S3：down + inject（K=HYPER 分块 fp32 累加）
    acc_dn = gemm_f32kblock(xn, wdown)
    acc_in = gemm_f32kblock(xn, winj[:INJ_N])
    lora = np.empty((hin.shape[0], LOWRANK + INJ_N), dtype=np.uint16)
    lora[:, :LOWRANK] = f32b(acc_dn)
    lora[:, LOWRANK:] = f32b(acc_in)
    # S4：silu(lora/HC)
    u = m20.f32(lora[:, :LOWRANK]).astype(np.float32) / np.float32(HC)
    ls = f32b((u / (np.float32(1.0) + np.exp(-u))).astype(np.float32))
    # S5：up（K=LOWRANK 分块 fp32 累加）
    gate = f32b(gemm_f32kblock(ls, wup))
    # S6：gate mix
    g = m20.f32(gate).astype(np.float32).reshape(hin.shape[0], HC, HID)
    xx = m20.f32(xn).astype(np.float32).reshape(hin.shape[0], HC, HID)
    sacc = np.zeros((hin.shape[0], HID), dtype=np.float32)
    for s in range(HC):
        sacc += ((np.float32(1.0) / (np.float32(1.0) + np.exp(-g[:, s, :]))) * xx[:, s, :]).astype(np.float32)
    blk = f32b((sacc / np.float32(HC)).astype(np.float32))
    return {"hc": hc, "xn": xn, "lora": lora, "ls": ls, "gate": gate, "blk": blk.reshape(hin.shape[0], HID)}


def report(name, got_u16, exp_f64):
    got_b = np.asarray(got_u16, dtype=np.uint16).reshape(-1)
    exp = np.asarray(exp_f64, dtype=np.float64).reshape(-1)
    exp_b = m20.b16(exp).reshape(-1)
    d = np.abs(m20.ulp_key(got_b).astype(np.int64) - m20.ulp_key(exp_b).astype(np.int64))
    scale = float(np.max(np.abs(exp))) if exp.size else 0.0
    sig = np.abs(exp) > 0.05 * scale
    got_f = m20.f32(got_b).astype(np.float64)
    norm_abs = float(np.max(np.abs(got_f - exp)) / scale) if scale > 0 else 0.0
    frac_all = float(np.mean(d == 0))
    frac_sig = float(np.mean(d[sig] == 0)) if np.any(sig) else 1.0
    maxulp = int(np.max(d[sig])) if np.any(sig) else 0
    over2 = float(np.mean(d[sig] > 2)) if np.any(sig) else 0.0
    print("[env] %-8s 逐位=%.4f 良态逐位=%.4f ulpMax(良态)=%-3d 良态里 ulp>2 的占比=%.4f 归一maxAbs=%.3e"
          % (name, frac_all, frac_sig, maxulp, over2, norm_abs))
    return dict(frac_all=frac_all, frac_sig=frac_sig, maxulp=maxulp, over2=over2, norm_abs=norm_abs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("case")
    ap.add_argument("tag", choices=["a1", "a2"])
    a = ap.parse_args()
    c27 = load_checker()
    D, case, tag = a.dump, a.case, a.tag
    try:
        cases = c27.read_cases(D)          # 逐 case 元数据（格式 `case <名> <key> <val> …`）
    except m20.MissingInput as e:
        print("[env] SKIPPED（缺 %s）" % e)
        return 2
    meta = next((c for c in cases if c["case"] == case), None)
    if meta is None:
        print("[env] SKIPPED（dump 里没有 case %s）" % case)
        return 2
    m = int(meta["m"])
    mode = int(meta["mode"])
    if mode in (MODE_MIX, MODE_COMBINE_ONLY):
        print("[env] SKIPPED（mode %d 没有可比的分段链）" % mode)
        return 2
    if m > 64:
        print("[env] SKIPPED（只对小档做，m=%d > 64；分段链的中间量在大档读不到）" % m)
        return 2
    try:
        hin_file = "%s_a1_hin" % case if tag == "a1" else "%s_a2_hin" % case
        ij_file = "%s_a1_ij" % case if tag == "a1" else "%s_a1_ij_handoff" % case
        wtag = "a1" if tag == "a1" else "a2"
        hin = c27.rd(D, hin_file, np.uint16)[:m]
        bo = c27.rd(D, "%s_%s_bo" % (case, tag), np.uint16)[:m]
        ij = c27.rd(D, ij_file, np.uint16)
        wdown = c27.rd(D, "w_%s_down" % wtag, np.uint16)
        winj = c27.rd(D, "w_%s_inj" % wtag, np.uint16)
        wup = c27.rd(D, "w_%s_up" % wtag, np.uint16)
        norm = c27.rd(D, "w_%s_norm" % wtag, np.uint16).reshape(-1)
        blk_dev = c27.rd(D, "%s_%s_blk" % (case, tag), np.uint16)[:m]
    except m20.MissingInput as e:
        print("[env] SKIPPED（缺 %s）" % e)
        return 2
    print("== ulp 可达性实验（零设备）：case=%s tag=%s m=%d mode=%d；A=fp64 参考，B=同落盘点+fp32 累加 ==" %
          (case, tag, m, mode))
    ref = m20.reference(m, mode, hin, bo, ij, wdown, winj, wup, norm)
    emu = chain_f32_emulated(hin, bo, ij, wdown, winj, wup, norm, mode)
    print("-- 参考侧：任何在这批落盘点取整的实现（B）相对 fp64 参考（A）--")
    for nm in ("xn", "lora", "ls", "gate", "blk"):
        if nm == "lora":
            report(nm, emu["lora"], ref["lora"] if ref["lora"].shape[1] == emu["lora"].shape[1]
                   else ref["lora"])
        else:
            report(nm, emu[nm], ref["hc" if nm == "hc" else nm])
    print("-- 对照：设备（D）相对同一条 fp64 参考（A）——同一批判据，读数取自 evidence/check_main*.log --")
    report("blk(D)", blk_dev, ref["blk"])
    print("[env] 读法：把 `blk` 的 A-vs-B 读数与上面 `blk(D)` 那行并排看 —— 若两者同量级（ulpMax 同阶、"
          "良态逐位率同为 ~0.99），则「ulp ≤ 2」在这条链深 + 这批输入分布上不是「实现质量」的判据；"
          "若 B 明显更好，则该门限在此可达，余差不能归因于链深。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
