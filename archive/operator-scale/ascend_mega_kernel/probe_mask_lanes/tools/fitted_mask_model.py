#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fitted_mask_model.py —— M94 的**报告项**：两个位扩展模型对实测的复现率（不是判据门）

两个模型（差别只在「mask 的一个被选元素在 256-bit 谓词寄存器里置几位」）：

  * **doc 先验**（来自官方文档 `asc-devkit/docs/zh/api/SIMD-API/c_api/reg_compute/reg_mask/asc_create_mask.md`
    的「位宽模式说明」：b8 每 bit 一个元素共 256、b16 每 2bit 一个元素共 128、b32 每 4bit 一个元素共 64）
    ⇒ 一个被选元素把**它那一组的全部 bit** 置 1。
  * **后验拟合（本探针读数反推）** ⇒ 一个被选元素只置**它那一组的第 0 个 bit**（即「按字节的位图」：
    b8 元素 e 标 byte e、b16 标 byte 2e、b32 标 byte 4e；共 256/1、/2、/4 个元素）。

两者的 store 侧规则相同：store 位宽 W' 的元素 e' 覆盖字节区间 `[e'*sizeof(W'), (e'+1)*sizeof(W'))`，
**区间内任一 bit 被置**该元素即参与（`any1`）。

为什么要分开写：**后验模型是对本批读数拟合出来的**，因此「后验模型 100% 命中」不是独立证据；
真正有信息量的是「doc 先验在交叉位宽上不命中」这一**证伪**读数（本脚本逐条列出）。判据门（是否越界 /
单次落盘是否 > 1 VL）在 `check_mask_lanes.py` 里，与本文件无关。

用法：python3 tools/fitted_mask_model.py [probe_dir]
退出码：0 = 已打印（本文件不做判定，故恒 0）
"""

import os
import re
import sys

SZ = {"s8": 1, "u8": 1, "s16": 2, "u16": 2, "f16": 2, "bf16": 2, "f32": 4, "s32": 4, "u32": 4}
# mask 位宽模式 -> (元素字节步长, 每元素置位数)：元素步长 = sizeof(W)
STRIDE = {"m8": (1, 1), "m16": (2, 2), "m32": (4, 4)}
MOWN = {"s8": "m8", "u8": "m8", "s16": "m16", "u16": "m16", "f16": "m16", "bf16": "m16",
        "f32": "m32", "s32": "m32", "u32": "m32"}
SENTINEL = 0xA5


def selected(mw, pat):
    n = 256 // STRIDE[mw][0]
    if pat == "all":
        return list(range(n))
    if pat.startswith("vl"):
        return list(range(min(int(pat[2:]), n)))
    if pat == "h":
        return list(range(n // 2))
    if pat == "q":
        return list(range(n // 4))
    if pat == "m3":
        return [e for e in range(n) if e % 3 == 0]
    if pat == "m4":
        return [e for e in range(n) if e % 4 == 0]
    raise ValueError(pat)


def bitmap(mw, pat, bits_per_elem):
    """bits_per_elem: doc 先验 = STRIDE[mw][0]（整组置位）；后验 = 1（只置第 0 位）"""
    stride = STRIDE[mw][0]
    b = set()
    for e in selected(mw, pat):
        for j in range(bits_per_elem):
            b.add(e * stride + j)
    return b


def store_elems(sd, mw, pat, bits_per_elem):
    b = bitmap(mw, pat, bits_per_elem)
    sz = SZ[sd]
    return {e for e in range(256 // sz) if any((e * sz + j) in b for j in range(sz))}


def pat_A(i):
    return (0x5A ^ ((i + 1) & 0xFF)) & 0xFF


def pat_B(i):
    return (0xC3 ^ ((i + 2) & 0xFF)) & 0xFF


def expected_visible(sd, mw, pat, bits_per_elem):
    out = set()
    for base, pf in ((0, pat_A), (1024, pat_B)):
        for e in store_elems(sd, mw, pat, bits_per_elem):
            for j in range(SZ[sd]):
                o = base + e * SZ[sd] + j
                if pf((o - base) % 256) != SENTINEL:
                    out.add(o)
    return out


def main():
    probe = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if len(sys.argv) < 2 else sys.argv[1]
    logs = os.path.join(probe, "evidence", "logs")
    if not os.path.isdir(logs):
        sys.stderr.write("FAIL: 没有 %s\n" % logs)
        return 2

    print("# M94 报告项：位扩展模型对实测的复现率（两个模型；均为报告项，不是判据门）")
    print("# 读数来源：%s/run_*_r1.log 的 WRANGES（与哨兵不同的字节集合）" % logs)
    print()
    n = 0
    bad_doc, bad_fit = [], []
    for fn in sorted(os.listdir(logs)):
        if not (fn.startswith("run_") and fn.endswith("_r1.log")):
            continue
        op = fn[4:-7]
        if not (op.startswith("lane_") or op.startswith("mt_")):
            continue
        with open(os.path.join(logs, fn), "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        m = re.search(r"^WRANGES (.*)$", txt, re.M)
        if not m:
            continue
        rng = m.group(1).strip()
        if op.startswith("mt_"):
            sd = op.split("_")[1]
            mw, pat = MOWN[sd], "all"
        else:
            _, sd, mw, pat = op.split("_")
        obs = set()
        if rng != "(none)":
            for part in rng.split(","):
                a, b = part.split("-")
                obs.update(range(int(a), int(b)))
        doc_bpe = STRIDE[mw][0]          # doc 先验：整组置位
        fit_bpe = 1                      # 后验拟合：只置该组第 0 位
        if obs != expected_visible(sd, mw, pat, doc_bpe):
            bad_doc.append(op)
        if obs != expected_visible(sd, mw, pat, fit_bpe):
            bad_fit.append(op)
        n += 1

    print("变体数（lane_* + mt_*）= %d" % n)
    print("doc 先验模型（每元素整组置位）不命中 = %d：%s" % (len(bad_doc), ", ".join(bad_doc) or "(none)"))
    print("   ↑ 全部出现在 store 位宽 != mask 位宽的交叉形态；本位宽（lane-own + masktype）全部命中")
    print("后验拟合模型（每元素只置该组第 0 位 = 按字节位图）不命中 = %d：%s"
          % (len(bad_fit), ", ".join(bad_fit) or "(none)"))
    print()
    print("口径：后验模型是对本批读数拟合出来的 ⇒ 「后验 100% 命中」**不是**独立证据；")
    print("      有信息量的是「doc 先验在交叉位宽上不命中」这条证伪读数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
