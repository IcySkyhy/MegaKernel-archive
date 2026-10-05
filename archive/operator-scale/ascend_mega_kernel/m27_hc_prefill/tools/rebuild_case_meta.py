#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m27_hc_prefill/tools/rebuild_case_meta.py —— 从**设备写的**布局文件重建 case 元数据（旧格式 dump 的补救）

## 为什么需要
元数据格式在 10-04 从"全局 `m27_cases.txt`"改成"逐 case 一份 `m27_case_<名>.txt`"。在那之前跑的档
（含 10-04 的 m=4097 两档）只留下**旧格式**，而 `check_ref.py` 按新格式 glob ⇒ 那份 dump 的真 shape
判定**判不了**（复审 r1 的 P2-2）。

## 它做什么（**不发明任何读数**）
`m27_layout_<case>.txt` 是**同一次设备运行**写出来的（`m15_hc_prefill_host.h::DumpLayout`），
里面有 `m` / `mode` / `mt` / `tiles` / `arena_bytes` / `arena_dumped` / `chain` —— 元数据所需的
case 相关字段**全部**来自它；其余三个字段是**该运行的已知事实**（`real_w` / `mutant` / `reloc_mask`，
由命令行决定、由本工具的选项显式给出，默认值 = 常规档：真实权重 / mutant=none / reloc_mask=15）。
`ub_peak_pf` 是编译期常量（`M15H::HcPF::UB_PEAK_PF`）。

⇒ 本工具**不改任何张量**，只把"这次运行的元数据"补写出来；每条字段的来源都在 `--verbose` 里逐行打印。
**它不能补救**：该批次的**二进制 sha**（若当时未归档，事后无法恢复）—— 那一条仍按 §4.6 的缺口登记。

用法：
    # 先看它打算写什么（不落盘）
    python3.12 m27_hc_prefill/tools/rebuild_case_meta.py /tmp/m27_out m4097 --dry-run --verbose
    # 真写（只在 m27_case_m4097.txt 不存在时写；存在则默认跳过，加 --force 覆盖）
    python3.12 m27_hc_prefill/tools/rebuild_case_meta.py /tmp/m27_out m4097 m4097_mix
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

UB_PEAK_PF = 94592          # = M15H::HcPf::UB_PEAK_PF（编译期常量）


def read_kv(path):
    kv = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            for pair in line.split():
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    kv[k] = v
    return kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--real-w", type=int, default=1, help="该运行的权重来源（1=真实 checkpoint / 0=合成）")
    ap.add_argument("--mutant", default="none", help="该运行的 M27_MUTANT（缺省 none）")
    ap.add_argument("--reloc-mask", type=int, default=15,
                    help="该运行的重定位目标位掩码（bit0=blk/1=injw/2=rstd/3=ijFlat；常规档 = 15）")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    wrote = 0
    for case in a.cases:
        layp = os.path.join(a.dump, "m27_layout_%s.txt" % case)
        outp = os.path.join(a.dump, "m27_case_%s.txt" % case)
        if not os.path.exists(layp):
            print("[meta] SKIP %s：缺 %s" % (case, layp))
            continue
        if os.path.exists(outp) and not a.force:
            print("[meta] SKIP %s：%s 已存在（要覆盖加 --force）" % (case, outp))
            continue
        L = read_kv(layp)
        need = ("m", "mode", "mt", "tiles", "arena_bytes")
        if any(k not in L for k in need):
            print("[meta] SKIP %s：布局文件缺字段（需要 %s）" % (case, ", ".join(need)))
            continue
        arena_dumped = L.get("arena_dumped", "0")
        chain = L.get("chain", "0")
        line = ("case %s m %s mode %s mt %s tiles %s arena_bytes %s real_w %d chain %s mutant %s "
                "ub_peak_pf %u arena_dumped %s reloc_mask %u\n"
                % (case, L["m"], L["mode"], L["mt"], L["tiles"], L["arena_bytes"], a.real_w, chain,
                   a.mutant, UB_PEAK_PF, arena_dumped, a.reloc_mask))
        if a.verbose:
            print("[meta] %s ← 字段来源：m/mode/mt/tiles/arena_bytes/arena_dumped/chain = %s（设备写的布局文件）；"
                  "real_w=%d / mutant=%s / reloc_mask=%u = 本次调用的命令行事实；ub_peak_pf=%u = 编译期常量"
                  % (case, os.path.basename(layp), a.real_w, a.mutant, a.reloc_mask, UB_PEAK_PF))
        if a.dry_run:
            print("[meta] %s（dry-run，未落盘）：%s" % (case, line.strip()))
            continue
        with open(outp, "w") as f:
            f.write("# 由 tools/rebuild_case_meta.py 从设备写的 %s 重建（旧格式 dump 的补救）\n"
                    % os.path.basename(layp))
            f.write(line)
        print("[meta] 写出 %s：%s" % (outp, line.strip()))
        wrote += 1
    print("[meta] 共处理 %d 个 case（写出 %d 份）" % (len(a.cases), wrote))
    return 0


if __name__ == "__main__":
    sys.exit(main())
