#!/usr/bin/env python3
"""probe_vf_ldst/tools/ub_regions.py —— 判定项 dump 跨进程一致性的**分区**证据。

为什么需要它（README §5 / §8 的口径）：探针 B 的 `b_*_ub.bin` 是**整段 UB 现场快照**，
里面既有判定项消费的区（槽区、输出区），也有内核在某个变体里**从不写**的 scratch
（`UB_MID` 256B、`UB_BACKLOG` 16KB）。后者是 UB 残留 ⇒ 跨进程内容随机 ⇒ 整文件 sha256 不稳定。
本工具**按区**比对，把「判定项 dump 逐字节一致」与「未写 scratch 不一致」分开报，
免得把前者写成无限定的"dump 逐字节一致"（这正是 review P2-1 指出的问题）。

退出码（三态）：0 = 分完区且判定项区全部一致；1 = 有判定项区跨进程不一致；2 = 没得比。

用法：python3 tools/ub_regions.py <evidence_dir> <procs> <A_MASKS_CONTIG> <A_MASKS_NC> <B_TARGETS> <C_TARGETS>
"""

import hashlib
import os
import sys

EV = sys.argv[1] if len(sys.argv) > 1 else "evidence"
PROCS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
A_CONTIG = (sys.argv[3].split() if len(sys.argv) > 3 else "0 1 8 64".split())
A_NC = (sys.argv[4].split() if len(sys.argv) > 4 else "0 1 2 3".split())
B_TARGETS = sys.argv[5].split() if len(sys.argv) > 5 else []
C_TARGETS = sys.argv[6].split() if len(sys.argv) > 6 else []
DUMPS = os.path.join(EV, "dumps")

# 探针 B 的 UB 布局（与 probe_b_ldst_handover.asc 一致）
B_REGIONS = [
    ("slot  (判定项)", 0, 16384),
    ("out   (判定项)", 16384, 32768),
    ("MID scratch (未写残留)", 32768, 33024),
    ("BACKLOG scratch (未写残留)", 33024, 49408),
]
# 探针 C 的 UB 布局（与 probe_c_gm_roundtrip.asc 一致）：UB_TAB / UB_SLOT / UB_OUT
C_REGIONS = [
    ("tab   (判定项)", 0, 16384),
    ("slot  (判定项)", 16384, 16640),
    ("out   (判定项)", 16640, 33024),
]


def read(path):
    with open(path, "rb") as f:
        return f.read()


def region_hashes(paths, start, end):
    hs = set()
    for p in paths:
        hs.add(hashlib.sha256(read(p)[start:end]).hexdigest()[:12])
    return hs


def check(kind, targets, regions, suffix):
    """返回 (判定项区不一致的条目数, 输出的行)"""
    lines = []
    bad = 0
    for t in targets:
        paths = []
        for k in range(PROCS):
            fp = os.path.join(DUMPS, "%s_%s_p%d_%s" % (kind, t, k, suffix))
            if os.path.isfile(fp):
                paths.append(fp)
        if len(paths) < 2:
            lines.append("   %-30s SKIP（只有 %d 份 dump）" % (t, len(paths)))
            continue
        for name, s, e in regions:
            hs = region_hashes(paths, s, e)
            verdict = "一致" if len(hs) == 1 else "不一致(%d 种)" % len(hs)
            judge = ("判定项" in name)
            mark = ""
            if judge and len(hs) != 1:
                mark = "  <== 判定项区跨进程不一致！"
                bad += 1
            lines.append("   %-30s %-28s [%6d,%6d)  %s%s" %
                         (t, name, s, e, verdict, mark))
    return bad, lines


def main():
    if not os.path.isdir(DUMPS):
        print("RESULT: SKIPPED（没有 dumps 目录 —— 没得比）")
        return 2
    print("=" * 92)
    print("判定项 dump 的跨进程一致性（分区；每变体 %d 个独立进程）" % PROCS)
    print("=" * 92)
    print("\n## 探针 A —— 256B 落盘快照（单区，全量初始化 ⇒ 应整体一致）")
    a_bad = 0
    n_a = 0
    for t in ("probe_a_first_elem", "probe_a_norm_b32"):
        for m in A_CONTIG:
            paths = [os.path.join(DUMPS, "a_%s_mask%s_p%d.bin" % (t, m, k)) for k in range(PROCS)]
            paths = [p for p in paths if os.path.isfile(p)]
            if len(paths) < 2:
                continue
            n_a += 1
            hs = region_hashes(paths, 0, 256)
            if len(hs) != 1:
                a_bad += 1
            print("   %-40s mask=%-3s %s" % (t, m, "一致" if len(hs) == 1 else "不一致! %d 种" % len(hs)))
    for t in ("probe_a_first_elem_nc", "probe_a_norm_b32_nc"):
        for m in A_NC:
            paths = [os.path.join(DUMPS, "a_%s_mask%s_p%d.bin" % (t, m, k)) for k in range(PROCS)]
            paths = [p for p in paths if os.path.isfile(p)]
            if len(paths) < 2:
                continue
            n_a += 1
            hs = region_hashes(paths, 0, 256)
            if len(hs) != 1:
                a_bad += 1
            print("   %-40s mask=%-3s %s" % (t, m, "一致" if len(hs) == 1 else "不一致! %d 种" % len(hs)))

    print("\n## 探针 B —— 整段 UB 快照（按区分栏）")
    b_bad, b_lines = check("b", B_TARGETS, B_REGIONS, "ub.bin")
    for ln in b_lines:
        print(ln)
    print("\n## 探针 B —— 输出缓冲 dump（判定项，单区）")
    b_out_bad, b_out_lines = check("b", B_TARGETS, [("out (判定项)", 0, 16384)], "out.bin")
    for ln in b_out_lines:
        print(ln)

    print("\n## 探针 C —— 整段 UB 快照（按区分栏）")
    c_bad, c_lines = check("c", C_TARGETS, C_REGIONS, "ub.bin")
    for ln in c_lines:
        print(ln)
    print("\n## 探针 C —— 输出缓冲 dump（判定项，单区）")
    c_out_bad, c_out_lines = check("c", C_TARGETS, [("out (判定项)", 0, 16384)], "out.bin")
    for ln in c_out_lines:
        print(ln)

    total_bad = a_bad + b_bad + b_out_bad + c_bad + c_out_bad
    print("\n## 小结")
    print("   探针 A 落盘快照：%d 个配置，跨进程不一致 %d 个" % (n_a, a_bad))
    print("   探针 B 判定项区（slot/out）：跨进程不一致 %d 处" % (b_bad + b_out_bad))
    print("   探针 C 判定项区（tab/slot/out）：跨进程不一致 %d 处" % (c_bad + c_out_bad))
    print("   ※ 探针 B 的 MID/BACKLOG scratch 在内核未写它们的变体里是 UB 残留，")
    print("     **设计上不做一致性要求**，它们的'不一致'不计入上面的判定项计数。")
    if total_bad == 0:
        print("\nRESULT: OK（判定项区跨进程全部逐字节一致）")
        return 0
    print("\nRESULT: DIFF（有 %d 处判定项区跨进程不一致）" % total_bad)
    return 1


if __name__ == "__main__":
    sys.exit(main())
