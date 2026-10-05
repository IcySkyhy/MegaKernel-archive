#!/usr/bin/env python3
"""probe_sync_quirks/tools/decode_mrgsort_dump.py

独立（numpy 之外零依赖）解析 probe_a_mrgsort4 落盘的 UB 快照，产出人读证据：

  * evidence/decode_A_summary.txt          —— 每个变体的判读摘要（与 kernel host 侧独立复算）
  * evidence/dumps_txt/variant_NN_<name>.txt —— dst 区逐对（value / index 分列）全量表格

UB 布局（与 probe_a_mrgsort4.asc 头部注释一致，字节偏移）：
    0     : 4 个输入块（各 32 对 @256B；对 = {fp32 value, u32 index}）
    1024  : dst     (2048B, 4 路输出 1024B + 1024B 观察窗)
    3072  : dst2    (1024B)
    4096  : tmp     (1024B, Sort32 输出 / A13 的 VMS4_SR 落点)
    5120  : s32 输入(vals fp32[128] + ids u32[128])
每变体 dump 6144B。

输入对契约：块 b、组内序 j -> value = 1000 - 4j - b, index = 1000b + j。
预置：sentinel = (0xDEADBEEF, 0xBEEFDEAD)；A03 为全 0；A04 不预置。

用法：
    python3 decode_mrgsort_dump.py <dump_dir> [<out_dir>]
        <dump_dir> 内含 input_vfirst.bin / variant_NN_<name>.bin
"""
import os
import struct
import sys

PAIRS, PAIR_B, BLK_B, NBLK = 32, 8, 256, 4
UB_IN, UB_DST, UB_DST_SPAN = 0, 1024, 2048
UB_DST2 = UB_DST + UB_DST_SPAN          # 3072
UB_TMP = UB_DST2 + 1024                 # 4096
UB_S32I = UB_TMP + 1024                 # 5120
UB_END = UB_S32I + 1024                 # 6144
SENT_VAL, SENT_IDX = 0xDEADBEEF, 0xBEEFDEAD

# 变体表（镜像 probe_a_mrgsort4.asc 的 kVariants；只保留判读所需字段）
#   dst   : 判读区起始字节
#   src   : src1..4 的输入块号
#   lens  : elementLengths
#   kind  : "merge" = 期望按值降序的全序；"concat" = 4 组各自有序的拼接；
#           "none"  = 语义不明/别名/重复次数异常，只取证 + 与规范 4 路归并做参考对比
VARIANTS = [
    ("A00_2way_v3_base",        UB_DST,                   [0, 1, 1, 1], [32, 32, 0, 0],   "merge"),
    ("A01_2way_tree_v3",        UB_DST2,                  [0, 1, 2, 3], [32, 32, 32, 32], "merge"),
    ("A02_4way_v15",            UB_DST,                   [0, 1, 2, 3], [32, 32, 32, 32], "merge"),
    ("A03_4way_v15_zero",       UB_DST,                   [0, 1, 2, 3], [32, 32, 32, 32], "merge"),
    ("A04_4way_v15_nopreset",   UB_DST,                   [0, 1, 2, 3], [32, 32, 32, 32], "merge"),
    ("A05_4way_v15_dst2",       UB_DST2,                  [0, 1, 2, 3], [32, 32, 32, 32], "merge"),
    ("A06_3way_v7",             UB_DST,                   [0, 1, 2, 2], [32, 32, 32, 0],  "merge"),
    ("A07_4way_v15_ph34_l0",    UB_DST,                   [0, 1, 1, 1], [32, 32, 0, 0],   "merge"),
    ("A08_4way_v15_ph34_l32",   UB_DST,                   [0, 1, 1, 1], [32, 32, 32, 32], "merge"),
    ("A09_4way_v15_noncontig",  UB_DST,                   [0, 2, 1, 3], [32, 32, 32, 32], "merge"),
    ("A10_2way_v3_len16",       UB_DST,                   [0, 1, 1, 1], [16, 16, 0, 0],   "merge"),
    ("A11_4way_v15_len16",      UB_DST,                   [0, 1, 2, 3], [16, 16, 16, 16], "merge"),
    ("A12_4way_v15_rep2",       UB_DST,                   [0, 1, 2, 3], [32, 32, 32, 32], "none"),
    ("A13_4way_v15_exhtrue",    UB_DST,                   [0, 1, 2, 3], [32, 32, 32, 32], "merge"),
    ("A14_sort32_layout",       UB_TMP,                   [0, 1, 2, 3], [32, 32, 32, 32], "concat"),
    ("A15_4way_v15_sort32src",  UB_DST,                   [0, 1, 2, 3], [32, 32, 32, 32], "merge"),
    ("A16_4way_v15_allb0",      UB_DST,                   [0, 0, 0, 0], [32, 32, 32, 32], "merge"),
    ("A17_4way_v15_len4",       UB_DST,                   [0, 1, 2, 3], [4, 4, 4, 4],     "merge"),
    ("A18_dst_alias_src1",      UB_IN,                    [0, 1, 2, 3], [32, 32, 32, 32], "none"),
    ("A19_dst_alias_src3",      UB_IN + 2 * BLK_B,        [0, 1, 2, 3], [32, 32, 32, 32], "none"),
    ("A20_sort32_nobar",        UB_DST,                   [0, 1, 2, 3], [32, 32, 32, 32], "merge"),
    ("A21_4way_v15_rep0",       UB_DST,                   [0, 1, 2, 3], [32, 32, 32, 32], "none"),
    ("A22_2way_v3_len4x32",     UB_DST,                   [0, 1, 2, 3], [32, 32, 32, 32], "none"),
    ("A23_4way_v15_len0_all",   UB_DST,                   [0, 1, 2, 3], [0, 0, 0, 0],     "none"),
    ("A24_4way_v15_rep2_len8",  UB_DST,                   [0, 1, 2, 3], [8, 8, 8, 8],     "none"),
    ("A25_4way_v15_len8",       UB_DST,                   [0, 1, 2, 3], [8, 8, 8, 8],     "merge"),
]


def pair_value(b, j):
    return 1000.0 - 4.0 * j - float(b)


def pair_index(b, j):
    return 1000 * b + j


def canonical_merge(nblk=4, npairs=PAIRS):
    """4 路全量归并的规范结果（按 value 降序；value 全局唯一）。"""
    ps = [(pair_value(b, j), pair_index(b, j)) for b in range(nblk) for j in range(npairs)]
    ps.sort(key=lambda t: -t[0])
    return ps


def expected_for(src, lens, kind):
    if kind == "none":
        return None
    out = []
    if kind == "concat":
        for b in range(4):
            for j in range(PAIRS):
                out.append((pair_value(b, j), pair_index(b, j)))
        return out
    for i in range(4):
        b = src[i]
        for j in range(lens[i]):
            out.append((pair_value(b, j), pair_index(b, j)))
    out.sort(key=lambda t: -t[0])
    return out


def words(buf, off, n):
    return list(struct.unpack_from("<%dI" % n, buf, off))


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def decode_region(buf, off, npairs):
    ws = words(buf, off, npairs * 2)
    return [(ws[2 * i], ws[2 * i + 1]) for i in range(npairs)]


def tag_pair(w0, w1):
    """把 (value 字, index 字) 翻译成"来自哪一块哪一项"，并标记 sentinel/全 0/非法。"""
    if w0 == SENT_VAL and w1 == SENT_IDX:
        return "  <sentinel 未写>"
    if w0 == 0 and w1 == 0:
        return "  <0 未写>"
    v = f32(w0)
    b, j = w1 // 1000, w1 % 1000
    if b < 4 and j < PAIRS and v == pair_value(b, j):
        return "  b%d j%-2d (value/index 自洽)" % (b, j)
    # value 合法但 index 对不上 -> index 串路/丢写
    for bb in range(4):
        for jj in range(PAIRS):
            if v == pair_value(bb, jj):
                return "  value=b%dj%-2d 但 index=%-10u <-- index 与 value 不配对" % (bb, jj, w1)
    return "  非法: value=%r(0x%08X) index=%u(0x%08X)" % (v, w0, w1, w1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    dump_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    ev_dir = os.path.join(out_dir, "..", "evidence")
    ev_dir = os.path.normpath(ev_dir)
    txt_dir = os.path.join(ev_dir, "dumps_txt")
    os.makedirs(txt_dir, exist_ok=True)

    canon = canonical_merge()
    summary = []
    summary.append("probe A（MrgSort 变体矩阵）UB dump 逐元素判读 —— 由 tools/decode_mrgsort_dump.py 生成")
    summary.append("对 = {fp32 value, u32 index}；输入块 b/j -> value=1000-4j-b, index=1000b+j")
    summary.append("")

    for idx, (name, dst, src, lens, kind) in enumerate(VARIANTS):
        path = os.path.join(dump_dir, "variant_%02d_%s.bin" % (idx, name))
        if not os.path.exists(path):
            summary.append("[A%02d] %-24s 缺 dump 文件：%s" % (idx, name, path))
            continue
        with open(path, "rb") as f:
            buf = f.read()
        if len(buf) != UB_END:
            summary.append("[A%02d] %-24s dump 长度异常 %d（期望 %d）" % (idx, name, len(buf), UB_END))

        max_pairs = (UB_DST_SPAN // PAIR_B) if dst == UB_DST else (1024 // PAIR_B)
        pairs = decode_region(buf, dst, max_pairs)
        exp = expected_for(src, lens, kind)

        n_sent = sum(1 for w0, w1 in pairs if (w0, w1) == (SENT_VAL, SENT_IDX))
        n_zero = sum(1 for w0, w1 in pairs if (w0, w1) == (0, 0))
        n_legal = n_ill = n_written = 0
        for w0, w1 in pairs:
            if (w0, w1) in ((SENT_VAL, SENT_IDX), (0, 0)):
                continue
            n_written += 1
            t = tag_pair(w0, w1)
            if "自洽" in t:
                n_legal += 1
            else:
                n_ill += 1

        # 与期望逐项比对（期望长度内）
        first_bad = None
        if exp is not None:
            for i, e in enumerate(exp):
                if i >= len(pairs):
                    break
                w0, w1 = pairs[i]
                if f32(w0) != e[0] or w1 != e[1]:
                    first_bad = i
                    break
            verdict = "PASS（%d 对全序与期望逐项一致）" % len(exp) if first_bad is None else \
                "FAIL 首个不符 pos=%d" % first_bad
        else:
            # 只取证：仍与规范 4 路归并做参考对比
            diff = [i for i in range(min(len(canon), len(pairs)))
                    if (f32(pairs[i][0]), pairs[i][1]) != canon[i]]
            verdict = "（不判序；与规范 4 路归并逐项一致的位点 = %d/%d%s）" % (
                min(len(canon), len(pairs)) - len(diff), min(len(canon), len(pairs)),
                "" if not diff else "，首个不符 pos=%d" % diff[0])

        summary.append("[A%02d] %-24s dst@%-5d  写出=%3d（自洽=%3d 非法=%3d）sentinel=%3d 零=%3d  %s"
                       % (idx, name, dst, n_written, n_legal, n_ill, n_sent, n_zero, verdict))
        if kind == "none" and pairs and canon:
            summary.append("        参考对比：pos0 得到 %s" % tag_pair(*pairs[0]))
        if idx == 13:
            reg = words(buf, UB_TMP, 4)
            summary.append("        VMS4_SR（4 个队列已完成元素数）= %s" % reg)

        # 逐元素表格落盘
        with open(os.path.join(txt_dir, "variant_%02d_%s.txt" % (idx, name)), "w") as f:
            f.write("# %s  dst 区 @ %d 字节，%d 对（value / index 分列）\n" % (name, dst, max_pairs))
            f.write("# preset/期望：%s\n" % ("concat(4 组各自有序)" if kind == "concat"
                                             else ("none(仅取证)" if kind == "none" else "merge(值降序全序)")))
            f.write("#  pos   word0(f32 value)      word1(u32 index)   标注 / 期望\n")
            for i, (w0, w1) in enumerate(pairs):
                e = ""
                if exp is not None and i < len(exp):
                    ok = (f32(w0) == exp[i][0] and w1 == exp[i][1])
                    e = "期望(%g, %u)%s" % (exp[i][0], exp[i][1], "" if ok else "  <== 不符")
                f.write("%5d  0x%08X %-14.6g 0x%08X %-10u %s %s\n"
                        % (i, w0, f32(w0), w1, w1, tag_pair(w0, w1), e))
            if idx == 14:
                f.write("\n# s32 输入区（Sort32 前，组内乱序）@ %d\n" % UB_S32I)
                sv = words(buf, UB_S32I, PAIRS * NBLK)
                si = words(buf, UB_S32I + PAIRS * NBLK * 4, PAIRS * NBLK)
                for i in range(8):
                    f.write("  vals[%2d]=%-12.6g ids[%2d]=%u\n" % (i, f32(sv[i]), i, si[i]))
            if idx == 13:
                f.write("\n# VMS4_SR @ %d：四个队列的已消费元素数 = %s\n" % (UB_TMP, words(buf, UB_TMP, 4)))

    with open(os.path.join(ev_dir, "decode_A_summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
