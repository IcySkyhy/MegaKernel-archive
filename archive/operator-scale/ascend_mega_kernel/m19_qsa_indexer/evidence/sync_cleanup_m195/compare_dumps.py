#!/usr/bin/env python3
"""dump_compare.txt 的生成 / 校验脚本（M195 证据，M199 补入仓）。

背景：`dump_compare.txt` 是 M195 改前 / 改后两个二进制在默认 4 档 dump 上的逐字节
比对报告。原报告由一次性脚本产出、未入仓 ⇒ 第三人无法离线复现。本脚本把该报告的
**生成器**与**期望值**一并落库，并提供离线校验入口。

两件事分开（device / offline）：
  * 纯离线（不需要设备）：校验已入仓的 `dump_compare.txt` 与本脚本内置的期望
    **逐字节一致**，并复核报告内部的结构性断言（非 stats 全同、DIFF 只落在
    `*_stats.bin`、stats 差异字段恰为 23..31 且 0..22 全同）。
  * 需设备（需要 NPU 0 + 两个二进制各跑一轮 dump）：从 before/after 两个 dump
    目录重新生成报告并与期望比对。

用法：
    # 纯离线：校验已入仓报告（默认动作）
    python3 m19_qsa_indexer/evidence/sync_cleanup_m195/compare_dumps.py

    # 需设备：从两个 dump 目录重新生成报告并与期望比对；--out 落盘
    python3 m19_qsa_indexer/evidence/sync_cleanup_m195/compare_dumps.py \
        --before /tmp/m195_base --after /tmp/m195_after \
        --out /tmp/dump_compare.regen.txt

退出码：0 = 一致；1 = 有差异（打印逐项）；2 = 用法 / 缺文件。

设备重跑产生 before/after 两个 dump 目录的完整命令见同目录 README §4.2。
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMITTED = HERE / "dump_compare.txt"

CASES = ["A_close_2048", "B_open_2048", "C_small_256", "D_max_65536"]
# 每档 dump 的文件顺序（= 报告里 case/dump 的行序）
SUFFIXES = [
    "ckout.bin", "cnt.bin", "comp.bin", "cosg.bin", "cosq.bin", "kout.bin",
    "logits.bin", "meta.txt", "out.bin", "qk.bin", "qout.bin", "ring.bin",
    "sing.bin", "sinq.bin", "stats.bin", "x.bin",
]
STATS_SUFFIX = "stats.bin"

# 报告抬头写死的 dump 目录标签（M195 时点）
BEFORE_LABEL = "/tmp/m195_base"
AFTER_LABEL = "/tmp/m195_after"

# 每个文件的 (before_sha256[:16], after_sha256[:16])。这是报告的独立锚点：
# 报告必须与这张表逐行相同。
EXPECTED = {
    "A_close_2048_ckout.bin": ("f20fb3da5bc566c0", "f20fb3da5bc566c0"),
    "A_close_2048_cnt.bin": ("ebba6bd66545379c", "ebba6bd66545379c"),
    "A_close_2048_comp.bin": ("bb8be67f65c7cfd2", "bb8be67f65c7cfd2"),
    "A_close_2048_cosg.bin": ("23ef822f4c9355f0", "23ef822f4c9355f0"),
    "A_close_2048_cosq.bin": ("2f0a5eb79301fafb", "2f0a5eb79301fafb"),
    "A_close_2048_kout.bin": ("67d115972bbf7926", "67d115972bbf7926"),
    "A_close_2048_logits.bin": ("3acc95212dc6eb32", "3acc95212dc6eb32"),
    "A_close_2048_meta.txt": ("f28fdadddcbe1060", "f28fdadddcbe1060"),
    "A_close_2048_out.bin": ("d858fd4f97b8319c", "d858fd4f97b8319c"),
    "A_close_2048_qk.bin": ("4534f44062b9cd21", "4534f44062b9cd21"),
    "A_close_2048_qout.bin": ("3c3ab3a6618ec0e7", "3c3ab3a6618ec0e7"),
    "A_close_2048_ring.bin": ("71e13d39588cd405", "71e13d39588cd405"),
    "A_close_2048_sing.bin": ("0aac22380ef2a00f", "0aac22380ef2a00f"),
    "A_close_2048_sinq.bin": ("5bc49bf1744cb9cc", "5bc49bf1744cb9cc"),
    "A_close_2048_stats.bin": ("c6d596c6e8e20987", "88e2864aba8d3e80"),
    "A_close_2048_x.bin": ("390f45e96d8631f4", "390f45e96d8631f4"),
    "B_open_2048_ckout.bin": ("5341e6b2646979a7", "5341e6b2646979a7"),
    "B_open_2048_cnt.bin": ("ebba6bd66545379c", "ebba6bd66545379c"),
    "B_open_2048_comp.bin": ("8d184ec1fb61eb55", "8d184ec1fb61eb55"),
    "B_open_2048_cosg.bin": ("2e53b5df1a7dc5fa", "2e53b5df1a7dc5fa"),
    "B_open_2048_cosq.bin": ("d5c4ca5068799659", "d5c4ca5068799659"),
    "B_open_2048_kout.bin": ("3281c2a951501920", "3281c2a951501920"),
    "B_open_2048_logits.bin": ("bde486482039fb96", "bde486482039fb96"),
    "B_open_2048_meta.txt": ("fefa67163387c0a4", "fefa67163387c0a4"),
    "B_open_2048_out.bin": ("99567130e5594878", "99567130e5594878"),
    "B_open_2048_qk.bin": ("44c44adc2eb832a2", "44c44adc2eb832a2"),
    "B_open_2048_qout.bin": ("6de3d3fda3700be8", "6de3d3fda3700be8"),
    "B_open_2048_ring.bin": ("7423e9d4aa8c3cd5", "7423e9d4aa8c3cd5"),
    "B_open_2048_sing.bin": ("59ff872a36b410de", "59ff872a36b410de"),
    "B_open_2048_sinq.bin": ("cf7531918f896fd0", "cf7531918f896fd0"),
    "B_open_2048_stats.bin": ("cf0577152b85e7cc", "e284548522ffa3dd"),
    "B_open_2048_x.bin": ("1087dbd43c6a367c", "1087dbd43c6a367c"),
    "C_small_256_ckout.bin": ("a7c36786326eb7a7", "a7c36786326eb7a7"),
    "C_small_256_cnt.bin": ("ebba6bd66545379c", "ebba6bd66545379c"),
    "C_small_256_comp.bin": ("3666c47d24dada52", "3666c47d24dada52"),
    "C_small_256_cosg.bin": ("ca20d96f71aebcf4", "ca20d96f71aebcf4"),
    "C_small_256_cosq.bin": ("c83f07f0646bcb51", "c83f07f0646bcb51"),
    "C_small_256_kout.bin": ("04db279a63c0b771", "04db279a63c0b771"),
    "C_small_256_logits.bin": ("a2a238071a2f4850", "a2a238071a2f4850"),
    "C_small_256_meta.txt": ("ae84068d88ce8096", "ae84068d88ce8096"),
    "C_small_256_out.bin": ("3580f7c629022cc5", "3580f7c629022cc5"),
    "C_small_256_qk.bin": ("7290789d792a9b98", "7290789d792a9b98"),
    "C_small_256_qout.bin": ("7eb7a1eeffd1139b", "7eb7a1eeffd1139b"),
    "C_small_256_ring.bin": ("29b830015709f16d", "29b830015709f16d"),
    "C_small_256_sing.bin": ("31b42b8d4a97430e", "31b42b8d4a97430e"),
    "C_small_256_sinq.bin": ("94d293344323472a", "94d293344323472a"),
    "C_small_256_stats.bin": ("d4c415a3b7f5b13f", "8a85ae31a7220c5e"),
    "C_small_256_x.bin": ("24fc6a0f9fbafd2b", "24fc6a0f9fbafd2b"),
    "D_max_65536_ckout.bin": ("bec26d52cbe1cc8d", "bec26d52cbe1cc8d"),
    "D_max_65536_cnt.bin": ("ebba6bd66545379c", "ebba6bd66545379c"),
    "D_max_65536_comp.bin": ("845eab7f90005707", "845eab7f90005707"),
    "D_max_65536_cosg.bin": ("7eeab731109ef4ba", "7eeab731109ef4ba"),
    "D_max_65536_cosq.bin": ("a776b33b690a209d", "a776b33b690a209d"),
    "D_max_65536_kout.bin": ("30cc904bac8daae0", "30cc904bac8daae0"),
    "D_max_65536_logits.bin": ("ddbcc45192a0335b", "ddbcc45192a0335b"),
    "D_max_65536_meta.txt": ("64931154dbadfb58", "64931154dbadfb58"),
    "D_max_65536_out.bin": ("bd1b16c97bdcdb39", "bd1b16c97bdcdb39"),
    "D_max_65536_qk.bin": ("18e62a78b801f91a", "18e62a78b801f91a"),
    "D_max_65536_qout.bin": ("54401ccbef416bfe", "54401ccbef416bfe"),
    "D_max_65536_ring.bin": ("3f361330d695b804", "3f361330d695b804"),
    "D_max_65536_sing.bin": ("f333d8e1a6bf3b7a", "f333d8e1a6bf3b7a"),
    "D_max_65536_sinq.bin": ("1ac8033015da9019", "1ac8033015da9019"),
    "D_max_65536_stats.bin": ("d35c6496332d0df6", "6100927d4146d423"),
    "D_max_65536_x.bin": ("1dbb51b3a3d3c0c4", "1dbb51b3a3d3c0c4"),
}

# 有序行（报告按 case 再按 SUFFIXES 展开）
ORDER = [f"{c}_{s}" for c in CASES for s in SUFFIXES]

# stats 文件的期望差异形态：字段 23..31（9 个 int32）× 32 个 AIV 记录 = 288，
# 字段 0..22 逐位相同。
EXPECTED_STATS_DIFF_COLS = list(range(23, 32))
EXPECTED_STATS_DIFF_COUNT = 288
STATS_RECORD_INTS = 32


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _unpack_i32(buf):
    n = len(buf) // 4
    return struct.unpack("<%di" % n, buf[: n * 4])


def stats_diff(before_bytes, after_bytes):
    """返回 (differing_int32_count, [differing_column_idx...], prefix_0_22_identical, rows)。"""
    bb = _unpack_i32(before_bytes)
    aa = _unpack_i32(after_bytes)
    n = min(len(bb), len(aa))
    rows = n // STATS_RECORD_INTS
    cols = set()
    count = 0
    for i in range(rows):
        base = i * STATS_RECORD_INTS
        for j in range(STATS_RECORD_INTS):
            if bb[base + j] != aa[base + j]:
                cols.add(j)
                count += 1
    prefix_ok = all(
        bb[i * STATS_RECORD_INTS + j] == aa[i * STATS_RECORD_INTS + j]
        for i in range(rows)
        for j in range(23)
    )
    return count, sorted(cols), prefix_ok, rows


def render(rows, stats, before_label=BEFORE_LABEL, after_label=AFTER_LABEL):
    """rows: [(name, b16, a16, same_bool)]；stats: {case: (count, cols, prefix_ok, rows)}。"""
    out = []
    out.append(
        "M195 device dumps: before(%s) vs after(%s) -- per-file"
        % (before_label, after_label)
    )
    out.append(
        "%-38s before_sha256[:16]  after_sha256[:16]  same" % "case/dump"
    )
    for name, b16, a16, same in rows:
        out.append("%-38s %s  %s  %s" % (name, b16, a16, "same" if same else "DIFF"))
    all_nonstats = all(
        same for name, _b, _a, same in rows if not name.endswith(STATS_SUFFIX)
    )
    out.append("")
    out.append("all non-stats dumps bit-identical: %s" % all_nonstats)
    out.append("")
    out.append(
        "=== *_stats.bin: only fields 23..31 differ "
        "(documented-reserved / uninitialized UB_TRACE tail) ==="
    )
    for case in CASES:
        count, cols, prefix_ok, _rows = stats[case]
        out.append(
            "%-14s differing int32 idx count=%d  fields=[%s]  fields[0..22] identical=%s"
            % (case, count, ", ".join(str(c) for c in cols), prefix_ok)
        )
    return "\n".join(out) + "\n"


def canonical_rows():
    return [
        (name, EXPECTED[name][0], EXPECTED[name][1],
         EXPECTED[name][0] == EXPECTED[name][1])
        for name in ORDER
    ]


def canonical_stats():
    return {
        case: (
            EXPECTED_STATS_DIFF_COUNT,
            list(EXPECTED_STATS_DIFF_COLS),
            True,
            STATS_RECORD_INTS,
        )
        for case in CASES
    }


def canonical_text():
    return render(canonical_rows(), canonical_stats())


def parse_report(text):
    """把报告解析回 (rows, nonstats_bool, stats_dict)；格式不符返回 None。"""
    lines = text.splitlines()
    if len(lines) < 4 or not lines[0].startswith("M195 device dumps:"):
        return None
    rows = []
    idx = 2
    while idx < len(lines) and lines[idx].strip():
        parts = lines[idx].split()
        if len(parts) != 4 or parts[3] not in ("same", "DIFF"):
            return None
        rows.append((parts[0], parts[1], parts[2], parts[3] == "same"))
        idx += 1
    nonstats = None
    stats = {}
    for ln in lines[idx:]:
        if ln.startswith("all non-stats dumps bit-identical:"):
            nonstats = ln.split(":", 1)[1].strip() == "True"
        elif "differing int32 idx count=" in ln:
            parts = ln.split()
            case = parts[0]
            count = int(parts[4].split("=", 1)[1])
            cols = [int(x) for x in ln.split("fields=[", 1)[1].split("]", 1)[0].split(",")]
            prefix = ln.rsplit("identical=", 1)[1].strip() == "True"
            stats[case] = (count, cols, prefix, STATS_RECORD_INTS)
    if nonstats is None or set(stats) != set(CASES):
        return None
    return rows, nonstats, stats


def check_committed(path):
    """离线：已入仓报告必须与本脚本内置的期望逐字节一致。"""
    path = Path(path)
    if not path.is_file():
        print("FAIL 缺报告文件：%s" % path)
        return 2
    got = path.read_text()
    want = canonical_text()
    if got != want:
        print("FAIL 已入仓 dump_compare.txt 与内置期望不一致：%s" % path)
        for i, (a, b) in enumerate(zip(got.splitlines(), want.splitlines()), 1):
            if a != b:
                print("  首个差异在行 %d" % i)
                print("    got : %s" % a)
                print("    want: %s" % b)
                break
        else:
            print("  行数不同：got=%d want=%d"
                  % (len(got.splitlines()), len(want.splitlines())))
        return 1
    # 结构性断言（即便文本相同也显式复核一遍语义）
    parsed = parse_report(got)
    if parsed is None:
        print("FAIL 报告格式无法解析")
        return 1
    rows, nonstats, stats = parsed
    bad = [n for n, b, a, s in rows
           if n.endswith(STATS_SUFFIX) and s]
    if bad:
        print("FAIL 以下 stats 文件被判为相同：%s" % bad)
        return 1
    if not nonstats:
        print("FAIL all non-stats dumps bit-identical 不为 True")
        return 1
    for case in CASES:
        count, cols, prefix, _ = stats[case]
        if not (count == EXPECTED_STATS_DIFF_COUNT
                and cols == EXPECTED_STATS_DIFF_COLS and prefix):
            print("FAIL stats 形态不符 case=%s: count=%d cols=%s prefix=%s"
                  % (case, count, cols, prefix))
            return 1
    print("OK   已入仓报告与内置期望逐字节一致（%d 行 / %d 文件 / DIFF=%d）"
          % (len(got.splitlines()), len(rows),
             sum(1 for _n, _b, _a, s in rows if not s)))
    return 0


def check_dirs(before, after, out_path=None):
    """需设备：从 before/after 两个 dump 目录重新生成报告并与期望比对。"""
    before, after = Path(before), Path(after)
    rows = []
    stats = {}
    problems = []
    for name in ORDER:
        bp, ap = before / name, after / name
        if not bp.is_file() or not ap.is_file():
            problems.append("缺文件：%s%s" % (name, "" if bp.is_file() else " (before)")
                            + ("" if ap.is_file() else " (after)"))
            continue
        b16 = sha16(bp)
        a16 = sha16(ap)
        rows.append((name, b16, a16, b16 == a16))
        if name.endswith(STATS_SUFFIX):
            case = name[: -len(STATS_SUFFIX) - 1]
            stats[case] = stats_diff(bp.read_bytes(), ap.read_bytes())
    if problems:
        for p in problems:
            print("FAIL " + p)
        return 1
    # --out 与比对都用仓内报告的固定抬头标签（BEFORE_LABEL/AFTER_LABEL），与调用者
    # 给的 dump 目录名无关 ⇒ 实测 dump 与 M195 时点一致时，重生成报告与入仓
    # dump_compare.txt 逐字节相等（目录可任意命名）。
    if out_path:
        Path(out_path).write_text(render(rows, stats))
        print("已写出重生成报告：%s" % out_path)
    mism = []
    for name, b16, a16, _same in rows:
        if (b16, a16) != EXPECTED[name]:
            mism.append(name)
    if mism:
        shown = ", ".join(mism[:8]) + (" ..." if len(mism) > 8 else "")
        print("FAIL %d 个文件的哈希与内置期望不一致：%s" % (len(mism), shown))
        return 1
    for case in CASES:
        count, cols, prefix, _ = stats[case]
        if not (count == EXPECTED_STATS_DIFF_COUNT
                and cols == EXPECTED_STATS_DIFF_COLS and prefix):
            print("FAIL stats 形态不符 case=%s: count=%d cols=%s prefix=%s"
                  % (case, count, cols, prefix))
            return 1
    if render(rows, stats, BEFORE_LABEL, AFTER_LABEL) != canonical_text():
        print("FAIL 重生成报告与内置期望文本不一致")
        return 1
    print("OK   设备 dump 重生成报告与内置期望一致（%d 文件）" % len(rows))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", nargs="?", const=str(COMMITTED), default=None,
                    metavar="REPORT",
                    help="离线校验已入仓报告（默认动作；REPORT 省略时用同目录 dump_compare.txt）")
    ap.add_argument("--before", metavar="DIR", help="改前 dump 目录（需设备）")
    ap.add_argument("--after", metavar="DIR", help="改后 dump 目录（需设备）")
    ap.add_argument("--out", metavar="FILE", help="把重生成的报告写到该文件（需设备）")
    args = ap.parse_args(argv)

    if args.before and args.after:
        return check_dirs(args.before, args.after, args.out)
    if args.before or args.after:
        ap.error("--before 与 --after 必须同时给出")
    return check_committed(args.check or COMMITTED)


if __name__ == "__main__":
    sys.exit(main())
