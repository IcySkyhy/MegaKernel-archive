#!/usr/bin/env bash
# [M114] 一键重建本目录的全部 log（在仓根跑；**零设备**）：bash docs/evidence/scan_doc_refs/repro.sh
#
# 「改前」的定义 = 本分支 base 那个 commit 上的 docs/scan_doc_refs.py。**钉不可移的 sha**，不钉 HEAD
# （M114 落库后 HEAD 上就是改后的扫描器；报告里的读数都标了这个 sha）。
set -u
cd "$(dirname "$0")/../../.."
E=docs/evidence/scan_doc_refs
BASE=0a9922374f55ba548ecab1e09ba11b0148cefcef      # = 本分支 base（main 的 merge M112）
BEFORE=/tmp/m114_before
mkdir -p "$BEFORE"
git show "$BASE:docs/scan_doc_refs.py" > "$BEFORE/scan_doc_refs.py" || exit 1

# ---- 1. 主扫描：改前 / 改后（双 locale + --include-self + --ignore-allowlist + --selftest 尾行）----
for v in before after; do
  if [ "$v" = before ]; then SC="$BEFORE/scan_doc_refs.py"; else SC=docs/scan_doc_refs.py; fi
  {
    echo "# [M114] 主扫描读数 —— 改$([ "$v" = before ] && echo 前 || echo 后)：$SC"
    echo "# 语料 base = $BASE（本 mission 未改任何 docs/*.md）"
    echo "# 命令：cd <repo root> && LC_ALL=C python3 $SC ; echo rc=\$?"
    echo
    echo "\$ LC_ALL=C python3 $SC"
    LC_ALL=C python3 "$SC"; echo "rc=$?"
    echo
    echo "\$ LC_ALL=C.UTF-8 python3 $SC | grep -E 'scanned=|RESULT|allowlist-unused'"
    LC_ALL=C.UTF-8 python3 "$SC" | grep -E 'scanned=|RESULT|allowlist-unused'; echo "rc=${PIPESTATUS[0]}"
    echo
    echo "\$ LC_ALL=C python3 $SC --include-self | grep -E 'SEC-REF|scanned=|RESULT'"
    LC_ALL=C python3 "$SC" --include-self | grep -E 'SEC-REF|scanned=|RESULT'; echo "rc=${PIPESTATUS[0]}"
    echo
    echo "\$ LC_ALL=C python3 $SC --ignore-allowlist | grep -E 'scanned=|RESULT'"
    LC_ALL=C python3 "$SC" --ignore-allowlist | grep -E 'scanned=|RESULT'; echo "rc=${PIPESTATUS[0]}"
    echo
    echo "\$ LC_ALL=C python3 $SC --selftest      # 全文"
    LC_ALL=C python3 "$SC" --selftest; echo "rc=$?"
    echo
    echo "\$ LC_ALL=C.UTF-8 python3 $SC --selftest | tail -1"
    LC_ALL=C.UTF-8 python3 "$SC" --selftest | tail -1; echo "rc=${PIPESTATUS[0]}"
  } > "$E/scan_$v.log" 2>&1
done

# ---- 2. 影响面量尺：改前 / 改后（同一份量尺脚本，只换被判的扫描器）----
LC_ALL=C python3 "$E/measure_sec_refs.py" . "$BEFORE/scan_doc_refs.py" > "$E/measure_before.log" 2>&1
LC_ALL=C python3 "$E/measure_sec_refs.py" . docs/scan_doc_refs.py     > "$E/measure_after.log"  2>&1

# ---- 3. 逆证（改前 / 永真 / 永假三档）与端到端 fixture ----
bash "$E/check_inverse_controls.sh" > "$E/nc_inverse_controls.log" 2>&1
bash "$E/run_e2e_fixture.sh"        > "$E/e2e_fixture.log"         2>&1

echo "wrote: $(cd "$E" && ls -1 *.log | tr '\n' ' ')"
