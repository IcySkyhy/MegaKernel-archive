#!/usr/bin/env bash
# [M114] 端到端复现（零设备）：造一个 /tmp fixture —— `docs/` 真拷贝 + 其余顶层项 symlink + `.git` symlink，
# 然后往 `docs/` 里注入一个探针文档，用**改前**（base commit 上的 `docs/scan_doc_refs.py`）与**改后**扫描器各跑一次。
# 用法（在仓根跑）：bash docs/evidence/scan_doc_refs/run_e2e_fixture.sh [fixture_dir]
# 期望：探针引用真实存在时 改前 rc=1（假阳性）/ 改后 rc=0；探针引用不存在时 **两个版本都 rc=1**。
# 注意**改前**必须钉**不可移的 sha**（下面 BASE），不能钉 HEAD —— M114 落库后 HEAD 上就是改后的扫描器。
set -u
WT=$(cd "$(dirname "$0")/../../.." && pwd)
BASE=0a9922374f55ba548ecab1e09ba11b0148cefcef      # = 本分支 base（main 的 merge M112）
FIX=${1:-/tmp/m114_e2e}
BEFORE=/tmp/m114_before
rm -rf "$FIX" "$BEFORE"
mkdir -p "$FIX" "$BEFORE"
git -C "$WT" show "$BASE:docs/scan_doc_refs.py" > "$BEFORE/scan_doc_refs.py" || exit 1
cd "$FIX"
for e in $(ls -A "$WT"); do
  [ "$e" = docs ] || ln -s "$WT/$e" "$e"
done
cp -r "$WT/docs" "$FIX/docs"
rm -rf "$FIX/docs/evidence"

run() {
  printf '# M114 e2e 探针\n%s\n' "$2" > docs/98-m114-probe.md
  if [ "$1" = before ]; then cp "$BEFORE/scan_doc_refs.py" docs/scan_doc_refs.py
  else cp "$WT/docs/scan_doc_refs.py" docs/scan_doc_refs.py; fi
  echo "--- 探针[$1]：$2"
  echo "\$ cd $FIX && LC_ALL=C python3 docs/scan_doc_refs.py    # 扫描器 = $1"
  LC_ALL=C python3 docs/scan_doc_refs.py > "$FIX/out.txt" 2>&1
  rc=$?
  grep -E 'SEC-REF|scanned=|RESULT' "$FIX/out.txt"
  echo "rc=$rc"
  echo
}

run before '见 docs/20 §3.2 与 docs/20 §5（两条都真实存在）'
run after  '见 docs/20 §3.2 与 docs/20 §5（两条都真实存在）'
run before '见 docs/20 §99.7（不存在）'
run after  '见 docs/20 §99.7（不存在）'
