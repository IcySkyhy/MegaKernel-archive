#!/usr/bin/env bash
# check_fingerprints.sh —— 核对 `evidence/logs/commands.txt` 里记的**源码指纹**是否等于**当前工作树**。
#
# 为什么需要它：`run_probes.sh` 在**提交之前**跑（证据本身就是提交内容之一），所以 commands.txt 的
# `HEAD_at_run` 天然会早于"把本批证据提交进去的那个 commit"。**判"这份 log 由哪份源码产出"要靠
# 内容寻址的 sha256 指纹，而不是 HEAD**。本脚本把这条核对变成一条命令。
#
# 用法（本目录下）：bash check_fingerprints.sh
#   rc=0 全部一致；rc=1 有不一致（列出差异）
#   注：先别把 evidence/ 本身纳入比对——只比对**源码/脚本**（它们是产出证据的输入）。

set -u
cd "$(dirname "$0")"

FILES=(probe_host_dma.asc CMakeLists.txt run_probes.sh mmap_ngram_plan.py check_locale.sh check_fingerprints.sh)
CMD="evidence/logs/commands.txt"

if [ ! -f "$CMD" ]; then
  echo "FAIL: $CMD 不存在"; exit 1
fi

fail=0
echo "== 源码指纹核对：commands.txt 记录 vs 当前工作树 =="
for f in "${FILES[@]}"; do
  now=$(sha256sum "$f" 2>/dev/null | awk '{print $1}')
  rec=$(grep -E "^[0-9a-f]{64}  ${f}\$" "$CMD" 2>/dev/null | awk '{print $1}')
  if [ -z "$rec" ]; then
    printf '  %-24s RECORDED=%-8s NOW=%s  -> 记录里没有该文件的指纹\n' "$f" "(missing)" "${now:0:12}"
    fail=1
  elif [ "$now" = "$rec" ]; then
    printf '  %-24s OK (%s)\n' "$f" "${now:0:12}"
  else
    printf '  %-24s MISMATCH  recorded=%s  now=%s\n' "$f" "${rec:0:12}" "${now:0:12}"
    fail=1
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "PASS: commands.txt 记的源码指纹与当前工作树逐字节一致"
  echo "      （⇒ 该批 log 由**当前这份源码**产出；HEAD_at_run 早晚于提交是正常的，见脚本注释）"
else
  echo "FAIL: 有不一致 —— 要么改了源码没重跑 run_probes.sh，要么 commands.txt 被手工改过"
fi
exit "$fail"
