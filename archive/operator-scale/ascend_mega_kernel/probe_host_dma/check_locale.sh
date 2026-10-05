#!/usr/bin/env bash
# check_locale.sh —— 判据命令的 locale 一致性 + 扫描正对照（M86 纪律要求）
# 用法（本目录下）：bash check_locale.sh
# 产物：evidence/locale_check.txt
#
# 做什么：
#   1) 在 LC_ALL=C / LC_ALL=C.UTF-8 / LC_ALL=POSIX 三种 locale 下，对同一批 log 跑**同一套**
#      判据抽取（只用 ASCII 字符类），要求三份读数**逐字节相同**；
#   2) 正对照（负向对照的正向版）：同一套抽取在**正例 log**（应咬到 aclError=0 / mismatches=0）
#      与**负例 log**（应咬到 507035）上给出**不同**读数 ⇒ 说明这套正则确实在判别形态，
#      而不是「怎么跑都返回同一串」。

set -u
cd "$(dirname "$0")"

EV="$PWD/evidence"
LOGS="$EV/logs"
OUT="$EV/locale_check.txt"
: > "$OUT"

# 判据抽取函数：只用 ASCII 字符类，输出与 locale 无关
extract() {
  local f="$1"
  printf '%s' "$(basename "$f")"
  printf ' | aclError=%s' "$(grep -m1 'aclError_after_last' "$f" | sed 's/.*=//')"
  printf ' | mism=%s' "$(grep -m1 '^\[VERIFY\] ' "$f" | sed 's/.*mismatches=\([0-9]*\).*/\1/')"
  printf ' | regRC=%s' "$(grep -m1 'HostRegisterV2' "$f" | sed 's/.*rc=//; s/ .*//')"
  printf ' | overall=%s\n' "$(grep -m1 '^== done' "$f" | sed 's/== done mode=[a-z]* overall=//; s/ ==//')"
}

FILES=(
  "$LOGS/read_run1.log"
  "$LOGS/hbm_run1.log"
  "$LOGS/unreg_unregistered_ptr.log"
  "$LOGS/page_partial_registration.log"
  "$LOGS/filemmap_file.log"
  "$LOGS/filemmap_anon_run1.log"
)

{
  echo "# 判据命令 locale 一致性 + 扫描正对照"
  echo "date: $(date -Is)"
  echo
  for LC in C C.UTF-8 POSIX; do
    {
      echo "===== LC_ALL=$LC ====="
      for f in "${FILES[@]}"; do LC_ALL="$LC" extract "$f"; done
      LC_ALL="$LC" locale 2>/dev/null | head -1
    } > "/tmp/m86_locale_${LC//./_}.txt"
    echo "===== LC_ALL=$LC ====="
    cat "/tmp/m86_locale_${LC//./_}.txt"
    echo
  done
} >> "$OUT" 2>&1

# 只比读数行（去掉自己写的 "===== LC_ALL=… =====" 抬头与 locale 打印行）
for LC in C C.UTF-8 POSIX; do
  grep -E '\.log \| aclError=' "/tmp/m86_locale_${LC//./_}.txt" > "/tmp/m86_locale_body_${LC//./_}.txt"
done

{
  echo "== 一致性判定（三份读数行逐字节 diff）=="
  if diff -q /tmp/m86_locale_body_C.txt /tmp/m86_locale_body_C_UTF-8.txt >/dev/null \
     && diff -q /tmp/m86_locale_body_C.txt /tmp/m86_locale_body_POSIX.txt >/dev/null; then
    echo "PASS: LC_ALL=C / C.UTF-8 / POSIX 三份读数行逐字节相同（$(wc -l < /tmp/m86_locale_body_C.txt) 行）"
  else
    echo "FAIL: locale 间读数不一致"
    diff /tmp/m86_locale_body_C.txt /tmp/m86_locale_body_C_UTF-8.txt
    diff /tmp/m86_locale_body_C.txt /tmp/m86_locale_body_POSIX.txt
  fi
  echo
  echo "== 正对照：同一套抽取在正例/负例上必须给出不同读数 =="
  echo "  正例 read_run1 (expect aclError=0,mism=0):        $(grep 'read_run1' /tmp/m86_locale_body_C.txt)"
  echo "  负例 unreg   (expect aclError=507035,mism>0):     $(grep 'unreg_' /tmp/m86_locale_body_C.txt)"
  echo "  负例 filemap (expect regRC=507899,overall=REJ):   $(grep 'filemmap_file' /tmp/m86_locale_body_C.txt)"
} >> "$OUT"

cat "$OUT"
