#!/usr/bin/env bash
# check_locale.sh —— M94 判据命令的 locale 一致性 + 扫描正对照
# 用法（本目录下）：bash check_locale.sh    （产物：evidence/locale_check.txt）
#
# 做什么：
#   1) 在 LC_ALL=C / C.UTF-8 / POSIX 三种 locale 下，对同一批 log 跑**同一套**判据抽取
#      （只用 ASCII 字符类），要求三份读数逐字节相同；
#   2) 正对照：同一套抽取必须在**形态不同的两组**上给出**不同**读数 ——
#      `row_nc_*`（应咬到 R1 越界 > 0）与 `row_m16_cur`（应咬到越界 = 0）。
#      若两者读数相同，说明抽取根本没在判别形态。

set -u
cd "$(dirname "$0")"

EV="$PWD/evidence"
LOGS="$EV/logs"
OUT="$EV/locale_check.txt"
: > "$OUT"

extract() {   # extract <logfile>
  local f="$1"
  printf '%s' "$(basename "$f")"
  printf ' | famount=%s' "$(grep -m1 '^LAUNCH' "$f" | sed 's/.*aclError=//; s/ .*//')"
  printf ' | nz=%s' "$(grep -m1 '^NZ ' "$f" | sed 's/^NZ bytes_written=//; s/ .*//')"
  printf ' | fnv=%s' "$(grep -m1 '^FNV ' "$f" | sed 's/^FNV //')"
  printf ' | nrange=%s\n' "$(grep -m1 '^WRANGES ' "$f" | tr ',' '\n' | grep -c .)"
}

FILES=(
  "$LOGS/run_row_m16_cur_r1.log"
  "$LOGS/run_row_m1_cur_r1.log"
  "$LOGS/run_row_nc_p192_r1.log"
  "$LOGS/run_row_nc_p128_b16_r1.log"
  "$LOGS/run_row_sparse4_p512_r1.log"
  "$LOGS/run_lane_s32_m32_all_r1.log"
)

{
  echo "# M94 判据命令 locale 一致性 + 扫描正对照"
  echo "date: $(date -Is)"
  echo
  for LC in C C.UTF-8 POSIX; do
    {
      echo "===== LC_ALL=$LC ====="
      for f in "${FILES[@]}"; do [ -f "$f" ] && LC_ALL="$LC" extract "$f"; done
      LC_ALL="$LC" locale 2>/dev/null | head -1
    } > "/tmp/m94_locale_${LC//./_}.txt"
    echo "===== LC_ALL=$LC ====="
    cat "/tmp/m94_locale_${LC//./_}.txt"
    echo
  done
} >> "$OUT" 2>&1

for LC in C C.UTF-8 POSIX; do
  grep -E '\.log \| famount=' "/tmp/m94_locale_${LC//./_}.txt" > "/tmp/m94_body_${LC//./_}.txt"
done

{
  echo "== 一致性判定（三份读数行逐字节 diff）=="
  if diff -q /tmp/m94_body_C.txt /tmp/m94_body_C_UTF-8.txt >/dev/null \
     && diff -q /tmp/m94_body_C.txt /tmp/m94_body_POSIX.txt >/dev/null; then
    echo "PASS: LC_ALL=C / C.UTF-8 / POSIX 三份读数行逐字节相同（$(wc -l < /tmp/m94_body_C.txt) 行）"
  else
    echo "FAIL: locale 间读数不一致"
    diff /tmp/m94_body_C.txt /tmp/m94_body_C_UTF-8.txt
  fi
  echo
  echo "== 正对照：同一套抽取在「越界形态」与「非越界形态」上必须给出不同读数 =="
  # 判据量 = nrange（与哨兵不同的**连续段数**）：非越界档 1 段；越界档落盘冲出末行槽 ⇒ 至少 2 段。
  # 注：**不要**拿 nz 比大小当判据 —— 行距从 256B 缩到 192B 后，槽内可写总量 = 16×192 + 4096 = 7232
  #     必然**小于** clean 档的 8192（行距组 nz 不是单调量）。
  echo "  非越界（判据 nrange == 1）:        $(grep 'row_m16_cur_r1' /tmp/m94_body_C.txt)"
  echo "  越界对照（判据 nrange > 1）:       $(grep 'row_nc_p192_r1' /tmp/m94_body_C.txt)"
  echo "  越界对照 b16（判据 nrange > 1）:   $(grep 'row_nc_p128_b16_r1' /tmp/m94_body_C.txt)"

  # ---- 真断言（不是展示）：失败必须反映到 rc ----
  nr() { grep "$1" /tmp/m94_body_C.txt | sed 's/.*nrange=//'; }
  nz() { grep "$1" /tmp/m94_body_C.txt | sed 's/.*nz=\([0-9]*\).*/\1/'; }
  fn() { grep "$1" /tmp/m94_body_C.txt | sed 's/.*fnv=//; s/ .*//'; }
  n_clean=$(nr row_m16_cur_r1);   z_clean=$(nz row_m16_cur_r1)
  n_p192=$(nr row_nc_p192_r1);    z_p192=$(nz row_nc_p192_r1)
  n_p128=$(nr row_nc_p128_b16_r1); z_p128=$(nz row_nc_p128_b16_r1)
  f_clean=$(fn row_m16_cur_r1); f_p192=$(fn row_nc_p192_r1)
  a_ok=1
  assert() {  # assert <name> <cond-result>
    if [ "$2" = "1" ]; then echo "  ASSERT PASS: $1"; else echo "  ASSERT FAIL: $1"; a_ok=0; fi
  }
  echo
  echo "== 断言（失败 ⇒ 本脚本 rc=1）=="
  assert "判据量 nrange 非空洞：clean=$n_clean（应为 1）" "$([ "$n_clean" = "1" ] && echo 1 || echo 0)"
  assert "越界档 nrange 必须 > 1：p192=$n_p192" "$([ "${n_p192:-0}" -gt 1 ] 2>/dev/null && echo 1 || echo 0)"
  assert "越界档 nrange 必须 > 1：p128_b16=$n_p128" "$([ "${n_p128:-0}" -gt 1 ] 2>/dev/null && echo 1 || echo 0)"
  assert "三档 nz 必须 > 0（非 0-vs-0）：clean=$z_clean p192=$z_p192 p128b16=$z_p128" \
         "$([ "${z_clean:-0}" -gt 0 ] && [ "${z_p192:-0}" -gt 0 ] && [ "${z_p128:-0}" -gt 0 ] && echo 1 || echo 0)"
  assert "越界档与 clean 的 dump 指纹必须不同：clean=$f_clean p192=$f_p192" \
         "$([ -n "$f_p192" ] && [ "$f_p192" != "$f_clean" ] && echo 1 || echo 0)"
  if diff -q /tmp/m94_body_C.txt /tmp/m94_body_C_UTF-8.txt >/dev/null \
     && diff -q /tmp/m94_body_C.txt /tmp/m94_body_POSIX.txt >/dev/null; then l_ok=1; else l_ok=0; fi
  assert "三 locale 读数行逐字节一致" "$l_ok"
  if [ "$a_ok" = "1" ]; then
    echo "RESULT: PASS（locale 一致性 + 正对照断言全部成立）"
  else
    echo "RESULT: FAIL（见上面的 ASSERT FAIL）"
  fi
} >> "$OUT" 2>&1

cat "$OUT"
# rc 契约：全断言成立 ⇒ 0；任一不成立 ⇒ 1（a_ok=1 表示**成立**，故此处要取反）
if [ "$a_ok" = "1" ]; then exit 0; else exit 1; fi
