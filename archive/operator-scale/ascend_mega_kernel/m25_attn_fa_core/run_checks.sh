#!/usr/bin/env bash
# ============================================================
# run_checks.sh —— M117 / Wave-B3 的一键复跑（**设备槽 + 确定性 + 负向对照**）
#
# 用法（仓库根目录，或任意目录；脚本自己解析路径）：
#   bash m25_attn_fa_core/run_checks.sh
# 环境变量：
#   M25FA_M="4097,1"   跑哪些 m（默认 4097,1）
#   M25FA_REPEAT=5     core 档连跑几次比 sha256（默认 5）
#   M25FA_OUT=<dir>    落盘根（默认 m25_attn_fa_core/out）
#   M25FA_NEG_M=4097   负向对照在哪个 m 上跑/断言（默认 4097 = prefill 档）。
#                      负控只在能改变**短上下文行**的档会咬：prefill m=4097 行 0 只见 j=0；
#                      而 decode m=1 的单行 posBase=ctx-1 已见全部上下文，negmask/negshift/negstart
#                      只引入 ~1e-3 级扰动（落在尺度型容差内）⇒ 在 m=1 上**不咬**。
#                      设备读数见 evidence/M196_readings.txt §3。
#
# **不写绝对断言**：脚本只打印每次读数与判定，把 PASS/FAIL 汇总成一张表；
# 退出码 0 = 「core 全 PASS 且负向对照全 FAIL 且 core 的 sha256 全一致」，否则 1。
# ============================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/build/m25_attn_fa_core"
OUT="${M25FA_OUT:-$HERE/out}"
MS="${M25FA_M:-4097,1}"
REPEAT="${M25FA_REPEAT:-5}"
NEG="${M25FA_NEG_M:-4097}"

if [[ ! -x "$BIN" ]]; then
  echo "[run_checks][FAIL] 找不到可执行 $BIN（先 cmake --build）"
  exit 1
fi

echo "=== df -h / ==="; df -h / | tail -1
echo "=== 设备槽：flock -w 900 /tmp/npu0.lock ==="

RC=0
# 1) core 档连跑 REPEAT 次（确定性）+ 三个负向对照各跑 1 次
flock -w 900 /tmp/npu0.lock bash -c '
  set -uo pipefail
  echo "--- npu-smi（进锁后复查）---"; npu-smi info -t common -i 0 2>/dev/null | head -4
  BIN="'"$BIN"'"; OUT="'"$OUT"'"; MS="'"$MS"'"; REPEAT="'"$REPEAT"'"; NEG="'"$NEG"'"
  for i in $(seq 1 "$REPEAT"); do
    M25FA_M="$MS" M25FA_OUT="$OUT/rep$i" "$BIN" core
  done
  for mode in negmask negshift negstart; do
    M25FA_M="$NEG" M25FA_OUT="$OUT/$mode" "$BIN" "$mode"
  done
'

echo
echo "=== 判定（check_ref.py；core 必须 PASS、三个负向对照必须 FAIL）==="
PY=/usr/local/python3.12.13/bin/python3
[[ -x "$PY" ]] || PY=python3

declare -a ROWS
for i in $(seq 1 "$REPEAT"); do
  for m in ${MS//,/ }; do
    d="$OUT/rep$i/m${m}_core"
    if [[ -f "$d/out.bin" ]]; then
      line=$("$PY" "$HERE/check_ref.py" "$d" 2>&1 | tail -1)
      echo "[core rep$i m=$m] $line"
      [[ "$line" == VERDICT=PASS* ]] || RC=1
      ROWS+=("core m=$m rep$i $(sha256sum "$d/out.bin" | cut -c1-16)")
    else
      echo "[core rep$i m=$m] 缺 out.bin（跑挂了？）"; RC=1
    fi
  done
done
for mode in negmask negshift negstart; do
  for m in ${NEG//,/ }; do
    d="$OUT/$mode/m${m}_${mode}"
    if [[ -f "$d/out.bin" ]]; then
      line=$("$PY" "$HERE/check_ref.py" "$d" --quiet 2>&1 | tail -1)
      echo "[$mode m=$m] $line"
      # 负向对照必须 FAIL（否则判据是空洞的）；只在 M25FA_NEG_M（默认 prefill 4097）上断言，
      # 因为 decode m=1 的单行已见全部上下文、这三个负控在该档不咬（见 evidence/M196_readings.txt §3）。
      [[ "$line" == VERDICT=FAIL* ]] || { echo "  ^^ 负向对照没变红 —— 判据空洞"; RC=1; }
    else
      echo "[$mode m=$m] 缺 out.bin"; RC=1
    fi
  done
done

echo
echo "=== core 档 sha256（确定性：同一 (m) 的多次必须一致）==="
printf '%s\n' "${ROWS[@]}"
for m in ${MS//,/ }; do
  n=$(printf '%s\n' "${ROWS[@]}" | grep " m=$m " | awk '{print $NF}' | sort -u | wc -l)
  echo "m=$m 的不同 sha256 数 = $n（期望 1）"
  [[ "$n" == "1" ]] || RC=1
done

echo
echo "RUN_CHECKS_RC=$RC"
exit $RC
