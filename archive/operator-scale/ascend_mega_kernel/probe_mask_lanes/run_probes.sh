#!/usr/bin/env bash
# probe_mask_lanes/run_probes.sh —— M94 一键复现：构建 → 逐变体独立进程 → 归档 evidence/ → 判据
# 用法（本目录下）：bash run_probes.sh [op_filter]
#   op_filter  只跑名字匹配该正则的 op（默认 '.' = 全部）。**过滤模式写到 evidence_partial/**，
#              不动 evidence/ 归档（部分刷新会让归档不再是一次完整运行）。
#
# 纪律：
#   - 每个变体一个独立进程（故障/挂死不污染后续；每条变体有独立 aclError）；
#   - 跑设备前/后各记一次 npu-smi（如实记录并发背景：fleet 上有多条 mission 并行）；
#   - 源码指纹（sha256）落 commands.txt：证据与「哪份源码」用内容寻址绑定（不靠 HEAD）；
#   - REP 数：lane/masktype/dual 组 = 1（同一 launch 内已跑两次图案 A/B 作内部重复）；
#             row 组 = 3（裁决悬案的那一组，要跨进程稳定性）。

set -u
cd "$(dirname "$0")"

FILTER="${1:-.}"
ROOT="$PWD"
if [ "$FILTER" = "." ]; then EV="$ROOT/evidence"; else EV="$ROOT/evidence_partial"; fi
LOGS="$EV/logs"
DUMPS="$EV/dumps"
BUILD="$ROOT/build"
mkdir -p "$LOGS" "$DUMPS"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
PY="/usr/local/python3.12.13/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"

{
  echo "# probe_mask_lanes（M94）证据生成记录"
  echo "date: $(date -Is)"
  echo "pwd: $ROOT"
  echo "HEAD_at_run: $(git rev-parse HEAD 2>/dev/null)  # 可能**早于**把本批证据提交进去的 commit（见 sha256 指纹）"
  echo "dirty_at_run: $(git status --porcelain 2>/dev/null | wc -l) files"
  echo "CANN: $(cat /usr/local/Ascend/cann-9.1.0/compiler/version.info 2>/dev/null | tr '\n' ' ')"
  echo "ASC arch: dav-3510 ; op filter: $FILTER"
  echo
  echo "## 源码内容指纹（内容寻址 ⇒ 用它核对「本 log 由哪份源码产出」）"
  (cd "$ROOT" && sha256sum probe_mask_lanes.asc CMakeLists.txt run_probes.sh check_locale.sh \
                 tools/check_mask_lanes.py 2>/dev/null)
  echo
  echo "## 复现命令（本目录执行）"
  echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
  echo "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8"
  echo "./build/probe_mask_lanes list > evidence/logs/oplist.txt"
  echo "bash run_probes.sh            # 本文（全矩阵 + 判据）"
  echo "$PY tools/check_mask_lanes.py .   # 独立判据（退出码 0=全成立 / 1=有反例 / 2=没得比）"
  echo
  echo "## 跑前 npu-smi（并发背景）"
  npu-smi info 2>&1
} > "$LOGS/commands.txt"

echo "== cmake 配置 =="
cmake -B "$BUILD" -S . -DCMAKE_BUILD_TYPE=Release > "$LOGS/cmake_configure.log" 2>&1 || {
  echo "cmake 配置失败，见 $LOGS/cmake_configure.log"; exit 1; }
echo "== 构建 =="
cmake --build "$BUILD" -j8 > "$LOGS/build.log" 2>&1 || {
  echo "构建失败，见 $LOGS/build.log"; tail -20 "$LOGS/build.log"; exit 1; }

"$BUILD/probe_mask_lanes" list > "$LOGS/oplist.txt"
echo "变体数：$(grep -c '^lane\|^mt_\|^dual_\|^row_' "$LOGS/oplist.txt")"

if [ "$FILTER" = "." ]; then
  rm -f "$LOGS"/run_*.log "$DUMPS"/*.bin
fi
: > "$LOGS/matrix.txt"

n_total=0; n_ok=0; n_fault=0; n_noout=0
runone() {   # runone <op> <rep>
  local op="$1" rep="$2" grp="$3"
  local tag="${op}_r${rep}"
  local log="$LOGS/run_${tag}.log"
  local outcome=""
  for attempt in 1 2 3; do
    timeout 300 "$BUILD/probe_mask_lanes" run "$op" "$DUMPS" > "$log" 2>&1
    rc=$?
    outcome="$(grep -m1 '^OUTCOME: ' "$log" | sed 's/^OUTCOME: //')"
    [ "$outcome" = "RAN" ] && break
    echo "  (attempt $attempt 无读数，重试：op=$op rep=$rep)" >&2
    sleep 3
  done
  [ -n "$outcome" ] || outcome="NO-OUTPUT"
  if [ -f "$DUMPS/ml_${op}.bin" ]; then mv -f "$DUMPS/ml_${op}.bin" "$DUMPS/ml_${tag}.bin"; fi
  echo "rc=$rc" >> "$log"
  local wr nz fnv
  wr="$(grep -m1 '^WRANGES ' "$log" | cut -c9- | cut -c1-80)"
  nz="$(grep -m1 '^NZ ' "$log" | sed 's/^NZ //')"
  fnv="$(grep -m1 '^FNV ' "$log" | awk '{print $2}')"
  printf '%-10s op=%-20s rep=%s grp=%-10s aclError=%-4s %s fnv=%s ranges=%.60s\n' \
      "$outcome" "$op" "$rep" "$grp" "$(grep -m1 '^LAUNCH' "$log" | sed 's/.*aclError=//; s/ .*//')" \
      "${nz:-no-NZ}" "${fnv:-NA}" "${wr:-NA}" | tee -a "$LOGS/matrix.txt"
  n_total=$((n_total + 1))
  case "$outcome" in
    RAN) n_ok=$((n_ok + 1)) ;;
    NO-OUTPUT|NOSYNC*) n_noout=$((n_noout + 1)) ;;
    *) n_fault=$((n_fault + 1)) ;;
  esac
}

echo "== 逐变体运行（每变体一个进程）=="
while read -r name kind group sd md pat rows pitchI pitchW im; do
  case "$name" in lane_*|mt_*|dual_*|row_*) ;; *) continue ;; esac
  echo "$name" | grep -qE "$FILTER" || continue
  if [ "$group" = "row" ]; then reps=3; else reps=1; fi
  for r in $(seq 1 "$reps"); do runone "$name" "$r" "$group"; done
done < "$LOGS/oplist.txt"

{
  echo "# 变体计数"
  echo "total=$n_total ok=$n_ok fault=$n_fault no_output=$n_noout"
} >> "$LOGS/matrix.txt"
echo "== 汇总：total=$n_total ok=$n_ok fault=$n_fault no_output=$n_noout =="

{
  echo "## 跑后 npu-smi（并发背景）"
  npu-smi info 2>&1
} >> "$LOGS/commands.txt"

if [ "$FILTER" = "." ]; then
  echo "== locale 一致性（判据抽取在 LC_ALL=C / C.UTF-8 下一致）=="
  bash "$ROOT/check_locale.sh" > "$EV/locale_check.txt" 2>&1
  echo "check_locale rc=$?  -> $EV/locale_check.txt（rc 含 locale 一致性与正对照断言）"
  tail -3 "$EV/locale_check.txt"
fi

echo "== 独立判据（tools/check_mask_lanes.py）=="
if [ "$FILTER" = "." ]; then
  "$PY" "$ROOT/tools/check_mask_lanes.py" "$ROOT" > "$EV/check_mask_lanes.log" 2>&1
else
  "$PY" "$ROOT/tools/check_mask_lanes.py" "$ROOT" --partial > "$EV/check_mask_lanes.log" 2>&1
fi
echo "check_mask_lanes rc=$?  → $EV/check_mask_lanes.log"
tail -12 "$EV/check_mask_lanes.log"
echo
echo "归档完成：$EV"
