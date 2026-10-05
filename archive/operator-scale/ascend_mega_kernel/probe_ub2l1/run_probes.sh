#!/usr/bin/env bash
# probe_ub2l1/run_probes.sh —— M121 一键复现：构建 → 逐条短命令取设备槽 → 归档 → 独立判据
#
# 设备纪律（塔裁 2026-10-04 收紧版，本脚本按它写）：
#   - **一次 flock 只跑一条短命令（≤120s）**，`flock -w 300`；
#   - **进锁前先 `flock -n` 探锁**，探不到就跳过（记 SKIP_BUSY），**不在锁里空等**；
#     所以本脚本可以**反复重跑**：每次只补上还没取到的档，已取到的会重取（用 logs/ 里同一文件名覆盖）；
#   - 编译/分析在锁外；锁内只放设备那一段；单进程；**timeout 放在锁内**（`flock … bash -c "timeout T cmd"`），
#     这样 rc=124 只表示"命令自己超时未返回"（挂死），不会把"等锁超时"混进来；
#     等锁没拿到记 `LOCK_ACQUIRE_FAILED`，按塔的口径记为**未取得读数**（不是"未复现"）；
#   - 跑前跑后 npu-smi 都记进各自 log；挂死档（kill=2/5）排在最后，免得影响紧随其后的档。
#
# 用法（本目录下）：bash run_probes.sh
# 读哪些档取到了：看末尾的"覆盖小结"或 evidence/logs/device_batch.log 里的 rc= 行。
#   111/LOCK_ACQUIRE_FAILED = 锁被别人占着（未执行 = **未取得读数**，下次重跑再取）；
#   124 = 命令自己超时未返回（**kill=2/5 的预期读数**）。

set -u
cd "$(dirname "$0")"

ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
DUMPS="$EV/dumps"
BUILD="$ROOT/build"
mkdir -p "$LOGS" "$DUMPS" "$DUMPS/capi_mmad" "$DUMPS/capi_mmad_opt1" "$DUMPS/capi_mmad_opt2" \
         "$DUMPS/basic_mmad_default" "$DUMPS/basic_mmad_ssbuf" "$DUMPS/capi_rt"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
PY="/usr/local/python3.12.13/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"

echo "== 磁盘先看一眼（根卷满时失败形态是静默截断）=="
df -h / | tail -1

{
  echo "# probe_ub2l1（M121）证据生成记录（本批 $(date -Is)）"
  echo "date: $(date -Is)"
  echo "pwd: $ROOT"
  echo "HEAD_at_run: $(git rev-parse HEAD 2>/dev/null)"
  echo "dirty_at_run: $(git status --porcelain 2>/dev/null | wc -l) files"
  echo "CANN: $(tr '\n' ' ' < /usr/local/Ascend/cann-9.1.0/compiler/version.info 2>/dev/null)"
  echo "ASC arch: dav-3510"
  echo
  echo "## 源码内容指纹（内容寻址 ⇒ 用它核对「本 log 由哪份源码产出」）"
  sha256sum probe_ub2l1.asc CMakeLists.txt run_probes.sh check_ref.py evidence/version_diff.sh \
            evidence/slot.sh evidence/retake.sh 2>/dev/null
  echo
  echo "## 复现命令（本目录执行）"
  echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
  echo "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4"
  echo "bash evidence/version_diff.sh    # host 侧版本差取证（零设备）"
  echo "bash run_probes.sh               # 本文（逐条短命令取设备槽；可反复重跑补齐）"
  echo
  echo "## 构建"
} > "$LOGS/commands.txt"

echo "== cmake 配置 =="
cmake -B "$BUILD" -S . -DCMAKE_BUILD_TYPE=Release > "$LOGS/cmake_configure.log" 2>&1 || {
  echo "cmake 配置失败，见 $LOGS/cmake_configure.log"; exit 1; }
echo "== 构建 =="
cmake --build "$BUILD" -j4 > "$LOGS/build.log" 2>&1 || {
  echo "构建失败，见 $LOGS/build.log"; tail -20 "$LOGS/build.log"; exit 1; }
echo "构建 OK"
{
  echo "cmake rc=0 ; build rc=0"
  md5sum "$BUILD"/probe_ub2l1 "$BUILD"/probe_ub2l1_ssbuf
} >> "$LOGS/commands.txt"

# ---- 设备档清单：每行 "name|timeout|二进制|参数..." ----
# 参数：all（全部）| core（README 直接引用的最小集，锁紧张时先取这批）
device_list() {
  local B="$BUILD" D="$DUMPS"
  local SEL="${1:-all}"
  # none：不动设备，只把本脚本的元数据（commands.txt）与 check_ref 复核刷一遍
  if [ "$SEL" = "none" ]; then
    return   # 不产出任何档（连注释行也不产出，免得被下面的 while 当成一条命令）
  fi
  if [ "$SEL" = "core" ]; then
    echo "capi_rt_k0_default|90|$B/probe_ub2l1|capi-rt|0|$D/capi_rt"
    echo "capi_rt_k0_ssbuf|90|$B/probe_ub2l1_ssbuf|capi-rt|0"
    echo "capi_rt_k1_nowait|90|$B/probe_ub2l1|capi-rt|1"
    echo "capi_mmad_k0_opt0|120|$B/probe_ub2l1|capi-mmad|0|$D/capi_mmad|0"
    echo "capi_mmad_k0_opt1|120|$B/probe_ub2l1|capi-mmad|0|$D/capi_mmad_opt1|1"
    echo "capi_mmad_k0_opt2|120|$B/probe_ub2l1|capi-mmad|0|$D/capi_mmad_opt2|2"
    echo "basic_mmad_k0_default|120|$B/probe_ub2l1|basic-mmad|0|$D/basic_mmad_default"
    echo "basic_mmad_k0_ssbuf|120|$B/probe_ub2l1_ssbuf|basic-mmad|0|$D/basic_mmad_ssbuf"
    echo "capi_bw_64k_300|120|$B/probe_ub2l1|capi-bw|65536|300"
    echo "basic_bw_64k_300_default|120|$B/probe_ub2l1|basic-bw|65536|300"
    echo "capi_rt_k2_noset|45|$B/probe_ub2l1|capi-rt|2"
    echo "basic_mmad_k2_noset|45|$B/probe_ub2l1|basic-mmad|2"
    return
  fi
  # ① 二分定位（step 1..9）——把 aicore exception 钉到指令；本机 9.1.0 实测全 NO_FAULT
  for s in 1 2 9 3 4 5 6 7 8; do echo "capi_step${s}|40|$B/probe_ub2l1|capi-step|$s"; done
  # ① C API 往返 + 同步变体
  echo "capi_rt_k0_default|90|$B/probe_ub2l1|capi-rt|0|$D/capi_rt"
  echo "capi_rt_k1_nowait|90|$B/probe_ub2l1|capi-rt|1"
  echo "capi_rt_k3_blockonly|90|$B/probe_ub2l1|capi-rt|3"
  echo "capi_rt_k4_intraonly|90|$B/probe_ub2l1|capi-rt|4"
  echo "capi_rt_k0_ssbuf|90|$B/probe_ub2l1_ssbuf|capi-rt|0"
  # ① 消费者验证：opt=0（asc_copy_l0c2gm + nz2nd）与 opt=2（基础 API Fixpipe 读同一块 CO1）
  #    都是**正确**的回写口径（实测逐位相同）；只有 opt=1（关 nz2nd）是错的对照
  echo "capi_mmad_k0_opt0|120|$B/probe_ub2l1|capi-mmad|0|$D/capi_mmad|0"
  echo "capi_mmad_k0_opt1|120|$B/probe_ub2l1|capi-mmad|0|$D/capi_mmad_opt1|1"
  echo "capi_mmad_k0_opt2|120|$B/probe_ub2l1|capi-mmad|0|$D/capi_mmad_opt2|2"
  # ③ 默认基础 API（塔的原假设："应走 GM+Matmul 注册 ⇒ 不可用"）与 ② 带宏版本
  echo "basic_mmad_k0_default|120|$B/probe_ub2l1|basic-mmad|0|$D/basic_mmad_default"
  echo "basic_mmad_k0_ssbuf|120|$B/probe_ub2l1_ssbuf|basic-mmad|0|$D/basic_mmad_ssbuf"
  echo "basic_mmad_k1_nowait|120|$B/probe_ub2l1|basic-mmad|1"
  # 带宽（本 mission 最载重的读数）：三档 N
  echo "capi_bw_1k_2k|120|$B/probe_ub2l1|capi-bw|1024|2000"
  echo "capi_bw_8k_1k|120|$B/probe_ub2l1|capi-bw|8192|1000"
  echo "capi_bw_64k_300|120|$B/probe_ub2l1|capi-bw|65536|300"
  echo "basic_bw_64k_300_default|120|$B/probe_ub2l1|basic-bw|65536|300"
  echo "basic_bw_64k_300_ssbuf|120|$B/probe_ub2l1_ssbuf|basic-bw|65536|300"
  # 挂死档（预期 rc=124）排最后，不让它污染前面的读数
  echo "capi_rt_k2_noset|45|$B/probe_ub2l1|capi-rt|2"
  echo "basic_mmad_k2_noset|45|$B/probe_ub2l1|basic-mmad|2"
  echo "capi_rt_k5_halfarrive_2|45|$B/probe_ub2l1|capi-rt|5"
  echo "basic_mmad_k5_halfarrive_2|45|$B/probe_ub2l1|basic-mmad|5"
}

BATCH_LOG="$LOGS/device_batch.log"
echo "== 逐条取设备槽（一次 flock 一条；M121_ONLY=${M121_ONLY:-all}，M121_WAIT=${M121_WAIT:-0}）=="
{
  echo "================ 批次 $(date -Is) HEAD=$(git rev-parse --short HEAD 2>/dev/null) ================"
} >> "$BATCH_LOG"

NOK=0; NSKIP=0
while IFS='|' read -r name tmo bin rest; do
  [ -z "${name:-}" ] && continue
  # rest 是 "arg1|arg2|..." → 展开成参数
  IFS='|' read -r -a args <<< "$rest"
  out="$(bash evidence/slot.sh "$name" "$tmo" "$bin" "${args[@]}")"
  echo "$name  ->  $out"
  echo "$out" >> "$BATCH_LOG"
  case "$out" in
    *"LOCK_ACQUIRE_FAILED"*|*"rc=111"*) NSKIP=$((NSKIP+1)) ;;
    *)          NOK=$((NOK+1)) ;;
  esac
done < <(device_list "${M121_ONLY:-all}")
if [ "${M121_ONLY:-all}" = "none" ]; then
  echo "覆盖小结：M121_ONLY=none —— 本轮**不取任何设备档**（只刷新 commands.txt / check_ref.log）" | tee -a "$BATCH_LOG"
else
  echo "覆盖小结：本轮执行 $NOK 条，因锁被占跳过 $NSKIP 条（跳过的不计入读数，重跑本脚本补齐）" | tee -a "$BATCH_LOG"
fi

echo
echo "== 独立判据（host 侧，零设备）=="
{
  # 注：capi_mmad_opt1 是**负向对照档**（关掉 nz2nd，结果是错的）⇒ 它那一段 rc=1 是**预期**；
  #     其余五段 rc=0 才是"判据成立"。拿不准就看该档自己的 log（README §5.4 有逐档读数）。
  for d in capi_mmad capi_mmad_opt1 capi_mmad_opt2 basic_mmad_default basic_mmad_ssbuf capi_rt; do
    case "$d" in
      capi_mmad_opt1) echo "### check_ref.py $d（负向对照：预期 rc=1）";;
      *)              echo "### check_ref.py $d";;
    esac
    "$PY" check_ref.py "$DUMPS/$d"; echo "rc=$?"
  done
} 2>&1 | tee "$LOGS/check_ref.log"

echo "完成。日志：$LOGS/"
