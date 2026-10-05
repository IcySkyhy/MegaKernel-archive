#!/usr/bin/env bash
# probe_ub2l1/evidence/retake.sh —— 在设备紧俏时**逐条**补关键读数（一次 flock 只跑一条短命令）
#
# 为什么单独有这么一个脚本：
#   整批 run_probes.sh 需要长时间占设备；11 个 agent 抢一把锁时很难整批拿全。
#   本脚本只取 README 直接引用的那几条，**一条命令一次 flock**，每条用 M121_WAIT（默认 200，上限 300）
#   做有界获取；拿不到就如实记 LOCK_ACQUIRE_FAILED（= 本轮未取到），下次重跑再补。
#
# 用法（本目录下）：bash evidence/retake.sh [extra|witness]
#   witness（默认）：同步语义见证（重复采样，因为 kill=1 是竞态，单次采样不足以定性）
#   extra          ：其余核心读数（消费者验证 / 带宽）
#
# 读数落在 evidence/logs/<name>.log，每档一行 rc= 追加进 evidence/logs/device_batch.log。
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"; LOGS="$ROOT/evidence/logs"; DUMPS="$ROOT/evidence/dumps"; B="$ROOT/build"
mkdir -p "$LOGS" "$DUMPS/capi_mmad" "$DUMPS/capi_mmad_opt0" "$DUMPS/basic_mmad_default" "$DUMPS/basic_mmad_ssbuf" "$DUMPS/capi_rt"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

run() {  # run <name> <timeout> <cmd...>
  local name="$1" tmo="$2"; shift 2
  local out; out="$(bash evidence/slot.sh "$name" "$tmo" "$@")"
  echo "$out"; echo "$out" >> "$LOGS/device_batch.log"
}

MODE="${1:-witness}"
echo "== retake mode=$MODE  M121_WAIT=${M121_WAIT:-200}  at=$(date -Is) =="

if [ "$MODE" = "witness" ]; then
  # kill=1（AIC 两侧 wait 全去掉）是**竞态**，单次采样不足以定性 ⇒ 重复采样给分布
  for r in 1 2 3 4; do run "capi_rt_k1_r${r}"  90 "$B/probe_ub2l1" capi-rt 1; done
  for r in 1 2;    do run "basic_mmad_k1_r${r}" 90 "$B/probe_ub2l1" basic-mmad 1; done
  # 挂死见证（预期 rc=124）
  run "capi_rt_k2_noset"        45 "$B/probe_ub2l1" capi-rt 2
  run "basic_mmad_k2_noset"     45 "$B/probe_ub2l1" basic-mmad 2
  # mode 2 是否要求配对的两个 AIV 都到
  run "capi_rt_k5_halfarrive"   45 "$B/probe_ub2l1" capi-rt 5
  run "basic_mmad_k5_halfarrive" 45 "$B/probe_ub2l1" basic-mmad 5
elif [ "$MODE" = "steps" ]; then
  # capi-step 的 9 步二分（r1 复审后用来重采 step1..4,9；`timeout` 在锁内，rc=124 才算挂死）
  for s in 1 2 3 4 9; do run "capi_step${s}" 60 "$B/probe_ub2l1" capi-step "$s"; done
  run "capi_rt_k4_intraonly" 90 "$B/probe_ub2l1" capi-rt 4
else
  run "capi_mmad_k0_opt0"           120 "$B/probe_ub2l1" capi-mmad 0 "$DUMPS/capi_mmad" 0
  run "capi_mmad_k0_opt2"           120 "$B/probe_ub2l1" capi-mmad 0 "$DUMPS/capi_mmad_opt2" 2
  run "capi_mmad_k0_opt0"           120 "$B/probe_ub2l1" capi-mmad 0 "$DUMPS/capi_mmad_opt0" 0
  run "basic_mmad_k0_default"       120 "$B/probe_ub2l1" basic-mmad 0 "$DUMPS/basic_mmad_default"
  run "basic_mmad_k0_ssbuf"         120 "$B/probe_ub2l1_ssbuf" basic-mmad 0 "$DUMPS/basic_mmad_ssbuf"
  run "capi_bw_64k_300"             120 "$B/probe_ub2l1" capi-bw 65536 300
  run "basic_bw_64k_300_default"    120 "$B/probe_ub2l1" basic-bw 65536 300
  run "basic_bw_64k_300_ssbuf"      120 "$B/probe_ub2l1_ssbuf" basic-bw 65536 300
  run "capi_bw_1k_2k"               120 "$B/probe_ub2l1" capi-bw 1024 2000
  run "capi_bw_8k_1k"               120 "$B/probe_ub2l1" capi-bw 8192 1000
fi
echo "== retake 结束 at=$(date -Is) =="
