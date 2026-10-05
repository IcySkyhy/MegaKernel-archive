#!/usr/bin/env bash
# M101 r2 的设备档（**一次进锁一条短命令**：按 mode 分开，别把四档压在一次持锁里）。
# 用法（在仓库/worktree 根）：
#   flock -w 300 /tmp/npu0.lock bash m15_layer_loop/evidence/attn_core/run_r2_device.sh core
#   flock -w 300 /tmp/npu0.lock bash m15_layer_loop/evidence/attn_core/run_r2_device.sh oproj
#   flock -w 300 /tmp/npu0.lock bash m15_layer_loop/evidence/attn_core/run_r2_device.sh negkv
# 锁内先 `npu-smi info` 复查；单进程；每条 `timeout`；rc 写进日志尾部。
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
EV=m15_layer_loop/evidence/attn_core
LOGS=$EV/logs
BIN=./m15_layer_loop/build/m15_attn_core
MODE="${1:-}"
mkdir -p "$LOGS"

echo "=== [锁内复查] npu-smi info ==="
npu-smi info 2>&1 | sed -n '1,20p'
echo "=== [锁内复查] df -h / ==="
df -h / | tail -1

run() {   # run <outdir> <mode> <cases> <logfile>
  local out="$1" mode="$2" cases="$3" log="$4"
  mkdir -p "$out"
  rm -f "$out"/*.bin
  M15AC_CASES="$cases" \
  M15AC_OUT="$ROOT/$out" \
  M15AC_REFDIR="$ROOT/m10_attn_decode/data" \
    timeout 280 "$BIN" "$mode" > "$log" 2>&1
  local rc=$?
  echo "rc=$rc" >> "$log"
  echo "[run] mode=$mode cases=$cases out=$out rc=$rc log=$log"
  return 0
}

case "$MODE" in
  core)
    # 第 9 类（RegBase 替换）的**等价性见证**：与归档 dump 逐字节对拍
    run "$EV/out" core 256,300,4096 "$LOGS/m101_core_run_20260927.log"
    ;;
  oproj)
    # 复审 F1：契约档 y + 指纹；并复跑一次比对设备确定性
    run "$EV/out_oproj"     oproj 1,3 "$LOGS/m101_oproj_run_20260927.log"
    run "$EV/out_oproj_rep" oproj 1,3 "$LOGS/m101_oproj_rerun_20260927.log"
    for m in 1 3; do
      for f in tdev yA_contract yA_kminus1 yC_contract yC_kminus1 attn gate tint; do
        a="$EV/out_oproj/m101op_m${m}_${f}.bin"; b="$EV/out_oproj_rep/m101op_m${m}_${f}.bin"
        if [ -f "$a" ] && [ -f "$b" ]; then
          cmp -s "$a" "$b" && echo "[det] m=$m $f 逐字节相同" || echo "[det] m=$m $f **不同**"
        fi
      done
    done
    ;;
  negkv)
    run "$EV/out_negkv" negkv 256,300 "$LOGS/m101_core_negkv_run_20260927.log"
    ;;
  *)
    echo "[usage] $0 {core|oproj|negkv}" >&2
    exit 2
    ;;
esac
echo "=== 设备档结束（mode=$MODE）==="
