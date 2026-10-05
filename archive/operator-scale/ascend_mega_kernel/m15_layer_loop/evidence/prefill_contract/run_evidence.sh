#!/usr/bin/env bash
# M110（Wave A）证据采集：读数的**可复跑**入口。
#
# 纪律（塔的环境硬纪律）：整脚本放在设备进程锁里跑，**绝不并发**；
#   flock -w 900 /tmp/npu0.lock bash m15_layer_loop/evidence/prefill_contract/run_evidence.sh
# 锁内每一步之前查一次 NPU 占用（`npu-smi`），被占用就等（最多 30 轮 × 30s）。
#
# 参数：$1 = 二进制路径（默认 ./m15_layer_loop/build/m15_layer_loop）
#       $2 = 日志前缀（默认 "new"）
# 用法（对照档：同一份 env/argv 打两个二进制，证明"本 mission 的改动零影响"）：
#   flock -w 900 /tmp/npu0.lock bash …/run_evidence.sh /tmp/m110_base_m15 base
#   flock -w 900 /tmp/npu0.lock bash …/run_evidence.sh ./m15_layer_loop/build/m15_layer_loop new
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
[ -f "$REPO/m15_layer_loop/m15_layer_loop.asc" ] || { echo "[FAIL] 推不出仓库根：$REPO"; exit 1; }
cd "$REPO" || exit 1

BIN="${1:-./m15_layer_loop/build/m15_layer_loop}"
TAG="${2:-new}"
MAN=m15_layer_loop/weights_manifest.txt
OUT=m15_layer_loop/evidence/prefill_contract

npu_guard() {
  local i=0
  # 判据取 **`npu-smi` 的空闲提示行**（"No running processes found in NPU 0"）——不能用
  # `grep "Process id"`：那个字符串在表头里**恒存在**，会把空闲误判成占用（本 mission 首跑踩过）。
  while ! npu-smi info 2>&1 | grep -q "No running processes found in NPU 0"; do
    i=$((i + 1))
    if [ "$i" -ge 30 ]; then
      echo "[guard] $(date -u +%FT%TZ) 等待 30 轮后 NPU 0 仍被占用 → 放弃" | tee -a "$OUT/guard_$TAG.log"
      return 1
    fi
    echo "[guard] $(date -u +%FT%TZ) NPU 0 被占用（第 $i 轮），等 30s" | tee -a "$OUT/guard_$TAG.log"
    sleep 30
  done
  return 0
}

run() {   # run <logname> <runsArg> <env...>
  local name="$1"; shift
  local runsArg="$1"; shift
  npu_guard || return 1
  echo "=== $(date -u +%FT%TZ)  $TAG/$name : env $* ; argv=$runsArg ==="
  env "$@" "$BIN" "$MAN" "$runsArg" > "$OUT/${TAG}_$name.log" 2>&1
  echo "rc=$?" >> "$OUT/${TAG}_$name.log"
  grep -E "^\[m15\] +Pf\.|^\[m15\]   Pf\.|^\[m15\] +Pw\.|^\[m15\]   Pw\.|^\[m15\] =====|^rc=" \
      "$OUT/${TAG}_$name.log"
}

sha256sum "$BIN" | cut -d' ' -f1 | sed "s/^/${TAG} binary sha256: /" | tee -a "$OUT/binary_sha256.txt"

# ---- M110 的主档（新档 runs=prefill）----
run pf_default prefill M15_PREFILL_WIRE=0
run pf_wired   prefill M15_PREFILL_WIRE=1
run pf_mutant  prefill M15_PREFILL_MUTANT=1
run pf_smallm  prefill M15_PREFILL_M=64,65
# ---- M100 的三档（runs=plewire；与本 mission 的改动无关，用来做"零影响"对照）----
run plewire_body  plewire M15_PLE_WIRE=1 M15_PLE_STAGE=15
run plewire_full  plewire M15_PLE_WIRE=1 M15_PLE_STAGE=31
run plewire_off   plewire M15_PLE_WIRE=0 M15_PLE_STAGE=15
# ---- 零回归：与基线**同一套 env**（M15_DUMP=1，基线就是带 dump 跑的）----
run all48 all M15_DUMP=1

# ---- 收尾：`runs=all M15_DUMP=1` 会把 dump 落在**仓库根**（1300+ 个 .bin + 3 个 layout txt）。
#      它们都是未跟踪的产物，测完即删（`git ls-files | grep -E '^[^/]+\.bin$'` 在 base 上是 0，
#      故这一句不会碰到任何入库文件）。
rm -f ./*.bin dump_manifest.txt moe_layout.txt hc_layout.txt
echo "=== $(date -u +%FT%TZ) done（仓库根 dump 已清理）==="
