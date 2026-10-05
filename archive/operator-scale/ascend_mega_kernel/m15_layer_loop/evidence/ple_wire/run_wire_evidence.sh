#!/usr/bin/env bash
# M100 证据采集：PLE 段接进融合 kernel 挂载点的读数（每一步之前查 NPU 占用，见塔的环境硬纪律 A）
#
# **本批（M111，二进制 e5971447…）是整脚本放在设备进程锁里跑的**（单次峰值 host RSS ≈15.6 GiB、
# cgroup 上限 32 GiB ⇒ 绝不并发）：
#   flock -w 900 /tmp/npu0.lock bash m15_layer_loop/evidence/ple_wire/run_wire_evidence.sh
# 锁内只放设备那一段（编译与日志整理在锁外）。下面 `npu_guard` 的 `npu-smi` 巡检**保留**作为
# 第二道网：用锁采集时它不会等待 ⇒ 本批没有 `guard.log`（更早那批不是锁内跑的，才留下它）。
#
# ⚠ 本脚本产出的 `*.log` 会被**覆盖**：跑之前先确认上游分支要不要留旧读数（旧批次可从
#   `git show <commit>:m15_layer_loop/evidence/ple_wire/<log>` 取回）。
#
# 用法： bash m15_layer_loop/evidence/ple_wire/run_wire_evidence.sh
set -u
# 仓库根 = 本脚本所在目录的上**三层**（`<repo>/m15_layer_loop/evidence/ple_wire/`）⇒ 从任何
# checkout 复跑都指向**它自己**那份源码，不钉任何固定的 worktree 路径（本队第六变体的同族要求）。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
[ -f "$REPO/m15_layer_loop/m15_layer_loop.asc" ] || { echo "[FAIL] 推不出仓库根：$REPO"; exit 1; }
cd "$REPO" || exit 1
BIN=./m15_layer_loop/build/m15_layer_loop
MAN=m15_layer_loop/weights_manifest.txt
OUT=m15_layer_loop/evidence/ple_wire

npu_guard() {
  # 纪律 A：跑之前必须确认 NPU 上没有别的同类进程（单次峰值 host RSS 15.6 GiB，两并发即 OOM）。
  # 最多等 30 轮 × 30s；仍被占用则放弃本次采集（绝不并发）。
  local i=0
  while ! npu-smi info 2>&1 | grep -q "No running processes found in NPU 0"; do
    i=$((i + 1))
    if [ "$i" -ge 30 ]; then
      echo "[guard] $(date -u +%FT%TZ) 等待 30 轮后仍被占用 → 放弃本次采集" | tee -a "$OUT"/guard.log
      return 1
    fi
    echo "[guard] $(date -u +%FT%TZ) NPU 0 被占用（第 $i 轮），等 30s（纪律 A）" | tee -a "$OUT"/guard.log
    sleep 30
  done
  return 0
}

run() {   # run <logname> <env...> -- <argv...>
  local name="$1"; shift
  npu_guard || return 1
  echo "=== $(date -u +%FT%TZ)  $name : env $* ==="
  env "$@" "$BIN" "$MAN" plewire > "$OUT/$name.log" 2>&1
  echo "rc=$?" >> "$OUT/$name.log"
  grep -E "^\[m15\]   Pw\.|^\[m15\] 验证 Pw|rc=" "$OUT/$name.log"
}

echo "bin sha256: $(sha256sum "$BIN" | cut -d' ' -f1)" | tee "$OUT/binary_sha256.txt"

# A：②③④⑤ 独跑（① 关，ids 由 host 参考灌入）—— 窗口模式 + 真实分片切片 20 MiB
run A_body_stage15 M15_PLE_WIRE=1 M15_PLE_STAGE=15
# B：① + ②③④⑤ 全跑（暴露 ①→② 的落盘可见性问题）
run B_full_stage31 M15_PLE_WIRE=1 M15_PLE_STAGE=31
# C：接线关（同一断点结构上空操作）—— 一条负向对照
run C_wire_off M15_PLE_WIRE=0 M15_PLE_STAGE=15
# D：实参错位 1（表基址接到 scratch slab）
run D_miswire_table M15_PLE_WIRE=1 M15_PLE_STAGE=15 M15_PLE_MISWIRE=1
# E：实参错位 2（行基址偏 1）
run E_miswire_winbase M15_PLE_WIRE=1 M15_PLE_STAGE=15 M15_PLE_MISWIRE=2
# F：真实词表档 + ① 全跑（含 U-A 的交接问题）
run F_real_vocab_full M15_PLE_WIRE=1 M15_PLE_STAGE=31 M15_PLE_VOCAB_REAL=1
# F2：真实词表档 + ② 独跑（ids 由 host 参考灌入）⇒ 干净的「单窗口装不下真实 id」读数（miss=16）
run F2_real_vocab_body M15_PLE_WIRE=1 M15_PLE_STAGE=15 M15_PLE_VOCAB_REAL=1

# G：runs=all 零回归（接线默认关）。**基线数 = 2095 + 302**（当前 main 的源码上；M100 那批记的
#   2068+290 出自不含 M98 attention-KV 判据的旧分支，见 WITNESS.md §3.5 / M111_LANDING_PATH.md §3.6）。
if npu_guard; then
  echo "=== $(date -u +%FT%TZ)  runs=all（接线未开）==="
  M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 "$BIN" "$MAN" all > "$OUT/runs_all.log" 2>&1
  echo "rc=$?" >> "$OUT/runs_all.log"
  tail -6 "$OUT/runs_all.log"
fi
