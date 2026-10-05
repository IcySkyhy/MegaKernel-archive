#!/usr/bin/env bash
# rerun_r2.sh —— M134 r1 复审整改用的取数驱动（每档由 run_probe.sh 各自进锁；本脚本只串起档位）
# 用法：bash tools/rerun_r2.sh
set -u
cd "$(dirname "$0")/.."
run() { echo "=== [$(date -Is)] $* ==="; "$@"; }
# 1) N=16 全变体（覆盖 §2 表；含 c_preset/c_m2sameid/c_m2mte3，补其 N=16 证据）
run env N=16 REPS=3 TO=20 LOCKWAIT=300 bash run_probe.sh
# 2) 逐轮打印（定位挂死点）：base / a_nowait / c_m2mte3 / c_m2setonly
run env TRACE=1 TAG=_trace N=2 REPS=2 TO=20 LOCKWAIT=300 VARIANTS="base a_nowait c_m2mte3 c_m2setonly" bash run_probe.sh
run env TRACE=1 TAG=_trace N=4 REPS=2 TO=20 LOCKWAIT=300 VARIANTS="base a_nowait c_m2mte3 c_m2setonly" bash run_probe.sh
run env TRACE=1 TAG=_trace N=16 REPS=1 TO=20 LOCKWAIT=300 VARIANTS="base c_m2setonly" bash run_probe.sh
# 3) N=4 内容读数（校验 README 里 N=4 的 rc 声明）
run env N=4 REPS=3 TO=20 LOCKWAIT=300 VARIANTS="base dbg_rdyonly c_mode4free c_m2canon" bash run_probe.sh
# 4) N=32/64 关键档刷新（保持与提交源码一致）
run env N=32 REPS=3 TO=20 LOCKWAIT=300 VARIANTS="base c_mode4free dbg_nosync dbg_rdyonly c_m2canon" bash run_probe.sh
run env N=64 REPS=3 TO=20 LOCKWAIT=300 VARIANTS="base c_mode4free dbg_nosync dbg_rdyonly c_m2canon" bash run_probe.sh
echo "=== ALL DONE $(date -Is) ==="
