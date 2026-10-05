#!/usr/bin/env bash
# gen_commands.sh —— 生成 evidence/commands.txt（复现命令 + 变体清单 + 生成时工作树源码 sha256）
# 由 run_probe.sh 调用；也可单独跑（零设备）以在最终改动后刷新 sha。
set -u
cd "$(dirname "$0")/.."
EV=evidence
mkdir -p "$EV"
{
    echo "# probe_crosscore_tail 证据生成记录（M134）"
    echo "date            : $(date -Is)"
    echo "host            : $(uname -srm)"
    echo "本文件由 tools/gen_commands.sh 生成；sha256 段始终反映**生成时的工作树源码**"
    echo
    echo "## 复现命令（本目录执行；每档各自进锁 flock -w 300 + 进锁先 npu-smi + 锁内 timeout）"
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4"
    echo "N=16 REPS=3 bash run_probe.sh                                  # N=16 全变体（默认 VARIANTS）"
    echo "N=32 REPS=3 VARIANTS=\"base c_mode4free dbg_nosync dbg_rdyonly c_m2canon\" bash run_probe.sh"
    echo "N=64 REPS=3 VARIANTS=\"base c_mode4free dbg_nosync dbg_rdyonly c_m2canon\" bash run_probe.sh"
    echo "N=4  REPS=3 VARIANTS=\"base dbg_rdyonly c_mode4free c_m2canon\" bash run_probe.sh"
    echo "TRACE=1 TAG=_trace N=4 REPS=2 VARIANTS=\"base a_nowait c_m2mte3 c_m2setonly\" bash run_probe.sh"
    echo "bash tools/summarize.sh                                        # 从逐次日志机械重算 matrix_N<N>.txt"
    echo
    echo "## 变体清单（默认 VARIANTS）"
    echo "base a_nowait b_via_gm c_rotate c_wait1 c_broadcast d_no_acquire dbg_rdyonly dbg_nosync c_mode4free"
    echo "c_m2setonly neg_nofree neg_wrongval c_preset c_m2sameid c_m2mte3 dbg_nodata dbg_nobar c_mode4full c_m2canon"
    echo
    echo "## 源码 sha256（生成时工作树）"
    sha256sum probe_crosscore_tail.asc CMakeLists.txt run_probe.sh tools/gen_commands.sh tools/summarize.sh tools/rerun_r2.sh 2>/dev/null
} > "$EV/commands.txt"
echo "wrote $EV/commands.txt"
