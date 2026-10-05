#!/bin/bash
#SBATCH --job-name=mkgen-calib
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0:40:00
# The machine model, on an exclusive GPU.  This is the ONLY measurement the
# compiler depends on, so it must not be taken on a shared card: a neighbour at
# 50% occupancy moves every number in it.
set -u
cd "${MKGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source env.sh
export CUDA_VISIBLE_DEVICES=0 MKGEN_GPU_AUTO=0
OUT=$MKGEN_WORK/hw.json
echo "== node $(hostname)"; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
nvcc -O3 -arch=sm_90a -std=c++17 -Iruntime/include -o $MKGEN_WORK/build/calib calib/calib.cu || exit 1
$MKGEN_WORK/build/calib "$OUT" 40
python3 - "$OUT" <<'PY'
import json, sys, collections
h = json.load(open(sys.argv[1]))
print("schema", h.get("schema"), h["device"], "stream_peak", h["stream_peak_gbs"])
print("gemv ", dict(collections.Counter(g["quant"] for g in h["gemv"])))
print("stage", dict(collections.Counter(g["quant"] for g in h["stage"])))
print("barrier", h["barrier_us"])
PY
echo "== done -> $OUT  (copy into hw/ from a login node)"
