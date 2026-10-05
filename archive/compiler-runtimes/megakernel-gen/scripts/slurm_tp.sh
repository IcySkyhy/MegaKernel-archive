#!/bin/bash
#SBATCH --job-name=mkgen-tp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=0:30:00
# What a tensor-parallel megakernel would cost, before building one.  At batch 1
# the bytes exchanged are kilobytes; the question is the latency of the
# rendezvous, twice per layer.
set -u
cd "${MKGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source env.sh
export MKGEN_GPU_AUTO=0
echo "== node $(hostname)"; nvidia-smi --query-gpu=index,name --format=csv,noheader
nvidia-smi topo -m 2>/dev/null | head -8
nvcc -O3 -arch=sm_90a -std=c++17 -o $MKGEN_WORK/build/tp_probe calib/tp_probe.cu || exit 1
for W in 2 4; do
  echo "-- world $W"
  $MKGEN_WORK/build/tp_probe $W 200
done

echo "== single-GPU grid barrier: is cooperative-groups leaving anything on the table?"
nvcc -O3 -arch=sm_90a -std=c++17 -o $MKGEN_WORK/build/bar_probe calib/bar_probe.cu || exit 1
CUDA_VISIBLE_DEVICES=0 $MKGEN_WORK/build/bar_probe 2000
echo "== done"
