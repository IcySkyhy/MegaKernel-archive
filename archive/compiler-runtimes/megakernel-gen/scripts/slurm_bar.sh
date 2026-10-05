#!/bin/bash
#SBATCH --job-name=mkgen-bar
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0:15:00
# A megakernel pays for one grid-wide rendezvous per stage per layer -- 170 of
# them on a 28-layer model.  This asks whether cooperative groups is the
# cheapest way to have one.
set -u
cd "${MKGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source env.sh
export CUDA_VISIBLE_DEVICES=0 MKGEN_GPU_AUTO=0
nvcc -O3 -arch=sm_90a -std=c++17 -Iruntime/include -o $MKGEN_WORK/build/bar_probe calib/bar_probe.cu || exit 1
$MKGEN_WORK/build/bar_probe 2000
echo "== done"
