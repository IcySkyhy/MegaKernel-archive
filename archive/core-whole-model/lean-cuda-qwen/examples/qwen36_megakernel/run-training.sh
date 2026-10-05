#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

example_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$example_dir/../../scripts/backend-env.sh"
cd "$example_dir"

"$stage1/bin/lean" --root="$LEAN_CUDA_QWEN_ROOT" -c Qwen36TrainingMegakernel.c \
  --cuda=Qwen36TrainingMegakernel.cu -Dcompiler.postponeCompile=false \
  Qwen36TrainingMegakernel.lean

grep -Fq 'lean_cuda_l_Qwen36Training_qwen36Train_kernel' Qwen36TrainingMegakernel.cu
grep -Fq 'lean_cuda_l_Qwen36Training_qwen36TrainForwardStep_kernel' \
  Qwen36TrainingMegakernel.cu
grep -Fq 'lean_cuda_l_Qwen36Training_qwen36TrainBackwardStep_kernel' \
  Qwen36TrainingMegakernel.cu
grep -Fq 'lean_cuda_grid_persistent_claim' Qwen36TrainingMegakernel.cu
grep -Fq 'lean_cuda_atomic_add_f32' Qwen36TrainingMegakernel.cu
grep -Fq 'lean_cuda_fexp' Qwen36TrainingMegakernel.cu
grep -Fq 'lean_cuda_frsqrt' Qwen36TrainingMegakernel.cu
! grep -Fq 'lean_alloc_ctor' Qwen36TrainingMegakernel.cu

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Skipping CUDA compilation: nvcc not found (set CUDA_HOME or put CUDA 13 on PATH)"
  exit 0
fi
if ! nvcc --version | grep -Fq 'release 13.'; then
  echo "Skipping CUDA compilation: CUDA 13 is required"
  exit 0
fi

nvcc -std=c++17 -arch="$cuda_arch" -rdc=true -Xptxas=-v -DLEAN_CUDA_HOST_LAUNCHERS \
  -I"$LEAN_CUDA_ROOT/src/include" -dc Qwen36TrainingMegakernel.cu \
  -o Qwen36TrainingMegakernel.cuda.o
nvcc -std=c++17 -arch="$cuda_arch" -rdc=true -Xcompiler=-fvisibility=hidden \
  -I"$LEAN_CUDA_ROOT/src/include" -dc "$LEAN_CUDA_ROOT/src/runtime/cuda_device.cu" \
  -o cuda_device.o
objcopy --localize-hidden cuda_device.o
objcopy --wildcard --globalize-symbol="__fatbinwrap_*" cuda_device.o
nvcc -arch="$cuda_arch" -dlink Qwen36TrainingMegakernel.cuda.o cuda_device.o \
  -o Qwen36TrainingMegakernel.dlink.o

"$stage1/bin/leanc" -c Qwen36TrainingMegakernel.c -o Qwen36TrainingMegakernel.o
"$stage1/bin/leanc" Qwen36TrainingMegakernel.o Qwen36TrainingMegakernel.cuda.o cuda_device.o \
  Qwen36TrainingMegakernel.dlink.o -L"$CUDA_HOME/lib64" \
  -Wl,-rpath,"$CUDA_HOME/lib64" -lcudart -lcudadevrt -o qwen36_training_megakernel

if ! command -v nvidia-smi >/dev/null 2>&1 ||
    ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -Fxq "$compute_capability"; then
  echo "Skipping CUDA execution: a $cuda_arch GPU is required"
  exit 0
fi

./qwen36_training_megakernel

if [[ ${LEAN_CUDA_COMPUTE_SANITIZER:-0} == 1 ]] &&
    command -v compute-sanitizer >/dev/null 2>&1; then
  compute-sanitizer --tool memcheck --error-exitcode 1 ./qwen36_training_megakernel
fi
