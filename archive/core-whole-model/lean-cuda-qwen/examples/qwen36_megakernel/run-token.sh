#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

example_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$example_dir/../../scripts/backend-env.sh"
cd "$example_dir"

"$stage1/bin/lean" --root="$LEAN_CUDA_QWEN_ROOT" -c Qwen36TokenMegakernel.c \
  --cuda=Qwen36TokenMegakernel.cu -Dcompiler.postponeCompile=false Qwen36TokenMegakernel.lean

grep -Fq 'lean_cuda_l_Qwen36Token_qwen36Token_kernel' Qwen36TokenMegakernel.cu
grep -Fq 'lean_cuda_l_Qwen36Token_qwen36TokenStep_kernel' Qwen36TokenMegakernel.cu
grep -Fq 'lean_cuda_l_Qwen36Token_initializeTokenInputs_kernel' Qwen36TokenMegakernel.cu
grep -Fq 'lean_cuda_grid_persistent_claim' Qwen36TokenMegakernel.cu
grep -Fq 'lean_cuda_grid_sync' Qwen36TokenMegakernel.cu
grep -Fq 'lean_cuda_atomic_min_u32' Qwen36TokenMegakernel.cu
grep -Fq 'lean_cuda_fexp2_fma' Qwen36TokenMegakernel.cu
grep -Fq 'lean_cuda_flog' Qwen36TokenMegakernel.cu
grep -Fq 'lean_cuda_frsqrt' Qwen36TokenMegakernel.cu
! grep -Fq 'lean_alloc_ctor' Qwen36TokenMegakernel.cu

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Skipping CUDA compilation: nvcc not found (set CUDA_HOME or put CUDA 13 on PATH)"
  exit 0
fi
if ! nvcc --version | grep -Fq 'release 13.'; then
  echo "Skipping CUDA compilation: CUDA 13 is required"
  exit 0
fi

nvcc -std=c++17 -arch="$cuda_arch" -rdc=true -Xptxas=-v -DLEAN_CUDA_HOST_LAUNCHERS \
  -I"$LEAN_CUDA_ROOT/src/include" -dc Qwen36TokenMegakernel.cu -o Qwen36TokenMegakernel.cuda.o
nvcc -std=c++17 -arch="$cuda_arch" -rdc=true -Xcompiler=-fvisibility=hidden \
  -I"$LEAN_CUDA_ROOT/src/include" -dc "$LEAN_CUDA_ROOT/src/runtime/cuda_device.cu" \
  -o cuda_device.o
objcopy --localize-hidden cuda_device.o
objcopy --wildcard --globalize-symbol="__fatbinwrap_*" cuda_device.o
nvcc -arch="$cuda_arch" -dlink Qwen36TokenMegakernel.cuda.o cuda_device.o \
  -o Qwen36TokenMegakernel.dlink.o

"$stage1/bin/leanc" -c Qwen36TokenMegakernel.c -o Qwen36TokenMegakernel.o
"$stage1/bin/leanc" Qwen36TokenMegakernel.o Qwen36TokenMegakernel.cuda.o cuda_device.o \
  Qwen36TokenMegakernel.dlink.o -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" \
  -lcudart -lcudadevrt -o qwen36_token_megakernel

if ! command -v nvidia-smi >/dev/null 2>&1 ||
    ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -Fxq "$compute_capability"; then
  echo "Skipping CUDA execution: a $cuda_arch GPU is required"
  exit 0
fi

./qwen36_token_megakernel

if [[ ${LEAN_CUDA_COMPUTE_SANITIZER:-0} == 1 ]] &&
    command -v compute-sanitizer >/dev/null 2>&1; then
  compute-sanitizer --tool memcheck --error-exitcode 1 ./qwen36_token_megakernel
fi
