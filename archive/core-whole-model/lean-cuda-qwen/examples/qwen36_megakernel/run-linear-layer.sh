#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

example_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$example_dir/../../scripts/backend-env.sh"
cd "$example_dir"

"$stage1/bin/lean" --root="$LEAN_CUDA_QWEN_ROOT" -c Qwen36LinearLayerMegakernel.c \
  --cuda=Qwen36LinearLayerMegakernel.cu -Dcompiler.postponeCompile=false \
  Qwen36LinearLayerMegakernel.lean

grep -Fq 'lean_cuda_l_Qwen36LinearLayer_qwen36LinearLayerToken_kernel' \
  Qwen36LinearLayerMegakernel.cu
grep -Fq 'lean_cuda_l_Qwen36LinearLayer_qwen36LinearPreprocess_kernel' \
  Qwen36LinearLayerMegakernel.cu
grep -Fq 'lean_cuda_l_Qwen36LinearLayer_qwen36LinearRecurrentPost_kernel' \
  Qwen36LinearLayerMegakernel.cu
grep -Fq 'lean_cuda_grid_persistent_claim' Qwen36LinearLayerMegakernel.cu
grep -Fq 'lean_cuda_grid_sync' Qwen36LinearLayerMegakernel.cu
grep -Fq 'lean_cuda_fexp' Qwen36LinearLayerMegakernel.cu
grep -Fq 'lean_cuda_flog' Qwen36LinearLayerMegakernel.cu
grep -Fq 'lean_cuda_frsqrt' Qwen36LinearLayerMegakernel.cu
! grep -Fq 'lean_alloc_ctor' Qwen36LinearLayerMegakernel.cu

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Skipping CUDA compilation: nvcc not found (set CUDA_HOME or put CUDA 13 on PATH)"
  exit 0
fi
if ! nvcc --version | grep -Fq 'release 13.'; then
  echo "Skipping CUDA compilation: CUDA 13 is required"
  exit 0
fi

nvcc -std=c++17 -arch="$cuda_arch" -rdc=true -Xptxas=-v -DLEAN_CUDA_HOST_LAUNCHERS \
  -I"$LEAN_CUDA_ROOT/src/include" -dc Qwen36LinearLayerMegakernel.cu \
  -o Qwen36LinearLayerMegakernel.cuda.o
nvcc -std=c++17 -arch="$cuda_arch" -rdc=true -Xcompiler=-fvisibility=hidden \
  -I"$LEAN_CUDA_ROOT/src/include" -dc "$LEAN_CUDA_ROOT/src/runtime/cuda_device.cu" \
  -o cuda_device.o
objcopy --localize-hidden cuda_device.o
objcopy --wildcard --globalize-symbol="__fatbinwrap_*" cuda_device.o
nvcc -arch="$cuda_arch" -dlink Qwen36LinearLayerMegakernel.cuda.o cuda_device.o \
  -o Qwen36LinearLayerMegakernel.dlink.o

"$stage1/bin/leanc" -c Qwen36LinearLayerMegakernel.c -o Qwen36LinearLayerMegakernel.o
"$stage1/bin/leanc" Qwen36LinearLayerMegakernel.o Qwen36LinearLayerMegakernel.cuda.o cuda_device.o \
  Qwen36LinearLayerMegakernel.dlink.o -L"$CUDA_HOME/lib64" \
  -Wl,-rpath,"$CUDA_HOME/lib64" -lcudart -lcudadevrt -o qwen36_linear_layer_megakernel

if ! command -v nvidia-smi >/dev/null 2>&1 ||
    ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -Fxq "$compute_capability"; then
  echo "Skipping CUDA execution: a $cuda_arch GPU is required"
  exit 0
fi

./qwen36_linear_layer_megakernel

if [[ ${LEAN_CUDA_COMPUTE_SANITIZER:-0} == 1 ]] &&
    command -v compute-sanitizer >/dev/null 2>&1; then
  compute-sanitizer --tool memcheck --error-exitcode 1 ./qwen36_linear_layer_megakernel
fi
