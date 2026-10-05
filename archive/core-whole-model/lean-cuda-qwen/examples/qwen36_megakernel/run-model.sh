#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

example_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$example_dir/../../scripts/backend-env.sh"
cd "$example_dir"
export LEAN_PATH="$example_dir"

"$stage1/bin/lean" --root="$example_dir" \
  -o Qwen36Checkpoint.olean -i Qwen36Checkpoint.ilean -c Qwen36Checkpoint.c \
  Qwen36Checkpoint.lean
"$stage1/bin/lean" --root="$example_dir" \
  -o Qwen36Architecture.olean -i Qwen36Architecture.ilean -c Qwen36Architecture.c \
  Qwen36Architecture.lean
"$stage1/bin/lean" --root="$example_dir" \
  -o Qwen36ModelMegakernel.olean -i Qwen36ModelMegakernel.ilean \
  -c Qwen36ModelMegakernel.c --cuda=Qwen36ModelMegakernel.cu \
  -Dcompiler.postponeCompile=false Qwen36ModelMegakernel.lean

grep -Fq 'lean_cuda_l_Qwen36Model_qwen36Model_kernel' Qwen36ModelMegakernel.cu
grep -Fq 'lean_cuda_l_Qwen36Model_initializeModelState_kernel' Qwen36ModelMegakernel.cu
grep -Fq 'lean_cuda_grid_persistent_claim' Qwen36ModelMegakernel.cu
grep -Fq 'lean_cuda_grid_sync' Qwen36ModelMegakernel.cu
grep -Fq 'lean_cuda_fexp' Qwen36ModelMegakernel.cu
grep -Fq 'lean_cuda_frsqrt' Qwen36ModelMegakernel.cu
! grep -Fiq 'Tyr' Qwen36ModelMegakernel.lean

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Skipping CUDA compilation: nvcc not found (set CUDA_HOME or put CUDA 13 on PATH)"
  exit 0
fi
if ! nvcc --version | grep -Fq 'release 13.'; then
  echo "Skipping CUDA compilation: CUDA 13 is required"
  exit 0
fi

nvcc -std=c++17 -arch="$cuda_arch" -rdc=true -Xptxas=-v -DLEAN_CUDA_HOST_LAUNCHERS \
  -I"$LEAN_CUDA_ROOT/src/include" -dc Qwen36ModelMegakernel.cu \
  -o Qwen36ModelMegakernel.cuda.o
nvcc -std=c++17 -arch="$cuda_arch" -rdc=true -Xcompiler=-fvisibility=hidden \
  -I"$LEAN_CUDA_ROOT/src/include" -dc "$LEAN_CUDA_ROOT/src/runtime/cuda_device.cu" \
  -o cuda_device.o
objcopy --localize-hidden cuda_device.o
objcopy --wildcard --globalize-symbol="__fatbinwrap_*" cuda_device.o
nvcc -arch="$cuda_arch" -dlink Qwen36ModelMegakernel.cuda.o cuda_device.o \
  -o Qwen36ModelMegakernel.dlink.o

"$stage1/bin/leanc" -c Qwen36Checkpoint.c -o Qwen36Checkpoint.o
"$stage1/bin/leanc" -c Qwen36Architecture.c -o Qwen36Architecture.o
"$stage1/bin/leanc" -c Qwen36ModelMegakernel.c -o Qwen36ModelMegakernel.o
"$stage1/bin/leanc" Qwen36Checkpoint.o Qwen36Architecture.o Qwen36ModelMegakernel.o \
  Qwen36ModelMegakernel.cuda.o cuda_device.o Qwen36ModelMegakernel.dlink.o \
  -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lcudart -lcudadevrt \
  -o qwen36_model_megakernel

if [[ ${QWEN36_COMPILE_ONLY:-0} == 1 ]]; then
  echo "Lean Qwen3.6 real-checkpoint 64-layer inference megakernel compiled"
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1 ||
    ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -Fxq "$compute_capability"; then
  echo "Skipping CUDA execution: a $cuda_arch GPU is required"
  exit 0
fi
if [[ -z ${QWEN36_MODEL_DIR:-} ]]; then
  echo "Set QWEN36_MODEL_DIR to the Qwen/Qwen3.6-27B snapshot" >&2
  exit 1
fi

./qwen36_model_megakernel

if [[ ${LEAN_CUDA_COMPUTE_SANITIZER:-0} == 1 ]] &&
    command -v compute-sanitizer >/dev/null 2>&1; then
  compute-sanitizer --tool memcheck --error-exitcode 1 ./qwen36_model_megakernel
fi
