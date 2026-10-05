#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
reference_dir="$script_dir/../cuda_qwen36_reference/fixtures/lora"
lean_test_root=${LEAN_TEST_ROOT:-"$repo_root/.lake/packages/LeanTest"}
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/lean-cuda-qwen36-lora.XXXXXX")
if [[ ${QWEN36_LORA_KEEP_BUILD:-0} == 1 ]]; then
  trap ':' EXIT
else
  trap 'rm -rf "$build_dir"' EXIT
fi

lean_bin=${LEAN:-"$(command -v lean)"}
leanc_bin=${LEANC:-"$(dirname "$lean_bin")/leanc"}
lean_libdir=$("$lean_bin" --print-libdir)
cp -as "$lean_libdir/Lean" "$build_dir/"
cp -as "$lean_libdir"/Lean.olean* "$build_dir/"
mkdir -p "$build_dir/LeanCudaQwen/Qwen36"
mkdir -p "$build_dir/LeanTest"
export LEAN_PATH="$build_dir:${LEAN_PATH:-$lean_libdir}"

if [[ ! -f "$lean_test_root/LeanTest.lean" ]]; then
  echo "Qwen3.6 LoRA suite requires LeanTest; set LEAN_TEST_ROOT" >&2
  exit 1
fi
for module in Basic Assert Attr Runner; do
  "$lean_bin" --root="$lean_test_root" -o "$build_dir/LeanTest/$module.olean" \
    -c "$build_dir/LeanTest.$module.c" "$lean_test_root/LeanTest/$module.lean"
done
"$lean_bin" --root="$lean_test_root" -o "$build_dir/LeanTest.olean" \
  -c "$build_dir/LeanTest.c" "$lean_test_root/LeanTest.lean"
"$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" \
  -o "$build_dir/LeanCudaQwen/Optimizer.olean" -c "$build_dir/Optimizer.c" \
  "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Optimizer.lean"


for module in Primitives Linear; do
  "$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" \
    -o "$build_dir/LeanCudaQwen/Qwen36/$module.olean" -c "$build_dir/$module.c" \
    --cuda="$build_dir/$module.cu" -Dcompiler.postponeCompile=false \
    "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/$module.lean"
done
"$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" \
  -o "$build_dir/LeanCudaQwen/Qwen36/LoRA.olean" -c "$build_dir/LoRA.c" \
  --cuda="$build_dir/LoRA.cu" -Dcompiler.postponeCompile=false \
  "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/LoRA.lean"
"$lean_bin" --root="$script_dir" -o "$build_dir/Cuda.olean" -c "$build_dir/Cuda.c" \
  "$script_dir/Cuda.lean"
"$lean_bin" --root="$script_dir" -o "$build_dir/Contract.olean" \
  -c "$build_dir/Contract.c" "$script_dir/Contract.lean"
"$lean_bin" --root="$script_dir" -o "$build_dir/Kernel.olean" -c "$build_dir/Kernel.c" \
  "$script_dir/Kernel.lean"
"$lean_bin" --root="$script_dir" -o "$build_dir/Driver.olean" -c "$build_dir/Driver.c" \
  "$script_dir/Driver.lean"

for symbol in initializeAdapterF32Kernel rankForwardF32Kernel forwardF32Kernel addForwardF32Kernel backwardInputF32Kernel \
    backwardAF32Kernel backwardBF32Kernel adamWF32Kernel; do
  grep -Fq "lean_cuda_l_Cuda_Qwen36_LoRA_${symbol}_kernel" "$build_dir/LoRA.cu"
done
if grep -Fq 'lean_apply_' "$build_dir/LoRA.cu" ||
    grep -Fq 'lean_alloc_ctor' "$build_dir/LoRA.cu"; then
  echo "Qwen3.6 LoRA retained boxed device code" >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Skipping Qwen3.6 LoRA CUDA compilation: nvcc is not installed"
  exit 0
fi
if ! nvcc --version | grep -Fq 'release 13.'; then
  echo "Skipping Qwen3.6 LoRA CUDA compilation: CUDA 13 is required"
  exit 0
fi

cuda_home=${CUDA_HOME:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}
for module in LeanTest.Basic LeanTest.Assert LeanTest.Attr LeanTest.Runner LeanTest \
    Optimizer Primitives Linear LoRA Cuda Contract Kernel Driver; do
  if [[ $module == LoRA || $module == Kernel ]]; then
    "$leanc_bin" -c "$build_dir/$module.c" -o "$build_dir/$module.o" \
      -DLEAN_EXPORTING
  else
    "$leanc_bin" -c "$build_dir/$module.c" -o "$build_dir/$module.o"
  fi
done
for module in Primitives Linear LoRA; do
  nvcc -std=c++17 -O3 -arch=sm_121a -rdc=true -DLEAN_CUDA_HOST_LAUNCHERS \
    -I"$LEAN_CUDA_ROOT/src/include" -dc "$build_dir/$module.cu" -o "$build_dir/$module.cuda.o"
done
nvcc -std=c++17 -arch=sm_121a -rdc=true -Xcompiler=-fvisibility=hidden \
  -I"$LEAN_CUDA_ROOT/src/include" -dc "$LEAN_CUDA_ROOT/src/runtime/cuda_device.cu" \
  -o "$build_dir/cuda_device.o"
objcopy --localize-hidden "$build_dir/cuda_device.o"
objcopy --wildcard --globalize-symbol='__fatbinwrap_*' "$build_dir/cuda_device.o"
nvcc -arch=sm_121a -dlink "$build_dir/Primitives.cuda.o" "$build_dir/Linear.cuda.o" \
  "$build_dir/LoRA.cuda.o" "$build_dir/cuda_device.o" -o "$build_dir/cuda_dlink.o"
"$leanc_bin" "$build_dir/Driver.o" "$build_dir/Kernel.o" "$build_dir/Contract.o" \
  "$build_dir/Cuda.o" "$build_dir/LeanTest.o" "$build_dir/LeanTest.Runner.o" \
  "$build_dir/LeanTest.Attr.o" "$build_dir/LeanTest.Assert.o" "$build_dir/LeanTest.Basic.o" \
  "$build_dir/LoRA.o" \
  "$build_dir/Linear.o" "$build_dir/Primitives.o" "$build_dir/Optimizer.o" "$build_dir/LoRA.cuda.o" \
  "$build_dir/Linear.cuda.o" "$build_dir/Primitives.cuda.o" "$build_dir/cuda_device.o" \
  "$build_dir/cuda_dlink.o" -rdynamic -L"$cuda_home/lib64" -Wl,-rpath,"$cuda_home/lib64" \
  -lcudart -lcudadevrt -o "$build_dir/cuda_qwen36_lora_test"

if ! command -v nvidia-smi >/dev/null 2>&1 ||
    ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -Fxq '12.1'; then
  echo "Skipping Qwen3.6 LoRA execution: an sm_121 GPU is required"
  exit 0
fi

QWEN36_LORA_FIXTURES="$reference_dir" \
  timeout 300 stdbuf -oL -eL "$build_dir/cuda_qwen36_lora_test"

if [[ ${LEAN_CUDA_COMPUTE_SANITIZER:-0} == 1 ]] &&
    command -v compute-sanitizer >/dev/null 2>&1; then
  QWEN36_LORA_FIXTURES="$reference_dir" \
    timeout 900 stdbuf -oL -eL compute-sanitizer --tool memcheck --error-exitcode 1 \
      "$build_dir/cuda_qwen36_lora_test"
fi
