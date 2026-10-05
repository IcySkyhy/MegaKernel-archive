#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
lean_test_root=${LEAN_TEST_ROOT:-"$repo_root/.lake/packages/LeanTest"}
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/lean-cuda-qwen36-weights.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT

lean_bin=${LEAN:-"$(command -v lean)"}
leanc_bin=${LEANC:-"$(dirname "$lean_bin")/leanc"}
lean_libdir=$($lean_bin --print-libdir)
cp -as "$lean_libdir/Lean" "$build_dir/"
cp -as "$lean_libdir"/Lean.olean* "$build_dir/"
mkdir -p "$build_dir/LeanCudaQwen/Qwen36" "$build_dir/LeanTest"
safetensors_olean="$build_dir/LeanCudaQwen/Qwen36/SafeTensors.olean"
if [[ -e $safetensors_olean || -L $safetensors_olean ]]; then
  unlink "$safetensors_olean"
fi
export LEAN_PATH="$build_dir:${LEAN_PATH:-$lean_libdir}"

if [[ ! -f "$lean_test_root/LeanTest.lean" ]]; then
  echo "Qwen3.6 weights suite requires LeanTest; set LEAN_TEST_ROOT" >&2
  exit 1
fi
for module in Basic Assert Attr Runner; do
  "$lean_bin" --root="$lean_test_root" -o "$build_dir/LeanTest/$module.olean" \
    -c "$build_dir/LeanTest.$module.c" "$lean_test_root/LeanTest/$module.lean"
done
"$lean_bin" --root="$lean_test_root" -o "$build_dir/LeanTest.olean" \
  -c "$build_dir/LeanTest.c" "$lean_test_root/LeanTest.lean"

# Shadow-compile the library module under test and the gate driver, then link a plain
# host binary: the CUDA host runtime dlopens libcudart, so no nvcc step is needed.
"$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" -o "$safetensors_olean" -c "$build_dir/SafeTensors.c" \
  "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/SafeTensors.lean"
"$lean_bin" --root="$script_dir" -o "$build_dir/Weights.olean" -c "$build_dir/Weights.c" \
  "$script_dir/Weights.lean"
"$lean_bin" --root="$script_dir" -o "$build_dir/Driver.olean" -c "$build_dir/Driver.c" \
  "$script_dir/Driver.lean"
"$leanc_bin" -c "$build_dir/SafeTensors.c" -o "$build_dir/SafeTensors.o"
"$leanc_bin" -c "$build_dir/Weights.c" -o "$build_dir/Weights.o" -DLEAN_EXPORTING
"$leanc_bin" -c "$build_dir/Driver.c" -o "$build_dir/Driver.o"
for module in LeanTest.Basic LeanTest.Assert LeanTest.Attr LeanTest.Runner LeanTest; do
  "$leanc_bin" -c "$build_dir/$module.c" -o "$build_dir/$module.o"
done
"$leanc_bin" "$build_dir/Driver.o" "$build_dir/Weights.o" "$build_dir/SafeTensors.o" \
  "$build_dir/LeanTest.o" "$build_dir/LeanTest.Runner.o" "$build_dir/LeanTest.Attr.o" \
  "$build_dir/LeanTest.Assert.o" "$build_dir/LeanTest.Basic.o" -rdynamic \
  -o "$build_dir/cuda_qwen36_weights_test"

run_device=0
if command -v nvidia-smi >/dev/null 2>&1 &&
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -Fxq '12.1'; then
  run_device=1
fi

if [[ -n ${QWEN36_STREAM_CHECKPOINT_DIR:-} ]]; then
  QWEN36_WEIGHTS_FIXTURES="$script_dir/fixtures" QWEN36_WEIGHTS_RUN_DEVICE="$run_device" \
    QWEN36_STREAM_CHECKPOINT_DIR="$QWEN36_STREAM_CHECKPOINT_DIR" \
    timeout 1800 "$build_dir/cuda_qwen36_weights_test"
else
  QWEN36_WEIGHTS_FIXTURES="$script_dir/fixtures" QWEN36_WEIGHTS_RUN_DEVICE="$run_device" \
    timeout 300 "$build_dir/cuda_qwen36_weights_test"
fi

if [[ ${LEAN_CUDA_COMPUTE_SANITIZER:-0} == 1 ]] &&
    command -v compute-sanitizer >/dev/null 2>&1; then
  QWEN36_WEIGHTS_FIXTURES="$script_dir/fixtures" QWEN36_WEIGHTS_RUN_DEVICE="$run_device" \
    timeout 600 compute-sanitizer --tool memcheck --error-exitcode 1 \
      "$build_dir/cuda_qwen36_weights_test"
fi
