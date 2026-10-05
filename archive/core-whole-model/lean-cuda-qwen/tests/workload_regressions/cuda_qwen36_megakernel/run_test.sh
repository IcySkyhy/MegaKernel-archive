#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
lean_test_root=${LEAN_TEST_ROOT:-"$repo_root/.lake/packages/LeanTest"}
source "$repo_root/scripts/qwen-cuda-profile.sh"
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/lean-cuda-qwen36-megakernel.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT

lean_bin=${LEAN:-"$(command -v lean)"}
leanc_bin=${LEANC:-"$(dirname "$lean_bin")/leanc"}
lean_libdir=$($lean_bin --print-libdir)
cp -as "$lean_libdir/Lean" "$build_dir/"
cp -as "$lean_libdir"/Lean.olean* "$build_dir/"
mkdir -p "$build_dir/LeanCudaQwen/Qwen36" "$build_dir/LeanTest"
export LEAN_PATH="$build_dir:${LEAN_PATH:-$lean_libdir}"

if [[ ! -f "$lean_test_root/LeanTest.lean" ]]; then
  echo "Qwen3.8 megakernel suite requires LeanTest; set LEAN_TEST_ROOT" >&2
  exit 1
fi
for module in Basic Assert Attr Runner; do
  "$lean_bin" --root="$lean_test_root" -o "$build_dir/LeanTest/$module.olean" \
    -c "$build_dir/LeanTest.$module.c" "$lean_test_root/LeanTest/$module.lean"
done
"$lean_bin" --root="$lean_test_root" -o "$build_dir/LeanTest.olean" \
  -c "$build_dir/LeanTest.c" "$lean_test_root/LeanTest.lean"
"$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" \
  -o "$build_dir/LeanCudaQwen/Foundation.olean" -c "$build_dir/Foundation.c" \
  "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Foundation.lean"
"$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" \
  -o "$build_dir/LeanCudaQwen/Optimizer.olean" -c "$build_dir/Optimizer.c" \
  "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Optimizer.lean"



"$lean_bin" --root="$LEAN_CUDA_ROOT/src" -o "$build_dir/Lean/Cuda/Mailbox.olean" \
  -c "$build_dir/Mailbox.c" --cuda="$build_dir/Mailbox.cu" \
  -Dcompiler.postponeCompile=false "$LEAN_CUDA_ROOT/src/Lean/Cuda/Mailbox.lean"

for module in Config SafeTensors; do
  "$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" \
    -o "$build_dir/LeanCudaQwen/Qwen36/$module.olean" -c "$build_dir/$module.c" \
    "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/$module.lean"
done
for module in Primitives Linear LoRA; do
  "$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" \
    -o "$build_dir/LeanCudaQwen/Qwen36/$module.olean" -c "$build_dir/$module.c" \
    --cuda="$build_dir/$module.cu" -Dcompiler.postponeCompile=false \
    "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/$module.lean"
done
"$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" -o "$build_dir/LeanCudaQwen/Qwen36/Projection.olean" \
  -c "$build_dir/Projection.c" "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/Projection.lean"
for module in DeltaNet Attention; do
  "$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" \
    -o "$build_dir/LeanCudaQwen/Qwen36/$module.olean" -c "$build_dir/$module.c" \
    --cuda="$build_dir/$module.cu" -Dcompiler.postponeCompile=false \
    "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/$module.lean"
done
"$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" -o "$build_dir/LeanCudaQwen/Qwen36/MLP.olean" \
  -c "$build_dir/MLP.c" "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/MLP.lean"
for module in Model Training FullTrainingMegakernelCore FullPretrainMegakernel \
    FullGRPOMegakernel Megakernel; do
  "$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" \
    -o "$build_dir/LeanCudaQwen/Qwen36/$module.olean" -c "$build_dir/$module.c" \
    --cuda="$build_dir/$module.cu" -Dcompiler.postponeCompile=false \
    "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/$module.lean"
done
"$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" -o "$build_dir/LeanCudaQwen/Qwen36/CheckpointLoRA.olean" \
  -c "$build_dir/CheckpointLoRA.c" "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/CheckpointLoRA.lean"
"$lean_bin" --root="$script_dir" -o "$build_dir/Cuda.olean" -c "$build_dir/Cuda.c" \
  "$script_dir/Cuda.lean"
"$lean_bin" --root="$script_dir" -o "$build_dir/Kernel.olean" -c "$build_dir/Kernel.c" \
  "$script_dir/Kernel.lean"
"$lean_bin" --root="$script_dir" -o "$build_dir/Driver.olean" -c "$build_dir/Driver.c" \
  "$script_dir/Driver.lean"

grep -Fq 'lean_cuda_grid_persistent_claim' "$build_dir/Megakernel.cu"
grep -Fq 'lean_cuda_grid_sync' "$build_dir/Megakernel.cu"
grep -Fq 'lean_cuda_l_Cuda_Qwen36_Megakernel_qwen36TrainingModel_kernel' "$build_dir/Megakernel.cu"
grep -Fq 'lean_cuda_l_Cuda_Qwen36_Megakernel_qwen36InferenceModel_kernel' "$build_dir/Megakernel.cu"
if grep -Fq 'lean_apply_' "$build_dir/Megakernel.cu" ||
    grep -Fq 'lean_alloc_ctor' "$build_dir/Megakernel.cu"; then
  echo "Qwen3.8 megakernel retained boxed device code" >&2
  exit 1
fi
if ! grep -Eq 'lean_cuda_load_u64\(_lean_cuda_state, v_input_[0-9]+_' \
    "$build_dir/Megakernel.cu"; then
  echo "Qwen3.8 projection loop did not retain packed BF16 input loads" >&2
  exit 1
fi
if ! grep -Eq 'lean_cuda_load_read_only_u64\(_lean_cuda_state, v_weight_[0-9]+_' \
    "$build_dir/Megakernel.cu"; then
  echo "Qwen3.8 projection loop did not retain read-only packed BF16 weight loads" >&2
  exit 1
fi
if ! grep -Fq 'dotProjectionBatch2RowStrided' "$build_dir/Megakernel.cu"; then
  echo "Qwen3.8 integrated prefill omitted the two-row projection body" >&2
  exit 1
fi
for pointer in firstInput secondInput; do
  if ! grep -Eq "lean_cuda_load_u64\(_lean_cuda_state, v_${pointer}_[0-9]+_" \
      "$build_dir/Megakernel.cu"; then
    echo "Qwen3.8 two-row prefill omitted packed ${pointer} loads" >&2
    exit 1
  fi
done
for pointer in firstWeight secondWeight; do
  if ! grep -Eq "lean_cuda_load_read_only_u64\\(_lean_cuda_state, v_${pointer}_[0-9]+_" \
      "$build_dir/Megakernel.cu"; then
    echo "Qwen3.8 paired projection loop did not retain read-only ${pointer} loads" >&2
    exit 1
  fi
done
if ! grep -Fq 'lean_cuda_load_read_only_u64' "$build_dir/Megakernel.cu"; then
  echo "Qwen3.8 resolved weight table did not retain read-only pointer loads" >&2
  exit 1
fi
if ! grep -Fq 'lean_cuda_buffer_table_create_at_byte_offsets' "$build_dir/Megakernel.c"; then
  echo "Qwen3.8 host path did not pre-resolve checkpoint weight addresses" >&2
  exit 1
fi
for pointer in firstScales secondScales; do
  if ! grep -Eq "lean_cuda_load_read_only_u8\\(_lean_cuda_state, v_${pointer}_[0-9]+_" \
      "$build_dir/Megakernel.cu"; then
    echo "Qwen3.8 paired MXFP8 projection loop did not retain read-only ${pointer} loads" >&2
    exit 1
  fi
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Skipping Qwen3.8 megakernel CUDA compilation: nvcc is not installed"
  exit 0
fi
if ! nvcc --version | grep -Fq 'release 13.'; then
  echo "Skipping Qwen3.8 megakernel CUDA compilation: CUDA 13 is required"
  exit 0
fi

cuda_home=${CUDA_HOME:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}
for module in LeanTest.Basic LeanTest.Assert LeanTest.Attr LeanTest.Runner LeanTest \
    Foundation Optimizer Config SafeTensors Primitives Linear LoRA Projection DeltaNet Attention \
    MLP Model Training FullTrainingMegakernelCore FullPretrainMegakernel FullGRPOMegakernel \
    Megakernel CheckpointLoRA Cuda Kernel Driver; do
  if [[ $module == Kernel ]]; then
    "$leanc_bin" -c "$build_dir/$module.c" -o "$build_dir/$module.o" -DLEAN_EXPORTING
  else
    "$leanc_bin" -c "$build_dir/$module.c" -o "$build_dir/$module.o"
  fi
done
for module in Mailbox Primitives Linear LoRA DeltaNet Attention Model Training \
    FullTrainingMegakernelCore FullPretrainMegakernel FullGRPOMegakernel Megakernel; do
  cuda_flags=(-std=c++17 -O3 "-arch=$qwen_cuda_arch" -rdc=true)
  if [[ $module == Mailbox || $module == Primitives || $module == Megakernel ]]; then
    cuda_flags=("${qwen_cuda_device_flags[@]}")
  fi
  nvcc "${cuda_flags[@]}" -DLEAN_CUDA_HOST_LAUNCHERS \
    -I"$LEAN_CUDA_ROOT/src/include" -dc "$build_dir/$module.cu" -o "$build_dir/$module.cuda.o"
done
nvcc "${qwen_cuda_device_flags[@]}" -Xcompiler=-fvisibility=hidden \
  -I"$LEAN_CUDA_ROOT/src/include" -dc "$LEAN_CUDA_ROOT/src/runtime/cuda_device.cu" \
  -o "$build_dir/cuda_device.o"
objcopy --localize-hidden "$build_dir/cuda_device.o"
objcopy --wildcard --globalize-symbol='__fatbinwrap_*' "$build_dir/cuda_device.o"
nvcc "${qwen_cuda_dlink_flags[@]}" -dlink \
  "$build_dir/Mailbox.cuda.o" "$build_dir/Primitives.cuda.o" \
  "$build_dir/Linear.cuda.o" "$build_dir/LoRA.cuda.o" "$build_dir/DeltaNet.cuda.o" \
  "$build_dir/Attention.cuda.o" "$build_dir/Model.cuda.o" "$build_dir/Training.cuda.o" \
  "$build_dir/FullTrainingMegakernelCore.cuda.o" "$build_dir/FullPretrainMegakernel.cuda.o" \
  "$build_dir/FullGRPOMegakernel.cuda.o" \
  "$build_dir/Megakernel.cuda.o" \
  "$build_dir/cuda_device.o" -o "$build_dir/cuda_dlink.o"
"$leanc_bin" "$build_dir/Driver.o" "$build_dir/Kernel.o" "$build_dir/Cuda.o" \
  "$build_dir/LeanTest.o" "$build_dir/LeanTest.Runner.o" "$build_dir/LeanTest.Attr.o" \
  "$build_dir/LeanTest.Assert.o" "$build_dir/LeanTest.Basic.o" "$build_dir/Megakernel.o" \
  "$build_dir/CheckpointLoRA.o" "$build_dir/FullGRPOMegakernel.o" \
  "$build_dir/FullPretrainMegakernel.o" "$build_dir/FullTrainingMegakernelCore.o" \
  "$build_dir/Training.o" "$build_dir/Model.o" \
  "$build_dir/MLP.o" "$build_dir/Attention.o" "$build_dir/DeltaNet.o" \
  "$build_dir/Projection.o" "$build_dir/LoRA.o" "$build_dir/Linear.o" \
  "$build_dir/Primitives.o" "$build_dir/Optimizer.o" "$build_dir/Foundation.o" "$build_dir/SafeTensors.o" "$build_dir/Config.o" \
  "$build_dir/Megakernel.cuda.o" "$build_dir/FullGRPOMegakernel.cuda.o" \
  "$build_dir/FullPretrainMegakernel.cuda.o" "$build_dir/FullTrainingMegakernelCore.cuda.o" \
  "$build_dir/Training.cuda.o" "$build_dir/Model.cuda.o" \
  "$build_dir/Attention.cuda.o" "$build_dir/DeltaNet.cuda.o" "$build_dir/LoRA.cuda.o" \
  "$build_dir/Linear.cuda.o" "$build_dir/Primitives.cuda.o" "$build_dir/Mailbox.cuda.o" \
  "$build_dir/cuda_device.o" "$build_dir/cuda_dlink.o" \
  -rdynamic -L"$cuda_home/lib64" -Wl,-rpath,"$cuda_home/lib64" -lcudart -lcudadevrt \
  -o "$build_dir/cuda_qwen36_megakernel_test"

case "$qwen_cuda_arch" in
  sm_90a) required_compute_capability=9.0 ;;
  sm_121|sm_121a) required_compute_capability=12.1 ;;
  *)
    echo "Skipping Qwen3.8 megakernel execution: no runtime gate for $qwen_cuda_arch"
    exit 0
    ;;
esac
if [[ -n ${QWEN_MODEL_DIR:-} ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 ||
      ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader |
        grep -Fxq "$required_compute_capability"; then
    echo "Skipping real Qwen3.8 megakernel tests: a $qwen_cuda_arch GPU is required"
    exit 0
  fi
fi

timeout 7200 stdbuf -oL -eL "$build_dir/cuda_qwen36_megakernel_test"

if [[ ${LEAN_CUDA_COMPUTE_SANITIZER:-0} == 1 ]] &&
    command -v compute-sanitizer >/dev/null 2>&1; then
  timeout 7200 stdbuf -oL -eL compute-sanitizer --tool memcheck --error-exitcode 1 \
    "$build_dir/cuda_qwen36_megakernel_test"
fi
