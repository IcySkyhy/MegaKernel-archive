#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
source "$repo_root/scripts/backend-env.sh"
if [[ -n ${LEAN_CUDA_TOOLCHAIN_ROOT:-} ]]; then
  exec bash "$repo_root/scripts/build-nightly-worker.sh"
fi
lean_cuda_build_qwen_library >&2
build_dir=${QWEN_CHAT_BUILD_DIR:-"$script_dir/.build"}
rebuild=${QWEN_CHAT_REBUILD:-0}
cuda_profile="$repo_root/scripts/qwen-cuda-profile.sh"
source "$cuda_profile"
worker="$build_dir/qwen38-chat-worker"
stamp="$build_dir/source.sha256"
lean_bin=${LEAN:-"$(command -v lean)"}
leanc_bin=${LEANC:-"$(dirname "$lean_bin")/leanc"}

for executable in "$lean_bin" "$leanc_bin"; do
  if [[ ! -x $executable ]]; then
    echo "missing executable: $executable" >&2
    exit 1
  fi
done
if ! command -v nvcc >/dev/null 2>&1; then
  echo "Qwen3.8 chat requires CUDA 13 nvcc" >&2
  exit 1
fi
if ! nvcc --version | grep -Fq 'release 13.'; then
  echo "Qwen3.8 chat requires CUDA 13 nvcc" >&2
  exit 1
fi

sources=(
  "$script_dir/build_worker.sh"
  "$cuda_profile"
  "$script_dir/Worker.lean"
  "$LEAN_CUDA_ROOT/src/Lean/Cuda/Types.lean"
  "$LEAN_CUDA_ROOT/src/Lean/Cuda/Mailbox.lean"
  "$LEAN_CUDA_ROOT/src/Lean/Cuda/MXFP8.lean"
  "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Foundation.lean"
  "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/Config.lean"
  "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/SafeTensors.lean"
  "$LEAN_CUDA_ROOT/src/Lean/Cuda/Collective.lean"
  "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/Primitives.lean"
  "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/Megakernel.lean"
  "$LEAN_CUDA_ROOT/src/runtime/cuda_device.cu"
  "$LEAN_CUDA_ROOT/src/runtime/cuda_host.cpp"
  "$LEAN_CUDA_ROOT/src/include/lean/lean_cuda.h"
  "$LEAN_CUDA_ROOT/src/include/lean/lean_cuda_primitives.h"
  "$LEAN_CUDA_ROOT/src/include/lean/lean_cuda_host.h"
)
source_hash=$({
  "$lean_bin" --version
  printf '%s\n' "$qwen_cuda_profile_id"
  sha256sum "${sources[@]}"
} | sha256sum | cut -d ' ' -f 1)
if [[ $rebuild != 1 && -x $worker && -f $stamp ]] &&
    [[ $(<"$stamp") == "$source_hash" ]]; then
  echo "$worker"
  exit 0
fi

mkdir -p "$build_dir/LeanCudaQwen/Qwen36"
lean_libdir=$($lean_bin --print-libdir)
if [[ ! -f $build_dir/.lean-links-ready ]]; then
  cp -as --update=none "$lean_libdir/Lean" "$build_dir/"
  cp -as --update=none "$lean_libdir"/Lean.olean* "$build_dir/"
  touch "$build_dir/.lean-links-ready"
fi
export LEAN_PATH="$build_dir:${LEAN_PATH:-$lean_libdir}"


"$lean_bin" --root="$LEAN_CUDA_ROOT/src" -o "$build_dir/Lean/Cuda/MXFP8.olean" \
  -c "$build_dir/MXFP8.c" "$LEAN_CUDA_ROOT/src/Lean/Cuda/MXFP8.lean"

"$lean_bin" --root="$LEAN_CUDA_ROOT/src" -o "$build_dir/Lean/Cuda/Mailbox.olean" \
  -c "$build_dir/Mailbox.c" --cuda="$build_dir/Mailbox.cu" \
  -Dcompiler.postponeCompile=false "$LEAN_CUDA_ROOT/src/Lean/Cuda/Mailbox.lean"

"$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" \
  -o "$build_dir/LeanCudaQwen/Foundation.olean" -c "$build_dir/Foundation.c" \
  "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Foundation.lean"

for module in Config SafeTensors Primitives Megakernel; do
  cuda_args=()
  if [[ $module == Primitives || $module == Megakernel ]]; then
    cuda_args=(--cuda="$build_dir/$module.cu" -Dcompiler.postponeCompile=false)
  fi
  "$lean_bin" --root="$LEAN_CUDA_QWEN_ROOT/lib" \
    -o "$build_dir/LeanCudaQwen/Qwen36/$module.olean" -c "$build_dir/$module.c" \
    "${cuda_args[@]}" "$LEAN_CUDA_QWEN_ROOT/lib/LeanCudaQwen/Qwen36/$module.lean"
done
"$lean_bin" --root="$script_dir" -o "$build_dir/Worker.olean" \
  -c "$build_dir/Worker.c" "$script_dir/Worker.lean"

grep -Fq 'lean_cuda_grid_persistent_claim' "$build_dir/Megakernel.cu"
grep -Fq 'lean_cuda_l_Cuda_Qwen36_Megakernel_qwen36TrainingModel_kernel' "$build_dir/Megakernel.cu"
grep -Fq 'lean_cuda_l_Cuda_Qwen36_Megakernel_qwen36InferenceModel_kernel' "$build_dir/Megakernel.cu"
if grep -Fq 'lean_apply_' "$build_dir/Megakernel.cu" ||
    grep -Fq 'lean_alloc_ctor' "$build_dir/Megakernel.cu"; then
  echo "Qwen3.8 chat megakernel retained boxed device code" >&2
  exit 1
fi

cuda_home=${CUDA_HOME:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}
for module in Foundation Config SafeTensors Primitives Megakernel Worker; do
  "$leanc_bin" -c "$build_dir/$module.c" -o "$build_dir/$module.o"
done
for module in Mailbox Primitives Megakernel; do
  nvcc "${qwen_cuda_device_flags[@]}" -DLEAN_CUDA_HOST_LAUNCHERS \
    -I"$LEAN_CUDA_ROOT/src/include" -dc "$build_dir/$module.cu" -o "$build_dir/$module.cuda.o"
done
nvcc "${qwen_cuda_device_flags[@]}" -Xcompiler=-fvisibility=hidden \
  -I"$LEAN_CUDA_ROOT/src/include" -dc "$LEAN_CUDA_ROOT/src/runtime/cuda_device.cu" \
  -o "$build_dir/cuda_device.o"
objcopy --localize-hidden "$build_dir/cuda_device.o"
objcopy --wildcard --globalize-symbol='__fatbinwrap_*' "$build_dir/cuda_device.o"
nvcc "${qwen_cuda_dlink_flags[@]}" -dlink \
  "$build_dir/Mailbox.cuda.o" "$build_dir/Primitives.cuda.o" \
  "$build_dir/Megakernel.cuda.o" "$build_dir/cuda_device.o" -o "$build_dir/cuda_dlink.o"
"$leanc_bin" "$build_dir/Worker.o" \
  "$build_dir/Megakernel.o" "$build_dir/Primitives.o" "$build_dir/Foundation.o" \
  "$build_dir/SafeTensors.o" "$build_dir/Config.o" "$build_dir/Megakernel.cuda.o" \
  "$build_dir/Primitives.cuda.o" "$build_dir/Mailbox.cuda.o" "$build_dir/cuda_device.o" \
  "$build_dir/cuda_dlink.o" -rdynamic -L"$cuda_home/lib64" -Wl,-rpath,"$cuda_home/lib64" \
  -lcudart -lcudadevrt -o "$worker"

printf '%s\n' "$source_hash" > "$stamp"
echo "$worker"
