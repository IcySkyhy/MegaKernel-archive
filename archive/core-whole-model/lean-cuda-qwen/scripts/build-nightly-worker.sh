#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. Apache-2.0.
# Compile AND native/device-link the chat worker using only the binary distribution.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${LEAN_CUDA_TOOLCHAIN_ROOT:?Set LEAN_CUDA_TOOLCHAIN_ROOT using scripts/install-nightly.sh}"
source "$repo_root/scripts/backend-env.sh"
source "$repo_root/scripts/qwen-cuda-profile.sh"
exec 3>&1
exec 1>&2
lean_cuda_build_qwen_library
nvcc --version | grep -F 'release 13.'
build_dir=${QWEN_CHAT_BUILD_DIR:-"$repo_root/examples/qwen38_chat/.build/nightly"}
mkdir -p "$build_dir/LeanCudaQwen/Qwen36"
lean_bin="$stage1/bin/lean"
leanc_bin="$stage1/bin/leanc"
lean_libdir=$("$lean_bin" --print-libdir)
export LEAN_PATH="$build_dir:$repo_root/.lake/build/lib/lean:$lean_libdir"
for module in Foundation Qwen36/Config Qwen36/SafeTensors Qwen36/Primitives Qwen36/Megakernel; do
  name=${module##*/}
  cuda_args=()
  if [[ $name == Primitives || $name == Megakernel ]]; then
    cuda_args=(--cuda="$build_dir/$name.cu" -Dcompiler.postponeCompile=false)
  fi
  "$lean_bin" --root="$repo_root/lib" -o "$build_dir/LeanCudaQwen/$module.olean" \
    -c "$build_dir/$name.c" "${cuda_args[@]}" "$repo_root/lib/LeanCudaQwen/$module.lean"
  "$leanc_bin" -c "$build_dir/$name.c" -o "$build_dir/$name.o"
done
"$lean_bin" --root="$repo_root/examples/qwen38_chat" \
  -c "$build_dir/Worker.c" "$repo_root/examples/qwen38_chat/Worker.lean"
"$leanc_bin" -c "$build_dir/Worker.c" -o "$build_dir/Worker.o"
"$lean_bin" --run "$repo_root/scripts/EmitNightlyModule.lean" Lean.Cuda.Mailbox "$build_dir/Mailbox.cu"
for module in Mailbox Primitives Megakernel; do
  nvcc "${qwen_cuda_device_flags[@]}" -DLEAN_CUDA_HOST_LAUNCHERS \
    -I"$stage1/include" -dc "$build_dir/$module.cu" -o "$build_dir/$module.cuda.o"
done
nvcc "${qwen_cuda_device_flags[@]}" -Xcompiler=-fvisibility=hidden \
  -I"$stage1/include" -dc "$repo_root/scripts/nightly-runtime.cu" -o "$build_dir/cuda_device.o"
objcopy --localize-hidden "$build_dir/cuda_device.o"
objcopy --wildcard --globalize-symbol='__fatbinwrap_*' "$build_dir/cuda_device.o"
nvcc "${qwen_cuda_dlink_flags[@]}" -dlink "$build_dir/Mailbox.cuda.o" "$build_dir/Primitives.cuda.o" \
  "$build_dir/Megakernel.cuda.o" "$build_dir/cuda_device.o" -o "$build_dir/cuda_dlink.o"
"$leanc_bin" "$build_dir/Worker.o" "$build_dir/Foundation.o" "$build_dir/Config.o" \
  "$build_dir/SafeTensors.o" "$build_dir/Primitives.o" "$build_dir/Megakernel.o" \
  "$build_dir/Mailbox.cuda.o" "$build_dir/Primitives.cuda.o" "$build_dir/Megakernel.cuda.o" "$build_dir/cuda_device.o" \
  "$build_dir/cuda_dlink.o" -rdynamic -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" \
  -lcudart -lcudadevrt -o "$build_dir/qwen38-chat-worker"
grep -Fq 'lean_cuda_l_Cuda_Qwen36_Megakernel_qwen36InferenceModel_kernel' "$build_dir/Megakernel.cu"
printf '%s\n' "$build_dir/qwen38-chat-worker" >&3
