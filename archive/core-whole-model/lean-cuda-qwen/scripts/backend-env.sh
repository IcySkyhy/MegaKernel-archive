#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

# Shared environment discovery for standalone examples. This file is sourced by example runners.

LEAN_CUDA_QWEN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export LEAN_CUDA_QWEN_ROOT
# shellcheck source=../toolchain/lean-cuda.env
source "$LEAN_CUDA_QWEN_ROOT/toolchain/lean-cuda.env"

if [[ -z ${LEAN_CUDA_TOOLCHAIN_ROOT:-} && -z ${LEAN_CUDA_ROOT:-} &&
      -x "$LEAN_CUDA_QWEN_ROOT/.lake/toolchains/$LEAN_CUDA_NIGHTLY_TAG/bin/lean" ]]; then
  export LEAN_CUDA_TOOLCHAIN_ROOT="$LEAN_CUDA_QWEN_ROOT/.lake/toolchains/$LEAN_CUDA_NIGHTLY_TAG"
fi

if [[ -n ${LEAN_CUDA_TOOLCHAIN_ROOT:-} ]]; then
  stage1=$(cd "$LEAN_CUDA_TOOLCHAIN_ROOT" && pwd -P)
  if [[ ! -x "$stage1/bin/lean" || ! -x "$stage1/bin/leanc" || ! -x "$stage1/bin/lake" ]]; then
    echo "LEAN_CUDA_TOOLCHAIN_ROOT must contain bin/lean, bin/leanc, and bin/lake" >&2
    return 1
  fi
  if [[ ${LEAN_CUDA_ALLOW_UNPINNED:-0} != 1 ]] &&
      [[ $("$stage1/bin/lean" --version) != *"$LEAN_CUDA_NIGHTLY_VERSION"* ]]; then
    echo "Expected the published $LEAN_CUDA_NIGHTLY_TAG binary" >&2
    return 1
  fi
else
  if [[ -n ${LEAN_CUDA_ROOT:-} ]]; then
    lean_cuda_root=$LEAN_CUDA_ROOT
  elif [[ -d "$LEAN_CUDA_QWEN_ROOT/.lake/lean4-cuda-backend" ]]; then
    lean_cuda_root="$LEAN_CUDA_QWEN_ROOT/.lake/lean4-cuda-backend"
  elif [[ -d "$LEAN_CUDA_QWEN_ROOT/../lean4-cuda-backend" ]]; then
    lean_cuda_root="$LEAN_CUDA_QWEN_ROOT/../lean4-cuda-backend"
  elif [[ -d "$LEAN_CUDA_QWEN_ROOT/../lean4/.cuda-backend-worktree" ]]; then
    lean_cuda_root="$LEAN_CUDA_QWEN_ROOT/../lean4/.cuda-backend-worktree"
  else
    echo "Cannot find the Lean CUDA backend; set LEAN_CUDA_ROOT to its worktree" >&2
    return 1
  fi

  LEAN_CUDA_ROOT=$(cd "$lean_cuda_root" && pwd -P)
  export LEAN_CUDA_ROOT

  stage1="$LEAN_CUDA_ROOT/build/release/stage1"
  if [[ ! -x "$stage1/bin/lean" || ! -x "$stage1/bin/leanc" || ! -x "$stage1/bin/lake" ]]; then
    echo "Missing staged Lean CUDA compiler under $stage1" >&2
    return 1
  fi

  if [[ ${LEAN_CUDA_ALLOW_UNPINNED:-0} != 1 ]]; then
    if ! git -C "$LEAN_CUDA_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "Lean CUDA backend is not a Git worktree: $LEAN_CUDA_ROOT" >&2
      return 1
    fi
    lean_cuda_revision=$(git -C "$LEAN_CUDA_ROOT" rev-parse HEAD)
    if [[ "$lean_cuda_revision" != "$LEAN_CUDA_REVISION" ]]; then
      echo "Lean CUDA backend revision is $lean_cuda_revision; expected $LEAN_CUDA_REVISION" >&2
      echo "Run scripts/setup-backend.sh or set LEAN_CUDA_ALLOW_UNPINNED=1 for local development" >&2
      return 1
    fi
  fi

fi

if [[ $("$stage1/bin/lean" --features) != *"[CUDA]"* ]]; then
  echo "Staged Lean compiler was not built with -DLEAN_CUDA=ON: $stage1/bin/lean" >&2
  return 1
fi

export PATH="$stage1/bin:$PATH"
export LEAN_SYSROOT="$stage1"

lean_cuda_build_qwen_library() {
  (cd "$LEAN_CUDA_QWEN_ROOT" && lake build LeanCudaQwen)
  export LEAN_PATH="$LEAN_CUDA_QWEN_ROOT/.lake/build/lib/lean${LEAN_PATH:+:$LEAN_PATH}"
}

cuda_arch=${LEAN_CUDA_ARCH:-sm_121}
case "$cuda_arch" in
  sm_90a) compute_capability=9.0 ;;
  sm_100a) compute_capability=10.0 ;;
  sm_103a) compute_capability=10.3 ;;
  sm_121|sm_121a) compute_capability=12.1 ;;
  *)
    echo "Unsupported CUDA example architecture: $cuda_arch" >&2
    return 1
    ;;
esac

if [[ -n ${CUDA_HOME:-} ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
elif command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
  export CUDA_HOME
fi
