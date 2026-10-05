#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../toolchain/lean-cuda.env
source "$repo_root/toolchain/lean-cuda.env"

target=${1:-"$repo_root/.lake/lean4-cuda-backend"}

if [[ -e "$target" ]] &&
    ! git -C "$target" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Backend target exists but is not a Git worktree: $target" >&2
  exit 1
fi

if [[ ! -e "$target" ]]; then
  mkdir -p "$(dirname "$target")"
  git clone --filter=blob:none --no-checkout "$LEAN_CUDA_REPOSITORY" "$target"
fi

if [[ -n $(git -C "$target" status --porcelain) ]]; then
  echo "Refusing to change dirty backend worktree: $target" >&2
  exit 1
fi

if ! git -C "$target" cat-file -e "$LEAN_CUDA_REVISION^{commit}" 2>/dev/null; then
  git -C "$target" fetch --depth 1 origin "$LEAN_CUDA_REVISION"
fi
git -C "$target" checkout --detach "$LEAN_CUDA_REVISION"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required to build the Lean CUDA backend" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA 13 nvcc must be on PATH" >&2
  exit 1
fi

jobs=${LEAN_CUDA_BUILD_JOBS:-}
if [[ -z "$jobs" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    jobs=$(nproc)
  else
    jobs=4
  fi
fi

(cd "$target" && cmake --preset release -DLEAN_CUDA=ON)
cmake --build "$target/build/release" --parallel "$jobs"

echo "Lean CUDA backend ready at $target ($LEAN_CUDA_REVISION)"
