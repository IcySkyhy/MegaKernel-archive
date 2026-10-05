#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

qwen_cuda_arch=${QWEN_CUDA_ARCH:-sm_121a}
qwen_cuda_use_fast_math=${QWEN_CUDA_USE_FAST_MATH:-1}
qwen_cuda_max_registers=${QWEN_CUDA_MAX_REGISTERS:-40}

if [[ ! $qwen_cuda_arch =~ ^sm_[0-9]+a?$ ]]; then
  echo "invalid QWEN_CUDA_ARCH: $qwen_cuda_arch" >&2
  return 1
fi
if [[ $qwen_cuda_use_fast_math != 0 && $qwen_cuda_use_fast_math != 1 ]]; then
  echo "QWEN_CUDA_USE_FAST_MATH must be 0 or 1" >&2
  return 1
fi
if [[ ! $qwen_cuda_max_registers =~ ^[0-9]+$ ]] ||
    ((qwen_cuda_max_registers != 0 && qwen_cuda_max_registers < 16)) ||
    ((qwen_cuda_max_registers > 255)); then
  echo "QWEN_CUDA_MAX_REGISTERS must be 0 or an integer from 16 through 255" >&2
  return 1
fi

qwen_cuda_device_flags=(-std=c++17 -O3 "-arch=$qwen_cuda_arch" -rdc=true)
if [[ $qwen_cuda_use_fast_math == 1 ]]; then
  qwen_cuda_device_flags+=(--use_fast_math)
fi
if ((qwen_cuda_max_registers != 0)); then
  qwen_cuda_device_flags+=("--maxrregcount=$qwen_cuda_max_registers")
fi
qwen_cuda_dlink_flags=("-arch=$qwen_cuda_arch")
qwen_cuda_profile_id="arch=$qwen_cuda_arch;fast_math=$qwen_cuda_use_fast_math;max_registers=$qwen_cuda_max_registers"
