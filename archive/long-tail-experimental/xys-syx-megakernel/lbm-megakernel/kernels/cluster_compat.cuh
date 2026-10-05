#pragma once
// Clang 23 supplies the SM90+ intrinsics, but this CUDA 12.9 header gates
// cluster_group on NVCC/NVRTC unless this compatibility switch is present.
// Do not impersonate NVCC or change the numerical compiler flags.
#if defined(__clang__) && !defined(_CG_CLUSTER_INTRINSICS_AVAILABLE)
#if defined(__CUDA_ARCH__) && (!__has_builtin(__nvvm_barrier_cluster_arrive) || !__has_builtin(__nvvm_mapa))
#error "This Clang lacks the native cluster intrinsics; use a supported compiler"
#endif
#define _CG_CLUSTER_INTRINSICS_AVAILABLE 1
#endif
#include <cooperative_groups.h>
