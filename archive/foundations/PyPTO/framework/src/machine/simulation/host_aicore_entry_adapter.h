/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <stdint.h>
#include <chrono>
#include "machine/simulation/host_core_context.h"
#include "machine/simulation/aicore_hardware.h"

namespace npu::tile_fwk {
struct CoreFuncParam;
}

#ifndef mem_dsb_t
typedef uint64_t mem_dsb_t;
#endif

#ifndef dsb
#define dsb(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

#ifndef dcci
#define dcci(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

#ifndef set_flag
#define set_flag(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

#ifndef wait_flag
#define wait_flag(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

#ifndef set_mask_norm
#define set_mask_norm(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

#ifndef ffts_cross_core_sync
#define ffts_cross_core_sync(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

#ifndef wait_flag_dev
#define wait_flag_dev(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

static inline uint64_t get_sys_cnt()
{
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    return ns / 10;
}

static inline int64_t get_coreid() { return npu::tile_fwk::dynamic::HostCoreCtx::Current().phyId; }

static inline int get_block_idx() { return npu::tile_fwk::dynamic::HostCoreCtx::Current().blockId; }

static inline int get_subblockdim() { return 0; }

static inline int get_subblockid() { return 0; }

static inline int get_block_num() { return 0; }

static inline void set_cond(uint64_t v)
{
    const auto& ctx = npu::tile_fwk::dynamic::HostCoreCtx::Current();
    npu::tile_fwk::dynamic::AicoreHardware::Global().WriteCond(static_cast<size_t>(ctx.phyId), v);
}

static inline uint64_t GetDataMainBase()
{
    const auto& ctx = npu::tile_fwk::dynamic::HostCoreCtx::Current();
    return npu::tile_fwk::dynamic::AicoreHardware::Global().ReadMainBase(static_cast<size_t>(ctx.phyId));
}

static inline void CallSubFuncTask(uint64_t, npu::tile_fwk::CoreFuncParam*, int64_t, int64_t*)
{
    std::this_thread::sleep_for(std::chrono::microseconds(5)); // fixed 5us modeling
}

template <typename T, typename V>
static inline T atomicAdd(__gm__ T* address, V value)
{
    return __atomic_fetch_add(address, static_cast<T>(value), __ATOMIC_RELAXED);
}

template <typename T>
static inline T atomicCAS(__gm__ T* address, T compare, T value)
{
    __atomic_compare_exchange_n(address, &compare, value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return compare;
}

template <typename T>
static inline T atomicExch(__gm__ T* address, T value)
{
    return __atomic_exchange_n(address, value, __ATOMIC_RELAXED);
}
