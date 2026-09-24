/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aikernel_runtime_data_ring_buffer.h
 * \brief
 */

#ifndef AIKERNEL_RUNTIME_DATA_RING_BUFFER_H
#define AIKERNEL_RUNTIME_DATA_RING_BUFFER_H

#include "tilefwk/aikernel_tensor.h"
#include "tilefwk/aikernel_device_task.h"

namespace npu::tile_fwk {

struct DevStartArgsBase {
    __gm__ DevTensorData* devTensorList;
    uint64_t inputTensorSize;
    uint64_t outputTensorSize;
    __gm__ int64_t* commContexts;
    uint64_t commGroupNum;
    volatile uint64_t syncFlag{0}; // sche and ctrl soft sync flag

    __gm__ DrcoDeviceTaskReadyQueue* drcoDeviceTaskReadyQueue{nullptr};

#ifdef __TILE_FWK_HOST__
    int GetInputTensorSize() const { return inputTensorSize; }
    const DevTensorData& GetInputTensor(int index) const { return devTensorList[index]; }
    DevTensorData& GetInputTensor(int index) { return devTensorList[index]; }

    int GetOutputTensorSize() const { return outputTensorSize; }
    const DevTensorData& GetOutputTensor(int index) const { return devTensorList[index + inputTensorSize]; }
    DevTensorData& GetOutputTensor(int index) { return devTensorList[index + inputTensorSize]; }
#endif
};

struct AtomicUint64 {
#ifdef __TILE_FWK_HOST__
    constexpr AtomicUint64() = default;
    constexpr AtomicUint64(uint64_t v) : value(v) {}

    uint64_t operator=(uint64_t v)
    {
        __atomic_store_n(&value, v, __ATOMIC_SEQ_CST);
        return v;
    }

    operator uint64_t() const { return __atomic_load_n(&value, __ATOMIC_SEQ_CST); }

    uint64_t operator+=(uint64_t delta) { return __atomic_add_fetch(&value, delta, __ATOMIC_SEQ_CST); }

    uint64_t operator++() { return __atomic_add_fetch(&value, 1, __ATOMIC_SEQ_CST); }
#endif
public:
    /* mutable: const readers (Full/Empty) must still perform an atomic load */
    uint64_t value{0};
};

enum class ArchInfo : uint32_t { DAV_1001 = 1001, DAV_2201 = 2201, DAV_3510 = 3510, DAV_UNKNOWN };

struct RuntimeDataRingBufferHeadData {
    uint64_t runtimeDataSize;
    uint64_t runtimeDataCount;

    ArchInfo archInfo{ArchInfo::DAV_2201};

    /* ringbuffer's end and begin */
    AtomicUint64 indexFinished;
    AtomicUint64 indexPending;

    static INLINE __gm__ uint8_t* GetRuntimeDataBase(__gm__ RuntimeDataRingBufferHeadData* data)
    {
        return ((__gm__ uint8_t*)data) + sizeof(RuntimeDataRingBufferHeadData);
    }
    static INLINE __gm__ DevStartArgsBase* GetRuntimeDataCurrent(__gm__ RuntimeDataRingBufferHeadData* data)
    {
        __gm__ uint8_t* base = GetRuntimeDataBase(data);
        uint64_t index = data->indexFinished.value + 1;
        uint64_t size = data->runtimeDataSize;
        uint64_t count = data->runtimeDataCount;
        __gm__ uint8_t* current = base + (index % count) * size;
        return (__gm__ DevStartArgsBase*)current;
    }
    static INLINE __gm__ DevStartArgsBase* GetRuntimeDataPending(__gm__ RuntimeDataRingBufferHeadData* data)
    {
        __gm__ uint8_t* base = GetRuntimeDataBase(data);
        uint64_t index = data->indexPending.value;
        uint64_t size = data->runtimeDataSize;
        uint64_t count = data->runtimeDataCount;
        __gm__ uint8_t* pending = base + (index % count) * size;
        return (__gm__ DevStartArgsBase*)pending;
    }
};
static_assert(sizeof(RuntimeDataRingBufferHeadData) % sizeof(uint64_t) == 0, "Invalid ring buffer head");

} // namespace npu::tile_fwk

#endif
