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
 * \file emulation_memory_utils.h
 * \brief emulation memory utils
 */

#pragma once

#include <memory>
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error.h"
#include "tilefwk/error_code.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "securec.h"
#include "machine/runtime/runner/runtime_utils.h"

namespace npu::tile_fwk::dynamic {
struct EmulationMemoryUtils {
    EmulationMemoryUtils() {}
    ~EmulationMemoryUtils() = default;
    static bool IsDevice() { return false; }
    uint8_t* AllocDev(size_t size, uint8_t** cachedDevAddrHolder)
    {
        (void)cachedDevAddrHolder;
        if (size == 0 || size >= 0xFFFFFFFFF) {
            MACHINE_LOGE(DevCommonErr::PARAM_INVALID, "AllocDev failed: size=%zu bytes", size);
            return nullptr;
        }
        uint8_t* rawPtr = (uint8_t*)malloc(size);
        if (rawPtr == nullptr) {
            return nullptr;
        }
        std::shared_ptr<uint8_t> ptr(rawPtr, free);
        EmulationAllocatePtrs_.push_back(ptr);
        return rawPtr;
    }

    uint8_t* AllocZero(uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        (void)cachedDevAddrHolder;
        uint8_t* devPtr = AllocDev(size, nullptr);
        memset_s(devPtr, size, 0, size);
        return devPtr;
    }

    uint8_t* CopyToDev(uint8_t* data, uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = AllocDev(size, cachedDevAddrHolder);
        if (devPtr != nullptr) {
            MemcpyS(devPtr, size, data, size);
        }
        return devPtr;
    }

    template <typename T>
    T* CopyToDev(std::vector<T> data, uint8_t** cachedDevAddrHolder)
    {
        (void)cachedDevAddrHolder;
        return (T*)CopyToDev((uint8_t*)data.data(), data.size() * sizeof(T), nullptr);
    }

    void CopyFromDev(uint8_t* data, uint8_t* devPtr, uint64_t size) { MemcpyS(data, size, devPtr, size); }

    uint8_t* CopyToDev(RawTensorData& data)
    {
        if (data.GetDevPtr() == nullptr) {
            auto devAddr = CopyToDev((uint8_t*)data.data(), data.size(), nullptr);
            data.SetDevPtr(devAddr);
        }
        return data.GetDevPtr();
    }

    void CopyFromDev(RawTensorData& t) { CopyFromDev(t.data(), t.GetDevPtr(), t.size()); }

    static uint64_t GetL2Offset() { return 0; }

private:
    std::vector<std::shared_ptr<uint8_t>> EmulationAllocatePtrs_;
};
} // namespace npu::tile_fwk::dynamic
