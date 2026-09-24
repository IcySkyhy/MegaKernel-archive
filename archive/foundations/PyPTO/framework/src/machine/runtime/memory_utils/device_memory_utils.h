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
 * \file device_memory_utils.h
 * \brief device memory utils
 */

#pragma once

#include "adapter/api/runtime_api.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "machine/runtime/runner/runtime_utils.h"

namespace npu::tile_fwk::dynamic {
struct DeviceMemoryUtils {
    DeviceMemoryUtils(bool isHugePage = true) : isUseHugePage_(isHugePage) {}
    static bool IsDevice() { return true; }
    uint8_t* AllocDev(size_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = nullptr;
        if (cachedDevAddrHolder == nullptr) {
            if (isUseHugePage_) {
                DevMemoryPool::Instance().AllocDevAddr(&devPtr, size);
            } else {
                RuntimeMalloc((void**)&devPtr, size, RT_MEMORY_HBM, 0);
            }
        } else if (*cachedDevAddrHolder == nullptr) {
            if (isUseHugePage_) {
                DevMemoryPool::Instance().AllocDevAddr(&devPtr, size);
            } else {
                RuntimeMalloc((void**)&devPtr, size, RT_MEMORY_HBM, 0);
            }
            *cachedDevAddrHolder = devPtr;
        } else {
            devPtr = *cachedDevAddrHolder;
        }
        return devPtr;
    }

    uint8_t* AllocZero(uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = AllocDev(size, cachedDevAddrHolder);
        (void)RuntimeMemset(devPtr, size, 0, size);
        return devPtr;
    }

    uint8_t* CopyToDev(uint8_t* data, uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = AllocDev(size, cachedDevAddrHolder);
        RuntimeMemcpy(devPtr, size, data, size, RtMemcpyKind::HOST_TO_DEVICE);
        return devPtr;
    }

    void CopyToDev(uint8_t* devPtr, uint8_t* data, uint64_t size)
    {
        RuntimeMemcpy(devPtr, size, data, size, RtMemcpyKind::HOST_TO_DEVICE);
    }

    template <typename T>
    T* CopyToDev(std::vector<T> data, uint8_t** cachedDevAddrHolder)
    {
        return (T*)CopyToDev((uint8_t*)data.data(), data.size() * sizeof(T), cachedDevAddrHolder);
    }

    void CopyFromDev(uint8_t* data, uint8_t* devPtr, uint64_t size)
    {
        RuntimeMemcpy(data, size, devPtr, size, RtMemcpyKind::DEVICE_TO_HOST);
    }

    uint8_t* CopyToDev(RawTensorData& data)
    {
        if (data.GetDevPtr() == nullptr) {
            uint8_t* devPtr = nullptr;
            DevMemoryPool::Instance().AllocDevAddr(&devPtr, data.size());
            if (devPtr == nullptr) {
                return nullptr;
            }
            RuntimeMemcpy(devPtr, data.size(), (uint8_t*)data.data(), data.size(), RtMemcpyKind::HOST_TO_DEVICE);
            data.SetDevPtr(devPtr);
        }
        return data.GetDevPtr();
    }

    void CopyFromDev(RawTensorData& data) { CopyFromDev(data.data(), data.GetDevPtr(), data.size()); }

    void Free(uint8_t* mem)
    {
        if (mem == nullptr) {
            return;
        }
        if (isUseHugePage_) {
            DevMemoryPool::Instance().FreeDevAddr(mem);
        } else {
            RuntimeFree(mem);
        }
    }

    static uint64_t GetL2Offset() { return GetRuntimeL2Offset(); }

    bool isUseHugePage_{true};
};
} // namespace npu::tile_fwk::dynamic
