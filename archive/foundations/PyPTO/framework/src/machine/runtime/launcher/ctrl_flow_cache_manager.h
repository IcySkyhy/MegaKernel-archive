/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ctrl_flow_cache_manager.h
 * \brief Control flow cache management for kernel launch.
 */

#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include "machine/runtime/launcher/device_launcher_types.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"

namespace npu::tile_fwk::dynamic {

class KernelBinary;

struct ControlFlowCache {
    int64_t hash;
    std::vector<DeviceTensorData> inputs;
    uint8_t* devCache{nullptr};

    ControlFlowCache(std::vector<DeviceTensorData>& datas, uint8_t* tcache) : inputs(datas), devCache(tcache)
    {
        hash = Hash(inputs);
    }

    static int64_t Hash(const std::vector<DeviceTensorData>& datas)
    {
        uint64_t h = 14695981039346656037ull;
        for (auto& data : datas) {
            for (auto x : data.GetShape()) {
                h ^= x;
                h *= 1099511628211ull;
            }
        }
        return h;
    }

    static int64_t Hash(const std::vector<std::vector<int64_t>>& shapes)
    {
        uint64_t h = 14695981039346656037ull;
        for (auto& shape : shapes) {
            for (auto x : shape) {
                h ^= x;
                h *= 1099511628211ull;
            }
        }
        return h;
    }
};

struct HostControlFlowCache {
    int64_t hash;
    std::vector<uint8_t> hostCache;

    HostControlFlowCache(std::vector<DeviceTensorData>& datas, std::vector<uint8_t>&& hcache)
        : hostCache(std::move(hcache))
    {
        hash = ControlFlowCache::Hash(datas);
    }
};

class CtrlFlowCacheManager {
public:
    static CtrlFlowCacheManager& Instance();

    CtrlFlowCacheManager(const CtrlFlowCacheManager&) = delete;
    CtrlFlowCacheManager& operator=(const CtrlFlowCacheManager&) = delete;

    uint8_t* FindOrBuildDevCache(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors);

    DevControlFlowCache* GetHostCtrlFlowCache(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors,
                                              uint8_t* devCache, std::vector<uint8_t>& hostCache);

private:
    CtrlFlowCacheManager() = default;
    ~CtrlFlowCacheManager() = default;

    DevControlFlowCache* FindHostCtrlFlowCache(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors,
                                               std::vector<uint8_t>& hostCache);
    void AddHostCtrlFlowCache(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors,
                              std::vector<uint8_t>&& hostCache);
};

} // namespace npu::tile_fwk::dynamic
