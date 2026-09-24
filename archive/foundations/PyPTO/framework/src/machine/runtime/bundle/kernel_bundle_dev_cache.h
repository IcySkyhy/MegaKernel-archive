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
 * \file kernel_bundle_dev_cache.h
 * \brief Process-wide cache of per-bundle control-flow-cache DEVICE buffers.
 *
 * The packed base-0 control-flow cache is copied host->device once per bundle (keyed by LoadedBundle::bundleKey,
 * a content digest -- NOT hashKey, which is equal across shape variants of one op) and reused across launches.
 * Cached device buffers live for the whole process; DevMemoryPool releases them at teardown.
 */

#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace npu::tile_fwk::bundle {

class KernelBundleDevCache {
public:
    static KernelBundleDevCache& Instance();

    // Return the device pointer holding `hostCtrlCache` for `key`, copying host->device once and caching it for
    // reuse. Returns nullptr when `hostCtrlCache` is empty. Held until process exit.
    uint8_t* GetOrCopy(uint64_t key, const std::vector<uint8_t>& hostCtrlCache);

    // Return a device buffer of at least `bytes` for bundle `key`'s cell-match metadata pool, allocated once and
    // reused (grown if a later launch needs more). Returns nullptr when bytes==0. The device inits the pool in
    // place, so no host->device copy here. Held until process exit.
    uint8_t* GetOrAllocCellMatch(uint64_t key, uint64_t bytes);

    KernelBundleDevCache(const KernelBundleDevCache&) = delete;
    KernelBundleDevCache& operator=(const KernelBundleDevCache&) = delete;

private:
    KernelBundleDevCache() = default;
    ~KernelBundleDevCache();

    struct CellMatchBuf {
        uint8_t* addr{nullptr};
        uint64_t capacity{0};
    };

    std::mutex mutex_;
    std::unordered_map<uint64_t, uint8_t*> cache_;              // bundleKey -> device ctrl-flow-cache buffer
    std::unordered_map<uint64_t, CellMatchBuf> cellMatchCache_; // bundleKey -> device cell-match metadata pool
};

} // namespace npu::tile_fwk::bundle
