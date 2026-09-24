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
 * \file aot_binary.cpp
 * \brief AOT control-flow code pool manager (flat array + LRU clock).
 */

#include "machine/device/dynamic/aot_binary.h"

namespace npu::tile_fwk::dynamic {

AOTCodePoolManager& AOTCodePoolManager::Instance()
{
    static AOTCodePoolManager mgr;
    return mgr;
}

int AOTCodePoolManager::FindEntry(uint64_t hashKey, uint8_t lastId) const
{
    if (lastId < AOT_CODE_POOL_NUM && ownerHashKey_[lastId] == hashKey) {
        return static_cast<int>(lastId);
    }
    for (int i = 0; i < AOT_CODE_POOL_NUM; ++i) {
        if (ownerHashKey_[i] == hashKey) {
            return i;
        }
    }
    return -1;
}

int AOTCodePoolManager::SelectVictimEntry() const
{
    for (int i = 0; i < AOT_CODE_POOL_NUM; ++i) {
        if (ownerHashKey_[i] == 0) {
            return i;
        }
    }
    uint64_t minLru = UINT64_MAX;
    int victim = 0;
    for (int i = 0; i < AOT_CODE_POOL_NUM; ++i) {
        if (lruSeq_[i] < minLru) {
            minLru = lruSeq_[i];
            victim = i;
        }
    }
    return victim;
}

void AOTCodePoolManager::LoadEntry(int entryId, uint64_t hashKey, const void* data, uint64_t size)
{
    const uintptr_t base = EntryCodeBase(entryId);
    if (size > 0) {
        PerfBegin(PERF_EVT_CONTROL_FLOW_MAPEXE_MEMCPY);
        DevMemcpyS(reinterpret_cast<void*>(base), size, data, size);
        __builtin___clear_cache(reinterpret_cast<char*>(base), reinterpret_cast<char*>(base) + size);
        PerfEnd(PERF_EVT_CONTROL_FLOW_MAPEXE_MEMCPY);
    }
    ownerHashKey_[entryId] = hashKey;
}

int AOTCodePoolManager::EnsureCached(uint64_t hashKey, uint8_t& lastId, const void* data, uint64_t size)
{
    int entryId = FindEntry(hashKey, lastId);
    if (entryId < 0) {
        entryId = SelectVictimEntry();
        LoadEntry(entryId, hashKey, data, size);
    }
    lruSeq_[entryId] = ++lruClock_;
    lastId = static_cast<uint8_t>(entryId);
    return entryId;
}

} // namespace npu::tile_fwk::dynamic
