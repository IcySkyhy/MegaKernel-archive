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
 * \file memory_pool.h
 * \brief
 */

#pragma once

#include <unordered_map>
#include <vector>
#include "adapter/api/runtime_define.h"

namespace npu::tile_fwk {
inline constexpr uint32_t TWO_MB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE_PAGE_FIRST;

struct MemoryBlock {
    void* baseAddr;
    size_t blockSize;
    size_t usedSize;

    MemoryBlock(void* addr, size_t size);
    void* Allocate(uint64_t alignSize);
};

RtError NormalizedRtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind);

void* DevAlloc(uint64_t size);
void* CopyDataToDevice(const void* dataPtr, uint64_t dataSize);

class DevMemoryPool {
public:
    static DevMemoryPool& Instance();
    void AllocDevAddr(uint8_t** devAddr, const uint64_t size);
    void FreeDevAddr(void* ptr);
    bool CheckAllSentinels();
    void DestroyPool();

private:
    DevMemoryPool();
    ~DevMemoryPool();
    bool AllocDevAddrInPool(uint8_t** devAddr, uint64_t size);
    static void FreeMemBlock(MemoryBlock* block);
    static void PrintSentinelVal(std::vector<uint64_t>& sentinelVal, uint8_t* sentinelAddr);
    void PutSentinelAddr(uint8_t* baseAddr, uint64_t baseSize);
    bool CheckSentinel(uint8_t* baseAddr, bool remove = true);
    void RecordAllocation(void* ptr, MemoryBlock* block);
    MemoryBlock* CreateNewBlock(uint64_t alignSize);
    void DynamicRecycle();
    void PrintPoolStatus() const;

    std::vector<MemoryBlock*> memoryBlocks_;
    std::unordered_map<void*, MemoryBlock*> addrToBlock_;

    bool needMemCheck_{false};
    std::vector<uint64_t> sentinelVec_;
    std::unordered_map<uint8_t*, std::vector<uint8_t*>> sentinelValMap_;
};
} // namespace npu::tile_fwk
