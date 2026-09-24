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
 * \file buffer_pool.h
 * \brief
 */

#ifndef PASS_BUFFER_POOL_H_
#define PASS_BUFFER_POOL_H_
#include "tilefwk/data_type.h"

#include <optional>

#include "interface/utils/common.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {

struct BufferAddrChange {
    int memId;
    uint64_t oldStart;
    uint64_t oldEnd;
    uint64_t newStart;
    uint64_t newEnd;
};

struct LocalBuffer {
    int id{0};
    uint64_t retireCycle{0};
    uint64_t startCycle{0};
    size_t start{0};
    size_t end{0};
    size_t size{0};
    MemoryType memType{MemoryType::MEM_UNKNOWN};
    bool operator<(const LocalBuffer& other) const
    {
        if (size < other.size) {
            return true;
        }
        if (size == other.size) {
            return id < other.id;
        }
        return false;
    }

    LocalBuffer(int tensorId, uint64_t shapeSize, MemoryType type)
    {
        id = tensorId;
        size = shapeSize;
        memType = type;
    }
};

using LocalBufferPtr = std::shared_ptr<LocalBuffer>;
// BufferSlice 一个时刻只能给一个tensor使用，允许生命周期不重叠的多个tensor分时复用
struct BufferSlice {
    uint64_t offset{0};
    uint64_t size{0};

    BufferSlice() = default;

    BufferSlice(uint64_t offset_, uint64_t size_) : offset(offset_), size(size_) {}
};

class BufferPool {
public:
    BufferPool(const MemoryType mem, const uint64_t memSize) : memType_(mem), memSize_(memSize) {}
    BufferPool() {}
    ~BufferPool() = default;
    // 返回tensorid 到 bufferblock的映射关系，value是bufferblock的index不是bufferblock的magic
    // 在已有的block中分配tensor空间
    Status Allocate(LocalBufferPtr tensor);
    std::map<uint64_t, uint64_t> GenFreeIntervals(const std::map<uint64_t, uint64_t>& occupiedSpace);
    std::map<uint64_t, std::map<uint64_t, uint64_t>> FindFreeIntervals();
    bool IsFull(const LocalBufferPtr tensor, bool isMainLoop);
    bool IsFullWithoutRearrange(const size_t size);
    Status Free(const int tensorId);
    uint64_t GetMemSize();

    size_t ObtainStartAddr(size_t i, const std::vector<std::tuple<int, size_t, size_t>>& allocatedBufs);
    size_t UpdateIdx(size_t& i, size_t sizeNeedSpill, size_t startAddr,
                     const std::vector<std::tuple<int, size_t, size_t>>& allocatedBufs);
    // 单池 spill 选组: 返回所有可能的 spill 组合 (空即"没有可 spill 候选")。
    // 与主线签名一致, 调用方按"空即失败"处理。
    std::vector<std::vector<int>> GetSpillGroup(size_t sizeNeedSpill);
    // 提取已分配 buf 列表 (按 startOffset 升序): 双池 spill 选组与单池 GetSpillGroup
    // 共用的"排序后已分配 buf"基本块, 拆出以便 OoOScheduler 层的双池 helper 直接复用。
    std::vector<std::tuple<int, size_t, size_t>> GetSortedAllocatedBufs();
    std::vector<int> GetBufferSlices();
    std::vector<int> GetAddrSortedBufs();
    bool isAllocate(const int tensorId);
    uint64_t GetAllocatedSize();
    uint64_t GetBufferOffset(int memId);
    uint64_t GetBufferSize(int memId);
    Status ModifyBufferRange(LocalBufferPtr localBuffer, size_t offset);
    bool CheckBufferSlicesOverlap();
    void PrintStatus();
    Status MakeBufferSlice(LocalBufferPtr tensor, BufferSlice& newSlice);
    void SelectHeadAndTail(LocalBufferPtr tensor, bool& head, bool& tail,
                           std::map<uint64_t, std::map<uint64_t, uint64_t>> freeIntervals);
    // Fills `changes` with one entry per moved memId for NotifyBufferRearrange.
    Status CompactBufferSlices(std::unordered_map<int, LocalBufferPtr>& localBufferMap,
                               std::vector<BufferAddrChange>& changes);

    // === DualDst 扩展 ===
    // 返回本池按起点升序的扁平 [start, end) 空闲段列表;OoOScheduler::FindCommonFreeOffset
    // 在两池列表上做归并求交集得到 dualdst 同址 offset。跨池交集决策不属于单池职责,
    // 因此 BufferPool 只提供"自己的空闲段"原料, 决策上提到 OoOScheduler。
    std::vector<std::pair<uint64_t, uint64_t>> GetSortedFreeIntervals();
    // 把 tensor 固定放到 offset;越界 / 与已有 slice 重叠返回 FAILED。
    Status AllocateAtOffset(LocalBufferPtr tensor, uint64_t offset);
    // 注: 历史上的 GetSpillGroupsWithFreedRange + SpillGroupWithRange + WithFreedRange tag
    //     已全部移除。dualdst 双池选组改由 OoOScheduler::GetDualSpillGroup 负责,
    //     内部嵌套两次单池滑窗 + 比较 startAddr 是否一致, 不再依赖富信息结构体。

private:
    MemoryType memType_{MemoryType::MEM_UNKNOWN};
    uint64_t memSize_{0};
    std::map<int, BufferSlice> bufferSlices;
    std::unordered_map<int, int> tensorIdToBuffer_;
};
} // namespace npu::tile_fwk

#endif
