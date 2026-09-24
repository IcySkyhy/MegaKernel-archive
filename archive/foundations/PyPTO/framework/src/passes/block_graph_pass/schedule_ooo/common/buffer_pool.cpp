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
 * \file buffer_pool.cpp
 * \brief
 */

#include "passes/block_graph_pass/schedule_ooo/common/buffer_pool.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {
constexpr size_t START_ADDR_IDX = 2;
constexpr double ONE_THIRD = 1.0 / 3.0;

std::map<uint64_t, uint64_t> BufferPool::GenFreeIntervals(const std::map<uint64_t, uint64_t>& occupiedSpace)
{
    std::map<uint64_t, uint64_t> freeIntervals;
    if (occupiedSpace.begin()->first > 0) {
        freeIntervals.insert({0, occupiedSpace.begin()->first});
    }
    // 遍历所有已占用的片段，查找相邻片段之间的空闲区域
    auto prevIt = occupiedSpace.begin();
    for (auto it = occupiedSpace.begin(); it != occupiedSpace.end(); ++it) {
        if (prevIt != it && prevIt->second < it->first) {
            freeIntervals.insert({prevIt->second, it->first});
        }
        prevIt = it;
    }
    // 检查末尾是否为空闲
    if (prevIt->second < memSize_) {
        freeIntervals.insert({prevIt->second, memSize_});
    }
    return freeIntervals;
}

std::map<uint64_t, std::map<uint64_t, uint64_t>> BufferPool::FindFreeIntervals()
{
    // 收集可用的offset + size
    std::map<uint64_t, uint64_t> occupiedSpace;
    std::map<uint64_t, std::map<uint64_t, uint64_t>> freeIntervalsMap;
    for (auto slice : bufferSlices) {
        // 当前slice被占用着
        auto tensorEnd = slice.second.offset + slice.second.size;
        occupiedSpace[slice.second.offset] = tensorEnd;
    }
    if (occupiedSpace.empty()) {
        freeIntervalsMap[memSize_].insert({0, memSize_});
        return freeIntervalsMap;
    }
    // 检查起始点是否为空闲
    std::map<uint64_t, uint64_t> freeIntervals = GenFreeIntervals(occupiedSpace);
    for (auto freeInterval : freeIntervals) {
        freeIntervalsMap[freeInterval.second - freeInterval.first].insert(freeInterval);
    }
    return freeIntervalsMap;
}

size_t BufferPool::ObtainStartAddr(size_t i, const std::vector<std::tuple<int, size_t, size_t>>& allocatedBufs)
{
    if (i == 0) {
        return 0;
    }
    return std::get<START_ADDR_IDX>(allocatedBufs[i - 1]);
}

size_t BufferPool::UpdateIdx(size_t& i, size_t sizeNeedSpill, size_t startAddr,
                             const std::vector<std::tuple<int, size_t, size_t>>& allocatedBufs)
{
    size_t j = i;
    while (j < allocatedBufs.size() && (std::get<1>(allocatedBufs[j]) - startAddr) < sizeNeedSpill) {
        j += 1;
    }
    size_t endAddr = memSize_;
    if (j < allocatedBufs.size()) {
        endAddr = std::get<1>(allocatedBufs[j]);
    }
    // 必须写成 i + 1 < j: 原先的 i < (j - 1) 在 j == 0 (没有 allocatedBufs 的 end addr
    // 落在 spill 窗口内, 首个 while 不推进) 时, j - 1 size_t 下溢成 SIZE_MAX, 本 while
    // 失控把 i 一路推过 allocatedBufs.size() (越界访问 + 让上游 GetDualSpillGroup 的
    // (jA - iA) 再次下溢, reserve 抛 length_error "vector::reserve").
    while (i + 1 < j && (endAddr - std::get<START_ADDR_IDX>(allocatedBufs[i])) >= sizeNeedSpill) {
        i += 1;
    }
    return j;
}

std::vector<std::tuple<int, size_t, size_t>> BufferPool::GetSortedAllocatedBufs()
{
    std::vector<std::tuple<int, size_t, size_t>> allocatedBufs;
    for (auto& [memId, bufferSlice] : bufferSlices) {
        allocatedBufs.push_back(std::make_tuple(memId, bufferSlice.offset, bufferSlice.offset + bufferSlice.size));
    }
    std::sort(allocatedBufs.begin(), allocatedBufs.end(),
              [&](std::tuple<int, size_t, size_t>& a, std::tuple<int, size_t, size_t>& b) {
                  return std::get<1>(a) < std::get<1>(b);
              });
    return allocatedBufs;
}

std::vector<std::vector<int>> BufferPool::GetSpillGroup(size_t sizeNeedSpill)
{
    std::vector<std::vector<int>> canSpillGroups;
    auto allocatedBufs = GetSortedAllocatedBufs();
    size_t i = 0;
    while (i < allocatedBufs.size()) {
        size_t startAddr = ObtainStartAddr(i, allocatedBufs);
        if ((memSize_ - startAddr) < sizeNeedSpill) {
            break;
        }
        size_t j = UpdateIdx(i, sizeNeedSpill, startAddr, allocatedBufs);
        if (i == j) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Incorrect idx for allocatedBufs.");
            return canSpillGroups;
        }
        std::vector<int> group;
        for (size_t k = i; k < j; k++) {
            group.push_back(std::get<0>(allocatedBufs[k]));
        }
        canSpillGroups.push_back(group);
        i += 1;
    }
    return canSpillGroups;
}

std::vector<int> BufferPool::GetBufferSlices()
{
    std::vector<int> res;
    for (auto bufferSlice : bufferSlices) {
        res.push_back(bufferSlice.first);
    }
    return res;
}

Status BufferPool::MakeBufferSlice(LocalBufferPtr tensor, BufferSlice& newSlice)
{
    newSlice.size = tensor->size;
    if (bufferSlices.find(tensor->id) != bufferSlices.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] already alloc in bufferSlices.", tensor->id);
        return FAILED;
    }
    bufferSlices[tensor->id] = newSlice;
    tensor->start = newSlice.offset;
    tensor->end = newSlice.offset + newSlice.size;
    APASS_LOG_DEBUG_F(Elements::Tensor, " Allocate Tensor[%d], range [%lu, %lu].", tensor->id, newSlice.offset,
                      newSlice.size + newSlice.offset);
    return SUCCESS;
}

void BufferPool::SelectHeadAndTail(LocalBufferPtr tensor, bool& head, bool& tail,
                                   std::map<uint64_t, std::map<uint64_t, uint64_t>> freeIntervals)
{
    for (auto& interval : freeIntervals) {
        if (interval.first < tensor->size) {
            continue;
        }
        for (auto& freeSpace : interval.second) {
            if (freeSpace.first == 0) {
                head = true;
            }
            if (freeSpace.second == memSize_) {
                tail = true;
            }
        }
    }
}

Status BufferPool::Allocate(LocalBufferPtr tensor)
{
    std::map<uint64_t, std::map<uint64_t, uint64_t>> freeIntervals = FindFreeIntervals();
    // size, {begin, end}
    // 创建新的bufferSlice
    if (tensor->memType == MemoryType::MEM_L0A || tensor->memType == MemoryType::MEM_L0B ||
        tensor->memType == MemoryType::MEM_L0C) {
        bool headFree = false;
        bool tailFree = false;
        SelectHeadAndTail(tensor, headFree, tailFree, freeIntervals);
        BufferSlice newSlice;
        if (headFree) {
            newSlice.offset = 0;
            if (MakeBufferSlice(tensor, newSlice) != SUCCESS) {
                return FAILED;
            } else {
                return SUCCESS;
            }
        } else if (tailFree) {
            newSlice.offset = memSize_ - tensor->size;
            if (MakeBufferSlice(tensor, newSlice) != SUCCESS) {
                return FAILED;
            } else {
                return SUCCESS;
            }
        }
    }
    for (auto& interval : freeIntervals) {
        if (interval.first < tensor->size) {
            continue;
        }
        for (auto& freeSpace : interval.second) {
            BufferSlice newSlice;
            newSlice.offset = freeSpace.first;
            if (MakeBufferSlice(tensor, newSlice) != SUCCESS) {
                return FAILED;
            } else {
                return SUCCESS;
            }
        }
    }
    APASS_LOG_ERROR_F(Elements::Tensor, "Buffer doesnot have enough memory to allocate Tensor[%d].", tensor->id);
    return FAILED;
}

std::vector<int> BufferPool::GetAddrSortedBufs()
{
    std::vector<int> memIds;
    for (auto& [memId, slice] : bufferSlices) {
        (void)slice;
        memIds.push_back(memId);
    }
    std::sort(memIds.begin(), memIds.end(),
              [&](int a, int b) { return bufferSlices[a].offset < bufferSlices[b].offset; });
    return memIds;
}

uint64_t BufferPool::GetMemSize() { return memSize_; }

bool BufferPool::isAllocate(const int tensorId)
{
    if (bufferSlices.find(tensorId) == bufferSlices.end()) {
        return false;
    }
    return true;
}

Status BufferPool::Free(const int tensorId)
{
    if (bufferSlices.find(tensorId) == bufferSlices.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] not in bufferSlices.", tensorId);
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "    Free tensor[%d], range:[%lu, %lu]", tensorId,
                      bufferSlices[tensorId].offset, bufferSlices[tensorId].size + bufferSlices[tensorId].offset);
    bufferSlices.erase(tensorId);
    return SUCCESS;
}

bool BufferPool::IsFull(const LocalBufferPtr tensor, bool isMainLoop)
{
    if (tensor->memType == MemoryType::MEM_BT) {
        if (bufferSlices.size() >= 1) {
            return true;
        }
    }
    auto freeSpace = FindFreeIntervals();
    if (tensor->memType == MemoryType::MEM_L0C && tensor->size >= ONE_THIRD * memSize_ && isMainLoop) {
        bool headFree = false;
        bool tailFree = false;
        SelectHeadAndTail(tensor, headFree, tailFree, freeSpace);
        return !(headFree || tailFree);
    }
    for (auto inter : freeSpace) {
        if (inter.first >= tensor->size) {
            return false;
        }
    }
    return true;
}

bool BufferPool::IsFullWithoutRearrange(const size_t size)
{
    auto freeSize = GetMemSize() - GetAllocatedSize();
    if (freeSize >= size) {
        return false;
    }
    return true;
}

uint64_t BufferPool::GetAllocatedSize()
{
    uint64_t allocatedSize = 0;
    for (auto& slice : bufferSlices) {
        allocatedSize += slice.second.size;
    }
    return allocatedSize;
}

uint64_t BufferPool::GetBufferOffset(int memId) { return bufferSlices.at(memId).offset; }

uint64_t BufferPool::GetBufferSize(int memId) { return bufferSlices.at(memId).size; }

bool BufferPool::CheckBufferSlicesOverlap()
{
    if (bufferSlices.size() <= 1) {
        return false;
    }
    std::vector<BufferSlice> items;
    items.reserve(bufferSlices.size());
    for (const auto& kv : bufferSlices) {
        items.push_back(kv.second);
    }
    std::sort(items.begin(), items.end(), [](BufferSlice& a, BufferSlice& b) {
        if (a.offset != b.offset) {
            return a.offset < b.offset;
        }
        return a.offset + a.size < b.offset + b.size;
    });
    auto prevEnd = items[0].offset + items[0].size;
    for (size_t i = 1; i < items.size(); ++i) {
        if (items[i].offset < prevEnd) {
            return true;
        }
        prevEnd = items[i].offset + items[i].size;
    }
    return false;
}

Status BufferPool::ModifyBufferRange(LocalBufferPtr localBuffer, size_t offset)
{
    // 调整localbuffer range
    localBuffer->start = offset;
    localBuffer->end = offset + localBuffer->size;
    // 调整bufferslice range
    auto it = bufferSlices.find(localBuffer->id);
    if (it != bufferSlices.end()) {
        it->second.offset = offset;
    } else {
        BufferSlice newSlice;
        newSlice.size = localBuffer->size;
        newSlice.offset = offset;
        bufferSlices[localBuffer->id] = newSlice;
    }
    if (CheckBufferSlicesOverlap()) {
        APASS_LOG_WARN_F(Elements::Tensor, "BufferSlices have overlap, ModifyBufferRange failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status BufferPool::CompactBufferSlices(std::unordered_map<int, LocalBufferPtr>& localBufferMap,
                                       std::vector<BufferAddrChange>& changes)
{
    if (bufferSlices.empty()) {
        return SUCCESS;
    }
    // 收集并按原 size 从大到小排序；同时记录每个 memId 的旧 offset
    std::vector<std::pair<int, BufferSlice>> items(bufferSlices.begin(), bufferSlices.end());
    std::unordered_map<int, uint64_t> oldOffsets;
    for (const auto& it : items) {
        oldOffsets[it.first] = it.second.offset;
    }
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.second.size > b.second.size; });

    // 紧凑重排
    uint64_t cursor = 0;
    for (auto& it : items) {
        if (cursor + it.second.size > memSize_) {
            return FAILED;
        }
        it.second.offset = cursor;
        cursor += it.second.size;
    }

    // 写回 + 收集 changes（只记录实际移动了的 slice）
    for (const auto& it : items) {
        auto memId = it.first;
        auto newOff = it.second.offset;
        auto sliceSize = it.second.size;
        auto oldOff = oldOffsets[memId];
        bufferSlices[memId].offset = newOff;

        auto localBufferIt = localBufferMap.find(memId);
        if (localBufferIt != localBufferMap.end() && localBufferIt->second) {
            auto& localBuffer = localBufferIt->second;
            localBuffer->start = newOff;
            localBuffer->end = newOff + localBuffer->size;
        } else {
            APASS_LOG_WARN_F(
                Elements::Tensor,
                "CompactBufferSlices: missing LocalBufferPtr for memId=%d, only updated bufferSlices offset", memId);
        }
        if (oldOff != newOff) {
            changes.push_back({memId, oldOff, oldOff + sliceSize, newOff, newOff + sliceSize});
        }
    }
    if (CheckBufferSlicesOverlap()) {
        return FAILED;
    }
    return SUCCESS;
}

void BufferPool::PrintStatus()
{
    std::vector<int> memIdList;
    for (auto& [memId, slice] : bufferSlices) {
        (void)slice;
        memIdList.push_back(memId);
    }
    std::sort(memIdList.begin(), memIdList.end(),
              [&](int a, int b) { return bufferSlices[a].offset < bufferSlices[b].offset; });

    uint64_t lastEnd = 0;
    for (auto memId : memIdList) {
        auto& slice = bufferSlices[memId];
        if (slice.offset != lastEnd) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "      |--- Space : [%lu, %lu], Size : %lu", lastEnd, slice.offset,
                              slice.offset - lastEnd);
        }
        APASS_LOG_DEBUG_F(Elements::Tensor, "  |--- MemId : %d, Span : [%lu, %lu], Size : %lu", memId, slice.offset,
                          slice.offset + slice.size, slice.size);
        lastEnd = slice.offset + slice.size;
    }
    if (lastEnd != memSize_) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "      |--- Space : [%lu, %lu], Size : %lu", lastEnd, memSize_,
                          memSize_ - lastEnd);
    }
}

// === DualDst 扩展 ===
// 把 FindFreeIntervals 返回的 size -> {start,end} 桶状 map 扁平化成
// 按 start 升序的 [start, end) 列表。OoOScheduler::FindCommonFreeOffset 在两池列表上
// 归并求交集;原 BufferPool::FindCommonFreeOffset (跨池决策) 已上提至 OoOScheduler。
std::vector<std::pair<uint64_t, uint64_t>> BufferPool::GetSortedFreeIntervals()
{
    auto freeMap = FindFreeIntervals();
    std::vector<std::pair<uint64_t, uint64_t>> v;
    for (auto& [sz, ranges] : freeMap) {
        (void)sz;
        for (auto& kv : ranges) {
            v.emplace_back(kv.first, kv.second);
        }
    }
    std::sort(v.begin(), v.end(), [](const std::pair<uint64_t, uint64_t>& a, const std::pair<uint64_t, uint64_t>& b) {
        return a.first < b.first;
    });
    return v;
}

Status BufferPool::AllocateAtOffset(LocalBufferPtr tensor, uint64_t offset)
{
    if (tensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "AllocateAtOffset: null tensor");
        return FAILED;
    }
    if (offset + tensor->size > memSize_) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "AllocateAtOffset: Tensor[%d] (offset %lu, size %lu) exceeds pool size %lu.", tensor->id,
                          offset, tensor->size, memSize_);
        return FAILED;
    }
    if (bufferSlices.find(tensor->id) != bufferSlices.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "AllocateAtOffset: Tensor[%d] already allocated.", tensor->id);
        return FAILED;
    }
    // 先保存 start/end 旧值, 用于 overlap 检查失败时回滚, 避免 tensor 处于
    // "已修改 start/end 但未实际占位" 的不一致状态。
    const uint64_t savedStart = tensor->start;
    const uint64_t savedEnd = tensor->end;

    BufferSlice ns(offset, tensor->size);
    bufferSlices[tensor->id] = ns;
    tensor->start = offset;
    tensor->end = offset + tensor->size;
    if (CheckBufferSlicesOverlap()) {
        APASS_LOG_WARN_F(Elements::Tensor, "AllocateAtOffset: Tensor[%d] at offset %lu overlaps existing slices.",
                         tensor->id, offset);
        bufferSlices.erase(tensor->id);
        tensor->start = savedStart;
        tensor->end = savedEnd;
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, " AllocateAtOffset Tensor[%d], range [%lu, %lu].", tensor->id, offset,
                      offset + tensor->size);
    return SUCCESS;
}

} // namespace npu::tile_fwk
