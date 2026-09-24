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
 * \file buffer_rearrange.cpp
 * \brief
 */
#include <climits>
#include "buffer_rearrange.h"

namespace npu {
namespace tile_fwk {
struct MoveFrontInfo {
    size_t l;
    size_t r;
    size_t cost;
    size_t moveToLeftStart;
    bool found;
};

inline void CopyVector(std::vector<int>& dst, const std::vector<int>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

inline void CopyMap(std::unordered_map<int, size_t>& dst, const std::unordered_map<int, size_t>& src)
{
    for (auto [a, b] : src) {
        dst[a] = b;
    }
}

void MoveFrontBuffer(RearrangeScheme& base, size_t sizeNeeded, MoveFrontInfo& moveInfo, size_t lStart,
                     RearrangeScheme& newScheme)
{
    bool loopFlag = true;
    while (loopFlag) {
        int rMemId = base.memIds[moveInfo.r];
        size_t rMemOffset = base.moveFrom[rMemId];
        size_t rMemSize = base.memSizeMap[rMemId];
        if ((moveInfo.moveToLeftStart + rMemSize) <= lStart) { // 如果能提前，
            newScheme.moveTo[rMemId] = moveInfo.moveToLeftStart;
            moveInfo.moveToLeftStart += rMemSize;
            moveInfo.cost += rMemSize;
        } else {
            loopFlag = false;
            continue;
        }
        if (moveInfo.r == moveInfo.l) {
            if ((moveInfo.moveToLeftStart + sizeNeeded) <= base.end) {
                moveInfo.found = true;
            }
            loopFlag = false;
            continue;
        } else {
            moveInfo.r -= 1;
            rMemId = base.memIds[moveInfo.r];
            rMemOffset = base.moveFrom[rMemId];
            rMemSize = base.memSizeMap[rMemId];
            // 如果尾部形成的空间已经满足条件，则判定为已找到结果。
            if ((rMemOffset + rMemSize + sizeNeeded) <= base.end) {
                moveInfo.found = true;
                loopFlag = false;
                continue;
            }
        }
    }
}

// 尝试在base scheme上获取cost更少的Scheme
// 尝试直接将区间内最靠后的buffer提前，而前面的buffer不改变位置。
//    from : | *** |  a  | *** |  b  | *** |
//    to :   |  b  |  a  | *** | *** | *** |
RearrangeScheme TryMoveBackToFront(RearrangeScheme base, size_t sizeNeeded)
{
    RearrangeScheme newScheme;
    newScheme.start = base.start;
    newScheme.end = base.end;
    CopyVector(newScheme.memIds, base.memIds);
    CopyMap(newScheme.moveFrom, base.moveFrom);
    CopyMap(newScheme.moveTo, base.moveFrom);
    CopyMap(newScheme.memSizeMap, base.memSizeMap);
    if (base.memIds.size() <= 1) {
        RearrangeScheme failed;
        return failed;
    }
    MoveFrontInfo moveInfo;
    // 指针，将r指向的元素移动到l指向的元素前面。
    moveInfo.l = 0;
    moveInfo.r = base.memIds.size() - 1;
    // Buffer前移的起始偏移
    moveInfo.moveToLeftStart = base.start;
    moveInfo.cost = 0;
    moveInfo.found = false;
    bool loopFlag = true;
    while (loopFlag) {
        int lMemId = base.memIds[moveInfo.l];
        size_t lMemSize = base.memSizeMap[lMemId];
        size_t lStart = base.moveFrom[lMemId];
        // 将后面的buffer一个个往前面移，到当前 lMemId 这个Buffer的前面
        MoveFrontBuffer(base, sizeNeeded, moveInfo, lStart, newScheme);
        if (moveInfo.found || moveInfo.r == moveInfo.l) {
            loopFlag = false;
            continue;
        }
        // 更新移动到前面的起始偏移
        moveInfo.moveToLeftStart = lStart + lMemSize;
        moveInfo.l += 1;
    }
    if (!moveInfo.found) {
        RearrangeScheme failed;
        return failed;
    }
    newScheme.cost = moveInfo.cost;
    return newScheme;
}

inline size_t CalcRevertOffset(size_t start, size_t end, size_t offset, size_t size)
{
    return start + (end - offset - size);
}

// 尝试在base Scheme上获取cost更少的Scheme
// 尝试直接将区间内最靠前的buffer挪到后面，而后面的buffer不改变位置。
//    from : | *** |  a  | *** |  b  | *** |
//    to :   | *** | *** | *** |  b  |  a  |
RearrangeScheme TryMoveFrontToBack(RearrangeScheme base, size_t sizeNeeded)
{
    RearrangeScheme revertScheme;
    revertScheme.start = base.start;
    revertScheme.end = base.end;
    CopyVector(revertScheme.memIds, base.memIds);
    std::reverse(revertScheme.memIds.begin(), revertScheme.memIds.end());
    CopyMap(revertScheme.memSizeMap, base.memSizeMap);
    for (auto memId : base.memIds) {
        revertScheme.moveFrom[memId] = CalcRevertOffset(base.start, base.end, base.moveFrom[memId],
                                                        base.memSizeMap[memId]);
    }
    CopyMap(revertScheme.moveTo, revertScheme.moveFrom);
    RearrangeScheme newScheme = TryMoveBackToFront(revertScheme, sizeNeeded);
    if (newScheme.cost != INT_MAX) {
        std::reverse(newScheme.memIds.begin(), newScheme.memIds.end());
        for (auto memId : base.memIds) {
            newScheme.moveFrom[memId] = CalcRevertOffset(newScheme.start, newScheme.end, newScheme.moveFrom[memId],
                                                         newScheme.memSizeMap[memId]);
            newScheme.moveTo[memId] = CalcRevertOffset(newScheme.start, newScheme.end, newScheme.moveTo[memId],
                                                       newScheme.memSizeMap[memId]);
        }
    }
    return newScheme;
}

void GroupBubbles(std::vector<std::pair<size_t, size_t>>& bubbleGroups, size_t sizeNeeded,
                  const std::vector<size_t>& bubbleSizeList, bool& failed)
{
    // 将碎片分组，满足每组碎片大小的和 >= 需要分配的大小。
    size_t start = 0;
    for (; start < bubbleSizeList.size(); start++) {
        size_t end = start;
        size_t groupBubbleSize = 0;
        // 扩充group的右边界，使得其总大小 >= 需要分配的大小。
        while (end < bubbleSizeList.size() && groupBubbleSize < sizeNeeded) {
            groupBubbleSize += bubbleSizeList[end];
            end += 1;
        }
        if (groupBubbleSize < sizeNeeded) {
            break;
        }
        // 收缩group的左边界，使得其总大小刚好 >= 需要分配的大小。
        while (start < bubbleSizeList.size() && (groupBubbleSize - bubbleSizeList[start]) >= sizeNeeded) {
            groupBubbleSize -= bubbleSizeList[start];
            start += 1;
        }
        if (start + 1 >= end) { // 至少要选中两个气泡作为一组
            APASS_LOG_WARN_F(Elements::Tensor,
                             "Rerange buffer unexpected result: only choose one bubble for rearange.");
            APASS_LOG_WARN_F(Elements::Tensor, "sizeNeeded : %zu.", sizeNeeded);
            failed = true;
            break;
        }
        bubbleGroups.push_back(std::make_pair(start, end));
        start += 1;
    }
}

RearrangeScheme MoveScheme(BufferPool& bufferManager, const std::vector<std::pair<size_t, size_t>>& rearangeSpan,
                           const std::vector<int>& memIds, size_t sizeNeeded)
{
    RearrangeScheme bestScheme;
    for (auto [memStart, memEnd] : rearangeSpan) {
        // 获取基础Scheme
        // 将区间内buffer按顺序排到区间前面连续地址内
        //    from : | *** |  a  | *** |  b  | *** |
        //    to :   |  a  |  b  | *** | *** | *** |
        RearrangeScheme baseScheme;
        baseScheme.start = memStart;
        baseScheme.end = memEnd;
        baseScheme.cost = 0;
        size_t offset = memStart;

        for (auto memId : memIds) {
            auto bufOffset = bufferManager.GetBufferOffset(memId);
            auto bufSize = bufferManager.GetBufferSize(memId);
            if (bufOffset >= memStart && (bufOffset + bufSize) <= memEnd) {
                baseScheme.memIds.push_back(memId);
                baseScheme.memSizeMap[memId] = bufSize;
                baseScheme.moveFrom[memId] = bufOffset;
                baseScheme.moveTo[memId] = offset;
                offset += bufSize;
                baseScheme.cost += bufSize;
            }
        }

        baseScheme.PrintScheme();
        if (bestScheme.cost == INT_MAX || baseScheme.cost < bestScheme.cost) {
            bestScheme = baseScheme;
        }

        RearrangeScheme newScheme = TryMoveBackToFront(baseScheme, sizeNeeded);
        if (bestScheme.cost == INT_MAX || newScheme.cost < bestScheme.cost) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Found better scheme by TryMoveBackToFront.");
            bestScheme = newScheme;
            newScheme.PrintScheme();
        }

        newScheme = TryMoveFrontToBack(baseScheme, sizeNeeded);
        if (bestScheme.cost == INT_MAX || newScheme.cost < bestScheme.cost) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Found better scheme by TryMoveFrontToBack.");
            bestScheme = newScheme;
            newScheme.PrintScheme();
        }
    }
    return bestScheme;
}

void GenOrderedMoveTo(RearrangeScheme& bestScheme)
{
    std::vector<std::pair<int, size_t>> vec;
    vec.reserve(bestScheme.moveTo.size());
    for (const auto& kv : bestScheme.moveTo) {
        vec.emplace_back(kv.first, kv.second);
    }
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) {
            return a.second < b.second;
        }
        return a.first < b.first;
    });
    bestScheme.orderedMoveTo = vec;
}

RearrangeScheme GetRearrangeScheme(BufferPool& bufferManager, size_t sizeNeeded)
{
    RearrangeScheme bestScheme;
    std::vector<int> memIds = bufferManager.GetAddrSortedBufs();
    auto totalSize = bufferManager.GetMemSize();
    auto allocedSize = bufferManager.GetAllocatedSize();
    if ((totalSize - allocedSize) < sizeNeeded) {
        return bestScheme;
    }

    std::vector<std::pair<size_t, size_t>> bubbleList; // [(free_start, free_end), ...]
    std::vector<size_t> bubbleSizeList;

    size_t lastEnd = 0;
    for (auto memId : memIds) {
        auto bufOffset = bufferManager.GetBufferOffset(memId);
        auto bufSize = bufferManager.GetBufferSize(memId);
        if (bufOffset != lastEnd) {
            bubbleList.push_back(std::make_pair(lastEnd, bufOffset));
            bubbleSizeList.push_back(bufOffset - lastEnd);
        }
        lastEnd = bufOffset + bufSize;
    }
    if (lastEnd != totalSize) {
        bubbleList.push_back(std::make_pair(lastEnd, totalSize));
        bubbleSizeList.push_back(totalSize - lastEnd);
    }

    // 将碎片分组，满足每组碎片大小的和 >= 需要分配的大小。
    std::vector<std::pair<size_t, size_t>> bubbleGroups;
    bool failedFlag{false};
    GroupBubbles(bubbleGroups, sizeNeeded, bubbleSizeList, failedFlag);
    if (failedFlag) {
        return bestScheme;
    }
    std::vector<std::pair<size_t, size_t>> rearangeSpan;
    for (size_t g = 0; g < bubbleGroups.size(); g++) {
        rearangeSpan.push_back(
            std::make_pair(bubbleList[bubbleGroups[g].first].first, bubbleList[bubbleGroups[g].second - 1].second));
    }
    bestScheme = MoveScheme(bufferManager, rearangeSpan, memIds, sizeNeeded);

    // 将moveTo的offset从大到小排序并赋值给orderedMoveTo
    GenOrderedMoveTo(bestScheme);

    return bestScheme;
}
} // namespace tile_fwk
} // namespace npu
