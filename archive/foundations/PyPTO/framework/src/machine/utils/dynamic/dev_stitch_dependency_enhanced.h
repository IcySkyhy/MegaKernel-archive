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
 * \file dev_stitch_dependency_enhanced.h
 * \brief Enhanced Stitch dependency processing with unified functions
 *
 * Design principles:
 * - Multi-concurrent types: Atomic-Write, Read (same type non-mutex, support multiple ops)
 * - Exclusive type: Normal-Write (all types mutex, only 1 op)
 * - Unified functions with opType parameter to reduce code duplication
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_types.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "machine/utils/dynamic/dev_encode_function_stitch.h"
#include "tilefwk/aicpu_common.h"
#include "tilefwk/aikernel_data.h"

namespace npu::tile_fwk::dynamic {

// Entry layout (uint64):
//   [63:48] epoch | [47:32] devTaskId[15:0] | [31:16] producerFuncIdx | [15:0] consumerFuncIdx
// Epoch replaces full-cache memset: previous-launch entries never match the current epoch.
// When monotonic taskId crosses a 16-bit boundary, runtime bumps epoch (AdvanceStitchCacheEpoch)
// so packing only the low 16 bits remains collision-free within a launch.
inline bool CheckStitchCacheDuplicate(uint64_t* stitchCache, uint32_t rootFuncMaxCallOpsize, uint16_t stitchCacheEpoch,
                                      uint32_t producerFuncIdx, uint32_t producerOpIdx, uint32_t consumerFuncIdx,
                                      uint32_t consumerOpIdx, uint64_t devTaskId)
{
    if (stitchCache == nullptr || rootFuncMaxCallOpsize == 0) {
        return false;
    }
    uint64_t& entry = stitchCache[consumerOpIdx * rootFuncMaxCallOpsize + producerOpIdx];
    uint64_t expected = (static_cast<uint64_t>(stitchCacheEpoch) << 48) | ((devTaskId & 0xFFFFULL) << 32) |
                        (static_cast<uint64_t>(producerFuncIdx & 0xFFFFU) << 16) | (consumerFuncIdx & 0xFFFFU);
    if (entry == expected) {
        return true;
    }
    entry = expected;
    return false;
}

inline void EstablishDependenciesWithType(uint64_t cellMemBase, uint64_t* cellMatchTableData, uint32_t opType,
                                          uint32_t opCount, DevAscendFunctionDupped* stitchingList, int stitchingSize,
                                          DevAscendFunctionDupped* currDup, uint64_t tagId, uint64_t stitchDevTaskId,
                                          size_t devCurrIdx, uint32_t operationIdx, DeviceWorkspaceAllocator* workspace,
                                          uint64_t* matchCount, int slotIdx, const DevCellMatchTableDesc& desc)
{
    (void)stitchingSize;
    (void)tagId;
    for (uint32_t i = 0; i < opCount; i++) {
        uint64_t opId = CellMatchGetOpId(cellMatchTableData, cellMemBase, opType, i, desc);
        if (opId != AICORE_TASK_INIT) {
            auto funcIdx = FuncID(static_cast<uint32_t>(opId));
            auto opIdx = TaskID(static_cast<uint32_t>(opId));

            if (funcIdx >= static_cast<uint32_t>(stitchingSize)) {
                continue;
            }

            (*matchCount)++;
            DeviceStitchContext::HandleOneStitch(
                stitchingList[funcIdx], *currDup, funcIdx, opIdx, devCurrIdx, operationIdx, workspace,
                DeviceStitchContext::StitchKind::StitchDefault, slotIdx, stitchDevTaskId);

            DEV_VERBOSE_DEBUG("Stitch dependency: (%u!%u) -> (%zu!%u), cellopType=%u, cellopCount=%u, "
                              "slotIdx=%d, opTagid=%x, curOpTagid=%lx",
                              funcIdx, opIdx, devCurrIdx, operationIdx, opType, opCount, slotIdx,
                              static_cast<uint32_t>(opId >> CELL_MATCH_META_TAGID_SHIFT32), tagId);
        }
    }
}

inline void CellMatchStitchHandle(int cellIndex, uint64_t* cellMatchTableData, uint32_t myOpType,
                                  DevAscendFunctionDupped* stitchingList, int stitchingSize,
                                  DevAscendFunctionDupped* currDup, uint64_t tagId, uint64_t stitchDevTaskId,
                                  size_t devCurrIdx, uint32_t operationIdx, DeviceWorkspaceAllocator* workspace,
                                  uint64_t* matchCount, int slotIdx, const DevCellMatchTableDesc& desc)
{
    DEV_VERBOSE_DEBUG(
        "CellMatchStitchHandle: cell[%d], cellMatchTableData=%p, myOpType=%u, mytagId=%lx, slotIdx=%d, myTaskid=%x",
        cellIndex, cellMatchTableData, myOpType, tagId, slotIdx, MakeTaskID(devCurrIdx, operationIdx));

    uint64_t cellMemBase = CellMatchCellIndexToMemBase(static_cast<uint64_t>(cellIndex), desc);
    uint64_t meta = cellMatchTableData[cellMemBase];

    uint32_t curActiveOpType = CellMatchGetCurrentOpType(meta);
    if (curActiveOpType == CELL_MATCH_OP_TYPE_NONE) {
        DEV_VERBOSE_DEBUG("CellMatchStitchHandle: cell[%d] early return, curActiveOpType=NONE", cellIndex);
        return;
    }

    if (CellMatchGetTagId(meta) != tagId) {
        DEV_VERBOSE_DEBUG("CellMatchStitchHandle: cell[%d] early return, storedTagId=%lx != tagId=%lx", cellIndex,
                          CellMatchGetTagId(meta), tagId);
        return;
    }

    uint32_t curActiveOpCount = CellMatchGetCurrentOpCount(meta);
    if (CellMatchIsMutexOp(myOpType, curActiveOpType)) {
        DEV_VERBOSE_DEBUG(
            "CellMatchStitchHandle: cell[%d] mutex op, establish deps for curActiveOpType=%u curActiveOpCnt=%u",
            cellIndex, curActiveOpType, curActiveOpCount);
        EstablishDependenciesWithType(cellMemBase, cellMatchTableData, curActiveOpType, curActiveOpCount, stitchingList,
                                      stitchingSize, currDup, tagId, stitchDevTaskId, devCurrIdx, operationIdx,
                                      workspace, matchCount, slotIdx, desc);
        return;
    }

    uint32_t prevMutexType = CellMatchGetPrevMutexOpType(meta);
    uint32_t prevMutexCount = CellMatchGetPrevMutexOpCount(meta);
    if (prevMutexType != CELL_MATCH_OP_TYPE_NONE && prevMutexCount > 0 &&
        prevMutexCount != CELL_MATCH_INVALID_OP_COUNT) {
        DEV_VERBOSE_DEBUG(
            "CellMatchStitchHandle: cell[%d] prev mutex, establish deps for prevMutexType=%u, prevMutexCount=%u",
            cellIndex, prevMutexType, prevMutexCount);
        EstablishDependenciesWithType(cellMemBase, cellMatchTableData, prevMutexType, prevMutexCount, stitchingList,
                                      stitchingSize, currDup, tagId, stitchDevTaskId, devCurrIdx, operationIdx,
                                      workspace, matchCount, slotIdx, desc);
    }
}

template <typename... TyArgs>
static void CellMatchStitchEnhance(const uint64_t offset[DEV_SHAPE_DIM_MAX], const uint64_t shape[DEV_SHAPE_DIM_MAX],
                                   const DevCellMatchTableDesc& cellMatchTableDesc, uint32_t opType, TyArgs... args)
{
    if constexpr (sizeof...(args) == 11) {
        auto argsTuple = std::make_tuple(args...);
        uint64_t* cellMatchTableData = std::get<0>(argsTuple);
        DevAscendFunctionDupped* stitchingList = std::get<1>(argsTuple);
        int stitchingSize = std::get<2>(argsTuple);
        DevAscendFunctionDupped* currDup = std::get<3>(argsTuple);
        uint64_t tagId = std::get<4>(argsTuple);
        uint64_t stitchDevTaskId = std::get<5>(argsTuple);
        size_t devCurrIdx = std::get<6>(argsTuple);
        DeviceWorkspaceAllocator* workspace = std::get<7>(argsTuple);
        uint32_t operationIdx = std::get<8>(argsTuple);
        int slotIdx = std::get<9>(argsTuple);
        uint64_t* matchCount = std::get<10>(argsTuple);

        struct HandleStitch {
            static inline uint32_t Process(int cellIndex, const DevCellMatchTableDesc& desc, uint64_t* data,
                                           DevAscendFunctionDupped* list, int size, DevAscendFunctionDupped* dup,
                                           uint64_t tag, uint64_t stitchDevTaskId, size_t currIdx, uint32_t opIdx,
                                           DeviceWorkspaceAllocator* ws, uint64_t* cnt, uint32_t type, int slot)
            {
                CellMatchStitchHandle(cellIndex, data, type, list, size, dup, tag, stitchDevTaskId, currIdx, opIdx, ws,
                                      cnt, slot, desc);
                return 0;
            }
        };

        CellMatchHandle<HandleStitch>(offset, shape, cellMatchTableDesc, cellMatchTableData, stitchingList,
                                      stitchingSize, currDup, tagId, stitchDevTaskId, devCurrIdx, operationIdx,
                                      workspace, matchCount, opType, slotIdx);
    }
}

} // namespace npu::tile_fwk::dynamic
