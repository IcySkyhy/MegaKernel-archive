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
 * \file dev_cell_match_mem_layout.h
 * \brief Cell Match memory layout definitions for enhanced stitch mechanism
 *
 * Memory Layout Per Cell (dynamic, see DevCellMatchTableDesc::cellUint64Size):
 * - uint64[0]: Metadata control word
 *   - bit 0-3:   Current active operation type (0xFF=NONE, 1=NORMAL_WRITE, 2=ATOMIC_WRITE, 3=READ)
 *   - bit 4-7:   Previous mutex operation type (for dependency tracking)
 *   - bit 8-15:  Current active operation count (8-bit field, max storable value 255)
 *   - bit 16-23: Previous mutex operation count (8-bit field, max storable value 255)
 *   - bit 32-63: Tag id bit (for dependency validation)
 *      - bit 32-41: slot alloc iter id, When the slot tensor memory is reallocated, the iter id changes accordingly
 *      - bit 42-63: Stitch devtask id
 * - uint64[1]:              Normal-write op-id (single op)
 * - uint64[2..]:            Atomic-write op-id list (max CELL_MATCH_MAX_ATOMIC_WRITE_COUNT ops)
 * - uint64[2+atomicMax..]:  Read op-id list (max CELL_MATCH_MAX_READ_COUNT ops)
 */

#pragma once

#include <cstdint>
#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"

namespace npu::tile_fwk::dynamic {

#define CELL_MATCH_META_TAGID_SHIFT32 32
#define CELL_MATCH_META_TAG_SLOT_ALLOC_ITER_ID_SHIFT 22
#define CELL_MATCH_TAG_SEQ_MAX (1u << CELL_MATCH_META_TAG_SLOT_ALLOC_ITER_ID_SHIFT)

// Operation Type Definitions
#define CELL_MATCH_OP_TYPE_NONE 0xF
#define CELL_MATCH_OP_TYPE_NORMAL_WRITE 0
#define CELL_MATCH_OP_TYPE_ATOMIC_WRITE 1
#define CELL_MATCH_OP_TYPE_READ 2

enum CellMatchOpType : uint32_t {
    NONE = CELL_MATCH_OP_TYPE_NONE,
    NORMAL_WRITE = CELL_MATCH_OP_TYPE_NORMAL_WRITE,
    ATOMIC_WRITE = CELL_MATCH_OP_TYPE_ATOMIC_WRITE,
    READ = CELL_MATCH_OP_TYPE_READ
};

/*
 * Operation count limits.
 * Metadata stores op count in 8 bits (bit 8-15 / 16-23): storable range is 0..255.
 * Atomic-write batch cap is 254 so 255 (0xFF) remains available as prev-mutex invalid sentinel.
 */
#define CELL_MATCH_METADATA_OP_COUNT_MAX 255
#define CELL_MATCH_MAX_NORMAL_WRITE_COUNT 1
#define CELL_MATCH_MAX_ATOMIC_WRITE_COUNT 254
#define CELL_MATCH_MAX_READ_COUNT 128
#define CELL_MATCH_INVALID_OP_COUNT 0xFF

/*
  Metadata Control Word Bit Layout (uint64_t)
  bit 0-3:   Current active operation types
  bit 4-7:   Previous mutex operation type
  bit 8-15:  Current active operation count
  bit 16-23: Previous mutex operation count
  bit 24-31: Reserved
  bit 32-63: Tag bit (for dependency validation)
*/
#define CELL_MATCH_BIT_CUR_OP_TYPE_START 0
#define CELL_MATCH_BIT_CUR_OP_TYPE_END 3
#define CELL_MATCH_BIT_PREV_MUTEX_OPTYPE_START 4
#define CELL_MATCH_BIT_PREV_MUTEX_OPTYPE_END 7
#define CELL_MATCH_BIT_CUR_OP_COUNT_START 8
#define CELL_MATCH_BIT_CUR_OP_COUNT_END 15
#define CELL_MATCH_BIT_PREV_MUTEX_OPCOUNT_START 16
#define CELL_MATCH_BIT_PREV_MUTEX_OPCOUNT_END 23
#define CELL_MATCH_BIT_RESERVED_START 24 // reserved
#define CELL_MATCH_BIT_RESERVED_END 31   // reserved
#define CELL_MATCH_BIT_TAG_ID_START 32
#define CELL_MATCH_BIT_TAG_ID_END 63

#define CELL_MATCH_GET_BITS(value, start_bit, end_bit) \
    (((value) >> (start_bit)) & ((1ULL << ((end_bit) - (start_bit) + 1)) - 1ULL))

#define CELL_MATCH_SET_BITS(value, start_bit, end_bit, new_val)                          \
    do {                                                                                 \
        uint64_t mask = ((1ULL << ((end_bit) - (start_bit) + 1)) - 1ULL) << (start_bit); \
        (value) = ((value) & ~mask) | (((uint64_t)(new_val) << (start_bit)) & mask);     \
    } while (0)

inline uint32_t CellMatchGetCurrentOpType(uint64_t meta)
{
    return (uint32_t)CELL_MATCH_GET_BITS(meta, CELL_MATCH_BIT_CUR_OP_TYPE_START, CELL_MATCH_BIT_CUR_OP_TYPE_END);
}

inline void CellMatchSetCurrentOpType(uint64_t& meta, uint32_t opType)
{
    CELL_MATCH_SET_BITS(meta, CELL_MATCH_BIT_CUR_OP_TYPE_START, CELL_MATCH_BIT_CUR_OP_TYPE_END, opType);
}

inline uint32_t CellMatchGetPrevMutexOpType(uint64_t meta)
{
    return (uint32_t)CELL_MATCH_GET_BITS(meta, CELL_MATCH_BIT_PREV_MUTEX_OPTYPE_START,
                                         CELL_MATCH_BIT_PREV_MUTEX_OPTYPE_END);
}

inline void CellMatchSetPrevMutexOpType(uint64_t& meta, uint32_t mutexType)
{
    CELL_MATCH_SET_BITS(meta, CELL_MATCH_BIT_PREV_MUTEX_OPTYPE_START, CELL_MATCH_BIT_PREV_MUTEX_OPTYPE_END, mutexType);
}

inline uint32_t CellMatchGetCurrentOpCount(uint64_t meta)
{
    return (uint32_t)CELL_MATCH_GET_BITS(meta, CELL_MATCH_BIT_CUR_OP_COUNT_START, CELL_MATCH_BIT_CUR_OP_COUNT_END);
}

inline void CellMatchSetCurrentOpCount(uint64_t& meta, uint32_t count)
{
    CELL_MATCH_SET_BITS(meta, CELL_MATCH_BIT_CUR_OP_COUNT_START, CELL_MATCH_BIT_CUR_OP_COUNT_END, count);
}

inline uint32_t CellMatchGetPrevMutexOpCount(uint64_t meta)
{
    return (uint32_t)CELL_MATCH_GET_BITS(meta, CELL_MATCH_BIT_PREV_MUTEX_OPCOUNT_START,
                                         CELL_MATCH_BIT_PREV_MUTEX_OPCOUNT_END);
}

inline void CellMatchSetPrevMutexOpCount(uint64_t& meta, uint32_t count)
{
    CELL_MATCH_SET_BITS(meta, CELL_MATCH_BIT_PREV_MUTEX_OPCOUNT_START, CELL_MATCH_BIT_PREV_MUTEX_OPCOUNT_END, count);
}

inline uint32_t CellMatchBuildTagId(uint32_t slotAllocIterId, uint32_t cellMatchTagSeq)
{
    return (static_cast<uint32_t>(slotAllocIterId) << CELL_MATCH_META_TAG_SLOT_ALLOC_ITER_ID_SHIFT) | cellMatchTagSeq;
}

inline uint32_t CellMatchGetCellMatchTagSeqFromTagId(uint32_t tagId)
{
    return (uint32_t)CELL_MATCH_GET_BITS(tagId, 0, CELL_MATCH_META_TAG_SLOT_ALLOC_ITER_ID_SHIFT - 1);
}

inline uint32_t CellMatchGetDevTaskIdFromTagId(uint32_t tagId) { return CellMatchGetCellMatchTagSeqFromTagId(tagId); }

inline uint64_t CellMatchGetTagId(uint64_t meta)
{
    return CELL_MATCH_GET_BITS(meta, CELL_MATCH_BIT_TAG_ID_START, CELL_MATCH_BIT_TAG_ID_END);
}

inline void CellMatchSetTagId(uint64_t& meta, uint64_t devTaskId)
{
    CELL_MATCH_SET_BITS(meta, CELL_MATCH_BIT_TAG_ID_START, CELL_MATCH_BIT_TAG_ID_END, devTaskId);
}

inline bool CellMatchIsMutexOp(uint32_t curOpType, uint32_t prepType)
{
    if (curOpType == prepType) {
        return (curOpType == CELL_MATCH_OP_TYPE_NORMAL_WRITE);
    }
    return true;
}

inline uint64_t CellMatchCellIndexToMemBase(uint64_t cellIndex, const DevCellMatchTableDesc& desc)
{
    return cellIndex * desc.cellUint64Size;
}

inline uint64_t CellMatchGetOpId(uint64_t* cellMatchTableData, uint64_t cellMemBase, uint32_t opType, uint32_t index,
                                 const DevCellMatchTableDesc& desc)
{
    return cellMatchTableData[cellMemBase + desc.opMemLayOutIndex[opType] + index];
}

inline void CellMatchAddOpId(uint64_t* cellMatchTableData, uint64_t cellMemBase, uint64_t taskId, uint32_t index,
                             uint32_t opType, const DevCellMatchTableDesc& desc)
{
    cellMatchTableData[cellMemBase + desc.opMemLayOutIndex[opType] + index] = taskId;
}

} // namespace npu::tile_fwk::dynamic
