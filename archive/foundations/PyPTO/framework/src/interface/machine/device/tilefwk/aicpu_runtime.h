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
 * \file aicpu_runtime.h
 * \brief
 */

#pragma once

#include <sys/cdefs.h>
#include <cstdint>
#include <vector>

#include "tilefwk/aikernel_runtime.h"

namespace npu::tile_fwk {

using RuntimeCallEntryType = void* (*)(void*, uint64_t);

enum RuntimeCallStage {
    T_RUNTIME_CALL_ROOT_ALLOC = 0,
    T_RUNTIME_CALL_ROOT_STITCH = 1,
    T_RUNTIME_CALL_LOG = 2,
    T_RUNTIME_CALL_SHMEM_ALLOC = 3,
    T_RUNTIME_CALL_SLOT_MARK_NEED_ALLOC = 4,
    T_RUNTIME_CALL_GET_DIE_ID = 5,
    T_RUNTIME_CALL_SET_DIE_ID = 6,
    T_RUNTIME_CALL_MAX = 7,
};

using Call1EntryType = uint64_t (*)(uint64_t);

using Call2EntryType = uint64_t (*)(uint64_t, uint64_t);

using Call3EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t);

using Call4EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t);

using Call5EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

using Call5EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

#define RUNTIME_FUNCKEY_ERROR (reinterpret_cast<void*>(static_cast<uintptr_t>(2)))

#define RUNTIME_FUNCKEY_FINISH (static_cast<uint64_t>(-1))
#define RUNTIME_FUNCKEY_CACHESTOP (static_cast<uint64_t>(-2))
#define RUNTIME_FUNCKEY_LOOP_BARRIER (static_cast<uint64_t>(-3))
#define RUNTIME_FUNCKEY_PARALLEL_FOR_BEGIN (static_cast<uint64_t>(-4))
#define RUNTIME_FUNCKEY_PARALLEL_FOR_END (static_cast<uint64_t>(-5))
#define RUNTIME_FUNCRET_CACHESTOP_CONTINUE (reinterpret_cast<void*>(static_cast<uintptr_t>(0)))
#define RUNTIME_FUNCRET_CACHESTOP_RETURN (reinterpret_cast<void*>(static_cast<uintptr_t>(1)))

#define RuntimeIsLoopBegin(idx, begin) ((idx) == (begin))
#define RuntimeIsLoopEnd(idx, end) ((int64_t)(idx) >= (int64_t)(end))
#define RuntimeTernaryOP(cond, lhs, rhs) ((cond) ? (lhs) : (rhs))

__always_inline int64_t RuntimeGetViewValidShapeDim(int64_t validshape, int64_t viewOffset, int64_t viewshape)
{
    validshape -= viewOffset;
    if (validshape > viewshape)
        validshape = viewshape;
    else if (validshape < 0)
        validshape = 0;
    return validshape;
}

__always_inline int64_t RuntimeMax(int64_t input1, int64_t input2)
{
    if (input1 > input2)
        return input1;
    else
        return input2;
}

__always_inline int64_t RuntimeMin(int64_t input1, int64_t input2)
{
    if (input1 < input2)
        return input1;
    else
        return input2;
}

__always_inline int64_t RuntimeEq(int64_t input1, int64_t input2) { return input1 == input2; }

__always_inline int64_t RuntimeNe(int64_t input1, int64_t input2) { return input1 != input2; }

#define RUNTIME_IsLoopBegin(idx, begin) RuntimeIsLoopBegin((idx), (begin))
#define RUNTIME_IsLoopEnd(idx, end) RuntimeIsLoopEnd((idx), (end))

#define RUNTIME_TernaryOP(cond, lhs, rhs) RuntimeTernaryOP((cond), (lhs), (rhs))

#define RUNTIME_GetViewValidShapeDim(validShape, viewOffset, viewShape) \
    RuntimeGetViewValidShapeDim(validShape, viewOffset, viewShape)
#define RUNTIME_Max(lhs, rhs) RuntimeMax(lhs, rhs)
#define RUNTIME_Min(lhs, rhs) RuntimeMin(lhs, rhs)
#define RUNTIME_And(lhs, rhs) ((lhs) && (rhs))
#define RUNTIME_Select(cond, set, unset) ((cond) ? (set) : (unset))

#define RUNTIME_CalcLoopDieId(var, idx, loopRange, step, dieNum)    \
    int8_t DIESET_##var = 0;                                        \
    if (*loopDieId == -1) {                                         \
        *loopDieId = ((idx) / (step)) % dieNum;                     \
        if (((idx) + (step) == (loopRange)) && (*loopDieId == 0)) { \
            *loopDieId = -1;                                        \
        } else {                                                    \
            DIESET_##var = 1;                                       \
        }                                                           \
    }

#define RUNTIME_ClearLoopDieId(var) \
    do {                            \
        if (DIESET_##var) {         \
            *loopDieId = -1;        \
        }                           \
    } while (0)

#define RUNTIME_SetExpr(exprList, index, value) \
    do {                                        \
        if (exprList) {                         \
            (exprList)[index] = (value);        \
        }                                       \
    } while (0)

#define RUNTIME_RootAlloc(funcKey) runtimeCallList[RuntimeCallStage::T_RUNTIME_CALL_ROOT_ALLOC](ctx, funcKey)
#define RUNTIME_RootStitch(funcKey)                                                        \
    do {                                                                                   \
        if (runtimeCallList[RuntimeCallStage::T_RUNTIME_CALL_ROOT_STITCH](ctx, funcKey) == \
            RUNTIME_FUNCRET_CACHESTOP_RETURN) {                                            \
            return 0;                                                                      \
        }                                                                                  \
    } while (0)

#define RUNTIME_SlotMarkNeedAlloc(slotIndex)                                                    \
    do {                                                                                        \
        runtimeCallList[RuntimeCallStage::T_RUNTIME_CALL_SLOT_MARK_NEED_ALLOC](ctx, slotIndex); \
    } while (0)

#define RUNTIME_RootGetDieId(funcKey) \
    int8_t* loopDieId = (int8_t*)runtimeCallList[RuntimeCallStage::T_RUNTIME_CALL_GET_DIE_ID](ctx, funcKey);

#define RUNTIME_RootSetDieId(funcKey) runtimeCallList[RuntimeCallStage::T_RUNTIME_CALL_SET_DIE_ID](ctx, funcKey)
} // namespace npu::tile_fwk
