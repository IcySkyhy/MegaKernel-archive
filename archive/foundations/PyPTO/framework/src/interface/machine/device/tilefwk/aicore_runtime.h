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
 * \file aicore_runtime.h
 * \brief
 */

#ifndef AICORE_RUNTIME_H
#define AICORE_RUNTIME_H

#include <cstdint>

#include "tilefwk/aikernel_data.h"
#include "tilefwk/aikernel_runtime.h"
#include "tileop/distributed/comm_context.h"

using CoreFuncParam = npu::tile_fwk::CoreFuncParam;

#define CACHELINE_SIZE_FOR_B32 128
#define CACHELINE_SIZE_FOR_B64 64
#define DEFAULT_TOTAL_BLOCK_NUM 75
#define DEBUG_OFFSET_FOR_B64 DEFAULT_TOTAL_BLOCK_NUM* CACHELINE_SIZE_FOR_B64 / sizeof(int64_t)
#define DEBUG_SIZE_PER_CORE (1 * 1024 * 1024)
#define PAD_LIMIT 512

const int SHAKE_SAY_HELLO = 100;
const int SHAKE_HELLO_ACK = 200;

struct RealizedVar {
    __gm__ void* Addr;
    int64_t offset0; // NEXTNEXT: should divide by TILESIZE or not? E.g . 128 or 128/128
    int64_t offset1;
};

struct GMTensorInfo {
    uint64_t Addr;
};

template <typename T>
struct IOCastInfo {
    __gm__ T* Addr;
    int64_t Size;
};

// | GmCount | IncastCount | OutcastCount | GmArrary | IncastArray | OutcastArray |

struct InvokeEntry {
    int64_t SubGraphProgramId;
    uint64_t gmCount;
    uint64_t incastCount;
    uint64_t outcastCount;
};

template <unsigned GRAPH_INVOKE_COUNT>
struct GraphInvokeInfo {
    uint64_t GraphInvokeCount{GRAPH_INVOKE_COUNT};
    uint64_t GraphInvokeOffset[GRAPH_INVOKE_COUNT];
};

template <typename T, unsigned SIZE>
struct RingBuffer {
    T elements[SIZE];
    uint64_t MAX_SIZE{SIZE};
    char pad1[PAD_LIMIT - SIZE * sizeof(T) - 1 * sizeof(uint64_t)];
    int64_t front{0};
    char pad2[PAD_LIMIT - 1 * sizeof(int64_t)];
    int64_t rear{0};
    char pad3[PAD_LIMIT - 1 * sizeof(int64_t)];
};

template <typename T, unsigned SIZE>
INLINE void InitRingBuffer(RingBuffer<T, SIZE>* Q)
{
    Q->front = 0;
    Q->rear = 0;
    Q->MAX_SIZE = SIZE;
}

// Enqueue:
template <typename T, unsigned SIZE>
INLINE bool EnQueue(volatile __gm__ RingBuffer<T, SIZE>* Q, T value)
{
    dcci(Q, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    if (Q->rear - Q->front == Q->MAX_SIZE) {
        return false;
    }
    Q->elements[Q->rear % Q->MAX_SIZE] = value;
    dsb((mem_dsb_t)0);
    Q->rear = Q->rear + 1;
    dcci(Q, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    return true;
}

// Enqueue:
template <typename T, unsigned SIZE>
INLINE bool EnQueuePrivate(volatile RingBuffer<T, SIZE>* Q, T value)
{
    if ((Q->rear + 1) % Q->MAX_SIZE == Q->front) {
        return false;
    }
    Q->elements[Q->rear] = value;
    dsb((mem_dsb_t)0);
    Q->rear = (Q->rear + 1) % Q->MAX_SIZE;
    return true;
}

// EnQueueLocalToGM:
template <typename T, unsigned SIZE>
INLINE bool EnQueueLocalToGM(volatile __gm__ RingBuffer<T, SIZE>* Q, T value)
{
    if ((Q->rear + 1) % Q->MAX_SIZE == Q->front) {
        return false;
    }
    Q->elements[Q->rear] = value;
    dsb((mem_dsb_t)0);
    Q->rear = (Q->rear + 1) % Q->MAX_SIZE;
    dcci(Q, 0);
    return true;
}

// dequeue:
template <typename T, unsigned SIZE>
INLINE bool DeQueue(volatile __gm__ RingBuffer<T, SIZE>* Q, T& value)
{
    dcci(Q, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    if (Q->front == Q->rear) {
        return false;
    }
    value = Q->elements[Q->front % Q->MAX_SIZE];
    dsb((mem_dsb_t)0);
    Q->front = Q->front + 1;
    dcci(Q, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    return true;
}

// dequeue:
template <typename T, unsigned SIZE>
INLINE volatile T* DeQueuePrivate(volatile RingBuffer<T, SIZE>* Q)
{
    if (Q->front == Q->rear) {
        return nullptr;
    }
    volatile T* ret = &(Q->elements[Q->front]);
    Q->front = (Q->front + 1) % Q->MAX_SIZE;
    return ret;
}

// Peek:
template <typename T, unsigned SIZE>
INLINE __gm__ T* Peek(volatile __gm__ RingBuffer<T, SIZE>* Q)
{
    if (Q->front == Q->rear) {
        return nullptr;
    }
    __gm__ T* ret = &(Q->elements[Q->front]);
    return ret;
}

// dequeue:
template <typename T, unsigned SIZE>
INLINE T* PeekPrivate(volatile RingBuffer<T, SIZE>* Q)
{
    if (Q->front == Q->rear) {
        return nullptr;
    }
    T* ret = &(Q->elements[Q->front]);
    return ret;
}

template <typename T, unsigned SIZE>
INLINE bool IsFull(volatile __gm__ RingBuffer<T, SIZE>* Q)
{
    dcci(Q, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    return (Q->rear - Q->front) == Q->MAX_SIZE;
}

template <typename T, unsigned SIZE>
INLINE bool IsFullPrivate(volatile RingBuffer<T, SIZE>* Q)
{
    return (Q->rear + 1) % Q->MAX_SIZE == Q->front;
}

template <typename T, unsigned SIZE>
INLINE bool IsEmpty(volatile __gm__ RingBuffer<T, SIZE>* Q)
{
    dcci(Q, 0);
    return Q->front == Q->rear;
}

template <typename T, unsigned SIZE>
INLINE bool IsEmptyPrivate(volatile RingBuffer<T, SIZE>* Q)
{
    return Q->front == Q->rear;
}

template <typename T, unsigned SIZE>
INLINE uint64_t GetLength(volatile __gm__ RingBuffer<T, SIZE>* Q)
{
    dcci(Q, 0);
    return (Q->rear - Q->front + Q->MAX_SIZE) % Q->MAX_SIZE;
}

template <typename T, unsigned SIZE>
INLINE uint64_t GetLengthPrivate(volatile RingBuffer<T, SIZE>* Q)
{
    return (Q->rear - Q->front + Q->MAX_SIZE) % Q->MAX_SIZE;
}

#define SYM_VALUE_LEN 63
#define SYM_VALUE_MASK ((1UL << SYM_VALUE_LEN) - 1)
#define SYM_IS_EXPR(val) (val & (1UL << SYM_VALUE_LEN))
#define SYM_VALUE(val) (val & SYM_VALUE_MASK)

#define RAW_TENSOR_ADDR_MASK ((1UL << 63) - 1)

INLINE __gm__ npu::tile_fwk::DevStartArgsBase* AiCoreRuntimeGetStartArgs(CoreFuncParam* param)
{
    return param->funcHeader->startArgs;
}

template <int mode, int64_t constval>
INLINE uint64_t GetTensorAddr(CoreFuncParam* ctx, int idx)
{
    auto func = ctx->funcData;
    auto desc = &func->rawTensorDesc[ctx->opAttrs[idx]];
    if constexpr (mode == 2) { // workspace
        return func->workspaceAddr + desc->offsetOrIndex;
    } else if constexpr (mode == 3) { // root function incast/outcast 内存
        return func->rawTensorAddr[desc->offsetOrIndex] & RAW_TENSOR_ADDR_MASK;
    } else {
        return (desc->location == npu::tile_fwk::RAW_TENSOR_LOCATION_LOCAL) ?
                   func->workspaceAddr + desc->offsetOrIndex :
                   func->rawTensorAddr[desc->offsetOrIndex] & RAW_TENSOR_ADDR_MASK;
    }
}

template <int mode, int64_t constval>
INLINE int64_t GetCoa(CoreFuncParam* ctx, int idx)
{
    if constexpr (mode == 1) { // 常量值
        return constval;
    }
    int64_t val = static_cast<int64_t>(ctx->opAttrs[idx]);
    if constexpr (mode == 2) { // 全变量
        return ctx->exprTbl[SYM_VALUE(val)];
    } else if constexpr (mode == 3) { //  全常量，但是每次取值不一样
        return SYM_VALUE(val);
    } else {
        return SYM_IS_EXPR(val) ? ctx->exprTbl[SYM_VALUE(val)] : SYM_VALUE(val);
    }
}

INLINE
int64_t RuntimeGetViewValidShapeDim(int64_t validshape, int64_t viewOffset, int64_t viewshape)
{
    validshape -= viewOffset;
    if (validshape > viewshape)
        validshape = viewshape;
    else if (validshape < 0)
        validshape = 0;
    return validshape;
}

#define RUNTIME_GetViewValidShapeDim(validShape, viewOffset, viewShape) \
    RuntimeGetViewValidShapeDim(validShape, viewOffset, viewShape)

#define RUNTIME_GET_PARAM_ADDR GET_PARAM_ADDR
#define RUNTIME_param param

#define GET_PARAM_ADDR(param, n, base) GetTensorAddr<0, 0>(param, base)

#define GET_PARAM_OFFSET_BY_IDX(param, n, base, dim, idx) GetCoa<0, 0>(param, ((base) + 1) + 0 * (dim) + idx)
#define GET_PARAM_SHAPE_BY_IDX(param, n, base, dim, idx) GetCoa<0, 0>(param, ((base) + 1) + 1 * (dim) + idx)
#define GET_PARAM_RAWSHAPE_BY_IDX(param, n, base, dim, idx) GetCoa<0, 0>(param, ((base) + 1) + 2 * (dim) + idx)
#define GET_PARAM_VALID_SHAPE_BY_IDX(param, n, base, dim, idx) GetCoa<0, 0>(param, ((base) + 1) + 3 * (dim) + idx)

#define GET_PARAM_ATTR_1(name, param, n, base) GET_PARAM_##name##_BY_IDX(param, n, base, 1, 0)
#define GET_PARAM_ATTR_2(name, param, n, base) \
    GET_PARAM_##name##_BY_IDX(param, n, base, 2, 0), GET_PARAM_##name##_BY_IDX(param, n, base, 2, 1)
#define GET_PARAM_ATTR_3(name, param, n, base)                                                        \
    GET_PARAM_##name##_BY_IDX(param, n, base, 3, 0), GET_PARAM_##name##_BY_IDX(param, n, base, 3, 1), \
        GET_PARAM_##name##_BY_IDX(param, n, base, 3, 2)
#define GET_PARAM_ATTR_4(name, param, n, base)                                                        \
    GET_PARAM_##name##_BY_IDX(param, n, base, 4, 0), GET_PARAM_##name##_BY_IDX(param, n, base, 4, 1), \
        GET_PARAM_##name##_BY_IDX(param, n, base, 4, 2), GET_PARAM_##name##_BY_IDX(param, n, base, 4, 3)
#define GET_PARAM_ATTR_5(name, param, n, base)                                                            \
    GET_PARAM_##name##_BY_IDX(param, n, base, 5, 0), GET_PARAM_##name##_BY_IDX(param, n, base, 5, 1),     \
        GET_PARAM_##name##_BY_IDX(param, n, base, 5, 2), GET_PARAM_##name##_BY_IDX(param, n, base, 5, 3), \
        GET_PARAM_##name##_BY_IDX(param, n, base, 5, 4)

#define GET_PARAM_ATTR_2_STRIDE(name, param, n, base) GET_PARAM_##name##_BY_IDX(param, n, base, 2, 1), 1
#define GET_PARAM_ATTR_3_STRIDE(name, param, n, base)                                                  \
    GET_PARAM_##name##_BY_IDX(param, n, base, 3, 1) * GET_PARAM_##name##_BY_IDX(param, n, base, 3, 2), \
        GET_PARAM_##name##_BY_IDX(param, n, base, 3, 2), 1
#define GET_PARAM_ATTR_4_STRIDE(name, param, n, base)                                                      \
    GET_PARAM_##name##_BY_IDX(param, n, base, 4, 1) * GET_PARAM_##name##_BY_IDX(param, n, base, 4, 2) *    \
        GET_PARAM_##name##_BY_IDX(param, n, base, 4, 3),                                                   \
        GET_PARAM_##name##_BY_IDX(param, n, base, 4, 2) * GET_PARAM_##name##_BY_IDX(param, n, base, 4, 3), \
        GET_PARAM_##name##_BY_IDX(param, n, base, 4, 3), 1
#define GET_PARAM_ATTR_5_STRIDE(name, param, n, base)                                                       \
    GET_PARAM_##name##_BY_IDX(param, n, base, 5, 1) * GET_PARAM_##name##_BY_IDX(param, n, base, 5, 2) *     \
        GET_PARAM_##name##_BY_IDX(param, n, base, 5, 3) * GET_PARAM_##name##_BY_IDX(param, n, base, 5, 4),  \
        GET_PARAM_##name##_BY_IDX(param, n, base, 5, 2) * GET_PARAM_##name##_BY_IDX(param, n, base, 5, 3) * \
            GET_PARAM_##name##_BY_IDX(param, n, base, 5, 4),                                                \
        GET_PARAM_##name##_BY_IDX(param, n, base, 5, 3) * GET_PARAM_##name##_BY_IDX(param, n, base, 5, 4),  \
        GET_PARAM_##name##_BY_IDX(param, n, base, 5, 4), 1

#define GET_PARAM_OFFSET_1(param, n, base) GET_PARAM_ATTR_1(OFFSET, param, n, base)
#define GET_PARAM_SHAPE_1(param, n, base) GET_PARAM_ATTR_1(SHAPE, param, n, base)
#define GET_PARAM_RAWSHAPE_1(param, n, base) GET_PARAM_ATTR_1(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_1(param, n, base) 1
#define GET_PARAM_STRIDE_DIM_1(s0) 1

#define GET_PARAM_OFFSET_2(param, n, base) GET_PARAM_ATTR_2(OFFSET, param, n, base)
#define GET_PARAM_SHAPE_2(param, n, base) GET_PARAM_ATTR_2(SHAPE, param, n, base)
#define GET_PARAM_RAWSHAPE_2(param, n, base) GET_PARAM_ATTR_2(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_2(param, n, base) GET_PARAM_ATTR_2_STRIDE(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_DIM_2(s0, s1) (s1), 1

#define GET_PARAM_OFFSET_3(param, n, base) GET_PARAM_ATTR_3(OFFSET, param, n, base)
#define GET_PARAM_SHAPE_3(param, n, base) GET_PARAM_ATTR_3(SHAPE, param, n, base)
#define GET_PARAM_RAWSHAPE_3(param, n, base) GET_PARAM_ATTR_3(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_3(param, n, base) GET_PARAM_ATTR_3_STRIDE(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_DIM_3(s0, s1, s2) (s1) * (s2), (s2), 1

#define GET_PARAM_OFFSET_4(param, n, base) GET_PARAM_ATTR_4(OFFSET, param, n, base)
#define GET_PARAM_SHAPE_4(param, n, base) GET_PARAM_ATTR_4(SHAPE, param, n, base)
#define GET_PARAM_RAWSHAPE_4(param, n, base) GET_PARAM_ATTR_4(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_4(param, n, base) GET_PARAM_ATTR_4_STRIDE(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_DIM_4(s0, s1, s2, s3) (s1) * (s2) * (s3), (s2) * (s3), (s3), 1

#define GET_PARAM_OFFSET_5(param, n, base) GET_PARAM_ATTR_5(OFFSET, param, n, base)
#define GET_PARAM_SHAPE_5(param, n, base) GET_PARAM_ATTR_5(SHAPE, param, n, base)
#define GET_PARAM_RAWSHAPE_5(param, n, base) GET_PARAM_ATTR_5(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_5(param, n, base) GET_PARAM_ATTR_5_STRIDE(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_DIM_5(s0, s1, s2, s3, s4) (s1) * (s2) * (s3) * (s4), (s2) * (s3) * (s4), (s3) * (s4), (s4), 1

#define GET_PARAM_ATTR_6(name, param, n, base)                                                            \
    GET_PARAM_##name##_BY_IDX(param, n, base, 6, 0), GET_PARAM_##name##_BY_IDX(param, n, base, 6, 1),     \
        GET_PARAM_##name##_BY_IDX(param, n, base, 6, 2), GET_PARAM_##name##_BY_IDX(param, n, base, 6, 3), \
        GET_PARAM_##name##_BY_IDX(param, n, base, 6, 4), GET_PARAM_##name##_BY_IDX(param, n, base, 6, 5)

#define GET_PARAM_ATTR_6_STRIDE(name, param, n, base)                                                          \
    GET_PARAM_##name##_BY_IDX(param, n, base, 6, 1) * GET_PARAM_##name##_BY_IDX(param, n, base, 6, 2) *        \
        GET_PARAM_##name##_BY_IDX(param, n, base, 6, 3) * GET_PARAM_##name##_BY_IDX(param, n, base, 6, 4) *    \
        GET_PARAM_##name##_BY_IDX(param, n, base, 6, 5),                                                       \
        GET_PARAM_##name##_BY_IDX(param, n, base, 6, 2) * GET_PARAM_##name##_BY_IDX(param, n, base, 6, 3) *    \
            GET_PARAM_##name##_BY_IDX(param, n, base, 6, 4) * GET_PARAM_##name##_BY_IDX(param, n, base, 6, 5), \
        GET_PARAM_##name##_BY_IDX(param, n, base, 6, 3) * GET_PARAM_##name##_BY_IDX(param, n, base, 6, 4) *    \
            GET_PARAM_##name##_BY_IDX(param, n, base, 6, 5),                                                   \
        GET_PARAM_##name##_BY_IDX(param, n, base, 6, 4) * GET_PARAM_##name##_BY_IDX(param, n, base, 6, 5),     \
        GET_PARAM_##name##_BY_IDX(param, n, base, 6, 5), 1

#define GET_PARAM_OFFSET_6(param, n, base) GET_PARAM_ATTR_6(OFFSET, param, n, base)
#define GET_PARAM_SHAPE_6(param, n, base) GET_PARAM_ATTR_6(SHAPE, param, n, base)
#define GET_PARAM_RAWSHAPE_6(param, n, base) GET_PARAM_ATTR_6(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_6(param, n, base) GET_PARAM_ATTR_6_STRIDE(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_DIM_6(s0, s1, s2, s3, s4, s5) \
    (s1) * (s2) * (s3) * (s4) * (s5), (s2) * (s3) * (s4) * (s5), (s3) * (s4) * (s5), (s4) * (s5), (s5), 1

INLINE int64_t RUNTIME_Min(int64_t input1, int64_t input2) { return input1 < input2 ? input1 : input2; }

INLINE int64_t RUNTIME_Max(int64_t input1, int64_t input2) { return input1 > input2 ? input1 : input2; }

INLINE int64_t RUNTIME_Eq(int64_t input1, int64_t input2) { return input1 == input2; }

INLINE int64_t RUNTIME_Ne(int64_t input1, int64_t input2) { return input1 != input2; }

INLINE int32_t GetTensorDataInt32(CoreFuncParam* ctx, uint64_t address)
{
    UNUSED(ctx);
    dcci((__gm__ uint32_t*)address, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    return *(__gm__ uint32_t*)(address);
}
#define RUNTIME_GetTensorDataInt32Dim1(index, ioType, ioTypeIndex, address, ...) GetTensorDataInt32(param, address)
#define RUNTIME_GetTensorDataInt32Dim2(index, ioType, ioTypeIndex, address, ...) GetTensorDataInt32(param, address)
#define RUNTIME_GetTensorDataInt32Dim3(index, ioType, ioTypeIndex, address, ...) GetTensorDataInt32(param, address)
#define RUNTIME_GetTensorDataInt32Dim4(index, ioType, ioTypeIndex, address, ...) GetTensorDataInt32(param, address)
#define RUNTIME_GetTensorDataInt32Dim5(index, ioType, ioTypeIndex, address, ...) GetTensorDataInt32(param, address)

#define RUNTIME_COA_GET_PARAM_OFFSET(dim, base, idx) GET_PARAM_OFFSET_BY_IDX(param, 0, base, dim, idx)
#define RUNTIME_COA_GET_PARAM_VALID_SHAPE(dim, base, idx) GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, base, dim, idx)
#define RUNTIME_COA_GET_PARAM_ADDR(_, idx) GET_PARAM_ADDR(param, _, idx)
#define RUNTIME_COA_GET_PARAM(idx) GetCoa<0, 0>(param, idx)
#define RUNTIME_COA_GET_PARAM_RAW_SHAPE(dim, base, idx) GetCoa<0, 0>(param, ((base) + 1) + 2 * (dim) + idx)

#define RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(mode, value, dim, base, idx) \
    GetCoa<mode, value>(param, ((base) + 1) + idx)
#define RUNTIME_COA_GET_PARAM_RAW_SHAPE_MAYBE_CONST(mode, value, dim, base, idx) \
    GetCoa<mode, value>(param, ((base) + 1) + 2 * (dim) + idx)
#define RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(mode, value, dim, base, idx) \
    GetCoa<mode, value>(param, ((base) + 1) + 3 * (dim) + idx)
#define RUNTIME_COA_GET_PARAM_MAYBE_CONST(mode, value, idx) GetCoa<mode, value>(param, idx)
#define RUNTIME_COA_GET_PARAM_ADDR_MAYBE_CONST(mode, value, tensorIdx, idx) GetTensorAddr<mode, value>(param, idx)

#define RUNTIME_TensorExtract(type, mem, dst, src) \
    do {                                           \
        pipe_barrier(PIPE_ALL);                    \
        *(mem type*)(dst) = *(mem type*)(src);     \
        pipe_barrier(PIPE_ALL);                    \
    } while (0)

#define RT_float32 float
#define RT_int64 int64_t
#define RT_int32 int32_t
#define RT_uint64 uint64_t
#define RT_uint32 uint32_t
#define RT_UB __ubuf__

#define RT_FUNCTION(name)                                                                                \
    extern "C"[aicore] void name(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t* hcclContext, \
                                 __gm__ TaskStat* taskStat)
#define RT_OPERATION(opcode, ...) RT_OPERATION_##opcode(__VA_ARGS__)
#define RT_OPERATION_MACRO(opcode, ...) RT_OPERATION_MACRO_##opcode(__VA_ARGS__)
#define RT_DECL_TYPE_TILE(name, primType, space, dim, ...) \
    using name = TileTensor<RT_##primType, LocalLayout##dim##Dim<__VA_ARGS__>, Hardware::space>;
#define RT_DECL_TYPE_TENSOR(name, primType, dim) \
    using name = TileTensor<__gm__ RT_##primType, DynLayout##dim##Dim, Hardware::GM>;
#define RT_DECL_VALUE_SCALAR(name, primType) RT_##primType name;
#define RT_DECL_VALUE_TILE(name, defType) defType name;
#define RT_DECL_VALUE_TENSOR(name, defType) defType name;

#define RT_INIT_ADDR(name, primType, space, addr, size)                             \
    RT_##primType RT_##space* name##_AS = (RT_##primType RT_##space*)get_imm(addr); \
    RT_##primType* name = (RT_##primType*)get_imm((addr));

#define RT_INIT_VALUE_TILE(name, defType, addr, dim, ...) \
    name = defType((uint64_t)(addr), Shape##dim##Dim(__VA_ARGS__));

#define RT_INIT_VALUE_TENSOR_1(shape0) DynLayout1Dim(Shape1Dim((shape0)), Stride1Dim(1))
#define RT_INIT_VALUE_TENSOR_2(shape0, shape1) DynLayout2Dim(Shape2Dim((shape0), (shape1)), Stride2Dim((shape1), 1))
#define RT_INIT_VALUE_TENSOR_3(shape0, shape1, shape2) \
    DynLayout3Dim(Shape3Dim((shape0), (shape1), (shape2)), Stride3Dim((shape1) * (shape2), (shape2), 1))
#define RT_INIT_VALUE_TENSOR_4(shape0, shape1, shape2, shape3)       \
    DynLayout4Dim(Shape4Dim((shape0), (shape1), (shape2), (shape3)), \
                  Stride4Dim((shape1) * (shape2) * (shape3), (shape2) * (shape3), (shape3), 1))
#define RT_INIT_VALUE_TENSOR_5(shape0, shape1, shape2, shape3, shape4)                                  \
    DynLayout5Dim(Shape5Dim((shape0), (shape1), (shape2), (shape3), (shape4)),                          \
                  Stride5Dim((shape1) * (shape2) * (shape3) * (shape4), (shape2) * (shape3) * (shape4), \
                             (shape3) * (shape4), (shape4), 1))
#define RT_INIT_VALUE_TENSOR(name, defType, primType, addr, dim, ...) \
    name = defType((__gm__ RT_##primType*)(addr), RT_INIT_VALUE_TENSOR_##dim(__VA_ARGS__));

#define RT_STMT_OP()
#define RT_STMT_IF()
#define RT_STMT_ELSE()
#define RT_STMT_FOR()
#define RT_STMT_YIELD()
#define RT_STMT_RETURN()
#define RT_LOOP_BEGIN()
#define RT_LOOP_ITER()
#define RT_LOOP_ASSIGN()
#define RT_IF_ASSIGN()

#define RT_OPERATION_OP_SCALAR_CALL_3(ret, arg0, arg1, arg2, callee) ret = callee(arg0, arg1, arg2);
#define RT_OPERATION_OP_SCALAR_CALL_5(ret, arg0, arg1, arg2, arg3, arg4, callee) \
    ret = callee(arg0, arg1, arg2, arg3, arg4);

#define RT_OPERATION_OP_SCALAR_CALL_3_RETVOID(ret, arg0, arg1, arg2, callee) callee(arg0, arg1, arg2);

#define RT_OPERATION_MACRO_OP_SCALAR_CALL_3(arg0, arg1, arg2, callee) callee(arg0, arg1, arg2)
#define RT_OPERATION_MACRO_OP_SCALAR_CALL_5(arg0, arg1, arg2, arg3, arg4, callee) callee(arg0, arg1, arg2, arg3, arg4)

#define RT_OPERATION_MACRO_OP_SCALAR_CALL_3_RETVOID(arg0, arg1, arg2, callee) callee(arg0, arg1, arg2)

#define RT_OPERATION_OP_SCALAR_ASSIGN(ret, arg) ret = (arg);

#define RT_OPERATION_MACRO_OP_SCALAR_ASSIGN(arg) (arg)

#define RT_OPERATION_OP_UB_COPY_IN(dst, src, off0, off1) TLoad((dst), (src), Coord2Dim((off0), (off1)));
#define RT_OPERATION_OP_ADD(dst, src0, src1, combineAxis, reverse) TAdd((dst), (src0), (src1));
#define RT_OPERATION_OP_UB_COPY_OUT(dst, src, off0, off1) TStoreVec((dst), (src), Coord2Dim((off0), (off1)));

#define RUNTIME_GetHcclRankId(groupIndex) \
    ((TileOp::CommContext*)(RuntimeGetStartArgs()->commContexts[groupIndex]))->rankId
#endif // AST_RUNTIME_H
