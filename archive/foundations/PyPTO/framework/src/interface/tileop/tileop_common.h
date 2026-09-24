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
 * \file tileop_common.h
 * \brief
 */

#ifndef __LOGICALTENSOR_TILEOP_COMMON__
#define __LOGICALTENSOR_TILEOP_COMMON__

#ifdef SUPPORT_TILE_TENSOR
#include "pto/pto-inst.hpp"
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#ifndef __aicore_host__
#define __aicore_host__ [ host, aicore ]
#endif

#ifndef TILEOP
#define TILEOP static __attribute__((always_inline))[aicore]
#endif

#ifndef INLINE
#define INLINE __attribute__((always_inline)) inline[aicore]
#endif

#ifndef CORELOG
#define CORELOG(x...)
#endif

#define SUBKERNEL_PHASE1
#define SUBKERNEL_PHASE2

#if defined(__DAV_C220_VEC__)
#define WAIT_TASK_FIN                       \
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7); \
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7)
#else
#define WAIT_TASK_FIN                      \
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7); \
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7)
#endif

#if defined(__DAV_C220_VEC__)
#define WAIT_PRE_TASK                          \
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7); \
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7)
#else
#define WAIT_PRE_TASK                          \
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7); \
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7)
#endif

enum class CopyInMode : int64_t { ND2ND = 0, ND2NZ = 1, NZ2NZ = 2, DN2NZ = 3 };

enum class CopyOutMode : int64_t { NZ2ND = 0, NZ2NZ = 1, ND2ND = 2, NZ2DN = 3 };

enum class CopyMode : int64_t { UNKNOWN = 0, EXTRACT = 1, INSERT = 2, MOVE = 3 };

enum class DualDstMode : int64_t { DUAL_DST_DISABLE = 0, DUAL_DST_SPLIT_M = 1, DUAL_DST_SPLIT_N = 2 };

enum class TransMode : int64_t { CAST_NONE = 0, CAST_RINT = 1, CAST_ROUND = 2 };

enum class PaddingMode : int64_t { NORMAL_PADDING_MODE = 0, MX_PADDING_MODE = 1 };

enum class ReLuType : int64_t { NoReLu = 0, ReLu = 1 };

namespace TileOp {
enum CastMode {
    CAST_NONE = 0,
    CAST_RINT = 1,  // round to nearest, tie to even
    CAST_ROUND = 2, // round to nearest, tie away from zero
    CAST_FLOOR = 3, // round to minus infinity
    CAST_CEIL = 4,  // round to positive infinity
    CAST_TRUNC = 5, // round to zero
    CAST_ODD = 6,   // round to odd (Von Neumann rounding)
};

enum class BroadcastOperand : int64_t {
    NONE = 0,
    LEFT_OPERAND = 1,
    RIGHT_OPERAND = 2,
};

constexpr uint64_t MASK_LEN = 64;
constexpr uint64_t BLOCK_NELEM_B16 = 16;
constexpr uint64_t BLOCK_NELEM_B32 = 8;
constexpr uint64_t NBLOCK_PER_MASK_B16 = 4;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t REPEAT_MAX = 255;
constexpr uint64_t REPEAT_BYTE = 256;
constexpr uint64_t REPEAT_STRIDE_MAX = 255;
constexpr uint64_t DUP_REPEAT_STRIDE_MAX = 4095;
constexpr uint64_t BLOCK_NUM_ONE_REPEAT = 8;
constexpr uint32_t BF16_FP32_MAN_LEN = 16;
constexpr float EPSILON = 1e-6f;

// fp32->bf16, rint mode
INLINE bfloat16_t Fp32ToBf16R(const float fVal)
{
    union Bfloat16Union {
        bfloat16_t bVal;
        uint16_t bNum;
    } bf16Union = {};
    union Float32Union {
        float fVal;
        uint32_t fNum;
    } fp32Union = {};
    fp32Union.fVal = fVal;
    uint32_t x = fp32Union.fNum;
    // 处理特殊值
    uint32_t exp = x & 0x7F800000;
    if (exp == 0x7F800000) { // NaN 或无穷大
        bf16Union.bNum = static_cast<uint16_t>((x >> BF16_FP32_MAN_LEN) | 0x7F80);
        return bf16Union.bVal;
    }
    if (exp == 0) { // 0或非规格化
        bf16Union.bNum = static_cast<uint16_t>((x >> BF16_FP32_MAN_LEN) & 0x8000);
        return bf16Union.bVal;
    }
    // RINT舍入
    uint32_t lsb = (x >> BF16_FP32_MAN_LEN) & 1;
    uint32_t roundingBit = (x >> (BF16_FP32_MAN_LEN - 1)) & 1;
    uint32_t sticky = x & 0x7FFF;

    uint32_t roundUp = 0;
    if (roundingBit) {
        roundUp = (sticky != 0) ? 1 : lsb;
    }

    uint32_t result = (x + (roundUp << (BF16_FP32_MAN_LEN - 1))) >> BF16_FP32_MAN_LEN;
    // 溢出检查
    if ((result & 0x7F80) == 0x7F80) {
        result = (result & 0x8000) | 0x7F80;
    }
    bf16Union.bNum = static_cast<uint16_t>(result);
    return bf16Union.bVal;
}

// bf16->fp32
INLINE float Bf16ToFp32(const bfloat16_t bVal)
{
    union Bfloat16Union {
        bfloat16_t bVal;
        uint16_t bNum;
    } bf16Union = {};
    union Float32Union {
        float fVal;
        uint32_t fNum;
    } fp32Union = {};
    bf16Union.bVal = bVal;
    fp32Union.fNum = static_cast<uint32_t>(bf16Union.bNum) << BF16_FP32_MAN_LEN;
    return fp32Union.fVal;
}

TILEOP bool IsInteger(float f)
{
    union {
        float f;
        uint32_t u;
    } converter;
    converter.f = f;
    uint32_t bits = converter.u;
    uint32_t exponent = (bits >> 23) & 0xFF;
    uint32_t fraction = bits & 0x7FFFFF;
    // NaN or Inf
    if (exponent == 0xFF) {
        return false;
    }
    int32_t e = static_cast<int32_t>(exponent) - 127;
    if (e < 0) {
        // +0 or -0
        return bits == 0 || bits == 0x80000000;
    }
    if (e >= 23) {
        return true;
    }
    uint32_t mask = (1u << (23 - e)) - 1;
    return (fraction & mask) == 0;
}

inline TILEOP void SetContinuousMask(unsigned n)
{
    set_vector_mask(
        static_cast<uint64_t>(
            (n > MASK_LEN) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(n - MASK_LEN)) - 1) : 0),
        static_cast<uint64_t>((n >= MASK_LEN) ? 0xffffffffffffffff :
                                                (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(n)) - 1)));
}

// Calculation linear offset for multi-dimension tensor
INLINE unsigned CalcLinearOffset(unsigned GmShape1, unsigned GmShape2, unsigned GmShape3, unsigned GmShape4,
                                 unsigned Offset0, unsigned Offset1, unsigned Offset2, unsigned Offset3,
                                 unsigned Offset4)
{
    return Offset4 + Offset3 * GmShape4 + Offset2 * (GmShape3 * GmShape4) + Offset1 * (GmShape2 * GmShape3 * GmShape4) +
           Offset0 * (GmShape1 * GmShape2 * GmShape3 * GmShape4);
}

// Calculation linear offset for multi-dimension tensor
INLINE unsigned CalcLinearOffset(unsigned GmShape1, unsigned GmShape2, unsigned GmShape3, unsigned Offset0,
                                 unsigned Offset1, unsigned Offset2, unsigned Offset3)
{
    return Offset3 + Offset2 * GmShape3 + Offset1 * (GmShape2 * GmShape3) + Offset0 * (GmShape1 * GmShape2 * GmShape3);
}

// Calculation linear offset for multi-dimension tensor
INLINE unsigned CalcLinearOffset(unsigned GmShape1, unsigned GmShape2, unsigned Offset0, unsigned Offset1,
                                 unsigned Offset2)
{
    return Offset2 + Offset1 * GmShape2 + Offset0 * (GmShape1 * GmShape2);
}

// Calculation linear offset for multi-dimension tensor
INLINE unsigned CalcLinearOffset(unsigned GmShape1, unsigned Offset0, unsigned Offset1)
{
    return Offset1 + Offset0 * GmShape1;
}
} // namespace TileOp

template <bool b>
struct BoolInst {
    using Type = BoolInst<b>;
    static constexpr bool value = b;
};
using TrueType = BoolInst<true>;
using FalseType = BoolInst<false>;
template <typename T, typename U>
struct IsSameType : public FalseType {};
template <typename T>
struct IsSameType<T, T> : public TrueType {};
template <typename T>
TILEOP void SetAtomicAddition()
{
    if constexpr (IsSameType<T, float>::value) {
        set_atomic_f32();
    } else if constexpr (IsSameType<T, half>::value) {
        set_atomic_f16();
    } else if constexpr (IsSameType<T, int16_t>::value) {
        set_atomic_s16();
    } else if constexpr (IsSameType<T, int32_t>::value) {
        set_atomic_s32();
    } else if constexpr (IsSameType<T, int8_t>::value) {
        set_atomic_s8();
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        set_atomic_bf16();
    }
    set_atomic_add();
}

#endif
