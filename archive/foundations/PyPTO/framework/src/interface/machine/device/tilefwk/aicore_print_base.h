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
 * \file aicore_print_base.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <cstring>
#include "aikernel_data.h"

#define ENABLE_AICORE_PRINT 0

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

// A5: FP8/HF8 supported, L1 not supported; A2/A3: opposite
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define SUPPORT_FP8_HF8_PRINT 1
#define SUPPORT_L1_COPY 0
#else
#define SUPPORT_FP8_HF8_PRINT 0
#define SUPPORT_L1_COPY 1
#endif

namespace AicorePrintConst {
constexpr size_t INDEXED_INDEX_SIZE = 8;
constexpr size_t TENSOR_RANGE_SIZE = 8;
constexpr size_t NAMELEN_FIELD_SIZE = 2;
constexpr size_t MAX_SHAPE_DIMS = 6;
constexpr size_t REMOTE_HEADER_SIZE = 16;
constexpr size_t WARNING_RESERVE_SPACE = 10;
constexpr size_t MIN_BUFFER_TOTAL_SIZE = WARNING_RESERVE_SPACE + REMOTE_HEADER_SIZE;
constexpr size_t SHORT_MAX_VALUE = 32767;
} // namespace AicorePrintConst

namespace Fp32Const {
constexpr uint32_t EXP_BIAS = 127;
constexpr uint32_t EXP_INF_NAN = 0xFFu;
constexpr uint32_t SIGN_SHIFT = 31;
constexpr uint32_t EXP_SHIFT = 23;
constexpr uint32_t MANT_BITS = 23;
} // namespace Fp32Const

namespace Fp16Const {
constexpr uint16_t SIGN_MASK = 0x8000u;
constexpr uint16_t EXP_MASK = 0x7C00u;
constexpr uint16_t MANT_MASK = 0x03FFu;
constexpr uint16_t NORM_HIDDEN = 0x0400u;
constexpr uint16_t EXP_INF_NAN = 0x1Fu;
constexpr uint16_t SIGN_SHIFT = 15;
constexpr uint16_t EXP_SHIFT = 10;
constexpr uint16_t EXP_BIAS = 15;
constexpr uint32_t MANT_TO_FP32_SHIFT = 13;
constexpr uint32_t SUBNORMAL_EXP_BASE = Fp32Const::EXP_BIAS - (EXP_BIAS - 1);
} // namespace Fp16Const

namespace Bf16Const {
constexpr uint32_t TO_FP32_SHIFT = 16;
}

namespace Fp8Const {
constexpr uint8_t SIGN_SHIFT = 7;
constexpr uint8_t BIT_MASK_1 = 0x1;
} // namespace Fp8Const

namespace Hf8Const {
constexpr uint8_t SIGN_SHIFT = 7;
constexpr uint8_t LOWER_BITS_MASK = 0x7F;
constexpr uint8_t MV_MASK_3BIT = 0x7;
constexpr uint8_t MV_MASK_2BIT = 0x3;
constexpr uint8_t MV_MASK_1BIT = 0x1;
constexpr uint8_t TOP4_SHIFT = 3;
constexpr uint8_t TOP3_SHIFT = 4;
constexpr uint8_t TOP2_SHIFT = 5;
constexpr uint8_t EB_SHIFT_MEDIUM = 3;
constexpr uint8_t EB_SHIFT_HUGE = 2; // 用于 DecodeHf8Huge 提取 eb
constexpr uint8_t EB_SHIFT_MAX = 1;  // 用于 DecodeHf8Max 提取 eb
constexpr uint8_t EB_MASK_WIDTH_1BIT = 1;
constexpr uint8_t MANT_BITS_3 = 3;
constexpr uint8_t MANT_BITS_2 = 2;
constexpr uint8_t MANT_BITS_1 = 1;
constexpr int MV_TO_FP32_EXP_BASE = 23;
constexpr int EB_MASK_WIDTH_LARGE = 2;
constexpr int EB_MASK_WIDTH_HUGE = 3;
constexpr int EB_MASK_WIDTH_MAX = 4;
constexpr int EV_BASE_LARGE = 2;
constexpr int EV_BASE_HUGE = 4;
constexpr int EV_BASE_MAX = 8;
constexpr int RANGE_TINY = 0;
constexpr int RANGE_SMALL = 1;
constexpr int RANGE_MEDIUM = 1;
constexpr int RANGE_LARGE = 1;
constexpr int RANGE_HUGE = 2;
constexpr int EB_ZERO = 0;
constexpr int EV_POSITIVE = 1;
constexpr int EV_NEGATIVE = -1;
constexpr int MV_ZERO = 0;
} // namespace Hf8Const

namespace Fp8E4M3Const {
constexpr uint8_t EXP_BITS = 4;
constexpr uint8_t MANT_BITS = 3;
constexpr uint8_t EXP_BIAS = 7;
constexpr uint8_t EXP_MASK = 0xF;
constexpr uint8_t EXP_MAX = 15;
} // namespace Fp8E4M3Const

namespace Fp8E5M2Const {
constexpr uint8_t EXP_BITS = 5;
constexpr uint8_t MANT_BITS = 2;
constexpr uint8_t EXP_BIAS = 15;
} // namespace Fp8E5M2Const

#ifdef __TILE_FWK_AICORE__
#include "tileop/utils/layout.h"
#endif

namespace AicorePrint {
enum class DataType : uint8_t {
    End = 0,
    Normal = 1,
    Fp32 = 2,
    Int64 = 3,
    Char = 4,
    String = 5,
    Pointer = 6,
    Bf16 = 7,
    Fp16 = 8,
    TensorHeader = 9,
    IndexedFp32 = 10,
    IndexedInt64 = 11,
    IndexedBf16 = 12,
    IndexedFp16 = 13,
    OverflowWarning = 14,
    Fp8E4M3 = 15,
    Fp8E5M2 = 16,
    Fp8E8M0 = 17,
    Hf8 = 18,
    IndexedFp8E4M3 = 19,
    IndexedFp8E5M2 = 20,
    IndexedFp8E8M0 = 21,
    IndexedHf8 = 22,
    Newline = 23,
    LogLevelMarker = 24,
    TimestampMarker = 25,
};
}

enum class AicoreLogLevel : uint8_t {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    NONE = 0xFF,
};

struct LogContext {
    void (*PrintInt64)(LogContext* ctx, __gm__ const char** fmt, int64_t val);
    void (*PrintFp32)(LogContext* ctx, __gm__ const char** fmt, float val);
    void (*PrintBf16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
    void (*PrintFp16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
    void (*PrintRaw)(LogContext* ctx, __gm__ const char* fmt);
    void (*PrintFp8E4M3)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8E5M2)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8E8M0)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintHf8)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintNewLine)(LogContext* ctx);
    uint8_t logLevel{static_cast<uint8_t>(AicoreLogLevel::NONE)};
};

template <typename ElemT>
struct IndexedTypeInfo {
    static constexpr AicorePrint::DataType Type = std::is_same_v<ElemT, float>   ? AicorePrint::DataType::IndexedFp32 :
                                                  std::is_same_v<ElemT, int64_t> ? AicorePrint::DataType::IndexedInt64 :
#if IS_AICORE
                                                  std::is_same_v<ElemT, bfloat16_t> ?
                                                                                   AicorePrint::DataType::IndexedBf16 :
                                                  std::is_same_v<ElemT, half> ? AicorePrint::DataType::IndexedFp16 :
#if SUPPORT_FP8_HF8_PRINT
                                                  std::is_same_v<ElemT, float8_e4m3_t> ?
                                                                                AicorePrint::DataType::IndexedFp8E4M3 :
                                                  std::is_same_v<ElemT, float8_e5m2_t> ?
                                                                                AicorePrint::DataType::IndexedFp8E5M2 :
                                                  std::is_same_v<ElemT, float8_e8m0_t> ?
                                                                                AicorePrint::DataType::IndexedFp8E8M0 :
                                                  std::is_same_v<ElemT, hifloat8_t> ?
                                                                                AicorePrint::DataType::IndexedHf8 :
#endif
#endif
                                                                                AicorePrint::DataType::End;
};

template <typename T, typename U>
INLINE void SafeBitCast(T& dst, const U& src)
{
    static_assert(sizeof(T) >= sizeof(U), "Target type too small");
    const unsigned char* srcBytes = reinterpret_cast<const unsigned char*>(&src);
    unsigned char* dstBytes = reinterpret_cast<unsigned char*>(&dst);

    for (std::size_t i = 0; i < sizeof(U); ++i) {
        dstBytes[i] = srcBytes[i];
    }

    for (std::size_t i = sizeof(U); i < sizeof(T); ++i) {
        dstBytes[i] = 0;
    }
}

template <typename T, typename U>
INLINE T SafeBitCast(const U& src)
{
    T dst{};
    SafeBitCast(dst, src);
    return dst;
}

INLINE float DecodeBf16(uint16_t bits)
{
    uint32_t u = static_cast<uint32_t>(bits) << Bf16Const::TO_FP32_SHIFT;
    return SafeBitCast<float>(u);
}

INLINE float DecodeF16(uint16_t bits)
{
    const uint16_t sign = (bits & Fp16Const::SIGN_MASK) >> Fp16Const::SIGN_SHIFT;
    const uint16_t exp = (bits & Fp16Const::EXP_MASK) >> Fp16Const::EXP_SHIFT;
    const uint16_t mant = bits & Fp16Const::MANT_MASK;

    uint32_t sign32 = static_cast<uint32_t>(sign) << Fp32Const::SIGN_SHIFT;
    uint32_t exp32 = 0;
    uint32_t mant32 = 0;

    if (exp == 0) {
        if (mant == 0) {
            exp32 = 0;
            mant32 = 0;
        } else {
            exp32 = Fp16Const::SUBNORMAL_EXP_BASE;
            uint16_t normMant = mant;

            while ((normMant & Fp16Const::NORM_HIDDEN) == 0) {
                normMant <<= 1;
                --exp32;
            }

            normMant &= Fp16Const::MANT_MASK;
            mant32 = static_cast<uint32_t>(normMant) << Fp16Const::MANT_TO_FP32_SHIFT;
        }
    } else if (exp == Fp16Const::EXP_INF_NAN) {
        exp32 = Fp32Const::EXP_INF_NAN;
        mant32 = static_cast<uint32_t>(mant) << Fp16Const::MANT_TO_FP32_SHIFT;
    } else {
        exp32 = static_cast<uint32_t>(exp) - Fp16Const::EXP_BIAS + Fp32Const::EXP_BIAS;
        mant32 = static_cast<uint32_t>(mant) << Fp16Const::MANT_TO_FP32_SHIFT;
    }

    uint32_t result = sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32;
    return SafeBitCast<float>(result);
}

template <uint8_t ExpBits, uint8_t MantBits, uint8_t ExpBias, bool HasInf>
INLINE float DecodeFp8Common(uint8_t bits)
{
    constexpr uint8_t SignShift = Fp8Const::SIGN_SHIFT;
    constexpr uint8_t ExpShift = MantBits;
    constexpr uint8_t ExpMax = (1u << ExpBits) - 1;
    constexpr uint8_t MantMask = (1u << MantBits) - 1;

    const uint8_t sign = (bits >> SignShift) & Fp8Const::BIT_MASK_1;
    const uint8_t exp = (bits >> ExpShift) & ((1u << ExpBits) - 1);
    const uint8_t mant = bits & MantMask;

    uint32_t sign32 = static_cast<uint32_t>(sign) << Fp32Const::SIGN_SHIFT;
    uint32_t exp32 = 0;
    uint32_t mant32 = 0;

    if (exp == 0) {
        if (mant == 0) {
            exp32 = 0;
            mant32 = 0;
        } else {
            exp32 = Fp32Const::EXP_BIAS - (ExpBias - 1);
            uint32_t normMant = static_cast<uint32_t>(mant);
            constexpr uint32_t hiddenBit = (1u << MantBits);

            while ((normMant & hiddenBit) == 0) {
                normMant <<= 1;
                --exp32;
            }

            normMant &= ((1u << MantBits) - 1);
            mant32 = normMant << (Fp32Const::MANT_BITS - MantBits);
        }
    } else if (HasInf && exp == ExpMax) {
        exp32 = Fp32Const::EXP_INF_NAN;
        mant32 = static_cast<uint32_t>(mant) << (Fp32Const::MANT_BITS - MantBits);
    } else {
        exp32 = static_cast<uint32_t>(exp) - ExpBias + Fp32Const::EXP_BIAS;
        mant32 = static_cast<uint32_t>(mant) << (Fp32Const::MANT_BITS - MantBits);
    }

    uint32_t result = sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32;
    return SafeBitCast<float>(result);
}

// float8_e4m3fn (OCP MX / PyTorch): no infinity.
// Only exp=0b1111 with mant=0b111 (0x7F / 0xFF) is NaN.
// Max finite: exp=0b1111, mant=0b110 (0x7E) → 448.0
INLINE float DecodeFp8E4M3(uint8_t bits)
{
    constexpr uint8_t MANT_MASK = (1u << Fp8E4M3Const::MANT_BITS) - 1;
    constexpr uint8_t NAN_MANT = MANT_MASK; // mant=0b111
    const uint8_t exp = (bits >> Fp8E4M3Const::MANT_BITS) & Fp8E4M3Const::EXP_MASK;
    const uint8_t mant = bits & MANT_MASK;
    if (exp == Fp8E4M3Const::EXP_MAX && mant == NAN_MANT) {
        const uint8_t sign = (bits >> Fp8Const::SIGN_SHIFT) & Fp8Const::BIT_MASK_1;
        constexpr uint32_t f32QNaN = (Fp32Const::EXP_INF_NAN << Fp32Const::EXP_SHIFT) | 0x400000u;
        uint32_t result = (static_cast<uint32_t>(sign) << Fp32Const::SIGN_SHIFT) | f32QNaN;
        return SafeBitCast<float>(result);
    }
    return DecodeFp8Common<Fp8E4M3Const::EXP_BITS, Fp8E4M3Const::MANT_BITS, Fp8E4M3Const::EXP_BIAS, false>(bits);
}

INLINE float DecodeFp8E5M2(uint8_t bits)
{
    return DecodeFp8Common<Fp8E5M2Const::EXP_BITS, Fp8E5M2Const::MANT_BITS, Fp8E5M2Const::EXP_BIAS, true>(bits);
}

// E8M0: 8-bit unsigned exponent, no sign, no mantissa, bias=127.
// value = 2^(bits - 127); exp=0 → 2^-127; exp=255 → NaN.
INLINE float DecodeFp8E8M0(uint8_t bits)
{
    if (bits == 0) {
        // exp=0 → 2^-127 (float32 subnormal)
        return SafeBitCast<float>(UINT32_C(0x00400000));
    }
    if (bits == UINT8_C(0xFF)) {
        // exp=255 → NaN (E8M0 has infinity=false)
        constexpr uint32_t f32QNaN = (Fp32Const::EXP_INF_NAN << Fp32Const::EXP_SHIFT) | 0x400000u;
        return SafeBitCast<float>(f32QNaN);
    }
    uint32_t result = static_cast<uint32_t>(bits) << Fp32Const::EXP_SHIFT;
    return SafeBitCast<float>(result);
}

// HiFloat8 Tiny (DML prefix 0000): mv=0 with sign=0 → ±0; sign=1 → NaN
INLINE float DecodeHf8Tiny(int signBit, int mv)
{
    if (mv == Hf8Const::MV_ZERO) {
        if (signBit) {
            // NaN: 1_0000_000 (HiF8 has no negative zero)
            constexpr uint32_t f32QNaN = (Fp32Const::EXP_INF_NAN << Fp32Const::EXP_SHIFT) | 0x400000u;
            return SafeBitCast<float>(f32QNaN);
        }
        return SafeBitCast<float>(0u);
    }

    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const int fp32Exp = mv - Hf8Const::MV_TO_FP32_EXP_BASE + Fp32Const::EXP_BIAS;
    const uint32_t result = sign32 | (static_cast<uint32_t>(fp32Exp) << Fp32Const::EXP_SHIFT);
    return SafeBitCast<float>(result);
}

INLINE float DecodeHf8Small(int signBit, int mv)
{
    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const uint32_t exp32 = Fp32Const::EXP_BIAS;
    const uint32_t mant32 = static_cast<uint32_t>(mv) << (Fp32Const::MANT_BITS - Hf8Const::MANT_BITS_3);
    return SafeBitCast<float>(sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32);
}

INLINE float DecodeHf8WithEvMv(int signBit, int ev, int mv, int mantBits)
{
    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const uint32_t exp32 = static_cast<uint32_t>(ev + Fp32Const::EXP_BIAS);
    const uint32_t mant32 = static_cast<uint32_t>(mv) << (Fp32Const::MANT_BITS - mantBits);
    return SafeBitCast<float>(sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32);
}

INLINE float DecodeHf8Medium(int signBit, int lower7)
{
    const int eb = (lower7 >> Hf8Const::EB_SHIFT_MEDIUM) & Hf8Const::EB_MASK_WIDTH_1BIT;
    const int ev = (eb == Hf8Const::EB_ZERO) ? Hf8Const::EV_POSITIVE : Hf8Const::EV_NEGATIVE;
    const int mv = lower7 & Hf8Const::MV_MASK_3BIT;
    return DecodeHf8WithEvMv(signBit, ev, mv, Hf8Const::MANT_BITS_3);
}

template <int EbShift, int EbMaskWidth, int EvBase, int MvMask, int MantBits>
INLINE float DecodeHf8EvMvPattern(int signBit, int lower7)
{
    const int ebMask = (1 << EbMaskWidth) - 1;
    const int eb = (lower7 >> EbShift) & ebMask;
    const int evSign = (eb >> (EbMaskWidth - 1)) & Hf8Const::EB_MASK_WIDTH_1BIT;
    const int addMask = (1 << (EbMaskWidth - 1)) - 1;
    const int evAbs = EvBase + (eb & addMask);
    const int ev = evSign ? -evAbs : evAbs;
    const int mv = lower7 & MvMask;
    return DecodeHf8WithEvMv(signBit, ev, mv, MantBits);
}

INLINE float DecodeHf8Large(int signBit, int lower7)
{
    return DecodeHf8EvMvPattern<Hf8Const::TOP4_SHIFT, Hf8Const::EB_MASK_WIDTH_LARGE, Hf8Const::EV_BASE_LARGE,
                                Hf8Const::MV_MASK_3BIT, Hf8Const::MANT_BITS_3>(signBit, lower7);
}

INLINE float DecodeHf8Huge(int signBit, int lower7)
{
    return DecodeHf8EvMvPattern<Hf8Const::EB_SHIFT_HUGE, Hf8Const::EB_MASK_WIDTH_HUGE, Hf8Const::EV_BASE_HUGE,
                                Hf8Const::MV_MASK_2BIT, Hf8Const::MANT_BITS_2>(signBit, lower7);
}

// HiFloat8 Max (D=4, prefix 11): ev=±[8,15], 1-bit mantissa.
// Infinity encoding: S_11_0111_1 (ev=15, mant=1).
INLINE float DecodeHf8Max(int signBit, int lower7)
{
    constexpr int EbMaskWidth = Hf8Const::EB_MASK_WIDTH_MAX;
    const int ebMask = (1 << EbMaskWidth) - 1;
    const int eb = (lower7 >> Hf8Const::EB_SHIFT_MAX) & ebMask;
    const int evSign = (eb >> (EbMaskWidth - 1)) & Hf8Const::EB_MASK_WIDTH_1BIT;
    const int addMask = (1 << (EbMaskWidth - 1)) - 1;
    const int evAbs = Hf8Const::EV_BASE_MAX + (eb & addMask);
    const int ev = evSign ? -evAbs : evAbs;
    const int mv = lower7 & Hf8Const::MV_MASK_1BIT;

    // Infinity: S_11_0111_1
    if (ev == 15 && mv == Hf8Const::MV_MASK_1BIT) {
        constexpr uint32_t f32InfBits = Fp32Const::EXP_INF_NAN << Fp32Const::EXP_SHIFT;
        uint32_t result = (static_cast<uint32_t>(signBit) << Fp32Const::SIGN_SHIFT) | f32InfBits;
        return SafeBitCast<float>(result);
    }

    return DecodeHf8WithEvMv(signBit, ev, mv, Hf8Const::MANT_BITS_1);
}

INLINE float DecodeHf8(uint8_t bits)
{
    const int signBit = (bits >> Hf8Const::SIGN_SHIFT) & Hf8Const::EB_MASK_WIDTH_1BIT;
    const int lower7 = bits & Hf8Const::LOWER_BITS_MASK;
    const int top4 = lower7 >> Hf8Const::TOP4_SHIFT;

    if (top4 == Hf8Const::RANGE_TINY) {
        return DecodeHf8Tiny(signBit, lower7 & Hf8Const::MV_MASK_3BIT);
    }

    if (top4 == Hf8Const::RANGE_SMALL) {
        return DecodeHf8Small(signBit, lower7 & Hf8Const::MV_MASK_3BIT);
    }

    const int top3 = lower7 >> Hf8Const::TOP3_SHIFT;

    if (top3 == Hf8Const::RANGE_MEDIUM) {
        return DecodeHf8Medium(signBit, lower7);
    }

    const int top2 = lower7 >> Hf8Const::TOP2_SHIFT;

    if (top2 == Hf8Const::RANGE_LARGE) {
        return DecodeHf8Large(signBit, lower7);
    }

    if (top2 == Hf8Const::RANGE_HUGE) {
        return DecodeHf8Huge(signBit, lower7);
    }

    return DecodeHf8Max(signBit, lower7);
}

// Host-Side Decode Helpers
