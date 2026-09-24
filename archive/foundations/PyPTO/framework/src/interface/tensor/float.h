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
 * \file float.h
 * \brief
 */

#pragma once
#include "interface/utils/common.h"

#include <stdio.h>
#include <stdint.h>

constexpr int SIGN_BIT_ONE = 1;
constexpr int EXP_BIT_FIVE = 5;
constexpr int FRAC_BIT_TEN = 10;
constexpr int EXP_BIT_EIGHT = 8;
constexpr int FRAC_BIT_SEVEN = 7;

namespace npu::tile_fwk {
template <typename TBase, uint32_t signBit, uint32_t expBit, uint32_t fracBit>
class Float {
    TBase value;

    enum class FloatExp {
        bitOfByte = 8,
        expZero = (1 << (expBit - 1)) - 1,

        /* fp32: 1-8-23 */
        fp32ExpBit = 8,
        fp32FracBit = 23,
        fp32ExpZero = (1 << (fp32ExpBit - 1)) - 1,
    };
    void InitFromFloat(float fv)
    {
        void* p = &fv;
        uint32_t v = *static_cast<uint32_t*>(p);
        value = BaseFromFp32(v);
    }

    float ToFloat() const
    {
        uint32_t v = BaseToFp32(value);
        void* p = &v;
        return *static_cast<float*>(p);
    }

    static bool isNaN(TBase v)
    {
        uint32_t exp = (v >> fracBit) & BitOf(expBit);
        uint32_t frac = v & BitOf(fracBit);
        return (exp == BitOf(expBit)) && (frac != 0);
    }

    static_assert(sizeof(TBase) * static_cast<uint32_t>(FloatExp::bitOfByte) >= signBit + expBit + fracBit,
                  "Invalid bit for float");

    static constexpr uint32_t BitOf(uint32_t n) { return (1 << n) - 1; }

    static constexpr uint32_t BaseFromFp32DivRound(uint32_t frac, uint32_t shift)
    {
        return (frac >> shift) + ((frac >> (shift - 1)) & 0x1);
    }

    static Float FromBase(TBase val)
    {
        Float ret;
        ret.value = val;
        return ret;
    }

    static constexpr TBase BaseFromFp32(uint32_t v32) NO_UBSAN
    {
        if (expBit == EXP_BIT_EIGHT) {
            /*  Converts a float point to bfloat16, with round-nearest-to-even as rounding method.
                最接近偶数舍入法:
                u32 : 0x3ed2f1aa -> 0011 1110 1101 0010 1111 0001 1010 1010
                                    ^         ^                           ^
                u32  format         s(1)      e(8)                        frac(23)
                                    ^         ^       ^
                bf16 format         s(1)      e(8)    frac(7)
                round-nearest-to-even :                 1111 0001 1010 1010
                Hex half-ULP : 0x8000                   0xf1aa > 0x8000    Round up (add 1 to the upper 7 bits)
                u16 : 0x3ed3     -> 0011 1110 1101 0011 = 0011 1110 1101 0010 + 1

                if 0xf1aa = X
                X > Hex half-ULP : Round up (add 1 to the upper 7 bits)
                X < Hex half-ULP : Round down (keep the original high 7 bits)
                X = Hex half-ULP : Round to the nearest even number
            */
            constexpr uint32_t floatToBf16FracShift = (static_cast<uint32_t>(FloatExp::fp32FracBit) - fracBit); // 16
            uint32_t fracLastBit = (v32 >> floatToBf16FracShift) & 0x1;
            uint32_t hexHalfUlp = (0x1 << (static_cast<uint32_t>(FloatExp::fp32FracBit) - fracBit - 1)) - 0x1; // 0x7fff
            uint32_t roundingBias = hexHalfUlp + fracLastBit;
            v32 += roundingBias;
            return (v32 >> floatToBf16FracShift);
        }
        uint32_t sign = (v32 >>
                         (static_cast<uint32_t>(FloatExp::fp32ExpBit) + static_cast<uint32_t>(FloatExp::fp32FracBit))) &
                        0x1;
        uint32_t exp32 = (v32 >> static_cast<uint32_t>(FloatExp::fp32FracBit)) &
                         BitOf(static_cast<uint32_t>(FloatExp::fp32ExpBit));
        uint32_t frac32 = v32 & BitOf(static_cast<uint32_t>(FloatExp::fp32FracBit));

        uint32_t exp = 0;
        uint32_t frac = 0;
        if (exp32 <
            static_cast<uint32_t>(FloatExp::fp32ExpZero) + -static_cast<uint32_t>(FloatExp::expZero) + 1 + -fracBit) {
            /*  Smallest number:
             *      format: 0bS X...X 0...0 == 0bS 0...0 0...01
             *                  (fp32)            (fp)
             *      compute: 2 ^ (X - fp32ExpZero) = 2 ^ (-expZero + 1) * 2 ^ (-fracBit)
             *          ==>: X = fp32ExpZero + -expZero + 1 + -fracBit
             */
            exp = 0;
            frac = 0;
        } else if (exp32 <
                   static_cast<uint32_t>(FloatExp::fp32ExpZero) + 1 - static_cast<uint32_t>(FloatExp::expZero)) {
            /*  Subnormal number:
             *      format: 0bS X...X 0...0 == 0bS 0...1 0...00
             *                  (fp32)            (fp)
             *      compute: 2 ^ (X - fp32ExpZero) == 2 ^ (1 - expZero)
             *          ==>: X = fp32ExpZero + 1 - expZero
             *
             *      value: 0bS X...X Y...Y = 0bS 0...0 Z...Z
             *      compute: (2 ^ (fp32FracBit) + Y) * 2 ^ (-fp32FracBit) * 2 ^ (X - fp32ExpZero) = Z * 2 ^ (-expZero +
             * 1) * 2 ^ (-fracBit)
             *          ==>: Z = (2 ^ (fp32FracBit) + Y) * 2 ^ (-fp32FracBit) * 2 ^ (X - fp32ExpZero) * 2 ^ (expZero -
             * 1) * 2 ^(fracBit) = (2 ^ (fp32FracBit) + Y) * 2 ^ (-fp32FracBit + X - fp32ExpZero + expZero - 1 +
             * fracBit) = (2 ^ (fp32FracBit) + Y) * 2 ^ -(fp32FracBit - X + fp32ExpZero - expZero + 1 - fracBit)
             */
            exp = 0;
            auto shift = static_cast<uint32_t>(FloatExp::fp32FracBit) - exp32 +
                         static_cast<uint32_t>(FloatExp::fp32ExpZero) - static_cast<uint32_t>(FloatExp::expZero) + 1 -
                         fracBit;
            frac = BaseFromFp32DivRound((1 << static_cast<uint32_t>(FloatExp::fp32FracBit)) | frac32, shift);
        } else if (exp32 < static_cast<uint32_t>(FloatExp::fp32ExpZero) + BitOf(expBit) -
                               static_cast<uint32_t>(FloatExp::expZero)) {
            /*  Normal number:
             *      format: 0bS X...X 0...0 == 0bS 1...1 0...00
             *                  (fp32)             (fp)
             *      compute: 2 ^ (X - fp32ExpZero) == 2 ^ (BitOf(expBit) - expZero)
             *          ==>: X = fp32ExpZero + BitOf(expBit) - expZero
             */
            exp = exp32 - static_cast<uint32_t>(FloatExp::fp32ExpZero) + static_cast<uint32_t>(FloatExp::expZero);
            frac = BaseFromFp32DivRound(frac32, static_cast<uint32_t>(FloatExp::fp32FracBit) - fracBit);
        } else {
            if (exp32 == BitOf(static_cast<uint32_t>(FloatExp::fp32ExpBit)) && frac32 != 0) {
                /* nan */
                exp = BitOf(expBit);
                frac = frac32 >> (static_cast<uint32_t>(FloatExp::fp32FracBit) - fracBit);
            } else {
                /* inf */
                exp = BitOf(expBit);
                frac = 0;
            }
        }
        return (sign << (expBit + fracBit)) | (exp << fracBit) | frac;
    }

    static constexpr uint32_t BaseToFp32(TBase v)
    {
        if (expBit == EXP_BIT_EIGHT) {
            // 常规值：BF16 高16位直接复制为 FP32 的高16位
            return static_cast<uint32_t>(v) << (static_cast<uint32_t>(FloatExp::fp32FracBit) - fracBit);
        }
        uint32_t sign = (v >> (expBit + fracBit)) & 0x1;
        uint32_t exp = (v >> fracBit) & BitOf(expBit);
        uint32_t frac = v & BitOf(fracBit);

        uint32_t exp32 = 0;
        uint32_t frac32 = 0;
        if (exp == 0) {
            if (frac == 0) {
                exp32 = 0;
                frac32 = 0;
            } else {
                /*  Subnormal:
                 *      format: 0bS 0000 0...01XY...Y
                 *                            |<-leadingBit
                 *      result: 0bS EEEE XY...Y0...0
                 *                       |<- fp32FracBit-1
                 *      compute:
                 *          shiftExp = (fracBit - 1) - (leadingBit - 1)
                 *          shiftFrac = (fp32FracBit - 1) - (leadingBit - 1)
                 *          EEEE = fp32ExpZero + -(expZero - 1) - shift
                 */
                uint32_t leadingBit = sizeof(unsigned int) * static_cast<uint32_t>(FloatExp::bitOfByte) -
                                      __builtin_clz(frac) - 1;
                uint32_t rest = frac ^ (1 << leadingBit);
                uint32_t shiftExp = (fracBit - 1) - (leadingBit - 1);
                uint32_t shiftFrac = (static_cast<uint32_t>(FloatExp::fp32FracBit) - 1) - (leadingBit - 1);
                exp32 = static_cast<uint32_t>(FloatExp::fp32ExpZero) + -(static_cast<uint32_t>(FloatExp::expZero) - 1) -
                        shiftExp;
                frac32 = rest << shiftFrac;
            }
        } else if (exp == BitOf(expBit)) {
            /*  Not a number:
             *      format: 0bS 1...1 X...X
             *      result: 0bS 1...1 X...X
             */
            exp32 = BitOf(static_cast<uint32_t>(FloatExp::fp32ExpBit));
            if (frac == 0) {
                /* inf */
                frac32 = 0;
            } else {
                /* nan */
                frac32 = frac << (static_cast<uint32_t>(FloatExp::fp32FracBit) - fracBit);
            }
        } else {
            /*  Normal:
             *      format: 0bS Y...Y Z...Z
             *                        |<-fracBit
             *      result: 0bS E...E Z...Z0...0
             *                        |<-fp32FracBit
             */
            exp32 = exp - static_cast<uint32_t>(FloatExp::expZero) + static_cast<uint32_t>(FloatExp::fp32ExpZero);
            frac32 = frac << (static_cast<uint32_t>(FloatExp::fp32FracBit) - fracBit);
        }
        return (sign << (static_cast<uint32_t>(FloatExp::fp32ExpBit) + static_cast<uint32_t>(FloatExp::fp32FracBit))) |
               (exp32 << static_cast<uint32_t>(FloatExp::fp32FracBit)) | frac32;
    }

    static void PrintMetadata()
    {
        printf("expBit=%d fracBit=%d expZero=%d fp32ExpBit=%d fp32FracBit=%d fp32ExpZero=%d\n", expBit, fracBit,
               static_cast<uint32_t>(FloatExp::expZero), static_cast<uint32_t>(FloatExp::fp32ExpBit),
               static_cast<uint32_t>(FloatExp::fp32FracBit), static_cast<uint32_t>(FloatExp::fp32ExpZero));
    }

public:
    Float() : value(0) {}

    template <typename T>
    Float(T fv)
    {
        InitFromFloat(static_cast<float>(fv));
    }

    operator float() const { return ToFloat(); }

    template <typename T>
    Float operator+(T fv)
    {
        return Float(this->ToFloat() + static_cast<float>(fv));
    }

    template <typename T>
    Float operator-(T fv)
    {
        return Float(this->ToFloat() - static_cast<float>(fv));
    }

    template <typename T>
    Float operator*(T fv)
    {
        return Float(this->ToFloat() * static_cast<float>(fv));
    }

    template <typename T>
    Float operator/(T fv)
    {
        return Float(this->ToFloat() / static_cast<float>(fv));
    }

    template <typename T>
    Float& operator+=(T fv)
    {
        InitFromFloat(this->ToFloat() + static_cast<float>(fv));
        return *this;
    }

    template <typename T>
    Float& operator-=(T fv)
    {
        InitFromFloat(this->ToFloat() - static_cast<float>(fv));
        return *this;
    }

    template <typename T>
    Float& operator*=(T fv)
    {
        InitFromFloat(this->ToFloat() * static_cast<float>(fv));
        return *this;
    }

    template <typename T>
    Float& operator/=(T fv)
    {
        InitFromFloat(this->ToFloat() / static_cast<float>(fv));
        return *this;
    }

    template <typename T>
    bool operator==(T fv) const
    {
        TBase thisBase = value;
        TBase otherBase;
        if constexpr (std::is_same_v<T, Float>) {
            otherBase = fv.value;
        } else {
            float temp = static_cast<float>(fv);
            otherBase = BaseFromFp32(*reinterpret_cast<const uint32_t*>(&temp));
        }
        if ((thisBase & ~(1 << (expBit + fracBit))) == 0 && (otherBase & ~(1 << (expBit + fracBit))) == 0) {
            return true;
        }
        if (isNaN(thisBase) || isNaN(otherBase)) {
            return false;
        }
        return thisBase == otherBase;
    }

    template <typename T>
    bool operator!=(T fv) const
    {
        return !(*this == fv);
    }

    template <typename T>
    bool operator>=(T fv) const
    {
        return this->ToFloat() >= static_cast<float>(fv);
    }

    template <typename T>
    bool operator<=(T fv) const
    {
        return this->ToFloat() <= static_cast<float>(fv);
    }

    template <typename T>
    bool operator>(T fv) const
    {
        return this->ToFloat() > static_cast<float>(fv);
    }

    template <typename T>
    bool operator<(T fv) const
    {
        return this->ToFloat() < static_cast<float>(fv);
    }
};

using bfloat16 = Float<uint16_t, SIGN_BIT_ONE, EXP_BIT_EIGHT, FRAC_BIT_SEVEN>;
using float16 = Float<uint16_t, SIGN_BIT_ONE, EXP_BIT_FIVE, FRAC_BIT_TEN>;

} // namespace npu::tile_fwk
