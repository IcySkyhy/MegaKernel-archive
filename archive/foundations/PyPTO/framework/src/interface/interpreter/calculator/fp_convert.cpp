/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fp_convert.cpp
 * \brief FP8/FP4 format conversion implementations.
 */

#include <array>
#include <cmath>
#include <limits>
#include "fp_convert.h"
#include "calc_error.h"
#include "tilefwk/error.h"

namespace npu::tile_fwk {

// OCP float8_e4m3fn encode thresholds; constexpr so divisors are provably non-zero (G.EXP.22-CPP).
static constexpr float kE4M3EncodeMinSubnormal = 1.0f / 512.0f;
static constexpr float kE4M3EncodeMinNormal = 1.0f / 64.0f;
static constexpr float kE4M3EncodeMaxFinite = 448.0f;

// Round to nearest integer, ties to even, for non-negative inputs.
static inline int RoundToNearestEvenFloatPos(float x)
{
    float floorVal = std::floor(x);
    float frac = x - floorVal;
    int base = static_cast<int>(floorVal);
    if (frac > 0.5f) {
        return base + 1;
    }
    if (frac < 0.5f) {
        return base;
    }
    return (base % 0x2 == 0) ? base : (base + 1);
}

// abs_times_512 = |value| / (1/512); OCP E4M3fn subnormal mantissa encoding (exp=0).
static uint8_t EncodeE4M3SubnormalFromAbsTimes512(float abs_times_512, int sign)
{
    if (abs_times_512 < 0.0f) {
        abs_times_512 = 0.0f;
    }
    const int mant = RoundToNearestEvenFloatPos(abs_times_512);
    if (mant <= 0) {
        return static_cast<uint8_t>(sign << 0x7);
    }
    if (mant >= 0x8) {
        return static_cast<uint8_t>((sign << 0x7) | (1 << 0x3));
    }
    return static_cast<uint8_t>((sign << 0x7) | mant);
}

// |v| in [1/64, 448); frexp path with exp=15 / overflow / renorm-to-subnormal.
static uint8_t EncodeE4M3NormalMagnitude(float absv, int sign)
{
    int exp_raw = 0;
    const float frac = std::frexp(absv, &exp_raw);
    const float norm_mant = frac * 2.0f;
    const int unbiased_exp = exp_raw - 1;
    int stored_exp = unbiased_exp + 7;

    float mant_scaled = (norm_mant - 1.0f) * 8.0f;
    if (mant_scaled < 0.0f) {
        mant_scaled = 0.0f;
    }
    int mant = RoundToNearestEvenFloatPos(mant_scaled);
    if (mant >= 0x8) {
        mant = 0;
        stored_exp += 1;
    }
    if (stored_exp > 0xf || (stored_exp == 0xf && mant >= 0x7)) {
        return static_cast<uint8_t>((sign << 0x7) | 0x7E);
    }
    if (stored_exp == 0xf) {
        return static_cast<uint8_t>((sign << 0x7) | (static_cast<uint8_t>(0xf) << 0x3) |
                                    static_cast<uint8_t>(mant & 0x7));
    }
    if (stored_exp <= 0) {
        return EncodeE4M3SubnormalFromAbsTimes512(absv / kE4M3EncodeMinSubnormal, sign);
    }

    const uint8_t exp_bits = static_cast<uint8_t>(stored_exp & 0xF);
    const uint8_t mant_bits = static_cast<uint8_t>(mant & 0x7);
    return static_cast<uint8_t>((sign << 0x7) | (exp_bits << 0x3) | mant_bits);
}

static torch::Tensor Fp8E4M3ToFloat32(const torch::Tensor& self)
{
    auto x = self.to(torch::kInt32);
    auto sign = 1.0f -
                (torch::bitwise_and(torch::bitwise_right_shift(x, at::Scalar(7)), at::Scalar(1))).to(torch::kFloat32) *
                    2.0f;
    auto exp_bits = torch::bitwise_and(torch::bitwise_right_shift(x, at::Scalar(3)), at::Scalar(0xF));
    auto mant_bits = torch::bitwise_and(x, at::Scalar(0x7));

    auto is_subnormal = (exp_bits == 0);
    auto subnormal_val = mant_bits.to(torch::kFloat32) * (1.0f / 8.0f) * (1.0f / 64.0f);

    auto is_normal = (exp_bits >= 1) & (exp_bits <= 0xe);
    auto exp_val = (exp_bits.to(torch::kFloat32) - 7.0f);
    auto mant_val = 1.0f + mant_bits.to(torch::kFloat32) / 8.0f;
    auto normal_val = torch::pow(2.0f, exp_val) * mant_val;

    auto is_e15 = (exp_bits == 0xf);
    auto is_e15_nan = is_e15 & (mant_bits == 0x7);
    auto is_e15_fin = is_e15 & (mant_bits < 0x7);
    auto e15_fin_val = 256.0f * (1.0f + mant_bits.to(torch::kFloat32) / 8.0f);
    auto nan_val = std::numeric_limits<float>::quiet_NaN();

    auto result = torch::zeros_like(self, torch::TensorOptions().dtype(torch::kFloat32));
    result = torch::where(is_subnormal, subnormal_val * sign, result);
    result = torch::where(is_normal, normal_val * sign, result);
    result = torch::where(is_e15_fin, sign * e15_fin_val, result);
    result = torch::where(is_e15_nan, nan_val, result);
    return result;
}

static torch::Tensor Fp8E5M2ToFloat32(const torch::Tensor& self)
{
    auto x = self.to(torch::kInt32);
    auto sign = 1.0f -
                (torch::bitwise_and(torch::bitwise_right_shift(x, at::Scalar(7)), at::Scalar(1))).to(torch::kFloat32) *
                    2.0f;
    auto exp_bits = torch::bitwise_and(torch::bitwise_right_shift(x, at::Scalar(2)), at::Scalar(0x1F));
    auto mant_bits = torch::bitwise_and(x, at::Scalar(0x3));

    auto is_subnormal = (exp_bits == 0);
    auto subnormal_val = mant_bits.to(torch::kFloat32) * (1.0f / 4.0f) * (1.0f / 16384.0f);

    auto is_normal = (exp_bits >= 1) & (exp_bits <= 0x1e);
    auto exp_val = (exp_bits.to(torch::kFloat32) - 15.0f);
    auto mant_val = 1.0f + mant_bits.to(torch::kFloat32) / 4.0f;
    auto normal_val = torch::pow(2.0f, exp_val) * mant_val;

    auto is_special = (exp_bits == 0x1f);
    auto is_inf = is_special & (mant_bits == 0);
    auto is_nan = is_special & (mant_bits != 0);
    auto inf_val = std::numeric_limits<float>::infinity();
    auto nan_val = std::numeric_limits<float>::quiet_NaN();

    auto result = torch::zeros_like(self, torch::TensorOptions().dtype(torch::kFloat32));
    result = torch::where(is_subnormal, subnormal_val * sign, result);
    result = torch::where(is_normal, normal_val * sign, result);
    result = torch::where(is_inf, sign * inf_val, result);
    result = torch::where(is_nan, nan_val, result);
    return result;
}

static torch::Tensor Fp8E8M0ToFloat32(const torch::Tensor& self)
{
    auto x = self.to(torch::kInt32);
    // E8M0 scale uses unsigned exponent semantics in practice for MX scaling.
    // Keep zero as exact zero, and decode others as power-of-two values.
    auto is_zero = (x == 0);
    auto exp_bits = torch::bitwise_and(x, at::Scalar(0xFF));
    auto exp_val = exp_bits.to(torch::kFloat32) - 127.0f;
    auto decoded = torch::pow(2.0f, exp_val);
    return torch::where(is_zero, torch::zeros_like(decoded), decoded);
}

static torch::Tensor Hf8ToFloat32(const torch::Tensor& self)
{
    auto x = self.to(torch::kInt32);
    auto sign = 1.0f -
                (torch::bitwise_and(torch::bitwise_right_shift(x, at::Scalar(7)), at::Scalar(1))).to(torch::kFloat32) *
                    2.0f;
    auto lower7 = torch::bitwise_and(x, at::Scalar(0x7F));
    auto out = torch::zeros_like(self, torch::TensorOptions().dtype(torch::kFloat32));

    // Subnormal: D=0000, M=bits[2:0], value=S_v*2^(M_v-23)
    auto isSub = (torch::bitwise_right_shift(lower7, at::Scalar(3)) == 0);
    auto subMant = torch::bitwise_and(lower7, at::Scalar(0x7)).to(torch::kFloat32);
    auto subVal = sign * torch::pow(2.0f, subMant - 23.0f);
    out = torch::where(isSub, subVal, out);

    // Normal-1: D=0001, E_v=0, M=bits[2:0]
    auto isN1 = (torch::bitwise_right_shift(lower7, at::Scalar(3)) == 1);
    auto n1Mant = torch::bitwise_and(lower7, at::Scalar(0x7)).to(torch::kFloat32) / 8.0f;
    auto n1Val = sign * (1.0f + n1Mant);
    out = torch::where(isN1, n1Val, out);

    // Remaining normal branches are prefix-coded by top bits of lower7.
    auto top3 = torch::bitwise_right_shift(lower7, at::Scalar(4));
    auto top2 = torch::bitwise_right_shift(lower7, at::Scalar(5));

    // N2: D=001, E=1bit -> E_v in {+1,-1}, M=bits[2:0]
    auto isN2 = (top3 == 1);
    auto n2ExpBits = torch::bitwise_and(torch::bitwise_right_shift(lower7, at::Scalar(3)), at::Scalar(0x1));
    auto n2Exp = 1.0f - 2.0f * n2ExpBits.to(torch::kFloat32); // 0->+1, 1->-1
    auto n2Mant = torch::bitwise_and(lower7, at::Scalar(0x7)).to(torch::kFloat32) / 8.0f;
    auto n2Val = sign * torch::pow(2.0f, n2Exp) * (1.0f + n2Mant);
    out = torch::where(isN2, n2Val, out);

    // N3: D=01, E=2bits (sign+magnitude with |E_v| in [2,3]), M=bits[2:0]
    auto isN3 = (top2 == 1);
    auto n3ExpBits = torch::bitwise_and(torch::bitwise_right_shift(lower7, at::Scalar(3)), at::Scalar(0x3));
    auto n3Sign = torch::bitwise_right_shift(n3ExpBits, at::Scalar(1));
    auto n3Mag = torch::bitwise_and(n3ExpBits, at::Scalar(0x1));
    auto n3Exp = (2.0f + n3Mag.to(torch::kFloat32)) * (1.0f - 2.0f * n3Sign.to(torch::kFloat32));
    auto n3Mant = torch::bitwise_and(lower7, at::Scalar(0x7)).to(torch::kFloat32) / 8.0f;
    auto n3Val = sign * torch::pow(2.0f, n3Exp) * (1.0f + n3Mant);
    out = torch::where(isN3, n3Val, out);

    // N4: D=10, E=3bits (|E_v| in [4,7]), M=bits[1:0]
    auto isN4 = (top2 == 0x2);
    auto n4ExpBits = torch::bitwise_and(torch::bitwise_right_shift(lower7, at::Scalar(2)), at::Scalar(0x7));
    auto n4Sign = torch::bitwise_right_shift(n4ExpBits, at::Scalar(2));
    auto n4Mag = torch::bitwise_and(n4ExpBits, at::Scalar(0x3));
    auto n4Exp = (4.0f + n4Mag.to(torch::kFloat32)) * (1.0f - 2.0f * n4Sign.to(torch::kFloat32));
    auto n4Mant = torch::bitwise_and(lower7, at::Scalar(0x3)).to(torch::kFloat32) / 4.0f;
    auto n4Val = sign * torch::pow(2.0f, n4Exp) * (1.0f + n4Mant);
    out = torch::where(isN4, n4Val, out);

    // N5: D=11, E=4bits (|E_v| in [8,15]), M=bits[0]
    auto isN5 = (top2 == 0x3);
    auto n5ExpBits = torch::bitwise_and(torch::bitwise_right_shift(lower7, at::Scalar(1)), at::Scalar(0xF));
    auto n5Sign = torch::bitwise_right_shift(n5ExpBits, at::Scalar(3));
    auto n5Mag = torch::bitwise_and(n5ExpBits, at::Scalar(0x7));
    auto n5Exp = (8.0f + n5Mag.to(torch::kFloat32)) * (1.0f - 2.0f * n5Sign.to(torch::kFloat32));
    auto n5Mant = torch::bitwise_and(lower7, at::Scalar(0x1)).to(torch::kFloat32) / 2.0f;
    auto n5Val = sign * torch::pow(2.0f, n5Exp) * (1.0f + n5Mant);
    out = torch::where(isN5, n5Val, out);
    return out;
}

torch::Tensor Fp8ToFloat32(const torch::Tensor& self, DataType actualType)
{
    if (actualType == DT_UINT8) {
        return self;
    }
    switch (actualType) {
        case DT_FP8:
        case DT_FP8E4M3:
            return Fp8E4M3ToFloat32(self);
        case DT_HF8:
            return Hf8ToFloat32(self);
        case DT_FP8E5M2:
            return Fp8E5M2ToFloat32(self);
        case DT_FP8E8M0:
            return Fp8E8M0ToFloat32(self);
        default:
            return self.to(torch::kFloat32);
    }
}

static inline uint8_t EncodeFloatToFp8E4M3(float v)
{
    if (std::isnan(v)) {
        const int sign = std::signbit(v) ? 1 : 0;
        return static_cast<uint8_t>((sign << 0x7) | 0x7F);
    }
    if (std::isinf(v)) {
        const int sign = std::signbit(v) ? 1 : 0;
        return static_cast<uint8_t>((sign << 0x7) | 0x7E);
    }
    if (std::fpclassify(v) == FP_ZERO) {
        return static_cast<uint8_t>(std::signbit(v) ? 0x80 : 0x00);
    }

    const float absv = std::fabs(v);
    const int sign = std::signbit(v) ? 1 : 0;
    if (absv < kE4M3EncodeMinSubnormal) {
        return static_cast<uint8_t>(sign << 0x7);
    }
    if (absv >= kE4M3EncodeMaxFinite) {
        return static_cast<uint8_t>((sign << 0x7) | 0x7E);
    }
    if (absv < kE4M3EncodeMinNormal) {
        return EncodeE4M3SubnormalFromAbsTimes512(absv / kE4M3EncodeMinSubnormal, sign);
    }
    return EncodeE4M3NormalMagnitude(absv, sign);
}

static inline uint8_t EncodeHf8Exponent(int exponent, int expBitCount)
{
    if (expBitCount <= 0) {
        return 0;
    }
    const int maxAbs = (1 << expBitCount) - 1;
    int absExp = std::abs(exponent);
    absExp = std::clamp(absExp, 1 << (expBitCount - 1), maxAbs);
    const int signBit = exponent < 0 ? (1 << (expBitCount - 1)) : 0;
    const int magnitudeMask = (1 << (expBitCount - 1)) - 1;
    const int encodedMagnitude = absExp - (1 << (expBitCount - 1));
    return static_cast<uint8_t>(signBit | (encodedMagnitude & magnitudeMask));
}

static inline uint8_t EncodeFloatToHf8(float v)
{
    if (std::fpclassify(v) == FP_ZERO || std::isnan(v)) {
        return 0;
    }
    const int sign = std::signbit(v) ? 1 : 0;
    const float absv = std::fabs(v);
    if (std::isinf(v)) {
        // Saturate to the largest representable normal branch.
        return static_cast<uint8_t>((sign << 0x7) | 0b11'0111'1);
    }
    int expRaw;
    float frac = std::frexp(absv, &expRaw); // absv = frac * 2^expRaw, frac in [0.5,1)
    float normalized = frac * 2.0f;
    int exponent = expRaw - 1;
    float mant = normalized - 1.0f;

    auto clampInt = [](int x, int lo, int hi) { return std::max(lo, std::min(hi, x)); };
    if (exponent <= -0x10) {
        // Subnormal branch: value=S_v*2^(M_v-23), M_v in [0,7]
        int mv = clampInt(static_cast<int>(std::round(std::log2(absv) + 23.0f)), 0, 0x7);
        return static_cast<uint8_t>((sign << 0x7) | mv);
    }
    if (exponent == 0) {
        int mv = clampInt(static_cast<int>(std::round(mant * 8.0f)), 0, 0x7);
        return static_cast<uint8_t>((sign << 0x7) | (0b0001 << 0x3) | mv);
    }
    if (std::abs(exponent) == 1) {
        int mvRaw = static_cast<int>(std::round(mant * 8.0f));
        if (mvRaw >= 0x8) {
            const float carried = std::ldexp(1.0f, exponent + 1);
            return EncodeFloatToHf8(sign ? -carried : carried);
        }
        int mv = clampInt(mvRaw, 0, 0x7);
        uint8_t e = EncodeHf8Exponent(exponent, 1);
        return static_cast<uint8_t>((sign << 0x7) | (0b001 << 0x4) | ((e & 0x1) << 0x3) | mv);
    }
    if (std::abs(exponent) <= 0x3) {
        int mvRaw = static_cast<int>(std::round(mant * 8.0f));
        if (mvRaw >= 0x8) {
            const float carried = std::ldexp(1.0f, exponent + 1);
            return EncodeFloatToHf8(sign ? -carried : carried);
        }
        int mv = clampInt(mvRaw, 0, 0x7);
        uint8_t e = EncodeHf8Exponent(exponent, 2);
        return static_cast<uint8_t>((sign << 0x7) | (0b01 << 0x5) | ((e & 0x3) << 0x3) | mv);
    }
    if (std::abs(exponent) <= 0x7) {
        int mvRaw = static_cast<int>(std::round(mant * 4.0f));
        if (mvRaw >= 0x4) {
            const float carried = std::ldexp(1.0f, exponent + 1);
            return EncodeFloatToHf8(sign ? -carried : carried);
        }
        int mv = clampInt(mvRaw, 0, 0x3);
        uint8_t e = EncodeHf8Exponent(exponent, 3);
        return static_cast<uint8_t>((sign << 0x7) | (0b10 << 0x5) | ((e & 0x7) << 0x2) | mv);
    }
    int mvRaw = static_cast<int>(std::round(mant * 2.0f));
    if (mvRaw >= 0x2) {
        const float carried = std::ldexp(1.0f, exponent + 1);
        return EncodeFloatToHf8(sign ? -carried : carried);
    }
    int mv = clampInt(mvRaw, 0, 1);
    uint8_t e = EncodeHf8Exponent(exponent, 4);
    return static_cast<uint8_t>((sign << 0x7) | (0b11 << 0x5) | ((e & 0xF) << 0x1) | mv);
}

static torch::Tensor Float32ToFp8E4M3(const torch::Tensor& self)
{
    auto x = self.to(torch::kFloat32).contiguous();
    auto flat = x.flatten();
    auto result = torch::empty_like(flat, torch::TensorOptions().dtype(torch::kUInt8));
    auto ptr = flat.data_ptr<float>();
    auto out_ptr = result.data_ptr<uint8_t>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
        out_ptr[i] = EncodeFloatToFp8E4M3(ptr[i]);
    }
    return result.reshape(x.sizes());
}

static torch::Tensor Float32ToHf8(const torch::Tensor& self)
{
    auto x = self.to(torch::kFloat32).contiguous();
    auto flat = x.flatten();
    auto result = torch::empty_like(flat, torch::TensorOptions().dtype(torch::kUInt8));
    auto ptr = flat.data_ptr<float>();
    auto outPtr = result.data_ptr<uint8_t>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
        outPtr[i] = EncodeFloatToHf8(ptr[i]);
    }
    return result.reshape(x.sizes());
}

static inline uint8_t EncodeFloatToFp8E5M2(float v)
{
    constexpr float kMinSubnormal = 1.0f / 65536.0f;
    constexpr float kMinNormal = 1.0f / 16384.0f;
    constexpr float kMaxVal = 57344.0f;
    if (std::isnan(v)) {
        return 0x7F;
    }
    if (std::isinf(v)) {
        return static_cast<uint8_t>((v < 0) ? 0xFC : 0x7C);
    }
    if (std::fpclassify(v) == FP_ZERO) {
        return static_cast<uint8_t>(std::signbit(v) ? 0x80 : 0x00);
    }
    float absv = std::fabs(v);
    int sign = std::signbit(v) ? 1 : 0;
    if (absv < kMinSubnormal) {
        return static_cast<uint8_t>(sign << 0x7);
    }
    if (absv > kMaxVal) {
        return static_cast<uint8_t>((sign << 0x7) | 0x7C);
    }
    if (absv < kMinNormal) {
        int mant = static_cast<int>(std::round(absv / kMinSubnormal));
        mant = std::clamp(mant, 1, 0x3);
        return static_cast<uint8_t>((sign << 0x7) | mant);
    }
    float log2v = std::log2(absv);
    int exp = static_cast<int>(std::round(log2v + 15.0f));
    exp = std::clamp(exp, 1, 0x1e);
    float scale = std::exp2(static_cast<float>(exp - 15));
    float scale_safe = (scale > 0.0f) ? scale : 1.0f;
    int mant = static_cast<int>(std::round((absv / scale_safe - 1.0f) * 4.0f));
    mant = std::clamp(mant, 0, 0x3);
    return static_cast<uint8_t>((sign << 0x7) | (exp << 0x2) | mant);
}

static torch::Tensor Float32ToFp8E5M2(const torch::Tensor& self)
{
    auto x = self.to(torch::kFloat32).contiguous();
    auto flat = x.flatten();
    auto result = torch::empty_like(flat, torch::TensorOptions().dtype(torch::kUInt8));
    auto ptr = flat.data_ptr<float>();
    auto out_ptr = result.data_ptr<uint8_t>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
        out_ptr[i] = EncodeFloatToFp8E5M2(ptr[i]);
    }
    return result.reshape(x.sizes());
}

static torch::Tensor Float32ToFp8E8M0(const torch::Tensor& self)
{
    auto x = self.to(torch::kFloat32).contiguous();
    auto flat = x.flatten();
    auto result = torch::empty_like(flat, torch::TensorOptions().dtype(torch::kUInt8));
    const float kMinVal = std::exp2(-126.0f);
    const float kMaxVal = std::exp2(127.0f);
    auto ptr = flat.data_ptr<float>();
    auto out_ptr = result.data_ptr<uint8_t>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
        float v = ptr[i];
        uint8_t enc = 0;
        if (std::isnan(v) || std::fpclassify(v) == FP_ZERO || v < 0.0f) {
            enc = 0;
        } else if (std::isinf(v)) {
            enc = 0xfe;
        } else {
            float clamped = std::clamp(v, kMinVal, kMaxVal);
            int exp = static_cast<int>(std::round(std::log2(clamped) + 127.0f));
            exp = std::clamp(exp, 1, 0xfe);
            enc = static_cast<uint8_t>(exp);
        }
        out_ptr[i] = enc;
    }
    return result.reshape(x.sizes());
}

torch::Tensor Float32ToFp8(const torch::Tensor& self, DataType actualType)
{
    switch (actualType) {
        case DT_FP8:
        case DT_FP8E4M3:
            return Float32ToFp8E4M3(self);
        case DT_HF8:
            return Float32ToHf8(self);
        case DT_FP8E5M2:
            return Float32ToFp8E5M2(self);
        case DT_FP8E8M0:
            return Float32ToFp8E8M0(self);
        default:
            return self.to(torch::kUInt8);
    }
}

static constexpr std::array<float, 16> kFp4E2M1DecodeTable = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static float DecodeFp4E2M1Nibble(uint8_t nib) { return kFp4E2M1DecodeTable[static_cast<size_t>(nib & 0x0F)]; }

static uint8_t EncodeFp4NibbleNearest(float v, float (*decodeFn)(uint8_t))
{
    if (std::isnan(v) || std::isinf(v)) {
        int sign = std::signbit(v) ? 1 : 0;
        return static_cast<uint8_t>((sign << 0x3) | 0x7);
    }
    if (std::fpclassify(v) == FP_ZERO) {
        return static_cast<uint8_t>(std::signbit(v) ? 0x8 : 0x0);
    }
    uint8_t best = 0;
    float bestErr = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 0x10; ++i) {
        float d = decodeFn(static_cast<uint8_t>(i));
        float err = std::fabs(d - v);
        if (err < bestErr) {
            bestErr = err;
            best = static_cast<uint8_t>(i);
        }
    }
    return best;
}

static uint8_t EncodeFp4E2M1Nibble(float v) { return EncodeFp4NibbleNearest(v, DecodeFp4E2M1Nibble); }

static constexpr std::array<float, 16> kFp4E1M2DecodeTable = {
    0.0f,  0.125f,  0.25f,  0.375f,  1.0f,  1.25f,  1.5f,  1.75f,
    -0.0f, -0.125f, -0.25f, -0.375f, -1.0f, -1.25f, -1.5f, -1.75f,
};

static float DecodeFp4E1M2Nibble(uint8_t nib) { return kFp4E1M2DecodeTable[static_cast<size_t>(nib & 0x0F)]; }

static uint8_t EncodeFp4E1M2Nibble(float v) { return EncodeFp4NibbleNearest(v, DecodeFp4E1M2Nibble); }

static float DecodeNibble(uint8_t nib, DataType actualType)
{
    switch (actualType) {
        case DT_FP4_E2M1:
            return DecodeFp4E2M1Nibble(nib);
        case DT_FP4_E1M2:
            return DecodeFp4E1M2Nibble(nib);
        default:
            return static_cast<float>(nib);
    }
}

static uint8_t EncodeNibble(float v, DataType actualType)
{
    switch (actualType) {
        case DT_FP4_E2M1:
            return EncodeFp4E2M1Nibble(v);
        case DT_FP4_E1M2:
            return EncodeFp4E1M2Nibble(v);
        default:
            return 0;
    }
}

torch::Tensor Fp4PackedToFloat32(const torch::Tensor& packed, DataType actualType)
{
    auto u8 = packed.to(torch::kUInt8).contiguous();
    auto sizes = u8.sizes().vec();
    if (sizes.empty()) {
        return torch::empty(sizes, torch::TensorOptions().dtype(torch::kFloat32));
    }
    int64_t lastPacked = sizes.back();
    std::vector<int64_t> outSizes = sizes;
    outSizes.back() = lastPacked * 0x2;

    auto out = torch::empty(outSizes, torch::TensorOptions().dtype(torch::kFloat32));
    if (lastPacked == 0) {
        return out;
    }
    int64_t outer = u8.numel() / lastPacked;
    const uint8_t* inPtr = u8.data_ptr<uint8_t>();
    float* outPtr = out.data_ptr<float>();

    for (int64_t i = 0; i < outer; ++i) {
        for (int64_t j = 0; j < lastPacked; ++j) {
            uint8_t b = inPtr[i * lastPacked + j];
            uint8_t hi = static_cast<uint8_t>((b >> 0x4) & 0x0F);
            uint8_t lo = static_cast<uint8_t>(b & 0x0F);
            // High nibble first: [e0, e1] -> byte = (e0 << 4) | e1
            outPtr[i * (lastPacked * 2) + j * 2] = DecodeNibble(hi, actualType);
            outPtr[i * (lastPacked * 2) + j * 2 + 1] = DecodeNibble(lo, actualType);
        }
    }
    return out;
}

torch::Tensor Float32ToFp4Packed(const torch::Tensor& self, DataType actualType)
{
    auto x = self.to(torch::kFloat32).contiguous();
    auto sizes = x.sizes().vec();
    if (sizes.empty()) {
        return torch::empty(sizes, torch::TensorOptions().dtype(torch::kUInt8));
    }
    int64_t lastFloat = sizes.back();
    ASSERT(CalculatorErrorScene::FP4_PACKED_LAST_DIM_INVALID, lastFloat % 0x2 == 0)
        << "FP4 packed conversion requires an even last dimension";
    int64_t lastPacked = lastFloat / 0x2;
    std::vector<int64_t> outSizes = sizes;
    outSizes.back() = lastPacked;

    auto out = torch::empty(outSizes, torch::TensorOptions().dtype(torch::kUInt8));
    if (lastFloat == 0) {
        return out;
    }
    int64_t outer = x.numel() / lastFloat;
    const float* inPtr = x.data_ptr<float>();
    uint8_t* outPtr = out.data_ptr<uint8_t>();

    for (int64_t i = 0; i < outer; ++i) {
        for (int64_t j = 0; j < lastPacked; ++j) {
            float f0 = inPtr[i * lastFloat + j * 2];
            float f1 = inPtr[i * lastFloat + j * 2 + 1];
            uint8_t n0 = EncodeNibble(f0, actualType);
            uint8_t n1 = EncodeNibble(f1, actualType);
            // High nibble first: [e0, e1] -> byte = (e0 << 4) | e1
            outPtr[i * lastPacked + j] = static_cast<uint8_t>((n0 << 0x4) | (n1 & 0x0F));
        }
    }
    return out;
}

} // namespace npu::tile_fwk
