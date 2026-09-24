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
 * \file test_interp_type_convert.cpp
 * \brief Unit tests for interpreter internal type conversion: FP8 (E4M3, E5M2, E8M0), FP4 packed (E2M1X2, E1M2X2).
 */

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk {

template <typename T>
static LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int64_t>& shape, const T& val)
{
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor(data, val));
}

template <typename T>
static LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int64_t>& shape, const std::vector<T>& vals)
{
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateTensor(data, vals));
}

#define ASSERT_ALLCLOSE(self, other) \
    ASSERT(calc::AllClose(self, other)) << "lhs:\n" << self->ToString() << "\nrhs:\n" << other->ToString() << "\n"

#define ASSERT_ALLCLOSE_ATOL(lhs, rhs, atol) \
    ASSERT(calc::AllClose(lhs, rhs, atol, 1e-5)) << "lhs:\n" << lhs->ToString() << "\nrhs:\n" << rhs->ToString() << "\n"

// Compare FP8 output with golden: Cast FP8 to FP32 first to avoid GetElement crash on FP8 (raw_tensor_data lacks FP8
// support)
#define ASSERT_FP8_ALLCLOSE_ATOL(fp8_out, golden, atol)                     \
    do {                                                                    \
        auto _out_f32 = makeTensorData(DT_FP32, fp8_out->GetShape(), 0.0f); \
        calc::Cast(_out_f32, fp8_out);                                      \
        ASSERT_ALLCLOSE_ATOL(_out_f32, golden, atol);                       \
    } while (0)

class InterpTypeConvertTest : public testing::Test {
public:
    void SetUp() override
    {
        if (!calc::IsVerifyEnabled()) {
            GTEST_SKIP() << "Torch verifier not enabled, skip type conversion tests";
        }
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(InterpTypeConvertTest, Fp8SameBitsDifferentFormats)
{
    // A few representative encodings with the *same* 8 bits across formats,
    // but interpreted differently by E4M3/E5M2/E8M0.
    // E8M0 here follows unsigned exponent semantics:
    //   bits == 0 -> 0
    //   bits != 0 -> 2^(bits - 127)
    //
    // 0x55 = 0101 0101:
    //   E4M3 [S][EEEE][MMM]: S=0, E=1010=10, M=101=5 -> 2^(10-7)*(1+5/8) = 13.0
    //   E5M2 [S][EEEEE][MM]: S=0, E=10101=21, M=01=1 -> 2^(21-15)*(1+1/4) = 80.0
    //   E8M0: bits=85 -> 2^(85-127) = 2^(-42)
    //
    // 0x38 = 0011 1000:
    //   E4M3: exp=7, mant=0 -> 2^(7-7)*(1+0) = 1.0
    //   E5M2: exp=14, mant=0 -> 2^(14-15)*(1+0) = 0.5
    //   E8M0: bits=56 -> 2^(56-127) = 2^(-71)
    //
    // 0xA0 = 1010 0000:
    //   E4M3: sign=-, exp=4, mant=0  -> -2^(4-7)  = -2^(-3)  = -0.125
    //   E5M2: sign=-, exp=8, mant=0  -> -2^(8-15) = -2^(-7)
    //   E8M0: bits=160 -> 2^(160-127) = 2^(33)
    struct {
        uint8_t bits;
        float expect_e4m3;
        float expect_e5m2;
        float expect_e8m0;
    } cases[] = {
        {0x55, 13.0f, 80.0f, std::exp2(-42.0f)},
        {0x38, 1.0f, 0.5f, std::exp2(-71.0f)},
        {0xA0, -0.125f, -std::exp2(-7.0f), std::exp2(33.0f)},
    };

    for (const auto& c : cases) {
        auto e4m3_src = makeTensorData(DT_FP8E4M3, {4}, static_cast<uint8_t>(c.bits));
        auto e5m2_src = makeTensorData(DT_FP8E5M2, {4}, static_cast<uint8_t>(c.bits));
        auto e8m0_src = makeTensorData(DT_FP8E8M0, {4}, static_cast<uint8_t>(c.bits));

        auto e4m3_out = makeTensorData(DT_FP32, {4}, 0.0f);
        auto e5m2_out = makeTensorData(DT_FP32, {4}, 0.0f);
        auto e8m0_out = makeTensorData(DT_FP32, {4}, 0.0f);

        calc::Cast(e4m3_out, e4m3_src);
        calc::Cast(e5m2_out, e5m2_src);
        calc::Cast(e8m0_out, e8m0_src);

        auto golden_e4m3 = makeTensorData(DT_FP32, {4}, c.expect_e4m3);
        auto golden_e5m2 = makeTensorData(DT_FP32, {4}, c.expect_e5m2);
        auto golden_e8m0 = makeTensorData(DT_FP32, {4}, c.expect_e8m0);

        ASSERT_ALLCLOSE_ATOL(e4m3_out, golden_e4m3, 1e-6f);
        ASSERT_ALLCLOSE_ATOL(e5m2_out, golden_e5m2, 1e-6f);
        ASSERT_ALLCLOSE_ATOL(e8m0_out, golden_e8m0, 1e-6f);
    }
}

TEST_F(InterpTypeConvertTest, Fp8SubnormalSameBitsDifferentFormats)
{
    // Encodings where E4M3/E5M2 see subnormals but E8M0 uses unsigned exponent decode.
    //
    // 0x01 = 0000 0001:
    //   E4M3: S=0, E=0, M=1 ->  1/512
    //   E5M2: S=0, E=0, M=1 ->  1/65536
    //   E8M0: bits=1        ->  2^(-126)
    //
    // 0x02 = 0000 0010:
    //   E4M3: S=0, E=0, M=2 ->  2/512  = 1/256
    //   E5M2: S=0, E=0, M=2 ->  2/65536 = 1/32768
    //   E8M0: bits=2        ->  2^(-125)
    //
    // 0x81 = 1000 0001:
    //   E4M3: S=1, E=0, M=1 -> -1/512
    //   E5M2: S=1, E=0, M=1 -> -1/65536
    //   E8M0: bits=129      ->  2^(2)
    struct {
        uint8_t bits;
        float expect_e4m3;
        float expect_e5m2;
        float expect_e8m0;
    } cases[] = {
        {0x01, 1.0f / 512.0f, 1.0f / 65536.0f, std::exp2(-126.0f)},
        {0x02, 2.0f / 512.0f, 2.0f / 65536.0f, std::exp2(-125.0f)},
        {0x81, -1.0f / 512.0f, -1.0f / 65536.0f, 4.0f},
    };

    for (const auto& c : cases) {
        auto e4m3_src = makeTensorData(DT_FP8E4M3, {4}, static_cast<uint8_t>(c.bits));
        auto e5m2_src = makeTensorData(DT_FP8E5M2, {4}, static_cast<uint8_t>(c.bits));
        auto e8m0_src = makeTensorData(DT_FP8E8M0, {4}, static_cast<uint8_t>(c.bits));

        auto e4m3_out = makeTensorData(DT_FP32, {4}, 0.0f);
        auto e5m2_out = makeTensorData(DT_FP32, {4}, 0.0f);
        auto e8m0_out = makeTensorData(DT_FP32, {4}, 0.0f);

        calc::Cast(e4m3_out, e4m3_src);
        calc::Cast(e5m2_out, e5m2_src);
        calc::Cast(e8m0_out, e8m0_src);

        auto golden_e4m3 = makeTensorData(DT_FP32, {4}, c.expect_e4m3);
        auto golden_e5m2 = makeTensorData(DT_FP32, {4}, c.expect_e5m2);
        auto golden_e8m0 = makeTensorData(DT_FP32, {4}, c.expect_e8m0);

        ASSERT_ALLCLOSE_ATOL(e4m3_out, golden_e4m3, 1e-8f);
        ASSERT_ALLCLOSE_ATOL(e5m2_out, golden_e5m2, 1e-10f); // 更小值，需更紧容差或用 rel tol
        ASSERT_ALLCLOSE_ATOL(e8m0_out, golden_e8m0, 1e-6f);
    }
}
namespace {
constexpr uint8_t kFp8OperandBits = 0x55;
constexpr DataType kFp8Types[] = {DT_FP8E4M3, DT_FP8E5M2, DT_FP8E8M0};
constexpr float kFp8GoldenPerType[] = {13.0f, 80.0f, 2.2737368e-13f}; // 2^-42
} // namespace

TEST_F(InterpTypeConvertTest, Fp8ToOperandCast)
{
    auto fp32_src = makeTensorData(DT_FP32, {4, 4}, 2.0f);
    auto golden = makeTensorData(DT_FP32, {4, 4}, 2.0f);
    for (DataType dtype : kFp8Types) {
        auto out = makeTensorData(dtype, {4, 4}, std::vector<uint8_t>(16, 0));
        calc::Cast(out, fp32_src);
        ASSERT_FP8_ALLCLOSE_ATOL(out, golden, 1e-1);
    }
}

TEST_F(InterpTypeConvertTest, Fp8ToOperandAddS)
{
    for (size_t i = 0; i < 3; ++i) {
        DataType dtype = kFp8Types[i];
        auto self = makeTensorData(dtype, {4, 4}, static_cast<uint8_t>(kFp8OperandBits));
        auto out = makeTensorData(dtype, {4, 4}, std::vector<uint8_t>(16, 0));
        auto golden = makeTensorData(DT_FP32, {4, 4}, kFp8GoldenPerType[i]);
        calc::AddS(out, self, Element(DT_FP32, 0.0f));
        ASSERT_FP8_ALLCLOSE_ATOL(out, golden, 1e-5);
    }
}

TEST_F(InterpTypeConvertTest, Fp8ToOperandNeg)
{
    for (size_t i = 0; i < 3; ++i) {
        DataType dtype = kFp8Types[i];
        auto self = makeTensorData(dtype, {4, 4}, static_cast<uint8_t>(kFp8OperandBits));
        auto out = makeTensorData(dtype, {4, 4}, std::vector<uint8_t>(16, 0));
        float expected = (dtype == DT_FP8E8M0) ? 0.0f : -kFp8GoldenPerType[i];
        auto golden = makeTensorData(DT_FP32, {4, 4}, expected);
        calc::Neg(out, self);
        ASSERT_FP8_ALLCLOSE_ATOL(out, golden, 1e-5);
    }
}

TEST_F(InterpTypeConvertTest, Fp8ToOperandSqrt)
{
    auto fp32_src = makeTensorData(DT_FP32, {4, 4}, 4.0f);
    auto golden = makeTensorData(DT_FP32, {4, 4}, 2.0f);
    for (DataType dtype : kFp8Types) {
        auto out = makeTensorData(dtype, {4, 4}, std::vector<uint8_t>(16, 0));
        calc::Sqrt(out, fp32_src);
        ASSERT_FP8_ALLCLOSE_ATOL(out, golden, 1e-1);
    }
}

TEST_F(InterpTypeConvertTest, Fp8ToOperandMul)
{
    constexpr uint8_t kMulFactorBits = 0x01;
    const float factor_per_type[3] = {1.0f / 512.0f, 1.0f / 65536.0f, std::exp2(-126.0f)};
    for (size_t i = 0; i < 3; ++i) {
        DataType dtype = kFp8Types[i];
        auto a = makeTensorData(dtype, {4, 4}, static_cast<uint8_t>(kFp8OperandBits));
        auto b = makeTensorData(dtype, {4, 4}, static_cast<uint8_t>(kMulFactorBits));
        auto out = makeTensorData(dtype, {4, 4}, std::vector<uint8_t>(16, 0));
        auto golden = makeTensorData(DT_FP32, {4, 4}, kFp8GoldenPerType[i] * factor_per_type[i]);
        calc::Mul(out, a, b);
        ASSERT_FP8_ALLCLOSE_ATOL(out, golden, 1e-5);
    }
}

// FP8 E4M3 (OCP float8_e4m3fn): +0(0x00), -0(0x80), +Max(0x7E)=448, -Max(0xFE), NaN(0x7F/0xFF)
TEST_F(InterpTypeConvertTest, Fp8E4M3SpecialValues)
{
    struct {
        uint8_t enc;
        float expected;
        bool is_nan;
    } cases[] = {
        {0x00, 0.0f, false},    {0x80, -0.0f, false}, {0x7E, 448.0f, false},
        {0xFE, -448.0f, false}, {0x7F, 0.0f, true},   {0xFF, 0.0f, true},
    };

    for (const auto& c : cases) {
        auto src = makeTensorData(DT_FP8E4M3, {4}, static_cast<uint8_t>(c.enc));
        auto out = makeTensorData(DT_FP32, {4}, 0.0f);
        calc::Cast(out, src);

        if (c.is_nan) {
            for (int64_t i = 0; i < 4; ++i) {
                ASSERT(std::isnan(out->Get<float>(i))) << "expected NaN at index " << i;
            }
        } else {
            auto golden = makeTensorData(DT_FP32, {4}, c.expected);
            ASSERT_ALLCLOSE(out, golden);
        }
    }
}

// FP8 E5M2 special values: +0(0x00), -0(0x80), +Inf(0x7C), -Inf(0xFC), NaN(0x7F)
TEST_F(InterpTypeConvertTest, Fp8E5M2SpecialValues)
{
    struct {
        uint8_t enc;
        float expected;
        bool is_nan;
        bool is_inf;
    } cases[] = {
        {0x00, 0.0f, false, false},
        {0x80, -0.0f, false, false},
        {0x7C, std::numeric_limits<float>::infinity(), false, true},  // +Inf
        {0xFC, -std::numeric_limits<float>::infinity(), false, true}, // -Inf
        {0x7F, 0.0f, true, false},                                    // NaN
    };

    for (const auto& c : cases) {
        auto src = makeTensorData(DT_FP8E5M2, {4}, static_cast<uint8_t>(c.enc));
        auto out = makeTensorData(DT_FP32, {4}, 0.0f);
        calc::Cast(out, src);

        if (c.is_nan) {
            for (int64_t i = 0; i < 4; ++i) {
                ASSERT(std::isnan(out->Get<float>(i))) << "expected NaN at index " << i;
            }
        } else if (c.is_inf) {
            for (int64_t i = 0; i < 4; ++i) {
                float v = out->Get<float>(i);
                ASSERT(std::isinf(v)) << "expected Inf at index " << i;
                if (c.expected > 0.0f) {
                    ASSERT(v > 0.0f) << "expected +Inf at index " << i;
                } else {
                    ASSERT(v < 0.0f) << "expected -Inf at index " << i;
                }
            }
        } else {
            auto golden = makeTensorData(DT_FP32, {4}, c.expected);
            ASSERT_ALLCLOSE(out, golden);
        }
    }
}

// FP4 E2M1 packed (DT_FP4_E2M1): 1 byte = 2 nibbles (high 4 bits first, low 4 bits second).
// Tensor shape uses logical FP4 elements.
TEST_F(InterpTypeConvertTest, Fp4E2M1PackedDecodeToFp32)
{
    // 0x10: high=1 -> E2M1 subnormal mant=1 -> 0.5; low=0 -> 0
    // 0x23: high=2 -> exp=1,m=0 -> 1.0; low=3 -> exp=1,m=1 -> 1.5
    const std::vector<uint8_t> packed = {0x10, 0x23};
    auto src = makeTensorData(DT_FP4_E2M1, {4}, packed);
    auto out = makeTensorData(DT_FP32, {4}, 0.0f);
    calc::Cast(out, src);
    const std::vector<float> goldenVals = {0.5f, 0.0f, 1.0f, 1.5f};
    auto golden = makeTensorData(DT_FP32, {4}, goldenVals);
    ASSERT_ALLCLOSE_ATOL(out, golden, 1e-6f);
}

// FP4 E1M2 packed (DT_FP4_E1M2): verify nibble decode order and value mapping.
TEST_F(InterpTypeConvertTest, Fp4E1M2PackedDecodeToFp32)
{
    // 0x14: high=1 -> E1M2 e=0,m=1 -> 0.125
    //       low=4  -> E1M2 e=1,m=0 -> 1.0
    // 0xFC: high=15-> sign=1,e=1,m=3 -> -1.75
    //       low=12 -> sign=1,e=1,m=0 -> -1.0
    const std::vector<uint8_t> packed = {0x14, 0xFC};
    auto src = makeTensorData(DT_FP4_E1M2, {4}, packed);
    auto out = makeTensorData(DT_FP32, {4}, 0.0f);
    calc::Cast(out, src);
    const std::vector<float> goldenVals = {0.125f, 1.0f, -1.75f, -1.0f};
    auto golden = makeTensorData(DT_FP32, {4}, goldenVals);
    ASSERT_ALLCLOSE_ATOL(out, golden, 1e-6f);
}

// End-to-end FP4 arithmetic validation:
// FP4(E2M1X2) + FP4(E2M1X2) -> FP4(E2M1X2), then cast back to FP32 and compare with golden.
// This validates decode (From), torch op, and encode back (ToOperand).
TEST_F(InterpTypeConvertTest, Fp4E2M1AddRoundTripViaFp32)
{
    // a (packed {0x12,0x9A}) -> [0.5, 1.0, -0.5, -1.0]
    // b (packed {0x21,0x92}) -> [1.0, 0.5, -0.5,  1.0]
    // sum                    -> [1.5, 1.5, -1.0,  0.0] (all exactly representable in E2M1)
    const std::vector<uint8_t> aPacked = {0x12, 0x9A};
    const std::vector<uint8_t> bPacked = {0x21, 0x92};

    auto a = makeTensorData(DT_FP4_E2M1, {4}, aPacked);
    auto b = makeTensorData(DT_FP4_E2M1, {4}, bPacked);
    auto out = makeTensorData(DT_FP4_E2M1, {4}, std::vector<uint8_t>(2, 0));
    calc::Add(out, a, b);

    auto outF32 = makeTensorData(DT_FP32, {4}, 0.0f);
    calc::Cast(outF32, out);

    const std::vector<float> goldenVals = {1.5f, 1.5f, -1.0f, 0.0f};
    auto golden = makeTensorData(DT_FP32, {4}, goldenVals);
    ASSERT_ALLCLOSE_ATOL(outF32, golden, 1e-6f);
}

TEST_F(InterpTypeConvertTest, Hf8DecodeSubnormalAndNormal)
{
    struct {
        uint8_t bits;
        float expected;
    } cases[] = {
        // Subnormal (D=0000): value = S_v * 2^(M_v - 23)
        {0x00, std::exp2(-23.0f)},  // +, M_v=0
        {0x07, std::exp2(-16.0f)},  // +, M_v=7
        {0x87, -std::exp2(-16.0f)}, // -, M_v=7
        // Normal-1 (D=0001): value = S_v * 2^0 * (1 + M_v)
        {0x08, 1.0f},    // M_v=0/8
        {0x0F, 1.875f},  // M_v=7/8
        {0x8F, -1.875f}, // sign=-
    };

    for (const auto& c : cases) {
        auto src = makeTensorData(DT_HF8, {4}, static_cast<uint8_t>(c.bits));
        auto out = makeTensorData(DT_FP32, {4}, 0.0f);
        calc::Cast(out, src);
        auto golden = makeTensorData(DT_FP32, {4}, c.expected);
        ASSERT_ALLCLOSE_ATOL(out, golden, 1e-6f);
    }
}

TEST_F(InterpTypeConvertTest, Hf8DecodeNormalExponentBranches)
{
    struct {
        uint8_t bits;
        float expected;
    } cases[] = {
        // D=001, E_v in {+1,-1}, M_v in [0/8,7/8]
        {0x10, 2.0f},    // E_v=+1, M_v=0
        {0x1F, 0.9375f}, // E_v=-1, M_v=7/8
        {0x18, 0.5f},    // E_v=-1, M_v=0
        // D=01, E_v in ±[2,3], M_v in [0/8,7/8]
        {0x20, 4.0f},  // E_v=+2, M_v=0
        {0x2F, 15.0f}, // E_v=+3, M_v=7/8
        {0x30, 0.25f}, // E_v=-2, M_v=0
        // D=10, E_v in ±[4,7], M_v in [0/4,3/4]
        {0x40, 16.0f}, // E_v=+4, M_v=0
        {0x43, 28.0f}, // E_v=+4, M_v=3/4
        // D=11, E_v in ±[8,15], M_v in [0/2,1/2]
        {0x60, 256.0f}, // E_v=+8, M_v=0
        {0x61, 384.0f}, // E_v=+8, M_v=1/2
    };

    for (const auto& c : cases) {
        auto src = makeTensorData(DT_HF8, {4}, static_cast<uint8_t>(c.bits));
        auto out = makeTensorData(DT_FP32, {4}, 0.0f);
        calc::Cast(out, src);
        auto golden = makeTensorData(DT_FP32, {4}, c.expected);
        ASSERT_ALLCLOSE_ATOL(out, golden, 1e-5f);
    }
}

TEST_F(InterpTypeConvertTest, Hf8RoundTripViaCast)
{
    // These values are selected from exact HF8 grid points across branches.
    const std::vector<float> vals = {
        std::exp2(-23.0f), // subnormal
        std::exp2(-16.0f), // subnormal max
        1.0f,              // D=0001, M=0
        1.875f,            // D=0001, M=7/8
        2.0f,              // D=001, E=+1, M=0
        0.5f,              // D=001, E=-1, M=0
        4.0f,              // D=01,  E=+2, M=0
        0.25f,             // D=01,  E=-2, M=0
        28.0f,             // D=10,  E=+4, M=3/4
        384.0f,            // D=11,  E=+8, M=1/2
        -1.875f,           // negative normal
        -0.5f,             // negative normal
    };

    auto src = makeTensorData(DT_FP32, {static_cast<int64_t>(vals.size())}, vals);
    auto hf8 = makeTensorData(DT_HF8, {static_cast<int64_t>(vals.size())}, std::vector<uint8_t>(vals.size(), 0));
    auto out = makeTensorData(DT_FP32, {static_cast<int64_t>(vals.size())}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {static_cast<int64_t>(vals.size())}, vals);

    calc::Cast(hf8, src); // Float32 -> HF8 encode
    calc::Cast(out, hf8); // HF8 -> Float32 decode
    ASSERT_ALLCLOSE_ATOL(out, golden, 1e-5f);
}

TEST_F(InterpTypeConvertTest, Fp32CastToHf8QuantizationCases)
{
    // Representative quantization checks around D=10/D=11 branches and negative carry boundary.
    const std::vector<float> vals = {-7640.0f, -258.0f, -247.0f, 220.0f, 200.0f, 180.0f, 160.0f};
    const std::vector<float> expected = {-8192.0f, -256.0f, -256.0f, 224.0f, 192.0f, 192.0f, 160.0f};

    auto src = makeTensorData(DT_FP32, {static_cast<int64_t>(vals.size())}, vals);
    auto hf8 = makeTensorData(DT_HF8, {static_cast<int64_t>(vals.size())}, std::vector<uint8_t>(vals.size(), 0));
    auto out = makeTensorData(DT_FP32, {static_cast<int64_t>(vals.size())}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {static_cast<int64_t>(vals.size())}, expected);

    calc::Cast(hf8, src); // FP32 -> HF8
    calc::Cast(out, hf8); // HF8 -> FP32 decode for validation
    ASSERT_ALLCLOSE_ATOL(out, golden, 1e-5f);
}

} // namespace npu::tile_fwk
