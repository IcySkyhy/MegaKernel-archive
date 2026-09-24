/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_flow_verifier.cpp
 * \brief Unit tests for FlowVerifier::VerifyResult (FP8*, DT_HF8 decode path, padded raw views).
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>

#include "interface/interpreter/flow_verifier.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {

namespace {

LogicalTensorDataPtr MakeFp8E4m3Scalar(uint8_t byte)
{
    Tensor t(DT_FP8E4M3, {1});
    auto raw = RawTensorData::CreateConstantTensor(t, byte);
    return std::make_shared<LogicalTensorData>(raw);
}

LogicalTensorDataPtr MakeFp8E5m2Scalar(uint8_t byte)
{
    Tensor t(DT_FP8E5M2, {1});
    auto raw = RawTensorData::CreateConstantTensor(t, byte);
    return std::make_shared<LogicalTensorData>(raw);
}

LogicalTensorDataPtr MakeFp8E8m0Scalar(uint8_t byte)
{
    Tensor t(DT_FP8E8M0, {1});
    auto raw = RawTensorData::CreateConstantTensor(t, byte);
    return std::make_shared<LogicalTensorData>(raw);
}

LogicalTensorDataPtr MakeHf8Scalar(uint8_t byte)
{
    Tensor t(DT_HF8, {1});
    auto raw = RawTensorData::CreateConstantTensor(t, byte);
    return std::make_shared<LogicalTensorData>(raw);
}

constexpr float kVerifyRtol = 1e-3f;
constexpr float kVerifyAtol = 1e-2f;

LogicalTensorDataPtr MakeInt8ViewOnPaddedRaw(const std::vector<int8_t>& rawVals, const std::vector<int64_t>& rawShape,
                                             const std::vector<int64_t>& viewShape, const std::vector<int64_t>& offset)
{
    Tensor t(DT_INT8, rawShape);
    auto raw = RawTensorData::CreateTensor<int8_t>(t, rawVals);
    return std::make_shared<LogicalTensorData>(raw, viewShape, viewShape, offset);
}

} // namespace

// Golden 0x0E and output 0x0F are adjacent E4M3 codes (~0.0273 vs ~0.0293). Raw uint8 differs, but decode
// comparison with moderate atol passes (see flow_verifier FP8 decode path).
TEST(FlowVerifierFp8Test, E4M3AdjacentCodesDecodePassBytesDiffer)
{
    constexpr uint8_t kGoldenByte = 0x0E;
    constexpr uint8_t kOutputByte = 0x0F;
    const float kApproxGolden = std::ldexp(1.0f + 6.0f / 8.0f, 1 - 7); // exp=1, mant=6
    const float kApproxOutput = std::ldexp(1.0f + 7.0f / 8.0f, 1 - 7); // exp=1, mant=7

    auto golden = MakeFp8E4m3Scalar(kGoldenByte);
    auto output = MakeFp8E4m3Scalar(kOutputByte);

    ASSERT_EQ(golden->GetDataType(), DT_FP8E4M3);
    ASSERT_EQ(output->GetDataType(), DT_FP8E4M3);
    EXPECT_NE(golden->Get<uint8_t>(0), output->Get<uint8_t>(0));
    EXPECT_GT(std::fabs(kApproxOutput - kApproxGolden), 0.0f);
    EXPECT_LT(std::fabs(kApproxOutput - kApproxGolden), kVerifyAtol);

    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_TRUE(result.Check()) << "E4M3 0x0E vs 0x0F: decode+tol should pass while storage bytes differ";
}

// E5M2 [S][EEEEE][MM]: 0x14 = exp=5 mant=0, 0x15 = exp=5 mant=1 -> 2^-10 vs 2^-10*1.25
TEST(FlowVerifierFp8Test, E5M2AdjacentCodesDecodePassBytesDiffer)
{
    constexpr uint8_t kGoldenByte = 0x14;
    constexpr uint8_t kOutputByte = 0x15;
    const float kApproxGolden = std::ldexp(1.0f + 0.0f / 4.0f, 5 - 15);
    const float kApproxOutput = std::ldexp(1.0f + 1.0f / 4.0f, 5 - 15);

    auto golden = MakeFp8E5m2Scalar(kGoldenByte);
    auto output = MakeFp8E5m2Scalar(kOutputByte);

    ASSERT_EQ(golden->GetDataType(), DT_FP8E5M2);
    ASSERT_EQ(output->GetDataType(), DT_FP8E5M2);
    EXPECT_NE(golden->Get<uint8_t>(0), output->Get<uint8_t>(0));
    EXPECT_GT(std::fabs(kApproxOutput - kApproxGolden), 0.0f);
    EXPECT_LT(std::fabs(kApproxOutput - kApproxGolden), kVerifyAtol);

    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_TRUE(result.Check()) << "E5M2 0x14 vs 0x15: decode+tol should pass while storage bytes differ";
}

// E8M0 [S][EEEEEEE]: 0x32 vs 0x33 -> exp 50 vs 51 (positive), 2^(50-63) vs 2^(51-63)
TEST(FlowVerifierFp8Test, E8M0AdjacentCodesDecodePassBytesDiffer)
{
    constexpr uint8_t kGoldenByte = 0x32;
    constexpr uint8_t kOutputByte = 0x33;
    const float kApproxGolden = std::ldexp(1.0f, 50 - 63);
    const float kApproxOutput = std::ldexp(1.0f, 51 - 63);

    auto golden = MakeFp8E8m0Scalar(kGoldenByte);
    auto output = MakeFp8E8m0Scalar(kOutputByte);

    ASSERT_EQ(golden->GetDataType(), DT_FP8E8M0);
    ASSERT_EQ(output->GetDataType(), DT_FP8E8M0);
    EXPECT_NE(golden->Get<uint8_t>(0), output->Get<uint8_t>(0));
    EXPECT_GT(std::fabs(kApproxOutput - kApproxGolden), 0.0f);
    EXPECT_LT(std::fabs(kApproxOutput - kApproxGolden), kVerifyAtol);

    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_TRUE(result.Check()) << "E8M0 0x32 vs 0x33: decode+tol should pass while storage bytes differ";
}

// DT_HF8: VerifyResult decodes uint8 storage (HF8.md) then applies rtol/atol like FP8.
TEST(FlowVerifierHf8Test, SameEncodingPasses)
{
    constexpr uint8_t kByte = 0x2F;
    auto golden = MakeHf8Scalar(kByte);
    auto output = MakeHf8Scalar(kByte);
    ASSERT_EQ(golden->GetDataType(), DT_HF8);
    ASSERT_EQ(output->GetDataType(), DT_HF8);
    auto result = FlowVerifier::VerifyResult(golden, output, 0.0f, 0.0f);
    EXPECT_TRUE(result.Check());
}

// Subnormal adjacent codes: M_v=6 vs 7 -> 2^-17 vs 2^-16, small decode gap; raw bytes differ.
TEST(FlowVerifierHf8Test, SubnormalAdjacentCodesDecodePassBytesDiffer)
{
    constexpr uint8_t kGoldenByte = 0x06;
    constexpr uint8_t kOutputByte = 0x07;
    const float kGolden = std::exp2(static_cast<float>(6 - 23));
    const float kOutput = std::exp2(static_cast<float>(7 - 23));

    auto golden = MakeHf8Scalar(kGoldenByte);
    auto output = MakeHf8Scalar(kOutputByte);
    ASSERT_EQ(golden->GetDataType(), DT_HF8);
    ASSERT_EQ(output->GetDataType(), DT_HF8);
    EXPECT_NE(golden->Get<uint8_t>(0), output->Get<uint8_t>(0));
    EXPECT_GT(std::fabs(kOutput - kGolden), 0.0f);
    EXPECT_LT(std::fabs(kOutput - kGolden), kVerifyAtol);

    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_TRUE(result.Check()) << "HF8 subnormal 0x06 vs 0x07: decode+tol should pass while storage bytes differ";
}

// D=0001 normal: 0x08 -> 1.0, 0x09 -> 1.125; decode gap exceeds zero tolerance -> verify fails.
TEST(FlowVerifierHf8Test, NormalAdjacentMismatchFailsWithZeroTol)
{
    auto golden = MakeHf8Scalar(0x08);
    auto output = MakeHf8Scalar(0x09);
    auto result = FlowVerifier::VerifyResult(golden, output, 0.0f, 0.0f);
    EXPECT_FALSE(result.Check());
}

// Regression guard for packed/physical stride usage in verifier recursion:
// view shape is [2,2] on top of raw [2,4] with offset [0,1].
// Correct row stepping must follow raw stride(0)=4 instead of logical stride(0)=2.
TEST(FlowVerifierStrideTest, PaddedRawViewUsesPhysicalStride)
{
    const std::vector<int64_t> rawShape = {2, 4};
    const std::vector<int64_t> viewShape = {2, 2};
    const std::vector<int64_t> offset = {0, 1};

    // raw index mapping for view:
    // row0 -> [1,2], row1 -> [5,6]
    const std::vector<int8_t> goldenRaw = {
        99, 10, 11, 98, // row0
        97, 12, 13, 96  // row1
    };
    const std::vector<int8_t> outputRaw = {-1, 10, 11, -2, // padding bytes intentionally differ
                                           -3, 12, 13, -4};

    auto golden = MakeInt8ViewOnPaddedRaw(goldenRaw, rawShape, viewShape, offset);
    auto output = MakeInt8ViewOnPaddedRaw(outputRaw, rawShape, viewShape, offset);

    auto result = FlowVerifier::VerifyResult(golden, output, 0.0f, 0.0f);
    EXPECT_TRUE(result.Check()) << "Verifier should compare valid view elements only with physical raw stride stepping";
}

} // namespace npu::tile_fwk
