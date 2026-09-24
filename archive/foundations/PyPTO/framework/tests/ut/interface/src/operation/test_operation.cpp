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
 * \file test_operation.cpp
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"

using namespace npu::tile_fwk;

class OperationOpsTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
    void TearDown() override {}
};

TEST_F(OperationOpsTest, CheckIndexAddParamsInvalid_FP16_Overflow)
{
    std::vector<int64_t> selfShape = {10, 10};
    std::vector<int64_t> srcShape = {5, 10};
    std::vector<int64_t> indicesShape = {5};
    int axis = 0;

    Tensor self(DT_FP16, selfShape);
    Tensor src(DT_FP16, srcShape);
    Tensor indices(DT_INT32, indicesShape);
    Element alpha(DT_FP16, 65505.0f);

    EXPECT_THROW(IndexAdd_(self, src, indices, axis, alpha), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedStartDataType)
{
    Element start(DT_INT8, 0);
    Element end(DT_INT32, 10);
    Element step(DT_INT32, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedEndDataType)
{
    Element start(DT_INT32, 0);
    Element end(DT_INT8, 10);
    Element step(DT_INT32, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedStepDataType)
{
    Element start(DT_INT32, 0);
    Element end(DT_INT32, 10);
    Element step(DT_INT8, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedOutputDataType)
{
    Element start(DT_INT64, 0);
    Element end(DT_INT64, INT64_MAX);
    Element step(DT_INT64, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, LogicalNot_UnsupportedDataType)
{
    std::vector<int64_t> shape = {10, 10};
    Tensor input(DT_INT32, shape);

    EXPECT_THROW(LogicalNot(input), std::exception);
}

TEST_F(OperationOpsTest, QuantMX_DefaultRoundDownFp8Output)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    TileShape::Current().SetVecTile(8, 64);
    Tensor input(DT_FP32, {8, 64});

    FUNCTION("QuantMXDefaultFp8", {input})
    {
        auto defaultRes = QuantMX(input);
        EXPECT_EQ(std::get<0>(defaultRes).GetDataType(), DT_FP8E4M3);
        EXPECT_EQ(std::get<1>(defaultRes).GetDataType(), DT_FP8E8M0);
        EXPECT_EQ(std::get<1>(defaultRes).GetShape(), std::vector<int64_t>({8, 1, 2}));
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantMX_RoundUpFp8AndFp4Output)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    TileShape::Current().SetVecTile(8, 128);
    Tensor fp16Input(DT_FP16, {8, 128});
    Tensor fp32Input(DT_FP32, {8, 128});

    FUNCTION("QuantMXRoundUp", {fp16Input, fp32Input})
    {
        auto fp8Res = QuantMX(fp32Input, DT_FP8E4M3, DequantScaleRoundingMode::ROUND_UP, -1, true);
        EXPECT_EQ(std::get<0>(fp8Res).GetDataType(), DT_FP8E4M3);
        EXPECT_EQ(std::get<1>(fp8Res).GetDataType(), DT_FP8E8M0);

        auto fp4Res = QuantMX(fp16Input, DT_FP4_E2M1X2, DequantScaleRoundingMode::ROUND_UP, -1, true);
        EXPECT_EQ(std::get<0>(fp4Res).GetDataType(), DT_FP4_E2M1X2);
        EXPECT_EQ(std::get<1>(fp4Res).GetDataType(), DT_FP8E8M0);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantMX_Fp32ToFp4Unsupported)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    TileShape::Current().SetVecTile(8, 64);
    Tensor input(DT_FP32, {8, 64});

    FUNCTION("QuantMXFp32ToFp4Unsupported", {input})
    {
        EXPECT_THROW(QuantMX(input, DT_FP4_E2M1X2, DequantScaleRoundingMode::ROUND_DOWN, -1, true), std::exception);
        EXPECT_THROW(QuantMX(input, DT_FP4_E2M1X2, DequantScaleRoundingMode::ROUND_UP, -1, true), std::exception);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, DumpAttr_PrintsInplaceInfoMap)
{
    Program::GetInstance().Reset();
    Tensor self(DT_FP32, {2, 5});
    Tensor indices(DT_INT64, {1, 4});

    FUNCTION("DumpAttrScatterInplace", {self, indices})
    {
        TileShape::Current().SetVecTile(8, 8);
        auto out = Scatter(self, indices, Element(DT_FP32, 2.0f), 0, ScatterMode::ADD);
        (void)out;
        auto* func = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(func, nullptr);
        ASSERT_EQ(func->GetOperationSize(), 1);
        auto opDump = func->Operations(false).back().DumpAttr();
        EXPECT_NE(opDump.find("#INPLACE_INFO{0:0}"), std::string::npos);
        EXPECT_EQ(opDump.find("unsupported type"), std::string::npos);

        auto opAttrJson = func->Operations(false).back().DumpAttrJson(OpAttributeKey::inplaceInfo);
        ASSERT_TRUE(opAttrJson.is_object());
        ASSERT_TRUE(opAttrJson.contains("0"));
        EXPECT_EQ(opAttrJson["0"], 0);
    }
}

TEST_F(OperationOpsTest, Axpy_UnsupportedSelfDim)
{
    std::vector<int64_t> shape5d = {2, 4, 8, 16, 32};
    Tensor self(DT_FP32, shape5d);
    Tensor other(DT_FP32, shape5d);

    FUNCTION("AxpyUnsupportedSelfDim", {self, other}) { EXPECT_THROW(Axpy(self, other, 1.0f), std::exception); }
}

TEST_F(OperationOpsTest, Axpy_UnsupportedOtherDim)
{
    std::vector<int64_t> shape4d = {4, 8, 16, 32};
    std::vector<int64_t> shape5d = {2, 4, 8, 16, 32};
    Tensor self(DT_FP32, shape4d);
    Tensor other(DT_FP32, shape5d);

    FUNCTION("AxpyUnsupportedOtherDim", {self, other}) { EXPECT_THROW(Axpy(self, other, 1.0f), std::exception); }
}
