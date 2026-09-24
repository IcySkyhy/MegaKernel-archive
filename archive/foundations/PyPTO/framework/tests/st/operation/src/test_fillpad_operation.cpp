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
 * \file test_fillpad_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct FillPadOpFuncArgs : public OpFuncArgs {
    FillPadOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, float padValue)
        : viewShape_(viewShape), tileShape_(tileShape), padValue_(padValue)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    float padValue_;
};

struct FillPadOpMetaData {
    explicit FillPadOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void FillPadOperationExeFunc1Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        const struct FillPadOpFuncArgs* args = static_cast<const FillPadOpFuncArgs*>(opArgs);
        const int firstViewShape = inputs[0].GetShape()[0];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            auto tileTensor = View(inputs[0], {firstViewShape}, SymbolicScalar::FromConcrete(args->viewShape_),
                                   {bIdx * firstViewShape});
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = FillPad(tileTensor, "constant", Element(tileTensor.GetDataType(), args->padValue_));
            Assemble(res, {bIdx * firstViewShape}, outputs[0]);
        }
    }
}

static void FillPadOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const struct FillPadOpFuncArgs* args = static_cast<const FillPadOpFuncArgs*>(opArgs);
        const int firstViewShape = inputs[0].GetShape()[0];
        const int secondViewShape = inputs[0].GetShape()[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                auto tileTensor = View(inputs[0], {firstViewShape, secondViewShape},
                                       SymbolicScalar::FromConcrete(args->viewShape_),
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = FillPad(tileTensor, "constant", Element(tileTensor.GetDataType(), args->padValue_));
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

class FillPadOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<FillPadOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestFillPad, FillPadOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<FillPadOpMetaData>(
                             {FillPadOperationExeFunc2Dims, FillPadOperationExeFunc1Dims}, "FillPad")));

TEST_P(FillPadOperationTest, TestFillPad)
{
    auto test_data = GetParam().test_data_;
    std::string padValueType = GetValueByName<std::string>(test_data, "pad_value_type");
    float padValue = 0.0f;
    if (padValueType == "min") {
        padValue = -std::numeric_limits<float>::infinity();
    } else if (padValueType == "max") {
        padValue = std::numeric_limits<float>::infinity();
    } else if (padValueType == "zero") {
        padValue = 0.0f;
    } else if (padValueType == "custom") {
        padValue = GetValueByName<float>(test_data, "pad_value");
    }
    auto args = FillPadOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), padValue);
    auto testCase = CreateTestCaseDesc<FillPadOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
