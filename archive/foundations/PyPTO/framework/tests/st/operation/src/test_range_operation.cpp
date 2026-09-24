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
 * \file test_range_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct RangeOpFuncArgs : public OpFuncArgs {
    RangeOpFuncArgs(const Element& start, const Element& end, const Element& step,
                    const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : start_(start), end_(end), step_(step), viewShape_(viewShape), tileShape_(tileShape)
    {}
    Element start_;
    Element end_;
    Element step_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct RangeOpMetaData {
    explicit RangeOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void RangeOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                  const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        auto args = static_cast<const RangeOpFuncArgs*>(opArgs);
        Element start = args->start_;
        Element end = args->end_;
        Element step = args->step_;
        int64_t bloop = 1;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Range(start, end, step);
            Assemble(res, {bIdx}, outputs[0]);
        }
    }
}

class RangeOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<RangeOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestRange, RangeOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<RangeOpMetaData>({RangeOperationExeFunc}, "Range")));

TEST_P(RangeOperationTest, TestRange)
{
    auto testCase = CreateTestCaseDesc<RangeOpMetaData>(GetParam(), nullptr);
    nlohmann::json test_data = GetParam().test_data_;
    Element start = GetElementByType(testCase.inputTensors[0].GetDataType(), test_data, "start");
    Element end = GetElementByType(testCase.inputTensors[1].GetDataType(), test_data, "end");
    Element step = GetElementByType(testCase.inputTensors[2].GetDataType(), test_data, "step");
    auto args = RangeOpFuncArgs(start, end, step, GetViewShape(test_data), GetTileShape(test_data));
    testCase.args = &args;
    TestExecutor::runTest(testCase);
}
} // namespace
