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
 * \file test_uniform_operation.cpp
 * \brief Test Uniform operation
 */

#include "test_operation.h"
#include <iostream>

using namespace tile_fwk::test_operation;
namespace {

struct UniformOpFuncArgs : public OpFuncArgs {
    UniformOpFuncArgs(uint64_t key, uint64_t counter0, uint64_t counter1, const std::vector<int64_t>& viewShape,
                      const std::vector<int64_t> tileShape, uint16_t rounds, DataType dtype)
        : key_(key),
          counter0_(counter0),
          counter1_(counter1),
          viewShape_(viewShape),
          tileShape_(tileShape),
          rounds_(rounds),
          dtype_(dtype)
    {}

    uint64_t key_;
    uint64_t counter0_;
    uint64_t counter1_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    uint16_t rounds_;
    DataType dtype_;
};

struct UniformOpMetaData {
    explicit UniformOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static DataType GetOutputDtype(const nlohmann::json& test_data)
{
    if (test_data.contains("output_tensors") && test_data["output_tensors"].is_array() &&
        !test_data["output_tensors"].empty()) {
        auto& outputTensor = test_data["output_tensors"][0];
        if (outputTensor.contains("dtype")) {
            std::string dtypeStr = outputTensor["dtype"].get<std::string>();
            if (dtypeStr == "fp32") {
                return DT_FP32;
            } else if (dtypeStr == "fp16") {
                return DT_FP16;
            } else if (dtypeStr == "bf16") {
                return DT_BF16;
            }
        }
    }
    return DT_FP32;
}

static void UniformOperationExeFunc([[maybe_unused]] const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                    const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        auto args = static_cast<const UniformOpFuncArgs*>(opArgs);

        const int64_t outputDim = outputs[0].GetShape()[0];
        const int64_t viewShape = args->viewShape_[0];

        const int loop = (outputDim + viewShape - 1) / viewShape;
        int64_t tileIdx = 0;

        LOOP("LOOP_L0_idx", FunctionType::DYNAMIC_LOOP, idx, LoopRange(0, loop, 1))
        {
            auto offset = idx * viewShape;
            int64_t validShape = std::min(outputDim - tileIdx * viewShape, viewShape);

            SymbolicScalar tileCounter0 = SymbolicScalar(static_cast<int64_t>(args->counter0_)) + idx * (viewShape / 4);
            uint64_t tileCounter1 = args->counter1_;

            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Uniform(Element(DT_UINT64, args->key_), tileCounter0, Element(DT_UINT64, tileCounter1),
                               {validShape}, Element(DT_UINT16, args->rounds_), args->dtype_);
            Assemble(res, {offset}, outputs[0]);

            ++tileIdx;
        }
    }
}

class UniformOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<UniformOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestUniform, UniformOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<UniformOpMetaData>({UniformOperationExeFunc}, "Uniform")));

TEST_P(UniformOperationTest, TestUniform)
{
    auto testCase = CreateTestCaseDesc<UniformOpMetaData>(GetParam(), nullptr);
    nlohmann::json test_data = GetParam().test_data_;

    auto viewShape = GetViewShape(test_data);

    auto tileShape = GetTileShape(test_data);

    uint16_t rounds = GetValueByName<int>(test_data, "rounds");

    uint64_t key = GetValueByName<uint64_t>(test_data, "key");

    uint64_t counter0 = 0;
    uint64_t counter1 = GetValueByName<uint64_t>(test_data, "counter_1");

    DataType dtype = GetOutputDtype(test_data);

    auto args = UniformOpFuncArgs(key, counter0, counter1, viewShape, tileShape, rounds, dtype);
    testCase.args = &args;
    TestExecutor::runTest(testCase);
}
} // namespace
