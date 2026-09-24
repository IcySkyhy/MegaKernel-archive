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
 * \file test_expand_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct ExpandOpFuncArgs : public OpFuncArgs {
    ExpandOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct ExpandOpMetaData {
    explicit ExpandOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

void UpdateInputExpandViewShape(std::vector<int64_t>& inputViewShape, const std::vector<SymbolicScalar>& inputsShape,
                                const std::vector<SymbolicScalar>& outputsShape)
{
    for (size_t i = 0; i < inputsShape.size(); i++) {
        if (inputsShape[i] == 1 && outputsShape[i] != 1) {
            inputViewShape[i] = 1;
        }
    }
}

void UpdateInputExpandVaildShape(std::vector<SymbolicScalar>& inputValidShape,
                                 const std::vector<SymbolicScalar>& inputsShape,
                                 const std::vector<SymbolicScalar>& outputsShape)
{
    for (size_t i = 0; i < inputsShape.size(); i++) {
        if (inputsShape[i] == 1 && outputsShape[i] != 1) {
            inputValidShape[i] = 1;
        }
    }
}

void UpdateInputExpandOffset(std::vector<SymbolicScalar>& inputOffset, const std::vector<SymbolicScalar>& inputsShape,
                             const std::vector<SymbolicScalar>& outputsShape)
{
    for (size_t i = 0; i < inputsShape.size(); i++) {
        if (inputsShape[i] == 1 && outputsShape[i] != 1) {
            inputOffset[i] = 0;
        }
    }
}

static void ExpandOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> inputsShape = {inputs[0].GetShape()[0], inputs[0].GetShape()[1]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0], outputs[0].GetShape()[1]};
        auto args = static_cast<const ExpandOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1]};
        std::vector<int64_t> inputViewShape = viewShape;
        UpdateInputExpandViewShape(inputViewShape, inputsShape, outputsShape);
        std::vector<SymbolicScalar> inputValidShape(2, 0);
        std::vector<SymbolicScalar> inputOffset(2, 0);
        const int bloop = CeilDiv(outputsShape[0], viewShape[0]);
        const int sloop = CeilDiv(outputsShape[1], viewShape[1]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                inputValidShape = {std::min(inputsShape[0] - bIdx * inputViewShape[0], inputViewShape[0]),
                                   std::min(inputsShape[1] - sIdx * inputViewShape[1], inputViewShape[1])};
                inputOffset = {bIdx * inputViewShape[0], sIdx * inputViewShape[1]};
                UpdateInputExpandVaildShape(inputValidShape, inputsShape, outputsShape);
                UpdateInputExpandOffset(inputOffset, inputsShape, outputsShape);
                Tensor tileTensor0 = View(inputs[0], inputViewShape, inputValidShape, inputOffset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Expand(tileTensor0, viewShape,
                                  {std::min(outputsShape[0] - bIdx * viewShape[0], viewShape[0]),
                                   std::min(outputsShape[1] - sIdx * viewShape[1], viewShape[1])});
                Assemble(res, {bIdx * viewShape[0], sIdx * viewShape[1]}, outputs[0]);
            }
        }
    }
}

static void ExpandOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> inputsShape = {inputs[0].GetShape()[0], inputs[0].GetShape()[1],
                                                   inputs[0].GetShape()[2]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0], outputs[0].GetShape()[1],
                                                    outputs[0].GetShape()[2]};
        auto args = static_cast<const ExpandOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2]};
        std::vector<int64_t> inputViewShape = viewShape;
        UpdateInputExpandViewShape(inputViewShape, inputsShape, outputsShape);
        std::vector<SymbolicScalar> inputValidShape(3, 0);
        std::vector<SymbolicScalar> inputOffset(3, 0);
        const int bloop = CeilDiv(outputsShape[0], viewShape[0]);
        const int sloop = CeilDiv(outputsShape[1], viewShape[1]);
        const int nloop = CeilDiv(outputsShape[2], viewShape[2]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    inputValidShape = {std::min(inputsShape[0] - bIdx * inputViewShape[0], inputViewShape[0]),
                                       std::min(inputsShape[1] - sIdx * inputViewShape[1], inputViewShape[1]),
                                       std::min(inputsShape[2] - nIdx * inputViewShape[2], inputViewShape[2])};
                    inputOffset = {bIdx * inputViewShape[0], sIdx * inputViewShape[1], nIdx * inputViewShape[2]};
                    UpdateInputExpandVaildShape(inputValidShape, inputsShape, outputsShape);
                    UpdateInputExpandOffset(inputOffset, inputsShape, outputsShape);
                    Tensor tileTensor0 = View(inputs[0], inputViewShape, inputValidShape, inputOffset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Expand(tileTensor0, viewShape,
                                      {std::min(outputsShape[0] - bIdx * viewShape[0], viewShape[0]),
                                       std::min(outputsShape[1] - sIdx * viewShape[1], viewShape[1]),
                                       std::min(outputsShape[2] - nIdx * viewShape[2], viewShape[2])});
                    Assemble(res, {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]}, outputs[0]);
                }
            }
        }
    }
}

static void ExpandOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> inputsShape = {inputs[0].GetShape()[0], inputs[0].GetShape()[1],
                                                   inputs[0].GetShape()[2], inputs[0].GetShape()[3]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0], outputs[0].GetShape()[1],
                                                    outputs[0].GetShape()[2], outputs[0].GetShape()[3]};
        auto args = static_cast<const ExpandOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2],
                                          args->viewShape_[3]};
        std::vector<int64_t> inputViewShape = viewShape;
        UpdateInputExpandViewShape(inputViewShape, inputsShape, outputsShape);
        std::vector<SymbolicScalar> inputValidShape(4, 0);
        std::vector<SymbolicScalar> inputOffset(4, 0);
        const int bloop = CeilDiv(outputsShape[0], viewShape[0]);
        const int sloop = CeilDiv(outputsShape[1], viewShape[1]);
        const int nloop = CeilDiv(outputsShape[2], viewShape[2]);
        const int mloop = CeilDiv(outputsShape[3], viewShape[3]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                    {
                        inputValidShape = {std::min(inputsShape[0] - bIdx * inputViewShape[0], inputViewShape[0]),
                                           std::min(inputsShape[1] - sIdx * inputViewShape[1], inputViewShape[1]),
                                           std::min(inputsShape[2] - nIdx * inputViewShape[2], inputViewShape[2]),
                                           std::min(inputsShape[3] - mIdx * inputViewShape[3], inputViewShape[3])};
                        inputOffset = {bIdx * inputViewShape[0], sIdx * inputViewShape[1], nIdx * inputViewShape[2],
                                       mIdx * inputViewShape[3]};
                        UpdateInputExpandVaildShape(inputValidShape, inputsShape, outputsShape);
                        UpdateInputExpandOffset(inputOffset, inputsShape, outputsShape);
                        Tensor tileTensor0 = View(inputs[0], inputViewShape, inputValidShape, inputOffset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Expand(tileTensor0, viewShape,
                                          {std::min(outputsShape[0] - bIdx * viewShape[0], viewShape[0]),
                                           std::min(outputsShape[1] - sIdx * viewShape[1], viewShape[1]),
                                           std::min(outputsShape[2] - nIdx * viewShape[2], viewShape[2]),
                                           std::min(outputsShape[3] - mIdx * viewShape[3], viewShape[3])});
                        Assemble(res,
                                 {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2], mIdx * viewShape[3]},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

class ExpandOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ExpandOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestExpand, ExpandOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<ExpandOpMetaData>(
                             {ExpandOperationExeFunc2Dims, ExpandOperationExeFunc3Dims, ExpandOperationExeFunc4Dims},
                             "Expand")));

TEST_P(ExpandOperationTest, TestExpand)
{
    auto test_data = GetParam().test_data_;
    auto args = ExpandOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<ExpandOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
