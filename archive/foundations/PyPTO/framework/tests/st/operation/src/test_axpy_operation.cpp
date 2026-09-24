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
 * \file test_axpy_operation.cpp
 * \brief Test for AXPY operation (y = alpha * x + y)
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct AxpyOpFuncArgs : public OpFuncArgs {
    AxpyOpFuncArgs(float alpha, const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : alpha_(alpha), viewShape_(viewShape), tileShape_(tileShape)
    {
        this->inplaceInfo[0] = 0; // 输出第0个tensor和输入第0个tensor（y）复用
    }

    float alpha_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct AxpyOpMetaData {
    explicit AxpyOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

void UpdateInputBrcViewShape(std::vector<int64_t>& inputBrcViewShape, const std::vector<SymbolicScalar>& inputsShape,
                             const std::vector<SymbolicScalar>& outputsShape)
{
    for (size_t i = 0; i < inputsShape.size(); i++) {
        if (inputsShape[i] == 1 && outputsShape[i] != 1) {
            inputBrcViewShape[i] = 1;
        }
    }
}

void UpdateInputBrcVaildShape(std::vector<SymbolicScalar>& inputValidShape,
                              const std::vector<SymbolicScalar>& inputsShape,
                              const std::vector<SymbolicScalar>& outputsShape)
{
    for (size_t i = 0; i < inputsShape.size(); i++) {
        if (inputsShape[i] == 1 && outputsShape[i] != 1) {
            inputValidShape[i] = 1;
        }
    }
}

void UpdateOffset(std::vector<SymbolicScalar>& offset, const std::vector<SymbolicScalar>& inputsShape,
                  const std::vector<SymbolicScalar>& outputsShape)
{
    for (size_t i = 0; i < inputsShape.size(); i++) {
        if (inputsShape[i] == 1 && outputsShape[i] != 1) {
            offset[i] = 0;
        }
    }
}

static void AxpyOperationExeFunc1Dim(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> yInputsShape = {inputs[0].GetShape()[0]};
        std::vector<SymbolicScalar> xInputsShape = {inputs[1].GetShape()[0]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0]};
        auto args = static_cast<const AxpyOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0]};
        std::vector<int64_t> yInputViewShape = viewShape;
        std::vector<int64_t> xInputViewShape = viewShape;
        UpdateInputBrcViewShape(yInputViewShape, yInputsShape, outputsShape);
        UpdateInputBrcViewShape(xInputViewShape, xInputsShape, outputsShape);

        const int sloop = CeilDiv(outputsShape[0], viewShape[0]);
        LOOP("LOOP_L0_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
        {
            std::vector<SymbolicScalar> yInputValidShape = {
                std::min(yInputsShape[0] - sIdx * yInputViewShape[0], yInputViewShape[0])};
            std::vector<SymbolicScalar> xInputValidShape = {
                std::min(xInputsShape[0] - sIdx * xInputViewShape[0], xInputViewShape[0])};
            std::vector<SymbolicScalar> yOffset = {sIdx * yInputViewShape[0]};
            std::vector<SymbolicScalar> xOffset = {sIdx * xInputViewShape[0]};

            UpdateInputBrcVaildShape(yInputValidShape, yInputsShape, outputsShape);
            UpdateInputBrcVaildShape(xInputValidShape, xInputsShape, outputsShape);
            UpdateOffset(yOffset, yInputsShape, outputsShape);
            UpdateOffset(xOffset, xInputsShape, outputsShape);
            Tensor yTile = View(inputs[0], yInputViewShape, yInputValidShape, yOffset);
            Tensor xTile = View(inputs[1], xInputViewShape, xInputValidShape, xOffset);
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Axpy(yTile, xTile, args->alpha_);
            Assemble(res, {sIdx * viewShape[0]}, outputs[0]);
        }
    }
}

static void AxpyOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                      const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> yInputsShape = {inputs[0].GetShape()[0], inputs[0].GetShape()[1]};
        std::vector<SymbolicScalar> xInputsShape = {inputs[1].GetShape()[0], inputs[1].GetShape()[1]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0], outputs[0].GetShape()[1]};
        auto args = static_cast<const AxpyOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1]};
        std::vector<int64_t> yInputViewShape = viewShape;
        std::vector<int64_t> xInputViewShape = viewShape;
        UpdateInputBrcViewShape(yInputViewShape, yInputsShape, outputsShape);
        UpdateInputBrcViewShape(xInputViewShape, xInputsShape, outputsShape);

        const int bloop = CeilDiv(outputsShape[0], viewShape[0]);
        const int sloop = CeilDiv(outputsShape[1], viewShape[1]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                std::vector<SymbolicScalar> yInputValidShape = {
                    std::min(yInputsShape[0] - bIdx * yInputViewShape[0], yInputViewShape[0]),
                    std::min(yInputsShape[1] - sIdx * yInputViewShape[1], yInputViewShape[1])};
                std::vector<SymbolicScalar> xInputValidShape = {
                    std::min(xInputsShape[0] - bIdx * xInputViewShape[0], xInputViewShape[0]),
                    std::min(xInputsShape[1] - sIdx * xInputViewShape[1], xInputViewShape[1])};
                std::vector<SymbolicScalar> yOffset = {bIdx * yInputViewShape[0], sIdx * yInputViewShape[1]};
                std::vector<SymbolicScalar> xOffset = {bIdx * xInputViewShape[0], sIdx * xInputViewShape[1]};

                UpdateInputBrcVaildShape(yInputValidShape, yInputsShape, outputsShape);
                UpdateInputBrcVaildShape(xInputValidShape, xInputsShape, outputsShape);
                UpdateOffset(yOffset, yInputsShape, outputsShape);
                UpdateOffset(xOffset, xInputsShape, outputsShape);
                Tensor yTile = View(inputs[0], yInputViewShape, yInputValidShape, yOffset);
                Tensor xTile = View(inputs[1], xInputViewShape, xInputValidShape, xOffset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Axpy(yTile, xTile, args->alpha_);
                Assemble(res, {bIdx * viewShape[0], sIdx * viewShape[1]}, outputs[0]);
            }
        }
    }
}

static void AxpyOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                      const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> yInputsShape = {inputs[0].GetShape()[0], inputs[0].GetShape()[1],
                                                    inputs[0].GetShape()[2]};
        std::vector<SymbolicScalar> xInputsShape = {inputs[1].GetShape()[0], inputs[1].GetShape()[1],
                                                    inputs[1].GetShape()[2]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0], outputs[0].GetShape()[1],
                                                    outputs[0].GetShape()[2]};
        auto args = static_cast<const AxpyOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2]};
        std::vector<int64_t> yInputViewShape = viewShape;
        std::vector<int64_t> xInputViewShape = viewShape;
        UpdateInputBrcViewShape(yInputViewShape, yInputsShape, outputsShape);
        UpdateInputBrcViewShape(xInputViewShape, xInputsShape, outputsShape);

        const int bloop = CeilDiv(outputsShape[0], viewShape[0]);
        const int sloop = CeilDiv(outputsShape[1], viewShape[1]);
        const int nloop = CeilDiv(outputsShape[2], viewShape[2]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    std::vector<SymbolicScalar> yInputValidShape = {
                        std::min(yInputsShape[0] - bIdx * yInputViewShape[0], yInputViewShape[0]),
                        std::min(yInputsShape[1] - sIdx * yInputViewShape[1], yInputViewShape[1]),
                        std::min(yInputsShape[2] - nIdx * yInputViewShape[2], yInputViewShape[2])};
                    std::vector<SymbolicScalar> xInputValidShape = {
                        std::min(xInputsShape[0] - bIdx * xInputViewShape[0], xInputViewShape[0]),
                        std::min(xInputsShape[1] - sIdx * xInputViewShape[1], xInputViewShape[1]),
                        std::min(xInputsShape[2] - nIdx * xInputViewShape[2], xInputViewShape[2])};
                    std::vector<SymbolicScalar> yOffset = {bIdx * yInputViewShape[0], sIdx * yInputViewShape[1],
                                                           nIdx * yInputViewShape[2]};
                    std::vector<SymbolicScalar> xOffset = {bIdx * xInputViewShape[0], sIdx * xInputViewShape[1],
                                                           nIdx * xInputViewShape[2]};

                    UpdateInputBrcVaildShape(yInputValidShape, yInputsShape, outputsShape);
                    UpdateInputBrcVaildShape(xInputValidShape, xInputsShape, outputsShape);
                    UpdateOffset(yOffset, yInputsShape, outputsShape);
                    UpdateOffset(xOffset, xInputsShape, outputsShape);
                    Tensor yTile = View(inputs[0], yInputViewShape, yInputValidShape, yOffset);
                    Tensor xTile = View(inputs[1], xInputViewShape, xInputValidShape, xOffset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Axpy(yTile, xTile, args->alpha_);
                    Assemble(res, {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]}, outputs[0]);
                }
            }
        }
    }
}

static void AxpyOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                      const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> yInputsShape = {inputs[0].GetShape()[0], inputs[0].GetShape()[1],
                                                    inputs[0].GetShape()[2], inputs[0].GetShape()[3]};
        std::vector<SymbolicScalar> xInputsShape = {inputs[1].GetShape()[0], inputs[1].GetShape()[1],
                                                    inputs[1].GetShape()[2], inputs[1].GetShape()[3]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0], outputs[0].GetShape()[1],
                                                    outputs[0].GetShape()[2], outputs[0].GetShape()[3]};
        auto args = static_cast<const AxpyOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2],
                                          args->viewShape_[3]};
        std::vector<int64_t> yInputViewShape = viewShape;
        std::vector<int64_t> xInputViewShape = viewShape;
        UpdateInputBrcViewShape(yInputViewShape, yInputsShape, outputsShape);
        UpdateInputBrcViewShape(xInputViewShape, xInputsShape, outputsShape);

        const int bloop = CeilDiv(outputsShape[0], viewShape[0]);
        const int sloop = CeilDiv(outputsShape[1], viewShape[1]);
        const int nloop = CeilDiv(outputsShape[2], viewShape[2]);
        const int mloop = CeilDiv(outputsShape[3], viewShape[3]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                    {
                        std::vector<SymbolicScalar> yInputValidShape = {
                            std::min(yInputsShape[0] - bIdx * yInputViewShape[0], yInputViewShape[0]),
                            std::min(yInputsShape[1] - sIdx * yInputViewShape[1], yInputViewShape[1]),
                            std::min(yInputsShape[2] - nIdx * yInputViewShape[2], yInputViewShape[2]),
                            std::min(yInputsShape[3] - mIdx * yInputViewShape[3], yInputViewShape[3])};
                        std::vector<SymbolicScalar> xInputValidShape = {
                            std::min(xInputsShape[0] - bIdx * xInputViewShape[0], xInputViewShape[0]),
                            std::min(xInputsShape[1] - sIdx * xInputViewShape[1], xInputViewShape[1]),
                            std::min(xInputsShape[2] - nIdx * xInputViewShape[2], xInputViewShape[2]),
                            std::min(xInputsShape[3] - mIdx * xInputViewShape[3], xInputViewShape[3])};
                        std::vector<SymbolicScalar> yOffset = {bIdx * yInputViewShape[0], sIdx * yInputViewShape[1],
                                                               nIdx * yInputViewShape[2], mIdx * yInputViewShape[3]};
                        std::vector<SymbolicScalar> xOffset = {bIdx * xInputViewShape[0], sIdx * xInputViewShape[1],
                                                               nIdx * xInputViewShape[2], mIdx * xInputViewShape[3]};

                        UpdateInputBrcVaildShape(yInputValidShape, yInputsShape, outputsShape);
                        UpdateInputBrcVaildShape(xInputValidShape, xInputsShape, outputsShape);
                        UpdateOffset(yOffset, yInputsShape, outputsShape);
                        UpdateOffset(xOffset, xInputsShape, outputsShape);
                        Tensor yTile = View(inputs[0], yInputViewShape, yInputValidShape, yOffset);
                        Tensor xTile = View(inputs[1], xInputViewShape, xInputValidShape, xOffset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Axpy(yTile, xTile, args->alpha_);
                        Assemble(res,
                                 {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2], mIdx * viewShape[3]},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

class AxpyOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<AxpyOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestAxpy, AxpyOperationTest,
    ::testing::ValuesIn(GetOpMetaData<AxpyOpMetaData>({AxpyOperationExeFunc1Dim, AxpyOperationExeFunc2Dims,
                                                       AxpyOperationExeFunc3Dims, AxpyOperationExeFunc4Dims},
                                                      "Axpy")));

TEST_P(AxpyOperationTest, TestAxpy)
{
    auto test_data = GetParam().test_data_;
    float alpha = GetValueByName<float>(test_data, "alpha");
    auto args = AxpyOpFuncArgs(alpha, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<AxpyOpMetaData>(GetParam(), &args);
    std::vector<OpFunc> opFuncs = {AxpyOperationExeFunc1Dim, AxpyOperationExeFunc2Dims, AxpyOperationExeFunc3Dims,
                                   AxpyOperationExeFunc4Dims};
    testCase.opFunc = opFuncs[GetViewShape(test_data).size() - 1];
    TestExecutor::runTest(testCase);
}
} // namespace
