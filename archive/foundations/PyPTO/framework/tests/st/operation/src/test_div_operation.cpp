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
 * \file test_div_operation.cpp
 * \brief
 */

#include <nlohmann/json.hpp>
#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct DivOpFuncArgs : public OpFuncArgs {
    DivOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct DivOpMetaData {
    explicit DivOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
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

static void DivOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> firstInputsShape = {inputs[0].GetShape()[0], inputs[0].GetShape()[1]};
        std::vector<SymbolicScalar> secondInputsShape = {inputs[1].GetShape()[0], inputs[1].GetShape()[1]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0], outputs[0].GetShape()[1]};
        auto args = static_cast<const DivOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1]};
        std::vector<int64_t> firstInputViewShape = viewShape;
        std::vector<int64_t> secondInputViewShape = viewShape;
        UpdateInputBrcViewShape(firstInputViewShape, firstInputsShape, outputsShape);
        UpdateInputBrcViewShape(secondInputViewShape, secondInputsShape, outputsShape);

        const int bloop = CeilDiv(outputsShape[0], viewShape[0]);
        const int sloop = CeilDiv(outputsShape[1], viewShape[1]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                std::vector<SymbolicScalar> firstInputValidShape = {
                    std::min(firstInputsShape[0] - bIdx * firstInputViewShape[0], firstInputViewShape[0]),
                    std::min(firstInputsShape[1] - sIdx * firstInputViewShape[1], firstInputViewShape[1])};
                std::vector<SymbolicScalar> secondInputValidShape = {
                    std::min(secondInputsShape[0] - bIdx * secondInputViewShape[0], secondInputViewShape[0]),
                    std::min(secondInputsShape[1] - sIdx * secondInputViewShape[1], secondInputViewShape[1])};
                std::vector<SymbolicScalar> firstOffset = {bIdx * firstInputViewShape[0],
                                                           sIdx * firstInputViewShape[1]};
                std::vector<SymbolicScalar> secondOffset = {bIdx * secondInputViewShape[0],
                                                            sIdx * secondInputViewShape[1]};

                UpdateInputBrcVaildShape(firstInputValidShape, firstInputsShape, outputsShape);
                UpdateInputBrcVaildShape(secondInputValidShape, secondInputsShape, outputsShape);
                UpdateOffset(firstOffset, firstInputsShape, outputsShape);
                UpdateOffset(secondOffset, secondInputsShape, outputsShape);
                Tensor tileTensor0 = View(inputs[0], firstInputViewShape, firstInputValidShape, firstOffset);
                Tensor tileTensor1 = View(inputs[1], secondInputViewShape, secondInputValidShape, secondOffset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Div(tileTensor0, tileTensor1);
                Assemble(res, {bIdx * viewShape[0], sIdx * viewShape[1]}, outputs[0]);
            }
        }
    }
}

static void DivOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> firstInputsShape = {inputs[0].GetShape()[0], inputs[0].GetShape()[1],
                                                        inputs[0].GetShape()[2]};
        std::vector<SymbolicScalar> secondInputsShape = {inputs[1].GetShape()[0], inputs[1].GetShape()[1],
                                                         inputs[1].GetShape()[2]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0], outputs[0].GetShape()[1],
                                                    outputs[0].GetShape()[2]};
        auto args = static_cast<const DivOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2]};
        std::vector<int64_t> firstInputViewShape = viewShape;
        std::vector<int64_t> secondInputViewShape = viewShape;
        UpdateInputBrcViewShape(firstInputViewShape, firstInputsShape, outputsShape);
        UpdateInputBrcViewShape(secondInputViewShape, secondInputsShape, outputsShape);

        const int bloop = CeilDiv(outputsShape[0], viewShape[0]);
        const int sloop = CeilDiv(outputsShape[1], viewShape[1]);
        const int nloop = CeilDiv(outputsShape[2], viewShape[2]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    std::vector<SymbolicScalar> firstInputValidShape = {
                        std::min(firstInputsShape[0] - bIdx * firstInputViewShape[0], firstInputViewShape[0]),
                        std::min(firstInputsShape[1] - sIdx * firstInputViewShape[1], firstInputViewShape[1]),
                        std::min(firstInputsShape[2] - nIdx * firstInputViewShape[2], firstInputViewShape[2])};
                    std::vector<SymbolicScalar> secondInputValidShape = {
                        std::min(secondInputsShape[0] - bIdx * secondInputViewShape[0], secondInputViewShape[0]),
                        std::min(secondInputsShape[1] - sIdx * secondInputViewShape[1], secondInputViewShape[1]),
                        std::min(secondInputsShape[2] - nIdx * secondInputViewShape[2], secondInputViewShape[2])};
                    std::vector<SymbolicScalar> firstOffset = {
                        bIdx * firstInputViewShape[0], sIdx * firstInputViewShape[1], nIdx * firstInputViewShape[2]};
                    std::vector<SymbolicScalar> secondOffset = {
                        bIdx * secondInputViewShape[0], sIdx * secondInputViewShape[1], nIdx * secondInputViewShape[2]};

                    UpdateInputBrcVaildShape(firstInputValidShape, firstInputsShape, outputsShape);
                    UpdateInputBrcVaildShape(secondInputValidShape, secondInputsShape, outputsShape);
                    UpdateOffset(firstOffset, firstInputsShape, outputsShape);
                    UpdateOffset(secondOffset, secondInputsShape, outputsShape);
                    Tensor tileTensor0 = View(inputs[0], firstInputViewShape, firstInputValidShape, firstOffset);
                    Tensor tileTensor1 = View(inputs[1], secondInputViewShape, secondInputValidShape, secondOffset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Div(tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]}, outputs[0]);
                }
            }
        }
    }
}

static void DivOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> firstInputsShape = {inputs[0].GetShape()[0], inputs[0].GetShape()[1],
                                                        inputs[0].GetShape()[2], inputs[0].GetShape()[3]};
        std::vector<SymbolicScalar> secondInputsShape = {inputs[1].GetShape()[0], inputs[1].GetShape()[1],
                                                         inputs[1].GetShape()[2], inputs[1].GetShape()[3]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0], outputs[0].GetShape()[1],
                                                    outputs[0].GetShape()[2], outputs[0].GetShape()[3]};
        auto args = static_cast<const DivOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2],
                                          args->viewShape_[3]};
        std::vector<int64_t> firstInputViewShape = viewShape;
        std::vector<int64_t> secondInputViewShape = viewShape;
        UpdateInputBrcViewShape(firstInputViewShape, firstInputsShape, outputsShape);
        UpdateInputBrcViewShape(secondInputViewShape, secondInputsShape, outputsShape);

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
                        std::vector<SymbolicScalar> firstInputValidShape = {
                            std::min(firstInputsShape[0] - bIdx * firstInputViewShape[0], firstInputViewShape[0]),
                            std::min(firstInputsShape[1] - sIdx * firstInputViewShape[1], firstInputViewShape[1]),
                            std::min(firstInputsShape[2] - nIdx * firstInputViewShape[2], firstInputViewShape[2]),
                            std::min(firstInputsShape[3] - mIdx * firstInputViewShape[3], firstInputViewShape[3])};
                        std::vector<SymbolicScalar> secondInputValidShape = {
                            std::min(secondInputsShape[0] - bIdx * secondInputViewShape[0], secondInputViewShape[0]),
                            std::min(secondInputsShape[1] - sIdx * secondInputViewShape[1], secondInputViewShape[1]),
                            std::min(secondInputsShape[2] - nIdx * secondInputViewShape[2], secondInputViewShape[2]),
                            std::min(secondInputsShape[3] - mIdx * secondInputViewShape[3], secondInputViewShape[3])};
                        std::vector<SymbolicScalar> firstOffset = {
                            bIdx * firstInputViewShape[0], sIdx * firstInputViewShape[1], nIdx * firstInputViewShape[2],
                            mIdx * firstInputViewShape[3]};
                        std::vector<SymbolicScalar> secondOffset = {
                            bIdx * secondInputViewShape[0], sIdx * secondInputViewShape[1],
                            nIdx * secondInputViewShape[2], mIdx * secondInputViewShape[3]};

                        UpdateInputBrcVaildShape(firstInputValidShape, firstInputsShape, outputsShape);
                        UpdateInputBrcVaildShape(secondInputValidShape, secondInputsShape, outputsShape);
                        UpdateOffset(firstOffset, firstInputsShape, outputsShape);
                        UpdateOffset(secondOffset, secondInputsShape, outputsShape);
                        Tensor tileTensor0 = View(inputs[0], firstInputViewShape, firstInputValidShape, firstOffset);
                        Tensor tileTensor1 = View(inputs[1], secondInputViewShape, secondInputValidShape, secondOffset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Div(tileTensor0, tileTensor1);
                        Assemble(res,
                                 {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2], mIdx * viewShape[3]},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

static void DivOperationExeFunc5Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> firstInputsShape = {inputs[0].GetShape()[0], inputs[0].GetShape()[1],
                                                        inputs[0].GetShape()[2], inputs[0].GetShape()[3],
                                                        inputs[0].GetShape()[4]};
        std::vector<SymbolicScalar> secondInputsShape = {inputs[1].GetShape()[0], inputs[1].GetShape()[1],
                                                         inputs[1].GetShape()[2], inputs[1].GetShape()[3],
                                                         inputs[1].GetShape()[4]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0], outputs[0].GetShape()[1],
                                                    outputs[0].GetShape()[2], outputs[0].GetShape()[3],
                                                    outputs[0].GetShape()[4]};
        auto args = static_cast<const DivOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2],
                                          args->viewShape_[3], args->viewShape_[4]};
        std::vector<int64_t> firstInputViewShape = viewShape;
        std::vector<int64_t> secondInputViewShape = viewShape;
        UpdateInputBrcViewShape(firstInputViewShape, firstInputsShape, outputsShape);
        UpdateInputBrcViewShape(secondInputViewShape, secondInputsShape, outputsShape);

        const int bloop = CeilDiv(outputsShape[0], viewShape[0]);
        const int sloop = CeilDiv(outputsShape[1], viewShape[1]);
        const int nloop = CeilDiv(outputsShape[2], viewShape[2]);
        const int mloop = CeilDiv(outputsShape[3], viewShape[3]);
        const int qloop = CeilDiv(outputsShape[4], viewShape[4]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                    {
                        LOOP("LOOP_L4_nIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(0, qloop, 1))
                        {
                            std::vector<SymbolicScalar> firstInputValidShape = {
                                std::min(firstInputsShape[0] - bIdx * firstInputViewShape[0], firstInputViewShape[0]),
                                std::min(firstInputsShape[1] - sIdx * firstInputViewShape[1], firstInputViewShape[1]),
                                std::min(firstInputsShape[2] - nIdx * firstInputViewShape[2], firstInputViewShape[2]),
                                std::min(firstInputsShape[3] - mIdx * firstInputViewShape[3], firstInputViewShape[3]),
                                std::min(firstInputsShape[4] - qIdx * firstInputViewShape[4], firstInputViewShape[4])};
                            std::vector<SymbolicScalar> secondInputValidShape = {
                                std::min(secondInputsShape[0] - bIdx * secondInputViewShape[0],
                                         secondInputViewShape[0]),
                                std::min(secondInputsShape[1] - sIdx * secondInputViewShape[1],
                                         secondInputViewShape[1]),
                                std::min(secondInputsShape[2] - nIdx * secondInputViewShape[2],
                                         secondInputViewShape[2]),
                                std::min(secondInputsShape[3] - mIdx * secondInputViewShape[3],
                                         secondInputViewShape[3]),
                                std::min(secondInputsShape[4] - qIdx * secondInputViewShape[4],
                                         secondInputViewShape[4])};
                            std::vector<SymbolicScalar> firstOffset = {
                                bIdx * firstInputViewShape[0], sIdx * firstInputViewShape[1],
                                nIdx * firstInputViewShape[2], mIdx * firstInputViewShape[3],
                                qIdx * firstInputViewShape[4]};
                            std::vector<SymbolicScalar> secondOffset = {
                                bIdx * secondInputViewShape[0], sIdx * secondInputViewShape[1],
                                nIdx * secondInputViewShape[2], mIdx * secondInputViewShape[3],
                                qIdx * secondInputViewShape[4]};

                            UpdateInputBrcVaildShape(firstInputValidShape, firstInputsShape, outputsShape);
                            UpdateInputBrcVaildShape(secondInputValidShape, secondInputsShape, outputsShape);
                            UpdateOffset(firstOffset, firstInputsShape, outputsShape);
                            UpdateOffset(secondOffset, secondInputsShape, outputsShape);
                            Tensor tileTensor0 = View(inputs[0], firstInputViewShape, firstInputValidShape,
                                                      firstOffset);
                            Tensor tileTensor1 = View(inputs[1], secondInputViewShape, secondInputValidShape,
                                                      secondOffset);
                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Div(tileTensor0, tileTensor1);
                            Assemble(res,
                                     {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2],
                                      mIdx * viewShape[3], qIdx * viewShape[4]},
                                     outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

class DivOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<DivOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestDiv, DivOperationTest,
    ::testing::ValuesIn(GetOpMetaData<DivOpMetaData>({DivOperationExeFunc2Dims, DivOperationExeFunc3Dims,
                                                      DivOperationExeFunc4Dims, DivOperationExeFunc5Dims},
                                                     "Div")));

TEST_P(DivOperationTest, TestDiv)
{
    auto test_data = GetParam().test_data_;
    auto args = DivOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<DivOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
