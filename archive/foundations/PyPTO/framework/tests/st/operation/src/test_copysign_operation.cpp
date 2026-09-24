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
 * \file test_copysign_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct CopySignOpFuncArgs : public OpFuncArgs {
    CopySignOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct CopySignOpMetaData {
    explicit CopySignOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

int BroadcastAxis(const std::vector<SymbolicScalar>& inputsShape, const std::vector<SymbolicScalar>& outputsShape)
{
    int brcAxis = -1;
    for (size_t i = 0; i < inputsShape.size(); i++) {
        if (inputsShape[i] != outputsShape[i]) {
            brcAxis = i;
        }
    }
    return brcAxis;
}

int BroadcastTensor(const std::vector<SymbolicScalar>& firstInputsShape,
                    const std::vector<SymbolicScalar>& secondInputsShape,
                    const std::vector<SymbolicScalar>& outputsShape)
{
    for (size_t i = 0; i < firstInputsShape.size(); i++) {
        if (firstInputsShape[i] != outputsShape[i]) {
            return 0;
        }
        if (secondInputsShape[i] != outputsShape[i]) {
            return 1;
        }
    }
    return -1;
}

void UpdateInputBrcViewShape(std::vector<int64_t>& inputBrcViewShape, const std::vector<SymbolicScalar>& inputsShape,
                             const std::vector<SymbolicScalar>& outputsShape)
{
    for (size_t i = 0; i < inputsShape.size(); i++) {
        if (inputsShape[i] != outputsShape[i]) {
            inputBrcViewShape[i] = inputsShape[i].Concrete();
        }
    }
}

static void CopySignOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        std::vector<SymbolicScalar> firstInputsShape = {inputs[0].GetShape()[0], inputs[0].GetShape()[1]};
        std::vector<SymbolicScalar> secondInputsShape = {inputs[1].GetShape()[0], inputs[1].GetShape()[1]};
        std::vector<SymbolicScalar> outputsShape = {outputs[0].GetShape()[0], outputs[0].GetShape()[1]};
        auto args = static_cast<const CopySignOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1]};
        std::vector<int64_t> firstInputViewShape = viewShape;
        std::vector<int64_t> secondInputViewShape = viewShape;
        std::vector<SymbolicScalar> firstInputValidShape(2, 0);
        std::vector<SymbolicScalar> secondInputValidShape(2, 0);
        std::vector<SymbolicScalar> firstOffset(2, 0);
        std::vector<SymbolicScalar> secondOffset(2, 0);
        int brcTensor = BroadcastTensor(firstInputsShape, secondInputsShape, outputsShape);
        int brcAxis = -1;
        if (brcTensor == 0) {
            brcAxis = BroadcastAxis(firstInputsShape, outputsShape);
            UpdateInputBrcViewShape(firstInputViewShape, firstInputsShape, outputsShape);
        } else if (brcTensor == 1) {
            brcAxis = BroadcastAxis(secondInputsShape, outputsShape);
            UpdateInputBrcViewShape(secondInputViewShape, secondInputsShape, outputsShape);
        }
        const int bloop = CeilDiv(outputsShape[0], viewShape[0]);
        const int sloop = CeilDiv(outputsShape[1], viewShape[1]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                firstInputValidShape = {
                    std::min(firstInputsShape[0] - bIdx * firstInputViewShape[0], firstInputViewShape[0]),
                    std::min(firstInputsShape[1] - sIdx * firstInputViewShape[1], firstInputViewShape[1])};
                secondInputValidShape = {
                    std::min(secondInputsShape[0] - bIdx * secondInputViewShape[0], secondInputViewShape[0]),
                    std::min(secondInputsShape[1] - sIdx * secondInputViewShape[1], secondInputViewShape[1])};
                firstOffset = {bIdx * firstInputViewShape[0], sIdx * firstInputViewShape[1]};
                secondOffset = {bIdx * secondInputViewShape[0], sIdx * secondInputViewShape[1]};

                if (brcTensor == 0 && brcAxis != -1) {
                    firstInputValidShape[brcAxis] = firstInputViewShape[brcAxis];
                    firstOffset[brcAxis] = 0;
                } else if (brcTensor == 1 && brcAxis != -1) {
                    secondInputValidShape[brcAxis] = secondInputViewShape[brcAxis];
                    secondOffset[brcAxis] = 0;
                }
                Tensor tileTensor0 = View(inputs[0], firstInputViewShape, firstInputValidShape, firstOffset);
                Tensor tileTensor1 = View(inputs[1], secondInputViewShape, secondInputValidShape, secondOffset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = CopySign(tileTensor0, tileTensor1);
                Assemble(res, {bIdx * viewShape[0], sIdx * viewShape[1]}, outputs[0]);
            }
        }
    }
}

static void CopySignOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
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
        auto args = static_cast<const CopySignOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2]};
        std::vector<int64_t> firstInputViewShape = viewShape;
        std::vector<int64_t> secondInputViewShape = viewShape;
        std::vector<SymbolicScalar> firstInputValidShape(3, 0);
        std::vector<SymbolicScalar> secondInputValidShape(3, 0);
        std::vector<SymbolicScalar> firstOffset(3, 0);
        std::vector<SymbolicScalar> secondOffset(3, 0);
        int brcTensor = BroadcastTensor(firstInputsShape, secondInputsShape, outputsShape);
        int brcAxis = -1;
        if (brcTensor == 0) {
            brcAxis = BroadcastAxis(firstInputsShape, outputsShape);
            UpdateInputBrcViewShape(firstInputViewShape, firstInputsShape, outputsShape);
        } else if (brcTensor == 1) {
            brcAxis = BroadcastAxis(secondInputsShape, outputsShape);
            UpdateInputBrcViewShape(secondInputViewShape, secondInputsShape, outputsShape);
        }
        const int bloop = CeilDiv(outputsShape[0], viewShape[0]);
        const int sloop = CeilDiv(outputsShape[1], viewShape[1]);
        const int nloop = CeilDiv(outputsShape[2], viewShape[2]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    firstInputValidShape = {
                        std::min(firstInputsShape[0] - bIdx * firstInputViewShape[0], firstInputViewShape[0]),
                        std::min(firstInputsShape[1] - sIdx * firstInputViewShape[1], firstInputViewShape[1]),
                        std::min(firstInputsShape[2] - nIdx * firstInputViewShape[2], firstInputViewShape[2])};
                    secondInputValidShape = {
                        std::min(secondInputsShape[0] - bIdx * secondInputViewShape[0], secondInputViewShape[0]),
                        std::min(secondInputsShape[1] - sIdx * secondInputViewShape[1], secondInputViewShape[1]),
                        std::min(secondInputsShape[2] - nIdx * secondInputViewShape[2], secondInputViewShape[2])};
                    firstOffset = {bIdx * firstInputViewShape[0], sIdx * firstInputViewShape[1],
                                   nIdx * firstInputViewShape[2]};
                    secondOffset = {bIdx * secondInputViewShape[0], sIdx * secondInputViewShape[1],
                                    nIdx * secondInputViewShape[2]};

                    if (brcTensor == 0 && brcAxis != -1) {
                        firstInputValidShape[brcAxis] = firstInputViewShape[brcAxis];
                        firstOffset[brcAxis] = 0;
                    } else if (brcTensor == 1 && brcAxis != -1) {
                        secondInputValidShape[brcAxis] = secondInputViewShape[brcAxis];
                        secondOffset[brcAxis] = 0;
                    }
                    Tensor tileTensor0 = View(inputs[0], firstInputViewShape, firstInputValidShape, firstOffset);
                    Tensor tileTensor1 = View(inputs[1], secondInputViewShape, secondInputValidShape, secondOffset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = CopySign(tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]}, outputs[0]);
                }
            }
        }
    }
}

static void CopySignOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
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
        auto args = static_cast<const CopySignOpFuncArgs*>(opArgs);
        std::vector<int64_t> viewShape = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2],
                                          args->viewShape_[3]};
        std::vector<int64_t> firstInputViewShape = viewShape;
        std::vector<int64_t> secondInputViewShape = viewShape;
        std::vector<SymbolicScalar> firstInputValidShape(4, 0);
        std::vector<SymbolicScalar> secondInputValidShape(4, 0);
        std::vector<SymbolicScalar> firstOffset(4, 0);
        std::vector<SymbolicScalar> secondOffset(4, 0);
        int brcTensor = BroadcastTensor(firstInputsShape, secondInputsShape, outputsShape);
        int brcAxis = -1;
        if (brcTensor == 0) {
            brcAxis = BroadcastAxis(firstInputsShape, outputsShape);
            UpdateInputBrcViewShape(firstInputViewShape, firstInputsShape, outputsShape);
        } else if (brcTensor == 1) {
            brcAxis = BroadcastAxis(secondInputsShape, outputsShape);
            UpdateInputBrcViewShape(secondInputViewShape, secondInputsShape, outputsShape);
        }
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
                        firstInputValidShape = {
                            std::min(firstInputsShape[0] - bIdx * firstInputViewShape[0], firstInputViewShape[0]),
                            std::min(firstInputsShape[1] - sIdx * firstInputViewShape[1], firstInputViewShape[1]),
                            std::min(firstInputsShape[2] - nIdx * firstInputViewShape[2], firstInputViewShape[2]),
                            std::min(firstInputsShape[3] - mIdx * firstInputViewShape[3], firstInputViewShape[3])};
                        secondInputValidShape = {
                            std::min(secondInputsShape[0] - bIdx * secondInputViewShape[0], secondInputViewShape[0]),
                            std::min(secondInputsShape[1] - sIdx * secondInputViewShape[1], secondInputViewShape[1]),
                            std::min(secondInputsShape[2] - nIdx * secondInputViewShape[2], secondInputViewShape[2]),
                            std::min(secondInputsShape[3] - mIdx * secondInputViewShape[3], secondInputViewShape[3])};
                        firstOffset = {bIdx * firstInputViewShape[0], sIdx * firstInputViewShape[1],
                                       nIdx * firstInputViewShape[2], mIdx * firstInputViewShape[3]};
                        secondOffset = {bIdx * secondInputViewShape[0], sIdx * secondInputViewShape[1],
                                        nIdx * secondInputViewShape[2], mIdx * secondInputViewShape[3]};

                        if (brcTensor == 0 && brcAxis != -1) {
                            firstInputValidShape[brcAxis] = firstInputViewShape[brcAxis];
                            firstOffset[brcAxis] = 0;
                        } else if (brcTensor == 1 && brcAxis != -1) {
                            secondInputValidShape[brcAxis] = secondInputViewShape[brcAxis];
                            secondOffset[brcAxis] = 0;
                        }
                        Tensor tileTensor0 = View(inputs[0], firstInputViewShape, firstInputValidShape, firstOffset);
                        Tensor tileTensor1 = View(inputs[1], secondInputViewShape, secondInputValidShape, secondOffset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = CopySign(tileTensor0, tileTensor1);
                        Assemble(res,
                                 {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2], mIdx * viewShape[3]},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

class CopySignOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<CopySignOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestCopySign, CopySignOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<CopySignOpMetaData>({CopySignOperationExeFunc2Dims,
                                                                                CopySignOperationExeFunc3Dims,
                                                                                CopySignOperationExeFunc4Dims},
                                                                               "CopySign")));

TEST_P(CopySignOperationTest, TestCopySign)
{
    auto test_data = GetParam().test_data_;
    auto args = CopySignOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<CopySignOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
