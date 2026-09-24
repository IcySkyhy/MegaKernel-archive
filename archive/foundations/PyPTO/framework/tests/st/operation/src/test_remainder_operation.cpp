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
 * \file test_remainder_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
const unsigned IDX_DIM0 = 0;
const unsigned IDX_DIM1 = 1;
const unsigned IDX_DIM2 = 2;
const unsigned IDX_DIM3 = 3;
const unsigned IDX_DIM4 = 4;
struct RemainderOpFuncArgs : public OpFuncArgs {
    RemainderOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct RemainderOpMetaData {
    explicit RemainderOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static Tensor GetBrcViewTensor(const Tensor& input, const Shape& outputShape, const Shape& viewShape,
                               const std::vector<SymbolicScalar>& validShape,
                               const std::vector<SymbolicScalar>& dynOffsets)
{
    Shape tmpViewShape = viewShape;
    std::vector<SymbolicScalar> tmpValidShape = validShape;
    std::vector<SymbolicScalar> tmpDynOffsets = dynOffsets;
    // 输入为 [1,n],[m,1]而输出为[m,n]时，需要将广播维度的viewShape和validShape设置为1，dynOffsets设置为0
    for (size_t i = 0; i < outputShape.size(); ++i) {
        if (input.GetShape()[i] == 1) {
            tmpViewShape[i] = 1;
            tmpValidShape[i] = 1;
            tmpDynOffsets[i] = 0;
        }
    }
    return View(input, tmpViewShape, tmpValidShape, tmpDynOffsets);
}

static void RemainderOperationExeFunc1Dim(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto outputShape = outputs[0].GetShape();
        auto args = static_cast<const RemainderOpFuncArgs*>(opArgs);
        auto viewShape = args->viewShape_;

        const int loop = CeilDiv(outputShape[0], viewShape[0]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop))
        {
            std::vector<SymbolicScalar> dynOffsets = {bIdx * viewShape[0]};
            std::vector<SymbolicScalar> validShape = {std::min(outputShape[0] - bIdx * viewShape[0], viewShape[0])};
            Tensor tileTensor0 = GetBrcViewTensor(inputs[0], outputShape, viewShape, validShape, dynOffsets);
            Tensor tileTensor1 = GetBrcViewTensor(inputs[1], outputShape, viewShape, validShape, dynOffsets);
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Remainder(tileTensor0, tileTensor1);
            Assemble(res, dynOffsets, outputs[0]);
        }
    }
}

static void RemainderOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto outputShape = outputs[0].GetShape();
        auto args = static_cast<const RemainderOpFuncArgs*>(opArgs);
        auto viewShape = args->viewShape_;

        const int loop[] = {CeilDiv(outputShape[0], viewShape[0]), CeilDiv(outputShape[1], viewShape[1])};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                std::vector<SymbolicScalar> dynOffsets = {bIdx * viewShape[0], sIdx * viewShape[1]};
                std::vector<SymbolicScalar> validShape = {std::min(outputShape[0] - bIdx * viewShape[0], viewShape[0]),
                                                          std::min(outputShape[1] - sIdx * viewShape[1], viewShape[1])};
                Tensor tileTensor0 = GetBrcViewTensor(inputs[0], outputShape, viewShape, validShape, dynOffsets);
                Tensor tileTensor1 = GetBrcViewTensor(inputs[1], outputShape, viewShape, validShape, dynOffsets);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Remainder(tileTensor0, tileTensor1);
                Assemble(res, dynOffsets, outputs[0]);
            }
        }
    }
}

static void RemainderOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto outputShape = outputs[0].GetShape();
        auto args = static_cast<const RemainderOpFuncArgs*>(opArgs);
        auto viewShape = args->viewShape_;

        const int loop[] = {CeilDiv(outputShape[0], viewShape[0]), CeilDiv(outputShape[1], viewShape[1]),
                            CeilDiv(outputShape[2], viewShape[2])};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2]))
                {
                    std::vector<SymbolicScalar> dynOffsets = {bIdx * viewShape[0], sIdx * viewShape[1],
                                                              nIdx * viewShape[2]};
                    std::vector<SymbolicScalar> validShape = {
                        std::min(outputShape[0] - bIdx * viewShape[0], viewShape[0]),
                        std::min(outputShape[1] - sIdx * viewShape[1], viewShape[1]),
                        std::min(outputShape[2] - nIdx * viewShape[2], viewShape[2])};
                    Tensor tileTensor0 = GetBrcViewTensor(inputs[0], outputShape, viewShape, validShape, dynOffsets);
                    Tensor tileTensor1 = GetBrcViewTensor(inputs[1], outputShape, viewShape, validShape, dynOffsets);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Remainder(tileTensor0, tileTensor1);
                    Assemble(res, dynOffsets, outputs[0]);
                }
            }
        }
    }
}

static void RemainderOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto outputShape = outputs[0].GetShape();
        auto args = static_cast<const RemainderOpFuncArgs*>(opArgs);
        auto viewShape = args->viewShape_;

        const int loop[] = {CeilDiv(outputShape[0], viewShape[0]), CeilDiv(outputShape[1], viewShape[1]),
                            CeilDiv(outputShape[2], viewShape[2]), CeilDiv(outputShape[3], viewShape[3])};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2]))
                {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(loop[IDX_DIM3]))
                    {
                        std::vector<SymbolicScalar> dynOffsets = {bIdx * viewShape[0], sIdx * viewShape[1],
                                                                  nIdx * viewShape[2], qIdx * viewShape[3]};
                        std::vector<SymbolicScalar> validShape = {
                            std::min(outputShape[0] - bIdx * viewShape[0], viewShape[0]),
                            std::min(outputShape[1] - sIdx * viewShape[1], viewShape[1]),
                            std::min(outputShape[2] - nIdx * viewShape[2], viewShape[2]),
                            std::min(outputShape[3] - qIdx * viewShape[3], viewShape[3])};
                        Tensor tileTensor0 = GetBrcViewTensor(inputs[0], outputShape, viewShape, validShape,
                                                              dynOffsets);
                        Tensor tileTensor1 = GetBrcViewTensor(inputs[1], outputShape, viewShape, validShape,
                                                              dynOffsets);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Remainder(tileTensor0, tileTensor1);
                        Assemble(res, dynOffsets, outputs[0]);
                    }
                }
            }
        }
    }
}

static void RemainderOperationExeFunc5Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto outputShape = outputs[0].GetShape();
        auto args = static_cast<const RemainderOpFuncArgs*>(opArgs);
        auto viewShape = args->viewShape_;

        const int loop[] = {CeilDiv(outputShape[0], viewShape[0]), CeilDiv(outputShape[1], viewShape[1]),
                            CeilDiv(outputShape[2], viewShape[2]), CeilDiv(outputShape[3], viewShape[3]),
                            CeilDiv(outputShape[4], viewShape[4])};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2]))
                {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(loop[IDX_DIM3]))
                    {
                        LOOP("LOOP_L4_rIdx", FunctionType::DYNAMIC_LOOP, rIdx, LoopRange(loop[IDX_DIM4]))
                        {
                            std::vector<SymbolicScalar> dynOffsets = {bIdx * viewShape[0], sIdx * viewShape[1],
                                                                      nIdx * viewShape[2], qIdx * viewShape[3],
                                                                      rIdx * viewShape[4]};
                            std::vector<SymbolicScalar> validShape = {
                                std::min(outputShape[0] - bIdx * viewShape[0], viewShape[0]),
                                std::min(outputShape[1] - sIdx * viewShape[1], viewShape[1]),
                                std::min(outputShape[2] - nIdx * viewShape[2], viewShape[2]),
                                std::min(outputShape[3] - qIdx * viewShape[3], viewShape[3]),
                                std::min(outputShape[4] - rIdx * viewShape[4], viewShape[4])};
                            Tensor tileTensor0 = GetBrcViewTensor(inputs[0], outputShape, viewShape, validShape,
                                                                  dynOffsets);
                            Tensor tileTensor1 = GetBrcViewTensor(inputs[1], outputShape, viewShape, validShape,
                                                                  dynOffsets);
                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Remainder(tileTensor0, tileTensor1);
                            Assemble(res, dynOffsets, outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

class RemainderOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<RemainderOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestRemainder, RemainderOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<RemainderOpMetaData>(
                             {RemainderOperationExeFunc1Dim, RemainderOperationExeFunc2Dims,
                              RemainderOperationExeFunc3Dims, RemainderOperationExeFunc4Dims,
                              RemainderOperationExeFunc5Dims},
                             "Remainder")));

TEST_P(RemainderOperationTest, TestRemainder)
{
    auto test_data = GetParam().test_data_;
    auto args = RemainderOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<RemainderOpMetaData>(GetParam(), &args);
    std::vector<OpFunc> opFuncs = {RemainderOperationExeFunc1Dim, RemainderOperationExeFunc2Dims,
                                   RemainderOperationExeFunc3Dims, RemainderOperationExeFunc4Dims,
                                   RemainderOperationExeFunc5Dims};
    testCase.opFunc = opFuncs[GetViewShape(test_data).size() - 1];
    TestExecutor::runTest(testCase);
}
} // namespace
