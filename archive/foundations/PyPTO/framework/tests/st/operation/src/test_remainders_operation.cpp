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
 * \file test_remainders_operation.cpp
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
struct RemainderSOpFuncArgs : public OpFuncArgs {
    RemainderSOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& tileShape, Element& scalar,
                         bool reverseOperand)
        : viewShape_(viewShape), tileShape_(tileShape), scalar_(scalar), reverseOperand_(reverseOperand)
    {}
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    Element scalar_;
    bool reverseOperand_;
};

struct RemainderSOpMetaData {
    explicit RemainderSOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void RemainderSOperationExeFunc1Dim(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    SymbolicScalar src_dim = outputs[0].GetShape()[0];
    auto args = static_cast<const RemainderSOpFuncArgs*>(opArgs);
    const std::vector<int64_t> viewShape = args->viewShape_;
    bool reverseOperand = args->reverseOperand_;
    const int loop = CeilDiv(src_dim, viewShape[0]);

    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, loop, 1))
        {
            std::vector<SymbolicScalar> dynOffsets = {bIdx * viewShape[0]};
            auto tileTensor = View(inputs[0], viewShape, {std::min(src_dim - bIdx * viewShape[0], viewShape[0])},
                                   dynOffsets);
            TileShape::Current().SetVecTile(args->tileShape_);
            Tensor res = reverseOperand ? Remainder(args->scalar_, tileTensor) : Remainder(tileTensor, args->scalar_);
            Assemble(res, dynOffsets, outputs[0]);
        }
    }
}

static void RemainderSOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    auto outputShape = outputs[0].GetShape();
    auto args = static_cast<const RemainderSOpFuncArgs*>(opArgs);
    const std::vector<int64_t> viewShape = args->viewShape_;
    bool reverseOperand = args->reverseOperand_;
    const int bloop = CeilDiv(outputShape[0], viewShape[0]);
    const int sloop = CeilDiv(outputShape[1], viewShape[1]);

    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                std::vector<SymbolicScalar> dynOffsets = {bIdx * viewShape[0], sIdx * viewShape[1]};
                auto tileTensor = View(inputs[0], viewShape,
                                       {std::min(outputShape[0] - bIdx * viewShape[0], viewShape[0]),
                                        std::min(outputShape[1] - sIdx * viewShape[1], viewShape[1])},
                                       dynOffsets);
                TileShape::Current().SetVecTile(args->tileShape_);
                Tensor res = reverseOperand ? Remainder(args->scalar_, tileTensor) :
                                              Remainder(tileTensor, args->scalar_);
                Assemble(res, dynOffsets, outputs[0]);
            }
        }
    }
}

static void RemainderSOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    auto outputShape = outputs[0].GetShape();
    auto args = static_cast<const RemainderSOpFuncArgs*>(opArgs);
    std::vector<int64_t> viewShape = args->viewShape_;
    bool reverseOperand = args->reverseOperand_;
    const int loop[] = {CeilDiv(outputShape[0], viewShape[0]), CeilDiv(outputShape[1], viewShape[1]),
                        CeilDiv(outputShape[2], viewShape[2])};

    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2]))
                {
                    std::vector<SymbolicScalar> dynOffsets = {bIdx * viewShape[0], sIdx * viewShape[1],
                                                              nIdx * viewShape[2]};
                    auto tileTensor = View(inputs[0], viewShape,
                                           {std::min(outputShape[0] - bIdx * viewShape[0], viewShape[0]),
                                            std::min(outputShape[1] - sIdx * viewShape[1], viewShape[1]),
                                            std::min(outputShape[2] - nIdx * viewShape[2], viewShape[2])},
                                           dynOffsets);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    Tensor res = reverseOperand ? Remainder(args->scalar_, tileTensor) :
                                                  Remainder(tileTensor, args->scalar_);
                    Assemble(res, dynOffsets, outputs[0]);
                }
            }
        }
    }
}

static void RemainderSOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    auto outputShape = outputs[0].GetShape();
    auto args = static_cast<const RemainderSOpFuncArgs*>(opArgs);
    std::vector<int64_t> viewShape = args->viewShape_;
    bool reverseOperand = args->reverseOperand_;
    const int loop[] = {CeilDiv(outputShape[0], viewShape[0]), CeilDiv(outputShape[1], viewShape[1]),
                        CeilDiv(outputShape[2], viewShape[2]), CeilDiv(outputShape[3], viewShape[3])};
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
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
                        auto tileTensor = View(inputs[0], viewShape,
                                               {std::min(outputShape[0] - bIdx * viewShape[0], viewShape[0]),
                                                std::min(outputShape[1] - sIdx * viewShape[1], viewShape[1]),
                                                std::min(outputShape[2] - nIdx * viewShape[2], viewShape[2]),
                                                std::min(outputShape[3] - qIdx * viewShape[3], viewShape[3])},
                                               dynOffsets);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        Tensor res = reverseOperand ? Remainder(args->scalar_, tileTensor) :
                                                      Remainder(tileTensor, args->scalar_);
                        Assemble(res, dynOffsets, outputs[0]);
                    }
                }
            }
        }
    }
}

static void RemainderSOperationExeFunc5Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    auto outputShape = outputs[0].GetShape();
    auto args = static_cast<const RemainderSOpFuncArgs*>(opArgs);
    std::vector<int64_t> viewShape = args->viewShape_;
    bool reverseOperand = args->reverseOperand_;
    const int loop[] = {CeilDiv(outputShape[0], viewShape[0]), CeilDiv(outputShape[1], viewShape[1]),
                        CeilDiv(outputShape[2], viewShape[2]), CeilDiv(outputShape[3], viewShape[3]),
                        CeilDiv(outputShape[4], viewShape[4])};
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
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
                            auto tileTensor = View(inputs[0], viewShape,
                                                   {std::min(outputShape[0] - bIdx * viewShape[0], viewShape[0]),
                                                    std::min(outputShape[1] - sIdx * viewShape[1], viewShape[1]),
                                                    std::min(outputShape[2] - nIdx * viewShape[2], viewShape[2]),
                                                    std::min(outputShape[3] - qIdx * viewShape[3], viewShape[3]),
                                                    std::min(outputShape[4] - rIdx * viewShape[4], viewShape[4])},
                                                   dynOffsets);
                            TileShape::Current().SetVecTile(args->tileShape_);
                            Tensor res = reverseOperand ? Remainder(args->scalar_, tileTensor) :
                                                          Remainder(tileTensor, args->scalar_);
                            Assemble(res, dynOffsets, outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

class RemainderRSOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<RemainderSOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestRemainderRS, RemainderRSOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<RemainderSOpMetaData>(
                             {RemainderSOperationExeFunc1Dim, RemainderSOperationExeFunc2Dims,
                              RemainderSOperationExeFunc3Dims, RemainderSOperationExeFunc4Dims,
                              RemainderSOperationExeFunc5Dims},
                             "RemainderRS")));

class RemainderSOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<RemainderSOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestRemainderS, RemainderSOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<RemainderSOpMetaData>(
                             {RemainderSOperationExeFunc1Dim, RemainderSOperationExeFunc2Dims,
                              RemainderSOperationExeFunc3Dims, RemainderSOperationExeFunc4Dims,
                              RemainderSOperationExeFunc5Dims},
                             "RemainderS")));

TEST_P(RemainderRSOperationTest, TestRemainderRS)
{
    auto test_data = GetParam().test_data_;
    auto dtype = GetDataType(GetValueByName<std::string>(test_data, "scalar_type"));
    Element scalar(dtype, GetValueByName<float>(test_data, "scalar"));
    auto args = RemainderSOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), scalar, true);
    auto testCase = CreateTestCaseDesc<RemainderSOpMetaData>(GetParam(), &args);
    std::vector<OpFunc> opFuncs = {RemainderSOperationExeFunc1Dim, RemainderSOperationExeFunc2Dims,
                                   RemainderSOperationExeFunc3Dims, RemainderSOperationExeFunc4Dims,
                                   RemainderSOperationExeFunc5Dims};
    testCase.opFunc = opFuncs[GetViewShape(test_data).size() - 1];
    TestExecutor::runTest(testCase);
}

TEST_P(RemainderSOperationTest, TestRemainderS)
{
    auto test_data = GetParam().test_data_;
    auto dtype = GetDataType(GetValueByName<std::string>(test_data, "scalar_type"));
    Element scalar(dtype, GetValueByName<float>(test_data, "scalar"));
    auto args = RemainderSOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), scalar, false);
    auto testCase = CreateTestCaseDesc<RemainderSOpMetaData>(GetParam(), &args);
    std::vector<OpFunc> opFuncs = {RemainderSOperationExeFunc1Dim, RemainderSOperationExeFunc2Dims,
                                   RemainderSOperationExeFunc3Dims, RemainderSOperationExeFunc4Dims,
                                   RemainderSOperationExeFunc5Dims};
    testCase.opFunc = opFuncs[GetViewShape(test_data).size() - 1];
    TestExecutor::runTest(testCase);
}
} // namespace
