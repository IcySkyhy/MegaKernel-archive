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
 * \file test_Tri_operation.cpp
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
struct TriOpFuncArgs : public OpFuncArgs {
    TriOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, int diagonal,
                  bool isUpper)
        : viewShape_(viewShape), tileShape_(tileShape), diagonal_(diagonal), isUpper_(isUpper)
    {}
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    int diagonal_;
    bool isUpper_;
};

struct TriOpMetaData {
    explicit TriOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void TriOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    size_t shapeSize = inputs[0].GetShape().size();
    SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
    SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
    auto args = static_cast<const TriOpFuncArgs*>(opArgs);
    const std::vector<int64_t> realViewShape = args->viewShape_;
    const int bloop = CeilDiv(src_firstDim, realViewShape[0]);
    const int sloop = CeilDiv(src_secondDim, realViewShape[1]);

    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                std::vector<SymbolicScalar> dynOffsets = {bIdx * realViewShape[0], sIdx * realViewShape[1]};
                auto tileTensor = View(inputs[0], realViewShape,
                                       {std::min(src_firstDim - bIdx * realViewShape[0], realViewShape[0]),
                                        std::min(src_secondDim - sIdx * realViewShape[1], realViewShape[1])},
                                       dynOffsets);
                auto realDiagonal = args->diagonal_ + dynOffsets[shapeSize - 2] - dynOffsets[shapeSize - 1];
                TileShape::Current().SetVecTile(args->tileShape_);
                Tensor res = args->isUpper_ ? TriU(tileTensor, realDiagonal) : TriL(tileTensor, realDiagonal);
                Assemble(res, dynOffsets, outputs[0]);
            }
        }
    }
}

static void TriOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    size_t shapeSize = inputs[0].GetShape().size();
    SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
    SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
    SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
    auto args = static_cast<const TriOpFuncArgs*>(opArgs);
    std::vector<int64_t> viewShape = args->viewShape_;
    const int loop[] = {CeilDiv(src_firstDim, viewShape[0]), CeilDiv(src_secondDim, viewShape[1]),
                        CeilDiv(src_thirdDim, viewShape[2])};
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
                                           {std::min(src_firstDim - bIdx * viewShape[0], viewShape[0]),
                                            std::min(src_secondDim - sIdx * viewShape[1], viewShape[1]),
                                            std::min(src_thirdDim - nIdx * viewShape[2], viewShape[2])},
                                           dynOffsets);
                    auto realDiagonal = args->diagonal_ + dynOffsets[shapeSize - 2] - dynOffsets[shapeSize - 1];
                    TileShape::Current().SetVecTile(args->tileShape_);
                    Tensor res = args->isUpper_ ? TriU(tileTensor, realDiagonal) : TriL(tileTensor, realDiagonal);
                    Assemble(res, dynOffsets, outputs[0]);
                }
            }
        }
    }
}

static void TriOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    size_t shapeSize = inputs[0].GetShape().size();
    SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
    SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
    SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
    SymbolicScalar src_forthDim = inputs[0].GetShape()[3];

    auto args = static_cast<const TriOpFuncArgs*>(opArgs);
    std::vector<int64_t> viewShape = args->viewShape_;
    const int loop[] = {CeilDiv(src_firstDim, viewShape[0]), CeilDiv(src_secondDim, viewShape[1]),
                        CeilDiv(src_thirdDim, viewShape[2]), CeilDiv(src_forthDim, viewShape[3])};
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
                                               {std::min(src_firstDim - bIdx * viewShape[0], viewShape[0]),
                                                std::min(src_secondDim - sIdx * viewShape[1], viewShape[1]),
                                                std::min(src_thirdDim - nIdx * viewShape[2], viewShape[2]),
                                                std::min(src_forthDim - qIdx * viewShape[3], viewShape[3])},
                                               dynOffsets);
                        auto realDiagonal = args->diagonal_ + dynOffsets[shapeSize - 2] - dynOffsets[shapeSize - 1];
                        TileShape::Current().SetVecTile(args->tileShape_);
                        Tensor res = args->isUpper_ ? TriU(tileTensor, realDiagonal) : TriL(tileTensor, realDiagonal);
                        Assemble(res, dynOffsets, outputs[0]);
                    }
                }
            }
        }
    }
}

static void TriOperationExeFunc5Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    size_t shapeSize = inputs[0].GetShape().size();
    SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
    SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
    SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
    SymbolicScalar src_forthDim = inputs[0].GetShape()[3];
    SymbolicScalar src_fifthDim = inputs[0].GetShape()[4];

    auto args = static_cast<const TriOpFuncArgs*>(opArgs);
    std::vector<int64_t> viewShape = args->viewShape_;
    const int loop[] = {CeilDiv(src_firstDim, viewShape[0]), CeilDiv(src_secondDim, viewShape[1]),
                        CeilDiv(src_thirdDim, viewShape[2]), CeilDiv(src_forthDim, viewShape[3]),
                        CeilDiv(src_fifthDim, viewShape[4])};
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
                                                   {std::min(src_firstDim - bIdx * viewShape[0], viewShape[0]),
                                                    std::min(src_secondDim - sIdx * viewShape[1], viewShape[1]),
                                                    std::min(src_thirdDim - nIdx * viewShape[2], viewShape[2]),
                                                    std::min(src_forthDim - qIdx * viewShape[3], viewShape[3]),
                                                    std::min(src_fifthDim - rIdx * viewShape[4], viewShape[4])},
                                                   dynOffsets);
                            auto realDiagonal = args->diagonal_ + dynOffsets[shapeSize - 2] - dynOffsets[shapeSize - 1];
                            TileShape::Current().SetVecTile(args->tileShape_);
                            Tensor res = args->isUpper_ ? TriU(tileTensor, realDiagonal) :
                                                          TriL(tileTensor, realDiagonal);
                            Assemble(res, dynOffsets, outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

class TriUOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<TriOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestTriU, TriUOperationTest,
    ::testing::ValuesIn(GetOpMetaData<TriOpMetaData>({TriOperationExeFunc2Dims, TriOperationExeFunc3Dims,
                                                      TriOperationExeFunc4Dims, TriOperationExeFunc5Dims},
                                                     "TriU")));

class TriLOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<TriOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestTriL, TriLOperationTest,
    ::testing::ValuesIn(GetOpMetaData<TriOpMetaData>({TriOperationExeFunc2Dims, TriOperationExeFunc3Dims,
                                                      TriOperationExeFunc4Dims, TriOperationExeFunc5Dims},
                                                     "TriL")));

TEST_P(TriUOperationTest, TestTriU)
{
    auto test_data = GetParam().test_data_;
    bool isUpper = true;
    int diagonal = GetValueByName<int>(test_data, "diagonal");
    auto args = TriOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), diagonal, isUpper);
    auto testCase = CreateTestCaseDesc<TriOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

TEST_P(TriLOperationTest, TestTriL)
{
    auto test_data = GetParam().test_data_;
    bool isUpper = false;
    int diagonal = GetValueByName<int>(test_data, "diagonal");
    auto args = TriOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), diagonal, isUpper);
    auto testCase = CreateTestCaseDesc<TriOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
