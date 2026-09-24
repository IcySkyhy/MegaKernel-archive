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
 * \file test_transpose_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct TransposeOpFuncArgs : public OpFuncArgs {
    TransposeOpFuncArgs(int first_dim, int second_dim, const std::vector<int64_t>& viewShape,
                        const std::vector<int64_t> tileShape)
        : first_dim_(first_dim), second_dim_(second_dim), viewShape_(viewShape), tileShape_(tileShape)
    {}
    int first_dim_;
    int second_dim_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct TransposeOpMetaData {
    explicit TransposeOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void TransposeOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    const TransposeOpFuncArgs* transposeInfo = static_cast<const TransposeOpFuncArgs*>(opArgs);
    const int firstViewShape = transposeInfo->viewShape_[0];
    const int secondViewShape = transposeInfo->viewShape_[1];
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1))
            {
                Tensor tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                          {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                           std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                          {bIdx * firstViewShape, sIdx * secondViewShape});
                TileShape::Current().SetVecTile(transposeInfo->tileShape_);
                auto res = Transpose(tileTensor0, {transposeInfo->first_dim_, transposeInfo->second_dim_});
                Assemble(res, {sIdx * secondViewShape, bIdx * firstViewShape}, outputs[0]);
            }
        }
    }
}

static void TransposeOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    const TransposeOpFuncArgs* transposeInfo = static_cast<const TransposeOpFuncArgs*>(opArgs);
    const int firstViewShape = transposeInfo->viewShape_[0];
    const int secondViewShape = transposeInfo->viewShape_[1];
    const int thirdViewShape = transposeInfo->viewShape_[2];
    int first_dim = transposeInfo->first_dim_ < 0 ? transposeInfo->first_dim_ + transposeInfo->viewShape_.size() :
                                                    transposeInfo->first_dim_;
    int second_dim = transposeInfo->second_dim_ < 0 ? transposeInfo->second_dim_ + transposeInfo->viewShape_.size() :
                                                      transposeInfo->second_dim_;
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1))
            {
                LOOP("LOOP_L2_tIdx", FunctionType::DYNAMIC_LOOP, tIdx,
                     LoopRange(0, CeilDiv(thirdDim, thirdViewShape), 1))
                {
                    Tensor tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                                              {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                               std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                               std::min(thirdDim - tIdx * thirdViewShape, thirdViewShape)},
                                              {bIdx * firstViewShape, sIdx * secondViewShape, tIdx * thirdViewShape});
                    TileShape::Current().SetVecTile(transposeInfo->tileShape_);
                    auto res = Transpose(tileTensor0, {transposeInfo->first_dim_, transposeInfo->second_dim_});
                    std::vector<SymbolicScalar> viewOffset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                              tIdx * thirdViewShape};
                    std::swap(viewOffset[first_dim], viewOffset[second_dim]);
                    Assemble(res, viewOffset, outputs[0]);
                }
            }
        }
    }
}

static void TransposeOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    const TransposeOpFuncArgs* transposeInfo = static_cast<const TransposeOpFuncArgs*>(opArgs);
    const int firstViewShape = transposeInfo->viewShape_[0];
    const int secondViewShape = transposeInfo->viewShape_[1];
    const int thirdViewShape = transposeInfo->viewShape_[2];
    const int forthViewShape = transposeInfo->viewShape_[3];
    int first_dim = transposeInfo->first_dim_ < 0 ? transposeInfo->first_dim_ + transposeInfo->viewShape_.size() :
                                                    transposeInfo->first_dim_;
    int second_dim = transposeInfo->second_dim_ < 0 ? transposeInfo->second_dim_ + transposeInfo->viewShape_.size() :
                                                      transposeInfo->second_dim_;
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar forthDim = inputs[0].GetShape()[3];
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1))
            {
                LOOP("LOOP_L2_tIdx", FunctionType::DYNAMIC_LOOP, tIdx,
                     LoopRange(0, CeilDiv(thirdDim, thirdViewShape), 1))
                {
                    LOOP("LOOP_L3_tIdx", FunctionType::DYNAMIC_LOOP, pIdx,
                         LoopRange(0, CeilDiv(forthDim, forthViewShape), 1))
                    {
                        Tensor tileTensor0 = View(inputs[0],
                                                  {firstViewShape, secondViewShape, thirdViewShape, forthViewShape},
                                                  {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                   std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                   std::min(thirdDim - tIdx * thirdViewShape, thirdViewShape),
                                                   std::min(forthDim - pIdx * forthViewShape, forthViewShape)},
                                                  {bIdx * firstViewShape, sIdx * secondViewShape, tIdx * thirdViewShape,
                                                   pIdx * forthViewShape});
                        TileShape::Current().SetVecTile(transposeInfo->tileShape_);
                        auto res = Transpose(tileTensor0, {transposeInfo->first_dim_, transposeInfo->second_dim_});
                        std::vector<SymbolicScalar> viewOffset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                                  tIdx * thirdViewShape, pIdx * forthViewShape};
                        std::swap(viewOffset[first_dim], viewOffset[second_dim]);
                        Assemble(res, viewOffset, outputs[0]);
                    }
                }
            }
        }
    }
}

static void TransposeOperationExeFunc5Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    const TransposeOpFuncArgs* transposeInfo = static_cast<const TransposeOpFuncArgs*>(opArgs);
    const int firstViewShape = transposeInfo->viewShape_[0];
    const int secondViewShape = transposeInfo->viewShape_[1];
    const int thirdViewShape = transposeInfo->viewShape_[2];
    const int forthViewShape = transposeInfo->viewShape_[3];
    const int fifthViewShape = transposeInfo->viewShape_[4];
    int first_dim = transposeInfo->first_dim_ < 0 ? transposeInfo->first_dim_ + transposeInfo->viewShape_.size() :
                                                    transposeInfo->first_dim_;
    int second_dim = transposeInfo->second_dim_ < 0 ? transposeInfo->second_dim_ + transposeInfo->viewShape_.size() :
                                                      transposeInfo->second_dim_;
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar forthDim = inputs[0].GetShape()[3];
        SymbolicScalar fifthDim = inputs[0].GetShape()[4];
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1))
            {
                LOOP("LOOP_L2_tIdx", FunctionType::DYNAMIC_LOOP, tIdx,
                     LoopRange(0, CeilDiv(thirdDim, thirdViewShape), 1))
                {
                    LOOP("LOOP_L3_pIdx", FunctionType::DYNAMIC_LOOP, pIdx,
                         LoopRange(0, CeilDiv(forthDim, forthViewShape), 1))
                    {
                        LOOP("LOOP_L4_qIdx", FunctionType::DYNAMIC_LOOP, qIdx,
                             LoopRange(0, CeilDiv(fifthDim, fifthViewShape), 1))
                        {
                            Tensor tileTensor0 = View(
                                inputs[0],
                                {firstViewShape, secondViewShape, thirdViewShape, forthViewShape, fifthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                 std::min(thirdDim - tIdx * thirdViewShape, thirdViewShape),
                                 std::min(forthDim - pIdx * forthViewShape, forthViewShape),
                                 std::min(fifthDim - qIdx * fifthViewShape, fifthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, tIdx * thirdViewShape,
                                 pIdx * forthViewShape, qIdx * fifthViewShape});
                            TileShape::Current().SetVecTile(transposeInfo->tileShape_);
                            auto res = Transpose(tileTensor0, {transposeInfo->first_dim_, transposeInfo->second_dim_});
                            std::vector<SymbolicScalar> viewOffset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                                      tIdx * thirdViewShape, pIdx * forthViewShape,
                                                                      qIdx * fifthViewShape};
                            std::swap(viewOffset[first_dim], viewOffset[second_dim]);
                            Assemble(res, viewOffset, outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

class TransposeOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<TransposeOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestTranspose, TransposeOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<TransposeOpMetaData>(
                             {TransposeOperationExeFunc2Dims, TransposeOperationExeFunc3Dims,
                              TransposeOperationExeFunc4Dims, TransposeOperationExeFunc5Dims},
                             "Transpose")));

TEST_P(TransposeOperationTest, TestTranspose)
{
    auto test_data = GetParam().test_data_;
    int first_dim = GetValueByName<int>(test_data, "first_dim");
    int second_dim = GetValueByName<int>(test_data, "second_dim");
    auto args = TransposeOpFuncArgs(first_dim, second_dim, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<TransposeOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
