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
 * \file test_Permute_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct PermuteOpFuncArgs : public OpFuncArgs {
    PermuteOpFuncArgs(const std::vector<int>& perm, const std::vector<int64_t>& viewShape,
                      const std::vector<int64_t> tileShape)
        : perm_(perm), viewShape_(viewShape), tileShape_(tileShape)
    {}
    std::vector<int> perm_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct PermuteOpMetaData {
    explicit PermuteOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void PermuteOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    const PermuteOpFuncArgs* PermuteInfo = static_cast<const PermuteOpFuncArgs*>(opArgs);
    const int firstViewShape = PermuteInfo->viewShape_[0];
    const int secondViewShape = PermuteInfo->viewShape_[1];
    const std::vector<int>& perm = PermuteInfo->perm_;
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
                TileShape::Current().SetVecTile(PermuteInfo->tileShape_);
                auto res = Permute(tileTensor0, perm);
                // Calculate assemble offset based on permutation
                std::vector<SymbolicScalar> viewOffset = {bIdx * firstViewShape, sIdx * secondViewShape};
                std::vector<SymbolicScalar> permutedOffset = {viewOffset[perm[0]], viewOffset[perm[1]]};
                Assemble(res, permutedOffset, outputs[0]);
            }
        }
    }
}

static void PermuteOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    const PermuteOpFuncArgs* PermuteInfo = static_cast<const PermuteOpFuncArgs*>(opArgs);
    const int firstViewShape = PermuteInfo->viewShape_[0];
    const int secondViewShape = PermuteInfo->viewShape_[1];
    const int thirdViewShape = PermuteInfo->viewShape_[2];
    const std::vector<int>& perm = PermuteInfo->perm_;
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
                    TileShape::Current().SetVecTile(PermuteInfo->tileShape_);
                    auto res = Permute(tileTensor0, perm);
                    // Calculate assemble offset based on permutation
                    std::vector<SymbolicScalar> viewOffset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                              tIdx * thirdViewShape};
                    std::vector<SymbolicScalar> permutedOffset = {viewOffset[perm[0]], viewOffset[perm[1]],
                                                                  viewOffset[perm[2]]};
                    Assemble(res, permutedOffset, outputs[0]);
                }
            }
        }
    }
}

static void PermuteOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    const PermuteOpFuncArgs* PermuteInfo = static_cast<const PermuteOpFuncArgs*>(opArgs);
    const int firstViewShape = PermuteInfo->viewShape_[0];
    const int secondViewShape = PermuteInfo->viewShape_[1];
    const int thirdViewShape = PermuteInfo->viewShape_[2];
    const int forthViewShape = PermuteInfo->viewShape_[3];
    const std::vector<int>& perm = PermuteInfo->perm_;
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
                        TileShape::Current().SetVecTile(PermuteInfo->tileShape_);
                        auto res = Permute(tileTensor0, perm);
                        // Calculate assemble offset based on permutation
                        std::vector<SymbolicScalar> viewOffset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                                  tIdx * thirdViewShape, pIdx * forthViewShape};
                        std::vector<SymbolicScalar> permutedOffset = {viewOffset[perm[0]], viewOffset[perm[1]],
                                                                      viewOffset[perm[2]], viewOffset[perm[3]]};
                        Assemble(res, permutedOffset, outputs[0]);
                    }
                }
            }
        }
    }
}

static void PermuteOperationExeFunc5Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    const PermuteOpFuncArgs* PermuteInfo = static_cast<const PermuteOpFuncArgs*>(opArgs);
    const int firstViewShape = PermuteInfo->viewShape_[0];
    const int secondViewShape = PermuteInfo->viewShape_[1];
    const int thirdViewShape = PermuteInfo->viewShape_[2];
    const int forthViewShape = PermuteInfo->viewShape_[3];
    const int fifthViewShape = PermuteInfo->viewShape_[4];
    const std::vector<int>& perm = PermuteInfo->perm_;
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
                            TileShape::Current().SetVecTile(PermuteInfo->tileShape_);
                            auto res = Permute(tileTensor0, perm);
                            // Calculate assemble offset based on permutation
                            std::vector<SymbolicScalar> viewOffset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                                      tIdx * thirdViewShape, pIdx * forthViewShape,
                                                                      qIdx * fifthViewShape};
                            std::vector<SymbolicScalar> permutedOffset = {viewOffset[perm[0]], viewOffset[perm[1]],
                                                                          viewOffset[perm[2]], viewOffset[perm[3]],
                                                                          viewOffset[perm[4]]};
                            Assemble(res, permutedOffset, outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

class PermuteOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<PermuteOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestPermute, PermuteOperationTest,
    ::testing::ValuesIn(GetOpMetaData<PermuteOpMetaData>({PermuteOperationExeFunc2Dims, PermuteOperationExeFunc3Dims,
                                                          PermuteOperationExeFunc4Dims, PermuteOperationExeFunc5Dims},
                                                         "Permute")));

TEST_P(PermuteOperationTest, TestPermute)
{
    auto test_data = GetParam().test_data_;
    std::vector<int> perm = GetValueByName<std::vector<int>>(test_data, "perm");
    auto args = PermuteOpFuncArgs(perm, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<PermuteOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
