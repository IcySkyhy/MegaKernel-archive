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
 * \file test_topk_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
const unsigned IDX_DIM0 = 0;
const unsigned IDX_DIM1 = 1;
const unsigned IDX_DIM2 = 2;
const unsigned IDX_DIM3 = 3;

struct TopKOpFuncArgs : public OpFuncArgs {
    TopKOpFuncArgs(std::vector<int64_t> viewShape, const std::vector<int64_t> tileShape, std::vector<int> count,
                   std::vector<int> dims, std::vector<bool> largest)
        : viewShape_(viewShape), tileShape_(tileShape), count_(count), dims_(dims), largest_(largest)
    {}
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    std::vector<int> count_;
    std::vector<int> dims_;
    std::vector<bool> largest_;
};

struct TopKOpMetadata {
    TopKOpMetadata(const OpFunc& opFunc, const nlohmann::json& test_data) : opFunc_(opFunc), test_data_(test_data) {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

void TopKOpExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    auto args = static_cast<const TopKOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = inputs[0].GetShape()[0];
    SymbolicScalar secondDim = inputs[0].GetShape()[1];
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];
    int loop[] = {CeilDiv(firstDim, firstViewShape), CeilDiv(secondDim, secondViewShape)};
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                std::vector<SymbolicScalar> offset = {bIdx * args->viewShape_[0], sIdx * args->viewShape_[1]};
                auto viewTensor = View(inputs[0], args->viewShape_,
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       offset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = TopK(viewTensor, args->count_[0], args->dims_[0], args->largest_[0]);
                Assemble(std::get<0>(res), offset, outputs[0]);
                Assemble(std::get<1>(res), offset, outputs[1]);
            }
        }
    }
}

void TopKOpExeFunc3D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    auto args = static_cast<const TopKOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = inputs[0].GetShape()[0];
    SymbolicScalar secondDim = inputs[0].GetShape()[1];
    SymbolicScalar thirdDim = inputs[0].GetShape()[2];
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];
    const int thirdViewShape = args->viewShape_[2];
    int loop[] = {CeilDiv(firstDim, firstViewShape), CeilDiv(secondDim, secondViewShape),
                  CeilDiv(thirdDim, thirdViewShape)};
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                LOOP("LOOP_L2_bIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2]))
                {
                    std::vector<SymbolicScalar> offset = {
                        bIdx * args->viewShape_[0],
                        sIdx * args->viewShape_[1],
                        nIdx * args->viewShape_[2],
                    };
                    auto viewTensor = View(inputs[0], args->viewShape_,
                                           {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                           offset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = TopK(viewTensor, args->count_[0], args->dims_[0], args->largest_[0]);
                    Assemble(std::get<0>(res), offset, outputs[0]);
                    Assemble(std::get<1>(res), offset, outputs[1]);
                }
            }
        }
    }
}

void TopKOpExeFunc4D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    auto args = static_cast<const TopKOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = inputs[0].GetShape()[0];
    SymbolicScalar secondDim = inputs[0].GetShape()[1];
    SymbolicScalar thirdDim = inputs[0].GetShape()[2];
    SymbolicScalar forthDim = inputs[0].GetShape()[3];
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];
    const int thirdViewShape = args->viewShape_[2];
    const int forthViewShape = args->viewShape_[3];
    int loop[] = {CeilDiv(firstDim, firstViewShape), CeilDiv(secondDim, secondViewShape),
                  CeilDiv(thirdDim, thirdViewShape), CeilDiv(forthDim, forthViewShape)};
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                LOOP("LOOP_L2_bIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2]))
                {
                    LOOP("LOOP_L3_bIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(loop[IDX_DIM3]))
                    {
                        std::vector<SymbolicScalar> offset = {
                            bIdx * args->viewShape_[0],
                            sIdx * args->viewShape_[1],
                            nIdx * args->viewShape_[2],
                            qIdx * args->viewShape_[3],
                        };
                        auto viewTensor = View(inputs[0], args->viewShape_,
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                                std::min(forthDim - qIdx * forthViewShape, forthViewShape)},
                                               offset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = TopK(viewTensor, args->count_[0], args->dims_[0], args->largest_[0]);
                        Assemble(std::get<0>(res), offset, outputs[0]);
                        Assemble(std::get<1>(res), offset, outputs[1]);
                    }
                }
            }
        }
    }
}

class TopKOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<TopKOpMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestTopK, TopKOperationTest,
    ::testing::ValuesIn(GetOpMetaData<TopKOpMetadata>({TopKOpExeFunc, TopKOpExeFunc3D, TopKOpExeFunc4D}, "TopK")));

TEST_P(TopKOperationTest, TestTopK)
{
    auto test_data = GetParam().test_data_;
    auto args = TopKOpFuncArgs(
        GetViewShape(test_data), GetTileShape(test_data), GetValueByName<std::vector<int>>(test_data, "count"),
        GetValueByName<std::vector<int>>(test_data, "dims"), GetValueByName<std::vector<bool>>(test_data, "islargest"));
    auto testCase = CreateTestCaseDesc<TopKOpMetadata>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

} // namespace
