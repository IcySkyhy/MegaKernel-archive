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
 * \file test_sub_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct ExpandExpDifOpFuncArgs : public OpFuncArgs {
    ExpandExpDifOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct ExpandExpDifOpMetaData {
    explicit ExpandExpDifOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void ExpandExpDifOperationExeFunc1Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                              const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstTensorDim = inputs[0].GetShape()[0];
        SymbolicScalar secondTensorDim = inputs[1].GetShape()[0];
        auto args = static_cast<const ExpandExpDifOpFuncArgs*>(opArgs);
        const int firstViewShape = std::min(args->viewShape_[0], firstTensorDim);

        const int bloop = CeilDiv(firstTensorDim, firstViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            Tensor tileTensor0 = View(inputs[0], {firstViewShape},
                                      {std::min(firstTensorDim - bIdx * firstViewShape, firstViewShape)},
                                      {bIdx * firstViewShape});
            Tensor tileTensor1;
            IF(secondTensorDim == 1) { tileTensor1 = View(inputs[1], {1}, {1}, {0}); }
            ELSE
            {
                tileTensor1 = View(inputs[1], {firstViewShape},
                                   {std::min(secondTensorDim - bIdx * firstViewShape, firstViewShape)},
                                   {bIdx * firstViewShape});
            }
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = ExpandExpDif(tileTensor0, tileTensor1);
            Assemble(res, {bIdx * firstViewShape}, outputs[0]);
        }
    }
}

static void ExpandExpDifOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                              const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstTensorDim0 = inputs[0].GetShape()[0];
        SymbolicScalar firstTensorDim1 = inputs[0].GetShape()[1];
        SymbolicScalar secondTensorDim0 = inputs[1].GetShape()[0];
        SymbolicScalar secondTensorDim1 = inputs[1].GetShape()[1];
        auto args = static_cast<const ExpandExpDifOpFuncArgs*>(opArgs);
        const int firstViewShape = std::min(args->viewShape_[0], firstTensorDim0);
        const int secondViewShape = std::min(args->viewShape_[1], firstTensorDim1);

        const int bloop = CeilDiv(firstTensorDim0, firstViewShape);
        const int sloop = CeilDiv(firstTensorDim1, secondViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                Tensor tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                          {std::min(firstTensorDim0 - bIdx * firstViewShape, firstViewShape),
                                           std::min(firstTensorDim1 - sIdx * secondViewShape, secondViewShape)},
                                          {bIdx * firstViewShape, sIdx * secondViewShape});
                Tensor tileTensor1;
                IF(secondTensorDim0 == 1 && secondTensorDim1 != 1)
                {
                    tileTensor1 = View(inputs[1], {1, secondViewShape},
                                       {1, std::min(secondTensorDim1 - sIdx * secondViewShape, secondViewShape)},
                                       {0, sIdx * secondViewShape});
                }
                ELSE IF(secondTensorDim0 != 1 && secondTensorDim1 == 1)
                {
                    tileTensor1 = View(inputs[1], {firstViewShape, 1},
                                       {std::min(secondTensorDim0 - bIdx * firstViewShape, firstViewShape), 1},
                                       {bIdx * firstViewShape, 0});
                }
                ELSE { tileTensor1 = View(inputs[1], {1, 1}, {1, 1}, {0, 0}); }
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = ExpandExpDif(tileTensor0, tileTensor1);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void ExpandExpDifOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                              const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstTensorDim0 = inputs[0].GetShape()[0];
        SymbolicScalar firstTensorDim1 = inputs[0].GetShape()[1];
        SymbolicScalar firstTensorDim2 = inputs[0].GetShape()[2];
        SymbolicScalar secondTensorDim0 = inputs[1].GetShape()[0];
        SymbolicScalar secondTensorDim1 = inputs[1].GetShape()[1];
        SymbolicScalar secondTensorDim2 = inputs[1].GetShape()[2];
        auto args = static_cast<const ExpandExpDifOpFuncArgs*>(opArgs);
        const int firstViewShape = std::min(args->viewShape_[0], firstTensorDim0);
        const int secondViewShape = std::min(args->viewShape_[1], firstTensorDim1);
        const int thirdViewShape = std::min(args->viewShape_[2], firstTensorDim2);

        const int bloop = CeilDiv(firstTensorDim0, firstViewShape);
        const int sloop = CeilDiv(firstTensorDim1, secondViewShape);
        const int nloop = CeilDiv(firstTensorDim2, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                                            {std::min(firstTensorDim0 - bIdx * firstViewShape, firstViewShape),
                                             std::min(firstTensorDim1 - sIdx * secondViewShape, secondViewShape),
                                             std::min(firstTensorDim2 - nIdx * thirdViewShape, thirdViewShape)},
                                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    Tensor tileTensor1;
                    IF(secondTensorDim1 == 1 && secondTensorDim2 != 1)
                    {
                        tileTensor1 = View(inputs[1], {firstViewShape, 1, thirdViewShape},
                                           {std::min(secondTensorDim0 - bIdx * firstViewShape, firstViewShape), 1,
                                            std::min(secondTensorDim2 - nIdx * thirdViewShape, thirdViewShape)},
                                           {bIdx * firstViewShape, 0, nIdx * thirdViewShape});
                    }
                    ELSE IF(secondTensorDim1 != 1 && secondTensorDim2 == 1)
                    {
                        tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape, 1},
                                           {std::min(secondTensorDim0 - bIdx * firstViewShape, firstViewShape),
                                            std::min(secondTensorDim1 - sIdx * secondViewShape, secondViewShape), 1},
                                           {bIdx * firstViewShape, sIdx * secondViewShape, 0});
                    }
                    ELSE
                    {
                        tileTensor1 = View(inputs[1], {firstViewShape, 1, 1},
                                           {std::min(secondTensorDim0 - bIdx * firstViewShape, firstViewShape), 1, 1},
                                           {bIdx * firstViewShape, 0, 0});
                    }
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = ExpandExpDif(tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void ExpandExpDifOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                              const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstTensorDim0 = inputs[0].GetShape()[0];
        SymbolicScalar firstTensorDim1 = inputs[0].GetShape()[1];
        SymbolicScalar firstTensorDim2 = inputs[0].GetShape()[2];
        SymbolicScalar firstTensorDim3 = inputs[0].GetShape()[3];
        SymbolicScalar secondTensorDim0 = inputs[1].GetShape()[0];
        SymbolicScalar secondTensorDim1 = inputs[1].GetShape()[1];
        SymbolicScalar secondTensorDim2 = inputs[1].GetShape()[2];
        SymbolicScalar secondTensorDim3 = inputs[1].GetShape()[3];
        auto args = static_cast<const ExpandExpDifOpFuncArgs*>(opArgs);
        const int firstViewShape = std::min(args->viewShape_[0], firstTensorDim0);
        const int secondViewShape = std::min(args->viewShape_[1], firstTensorDim1);
        const int thirdViewShape = std::min(args->viewShape_[2], firstTensorDim2);
        const int fourthViewShape = std::min(args->viewShape_[3], firstTensorDim3);

        const int bloop = CeilDiv(firstTensorDim0, firstViewShape);
        const int sloop = CeilDiv(firstTensorDim1, secondViewShape);
        const int mloop = CeilDiv(firstTensorDim2, thirdViewShape);
        const int nloop = CeilDiv(firstTensorDim3, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        Tensor tileTensor0 = View(inputs[0],
                                                  {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                                  {std::min(firstTensorDim0 - bIdx * firstViewShape, firstViewShape),
                                                   std::min(firstTensorDim1 - sIdx * secondViewShape, secondViewShape),
                                                   std::min(firstTensorDim2 - mIdx * thirdViewShape, thirdViewShape),
                                                   std::min(firstTensorDim3 - nIdx * fourthViewShape, fourthViewShape)},
                                                  {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                   nIdx * fourthViewShape});
                        Tensor tileTensor1;
                        IF(secondTensorDim2 == 1 && secondTensorDim3 != 1)
                        {
                            tileTensor1 = View(
                                inputs[1], {firstViewShape, secondViewShape, 1, fourthViewShape},
                                {std::min(secondTensorDim0 - bIdx * firstViewShape, firstViewShape),
                                 std::min(secondTensorDim1 - sIdx * secondViewShape, secondViewShape), 1,
                                 std::min(secondTensorDim3 - nIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, 0, nIdx * fourthViewShape});
                        }
                        ELSE IF(secondTensorDim2 != 1 && secondTensorDim3 == 1)
                        {
                            tileTensor1 = View(
                                inputs[1], {firstViewShape, secondViewShape, thirdViewShape, 1},
                                {std::min(secondTensorDim0 - bIdx * firstViewShape, firstViewShape),
                                 std::min(secondTensorDim1 - sIdx * secondViewShape, secondViewShape),
                                 std::min(secondTensorDim2 - mIdx * thirdViewShape, thirdViewShape), 1},
                                {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape, 0});
                        }
                        ELSE
                        {
                            tileTensor1 = View(
                                inputs[1], {firstViewShape, secondViewShape, 1, 1},
                                {std::min(secondTensorDim0 - bIdx * firstViewShape, firstViewShape),
                                 std::min(secondTensorDim1 - sIdx * secondViewShape, secondViewShape), 1, 1},
                                {bIdx * firstViewShape, sIdx * secondViewShape, 0, 0});
                        }
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = ExpandExpDif(tileTensor0, tileTensor1);
                        Assemble(res,
                                 {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                  nIdx * fourthViewShape},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

class ExpandExpDifOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ExpandExpDifOpMetaData> {
};

INSTANTIATE_TEST_SUITE_P(TestExpandExpDif, ExpandExpDifOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<ExpandExpDifOpMetaData, 1>(
                             {ExpandExpDifOperationExeFunc1Dims, ExpandExpDifOperationExeFunc2Dims,
                              ExpandExpDifOperationExeFunc3Dims, ExpandExpDifOperationExeFunc4Dims},
                             "ExpandExpDif")));

TEST_P(ExpandExpDifOperationTest, TestExpandExpDif)
{
    auto test_data = GetParam().test_data_;
    auto args = ExpandExpDifOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<ExpandExpDifOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
