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
 * \file test_hypot_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct HypotOpFuncArgs : public OpFuncArgs {
    HypotOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct HypotOpMetaData {
    explicit HypotOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void HypotOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                       const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        auto args = static_cast<const HypotOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int broadcastFlag = 1;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                Tensor tileTensor0;
                Tensor tileTensor1;
                IF(inputs[0].GetShape()[1] != broadcastFlag && inputs[1].GetShape()[1] == broadcastFlag)
                {
                    tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                    tileTensor1 = View(inputs[1], {firstViewShape, 1},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1},
                                       {bIdx * firstViewShape, 0});
                }
                ELSE IF(inputs[0].GetShape()[0] != broadcastFlag && inputs[1].GetShape()[0] == broadcastFlag)
                {
                    tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                    tileTensor1 = View(inputs[1], {1, secondViewShape},
                                       {1, std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {0, sIdx * secondViewShape});
                }
                ELSE
                {
                    tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                    tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                }
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Hypot(tileTensor0, tileTensor1);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void HypotOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                       const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        auto args = static_cast<const HypotOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                             std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                             std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    auto tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape, thirdViewShape},
                                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                             std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                             std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Hypot(tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void HypotOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                       const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        auto args = static_cast<const HypotOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        const int broadcastFlag = 1;

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int mloop = CeilDiv(thirdDim, thirdViewShape);
        const int nloop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        Tensor tileTensor0;
                        Tensor tileTensor1;
                        IF(inputs[1].GetShape()[2] == broadcastFlag && inputs[0].GetShape()[2] != broadcastFlag)
                        {
                            tileTensor0 = View(inputs[0],
                                               {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                               {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                nIdx * fourthViewShape});
                            tileTensor1 = View(
                                inputs[1], {firstViewShape, secondViewShape, 1, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape), 1,
                                 std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, 0, nIdx * fourthViewShape});
                        }
                        ELSE
                        {
                            tileTensor0 = View(inputs[0],
                                               {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                               {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                nIdx * fourthViewShape});
                            tileTensor1 = View(inputs[1],
                                               {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                               {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                nIdx * fourthViewShape});
                        }
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Hypot(tileTensor0, tileTensor1);
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

class HypotOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<HypotOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestHypot, HypotOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<HypotOpMetaData>(
                             {HypotOperationExeFunc2Dims, HypotOperationExeFunc3Dims, HypotOperationExeFunc4Dims},
                             "Hypot")));

TEST_P(HypotOperationTest, TestHypot)
{
    auto test_data = GetParam().test_data_;
    auto args = HypotOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<HypotOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
