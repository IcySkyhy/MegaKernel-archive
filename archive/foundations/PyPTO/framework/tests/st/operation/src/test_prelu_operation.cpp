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
 * \file test_prelu_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct PReLUOpFuncArgs : public OpFuncArgs {
    PReLUOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct PReLUOpMetaData {
    explicit PReLUOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void PReLUOperationExeFunc1Dim(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                      const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto args = static_cast<const PReLUOpFuncArgs*>(opArgs);

        SymbolicScalar firstDim = inputs[0].GetShape()[0];

        const int firstViewShape = args->viewShape_[0];

        const int bloop = CeilDiv(firstDim, firstViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            auto tileTensor0 = View(inputs[0], {firstViewShape},
                                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape)},
                                    {bIdx * firstViewShape});

            // 1D 输入时，weight 恒为 [1]
            auto tileTensor1 = View(inputs[1], {1}, {1}, {0});

            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = PReLU(tileTensor0, tileTensor1);
            Assemble(res, {bIdx * firstViewShape}, outputs[0]);
        }
    }
}

static void PReLUOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                       const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto args = static_cast<const PReLUOpFuncArgs*>(opArgs);

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];

        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                         std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                        {bIdx * firstViewShape, sIdx * secondViewShape});

                auto weightViewShape = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                auto tileTensor1 = View(inputs[1], {secondViewShape}, {weightViewShape}, {sIdx * secondViewShape});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = PReLU(tileTensor0, tileTensor1);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void PReLUOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                       const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto args = static_cast<const PReLUOpFuncArgs*>(opArgs);

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];

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

                    auto weightViewShape = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                    auto tileTensor1 = View(inputs[1], {secondViewShape}, {weightViewShape}, {sIdx * secondViewShape});

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = PReLU(tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void PReLUOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                       const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto args = static_cast<const PReLUOpFuncArgs*>(opArgs);

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];

        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];

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
                        auto tileTensor0 = View(inputs[0],
                                                {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                 std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                 std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                                {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                 nIdx * fourthViewShape});

                        auto weightViewShape = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                        auto tileTensor1 = View(inputs[1], {secondViewShape}, {weightViewShape},
                                                {sIdx * secondViewShape});

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = PReLU(tileTensor0, tileTensor1);
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

class PReLUOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<PReLUOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestPReLU, PReLUOperationTest,
    ::testing::ValuesIn(GetOpMetaData<PReLUOpMetaData, 1>({PReLUOperationExeFunc1Dim, PReLUOperationExeFunc2Dims,
                                                           PReLUOperationExeFunc3Dims, PReLUOperationExeFunc4Dims},
                                                          "PReLU")));

TEST_P(PReLUOperationTest, TestPReLU)
{
    auto test_data = GetParam().test_data_;
    auto args = PReLUOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<PReLUOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
