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
 * \file test_logical_and_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct LogicalAndOpFuncArgs : public OpFuncArgs {
    LogicalAndOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct LogicalAndOpMetaData {
    explicit LogicalAndOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void LogicalAndOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        const struct LogicalAndOpFuncArgs* args = static_cast<const LogicalAndOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                auto createTensorView = [&](const Tensor& tensor, SymbolicScalar bOffset, SymbolicScalar sOffset) {
                    SymbolicScalar validDim0 = std::min(firstDim - bOffset, SymbolicScalar(firstViewShape));
                    SymbolicScalar validDim1 = std::min(secondDim - sOffset, SymbolicScalar(secondViewShape));
                    SymbolicScalar offset0 = bOffset;
                    SymbolicScalar offset1 = sOffset;
                    SymbolicScalar inputviewshape0 = firstViewShape;
                    SymbolicScalar inputviewshape1 = secondViewShape;

                    if (tensor.GetShape()[0] == 1) {
                        validDim0 = 1;
                        offset0 = 0;
                        inputviewshape0 = 1;
                    }
                    if (tensor.GetShape()[1] == 1) {
                        validDim1 = 1;
                        offset1 = 0;
                        inputviewshape1 = 1;
                    }
                    return View(tensor, {inputviewshape0, inputviewshape1}, {validDim0, validDim1}, {offset0, offset1});
                };
                auto tileTensor0 = createTensorView(inputs[0], SymbolicScalar(bIdx * firstViewShape),
                                                    SymbolicScalar(sIdx * secondViewShape));
                auto tileTensor1 = createTensorView(inputs[1], SymbolicScalar(bIdx * firstViewShape),
                                                    SymbolicScalar(sIdx * secondViewShape));
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = LogicalAnd(tileTensor0, tileTensor1);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}
static void LogicalAndOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        SymbolicScalar thirdDim = std::max(inputs[0].GetShape()[2], inputs[1].GetShape()[2]);
        const struct LogicalAndOpFuncArgs* args = static_cast<const LogicalAndOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int mloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                {
                    auto createTensorView = [&](const Tensor& tensor, SymbolicScalar bOffset, SymbolicScalar sOffset,
                                                SymbolicScalar mOffset) {
                        SymbolicScalar validDim0 = std::min(firstDim - bOffset, SymbolicScalar(firstViewShape));
                        SymbolicScalar validDim1 = std::min(secondDim - sOffset, SymbolicScalar(secondViewShape));
                        SymbolicScalar validDim2 = std::min(thirdDim - mOffset, SymbolicScalar(thirdViewShape));
                        SymbolicScalar offset0 = bOffset;
                        SymbolicScalar offset1 = sOffset;
                        SymbolicScalar offset2 = mOffset;
                        SymbolicScalar inputviewshape0 = firstViewShape;
                        SymbolicScalar inputviewshape1 = secondViewShape;
                        SymbolicScalar inputviewshape2 = thirdViewShape;
                        if (tensor.GetShape()[0] == 1) {
                            validDim0 = 1;
                            offset0 = 0;
                            inputviewshape0 = 1;
                        }
                        if (tensor.GetShape()[1] == 1) {
                            validDim1 = 1;
                            offset1 = 0;
                            inputviewshape1 = 1;
                        }
                        if (tensor.GetShape()[2] == 1) {
                            validDim2 = 1;
                            offset2 = 0;
                            inputviewshape2 = 1;
                        }
                        return View(tensor, {inputviewshape0, inputviewshape1, inputviewshape2},
                                    {validDim0, validDim1, validDim2}, {offset0, offset1, offset2});
                    };
                    auto tileTensor0 = createTensorView(inputs[0], SymbolicScalar(bIdx * firstViewShape),
                                                        SymbolicScalar(sIdx * secondViewShape),
                                                        SymbolicScalar(mIdx * thirdViewShape));
                    auto tileTensor1 = createTensorView(inputs[1], SymbolicScalar(bIdx * firstViewShape),
                                                        SymbolicScalar(sIdx * secondViewShape),
                                                        SymbolicScalar(mIdx * thirdViewShape));
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = LogicalAnd(tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void LogicalAndOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        SymbolicScalar thirdDim = std::max(inputs[0].GetShape()[2], inputs[1].GetShape()[2]);
        SymbolicScalar fourthDim = std::max(inputs[0].GetShape()[3], inputs[1].GetShape()[3]);
        auto args = static_cast<const LogicalAndOpFuncArgs*>(opArgs);
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
                        auto createTensorView = [&](const Tensor& tensor, SymbolicScalar bOffset,
                                                    SymbolicScalar sOffset, SymbolicScalar mOffset,
                                                    SymbolicScalar nOffset) {
                            SymbolicScalar validDim0 = std::min(firstDim - bOffset, SymbolicScalar(firstViewShape));
                            SymbolicScalar validDim1 = std::min(secondDim - sOffset, SymbolicScalar(secondViewShape));
                            SymbolicScalar validDim2 = std::min(thirdDim - mOffset, SymbolicScalar(thirdViewShape));
                            SymbolicScalar validDim3 = std::min(fourthDim - nOffset, SymbolicScalar(fourthViewShape));
                            SymbolicScalar offset0 = bOffset;
                            SymbolicScalar offset1 = sOffset;
                            SymbolicScalar offset2 = mOffset;
                            SymbolicScalar offset3 = nOffset;
                            SymbolicScalar inputviewshape0 = firstViewShape;
                            SymbolicScalar inputviewshape1 = secondViewShape;
                            SymbolicScalar inputviewshape2 = thirdViewShape;
                            SymbolicScalar inputviewshape3 = fourthViewShape;
                            if (tensor.GetShape()[0] == 1) {
                                validDim0 = 1;
                                offset0 = 0;
                                inputviewshape0 = 1;
                            }
                            if (tensor.GetShape()[1] == 1) {
                                validDim1 = 1;
                                offset1 = 0;
                                inputviewshape1 = 1;
                            }
                            if (tensor.GetShape()[2] == 1) {
                                validDim2 = 1;
                                offset2 = 0;
                                inputviewshape2 = 1;
                            }
                            if (tensor.GetShape()[3] == 1) {
                                validDim3 = 1;
                                offset3 = 0;
                                inputviewshape3 = 1;
                            }
                            return View(tensor, {inputviewshape0, inputviewshape1, inputviewshape2, inputviewshape3},
                                        {validDim0, validDim1, validDim2, validDim3},
                                        {offset0, offset1, offset2, offset3});
                        };
                        auto tileTensor0 = createTensorView(
                            inputs[0], SymbolicScalar(bIdx * firstViewShape), SymbolicScalar(sIdx * secondViewShape),
                            SymbolicScalar(mIdx * thirdViewShape), SymbolicScalar(nIdx * fourthViewShape));
                        auto tileTensor1 = createTensorView(
                            inputs[1], SymbolicScalar(bIdx * firstViewShape), SymbolicScalar(sIdx * secondViewShape),
                            SymbolicScalar(mIdx * thirdViewShape), SymbolicScalar(nIdx * fourthViewShape));
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = LogicalAnd(tileTensor0, tileTensor1);
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

class LogicalAndOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<LogicalAndOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestLogicalAnd, LogicalAndOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<LogicalAndOpMetaData>({LogicalAndOperationExeFunc2Dims,
                                                                                  LogicalAndOperationExeFunc3Dims,
                                                                                  LogicalAndOperationExeFunc4Dims},
                                                                                 "LogicalAnd")));

TEST_P(LogicalAndOperationTest, TestLogicalAnd)
{
    auto test_data = GetParam().test_data_;
    auto args = LogicalAndOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<LogicalAndOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
