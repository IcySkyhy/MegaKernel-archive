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
 * \file test_log1p_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
using npu::tile_fwk::LogBaseType;
namespace {
struct Log1pOpFuncArgs : public OpFuncArgs {
    Log1pOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct Log1pOpMetaData {
    explicit Log1pOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void Log1pOperationExeFunc1Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                       const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        const struct Log1pOpFuncArgs* args = static_cast<const Log1pOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int bloop = CeilDiv(firstDim, firstViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            auto tileTensor = View(inputs[0], {firstViewShape},
                                   {std::min(firstDim - bIdx * firstViewShape, firstViewShape)},
                                   {bIdx * firstViewShape});
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Log1p(tileTensor);
            Assemble(res, {bIdx * firstViewShape}, outputs[0]);
        }
    }
}

static void Log1pOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                       const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const struct Log1pOpFuncArgs* args = static_cast<const Log1pOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                auto tileTensor = View(inputs[0], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Log1p(tileTensor);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void Log1pOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                       const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        const struct Log1pOpFuncArgs* args = static_cast<const Log1pOpFuncArgs*>(opArgs);
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
                LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    auto tileTensor = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                                           {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                           {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Log1p(tileTensor);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void Log1pOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                       const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        auto args = static_cast<const Log1pOpFuncArgs*>(opArgs);
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
                        Tensor tileTensor0 = View(inputs[0],
                                                  {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                                  {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                   std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                   std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                   std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                                  {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                   nIdx * fourthViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Log1p(tileTensor0);
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

class Log1pOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<Log1pOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestLog1p, Log1pOperationTest,
    ::testing::ValuesIn(GetOpMetaData<Log1pOpMetaData>({Log1pOperationExeFunc2Dims, Log1pOperationExeFunc3Dims,
                                                        Log1pOperationExeFunc4Dims, Log1pOperationExeFunc1Dims},
                                                       "Log1p")));

TEST_P(Log1pOperationTest, TestLog1p)
{
    auto test_data = GetParam().test_data_;
    auto args = Log1pOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<Log1pOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
