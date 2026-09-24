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
 * \file test_s_bitwise_right_shift_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct SBitwiseRightShiftOpFuncArgs : public OpFuncArgs {
    SBitwiseRightShiftOpFuncArgs(const Element& value, const std::vector<int64_t>& viewShape,
                                 const std::vector<int64_t> tileShape)
        : value_(value), viewShape_(viewShape), tileShape_(tileShape)
    {}

    Element value_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct SBitwiseRightShiftOpMetaData {
    explicit SBitwiseRightShiftOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void SBitwiseRightShiftOperationExeFuncDoubleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        auto args = static_cast<const SBitwiseRightShiftOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                         std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                        {bIdx * firstViewShape, sIdx * secondViewShape});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = BitwiseRightShift(args->value_, tileTensor0);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void SBitwiseRightShiftOperationExeFuncTripleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        auto* args = static_cast<const SBitwiseRightShiftOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);

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
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = BitwiseRightShift(args->value_, tileTensor0);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void SBitwiseRightShiftOperationExeFuncQuadrupleCut(const std::vector<Tensor>& inputs,
                                                           std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        auto args = static_cast<const SBitwiseRightShiftOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);
        int qloop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(0, qloop, 1))
                    {
                        auto tileTensor0 = View(inputs[0],
                                                {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                 std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                                 std::min(fourthDim - qIdx * fourthViewShape, fourthViewShape)},
                                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                                 qIdx * fourthViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = BitwiseRightShift(args->value_, tileTensor0);
                        Assemble(res,
                                 {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                  qIdx * fourthViewShape},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

class SBitwiseRightShiftOperationTest
    : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<SBitwiseRightShiftOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestSBitwiseRightShift, SBitwiseRightShiftOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<SBitwiseRightShiftOpMetaData>(
                             {SBitwiseRightShiftOperationExeFuncDoubleCut, SBitwiseRightShiftOperationExeFuncTripleCut,
                              SBitwiseRightShiftOperationExeFuncQuadrupleCut},
                             "SBitwiseRightShift")));

TEST_P(SBitwiseRightShiftOperationTest, TestSBitwiseRightShift)
{
    auto test_data = GetParam().test_data_;
    auto dtype = GetDataType(GetValueByName<std::string>(test_data, "scalar_type"));
    Element value(dtype, GetValueByName<float>(test_data, "scalar"));
    auto args = SBitwiseRightShiftOpFuncArgs(value, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<SBitwiseRightShiftOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
