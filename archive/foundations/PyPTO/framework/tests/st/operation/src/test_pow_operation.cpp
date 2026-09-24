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
 * \file test_pow_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct PowOpFuncArgs : public OpFuncArgs {
    PowOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct PowOpMetaData {
    explicit PowOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static inline void DoPow(const std::vector<int64_t>& shape0, const std::vector<int64_t>& shape1,
                         const std::vector<SymbolicScalar>& validShape0, const std::vector<SymbolicScalar>& validShape1,
                         const std::vector<SymbolicScalar>& offset0, const std::vector<SymbolicScalar>& offset1,
                         const std::vector<SymbolicScalar>& offset2, const Tensor& input0, const Tensor& input1,
                         Tensor& output)
{
    Tensor tileTensor0 = View(input0, shape0, validShape0, offset0);
    Tensor tileTensor1 = View(input1, shape1, validShape1, offset1);
    auto res = Pow(tileTensor0, tileTensor1);
    Assemble(res, offset2, output);
}

template <size_t shapeDim, size_t inputSize>
static inline void CalcWillBroadcast(bool willBroadcast[][shapeDim], const std::vector<Tensor>& inputs)
{
    constexpr int64_t broadcast = 1;
    for (size_t i = 0; i < shapeDim; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            willBroadcast[j][i] = inputs[j].GetShape()[i] == broadcast && inputs[j ^ 1].GetShape()[i] != broadcast;
        }
    }
}

static void PowOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        constexpr size_t shapeDim = 2;
        constexpr size_t inputSize = 2;
        bool willBroadcast[inputSize][shapeDim] = {};
        CalcWillBroadcast<shapeDim, inputSize>(willBroadcast, inputs);
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        const std::vector<Tensor> tensors = {inputs[0], inputs[1], outputs[0]};
        auto args = static_cast<const PowOpFuncArgs*>(opArgs);
        const int64_t firstViewShape = args->viewShape_[0];
        const int64_t secondViewShape = args->viewShape_[1];
        const std::vector<int64_t> shape0 = {willBroadcast[0][0] ? 1 : firstViewShape,
                                             willBroadcast[0][1] ? 1 : secondViewShape};
        const std::vector<int64_t> shape1 = {willBroadcast[1][0] ? 1 : firstViewShape,
                                             willBroadcast[1][1] ? 1 : secondViewShape};
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                std::vector<SymbolicScalar> offset0 = {
                    willBroadcast[0][0] ? SymbolicScalar(0) : bIdx * firstViewShape,
                    willBroadcast[0][1] ? SymbolicScalar(0) : sIdx * secondViewShape};
                std::vector<SymbolicScalar> offset1 = {
                    willBroadcast[1][0] ? SymbolicScalar(0) : bIdx * firstViewShape,
                    willBroadcast[1][1] ? SymbolicScalar(0) : sIdx * secondViewShape};
                std::vector<SymbolicScalar> offset2 = {bIdx * firstViewShape, sIdx * secondViewShape};
                std::vector<SymbolicScalar> validShape0 = {
                    willBroadcast[0][0] ? SymbolicScalar(1) :
                                          std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                    willBroadcast[0][1] ? SymbolicScalar(1) :
                                          std::min(secondDim - sIdx * secondViewShape, secondViewShape)};
                std::vector<SymbolicScalar> validShape1 = {
                    willBroadcast[1][0] ? SymbolicScalar(1) :
                                          std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                    willBroadcast[1][1] ? SymbolicScalar(1) :
                                          std::min(secondDim - sIdx * secondViewShape, secondViewShape)};
                TileShape::Current().SetVecTile(args->tileShape_);
                DoPow(shape0, shape1, validShape0, validShape1, offset0, offset1, offset2, inputs[0], inputs[1],
                      outputs[0]);
            }
        }
    }
}

static void PowOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        constexpr size_t shapeDim = 3;
        constexpr size_t inputSize = 2;
        bool willBroadcast[inputSize][shapeDim] = {};
        CalcWillBroadcast<shapeDim, inputSize>(willBroadcast, inputs);
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        SymbolicScalar thirdDim = std::max(inputs[0].GetShape()[2], inputs[1].GetShape()[2]);
        const std::vector<Tensor> tensors = {inputs[0], inputs[1], outputs[0]};
        auto args = static_cast<const PowOpFuncArgs*>(opArgs);
        const int64_t firstViewShape = args->viewShape_[0];
        const int64_t secondViewShape = args->viewShape_[1];
        const int64_t thirdViewShape = args->viewShape_[2];
        const std::vector<int64_t> shape0 = {willBroadcast[0][0] ? 1 : firstViewShape,
                                             willBroadcast[0][1] ? 1 : secondViewShape,
                                             willBroadcast[0][2] ? 1 : thirdViewShape};
        const std::vector<int64_t> shape1 = {willBroadcast[1][0] ? 1 : firstViewShape,
                                             willBroadcast[1][1] ? 1 : secondViewShape,
                                             willBroadcast[1][2] ? 1 : thirdViewShape};
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int mloop = CeilDiv(thirdDim, thirdViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                {
                    std::vector<SymbolicScalar> offset0 = {
                        willBroadcast[0][0] ? SymbolicScalar(0) : bIdx * firstViewShape,
                        willBroadcast[0][1] ? SymbolicScalar(0) : sIdx * secondViewShape,
                        willBroadcast[0][2] ? SymbolicScalar(0) : mIdx * thirdViewShape};
                    std::vector<SymbolicScalar> offset1 = {
                        willBroadcast[1][0] ? SymbolicScalar(0) : bIdx * firstViewShape,
                        willBroadcast[1][1] ? SymbolicScalar(0) : sIdx * secondViewShape,
                        willBroadcast[1][2] ? SymbolicScalar(0) : mIdx * thirdViewShape};
                    std::vector<SymbolicScalar> offset2 = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                           mIdx * thirdViewShape};
                    std::vector<SymbolicScalar> validShape0 = {
                        willBroadcast[0][0] ? SymbolicScalar(1) :
                                              std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        willBroadcast[0][1] ? SymbolicScalar(1) :
                                              std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                        willBroadcast[0][2] ? SymbolicScalar(1) :
                                              std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape)};
                    std::vector<SymbolicScalar> validShape1 = {
                        willBroadcast[1][0] ? SymbolicScalar(1) :
                                              std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        willBroadcast[1][1] ? SymbolicScalar(1) :
                                              std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                        willBroadcast[1][2] ? SymbolicScalar(1) :
                                              std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape)};
                    TileShape::Current().SetVecTile(args->tileShape_);
                    DoPow(shape0, shape1, validShape0, validShape1, offset0, offset1, offset2, inputs[0], inputs[1],
                          outputs[0]);
                }
            }
        }
    }
}

static void PowOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        constexpr size_t shapeDim = 4;
        constexpr size_t inputSize = 2;
        bool willBroadcast[inputSize][shapeDim] = {};
        CalcWillBroadcast<shapeDim, inputSize>(willBroadcast, inputs);
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        SymbolicScalar thirdDim = std::max(inputs[0].GetShape()[2], inputs[1].GetShape()[2]);
        SymbolicScalar fourthDim = std::max(inputs[0].GetShape()[3], inputs[1].GetShape()[3]);
        const std::vector<Tensor> tensors = {inputs[0], inputs[1], outputs[0]};
        auto args = static_cast<const PowOpFuncArgs*>(opArgs);
        const int64_t firstViewShape = args->viewShape_[0];
        const int64_t secondViewShape = args->viewShape_[1];
        const int64_t thirdViewShape = args->viewShape_[2];
        const int64_t fourthViewShape = args->viewShape_[3];
        const std::vector<int64_t> shape0 = {
            willBroadcast[0][0] ? 1 : firstViewShape, willBroadcast[0][1] ? 1 : secondViewShape,
            willBroadcast[0][2] ? 1 : thirdViewShape, willBroadcast[0][3] ? 1 : fourthViewShape};
        const std::vector<int64_t> shape1 = {
            willBroadcast[1][0] ? 1 : firstViewShape, willBroadcast[1][1] ? 1 : secondViewShape,
            willBroadcast[1][2] ? 1 : thirdViewShape, willBroadcast[1][3] ? 1 : fourthViewShape};
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
                        std::vector<SymbolicScalar> offset0 = {
                            willBroadcast[0][0] ? SymbolicScalar(0) : bIdx * firstViewShape,
                            willBroadcast[0][1] ? SymbolicScalar(0) : sIdx * secondViewShape,
                            willBroadcast[0][2] ? SymbolicScalar(0) : mIdx * thirdViewShape,
                            willBroadcast[0][3] ? SymbolicScalar(0) : nIdx * fourthViewShape};
                        std::vector<SymbolicScalar> offset1 = {
                            willBroadcast[1][0] ? SymbolicScalar(0) : bIdx * firstViewShape,
                            willBroadcast[1][1] ? SymbolicScalar(0) : sIdx * secondViewShape,
                            willBroadcast[1][2] ? SymbolicScalar(0) : mIdx * thirdViewShape,
                            willBroadcast[1][3] ? SymbolicScalar(0) : nIdx * fourthViewShape};
                        std::vector<SymbolicScalar> offset2 = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                               mIdx * thirdViewShape, nIdx * fourthViewShape};
                        std::vector<SymbolicScalar> validShape0 = {
                            willBroadcast[0][0] ? SymbolicScalar(1) :
                                                  std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            willBroadcast[0][1] ? SymbolicScalar(1) :
                                                  std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            willBroadcast[0][2] ? SymbolicScalar(1) :
                                                  std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                            willBroadcast[0][3] ? SymbolicScalar(1) :
                                                  std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)};
                        std::vector<SymbolicScalar> validShape1 = {
                            willBroadcast[1][0] ? SymbolicScalar(1) :
                                                  std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            willBroadcast[1][1] ? SymbolicScalar(1) :
                                                  std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            willBroadcast[1][2] ? SymbolicScalar(1) :
                                                  std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                            willBroadcast[1][3] ? SymbolicScalar(1) :
                                                  std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)};
                        TileShape::Current().SetVecTile(args->tileShape_);
                        DoPow(shape0, shape1, validShape0, validShape1, offset0, offset1, offset2, inputs[0], inputs[1],
                              outputs[0]);
                    }
                }
            }
        }
    }
}

class PowOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<PowOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestPow, PowOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<PowOpMetaData>(
                             {PowOperationExeFunc2Dims, PowOperationExeFunc3Dims, PowOperationExeFunc4Dims}, "Pow")));

TEST_P(PowOperationTest, TestPow)
{
    auto test_data = GetParam().test_data_;
    auto args = PowOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<PowOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
