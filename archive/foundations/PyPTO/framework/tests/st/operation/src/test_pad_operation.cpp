/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_pad_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct PadOpFuncArgs : public OpFuncArgs {
    PadOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, float padValue)
        : viewShape_(viewShape), tileShape_(tileShape), padValue_(padValue)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    float padValue_;
};

struct PadOpMetaData {
    explicit PadOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void PadOperationExeFunc1Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        const struct PadOpFuncArgs* args = static_cast<const PadOpFuncArgs*>(opArgs);
        // viewshape不切分尾部两轴
        const int firstViewShape = inputs[0].GetShape()[0];
        const int bloop = CeilDiv(firstDim, firstViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            auto tileTensor = View(inputs[0], {firstViewShape},
                                   {std::min(firstDim - bIdx * firstViewShape, firstViewShape)},
                                   {bIdx * firstViewShape});
            TileShape::Current().SetVecTile(args->tileShape_);
            int64_t padRight = outputs[0].GetShape()[0] - inputs[0].GetShape()[0];
            auto res = Pad(tileTensor, {0, padRight}, "constant", Element(tileTensor.GetDataType(), args->padValue_));
            Assemble(res, {bIdx * firstViewShape}, outputs[0]);
        }
    }
}

static void PadOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const struct PadOpFuncArgs* args = static_cast<const PadOpFuncArgs*>(opArgs);
        // viewshape不切分尾部两轴
        const int firstViewShape = inputs[0].GetShape()[0];
        const int secondViewShape = inputs[0].GetShape()[1];
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
                int64_t padRight = outputs[0].GetShape()[1] - inputs[0].GetShape()[1];
                int64_t padBottom = outputs[0].GetShape()[0] - inputs[0].GetShape()[0];
                auto res = Pad(tileTensor, {0, padRight, 0, padBottom}, "constant",
                               Element(tileTensor.GetDataType(), args->padValue_));
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void PadOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        const struct PadOpFuncArgs* args = static_cast<const PadOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        // viewshape不切分尾部两轴
        const int secondViewShape = inputs[0].GetShape()[1];
        const int thirdViewShape = inputs[0].GetShape()[2];
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
                    int64_t padRight = outputs[0].GetShape()[2] - inputs[0].GetShape()[2];
                    int64_t padBottom = outputs[0].GetShape()[1] - inputs[0].GetShape()[1];
                    auto res = Pad(tileTensor, {0, padRight, 0, padBottom}, "constant",
                                   Element(tileTensor.GetDataType(), args->padValue_));
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void PadOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        const struct PadOpFuncArgs* args = static_cast<const PadOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        // viewshape不切分尾部两轴
        const int thirdViewShape = inputs[0].GetShape()[2];
        const int fourthViewShape = inputs[0].GetShape()[3];

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
                        int64_t padRight = outputs[0].GetShape()[3] - inputs[0].GetShape()[3];
                        int64_t padBottom = outputs[0].GetShape()[2] - inputs[0].GetShape()[2];
                        auto res = Pad(tileTensor0, {0, padRight, 0, padBottom}, "constant",
                                       Element(tileTensor0.GetDataType(), args->padValue_));
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

class PadOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<PadOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestPad, PadOperationTest,
    ::testing::ValuesIn(GetOpMetaData<PadOpMetaData>({PadOperationExeFunc2Dims, PadOperationExeFunc3Dims,
                                                      PadOperationExeFunc4Dims, PadOperationExeFunc1Dims},
                                                     "Pad")));

TEST_P(PadOperationTest, TestPad)
{
    auto test_data = GetParam().test_data_;
    std::string padValueType = GetValueByName<std::string>(test_data, "pad_value_type");
    float padValue = 0.0f;
    if (padValueType == "min") {
        padValue = -std::numeric_limits<float>::infinity();
    } else if (padValueType == "max") {
        padValue = std::numeric_limits<float>::infinity();
    } else if (padValueType == "zero") {
        padValue = 0.0f;
    } else if (padValueType == "custom") {
        padValue = GetValueByName<float>(test_data, "pad_value");
    }
    auto args = PadOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), padValue);
    auto testCase = CreateTestCaseDesc<PadOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
