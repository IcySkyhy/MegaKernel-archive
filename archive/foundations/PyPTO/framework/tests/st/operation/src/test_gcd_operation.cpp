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
 * \file test_gcd_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct GcdOpFuncArgs : public OpFuncArgs {
    GcdOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct GcdOpMetaData {
    explicit GcdOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static std::vector<std::vector<int64_t>> GetBroadcastInfo(const std::vector<Tensor>& inputs, const Tensor& output)
{
    std::vector<std::vector<int64_t>> res;
    for (auto input : inputs) {
        std::vector<int64_t> tmpVec;
        for (size_t i = 0; i < input.GetShape().size(); i++) {
            if (input.GetShape()[i] != output.GetShape()[i]) {
                tmpVec.push_back(1);
            } else {
                tmpVec.push_back(0);
            }
        }
        res.push_back(tmpVec);
    }
    return res;
}

static std::vector<int64_t> GetBroadcastedViewShape(std::vector<int64_t> viewShape, const Tensor& input)
{
    std::vector<int64_t> res;
    for (size_t i = 0; i < viewShape.size(); i++) {
        if (input.GetShape()[i] == 1) {
            res.push_back(1);
        } else {
            res.push_back(viewShape[i]);
        }
    }
    return res;
}

static std::vector<SymbolicScalar> GetBrcedValidShape(std::vector<int64_t>& broadcastInfo,
                                                      std::vector<SymbolicScalar>& indices,
                                                      std::vector<int64_t> inputViewShape, const Tensor& input)
{
    std::vector<SymbolicScalar> res;
    for (size_t i = 0; i < inputViewShape.size(); i++) {
        SymbolicScalar tmp = broadcastInfo[i] == 1 ?
                                 SymbolicScalar(1) :
                                 std::min(input.GetShape()[i] - indices[i] * inputViewShape[i], inputViewShape[i]);
        res.push_back(tmp);
    }
    return res;
}

static std::vector<SymbolicScalar> GetBrcedOffset(std::vector<int64_t>& broadcastInfo,
                                                  std::vector<SymbolicScalar>& indices,
                                                  std::vector<int64_t> inputViewShape)
{
    std::vector<SymbolicScalar> res;
    for (size_t i = 0; i < inputViewShape.size(); i++) {
        SymbolicScalar tmp = broadcastInfo[i] == 1 ? SymbolicScalar(0) : indices[i] * inputViewShape[i];
        res.push_back(tmp);
    }
    return res;
}

static void GcdOperationExeFuncDoubleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    auto args = static_cast<const GcdOpFuncArgs*>(opArgs);

    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        if (outputs[0].GetShape().size() == 1) {
            SymbolicScalar firstLoopVar = outputs[0].GetShape()[0];
            const int firstViewShape = args->viewShape_[0];
            const int bloop = CeilDiv(firstLoopVar, firstViewShape);

            auto broadcastInfo = GetBroadcastInfo(inputs, outputs[0]);
            auto input0ViewShape = GetBroadcastedViewShape(args->viewShape_, inputs[0]);
            auto input1ViewShape = GetBroadcastedViewShape(args->viewShape_, inputs[1]);

            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                Tensor tileTensor0;
                Tensor tileTensor1;
                std::vector<SymbolicScalar> indices = {bIdx};
                auto tensor0ValidShape = GetBrcedValidShape(broadcastInfo[0], indices, input0ViewShape, inputs[0]);
                auto tensor1ValidShape = GetBrcedValidShape(broadcastInfo[1], indices, input1ViewShape, inputs[1]);
                auto tensor0Offset = GetBrcedOffset(broadcastInfo[0], indices, input0ViewShape);
                auto tensor1Offset = GetBrcedOffset(broadcastInfo[1], indices, input1ViewShape);

                tileTensor0 = View(inputs[0], {input0ViewShape[0]}, {tensor0ValidShape[0]}, {tensor0Offset[0]});
                tileTensor1 = View(inputs[1], {input1ViewShape[0]}, {tensor1ValidShape[0]}, {tensor1Offset[0]});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Gcd(tileTensor0, tileTensor1);
                Assemble(res, {bIdx * firstViewShape}, outputs[0]);
            }
        } else {
            SymbolicScalar firstLoopVar = outputs[0].GetShape()[0];
            SymbolicScalar secondLoopVar = outputs[0].GetShape()[1];
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];
            auto broadcastInfo = GetBroadcastInfo(inputs, outputs[0]);
            auto input0ViewShape = GetBroadcastedViewShape(args->viewShape_, inputs[0]);
            auto input1ViewShape = GetBroadcastedViewShape(args->viewShape_, inputs[1]);

            const int bloop = CeilDiv(firstLoopVar, firstViewShape);
            const int sloop = CeilDiv(secondLoopVar, secondViewShape);

            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    Tensor tileTensor0;
                    Tensor tileTensor1;
                    std::vector<SymbolicScalar> indices = {bIdx, sIdx};
                    auto tensor0ValidShape = GetBrcedValidShape(broadcastInfo[0], indices, input0ViewShape, inputs[0]);
                    auto tensor1ValidShape = GetBrcedValidShape(broadcastInfo[1], indices, input1ViewShape, inputs[1]);
                    auto tensor0Offset = GetBrcedOffset(broadcastInfo[0], indices, input0ViewShape);
                    auto tensor1Offset = GetBrcedOffset(broadcastInfo[1], indices, input1ViewShape);

                    tileTensor0 = View(inputs[0], {input0ViewShape[0], input0ViewShape[1]},
                                       {tensor0ValidShape[0], tensor0ValidShape[1]},
                                       {tensor0Offset[0], tensor0Offset[1]});
                    tileTensor1 = View(inputs[1], {input1ViewShape[0], input1ViewShape[1]},
                                       {tensor1ValidShape[0], tensor1ValidShape[1]},
                                       {tensor1Offset[0], tensor1Offset[1]});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Gcd(tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void GcdOperationExeFuncTripleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    auto args = static_cast<const GcdOpFuncArgs*>(opArgs);

    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstLoopVar = outputs[0].GetShape()[0];
        SymbolicScalar secondLoopVar = outputs[0].GetShape()[1];
        SymbolicScalar thirdLoopVar = outputs[0].GetShape()[2];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        auto broadcastInfo = GetBroadcastInfo(inputs, outputs[0]);
        auto input0ViewShape = GetBroadcastedViewShape(args->viewShape_, inputs[0]);
        auto input1ViewShape = GetBroadcastedViewShape(args->viewShape_, inputs[1]);

        const int bloop = CeilDiv(firstLoopVar, firstViewShape);
        const int sloop = CeilDiv(secondLoopVar, secondViewShape);
        const int kloop = CeilDiv(thirdLoopVar, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L1_kIdx", FunctionType::DYNAMIC_LOOP, kIdx, LoopRange(0, kloop, 1))
                {
                    Tensor tileTensor0;
                    Tensor tileTensor1;
                    std::vector<SymbolicScalar> indices = {bIdx, sIdx, kIdx};
                    auto tensor0ValidShape = GetBrcedValidShape(broadcastInfo[0], indices, input0ViewShape, inputs[0]);
                    auto tensor1ValidShape = GetBrcedValidShape(broadcastInfo[1], indices, input1ViewShape, inputs[1]);
                    auto tensor0Offset = GetBrcedOffset(broadcastInfo[0], indices, input0ViewShape);
                    auto tensor1Offset = GetBrcedOffset(broadcastInfo[1], indices, input1ViewShape);

                    tileTensor0 = View(inputs[0], {input0ViewShape[0], input0ViewShape[1], input0ViewShape[2]},
                                       {tensor0ValidShape[0], tensor0ValidShape[1], tensor0ValidShape[2]},
                                       {tensor0Offset[0], tensor0Offset[1], tensor0Offset[2]});
                    tileTensor1 = View(inputs[1], {input1ViewShape[0], input1ViewShape[1], input1ViewShape[2]},
                                       {tensor1ValidShape[0], tensor1ValidShape[1], tensor1ValidShape[2]},
                                       {tensor1Offset[0], tensor1Offset[1], tensor1Offset[2]});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Gcd(tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, kIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void GcdOperationExeFuncQuadraticCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    auto args = static_cast<const GcdOpFuncArgs*>(opArgs);

    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstLoopVar = outputs[0].GetShape()[0];
        SymbolicScalar secondLoopVar = outputs[0].GetShape()[1];
        SymbolicScalar thirdLoopVar = outputs[0].GetShape()[2];
        SymbolicScalar fourthLoopVar = outputs[0].GetShape()[3];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        auto broadcastInfo = GetBroadcastInfo(inputs, outputs[0]);
        auto input0ViewShape = GetBroadcastedViewShape(args->viewShape_, inputs[0]);
        auto input1ViewShape = GetBroadcastedViewShape(args->viewShape_, inputs[1]);

        const int bloop = CeilDiv(firstLoopVar, firstViewShape);
        const int sloop = CeilDiv(secondLoopVar, secondViewShape);
        const int kloop = CeilDiv(thirdLoopVar, thirdViewShape);
        const int mloop = CeilDiv(fourthLoopVar, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L1_kIdx", FunctionType::DYNAMIC_LOOP, kIdx, LoopRange(0, kloop, 1))
                {
                    LOOP("LOOP_L1_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                    {
                        Tensor tileTensor0;
                        Tensor tileTensor1;
                        std::vector<SymbolicScalar> indices = {bIdx, sIdx, kIdx, mIdx};
                        auto tensor0ValidShape = GetBrcedValidShape(broadcastInfo[0], indices, input0ViewShape,
                                                                    inputs[0]);
                        auto tensor1ValidShape = GetBrcedValidShape(broadcastInfo[1], indices, input1ViewShape,
                                                                    inputs[1]);
                        auto tensor0Offset = GetBrcedOffset(broadcastInfo[0], indices, input0ViewShape);
                        auto tensor1Offset = GetBrcedOffset(broadcastInfo[1], indices, input1ViewShape);

                        tileTensor0 = View(
                            inputs[0], {input0ViewShape[0], input0ViewShape[1], input0ViewShape[2], input0ViewShape[3]},
                            {tensor0ValidShape[0], tensor0ValidShape[1], tensor0ValidShape[2], tensor0ValidShape[3]},
                            {tensor0Offset[0], tensor0Offset[1], tensor0Offset[2], tensor0Offset[3]});
                        tileTensor1 = View(
                            inputs[1], {input1ViewShape[0], input1ViewShape[1], input1ViewShape[2], input1ViewShape[3]},
                            {tensor1ValidShape[0], tensor1ValidShape[1], tensor1ValidShape[2], tensor1ValidShape[3]},
                            {tensor1Offset[0], tensor1Offset[1], tensor1Offset[2], tensor1Offset[3]});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Gcd(tileTensor0, tileTensor1);
                        Assemble(res,
                                 {bIdx * firstViewShape, sIdx * secondViewShape, kIdx * thirdViewShape,
                                  mIdx * fourthViewShape},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

static void GcdOperationExeFuncPentaCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    auto args = static_cast<const GcdOpFuncArgs*>(opArgs);

    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstLoopVar = outputs[0].GetShape()[0];
        SymbolicScalar secondLoopVar = outputs[0].GetShape()[1];
        SymbolicScalar thirdLoopVar = outputs[0].GetShape()[2];
        SymbolicScalar fourthLoopVar = outputs[0].GetShape()[3];
        SymbolicScalar fifthLoopVar = outputs[0].GetShape()[4];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        const int fifthViewShape = args->viewShape_[4];
        auto broadcastInfo = GetBroadcastInfo(inputs, outputs[0]);
        auto input0ViewShape = GetBroadcastedViewShape(args->viewShape_, inputs[0]);
        auto input1ViewShape = GetBroadcastedViewShape(args->viewShape_, inputs[1]);

        const int bloop = CeilDiv(firstLoopVar, firstViewShape);
        const int sloop = CeilDiv(secondLoopVar, secondViewShape);
        const int kloop = CeilDiv(thirdLoopVar, thirdViewShape);
        const int mloop = CeilDiv(fourthLoopVar, fourthViewShape);
        const int nloop = CeilDiv(fifthLoopVar, fifthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L1_kIdx", FunctionType::DYNAMIC_LOOP, kIdx, LoopRange(0, kloop, 1))
                {
                    LOOP("LOOP_L1_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                    {
                        LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                        {
                            Tensor tileTensor0;
                            Tensor tileTensor1;
                            std::vector<SymbolicScalar> indices = {bIdx, sIdx, kIdx, mIdx, nIdx};
                            auto tensor0ValidShape = GetBrcedValidShape(broadcastInfo[0], indices, input0ViewShape,
                                                                        inputs[0]);
                            auto tensor1ValidShape = GetBrcedValidShape(broadcastInfo[1], indices, input1ViewShape,
                                                                        inputs[1]);
                            auto tensor0Offset = GetBrcedOffset(broadcastInfo[0], indices, input0ViewShape);
                            auto tensor1Offset = GetBrcedOffset(broadcastInfo[1], indices, input1ViewShape);

                            tileTensor0 = View(inputs[0],
                                               {input0ViewShape[0], input0ViewShape[1], input0ViewShape[2],
                                                input0ViewShape[3], input0ViewShape[4]},
                                               {tensor0ValidShape[0], tensor0ValidShape[1], tensor0ValidShape[2],
                                                tensor0ValidShape[3], tensor0ValidShape[4]},
                                               {tensor0Offset[0], tensor0Offset[1], tensor0Offset[2], tensor0Offset[3],
                                                tensor0Offset[4]});
                            tileTensor1 = View(inputs[1],
                                               {input1ViewShape[0], input1ViewShape[1], input1ViewShape[2],
                                                input1ViewShape[3], input1ViewShape[4]},
                                               {tensor1ValidShape[0], tensor1ValidShape[1], tensor1ValidShape[2],
                                                tensor1ValidShape[3], tensor1ValidShape[4]},
                                               {tensor1Offset[0], tensor1Offset[1], tensor1Offset[2], tensor1Offset[3],
                                                tensor1Offset[4]});
                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Gcd(tileTensor0, tileTensor1);
                            Assemble(res,
                                     {bIdx * firstViewShape, sIdx * secondViewShape, kIdx * thirdViewShape,
                                      mIdx * fourthViewShape, nIdx * fifthViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

class GcdOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<GcdOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestGcd, GcdOperationTest,
    ::testing::ValuesIn(GetOpMetaData<GcdOpMetaData>({GcdOperationExeFuncDoubleCut, GcdOperationExeFuncTripleCut,
                                                      GcdOperationExeFuncQuadraticCut, GcdOperationExeFuncPentaCut},
                                                     "Gcd")));

TEST_P(GcdOperationTest, TestGcd)
{
    auto test_data = GetParam().test_data_;
    auto args = GcdOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<GcdOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
