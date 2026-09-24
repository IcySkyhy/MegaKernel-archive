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
 * \file test_atan2_operation.cpp
 * \brief
 */
#include <nlohmann/json.hpp>
#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct Atan2OpFuncArgs : public OpFuncArgs {
    Atan2OpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct Atan2OpMetaData {
    explicit Atan2OpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

Shape GetBroadCastViewShape(const Tensor& self, const Tensor& other, const Shape& viewShape)
{
    ASSERT(self.GetShape().size() == other.GetShape().size());
    Shape result = viewShape;
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        int64_t selfDim = self.GetShape()[i];
        int64_t otherDim = other.GetShape()[i];
        if (selfDim != otherDim && selfDim == 1 && otherDim != 1) {
            result[i] = 1;
        } else {
            result[i] = std::min(selfDim, viewShape[i]);
        }
    }
    return result;
}

std::vector<int64_t> GetBroadCastOffsetRatio(const Tensor& self, const Tensor& other, const Shape& viewShape)
{
    ASSERT(self.GetShape().size() == other.GetShape().size());
    Shape result(viewShape.size(), 1);
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        int64_t selfDim = self.GetShape()[i];
        int64_t otherDim = other.GetShape()[i];
        if (selfDim != otherDim && selfDim == 1 && otherDim != 1) {
            result[i] = 0;
        }
    }
    return result;
}

void Atan2OperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        if (inputs[0].GetShape().size() == 1) {
            SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
            auto args = static_cast<const Atan2OpFuncArgs*>(opArgs);
            const int firstViewShape = args->viewShape_[0];

            const int bloop = CeilDiv(firstDim, firstViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                const Shape& tile0ViewShape = GetBroadCastViewShape(inputs[0], inputs[1], args->viewShape_);
                const std::vector<int64_t>& tile0OffsetRatio = GetBroadCastOffsetRatio(inputs[0], inputs[1],
                                                                                       args->viewShape_);
                const Shape& tile1ViewShape = GetBroadCastViewShape(inputs[1], inputs[0], args->viewShape_);
                const std::vector<int64_t>& tile1OffsetRatio = GetBroadCastOffsetRatio(inputs[1], inputs[0],
                                                                                       args->viewShape_);
                Tensor tileTensor0 = View(inputs[0], {tile0ViewShape[0]},
                                          {std::min(firstDim - bIdx * tile0ViewShape[0], tile0ViewShape[0])},
                                          {bIdx * tile0ViewShape[0] * tile0OffsetRatio[0]});
                Tensor tileTensor1 = View(inputs[1], {tile1ViewShape[0]},
                                          {std::min(firstDim - bIdx * tile1ViewShape[0], tile1ViewShape[0])},
                                          {bIdx * tile1ViewShape[0] * tile1OffsetRatio[0]});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Atan2(tileTensor0, tileTensor1);
                Assemble(res, {bIdx * firstViewShape}, outputs[0]);
            }
        } else {
            SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
            SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
            auto args = static_cast<const Atan2OpFuncArgs*>(opArgs);
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];

            const int bloop = CeilDiv(firstDim, firstViewShape);
            const int sloop = CeilDiv(secondDim, secondViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    const Shape& tile0ViewShape = GetBroadCastViewShape(inputs[0], inputs[1], args->viewShape_);
                    const std::vector<int64_t>& tile0OffsetRatio = GetBroadCastOffsetRatio(inputs[0], inputs[1],
                                                                                           args->viewShape_);
                    const Shape& tile1ViewShape = GetBroadCastViewShape(inputs[1], inputs[0], args->viewShape_);
                    const std::vector<int64_t>& tile1OffsetRatio = GetBroadCastOffsetRatio(inputs[1], inputs[0],
                                                                                           args->viewShape_);
                    Tensor tileTensor0 = View(inputs[0], {tile0ViewShape[0], tile0ViewShape[1]},
                                              {std::min(firstDim - bIdx * tile0ViewShape[0], tile0ViewShape[0]),
                                               std::min(secondDim - sIdx * tile0ViewShape[1], tile0ViewShape[1])},
                                              {bIdx * tile0ViewShape[0] * tile0OffsetRatio[0],
                                               sIdx * tile0ViewShape[1] * tile0OffsetRatio[1]});
                    Tensor tileTensor1 = View(inputs[1], {tile1ViewShape[0], tile1ViewShape[1]},
                                              {std::min(firstDim - bIdx * tile1ViewShape[0], tile1ViewShape[0]),
                                               std::min(secondDim - sIdx * tile1ViewShape[1], tile1ViewShape[1])},
                                              {bIdx * tile1ViewShape[0] * tile1OffsetRatio[0],
                                               sIdx * tile1ViewShape[1] * tile1OffsetRatio[1]});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Atan2(tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                }
            }
        }
    }
}

void Atan2OperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        SymbolicScalar thirdDim = std::max(inputs[0].GetShape()[2], inputs[1].GetShape()[2]);
        auto args = static_cast<const Atan2OpFuncArgs*>(opArgs);
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
                    const Shape& tile0ViewShape = GetBroadCastViewShape(inputs[0], inputs[1], args->viewShape_);
                    const std::vector<int64_t>& tile0OffsetRatio = GetBroadCastOffsetRatio(inputs[0], inputs[1],
                                                                                           args->viewShape_);
                    const Shape& tile1ViewShape = GetBroadCastViewShape(inputs[1], inputs[0], args->viewShape_);
                    const std::vector<int64_t>& tile1OffsetRatio = GetBroadCastOffsetRatio(inputs[1], inputs[0],
                                                                                           args->viewShape_);
                    Tensor tileTensor0 = View(
                        inputs[0], {tile0ViewShape[0], tile0ViewShape[1], tile0ViewShape[2]},
                        {std::min(firstDim - bIdx * tile0ViewShape[0], tile0ViewShape[0]),
                         std::min(secondDim - sIdx * tile0ViewShape[1], tile0ViewShape[1]),
                         std::min(thirdDim - nIdx * tile0ViewShape[2], tile0ViewShape[2])},
                        {bIdx * tile0ViewShape[0] * tile0OffsetRatio[0], sIdx * tile0ViewShape[1] * tile0OffsetRatio[1],
                         nIdx * tile0ViewShape[2] * tile0OffsetRatio[2]});
                    Tensor tileTensor1 = View(
                        inputs[1], {tile1ViewShape[0], tile1ViewShape[1], tile1ViewShape[2]},
                        {std::min(firstDim - bIdx * tile1ViewShape[0], tile1ViewShape[0]),
                         std::min(secondDim - sIdx * tile1ViewShape[1], tile1ViewShape[1]),
                         std::min(thirdDim - nIdx * tile1ViewShape[2], tile1ViewShape[2])},
                        {bIdx * tile1ViewShape[0] * tile1OffsetRatio[0], sIdx * tile1ViewShape[1] * tile1OffsetRatio[1],
                         nIdx * tile1ViewShape[2] * tile1OffsetRatio[2]});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Atan2(tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

void Atan2OperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        SymbolicScalar thirdDim = std::max(inputs[0].GetShape()[2], inputs[1].GetShape()[2]);
        SymbolicScalar fourthDim = std::max(inputs[0].GetShape()[3], inputs[1].GetShape()[3]);
        auto args = static_cast<const Atan2OpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);
        const int kloop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    LOOP("LOOP_L3_kIdx", FunctionType::DYNAMIC_LOOP, kIdx, LoopRange(0, kloop, 1))
                    {
                        const Shape& tile0ViewShape = GetBroadCastViewShape(inputs[0], inputs[1], args->viewShape_);
                        const std::vector<int64_t>& tile0OffsetRatio = GetBroadCastOffsetRatio(inputs[0], inputs[1],
                                                                                               args->viewShape_);
                        const Shape& tile1ViewShape = GetBroadCastViewShape(inputs[1], inputs[0], args->viewShape_);
                        const std::vector<int64_t>& tile1OffsetRatio = GetBroadCastOffsetRatio(inputs[1], inputs[0],
                                                                                               args->viewShape_);
                        Tensor tileTensor0 = View(
                            inputs[0], {tile0ViewShape[0], tile0ViewShape[1], tile0ViewShape[2], tile0ViewShape[3]},
                            {std::min(firstDim - bIdx * tile0ViewShape[0], tile0ViewShape[0]),
                             std::min(secondDim - sIdx * tile0ViewShape[1], tile0ViewShape[1]),
                             std::min(thirdDim - nIdx * tile0ViewShape[2], tile0ViewShape[2]),
                             std::min(fourthDim - kIdx * tile0ViewShape[3], tile0ViewShape[3])},
                            {bIdx * tile0ViewShape[0] * tile0OffsetRatio[0],
                             sIdx * tile0ViewShape[1] * tile0OffsetRatio[1],
                             nIdx * tile0ViewShape[2] * tile0OffsetRatio[2],
                             kIdx * tile0ViewShape[3] * tile0OffsetRatio[3]});
                        Tensor tileTensor1 = View(
                            inputs[1], {tile1ViewShape[0], tile1ViewShape[1], tile1ViewShape[2], tile1ViewShape[3]},
                            {std::min(firstDim - bIdx * tile1ViewShape[0], tile1ViewShape[0]),
                             std::min(secondDim - sIdx * tile1ViewShape[1], tile1ViewShape[1]),
                             std::min(thirdDim - nIdx * tile1ViewShape[2], tile1ViewShape[2]),
                             std::min(fourthDim - kIdx * tile1ViewShape[3], tile1ViewShape[3])},
                            {bIdx * tile1ViewShape[0] * tile1OffsetRatio[0],
                             sIdx * tile1ViewShape[1] * tile1OffsetRatio[1],
                             nIdx * tile1ViewShape[2] * tile1OffsetRatio[2],
                             kIdx * tile1ViewShape[3] * tile1OffsetRatio[3]});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Atan2(tileTensor0, tileTensor1);
                        Assemble(res,
                                 {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                  kIdx * fourthViewShape},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

class Atan2OperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<Atan2OpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestAtan2, Atan2OperationTest,
                         ::testing::ValuesIn(GetOpMetaData<Atan2OpMetaData>(
                             {Atan2OperationExeFunc2Dims, Atan2OperationExeFunc3Dims, Atan2OperationExeFunc4Dims},
                             "Atan2")));

TEST_P(Atan2OperationTest, TestAtan2)
{
    auto test_data = GetParam().test_data_;
    Shape viewShape = GetViewShape(test_data);
    Shape tileShape = GetTileShape(test_data);
    auto args = Atan2OpFuncArgs(viewShape, tileShape);
    auto testCase = CreateTestCaseDesc<Atan2OpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
