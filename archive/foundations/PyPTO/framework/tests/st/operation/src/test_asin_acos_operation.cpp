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
 * \file test_asin_acos_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {

enum class AsinAcosOp { Asin, Acos };

struct AsinAcosOpFuncArgs : public OpFuncArgs {
    AsinAcosOpFuncArgs(AsinAcosOp op, const std::vector<int64_t>& viewShape, const std::vector<int64_t>& tileShape)
        : op_(op), viewShape_(viewShape), tileShape_(tileShape)
    {}

    AsinAcosOp op_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct AsinAcosOpMetaData {
    explicit AsinAcosOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static inline Tensor ApplyAsinAcosOp(AsinAcosOp op, const Tensor& t0)
{
    switch (op) {
        case AsinAcosOp::Asin:
            return Asin(t0);
        case AsinAcosOp::Acos:
            return Acos(t0);
        default:
            throw std::invalid_argument("Unsupported asin/acos operation");
    }
}

static void AsinAcosOpOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        if (inputs[0].GetShape().size() == 1) {
            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            const struct AsinAcosOpFuncArgs* args = static_cast<const AsinAcosOpFuncArgs*>(opArgs);
            const int firstViewShape = args->viewShape_[0];
            const int bloop = CeilDiv(firstDim, firstViewShape);

            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                std::vector<SymbolicScalar> offset = {bIdx * args->viewShape_[0]};
                auto viewTensor = View(inputs[0], args->viewShape_,
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape)}, offset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = ApplyAsinAcosOp(args->op_, viewTensor);
                Assemble(res, offset, outputs[0]);
            }
        } else {
            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            SymbolicScalar secondDim = inputs[0].GetShape()[1];
            const struct AsinAcosOpFuncArgs* args = static_cast<const AsinAcosOpFuncArgs*>(opArgs);
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];
            const int bloop = CeilDiv(firstDim, firstViewShape);
            const int sloop = CeilDiv(secondDim, secondViewShape);

            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape};
                    auto viewTensor = View(inputs[0], args->viewShape_,
                                           {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                            std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                           offset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = ApplyAsinAcosOp(args->op_, viewTensor);
                    Assemble(res, offset, outputs[0]);
                }
            }
        }
    }
}

static void AsinAcosOpOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        const struct AsinAcosOpFuncArgs* args = static_cast<const AsinAcosOpFuncArgs*>(opArgs);
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
                    std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                          nIdx * thirdViewShape};
                    auto viewTensor = View(inputs[0], args->viewShape_,
                                           {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                           offset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = ApplyAsinAcosOp(args->op_, viewTensor);
                    Assemble(res, offset, outputs[0]);
                }
            }
        }
    }
}

static void AsinAcosOpOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        auto args = static_cast<const AsinAcosOpFuncArgs*>(opArgs);
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
                        std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                              mIdx * thirdViewShape, nIdx * fourthViewShape};
                        auto viewTensor = View(inputs[0], args->viewShape_,
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                               offset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = ApplyAsinAcosOp(args->op_, viewTensor);
                        Assemble(res, offset, outputs[0]);
                    }
                }
            }
        }
    }
}

class AsinOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<AsinAcosOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestAsin, AsinOperationTest,
    ::testing::ValuesIn(tile_fwk::test_operation::GetOpMetaData<AsinAcosOpMetaData>(
        {AsinAcosOpOperationExeFunc2Dims, AsinAcosOpOperationExeFunc3Dims, AsinAcosOpOperationExeFunc4Dims}, "Asin")));

TEST_P(AsinOperationTest, TestAsin)
{
    auto test_data = GetParam().test_data_;
    auto args = AsinAcosOpFuncArgs(AsinAcosOp::Asin, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = tile_fwk::test_operation::CreateTestCaseDesc<AsinAcosOpMetaData>(GetParam(), &args);
    tile_fwk::test_operation::TestExecutor::runTest(testCase);
}

class AcosOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<AsinAcosOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestAcos, AcosOperationTest,
    ::testing::ValuesIn(tile_fwk::test_operation::GetOpMetaData<AsinAcosOpMetaData>(
        {AsinAcosOpOperationExeFunc2Dims, AsinAcosOpOperationExeFunc3Dims, AsinAcosOpOperationExeFunc4Dims}, "Acos")));

TEST_P(AcosOperationTest, TestAcos)
{
    auto test_data = GetParam().test_data_;
    auto args = AsinAcosOpFuncArgs(AsinAcosOp::Acos, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = tile_fwk::test_operation::CreateTestCaseDesc<AsinAcosOpMetaData>(GetParam(), &args);
    tile_fwk::test_operation::TestExecutor::runTest(testCase);
}

} // namespace
