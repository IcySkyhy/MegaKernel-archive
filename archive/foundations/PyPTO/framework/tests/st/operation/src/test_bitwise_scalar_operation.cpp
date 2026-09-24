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
 * \file test_bitwise_scalar_operation.cpp
 * \brief Test for BitwiseAnds / BitwiseOrs / BitwiseXors (tensor op scalar)
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {

enum class BitwiseScalarOp { AND, OR, XOR };

struct BitwiseScalarOpFuncArgs : public OpFuncArgs {
    BitwiseScalarOpFuncArgs(BitwiseScalarOp op, const Element& value, const std::vector<int64_t>& viewShape,
                            const std::vector<int64_t>& tileShape)
        : op_(op), value_(value), viewShape_(viewShape), tileShape_(tileShape)
    {}

    BitwiseScalarOp op_;
    Element value_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct BitwiseScalarOpMetaData {
    explicit BitwiseScalarOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static inline Tensor ApplyBitwiseScalarOp(BitwiseScalarOp op, const Tensor& t, const Element& scalar)
{
    switch (op) {
        case BitwiseScalarOp::AND:
            return BitwiseAnd(t, scalar);
        case BitwiseScalarOp::OR:
            return BitwiseOr(t, scalar);
        case BitwiseScalarOp::XOR:
            return BitwiseXor(t, scalar);
        default:
            throw std::invalid_argument("Unsupported bitwise scalar operation");
    }
}

static void BitwiseScalarOpExeFunc2D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        auto args = static_cast<const BitwiseScalarOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                auto fullSize0 = std::min(firstDim - bIdx * firstViewShape, firstViewShape);
                auto fullSize1 = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape}, {fullSize0, fullSize1},
                                        {bIdx * firstViewShape, sIdx * secondViewShape});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = ApplyBitwiseScalarOp(args->op_, tileTensor0, args->value_);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void BitwiseScalarOpExeFunc3D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        auto args = static_cast<const BitwiseScalarOpFuncArgs*>(opArgs);
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
                    auto fullSize0 = std::min(firstDim - bIdx * firstViewShape, firstViewShape);
                    auto fullSize1 = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                    auto fullSize2 = std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape);
                    auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                                            {fullSize0, fullSize1, fullSize2},
                                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = ApplyBitwiseScalarOp(args->op_, tileTensor0, args->value_);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void BitwiseScalarOpExeFunc4D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        auto args = static_cast<const BitwiseScalarOpFuncArgs*>(opArgs);
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
                        auto fullSize0 = std::min(firstDim - bIdx * firstViewShape, firstViewShape);
                        auto fullSize1 = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                        auto fullSize2 = std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape);
                        auto fullSize3 = std::min(fourthDim - qIdx * fourthViewShape, fourthViewShape);
                        auto tileTensor0 = View(inputs[0],
                                                {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                                {fullSize0, fullSize1, fullSize2, fullSize3},
                                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                                 qIdx * fourthViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = ApplyBitwiseScalarOp(args->op_, tileTensor0, args->value_);
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

class BitwiseAndsOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<BitwiseScalarOpMetaData> {
};

INSTANTIATE_TEST_SUITE_P(TestBitwiseAnds, BitwiseAndsOperationTest,
                         ::testing::ValuesIn(tile_fwk::test_operation::GetOpMetaData<BitwiseScalarOpMetaData>(
                             {BitwiseScalarOpExeFunc2D, BitwiseScalarOpExeFunc3D, BitwiseScalarOpExeFunc4D},
                             "BitwiseAnds")));

TEST_P(BitwiseAndsOperationTest, TestBitwiseAnds)
{
    auto test_data = GetParam().test_data_;
    auto dtype = tile_fwk::test_operation::GetDataType(
        tile_fwk::test_operation::GetValueByName<std::string>(test_data, "scalar_type"));
    Element value(dtype, tile_fwk::test_operation::GetValueByName<float>(test_data, "scalar"));
    auto args = BitwiseScalarOpFuncArgs(BitwiseScalarOp::AND, value, tile_fwk::test_operation::GetViewShape(test_data),
                                        tile_fwk::test_operation::GetTileShape(test_data));
    auto testCase = tile_fwk::test_operation::CreateTestCaseDesc<BitwiseScalarOpMetaData>(GetParam(), &args);
    tile_fwk::test_operation::TestExecutor::runTest(testCase);
}

class BitwiseOrsOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<BitwiseScalarOpMetaData> {
};

INSTANTIATE_TEST_SUITE_P(TestBitwiseOrs, BitwiseOrsOperationTest,
                         ::testing::ValuesIn(tile_fwk::test_operation::GetOpMetaData<BitwiseScalarOpMetaData>(
                             {BitwiseScalarOpExeFunc2D, BitwiseScalarOpExeFunc3D, BitwiseScalarOpExeFunc4D},
                             "BitwiseOrs")));

TEST_P(BitwiseOrsOperationTest, TestBitwiseOrs)
{
    auto test_data = GetParam().test_data_;
    auto dtype = tile_fwk::test_operation::GetDataType(
        tile_fwk::test_operation::GetValueByName<std::string>(test_data, "scalar_type"));
    Element value(dtype, tile_fwk::test_operation::GetValueByName<float>(test_data, "scalar"));
    auto args = BitwiseScalarOpFuncArgs(BitwiseScalarOp::OR, value, tile_fwk::test_operation::GetViewShape(test_data),
                                        tile_fwk::test_operation::GetTileShape(test_data));
    auto testCase = tile_fwk::test_operation::CreateTestCaseDesc<BitwiseScalarOpMetaData>(GetParam(), &args);
    tile_fwk::test_operation::TestExecutor::runTest(testCase);
}

class BitwiseXorsOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<BitwiseScalarOpMetaData> {
};

INSTANTIATE_TEST_SUITE_P(TestBitwiseXors, BitwiseXorsOperationTest,
                         ::testing::ValuesIn(tile_fwk::test_operation::GetOpMetaData<BitwiseScalarOpMetaData>(
                             {BitwiseScalarOpExeFunc2D, BitwiseScalarOpExeFunc3D, BitwiseScalarOpExeFunc4D},
                             "BitwiseXors")));

TEST_P(BitwiseXorsOperationTest, TestBitwiseXors)
{
    auto test_data = GetParam().test_data_;
    auto dtype = tile_fwk::test_operation::GetDataType(
        tile_fwk::test_operation::GetValueByName<std::string>(test_data, "scalar_type"));
    Element value(dtype, tile_fwk::test_operation::GetValueByName<float>(test_data, "scalar"));
    auto args = BitwiseScalarOpFuncArgs(BitwiseScalarOp::XOR, value, tile_fwk::test_operation::GetViewShape(test_data),
                                        tile_fwk::test_operation::GetTileShape(test_data));
    auto testCase = tile_fwk::test_operation::CreateTestCaseDesc<BitwiseScalarOpMetaData>(GetParam(), &args);
    tile_fwk::test_operation::TestExecutor::runTest(testCase);
}

} // namespace
