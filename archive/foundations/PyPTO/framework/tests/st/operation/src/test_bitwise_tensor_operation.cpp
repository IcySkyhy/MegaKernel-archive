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
 * \file test_bitwise_tensor_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {

enum class BitwiseOp { AND, OR, XOR };

struct BitwiseOpFuncArgs : public OpFuncArgs {
    BitwiseOpFuncArgs(BitwiseOp op, const std::vector<int64_t>& viewShape, const std::vector<int64_t>& tileShape)
        : op_(op), viewShape_(viewShape), tileShape_(tileShape)
    {}

    BitwiseOp op_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct BitwiseOpMetaData {
    explicit BitwiseOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static inline Tensor ApplyBitwiseOp(BitwiseOp op, const Tensor& t0, const Tensor& t1)
{
    switch (op) {
        case BitwiseOp::AND:
            return BitwiseAnd(t0, t1);
        case BitwiseOp::OR:
            return BitwiseOr(t0, t1);
        case BitwiseOp::XOR:
            return BitwiseXor(t0, t1);
        default:
            throw std::invalid_argument("Unsupported bitwise operation");
    }
}

static void BitwiseOpOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto args = static_cast<const BitwiseOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int broadcastFlag = 1;

        SymbolicScalar firstDim, secondDim;

        firstDim = outputs[0].GetShape()[0];
        secondDim = outputs[0].GetShape()[1];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                auto fullSize0 = std::min(firstDim - bIdx * firstViewShape, firstViewShape);
                auto fullSize1 = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                auto fullOffset0 = bIdx * firstViewShape;
                auto fullOffset1 = sIdx * secondViewShape;

                Tensor tileTensor0, tileTensor1;

                bool input0_bcast_dim0 = (inputs[0].GetShape()[0] == broadcastFlag);
                bool input0_bcast_dim1 = (inputs[0].GetShape()[1] == broadcastFlag);
                bool input1_bcast_dim0 = (inputs[1].GetShape()[0] == broadcastFlag);
                bool input1_bcast_dim1 = (inputs[1].GetShape()[1] == broadcastFlag);

                // 构造 input0 的 tile
                if (input0_bcast_dim0 && !input0_bcast_dim1) {
                    tileTensor0 = View(inputs[0], {1, secondViewShape}, {1, fullSize1}, {0, fullOffset1});
                } else if (!input0_bcast_dim0 && input0_bcast_dim1) {
                    tileTensor0 = View(inputs[0], {firstViewShape, 1}, {fullSize0, 1}, {fullOffset0, 0});
                } else {
                    tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape}, {fullSize0, fullSize1},
                                       {fullOffset0, fullOffset1});
                }

                // 构造 input1 的 tile
                if (input1_bcast_dim0 && !input1_bcast_dim1) {
                    tileTensor1 = View(inputs[1], {1, secondViewShape}, {1, fullSize1}, {0, fullOffset1});
                } else if (!input1_bcast_dim0 && input1_bcast_dim1) {
                    tileTensor1 = View(inputs[1], {firstViewShape, 1}, {fullSize0, 1}, {fullOffset0, 0});
                } else {
                    tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape}, {fullSize0, fullSize1},
                                       {fullOffset0, fullOffset1});
                }

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = ApplyBitwiseOp(args->op_, tileTensor0, tileTensor1);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void BitwiseOpOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto args = static_cast<const BitwiseOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int broadcastFlag = 1;

        SymbolicScalar firstDim, secondDim, thirdDim;

        firstDim = outputs[0].GetShape()[0];
        secondDim = outputs[0].GetShape()[1];
        thirdDim = outputs[0].GetShape()[2];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    auto fullSize0 = std::min(firstDim - bIdx * firstViewShape, firstViewShape);
                    auto fullSize1 = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                    auto fullSize2 = std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape);
                    auto fullOffset0 = bIdx * firstViewShape;
                    auto fullOffset1 = sIdx * secondViewShape;
                    auto fullOffset2 = nIdx * thirdViewShape;

                    bool i0_b0 = (inputs[0].GetShape()[0] == broadcastFlag);
                    bool i0_b1 = (inputs[0].GetShape()[1] == broadcastFlag);
                    bool i0_b2 = (inputs[0].GetShape()[2] == broadcastFlag);

                    bool i1_b0 = (inputs[1].GetShape()[0] == broadcastFlag);
                    bool i1_b1 = (inputs[1].GetShape()[1] == broadcastFlag);
                    bool i1_b2 = (inputs[1].GetShape()[2] == broadcastFlag);

                    Tensor tileTensor0, tileTensor1;
                    // input0
                    if (i0_b0 && !i0_b1 && !i0_b2) {
                        tileTensor0 = View(inputs[0], {1, secondViewShape, thirdViewShape}, {1, fullSize1, fullSize2},
                                           {0, fullOffset1, fullOffset2});
                    } else if (!i0_b0 && i0_b1 && !i0_b2) {
                        tileTensor0 = View(inputs[0], {firstViewShape, 1, thirdViewShape}, {fullSize0, 1, fullSize2},
                                           {fullOffset0, 0, fullOffset2});
                    } else if (!i0_b0 && !i0_b1 && i0_b2) {
                        tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, 1}, {fullSize0, fullSize1, 1},
                                           {fullOffset0, fullOffset1, 0});
                    } else {
                        tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                                           {fullSize0, fullSize1, fullSize2}, {fullOffset0, fullOffset1, fullOffset2});
                    }
                    // input1
                    if (i1_b0 && !i1_b1 && !i1_b2) {
                        tileTensor1 = View(inputs[1], {1, secondViewShape, thirdViewShape}, {1, fullSize1, fullSize2},
                                           {0, fullOffset1, fullOffset2});
                    } else if (!i1_b0 && i1_b1 && !i1_b2) {
                        tileTensor1 = View(inputs[1], {firstViewShape, 1, thirdViewShape}, {fullSize0, 1, fullSize2},
                                           {fullOffset0, 0, fullOffset2});
                    } else if (!i1_b0 && !i1_b1 && i1_b2) {
                        tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape, 1}, {fullSize0, fullSize1, 1},
                                           {fullOffset0, fullOffset1, 0});
                    } else {
                        tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape, thirdViewShape},
                                           {fullSize0, fullSize1, fullSize2}, {fullOffset0, fullOffset1, fullOffset2});
                    }

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = ApplyBitwiseOp(args->op_, tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void BitwiseOpOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        auto args = static_cast<const BitwiseOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        const int broadcastFlag = 1;

        SymbolicScalar firstDim, secondDim, thirdDim, fourthDim;

        firstDim = outputs[0].GetShape()[0];
        secondDim = outputs[0].GetShape()[1];
        thirdDim = outputs[0].GetShape()[2];
        fourthDim = outputs[0].GetShape()[3];

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
                        auto fullSize0 = std::min(firstDim - bIdx * firstViewShape, firstViewShape);
                        auto fullSize1 = std::min(secondDim - sIdx * secondViewShape, secondViewShape);
                        auto fullSize2 = std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape);
                        auto fullSize3 = std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape);
                        auto fullOffset0 = bIdx * firstViewShape;
                        auto fullOffset1 = sIdx * secondViewShape;
                        auto fullOffset2 = mIdx * thirdViewShape;
                        auto fullOffset3 = nIdx * fourthViewShape;

                        bool i0_b0 = (inputs[0].GetShape()[0] == broadcastFlag);
                        bool i0_b1 = (inputs[0].GetShape()[1] == broadcastFlag);
                        bool i0_b2 = (inputs[0].GetShape()[2] == broadcastFlag);
                        bool i0_b3 = (inputs[0].GetShape()[3] == broadcastFlag);

                        bool i1_b0 = (inputs[1].GetShape()[0] == broadcastFlag);
                        bool i1_b1 = (inputs[1].GetShape()[1] == broadcastFlag);
                        bool i1_b2 = (inputs[1].GetShape()[2] == broadcastFlag);
                        bool i1_b3 = (inputs[1].GetShape()[3] == broadcastFlag);

                        Tensor tileTensor0, tileTensor1;
                        // input0
                        if (i0_b0 && !i0_b1 && !i0_b2 && !i0_b3) {
                            tileTensor0 = View(inputs[0], {1, secondViewShape, thirdViewShape, fourthViewShape},
                                               {1, fullSize1, fullSize2, fullSize3},
                                               {0, fullOffset1, fullOffset2, fullOffset3});
                        } else if (!i0_b0 && i0_b1 && !i0_b2 && !i0_b3) {
                            tileTensor0 = View(inputs[0], {firstViewShape, 1, thirdViewShape, fourthViewShape},
                                               {fullSize0, 1, fullSize2, fullSize3},
                                               {fullOffset0, 0, fullOffset2, fullOffset3});
                        } else if (!i0_b0 && !i0_b1 && i0_b2 && !i0_b3) {
                            tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, 1, fourthViewShape},
                                               {fullSize0, fullSize1, 1, fullSize3},
                                               {fullOffset0, fullOffset1, 0, fullOffset3});
                        } else if (!i0_b0 && !i0_b1 && !i0_b2 && i0_b3) {
                            tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, 1},
                                               {fullSize0, fullSize1, fullSize2, 1},
                                               {fullOffset0, fullOffset1, fullOffset2, 0});
                        } else {
                            tileTensor0 = View(inputs[0],
                                               {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                               {fullSize0, fullSize1, fullSize2, fullSize3},
                                               {fullOffset0, fullOffset1, fullOffset2, fullOffset3});
                        }
                        // input1
                        if (i1_b0 && !i1_b1 && !i1_b2 && !i1_b3) {
                            tileTensor1 = View(inputs[1], {1, secondViewShape, thirdViewShape, fourthViewShape},
                                               {1, fullSize1, fullSize2, fullSize3},
                                               {0, fullOffset1, fullOffset2, fullOffset3});
                        } else if (!i1_b0 && i1_b1 && !i1_b2 && !i1_b3) {
                            tileTensor1 = View(inputs[1], {firstViewShape, 1, thirdViewShape, fourthViewShape},
                                               {fullSize0, 1, fullSize2, fullSize3},
                                               {fullOffset0, 0, fullOffset2, fullOffset3});
                        } else if (!i1_b0 && !i1_b1 && i1_b2 && !i1_b3) {
                            tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape, 1, fourthViewShape},
                                               {fullSize0, fullSize1, 1, fullSize3},
                                               {fullOffset0, fullOffset1, 0, fullOffset3});
                        } else if (!i1_b0 && !i1_b1 && !i1_b2 && i1_b3) {
                            tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape, thirdViewShape, 1},
                                               {fullSize0, fullSize1, fullSize2, 1},
                                               {fullOffset0, fullOffset1, fullOffset2, 0});
                        } else {
                            tileTensor1 = View(inputs[1],
                                               {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                               {fullSize0, fullSize1, fullSize2, fullSize3},
                                               {fullOffset0, fullOffset1, fullOffset2, fullOffset3});
                        }

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = ApplyBitwiseOp(args->op_, tileTensor0, tileTensor1);
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

class BitwiseAndOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<BitwiseOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestBitwiseAnd, BitwiseAndOperationTest,
                         ::testing::ValuesIn(tile_fwk::test_operation::GetOpMetaData<BitwiseOpMetaData>(
                             {BitwiseOpOperationExeFunc2Dims, BitwiseOpOperationExeFunc3Dims,
                              BitwiseOpOperationExeFunc4Dims},
                             "BitwiseAnd")));

TEST_P(BitwiseAndOperationTest, TestBitwiseAnd)
{
    auto test_data = GetParam().test_data_;
    auto args = BitwiseOpFuncArgs(BitwiseOp::AND, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = tile_fwk::test_operation::CreateTestCaseDesc<BitwiseOpMetaData>(GetParam(), &args);
    tile_fwk::test_operation::TestExecutor::runTest(testCase);
}

class BitwiseOrOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<BitwiseOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestBitwiseOr, BitwiseOrOperationTest,
                         ::testing::ValuesIn(tile_fwk::test_operation::GetOpMetaData<BitwiseOpMetaData>(
                             {BitwiseOpOperationExeFunc2Dims, BitwiseOpOperationExeFunc3Dims,
                              BitwiseOpOperationExeFunc4Dims},
                             "BitwiseOr")));

TEST_P(BitwiseOrOperationTest, TestBitwiseOr)
{
    auto test_data = GetParam().test_data_;
    auto args = BitwiseOpFuncArgs(BitwiseOp::OR, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = tile_fwk::test_operation::CreateTestCaseDesc<BitwiseOpMetaData>(GetParam(), &args);
    tile_fwk::test_operation::TestExecutor::runTest(testCase);
}

class BitwiseXorOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<BitwiseOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestBitwiseXor, BitwiseXorOperationTest,
                         ::testing::ValuesIn(tile_fwk::test_operation::GetOpMetaData<BitwiseOpMetaData>(
                             {BitwiseOpOperationExeFunc2Dims, BitwiseOpOperationExeFunc3Dims,
                              BitwiseOpOperationExeFunc4Dims},
                             "BitwiseXor")));

TEST_P(BitwiseXorOperationTest, TestBitwiseXor)
{
    auto test_data = GetParam().test_data_;
    auto args = BitwiseOpFuncArgs(BitwiseOp::XOR, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = tile_fwk::test_operation::CreateTestCaseDesc<BitwiseOpMetaData>(GetParam(), &args);
    tile_fwk::test_operation::TestExecutor::runTest(testCase);
}

} // namespace
