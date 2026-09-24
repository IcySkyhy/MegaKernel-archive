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
 * \file test_sincos_operation.cpp
 * \brief Support for both Sin and Cos operations using an enum-based approach.
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {

// 定义支持的操作类型
enum class SinCosOp { SIN, COS };

struct SinCosOpFuncArgs : public OpFuncArgs {
    SinCosOpFuncArgs(SinCosOp op, const std::vector<int64_t>& viewShape, const std::vector<int64_t>& tileShape)
        : op_(op), viewShape_(viewShape), tileShape_(tileShape)
    {}

    SinCosOp op_; // 添加操作类型字段
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct SinCosOpMetaData {
    explicit SinCosOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

// 定义一个函数来根据枚举应用对应的操作
static inline Tensor ApplySinCosOp(SinCosOp op, const Tensor& input)
{
    switch (op) {
        case SinCosOp::SIN:
            return Sin(input);
        case SinCosOp::COS:
            return Cos(input);
        default:
            throw std::invalid_argument("Unsupported trigonometric operation");
    }
}

static void SinCosOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        if (inputs[0].GetShape().size() == 1) {
            const struct SinCosOpFuncArgs* args = static_cast<const SinCosOpFuncArgs*>(opArgs);
            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            const int firstViewShape = args->viewShape_[0];
            int loop[] = {CeilDiv(firstDim, firstViewShape)};
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[0]))
            {
                std::vector<SymbolicScalar> offset = {bIdx * args->viewShape_[0]};
                auto viewTensor = View(inputs[0], args->viewShape_,
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape)}, offset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = ApplySinCosOp(args->op_, viewTensor);
                Assemble(res, offset, outputs[0]);
            }
        } else {
            auto args = static_cast<const SinCosOpFuncArgs*>(opArgs);
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];

            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            SymbolicScalar secondDim = inputs[0].GetShape()[1];

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
                    // 使用 ApplySinCosOp 函数，根据 args->op_ 的值动态决定是 sin 还是 cos
                    auto res = ApplySinCosOp(args->op_, tileTensor);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void SinCosOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        auto args = static_cast<const SinCosOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];

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
                    auto res = ApplySinCosOp(args->op_, tileTensor);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void SinCosOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        auto args = static_cast<const SinCosOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];

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
                        auto res = ApplySinCosOp(args->op_, tileTensor0);
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

// 为每种操作创建独立的测试类
class SinOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<SinCosOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestSin, SinOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<SinCosOpMetaData>(
                             {SinCosOperationExeFunc2Dims, SinCosOperationExeFunc3Dims, SinCosOperationExeFunc4Dims},
                             "Sin"))); // 注意这里的字符串参数是 "Sin"

TEST_P(SinOperationTest, TestSin)
{
    auto test_data = GetParam().test_data_;
    // 创建 SinCosOpFuncArgs 时传入 SinCosOp::SIN
    auto args = SinCosOpFuncArgs(SinCosOp::SIN, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<SinCosOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

class CosOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<SinCosOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestCos, CosOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<SinCosOpMetaData>(
                             {SinCosOperationExeFunc2Dims, SinCosOperationExeFunc3Dims, SinCosOperationExeFunc4Dims},
                             "Cos"))); // 注意这里的字符串参数是 "Cos"

TEST_P(CosOperationTest, TestCos)
{
    auto test_data = GetParam().test_data_;
    // 创建 SinCosOpFuncArgs 时传入 SinCosOp::COS
    auto args = SinCosOpFuncArgs(SinCosOp::COS, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<SinCosOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

} // namespace
