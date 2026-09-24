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
 * \file test_onehot_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct OneHotOpFuncArgs : public OpFuncArgs {
    OneHotOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, int numClasses)
        : viewShape_(viewShape), tileShape_(tileShape), numClasses_(numClasses)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    int numClasses_;
};

struct OneHotOpMetaData {
    explicit OneHotOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void OneHotOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        const struct OneHotOpFuncArgs* args = static_cast<const OneHotOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = args->numClasses_;
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                auto tileTensor = View(inputs[0], {firstViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape)},
                                       {bIdx * firstViewShape});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = OneHot(tileTensor, args->numClasses_);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void OneHotOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        const struct OneHotOpFuncArgs* args = static_cast<const OneHotOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = args->numClasses_;
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
                    auto tileTensor = View(inputs[0], {firstViewShape, secondViewShape},
                                           {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                            std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                           {bIdx * firstViewShape, sIdx * secondViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = OneHot(tileTensor, args->numClasses_);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void OneHotOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        auto args = static_cast<const OneHotOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = args->numClasses_;
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
                        Tensor tileTensor = View(
                            inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                             std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                             std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = OneHot(tileTensor, args->numClasses_);
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

class OneHotOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<OneHotOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestOneHot, OneHotOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<OneHotOpMetaData>(
                             {OneHotOperationExeFunc2Dims, OneHotOperationExeFunc3Dims, OneHotOperationExeFunc4Dims},
                             "OneHot")));

TEST_P(OneHotOperationTest, TestOneHot)
{
    auto test_data = GetParam().test_data_;
    int numClasses = GetValueByName<int>(test_data, "num_classes");
    auto args = OneHotOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), numClasses);
    auto testCase = CreateTestCaseDesc<OneHotOpMetaData>(GetParam(), &args);
    std::vector<OpFunc> func{OneHotOperationExeFunc2Dims, OneHotOperationExeFunc3Dims, OneHotOperationExeFunc4Dims};
    int dim = testCase.inputTensors[0].GetShape().size();
    ASSERT(dim >= 1 && dim <= 3) << "unsupport input dim";
    testCase.opFunc = func[dim - 1];
    TestExecutor::runTest(testCase);
}
} // namespace
