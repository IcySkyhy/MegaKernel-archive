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
 * \file test_scatter_tensor_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace ScatterOperation {
extern const std::map<std::string, ScatterMode>& GetScatterModeMap();
}
namespace ScatterTensorOperation {
struct ScatterTensorOpFuncArgs : public OpFuncArgs {
    ScatterTensorOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, int axis,
                            ScatterMode reduce)
        : viewShape_(viewShape), tileShape_(tileShape), axis_(axis), reduce_(reduce)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    int axis_;
    ScatterMode reduce_;
};

struct ScatterTensorOpMetaData {
    explicit ScatterTensorOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void ScatterTensorOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        SymbolicScalar self_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar self_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar src_firstDim = inputs[2].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[2].GetShape()[1];
        auto args = static_cast<const ScatterTensorOpFuncArgs*>(opArgs);
        const int64_t firstViewShape = args->viewShape_[0];
        const int64_t secondViewShape = args->viewShape_[1];

        const int64_t bloop = CeilDiv(idx_firstDim, firstViewShape);
        const int64_t sloop = CeilDiv(idx_secondDim, secondViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                        {std::min(self_firstDim - bIdx * firstViewShape, firstViewShape),
                                         std::min(self_secondDim - sIdx * secondViewShape, secondViewShape)},
                                        {bIdx * firstViewShape, sIdx * secondViewShape});
                auto tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                                        {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                                         std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape)},
                                        {bIdx * firstViewShape, sIdx * secondViewShape});
                auto tileTensor2 = View(inputs[2], {firstViewShape, secondViewShape},
                                        {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                         std::min(src_secondDim - sIdx * secondViewShape, secondViewShape)},
                                        {bIdx * firstViewShape, sIdx * secondViewShape});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Scatter(tileTensor0, tileTensor1, tileTensor2, args->axis_, args->reduce_);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void ScatterTensorOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        SymbolicScalar self_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar self_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar self_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar idx_thirdDim = inputs[1].GetShape()[2];
        SymbolicScalar src_firstDim = inputs[2].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[2].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[2].GetShape()[2];
        auto args = static_cast<const ScatterTensorOpFuncArgs*>(opArgs);
        const int64_t firstViewShape = args->viewShape_[0];
        const int64_t secondViewShape = args->viewShape_[1];
        const int64_t thirdViewShape = args->viewShape_[2];

        const int64_t bloop = CeilDiv(idx_firstDim, firstViewShape);
        const int64_t sloop = CeilDiv(idx_secondDim, secondViewShape);
        const int64_t nloop = CeilDiv(idx_thirdDim, thirdViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                                            {std::min(self_firstDim - bIdx * firstViewShape, firstViewShape),
                                             std::min(self_secondDim - sIdx * secondViewShape, secondViewShape),
                                             std::min(self_thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    auto tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape, thirdViewShape},
                                            {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                                             std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape),
                                             std::min(idx_thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    auto tileTensor2 = View(inputs[2], {firstViewShape, secondViewShape, thirdViewShape},
                                            {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                             std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                                             std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Scatter(tileTensor0, tileTensor1, tileTensor2, args->axis_, args->reduce_);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void ScatterTensorOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        SymbolicScalar self_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar self_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar self_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar self_fourthDim = inputs[0].GetShape()[3];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar idx_thirdDim = inputs[1].GetShape()[2];
        SymbolicScalar idx_fourthDim = inputs[1].GetShape()[3];
        SymbolicScalar src_firstDim = inputs[2].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[2].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[2].GetShape()[2];
        SymbolicScalar src_fourthDim = inputs[2].GetShape()[3];
        auto args = static_cast<const ScatterTensorOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];

        const int bloop = CeilDiv(idx_firstDim, firstViewShape);
        const int sloop = CeilDiv(idx_secondDim, secondViewShape);
        const int mloop = CeilDiv(idx_thirdDim, thirdViewShape);
        const int nloop = CeilDiv(idx_fourthDim, fourthViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        auto tileTensor0 = View(inputs[0],
                                                {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                                {std::min(self_firstDim - bIdx * firstViewShape, firstViewShape),
                                                 std::min(self_secondDim - sIdx * secondViewShape, secondViewShape),
                                                 std::min(self_thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                 std::min(self_fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                                {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                 nIdx * fourthViewShape});
                        auto tileTensor1 = View(inputs[1],
                                                {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                                {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                                                 std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape),
                                                 std::min(idx_thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                 std::min(idx_fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                                {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                 nIdx * fourthViewShape});
                        auto tileTensor2 = View(inputs[2],
                                                {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                                {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                                 std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                                                 std::min(src_thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                 std::min(src_fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                                {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                 nIdx * fourthViewShape});

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Scatter(tileTensor0, tileTensor1, tileTensor2, args->axis_, args->reduce_);
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

class ScatterTensorOperationTest
    : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ScatterTensorOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestScatterTensor, ScatterTensorOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<ScatterTensorOpMetaData>(
                             {ScatterTensorOperationExeFunc2Dims, ScatterTensorOperationExeFunc3Dims,
                              ScatterTensorOperationExeFunc4Dims},
                             "ScatterTensor")));

TEST_P(ScatterTensorOperationTest, TestScatterTensor)
{
    auto test_data = GetParam().test_data_;
    auto axis = GetValueByName<int>(test_data, "axis");
    auto reduce = GetMapValByName(ScatterOperation::GetScatterModeMap(),
                                  GetValueByName<std::string>(test_data, "reduce"));
    auto args = ScatterTensorOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), axis, reduce);
    auto testCase = CreateTestCaseDesc<ScatterTensorOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace ScatterTensorOperation
