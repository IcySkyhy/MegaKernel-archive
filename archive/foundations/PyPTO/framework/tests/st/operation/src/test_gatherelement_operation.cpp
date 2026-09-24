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
 * \file test_GatherElement_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
const unsigned IDX_DIM0 = 0;
const unsigned IDX_DIM1 = 1;
const unsigned IDX_DIM2 = 2;
const unsigned IDX_DIM3 = 3;
const unsigned IDX_DIM4 = 4;
struct GatherElementOpFuncArgs : public OpFuncArgs {
    GatherElementOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, int axis)
        : viewShape_(viewShape), tileShape_(tileShape), axis_(axis)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    int axis_;
};

struct GatherElementOpMetaData {
    explicit GatherElementOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void GatherElementOperationExeFunc1Dim(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                              const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        auto args = static_cast<const GatherElementOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + inputs[0].GetShape().size();
        std::vector<int64_t> viewShape = args->viewShape_;
        ASSERT(viewShape[axis] >= std::max(inputs[0].GetShape()[axis], inputs[1].GetShape()[axis]));
        const int firstViewShape = viewShape[0];

        // gather操作src axis轴不能切分 ，其他轴可正常切分，index和最终输出都可正常切分, 切分以index为准
        const int loop = CeilDiv(idx_firstDim, firstViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop))
        {
            auto tileTensor0 = View(inputs[0], {firstViewShape},
                                    {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape)},
                                    {bIdx * firstViewShape});
            auto tileTensor1 = View(inputs[1], {firstViewShape},
                                    {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape)},
                                    {bIdx * firstViewShape});
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = GatherElements(tileTensor0, tileTensor1, args->axis_);
            Assemble(res, {bIdx * firstViewShape}, outputs[0]);
        }
    }
}

static void GatherElementOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        auto args = static_cast<const GatherElementOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + inputs[0].GetShape().size();
        std::vector<int64_t> viewShape = args->viewShape_;
        ASSERT(viewShape[axis] >= std::max(inputs[0].GetShape()[axis], inputs[1].GetShape()[axis]));
        const int firstViewShape = viewShape[0];
        const int secondViewShape = viewShape[1];

        // gather操作src axis轴不能切分 ，其他轴可正常切分，index和最终输出都可正常切分, 切分以index为准
        const int loop[] = {CeilDiv(idx_firstDim, firstViewShape), CeilDiv(idx_secondDim, secondViewShape)};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                                        {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                         std::min(src_secondDim - sIdx * secondViewShape, secondViewShape)},
                                        {bIdx * firstViewShape, sIdx * secondViewShape});
                auto tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                                        {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                                         std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape)},
                                        {bIdx * firstViewShape, sIdx * secondViewShape});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = GatherElements(tileTensor0, tileTensor1, args->axis_);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void GatherElementOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar idx_thirdDim = inputs[1].GetShape()[2];

        auto args = static_cast<const GatherElementOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + inputs[0].GetShape().size();
        std::vector<int64_t> viewShape = args->viewShape_;
        ASSERT(viewShape[axis] >= std::max(inputs[0].GetShape()[axis], inputs[1].GetShape()[axis]));
        const int firstViewShape = viewShape[0];
        const int secondViewShape = viewShape[1];
        const int thirdViewShape = viewShape[2];

        // gather操作src axis轴不能切分 ，其他轴可正常切分，index和最终输出都可正常切分, 切分以index为准
        const int loop[] = {CeilDiv(idx_firstDim, firstViewShape), CeilDiv(idx_secondDim, secondViewShape),
                            CeilDiv(idx_thirdDim, thirdViewShape)};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2]))
                {
                    auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                                            {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                             std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                                             std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    auto tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape, thirdViewShape},
                                            {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                                             std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape),
                                             std::min(idx_thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = GatherElements(tileTensor0, tileTensor1, args->axis_);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void GatherElementOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar src_forthDim = inputs[0].GetShape()[3];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar idx_thirdDim = inputs[1].GetShape()[2];
        SymbolicScalar idx_forthDim = inputs[1].GetShape()[3];
        auto args = static_cast<const GatherElementOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + inputs[0].GetShape().size();
        std::vector<int64_t> viewShape = args->viewShape_;
        ASSERT(viewShape[axis] >= std::max(inputs[0].GetShape()[axis], inputs[1].GetShape()[axis]));
        const int firstViewShape = viewShape[0];
        const int secondViewShape = viewShape[1];
        const int thirdViewShape = viewShape[2];
        const int forthViewShape = viewShape[3];
        const int loop[] = {CeilDiv(idx_firstDim, firstViewShape), CeilDiv(idx_secondDim, secondViewShape),
                            CeilDiv(idx_thirdDim, thirdViewShape), CeilDiv(idx_forthDim, forthViewShape)};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2]))
                {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(loop[IDX_DIM3]))
                    {
                        auto tileTensor0 = View(inputs[0],
                                                {firstViewShape, secondViewShape, thirdViewShape, forthViewShape},
                                                {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                                 std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                                                 std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                                 std::min(src_forthDim - qIdx * forthViewShape, forthViewShape)},
                                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                                 qIdx * forthViewShape});
                        auto tileTensor1 = View(inputs[1],
                                                {firstViewShape, secondViewShape, thirdViewShape, forthViewShape},
                                                {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                                                 std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape),
                                                 std::min(idx_thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                                 std::min(idx_forthDim - qIdx * forthViewShape, forthViewShape)},
                                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                                 qIdx * forthViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = GatherElements(tileTensor0, tileTensor1, args->axis_);
                        Assemble(res,
                                 {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                  qIdx * forthViewShape},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

static void GatherElementOperationExeFunc5Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar src_forthDim = inputs[0].GetShape()[3];
        SymbolicScalar src_fifthDim = inputs[0].GetShape()[4];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar idx_thirdDim = inputs[1].GetShape()[2];
        SymbolicScalar idx_forthDim = inputs[1].GetShape()[3];
        SymbolicScalar idx_fifthDim = inputs[1].GetShape()[4];

        auto args = static_cast<const GatherElementOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + inputs[0].GetShape().size();
        std::vector<int64_t> viewShape = args->viewShape_;
        ASSERT(viewShape[axis] >= std::max(inputs[0].GetShape()[axis], inputs[1].GetShape()[axis]));
        const int firstViewShape = viewShape[0];
        const int secondViewShape = viewShape[1];
        const int thirdViewShape = viewShape[2];
        const int forthViewShape = viewShape[3];
        const int fifthViewShape = viewShape[4];

        // gather操作src axis轴不能切分 ，其他轴可正常切分，index和最终输出都可正常切分, 切分以index为准
        const int loop[] = {CeilDiv(idx_firstDim, firstViewShape), CeilDiv(idx_secondDim, secondViewShape),
                            CeilDiv(idx_thirdDim, thirdViewShape), CeilDiv(idx_forthDim, forthViewShape),
                            CeilDiv(idx_fifthDim, fifthViewShape)};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0]))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1]))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2]))
                {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(loop[IDX_DIM3]))
                    {
                        LOOP("LOOP_L4_rIdx", FunctionType::DYNAMIC_LOOP, rIdx, LoopRange(loop[IDX_DIM4]))
                        {
                            auto tileTensor0 = View(
                                inputs[0],
                                {firstViewShape, secondViewShape, thirdViewShape, forthViewShape, fifthViewShape},
                                {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                                 std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                 std::min(src_forthDim - qIdx * forthViewShape, forthViewShape),
                                 std::min(src_fifthDim - rIdx * fifthViewShape, fifthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                 qIdx * forthViewShape, rIdx * fifthViewShape});
                            auto tileTensor1 = View(
                                inputs[1],
                                {firstViewShape, secondViewShape, thirdViewShape, forthViewShape, fifthViewShape},
                                {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape),
                                 std::min(idx_thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                 std::min(idx_forthDim - qIdx * forthViewShape, forthViewShape),
                                 std::min(idx_fifthDim - rIdx * fifthViewShape, fifthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                 qIdx * forthViewShape, rIdx * fifthViewShape});

                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = GatherElements(tileTensor0, tileTensor1, args->axis_);
                            Assemble(res,
                                     {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                      qIdx * forthViewShape, rIdx * fifthViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        }
    }
}
class GatherElementOperationTest
    : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<GatherElementOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestGatherElement, GatherElementOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<GatherElementOpMetaData>(
                             {GatherElementOperationExeFunc1Dim, GatherElementOperationExeFunc2Dims,
                              GatherElementOperationExeFunc3Dims, GatherElementOperationExeFunc4Dims,
                              GatherElementOperationExeFunc5Dims},
                             "GatherElement")));

TEST_P(GatherElementOperationTest, TestGatherElement)
{
    auto test_data = GetParam().test_data_;
    auto axis = static_cast<CastMode>(GetValueByName<int>(test_data, "axis"));
    auto args = GatherElementOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), axis);
    auto testCase = CreateTestCaseDesc<GatherElementOpMetaData>(GetParam(), &args);
    std::vector<OpFunc> opFuncs = {GatherElementOperationExeFunc1Dim, GatherElementOperationExeFunc2Dims,
                                   GatherElementOperationExeFunc3Dims, GatherElementOperationExeFunc4Dims,
                                   GatherElementOperationExeFunc5Dims};
    testCase.opFunc = opFuncs[GetViewShape(test_data).size() - 1];
    TestExecutor::runTest(testCase);
}
} // namespace
