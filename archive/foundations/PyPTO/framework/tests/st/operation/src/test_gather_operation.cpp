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
 * \file test_Gather_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct GatherOpFuncArgs : public OpFuncArgs {
    GatherOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, int axis)
        : viewShape_(viewShape), tileShape_(tileShape), axis_(axis)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    int axis_;
};

struct GatherOpMetaData {
    explicit GatherOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};
constexpr int AXIS0 = 0;
constexpr int AXIS1 = 1;
constexpr int AXIS2 = 2;
constexpr int AXIS3 = 3;
static void GatherOperationExeFunc1_1Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        auto args = static_cast<const GatherOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int bloop = CeilDiv(idx_firstDim, firstViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            auto tileTensor0 = View(inputs[0], {src_firstDim}, {src_firstDim}, {0});
            auto tileTensor1 = View(inputs[1], {firstViewShape},
                                    {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape)},
                                    {bIdx * firstViewShape});
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Gather(tileTensor0, tileTensor1, args->axis_);
            Assemble(res, {bIdx * firstViewShape}, outputs[0]);
        }
    }
}

static void GatherOperationExeFunc1_2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        auto args = static_cast<const GatherOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int bloop = CeilDiv(idx_firstDim, firstViewShape);
        const int sloop = CeilDiv(idx_secondDim, secondViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                auto tileTensor0 = View(inputs[0], {src_firstDim}, {src_firstDim}, {0});
                auto tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                                        {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                                         std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape)},
                                        {bIdx * firstViewShape, sIdx * secondViewShape});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void GatherOperationExeFunc2_1Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        auto args = static_cast<const GatherOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis < 0 ? axis + inputs[0].GetShape().size() : axis;
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        /* gather操作src axis轴不能切分 ，其他轴可正常切分，index和最终输出都可正常切分。切分以index为准
         * src axis不能切分，需要保证axis轴的viewshape=src的shape */
        if (axis == AXIS0) {
            const int bloop = CeilDiv(idx_firstDim, firstViewShape);
            const int sloop = CeilDiv(src_secondDim, secondViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    auto tileTensor0 = View(
                        inputs[0], {src_firstDim, secondViewShape},
                        {src_firstDim, std::min(src_secondDim - sIdx * secondViewShape, secondViewShape)},
                        {0, sIdx * secondViewShape});
                    auto tileTensor1 = View(inputs[1], {firstViewShape},
                                            {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape)},
                                            {bIdx * firstViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                }
            }
        } else {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(idx_firstDim, secondViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    auto tileTensor0 = View(
                        inputs[0], {firstViewShape, src_secondDim},
                        {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape), src_secondDim},
                        {bIdx * firstViewShape, 0});
                    auto tileTensor1 = View(inputs[1], {secondViewShape},
                                            {std::min(idx_firstDim - sIdx * secondViewShape, secondViewShape)},
                                            {sIdx * secondViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void GatherOperationExeFunc2_2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        auto args = static_cast<const GatherOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis < 0 ? axis + inputs[0].GetShape().size() : axis;
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        /* gather操作src axis轴不能切分 ，其他轴可正常切分，index和最终输出都可正常切分。切分以index为准
         * src axis不能切分，需要保证axis轴的viewshape=src的shape */
        if (axis == AXIS0) {
            const int bloop = CeilDiv(idx_firstDim, firstViewShape);
            const int sloop = CeilDiv(idx_secondDim, secondViewShape);
            const int nloop = CeilDiv(src_secondDim, thirdViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        auto tileTensor0 = View(
                            inputs[0], {src_firstDim, thirdViewShape},
                            {src_firstDim, std::min(src_secondDim - nIdx * thirdViewShape, thirdViewShape)},
                            {0, nIdx * thirdViewShape});
                        auto tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                                                {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                                                 std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape)},
                                                {bIdx * firstViewShape, sIdx * secondViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape},
                                 outputs[0]);
                    }
                }
            }
        } else {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(idx_firstDim, secondViewShape);
            const int nloop = CeilDiv(idx_secondDim, thirdViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        auto tileTensor0 = View(
                            inputs[0], {firstViewShape, src_secondDim},
                            {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape), src_secondDim},
                            {bIdx * firstViewShape, 0});
                        auto tileTensor1 = View(inputs[1], {secondViewShape, thirdViewShape},
                                                {std::min(idx_firstDim - sIdx * secondViewShape, secondViewShape),
                                                 std::min(idx_secondDim - nIdx * thirdViewShape, thirdViewShape)},
                                                {sIdx * secondViewShape, nIdx * thirdViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

static void GatherOperationExeFunc3_1Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        auto args = static_cast<const GatherOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis < 0 ? axis + inputs[0].GetShape().size() : axis;
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        /* gather操作src axis轴不能切分 ，其他轴可正常切分，index和最终输出都可正常切分。切分以index为准
         * src axis不能切分，需要保证axis轴的viewshape=src的shape */
        if (axis == AXIS0) {
            const int bloop = CeilDiv(idx_firstDim, firstViewShape);
            const int sloop = CeilDiv(src_secondDim, secondViewShape);
            const int nloop = CeilDiv(src_thirdDim, thirdViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        auto tileTensor0 = View(
                            inputs[0], {src_firstDim, secondViewShape, thirdViewShape},
                            {src_firstDim, std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                             std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                            {0, sIdx * secondViewShape, nIdx * thirdViewShape});
                        auto tileTensor1 = View(inputs[1], {firstViewShape},
                                                {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape)},
                                                {bIdx * firstViewShape});

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape},
                                 outputs[0]);
                    }
                }
            }
        } else if (axis == AXIS1) {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(idx_firstDim, secondViewShape);
            const int nloop = CeilDiv(src_thirdDim, thirdViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        auto tileTensor0 = View(
                            inputs[0], {firstViewShape, src_secondDim, thirdViewShape},
                            {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape), src_secondDim,
                             std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                            {bIdx * firstViewShape, 0, nIdx * thirdViewShape});
                        auto tileTensor1 = View(inputs[1], {secondViewShape},
                                                {std::min(idx_firstDim - sIdx * secondViewShape, secondViewShape)},
                                                {sIdx * secondViewShape});

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape},
                                 outputs[0]);
                    }
                }
            }
        } else {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(src_secondDim, secondViewShape);
            const int nloop = CeilDiv(idx_firstDim, thirdViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        auto tileTensor0 = View(
                            inputs[0], {firstViewShape, secondViewShape, src_thirdDim},
                            {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                             std::min(src_secondDim - sIdx * secondViewShape, secondViewShape), src_thirdDim},
                            {bIdx * firstViewShape, sIdx * secondViewShape, 0});
                        auto tileTensor1 = View(inputs[1], {thirdViewShape},
                                                {std::min(idx_firstDim - nIdx * thirdViewShape, thirdViewShape)},
                                                {nIdx * thirdViewShape});

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

static void GatherOperationExeFunc3_2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        auto args = static_cast<const GatherOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis < 0 ? axis + inputs[0].GetShape().size() : axis;
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        /* gather操作src axis轴不能切分 ，其他轴可正常切分，index和最终输出都可正常切分。切分以index为准
         * src axis不能切分，需要保证axis轴的viewshape=src的shape */
        if (axis == AXIS0) {
            const int bloop = CeilDiv(idx_firstDim, firstViewShape);
            const int sloop = CeilDiv(idx_secondDim, secondViewShape);
            const int nloop = CeilDiv(src_secondDim, thirdViewShape);
            const int mloop = CeilDiv(src_thirdDim, fourthViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                        {
                            auto tileTensor0 = View(
                                inputs[0], {src_firstDim, thirdViewShape, fourthViewShape},
                                {src_firstDim, std::min(src_secondDim - nIdx * thirdViewShape, thirdViewShape),
                                 std::min(src_thirdDim - mIdx * fourthViewShape, fourthViewShape)},
                                {0, nIdx * thirdViewShape, mIdx * fourthViewShape});

                            auto tileTensor1 = View(inputs[1], {firstViewShape, secondViewShape},
                                                    {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                                                     std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape)},
                                                    {bIdx * firstViewShape, sIdx * secondViewShape});

                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                            Assemble(res,
                                     {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                      mIdx * fourthViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        } else if (axis == AXIS1) {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(idx_firstDim, secondViewShape);
            const int nloop = CeilDiv(idx_secondDim, thirdViewShape);
            const int mloop = CeilDiv(src_thirdDim, fourthViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                        {
                            auto tileTensor0 = View(
                                inputs[0], {firstViewShape, src_secondDim, fourthViewShape},
                                {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape), src_secondDim,
                                 std::min(src_thirdDim - mIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, 0, mIdx * fourthViewShape});
                            auto tileTensor1 = View(inputs[1], {secondViewShape, thirdViewShape},
                                                    {std::min(idx_firstDim - sIdx * secondViewShape, secondViewShape),
                                                     std::min(idx_secondDim - nIdx * thirdViewShape, thirdViewShape)},
                                                    {sIdx * secondViewShape, nIdx * thirdViewShape});

                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                            Assemble(res,
                                     {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                      mIdx * fourthViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        } else {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(src_secondDim, secondViewShape);
            const int nloop = CeilDiv(idx_firstDim, thirdViewShape);
            const int mloop = CeilDiv(idx_secondDim, fourthViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                        {
                            auto tileTensor0 = View(
                                inputs[0], {firstViewShape, secondViewShape, src_thirdDim},
                                {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(src_secondDim - sIdx * secondViewShape, secondViewShape), src_thirdDim},
                                {bIdx * firstViewShape, sIdx * secondViewShape, 0});
                            auto tileTensor1 = View(inputs[1], {thirdViewShape, fourthViewShape},
                                                    {std::min(idx_firstDim - nIdx * thirdViewShape, thirdViewShape),
                                                     std::min(idx_secondDim - mIdx * fourthViewShape, fourthViewShape)},
                                                    {nIdx * thirdViewShape, mIdx * fourthViewShape});

                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                            Assemble(res,
                                     {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                      mIdx * fourthViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

static void GatherOperationExeFunc4_1Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar src_fourthDim = inputs[0].GetShape()[3];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        auto args = static_cast<const GatherOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis < 0 ? axis + inputs[0].GetShape().size() : axis;
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        /* gather操作src axis轴不能切分 ，其他轴可正常切分，index和最终输出都可正常切分。切分以index为准
         * src axis不能切分，需要保证axis轴的viewshape=src的shape */
        if (axis == AXIS0) {
            const int bloop = CeilDiv(idx_firstDim, firstViewShape);
            const int sloop = CeilDiv(src_secondDim, secondViewShape);
            const int nloop = CeilDiv(src_thirdDim, thirdViewShape);
            const int mloop = CeilDiv(src_fourthDim, fourthViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                        {
                            auto tileTensor0 = View(
                                inputs[0], {src_firstDim, secondViewShape, thirdViewShape, fourthViewShape},
                                {src_firstDim, std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                                 std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                 std::min(src_fourthDim - mIdx * fourthViewShape, fourthViewShape)},
                                {0, sIdx * secondViewShape, nIdx * thirdViewShape, mIdx * fourthViewShape});
                            auto tileTensor1 = View(inputs[1], {firstViewShape},
                                                    {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape)},
                                                    {bIdx * firstViewShape});

                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                            Assemble(res,
                                     {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                      mIdx * fourthViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        } else if (axis == AXIS1) {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(idx_firstDim, secondViewShape);
            const int nloop = CeilDiv(src_thirdDim, thirdViewShape);
            const int mloop = CeilDiv(src_fourthDim, fourthViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                        {
                            auto tileTensor0 = View(
                                inputs[0], {firstViewShape, src_secondDim, thirdViewShape, fourthViewShape},
                                {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape), src_secondDim,
                                 std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                 std::min(src_fourthDim - mIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, 0, nIdx * thirdViewShape, mIdx * fourthViewShape});
                            auto tileTensor1 = View(inputs[1], {secondViewShape},
                                                    {std::min(idx_firstDim - sIdx * secondViewShape, secondViewShape)},
                                                    {sIdx * secondViewShape});

                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                            Assemble(res,
                                     {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                      mIdx * fourthViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        } else if (axis == AXIS2) {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(src_secondDim, secondViewShape);
            const int nloop = CeilDiv(idx_firstDim, thirdViewShape);
            const int mloop = CeilDiv(src_fourthDim, fourthViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                        {
                            auto tileTensor0 = View(
                                inputs[0], {firstViewShape, secondViewShape, src_thirdDim, fourthViewShape},
                                {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(src_secondDim - sIdx * secondViewShape, secondViewShape), src_thirdDim,
                                 std::min(src_fourthDim - mIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, 0, mIdx * fourthViewShape});
                            auto tileTensor1 = View(inputs[1], {thirdViewShape},
                                                    {std::min(idx_firstDim - nIdx * thirdViewShape, thirdViewShape)},
                                                    {nIdx * thirdViewShape});

                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                            Assemble(res,
                                     {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                      mIdx * fourthViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        } else if (axis == AXIS3) {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(src_secondDim, secondViewShape);
            const int nloop = CeilDiv(src_thirdDim, thirdViewShape);
            const int mloop = CeilDiv(idx_firstDim, fourthViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                        {
                            auto tileTensor0 = View(
                                inputs[0], {firstViewShape, secondViewShape, thirdViewShape, src_fourthDim},
                                {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                                 std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape), src_fourthDim},
                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, 0});
                            auto tileTensor1 = View(inputs[1], {fourthViewShape},
                                                    {std::min(idx_firstDim - mIdx * fourthViewShape, fourthViewShape)},
                                                    {mIdx * fourthViewShape});

                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                            Assemble(res,
                                     {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                      mIdx * fourthViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

static void GatherOperationExeFunc4_2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar src_fourthDim = inputs[0].GetShape()[3];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        auto args = static_cast<const GatherOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis < 0 ? axis + inputs[0].GetShape().size() : axis;
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        const int fifthViewShape = args->viewShape_[4];
        /* gather操作src axis轴不能切分 ，其他轴可正常切分，index和最终输出都可正常切分。切分以index为准
         * src axis不能切分，需要保证axis轴的viewshape=src的shape */
        if (axis == AXIS0) {
            const int bloop = CeilDiv(idx_firstDim, firstViewShape);
            const int sloop = CeilDiv(idx_secondDim, secondViewShape);
            const int nloop = CeilDiv(src_secondDim, thirdViewShape);
            const int mloop = CeilDiv(src_thirdDim, fourthViewShape);
            const int kloop = CeilDiv(src_fourthDim, fifthViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                        {
                            LOOP("LOOP_L4_kIdx", FunctionType::DYNAMIC_LOOP, kIdx, LoopRange(0, kloop, 1))
                            {
                                auto tileTensor0 = View(
                                    inputs[0], {src_firstDim, thirdViewShape, fourthViewShape, fifthViewShape},
                                    {src_firstDim, std::min(src_secondDim - nIdx * thirdViewShape, thirdViewShape),
                                     std::min(src_thirdDim - mIdx * fourthViewShape, fourthViewShape),
                                     std::min(src_fourthDim - kIdx * fifthViewShape, fifthViewShape)},
                                    {0, nIdx * thirdViewShape, mIdx * fourthViewShape, kIdx * fifthViewShape});
                                auto tileTensor1 = View(
                                    inputs[1], {firstViewShape, secondViewShape},
                                    {std::min(idx_firstDim - bIdx * firstViewShape, firstViewShape),
                                     std::min(idx_secondDim - sIdx * secondViewShape, secondViewShape)},
                                    {bIdx * firstViewShape, sIdx * secondViewShape});

                                TileShape::Current().SetVecTile(args->tileShape_);
                                auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                                Assemble(res,
                                         {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                          mIdx * fourthViewShape, kIdx * fifthViewShape},
                                         outputs[0]);
                            }
                        }
                    }
                }
            }
        } else if (axis == AXIS1) {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(idx_firstDim, secondViewShape);
            const int nloop = CeilDiv(idx_secondDim, thirdViewShape);
            const int mloop = CeilDiv(src_thirdDim, fourthViewShape);
            const int kloop = CeilDiv(src_fourthDim, fifthViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                        {
                            LOOP("LOOP_L4_kIdx", FunctionType::DYNAMIC_LOOP, kIdx, LoopRange(0, kloop, 1))
                            {
                                auto tileTensor0 = View(
                                    inputs[0], {firstViewShape, src_secondDim, fourthViewShape, fifthViewShape},
                                    {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape), src_secondDim,
                                     std::min(src_thirdDim - mIdx * fourthViewShape, fourthViewShape),
                                     std::min(src_fourthDim - kIdx * fifthViewShape, fifthViewShape)},
                                    {bIdx * firstViewShape, 0, mIdx * fourthViewShape, kIdx * fifthViewShape});
                                auto tileTensor1 = View(
                                    inputs[1], {secondViewShape, thirdViewShape},
                                    {std::min(idx_firstDim - sIdx * secondViewShape, secondViewShape),
                                     std::min(idx_secondDim - nIdx * thirdViewShape, thirdViewShape)},
                                    {sIdx * secondViewShape, nIdx * thirdViewShape});

                                TileShape::Current().SetVecTile(args->tileShape_);
                                auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                                Assemble(res,
                                         {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                          mIdx * fourthViewShape, kIdx * fifthViewShape},
                                         outputs[0]);
                            }
                        }
                    }
                }
            }
        } else if (axis == AXIS2) {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(src_secondDim, secondViewShape);
            const int nloop = CeilDiv(idx_firstDim, thirdViewShape);
            const int mloop = CeilDiv(idx_secondDim, fourthViewShape);
            const int kloop = CeilDiv(src_fourthDim, fifthViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                        {
                            LOOP("LOOP_L4_kIdx", FunctionType::DYNAMIC_LOOP, kIdx, LoopRange(0, kloop, 1))
                            {
                                auto tileTensor0 = View(
                                    inputs[0], {firstViewShape, secondViewShape, src_thirdDim, fifthViewShape},
                                    {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                     std::min(src_secondDim - sIdx * secondViewShape, secondViewShape), src_thirdDim,
                                     std::min(src_fourthDim - kIdx * fifthViewShape, fifthViewShape)},
                                    {bIdx * firstViewShape, sIdx * secondViewShape, 0, kIdx * fifthViewShape});
                                auto tileTensor1 = View(
                                    inputs[1], {thirdViewShape, fourthViewShape},
                                    {std::min(idx_firstDim - nIdx * thirdViewShape, thirdViewShape),
                                     std::min(idx_secondDim - mIdx * fourthViewShape, fourthViewShape)},
                                    {nIdx * thirdViewShape, mIdx * fourthViewShape});

                                TileShape::Current().SetVecTile(args->tileShape_);
                                auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                                Assemble(res,
                                         {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                          mIdx * fourthViewShape, kIdx * fifthViewShape},
                                         outputs[0]);
                            }
                        }
                    }
                }
            }
        } else if (axis == AXIS3) {
            const int bloop = CeilDiv(src_firstDim, firstViewShape);
            const int sloop = CeilDiv(src_secondDim, secondViewShape);
            const int nloop = CeilDiv(src_thirdDim, thirdViewShape);
            const int mloop = CeilDiv(idx_firstDim, fourthViewShape);
            const int kloop = CeilDiv(idx_secondDim, fifthViewShape);
            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        LOOP("LOOP_L3_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                        {
                            LOOP("LOOP_L4_kIdx", FunctionType::DYNAMIC_LOOP, kIdx, LoopRange(0, kloop, 1))
                            {
                                auto tileTensor0 = View(
                                    inputs[0], {firstViewShape, secondViewShape, thirdViewShape, src_fourthDim},
                                    {std::min(src_firstDim - bIdx * firstViewShape, firstViewShape),
                                     std::min(src_secondDim - sIdx * secondViewShape, secondViewShape),
                                     std::min(src_thirdDim - nIdx * thirdViewShape, thirdViewShape), src_fourthDim},
                                    {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, 0});
                                auto tileTensor1 = View(
                                    inputs[1], {fourthViewShape, fifthViewShape},
                                    {std::min(idx_firstDim - mIdx * fourthViewShape, fourthViewShape),
                                     std::min(idx_secondDim - kIdx * fifthViewShape, fifthViewShape)},
                                    {mIdx * fourthViewShape, kIdx * fifthViewShape});

                                TileShape::Current().SetVecTile(args->tileShape_);
                                auto res = Gather(tileTensor0, tileTensor1, args->axis_);
                                Assemble(res,
                                         {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                          mIdx * fourthViewShape, kIdx * fifthViewShape},
                                         outputs[0]);
                            }
                        }
                    }
                }
            }
        }
    }
}
class GatherOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<GatherOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestGather, GatherOperationTest,
    ::testing::ValuesIn(GetOpMetaData<GatherOpMetaData>({GatherOperationExeFunc1_1Dims, GatherOperationExeFunc1_2Dims,
                                                         GatherOperationExeFunc2_1Dims, GatherOperationExeFunc2_2Dims,
                                                         GatherOperationExeFunc3_1Dims, GatherOperationExeFunc3_2Dims,
                                                         GatherOperationExeFunc4_1Dims, GatherOperationExeFunc4_2Dims},
                                                        "Gather")));

TEST_P(GatherOperationTest, TestGather)
{
    std::unordered_map<int, std::unordered_map<int, OpFunc>> func{
        {1, {{1, GatherOperationExeFunc1_1Dims}, {2, GatherOperationExeFunc1_2Dims}}},
        {2, {{1, GatherOperationExeFunc2_1Dims}, {2, GatherOperationExeFunc2_2Dims}}},
        {3, {{1, GatherOperationExeFunc3_1Dims}, {2, GatherOperationExeFunc3_2Dims}}},
        {4, {{1, GatherOperationExeFunc4_1Dims}, {2, GatherOperationExeFunc4_2Dims}}},
    };
    auto test_data = GetParam().test_data_;
    ConfigManagerNg::JitScopeGuard configGuard(
        "matmul_test", std::map<std::string, std::any>{{"pass." + std::string(ENABLE_SLICE), false}});
    auto axis = static_cast<CastMode>(GetValueByName<int>(test_data, "axis"));
    auto args = GatherOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), axis);
    auto testCase = CreateTestCaseDesc<GatherOpMetaData>(GetParam(), &args);
    int params_rank = testCase.inputTensors[0].GetShape().size();
    int indices_rank = testCase.inputTensors[1].GetShape().size();

    if ((func.find(params_rank) == func.end()) || (func[params_rank].find(indices_rank) == func[params_rank].end())) {
        std::cerr << "测试不支持这个形状" << std::endl;
        ASSERT(false);
    }
    testCase.opFunc = func[params_rank][indices_rank];
    TestExecutor::runTest(testCase);
}
} // namespace
