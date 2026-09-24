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
 * \file test_var_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct VarOpFuncArgs : public OpFuncArgs {
    VarOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& tileShape,
                  const std::vector<int>& dim, float correction, bool keepDim)
        : viewShape_(viewShape), tileShape_(tileShape), dim_(dim), correction_(correction), keepDim_(keepDim)
    {
        dimReduceFlag_.resize(viewShape_.size(), false);
        if (dim_.empty()) {
            std::fill(dimReduceFlag_.begin(), dimReduceFlag_.end(), true);
        }
        for (auto i : dim_) {
            int index = (i < 0) ? i + static_cast<int>(viewShape_.size()) : i;
            dimReduceFlag_[index] = true;
        }
    }

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    std::vector<int> dim_;
    float correction_;
    bool keepDim_;
    std::vector<bool> dimReduceFlag_;
};

struct VarOpMetaData {
    explicit VarOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void VarSetTileShapeBeforeAssemble(const std::vector<int64_t>& oriTileShape, const std::vector<int>& oriDim,
                                          bool keepDim, DataType dtype)
{
    if (keepDim) {
        return;
    }
    std::vector<int64_t> vecTile(oriTileShape.begin(), oriTileShape.end());
    std::vector<int> dim(oriDim.begin(), oriDim.end());
    if (dim.empty()) {
        for (size_t i = 0; i < vecTile.size(); ++i) {
            dim.push_back(static_cast<int>(i));
        }
    }
    std::sort(dim.begin(), dim.end());
    for (auto it = dim.rbegin(); it != dim.rend(); ++it) {
        vecTile.erase(vecTile.begin() + *it);
    }
    int64_t algnedSize = BLOCK_SIZE / BytesOf(dtype);
    if (vecTile.empty()) {
        vecTile.push_back(algnedSize);
    }
    int64_t lastDimSize = vecTile.back();
    if (lastDimSize % algnedSize != 0) {
        vecTile.back() = CeilDiv(lastDimSize, algnedSize) * algnedSize;
    }
    TileShape::Current().SetVecTile(vecTile);
}

static void VarOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const struct VarOpFuncArgs* args = static_cast<const VarOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const vector<bool>& dimFlag = args->dimReduceFlag_;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                auto tileTensor = View(inputs[0], {firstViewShape, secondViewShape},
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       {bIdx * firstViewShape, sIdx * secondViewShape});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Var(tileTensor, args->dim_, args->correction_, args->keepDim_);
                VarSetTileShapeBeforeAssemble(args->tileShape_, args->dim_, args->keepDim_, inputs[0].GetDataType());
                IF(args->keepDim_) { Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]); }
                ELSE IF(dimFlag[0] && dimFlag[1]) { Assemble(res, {0}, outputs[0]); }
                ELSE IF(dimFlag[0]) { Assemble(res, {sIdx * secondViewShape}, outputs[0]); }
                ELSE IF(dimFlag[1]) { Assemble(res, {bIdx * firstViewShape}, outputs[0]); }
            }
        }
    }
}

static void VarOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        const struct VarOpFuncArgs* args = static_cast<const VarOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);
        const vector<bool>& dimFlag = args->dimReduceFlag_;

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
                    auto res = Var(tileTensor, args->dim_, args->correction_, args->keepDim_);
                    VarSetTileShapeBeforeAssemble(args->tileShape_, args->dim_, args->keepDim_,
                                                  inputs[0].GetDataType());
                    IF(args->keepDim_)
                    {
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape},
                                 outputs[0]);
                    }
                    ELSE IF(dimFlag[0] && dimFlag[1] && dimFlag[2]) { Assemble(res, {0}, outputs[0]); }
                    ELSE IF(dimFlag[0] && dimFlag[1]) { Assemble(res, {nIdx * thirdViewShape}, outputs[0]); }
                    ELSE IF(dimFlag[0] && dimFlag[2]) { Assemble(res, {sIdx * secondViewShape}, outputs[0]); }
                    ELSE IF(dimFlag[1] && dimFlag[2]) { Assemble(res, {bIdx * firstViewShape}, outputs[0]); }
                    ELSE IF(dimFlag[0]) { Assemble(res, {sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]); }
                    ELSE IF(dimFlag[1]) { Assemble(res, {bIdx * firstViewShape, nIdx * thirdViewShape}, outputs[0]); }
                    ELSE { Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]); }
                }
            }
        }
    }
}

static void VarOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                     const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        auto args = static_cast<const VarOpFuncArgs*>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int mloop = CeilDiv(thirdDim, thirdViewShape);
        const int nloop = CeilDiv(fourthDim, fourthViewShape);
        const vector<bool>& dimFlag = args->dimReduceFlag_;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1))
                {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        Tensor tileTensor = View(inputs[0],
                                                 {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                                 {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                  std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                  std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                  std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                                 {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                                  nIdx * fourthViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Var(tileTensor, args->dim_, args->correction_, args->keepDim_);
                        VarSetTileShapeBeforeAssemble(args->tileShape_, args->dim_, args->keepDim_,
                                                      inputs[0].GetDataType());
                        IF(args->keepDim_)
                        {
                            Assemble(res,
                                     {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                      nIdx * fourthViewShape},
                                     outputs[0]);
                        }
                        ELSE IF(dimFlag[0] && dimFlag[1] && dimFlag[2] && dimFlag[3])
                        {
                            Assemble(res, {0}, outputs[0]);
                        }
                        ELSE IF(dimFlag[0] && dimFlag[1] && dimFlag[2])
                        {
                            Assemble(res, {nIdx * fourthViewShape}, outputs[0]);
                        }
                        ELSE IF(dimFlag[0] && dimFlag[1] && dimFlag[3])
                        {
                            Assemble(res, {mIdx * thirdViewShape}, outputs[0]);
                        }
                        ELSE IF(dimFlag[0] && dimFlag[2] && dimFlag[3])
                        {
                            Assemble(res, {sIdx * secondViewShape}, outputs[0]);
                        }
                        ELSE IF(dimFlag[0] && dimFlag[1])
                        {
                            Assemble(res, {mIdx * thirdViewShape, nIdx * fourthViewShape}, outputs[0]);
                        }
                        ELSE IF(dimFlag[0] && dimFlag[2])
                        {
                            Assemble(res, {sIdx * secondViewShape, nIdx * fourthViewShape}, outputs[0]);
                        }
                        ELSE IF(dimFlag[0] && dimFlag[3])
                        {
                            Assemble(res, {sIdx * secondViewShape, nIdx * fourthViewShape}, outputs[0]);
                        }
                        ELSE IF(dimFlag[1] && dimFlag[2])
                        {
                            Assemble(res, {bIdx * firstViewShape, nIdx * fourthViewShape}, outputs[0]);
                        }
                        ELSE IF(dimFlag[1] && dimFlag[3])
                        {
                            Assemble(res, {bIdx * firstViewShape, mIdx * thirdViewShape}, outputs[0]);
                        }
                        ELSE IF(dimFlag[2] && dimFlag[3])
                        {
                            Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                        }
                        ELSE IF(dimFlag[0])
                        {
                            Assemble(res, {sIdx * secondViewShape, mIdx * thirdViewShape, nIdx * fourthViewShape},
                                     outputs[0]);
                        }
                        ELSE IF(dimFlag[1])
                        {
                            Assemble(res, {bIdx * firstViewShape, mIdx * thirdViewShape, nIdx * fourthViewShape},
                                     outputs[0]);
                        }
                        ELSE IF(dimFlag[2])
                        {
                            Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * fourthViewShape},
                                     outputs[0]);
                        }
                        ELSE
                        {
                            Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

class VarOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<VarOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestVar, VarOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<VarOpMetaData>(
                             {VarOperationExeFunc2Dims, VarOperationExeFunc3Dims, VarOperationExeFunc4Dims}, "Var")));

TEST_P(VarOperationTest, TestVar)
{
    auto test_data = GetParam().test_data_;
    std::vector<int> dim = GetValueByName<std::vector<int>>(test_data, "dim");
    float correction = GetValueByName<float>(test_data, "correction");
    bool keepDim = GetValueByName<bool>(test_data, "keepDim");
    auto args = VarOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), dim, correction, keepDim);
    auto testCase = CreateTestCaseDesc<VarOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
