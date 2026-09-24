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
 * \file test_reduce_prod_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
const unsigned IDX_DIM0 = 0;
const unsigned IDX_DIM1 = 1;
const unsigned IDX_DIM2 = 2;
const unsigned IDX_DIM3 = 3;

struct ProdOpFuncArgs : public OpFuncArgs {
    ProdOpFuncArgs(std::vector<int64_t> viewShape, const std::vector<int64_t> tileShape, std::vector<int64_t> dims,
                   const bool keepDim)
        : viewShape_(viewShape), tileShape_(tileShape), dims_(dims), keepDim_(keepDim)
    {}
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    std::vector<int64_t> dims_;
    bool keepDim_;
};

struct ProdOpMetadata {
    explicit ProdOpMetadata(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

void AdjustTileShapeForReduce(const int dim, const Tensor& result, std::vector<int64_t> tileshape)
{
    tileshape.erase(tileshape.begin() + dim);
    const int alignNum = BLOCK_SIZE / BytesOf(result.GetStorage()->tensor->datatype);
    tileshape[tileshape.size() - 1] = (tileshape[tileshape.size() - 1] + alignNum - 1) / alignNum * alignNum;
    TileShape::Current().SetVecTile(tileshape);
}

void ProdOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    auto args = static_cast<const ProdOpFuncArgs*>(opArgs);
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        int dim = args->dims_[0];
        bool keepDim = args->keepDim_;
        if (dim < 0) {
            dim = static_cast<int>(inputs[0].GetShape().size()) + dim;
        }
        SymbolicScalar viewShape[] = {args->viewShape_[0], args->viewShape_[1]};
        viewShape[dim] = 0;
        const int batch = CeilDiv(inputs[0].GetShape()[1 - dim], viewShape[1 - dim]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(batch))
        {
            auto viewTensor = View(
                inputs[0], {viewShape[0] == 0 ? firstDim : viewShape[0], viewShape[1] == 0 ? secondDim : viewShape[1]},
                {viewShape[0] == 0 ? firstDim : std::min(firstDim - bIdx * viewShape[0], viewShape[0]),
                 viewShape[1] == 0 ? secondDim : std::min(secondDim - bIdx * viewShape[1], viewShape[1])},
                {bIdx * viewShape[0], bIdx * viewShape[1]});
            TileShape::Current().SetVecTile(args->tileShape_);
            std::vector<SymbolicScalar> offset = {bIdx * viewShape[0], bIdx * viewShape[1]};
            auto res = Prod(viewTensor, args->dims_[0], keepDim);
            if (!keepDim) {
                offset.erase(offset.begin() + dim);
                AdjustTileShapeForReduce(dim, res, args->tileShape_);
            }
            Assemble(res, offset, outputs[0]);
        }
    }
}

void Prod3DOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    auto args = static_cast<const ProdOpFuncArgs*>(opArgs);
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar lastDim = inputs[0].GetShape()[2];
        int dim = args->dims_[0];
        bool keepDim = args->keepDim_;
        if (dim < 0) {
            dim = static_cast<int>(inputs[0].GetShape().size()) + dim;
        }
        SymbolicScalar viewShape[] = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2]};
        int loops[] = {CeilDiv(inputs[0].GetShape()[0], viewShape[0]), CeilDiv(inputs[0].GetShape()[1], viewShape[1]),
                       CeilDiv(inputs[0].GetShape()[2], viewShape[2])};
        viewShape[dim] = 0;
        loops[dim] = 1;
        if (dim == IDX_DIM0) {
            viewShape[IDX_DIM1] = 0;
            loops[IDX_DIM1] = 1;
        } else {
            viewShape[IDX_DIM0] = 0;
            loops[IDX_DIM0] = 1;
        }
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loops[IDX_DIM0]))
        {
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loops[IDX_DIM1]))
            {
                LOOP("LOOP_L2_bIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loops[IDX_DIM2]))
                {
                    auto viewTensor = View(
                        inputs[0],
                        {viewShape[0] == 0 ? firstDim : viewShape[0], viewShape[1] == 0 ? secondDim : viewShape[1],
                         viewShape[2] == 0 ? lastDim : viewShape[2]},
                        {viewShape[0] == 0 ? firstDim : std::min(firstDim - bIdx * viewShape[0], viewShape[0]),
                         viewShape[1] == 0 ? secondDim : std::min(secondDim - sIdx * viewShape[1], viewShape[1]),
                         viewShape[2] == 0 ? lastDim : std::min(lastDim - nIdx * viewShape[2], viewShape[2])},
                        {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    std::vector<SymbolicScalar> offset = {bIdx * viewShape[0], sIdx * viewShape[1],
                                                          nIdx * viewShape[2]};
                    auto res = Prod(viewTensor, args->dims_[0], keepDim);
                    if (!keepDim) {
                        offset.erase(offset.begin() + dim);
                        AdjustTileShapeForReduce(dim, res, args->tileShape_);
                    }
                    Assemble(res, offset, outputs[0]);
                }
            }
        }
    }
}

void Prod4DOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    auto args = static_cast<const ProdOpFuncArgs*>(opArgs);
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar lastDim = inputs[0].GetShape()[3];
        int dim = args->dims_[0];
        bool keepDim = args->keepDim_;
        if (dim < 0) {
            dim = static_cast<int>(inputs[0].GetShape().size()) + dim;
        }
        SymbolicScalar viewShape[] = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2],
                                      args->viewShape_[3]};
        int loops[] = {CeilDiv(inputs[0].GetShape()[0], viewShape[0]), CeilDiv(inputs[0].GetShape()[1], viewShape[1]),
                       CeilDiv(inputs[0].GetShape()[2], viewShape[2]), CeilDiv(inputs[0].GetShape()[3], viewShape[3])};
        viewShape[dim] = 0;
        loops[dim] = 1;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loops[IDX_DIM0]))
        {
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loops[IDX_DIM1]))
            {
                LOOP("LOOP_L2_bIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loops[IDX_DIM2]))
                {
                    LOOP("LOOP_L3_bIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(loops[IDX_DIM3]))
                    {
                        std::vector<SymbolicScalar> offset = {bIdx * viewShape[0], sIdx * viewShape[1],
                                                              nIdx * viewShape[2], qIdx * viewShape[3]};
                        auto viewTensor = View(
                            inputs[0],
                            {viewShape[0] == 0 ? firstDim : viewShape[0], viewShape[1] == 0 ? secondDim : viewShape[1],
                             viewShape[2] == 0 ? thirdDim : viewShape[2], viewShape[3] == 0 ? lastDim : viewShape[3]},
                            {viewShape[0] == 0 ? firstDim : std::min(firstDim - bIdx * viewShape[0], viewShape[0]),
                             viewShape[1] == 0 ? secondDim : std::min(secondDim - sIdx * viewShape[1], viewShape[1]),
                             viewShape[2] == 0 ? thirdDim : std::min(thirdDim - nIdx * viewShape[2], viewShape[2]),
                             viewShape[3] == 0 ? lastDim : std::min(lastDim - qIdx * viewShape[3], viewShape[3])},
                            offset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Prod(viewTensor, args->dims_[0], keepDim);
                        if (!keepDim) {
                            offset.erase(offset.begin() + dim);
                            AdjustTileShapeForReduce(dim, res, args->tileShape_);
                        }
                        Assemble(res, offset, outputs[0]);
                    }
                }
            }
        }
    }
}

class ProdOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ProdOpMetadata> {};

INSTANTIATE_TEST_SUITE_P(TestProd, ProdOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<ProdOpMetadata>(
                             {ProdOperationExeFunc, Prod3DOperationExeFunc, Prod4DOperationExeFunc}, "Prod")));

TEST_P(ProdOperationTest, TestProd)
{
    auto test_data = GetParam().test_data_;
    auto args = ProdOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data),
                               GetValueByName<std::vector<int64_t>>(test_data, "dims"),
                               GetValueByName<bool>(test_data, "keepDim"));
    auto testCase = CreateTestCaseDesc<ProdOpMetadata>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

} // namespace
