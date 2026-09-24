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
 * \file test_batchmatmul_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct BatchMatmulOpFuncArgs : public OpFuncArgs {
    BatchMatmulOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<std::vector<int64_t>>& tileShape,
                          const MatmulTestCaseParam& param)
        : viewShape_(viewShape), tileShape_(tileShape), param_(param)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<std::vector<int64_t>> tileShape_;
    MatmulTestCaseParam param_;
};

struct BatchMatmulOpMetaData {
    explicit BatchMatmulOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

struct BatchMatmulTileParam {
    bool transA;
    bool transB;
    int mDim;
    int kDim;
    int nDim;
    int mView;
    int nView;

    std::vector<int64_t> aViewShape;
    std::vector<int64_t> bViewShape;
    std::vector<SymbolicScalar> aValidShape;
    std::vector<SymbolicScalar> bValidShape;
    std::vector<SymbolicScalar> aOffset;
    std::vector<SymbolicScalar> bOffset;
    std::vector<SymbolicScalar> cOffset;

    std::vector<int64_t> vecTileShape;
};

static void GetBatchMatmulTileParam(const std::vector<Tensor>& inputs, const OpFuncArgs* opArgs,
                                    BatchMatmulTileParam& tileParam)
{
    auto args = static_cast<const BatchMatmulOpFuncArgs*>(opArgs);
    tileParam.transA = args->param_.transA;
    tileParam.transB = args->param_.transB;
    size_t inputDim = inputs[0].GetShape().size();
    const size_t DIM_OFFSET_2 = 2;
    tileParam.mDim = tileParam.transA ? inputs[0].GetShape()[inputDim - 1] :
                                        inputs[0].GetShape()[inputDim - DIM_OFFSET_2];
    tileParam.kDim = tileParam.transA ? inputs[0].GetShape()[inputDim - DIM_OFFSET_2] :
                                        inputs[0].GetShape()[inputDim - 1];
    tileParam.nDim = tileParam.transB ? inputs[1].GetShape()[inputDim - DIM_OFFSET_2] :
                                        inputs[1].GetShape()[inputDim - 1];
    tileParam.mView = args->viewShape_[inputDim - DIM_OFFSET_2];
    tileParam.nView = args->viewShape_[inputDim - 1UL];

    tileParam.aViewShape = {inputs[0].GetShape().begin(), inputs[0].GetShape().end() - DIM_OFFSET_2};
    tileParam.bViewShape = {inputs[1].GetShape().begin(), inputs[1].GetShape().end() - DIM_OFFSET_2};

    tileParam.aValidShape = {inputs[0].GetShape().begin(), inputs[0].GetShape().end() - DIM_OFFSET_2};
    tileParam.bValidShape = {inputs[1].GetShape().begin(), inputs[1].GetShape().end() - DIM_OFFSET_2};

    tileParam.aOffset = std::vector<SymbolicScalar>(inputDim - DIM_OFFSET_2, 0);
    tileParam.bOffset = std::vector<SymbolicScalar>(inputDim - DIM_OFFSET_2, 0);
    tileParam.cOffset = std::vector<SymbolicScalar>(inputDim - DIM_OFFSET_2, 0);

    tileParam.vecTileShape = std::vector<int64_t>(inputDim - DIM_OFFSET_2, 1);
    tileParam.vecTileShape.insert(tileParam.vecTileShape.end(), {args->tileShape_[0][1], args->tileShape_[1][1]});
}

static Tensor CallBatchMatmulOp(const Tensor& tensorA, const Tensor& tensorB, const MatmulTestCaseParam& param)
{
    if (!param.transA && !param.transB && !param.isCMatrixNz) {
        return Matrix::BatchMatmul(param.outDtype, tensorA, tensorB, false, false, false);
    } else if (!param.transA && !param.transB && param.isCMatrixNz) {
        return Matrix::BatchMatmul(param.outDtype, tensorA, tensorB, false, false, true);
    } else if (!param.transA && param.transB && !param.isCMatrixNz) {
        return Matrix::BatchMatmul(param.outDtype, tensorA, tensorB, false, true, false);
    } else if (!param.transA && param.transB && param.isCMatrixNz) {
        return Matrix::BatchMatmul(param.outDtype, tensorA, tensorB, false, true, true);
    } else if (param.transA && !param.transB && !param.isCMatrixNz) {
        return Matrix::BatchMatmul(param.outDtype, tensorA, tensorB, true, false, false);
    } else if (param.transA && !param.transB && param.isCMatrixNz) {
        return Matrix::BatchMatmul(param.outDtype, tensorA, tensorB, true, false, true);
    } else if (param.transA && param.transB && !param.isCMatrixNz) {
        return Matrix::BatchMatmul(param.outDtype, tensorA, tensorB, true, true, false);
    } else {
        return Matrix::BatchMatmul(param.outDtype, tensorA, tensorB, true, true, true);
    }
}

static void BatchMatmulOperationExeFuncNoSplit(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    auto args = static_cast<const BatchMatmulOpFuncArgs*>(opArgs);
    BatchMatmulTileParam tileParam;
    GetBatchMatmulTileParam(inputs, opArgs, tileParam);
    size_t inputDim = inputs[0].GetShape().size();
    tileParam.aOffset = std::vector<SymbolicScalar>(inputDim, 0);
    tileParam.bOffset = std::vector<SymbolicScalar>(inputDim, 0);
    tileParam.aValidShape = {inputs[0].GetShape().begin(), inputs[0].GetShape().end()};
    tileParam.bValidShape = {inputs[1].GetShape().begin(), inputs[1].GetShape().end()};

    FUNCTION("testNoSplit", {inputs[0], inputs[1]}, {outputs[0]})
    {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1))
        {
            tileParam.aOffset[inputDim - 1] = mIdx;
            Tensor tensorA = View(inputs[0], inputs[0].GetShape(), tileParam.aValidShape, tileParam.aOffset);
            Tensor tensorB = View(inputs[1], inputs[1].GetShape(), tileParam.bValidShape, tileParam.bOffset);
            TileShape::Current().SetCubeTile({args->tileShape_[0][0], args->tileShape_[0][1]},
                                             {args->tileShape_[1][0], args->tileShape_[1][1]},
                                             {args->tileShape_[2][0], args->tileShape_[2][1]});
            if (args->param_.isAMatrixNz || args->param_.isBMatrixNz || args->param_.isCMatrixNz) {
                TileShape::Current().SetMatrixSize({tileParam.mDim, tileParam.kDim, tileParam.nDim});
            }
            outputs[0] = CallBatchMatmulOp(tensorA, tensorB, args->param_);
        }
    }
}

static void BatchMatmulOperationExeFuncSplitM(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                              const OpFuncArgs* opArgs)
{
    auto args = static_cast<const BatchMatmulOpFuncArgs*>(opArgs);
    BatchMatmulTileParam tileParam;
    GetBatchMatmulTileParam(inputs, opArgs, tileParam);
    tileParam.bValidShape = {inputs[1].GetShape().begin(), inputs[1].GetShape().end()};

    FUNCTION("testMSplit", {inputs[0], inputs[1]}, {outputs[0]})
    {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx,
             LoopRange(0, CeilDivSymbolicScalar(tileParam.mDim, tileParam.mView), 1))
        {
            if (tileParam.transA) {
                tileParam.aOffset.insert(tileParam.aOffset.end(), {0, mIdx * tileParam.mView});
                tileParam.aViewShape.insert(tileParam.aViewShape.end(), {tileParam.kDim, tileParam.mView});
                tileParam.aValidShape.insert(
                    tileParam.aValidShape.end(),
                    {tileParam.kDim, std::min(tileParam.mDim - tileParam.mView * mIdx, tileParam.mView)});
            } else {
                tileParam.aOffset.insert(tileParam.aOffset.end(), {mIdx * tileParam.mView, 0});
                tileParam.aViewShape.insert(tileParam.aViewShape.end(), {tileParam.mView, tileParam.kDim});
                tileParam.aValidShape.insert(
                    tileParam.aValidShape.end(),
                    {std::min(tileParam.mDim - tileParam.mView * mIdx, tileParam.mView), tileParam.kDim});
            }
            Tensor tensorA = View(inputs[0], tileParam.aViewShape, tileParam.aValidShape, tileParam.aOffset);

            tileParam.bOffset.insert(tileParam.bOffset.end(), {0, 0});
            Tensor tensorB = View(inputs[1], inputs[1].GetShape(), tileParam.bValidShape, tileParam.bOffset);

            TileShape::Current().SetVecTile(tileParam.vecTileShape);
            TileShape::Current().SetCubeTile({args->tileShape_[0][0], args->tileShape_[0][1]},
                                             {args->tileShape_[1][0], args->tileShape_[1][1]},
                                             {args->tileShape_[2][0], args->tileShape_[2][1]});
            if (args->param_.isAMatrixNz || args->param_.isBMatrixNz || args->param_.isCMatrixNz) {
                TileShape::Current().SetMatrixSize({tileParam.mDim, tileParam.kDim, tileParam.nDim});
            }
            Tensor tensorC = CallBatchMatmulOp(tensorA, tensorB, args->param_);
            tileParam.cOffset.insert(tileParam.cOffset.end(), {mIdx * tileParam.mView, 0});
            Assemble(tensorC, tileParam.cOffset, outputs[0]);
        }
    }
}

static void BatchMatmulOperationExeFuncSplitN(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                              const OpFuncArgs* opArgs)
{
    auto args = static_cast<const BatchMatmulOpFuncArgs*>(opArgs);
    BatchMatmulTileParam tileParam;
    GetBatchMatmulTileParam(inputs, opArgs, tileParam);
    tileParam.aValidShape = {inputs[0].GetShape().begin(), inputs[0].GetShape().end()};

    FUNCTION("testNSplit", {inputs[0], inputs[1]}, {outputs[0]})
    {
        LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx,
             LoopRange(0, CeilDivSymbolicScalar(tileParam.nDim, tileParam.nView), 1))
        {
            tileParam.aOffset.insert(tileParam.aOffset.end(), {0, 0});
            Tensor tensorA = View(inputs[0], inputs[0].GetShape(), tileParam.aValidShape, tileParam.aOffset);
            if (tileParam.transB) {
                tileParam.bOffset.insert(tileParam.bOffset.end(), {nIdx * tileParam.nView, 0});
                tileParam.bViewShape.insert(tileParam.bViewShape.end(), {tileParam.nView, tileParam.kDim});
                tileParam.bValidShape.insert(
                    tileParam.bValidShape.end(),
                    {std::min(tileParam.nDim - nIdx * tileParam.nView, tileParam.nView), tileParam.kDim});
            } else {
                tileParam.bOffset.insert(tileParam.bOffset.end(), {0, nIdx * tileParam.nView});
                tileParam.bViewShape.insert(tileParam.bViewShape.end(), {tileParam.kDim, tileParam.nView});
                tileParam.bValidShape.insert(
                    tileParam.bValidShape.end(),
                    {tileParam.kDim, std::min(tileParam.nDim - nIdx * tileParam.nView, tileParam.nView)});
            }
            Tensor tensorB = View(inputs[1], tileParam.bViewShape, tileParam.bValidShape, tileParam.bOffset);
            TileShape::Current().SetCubeTile({args->tileShape_[0][0], args->tileShape_[0][1]},
                                             {args->tileShape_[1][0], args->tileShape_[1][1]},
                                             {args->tileShape_[2][0], args->tileShape_[2][1]});
            TileShape::Current().SetVecTile(tileParam.vecTileShape);
            if (args->param_.isAMatrixNz || args->param_.isBMatrixNz || args->param_.isCMatrixNz) {
                TileShape::Current().SetMatrixSize({tileParam.mDim, tileParam.kDim, tileParam.nDim});
            }
            Tensor tensorC = CallBatchMatmulOp(tensorA, tensorB, args->param_);
            tileParam.cOffset.insert(tileParam.cOffset.end(), {0, nIdx * tileParam.nView});
            Assemble(tensorC, tileParam.cOffset, outputs[0]);
        }
    }
}

static void BatchMatmulOperationExeFuncSplitMN(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    auto args = static_cast<const BatchMatmulOpFuncArgs*>(opArgs);
    BatchMatmulTileParam tileParam;
    GetBatchMatmulTileParam(inputs, opArgs, tileParam);

    FUNCTION("testMNSplit", {inputs[0], inputs[1]}, {outputs[0]})
    {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx,
             LoopRange(0, CeilDivSymbolicScalar(tileParam.mDim, tileParam.mView), 1))
        {
            LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx,
                 LoopRange(0, CeilDivSymbolicScalar(tileParam.nDim, tileParam.nView), 1))
            {
                if (tileParam.transA) {
                    tileParam.aViewShape.insert(tileParam.aViewShape.end(), {tileParam.kDim, tileParam.mView});
                    tileParam.aValidShape.insert(
                        tileParam.aValidShape.end(),
                        {tileParam.kDim, std::min(tileParam.mDim - tileParam.mView * mIdx, tileParam.mView)});
                    tileParam.aOffset.insert(tileParam.aOffset.end(), {0, mIdx * tileParam.mView});
                } else {
                    tileParam.aViewShape.insert(tileParam.aViewShape.end(), {tileParam.mView, tileParam.kDim});
                    tileParam.aValidShape.insert(
                        tileParam.aValidShape.end(),
                        {std::min(tileParam.mDim - tileParam.mView * mIdx, tileParam.mView), tileParam.kDim});
                    tileParam.aOffset.insert(tileParam.aOffset.end(), {mIdx * tileParam.mView, 0});
                }
                Tensor tensorA = View(inputs[0], tileParam.aViewShape, tileParam.aValidShape, tileParam.aOffset);

                if (tileParam.transB) {
                    tileParam.bViewShape.insert(tileParam.bViewShape.end(), {tileParam.nView, tileParam.kDim});
                    tileParam.bValidShape.insert(
                        tileParam.bValidShape.end(),
                        {std::min(tileParam.nDim - nIdx * tileParam.nView, tileParam.nView), tileParam.kDim});
                    tileParam.bOffset.insert(tileParam.bOffset.end(), {nIdx * tileParam.nView, 0});
                } else {
                    tileParam.bViewShape.insert(tileParam.bViewShape.end(), {tileParam.kDim, tileParam.nView});
                    tileParam.bValidShape.insert(
                        tileParam.bValidShape.end(),
                        {tileParam.kDim, std::min(tileParam.nDim - nIdx * tileParam.nView, tileParam.nView)});
                    tileParam.bOffset.insert(tileParam.bOffset.end(), {0, nIdx * tileParam.nView});
                }
                Tensor tensorB = View(inputs[1], tileParam.bViewShape, tileParam.bValidShape, tileParam.bOffset);

                TileShape::Current().SetVecTile(tileParam.vecTileShape);
                TileShape::Current().SetCubeTile({args->tileShape_[0][0], args->tileShape_[0][1]},
                                                 {args->tileShape_[1][0], args->tileShape_[1][1]},
                                                 {args->tileShape_[2][0], args->tileShape_[2][1]});
                if (args->param_.isAMatrixNz || args->param_.isBMatrixNz || args->param_.isCMatrixNz) {
                    TileShape::Current().SetMatrixSize({tileParam.mDim, tileParam.kDim, tileParam.nDim});
                }
                Tensor tensorC = CallBatchMatmulOp(tensorA, tensorB, args->param_);
                tileParam.cOffset.insert(tileParam.cOffset.end(), {mIdx * tileParam.mView, nIdx * tileParam.nView});
                Assemble(tensorC, tileParam.cOffset, outputs[0]);
            }
        }
    }
}

static void BatchMatmulOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    ASSERT(inputs[0].GetShape().size() == inputs[1].GetShape().size());
    auto args = static_cast<const BatchMatmulOpFuncArgs*>(opArgs);
    size_t inputDim = inputs[0].GetShape().size();
    ASSERT(args->viewShape_.size() == inputDim);
    const size_t DIM_OFFSET_2 = 2;
    const int mView = args->viewShape_[inputDim - DIM_OFFSET_2];
    const int nView = args->viewShape_[inputDim - 1UL];

    if (mView > 0 && nView > 0) {
        return BatchMatmulOperationExeFuncSplitMN(inputs, outputs, opArgs);
    } else if (mView > 0) {
        return BatchMatmulOperationExeFuncSplitM(inputs, outputs, opArgs);
    } else if (nView > 0) {
        return BatchMatmulOperationExeFuncSplitN(inputs, outputs, opArgs);
    } else {
        return BatchMatmulOperationExeFuncNoSplit(inputs, outputs, opArgs);
    }
}

class BatchMatmulOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<BatchMatmulOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestBatchMatmul, BatchMatmulOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<BatchMatmulOpMetaData>({BatchMatmulOperationExeFunc},
                                                                                  "BatchMatmul")));

TEST_P(BatchMatmulOperationTest, TestBatchMatmul)
{
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetMatmulTensors(test_data, "input_tensors");
    testCase.outputTensors = GetMatmulTensors(test_data, "output_tensors");
    auto args = BatchMatmulOpFuncArgs(GetViewShape(test_data), GetMatmulTileShape(test_data),
                                      GetMatmulParam(test_data));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0].GetStorage()->Symbol() + ".bin",
                           GetGoldenDir() + "/" + testCase.inputTensors[1].GetStorage()->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0].GetStorage()->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}

class BatchMatmulVerifyOperationTest
    : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<BatchMatmulOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestBatchMatmulVerify, BatchMatmulVerifyOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<BatchMatmulOpMetaData>({BatchMatmulOperationExeFunc},
                                                                                  "BatchMatmulVerify")));

TEST_P(BatchMatmulVerifyOperationTest, TestBatchMatmulVerify)
{
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.outputTensors = GetMatmulTensors(test_data, "output_tensors");
    testCase.inputTensors = GetMatmulTensors(test_data, "input_tensors");
    auto args = BatchMatmulOpFuncArgs(GetViewShape(test_data), GetMatmulTileShape(test_data),
                                      GetMatmulParam(test_data));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0].GetStorage()->Symbol() + ".bin",
                           GetGoldenDir() + "/" + testCase.inputTensors[1].GetStorage()->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0].GetStorage()->Symbol() + ".bin"};
    TestFlowVerifier::runTest(testCase);
}
} // namespace
