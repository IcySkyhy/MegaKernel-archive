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
 * \file test_matmul_operation.cpp
 * \brief
 */

#include "test_operation.h"
#include "interface/operation/operation_impl.h"

using namespace tile_fwk::test_operation;
namespace {
constexpr int scaleIndex = 2;
constexpr int biasIndex = 3;
constexpr int l0cToL1Index = 4;

struct MatmulOpFuncArgs : public OpFuncArgs {
    MatmulOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<std::vector<int64_t>>& tileShape,
                     const MatmulTestCaseParam& param)
        : viewShape_(viewShape), tileShape_(tileShape), param_(param)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<std::vector<int64_t>> tileShape_;
    MatmulTestCaseParam param_;
};

struct MatmulOpMetaData {
    explicit MatmulOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static Tensor CallMatmulOp(const Tensor& tensorA, const Tensor& tensorB, const MatmulTestCaseParam& param,
                           const Matrix::MatmulExtendParam& matmulExtendParam)
{
    return Matrix::Matmul(param.outDtype, tensorA, tensorB, matmulExtendParam, param.transA, param.transB,
                          param.isCMatrixNz);
}

static Tensor CallMatmulOpWithL0C2L1(const Tensor& tensorA, const Tensor& tensorB, const vector<int>& transInfo,
                                     const bool& isCMatrixNz, const DataType& outDtype)
{
    // 2: transinfo vector has two elements
    ASSERT(transInfo.size() == 2);
    return Matrix::Matmul(outDtype, tensorA, tensorB, transInfo.at(0), transInfo.at(1), isCMatrixNz);
}

static Tensor CallMatmulOpWithL0C2L1AndScale(const Tensor& tensorA, const Tensor& tensorB,
                                             const vector<int>& transInform, const bool& isCMatrixNz,
                                             const DataType& outDtype,
                                             const Matrix::MatmulExtendParam& matmulExtendParam)
{
    // 2: transinfo vector has two elements
    ASSERT(transInform.size() == 2);
    return Matrix::Matmul(outDtype, tensorA, tensorB, matmulExtendParam, transInform.at(0), transInform.at(1),
                          isCMatrixNz);
}

static void MatmulOperationExeFuncNoSplitWithL0C2L1(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                                    const OpFuncArgs* opArgs)
{
    auto args = static_cast<const MatmulOpFuncArgs*>(opArgs);
    SymbolicScalar mDim = args->param_.transA ? inputs[0].GetShape()[1] : inputs[0].GetShape()[0];
    SymbolicScalar kDim = args->param_.transA ? inputs[0].GetShape()[0] : inputs[0].GetShape()[1];
    SymbolicScalar nDim = args->param_.transB ? inputs[1].GetShape()[0] : inputs[1].GetShape()[1];
    SymbolicScalar tensorcMdim = args->param_.l0c2l1IsTrans ? inputs[l0cToL1Index].GetShape()[1] :
                                                              inputs[l0cToL1Index].GetShape()[0];
    SymbolicScalar tensorcNdim = args->param_.l0c2l1IsTrans ? inputs[l0cToL1Index].GetShape()[0] :
                                                              inputs[l0cToL1Index].GetShape()[1];

    FUNCTION("testNoSplit", {inputs[0], inputs[1], inputs[scaleIndex], inputs[biasIndex], inputs[l0cToL1Index]},
             {outputs[0]})
    {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1))
        {
            Tensor tensorL0c2L1;
            if (args->param_.l0c2l1IsTrans) {
                // 2: l0c2l1的tensor的index
                tensorL0c2L1 = View(inputs[l0cToL1Index], {tensorcNdim, tensorcMdim}, {tensorcNdim, tensorcMdim},
                                    {0, 0});
            } else {
                // 2: l0c2l1的tensor的index
                tensorL0c2L1 = View(inputs[l0cToL1Index], {tensorcMdim, tensorcNdim}, {tensorcMdim, tensorcNdim},
                                    {0, 0});
            }
            Tensor tensorB;
            if (args->param_.transB) {
                tensorB = View(inputs[1], {nDim, kDim}, {nDim, kDim}, {0, 0});
            } else {
                tensorB = View(inputs[1], {kDim, nDim}, {kDim, nDim}, {0, 0});
            }
            Tensor tensorA;
            if (args->param_.transA) {
                tensorA = View(inputs[0], {kDim, mDim}, {kDim, mDim}, {0, 0});
            } else {
                tensorA = View(inputs[0], {mDim, kDim}, {mDim, kDim}, {mIdx, 0});
            }
            Matrix::MatmulExtendParam param;
            param.scaleValue = args->param_.scaleValue;
            param.reluType = static_cast<Matrix::ReLuType>(args->param_.reluTypeInt);
            if (args->param_.hasBias) {
                param.biasTensor = View(inputs[biasIndex], {1, nDim}, {1, nDim}, {0, 0});
            }
            if (args->param_.hasScale) {
                param.scaleTensor = View(inputs[scaleIndex], {1, nDim}, {1, nDim}, {0, 0});
            }
            Tensor tensorTmp = CallMatmulOpWithL0C2L1AndScale(tensorA, tensorB,
                                                              {args->param_.transA, args->param_.transB},
                                                              args->param_.l0c2l1IsNz, args->param_.outDtype, param);
            if (args->param_.l0c2l1AsLeftMatrix) {
                outputs[0] = CallMatmulOpWithL0C2L1(tensorL0c2L1, tensorTmp,
                                                    {args->param_.l0c2l1IsTrans, args->param_.l0c2l1TmpIsTrans},
                                                    args->param_.isCMatrixNz, args->param_.outDtype);
            } else {
                outputs[0] = CallMatmulOpWithL0C2L1(tensorTmp, tensorL0c2L1,
                                                    {args->param_.l0c2l1TmpIsTrans, args->param_.l0c2l1IsTrans},
                                                    args->param_.isCMatrixNz, args->param_.outDtype);
            }
        }
    }
}

static void MatmulOperationExeFuncNoSplit(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    auto args = static_cast<const MatmulOpFuncArgs*>(opArgs);
    bool transA = args->param_.transA;
    bool transB = args->param_.transB;
    float scaleValue = args->param_.scaleValue;
    int reluTypeInt = args->param_.reluTypeInt;
    Matrix::ReLuType reluType = static_cast<Matrix::ReLuType>(reluTypeInt);
    SymbolicScalar mDim = transA ? inputs[0].GetShape()[1] : inputs[0].GetShape()[0];
    SymbolicScalar kDim = transA ? inputs[0].GetShape()[0] : inputs[0].GetShape()[1];
    SymbolicScalar nDim = transB ? inputs[1].GetShape()[0] : inputs[1].GetShape()[1];

    FUNCTION("testNoSplit", {inputs[0], inputs[1], inputs[scaleIndex], inputs[biasIndex]}, {outputs[0]})
    {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1))
        {
            Tensor tensorA;
            if (transA) {
                tensorA = View(inputs[0], {kDim, mDim}, {kDim, mDim}, {0, 0});
            } else {
                tensorA = View(inputs[0], {mDim, kDim}, {mDim, kDim}, {mIdx, 0});
            }
            Tensor tensorB;
            if (transB) {
                tensorB = View(inputs[1], {nDim, kDim}, {nDim, kDim}, {0, 0});
            } else {
                tensorB = View(inputs[1], {kDim, nDim}, {kDim, nDim}, {0, 0});
            }
            Matrix::MatmulExtendParam param;
            param.scaleValue = scaleValue;
            param.reluType = reluType;
            if (args->param_.hasScale) {
                param.scaleTensor = View(inputs[scaleIndex], {1, nDim}, {1, nDim}, {0, 0});
            }
            if (args->param_.hasBias) {
                param.biasTensor = View(inputs[biasIndex], {1, nDim}, {1, nDim}, {0, 0});
            }
            outputs[0] = CallMatmulOp(tensorA, tensorB, args->param_, param);
        }
    }
}

static void MatmulOperationExeFuncSplitM(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    auto args = static_cast<const MatmulOpFuncArgs*>(opArgs);
    const int64_t mView = args->viewShape_[0];
    bool transA = args->param_.transA;
    bool transB = args->param_.transB;
    SymbolicScalar mDim = transA ? inputs[0].GetShape()[1] : inputs[0].GetShape()[0];
    SymbolicScalar kDim = transA ? inputs[0].GetShape()[0] : inputs[0].GetShape()[1];
    SymbolicScalar nDim = transB ? inputs[1].GetShape()[0] : inputs[1].GetShape()[1];
    float scaleValue = args->param_.scaleValue;
    int reluTypeInt = args->param_.reluTypeInt;
    Matrix::ReLuType reluType = static_cast<Matrix::ReLuType>(reluTypeInt);

    FUNCTION("testMSplit", {inputs[0], inputs[1], inputs[scaleIndex], inputs[biasIndex]}, {outputs[0]})
    {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, CeilDivSymbolicScalar(mDim, mView), 1))
        {
            Tensor tensorA;
            if (transA) {
                tensorA = View(inputs[0], {kDim, mView}, {kDim, std::min(mDim - mView * mIdx, mView)},
                               {0, mIdx * mView});
            } else {
                tensorA = View(inputs[0], {mView, kDim}, {std::min(mDim - mView * mIdx, mView), kDim},
                               {mIdx * mView, 0});
            }
            Tensor tensorB;
            if (transB) {
                tensorB = View(inputs[1], {nDim, kDim}, {nDim, kDim}, {0, 0});
            } else {
                tensorB = View(inputs[1], {kDim, nDim}, {kDim, nDim}, {0, 0});
            }
            Matrix::MatmulExtendParam param;
            if (args->param_.hasBias) {
                param.biasTensor = View(inputs[biasIndex], {1, nDim}, {1, nDim}, {0, 0});
            }
            if (args->param_.hasScale) {
                param.scaleTensor = View(inputs[scaleIndex], {1, nDim}, {1, nDim}, {0, 0});
            }
            param.reluType = reluType;
            param.scaleValue = scaleValue;
            Tensor tensorC = CallMatmulOp(tensorA, tensorB, args->param_, param);
            Assemble(tensorC, {mIdx * mView, 0}, outputs[0]);
        }
    }
}

static void MatmulOperationExeFuncSplitN(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    auto args = static_cast<const MatmulOpFuncArgs*>(opArgs);
    bool transA = args->param_.transA;
    bool transB = args->param_.transB;
    int reluTypeInt = args->param_.reluTypeInt;
    Matrix::ReLuType reluType = static_cast<Matrix::ReLuType>(reluTypeInt);
    float scaleValue = args->param_.scaleValue;
    SymbolicScalar mDim = transA ? inputs[0].GetShape()[1] : inputs[0].GetShape()[0];
    SymbolicScalar kDim = transA ? inputs[0].GetShape()[0] : inputs[0].GetShape()[1];
    SymbolicScalar nDim = transB ? inputs[1].GetShape()[0] : inputs[1].GetShape()[1];
    const int64_t nView = args->viewShape_[1];

    FUNCTION("testNSplit", {inputs[0], inputs[1], inputs[scaleIndex], inputs[biasIndex]}, {outputs[0]})
    {
        LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, CeilDivSymbolicScalar(nDim, nView), 1))
        {
            Tensor tensorA;
            if (transA) {
                tensorA = View(inputs[0], {kDim, mDim}, {kDim, mDim}, {0, 0});
            } else {
                tensorA = View(inputs[0], {mDim, kDim}, {mDim, kDim}, {0, 0});
            }
            Tensor tensorB;
            if (transB) {
                tensorB = View(inputs[1], {nView, kDim}, {std::min(nDim - nIdx * nView, nView), kDim},
                               {nIdx * nView, 0});
            } else {
                tensorB = View(inputs[1], {kDim, nView}, {kDim, std::min(nDim - nIdx * nView, nView)},
                               {0, nIdx * nView});
            }
            Matrix::MatmulExtendParam param;
            param.reluType = reluType;
            param.scaleValue = scaleValue;
            if (args->param_.hasBias) {
                param.biasTensor = View(inputs[biasIndex], {1, nView}, {1, std::min(nDim - nIdx * nView, nView)},
                                        {0, nIdx * nView});
            }
            if (args->param_.hasScale) {
                param.scaleTensor = View(inputs[scaleIndex], {1, nView}, {1, std::min(nDim - nIdx * nView, nView)},
                                         {0, nIdx * nView});
            }
            Tensor tensorC = CallMatmulOp(tensorA, tensorB, args->param_, param);
            Assemble(tensorC, {0, nIdx * nView}, outputs[0]);
        }
    }
}

static void MatmulOperationExeFuncSplitMN(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    auto args = static_cast<const MatmulOpFuncArgs*>(opArgs);
    float scaleValue = args->param_.scaleValue;
    int reluTypeInt = args->param_.reluTypeInt;
    Matrix::ReLuType reluType = static_cast<Matrix::ReLuType>(reluTypeInt);
    bool transA = args->param_.transA;
    bool transB = args->param_.transB;
    SymbolicScalar mDim = transA ? inputs[0].GetShape()[1] : inputs[0].GetShape()[0];
    SymbolicScalar kDim = transA ? inputs[0].GetShape()[0] : inputs[0].GetShape()[1];
    SymbolicScalar nDim = transB ? inputs[1].GetShape()[0] : inputs[1].GetShape()[1];
    const int64_t mView = args->viewShape_[0];
    const int64_t nView = args->viewShape_[1];

    FUNCTION("testMNSplit", {inputs[0], inputs[1], inputs[scaleIndex], inputs[biasIndex]}, {outputs[0]})
    {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, CeilDivSymbolicScalar(mDim, mView), 1))
        {
            LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, CeilDivSymbolicScalar(nDim, nView), 1))
            {
                Tensor tensorA;
                if (transA) {
                    tensorA = View(inputs[0], {kDim, mView}, {kDim, std::min(mDim - mView * mIdx, mView)},
                                   {0, mIdx * mView});
                } else {
                    tensorA = View(inputs[0], {mView, kDim}, {std::min(mDim - mView * mIdx, mView), kDim},
                                   {mIdx * mView, 0});
                }
                Tensor tensorB;
                if (transB) {
                    tensorB = View(inputs[1], {nView, kDim}, {std::min(nDim - nIdx * nView, nView), kDim},
                                   {nIdx * nView, 0});
                } else {
                    tensorB = View(inputs[1], {kDim, nView}, {kDim, std::min(nDim - nIdx * nView, nView)},
                                   {0, nIdx * nView});
                }
                Matrix::MatmulExtendParam param;
                if (args->param_.hasScale) {
                    param.scaleTensor = View(inputs[scaleIndex], {1, nView}, {1, std::min(nDim - nIdx * nView, nView)},
                                             {0, nIdx * nView});
                }
                if (args->param_.hasBias) {
                    param.biasTensor = View(inputs[biasIndex], {1, nView}, {1, std::min(nDim - nIdx * nView, nView)},
                                            {0, nIdx * nView});
                }
                param.reluType = reluType;
                param.scaleValue = scaleValue;
                Tensor tensorC = CallMatmulOp(tensorA, tensorB, args->param_, param);
                Assemble(tensorC, {mIdx * mView, nIdx * nView}, outputs[0]);
            }
        }
    }
}

static void MatmulOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                   const OpFuncArgs* opArgs)
{
    auto args = static_cast<const MatmulOpFuncArgs*>(opArgs);
    if (args->param_.hasScale || args->param_.hasBias) {
        int64_t nTile = (args->tileShape_[2][0] < args->tileShape_[2][1]) ? args->tileShape_[2][0] :
                                                                            args->tileShape_[2][1];
        TileShape::Current().SetCubeTile({args->tileShape_[0][0], args->tileShape_[0][1]},
                                         {args->tileShape_[1][0], args->tileShape_[1][1]}, {nTile, nTile},
                                         args->param_.enableKSplit);
    } else {
        TileShape::Current().SetCubeTile({args->tileShape_[0][0], args->tileShape_[0][1]},
                                         {args->tileShape_[1][0], args->tileShape_[1][1]},
                                         {args->tileShape_[2][0], args->tileShape_[2][1]}, args->param_.enableKSplit);
    }

    const size_t MM_VIEW_SHAPE_DIM = 2;
    ASSERT(args->viewShape_.size() == MM_VIEW_SHAPE_DIM);
    const int64_t mView = args->viewShape_[0];
    const int64_t nView = args->viewShape_[1];
    if (args->param_.enable_l0c2l1) {
        return MatmulOperationExeFuncNoSplitWithL0C2L1(inputs, outputs, opArgs);
    }
    if (mView > 0 && nView > 0) {
        return MatmulOperationExeFuncSplitMN(inputs, outputs, opArgs);
    } else if (mView > 0) {
        return MatmulOperationExeFuncSplitM(inputs, outputs, opArgs);
    } else if (nView > 0) {
        return MatmulOperationExeFuncSplitN(inputs, outputs, opArgs);
    } else {
        return MatmulOperationExeFuncNoSplit(inputs, outputs, opArgs);
    }
}

static void CheckBTransNZUnaligned(bool transB, bool isCMatrixNz, const Tensor& b, const Tensor& c)
{
    constexpr int blockAlignBytes = 32;
    ASSERT(BytesOf(c.GetDataType()) != 0) << "wrong data type";
    int innerNum = blockAlignBytes / BytesOf(c.GetDataType());
    SymbolicScalar bN = transB ? b.GetShape()[0] : b.GetShape()[1];
    SymbolicScalar cN = c.GetShape()[1];
    bool nNotEqualCase = !transB && isCMatrixNz && cN % innerNum == 0;
    // (N, K)输入，NZ输出是，由于ND2NZ指令无法在N轴补零，最终NPU输出在N轴可能存在脏数据，暂不支持。
    ASSERT(bN == cN || nNotEqualCase) << "N, K shape for NZ output format with N unaligned is not supported";
}

class MatmulOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<MatmulOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestMatmul, MatmulOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<MatmulOpMetaData>({MatmulOperationExeFunc}, "Matmul")));

TEST_P(MatmulOperationTest, TestMatmul)
{
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    ConfigManagerNg::JitScopeGuard configGuard(
        "matmul_test", std::map<std::string, std::any>{{"pass." + std::string(ENABLE_SLICE), false}});
    testCase.inputTensors = GetMatmulTensors(test_data, "input_tensors");
    testCase.outputTensors = GetMatmulTensors(test_data, "output_tensors");
    auto args = MatmulOpFuncArgs(GetViewShape(test_data), GetMatmulTileShape(test_data), GetMatmulParam(test_data));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0].GetStorage()->Symbol() + ".bin",
                           GetGoldenDir() + "/" + testCase.inputTensors[1].GetStorage()->Symbol() + ".bin"};
    // scale tensor
    Tensor scaleTensor = GetParamTensor(test_data, "scale_tensors");
    testCase.inputTensors.push_back(scaleTensor);
    if (scaleTensor.GetStorage() != nullptr) {
        testCase.inputPaths.push_back(GetGoldenDir() + "/" + scaleTensor.GetStorage()->Symbol() + ".bin");
    } else {
        testCase.inputPaths.push_back("");
    }
    // bias tensor
    Tensor biasTensor = GetParamTensor(test_data, "bias_tensors");
    testCase.inputTensors.push_back(biasTensor);
    if (biasTensor.GetStorage() != nullptr) {
        testCase.inputPaths.push_back(GetGoldenDir() + "/" + biasTensor.GetStorage()->Symbol() + ".bin");
    } else {
        testCase.inputPaths.push_back("");
    }
    if (args.param_.enable_l0c2l1) {
        Tensor l0c2L1Tensor = GetParamTensor(test_data, "l0c2l1_tensor");
        testCase.inputTensors.push_back(l0c2L1Tensor);
        if (l0c2L1Tensor.GetStorage() != nullptr) {
            testCase.inputPaths.push_back(GetGoldenDir() + "/" + l0c2L1Tensor.GetStorage()->Symbol() + ".bin");
        } else {
            testCase.inputPaths.push_back("");
        }
    }
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0].GetStorage()->Symbol() + ".bin"};
    CheckBTransNZUnaligned(args.param_.transB, args.param_.isCMatrixNz, testCase.inputTensors[1],
                           testCase.outputTensors[0]);
    if (args.param_.enableKSplit) {
        TestExecutor::setGMNotClear();
    }
    TestExecutor::runTest(testCase);
}

class MatmulVerifyOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<MatmulOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestMatmulVerify, MatmulVerifyOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<MatmulOpMetaData>({MatmulOperationExeFunc},
                                                                             "MatmulVerify")));

TEST_P(MatmulVerifyOperationTest, TestMatmulVerify)
{
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.outputTensors = GetMatmulTensors(test_data, "output_tensors");
    testCase.inputTensors = GetMatmulTensors(test_data, "input_tensors");
    auto args = MatmulOpFuncArgs(GetViewShape(test_data), GetMatmulTileShape(test_data), GetMatmulParam(test_data));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0].GetStorage()->Symbol() + ".bin",
                           GetGoldenDir() + "/" + testCase.inputTensors[1].GetStorage()->Symbol() + ".bin"};
    Tensor biasTensor = GetParamTensor(test_data, "bias_tensors");
    Tensor scaleTensor = GetParamTensor(test_data, "scale_tensors");
    if (scaleTensor.GetStorage() == nullptr) {
        testCase.inputPaths.push_back("");
    } else {
        testCase.inputPaths.push_back(GetGoldenDir() + "/" + scaleTensor.GetStorage()->Symbol() + ".bin");
    }
    if (biasTensor.GetStorage() == nullptr) {
        testCase.inputPaths.push_back("");
    } else {
        testCase.inputPaths.push_back(GetGoldenDir() + "/" + biasTensor.GetStorage()->Symbol() + ".bin");
    }
    testCase.inputTensors.push_back(scaleTensor);
    testCase.inputTensors.push_back(biasTensor);
    CheckBTransNZUnaligned(args.param_.transB, args.param_.isCMatrixNz, testCase.inputTensors[1],
                           testCase.outputTensors[0]);
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0].GetStorage()->Symbol() + ".bin"};
    if (args.param_.enable_l0c2l1) {
        Tensor l0c2L1Tensor = GetParamTensor(test_data, "l0c2l1_tensor");
        testCase.inputTensors.push_back(l0c2L1Tensor);
        if (l0c2L1Tensor.GetStorage() != nullptr) {
            testCase.inputPaths.push_back(GetGoldenDir() + "/" + l0c2L1Tensor.GetStorage()->Symbol() + ".bin");
        } else {
            testCase.inputPaths.push_back("");
        }
    }
    TestFlowVerifier::runTest(testCase);
}
} // namespace
