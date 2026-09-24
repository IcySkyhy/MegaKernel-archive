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
 * \file test_clip_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
constexpr const unsigned ID0 = 0;
constexpr const unsigned ID1 = 1;
constexpr const unsigned ID2 = 2;
constexpr const unsigned ID3 = 3;

std::vector<std::reference_wrapper<const Tensor>> AsRef(const std::vector<Tensor>& tensors)
{
    std::vector<std::reference_wrapper<const Tensor>> results;
    results.reserve(tensors.size());
    for (const Tensor& tensor : tensors) {
        results.emplace_back(std::cref(tensor));
    }
    return results;
}

Shape GetBroadCastViewShape(const Tensor& self, const Tensor& other, const Shape& viewShape)
{
    ASSERT(self.GetShape().size() == other.GetShape().size());
    Shape result = viewShape;
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        int64_t selfDim = self.GetShape()[i];
        int64_t otherDim = other.GetShape()[i];
        if (selfDim != otherDim && selfDim == 1 && otherDim != 1) {
            result[i] = 1;
        } else {
            result[i] = std::min(selfDim, viewShape[i]);
        }
    }
    return result;
}

std::vector<int> GetBroadCastOffsetRatio(const Tensor& self, const Tensor& other, const Shape& viewShape)
{
    ASSERT(self.GetShape().size() == other.GetShape().size());
    std::vector<int> result(viewShape.size(), 1);
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        int64_t selfDim = self.GetShape()[i];
        int64_t otherDim = other.GetShape()[i];
        if (selfDim != otherDim && selfDim == 1 && otherDim != 1) {
            result[i] = 0;
        }
    }
    return result;
}

std::vector<SymbolicScalar> GetValidShape(const Shape& originShapes, const Shape& viewShapes,
                                          const std::vector<SymbolicScalar> loopVars)
{
    if (loopVars.size() != viewShapes.size() || originShapes.size() != viewShapes.size()) {
        throw std::invalid_argument("Length of `originShapes`/`viewShapes`/`loopVars` should be the same!");
    }
    std::vector<SymbolicScalar> validShapes(loopVars.size(), 0);
    for (size_t i = 0; i < viewShapes.size(); i++) {
        validShapes[i] = std::min(originShapes[i] - loopVars[i] * viewShapes[i], viewShapes[i]);
    }
    return validShapes;
}

std::vector<SymbolicScalar> GetOffsets(const Shape& viewShapes, const std::vector<SymbolicScalar> loopVars,
                                       std::vector<int> ratios = {})
{
    if (loopVars.size() != viewShapes.size()) {
        throw std::invalid_argument("Length of `loopVars` and `viewShapes` should be the same!");
    }
    if (ratios.empty()) {
        ratios = std::vector<int>(viewShapes.size(), 1);
    }
    std::vector<SymbolicScalar> offsets(loopVars.size(), 0);
    for (size_t i = 0; i < viewShapes.size(); i++) {
        offsets[i] = loopVars[i] * viewShapes[i] * ratios[i];
    }
    return offsets;
}

Tensor BroadCastView(const Tensor& needBroadCast, const Tensor& broadCasted, const Shape& viewShapes,
                     const std::vector<SymbolicScalar> loopVars)
{
    const Shape& tileViewShape = GetBroadCastViewShape(needBroadCast, broadCasted, viewShapes);
    const std::vector<int>& tile0OffsetRatio = GetBroadCastOffsetRatio(needBroadCast, broadCasted, viewShapes);
    std::vector<SymbolicScalar> validShapes = GetValidShape(broadCasted.GetShape(), tileViewShape, loopVars);
    std::vector<SymbolicScalar> offsets = GetOffsets(tileViewShape, loopVars, tile0OffsetRatio);
    Tensor result = View(needBroadCast, tileViewShape, validShapes, offsets);
    return result;
}

enum class TestType : int {
    NotDefault2D = 0,
    NotDefault3D = 1,
    NotDefault4D = 2,
    TensorDefaultMinDefaultMax = 3,
    TensorDefaultMinNotDefaultMax = 4,
    TensorNotDefaultMinDefaultMax = 5,
    ElementDefaultMinDefaultMax = 6,
    ElementNotDefaultMinDefaultMax = 7,
    ElementDefaultMinNotDefaultMax = 8,
    NoValue,
};

struct ClipOpFuncArgs : public OpFuncArgs {
    ClipOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& tileShape, int type,
                   const Element& min = {}, const Element& max = {}, bool isElement = false)
        : viewShape_(viewShape),
          tileShape_(tileShape),
          type_(static_cast<TestType>(type)),
          min_(min),
          max_(max),
          isElement_(isElement)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    TestType type_;
    Element min_;
    Element max_;
    bool isElement_ = false;
};

struct ClipOpMetaData {
    explicit ClipOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

Tensor ProcessElementModeClip(const Tensor& tileTensor0, const ClipOpFuncArgs* args)
{
    Tensor res;
    switch (args->type_) {
        case TestType::NotDefault2D:
        case TestType::NotDefault3D:
        case TestType::NotDefault4D: {
            res = Clip(tileTensor0, args->min_, args->max_);
            break;
        }
        case TestType::ElementDefaultMinDefaultMax: {
            res = Clip(tileTensor0, Element{}, Element{});
            break;
        }
        case TestType::ElementDefaultMinNotDefaultMax: {
            res = Clip(tileTensor0, {}, args->max_);
            break;
        }
        case TestType::ElementNotDefaultMinDefaultMax: {
            res = Clip(tileTensor0, args->min_, {});
            break;
        }
        default: {
            throw std::invalid_argument("This is a NOT supported Tensor-Element case!");
            break;
        }
    }
    return res;
}

Tensor ProcessTensorModeClip(const Tensor& tileTensor0, const std::vector<Tensor>& inputs, const ClipOpFuncArgs* args,
                             const std::vector<SymbolicScalar> loopVars)
{
    Tensor res;
    switch (args->type_) {
        case TestType::NotDefault2D:
        case TestType::NotDefault3D:
        case TestType::NotDefault4D: {
            Tensor minTensor = BroadCastView(inputs[ID1], inputs[ID0], args->viewShape_, loopVars);
            Tensor maxTensor = BroadCastView(inputs[ID2], inputs[ID0], args->viewShape_, loopVars);
            res = Clip(tileTensor0, minTensor, maxTensor);
            break;
        }
        case TestType::TensorDefaultMinDefaultMax: {
            res = Clip(tileTensor0, Tensor{}, Tensor{});
            break;
        }
        case TestType::TensorDefaultMinNotDefaultMax: {
            Tensor maxTensor = BroadCastView(inputs[ID2], inputs[ID0], args->viewShape_, loopVars);
            res = Clip(tileTensor0, {}, maxTensor);
            break;
        }
        case TestType::TensorNotDefaultMinDefaultMax: {
            Tensor minTensor = BroadCastView(inputs[ID1], inputs[ID0], args->viewShape_, loopVars);
            res = Clip(tileTensor0, minTensor, {});
            break;
        }
        default: {
            throw std::invalid_argument("This is a NOT supported Tensor-Tensor case!");
            break;
        }
    }
    return res;
}

static void ClipOperationExeFuncDoubleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    auto inputRefs = AsRef(inputs);
    FUNCTION("main", inputRefs, {outputs[ID0]})
    {
        SymbolicScalar firstDim = inputs[ID0].GetShape()[ID0];
        SymbolicScalar secondDim = inputs[ID0].GetShape()[ID1];

        auto args = static_cast<const ClipOpFuncArgs*>(opArgs);
        Shape viewShapes(args->viewShape_.size(), 0);
        viewShapes[ID0] = std::min(args->viewShape_[ID0], firstDim);
        viewShapes[ID1] = std::min(args->viewShape_[ID1], secondDim);
        int bloop = CeilDiv(firstDim, viewShapes[ID0]);
        int sloop = CeilDiv(secondDim, viewShapes[ID1]);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                std::vector<SymbolicScalar> loopVars = {bIdx, sIdx};
                std::vector<SymbolicScalar> offsets = GetOffsets(viewShapes, loopVars);
                auto tileTensor0 = View(inputs[ID0], viewShapes,
                                        GetValidShape(inputs[ID0].GetShape(), viewShapes, loopVars), offsets);
                TileShape::Current().SetVecTile(args->tileShape_);
                Tensor res;
                if (args->isElement_) {
                    res = ProcessElementModeClip(tileTensor0, args);
                } else {
                    res = ProcessTensorModeClip(tileTensor0, inputs, args, loopVars);
                }
                Assemble(res, offsets, outputs[ID0]);
            }
        }
    }
}

static void ClipOperationExeFuncTripleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs)
{
    auto inputRefs = AsRef(inputs);
    FUNCTION("main", inputRefs, {outputs[ID0]})
    {
        SymbolicScalar firstDim = inputs[ID0].GetShape()[ID0];
        SymbolicScalar secondDim = inputs[ID0].GetShape()[ID1];
        SymbolicScalar thirdDim = inputs[ID0].GetShape()[ID2];

        auto args = static_cast<const ClipOpFuncArgs*>(opArgs);
        Shape viewShapes(args->viewShape_.size(), 0);
        viewShapes[ID0] = std::min(args->viewShape_[ID0], firstDim);
        viewShapes[ID1] = std::min(args->viewShape_[ID1], secondDim);
        viewShapes[ID2] = std::min(args->viewShape_[ID2], thirdDim);
        int bloop = CeilDiv(firstDim, viewShapes[ID0]);
        int sloop = CeilDiv(secondDim, viewShapes[ID1]);
        int nloop = CeilDiv(thirdDim, viewShapes[ID2]);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    std::vector<SymbolicScalar> loopVars = {bIdx, sIdx, nIdx};
                    std::vector<SymbolicScalar> offsets = GetOffsets(viewShapes, loopVars);
                    Tensor tileTensor0 = View(inputs[ID0], viewShapes,
                                              GetValidShape(inputs[ID0].GetShape(), viewShapes, loopVars), offsets);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    Tensor res;
                    if (args->isElement_) {
                        res = ProcessElementModeClip(tileTensor0, args);
                    } else {
                        res = ProcessTensorModeClip(tileTensor0, inputs, args, loopVars);
                    }
                    Assemble(res, offsets, outputs[ID0]);
                }
            }
        }
    }
}

static void ClipOperationExeFuncQuadrupleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                             const OpFuncArgs* opArgs)
{
    auto inputRefs = AsRef(inputs);
    FUNCTION("main", inputRefs, {outputs[ID0]})
    {
        SymbolicScalar firstDim = inputs[ID0].GetShape()[ID0];
        SymbolicScalar secondDim = inputs[ID0].GetShape()[ID1];
        SymbolicScalar thirdDim = inputs[ID0].GetShape()[ID2];
        SymbolicScalar fourthDim = inputs[ID0].GetShape()[ID3];

        auto args = static_cast<const ClipOpFuncArgs*>(opArgs);
        Shape viewShapes(args->viewShape_.size(), 0);
        viewShapes[ID0] = std::min(args->viewShape_[ID0], firstDim);
        viewShapes[ID1] = std::min(args->viewShape_[ID1], secondDim);
        viewShapes[ID2] = std::min(args->viewShape_[ID2], thirdDim);
        viewShapes[ID3] = std::min(args->viewShape_[ID3], fourthDim);
        int bloop = CeilDiv(firstDim, viewShapes[ID0]);
        int sloop = CeilDiv(secondDim, viewShapes[ID1]);
        int nloop = CeilDiv(thirdDim, viewShapes[ID2]);
        int qloop = CeilDiv(fourthDim, viewShapes[ID3]);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(0, qloop, 1))
                    {
                        std::vector<SymbolicScalar> loopVars = {bIdx, sIdx, nIdx, qIdx};
                        std::vector<SymbolicScalar> offsets = GetOffsets(viewShapes, loopVars);
                        Tensor tileTensor0 = View(inputs[ID0], viewShapes,
                                                  GetValidShape(inputs[ID0].GetShape(), viewShapes, loopVars), offsets);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        Tensor res;
                        if (args->isElement_) {
                            res = ProcessElementModeClip(tileTensor0, args);
                        } else {
                            res = ProcessTensorModeClip(tileTensor0, inputs, args, loopVars);
                        }
                        Assemble(res, offsets, outputs[ID0]);
                    }
                }
            }
        }
    }
}

class ClipOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ClipOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestClip, ClipOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<ClipOpMetaData>({ClipOperationExeFuncDoubleCut,
                                                                            ClipOperationExeFuncTripleCut,
                                                                            ClipOperationExeFuncQuadrupleCut},
                                                                           "Clip")));

TEST_P(ClipOperationTest, TestClip)
{
    auto test_data = GetParam().test_data_;
    int testType = GetValueByName<int>(test_data, "test_type");
    if (testType == -1) {
        // same as the logic for func_id
        testType = GetViewShape(test_data).size() - 2;
    }
    std::string minDtypeStr = GetValueByName<std::string>(test_data, "min_dtype");
    std::string maxDtypeStr = GetValueByName<std::string>(test_data, "max_dtype");

    bool isElement = !minDtypeStr.empty() || !maxDtypeStr.empty();
    ClipOpFuncArgs args({}, {}, ToUnderlying(TestType::NoValue));
    Element min = {}, max = {};

    if (testType == ToUnderlying(TestType::ElementDefaultMinDefaultMax) ||
        testType == ToUnderlying(TestType::ElementDefaultMinNotDefaultMax) ||
        testType == ToUnderlying(TestType::ElementNotDefaultMinDefaultMax) ||
        (testType == ToUnderlying(TestType::NotDefault2D) && isElement) ||
        (testType == ToUnderlying(TestType::NotDefault3D) && isElement) ||
        (testType == ToUnderlying(TestType::NotDefault4D) && isElement)) {
        if (!minDtypeStr.empty()) {
            DataType minDtype = GetDataType(minDtypeStr);
            min = GetElementByType(minDtype, test_data, "min_value");
        }
        if (!maxDtypeStr.empty()) {
            DataType maxDtype = GetDataType(maxDtypeStr);
            max = GetElementByType(maxDtype, test_data, "max_value");
        }
    }
    args = ClipOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), testType, min, max, isElement);
    auto testCase = CreateTestCaseDesc<ClipOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
