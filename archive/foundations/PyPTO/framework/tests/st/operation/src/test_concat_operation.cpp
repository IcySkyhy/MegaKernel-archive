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
 * \file test_add_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct ConcatOpFuncArgs : public OpFuncArgs {
    ConcatOpFuncArgs(int axis, const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : axis_(axis), viewShape_(viewShape), tileShape_(tileShape)
    {}

    int axis_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct ConcatOpMetaData {
    explicit ConcatOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static std::vector<int64_t> GetConcatViewShape(Tensor tensor, const std::vector<int64_t>& viewShape, int axis)
{
    std::vector<int64_t> resultViewShape = {};
    for (size_t i = 0; i < viewShape.size(); i++) {
        if (static_cast<int>(i) != axis) {
            resultViewShape.push_back(std::min(viewShape[i], tensor.GetShape()[i]));
        } else {
            resultViewShape.push_back(tensor.GetShape()[i]);
        }
    }
    return resultViewShape;
}

static std::vector<SymbolicScalar> GetNoAxisDims(const std::vector<Tensor>& inputs, int axis)
{
    Tensor input0 = inputs[0];
    std::vector<SymbolicScalar> dimensions = {};
    for (size_t i = 0; i < input0.GetShape().size(); i++) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        dimensions.push_back(input0.GetShape()[i]);
    }
    return dimensions;
}

static std::vector<int64_t> GetNoAxisViewShapes(const std::vector<int64_t>& viewShapes, int axis)
{
    std::vector<int64_t> shapes = {};
    for (size_t i = 0; i < viewShapes.size(); i++) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        shapes.push_back(viewShapes[i]);
    }
    return shapes;
}

static std::vector<std::reference_wrapper<const Tensor>> AsRef(const std::vector<Tensor>& tensors)
{
    std::vector<std::reference_wrapper<const Tensor>> results;
    results.reserve(tensors.size());
    for (const Tensor& tensor : tensors) {
        results.emplace_back(std::cref(tensor));
    }
    return results;
}

static void ConcatOperationExeFuncDoubleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    auto inputRefs = AsRef(inputs);
    auto args = static_cast<const ConcatOpFuncArgs*>(opArgs);
    int axis = 0;
    if (args->axis_ < 0) {
        axis = args->axis_ + inputs[0].GetShape().size();
    } else {
        axis = args->axis_;
    }

    FUNCTION("main", inputRefs, {outputs[0]})
    {
        std::vector<SymbolicScalar> noConcatAxisDimensions = GetNoAxisDims(inputs, axis);
        std::vector<int64_t> noConcatAxisViewShapes = GetNoAxisViewShapes(args->viewShape_, axis);
        const int bloop = CeilDiv(noConcatAxisDimensions[0], noConcatAxisViewShapes[0]);
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            std::vector<SymbolicScalar> indices = {bIdx};
            indices.insert(indices.begin() + axis, 0);
            std::vector<Tensor> concatTensors = {};
            for (size_t tensorId = 0; tensorId < inputs.size(); tensorId++) {
                const std::vector<int64_t> tensorViewShape = GetConcatViewShape(inputs[tensorId], args->viewShape_,
                                                                                axis);
                // 本质是在找 bIdx 循环的是哪一根轴
                SymbolicScalar validShape0 = std::min(
                    SymbolicScalar(inputs[tensorId].GetShape()[0]) - indices[0] * tensorViewShape[0],
                    tensorViewShape[0]);
                SymbolicScalar validShape1 = std::min(
                    SymbolicScalar(inputs[tensorId].GetShape()[1]) - indices[1] * tensorViewShape[1],
                    tensorViewShape[1]);
                auto tileTensor = View(inputs[tensorId], tensorViewShape, {validShape0, validShape1},
                                       {indices[0] * tensorViewShape[0], indices[1] * tensorViewShape[1]});
                concatTensors.push_back(tileTensor);
            }
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Cat(concatTensors, axis);
            Assemble(res, {indices[0] * args->viewShape_[0], indices[1] * args->viewShape_[1]}, outputs[0]);
        }
    }
}

static void ConcatOperationExeFuncTripleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                            const OpFuncArgs* opArgs)
{
    auto inputRefs = AsRef(inputs);
    auto args = static_cast<const ConcatOpFuncArgs*>(opArgs);
    int axis = 0;
    if (args->axis_ < 0) {
        axis = args->axis_ + inputs[0].GetShape().size();
    } else {
        axis = args->axis_;
    }
    FUNCTION("main", inputRefs, {outputs[0]})
    {
        std::vector<SymbolicScalar> noConcatAxisDimensions = GetNoAxisDims(inputs, axis);
        std::vector<int64_t> noConcatAxisViewShapes = GetNoAxisViewShapes(args->viewShape_, axis);
        const int bloop = CeilDiv(noConcatAxisDimensions[0], noConcatAxisViewShapes[0]);
        const int sloop = CeilDiv(noConcatAxisDimensions[1], noConcatAxisViewShapes[1]);
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                std::vector<SymbolicScalar> indices = {bIdx, sIdx};
                indices.insert(indices.begin() + axis, 0);
                std::vector<Tensor> concatTensors = {};
                for (size_t tensorId = 0; tensorId < inputs.size(); tensorId++) {
                    const std::vector<int64_t> tensorViewShape = GetConcatViewShape(inputs[tensorId], args->viewShape_,
                                                                                    axis);
                    SymbolicScalar validShape0 = std::min(
                        SymbolicScalar(inputs[tensorId].GetShape()[0]) - indices[0] * tensorViewShape[0],
                        tensorViewShape[0]);
                    SymbolicScalar validShape1 = std::min(
                        SymbolicScalar(inputs[tensorId].GetShape()[1]) - indices[1] * tensorViewShape[1],
                        tensorViewShape[1]);
                    SymbolicScalar validShape2 = std::min(
                        SymbolicScalar(inputs[tensorId].GetShape()[2]) - indices[2] * tensorViewShape[2],
                        tensorViewShape[2]);
                    auto tileTensor = View(inputs[tensorId], tensorViewShape, {validShape0, validShape1, validShape2},
                                           {
                                               indices[0] * tensorViewShape[0],
                                               indices[1] * tensorViewShape[1],
                                               indices[2] * tensorViewShape[2],
                                           });
                    concatTensors.push_back(tileTensor);
                }
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Cat(concatTensors, axis);
                Assemble(res,
                         {
                             indices[0] * args->viewShape_[0],
                             indices[1] * args->viewShape_[1],
                             indices[2] * args->viewShape_[2],
                         },
                         outputs[0]);
            }
        }
    }
}

static void ConcatOperationExeFuncQuadraticCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    auto inputRefs = AsRef(inputs);
    int axis = 0;
    auto args = static_cast<const ConcatOpFuncArgs*>(opArgs);
    if (args->axis_ < 0) {
        axis = args->axis_ + inputs[0].GetShape().size();
    } else {
        axis = args->axis_;
    }
    FUNCTION("main", inputRefs, {outputs[0]})
    {
        std::vector<SymbolicScalar> noConcatAxisDimensions = GetNoAxisDims(inputs, axis);
        std::vector<int64_t> noConcatAxisViewShapes = GetNoAxisViewShapes(args->viewShape_, axis);
        const int bloop = CeilDiv(noConcatAxisDimensions[0], noConcatAxisViewShapes[0]);
        const int sloop = CeilDiv(noConcatAxisDimensions[1], noConcatAxisViewShapes[1]);
        const int kloop = CeilDiv(noConcatAxisDimensions[2], noConcatAxisViewShapes[2]);
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                LOOP("LOOP_L1_kIdx", FunctionType::DYNAMIC_LOOP, kIdx, LoopRange(0, kloop, 1))
                {
                    std::vector<SymbolicScalar> indices = {bIdx, sIdx, kIdx};
                    indices.insert(indices.begin() + axis, 0);
                    std::vector<Tensor> concatTensors = {};
                    for (size_t tensorId = 0; tensorId < inputs.size(); tensorId++) {
                        const std::vector<int64_t> tensorViewShape = GetConcatViewShape(inputs[tensorId],
                                                                                        args->viewShape_, axis);
                        SymbolicScalar validShape0 = std::min(
                            SymbolicScalar(inputs[tensorId].GetShape()[0]) - indices[0] * tensorViewShape[0],
                            tensorViewShape[0]);
                        SymbolicScalar validShape1 = std::min(
                            SymbolicScalar(inputs[tensorId].GetShape()[1]) - indices[1] * tensorViewShape[1],
                            tensorViewShape[1]);
                        SymbolicScalar validShape2 = std::min(
                            SymbolicScalar(inputs[tensorId].GetShape()[2]) - indices[2] * tensorViewShape[2],
                            tensorViewShape[2]);
                        SymbolicScalar validShape3 = std::min(
                            SymbolicScalar(inputs[tensorId].GetShape()[3]) - indices[3] * tensorViewShape[3],
                            tensorViewShape[3]);
                        auto tileTensor = View(inputs[tensorId], tensorViewShape,
                                               {validShape0, validShape1, validShape2, validShape3},
                                               {
                                                   indices[0] * tensorViewShape[0],
                                                   indices[1] * tensorViewShape[1],
                                                   indices[2] * tensorViewShape[2],
                                                   indices[3] * tensorViewShape[3],
                                               });
                        concatTensors.push_back(tileTensor);
                    }
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Cat(concatTensors, axis);
                    Assemble(res,
                             {
                                 indices[0] * args->viewShape_[0],
                                 indices[1] * args->viewShape_[1],
                                 indices[2] * args->viewShape_[2],
                                 indices[3] * args->viewShape_[3],
                             },
                             outputs[0]);
                }
            }
        }
    }
}

class ConcatOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ConcatOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestConcat, ConcatOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<ConcatOpMetaData>({ConcatOperationExeFuncDoubleCut,
                                                                              ConcatOperationExeFuncTripleCut,
                                                                              ConcatOperationExeFuncQuadraticCut},
                                                                             "Concat")));

TEST_P(ConcatOperationTest, TestConcat)
{
    auto test_data = GetParam().test_data_;
    int axis = GetValueByName<int>(test_data, "axis");
    auto args = ConcatOpFuncArgs(axis, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<ConcatOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
