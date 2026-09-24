/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_indexput_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;

namespace {
struct IndexPut_OpFuncArgs : public OpFuncArgs {
    IndexPut_OpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, bool accumulate)
        : viewShape_(viewShape), tileShape_(tileShape), accumulate_(accumulate)
    {
        this->inplaceInfo[0] = 0;
    }

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    bool accumulate_;
};

struct IndexPut_OpMetaData {
    explicit IndexPut_OpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

/* inputs:{self, values, indices[0]} */
template <typename T>
static void IndexPut_OperationExeFunc1Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        const T* args = static_cast<const T*>(opArgs);
        SymbolicScalar indicesFirstDim = inputs[2].GetShape()[0];
        std::vector<int64_t> valuesShapes = inputs[1].GetShape();
        const int viewShape = args->viewShape_[0]; // 只切首轴
        std::vector<int64_t> valuesViewShapes = valuesShapes;
        valuesViewShapes[0] = viewShape;
        std::vector<SymbolicScalar> valuesValidShapes;
        for (int64_t vs : valuesShapes) {
            valuesValidShapes.emplace_back(vs);
        }
        std::vector<SymbolicScalar> valuesNewOffsets(valuesShapes.size(), 0);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(indicesFirstDim, viewShape), 1))
        {
            valuesValidShapes[0] = std::min(valuesShapes[0] - bIdx * viewShape, viewShape);
            valuesNewOffsets[0] = bIdx * viewShape;
            auto viewValues = View(inputs[1], valuesViewShapes, valuesValidShapes, valuesNewOffsets);
            auto viewIndices1 = View(inputs[2], {viewShape}, {std::min(indicesFirstDim - bIdx * viewShape, viewShape)},
                                     {bIdx * viewShape});
            std::vector<Tensor> viewIndices = {viewIndices1};
            TileShape::Current().SetVecTile(args->tileShape_);
            IndexPut_(outputs[0], viewIndices, viewValues, args->accumulate_);
        }
    }
}

/* inputs:{self, values, indices[0], indices[1]} */
template <typename T>
static void IndexPut_OperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2], inputs[3]}, {outputs[0]})
    {
        const T* args = static_cast<const T*>(opArgs);
        std::vector<int64_t> valuesShapes = inputs[1].GetShape();
        const int viewShape = args->viewShape_[0];
        std::vector<int64_t> valuesViewShapes = valuesShapes;
        valuesViewShapes[0] = viewShape;
        std::vector<SymbolicScalar> valuesValidShapes;
        for (int64_t vs : valuesShapes) {
            valuesValidShapes.emplace_back(vs);
        }
        SymbolicScalar indicesSecondDim = inputs[3].GetShape()[0];
        SymbolicScalar indicesFirstDim = inputs[2].GetShape()[0];
        SymbolicScalar maxIndices = std::max({indicesFirstDim, indicesSecondDim});
        std::vector<SymbolicScalar> valuesNewOffsets(valuesShapes.size(), 0);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(maxIndices, viewShape), 1))
        {
            valuesValidShapes[0] = std::min(valuesShapes[0] - bIdx * viewShape, viewShape);
            valuesNewOffsets[0] = bIdx * viewShape;
            auto viewValues = View(inputs[1], valuesViewShapes, valuesValidShapes, valuesNewOffsets);
            auto viewIndices1 = View(inputs[2], {viewShape}, {std::min(indicesFirstDim - bIdx * viewShape, viewShape)},
                                     {bIdx * viewShape});
            auto viewIndices2 = View(inputs[3], {viewShape}, {std::min(indicesSecondDim - bIdx * viewShape, viewShape)},
                                     {bIdx * viewShape});
            std::vector<Tensor> viewIndices = {viewIndices1, viewIndices2};
            TileShape::Current().SetVecTile(args->tileShape_);
            IndexPut_(outputs[0], viewIndices, viewValues, args->accumulate_);
        }
    }
}

/* inputs: self, values, indices[0], indices[1], indices[2] */
template <typename T>
static void IndexPut_OperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]}, {outputs[0]})
    {
        const T* args = static_cast<const T*>(opArgs);
        std::vector<int64_t> valuesShapes = inputs[1].GetShape();
        const int viewShape = args->viewShape_[0];
        std::vector<int64_t> valuesViewShapes = valuesShapes;
        valuesViewShapes[0] = viewShape;
        std::vector<SymbolicScalar> valuesValidShapes;
        for (int64_t vs : valuesShapes) {
            valuesValidShapes.emplace_back(vs);
        }
        SymbolicScalar indicesThirdDim = inputs[4].GetShape()[0];
        SymbolicScalar indicesSecondDim = inputs[3].GetShape()[0];
        SymbolicScalar indicesFirstDim = inputs[2].GetShape()[0];
        SymbolicScalar maxIndices = std::max({indicesFirstDim, indicesSecondDim, indicesThirdDim});
        std::vector<SymbolicScalar> valuesNewOffsets(valuesShapes.size(), 0);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(maxIndices, viewShape), 1))
        {
            valuesValidShapes[0] = std::min(valuesShapes[0] - bIdx * viewShape, viewShape);
            valuesNewOffsets[0] = bIdx * viewShape;
            auto viewIndices1 = View(inputs[2], {viewShape}, {std::min(indicesFirstDim - bIdx * viewShape, viewShape)},
                                     {bIdx * viewShape});
            auto viewIndices2 = View(inputs[3], {viewShape}, {std::min(indicesSecondDim - bIdx * viewShape, viewShape)},
                                     {bIdx * viewShape});
            auto viewIndices3 = View(inputs[4], {viewShape}, {std::min(indicesThirdDim - bIdx * viewShape, viewShape)},
                                     {bIdx * viewShape});
            auto viewValues = View(inputs[1], valuesViewShapes, valuesValidShapes, valuesNewOffsets);
            std::vector<Tensor> viewIndices = {viewIndices1, viewIndices2, viewIndices3};
            TileShape::Current().SetVecTile(args->tileShape_);
            IndexPut_(outputs[0], viewIndices, viewValues, args->accumulate_);
        }
    }
}

/* inputs: self, values, indices[0], indices[1], indices[2], indices[3] */
template <typename T>
static void IndexPut_OperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]}, {outputs[0]})
    {
        const T* args = static_cast<const T*>(opArgs);
        std::vector<int64_t> valuesShapes = inputs[1].GetShape();
        const int viewShape = args->viewShape_[0];
        std::vector<int64_t> valuesViewShapes = valuesShapes;
        valuesViewShapes[0] = viewShape;
        std::vector<SymbolicScalar> valuesValidShapes;
        for (int64_t vs : valuesShapes) {
            valuesValidShapes.emplace_back(vs);
        }
        SymbolicScalar indicesForthDim = inputs[5].GetShape()[0];
        SymbolicScalar indicesThirdDim = inputs[4].GetShape()[0];
        SymbolicScalar indicesSecondDim = inputs[3].GetShape()[0];
        SymbolicScalar indicesFirstDim = inputs[2].GetShape()[0];
        SymbolicScalar maxIndices = std::max({indicesFirstDim, indicesSecondDim, indicesThirdDim, indicesForthDim});
        std::vector<SymbolicScalar> valuesNewOffsets(valuesShapes.size(), 0);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(maxIndices, viewShape), 1))
        {
            valuesValidShapes[0] = std::min(valuesShapes[0] - bIdx * viewShape, viewShape);
            valuesNewOffsets[0] = bIdx * viewShape;
            auto viewValues = View(inputs[1], valuesViewShapes, valuesValidShapes, valuesNewOffsets);
            auto viewIndices1 = View(inputs[2], {viewShape}, {std::min(indicesFirstDim - bIdx * viewShape, viewShape)},
                                     {bIdx * viewShape});
            auto viewIndices2 = View(inputs[3], {viewShape}, {std::min(indicesSecondDim - bIdx * viewShape, viewShape)},
                                     {bIdx * viewShape});
            auto viewIndices3 = View(inputs[4], {viewShape}, {std::min(indicesThirdDim - bIdx * viewShape, viewShape)},
                                     {bIdx * viewShape});
            auto viewIndices4 = View(inputs[5], {viewShape}, {std::min(indicesForthDim - bIdx * viewShape, viewShape)},
                                     {bIdx * viewShape});
            std::vector<Tensor> viewIndices = {viewIndices1, viewIndices2, viewIndices3, viewIndices4};
            TileShape::Current().SetVecTile(args->tileShape_);
            IndexPut_(outputs[0], viewIndices, viewValues, args->accumulate_);
        }
    }
}

class IndexPut_OperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<IndexPut_OpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestIndexPut_, IndexPut_OperationTest,
    ::testing::ValuesIn(GetOpMetaData<IndexPut_OpMetaData>(
        {IndexPut_OperationExeFunc1Dims<IndexPut_OpFuncArgs>, IndexPut_OperationExeFunc2Dims<IndexPut_OpFuncArgs>,
         IndexPut_OperationExeFunc3Dims<IndexPut_OpFuncArgs>, IndexPut_OperationExeFunc4Dims<IndexPut_OpFuncArgs>},
        "IndexPut_")));

template <typename T>
void IndexPutTestCase(TestCaseDesc& testCase_, const nlohmann::json& test_data)
{
    testCase_.inputTensors = GetInputTensors(test_data);
    testCase_.outputTensors = GetOutputTensors(test_data);
    bool accumulate = GetValueByName<bool>(test_data, "accumulate");
    T args(GetViewShape(test_data), GetTileShape(test_data), accumulate);
    testCase_.args = &args;
    std::vector<OpFunc> func{IndexPut_OperationExeFunc1Dims<T>, IndexPut_OperationExeFunc2Dims<T>,
                             IndexPut_OperationExeFunc3Dims<T>, IndexPut_OperationExeFunc4Dims<T>};
    size_t sizeIndices = testCase_.inputTensors.size() - NUM2;
    size_t sizeIndicesMax = 4;
    size_t sizeIndicesMin = 1;
    ASSERT(sizeIndices >= sizeIndicesMin && sizeIndices <= sizeIndicesMax) << "unsupport the input indices dim";
    for (size_t i = 0; i < sizeIndices; i++) {
        ASSERT(testCase_.inputTensors[i + NUM2].GetShape().size() == 1) << "indices must be a one-dimensional array";
    }
    testCase_.opFunc = func[sizeIndices - 1];
    std::vector<std::string> paths = {GetGoldenDir() + "/" + testCase_.inputTensors[0].GetStorage()->Symbol() + ".bin",
                                      GetGoldenDir() + "/" + testCase_.inputTensors[1].GetStorage()->Symbol() + ".bin"};
    for (size_t indicesIdx = 0; indicesIdx < sizeIndices; indicesIdx++) {
        paths.push_back(GetGoldenDir() + "/" + testCase_.inputTensors[indicesIdx + NUM2].GetStorage()->Symbol() +
                        ".bin");
    }
    testCase_.inputPaths = paths;
    testCase_.goldenPaths = {GetGoldenDir() + "/" + testCase_.outputTensors[0].GetStorage()->Symbol() + ".bin"};
    TestExecutor::runTest(testCase_);
}

TEST_P(IndexPut_OperationTest, TestIndexPut_)
{
    TestCaseDesc testCase_;
    auto test_data = GetParam().test_data_;
    IndexPutTestCase<IndexPut_OpFuncArgs>(testCase_, test_data);
}

} // namespace
