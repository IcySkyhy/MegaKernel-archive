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
 * \file test_dequantize_operation.cpp
 * \brief Test cases for Dequantize operation
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;

namespace {

struct DequantizeOpFuncArgs : public OpFuncArgs {
    DequantizeOpFuncArgs(std::vector<int64_t> shape, std::vector<int64_t> vecTileShapes, DataType inputDtype, int axis,
                         bool useZeroPoints = false)
        : viewShape_(shape),
          tileShape_(vecTileShapes),
          inputDtype_(inputDtype),
          axis_(axis),
          useZeroPoints_(useZeroPoints)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    DataType inputDtype_; // INT8 or INT16
    int axis_;
    bool useZeroPoints_;
};

struct DequantizeOpMetaData {
    explicit DequantizeOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

// ============================================================
// 1D Tensor Tests
// ============================================================

// 1D dequantization with axis=-1, symmetric/asymmetric
static void Dequantize1DOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    auto args = static_cast<const DequantizeOpFuncArgs*>(opArgs);
    const bool useZeroPoints = args->useZeroPoints_;

    if (useZeroPoints) {
        FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
        {
            const int viewShape = args->viewShape_[0];

            SymbolicScalar dim = inputs[0].GetShape()[0];
            const int loop = CeilDiv(dim, viewShape);

            int axis = args->axis_;

            LOOP("LOOP_L0_idx", FunctionType::DYNAMIC_LOOP, idx, LoopRange(0, loop, 1))
            {
                auto tileTensorInput = View(inputs[0], {viewShape}, {std::min(dim - idx * viewShape, viewShape)},
                                            {idx * viewShape});

                // Scale shape: [1] for axis=-1 on 1D tensor
                auto tileTensorScale = inputs[1];
                auto tileTensorZeroPoints = inputs[2];

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, tileTensorZeroPoints);
                Assemble(res, {idx * viewShape}, outputs[0]);
            }
        }
    } else {
        FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
        {
            const int viewShape = args->viewShape_[0];

            SymbolicScalar dim = inputs[0].GetShape()[0];
            const int loop = CeilDiv(dim, viewShape);

            int axis = args->axis_;

            LOOP("LOOP_L0_idx", FunctionType::DYNAMIC_LOOP, idx, LoopRange(0, loop, 1))
            {
                auto tileTensorInput = View(inputs[0], {viewShape}, {std::min(dim - idx * viewShape, viewShape)},
                                            {idx * viewShape});

                // Scale shape: [1] for axis=-1 on 1D tensor
                auto tileTensorScale = inputs[1];

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                Assemble(res, {idx * viewShape}, outputs[0]);
            }
        }
    }
}

// ============================================================
// 2D Tensor Tests
// ============================================================

// 2D dequantization with axis=-1 / -2, symmetric/asymmetric
static void Dequantize2DOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    auto args = static_cast<const DequantizeOpFuncArgs*>(opArgs);
    const bool useZeroPoints = args->useZeroPoints_;

    if (useZeroPoints) {
        FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
        {
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];

            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            SymbolicScalar secondDim = inputs[0].GetShape()[1];
            const int bloop = CeilDiv(firstDim, firstViewShape);
            const int sloop = CeilDiv(secondDim, secondViewShape);

            int axis = args->axis_;

            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    auto tileTensorInput = View(inputs[0], {firstViewShape, secondViewShape},
                                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                                {bIdx * firstViewShape, sIdx * secondViewShape});

                    // shape: [..., row] for axis=-1 / [..., col] for axis=-2
                    SymbolicScalar paraViewShape = (axis == -1) ? firstViewShape : secondViewShape;
                    SymbolicScalar paraViewShapeBefore = (axis == -1) ? bIdx * firstViewShape : sIdx * secondViewShape;
                    SymbolicScalar paraViewShapeTail = (axis == -1) ? firstDim - paraViewShapeBefore :
                                                                      secondDim - paraViewShapeBefore;

                    auto tileTensorScale = View(inputs[1], {paraViewShape},
                                                {std::min(paraViewShapeTail, paraViewShape)}, {paraViewShapeBefore});
                    auto tileTensorZeroPoints = View(inputs[2], {paraViewShape},
                                                     {std::min(paraViewShapeTail, paraViewShape)},
                                                     {paraViewShapeBefore});

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis,
                                          tileTensorZeroPoints);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                }
            }
        }
    } else {
        FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
        {
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];

            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            SymbolicScalar secondDim = inputs[0].GetShape()[1];
            const int bloop = CeilDiv(firstDim, firstViewShape);
            const int sloop = CeilDiv(secondDim, secondViewShape);

            int axis = args->axis_;

            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    auto tileTensorInput = View(inputs[0], {firstViewShape, secondViewShape},
                                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                                {bIdx * firstViewShape, sIdx * secondViewShape});

                    // shape: [..., row] for axis=-1 / [..., col] for axis=-2
                    SymbolicScalar paraViewShape = (axis == -1) ? firstViewShape : secondViewShape;
                    SymbolicScalar paraViewShapeBefore = (axis == -1) ? bIdx * firstViewShape : sIdx * secondViewShape;
                    SymbolicScalar paraViewShapeTail = (axis == -1) ? firstDim - paraViewShapeBefore :
                                                                      secondDim - paraViewShapeBefore;

                    auto tileTensorScale = View(inputs[1], {paraViewShape},
                                                {std::min(paraViewShapeTail, paraViewShape)}, {paraViewShapeBefore});

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
                }
            }
        }
    }
}

// ============================================================
// 3D Tensor Tests
// ============================================================

// 3D dequantization with axis=-1 / -2, symmetric/asymmetric
static void Dequantize3DOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    auto args = static_cast<const DequantizeOpFuncArgs*>(opArgs);
    const bool useZeroPoints = args->useZeroPoints_;

    if (useZeroPoints) {
        FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
        {
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];
            const int thirdViewShape = args->viewShape_[2];

            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            SymbolicScalar secondDim = inputs[0].GetShape()[1];
            SymbolicScalar thirdDim = inputs[0].GetShape()[2];
            const int bloop = CeilDiv(firstDim, firstViewShape);
            const int sloop = CeilDiv(secondDim, secondViewShape);
            const int nloop = CeilDiv(thirdDim, thirdViewShape);

            int axis = args->axis_;

            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        auto tileTensorInput = View(
                            inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                             std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                             std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});

                        SymbolicScalar paraViewShape = (axis == -1) ? secondViewShape : thirdViewShape;
                        SymbolicScalar paraViewShapeBefore = (axis == -1) ? sIdx * secondViewShape :
                                                                            nIdx * thirdViewShape;
                        SymbolicScalar paraViewShapeTail = (axis == -1) ? secondDim - paraViewShapeBefore :
                                                                          thirdDim - paraViewShapeBefore;

                        auto tileTensorScale = View(inputs[1], {firstViewShape, paraViewShape},
                                                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                     std::min(paraViewShapeTail, paraViewShape)},
                                                    {bIdx * firstViewShape, paraViewShapeBefore});
                        auto tileTensorZeroPoints = View(inputs[2], {firstViewShape, paraViewShape},
                                                         {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                          std::min(paraViewShapeTail, paraViewShape)},
                                                         {bIdx * firstViewShape, paraViewShapeBefore});

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis,
                                              tileTensorZeroPoints);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape},
                                 outputs[0]);
                    }
                }
            }
        }
    } else {
        FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
        {
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];
            const int thirdViewShape = args->viewShape_[2];

            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            SymbolicScalar secondDim = inputs[0].GetShape()[1];
            SymbolicScalar thirdDim = inputs[0].GetShape()[2];
            const int bloop = CeilDiv(firstDim, firstViewShape);
            const int sloop = CeilDiv(secondDim, secondViewShape);
            const int nloop = CeilDiv(thirdDim, thirdViewShape);

            int axis = args->axis_;

            LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
            {
                LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
                {
                    LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1))
                    {
                        auto tileTensorInput = View(
                            inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                             std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                             std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});

                        SymbolicScalar paraViewShape = (axis == -1) ? secondViewShape : thirdViewShape;
                        SymbolicScalar paraViewShapeBefore = (axis == -1) ? sIdx * secondViewShape :
                                                                            nIdx * thirdViewShape;
                        SymbolicScalar paraViewShapeTail = (axis == -1) ? secondDim - paraViewShapeBefore :
                                                                          thirdDim - paraViewShapeBefore;

                        auto tileTensorScale = View(inputs[1], {firstViewShape, paraViewShape},
                                                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                     std::min(paraViewShapeTail, paraViewShape)},
                                                    {bIdx * firstViewShape, paraViewShapeBefore});

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape},
                                 outputs[0]);
                    }
                }
            }
        }
    }
}

// ============================================================
// 4D Tensor Tests
// ============================================================

// 4D dequantization with axis=-1 / -2, symmetric/asymmetric
static void Dequantize4DOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    auto args = static_cast<const DequantizeOpFuncArgs*>(opArgs);
    const bool useZeroPoints = args->useZeroPoints_;

    if (useZeroPoints) {
        FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
        {
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];
            const int thirdViewShape = args->viewShape_[2];
            const int fourthViewShape = args->viewShape_[3];

            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            SymbolicScalar secondDim = inputs[0].GetShape()[1];
            SymbolicScalar thirdDim = inputs[0].GetShape()[2];
            SymbolicScalar fourthDim = inputs[0].GetShape()[3];
            const int loop0 = CeilDiv(firstDim, firstViewShape);
            const int loop1 = CeilDiv(secondDim, secondViewShape);
            const int loop2 = CeilDiv(thirdDim, thirdViewShape);
            const int loop3 = CeilDiv(fourthDim, fourthViewShape);

            int axis = args->axis_;

            LOOP("LOOP_L0", FunctionType::DYNAMIC_LOOP, idx0, LoopRange(0, loop0, 1))
            {
                LOOP("LOOP_L1", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(0, loop1, 1))
                {
                    LOOP("LOOP_L2", FunctionType::DYNAMIC_LOOP, idx2, LoopRange(0, loop2, 1))
                    {
                        LOOP("LOOP_L3", FunctionType::DYNAMIC_LOOP, idx3, LoopRange(0, loop3, 1))
                        {
                            auto tileTensorInput = View(
                                inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                 std::min(secondDim - idx1 * secondViewShape, secondViewShape),
                                 std::min(thirdDim - idx2 * thirdViewShape, thirdViewShape),
                                 std::min(fourthDim - idx3 * fourthViewShape, fourthViewShape)},
                                {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape,
                                 idx3 * fourthViewShape});

                            SymbolicScalar paraViewShape = (axis == -1) ? thirdViewShape : fourthViewShape;
                            SymbolicScalar paraViewShapeBefore = (axis == -1) ? idx2 * thirdViewShape :
                                                                                idx3 * fourthViewShape;
                            SymbolicScalar paraViewShapeTail = (axis == -1) ? thirdDim - paraViewShapeBefore :
                                                                              fourthDim - paraViewShapeBefore;

                            auto tileTensorScale = View(
                                inputs[1], {firstViewShape, secondViewShape, paraViewShape},
                                {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                 std::min(secondDim - idx1 * secondViewShape, secondViewShape),
                                 std::min(paraViewShapeTail, paraViewShape)},
                                {idx0 * firstViewShape, idx1 * secondViewShape, paraViewShapeBefore});
                            auto tileTensorZeroPoints = View(
                                inputs[2], {firstViewShape, secondViewShape, paraViewShape},
                                {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                 std::min(secondDim - idx1 * secondViewShape, secondViewShape),
                                 std::min(paraViewShapeTail, paraViewShape)},
                                {idx0 * firstViewShape, idx1 * secondViewShape, paraViewShapeBefore});

                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis,
                                                  tileTensorZeroPoints);
                            Assemble(res,
                                     {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape,
                                      idx3 * fourthViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        }
    } else {
        FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]})
        {
            const int firstViewShape = args->viewShape_[0];
            const int secondViewShape = args->viewShape_[1];
            const int thirdViewShape = args->viewShape_[2];
            const int fourthViewShape = args->viewShape_[3];

            SymbolicScalar firstDim = inputs[0].GetShape()[0];
            SymbolicScalar secondDim = inputs[0].GetShape()[1];
            SymbolicScalar thirdDim = inputs[0].GetShape()[2];
            SymbolicScalar fourthDim = inputs[0].GetShape()[3];
            const int loop0 = CeilDiv(firstDim, firstViewShape);
            const int loop1 = CeilDiv(secondDim, secondViewShape);
            const int loop2 = CeilDiv(thirdDim, thirdViewShape);
            const int loop3 = CeilDiv(fourthDim, fourthViewShape);

            int axis = args->axis_;

            LOOP("LOOP_L0", FunctionType::DYNAMIC_LOOP, idx0, LoopRange(0, loop0, 1))
            {
                LOOP("LOOP_L1", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(0, loop1, 1))
                {
                    LOOP("LOOP_L2", FunctionType::DYNAMIC_LOOP, idx2, LoopRange(0, loop2, 1))
                    {
                        LOOP("LOOP_L3", FunctionType::DYNAMIC_LOOP, idx3, LoopRange(0, loop3, 1))
                        {
                            auto tileTensorInput = View(
                                inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                 std::min(secondDim - idx1 * secondViewShape, secondViewShape),
                                 std::min(thirdDim - idx2 * thirdViewShape, thirdViewShape),
                                 std::min(fourthDim - idx3 * fourthViewShape, fourthViewShape)},
                                {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape,
                                 idx3 * fourthViewShape});

                            SymbolicScalar paraViewShape = (axis == -1) ? thirdViewShape : fourthViewShape;
                            SymbolicScalar paraViewShapeBefore = (axis == -1) ? idx2 * thirdViewShape :
                                                                                idx3 * fourthViewShape;
                            SymbolicScalar paraViewShapeTail = (axis == -1) ? thirdDim - paraViewShapeBefore :
                                                                              fourthDim - paraViewShapeBefore;

                            auto tileTensorScale = View(
                                inputs[1], {firstViewShape, secondViewShape, paraViewShape},
                                {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                 std::min(secondDim - idx1 * secondViewShape, secondViewShape),
                                 std::min(paraViewShapeTail, paraViewShape)},
                                {idx0 * firstViewShape, idx1 * secondViewShape, paraViewShapeBefore});

                            TileShape::Current().SetVecTile(args->tileShape_);
                            auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                            Assemble(res,
                                     {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape,
                                      idx3 * fourthViewShape},
                                     outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

// Helper function to select the appropriate execution function based on test parameters
OpFunc SelectDequantizeOpFunc(int ndim)
{
    // inputDtype (INT8/INT16) doesn't affect loop structure, only ndim matters
    if (ndim == 1) {
        return Dequantize1DOperationExeFunc;
    } else if (ndim == 2) {
        return Dequantize2DOperationExeFunc;
    } else if (ndim == 3) {
        return Dequantize3DOperationExeFunc;
    } else if (ndim == 4) {
        return Dequantize4DOperationExeFunc;
    }

    // Default to 2D
    return Dequantize2DOperationExeFunc;
}

class DequantizeOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<DequantizeOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestDequantize, DequantizeOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<DequantizeOpMetaData>(
                             {Dequantize1DOperationExeFunc, Dequantize2DOperationExeFunc, Dequantize3DOperationExeFunc,
                              Dequantize4DOperationExeFunc},
                             "Dequantize")));

TEST_P(DequantizeOperationTest, TestDequantize)
{
    auto test_data = GetParam().test_data_;
    auto inputDtype = static_cast<DataType>(GetValueByName<int>(test_data, "param_input_dtype"));
    auto axis = GetValueByName<int>(test_data, "axis");
    auto useZeroPoints = GetValueByName<bool>(test_data, "use_zero_points");
    auto viewShape = GetViewShape(test_data);
    int ndim = static_cast<int>(viewShape.size());

    // Dynamically select the appropriate execution function
    auto selectedOpFunc = SelectDequantizeOpFunc(ndim);

    auto args = DequantizeOpFuncArgs(viewShape, GetTileShape(test_data), inputDtype, axis, useZeroPoints);

    TestCaseDesc testCase;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    testCase.args = &args;
    testCase.opFunc = selectedOpFunc;
    std::transform(testCase.inputTensors.begin(), testCase.inputTensors.end(), std::back_inserter(testCase.inputPaths),
                   [](const auto& tensor) { return GetGoldenDir() + "/" + tensor.GetStorage()->Symbol() + ".bin"; });
    std::transform(testCase.outputTensors.begin(), testCase.outputTensors.end(),
                   std::back_inserter(testCase.goldenPaths),
                   [](const auto& tensor) { return GetGoldenDir() + "/" + tensor.GetStorage()->Symbol() + ".bin"; });
    auto params_dict = test_data.at("params");
    testCase.onBoard = params_dict.find("on_board") == params_dict.end() || GetValueByName<bool>(test_data, "on_board");

    TestExecutor::runTest(testCase);
}

} // namespace
