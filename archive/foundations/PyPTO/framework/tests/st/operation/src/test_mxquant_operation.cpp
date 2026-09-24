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
 * \file test_mxquant_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
constexpr int64_t QUANT_MX_SCALE_GROUP_COLS = 64;

struct QuantMXOpFuncArgs : public OpFuncArgs {
    QuantMXOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& tileShape, DataType quantDtype,
                      DequantScaleRoundingMode mode, int64_t axis, bool performanceMode)
        : viewShape_(viewShape),
          tileShape_(tileShape),
          quantDtype_(quantDtype),
          mode_(mode),
          axis_(axis),
          performanceMode_(performanceMode)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    DataType quantDtype_;
    DequantScaleRoundingMode mode_;
    int64_t axis_;
    bool performanceMode_;
};

struct QuantMXOpMetaData {
    explicit QuantMXOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

std::vector<SymbolicScalar> BuildQuantMXScaleOffset(const std::vector<SymbolicScalar>& offset, int64_t axis)
{
    const auto rank = offset.size();
    int64_t normalizedAxis = axis;
    if (normalizedAxis < 0) {
        normalizedAxis += static_cast<int64_t>(rank);
    }
    auto scaleOffset = offset;
    if (rank >= 2 && normalizedAxis == static_cast<int64_t>(rank) - 2) {
        scaleOffset[rank - 2] = scaleOffset[rank - 2] / QUANT_MX_SCALE_GROUP_COLS;
    } else {
        scaleOffset.back() = scaleOffset.back() / QUANT_MX_SCALE_GROUP_COLS;
    }
    scaleOffset.push_back(0);
    return scaleOffset;
}

static void QuantMXOperationExeFunc1D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                      const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        auto args = static_cast<const QuantMXOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        const int firstViewShape = args->viewShape_[0];
        const int firstLoop = CeilDiv(firstDim, firstViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, firstLoop, 1))
        {
            std::vector<SymbolicScalar> offset = {bIdx * firstViewShape};
            auto viewTensor = View(inputs[0], args->viewShape_,
                                   {std::min(firstDim - bIdx * firstViewShape, firstViewShape)}, offset);
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = QuantMX(viewTensor, args->quantDtype_, args->mode_, args->axis_, args->performanceMode_);
            auto scaleOffset = BuildQuantMXScaleOffset(offset, args->axis_);
            Assemble(std::get<0>(res), offset, outputs[0]);
            Assemble(std::get<1>(res), scaleOffset, outputs[1]);
        }
    }
}

static void QuantMXOperationExeFunc2D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                      const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        auto args = static_cast<const QuantMXOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int firstLoop = CeilDiv(firstDim, firstViewShape);
        const int secondLoop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, firstLoop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, secondLoop, 1))
            {
                std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape};
                auto viewTensor = View(inputs[0], args->viewShape_,
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       offset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = QuantMX(viewTensor, args->quantDtype_, args->mode_, args->axis_, args->performanceMode_);
                auto scaleOffset = BuildQuantMXScaleOffset(offset, args->axis_);
                Assemble(std::get<0>(res), offset, outputs[0]);
                Assemble(std::get<1>(res), scaleOffset, outputs[1]);
            }
        }
    }
}

static void QuantMXOperationExeFunc3D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                      const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        auto args = static_cast<const QuantMXOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int firstLoop = CeilDiv(firstDim, firstViewShape);
        const int secondLoop = CeilDiv(secondDim, secondViewShape);
        const int thirdLoop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, firstLoop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, secondLoop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, thirdLoop, 1))
                {
                    std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                          nIdx * thirdViewShape};
                    auto viewTensor = View(inputs[0], args->viewShape_,
                                           {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                           offset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = QuantMX(viewTensor, args->quantDtype_, args->mode_, args->axis_, args->performanceMode_);
                    auto scaleOffset = BuildQuantMXScaleOffset(offset, args->axis_);
                    Assemble(std::get<0>(res), offset, outputs[0]);
                    Assemble(std::get<1>(res), scaleOffset, outputs[1]);
                }
            }
        }
    }
}

static void QuantMXOperationExeFunc4D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                      const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        auto args = static_cast<const QuantMXOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        const int firstLoop = CeilDiv(firstDim, firstViewShape);
        const int secondLoop = CeilDiv(secondDim, secondViewShape);
        const int thirdLoop = CeilDiv(thirdDim, thirdViewShape);
        const int fourthLoop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, firstLoop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, secondLoop, 1))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, thirdLoop, 1))
                {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, fourthLoop, 1))
                    {
                        std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                              mIdx * thirdViewShape, nIdx * fourthViewShape};
                        auto viewTensor = View(inputs[0], args->viewShape_,
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                               offset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = QuantMX(viewTensor, args->quantDtype_, args->mode_, args->axis_,
                                           args->performanceMode_);
                        auto scaleOffset = BuildQuantMXScaleOffset(offset, args->axis_);
                        Assemble(std::get<0>(res), offset, outputs[0]);
                        Assemble(std::get<1>(res), scaleOffset, outputs[1]);
                    }
                }
            }
        }
    }
}

class QuantMXOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<QuantMXOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestQuantMX, QuantMXOperationTest,
    ::testing::ValuesIn(GetOpMetaData<QuantMXOpMetaData, 1>({QuantMXOperationExeFunc1D, QuantMXOperationExeFunc2D,
                                                             QuantMXOperationExeFunc3D, QuantMXOperationExeFunc4D},
                                                            "QuantMX")));

TEST_P(QuantMXOperationTest, TestQuantMX)
{
    auto test_data = GetParam().test_data_;
    auto quantDtype = GetDataType(test_data.at("output_tensors")[0].at("dtype"));
    std::string modeStr = GetValueByName<std::string>(test_data, "mode");
    DequantScaleRoundingMode mode;
    if (modeStr == "ROUND_UP") {
        mode = DequantScaleRoundingMode::ROUND_UP;
    } else if (modeStr == "ROUND_DOWN") {
        mode = DequantScaleRoundingMode::ROUND_DOWN;
    } else {
        throw std::invalid_argument("Unsupported QuantMX mode: " + modeStr);
    }
    bool performanceMode = true;
    if (test_data.at("params").contains("performance_mode")) {
        performanceMode = GetValueByName<bool>(test_data, "performance_mode");
    }
    int64_t axis = -1;
    if (test_data.at("params").contains("axis")) {
        axis = GetValueByName<int64_t>(test_data, "axis");
    }
    auto args = QuantMXOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), quantDtype, mode, axis,
                                  performanceMode);
    auto testCase = CreateTestCaseDesc<QuantMXOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
