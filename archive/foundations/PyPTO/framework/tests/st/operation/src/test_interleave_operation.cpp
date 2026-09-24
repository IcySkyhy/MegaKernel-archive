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
 * \file test_interleave_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
using TileBody = std::function<void(const std::vector<SymbolicScalar>&, const std::vector<SymbolicScalar>&)>;

struct InterleaveOpFuncArgs : public OpFuncArgs {
    InterleaveOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape,
                         bool singleInput = false)
        : viewShape_(viewShape), tileShape_(tileShape), singleInput_(singleInput)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    bool singleInput_;
};

struct InterleaveOpMetaData {
    explicit InterleaveOpMetaData(const OpFunc& opFunc, const nlohmann::json& testData)
        : opFunc_(opFunc), test_data_(testData)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static std::vector<SymbolicScalar> GetTensorShape(const Tensor& tensor, size_t dim)
{
    std::vector<SymbolicScalar> shape;
    shape.reserve(dim);
    for (size_t i = 0; i < dim; ++i) {
        shape.push_back(tensor.GetShape()[i]);
    }
    return shape;
}

static std::vector<int> GetLoops(const std::vector<SymbolicScalar>& shape, const std::vector<int64_t>& viewShape)
{
    std::vector<int> loops;
    loops.reserve(viewShape.size());
    for (size_t i = 0; i < viewShape.size(); ++i) {
        loops.push_back(CeilDiv(shape[i], viewShape[i]));
    }
    return loops;
}

static std::vector<SymbolicScalar> GetValidShape(const std::vector<SymbolicScalar>& shape,
                                                 const std::vector<int64_t>& viewShape,
                                                 const std::vector<SymbolicScalar>& offset)
{
    std::vector<SymbolicScalar> validShape;
    validShape.reserve(viewShape.size());
    for (size_t i = 0; i < viewShape.size(); ++i) {
        validShape.push_back(std::min(shape[i] - offset[i], viewShape[i]));
    }
    return validShape;
}

static void RunTileLoop(size_t cur, const std::vector<SymbolicScalar>& shape, const std::vector<int>& loops,
                        const std::vector<int64_t>& viewShape, std::vector<SymbolicScalar>& offset,
                        const TileBody& body)
{
    if (cur == viewShape.size()) {
        body(offset, GetValidShape(shape, viewShape, offset));
        return;
    }

    switch (cur) {
        case 0:
            LOOP("LOOP_INTERLEAVE_DIM0", FunctionType::DYNAMIC_LOOP, idx0, LoopRange(0, loops[cur], 1))
            {
                offset[cur] = idx0 * viewShape[cur];
                RunTileLoop(cur + 1, shape, loops, viewShape, offset, body);
            }
            break;
        case 1:
            LOOP("LOOP_INTERLEAVE_DIM1", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(0, loops[cur], 1))
            {
                offset[cur] = idx1 * viewShape[cur];
                RunTileLoop(cur + 1, shape, loops, viewShape, offset, body);
            }
            break;
        case 2:
            LOOP("LOOP_INTERLEAVE_DIM2", FunctionType::DYNAMIC_LOOP, idx2, LoopRange(0, loops[cur], 1))
            {
                offset[cur] = idx2 * viewShape[cur];
                RunTileLoop(cur + 1, shape, loops, viewShape, offset, body);
            }
            break;
        default:
            LOOP("LOOP_INTERLEAVE_DIM3", FunctionType::DYNAMIC_LOOP, idx3, LoopRange(0, loops[cur], 1))
            {
                offset[cur] = idx3 * viewShape[cur];
                RunTileLoop(cur + 1, shape, loops, viewShape, offset, body);
            }
            break;
    }
}

static void RunRankedTileLoop(const Tensor& input, const std::vector<int64_t>& viewShape, const TileBody& body)
{
    auto shape = GetTensorShape(input, viewShape.size());
    auto loops = GetLoops(shape, viewShape);
    std::vector<SymbolicScalar> offset(viewShape.size(), 0);
    RunTileLoop(0, shape, loops, viewShape, offset, body);
}

static std::vector<SymbolicScalar> GetSingleInputDeInterleaveOutputOffset(std::vector<SymbolicScalar> inputOffset)
{
    inputOffset.back() = inputOffset.back() / 2;
    return inputOffset;
}

template <size_t RANK>
static void InterleaveOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                       const OpFuncArgs* opArgs)
{
    static_assert(RANK >= 1 && RANK <= 4);
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0], outputs[1]})
    {
        auto args = static_cast<const InterleaveOpFuncArgs*>(opArgs);
        RunRankedTileLoop(inputs[0], args->viewShape_, [&](const auto& offset, const auto& validShape) {
            Tensor tileTensor0 = View(inputs[0], args->viewShape_, validShape, offset);
            Tensor tileTensor1 = View(inputs[1], args->viewShape_, validShape, offset);
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Interleave(tileTensor0, tileTensor1);
            Assemble(std::get<0>(res), offset, outputs[0]);
            Assemble(std::get<1>(res), offset, outputs[1]);
        });
    }
}

template <size_t RANK>
static void DeInterleaveOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs)
{
    static_assert(RANK >= 1 && RANK <= 4);
    auto args = static_cast<const InterleaveOpFuncArgs*>(opArgs);
    if (args->singleInput_) {
        FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
        {
            RunRankedTileLoop(inputs[0], args->viewShape_, [&](const auto& offset, const auto& validShape) {
                Tensor tileTensor = View(inputs[0], args->viewShape_, validShape, offset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = DeInterleave(tileTensor);
                auto outputOffset = GetSingleInputDeInterleaveOutputOffset(offset);
                Assemble(std::get<0>(res), outputOffset, outputs[0]);
                Assemble(std::get<1>(res), outputOffset, outputs[1]);
            });
        }
        return;
    }

    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0], outputs[1]})
    {
        RunRankedTileLoop(inputs[0], args->viewShape_, [&](const auto& offset, const auto& validShape) {
            Tensor tileTensor0 = View(inputs[0], args->viewShape_, validShape, offset);
            Tensor tileTensor1 = View(inputs[1], args->viewShape_, validShape, offset);
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = DeInterleave(tileTensor0, tileTensor1);
            Assemble(std::get<0>(res), offset, outputs[0]);
            Assemble(std::get<1>(res), offset, outputs[1]);
        });
    }
}

class InterleaveOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<InterleaveOpMetaData> {};
class DeInterleaveOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<InterleaveOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestInterleave, InterleaveOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<InterleaveOpMetaData, 1>(
                             {InterleaveOperationExeFunc<1>, InterleaveOperationExeFunc<2>,
                              InterleaveOperationExeFunc<3>, InterleaveOperationExeFunc<4>},
                             "Interleave")));

INSTANTIATE_TEST_SUITE_P(TestDeInterleave, DeInterleaveOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<InterleaveOpMetaData, 1>(
                             {DeInterleaveOperationExeFunc<1>, DeInterleaveOperationExeFunc<2>,
                              DeInterleaveOperationExeFunc<3>, DeInterleaveOperationExeFunc<4>},
                             "DeInterleave")));

TEST_P(InterleaveOperationTest, TestInterleave)
{
    auto testData = GetParam().test_data_;
    auto args = InterleaveOpFuncArgs(GetViewShape(testData), GetTileShape(testData));
    auto testCase = CreateTestCaseDesc<InterleaveOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

TEST_P(DeInterleaveOperationTest, TestDeInterleave)
{
    auto testData = GetParam().test_data_;
    bool singleInput = false;
    if (testData.at("params").find("single_input") != testData.at("params").end()) {
        singleInput = GetValueByName<bool>(testData, "single_input");
    }
    auto args = InterleaveOpFuncArgs(GetViewShape(testData), GetTileShape(testData), singleInput);
    auto testCase = CreateTestCaseDesc<InterleaveOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
