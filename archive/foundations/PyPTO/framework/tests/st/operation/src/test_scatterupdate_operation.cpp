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
 * \file test_scatterupdate_operation.cpp
 * \brief
 */

#include <nlohmann/json.hpp>
#include "test_operation.h"
#include "tilefwk/tensor.h"

using namespace tile_fwk::test_operation;
namespace {
struct ScatterUpdateOpFuncArgs : public OpFuncArgs {
    ScatterUpdateOpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape)
    {
        this->inplaceInfo[0] = 2;
    }

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct ScatterUpdateOpMetaData {
    explicit ScatterUpdateOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void ScatterUpdateOperationExeFunc4Dims(const std::vector<Tensor>& input, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    std::vector<Tensor>& inputs = const_cast<std::vector<Tensor>&>(input);
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        auto args = static_cast<const ScatterUpdateOpFuncArgs*>(opArgs);
        const int64_t b = inputs[0].GetShape()[0];
        const int64_t s = inputs[0].GetShape()[1];
        const int64_t n = inputs[0].GetShape()[2];
        const int64_t d = inputs[0].GetShape()[3];
        const int64_t bViewShape = args->viewShape_[0];
        const int64_t sViewShape = args->viewShape_[1];

        const int64_t bloop = CeilDiv(b, bViewShape);
        const int64_t sloop = CeilDiv(s, sViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1))
            {
                TileShape::Current().SetVecTile(args->tileShape_);
                Tensor srcView = View(
                    inputs[0], {bViewShape, sViewShape, n, d},
                    {std::min(b - bIdx * bViewShape, bViewShape), std::min(s - sIdx * sViewShape, sViewShape), n, d},
                    {bIdx * bViewShape, sIdx * sViewShape, 0, 0});
                Tensor indexView = View(
                    inputs[1], {bViewShape, sViewShape},
                    {std::min(b - bIdx * bViewShape, bViewShape), std::min(s - sIdx * sViewShape, sViewShape)},
                    {bIdx * bViewShape, sIdx * sViewShape});
                Tensor dst = View(inputs[2], inputs[2].GetShape(), {0, 0, 0, 0});
                dst = ScatterUpdate(dst, indexView, srcView, -2, "PA_BSND", 1);
                TileShape::Current().SetVecTile({1, 64, 1, d});
                Assemble(dst, {0, 0, 0, 0}, outputs[0]);
            }
        }
    }
}

static void ScatterUpdateOperationExeFunc2Dims(const std::vector<Tensor>& input, std::vector<Tensor>& outputs,
                                               const OpFuncArgs* opArgs)
{
    std::vector<Tensor>& inputs = const_cast<std::vector<Tensor>&>(input);
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        auto args = static_cast<const ScatterUpdateOpFuncArgs*>(opArgs);
        const int64_t b = inputs[1].GetShape()[0];
        const int64_t s = inputs[1].GetShape()[1];
        const int64_t bs = inputs[0].GetShape()[0];
        const int64_t d = inputs[0].GetShape()[1];
        const int64_t bsViewShape = args->viewShape_[0];
        const int64_t bViewShape = bsViewShape / s;
        const int64_t bloop = CeilDiv(b, bViewShape);
        LOOP("LOOP_L0_bsIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1))
        {
            TileShape::Current().SetVecTile(args->tileShape_);
            Tensor srcView = View(inputs[0], {bsViewShape, d}, {std::min(bs - bIdx * bsViewShape, bsViewShape), d},
                                  {bIdx * bsViewShape, 0});
            Tensor indexView = View(inputs[1], {bViewShape, s}, {std::min(b - bIdx * bViewShape, bViewShape), s},
                                    {bIdx * bViewShape, 0});
            Tensor dst = View(inputs[2], inputs[2].GetShape(), {0, 0});
            dst = ScatterUpdate(dst, indexView, srcView, -2, "PA_BSND", 1);
            TileShape::Current().SetVecTile({32, d});
            Assemble(dst, {0, 0}, outputs[0]);
        }
    }
}

class ScatterUpdateOperationTest
    : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ScatterUpdateOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestScatterUpdate, ScatterUpdateOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<ScatterUpdateOpMetaData>(
                             {ScatterUpdateOperationExeFunc2Dims, ScatterUpdateOperationExeFunc4Dims,
                              ScatterUpdateOperationExeFunc4Dims},
                             "ScatterUpdate")));

TEST_P(ScatterUpdateOperationTest, TestScatterUpdate)
{
    auto test_data = GetParam().test_data_;
    auto args = ScatterUpdateOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<ScatterUpdateOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
