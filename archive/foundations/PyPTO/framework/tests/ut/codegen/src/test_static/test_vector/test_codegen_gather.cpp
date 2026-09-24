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
 * \file test_codegen_gather.cpp
 * \brief Unit test for codegen.
 */
#include <vector>
#include <string>
using std::string;
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenGather : public CodegenTestBase {
public:
    TestCodegenGather() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .setTileTensor = true}) {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

constexpr const int GATHER_SHAPE0 = 16;
constexpr const int GATHER_SHAPE1 = 32;

Function& testGatherEle(bool isSupportTileTensor, string funcName)
{
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    } else {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }
    constexpr const int32_t nRoutedExperts = 32;
    constexpr const int32_t numExpertsPerTopk = 8;
    constexpr const int32_t S = 1;
    constexpr const int32_t B = 2;

    std::vector<int64_t> inputShape = {B * S, nRoutedExperts};
    std::vector<int64_t> outputShape = {B * S, numExpertsPerTopk};
    TileShape::Current().SetVecTile({GATHER_SHAPE0, GATHER_SHAPE1});
    Tensor inputScores(DT_FP32, inputShape, "input_scores");
    Tensor inputTmpScores(DT_FP32, inputShape, "input_tmp_scores");
    Tensor outputTensor(DT_FP32, outputShape, "output_tensor");

    config::SetBuildStatic(true);
    FUNCTION(funcName, {inputScores, inputTmpScores, outputTensor})
    {
        auto topkIdx = std::get<1>(TopK(inputScores, numExpertsPerTopk, -1));      // [b*s,256]->[b*s,8]
        auto topkWeight = GatherElements(inputTmpScores, topkIdx, 1);              // [b*s,8]
        auto topkWeightSum = Sum(topkWeight, 1, true);                             // [b*s,8]->[b*s,1]
        auto denominator = Add(topkWeightSum, Element(DataType::DT_FP32, 1e-20f)); // [b*s,1]
        outputTensor = Div(topkWeight, denominator);                               // [b*s,numExpertsPerTok]
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
    return *function;
}
TEST_F(TestCodegenGather, TestGatherEle) { testGatherEle(false, "GATHER_ELEMET_T"); }

TEST_F(TestCodegenGather, TestGatherEleTileTensor)
{
    Function& func = testGatherEle(true, "GATHER_ELEMET_TILETENSOR");
    std::string res = GetResultFromCpp(func);
    std::string expect = R"!!!(TgatherElement<4>(ubTensor_12, ubTensor_5, ubTensor_10, ubTensor_13);
)!!!";
    CheckStringExist(expect, res);
}

} // namespace npu::tile_fwk
