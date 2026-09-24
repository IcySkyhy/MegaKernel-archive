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
 * \file test_codegen_scatterupdate.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"
#include "test_codegen_common.h"
#include "codegen/codegen.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_mgr/pass_manager.h"
#include "tilefwk/tilefwk_op.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {

class TestCodegenScatterUpdate : public CodegenTestBase {
public:
    TestCodegenScatterUpdate()
        : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH,
                           .setTileTensor = true,
                           .tileTensorValue = false,
                           .resetTileTensorOnTearDown = true})
    {}
};

// ScatterUpdate
void TestScatterUpdate(std::vector<int64_t> tileShape)
{
    TileShape::Current().SetVecTile(tileShape);

    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                     {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                     {"DuplicateOp", PassName::DUPLICATE_OP},
                                     {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                     {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                     {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                     {"SplitReshape", PassName::SPLIT_RESHAPE},
                                     {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
                                     {"GenerateMoveOp", PassName::GENERATE_MOVE_OP},
                                 });

    int h = 128, minusTwo = -2;
    Tensor output(DT_INT32, {h, h}, "output");
    Tensor idxs(DT_INT32, {h, h}, "idxs");
    Tensor keyStates(DT_INT32, {h, h}, "keyStates");

    std::string funcName = "ScatterUpdate";
    FUNCTION(funcName) { output = ScatterUpdate(output, idxs, keyStates, minusTwo); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenScatterUpdate, TestScatterupdateDim2) { TestScatterUpdate({16, 32}); }

TEST_F(TestCodegenScatterUpdate, TestBatchMatmul)
{
    int bs = 1;
    int m = 32;
    int k = 32;
    int n = 32;

    std::vector<int64_t> shapeA = {bs, m, k};
    std::vector<int64_t> shapeB = {bs, k, n};
    std::vector<int64_t> shapeC = {bs, m, n};

    config::Reset();
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    Tensor matA(DT_FP16, shapeA, "MatA", TileOpFormat::TILEOP_NZ);
    Tensor matB(DT_FP16, shapeB, "MatB", TileOpFormat::TILEOP_ND);
    Tensor matC(DT_FP32, shapeC, "MatC");
    std::string funcName = "BATCHMATMUL";
    config::SetBuildStatic(true);
    FUNCTION(funcName, {matA, matB, matC})
    {
        matC = npu::tile_fwk::Matrix::BatchMatmul(DT_FP32, matA, matB, false, false);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenScatterUpdate, TestScatterUpdate)
{
    int S = 1;
    int S2 = 16;
    int kvLoraRank = 8;
    int qkRopeHeadDim = 8;

    std::vector<int64_t> shape0 = {S2, kvLoraRank + qkRopeHeadDim}; // [16, 16]
    std::vector<int64_t> shape1 = {1, S};
    std::vector<int64_t> shape2 = {S, kvLoraRank + qkRopeHeadDim}; // [1, 16]

    TileShape::Current().SetVecTile(16, 16);

    Tensor kv_len(DT_INT64, shape1, "kv_len");
    Tensor past_key_states(DT_FP32, shape0, "past_key_states");
    Tensor key_states(DT_FP32, shape2, "key_states"); // [16,16]

    // Tensor past_key_states_new(DT_FP32, shape0, "past_key_states_new");

    /* torch capture */
    std::string funcName = "ScatterUpdate";
    FUNCTION(funcName) { past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
} // namespace npu::tile_fwk
