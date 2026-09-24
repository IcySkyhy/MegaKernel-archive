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
 * \file test_remove_alloc.cpp
 * \brief Unit test for Remove Alloc pass.
 */
#include <vector>
#include "symbolic_scalar_test_utils.h"
#include <string>

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/remove_alloc.h"
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;

class RemoveAllocTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "RemoveAllocTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(RemoveAllocTest, RemoveAlloc)
{
    constexpr int CP_NUM16 = 16;
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestSaveGmTensorParamIdxToOp",
                                                  "TestSaveGmTensorParamIdxToOp", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSaveGmTensorParamIdxToOpLeaf",
                                                      "TestSaveGmTensorParamIdxToOpLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    currFunctionPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<int64_t> shape = {CP_NUM16, CP_NUM16};
    auto tensor1 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor4 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor5 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor6 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor2}, {tensor4});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {tensor3, tensor4}, {tensor5});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensor5}, {tensor6});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_UB_ALLOC, {}, {tensor5});

    npu::tile_fwk::RemoveAlloc removeAllocPass;
    removeAllocPass.RemoveAllocCall(*rootFuncPtr);

    // ================== Verify Pass Effect ==================
    constexpr int expectOpNum = 4;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expectOpNum) << expectOpNum << " operations";
}
