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
 * \file test_insert_op_for_viewassemble.cpp
 * \brief Unit test for InsertOpForViewAssemble pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include <vector>
#include <string>
#include "computational_graph_builder.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "ut_json/ut_json_tool.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"

#include "interface/tensor/irbuilder.h"
#define private public
#include "passes/tile_graph_pass/graph_optimization/insert_op_for_viewassemble.h"

namespace npu {
namespace tile_fwk {
static const size_t kSizeZero = 0UL;
static const uint16_t kNumFour = 4u;
static const size_t kSizeEight = 8UL;
static const size_t kSizeTwelve = 12UL;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumExpEight = 64u;

class TestInsertCopyPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(TestInsertCopyPass, TestNormalCase)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNormalCase", "TestNormalCase",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    /*
               | ------- view --- t0 --- assemble ------- |
             | ------- view --- t1 --- assemble  ---------- |
    inTensor [16, 16]                                         outTensor [16, 16]
             | ------- view --- t2 --- assemble  ---------- |
               | ------- view --- t3 --- assemble ------- |
 */
    // Prepare the graph
    std::vector<int64_t> shape = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> midShape = {kNumExpFour, kNumFour};
    std::vector<int64_t> offset0 = {kSizeZero, kSizeZero};
    std::vector<int64_t> offset1 = {kSizeZero, kNumFour};
    std::vector<int64_t> offset2 = {kSizeZero, kSizeEight};
    std::vector<int64_t> offset3 = {kSizeZero, kSizeTwelve};

    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto midTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor0->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto midTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto midTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto midTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);

    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor0});
    viewOp0.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0, MemoryType::MEM_DEVICE_DDR));
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor1});
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_DEVICE_DDR));
    auto& viewOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor2});
    viewOp2.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_DEVICE_DDR));
    auto& viewOp3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor3});
    viewOp3.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset3, MemoryType::MEM_DEVICE_DDR));

    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor0}, {outTensor});
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset0));
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor1}, {outTensor});
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset1));
    auto& assOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor2}, {outTensor});
    assOp2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset2));
    auto& assOp3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor3}, {outTensor});
    assOp3.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset3));

    InsertOpForViewAssemble pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), kSizeEight);
    EXPECT_EQ(midTensor0->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
}

TEST_F(TestInsertCopyPass, TestNoEqualSize)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNormalCase", "TestNormalCase",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    /*
               | ------- view --- t0 --- assemble ------- |
             | ------- view --- t1 --- assemble  ---------- |
    inTensor [16, 16]                                         outTensor [16, 64]
             | ------- view --- t2 --- assemble  ---------- |
               | ------- view --- t3 --- assemble ------- |
 */
    // Prepare the graph
    std::vector<int64_t> shape = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> shape1 = {kNumExpFour, kNumExpEight};
    std::vector<int64_t> midShape = {kNumExpFour, kNumFour};
    std::vector<int64_t> offset0 = {kSizeZero, kSizeZero};
    std::vector<int64_t> offset1 = {kSizeZero, kNumFour};
    std::vector<int64_t> offset2 = {kSizeZero, kSizeEight};
    std::vector<int64_t> offset3 = {kSizeZero, kSizeTwelve};

    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    outTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor0->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor3->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);

    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor0});
    viewOp0.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0, MemoryType::MEM_UB));
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor1});
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UB));
    auto& viewOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor2});
    viewOp2.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UB));
    auto& viewOp3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor3});
    viewOp3.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset3, MemoryType::MEM_UB));

    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor0}, {outTensor});
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset0));
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor1}, {outTensor});
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1));
    auto& assOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor2}, {outTensor});
    assOp2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset2));
    auto& assOp3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor3}, {outTensor});
    assOp3.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset3));

    InsertOpForViewAssemble pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), kNumExpFour);
}

TEST_F(TestInsertCopyPass, TestInsert)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNormalCase", "TestNormalCase",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    /*

             | ------- view ---------- t1 ----------- assemble  ---------- |
    inTensor [16, 16]                                                       outTensor [16, 16]
             | ------- view --- t2 --- EXP --- t3 --- assemble  ---------- |

 */
    std::vector<int64_t> shape = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> midShape = {kNumExpFour, kSizeEight};
    std::vector<int64_t> offset0 = {kSizeZero, kSizeZero};
    std::vector<int64_t> offset1 = {kSizeZero, kSizeEight};

    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor0->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);

    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor0});
    viewOp0.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0, MemoryType::MEM_UB));
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor1});
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UB));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {midTensor1}, {midTensor2});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor0}, {outTensor});
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset0));
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor2}, {outTensor});
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1));

    InsertOpForViewAssemble pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    const int result = 7;
    EXPECT_EQ(currFunctionPtr->Operations().size(), result);
}
} // namespace tile_fwk
} // namespace npu
