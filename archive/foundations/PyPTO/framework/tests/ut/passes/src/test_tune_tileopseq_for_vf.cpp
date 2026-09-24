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
 * \file test_tune_tileopseq_for_vf.cpp
 * \brief Unit test for TuneTileOpSeqForVF.
 */
#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include <algorithm>
#include "computational_graph_builder.h"
#include "tilefwk/platform.h"
#include "passes/block_graph_pass/tune_tileopseq_for_vf.h"
#include "interface/tensor/irbuilder.h"
#define private public

namespace npu {
namespace tile_fwk {
constexpr int TT_NUM10 = 10;
constexpr int TT_NUM20 = 20;
constexpr int TT_NUM30 = 30;
constexpr int TT_NUM40 = 40;
constexpr int TT_NUM50 = 50;
constexpr int TT_NUM60 = 60;
constexpr int TT_NUM70 = 70;
constexpr int TT_NUM80 = 80;
constexpr int TT_NUM90 = 90;
constexpr int TT_NUM100 = 100;
constexpr int TT_NUM16 = 16;
constexpr int TT_NUM5 = 5;

class TuneTileopseqForVFTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPassGlobalConfig(KEY_ENABLE_VF, true);
    }
    void TearDown() override {}

protected:
    std::shared_ptr<LogicalTensor> CreateTensor(int64_t start, int64_t end)
    {
        std::vector<int64_t> shape = {TT_NUM16, TT_NUM16};
        auto tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        tensor->memoryrange.start = start;
        tensor->memoryrange.end = end;
        return tensor;
    }

    std::pair<std::shared_ptr<Function>, std::shared_ptr<Function>> CreateFunctionPair(const std::string& rootName,
                                                                                       const std::string& leafName)
    {
        auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), rootName, rootName, nullptr);
        rootFuncPtr->rootFunc_ = rootFuncPtr.get();
        auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), leafName, leafName,
                                                          rootFuncPtr.get());
        rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
        return {rootFuncPtr, currFunctionPtr};
    }

    void SetupAndRunAdjustUbCopyNd2NzOrder(TuneTileOpSeqForVF& tuneTileop, PipeSync& ps,
                                           const std::vector<Operation*>& ops,
                                           const std::vector<std::vector<Operation*>>& mergedGroups)
    {
        tuneTileop.opList_ = ops;
        for (auto& op : tuneTileop.opList_) {
            op->SetAIVCore(AIVCore::AIV0);
            ps.BuildTensorRangeMap(op);
        }
        tuneTileop.mergedOps = mergedGroups;
        tuneTileop.AdjustUbCopyNd2NzOrder(ps);
    }
};

void BuildGraphForTest(std::shared_ptr<Function> currFunctionPtr, std::vector<Operation*>& opListPtr)
{
    std::vector<int64_t> shape = {TT_NUM16, TT_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->memoryrange.start = 0;
    tensor1->memoryrange.end = TT_NUM10;
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->memoryrange.start = TT_NUM10;
    tensor2->memoryrange.end = TT_NUM20;
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->memoryrange.start = TT_NUM20;
    tensor3->memoryrange.end = TT_NUM30;
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor4->memoryrange.start = TT_NUM30;
    tensor4->memoryrange.end = TT_NUM40;
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor5->memoryrange.start = TT_NUM40;
    tensor5->memoryrange.end = TT_NUM50;
    auto tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor6->memoryrange.start = TT_NUM50;
    tensor6->memoryrange.end = TT_NUM60;
    auto tensor7 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor7->memoryrange.start = TT_NUM60;
    tensor7->memoryrange.end = TT_NUM70;
    auto tensor8 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor8->memoryrange.start = TT_NUM70;
    tensor8->memoryrange.end = TT_NUM80;
    auto tensor9 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor9->memoryrange.start = TT_NUM80;
    tensor9->memoryrange.end = TT_NUM90;
    auto tensor10 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor10->memoryrange.start = TT_NUM90;
    tensor10->memoryrange.end = TT_NUM100;
    auto& vecop1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {tensor2});
    opListPtr.emplace_back(&vecop1);
    auto& vecop2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SQRT, {tensor3}, {tensor4});
    opListPtr.emplace_back(&vecop2);
    auto& vecop3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RECIPROCAL, {tensor5}, {tensor6});
    opListPtr.emplace_back(&vecop3);
    auto& op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_TRANSPOSE_MOVEIN, {tensor7}, {tensor8});
    opListPtr.emplace_back(&op1);
    auto& op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_TRANSPOSE_MOVEOUT, {tensor6}, {tensor9});
    opListPtr.emplace_back(&op2);
    auto& vecop4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXPAND, {tensor8}, {tensor10});
    opListPtr.emplace_back(&vecop4);
}

TEST_F(TuneTileopseqForVFTest, TestMergeForTuneTileop)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDep", "TestFindDep", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDepLeaf", "TestFindDepLeaf",
                                                      rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<Operation*> opListPtr;
    BuildGraphForTest(currFunctionPtr, opListPtr);
    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    tuneTileop.opList_ = opListPtr;
    for (auto& op : tuneTileop.opList_) {
        op->SetAIVCore(AIVCore::AIV0);
    }
    tuneTileop.ChangeOpSeq(ps, false);
    EXPECT_EQ(tuneTileop.opList_[0]->GetOpcode(), Opcode::OP_TRANSPOSE_MOVEIN);
    EXPECT_EQ(tuneTileop.opList_[TT_NUM5]->GetOpcode(), Opcode::OP_TRANSPOSE_MOVEOUT);
}

TEST_F(TuneTileopseqForVFTest, TestNotMergeForTuneTileop)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestTuneTileop", "TestTuneTileop", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestTuneTileopLeaf",
                                                      "TestTuneTileopLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<Operation*> opListPtr;
    BuildGraphForTest(currFunctionPtr, opListPtr);
    opListPtr[3]->GetIOperands()[0]->memoryrange.start = TT_NUM50;
    opListPtr[3]->GetIOperands()[0]->memoryrange.end = TT_NUM60;
    opListPtr[3]->GetOOperands()[0]->memoryrange.start = TT_NUM60;
    opListPtr[3]->GetOOperands()[0]->memoryrange.end = TT_NUM70;
    opListPtr[4]->GetIOperands()[0]->memoryrange.start = TT_NUM60;
    opListPtr[4]->GetIOperands()[0]->memoryrange.end = TT_NUM70;
    opListPtr[4]->GetOOperands()[0]->memoryrange.start = TT_NUM70;
    opListPtr[4]->GetOOperands()[0]->memoryrange.end = TT_NUM80;
    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    tuneTileop.opList_ = opListPtr;
    for (auto& op : tuneTileop.opList_) {
        op->SetAIVCore(AIVCore::AIV0);
    }
    tuneTileop.ChangeOpSeq(ps, false);
    EXPECT_EQ(tuneTileop.opList_[0]->GetOpcode(), Opcode::OP_EXP);
    EXPECT_EQ(tuneTileop.opList_[TT_NUM5]->GetOpcode(), Opcode::OP_EXPAND);
}

TEST_F(TuneTileopseqForVFTest, TestMainProcess)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestMainProcess", "TestMainProcess",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMainProcessLeaf",
                                                      "TestMainProcessLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {input}, {output});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MULACC_B, {input}, {output});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SYNC_SRC, {input}, {output});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SYNC_DST, {input}, {output});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_COPY_UB, {input}, {output});
    TuneTileOpSeqForVF tuneSync;
    tuneSync.RunOnFunction(*rootFuncPtr);
    auto it = rootFuncPtr->rootFunc_->programs_.begin();
    auto funcPtr = it->second;
    std::vector<Operation*> opList(funcPtr->Operations(false).DuplicatedOpList());
    EXPECT_EQ(opList.size(), TT_NUM5);
}

void BuildGraphForNonGroup(std::shared_ptr<Function> currFunctionPtr, std::vector<Operation*>& opListPtr)
{
    std::vector<int64_t> shape = {TT_NUM16, TT_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->memoryrange.start = TT_NUM40;
    tensor1->memoryrange.end = TT_NUM50;
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->memoryrange.start = TT_NUM50;
    tensor2->memoryrange.end = TT_NUM60;
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->memoryrange.start = TT_NUM60;
    tensor3->memoryrange.end = TT_NUM70;
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor4->memoryrange.start = TT_NUM70;
    tensor4->memoryrange.end = TT_NUM80;
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor5->memoryrange.start = TT_NUM90;
    tensor5->memoryrange.end = TT_NUM100;
    auto& vecop1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RECIPROCAL, {tensor1}, {tensor2});
    opListPtr.emplace_back(&vecop1);
    auto& op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_TRANSPOSE_MOVEIN, {tensor2}, {tensor3});
    opListPtr.emplace_back(&op1);
    auto& op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_TRANSPOSE_MOVEOUT, {tensor3}, {tensor4});
    opListPtr.emplace_back(&op2);
    auto& vecop2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXPAND, {tensor4}, {tensor5});
    opListPtr.emplace_back(&vecop2);
}

TEST_F(TuneTileopseqForVFTest, TestNonGroupCase)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestNonGroup", "TestNonGroup", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNonGroupLeaf", "TestNonGroupLeaf",
                                                      rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<Operation*> opListPtr;
    BuildGraphForNonGroup(currFunctionPtr, opListPtr);
    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    tuneTileop.opList_ = opListPtr;
    for (auto& op : tuneTileop.opList_) {
        op->SetAIVCore(AIVCore::AIV0);
    }
    tuneTileop.ChangeOpSeq(ps, false);
    EXPECT_EQ(tuneTileop.opList_[0]->GetOpcode(), Opcode::OP_RECIPROCAL);
    EXPECT_EQ(tuneTileop.opList_[3]->GetOpcode(), Opcode::OP_EXPAND);
}

TEST_F(TuneTileopseqForVFTest, TestSkip)
{
    config::SetPassGlobalConfig(KEY_ENABLE_VF, false);
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestSkip", "TestSkip", nullptr);
    TuneTileOpSeqForVF tuneSync;
    EXPECT_EQ(tuneSync.RunOnFunction(*rootFuncPtr.get()), SUCCESS);
}

/*
 * TestAdjustUbCopyNd2NzOrder_EmptyMergedOps
 * 测试mergedOps为空时，AdjustUbCopyNd2NzOrder函数的行为
 * 预期：函数正常执行，不做任何调整
 */
TEST_F(TuneTileopseqForVFTest, TestAdjustUbCopyNd2NzOrder_EmptyMergedOps)
{
    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    tuneTileop.mergedOps.clear();
    tuneTileop.AdjustUbCopyNd2NzOrder(ps);
    EXPECT_TRUE(tuneTileop.mergedOps.empty());
}

/*
 * TestAdjustUbCopyNd2NzOrder_NoUbCopyOp
 * 测试组内无UB_COPY_ND2NZ op时，AdjustUbCopyNd2NzOrder函数的行为
 * 预期：函数正常执行，不调整操作顺序
 */
TEST_F(TuneTileopseqForVFTest, TestAdjustUbCopyNd2NzOrder_NoUbCopyOp)
{
    auto [rootFuncPtr, currFunctionPtr] = CreateFunctionPair("TestNoUbCopy", "TestNoUbCopyLeaf");
    EXPECT_TRUE(currFunctionPtr != nullptr);

    auto tensor1 = CreateTensor(0, TT_NUM10);
    auto tensor2 = CreateTensor(TT_NUM10, TT_NUM20);
    auto tensor3 = CreateTensor(TT_NUM20, TT_NUM30);
    auto tensor4 = CreateTensor(TT_NUM30, TT_NUM40);

    auto& op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {tensor2});
    auto& op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SQRT, {tensor3}, {tensor4});

    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    SetupAndRunAdjustUbCopyNd2NzOrder(tuneTileop, ps, {&op1, &op2}, {{&op1, &op2}});

    EXPECT_EQ(tuneTileop.opList_.size(), 2U);
    EXPECT_EQ(tuneTileop.opList_[0]->GetOpcode(), Opcode::OP_EXP);
    EXPECT_EQ(tuneTileop.opList_[1]->GetOpcode(), Opcode::OP_SQRT);
}

/*
 * TestAdjustUbCopyNd2NzOrder_NoNonUbCopyOp
 * 测试组内只有UB_COPY_ND2NZ op时，AdjustUbCopyNd2NzOrder函数的行为
 * 预期：函数正常执行，不调整操作顺序
 */
TEST_F(TuneTileopseqForVFTest, TestAdjustUbCopyNd2NzOrder_NoNonUbCopyOp)
{
    auto [rootFuncPtr, currFunctionPtr] = CreateFunctionPair("TestNoNonUbCopy", "TestNoNonUbCopyLeaf");
    EXPECT_TRUE(currFunctionPtr != nullptr);

    auto tensor1 = CreateTensor(0, TT_NUM10);
    auto tensor2 = CreateTensor(TT_NUM10, TT_NUM20);
    auto tensor3 = CreateTensor(TT_NUM20, TT_NUM30);
    auto tensor4 = CreateTensor(TT_NUM30, TT_NUM40);

    auto& ubCopyOp1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_UB_COPY_ND2NZ, {tensor1}, {tensor2});
    auto& ubCopyOp2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_UB_COPY_ND2NZ, {tensor3}, {tensor4});

    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    SetupAndRunAdjustUbCopyNd2NzOrder(tuneTileop, ps, {&ubCopyOp1, &ubCopyOp2}, {{&ubCopyOp1, &ubCopyOp2}});

    EXPECT_EQ(tuneTileop.opList_.size(), 2U);
    EXPECT_EQ(tuneTileop.opList_[0]->GetOpcode(), Opcode::OP_UB_COPY_ND2NZ);
    EXPECT_EQ(tuneTileop.opList_[1]->GetOpcode(), Opcode::OP_UB_COPY_ND2NZ);
}

/*
 * TestAdjustUbCopyNd2NzOrder_UbCopyMoveFront
 * 测试UB_COPY_ND2NZ可以前移的情况
 * 初始顺序：[VEC_OP1, UB_COPY_ND2NZ, VEC_OP2]
 * 预期结果：[UB_COPY_ND2NZ, VEC_OP1, VEC_OP2]
 */
TEST_F(TuneTileopseqForVFTest, TestAdjustUbCopyNd2NzOrder_UbCopyMoveFront)
{
    auto [rootFuncPtr, currFunctionPtr] = CreateFunctionPair("TestMoveFront", "TestMoveFrontLeaf");
    EXPECT_TRUE(currFunctionPtr != nullptr);

    auto tensor1 = CreateTensor(0, TT_NUM10);
    auto tensor2 = CreateTensor(TT_NUM10, TT_NUM20);
    auto tensor3 = CreateTensor(TT_NUM40, TT_NUM50);
    auto tensor4 = CreateTensor(TT_NUM50, TT_NUM60);
    auto tensor5 = CreateTensor(TT_NUM60, TT_NUM70);
    auto tensor6 = CreateTensor(TT_NUM70, TT_NUM80);

    auto& vecOp1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {tensor2});
    auto& ubCopyOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_UB_COPY_ND2NZ, {tensor3}, {tensor4});
    auto& vecOp2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SQRT, {tensor5}, {tensor6});

    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    SetupAndRunAdjustUbCopyNd2NzOrder(tuneTileop, ps, {&vecOp1, &ubCopyOp, &vecOp2}, {{&vecOp1, &ubCopyOp, &vecOp2}});

    EXPECT_EQ(tuneTileop.opList_.size(), 3U);
    EXPECT_EQ(tuneTileop.opList_[0]->GetOpcode(), Opcode::OP_UB_COPY_ND2NZ);
}

/*
 * TestAdjustUbCopyNd2NzOrder_UbCopyMoveBack
 * 测试UB_COPY_ND2NZ可以后移的情况
 * 初始顺序：[VEC_OP1, UB_COPY_ND2NZ, VEC_OP2]
 * 当UB_COPY_ND2NZ与VEC_OP1存在依赖但与VEC_OP2无依赖时，UB_COPY_ND2NZ应后移
 * 预期结果：[VEC_OP1, VEC_OP2, UB_COPY_ND2NZ]
 */
TEST_F(TuneTileopseqForVFTest, TestAdjustUbCopyNd2NzOrder_UbCopyMoveBack)
{
    auto [rootFuncPtr, currFunctionPtr] = CreateFunctionPair("TestMoveBack", "TestMoveBackLeaf");
    EXPECT_TRUE(currFunctionPtr != nullptr);

    auto tensor1 = CreateTensor(0, TT_NUM10);
    auto tensor2 = CreateTensor(TT_NUM10, TT_NUM20);
    auto tensor3 = CreateTensor(TT_NUM20, TT_NUM30);
    auto tensor4 = CreateTensor(TT_NUM40, TT_NUM50);
    auto tensor5 = CreateTensor(TT_NUM50, TT_NUM60);

    auto& vecOp1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {tensor2});
    auto& ubCopyOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_UB_COPY_ND2NZ, {tensor2}, {tensor3});
    auto& vecOp2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SQRT, {tensor4}, {tensor5});

    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    SetupAndRunAdjustUbCopyNd2NzOrder(tuneTileop, ps, {&vecOp1, &ubCopyOp, &vecOp2}, {{&vecOp1, &ubCopyOp, &vecOp2}});

    EXPECT_EQ(tuneTileop.opList_.size(), 3U);
    EXPECT_EQ(tuneTileop.opList_[2]->GetOpcode(), Opcode::OP_UB_COPY_ND2NZ);
}

/*
 * TestAdjustUbCopyNd2NzOrder_UbCopyCannotMove
 * 测试UB_COPY_ND2NZ因数据依赖无法移动的情况
 * 当UB_COPY_ND2NZ与前后的VEC_OP都有依赖时，无法移动
 * 初始顺序：[VEC_OP1, UB_COPY_ND2NZ, VEC_OP2]
 * 预期结果：顺序不变
 */
TEST_F(TuneTileopseqForVFTest, TestAdjustUbCopyNd2NzOrder_UbCopyCannotMove)
{
    auto [rootFuncPtr, currFunctionPtr] = CreateFunctionPair("TestCannotMove", "TestCannotMoveLeaf");
    EXPECT_TRUE(currFunctionPtr != nullptr);

    auto tensor1 = CreateTensor(0, TT_NUM10);
    auto tensor2 = CreateTensor(TT_NUM10, TT_NUM20);
    auto tensor3 = CreateTensor(TT_NUM20, TT_NUM30);
    auto tensor4 = CreateTensor(TT_NUM30, TT_NUM40);

    auto& vecOp1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {tensor2});
    auto& ubCopyOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_UB_COPY_ND2NZ, {tensor2}, {tensor3});
    auto& vecOp2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SQRT, {tensor3}, {tensor4});

    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    SetupAndRunAdjustUbCopyNd2NzOrder(tuneTileop, ps, {&vecOp1, &ubCopyOp, &vecOp2}, {{&vecOp1, &ubCopyOp, &vecOp2}});

    EXPECT_EQ(tuneTileop.opList_.size(), 3U);
    EXPECT_EQ(tuneTileop.opList_[0]->GetOpcode(), Opcode::OP_EXP);
    EXPECT_EQ(tuneTileop.opList_[1]->GetOpcode(), Opcode::OP_UB_COPY_ND2NZ);
    EXPECT_EQ(tuneTileop.opList_[2]->GetOpcode(), Opcode::OP_SQRT);
}

/*
 * TestAdjustUbCopyNd2NzOrder_MultipleUbCopyOps
 * 测试多个UB_COPY_ND2NZ操作的情况
 * 初始顺序：[UB_COPY_ND2NZ_1, VEC_OP, UB_COPY_ND2NZ_2]
 * 预期：UB_COPY_ND2NZ_1前移，UB_COPY_ND2NZ_2后移或保持
 */
TEST_F(TuneTileopseqForVFTest, TestAdjustUbCopyNd2NzOrder_MultipleUbCopyOps)
{
    auto [rootFuncPtr, currFunctionPtr] = CreateFunctionPair("TestMultipleUbCopy", "TestMultipleUbCopyLeaf");
    EXPECT_TRUE(currFunctionPtr != nullptr);

    auto tensor1 = CreateTensor(0, TT_NUM10);
    auto tensor2 = CreateTensor(TT_NUM10, TT_NUM20);
    auto tensor3 = CreateTensor(TT_NUM30, TT_NUM40);
    auto tensor4 = CreateTensor(TT_NUM40, TT_NUM50);
    auto tensor5 = CreateTensor(TT_NUM60, TT_NUM70);
    auto tensor6 = CreateTensor(TT_NUM70, TT_NUM80);

    auto& ubCopyOp1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_UB_COPY_ND2NZ, {tensor1}, {tensor2});
    auto& vecOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXP, {tensor3}, {tensor4});
    auto& ubCopyOp2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_UB_COPY_ND2NZ, {tensor5}, {tensor6});

    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    SetupAndRunAdjustUbCopyNd2NzOrder(tuneTileop, ps, {&ubCopyOp1, &vecOp, &ubCopyOp2},
                                      {{&ubCopyOp1, &vecOp, &ubCopyOp2}});

    EXPECT_EQ(tuneTileop.opList_.size(), 3U);
    // 验证UB_COPY_ND2NZ操作在VEC操作之前或之后
    auto findVecOp = std::find(tuneTileop.opList_.begin(), tuneTileop.opList_.end(), &vecOp);
    EXPECT_NE(findVecOp, tuneTileop.opList_.end());
}

/*
 * TestAdjustUbCopyNd2NzOrder_MultipleGroups
 * 测试多个融合组的情况
 * 预期：每个组内的UB_COPY_ND2NZ操作都会被调整
 */
TEST_F(TuneTileopseqForVFTest, TestAdjustUbCopyNd2NzOrder_MultipleGroups)
{
    auto [rootFuncPtr, currFunctionPtr] = CreateFunctionPair("TestMultipleGroups", "TestMultipleGroupsLeaf");
    EXPECT_TRUE(currFunctionPtr != nullptr);

    auto tensor1 = CreateTensor(0, TT_NUM10);
    auto tensor2 = CreateTensor(TT_NUM10, TT_NUM20);
    auto tensor3 = CreateTensor(TT_NUM20, TT_NUM30);
    auto tensor4 = CreateTensor(TT_NUM30, TT_NUM40);
    auto tensor5 = CreateTensor(TT_NUM50, TT_NUM60);
    auto tensor6 = CreateTensor(TT_NUM60, TT_NUM70);
    auto tensor7 = CreateTensor(TT_NUM70, TT_NUM80);
    auto tensor8 = CreateTensor(TT_NUM80, TT_NUM90);

    auto& vecOp1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {tensor2});
    auto& ubCopyOp1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_UB_COPY_ND2NZ, {tensor3}, {tensor4});
    auto& vecOp2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SQRT, {tensor5}, {tensor6});
    auto& ubCopyOp2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_UB_COPY_ND2NZ, {tensor7}, {tensor8});

    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    SetupAndRunAdjustUbCopyNd2NzOrder(tuneTileop, ps, {&vecOp1, &ubCopyOp1, &vecOp2, &ubCopyOp2},
                                      {{&vecOp1, &ubCopyOp1}, {&vecOp2, &ubCopyOp2}});
    EXPECT_EQ(tuneTileop.opList_.size(), 4U);
}

} // namespace tile_fwk
} // namespace npu

#undef private
