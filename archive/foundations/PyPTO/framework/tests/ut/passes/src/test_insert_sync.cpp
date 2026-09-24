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
 * \file test_insert_sync.cpp
 * \brief Unit test for InsertSync.
 */
#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/platform.h"
#include "interface/tensor/irbuilder.h"
#define private public
#include "passes/block_graph_pass/insert_sync.h"
#include "ut_json/ut_json_tool.h"

namespace npu {
namespace tile_fwk {
constexpr int IS_NUM1 = 1;
constexpr int IS_NUM2 = 2;
constexpr int IS_NUM3 = 3;
constexpr int IS_NUM4 = 4;
constexpr int IS_NUM5 = 5;
constexpr int IS_NUM6 = 6;
constexpr int IS_NUM7 = 7;
constexpr int IS_NUM8 = 8;
constexpr int IS_NUM9 = 9;
constexpr int IS_NUM10 = 10;
constexpr int IS_NUM11 = 11;
constexpr int IS_NUM12 = 12;
constexpr int IS_NUM16 = 16;
constexpr int IS_NUM18 = 18;
constexpr int IS_NUM19 = 19;
constexpr int IS_NUM20 = 20;
constexpr int IS_NUM29 = 29;
constexpr int IS_NUM30 = 30;
constexpr int IS_NUM32 = 32;
constexpr int IS_NUM39 = 39;
constexpr int IS_NUM40 = 40;
constexpr int IS_NUM49 = 49;
constexpr int IS_NUM50 = 50;
constexpr int IS_NUM59 = 59;
constexpr int IS_NUM60 = 60;
constexpr int IS_NUM69 = 69;
constexpr int IS_NUM70 = 70;
constexpr int IS_NUM79 = 79;
constexpr int IS_NUM80 = 80;
constexpr int IS_NUM89 = 89;
constexpr int IS_NUM90 = 90;
constexpr int IS_NUM99 = 99;
constexpr int IS_NUM100 = 100;
constexpr int IS_NUM101 = 101;
constexpr int IS_NUM109 = 109;
constexpr int IS_NUM110 = 110;
constexpr int IS_NUM119 = 119;
constexpr int IS_NUM120 = 120;
constexpr int IS_NUM129 = 129;
constexpr int IS_NUM130 = 130;
constexpr int IS_NUM139 = 139;
constexpr int IS_NUM140 = 140;
constexpr int IS_NUM149 = 149;
constexpr int IS_NUM150 = 150;
constexpr int IS_NUM159 = 159;
constexpr int IS_NUM160 = 160;
constexpr int IS_NUM169 = 169;
constexpr int IS_NUM170 = 170;
constexpr int IS_NUM179 = 179;
constexpr int IS_NUM180 = 180;
constexpr int IS_NUM189 = 189;
constexpr int IS_NUM190 = 190;
constexpr int IS_NUM199 = 199;
constexpr int IS_NUM200 = 200;
constexpr int IS_NUM209 = 209;
constexpr int IS_NUM210 = 210;
constexpr int IS_NUM219 = 219;
constexpr int IS_NUM220 = 220;
constexpr int IS_NUM229 = 229;
constexpr int IS_NUM230 = 230;
constexpr int IS_NUM239 = 239;
constexpr int IS_NUM240 = 240;
constexpr int IS_NUM249 = 249;
constexpr int IS_NUM250 = 250;
constexpr int IS_NUM259 = 259;
constexpr int IS_NUM260 = 260;
constexpr int IS_NUM269 = 269;
constexpr int IS_NUM270 = 270;
constexpr int IS_NUM279 = 279;
constexpr int IS_NUM280 = 280;
constexpr int IS_NUM289 = 289;
constexpr int IS_NUM290 = 290;
constexpr int IS_NUM299 = 299;
constexpr int IS_NUM300 = 300;
constexpr int IS_NUM499 = 499;
constexpr int IS_NUM500 = 500;
constexpr int IS_NUM600 = 600;
constexpr int IS_NUM699 = 699;
constexpr int IS_NUM700 = 700;
constexpr int IS_NUM800 = 800;
constexpr int IS_NUM900 = 900;
constexpr int IS_NUM1000 = 1000;
constexpr int IS_NUM1100 = 1100;
class InsertSyncTest : public ::testing::Test {
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

    void AdjustCopyOpTileCfg(Operation& op, TileOpCfg& opcfg)
    {
        opcfg.aivCore_ = AIVCore::AIV0;
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
            opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
            opcfg.coreType_ = CoreType::AIV;
        } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
            opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
            opcfg.coreType_ = CoreType::AIV;
        }
    }

    void BuildDeps(PipeSync& ps, DataDependencySearcher& dataDependencySearcher, std::vector<Operation*>& opLogPtr,
                   std::vector<IndexOp>& synced)
    {
        for (size_t i = 0; i < opLogPtr.size(); i++) {
            auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[i]->GetOpcode());
            AdjustCopyOpTileCfg(*opLogPtr[i], opcfg);
            PipeSync::DepOp op(i, {opcfg.pipeIdStart_, opcfg.pipeIdEnd_, opcfg.coreType_, opcfg.aivCore_});
            PipeSync::DepOp& currOp = ps.depOps_.emplace_back(op);
            auto dataDependencySet = dataDependencySearcher.Find(opLogPtr[i]);
            for (auto it = dataDependencySet.rbegin(); it != dataDependencySet.rend(); it++) {
                size_t k = *it;
                PipeSync::DepOp& prevOp = ps.depOps_[k];
                if (ps.HasDataDependency(*opLogPtr[k], *opLogPtr[i])) {
                    ps.UpdateDep(currOp, prevOp);
                }
            }
            dataDependencySearcher.Insert(opLogPtr[i], i);
            ps.EnqueueOp(currOp, opLogPtr, synced);
        }
    }
};

TEST_F(InsertSyncTest, TestEnableDebug)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddParams", "TestAddParams",
                                                      rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    // Prepare the graph
    std::vector<int64_t> shape = {IS_NUM8, IS_NUM16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& copy_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    (void)copy_op1;
    auto& copy_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    (void)copy_op2;
    auto& add_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2},
                                                  {ubTensor3});
    (void)add_op;
    auto& copy_out_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void)copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);
    InsertSync syncPass;
    syncPass.SetEnableDebug(true);
    syncPass.RunOnFunction(*rootFuncPtr);
    EXPECT_TRUE(true);
}

std::vector<std::shared_ptr<LogicalTensor>> AddOpForTestFindDep(std::vector<Operation*>& opLogPtr,
                                                                std::shared_ptr<Function> currFunctionPtr)
{
    // Build graph
    std::vector<int64_t> shape1 = {IS_NUM16, IS_NUM16};
    std::vector<int64_t> shape2 = {IS_NUM8, IS_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor1->memoryrange.start = 0;
    tensor1->memoryrange.end = IS_NUM100;
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor2->memoryrange.start = 0;
    tensor2->memoryrange.end = IS_NUM200;
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor3->memoryrange.start = IS_NUM300;
    tensor3->memoryrange.end = IS_NUM499;
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor4->memoryrange.start = IS_NUM500;
    tensor4->memoryrange.end = IS_NUM699;
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor5->memoryrange.start = IS_NUM700;
    tensor5->memoryrange.end = IS_NUM900;
    auto tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    tensor6->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor6->memoryrange.start = IS_NUM150;
    tensor6->memoryrange.end = IS_NUM200;
    auto& expend = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXPAND, {tensor1}, {tensor2});
    opLogPtr.emplace_back(&expend);
    auto& copyin1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor2}, {tensor3});
    opLogPtr.emplace_back(&copyin1);
    auto& copyin2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor2}, {tensor4});
    opLogPtr.emplace_back(&copyin2);
    auto& copyin3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor2}, {tensor5});
    opLogPtr.emplace_back(&copyin3);
    auto& exp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXP, {tensor3}, {tensor6});
    opLogPtr.emplace_back(&exp);
    return {tensor1, tensor2, tensor3, tensor4, tensor5, tensor6};
}

void CheckDependencyForTestFindDep(PipeSync& ps, std::set<int> dataDependencySet, std::vector<Operation*>& opLogPtr,
                                   size_t i)
{
    for (auto it = dataDependencySet.rbegin(); it != dataDependencySet.rend(); it++) {
        size_t k = *it;
        // start tests
        if (i == IS_NUM1 && k == 0) {
            EXPECT_EQ(ps.CheckRawDependency(*opLogPtr[k], *opLogPtr[i]), true);
            EXPECT_EQ(ps.CheckWarDependency(*opLogPtr[k], *opLogPtr[i]), false);
            EXPECT_EQ(ps.CheckWawDependency(*opLogPtr[k], *opLogPtr[i]), false);
        }
        if (i == IS_NUM4 && k == 0) {
            EXPECT_EQ(ps.CheckRawDependency(*opLogPtr[k], *opLogPtr[i]), false);
            EXPECT_EQ(ps.CheckWarDependency(*opLogPtr[k], *opLogPtr[i]), false);
            EXPECT_EQ(ps.CheckWawDependency(*opLogPtr[k], *opLogPtr[i]), true);
        }
        if (i == IS_NUM4 && k == IS_NUM1) {
            EXPECT_EQ(ps.CheckRawDependency(*opLogPtr[k], *opLogPtr[i]), true);
            EXPECT_EQ(ps.CheckWarDependency(*opLogPtr[k], *opLogPtr[i]), true);
            EXPECT_EQ(ps.CheckWawDependency(*opLogPtr[k], *opLogPtr[i]), false);
        }
        // end tests
    }
}

void ProcessOpList(PipeSync& ps, DataDependencySearcher& dataDependencySearcher, std::vector<Operation*>& opLogPtr)
{
    ps.oriOpList_ = opLogPtr;
    for (auto& op : opLogPtr) {
        bool isCubeComponent = op->HasAttr(OpAttributeKey::isCube) && op->GetAttr<bool>(OpAttributeKey::isCube);
        if (!isCubeComponent) {
            op->SetAIVCore(AIVCore::AIV0);
        }
        ps.BuildTensorRangeMap(op);
    }
    dataDependencySearcher.ubTensorRangeMap = ps.ubTensorRangeMap;
}

TEST_F(InsertSyncTest, TestFindDep)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDep", "TestFindDep", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDepLeaf", "TestFindDepLeaf",
                                                      rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<Operation*> opLogPtr;
    auto tensors = AddOpForTestFindDep(opLogPtr, currFunctionPtr);
    PipeSync ps;
    DataDependencySearcher dataDependencySearcher;
    ProcessOpList(ps, dataDependencySearcher, opLogPtr);
    for (size_t i = 0; i < opLogPtr.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[i]->GetOpcode());
        AdjustCopyOpTileCfg(*opLogPtr[i], opcfg);
        PipeSync::DepOp op(i, {opcfg.pipeIdStart_, opcfg.pipeIdEnd_, opcfg.coreType_, opcfg.aivCore_});
        ps.depOps_.emplace_back(op);
        auto dataDependencySet = dataDependencySearcher.Find(opLogPtr[i]);
        // start tests
        if (i == IS_NUM1 || i == IS_NUM2 || i == IS_NUM3) {
            std::set<int> res = {0};
            EXPECT_EQ(dataDependencySet, res);
        }
        if (i == IS_NUM4) {
            std::set<int> res = {0, IS_NUM1, IS_NUM2, IS_NUM3};
            EXPECT_EQ(dataDependencySet, res);
        }
        // end tests
        CheckDependencyForTestFindDep(ps, dataDependencySet, opLogPtr, i);
        dataDependencySearcher.Insert(opLogPtr[i], i);
    }

    // test ignorable intra pipe dep
    EXPECT_EQ(ps.IgnorableIntraPipeDep(0, IS_NUM4, opLogPtr), false);
    tensors[1]->shape = {IS_NUM16, IS_NUM32, IS_NUM32};
    EXPECT_EQ(ps.IgnorableIntraPipeDep(0, IS_NUM4, opLogPtr), false);

    // test AdjustOpCfg
    auto opcfg1 = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[1]->GetOpcode());
    EXPECT_EQ(ps.AdjustOpCfg(opcfg1, *opLogPtr[1]), FAILED);

    std::vector<int64_t> shape = {IS_NUM8, IS_NUM16};
    auto tensor7 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor7->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor7->memoryrange.start = IS_NUM1000;
    tensor7->memoryrange.end = IS_NUM1100;
    auto& copyout = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensors[4]}, {tensor7});
    opLogPtr.emplace_back(&copyout);
    auto opcfg2 = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[IS_NUM1]->GetOpcode());
    EXPECT_EQ(ps.AdjustOpCfg(opcfg2, *opLogPtr[IS_NUM5]), FAILED);
}

TEST_F(InsertSyncTest, TestPhaseKernelProcess)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestPhaseKernelProcess",
                                                  "TestPhaseKernelProcess", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPhaseKernelProcessLeaf",
                                                      "TestPhaseKernelProcessLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<int64_t> shape = {IS_NUM16, IS_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    std::vector<Operation*> opLogPtr;
    auto& copyin1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    opLogPtr.emplace_back(&copyin1);
    auto& copyin2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor2}, {tensor4});
    opLogPtr.emplace_back(&copyin2);
    auto& add = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {tensor3, tensor4}, {tensor5});
    opLogPtr.emplace_back(&add);
    auto& copyout = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensor5}, {tensor6});
    opLogPtr.emplace_back(&copyout);

    PipeSync ps;
    std::vector<Operation*> resLogPtr;
    ps.PhaseKernelProcess(*currFunctionPtr, opLogPtr, resLogPtr);
    EXPECT_EQ(resLogPtr[0]->GetOpcode(), Opcode::OP_PHASE1);
    EXPECT_EQ(resLogPtr[IS_NUM3]->GetOpcode(), Opcode::OP_PHASE2);
}

void AddOpForTestUpdateDep(std::vector<Operation*>& opLogPtr, std::shared_ptr<Function> currFunctionPtr)
{
    std::vector<int64_t> shape = {IS_NUM16, IS_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor1->memoryrange.start = 0;
    tensor1->memoryrange.end = IS_NUM100;
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor2->memoryrange.start = IS_NUM200;
    tensor2->memoryrange.end = IS_NUM300;
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor3->memoryrange.start = IS_NUM200;
    tensor3->memoryrange.end = IS_NUM300;
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor4->memoryrange.start = IS_NUM200;
    tensor4->memoryrange.end = IS_NUM300;
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor5->memoryrange.start = IS_NUM500;
    tensor5->memoryrange.end = IS_NUM600;
    auto tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor6->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor6->memoryrange.start = IS_NUM500;
    tensor6->memoryrange.end = IS_NUM600;
    auto tensor7 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor7->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor7->memoryrange.start = 0;
    tensor7->memoryrange.end = IS_NUM100;
    auto tensor8 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor8->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor8->memoryrange.start = IS_NUM101;
    tensor8->memoryrange.end = IS_NUM199;
    auto& copyin1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor8});
    opLogPtr.emplace_back(&copyin1);
    auto& copyin2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor2}, {tensor3});
    opLogPtr.emplace_back(&copyin2);
    auto& cast1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor3}, {tensor4});
    opLogPtr.emplace_back(&cast1);
    auto& cast2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor5}, {tensor6});
    opLogPtr.emplace_back(&cast2);
    auto& copyout = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensor6}, {tensor7});
    opLogPtr.emplace_back(&copyout);
}

TEST_F(InsertSyncTest, TestUpdateDep)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestUpdateDep", "TestUpdateDep", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestUpdateDepLeaf", "TestUpdateDepLeaf",
                                                      rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<Operation*> opLogPtr;
    AddOpForTestUpdateDep(opLogPtr, currFunctionPtr);

    PipeSync ps;
    DataDependencySearcher dataDependencySearcher;
    ProcessOpList(ps, dataDependencySearcher, opLogPtr);
    for (size_t i = 0; i < opLogPtr.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[i]->GetOpcode());
        AdjustCopyOpTileCfg(*opLogPtr[i], opcfg);
        PipeSync::DepOp op(i, {opcfg.pipeIdStart_, opcfg.pipeIdEnd_, opcfg.coreType_, opcfg.aivCore_});
        auto& currOp = ps.depOps_.emplace_back(op);
        auto dataDependencySet = dataDependencySearcher.Find(opLogPtr[i]);
        for (auto it = dataDependencySet.rbegin(); it != dataDependencySet.rend(); it++) {
            size_t k = *it;
            auto& prevOp = ps.depOps_[k];
            if (ps.HasDataDependency(*opLogPtr[k], *opLogPtr[i])) {
                // start tests
                ps.UpdateDep(currOp, prevOp);
                PipeSync::PipeCoreRealEx pcCurr(PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
                PipeSync::PipeCoreRealEx pcSet1(PipeType::PIPE_V, CoreType::AIV, AIVCore::AIV0);
                PipeSync::PipeCoreRealEx pcSet2(PipeType::PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
                auto setPipeIdx1 = ps.latestPipeDep_[pcCurr].setPipes[pcSet1];
                auto setPipeIdx2 = ps.latestPipeDep_[pcCurr].setPipes[pcSet2];
                ps.latestPipeDep_[pcCurr].DumpPipeDepInfo();
                if (i == IS_NUM4 && k == IS_NUM3) {
                    EXPECT_EQ(setPipeIdx1, static_cast<size_t>(IS_NUM3));
                    EXPECT_EQ(setPipeIdx2, static_cast<size_t>(IS_NUM1));
                }
                if (i == IS_NUM4 && k == 0) {
                    EXPECT_EQ(setPipeIdx1, static_cast<size_t>(IS_NUM3));
                    EXPECT_EQ(setPipeIdx2, static_cast<size_t>(IS_NUM1));
                }
            }
        }
        dataDependencySearcher.Insert(opLogPtr[i], i);
    }
}

void AddOpForTestHandleEventID(std::vector<Operation*>& opLogPtr, std::shared_ptr<Function> currFunctionPtr)
{
    std::vector<int64_t> shape = {IS_NUM16, IS_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor1->memoryrange.start = 0;
    tensor1->memoryrange.end = IS_NUM99;
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor2->memoryrange.start = IS_NUM100;
    tensor2->memoryrange.end = IS_NUM199;
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor3->memoryrange.start = IS_NUM200;
    tensor3->memoryrange.end = IS_NUM300;
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor4->memoryrange.start = IS_NUM500;
    tensor4->memoryrange.end = IS_NUM600;
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor5->memoryrange.start = IS_NUM700;
    tensor5->memoryrange.end = IS_NUM800;
    auto& copyin1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor2});
    opLogPtr.emplace_back(&copyin1); // index 0
    auto& copyin2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor4}, {tensor5});
    opLogPtr.emplace_back(&copyin2); // index 1 - moved before cast to satisfy AdjustOpDep conditions
    auto& cast = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor2}, {tensor3});
    opLogPtr.emplace_back(&cast); // index 2
}

TEST_F(InsertSyncTest, TestHandleEventID)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestHandleEventID", "TestHandleEventID",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestHandleEventIDLeaf",
                                                      "TestHandleEventIDLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<Operation*> opLogPtr;
    AddOpForTestHandleEventID(opLogPtr, currFunctionPtr);

    PipeSync ps;
    ps.InitIssueQueue(); // Initialize issueState_ before BuildDeps
    DataDependencySearcher dataDependencySearcher;
    ProcessOpList(ps, dataDependencySearcher, opLogPtr);
    std::vector<IndexOp> synced;
    BuildDeps(ps, dataDependencySearcher, opLogPtr, synced);
    // After reorder: copyin1(0), copyin2(1), cast(2)
    EXPECT_EQ(ps.depOps_[0].setPipe[0], static_cast<size_t>(IS_NUM2));  // copyin1 sets for cast
    EXPECT_EQ(ps.depOps_[IS_NUM2].waitPipe[0], static_cast<size_t>(0)); // cast waits for copyin1

    // HandleEventID - Test AdjustOpDep path
    // Set up conditions to trigger AdjustOpDep:
    // 1. currIssueNum >= maxIssueNum (both set to 8)
    // 2. deadlock = true
    // 3. issueQ has enough elements for nextOp
    bool eventIdDeadlock = true;
    bool res = false;
    PipeSync::IssueNum issuenum;
    // issueState_[4] is AIV0_MTE2 queue, contains copyin1(0) and copyin2(1)
    PipeSync::IssueQueue& issueQ = ps.issueState_[IS_NUM4];
    PipeSync::DepOp& handleOp = ps.depOps_[0];    // copyin1
    PipeSync::DepOp& eleOp = ps.depOps_[IS_NUM2]; // cast
    PipeSync::PipeCoreRealEx currPipeCoreEx(handleOp.selfPipeCore.pipeEnd, handleOp.selfPipeCore.core,
                                            handleOp.selfPipeCore.aivCore);
    PipeSync::PipeCoreRealEx elePipeCoreEx(eleOp.selfPipeCore.pipeStart, eleOp.selfPipeCore.core,
                                           eleOp.selfPipeCore.aivCore);
    PipeSync::PipePairEx pp{currPipeCoreEx, elePipeCoreEx};

    // Pre-set issuenum to trigger AdjustOpDep: currIssueNum >= maxIssueNum
    issuenum.maxIssueNum[pp] = IS_NUM8;
    issuenum.currIssueNum[pp] = IS_NUM8;

    // Verify initial state before HandleEventID
    EXPECT_EQ(ps.depOps_[IS_NUM2].waitPipe[0], static_cast<size_t>(0)); // cast waits for copyin1
    EXPECT_EQ(ps.depOps_[0].setPipe[0], static_cast<size_t>(IS_NUM2));  // copyin1 sets for cast

    ps.HandleEventID(handleOp, issueQ, issuenum, eventIdDeadlock, res, synced);
    issueQ.DumpIssueQueue(ps.oriOpList_);
    ps.DumpLatestPipeDepMap();

    // After HandleEventID with AdjustOpDep:
    // RemoveOpDep: copyin1's setPipe removes cast, cast's waitPipe removes copyin1
    // AddOpDep: copyin2's setPipe adds cast, cast's waitPipe adds copyin2
    EXPECT_EQ(ps.depOps_[IS_NUM2].waitPipe[0], static_cast<size_t>(IS_NUM1)); // cast now waits for copyin2
    EXPECT_EQ(ps.depOps_[IS_NUM1].setPipe[0], static_cast<size_t>(IS_NUM2));  // copyin2 sets for cast
    EXPECT_EQ(ps.depOps_[0].setPipe.size(), static_cast<size_t>(0));          // copyin1 has no setPipe

    // InitCVEventIdQ
    ps.InitCVEventIdQ();
    PipeSync::PipeCoreRealEx pcAIC(PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED);
    PipeSync::PipeCoreRealEx pcAIV1(PIPE_V, CoreType::AIV, AIVCore::AIV1);
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIC][0].size(), static_cast<size_t>(IS_NUM16));
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIV1][0].size(), static_cast<size_t>(IS_NUM16));
}

TEST_F(InsertSyncTest, TestRelaxFakeDataDep)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestRelaxFakeDataDep",
                                                  "TestRelaxFakeDataDep", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRelaxFakeDataDepLeaf",
                                                      "TestRelaxFakeDataDepLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<int64_t> shape = {IS_NUM8, IS_NUM8};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor1->memoryrange.start = IS_NUM100;
    tensor1->memoryrange.end = IS_NUM109;
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor2->memoryrange.start = 0;
    tensor2->memoryrange.end = IS_NUM9;
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor3->memoryrange.start = IS_NUM10;
    tensor3->memoryrange.end = IS_NUM19;
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor4->memoryrange.start = IS_NUM20;
    tensor4->memoryrange.end = IS_NUM29;
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor5->memoryrange.start = IS_NUM30;
    tensor5->memoryrange.end = IS_NUM39;
    auto tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor6->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor6->memoryrange.start = IS_NUM40;
    tensor6->memoryrange.end = IS_NUM49;
    auto tensor7 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor7->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor7->memoryrange.start = IS_NUM50;
    tensor7->memoryrange.end = IS_NUM59;
    auto tensor8 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor8->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor8->memoryrange.start = IS_NUM60;
    tensor8->memoryrange.end = IS_NUM69;
    auto tensor9 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor9->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor9->memoryrange.start = IS_NUM70;
    tensor9->memoryrange.end = IS_NUM79;
    auto tensor10 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor10->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor10->memoryrange.start = IS_NUM80;
    tensor10->memoryrange.end = IS_NUM89;
    auto tensor11 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor11->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor11->memoryrange.start = IS_NUM90;
    tensor11->memoryrange.end = IS_NUM99;
    auto tensor12 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor12->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor12->memoryrange.start = IS_NUM200;
    tensor12->memoryrange.end = IS_NUM209;
    auto tensor13 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor13->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor13->memoryrange.start = IS_NUM210;
    tensor13->memoryrange.end = IS_NUM219;
    auto tensor14 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor14->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor14->memoryrange.start = IS_NUM220;
    tensor14->memoryrange.end = IS_NUM229;
    auto tensor15 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor15->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor15->memoryrange.start = IS_NUM230;
    tensor15->memoryrange.end = IS_NUM239;
    auto tensor16 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor16->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor16->memoryrange.start = IS_NUM240;
    tensor16->memoryrange.end = IS_NUM249;
    auto tensor17 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor17->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor17->memoryrange.start = IS_NUM250;
    tensor17->memoryrange.end = IS_NUM259;
    auto tensor18 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor18->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor18->memoryrange.start = IS_NUM260;
    tensor18->memoryrange.end = IS_NUM269;
    auto tensor19 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor19->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor19->memoryrange.start = IS_NUM270;
    tensor19->memoryrange.end = IS_NUM279;
    auto tensor20 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor20->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor20->memoryrange.start = IS_NUM280;
    tensor20->memoryrange.end = IS_NUM289;
    auto tensor21 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor21->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor21->memoryrange.start = IS_NUM290;
    tensor21->memoryrange.end = IS_NUM299;
    auto tensor22 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor22->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor22->memoryrange.start = IS_NUM110;
    tensor22->memoryrange.end = IS_NUM119;
    auto tensor23 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor23->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor23->memoryrange.start = IS_NUM120;
    tensor23->memoryrange.end = IS_NUM129;
    auto tensor24 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor24->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor24->memoryrange.start = IS_NUM130;
    tensor24->memoryrange.end = IS_NUM139;
    auto tensor25 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor25->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor25->memoryrange.start = IS_NUM140;
    tensor25->memoryrange.end = IS_NUM149;
    auto tensor26 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor26->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor26->memoryrange.start = IS_NUM150;
    tensor26->memoryrange.end = IS_NUM159;
    auto tensor27 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor27->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor27->memoryrange.start = IS_NUM160;
    tensor27->memoryrange.end = IS_NUM169;
    auto tensor28 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor28->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor28->memoryrange.start = IS_NUM170;
    tensor28->memoryrange.end = IS_NUM179;
    auto tensor29 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor29->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor29->memoryrange.start = IS_NUM180;
    tensor29->memoryrange.end = IS_NUM189;
    std::vector<Operation*> opLogPtr;
    auto& copyin1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor2});
    opLogPtr.emplace_back(&copyin1);
    auto& copyin2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    opLogPtr.emplace_back(&copyin2);
    auto& copyin3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor4});
    opLogPtr.emplace_back(&copyin3);
    auto& copyin4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor5});
    opLogPtr.emplace_back(&copyin4);
    auto& copyin5 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor6});
    opLogPtr.emplace_back(&copyin5);
    auto& copyin6 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor7});
    opLogPtr.emplace_back(&copyin6);
    auto& copyin7 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor8});
    opLogPtr.emplace_back(&copyin7);
    auto& copyin8 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor9});
    opLogPtr.emplace_back(&copyin8);
    auto& copyin9 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor10});
    opLogPtr.emplace_back(&copyin9);
    auto& copyin10 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor11});
    opLogPtr.emplace_back(&copyin10);
    auto& cast1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor22});
    opLogPtr.emplace_back(&cast1);
    auto& cast2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor23});
    opLogPtr.emplace_back(&cast2);
    auto& cast3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor24});
    opLogPtr.emplace_back(&cast3);
    auto& cast4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor25});
    opLogPtr.emplace_back(&cast4);
    auto& cast5 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor26});
    opLogPtr.emplace_back(&cast5);
    auto& cast6 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor27});
    opLogPtr.emplace_back(&cast6);
    auto& cast7 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor28});
    opLogPtr.emplace_back(&cast7);
    auto& cast8 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor29});
    opLogPtr.emplace_back(&cast8);
    auto& cast9 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor2}, {tensor12});
    opLogPtr.emplace_back(&cast9);
    auto& cast10 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor3}, {tensor13});
    opLogPtr.emplace_back(&cast10);
    auto& cast11 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor4}, {tensor14});
    opLogPtr.emplace_back(&cast11);
    auto& cast12 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor5}, {tensor15});
    opLogPtr.emplace_back(&cast12);
    auto& cast13 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor6}, {tensor16});
    opLogPtr.emplace_back(&cast13);
    auto& cast14 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor7}, {tensor17});
    opLogPtr.emplace_back(&cast14);
    auto& cast15 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor8}, {tensor18});
    opLogPtr.emplace_back(&cast15);
    auto& cast16 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor9}, {tensor19});
    opLogPtr.emplace_back(&cast16);
    auto& cast17 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor10}, {tensor20});
    opLogPtr.emplace_back(&cast17);
    auto& cast18 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor11}, {tensor21});
    opLogPtr.emplace_back(&cast18);

    PipeSync ps;
    DataDependencySearcher dataDependencySearcher;
    ProcessOpList(ps, dataDependencySearcher, opLogPtr);
    std::vector<IndexOp> synced;
    size_t index = UINT64_MAX;
    EXPECT_EQ(ps.InjectSync(*currFunctionPtr, opLogPtr, index, synced), FAILED);
    BuildDeps(ps, dataDependencySearcher, opLogPtr, synced);

    // Issue op
    size_t totalIssued = 0;
    size_t allIssued = 0;
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        allIssued += ps.issueState_[i].ops.size();
    }
    bool eventIdDeadlock = false;
    uint64_t eventIdDeadlockEnterTimes = 0;

    while (totalIssued < allIssued) {
        size_t issued = 0;
        size_t issuedTest = IS_NUM100;
        for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
            std::vector<size_t> issuedOps;
            ps.PopFromQueue(ps.issueState_[i], issuedOps, eventIdDeadlock, synced);
            issued += issuedOps.size();
            if (i == IS_NUM4) {
                issuedTest = issuedOps.size();
            }
            for (auto idx : issuedOps) {
                ps.InjectSync(*currFunctionPtr, opLogPtr, idx, synced);
            }
            if (issuedTest == static_cast<size_t>(0)) {
                break;
            }
        }
        totalIssued += issued;
        // eventIdDeadlockEnterTimes eventIdDeadlock syncedOpLog
        if (issuedTest == static_cast<size_t>(0)) {
            EXPECT_EQ(ps.depOps_[0].setPipe[0], static_cast<size_t>(IS_NUM18));
            EXPECT_EQ(ps.depOps_[IS_NUM1].setPipe[0], static_cast<size_t>(IS_NUM19));
            ps.ProcessDeadLock(eventIdDeadlockEnterTimes, eventIdDeadlock, synced);
            EXPECT_EQ(ps.depOps_[IS_NUM1].setPipe[0], static_cast<size_t>(IS_NUM18));
            continue;
        }
        eventIdDeadlock = false;
        eventIdDeadlockEnterTimes = static_cast<size_t>(0);
        break;
    }
}

TEST_F(InsertSyncTest, TestGetDepInfoSizeMismatch)
{
    PipeSync ps;
    std::vector<IndexOp> emptySyncedOpLog;
    auto pipePair = PipeSync::dataDepPair[0];
    PipeSync::DataDepInfo depInfo;
    EXPECT_EQ(ps.GetDepInfo(emptySyncedOpLog, pipePair, depInfo), FAILED);
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV0_MTE2), "AIV0_MTE2");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV1_MTE2), "AIV1_MTE2");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV0_V), "AIV0_V");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV1_V), "AIV1_V");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV0_MTE3), "AIV0_MTE3");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV1_MTE3), "AIV1_MTE3");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV0_S), "AIV0_S");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV1_S), "AIV1_S");
}

TEST_F(InsertSyncTest, TestRecycleCrossCoreEventIds)
{
    PipeSync ps;
    ps.InitCVEventIdQ();

    // 场景1: currPipeRealEx 不在 crossCoreFreeEventId_ 中 -> 返回 FAILED
    PipeSync::PipeCoreRealEx notExistPipe(PIPE_V, CoreType::AIV, AIVCore::UNSPECIFIED);
    EXPECT_EQ(ps.RecycleCrossCoreEventIds(notExistPipe), FAILED);

    using CoreDetail = PipeSync::CoreTypeDetail;
    auto makeEr = [](int id, CoreDetail src, CoreDetail dst) {
        return PipeSync::EventResource{id, src, dst, PIPE_S, PIPE_V};
    };
    CoreDetail aicCore{CoreType::AIC, AIVCore::UNSPECIFIED};
    CoreDetail aiv0Core{CoreType::AIV, AIVCore::AIV0};
    CoreDetail aiv1Core{CoreType::AIV, AIVCore::AIV1};

    // 场景2: AIC core - 覆盖 srcCore==dstCore(continue) + eventId<16(queue0) + eventId>=16(queue1)
    PipeSync::PipeCoreRealEx pcAIC(PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED);
    ps.syncArriveStatus[pcAIC].clear();
    ps.crossCoreFreeEventId_[pcAIC][0].clear();
    ps.crossCoreFreeEventId_[pcAIC][1].clear();
    ps.syncArriveStatus[pcAIC].insert(makeEr(IS_NUM5, aicCore, aicCore));   // srcCore==dstCore -> continue
    ps.syncArriveStatus[pcAIC].insert(makeEr(IS_NUM3, aicCore, aiv0Core));  // eventId<16 -> queue0
    ps.syncArriveStatus[pcAIC].insert(makeEr(IS_NUM20, aicCore, aiv1Core)); // eventId>=16 -> queue1
    EXPECT_EQ(ps.RecycleCrossCoreEventIds(pcAIC), SUCCESS);
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIC][0].size(), static_cast<size_t>(IS_NUM1));
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIC][0].front(), IS_NUM3);
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIC][1].size(), static_cast<size_t>(IS_NUM1));
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIC][1].front(), IS_NUM20);

    // 场景3: AIV0 core - 覆盖 eventId<16(queue0) + eventId>=16(跳过)
    PipeSync::PipeCoreRealEx pcAIV0(PIPE_S, CoreType::AIV, AIVCore::AIV0);
    ps.syncArriveStatus[pcAIV0].clear();
    ps.crossCoreFreeEventId_[pcAIV0][0].clear();
    ps.syncArriveStatus[pcAIV0].insert(makeEr(IS_NUM9, aicCore, aiv0Core));  // eventId<16 -> queue0
    ps.syncArriveStatus[pcAIV0].insert(makeEr(IS_NUM29, aicCore, aiv0Core)); // eventId>=16 -> 跳过
    EXPECT_EQ(ps.RecycleCrossCoreEventIds(pcAIV0), SUCCESS);
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIV0][0].size(), static_cast<size_t>(IS_NUM1));
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIV0][0].front(), IS_NUM9);

    // 场景4: AIV1 core - 覆盖 eventId>=16(queue0) + eventId<16(跳过)
    PipeSync::PipeCoreRealEx pcAIV1(PIPE_S, CoreType::AIV, AIVCore::AIV1);
    ps.syncArriveStatus[pcAIV1].clear();
    ps.crossCoreFreeEventId_[pcAIV1][0].clear();
    ps.syncArriveStatus[pcAIV1].insert(makeEr(IS_NUM30, aicCore, aiv1Core)); // eventId>=16 -> queue0
    ps.syncArriveStatus[pcAIV1].insert(makeEr(IS_NUM10, aicCore, aiv1Core)); // eventId<16 -> 跳过
    EXPECT_EQ(ps.RecycleCrossCoreEventIds(pcAIV1), SUCCESS);
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIV1][0].size(), static_cast<size_t>(IS_NUM1));
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIV1][0].front(), IS_NUM30);
}

// Helper: create root + leaf Function pair for enableDebug tests
static std::pair<std::shared_ptr<Function>, std::shared_ptr<Function>> SetupDebugFunc(const std::string& name)
{
    auto root = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
    root->rootFunc_ = root.get();
    auto leaf = std::make_shared<Function>(Program::GetInstance(), name + "Leaf", name + "Leaf", root.get());
    root->rootFunc_->programs_.emplace(leaf->GetFuncMagic(), leaf.get());
    return {root, leaf};
}

// Helper: batch-create N FP32 tensors with given shape
static std::vector<std::shared_ptr<LogicalTensor>> MakeTensors(int n, const std::vector<int64_t>& shape)
{
    std::vector<std::shared_ptr<LogicalTensor>> ts;
    for (int i = 0; i < n; i++) {
        ts.push_back(npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape)));
    }
    return ts;
}

// Helper: verify one sync op's opcode and AIVCore
static void VerifySyncOp(const Operation* op, Opcode expectedOp, AIVCore expectedCore)
{
    EXPECT_EQ(op->GetOpcode(), expectedOp);
    EXPECT_EQ(op->GetAIVCore(), expectedCore);
}

// Helper: verify BAR_ALL inserted between same-coreType ops
static void VerifyBarAll(const std::vector<Operation*>& ops, size_t idx)
{
    VerifySyncOp(ops[idx], Opcode::OP_BAR_ALL, AIVCore::UNSPECIFIED);
}

// Helper: verify full InsertCvSyncOps 12-op sequence, return next index
static size_t VerifyCvSyncOps(const std::vector<Operation*>& ops, size_t i)
{
    // UNSPECIFIED group (ops 1-4)
    VerifySyncOp(ops[i], Opcode::OP_BAR_ALL, AIVCore::UNSPECIFIED);
    VerifySyncOp(ops[i + IS_NUM1], Opcode::OP_FFTS_CROSS_CORE_SYNC, AIVCore::UNSPECIFIED);
    VerifySyncOp(ops[i + IS_NUM2], Opcode::OP_WAIT_FLAG_DEV, AIVCore::UNSPECIFIED);
    VerifySyncOp(ops[i + IS_NUM3], Opcode::OP_BAR_ALL, AIVCore::UNSPECIFIED);
    // AIV0 group (ops 5-8)
    VerifySyncOp(ops[i + IS_NUM4], Opcode::OP_BAR_ALL, AIVCore::AIV0);
    VerifySyncOp(ops[i + IS_NUM5], Opcode::OP_WAIT_FLAG_DEV, AIVCore::AIV0);
    VerifySyncOp(ops[i + IS_NUM6], Opcode::OP_FFTS_CROSS_CORE_SYNC, AIVCore::AIV0);
    VerifySyncOp(ops[i + IS_NUM7], Opcode::OP_BAR_ALL, AIVCore::AIV0);
    // AIV1 group (ops 9-12)
    VerifySyncOp(ops[i + IS_NUM8], Opcode::OP_BAR_ALL, AIVCore::AIV1);
    VerifySyncOp(ops[i + IS_NUM9], Opcode::OP_WAIT_FLAG_DEV, AIVCore::AIV1);
    VerifySyncOp(ops[i + IS_NUM10], Opcode::OP_FFTS_CROSS_CORE_SYNC, AIVCore::AIV1);
    VerifySyncOp(ops[i + IS_NUM11], Opcode::OP_BAR_ALL, AIVCore::AIV1);
    return i + IS_NUM12;
}

// 覆盖 InsertCvPipeAll 全部 3 条分支：
//   ADD(AIV)→CAST(AIV): 同coreType → BAR_ALL
//   CAST(AIV)→A_MUL_B(AIC): 异coreType → InsertCvSyncOps
//   A_MUL_B(AIC)→EXP(AIV): 异coreType → InsertCvSyncOps
//   EXP 是最后一个 op → skip
TEST_F(InsertSyncTest, TestEnableDebugAllPaths)
{
    auto [root, leaf] = SetupDebugFunc("TestDbgAll");
    auto ts = MakeTensors(IS_NUM7, {IS_NUM8, IS_NUM16});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_A_MUL_B, {ts[3], ts[4], ts[5]}, {ts[6]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[6]}, {ts[0]});
    InsertSync syncPass;
    syncPass.SetEnableDebug(true);
    syncPass.RunOnFunction(*root);
    // 验证 oriOpList
    EXPECT_EQ(leaf->oriOpList.size(), static_cast<size_t>(IS_NUM4));
    // 验证新 op 列表: 4原始 + 1 BAR_ALL + 12 CvSync + 12 CvSync = 29
    auto ops = leaf->Operations(false).DuplicatedOpList();
    EXPECT_EQ(ops.size(), static_cast<size_t>(IS_NUM29));
    size_t idx = 0;
    EXPECT_EQ(ops[idx++]->GetOpcode(), Opcode::OP_ADD);
    VerifyBarAll(ops, idx++);
    EXPECT_EQ(ops[idx++]->GetOpcode(), Opcode::OP_CAST);
    idx = VerifyCvSyncOps(ops, idx);
    EXPECT_EQ(ops[idx++]->GetOpcode(), Opcode::OP_A_MUL_B);
    idx = VerifyCvSyncOps(ops, idx);
    EXPECT_EQ(ops[idx++]->GetOpcode(), Opcode::OP_EXP);
    EXPECT_EQ(idx, static_cast<size_t>(IS_NUM29));
}

// 边界: 单 op 时 i==oriOpList.size()-1 直接 skip，不插入任何 sync op
TEST_F(InsertSyncTest, TestEnableDebugSingleOp)
{
    auto [root, leaf] = SetupDebugFunc("TestDbgSingle");
    auto ts = MakeTensors(IS_NUM2, {IS_NUM8, IS_NUM16});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0]}, {ts[1]});
    InsertSync syncPass;
    syncPass.SetEnableDebug(true);
    syncPass.RunOnFunction(*root);
    EXPECT_EQ(leaf->oriOpList.size(), static_cast<size_t>(IS_NUM1));
    auto ops = leaf->Operations(false).DuplicatedOpList();
    EXPECT_EQ(ops.size(), static_cast<size_t>(IS_NUM1));
    EXPECT_EQ(ops[0]->GetOpcode(), Opcode::OP_ADD);
}

// 全同 coreType(AIV): 每对相邻 op 之间都插入 BAR_ALL
TEST_F(InsertSyncTest, TestEnableDebugAllSameCoreType)
{
    auto [root, leaf] = SetupDebugFunc("TestDbgSame");
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[3]}, {ts[0]});
    InsertSync syncPass;
    syncPass.SetEnableDebug(true);
    syncPass.RunOnFunction(*root);
    EXPECT_EQ(leaf->oriOpList.size(), static_cast<size_t>(IS_NUM3));
    // 3 原始 + 2 BAR_ALL = 5
    auto ops = leaf->Operations(false).DuplicatedOpList();
    EXPECT_EQ(ops.size(), static_cast<size_t>(IS_NUM5));
    EXPECT_EQ(ops[0]->GetOpcode(), Opcode::OP_ADD);
    VerifyBarAll(ops, IS_NUM1);
    EXPECT_EQ(ops[IS_NUM2]->GetOpcode(), Opcode::OP_CAST);
    VerifyBarAll(ops, IS_NUM3);
    EXPECT_EQ(ops[IS_NUM4]->GetOpcode(), Opcode::OP_EXP);
}

// 全同 coreType(AIC): AIC→AIC 同核心场景，BAR_ALL 的 AIVCore 取自 nextOp(AIC 算子)，其值为 UNSPECIFIED
TEST_F(InsertSyncTest, TestEnableDebugAllAic)
{
    auto [root, leaf] = SetupDebugFunc("TestDbgAic");
    auto ts = MakeTensors(IS_NUM10, {IS_NUM8, IS_NUM16});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_A_MUL_B, {ts[0], ts[1], ts[2]}, {ts[3]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_A_MULACC_B, {ts[3], ts[4], ts[5]}, {ts[6]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_A_MUL_BT, {ts[6], ts[7], ts[8]}, {ts[9]});
    InsertSync syncPass;
    syncPass.SetEnableDebug(true);
    syncPass.RunOnFunction(*root);
    EXPECT_EQ(leaf->oriOpList.size(), static_cast<size_t>(IS_NUM3));
    // 3 原始 + 2 BAR_ALL = 5
    auto ops = leaf->Operations(false).DuplicatedOpList();
    EXPECT_EQ(ops.size(), static_cast<size_t>(IS_NUM5));
    EXPECT_EQ(ops[0]->GetOpcode(), Opcode::OP_A_MUL_B);
    VerifyBarAll(ops, IS_NUM1);
    EXPECT_EQ(ops[IS_NUM2]->GetOpcode(), Opcode::OP_A_MULACC_B);
    VerifyBarAll(ops, IS_NUM3);
    EXPECT_EQ(ops[IS_NUM4]->GetOpcode(), Opcode::OP_A_MUL_BT);
}

} // namespace tile_fwk
} // namespace npu

#undef private
