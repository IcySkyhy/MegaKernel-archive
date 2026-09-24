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
 * \file test_intra_subgraph_adapter.cpp
 * \brief Unit test for IntraSubgraphAdapter.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/data_path/intra_subgraph_adapter.h"

namespace npu::tile_fwk {
class IntraSubgraphAdapterTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    }

    void TearDown() override {}
};

TEST_F(IntraSubgraphAdapterTest, TestBoundaryConvert)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_L1,
                                           MemoryType::MEM_L0A};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_CONVERT, Opcode::OP_L1_TO_L0A};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"adds", "convert", "L1ToL0A"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    subGraph.GetOp("adds")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->SetOpAttribute(
        std::make_shared<ConvertOpAttribute>(MemoryType::MEM_UB, MemoryType::MEM_L1));
    subGraph.GetOp("L1ToL0A")->UpdateSubgraphID(1);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    IntraSubgraphAdapter adapter;
    EXPECT_EQ(adapter.PostCheck(*function), FAILED);
    function->SetTotalSubGraphCount(2);
    adapter.RunOnFunction(*function);
    EXPECT_EQ(adapter.PostCheck(*function), SUCCESS);
    const int opNum = 4;
    EXPECT_EQ(function->Operations().DuplicatedOpList().size(), opNum);
    const int copyOutIdx = 1;
    EXPECT_EQ(function->Operations().DuplicatedOpList()[copyOutIdx]->GetOpcode(), Opcode::OP_COPY_OUT);
    const int viewIdx = 2;
    EXPECT_EQ(function->Operations().DuplicatedOpList()[viewIdx]->GetOpcode(), Opcode::OP_SLICE);
    auto copyOpAttr = dynamic_cast<CopyOpAttribute*>(subGraph.GetOp("convert")->GetOpAttribute().get());
    EXPECT_NE(copyOpAttr, nullptr);
}

TEST_F(IntraSubgraphAdapterTest, TestBoundaryConvertFailed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_L1,
                                           MemoryType::MEM_L0B};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_CONVERT, Opcode::OP_L1_TO_L0B};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"adds", "convert", "L1ToL0B"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    subGraph.GetOp("adds")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->UpdateSubgraphID(1);
    subGraph.GetOp("convert")->SetOpAttribute(
        std::make_shared<ConvertOpAttribute>(MemoryType::MEM_UB, MemoryType::MEM_L1));
    subGraph.GetOp("L1ToL0B")->UpdateSubgraphID(1);
    Platform::Instance().GetDie().SetMemoryPath({{MemoryType::MEM_L1, MemoryType::MEM_DEVICE_DDR}});
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    IntraSubgraphAdapter adapter;
    function->SetTotalSubGraphCount(2);
    EXPECT_EQ(adapter.RunOnFunction(*function), FAILED);
    EXPECT_EQ(adapter.PostCheck(*function), FAILED);
    Platform::Instance().ReloadMemoryPaths("3510");
}

TEST_F(IntraSubgraphAdapterTest, TestInnerConvert)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_L1,
                                           MemoryType::MEM_L0A};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_CONVERT, Opcode::OP_L1_TO_L0A};
    std::vector<std::vector<std::string>> ioperands{{"t0"}, {"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"adds", "convert", "L1ToL0A"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    subGraph.GetOp("adds")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->SetOpAttribute(
        std::make_shared<ConvertOpAttribute>(MemoryType::MEM_UB, MemoryType::MEM_L1));
    subGraph.GetOp("L1ToL0A")->UpdateSubgraphID(0);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    IntraSubgraphAdapter adapter;
    adapter.RunOnFunction(*function);
    const int opNum = 3;
    EXPECT_EQ(function->Operations().DuplicatedOpList().size(), opNum);
    const int convertIdx = 1;
    EXPECT_EQ(function->Operations().DuplicatedOpList()[convertIdx]->GetOpcode(), Opcode::OP_CONVERT);
}

TEST_F(IntraSubgraphAdapterTest, TestInheritScopeInfo)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);

    std::vector<Opcode> opCodes{Opcode::OP_SLICE,    Opcode::OP_MULS, Opcode::OP_ADDS,
                                Opcode::OP_CONTRACT, Opcode::OP_ADDS, Opcode::OP_CONTRACT};
    std::vector<std::vector<std::string>> ioperands{{"t0"}, {"t1"}, {"t2"}, {"t3"}, {"t2"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"view_in", "muls", "adds1", "asm_out1", "adds2", "asm_out2"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    std::vector<int64_t> offset{0, 0};
    subGraph.GetOp("view_in")->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_UB));
    subGraph.GetOp("asm_out1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset));
    subGraph.GetOp("asm_out2")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset));
    EXPECT_EQ(subGraph.SetInCast({"t0"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"t4", "t6"}), true);

    const int scopeId = 20001;
    const int cvFuseId = 0;
    std::vector<int> subgraphIds{0, 0, 1, 1, 2, 2};
    for (size_t i = 0; i < opNames.size(); i++) {
        Operation* op = subGraph.GetOp(opNames[i]);
        op->UpdateSubgraphID(subgraphIds[i]);
        op->SetScopeId(scopeId);
        op->scopeInfo_.SetCvFuseId(cvFuseId);
    }

    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetTotalSubGraphCount(3);

    IntraSubgraphAdapter adapter;
    EXPECT_EQ(adapter.RunOnFunction(*function), SUCCESS);

    int contractCnt = 0;
    int sliceCnt = 0;
    for (const auto& op : function->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_CONTRACT || op->GetOpcode() == Opcode::OP_SLICE) {
            EXPECT_EQ(op->GetScopeId(), scopeId);
            EXPECT_EQ(op->GetCvFuseId(), cvFuseId);
        }
        if (op->GetOpcode() == Opcode::OP_CONTRACT) {
            contractCnt++;
        }
        if (op->GetOpcode() == Opcode::OP_SLICE) {
            sliceCnt++;
        }
    }
    EXPECT_EQ(contractCnt, 3);
    EXPECT_EQ(sliceCnt, 3);
}

TEST_F(IntraSubgraphAdapterTest, TestValidShapeInfer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_L1,
                                           MemoryType::MEM_L0A};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_CONVERT, Opcode::OP_L1_TO_L0A};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"adds", "convert", "L1ToL0A"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {32, 32}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    auto t2 = subGraph.GetTensor("t2");
    t2->UpdateDynValidShape({IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(32)});

    subGraph.GetOp("adds")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->SetOpAttribute(
        std::make_shared<ConvertOpAttribute>(MemoryType::MEM_UB, MemoryType::MEM_L1));
    subGraph.GetOp("L1ToL0A")->UpdateSubgraphID(1);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetTotalSubGraphCount(2);

    IntraSubgraphAdapter adapter;
    EXPECT_EQ(adapter.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(adapter.PostCheck(*function), SUCCESS);

    bool foundAssembleOrView = false;
    for (const auto& op : function->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_CONTRACT || op->GetOpcode() == Opcode::OP_SLICE) {
            foundAssembleOrView = true;
            auto oOperand = op->GetOOperands().front();
            auto dynValidShape = oOperand->GetDynValidShape();
            EXPECT_FALSE(dynValidShape.empty());
            EXPECT_EQ(dynValidShape.size(), 2);
        }
    }
    EXPECT_TRUE(foundAssembleOrView);
}

TEST_F(IntraSubgraphAdapterTest, BoundaryViewShouldChangeToSlice)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1,
                                           MemoryType::MEM_L1};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_VIEW, Opcode::OP_EXP};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"adds", "view", "exp"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {32, 32}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(subGraph.SetInCast({"t1"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"t4"}), true);

    subGraph.GetOp("adds")->UpdateSubgraphID(0);
    subGraph.GetOp("view")->UpdateSubgraphID(1);
    subGraph.GetOp("exp")->UpdateSubgraphID(1);
    subGraph.GetOp("view")->SetOpAttribute(
        std::make_shared<ViewOpAttribute>(std::vector<int64_t>{32, 32}, MemoryType::MEM_L1));
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetTotalSubGraphCount(2);
    Operation* viewOp = subGraph.GetOp("view");

    IntraSubgraphAdapter adapter;
    EXPECT_EQ(adapter.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(adapter.PostCheck(*function), SUCCESS);

    int sliceNum = 0;
    int viewNum = 0;
    Operation* sliceOp = nullptr;
    for (const auto& op : function->Operations(false).DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_SLICE) {
            ++sliceNum;
            sliceOp = op;
        }
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
    }
    EXPECT_EQ(sliceNum, 1);
    EXPECT_EQ(viewNum, 0);
    ASSERT_NE(sliceOp, nullptr);
    EXPECT_EQ(function->Operations(false).DuplicatedOpList().size(), 4);
    EXPECT_EQ(sliceOp, viewOp);
    EXPECT_EQ(sliceOp->GetIOperands().front(), subGraph.GetTensor("t2"));
    EXPECT_EQ(sliceOp->GetOOperands().front(), subGraph.GetTensor("t3"));
}

TEST_F(IntraSubgraphAdapterTest, BoundaryAssembleShouldChangeToContract)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1};
    std::vector<Opcode> opCodes{Opcode::OP_ASSEMBLE, Opcode::OP_EXP};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}};
    std::vector<std::string> opNames{"assemble", "exp"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {32, 32}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(subGraph.SetInCast({"t1"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"t3"}), true);

    Operation* assembleOp = subGraph.GetOp("assemble");
    assembleOp->UpdateSubgraphID(0);
    assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 0}));
    subGraph.GetOp("exp")->UpdateSubgraphID(1);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetTotalSubGraphCount(2);

    IntraSubgraphAdapter adapter;
    EXPECT_EQ(adapter.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(adapter.PostCheck(*function), SUCCESS);

    EXPECT_EQ(function->Operations(false).DuplicatedOpList().size(), 3);
    EXPECT_EQ(assembleOp->GetOpcode(), Opcode::OP_CONTRACT);
    EXPECT_EQ(assembleOp->GetIOperands().front(), subGraph.GetTensor("t1"));
    EXPECT_EQ(assembleOp->GetOOperands().front(), subGraph.GetTensor("t2"));
    EXPECT_EQ(subGraph.GetTensor("t2")->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
}

TEST_F(IntraSubgraphAdapterTest, BoundaryViewShouldKeepViewWhenDisableSlice)
{
    config::SetPassOption(ENABLE_SLICE, false);
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1,
                                           MemoryType::MEM_L1};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_VIEW, Opcode::OP_EXP};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"adds", "view", "exp"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {32, 32}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(subGraph.SetInCast({"t1"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"t4"}), true);

    subGraph.GetOp("adds")->UpdateSubgraphID(0);
    subGraph.GetOp("view")->UpdateSubgraphID(1);
    subGraph.GetOp("exp")->UpdateSubgraphID(1);
    subGraph.GetOp("view")->SetOpAttribute(
        std::make_shared<ViewOpAttribute>(std::vector<int64_t>{32, 32}, MemoryType::MEM_L1));
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetTotalSubGraphCount(2);
    Operation* viewOp = subGraph.GetOp("view");

    IntraSubgraphAdapter adapter;
    EXPECT_EQ(adapter.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(adapter.PostCheck(*function), SUCCESS);

    int sliceNum = 0;
    int contractNum = 0;
    int viewNum = 0;
    int assembleNum = 0;
    for (const auto& op : function->Operations(false).DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_SLICE) {
            ++sliceNum;
        }
        if (op->GetOpcode() == Opcode::OP_CONTRACT) {
            ++contractNum;
        }
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
        if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(sliceNum, 0);
    EXPECT_EQ(contractNum, 0);
    EXPECT_GT(viewNum, 0);
    EXPECT_GT(assembleNum, 0);
    EXPECT_EQ(viewOp->GetOpcode(), Opcode::OP_VIEW);
    EXPECT_EQ(subGraph.GetTensor("t2")->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
}

TEST_F(IntraSubgraphAdapterTest, BoundaryAssembleShouldKeepAssembleWhenDisableSlice)
{
    config::SetPassOption(ENABLE_SLICE, false);
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1};
    std::vector<Opcode> opCodes{Opcode::OP_ASSEMBLE, Opcode::OP_EXP};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}};
    std::vector<std::string> opNames{"assemble", "exp"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {32, 32}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(subGraph.SetInCast({"t1"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"t3"}), true);

    Operation* assembleOp = subGraph.GetOp("assemble");
    assembleOp->UpdateSubgraphID(0);
    assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 0}));
    subGraph.GetOp("exp")->UpdateSubgraphID(1);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    function->SetTotalSubGraphCount(2);

    IntraSubgraphAdapter adapter;
    EXPECT_EQ(adapter.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(adapter.PostCheck(*function), SUCCESS);

    int sliceNum = 0;
    int contractNum = 0;
    int viewNum = 0;
    for (const auto& op : function->Operations(false).DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_SLICE) {
            ++sliceNum;
        }
        if (op->GetOpcode() == Opcode::OP_CONTRACT) {
            ++contractNum;
        }
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
    }
    EXPECT_EQ(sliceNum, 0);
    EXPECT_EQ(contractNum, 0);
    EXPECT_GT(viewNum, 0);
    EXPECT_EQ(assembleOp->GetOpcode(), Opcode::OP_ASSEMBLE);
    EXPECT_EQ(assembleOp->GetOOperands().front(), subGraph.GetTensor("t2"));
    EXPECT_EQ(subGraph.GetTensor("t2")->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
}

TEST_F(IntraSubgraphAdapterTest, TestL1BoundaryNoDirectPathToDDR)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_L1, MemoryType::MEM_L0A,
                                           MemoryType::MEM_L0A, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);

    std::vector<Opcode> opCodes{Opcode::OP_CONTRACT, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t0"}, {"t1"}, {"t1"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"asm", "L1ToL0A_inner", "L1ToL0A_outer", "copyOut"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    std::vector<int64_t> offset{0, 0};
    subGraph.GetOp("asm")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset));

    subGraph.GetOp("asm")->UpdateSubgraphID(0);
    subGraph.GetOp("L1ToL0A_inner")->UpdateSubgraphID(0);
    subGraph.GetOp("L1ToL0A_outer")->UpdateSubgraphID(1);
    subGraph.GetOp("copyOut")->UpdateSubgraphID(1);

    Platform::Instance().ReloadMemoryPaths("");
    Function* function = subGraph.GetFunction();
    function->SetTotalSubGraphCount(2);

    IntraSubgraphAdapter adapter;
    EXPECT_EQ(adapter.RunOnFunction(*function), SUCCESS);

    EXPECT_EQ(subGraph.GetTensor("t1")->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);

    int sliceCnt = 0;
    for (const auto& op : function->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_SLICE) {
            sliceCnt++;
        }
    }
    EXPECT_GE(sliceCnt, 1);

    Platform::Instance().ReloadMemoryPaths("3510");
}

TEST_F(IntraSubgraphAdapterTest, TestL1BoundaryNoDirectPathToDDRFailed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_L1, MemoryType::MEM_L0A,
                                           MemoryType::MEM_UB};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);

    std::vector<Opcode> opCodes{Opcode::OP_CONVERT, Opcode::OP_L1_TO_L0A, Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"t0"}, {"t1"}, {"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"convert_in", "L1ToL0A_inner", "convert_out"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    subGraph.GetOp("convert_in")->UpdateSubgraphID(0);
    subGraph.GetOp("convert_in")
        ->SetOpAttribute(std::make_shared<ConvertOpAttribute>(MemoryType::MEM_UB, MemoryType::MEM_L1));
    subGraph.GetOp("L1ToL0A_inner")->UpdateSubgraphID(0);
    subGraph.GetOp("convert_out")->UpdateSubgraphID(1);
    subGraph.GetOp("convert_out")
        ->SetOpAttribute(std::make_shared<ConvertOpAttribute>(MemoryType::MEM_L1, MemoryType::MEM_UB));

    Platform::Instance().ReloadMemoryPaths("");
    Function* function = subGraph.GetFunction();
    function->SetTotalSubGraphCount(2);

    IntraSubgraphAdapter adapter;
    EXPECT_EQ(adapter.RunOnFunction(*function), FAILED);

    Platform::Instance().ReloadMemoryPaths("3510");
}

} // namespace npu::tile_fwk
