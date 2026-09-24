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
 * \file test_common_operation_eliminate.cpp
 * \brief Unit test for common_operation_eliminate pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/graph_partition/common_operation_eliminate.h"
#include <fstream>
#include <vector>
#include <string>

namespace npu {
namespace tile_fwk {

class CommonOperationEliminateTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }
    void TearDown() override {}
};

namespace {
void ExpectCommonOperationEliminateOpCount(ComputationalGraphBuilder& graph, size_t expectedOpCount)
{
    Function* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    CommonOperationEliminate coe;
    EXPECT_EQ(coe.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(function->Operations().size(), expectedOpCount);
}

bool SetAssembleAttributes(ComputationalGraphBuilder& graph)
{
    auto* assemble1 = graph.GetOp("ASSEMBLE1");
    auto* assemble2 = graph.GetOp("ASSEMBLE2");
    if (assemble1 == nullptr || assemble2 == nullptr) {
        return false;
    }
    assemble1->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int64_t>{0, 0}));
    assemble2->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int64_t>{16, 0}));
    return true;
}

bool BuildVecDupAssembleGraph(ComputationalGraphBuilder& graph, MemoryType outMemoryType,
                              bool setVecDupMemoryType = true, bool setAssembleAttr = true)
{
    std::vector<Opcode> opCodes{Opcode::OP_VEC_DUP, Opcode::OP_VEC_DUP, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"out"}, {"out"}};
    std::vector<std::string> opNames{"VECDUP1", "VECDUP2", "ASSEMBLE1", "ASSEMBLE2"};
    bool vecDupOk = setVecDupMemoryType ? graph.AddTensors(DataType::DT_FP32, {16, 16},
                                                           {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"t1", "t2"}) :
                                          graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t1", "t2"});
    if (!vecDupOk || !graph.AddTensor(DataType::DT_FP32, {32, 16}, outMemoryType, "out") ||
        !graph.AddOps(opCodes, ioperands, ooperands, opNames, true)) {
        return false;
    }
    if (!setAssembleAttr) {
        return graph.SetOutCast({"out"});
    }
    if (!SetAssembleAttributes(graph)) {
        return false;
    }
    return graph.SetOutCast({"out"});
}

bool BuildUnaryOpAssembleGraph(ComputationalGraphBuilder& graph, MemoryType outMemoryType)
{
    std::vector<Opcode> opCodes{Opcode::OP_EXP, Opcode::OP_EXP, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{"input"}, {"input"}, {"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"out"}, {"out"}};
    std::vector<std::string> opNames{"EXP1", "EXP2", "ASSEMBLE1", "ASSEMBLE2"};
    if (!graph.AddTensors(DataType::DT_FP32, {16, 16}, {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                          {"input", "t1", "t2"}) ||
        !graph.AddTensor(DataType::DT_FP32, {32, 16}, outMemoryType, "out") ||
        !graph.AddOps(opCodes, ioperands, ooperands, opNames, true) || !SetAssembleAttributes(graph)) {
        return false;
    }
    return graph.SetInCast({"input"}) && graph.SetOutCast({"out"});
}
} // namespace

TEST_F(CommonOperationEliminateTest, EliminateRedundantOps)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 2;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantMultiInputOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t1", "t2"}, {"t3", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"MUL1", "MUL2", "MUL3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t5"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 2;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantMultiOutputOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t4"}, {"t3", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2", "t3"}, {"t4", "t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"RowMax1", "RowMax2", "MUL1", "MUL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t6", "t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantCascadeOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_EXP, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2"}, {"t3"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "EXP1", "EXP2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
    std::shared_ptr<LogicalTensor> tensorPtr = G.GetTensor("t1");
    EXPECT_NE(tensorPtr, nullptr);
    EXPECT_EQ(tensorPtr->GetConsumers().size(), 1);
    tensorPtr = G.GetTensor("t6");
    EXPECT_NE(tensorPtr, nullptr);
    EXPECT_EQ(tensorPtr->GetProducers().size(), 1);
}

TEST_F(CommonOperationEliminateTest, IgnoreSingleInputOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t3"}, {"t2", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t3"}), true);
    EXPECT_EQ(G.SetOutCast({"t5"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreMultiInputOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t1", "t4"}, {"t3", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"MUL1", "MUL2", "MUL3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2", "t4"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentAttr)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_ADDS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ADDS1", "ADDS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Operation* opPtr = G.GetOp("ADDS1");
    EXPECT_NE(opPtr, nullptr);
    opPtr->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 1.0));
    opPtr = G.GetOp("ADDS2");
    EXPECT_NE(opPtr, nullptr);
    const double value2 = 2.0;
    opPtr->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, value2));
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3; // 修复后有序遍历tensor，使得连续冗余场景正确消除
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS", "EXP", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentSubgraph)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Operation* opPtr = G.GetOp("ABS1");
    EXPECT_NE(opPtr, nullptr);
    opPtr->UpdateSubgraphID(0);
    opPtr = G.GetOp("ABS2");
    EXPECT_NE(opPtr, nullptr);
    opPtr->UpdateSubgraphID(1);
    opPtr = G.GetOp("MUL");
    EXPECT_NE(opPtr, nullptr);
    const int subgraphID2 = 2;
    opPtr->UpdateSubgraphID(subgraphID2);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    const int subgraphNum = 3;
    function->SetTotalSubGraphCount(subgraphNum);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, RespectReduceCopyPreSubgraphId)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    auto* abs1 = G.GetOp("ABS1");
    auto* abs2 = G.GetOp("ABS2");
    auto* mul = G.GetOp("MUL");
    ASSERT_NE(abs1, nullptr);
    ASSERT_NE(abs2, nullptr);
    ASSERT_NE(mul, nullptr);
    // The current subgraph id is identical; only the ReduceCopy source subgraph id differs.
    abs1->UpdateSubgraphID(0);
    abs1->SetAttr(OpAttributeKey::reduceCopyPreSubgraphId, static_cast<int64_t>(0));
    abs2->UpdateSubgraphID(0);
    abs2->SetAttr(OpAttributeKey::reduceCopyPreSubgraphId, static_cast<int64_t>(1));
    mul->UpdateSubgraphID(0);
    EXPECT_NE(abs1->DumpAttr().find(OpAttributeKey::reduceCopyPreSubgraphId), std::string::npos);
    EXPECT_NE(abs2->DumpAttr().find(OpAttributeKey::reduceCopyPreSubgraphId), std::string::npos);
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    CommonOperationEliminate COE;
    EXPECT_EQ(COE.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(function->Operations().size(), 3);
}

TEST_F(CommonOperationEliminateTest, IgnoreSpecialOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW,      Opcode::OP_VIEW,      Opcode::OP_MUL,
                                Opcode::OP_L1_TO_FIX, Opcode::OP_L1_TO_FIX, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}, {"t1"}, {"t1"}, {"t5", "t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"VIEW1", "VIEW2", "MUL1", "COPY1", "COPY2", "MUL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4", "t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 6;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateVecDupWithoutAssemblePlacement)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_VEC_DUP, Opcode::OP_VEC_DUP, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"VECDUP1", "VECDUP2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetOutCast({"t3"}), true);
    ExpectCommonOperationEliminateOpCount(G, 2);
}

TEST_F(CommonOperationEliminateTest, PreserveVecDupAssembledToSameMemory)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(BuildVecDupAssembleGraph(G, MemoryType::MEM_UB));
    ExpectCommonOperationEliminateOpCount(G, 4);
}

TEST_F(CommonOperationEliminateTest, EliminateVecDupAssembledToDifferentMemory)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(BuildVecDupAssembleGraph(G, MemoryType::MEM_DEVICE_DDR));
    ExpectCommonOperationEliminateOpCount(G, 3);
}

TEST_F(CommonOperationEliminateTest, PreserveCommonOpAssembledToSameMemory)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(BuildUnaryOpAssembleGraph(G, MemoryType::MEM_UB));
    ExpectCommonOperationEliminateOpCount(G, 4);
}

TEST_F(CommonOperationEliminateTest, PreserveAssembledWhenMemoryTypeUnknown)
{
    ComputationalGraphBuilder attrOnlyGraph;
    ASSERT_TRUE(BuildVecDupAssembleGraph(attrOnlyGraph, MemoryType::MEM_UB, false));
    ExpectCommonOperationEliminateOpCount(attrOnlyGraph, 4);

    ComputationalGraphBuilder unknownMemoryGraph;
    ASSERT_TRUE(BuildVecDupAssembleGraph(unknownMemoryGraph, MemoryType::MEM_UNKNOWN, false, false));
    ExpectCommonOperationEliminateOpCount(unknownMemoryGraph, 4);
}

TEST_F(CommonOperationEliminateTest, TestShmemLoadChecker)
{
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensors(DataType::DT_INT32, {1, 1}, {"dummy"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_INT32, {1, 1, 4, 64}, {"shmemData"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_INT32, {4, 64}, {"out"}), true);
    std::vector<Opcode> opCodes{Opcode::OP_SHMEM_LOAD};
    std::vector<std::vector<std::string>> ioperands{{"dummy", "shmemData"}};
    std::vector<std::vector<std::string>> ooperands{{"out"}};
    std::vector<std::string> opNames{"TILE_SHMEM_LOAD"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"dummy", "shmemData"}), true);
    EXPECT_EQ(G.SetOutCast({"out"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    Status preCheckStatus = COE.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, SUCCESS) << "COE Precheck failed for OP_SHMEM_LOAD!";
}

TEST_F(CommonOperationEliminateTest, PreCheck_CopyIn_InvalidInputNum)
{
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, {"t1", "t2", "t3"}), true);
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"COPY_IN_InvalidInput"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    CommonOperationEliminate COE;
    Status preCheckStatus = COE.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(CommonOperationEliminateTest, PreCheck_CopyIn_OffsetShapeMismatch)
{
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, {"t1", "t2"}), true);
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"COPY_IN_OffsetMismatch"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    Operation* copyOp = G.GetOp("COPY_IN_OffsetMismatch");
    ASSERT_NE(copyOp, nullptr);
    auto opAttr = copyOp->GetOpAttribute();
    ASSERT_NE(opAttr, nullptr);
    auto copyAttr = dynamic_cast<CopyOpAttribute*>(opAttr.get());
    ASSERT_NE(copyAttr, nullptr);
    std::vector<OpImmediate> newFromOffset;
    newFromOffset.emplace_back(0);
    newFromOffset.emplace_back(1);
    newFromOffset.emplace_back(2);
    copyAttr->SetFromOffset(newFromOffset);
    G.GetTensor("t1")->offset = {0, 0};
    CommonOperationEliminate COE;
    Status preCheckStatus = COE.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}
} // namespace tile_fwk
} // namespace npu
