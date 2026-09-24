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
 * \file test_graph_partition.cpp
 * \brief Unit test for GraphPartition pass.
 */

#include <fstream>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#define private public
#include "passes/tile_graph_pass/graph_partition/iso_partitioner.h"
#include "passes/tile_graph_pass/graph_partition/graph_partition.h"
#undef private
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk {

class GraphPartitionTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "GraphPartitionTestStrategy");
        Platform::Instance().ObtainPlatformInfo();
    }
    void TearDown() override { Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN); }
};

void SetScopeInfoForOps(ComputationalGraphBuilder& G, const std::vector<std::string>& opNames,
                        const Operation::ScopeInfo& info)
{
    for (const auto& name : opNames) {
        G.GetOp(name)->SetScopeInfo(info);
    }
}

void GetPairSumGraph(ComputationalGraphBuilder& G)
{
    const int brNum = 4;
    std::vector<int64_t> tileShape{16, 16};
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"t1" + br, "t2" + br, "t3" + br, "t4" + br};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_MULS, Opcode::OP_ADDS};
        std::vector<std::vector<std::string>> ioperands{{"t1" + br}, {"t2" + br}, {"t3" + br}};
        std::vector<std::vector<std::string>> ooperands{{"t2" + br}, {"t3" + br}, {"t4" + br}};
        std::vector<std::string> opNames{"COPY_IN" + br, "MULS" + br, "ADDS" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    std::vector<std::string> sumTensorNames{"s1", "s2", "s3"};
    std::vector<Opcode> sumOpCodes{Opcode::OP_PAIRSUM, Opcode::OP_PAIRSUM, Opcode::OP_PAIRSUM};
    std::vector<std::vector<std::string>> sumIoperands{{"t40", "t41"}, {"t42", "s1"}, {"t43", "s2"}};
    std::vector<std::vector<std::string>> sumOoperands{{"s1"}, {"s2"}, {"s3"}};
    std::vector<std::string> sumOpNames{"SUM1", "SUM2", "SUM3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, sumTensorNames), true);
    EXPECT_EQ(G.AddOps(sumOpCodes, sumIoperands, sumOoperands, sumOpNames, true), true);
    EXPECT_EQ(G.SetInCast({"t10", "t11", "t12", "t13"}), true);
    EXPECT_EQ(G.SetOutCast({"s3"}), true);
}

TEST_F(GraphPartitionTest, TestBuildOpGraph)
{
    ComputationalGraphBuilder G;
    GetPairSumGraph(G);
    Function* function = G.GetFunction();
    const int parallelTH = 10;
    const int cycleLB = 10;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    EXPECT_EQ(partitioner.operationInfo_->opList_.size(), function->Operations().size());
    EXPECT_EQ(partitioner.operationInfo_->magic2Idx_.size(), function->Operations().size());
    EXPECT_EQ(partitioner.operationInfo_->inGraph_.size(), function->Operations().size());
    EXPECT_EQ(partitioner.operationInfo_->opHashList_.size(), function->Operations().size());
    EXPECT_EQ(partitioner.operationInfo_->opCoreType_.size(), function->Operations().size());
    const std::vector<std::pair<std::string, int>> inLinkNum{{"COPY_IN0", 0}, {"MULS0", 1}, {"ADDS0", 1},
                                                             {"SUM1", 2},     {"SUM2", 2},  {"SUM3", 2}};
    for (auto& pr : inLinkNum) {
        EXPECT_NE(G.GetOp(pr.first), nullptr);
        int opMagic = G.GetOp(pr.first)->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        EXPECT_EQ(partitioner.operationInfo_->inGraph_[opIdx].size(), pr.second);
    }
    const std::vector<std::pair<std::string, int>> outLinkNum{{"COPY_IN0", 1}, {"MULS0", 1}, {"ADDS0", 1},
                                                              {"SUM1", 1},     {"SUM2", 1},  {"SUM3", 0}};
    for (auto& pr : outLinkNum) {
        EXPECT_NE(G.GetOp(pr.first), nullptr);
        int opMagic = G.GetOp(pr.first)->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        EXPECT_EQ(partitioner.operationInfo_->outGraph_[opIdx].size(), pr.second);
    }
    int copyIdx0 = partitioner.operationInfo_->magic2Idx_[G.GetOp("COPY_IN0")->GetOpMagic()];
    int copyIdx1 = partitioner.operationInfo_->magic2Idx_[G.GetOp("COPY_IN1")->GetOpMagic()];
    int sumIdx1 = partitioner.operationInfo_->magic2Idx_[G.GetOp("SUM1")->GetOpMagic()];
    int sumIdx2 = partitioner.operationInfo_->magic2Idx_[G.GetOp("SUM2")->GetOpMagic()];
    EXPECT_EQ(partitioner.operationInfo_->opHashList_[copyIdx0], partitioner.operationInfo_->opHashList_[copyIdx1]);
    EXPECT_EQ(partitioner.operationInfo_->opHashList_[sumIdx1], partitioner.operationInfo_->opHashList_[sumIdx2]);
    EXPECT_NE(partitioner.operationInfo_->opHashList_[copyIdx0], partitioner.operationInfo_->opHashList_[sumIdx2]);
    std::unordered_set<uint64_t> copyInHash;
    const int brNum = 4;
    for (int i = 0; i < brNum; i++) {
        EXPECT_NE(G.GetOp("COPY_IN" + std::to_string(i)), nullptr);
        int opMagic = G.GetOp("COPY_IN" + std::to_string(i))->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        int nodeIdx = partitioner.superNodeInfo_->op2Node_[opIdx];
        copyInHash.insert(partitioner.superNodeInfo_->nodeHashList_[nodeIdx]);
    }
    const int copyInHashNum = 3;
    EXPECT_EQ(copyInHash.size(), copyInHashNum);
}

void GetReshapeGraph(ComputationalGraphBuilder& G)
{
    const int brNum = 4;
    std::vector<int64_t> tileShape{16, 16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"rin", "rout"}), true);
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"t1" + br, "t2" + br, "t3" + br};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_RESHAPE, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> ioperands{{"t1" + br}, {"t2" + br}, {"t3" + br}};
        std::vector<std::vector<std::string>> ooperands{{"t2" + br}, {"t3" + br}, {"rin"}};
        std::vector<std::string> opNames{"COPY_IN" + br, "RESHAPE_IN" + br, "ASSEMBLE" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    EXPECT_EQ(G.AddOp(Opcode::OP_RESHAPE, {"rin"}, {"rout"}, "MULTI_RESHAPE", true), true);
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"b1" + br, "b2" + br, "b3" + br};
        std::vector<Opcode> opCodes{Opcode::OP_VIEW, Opcode::OP_RESHAPE, Opcode::OP_COPY_OUT};
        std::vector<std::vector<std::string>> ioperands{{"rout"}, {"b1" + br}, {"b2" + br}};
        std::vector<std::vector<std::string>> ooperands{{"b1" + br}, {"b2" + br}, {"b3" + br}};
        std::vector<std::string> opNames{"VIEW" + br, "RESHAPE_OUT" + br, "COPY_OUT" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    EXPECT_EQ(G.SetInCast({"t10", "t11", "t12", "t13"}), true);
    EXPECT_EQ(G.SetOutCast({"b30", "b31", "b32", "b33"}), true);
}

TEST_F(GraphPartitionTest, TestSuperNode)
{
    ComputationalGraphBuilder G;
    GetReshapeGraph(G);
    Function* function = G.GetFunction();
    const int parallelTH = 10;
    const int cycleLB = 10;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    std::unordered_set<int> frontReshapeNode;
    const int brNum = 4;
    for (int i = 0; i < brNum; i++) {
        EXPECT_NE(G.GetOp("RESHAPE_IN" + std::to_string(i)), nullptr);
        int opMagic = G.GetOp("RESHAPE_IN" + std::to_string(i))->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        frontReshapeNode.insert(partitioner.superNodeInfo_->op2Node_[opIdx]);
    }
    EXPECT_EQ(frontReshapeNode.size(), brNum);
    std::unordered_set<int> backReshapeNode;
    for (int i = 0; i < brNum; i++) {
        EXPECT_NE(G.GetOp("RESHAPE_OUT" + std::to_string(i)), nullptr);
        int opMagic = G.GetOp("RESHAPE_OUT" + std::to_string(i))->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        backReshapeNode.insert(partitioner.superNodeInfo_->op2Node_[opIdx]);
    }
    EXPECT_EQ(backReshapeNode.size(), brNum);
    const int subGraphNum = 9;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestReduceNodeHash)
{
    ComputationalGraphBuilder G;
    GetPairSumGraph(G);
    Function* function = G.GetFunction();
    const int parallelTH = 10;
    const int cycleLB = 10;
    const int useNodeHash = true;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    std::unordered_set<uint64_t> copyInHash;
    const int brNum = 4;
    for (int i = 0; i < brNum; i++) {
        EXPECT_NE(G.GetOp("COPY_IN" + std::to_string(i)), nullptr);
        int opMagic = G.GetOp("COPY_IN" + std::to_string(i))->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        int nodeIdx = partitioner.superNodeInfo_->op2Node_[opIdx];
        copyInHash.insert(partitioner.superNodeInfo_->nodeHashList_[nodeIdx]);
    }
    EXPECT_EQ(copyInHash.size(), 1);
}

void GetCrossGraph(ComputationalGraphBuilder& G)
{
    const int brNum = 4;
    std::vector<int64_t> tileShape{16, 16};
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"t1" + br, "t2" + br, "t3" + br, "t4" + br};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_RESHAPE, Opcode::OP_ABS};
        std::vector<std::vector<std::string>> ioperands{{"t1" + br}, {"t2" + br}, {"t3" + br}};
        std::vector<std::vector<std::string>> ooperands{{"t2" + br}, {"t3" + br}, {"t4" + br}};
        std::vector<std::string> opNames{"COPY_IN" + br, "RESHAPE_IN" + br, "ABS" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::string sbr = std::to_string((i + 1) % brNum);
        std::vector<std::string> tensorNames{"b1" + br, "b2" + br, "b3" + br};
        std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_RESHAPE, Opcode::OP_COPY_OUT};
        std::vector<std::vector<std::string>> ioperands{{"t4" + br, "t4" + sbr}, {"b1" + br}, {"b2" + br}};
        std::vector<std::vector<std::string>> ooperands{{"b1" + br}, {"b2" + br}, {"b3" + br}};
        std::vector<std::string> opNames{"MUL" + br, "RESHAPE_OUT" + br, "COPY_OUT" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    EXPECT_EQ(G.SetInCast({"t10", "t11", "t12", "t13"}), true);
    EXPECT_EQ(G.SetOutCast({"b30", "b31", "b32", "b33"}), true);
}

TEST_F(GraphPartitionTest, TestBuildIsomorphismGraph)
{
    ComputationalGraphBuilder G;
    GetCrossGraph(G);
    Function* function = G.GetFunction();
    const int parallelTH = 10;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = 8;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestIsoParameterFailure)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_EQ(function->Operations().size(), 0);

    const int parallelTHFail = -2;
    const int parallelTH = 10;
    const int cycleLBFail = -1;
    const int cycleLB = 100000;
    const int useNodeHash = false;

    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTHFail, cycleLB, useNodeHash), FAILED);
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLBFail, useNodeHash), FAILED);

    EXPECT_EQ(partitioner.PartitionGraph(*function), FAILED);
}

TEST_F(GraphPartitionTest, TestOspParameter)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_EQ(function->Operations().size(), 0);

    OspPartitioner partitioner(OspMode::SARKAR);

    const int cycleLBFail = -11;
    const int cycleLB = 10000;

    function->paramConfigs_.sgPgLowerBound = cycleLBFail;
    EXPECT_EQ(partitioner.SetParameter(*function), FAILED);

    function->paramConfigs_.sgPgLowerBound = cycleLB;
    EXPECT_EQ(partitioner.SetParameter(*function), SUCCESS);
}

TEST_F(GraphPartitionTest, TestEmptyGraph)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_EQ(function->Operations().size(), 0);
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), 0);
}

TEST_F(GraphPartitionTest, TestSarkarEmptyGraph)
{
    const std::string partitionAlg = "OspSarkar";

    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    function->paramConfigs_.sgPartitionAlgorithm = partitionAlg;
    EXPECT_EQ(function->paramConfigs_.sgPartitionAlgorithm, partitionAlg);
    EXPECT_EQ(function->Operations().size(), 0);
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);
}

TEST_F(GraphPartitionTest, TestPartitionerParameterFailure)
{
    const std::string partitionAlg = "NotExistent";

    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    function->paramConfigs_.sgPartitionAlgorithm = partitionAlg;
    EXPECT_EQ(function->paramConfigs_.sgPartitionAlgorithm, partitionAlg);
    EXPECT_EQ(function->Operations().size(), 0);
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), FAILED);
}

void GetCubeVectorGraph(ComputationalGraphBuilder& G, int brNum)
{
    std::vector<int64_t> tileShape{16, 16};
    std::vector<std::string> inTensorNames;
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        inTensorNames.push_back("ta1" + br);
        std::vector<std::string> tensorNames{"ta1" + br, "ta2" + br, "ta3" + br, "ta4" + br, "ta5" + br, "ta6" + br};
        std::vector<MemoryType> tensorMemType{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                              MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0A};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_CAST, Opcode::OP_COPY_OUT, Opcode::OP_COPY_IN,
                                    Opcode::OP_L1_TO_L0A};
        std::vector<std::vector<std::string>> ioperands{
            {"ta1" + br}, {"ta2" + br}, {"ta3" + br}, {"ta4" + br}, {"ta5" + br}};
        std::vector<std::vector<std::string>> ooperands{
            {"ta2" + br}, {"ta3" + br}, {"ta4" + br}, {"ta5" + br}, {"ta6" + br}};
        std::vector<std::string> opNames{"IN_A" + br, "CAST_A" + br, "OUT_A" + br, "IN_L1_A" + br, "L1_TO_L0A" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorMemType, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        inTensorNames.push_back("tb1" + br);
        std::vector<std::string> tensorNames{"tb1" + br, "tb2" + br, "tb3" + br, "tb4" + br, "tb5" + br, "tb6" + br};
        std::vector<MemoryType> tensorMemType{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                              MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0B};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_CAST, Opcode::OP_COPY_OUT, Opcode::OP_COPY_IN,
                                    Opcode::OP_L1_TO_L0B};
        std::vector<std::vector<std::string>> ioperands{
            {"tb1" + br}, {"tb2" + br}, {"tb3" + br}, {"tb4" + br}, {"tb5" + br}};
        std::vector<std::vector<std::string>> ooperands{
            {"tb2" + br}, {"tb3" + br}, {"tb4" + br}, {"tb5" + br}, {"tb6" + br}};
        std::vector<std::string> opNames{"IN_B" + br, "CAST_B" + br, "OUT_B" + br, "IN_L1_B" + br, "L1_TO_L0B" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorMemType, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    for (int i = 0; i < brNum; i++) {
        EXPECT_EQ(G.AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_L0C, "tc" + std::to_string(i)), true);
    }
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_DEVICE_DDR, "tout"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_A_MUL_B, {"ta60", "tb60"}, {"tc0"}, "MUL1", true), true);
    for (int i = 1; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::string lbr = std::to_string(i - 1);
        EXPECT_EQ(G.AddOp(Opcode::OP_A_MULACC_B, {"ta6" + br, "tb6" + br, "tc" + lbr}, {"tc" + br}, "MC" + br, true),
                  true);
    }
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"tc" + std::to_string(brNum - 1)}, {"tout"}, "COPY_OUT_C", true), true);
    EXPECT_EQ(G.SetInCast(inTensorNames), true);
    EXPECT_EQ(G.SetOutCast({"tout"}), true);
}

TEST_F(GraphPartitionTest, TestCVGraph)
{
    ComputationalGraphBuilder G;
    const int brNum = 4;
    GetCubeVectorGraph(G, brNum);
    Function* function = G.GetFunction();
    const int parallelTH = 1;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    std::unordered_set<std::string> cubeOp{"MUL1", "MC1", "MC2", "MC3", "COPY_OUT_C"};
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        cubeOp.insert("IN_L1_A" + br);
        cubeOp.insert("IN_L1_B" + br);
        cubeOp.insert("L1_TO_L0A" + br);
        cubeOp.insert("L1_TO_L0B" + br);
    }
    const int subGraphNum = 9;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
    std::unordered_set<int> subgraphIDs;
    for (auto& opPair : G.operations_) {
        Operation* op = opPair.second;
        EXPECT_NE(op, nullptr);
        if (cubeOp.count(opPair.first) > 0) {
            EXPECT_EQ(op->HasAttr(OpAttributeKey::isCube) && op->GetBoolAttribute(OpAttributeKey::isCube), true);
        } else {
            EXPECT_EQ(op->HasAttr(OpAttributeKey::isCube) && !op->GetBoolAttribute(OpAttributeKey::isCube), true);
        }
        EXPECT_EQ(op->GetSubgraphID() >= 0 && op->GetSubgraphID() < subGraphNum, true);
        subgraphIDs.insert(op->GetSubgraphID());
    }
    EXPECT_EQ(subgraphIDs.size(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestOspCVGraph)
{
    for (const auto partitionAlg : {"Iso", "OspSarkar", "OspBsp"}) {
        ComputationalGraphBuilder G;
        const int brNum = 4;
        GetCubeVectorGraph(G, brNum);
        Function* function = G.GetFunction();
        function->paramConfigs_.sgPartitionAlgorithm = partitionAlg;

        GraphPartition gpp;
        EXPECT_EQ(gpp.PreCheck(*function), SUCCESS);
        EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);

        std::unordered_set<std::string> cubeOp{"MUL1", "MC1", "MC2", "MC3", "COPY_OUT_C"};
        for (int i = 0; i < brNum; i++) {
            std::string br = std::to_string(i);
            cubeOp.insert("IN_L1_A" + br);
            cubeOp.insert("IN_L1_B" + br);
            cubeOp.insert("L1_TO_L0A" + br);
            cubeOp.insert("L1_TO_L0B" + br);
        }
        const int subGraphNum = function->GetTotalSubGraphCount();
        std::unordered_map<int, bool> subgraphIDs2IsCube;
        for (auto& opPair : G.operations_) {
            Operation* op = opPair.second;
            EXPECT_NE(op, nullptr);
            if (cubeOp.count(opPair.first) > 0) {
                EXPECT_EQ(op->HasAttr(OpAttributeKey::isCube) && op->GetBoolAttribute(OpAttributeKey::isCube), true);
            } else {
                EXPECT_EQ(op->HasAttr(OpAttributeKey::isCube) && !op->GetBoolAttribute(OpAttributeKey::isCube), true);
            }
            EXPECT_EQ(op->GetSubgraphID() >= 0 && op->GetSubgraphID() < subGraphNum, true);

            const auto subgraphId = op->GetSubgraphID();
            if (subgraphIDs2IsCube.count(subgraphId) > 0) {
                EXPECT_EQ(subgraphIDs2IsCube.at(subgraphId), op->GetBoolAttribute(OpAttributeKey::isCube));
            } else {
                subgraphIDs2IsCube.emplace(subgraphId, op->GetBoolAttribute(OpAttributeKey::isCube));
            }
        }
        EXPECT_EQ(subgraphIDs2IsCube.size(), subGraphNum);
        EXPECT_EQ(gpp.PostCheck(*function), SUCCESS);
        Program::GetInstance().Reset();
    }
}

TEST_F(GraphPartitionTest, TestMixCVGraph)
{
    for (const auto mode : {OspMode::SARKAR, OspMode::MERKLEBSP}) {
        ComputationalGraphBuilder G;
        const int brNum = 4;
        GetCubeVectorGraph(G, brNum);
        Function* function = G.GetFunction();

        for (const bool cvMix : {true, false}) {
            OspPartitioner partitioner(mode, cvMix);
            EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);

            const int subGraphNum = function->GetTotalSubGraphCount();
            for (auto& opPair : G.operations_) {
                Operation* op = opPair.second;
                EXPECT_NE(op, nullptr);
                EXPECT_EQ(op->GetSubgraphID() >= 0 && op->GetSubgraphID() < subGraphNum, true);
            }
        }
    }
}

void GetMergeableGraph(ComputationalGraphBuilder& G, int brNum)
{
    std::vector<int64_t> tileShape{16, 16};
    std::vector<std::string> inCast;
    std::vector<std::string> outCast;
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"t1" + br, "t2" + br, "t3" + br, "t4" + br,
                                             "b1" + br, "b2" + br, "b3" + br, "b4" + br};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_RESHAPE, Opcode::OP_ABS,
                                    Opcode::OP_COPY_IN, Opcode::OP_RESHAPE, Opcode::OP_ABS};
        std::vector<std::vector<std::string>> ioperands{{"t1" + br}, {"t2" + br}, {"t3" + br},
                                                        {"b1" + br}, {"b2" + br}, {"b3" + br}};
        std::vector<std::vector<std::string>> ooperands{{"t2" + br}, {"t3" + br}, {"t4" + br},
                                                        {"b2" + br}, {"b3" + br}, {"b4" + br}};
        std::vector<std::string> opNames{"COPY_INt" + br, "RESHAPE_INt" + br, "ABSt" + br,
                                         "COPY_INb" + br, "RESHAPE_INb" + br, "ABSb" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
        inCast.push_back("t1" + br);
        inCast.push_back("b1" + br);
    }
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"m1" + br, "m2" + br, "m3" + br};
        std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_RESHAPE, Opcode::OP_COPY_OUT};
        std::vector<std::vector<std::string>> ioperands{{"t4" + br, "b4" + br}, {"m1" + br}, {"m2" + br}};
        std::vector<std::vector<std::string>> ooperands{{"m1" + br}, {"m2" + br}, {"m3" + br}};
        std::vector<std::string> opNames{"MUL" + br, "RESHAPE_OUT" + br, "COPY_OUT" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
        outCast.push_back("m3" + br);
    }
    EXPECT_EQ(G.SetInCast(inCast), true);
    EXPECT_EQ(G.SetOutCast(outCast), true);
}

TEST_F(GraphPartitionTest, TestDynamicCycleEstimation)
{
    ComputationalGraphBuilder G;
    const int brNum = 4;
    GetMergeableGraph(G, brNum);
    Function* function = G.GetFunction();
    const int parallelTH = 1;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = brNum;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestParallelThreshold)
{
    ComputationalGraphBuilder G;
    const int brNum = 4;
    GetMergeableGraph(G, brNum);
    Function* function = G.GetFunction();
    const int parallelTH = brNum * 2;
    const int cycleLB = 0;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = 3 * brNum;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestSmallGraphBound)
{
    ComputationalGraphBuilder G;
    const int brNum = 4;
    GetMergeableGraph(G, brNum);
    Function* function = G.GetFunction();
    const int parallelTH = brNum * 2;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = brNum;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestLargeSuperNode)
{
    ComputationalGraphBuilder G;
    const int brNum = 5000;
    GetCubeVectorGraph(G, brNum);
    Function* function = G.GetFunction();
    const int parallelTH = brNum * 2;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
}

void GetWideGraph(ComputationalGraphBuilder& G, int brNum)
{
    std::vector<int64_t> tileShape{16, 16};
    std::vector<std::string> outCast;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"h1", "h2", "h3"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"h1"}, {"h2"}, "COPY_IN", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ABS, {"h2"}, {"h3"}, "ABS", true), true);
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"t1" + br, "t2" + br, "t3" + br};
        std::vector<Opcode> opCodes{Opcode::OP_EXP, Opcode::OP_RESHAPE, Opcode::OP_COPY_OUT};
        std::vector<std::vector<std::string>> ioperands{{"h3"}, {"t1" + br}, {"t2" + br}};
        std::vector<std::vector<std::string>> ooperands{{"t1" + br}, {"t2" + br}, {"t3" + br}};
        std::vector<std::string> opNames{"EXP" + br, "RESHAPE" + br, "COPY_OUT" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
        outCast.push_back("t3" + br);
    }
    EXPECT_EQ(G.SetInCast({"h1"}), true);
    EXPECT_EQ(G.SetOutCast(outCast), true);
}

TEST_F(GraphPartitionTest, TestLargeWideGraph)
{
    ComputationalGraphBuilder G;
    const int brNum = 5000;
    GetWideGraph(G, brNum);
    Function* function = G.GetFunction();
    const int parallelTH = 20;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
}

void GetDeepGraph(ComputationalGraphBuilder& G, int brNum)
{
    std::vector<int64_t> tileShape{16, 16};
    std::vector<std::string> outCast;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"ha", "a0", "hb", "b0"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"ha"}, {"a0"}, "COPY_INa", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"hb"}, {"b0"}, "COPY_INb", true), true);
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::string nbr = std::to_string(i + 1);
        std::vector<std::string> tensorNames{"a" + nbr, "b" + nbr};
        std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS};
        std::vector<std::vector<std::string>> ioperands{{"a" + br}, {"b" + br}};
        std::vector<std::vector<std::string>> ooperands{{"a" + nbr}, {"b" + nbr}};
        std::vector<std::string> opNames{"ABSa" + br, "ABSb" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    std::string tbr = std::to_string(brNum);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"ta", "tb"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"a" + tbr}, {"ta"}, "COPY_OUTa", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"b" + tbr}, {"tb"}, "COPY_OUTb", true), true);
    EXPECT_EQ(G.SetInCast({"ha", "hb"}), true);
    EXPECT_EQ(G.SetOutCast({"ta", "tb"}), true);
}

TEST_F(GraphPartitionTest, TestLargeDeepGraph)
{
    ComputationalGraphBuilder G;
    const int brNum = 5000;
    GetDeepGraph(G, brNum);
    Function* function = G.GetFunction();
    const int parallelTH = 20;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
}

TEST_F(GraphPartitionTest, TestIsomorphismGraph)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"h1", "h2", "h3", "h41", "h42", "h5", "h6"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"h1"}, {"h2"}, "COPY_IN", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ABS, {"h2"}, {"h3"}, "ABS", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {"h3"}, {"h41"}, "ADDS1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {"h3"}, {"h42"}, "ADDS2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MUL, {"h41", "h42"}, {"h5"}, "MUL", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"h5"}, {"h6"}, "COPY_OUT", true), true);
    EXPECT_EQ(G.SetInCast({"h1"}), true);
    EXPECT_EQ(G.SetOutCast({"h6"}), true);

    Function* function = G.GetFunction();
    const int parallelTH = 20;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = 4;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

void RunScopeTest(bool allowCrossScopeMerge, int expectedSubGraphNum)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{32, 32};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"h1", "h2", "h3", "h41", "h42", "h5", "h6"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"h1"}, {"h2"}, "COPY_IN", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ABS, {"h2"}, {"h3"}, "ABS", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {"h3"}, {"h41"}, "A1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {"h3"}, {"h42"}, "A2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MUL, {"h41", "h42"}, {"h5"}, "M", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"h5"}, {"h6"}, "COPY_OUT", true), true);
    EXPECT_EQ(G.SetInCast({"h1"}), true);
    EXPECT_EQ(G.SetOutCast({"h6"}), true);
    Operation::ScopeInfo info;
    info.scopeId = 1;
    info.allowCrossScopeMerge = allowCrossScopeMerge;
    for (auto* op : {G.GetOp("A1"), G.GetOp("A2"), G.GetOp("M"), G.GetOp("COPY_OUT")}) {
        op->SetScopeInfo(info);
    }

    Function* function = G.GetFunction();
    const int parallelTH = 20;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), expectedSubGraphNum);
}

TEST_F(GraphPartitionTest, TestScopeCase1) { RunScopeTest(false, 2); }

TEST_F(GraphPartitionTest, TestScopeCase2) { RunScopeTest(true, 1); }

void RunScopeTest2(bool paraller, bool crossScopeMerge, int expectedSubGraphNum)
{
    ComputationalGraphBuilder G;
    const int brNum = 1;
    GetMergeableGraph(G, brNum);
    Function* function = G.GetFunction();
    Operation::ScopeInfo info(1);
    info.allowParallelMerge = paraller;
    info.allowCrossScopeMerge = crossScopeMerge,
    SetScopeInfoForOps(G, {"COPY_INt0", "RESHAPE_INt0", "ABSt0", "COPY_INb0", "RESHAPE_INb0", "ABSb0"}, info);
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(100000, 20, 0), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), expectedSubGraphNum);
}

TEST_F(GraphPartitionTest, TestScopeCase3) { RunScopeTest2(true, false, 2); }

TEST_F(GraphPartitionTest, TestScopeCase4) { RunScopeTest2(true, true, 1); }

void AddGLMTensors(ComputationalGraphBuilder& G)
{
    std::vector<int64_t> shapeSij{12, 512};
    std::vector<int64_t> shapeOne{12, 1};
    std::vector<int64_t> shapeOi{12, 128};
    std::vector<int64_t> shapeVBlock{128, 128};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shapeSij, {"sij", "sij_ub", "sij_scale", "tsub", "tilda_pij"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shapeOne,
                           {"max_update", "sum_update", "max_update_ub", "tilda_mij", "max_new", "sum_local",
                            "max_new_ddr", "max_update_ub2", "tsub2", "update_mul", "sum_update_ub", "tmp_mul",
                            "sum_update_out", "sum_update_out_ddr"}),
              true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shapeOi,
                           {"oi_update", "oi_tmp_ddr", "oi_update_ub", "oi_tmp_ub", "tmp_oi", "oi_update_out",
                            "oi_update_out_ddr"}),
              true);
    EXPECT_EQ(G.AddTensors(DataType::DT_BF16, shapeSij, {"tilda_pij_fp16", "pij_assembled_ddr"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_BF16, std::vector<int64_t>{-1, 128}, {"value_cache_2d"}), true);

    std::vector<MemoryType> memL1(4, MemoryType::MEM_L1);
    std::vector<MemoryType> memL0B(4, MemoryType::MEM_L0B);
    std::vector<MemoryType> memL0A(4, MemoryType::MEM_L0A);
    std::vector<MemoryType> memL0C(4, MemoryType::MEM_L0C);

    EXPECT_EQ(G.AddTensors(DataType::DT_BF16, shapeVBlock, memL1, {"v_l1_0", "v_l1_1", "v_l1_2", "v_l1_3"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_BF16, shapeVBlock, memL0B, {"v_l0b_0", "v_l0b_1", "v_l0b_2", "v_l0b_3"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_BF16, shapeOi, memL1, {"a_l1_0", "a_l1_1", "a_l1_2", "a_l1_3"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_BF16, shapeOi, memL0A, {"a_l0a_0", "a_l0a_1", "a_l0a_2", "a_l0a_3"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shapeOi, memL0C, {"mmul_a", "mmul_b", "mmul_c", "mmul_d"}), true);

    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {12, 128}, "div_out"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {1, 12, 128}, "reshape_out"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {1, 12, 128}, "cast_out"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {1, 12, 128}, "atten_out"), true);
}

void AddGLMSoftmaxOps(ComputationalGraphBuilder& G)
{
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"sij"}, {"sij_ub"}, "VIEW_sij", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MULS, {"sij_ub"}, {"sij_scale"}, "MULS_scale", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ROWMAX_SINGLE, {"sij_scale"}, {"tilda_mij"}, "ROWMAX_mij", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"max_update"}, {"max_update_ub"}, "VIEW_max_update", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MAXIMUM, {"max_update_ub", "tilda_mij"}, {"max_new"}, "MAX_new", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_SUB, {"sij_scale", "max_new"}, {"tsub"}, "SUB_tsub", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_EXP, {"tsub"}, {"tilda_pij"}, "EXP_pij", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"tilda_pij"}, {"tilda_pij_fp16"}, "CAST_fp16", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tilda_pij_fp16"}, {"pij_assembled_ddr"}, "ASSEMBLE_pij_ddr", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ROWSUM_SINGLE, {"tilda_pij"}, {"sum_local"}, "ROWSUM_local", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"max_new"}, {"max_new_ddr"}, "ASSEMBLE_max_new_ddr", true), true);
}

void AddGLMCubeOps(ComputationalGraphBuilder& G)
{
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"value_cache_2d"}, {"v_l1_0"}, "VIEW_v_l1_0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"value_cache_2d"}, {"v_l1_1"}, "VIEW_v_l1_1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"value_cache_2d"}, {"v_l1_2"}, "VIEW_v_l1_2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"value_cache_2d"}, {"v_l1_3"}, "VIEW_v_l1_3", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"pij_assembled_ddr"}, {"a_l1_0"}, "VIEW_a_l1_0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"pij_assembled_ddr"}, {"a_l1_1"}, "VIEW_a_l1_1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"pij_assembled_ddr"}, {"a_l1_2"}, "VIEW_a_l1_2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"pij_assembled_ddr"}, {"a_l1_3"}, "VIEW_a_l1_3", true), true);
    {
        std::vector<std::pair<std::string, std::string>> vViews = {
            {"v_l1_0", "v_l0b_0"}, {"v_l1_1", "v_l0b_1"}, {"v_l1_2", "v_l0b_2"}, {"v_l1_3", "v_l0b_3"}};
        for (int i = 0; i < static_cast<int>(vViews.size()); i++) {
            std::string opName = "VIEW_L0B_v" + std::to_string(i);
            EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {vViews[i].first}, {vViews[i].second}, opName, true), true);
            auto viewOp = G.GetOp(opName);
            std::vector<int64_t> offset = {0, 0, 0};
            viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_L0B));
        }
    }
    {
        std::vector<std::pair<std::string, std::string>> aViews = {
            {"a_l1_0", "a_l0a_0"}, {"a_l1_1", "a_l0a_1"}, {"a_l1_2", "a_l0a_2"}, {"a_l1_3", "a_l0a_3"}};
        for (int i = 0; i < static_cast<int>(aViews.size()); i++) {
            std::string opName = "VIEW_L0A_a" + std::to_string(i);
            EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {aViews[i].first}, {aViews[i].second}, opName, true), true);
            auto viewOp = G.GetOp(opName);
            std::vector<int64_t> offset = {0, 0, 0};
            viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_L0A));
        }
    }
    EXPECT_EQ(G.AddOp(Opcode::OP_A_MUL_B, {"a_l0a_0", "v_l0b_0"}, {"mmul_a"}, "MMUL_a", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_A_MULACC_B, {"a_l0a_1", "v_l0b_1", "mmul_a"}, {"mmul_b"}, "MMACC_b", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_A_MULACC_B, {"a_l0a_2", "v_l0b_2", "mmul_b"}, {"mmul_c"}, "MMACC_c", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_A_MULACC_B, {"a_l0a_3", "v_l0b_3", "mmul_c"}, {"mmul_d"}, "MMACC_d", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"mmul_d"}, {"oi_tmp_ddr"}, "ASSEMBLE_oi_tmp_ddr", true), true);
}

void AddGLMUpdateOps(ComputationalGraphBuilder& G)
{
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"max_update"}, {"max_update_ub2"}, "VIEW_max_update2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_SUB, {"max_update_ub2", "max_new"}, {"tsub2"}, "SUB_tsub2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_EXP, {"tsub2"}, {"update_mul"}, "EXP_update_mul", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"sum_update"}, {"sum_update_ub"}, "VIEW_sum_update", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MUL, {"sum_update_ub", "update_mul"}, {"tmp_mul"}, "MUL_tmp", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADD, {"tmp_mul", "sum_local"}, {"sum_update_out"}, "ADD_sum_update", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"sum_update_out"}, {"sum_update_out_ddr"}, "ASSEMBLE_sum_ddr", true), true);

    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"oi_update"}, {"oi_update_ub"}, "VIEW_oi_update", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"oi_tmp_ddr"}, {"oi_tmp_ub"}, "VIEW_oi_tmp", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MUL, {"oi_update_ub", "update_mul"}, {"tmp_oi"}, "MUL_oi", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADD, {"tmp_oi", "oi_tmp_ub"}, {"oi_update_out"}, "ADD_oi_update", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"oi_update_out"}, {"oi_update_out_ddr"}, "ASSEMBLE_oi_ddr", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_DIV, {"oi_update_out", "sum_update_out"}, {"div_out"}, "DIV_1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_RESHAPE, {"div_out"}, {"reshape_out"}, "RESHAPE_1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"reshape_out"}, {"cast_out"}, "CAST_1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"cast_out"}, {"atten_out"}, "ASSEMBLE_1", true), true);
}

int VerifyOpsInSameSubgraph(ComputationalGraphBuilder& G, const std::vector<std::string>& opNames)
{
    int subgraphId = G.GetOp(opNames[0])->GetSubgraphID();
    for (const auto& name : opNames) {
        EXPECT_EQ(G.GetOp(name)->GetSubgraphID(), subgraphId);
    }
    return subgraphId;
}

void ConstructGLMAttentionCase(ComputationalGraphBuilder& G)
{
    AddGLMTensors(G);
    AddGLMSoftmaxOps(G);
    AddGLMCubeOps(G);
    AddGLMUpdateOps(G);

    EXPECT_EQ(G.SetInCast({"sij", "max_update", "sum_update", "oi_update", "value_cache_2d"}), true);
    EXPECT_EQ(G.SetOutCast({"max_new_ddr", "sum_update_out_ddr", "oi_update_out_ddr", "atten_out"}), true);
}

TEST_F(GraphPartitionTest, TestScopeCase5)
{
    ComputationalGraphBuilder G;
    ConstructGLMAttentionCase(G);

    Operation::ScopeInfo scope1(1);
    scope1.allowParallelMerge = true;
    scope1.allowCrossScopeMerge = false;
    std::vector<std::string> scope1Ops = {"MULS_scale", "ROWMAX_mij", "MAX_new",     "SUB_tsub",
                                          "EXP_pij",    "CAST_fp16",  "ROWSUM_local"};
    SetScopeInfoForOps(G, scope1Ops, scope1);

    Operation::ScopeInfo scope2(2);
    scope2.allowParallelMerge = true;
    scope2.allowCrossScopeMerge = false;
    std::vector<std::string> scope2Ops = {"SUB_tsub2", "EXP_update_mul", "MUL_tmp", "ADD_sum_update"};
    SetScopeInfoForOps(G, scope2Ops, scope2);

    Function* function = G.GetFunction();
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(100000, 20, 0), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    for (auto& op : function->Operations()) {
        EXPECT_GE(op.GetSubgraphID(), 0);
    }

    int scope1Subgraph = VerifyOpsInSameSubgraph(G, scope1Ops);
    int scope2Subgraph = VerifyOpsInSameSubgraph(G, scope2Ops);
    EXPECT_NE(scope1Subgraph, scope2Subgraph);

    std::vector<std::string> cubeOps = {
        "VIEW_v_l1_0", "VIEW_v_l1_1", "VIEW_v_l1_2",        "VIEW_v_l1_3", "VIEW_a_l1_0", "VIEW_a_l1_1",
        "VIEW_a_l1_2", "VIEW_a_l1_3", "VIEW_L0B_v0",        "VIEW_L0B_v1", "VIEW_L0B_v2", "VIEW_L0B_v3",
        "VIEW_L0A_a0", "VIEW_L0A_a1", "VIEW_L0A_a2",        "VIEW_L0A_a3", "MMUL_a",      "MMACC_b",
        "MMACC_c",     "MMACC_d",     "ASSEMBLE_oi_tmp_ddr"};
    int cubeSubgraph = VerifyOpsInSameSubgraph(G, cubeOps);
    for (const auto& name : cubeOps) {
        auto op = G.GetOp(name);
        auto isCube = op->HasAttr(OpAttributeKey::isCube) && op->GetBoolAttribute(OpAttributeKey::isCube);
        EXPECT_EQ(isCube, true);
        EXPECT_EQ(op->GetCvFuseId(), -1);
    }
    EXPECT_NE(cubeSubgraph, scope1Subgraph);
    EXPECT_NE(cubeSubgraph, scope2Subgraph);

    std::vector<std::string> v2Ops = {"VIEW_oi_update", "VIEW_oi_tmp", "MUL_oi", "ADD_oi_update", "ASSEMBLE_oi_ddr"};
    int v2Subgraph = VerifyOpsInSameSubgraph(G, v2Ops);
    EXPECT_NE(v2Subgraph, scope1Subgraph);
    EXPECT_NE(v2Subgraph, scope2Subgraph);
    EXPECT_NE(v2Subgraph, cubeSubgraph);
}

TEST_F(GraphPartitionTest, TestScopeParallelMergeRebuildAvoidLoop)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape{16, 16};
    EXPECT_EQ(
        G.AddTensors(DataType::DT_FP32, shape, {"input", "scope_head_out", "bridge_out", "scope_tail_out", "output"}),
        true);
    EXPECT_EQ(G.AddOps({Opcode::OP_ADDS, Opcode::OP_MULS, Opcode::OP_ADDS, Opcode::OP_COPY_OUT},
                       {{"input"}, {"scope_head_out"}, {"bridge_out"}, {"scope_tail_out"}},
                       {{"scope_head_out"}, {"bridge_out"}, {"scope_tail_out"}, {"output"}},
                       {"SCOPE_HEAD", "BRIDGE", "SCOPE_TAIL", "COPY_OUT"}, true),
              true);
    EXPECT_EQ(G.SetInCast({"input"}), true);
    EXPECT_EQ(G.SetOutCast({"output"}), true);

    Operation::ScopeInfo scope(5002);
    scope.allowParallelMerge = true;
    scope.allowCrossScopeMerge = false;
    SetScopeInfoForOps(G, {"SCOPE_HEAD", "SCOPE_TAIL"}, scope);

    Function* function = G.GetFunction();
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(100000, 20, 0), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);

    for (auto& op : function->Operations()) {
        EXPECT_GE(op.GetSubgraphID(), 0);
    }
    EXPECT_EQ(G.GetOp("SCOPE_HEAD")->GetSubgraphID(), G.GetOp("SCOPE_TAIL")->GetSubgraphID());
    EXPECT_EQ(G.GetOp("SCOPE_HEAD")->GetSubgraphID(), G.GetOp("BRIDGE")->GetSubgraphID());
}

TEST_F(GraphPartitionTest, TestUpdatePartitionResultInvalidSubgraphId)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape{16, 16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shape, {"input", "output"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_EXP, {"input"}, {"output"}, "EXP", true), true);
    EXPECT_EQ(G.SetInCast({"input"}), true);
    EXPECT_EQ(G.SetOutCast({"output"}), true);

    Function* function = G.GetFunction();
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.UpdatePartitionResult(*function), FAILED);
}

// cvmix
TEST_F(GraphPartitionTest, TestScopeCase6)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);

    ComputationalGraphBuilder G;
    ConstructGLMAttentionCase(G);

    Operation::ScopeInfo scope1;
    scope1.scopeId = 1;
    std::vector<std::string> vecOps = {"MULS_scale",     "ROWMAX_mij", "MAX_new",       "SUB_tsub",
                                       "EXP_pij",        "CAST_fp16",  "ROWSUM_local",  "SUB_tsub2",
                                       "EXP_update_mul", "MUL_tmp",    "ADD_sum_update"};
    SetScopeInfoForOps(G, vecOps, scope1);
    std::vector<std::string> cubeOps = {
        "VIEW_v_l1_0", "VIEW_v_l1_1", "VIEW_v_l1_2",        "VIEW_v_l1_3", "VIEW_a_l1_0", "VIEW_a_l1_1",
        "VIEW_a_l1_2", "VIEW_a_l1_3", "VIEW_L0B_v0",        "VIEW_L0B_v1", "VIEW_L0B_v2", "VIEW_L0B_v3",
        "VIEW_L0A_a0", "VIEW_L0A_a1", "VIEW_L0A_a2",        "VIEW_L0A_a3", "MMUL_a",      "MMACC_b",
        "MMACC_c",     "MMACC_d",     "ASSEMBLE_oi_tmp_ddr"};
    SetScopeInfoForOps(G, cubeOps, scope1);

    Function* function = G.GetFunction();
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(100000, 20, 0), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    for (const auto& name : cubeOps) {
        const auto& op = G.GetOp(name);
        EXPECT_EQ(op->HasAttr(OpAttributeKey::isCube), true);
        EXPECT_EQ(op->GetBoolAttribute(OpAttributeKey::isCube), true);
        EXPECT_EQ(op->GetCvFuseId(), 0);
    }
    std::unordered_set<std::string> cubeOpSet(cubeOps.begin(), cubeOps.end());
    std::unordered_set<std::string> vecOpSet(vecOps.begin(), vecOps.end());
    for (const auto& opPair : G.operations_) {
        if (cubeOpSet.count(opPair.first) == 0) {
            const auto& op = opPair.second;
            EXPECT_EQ(op->HasAttr(OpAttributeKey::isCube), true) << op->GetOpcodeStr() << op->GetOpMagic();
            EXPECT_EQ(op->GetBoolAttribute(OpAttributeKey::isCube), false) << op->GetOpcodeStr() << op->GetOpMagic();
            if (vecOpSet.count(opPair.first) != 0) {
                EXPECT_EQ(op->GetCvFuseId(), 0);
            }
        }
    }
}

void GetViewAssembleOnlySuperNodeGraph(ComputationalGraphBuilder& G)
{
    std::vector<int64_t> tileShape{16, 16};
    G.AddTensors(DataType::DT_FP32, tileShape,
                 {"t_in", "v_out1", "v_out2", "t_mid", "t_reshape_out", "t_view_in2", "t_view_out2", "t_mul_out",
                  "t_div_out", "a_out1", "v_out3", "v_out4", "a_out2", "a_out3", "t_final"});
    std::vector<Opcode> opCodes{Opcode::OP_VIEW,     Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE,
                                Opcode::OP_RESHAPE,  Opcode::OP_VIEW, Opcode::OP_MUL,      Opcode::OP_DIV,
                                Opcode::OP_ASSEMBLE, Opcode::OP_VIEW, Opcode::OP_VIEW,     Opcode::OP_ASSEMBLE,
                                Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{"t_in"},
                                                    {"t_in"},
                                                    {"v_out1"},
                                                    {"v_out2"},
                                                    {"t_mid"},
                                                    {"t_view_in2"},
                                                    {"t_view_out2", "t_reshape_out"},
                                                    {"t_mul_out"},
                                                    {"t_div_out"},
                                                    {"t_div_out"},
                                                    {"a_out1"},
                                                    {"v_out3"},
                                                    {"v_out4"}};
    std::vector<std::vector<std::string>> ooperands{
        {"v_out1"},    {"v_out2"}, {"t_mid"},  {"t_mid"},  {"t_reshape_out"}, {"t_view_out2"}, {"t_mul_out"},
        {"t_div_out"}, {"a_out1"}, {"v_out3"}, {"v_out4"}, {"t_final"},       {"t_final"}};
    std::vector<std::string> opNames{"VIEW1", "VIEW2",     "ASSEMBLE1", "ASSEMBLE2", "RESHAPE",   "VIEW3",    "MUL",
                                     "DIV",   "ASSEMBLE3", "VIEW4",     "VIEW5",     "ASSEMBLE4", "ASSEMBLE5"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t_in", "t_view_in2"}), true);
    EXPECT_EQ(G.SetOutCast({"t_final", "t_mul_out"}), true);
    Operation::ScopeInfo scope1(1);
    scope1.allowCrossScopeMerge = true;
    for (const auto& name : opNames) {
        G.GetOp(name)->SetScopeInfo(scope1);
    }
}

TEST_F(GraphPartitionTest, TestViewAssembleOnlySuperNodeScopeId)
{
    ComputationalGraphBuilder G;
    GetViewAssembleOnlySuperNodeGraph(G);
    Function* function = G.GetFunction();
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(100000, 20, 0), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), 1);
}

TEST_F(GraphPartitionTest, TestNonIsomorphismGraph)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"hin", "h2", "h3", "h41", "h42", "h5", "hout"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"hin"}, {"h2"}, "COPY_IN", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ABS, {"h2"}, {"h3"}, "ABS", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {"h3"}, {"h41"}, "ADDS1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MULS, {"h3"}, {"h42"}, "MULS2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MUL, {"h41", "h42"}, {"h5"}, "MUL", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"h5"}, {"hout"}, "COPY_OUT", true), true);
    EXPECT_EQ(G.SetInCast({"hin"}), true);
    EXPECT_EQ(G.SetOutCast({"hout"}), true);

    Function* function = G.GetFunction();
    const int parallelTH = 20;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestAvoidSuperNodeLoop)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"}), true);
    std::vector<Opcode> opCodes{Opcode::OP_A_MUL_B, Opcode::OP_A_MUL_B, Opcode::OP_A_MUL_B, Opcode::OP_A_MULACC_B};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t2", "t3"}, {"t4", "t5"}, {"t3", "t6", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t6"}, {"t8"}};
    std::vector<std::string> opNames{"MUL1", "MUL2", "MUL3", "MULACC"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2", "t5", "t7"}), true);
    EXPECT_EQ(G.SetOutCast({"t8"}), true);
    Function* function = G.GetFunction();
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(gpp.PostCheck(*function), SUCCESS);
}

TEST_F(GraphPartitionTest, TestBoundaryConvert)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<int64_t> tileShape{16, 16};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_L1,
                                           MemoryType::MEM_L0A};
    std::vector<Opcode> opCodes{Opcode::OP_MULS, Opcode::OP_CONVERT, Opcode::OP_L1_TO_L0A};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"muls", "convert", "L1ToL0A"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    G.GetOp("convert")->SetOpAttribute(std::make_shared<ConvertOpAttribute>(MemoryType::MEM_UB, MemoryType::MEM_L1));
    Function* function = G.GetFunction();
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(G.GetOp("muls")->GetSubgraphID(), G.GetOp("convert")->GetSubgraphID());
    EXPECT_NE(G.GetOp("L1ToL0A")->GetSubgraphID(), G.GetOp("convert")->GetSubgraphID());
}

void ConstructGraphForMatMulViewFormSuperNode(ComputationalGraphBuilder& G)
{
    // add tensor
    DataType dataType = DataType::DT_FP16;
    Shape shape = {16, 16};
    Shape viewShape{8, 16};
    std::vector<std::string> oriTensorNames{"matA1DDR", "matB1DDR", "matA1L1", "matB1L1",
                                            "matA1L0A", "matB1L0B", "matC1L0C"};
    std::vector<MemoryType> oriTensorMemoryType{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L1,
        MemoryType::MEM_L0A,        MemoryType::MEM_L0B,        MemoryType::MEM_L0C};
    EXPECT_EQ(G.AddTensors(dataType, shape, oriTensorMemoryType, oriTensorNames, 0), true);
    std::vector<std::string> afterViewTensorNames{"viewC1L0C", "outcast1"};
    std::vector<MemoryType> afterViewTensorMemoryType{MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(G.AddTensors(dataType, viewShape, afterViewTensorMemoryType, afterViewTensorNames, 0), true);
    // add operation
    std::vector<Opcode> opCodes{Opcode::OP_VIEW,    Opcode::OP_VIEW, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0B,
                                Opcode::OP_A_MUL_B, Opcode::OP_VIEW, Opcode::OP_ASSEMBLE};
    std::vector<std::string> opNames{"View1", "View2", "L1ToL0A1", "L1ToL0B1", "Mul1", "View3", "Assemble1"};
    std::vector<std::vector<std::string>> iOperands{
        {"matA1DDR"}, {"matB1DDR"}, {"matA1L1"}, {"matB1L1"}, {"matA1L0A", "matB1L0B"}, {"matC1L0C"}, {"viewC1L0C"}};
    std::vector<std::vector<std::string>> oOperands{{"matA1L1"},  {"matB1L1"},   {"matA1L0A"}, {"matB1L0B"},
                                                    {"matC1L0C"}, {"viewC1L0C"}, {"outcast1"}};
    EXPECT_EQ(G.AddOps(opCodes, iOperands, oOperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"matA1DDR", "matB1DDR"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast1"}), true);
}

TEST_F(GraphPartitionTest, TestMatMulViewFormSuperNode)
{
    ComputationalGraphBuilder G;
    ConstructGraphForMatMulViewFormSuperNode(G);

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);

    auto mulOp = G.GetOp("Mul1");
    auto viewOp = G.GetOp("View3");
    EXPECT_EQ(mulOp->GetSubgraphID(), viewOp->GetSubgraphID());
}

void ConstructGraphForMatMulMultipleViewSuccessors(ComputationalGraphBuilder& G)
{
    DataType dataType = DataType::DT_FP16;
    Shape shape = {16, 16};
    Shape viewShape{8, 16};
    std::vector<std::string> oriTensorNames{"matA3DDR", "matB3DDR", "matA3L1", "matB3L1",
                                            "matA3L0A", "matB3L0B", "matC3L0C"};
    std::vector<MemoryType> oriTensorMemoryType{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L1,
        MemoryType::MEM_L0A,        MemoryType::MEM_L0B,        MemoryType::MEM_L0C};
    EXPECT_EQ(G.AddTensors(dataType, shape, oriTensorMemoryType, oriTensorNames, 0), true);
    std::vector<std::string> afterViewTensorNames{"viewC3L0C_1", "viewC3L0C_2", "outcast3"};
    std::vector<MemoryType> afterViewTensorMemoryType{MemoryType::MEM_L0C, MemoryType::MEM_L0C,
                                                      MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(G.AddTensors(dataType, viewShape, afterViewTensorMemoryType, afterViewTensorNames, 0), true);
    std::vector<Opcode> opCodes{Opcode::OP_VIEW,    Opcode::OP_VIEW, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0B,
                                Opcode::OP_A_MUL_B, Opcode::OP_VIEW, Opcode::OP_VIEW,      Opcode::OP_ASSEMBLE};
    std::vector<std::string> opNames{"View1", "View2",     "L1ToL0A3",  "L1ToL0B3",
                                     "Mul3",  "ViewL0C_1", "ViewL0C_2", "Assemble3"};
    std::vector<std::vector<std::string>> iOperands{
        {"matA3DDR"}, {"matB3DDR"}, {"matA3L1"},    {"matB3L1"}, {"matA3L0A", "matB3L0B"},
        {"matC3L0C"}, {"matC3L0C"}, {"viewC3L0C_1"}};
    std::vector<std::vector<std::string>> oOperands{{"matA3L1"},  {"matB3L1"},     {"matA3L0A"},    {"matB3L0B"},
                                                    {"matC3L0C"}, {"viewC3L0C_1"}, {"viewC3L0C_2"}, {"outcast3"}};
    EXPECT_EQ(G.AddOps(opCodes, iOperands, oOperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"matA3DDR", "matB3DDR"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast3", "viewC3L0C_2"}), true);
}

TEST_F(GraphPartitionTest, TestMatMulMultipleViewSuccessors)
{
    ComputationalGraphBuilder G;
    ConstructGraphForMatMulMultipleViewSuccessors(G);

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);

    auto mulOp = G.GetOp("Mul3");
    auto viewL0C_1 = G.GetOp("ViewL0C_1");
    auto viewL0C_2 = G.GetOp("ViewL0C_2");
    EXPECT_EQ(mulOp->GetSubgraphID(), viewL0C_1->GetSubgraphID());
    EXPECT_EQ(mulOp->GetSubgraphID(), viewL0C_2->GetSubgraphID());
}

void ConstructGraphForMatMulViewNonL0C(ComputationalGraphBuilder& G)
{
    DataType dataType = DataType::DT_FP16;
    Shape shape = {16, 16};
    Shape viewShape{8, 16};
    std::vector<std::string> oriTensorNames{"matA4DDR", "matB4DDR", "matA4L1", "matB4L1",
                                            "matA4L0A", "matB4L0B", "matC4L0C"};
    std::vector<MemoryType> oriTensorMemoryType{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L1,
        MemoryType::MEM_L0A,        MemoryType::MEM_L0B,        MemoryType::MEM_L0C};
    EXPECT_EQ(G.AddTensors(dataType, shape, oriTensorMemoryType, oriTensorNames, 0), true);
    std::vector<std::string> afterViewTensorNames{"outcast1", "viewC4L1", "outcast2"};
    std::vector<MemoryType> afterViewTensorMemoryType{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1,
                                                      MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(G.AddTensors(dataType, viewShape, afterViewTensorMemoryType, afterViewTensorNames, 0), true);
    std::vector<Opcode> opCodes{Opcode::OP_VIEW,    Opcode::OP_VIEW, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0B,
                                Opcode::OP_A_MUL_B, Opcode::OP_VIEW, Opcode::OP_VIEW,      Opcode::OP_ASSEMBLE};
    std::vector<std::string> opNames{"View1", "View2",   "L1ToL0A4", "L1ToL0B4",
                                     "Mul4",  "ViewDDR", "ViewL1",   "Assemble4"};
    std::vector<std::vector<std::string>> iOperands{
        {"matA4DDR"}, {"matB4DDR"}, {"matA4L1"}, {"matB4L1"}, {"matA4L0A", "matB4L0B"},
        {"matC4L0C"}, {"matC4L0C"}, {"viewC4L1"}};
    std::vector<std::vector<std::string>> oOperands{{"matA4L1"},  {"matB4L1"},  {"matA4L0A"}, {"matB4L0B"},
                                                    {"matC4L0C"}, {"outcast1"}, {"viewC4L1"}, {"outcast2"}};
    EXPECT_EQ(G.AddOps(opCodes, iOperands, oOperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"matA4DDR", "matB4DDR"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast1", "outcast2"}), true);
}

TEST_F(GraphPartitionTest, TestMatMulViewNonL0C)
{
    ComputationalGraphBuilder G;
    ConstructGraphForMatMulViewNonL0C(G);

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);

    auto mulOp = G.GetOp("Mul4");
    auto viewDDROp = G.GetOp("ViewDDR");
    auto viewL1Op = G.GetOp("ViewL1");
    EXPECT_NE(mulOp->GetSubgraphID(), viewDDROp->GetSubgraphID());
    EXPECT_EQ(mulOp->GetSubgraphID(), viewL1Op->GetSubgraphID());
}

TEST_F(GraphPartitionTest, TestViewAssembleScopeId)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<int64_t> tileShape{16, 16};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_MULS, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE,
                                Opcode::OP_VIEW, Opcode::OP_VIEW,     Opcode::OP_MULS};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"m1", "a1", "a2", "v1", "v2", "m2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetOutCast({"t3", "t4", "t7"}), true);
    G.GetOp("m1")->SetScopeId(0);
    G.GetOp("m2")->SetScopeId(1);
    Function* function = G.GetFunction();
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(G.GetOp("a1")->GetScopeId(), 0);
    EXPECT_EQ(G.GetOp("a2")->GetScopeId(), 0);
    EXPECT_EQ(G.GetOp("v1")->GetScopeId(), 1);
    EXPECT_EQ(G.GetOp("v2")->GetScopeId(), 1);
}

/**
 * Test UB-to-UB VIEW with dynamic offset should not be mergeable
 *
 * Graph structure:
 *   input_ub[MEM_UB, (32, 16)] --> VIEW[fromOffset=(0,0), dynOffset=(dynamic_offset_dim0, 0)] --> output_ub[MEM_UB,
 * (32, 16)]
 *
 * Expected behavior:
 *   Node containing UB-to-UB VIEW with dynamic offset should be marked as not mergeable
 */
TEST_F(GraphPartitionTest, TestUbToUbViewWithDynOffsetNotMergeable)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape = {32, 16};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<std::string> tensorNames{"inputUb", "outputUb"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shape, tensorMemTypes, tensorNames, 0), true);

    std::vector<std::vector<std::string>> iOperands{{"inputUb"}};
    std::vector<std::vector<std::string>> oOperands{{"outputUb"}};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::string> opNames{"viewOp"};
    EXPECT_EQ(G.AddOps(opCodes, iOperands, oOperands, opNames, true), true);

    auto viewOp = G.GetOp("viewOp");
    ASSERT_NE(viewOp, nullptr);

    std::vector<SymbolicScalar> dynOffset = {CreateTestScalarVar("dynamicOffsetDim0"), IRBuilder().CreateConstInt(0)};
    auto viewAttr = std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0}, dynOffset,
        std::vector<SymbolicScalar>{IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(16)});
    viewOp->SetOpAttribute(viewAttr);

    Function* function = G.GetFunction();
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(10, 10, false), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);

    int opMagic = viewOp->GetOpMagic();
    int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
    int nodeIdx = partitioner.superNodeInfo_->op2Node_[opIdx];
    bool mergeable = partitioner.superNodeInfo_->GetNodeMergeable(partitioner.operationInfo_, nodeIdx);

    EXPECT_EQ(mergeable, false);
}

/**
 * Test UB-to-UB VIEW without dynamic offset should be mergeable
 *
 * Graph structure:
 *   inputUb[MEM_UB, (32, 16)] --> VIEW[fromOffset=(0,0), dynOffset=empty] --> outputUb[MEM_UB, (32, 16)]
 *
 * Expected behavior:
 *   Node containing UB-to-UB VIEW without dynamic offset should be marked as mergeable
 */
TEST_F(GraphPartitionTest, TestUbToUbViewWithoutDynOffsetMergeable)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape = {32, 16};
    std::vector<std::string> tensorNames{"inputUb", "outputUb"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shape, tensorMemTypes, tensorNames, 0), true);

    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> iOperands{{"inputUb"}};
    std::vector<std::vector<std::string>> oOperands{{"outputUb"}};
    std::vector<std::string> opNames{"viewOp"};
    EXPECT_EQ(G.AddOps(opCodes, iOperands, oOperands, opNames, true), true);

    auto viewOp = G.GetOp("viewOp");
    ASSERT_NE(viewOp, nullptr);

    auto viewAttr = std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{},
        std::vector<SymbolicScalar>{IRBuilder().CreateConstInt(32), IRBuilder().CreateConstInt(16)});
    viewOp->SetOpAttribute(viewAttr);

    Function* function = G.GetFunction();
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(10, 10, false), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);

    int opMagic = viewOp->GetOpMagic();
    int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
    int nodeIdx = partitioner.superNodeInfo_->op2Node_[opIdx];
    bool mergeable = partitioner.superNodeInfo_->GetNodeMergeable(partitioner.operationInfo_, nodeIdx);

    EXPECT_EQ(mergeable, true);
}

/**
 * Test UB-to-UB ASSEMBLE with dynamic offset should not be mergeable
 *
 * Graph structure:
 *   inputUb[MEM_UB, (32, 16)] --> ASSEMBLE[toOffset=(0,0), toDynOffset=(dynamicOffsetDim0,0)] --> outputUb[MEM_UB, (32,
 * 16)]
 *
 * Expected behavior:
 *   Node containing UB-to-UB ASSEMBLE with dynamic offset should be marked as not mergeable
 */
TEST_F(GraphPartitionTest, TestUbToUbAssembleWithDynOffsetNotMergeable)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape = {32, 16};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<std::string> tensorNames{"inputUb", "outputUb"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shape, tensorMemTypes, tensorNames, 0), true);

    std::vector<std::vector<std::string>> iOperands{{"inputUb"}};
    std::vector<std::vector<std::string>> oOperands{{"outputUb"}};
    std::vector<Opcode> opCodes{Opcode::OP_ASSEMBLE};
    std::vector<std::string> opNames{"assembleOp"};
    EXPECT_EQ(G.AddOps(opCodes, iOperands, oOperands, opNames, true), true);

    auto assembleOp = G.GetOp("assembleOp");
    ASSERT_NE(assembleOp, nullptr);

    std::vector<SymbolicScalar> toDynOffset = {CreateTestScalarVar("dynamicOffsetDim0"), IRBuilder().CreateConstInt(0)};
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}, toDynOffset);
    assembleOp->SetOpAttribute(assembleAttr);

    Function* function = G.GetFunction();
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(10, 10, false), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);

    int opMagic = assembleOp->GetOpMagic();
    int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
    int nodeIdx = partitioner.superNodeInfo_->op2Node_[opIdx];
    bool mergeable = partitioner.superNodeInfo_->GetNodeMergeable(partitioner.operationInfo_, nodeIdx);

    EXPECT_EQ(mergeable, false);
}

/**
 * Test UB-to-UB ASSEMBLE without dynamic offset should be mergeable
 *
 * Graph structure:
 *   inputUb[MEM_UB, (32, 16)] --> ASSEMBLE[toOffset=(0,0), toDynOffset=empty] --> outputUb[MEM_UB, (32, 16)]
 *
 * Expected behavior:
 *   Node containing UB-to-UB ASSEMBLE without dynamic offset should be marked as mergeable
 */
TEST_F(GraphPartitionTest, TestUbToUbAssembleWithoutDynOffsetMergeable)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape = {32, 16};
    std::vector<std::string> tensorNames{"inputUb", "outputUb"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shape, tensorMemTypes, tensorNames, 0), true);

    std::vector<Opcode> opCodes{Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> iOperands{{"inputUb"}};
    std::vector<std::vector<std::string>> oOperands{{"outputUb"}};
    std::vector<std::string> opNames{"assembleOp"};
    EXPECT_EQ(G.AddOps(opCodes, iOperands, oOperands, opNames, true), true);

    auto assembleOp = G.GetOp("assembleOp");
    ASSERT_NE(assembleOp, nullptr);

    auto assembleAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0},
                                                              std::vector<SymbolicScalar>{});
    assembleOp->SetOpAttribute(assembleAttr);

    Function* function = G.GetFunction();
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(10, 10, false), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);

    int opMagic = assembleOp->GetOpMagic();
    int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
    int nodeIdx = partitioner.superNodeInfo_->op2Node_[opIdx];
    bool mergeable = partitioner.superNodeInfo_->GetNodeMergeable(partitioner.operationInfo_, nodeIdx);

    EXPECT_EQ(mergeable, true);
}

/**
 * Test case for PostCheck: VIEW op must not be the end node of a subgraph.
 *
 * After L1CopyInCombine, a L0C→L1 VIEW should be merged with its consumer ops
 * (not just producers), ensuring the VIEW's output tensor has at least one
 * consumer in the same subgraph.
 *
 * This graph contains multiple L0C→L1 VIEWs (VIEW_c1, VIEW_c2, VIEW_e1, VIEW_e2)
 * whose outputs feed into L1_TO_L0A ops. The PostCheck verifies that each VIEW
 * and its successor L1_TO_L0A are placed in the same subgraph.
 *
 * Graph structure:
 *   Branch1: a1/b1 DDR→L1→L0A/L0B → MUL + 2xMULACC → acc2_l0c
 *   Branch2: a2/b2 DDR→L1→L0A/L0B → MUL + 2xMULACC → acc2_l0c
 *   Shared s: s_ddr→VIEW→L1 → fan-out to both branches' L0B
 *   Stage2: acc2→VIEW→L1→L0A → MUL(s_l0b) → e1/e2_l0c
 *   Reduce: e1/e2_l0c→VIEW→UB → ADD → ASSEMBLE→out_ddr
 */
static void AddBranch1Matmul(ComputationalGraphBuilder& G, DataType dtype, const Shape& shape)
{
    std::vector<std::string> a1Names{"a1_ddr", "a1_l1", "a1_l0a"};
    std::vector<MemoryType> a1Mems{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0A};
    EXPECT_EQ(G.AddTensors(dtype, shape, a1Mems, a1Names, 0), true);
    std::vector<std::string> b1Names{"b1_ddr", "b1_l1", "b1_l0b"};
    std::vector<MemoryType> b1Mems{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0B};
    EXPECT_EQ(G.AddTensors(dtype, shape, b1Mems, b1Names, 0), true);

    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"a1_ddr"}, {"a1_l1"}, "VIEW_a1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"b1_ddr"}, {"b1_l1"}, "VIEW_b1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0A, {"a1_l1"}, {"a1_l0a"}, "L1TO0A_a1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0B, {"b1_l1"}, {"b1_l0b"}, "L1TO0B_b1", true), true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0C, "br1_acc0_l0c"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_A_MUL_B, {"a1_l0a", "b1_l0b"}, {"br1_acc0_l0c"}, "MUL1", true), true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0A, "br1_a1_l0a"), true);
    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0B, "br1_b1_l0b"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0A, {"a1_l1"}, {"br1_a1_l0a"}, "L1TO0A_a1_iter2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0B, {"b1_l1"}, {"br1_b1_l0b"}, "L1TO0B_b1_iter2", true), true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0C, "br1_acc1_l0c"), true);
    EXPECT_EQ(
        G.AddOp(Opcode::OP_A_MULACC_B, {"br1_a1_l0a", "br1_b1_l0b", "br1_acc0_l0c"}, {"br1_acc1_l0c"}, "MULACC1", true),
        true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0A, "br1_a2_l0a"), true);
    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0B, "br1_b2_l0b"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0A, {"a1_l1"}, {"br1_a2_l0a"}, "L1TO0A_a1_iter3", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0B, {"b1_l1"}, {"br1_b2_l0b"}, "L1TO0B_b1_iter3", true), true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0C, "br1_acc2_l0c"), true);
    EXPECT_EQ(
        G.AddOp(Opcode::OP_A_MULACC_B, {"br1_a2_l0a", "br1_b2_l0b", "br1_acc1_l0c"}, {"br1_acc2_l0c"}, "MULACC2", true),
        true);
}

static void AddBranch2Matmul(ComputationalGraphBuilder& G, DataType dtype, const Shape& shape)
{
    std::vector<std::string> a2Names{"a2_ddr", "a2_l1", "a2_l0a"};
    std::vector<MemoryType> a2Mems{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0A};
    EXPECT_EQ(G.AddTensors(dtype, shape, a2Mems, a2Names, 0), true);
    std::vector<std::string> b2Names{"b2_ddr", "b2_l1", "b2_l0b"};
    std::vector<MemoryType> b2Mems{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0B};
    EXPECT_EQ(G.AddTensors(dtype, shape, b2Mems, b2Names, 0), true);

    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"a2_ddr"}, {"a2_l1"}, "VIEW_a2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"b2_ddr"}, {"b2_l1"}, "VIEW_b2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0A, {"a2_l1"}, {"a2_l0a"}, "L1TO0A_a2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0B, {"b2_l1"}, {"b2_l0b"}, "L1TO0B_b2", true), true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0C, "br2_acc0_l0c"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_A_MUL_B, {"a2_l0a", "b2_l0b"}, {"br2_acc0_l0c"}, "MUL3", true), true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0A, "br2_a1_l0a"), true);
    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0B, "br2_b1_l0b"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0A, {"a2_l1"}, {"br2_a1_l0a"}, "L1TO0A_a2_iter2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0B, {"b2_l1"}, {"br2_b1_l0b"}, "L1TO0B_b2_iter2", true), true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0C, "br2_acc1_l0c"), true);
    EXPECT_EQ(
        G.AddOp(Opcode::OP_A_MULACC_B, {"br2_a1_l0a", "br2_b1_l0b", "br2_acc0_l0c"}, {"br2_acc1_l0c"}, "MULACC3", true),
        true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0A, "br2_a2_l0a"), true);
    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0B, "br2_b2_l0b"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0A, {"a2_l1"}, {"br2_a2_l0a"}, "L1TO0A_a2_iter3", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0B, {"b2_l1"}, {"br2_b2_l0b"}, "L1TO0B_b2_iter3", true), true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0C, "br2_acc2_l0c"), true);
    EXPECT_EQ(
        G.AddOp(Opcode::OP_A_MULACC_B, {"br2_a2_l0a", "br2_b2_l0b", "br2_acc1_l0c"}, {"br2_acc2_l0c"}, "MULACC4", true),
        true);
}

static void AddSharedAndReduce(ComputationalGraphBuilder& G, DataType dtype, const Shape& shape)
{
    std::vector<std::string> sNames{"s_ddr", "s_l1"};
    std::vector<MemoryType> sMems{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1};
    EXPECT_EQ(G.AddTensors(dtype, shape, sMems, sNames, 0), true);
    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0B, "s_l0b_1"), true);
    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0B, "s_l0b_2"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"s_ddr"}, {"s_l1"}, "VIEW_s", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0B, {"s_l1"}, {"s_l0b_1"}, "L1TO0B_s1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0B, {"s_l1"}, {"s_l0b_2"}, "L1TO0B_s2", true), true);

    std::vector<std::string> c1Names{"c1_l1_new", "c1_l0a_new"};
    std::vector<MemoryType> c1Mems{MemoryType::MEM_L1, MemoryType::MEM_L0A};
    EXPECT_EQ(G.AddTensors(dtype, shape, c1Mems, c1Names, 0), true);
    std::vector<std::string> c2Names{"c2_l1_new", "c2_l0a_new"};
    std::vector<MemoryType> c2Mems{MemoryType::MEM_L1, MemoryType::MEM_L0A};
    EXPECT_EQ(G.AddTensors(dtype, shape, c2Mems, c2Names, 0), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"br1_acc2_l0c"}, {"c1_l1_new"}, "VIEW_c1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"br2_acc2_l0c"}, {"c2_l1_new"}, "VIEW_c2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0A, {"c1_l1_new"}, {"c1_l0a_new"}, "L1TO0A_c1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0A, {"c2_l1_new"}, {"c2_l0a_new"}, "L1TO0A_c2", true), true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0C, "e1_l0c"), true);
    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_L0C, "e2_l0c"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_A_MUL_B, {"c1_l0a_new", "s_l0b_1"}, {"e1_l0c"}, "MUL5", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_A_MUL_B, {"c2_l0a_new", "s_l0b_2"}, {"e2_l0c"}, "MUL6", true), true);

    std::vector<std::string> addNames{"e1_ub", "e2_ub", "add_ub"};
    EXPECT_EQ(G.AddTensors(dtype, shape, addNames), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"e1_l0c"}, {"e1_ub"}, "VIEW_e1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {"e2_l0c"}, {"e2_ub"}, "VIEW_e2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADD, {"e1_ub", "e2_ub"}, {"add_ub"}, "ADD1", true), true);

    EXPECT_EQ(G.AddTensor(dtype, shape, MemoryType::MEM_DEVICE_DDR, "out_ddr"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"add_ub"}, {"out_ddr"}, "ASSEMBLE_out", true), true);
}

void GetDualBranchMatmulGraph(ComputationalGraphBuilder& G)
{
    DataType dtype = DataType::DT_FP16;
    Shape shape{32, 32};

    AddBranch1Matmul(G, dtype, shape);
    AddBranch2Matmul(G, dtype, shape);
    AddSharedAndReduce(G, dtype, shape);

    EXPECT_EQ(G.SetInCast({"a1_ddr", "b1_ddr", "a2_ddr", "b2_ddr", "s_ddr"}), true);
    EXPECT_EQ(G.SetOutCast({"out_ddr"}), true);
}

static bool HasConsumerInSameSubgraph(const Operation* op, const std::shared_ptr<LogicalTensor>& oTensor)
{
    for (auto& consumer : oTensor->GetConsumers()) {
        if (consumer->GetSubgraphID() == op->GetSubgraphID()) {
            return true;
        }
    }
    return false;
}

static void VerifyViewNotEndNode(const std::unordered_map<std::string, Operation*>& operations)
{
    for (auto& opPair : operations) {
        Operation* op = opPair.second;
        EXPECT_NE(op, nullptr);
        EXPECT_GE(op->GetSubgraphID(), 0);
        if (!IsViewLike(op->GetOpcode())) {
            continue;
        }
        for (auto oTensor : op->GetOOperands()) {
            EXPECT_TRUE(HasConsumerInSameSubgraph(op, oTensor))
                << "View-like op " << opPair.first << " (opmagic: " << op->GetOpMagic() << ") is end node in subgraph "
                << op->GetSubgraphID();
        }
    }
}

TEST_F(GraphPartitionTest, TestDualBranchMatmulPostCheck)
{
    ComputationalGraphBuilder G;
    GetDualBranchMatmulGraph(G);
    Function* function = G.GetFunction();
    GraphPartition gpp;
    EXPECT_EQ(gpp.PreCheck(*function), SUCCESS);
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(gpp.PostCheck(*function), SUCCESS);
    VerifyViewNotEndNode(G.operations_);
}

/**
 * sum_update path: 4 SUB→EXP chains share a max tensor. Each EXP feeds CAST and
 * PAIRSUM. Chains 0-2 create CAST before PAIRSUM; chain 3 creates PAIRSUM before
 * CAST, reversing the consumer index order in nodeOutGraph_.
 */
void GetSumUpdatePathGraph(ComputationalGraphBuilder& G)
{
    std::vector<int64_t> shape{16, 16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shape,
                           {"mx", "d0", "d1", "d2", "d3", "s0", "s1", "s2", "s3", "e0", "e1", "e2", "e3", "c0", "c1",
                            "c2", "c3", "p01", "p23"}),
              true);

    EXPECT_EQ(G.AddOp(Opcode::OP_SUB, {"d0", "mx"}, {"s0"}, "SUB0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_EXP, {"s0"}, {"e0"}, "EXP0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"e0"}, {"c0"}, "CAST0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_SUB, {"d1", "mx"}, {"s1"}, "SUB1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_EXP, {"s1"}, {"e1"}, "EXP1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"e1"}, {"c1"}, "CAST1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_PAIRSUM, {"e0", "e1"}, {"p01"}, "PS01", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_SUB, {"d2", "mx"}, {"s2"}, "SUB2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_EXP, {"s2"}, {"e2"}, "EXP2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"e2"}, {"c2"}, "CAST2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_SUB, {"d3", "mx"}, {"s3"}, "SUB3", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_EXP, {"s3"}, {"e3"}, "EXP3", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_PAIRSUM, {"e2", "e3"}, {"p23"}, "PS23", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"e3"}, {"c3"}, "CAST3", true), true);

    EXPECT_EQ(G.SetInCast({"mx", "d0", "d1", "d2", "d3"}), true);
    EXPECT_EQ(G.SetOutCast({"c0", "c1", "c2", "c3", "p01", "p23"}), true);
}

TEST_F(GraphPartitionTest, TestSumUpdatePathMerge)
{
    ComputationalGraphBuilder G;
    GetSumUpdatePathGraph(G);
    Function* function = G.GetFunction();
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(20, 100000, true), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);

    // All EXPs get the same supernode hash despite different consumer order
    int e0Node = partitioner.superNodeInfo_
                     ->op2Node_[partitioner.operationInfo_->magic2Idx_[G.GetOp("EXP0")->GetOpMagic()]];
    int e3Node = partitioner.superNodeInfo_
                     ->op2Node_[partitioner.operationInfo_->magic2Idx_[G.GetOp("EXP3")->GetOpMagic()]];
    EXPECT_EQ(partitioner.superNodeInfo_->nodeHashList_[e0Node], partitioner.superNodeInfo_->nodeHashList_[e3Node]);

    // PAIRSUMs merge with their EXP predecessors into the same subgraph
    EXPECT_EQ(G.GetOp("PS01")->GetSubgraphID(), G.GetOp("EXP0")->GetSubgraphID());
    EXPECT_EQ(G.GetOp("PS23")->GetSubgraphID(), G.GetOp("EXP3")->GetSubgraphID());
}

TEST_F(GraphPartitionTest, TestIsViewLikeAndIsAssembleLikeCompatibility)
{
    EXPECT_TRUE(IsViewLike(Opcode::OP_VIEW));
    EXPECT_TRUE(IsViewLike(Opcode::OP_SLICE));
    EXPECT_FALSE(IsViewLike(Opcode::OP_ASSEMBLE));
    EXPECT_FALSE(IsViewLike(Opcode::OP_CONTRACT));
    EXPECT_TRUE(IsAssembleLike(Opcode::OP_ASSEMBLE));
    EXPECT_TRUE(IsAssembleLike(Opcode::OP_CONTRACT));
    EXPECT_FALSE(IsAssembleLike(Opcode::OP_VIEW));
    EXPECT_FALSE(IsAssembleLike(Opcode::OP_SLICE));
}

void GetUbToUbTransposeCubeGraph(ComputationalGraphBuilder& G)
{
    std::vector<int64_t> tileShape{16, 16};
    std::vector<std::string> tensorNames{"t_in",   "t_ub1", "t_ub2", "t_l1_a", "t_l0a",
                                         "t_l1_b", "t_l0b", "t_l0c", "t_out"};
    std::vector<MemoryType> tensorMemType{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,  MemoryType::MEM_UB,
                                          MemoryType::MEM_L1,         MemoryType::MEM_L0A, MemoryType::MEM_L1,
                                          MemoryType::MEM_L0B,        MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorMemType, tensorNames), true);

    std::vector<Opcode> opCodes{
        Opcode::OP_COPY_IN, Opcode::OP_TRANSPOSE_VNCHWCONV, Opcode::OP_VIEW,    Opcode::OP_L1_TO_L0A,
        Opcode::OP_VIEW,    Opcode::OP_L1_TO_L0B,           Opcode::OP_A_MUL_B, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t_in"},  {"t_ub1"},  {"t_ub2"},          {"t_l1_a"},
                                                    {"t_ub2"}, {"t_l1_b"}, {"t_l0a", "t_l0b"}, {"t_l0c"}};
    std::vector<std::vector<std::string>> ooperands{{"t_ub1"},  {"t_ub2"}, {"t_l1_a"}, {"t_l0a"},
                                                    {"t_l1_b"}, {"t_l0b"}, {"t_l0c"},  {"t_out"}};
    std::vector<std::string> opNames{"COPY_IN",         "TRANSPOSE_UB", "VIEW_UB_TO_L1_A", "L1_TO_L0A",
                                     "VIEW_UB_TO_L1_B", "L1_TO_L0B",    "MATMUL",          "COPY_OUT"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t_in"}), true);
    EXPECT_EQ(G.SetOutCast({"t_out"}), true);
}

TEST_F(GraphPartitionTest, TestUbToUbTransposeNotMergeWithCube)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);

    ComputationalGraphBuilder G;
    GetUbToUbTransposeCubeGraph(G);
    Function* function = G.GetFunction();

    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(1, 100000, false), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);

    Operation* transposeOp = G.GetOp("TRANSPOSE_UB");
    EXPECT_NE(transposeOp, nullptr);
    EXPECT_EQ(transposeOp->HasAttr(OpAttributeKey::isCube), true);
    EXPECT_EQ(transposeOp->GetBoolAttribute(OpAttributeKey::isCube), false);

    Operation* matmulOp = G.GetOp("MATMUL");
    EXPECT_NE(matmulOp, nullptr);
    EXPECT_EQ(matmulOp->HasAttr(OpAttributeKey::isCube), true);
    EXPECT_EQ(matmulOp->GetBoolAttribute(OpAttributeKey::isCube), true);

    EXPECT_NE(transposeOp->GetSubgraphID(), matmulOp->GetSubgraphID());
}

} // namespace tile_fwk
} // namespace npu
