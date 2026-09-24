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
 * \file test_l1_copy_reuse.cpp
 * \brief Unit test for L1CopyInReuseMerge pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include <fstream>
#include <set>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/graph_partition/graph_partition.h"
#include "passes/tile_graph_pass/graph_partition/l1_copy_reuse.h"
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;
namespace npu {
namespace tile_fwk {
class L1CopyInReuseTest : public testing::Test {
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

TEST_F(L1CopyInReuseTest, TwoCopyIn)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestL1CopyInReuse", "TestL1CopyInReuse",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    incast1->tensor->rawmagic = 1;
    incast1->memoryTypeToBe_ = MEM_DEVICE_DDR;
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    incast2->tensor->rawmagic = 1;
    incast2->memoryTypeOriginal_ = MEM_DEVICE_DDR;
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->memoryTypeOriginal_ = MEM_L1;
    tensor1->tensor->rawmagic = 2;
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->memoryTypeOriginal_ = MEM_L1;
    tensor3->tensor->rawmagic = 3;
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& copy_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SLICE, {incast1}, {tensor1});
    copy_op1.UpdateSubgraphID(subGraphID0);
    copy_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0}, MEM_L1, std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>()));
    auto& copy_out1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L1_TO_L0A, {tensor1}, {tensor2});
    copy_out1.UpdateSubgraphID(subGraphID0);

    auto& view_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast1}, {incast2});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op1.UpdateSubgraphID(subGraphID1);
    auto& alloc_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L1_ALLOC, {}, {tensor3});
    alloc_op1.UpdateSubgraphID(subGraphID1);
    auto& copy_op2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SLICE, {incast2}, {tensor3});
    copy_op2.UpdateSubgraphID(subGraphID1);
    incast2->AddConsumer(copy_op2);
    copy_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0}, MEM_L1, std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>()));
    auto& copy_out2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L1_TO_L0A, {tensor3}, {tensor4});
    copy_out2.UpdateSubgraphID(subGraphID1);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(tensor2);
    currFunctionPtr->outCasts_.push_back(tensor4);

    // Call the pass
    L1CopyInReuseMerge pass;
    pass.PreCheck(*currFunctionPtr);
    pass.RunOnFunction(*currFunctionPtr);
    pass.PostCheck(*currFunctionPtr);
}

void InitGraphBuilder(ComputationalGraphBuilder& G, std::vector<int64_t> tileShape, const int subGraphNum)
{
    auto shapeImme = OpImmediate::Specified(tileShape);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_VIEW}, {{"incast0"}}, {{"incast1"}}, {"view"}, true), true);
    G.GetOp("view")->UpdateSubgraphID(0);
    G.GetTensor("incast1")->tensor->rawmagic = 1;
    G.GetTensor("incast1")->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_SLICE, Opcode::OP_EXP};
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"SLICE_" + strID, "EXP_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("SLICE_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("SLICE_" + strID)
            ->SetOpAttribute(std::make_shared<ViewOpAttribute>(
                std::vector<int64_t>{0, 0}, MEM_L1, std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>()));
        G.GetTensor("tensor" + strID)->SetMemoryTypeOriginal(MEM_L1);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
}

void BuildParallelAssembleSource(ComputationalGraphBuilder& G, const std::string& prefix, int branchNum,
                                 std::vector<std::string>& gmTensors)
{
    std::vector<int64_t> tileShape{16, 16};
    std::string input = prefix + "In";
    std::string viewUb = prefix + "ViewUb";
    std::string cast0Ub = prefix + "Cast0Ub";
    std::string addsUb = prefix + "AddsUb";
    std::string cast1Ub = prefix + "Cast1Ub";

    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_DEVICE_DDR, input), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_UB, viewUb), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_UB, cast0Ub), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_UB, addsUb), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_UB, cast1Ub), true);

    EXPECT_EQ(G.AddOp(Opcode::OP_VIEW, {input}, {viewUb}, prefix + "View"), true);
    auto viewOp = G.GetOp(prefix + "View");
    viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0}, MEM_UB, std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>()));

    EXPECT_EQ(G.AddOp(Opcode::OP_CONVERT, {viewUb}, {cast0Ub}, prefix + "Cast0"), true);

    EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {cast0Ub}, {addsUb}, prefix + "Adds"), true);

    EXPECT_EQ(G.AddOp(Opcode::OP_CONVERT, {addsUb}, {cast1Ub}, prefix + "Cast1"), true);

    for (int i = 0; i < branchNum; ++i) {
        std::string branchId = std::to_string(i);
        std::string gmTensor = prefix + "Gm" + branchId;
        std::string assembleName = prefix + "Assemble" + branchId;
        EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_DEVICE_DDR, gmTensor), true);
        EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {cast1Ub}, {gmTensor}, assembleName), true);
        auto assembleOp = G.GetOp(assembleName);
        assembleOp->SetOpAttribute(
            std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{0, 0}));
        gmTensors.push_back(gmTensor);
    }
}

void BuildParallelMatmulBranch(ComputationalGraphBuilder& G, const std::vector<std::string>& gmTensorsA,
                               const std::vector<std::string>& gmTensorsB, int branchIndex,
                               std::vector<std::string>& outcasts)
{
    std::vector<int64_t> tileShape{16, 16};
    std::string branchId = std::to_string(branchIndex);
    std::string viewA = "viewA" + branchId;
    std::string viewB = "viewB" + branchId;
    std::string l1A = "l1A" + branchId;
    std::string l1B = "l1B" + branchId;
    std::string l0A = "l0A" + branchId;
    std::string l0B = "l0B" + branchId;
    std::string l0C = "l0C" + branchId;
    std::string matmul = "matmul" + branchId;
    std::string copyOut = "copyOut" + branchId;
    std::string out = "out" + branchId;

    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_L1, l1A), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_L1, l1B), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_L0A, l0A), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_L0B, l0B), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_L0C, l0C), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_DEVICE_DDR, out), true);

    EXPECT_EQ(G.AddOp(Opcode::OP_SLICE, {gmTensorsA[branchIndex]}, {l1A}, viewA), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_SLICE, {gmTensorsB[branchIndex]}, {l1B}, viewB), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0A, {l1A}, {l0A}, "toA" + branchId), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0B, {l1B}, {l0B}, "toB" + branchId), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_A_MUL_B, {l0A, l0B}, {l0C}, matmul), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {l0C}, {out}, copyOut), true);

    auto viewAOp = G.GetOp(viewA);
    auto viewBOp = G.GetOp(viewB);
    viewAOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0}, MEM_L1, std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>()));
    viewBOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0}, MEM_L1, std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>()));

    auto toAOp = G.GetOp("toA" + branchId);
    auto toBOp = G.GetOp("toB" + branchId);
    auto matmulOp = G.GetOp(matmul);
    auto copyOutOp = G.GetOp(copyOut);

    viewAOp->SetAttr(OpAttributeKey::isCube, true);
    viewBOp->SetAttr(OpAttributeKey::isCube, true);
    toAOp->SetAttr(OpAttributeKey::isCube, true);
    toBOp->SetAttr(OpAttributeKey::isCube, true);
    matmulOp->SetAttr(OpAttributeKey::isCube, true);
    copyOutOp->SetAttr(OpAttributeKey::isCube, true);

    outcasts.push_back(out);
}

void BuildParallelMatmulGraph(ComputationalGraphBuilder& G, int branchNum)
{
    std::vector<std::string> gmTensorsA;
    std::vector<std::string> gmTensorsB;
    BuildParallelAssembleSource(G, "srcA", branchNum, gmTensorsA);
    BuildParallelAssembleSource(G, "srcB", branchNum, gmTensorsB);

    std::vector<std::string> outcasts;
    for (int i = 0; i < branchNum; ++i) {
        BuildParallelMatmulBranch(G, gmTensorsA, gmTensorsB, i, outcasts);
    }

    EXPECT_EQ(G.SetInCast({"srcAIn", "srcBIn"}), true);
    EXPECT_EQ(G.SetOutCast(outcasts), true);
}

std::set<int> GetTensorRawMagics(ComputationalGraphBuilder& G, const std::string& prefix, int branchNum)
{
    std::set<int> rawMagics;
    for (int i = 0; i < branchNum; ++i) {
        rawMagics.insert(G.GetTensor(prefix + "Gm" + std::to_string(i))->GetRawMagic());
    }
    return rawMagics;
}

int CountOpcode(Function& function, Opcode opcode)
{
    int count = 0;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == opcode) {
            ++count;
        }
    }
    return count;
}

std::set<int> GetMatmulSubgraphIds(ComputationalGraphBuilder& G, int branchNum)
{
    std::set<int> subgraphIds;
    for (int i = 0; i < branchNum; ++i) {
        subgraphIds.insert(G.GetOp("matmul" + std::to_string(i))->GetSubgraphID());
    }
    return subgraphIds;
}

TEST_F(L1CopyInReuseTest, TestInvalidOp)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensorL1"}), true);
    G.GetTensor("tensorL1")->SetMemoryTypeOriginal(MEM_L1);
    EXPECT_EQ(G.AddOps({Opcode::OP_GATHER_IN_L1}, {{"incast1"}}, {{"tensorL1"}}, {"gather_in_l1"}, true), true);
    G.GetOp("gather_in_l1")->UpdateSubgraphID(1);
    Function* function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, 4}};
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}, {-1, 2}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
}

TEST_F(L1CopyInReuseTest, TestNormal)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int result = 5;
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function* function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, 4}};
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}, {-1, 2}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

/*
GraphPartitionCoeThenL1ReuseMerge
    srcAIn -> view -> cast -> adds -> cast -> assemble0/1/2/3 -> srcAGm0/1/2/3 -> view(L1) -> l0A -> matmul
    srcBIn -> view -> cast -> adds -> cast -> assemble0/1/2/3 -> srcBGm0/1/2/3 -> view(L1) -> l0B -> matmul

GraphPartition 内部新增的 COE 会删除冗余 assemble，只保留一份共享的 GM 输出；
随后 L1CopyInReuseMerge 基于统一后的 GM->L1 view，将多路并行 matmul 合并到同一子图。
*/
TEST_F(L1CopyInReuseTest, TestParallelAssembleEnableParallelMatmulL1ReuseMerge)
{
    constexpr int branchNum = 4;

    auto func = std::make_shared<Function>(Program::GetInstance(), "TestGraphPartitionCoeL1Reuse",
                                           "TestGraphPartitionCoeL1Reuse", nullptr);
    ComputationalGraphBuilder graph(func.get());
    BuildParallelMatmulGraph(graph, branchNum);

    auto rawMagicsABefore = GetTensorRawMagics(graph, "srcA", branchNum);
    auto rawMagicsBBefore = GetTensorRawMagics(graph, "srcB", branchNum);
    EXPECT_EQ(rawMagicsABefore.size(), static_cast<size_t>(branchNum));
    EXPECT_EQ(rawMagicsBBefore.size(), static_cast<size_t>(branchNum));

    Function* function = graph.GetFunction();
    function->paramConfigs_.sgPartitionAlgorithm = "Iso";
    GraphPartition graphPartition;
    EXPECT_EQ(graphPartition.PreCheck(*function), SUCCESS);
    EXPECT_EQ(graphPartition.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(graphPartition.PostCheck(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), 8);

    auto matmulSubgraphIdsAfterPartition = GetMatmulSubgraphIds(graph, branchNum);
    EXPECT_EQ(matmulSubgraphIdsAfterPartition.size(), static_cast<size_t>(branchNum));

    const int funcMagic = function->GetFuncMagic();
    std::string funcHashOrderKey = "func" + std::to_string(funcMagic) + "_0";
    function->paramConfigs_.cubeL1ReuseSetting.clear();
    function->paramConfigs_.cubeNBufferSetting.clear();
    function->paramConfigs_.cubeL1ReuseSettingByFunc = {{"DEFAULT", 1}, {funcHashOrderKey, branchNum}};
    function->paramConfigs_.cubeNBufferSettingByFunc = {{"DEFAULT", 1}, {funcHashOrderKey, 1}};
    L1CopyInReuseMerge l1CopyInReuse;
    EXPECT_EQ(l1CopyInReuse.RunOnFunction(*function), SUCCESS);

    auto matmulSubgraphIdsAfterL1Reuse = GetMatmulSubgraphIds(graph, branchNum);
    EXPECT_EQ(matmulSubgraphIdsAfterL1Reuse.size(), 1UL);
}

// Helper: run GraphPartition (COE dedup) then L1CopyInReuseMerge with a given matrix-side
// preference and merge count, returning the number of distinct matmul subgraphs after L1
// reuse. mergeCount <= 0 means "use branchNum" (i.e. allow merging all branches into one).
static size_t RunMatmulSideMerge(int branchNum, int64_t side, int mergeCount = 0)
{
    int count = mergeCount > 0 ? mergeCount : branchNum;
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestL1ReuseSide", "TestL1ReuseSide", nullptr);
    ComputationalGraphBuilder graph(func.get());
    BuildParallelMatmulGraph(graph, branchNum);

    Function* function = graph.GetFunction();
    function->paramConfigs_.sgPartitionAlgorithm = "Iso";
    GraphPartition graphPartition;
    EXPECT_EQ(graphPartition.PreCheck(*function), SUCCESS);
    EXPECT_EQ(graphPartition.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(graphPartition.PostCheck(*function), SUCCESS);

    // Matrix-side is packed into the merge-count value: side * L1_REUSE_SIDE_BASE + count
    // (side 1=left/L0A, 2=right/L0B).
    function->paramConfigs_.cubeL1ReuseSetting = {{-1, side * L1_REUSE_SIDE_BASE + count}};
    function->paramConfigs_.cubeNBufferSetting = {{-1, 1}};
    L1CopyInReuseMerge l1CopyInReuse;
    EXPECT_EQ(l1CopyInReuse.RunOnFunction(*function), SUCCESS);
    return GetMatmulSubgraphIds(graph, branchNum).size();
}

// In this graph every branch shares both the A and B GM tensors after COE dedup, so
// biasing the merge towards the left(L0A) matrix still finds the shared-A copy-in and
// merges all branches into a single subgraph.
TEST_F(L1CopyInReuseTest, TestL1ReuseSideLeftMatmulMerge) { EXPECT_EQ(RunMatmulSideMerge(4, 1), 1UL); }

// Symmetric to the left case: biasing towards the right(L0B) matrix finds the shared-B
// copy-in and merges all branches into a single subgraph.
TEST_F(L1CopyInReuseTest, TestL1ReuseSideRightMatmulMerge) { EXPECT_EQ(RunMatmulSideMerge(4, 2), 1UL); }

// Partial merge: with a merge count (2) smaller than the branch count (4), the side-preferred
// subgraphs merge in groups rather than all into one, so the final subgraph count stays > 1
// (and <= branchNum). This exercises the per-group anchor + capacity-limited merge path.
TEST_F(L1CopyInReuseTest, TestL1ReuseSidePartialMerge)
{
    size_t merged = RunMatmulSideMerge(4, 2, 2); // side=right, merge count=2
    EXPECT_GT(merged, 1UL);
    EXPECT_LE(merged, 4UL);
}

// Hard restriction (no fall-back): the InitGraphBuilder graph has no cube matmul (consumers
// are EXP, not L0A/L0B), so a left-matrix side preference matches nothing AND does not fall
// back -> nothing is merged on the L1-reuse side. With cube_nbuffer set to skip, the total
// subgraph count therefore stays at subGraphNum (whereas the unfiltered/auto merge would
// reduce it, cf. TestNormal). This verifies that side is a hard restriction.
TEST_F(L1CopyInReuseTest, TestL1ReuseSideHardRestrictNonMatmul)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function* function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{-1, 1}}; // skip the cube-nbuffer merge
    // global side=left (1) packed into the -1 entry's value: 1 * L1_REUSE_SIDE_BASE + 2.
    function->paramConfigs_.cubeL1ReuseSetting = {{-1, 1 * L1_REUSE_SIDE_BASE + 2}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    // No L0A consumer anywhere -> left side finds no partner -> no merge -> count unchanged.
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

// Deterministic assertion of the side-detection itself: a GM->L1 copy-in is classified as
// left/right by whether its L1 buffer is consumed by a copy into L0A / L0B. This isolates
// the new logic from the merge/fallback machinery (which subgraph-count tests cannot do).
TEST_F(L1CopyInReuseTest, TestIsLeftRightMatrixCopy)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{128, 128};
    // GM (DDR) sources
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_DEVICE_DDR, "gmA"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_DEVICE_DDR, "gmB"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_DEVICE_DDR, "gmC"), true);
    // L1 copy-in outputs
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_L1, "l1A"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_L1, "l1B"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_L1, "l1C"), true);
    // Consumers' outputs: L0A (left), L0B (right), UB (non-cube)
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_L0A, "l0A"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_L0B, "l0B"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, tileShape, MemoryType::MEM_UB, "ubC"), true);

    // GM -> L1 copy-ins
    EXPECT_EQ(G.AddOp(Opcode::OP_SLICE, {"gmA"}, {"l1A"}, "viewA"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_SLICE, {"gmB"}, {"l1B"}, "viewB"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_SLICE, {"gmC"}, {"l1C"}, "viewC"), true);
    // L1 -> L0A / L0B / UB consumers
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0A, {"l1A"}, {"l0A"}, "toL0A"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_L1_TO_L0B, {"l1B"}, {"l0B"}, "toL0B"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_EXP, {"l1C"}, {"ubC"}, "expC"), true);

    // Register the consumer links explicitly so GetConsumers() is deterministic.
    G.GetTensor("l1A")->AddConsumer(G.GetOp("toL0A"));
    G.GetTensor("l1B")->AddConsumer(G.GetOp("toL0B"));
    G.GetTensor("l1C")->AddConsumer(G.GetOp("expC"));

    auto* viewA = G.GetOp("viewA");
    auto* viewB = G.GetOp("viewB");
    auto* viewC = G.GetOp("viewC");
    ASSERT_NE(viewA, nullptr);
    ASSERT_NE(viewB, nullptr);
    ASSERT_NE(viewC, nullptr);

    // Left-matrix copy-in: consumer outputs to L0A.
    EXPECT_TRUE(L1CopyInReuseRunner::IsLeftMatrixCopy(*viewA));
    EXPECT_FALSE(L1CopyInReuseRunner::IsRightMatrixCopy(*viewA));
    // Right-matrix copy-in: consumer outputs to L0B.
    EXPECT_FALSE(L1CopyInReuseRunner::IsLeftMatrixCopy(*viewB));
    EXPECT_TRUE(L1CopyInReuseRunner::IsRightMatrixCopy(*viewB));
    // Non-cube copy-in (consumer outputs UB): neither side -> this is exactly the case
    // where a left/right preference finds no candidate and falls back.
    EXPECT_FALSE(L1CopyInReuseRunner::IsLeftMatrixCopy(*viewC));
    EXPECT_FALSE(L1CopyInReuseRunner::IsRightMatrixCopy(*viewC));
}

TEST_F(L1CopyInReuseTest, TestNoL1Num)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int cube_nbuffer = 2;
    const int result = 11;
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function* function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, cube_nbuffer}};
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(L1CopyInReuseTest, TestNoL1Map)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int result = 5;
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function* function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, 4}};
    function->paramConfigs_.cubeL1ReuseSetting = {{-1, 2}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(L1CopyInReuseTest, TestNoBufferMap)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int result = 5;
    const int subGraphNum = 20;
    auto shapeImme = OpImmediate::Specified(tileShape);
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function* function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{-1, 4}};
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}, {-1, 2}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(L1CopyInReuseTest, TestNoParam)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int result = 20;
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(L1CopyInReuseTest, TestInvalidL1Num)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function* function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{-1, 4}};
    function->paramConfigs_.cubeL1ReuseSetting = {{-1, -1}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), FAILED);
}

TEST_F(L1CopyInReuseTest, TestInvalidSubgraphId)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"input", "output"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_EXP, {"input"}, {"output"}, "EXP", true), true);
    EXPECT_EQ(G.SetInCast({"input"}), true);
    EXPECT_EQ(G.SetOutCast({"output"}), true);

    Function* function = G.GetFunction();
    function->paramConfigs_.cubeL1ReuseSetting = {{-1, 1}};
    function->SetTotalSubGraphCount(1);

    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), FAILED);
}

TEST_F(L1CopyInReuseTest, TestInvalidL1Map)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function* function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{-2, 2}, {-1, 4}};
    function->paramConfigs_.cubeL1ReuseSetting = {{-2, 2}, {-1, 2}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    function->paramConfigs_.cubeL1ReuseSetting = {{-1, -3}};
    EXPECT_EQ(LCRM.RunOnFunction(*function), FAILED);
    function->paramConfigs_.cubeL1ReuseSetting = {{-2, 2}};
    function->paramConfigs_.cubeNBufferSetting = {{-1, -5}};
    EXPECT_EQ(LCRM.RunOnFunction(*function), FAILED);
}

// 健康检查用例:静态图和非静态图
TEST_F(L1CopyInReuseTest, TestHealthReport)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int result = 5;
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);

    Function* function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, 4}};
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}, {-1, 2}};
    function->SetTotalSubGraphCount(subGraphNum);

    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);

    nlohmann::json report;
    const int maxFaninOpsResult = 29;
    const int maxFanoutOpsResult = 1;
    const int totalOpCount = 30;
    const int peakMemoryUsage = 512;
    const int copyDataCount = 0;
    // 计算operation节点信息
    CalcOperatorInfo(*function, report);
    EXPECT_EQ(report["totalOpCount"], totalOpCount);
    EXPECT_EQ(report["peakMemory"]["peakMemoryUsage"], peakMemoryUsage);
    EXPECT_EQ(report["copyDataCount"], copyDataCount);

    // 构建operation节点图
    std::vector<std::vector<int>> inMap; // magic到magic的映射，in - parent, out - child
    std::vector<std::vector<int>> outMap;
    std::vector<bool> actualMagic;
    GetOpConnectionMap(*function, inMap, outMap, actualMagic);

    // 计算图信息
    CalcGraphMetrics(inMap, outMap, actualMagic, report);
    EXPECT_EQ(report["maxFaninOps"].size(), maxFaninOpsResult);
    EXPECT_EQ(report["maxFanoutOps"].size(), maxFanoutOpsResult);

    // 计算operation节点信息，静态图下部分字段不计算
    nlohmann::json reportNull;
    function->SetFunctionType(FunctionType::DYNAMIC);
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    CalcOperatorInfo(*function, reportNull);
    EXPECT_EQ(reportNull["peakMemory"], nullptr);
    EXPECT_EQ(reportNull["copyDataCount"], nullptr);
}

TEST_F(L1CopyInReuseTest, TestGeneralizationL1CopyIn)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int result = 5;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast0", "outcast1"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_VIEW}, {{"incast0"}}, {{"incast1"}}, {"view"}, true), true);
    G.GetOp("view")->UpdateSubgraphID(0);
    const int subGraphNum = 20;
    G.GetTensor("incast1")->tensor->rawmagic = 1;
    G.GetTensor("incast1")->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_CONVERT, Opcode::OP_MUL};
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor" + strID}, {"outcast0", "outcast1"}};
        std::vector<std::string> opNames{"CONVERT_" + strID, "MUL_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("CONVERT_" + strID)->UpdateSubgraphID(i);
        G.GetOp("MUL_" + strID)->UpdateSubgraphID(i);
        G.GetOp("CONVERT_" + strID)->SetOpAttribute(std::make_shared<ConvertOpAttribute>(MEM_DEVICE_DDR, MEM_L1));
        G.GetTensor("tensor" + strID)->SetMemoryTypeOriginal(MEM_L1);
    }

    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast0", "outcast1"}), true);
    Function* function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, 4}};
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}, {-1, 2}};
    function->SetTotalSubGraphCount(subGraphNum);
    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("myStrategy", {
                                                   {"L1CopyInReuseMerge", PassName::L1_COPY_IN_REUSE_MERGE},
                                               });
    auto ret = passManager.RunPass(Program::GetInstance(), *function, "myStrategy");
    // L1CopyInReuseMerge LCRM;
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(L1CopyInReuseTest, TestTensorReuseFailed)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor_before" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_EXP};
        std::vector<std::vector<std::string>> iOperands{{"tensor_before" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor" + strID}};
        std::vector<std::string> opNames{"EXP_BEFORE_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("EXP_BEFORE_" + strID)->UpdateSubgraphID(i);
        G.GetTensor("tensor_before" + strID)->SetMemoryTypeOriginal(MEM_L1);
    }
    G.GetTensor("tensor_before1")->tensor->datatype = DataType::DT_FP16;
    G.GetTensor("tensor1")->tensor->datatype = DataType::DT_FP16;
    Function* function = G.GetFunction();
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}, {-1, 2}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), FAILED);
}

TEST_F(L1CopyInReuseTest, TestSemanticLabelSetting)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);

    // Set semantic label for first few subgraphs
    auto cubeLabel = std::make_shared<SemanticLabel>("CubeLabel", __FILE__, __LINE__);
    for (int i = 1; i <= 3; i++) {
        std::string strID = std::to_string(i);
        G.GetOp("EXP_" + strID)->SetSemanticLabel(cubeLabel);
    }

    Function* function = G.GetFunction();

    // L1Reuse: default merge=2, "CubeLabel" subgraphs override to 1 (subgraph granularity)
    function->paramConfigs_.cubeL1ReuseSetting = {{-1, 2}};
    function->paramConfigs_.cubeL1ReuseSettingByLabel = {{"CubeLabel", 1}};

    // CubeNBuffer: default merge=4, "CubeLabel" group override to 2
    function->paramConfigs_.cubeNBufferSetting = {{-1, 4}};
    function->paramConfigs_.cubeNBufferSettingByLabel = {{"CubeLabel", 2}};

    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
}

// Unmatched semantic labels are ignored (warned, not failed).
TEST_F(L1CopyInReuseTest, TestSemanticLabelSettingNotFound)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function* function = G.GetFunction();
    function->paramConfigs_.cubeL1ReuseSetting = {{-1, 2}};
    function->paramConfigs_.cubeL1ReuseSettingByLabel = {{"NoSuchLabel", 1}};
    function->paramConfigs_.cubeNBufferSetting = {{-1, 4}};
    function->paramConfigs_.cubeNBufferSettingByLabel = {{"NoSuchLabel", 2}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
}

// ===== ByFunc Integration Tests =====
TEST_F(L1CopyInReuseTest, ByFuncL1ReuseFuncSpecificMerge)
{
    // InitGraphBuilder: 20 subgraphs.
    // Subgraph 0 (VIEW DDR→DDR): CanReuse=false, not in L1Reuse → stays alone.
    // Subgraphs 1..19 (VIEW DDR→L1 + EXP): CanReuse=true, same hash → L1Reuse hashOrder 0.
    // L1Reuse: DEFAULT=2, func{magic}_0:4 → hashOrder 0 merges 4 per group: ceil(19/4)=5
    // NBuffer: DEFAULT=1 → no merge.
    // Total: 1 (subgraph 0) + 5 = 6
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function* function = G.GetFunction();

    int fm = function->GetFuncMagic();
    std::string key = "func" + std::to_string(fm) + "_0";
    function->paramConfigs_.cubeL1ReuseSettingByFunc = {{"DEFAULT", 2}, {key, 4}};
    function->paramConfigs_.cubeNBufferSettingByFunc = {{"DEFAULT", 1}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), 6);
}

TEST_F(L1CopyInReuseTest, ByFuncL1ReuseDefaultOneNoMerge)
{
    // cubeL1Reuse DEFAULT:1 → GetModeBySetting returns 0 → skip L1 reuse merge
    // cubeNBuffer DEFAULT:1 → GetModeBySetting returns 0 → skip NBuffer merge
    // 20 subgraphs unchanged.
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function* function = G.GetFunction();

    function->paramConfigs_.cubeL1ReuseSettingByFunc = {{"DEFAULT", 1}};
    function->paramConfigs_.cubeNBufferSettingByFunc = {{"DEFAULT", 1}};
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

} // namespace tile_fwk
} // namespace npu
