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
 * \file test_reduce_copy.cpp
 * \brief Unit test for ReduceCopy pass.
 */

#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/platform.h"
#include "interface/function/function.h"
#define private public
#include "passes/tile_graph_pass/graph_partition/reduce_copy.h"
#undef private
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk {

class ReduceCopyTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ReduceCopyTestStrategy");
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    }
    void TearDown() override { Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN); }
};

void BuildMatmulAddBranch(ComputationalGraphBuilder& G, int brId, std::vector<std::string>& incasts,
                          std::vector<std::string>& outcasts)
{
    std::vector<int64_t> tileShape{16, 16};
    std::string br = std::to_string(brId);
    std::vector<std::string> tensorNames{"tRA" + br, "tRB" + br, "tL1A" + br, "tL1B" + br,
                                         "tA" + br,  "tB" + br,  "tC" + br,   "tUB" + br};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW,      Opcode::OP_VIEW,    Opcode::OP_L1_TO_L0A,
                                Opcode::OP_L1_TO_L0B, Opcode::OP_A_MUL_B, Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"tRA" + br},  {"tRB" + br},           {"tL1A" + br},
                                                    {"tL1B" + br}, {"tA" + br, "tB" + br}, {"tC" + br}};
    std::vector<std::vector<std::string>> ooperands{{"tL1A" + br}, {"tL1B" + br}, {"tA" + br},
                                                    {"tB" + br},   {"tC" + br},   {"tUB" + br}};
    std::vector<std::string> opNames{"view" + br, "view2" + br, "toA" + br, "toB" + br, "matmul" + br, "convert" + br};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    for (auto aicOp : opNames) {
        G.GetOp(aicOp)->SetAttr(OpAttributeKey::isCube, true);
    }
    incasts.push_back("tRA" + br);
    incasts.push_back("tRB" + br);
    const int Num50 = 50;
    for (auto opName : opNames) {
        G.GetOp(opName)->UpdateSubgraphID(brId);
        G.GetOp(opName)->UpdateLatency(Num50);
    }
    const int Num2 = 2;
    for (int k = 0; k < Num2; k++) {
        std::string brv1 = std::to_string(brId + 1 + k);
        std::vector<std::string> tensorNamesV1{"add1" + brv1, "add2" + brv1, "out" + brv1};
        std::vector<Opcode> opCodesV1{Opcode::OP_ADDS, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> ioperandsV1{
            {"tUB" + br},
            {"add1" + brv1},
            {"add2" + brv1},
        };
        std::vector<std::vector<std::string>> ooperandsV1{
            {"add1" + brv1},
            {"add2" + brv1},
            {"out" + brv1},
        };
        std::vector<std::string> opNamesV1{"add1" + brv1, "add2" + brv1, "assemble" + brv1};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNamesV1), true);
        EXPECT_EQ(G.AddOps(opCodesV1, ioperandsV1, ooperandsV1, opNamesV1, true), true);
        outcasts.push_back("out" + brv1);
        for (auto opName : opNamesV1) {
            G.GetOp(opName)->UpdateSubgraphID(brId + 1 + k);
            G.GetOp(opName)->UpdateLatency(Num50);
            G.GetOp(opName)->SetAttr(OpAttributeKey::isCube, false);
        }
    }
}

void BuildMatmulAddsGraph(ComputationalGraphBuilder& G)
{
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    const int Num3 = 3;
    BuildMatmulAddBranch(G, 0, incasts, outcasts);
    BuildMatmulAddBranch(G, Num3, incasts, outcasts);
    Function* function = G.GetFunction();
    const int Num6 = 6;
    function->SetTotalSubGraphCount(Num6);
    EXPECT_EQ(G.SetInCast(incasts), true);
    EXPECT_EQ(G.SetOutCast(outcasts), true);
}

TEST_F(ReduceCopyTest, TestCase0)
{
    ComputationalGraphBuilder G;
    BuildMatmulAddsGraph(G);
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    ReduceCopyMerge merger;
    function->paramConfigs_.autoMixPartition = 1;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num2 = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num2);
}

void BuildConnectMatmul(ComputationalGraphBuilder& G, int brId, std::vector<std::string>& incasts,
                        std::vector<std::string>& outcasts)
{
    std::vector<int64_t> tileShape{16, 16};
    const int Num50 = 50;
    std::string br = std::to_string(brId);
    std::vector<std::string> tensorNames{"tRA" + br, "tRB" + br, "tL1A" + br, "tL1B" + br,
                                         "tA" + br,  "tB" + br,  "tC" + br,   "tGM" + br};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW,      Opcode::OP_VIEW,    Opcode::OP_L1_TO_L0A,
                                Opcode::OP_L1_TO_L0B, Opcode::OP_A_MUL_B, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{"tRA" + br},  {"tRB" + br},           {"tL1A" + br},
                                                    {"tL1B" + br}, {"tA" + br, "tB" + br}, {"tC" + br}};
    std::vector<std::vector<std::string>> ooperands{{"tL1A" + br}, {"tL1B" + br}, {"tA" + br},
                                                    {"tB" + br},   {"tC" + br},   {"tGM" + br}};
    std::vector<std::string> opNames{"view" + br, "view2" + br, "toA" + br, "toB" + br, "matmul" + br, "convert" + br};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    incasts.push_back("tRA" + br);
    incasts.push_back("tRB" + br);
    for (auto opName : opNames) {
        G.GetOp(opName)->UpdateSubgraphID(brId);
        G.GetOp(opName)->UpdateLatency(Num50);
        G.GetOp(opName)->SetAttr(OpAttributeKey::isCube, true);
    }
    std::string br2 = std::to_string(brId + 1);
    std::vector<std::string> tensorNames2{"tRB" + br2, "tL1A" + br2, "tL1B" + br2, "tA" + br2,
                                          "tB" + br2,  "tC" + br2,   "tGM" + br2};
    std::vector<Opcode> opCodes2{Opcode::OP_VIEW,      Opcode::OP_VIEW,    Opcode::OP_L1_TO_L0A,
                                 Opcode::OP_L1_TO_L0B, Opcode::OP_A_MUL_B, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands2{
        {"tGM" + br}, {"tRB" + br2}, {"tL1A" + br2}, {"tL1B" + br2}, {"tA" + br2, "tB" + br2}, {"tC" + br2}};
    std::vector<std::vector<std::string>> ooperands2{{"tL1A" + br2}, {"tL1B" + br2}, {"tA" + br2},
                                                     {"tB" + br2},   {"tC" + br2},   {"tGM" + br2}};
    std::vector<std::string> opNames2{"view" + br2, "view2" + br2,  "toA" + br2,
                                      "toB" + br2,  "matmul" + br2, "convert" + br2};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames2), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands2, ooperands2, opNames2, true), true);
    incasts.push_back("tRB" + br2);
    outcasts.push_back("tGM" + br2);
    for (auto opName : opNames2) {
        G.GetOp(opName)->UpdateSubgraphID(brId + 1);
        G.GetOp(opName)->UpdateLatency(Num50);
    }
}

void BuildConnectVector(ComputationalGraphBuilder& G, int brId, std::vector<std::string>& incasts,
                        std::vector<std::string>& outcasts)
{
    std::vector<int64_t> tileShape{16, 16};
    std::string br = std::to_string(brId);
    std::vector<std::string> tensorNames{"tin" + br, "tadds1" + br, "tout" + br};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_ADDS};
    std::vector<std::vector<std::string>> ioperands{{"tin" + br}, {"tadds1" + br}};
    std::vector<std::vector<std::string>> ooperands{{"tadds1" + br}, {"tout" + br}};
    std::vector<std::string> opNames{"adds1" + br, "adds2" + br};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    incasts.push_back("tin" + br);
    outcasts.push_back("tout" + br);
    const int Num50 = 50;
    G.GetOp("adds1" + br)->UpdateSubgraphID(brId);
    G.GetOp("adds1" + br)->UpdateLatency(Num50);
    G.GetOp("adds1" + br)->SetAttr(OpAttributeKey::isCube, false);
    G.GetOp("adds2" + br)->UpdateSubgraphID(brId + 1);
    G.GetOp("adds2" + br)->UpdateLatency(Num50);
    G.GetOp("adds2" + br)->SetAttr(OpAttributeKey::isCube, false);
}

void BuildConnect(ComputationalGraphBuilder& G)
{
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    const int Num2 = 2;
    BuildConnectMatmul(G, 0, incasts, outcasts);
    BuildConnectVector(G, Num2, incasts, outcasts);
    Function* function = G.GetFunction();
    const int Num4 = 4;
    function->SetTotalSubGraphCount(Num4);
    EXPECT_EQ(G.SetInCast(incasts), true);
    EXPECT_EQ(G.SetOutCast(outcasts), true);
}

TEST_F(ReduceCopyTest, TestCase1)
{
    ComputationalGraphBuilder G;
    BuildConnect(G);
    Function* function = G.GetFunction();
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    merger.RunOnFunction(*function);
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

TEST_F(ReduceCopyTest, TestCase2)
{
    ComputationalGraphBuilder G;
    BuildConnect(G);
    Function* function = G.GetFunction();
    function->paramConfigs_.autoMixPartition = 1;
    G.GetOp("adds12")->scopeInfo_.cvFuseId = 0;
    G.GetOp("adds22")->scopeInfo_.cvFuseId = 0;
    ReduceCopyMerge merger;
    merger.RunOnFunction(*function);
    const int Num2 = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num2);
}

TEST_F(ReduceCopyTest, PreserveOriginalSubgraphId)
{
    ComputationalGraphBuilder G;
    BuildMatmulAddsGraph(G);
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    auto* add11 = G.GetOp("add11");
    ASSERT_NE(add11, nullptr);
    const int64_t add11SubgraphId = static_cast<int64_t>(add11->GetSubgraphID());
    auto* add14 = G.GetOp("add14");
    ASSERT_NE(add14, nullptr);
    const int64_t add14SubgraphId = static_cast<int64_t>(add14->GetSubgraphID());
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);

    int64_t preSubgraphId = -1;
    ASSERT_TRUE(add11->GetAttr(OpAttributeKey::reduceCopyPreSubgraphId, preSubgraphId));
    EXPECT_EQ(preSubgraphId, add11SubgraphId);
    ASSERT_TRUE(add14->GetAttr(OpAttributeKey::reduceCopyPreSubgraphId, preSubgraphId));
    EXPECT_EQ(preSubgraphId, add14SubgraphId);
}

TEST_F(ReduceCopyTest, TestCase3)
{
    ComputationalGraphBuilder G;
    BuildConnect(G);
    Function* function = G.GetFunction();
    function->paramConfigs_.autoMixPartition = 1;
    const int largeNum = 2e7; // latency超过阈值的子图不会合并
    G.GetOp("matmul0")->UpdateLatency(largeNum);
    ReduceCopyMerge merger;
    merger.RunOnFunction(*function);
    const int Num4 = 4;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num4);
}

// ============================================================================
// ST: automix sink 保护与输出级旁路 (简化自 glm / gqa / mha 调试用例)
static std::string AddCubeMatmulFrom(ComputationalGraphBuilder& G, int sg, const std::string& inA,
                                     const std::string& inB);
static std::string AddCubeMatmulSG(ComputationalGraphBuilder& G, int sg, std::vector<std::string>& incasts)
{
    std::vector<int64_t> sh{16, 16};
    std::string b = std::to_string(sg);
    std::string inA = "tRA" + b;
    std::string inB = "tRB" + b;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {inA, inB}), true);
    incasts.push_back(inA);
    incasts.push_back(inB);
    return AddCubeMatmulFrom(G, sg, inA, inB);
}

static std::string AddIncast(ComputationalGraphBuilder& G, const std::string& name, std::vector<std::string>& incasts)
{
    std::vector<int64_t> sh{16, 16};
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, name), true);
    incasts.push_back(name);
    return name;
}

static std::string AddCubeMatmulFrom(ComputationalGraphBuilder& G, int sg, const std::string& inA,
                                     const std::string& inB)
{
    std::vector<int64_t> sh{16, 16};
    std::string b = std::to_string(sg);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tVA" + b, "tVB" + b, "tL0A" + b, "tL0B" + b, "tC" + b}), true);
    EXPECT_EQ(
        G.AddOps({Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0B, Opcode::OP_A_MUL_B},
                 {{inA}, {inB}, {"tVA" + b}, {"tVB" + b}, {"tL0A" + b, "tL0B" + b}},
                 {{"tVA" + b}, {"tVB" + b}, {"tL0A" + b}, {"tL0B" + b}, {"tC" + b}},
                 {"viewA" + b, "viewB" + b, "toA" + b, "toB" + b, "mm" + b}, true),
        true);
    const int Num50 = 50;
    for (auto& n : std::vector<std::string>{"viewA" + b, "viewB" + b, "toA" + b, "toB" + b, "mm" + b}) {
        G.GetOp(n)->UpdateSubgraphID(sg);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, true);
    }
    return "tC" + b;
}

static std::string AddVecSG(ComputationalGraphBuilder& G, int sg, const std::string& in, const std::string& out)
{
    std::vector<int64_t> sh{16, 16};
    std::string b = std::to_string(sg);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"vAdd" + b, out}), true);
    std::string addsName = "vadds" + b;
    std::string asmName = "vasm" + b;
    EXPECT_EQ(G.AddOps({Opcode::OP_ADDS, Opcode::OP_ASSEMBLE}, {{in}, {"vAdd" + b}}, {{"vAdd" + b}, {out}},
                       {addsName, asmName}, true),
              true);
    const int Num50 = 50;
    for (auto& n : std::vector<std::string>{addsName, asmName}) {
        G.GetOp(n)->UpdateSubgraphID(sg);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    return out;
}

static void AddVecSinkSG(ComputationalGraphBuilder& G, int sg, const std::vector<std::string>& inputs,
                         const std::string& outName, std::vector<std::string>& outcasts)
{
    std::vector<int64_t> sh{16, 16};
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, outName), true);
    const int Num50 = 50;
    std::vector<std::string> addOuts;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::string b = std::to_string(sg) + "_" + std::to_string(i);
        std::string tName = "sinkadd" + b;
        std::string opName = "vadds" + b;
        EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, tName), true);
        EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {inputs[i]}, {tName}, opName, true), true);
        G.GetOp(opName)->UpdateSubgraphID(sg);
        G.GetOp(opName)->UpdateLatency(Num50);
        G.GetOp(opName)->SetAttr(OpAttributeKey::isCube, false);
        addOuts.push_back(tName);
    }
    std::string asmName = "vasm" + std::to_string(sg);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, addOuts, {outName}, asmName, true), true);
    G.GetOp(asmName)->UpdateSubgraphID(sg);
    G.GetOp(asmName)->UpdateLatency(Num50);
    G.GetOp(asmName)->SetAttr(OpAttributeKey::isCube, false);
    outcasts.push_back(outName);
}

// glm 场景: 两条 attention 分支 (C1->V1->C2) 汇入单个 sink, 两 C2 root 不归一 -> sink rootInDeg=2 被保护。
TEST_F(ReduceCopyTest, SinkProtectedWhenMultiProducerRoots)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    AddCubeMatmulFrom(G, 0, AddIncast(G, "QA0", incasts), AddIncast(G, "QB0", incasts));
    AddVecSG(G, 1, "tC0", "v1a");
    std::string c2a = AddCubeMatmulFrom(G, 2, "v1a", AddIncast(G, "VA0", incasts));
    AddCubeMatmulFrom(G, 3, AddIncast(G, "QA1", incasts), AddIncast(G, "QB1", incasts));
    AddVecSG(G, 4, "tC3", "v1b");
    std::string c2b = AddCubeMatmulFrom(G, 5, "v1b", AddIncast(G, "VA1", incasts));
    AddVecSinkSG(G, 6, {c2a, c2b}, "sinkGlmOut", outcasts);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(7);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

// gqa 场景: 一个 cube fan-out 到两个 vec 再汇入 sink, vec 先并入 cube 归一 -> sink rootInDeg=1 放行。
TEST_F(ReduceCopyTest, SinkMergesWhenProducersCollapse)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::string c = AddCubeMatmulSG(G, 0, incasts);
    std::string v1 = AddVecSG(G, 1, c, "v1");
    std::string v2 = AddVecSG(G, 2, c, "v2");
    AddVecSinkSG(G, 3, {v1, v2}, "sinkGqaOut", outcasts);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(4);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num1 = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num1);
}

// mha 场景: 两个 cube 各 fan-out 到两个 sink (2:2 完全二分), 2 个出度0 sink -> 输出级旁路放行。
TEST_F(ReduceCopyTest, OutputStageBypassMergesAllSinks)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::string c0 = AddCubeMatmulSG(G, 0, incasts);
    std::string c1 = AddCubeMatmulSG(G, 1, incasts);
    AddVecSinkSG(G, 2, {c0, c1}, "sinkA", outcasts);
    AddVecSinkSG(G, 3, {c0, c1}, "sinkB", outcasts);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(4);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num1 = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num1);
}

// ============================================================================
// MixGraphMerger 缓存优化 UT

static MergeInput BuildSimpleMergeInput(int numSubgraph, const std::vector<std::set<int>>& outGraph,
                                        const std::vector<std::vector<int>>& groups)
{
    MergeInput input;
    input.numSubgraph = numSubgraph;
    input.maxLatency = 1e7;
    input.aivRatio = {1e-6, 1e6};
    input.subgraphAICLatency.assign(numSubgraph, 50);
    input.subgraphAIVLatency.assign(numSubgraph, 50);
    input.subGraphOutGraph = outGraph;
    input.mergeGroup = groups;
    input.isEnforceMergeGroup.assign(groups.size(), true);
    input.isValidMergeGroup.assign(groups.size(), true);
    return input;
}

// 两对独立子图 0->1, 2->3, 强制合并 {0,1} 和 {2,3}
// 验证 ApplyMergeToGraph 在 UnionSets 之后用 FindParent 获取正确 root, 增量更新缓存图
TEST_F(ReduceCopyTest, MixGraphMerger_ApplyMergeToGraphUpdatesCache)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {3}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1}, {2, 3}});
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 2);
    EXPECT_EQ(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_EQ(output.subgraphIdUpdated[2], output.subgraphIdUpdated[3]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[2]);
}

// 0->1->2, 0->3, 强制合并 {0,2}(因成环被拒) 然后 {0,1}(成功)
// 验证 CanMergeWithoutCycle 被拒后 save/restore 正确恢复缓存, 随后 {0,1} 合并成功
TEST_F(ReduceCopyTest, MixGraphMerger_CacheRestoreAfterRejection)
{
    std::vector<std::set<int>> outGraph{{1, 3}, {2}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 2}, {0, 1}});
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    const int Num2 = 2;
    EXPECT_EQ(output.numSubgraphUpdated, Num2);
    EXPECT_EQ(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_EQ(output.subgraphIdUpdated[1], output.subgraphIdUpdated[2]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[3]);
}

// 经历多轮「合并-拒绝-合并」后, 直接对比缓存图与 BuildMergedGraph 全量重建结果
// 守护「增量更新 ≡ 全量重建」这一核心不变量
TEST_F(ReduceCopyTest, MixGraphMerger_CacheConsistentWithFullRebuild)
{
    std::vector<std::set<int>> outGraph{{1, 3}, {2}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 2}, {0, 1}});
    MixGraphMerger merger;
    merger.Merge(input);
    std::vector<std::set<int>> freshOut, freshIn;
    merger.BuildMergedGraph(freshOut, freshIn);
    for (int i = 0; i < input.numSubgraph; i++) {
        EXPECT_EQ(merger.mCachedOutGraph[i], freshOut[i]) << "outGraph mismatch at node " << i;
        EXPECT_EQ(merger.mCachedInGraph[i], freshIn[i]) << "inGraph mismatch at node " << i;
    }
}

// 正向: sg0 同时是 T 的 producer(ASSEMBLE) 和 consumer(CAST), sg1 是外部 ASSEMBLE producer.
// 复刻 gdr_fwd tensor 522 结构: producer 和 consumer 在同一子图, 外部 producer 在另一子图.
// T(UB) 多 ASSEMBLE producer(全 MOVE_LOCAL) -> WillBeDdrWithoutNewCopyOp 命中 -> isDDR=true ->
// 合并 sg0+sg2 时 T 满足三条件(producer in sg0 + consumer in sg0 + external sg1) -> 拒绝.
TEST_F(ReduceCopyTest, DdrPredictRejectsMultiAssembleProducerOnUbTensor)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::vector<int64_t> sh{16, 16};
    // sg0: cube matmul(AIC) -> ASSEMBLE 写 T(UB); CAST 读 T -> ASSEMBLE 写 out (AIV)
    std::string c0 = AddCubeMatmulSG(G, 0, incasts);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, MemoryType::MEM_UB, "T"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {c0}, {"T"}, "asm0", true), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast0", "out0"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast0"}, "cast0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast0"}, {"out0"}, "asmOut0", true), true);
    // sg1: 外部 ASSEMBLE producer of T (多 producer, MOVE_LOCAL -> WillBeDdr 命中)
    AddIncast(G, "in1", incasts);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"in1"}, {"T"}, "asm1", true), true);
    // sg2: 独立 vec 子图消费 out0 (与 sg0 构成合并候选)
    AddVecSG(G, 2, "out0", "v2");
    outcasts.push_back("v2");
    const int Num50 = 50;
    G.GetOp("asm0")->UpdateSubgraphID(0);
    G.GetOp("asm0")->UpdateLatency(Num50);
    G.GetOp("asm0")->SetAttr(OpAttributeKey::isCube, true);
    for (auto& n : std::vector<std::string>{"cast0", "asmOut0"}) {
        G.GetOp(n)->UpdateSubgraphID(0);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    const int largeNum = 2e7;
    G.GetOp("asm1")->UpdateSubgraphID(1);
    G.GetOp("asm1")->UpdateLatency(largeNum);
    G.GetOp("asm1")->SetAttr(OpAttributeKey::isCube, false);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(3);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    // T(UB) 多 ASSEMBLE producer -> WillBeDdr 命中 -> isDDR=true ->
    // inner-external-use 拒绝 sg0+sg2 合并(sg1 为外部 producer 端点) -> 子图数 == 3.
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

// 反向: 同上结构, 但外部 producer 换成 ADD(calcType=BROADCAST, 非 MOVE) ->
// WillBeDdrWithoutNewCopyOp 返回 false -> isDDR=false -> 跳过 inner-external-use -> 合并放行.
TEST_F(ReduceCopyTest, DdrPredictNotTriggeredWhenProducerMixNonMoveOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::vector<int64_t> sh{16, 16};
    std::string c0 = AddCubeMatmulSG(G, 0, incasts);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, MemoryType::MEM_UB, "T"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {c0}, {"T"}, "asm0", true), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast0", "out0"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast0"}, "cast0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast0"}, {"out0"}, "asmOut0", true), true);
    // sg1: 外部 ADD producer of T (BROADCAST -> WillBeDdr 返回 false)
    AddIncast(G, "in1", incasts);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADD, {"in1", "in1"}, {"T"}, "add1", true), true);
    AddVecSG(G, 2, "out0", "v2");
    outcasts.push_back("v2");
    const int Num50 = 50;
    G.GetOp("asm0")->UpdateSubgraphID(0);
    G.GetOp("asm0")->UpdateLatency(Num50);
    G.GetOp("asm0")->SetAttr(OpAttributeKey::isCube, true);
    for (auto& n : std::vector<std::string>{"cast0", "asmOut0"}) {
        G.GetOp(n)->UpdateSubgraphID(0);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    const int largeNum = 2e7;
    G.GetOp("add1")->UpdateSubgraphID(1);
    G.GetOp("add1")->UpdateLatency(largeNum);
    G.GetOp("add1")->SetAttr(OpAttributeKey::isCube, false);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(3);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    // T(UB) 多 producer 混入 ADD(BROADCAST) -> WillBeDdr 返回 false -> isDDR=false ->
    // 跳过 inner-external-use -> sg0+sg2 合并放行(sg1 高 latency 不参与) -> 子图数 == 2.
    const int Num2 = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num2);
}

// 单 producer 跨核 COPY_OUT(L1->UB): sg0 cube matmul 产 L1 tensor, COPY_OUT(L1->UB) 写 T(UB).
TEST_F(ReduceCopyTest, DdrPredictRejectsCrossCoreCopyOutProducerOnUbTensor)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::vector<int64_t> sh{16, 16};
    // sg0: cube matmul(AIC) -> L1 tensor -> 跨核 COPY_OUT(L1->UB) 写 T(UB), 单 producer
    std::string c0 = AddCubeMatmulSG(G, 0, incasts);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, MemoryType::MEM_L1, "tL1"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {c0}, {"tL1"}, "cpOut0", true), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, MemoryType::MEM_UB, "T"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"tL1"}, {"T"}, "crossCoreCopy", true), true);
    // sg0: CAST 读 T -> ASSEMBLE 写 out (AIV, 与 cube 构成混合子图)
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast0", "out0"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast0"}, "cast0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast0"}, {"out0"}, "asmOut0", true), true);
    // sg1: 外部 CAST consumer of T (外部端点, 高 latency 阻止合并)
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast1", "out1"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast1"}, "cast1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast1"}, {"out1"}, "asmOut1", true), true);
    // sg2: 独立 vec 子图消费 out0 (与 sg0 构成合并候选)
    AddVecSG(G, 2, "out0", "v2");
    outcasts.push_back("v2");
    const int Num50 = 50;
    for (auto& n : std::vector<std::string>{"cpOut0", "crossCoreCopy"}) {
        G.GetOp(n)->UpdateSubgraphID(0);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, true);
    }
    for (auto& n : std::vector<std::string>{"cast0", "asmOut0"}) {
        G.GetOp(n)->UpdateSubgraphID(0);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    const int largeNum = 2e7;
    for (auto& n : std::vector<std::string>{"cast1", "asmOut1"}) {
        G.GetOp(n)->UpdateSubgraphID(1);
        G.GetOp(n)->UpdateLatency(largeNum);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(3);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    // T(UB) 单 producer 跨核 COPY_OUT(L1->UB) -> isDDR=true -> 拒绝 sg0+sg2 合并 -> 子图数 == 3.
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

// 单 producer OP_ASSEMBLE 写 UB tensor T: WillBeDdrWithoutNewCopyOp 单 producer 分支命中 -> isDDR=true -> 拒绝合并.
TEST_F(ReduceCopyTest, DdrPredictRejectsSingleAssembleProducerOnUbTensor)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::vector<int64_t> sh{16, 16};
    // sg0: cube matmul(AIC) -> ASSEMBLE 写 T(UB), 单 producer
    std::string c0 = AddCubeMatmulSG(G, 0, incasts);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, MemoryType::MEM_UB, "T"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {c0}, {"T"}, "asm0", true), true);
    // sg0: CAST 读 T -> ASSEMBLE 写 out (AIV, 与 cube 构成混合子图)
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast0", "out0"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast0"}, "cast0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast0"}, {"out0"}, "asmOut0", true), true);
    // sg1: 外部 CAST consumer of T (外部端点, 高 latency 阻止合并)
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast1", "out1"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast1"}, "cast1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast1"}, {"out1"}, "asmOut1", true), true);
    // sg2: 独立 vec 子图消费 out0 (与 sg0 构成合并候选)
    AddVecSG(G, 2, "out0", "v2");
    outcasts.push_back("v2");
    const int Num50 = 50;
    G.GetOp("asm0")->UpdateSubgraphID(0);
    G.GetOp("asm0")->UpdateLatency(Num50);
    G.GetOp("asm0")->SetAttr(OpAttributeKey::isCube, true);
    for (auto& n : std::vector<std::string>{"cast0", "asmOut0"}) {
        G.GetOp(n)->UpdateSubgraphID(0);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    const int largeNum = 2e7;
    for (auto& n : std::vector<std::string>{"cast1", "asmOut1"}) {
        G.GetOp(n)->UpdateSubgraphID(1);
        G.GetOp(n)->UpdateLatency(largeNum);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(3);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    // T(UB) 单 ASSEMBLE producer -> WillBeDdr 命中 -> isDDR=true -> 拒绝 sg0+sg2 合并 -> 子图数 == 3.
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

} // namespace tile_fwk
} // namespace npu
