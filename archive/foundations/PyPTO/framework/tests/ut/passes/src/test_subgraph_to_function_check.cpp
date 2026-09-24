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
 * \file test_subgraph_to_function_check.cpp
 * \brief Unit test for SubgraphToFunction preCheck and postCheck.
 */

#include "gtest/gtest.h"
#include "symbolic_scalar_test_utils.h"
#include <algorithm>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/inner/tile_shape.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#define private public
#include "passes/pass_check/subgraph_to_function_checker.h"
#undef private
#include "computational_graph_builder.h"
#include "tilefwk/tilefwk_op.h"

using namespace npu::tile_fwk;
using namespace std;

static const std::vector<int64_t> kShape88 = {8, 8};

static void RunPreCheckTest(const std::string& funcName, Opcode opcode, const std::vector<std::string>& iops,
                            const std::vector<std::string>& oops, const std::string& opName, int opSubGraphId,
                            int totalSubGraphCount, Status expectedStatus)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), funcName, funcName, nullptr);
    Program::GetInstance().InsertFuncToFunctionMap(funcName, f);
    ComputationalGraphBuilder G(f.get());
    for (const auto& t : iops)
        EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, t));
    for (const auto& t : oops)
        EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, t));
    EXPECT_TRUE(G.AddOp(opcode, iops, oops, opName));
    G.GetOp(opName)->UpdateSubgraphID(opSubGraphId);
    G.GetFunction()->SetTotalSubGraphCount(totalSubGraphCount);
    SubGraphToFuncChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*G.GetFunction()), expectedStatus);
}

static void RunColorOutGraphCheckTest(const std::string& funcName, const std::vector<std::vector<int>>& colorInGraph,
                                      const std::vector<std::vector<int>>& colorOutGraph,
                                      Status expectedPostCheckStatus, bool threeOpFork = false)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), funcName, funcName, nullptr);
    Program::GetInstance().InsertFuncToFunctionMap(funcName, f);
    ComputationalGraphBuilder G(f.get());
    for (const auto& t : {"a", "b", "c"})
        EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, t));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add1"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"b"}, {"c"}, "add2"));
    if (threeOpFork) {
        EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, "d"));
        EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"b"}, {"d"}, "add3"));
        G.GetOp("add3")->UpdateSubgraphID(2);
    }
    G.GetOp("add1")->UpdateSubgraphID(0);
    G.GetOp("add2")->UpdateSubgraphID(1);
    for (const auto& t : {"a", "b", "c"})
        G.GetTensor(t)->SetMemoryTypeBoth(MemoryType::MEM_UB);
    if (threeOpFork)
        G.GetTensor("d")->SetMemoryTypeBoth(MemoryType::MEM_UB);
    G.GetFunction()->SetTotalSubGraphCount(threeOpFork ? 3 : 2);
    G.GetFunction()->SetFunctionType(FunctionType::STATIC);
    SubGraphToFuncChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*G.GetFunction()), SUCCESS);
    checker.SetColorGraph(colorInGraph, colorOutGraph);
    EXPECT_EQ(checker.DoPostCheck(*G.GetFunction()), expectedPostCheckStatus);
}

class SubgraphToFunctionCheckTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(SubgraphToFunctionCheckTest, TestPrePostCheck)
{
    constexpr int kTileSize = 32;
    constexpr int kVectorSize = 64;
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_PRE_CHECK, true);
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_POST_CHECK, true);
    TileShape::Current().SetVecTile(kTileSize, kTileSize);
    TileShape::Current().SetCubeTile({kTileSize, kTileSize}, {kTileSize, kTileSize}, {kTileSize, kTileSize});

    std::vector<int64_t> shape = {kVectorSize, kVectorSize};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c1(DT_FP32, shape, "c1");
    Tensor c2(DT_FP32, shape, "c2");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(a, 1.0f),
        RawTensorData::CreateConstantTensor<float>(b, 2.0f),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(c1, 0.0f),
        RawTensorData::CreateConstantTensor<float>(c2, 0.0f),
    });

    config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
    FUNCTION("SimpleTest", {a, b}, {c1, c2})
    {
        Tensor temp1 = Add(a, b);
        temp1 = Mul(temp1, a);
        c1 = Sub(temp1, b);

        Tensor temp2 = Add(a, b);
        temp2 = Mul(temp2, a);
        c2 = Sub(temp2, b);
    }
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_SimpleTest_2");
    EXPECT_NE(mainFunc, nullptr);
}

TEST_F(SubgraphToFunctionCheckTest, NOPCheck_NonNOP_Fail)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), "NOPCheckTest", "NOPCheckTest", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("NOPCheckTest", f);
    ComputationalGraphBuilder G(f.get());
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, "a"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, "b"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add_op"));
    SubGraphToFuncChecker checker;
    EXPECT_NE(checker.NOPCheck(G.GetFunction()->Operations()[0]), SUCCESS);
}

TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_InvalidTotalSubGraphNum_Fail)
{
    RunPreCheckTest("TopoTest", Opcode::OP_ADD, {"x"}, {"y"}, "add_op", 0, 0, FAILED);
}

TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_NegativeSubGraphId_NotNOP_Fail)
{
    RunPreCheckTest("NegIdTest", Opcode::OP_ADD, {"a"}, {"b"}, "add_op", -1, 1, FAILED);
}

TEST_F(SubgraphToFunctionCheckTest, NOPCheck_NOPWithIOperands_Fail)
{
    RunPreCheckTest("NOPIOpTest", Opcode::OP_NOP, {"a"}, {"b"}, "nop_op", -1, 1, FAILED);
}

TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_SubGraphIdOutOfRange_Fail)
{
    RunPreCheckTest("OutOfRangeTest", Opcode::OP_ADD, {"a"}, {"b"}, "add_op", 1, 1, FAILED);
}

static void RunPreCheck2OpTest(const std::string& funcName, int add1SgId, int add2SgId, int totalSubGraphCount,
                               Status expectedStatus)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), funcName, funcName, nullptr);
    Program::GetInstance().InsertFuncToFunctionMap(funcName, f);
    ComputationalGraphBuilder G(f.get());
    for (const auto& t : {"a", "b", "c"})
        EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, t));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add1"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"b"}, {"c"}, "add2"));
    G.GetOp("add1")->UpdateSubgraphID(add1SgId);
    G.GetOp("add2")->UpdateSubgraphID(add2SgId);
    G.GetFunction()->SetTotalSubGraphCount(totalSubGraphCount);
    SubGraphToFuncChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*G.GetFunction()), expectedStatus);
}

TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_ParentSubGraphIdGreater_Fail)
{
    RunPreCheck2OpTest("ParentGreaterTest", 1, 0, 2, FAILED);
}

TEST_F(SubgraphToFunctionCheckTest, InAndOutGraphConsistencyCheck_ParentSeqNoExceeds_Fail)
{
    SubGraphToFuncChecker checker;
    std::vector<std::vector<size_t>> inGraph = {{0, 0}};
    std::vector<std::vector<size_t>> outGraph = {{0}};
    EXPECT_NE(checker.InAndOutGraphConsistencyCheck(inGraph, outGraph), SUCCESS);
}

TEST_F(SubgraphToFunctionCheckTest, InAndOutGraphConsistencyCheck_NodeMismatch_Fail)
{
    SubGraphToFuncChecker checker;
    std::vector<std::vector<size_t>> inGraph = {{0}};
    std::vector<std::vector<size_t>> outGraph = {{1}};
    EXPECT_NE(checker.InAndOutGraphConsistencyCheck(inGraph, outGraph), SUCCESS);
}

TEST_F(SubgraphToFunctionCheckTest, ColorOutGraphCheck_EdgeMissedInColorOutGraph_Fail)
{
    RunColorOutGraphCheckTest("ColorTest_EdgeMissed", {{}, {0}, {}}, {{1}, {}, {}}, FAILED, true);
}

TEST_F(SubgraphToFunctionCheckTest, ColorOutGraphCheck_EdgeInColorNotInOutGraph_Fail)
{
    RunColorOutGraphCheckTest("ColorTest_EdgeInColor", {{}, {0}, {0}, {0}}, {{1, 2, 3}, {}, {}, {}}, FAILED, true);
}

TEST_F(SubgraphToFunctionCheckTest, NOPCheckHasInCtrlOperations)
{
    constexpr int kTileSize = 32;
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_PRE_CHECK, true);
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_POST_CHECK, true);
    TileShape::Current().SetVecTile(kTileSize, kTileSize);
    auto func = std::make_shared<Function>(Program::GetInstance(), "NopCtrlInTest", "NopCtrlInTest", nullptr);
    Operation& nopOp = PassOperationUtils::AddOperation(*func, Opcode::OP_NOP, {}, {});
    Operation& dummyOp = PassOperationUtils::AddOperation(*func, Opcode::OP_NOP, {}, {});
    nopOp.AddInCtrlOperation(dummyOp);
    SubGraphToFuncChecker checker;
    Status ret = checker.NOPCheck(nopOp);
    EXPECT_EQ(ret, FAILED);
}

TEST_F(SubgraphToFunctionCheckTest, NOPCheckHasOOperands)
{
    constexpr int kTileSize = 32;
    constexpr int kVectorSize = 64;
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_PRE_CHECK, true);
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_POST_CHECK, true);
    TileShape::Current().SetVecTile(kTileSize, kTileSize);
    std::vector<int64_t> shape = {kVectorSize, kVectorSize};
    auto func = std::make_shared<Function>(Program::GetInstance(), "NopOutputTest", "NopOutputTest", nullptr);
    LogicalTensorPtr outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
                                                                            CreateTestConstIntVector(shape));
    Operation& nopOp = PassOperationUtils::AddOperation(*func, Opcode::OP_NOP, {}, {outTensor});
    SubGraphToFuncChecker checker;
    Status ret = checker.NOPCheck(nopOp);
    EXPECT_EQ(ret, FAILED);
}

TEST_F(SubgraphToFunctionCheckTest, NOPCheckHasOutCtrlOperations)
{
    constexpr int kTileSize = 32;
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_PRE_CHECK, true);
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_POST_CHECK, true);
    TileShape::Current().SetVecTile(kTileSize, kTileSize);
    auto func = std::make_shared<Function>(Program::GetInstance(), "NopCtrlOutTest", "NopCtrlOutTest", nullptr);
    Operation& nopOp = PassOperationUtils::AddOperation(*func, Opcode::OP_NOP, {}, {});
    Operation& dummyOp = PassOperationUtils::AddOperation(*func, Opcode::OP_NOP, {}, {});
    nopOp.AddOutCtrlOperation(dummyOp);
    SubGraphToFuncChecker checker;
    Status ret = checker.NOPCheck(nopOp);
    EXPECT_EQ(ret, FAILED);
}

TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_EmptySubgraph)
{
    constexpr int kTileSize = 32;
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_PRE_CHECK, true);
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_POST_CHECK, true);
    TileShape::Current().SetVecTile(kTileSize, kTileSize);

    auto func = std::make_shared<Function>(Program::GetInstance(), "EmptySubgraphTest", "EmptySubgraphTest", nullptr);
    func->SetTotalSubGraphCount(1);

    SubGraphToFuncChecker checker;
    Status status = checker.CheckSubGraphTopo(*func);
    EXPECT_EQ(status, FAILED);
}

TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphBoundary_Rule1_DDR_NotBoundary)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    auto f = std::make_shared<Function>(Program::GetInstance(), "test_rule1_ddr", "test_rule1_ddr", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("test_rule1_ddr", f);

    ComputationalGraphBuilder G(f.get());
    G.AddTensor(DataType::DT_FP32, kShape88, "in");
    G.AddTensor(DataType::DT_FP32, kShape88, "out");
    G.AddOp(Opcode::OP_ADD, {"in"}, {"out"}, "add_op");

    auto op = G.GetOp("add_op");
    auto inTensor = G.GetTensor("in");
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    auto outTensor = G.GetTensor("out");
    outTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    op->UpdateSubgraphID(0);

    SubGraphToFuncChecker checker;
    auto status = checker.CheckSubGraphBoundary(*f);
    EXPECT_EQ(status, FAILED);
}

TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphBoundary_Rule3_CopyIn_NoBoundary)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    auto f = std::make_shared<Function>(Program::GetInstance(), "test_rule3_clean", "test_rule3_clean", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("test_rule3_clean", f);

    ComputationalGraphBuilder G(f.get());
    G.AddTensor(DataType::DT_FP32, kShape88, "in");
    G.AddTensor(DataType::DT_FP32, kShape88, "out");

    G.AddOp(Opcode::OP_UB_COPY_IN, {"in"}, {"out"}, "copy_in");

    auto op = G.GetOp("copy_in");
    auto tensor = G.GetTensor("in");

    op->UpdateSubgraphID(0);
    SubGraphToFuncChecker checker;
    auto status = checker.CheckSubGraphBoundary(*f);
    EXPECT_EQ(status, SUCCESS);
}

TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphBoundary_Output_DDR_Only)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    auto f = std::make_shared<Function>(Program::GetInstance(), "test_out_ddr_clean", "test_out_ddr_clean", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("test_out_ddr_clean", f);

    ComputationalGraphBuilder G(f.get());
    G.AddTensor(DataType::DT_FP32, kShape88, "in");
    G.AddTensor(DataType::DT_FP32, kShape88, "out");
    G.AddOp(Opcode::OP_ADD, {"in"}, {"out"}, "add_op");

    auto op = G.GetOp("add_op");
    auto in_tensor = G.GetTensor("in");
    auto out_tensor = G.GetTensor("out");

    op->UpdateSubgraphID(0);

    out_tensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);

    SubGraphToFuncChecker checker;
    auto status = checker.CheckSubGraphBoundary(*f);
    EXPECT_EQ(status, FAILED);
}
