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
 * \file test_process_atomic.cpp
 * \brief Unit test for ProcessAtomic pass.
 */

#include <fstream>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/graph_optimization/process_atomic.h"
#include "passes/tile_graph_pass/graph_constraint/pre_graph/pre_graph.h"
#include "computational_graph_builder.h"
#include "ut_json/ut_json_tool.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

size_t CountOpsByType(Function* function, Opcode opcode)
{
    if (function == nullptr) {
        return 0;
    }
    size_t count = 0;
    for (const auto& op : function->Operations()) {
        if (!op.IsDeleted() && op.GetOpcode() == opcode) {
            ++count;
        }
    }
    return count;
}

class ProcessAtomicTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        // config::SetHostConfig(KEY_STRATEGY, "SplitKTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void SetMatMulAttr(ComputationalGraphBuilder& G, const std::string name, bool isAtomic = false,
                       const int nzFormat = 0)
    {
        auto op = G.GetOp(name);
        if (op == nullptr) {
            return;
        }
        if (isAtomic) {
            op->SetAttribute(RMW_MODE_ATTR_ADD, 1);
        } else {
            op->SetAttribute(RMW_MODE_ATTR_ADD, 0);
        }
        op->SetAttribute(MATMUL_NZ_ATTR, nzFormat);
        op->SetAttribute(A_MUL_B_ACT_M, 0L);
        op->SetAttribute(A_MUL_B_ACT_K, 0L);
        op->SetAttribute(A_MUL_B_ACT_N, 0L);
    }

    void SetMatmulMatrixSize(ComputationalGraphBuilder& G, const std::string name,
                             const std::vector<int64_t>& matrixSize)
    {
        auto op = G.GetOp(name);
        op->SetAttribute(A_MUL_B_ACT_M, matrixSize[0]);
        op->SetAttribute(A_MUL_B_ACT_K, matrixSize[1]);
        op->SetAttribute(A_MUL_B_ACT_N, matrixSize[2]);
    }

    void CheckL0cType(DataType inputAstDtype, DataType outputAstDtype, DataType l0cDtype)
    {
        ComputationalGraphBuilder G;
        // add tensor
        G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
        auto mat_a = G.GetTensor("mat_a");
        mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
        auto mat_b = G.GetTensor("mat_b");
        mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        G.AddTensor(outputAstDtype, {64, 128}, "mat_c");
        auto mat_c = G.GetTensor("mat_c");
        mat_c->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        G.AddTensor(inputAstDtype, {64, 128}, "l1_a");
        auto l1_a = G.GetTensor("l1_a");
        l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
        G.AddTensor(inputAstDtype, {64, 128}, "l0_a");
        auto l0_a = G.GetTensor("l0_a");
        l0_a->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
        G.AddTensor(inputAstDtype, {128, 128}, "l1_b");
        auto l1_b = G.GetTensor("l1_b");
        l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
        G.AddTensor(inputAstDtype, {128, 128}, "l0_b");
        auto l0_b = G.GetTensor("l0_b");
        l0_b->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
        G.AddTensor(outputAstDtype, {64, 128}, "l0_c");
        auto l0_c = G.GetTensor("l0_c");
        l0_c->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
        // add op
        G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
        G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
        G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a"}, {"l0_a"}, "L1_To_L0A");
        G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b"}, {"l0_b"}, "L1_To_L0B");
        G.AddOp(Opcode::OP_A_MUL_B, {"l0_a", "l0_b"}, {"l0_c"}, "A_MUL_B");
        SetMatMulAttr(G, "A_MUL_B", false, 0);
        G.AddOp(Opcode::OP_COPY_OUT, {"l0_c"}, {"mat_c"}, "L0C_Copy_out");
        // set incast and outcast
        G.SetInCast({"mat_a", "mat_b"});
        G.SetOutCast({"mat_c"});
        // check before pass
        auto l0cBefore = G.GetTensor("l0_c");
        EXPECT_EQ(l0cBefore->Datatype(), outputAstDtype);
        // run pass
        Function* function = G.GetFunction();
        EXPECT_NE(function, nullptr);
        ProcessAtomic passLocal;
        passLocal.Run(*function, "", "", 0);
        CubeProcess cubeProcess;
        cubeProcess.UpdateCubeOp(*function);
        // check after pass
        auto l0cAfter = G.GetTensor("l0_c");
        EXPECT_EQ(l0cAfter->Datatype(), l0cDtype);
    }
    void TearDown() override {}

    std::shared_ptr<LogicalTensor> AddDdrTensor(ComputationalGraphBuilder& G, DataType dtype,
                                                const std::vector<int64_t>& shape, const std::string& name)
    {
        G.AddTensor(dtype, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        return tensor;
    }
};

TEST_F(ProcessAtomicTest, TestMMFP16) { CheckL0cType(DataType::DT_FP16, DataType::DT_FP16, DataType::DT_FP32); }
TEST_F(ProcessAtomicTest, TestMMBF16) { CheckL0cType(DataType::DT_BF16, DataType::DT_BF16, DataType::DT_FP32); }
TEST_F(ProcessAtomicTest, TestMMFP32) { CheckL0cType(DataType::DT_FP32, DataType::DT_FP32, DataType::DT_FP32); }
TEST_F(ProcessAtomicTest, TestMMINT8) { CheckL0cType(DataType::DT_INT8, DataType::DT_INT8, DataType::DT_INT32); }
TEST_F(ProcessAtomicTest, TestMMINT16) { CheckL0cType(DataType::DT_INT16, DataType::DT_INT16, DataType::DT_INT32); }
TEST_F(ProcessAtomicTest, TestMMINT32) { CheckL0cType(DataType::DT_INT32, DataType::DT_INT32, DataType::DT_INT32); }

TEST_F(ProcessAtomicTest, TestGetL0CCopyOutsInvalidOutputMemory)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, {16, 16}, "l0_c");
    auto l0c = G.GetTensor("l0_c");
    l0c->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddTensor(DataType::DT_FP32, {16, 16}, "bad_out");
    auto badOut = G.GetTensor("bad_out");
    badOut->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c"}, {"bad_out"}, "L0C_Copy_out");
    CubeProcess cubeProcess;
    std::vector<Operation*> copyOuts;
    EXPECT_EQ(cubeProcess.GetL0CCopyOuts(*G.GetOp("L0C_Copy_out"), copyOuts), FAILED);
}

TEST_F(ProcessAtomicTest, TestReducAccProcessAtomicOn)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_0");
    auto mat_c_before_reduce_acc_0 = G.GetTensor("mat_c_before_reduce_acc_0");
    mat_c_before_reduce_acc_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_1");
    auto mat_c_before_reduce_acc_1 = G.GetTensor("mat_c_before_reduce_acc_1");
    mat_c_before_reduce_acc_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_2");
    auto mat_c_before_reduce_acc_2 = G.GetTensor("mat_c_before_reduce_acc_2");
    mat_c_before_reduce_acc_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_3");
    auto mat_c_before_reduce_acc_3 = G.GetTensor("mat_c_before_reduce_acc_3");
    mat_c_before_reduce_acc_3->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_after_reduce_acc");
    auto mat_c_after_reduce_acc = G.GetTensor("mat_c_after_reduce_acc");
    mat_c_after_reduce_acc->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l1_a_0", "l1_a_1", "l1_a_2", "l1_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l1_b_0", "l1_b_1", "l1_b_2", "l1_b_3"});
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_2 = G.GetTensor("l1_a_2");
    l1_a_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_3 = G.GetTensor("l1_a_3");
    l1_a_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_2 = G.GetTensor("l1_b_2");
    l1_b_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_3 = G.GetTensor("l1_b_3");
    l1_b_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l0_a_0", "l0_a_1", "l0_a_2", "l0_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l0_b_0", "l0_b_1", "l0_b_2", "l0_b_3"});
    G.AddTensors(outputAstDtype, {64, 128}, {"l0_c_0", "l0_c_1", "l0_c_2", "l0_c_3"});
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_2 = G.GetTensor("l0_a_2");
    l0_a_2->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_3 = G.GetTensor("l0_a_3");
    l0_a_3->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_2 = G.GetTensor("l0_b_2");
    l0_b_2->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_3 = G.GetTensor("l0_b_3");
    l0_b_3->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_2 = G.GetTensor("l0_c_2");
    l0_c_2->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_3 = G.GetTensor("l0_c_3");
    l0_c_3->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_0"}, "L1_Copy_In_A_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_1"}, "L1_Copy_In_A_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_2"}, "L1_Copy_In_A_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_3"}, "L1_Copy_In_A_3");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_0"}, "L1_Copy_In_B_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_1"}, "L1_Copy_In_B_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_2"}, "L1_Copy_In_B_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_3"}, "L1_Copy_In_B_3");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_2"}, {"l0_a_2"}, "L1_To_L0A_2");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_3"}, {"l0_a_3"}, "L1_To_L0A_3");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_2"}, {"l0_b_2"}, "L1_To_L0B_2");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_3"}, {"l0_b_3"}, "L1_To_L0B_3");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_2", "l0_b_2"}, {"l0_c_2"}, "A_MUL_B_2");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_3", "l0_b_3"}, {"l0_c_3"}, "A_MUL_B_3");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_before_reduce_acc_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_before_reduce_acc_1"}, "L0C_Copy_out_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_2"}, {"mat_c_before_reduce_acc_2"}, "L0C_Copy_out_2");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_3"}, {"mat_c_before_reduce_acc_3"}, "L0C_Copy_out_3");
    G.AddOp(Opcode::OP_REDUCE_ACC,
            {"mat_c_before_reduce_acc_0", "mat_c_before_reduce_acc_1", "mat_c_before_reduce_acc_2",
             "mat_c_before_reduce_acc_3"},
            {"mat_c_after_reduce_acc"}, "Reduce_Acc");
    SetMatMulAttr(G, "A_MUL_B_0", false, 0);
    SetMatMulAttr(G, "A_MUL_B_1", false, 0);
    SetMatMulAttr(G, "A_MUL_B_2", false, 0);
    SetMatMulAttr(G, "A_MUL_B_3", false, 0);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_after_reduce_acc"});
    // check before pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    int opReduceAccCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_NE(opReduceAccCount, 0);
    // run pass
    ProcessAtomic passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    opReduceAccCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_EQ(opReduceAccCount, 0);
}

TEST_F(ProcessAtomicTest, TestReducAccProcessAtomicOff)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_0");
    auto mat_c_before_reduce_acc_0 = G.GetTensor("mat_c_before_reduce_acc_0");
    mat_c_before_reduce_acc_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_1");
    auto mat_c_before_reduce_acc_1 = G.GetTensor("mat_c_before_reduce_acc_1");
    mat_c_before_reduce_acc_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_2");
    auto mat_c_before_reduce_acc_2 = G.GetTensor("mat_c_before_reduce_acc_2");
    mat_c_before_reduce_acc_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_3");
    auto mat_c_before_reduce_acc_3 = G.GetTensor("mat_c_before_reduce_acc_3");
    mat_c_before_reduce_acc_3->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l1_a_0", "l1_a_1", "l1_a_2", "l1_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l1_b_0", "l1_b_1", "l1_b_2", "l1_b_3"});
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_2 = G.GetTensor("l1_a_2");
    l1_a_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_3 = G.GetTensor("l1_a_3");
    l1_a_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_2 = G.GetTensor("l1_b_2");
    l1_b_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_3 = G.GetTensor("l1_b_3");
    l1_b_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l0_a_0", "l0_a_1", "l0_a_2", "l0_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l0_b_0", "l0_b_1", "l0_b_2", "l0_b_3"});
    G.AddTensors(outputAstDtype, {64, 128}, {"l0_c_0", "l0_c_1", "l0_c_2", "l0_c_3"});
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_2 = G.GetTensor("l0_a_2");
    l0_a_2->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_3 = G.GetTensor("l0_a_3");
    l0_a_3->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_2 = G.GetTensor("l0_b_2");
    l0_b_2->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_3 = G.GetTensor("l0_b_3");
    l0_b_3->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_2 = G.GetTensor("l0_c_2");
    l0_c_2->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_3 = G.GetTensor("l0_c_3");
    l0_c_3->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_0"}, "L1_Copy_In_A_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_1"}, "L1_Copy_In_A_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_2"}, "L1_Copy_In_A_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_3"}, "L1_Copy_In_A_3");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_0"}, "L1_Copy_In_B_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_1"}, "L1_Copy_In_B_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_2"}, "L1_Copy_In_B_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_3"}, "L1_Copy_In_B_3");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_2"}, {"l0_a_2"}, "L1_To_L0A_2");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_3"}, {"l0_a_3"}, "L1_To_L0A_3");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_2"}, {"l0_b_2"}, "L1_To_L0B_2");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_3"}, {"l0_b_3"}, "L1_To_L0B_3");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_2", "l0_b_2"}, {"l0_c_2"}, "A_MUL_B_2");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_3", "l0_b_3"}, {"l0_c_3"}, "A_MUL_B_3");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_before_reduce_acc_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_before_reduce_acc_1"}, "L0C_Copy_out_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_2"}, {"mat_c_before_reduce_acc_2"}, "L0C_Copy_out_2");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_3"}, {"mat_c_before_reduce_acc_3"}, "L0C_Copy_out_3");
    SetMatMulAttr(G, "A_MUL_B_0", false, 0);
    SetMatMulAttr(G, "A_MUL_B_1", false, 0);
    SetMatMulAttr(G, "A_MUL_B_2", false, 0);
    SetMatMulAttr(G, "A_MUL_B_3", false, 0);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_before_reduce_acc_0", "mat_c_before_reduce_acc_1", "mat_c_before_reduce_acc_2",
                  "mat_c_before_reduce_acc_3"});
    // check before pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    int opReduceAccCount = 0;
    int opCountBefore = 0;
    for (auto& op : function->Operations()) {
        opCountBefore++;
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_EQ(opReduceAccCount, 0);
    // run pass
    ProcessAtomic passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    opReduceAccCount = 0;
    int opCountAfter = 0;
    for (auto& op : function->Operations()) {
        opCountAfter++;
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_EQ(opReduceAccCount, 0);
    EXPECT_EQ(opCountBefore, opCountAfter);
}

TEST_F(ProcessAtomicTest, TestReducAccInputLess)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc");
    auto mat_c_before_reduce_acc = G.GetTensor("mat_c_before_reduce_acc");
    mat_c_before_reduce_acc->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_after_reduce_acc");
    auto mat_c_after_reduce_acc = G.GetTensor("mat_c_after_reduce_acc");
    mat_c_after_reduce_acc->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a");
    auto l1_a = G.GetTensor("l1_a");
    l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a");
    auto l0_a = G.GetTensor("l0_a");
    l0_a->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b");
    auto l1_b = G.GetTensor("l1_b");
    l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b");
    auto l0_b = G.GetTensor("l0_b");
    l0_b->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c");
    auto l0_c = G.GetTensor("l0_c");
    l0_c->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a"}, {"l0_a"}, "L1_To_L0A");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b"}, {"l0_b"}, "L1_To_L0B");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a", "l0_b"}, {"l0_c"}, "A_MUL_B");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c"}, {"mat_c_before_reduce_acc"}, "L0C_Copy_out");
    G.AddOp(Opcode::OP_REDUCE_ACC, {"mat_c_before_reduce_acc"}, {"mat_c_after_reduce_acc"}, "Reduce_Acc");
    SetMatMulAttr(G, "A_MUL_B", false, 0);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_after_reduce_acc"});
    // check before pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    int opReduceAccCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_NE(opReduceAccCount, 0);
    // run pass
    ProcessAtomic passLocal;
    Status preCheckResult = passLocal.PreCheck(*function);
    EXPECT_EQ(preCheckResult, SUCCESS);
    passLocal.Run(*function, "", "", 0);
    // check after pass
    opReduceAccCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_EQ(opReduceAccCount, 0);
}

TEST_F(ProcessAtomicTest, TestReducAccOutPutMore)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_0");
    auto mat_c_before_reduce_acc_0 = G.GetTensor("mat_c_before_reduce_acc_0");
    mat_c_before_reduce_acc_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_1");
    auto mat_c_before_reduce_acc_1 = G.GetTensor("mat_c_before_reduce_acc_1");
    mat_c_before_reduce_acc_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_after_reduce_acc_0");
    auto mat_c_after_reduce_acc_0 = G.GetTensor("mat_c_after_reduce_acc_0");
    mat_c_after_reduce_acc_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_after_reduce_acc_1");
    auto mat_c_after_reduce_acc_1 = G.GetTensor("mat_c_after_reduce_acc_1");
    mat_c_after_reduce_acc_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_2");
    auto mat_c_before_reduce_acc_2 = G.GetTensor("mat_c_before_reduce_acc_2");
    mat_c_before_reduce_acc_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_3");
    auto mat_c_before_reduce_acc_3 = G.GetTensor("mat_c_before_reduce_acc_3");
    mat_c_before_reduce_acc_3->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l1_a_0", "l1_a_1", "l1_a_2", "l1_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l1_b_0", "l1_b_1", "l1_b_2", "l1_b_3"});
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_2 = G.GetTensor("l1_a_2");
    l1_a_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_3 = G.GetTensor("l1_a_3");
    l1_a_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_2 = G.GetTensor("l1_b_2");
    l1_b_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_3 = G.GetTensor("l1_b_3");
    l1_b_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l0_a_0", "l0_a_1", "l0_a_2", "l0_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l0_b_0", "l0_b_1", "l0_b_2", "l0_b_3"});
    G.AddTensors(outputAstDtype, {64, 128}, {"l0_c_0", "l0_c_1", "l0_c_2", "l0_c_3"});
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_2 = G.GetTensor("l0_a_2");
    l0_a_2->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_3 = G.GetTensor("l0_a_3");
    l0_a_3->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_2 = G.GetTensor("l0_b_2");
    l0_b_2->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_3 = G.GetTensor("l0_b_3");
    l0_b_3->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_2 = G.GetTensor("l0_c_2");
    l0_c_2->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_3 = G.GetTensor("l0_c_3");
    l0_c_3->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_0"}, "L1_Copy_In_A_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_1"}, "L1_Copy_In_A_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_2"}, "L1_Copy_In_A_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_3"}, "L1_Copy_In_A_3");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_0"}, "L1_Copy_In_B_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_1"}, "L1_Copy_In_B_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_2"}, "L1_Copy_In_B_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_3"}, "L1_Copy_In_B_3");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_2"}, {"l0_a_2"}, "L1_To_L0A_2");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_3"}, {"l0_a_3"}, "L1_To_L0A_3");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_2"}, {"l0_b_2"}, "L1_To_L0B_2");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_3"}, {"l0_b_3"}, "L1_To_L0B_3");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_2", "l0_b_2"}, {"l0_c_2"}, "A_MUL_B_2");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_3", "l0_b_3"}, {"l0_c_3"}, "A_MUL_B_3");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_before_reduce_acc_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_before_reduce_acc_1"}, "L0C_Copy_out_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_2"}, {"mat_c_before_reduce_acc_2"}, "L0C_Copy_out_2");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_3"}, {"mat_c_before_reduce_acc_3"}, "L0C_Copy_out_3");
    G.AddOp(Opcode::OP_REDUCE_ACC,
            {"mat_c_before_reduce_acc_0", "mat_c_before_reduce_acc_1", "mat_c_before_reduce_acc_2",
             "mat_c_before_reduce_acc_3"},
            {"mat_c_after_reduce_acc_0", "mat_c_after_reduce_acc_1"}, "Reduce_Acc");
    SetMatMulAttr(G, "A_MUL_B_0", false, 0);
    SetMatMulAttr(G, "A_MUL_B_1", false, 0);
    SetMatMulAttr(G, "A_MUL_B_2", false, 0);
    SetMatMulAttr(G, "A_MUL_B_3", false, 0);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_after_reduce_acc_0", "mat_c_after_reduce_acc_1"});
    // check before pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    int opReduceAccCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_NE(opReduceAccCount, 0);
    // run pass
    ProcessAtomic passLocal;
    Status preCheckResult = passLocal.PreCheck(*function);
    EXPECT_NE(preCheckResult, SUCCESS);
}

TEST_F(ProcessAtomicTest, TestMMFP16AtomicOn)
{
    int m = 32;
    int n = 512;
    int k = 128;
    std::vector<int64_t> shape_a = {m, k};
    std::vector<int64_t> shape_b = {k, n};
    std::vector<int64_t> shape_c = {m, n};
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP32;
    config::SetHostConfig(KEY_STRATEGY, "PVC2_OOO");

    PROGRAM("Test_MM_FP16_Atomic_On")
    {
        Tensor mat_a(inputAstDtype, shape_a, "mat_a");
        Tensor mat_b(inputAstDtype, shape_b, "mat_b");
        Tensor final_out(outputAstDtype, shape_c, "final_out");
        config::SetBuildStatic(true);
        FUNCTION("MM_FP16_Atomic_On", {mat_a, mat_b, final_out})
        {
            TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {64, 64}, true);
            auto tmpC = Matrix::Matmul(outputAstDtype, mat_a, mat_b, false, false);
            TileShape::Current().SetVecTile(32, 32);
            final_out = Add(tmpC, Element(DataType::DT_FP32, 0.0));
        }
    }
}

TEST_F(ProcessAtomicTest, TestAnzBnd)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c");
    auto mat_c = G.GetTensor("mat_c");
    mat_c->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a");
    auto l1_a = G.GetTensor("l1_a");
    l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a");
    auto l0_a = G.GetTensor("l0_a");
    l0_a->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b");
    auto l1_b = G.GetTensor("l1_b");
    l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b");
    auto l0_b = G.GetTensor("l0_b");
    l0_b->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c");
    auto l0_c = G.GetTensor("l0_c");
    l0_c->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a"}, {"l0_a"}, "L1_To_L0A");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b"}, {"l0_b"}, "L1_To_L0B");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a", "l0_b"}, {"l0_c"}, "A_MUL_B");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c"}, {"mat_c"}, "L0C_Copy_out");
    SetMatMulAttr(G, "A_MUL_B", false, 1);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c"});
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    ProcessAtomic passLocal;
    passLocal.Run(*function, "", "", 0);
    CubeProcess cubeProcess;
    cubeProcess.UpdateCubeOp(*function);
    // check after pass
    auto opL1CopyInA = G.GetOp("L1_Copy_In_A");
    EXPECT_NE(opL1CopyInA, nullptr);
    EXPECT_EQ(opL1CopyInA->GetIntAttribute(COPY_IS_NZ), 1);
    auto opL1CopyInB = G.GetOp("L1_Copy_In_B");
    EXPECT_NE(opL1CopyInB, nullptr);
    EXPECT_EQ(opL1CopyInB->GetIntAttribute(COPY_IS_NZ), 0);
}

TEST_F(ProcessAtomicTest, TestAnzBndL1)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_0");
    auto mat_c_0 = G.GetTensor("mat_c_0");
    mat_c_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_1");
    auto mat_c_1 = G.GetTensor("mat_c_1");
    mat_c_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a");
    auto l1_a = G.GetTensor("l1_a");
    l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b");
    auto l1_b = G.GetTensor("l1_b");
    l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a_0");
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b_0");
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a_1");
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b_1");
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a_0");
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b_0");
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_0");
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a_1");
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b_1");
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_1");
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_VIEW, {"l1_a"}, {"l1_a_0", "l1_a_1"}, "A_OP_VIEW");
    G.AddOp(Opcode::OP_VIEW, {"l1_b"}, {"l1_b_0", "l1_b_1"}, "B_OP_VIEW");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_1"}, "L0C_Copy_out_1");
    SetMatMulAttr(G, "A_MUL_B_0", false, 1);
    SetMatMulAttr(G, "A_MUL_B_1", false, 1);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_0", "mat_c_1"});
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    ProcessAtomic passLocal;
    passLocal.Run(*function, "", "", 0);
    CubeProcess cubeProcess;
    cubeProcess.UpdateCubeOp(*function);
    // check after pass
    auto opL1CopyInA = G.GetOp("L1_Copy_In_A");
    EXPECT_NE(opL1CopyInA, nullptr);
    EXPECT_EQ(opL1CopyInA->GetIntAttribute(COPY_IS_NZ), 1);
    auto opL1CopyInB = G.GetOp("L1_Copy_In_B");
    EXPECT_NE(opL1CopyInB, nullptr);
    EXPECT_EQ(opL1CopyInB->GetIntAttribute(COPY_IS_NZ), 0);
}

TEST_F(ProcessAtomicTest, TestAndBndCnz)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_0");
    auto mat_c_0 = G.GetTensor("mat_c_0");
    mat_c_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_1");
    auto mat_c_1 = G.GetTensor("mat_c_1");
    mat_c_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a");
    auto l1_a = G.GetTensor("l1_a");
    l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b");
    auto l1_b = G.GetTensor("l1_b");
    l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a_0");
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b_0");
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a_1");
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b_1");
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a_0");
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b_0");
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_0");
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a_1");
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b_1");
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_1");
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_VIEW, {"l1_a"}, {"l1_a_0", "l1_a_1"}, "A_OP_VIEW");
    G.AddOp(Opcode::OP_VIEW, {"l1_b"}, {"l1_b_0", "l1_b_1"}, "B_OP_VIEW");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_1"}, "L0C_Copy_out_1");
    SetMatMulAttr(G, "A_MUL_B_0", false, 4);
    SetMatmulMatrixSize(G, "A_MUL_B_0", {64, 128, 128});
    SetMatMulAttr(G, "A_MUL_B_1", false, 4);
    SetMatmulMatrixSize(G, "A_MUL_B_1", {64, 128, 128});
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_0", "mat_c_1"});
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    ProcessAtomic passLocal;
    passLocal.Run(*function, "", "", 0);
    CubeProcess cubeProcess;
    cubeProcess.UpdateCubeOp(*function);
    // check after pass
    auto opL1CopyInA = G.GetOp("L1_Copy_In_A");
    EXPECT_NE(opL1CopyInA, nullptr);
    EXPECT_EQ(opL1CopyInA->GetIntAttribute(COPY_IS_NZ), 0);
    auto opL1CopyInB = G.GetOp("L1_Copy_In_B");
    EXPECT_NE(opL1CopyInB, nullptr);
    EXPECT_EQ(opL1CopyInB->GetIntAttribute(COPY_IS_NZ), 0);
    auto opL0cCopyOut0 = G.GetOp("L0C_Copy_out_0");
    EXPECT_NE(opL0cCopyOut0, nullptr);
    EXPECT_EQ(opL0cCopyOut0->GetIntAttribute(COPY_IS_NZ), 1);
    EXPECT_EQ(opL0cCopyOut0->GetIntAttribute(L0C_COPY_OUT_OUTER), 64);
    EXPECT_EQ(opL0cCopyOut0->GetIntAttribute(L0C_COPY_OUT_INNER), 128);
    auto opL0cCopyOut1 = G.GetOp("L0C_Copy_out_1");
    EXPECT_NE(opL0cCopyOut1, nullptr);
    EXPECT_EQ(opL0cCopyOut1->GetIntAttribute(COPY_IS_NZ), 1);
    EXPECT_EQ(opL0cCopyOut1->GetIntAttribute(L0C_COPY_OUT_OUTER), 64);
    EXPECT_EQ(opL0cCopyOut1->GetIntAttribute(L0C_COPY_OUT_INNER), 128);
}

TEST_F(ProcessAtomicTest, TestGatherOnL1)
{
    ComputationalGraphBuilder G;
    // INCAST mat_a, mat_b, OUTCAST mat_c
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {1024, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_c");
    auto mat_c = G.GetTensor("mat_c");
    mat_c->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    // paritial mat_a on L1
    G.AddTensor(outputAstDtype, {64, 64}, "mat_a_partial_0");
    auto mat_a_partial_0 = G.GetTensor("mat_a_partial_0");
    mat_a_partial_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(outputAstDtype, {64, 64}, "mat_a_partial_1");
    auto mat_a_partial_1 = G.GetTensor("mat_a_partial_1");
    mat_a_partial_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(outputAstDtype, {64, 64}, "mat_a_partial_2");
    auto mat_a_partial_2 = G.GetTensor("mat_a_partial_2");
    mat_a_partial_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(outputAstDtype, {64, 64}, "mat_a_partial_3");
    auto mat_a_partial_3 = G.GetTensor("mat_a_partial_3");
    mat_a_partial_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    // Contract on L1
    G.AddTensor(outputAstDtype, {128, 128}, "mat_a_L1");
    auto mat_a_L1 = G.GetTensor("mat_a_L1");
    mat_a_L1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(outputAstDtype, {128, 128}, "mat_b_L1");
    auto mat_b_L1 = G.GetTensor("mat_b_L1");
    mat_b_L1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    // L0
    G.AddTensor(outputAstDtype, {128, 128}, "mat_a_L0");
    auto mat_a_L0 = G.GetTensor("mat_a_L0");
    mat_a_L0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(outputAstDtype, {128, 128}, "mat_b_L0");
    auto mat_b_L0 = G.GetTensor("mat_b_L0");
    mat_b_L0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {128, 128}, "mat_c_L0");
    auto mat_c_L0 = G.GetTensor("mat_c_L0");
    mat_c_L0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);

    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"mat_a_partial_0"}, "L1copyInA_0");
    auto L1copyInA_0 = G.GetOp("L1copyInA_0");
    auto attrCopyInA_0 = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({256, 0}), MemoryType::MEM_L1,
                                                           OpImmediate::Specified(mat_a->GetShape()),
                                                           OpImmediate::Specified(mat_a->tensor->GetRawShape()));
    L1copyInA_0->SetOpAttribute(attrCopyInA_0);
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"mat_a_partial_1"}, "L1copyInA_1");
    auto L1copyInA_1 = G.GetOp("L1copyInA_1");
    auto attrCopyInA_1 = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({256, 64}), MemoryType::MEM_L1,
                                                           OpImmediate::Specified(mat_a->GetShape()),
                                                           OpImmediate::Specified(mat_a->tensor->GetRawShape()));
    L1copyInA_1->SetOpAttribute(attrCopyInA_1);
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"mat_a_partial_2"}, "L1copyInA_2");
    auto L1copyInA_2 = G.GetOp("L1copyInA_2");
    auto attrCopyInA_2 = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({512, 0}), MemoryType::MEM_L1,
                                                           OpImmediate::Specified(mat_a->GetShape()),
                                                           OpImmediate::Specified(mat_a->tensor->GetRawShape()));
    L1copyInA_2->SetOpAttribute(attrCopyInA_2);
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"mat_a_partial_3"}, "L1copyInA_3");
    auto L1copyInA_3 = G.GetOp("L1copyInA_3");
    auto attrCopyInA_3 = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({512, 64}), MemoryType::MEM_L1,
                                                           OpImmediate::Specified(mat_a->GetShape()),
                                                           OpImmediate::Specified(mat_a->tensor->GetRawShape()));
    L1copyInA_3->SetOpAttribute(attrCopyInA_3);

    G.AddOp(Opcode::OP_CONTRACT, {"mat_a_partial_0"}, {"mat_a_L1"}, "contract_A_0");
    auto contract_A_0 = G.GetOp("contract_A_0");
    auto attrContract_0 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 0});
    contract_A_0->SetOpAttribute(attrContract_0);
    G.AddOp(Opcode::OP_CONTRACT, {"mat_a_partial_1"}, {"mat_a_L1"}, "contract_A_1");
    auto contract_A_1 = G.GetOp("contract_A_1");
    auto attrContract_1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 64});
    contract_A_1->SetOpAttribute(attrContract_1);
    G.AddOp(Opcode::OP_CONTRACT, {"mat_a_partial_2"}, {"mat_a_L1"}, "contract_A_2");
    auto contract_A_2 = G.GetOp("contract_A_2");
    auto attrContract_2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{64, 0});
    contract_A_2->SetOpAttribute(attrContract_2);
    G.AddOp(Opcode::OP_CONTRACT, {"mat_a_partial_3"}, {"mat_a_L1"}, "contract_A_3");
    auto contract_A_3 = G.GetOp("contract_A_3");
    auto attrContract_3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{64, 64});
    contract_A_3->SetOpAttribute(attrContract_3);

    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"mat_b_L1"}, "L1_Copy_In_B");
    auto L1copyInB = G.GetOp("L1_Copy_In_B");
    auto attrCopyInB = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
                                                         OpImmediate::Specified(mat_b->GetShape()),
                                                         OpImmediate::Specified(mat_b->tensor->GetRawShape()));
    L1copyInB->SetOpAttribute(attrCopyInB);

    G.AddOp(Opcode::OP_L1_TO_L0A, {"mat_a_L1"}, {"mat_a_L0"}, "L1_To_L0A");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"mat_b_L1"}, {"mat_b_L0"}, "L1_To_L0B");

    G.AddOp(Opcode::OP_A_MUL_B, {"mat_a_L0", "mat_b_L0"}, {"mat_c_L0"}, "A_MUL_B");
    SetMatMulAttr(G, "A_MUL_B", false, 0);

    G.AddOp(Opcode::OP_COPY_OUT, {"mat_c_L0"}, {"mat_c"}, "L0C_Copy_out");
    auto copyOutOp = G.GetOp("L0C_Copy_out");
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L0C,
                                                         OpImmediate::Specified(mat_c->GetShape()),
                                                         OpImmediate::Specified(mat_c->tensor->GetRawShape()));
    copyOutOp->SetOpAttribute(attrCopyOut);

    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c"});
    // check before pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    // run pass
    ProcessAtomic passLocal;
    Status res = passLocal.Run(*function, "", "", 0);
    CubeProcess cubeProcess;
    cubeProcess.UpdateCubeOp(*function);
    // check after pass
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(mat_c->Datatype(), outputAstDtype);
    EXPECT_EQ(mat_c_L0->Datatype(), DataType::DT_FP32);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWBasic)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;

    G.AddTensor(dtype, {64, 128}, "inputDdr");
    auto inputDdr = G.GetTensor("inputDdr");
    inputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {64, 128}, "contractInput");
    auto contractInput = G.GetTensor("contractInput");
    contractInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {128, 128}, "outputDdr");
    auto outputDdr = G.GetTensor("outputDdr");
    outputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_CONTRACT, {"contractInput"}, {"inputDdr"}, "contractOp");
    auto contractOp = G.GetOp("contractOp");
    auto contractAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    contractOp->SetOpAttribute(contractAttr);

    G.AddOp(Opcode::OP_ATOMIC_RMW, {"inputDdr"}, {"outputDdr"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    auto atomicRmwAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0});
    atomicRmwOp->SetOpAttribute(atomicRmwAttr);
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, (int)AtomicRMWMode::ADD);

    G.SetInCast({"contractInput"});
    G.SetOutCast({"outputDdr"});

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    int atomicRmwCount = 0;

    ProcessAtomic passLocal;
    Status preCheckResult = passLocal.PreCheck(*function);
    EXPECT_EQ(preCheckResult, SUCCESS);
    bool hasReduceAccCascade = false;
    EXPECT_EQ(passLocal.EliminateVecDupBranch(*function, hasReduceAccCascade), SUCCESS);
    EXPECT_FALSE(hasReduceAccCascade);
    EXPECT_EQ(passLocal.EliminateReduceAcc(*function), SUCCESS);
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*function), SUCCESS);

    atomicRmwCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            atomicRmwCount++;
        }
    }
    EXPECT_EQ(atomicRmwCount, 0);

    auto updatedContractOp = G.GetOp("contractOp");
    EXPECT_NE(updatedContractOp, nullptr);
    EXPECT_EQ(updatedContractOp->HasAttr(RMW_MODE_ATTR_ADD), true);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWMaxModeUnsupported)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;

    G.AddTensor(dtype, {64, 128}, "inputDdr");
    auto inputDdr = G.GetTensor("inputDdr");
    inputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {64, 128}, "contractInput");
    auto contractInput = G.GetTensor("contractInput");
    contractInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {128, 128}, "outputDdr");
    auto outputDdr = G.GetTensor("outputDdr");
    outputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_CONTRACT, {"contractInput"}, {"inputDdr"}, "contractOp");
    auto contractOp = G.GetOp("contractOp");
    auto contractAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    contractOp->SetOpAttribute(contractAttr);

    G.AddOp(Opcode::OP_ATOMIC_RMW, {"inputDdr"}, {"outputDdr"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    auto atomicRmwAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0});
    atomicRmwOp->SetOpAttribute(atomicRmwAttr);
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, (int)AtomicRMWMode::MAX);

    G.SetInCast({"contractInput"});
    G.SetOutCast({"outputDdr"});

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    ProcessAtomic passLocal;
    Status preCheckResult = passLocal.PreCheck(*function);
    EXPECT_EQ(preCheckResult, SUCCESS);
    Status runResult = passLocal.Run(*function, "", "", 0);
    EXPECT_NE(runResult, SUCCESS);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWMinModeUnsupported)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;

    G.AddTensor(dtype, {64, 128}, "inputDdr");
    auto inputDdr = G.GetTensor("inputDdr");
    inputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {64, 128}, "contractInput");
    auto contractInput = G.GetTensor("contractInput");
    contractInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {128, 128}, "outputDdr");
    auto outputDdr = G.GetTensor("outputDdr");
    outputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_CONTRACT, {"contractInput"}, {"inputDdr"}, "contractOp");
    auto contractOp = G.GetOp("contractOp");
    auto contractAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    contractOp->SetOpAttribute(contractAttr);

    G.AddOp(Opcode::OP_ATOMIC_RMW, {"inputDdr"}, {"outputDdr"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    auto atomicRmwAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0});
    atomicRmwOp->SetOpAttribute(atomicRmwAttr);
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, (int)AtomicRMWMode::MIN);

    G.SetInCast({"contractInput"});
    G.SetOutCast({"outputDdr"});

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    ProcessAtomic passLocal;
    Status preCheckResult = passLocal.PreCheck(*function);
    EXPECT_EQ(preCheckResult, SUCCESS);
    Status runResult = passLocal.Run(*function, "", "", 0);
    EXPECT_NE(runResult, SUCCESS);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWInvalidInputMemory)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;

    G.AddTensor(dtype, {64, 128}, "inputUb");
    auto inputUb = G.GetTensor("inputUb");
    inputUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    G.AddTensor(dtype, {128, 128}, "outputDdr");
    auto outputDdr = G.GetTensor("outputDdr");
    outputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_ATOMIC_RMW, {"inputUb"}, {"outputDdr"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    auto atomicRmwAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    atomicRmwOp->SetOpAttribute(atomicRmwAttr);
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, (int)AtomicRMWMode::ADD);

    G.SetInCast({"inputUb"});
    G.SetOutCast({"outputDdr"});

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    ProcessAtomic passLocal;
    Status preCheckResult = passLocal.PreCheck(*function);
    EXPECT_NE(preCheckResult, SUCCESS);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWInvalidOutputMemory)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;

    G.AddTensor(dtype, {64, 128}, "inputDdr");
    auto inputDdr = G.GetTensor("inputDdr");
    inputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {128, 128}, "outputUb");
    auto outputUb = G.GetTensor("outputUb");
    outputUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    G.AddOp(Opcode::OP_ATOMIC_RMW, {"inputDdr"}, {"outputUb"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    auto atomicRmwAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    atomicRmwOp->SetOpAttribute(atomicRmwAttr);
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, (int)AtomicRMWMode::ADD);

    G.SetInCast({"inputDdr"});
    G.SetOutCast({"outputUb"});

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    ProcessAtomic passLocal;
    Status preCheckResult = passLocal.PreCheck(*function);
    EXPECT_NE(preCheckResult, SUCCESS);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWWithReduceAcc)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;

    G.AddTensor(dtype, {64, 128}, "matCBeforeReduce0");
    auto matCBeforeReduce0 = G.GetTensor("matCBeforeReduce0");
    matCBeforeReduce0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {64, 128}, "matCBeforeReduce1");
    auto matCBeforeReduce1 = G.GetTensor("matCBeforeReduce1");
    matCBeforeReduce1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {64, 128}, "matCAfterReduce");
    auto matCAfterReduce = G.GetTensor("matCAfterReduce");
    matCAfterReduce->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {64, 128}, "contractInput");
    auto contractInput = G.GetTensor("contractInput");
    contractInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {128, 128}, "atomicOutput");
    auto atomicOutput = G.GetTensor("atomicOutput");
    atomicOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_REDUCE_ACC, {"matCBeforeReduce0", "matCBeforeReduce1"}, {"matCAfterReduce"}, "ReduceAcc");

    G.AddOp(Opcode::OP_CONTRACT, {"contractInput"}, {"matCAfterReduce"}, "contractOp");
    auto contractOp = G.GetOp("contractOp");
    auto contractAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    contractOp->SetOpAttribute(contractAttr);

    G.AddOp(Opcode::OP_ATOMIC_RMW, {"matCAfterReduce"}, {"atomicOutput"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    auto atomicRmwAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0});
    atomicRmwOp->SetOpAttribute(atomicRmwAttr);
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, (int)AtomicRMWMode::ADD);

    G.SetInCast({"matCBeforeReduce0", "matCBeforeReduce1", "contractInput"});
    G.SetOutCast({"atomicOutput"});

    Function* function = G.GetFunction();

    ProcessAtomic passLocal;
    bool hasReduceAccCascade = false;
    EXPECT_EQ(passLocal.EliminateVecDupBranch(*function, hasReduceAccCascade), SUCCESS);
    EXPECT_TRUE(hasReduceAccCascade);
    EXPECT_EQ(passLocal.EliminateReduceAcc(*function), SUCCESS);
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*function), SUCCESS);

    int reduceAccCount = 0;
    int atomicRmwCount = 0;
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            reduceAccCount++;
        }
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            atomicRmwCount++;
        }
    }
    EXPECT_EQ(reduceAccCount, 0);
    EXPECT_EQ(atomicRmwCount, 0);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWModeConflict)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;

    G.AddTensor(dtype, {64, 128}, "contractInput");
    auto contractInput = G.GetTensor("contractInput");
    contractInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {64, 128}, "inputDdr");
    auto inputDdr = G.GetTensor("inputDdr");
    inputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {128, 128}, "outputDdr");
    auto outputDdr = G.GetTensor("outputDdr");
    outputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_CONTRACT, {"contractInput"}, {"inputDdr"}, "contractOp");
    auto contractOp = G.GetOp("contractOp");
    auto contractAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    contractOp->SetOpAttribute(contractAttr);
    contractOp->SetAttribute(RMW_MODE_ATTR_ADD, 1);

    G.AddOp(Opcode::OP_ATOMIC_RMW, {"inputDdr"}, {"outputDdr"}, "atomicRmwMax");
    auto atomicRmwMaxOp = G.GetOp("atomicRmwMax");
    auto atomicRmwMaxAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0});
    atomicRmwMaxOp->SetOpAttribute(atomicRmwMaxAttr);
    atomicRmwMaxOp->SetAttribute(OpAttributeKey::rmwMode, (int)AtomicRMWMode::MAX);

    G.SetInCast({"contractInput"});
    G.SetOutCast({"outputDdr"});

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    ProcessAtomic passLocal;
    Status result = passLocal.Run(*function, "", "", 0);
    EXPECT_NE(result, SUCCESS);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWSameModeNoConflict)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;

    G.AddTensor(dtype, {64, 128}, "contractInput");
    auto contractInput = G.GetTensor("contractInput");
    contractInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {64, 128}, "inputDdr");
    auto inputDdr = G.GetTensor("inputDdr");
    inputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {128, 128}, "outputDdr");
    auto outputDdr = G.GetTensor("outputDdr");
    outputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_CONTRACT, {"contractInput"}, {"inputDdr"}, "contractOp");
    auto contractOp = G.GetOp("contractOp");
    auto contractAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    contractOp->SetOpAttribute(contractAttr);
    contractOp->SetAttribute(RMW_MODE_ATTR_ADD, 1);

    G.AddOp(Opcode::OP_ATOMIC_RMW, {"inputDdr"}, {"outputDdr"}, "atomicRmwAdd");
    auto atomicRmwAddOp = G.GetOp("atomicRmwAdd");
    auto atomicRmwAddAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0});
    atomicRmwAddOp->SetOpAttribute(atomicRmwAddAttr);
    atomicRmwAddOp->SetAttribute(OpAttributeKey::rmwMode, (int)AtomicRMWMode::ADD);

    G.SetInCast({"contractInput"});
    G.SetOutCast({"outputDdr"});

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    ProcessAtomic passLocal;
    bool hasReduceAccCascade = false;
    EXPECT_EQ(passLocal.EliminateVecDupBranch(*function, hasReduceAccCascade), SUCCESS);
    EXPECT_FALSE(hasReduceAccCascade);
    EXPECT_EQ(passLocal.EliminateReduceAcc(*function), SUCCESS);
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*function), SUCCESS);

    int atomicRmwCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            atomicRmwCount++;
        }
    }
    EXPECT_EQ(atomicRmwCount, 0);

    auto updatedContractOp = G.GetOp("contractOp");
    EXPECT_NE(updatedContractOp, nullptr);
    EXPECT_EQ(updatedContractOp->HasAttr(RMW_MODE_ATTR_ADD), true);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWSharedInputCloneContractProducer)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;

    auto contractInput = AddDdrTensor(G, dtype, {64, 128}, "contractInput");
    auto inputDdr = AddDdrTensor(G, dtype, {64, 128}, "inputDdr");
    auto viewOut = AddDdrTensor(G, dtype, {64, 128}, "viewOut");
    auto outputDdr = AddDdrTensor(G, dtype, {128, 128}, "outputDdr");

    G.AddOp(Opcode::OP_CONTRACT, {"contractInput"}, {"inputDdr"}, "contractOp");
    auto contractOp = G.GetOp("contractOp");
    contractOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    contractOp->SetScopeId(7);
    const std::string copyAttr = OP_ATTR_PREFIX + "copy_test_attr";
    contractOp->SetAttribute(copyAttr, 123L);

    G.AddOp(Opcode::OP_VIEW, {"inputDdr"}, {"viewOut"}, "viewOp");
    G.GetOp("viewOp")->SetOpAttribute(std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{}));

    G.AddOp(Opcode::OP_ATOMIC_RMW, {"inputDdr"}, {"outputDdr"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    atomicRmwOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0}));
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, static_cast<int>(AtomicRMWMode::ADD));

    G.SetInCast({"contractInput"});
    G.SetOutCast({"outputDdr", "viewOut"});

    ProcessAtomic passLocal;
    EXPECT_EQ(passLocal.CheckAtomicRMWUnsupportedMode(*G.GetFunction()), SUCCESS);
    bool hasReduceAccCascade = false;
    EXPECT_EQ(passLocal.EliminateVecDupBranch(*G.GetFunction(), hasReduceAccCascade), SUCCESS);
    EXPECT_FALSE(hasReduceAccCascade);
    EXPECT_EQ(passLocal.EliminateReduceAcc(*G.GetFunction()), SUCCESS);
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*G.GetFunction()), SUCCESS);

    Operation* atomicContract = nullptr;
    Operation* viewContract = nullptr;
    for (auto& op : G.GetFunction()->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            if (op.GetOutputOperand(0) == outputDdr)
                atomicContract = &op;
            if (op.GetOutputOperand(0) == inputDdr)
                viewContract = &op;
        }
    }

    ASSERT_NE(atomicContract, nullptr);
    ASSERT_NE(viewContract, nullptr);
    EXPECT_TRUE(atomicContract->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_FALSE(viewContract->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_EQ(atomicContract->GetScopeId(), 7);
    EXPECT_TRUE(atomicContract->HasAttr(copyAttr));
    EXPECT_EQ(G.GetOp("viewOp")->GetInputOperand(0), inputDdr);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWReduceAccChainVecDupBranchRemove)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;
    auto matb1 = AddDdrTensor(G, dtype, {128, 128}, "matb1");
    auto matb2 = AddDdrTensor(G, dtype, {128, 128}, "matb2");
    auto matb3 = AddDdrTensor(G, dtype, {128, 128}, "matb3");
    AddDdrTensor(G, dtype, {1}, "vecdupIn");
    AddDdrTensor(G, dtype, {64, 128}, "vecdupOut");
    AddDdrTensor(G, dtype, {64, 128}, "contractOut1");
    AddDdrTensor(G, dtype, {64, 128}, "mulbOut");
    AddDdrTensor(G, dtype, {64, 128}, "mulaccOut");
    AddDdrTensor(G, dtype, {64, 128}, "contractOut2");
    AddDdrTensor(G, dtype, {64, 128}, "reduceOut");
    AddDdrTensor(G, dtype, {64, 128}, "atomicOut");
    G.AddOp(Opcode::OP_VEC_DUP, {"vecdupIn"}, {"vecdupOut"}, "vecdupOp");
    G.AddOp(Opcode::OP_CONTRACT, {"vecdupOut"}, {"contractOut1"}, "contractOp1");
    G.GetOp("contractOp1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_A_MUL_B, {"contractOut1", "matb1", "matb2"}, {"mulbOut"}, "mulbOp");
    G.AddOp(Opcode::OP_A_MULACC_B, {"mulbOut", "matb3"}, {"mulaccOut"}, "mulaccOp");
    G.AddOp(Opcode::OP_CONTRACT, {"mulaccOut"}, {"contractOut2"}, "contractOp2");
    G.GetOp("contractOp2")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_REDUCE_ACC, {"contractOut2"}, {"reduceOut"}, "reduceOp");
    G.AddOp(Opcode::OP_ATOMIC_RMW, {"reduceOut"}, {"atomicOut"}, "atomicrmwOp");
    G.GetOp("atomicrmwOp")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0}));
    G.GetOp("atomicrmwOp")->SetAttribute(OpAttributeKey::rmwMode, static_cast<int>(AtomicRMWMode::ADD));
    G.SetInCast({"vecdupIn", "matb1", "matb2", "matb3"});
    G.SetOutCast({"atomicOut"});
    ProcessAtomic passLocal;
    auto* function = G.GetFunction();
    bool hasReduceAccCascade = false;
    EXPECT_EQ(passLocal.EliminateVecDupBranch(*function, hasReduceAccCascade), SUCCESS);
    EXPECT_TRUE(hasReduceAccCascade);
    EXPECT_EQ(passLocal.EliminateReduceAcc(*function), SUCCESS);
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*function), SUCCESS);
    auto mulbOp = G.GetOp("mulbOp");
    ASSERT_NE(mulbOp, nullptr);
    EXPECT_EQ(mulbOp->GetInputOperandSize(), 2);
    EXPECT_EQ(mulbOp->GetInputOperand(0), matb1);
    EXPECT_EQ(mulbOp->GetInputOperand(1), matb2);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_ATOMIC_RMW), 0);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_REDUCE_ACC), 0);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_VEC_DUP), 0);
    auto contractOp2 = G.GetOp("contractOp2");
    ASSERT_NE(contractOp2, nullptr);
    EXPECT_TRUE(contractOp2->HasAttr(RMW_MODE_ATTR_ADD));
}

TEST_F(ProcessAtomicTest, TestAtomicRMWNoReduceAccVecDupBranchKeep)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;
    auto matb1 = AddDdrTensor(G, dtype, {128, 128}, "matb1");
    auto matb2 = AddDdrTensor(G, dtype, {128, 128}, "matb2");
    AddDdrTensor(G, dtype, {1}, "vecdupIn");
    AddDdrTensor(G, dtype, {64, 128}, "vecdupOut");
    AddDdrTensor(G, dtype, {64, 128}, "contractOut1");
    AddDdrTensor(G, dtype, {64, 128}, "mulbOut");
    AddDdrTensor(G, dtype, {64, 128}, "contractOut2");
    AddDdrTensor(G, dtype, {64, 128}, "atomicOut");
    G.AddOp(Opcode::OP_VEC_DUP, {"vecdupIn"}, {"vecdupOut"}, "vecdupOp");
    G.AddOp(Opcode::OP_CONTRACT, {"vecdupOut"}, {"contractOut1"}, "contractOp1");
    G.GetOp("contractOp1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_A_MUL_B, {"contractOut1", "matb1", "matb2"}, {"mulbOut"}, "mulbOp");
    G.AddOp(Opcode::OP_CONTRACT, {"mulbOut"}, {"contractOut2"}, "contractOp2");
    G.GetOp("contractOp2")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_ATOMIC_RMW, {"contractOut2"}, {"atomicOut"}, "atomicrmwOp");
    G.GetOp("atomicrmwOp")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0}));
    G.GetOp("atomicrmwOp")->SetAttribute(OpAttributeKey::rmwMode, static_cast<int>(AtomicRMWMode::ADD));
    G.SetInCast({"vecdupIn", "matb1", "matb2"});
    G.SetOutCast({"atomicOut"});
    ProcessAtomic passLocal;
    auto* function = G.GetFunction();
    bool hasReduceAccCascade = false;
    EXPECT_EQ(passLocal.EliminateVecDupBranch(*function, hasReduceAccCascade), SUCCESS);
    EXPECT_FALSE(hasReduceAccCascade);
    EXPECT_EQ(passLocal.EliminateReduceAcc(*function), SUCCESS);
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*function), SUCCESS);
    auto mulbOp = G.GetOp("mulbOp");
    ASSERT_NE(mulbOp, nullptr);
    EXPECT_EQ(mulbOp->GetInputOperandSize(), 3);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_VEC_DUP), 1);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWThroughReshape)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP32;
    G.AddTensor(dtype, {16, 32}, "assembleInput");
    G.AddTensor(dtype, {16, 32}, "reshapeInput");
    G.AddTensor(dtype, {1, 16, 32}, "reshapeMid");
    G.AddTensor(dtype, {2, 16, 32}, "atomicOutput");
    auto assembleInput = G.GetTensor("assembleInput");
    auto reshapeInput = G.GetTensor("reshapeInput");
    assembleInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto reshapeMid = G.GetTensor("reshapeMid");
    auto atomicOutput = G.GetTensor("atomicOutput");
    reshapeInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    reshapeMid->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    atomicOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_CONTRACT, {"assembleInput"}, {"reshapeInput"}, "assembleOp");
    G.GetOp("assembleOp")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_RESHAPE, {"reshapeInput"}, {"reshapeMid"}, "reshapeOp");
    G.AddOp(Opcode::OP_ATOMIC_RMW, {"reshapeMid"}, {"atomicOutput"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    atomicRmwOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0, 0}));
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, static_cast<int>(AtomicRMWMode::ADD));
    G.SetInCast({"assembleInput"});
    G.SetOutCast({"atomicOutput"});

    auto* function = G.GetFunction();
    ProcessAtomic passLocal;
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*function), SUCCESS);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_ATOMIC_RMW), 0);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_RESHAPE), 1);

    auto* assembleOp = G.GetOp("assembleOp");
    EXPECT_NE(assembleOp, nullptr);
    EXPECT_TRUE(assembleOp->HasAttr(RMW_MODE_ATTR_ADD));
    auto updatedAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(assembleOp->GetOpAttribute());
    ASSERT_NE(updatedAttr, nullptr);
    EXPECT_EQ(updatedAttr->GetToOffset(), (std::vector<int64_t>{16, 0}));
    EXPECT_NE(assembleOp->GetOutputOperand(0)->GetRawTensor(), atomicOutput->GetRawTensor());
    EXPECT_EQ(assembleOp->GetOutputOperand(0)->GetShape(), (std::vector<int64_t>{32, 32}));
    EXPECT_EQ(assembleOp->GetOutputOperand(0)->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(assembleOp->GetOutputOperand(0)->GetMemoryTypeToBe(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(G.GetOp("reshapeOp")->GetOutputOperand(0), atomicOutput);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_CONTRACT), 1);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWThroughReshapeWithAssembleFanin)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP32;
    G.AddTensor(dtype, {16, 16}, "assembleInput0");
    G.AddTensor(dtype, {16, 16}, "assembleInput1");
    G.AddTensor(dtype, {16, 32}, "reshapeInput");
    G.AddTensor(dtype, {1, 16, 32}, "reshapeOutput");
    G.AddTensor(dtype, {2, 16, 32}, "atomicOutput");
    for (const char* name : {"assembleInput0", "assembleInput1", "reshapeInput", "reshapeOutput", "atomicOutput"}) {
        G.GetTensor(name)->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    G.AddOp(Opcode::OP_CONTRACT, {"assembleInput0"}, {"reshapeInput"}, "assembleOp0");
    G.GetOp("assembleOp0")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_CONTRACT, {"assembleInput1"}, {"reshapeInput"}, "assembleOp1");
    G.GetOp("assembleOp1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 16}));
    G.AddOp(Opcode::OP_RESHAPE, {"reshapeInput"}, {"reshapeOutput"}, "reshapeOp");
    G.AddOp(Opcode::OP_ATOMIC_RMW, {"reshapeOutput"}, {"atomicOutput"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    atomicRmwOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0, 0}));
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, static_cast<int>(AtomicRMWMode::ADD));
    G.SetInCast({"assembleInput0", "assembleInput1"});
    G.SetOutCast({"atomicOutput"});

    auto* function = G.GetFunction();
    auto atomicOutput = G.GetTensor("atomicOutput");
    ProcessAtomic passLocal;
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*function), SUCCESS);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_ATOMIC_RMW), 0);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_CONTRACT), 2);

    auto* assembleOp0 = G.GetOp("assembleOp0");
    auto* assembleOp1 = G.GetOp("assembleOp1");
    ASSERT_NE(assembleOp0, nullptr);
    ASSERT_NE(assembleOp1, nullptr);
    EXPECT_TRUE(assembleOp0->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_TRUE(assembleOp1->HasAttr(RMW_MODE_ATTR_ADD));
    auto attr0 = std::dynamic_pointer_cast<AssembleOpAttribute>(assembleOp0->GetOpAttribute());
    auto attr1 = std::dynamic_pointer_cast<AssembleOpAttribute>(assembleOp1->GetOpAttribute());
    ASSERT_NE(attr0, nullptr);
    ASSERT_NE(attr1, nullptr);
    EXPECT_EQ(attr0->GetToOffset(), (std::vector<int64_t>{16, 0}));
    EXPECT_EQ(attr1->GetToOffset(), (std::vector<int64_t>{16, 16}));
    EXPECT_EQ(assembleOp0->GetOutputOperand(0), assembleOp1->GetOutputOperand(0));
    EXPECT_EQ(assembleOp0->GetOutputOperand(0)->GetShape(), (std::vector<int64_t>{32, 32}));
    EXPECT_EQ(assembleOp0->GetOutputOperand(0)->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(G.GetOp("reshapeOp")->GetOutputOperand(0), atomicOutput);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWThroughMultiDimReshape)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP32;
    G.AddTensor(dtype, {8, 32}, "assembleInput");
    G.AddTensor(dtype, {8, 32}, "reshapeInput");
    G.AddTensor(dtype, {2, 4, 32}, "reshapeMid");
    G.AddTensor(dtype, {1, 2, 4, 32}, "atomicInput");
    G.AddTensor(dtype, {3, 2, 4, 32}, "atomicOutput");
    for (const char* name : {"assembleInput", "reshapeInput", "reshapeMid", "atomicInput", "atomicOutput"}) {
        G.GetTensor(name)->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }
    auto atomicOutput = G.GetTensor("atomicOutput");

    G.AddOp(Opcode::OP_CONTRACT, {"assembleInput"}, {"reshapeInput"}, "assembleOp");
    G.GetOp("assembleOp")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_RESHAPE, {"reshapeInput"}, {"reshapeMid"}, "reshapeOp0");
    G.AddOp(Opcode::OP_RESHAPE, {"reshapeMid"}, {"atomicInput"}, "reshapeOp1");
    G.AddOp(Opcode::OP_ATOMIC_RMW, {"atomicInput"}, {"atomicOutput"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    atomicRmwOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 1, 0, 0}));
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, static_cast<int>(AtomicRMWMode::ADD));
    G.SetInCast({"assembleInput"});
    G.SetOutCast({"atomicOutput"});

    auto* function = G.GetFunction();
    ProcessAtomic passLocal;
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*function), SUCCESS);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_ATOMIC_RMW), 0);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_RESHAPE), 2);

    auto* assembleOp = G.GetOp("assembleOp");
    ASSERT_NE(assembleOp, nullptr);
    EXPECT_TRUE(assembleOp->HasAttr(RMW_MODE_ATTR_ADD));
    auto updatedAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(assembleOp->GetOpAttribute());
    ASSERT_NE(updatedAttr, nullptr);
    EXPECT_EQ(updatedAttr->GetToOffset(), (std::vector<int64_t>{20, 0}));
    EXPECT_NE(assembleOp->GetOutputOperand(0)->GetRawTensor(), atomicOutput->GetRawTensor());
    EXPECT_EQ(assembleOp->GetOutputOperand(0)->GetShape(), (std::vector<int64_t>{24, 32}));
    EXPECT_EQ(assembleOp->GetOutputOperand(0)->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(assembleOp->GetOutputOperand(0)->GetMemoryTypeToBe(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(G.GetOp("reshapeOp0")->GetOutputOperand(0)->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(G.GetOp("reshapeOp0")->GetOutputOperand(0)->GetMemoryTypeToBe(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(G.GetOp("reshapeOp1")->GetOutputOperand(0), atomicOutput);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_CONTRACT), 1);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWWithReshapeBeforeAssemble)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP32;
    G.AddTensor(dtype, {16, 32}, "matA");
    G.AddTensor(dtype, {16, 32}, "matB");
    G.AddTensor(dtype, {16, 32}, "matmulOutput");
    G.AddTensor(dtype, {16, 32}, "transported");
    G.AddTensor(dtype, {1, 16, 32}, "reshaped");
    G.AddTensor(dtype, {1, 16, 32}, "assembled");
    G.AddTensor(dtype, {1, 16, 32}, "atomicOutput");
    for (const char* name : {"matA", "matB", "matmulOutput", "transported", "reshaped", "assembled", "atomicOutput"}) {
        G.GetTensor(name)->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    G.AddOp(Opcode::OP_A_MUL_B, {"matA", "matB"}, {"matmulOutput"}, "matmulOp");
    G.AddOp(Opcode::OP_CONTRACT, {"matmulOutput"}, {"transported"}, "transportAssemble");
    G.GetOp("transportAssemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_RESHAPE, {"transported"}, {"reshaped"}, "reshapeOp");
    G.AddOp(Opcode::OP_CONTRACT, {"reshaped"}, {"assembled"}, "wrapperAssemble");
    G.GetOp("wrapperAssemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0, 0}));
    G.AddOp(Opcode::OP_ATOMIC_RMW, {"assembled"}, {"atomicOutput"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    atomicRmwOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0, 0}));
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, static_cast<int>(AtomicRMWMode::ADD));
    G.SetInCast({"matA", "matB"});
    G.SetOutCast({"atomicOutput"});

    auto* function = G.GetFunction();
    auto atomicOutput = G.GetTensor("atomicOutput");
    ProcessAtomic passLocal;
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*function), SUCCESS);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_ATOMIC_RMW), 0);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_RESHAPE), 1);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_A_MUL_B), 1);

    auto* transportAssemble = G.GetOp("transportAssemble");
    EXPECT_NE(transportAssemble, nullptr);
    EXPECT_TRUE(transportAssemble->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_NE(transportAssemble->GetOutputOperand(0)->GetRawTensor(), atomicOutput->GetRawTensor());
    EXPECT_EQ(G.GetOp("reshapeOp")->GetOutputOperand(0), atomicOutput);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_CONTRACT), 1);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWWithContract)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;

    G.AddTensor(dtype, {64, 128}, "inputDdr");
    auto inputDdr = G.GetTensor("inputDdr");
    inputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {64, 128}, "contractInput");
    auto contractInput = G.GetTensor("contractInput");
    contractInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(dtype, {128, 128}, "outputDdr");
    auto outputDdr = G.GetTensor("outputDdr");
    outputDdr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_CONTRACT, {"contractInput"}, {"inputDdr"}, "contractOp");
    auto contractOp = G.GetOp("contractOp");
    auto contractAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    contractOp->SetOpAttribute(contractAttr);

    G.AddOp(Opcode::OP_ATOMIC_RMW, {"inputDdr"}, {"outputDdr"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    auto atomicRmwAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0});
    atomicRmwOp->SetOpAttribute(atomicRmwAttr);
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, (int)AtomicRMWMode::ADD);

    G.SetInCast({"contractInput"});
    G.SetOutCast({"outputDdr"});

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    ProcessAtomic passLocal;
    bool hasReduceAccCascade = false;
    EXPECT_EQ(passLocal.EliminateVecDupBranch(*function, hasReduceAccCascade), SUCCESS);
    EXPECT_FALSE(hasReduceAccCascade);
    EXPECT_EQ(passLocal.EliminateReduceAcc(*function), SUCCESS);
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*function), SUCCESS);

    int atomicRmwCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            atomicRmwCount++;
        }
    }
    EXPECT_EQ(atomicRmwCount, 0);

    auto updatedContractOp = G.GetOp("contractOp");
    EXPECT_NE(updatedContractOp, nullptr);
    EXPECT_EQ(updatedContractOp->HasAttr(RMW_MODE_ATTR_ADD), true);
}

TEST_F(ProcessAtomicTest, TestAtomicRMWContractVecDupBranchRemove)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;
    auto matb1 = AddDdrTensor(G, dtype, {128, 128}, "matb1");
    auto matb2 = AddDdrTensor(G, dtype, {128, 128}, "matb2");
    auto matb3 = AddDdrTensor(G, dtype, {128, 128}, "matb3");
    AddDdrTensor(G, dtype, {1}, "vecdupIn");
    AddDdrTensor(G, dtype, {64, 128}, "vecdupOut");
    AddDdrTensor(G, dtype, {64, 128}, "contractOut1");
    AddDdrTensor(G, dtype, {64, 128}, "mulbOut");
    AddDdrTensor(G, dtype, {64, 128}, "mulaccOut");
    AddDdrTensor(G, dtype, {64, 128}, "contractOut2");
    AddDdrTensor(G, dtype, {64, 128}, "reduceOut");
    AddDdrTensor(G, dtype, {64, 128}, "atomicOut");
    G.AddOp(Opcode::OP_VEC_DUP, {"vecdupIn"}, {"vecdupOut"}, "vecdupOp");
    G.AddOp(Opcode::OP_CONTRACT, {"vecdupOut"}, {"contractOut1"}, "contractOp1");
    G.GetOp("contractOp1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_A_MUL_B, {"contractOut1", "matb1", "matb2"}, {"mulbOut"}, "mulbOp");
    G.AddOp(Opcode::OP_A_MULACC_B, {"mulbOut", "matb3"}, {"mulaccOut"}, "mulaccOp");
    G.AddOp(Opcode::OP_CONTRACT, {"mulaccOut"}, {"contractOut2"}, "contractOp2");
    G.GetOp("contractOp2")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_REDUCE_ACC, {"contractOut2"}, {"reduceOut"}, "reduceOp");
    G.AddOp(Opcode::OP_ATOMIC_RMW, {"reduceOut"}, {"atomicOut"}, "atomicrmwOp");
    G.GetOp("atomicrmwOp")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0}));
    G.GetOp("atomicrmwOp")->SetAttribute(OpAttributeKey::rmwMode, static_cast<int>(AtomicRMWMode::ADD));
    G.SetInCast({"vecdupIn", "matb1", "matb2", "matb3"});
    G.SetOutCast({"atomicOut"});
    ProcessAtomic passLocal;
    auto* function = G.GetFunction();
    bool hasReduceAccCascade = false;
    EXPECT_EQ(passLocal.EliminateVecDupBranch(*function, hasReduceAccCascade), SUCCESS);
    EXPECT_TRUE(hasReduceAccCascade);
    EXPECT_EQ(passLocal.EliminateReduceAcc(*function), SUCCESS);
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*function), SUCCESS);
    auto mulbOp = G.GetOp("mulbOp");
    ASSERT_NE(mulbOp, nullptr);
    EXPECT_EQ(mulbOp->GetInputOperandSize(), 2);
    EXPECT_EQ(mulbOp->GetInputOperand(0), matb1);
    EXPECT_EQ(mulbOp->GetInputOperand(1), matb2);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_ATOMIC_RMW), 0);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_REDUCE_ACC), 0);
    EXPECT_EQ(CountOpsByType(function, Opcode::OP_VEC_DUP), 0);
    auto contractOp2 = G.GetOp("contractOp2");
    ASSERT_NE(contractOp2, nullptr);
    EXPECT_TRUE(contractOp2->HasAttr(RMW_MODE_ATTR_ADD));
}

TEST_F(ProcessAtomicTest, TestAtomicRMWContractSharedInputCloneProducer)
{
    ComputationalGraphBuilder G;
    DataType dtype = DataType::DT_FP16;

    auto contractInput = AddDdrTensor(G, dtype, {64, 128}, "contractInput");
    auto inputDdr = AddDdrTensor(G, dtype, {64, 128}, "inputDdr");
    auto viewOut = AddDdrTensor(G, dtype, {64, 128}, "viewOut");
    auto outputDdr = AddDdrTensor(G, dtype, {128, 128}, "outputDdr");

    G.AddOp(Opcode::OP_CONTRACT, {"contractInput"}, {"inputDdr"}, "contractOp");
    auto contractOp = G.GetOp("contractOp");
    contractOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    contractOp->SetScopeId(7);

    G.AddOp(Opcode::OP_VIEW, {"inputDdr"}, {"viewOut"}, "viewOp");
    G.GetOp("viewOp")->SetOpAttribute(std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{}));

    G.AddOp(Opcode::OP_ATOMIC_RMW, {"inputDdr"}, {"outputDdr"}, "atomicRmwOp");
    auto atomicRmwOp = G.GetOp("atomicRmwOp");
    atomicRmwOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0}));
    atomicRmwOp->SetAttribute(OpAttributeKey::rmwMode, static_cast<int>(AtomicRMWMode::ADD));

    G.SetInCast({"contractInput"});
    G.SetOutCast({"outputDdr", "viewOut"});

    ProcessAtomic passLocal;
    bool hasReduceAccCascade = false;
    EXPECT_EQ(passLocal.EliminateVecDupBranch(*G.GetFunction(), hasReduceAccCascade), SUCCESS);
    EXPECT_FALSE(hasReduceAccCascade);
    EXPECT_EQ(passLocal.EliminateReduceAcc(*G.GetFunction()), SUCCESS);
    EXPECT_EQ(passLocal.EliminateAtomicRMW(*G.GetFunction()), SUCCESS);

    Operation* atomicContract = nullptr;
    Operation* viewContract = nullptr;
    for (auto& op : G.GetFunction()->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            if (op.GetOutputOperand(0) == outputDdr)
                atomicContract = &op;
            if (op.GetOutputOperand(0) == inputDdr)
                viewContract = &op;
        }
    }

    ASSERT_NE(atomicContract, nullptr);
    ASSERT_NE(viewContract, nullptr);
    EXPECT_TRUE(atomicContract->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_FALSE(viewContract->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_EQ(atomicContract->GetScopeId(), 7);
    EXPECT_EQ(G.GetOp("viewOp")->GetInputOperand(0), inputDdr);
}

} // namespace tile_fwk
} // namespace npu
