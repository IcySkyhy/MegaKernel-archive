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
 * \file test_inplace_process.cpp
 * \brief Unit test for InplaceProcess pass.
 */

#include <vector>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_constraint/inplace_process.h"
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

class InplaceProcessTest : public testing::Test {
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
    bool IsInplace(const Operation& op)
    {
        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        return input->tensor->GetRawMagic() == output->tensor->GetRawMagic();
    }

    void CheckInplace(Function& function)
    {
        for (auto& op : function.Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                EXPECT_EQ(IsInplace(op), true)
                    << op.GetOpcodeStr() << " " << op.GetOpMagic() << " should be processed.";
            } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                EXPECT_EQ(IsInplace(op), true)
                    << op.GetOpcodeStr() << " " << op.GetOpMagic() << " should be processed.";
            } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
                EXPECT_EQ(IsInplace(op), true)
                    << op.GetOpcodeStr() << " " << op.GetOpMagic() << " should be processed.";
            }
        }
    }
};

TEST_F(InplaceProcessTest, CopyInDirectAssemble)
{
    int NUM_16 = 16;
    int NUM_32 = 32;
    std::vector<int64_t> shape0{NUM_16, NUM_16};
    std::vector<int64_t> shape1{NUM_32, NUM_32};
    std::vector<int64_t> shape2{NUM_32, NUM_16};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape1, "c");
    auto c = G.GetTensor("c");
    c->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape2, "out");
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // a [16, 16] --> Copy_IN (0, 0) --> [16, 16]
    G.AddTensor(DataType::DT_FP32, shape0, "a_ub");
    auto a_ub = G.GetTensor("a_ub");
    a_ub->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_COPY_IN, {"a"}, {"a_ub"}, "Copy_In_a");
    auto copyInA = G.GetOp("Copy_In_a");
    std::vector<int64_t> offsetA = {0, 0};
    auto attrCopyInA = std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offsetA), MemoryType::MEM_UB,
                                                         OpImmediate::Specified(a_ub->GetShape()),
                                                         OpImmediate::Specified(a_ub->tensor->GetRawShape()));
    copyInA->SetOpAttribute(attrCopyInA);

    // b [16, 16] --> Copy_IN (0, 0) --> [16, 16]
    G.AddTensor(DataType::DT_FP32, shape0, "b_ub");
    auto b_ub = G.GetTensor("b_ub");
    b_ub->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_COPY_IN, {"b"}, {"b_ub"}, "Copy_In_b");
    auto copyInB = G.GetOp("Copy_In_b");
    std::vector<int64_t> offsetB = {0, 0};
    auto attrCopyInB = std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offsetB), MemoryType::MEM_UB,
                                                         OpImmediate::Specified(b_ub->GetShape()),
                                                         OpImmediate::Specified(b_ub->tensor->GetRawShape()));
    copyInB->SetOpAttribute(attrCopyInB);

    // a[16, 16] + b[16, 16] --> add_out[16, 16]
    G.AddTensor(DataType::DT_FP32, shape0, "add_out");
    G.AddOp(Opcode::OP_ADD, {"a_ub", "b_ub"}, {"add_out"}, "Add");
    auto addOut = G.GetTensor("add_out");
    addOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // c[32, 32] --> Copy_IN (16, 0) --> c1[16, 16]
    G.AddTensor(DataType::DT_FP32, shape0, "c1_ub");
    auto c1_ub = G.GetTensor("c1_ub");
    c1_ub->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_COPY_IN, {"c"}, {"c1_ub"}, "Copy_In_C");
    auto copyInC = G.GetOp("Copy_In_C");
    std::vector<int64_t> offsetC = {0, NUM_16};
    auto attrCopyInC = std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offsetC), MemoryType::MEM_UB,
                                                         OpImmediate::Specified(c1_ub->GetShape()),
                                                         OpImmediate::Specified(c1_ub->tensor->GetRawShape()));
    copyInC->SetOpAttribute(attrCopyInC);

    // c1[16, 16]  --> Assemble(16, 0) --> [32, 16]
    G.AddTensor(DataType::DT_FP32, shape2, "assembleOut");
    auto assembleOut = G.GetTensor("assembleOut");
    assembleOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"c1_ub"}, {"assembleOut"}, "Assemble_1");
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int64_t>{16, 0});
    auto assemble1 = G.GetOp("Assemble_1");
    assemble1->SetOpAttribute(attrAssemble1);

    // add_out[16, 16]  --> Assemble(0, 0) --> [32, 16]
    G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"assembleOut"}, "Assemble_2");
    auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int64_t>{0, 0});
    auto assemble2 = G.GetOp("Assemble_2");
    assemble2->SetOpAttribute(attrAssemble2);

    // [32, 16] --> Exp --> [32, 16] --> Copy Out
    G.AddTensor(DataType::DT_FP32, shape2, "exp_out");
    G.AddOp(Opcode::OP_EXP, {"assembleOut"}, {"exp_out"}, "Exp");
    auto expOut = G.GetTensor("exp_out");
    expOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_COPY_OUT, {"exp_out"}, {"out"}, "Copy_Out");
    auto copyOut = G.GetOp("Copy_Out");
    std::vector<int64_t> offsetOut = {0, 0};
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB, OpImmediate::Specified(offsetOut),
                                                         OpImmediate::Specified(expOut->GetShape()),
                                                         OpImmediate::Specified(expOut->tensor->GetRawShape()));
    copyOut->SetOpAttribute(attrCopyOut);

    G.SetInCast({"a", "b", "c"});
    G.SetOutCast({"out"});
    Function* function = G.GetFunction();

    // 确认构图完毕
    constexpr int opNumBefore = 8;
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::InplaceProcess inplaceProcess;
    inplaceProcess.PreCheck(*function);
    inplaceProcess.RunOnFunction(*function);
    inplaceProcess.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    CheckInplace(*function);
}

TEST_F(InplaceProcessTest, InplaceProcessViewOnL1)
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
    G.AddTensor(inputAstDtype, {64, 64}, "l1_a_0");
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_b_0");
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 64}, "l1_a_1");
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_b_1");
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 64}, "l0_a_0");
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_b_0");
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_0");
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddTensor(inputAstDtype, {64, 64}, "l0_a_1");
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_b_1");
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_1");
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_VIEW, {"l1_a"}, {"l1_a_0"}, "A_OP_VIEW_0");
    auto a_op_view_0 = G.GetOp("A_OP_VIEW_0");
    std::vector<int64_t> offestAOpView0 = {0, 0};
    auto attrAOpView0 = std::make_shared<ViewOpAttribute>(offestAOpView0, MemoryType::MEM_L1);
    a_op_view_0->SetOpAttribute(attrAOpView0);
    G.AddOp(Opcode::OP_VIEW, {"l1_a"}, {"l1_a_1"}, "A_OP_VIEW_1");
    auto a_op_view_1 = G.GetOp("A_OP_VIEW_1");
    std::vector<int64_t> offestAOpView1 = {0, 64};
    auto attrAOpView1 = std::make_shared<ViewOpAttribute>(offestAOpView1, MemoryType::MEM_L1);
    a_op_view_1->SetOpAttribute(attrAOpView1);
    G.AddOp(Opcode::OP_VIEW, {"l1_b"}, {"l1_b_0"}, "B_OP_VIEW_0");
    auto b_op_view_0 = G.GetOp("B_OP_VIEW_0");
    std::vector<int64_t> offestBOpView0 = {0, 0};
    auto attrBOpView0 = std::make_shared<ViewOpAttribute>(offestBOpView0, MemoryType::MEM_L1);
    b_op_view_0->SetOpAttribute(attrBOpView0);
    G.AddOp(Opcode::OP_VIEW, {"l1_b"}, {"l1_b_1"}, "B_OP_VIEW_1");
    auto b_op_view_1 = G.GetOp("B_OP_VIEW_1");
    std::vector<int64_t> offestBOpView1 = {64, 0};
    auto attrBOpView1 = std::make_shared<ViewOpAttribute>(offestBOpView1, MemoryType::MEM_L1);
    b_op_view_1->SetOpAttribute(attrBOpView1);
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_1"}, "L0C_Copy_out_1");
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_0", "mat_c_1"});
    // check before pass
    auto l1ArawshapeBefore = l1_a->GetRawTensor()->GetRawShape();
    auto l1A0rawshapeBefore = l1_a_0->GetRawTensor()->GetRawShape();
    auto l1A1rawshapeBefore = l1_a_1->GetRawTensor()->GetRawShape();
    auto l1BrawshapeBefore = l1_b->GetRawTensor()->GetRawShape();
    auto l1B0rawshapeBefore = l1_b_0->GetRawTensor()->GetRawShape();
    auto l1B1rawshapeBefore = l1_b_1->GetRawTensor()->GetRawShape();
    EXPECT_NE(l1ArawshapeBefore, l1A0rawshapeBefore);
    EXPECT_NE(l1ArawshapeBefore, l1A1rawshapeBefore);
    EXPECT_NE(l1BrawshapeBefore, l1B0rawshapeBefore);
    EXPECT_NE(l1BrawshapeBefore, l1B1rawshapeBefore);
    auto l1ARawMagicBefore = l1_a->GetRawMagic();
    auto l1A0RawMagicBefore = l1_a_0->GetRawMagic();
    auto l1A1RawMagicBefore = l1_a_1->GetRawMagic();
    auto l1BRawMagicBefore = l1_b->GetRawMagic();
    auto l1B0RawMagicBefore = l1_b_0->GetRawMagic();
    auto l1B1RawMagicBefore = l1_b_1->GetRawMagic();
    EXPECT_NE(l1ARawMagicBefore, l1A0RawMagicBefore);
    EXPECT_NE(l1ARawMagicBefore, l1A1RawMagicBefore);
    EXPECT_NE(l1BRawMagicBefore, l1B0RawMagicBefore);
    EXPECT_NE(l1BRawMagicBefore, l1B1RawMagicBefore);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    InplaceProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    auto l1ARawshapeAfter = l1_a->GetRawTensor()->GetRawShape();
    auto l1A0RawshapeAfter = l1_a_0->GetRawTensor()->GetRawShape();
    auto l1A1RawshapeAfter = l1_a_1->GetRawTensor()->GetRawShape();
    auto l1BRawshapeAfter = l1_b->GetRawTensor()->GetRawShape();
    auto l1B0RawshapeAfter = l1_b_0->GetRawTensor()->GetRawShape();
    auto l1B1RawshapeAfter = l1_b_1->GetRawTensor()->GetRawShape();
    EXPECT_EQ(l1ARawshapeAfter, l1A0RawshapeAfter);
    EXPECT_EQ(l1ARawshapeAfter, l1A1RawshapeAfter);
    EXPECT_EQ(l1BRawshapeAfter, l1B0RawshapeAfter);
    EXPECT_EQ(l1BRawshapeAfter, l1B1RawshapeAfter);
    EXPECT_EQ(l1ArawshapeBefore, l1ARawshapeAfter);
    EXPECT_EQ(l1BrawshapeBefore, l1BRawshapeAfter);
    auto l1A0OffestAfter = l1_a_0->GetOffset();
    auto l1A1OffestAfter = l1_a_1->GetOffset();
    auto l1B0OffestAfter = l1_b_0->GetOffset();
    auto l1B1OffestAfter = l1_b_1->GetOffset();
    EXPECT_EQ(offestAOpView0, l1A0OffestAfter);
    EXPECT_EQ(offestAOpView1, l1A1OffestAfter);
    EXPECT_EQ(offestBOpView0, l1B0OffestAfter);
    EXPECT_EQ(offestBOpView1, l1B1OffestAfter);
    auto l1ARawMagicAfter = l1_a->GetRawMagic();
    auto l1A0RawMagicAfter = l1_a_0->GetRawMagic();
    auto l1A1RawMagicAfter = l1_a_1->GetRawMagic();
    auto l1BRawMagicAfter = l1_b->GetRawMagic();
    auto l1B0RawMagicAfter = l1_b_0->GetRawMagic();
    auto l1B1RawMagicAfter = l1_b_1->GetRawMagic();
    EXPECT_EQ(l1ARawMagicAfter, l1A0RawMagicAfter);
    EXPECT_EQ(l1ARawMagicAfter, l1A1RawMagicAfter);
    EXPECT_EQ(l1BRawMagicAfter, l1B0RawMagicAfter);
    EXPECT_EQ(l1BRawMagicAfter, l1B1RawMagicAfter);
    EXPECT_EQ(l1ARawMagicBefore, l1ARawMagicAfter);
    EXPECT_EQ(l1BRawMagicBefore, l1BRawMagicAfter);
}

TEST_F(InplaceProcessTest, InplaceProcessAssembleOnGm)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 64}, "copy_in_0");
    auto copy_in_0 = G.GetTensor("copy_in_0");
    copy_in_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 64}, "copy_in_1");
    auto copy_in_1 = G.GetTensor("copy_in_1");
    copy_in_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 64}, "vec_in_0");
    auto vec_in_0 = G.GetTensor("vec_in_0");
    vec_in_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 64}, "vec_in_1");
    auto vec_in_1 = G.GetTensor("vec_in_1");
    vec_in_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    vec_out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"copy_in_0"}, {"vec_in_0"}, "COPYIN_0");
    G.AddOp(Opcode::OP_COPY_IN, {"copy_in_1"}, {"vec_in_1"}, "COPYIN_1");
    G.AddOp(Opcode::OP_ASSEMBLE, {"vec_in_0"}, {"vec_out"}, "ASSEMBLE_0");
    auto assemble0 = G.GetOp("ASSEMBLE_0");
    std::vector<int64_t> offestAssemble0 = {0, 0};
    auto attrAssemble0 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offestAssemble0);
    assemble0->SetOpAttribute(attrAssemble0);
    G.AddOp(Opcode::OP_ASSEMBLE, {"vec_in_1"}, {"vec_out"}, "ASSEMBLE_1");
    auto assemble1 = G.GetOp("ASSEMBLE_1");
    std::vector<int64_t> offestAssemble1 = {0, 64};
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offestAssemble1);
    assemble1->SetOpAttribute(attrAssemble1);
    // set incast and outcast
    G.SetInCast({"copy_in_0", "copy_in_1"});
    G.SetOutCast({"vec_out"});
    // check before pass
    auto vecIn0RawshapeBefore = vec_in_0->GetRawTensor()->GetRawShape();
    auto vecIn1RawshapeBefore = vec_in_1->GetRawTensor()->GetRawShape();
    auto vecOutRawshapeBefore = vec_out->GetRawTensor()->GetRawShape();
    EXPECT_NE(vecIn0RawshapeBefore, vecOutRawshapeBefore);
    EXPECT_NE(vecIn1RawshapeBefore, vecOutRawshapeBefore);
    auto vecIn0RawMagicBefore = vec_in_0->GetRawMagic();
    auto vecIn1RawMagicBefore = vec_in_1->GetRawMagic();
    auto vecOutRawMagicBefore = vec_out->GetRawMagic();
    EXPECT_NE(vecIn0RawMagicBefore, vecOutRawMagicBefore);
    EXPECT_NE(vecIn1RawMagicBefore, vecOutRawMagicBefore);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    InplaceProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    auto vecIn0RawshapeAfter = vec_in_0->GetRawTensor()->GetRawShape();
    auto vecIn1RawshapeAfter = vec_in_1->GetRawTensor()->GetRawShape();
    auto vecOutRawshapeAfter = vec_out->GetRawTensor()->GetRawShape();
    EXPECT_EQ(vecIn0RawshapeAfter, vecOutRawshapeAfter);
    EXPECT_EQ(vecIn1RawshapeAfter, vecOutRawshapeAfter);
    EXPECT_EQ(vecOutRawshapeBefore, vecOutRawshapeAfter);
    auto vecIn0OffestAfter = vec_in_0->GetOffset();
    auto vecIn1OffestAfter = vec_in_1->GetOffset();
    EXPECT_EQ(offestAssemble0, vecIn0OffestAfter);
    EXPECT_EQ(offestAssemble1, vecIn1OffestAfter);
    auto vecIn0RawMagicAfter = vec_in_0->GetRawMagic();
    auto vecIn1RawMagicAfter = vec_in_1->GetRawMagic();
    auto vecOutRawMagicAfter = vec_out->GetRawMagic();
    EXPECT_EQ(vecIn0RawMagicAfter, vecOutRawMagicAfter);
    EXPECT_EQ(vecIn1RawMagicAfter, vecOutRawMagicAfter);
}

TEST_F(InplaceProcessTest, InplaceProcessAssembleOnUb)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 64}, "vec_in_0");
    auto vec_in_0 = G.GetTensor("vec_in_0");
    vec_in_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 64}, "vec_in_1");
    auto vec_in_1 = G.GetTensor("vec_in_1");
    vec_in_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    vec_out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 64}, "ub_in_0");
    auto ub_in_0 = G.GetTensor("ub_in_0");
    ub_in_0->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddTensor(inputAstDtype, {64, 64}, "ub_in_1");
    auto ub_in_1 = G.GetTensor("ub_in_1");
    ub_in_1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddTensor(outputAstDtype, {64, 128}, "ub_out");
    auto ub_out = G.GetTensor("ub_out");
    ub_out->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    // add op
    G.AddOp(Opcode::OP_UB_COPY_IN, {"vec_in_0"}, {"ub_in_0"}, "UB_COPY_IN");
    G.AddOp(Opcode::OP_UB_COPY_IN, {"vec_in_1"}, {"ub_in_1"}, "UB_COPY_IN");
    G.AddOp(Opcode::OP_ASSEMBLE, {"ub_in_0"}, {"ub_out"}, "ASSEMBLE_0");
    auto assemble0 = G.GetOp("ASSEMBLE_0");
    std::vector<int64_t> offestAssemble0 = {0, 0};
    auto attrAssemble0 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offestAssemble0);
    assemble0->SetOpAttribute(attrAssemble0);
    G.AddOp(Opcode::OP_ASSEMBLE, {"ub_in_1"}, {"ub_out"}, "ASSEMBLE_1");
    auto assemble1 = G.GetOp("ASSEMBLE_1");
    std::vector<int64_t> offestAssemble1 = {0, 64};
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offestAssemble1);
    assemble1->SetOpAttribute(attrAssemble1);
    G.AddOp(Opcode::OP_UB_COPY_OUT, {"ub_out"}, {"vec_out"}, "UB_COPY_OUT");
    // set incast and outcast
    G.SetInCast({"vec_in_0", "vec_in_1"});
    G.SetOutCast({"vec_out"});
    // check before pass
    auto ubIn0RawshapeBefore = ub_in_0->GetRawTensor()->GetRawShape();
    auto ubIn1RawshapeBefore = ub_in_1->GetRawTensor()->GetRawShape();
    auto ubOutRawshapeBefore = ub_out->GetRawTensor()->GetRawShape();
    EXPECT_NE(ubIn0RawshapeBefore, ubOutRawshapeBefore);
    EXPECT_NE(ubIn1RawshapeBefore, ubOutRawshapeBefore);
    auto ubIn0RawMagicBefore = ub_in_0->GetRawMagic();
    auto ubIn1RawMagicBefore = ub_in_1->GetRawMagic();
    auto ubOutRawMagicBefore = ub_out->GetRawMagic();
    EXPECT_NE(ubIn0RawMagicBefore, ubOutRawMagicBefore);
    EXPECT_NE(ubIn1RawMagicBefore, ubOutRawMagicBefore);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    InplaceProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    auto ubIn0RawshapeAfter = ub_in_0->GetRawTensor()->GetRawShape();
    auto ubIn1RawshapeAfter = ub_in_1->GetRawTensor()->GetRawShape();
    auto ubOutRawshapeAfter = ub_out->GetRawTensor()->GetRawShape();
    EXPECT_EQ(ubIn0RawshapeAfter, ubOutRawshapeAfter);
    EXPECT_EQ(ubIn1RawshapeAfter, ubOutRawshapeAfter);
    EXPECT_EQ(ubOutRawshapeBefore, ubOutRawshapeAfter);
    auto ubIn0OffestAfter = ub_in_0->GetOffset();
    auto ubIn1OffestAfter = ub_in_1->GetOffset();
    EXPECT_EQ(offestAssemble0, ubIn0OffestAfter);
    EXPECT_EQ(offestAssemble1, ubIn1OffestAfter);
    auto ubIn0RawMagicAfter = ub_in_0->GetRawMagic();
    auto ubIn1RawMagicAfter = ub_in_1->GetRawMagic();
    auto ubOutRawMagicAfter = ub_out->GetRawMagic();
    EXPECT_EQ(ubIn0RawMagicAfter, ubOutRawMagicAfter);
    EXPECT_EQ(ubIn1RawMagicAfter, ubOutRawMagicAfter);
    EXPECT_EQ(ubOutRawMagicBefore, ubOutRawMagicAfter);
}

TEST_F(InplaceProcessTest, InplaceProcessReShapeOnGm)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 8, 16}, "vec_in");
    auto vec_in = G.GetTensor("vec_in");
    vec_in->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    vec_out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "vec_out_rel");
    auto vec_out_rel = G.GetTensor("vec_out_rel");
    vec_out_rel->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    // add op
    G.AddOp(Opcode::OP_RESHAPE, {"vec_in"}, {"vec_out"}, "RESHAPE");
    G.AddOp(Opcode::OP_VIEW, {"vec_out"}, {"vec_out_rel"}, "VIEW");
    // set incast and outcast
    G.SetInCast({"vec_in"});
    G.SetOutCast({"vec_out_rel"});
    // check before pass
    auto inRawMagicBefore = vec_in->GetRawMagic();
    auto outRawMagicBefore = vec_out->GetRawMagic();
    EXPECT_NE(inRawMagicBefore, outRawMagicBefore);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    InplaceProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    auto inRawMagicAfter = vec_in->GetRawMagic();
    auto outRawMagicAfter = vec_out->GetRawMagic();
    EXPECT_EQ(inRawMagicAfter, outRawMagicAfter);
    EXPECT_EQ(inRawMagicBefore, inRawMagicAfter);
}

TEST_F(InplaceProcessTest, InplaceProcessReShapeOnUb)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 8, 16}, "vec_in");
    auto vec_in = G.GetTensor("vec_in");
    vec_in->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    vec_out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 8, 16}, "ub_in");
    auto ub_in = G.GetTensor("ub_in");
    ub_in->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddTensor(outputAstDtype, {64, 128}, "ub_out");
    auto ub_out = G.GetTensor("ub_out");
    ub_out->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    // add op
    G.AddOp(Opcode::OP_UB_COPY_IN, {"vec_in"}, {"ub_in"}, "UB_COPY_IN");
    G.AddOp(Opcode::OP_RESHAPE, {"ub_in"}, {"ub_out"}, "RESHAPE");
    G.AddOp(Opcode::OP_UB_COPY_OUT, {"ub_out"}, {"vec_out"}, "UB_COPY_OUT");
    // set incast and outcast
    G.SetInCast({"vec_in"});
    G.SetOutCast({"vec_out"});
    // check before pass
    auto inRawMagicBefore = ub_in->GetRawMagic();
    auto outRawMagicBefore = ub_out->GetRawMagic();
    EXPECT_NE(inRawMagicBefore, outRawMagicBefore);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    InplaceProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    auto inRawMagicAfter = ub_in->GetRawMagic();
    auto outRawMagicAfter = ub_out->GetRawMagic();
    EXPECT_EQ(inRawMagicAfter, outRawMagicAfter);
    EXPECT_EQ(inRawMagicBefore, inRawMagicAfter);
}

TEST_F(InplaceProcessTest, InplaceProcessViewReshape)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 4, 32}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {4, 32, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_0");
    auto mat_c_0 = G.GetTensor("mat_c_0");
    mat_c_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_1");
    auto mat_c_1 = G.GetTensor("mat_c_1");
    mat_c_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 4, 32}, "l1_a");
    auto l1_a = G.GetTensor("l1_a");
    l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {4, 32, 128}, "l1_b");
    auto l1_b = G.GetTensor("l1_b");
    l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 4, 16}, "l1_a_0");
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {4, 16, 128}, "l1_b_0");
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 4, 16}, "l1_a_1");
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {4, 16, 128}, "l1_b_1");
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 64}, "l1_a_2");
    auto l1_a_2 = G.GetTensor("l1_a_2");
    l1_a_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_b_2");
    auto l1_b_2 = G.GetTensor("l1_b_2");
    l1_b_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 64}, "l1_a_3");
    auto l1_a_3 = G.GetTensor("l1_a_3");
    l1_a_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_b_3");
    auto l1_b_3 = G.GetTensor("l1_b_3");
    l1_b_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 64}, "l0_a_0");
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_b_0");
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_0");
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddTensor(inputAstDtype, {64, 64}, "l0_a_1");
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_b_1");
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_1");
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_VIEW, {"l1_a"}, {"l1_a_0"}, "A_OP_VIEW_0");
    auto a_op_view_0 = G.GetOp("A_OP_VIEW_0");
    std::vector<int64_t> offestAOpView0 = {0, 0, 0};
    auto attrAOpView0 = std::make_shared<ViewOpAttribute>(offestAOpView0, MemoryType::MEM_L1);
    a_op_view_0->SetOpAttribute(attrAOpView0);
    G.AddOp(Opcode::OP_VIEW, {"l1_a"}, {"l1_a_1"}, "A_OP_VIEW_1");
    auto a_op_view_1 = G.GetOp("A_OP_VIEW_1");
    std::vector<int64_t> offestAOpView1 = {0, 0, 16};
    auto attrAOpView1 = std::make_shared<ViewOpAttribute>(offestAOpView1, MemoryType::MEM_L1);
    a_op_view_1->SetOpAttribute(attrAOpView1);
    G.AddOp(Opcode::OP_VIEW, {"l1_b"}, {"l1_b_0"}, "B_OP_VIEW_0");
    auto b_op_view_0 = G.GetOp("B_OP_VIEW_0");
    std::vector<int64_t> offestBOpView0 = {0, 0, 0};
    auto attrBOpView0 = std::make_shared<ViewOpAttribute>(offestBOpView0, MemoryType::MEM_L1);
    b_op_view_0->SetOpAttribute(attrBOpView0);
    G.AddOp(Opcode::OP_VIEW, {"l1_b"}, {"l1_b_1"}, "B_OP_VIEW_1");
    auto b_op_view_1 = G.GetOp("B_OP_VIEW_1");
    std::vector<int64_t> offestBOpView1 = {0, 16, 0};
    auto attrBOpView1 = std::make_shared<ViewOpAttribute>(offestBOpView1, MemoryType::MEM_L1);
    b_op_view_1->SetOpAttribute(attrBOpView1);
    G.AddOp(Opcode::OP_RESHAPE, {"l1_a_0"}, {"l1_a_2"}, "RESHAPE_0");
    G.AddOp(Opcode::OP_RESHAPE, {"l1_a_1"}, {"l1_a_3"}, "RESHAPE_1");
    G.AddOp(Opcode::OP_RESHAPE, {"l1_b_0"}, {"l1_b_2"}, "RESHAPE_2");
    G.AddOp(Opcode::OP_RESHAPE, {"l1_b_1"}, {"l1_b_3"}, "RESHAPE_3");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_2"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_3"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_2"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_3"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_1"}, "L0C_Copy_out_1");
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_0", "mat_c_1"});
    // view check before pass
    auto l1ARawshapeBefore = l1_a->GetRawTensor()->GetRawShape();
    auto l1A0RawshapeBefore = l1_a_0->GetRawTensor()->GetRawShape();
    auto l1A1RawshapeBefore = l1_a_1->GetRawTensor()->GetRawShape();
    auto l1BRawshapeBefore = l1_b->GetRawTensor()->GetRawShape();
    auto l1B0RawshapeBefore = l1_b_0->GetRawTensor()->GetRawShape();
    auto l1B1RawshapeBefore = l1_b_1->GetRawTensor()->GetRawShape();
    EXPECT_NE(l1ARawshapeBefore, l1A0RawshapeBefore);
    EXPECT_NE(l1ARawshapeBefore, l1A1RawshapeBefore);
    EXPECT_NE(l1BRawshapeBefore, l1B0RawshapeBefore);
    EXPECT_NE(l1BRawshapeBefore, l1B1RawshapeBefore);
    auto l1ARawMagicBefore = l1_a->GetRawMagic();
    auto l1A0RawMagicBefore = l1_a_0->GetRawMagic();
    auto l1A1RawMagicBefore = l1_a_1->GetRawMagic();
    auto l1BRawMagicBefore = l1_b->GetRawMagic();
    auto l1B0RawMagicBefore = l1_b_0->GetRawMagic();
    auto l1B1RawMagicBefore = l1_b_1->GetRawMagic();
    EXPECT_NE(l1ARawMagicBefore, l1A0RawMagicBefore);
    EXPECT_NE(l1ARawMagicBefore, l1A1RawMagicBefore);
    EXPECT_NE(l1BRawMagicBefore, l1B0RawMagicBefore);
    EXPECT_NE(l1BRawMagicBefore, l1B1RawMagicBefore);
    // reshape check before pass
    auto l1A2RawMagicBefore = l1_a_2->GetRawMagic();
    auto l1A3RawMagicBefore = l1_a_3->GetRawMagic();
    auto l1B2RawMagicBefore = l1_b_2->GetRawMagic();
    auto l1B3RawMagicBefore = l1_b_3->GetRawMagic();
    EXPECT_NE(l1A0RawMagicBefore, l1A2RawMagicBefore);
    EXPECT_NE(l1A1RawMagicBefore, l1A3RawMagicBefore);
    EXPECT_NE(l1B0RawMagicBefore, l1B2RawMagicBefore);
    EXPECT_NE(l1B1RawMagicBefore, l1B3RawMagicBefore);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    InplaceProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // view check after pass
    auto l1ARawshapeAfter = l1_a->GetRawTensor()->GetRawShape();
    auto l1A0RawshapeAfter = l1_a_0->GetRawTensor()->GetRawShape();
    auto l1A1RawshapeAfter = l1_a_1->GetRawTensor()->GetRawShape();
    auto l1BRawshapeAfter = l1_b->GetRawTensor()->GetRawShape();
    auto l1B0RawshapeAfter = l1_b_0->GetRawTensor()->GetRawShape();
    auto l1B1RawshapeAfter = l1_b_1->GetRawTensor()->GetRawShape();
    EXPECT_EQ(l1ARawshapeAfter, l1A0RawshapeAfter);
    EXPECT_EQ(l1ARawshapeAfter, l1A1RawshapeAfter);
    EXPECT_EQ(l1BRawshapeAfter, l1B0RawshapeAfter);
    EXPECT_EQ(l1BRawshapeAfter, l1B1RawshapeAfter);
    EXPECT_EQ(l1ARawshapeBefore, l1ARawshapeAfter);
    EXPECT_EQ(l1BRawshapeBefore, l1BRawshapeAfter);
    auto l1A0OffestAfter = l1_a_0->GetOffset();
    auto l1A1OffestAfter = l1_a_1->GetOffset();
    auto l1B0OffestAfter = l1_b_0->GetOffset();
    auto l1B1OffestAfter = l1_b_1->GetOffset();
    EXPECT_EQ(offestAOpView0, l1A0OffestAfter);
    EXPECT_EQ(offestAOpView1, l1A1OffestAfter);
    EXPECT_EQ(offestBOpView0, l1B0OffestAfter);
    EXPECT_EQ(offestBOpView1, l1B1OffestAfter);
    auto l1ARawMagicAfter = l1_a->GetRawMagic();
    auto l1A0RawMagicAfter = l1_a_0->GetRawMagic();
    auto l1A1RawMagicAfter = l1_a_1->GetRawMagic();
    auto l1BRawMagicAfter = l1_b->GetRawMagic();
    auto l1B0RawMagicAfter = l1_b_0->GetRawMagic();
    auto l1B1RawMagicAfter = l1_b_1->GetRawMagic();
    EXPECT_EQ(l1ARawMagicAfter, l1A0RawMagicAfter);
    EXPECT_EQ(l1ARawMagicAfter, l1A1RawMagicAfter);
    EXPECT_EQ(l1BRawMagicAfter, l1B0RawMagicAfter);
    EXPECT_EQ(l1BRawMagicAfter, l1B1RawMagicAfter);
    EXPECT_EQ(l1ARawMagicBefore, l1ARawMagicAfter);
    EXPECT_EQ(l1BRawMagicBefore, l1BRawMagicAfter);
    // reshape check after pass
    auto l1A2RawMagicAfter = l1_a_2->GetRawMagic();
    auto l1A3RawMagicAfter = l1_a_3->GetRawMagic();
    auto l1B2RawMagicAfter = l1_b_2->GetRawMagic();
    auto l1B3RawMagicAfter = l1_b_3->GetRawMagic();
    EXPECT_EQ(l1A0RawMagicAfter, l1A2RawMagicAfter);
    EXPECT_EQ(l1A1RawMagicAfter, l1A3RawMagicAfter);
    EXPECT_EQ(l1B0RawMagicAfter, l1B2RawMagicAfter);
    EXPECT_EQ(l1B1RawMagicAfter, l1B3RawMagicAfter);
}

TEST_F(InplaceProcessTest, InplaceProcessReshapeView)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 4, 32}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {4, 32, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_0");
    auto mat_c_0 = G.GetTensor("mat_c_0");
    mat_c_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_1");
    auto mat_c_1 = G.GetTensor("mat_c_1");
    mat_c_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 4, 32}, "l1_a");
    auto l1_a = G.GetTensor("l1_a");
    l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {4, 32, 128}, "l1_b");
    auto l1_b = G.GetTensor("l1_b");
    l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a_0");
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b_0");
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 64}, "l1_a_1");
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_b_1");
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 64}, "l1_a_2");
    auto l1_a_2 = G.GetTensor("l1_a_2");
    l1_a_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_b_2");
    auto l1_b_2 = G.GetTensor("l1_b_2");
    l1_b_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 64}, "l0_a_0");
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_b_0");
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_0");
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddTensor(inputAstDtype, {64, 64}, "l0_a_1");
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_b_1");
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_1");
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_RESHAPE, {"l1_a"}, {"l1_a_0"}, "RESHAPE_0");
    G.AddOp(Opcode::OP_RESHAPE, {"l1_b"}, {"l1_b_0"}, "RESHAPE_1");
    G.AddOp(Opcode::OP_VIEW, {"l1_a_0"}, {"l1_a_1"}, "A_OP_VIEW_0");
    auto a_op_view_0 = G.GetOp("A_OP_VIEW_0");
    std::vector<int64_t> offestAOpView0 = {0, 0};
    auto attrAOpView0 = std::make_shared<ViewOpAttribute>(offestAOpView0, MemoryType::MEM_L1);
    a_op_view_0->SetOpAttribute(attrAOpView0);
    G.AddOp(Opcode::OP_VIEW, {"l1_a_0"}, {"l1_a_2"}, "A_OP_VIEW_1");
    auto a_op_view_1 = G.GetOp("A_OP_VIEW_1");
    std::vector<int64_t> offestAOpView1 = {0, 64};
    auto attrAOpView1 = std::make_shared<ViewOpAttribute>(offestAOpView1, MemoryType::MEM_L1);
    a_op_view_1->SetOpAttribute(attrAOpView1);
    G.AddOp(Opcode::OP_VIEW, {"l1_b_0"}, {"l1_b_1"}, "B_OP_VIEW_0");
    auto b_op_view_0 = G.GetOp("B_OP_VIEW_0");
    std::vector<int64_t> offestBOpView0 = {0, 0};
    auto attrBOpView0 = std::make_shared<ViewOpAttribute>(offestBOpView0, MemoryType::MEM_L1);
    b_op_view_0->SetOpAttribute(attrBOpView0);
    G.AddOp(Opcode::OP_VIEW, {"l1_b_0"}, {"l1_b_2"}, "B_OP_VIEW_1");
    auto b_op_view_1 = G.GetOp("B_OP_VIEW_1");
    std::vector<int64_t> offestBOpView1 = {64, 0};
    auto attrBOpView1 = std::make_shared<ViewOpAttribute>(offestBOpView1, MemoryType::MEM_L1);
    b_op_view_1->SetOpAttribute(attrBOpView1);
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_1"}, "L0C_Copy_out_1");
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_0", "mat_c_1"});
    // reshape check before pass
    auto l1ARawMagicBefore = l1_a->GetRawMagic();
    auto l1A0RawMagicBefore = l1_a_0->GetRawMagic();
    EXPECT_NE(l1ARawMagicBefore, l1A0RawMagicBefore);
    auto l1BRawMagicBefore = l1_b->GetRawMagic();
    auto l1B0RawMagicBefore = l1_b_0->GetRawMagic();
    EXPECT_NE(l1BRawMagicBefore, l1B0RawMagicBefore);
    // view check before pass
    auto l1A0RawshapeBefore = l1_a_0->GetRawTensor()->GetRawShape();
    auto l1A1RawshapeBefore = l1_a_1->GetRawTensor()->GetRawShape();
    auto l1A2RawshapeBefore = l1_a_2->GetRawTensor()->GetRawShape();
    auto l1B0RawshapeBefore = l1_b_0->GetRawTensor()->GetRawShape();
    auto l1B1RawshapeBefore = l1_b_1->GetRawTensor()->GetRawShape();
    auto l1B2RawshapeBefore = l1_b_2->GetRawTensor()->GetRawShape();
    EXPECT_NE(l1A0RawshapeBefore, l1A1RawshapeBefore);
    EXPECT_NE(l1A0RawshapeBefore, l1A2RawshapeBefore);
    EXPECT_NE(l1B0RawshapeBefore, l1B1RawshapeBefore);
    EXPECT_NE(l1B0RawshapeBefore, l1B2RawshapeBefore);
    auto l1A1RawMagicBefore = l1_a_1->GetRawMagic();
    auto l1A2RawMagicBefore = l1_a_2->GetRawMagic();
    auto l1B1RawMagicBefore = l1_b_1->GetRawMagic();
    auto l1B2RawMagicBefore = l1_b_2->GetRawMagic();
    EXPECT_NE(l1A0RawMagicBefore, l1A1RawMagicBefore);
    EXPECT_NE(l1A0RawMagicBefore, l1A2RawMagicBefore);
    EXPECT_NE(l1B0RawMagicBefore, l1B1RawMagicBefore);
    EXPECT_NE(l1B0RawMagicBefore, l1B2RawMagicBefore);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    InplaceProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // reshape check after pass
    auto l1ARawMagicAfter = l1_a->GetRawMagic();
    auto l1A0RawMagicAfter = l1_a_0->GetRawMagic();
    EXPECT_EQ(l1ARawMagicAfter, l1A0RawMagicAfter);
    auto l1BRawMagicAfter = l1_b->GetRawMagic();
    auto l1B0RawMagicAfter = l1_b_0->GetRawMagic();
    EXPECT_EQ(l1BRawMagicAfter, l1B0RawMagicAfter);
    EXPECT_EQ(l1ARawMagicBefore, l1ARawMagicAfter);
    EXPECT_EQ(l1BRawMagicBefore, l1BRawMagicAfter);
    // view check after pass
    auto l1A0RawshapeAfter = l1_a_0->GetRawTensor()->GetRawShape();
    auto l1A1RawshapeAfter = l1_a_1->GetRawTensor()->GetRawShape();
    auto l1A2RawshapeAfter = l1_a_2->GetRawTensor()->GetRawShape();
    auto l1B0RawshapeAfter = l1_b_0->GetRawTensor()->GetRawShape();
    auto l1B1RawshapeAfter = l1_b_1->GetRawTensor()->GetRawShape();
    auto l1B2RawshapeAfter = l1_b_2->GetRawTensor()->GetRawShape();
    EXPECT_EQ(l1A0RawshapeAfter, l1A1RawshapeAfter);
    EXPECT_EQ(l1A0RawshapeAfter, l1A2RawshapeAfter);
    EXPECT_EQ(l1B0RawshapeAfter, l1B1RawshapeAfter);
    EXPECT_EQ(l1B0RawshapeAfter, l1B2RawshapeAfter);
    EXPECT_EQ(l1A0RawshapeBefore, l1A0RawshapeAfter);
    EXPECT_EQ(l1B0RawshapeBefore, l1B0RawshapeAfter);
    auto l1A1OffestAfter = l1_a_1->GetOffset();
    auto l1A2OffestAfter = l1_a_2->GetOffset();
    auto l1B1OffestAfter = l1_b_1->GetOffset();
    auto l1B2OffestAfter = l1_b_2->GetOffset();
    EXPECT_EQ(offestAOpView0, l1A1OffestAfter);
    EXPECT_EQ(offestAOpView1, l1A2OffestAfter);
    EXPECT_EQ(offestBOpView0, l1B1OffestAfter);
    EXPECT_EQ(offestBOpView1, l1B2OffestAfter);
    auto l1A1RawMagicAfter = l1_a_1->GetRawMagic();
    auto l1A2RawMagicAfter = l1_a_2->GetRawMagic();
    auto l1B1RawMagicAfter = l1_b_1->GetRawMagic();
    auto l1B2RawMagicAfter = l1_b_2->GetRawMagic();
    EXPECT_EQ(l1A0RawMagicAfter, l1A1RawMagicAfter);
    EXPECT_EQ(l1A0RawMagicAfter, l1A2RawMagicAfter);
    EXPECT_EQ(l1B0RawMagicAfter, l1B1RawMagicAfter);
    EXPECT_EQ(l1B0RawMagicAfter, l1B2RawMagicAfter);
}

TEST_F(InplaceProcessTest, InplaceProcessAssembleReshape)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 64}, "copy_in_0");
    auto copy_in_0 = G.GetTensor("copy_in_0");
    copy_in_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 64}, "copy_in_1");
    auto copy_in_1 = G.GetTensor("copy_in_1");
    copy_in_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 64}, "vec_in_0");
    auto vec_in_0 = G.GetTensor("vec_in_0");
    vec_in_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 64}, "vec_in_1");
    auto vec_in_1 = G.GetTensor("vec_in_1");
    vec_in_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "vec");
    auto vec = G.GetTensor("vec");
    vec->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 8, 16}, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    vec_out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 8, 16}, "vec_out_rel");
    auto vec_out_rel = G.GetTensor("vec_out_rel");
    vec_out_rel->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"copy_in_0"}, {"vec_in_0"}, "COPYIN_0");
    G.AddOp(Opcode::OP_COPY_IN, {"copy_in_1"}, {"vec_in_1"}, "COPYIN_1");
    G.AddOp(Opcode::OP_ASSEMBLE, {"vec_in_0"}, {"vec"}, "ASSEMBLE_0");
    auto assemble0 = G.GetOp("ASSEMBLE_0");
    std::vector<int64_t> offestAssemble0 = {0, 0};
    auto attrAssemble0 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offestAssemble0);
    assemble0->SetOpAttribute(attrAssemble0);
    G.AddOp(Opcode::OP_ASSEMBLE, {"vec_in_1"}, {"vec"}, "ASSEMBLE_1");
    auto assemble1 = G.GetOp("ASSEMBLE_1");
    std::vector<int64_t> offestAssemble1 = {0, 64};
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offestAssemble1);
    assemble1->SetOpAttribute(attrAssemble1);
    G.AddOp(Opcode::OP_RESHAPE, {"vec"}, {"vec_out"}, "RESHAPE");
    G.AddOp(Opcode::OP_VIEW, {"vec_out"}, {"vec_out_rel"}, "VIEW");
    // set incast and outcast
    G.SetInCast({"copy_in_0", "copy_in_1"});
    G.SetOutCast({"vec_out_rel"});
    // assemble check before pass
    auto vecIn0RawshapeBefore = vec_in_0->GetRawTensor()->GetRawShape();
    auto vecIn1RawshapeBefore = vec_in_1->GetRawTensor()->GetRawShape();
    auto vecRawshapeBefore = vec->GetRawTensor()->GetRawShape();
    EXPECT_NE(vecIn0RawshapeBefore, vecRawshapeBefore);
    EXPECT_NE(vecIn1RawshapeBefore, vecRawshapeBefore);
    auto vecIn0RawMagicBefore = vec_in_0->GetRawMagic();
    auto vecIn1RawMagicBefore = vec_in_1->GetRawMagic();
    auto vecRawMagicBefore = vec->GetRawMagic();
    EXPECT_NE(vecIn0RawMagicBefore, vecRawMagicBefore);
    EXPECT_NE(vecIn1RawMagicBefore, vecRawMagicBefore);
    // reshape check before pass
    auto vecOutRawMagicBefore = vec_out->GetRawMagic();
    EXPECT_NE(vecRawMagicBefore, vecOutRawMagicBefore);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    InplaceProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // assemble check after pass
    auto vecIn0RawshapeAfter = vec_in_0->GetRawTensor()->GetRawShape();
    auto vecIn1RawshapeAfter = vec_in_1->GetRawTensor()->GetRawShape();
    auto vecRawshapeAfter = vec->GetRawTensor()->GetRawShape();
    EXPECT_EQ(vecIn0RawshapeAfter, vecRawshapeAfter);
    EXPECT_EQ(vecIn1RawshapeAfter, vecRawshapeAfter);
    EXPECT_EQ(vecRawshapeBefore, vecRawshapeAfter);
    auto vecIn0OffestAfter = vec_in_0->GetOffset();
    auto vecIn1OffestAfter = vec_in_1->GetOffset();
    EXPECT_EQ(offestAssemble0, vecIn0OffestAfter);
    EXPECT_EQ(offestAssemble1, vecIn1OffestAfter);
    auto vecIn0RawMagicAfter = vec_in_0->GetRawMagic();
    auto vecIn1RawMagicAfter = vec_in_1->GetRawMagic();
    auto vecRawMagicAfter = vec->GetRawMagic();
    EXPECT_EQ(vecIn0RawMagicAfter, vecRawMagicAfter);
    EXPECT_EQ(vecIn1RawMagicAfter, vecRawMagicAfter);
    EXPECT_EQ(vecRawMagicBefore, vecRawMagicAfter);
    // reshape check after pass
    auto vecOutRawMagicAfter = vec_out->GetRawMagic();
    EXPECT_EQ(vecRawMagicAfter, vecOutRawMagicAfter);
}

TEST_F(InplaceProcessTest, InplaceProcessReShapeReshape)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 8, 16}, "vec_in");
    auto vec_in = G.GetTensor("vec_in");
    vec_in->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    vec_out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 8, 16}, "ub_in");
    auto ub_in = G.GetTensor("ub_in");
    ub_in->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddTensor(inputAstDtype, {64, 4, 32}, "ub");
    auto ub = G.GetTensor("ub");
    ub->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddTensor(outputAstDtype, {64, 128}, "ub_out");
    auto ub_out = G.GetTensor("ub_out");
    ub_out->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    // add op
    G.AddOp(Opcode::OP_UB_COPY_IN, {"vec_in"}, {"ub_in"}, "UB_COPY_IN");
    G.AddOp(Opcode::OP_RESHAPE, {"ub_in"}, {"ub"}, "RESHAPE_1");
    G.AddOp(Opcode::OP_RESHAPE, {"ub"}, {"ub_out"}, "RESHAPE_2");
    G.AddOp(Opcode::OP_UB_COPY_OUT, {"ub_out"}, {"vec_out"}, "UB_COPY_OUT");
    // set incast and outcast
    G.SetInCast({"vec_in"});
    G.SetOutCast({"vec_out"});
    // check before pass
    auto inRawMagicBefore = ub_in->GetRawMagic();
    auto outRawMagicBefore = ub_out->GetRawMagic();
    EXPECT_NE(inRawMagicBefore, outRawMagicBefore);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    InplaceProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    auto inRawMagicAfter = ub_in->GetRawMagic();
    auto outRawMagicAfter = ub_out->GetRawMagic();
    EXPECT_EQ(inRawMagicAfter, outRawMagicAfter);
    EXPECT_EQ(inRawMagicBefore, inRawMagicAfter);
}

TEST_F(InplaceProcessTest, TestAssembleOnL1)
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
    // Assemble on L1
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

    G.AddOp(Opcode::OP_ASSEMBLE, {"mat_a_partial_0"}, {"mat_a_L1"}, "assemble_A_0");
    auto assemble_A_0 = G.GetOp("assemble_A_0");
    auto attrAssemble_0 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 0});
    assemble_A_0->SetOpAttribute(attrAssemble_0);
    G.AddOp(Opcode::OP_ASSEMBLE, {"mat_a_partial_1"}, {"mat_a_L1"}, "assemble_A_1");
    auto assemble_A_1 = G.GetOp("assemble_A_1");
    auto attrAssemble_1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 64});
    assemble_A_1->SetOpAttribute(attrAssemble_1);
    G.AddOp(Opcode::OP_ASSEMBLE, {"mat_a_partial_2"}, {"mat_a_L1"}, "assemble_A_2");
    auto assemble_A_2 = G.GetOp("assemble_A_2");
    auto attrAssemble_2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{64, 0});
    assemble_A_2->SetOpAttribute(attrAssemble_2);
    G.AddOp(Opcode::OP_ASSEMBLE, {"mat_a_partial_3"}, {"mat_a_L1"}, "assemble_A_3");
    auto assemble_A_3 = G.GetOp("assemble_A_3");
    auto attrAssemble_3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{64, 64});
    assemble_A_3->SetOpAttribute(attrAssemble_3);

    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"mat_b_L1"}, "L1_Copy_In_B");
    auto L1copyInB = G.GetOp("L1_Copy_In_B");
    auto attrCopyInB = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
                                                         OpImmediate::Specified(mat_b->GetShape()),
                                                         OpImmediate::Specified(mat_b->tensor->GetRawShape()));
    L1copyInB->SetOpAttribute(attrCopyInB);

    G.AddOp(Opcode::OP_L1_TO_L0A, {"mat_a_L1"}, {"mat_a_L0"}, "L1_To_L0A");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"mat_b_L1"}, {"mat_b_L0"}, "L1_To_L0B");

    G.AddOp(Opcode::OP_A_MUL_B, {"mat_a_L0", "mat_b_L0"}, {"mat_c_L0"}, "A_MUL_B");

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
    InplaceProcess passLocal;
    Status res = passLocal.Run(*function, "", "", 0);
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(mat_c->Datatype(), outputAstDtype);
    EXPECT_EQ(mat_a_partial_0->GetRawMagic(), mat_a_L1->GetRawMagic());
    EXPECT_EQ(mat_a_partial_0->GetOffset(), attrAssemble_0->GetToOffset());
    EXPECT_EQ(mat_a_partial_1->GetRawMagic(), mat_a_L1->GetRawMagic());
    EXPECT_EQ(mat_a_partial_1->GetOffset(), attrAssemble_1->GetToOffset());
    EXPECT_EQ(mat_a_partial_2->GetRawMagic(), mat_a_L1->GetRawMagic());
    EXPECT_EQ(mat_a_partial_2->GetOffset(), attrAssemble_2->GetToOffset());
    EXPECT_EQ(mat_a_partial_3->GetRawMagic(), mat_a_L1->GetRawMagic());
    EXPECT_EQ(mat_a_partial_3->GetOffset(), attrAssemble_3->GetToOffset());
}

inline void InplaceAssembleAddOp(ComputationalGraphBuilder& G)
{
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"mat_a_partial_0"}, "L1copyInA_0");
    auto L1copyInA_0 = G.GetOp("L1copyInA_0");
    auto attrCopyInA_0 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_L1, OpImmediate::Specified(G.GetTensor("mat_a")->GetShape()),
        OpImmediate::Specified(G.GetTensor("mat_a")->tensor->GetRawShape()));
    L1copyInA_0->SetOpAttribute(attrCopyInA_0);
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"mat_a_partial_1"}, "L1copyInA_1");
    auto L1copyInA_1 = G.GetOp("L1copyInA_1");
    auto attrCopyInA_1 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_L1, OpImmediate::Specified(G.GetTensor("mat_a")->GetShape()),
        OpImmediate::Specified(G.GetTensor("mat_a")->tensor->GetRawShape()));
    L1copyInA_1->SetOpAttribute(attrCopyInA_1);
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"mat_b_partial_0"}, "L1copyInB_0");
    auto L1copyInB_0 = G.GetOp("L1copyInB_0");
    auto attrCopyInB_0 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_L1, OpImmediate::Specified(G.GetTensor("mat_b")->GetShape()),
        OpImmediate::Specified(G.GetTensor("mat_b")->tensor->GetRawShape()));
    L1copyInB_0->SetOpAttribute(attrCopyInB_0);
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"mat_b_partial_1"}, "L1copyInB_1");
    auto L1copyInB_1 = G.GetOp("L1copyInB_1");
    auto attrCopyInB_1 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_L1, OpImmediate::Specified(G.GetTensor("mat_b")->GetShape()),
        OpImmediate::Specified(G.GetTensor("mat_b")->tensor->GetRawShape()));
    L1copyInB_1->SetOpAttribute(attrCopyInB_1);

    G.AddOp(Opcode::OP_L1_TO_L0A, {"mat_a_partial_0"}, {"mat_a_L0_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"mat_a_partial_1"}, {"mat_a_L0_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"mat_b_partial_0"}, {"mat_b_L0_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"mat_b_partial_1"}, {"mat_b_L0_1"}, "L1_To_L0B_1");

    G.AddOp(Opcode::OP_A_MUL_B, {"mat_a_L0_0", "mat_b_L0_0"}, {"mat_c_L0_0"}, "A_MUL_B");
    G.AddOp(Opcode::OP_A_MULACC_B, {"mat_a_L0_1", "mat_b_L0_1", "mat_c_L0_0"}, {"mat_c_L0_1"}, "A_MULACC_B");

    G.AddOp(Opcode::OP_ASSEMBLE, {"mat_c_L0_1"}, {"assemble_out_c"}, "assemble_c_1");
    auto assemble_c_1 = G.GetOp("assemble_c_1");
    auto attrAssemble_c_1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 64});
    assemble_c_1->SetOpAttribute(attrAssemble_c_1);
    G.AddOp(Opcode::OP_COPY_OUT, {"assemble_out_c"}, {"out_c"}, "L0C_Copy_out");
    auto L0C_Copy_out = G.GetOp("L0C_Copy_out");
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_L0C, OpImmediate::Specified(G.GetTensor("out_c")->GetShape()),
        OpImmediate::Specified(G.GetTensor("out_c")->tensor->GetRawShape()));
    L0C_Copy_out->SetOpAttribute(attrCopyOut);
}

inline void AssembleViewAddOp(ComputationalGraphBuilder& G)
{
    G.AddOp(Opcode::OP_COPY_IN, {"vec_in_0"}, {"copy_in_0"}, "copy_0");
    auto attrCopy_0 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_DEVICE_DDR,
        OpImmediate::Specified(G.GetTensor("vec_in_0")->GetShape()),
        OpImmediate::Specified(G.GetTensor("vec_in_0")->tensor->GetRawShape()));
    G.GetOp("copy_0")->SetOpAttribute(attrCopy_0);
    G.AddOp(Opcode::OP_COPY_IN, {"vec_in_1"}, {"copy_in_1"}, "copy_1");
    auto attrCopy_1 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_DEVICE_DDR,
        OpImmediate::Specified(G.GetTensor("vec_in_1")->GetShape()),
        OpImmediate::Specified(G.GetTensor("vec_in_1")->tensor->GetRawShape()));
    G.GetOp("copy_1")->SetOpAttribute(attrCopy_1);
    G.AddOp(Opcode::OP_COPY_IN, {"vec_in_2"}, {"copy_in_2"}, "copy_2");
    auto attrCopy_2 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_DEVICE_DDR,
        OpImmediate::Specified(G.GetTensor("vec_in_2")->GetShape()),
        OpImmediate::Specified(G.GetTensor("vec_in_2")->tensor->GetRawShape()));
    G.GetOp("copy_2")->SetOpAttribute(attrCopy_2);
    G.AddOp(Opcode::OP_COPY_IN, {"vec_in_3"}, {"copy_in_3"}, "copy_3");
    auto attrCopy_3 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_DEVICE_DDR,
        OpImmediate::Specified(G.GetTensor("vec_in_3")->GetShape()),
        OpImmediate::Specified(G.GetTensor("vec_in_3")->tensor->GetRawShape()));
    G.GetOp("copy_3")->SetOpAttribute(attrCopy_3);

    G.AddOp(Opcode::OP_ASSEMBLE, {"copy_in_0"}, {"assemble_out_0"}, "assemble_0");
    std::vector<int64_t> offestAssemble_0 = {0, 0};
    auto attrAssemble_0 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offestAssemble_0);
    G.GetOp("assemble_0")->SetOpAttribute(attrAssemble_0);
    G.AddOp(Opcode::OP_ASSEMBLE, {"copy_in_1"}, {"assemble_out_0"}, "assemble_1");
    std::vector<int64_t> offestAssemble_1 = {128, 0};
    auto attrAssemble_1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offestAssemble_1);
    G.GetOp("assemble_1")->SetOpAttribute(attrAssemble_1);

    G.AddOp(Opcode::OP_ASSEMBLE, {"copy_in_2"}, {"assemble_out_0"}, "assemble_2");
    std::vector<int64_t> offestAssemble_2 = {256, 0};
    auto attrAssemble_2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offestAssemble_2);
    G.GetOp("assemble_2")->SetOpAttribute(attrAssemble_2);

    G.AddOp(Opcode::OP_ASSEMBLE, {"copy_in_3"}, {"assemble_out_0"}, "assemble_3");
    std::vector<int64_t> offestAssemble_3 = {384, 0};
    auto attrAssemble_3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offestAssemble_3);
    G.GetOp("assemble_3")->SetOpAttribute(attrAssemble_3);

    G.AddOp(Opcode::OP_VIEW, {"assemble_out_0"}, {"view_0"}, "OP_VIEW_0");
    std::vector<int64_t> offestOpView0 = {256, 0};
    auto attrOpView0 = std::make_shared<ViewOpAttribute>(offestOpView0, MemoryType::MEM_DEVICE_DDR);
    G.GetOp("OP_VIEW_0")->SetOpAttribute(attrOpView0);

    G.AddOp(Opcode::OP_COPY_OUT, {"view_0"}, {"vec_out_0"}, "Copy_Out");
    std::vector<int64_t> offsetOut = {0, 0};
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified(offsetOut),
        OpImmediate::Specified(G.GetTensor("view_0")->GetShape()),
        OpImmediate::Specified(G.GetTensor("view_0")->tensor->GetRawShape()));
    G.GetOp("Copy_Out")->SetOpAttribute(attrCopyOut);
}

} // namespace tile_fwk
} // namespace npu
