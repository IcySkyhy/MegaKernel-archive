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
 * \file test_memory_reuse.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/graph_optimization/infer_discontinuous_input.h"

namespace npu {
namespace tile_fwk {

class TestInferDiscontinuousInput : public ::testing::Test {
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
};

void Construct(ComputationalGraphBuilder& G)
{
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor0");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor1");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor2");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor3");
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_DEVICE_DDR, "outputTensor");

    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor0"}, {"outputTensor"}, "assemble_0");
    auto assemble_0 = G.GetOp("assemble_0");
    auto attrAssemble_0 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{0, 0});
    assemble_0->SetOpAttribute(attrAssemble_0);

    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor1"}, {"outputTensor"}, "assemble_1");
    auto assemble_1 = G.GetOp("assemble_1");
    auto attrAssemble_1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR,
                                                                std::vector<int64_t>{16, 0});
    assemble_1->SetOpAttribute(attrAssemble_1);

    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor2"}, {"outputTensor"}, "assemble_2");
    auto assemble_2 = G.GetOp("assemble_2");
    auto attrAssemble_2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR,
                                                                std::vector<int64_t>{32, 0});
    assemble_2->SetOpAttribute(attrAssemble_2);

    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor3"}, {"outputTensor"}, "assemble_3");
    auto assemble_3 = G.GetOp("assemble_3");
    auto attrAssemble_3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR,
                                                                std::vector<int64_t>{48, 0});
    assemble_3->SetOpAttribute(attrAssemble_3);

    G.SetInCast({});
    G.SetOutCast({"outputTensor"});
}

TEST_F(TestInferDiscontinuousInput, testScenarioWithoutInsert_1)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    Construct(G);
    // run pass
    InferDiscontinuousInput inferDiscontinuousInput;
    EXPECT_EQ(inferDiscontinuousInput.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), SUCCESS);

    EXPECT_EQ(function->Operations().size(), 4);
}

TEST_F(TestInferDiscontinuousInput, testScenarioWithoutInsert_2)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    Construct(G);

    auto inputTensor2 = G.GetTensor("inputTensor2");
    auto inputTensor3 = G.GetTensor("inputTensor3");
    inputTensor3->tensor = inputTensor2->tensor;
    inputTensor2->tensor->UpdateRawShape({32, 128});
    inputTensor3->UpdateOffset({16, 0});

    // run pass
    InferDiscontinuousInput inferDiscontinuousInput;
    EXPECT_EQ(inferDiscontinuousInput.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), SUCCESS);

    EXPECT_EQ(function->Operations().size(), 4);
}

void check(Function* function, ComputationalGraphBuilder& G)
{
    InferDiscontinuousInput inferDiscontinuousInput;
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), FAILED);
    EXPECT_EQ(inferDiscontinuousInput.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), SUCCESS);
    EXPECT_EQ(function->Operations().size(), 12);

    auto inputTensor0 = G.GetTensor("inputTensor0");
    auto inputTensor1 = G.GetTensor("inputTensor1");
    auto inputTensor2 = G.GetTensor("inputTensor2");
    auto inputTensor3 = G.GetTensor("inputTensor3");
    auto viewOp0 = *inputTensor0->GetConsumers().begin();
    EXPECT_EQ(viewOp0->GetOpcode(), Opcode::OP_SLICE);
    auto viewOp1 = *inputTensor1->GetConsumers().begin();
    EXPECT_EQ(viewOp1->GetOpcode(), Opcode::OP_SLICE);
    auto viewOp2 = *inputTensor2->GetConsumers().begin();
    EXPECT_EQ(viewOp2->GetOpcode(), Opcode::OP_SLICE);
    auto viewOp3 = *inputTensor3->GetConsumers().begin();
    EXPECT_EQ(viewOp3->GetOpcode(), Opcode::OP_SLICE);
    auto insertTensor0 = viewOp0->GetOOperands()[0];
    EXPECT_EQ(insertTensor0->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    auto insertTensor1 = viewOp1->GetOOperands()[0];
    EXPECT_EQ(insertTensor1->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    auto insertTensor2 = viewOp2->GetOOperands()[0];
    EXPECT_EQ(insertTensor2->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    auto insertTensor3 = viewOp3->GetOOperands()[0];
    EXPECT_EQ(insertTensor3->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    auto assembleOp0 = *insertTensor0->GetConsumers().begin();
    EXPECT_EQ(assembleOp0->GetOpcode(), Opcode::OP_CONTRACT);
    auto assembleOp1 = *insertTensor1->GetConsumers().begin();
    EXPECT_EQ(assembleOp1->GetOpcode(), Opcode::OP_CONTRACT);
    auto assembleOp2 = *insertTensor2->GetConsumers().begin();
    EXPECT_EQ(assembleOp2->GetOpcode(), Opcode::OP_CONTRACT);
    auto assembleOp3 = *insertTensor3->GetConsumers().begin();
    EXPECT_EQ(assembleOp3->GetOpcode(), Opcode::OP_CONTRACT);
}
TEST_F(TestInferDiscontinuousInput, testScenarioInsert_1)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    Construct(G);
    auto inputTensor0 = G.GetTensor("inputTensor0");
    auto inputTensor1 = G.GetTensor("inputTensor1");
    auto inputTensor2 = G.GetTensor("inputTensor2");
    auto inputTensor3 = G.GetTensor("inputTensor3");
    inputTensor3->tensor = inputTensor1->tensor;
    inputTensor1->tensor->UpdateRawShape({32, 128});
    inputTensor3->UpdateOffset({16, 0});

    // run pass
    check(function, G);
}

TEST_F(TestInferDiscontinuousInput, testScenarioInsert_2)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    Construct(G);
    auto inputTensor0 = G.GetTensor("inputTensor0");
    auto inputTensor1 = G.GetTensor("inputTensor1");
    auto inputTensor2 = G.GetTensor("inputTensor2");
    auto inputTensor3 = G.GetTensor("inputTensor3");
    inputTensor0->tensor->UpdateRawShape({32, 128});
    inputTensor1->tensor->UpdateRawShape({32, 128});
    inputTensor2->tensor->UpdateRawShape({32, 128});
    inputTensor3->tensor->UpdateRawShape({32, 128});
    // run pass
    check(function, G);
}

TEST_F(TestInferDiscontinuousInput, testViewAssembleScenario)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "t1");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "t2");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "t3");
    G.AddOp(Opcode::OP_SLICE, {"t1"}, {"t2"}, "view");
    G.AddOp(Opcode::OP_CONTRACT, {"t2"}, {"t3"}, "assemble");
    auto view = G.GetOp("view");
    view->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto assemble = G.GetOp("assemble");
    assemble->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    // run pass
    InferDiscontinuousInput inferDiscontinuousInput;
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), FAILED);
    EXPECT_EQ(inferDiscontinuousInput.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), SUCCESS);
}

TEST_F(TestInferDiscontinuousInput, testCheckTensorAssembleNullAttr)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_UB, "t1");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_UB, "t2");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_UB, "t3");

    G.AddOp(Opcode::OP_VIEW, {"t1"}, {"t2"}, "view");
    G.AddOp(Opcode::OP_ASSEMBLE, {"t2"}, {"t3"}, "assemble");

    auto view = G.GetOp("view");
    view->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    InferDiscontinuousInput inferDiscontinuousInput;
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), FAILED);
}

TEST_F(TestInferDiscontinuousInput, testValidShapeInfer)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor0");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor1");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor2");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor3");
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_DEVICE_DDR, "outputTensor");

    G.GetTensor("inputTensor0")->UpdateDynValidShape({IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    G.GetTensor("inputTensor1")->UpdateDynValidShape({IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    G.GetTensor("inputTensor2")->UpdateDynValidShape({IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});
    G.GetTensor("inputTensor3")->UpdateDynValidShape({IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(128)});

    G.GetTensor("inputTensor3")->tensor = G.GetTensor("inputTensor1")->tensor;
    G.GetTensor("inputTensor1")->tensor->UpdateRawShape({32, 128});
    G.GetTensor("inputTensor3")->UpdateOffset({16, 0});

    G.SetInCast({"inputTensor0", "inputTensor1", "inputTensor2", "inputTensor3"});

    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor0"}, {"outputTensor"}, "assemble_0");
    auto attrAssemble_0 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{0, 0});
    G.GetOp("assemble_0")->SetOpAttribute(attrAssemble_0);

    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor1"}, {"outputTensor"}, "assemble_1");
    auto attrAssemble_1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR,
                                                                std::vector<int64_t>{16, 0});
    G.GetOp("assemble_1")->SetOpAttribute(attrAssemble_1);

    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor2"}, {"outputTensor"}, "assemble_2");
    auto attrAssemble_2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR,
                                                                std::vector<int64_t>{32, 0});
    G.GetOp("assemble_2")->SetOpAttribute(attrAssemble_2);

    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor3"}, {"outputTensor"}, "assemble_3");
    auto attrAssemble_3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR,
                                                                std::vector<int64_t>{48, 0});
    G.GetOp("assemble_3")->SetOpAttribute(attrAssemble_3);

    G.SetOutCast({"outputTensor"});

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    InferDiscontinuousInput inferDiscontinuousInput;
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), FAILED);
    EXPECT_EQ(inferDiscontinuousInput.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), SUCCESS);

    auto insertedViewOps = function->Operations().DuplicatedOpList();
    bool foundViewLikeWithValidShape = false;
    for (const auto& op : insertedViewOps) {
        if (IsViewLike(op->GetOpcode())) {
            auto oOperand = op->GetOOperands().front();
            auto dynValidShape = oOperand->GetDynValidShape();
            if (!dynValidShape.empty()) {
                foundViewLikeWithValidShape = true;
                EXPECT_EQ(dynValidShape.size(), 2);
            }
        }
    }
    EXPECT_TRUE(foundViewLikeWithValidShape);
}

// Construct DDR VIEW->ASSEMBLE chain: DDRTensorAssignUB converts tensor type to UB
// so InsertTensorCopy skips copy insertion, no extra VIEW/ASSEMBLE inserted.
void ConstructDDRTensorAssignUB(ComputationalGraphBuilder& G)
{
    G.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_DEVICE_DDR, "t80");
    G.GetTensor("t80")->tensor->UpdateRawShape({128, 128});
    G.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "t327");
    G.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "t339");

    // === Intermediate tensors ===
    // VIEW from t80
    G.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_DEVICE_DDR, "t61");
    G.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_DEVICE_DDR, "t60");
    G.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_DEVICE_DDR, "t63");
    G.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_DEVICE_DDR, "t62");
    // ASSEMBLE output
    G.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "t609");
    G.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "t613");
    // VIEW from assembled
    G.AddTensor(DataType::DT_FP32, {128, 1}, "t397");
    G.AddTensor(DataType::DT_FP32, {128, 1}, "t401");
    // SUB output
    G.AddTensor(DataType::DT_FP32, {128, 1}, "t396");
    G.AddTensor(DataType::DT_FP32, {128, 1}, "t400");
    // EXP output -> final
    G.AddTensor(DataType::DT_FP32, {128, 1}, "t403");
    G.AddTensor(DataType::DT_FP32, {128, 1}, "t405");
    // COPY_OUT output
    G.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "t900");
    G.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "t901");

    // === DDR VIEW -> ASSEMBLE chain ===

    G.AddOp(Opcode::OP_VIEW, {"t80"}, {"t61"}, "view_80_61");
    G.GetOp("view_80_61")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_VIEW, {"t80"}, {"t60"}, "view_80_60");
    G.GetOp("view_80_60")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_VIEW, {"t80"}, {"t63"}, "view_80_63");
    G.GetOp("view_80_63")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_VIEW, {"t80"}, {"t62"}, "view_80_62");
    G.GetOp("view_80_62")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));

    G.AddOp(Opcode::OP_ASSEMBLE, {"t61"}, {"t609"}, "asm_61_609");
    G.GetOp("asm_61_609")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0}));
    G.AddOp(Opcode::OP_ASSEMBLE, {"t60"}, {"t609"}, "asm_60_609");
    G.GetOp("asm_60_609")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_ASSEMBLE, {"t63"}, {"t613"}, "asm_63_613");
    G.GetOp("asm_63_613")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{64, 0}));
    G.AddOp(Opcode::OP_ASSEMBLE, {"t62"}, {"t613"}, "asm_62_613");
    G.GetOp("asm_62_613")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    G.AddOp(Opcode::OP_VIEW, {"t609"}, {"t397"}, "view_609_397");
    G.GetOp("view_609_397")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_VIEW, {"t613"}, {"t401"}, "view_613_401");
    G.GetOp("view_613_401")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));

    G.AddOp(Opcode::OP_SUB, {"t397", "t327"}, {"t396"}, "sub_397_327");
    G.AddOp(Opcode::OP_SUB, {"t401", "t339"}, {"t400"}, "sub_401_339");

    G.AddOp(Opcode::OP_EXP, {"t396"}, {"t403"}, "exp_396");
    G.AddOp(Opcode::OP_EXP, {"t400"}, {"t405"}, "exp_400");

    G.AddOp(Opcode::OP_COPY_OUT, {"t403"}, {"t900"}, "copyout_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"t405"}, {"t901"}, "copyout_1");

    // === Set InCast and OutCast ===
    G.SetInCast({"t80", "t327", "t339"});
    G.SetOutCast({"t900", "t901"});
}

// DDR discontinuous input through VIEW->ASSEMBLE: DDRTensorAssignUB converts type,
// InsertTensorCopy skips copy, no extra VIEW/ASSEMBLE inserted.
TEST_F(TestInferDiscontinuousInput, testDDRTensorAssignUBNoInsert)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    ConstructDDRTensorAssignUB(G);
    // run pass
    InferDiscontinuousInput inferDiscontinuousInput;
    EXPECT_EQ(inferDiscontinuousInput.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), SUCCESS);

    // Verify no new VIEW or ASSEMBLE ops were inserted (SLICE/CONTRACT may be inserted instead)
    auto opList = function->Operations().DuplicatedOpList();
    int viewCount = 0;
    int assembleCount = 0;
    for (auto& op : opList) {
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            viewCount++;
        }
        if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleCount++;
        }
    }
    EXPECT_EQ(viewCount, 6);
    EXPECT_EQ(assembleCount, 4);
}
} // namespace tile_fwk
} // namespace npu
