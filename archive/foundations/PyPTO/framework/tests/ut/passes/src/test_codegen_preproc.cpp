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
 * \file test_codegen_preproc.cpp
 * \brief Unit test for codegen_preproc pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/block_graph_pass/codegen_preproc.h"
#include "passes/tile_graph_pass/graph_constraint/pad_local_buffer.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"
#include "computational_graph_builder.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/tensor_offset.h"
#include "ut_json/ut_json_tool.h"
#include <vector>
#include <string>
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
constexpr int CP_NUM1 = 1;
constexpr int CP_NUM16 = 16;
constexpr int CP_NUM64 = 64;
constexpr int CP_NUM256 = 256;
const std::string REDUCE_AXIS = OP_ATTR_PREFIX + "AXIS";

class CodegenPreprocTest : public testing::Test {
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

TEST_F(CodegenPreprocTest, TestSaveGmTensorParamIdxToOp)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestSaveGmTensorParamIdxToOp",
                                                  "TestSaveGmTensorParamIdxToOp", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSaveGmTensorParamIdxToOpLeaf",
                                                      "TestSaveGmTensorParamIdxToOpLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);

    std::vector<int64_t> shape = {CP_NUM16, CP_NUM16};
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

    int index{0};
    for (auto op : opLogPtr) {
        if (OpcodeManager::Inst().IsCopyInOrOut(op->GetOpcode())) {
            if (IsCopyIn(op->GetOpcode()))
                op->SetIOpAtt(0, index++);
            else
                op->SetOOpAtt(0, index++);
        }
    }

    CodegenPreproc codegenPreprocPass;
    codegenPreprocPass.SaveGmTensorParamIdxToOp(*rootFuncPtr);

    for (const auto& op : opLogPtr) {
        if (OpcodeManager::Inst().IsCopyInOrOut(op->GetOpcode())) {
            EXPECT_TRUE(op->HasAttr(OpAttributeKey::gmTensorParamIdxInCall));
        }
    }
}

TEST_F(CodegenPreprocTest, TestCombineAxisRowSumLine)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 12, 1}, MemoryType::MEM_DEVICE_DDR, "gm_in"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 12, 1}, MemoryType::MEM_UB, "in"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 12, 1}, MemoryType::MEM_UB, "out"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 12, 1}, MemoryType::MEM_DEVICE_DDR, "gm_out"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 8}, MemoryType::MEM_UB, "tmp"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm_in"}, {"in"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWSUMLINE, {"in"}, {"out", "tmp"}, "sumline", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"out"}, {"gm_out"}, "copy_out", true), true);
    auto sumline = graph.GetOp("sumline");
    sumline->SetAttribute(REDUCE_AXIS, 0);

    auto funcPtr = graph.GetFunction();
    funcPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*funcPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*funcPtr), SUCCESS);

    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxis", "TestCombineAxis",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxisLeaf",
                                                      "TestCombineAxisLeaf", graph.GetFunction());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), graph.GetFunction());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);
    rootFuncPtr->paramConfigs_.combineAxis = true;

    CodegenPreproc codegenPreprocPass;
    EXPECT_EQ(codegenPreprocPass.RunOnFunction(*rootFuncPtr), SUCCESS);
    // Verify AxisCombine
    auto updatedOperations = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto& op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
    // Verify PadLocalBuffer
    auto tmp = graph.GetTensor("tmp");
    auto shape = tmp->GetRawTensor()->GetRawShape();
    EXPECT_EQ(shape[shape.size() - 1], CP_NUM16);
    // Verify CodegenPreproc
    sumline = graph.GetOp("sumline");
    std::vector<bool> attr;
    EXPECT_TRUE(sumline->HasAttr(OpAttributeKey::outputCombineAxis));
    sumline->GetAttr(OpAttributeKey::outputCombineAxis, attr);
    EXPECT_EQ(attr, (std::vector<bool>{true, false}));
}

TEST_F(CodegenPreprocTest, TestCombineAxisExpand)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "c1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t1"}, {"t2"}, "expand", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t3"}, "c2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_SUB, {"t2", "t3"}, {"t4"}, "sub", true), true);
    auto expand = graph.GetOp("expand");
    expand->SetAttribute(OpAttributeKey::expandDims, std::vector<int>{0});

    auto funcPtr = graph.GetFunction();
    funcPtr->paramConfigs_.combineAxis = true;
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*funcPtr), SUCCESS);

    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxis", "TestCombineAxis",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxisLeaf",
                                                      "TestCombineAxisLeaf", graph.GetFunction());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), graph.GetFunction());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);
    rootFuncPtr->paramConfigs_.combineAxis = true;

    CodegenPreproc codegenPreprocPass;
    EXPECT_EQ(codegenPreprocPass.RunOnFunction(*rootFuncPtr), SUCCESS);
    // Verify CodegenPreproc
    auto afterExpand = graph.GetOp("expand");
    std::vector<int64_t> axes = afterExpand->GetVectorIntAttribute(OpAttributeKey::expandDims);
    ASSERT_EQ(axes.size(), 1);
    EXPECT_EQ(axes[0], 1);
}

// 隐式expand
TEST_F(CodegenPreprocTest, TestCombineAxisExpandinline)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_DEVICE_DDR, "in2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "c1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in2"}, {"t2"}, "c2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_SUB, {"t1", "t2"}, {"t3"}, "sub", true), true);
    auto sub = graph.GetOp("sub");
    sub->SetAttribute(OpAttributeKey::brcOperand, std::vector<int64_t>{1, 0});

    auto funcPtr = graph.GetFunction();
    funcPtr->paramConfigs_.combineAxis = true;
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*funcPtr), SUCCESS);

    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxis", "TestCombineAxis",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxisLeaf",
                                                      "TestCombineAxisLeaf", graph.GetFunction());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), graph.GetFunction());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);
    rootFuncPtr->paramConfigs_.combineAxis = true;

    CodegenPreproc codegenPreprocPass;
    EXPECT_EQ(codegenPreprocPass.RunOnFunction(*rootFuncPtr), SUCCESS);
    // Verify AxisCombine
    auto updatedOperations = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto& op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
    // Verify PadLocalBuffer
    EXPECT_EQ(graph.GetTensor("t1")->GetRawTensor()->GetRawShape(), (Shape{8, 1}));
    EXPECT_EQ(graph.GetTensor("t2")->GetRawTensor()->GetRawShape(), (Shape{16, 1}));
    // Verify CodegenPreproc
    EXPECT_EQ(sub->GetVectorIntAttribute(OpAttributeKey::brcOperand), (std::vector<int64_t>{0, 1}));
    EXPECT_EQ(sub->GetIntAttribute(OpAttributeKey::brcbIdx), 1);
}

// expand input have multi consumer
TEST_F(CodegenPreprocTest, TestCombineAxisExpand2)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t2"}, "copyin1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXP, {"t2"}, {"t3"}, "exp", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t3"}, {"t4"}, "expand", true), true);
    auto expand = graph.GetOp("expand");
    expand->SetAttribute(OpAttributeKey::expandDims, std::vector<int64_t>{0});
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "in2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t22"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in2"}, {"t22"}, "copyin12", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t32"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_MUL, {"t3", "t22"}, {"t32"}, "mul1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "out1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t32"}, {"out1"}, "copyout", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_DEVICE_DDR, "in3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_UB, "t23"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in3"}, {"t23"}, "copyin13", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_UB, "t33"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_MUL, {"t23", "t4"}, {"t33"}, "mul2", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_DEVICE_DDR, "out2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t33"}, {"out2"}, "copyout2", true), true);

    auto funcPtr = graph.GetFunction();
    funcPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*funcPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*funcPtr), SUCCESS);

    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxis", "TestCombineAxis",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxisLeaf",
                                                      "TestCombineAxisLeaf", graph.GetFunction());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), graph.GetFunction());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);
    rootFuncPtr->paramConfigs_.combineAxis = true;

    CodegenPreproc codegenPreprocPass;
    EXPECT_EQ(codegenPreprocPass.RunOnFunction(*rootFuncPtr), SUCCESS);
    // Verify PadLocalBuffer
    EXPECT_EQ(graph.GetTensor("t3")->GetRawTensor()->GetRawShape(), (Shape{8, 1}));
    // Verify CodegenPreproc
    auto afterExpand = graph.GetOp("expand");
    std::vector<int64_t> axes = afterExpand->GetVectorIntAttribute(OpAttributeKey::expandDims);
    EXPECT_EQ(axes.size(), 1);
    EXPECT_EQ(axes[0], 1);
    std::vector<bool> inputAttr;
    EXPECT_TRUE(expand->HasAttr(OpAttributeKey::inputCombineAxis));
    expand->GetAttr(OpAttributeKey::inputCombineAxis, inputAttr);
    EXPECT_EQ(inputAttr, (std::vector<bool>{true}));
    std::vector<bool> outAttr;
    EXPECT_TRUE(expand->HasAttr(OpAttributeKey::outputCombineAxis));
    expand->GetAttr(OpAttributeKey::outputCombineAxis, outAttr);
    EXPECT_EQ(outAttr, (std::vector<bool>{true}));
}

TEST_F(CodegenPreprocTest, TestCombineAxis3510)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 32}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 32}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "c1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t1"}, {"t2"}, "expand", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t3"}, "c2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_SUB, {"t2", "t3"}, {"t4"}, "sub", true), true);
    auto expand = graph.GetOp("expand");
    expand->SetAttribute(OpAttributeKey::expandDims, std::vector<int>{0});

    auto funcPtr = graph.GetFunction();
    funcPtr->paramConfigs_.combineAxis = true;
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*funcPtr), SUCCESS);

    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxis", "TestCombineAxis",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxisLeaf",
                                                      "TestCombineAxisLeaf", graph.GetFunction());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), graph.GetFunction());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);
    rootFuncPtr->paramConfigs_.combineAxis = true;

    CodegenPreproc codegenPreprocPass;
    EXPECT_EQ(codegenPreprocPass.RunOnFunction(*rootFuncPtr), SUCCESS);
    // Verify CodegenPreproc
    auto afterExpand = graph.GetOp("sub");
    EXPECT_EQ(afterExpand->HasAttr(OpAttributeKey::outputCombineAxis), false);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(CodegenPreprocTest, TestSaveGmTensorParamIdxToOpPermute)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestSaveGmParamPermute",
                                                  "TestSaveGmParamPermute", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSaveGmParamPermuteLeaf",
                                                      "TestSaveGmParamPermuteLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);

    std::vector<int64_t> shape = {CP_NUM16, CP_NUM16, CP_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& copyin = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    copyin.SetIOpAtt(0, 0);
    auto& permute_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_PERMUTE, {tensor3}, {tensor2});
    permute_op.SetIOpAtt(0, 0);
    auto& copyout = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensor2}, {tensor4});
    copyout.SetOOpAtt(0, 1);

    CodegenPreproc codegenPreprocPass;
    codegenPreprocPass.SaveGmTensorParamIdxToOp(*rootFuncPtr);

    EXPECT_TRUE(copyin.HasAttr(OpAttributeKey::gmTensorParamIdxInCall));
    EXPECT_TRUE(permute_op.HasAttr(OpAttributeKey::gmTensorParamIdxInCall));
    EXPECT_TRUE(copyout.HasAttr(OpAttributeKey::gmTensorParamIdxInCall));
}

TEST_F(CodegenPreprocTest, TestSaveGmTensorParamIdxToOpPermuteElement)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestSaveGmParamPermuteElem",
                                                  "TestSaveGmParamPermuteElem", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSaveGmParamPermuteElemLeaf",
                                                      "TestSaveGmParamPermuteElemLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);

    std::vector<int64_t> shape = {CP_NUM16, CP_NUM16, CP_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& copyin = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    copyin.SetIOpAtt(0, 0);
    auto& permute_elem_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_PERMUTE_ELEMENT, {tensor3},
                                                           {tensor2});
    permute_elem_op.SetIOpAtt(0, 0);
    auto& copyout = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensor2}, {tensor4});
    copyout.SetOOpAtt(0, 1);

    CodegenPreproc codegenPreprocPass;
    codegenPreprocPass.SaveGmTensorParamIdxToOp(*rootFuncPtr);

    EXPECT_TRUE(copyin.HasAttr(OpAttributeKey::gmTensorParamIdxInCall));
    EXPECT_TRUE(permute_elem_op.HasAttr(OpAttributeKey::gmTensorParamIdxInCall));
    EXPECT_TRUE(copyout.HasAttr(OpAttributeKey::gmTensorParamIdxInCall));
}

TEST_F(CodegenPreprocTest, TestCombineAxis3510BothLastDimOne)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_DEVICE_DDR, "in2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "c1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in2"}, {"t2"}, "c2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t1", "t2"}, {"t3"}, "add", true), true);

    auto funcPtr = graph.GetFunction();
    funcPtr->paramConfigs_.combineAxis = true;
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*funcPtr), SUCCESS);

    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxis3510BLast1",
                                                  "TestCombineAxis3510BLast1", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCombineAxis3510BLast1Leaf",
                                                      "TestCombineAxis3510BLast1Leaf", graph.GetFunction());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), graph.GetFunction());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);
    rootFuncPtr->paramConfigs_.combineAxis = true;

    CodegenPreproc codegenPreprocPass;
    EXPECT_EQ(codegenPreprocPass.RunOnFunction(*rootFuncPtr), SUCCESS);

    auto addOp = graph.GetOp("add");
    EXPECT_TRUE(addOp->HasAttr(OpAttributeKey::outputCombineAxis));
    EXPECT_TRUE(addOp->HasAttr(OpAttributeKey::inputCombineAxis));
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(CodegenPreprocTest, TestGenGmOoRCheckInfoCopyInCopyOut)
{
    ComputationalGraphBuilder graph;
    std::vector<int64_t> shape = {4, 8};

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, shape, MemoryType::MEM_DEVICE_DDR, "ddr_in"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, shape, MemoryType::MEM_UB, "ub_buf"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, shape, MemoryType::MEM_DEVICE_DDR, "ddr_out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"ddr_in"}, {"ub_buf"}, "copyin", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"ub_buf"}, {"ddr_out"}, "copyout", true), true);

    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestGmOoRCheck", "TestGmOoRCheck", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGmOoRCheckLeaf",
                                                      "TestGmOoRCheckLeaf", graph.GetFunction());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), graph.GetFunction());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);

    CodegenPreproc codegenPreprocPass;
    EXPECT_EQ(codegenPreprocPass.RunOnFunction(*rootFuncPtr), SUCCESS);

    // Expected values for shape={4,8}, rawShape={4,8}, offset={0,0}:
    //   strides = [8, 1]
    //   oneDimOffset = 0*8 + 0*1 = 0
    //   oneDimExtent = (4-1)*8 + (8-1)*1 + 1 = 24 + 7 + 1 = 32
    //   totalSize = 4 * 8 = 32

    auto copyinOp = graph.GetOp("copyin");
    auto copyinAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyinOp->GetOpAttribute());
    ASSERT_NE(copyinAttr, nullptr);
    const auto* copyinCheck = copyinAttr->GetGmOutOfRangeCheck();
    ASSERT_NE(copyinCheck, nullptr);
    EXPECT_EQ(copyinCheck->accessType, GmOutOfRangeCheckInfo::AccessType::READ_GM);
    EXPECT_EQ(copyinCheck->oneDimOffset.GetSpecifiedValue().Dump(), "0");
    EXPECT_EQ(copyinCheck->oneDimExtent.GetSpecifiedValue().Dump(), "32");
    EXPECT_EQ(copyinCheck->totalSize.GetSpecifiedValue().Dump(), "32");

    auto copyoutOp = graph.GetOp("copyout");
    auto copyoutAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyoutOp->GetOpAttribute());
    ASSERT_NE(copyoutAttr, nullptr);
    const auto* copyoutCheck = copyoutAttr->GetGmOutOfRangeCheck();
    ASSERT_NE(copyoutCheck, nullptr);
    EXPECT_EQ(copyoutCheck->accessType, GmOutOfRangeCheckInfo::AccessType::WRITE_GM);
    EXPECT_EQ(copyoutCheck->oneDimOffset.GetSpecifiedValue().Dump(), "0");
    EXPECT_EQ(copyoutCheck->oneDimExtent.GetSpecifiedValue().Dump(), "32");
    EXPECT_EQ(copyoutCheck->totalSize.GetSpecifiedValue().Dump(), "32");
}

TEST_F(CodegenPreprocTest, TestGenGmOoRCheckInfoWithOffset)
{
    ComputationalGraphBuilder graph;
    std::vector<int64_t> shape = {4, 8};

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, shape, MemoryType::MEM_DEVICE_DDR, "ddr_in"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, shape, MemoryType::MEM_UB, "ub_buf"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"ddr_in"}, {"ub_buf"}, "copyin", true), true);

    auto copyinOp = graph.GetOp("copyin");
    auto copyinAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyinOp->GetOpAttribute());
    ASSERT_NE(copyinAttr, nullptr);
    copyinAttr->SetFromOffset(OpImmediate::Specified({2, 3}));

    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestGmOoROffset", "TestGmOoROffset",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGmOoROffsetLeaf",
                                                      "TestGmOoROffsetLeaf", graph.GetFunction());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), graph.GetFunction());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);

    CodegenPreproc codegenPreprocPass;
    EXPECT_EQ(codegenPreprocPass.RunOnFunction(*rootFuncPtr), SUCCESS);

    // Expected values for shape={4,8}, rawShape={4,8}, offset={2,3}:
    //   strides = [8, 1]
    //   oneDimOffset = 2*8 + 3*1 = 19
    //   oneDimExtent = (4-1)*8 + (8-1)*1 + 1 = 32
    //   totalSize = 4 * 8 = 32
    const auto* check = copyinAttr->GetGmOutOfRangeCheck();
    ASSERT_NE(check, nullptr);
    EXPECT_EQ(check->accessType, GmOutOfRangeCheckInfo::AccessType::READ_GM);
    EXPECT_EQ(check->oneDimOffset.GetSpecifiedValue().Dump(), "19");
    EXPECT_EQ(check->oneDimExtent.GetSpecifiedValue().Dump(), "32");
    EXPECT_EQ(check->totalSize.GetSpecifiedValue().Dump(), "32");
}

TEST_F(CodegenPreprocTest, TestGenGmOoRCheckInfoNonDdrSkipped)
{
    ComputationalGraphBuilder graph;
    std::vector<int64_t> shape = {4, 8};

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, shape, MemoryType::MEM_UB, "ub_in"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, shape, MemoryType::MEM_UB, "ub_out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"ub_in"}, {"ub_out"}, "copyin", true), true);

    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestGmOoRSkip", "TestGmOoRSkip", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGmOoRSkipLeaf", "TestGmOoRSkipLeaf",
                                                      graph.GetFunction());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), graph.GetFunction());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);

    CodegenPreproc codegenPreprocPass;
    EXPECT_EQ(codegenPreprocPass.RunOnFunction(*rootFuncPtr), SUCCESS);

    auto copyinOp = graph.GetOp("copyin");
    auto copyinAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyinOp->GetOpAttribute());
    ASSERT_NE(copyinAttr, nullptr);
    EXPECT_EQ(copyinAttr->GetGmOutOfRangeCheck(), nullptr);
}

} // namespace tile_fwk
} // namespace npu
