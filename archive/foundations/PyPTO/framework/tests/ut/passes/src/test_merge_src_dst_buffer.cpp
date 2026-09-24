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
 * \file test_merge_src_dst_buffer.cpp
 * \brief Unit test for SrcDstBufferMerge pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/block_graph_pass/memory_reuse/merge_src_dst_buffer.h"
#include "passes/pass_utils/pass_utils.h"
#include "computational_graph_builder.h"
#include <fstream>
#include <vector>
#include <string>
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {

class MergeSrcDstBufferTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void StubInputOutput(Function* function);

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(MergeSrcDstBufferTest, AppointInplace)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<int64_t> shape = {128, 128};
    auto shapeImme = OpImmediate::Specified(shape);
    std::vector<int64_t> offset = {0, 0};

    std::shared_ptr<LogicalTensor> tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor1->SetMemoryTypeToBe(MEM_DEVICE_DDR);

    std::shared_ptr<LogicalTensor> tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor2->SetMemoryTypeToBe(MEM_DEVICE_DDR);

    std::shared_ptr<LogicalTensor> tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);

    std::shared_ptr<LogicalTensor> tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor4->SetMemoryTypeOriginal(MEM_UB);
    tensor4->SetMemoryTypeToBe(MEM_UB);

    std::shared_ptr<LogicalTensor> tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor5->SetMemoryTypeOriginal(MEM_UB);
    tensor5->SetMemoryTypeToBe(MEM_UB);

    auto& alloc1 = PassOperationUtils::AddOperation(function, Opcode::OP_UB_ALLOC, {}, {tensor3});
    alloc1.UpdateLatency(1);
    auto& copyin1 = PassOperationUtils::AddOperation(function, Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    copyin1.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MEM_UB, shapeImme,
                                                             shapeImme, std::vector<npu::tile_fwk::OpImmediate>()));

    auto& alloc2 = PassOperationUtils::AddOperation(function, Opcode::OP_UB_ALLOC, {}, {tensor4});
    alloc2.UpdateLatency(1);
    auto& copyin2 = PassOperationUtils::AddOperation(function, Opcode::OP_COPY_IN, {tensor2}, {tensor4});
    copyin2.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MEM_UB, shapeImme,
                                                             shapeImme, std::vector<npu::tile_fwk::OpImmediate>()));

    auto& alloc3 = PassOperationUtils::AddOperation(function, Opcode::OP_UB_ALLOC, {}, {tensor5});
    alloc3.UpdateLatency(1);
    auto& add1 = PassOperationUtils::AddOperation(function, Opcode::OP_ADD, {tensor3, tensor4}, {tensor5});
    add1.SetAttribute(OpAttributeKey::inplaceIdx, 0);

    SrcDstBufferMergeImpl srcDstMerge;
    Function func(Program::GetInstance(), "", "", nullptr);
    Function func1(Program::GetInstance(), "", "", nullptr);
    Function* rootFunc = &func1;
    rootFunc->programs_.insert(std::pair<uint64_t, Function*>(1, &function));
    func.rootFunc_ = rootFunc;
    srcDstMerge.Run(func);
}

void MergeSrcDstBufferTest::StubInputOutput(Function* function)
{
    Function* rootFunc = function;
    rootFunc->programs_.insert(std::pair<uint64_t, Function*>(1, function));
    function->rootFunc_ = rootFunc;
}

TEST_F(MergeSrcDstBufferTest, AddReplaced)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3", "t4"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"COPYIN1", "COPYIN2", "ADD", "COPYOUT"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);

    std::string jsonFilePath = "./config/pass/json/merge_src_dst_buffer_add_replaced.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        function->DumpJsonFile(jsonFilePath);
    }

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ADD) {
            auto outputTensor = op.GetOOperands()[0];
            auto inputTensor = op.GetIOperands()[0];
            EXPECT_NE(outputTensor->memoryrange.memId, -1);
            EXPECT_EQ(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
            break;
        }
    }
}

TEST_F(MergeSrcDstBufferTest, AddHasInReplaced)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3", "t4"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"COPYIN1", "COPYIN2", "ADD", "COPYOUT"};
    EXPECT_EQ(G.AddTensors(DataType::DT_INT32, {32, 32}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);
    for (auto& subProgram : function->rootFunc_->programs_) {
        auto opList = subProgram.second->Operations().DuplicatedOpList();
        for (auto& op : opList) {
            if (op->GetOpcode() != Opcode::OP_ADD) {
                continue;
            }
            op->SetAttribute(OpAttributeKey::inplaceIdx, 0);
        }
    }

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ADD) {
            auto outputTensor = op.GetOOperands()[0];
            auto inputTensor = op.GetIOperands()[0];
            EXPECT_NE(outputTensor->memoryrange.memId, -1);
            EXPECT_EQ(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
            break;
        }
    }
}

TEST_F(MergeSrcDstBufferTest, CopyInNotReplaced)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"COPYIN1"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t2"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            auto outputTensor = op.GetOOperands()[0];
            auto inputTensor = op.GetIOperands()[0];
            EXPECT_NE(outputTensor->memoryrange.memId, -1);
            EXPECT_NE(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
            break;
        }
    }
}

TEST_F(MergeSrcDstBufferTest, PairMaxNotReplaced)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_PAIRMAX};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"PAIRMAX"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Operation* pairMaxOp = G.GetOp("PAIRMAX");
    pairMaxOp->SetAttribute(OpAttributeKey::excludeBufferReuse, true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t2"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);
    function->GetRootFunction()->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->GetRootFunction()->SetGraphType(GraphType::EXECUTE_GRAPH);

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_PAIRMAX) {
            auto outputTensor = op.GetOOperands()[0];
            auto inputTensor = op.GetIOperands()[0];
            EXPECT_NE(outputTensor->memoryrange.memId, -1);
            EXPECT_NE(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
            break;
        }
    }
}

TEST_F(MergeSrcDstBufferTest, IsCubeNotReplaced)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_PAIRMAX};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"PAIRMAX"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t2"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);
    for (auto& subProgram : function->rootFunc_->programs_) {
        auto opList = subProgram.second->Operations().DuplicatedOpList();
        for (auto& op : opList) {
            op->SetAttr(OpAttributeKey::isCube, true);
        }
    }

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_PAIRMAX) {
            auto outputTensor = op.GetOOperands()[0];
            auto inputTensor = op.GetIOperands()[0];
            EXPECT_NE(outputTensor->memoryrange.memId, -1);
            EXPECT_NE(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
            break;
        }
    }
}

TEST_F(MergeSrcDstBufferTest, AddDiffMemTypeNotReplaced)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_SUB, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3", "t4"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"COPYIN1", "COPYIN2", "SUB", "COPYOUT"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);
    for (auto& subProgram : function->rootFunc_->programs_) {
        auto opList = subProgram.second->Operations().DuplicatedOpList();
        for (auto& op : opList) {
            if (op->GetOpcode() != Opcode::OP_SUB) {
                continue;
            }
            for (auto& output : op->GetOOperands()) {
                output->SetMemoryTypeOriginal(MemoryType::MEM_UB);
            }
            for (auto& in : op->GetIOperands()) {
                in->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
            }
        }
    }

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SUB) {
            auto outputTensor = op.GetOOperands()[0];
            auto inputTensor = op.GetIOperands()[0];
            EXPECT_NE(outputTensor->memoryrange.memId, -1);
            EXPECT_NE(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
            break;
        }
    }
}

TEST_F(MergeSrcDstBufferTest, AddDiffShapeNotReplaced)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<std::string> tensorNames1{"t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_MUL, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3", "t4"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"COPYIN1", "COPYIN2", "MUL", "COPYOUT"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 32}, tensorNames1), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_MUL) {
            auto outputTensor = op.GetOOperands()[0];
            auto inputTensor = op.GetIOperands()[0];
            EXPECT_NE(outputTensor->memoryrange.memId, -1);
            EXPECT_NE(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
            break;
        }
    }
}

TEST_F(MergeSrcDstBufferTest, AddMultiConsumerNotReplaced)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN,
                                Opcode::OP_DIV,     Opcode::OP_SUB,     Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}, {"t4", "t5"}, {"t4", "t6"}, {"t7", "t8"}};
    std::vector<std::vector<std::string>> ooperands{{"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"COPYIN1", "COPYIN2", "COPYIN3", "DIV", "SUB", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2", "t3"}), true);
    EXPECT_EQ(G.SetOutCast({"t9"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_DIV) {
            auto outputTensor = op.GetOOperands()[0];
            auto inputTensor = op.GetIOperands()[0];
            EXPECT_NE(outputTensor->memoryrange.memId, -1);
            EXPECT_NE(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
            break;
        }
    }
}

TEST_F(MergeSrcDstBufferTest, ReplaceOnBitWidthMatch)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<std::string> tensorNames1{"t5", "t6", "t7", "t8"};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_CAST,
                                Opcode::OP_CAST,    Opcode::OP_ADD,     Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}};
    std::vector<std::string> opNames{"COPYIN1", "COPYIN2", "CAST1", "CAST2", "ADD", "COPYOUT"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_INT32, {16, 16}, tensorNames1), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t8"}), true);
    Function* function = G.GetFunction();

    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CAST) {
            auto outputTensor = op.GetOOperands()[0];
            auto inputTensor = op.GetIOperands()[0];
            EXPECT_NE(outputTensor->memoryrange.memId, -1);
            EXPECT_EQ(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
            break;
        }
    }
}

void ConstructGrapgForReusePreMulMem(ComputationalGraphBuilder& G)
{
    // add tensor
    DataType dateType = DataType::DT_FP16;
    Shape shape = {16, 16};
    std::vector<std::string> tensorNames{"matA1DDR", "matA2DDR", "matB1DDR", "matA1L1",  "matA2L1",
                                         "matB1L1",  "matB2L1",  "matA1L0A", "matA2L0A", "matB1L0B",
                                         "matB2L0B", "matC1L0C", "matC2L0C", "outcast"};
    std::vector<MemoryType> tensorMemoryType{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1,
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L0A,
        MemoryType::MEM_L0A,        MemoryType::MEM_L0B,        MemoryType::MEM_L0B,        MemoryType::MEM_L0C,
        MemoryType::MEM_L0C,        MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(G.AddTensors(dateType, shape, tensorMemoryType, tensorNames, 0), true);
    // add operation
    std::vector<Opcode> opCodes{Opcode::OP_L1_COPY_IN, Opcode::OP_L1_COPY_IN,  Opcode::OP_L1_TO_L0A,
                                Opcode::OP_L1_TO_L0B,  Opcode::OP_A_MUL_B,     Opcode::OP_L0C_TO_L1,
                                Opcode::OP_L1_TO_L0B,  Opcode::OP_L1_COPY_IN,  Opcode::OP_L1_TO_L0A,
                                Opcode::OP_A_MUL_B,    Opcode::OP_L0C_COPY_OUT};
    std::vector<std::string> opNames{"CopyIn1",  "CopyIn2", "L1ToL0A1", "L1ToL0B1", "Mul1",       "L0CToL11",
                                     "L1ToL0B2", "CopyIn3", "L1ToL0A2", "Mul2",     "L0CCopyOut1"};
    std::vector<std::vector<std::string>> iOperands{
        {"matA1DDR"}, {"matB1DDR"}, {"matA1L1"},  {"matB1L1"}, {"matA1L0A", "matB1L0B"},
        {"matC1L0C"}, {"matB2L1"},  {"matA2DDR"}, {"matA2L1"}, {"matA2L0A", "matB2L0B"},
        {"matC2L0C"}};
    std::vector<std::vector<std::string>> oOperands{{"matA1L1"},  {"matB1L1"},  {"matA1L0A"}, {"matB1L0B"},
                                                    {"matC1L0C"}, {"matB2L1"},  {"matB2L0B"}, {"matA2L1"},
                                                    {"matA2L0A"}, {"matC2L0C"}, {"outcast"}};
    EXPECT_EQ(G.AddOps(opCodes, iOperands, oOperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"matA1DDR", "matB1DDR", "matA2DDR"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
}

TEST_F(MergeSrcDstBufferTest, DircetReusePreMulL0BMemory)
{
    ComputationalGraphBuilder G;
    ConstructGrapgForReusePreMulMem(G);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    auto firstMulOp = G.GetOp("Mul1");
    auto secondMulOp = G.GetOp("Mul2");
    int srcMemId = -1;
    for (auto inputTensor : firstMulOp->GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            EXPECT_NE(inputTensor->memoryrange.memId, -1);
            srcMemId = inputTensor->memoryrange.memId;
        }
    }
    for (auto inputTensor : secondMulOp->GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            EXPECT_NE(inputTensor->memoryrange.memId, -1);
            EXPECT_EQ(inputTensor->memoryrange.memId, srcMemId);
        }
    }
}

void ConstructGrapgForUnReuseMultiCons(ComputationalGraphBuilder& G)
{
    // add tensor
    DataType dateType = DataType::DT_FP16;
    Shape shape = {16, 16};
    std::vector<std::string> tensorNames{"matA1DDR", "matA2DDR", "matA3DDR", "matB1DDR", "matA1L1",
                                         "matA2L1",  "matA3L1",  "matB1L1",  "matB2L1",  "matA1L0A",
                                         "matA2L0A", "matA3L0A", "matB1L0B", "matB2L0B", "matC1L0C",
                                         "matC2L0C", "matC3L0C", "outcast1", "outcast2"};
    std::vector<MemoryType> tensorMemoryType{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,
        MemoryType::MEM_L1,         MemoryType::MEM_L0A,        MemoryType::MEM_L0A,        MemoryType::MEM_L0A,
        MemoryType::MEM_L0B,        MemoryType::MEM_L0B,        MemoryType::MEM_L0C,        MemoryType::MEM_L0C,
        MemoryType::MEM_L0C,        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(G.AddTensors(dateType, shape, tensorMemoryType, tensorNames, 0), true);
    // add operation
    std::vector<Opcode> opCodes{Opcode::OP_L1_COPY_IN, Opcode::OP_L1_COPY_IN,   Opcode::OP_L1_TO_L0A,
                                Opcode::OP_L1_TO_L0B,  Opcode::OP_A_MUL_B,      Opcode::OP_L0C_TO_L1,
                                Opcode::OP_L1_TO_L0B,  Opcode::OP_L1_COPY_IN,   Opcode::OP_L1_TO_L0A,
                                Opcode::OP_A_MUL_B,    Opcode::OP_L0C_COPY_OUT, Opcode::OP_L1_COPY_IN,
                                Opcode::OP_L1_TO_L0A,  Opcode::OP_A_MUL_B,      Opcode::OP_L0C_COPY_OUT};
    std::vector<std::string> opNames{"CopyIn1",     "CopyIn2",  "L1ToL0A1", "L1ToL0B1", "Mul1",
                                     "L0CToL11",    "L1ToL0B2", "CopyIn3",  "L1ToL0A2", "Mul2",
                                     "L0CCopyOut1", "CopyIn4",  "L1ToL0A3", "Mul3",     "L0CCopyOut2"};
    std::vector<std::vector<std::string>> iOperands{{"matA1DDR"},
                                                    {"matB1DDR"},
                                                    {"matA1L1"},
                                                    {"matB1L1"},
                                                    {"matA1L0A", "matB1L0B"},
                                                    {"matC1L0C"},
                                                    {"matB2L1"},
                                                    {"matA2DDR"},
                                                    {"matA2L1"},
                                                    {"matA2L0A", "matB2L0B"},
                                                    {"matC2L0C"},
                                                    {"matA3DDR"},
                                                    {"matA3L1"},
                                                    {"matA3L0A", "matB1L0B"},
                                                    {"matC3L0C"}};
    std::vector<std::vector<std::string>> oOperands{
        {"matA1L1"},  {"matB1L1"},  {"matA1L0A"}, {"matB1L0B"}, {"matC1L0C"}, {"matB2L1"},  {"matB2L0B"}, {"matA2L1"},
        {"matA2L0A"}, {"matC2L0C"}, {"outcast1"}, {"matA3L1"},  {"matA3L0A"}, {"matC3L0C"}, {"outcast2"}};
    EXPECT_EQ(G.AddOps(opCodes, iOperands, oOperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"matA1DDR", "matB1DDR", "matA2DDR", "matA3DDR"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast1", "outcast2"}), true);
}

TEST_F(MergeSrcDstBufferTest, UnReusePreMulL0BMemoryForMultiConsumers)
{
    ComputationalGraphBuilder G;
    ConstructGrapgForUnReuseMultiCons(G);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    auto firstMulOp = G.GetOp("Mul1");
    auto secondMulOp = G.GetOp("Mul2");
    auto thirdMulOp = G.GetOp("Mul3");
    int srcMemId = -1;
    for (auto inputTensor : firstMulOp->GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            EXPECT_NE(inputTensor->memoryrange.memId, -1);
            srcMemId = inputTensor->memoryrange.memId;
        }
    }
    // 第一路和第三路matmul复用同一个L0B tensor进行计算
    for (auto inputTensor : thirdMulOp->GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            EXPECT_NE(inputTensor->memoryrange.memId, -1);
            EXPECT_EQ(inputTensor->memoryrange.memId, srcMemId);
        }
    }
    // 因为第一路matmul的L0B tensor有两个消费者 因此这里不进行复用
    for (auto inputTensor : secondMulOp->GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            EXPECT_NE(inputTensor->memoryrange.memId, -1);
            EXPECT_NE(inputTensor->memoryrange.memId, srcMemId);
        }
    }
}

void ConstructGraphForDisableReuseMultiPath(ComputationalGraphBuilder& G)
{
    // add tensor
    DataType dateType = DataType::DT_FP16;
    Shape shape = {16, 16};
    std::vector<std::string> tensorNames{"matA1DDR", "matA2DDR", "matA3DDR", "matB1DDR", "matA1L1",
                                         "matA2L1",  "matA3L1",  "matB1L1",  "matB2L1",  "matA1L0A",
                                         "matA2L0A", "matA3L0A", "matB1L0B", "matB2L0B", "matB3L0B",
                                         "matC1L0C", "matC2L0C", "matC3L0C", "outcast1", "outcast2"};
    std::vector<MemoryType> tensorMemoryType{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,
        MemoryType::MEM_L1,         MemoryType::MEM_L0A,        MemoryType::MEM_L0A,        MemoryType::MEM_L0A,
        MemoryType::MEM_L0B,        MemoryType::MEM_L0B,        MemoryType::MEM_L0B,        MemoryType::MEM_L0C,
        MemoryType::MEM_L0C,        MemoryType::MEM_L0C,        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(G.AddTensors(dateType, shape, tensorMemoryType, tensorNames, 0), true);
    // add operation
    std::vector<Opcode> opCodes{
        Opcode::OP_L1_COPY_IN, Opcode::OP_L1_COPY_IN, Opcode::OP_L1_TO_L0A,    Opcode::OP_L1_TO_L0B,
        Opcode::OP_A_MUL_B,    Opcode::OP_L0C_TO_L1,  Opcode::OP_L1_TO_L0B,    Opcode::OP_L1_COPY_IN,
        Opcode::OP_L1_TO_L0A,  Opcode::OP_A_MUL_B,    Opcode::OP_L0C_COPY_OUT, Opcode::OP_L1_COPY_IN,
        Opcode::OP_L1_TO_L0A,  Opcode::OP_L1_TO_L0B,  Opcode::OP_A_MUL_B,      Opcode::OP_L0C_COPY_OUT};
    std::vector<std::string> opNames{"CopyIn1",  "CopyIn2",  "L1ToL0A1", "L1ToL0B1",   "Mul1",        "L0CToL11",
                                     "L1ToL0B2", "CopyIn3",  "L1ToL0A2", "Mul2",       "L0CCopyOut1", "CopyIn4",
                                     "L1ToL0A3", "L1ToL0B3", "Mul3",     "L0CCopyOut2"};
    std::vector<std::vector<std::string>> iOperands{
        {"matA1DDR"}, {"matB1DDR"}, {"matA1L1"},  {"matB1L1"}, {"matA1L0A", "matB1L0B"},
        {"matC1L0C"}, {"matB2L1"},  {"matA2DDR"}, {"matA2L1"}, {"matA2L0A", "matB2L0B"},
        {"matC2L0C"}, {"matA3DDR"}, {"matA3L1"},  {"matB2L1"}, {"matA3L0A", "matB3L0B"},
        {"matC3L0C"}};
    std::vector<std::vector<std::string>> oOperands{
        {"matA1L1"},  {"matB1L1"},  {"matA1L0A"}, {"matB1L0B"}, {"matC1L0C"}, {"matB2L1"},  {"matB2L0B"}, {"matA2L1"},
        {"matA2L0A"}, {"matC2L0C"}, {"outcast1"}, {"matA3L1"},  {"matA3L0A"}, {"matB3L0B"}, {"matC3L0C"}, {"outcast2"}};
    EXPECT_EQ(G.AddOps(opCodes, iOperands, oOperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"matA1DDR", "matB1DDR", "matA2DDR", "matA3DDR"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast1", "outcast2"}), true);
}

TEST_F(MergeSrcDstBufferTest, DisableReuseWhenMultiSubPath)
{
    ComputationalGraphBuilder G;
    ConstructGraphForDisableReuseMultiPath(G);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    StubInputOutput(function);

    SrcDstBufferMerge mergePass;
    mergePass.RunOnFunction(*function);

    auto firstMulOp = G.GetOp("Mul1");
    auto secondMulOp = G.GetOp("Mul2");
    auto thirdMulOp = G.GetOp("Mul3");
    int firstMemId = -1;
    for (auto inputTensor : firstMulOp->GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            EXPECT_NE(inputTensor->memoryrange.memId, -1);
            firstMemId = inputTensor->memoryrange.memId;
        }
    }
    int secondMemId = -1;
    for (auto inputTensor : secondMulOp->GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            EXPECT_NE(inputTensor->memoryrange.memId, -1);
            secondMemId = inputTensor->memoryrange.memId;
        }
    }
    int thirdMemId = -1;
    for (auto inputTensor : thirdMulOp->GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            EXPECT_NE(inputTensor->memoryrange.memId, -1);
            thirdMemId = inputTensor->memoryrange.memId;
        }
    }
    EXPECT_TRUE(((firstMemId != secondMemId) && (firstMemId != thirdMemId)));
    EXPECT_NE(secondMemId, thirdMemId);
}

} // namespace tile_fwk
} // namespace npu
