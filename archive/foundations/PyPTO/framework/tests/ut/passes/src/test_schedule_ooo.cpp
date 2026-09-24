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
 * \file test_schedule_ooo.cpp
 * \brief Unit test for OoOSchedule.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_mgr/pass_manager.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/platform.h"
#include "tilefwk/tilefwk.h"
#define private public
#include "computational_graph_builder.h"
#include "passes/block_graph_pass/schedule_ooo/common/iso_matcher.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/buffer_rearrange.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/cluster_list_sort.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/core_assign.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/prior_dfs_sort.h"
#include "passes/block_graph_pass/schedule_ooo/schedule_ooo.h"
#include "passes/tile_graph_pass/graph_constraint/infer_dyn_shape.h"

namespace npu::tile_fwk {
constexpr int OOO_NUM2 = 2;
constexpr int OOO_NUM209 = 209;
constexpr int UBPoolSize = 192 * 1024;
std::unordered_map<Opcode, int> preNodePriority = {
    {Opcode::OP_UB_ALLOC, 0},
    {Opcode::OP_L1_ALLOC, 0},
    {Opcode::OP_L0A_ALLOC, 0},
    {Opcode::OP_L0B_ALLOC, 0},
    {Opcode::OP_L0C_ALLOC, 0},
    {Opcode::OP_BT_ALLOC, 0},
    {Opcode::OP_FIX_ALLOC, 0},
    {Opcode::OP_L1_TO_L0A, 1},
    {Opcode::OP_L1_TO_L0B, 1},
    {Opcode::OP_L1_TO_L0_AT, 1},
    {Opcode::OP_L1_TO_L0_BT, 1},
    {Opcode::OP_L1_TO_FIX, 1},
    {Opcode::OP_L1_TO_FIX_QUANT_PRE, 1},
    {Opcode::OP_L1_TO_FIX_RELU_PRE, 1},
    {Opcode::OP_L1_TO_FIX_RELU_POST, 1},
    {Opcode::OP_L1_TO_FIX_QUANT_POST, 1},
    {Opcode::OP_L1_TO_FIX_ELT_ANTIQ, 1},
    {Opcode::OP_L1_TO_FIX_MTE2_ANTIQ, 1},
    {Opcode::OP_L1_TO_BT, 1},
    {Opcode::OP_COPY_IN, 2},
    {Opcode::OP_UB_COPY_IN, 2},
    {Opcode::OP_L1_COPY_IN, 2},
    {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2},
    {Opcode::OP_L1_COPY_UB, 2},
    {Opcode::OP_L0C_COPY_UB, 2},
    {Opcode::OP_UB_COPY_L1, 2},
};

class ScheduleOoOTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }
    void TearDown() override {}
};

void SetTensorAttr(LogicalTensorPtr tensor, MemoryType memType, int memId)
{
    tensor->SetMemoryTypeOriginal(memType);
    tensor->SetMemoryTypeToBe(memType);
    tensor->memoryrange.memId = memId;
    tensor->UpdateDynValidShape({CreateTestScalarVar("S0"), CreateTestScalarVar("S1")});
}

void SetAllocAttr(Operation& alloc, int latency) { alloc.UpdateLatency(latency); }

LogicalTensorPtr CreateTensor(DataType dateType, std::vector<int64_t> shape, MemoryType memType, int memId)
{
    LogicalTensorPtr tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dateType, shape,
                                                                         CreateTestConstIntVector(shape));
    SetTensorAttr(tensor, memType, memId);
    return tensor;
}

Operation& CreateAllocOp(Function& currFunction, LogicalTensorPtr tensor, int latency)
{
    Operation& alloc = PassOperationUtils::AddOperation(currFunction, Opcode::OP_UB_ALLOC, {},
                                                        LogicalTensors({tensor}));
    SetAllocAttr(alloc, latency);
    return alloc;
}

Operation& CreateCopyOp(Function& currFunction, Opcode opcode, LogicalTensorPtr inTensor, LogicalTensorPtr outTensor,
                        std::vector<int64_t> shape)
{
    std::vector<int64_t> offset = {0, 0};
    auto& copy = PassOperationUtils::AddOperation(currFunction, opcode, LogicalTensors({inTensor}),
                                                  LogicalTensors({outTensor}));
    auto shapeImme = OpImmediate::Specified(shape);
    if (opcode == Opcode::OP_COPY_IN) {
        copy.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MEM_UB, shapeImme, shapeImme));
    }
    if (opcode == Opcode::OP_COPY_OUT) {
        copy.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified(offset), shapeImme, shapeImme));
    }
    return copy;
}

Operation& CreateAddOp(Function& currFunction, LogicalTensorPtr inTensor1, LogicalTensorPtr inTensor2,
                       LogicalTensorPtr outTensor)
{
    auto& add = PassOperationUtils::AddOperation(currFunction, Opcode::OP_ADD, LogicalTensors({inTensor1, inTensor2}),
                                                 LogicalTensors({outTensor}));
    return add;
}

void ReorderOperations(Function& function)
{
    auto opList = function.Operations().DuplicatedOpList();
    std::vector<Operation*> newOperations;
    for (auto& op : opList) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            newOperations.insert(newOperations.begin(), op);
        } else {
            newOperations.push_back(op);
        }
    }
    function.ScheduleBy(newOperations);
}

TEST_F(ScheduleOoOTest, TestMainScheduleOoO)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestOOO", "TestOOO", rootFuncPtr.get());
    currFunctionPtr->paramConfigs_.OoOPreScheduleMethod = "PriorDFS";
    auto emptyOpFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "", "", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    EXPECT_TRUE(emptyOpFunctionPtr != nullptr);
    currFunctionPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    emptyOpFunctionPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->rootFunc_->programs_.emplace(emptyOpFunctionPtr->GetFuncMagic(), emptyOpFunctionPtr.get());
    std::vector<int64_t> shape = {128, 128};
    auto shapeImme = OpImmediate::Specified(shape);

    auto tensor1 = CreateTensor(DataType::DT_FP32, shape, MEM_DEVICE_DDR, 0);
    auto tensor2 = CreateTensor(DataType::DT_FP32, shape, MEM_DEVICE_DDR, 1);
    auto tensor3 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 2);
    auto tensor4 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 3);
    auto tensor5 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 4);
    auto tensor6 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 5);
    auto tensor7 = CreateTensor(DataType::DT_FP32, shape, MEM_DEVICE_DDR, 6);
    auto tensor8 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 7);
    auto tensor9 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 8);
    auto& alloc1 = CreateAllocOp(*currFunctionPtr, tensor3, 1);
    auto& alloc2 = CreateAllocOp(*currFunctionPtr, tensor4, 1);
    auto& alloc3 = CreateAllocOp(*currFunctionPtr, tensor5, 1);
    auto& alloc4 = CreateAllocOp(*currFunctionPtr, tensor6, 1);
    auto& alloc5 = CreateAllocOp(*currFunctionPtr, tensor8, 1);
    auto& alloc6 = CreateAllocOp(*currFunctionPtr, tensor9, 1);
    auto& copyin1 = CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor1, tensor3, shape);
    auto& copyin2 = CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor2, tensor4, shape);
    auto& add1 = CreateAddOp(*currFunctionPtr, tensor3, tensor4, tensor5);
    auto& add2 = CreateAddOp(*currFunctionPtr, tensor3, tensor4, tensor6);
    auto& add3 = CreateAddOp(*currFunctionPtr, tensor6, tensor4, tensor8);
    auto& add4 = CreateAddOp(*currFunctionPtr, tensor8, tensor5, tensor9);
    (void)alloc1, (void)alloc2, (void)alloc3, (void)alloc4, (void)alloc5, (void)alloc6, (void)copyin1, (void)copyin2,
        (void)add1, (void)add2, (void)add3, (void)add4;
    for (auto& program : rootFuncPtr->rootFunc_->programs_) {
        ReorderOperations(*(program.second));
    }
    currFunctionPtr->EndFunction(nullptr);
    emptyOpFunctionPtr->EndFunction(nullptr);
    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.PreCheck(*rootFuncPtr), SUCCESS);
    oooSchedule.RunOnFunction(*rootFuncPtr);
    EXPECT_EQ(oooSchedule.PostCheck(*rootFuncPtr), SUCCESS);
}

static bool CheckSkipOps(std::vector<Operation*>& skipOps, Operation* op)
{
    for (auto skipOp : skipOps) {
        if (skipOp == op) {
            return true;
        }
    }
    return false;
}

TEST_F(ScheduleOoOTest, TestDependencies)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* op = subGraph.GetOp("RowMax1");
    EXPECT_NE(op, nullptr);
    EXPECT_EQ(ooOScheduler.state_.depManager.GetPredecessors(op).size(), 3);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(op).count(subGraph.GetOp("Alloc4")) > 0);
    EXPECT_EQ(ooOScheduler.state_.depManager.GetSuccessors(op).size(), 2);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(op).count(subGraph.GetOp("Add1")) > 0);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_VIEW,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}, {"t2"}, {"t2"}, {"t3", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t6"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Copyin1", "View1", "View2", "View3", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t3");
    tensor1->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t4");
    tensor2->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t5");
    tensor3->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;
    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* copyin = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin, nullptr);
    Operation* add = subGraph.GetOp("Add1");
    EXPECT_NE(add, nullptr);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(copyin).count(subGraph.GetOp("Alloc1")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(subGraph.GetOp("Alloc2")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(subGraph.GetOp("Copyin1")) > 0);
    EXPECT_TRUE(CheckSkipOps(ooOScheduler.GetSkipOps(add), subGraph.GetOp("View1")));
    EXPECT_TRUE(CheckSkipOps(ooOScheduler.GetSkipOps(add), subGraph.GetOp("View2")));
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesReshape)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_RESHAPE,
                                Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}, {"t3", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Copyin1", "Reshape1", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    std::shared_ptr<LogicalTensor> reshapeOut = subGraph.GetTensor("t3");
    reshapeOut->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);

    Operation* add = subGraph.GetOp("Add1");
    Operation* reshape = subGraph.GetOp("Reshape1");
    EXPECT_NE(add, nullptr);
    EXPECT_NE(reshape, nullptr);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(subGraph.GetOp("Copyin1")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(reshape) == 0);
    EXPECT_TRUE(CheckSkipOps(ooOScheduler.GetSkipOps(add), reshape));
    EXPECT_EQ(ooOScheduler.GetSkipOps(add).size(), 1UL);
}

// copyin -> t2 -> reshape1 -> t3 -+-> reshape2 -> t4 -> add(t4, t4)
//                                 +-> mul(t3, t3)
// t3 是链中间的张量且有旁支消费者 mul: 克隆范围必须覆盖到链尾 reshape2, 且只动 add 那一支。
TEST_F(ScheduleOoOTest, TestSpillClonesSharedReshapeChain)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_RESHAPE,  Opcode::OP_RESHAPE,  Opcode::OP_ADD,      Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4", "t4"}, {"t3", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t5"}, {"t6"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Copyin1", "Reshape1", "Reshape2", "Add1", "Mul1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    int spillMemId = subGraph.GetTensor("t2")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> midTensor = subGraph.GetTensor("t3");
    std::shared_ptr<LogicalTensor> tailOut = subGraph.GetTensor("t4");

    OoOScheduler ooOScheduler(*function);
    EXPECT_EQ(ooOScheduler.Init(function->Operations().DuplicatedOpList()), SUCCESS);

    Operation* reshape1 = subGraph.GetOp("Reshape1");
    Operation* reshape2 = subGraph.GetOp("Reshape2");
    Operation* add = subGraph.GetOp("Add1");
    Operation* mul = subGraph.GetOp("Mul1");
    LogicalTensorPtr reshape1InBefore = reshape1->GetInputOperand(0);

    midTensor->memoryrange.memId = spillMemId;
    tailOut->memoryrange.memId = spillMemId;
    EXPECT_EQ(ooOScheduler.GetSkipOps(add).size(), 2UL);

    LogicalTensorPtr reloadTensor = subGraph.GetTensor("t2")->Clone(*function, true);
    reloadTensor->memoryrange.memId = spillMemId + 1000;
    EXPECT_EQ(ooOScheduler.spillEngine_.UpdateOperationInput(add, subGraph.GetOp("Copyin1"), reloadTensor, spillMemId),
              SUCCESS);

    EXPECT_EQ(reshape1->GetInputOperand(0), reshape1InBefore);
    EXPECT_EQ(reshape2->GetInputOperand(0), midTensor);
    EXPECT_EQ(mul->GetIOperands()[0], midTensor);
    EXPECT_EQ(midTensor->memoryrange.memId, spillMemId);
    EXPECT_NE(add->GetIOperands()[0], tailOut);
    EXPECT_EQ(add->GetIOperands()[0]->memoryrange.memId, reloadTensor->memoryrange.memId);
    EXPECT_EQ(add->GetIOperands()[1]->memoryrange.memId, reloadTensor->memoryrange.memId);
    EXPECT_EQ(add->GetIOperands()[0]->Datatype(), tailOut->Datatype());
    EXPECT_EQ(add->GetIOperands()[0]->Format(), tailOut->Format());
    EXPECT_EQ(add->GetIOperands()[0]->GetShape(), tailOut->GetShape());
    EXPECT_EQ(add->GetIOperands()[0]->GetRawTensor()->rawshape, tailOut->GetRawTensor()->rawshape);

    // add 两个 operand 都改指到克隆链, 原链整条从 add 自己的 skipOps 摘掉;
    // 链首 reshape1 还有旁支 mul 读着所以不标删, 由 mul 的 skipOps 带走。
    auto& skipOps = ooOScheduler.GetSkipOps(add);
    EXPECT_EQ(skipOps.size(), 4UL);
    EXPECT_FALSE(CheckSkipOps(skipOps, reshape1));
    EXPECT_FALSE(CheckSkipOps(skipOps, reshape2));
    EXPECT_TRUE(CheckSkipOps(ooOScheduler.GetSkipOps(mul), reshape1));
    EXPECT_FALSE(reshape1->IsDeleted());
    EXPECT_TRUE(reshape2->IsDeleted());

    Operation* cloneTail = *add->GetIOperands()[0]->GetProducers().begin();
    Operation* cloneHead = *cloneTail->GetInputOperand(0)->GetProducers().begin();
    EXPECT_NE(cloneTail, reshape2);
    EXPECT_NE(cloneHead, reshape1);
    EXPECT_TRUE(CheckSkipOps(skipOps, cloneTail));
    EXPECT_TRUE(CheckSkipOps(skipOps, cloneHead));
    EXPECT_EQ(cloneHead->GetInputOperand(0), reloadTensor);
    EXPECT_EQ(cloneTail->GetOutputOperand(0)->memoryrange.memId, reloadTensor->memoryrange.memId);
    // 克隆体持有私有 RawTensor, 改它不会连坐原链。
    EXPECT_NE(cloneTail->GetOutputOperand(0)->GetRawTensor(), tailOut->GetRawTensor());
    EXPECT_TRUE(tailOut->GetConsumers().empty());
    // 被 spill 的 t2 不受连坐。
    EXPECT_EQ(subGraph.GetTensor("t2")->memoryrange.memId, spillMemId);
}

TEST_F(ScheduleOoOTest, TestDependenciesAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_SUB,      Opcode::OP_SUB,      Opcode::OP_SUB,
                                Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t4"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t7"}, {"t7"}, {"t8"}};
    std::vector<std::string> opNames{"Alloc1", "Sub1", "Sub2", "Sub3", "Assemble1", "Assemble2", "Assemble3", "Mul1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    for (size_t i = 3; i < tensorNames.size() - 1; i++) {
        EXPECT_NE(subGraph.GetTensor(tensorNames[i]), nullptr);
        std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor(tensorNames[i]);
        tensor->memoryrange.memId = subGraph.GetTensor("t4")->memoryrange.memId;
    }

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* alloc = subGraph.GetOp("Alloc1");
    EXPECT_NE(alloc, nullptr);
    Operation* sub = subGraph.GetOp("Sub1");
    EXPECT_NE(sub, nullptr);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(alloc).count(subGraph.GetOp("Sub3")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(sub).count(subGraph.GetOp("Alloc1")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(sub).count(subGraph.GetOp("Assemble1")) > 0);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* add = subGraph.GetOp("Add1");
    EXPECT_NE(add, nullptr);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(add).count(subGraph.GetOp("Copyout1")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(subGraph.GetOp("Copyin1")) > 0);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesTrue)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Copyin1", "Alloc1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 1,
                ooOScheduler.state_.orderedOps.end());
    res = ooOScheduler.state_.depManager.InitDependencies(ooOScheduler.state_.orderedOps, false);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillCopyIn)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{},     {},           {},           {},           {},    {"t1"},
                                                    {"t2"}, {"t3", "t4"}, {"t3", "t5"}, {"t4", "t6"}, {"t8"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.orderedOps[8]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.orderedOps[9]->GetOpcodeStr(), "COPY_IN");
}

TEST_F(ScheduleOoOTest, TestSpill)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Add4",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.orderedOps[8]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.orderedOps[13]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.orderedOps[14]->GetOpcodeStr(), "COPY_IN");
}

TEST_F(ScheduleOoOTest, TestSpillInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},           {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"},  {"t9"}, {"t5"},
                                                    {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 6,
                ooOScheduler.state_.orderedOps.begin() + 11);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    Operation* add1 = subGraph.GetOp("Add1");
    EXPECT_NE(add1, nullptr);
    Operation* add3 = subGraph.GetOp("Add3");
    EXPECT_NE(add3, nullptr);
    EXPECT_EQ((*ooOScheduler.state_.depManager.GetSuccessors(add1).begin())->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.depManager.GetPredecessors(add3).size(), 3);
}

TEST_F(ScheduleOoOTest, TestSpillMultiTensor)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {"t1"},
                                                    {"t2"},
                                                    {"t3"},
                                                    {"t4"},
                                                    {"t7", "t8"},
                                                    {"t5", "t6", "t9"},
                                                    {"t7", "t8", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"},
                                                    {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4",  "Alloc5", "Alloc6", "Alloc7",
                                     "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {50, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->shape = {128, 128};
    tensor1->tensor->rawshape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->shape = {128, 128};
    tensor2->tensor->rawshape = {128, 128};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.orderedOps.size(), 23);
}

TEST_F(ScheduleOoOTest, TestSpillView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},     {"t1"},
                                                    {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "View1",  "View2",  "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3",  "t4",  "t5",  "t6",  "t7",
                                         "t8", "t9", "t10", "t11", "t12", "t13", "t14"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_SUB,
                                Opcode::OP_SUB,      Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE,
                                Opcode::OP_ASSEMBLE, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},      {},      {},      {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"},  {"t5"},  {"t6"},  {"t7"},
                                                    {"t8"}, {"t9"}, {"t10"}, {"t11"}, {"t12"}, {"t13"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"},  {"t6"},  {"t7"},  {"t8"},  {"t12"}, {"t5"},
                                                    {"t6"},  {"t7"},  {"t8"},  {"t9"},  {"t10"}, {"t11"},
                                                    {"t12"}, {"t13"}, {"t13"}, {"t13"}, {"t13"}, {"t14"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",    "Alloc3",    "Alloc4",    "Alloc5",    "Copyin1",
                                     "Copyin2", "Copyin3",   "Copyin4",   "Add1",      "Add2",      "Sub1",
                                     "Sub2",    "Assemble1", "Assemble2", "Assemble3", "Assemble4", "Mul1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor0 = subGraph.GetTensor("t9");
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor0->tensor->rawshape = {128, 192};
    tensor1->tensor->rawshape = {128, 192};
    tensor1->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor2->shape = {128, 128};
    tensor2->tensor->rawshape = {128, 192};
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t12");
    tensor3->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor3->shape = {128, 128};
    tensor3->tensor->rawshape = {128, 192};
    std::shared_ptr<LogicalTensor> tensor4 = subGraph.GetTensor("t13");
    tensor4->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor4->shape = {128, 192};
    tensor4->tensor->rawshape = {128, 192};
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("t14");
    tensor5->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor5->shape = {128, 192};
    tensor5->tensor->rawshape = {128, 192};
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("t7");
    tensor6->shape = {128, 128};
    tensor6->tensor->rawshape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor7 = subGraph.GetTensor("t8");
    tensor7->shape = {128, 128};
    tensor7->tensor->rawshape = {128, 128};
    std::vector<int64_t> offset1 = {0, 0};
    std::vector<int64_t> offset2 = {0, 128};
    std::vector<int64_t> offset3 = {64, 0};
    std::vector<int64_t> offset4 = {64, 128};
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1);
    auto assemble1 = subGraph.GetOp("Assemble1");
    assemble1->SetOpAttribute(assembleAttr);
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset2);
    auto assemble2 = subGraph.GetOp("Assemble2");
    assemble2->SetOpAttribute(assembleAttr2);
    auto assembleAttr3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset3);
    auto assemble3 = subGraph.GetOp("Assemble3");
    assemble3->SetOpAttribute(assembleAttr3);
    auto assembleAttr4 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset4);
    auto assemble4 = subGraph.GetOp("Assemble4");
    assemble4->SetOpAttribute(assembleAttr4);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSchedule)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Add4",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* add = subGraph.GetOp("Add2");
    EXPECT_NE(add, nullptr);
    EXPECT_EQ(add->oOperand[0]->memoryrange.start, 32768);
    EXPECT_EQ(add->oOperand[0]->memoryrange.end, 49152);
}

TEST_F(ScheduleOoOTest, TestScheduleInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},           {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"},  {"t9"}, {"t5"},
                                                    {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* copyin = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin, nullptr);
    Operation* add1 = subGraph.GetOp("Add1");
    EXPECT_NE(add1, nullptr);
    Operation* add3 = subGraph.GetOp("Add3");
    EXPECT_NE(add3, nullptr);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(add1->oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(add1->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(add3->oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(add3->oOperand[0]->memoryrange.end, 65536);
}

TEST_F(ScheduleOoOTest, TestScheduleView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},     {"t1"},
                                                    {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "View1",  "View2",  "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    tensor1->shape = {32, 32};
    tensor1->tensor->rawshape = {64, 64};
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    tensor2->shape = {32, 32};
    tensor2->tensor->rawshape = {64, 64};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* copyin = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin, nullptr);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(subGraph.GetOp("View1")->GetOutputOperand(0)->memoryrange.start, 0);
    EXPECT_EQ(subGraph.GetOp("View1")->GetOutputOperand(0)->memoryrange.end, 16384);
    EXPECT_EQ(subGraph.GetOp("View2")->GetOutputOperand(0)->memoryrange.start, 0);
    EXPECT_EQ(subGraph.GetOp("View2")->GetOutputOperand(0)->memoryrange.end, 16384);
}

TEST_F(ScheduleOoOTest, TestScheduleAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD,
                                Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},           {"t1"},       {"t2"},
                                                    {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t7"}, {"t8"}, {"t10"}, {"t11"}, {"t5"}, {"t6"},
                                                    {"t7"}, {"t8"}, {"t9"}, {"t9"},  {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",    "Alloc4",    "Alloc5", "Copyin1", "Copyin2",
                                     "Copyin3", "Copyin4", "Assemble1", "Assemble2", "Add1",   "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    tensor2->shape = {32, 32};
    tensor2->tensor->rawshape = {64, 64};
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t5");
    tensor3->shape = {32, 32};
    tensor3->tensor->rawshape = {64, 64};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* copyin1 = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin1, nullptr);
    Operation* copyin2 = subGraph.GetOp("Copyin2");
    EXPECT_NE(copyin2, nullptr);
    Operation* assemble1 = subGraph.GetOp("Assemble1");
    EXPECT_NE(assemble1, nullptr);
    Operation* assemble2 = subGraph.GetOp("Assemble2");
    EXPECT_NE(assemble2, nullptr);
    EXPECT_EQ(copyin1->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin1->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(copyin2->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin2->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(assemble1->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(assemble1->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(assemble2->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(assemble2->oOperand[0]->memoryrange.end, 16384);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillCopyIn)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{},     {},           {},           {},           {},    {"t1"},
                                                    {"t2"}, {"t3", "t4"}, {"t3", "t5"}, {"t4", "t6"}, {"t8"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[10]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.state_.newOperations[10]->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(ooOScheduler.state_.newOperations[10]->oOperand[0]->memoryrange.end, 65536);
}

TEST_F(ScheduleOoOTest, TestScheduleSpill)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Add4",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.end, 131072);
    EXPECT_EQ(ooOScheduler.state_.newOperations[15]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.state_.newOperations[15]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[15]->oOperand[0]->memoryrange.end, 131072);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},           {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"},  {"t9"}, {"t5"},
                                                    {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 6,
                ooOScheduler.state_.orderedOps.begin() + 11);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.newOperations[6]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.end, 131072);
    EXPECT_EQ(ooOScheduler.state_.newOperations[14]->GetOpcodeStr(), "ADD");
    EXPECT_EQ(ooOScheduler.state_.newOperations[14]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[14]->oOperand[0]->memoryrange.end, 131072);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},     {"t1"},
                                                    {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "View1",  "View2",  "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD,
                                Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},           {"t1"},       {"t2"},
                                                    {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t7"}, {"t8"}, {"t10"}, {"t11"}, {"t5"}, {"t6"},
                                                    {"t7"}, {"t8"}, {"t9"}, {"t9"},  {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",    "Alloc4",    "Alloc5", "Copyin1", "Copyin2",
                                     "Copyin3", "Copyin4", "Assemble1", "Assemble2", "Add1",   "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("t5");
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("t6");
    tensor5->shape = {64, 128};
    tensor5->tensor->rawshape = {128, 128};
    tensor6->shape = {64, 128};
    tensor6->tensor->rawshape = {128, 128};
    std::vector<int64_t> offset1 = {0, 0};
    std::vector<int64_t> offset2 = {64, 0};
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1);
    auto assemble1 = subGraph.GetOp("Assemble1");
    assemble1->SetOpAttribute(assembleAttr1);
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset2);
    auto assemble2 = subGraph.GetOp("Assemble2");
    assemble2->SetOpAttribute(assembleAttr2);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillMultiProducerBuffer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"DDR1", "DDR2", "DDR3", "UB1",  "UB2",  "UB3",
                                         "UB4",  "L1_1", "L0C1", "L0C2", "L1_2", "L0C3"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_L1,
        MemoryType::MEM_L0C,        MemoryType::MEM_L0C,        MemoryType::MEM_L1,         MemoryType::MEM_L0C};
    std::vector<Opcode> opCodes{
        Opcode::OP_COPY_IN,    Opcode::OP_COPY_IN,    Opcode::OP_UB_COPY_ND2NZ, Opcode::OP_UB_COPY_ND2NZ,
        Opcode::OP_UB_COPY_L1, Opcode::OP_UB_COPY_L1, Opcode::OP_A_MUL_B,       Opcode::OP_A_MUL_B,
        Opcode::OP_COPY_IN,    Opcode::OP_A_MUL_B,    Opcode::OP_UB_ALLOC,      Opcode::OP_UB_ALLOC,
        Opcode::OP_UB_ALLOC,   Opcode::OP_UB_ALLOC,   Opcode::OP_L1_ALLOC,      Opcode::OP_L1_ALLOC,
        Opcode::OP_L0C_ALLOC,  Opcode::OP_L0C_ALLOC,  Opcode::OP_L0C_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{"DDR1"}, {"DDR2"}, {"UB1"},  {"UB2"}, {"UB3"}, {"UB4"}, {"L1_1"},
                                                    {"L1_1"}, {"DDR3"}, {"L1_2"}, {},      {},      {},      {},
                                                    {},       {},       {},       {},      {}};
    std::vector<std::vector<std::string>> ooperands{
        {"UB1"}, {"UB2"}, {"UB3"}, {"UB4"}, {"L1_1"}, {"L1_1"}, {"L0C1"}, {"L0C2"}, {"L1_2"}, {"L0C3"},
        {"UB1"}, {"UB2"}, {"UB3"}, {"UB4"}, {"L1_1"}, {"L1_2"}, {"L0C1"}, {"L0C2"}, {"L0C3"}};
    std::vector<std::string> opNames{"copyin1",  "copyin2",   "ubNd2nz1",  "ubNd2nz2", "ub_l11",
                                     "ub_l12",   "aMulB1",    "aMulB2",    "copyin3",  "aMulB3",
                                     "ubAlloc1", "ubAlloc2",  "ubAlloc3",  "ubAlloc4", "l1Alloc1",
                                     "l1Alloc2", "l0cAlloc1", "l0cAlloc2", "l0cAlloc3"};
    subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0);
    subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("L1_1");
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("L1_2");
    tensor5->shape = {256, 300};
    tensor5->tensor->rawshape = {256, 300};
    tensor6->shape = {256, 300};
    tensor6->tensor->rawshape = {256, 300};
    auto* ubToL11 = subGraph.GetOp("ub_l11");
    ubToL11->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    auto* ubToL12 = subGraph.GetOp("ub_l12");
    ubToL12->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({64, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    OoOScheduler ooOScheduler(*function);
    rotate(sort.operations.begin() + 12, sort.operations.begin() + 15, sort.operations.begin() + 18);
    EXPECT_EQ(ooOScheduler.Init(sort.operations), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    EXPECT_EQ(ooOScheduler.SeqSchedule(), SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillMultiProducerBufferNotReady)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"DDR1", "DDR2", "DDR3", "UB1",  "UB2", "UB3",
                                         "UB4",  "L1_1", "L0C1", "L0C2", "L1_2"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_L1,
        MemoryType::MEM_L0C,        MemoryType::MEM_L0C,        MemoryType::MEM_L1};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,    Opcode::OP_UB_COPY_ND2NZ,
                                Opcode::OP_UB_COPY_ND2NZ, Opcode::OP_UB_COPY_L1, Opcode::OP_UB_COPY_L1,
                                Opcode::OP_A_MUL_B,       Opcode::OP_A_MUL_B,    Opcode::OP_L1_COPY_UB,
                                Opcode::OP_UB_ALLOC,      Opcode::OP_UB_ALLOC,   Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC,      Opcode::OP_L1_ALLOC,   Opcode::OP_L1_ALLOC,
                                Opcode::OP_L0C_ALLOC,     Opcode::OP_L0C_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{"DDR1"}, {"DDR2"}, {"UB1"},  {"UB2"}, {"UB3"}, {"UB4"},
                                                    {"L1_1"}, {"L1_1"}, {"L1_2"}, {},      {},      {},
                                                    {},       {},       {},       {},      {}};
    std::vector<std::vector<std::string>> ooperands{{"UB1"},  {"L1_2"}, {"UB3"},  {"UB4"},  {"L1_1"}, {"L1_1"},
                                                    {"L0C1"}, {"L0C2"}, {"UB2"},  {"UB1"},  {"UB2"},  {"UB3"},
                                                    {"UB4"},  {"L1_1"}, {"L1_2"}, {"L0C1"}, {"L0C2"}};
    std::vector<std::string> opNames{"copyin1",  "copyin2",  "ubNd2nz1", "ubNd2nz2",  "ub_l11",   "ub_l12",
                                     "aMulB1",   "aMulB2",   "l1CopyUb", "ubAlloc1",  "ubAlloc2", "ubAlloc3",
                                     "ubAlloc4", "l1Alloc1", "l1Alloc2", "l0cAlloc1", "l0cAlloc2"};
    subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0);
    subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("L1_1");
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("L1_2");
    tensor5->shape = {256, 300};
    tensor5->tensor->rawshape = {256, 300};
    tensor6->shape = {256, 300};
    tensor6->tensor->rawshape = {256, 300};
    auto* ubToL11 = subGraph.GetOp("ub_l11");
    ubToL11->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    auto* ubToL12 = subGraph.GetOp("ub_l12");
    ubToL12->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({64, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    OoOScheduler ooOScheduler(*function);
    rotate(sort.operations.begin(), sort.operations.begin() + 8, sort.operations.begin() + 14);
    rotate(sort.operations.begin() + 4, sort.operations.begin() + 12, sort.operations.begin() + 13);
    EXPECT_EQ(ooOScheduler.Init(sort.operations), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    EXPECT_EQ(ooOScheduler.SeqSchedule(), SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillL0CMultiConsumer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::tuple<DataType, MemoryType, std::string>> tensors{
        {DT_FP32, MEM_L0A, "L0A"},          {DT_FP32, MEM_L0B, "L0B"},  {DT_FP32, MEM_L0C, "L0C1"},
        {DT_FP32, MEM_L0C, "L0C2"},         {DT_FP16, MEM_UB, "UBDst"}, {DT_FP32, MEM_L1, "L1Dst"},
        {DT_FP32, MEM_DEVICE_DDR, "DDROut"}};
    for (auto& [dt, mem, name] : tensors) {
        subGraph.AddTensor(dt, {128, 128}, mem, name, 0);
    }
    std::vector<Opcode> opCodes{Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC,
                                Opcode::OP_A_MUL_B,   Opcode::OP_UB_ALLOC,  Opcode::OP_L0C_COPY_UB,
                                Opcode::OP_L1_ALLOC,  Opcode::OP_L0C_TO_L1, Opcode::OP_L0C_COPY_OUT,
                                Opcode::OP_L0C_ALLOC, Opcode::OP_A_MUL_B};
    std::vector<std::vector<std::string>> ioperands{{},       {},       {}, {"L0A", "L0B"}, {}, {"L0C1"}, {},
                                                    {"L0C1"}, {"L0C1"}, {}, {"L0A", "L0B"}};
    std::vector<std::vector<std::string>> ooperands{{"L0A"},   {"L0B"},   {"L0C1"},   {"L0C1"}, {"UBDst"}, {"UBDst"},
                                                    {"L1Dst"}, {"L1Dst"}, {"DDROut"}, {"L0C2"}, {"L0C2"}};
    std::vector<std::string> opNames{"L0AAlloc", "L0BAlloc", "L0CAlloc1",  "Matmul1",   "UBAlloc", "L0CCopyUB",
                                     "L1Alloc",  "L0CToL1",  "L0CCopyOut", "L0CAlloc2", "Matmul2"};
    subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true);
    Function* function = subGraph.GetFunction();

    auto shapeImme = OpImmediate::Specified(std::vector<int64_t>{128, 128});
    auto* copyOutOp = subGraph.GetOp("L0CCopyOut");
    copyOutOp->SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(std::vector<int64_t>{0, 0}), shapeImme, shapeImme));

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    OoOScheduler ooOScheduler(*function);
    EXPECT_EQ(ooOScheduler.Init(sort.operations), SUCCESS);

    ooOScheduler.state_.bufferManagerMap[CoreLocationType::AIC][MemoryType::MEM_L0C] = BufferPool(MemoryType::MEM_L0C,
                                                                                                  64 * 1024);
    ooOScheduler.state_.orderedOps = {subGraph.GetOp("L0AAlloc"),
                                      subGraph.GetOp("L0BAlloc"),
                                      subGraph.GetOp("L0CAlloc1"),
                                      subGraph.GetOp("Matmul1"),
                                      copyOutOp,
                                      subGraph.GetOp("L0CAlloc2"),
                                      subGraph.GetOp("Matmul2"),
                                      subGraph.GetOp("UBAlloc"),
                                      subGraph.GetOp("L0CCopyUB"),
                                      subGraph.GetOp("L1Alloc"),
                                      subGraph.GetOp("L0CToL1")};
    for (size_t i = 0; i < ooOScheduler.state_.orderedOps.size(); i++) {
        ooOScheduler.state_.schedInfoMap[ooOScheduler.state_.orderedOps[i]].execOrder = static_cast<int>(i);
    }

    EXPECT_EQ(ooOScheduler.SeqSchedule(), SUCCESS);

    auto spillCopyOut = std::count_if(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.end(),
                                      [](Operation* op) { return op->GetOpcodeStr() == "COPY_OUT"; });
    EXPECT_EQ(spillCopyOut, 2);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillWithInplaceView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}, {"t11"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"},  {"t8"}, {"t9"},  {"t5"}, {"t6"},
                                                    {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}, {"t12"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2",
                                     "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3",   "View1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t12");
    tensor3->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    std::vector<int64_t> offset1 = {0, 0};
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UB);
    auto view1 = subGraph.GetOp("View1");
    view1->SetOpAttribute(viewAttr1);
    auto inputTensor = view1->GetIOperands()[0];
    auto outputTensor = view1->GetOOperands()[0];
    EXPECT_EQ(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 6,
                ooOScheduler.state_.orderedOps.begin() + 11);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    // The tensor's memId remains the same before and after the view operation following a spill.
    EXPECT_EQ(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillFragFailed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->shape = {32, 32};
    tensor->tensor->rawshape = {32, 32};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestEmptyOplist)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation*> scheduleOpList;
    OoOScheduler ooOScheduler(function);
    Status res = ooOScheduler.Schedule(scheduleOpList);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestScheduleReshape)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_RESHAPE};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Reshape1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSingleCopyin1)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Copyin1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestSingleCopyin2)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestDelBufCount)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    OoOScheduler oooSchedule(function);
    oooSchedule.state_.DelBufRefCount(-1);
}

TEST_F(ScheduleOoOTest, TestDelBufCount_1)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    OoOScheduler oooSchedule(function);
    oooSchedule.state_.bufRefCount[1] = -1;
    oooSchedule.state_.DelBufRefCount(1);
}

TEST_F(ScheduleOoOTest, TestGetSpillTensor)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation*> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    auto& alloc1 = PassOperationUtils::AddOperation(function, Opcode::OP_UB_ALLOC, {}, {tensor3});
    alloc1.UpdateLatency(1);

    OoOScheduler oooSchedule(function);
    LogicalTensorPtr tensor = oooSchedule.spillEngine_.GetSpillTensor(&alloc1, 1);
    EXPECT_EQ(tensor, nullptr);
}

TEST_F(ScheduleOoOTest, TestCheckAllocIssue)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation*> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    std::shared_ptr<LogicalTensor> tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 1;

    auto& alloc1 = PassOperationUtils::AddOperation(function, Opcode::OP_UB_ALLOC, {}, {tensor3, tensor2});
    alloc1.UpdateLatency(1);

    OoOScheduler oooSchedule(function);
    auto opList = function.Operations().DuplicatedOpList();
    oooSchedule.Init(opList);
}

TEST_F(ScheduleOoOTest, TestBufferUsage)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    OoOScheduleStatistic testCheck;
    ooOScheduler.AddObserver(&testCheck);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    std::unordered_map<MemoryType, uint64_t> invalidBufferTotalUsage = {{MemoryType::MEM_UB, 0},
                                                                        {MemoryType::MEM_L1, 0},
                                                                        {MemoryType::MEM_L0A, 0},
                                                                        {MemoryType::MEM_L0B, 0},
                                                                        {MemoryType::MEM_L0C, 0}};
    std::unordered_map<MemoryType, uint64_t> invalidBufferMaxUsage = {{MemoryType::MEM_UB, 0},
                                                                      {MemoryType::MEM_L1, 0},
                                                                      {MemoryType::MEM_L0A, 0},
                                                                      {MemoryType::MEM_L0B, 0},
                                                                      {MemoryType::MEM_L0C, 0}};
    EXPECT_NE(testCheck.bufferTotalUsage, invalidBufferTotalUsage);
    EXPECT_NE(testCheck.bufferMaxUsage, invalidBufferMaxUsage);

    testCheck.clock = 3; // 模拟数据
    res = testCheck.HealthCheckOoOSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_NE(testCheck.report, nullptr);
}

TEST_F(ScheduleOoOTest, TestScheduleGenSpillInfiniteLoop)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_SUB,
                                Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},
                                                    {"t1"}, {"t2"}, {"t3"}, {"t4", "t5"}, {"t3", "t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"},
                                                    {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3", "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "Sub1",   "Add1",   "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP16, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->shape = {80, 128};
    tensor->tensor->rawshape = {80, 128};

    EXPECT_NE(subGraph.GetTensor("t4"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t4");
    tensor1->shape = {176, 256};
    tensor1->tensor->rawshape = {176, 256};

    EXPECT_NE(subGraph.GetTensor("t5"), nullptr);
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t5");
    tensor2->shape = {176, 256};
    tensor2->tensor->rawshape = {176, 256};

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t6");
    tensor3->shape = {64, 128};
    tensor3->tensor->rawshape = {64, 128};

    EXPECT_NE(subGraph.GetTensor("t7"), nullptr);
    std::shared_ptr<LogicalTensor> tensor4 = subGraph.GetTensor("t7");
    tensor4->shape = {16, 16};
    tensor4->tensor->rawshape = {16, 16};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestCheckOpBufferSize)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {144, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestInitLocalBufferFailed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {62, 69}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestCheckAllocBufferSize)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestOoORollbackMix)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"T3",  "T1",   "T6",   "T8",   "T11",  "T13",  "T16",  "T18",
                                         "T21", "DDR1", "DDR2", "DDR3", "DDR4", "DDR5", "DDR6", "DDR7"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,
        MemoryType::MEM_L1,         MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<std::string> tensorNames_L0AB{"T4", "T2", "T7", "T9", "T12", "T14", "T19", "T17"};
    std::vector<MemoryType> tensorMemTypes_L0AB{MemoryType::MEM_L0B, MemoryType::MEM_L0A, MemoryType::MEM_L0A,
                                                MemoryType::MEM_L0B, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
                                                MemoryType::MEM_L0B, MemoryType::MEM_L0A};
    std::vector<std::string> tensorNames_L0C{"T5", "T10", "T15", "T20"};
    std::vector<MemoryType> tensorMemTypes_L0C{MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C,
                                               MemoryType::MEM_L0C};
    std::vector<Opcode> opCodes{
        Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,
        Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L0A_ALLOC, Opcode::OP_L1_ALLOC,
        Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC,
        Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN,   Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A,
        Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L0C_TO_L1,
        Opcode::OP_L0C_TO_L1, Opcode::OP_L0C_TO_L1, Opcode::OP_A_MUL_B,   Opcode::OP_A_MUL_B,   Opcode::OP_A_MUL_B,
        Opcode::OP_COPY_OUT,  Opcode::OP_A_MULACC_B};
    std::vector<std::vector<std::string>> inputoperands{{},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {"DDR2"},     {"DDR1"},
                                                        {"DDR3"},     {"DDR4"},
                                                        {"DDR5"},     {"DDR6"},
                                                        {"T1"},       {"T6"},
                                                        {"T11"},      {"T16"},
                                                        {"T3"},       {"T8"},
                                                        {"T13"},      {"T18"},
                                                        {"T5"},       {"T15"},
                                                        {"T20"},      {"T2", "T4"},
                                                        {"T7", "T9"}, {"T12", "T14"},
                                                        {"T21"},      {"T10", "T17", "T19"}};
    std::vector<std::vector<std::string>> outputoperands{
        {"T1"},  {"T3"},  {"T6"},  {"T8"},  {"T11"}, {"T13"}, {"T16"}, {"T18"},  {"T2"},  {"T21"}, {"T7"},
        {"T12"}, {"T17"}, {"T4"},  {"T9"},  {"T14"}, {"T19"}, {"T5"},  {"T10"},  {"T15"}, {"T3"},  {"T1"},
        {"T8"},  {"T11"}, {"T13"}, {"T18"}, {"T2"},  {"T7"},  {"T12"}, {"T17"},  {"T4"},  {"T9"},  {"T14"},
        {"T19"}, {"T6"},  {"T16"}, {"T21"}, {"T5"},  {"T10"}, {"T15"}, {"DDR7"}, {"T20"}};
    std::vector<std::string> operationNames{
        "L1_Alloc1",      "L1_Alloc2",      "L1_Alloc3",      "L1_Alloc4",      "L1_Alloc5",      "L1_Alloc6",
        "L1_Alloc7",      "L1_Alloc8",      "L0A_Alloc1",     "L1_Alloc9",      "L0A_Alloc2",     "L0A_Alloc3",
        "L0A_Alloc4",     "L0B_Alloc1",     "L0B_Alloc2",     "L0B_Alloc3",     "L0B_Alloc4",     "L0C_Alloc1",
        "L0C_Alloc2",     "L0C_Alloc3",     "Copyin2",        "Copyin1",        "Copyin3",        "Copyin4",
        "Copyin5",        "Copyin6",        "OP_L1_TO_L0A_1", "OP_L1_TO_L0A_2", "OP_L1_TO_L0A_3", "OP_L1_TO_L0A_4",
        "OP_L1_TO_L0B_1", "OP_L1_TO_L0B_2", "OP_L1_TO_L0B_3", "OP_L1_TO_L0B_4", "OP_L0C_TO_L1_1", "OP_L0C_TO_L1_2",
        "OP_L0C_TO_L1_3", "OP_A_MUL_B_1",   "OP_A_MUL_B_2",   "OP_A_MUL_B_3",   "Copyout",        "OP_A_MULACC_B"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes_L0AB, tensorNames_L0AB, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 256}, tensorMemTypes_L0C, tensorNames_L0C, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, inputoperands, outputoperands, operationNames, true), true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("T10");
    tensor->memoryrange.memId = subGraph.GetTensor("T20")->memoryrange.memId;
    EXPECT_NE(function, nullptr);

    OptimizeSort optimizeSort(function->Operations().DuplicatedOpList(), *function);
    Status res = optimizeSort.SortOps();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestHasEnoughBuffer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}};
    std::vector<std::vector<std::string>> ooperands{{"t1", "t2"}};
    std::vector<std::string> opNames{"Alloc1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto op = subGraph.GetOp("Alloc1");
    auto tensor1 = subGraph.GetTensor("t1");
    int memId = tensor1->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    ooOScheduler.state_.orderedOps.push_back(op);
    ooOScheduler.state_.schedInfoMap[op].isAlloc = true;
    ooOScheduler.state_.depManager.GetSuccessors(op).clear();
    ooOScheduler.state_.opReqMemIdsMap[op] = {memId};
    ooOScheduler.SetCoreLocation(op, CoreLocationType::AIV0);
    EXPECT_EQ(ooOScheduler.state_.InitLocalBuffer(tensor1, memId), SUCCESS);
    ooOScheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB] = BufferPool(MemoryType::MEM_UB,
                                                                                                  0);
    bool res = ooOScheduler.HasEnoughBuffer(op, MemoryType::MEM_UB);
    EXPECT_EQ(res, false);
}

TEST_F(ScheduleOoOTest, TestHasEnoughBufferAddMemId)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t1"}};
    std::vector<std::string> opNames{"Alloc1", "COPY_IN"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto op = subGraph.GetOp("Alloc1");
    auto opCopyIn = subGraph.GetOp("COPY_IN");
    auto tensor1 = subGraph.GetTensor("t1");
    auto tensor2 = subGraph.GetTensor("t2");
    int memId1 = tensor1->memoryrange.memId;
    op->GetOutputOperand(0)->ClearAllProducers();
    op->GetOutputOperand(0)->AddProducer(*opCopyIn);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.state_.orderedOps.push_back(op);
    ooOScheduler.state_.orderedOps.push_back(opCopyIn);
    ooOScheduler.state_.schedInfoMap[op].isAlloc = true;
    ooOScheduler.state_.depManager.InsertSuccessor(op, opCopyIn);
    ooOScheduler.state_.opReqMemIdsMap[opCopyIn] = {1};
    ooOScheduler.state_.opReqMemIdsMap[op] = {memId1};
    ooOScheduler.SetCoreLocation(op, CoreLocationType::AIV0);
    EXPECT_EQ(ooOScheduler.state_.InitLocalBuffer(tensor1, memId1), SUCCESS);
    ooOScheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB] = BufferPool(MemoryType::MEM_UB,
                                                                                                  0);
    EXPECT_EQ(ooOScheduler.state_.InitLocalBuffer(tensor2, 1), SUCCESS);
    bool res = ooOScheduler.HasEnoughBuffer(op, MemoryType::MEM_UB);
    EXPECT_EQ(res, false);
}

TEST_F(ScheduleOoOTest, TestCoreAssign)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"};
    std::vector<Opcode> opCodes{Opcode::OP_A_MUL_B, Opcode::OP_ADDS,      Opcode::OP_ADDS, Opcode::OP_ADDS,
                                Opcode::OP_ADDS,    Opcode::OP_A_MUL_B,   Opcode::OP_ADDS, Opcode::OP_A_MUL_B,
                                Opcode::OP_ADD,     Opcode::OP_A_MULACC_B};
    std::vector<std::vector<std::string>> ioperands{
        {"t0", "t0"}, {"t1"}, {"t1"},       {"t1"},       {"t2", "t2"},
        {"t2"},       {"t4"}, {"t5", "t5"}, {"t3", "t6"}, {"t7", "t8", "t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t10"}};
    std::vector<std::string> opNames{"op1", "op2", "op3", "op4", "op5", "op6", "op7", "op8", "op9", "op10"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorNames), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto opList = function->Operations(false).DuplicatedOpList();
    TaskSplitter splitter;
    splitter.SplitGraph(opList);
    CoreScheduler coreScheduler;
    coreScheduler.Schedule(splitter.GetTaskGraph());
    const int taskNum = 10;
    EXPECT_EQ(splitter.GetTaskGraph().tasks.size(), taskNum);
    splitter.MergeTask();
    OoOScheduler ooOScheduler(*function);
    splitter.MarkInternalSubgraphID();
    EXPECT_EQ(splitter.GetMergedOperations().size(), opList.size());
}

TEST_F(ScheduleOoOTest, TestOooScopeMerge)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_A_MUL_B, Opcode::OP_ADDS, Opcode::OP_ADDS, Opcode::OP_EXP};
    std::vector<std::vector<std::string>> ioperands{{"t0", "t0"}, {"t1"}, {"t2"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"op1", "op2", "op3", "op4"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorNames), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    auto op2 = subGraph.GetOp("op2");
    auto op3 = subGraph.GetOp("op3");
    auto op4 = subGraph.GetOp("op4");
    ASSERT_NE(op2, nullptr);
    ASSERT_NE(op3, nullptr);
    ASSERT_NE(op4, nullptr);
    op2->SetOooScopeId(1);
    op4->SetOooScopeId(1);
    int magic2 = op2->GetOpMagic();
    int magic3 = op3->GetOpMagic();
    int magic4 = op4->GetOpMagic();

    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto opList = function->Operations(false).DuplicatedOpList();
    EXPECT_EQ(opList.size(), 4U);

    TaskSplitter splitter;
    splitter.SplitGraph(opList);
    auto& tasks = splitter.GetTaskGraph().tasks;

    bool foundMerged23 = false;
    bool foundMerged24 = false;
    for (auto& task : tasks) {
        bool hasOp2 = false, hasOp3 = false, hasOp4 = false;
        for (auto* op : task.opList_) {
            if (op->GetOpMagic() == magic2)
                hasOp2 = true;
            if (op->GetOpMagic() == magic3)
                hasOp3 = true;
            if (op->GetOpMagic() == magic4)
                hasOp4 = true;
        }
        if (hasOp2 && hasOp3) {
            foundMerged23 = true;
        }
        if (hasOp2 && hasOp4) {
            foundMerged24 = true;
        }
    }
    EXPECT_TRUE(!foundMerged23) << "op2 and op3 should not be merged into one task by ooo_scope";
    EXPECT_TRUE(foundMerged24) << "op2 and op4 should be merged into one task by ooo_scope";
}

TEST_F(ScheduleOoOTest, TestBufferPollRearrange)
{
    BufferPool pool;
    pool.memSize_ = UBPoolSize;
    BufferSlice s1(32768, 65536);
    BufferSlice s2(98304, 98304);
    pool.bufferSlices[1] = s1;
    pool.bufferSlices[2] = s2;
    EXPECT_FALSE(pool.CheckBufferSlicesOverlap());

    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto alloc1 = subGraph.GetOp("Alloc1");
    auto alloc2 = subGraph.GetOp("Alloc2");
    auto alloc3 = subGraph.GetOp("Alloc3");

    OoOScheduler oooSchedule(*function);
    auto corePair = CoreLocationType::AIV0;
    oooSchedule.state_.schedInfoMap[alloc3].coreLocation = corePair;
    oooSchedule.state_.bufferManagerMap[corePair][MemoryType::MEM_UB] = pool;
    oooSchedule.state_.tensorOccupyMap[1] = alloc1;
    oooSchedule.state_.tensorOccupyMap[2] = alloc2;

    oooSchedule.state_.localBufferMap[1] = std::make_shared<LocalBuffer>(1, 65536, MemoryType::MEM_UB);
    oooSchedule.state_.localBufferMap[2] = std::make_shared<LocalBuffer>(2, 98304, MemoryType::MEM_UB);
    EXPECT_EQ(oooSchedule.RearrangeBuffer(alloc3, MemoryType::MEM_UB), SUCCESS);
    auto& ubPool = oooSchedule.state_.bufferManagerMap[corePair][MemoryType::MEM_UB];
    EXPECT_EQ(ubPool.GetBufferSize(1), 65536);
    EXPECT_EQ(ubPool.GetBufferSize(2), 98304);
    EXPECT_EQ(ubPool.GetBufferOffset(1), 98304);
    EXPECT_EQ(ubPool.GetBufferOffset(2), 0);
}

TEST_F(ScheduleOoOTest, TestBufferPoolMakeBufferSliceAlreadyAlloc)
{
    BufferPool pool(MemoryType::MEM_UB, 1024);
    auto tensor = std::make_shared<LocalBuffer>(1, 64, MemoryType::MEM_UB);
    BufferSlice slice1(0, 64);
    EXPECT_EQ(pool.MakeBufferSlice(tensor, slice1), SUCCESS);
    BufferSlice slice2(128, 64);
    EXPECT_EQ(pool.MakeBufferSlice(tensor, slice2), FAILED);
}

TEST_F(ScheduleOoOTest, TestBufferPoolAllocateNoFreeSpace)
{
    BufferPool pool(MemoryType::MEM_UB, 256);
    auto tensor1 = std::make_shared<LocalBuffer>(1, 256, MemoryType::MEM_UB);
    EXPECT_EQ(pool.Allocate(tensor1), SUCCESS);
    auto tensor2 = std::make_shared<LocalBuffer>(2, 64, MemoryType::MEM_UB);
    EXPECT_EQ(pool.Allocate(tensor2), FAILED);
}

TEST_F(ScheduleOoOTest, TestBufferRearrangeSingleBubble)
{
    BufferPool pool(MemoryType::MEM_UB, 100);
    auto tensor = std::make_shared<LocalBuffer>(1, 50, MemoryType::MEM_UB);
    BufferSlice s1(0, 50);
    EXPECT_EQ(pool.MakeBufferSlice(tensor, s1), SUCCESS);
    RearrangeScheme scheme = GetRearrangeScheme(pool, 50);
    EXPECT_EQ(scheme.cost, static_cast<size_t>(INT_MAX));
}

TEST_F(ScheduleOoOTest, TestSchedulerAllocTensorMemRangeNonViewOp)
{
    ComputationalGraphBuilder subGraph;
    subGraph.AddTensor(DataType::DT_FP32, {8, 8}, "a");
    subGraph.AddTensor(DataType::DT_FP32, {8, 8}, "b");
    subGraph.AddTensor(DataType::DT_FP32, {8, 8}, "c");
    subGraph.AddOp(Opcode::OP_ADD, {"a", "b"}, {"c"}, "add1");
    Function* function = subGraph.GetFunction();
    OoOScheduler oooSchedule(*function);
    auto addOp = subGraph.GetOp("add1");
    oooSchedule.GetSkipOps(addOp).push_back(addOp);
    EXPECT_EQ(oooSchedule.AllocTensorMemRange(addOp), FAILED);
}

TEST_F(ScheduleOoOTest, TestSpillOnBlockFailedAtL0)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_L0A, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
                                           MemoryType::MEM_L0B};
    std::vector<Opcode> opCodes{Opcode::OP_L1_TO_L0A, Opcode::OP_L0A_ALLOC, Opcode::OP_L1_TO_L0B, Opcode::OP_L0B_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"L1toL0A", "AllocL0A", "L1toL0B", "AllocL0B"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP16, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto L1toL0A = subGraph.GetOp("L1toL0A");
    auto L1toL0B = subGraph.GetOp("L1toL0B");
    auto AllocL0A = subGraph.GetOp("AllocL0A");
    auto AllocL0B = subGraph.GetOp("AllocL0B");
    OoOScheduler oooSchedule(*function);
    oooSchedule.state_.SetOpMemIds(AllocL0A, {3});
    oooSchedule.state_.SetOpMemIds(AllocL0B, {4});
    auto corePair = CoreLocationType::AIC;
    oooSchedule.state_.allocIssueQueue[corePair][MemoryType::MEM_L0A].Insert(AllocL0A);
    oooSchedule.state_.allocIssueQueue[corePair][MemoryType::MEM_L0B].Insert(AllocL0B);
    oooSchedule.state_.tensorOccupyMap.emplace(1, L1toL0A);
    oooSchedule.state_.tensorOccupyMap.emplace(2, L1toL0B);
    oooSchedule.state_.localBufferMap[1] = std::make_shared<LocalBuffer>(1, 32768, MemoryType::MEM_L0A);
    oooSchedule.state_.localBufferMap[2] = std::make_shared<LocalBuffer>(2, 32768, MemoryType::MEM_L0B);
    oooSchedule.state_.localBufferMap[3] = std::make_shared<LocalBuffer>(3, 32768, MemoryType::MEM_L0A);
    oooSchedule.state_.localBufferMap[4] = std::make_shared<LocalBuffer>(4, 32768, MemoryType::MEM_L0B);
    oooSchedule.state_.localBufferMap[1]->start = 512;
    oooSchedule.state_.localBufferMap[1]->end = 33280;
    oooSchedule.state_.localBufferMap[2]->start = 512;
    oooSchedule.state_.localBufferMap[2]->end = 33280;
    EXPECT_EQ(oooSchedule.SpillOnBlock(), FAILED);
}

TEST_F(ScheduleOoOTest, TestOoO1C2V)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1",   "t3",   "t6",   "t7",   "t8",  "t10",
                                         "DDR1", "DDR2", "DDR3", "DDR4", "t11", "t12"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<std::string> tensorNames_L0{"t2", "t4", "t5"};
    std::vector<MemoryType> tensorMemTypes_L0AB{MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C};

    std::vector<Opcode> opCodes{
        Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC,  Opcode::OP_L0A_ALLOC,  Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,  Opcode::OP_UB_ALLOC,   Opcode::OP_UB_ALLOC,  Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN,  Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0B,  Opcode::OP_A_MUL_B,   Opcode::OP_L0C_COPY_UB,
        Opcode::OP_ADDS,     Opcode::OP_COPY_OUT,  Opcode::OP_L1_COPY_UB, Opcode::OP_ADDS,      Opcode::OP_COPY_OUT,
        Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,  Opcode::OP_ADDS,       Opcode::OP_UB_COPY_L1};
    std::vector<std::vector<std::string>> ioperands{
        {},     {},           {},     {},     {},     {},     {},     {},      {}, {"DDR1"}, {"DDR2"}, {"t1"},
        {"t3"}, {"t2", "t4"}, {"t5"}, {"t6"}, {"t7"}, {"t3"}, {"t8"}, {"t10"}, {}, {},       {"t11"},  {"t12"}};
    std::vector<std::vector<std::string>> ooperands{
        {"t1"}, {"t3"}, {"t2"}, {"t4"}, {"t5"},   {"t6"}, {"t7"},  {"t8"},   {"t10"}, {"t11"}, {"t3"},  {"t2"},
        {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"DDR3"}, {"t8"}, {"t10"}, {"DDR4"}, {"t11"}, {"t12"}, {"t12"}, {"t1"}};
    std::vector<std::string> opNames{"L1_Alloc1", "L1_Alloc2", "L0A_Alloc1",  "L0B_Alloc1", "L0C_Alloc1", "UB_Alloc1",
                                     "UB_Alloc2", "UB_Alloc3", "UB_Alloc4",   "COPY_IN1",   "COPY_IN2",   "L1_TO_L0A",
                                     "L1_TO_L0B", "A_MUL_B",   "L0C_COPY_UB", "ADDS1",      "COPY_OUT1",  "L1_COPY_UB",
                                     "ADDS2",     "COPY_OUT2", "UB_Alloc5",   "UB_Alloc6",  "ADDS3",      "UB_COPY_L1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes_L0AB, tensorNames_L0, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    subGraph.GetOp("ADDS1")->SetAttribute(OpAttributeKey::isCube, false);
    Function* function = subGraph.GetFunction();
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    auto op1 = subGraph.GetOp("ADDS3");
    auto op2 = subGraph.GetOp("ADDS2");
    auto op3 = subGraph.GetOp("ADDS1");
    auto op4 = subGraph.GetOp("L1_TO_L0A");
    OoOSchedule oooSchedule;
    std::pair<uint64_t, Function*> functionPair = std::make_pair(0, function);
    int64_t size = 0;
    std::vector<Operation*> opList = function->Operations().DuplicatedOpList();
    auto res = oooSchedule.Schedule(opList, *function, functionPair, size);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(op1->GetInternalSubgraphID(), 1);
    EXPECT_EQ(op2->GetInternalSubgraphID(), 2);
    EXPECT_EQ(op3->GetInternalSubgraphID(), 1);
    EXPECT_EQ(op4->GetInternalSubgraphID(), 0);
}

void SetInternalSubgraphIDAndAIVCore(Operation* op, int id)
{
    op->UpdateInternalSubgraphID(id);
    if (id == 0) {
        op->SetAIVCore(AIVCore::AIV0);
    }
}

void SetAttribute(ComputationalGraphBuilder& subGraph, OoOScheduler& oooSchedule, Operation*& ubCopyL1,
                  Operation*& alloc3)
{
    Operation* adds = subGraph.GetOp("ADDS");
    ubCopyL1 = subGraph.GetOp("UB_COPY_L1");
    Operation* copyin1 = subGraph.GetOp("COPY_IN1");
    Operation* copyin2 = subGraph.GetOp("COPY_IN2");
    Operation* copyin3 = subGraph.GetOp("COPY_IN3");
    Operation* copyin4 = subGraph.GetOp("COPY_IN4");
    Operation* copyout1 = subGraph.GetOp("COPY_OUT1");
    Operation* copyout2 = subGraph.GetOp("COPY_OUT2");

    Operation* alloc1 = subGraph.GetOp("UB_Alloc2");
    Operation* alloc2 = subGraph.GetOp("L1_Alloc1");
    alloc3 = subGraph.GetOp("L1_Alloc3");
    Operation* alloc4 = subGraph.GetOp("L1_Alloc2");

    Operation* alloc5 = subGraph.GetOp("UB_Alloc1");
    Operation* alloc6 = subGraph.GetOp("L0A_Alloc1");
    Operation* alloc7 = subGraph.GetOp("L0A_Alloc2");

    SetInternalSubgraphIDAndAIVCore(adds, 0);
    SetInternalSubgraphIDAndAIVCore(alloc5, 0);
    SetInternalSubgraphIDAndAIVCore(alloc1, 0);
    SetInternalSubgraphIDAndAIVCore(ubCopyL1, 0);

    SetInternalSubgraphIDAndAIVCore(alloc2, 1);
    SetInternalSubgraphIDAndAIVCore(alloc3, 1);
    SetInternalSubgraphIDAndAIVCore(alloc4, 1);
    SetInternalSubgraphIDAndAIVCore(alloc6, 1);
    SetInternalSubgraphIDAndAIVCore(alloc7, 1);
    SetInternalSubgraphIDAndAIVCore(copyin1, 1);
    SetInternalSubgraphIDAndAIVCore(copyin2, 1);
    SetInternalSubgraphIDAndAIVCore(copyin3, 1);
    SetInternalSubgraphIDAndAIVCore(copyin4, 1);
    SetInternalSubgraphIDAndAIVCore(copyout1, 1);
    SetInternalSubgraphIDAndAIVCore(copyout2, 1);

    oooSchedule.SetIsRetired(alloc5, true);
    oooSchedule.SetIsRetired(adds, true);
    oooSchedule.SetIsRetired(alloc1, true);
    oooSchedule.SetIsRetired(ubCopyL1, true);
    oooSchedule.SetIsRetired(alloc2, true);
    oooSchedule.SetIsRetired(alloc7, true);
    oooSchedule.SetIsRetired(copyin2, true);

    auto localBuffer1 = oooSchedule.state_.localBufferMap[0];
    auto coreAIC = CoreLocationType::AIC;
    oooSchedule.state_.bufferManagerMap[coreAIC][MemoryType::MEM_L1].Allocate(localBuffer1);
    oooSchedule.state_.tensorOccupyMap.emplace(0, copyin2);
}

TEST_F(ScheduleOoOTest, TensorMemTypeMismatch)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestMemTypeMismatch", "TestMemTypeMismatch",
                                           nullptr);
    std::vector<int64_t> shape = {16, 16};

    auto t = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    t->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    t->SetMemoryTypeToBe(MemoryType::MEM_UB); // 不一�?

    PassOperationUtils::AddOperation(*func, Opcode::OP_NOP, {t}, {t});
    func->inCasts_.push_back(t);

    OoOScheduleChecker checker;
    bool ok = checker.PreCheckTensorInfo(t);
    EXPECT_FALSE(ok);
}

TEST_F(ScheduleOoOTest, TensorMemIdInvalid)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestMemIdInvalid", "TestMemIdInvalid", nullptr);
    std::vector<int64_t> shape = {16, 16};

    auto t = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    t->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    t->SetMemoryTypeToBe(MemoryType::MEM_UB);
    t->memoryrange.memId = -1; // 非法

    OoOScheduleChecker checker;
    bool ok = checker.PreCheckTensorInfo(t);
    EXPECT_FALSE(ok);
}

TEST_F(ScheduleOoOTest, CallOpNotAllowed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames = {"t1", "t2"};
    std::vector<MemoryType> memTypes = {MEM_DEVICE_DDR, MEM_DEVICE_DDR};
    subGraph.AddTensors(DT_FP32, {16, 16}, memTypes, tensorNames, 0);

    std::vector<Opcode> opCodes = {Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ins = {{"t1"}};
    std::vector<std::vector<std::string>> outs = {{"t2"}};
    std::vector<std::string> opNames = {"CALL_OP"};
    subGraph.AddOps(opCodes, ins, outs, opNames, true);

    Function* function = subGraph.GetFunction();
    OoOScheduleChecker checker;
    bool ret = checker.PreCheckOpInfo(function->Operations().DuplicatedOpList()[0]);
    EXPECT_FALSE(ret);
}

TEST_F(ScheduleOoOTest, ViewMemIdMismatch)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "ViewMemIdMismatch", "ViewMemIdMismatch", nullptr);
    std::vector<int64_t> shape = {16, 16};
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    inTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    inTensor->memoryrange.memId = 0;
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    outTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    outTensor->memoryrange.memId = 1;
    PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {inTensor}, {outTensor});
    OoOScheduleChecker checker;
    bool ret = checker.PreCheckOpInfo(func->Operations().DuplicatedOpList()[0]);
    EXPECT_FALSE(ret);
}

namespace dualdst_ut {

constexpr int64_t TILE_M = 64;
constexpr int64_t TILE_N = 64;
constexpr size_t SMALL_UB_POOL = 8 * 1024;

void SetCopyL0cToUbAttr(Operation& op, const std::vector<int64_t>& fromOff, const std::vector<int64_t>& tileShape)
{
    auto fromOffImme = OpImmediate::Specified(fromOff);
    auto shapeImme = OpImmediate::Specified(tileShape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(fromOffImme, MemoryType::MEM_UB, shapeImme, shapeImme));
}

void InjectStaticValidShape(Operation& op, const std::vector<int64_t>& vals)
{
    op.SetAttribute(OpAttributeKey::staticValidShape, vals);
}

struct DualDstGraph {
    std::shared_ptr<ComputationalGraphBuilder> builder;
    Function* func{nullptr};
    Operation* allocL0c{nullptr};
    Operation* allocUb0{nullptr};
    Operation* allocUb1{nullptr};
    Operation* allocOut0{nullptr};
    Operation* allocOut1{nullptr};
    Operation* copyinL0c{nullptr};
    Operation* copy0{nullptr};
    Operation* copy1{nullptr};
    Operation* add0{nullptr};
    Operation* add1{nullptr};
};

DualDstGraph BuildDualDstGraph(const std::vector<int64_t>& l0cShape, const std::vector<int64_t>& tileShape,
                               const std::vector<int64_t>& fromOff0, const std::vector<int64_t>& fromOff1)
{
    DualDstGraph g;
    g.builder = std::make_shared<ComputationalGraphBuilder>();
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, l0cShape, MemoryType::MEM_L0C, "t_l0c"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_ub0"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_ub1"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_out0"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_out1"), true);

    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_ALLOC, {}, {"t_l0c"}, "alloc_l0c"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub0"}, "alloc_ub0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub1"}, "alloc_ub1"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_out0"}, "alloc_out0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_out1"}, "alloc_out1"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_COPY_UB, {"t_l0c"}, {"t_ub0"}, "copy0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_COPY_UB, {"t_l0c"}, {"t_ub1"}, "copy1"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_ADD, {"t_ub0", "t_ub0"}, {"t_out0"}, "add0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_ADD, {"t_ub1", "t_ub1"}, {"t_out1"}, "add1"), true);

    g.func = g.builder->GetFunction();
    g.allocL0c = g.builder->GetOp("alloc_l0c");
    g.allocUb0 = g.builder->GetOp("alloc_ub0");
    g.allocUb1 = g.builder->GetOp("alloc_ub1");
    g.allocOut0 = g.builder->GetOp("alloc_out0");
    g.allocOut1 = g.builder->GetOp("alloc_out1");
    g.copy0 = g.builder->GetOp("copy0");
    g.copy1 = g.builder->GetOp("copy1");
    g.add0 = g.builder->GetOp("add0");
    g.add1 = g.builder->GetOp("add1");

    SetCopyL0cToUbAttr(*g.copy0, fromOff0, tileShape);
    SetCopyL0cToUbAttr(*g.copy1, fromOff1, tileShape);
    return g;
}

DualDstGraph BuildDualDstGraph_2(const std::vector<int64_t>& l0cShape, const std::vector<int64_t>& tileShape,
                                 const std::vector<int64_t>& fromOff0, const std::vector<int64_t>& fromOff1)
{
    DualDstGraph g;
    g.builder = std::make_shared<ComputationalGraphBuilder>();
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, l0cShape, MemoryType::MEM_DEVICE_DDR, "t_ddr"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, l0cShape, MemoryType::MEM_L0C, "t_l0c"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_ub0"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_ub1"), true);
    for (int side = 0; side < 2; ++side) {
        for (int idx = 0; idx < 5; ++idx) {
            EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB,
                                           "t_softmax" + std::to_string(side) + "_out" + std::to_string(idx)),
                      true);
        }
    }

    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_ALLOC, {}, {"t_l0c"}, "alloc_l0c"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub0"}, "alloc_ub0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub1"}, "alloc_ub1"), true);
    for (int side = 0; side < 2; ++side) {
        for (int idx = 0; idx < 5; ++idx) {
            EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {},
                                       {"t_softmax" + std::to_string(side) + "_out" + std::to_string(idx)},
                                       "alloc_softmax" + std::to_string(side) + "_out" + std::to_string(idx)),
                      true);
        }
    }
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_COPY_IN, {"t_ddr"}, {"t_l0c"}, "copyin_l0c"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_COPY_UB, {"t_l0c"}, {"t_ub0"}, "copy0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_COPY_UB, {"t_l0c"}, {"t_ub1"}, "copy1"), true);
    EXPECT_EQ(g.builder->AddOp(
                  Opcode::OP_ONLINE_SOFTMAX, {"t_ub0"},
                  {"t_softmax0_out0", "t_softmax0_out1", "t_softmax0_out2", "t_softmax0_out3", "t_softmax0_out4"},
                  "softmax0"),
              true);
    EXPECT_EQ(g.builder->AddOp(
                  Opcode::OP_ONLINE_SOFTMAX, {"t_ub1"},
                  {"t_softmax1_out0", "t_softmax1_out1", "t_softmax1_out2", "t_softmax1_out3", "t_softmax1_out4"},
                  "softmax1"),
              true);

    g.func = g.builder->GetFunction();
    g.allocL0c = g.builder->GetOp("alloc_l0c");
    g.allocUb0 = g.builder->GetOp("alloc_ub0");
    g.allocUb1 = g.builder->GetOp("alloc_ub1");
    g.allocOut0 = g.builder->GetOp("alloc_softmax0_out0");
    g.allocOut1 = g.builder->GetOp("alloc_softmax1_out0");
    g.copyinL0c = g.builder->GetOp("copyin_l0c");
    g.copy0 = g.builder->GetOp("copy0");
    g.copy1 = g.builder->GetOp("copy1");
    g.add0 = g.builder->GetOp("softmax0");
    g.add1 = g.builder->GetOp("softmax1");

    SetCopyL0cToUbAttr(*g.copy0, fromOff0, tileShape);
    SetCopyL0cToUbAttr(*g.copy1, fromOff1, tileShape);
    auto copyinOffImme = OpImmediate::Specified(std::vector<int64_t>{0, 0});
    auto copyinShapeImme = OpImmediate::Specified(l0cShape);
    g.copyinL0c->SetOpAttribute(
        std::make_shared<CopyOpAttribute>(copyinOffImme, MemoryType::MEM_L0C, copyinShapeImme, copyinShapeImme));
    return g;
}

void InjectCoreMap(OoOScheduler& s, const DualDstGraph& g, bool sameCoreForAdds = false)
{
    s.state_.schedInfoMap[g.copy0].coreLocation = CoreLocationType::AIC;
    s.state_.schedInfoMap[g.copy1].coreLocation = CoreLocationType::AIC;
    s.state_.schedInfoMap[g.add0].coreLocation = CoreLocationType::AIV0;
    s.state_.schedInfoMap[g.add1].coreLocation = sameCoreForAdds ? CoreLocationType::AIV0 : CoreLocationType::AIV1;
    s.state_.schedInfoMap[g.allocL0c].coreLocation = CoreLocationType::AIC;
    if (g.copyinL0c != nullptr) {
        s.state_.schedInfoMap[g.copyinL0c].coreLocation = CoreLocationType::AIC;
    }
    s.state_.schedInfoMap[g.allocUb0].coreLocation = CoreLocationType::AIV0;
    s.state_.schedInfoMap[g.allocUb1].coreLocation = CoreLocationType::AIV1;
    s.state_.schedInfoMap[g.allocOut0].coreLocation = CoreLocationType::AIV0;
    s.state_.schedInfoMap[g.allocOut1].coreLocation = CoreLocationType::AIV1;
}

void UpdateCopyDynValidShape(DualDstGraph& g)
{
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar(TILE_M), SymbolicScalar(TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar(TILE_M), SymbolicScalar(TILE_N)});
}

DualDstGraph BuildOnlineSoftmaxDualDstGraph()
{
    auto g = BuildDualDstGraph_2({TILE_M, TILE_N * 2}, {TILE_M, TILE_N}, {0, 0}, {0, TILE_N});
    UpdateCopyDynValidShape(g);
    return g;
}

bool TopoSortInPlace(DependencyManager& depManager, std::vector<Operation*>& ops)
{
    std::vector<Operation*> sorted;
    sorted.reserve(ops.size());
    std::unordered_set<Operation*> placed;
    while (sorted.size() < ops.size()) {
        bool progressed = false;
        for (auto* op : ops) {
            if (placed.count(op) != 0) {
                continue;
            }
            bool ready = true;
            for (auto* pred : depManager.GetPredecessors(op)) {
                if (pred != nullptr && placed.count(pred) == 0 &&
                    std::find(ops.begin(), ops.end(), pred) != ops.end()) {
                    ready = false;
                    break;
                }
            }
            if (!ready) {
                continue;
            }
            sorted.push_back(op);
            placed.insert(op);
            progressed = true;
        }
        if (!progressed) {
            return false;
        }
    }
    ops = sorted;
    return true;
}

Status InitDualDstScheduler(OoOScheduler& s, const DualDstGraph& g, bool enableGuard = false)
{
    Status st = s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO);
    if (st != SUCCESS)
        return st;
    InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    if (enableGuard)
        s.SetEnableDualDstAllocGuard(true);
    return SUCCESS;
}

bool HasAiv0AndAiv1Tasks(TaskSplitter& splitter)
{
    bool hasAiv0Task = false;
    bool hasAiv1Task = false;
    for (const auto& task : splitter.GetTaskGraph().tasks) {
        hasAiv0Task = hasAiv0Task || task.targetCoreType == TargetCoreType::AIV0;
        hasAiv1Task = hasAiv1Task || task.targetCoreType == TargetCoreType::AIV1;
    }
    return hasAiv0Task && hasAiv1Task;
}

bool HasIsoAllocPair(const std::unordered_map<Operation*, Operation*>& pairs, Operation* lhs, Operation* rhs)
{
    auto lhsIt = pairs.find(lhs);
    if (lhsIt != pairs.end() && lhsIt->second == rhs)
        return true;
    auto rhsIt = pairs.find(rhs);
    return rhsIt != pairs.end() && rhsIt->second == lhs;
}

Operation* FindDualDstOp(Function& func)
{
    for (auto& op : func.Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST)
            return &op;
    }
    return nullptr;
}

bool HasDualDstOp(const std::vector<Operation*>& ops)
{
    return std::any_of(ops.begin(), ops.end(), [](Operation* op) {
        return op != nullptr && op->GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST;
    });
}

Operation* FindUbAllocPred(OoOScheduler& s, Operation* op)
{
    for (auto* pred : s.state_.depManager.GetPredecessors(op)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos)
            return pred;
    }
    return nullptr;
}

Operation* AddGuardedUbAlloc(Function& func, OoOScheduler& s, int memId, CoreLocationType core, int64_t execOrder)
{
    auto tensor = CreateTensor(DataType::DT_FP32, {TILE_M, TILE_N}, MemoryType::MEM_UB, memId);
    Operation* alloc = &PassOperationUtils::AddOperation(func, Opcode::OP_UB_ALLOC, {}, LogicalTensors({tensor}));
    s.state_.schedInfoMap[alloc].isAlloc = true;
    s.state_.schedInfoMap[alloc].isRetired = false;
    s.state_.schedInfoMap[alloc].coreLocation = core;
    s.state_.schedInfoMap[alloc].execOrder = execOrder;
    s.state_.SetOpMemIds(alloc, {tensor->memoryrange.memId});
    return alloc;
}

Status FillAivPoolsWithPlaceholderBuffers(OoOScheduler& s, const DualDstGraph& g, size_t needSize, int memIdA,
                                          int memIdB)
{
    auto& poolA = s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& poolB = s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    if (poolA.GetMemSize() < needSize || poolB.GetMemSize() < needSize)
        return FAILED;
    auto bufHolderA = std::make_shared<LocalBuffer>(memIdA, needSize, MemoryType::MEM_UB);
    auto bufHolderB = std::make_shared<LocalBuffer>(memIdB, needSize, MemoryType::MEM_UB);
    if (poolA.AllocateAtOffset(bufHolderA, 0) != SUCCESS || poolB.AllocateAtOffset(bufHolderB, 0) != SUCCESS)
        return FAILED;
    s.state_.tensorOccupyMap[memIdA] = g.add0;
    s.state_.tensorOccupyMap[memIdB] = g.add1;
    return SUCCESS;
}
} // namespace dualdst_ut

TEST_F(ScheduleOoOTest, DualDst_DynShapeEq_DumpEqual_HitsIdentify)
{
    auto g = dualdst_ut::BuildDualDstGraph_2(
        /*l0cShape*/ {dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
        /*tileShape*/ {dualdst_ut::TILE_M, dualdst_ut::TILE_N},
        /*fromOff0*/ {0, 0},
        /*fromOff1*/ {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 1u);
}

TEST_F(ScheduleOoOTest, DualDst_DynShapeEq_ConcreteEqualButDifferentDump_StillHits)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar("a", dualdst_ut::TILE_M), SymbolicScalar("b", dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar("c", dualdst_ut::TILE_M), SymbolicScalar("d", dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 1u);
}

TEST_F(ScheduleOoOTest, DualDst_DynShapeEq_DumpDifferAndNoConcrete_NoPair)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("X0"), SymbolicScalar("X1")});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("Y0"), SymbolicScalar("Y1")});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 0u);
}
TEST_F(ScheduleOoOTest, DualDst_ReadGeometry_PrefersStaticValidShape)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("X0"), SymbolicScalar("X1")});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("Y0"), SymbolicScalar("Y1")});
    dualdst_ut::InjectStaticValidShape(*g.copy0, {dualdst_ut::TILE_M, dualdst_ut::TILE_N});
    dualdst_ut::InjectStaticValidShape(*g.copy1, {dualdst_ut::TILE_M, dualdst_ut::TILE_N});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 1u);
}

TEST_F(ScheduleOoOTest, DualDst_Identify_SplitN_HappyPath)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].opEarly, g.copy0);
    EXPECT_EQ(pairs[0].opLate, g.copy1);
    EXPECT_NE(pairs[0].allocEarly, nullptr);
    EXPECT_NE(pairs[0].allocLate, nullptr);
}

TEST_F(ScheduleOoOTest, DualDst_ShouldEnableDualDst_WithOnlineSoftmaxTasks)
{
    auto g = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    auto opList = g.func->Operations().DuplicatedOpList();
    TaskSplitter splitter;
    splitter.SplitGraph(opList);
    OoOSchedule oooSchedule;
    ASSERT_EQ(oooSchedule.TaskSchedule(opList, *g.func, splitter), SUCCESS);

    EXPECT_TRUE(dualdst_ut::HasAiv0AndAiv1Tasks(splitter));
    ASSERT_TRUE(oooSchedule.ShouldEnableDualDst(splitter));
    ASSERT_FALSE(oooSchedule.dualDstPairs_.empty());
    EXPECT_TRUE(dualdst_ut::HasIsoAllocPair(oooSchedule.dualDstPairs_, g.allocUb0, g.allocUb1) ||
                dualdst_ut::HasIsoAllocPair(oooSchedule.dualDstPairs_, g.allocOut0, g.allocOut1));
    EXPECT_TRUE(dualdst_ut::HasIsoAllocPair(oooSchedule.dualDstOpPairs_, g.add0, g.add1));

    OoOScheduler scheduler(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(scheduler, g), SUCCESS);
    scheduler.SetDualDstPairs(oooSchedule.dualDstPairs_);
    scheduler.SetDualDstOpPairs(oooSchedule.dualDstOpPairs_);
    std::vector<DualDstPair> pairs;
    EXPECT_EQ(scheduler.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    ASSERT_TRUE(scheduler.state_.enableDualDst);
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].opEarly, g.copy0);
    EXPECT_EQ(pairs[0].opLate, g.copy1);
    EXPECT_EQ(scheduler.dualDstEngine_.RealignAllocByIso(scheduler.state_.orderedOps), SUCCESS);
    EXPECT_EQ(scheduler.dualDstEngine_.RunDualDstFuse(), SUCCESS);
    EXPECT_TRUE(dualdst_ut::HasDualDstOp(scheduler.state_.orderedOps));
}

TEST_F(ScheduleOoOTest, DualDst_Realign_MovesConsumersWithAllocs)
{
    auto g = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler s(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(s, g), SUCCESS);
    s.SetDualDstPairs({{g.allocUb0, g.allocUb1}});
    s.SetDualDstOpPairs({{g.add0, g.add1}});

    auto& ops = s.state_.orderedOps;

    // 构造出的 opList 并非拓扑序（copy 排在自己的 UB_ALLOC 之前），拓扑守卫会直接
    // 否掉整次重排。先归位成拓扑序，让重排本身有机会执行。
    ASSERT_TRUE(dualdst_ut::TopoSortInPlace(s.state_.depManager, ops));

    auto slotOf = [&ops](Operation* op) {
        return static_cast<int>(std::find(ops.begin(), ops.end(), op) - ops.begin());
    };
    int allocSlot = slotOf(g.allocUb1);
    int addSlot = slotOf(g.add1);
    ASSERT_LT(addSlot, static_cast<int>(ops.size()));
    ASSERT_LT(allocSlot, addSlot);
    ASSERT_LT(slotOf(g.allocUb0), slotOf(g.add0));

    std::swap(ops[allocSlot], ops[addSlot]);
    auto before = ops;
    ASSERT_EQ(s.dualDstEngine_.RealignAllocByIso(ops), SUCCESS);

    ASSERT_EQ(before.size(), ops.size());
    EXPECT_EQ(std::unordered_set<Operation*>(before.begin(), before.end()),
              std::unordered_set<Operation*>(ops.begin(), ops.end()));
    EXPECT_EQ(slotOf(g.allocUb1), allocSlot);
    EXPECT_EQ(slotOf(g.add1), addSlot);
}

TEST_F(ScheduleOoOTest, DualDst_Realign_SkipsWhenTopoOrderWouldBreak)
{
    auto g = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler s(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(s, g), SUCCESS);
    // Pairing an alloc with a consumer of the other core forces the mirrored order to
    // place a consumer ahead of its own producer, which the topo guard must reject.
    s.SetDualDstPairs({{g.allocUb0, g.allocUb1}});
    s.SetDualDstOpPairs({{g.add0, g.allocOut1}});

    auto before = s.state_.orderedOps;
    EXPECT_EQ(s.dualDstEngine_.RealignAllocByIso(s.state_.orderedOps), SUCCESS);
    EXPECT_EQ(before, s.state_.orderedOps);
}

TEST_F(ScheduleOoOTest, DualDst_Realign_SkipsPairsOutsideOpList)
{
    auto g = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    auto other = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler s(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(s, g), SUCCESS);
    s.SetDualDstPairs({{g.allocUb0, g.allocUb1}});
    s.SetDualDstOpPairs({{other.add0, other.add1}});

    auto before = s.state_.orderedOps;
    EXPECT_EQ(s.dualDstEngine_.RealignAllocByIso(s.state_.orderedOps), SUCCESS);
    EXPECT_EQ(before, s.state_.orderedOps);
}

TEST_F(ScheduleOoOTest, DualDst_Realign_SkipsWhenOneAiv1OpHasTwoAiv0Pairs)
{
    auto g = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler s(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(s, g), SUCCESS);
    s.SetDualDstPairs({{g.allocUb0, g.allocUb1}});
    s.SetDualDstOpPairs({{g.add0, g.allocUb1}});

    auto before = s.state_.orderedOps;
    EXPECT_EQ(s.dualDstEngine_.RealignAllocByIso(s.state_.orderedOps), SUCCESS);
    EXPECT_EQ(before, s.state_.orderedOps);
}

TEST_F(ScheduleOoOTest, DualDst_Identify_SplitM_HappyPath)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M * 2, dualdst_ut::TILE_N},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {dualdst_ut::TILE_M, 0});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].opEarly, g.copy0); // fromM 0 < TILE_M
    EXPECT_EQ(pairs[0].opLate, g.copy1);
    auto l0c = g.copy0->GetInputOperand(0);
    EXPECT_EQ(s.dualDstEngine_.dualDstL0CDirection_.count(l0c), 1u);
    EXPECT_EQ(s.dualDstEngine_.dualDstL0CDirection_[l0c], 0);
}

TEST_F(ScheduleOoOTest, DualDst_Identify_NotAdjacent_NoPair)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 4},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0},
                                             {0, dualdst_ut::TILE_N * 2});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 0u);
}

TEST_F(ScheduleOoOTest, DualDst_Identify_AddConsumerUnsupported_NoPair)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 0u);
}

TEST_F(ScheduleOoOTest, DualDst_Identify_SameConsumerCore_NoPair)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g, /*sameCoreForAdds=*/true); // both consumers AIV0

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 0u);
}

TEST_F(ScheduleOoOTest, DualDst_RunDualDstFuse_DisabledIsNoOp)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    s.SetEnableDualDst(false);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);
}

TEST_F(ScheduleOoOTest, DualDst_RunDualDstFuse_SingleAivPoolEarlyExit)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_ONE), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);
}

TEST_F(ScheduleOoOTest, DualDst_RunDualDstFuse_ActuallyFusesAndMutatesFunction)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    size_t opsBefore = g.func->Operations().size();
    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);

    size_t opsAfter = g.func->Operations().size();
    EXPECT_EQ(opsBefore, opsAfter + 2);

    bool hasFused = false;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            hasFused = true;
            break;
        }
    }
    EXPECT_TRUE(hasFused);
}

TEST_F(ScheduleOoOTest, DualDst_AllocQueryHelpers_AfterFuse)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);

    Operation* dual = nullptr;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            dual = &op;
            break;
        }
    }
    ASSERT_NE(dual, nullptr);

    Operation* survivingUbAlloc = nullptr;
    for (auto* pred : s.state_.depManager.GetPredecessors(dual)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos) {
            survivingUbAlloc = pred;
            break;
        }
    }
    ASSERT_NE(survivingUbAlloc, nullptr);

    EXPECT_TRUE(s.state_.IsDualDstAlloc(survivingUbAlloc));
    EXPECT_EQ(s.dualDstEngine_.GetDualDstCopyOpFor(survivingUbAlloc), dual);
    int paired = s.dualDstEngine_.GetDualDstPairedMemId(survivingUbAlloc);
    EXPECT_NE(paired, -1);
    EXPECT_NE(paired, survivingUbAlloc->GetOutputOperand(0)->memoryrange.memId);

    EXPECT_FALSE(s.state_.IsDualDstAlloc(g.allocL0c));
    EXPECT_EQ(s.dualDstEngine_.GetDualDstCopyOpFor(g.allocL0c), nullptr);
    EXPECT_EQ(s.dualDstEngine_.GetDualDstPairedMemId(nullptr), -1);
}

TEST_F(ScheduleOoOTest, DualDst_AllocGuardBlocksAiv0UntilAiv1DualDstAllocRetires)
{
    auto g = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler s(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(s, g, true), SUCCESS);
    s.state_.schedInfoMap[g.allocUb1].execOrder = -1;
    s.state_.schedInfoMap[g.allocUb0].execOrder = 1;
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);

    Operation* dual = dualdst_ut::FindDualDstOp(*g.func);
    ASSERT_NE(dual, nullptr);
    Operation* aiv1DualDstAlloc = dualdst_ut::FindUbAllocPred(s, dual);
    ASSERT_NE(aiv1DualDstAlloc, nullptr);
    EXPECT_EQ(s.state_.schedInfoMap[aiv1DualDstAlloc].coreLocation, CoreLocationType::AIV1);
    ASSERT_TRUE(s.state_.IsDualDstAlloc(aiv1DualDstAlloc));

    Operation* aiv0GuardedAlloc = dualdst_ut::AddGuardedUbAlloc(*g.func, s, 91001, CoreLocationType::AIV0, 0);
    Operation* aiv1GuardedAlloc = dualdst_ut::AddGuardedUbAlloc(*g.func, s, 91002, CoreLocationType::AIV1, 0);
    Operation* aiv0LaterGuardedAlloc = dualdst_ut::AddGuardedUbAlloc(*g.func, s, 91003, CoreLocationType::AIV0, 1);
    s.state_.depManager.AddAllocDependency(aiv0GuardedAlloc, g.add0);
    s.state_.depManager.AddAllocDependency(aiv0LaterGuardedAlloc, g.add0);
    s.state_.depManager.AddAllocDependency(aiv1GuardedAlloc, g.add1);

    EXPECT_EQ(s.dualDstEngine_.BuildDualDstAllocGuards(), SUCCESS);
    EXPECT_FALSE(s.dualDstEngine_.IsDualDstAllocGuardSatisfied(aiv0GuardedAlloc));
    auto guardAllocs = s.dualDstEngine_.GetUnretiredGuardAllocs(aiv0GuardedAlloc);
    ASSERT_EQ(guardAllocs.size(), 1u);
    EXPECT_EQ(guardAllocs[0], aiv1DualDstAlloc);
    EXPECT_TRUE(s.dualDstEngine_.IsDualDstAllocGuardSatisfied(aiv0LaterGuardedAlloc));

    EXPECT_FALSE(s.dualDstEngine_.IsDualDstAllocGuardSatisfied(aiv1GuardedAlloc));
    guardAllocs = s.dualDstEngine_.GetUnretiredGuardAllocs(aiv1GuardedAlloc);
    ASSERT_EQ(guardAllocs.size(), 1u);
    EXPECT_EQ(guardAllocs[0], aiv1DualDstAlloc);

    s.state_.schedInfoMap[aiv1DualDstAlloc].isRetired = true;
    EXPECT_TRUE(s.dualDstEngine_.IsDualDstAllocGuardSatisfied(aiv0GuardedAlloc));
    EXPECT_TRUE(s.dualDstEngine_.GetUnretiredGuardAllocs(aiv0GuardedAlloc).empty());
}

TEST_F(ScheduleOoOTest, DualDst_AivUbAllocUsesMatchedPeerOffset)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    s.SetEnableDualDst(true);
    s.SetEnableDualDstAllocGuard(true);

    constexpr int kAiv0MemId = 92001;
    constexpr int kAiv1MemId = 92002;
    constexpr int kPlaceholderMemId = 92003;
    constexpr uint64_t kAllocSize = 256;
    constexpr uint64_t kPlaceholderSize = 128;
    constexpr uint64_t kPoolSize = 2048;

    s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB] = BufferPool(MemoryType::MEM_UB, kPoolSize);
    s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB] = BufferPool(MemoryType::MEM_UB, kPoolSize);
    auto& aiv0Pool = s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& aiv1Pool = s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    auto placeholder = std::make_shared<LocalBuffer>(kPlaceholderMemId, kPlaceholderSize, MemoryType::MEM_UB);
    ASSERT_EQ(aiv0Pool.AllocateAtOffset(placeholder, 0), SUCCESS);

    auto addAivAlloc = [&](int memId, CoreLocationType core) {
        auto tensor = CreateTensor(DataType::DT_FP32, {8, 8}, MemoryType::MEM_UB, memId);
        Operation* alloc = &PassOperationUtils::AddOperation(*g.func, Opcode::OP_UB_ALLOC, {},
                                                             LogicalTensors({tensor}));
        s.state_.schedInfoMap[alloc].isAlloc = true;
        s.state_.schedInfoMap[alloc].isRetired = false;
        s.state_.schedInfoMap[alloc].coreLocation = core;
        s.state_.SetOpMemIds(alloc, {memId});
        s.state_.localBufferMap[memId] = std::make_shared<LocalBuffer>(memId, kAllocSize, MemoryType::MEM_UB);
        s.state_.bufRefCount[memId] = 2;
        return alloc;
    };

    Operation* aiv0Alloc = addAivAlloc(kAiv0MemId, CoreLocationType::AIV0);
    Operation* aiv1Alloc = addAivAlloc(kAiv1MemId, CoreLocationType::AIV1);
    s.dualDstEngine_.guardedAllocToDualDstAllocs_[aiv1Alloc].push_back(aiv0Alloc);

    uint64_t commitCnt = 0;
    bool allocated = false;
    EXPECT_EQ(s.TryRegularAllocOnce(aiv0Alloc, MemoryType::MEM_UB, CoreLocationType::AIV0,
                                    s.state_.GetOpMemIds(aiv0Alloc), commitCnt, allocated),
              SUCCESS);
    ASSERT_TRUE(allocated);
    ASSERT_EQ(aiv0Pool.GetBufferOffset(kAiv0MemId), kPlaceholderSize);

    allocated = false;
    EXPECT_EQ(s.TryRegularAllocOnce(aiv1Alloc, MemoryType::MEM_UB, CoreLocationType::AIV1,
                                    s.state_.GetOpMemIds(aiv1Alloc), commitCnt, allocated),
              SUCCESS);
    ASSERT_TRUE(allocated);
    EXPECT_EQ(aiv1Pool.GetBufferOffset(kAiv1MemId), kPlaceholderSize);
    EXPECT_NE(aiv1Pool.GetBufferOffset(kAiv1MemId), 0u);
}

TEST_F(ScheduleOoOTest, DualDst_AllocateDualDstAtCurrent_HappyPath)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);

    Operation* dual = nullptr;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            dual = &op;
            break;
        }
    }
    ASSERT_NE(dual, nullptr);
    Operation* survivingUbAlloc = nullptr;
    for (auto* pred : s.state_.depManager.GetPredecessors(dual)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos) {
            survivingUbAlloc = pred;
            break;
        }
    }
    ASSERT_NE(survivingUbAlloc, nullptr);

    int memIdA = survivingUbAlloc->GetOutputOperand(0)->memoryrange.memId;
    int memIdB = s.dualDstEngine_.GetDualDstPairedMemId(survivingUbAlloc);
    ASSERT_NE(s.state_.localBufferMap.find(memIdA), s.state_.localBufferMap.end());
    ASSERT_NE(s.state_.localBufferMap.find(memIdB), s.state_.localBufferMap.end());

    bool allocated = false;
    EXPECT_EQ(s.dualDstEngine_.AllocateDualDstAtCurrent(survivingUbAlloc, allocated), SUCCESS);
    EXPECT_TRUE(allocated);
}

TEST_F(ScheduleOoOTest, DualDst_SelectSpillBuffers_PicksMatchingGroupAcrossAivPools)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    dualdst_ut::UpdateCopyDynValidShape(g);

    OoOScheduler s(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(s, g), SUCCESS);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);

    Operation* dual = dualdst_ut::FindDualDstOp(*g.func);
    ASSERT_NE(dual, nullptr);
    Operation* survivingUbAlloc = dualdst_ut::FindUbAllocPred(s, dual);
    ASSERT_NE(survivingUbAlloc, nullptr);
    ASSERT_TRUE(s.state_.IsDualDstAlloc(survivingUbAlloc));

    int memIdA = survivingUbAlloc->GetOutputOperand(0)->memoryrange.memId;
    ASSERT_NE(s.state_.localBufferMap.find(memIdA), s.state_.localBufferMap.end());
    size_t needSize = s.state_.localBufferMap[memIdA]->size;

    constexpr int kPlaceholderMemIdA = 90001;
    constexpr int kPlaceholderMemIdB = 90002;
    ASSERT_EQ(dualdst_ut::FillAivPoolsWithPlaceholderBuffers(s, g, needSize, kPlaceholderMemIdA, kPlaceholderMemIdB),
              SUCCESS);

    auto spillGroup = s.SelectSpillBuffers(survivingUbAlloc);
    ASSERT_EQ(spillGroup.size(), 2u);
    EXPECT_NE(std::find(spillGroup.begin(), spillGroup.end(), kPlaceholderMemIdA), spillGroup.end());
    EXPECT_NE(std::find(spillGroup.begin(), spillGroup.end(), kPlaceholderMemIdB), spillGroup.end());
}

TEST_F(ScheduleOoOTest, DualDst_SelectSpillBuffers_EmptyPoolsReturnEmpty)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);

    Operation* dual = nullptr;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            dual = &op;
            break;
        }
    }
    ASSERT_NE(dual, nullptr);
    Operation* survivingUbAlloc = nullptr;
    for (auto* pred : s.state_.depManager.GetPredecessors(dual)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos) {
            survivingUbAlloc = pred;
            break;
        }
    }
    ASSERT_NE(survivingUbAlloc, nullptr);
    ASSERT_TRUE(s.state_.IsDualDstAlloc(survivingUbAlloc));

    auto spillGroup = s.SelectSpillBuffers(survivingUbAlloc);
    EXPECT_TRUE(spillGroup.empty());
}

TEST_F(ScheduleOoOTest, DualDst_GetDualSpillGroup_FindsSharedStartAddrCandidate)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);

    auto& poolA = s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& poolB = s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];

    constexpr int kBufMemIdA = 80001;
    constexpr int kBufMemIdB = 80002;
    constexpr size_t kBufSize = 1024;
    constexpr size_t kNeedSize = 512;
    ASSERT_GE(poolA.GetMemSize(), kBufSize);
    ASSERT_GE(poolB.GetMemSize(), kBufSize);

    auto bufA = std::make_shared<LocalBuffer>(kBufMemIdA, kBufSize, MemoryType::MEM_UB);
    auto bufB = std::make_shared<LocalBuffer>(kBufMemIdB, kBufSize, MemoryType::MEM_UB);
    ASSERT_EQ(poolA.AllocateAtOffset(bufA, 0), SUCCESS);
    ASSERT_EQ(poolB.AllocateAtOffset(bufB, 0), SUCCESS);

    auto groups = s.GetDualSpillGroup(poolA, poolB, kNeedSize);
    ASSERT_EQ(groups.size(), 1u);
    ASSERT_EQ(groups[0].size(), 2u);
    EXPECT_NE(std::find(groups[0].begin(), groups[0].end(), kBufMemIdA), groups[0].end());
    EXPECT_NE(std::find(groups[0].begin(), groups[0].end(), kBufMemIdB), groups[0].end());
}

TEST_F(ScheduleOoOTest, DualDst_GetDualSpillGroup_NeedSizeExceedsPoolReturnsEmpty)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);

    auto& poolA = s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& poolB = s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];

    constexpr int kBufMemIdA = 80003;
    constexpr int kBufMemIdB = 80004;
    auto bufA = std::make_shared<LocalBuffer>(kBufMemIdA, 1024, MemoryType::MEM_UB);
    auto bufB = std::make_shared<LocalBuffer>(kBufMemIdB, 1024, MemoryType::MEM_UB);
    ASSERT_EQ(poolA.AllocateAtOffset(bufA, 0), SUCCESS);
    ASSERT_EQ(poolB.AllocateAtOffset(bufB, 0), SUCCESS);

    size_t needSize = poolA.GetMemSize() + 1024;
    auto groups = s.GetDualSpillGroup(poolA, poolB, needSize);
    EXPECT_TRUE(groups.empty());
}

TEST_F(ScheduleOoOTest, DualDst_InferDynShape_RecordsStaticValidShape)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    InferDynShape pass;
    pass.RecordStaticValidShapeOnL0CCopyUB(*g.func);

    EXPECT_TRUE(g.copy0->HasAttribute(OpAttributeKey::staticValidShape));
    EXPECT_TRUE(g.copy1->HasAttribute(OpAttributeKey::staticValidShape));
    auto v0 = g.copy0->GetVectorIntAttribute<int64_t>(OpAttributeKey::staticValidShape);
    EXPECT_EQ(v0.size(), 2u);
    EXPECT_EQ(v0[0], dualdst_ut::TILE_M);
    EXPECT_EQ(v0[1], dualdst_ut::TILE_N);
}

TEST_F(ScheduleOoOTest, DualDst_InferDynShape_SkipsDynamicValidShape)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("dyn0"), SymbolicScalar(dualdst_ut::TILE_N)});

    InferDynShape pass;
    pass.RecordStaticValidShapeOnL0CCopyUB(*g.func);

    EXPECT_FALSE(g.copy0->HasAttribute(OpAttributeKey::staticValidShape));
}

TEST_F(ScheduleOoOTest, DualDst_GetNewOperations_DedupePreservesFirstOccurrence)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);

    s.state_.newOperations.clear();
    s.state_.newOperations.push_back(g.copy0);
    s.state_.newOperations.push_back(g.copy1);
    s.state_.newOperations.push_back(g.copy0); // dup
    s.state_.newOperations.push_back(nullptr); // null

    auto uniq = s.GetNewOperations();
    EXPECT_EQ(uniq.size(), 2u);
    EXPECT_EQ(uniq[0], g.copy0);
    EXPECT_EQ(uniq[1], g.copy1);
}

TEST_F(ScheduleOoOTest, IsoMatch_IsoMatchChains_RootSignatureMismatch)
{
    auto builder = std::make_shared<ComputationalGraphBuilder>();
    EXPECT_EQ(builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "ta0"), true);
    EXPECT_EQ(builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "tb0"), true);
    EXPECT_EQ(builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "ta_out"), true);
    EXPECT_EQ(builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "tb_out"), true);
    EXPECT_EQ(builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"ta0"}, "alloc_a0"), true);
    EXPECT_EQ(builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"tb0"}, "alloc_b0"), true);
    EXPECT_EQ(builder->AddOp(Opcode::OP_ADD, {"ta0", "ta0"}, {"ta_out"}, "add_a0"), true);
    EXPECT_EQ(builder->AddOp(Opcode::OP_SUB, {"tb0", "tb0"}, {"tb_out"}, "sub_b0"), true);

    std::vector<Operation*> opsA = {builder->GetOp("alloc_a0"), builder->GetOp("add_a0")};
    std::vector<Operation*> opsB = {builder->GetOp("alloc_b0"), builder->GetOp("sub_b0")};
    std::unordered_set<Operation*> setA(opsA.begin(), opsA.end());
    std::unordered_set<Operation*> setB(opsB.begin(), opsB.end());

    auto rootsA = FindTaskEntryOps(opsA, setA);
    auto rootsB = FindTaskEntryOps(opsB, setB);
    auto res = IsoMatchChains(rootsA, rootsB, setA, setB);
    EXPECT_FALSE(res.rootIsomorphic);
    EXPECT_EQ(res.pairs.size(), 0u);
}

// === upstream/master: ModifyAllocOrder tests ===

static int IndexOf(const std::vector<Operation*>& opList, Operation* op)
{
    for (size_t i = 0; i < opList.size(); i++) {
        if (opList[i] == op) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

TEST_F(ScheduleOoOTest, TestModifyAllocOrderPushesEarlyAllocDown)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> names{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> memTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                     MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, names, 0), true);
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{}, {}, {}, {"t1"}, {"t3"}, {"t2", "t4"}, {"t5"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t4"}, {"t5"}, {"t2"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"AllocT2", "AllocT4", "AllocT5", "Copyin1", "Copyin2", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ins, outs, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);

    std::vector<Operation*> opList{subGraph.GetOp("AllocT2"), subGraph.GetOp("AllocT4"), subGraph.GetOp("AllocT5"),
                                   subGraph.GetOp("Copyin1"), subGraph.GetOp("Copyin2"), subGraph.GetOp("Add1"),
                                   subGraph.GetOp("Copyout1")};
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Copyin1")) - IndexOf(opList, subGraph.GetOp("AllocT2")), 3);

    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.ModifyAllocOrder(opList), SUCCESS);

    EXPECT_EQ(opList.size(), 7u);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Copyin1")) - IndexOf(opList, subGraph.GetOp("AllocT2")), 1);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Copyin2")) - IndexOf(opList, subGraph.GetOp("AllocT4")), 1);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Add1")) - IndexOf(opList, subGraph.GetOp("AllocT5")), 1);
}

TEST_F(ScheduleOoOTest, TestModifyAllocOrderPullsLateAllocUp)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> names{"t1", "t2", "t3"};
    std::vector<MemoryType> memTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, names, 0), true);
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{"t1"}, {}, {"t2"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"Copyin1", "AllocT2", "Copyout1"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ins, outs, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);

    std::vector<Operation*> opList{subGraph.GetOp("Copyin1"), subGraph.GetOp("AllocT2"), subGraph.GetOp("Copyout1")};
    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.ModifyAllocOrder(opList), SUCCESS);

    EXPECT_EQ(opList.size(), 3u);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("AllocT2")), 0);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Copyin1")), 1);
}

TEST_F(ScheduleOoOTest, TestModifyAllocOrderKeepsUnreferencedAllocInPlace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> names{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> memTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                     MemoryType::MEM_UB};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, names, 0), true);
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{}, {}, {"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> outs{{"t4"}, {"t2"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"AllocT4", "AllocT2", "Copyin1", "Copyout1"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ins, outs, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);

    std::vector<Operation*> opList{subGraph.GetOp("AllocT4"), subGraph.GetOp("AllocT2"), subGraph.GetOp("Copyin1"),
                                   subGraph.GetOp("Copyout1")};
    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.ModifyAllocOrder(opList), SUCCESS);

    EXPECT_EQ(opList.size(), 4u);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("AllocT4")), 0);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Copyin1")) - IndexOf(opList, subGraph.GetOp("AllocT2")), 1);
}

TEST_F(ScheduleOoOTest, TestModifyAllocOrderInsertsAllocOnceForInPlaceOp)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> names{"t1", "t2", "t3"};
    std::vector<MemoryType> memTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, names, 0), true);
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{}, {"t1"}, {"t2", "t2"}, {"t2"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t2"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"AllocT2", "Copyin1", "AddInPlace", "Copyout1"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ins, outs, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);

    std::vector<Operation*> opList{subGraph.GetOp("AllocT2"), subGraph.GetOp("AddInPlace"), subGraph.GetOp("Copyout1")};
    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.ModifyAllocOrder(opList), SUCCESS);

    EXPECT_EQ(opList.size(), 3u);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("AllocT2")), 0);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("AddInPlace")), 1);
}

// === vecSortAlgo: OptimizeSort factory interface tests ===

// 用按 mode 实例化正确的 sorter 子类。
TEST_F(ScheduleOoOTest, OptimizeSortCreateFactory)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCreate", "TestCreate",
                                                      rootFuncPtr.get());
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<Operation*> emptyOps;
    auto priordfs = OptimizeSort::Create(emptyOps, *currFunctionPtr, "PriorDFS");
    EXPECT_NE(priordfs, nullptr);
    EXPECT_NE(dynamic_cast<PriorDFSSort*>(priordfs.get()), nullptr);

    auto clusterList = OptimizeSort::Create(emptyOps, *currFunctionPtr, "ClusterList");
    EXPECT_NE(clusterList, nullptr);
    EXPECT_NE(dynamic_cast<ClusterListSort*>(clusterList.get()), nullptr);

    auto unknown = OptimizeSort::Create(emptyOps, *currFunctionPtr, "UnknownMode");
    EXPECT_EQ(unknown, nullptr);
}

// SortOps 端到端——oooSortModeAiv="ClusterList" + 纯 AIV 图 → SUCCESS。
TEST_F(ScheduleOoOTest, SortOpsClusterListMode)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Copyin1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);
    function->paramConfigs_.oooSortModeAiv = "ClusterList";

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    auto result = sort.GetOperations();
    EXPECT_EQ(result.size(), 5U);
    Operation* copyin = subGraph.GetOp("Copyin1");
    Operation* add = subGraph.GetOp("Add1");
    Operation* copyout = subGraph.GetOp("Copyout1");
    auto pos = [&](Operation* target) {
        for (size_t i = 0; i < result.size(); ++i) {
            if (result[i] == target) {
                return i;
            }
        }
        return result.size();
    };
    EXPECT_LT(pos(copyin), pos(add));
    EXPECT_LT(pos(add), pos(copyout));
}

// SortOps 端到端——oooSortModeAiv="" + 纯 AIV 图 → SUCCESS（默认走 PriorDFS）。
TEST_F(ScheduleOoOTest, SortOpsDefaultPriorDFSMode)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Copyin1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);
    function->paramConfigs_.oooSortModeAiv = "";

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    auto result = sort.GetOperations();
    EXPECT_EQ(result.size(), 5U);
}

// SortOps 端到端——空 operations → SUCCESS。
TEST_F(ScheduleOoOTest, SortOpsEmptyInput)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestEmpty", "TestEmpty",
                                                      rootFuncPtr.get());
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<Operation*> emptyOps;
    OptimizeSort sort(emptyOps, *currFunctionPtr);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    EXPECT_TRUE(sort.GetOperations().empty());
}

// 图结构：
//   copy_in1 -> add1 -> add2 -> add3 -> copy_out1   (depth=4, 源点={copy_in1})
//   copy_in1 -> mul1 -> copy_out2                    (depth=2, 源点={copy_in1})
//   copy_in2 -> sub1 -> sub2 -> copy_out3            (depth=3, 源点={copy_in2})
//
// 三个出口节点：copy_out1(depth=4), copy_out2(depth=2), copy_out3(depth=3)
// copy_out1 和 copy_out2 同源（源点都是 copy_in1），copy_out3 不同源（源点是 copy_in2）
// 正确排序：copy_out1(4), copy_out2(2), copy_out3(3)  — 同源排一起，组间按最大深度降序
// 错误排序：copy_out1(4), copy_out3(3), copy_out2(2)  — 纯深度排序，同源被拆开
TEST_F(ScheduleOoOTest, PriorDFSSort_SortOutNodeQueue_SameSourceGrouped)
{
    ComputationalGraphBuilder b;
    // t0:DDR(输入)  t1:UB(copy_in1输出)  t2:UB(add1输出)  t3:UB(add2输出)  t4:UB(add3输出/copy_out1输入)
    // t5:UB(mul1输出/copy_out2输入)  t6:DDR(输入)  t7:UB(copy_in2输出)  t8:UB(sub1输出)  t9:UB(sub2输出/copy_out3输入)
    // t10:DDR(copy_out输出)
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"};
    std::vector<MemoryType> memTypes{MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_UB, MEM_UB,        MEM_UB,
                                     MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_UB, MEM_DEVICE_DDR};
    EXPECT_TRUE(b.AddTensors(DT_FP32, {64, 64}, memTypes, tensorNames, 0));

    // 每个UB张量都需要一个alloc
    std::vector<Opcode> opcodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_COPY_OUT,
                                Opcode::OP_MUL,      Opcode::OP_COPY_OUT, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_SUB,      Opcode::OP_SUB,      Opcode::OP_COPY_OUT};
    // Alloc1->t1, Alloc2->t2, Alloc3->t3, Alloc4->t4, Alloc5->t5, Alloc6->t7, Alloc7->t8
    // CopyIn1: t0->t1, Add1: t1->t2, Add2: t2->t3, Add3: t3->t4, CopyOut1: t4->t10
    // Mul1: t1->t5, CopyOut2: t5->t10
    // Alloc8->t7(已有Alloc6), 这里t9也需要alloc
    // CopyIn2: t6->t7, Sub1: t7->t8, Sub2: t8->t9, CopyOut3: t9->t10
    std::vector<std::vector<std::string>> ins{{},     {},     {},     {},     {},     {},     {},
                                              {"t0"}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t1"}, {"t5"},
                                              {},     {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::vector<std::string>> outs{{"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"},  {"t7"}, {"t8"},
                                               {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t10"}, {"t5"}, {"t10"},
                                               {"t9"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3", "Alloc4", "Alloc5",   "Alloc6", "Alloc7",
                                     "CopyIn1", "Add1",    "Add2",   "Add3",   "CopyOut1", "Mul1",   "CopyOut2",
                                     "Alloc8",  "CopyIn2", "Sub1",   "Sub2",   "CopyOut3"};
    EXPECT_TRUE(b.AddOps(opcodes, ins, outs, opNames, true));
    Function* func = b.GetFunction();
    ASSERT_NE(func, nullptr);

    auto ops = func->Operations().DuplicatedOpList();
    PriorDFSSort sorter(ops, *func);
    sorter.state_.Init(ops);

    std::vector<Operation*> outNodeQueue;
    sorter.depthCache_.clear();
    for (auto* op : ops) {
        if (sorter.state_.depManager.GetSuccessors(op).empty()) {
            outNodeQueue.emplace_back(op);
        }
    }
    ASSERT_EQ(outNodeQueue.size(), 3U);

    sorter.SortOutNodeQueue(outNodeQueue);

    EXPECT_EQ(outNodeQueue[0], b.GetOp("CopyOut1"));
    EXPECT_EQ(outNodeQueue[1], b.GetOp("CopyOut2"));
    EXPECT_EQ(outNodeQueue[2], b.GetOp("CopyOut3"));
}

} // namespace npu::tile_fwk
