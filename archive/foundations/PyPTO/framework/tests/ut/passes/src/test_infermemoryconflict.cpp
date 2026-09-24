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
 * \file test_infermemoryconflict.cpp
 * \brief Unit test for InferMemoryConflict pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"

#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#define private public
#include "passes/tensor_graph_pass/infer_memory_conflict.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

const int NUM_ZERO = 0;
const int NUM_ONE = 1;
const int NUM_2 = 2;
const int NUM_3 = 3;
const int NUM_4 = 4;
const int NUM_6 = 6;
const int NUM_8 = 8;
const int NUM_10 = 10;
const int NUM_11 = 11;
const int NUM_32 = 32;
const int NUM_64 = 64;
const int NUM_127 = 127;
const int NUM_128 = 128;
const int NUM_129 = 129;

class InferMemoryConflictTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "InferMemoryTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

int CountRegisterCopy(Function* function)
{
    int cnt = 0;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            cnt += 1;
        }
    }
    return cnt;
}

void ExpectNoRegisterCopyForViewReshapeMatmul(const Shape& reshapeInputShape)
{
    const Shape reshapeOutputShape = {NUM_64, NUM_128, NUM_64};
    const Shape matmulRhsShape = {NUM_64, NUM_64};
    const Shape conflictRawShape = {NUM_64, NUM_128, NUM_32};

    ComputationalGraphBuilder G;
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, reshapeInputShape, "t1"));
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, reshapeInputShape, "t2"));
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, reshapeInputShape, "t3"));
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, reshapeOutputShape, "t4"));
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, matmulRhsShape, "t5"));
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, matmulRhsShape, "t6"));
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, reshapeOutputShape, "o1"));
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, reshapeOutputShape, "o2"));

    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"t1"}, {"t2"}, "V1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"t2"}, {"t3"}, "V2"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_RESHAPE, {"t3"}, {"t4"}, "R1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_A_MUL_B, {"t4", "t5"}, {"o1"}, "MUL1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_A_MUL_B, {"t4", "t6"}, {"o2"}, "MUL2"));

    G.GetTensor("t4")->tensor->UpdateRawShape(conflictRawShape);
    ASSERT_TRUE(G.SetInCast({"t1", "t5", "t6"}));
    ASSERT_TRUE(G.SetOutCast({"o1", "o2"}));

    auto currFunctionPtr = G.GetFunction();
    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountRegisterCopy(currFunctionPtr), NUM_ZERO);
}

TEST_F(InferMemoryConflictTest, CheckRawShapeConflictInShapeNegative)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "CheckRawShapeTest", "CheckRawShapeTest",
                                                      nullptr);
    std::vector<int64_t> inShape = {-1, 4};
    std::vector<int64_t> outShape = {2, 4};
    std::shared_ptr<RawTensor> inRaw = std::make_shared<RawTensor>(DT_FP32, inShape);
    std::shared_ptr<RawTensor> outRaw = std::make_shared<RawTensor>(DT_FP32, outShape);
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(inRaw, std::vector<int64_t>{0, 0}, inShape,
                                                               CreateTestConstIntVector(inShape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(outRaw, std::vector<int64_t>{0, 0}, outShape,
                                                                CreateTestConstIntVector(outShape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {outTensor});
    InferMemoryConflict pass;
    // 这里只关注 inShape 中存在负值时的早返回分支，reshapeOp 不会被访问，传入 nullptr 即可
    EXPECT_TRUE(pass.CheckRawShapeConflict(inTensor, outTensor));
}

TEST_F(InferMemoryConflictTest, CheckRawShapeConflictOutShapeNegative)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "CheckRawShapeTest2",
                                                      "CheckRawShapeTest2", nullptr);
    std::vector<int64_t> inShape = {2, 4};
    std::vector<int64_t> outShape = {-1, 4};
    std::shared_ptr<RawTensor> inRaw = std::make_shared<RawTensor>(DT_FP32, inShape);
    std::shared_ptr<RawTensor> outRaw = std::make_shared<RawTensor>(DT_FP32, outShape);
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(inRaw, std::vector<int64_t>{0, 0}, inShape,
                                                               CreateTestConstIntVector(inShape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(outRaw, std::vector<int64_t>{0, 0}, outShape,
                                                                CreateTestConstIntVector(outShape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {outTensor});
    InferMemoryConflict pass;
    EXPECT_TRUE(pass.CheckRawShapeConflict(inTensor, outTensor));
}

TEST_F(InferMemoryConflictTest, TestInit)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {NUM_2, NUM_4};
    std::vector<int64_t> shape1 = {NUM_2, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_2};

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor, offset1, shape1,
                                                             CreateTestConstIntVector(shape1));
    auto input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor, offset2, shape1,
                                                             CreateTestConstIntVector(shape1));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    currFunctionPtr->inCasts_.push_back(input1);
    currFunctionPtr->inCasts_.push_back(input2);
    currFunctionPtr->outCasts_.push_back(output);

    auto& assembleOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {input1}, {ubTensor});
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assembleOp1.SetOpAttribute(assembleAttr1);

    auto& assembleOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {input2}, {ubTensor});
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp2.SetOpAttribute(assembleAttr2);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor}, {output});

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.memoryInfo.size(), NUM_3);
    EXPECT_NE(pass.memoryInfo.find(input1), pass.memoryInfo.end());
    EXPECT_NE(pass.memoryInfo.find(input2), pass.memoryInfo.end());
    EXPECT_NE(pass.memoryInfo.find(output), pass.memoryInfo.end());
    EXPECT_EQ(pass.memoryInfo[input1], input1);
    EXPECT_EQ(pass.memoryInfo[input2], input2);
    EXPECT_EQ(pass.memoryInfo[output], output);
}

/*
Case:
outCast->exp->T
Backward inference skips outcast input and does not insert register copy.
*/
TEST_F(InferMemoryConflictTest, TestSkipOutcastInputBackward)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSkipOutcast", "TestSkipOutcast",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {NUM_2, NUM_4};
    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};
    std::shared_ptr<RawTensor> outcastRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> curRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    outcastRawTensor->SetSymbol("outcast");
    curRawTensor->SetSymbol("cur");
    outcastRawTensor->memoryId = NUM_ZERO;
    curRawTensor->memoryId = NUM_ONE;

    auto outcastInput = npu::tile_fwk::IRBuilder().CreateTensorVar(outcastRawTensor, offset, shape,
                                                                   CreateTestConstIntVector(shape));
    auto curTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(curRawTensor, offset, shape,
                                                                CreateTestConstIntVector(shape));
    auto normalTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    currFunctionPtr->outCasts_.push_back(outcastInput);

    auto& producer = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {outcastInput}, {curTensor});

    InferMemoryConflict pass;
    EXPECT_TRUE(pass.ShouldSkipOutcastInput(outcastInput, *currFunctionPtr));
    EXPECT_FALSE(pass.ShouldSkipOutcastInput(normalTensor, *currFunctionPtr));

    pass.memoryInfo[curTensor] = curTensor;
    pass.memoryInfo[outcastInput] = outcastInput;
    auto status = pass.HandleConflictBackward(*currFunctionPtr, curTensor, outcastInput, &producer, curTensor);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(pass.preregcopys.size(), NUM_ZERO);
    EXPECT_EQ(pass.postregcopys.size(), NUM_ZERO);

    InferMemoryConflict updatePass;
    updatePass.memoryInfo[curTensor] = curTensor;
    std::queue<LogicalTensorPtr> curTensors;
    status = updatePass.UpdateBackwardTensor(*currFunctionPtr, curTensor, &producer, curTensors);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(curTensors.empty());
    EXPECT_EQ(updatePass.memoryInfo.find(outcastInput), updatePass.memoryInfo.end());
    EXPECT_EQ(updatePass.preregcopys.size(), NUM_ZERO);
    EXPECT_EQ(updatePass.postregcopys.size(), NUM_ZERO);
}

/*
Case 1:
input->view->T1->reshape->T2->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestForwardPropagation1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> shape = {NUM_2, NUM_4};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset, shape,
                                                            CreateTestConstIntVector(shape));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);

    currFunctionPtr->inCasts_.push_back(input);

    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset, shape,
                                                             CreateTestConstIntVector(shape));

    currFunctionPtr->outCasts_.push_back(output);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {T1}, {T2});

    auto& assembleOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output});
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp1.SetOpAttribute(assembleAttr1);

    auto& viewOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(offset);
    viewOp.SetOpAttribute(viewAttr);

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(&assembleOp1), pass.preregcopys.end());
}

/*
Case 3:
input1->view->T1->reshape->T2->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestForwardPropagation2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_4, NUM_2};
    std::vector<int64_t> shape1 = {NUM_2, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_ONE, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset1, shape1,
                                                            CreateTestConstIntVector(shape1));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset2, shape2,
                                                             CreateTestConstIntVector(shape2));

    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    ddrRawTensor1->SetSymbol("input1");
    ddrRawTensor2->SetSymbol("output");

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp.SetOpAttribute(assembleAttr);

    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp1.SetOpAttribute(viewAttr1);
    auto& reshapeOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {T1}, {T2});

    InferMemoryConflict pass;
    auto passStatus = pass.Init(*currFunctionPtr);
    passStatus = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(passStatus, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(&reshapeOp), pass.preregcopys.end());
}

/*
Case 4:
input->view->T->assemble->output(same memoryid)
*/
TEST_F(InferMemoryConflictTest, TestForwardPropagation3)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {NUM_2, NUM_2};
    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset, shape,
                                                            CreateTestConstIntVector(shape));
    auto T = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset, shape,
                                                             CreateTestConstIntVector(shape));

    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 0;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset);
    viewOp1.SetOpAttribute(viewAttr1);

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr);

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ZERO);
}

/*
Case 5:
T2->
input->index_outcast->T1->assemble->output
T0->
*/
TEST_F(InferMemoryConflictTest, TestForwardPropagation4)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_4, NUM_4};
    std::vector<int64_t> shape2 = {NUM_2, NUM_4};
    std::vector<int64_t> shape1 = {NUM_ONE, NUM_2};

    auto tensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));

    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape0, CreateTestConstIntVector(shape0));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset, shape0,
                                                             CreateTestConstIntVector(shape0));

    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset, shape0,
                                                            CreateTestConstIntVector(shape0));
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    currFunctionPtr->outCasts_.push_back(output);
    currFunctionPtr->inCasts_.push_back(input);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_OUTCAST, {tensor0, tensor2, input}, {tensor1});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {tensor1}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr);

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ZERO);
}

/*
Case 6:
T2->
input->index_outcast->T1->reshape->output
T0->
*/
TEST_F(InferMemoryConflictTest, TestForwardPropagation5)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_4, NUM_4};
    std::vector<int64_t> shape1 = {NUM_ONE, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_4};

    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};

    auto logicalTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto logicalTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset, shape0,
                                                            CreateTestConstIntVector(shape0));
    auto logicalTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape0, CreateTestConstIntVector(shape0));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset, shape0,
                                                             CreateTestConstIntVector(shape0));

    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_OUTCAST,
                                     {logicalTensor0, logicalTensor2, input}, {logicalTensor1});

    auto& reshapeOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {logicalTensor1},
                                                       {output});

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(&reshapeOp), pass.preregcopys.end());
}

/*
Case 1:
input1->view->T1->exp->T2->assemble->output
                         ->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestBackwardPropagation1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape1 = {NUM_2, NUM_4};
    std::vector<int64_t> shape2 = {NUM_2, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_2};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset1, shape1,
                                                            CreateTestConstIntVector(shape1));
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset1, shape2,
                                                              CreateTestConstIntVector(shape2));
    std::shared_ptr<RawTensor> ddrRawTensor3 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor3, offset2, shape2,
                                                              CreateTestConstIntVector(shape2));

    ddrRawTensor1->SetSymbol("input1");
    ddrRawTensor2->SetSymbol("output1");
    ddrRawTensor3->SetSymbol("output2");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    ddrRawTensor3->memoryId = 2;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output1);
    currFunctionPtr->outCasts_.push_back(output2);

    auto& viewOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(offset1);
    viewOp.SetOpAttribute(viewAttr);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {T1}, {T2});

    auto& assembleOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output1});
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assembleOp1.SetOpAttribute(assembleAttr1);

    auto& assembleOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output2});
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp2.SetOpAttribute(assembleAttr2);

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(&assembleOp2), pass.postregcopys.end());
}

/*
Case 2:
input1->view->T1->exp->T2->reshape->T3->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestBackwardPropagation2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_2, NUM_2, NUM_2};
    std::vector<int64_t> shape1 = {NUM_2, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_ONE, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset1, shape1,
                                                            CreateTestConstIntVector(shape1));
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto T3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset2, shape2,
                                                             CreateTestConstIntVector(shape2));

    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp1.SetOpAttribute(viewAttr1);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {T1}, {T2});

    auto& reshapeOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {T2}, {T3});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T3}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp.SetOpAttribute(assembleAttr);

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.postregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.postregcopys.find(&reshapeOp), pass.postregcopys.end());
}

/*
Case 3:
input->view->T1->exp->T2->assemble->output1
                        ->assemble->output2(same memoryId)
                        ->assemble->output3(same symbol)
*/
TEST_F(InferMemoryConflictTest, TestBackwardPropagation3)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape1 = {NUM_2, NUM_6};
    std::vector<int64_t> shape2 = {NUM_2, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_2};
    std::vector<int64_t> offset3 = {NUM_ZERO, NUM_4};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset1, shape1,
                                                            CreateTestConstIntVector(shape1));
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset1, shape2,
                                                              CreateTestConstIntVector(shape2));
    std::shared_ptr<RawTensor> ddrRawTensor3 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor3, offset2, shape2,
                                                              CreateTestConstIntVector(shape2));
    std::shared_ptr<RawTensor> ddrRawTensor4 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output3 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor4, offset3, shape2,
                                                              CreateTestConstIntVector(shape2));

    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output1");
    ddrRawTensor3->SetSymbol("output2");
    ddrRawTensor4->SetSymbol("output1");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    ddrRawTensor3->memoryId = 1;
    ddrRawTensor4->memoryId = 2;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output1);
    currFunctionPtr->outCasts_.push_back(output2);
    currFunctionPtr->outCasts_.push_back(output3);

    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp1.SetOpAttribute(viewAttr1);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {T1}, {T2});

    auto& assembleOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output1});
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assembleOp2.SetOpAttribute(assembleAttr2);

    auto& assembleOp3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output2});
    auto assembleAttr3 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp3.SetOpAttribute(assembleAttr3);

    auto& assembleOp4 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output3});
    auto assembleAttr4 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset3);
    assembleOp4.SetOpAttribute(assembleAttr4);

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ZERO);
    EXPECT_EQ(pass.postregcopys.size(), NUM_ZERO);
}

/*
Case 4:
T2->
input->index_outcast->T1->assemble->output
T0->
*/
TEST_F(InferMemoryConflictTest, TestBackwardPropagation4)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_4, NUM_4};
    std::vector<int64_t> shape1 = {1, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_4};

    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};

    auto T0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset, shape0,
                                                            CreateTestConstIntVector(shape0));
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape0, CreateTestConstIntVector(shape0));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset, shape0,
                                                              CreateTestConstIntVector(shape0));

    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output1");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output1);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_OUTCAST, {T0, T2, input}, {T1});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T1}, {output1});
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr1);

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ZERO);
    EXPECT_EQ(pass.postregcopys.size(), NUM_ZERO);
}

/*
Case 5:
T2->
input->index_outcast->T1->reshape->output
T0->
*/
TEST_F(InferMemoryConflictTest, TestBackwardPropagation5)
{
    auto currFunctionPtr1 = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                       nullptr);
    EXPECT_TRUE(currFunctionPtr1 != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_4, NUM_4};
    std::vector<int64_t> shape1 = {1, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_4};
    std::vector<int64_t> shape3 = {NUM_ONE, NUM_4, NUM_4};
    std::vector<int64_t> shape4 = {NUM_2, NUM_ONE, NUM_4};

    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};

    auto T0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset1, shape2,
                                                            CreateTestConstIntVector(shape2));
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset2, shape4,
                                                             CreateTestConstIntVector(shape4));

    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr1->inCasts_.push_back(input);
    currFunctionPtr1->outCasts_.push_back(output);

    PassOperationUtils::AddOperation(*currFunctionPtr1, Opcode::OP_INDEX_OUTCAST, {T0, T2, input}, {T1});

    auto& reshapeOp1 = PassOperationUtils::AddOperation(*currFunctionPtr1, Opcode::OP_RESHAPE, {T1}, {output});

    InferMemoryConflict testPass;
    auto status = testPass.Init(*currFunctionPtr1);
    status = testPass.BackwardPropagation(*currFunctionPtr1);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(testPass.postregcopys.size(), NUM_ONE);
    EXPECT_NE(testPass.postregcopys.find(&reshapeOp1), testPass.postregcopys.end());
}

/*
Case 1:
input1->view->T1->reshape->T2->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestBothPropagation1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {NUM_2, NUM_4};
    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset, shape,
                                                            CreateTestConstIntVector(shape));

    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset, shape,
                                                             CreateTestConstIntVector(shape));
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    auto& viewOperation = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(offset);
    viewOperation.SetOpAttribute(viewAttr);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {T1}, {T2});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr);

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_EQ(pass.postregcopys.size(), NUM_ZERO);
    EXPECT_NE(pass.preregcopys.find(&assembleOp), pass.preregcopys.end());
}

/*
Case 2:
input1->view->T1->reshape->T2->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestBothPropagation2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_2, NUM_4};
    std::vector<int64_t> shape1 = {NUM_2, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_ONE, NUM_2};
    std::vector<int64_t> shape3 = {NUM_2, NUM_2, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset1, shape1,
                                                            CreateTestConstIntVector(shape1));
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset2, shape2,
                                                              CreateTestConstIntVector(shape2));
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->SetSymbol("input1");

    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output1);

    auto& viewOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr2 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp2.SetOpAttribute(viewAttr2);

    auto& reshapeOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {T1}, {T2});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output1});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp.SetOpAttribute(assembleAttr);

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_EQ(pass.postregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(&reshapeOp), pass.preregcopys.end());
    EXPECT_NE(pass.postregcopys.find(&reshapeOp), pass.postregcopys.end());
}

/*
Case 2:
input1->view->T1->reshape->T2->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestInsertCopys)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape1 = {NUM_2, NUM_32};
    std::vector<int64_t> shape2 = {NUM_ONE, NUM_2, NUM_32};
    std::vector<int64_t> offset3 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset3, shape1,
                                                            CreateTestConstIntVector(shape1));
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);

    ddrRawTensor1->SetSymbol("input1");
    ddrRawTensor2->SetSymbol("output");
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset2, shape2,
                                                             CreateTestConstIntVector(shape2));
    ddrRawTensor1->memoryId = 0;
    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);
    ddrRawTensor2->memoryId = 1;

    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset3);
    viewOp1.SetOpAttribute(viewAttr1);

    auto& reshapeOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {T1}, {T2});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp.SetOpAttribute(assembleAttr);

    InferMemoryConflict pass;
    pass.preregcopys.insert(&reshapeOp);
    pass.postregcopys.insert(&reshapeOp);
    auto status = pass.InsertCopys(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    int cnt = 0;
    Operation* copy1 = nullptr;
    Operation* copy2 = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            if (*(op->GetOOperands().begin()) == T2) {
                copy2 = op;
                cnt += 1;
            } else {
                copy1 = op;
                cnt += 10;
            }
        }
    }
    EXPECT_EQ(cnt, NUM_11);
    EXPECT_NE(copy1, nullptr);
    EXPECT_EQ(*(copy1->GetIOperands().begin()), T1);
    auto newTensorOut1 = *(copy1->GetOOperands().begin());
    EXPECT_EQ(*(newTensorOut1->GetConsumers().begin()), &reshapeOp);
    EXPECT_NE(copy2, nullptr);
    auto newTensorIn2 = *(copy2->GetIOperands().begin());
    EXPECT_EQ(*(newTensorIn2->GetProducers().begin()), &reshapeOp);
}

/*
STest1
input1->view->T1->reshape->T2->assemble->output
单链，存在地址冲突
*/
TEST_F(InferMemoryConflictTest, STest1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {NUM_129, NUM_127};
    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset, shape,
                                                            CreateTestConstIntVector(shape));
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset, shape,
                                                             CreateTestConstIntVector(shape));

    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    auto& viewOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(offset);
    viewOp.SetOpAttribute(viewAttr);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {T1}, {T2});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr);

    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    int cnt = 0;
    Operation* copy = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
            cnt += 1;
        }
    }
    EXPECT_EQ(cnt, NUM_ONE);
    EXPECT_NE(copy, nullptr);
    EXPECT_EQ(*(copy->GetIOperands().begin()), T2);
    auto newTensorOut1 = *(copy->GetOOperands().begin());
    EXPECT_EQ(copy->GetTileShape().GetVecTile().size(), NUM_2);
    std::vector<int64_t> expectShape = {NUM_128, NUM_128};
    EXPECT_EQ(copy->GetTileShape().GetVecTile().tile, expectShape);
    EXPECT_EQ(*(newTensorOut1->GetConsumers().begin()), &assembleOp);
}

/*
STest2
input1->view->T1->index_outcast->T2->reshape->T3->exp->output
单链，存在reshape
*/
TEST_F(InferMemoryConflictTest, STest2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_32, NUM_32, NUM_128};
    std::vector<int64_t> shape1 = {NUM_32, NUM_32, NUM_64};
    std::vector<int64_t> shape2 = {NUM_ONE, NUM_32, NUM_32, NUM_64};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset1, shape1,
                                                            CreateTestConstIntVector(shape1));
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto T4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto T5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));

    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto T3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset2, shape2,
                                                             CreateTestConstIntVector(shape2));

    ddrRawTensor1->SetSymbol("input1");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp1.SetOpAttribute(viewAttr1);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_OUTCAST, {T4, T5, T1}, {T2});

    auto& reshapeOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {T2}, {T3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {T3}, {output});

    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    int cnt = 0;
    Operation* copy = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
            cnt += 1;
        }
    }
    EXPECT_EQ(cnt, NUM_ONE);
    EXPECT_NE(copy, nullptr);
    EXPECT_EQ(*(copy->GetIOperands().begin()), T2);
    EXPECT_EQ(copy->GetTileShape().GetVecTile().size(), NUM_3);
    std::vector<int64_t> expectShape = {NUM_8, NUM_32, NUM_64};
    EXPECT_EQ(copy->GetTileShape().GetVecTile().tile, expectShape);
    auto newTensorOut = *(copy->GetOOperands().begin());
    EXPECT_EQ(*(newTensorOut->GetConsumers().begin()), &reshapeOp);
}

/*
STest3
input1->view->T1->exp->T2->assemble->output
                         ->assemble->output
同一tensor assemble输出到不同outcast
*/
TEST_F(InferMemoryConflictTest, STest3)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape1 = {NUM_2, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_4};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_2};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset1, shape1,
                                                            CreateTestConstIntVector(shape1));
    auto T1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto T2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset1, shape2,
                                                              CreateTestConstIntVector(shape2));
    std::shared_ptr<RawTensor> ddrRawTensor3 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor3, offset2, shape2,
                                                              CreateTestConstIntVector(shape2));

    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output1");
    ddrRawTensor3->SetSymbol("output2");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    ddrRawTensor3->memoryId = 2;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output1);
    currFunctionPtr->outCasts_.push_back(output2);

    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp1.SetOpAttribute(viewAttr1);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {T1}, {T2});

    auto& assembleOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output1});
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assembleOp1.SetOpAttribute(assembleAttr1);

    auto& assembleOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {T2}, {output2});
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp2.SetOpAttribute(assembleAttr2);

    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    int cnt = 0;
    Operation* copy = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
            cnt += 1;
        }
    }
    EXPECT_EQ(cnt, NUM_ONE);
    EXPECT_NE(copy, nullptr);
    EXPECT_EQ(*(copy->GetIOperands().begin()), T2);
    EXPECT_EQ(copy->GetTileShape().GetVecTile().size(), NUM_2);
    std::vector<int64_t> expectShape = {NUM_2, NUM_32};
    EXPECT_EQ(copy->GetTileShape().GetVecTile().tile, expectShape);
    auto newTensorOut = *(copy->GetOOperands().begin());
    EXPECT_EQ(*(newTensorOut->GetConsumers().begin()), &assembleOp2);
}

struct PreMergeAssembleGraph {
    Function* function = nullptr;
    Operation* assembleOp1 = nullptr;
    Operation* assembleOp2 = nullptr;
    Operation* finalAssemble = nullptr;
    Operation* finalAssemble2 = nullptr;
    LogicalTensorPtr viewTensor;
    LogicalTensorPtr expTensor;
    LogicalTensorPtr mergeTensor;
};

void SetRawMemoryId(const LogicalTensorPtr& tensor, int64_t memoryId)
{
    if (tensor != nullptr) {
        tensor->GetRawTensor()->memoryId = memoryId;
    }
}

void SetAssembleLikeAttr(Operation* op, MemoryType from, const std::vector<int64_t>& offset)
{
    if (op == nullptr) {
        return;
    }
    op->SetOpAttribute(std::make_shared<AssembleOpAttribute>(from, offset));
    if (op->GetOpcode() == Opcode::OP_ATOMIC_RMW) {
        op->SetAttribute(OpAttributeKey::rmwMode, static_cast<int>(AtomicRMWMode::ADD));
    }
}

PreMergeAssembleGraph BuildPreMergeAssembleGraph(bool allPreMergeBranchesPure,
                                                 Opcode preMergeOpcode = Opcode::OP_ASSEMBLE)
{
    PreMergeAssembleGraph g;
    std::vector<int64_t> branchShape = {NUM_2, NUM_2};
    std::vector<int64_t> mergeShape = {NUM_2, NUM_4};
    std::vector<int64_t> offset0 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_2};

    ComputationalGraphBuilder graph;
    EXPECT_TRUE(graph.AddTensors(DataType::DT_FP32, branchShape, {"input1", "input2", "branch1", "branch2"}));
    EXPECT_TRUE(graph.AddTensors(DataType::DT_FP32, mergeShape, {"mergeTensor", "output"}));

    std::vector<Opcode> opCodes = {Opcode::OP_VIEW, allPreMergeBranchesPure ? Opcode::OP_VIEW : Opcode::OP_EXP,
                                   preMergeOpcode, preMergeOpcode, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands = {
        {"input1"}, {"input2"}, {"branch1"}, {"branch2"}, {"mergeTensor"}};
    std::vector<std::vector<std::string>> ooperands = {
        {"branch1"}, {"branch2"}, {"mergeTensor"}, {"mergeTensor"}, {"output"}};
    std::vector<std::string> opNames = {"view1", "branch2Op", "assembleOp1", "assembleOp2", "finalAssemble"};
    EXPECT_TRUE(graph.AddOps(opCodes, ioperands, ooperands, opNames, true));
    EXPECT_TRUE(graph.SetInCast({"input1", "input2"}));
    EXPECT_TRUE(graph.SetOutCast({"output"}));

    SetRawMemoryId(graph.GetTensor("input1"), NUM_ZERO);
    SetRawMemoryId(graph.GetTensor("input2"), NUM_ONE);
    SetRawMemoryId(graph.GetTensor("output"), NUM_2);

    graph.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0));
    if (allPreMergeBranchesPure) {
        graph.GetOp("branch2Op")->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0));
    }
    SetAssembleLikeAttr(graph.GetOp("assembleOp1"), MEM_DEVICE_DDR, offset0);
    SetAssembleLikeAttr(graph.GetOp("assembleOp2"), MEM_DEVICE_DDR, offset1);
    SetAssembleLikeAttr(graph.GetOp("finalAssemble"), MEM_DEVICE_DDR, offset0);

    g.function = graph.GetFunction();
    g.assembleOp1 = graph.GetOp("assembleOp1");
    g.assembleOp2 = graph.GetOp("assembleOp2");
    g.finalAssemble = graph.GetOp("finalAssemble");
    g.viewTensor = graph.GetTensor("branch1");
    g.expTensor = graph.GetTensor("branch2");
    g.mergeTensor = graph.GetTensor("mergeTensor");
    return g;
}

PreMergeAssembleGraph BuildPreMergeBackwardAssembleGraph()
{
    PreMergeAssembleGraph g;
    std::vector<int64_t> branchShape = {NUM_2, NUM_2};
    std::vector<int64_t> mergeShape = {NUM_2, NUM_4};
    std::vector<int64_t> offset0 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_2};

    ComputationalGraphBuilder graph;
    EXPECT_TRUE(graph.AddTensors(DataType::DT_FP32, branchShape, {"input1", "input2", "branch1", "branch2"}));
    EXPECT_TRUE(graph.AddTensors(DataType::DT_FP32, mergeShape, {"mergeTensor", "output1", "output2"}));

    std::vector<Opcode> opCodes = {Opcode::OP_VIEW,     Opcode::OP_EXP,      Opcode::OP_ASSEMBLE,
                                   Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands = {{"input1"},  {"input2"},      {"branch1"},
                                                       {"branch2"}, {"mergeTensor"}, {"mergeTensor"}};
    std::vector<std::vector<std::string>> ooperands = {{"branch1"},     {"branch2"}, {"mergeTensor"},
                                                       {"mergeTensor"}, {"output1"}, {"output2"}};
    std::vector<std::string> opNames = {"view1",       "branch2Op",      "assembleOp1",
                                        "assembleOp2", "finalAssemble1", "finalAssemble2"};
    EXPECT_TRUE(graph.AddOps(opCodes, ioperands, ooperands, opNames, true));
    EXPECT_TRUE(graph.SetInCast({"input1", "input2"}));
    EXPECT_TRUE(graph.SetOutCast({"output1", "output2"}));

    SetRawMemoryId(graph.GetTensor("input1"), NUM_ZERO);
    SetRawMemoryId(graph.GetTensor("input2"), NUM_ZERO);
    SetRawMemoryId(graph.GetTensor("output1"), NUM_ZERO);
    SetRawMemoryId(graph.GetTensor("output2"), NUM_2);

    graph.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0));
    SetAssembleLikeAttr(graph.GetOp("assembleOp1"), MEM_DEVICE_DDR, offset0);
    SetAssembleLikeAttr(graph.GetOp("assembleOp2"), MEM_DEVICE_DDR, offset1);
    SetAssembleLikeAttr(graph.GetOp("finalAssemble1"), MEM_DEVICE_DDR, offset0);
    SetAssembleLikeAttr(graph.GetOp("finalAssemble2"), MEM_DEVICE_DDR, offset0);

    g.function = graph.GetFunction();
    g.assembleOp1 = graph.GetOp("assembleOp1");
    g.assembleOp2 = graph.GetOp("assembleOp2");
    g.finalAssemble = graph.GetOp("finalAssemble1");
    g.finalAssemble2 = graph.GetOp("finalAssemble2");
    g.viewTensor = graph.GetTensor("branch1");
    g.expTensor = graph.GetTensor("branch2");
    g.mergeTensor = graph.GetTensor("mergeTensor");
    return g;
}

/*
STest3_1
input1->view->T1->assemble----------\
                                      ->merge->assemble->output
input2->exp->T2->assemble------------/
多分支先 assemble 汇合后再 assemble 输出，纯视图分支应在汇合前 assemble 前插 copy，
而不是在最后 assemble 前插 copy。
*/
TEST_F(InferMemoryConflictTest, STest3_1)
{
    auto g = BuildPreMergeAssembleGraph(false);
    ASSERT_NE(g.function, nullptr);
    auto currFunctionPtr = g.function;
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    status = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(g.assembleOp1), pass.preregcopys.end());
    EXPECT_EQ(pass.preregcopys.find(g.assembleOp2), pass.preregcopys.end());
    EXPECT_EQ(pass.preregcopys.find(g.finalAssemble), pass.preregcopys.end());

    status = pass.InsertCopys(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountRegisterCopy(currFunctionPtr), NUM_ONE);
    Operation* copy = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
        }
    }
    EXPECT_NE(copy, nullptr);
    EXPECT_EQ(copy->GetIOperands().front(), g.viewTensor);
    EXPECT_EQ(g.assembleOp1->GetIOperands().front(), copy->GetOOperands().front());
    EXPECT_EQ(g.assembleOp2->GetIOperands().front(), g.expTensor);
}

/*
STest3_2
input1->view->T1->assemble----------\
                                      ->merge->assemble->output
input2->view->T2->assemble-----------/
多分支 pre-merge producer 都是纯视图分支且都与 output 冲突时，
应回退到最终 assemble 前插入一次 copy。
*/
TEST_F(InferMemoryConflictTest, STest3_2)
{
    auto g = BuildPreMergeAssembleGraph(true);
    ASSERT_NE(g.function, nullptr);
    auto currFunctionPtr = g.function;
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    status = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_EQ(pass.preregcopys.find(g.assembleOp1), pass.preregcopys.end());
    EXPECT_EQ(pass.preregcopys.find(g.assembleOp2), pass.preregcopys.end());
    EXPECT_NE(pass.preregcopys.find(g.finalAssemble), pass.preregcopys.end());

    status = pass.InsertCopys(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountRegisterCopy(currFunctionPtr), NUM_ONE);
    Operation* copy = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
        }
    }
    EXPECT_NE(copy, nullptr);
    EXPECT_EQ(copy->GetIOperands().front(), g.mergeTensor);
    EXPECT_EQ(g.finalAssemble->GetIOperands().front(), copy->GetOOperands().front());
}

/*
STest3_3
input1->view->T1->assemble----------\
                                      ->merge->assemble->output1
input2->exp->T2->assemble------------/      \
                                             ->assemble->output2
反向遍历从 output2 回到已由 output1 标记过的 merge 时，应在纯视图分支的
pre-merge assemble 前插 copy，而不是在 output2 对应的最终 assemble 前插 copy。
*/
TEST_F(InferMemoryConflictTest, STest3_3)
{
    auto g = BuildPreMergeBackwardAssembleGraph();
    ASSERT_NE(g.function, nullptr);
    auto currFunctionPtr = g.function;
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(g.assembleOp1), pass.preregcopys.end());
    EXPECT_EQ(pass.preregcopys.find(g.assembleOp2), pass.preregcopys.end());
    EXPECT_EQ(pass.preregcopys.find(g.finalAssemble), pass.preregcopys.end());
    EXPECT_EQ(pass.preregcopys.find(g.finalAssemble2), pass.preregcopys.end());

    status = pass.InsertCopys(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountRegisterCopy(currFunctionPtr), NUM_ONE);
    Operation* copy = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
        }
    }
    EXPECT_NE(copy, nullptr);
    EXPECT_EQ(copy->GetIOperands().front(), g.viewTensor);
    EXPECT_EQ(g.assembleOp1->GetIOperands().front(), copy->GetOOperands().front());
    EXPECT_EQ(g.assembleOp2->GetIOperands().front(), g.expTensor);
}

/*
STest3_4
input1->view->T1->atomic_rmw--------\
                                      ->merge->assemble->output
input2->exp->T2->atomic_rmw----------/
OP_ATOMIC_RMW 作为 pre-merge producer 时，纯视图分支仍应在 atomic_rmw 前插 copy。
*/
TEST_F(InferMemoryConflictTest, STest3_4)
{
    auto g = BuildPreMergeAssembleGraph(false, Opcode::OP_ATOMIC_RMW);
    ASSERT_NE(g.function, nullptr);
    auto currFunctionPtr = g.function;
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    status = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(g.assembleOp1->GetOpcode(), Opcode::OP_ATOMIC_RMW);
    EXPECT_EQ(g.assembleOp2->GetOpcode(), Opcode::OP_ATOMIC_RMW);
    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(g.assembleOp1), pass.preregcopys.end());
    EXPECT_EQ(pass.preregcopys.find(g.assembleOp2), pass.preregcopys.end());
    EXPECT_EQ(pass.preregcopys.find(g.finalAssemble), pass.preregcopys.end());

    status = pass.InsertCopys(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountRegisterCopy(currFunctionPtr), NUM_ONE);
    Operation* copy = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
        }
    }
    EXPECT_NE(copy, nullptr);
    EXPECT_EQ(copy->GetIOperands().front(), g.viewTensor);
    EXPECT_EQ(g.assembleOp1->GetIOperands().front(), copy->GetOOperands().front());
    EXPECT_EQ(g.assembleOp2->GetIOperands().front(), g.expTensor);
}

/*
STest4
view->reshape->matmul
优化场景不插入 registery copy
*/
TEST_F(InferMemoryConflictTest, STest4)
{
    PassManager& passManager = PassManager::Instance();
    Tensor in0(DT_FP32, Shape{3, 128, 64}, "in0");
    Tensor in1(DT_FP32, Shape{3, 64, 256}, "in1");
    Tensor out(DT_FP32, Shape{128, 256}, "out");
    TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
    FUNCTION("InferMemoryConflictTest")
    {
        auto a = View(in0, Shape{1, 128, 64}, {0, 0, 0});
        auto b = View(in1, Shape{1, 64, 256}, {0, 0, 0});
        auto a0 = Reshape(a, Shape{128, 64});
        auto b0 = Reshape(b, Shape{64, 256});
        out = Matrix::Matmul(DataType::DT_FP32, a0, b0, false, false);
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_InferMemoryConflictTest");
    int cnt = 0;
    for (auto e : func->Operations().DuplicatedOpList()) {
        if (e->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
    passManager.RegisterStrategy("InferMemoryConflictTestStrategy",
                                 {
                                     {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                 });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "InferMemoryConflictTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    for (auto e : func->Operations().DuplicatedOpList()) {
        if (e->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
}

/*
STest4_4DTo3D
view->reshape(4D->3D)->matmul
raw shape 冲突但 reshape shape 满足 MatchReshapeDimensionPair，优化场景不插入 register copy
*/
TEST_F(InferMemoryConflictTest, STest4_4DTo3D)
{
    ExpectNoRegisterCopyForViewReshapeMatmul({1, NUM_64, NUM_128, NUM_64});
}

/*
STest4_5DTo3D
view->reshape(5D->3D)->matmul
raw shape 冲突但 reshape shape 满足 MatchReshapeDimensionPair，优化场景不插入 register copy
*/
TEST_F(InferMemoryConflictTest, STest4_5DTo3D)
{
    ExpectNoRegisterCopyForViewReshapeMatmul({1, 1, NUM_64, NUM_128, NUM_64});
}

/*
STest5
动态tensor->view->reshape->matmul 需插入 register copy
       t5  -----------------------------------|
                                                +-> A_MUL_B -> o1
t1 -> VIEW -> t2 -> VIEW -> t3 -> RESHAPE -> t4
                                                +-> A_MUL_B -> o2
       t6   ----------------------------------|
*/
TEST_F(InferMemoryConflictTest, STest5)
{
    ComputationalGraphBuilder G;
    // add tensor
    G.AddTensor(DataType::DT_FP32, {-1, -1, 128}, "t1");
    G.AddTensor(DataType::DT_FP32, {-1, -1, 128}, "t2");
    G.AddTensor(DataType::DT_FP32, {1, 64, 128}, "t3");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "t4");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "t5");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "t6");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "o1");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "o2");

    G.AddOp(Opcode::OP_VIEW, {"t1"}, {"t2"}, "V1");
    G.AddOp(Opcode::OP_VIEW, {"t2"}, {"t3"}, "V2");
    G.AddOp(Opcode::OP_RESHAPE, {"t3"}, {"t4"}, "R1");
    G.AddOp(Opcode::OP_A_MUL_B, {"t4", "t5"}, {"o1"}, "MUL1");
    G.AddOp(Opcode::OP_A_MUL_B, {"t4", "t6"}, {"o2"}, "MUL2");

    // set incast and outcast
    G.SetInCast({"t1", "t5", "t6"});
    G.SetOutCast({"o1", "o2"});

    auto currFunctionPtr = G.GetFunction();
    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    int cnt = 0;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            cnt += 1;
        }
    }
    EXPECT_EQ(cnt, NUM_ONE);
}

/*
STest6
NZ格式动态tensor->view->reshape->matmul 两根动轴场景编译失败
       t5  -----------------------------------|
                                                +-> A_MUL_B -> o1
t1 -> VIEW -> t2 -> VIEW -> t3 -> RESHAPE -> t4
                                                +-> A_MUL_B -> o2
       t6   ----------------------------------|
*/
TEST_F(InferMemoryConflictTest, STest6)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, {-1, -1, 128}, "t1");
    G.AddTensor(DataType::DT_FP32, {-1, -1, 128}, "t2");
    G.AddTensor(DataType::DT_FP32, {1, 64, 128}, "t3");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "t4");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "t5");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "t6");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "o1");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "o2");
    G.GetTensor("t1")->tensor->format = TileOpFormat::TILEOP_NZ;

    G.AddOp(Opcode::OP_VIEW, {"t1"}, {"t2"}, "V1");
    G.AddOp(Opcode::OP_VIEW, {"t2"}, {"t3"}, "V2");
    G.AddOp(Opcode::OP_RESHAPE, {"t3"}, {"t4"}, "R1");
    G.AddOp(Opcode::OP_A_MUL_B, {"t4", "t5"}, {"o1"}, "MUL1");
    G.AddOp(Opcode::OP_A_MUL_B, {"t4", "t6"}, {"o2"}, "MUL2");

    G.SetInCast({"t1", "t5", "t6"});
    G.SetOutCast({"o1", "o2"});

    auto currFunctionPtr = G.GetFunction();
    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, FAILED);
}

/*
BMM_OnlyIncastContainsMoreThanOneDyn
incast 为多维动态 shape {-1, -1, 128}，View->Reshape->MatMul 场景
HasNegativeDimAfterFirstCast 返回 true，应插入 register copy
incast 与 outcast 同 shape size (均为 3D)，outcast 仅第0轴为-1

t1(incast{-1,-1,128}) -> VIEW -> t2{1,64,128} -> RESHAPE(R1) -> t3{64,128}
t4{64,64} -----------+-> A_MUL_B(t3,t4) -> t6{64,64} -> RESHAPE(R2) -> t7{1,64,64} -> ASSEMBLE -> o1{-1,64,64}
t5{64,64} -----------+-> A_MUL_B(t3,t5) -> t8{64,64} -> RESHAPE(R3) -> t9{1,64,64} -> ASSEMBLE -> o2{-1,64,64}
*/
TEST_F(InferMemoryConflictTest, BMM_OnlyIncastContainsMoreThanOneDyn)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, {-1, -1, 128}, "t1");
    G.AddTensor(DataType::DT_FP32, {1, 64, 128}, "t2");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "t3");
    G.AddTensor(DataType::DT_FP32, {64, 64}, "t4");
    G.AddTensor(DataType::DT_FP32, {64, 64}, "t5");
    G.AddTensor(DataType::DT_FP32, {64, 64}, "t6");
    G.AddTensor(DataType::DT_FP32, {1, 64, 64}, "t7");
    G.AddTensor(DataType::DT_FP32, {64, 64}, "t8");
    G.AddTensor(DataType::DT_FP32, {1, 64, 64}, "t9");
    G.AddTensor(DataType::DT_FP32, {-1, 64, 64}, "o1");
    G.AddTensor(DataType::DT_FP32, {-1, 64, 64}, "o2");

    G.AddOp(Opcode::OP_VIEW, {"t1"}, {"t2"}, "V1");
    G.AddOp(Opcode::OP_RESHAPE, {"t2"}, {"t3"}, "R1");
    G.AddOp(Opcode::OP_A_MUL_B, {"t3", "t4"}, {"t6"}, "MUL1");
    G.AddOp(Opcode::OP_RESHAPE, {"t6"}, {"t7"}, "R2");
    G.AddOp(Opcode::OP_ASSEMBLE, {"t7"}, {"o1"}, "ASM1");
    G.AddOp(Opcode::OP_A_MUL_B, {"t3", "t5"}, {"t8"}, "MUL2");
    G.AddOp(Opcode::OP_RESHAPE, {"t8"}, {"t9"}, "R3");
    G.AddOp(Opcode::OP_ASSEMBLE, {"t9"}, {"o2"}, "ASM2");

    G.SetInCast({"t1", "t4", "t5"});
    G.SetOutCast({"o1", "o2"});

    auto currFunctionPtr = G.GetFunction();
    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountRegisterCopy(currFunctionPtr), NUM_3);
}

/*
BMM_OnlyOutcastContainsMoreThanOneDyn
outcast 为多维动态 shape {-1, -1, 64}，MatMul->Reshape->Assemble->outcast 场景
HasNegativeDimAfterFirstCast 返回 true，应插入 register copy
incast 与 outcast 同 shape size (均为 3D)，incast 仅第0轴为-1

t1(incast{-1,64,64}) -> RESHAPE -> t2{64,64}
                                          +-> A_MUL_B -> t5{64,64} -> RESHAPE(R3) -> t6{1,64,64} -> ASSEMBLE ->
o1{-1,-1,64} t3(incast{-1,64,64}) -> RESHAPE -> t4{64,64}
*/
TEST_F(InferMemoryConflictTest, BMM_OnlyOutcastContainsMoreThanOneDyn)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, {-1, 64, 64}, "t1");
    G.AddTensor(DataType::DT_FP32, {64, 64}, "t2");
    G.AddTensor(DataType::DT_FP32, {-1, 64, 64}, "t3");
    G.AddTensor(DataType::DT_FP32, {64, 64}, "t4");
    G.AddTensor(DataType::DT_FP32, {64, 64}, "t5");
    G.AddTensor(DataType::DT_FP32, {1, 64, 64}, "t6");
    G.AddTensor(DataType::DT_FP32, {-1, -1, 64}, "o1");

    G.AddOp(Opcode::OP_RESHAPE, {"t1"}, {"t2"}, "R1");
    G.AddOp(Opcode::OP_RESHAPE, {"t3"}, {"t4"}, "R2");
    G.AddOp(Opcode::OP_A_MUL_B, {"t2", "t4"}, {"t5"}, "MUL1");
    G.AddOp(Opcode::OP_RESHAPE, {"t5"}, {"t6"}, "R3");
    G.AddOp(Opcode::OP_ASSEMBLE, {"t6"}, {"o1"}, "ASM1");

    G.SetInCast({"t1", "t3"});
    G.SetOutCast({"o1"});

    auto currFunctionPtr = G.GetFunction();
    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountRegisterCopy(currFunctionPtr), NUM_3);
}

/*
BMM_IncastAndOutcastOnlyOneDyn
incast 与 outcast 均为 4D shape，仅最高轴为动轴，raw shape 为具体值，
View->Reshape(4D->2D)->MatMul->Reshape(2D->4D)->Assemble 场景。
HasNegativeDimAfterFirstCast 返回 false，4D->2D 和 2D->4D pattern 均匹配，不插入 register copy
incast 与 outcast 同 shape size (均为 4D)

t1(incast{-1,1,64,128}) -> VIEW -> t2{1,1,64,128} -> RESHAPE(R1) -> t3{64,128}
t4{128,64} -------+-> A_MUL_B(t3,t4) -> t6{64,64} -> RESHAPE(R2) -> t7{1,1,64,64} -> ASSEMBLE -> o1{-1,1,64,64}
t5{128,64} -------+-> A_MUL_B(t3,t5) -> t8{64,64} -> RESHAPE(R3) -> t9{1,1,64,64} -> ASSEMBLE -> o2{-1,1,64,64}
*/
TEST_F(InferMemoryConflictTest, BMM_IncastAndOutcastOnlyOneDyn)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, {-1, 1, 64, 128}, "t1");
    G.AddTensor(DataType::DT_FP32, {1, 1, 64, 128}, "t2");
    G.AddTensor(DataType::DT_FP32, {64, 128}, "t3");
    G.AddTensor(DataType::DT_FP32, {128, 64}, "t4");
    G.AddTensor(DataType::DT_FP32, {128, 64}, "t5");
    G.AddTensor(DataType::DT_FP32, {64, 64}, "t6");
    G.AddTensor(DataType::DT_FP32, {1, 1, 64, 64}, "t7");
    G.AddTensor(DataType::DT_FP32, {64, 64}, "t8");
    G.AddTensor(DataType::DT_FP32, {1, 1, 64, 64}, "t9");
    G.AddTensor(DataType::DT_FP32, {-1, 1, 64, 64}, "o1");
    G.AddTensor(DataType::DT_FP32, {-1, 1, 64, 64}, "o2");

    G.GetTensor("t1")->tensor->UpdateRawShape({8, 1, 64, 128});

    G.AddOp(Opcode::OP_VIEW, {"t1"}, {"t2"}, "V1");
    G.AddOp(Opcode::OP_RESHAPE, {"t2"}, {"t3"}, "R1");
    G.AddOp(Opcode::OP_A_MUL_B, {"t3", "t4"}, {"t6"}, "MUL1");
    G.AddOp(Opcode::OP_RESHAPE, {"t6"}, {"t7"}, "R2");
    G.AddOp(Opcode::OP_ASSEMBLE, {"t7"}, {"o1"}, "ASM1");
    G.AddOp(Opcode::OP_A_MUL_B, {"t3", "t5"}, {"t8"}, "MUL2");
    G.AddOp(Opcode::OP_RESHAPE, {"t8"}, {"t9"}, "R3");
    G.AddOp(Opcode::OP_ASSEMBLE, {"t9"}, {"o2"}, "ASM2");

    G.SetInCast({"t1", "t4", "t5"});
    G.SetOutCast({"o1", "o2"});

    auto currFunctionPtr = G.GetFunction();
    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountRegisterCopy(currFunctionPtr), NUM_ZERO);
}
} // namespace tile_fwk
} // namespace npu
