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
 * \file test_duplicate_op.cpp
 * \brief Unit test for DuplicateOp pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "ut_json/ut_json_tool.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/graph_optimization/duplicate_op.h"
#include "passes/pass_check/duplicate_op_checker.h"
#include "symbolic_scalar_test_utils.h"
#include "pass_test_utils.h"

#include "interface/tensor/irbuilder.h"
#define private public
namespace npu {
namespace tile_fwk {
static const size_t kSizeZero = 0UL;
static const size_t kSizeOne = 1UL;
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;
static const uint16_t KNumThree = 3u;
static const uint16_t kNumFour = 4u;
static const uint16_t kNumFive = 5u;
static const uint16_t kNumSix = 6u;
static const uint16_t kNumSeven = 7u;
static const uint16_t kNumEight = 8u;
static const uint16_t kNumEleven = 11u;
static const uint16_t kNumForteen = 14u;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumExpFive = 32u;
static const uint16_t kNumExpSix = 64u;
static const uint16_t kNumExpSeven = 128u;

class TestDuplicateOpPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "DuplicateOpTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

static std::shared_ptr<LogicalTensor> CreateGatherInAuxTensor(int64_t rows = 1)
{
    std::vector<int64_t> shape = {1, rows};
    auto tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT64, shape, CreateTestConstIntVector(shape));
    std::vector<SymbolicScalar> dynValidShape = {IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(rows)};
    tensor->UpdateDynValidShape(dynValidShape);
    return tensor;
}

/*
TESTDuplicateViewSingleConsumer
inCast{8,16}->view->ubTensor{1,8,16}->exp->outCast{1,8,16}

inCast{8,16}->view->ubTensor{1,8,16}->exp->outCast{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateViewUTest1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto& tensorOffset = inCast->GetTensorOffset();
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ubTensor}, [&tensorOffset, &ubTensor](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset(),
                                                                ubTensor->GetDynValidShape()));
        });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t viewNum = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
    }
    EXPECT_EQ(viewNum, kNumOne);
}

/*
TESTDuplicateViewThreeConsumer
inCast{8,16}->view->ubTensor{1,8,16}->exp->outCast1{1,8,16}
                                    ->view->outCast2{8,16}
                                    ->sqrt->outCast3{1,8,16}
inCast{8,16}->view->ubTensor{1,8,16}->view->outCast2{8,16}
            ->view->viewTensor1{1,8,16}->exp->outCast1{1,8,16}
            ->view->viewTensor2{1,8,16}->sqrt->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateViewUTest2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(kNumEight),
                                 IRBuilder().CreateConstInt(kNumExpFour)});
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(kNumEight),
                                   IRBuilder().CreateConstInt(kNumExpFour)});
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    outCast2->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(kNumEight),
                                   IRBuilder().CreateConstInt(kNumExpFour)});
    auto outCast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto& tensorOffset = inCast->GetTensorOffset();
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ubTensor}, [&tensorOffset, &ubTensor](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset(),
                                                                ubTensor->GetDynValidShape()));
        });
    auto& tensorOffset1 = ubTensor->GetTensorOffset();
    auto& viewOp1 = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {ubTensor}, {outCast2}, [&tensorOffset1, &outCast2](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(),
                                                                outCast2->GetDynValidShape()));
        });
    auto& sqrtOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor}, {outCast3});
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor}, {outCast1});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);
    currFunctionPtr->outCasts_.push_back(outCast1);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t viewNum = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
    }
    // 旧的两个 + 新的两个
    EXPECT_EQ(viewNum, kNumFour);
    EXPECT_EQ(expOp.GetInputOperandSize(), kNumOne);
    EXPECT_NE(expOp.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(viewOp1.GetInputOperandSize(), kNumOne);
    EXPECT_EQ(viewOp1.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(sqrtOp.GetInputOperandSize(), kNumOne);
    EXPECT_NE(sqrtOp.GetInputOperand(kSizeZero), ubTensor);
}

/*
TESTDuplicateSliceTwoConsumer
inCast{8,16}->slice->sliceTensor{4,16}->exp->outCast1{4,16}
                                      ->sqrt->outCast2{4,16}

inCast{8,16}->slice->sliceTensor1{4,16}->exp->outCast1{4,16}
            ->slice->sliceTensor2{4,16}->sqrt->outCast2{4,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateSliceUTest1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateSlice",
                                                      "TestDuplicateSlice", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inputShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> sliceShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto inCast = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto sliceTensor = IRBuilder().CreateTensorVar(DT_FP32, sliceShape, CreateTestConstIntVector(sliceShape));
    auto outCast1 = IRBuilder().CreateTensorVar(DT_FP32, sliceShape, CreateTestConstIntVector(sliceShape));
    auto outCast2 = IRBuilder().CreateTensorVar(DT_FP32, sliceShape, CreateTestConstIntVector(sliceShape));

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_SLICE, {inCast}, {sliceTensor}, [&offset](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MEM_VECTOR_REG, std::vector<SymbolicScalar>{}));
        });
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {sliceTensor}, {outCast1});
    auto& sqrtOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {sliceTensor}, {outCast2});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_SLICE), kNumTwo);
    EXPECT_EQ(expOp.GetInputOperandSize(), kNumOne);
    EXPECT_EQ(sqrtOp.GetInputOperandSize(), kNumOne);
    EXPECT_NE(expOp.GetInputOperand(kSizeZero), sliceTensor);
    EXPECT_NE(sqrtOp.GetInputOperand(kSizeZero), sliceTensor);
    EXPECT_NE(expOp.GetInputOperand(kSizeZero), sqrtOp.GetInputOperand(kSizeZero));

    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        auto sliceAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        ASSERT_NE(sliceAttr, nullptr);
        EXPECT_EQ(sliceAttr->GetFromOffset(), offset);
        EXPECT_EQ(sliceAttr->GetTo(), MEM_VECTOR_REG);
    }
}

/*
TESTDuplicateViewAlternativeConsumer
inCast{8,16}->view->ubTensor1{1,8,16}
            ->view->ubTensor2{1,8,16}
ubTensor1+ubTensor1->div->outCast1{1,8,16}
ubTensor1+ubTensor2->div->outCast2{1,8,16}
ubTensor2+ubTensor2->div->outCast3{1,8,16}

inCast{8,16}->view->ubTensor1'{1,8,16}
            ->view->ubTensor2'{1,8,16}
            ->view->ubTensor3'{1,8,16}
            ->view->ubTensor4'{1,8,16}
            ->view->ubTensor5'{1,8,16}
            ->view->ubTensor6'{1,8,16}
ubTensor1'+ubTensor2'->div->outCast1{1,8,16}
ubTensor3'+ubTensor4'->div->outCast2{1,8,16}
ubTensor5'+ubTensor6'->div->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateViewUTest3)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(kNumEight),
                                 IRBuilder().CreateConstInt(kNumExpFour)});
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor1->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(kNumEight),
                                    IRBuilder().CreateConstInt(kNumExpFour)});
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor2->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(kNumEight),
                                    IRBuilder().CreateConstInt(kNumExpFour)});
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto& tensorOffset = inCast->GetTensorOffset();
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ubTensor1}, [&tensorOffset, &ubTensor1](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset(),
                                                                ubTensor1->GetDynValidShape()));
        });
    auto& tensorOffset1 = inCast->GetTensorOffset();
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ubTensor2}, [&tensorOffset1, &ubTensor2](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(),
                                                                ubTensor2->GetDynValidShape()));
        });
    auto& div1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_DIV, {ubTensor1, ubTensor1}, {outCast1});
    auto& div2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_DIV, {ubTensor1, ubTensor2}, {outCast2});
    auto& div3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_DIV, {ubTensor2, ubTensor2}, {outCast3});
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast3);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast1);
    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);
    uint32_t viewNum = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
    }
    EXPECT_EQ(viewNum, kNumFour);
    EXPECT_EQ(div1.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(div2.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(div3.GetInputOperandSize(), kNumTwo);
    auto div1Input1 = div1.GetInputOperand(kSizeZero);
    auto div1Input2 = div1.GetInputOperand(kSizeOne);
    EXPECT_NE(div1Input1, ubTensor1);
    EXPECT_EQ(div1Input1, div1Input2);
    auto div2Input1 = div2.GetInputOperand(kSizeZero);
    auto div2Input2 = div2.GetInputOperand(kSizeOne);
    EXPECT_NE(div2Input2, ubTensor2);
    EXPECT_NE(div2Input1, div2Input2);
    auto div3Input1 = div3.GetInputOperand(kSizeZero);
    auto div3Input2 = div3.GetInputOperand(kSizeOne);
    EXPECT_NE(div3Input2, ubTensor2);
    EXPECT_EQ(div3Input1, div3Input2);
}

/*
incast    ->view  -> tensor1 -> exp  -> outcast1
                             -> exp  -> outcast2
incast    ->view  -> tensor1 -> exp  -> outcast1
                             -> exp  -> outcast2
*/
TEST_F(TestDuplicateOpPass, TestDupViewL1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDupViewL1", "TestDupViewL1",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto viewAttr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MEM_L1, std::vector<SymbolicScalar>(),
                                                      std::vector<SymbolicScalar>());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {tensor1},
                                     [&viewAttr](Operation& op) { op.SetOpAttribute(viewAttr); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {outcast1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {outcast2});
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);
    DuplicateOp duplicateoppass;
    duplicateoppass.RunOnFunction(*currFunctionPtr);
    int viewNum = 0;
    auto opList = currFunctionPtr->Operations();
    for (auto& op : opList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            viewNum++;
        }
    }
    EXPECT_EQ(viewNum, 1);
}

/*
TESTDuplicateGatherinSingleConsumer
inCast{8,16}->Gatherin->ubTensor{8,16}->exp->outCast{8,16}

inCast{8,16}->Gatherin->ubTensor{8,16}->exp->outCast{8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->UpdateDynValidShape({IRBuilder().CreateConstInt(kNumEight), IRBuilder().CreateConstInt(kNumExpFour)});
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto offsets = CreateGatherInAuxTensor(kNumEight);
    auto blockTable = CreateGatherInAuxTensor(kNumOne);

    int64_t startOffset = 0;
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_GATHER_IN_L1, {inCast, offsets, blockTable}, {ubTensor},
        [&startOffset](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, startOffset); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t gatherinNum = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherinNum;
        }
    }
    EXPECT_EQ(gatherinNum, kNumOne);
}

/*
TESTDuplicateGatherinThreeConsumer
inCast{8,16}->Gatherin->ubTensor{1,8,16}->exp->outCast1{1,8,16}
                                    ->view->outCast2{8,16}
                                    ->sqrt->outCast3{1,8,16}
inCast{8,16}->Gatherin->ubTensor{1,8,16}->view->outCast2{8,16}
            ->Gatherin->GatherinTensor1{1,8,16}->exp->outCast1{1,8,16}
            ->Gatherin->GatherinTensor2{1,8,16}->sqrt->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    int64_t j = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->UpdateDynValidShape({IRBuilder().CreateConstInt(kNumEight), IRBuilder().CreateConstInt(kNumExpFour)});
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto outCast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto offsets = CreateGatherInAuxTensor(kNumEight);
    auto blockTable = CreateGatherInAuxTensor(kNumOne);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_GATHER_IN_L1, {inCast, offsets, blockTable},
                                     {ubTensor},
                                     [&j](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, j); });
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor}, {outCast1});
    auto& tensorOffset1 = ubTensor->GetTensorOffset();
    auto& viewOp = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {ubTensor}, {outCast2}, [&tensorOffset1, &outCast2](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(),
                                                                outCast2->GetDynValidShape()));
        });
    auto& sqrtOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t gatherinnum = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherinnum;
        }
    }
    // 旧的两个 + 新的两个
    EXPECT_EQ(gatherinnum, KNumThree);
    EXPECT_EQ(expOp.GetInputOperandSize(), kNumOne);
    EXPECT_EQ(expOp.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(viewOp.GetInputOperandSize(), kNumOne);
    EXPECT_NE(viewOp.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(sqrtOp.GetInputOperandSize(), kNumOne);
    EXPECT_NE(sqrtOp.GetInputOperand(kSizeZero), ubTensor);
}

/*
TESTDuplicateViewAlternativeConsumer
inCast{8,16}->gatherin->ubTensor1{1,8,16}
            ->gatherin->ubTensor2{1,8,16}
ubTensor1+ubTensor1->div->outCast1{1,8,16}
ubTensor1+ubTensor2->div->outCast2{1,8,16}
ubTensor2+ubTensor2->div->outCast3{1,8,16}

inCast{8,16}->gatherin->ubTensor1'{1,8,16}
            ->gatherin->ubTensor2'{1,8,16}
            ->gatherin->ubTensor3'{1,8,16}
            ->gatherin->ubTensor4'{1,8,16}
ubTensor1'+ubTensor1'->div->outCast1{1,8,16}
ubTensor2'+ubTensor3'->div->outCast2{1,8,16}
ubTensor4'+ubTensor4'->div->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest3)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    int64_t j = 1;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->UpdateDynValidShape({IRBuilder().CreateConstInt(kNumEight), IRBuilder().CreateConstInt(kNumExpFour)});
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto offsets1 = CreateGatherInAuxTensor(kNumEight);
    auto blockTable1 = CreateGatherInAuxTensor(kNumOne);
    auto offsets2 = CreateGatherInAuxTensor(kNumEight);
    auto blockTable2 = CreateGatherInAuxTensor(kNumOne);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_GATHER_IN_L1, {inCast, offsets1, blockTable1},
                                     {ubTensor1},
                                     [&i](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, i); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_GATHER_IN_L1, {inCast, offsets2, blockTable2},
                                     {ubTensor2},
                                     [&j](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, j); });
    auto& divOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_DIV, {ubTensor1, ubTensor1},
                                                    {outCast1});
    auto& divOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_DIV, {ubTensor1, ubTensor2},
                                                    {outCast2});
    auto& divOp3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_DIV, {ubTensor2, ubTensor2},
                                                    {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t gatherinNum = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherinNum;
        }
    }
    EXPECT_EQ(gatherinNum, kNumFour);
    EXPECT_EQ(divOp1.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(divOp2.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(divOp3.GetInputOperandSize(), kNumTwo);

    auto div1Input1 = divOp1.GetInputOperand(kSizeZero);
    auto div1Input2 = divOp1.GetInputOperand(kSizeOne);
    EXPECT_EQ(div1Input2, ubTensor1);
    EXPECT_EQ(div1Input1, div1Input2);

    auto div2Input1 = divOp2.GetInputOperand(kSizeZero);
    auto div2Input2 = divOp2.GetInputOperand(kSizeOne);
    EXPECT_NE(div2Input1, ubTensor1);
    EXPECT_NE(div2Input1, div2Input2);

    auto div3Input1 = divOp3.GetInputOperand(kSizeZero);
    auto div3Input2 = divOp3.GetInputOperand(kSizeOne);
    EXPECT_NE(div3Input2, ubTensor2);
    EXPECT_EQ(div3Input1, div3Input2);
}

/*
TESTDuplicateGatherinConsumerGatherin(ERROR)
inCast{8,16}->Gatherin->ubTensor{1,8,16}->Gatherin->outCast{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest4)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->UpdateDynValidShape({IRBuilder().CreateConstInt(kNumEight), IRBuilder().CreateConstInt(kNumExpFour)});
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto offsets1 = CreateGatherInAuxTensor(kNumEight);
    auto blockTable1 = CreateGatherInAuxTensor(kNumOne);
    auto offsets2 = CreateGatherInAuxTensor(kNumEight);
    auto blockTable2 = CreateGatherInAuxTensor(kNumOne);

    int64_t startOffset1 = 0;
    int64_t startOffset2 = 0;
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_GATHER_IN_L1, {inCast, offsets1, blockTable1}, {ubTensor},
        [&startOffset1](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, startOffset1); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_GATHER_IN_L1, {ubTensor, offsets2, blockTable2}, {outCast},
        [&startOffset2](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, startOffset2); });

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    DuplicateOp duplicateoppass;
    EXPECT_NE(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
}

/*
TESTDuplicateGatherinViewnNormal
inCast{8,16}->Gatherin->ubTensor1{8,16}->exp->outCast1{8,16}
                      ->view->ubtensor2 ->exp->outcast2
                                        ->view->outcast3
inCast{8,16}->Gatherin->ubTensor1{8,16}->exp->outCast1{8,16}
            ->Gatherin->ubTensor1'{8,16}->view->ubtensor2 ->exp->outcast2
            ->Gatherin->ubTensor1'{8,16}->view->ubtensor2 ->view->outcast3
*/
TEST_F(TestDuplicateOpPass, DuplicateViewGatherInUTest1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->UpdateDynValidShape({IRBuilder().CreateConstInt(kNumEight), IRBuilder().CreateConstInt(kNumExpFour)});
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto offsets = CreateGatherInAuxTensor(kNumEight);
    auto blockTable = CreateGatherInAuxTensor(kNumOne);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_GATHER_IN_L1, {inCast, offsets, blockTable},
                                     {ubTensor1},
                                     [&i](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, i); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor1}, {outCast1});
    auto& tensorOffset1 = ubTensor1->GetTensorOffset();
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {ubTensor1}, {ubTensor2}, [&tensorOffset1, &ubTensor2](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(),
                                                                ubTensor2->GetDynValidShape()));
        });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor2}, {outCast2});
    auto& tensorOffset2 = ubTensor2->GetTensorOffset();
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {ubTensor2}, {outCast3}, [&tensorOffset2, &outCast3](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset2.GetOffset(), tensorOffset2.GetDynOffset(),
                                                                outCast3->GetDynValidShape()));
        });
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_GATHER_IN_L1), KNumThree);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_VIEW), KNumThree);
}

/*
TESTDuplicateGatherinView(special1)
inCast{8,16}->Gatherin->ubtensor0 ->view->ubtensor1 ->gatherIn->ubtensor2->sqrt->outcast1
                                                             ->exp->outcast2
inCast{8,16}->Gatherin->ubtensor0 ->view->ubtensor1 ->gatherIn->ubtensor2->sqrt->outcast1
            ->Gatherin->ubtensor0'->view->ubtensor1'->gatherIn->ubtensor2'->exp->outcast2
*/
TEST_F(TestDuplicateOpPass, DuplicateViewGatherInUTest2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    int64_t j = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->UpdateDynValidShape({IRBuilder().CreateConstInt(kNumEight), IRBuilder().CreateConstInt(kNumExpFour)});
    auto ubTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto offsets1 = CreateGatherInAuxTensor(kNumEight);
    auto blockTable1 = CreateGatherInAuxTensor(kNumOne);
    auto offsets2 = CreateGatherInAuxTensor(kNumEight);
    auto blockTable2 = CreateGatherInAuxTensor(kNumOne);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_GATHER_IN_L1, {inCast, offsets1, blockTable1},
                                     {ubTensor0},
                                     [&i](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, i); });
    auto& tensorOffset1 = ubTensor0->GetTensorOffset();
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {ubTensor0}, {ubTensor1}, [&tensorOffset1, &ubTensor1](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(),
                                                                ubTensor1->GetDynValidShape()));
        });
    ubTensor1->UpdateDynValidShape({IRBuilder().CreateConstInt(kNumEight), IRBuilder().CreateConstInt(kNumExpFour)});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_GATHER_IN_L1, {ubTensor1, offsets2, blockTable2},
                                     {ubTensor2},
                                     [&j](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, j); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor2}, {outCast1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor2}, {outCast2});
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast2);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_GATHER_IN_L1), kNumFour);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_VIEW), kNumTwo);
}

/*
TESTDuplicateGatherinView(special2)
inCast{8,16}->View->ubtensor0 ->GatherIn->ubtensor1 ->View->ubtensor2->sqrt->outcast1
                                                                      ->exp->outcast2
inCast{8,16}->View->ubtensor0 ->GatherIn->ubtensor1 ->View->ubtensor2->sqrt->outcast1
            ->View->ubtensor0' ->GatherIn->ubtensor1' ->View->ubtensor2'->exp->outcast2
*/
TEST_F(TestDuplicateOpPass, DuplicateViewGatherInUTest3)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->UpdateDynValidShape({IRBuilder().CreateConstInt(kNumEight), IRBuilder().CreateConstInt(kNumExpFour)});
    auto ubTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto offsets = CreateGatherInAuxTensor(kNumEight);
    auto blockTable = CreateGatherInAuxTensor(kNumOne);

    auto& tensorOffset1 = inCast->GetTensorOffset();
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ubTensor0}, [&tensorOffset1, &ubTensor0](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(),
                                                                ubTensor0->GetDynValidShape()));
        });
    ubTensor0->UpdateDynValidShape({IRBuilder().CreateConstInt(kNumEight), IRBuilder().CreateConstInt(kNumExpFour)});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_GATHER_IN_L1, {ubTensor0, offsets, blockTable},
                                     {ubTensor1},
                                     [&i](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, i); });
    auto& tensorOffset2 = ubTensor1->GetTensorOffset();
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {ubTensor1}, {ubTensor2},
        [&tensorOffset1, &tensorOffset2, &ubTensor2](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset2.GetDynOffset(),
                                                                ubTensor2->GetDynValidShape()));
        });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor2}, {outCast2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor2}, {outCast1});
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->inCasts_.push_back(inCast);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_GATHER_IN_L1), kNumTwo);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_VIEW), kNumFour);
}

/*
incast    ->view  -> tensor1 -> sqrt  -> tensor1 -> gatherIn -> tensor3 -> exp -> Outcast2
                             -> sqrt  -> Outcast1                       -> exp -> Outcast3
incast    ->view  -> tensor1 -> sqrt  -> tensor1 -> gatherIn -> tensor3 -> exp -> Outcast2
                             -> sqrt  -> Outcast1   gatherIn -> tensor4 -> exp -> Outcast3
*/
TEST_F(TestDuplicateOpPass, TestCheck1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPostCheck", "TestPostCheck",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    int64_t i = 0;
    std::vector<int64_t> shape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    incast->UpdateDynValidShape({IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(16)});
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->UpdateDynValidShape({IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(16)});
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->UpdateDynValidShape({IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(16)});
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->UpdateDynValidShape({IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(16)});
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto offsets = CreateGatherInAuxTensor(8);
    auto blockTable = CreateGatherInAuxTensor(1);

    auto viewAttr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MEM_L1, std::vector<SymbolicScalar>(),
                                                      std::vector<SymbolicScalar>());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {tensor1},
                                     [&viewAttr](Operation& op) { op.SetOpAttribute(viewAttr); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor3}, {outcast2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor3}, {outcast3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {tensor1}, {tensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {tensor1}, {outcast1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_GATHER_IN_L1, {tensor2, offsets, blockTable},
                                     {tensor3},
                                     [&i](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, i); });
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);
    currFunctionPtr->outCasts_.push_back(outcast3);
    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_GATHER_IN_L1), kNumTwo);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_VIEW), kNumOne);
}

/*
incast    ->gatherIn  -> tensor1 -> sqrt  -> tensor2 -> view -> tensor3 -> exp -> Outcast2
                      -> sqrt  -> Outcast1                               -> exp -> Outcast3
incast    ->gatherIn  -> tensor1 -> sqrt  -> tensor2 -> view -> tensor3 -> exp -> Outcast2
            gatherIn  -> tensor1'-> sqrt  -> Outcast1   view -> tensor4 -> exp -> Outcast3
*/
TEST_F(TestDuplicateOpPass, TestCheck2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPostCheck", "TestPostCheck",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    int64_t i = 0;
    std::vector<int64_t> shape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    incast->UpdateDynValidShape({IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(16)});
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->UpdateDynValidShape({IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(16)});
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->UpdateDynValidShape({IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(16)});
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->UpdateDynValidShape({IRBuilder().CreateConstInt(8), IRBuilder().CreateConstInt(16)});

    auto offsets = CreateGatherInAuxTensor(8);
    auto blockTable = CreateGatherInAuxTensor(1);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_GATHER_IN_L1, {incast, offsets, blockTable},
                                     {tensor1},
                                     [&i](Operation& op) { op.SetAttribute(OpAttributeKey::startOffset, i); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor3}, {outcast2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor3}, {outcast3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {tensor1}, {tensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {tensor1}, {outcast1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MEM_VECTOR_REG,
                                                      std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {tensor2}, {tensor3},
                                     [&viewAttr](Operation& op) { op.SetOpAttribute(viewAttr); });
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);
    currFunctionPtr->outCasts_.push_back(outcast3);
    currFunctionPtr->inCasts_.push_back(incast);
    DuplicateOp duplicateoppass;
    EXPECT_NE(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_GATHER_IN_L1), kNumTwo);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_VIEW), kNumTwo);
}

/* ERROR
incast    ->view  -> tensor1 -> exp - >output1
*/
TEST_F(TestDuplicateOpPass, TestCheck3)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPostCheck", "TestPostCheck",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {outcast1});
    currFunctionPtr->inCasts_.push_back(incast);
    DuplicateOp duplicateoppass;
    EXPECT_NE(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_NE(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);
}

/* ERROR
incast    ->view  -> tensor1 -> exp - >output1
                             -> exp - >output2
*/
TEST_F(TestDuplicateOpPass, TestCheck4)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPostCheck", "TestPostCheck",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto& tensorOffset = incast->GetTensorOffset();
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {incast}, {tensor1}, [&tensorOffset, &tensor1](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset(),
                                                                tensor1->GetDynValidShape()));
        });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {outcast1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {outcast2});
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast2);
    currFunctionPtr->outCasts_.push_back(outcast1);
    DuplicateOp duplicateoppass;
    EXPECT_NE(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(TestDuplicateOpPass, GatherIn_OOperandNull)
{
    PROGRAM("GenerateMoveOpPassTest")
    {
        std::vector<int64_t> shape1{256, 256};
        std::vector<int64_t> shape2{128, 128};
        TileShape::Current().SetVecTile({128, 128});

        Tensor input_1(DT_FP32, shape1, "input_1");
        Tensor output(DT_FP32, shape2, "output");

        Function* originFunction = nullptr;
        config::SetBuildStatic(true);

        FUNCTION("VIEW", {input_1, output})
        {
            auto tmp_view = View(input_1, shape2, {0, 0});
            output = tmp_view;
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_VIEW");
        ASSERT_NE(originFunction, nullptr);
        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                op.SetOpCode(Opcode::OP_GATHER_IN_L1);
                auto& outputs = op.GetOOperands();
                if (!outputs.empty()) {
                    outputs[0] = nullptr;
                }
            }
        }
        DuplicateOpChecker duplicatePass;
        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
                Status ret = duplicatePass.PreCheckGatherIn(op);
                EXPECT_EQ(ret, FAILED);
            }
        }
    }
}

/*
TestSpan
inCast{16,16}->gatherIn->ubTensor{16,16}->sqrt->outCast1{16,16}
                                        ->sqrt->outCast2{16,16}
inCast{16,16}->gatherIn->ubTensor{16,16}->sqrt->outCast1{16,16}
            ->gatherIn->ubTensor'{16,16}->sqrt->outCast2{16,16}
*/
TEST_F(TestDuplicateOpPass, TestSpan)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    // Prepare the graph
    std::vector<int64_t> shape = {kNumExpFour, kNumExpFour};
    auto span = ir::Span("gatherIn", kNumOne, 0);
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inCast->UpdateDynValidShape({IRBuilder().CreateConstInt(kNumExpFour), IRBuilder().CreateConstInt(kNumExpFour)});
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto offsets = CreateGatherInAuxTensor(kNumExpFour);
    auto blockTable = CreateGatherInAuxTensor(kNumOne);

    auto& gatherIn = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_GATHER_IN_L1,
                                                    {inCast, offsets, blockTable}, {ubTensor}, span);
    gatherIn.SetAttribute(OpAttributeKey::startOffset, i);
    gatherIn.SetScopeId(1);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor}, {outCast1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor}, {outCast2});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);

    DuplicateOp duplicateOpPass;
    EXPECT_EQ(duplicateOpPass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateOpPass.PostCheck(*currFunctionPtr), SUCCESS);

    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            EXPECT_EQ(op.GetSpan().BeginLine(), kNumOne);
            EXPECT_EQ(op.GetScopeId(), 1);
        }
    }
}
} // namespace tile_fwk
} // namespace npu
