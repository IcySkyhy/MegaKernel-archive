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
 * \file test_pass_check.cpp
 * \brief Unit test for pass checkers.
 */

#include "gtest/gtest.h"
#include <algorithm>
#include "tilefwk/tilefwk.h"
#include "symbolic_scalar_test_utils.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "interface/inner/tilefwk.h"
#include "interface/inner/tile_shape.h"
#include "interface/function/function.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "passes/pass_check/checker.h"
#include "passes/pass_check/pre_graph_checker.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"

using namespace npu::tile_fwk;
using namespace std;

class PassCheckTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(PassCheckTest, TestCheckCompletenessWithNullIncast)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest",
                                                      "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    currFunctionPtr->inCasts_.push_back(nullptr);
    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompletenessWithIncastHasNoConsumer)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest",
                                                      "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    currFunctionPtr->inCasts_.push_back(incast1);
    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompletenessWithoutOutcast)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "PassCheckTest", "PassCheckTest",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto& exp_op0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {incast1}, {tensor1});
    exp_op0.UpdateSubgraphID(0);

    currFunctionPtr->inCasts_.push_back(incast1);
    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompletenessWithNullOutcast)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest",
                                                      "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast1}, {tensor1});
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(nullptr);
    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompletenessWithOutcastHasNoProducer)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest",
                                                      "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast1}, {tensor1});
    currFunctionPtr->inCasts_.push_back(incast1);
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), SUCCESS);
}

TEST_F(PassCheckTest, TestCheckGraphLoop)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest",
                                                      "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {incast1}, {tensor1});
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {tensor2, outcast1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor2}, {tensor1});

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckGraphLoop(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestPublicCheck)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest",
                                                      "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "");
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {incast1}, {tensor1});
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                               TileOpFormat::TILEOP_ND, "");
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {tensor1}, {outcast1});

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.PublicCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(PassCheckTest, TestCheckDynAttrForView)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest",
                                                      "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto& tensorOffset = incast1->GetTensorOffset();
    auto viewAttr = std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast1}, {outcast1},
                                     [&viewAttr](Operation& op) { op.SetOpAttribute(viewAttr); });

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckDynAttrForView(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckDynAttrForViewWithoutToDynValidShape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest",
                                                      "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {16, 16};

    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    Offset offset = {0, 128};
    std::vector<SymbolicScalar> dynOffset = {0, 128};
    auto viewAttr = std::make_shared<ViewOpAttribute>(offset, dynOffset, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast1}, {outcast1},
                                     [&viewAttr](Operation& op) { op.SetOpAttribute(viewAttr); });
    viewAttr->SetToDynValidShape({});

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckDynAttrForView(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckToDynOffsetForAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest",
                                                      "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto& tensorOffset = incast1->GetTensorOffset();
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(tensorOffset.GetOffset());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {incast1}, {outcast1},
                                     [&assembleAttr](Operation& op) { op.SetOpAttribute(assembleAttr); });

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckToDynOffsetForAssemble(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestPreGraphCheckerAssembleViewReshapeInvalidIO)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "PreGraphCheckerTest",
                                                      "PreGraphCheckerTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast1, incast2}, {outcast1});
    currFunctionPtr->Operations().back().UpdateSubgraphID(0);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outcast1);

    PreGraphProcessChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestPreGraphCheckerTensorNotInSubgraph)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "PreGraphCheckerTest2",
                                                      "PreGraphCheckerTest2", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {incast1, incast2}, {tensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {tensor1, incast2}, {outcast1});
    currFunctionPtr->Operations()[0].UpdateSubgraphID(0);
    currFunctionPtr->Operations()[1].UpdateSubgraphID(1);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->SetTotalSubGraphCount(2);

    PreGraphProcessChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(checker.DoPostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(PassCheckTest, TestCheckConsumerProducer_ProducerIsNull)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestCheckConsumerProducer_ProducerIsNull",
                                                      "TestCheckConsumerProducer_ProducerIsNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    tensor->GetProducers().insert(nullptr);

    Checker checker;
    EXPECT_EQ(checker.CheckConsumerProducer(tensor), FAILED);
}

TEST_F(PassCheckTest, TestCheckConsumerProducer_ConsumerIsNull)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestCheckConsumerProducer_ConsumerIsNull",
                                                      "TestCheckConsumerProducer_ConsumerIsNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    tensor->GetConsumers().insert(nullptr);

    Checker checker;
    EXPECT_EQ(checker.CheckConsumerProducer(tensor), FAILED);
}

TEST_F(PassCheckTest, TestCheckOpIOValid_OutputIsNull)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCheckOpIOValid_OutputIsNull",
                                                      "TestCheckOpIOValid_OutputIsNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "");
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "");
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                               TileOpFormat::TILEOP_ND, "");

    auto& addOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {incast1, incast2}, {outcast1});
    addOp.oOperand[0] = nullptr;
    (void)addOp;

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckOpIOValid(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompleteness_IncastIsNull)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCheckCompleteness_IncastIsNull",
                                                      "TestCheckCompleteness_IncastIsNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                               TileOpFormat::TILEOP_ND, "");

    currFunctionPtr->inCasts_.push_back(nullptr);
    currFunctionPtr->outCasts_.push_back(outcast1);

    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompleteness_OutcastEmpty)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCheckCompleteness_OutcastEmpty",
                                                      "TestCheckCompleteness_OutcastEmpty", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "");

    currFunctionPtr->inCasts_.push_back(incast1);

    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompleteness_OutcastIsNull)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCheckCompleteness_OutcastIsNull",
                                                      "TestCheckCompleteness_OutcastIsNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "");

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(nullptr);

    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckGraphLoop_HasLoop)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCheckGraphLoop_HasLoop",
                                                      "TestCheckGraphLoop_HasLoop", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "");
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {incast1, tensor2}, {tensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {tensor1, incast1}, {tensor2});

    currFunctionPtr->inCasts_.push_back(incast1);
    Checker checker;
    EXPECT_EQ(checker.CheckGraphLoop(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckDynAttrForView_FromDynOffsetEmpty)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestCheckDynAttrForView_FromDynOffsetEmpty",
                                                      "TestCheckDynAttrForView_FromDynOffsetEmpty", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "");
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                               TileOpFormat::TILEOP_ND, "");
    auto viewAttr = std::make_shared<ViewOpAttribute>(Offset(0, 0), CreateTestConstIntVector(shape),
                                                      CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast1}, {outcast1},
                                     [&viewAttr](Operation& op) { op.SetOpAttribute(viewAttr); });
    viewAttr->GetFromDynOffset().clear();

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckDynAttrForView(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckDynAttrForView_ToDynValidShapeEmpty)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestCheckDynAttrForView_ToDynValidShapeEmpty",
                                                      "TestCheckDynAttrForView_ToDynValidShapeEmpty", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "");
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                               TileOpFormat::TILEOP_ND, "");

    auto viewAttr = std::make_shared<ViewOpAttribute>(Offset(0, 0), CreateTestConstIntVector(shape),
                                                      CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast1}, {outcast1},
                                     [&viewAttr](Operation& op) { op.SetOpAttribute(viewAttr); });
    viewAttr->GetToDynValidShape().clear();

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckDynAttrForView(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckToDynOffsetForAssemble_ToDynOffsetEmpty)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestCheckToDynOffsetForAssemble_ToDynOffsetEmpty",
                                                      "TestCheckToDynOffsetForAssemble_ToDynOffsetEmpty", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "");
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                               TileOpFormat::TILEOP_ND, "");

    auto& op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {incast1}, {outcast1});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(Offset(0, 0));
    assembleAttr->GetToDynOffset().clear();
    op.SetOpAttribute(assembleAttr);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckToDynOffsetForAssemble(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckLocalTensor_LocalInput)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCheckLocalTensor_LocalInput",
                                                      "TestCheckLocalTensor_LocalInput", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};

    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "");
    auto localTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                               TileOpFormat::TILEOP_ND, "");
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {incast1, localTensor}, {outcast1});

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckLocalTensor(*currFunctionPtr), FAILED);
}

// Some computation graphs may not include incast because they contain operations such as VEC_DUP.
TEST_F(PassCheckTest, TestPublicCheck_IncastEmpty)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPublicCheck_IncastEmpty",
                                                      "TestPublicCheck_IncastEmpty", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                               TileOpFormat::TILEOP_ND, "");

    currFunctionPtr->outCasts_.push_back(outcast1);

    Checker checker;
    EXPECT_EQ(checker.PublicCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(PassCheckTest, TestDefaultCheckItems)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDefaultCheckItems",
                                                      "TestDefaultCheckItems", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Checker checker;
    EXPECT_EQ(checker.DoDefaultEnabledPreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(checker.DoDefaultEnabledPostCheck(*currFunctionPtr), SUCCESS);
}
