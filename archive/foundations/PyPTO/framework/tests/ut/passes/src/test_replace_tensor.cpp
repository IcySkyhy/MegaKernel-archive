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
 * \file test_replace_tensor.cpp
 * \brief Unit test for ReplaceTensor pass.
 */

#include "gtest/gtest.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_constraint/replace_tensor.h"
#include <fstream>
#include <vector>
#include <string>
#include "computational_graph_builder.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
static const uint32_t kNumZero = 0u;
static const uint32_t kNumOne = 1u;
static const uint32_t kNumTwo = 2u;
static const uint32_t kNumThree = 3u;
static const uint32_t kNumFour = 4u;
static const uint32_t kNumSix = 6u;
static const uint32_t kNumEight = 8u;
static const uint32_t kNumTwelve = 12u;
static const uint32_t kNumSixteen = 16u;

class ReplaceTensorTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(ReplaceTensorTest, TestViewAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewAssemble", "TestViewAssemble",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                             CreateTestConstIntVector(shape));
    auto viewOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor0, offset0, shape1,
                                                               CreateTestConstIntVector(shape1));
    auto viewOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor1, offset1, shape1,
                                                               CreateTestConstIntVector(shape1));
    auto copyOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto copyOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto assOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor0, offset0, shape1,
                                                              CreateTestConstIntVector(shape1));
    auto assOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor1, offset1, shape1,
                                                              CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, shape,
                                                              CreateTestConstIntVector(shape));
    /*       Init Graph
                /————> view0 ————> copy ————> assemble \
        incast -                                        - outcast
                \————> view1 ————> copy ————> assemble /
    */
    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut0});
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {viewOut0}, {copyOut0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {viewOut1}, {copyOut1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyOut0}, {assOut0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyOut1}, {assOut1});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {assOut0}, {outcast});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {assOut1}, {outcast});
    // Init Attribute
    auto viewAttr0 = std::make_shared<ViewOpAttribute>(offset0);
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    auto assAttr0 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0);
    auto assAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    viewOp0.SetOpAttribute(viewAttr0);
    viewOp1.SetOpAttribute(viewAttr1);
    assOp0.SetOpAttribute(assAttr0);
    assOp1.SetOpAttribute(assAttr1);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(incast->GetRawMagic(), viewOut0->GetRawMagic());
    EXPECT_EQ(incast->GetRawMagic(), viewOut1->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), assOut0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), assOut1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshape", "TestReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumOne, kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape1);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto reshape0 = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    auto reshape1 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset1, shape1,
                                                               CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    /* Init Graph
        incast -> CopyIn -> Reshape -> CopyOut -> outCast
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {reshape0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {reshape0}, {reshape1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {reshape1}, {outcast});
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(reshape0->GetRawMagic(), reshape1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestIndexOutCast)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestIndexOutCast", "TestIndexOutCast",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto inTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                                CreateTestConstIntVector(shape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, shape,
                                                              CreateTestConstIntVector(shape));
    /* Init Graph
        incast -> Index_OutCast -> outCast
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_OUTCAST, {inTensor0, inTensor1, inTensor2},
                                     {outcast});
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(inTensor2);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_NE(inTensor2->GetRawMagic(), outcast->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestViewType)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewType", "TestViewType", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumTwo};
    std::vector<int64_t> shape1 = {kNumOne, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_INT8, shape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape1);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT8, shape, CreateTestConstIntVector(shape));
    auto viewType0 = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                                CreateTestConstIntVector(shape));
    auto viewType1 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset1, shape1,
                                                                CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    /* Init Graph
        incast -> CopyIn -> ViewType -> CopyOut -> outCast
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {viewType0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW_TYPE, {viewType0}, {viewType1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {viewType1}, {outcast});
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(viewType0->GetRawMagic(), viewType1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestHasSameConsecutive_True)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestHasSameConsecutive_True",
                                                      "TestHasSameConsecutive_True", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {tensor1}, {tensor2});
    auto& viewOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {tensor2}, {tensor3});

    // 设置操作连接
    tensor2->AddConsumer(&viewOp2);

    ReplaceTensor pass;
    bool result = pass.HasSameConsecutive(viewOp1);
    EXPECT_TRUE(result);
}

TEST_F(ReplaceTensorTest, TestHasSameConsecutive_False)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestHasSameConsecutive_False",
                                                      "TestHasSameConsecutive_False", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {tensor1}, {tensor2});
    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {tensor2}, {tensor3});

    // 设置操作连接
    tensor2->AddConsumer(&assembleOp);

    ReplaceTensor pass;
    bool result = pass.HasSameConsecutive(viewOp1);
    EXPECT_FALSE(result);
}

TEST_F(ReplaceTensorTest, TestPreCheck_FailNoSubgraphID)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPreCheck_FailNoSubgraphID",
                                                      "TestPreCheck_FailNoSubgraphID", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {tensor1}, {tensor2});
    // 不设置subgraph ID

    ReplaceTensor pass;
    Status result = pass.PreCheck(*currFunctionPtr);
    EXPECT_EQ(result, FAILED);
}

/*
            /————> copy ————> view ————> viewtype ————> assemble \
    incast -                                                      - outcast
            \————> copy ————> view ————> viewtype ————> assemble /
*/
TEST_F(ReplaceTensorTest, TestBackView)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewAssemble", "TestViewAssemble",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    // init RawTensor
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewTypeRaw0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewTypeRaw1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto copy0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto copy1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto viewIn0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor0, offset0, shape1,
                                                              CreateTestConstIntVector(shape1));
    auto viewIn1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor1, offset0, shape1,
                                                              CreateTestConstIntVector(shape1));
    auto viewTypeIn0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewTypeRaw0, offset0, shape1,
                                                                  CreateTestConstIntVector(shape1));
    auto viewTypeIn1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewTypeRaw1, offset0, shape1,
                                                                  CreateTestConstIntVector(shape1));
    auto assIn0 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor0, offset0, shape1,
                                                             CreateTestConstIntVector(shape1));
    auto assIn1 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor1, offset1, shape1,
                                                             CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, shape,
                                                              CreateTestConstIntVector(shape));
    // Init Graph
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copy0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copy1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copy0}, {viewIn0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copy1}, {viewIn1});
    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {viewIn0}, {viewTypeIn0});
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {viewIn1}, {viewTypeIn1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW_TYPE, {viewTypeIn0}, {assIn0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW_TYPE, {viewTypeIn1}, {assIn1});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {assIn0}, {outcast});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {assIn1}, {outcast});
    // Init Attribute
    auto view_Attr0 = std::make_shared<ViewOpAttribute>(offset0);
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(offset1);
    auto ass_Attr0 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0);
    auto ass_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    viewOp0.SetOpAttribute(view_Attr0);
    viewOp1.SetOpAttribute(view_Attr1);
    assOp0.SetOpAttribute(ass_Attr0);
    assOp1.SetOpAttribute(ass_Attr1);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(outcast->GetRawMagic(), assIn0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), assIn1->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewTypeIn0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewTypeIn1->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewIn0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewIn1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestProcessHubAssembleOp_Success)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestProcessHubAssembleOp_Success",
                                                      "TestProcessHubAssembleOp_Success", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // 准备HUB-ASSEMBLE-OUTCAST链
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset = {kNumZero, kNumZero};

    // 创建共享的raw tensor
    std::shared_ptr<RawTensor> rawTensor = std::make_shared<RawTensor>(DT_FP32, shape);

    // 创建HUB操作相关张量
    auto hubInput = npu::tile_fwk::IRBuilder().CreateTensorVar(rawTensor, offset, shape,
                                                               CreateTestConstIntVector(shape));
    auto hubOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(rawTensor, offset, shape,
                                                                CreateTestConstIntVector(shape));

    // 创建ASSEMBLE操作相关张量
    auto assembleOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(rawTensor, offset, shape,
                                                                     CreateTestConstIntVector(shape));

    // 设置内存类型
    hubInput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    hubOutput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    assembleOutput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);

    // 创建操作
    auto& hubOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_HUB, {hubInput}, {hubOutput});
    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {hubOutput},
                                                        {assembleOutput});

    // 设置操作连接
    hubOutput->AddConsumer(&assembleOp);

    // 设置ASSEMBLE属性
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr);

    // 将assembleOutput设置为outcast
    currFunctionPtr->outCasts_.push_back(assembleOutput);

    ReplaceTensor pass;
    pass.ProcessHubAssembleOp(*currFunctionPtr, hubOp, assembleOp, hubInput, hubOutput);

    // 验证hubInput和hubOutput共享了assembleOutput的tensor
    EXPECT_EQ(hubInput->GetRawTensor(), assembleOutput->GetRawTensor());
    EXPECT_EQ(hubOutput->GetRawTensor(), assembleOutput->GetRawTensor());
}

TEST_F(ReplaceTensorTest, TestProcessHubOpUpdateCopyOutRawShape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestProcessHubOpUpdateCopyOutRawShape",
                                                      "TestProcessHubOpUpdateCopyOutRawShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumEight};
    std::vector<int64_t> partShape = {kNumFour, kNumEight};
    std::vector<int64_t> copyToOffset = {kNumOne, kNumZero};
    std::vector<int64_t> hubOffset = {kNumTwo, kNumZero};

    auto fullRawTensor = std::make_shared<RawTensor>(DT_FP32, fullShape);
    auto copyIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    copyIn->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto hubInput = npu::tile_fwk::IRBuilder().CreateTensorVar(fullRawTensor, hubOffset, partShape,
                                                               CreateTestConstIntVector(partShape));
    hubInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto hubOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, partShape,
                                                                CreateTestConstIntVector(partShape));
    hubOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    auto& copyOutOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyIn}, {hubInput});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_HUB, {hubInput}, {hubOutput});

    auto copyOutAttr = std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB, OpImmediate::Specified(copyToOffset),
                                                         OpImmediate::Specified(partShape),
                                                         OpImmediate::Specified(partShape));
    copyOutOp.SetOpAttribute(copyOutAttr);

    ReplaceTensor pass;
    EXPECT_EQ(pass.ProcessHubOp(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(OpImmediate::ToSpecified(copyOutAttr->GetToOffset()), CreateTestConstIntVector({kNumThree, kNumZero}));
    EXPECT_EQ(OpImmediate::ToSpecified(copyOutAttr->GetRawShape()), CreateTestConstIntVector(fullShape));
}

TEST_F(ReplaceTensorTest, TestA_MULACC_B)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestProcessHubAssembleOp_BrokenChain",
                                                      "TestProcessHubAssembleOp_BrokenChain", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> mulAccshape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    // init LogicalTensor
    auto inTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto inTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto mulAccIn = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, mulAccshape,
                                                               CreateTestConstIntVector(mulAccshape));
    auto mulAccOut = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    /* Init Graph
        incast -> Index_OutCast -> mulAccOut-> op
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MULACC_B, {inTensor0, inTensor1, mulAccIn},
                                     {mulAccOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {mulAccOut}, {outTensor});
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(mulAccIn);
    currFunctionPtr->outCasts_.push_back(outTensor);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(mulAccIn->GetRawMagic(), mulAccOut->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestSameAssembleOut)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSameAssembleOut",
                                                      "TestSameAssembleOut", nullptr);
    EXPECT_NE(currFunctionPtr, nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> copyInRawTensor = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> outRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                             CreateTestConstIntVector(shape));
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto copyInOut = npu::tile_fwk::IRBuilder().CreateTensorVar(copyInRawTensor, offset0, shape1,
                                                                CreateTestConstIntVector(shape1));
    copyInOut->SetMemoryTypeBoth(MEM_UB, true);
    auto outcast0 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor0, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    outcast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor1, offset0, shape1,
                                                               CreateTestConstIntVector(shape1));
    outcast1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    /*       Init Graph
                            /—————> assemble -outcast1
                             /————> assemble \
        incast ————> copyIn -                 - outcast0
                             \————> assemble /
    */
    auto& copyInOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyInOut});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyInOut}, {outcast0});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyInOut}, {outcast0});
    auto& assOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyInOut}, {outcast1});
    // Init Attribute
    auto copyInAttr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset0), MEM_UB,
                                                        OpImmediate::Specified(shape), OpImmediate::Specified(shape));
    auto assAttr0 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset0);
    auto assAttr1 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset1);
    auto assAttr2 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset0);
    copyInOp.SetOpAttribute(copyInAttr);
    assOp0.SetOpAttribute(assAttr0);
    assOp1.SetOpAttribute(assAttr1);
    assOp2.SetOpAttribute(assAttr2);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast0);
    currFunctionPtr->outCasts_.push_back(outcast1);
    int opSumBefore = currFunctionPtr->Operations().size();
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opSumBefore + 6);
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestNotInplaceReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNotInplaceReshape",
                                                      "TestNotInplaceReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumFour, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> shape2 = {kNumFour, kNumFour};
    std::vector<int64_t> shape3 = {kNumOne, kNumFour, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> reshapeRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto reshapeOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(reshapeRawTensor0, offset0, shape1,
                                                                  CreateTestConstIntVector(shape1));
    auto viewOut = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor, offset0, shape2,
                                                              CreateTestConstIntVector(shape2));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    /* Init Graph
        incast0 -> Reshape -> View -> Reshape -> outcast
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {incast}, {reshapeOut0});
    auto& viewOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {reshapeOut0}, {viewOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {viewOut}, {outcast});

    auto view_Attr = std::make_shared<ViewOpAttribute>(offset0, MEM_DEVICE_DDR);
    viewOp.SetOpAttribute(view_Attr);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_NE(viewOut->GetRawMagic(), outcast->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, UpdateCopyInAttrAfterBackAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "UpdateCopyInAttrAfterBackAssemble",
                                                      "UpdateCopyInAttrAfterBackAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {4, 4};
    std::vector<int64_t> outshape1 = {8, 4};
    std::vector<int64_t> outshape2 = {2, 4};

    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInout1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    copyInout1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyOutout1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    copyOutout1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInout2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    copyInout2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape1, CreateTestConstIntVector(outshape1));
    outcast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape2, CreateTestConstIntVector(outshape2));
    outcast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    /* Init Graph
        incast -- CopyIn -- copyInout1 -- CopyOut -- copyOutOut1 -- Assemble -- outcast1
                                                                 -- CopyIn   -- copyInout2 -- CopyOut -- outcast2
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyInout1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyInout1}, {copyOutout1});
    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyOutout1},
                                                        {outcast1});
    auto& copyInOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {copyOutout1},
                                                      {copyInout2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyInout2}, {outcast2});

    Offset assembleToOffset = {4, 0};
    auto assembleOpAttribute = std::make_shared<AssembleOpAttribute>(assembleToOffset,
                                                                     CreateTestConstIntVector(assembleToOffset));
    assembleOp.SetOpAttribute(assembleOpAttribute);
    Offset copyIn2FromOffset = {2, 0};
    auto copyInOpAttribute = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(copyIn2FromOffset), MEM_UB, OpImmediate::Specified(copyInout2->GetShape()),
        OpImmediate::Specified(copyInout2->tensor->GetDynRawShape()),
        OpImmediate::Specified(copyInout2->GetDynValidShape()));
    copyInOp.SetOpAttribute(copyInOpAttribute);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);

    ReplaceTensor replaceTensorPass;
    int opSumBefore = currFunctionPtr->Operations().size();
    replaceTensorPass.RunOnFunction(*currFunctionPtr);
    int opSumExpAfter = opSumBefore + 2;
    EXPECT_EQ(replaceTensorPass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opSumExpAfter);
}

/*
 * 验证场景：SHMEM_WAIT_UNTIL 的输出 tensor 被多个 ASSEMBLE 消费且输出到不同 outcast 时，
 * ReplaceTensor 不插入 COPY_IN/COPY_OUT。
 */
TEST_F(ReplaceTensorTest, TestShmemWaitUntilWithDiffAssembleOut)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestShmemWaitUntilAssemble",
                                                      "TestShmemWaitUntilAssemble", nullptr);
    EXPECT_NE(currFunctionPtr, nullptr);
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    std::shared_ptr<RawTensor> inRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> inRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> shmemRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto incast0 = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor0, offset0, shape);
    incast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor1, offset0, shape);
    incast1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto shmemOut = std::make_shared<LogicalTensor>(*currFunctionPtr, shmemRawTensor, offset0, shape);
    shmemOut->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast0 = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor0, offset0, shape);
    outcast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor1, offset1, shape1);
    outcast1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SHMEM_WAIT_UNTIL, {incast0, incast1}, {shmemOut});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {shmemOut}, {outcast0});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {shmemOut}, {outcast1});
    auto assAttr0 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0);
    auto assAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assOp0.SetOpAttribute(assAttr0);
    assOp1.SetOpAttribute(assAttr1);
    currFunctionPtr->inCasts_.push_back(incast0);
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast0);
    currFunctionPtr->outCasts_.push_back(outcast1);
    ReplaceTensor pass;
    int opSumBefore = currFunctionPtr->Operations().size();
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opSumBefore);
    for (auto& op : currFunctionPtr->Operations()) {
        EXPECT_TRUE(op.GetOpcode() != Opcode::OP_COPY_IN && op.GetOpcode() != Opcode::OP_COPY_OUT);
    }
    EXPECT_EQ(shmemOut->GetRawMagic(), outcast0->GetRawMagic());
    EXPECT_EQ(shmemOut->GetRawMagic(), outcast1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

// 场景：L0C2UB copy 携带非零 fromOffset，下游 assemble 又携带 toOffset，
// 源与目的同时携带偏移无法折叠，pass 应报错返回 FAILED
TEST_F(ReplaceTensorTest, FoldL0C2UBCopyOffsetSrcDstConflictFail)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "FoldL0C2UBOffsetConflict",
                                                      "FoldL0C2UBOffsetConflict", nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    const std::vector<int64_t> l0cShape = {64, 128};
    const std::vector<int64_t> bigShape = {128, 128};
    const std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    const std::vector<int64_t> srcOffset = {kNumZero, 32};
    const std::vector<int64_t> dstOffset = {kNumZero, 64};

    auto l0a = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0b = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0cRaw = std::make_shared<RawTensor>(DT_FP32, l0cShape);
    auto l0c = npu::tile_fwk::IRBuilder().CreateTensorVar(l0cRaw, offset0, l0cShape,
                                                          CreateTestConstIntVector(l0cShape));
    l0c->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto ub = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    ub->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto ubBigRaw = std::make_shared<RawTensor>(DT_FP32, bigShape);
    auto ubBig = npu::tile_fwk::IRBuilder().CreateTensorVar(ubBigRaw, offset0, bigShape,
                                                            CreateTestConstIntVector(bigShape));
    ubBig->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto mulIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));
    auto mulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a, l0b}, {l0c});
    auto& copyOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L0C_COPY_UB, {l0c}, {ub});
    auto& assOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ub}, {ubBig});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {ubBig, mulIn}, {mulOut});

    // copy 携带非零 fromOffset [0,32]（其余属性与 GenerateMoveOp 产物对齐）
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(srcOffset), MemoryType::MEM_UB, OpImmediate::Specified(l0cShape),
        OpImmediate::Specified(l0cShape), OpImmediate::Specified(CreateTestConstIntVector(l0cShape)));
    copyAttr->SetToOffset(OpImmediate::Specified(offset0));
    copyOp.SetOpAttribute(copyAttr);
    copyOp.SetAttribute(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(Matrix::CopyMode::EXTRACT));
    // assemble 携带非零 toOffset [0,64]
    assOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, dstOffset));

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), FAILED);
}

// 场景：两个 matmul 结果经 L0C_COPY_UB(EXTRACT) 搬到 UB，各自 ASSEMBLE 到同一个大 tensor 供 MUL 使用。
// ASSEMBLE toOffset 非零([0,64])且 copy fromOffset 全零 → 折叠为 INSERT 并携带 toOffset；
// ASSEMBLE toOffset 全零([0,0]) → 维持 EXTRACT 不动作（参考 chunk_kda tensor 520 计算路径）
TEST_F(ReplaceTensorTest, FoldL0C2UBCopyOffsetForDualAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "FoldL0C2UBCopyOffset",
                                                      "FoldL0C2UBCopyOffset", nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    const std::vector<int64_t> l0cShape = {64, 128};
    const std::vector<int64_t> bigShape = {128, 128};
    const std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    const std::vector<int64_t> offset1 = {kNumZero, 64};

    // l0a/l0b 输入与两路 matmul 的 L0C 输出
    auto l0a1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0b1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0a2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0b2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0cRaw1 = std::make_shared<RawTensor>(DT_FP32, l0cShape);
    auto l0cRaw2 = std::make_shared<RawTensor>(DT_FP32, l0cShape);
    auto l0c1 = npu::tile_fwk::IRBuilder().CreateTensorVar(l0cRaw1, offset0, l0cShape,
                                                           CreateTestConstIntVector(l0cShape));
    auto l0c2 = npu::tile_fwk::IRBuilder().CreateTensorVar(l0cRaw2, offset0, l0cShape,
                                                           CreateTestConstIntVector(l0cShape));
    l0c1->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    l0c2->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    // L0C2UB copy 输出的 UB 小 tensor
    auto ub1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto ub2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    ub1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    ub2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    // assemble 汇聚的大 UB tensor 与 MUL 侧
    auto ubBigRaw = std::make_shared<RawTensor>(DT_FP32, bigShape);
    auto ubBig = npu::tile_fwk::IRBuilder().CreateTensorVar(ubBigRaw, offset0, bigShape,
                                                            CreateTestConstIntVector(bigShape));
    ubBig->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto mulIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));
    auto mulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a1, l0b1}, {l0c1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a2, l0b2}, {l0c2});
    auto& copyOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L0C_COPY_UB, {l0c1}, {ub1});
    auto& copyOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L0C_COPY_UB, {l0c2}, {ub2});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ub1}, {ubBig});
    auto& assOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ub2}, {ubBig});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {ubBig, mulIn}, {mulOut});

    // copy 属性对齐 GenerateMoveOp::SetL0C2UBCopyAttr 产物：EXTRACT + 零 toOffset
    auto makeCopyAttr = [&l0cShape, &offset0](Operation& op) {
        auto attr = std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(offset0), MemoryType::MEM_UB, OpImmediate::Specified(l0cShape),
            OpImmediate::Specified(l0cShape), OpImmediate::Specified(CreateTestConstIntVector(l0cShape)));
        attr->SetToOffset(OpImmediate::Specified(offset0));
        op.SetOpAttribute(attr);
        op.SetAttr(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(Matrix::CopyMode::EXTRACT));
    };
    makeCopyAttr(copyOp1);
    makeCopyAttr(copyOp2);
    // assemble1 写大 tensor 偏移 [0,64]，assemble2 写 [0,0]
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1));
    assOp2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset0));

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // toOffset 非零分支：折叠为 INSERT，copy toOffset 携带 assemble 偏移
    EXPECT_EQ(copyOp1.GetIntAttribute(OpAttributeKey::localCopyLocalMode),
              static_cast<int64_t>(Matrix::CopyMode::INSERT));
    auto foldedAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp1.GetOpAttribute());
    ASSERT_NE(foldedAttr, nullptr);
    const auto& foldedToOffset = foldedAttr->GetToOffset();
    ASSERT_EQ(foldedToOffset.size(), offset1.size());
    for (size_t i = 0; i < offset1.size(); i++) {
        EXPECT_EQ(foldedToOffset[i].GetSpecifiedValue().Raw()->GetImmediateValue(), offset1[i]);
    }
    // toOffset 全零分支：维持 EXTRACT，toOffset 不变
    EXPECT_EQ(copyOp2.GetIntAttribute(OpAttributeKey::localCopyLocalMode),
              static_cast<int64_t>(Matrix::CopyMode::EXTRACT));
    auto unfoldAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp2.GetOpAttribute());
    ASSERT_NE(unfoldAttr, nullptr);
    const auto& unfoldToOffset = unfoldAttr->GetToOffset();
    for (size_t i = 0; i < offset0.size(); i++) {
        EXPECT_EQ(unfoldToOffset[i].GetSpecifiedValue().Raw()->GetImmediateValue(), offset0[i]);
    }
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

// 场景：copy 输出同时被 ASSEMBLE 和普通算子（MUL）消费，consumers 非唯一触发跳过守卫
TEST_F(ReplaceTensorTest, FoldL0C2UBCopyOffsetSkipForMultiConsumer)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "FoldL0C2UBSkipMultiCons",
                                                      "FoldL0C2UBSkipMultiCons", nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    const std::vector<int64_t> l0cShape = {64, 128};
    const std::vector<int64_t> bigShape = {128, 128};
    const std::vector<int64_t> offset0 = {kNumZero, kNumZero};

    auto l0a = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0b = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0cRaw = std::make_shared<RawTensor>(DT_FP32, l0cShape);
    auto l0c = npu::tile_fwk::IRBuilder().CreateTensorVar(l0cRaw, offset0, l0cShape,
                                                          CreateTestConstIntVector(l0cShape));
    l0c->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto ub = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    ub->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto ubBigRaw = std::make_shared<RawTensor>(DT_FP32, bigShape);
    auto ubBig = npu::tile_fwk::IRBuilder().CreateTensorVar(ubBigRaw, offset0, bigShape,
                                                            CreateTestConstIntVector(bigShape));
    ubBig->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto mulIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));
    auto mulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));
    auto mulIn2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto mulOut2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a, l0b}, {l0c});
    auto& copyOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L0C_COPY_UB, {l0c}, {ub});
    auto& assOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ub}, {ubBig});
    // copy 输出同时被普通算子直接消费：fold 改写 toOffset 会使其读错位
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {ub, mulIn2}, {mulOut2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {ubBig, mulIn}, {mulOut});

    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(offset0), MemoryType::MEM_UB, OpImmediate::Specified(l0cShape),
        OpImmediate::Specified(l0cShape), OpImmediate::Specified(CreateTestConstIntVector(l0cShape)));
    copyAttr->SetToOffset(OpImmediate::Specified(offset0));
    copyOp.SetOpAttribute(copyAttr);
    copyOp.SetAttribute(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(Matrix::CopyMode::EXTRACT));
    assOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset0));

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // consumers 非唯一跳过折叠：copy 维持 EXTRACT，toOffset 保持全零不被覆盖
    EXPECT_EQ(copyOp.GetIntAttribute(OpAttributeKey::localCopyLocalMode),
              static_cast<int64_t>(Matrix::CopyMode::EXTRACT));
    auto skipAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp.GetOpAttribute());
    ASSERT_NE(skipAttr, nullptr);
    const auto& skipToOffset = skipAttr->GetToOffset();
    for (size_t i = 0; i < offset0.size(); i++) {
        EXPECT_EQ(skipToOffset[i].GetSpecifiedValue().Raw()->GetImmediateValue(), offset0[i]);
    }
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

/*
 * A5(DAV_3510)场景下A_MULACC_B支持最多5个输入
 */
TEST_F(ReplaceTensorTest, TestA_MULACC_B_5Inputs_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestA_MULACC_B_5Inputs_A5",
                                                      "TestA_MULACC_B_5Inputs_A5", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> mulAccshape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    auto inTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto inTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto mulAccIn = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, mulAccshape,
                                                               CreateTestConstIntVector(mulAccshape));
    auto bias = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape, CreateTestConstIntVector(mulAccshape));
    auto deqScale = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                               CreateTestConstIntVector(mulAccshape));
    auto mulAccOut = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MULACC_B,
                                     {inTensor0, inTensor1, mulAccIn, bias, deqScale}, {mulAccOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {mulAccOut}, {outTensor});
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.insert(currFunctionPtr->inCasts_.end(), {inTensor0, inTensor1, mulAccIn, bias, deqScale});
    currFunctionPtr->outCasts_.push_back(outTensor);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(mulAccIn->GetRawMagic(), mulAccOut->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}
} // namespace tile_fwk
} // namespace npu
