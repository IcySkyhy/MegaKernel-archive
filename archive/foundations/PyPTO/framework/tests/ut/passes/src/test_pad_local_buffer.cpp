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
 * \file test_pad_local_buffer.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/tile_graph_pass/graph_constraint/pad_local_buffer.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"
#include "computational_graph_builder.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;
constexpr size_t K_1 = 1;
constexpr size_t K_4 = 4;
constexpr size_t K_8 = 8;
constexpr size_t K_16 = 16;
constexpr size_t K_64 = 64;
constexpr size_t K_128 = 128;
class TestPadLocalBuffer : public ::testing::Test {
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

namespace {
inline void ConstructGraphWithTwoInputs(std::shared_ptr<Function>& currFunctionPtr, const std::vector<int64_t>& shape)
{
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor3->SetMemoryTypeBoth(MEM_UB);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto& copy_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                                            OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& copy_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(CreateTestScalarVar("Input_1_Dim_0")),
                                                             OpImmediate(CreateTestScalarVar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);
    auto& add_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2},
                                                  {ubTensor3});
    (void)add_op;
    auto& copy_out_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void)copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph1(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    ConstructGraphWithTwoInputs(currFunctionPtr, {8, 15});
}

inline void ConstructGraph2(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor3->SetMemoryTypeBoth(MEM_UB);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto& copy_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                                            OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& copy_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(CreateTestScalarVar("Input_1_Dim_0")),
                                                             OpImmediate(CreateTestScalarVar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);
    auto& add_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2},
                                                  {ubTensor3});
    (void)add_op;
    auto& copy_out_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void)copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph3(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    std::vector<int64_t> AShape = {8, 15};
    std::vector<int64_t> BShape = {15, 15};
    std::vector<int64_t> CShape = {8, 16};
    std::vector<int64_t> expOriShape = {16, 16};
    auto AShapeImme = OpImmediate::Specified(AShape);
    auto BShapeImme = OpImmediate::Specified(BShape);
    auto CShapeImme = OpImmediate::Specified(CShape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, AShape, CreateTestConstIntVector(AShape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, BShape, CreateTestConstIntVector(BShape));
    auto l0Atensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, AShape, CreateTestConstIntVector(AShape));
    l0Atensor0->SetMemoryTypeBoth(MEM_L0A);
    auto l0Btensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, BShape, CreateTestConstIntVector(BShape));
    l0Btensor0->SetMemoryTypeBoth(MEM_L0B);
    auto l0Ctensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, CShape, CreateTestConstIntVector(CShape));
    l0Ctensor0->SetMemoryTypeBoth(MEM_L0C);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, expOriShape,
                                                              CreateTestConstIntVector(expOriShape));
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto& copy_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {l0Atensor0});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_L0A, AShapeImme,
                                                         AShapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                                            OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& copy_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {l0Btensor0});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_L0B, BShapeImme,
                                                         BShapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(CreateTestScalarVar("Input_1_Dim_0")),
                                                             OpImmediate(CreateTestScalarVar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);
    auto& matmul_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MULACC_B, {l0Atensor0, l0Btensor0},
                                                     {l0Ctensor0});
    (void)matmul_op;
    auto& copy_out_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {l0Ctensor0}, {outCast});
    (void)copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph9(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    std::vector<int64_t> shape = {16, 6};
    std::vector<int64_t> trans_shape = {6, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, trans_shape,
                                                                CreateTestConstIntVector(trans_shape));
    ubTensor3->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, trans_shape,
                                                                CreateTestConstIntVector(trans_shape));
    ubTensor4->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, trans_shape,
                                                                CreateTestConstIntVector(trans_shape));
    ubTensor5->SetMemoryTypeBoth(MEM_UB);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, trans_shape,
                                                              CreateTestConstIntVector(trans_shape));
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outCast2->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto& copy_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                                            OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& abs_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ABS, {ubTensor1}, {ubTensor2});
    (void)abs_op;
    auto& transpose_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_TRANSPOSE_VNCHWCONV, {ubTensor2},
                                                        {ubTensor3, ubTensor4});
    (void)transpose_op;
    transpose_op.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{1, 0});
    auto& exp_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXP, {ubTensor3}, {ubTensor5});
    (void)exp_op;
    auto& copy_out_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor5}, {outCast});
    (void)copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
}
} // namespace

/*
before:
  copyin  copyin
  [8,15]  [8,15]
    \       /
       add
      [8,15]
        |
      copyout

after:
  copyin  copyin
  [8,16]  [8,16]
    \       /
       add
      [8,16]
        |
      copyout
*/
TEST_F(TestPadLocalBuffer, no_reduce_last_dim_all_vec_last_dim_unpadded)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer",
                                                      "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 15};
    std::vector<int64_t> expOriShape = {8, 16};
    ConstructGraph1(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ADD) {
            for (auto& in : op.iOperand) {
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, shape);
                    EXPECT_EQ(in->tensor->rawshape, expOriShape);
                }
            }
            for (auto& out : op.oOperand) {
                if (out->oriShape == shape) {
                    EXPECT_EQ(out->shape, shape);
                    EXPECT_EQ(out->tensor->rawshape, expOriShape);
                }
            }
        }
    }
}

/*
before:
    copyin  copyin
    [8,16]  [8,16]
      \       /
       add
      [8,16]
        |
      copyout

after:
    copyin  copyin
    [8,16]  [8,16]
      \       /
       add
      [8,16]
        |
      copyout
*/
TEST_F(TestPadLocalBuffer, no_reduce_last_dim_all_vec_last_dim_padded)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer",
                                                      "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> expOriShape = {8, 16};
    ConstructGraph2(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ADD) {
            for (auto& in : op.iOperand) {
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, expOriShape);
                    EXPECT_EQ(in->tensor->rawshape, expOriShape);
                }
            }
            for (auto& out : op.oOperand) {
                if (out->oriShape == shape) {
                    EXPECT_EQ(out->shape, expOriShape);
                    EXPECT_EQ(out->tensor->rawshape, expOriShape);
                }
            }
        }
    }
}

/*
before:
    copyin  copyin
    [8,15]  [15,15]
      \       /
       matmul
       [8,15]
          |
       copyout

after:
    copyin  copyin
    [16,16]  [16,16]
      \       /
       matmul
       [16,16]
          |
       copyout
*/
TEST_F(TestPadLocalBuffer, no_reduce_last_dim_mm)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer",
                                                      "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> AShape = {8, 15};
    std::vector<int64_t> BShape = {15, 15};
    std::vector<int64_t> CShape = {8, 16};
    std::vector<int64_t> expOriShape = {16, 16};
    ConstructGraph3(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_A_MULACC_B) {
            for (auto& in : op.iOperand) {
                if (in->oriShape == AShape) {
                    EXPECT_EQ(in->shape, AShape);
                    EXPECT_EQ(in->tensor->rawshape, expOriShape);
                }
                if (in->oriShape == BShape) {
                    EXPECT_EQ(in->shape, BShape);
                    EXPECT_EQ(in->tensor->rawshape, expOriShape);
                }
            }
            for (auto& out : op.oOperand) {
                if (out->oriShape == CShape) {
                    EXPECT_EQ(out->shape, CShape);
                    EXPECT_EQ(out->tensor->rawshape, expOriShape);
                }
            }
        }
    }
}

TEST_F(TestPadLocalBuffer, axiscombine)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 127}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 127}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t1", "t2"}, {"t3"}, "add", true), true);
    auto* rootFuncPtr = graph.GetFunction();
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*rootFuncPtr);
    // ================== Verify Pass Effect ==================
    auto updatedOps = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto& op : updatedOps) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
            auto outputTensor = op.GetOOperands()[0];
            EXPECT_TRUE(outputTensor->GetConsumers().size() != 0);
        }
        if (op.HasAttr(OpAttributeKey::brcbIdx)) {
            auto idx = op.GetIntAttribute(OpAttributeKey::brcbIdx) - 1;
            auto tensor = op.GetIOperands()[idx];
            EXPECT_TRUE(tensor != nullptr);
            EXPECT_EQ(tensor->shape[0], K_4);
            EXPECT_EQ(tensor->shape[1], K_8);
            EXPECT_EQ(tensor->GetRawTensor()->GetRawShape()[0], K_8);
            EXPECT_EQ(tensor->GetRawTensor()->GetRawShape()[1], K_8);
        }
    }
    EXPECT_EQ(cnt, K_1);
}

TEST_F(TestPadLocalBuffer, axiscombineDisable)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 127}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 2}, MemoryType::MEM_UB, "gm"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 127}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm"}, {"t2"}, "copyin", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t1", "t2"}, {"t3"}, "add", true), true);
    auto* rootFuncPtr = graph.GetFunction();
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*rootFuncPtr);
    // ================== Verify Pass Effect ==================
    auto updatedOps = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto& op : updatedOps) {
        if (op.GetOpcode() == Opcode::OP_EXPAND) {
            ++cnt;
        }
        for (auto& inTensor : op.GetIOperands()) {
            if (inTensor->Symbol() == "t2") {
                std::vector<int64_t> targetShape = {4, 8};
                EXPECT_EQ(inTensor->GetRawTensor()->GetRawShape(), targetShape);
            }
        }
    }
    EXPECT_EQ(cnt, K_1);
}

TEST_F(TestPadLocalBuffer, axiscombineDisable1)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_DEVICE_DDR, "gm1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 2}, MemoryType::MEM_DEVICE_DDR, "gm2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm1"}, {"t1"}, "copyin1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm2"}, {"t2"}, "copyin2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t1", "t2"}, {"t3"}, "add", true), true);
    auto* rootFuncPtr = graph.GetFunction();
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*rootFuncPtr);
    // ================== Verify Pass Effect ==================
    auto updatedOps = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto& op : updatedOps) {
        if (op.GetOpcode() == Opcode::OP_EXPAND || op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
        for (auto& inTensor : op.GetIOperands()) {
            if (inTensor->Symbol() == "t2" || inTensor->Symbol() == "t1") {
                std::vector<int64_t> targetShape = {4, 8};
                EXPECT_EQ(inTensor->GetRawTensor()->GetRawShape(), targetShape);
            }
        }
    }
    EXPECT_EQ(cnt, 0);
}

TEST_F(TestPadLocalBuffer, axiscombineDisable2)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 3}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_DEVICE_DDR, "gm2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 2}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t1"}, {"t2"}, "view1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm2"}, {"t3"}, "copyin2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t2", "t3"}, {"t4"}, "add", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"t4"}, {"t5"}, "assemble", true), true);
    auto* rootFuncPtr = graph.GetFunction();
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*rootFuncPtr);
    // ================== Verify Pass Effect ==================
    auto updatedOps = rootFuncPtr->Operations();
    for (const auto& op : updatedOps) {
        for (auto& inTensor : op.GetIOperands()) {
            if (inTensor->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                std::vector<int64_t> targetShape = {4, 8};
                EXPECT_EQ(inTensor->GetRawTensor()->GetRawShape(), targetShape);
            }
        }
    }
}

TEST_F(TestPadLocalBuffer, axiscombine2)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {32, 4, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {32, 4, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {32, 4, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_DIV, {"t1", "t2"}, {"t3"}, "div", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {32, 4, 1, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_RESHAPE, {"t3"}, {"t4"}, "reshape", true), true);
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    // ================== Verify Pass Effect ==================
    auto updatedOperations = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto& op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
    auto t3 = graph.GetTensor("t3");
    auto shape = t3->GetRawTensor()->GetRawShape();
    EXPECT_EQ(shape[shape.size() - 1], K_1);
    EXPECT_EQ(shape[shape.size() - 2], K_8);
}

TEST_F(TestPadLocalBuffer, axiscombineDisable3)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "gm"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm"}, {"t1"}, "copyin", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {24, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t1"}, {"t2"}, "expand", true), true);
    graph.GetOp("expand")->SetAttribute(OpAttributeKey::expandDims, std::vector<int>{0});

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {24, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {24, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_DIV, {"t2", "t3"}, {"t4"}, "div", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {24, 1}, MemoryType::MEM_DEVICE_DDR, "gm2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t4"}, {"gm2"}, "copyout", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t4"}, {"t5"}, "view", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 2}, MemoryType::MEM_UB, "t6"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "t7"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t6"}, {"t7"}, "view1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "resres"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t5", "t7"}, {"resres"}, "add", true), true);
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    auto updatedOperations = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto& op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
}

TEST_F(TestPadLocalBuffer, axiscombineTest)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {24, 16}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t1"}, {"t2"}, "view", true), true);
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*rootFuncPtr), SUCCESS);
}

TEST_F(TestPadLocalBuffer, axiscombineTest1)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {24, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t3"}, {"t4"}, "view1", true), true);
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*rootFuncPtr), SUCCESS);
}

TEST_F(TestPadLocalBuffer, axiscombineEnable)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 16}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {24, 16}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t1"}, {"t2"}, "expand", true), true);
    graph.GetOp("expand")->SetAttribute(OpAttributeKey::expandDims, std::vector<int>{0});
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {24, 1}, MemoryType::MEM_DEVICE_DDR, "gm"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {24, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm"}, {"t3"}, "copyin", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {24, 16}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_DIV, {"t2", "t3"}, {"t4"}, "div", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {24, 1}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t4"}, {"t5"}, "view", true), true);
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    auto updatedOperations = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto& op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 1);
}

TEST_F(TestPadLocalBuffer, axiscombine3)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1, 160}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_RESHAPE, {"t1"}, {"t2"}, "reshape", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t2"}, {"t3"}, "expand", true), true);
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    // ================== Verify Pass Effect ==================
    auto updatedOperations = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto& op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
    auto t2 = graph.GetTensor("t2");
    auto shape = t2->GetRawTensor()->GetRawShape();
    EXPECT_EQ(shape[shape.size() - 1], K_8);
    EXPECT_EQ(shape[shape.size() - 2], K_1);
    EXPECT_EQ(shape[shape.size() - 3], K_1);
}

// deepseek lightning_indexer_prolog_quant case
TEST_F(TestPadLocalBuffer, axiscombine4)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {2, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"t1"}, {"t2"}, "cast", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {2, 1, 1, 1}, MemoryType::MEM_UB, "index-t0"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_RESHAPE, {"t2"}, {"index-t0"}, "reshape", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 1}, MemoryType::MEM_UB, "index-t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 128, 1, 1}, MemoryType::MEM_UB, "index-t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 128, 1, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_INDEX_OUTCAST, {"index-t0", "index-t1", "index-t2"}, {"t3"}, "index", true), true);
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    // ================== Verify Pass Effect ==================
    EXPECT_EQ(graph.GetTensor("index-t0")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>({2, 1, 1, 16})));
    EXPECT_EQ(graph.GetTensor("t2")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>({16, 1})));
}

// 1dim brcb
TEST_F(TestPadLocalBuffer, axiscombine5)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1}, MemoryType::MEM_DEVICE_DDR, "gm1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16}, MemoryType::MEM_DEVICE_DDR, "gm2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm1"}, {"t1"}, "copyin1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm2"}, {"t2"}, "copyin2", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t1", "t2"}, {"t3"}, "add", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16}, MemoryType::MEM_DEVICE_DDR, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t3"}, {"out"}, "copyout", true), true);
    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    auto updatedOps = rootFuncPtr->Operations();
    for (const auto& op : updatedOps) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            auto inputTensor = op.GetIOperands()[0];
            EXPECT_EQ(inputTensor->GetRawTensor()->GetRawShape()[0], K_8);
            break;
        }
    }
}

// scale_ub have multiple BRCB consumer
TEST_F(TestPadLocalBuffer, axiscombine6)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {4, 256}, MemoryType::MEM_DEVICE_DDR, "x_gm"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_DEVICE_DDR, "scale_gm"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 6144}, MemoryType::MEM_DEVICE_DDR, "w_scale_gm"), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {4, 64}, MemoryType::MEM_UB, "x_ub1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {4, 64}, MemoryType::MEM_UB, "x_ub2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 64}, MemoryType::MEM_UB, "x_fp32_1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 64}, MemoryType::MEM_UB, "x_fp32_2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, MemoryType::MEM_UB, "scale_ub"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 64}, MemoryType::MEM_UB, "w_scale_ub1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 64}, MemoryType::MEM_UB, "w_scale_ub2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 64}, MemoryType::MEM_UB, "mul1_out1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 64}, MemoryType::MEM_UB, "mul1_out2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 64}, MemoryType::MEM_UB, "mul2_out1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 64}, MemoryType::MEM_UB, "mul2_out2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 64}, MemoryType::MEM_UB, "cast_out1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 64}, MemoryType::MEM_UB, "cast_out2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {4, 128}, MemoryType::MEM_UB, "assemble_out"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {4, 1, 128}, MemoryType::MEM_UB, "reshape_out"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 4, 128}, MemoryType::MEM_DEVICE_DDR, "trans_out"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {4, 128}, MemoryType::MEM_DEVICE_DDR, "output"), true);

    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"x_gm"}, {"x_ub1"}, "copyin1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"scale_gm"}, {"scale_ub"}, "copyin_scale", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"w_scale_gm"}, {"w_scale_ub1"}, "copyin_wscale1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"w_scale_gm"}, {"w_scale_ub2"}, "copyin_wscale2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"x_ub1"}, {"x_fp32_1"}, "cast1", true), true);

    EXPECT_EQ(graph.AddOp(Opcode::OP_MUL, {"x_fp32_1", "scale_ub"}, {"mul1_out1"}, "mul1_1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_MUL, {"mul1_out1", "w_scale_ub1"}, {"mul2_out1"}, "mul2_1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"mul2_out1"}, {"cast_out1"}, "cast_out1", true), true);

    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"x_gm"}, {"x_ub2"}, "copyin2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"x_ub2"}, {"x_fp32_2"}, "cast2", true), true);

    EXPECT_EQ(graph.AddOp(Opcode::OP_MUL, {"x_fp32_2", "scale_ub"}, {"mul1_out2"}, "mul1_2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_MUL, {"mul1_out2", "w_scale_ub2"}, {"mul2_out2"}, "mul2_2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"mul2_out2"}, {"cast_out2"}, "cast_out2", true), true);

    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"cast_out1"}, {"assemble_out"}, "assemble1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"cast_out2"}, {"assemble_out"}, "assemble2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_RESHAPE, {"assemble_out"}, {"reshape_out"}, "reshape1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_TRANSPOSE_MOVEOUT, {"reshape_out"}, {"trans_out"}, "trans_moveout", true), true);
    graph.GetOp("trans_moveout")->SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{0, 1});
    EXPECT_EQ(graph.AddOp(Opcode::OP_RESHAPE, {"trans_out"}, {"output"}, "reshape2", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*rootFuncPtr), SUCCESS);

    for (const auto& op : rootFuncPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            EXPECT_EQ(op.oOperand[0]->GetRawTensor()->GetRawShape()[0], K_8);
        }
    }
}

/*
Test case from PROD
  t_pairprod_1 [4,1,3,1]  t_pairprod_2 [4,1,3,1]
         \                   /
          PAIRPROD [4,1,3,1] (t_pairprod_3)
                |
          ROWPRODLINE (AXIS=2) [4,1,1,1]
                |
             RESHAPE
*/
TEST_F(TestPadLocalBuffer, axiscombine7)
{
    ComputationalGraphBuilder graph;
    // Input tensors for the first PAIRPROD: [4,1,3,1] x [4,1,3,1] -> [4,1,3,1]
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1, 3, 1}, MemoryType::MEM_UB, "t_input_1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1, 3, 1}, MemoryType::MEM_UB, "t_input_2"), true);
    // Input tensors for the second PAIRPROD: [4,1,3,1] x [4,1,1,1] -> [4,1,3,1]
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1, 3, 1}, MemoryType::MEM_UB, "t_input_3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1, 1, 1}, MemoryType::MEM_UB, "t_input_4"), true);

    // Output of first PAIRPROD
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1, 3, 1}, MemoryType::MEM_UB, "t_pairprod_1"), true);
    // Output of second PAIRPROD
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1, 3, 1}, MemoryType::MEM_UB, "t_pairprod_2"), true);
    // Output of third PAIRPROD (combines results from first two): [4,1,3,1] x [4,1,3,1] -> [4,1,3,1]
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1, 3, 1}, MemoryType::MEM_UB, "t_pairprod_3"), true);
    // Output of ROWPRODLINE (reduce along AXIS=2): [4,1,3,1] -> [4,1,1,1]
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1, 1, 1}, MemoryType::MEM_UB, "t_rowprodline"), true);
    // Output of RESHAPE: [4,1,1,1] -> [4,1,1]
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1, 1}, MemoryType::MEM_UB, "t_reshape_out"), true);
    // GM output tensor
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1, 1}, MemoryType::MEM_DEVICE_DDR, "t_gm_out"), true);

    // Build the graph
    EXPECT_EQ(graph.AddOp(Opcode::OP_PAIRPROD, {"t_input_1", "t_input_2"}, {"t_pairprod_1"}, "pairprod1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_PAIRPROD, {"t_input_3", "t_input_4"}, {"t_pairprod_2"}, "pairprod2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_PAIRPROD, {"t_pairprod_1", "t_pairprod_2"}, {"t_pairprod_3"}, "pairprod3", true),
              true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWPRODLINE, {"t_pairprod_3"}, {"t_rowprodline"}, "rowprodline", true), true);
    graph.GetOp("rowprodline")->SetAttribute(OP_ATTR_PREFIX + "AXIS", 2);
    EXPECT_EQ(graph.AddOp(Opcode::OP_RESHAPE, {"t_rowprodline"}, {"t_reshape_out"}, "reshape", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t_reshape_out"}, {"t_gm_out"}, "copyout", true), true);

    auto* currFunctionPtr = graph.GetFunction();
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    currFunctionPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*currFunctionPtr), SUCCESS);

    // Verify that the tensors are padded correctly
    // The last dim 1 should be padded to 8 for vector operations
    auto tPairprod3 = graph.GetTensor("t_pairprod_3");
    EXPECT_EQ(tPairprod3->GetRawTensor()->GetRawShape()[3], 8);

    auto tRowprodline = graph.GetTensor("t_rowprodline");
    EXPECT_EQ(tRowprodline->GetRawTensor()->GetRawShape()[3], 8);

    auto tReshape = graph.GetTensor("t_reshape_out");
    EXPECT_EQ(tReshape->GetRawTensor()->GetRawShape()[2], 8);
}

TEST_F(TestPadLocalBuffer, L1toBt1)
{
    ComputationalGraphBuilder graph;
    // bias
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 15}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 15}, MemoryType::MEM_L1, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "COPY1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 15}, MemoryType::MEM_BT, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_L1_TO_BT, {"t2"}, {"t3"}, "L1TOBT", true), true);
    // a
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {32, 32}, MemoryType::MEM_DEVICE_DDR, "t1a"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {32, 32}, MemoryType::MEM_L1, "t2a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1a"}, {"t2a"}, "COPYA", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {1, 32}, MemoryType::MEM_L0A, "t3a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_L1_TO_L0A, {"t2a"}, {"t3a"}, "L1TOL0A", true), true);
    // b
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {32, 15}, MemoryType::MEM_DEVICE_DDR, "t1b"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {32, 15}, MemoryType::MEM_L1, "t2b"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1b"}, {"t2b"}, "COPYB", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {1, 15}, MemoryType::MEM_L0B, "t3b"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_L1_TO_L0_BT, {"t2b"}, {"t3b"}, "L1TOL0B", true), true);
    // fix
    EXPECT_EQ(graph.AddTensor(DataType::DT_UINT64, {1, 15}, MemoryType::MEM_DEVICE_DDR, "t1f"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_UINT64, {1, 15}, MemoryType::MEM_L1, "t2f"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1f"}, {"t2f"}, "COPYFIX", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_UINT64, {1, 15}, MemoryType::MEM_FIX_QUANT_PRE, "t3f"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_L1_TO_FIX_QUANT_PRE, {"t2f"}, {"t3f"}, "L1TOFIX", true), true);
    // amulb
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {32, 15}, MemoryType::MEM_L0C, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_A_MUL_B, {"t3a", "t3b", "t3f", "t3"}, {"out"}, "AMULB", true), true);

    auto* currFunctionPtr = graph.GetFunction();
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    std::vector<int64_t> expectShape{1, 32};
    auto t2 = graph.GetTensor("t2");
    EXPECT_EQ(t2->tensor->GetRawShape(), expectShape);
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(t3->tensor->GetRawShape(), expectShape);
}

TEST_F(TestPadLocalBuffer, padDimTest1)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 1}, MemoryType::MEM_DEVICE_DDR, "gm"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 16}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_BRCB, {"t1"}, {"t2"}, "view1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm"}, {"t1"}, "copyin", true), true);
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    EXPECT_EQ(graph.GetTensor("t1")->GetRawTensor()->GetRawShape()[0], 16);
}

TEST_F(TestPadLocalBuffer, axiscombineCastCase)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_BF16, {2, 1}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_BF16, {2, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t2"}, "copyin1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"t2"}, {"t3"}, "cast1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t3"}, {"t4"}, "copyout1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 1}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 1}, MemoryType::MEM_UB, "t6"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t4"}, {"t5"}, "copyin2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t4"}, {"t6"}, "copyin3", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 16}, MemoryType::MEM_UB, "t7"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 16}, MemoryType::MEM_UB, "t8"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 16}, MemoryType::MEM_UB, "t9"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t7", "t5"}, {"t8"}, "add1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPANDEXPDIF, {"t7", "t6"}, {"t9"}, "add2", true), true);

    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    int64_t cnt = 0;
    for (const auto& op : rootFuncPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 2);
    EXPECT_EQ(graph.GetTensor("t2")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{16, 1}));
    EXPECT_EQ(graph.GetTensor("t3")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{8, 1}));
}

/*
before:
  copyin0[8,1,224] -> rowargmaxwithvalue_single -> value0[8,1,1]
                                                -> index0[8,1,1] -> adds0[8,1,1] -> cast0_fp32[8,1,1]
                                                                  -> cast0_i32[8,1,1] -> copyout0
                                                -> tmp0[1,224]
  copyin1[6,1,224] -> rowargmaxwithvalue_single -> value1[6,1,1]
                                                -> index1[6,1,1] -> adds1[6,1,1] -> cast1_fp32[6,1,1]
                                                                  -> cast1_i32[6,1,1] -> copyout1
                                                -> tmp1[1,224]

after rawshape:
  value{0,1}      : [*,16,1]
  index/adds      : [*,16,1]
  cast/copyout in : [*,8,1]
*/
TEST_F(TestPadLocalBuffer, axiscombineRowArgMaxWithValueSingle)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {8, 1, 224}, MemoryType::MEM_DEVICE_DDR, "in0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {6, 1, 224}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {8, 1, 224}, MemoryType::MEM_UB, "copyin0_out"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {6, 1, 224}, MemoryType::MEM_UB, "copyin1_out"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {8, 1, 1}, MemoryType::MEM_UB, "value0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT16, {8, 1, 1}, MemoryType::MEM_UB, "index0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 224}, MemoryType::MEM_UB, "tmp0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {6, 1, 1}, MemoryType::MEM_UB, "value1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT16, {6, 1, 1}, MemoryType::MEM_UB, "index1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 224}, MemoryType::MEM_UB, "tmp1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT16, {8, 1, 1}, MemoryType::MEM_UB, "adds0_out"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT16, {6, 1, 1}, MemoryType::MEM_UB, "adds1_out"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1, 1}, MemoryType::MEM_UB, "cast0_fp32"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {6, 1, 1}, MemoryType::MEM_UB, "cast1_fp32"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT32, {8, 1, 1}, MemoryType::MEM_UB, "cast0_i32"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT32, {6, 1, 1}, MemoryType::MEM_UB, "cast1_i32"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT32, {8, 1, 1}, MemoryType::MEM_DEVICE_DDR, "out0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT32, {6, 1, 1}, MemoryType::MEM_DEVICE_DDR, "out1"), true);

    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in0"}, {"copyin0_out"}, "copyin0", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"copyin1_out"}, "copyin1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWARGMAXWITHVALUE_SINGLE, {"copyin0_out"}, {"value0", "index0", "tmp0"},
                          "reduce0", true),
              true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWARGMAXWITHVALUE_SINGLE, {"copyin1_out"}, {"value1", "index1", "tmp1"},
                          "reduce1", true),
              true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADDS, {"index0"}, {"adds0_out"}, "adds0", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADDS, {"index1"}, {"adds1_out"}, "adds1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"adds0_out"}, {"cast0_fp32"}, "cast0_fp32_op", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"adds1_out"}, {"cast1_fp32"}, "cast1_fp32_op", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"cast0_fp32"}, {"cast0_i32"}, "cast0_i32_op", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"cast1_fp32"}, {"cast1_i32"}, "cast1_i32_op", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"cast0_i32"}, {"out0"}, "copyout0", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"cast1_i32"}, {"out1"}, "copyout1", true), true);

    graph.GetOp("reduce0")->SetAttribute(OP_ATTR_PREFIX + "AXIS", 2);
    graph.GetOp("reduce1")->SetAttribute(OP_ATTR_PREFIX + "AXIS", 2);
    graph.GetOp("adds0")->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_INT16, static_cast<int16_t>(0)));
    graph.GetOp("adds1")->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_INT16, static_cast<int16_t>(0)));

    auto* functionPtr = graph.GetFunction();
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    functionPtr->paramConfigs_.combineAxis = true;
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*functionPtr), SUCCESS);

    EXPECT_EQ(graph.GetTensor("value0")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{8, 16, 1}));
    EXPECT_EQ(graph.GetTensor("value1")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{6, 16, 1}));
    EXPECT_EQ(graph.GetTensor("index0")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{8, 16, 1}));
    EXPECT_EQ(graph.GetTensor("index1")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{6, 16, 1}));
    EXPECT_EQ(graph.GetTensor("adds0_out")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{8, 16, 1}));
    EXPECT_EQ(graph.GetTensor("adds1_out")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{6, 16, 1}));
    EXPECT_EQ(graph.GetTensor("cast0_fp32")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{8, 8, 1}));
    EXPECT_EQ(graph.GetTensor("cast1_fp32")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{6, 8, 1}));
    EXPECT_EQ(graph.GetTensor("cast0_i32")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{8, 8, 1}));
    EXPECT_EQ(graph.GetTensor("cast1_i32")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{6, 8, 1}));
    EXPECT_EQ(graph.GetTensor("tmp0")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{1, 224}));
    EXPECT_EQ(graph.GetTensor("tmp1")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{1, 224}));
}

TEST_F(TestPadLocalBuffer, padCmpInputTo256)
{
    ComputationalGraphBuilder graphBuilder;
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "in2"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "copyin1", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in2"}, {"t2"}, "copyin2", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_CMP, {"t1", "t2"}, {"t3"}, "cmp1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_OUT, {"t3"}, {"t4"}, "copyout1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t6"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"t4"}, {"t5"}, "copyin3", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"t4"}, {"t6"}, "copyin4", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t7"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_ADD, {"t6", "t5"}, {"t7"}, "add1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "out1"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_OUT, {"t7"}, {"out1"}, "copyout2", true), true);
    std::vector<bool> dimMap({true, false});
    graphBuilder.GetOp("cmp1")->SetAttr(OpAttributeKey::rowPad, dimMap);
    auto* functionPtr = graphBuilder.GetFunction();
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*functionPtr), SUCCESS);
    EXPECT_EQ(graphBuilder.GetTensor("t1")->GetRawTensor()->GetRawShape()[0], 10);
    EXPECT_EQ(graphBuilder.GetTensor("t2")->GetRawTensor()->GetRawShape()[0], 3);
}

TEST_F(TestPadLocalBuffer, padCmpInputSharedWithVnchwconvTo256)
{
    ComputationalGraphBuilder graphBuilder;
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 3}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 3}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "copyin1", true), true);

    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 3}, MemoryType::MEM_UB, "transOut"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_TRANSPOSE_VNCHWCONV, {"t1"}, {"transOut"}, "transpose1", true), true);
    graphBuilder.GetOp("transpose1")->SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{1, 0});

    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_BOOL, {3, 3}, MemoryType::MEM_UB, "cmpOut"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_CMP, {"t1", "transOut"}, {"cmpOut"}, "cmp1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_BOOL, {3, 3}, MemoryType::MEM_DEVICE_DDR, "out1"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_OUT, {"cmpOut"}, {"out1"}, "copyout1", true), true);

    std::vector<bool> dimMap({true, true});
    graphBuilder.GetOp("cmp1")->SetAttr(OpAttributeKey::rowPad, dimMap);
    auto* functionPtr = graphBuilder.GetFunction();
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*functionPtr), SUCCESS);
    EXPECT_EQ(graphBuilder.GetTensor("t1")->GetRawTensor()->GetRawShape()[0], 10);
    EXPECT_EQ(graphBuilder.GetTensor("transOut")->GetRawTensor()->GetRawShape()[0], 10);
    EXPECT_EQ(graphBuilder.GetTensor("t1")->GetRawTensor()->GetRawShape()[1], 8);
    EXPECT_EQ(graphBuilder.GetTensor("transOut")->GetRawTensor()->GetRawShape()[1], 8);
}

TEST_F(TestPadLocalBuffer, padCmpsInputTo256)
{
    ComputationalGraphBuilder graphBuilder;
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "in2"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "copyin1", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in2"}, {"t2"}, "copyin2", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_CMPS, {"t1", "t2"}, {"t3"}, "cmps1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_OUT, {"t3"}, {"t4"}, "copyout1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t6"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"t4"}, {"t5"}, "copyin3", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"t4"}, {"t6"}, "copyin4", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t7"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_ADD, {"t6", "t5"}, {"t7"}, "add1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "out1"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_OUT, {"t7"}, {"out1"}, "copyout2", true), true);
    std::vector<bool> dimMap({false, true});
    graphBuilder.GetOp("cmps1")->SetAttr(OpAttributeKey::rowPad, dimMap);
    auto* functionPtr = graphBuilder.GetFunction();
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*functionPtr), SUCCESS);
    EXPECT_EQ(graphBuilder.GetTensor("t1")->GetRawTensor()->GetRawShape()[0], 3);
    EXPECT_EQ(graphBuilder.GetTensor("t2")->GetRawTensor()->GetRawShape()[0], 10);
}

TEST_F(TestPadLocalBuffer, padCmps1DInputTo256)
{
    ComputationalGraphBuilder graphBuilder;
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {8}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {8}, MemoryType::MEM_DEVICE_DDR, "in2"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {8}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {8}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "copyin1", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in2"}, {"t2"}, "copyin2", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_BOOL, {8}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_CMPS, {"t1", "t2"}, {"t3"}, "cmps1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_BOOL, {8}, MemoryType::MEM_DEVICE_DDR, "out1"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_OUT, {"t3"}, {"out1"}, "copyout1", true), true);
    graphBuilder.GetOp("cmps1")->SetAttr(OpAttributeKey::rowPad, std::vector<bool>({true, false}));

    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*graphBuilder.GetFunction()), SUCCESS);
    EXPECT_EQ(graphBuilder.GetTensor("t1")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{64}));
    EXPECT_EQ(graphBuilder.GetTensor("t2")->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{8}));
}

TEST_F(TestPadLocalBuffer, padPreluInputTo256)
{
    ComputationalGraphBuilder graphBuilder;
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "in2"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "copyin1", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in2"}, {"t2"}, "copyin2", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_PRELU, {"t1", "t2"}, {"t3"}, "prelu1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_OUT, {"t3"}, {"t4"}, "copyout1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t6"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"t4"}, {"t5"}, "copyin3", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"t4"}, {"t6"}, "copyin4", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t7"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_ADD, {"t6", "t5"}, {"t7"}, "add1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "out1"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_OUT, {"t7"}, {"out1"}, "copyout2", true), true);
    std::vector<bool> dimMap({true, false});
    graphBuilder.GetOp("prelu1")->SetAttr(OpAttributeKey::rowPad, dimMap);
    auto* functionPtr = graphBuilder.GetFunction();
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*functionPtr), SUCCESS);
    EXPECT_EQ(graphBuilder.GetTensor("t1")->GetRawTensor()->GetRawShape()[0], 10);
    EXPECT_EQ(graphBuilder.GetTensor("t2")->GetRawTensor()->GetRawShape()[0], 3);
}

// 一个tensor同时是cmps1和cmps2的输入
TEST_F(TestPadLocalBuffer, padTwoCmpsInputTo256)
{
    ComputationalGraphBuilder graphBuilder;
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "in2"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "in3"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "copyin1", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in2"}, {"t2"}, "copyin2", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"in3"}, {"t3"}, "copyin3", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_CMPS, {"t1", "t2"}, {"t4"}, "cmps1", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_CMPS, {"t3", "t2"}, {"t5"}, "cmps2", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "t6"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "t7"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_OUT, {"t4"}, {"t6"}, "copyout1", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_OUT, {"t5"}, {"t7"}, "copyout2", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t8"), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t9"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"t6"}, {"t8"}, "copyin4", true), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_IN, {"t7"}, {"t9"}, "copyin5", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_UB, "t10"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_ADD, {"t8", "t9"}, {"t10"}, "add1", true), true);
    EXPECT_EQ(graphBuilder.AddTensor(DataType::DT_FP32, {3, 7}, MemoryType::MEM_DEVICE_DDR, "out1"), true);
    EXPECT_EQ(graphBuilder.AddOp(Opcode::OP_COPY_OUT, {"t10"}, {"out1"}, "copyout3", true), true);
    std::vector<bool> dimMap({false, true});
    graphBuilder.GetOp("cmps1")->SetAttr(OpAttributeKey::rowPad, dimMap);
    graphBuilder.GetOp("cmps2")->SetAttr(OpAttributeKey::rowPad, dimMap);
    auto* functionPtr = graphBuilder.GetFunction();
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*functionPtr), SUCCESS);
    EXPECT_EQ(graphBuilder.GetTensor("t1")->GetRawTensor()->GetRawShape()[0], 3);
    EXPECT_EQ(graphBuilder.GetTensor("t2")->GetRawTensor()->GetRawShape()[0], 10);
    EXPECT_EQ(graphBuilder.GetTensor("t3")->GetRawTensor()->GetRawShape()[0], 3);
    EXPECT_EQ(graphBuilder.GetTensor("t1")->GetShape()[0], 3);
    EXPECT_EQ(graphBuilder.GetTensor("t3")->GetShape()[0], 3);
    EXPECT_EQ(graphBuilder.GetTensor("t1")->GetRawTensor()->GetRawShape()[1], 8);
    EXPECT_EQ(graphBuilder.GetTensor("t2")->GetRawTensor()->GetRawShape()[1], 8);
    EXPECT_EQ(graphBuilder.GetTensor("t3")->GetRawTensor()->GetRawShape()[1], 8);
}

TEST_F(TestPadLocalBuffer, UB2L1)
{
    ComputationalGraphBuilder graph;
    // a from vec to cube
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {15, 32}, MemoryType::MEM_DEVICE_DDR, "t1a"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {15, 32}, MemoryType::MEM_UB, "t2a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1a"}, {"t2a"}, "COPYA1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {15, 32}, MemoryType::MEM_UB, "t3a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_UB_COPY_ND2NZ, {"t2a"}, {"t3a"}, "ND2NZ", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {15, 32}, MemoryType::MEM_L1, "t4a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_UB_COPY_L1, {"t3a"}, {"t4a"}, "COPYA2", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {15, 32}, MemoryType::MEM_L0A, "t5a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_L1_TO_L0A, {"t4a"}, {"t5a"}, "L1TOL0A", true), true);
    // b
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {32, 32}, MemoryType::MEM_DEVICE_DDR, "t1b"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {32, 32}, MemoryType::MEM_L1, "t2b"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1b"}, {"t2b"}, "COPYB", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {32, 32}, MemoryType::MEM_L0B, "t3b"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_L1_TO_L0_BT, {"t2b"}, {"t3b"}, "L1TOL0B", true), true);
    // amulb
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {15, 32}, MemoryType::MEM_L0C, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_A_MUL_B, {"t5a", "t3b"}, {"out"}, "AMULB", true), true);

    auto* currFunctionPtr = graph.GetFunction();
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    std::vector<int64_t> expectShape{33, 32};
    auto t3a = graph.GetTensor("t3a");
    EXPECT_EQ(t3a->tensor->GetRawShape(), expectShape);
}

TEST_F(TestPadLocalBuffer, axiscombineAssemblePadSecondLast)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_BF16, {22, 16}, MemoryType::MEM_DEVICE_DDR, "gm_in0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_BF16, {22, 16}, MemoryType::MEM_UB, "t79"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm_in0"}, {"t79"}, "copyin_10133", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {22, 16}, MemoryType::MEM_UB, "t372"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"t79"}, {"t372"}, "cast_10041", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {22, 16}, MemoryType::MEM_DEVICE_DDR, "t80"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t372"}, {"t80"}, "copyout_10169", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {22, 16}, MemoryType::MEM_UB, "t373"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t80"}, {"t373"}, "copyin_10170", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {22, 7}, MemoryType::MEM_DEVICE_DDR, "gm_in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {22, 7}, MemoryType::MEM_UB, "t371"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm_in1"}, {"t371"}, "copyin_10168", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {22, 16}, MemoryType::MEM_UB, "t145"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_PAIRSUM, {"t373", "t371"}, {"t145"}, "pairsum_10083", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {22, 1}, MemoryType::MEM_UB, "t139"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8}, MemoryType::MEM_UB, "t148"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWSUM_SINGLE, {"t145"}, {"t139", "t148"}, "rowsum_10086", true), true);
    graph.GetOp("rowsum_10086")->SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);
    EXPECT_EQ(graph.AddTensor(DataType::DT_BF16, {22, 1}, MemoryType::MEM_UB, "t160"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"t139"}, {"t160"}, "cast_10104", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_BF16, {22, 1}, MemoryType::MEM_UB, "t224"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"t160"}, {"t224"}, "assemble_10140", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_BF16, {137, 1}, MemoryType::MEM_DEVICE_DDR, "t7"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t160"}, {"t7"}, "copyout_10105", true), true);
    EXPECT_EQ(graph.SetOutCast({"t224", "t7"}), true);

    auto* currFunctionPtr = graph.GetFunction();
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    currFunctionPtr->paramConfigs_.combineAxis = true;
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*currFunctionPtr), SUCCESS);

    auto t160 = graph.GetTensor("t160");
    std::vector<int64_t> expectShape{32, 1};
    EXPECT_EQ(t160->GetRawTensor()->GetRawShape(), expectShape);
}

TEST_F(TestPadLocalBuffer, axiscombineViewPadSecondLast)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_DEVICE_DDR, "gm_in0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_UB, "t48"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm_in0"}, {"t48"}, "copyin_10022", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_DEVICE_DDR, "gm_in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_UB, "t49"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm_in1"}, {"t49"}, "copyin_10021", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_UB, "t50"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t48", "t49"}, {"t50"}, "add_10012", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t51"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t50"}, {"t51"}, "view_10023", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t12"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t50"}, {"t12"}, "view_10018", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t52"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADDS, {"t51"}, {"t52"}, "adds_10016", true), true);
    graph.GetOp("adds_10016")->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 0.0));
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_DEVICE_DDR, "t20"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t12"}, {"t20"}, "copyout_10020", true), true);

    auto* currFunctionPtr = graph.GetFunction();
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    currFunctionPtr->paramConfigs_.combineAxis = true;
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*currFunctionPtr), SUCCESS);

    auto t50 = graph.GetTensor("t50");
    EXPECT_EQ(t50->GetRawTensor()->GetRawShape(), (std::vector<int64_t>{16, 1}));
}

TEST_F(TestPadLocalBuffer, UB2L1_WithAxisCombine)
{
    ComputationalGraphBuilder graph;
    // a from vec to cube
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {15, 32}, MemoryType::MEM_DEVICE_DDR, "t1a"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {15, 32}, MemoryType::MEM_UB, "t2a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1a"}, {"t2a"}, "COPYA1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {15, 32}, MemoryType::MEM_UB, "t3a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_UB_COPY_ND2NZ, {"t2a"}, {"t3a"}, "ND2NZ", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {15, 32}, MemoryType::MEM_L1, "t4a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_UB_COPY_L1, {"t3a"}, {"t4a"}, "COPYA2", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {15, 32}, MemoryType::MEM_L0A, "t5a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_L1_TO_L0A, {"t4a"}, {"t5a"}, "L1TOL0A", true), true);
    // b
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {32, 32}, MemoryType::MEM_DEVICE_DDR, "t1b"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {32, 32}, MemoryType::MEM_L1, "t2b"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1b"}, {"t2b"}, "COPYB", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_INT8, {32, 32}, MemoryType::MEM_L0B, "t3b"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_L1_TO_L0_BT, {"t2b"}, {"t3b"}, "L1TOL0B", true), true);
    // amulb
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {15, 32}, MemoryType::MEM_L0C, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_A_MUL_B, {"t5a", "t3b"}, {"out"}, "AMULB", true), true);

    auto* currFunctionPtr = graph.GetFunction();
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    currFunctionPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    std::vector<int64_t> expectShape{33, 32};
    auto t2 = graph.GetTensor("t3a");
    EXPECT_EQ(t2->tensor->GetRawShape(), expectShape);
}

TEST_F(TestPadLocalBuffer, GemvMatmulKAlign)
{
    ComputationalGraphBuilder graph;
    // a: M=1 GEMV，L1 -> L0A，producer(COPYIN) 带 isGemv 属性
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 32}, MemoryType::MEM_DEVICE_DDR, "t1a"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 32}, MemoryType::MEM_L1, "t2a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1a"}, {"t2a"}, "COPYA", true), true);
    graph.GetOp("COPYA")->SetAttribute(OpAttributeKey::isGemv, static_cast<int64_t>(1));
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 32}, MemoryType::MEM_L0A, "t3a"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_L1_TO_L0A, {"t2a"}, {"t3a"}, "L1TOL0A", true), true);
    // GEMV 的 L1→L0A VIEW 也带 isGemv（IsGemvL0ATensor 读 producer 的 isGemv）
    graph.GetOp("L1TOL0A")->SetAttribute(OpAttributeKey::isGemv, static_cast<int64_t>(1));
    // b
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {32, 16}, MemoryType::MEM_DEVICE_DDR, "t1b"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {32, 16}, MemoryType::MEM_L1, "t2b"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1b"}, {"t2b"}, "COPYB", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {32, 16}, MemoryType::MEM_L0B, "t3b"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_L1_TO_L0_BT, {"t2b"}, {"t3b"}, "L1TOL0B", true), true);
    // amulb
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP16, {1, 16}, MemoryType::MEM_L0C, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_A_MUL_B, {"t3a", "t3b"}, {"out"}, "AMULB", true), true);

    auto* currFunctionPtr = graph.GetFunction();
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);

    // GEMV A 的 M 轴不打 pad（保持 1），L0A K 对齐 512B（fp16 256 元素）
    auto t2a = graph.GetTensor("t2a");
    auto rawShapeL1 = t2a->tensor->GetRawShape();
    EXPECT_EQ(rawShapeL1[0], 1);
    EXPECT_EQ(rawShapeL1[1], 32);
    auto t3a = graph.GetTensor("t3a");
    auto rawShapeL0A = t3a->tensor->GetRawShape();
    EXPECT_EQ(rawShapeL0A[0], 1);
    EXPECT_EQ(rawShapeL0A[1], 256);
}
