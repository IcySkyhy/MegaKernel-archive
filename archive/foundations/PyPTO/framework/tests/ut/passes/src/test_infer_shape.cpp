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
 * \file test_infer_shape.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "symbolic_scalar_test_utils.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/op_infer_shape_impl.h"
#include "passes/tile_graph_pass/graph_constraint/infer_dyn_shape.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
class InferShapeTest : public testing::Test {
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

namespace {
std::shared_ptr<CopyOpAttribute> CreateCopyInAttribute(const std::vector<OpImmediate>& offset, MemoryType memoryType,
                                                       const std::vector<OpImmediate>& fromShape,
                                                       const std::vector<OpImmediate>& toShape,
                                                       std::initializer_list<const char*> dimNames = {})
{
    auto attr = std::make_shared<CopyOpAttribute>(offset, memoryType, fromShape, toShape, std::vector<OpImmediate>());
    if (dimNames.size() != 0) {
        std::vector<OpImmediate> dynValidShape;
        dynValidShape.reserve(dimNames.size());
        for (const char* dimName : dimNames) {
            dynValidShape.emplace_back(CreateTestScalarVar(dimName));
        }
        attr->SetToDynValidShape(dynValidShape);
    }
    return attr;
}

std::shared_ptr<CopyOpAttribute> CreateCopyOutAttribute(MemoryType memoryType, const std::vector<OpImmediate>& offset,
                                                        const std::vector<OpImmediate>& fromShape,
                                                        const std::vector<OpImmediate>& toShape)
{
    return std::make_shared<CopyOpAttribute>(memoryType, offset, fromShape, toShape, std::vector<OpImmediate>());
}

void AddCopyOp(const std::shared_ptr<Function>& currFunctionPtr, Opcode opcode,
               const std::shared_ptr<LogicalTensor>& input, const std::shared_ptr<LogicalTensor>& output,
               const std::shared_ptr<CopyOpAttribute>& attr)
{
    PassOperationUtils::AddOperation(*currFunctionPtr, opcode, {input}, {output},
                                     [&attr](Operation& op) { op.SetOpAttribute(attr); });
}

void AppendFunctionIO(const std::shared_ptr<Function>& currFunctionPtr,
                      const std::vector<std::shared_ptr<LogicalTensor>>& inputs,
                      const std::vector<std::shared_ptr<LogicalTensor>>& outputs)
{
    currFunctionPtr->inCasts_.insert(currFunctionPtr->inCasts_.end(), inputs.begin(), inputs.end());
    currFunctionPtr->outCasts_.insert(currFunctionPtr->outCasts_.end(), outputs.begin(), outputs.end());
}

void RunInferShapeAndExpect(const std::shared_ptr<Function>& currFunctionPtr, Status expected)
{
    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), expected);
}
} // namespace

TEST_F(InferShapeTest, TestAdd)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                                            OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1},
                                     [&copyin1Attr](Operation& op) { op.SetOpAttribute(copyin1Attr); });

    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(CreateTestScalarVar("Input_1_Dim_0")),
                                                             OpImmediate(CreateTestScalarVar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2},
                                     [&copyin2Attr](Operation& op) { op.SetOpAttribute(copyin2Attr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAddAlignCase)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});

    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1},
                                     [&copyin1Attr](Operation& op) { op.SetOpAttribute(copyin1Attr); });

    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2},
                                     [&copyin2Attr](Operation& op) { op.SetOpAttribute(copyin2Attr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    auto copyoutAttr = std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {outCast},
                                     [&copyoutAttr](Operation& op) { op.SetOpAttribute(copyoutAttr); });

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAddExp)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_IN, incast1, ubTensor1,
              CreateCopyInAttribute(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                    {"Input_0_Dim_0", "Input_0_Dim_1"}));
    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_IN, incast2, ubTensor2,
              CreateCopyInAttribute(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                    {"Input_1_Dim_0", "Input_1_Dim_1"}));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    auto tmpCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_OUT, ubTensor3, tmpCast,
              CreateCopyOutAttribute(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));

    auto ubTensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_IN, tmpCast, ubTensor4,
              CreateCopyInAttribute(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme));

    auto ubTensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor4}, {ubTensor5});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor5}, {outCast});

    AppendFunctionIO(currFunctionPtr, {incast1, incast2}, {outCast});
    RunInferShapeAndExpect(currFunctionPtr, SUCCESS);
}

TEST_F(InferShapeTest, TestReduce)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReduceInferShape",
                                                      "TestReduceInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {4, 8, 16};
    std::vector<int64_t> outshape = {4, 8, 8};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_2"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    auto axis = inshape.size() - 1;
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ROWMAX_SINGLE, {inTensor}, {outTensor},
        [&axis](Operation& op) { op.SetAttribute(OP_ATTR_PREFIX + "AXIS", static_cast<int>(axis)); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestView)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TesViewInferShape", "TesViewInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});

    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});
    auto view_Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>(), MEM_UNKNOWN,
                                                       std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>());
    view_Attr->SetFromOffset(std::vector<int64_t>(),
                             {CreateTestScalarVar("Offset_0_Dim_0"), CreateTestScalarVar("Offset_0_Dim_1")});
    auto& view_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {outcast},
                                                     [&view_Attr](Operation& op) { op.SetOpAttribute(view_Attr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << view_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestViewAlign)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TesViewInferShape", "TesViewInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> offset = {2, 0};
    std::vector<int64_t> viewshape = {8, 4};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, viewshape, std::vector<SymbolicScalar>{});

    auto view_Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>(), MEM_UNKNOWN,
                                                       std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>());
    view_Attr->SetFromOffset(offset, std::vector<SymbolicScalar>());
    auto& view_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {outcast},
                                                     [&view_Attr](Operation& op) { op.SetOpAttribute(view_Attr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << view_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAssembleInferShape",
                                                      "TestAssembleInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});

    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});
    outcast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(
        MEM_UNKNOWN, std::vector<int64_t>(), std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>());

    auto dynOffset = {CreateTestScalarVar("DynOffset_0_Dim_0"), CreateTestScalarVar("DynOffset_0_Dim_1")};
    assemble_Attr->SetToOffset({2, 2}, dynOffset);
    auto& assemble_op = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {incast}, {outcast},
        [&assemble_Attr](Operation& op) { op.SetOpAttribute(assemble_Attr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << assemble_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestFailCopyOut)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyOutInferShape",
                                                      "TestCopyOutInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 8};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {incast}, {outcast});
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), FAILED);
}

TEST_F(InferShapeTest, TestCopyOut)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyOutInferShape",
                                                      "TestCopyOutInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 16};
    auto toOffsetImme = OpImmediate::Specified({4, 4});
    auto inshapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});

    auto copyout_Attr = std::make_shared<CopyOpAttribute>(MEM_DEVICE_DDR, toOffsetImme, inshapeImme, inshapeImme,
                                                          std::vector<OpImmediate>());
    auto& copyout_op = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_COPY_OUT, {incast}, {outcast},
        [&copyout_Attr](Operation& op) { op.SetOpAttribute(copyout_Attr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << copyout_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestCopyIn)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyInInferShape",
                                                      "TestCopyInInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 8};
    auto fromOffsetImme = OpImmediate::Specified({4, 4});
    auto inshapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(fromOffsetImme, MEM_UNKNOWN, inshapeImme, inshapeImme,
                                                         OpImmediate::Specified({4, 4}));
    auto& copyin_op = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {outcast},
        [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << copyin_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeInferShape",
                                                      "TestReshapeInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {4, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});
    outcast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UNKNOWN, shapeImme,
                                                         shapeImme, std::vector<OpImmediate>());

    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_1_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_1_Dim_1"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    auto& reshape_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inTensor}, {outTensor});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << reshape_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestSHMEM_LOAD)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSHMEM_LOAD", "TestSHMEM_LOAD",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape0 = {1, 1};
    std::vector<int64_t> inshape1 = {1, 1, 8, 16};
    std::vector<int64_t> outshape = {8, 16};
    auto shapeImme = OpImmediate::Specified(outshape);

    auto incast0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape0, std::vector<SymbolicScalar>{});
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape1, std::vector<SymbolicScalar>{});
    auto shmemLoadOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto shmemLoad_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme,
                                                            shapeImme, std::vector<OpImmediate>());

    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    shmemLoad_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SHMEM_LOAD, {incast0, incast1}, {shmemLoadOut},
                                     [&shmemLoad_Attr](Operation& op) { op.SetOpAttribute(shmemLoad_Attr); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {shmemLoadOut}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast0);
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_NE(shmemLoadOut->GetDynValidShape().size(), 0);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestPad)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadInferShape", "TestPadInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {2, 2};
    std::vector<int64_t> outshape = {3, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_PAD, {inTensor}, {outTensor}, [](Operation& op) {
        op.SetAttribute(OP_ATTR_PREFIX + "pad_right", 2);
        op.SetAttribute(OP_ATTR_PREFIX + "pad_bottom", 1);
        op.SetAttribute(OpAttributeKey::scalar, Element(DT_FP32, 0.0f));
    });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestFillPad)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestFillPadInferShape",
                                                      "TestFillPadInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {3, 4};
    std::vector<int64_t> outshape = {3, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_FILLPAD, {inTensor}, {outTensor}, [](Operation& op) {
        op.SetAttribute(OpAttributeKey::scalar, Element(DT_FP32, 0.0f));
    });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestIndexOutCast)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestIndexOutCast", "TestIndexOutCast",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape0 = {1, 1};
    std::vector<int64_t> inshape1 = {2, 2};
    std::vector<int64_t> inshape2 = {4, 4};
    std::vector<int64_t> outshape = {4, 4};

    auto incast0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape0, std::vector<SymbolicScalar>{});
    incast0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto view0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape0, std::vector<SymbolicScalar>{});
    view0->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape1, std::vector<SymbolicScalar>{});
    incast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto view1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape1, std::vector<SymbolicScalar>{});
    view1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape2, std::vector<SymbolicScalar>{});
    std::vector<SymbolicScalar> validShape = {CreateTestScalarVar("Input_0_Dim_0"),
                                              CreateTestScalarVar("Input_0_Dim_1")};
    incast2->UpdateDynValidShape(validShape);
    incast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    Offset offsets = {0, 0};
    auto viewOpAttribute0 = std::make_shared<ViewOpAttribute>(offsets);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast0}, {view0},
                                     [&viewOpAttribute0](Operation& op) { op.SetOpAttribute(viewOpAttribute0); });
    auto viewOpAttribute1 = std::make_shared<ViewOpAttribute>(offsets);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast1}, {view1},
                                     [&viewOpAttribute1](Operation& op) { op.SetOpAttribute(viewOpAttribute1); });
    auto indexoutcastOpAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified(offsets), OpImmediate::Specified(inshape1),
        OpImmediate::Specified(incast2->tensor->GetDynRawShape()));
    auto& indexoutcastOp = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_INDEX_OUTCAST, {view0, view1, incast2}, {outcast},
        [&indexoutcastOpAttr](Operation& op) { op.SetOpAttribute(indexoutcastOpAttr); });

    AppendFunctionIO(currFunctionPtr, {incast0, incast1, incast2}, {outcast});
    RunInferShapeAndExpect(currFunctionPtr, SUCCESS);
    EXPECT_NE(outcast->GetDynValidShape().size(), 0);
    auto indexOutCastOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(indexoutcastOp.GetOpAttribute());
    const auto& fromDynValidShape = indexOutCastOpAttribute->GetFromDynValidShape();
    EXPECT_NE(fromDynValidShape.size(), 0U);
}

TEST_F(InferShapeTest, TestPermute)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPermuteInferShape",
                                                      "TestPermuteInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape = {2, 3, 4};
    std::vector<int64_t> outshape = {3, 2, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0, 0}), MEM_UB, shapeImme,
                                                         shapeImme, std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_2"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_PERMUTE, {inTensor}, {outTensor}, [](Operation& op) {
        op.SetAttribute(OpAttributeKey::perm, std::vector<int>{1, 0, 2});
    });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestPermuteElement)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPermuteElemInferShape",
                                                      "TestPermuteElemInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape = {2, 3, 4};
    std::vector<int64_t> outshape = {2, 4, 3};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0, 0}), MEM_UB, shapeImme,
                                                         shapeImme, std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_2"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_PERMUTE_ELEMENT, {inTensor}, {outTensor},
        [](Operation& op) { op.SetAttribute(OpAttributeKey::perm, std::vector<int>{0, 2, 1}); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestErfcUnary)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestErfcInferShape",
                                                      "TestErfcInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensorOut = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    auto copyinAttr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                        std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                                            OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyinAttr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {ubTensor},
                                     [&copyinAttr](Operation& op) { op.SetOpAttribute(copyinAttr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ERFC, {ubTensor}, {ubTensorOut});

    auto copyoutAttr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensorOut}, {outCast},
                                     [&copyoutAttr](Operation& op) { op.SetOpAttribute(copyoutAttr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

// BRCB op output whose DynValidShape exceeds tensor shape should be skipped and return SUCCESS.
// Graph:
//   incast1[16,16] --copy_in--> ub1[16,16] ------------------ add --> addOut[16,16] --copy_out--> outcast[16,16]
//                                                            /
//   incast2[16,1]  --copy_in--> ub2[16,1]  --brcb--> brcbOut[16,8](validShape=[16,16] > shape, skipped)
TEST_F(InferShapeTest, BRCBSkipsValidShapeValidation)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "BRCBSkipsValidShapeValidation",
                                                      "BRCBSkipsValidShapeValidation", nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape1 = {16, 16};
    std::vector<int64_t> shape2 = {16, 1};
    std::vector<int64_t> brcbShape = {16, 8};
    auto shape1Imme = OpImmediate::Specified(shape1);
    auto shape2Imme = OpImmediate::Specified(shape2);

    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, std::vector<SymbolicScalar>{});
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, std::vector<SymbolicScalar>{});
    auto ub1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, std::vector<SymbolicScalar>{});
    auto ub2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, std::vector<SymbolicScalar>{});
    auto brcbOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, brcbShape, std::vector<SymbolicScalar>{});
    auto addOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, std::vector<SymbolicScalar>{});

    auto copyin1Attr = CreateCopyInAttribute(OpImmediate::Specified({0, 0}), MEM_UB, shape1Imme, shape1Imme,
                                             {"Input_0_Dim_0", "Input_0_Dim_1"});
    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_IN, incast1, ub1, copyin1Attr);

    auto copyin2Attr = CreateCopyInAttribute(OpImmediate::Specified({0, 0}), MEM_UB, shape2Imme, shape2Imme,
                                             {"Input_1_Dim_0", "Input_1_Dim_1"});
    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_IN, incast2, ub2, copyin2Attr);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_BRCB, {ub2}, {brcbOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ub1, brcbOut}, {addOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {addOut}, {outcast});

    ub1->UpdateDynValidShape(CreateTestConstIntVector(shape1));
    ub2->UpdateDynValidShape(CreateTestConstIntVector(shape2));
    brcbOut->UpdateDynValidShape({IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(16)});
    addOut->UpdateDynValidShape(CreateTestConstIntVector(shape1));
    outcast->UpdateDynValidShape(CreateTestConstIntVector(shape1));

    AppendFunctionIO(currFunctionPtr, {incast1, incast2}, {outcast});

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

// A concrete DynValidShape larger than its tensor shape must fail post-check.
TEST_F(InferShapeTest, CheckDynValidShapeFailed)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "CheckDynValidShapeFailed",
                                                      "CheckDynValidShapeFailed", nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);

    const std::vector<int64_t> shape = {16, 16};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    output->UpdateDynValidShape({IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(17)});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {input}, {output});
    AppendFunctionIO(currFunctionPtr, {input}, {output});

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), FAILED);
}

} // namespace tile_fwk
} // namespace npu
