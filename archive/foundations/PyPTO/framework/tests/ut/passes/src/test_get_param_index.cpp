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
 * \file test_get_param_index.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/op_infer_shape_impl.h"
#include "passes/block_graph_pass/infer_param_index.h"
#include "interface/operation/attribute.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;

class GetParamIdxTest : public testing::Test {
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

TEST_F(GetParamIdxTest, TestAdd)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddParams", "TestAddParams",
                                                      rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor1->UpdateDynValidShape({CreateTestScalarVar("S0"), CreateTestScalarVar("S1")});
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor2->UpdateDynValidShape({CreateTestScalarVar("Z0"), CreateTestScalarVar("Z1")});
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor3->UpdateDynValidShape({CreateTestScalarVar("S0"), CreateTestScalarVar("Z1")});
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& copy_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    std::vector<npu::tile_fwk::OpImmediate> fromOffset = {OpImmediate::Parameter(1), OpImmediate::Parameter(2)};
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(fromOffset, MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate::Parameter(3), OpImmediate::Parameter(4)};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetIOpAtt(0, 0);
    copy_op1.SetOpAttribute(copyin1Attr);

    auto& copy_op2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    fromOffset = {OpImmediate::Parameter(6), OpImmediate::Parameter(7)};
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(fromOffset, MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate::Parameter(8), OpImmediate::Parameter(9)};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetIOpAtt(0, 5);
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2},
                                                    {ubTensor3});
    (void)add_op;
    auto& copy_out_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    copy_out_op.SetOOpAtt(0, 11);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferParamIndex getParamIndexTest;
    getParamIndexTest.RunOnFunction(*rootFuncPtr);
    EXPECT_TRUE(true);
}

TEST_F(GetParamIdxTest, TestAddExp)
{
    auto rootGraphPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    EXPECT_TRUE(rootGraphPtr != nullptr);
    rootGraphPtr->rootFunc_ = rootGraphPtr.get();
    auto subGraphPtr0 = std::make_shared<Function>(Program::GetInstance(), "TestAddParams", "TestAddParams",
                                                   rootGraphPtr.get());
    auto subGraphPtr1 = std::make_shared<Function>(Program::GetInstance(), "TestExpParams", "TestExpParams",
                                                   rootGraphPtr.get());
    rootGraphPtr->rootFunc_->programs_.emplace(subGraphPtr0->GetFuncMagic(), subGraphPtr0.get());
    rootGraphPtr->rootFunc_->programs_.emplace(subGraphPtr1->GetFuncMagic(), subGraphPtr1.get());
    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor1->UpdateDynValidShape({CreateTestScalarVar("S0"), CreateTestScalarVar("S1")});
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor2->UpdateDynValidShape({CreateTestScalarVar("Z0"), CreateTestScalarVar("Z1")});
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor3->UpdateDynValidShape({CreateTestScalarVar("S0"), CreateTestScalarVar("Z1")});
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& copy_op1 = PassOperationUtils::AddOperation(*subGraphPtr0, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    std::vector<npu::tile_fwk::OpImmediate> fromOffset = {OpImmediate::Parameter(1), OpImmediate::Parameter(2)};
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(fromOffset, MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate::Parameter(3), OpImmediate::Parameter(4)};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetIOpAtt(0, 0);
    copy_op1.SetOpAttribute(copyin1Attr);

    auto& copy_op2 = PassOperationUtils::AddOperation(*subGraphPtr0, Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    fromOffset = {OpImmediate::Parameter(6), OpImmediate::Parameter(7)};
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(fromOffset, MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate::Parameter(8), OpImmediate::Parameter(9)};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetIOpAtt(0, 5);
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = PassOperationUtils::AddOperation(*subGraphPtr0, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void)add_op;
    auto tmpCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto& copy_out_op = PassOperationUtils::AddOperation(*subGraphPtr0, Opcode::OP_COPY_OUT, {ubTensor3}, {tmpCast});
    auto copyout1Attr = std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme);
    copy_out_op.SetOpAttribute(copyout1Attr);
    copy_out_op.SetOOpAtt(0, 10);

    auto ubTensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor4->UpdateDynValidShape({CreateTestScalarVar("X0"), CreateTestScalarVar("X1")});
    auto& copy_op3 = PassOperationUtils::AddOperation(*subGraphPtr1, Opcode::OP_COPY_IN, {tmpCast}, {ubTensor4});
    fromOffset = {OpImmediate::Parameter(16), OpImmediate::Parameter(17)};
    auto copyin3Attr = std::make_shared<CopyOpAttribute>(fromOffset, MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape2 = {OpImmediate::Parameter(18), OpImmediate::Parameter(19)};
    copyin3Attr->SetToDynValidShape(toValidShape2);
    copy_op3.SetIOpAtt(0, 15);
    copy_op3.SetOpAttribute(copyin3Attr);

    auto ubTensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto& exp = PassOperationUtils::AddOperation(*subGraphPtr1, Opcode::OP_EXP, {ubTensor4}, {ubTensor5});
    (void)exp;

    auto& copy_out_op1 = PassOperationUtils::AddOperation(*subGraphPtr1, Opcode::OP_COPY_OUT, {ubTensor5}, {outCast});
    copy_out_op1.SetOOpAtt(0, 11);

    subGraphPtr0->inCasts_.push_back(incast1);
    subGraphPtr0->inCasts_.push_back(incast2);
    subGraphPtr0->outCasts_.push_back(tmpCast);

    subGraphPtr1->inCasts_.push_back(tmpCast);
    subGraphPtr1->outCasts_.push_back(outCast);

    InferParamIndex getParamIndexTest;
    getParamIndexTest.RunOnFunction(*rootGraphPtr);
    EXPECT_TRUE(true);
}

/**
 * @brief Verify ResetGmCopyDynValidShape handles a GM-spill CopyIn whose
 *        input is produced by Reshape rather than an InCast.
 *
 * Graph: inCast(DDR) -> Reshape -> reshapeOut(DDR) -> CopyIn -> ubTensor(UB)
 *        -> CopyOut -> outCast
 */
TEST_F(GetParamIdxTest, TestGmSpillCopyInWithOutNoramlize)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGmSpillCopyIn", "TestGmSpillCopyIn",
                                                      rootFuncPtr.get());
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<int64_t> shape1 = {8, 16};
    std::vector<int64_t> shape2 = {16, 8};
    auto shapeImme2 = OpImmediate::Specified(shape2);
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    inCast->UpdateDynValidShape({CreateTestScalarVar("inCast1_dim0"), CreateTestScalarVar("inCast1_dim1")});
    auto reshapeOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    reshapeOut->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    reshapeOut->UpdateDynValidShape({CreateTestScalarVar("reshapeOut_dim0"), CreateTestScalarVar("reshapeOut_dim1")});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inCast}, {reshapeOut});

    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    ubTensor->UpdateDynValidShape({CreateTestScalarVar("ubTensor_dim0"), CreateTestScalarVar("ubTensor_dim1")});
    auto& copyIn = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeOut}, {ubTensor});
    std::vector<npu::tile_fwk::OpImmediate> fromOffset = {OpImmediate::Specified(0), OpImmediate::Specified(0)};
    auto copyinAttr = std::make_shared<CopyOpAttribute>(fromOffset, MEM_UB, shapeImme2, shapeImme2);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate::Specified(16), OpImmediate::Specified(8)};
    copyinAttr->SetToDynValidShape(toValidShape);
    copyIn.SetOpAttribute(copyinAttr);

    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto& copyOut = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor}, {outCast});
    auto copyoutAttr = std::make_shared<CopyOpAttribute>(MEM_UB, fromOffset, shapeImme2, shapeImme2, shapeImme2);
    copyOut.SetOOpAtt(0, 11);
    copyOut.SetOpAttribute(copyoutAttr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferParamIndex getParamIndexTest;
    getParamIndexTest.RunOnFunction(*rootFuncPtr);
}
