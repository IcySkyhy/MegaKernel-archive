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
 * \file test_expand_function.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include <vector>
#include <string>
#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "passes/pass_mgr/pass_manager.h"

#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#define private public
#include "passes/block_graph_pass/loopaxes_proc.h"

namespace npu {
namespace tile_fwk {
static const int kKeepOut = -1;
static const int kNum0 = 0;
static const int kNum1 = 1;
static const int kNum2 = 2;
static const int kNum3 = 3;
static const int kNum4 = 4;
static const int kNum16 = 2;
static const std::vector<int64_t> shape1 = {kNum2, kNum16};
static const std::vector<int64_t> shape2 = {kNum2, kNum2, kNum2, kNum4};
static const std::vector<int64_t> shape3 = {kNum4, kNum2, kNum4};
static const std::vector<int64_t> shape4 = {kNum3, kNum2, kNum2, kNum4};
static const std::vector<SymbolicScalar> symShape1 = {kNum2, kNum16};
static const std::vector<SymbolicScalar> symShape2 = {kNum2, kNum2, kNum2, kNum4};
static const std::vector<SymbolicScalar> symShape3 = {kNum4, kNum2, kNum4};
static const std::vector<SymbolicScalar> symShape4 = {kNum3, kNum2, kNum2, kNum4};
static const std::vector<SymbolicScalar> expectedLoopAxis1 = {kNum2, kNum2};
static const std::vector<SymbolicScalar> expectedLoopAxis2 = {kNum3, kNum2};

class TestLoopaxesProcPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPassGlobalConfig(KEY_VF_OPT_MARK_FOR, true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

bool EqualSymShape(const std::vector<SymbolicScalar>& A, const std::vector<SymbolicScalar>& B)
{
    if (A.size() != B.size()) {
        return false;
    }
    for (size_t i = 0; i < A.size(); ++i) {
        if (A[i].Dump() != B[i].Dump()) {
            return false;
        }
    }
    return true;
}

TEST_F(TestLoopaxesProcPass, LoopaxesProcUTest1)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestLoopaxesProcPass",
                                                  "TestLoopaxesProcPass", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestLoopaxesProcPassLeaf",
                                                      "TestLoopaxesProcPassLeaf", rootFuncPtr.get());
    // Register the leaf function in Program's functionmap_ so GetFunctionByMagicName can find it
    Program::GetInstance().InsertFuncToFunctionMap(currFunctionPtr->GetMagicName(), currFunctionPtr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetGraphType(GraphType::EXECUTE_GRAPH);
    currFunctionPtr->SetGraphType(GraphType::TILE_GRAPH);
    currFunctionPtr->SetFunctionType(FunctionType::STATIC);
    rootFuncPtr->SetUnderDynamicFunction(true);

    auto inCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast1->UpdateDynValidShape(symShape1);
    auto inCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    inCast2->UpdateDynValidShape(symShape2);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor1->UpdateDynValidShape(symShape1);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor2->UpdateDynValidShape(symShape2);
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor3->UpdateDynValidShape(symShape2);
    auto ubTensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor4->UpdateDynValidShape(symShape2);
    auto ubTensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor5->UpdateDynValidShape(symShape2);
    auto ubTensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape4, CreateTestConstIntVector(shape4));
    ubTensor6->UpdateDynValidShape(symShape4);
    auto ubTensor7 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape4, CreateTestConstIntVector(shape4));
    ubTensor7->UpdateDynValidShape(symShape4);
    auto ubTensor8 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape4, CreateTestConstIntVector(shape4));
    ubTensor8->UpdateDynValidShape(symShape4);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    outCast->UpdateDynValidShape(symShape3);

    auto& expand = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXPAND, {inCast2}, {ubTensor2});
    expand.SetAttribute(OpAttributeKey::expandDims, std::vector<int>{kNum3});
    PassOperationUtils::AddOperation(*currFunctionPtr, npu::tile_fwk::Opcode::OP_BAR_ALL, {inCast1}, {ubTensor2});
    auto& add = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor2, ubTensor3}, {ubTensor4});
    auto& mul = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {ubTensor2, ubTensor4}, {ubTensor5});
    auto& sub = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SUB, {ubTensor6, ubTensor7}, {ubTensor8});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor5}, {outCast});
    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    // Create a call operation in rootFunc to connect to currFunctionPtr
    auto rootInCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto rootInCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto rootOutCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    auto& callOp = IRBuilder().CreateTensorOpStmt(*rootFuncPtr, Opcode::OP_CALL, {rootInCast1, rootInCast2},
                                                  {rootOutCast});
    std::vector<std::vector<SymbolicScalar>> argList;
    std::map<int, SymbolicScalar> outIndexToExpr;
    callOp.SetOpAttribute(currFunctionPtr->CreateCallOpAttribute(argList, outIndexToExpr));
    callOp.SetIOpAtt(0, 0);
    callOp.SetIOpAtt(1, 0);
    callOp.SetOOpAtt(0, 0);

    LoopaxesProc loopaxesprocpass;
    EXPECT_EQ(loopaxesprocpass.RunOnFunction(*rootFuncPtr), SUCCESS);

    EXPECT_TRUE(expand.HasAttr(OpAttributeKey::dynloopGroup));
    EXPECT_EQ(expand.GetIntAttribute(OpAttributeKey::dynloopGroup), kNum0);
    EXPECT_TRUE(expand.HasAttr(OpAttributeKey::dynloopAxes));
    EXPECT_TRUE(EqualSymShape(expand.GetVectorSymbolicScalarAttribute(OpAttributeKey::dynloopAxes), expectedLoopAxis1));

    EXPECT_TRUE(add.HasAttr(OpAttributeKey::dynloopGroup));
    EXPECT_EQ(add.GetIntAttribute(OpAttributeKey::dynloopGroup), kNum1);
    EXPECT_TRUE(add.HasAttr(OpAttributeKey::dynloopAxes));
    EXPECT_TRUE(EqualSymShape(add.GetVectorSymbolicScalarAttribute(OpAttributeKey::dynloopAxes), expectedLoopAxis1));

    EXPECT_TRUE(mul.HasAttr(OpAttributeKey::dynloopGroup));
    EXPECT_EQ(mul.GetIntAttribute(OpAttributeKey::dynloopGroup), kNum1);
    EXPECT_TRUE(mul.HasAttr(OpAttributeKey::dynloopAxes));
    EXPECT_TRUE(EqualSymShape(mul.GetVectorSymbolicScalarAttribute(OpAttributeKey::dynloopAxes), expectedLoopAxis1));

    EXPECT_TRUE(sub.HasAttr(OpAttributeKey::dynloopGroup));
    EXPECT_EQ(sub.GetIntAttribute(OpAttributeKey::dynloopGroup), kNum2);
    EXPECT_TRUE(sub.HasAttr(OpAttributeKey::dynloopAxes));
    EXPECT_TRUE(EqualSymShape(sub.GetVectorSymbolicScalarAttribute(OpAttributeKey::dynloopAxes), expectedLoopAxis2));
}

TEST_F(TestLoopaxesProcPass, LoopaxesProcSubProgramNullptr)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "LoopaxesProcNullTest",
                                                  "LoopaxesProcNullTest", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->programs_[0] = nullptr;
    rootFuncPtr->programs_[1] = rootFuncPtr.get();

    LoopaxesProc loopaxesprocpass;
    EXPECT_EQ(loopaxesprocpass.RunOnFunction(*rootFuncPtr), SUCCESS);
}
} // namespace tile_fwk
} // namespace npu
