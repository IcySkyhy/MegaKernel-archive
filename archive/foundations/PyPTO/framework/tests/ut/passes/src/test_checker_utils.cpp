/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_checker_utils.cpp
 * \brief Unit test for pass utils.
 */

#include <gtest/gtest.h>
#include "passes/pass_utils/checker_utils.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {
const int NUM_8 = 8;
const int NUM_16 = 16;
const int NUM_32 = 32;

class TestCheckerUtils : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

TEST_F(TestCheckerUtils, TestOpChecker)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestOpChecker", "TestOpChecker",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape1 = {NUM_32, NUM_16};
    std::vector<int64_t> shape2 = {NUM_16, NUM_8};
    std::vector<int64_t> shape3 = {NUM_32, NUM_8};

    Program::GetInstance().InsertFuncToFunctionMap("TestOpChecker", currFunctionPtr);

    std::shared_ptr<LogicalTensor> incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, npu::tile_fwk::SymbolicScalar::FromConcrete(shape1));
    std::shared_ptr<LogicalTensor> incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape2, npu::tile_fwk::SymbolicScalar::FromConcrete(shape2));
    std::shared_ptr<LogicalTensor> L1A = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, npu::tile_fwk::SymbolicScalar::FromConcrete(shape1));
    std::shared_ptr<LogicalTensor> L1B = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape2, npu::tile_fwk::SymbolicScalar::FromConcrete(shape2));
    auto& viewA = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {incast1}, {L1A});
    auto& viewB = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {incast2}, {L1B});

    std::shared_ptr<LogicalTensor> L0A = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, npu::tile_fwk::SymbolicScalar::FromConcrete(shape1));
    std::shared_ptr<LogicalTensor> L0B = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape2, npu::tile_fwk::SymbolicScalar::FromConcrete(shape2));
    auto& l1ToL0A = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0A, {L1A}, {L0A});
    auto& l1ToL0B = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0B, {L1B}, {L0B});

    std::shared_ptr<LogicalTensor> L0C = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape3, npu::tile_fwk::SymbolicScalar::FromConcrete(shape3));
    auto& aMulB = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {L0A, L0B}, {L0C});

    std::shared_ptr<LogicalTensor> outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape3, npu::tile_fwk::SymbolicScalar::FromConcrete(shape3));
    auto& copyout = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L0C_COPY_OUT, {L0C}, {outcast1});

    EXPECT_TRUE(OpChecker::check(aMulB, OpChecker::CalcTypeChecker(OpCalcType::MATMUL)));
    EXPECT_FALSE(OpChecker::check(aMulB, OpChecker::CoreTypeChecker(OpCoreType::AIV)));
    EXPECT_TRUE(OpChecker::check(l1ToL0A, OpChecker::CalcTypeChecker(OpCalcType::MOVE_LOCAL),
                                 OpChecker::InputMemTypeChecker(MemoryType::MEM_L1),
                                 OpChecker::OutputMemTypeChecker({MemoryType::MEM_L0A, MemoryType::MEM_L0B})));
    EXPECT_FALSE(OpChecker::check(l1ToL0B, OpChecker::CalcTypeChecker(OpCalcType::MOVE_LOCAL),
                                  OpChecker::InputMemTypeChecker(MemoryType::MEM_L1),
                                  OpChecker::OutputMemTypeChecker(MemoryType::MEM_L0C)));
    EXPECT_TRUE(OpChecker::check(viewA, OpChecker::CalcTypeChecker(OpCalcType::MOVE_LOCAL)));
    EXPECT_FALSE(OpChecker::check(viewB, OpChecker::CalcTypeChecker(OpCalcType::MATMUL)));
    EXPECT_TRUE(OpChecker::check(copyout, OpChecker::CalcTypeChecker(OpCalcType::MOVE_OUT),
                                 OpChecker::InputMemTypeChecker(MemoryType::MEM_L0C),
                                 OpChecker::OutputMemTypeChecker({MemoryType::MEM_DEVICE_DDR})));
}
} // namespace tile_fwk
} // namespace npu
