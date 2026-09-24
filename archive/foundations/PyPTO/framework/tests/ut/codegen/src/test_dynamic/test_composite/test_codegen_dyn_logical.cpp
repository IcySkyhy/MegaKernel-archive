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
 * \file test_codegen_dyn_logical.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
class TestCodegenDynLogical : public CodegenTestBase {
public:
    TestCodegenDynLogical() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}

    static void SetUpTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false); }

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

TEST_F(TestCodegenDynLogical, TestDynOpLogicalAnd)
{
    auto function = GenMockFuncDyn("TestDynOpLogicalAnd");
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensorInput1 = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorInput2 = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorRes = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_LOGICALAND, {localTensorInput1, localTensorInput2},
                                      {localTensorRes, localTensorTmp});

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynTlogicalAnd<float, float, 64, 64, 64>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynLogical, TestDynOpLogicalNot)
{
    auto function = GenMockFuncDyn("TestDynOpLogicalNot");
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensorInput = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorRes = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorTmpCond = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_LOGICALNOT, {localTensorInput}, {localTensorRes, localTensorTmpCond});

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynTlogicalNot<float, 64, 64>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

std::string TestLogicalBody(Opcode opcode)
{
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);

    auto function = GenMockFuncDyn(OpcodeManager::Inst().GetOpcodeStr(opcode));

    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto logicalInTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};

    std::vector<LogicalTensorPtr> input = {logicalInTensor};
    if (opcode == Opcode::OP_LOGICALAND) {
        input.emplace_back(logicalInTensor);
    }
    auto& op = function->AddOperation(opcode, input, {localOutTensor, localOutTensor});

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynLogical, LogicalAndTileTensor)
{
    std::string res = TestLogicalBody(Opcode::OP_LOGICALAND);
    std::string expect = R"!!!(TLogicalAnd(ubTensor_0, ubTensor_0, ubTensor_0, ubTensor_0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynLogical, LogicalNotTileTensor)
{
    std::string res = TestLogicalBody(Opcode::OP_LOGICALNOT);
    std::string expect = R"!!!(TLogicalNot(ubTensor_0, ubTensor_0, ubTensor_0);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk
