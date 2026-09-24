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
 * \file test_codegen_dyn_cmp.cpp
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
class TestCodegenDynCmp : public CodegenTestBase {
public:
    TestCodegenDynCmp() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

TEST_F(TestCodegenDynCmp, TestDynOpCmp)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);

    auto function = GenMockFuncDyn("TestDynOpCmp");

    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensorInput1 = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorInput2 = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorRes = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_CMP, {localTensorInput1, localTensorInput2},
                                      {localTensorRes, localTensorTmp});
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", 0);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynCompare<float, 1, 64, 64, 1, 64, 64, 1, 64, 64, 0, 0>((__ubuf__ uint8_t*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64, (__ubuf__ uint8_t*)UB_S0_E0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCmp, TestDynOpCmpS)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);

    auto function = GenMockFuncDyn("TestDynOpCmpS");

    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    Element scalaVal(DataType::DT_FP32, 1.0);

    auto localTensorInput1 = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorRes = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_CMPS, {localTensorInput1}, {localTensorRes, localTensorTmp});
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", 0);
    op.SetAttribute(OpAttributeKey::scalar, scalaVal);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynCmps<float, 1, 64, 64, 1, 64, 64, 0, 0>((__ubuf__ uint8_t*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64, (__ubuf__ uint8_t*)UB_S0_E0, 1.f);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCmp, CmpTileTensor)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    auto function = GenMockFuncDyn("CmpTileTensor");

    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    localTensor->UpdateDynValidShape(dynValidShape);
    localOutTensor->UpdateDynValidShape(dynValidShape);
    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};

    auto& op = function->AddOperation(Opcode::OP_CMP, {localTensor, localTensor}, {localOutTensor, localOutTensor});
    int64_t cmpParam = 0;
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", cmpParam);
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", cmpParam);

    std::string res = GenOpCodeFromOp(*function, op);

    std::string expect =
        R"!!!(TCompare<0, 0>(ubTensor_0, ubTensor_0, ubTensor_0, ubTensor_0);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk
