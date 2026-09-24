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
 * \file test_codegen_dyn_scatter.cpp
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
class TestCodegenDynScatter : public CodegenTestBase {
public:
    TestCodegenDynScatter() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}

    static void SetUpTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false); }

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

TEST_F(TestCodegenDynScatter, TestDynOpScatterElement)
{
    std::vector<int64_t> shape = {64, 64};
    auto function = GenMockFuncDyn("TestDynOpScatterElement");

    Element scalaVal(DataType::DT_FP32, 1.0);

    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShapeIdx = {32};
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorIdx = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, {32}, dynValidShapeIdx});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_SCATTER_ELEMENT, {localTensorSrc, localTensorIdx}, {localTensorDst});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", 0);
    op.SetAttribute(OpAttributeKey::scalar, scalaVal);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", 0);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynTscatterElementS<float, float, float, 1, 1, 32, 1, 64, 64, 3, 0>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (float)1.f, 1, 1, 1, 32);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynScatter, TestOpDynScatter)
{
    auto function = GenMockFuncDyn("TestOpDynScatter");

    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShapeIdx = {32};
    auto localTensorSelf = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorIdx = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, {32}, dynValidShapeIdx});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_INT32, MemoryType::MEM_UB, {32}});

    auto& op = function->AddOperation(Opcode::OP_SCATTER, {localTensorSelf, localTensorIdx, localTensorSrc},
                                      {localTensorDst, localTensorTmp});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", 0);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynTscatter<float, float, 1, 1, 32, 1, 64, 64, 1, 64, 64, 3, 0>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 1, 32);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk
