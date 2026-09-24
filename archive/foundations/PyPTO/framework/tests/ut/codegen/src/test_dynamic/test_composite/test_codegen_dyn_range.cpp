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
 * \file test_codegen_dyn_range.cpp
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
class TestCodegenDynRange : public CodegenTestBase {
public:
    TestCodegenDynRange() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

TEST_F(TestCodegenDynRange, TestDynOpRange)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);

    auto function = GenMockFuncDyn("TestDynOpRange");
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    Element start(DataType::DT_FP32, 1.0);
    Element step(DataType::DT_FP32, 2.0);
    Element size(DataType::DT_FP32, 3.0);
    int64_t idx = 0;
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_RANGE, {}, {localTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "START", start);
    op.SetAttribute(OP_ATTR_PREFIX + "STEP", step);
    op.SetAttribute(OP_ATTR_PREFIX + "SIZE", size);
    SymbolicScalar tileIdx(idx);
    op.SetAttribute(OpAttributeKey::dynScalar, tileIdx);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynRange<float, 64>((__ubuf__ float*)UB_S0_E0, 64, 1.f, 2.f, ((int64_t)(0)));
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynRange, RangeTileTensor)
{
    auto function = GenMockFuncDyn("RangeTileTensor");
    std::vector<int64_t> rangeShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, rangeShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, rangeShape});
    localOutTensor->UpdateDynValidShape(dynValidShape);
    localTensor->UpdateDynValidShape(dynValidShape);
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    std::vector<int64_t> offset = {0, 0};

    auto& op = function->AddOperation(Opcode::OP_RANGE, {localTensor}, {localOutTensor});
    Element start(DataType::DT_FP32, 1.0);
    op.SetAttribute(OP_ATTR_PREFIX + "START", start);
    op.SetAttribute(OP_ATTR_PREFIX + "STEP", start);
    op.SetAttribute(OP_ATTR_PREFIX + "SIZE", start);

    GenOpCodeFromOp(*function, op, {.isMainBlk = true});
}

TEST_F(TestCodegenDynRange, RangeUnsupportedDtype)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    auto function = GenMockFuncDyn("RangeUnsupportedDtype");
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_BOOL, MemoryType::MEM_UB, {64, 64}});
    auto& op = function->AddOperation(Opcode::OP_RANGE, {}, {localTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "START", Element(DataType::DT_INT32, 1));
    op.SetAttribute(OP_ATTR_PREFIX + "STEP", Element(DataType::DT_INT32, 1));
    op.SetAttribute(OP_ATTR_PREFIX + "SIZE", Element(DataType::DT_INT32, 1));
    EXPECT_EQ(GenOpCodeFromOp(*function, op), CG_ERROR);
}
} // namespace npu::tile_fwk
