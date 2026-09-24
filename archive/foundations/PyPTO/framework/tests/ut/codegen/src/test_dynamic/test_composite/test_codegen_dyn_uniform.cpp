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
 * \file test_codegen_dyn_uniform.cpp
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
class TestCodegenDynUniform : public CodegenTestBase {
public:
    TestCodegenDynUniform() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

TEST_F(TestCodegenDynUniform, UniformTileTensorFP32)
{
    auto function = GenMockFuncDyn("UniformTileTensorFP32");
    std::vector<int64_t> shape = {64};
    std::vector<SymbolicScalar> dynValidShape = {64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto tempTensor = CreateLogicalTensor({*function, DataType::DT_UINT8, MemoryType::MEM_UB, {256}});
    localTensor->UpdateDynValidShape(dynValidShape);
    tempTensor->UpdateDynValidShape({256});
    std::vector<SymbolicScalar> dynoffset = {0};
    std::vector<int64_t> offset = {0};

    uint64_t key = 12345678901234;
    uint64_t counter1 = 0;
    uint16_t rounds = 10;

    auto& op = function->AddOperation(Opcode::OP_UNIFORM, {}, {localTensor, tempTensor});
    std::vector<Element> scalars = {Element(DT_UINT64, key), Element(DT_UINT64, counter1), Element(DT_UINT16, rounds),
                                    Element(DT_INT32, static_cast<int32_t>(DT_FP32))};
    op.SetAttribute(OpAttributeKey::vectorScalar, scalars);
    SymbolicScalar tileIdx(0);
    op.SetAttribute(OpAttributeKey::dynScalar, tileIdx);
    op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", shape);

    std::string res = GenOpCodeFromOp(*function, op);
    EXPECT_TRUE(res.find("TUniform") != std::string::npos);
}

TEST_F(TestCodegenDynUniform, UniformTileTensorFP16)
{
    auto function = GenMockFuncDyn("UniformTileTensorFP16");
    std::vector<int64_t> shape = {64};
    std::vector<SymbolicScalar> dynValidShape = {64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_UB, shape});
    auto tempTensor = CreateLogicalTensor({*function, DataType::DT_UINT8, MemoryType::MEM_UB, {512}});
    localTensor->UpdateDynValidShape(dynValidShape);
    tempTensor->UpdateDynValidShape({512});
    std::vector<SymbolicScalar> dynoffset = {0};
    std::vector<int64_t> offset = {0};

    uint64_t key = 12345678901234;
    uint64_t counter1 = 0;
    uint16_t rounds = 10;

    auto& op = function->AddOperation(Opcode::OP_UNIFORM, {}, {localTensor, tempTensor});
    std::vector<Element> scalars = {Element(DT_UINT64, key), Element(DT_UINT64, counter1), Element(DT_UINT16, rounds),
                                    Element(DT_INT32, static_cast<int32_t>(DT_FP16))};
    op.SetAttribute(OpAttributeKey::vectorScalar, scalars);
    SymbolicScalar tileIdx(0);
    op.SetAttribute(OpAttributeKey::dynScalar, tileIdx);
    op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", shape);

    std::string res = GenOpCodeFromOp(*function, op);
    EXPECT_TRUE(res.find("TUniform") != std::string::npos);
}

TEST_F(TestCodegenDynUniform, UniformTileTensorBF16)
{
    auto function = GenMockFuncDyn("UniformTileTensorBF16");
    std::vector<int64_t> shape = {64};
    std::vector<SymbolicScalar> dynValidShape = {64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_BF16, MemoryType::MEM_UB, shape});
    auto tempTensor = CreateLogicalTensor({*function, DataType::DT_UINT8, MemoryType::MEM_UB, {512}});
    localTensor->UpdateDynValidShape(dynValidShape);
    tempTensor->UpdateDynValidShape({512});
    std::vector<SymbolicScalar> dynoffset = {0};
    std::vector<int64_t> offset = {0};

    uint64_t key = 12345678901234;
    uint64_t counter1 = 0;
    uint16_t rounds = 10;

    auto& op = function->AddOperation(Opcode::OP_UNIFORM, {}, {localTensor, tempTensor});
    std::vector<Element> scalars = {Element(DT_UINT64, key), Element(DT_UINT64, counter1), Element(DT_UINT16, rounds),
                                    Element(DT_INT32, static_cast<int32_t>(DT_BF16))};
    op.SetAttribute(OpAttributeKey::vectorScalar, scalars);
    SymbolicScalar tileIdx(0);
    op.SetAttribute(OpAttributeKey::dynScalar, tileIdx);
    op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", shape);

    std::string res = GenOpCodeFromOp(*function, op);
    EXPECT_TRUE(res.find("TUniform") != std::string::npos);
}

} // namespace npu::tile_fwk
