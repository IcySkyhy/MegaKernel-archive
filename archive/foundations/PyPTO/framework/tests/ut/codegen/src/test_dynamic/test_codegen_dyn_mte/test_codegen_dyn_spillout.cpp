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
 * \file test_codegen_dyn_spillout.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynSpillOut : public CodegenTestBase {
public:
    TestCodegenDynSpillOut()
        : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION, .resetTileTensorOnTearDown = true})
    {}
};

TEST_F(TestCodegenDynSpillOut, UBSpillOut)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);

    const std::vector<int64_t> shape = {64, 64};
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto function = GenMockFuncDyn("UBSpillOut");
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "UBSpillOut",
                                          SYMBOL_STACK_BASE, dynValidShape});
    int64_t baseOffset{0};
    ddrTensor->SetAttr(OpAttributeKey::workspaceBaseOffset, baseOffset);
    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_COPY_OUT, {ubTensor}, {ddrTensor});
    auto shapeImme = OpImmediate::Specified(shape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);

    GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynSpillOut, L1SpillOut)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);

    std::vector<int64_t> shape = {64, 64};
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto function = GenMockFuncDyn("L1SpillOut");
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "L1SpillOut",
                                          SYMBOL_STACK_BASE, dynValidShape});
    int64_t baseOffset{0};
    ddrTensor->SetAttr(OpAttributeKey::workspaceBaseOffset, baseOffset);
    auto l1Tensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_COPY_OUT, {l1Tensor}, {ddrTensor});
    auto shapeImme = OpImmediate::Specified(shape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);

    GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynSpillOut, L1SpillTileTensor)
{
    std::vector<int64_t> shape = {64, 64};
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto function = GenMockFuncDyn("L1SpillTileTensor");
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "L1SpillOut",
                                          SYMBOL_STACK_BASE, dynValidShape});
    int64_t baseOffset{0};
    ddrTensor->SetAttr(OpAttributeKey::workspaceBaseOffset, baseOffset);
    auto l1Tensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto& op = function->rootFunc_->programs_[0]->AddOperation(Opcode::OP_COPY_OUT, {l1Tensor}, {ddrTensor});
    auto shapeImme = OpImmediate::Specified(shape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);

    auto& op2 = function->rootFunc_->programs_[0]->AddOperation(Opcode::OP_COPY_IN, {ddrTensor}, {l1Tensor});
    op2.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_L1, shapeImme, shapeImme));
    op2.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);
    op2.SetAttribute(OP_ATTR_PREFIX + "copy_in_mode", 0);

    std::string res = GenCodeByFunction(*function);
    std::string
        expect = R"!!!(TStore<TStoreConfig<CopyOutMode::NZ2ND, 0, 0>>(gmTensor_9, l1Tensor_10, Coord2Dim(0, 0));)!!!";
    CheckStringExist(expect, res);

    expect =
        R"!!!(TLoad<CopyInMode::ND2ND, PaddingMode::NORMAL_PADDING_MODE, -1>(l1Tensor_10, gmTensor_9, Coord2Dim(0, 0), Coord2Dim(0, 0), 64, 64);)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
