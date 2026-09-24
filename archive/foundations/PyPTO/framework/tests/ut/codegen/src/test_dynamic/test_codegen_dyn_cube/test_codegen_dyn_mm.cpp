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
 * \file test_codegen_dyn_mm.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
class TestCodegenDynMM : public CodegenTestBase {
public:
    TestCodegenDynMM() : CodegenTestBase({.setIdGen = true}) {}
};

TEST_F(TestCodegenDynMM, TestDynMatmulTileTensor)
{
    auto function = GenMockFuncDyn("TestDynMatmulTileTensor");

    std::vector<int64_t> shape = {64, 64};
    std::vector<int64_t> shapeBias = {1, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensorA = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L0A, shape, dynValidShape});
    auto localTensorB = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L0B, shape, dynValidShape});
    auto localTensorBias = CreateLogicalTensor(
        {*function, DataType::DT_FP16, MemoryType::MEM_BT, shapeBias, dynValidShape});
    auto localOutTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_L0C, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_A_MUL_B, {localTensorA, localTensorB, localTensorBias},
                                      {localOutTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "has_bias", true);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TMatmul<TransMode::CAST_NONE, 1>(l0cTensor_0, l0aTensor_1, l0bTensor_2, btTensor_3);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynMM, TestMatmulMXTileTensor)
{
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    auto function = GenMockFuncDyn("TestMatmulMXTileTensor");

    std::vector<int64_t> mxShape = {64, 64};
    std::vector<int64_t> shapeBias = {1, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensorAMX = CreateLogicalTensor(
        {*function, DataType::DT_FP16, MemoryType::MEM_L0AMX, mxShape, dynValidShape});
    auto localTensorBMX = CreateLogicalTensor(
        {*function, DataType::DT_FP16, MemoryType::MEM_L0BMX, mxShape, dynValidShape});
    auto localTensorBias = CreateLogicalTensor(
        {*function, DataType::DT_FP16, MemoryType::MEM_BT, shapeBias, dynValidShape});
    auto localOutTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_L0C, mxShape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_A_MUL_B, {localTensorAMX, localTensorBMX, localTensorBias},
                                      {localOutTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "has_bias", true);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect = R"!!!(MatmulMX(l0cTensor_0, l0a_mxTensor_1, l0b_mxTensor_2, btTensor_3);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk
