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
 * \file test_codegen_dyn_binary_brc.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynBinaryBrc : public CodegenTestBase {
public:
    TestCodegenDynBinaryBrc() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

// mul (32, 512), (32, 1)
TEST_F(TestCodegenDynBinaryBrc, TestMulDynamic)
{
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    std::vector<int64_t> shape1 = {32, 512};
    std::vector<int64_t> shape2 = {32, 1};
    TileShape::Current().SetVecTile({32, 256});
    Tensor input_a(DataType::DT_FP32, shape1, "A");
    Tensor input_b(DataType::DT_FP32, shape1, "B");
    Tensor output(DataType::DT_FP32, shape1, "C");
    ConfigManager::Instance();

    std::string funcName = "MUL_T";
    FUNCTION(funcName, {input_a, input_b, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            // add RowSumSingle to test brc case
            auto input_c = Sum(input_b, -1, true);
            output = Mul(input_a, input_c);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    (void)GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynBinaryBrc, TestAddBrcTileTensorDynamic)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
    ConfigManager::Instance();
    auto function = GenMockFuncDyn("TestAddBrcTileTensorDynamic", {32, 256});
    for (auto& subFunc : function->rootFunc_->programs_) {
        for (auto& op : subFunc.second->Operations()) {
            if (op.GetOpcode() == Opcode::OP_ADD) {
                op.SetAttribute(OpAttributeKey::brcOperand, std::vector<int64_t>{0, 0, 0, 0, 1});
                std::string res = GenOpCodeFromOp(*function, op);
                std::string expect =
                    R"!!!(TAdd<LastUse3Dim<0, 0, 1>, 0, 0, 0, 0, 1>(ubTensor_0, ubTensor_0, ubTensor_2);
)!!!";
                EXPECT_EQ(res, expect);
                break;
            }
        }
    }
}

} // namespace npu::tile_fwk
