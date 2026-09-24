/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_erfc.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
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

class TestCodegenDynErfc : public CodegenTestBase {
public:
    TestCodegenDynErfc() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .setIdGen = true}) {}
};

TEST_F(TestCodegenDynErfc, ErfcFP32)
{
    std::vector<int64_t> shape = {16, 16};
    TileShape::Current().SetVecTile({16, 16});
    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, shape, "C");
    std::string funcName = "ErfcFP32";
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Erfc(input_a);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    std::vector<SymbolicScalar> dynValidShape = {16, 16};
    std::string res = GenCodeByFunction(*function);
    std::string expect = R"(TErfc(ubTensor_2, ubTensor_3, ubTensor_0);)";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
