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
 * \file test_codegen_dyn_permute.cpp
 * \brief Unit test for codegen OP_PERMUTE and OP_PERMUTE_ELEMENT dynamic layout coverage.
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
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynPermute : public CodegenTestBase {
public:
    TestCodegenDynPermute() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

TEST_F(TestCodegenDynPermute, TestPermuteLayout)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int DIM0 = 4;
    constexpr const int DIM1 = 8;
    constexpr const int DIM2 = 16;
    std::vector<int64_t> shape = {DIM0, DIM1, DIM2};

    TileShape::Current().SetVecTile({DIM0, DIM1, DIM2});

    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    ConfigManager::Instance();
    std::string funcName = "PERMUTE_LAYOUT_T";
    FUNCTION(funcName, {input}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Permute(input, {1, 0, 2});
        }
    }
#if ENABLE_HIDDENLOOP
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
#endif
    std::string res = GenCodeByFunction(*function);
    std::string expect = "TPermute<1, 0, 2, -1, -1, 3>";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynPermute, TestPermuteElementLayout)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int DIM0 = 4;
    constexpr const int DIM1 = 8;
    constexpr const int DIM2 = 16;
    std::vector<int64_t> shape = {DIM0, DIM1, DIM2};

    TileShape::Current().SetVecTile({DIM0, DIM1, DIM2});

    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    ConfigManager::Instance();
    std::string funcName = "PERMUTE_ELEM_LAYOUT_T";
    FUNCTION(funcName, {input}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Permute(input, {0, 2, 1});
        }
    }
#if ENABLE_HIDDENLOOP
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
#endif
    std::string res = GenCodeByFunction(*function);
    std::string expect = "TPermuteElewise<0, 2, 1, -1, -1, 3>";
    CheckStringExist(expect, res);
}

} // namespace npu::tile_fwk
