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
 * \file test_codegen_dyn_tri.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynTri : public CodegenTestBase {
public:
    TestCodegenDynTri() : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION}) {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

TEST_F(TestCodegenDynTri, TestTriU)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int S = 16;
    constexpr const int D = 32;
    std::vector<int64_t> shape = {S, D};
    SymbolicScalar diagonal = 0;
    TileShape::Current().SetVecTile({S, D});

    Tensor inputSrc(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    ConfigManager::Instance();
    std::string funcName = "TRIU";
    FUNCTION(funcName, {inputSrc, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = TriU(inputSrc, diagonal);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TTriUL<1>(ubTensor_2, ubTensor_0, (int)((RUNTIME_COA_GET_PARAM(37))));
)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
