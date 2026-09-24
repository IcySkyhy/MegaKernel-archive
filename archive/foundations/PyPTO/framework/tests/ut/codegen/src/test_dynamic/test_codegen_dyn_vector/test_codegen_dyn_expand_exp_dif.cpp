/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_expand_exp_dif.cpp
 * \brief Unit test for expand_exp_dif codegen.
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

class TestCodegenDynExpandExpDif : public CodegenTestBase {
public:
    TestCodegenDynExpandExpDif()
        : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION, .setTileTensor = true, .tileTensorValue = true})
    {}
};

void TestExpandExpDif(const Shape& shape_x, const Shape& shape_y, const std::string& expect)
{
    TileShape::Current().SetVecTile(shape_x);
    Tensor input_x(DT_FP32, shape_x, "input_x");
    Tensor input_y(DT_FP32, shape_y, "input_y");
    Tensor output(DT_FP32, shape_x, "output");

    ConfigManager::Instance();
    std::string funcName = "TestExpandExpDif";
    FUNCTION(funcName, {input_x, input_y, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = ExpandExpDif(input_x, input_y);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    CheckStringExist(expect, GenCodeByFunction(*function));
}

TEST_F(TestCodegenDynExpandExpDif, TestExpandExpDifLastAxis)
{
    const Shape shape_x = {16, 128};
    const Shape shape_y = {16, 1};
    std::string expect = R"!!!(TExpandExpDif<0, 0, 0, 0, 2>(ubTensor_4, ubTensor_4, ubTensor_2);)!!!";
    TestExpandExpDif(shape_x, shape_y, expect);
}

TEST_F(TestCodegenDynExpandExpDif, TestExpandExpDifSecondaryLastAxis)
{
    const Shape shape_x = {16, 128};
    const Shape shape_y = {1, 128};
    std::string expect = R"!!!(TExpandExpDif<0, 0, 0, 2, 0>(ubTensor_0, ubTensor_0, ubTensor_2);)!!!";
    TestExpandExpDif(shape_x, shape_y, expect);
}
} // namespace npu::tile_fwk
