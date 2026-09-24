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
 * \file test_codegen_dyn_transpose_conv.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynVnchwconv : public CodegenTestBase {
public:
    TestCodegenDynVnchwconv() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

void TestDynVnchwconvBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int> transposeShape,
                          std::vector<int64_t> tileShape, std::string funcName)
{
    TileShape::Current().SetVecTile(tileShape);
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, outShape, "output");

    FUNCTION(funcName, {input, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Transpose(input, transposeShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    (void)GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynVnchwconv, TransposeDynVnchwconvDim3)
{
    TestDynVnchwconvBody({2, 32, 16}, {2, 16, 32}, {2, 1}, {1, 16, 16}, "TRANSPOSE_DYN_VNCHWCONV_DIM4");
}
} // namespace npu::tile_fwk
