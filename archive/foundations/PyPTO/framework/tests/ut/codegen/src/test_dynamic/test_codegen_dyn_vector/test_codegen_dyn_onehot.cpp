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
 * \file test_codegen_dyn_onehot.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynOneHot : public CodegenTestBase {
public:
    TestCodegenDynOneHot() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

TEST_F(TestCodegenDynOneHot, OneHotLayout)
{
    std::vector<int64_t> indicesShape = {32, 32};
    std::vector<int64_t> outputShape = {32, 32, 1};
    TileShape::Current().SetVecTile(outputShape);
    Tensor inputStub(DT_INT32, indicesShape, "input");
    Tensor outputStub(DT_INT32, outputShape, "output");

    std::string funcName = "OneHotLayout";
    FUNCTION(funcName, {inputStub}, {outputStub})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            outputStub = OneHot(inputStub, 1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    (void)GenCodeByFunction(*function);
}
} // namespace npu::tile_fwk
