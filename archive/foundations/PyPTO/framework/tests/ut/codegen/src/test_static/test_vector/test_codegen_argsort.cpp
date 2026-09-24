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
 * \file test_codegen_argsort.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include "test_codegen_common.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include <vector>
#include <string>
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {

class TestCodegenArgSort : public CodegenTestBase {
public:
    TestCodegenArgSort() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

TEST_F(TestCodegenArgSort, TestArgSort)
{
    constexpr const int32_t shape0 = 64;

    std::vector<int64_t> input_shape = {shape0};
    std::vector<int64_t> output_shape = {shape0};

    TileShape::Current().SetVecTile({shape0});
    Tensor input_a(DT_FP32, input_shape, "A");
    Tensor output(DT_FP32, output_shape, "resDics");

    std::string funcName = "ARGSORT_T";
    config::SetBuildStatic(true);
    FUNCTION(funcName, {input_a, output}) { output = ArgSort(input_a, -1); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
} // namespace npu::tile_fwk
