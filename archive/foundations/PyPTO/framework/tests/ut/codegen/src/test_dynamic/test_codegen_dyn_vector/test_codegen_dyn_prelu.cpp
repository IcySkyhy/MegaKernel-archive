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
 * \file test_codegen_dyn_prelu.cpp
 * \brief Unit test for prelu codegen.
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

class TestCodegenDynPrelu : public CodegenTestBase {
public:
    TestCodegenDynPrelu() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

TEST_F(TestCodegenDynPrelu, PreluNormal)
{
    MockFuncDynBinaryConf config;
    config.shapeA = {32, 256};
    config.shapeB = {256};
    config.outputShape = {32, 256};
    auto function = GenMockFuncDynBinary(
        "PRELU_NORMAL", config, [](Tensor& input, Tensor& weight, Tensor& output) { output = PReLU(input, weight); });

    (void)GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynPrelu, PreluFP16)
{
    MockFuncDynBinaryConf config;
    config.shapeA = {16, 128};
    config.shapeB = {128};
    config.outputShape = {16, 128};
    config.dtype = DT_FP16;
    auto function = GenMockFuncDynBinary(
        "PRELU_FP16", config, [](Tensor& input, Tensor& weight, Tensor& output) { output = PReLU(input, weight); });

    (void)GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynPrelu, PreluBF16)
{
    MockFuncDynBinaryConf config;
    config.shapeA = {8, 64};
    config.shapeB = {64};
    config.outputShape = {8, 64};
    config.dtype = DT_BF16;
    auto function = GenMockFuncDynBinary(
        "PRELU_BF16", config, [](Tensor& input, Tensor& weight, Tensor& output) { output = PReLU(input, weight); });

    (void)GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynPrelu, Prelu4D)
{
    MockFuncDynBinaryConf config;
    config.shapeA = {2, 64, 8, 8};
    config.shapeB = {64};
    config.outputShape = {2, 64, 8, 8};
    auto function = GenMockFuncDynBinary(
        "PRELU_4D", config, [](Tensor& input, Tensor& weight, Tensor& output) { output = PReLU(input, weight); });

    (void)GenCodeByFunction(*function);
}

} // namespace npu::tile_fwk
