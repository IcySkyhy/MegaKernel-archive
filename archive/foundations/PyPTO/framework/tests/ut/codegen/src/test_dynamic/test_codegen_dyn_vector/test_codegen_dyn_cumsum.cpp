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
 * \file test_codegen_dyn_cumsum.cpp
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
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynCumSum : public CodegenTestBase {
public:
    TestCodegenDynCumSum() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

void TestCodegenDynCumSumBody(int axis)
{
    std::vector<int64_t> vecTileShape = {5, 9};
    std::vector<int64_t> shape{12, 14};

    auto function = GenMockFuncDynUnary("CumSum", {shape, vecTileShape},
                                        [axis](Tensor& input, Tensor& output) { output = CumSum(input, axis); });

    std::string res = GenCodeByFunction(*function);
    std::string expect = axis == 0 ? "TCumOperation<3, 1>(ubTensor_2, ubTensor_0);" :
                                     "TCumOperation<3, 1>(ubTensor_7, ubTensor_2);";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynCumSum, test_CumSum_dim2_0) { TestCodegenDynCumSumBody(0); }

TEST_F(TestCodegenDynCumSum, test_CumSum_dim2_1) { TestCodegenDynCumSumBody(1); }
} // namespace npu::tile_fwk
