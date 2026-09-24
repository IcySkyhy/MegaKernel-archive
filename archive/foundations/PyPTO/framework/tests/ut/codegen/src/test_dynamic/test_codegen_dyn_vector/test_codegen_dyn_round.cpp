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
 * \file test_codegen_dyn_round.cpp
 * \brief Unit test for codegen.
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
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynRound : public CodegenTestBase {
public:
    TestCodegenDynRound()
        : CodegenTestBase(
              {.compileStage = CS_EXECUTE_GRAPH, .setTileTensor = true, .tileTensorValue = true, .setIdGen = true})
    {}

    static void SetUpTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false); }

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

TEST_F(TestCodegenDynRound, TestDynOpRound)
{
    std::vector<int64_t> shape = {64, 64};
    auto function = GenMockFuncDynUnary("TestDynOpRound", {shape},
                                        [](Tensor& input, Tensor& output) { output = Round(input, 1); });
    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TRound<float>(ubTensor_2, ubTensor_3, ubTensor_0, 10.f);
)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
