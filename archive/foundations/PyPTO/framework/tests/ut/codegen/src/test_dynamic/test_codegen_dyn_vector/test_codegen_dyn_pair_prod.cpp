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
 * \file test_codegen_dyn_pair_prod.cpp
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

class TestCodegenDynPairProd : public CodegenTestBase {
public:
    TestCodegenDynPairProd() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .setIdGen = true}) {}
};

TEST_F(TestCodegenDynPairProd, PairProdLayout)
{
    int shape0 = 32;
    int shape1 = 32;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};

    TileShape::Current().SetVecTile({16, 16});

    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, shape, "B");

    std::string funcName = "PairProd";
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Prod(input_a, -1, true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    std::string res = GenCodeByFunction(*function);
    const std::string expect = R"(TPairProd(ubTensor_4, ubTensor_0, ubTensor_2);)";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
