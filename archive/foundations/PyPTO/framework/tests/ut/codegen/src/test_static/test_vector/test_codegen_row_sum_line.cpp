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
 * \file test_codegen_row_sum_line.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/function/function.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenRowSumLine : public CodegenTestBase {
public:
    TestCodegenRowSumLine() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .buildStatic = true}) {}
};

TEST_F(TestCodegenRowSumLine, TestOperationRowSumLineTileTensor)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int shape0 = 6;
    int shape1 = 1;
    int shape2 = 8;
    int shape3 = 1024;
    std::vector<int64_t> shape = {shape0 * shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0 * shape1, 1, shape3};
    TileShape::Current().SetVecTile({2, 8, 512});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, outshape, "C");

    std::string funcName = "Reduce3dimMoe_TILERENSOR";
    FUNCTION(funcName, {input_a, output}) { output = Sum(input_a, 1, true); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TRowSumLine<3>(ubTensor_2, ubTensor_0, ubTensor_3);
)!!!";
    CheckStringExist(expect, res);
}

} // namespace npu::tile_fwk
