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
 * \file test_codegen_sort.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
constexpr const unsigned OP_MAGIC3 = 3;
constexpr const unsigned OP_MAGIC4 = 4;
class TestCodegenSort : public CodegenTestBase {
public:
    TestCodegenSort() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .buildStatic = true}) {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

struct TopKParams {
    int32_t shape0;
    int32_t shape1;
    int32_t k;
    bool isLargest;
};
void TopKOnBoardFunc(TopKParams& params)
{
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 1}});

    int32_t shape0 = params.shape0;
    int32_t shape1 = params.shape1;
    int32_t k = params.k;
    bool isLargest = params.isLargest;

    std::vector<int64_t> input_shape = {shape0, shape1};
    std::vector<int64_t> output_shape = {shape0, k};
    TileShape::Current().SetVecTile({shape0, shape1});
    Tensor input_a(DataType::DT_FP32, input_shape, "A");
    auto output = std::make_tuple(Tensor(DataType::DT_FP32, output_shape, "npu_val"),
                                  Tensor(DataType::DT_FP32, output_shape, "resDics"));

    FUNCTION("TOPK_T_TILETENSOR", {input_a, std::get<0>(output), std::get<1>(output)})
    {
        output = TopK(input_a, k, -1, isLargest);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TOPK_T_TILETENSOR");
    std::string res = GenCodeByFunction(*function);

    std::string expect = R"(TBitSort<1, 0, 1>(ubTensor_2, ubTensor_0, ubTensor_3); // [opMagic:10006])";
    CheckStringExist(expect, res);

    expect = R"(TMrgSort<1, 32, 32>(ubTensor_2, ubTensor_2, ubTensor_6); // [opMagic:10007])";
    CheckStringExist(expect, res);

    expect = R"(TExtract<32, 0, 1>(ubTensor_8, ubTensor_2); // [opMagic:10008])";
    CheckStringExist(expect, res);

    expect = R"(TExtract<32, 1, 1>(ubTensor_10, ubTensor_2); // [opMagic:10010])";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenSort, TestTopKTileTensor)
{
    config::SetDebugOption(CFG_COMPILE_DBEUG_MODE, CFG_DEBUG_ALL);

    TopKParams params;
    params.shape0 = 128;
    params.shape1 = 32;
    params.k = 32;
    params.isLargest = true;
    TopKOnBoardFunc(params);
}

} // namespace npu::tile_fwk
