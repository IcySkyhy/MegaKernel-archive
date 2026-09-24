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
 * \file test_codegen_dyn_scalar.cpp
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

class TestCodegenDynScalar : public CodegenTestBase {
public:
    TestCodegenDynScalar()
        : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION, .resetTileTensorOnTearDown = true})
    {}
};

TEST_F(TestCodegenDynScalar, TestScalarAdds)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);

    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 35};

    TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "res");
    std::string funcName = "ScalarAddS";
    FUNCTION(funcName, {input, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = ScalarAddS(input, Element(DataType::DT_FP32, 127.0), true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TileOp::DynTSadds<float, /*DstRawShape*/ 2, 40, /*Src0RawShape*/ 2, 40, 1>((__ubuf__ float*)UB_S0_E320, (__ubuf__ float*)UB_S0_E320, 127.f,)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynScalar, TestScalarDivs)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);

    std::vector<int64_t> vecTileShape = {128, 128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{s, b * s, 35};

    TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1], vecTileShape[2]);
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "res");
    std::string funcName = "ScalarDivS";
    FUNCTION(funcName, {input, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = ScalarDivS(input, Element(DataType::DT_FP32, 127.0), true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TileOp::DynTSdivs<float, /*DstRawShape*/ 1, 2, 40, /*Src0RawShape*/ 1, 2, 40, 1>((__ubuf__ float*)UB_S0_E320, (__ubuf__ float*)UB_S0_E320, 127.f,)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynScalar, TestAddsTileTensor)
{
    int s = 32;
    Tensor t0(DT_FP32, {-1, s}, "t0"); // [32*8, 32]
    Tensor out(DT_FP32, {-1, s}, "out");
    TileShape::Current().SetVecTile({128, 64});

    auto funcName = "ADDS_TILETENSOR";
    FUNCTION(funcName, {t0}, {out})
    {
        auto shape0 = GetInputShape(t0, 0);
        auto loop1 = (shape0 + s - 1) / s;
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, idx, LoopRange(loop1))
        {
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});
            auto t = Add(t0s, Element(DT_FP32, 3.0));
            Assemble(t, {idx * s, 0}, out);
        }
    }

#if ENABLE_HIDDENLOOP
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX);
#endif
    std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TAddS<LastUse2Dim<0, 0>, float>(ubTensor_0, ubTensor_0, 3.f);
)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
