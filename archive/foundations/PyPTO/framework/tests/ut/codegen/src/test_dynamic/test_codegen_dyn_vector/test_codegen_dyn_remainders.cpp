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
 * \file test_codegen_dyn_remainders.cpp
 * \brief Unit test for remainders codegen.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynRemainderS : public CodegenTestBase {
public:
    TestCodegenDynRemainderS() : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION}) {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

TEST_F(TestCodegenDynRemainderS, TestRemainderS)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int S = 16;
    constexpr const int D = 32;
    std::vector<int64_t> shape = {S, D};
    Element other(DT_INT16, static_cast<int16_t>(2));
    TileShape::Current().SetVecTile({S, D});

    Tensor inputSrc(DT_INT16, shape, "input");
    Tensor output(DT_INT16, shape, "output");
    ConfigManager::Instance();
    std::string funcName = "RemainderS";
    FUNCTION(funcName, {inputSrc, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Remainder(inputSrc, other);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    std::string
        expect = R"!!!(TRemainderS<int16_t, pto::RemSAlgorithm::DEFAULT>(ubTensor_4, ubTensor_2, 2, ubTensor_5);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynRemainderS, TestRemainderRS)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int S = 16;
    constexpr const int D = 32;
    std::vector<int64_t> shape = {S, D};
    Element self(DT_INT32, static_cast<int32_t>(10));
    TileShape::Current().SetVecTile({S, D});

    Tensor inputSrc(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    ConfigManager::Instance();
    std::string funcName = "RemainderRS";
    FUNCTION(funcName, {inputSrc, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Remainder(self, inputSrc);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    std::string
        expect = R"!!!(TRemainderRS<float, pto::RemAlgorithm::DEFAULT>(ubTensor_2, ubTensor_0, 10.f, ubTensor_3);)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
