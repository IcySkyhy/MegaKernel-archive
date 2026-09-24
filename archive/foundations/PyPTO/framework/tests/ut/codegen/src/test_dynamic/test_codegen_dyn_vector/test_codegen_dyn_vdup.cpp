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
 * \file test_codegen_dyn_vdup.cpp
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
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynVdup : public CodegenTestBase {
public:
    TestCodegenDynVdup() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

TEST_F(TestCodegenDynVdup, TestDynVdupUnaligned)
{
    std::vector<int64_t> shape{2, 256, 7};
    Element src(DataType::DT_FP32, 2.0);

    TileShape::Current().SetVecTile({1, 256, 16});

    Tensor output(DataType::DT_FP32, shape, "C");
    std::string funcName = "TestDynVdupUnaligned";

    FUNCTION(funcName, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = npu::tile_fwk::Full(src, DataType::DT_FP32, shape);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    (void)GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynVdup, TestDynVdupInt16)
{
    std::vector<int64_t> shape{2, 256, 7};
    Element src(DataType::DT_INT16, static_cast<int16_t>(2));

    TileShape::Current().SetVecTile({1, 256, 16});

    Tensor output(DataType::DT_INT16, shape, "C");
    std::string funcName = "TestDynVdupInt16";

    FUNCTION(funcName, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = npu::tile_fwk::Full(src, DataType::DT_INT16, shape);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    (void)GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynVdup, TestDynVdupUInt16)
{
    std::vector<int64_t> shape{2, 256, 7};
    Element src(DataType::DT_UINT16, static_cast<uint16_t>(2));

    TileShape::Current().SetVecTile({1, 256, 16});

    Tensor output(DataType::DT_UINT16, shape, "C");
    std::string funcName = "TestDynVdupUInt16";

    FUNCTION(funcName, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = npu::tile_fwk::Full(src, DataType::DT_UINT16, shape);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    (void)GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynVdup, TestDynVdupUInt32)
{
    std::vector<int64_t> shape{2, 256, 7};
    Element src(DataType::DT_UINT32, static_cast<uint32_t>(2));

    TileShape::Current().SetVecTile({1, 256, 16});

    Tensor output(DataType::DT_UINT32, shape, "C");
    std::string funcName = "TestDynVdupUInt32";

    FUNCTION(funcName, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = npu::tile_fwk::Full(src, DataType::DT_UINT32, shape);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    (void)GenCodeByFunction(*function);
}
} // namespace npu::tile_fwk
