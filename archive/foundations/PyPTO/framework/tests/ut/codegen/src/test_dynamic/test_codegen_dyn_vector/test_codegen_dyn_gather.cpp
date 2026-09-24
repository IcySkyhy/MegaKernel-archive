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
 * \file test_codegen_dyn_gather.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynGather : public CodegenTestBase {
public:
    TestCodegenDynGather() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH}) {}
};

constexpr const int GATHER_SHAPE0 = 16;
constexpr const int GATHER_SHAPE1 = 32;

TEST_F(TestCodegenDynGather, TestGather)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    constexpr const int S2 = 32;
    constexpr const int D = 64;
    constexpr const int B = 1;
    constexpr const int S = 32;
    std::vector<int64_t> shape0 = {S2, D};
    std::vector<int64_t> shape1 = {B, S};
    int axis = 0;
    std::vector<int64_t> shape2 = {B, S, D};

    TileShape::Current().SetVecTile({1, GATHER_SHAPE0, GATHER_SHAPE1});

    Tensor inputSrc0(DT_FP32, shape0, "x");
    Tensor inputSrc1(DT_INT32, shape1, "indices");
    Tensor output(DT_FP32, shape2, "output");

    ConfigManager::Instance();
    std::string funcName = "GATHER_T";
    FUNCTION(funcName, {inputSrc0, inputSrc1, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Gather(inputSrc0, inputSrc1, axis);
        }
    }
#if ENABLE_HIDDENLOOP
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
#endif
    (void)GenCodeByFunction(*function);
}
TEST_F(TestCodegenDynGather, TestGatherLayout)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int S2 = 32;
    constexpr const int D = 64;
    constexpr const int B = 1;
    constexpr const int S = 32;
    std::vector<int64_t> shape0 = {S2, D};
    std::vector<int64_t> shape1 = {B, S};
    int axis = 0;
    std::vector<int64_t> shape2 = {B, S, D};

    TileShape::Current().SetVecTile({1, GATHER_SHAPE0, GATHER_SHAPE1});

    Tensor inputSrc0(DT_FP32, shape0, "x");
    Tensor inputSrc1(DT_INT32, shape1, "indices");
    Tensor output(DT_FP32, shape2, "output");

    ConfigManager::Instance();
    std::string funcName = "GATHER_T";
    FUNCTION(funcName, {inputSrc0, inputSrc1, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Gather(inputSrc0, inputSrc1, axis);
        }
    }
#if ENABLE_HIDDENLOOP
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
#endif
    (void)GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynGather, GatherFromUB)
{
    auto function = GenMockFuncDyn("GatherFromUB");
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShapeIdx = {32};
    auto params = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto indices = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, {32}, dynValidShapeIdx});
    auto result = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_GATHER_FROM_UB, {params, indices}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", 0);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynTgatherFromUB_<float, float, /*before*/ 1, /*after*/ 64, /*axis_shape*/ 64, 1, 1, 1, 32, 64>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 1, 32);
)!!!";
    EXPECT_EQ(res, expect);
}
} // namespace npu::tile_fwk
