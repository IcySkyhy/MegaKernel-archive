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
 * \file test_codegen_dyn_indexput.cpp
 * \brief Unit test for codegen.
 */

#include <iostream>

#include "gtest/gtest.h"

#include "interface/operation/opcode.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "passes/pass_mgr/pass_manager.h"
#include "codegen/codegen.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"
#include "interface/utils/id_gen.h"

namespace npu::tile_fwk {
class TestCodegenDynIndexPut : public CodegenTestBase {
public:
    TestCodegenDynIndexPut()
        : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .setIdGen = true, .resetTileTensorOnTearDown = true})
    {}
};

TEST_F(TestCodegenDynIndexPut, DynIndexPutTileTensor)
{
    int h = 8;
    std::vector<int64_t> vecTileShape = {h};
    std::vector<int64_t> shape1 = {h, h};
    std::vector<int64_t> shape2 = {h, h};
    std::vector<int64_t> shape3 = {h};
    TileShape::Current().SetVecTile(vecTileShape);
    Tensor input(DataType::DT_FP32, shape1, "input");
    Tensor values(DataType::DT_FP32, shape2, "values");
    Tensor indices1(DataType::DT_INT32, shape3, "indices1");
    Tensor output(DataType::DT_FP32, shape1, "output");
    std::string funcName = "IndexPut";
    FUNCTION(funcName, {input, values, indices1}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            IndexPut_(input, {indices1}, values);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);

    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TIndexPut<0, 1>(gmTensor_4, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 24, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 24, 1))), ubTensor_0, ubTensor_2, ubTensor_2, ubTensor_2, ubTensor_2);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynIndexPut, DynIndexPutDynUnaligned)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    auto function = GenMockFuncDyn("DynIndexPutDynUnaligned");
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShapeIdx = {32};
    auto input = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto value = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto indices = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, {32}, dynValidShapeIdx});
    auto result = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_INDEX_PUT, {input, value, indices}, {result});
    op.SetAttribute(OpAttributeKey::accumulate, true);
    op.SetAttribute(OpAttributeKey::indicesSize, 1);
    auto shapeImme = OpImmediate::Specified(shape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynTIndexPut<float, float, 2, 1, 64, 64, 1>((__gm__ float*)GET_PARAM_ADDR(param, 0, 1), (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 64, 1, 1, GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 1, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 1, 2, 1));
)!!!";
    EXPECT_EQ(res, expect);
}
} // namespace npu::tile_fwk
