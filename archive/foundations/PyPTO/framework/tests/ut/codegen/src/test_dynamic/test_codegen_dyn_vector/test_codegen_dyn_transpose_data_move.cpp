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
 * \file test_codegen_dyn_transpose_data_move.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynTransposeDataMove : public CodegenTestBase {
public:
    TestCodegenDynTransposeDataMove() : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION, .setTileTensor = true})
    {}
};

void TestTransposeDataMoveBody(int dim = 3)
{
    std::vector<int64_t> shape = {2, 2, 8};
    std::vector<SymbolicScalar> dynValidShape = {2, 2, 8};
    if (dim == SHAPE_DIM4) {
        shape = {2, 2, 2, 8};
        dynValidShape = {2, 2, 2, 8};
    }
    auto function = GenMockFuncDyn("TestCodegenDynTransposeDataMove_dim" + std::to_string(dim), shape);
    auto ddrTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "TransposeDataMove"});
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {localTensor}, {ddrTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
    auto to_offset = OpImmediate::Specified({0, 0, 0});
    if (dim == SHAPE_DIM4) {
        to_offset = OpImmediate::Specified({0, 0, 0, 0});
    }
    auto shapeImme = OpImmediate::Specified(shape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, to_offset, shapeImme, shapeImme));
    op.SetOOpAtt(0, 0);
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);

    GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynTransposeDataMove, TransposeDataMoveDim3) { TestTransposeDataMoveBody(); }

TEST_F(TestCodegenDynTransposeDataMove, TransposeDataMoveDim4) { TestTransposeDataMoveBody(SHAPE_DIM4); }

class TestCodegenLayoutTransposeDataMove : public CodegenTestBase {
public:
    TestCodegenLayoutTransposeDataMove() : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION}) {}
};

TEST_F(TestCodegenLayoutTransposeDataMove, TransposeDataMoveLayout)
{
    constexpr int64_t dim = 3;
    constexpr int64_t shape0 = 8;
    constexpr int dim0 = 0;
    constexpr int dim1 = 1;
    std::vector<int64_t> shape(dim, shape0);
    TileShape::Current().SetVecTile(shape);
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");
    std::string funcName = "TransposeDataMoveLayout";
    FUNCTION(funcName, {input}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Transpose(input, {dim0, dim1});
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    (void)GenCodeByFunction(*function);
}

} // namespace npu::tile_fwk
