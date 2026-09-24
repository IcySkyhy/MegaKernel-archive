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
 * \file test_codegen_where.cpp
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
#include "codegen/npu/op_print_param_def.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {
class TestCodegenWhere : public CodegenTestBase {
public:
    TestCodegenWhere() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .buildStatic = true}) {}

    static void SetUpTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false); }

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

Operation& GetWhereOp(Function* function, Opcode opCode, const LogicalTensors& inputs)
{
    Element scalaVal1(DataType::DT_FP32, 1.0);
    Element scalaVal2(DataType::DT_FP32, 2.0);

    if (opCode == Opcode::OP_WHERE_SS) {
        auto& op = function->AddOperation(
            opCode, {inputs[ToUnderlying(WhereOpIdx::condIdx)]},
            {inputs[ToUnderlying(WhereOpIdx::resIdx)], inputs[ToUnderlying(WhereOpIdx::tempIdx)]});
        std::vector<Element> scalars = {scalaVal1, scalaVal2};
        op.SetAttribute(OpAttributeKey::vectorScalar, scalars);
        op.SetAttribute(OP_ATTR_PREFIX + "whereBitMode", 0);
        return op;
    } else if (opCode == Opcode::OP_WHERE_ST || opCode == Opcode::OP_WHERE_TS) {
        auto& op = function->AddOperation(
            opCode, {inputs[ToUnderlying(WhereOpIdx::condIdx)], inputs[ToUnderlying(WhereOpIdx::src1Idx)]},
            {inputs[ToUnderlying(WhereOpIdx::resIdx)], inputs[ToUnderlying(WhereOpIdx::tempIdx)]});
        op.SetAttribute(OpAttributeKey::scalar, scalaVal1);
        op.SetAttribute(OP_ATTR_PREFIX + "whereBitMode", 0);
        return op;
    }

    EXPECT_EQ(opCode, Opcode::OP_WHERE_TT) << "Unsupported opcode: " << OpcodeManager::Inst().GetOpcodeStr(opCode);
    auto& op = function->AddOperation(
        opCode,
        {inputs[ToUnderlying(WhereOpIdx::condIdx)], inputs[ToUnderlying(WhereOpIdx::src0Idx)],
         inputs[ToUnderlying(WhereOpIdx::src1Idx)]},
        {inputs[ToUnderlying(WhereOpIdx::resIdx)], inputs[ToUnderlying(WhereOpIdx::tempIdx)]});
    op.SetAttribute(OP_ATTR_PREFIX + "whereBitMode", 0);
    return op;
}

void TestWhereBody(const Opcode opCode, const std::string& caseName, const std::string& expect,
                   bool isSupportTileTensor = false)
{
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }
    auto function = GenMockFuncStatic(caseName);
    std::vector<int64_t> shape = {64, 64};
    auto localTensorCond = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorInput = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorOther = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorResult = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto& op = GetWhereOp(function, opCode,
                          {localTensorResult, localTensorCond, localTensorTmp, localTensorInput, localTensorOther});

    std::string res = GenOpCodeFromOp(*function, op);
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenWhere, TestOpWhereSS)
{
    std::string expect =
        R"!!!(TileOp::Where_SS<float, float, /*DstRawShape*/ 1, 64, 64, /*ConditionRawShape*/ 1, 64, 64, /*Src0RawShape*/ 1, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, float(1.f), float(2.f), 1, 1, 1, 1);
)!!!";
    TestWhereBody(Opcode::OP_WHERE_SS, "TestOpWhereSS", expect);
}

TEST_F(TestCodegenWhere, TestOpWhereST)
{
    std::string expect =
        R"!!!(TileOp::Where_ST<float, float, /*DstRawShape*/ 1, 64, 64, /*ConditionRawShape*/ 1, 64, 64, /*Src0RawShape*/ 1, 64, 64>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, float(1.f), (__ubuf__ float*)UB_S0_E0, 1, 1, 1, 1);
)!!!";
    TestWhereBody(Opcode::OP_WHERE_ST, "TestOpWhereST", expect);
}

TEST_F(TestCodegenWhere, TestOpWhereTS)
{
    std::string expect =
        R"!!!(TileOp::Where_TS<float, float, /*DstRawShape*/ 1, 64, 64, /*ConditionRawShape*/ 1, 64, 64, /*Src0RawShape*/ 1, 64, 64>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, float(1.f), 1, 1, 1, 1);
)!!!";
    TestWhereBody(Opcode::OP_WHERE_TS, "TestOpWhereTS", expect);
}

TEST_F(TestCodegenWhere, TestOpWhereTT)
{
    std::string expect =
        R"!!!(TileOp::Where_TT<float, float, /*DstRawShape*/ 1, 64, 64, /*ConditionRawShape*/ 1, 64, 64, /*Src0RawShape*/ 1, 64, 64>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 1, 1);
)!!!";
    TestWhereBody(Opcode::OP_WHERE_TT, "TestOpWhereTT", expect);
}

TEST_F(TestCodegenWhere, TestOpWhereSS_TileTensor)
{
    std::string expect =
        R"!!!(TWhereSS(ubTensor_0, ubTensor_0, ubTensor_0, float(1.f), float(2.f));
)!!!";
    TestWhereBody(Opcode::OP_WHERE_SS, "TestOpWhereSS", expect, true);
}

TEST_F(TestCodegenWhere, TestOpWhereST_TileTensor)
{
    std::string expect =
        R"!!!(TWhereST(ubTensor_0, ubTensor_0, ubTensor_0, float(1.f), ubTensor_0);
)!!!";
    TestWhereBody(Opcode::OP_WHERE_ST, "TestOpWhereST", expect, true);
}

TEST_F(TestCodegenWhere, TestOpWhereTS_TileTensor)
{
    std::string expect =
        R"!!!(TWhereTS(ubTensor_0, ubTensor_0, ubTensor_0, ubTensor_0, float(1.f));
)!!!";
    TestWhereBody(Opcode::OP_WHERE_TS, "TestOpWhereTS", expect, true);
}

TEST_F(TestCodegenWhere, TestOpWhereTT_TileTensor)
{
    std::string expect =
        R"!!!(TWhereTT(ubTensor_0, ubTensor_0, ubTensor_0, ubTensor_0, ubTensor_0);
)!!!";
    TestWhereBody(Opcode::OP_WHERE_TT, "TestOpWhereTT", expect, true);
}

} // namespace npu::tile_fwk
