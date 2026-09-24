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
 * \file test_codegen_spillout.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen_op.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenSpillOut : public CodegenTestBase {
public:
    TestCodegenSpillOut()
        : CodegenTestBase(
              {.compileStage = CS_EXECUTE_GRAPH, .buildStatic = true, .setTileTensor = true, .tileTensorValue = false})
    {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override
    {
        CodegenTestBase::SetUp();
        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
    }
};

TEST_F(TestCodegenSpillOut, UBSpillOut)
{
    auto function = GenMockFuncStatic("UBSpillOut");
    std::vector<int64_t> shape = {64, 64};
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "UBSpillOut",
                                          SYMBOL_STACK_BASE, dynValidShape});
    int64_t baseOffset{0};
    ddrTensor->SetAttr(OpAttributeKey::workspaceBaseOffset, baseOffset);
    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto& op = function->AddOperation(Opcode::OP_COPY_OUT, {ubTensor}, {ddrTensor});
    auto shapeImme = OpImmediate::Specified(shape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::UBCopyOut<float, 1, 1, 1, 64, 64, /*dst stride*/ 1, 1, 64, 64,/*src stride*/ 1, 1, 64, 64 >((__gm__ float*)GMStackBase, (__ubuf__ float*)UB_S0_E0);
)!!!";

    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenSpillOut, UBSpillOutTileTensor)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);

    auto function = GenMockFuncStatic("UBSpillOutTileTensor");
    std::vector<int64_t> shape = {64, 64};
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape,
                                          "UBSpillOutTileTensor", SYMBOL_STACK_BASE, dynValidShape});
    int64_t baseOffset{16};
    ddrTensor->SetAttr(OpAttributeKey::workspaceBaseOffset, baseOffset);
    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto& op = function->AddOperation(Opcode::OP_COPY_OUT, {ubTensor}, {ddrTensor});
    auto shapeImme = OpImmediate::Specified(shape);
    op.SetOpAttribute(
        std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({16, 16}), shapeImme, shapeImme));
    op.SetAttr(OpAttributeKey::workspaceBaseOffset, static_cast<int64_t>(16));

    std::shared_ptr<SymbolManager> symbolManager;
    auto cop = GenOpCloudNPUFromOp(*function, op, symbolManager);

    std::string res = symbolManager->GenTileTensorDefList();
    std::string expect =
        R"!!!(GMTileTensorFP32Dim2_0 gmTensor_0((__gm__ float*)((__gm__ uint8_t*)GMStackBase + 16), DynLayout2Dim(Shape2Dim(64, 64), Stride2Dim(64, 1)));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E0_T);
)!!!";
    EXPECT_EQ(res, expect);

    res = cop.GenOpCode();
    expect = R"!!!(TStoreVec<TStoreConfigVec<pto::AtomicType::AtomicNone>>(gmTensor_0, ubTensor_1, Coord2Dim(0, 0));
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenSpillOut, L1SpillOut)
{
    auto function = GenMockFuncStatic("L1SpillOut");
    std::vector<int64_t> shape = {64, 64};
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "L1SpillOut",
                                          SYMBOL_STACK_BASE, dynValidShape});
    int64_t baseOffset{0};
    ddrTensor->SetAttr(OpAttributeKey::workspaceBaseOffset, baseOffset);
    auto l1Tensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_L1_COPY_OUT, {l1Tensor}, {ddrTensor});
    auto shapeImme = OpImmediate::Specified(shape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));

    GenOpCodeFromOp(*function, op);
}
} // namespace npu::tile_fwk
