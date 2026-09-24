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
 * \file test_codegen_dyn_sort.cpp
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
constexpr const unsigned OP_MAGIC5 = 5;
constexpr const unsigned TOPK_OP_X_IDX = 0;
constexpr const unsigned TOPK_OP_Y_IDX = 1;
constexpr const unsigned TOPK_OP_TMP_IDX = 2;

class TestCodegenDynSort : public CodegenTestBase {
public:
    TestCodegenDynSort() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .setTileTensor = true}) {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

struct TestContext {
    Function* function;
    std::shared_ptr<LogicalTensor> localTensor;
    std::shared_ptr<LogicalTensor> localOutTensor;
    std::shared_ptr<LogicalTensor> localTmpTensor;
    Operation* op;
};

std::string generateCodeForOp(TestContext& tc) { return GenOpCodeFromOp(*tc.function, *tc.op, {.isMainBlk = true}); }

TestContext prepareSortParamForUT(Opcode opcode)
{
    auto function = GenMockFuncDyn(OpcodeManager::Inst().GetOpcodeStr(opcode));
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, OP_MAGIC3, dynValidShape});
    auto localOutTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, OP_MAGIC4, dynValidShape});
    auto localTmpTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, OP_MAGIC5, dynValidShape});

    TestContext param;
    param.function = function;
    param.localTensor = localTensor;
    param.localOutTensor = localOutTensor;
    param.localTmpTensor = localTmpTensor;
    if (opcode == Opcode::OP_BITSORT || opcode == Opcode::OP_MRGSORT) {
        auto& op = function->AddOperation(opcode, {localTensor}, {localOutTensor, localTmpTensor});
        param.op = &op;
    } else {
        auto& op = function->AddOperation(opcode, {localTensor}, {localOutTensor});
        param.op = &op;
    }

    return param;
}

TEST_F(TestCodegenDynSort, TestDynBitSort)
{
    auto param = prepareSortParamForUT(Opcode::OP_BITSORT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "axis", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);

    std::string res = generateCodeForOp(param);
    std::string expect =
        R"!!!(TileOp::DynBitSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynMrgSort)
{
    auto param = prepareSortParamForUT(Opcode::OP_MRGSORT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "axis", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "kvalue", 1);

    std::string res = generateCodeForOp(param);
    std::string expect =
        R"!!!(TileOp::DynMrgSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 1, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynExtract)
{
    auto param = prepareSortParamForUT(Opcode::OP_EXTRACT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "kvalue", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "mode", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);

    std::string res = generateCodeForOp(param);
    std::string expect =
        R"!!!(TileOp::DynExtract<float, float, 1, 64, 64, 1, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynTiledMgrSort)
{
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto function = GenMockFuncDyn("TestDynTiledMgrSort");

    Element scalaVal(DataType::DT_FP32, 1.0);

    auto localTensorInput1 = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorInput2 = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorInput3 = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorRes = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_TILEDMRGSORT,
                                      {localTensorInput1, localTensorInput2, localTensorInput3, localTensorInput3},
                                      {localTensorRes, localTensorTmp});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", 0);
    op.SetAttribute(OpAttributeKey::scalar, scalaVal);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynTiledMrgSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 64, 0>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64, 64, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

Operation& GetTopkOp(Function* function, Opcode opCode, const LogicalTensors& tensors)
{
    if (opCode == Opcode::OP_TOPK_SORT) {
        auto& op = function->AddOperation(opCode, {tensors[TOPK_OP_Y_IDX]},
                                          {tensors[TOPK_OP_TMP_IDX], tensors[TOPK_OP_X_IDX]});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", 0);
        SymbolicScalar startIdx(1);
        op.SetAttribute(OpAttributeKey::dynScalar, startIdx);
        return op;
    }

    auto& op = function->AddOperation(opCode, {tensors[TOPK_OP_X_IDX]}, {tensors[TOPK_OP_Y_IDX]});
    return op;
}

void TestTopkBody(Opcode opCode, const std::string& expect)
{
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto function = GenMockFuncDyn("TestDynTopkSort");
    auto yVar = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto tmpVar = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto xVar = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = GetTopkOp(function, opCode, {xVar, yVar, tmpVar});

    std::string res = GenOpCodeFromOp(*function, op);
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynTopkSort)
{
    std::string expect =
        R"!!!(TileOp::DynTopKSort<float, 64, 64>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1);
)!!!";
    TestTopkBody(Opcode::OP_TOPK_SORT, expect);
}

TEST_F(TestCodegenDynSort, TestDynTopkMerge)
{
    std::string expect =
        R"!!!(TileOp::DynTopKMerge<float, 64, 32>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0);
)!!!";
    TestTopkBody(Opcode::OP_TOPK_MERGE, expect);
}

TEST_F(TestCodegenDynSort, TestDynTopkExract)
{
    std::string expect =
        R"!!!(TileOp::DynTopKExtract<float, float, 64, 64, 64, 32>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0);
)!!!";
    TestTopkBody(Opcode::OP_TOPK_EXTRACT, expect);
}

TEST_F(TestCodegenDynSort, TestDynTwoTileMrgSort)
{
    auto param = prepareSortParamForUT(Opcode::OP_TWOTILEMRGSORT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "firstshape", 32);

    std::string res = generateCodeForOp(param);
    std::string expect =
        R"!!!(TileOp::DynTwoTileMrgSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 32>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynExtractSingle)
{
    auto param = prepareSortParamForUT(Opcode::OP_EXTRACT_SINGLE);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "maskmode", 0);
    std::string res = generateCodeForOp(param);
    std::string expect =
        R"!!!(TileOp::DynExtractSingle<float, float, 1, 1, 64, 64, 1, 1, 64, 64, 0, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestRadixSelectFP32)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);

    std::vector<int64_t> srcShape = {64, 64};
    std::vector<int64_t> dstShape = {64, 16};
    std::vector<int64_t> tmpShape = {64, 1024};
    std::vector<SymbolicScalar> srcDynValidShape = {64, 64};
    std::vector<SymbolicScalar> dstDynValidShape = {64, 16};
    std::vector<SymbolicScalar> tmpDynValidShape = {64, 1024};
    TileShape::Current().SetVecTile({64, 64});
    auto function = GenMockFuncDyn("TestRadixSelectFP32");
    auto localTensorInput = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, srcShape, srcDynValidShape});
    auto localTensorValue = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, dstShape, dstDynValidShape});
    auto localTensorIndex = CreateLogicalTensor(
        {*function, DataType::DT_INT32, MemoryType::MEM_UB, dstShape, dstDynValidShape});
    auto localTensorTmp = CreateLogicalTensor(
        {*function, DataType::DT_UINT8, MemoryType::MEM_UB, tmpShape, tmpDynValidShape});
    auto& op = function->AddOperation(Opcode::OP_RADIX_SELECT, {localTensorInput},
                                      {localTensorValue, localTensorIndex, localTensorTmp});
    op.SetAttribute(OP_ATTR_PREFIX + "kvalue", 16);
    op.SetAttribute(OP_ATTR_PREFIX + "order", 1);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TRadixSelect<16, 1>(ubTensor_0, ubTensor_1, ubTensor_2, ubTensor_3);
)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
