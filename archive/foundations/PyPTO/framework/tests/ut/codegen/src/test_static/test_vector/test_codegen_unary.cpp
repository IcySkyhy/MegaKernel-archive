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
 * \file test_codegen_unary.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include <vector>
#include <string>
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenUnary : public CodegenTestBase {
public:
    TestCodegenUnary()
        : CodegenTestBase(
              {.compileStage = CS_EXECUTE_GRAPH, .buildStatic = true, .setTileTensor = true, .tileTensorValue = false})
    {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

void TestRowMaxSingleBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape,
                          std::string name)
{
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");
    FUNCTION(name, {input_a, output}) { output = Amax(input_a, -1, true); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenUnary, RowMaxSingleDim2) { TestRowMaxSingleBody({8, 128}, {8, 1}, {2, 64}, "ROWMAXSINGLE_DIM2"); }

TEST_F(TestCodegenUnary, RowMaxSingleDim3)
{
    TestRowMaxSingleBody({8, 4, 128}, {8, 4, 1}, {2, 1, 64}, "ROWMAXSINGLE_DIM3");
}

TEST_F(TestCodegenUnary, RowMaxSingleDim4)
{
    TestRowMaxSingleBody({8, 4, 4, 128}, {8, 4, 4, 1}, {2, 1, 1, 64}, "ROWMAXSINGLE_DIM4");
}

void TestRowSumSingleBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape,
                          std::string name)
{
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");
    FUNCTION(name, {input_a, output}) { output = Sum(input_a, -1, true); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenUnary, RowSumSingleDim2) { TestRowSumSingleBody({8, 128}, {8, 1}, {2, 64}, "ROWSUMSINGLE_DIM2"); }

TEST_F(TestCodegenUnary, RowSumSingleDim3)
{
    TestRowSumSingleBody({8, 4, 128}, {8, 4, 1}, {2, 1, 64}, "ROWSUMSINGLE_DIM3");
}

TEST_F(TestCodegenUnary, RowSumSingleDim4)
{
    TestRowSumSingleBody({8, 4, 4, 128}, {8, 4, 4, 1}, {2, 1, 1, 64}, "ROWSUMSINGLE_DIM4");
}

void TestTransposeVnchwconvBody(std::vector<int64_t> shape, std::vector<int64_t> outShape,
                                std::vector<int> transposeShape, std::vector<int64_t> tileShape, std::string name,
                                bool isSupportTileTensor = false)
{
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }
    TileShape::Current().SetVecTile(tileShape);
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, outShape, "output");
    FUNCTION(name, {input, output}) { output = Transpose(input, transposeShape); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenUnary, TransposeVnchwconvDim2)
{
    TestTransposeVnchwconvBody({16, 32}, {32, 16}, {0, 1}, {16, 16}, "TRANSPOSE_VNCHWCONV_DIM2");
}

TEST_F(TestCodegenUnary, TransposeVnchwconvDim4)
{
    TestTransposeVnchwconvBody({1, 2, 32, 16}, {1, 2, 16, 32}, {3, 2}, {1, 1, 16, 16}, "TRANSPOSE_VNCHWCONV_DIM4");
}

TEST_F(TestCodegenUnary, TransposeVnchwconvDim5)
{
    TestTransposeVnchwconvBody({1, 1, 2, 32, 16}, {1, 1, 2, 16, 32}, {3, 4}, {1, 1, 1, 16, 16},
                               "TRANSPOSE_VNCHWCONV_DIM5");
}

TEST_F(TestCodegenUnary, TransposeVnchwconvDim2TileTensor)
{
    TestTransposeVnchwconvBody({16, 32}, {32, 16}, {0, 1}, {16, 16}, "TransposeVnchwconvDim2TileTensor", true);
}

Function& TestFullBody(std::vector<int64_t> shape, std::vector<int64_t> tileShape, std::string name,
                       bool isSupportTileTensor = false)
{
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, shape, "C");
    Element value(DataType::DT_FP32, 2.0f);
    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) { output = Full(value, DT_FP32, shape, {}); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
    return *function;
}

TEST_F(TestCodegenUnary, FullDim2TileTensor)
{
    Function& func = TestFullBody({32, 32}, {16, 16}, "FULL_DIM2_TILETENSOR", true);
    std::string res = GetResultFromCpp(func);
    std::string expect = R"!!!(TVecDup<float>(ubTensor_0, 2.f);)!!!";
    CheckStringExist(expect, res);
}

Function& TestCastBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape,
                       std::string name, bool isSupportTileTensor = false)
{
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_INT32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");
    FUNCTION(name, {input_a, output}) { output = Cast(input_a, DT_FP32); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
    return *function;
}

TEST_F(TestCodegenUnary, CastDim1) { TestCastBody({128}, {128}, {64}, "CastDim1"); }

TEST_F(TestCodegenUnary, CastDim1TileTensor)
{
    Function& func = TestCastBody({128}, {128}, {64}, "CastDim1TileTensor", true);
    std::string res = GetResultFromCpp(func);
    std::string expect = R"!!!(TCast<LastUse2Dim<0, 0>, 0, pto::SaturationMode::OFF>(ubTensor_2, ubTensor_0);
)!!!";
    CheckStringExist(expect, res);
}

Function& TestExpandBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape,
                         std::string name, bool isSupportTileTensor = false)
{
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    } else {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");

    FUNCTION(name, {input_a, output}) { output = Expand(input_a, outShape); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
    return *function;
}

TEST_F(TestCodegenUnary, ExpandDim2Axis0TileTensor)
{
    Function& func = TestExpandBody({1, 22}, {22, 22}, {2, 2}, "ExpandDim2Axis0TileTensor", true);
    std::string res = GetResultFromCpp(func);
    std::string expect = R"!!!(TExpand<LastUse2Dim<0, 0>, 3>(ubTensor_2, ubTensor_0);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenUnary, ExpandDim2Axis0) { TestExpandBody({1, 22}, {22, 22}, {2, 2}, "ExpandDim2Axis0"); }

TEST_F(TestCodegenUnary, ExpandDim4Axis0)
{
    TestExpandBody({1, 22, 8, 17}, {4, 22, 8, 17}, {2, 16, 4, 8}, "ExpandDim4Axis0");
}

TEST_F(TestCodegenUnary, ExpandDim4Axis1)
{
    TestExpandBody({4, 1, 8, 17}, {4, 22, 8, 17}, {2, 16, 4, 8}, "ExpandDim4Axis1");
}

void TestRowSumBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape,
                    std::string name, unsigned axis)
{
    TileShape::Current().SetVecTile(tileShape);

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, outShape, "C");

    FUNCTION(name, {input_a, output}) { output = Sum(input_a, axis, true); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenUnary, RowSumDim4Axis2)
{
    TestRowSumBody({3, 2, 8, 255}, {3, 2, 1, 255}, {2, 8, 8, 255}, "ROWSUMAXIS2", 2);
}

TEST_F(TestCodegenUnary, RowSumDim3Axis1) { TestRowSumBody({2, 8, 255}, {2, 1, 255}, {8, 8, 255}, "ROWSUMAXIS1", 1); }

TEST_F(TestCodegenUnary, RowSumDim2Axis0) { TestRowSumBody({8, 255}, {1, 255}, {8, 255}, "ROWSUMAXIS0", 0); }

TEST_F(TestCodegenUnary, TestVecDup)
{
    std::vector<int64_t> shape{32, 1, 32};
    Element src(DataType::DT_INT32, static_cast<int64_t>(2));
    std::string funcName = "VECDUP";
    TileShape::Current().SetVecTile({16, 1, 16});

    Tensor output(DataType::DT_INT32, shape, "C");
    FUNCTION(funcName, {output}) { output = npu::tile_fwk::Full(src, DT_INT32, shape); }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenUnary, TestVecDupUnaligned)
{
    std::vector<int64_t> shape{2, 2, 256, 7};
    Element src(DataType::DT_FP32, 2.0);
    TileShape::Current().SetVecTile({1, 1, 256, 16});

    Tensor output(DataType::DT_FP32, shape, "C");

    std::string funcName = "VECDUP_T";
    FUNCTION(funcName, {output}) { output = npu::tile_fwk::Full(src, DT_FP32, shape); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenUnary, TestVecDupInt16)
{
    std::vector<int64_t> shape{32, 1, 32};
    Element src(DataType::DT_INT16, static_cast<int16_t>(2));
    std::string funcName = "VECDUP_INT16";
    TileShape::Current().SetVecTile({16, 1, 16});

    Tensor output(DataType::DT_INT16, shape, "C");
    FUNCTION(funcName, {output}) { output = npu::tile_fwk::Full(src, DT_INT16, shape); }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenUnary, TestVecDupUInt16)
{
    std::vector<int64_t> shape{32, 1, 32};
    Element src(DataType::DT_UINT16, static_cast<uint16_t>(2));
    std::string funcName = "VECDUP_UINT16";
    TileShape::Current().SetVecTile({16, 1, 16});

    Tensor output(DataType::DT_UINT16, shape, "C");
    FUNCTION(funcName, {output}) { output = npu::tile_fwk::Full(src, DT_UINT16, shape); }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenUnary, TestVecDupUInt32)
{
    std::vector<int64_t> shape{32, 1, 32};
    Element src(DataType::DT_UINT32, static_cast<uint32_t>(2));
    std::string funcName = "VECDUP_UINT32";
    TileShape::Current().SetVecTile({16, 1, 16});

    Tensor output(DataType::DT_UINT32, shape, "C");
    FUNCTION(funcName, {output}) { output = npu::tile_fwk::Full(src, DT_UINT32, shape); }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenUnary, TestRowMaxLine)
{
    config::SetBuildStatic(true);
    std::vector<int64_t> shape = {2, 2, 64};
    auto function = GenMockFuncStatic("TestCVSyncBody", shape);
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto& op = function->AddOperation(Opcode::OP_ROWMAXLINE, {localTensorSrc}, {localTensorDst});
    op.SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::Trowmaxline_<float, 1, 2, 2, 64, 2, 2, 64, 2, 2, 64, 2>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenUnary, TestExpDefaultPrecision)
{
    config::SetBuildStatic(true);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    std::vector<int64_t> shape = {2, 2, 64};
    auto function = GenMockFuncStatic("TestExpDefaultPrecision", shape);
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto& op = function->AddOperation(Opcode::OP_EXP, {localTensorSrc}, {localTensorDst});

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect = R"!!!(TExp<pto::ExpAlgorithm::DEFAULT>(ubTensor_0, ubTensor_0);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenUnary, TestExpHighPrecision)
{
    config::SetBuildStatic(true);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    std::vector<int64_t> shape = {2, 2, 64};
    auto function = GenMockFuncStatic("TestExpHighPrecision", shape);
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto& op = function->AddOperation(Opcode::OP_EXP, {localTensorSrc}, {localTensorDst});
    op.SetAttribute("precision_type", static_cast<int64_t>(1)); // HIGH_PRECISION

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect = R"!!!(TExp<pto::ExpAlgorithm::HIGH_PRECISION>(ubTensor_0, ubTensor_0);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenUnary, TestSqrtDefaultPrecision)
{
    config::SetBuildStatic(true);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    std::vector<int64_t> shape = {2, 2, 64};
    auto function = GenMockFuncStatic("TestSqrtDefaultPrecision", shape);
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto& op = function->AddOperation(Opcode::OP_SQRT, {localTensorSrc}, {localTensorDst});

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect = R"!!!(TSqrt<pto::SqrtAlgorithm::DEFAULT>(ubTensor_0, ubTensor_0);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenUnary, TestSqrtHighPrecision)
{
    config::SetBuildStatic(true);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    std::vector<int64_t> shape = {2, 2, 64};
    auto function = GenMockFuncStatic("TestSqrtHighPrecision", shape);
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto& op = function->AddOperation(Opcode::OP_SQRT, {localTensorSrc}, {localTensorDst});
    op.SetAttribute("precision_type", static_cast<int64_t>(1)); // HIGH_PRECISION

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect = R"!!!(TSqrt<pto::SqrtAlgorithm::HIGH_PRECISION>(ubTensor_0, ubTensor_0);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenUnary, TestLogDefaultPrecision)
{
    config::SetBuildStatic(true);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    std::vector<int64_t> shape = {2, 2, 64};
    auto function = GenMockFuncStatic("TestLogDefaultPrecision", shape);
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto& op = function->AddOperation(Opcode::OP_LN, {localTensorSrc}, {localTensorDst});

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect = R"!!!(TLog<pto::LogAlgorithm::DEFAULT>(ubTensor_0, ubTensor_0);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenUnary, TestLogHighPrecision)
{
    config::SetBuildStatic(true);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    std::vector<int64_t> shape = {2, 2, 64};
    auto function = GenMockFuncStatic("TestLogHighPrecision", shape);
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto& op = function->AddOperation(Opcode::OP_LN, {localTensorSrc}, {localTensorDst});
    op.SetAttribute("precision_type", static_cast<int64_t>(1)); // HIGH_PRECISION

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect = R"!!!(TLog<pto::LogAlgorithm::HIGH_PRECISION>(ubTensor_0, ubTensor_0);
)!!!";
    CheckStringExist(expect, res);
}

} // namespace npu::tile_fwk
