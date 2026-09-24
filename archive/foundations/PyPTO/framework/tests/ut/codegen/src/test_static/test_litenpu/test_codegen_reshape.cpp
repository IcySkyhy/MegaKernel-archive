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
 * \file test_codegen_reshape.cpp
 * \brief
 */

#include "include/test_codegen_reshape.h"

using namespace npu::tile_fwk;

TestCodeGenReshape::TestCodeGenReshape() = default;
TestCodeGenReshape::~TestCodeGenReshape() = default;

TestCodeGenReshape& TestCodeGenReshape::Instance()
{
    static TestCodeGenReshape instance;
    return instance;
}

void TestCodeGenReshape::test_reshape_fp16_001()
{
    PROGRAM("RESHAPE_FP16_001")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor operand(DT_FP16, {2, 2}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP16_001") { result = Reshape(operand, {1, 2, 1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp16_002()
{
    PROGRAM("RESHAPE_FP16_002")
    {
        TileShape::Current().SetVecTile({2, 16});
        Tensor operand(DT_FP16, {4, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP16_002") { result = Reshape(operand, {16}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp16_003()
{
    PROGRAM("RESHAPE_FP16_003")
    {
        TileShape::Current().SetVecTile({1, 1, 16});
        Tensor operand(DT_FP16, {2, 3, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP16_003") { result = Reshape(operand, {2, 12}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp16_004()
{
    PROGRAM("RESHAPE_FP16_004")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 16});
        Tensor operand(DT_FP16, {2, 2, 2, 2}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP16_004") { result = Reshape(operand, {4, 4}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp16_005()
{
    PROGRAM("RESHAPE_FP16_005")
    {
        TileShape::Current().SetVecTile({16});
        Tensor operand(DT_FP16, {8}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP16_005") { result = Reshape(operand, {2, 4}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp16_006()
{
    PROGRAM("RESHAPE_FP16_006")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor operand(DT_FP16, {3, 9}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP16_006") { result = Reshape(operand, {3, 3, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP16_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp16_007()
{
    PROGRAM("RESHAPE_FP16_007")
    {
        TileShape::Current().SetVecTile({1, 1, 16});
        Tensor operand(DT_FP16, {2, 2, 6}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP16_007") { result = Reshape(operand, {2, 2, 2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP16_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp16_008()
{
    PROGRAM("RESHAPE_FP16_008")
    {
        TileShape::Current().SetVecTile({2, 2, 2, 16});
        Tensor operand(DT_FP16, {4, 4, 4, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP16_008") { result = Reshape(operand, {256}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP16_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp16_009()
{
    PROGRAM("RESHAPE_FP16_009")
    {
        TileShape::Current().SetVecTile({16});
        Tensor operand(DT_FP16, {12}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP16_009") { result = Reshape(operand, {2, 2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP16_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp16_010()
{
    PROGRAM("RESHAPE_FP16_010")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor operand(DT_FP16, {2, 6}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP16_010") { result = Reshape(operand, {2, 2, 3, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP16_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp32_001()
{
    PROGRAM("RESHAPE_FP32_001")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor operand(DT_FP32, {2, 2}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP32_001") { result = Reshape(operand, {1, 2, 1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp32_002()
{
    PROGRAM("RESHAPE_FP32_002")
    {
        TileShape::Current().SetVecTile({2, 8});
        Tensor operand(DT_FP32, {4, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP32_002") { result = Reshape(operand, {16}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp32_003()
{
    PROGRAM("RESHAPE_FP32_003")
    {
        TileShape::Current().SetVecTile({1, 1, 8});
        Tensor operand(DT_FP32, {2, 3, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP32_003") { result = Reshape(operand, {2, 12}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp32_004()
{
    PROGRAM("RESHAPE_FP32_004")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 8});
        Tensor operand(DT_FP32, {2, 2, 2, 2}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP32_004") { result = Reshape(operand, {4, 4}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp32_005()
{
    PROGRAM("RESHAPE_FP32_005")
    {
        TileShape::Current().SetVecTile({8});
        Tensor operand(DT_FP32, {8}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP32_005") { result = Reshape(operand, {2, 4}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp32_006()
{
    PROGRAM("RESHAPE_FP32_006")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor operand(DT_FP32, {3, 9}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP32_006") { result = Reshape(operand, {3, 3, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp32_007()
{
    PROGRAM("RESHAPE_FP32_007")
    {
        TileShape::Current().SetVecTile({1, 1, 8});
        Tensor operand(DT_FP32, {2, 2, 6}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP32_007") { result = Reshape(operand, {2, 2, 2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP32_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp32_008()
{
    PROGRAM("RESHAPE_FP32_008")
    {
        TileShape::Current().SetVecTile({2, 2, 2, 8});
        Tensor operand(DT_FP32, {4, 4, 4, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP32_008") { result = Reshape(operand, {256}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp32_009()
{
    PROGRAM("RESHAPE_FP32_009")
    {
        TileShape::Current().SetVecTile({8});
        Tensor operand(DT_FP32, {12}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP32_009") { result = Reshape(operand, {2, 2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP32_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_fp32_010()
{
    PROGRAM("RESHAPE_FP32_010")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor operand(DT_FP32, {2, 6}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_FP32_010") { result = Reshape(operand, {2, 2, 3, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_FP32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_inplace_fp32_001()
{
    PROGRAM("RESHAPE_INPLACE_FP32_001")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor operand(DT_FP32, {2, 2}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INPLACE_FP32_001") { result = Reshape(operand, {1, 2, 1, 2}, true); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INPLACE_FP32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_inplace_fp32_002()
{
    PROGRAM("RESHAPE_INPLACE_FP32_002")
    {
        TileShape::Current().SetVecTile({2, 8});
        Tensor operand(DT_FP32, {4, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INPLACE_FP32_002") { result = Reshape(operand, {16}, true); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INPLACE_FP32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_inplace_fp32_003()
{
    PROGRAM("RESHAPE_INPLACE_FP32_003")
    {
        TileShape::Current().SetVecTile({1, 1, 8});
        Tensor operand(DT_FP32, {2, 3, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INPLACE_FP32_003") { result = Reshape(operand, {2, 12}, true); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INPLACE_FP32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_inplace_fp32_004()
{
    PROGRAM("RESHAPE_INPLACE_FP32_004")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 8});
        Tensor operand(DT_FP32, {2, 2, 2, 2}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INPLACE_FP32_004") { result = Reshape(operand, {4, 4}, true); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INPLACE_FP32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_inplace_fp32_005()
{
    PROGRAM("RESHAPE_INPLACE_FP32_005")
    {
        TileShape::Current().SetVecTile({8});
        Tensor operand(DT_FP32, {8}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INPLACE_FP32_005") { result = Reshape(operand, {2, 4}, true); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INPLACE_FP32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int8_001()
{
    PROGRAM("RESHAPE_INT8_001")
    {
        TileShape::Current().SetVecTile({1, 32});
        Tensor operand(DT_INT8, {2, 2}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT8_001") { result = Reshape(operand, {1, 2, 1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT8_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int8_002()
{
    PROGRAM("RESHAPE_INT8_002")
    {
        TileShape::Current().SetVecTile({2, 32});
        Tensor operand(DT_INT8, {4, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT8_002") { result = Reshape(operand, {16}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT8_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int8_003()
{
    PROGRAM("RESHAPE_INT8_003")
    {
        TileShape::Current().SetVecTile({32});
        Tensor operand(DT_INT8, {8}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT8_003") { result = Reshape(operand, {2, 4}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT8_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int8_004()
{
    PROGRAM("RESHAPE_INT8_004")
    {
        TileShape::Current().SetVecTile({1, 1, 32});
        Tensor operand(DT_INT8, {2, 3, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT8_004") { result = Reshape(operand, {2, 12}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT8_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int8_005()
{
    PROGRAM("RESHAPE_INT8_005")
    {
        TileShape::Current().SetVecTile({32});
        Tensor operand(DT_INT8, {12}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT8_005") { result = Reshape(operand, {2, 2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT8_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int16_001()
{
    PROGRAM("RESHAPE_INT16_001")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor operand(DT_INT16, {2, 2}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT16_001") { result = Reshape(operand, {1, 2, 1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int16_002()
{
    PROGRAM("RESHAPE_INT16_002")
    {
        TileShape::Current().SetVecTile({2, 16});
        Tensor operand(DT_INT16, {4, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT16_002") { result = Reshape(operand, {16}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int16_003()
{
    PROGRAM("RESHAPE_INT16_003")
    {
        TileShape::Current().SetVecTile({16});
        Tensor operand(DT_INT16, {8}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT16_003") { result = Reshape(operand, {2, 4}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int16_004()
{
    PROGRAM("RESHAPE_INT16_004")
    {
        TileShape::Current().SetVecTile({1, 1, 16});
        Tensor operand(DT_INT16, {2, 3, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT16_004") { result = Reshape(operand, {2, 12}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int16_005()
{
    PROGRAM("RESHAPE_INT16_005")
    {
        TileShape::Current().SetVecTile({16});
        Tensor operand(DT_INT16, {12}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT16_005") { result = Reshape(operand, {2, 2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int32_001()
{
    PROGRAM("RESHAPE_INT32_001")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor operand(DT_INT32, {2, 2}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT32_001") { result = Reshape(operand, {1, 2, 1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int32_002()
{
    PROGRAM("RESHAPE_INT32_002")
    {
        TileShape::Current().SetVecTile({2, 8});
        Tensor operand(DT_INT32, {4, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT32_002") { result = Reshape(operand, {16}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int32_003()
{
    PROGRAM("RESHAPE_INT32_003")
    {
        TileShape::Current().SetVecTile({8});
        Tensor operand(DT_INT32, {8}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT32_003") { result = Reshape(operand, {2, 4}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int32_004()
{
    PROGRAM("RESHAPE_INT32_004")
    {
        TileShape::Current().SetVecTile({1, 1, 8});
        Tensor operand(DT_INT32, {2, 3, 4}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT32_004") { result = Reshape(operand, {2, 12}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenReshape::test_reshape_int32_005()
{
    PROGRAM("RESHAPE_INT32_005")
    {
        TileShape::Current().SetVecTile({8});
        Tensor operand(DT_INT32, {12}, "operand");
        Tensor result;
        FUNCTION("RESHAPE_INT32_005") { result = Reshape(operand, {2, 2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RESHAPE_INT32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
