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
 * \file test_codegen_transpose.cpp
 * \brief
 */

#include "include/test_codegen_transpose.h"

using namespace npu::tile_fwk;

TestCodeGenTranspose::TestCodeGenTranspose() = default;
TestCodeGenTranspose::~TestCodeGenTranspose() = default;

TestCodeGenTranspose& TestCodeGenTranspose::Instance()
{
    static TestCodeGenTranspose instance;
    return instance;
}

void TestCodeGenTranspose::test_transpose_fp16_001()
{
    PROGRAM("TRANSPOSE_FP16_001")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor operand(DT_FP16, {2, 3}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_001") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp16_002()
{
    PROGRAM("TRANSPOSE_FP16_002")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor operand(DT_FP16, {3, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_002") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp16_003()
{
    PROGRAM("TRANSPOSE_FP16_003")
    {
        TileShape::Current().SetVecTile({1, 2, 16});
        Tensor operand(DT_FP16, {3, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_003") { result = Transpose(operand, {1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp16_004()
{
    PROGRAM("TRANSPOSE_FP16_004")
    {
        TileShape::Current().SetVecTile({1, 3, 16});
        Tensor operand(DT_FP16, {2, 3, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_004") { result = Transpose(operand, {0, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp16_005()
{
    PROGRAM("TRANSPOSE_FP16_005")
    {
        TileShape::Current().SetVecTile({1, 1, 2, 16});
        Tensor operand(DT_FP16, {2, 3, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_005") { result = Transpose(operand, {2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp16_006()
{
    PROGRAM("TRANSPOSE_FP16_006")
    {
        TileShape::Current().SetVecTile({1, 3, 1, 16});
        Tensor operand(DT_FP16, {2, 3, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_006") { result = Transpose(operand, {1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP16_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp16_007()
{
    PROGRAM("TRANSPOSE_FP16_007")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 2, 16});
        Tensor operand(DT_FP16, {2, 3, 4, 5, 6}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_007") { result = Transpose(operand, {3, 4}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP16_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp16_008()
{
    PROGRAM("TRANSPOSE_FP16_008")
    {
        TileShape::Current().SetVecTile({2, 16});
        Tensor operand(DT_FP16, {4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_008") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP16_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp16_009()
{
    PROGRAM("TRANSPOSE_FP16_009")
    {
        TileShape::Current().SetVecTile({1, 5, 16});
        Tensor operand(DT_FP16, {2, 5, 6}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_009") { result = Transpose(operand, {1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP16_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp16_010()
{
    PROGRAM("TRANSPOSE_FP16_010")
    {
        TileShape::Current().SetVecTile({1, 2, 2, 16});
        Tensor operand(DT_FP16, {3, 2, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_010") { result = Transpose(operand, {2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP16_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp32_001()
{
    PROGRAM("TRANSPOSE_FP32_001")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor operand(DT_FP32, {2, 3}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_001") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp32_002()
{
    PROGRAM("TRANSPOSE_FP32_002")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor operand(DT_FP32, {3, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_002") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp32_003()
{
    PROGRAM("TRANSPOSE_FP32_003")
    {
        TileShape::Current().SetVecTile({1, 2, 8});
        Tensor operand(DT_FP32, {3, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_003") { result = Transpose(operand, {1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp32_004()
{
    PROGRAM("TRANSPOSE_FP32_004")
    {
        TileShape::Current().SetVecTile({1, 3, 8});
        Tensor operand(DT_FP32, {2, 3, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_004") { result = Transpose(operand, {0, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp32_005()
{
    PROGRAM("TRANSPOSE_FP32_005")
    {
        TileShape::Current().SetVecTile({1, 1, 2, 8});
        Tensor operand(DT_FP32, {2, 3, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_005") { result = Transpose(operand, {2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp32_006()
{
    PROGRAM("TRANSPOSE_FP32_006")
    {
        TileShape::Current().SetVecTile({1, 3, 1, 8});
        Tensor operand(DT_FP32, {2, 3, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_006") { result = Transpose(operand, {1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp32_007()
{
    PROGRAM("TRANSPOSE_FP32_007")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 2, 8});
        Tensor operand(DT_FP32, {2, 3, 4, 5, 6}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_007") { result = Transpose(operand, {3, 4}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp32_008()
{
    PROGRAM("TRANSPOSE_FP32_008")
    {
        TileShape::Current().SetVecTile({2, 8});
        Tensor operand(DT_FP32, {4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_008") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp32_009()
{
    PROGRAM("TRANSPOSE_FP32_009")
    {
        TileShape::Current().SetVecTile({1, 5, 8});
        Tensor operand(DT_FP32, {2, 5, 6}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_009") { result = Transpose(operand, {1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_fp32_010()
{
    PROGRAM("TRANSPOSE_FP32_010")
    {
        TileShape::Current().SetVecTile({1, 2, 2, 8});
        Tensor operand(DT_FP32, {3, 2, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_010") { result = Transpose(operand, {2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int32_001()
{
    PROGRAM("TRANSPOSE_INT32_001")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor operand(DT_INT32, {2, 3}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT32_001") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int32_002()
{
    PROGRAM("TRANSPOSE_INT32_002")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor operand(DT_INT32, {3, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT32_002") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int32_003()
{
    PROGRAM("TRANSPOSE_INT32_003")
    {
        TileShape::Current().SetVecTile({1, 2, 8});
        Tensor operand(DT_INT32, {3, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT32_003") { result = Transpose(operand, {1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int32_004()
{
    PROGRAM("TRANSPOSE_INT32_004")
    {
        TileShape::Current().SetVecTile({1, 1, 2, 8});
        Tensor operand(DT_INT32, {2, 3, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT32_004") { result = Transpose(operand, {2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int32_005()
{
    PROGRAM("TRANSPOSE_INT32_005")
    {
        TileShape::Current().SetVecTile({2, 8});
        Tensor operand(DT_INT32, {4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT32_005") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int16_001()
{
    PROGRAM("TRANSPOSE_INT16_001")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor operand(DT_INT16, {2, 3}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT16_001") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int16_002()
{
    PROGRAM("TRANSPOSE_INT16_002")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor operand(DT_INT16, {3, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT16_002") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int16_003()
{
    PROGRAM("TRANSPOSE_INT16_003")
    {
        TileShape::Current().SetVecTile({1, 2, 16});
        Tensor operand(DT_INT16, {3, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT16_003") { result = Transpose(operand, {1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int16_004()
{
    PROGRAM("TRANSPOSE_INT16_004")
    {
        TileShape::Current().SetVecTile({1, 3, 16});
        Tensor operand(DT_INT16, {2, 3, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT16_004") { result = Transpose(operand, {0, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int16_005()
{
    PROGRAM("TRANSPOSE_INT16_005")
    {
        TileShape::Current().SetVecTile({1, 1, 2, 16});
        Tensor operand(DT_INT16, {2, 3, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT16_005") { result = Transpose(operand, {2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int16_006()
{
    PROGRAM("TRANSPOSE_INT16_006")
    {
        TileShape::Current().SetVecTile({1, 3, 1, 16});
        Tensor operand(DT_INT16, {2, 3, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT16_006") { result = Transpose(operand, {1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT16_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int16_007()
{
    PROGRAM("TRANSPOSE_INT16_007")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 2, 16});
        Tensor operand(DT_INT16, {2, 3, 4, 5, 6}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT16_007") { result = Transpose(operand, {3, 4}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT16_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int16_008()
{
    PROGRAM("TRANSPOSE_INT16_008")
    {
        TileShape::Current().SetVecTile({2, 16});
        Tensor operand(DT_INT16, {4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT16_008") { result = Transpose(operand, {0, 1}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT16_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int16_009()
{
    PROGRAM("TRANSPOSE_INT16_009")
    {
        TileShape::Current().SetVecTile({1, 5, 16});
        Tensor operand(DT_INT16, {2, 5, 6}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT16_009") { result = Transpose(operand, {1, 2}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT16_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int16_010()
{
    PROGRAM("TRANSPOSE_INT16_010")
    {
        TileShape::Current().SetVecTile({1, 2, 2, 16});
        Tensor operand(DT_INT16, {3, 2, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT16_010") { result = Transpose(operand, {2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT16_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenTranspose::test_transpose_int16_011()
{
    PROGRAM("TRANSPOSE_INT16_011")
    {
        TileShape::Current().SetVecTile({2, 2, 2, 16});
        Tensor operand(DT_INT16, {3, 2, 4, 5}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_INT16_011") { result = Transpose(operand, {2, 3}); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_INT16_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
