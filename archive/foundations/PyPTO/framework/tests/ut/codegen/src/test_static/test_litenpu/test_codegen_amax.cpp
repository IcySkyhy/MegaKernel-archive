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
 * \file test_codegen_amax.cpp
 * \brief
 */

#include "include/test_codegen_amax.h"

using namespace npu::tile_fwk;

TestCodeGenAmax::TestCodeGenAmax() = default;
TestCodeGenAmax::~TestCodeGenAmax() = default;

TestCodeGenAmax& TestCodeGenAmax::Instance()
{
    static TestCodeGenAmax instance;
    return instance;
}

void TestCodeGenAmax::test_amax_fp16_003()
{
    PROGRAM("AMAX_FP16_003")
    {
        TileShape::Current().SetVecTile({2, 32});
        Tensor operand(DT_FP16, {4, 128}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_003") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_004()
{
    PROGRAM("AMAX_FP16_004")
    {
        TileShape::Current().SetVecTile({1, 128});
        Tensor operand(DT_FP16, {4, 130}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_004") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_005()
{
    PROGRAM("AMAX_FP16_005")
    {
        TileShape::Current().SetVecTile({1, 2, 32});
        Tensor operand(DT_FP16, {2, 4, 160}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_005") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_006()
{
    PROGRAM("AMAX_FP16_006")
    {
        TileShape::Current().SetVecTile({1, 2, 128});
        Tensor operand(DT_FP16, {2, 4, 140}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_006") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_007()
{
    PROGRAM("AMAX_FP16_007")
    {
        TileShape::Current().SetVecTile({1, 5, 32});
        Tensor operand(DT_FP16, {2, 5, 152}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_007") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_008()
{
    PROGRAM("AMAX_FP16_008")
    {
        TileShape::Current().SetVecTile({1, 3, 160});
        Tensor operand(DT_FP16, {2, 3, 170}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_008") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_009()
{
    PROGRAM("AMAX_FP16_009")
    {
        TileShape::Current().SetVecTile({2, 1, 2, 16});
        Tensor operand(DT_FP16, {5, 2, 4, 176}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_009") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_010()
{
    PROGRAM("AMAX_FP16_010")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 128});
        Tensor operand(DT_FP16, {5, 2, 4, 130}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_010") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_011()
{
    PROGRAM("AMAX_FP16_011")
    {
        TileShape::Current().SetVecTile({1, 1, 5, 32});
        Tensor operand(DT_FP16, {2, 3, 5, 134}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_011") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_012()
{
    PROGRAM("AMAX_FP16_012")
    {
        TileShape::Current().SetVecTile({2, 2, 3, 32});
        Tensor operand(DT_FP16, {4, 2, 6, 135}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_012") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_013()
{
    PROGRAM("AMAX_FP16_013")
    {
        TileShape::Current().SetVecTile({1, 1, 4, 128});
        Tensor operand(DT_FP16, {6, 2, 4, 130}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_013") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_014()
{
    PROGRAM("AMAX_FP16_014")
    {
        TileShape::Current().SetVecTile({1, 2, 1, 128});
        Tensor operand(DT_FP16, {3, 2, 3, 139}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_014") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp16_015()
{
    PROGRAM("AMAX_FP16_015")
    {
        TileShape::Current().SetVecTile({3, 3, 5, 32});
        Tensor operand(DT_FP16, {6, 3, 5, 141}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP16_015") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP16_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_003()
{
    PROGRAM("AMAX_FP32_003")
    {
        TileShape::Current().SetVecTile({2, 32});
        Tensor operand(DT_FP32, {4, 128}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_003") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_004()
{
    PROGRAM("AMAX_FP32_004")
    {
        TileShape::Current().SetVecTile({1, 128});
        Tensor operand(DT_FP32, {4, 130}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_004") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_005()
{
    PROGRAM("AMAX_FP32_005")
    {
        TileShape::Current().SetVecTile({1, 2, 32});
        Tensor operand(DT_FP32, {2, 4, 160}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_005") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_006()
{
    PROGRAM("AMAX_FP32_006")
    {
        TileShape::Current().SetVecTile({1, 2, 128});
        Tensor operand(DT_FP32, {2, 4, 140}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_006") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_007()
{
    PROGRAM("AMAX_FP32_007")
    {
        TileShape::Current().SetVecTile({1, 5, 32});
        Tensor operand(DT_FP32, {2, 5, 152}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_007") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_008()
{
    PROGRAM("AMAX_FP32_008")
    {
        TileShape::Current().SetVecTile({1, 3, 168});
        Tensor operand(DT_FP32, {2, 3, 170}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_008") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_009()
{
    PROGRAM("AMAX_FP32_009")
    {
        TileShape::Current().SetVecTile({2, 1, 2, 16});
        Tensor operand(DT_FP32, {5, 2, 4, 176}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_009") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_010()
{
    PROGRAM("AMAX_FP32_010")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 128});
        Tensor operand(DT_FP32, {5, 2, 4, 130}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_010") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_011()
{
    PROGRAM("AMAX_FP32_011")
    {
        TileShape::Current().SetVecTile({1, 1, 5, 32});
        Tensor operand(DT_FP32, {2, 3, 5, 134}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_011") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_012()
{
    PROGRAM("AMAX_FP32_012")
    {
        TileShape::Current().SetVecTile({2, 2, 3, 32});
        Tensor operand(DT_FP32, {4, 2, 6, 135}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_012") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_013()
{
    PROGRAM("AMAX_FP32_013")
    {
        TileShape::Current().SetVecTile({1, 1, 4, 128});
        Tensor operand(DT_FP32, {6, 2, 4, 130}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_013") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_014()
{
    PROGRAM("AMAX_FP32_014")
    {
        TileShape::Current().SetVecTile({1, 2, 1, 136});
        Tensor operand(DT_FP32, {3, 2, 3, 139}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_014") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAmax::test_amax_fp32_015()
{
    PROGRAM("AMAX_FP32_015")
    {
        TileShape::Current().SetVecTile({3, 3, 5, 32});
        Tensor operand(DT_FP32, {6, 3, 5, 141}, "operand");
        Tensor result;
        FUNCTION("AMAX_FP32_015") { result = Amax(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMAX_FP32_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
