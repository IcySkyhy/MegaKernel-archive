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
 * \file test_codegen_sum.cpp
 * \brief
 */

#include "include/test_codegen_sum.h"

using namespace npu::tile_fwk;

TestCodeGenSum::TestCodeGenSum() = default;
TestCodeGenSum::~TestCodeGenSum() = default;

TestCodeGenSum& TestCodeGenSum::Instance()
{
    static TestCodeGenSum instance;
    return instance;
}

void TestCodeGenSum::test_sum_fp32_003()
{
    PROGRAM("SUM_FP32_003")
    {
        TileShape::Current().SetVecTile({2, 32});
        Tensor operand(DT_FP32, {4, 128}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_003") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_004()
{
    PROGRAM("SUM_FP32_004")
    {
        TileShape::Current().SetVecTile({1, 128});
        Tensor operand(DT_FP32, {4, 130}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_004") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_005()
{
    PROGRAM("SUM_FP32_005")
    {
        TileShape::Current().SetVecTile({1, 2, 32});
        Tensor operand(DT_FP32, {2, 4, 160}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_005") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_006()
{
    PROGRAM("SUM_FP32_006")
    {
        TileShape::Current().SetVecTile({1, 2, 128});
        Tensor operand(DT_FP32, {2, 4, 140}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_006") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_007()
{
    PROGRAM("SUM_FP32_007")
    {
        TileShape::Current().SetVecTile({1, 5, 32});
        Tensor operand(DT_FP32, {2, 5, 152}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_007") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_008()
{
    PROGRAM("SUM_FP32_008")
    {
        TileShape::Current().SetVecTile({1, 3, 168});
        Tensor operand(DT_FP32, {2, 3, 170}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_008") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_009()
{
    PROGRAM("SUM_FP32_009")
    {
        TileShape::Current().SetVecTile({2, 1, 2, 16});
        Tensor operand(DT_FP32, {5, 2, 4, 176}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_009") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_010()
{
    PROGRAM("SUM_FP32_010")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 128});
        Tensor operand(DT_FP32, {5, 2, 4, 130}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_010") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_011()
{
    PROGRAM("SUM_FP32_011")
    {
        TileShape::Current().SetVecTile({1, 1, 5, 32});
        Tensor operand(DT_FP32, {2, 3, 5, 134}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_011") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_012()
{
    PROGRAM("SUM_FP32_012")
    {
        TileShape::Current().SetVecTile({2, 2, 3, 32});
        Tensor operand(DT_FP32, {4, 2, 6, 135}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_012") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_013()
{
    PROGRAM("SUM_FP32_013")
    {
        TileShape::Current().SetVecTile({1, 1, 4, 128});
        Tensor operand(DT_FP32, {6, 2, 4, 130}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_013") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_014()
{
    PROGRAM("SUM_FP32_014")
    {
        TileShape::Current().SetVecTile({1, 2, 1, 136});
        Tensor operand(DT_FP32, {3, 2, 3, 139}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_014") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_fp32_015()
{
    PROGRAM("SUM_FP32_015")
    {
        TileShape::Current().SetVecTile({3, 3, 5, 32});
        Tensor operand(DT_FP32, {6, 3, 5, 141}, "operand");
        Tensor result;
        FUNCTION("SUM_FP32_015") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_FP32_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_003()
{
    PROGRAM("SUM_INT32_003")
    {
        TileShape::Current().SetVecTile({2, 32});
        Tensor operand(DT_INT32, {4, 128}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_003") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_004()
{
    PROGRAM("SUM_INT32_004")
    {
        TileShape::Current().SetVecTile({1, 128});
        Tensor operand(DT_INT32, {4, 130}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_004") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_005()
{
    PROGRAM("SUM_INT32_005")
    {
        TileShape::Current().SetVecTile({1, 2, 32});
        Tensor operand(DT_INT32, {2, 4, 160}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_005") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_006()
{
    PROGRAM("SUM_INT32_006")
    {
        TileShape::Current().SetVecTile({1, 2, 128});
        Tensor operand(DT_INT32, {2, 4, 140}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_006") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_007()
{
    PROGRAM("SUM_INT32_007")
    {
        TileShape::Current().SetVecTile({1, 5, 32});
        Tensor operand(DT_INT32, {2, 5, 152}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_007") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_008()
{
    PROGRAM("SUM_INT32_008")
    {
        TileShape::Current().SetVecTile({1, 3, 168});
        Tensor operand(DT_INT32, {2, 3, 170}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_008") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_009()
{
    PROGRAM("SUM_INT32_009")
    {
        TileShape::Current().SetVecTile({2, 1, 2, 16});
        Tensor operand(DT_INT32, {5, 2, 4, 176}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_009") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_010()
{
    PROGRAM("SUM_INT32_010")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 128});
        Tensor operand(DT_INT32, {5, 2, 4, 130}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_010") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_011()
{
    PROGRAM("SUM_INT32_011")
    {
        TileShape::Current().SetVecTile({1, 1, 5, 32});
        Tensor operand(DT_INT32, {2, 3, 5, 134}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_011") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_012()
{
    PROGRAM("SUM_INT32_012")
    {
        TileShape::Current().SetVecTile({2, 2, 3, 32});
        Tensor operand(DT_INT32, {4, 2, 6, 135}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_012") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_013()
{
    PROGRAM("SUM_INT32_013")
    {
        TileShape::Current().SetVecTile({1, 1, 4, 128});
        Tensor operand(DT_INT32, {6, 2, 4, 130}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_013") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_014()
{
    PROGRAM("SUM_INT32_014")
    {
        TileShape::Current().SetVecTile({1, 2, 1, 136});
        Tensor operand(DT_INT32, {3, 2, 3, 139}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_014") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSum::test_sum_int32_015()
{
    PROGRAM("SUM_INT32_015")
    {
        TileShape::Current().SetVecTile({3, 3, 5, 32});
        Tensor operand(DT_INT32, {6, 3, 5, 141}, "operand");
        Tensor result;
        FUNCTION("SUM_INT32_015") { result = Sum(operand, -1, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUM_INT32_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
