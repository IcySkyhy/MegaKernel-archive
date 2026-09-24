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
 * \brief
 */

#include "include/test_codegen_where.h"

using namespace npu::tile_fwk;

TestCodeGenWhere::TestCodeGenWhere() = default;
TestCodeGenWhere::~TestCodeGenWhere() = default;

TestCodeGenWhere& TestCodeGenWhere::Instance()
{
    static TestCodeGenWhere instance;
    return instance;
}

void TestCodeGenWhere::test_where_fp16_001()
{
    PROGRAM("WHERE_FP16_001")
    {
        TileShape::Current().SetVecTile(1, 64);
        Tensor condition(DT_BOOL, {2, 1}, "condition");
        Tensor input(DT_FP16, {2, 64}, "input");
        Tensor other(DT_FP16, {2, 64}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 64}, "output");
        FUNCTION("WHERE_FP16_001") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_002()
{
    PROGRAM("WHERE_FP16_002")
    {
        TileShape::Current().SetVecTile(4, 16);
        Tensor condition(DT_BOOL, {4, 1}, "condition");
        Tensor input(DT_FP16, {4, 32}, "input");
        Tensor other(DT_FP16, {4, 32}, "other");
        auto output = Tensor(DataType::DT_FP16, {4, 32}, "output");
        FUNCTION("WHERE_FP16_002") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_003()
{
    PROGRAM("WHERE_FP16_003")
    {
        TileShape::Current().SetVecTile(2, 32);
        Tensor condition(DT_BOOL, {2, 1}, "condition");
        Tensor input(DT_FP16, {2, 64}, "input");
        Tensor other(DT_FP16, {1, 64}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 64}, "output");
        FUNCTION("WHERE_FP16_003") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_004()
{
    PROGRAM("WHERE_FP16_004")
    {
        TileShape::Current().SetVecTile(2, 32);
        Tensor condition(DT_BOOL, {4, 1}, "condition");
        Tensor input(DT_FP16, {4, 32}, "input");
        Element other(DT_FP16, 1.0f);
        auto output = Tensor(DataType::DT_FP16, {4, 32}, "output");
        FUNCTION("WHERE_FP16_004") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_005()
{
    PROGRAM("WHERE_FP16_005")
    {
        TileShape::Current().SetVecTile(1, 20);
        Tensor condition(DT_BOOL, {2, 1}, "condition");
        Tensor input(DT_FP16, {2, 40}, "input");
        Tensor other(DT_FP16, {2, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 40}, "output");
        FUNCTION("WHERE_FP16_005") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_006()
{
    PROGRAM("WHERE_FP16_006")
    {
        TileShape::Current().SetVecTile(2, 16, 32);
        Tensor condition(DT_BOOL, {2, 1, 32}, "condition");
        Tensor input(DT_FP16, {2, 1, 32}, "input");
        Tensor other(DT_FP16, {2, 32, 32}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 32}, "output");
        FUNCTION("WHERE_FP16_006") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_007()
{
    PROGRAM("WHERE_FP16_007")
    {
        TileShape::Current().SetVecTile(2, 32, 16);
        Tensor condition(DT_BOOL, {2, 32, 1}, "condition");
        Tensor input(DT_FP16, {1, 32, 32}, "input");
        Tensor other(DT_FP16, {2, 1, 32}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 32}, "output");
        FUNCTION("WHERE_FP16_007") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_008()
{
    PROGRAM("WHERE_FP16_008")
    {
        TileShape::Current().SetVecTile(1, 24, 24);
        Tensor condition(DT_BOOL, {2, 1, 24}, "condition");
        Tensor input(DT_FP16, {2, 24, 1}, "input");
        Tensor other(DT_FP16, {1, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 24, 24}, "output");
        FUNCTION("WHERE_FP16_008") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_009()
{
    PROGRAM("WHERE_FP16_009")
    {
        TileShape::Current().SetVecTile(3, 8, 48);
        Tensor condition(DT_BOOL, {3, 1, 48}, "condition");
        Tensor input(DT_FP16, {3, 16, 48}, "input");
        Element other(DT_FP16, 1.0f);
        auto output = Tensor(DataType::DT_FP16, {3, 16, 48}, "output");
        FUNCTION("WHERE_FP16_009") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_010()
{
    PROGRAM("WHERE_FP16_010")
    {
        TileShape::Current().SetVecTile(1, 16, 40);
        Tensor condition(DT_BOOL, {2, 1, 40}, "condition");
        Tensor input(DT_FP16, {2, 32, 1}, "input");
        Tensor other(DT_FP16, {1, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP16_010") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_011()
{
    PROGRAM("WHERE_FP16_011")
    {
        TileShape::Current().SetVecTile(1, 32, 32);
        Tensor condition(DT_BOOL, {2, 32, 1}, "condition");
        Tensor input(DT_FP16, {2, 32, 40}, "input");
        Tensor other(DT_FP16, {2, 1, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP16_011") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_012()
{
    PROGRAM("WHERE_FP16_012")
    {
        TileShape::Current().SetVecTile(1, 2, 20, 40);
        Tensor condition(DT_BOOL, {1, 2, 1, 40}, "condition");
        Tensor input(DT_FP16, {1, 2, 1, 40}, "input");
        Tensor other(DT_FP16, {1, 2, 40, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {1, 2, 40, 40}, "output");
        FUNCTION("WHERE_FP16_012") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_013()
{
    PROGRAM("WHERE_FP16_013")
    {
        TileShape::Current().SetVecTile(1, 2, 40, 16);
        Tensor condition(DT_BOOL, {1, 2, 40, 1}, "condition");
        Tensor input(DT_FP16, {1, 2, 40, 1}, "input");
        Tensor other(DT_FP16, {1, 1, 40, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {1, 2, 40, 40}, "output");
        FUNCTION("WHERE_FP16_013") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_014()
{
    PROGRAM("WHERE_FP16_014")
    {
        TileShape::Current().SetVecTile(1, 3, 24, 24);
        Tensor condition(DT_BOOL, {2, 3, 1, 24}, "condition");
        Tensor input(DT_FP16, {2, 3, 1, 24}, "input");
        Tensor other(DT_FP16, {2, 3, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 3, 24, 24}, "output");
        FUNCTION("WHERE_FP16_014") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_015()
{
    PROGRAM("WHERE_FP16_015")
    {
        TileShape::Current().SetVecTile(1, 2, 32, 32);
        Tensor condition(DT_BOOL, {1, 4, 32, 1}, "condition");
        Tensor input(DT_FP16, {1, 4, 1, 32}, "input");
        Tensor other(DT_FP16, {1, 1, 32, 32}, "other");
        auto output = Tensor(DataType::DT_FP16, {1, 4, 32, 32}, "output");
        FUNCTION("WHERE_FP16_015") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_016()
{
    PROGRAM("WHERE_FP16_016")
    {
        TileShape::Current().SetVecTile(2, 2, 16, 16);
        Tensor condition(DT_BOOL, {2, 2, 24, 1}, "condition");
        Tensor input(DT_FP16, {2, 2, 1, 40}, "input");
        Tensor other(DT_FP16, {2, 2, 24, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 2, 24, 40}, "output");
        FUNCTION("WHERE_FP16_016") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_017()
{
    PROGRAM("WHERE_FP16_017")
    {
        TileShape::Current().SetVecTile(1, 32, 24);
        Tensor condition(DT_BOOL, {2, 1, 40}, "condition");
        Tensor input(DT_FP16, {2, 1, 40}, "input");
        Tensor other(DT_FP16, {2, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP16_017") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_018()
{
    PROGRAM("WHERE_FP16_018")
    {
        TileShape::Current().SetVecTile(2, 16, 24);
        Tensor condition(DT_BOOL, {2, 32, 1}, "condition");
        Tensor input(DT_FP16, {2, 32, 40}, "input");
        Tensor other(DT_FP16, {2, 1, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP16_018") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_019()
{
    PROGRAM("WHERE_FP16_019")
    {
        TileShape::Current().SetVecTile(1, 4, 12, 8);
        Tensor condition(DT_BOOL, {2, 4, 24, 1}, "condition");
        Tensor input(DT_FP16, {2, 4, 1, 24}, "input");
        Tensor other(DT_FP16, {2, 4, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP16_019") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_020()
{
    PROGRAM("WHERE_FP16_020")
    {
        TileShape::Current().SetVecTile(2, 2, 12, 8);
        Tensor condition(DT_BOOL, {2, 4, 1, 24}, "condition");
        Tensor input(DT_FP16, {2, 1, 24, 24}, "input");
        Tensor other(DT_FP16, {2, 4, 24, 1}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP16_020") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_020");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_021()
{
    PROGRAM("WHERE_FP16_021")
    {
        TileShape::Current().SetVecTile(1, 2, 12, 24);
        Tensor condition(DT_BOOL, {2, 1, 24, 24}, "condition");
        Tensor input(DT_FP16, {2, 4, 1, 24}, "input");
        Tensor other(DT_FP16, {2, 4, 24, 1}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP16_021") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_021");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_022()
{
    PROGRAM("WHERE_FP16_022")
    {
        TileShape::Current().SetVecTile(1, 2, 12, 8);
        Tensor condition(DT_BOOL, {2, 4, 24, 24}, "condition");
        Tensor input(DT_FP16, {2, 4, 24, 1}, "input");
        Tensor other(DT_FP16, {2, 4, 1, 24}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP16_022") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_022");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_023()
{
    PROGRAM("WHERE_FP16_023")
    {
        TileShape::Current().SetVecTile(1, 4, 16, 16);
        Tensor condition(DT_BOOL, {2, 4, 32, 1}, "condition");
        Tensor input(DT_FP16, {2, 4, 1, 40}, "input");
        Element other(DT_FP16, 1.0f);
        auto output = Tensor(DataType::DT_FP16, {2, 4, 32, 40}, "output");
        FUNCTION("WHERE_FP16_023") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_023");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_024()
{
    PROGRAM("WHERE_FP16_024")
    {
        TileShape::Current().SetVecTile(2, 2, 20, 16);
        Tensor condition(DT_BOOL, {2, 4, 40, 1}, "condition");
        Tensor input(DT_FP16, {2, 4, 1, 40}, "input");
        Element other(DT_FP16, 1.0f);
        auto output = Tensor(DataType::DT_FP16, {2, 4, 40, 40}, "output");
        FUNCTION("WHERE_FP16_024") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_024");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_025()
{
    PROGRAM("WHERE_FP16_025")
    {
        TileShape::Current().SetVecTile(1, 1, 32, 16);
        Tensor condition(DT_BOOL, {2, 3, 1, 40}, "condition");
        Tensor input(DT_FP16, {2, 3, 1, 40}, "input");
        Tensor other(DT_FP16, {2, 3, 32, 1}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 3, 32, 40}, "output");
        FUNCTION("WHERE_FP16_025") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_025");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_026()
{
    PROGRAM("WHERE_FP16_026")
    {
        TileShape::Current().SetVecTile(1, 8);
        Tensor condition(DT_BOOL, {2, 8}, "condition");
        Element input(DT_FP16, 1.0f);
        Element other(DT_FP16, 2.0f);
        auto output = Tensor(DataType::DT_FP16, {2, 8}, "output");
        FUNCTION("WHERE_FP16_026") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_026");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_027()
{
    PROGRAM("WHERE_FP16_027")
    {
        TileShape::Current().SetVecTile(1, 4, 8);
        Tensor condition(DT_BOOL, {1, 8, 16}, "condition");
        Element input(DT_FP16, 1.0f);
        Element other(DT_FP16, 2.0f);
        auto output = Tensor(DataType::DT_FP16, {1, 8, 16}, "output");
        FUNCTION("WHERE_FP16_027") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_027");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_028()
{
    PROGRAM("WHERE_FP16_028")
    {
        TileShape::Current().SetVecTile(1, 1, 2, 4);
        Tensor condition(DT_BOOL, {2, 2, 4, 8}, "condition");
        Element input(DT_FP16, 1.0f);
        Element other(DT_FP16, 2.0f);
        auto output = Tensor(DataType::DT_FP16, {2, 2, 4, 8}, "output");
        FUNCTION("WHERE_FP16_028") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_028");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_029()
{
    PROGRAM("WHERE_FP16_029")
    {
        TileShape::Current().SetVecTile(2, 16, 16);
        Tensor condition(DT_BOOL, {2, 32, 1}, "condition");
        Tensor input(DT_FP16, {2, 1, 32}, "input");
        Tensor other(DT_FP16, {1, 32, 32}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 32}, "output");
        FUNCTION("WHERE_FP16_029") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_029");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_030()
{
    PROGRAM("WHERE_FP16_030")
    {
        TileShape::Current().SetVecTile(1, 3, 16);
        Tensor condition(DT_BOOL, {2, 3, 16}, "condition");
        Tensor input(DT_FP16, {2, 3, 16}, "input");
        Tensor other(DT_FP16, {2, 3, 1}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 3, 16}, "output");
        FUNCTION("WHERE_FP16_030") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_030");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_031()
{
    PROGRAM("WHERE_FP16_031")
    {
        TileShape::Current().SetVecTile(2, 16, 8);
        Tensor condition(DT_BOOL, {2, 32, 16}, "condition");
        Tensor input(DT_FP16, {2, 1, 16}, "input");
        Tensor other(DT_FP16, {1, 32, 16}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 16}, "output");
        FUNCTION("WHERE_FP16_031") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_031");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp16_032()
{
    PROGRAM("WHERE_FP16_032")
    {
        TileShape::Current().SetVecTile(1, 3, 16, 8);
        Tensor condition(DT_BOOL, {2, 3, 32, 16}, "condition");
        Tensor input(DT_FP16, {1, 3, 32, 16}, "input");
        Tensor other(DT_FP16, {2, 3, 1, 16}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 3, 32, 16}, "output");
        FUNCTION("WHERE_FP16_032") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_032");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_001()
{
    PROGRAM("WHERE_FP32_001")
    {
        TileShape::Current().SetVecTile(1, 64);
        Tensor condition(DT_BOOL, {1, 64}, "condition");
        Tensor input(DT_FP32, {2, 64}, "input");
        Tensor other(DT_FP32, {2, 64}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 64}, "output");
        FUNCTION("WHERE_FP32_001") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_002()
{
    PROGRAM("WHERE_FP32_002")
    {
        TileShape::Current().SetVecTile(4, 16);
        Tensor condition(DT_BOOL, {1, 32}, "condition");
        Tensor input(DT_FP32, {4, 32}, "input");
        Tensor other(DT_FP32, {4, 32}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 32}, "output");
        FUNCTION("WHERE_FP32_002") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_003()
{
    PROGRAM("WHERE_FP32_003")
    {
        TileShape::Current().SetVecTile(2, 32);
        Tensor condition(DT_BOOL, {2, 1}, "condition");
        Tensor input(DT_FP32, {2, 64}, "input");
        Tensor other(DT_FP32, {1, 64}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 64}, "output");
        FUNCTION("WHERE_FP32_003") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_004()
{
    PROGRAM("WHERE_FP32_004")
    {
        TileShape::Current().SetVecTile(2, 32);
        Tensor condition(DT_BOOL, {4, 1}, "condition");
        Tensor input(DT_FP32, {4, 32}, "input");
        Element other(DT_FP32, 1.0f);
        auto output = Tensor(DataType::DT_FP32, {4, 32}, "output");
        FUNCTION("WHERE_FP32_004") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_005()
{
    PROGRAM("WHERE_FP32_005")
    {
        TileShape::Current().SetVecTile(1, 16);
        Tensor condition(DT_BOOL, {2, 1}, "condition");
        Tensor input(DT_FP32, {2, 40}, "input");
        Tensor other(DT_FP32, {2, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 40}, "output");
        FUNCTION("WHERE_FP32_005") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_006()
{
    PROGRAM("WHERE_FP32_006")
    {
        TileShape::Current().SetVecTile(2, 16, 32);
        Tensor condition(DT_BOOL, {2, 1, 32}, "condition");
        Tensor input(DT_FP32, {2, 32, 32}, "input");
        Tensor other(DT_FP32, {2, 32, 1}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 32}, "output");
        FUNCTION("WHERE_FP32_006") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_007()
{
    PROGRAM("WHERE_FP32_007")
    {
        TileShape::Current().SetVecTile(2, 32, 16);
        Tensor condition(DT_BOOL, {2, 32, 1}, "condition");
        Tensor input(DT_FP32, {2, 1, 32}, "input");
        Tensor other(DT_FP32, {2, 32, 1}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 32}, "output");
        FUNCTION("WHERE_FP32_007") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_008()
{
    PROGRAM("WHERE_FP32_008")
    {
        TileShape::Current().SetVecTile(1, 24, 24);
        Tensor condition(DT_BOOL, {2, 24, 1}, "condition");
        Tensor input(DT_FP32, {1, 24, 24}, "input");
        Tensor other(DT_FP32, {2, 1, 24}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 24, 24}, "output");
        FUNCTION("WHERE_FP32_008") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_009()
{
    PROGRAM("WHERE_FP32_009")
    {
        TileShape::Current().SetVecTile(3, 8, 48);
        Tensor condition(DT_BOOL, {3, 1, 48}, "condition");
        Tensor input(DT_FP32, {1, 16, 48}, "input");
        Element other(DT_FP32, 1.0f);
        auto output = Tensor(DataType::DT_FP32, {3, 16, 48}, "output");
        FUNCTION("WHERE_FP32_009") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_010()
{
    PROGRAM("WHERE_FP32_010")
    {
        TileShape::Current().SetVecTile(1, 16, 40);
        Tensor condition(DT_BOOL, {2, 32, 1}, "condition");
        Tensor input(DT_FP32, {2, 32, 40}, "input");
        Tensor other(DT_FP32, {2, 1, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP32_010") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_011()
{
    PROGRAM("WHERE_FP32_011")
    {
        TileShape::Current().SetVecTile(1, 32, 16);
        Tensor condition(DT_BOOL, {2, 1, 40}, "condition");
        Tensor input(DT_FP32, {2, 32, 1}, "input");
        Tensor other(DT_FP32, {2, 1, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP32_011") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_012()
{
    PROGRAM("WHERE_FP32_012")
    {
        TileShape::Current().SetVecTile(1, 2, 20, 40);
        Tensor condition(DT_BOOL, {1, 2, 40, 1}, "condition");
        Tensor input(DT_FP32, {1, 2, 40, 1}, "input");
        Tensor other(DT_FP32, {1, 2, 1, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {1, 2, 40, 40}, "output");
        FUNCTION("WHERE_FP32_012") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_013()
{
    PROGRAM("WHERE_FP32_013")
    {
        TileShape::Current().SetVecTile(1, 2, 40, 16);
        Tensor condition(DT_BOOL, {1, 2, 1, 40}, "condition");
        Tensor input(DT_FP32, {1, 2, 40, 40}, "input");
        Tensor other(DT_FP32, {1, 2, 40, 1}, "other");
        auto output = Tensor(DataType::DT_FP32, {1, 2, 40, 40}, "output");
        FUNCTION("WHERE_FP32_013") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_014()
{
    PROGRAM("WHERE_FP32_014")
    {
        TileShape::Current().SetVecTile(1, 3, 24, 24);
        Tensor condition(DT_BOOL, {2, 3, 24, 1}, "condition");
        Tensor input(DT_FP32, {2, 3, 1, 24}, "input");
        Tensor other(DT_FP32, {2, 3, 24, 1}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 3, 24, 24}, "output");
        FUNCTION("WHERE_FP32_014") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_015()
{
    PROGRAM("WHERE_FP32_015")
    {
        TileShape::Current().SetVecTile(1, 2, 32, 32);
        Tensor condition(DT_BOOL, {1, 4, 32, 1}, "condition");
        Tensor input(DT_FP32, {1, 4, 1, 32}, "input");
        Tensor other(DT_FP32, {1, 1, 32, 32}, "other");
        auto output = Tensor(DataType::DT_FP32, {1, 4, 32, 32}, "output");
        FUNCTION("WHERE_FP32_015") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_016()
{
    PROGRAM("WHERE_FP32_016")
    {
        TileShape::Current().SetVecTile(1, 2, 12, 16);
        Tensor condition(DT_BOOL, {2, 2, 24, 1}, "condition");
        Tensor input(DT_FP32, {2, 2, 24, 1}, "input");
        Tensor other(DT_FP32, {2, 2, 1, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 2, 24, 40}, "output");
        FUNCTION("WHERE_FP32_016") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_017()
{
    PROGRAM("WHERE_FP32_017")
    {
        TileShape::Current().SetVecTile(1, 16, 40);
        Tensor condition(DT_BOOL, {2, 32, 1}, "condition");
        Tensor input(DT_FP32, {2, 32, 1}, "input");
        Tensor other(DT_FP32, {2, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP32_017") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_018()
{
    PROGRAM("WHERE_FP32_018")
    {
        TileShape::Current().SetVecTile(2, 16, 16);
        Tensor condition(DT_BOOL, {2, 1, 40}, "condition");
        Tensor input(DT_FP32, {2, 1, 40}, "input");
        Tensor other(DT_FP32, {2, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP32_018") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_019()
{
    PROGRAM("WHERE_FP32_019")
    {
        TileShape::Current().SetVecTile(1, 4, 12, 8);
        Tensor condition(DT_BOOL, {2, 4, 24, 1}, "condition");
        Tensor input(DT_FP32, {2, 4, 1, 24}, "input");
        Tensor other(DT_FP32, {2, 4, 24, 1}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP32_019") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_020()
{
    PROGRAM("WHERE_FP32_020")
    {
        TileShape::Current().SetVecTile(2, 2, 12, 8);
        Tensor condition(DT_BOOL, {2, 4, 1, 24}, "condition");
        Tensor input(DT_FP32, {2, 4, 24, 1}, "input");
        Tensor other(DT_FP32, {2, 1, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP32_020") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_020");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_021()
{
    PROGRAM("WHERE_FP32_021")
    {
        TileShape::Current().SetVecTile(1, 2, 12, 24);
        Tensor condition(DT_BOOL, {2, 4, 24, 24}, "condition");
        Tensor input(DT_FP32, {2, 1, 24, 24}, "input");
        Tensor other(DT_FP32, {1, 4, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP32_021") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_021");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_022()
{
    PROGRAM("WHERE_FP32_022")
    {
        TileShape::Current().SetVecTile(1, 2, 12, 16);
        Tensor condition(DT_BOOL, {2, 1, 24, 24}, "condition");
        Tensor input(DT_FP32, {2, 4, 1, 24}, "input");
        Tensor other(DT_FP32, {2, 4, 24, 1}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP32_022") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_022");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_023()
{
    PROGRAM("WHERE_FP32_023")
    {
        TileShape::Current().SetVecTile(1, 4, 16, 16);
        Tensor condition(DT_BOOL, {2, 4, 32, 1}, "condition");
        Tensor input(DT_FP32, {2, 4, 1, 40}, "input");
        Element other(DT_FP32, 1.0f);
        auto output = Tensor(DataType::DT_FP32, {2, 4, 32, 40}, "output");
        FUNCTION("WHERE_FP32_023") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_023");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_024()
{
    PROGRAM("WHERE_FP32_024")
    {
        TileShape::Current().SetVecTile(2, 2, 16, 16);
        Tensor condition(DT_BOOL, {2, 4, 40, 1}, "condition");
        Tensor input(DT_FP32, {2, 4, 40, 40}, "input");
        Element other(DT_FP32, 1.0f);
        auto output = Tensor(DataType::DT_FP32, {2, 4, 40, 40}, "output");
        FUNCTION("WHERE_FP32_024") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_024");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_025()
{
    PROGRAM("WHERE_FP32_025")
    {
        TileShape::Current().SetVecTile(1, 1, 32, 16);
        Tensor condition(DT_BOOL, {2, 3, 32, 1}, "condition");
        Tensor input(DT_FP32, {1, 3, 32, 40}, "input");
        Tensor other(DT_FP32, {2, 3, 1, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 3, 32, 40}, "output");
        FUNCTION("WHERE_FP32_025") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_025");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_026()
{
    PROGRAM("WHERE_FP32_026")
    {
        TileShape::Current().SetVecTile(1, 4);
        Tensor condition(DT_BOOL, {2, 8}, "condition");
        Element input(DT_FP32, 1.0f);
        Element other(DT_FP32, 2.0f);
        auto output = Tensor(DataType::DT_FP32, {2, 8}, "output");
        FUNCTION("WHERE_FP32_026") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_026");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_027()
{
    PROGRAM("WHERE_FP32_027")
    {
        TileShape::Current().SetVecTile(1, 4, 8);
        Tensor condition(DT_BOOL, {2, 8, 64}, "condition");
        Element input(DT_FP32, 1.0f);
        Element other(DT_FP32, 2.0f);
        auto output = Tensor(DataType::DT_FP32, {2, 8, 64}, "output");
        FUNCTION("WHERE_FP32_027") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_027");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_028()
{
    PROGRAM("WHERE_FP32_028")
    {
        TileShape::Current().SetVecTile(1, 2, 5, 16);
        Tensor condition(DT_BOOL, {2, 4, 16, 32}, "condition");
        Element input(DT_FP32, 1.0f);
        Element other(DT_FP32, 2.0f);
        auto output = Tensor(DataType::DT_FP32, {2, 4, 16, 32}, "output");
        FUNCTION("WHERE_FP32_028") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_028");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_029()
{
    PROGRAM("WHERE_FP32_029")
    {
        TileShape::Current().SetVecTile(4, 4);
        Tensor condition(DT_BOOL, {8, 8}, "condition");
        Tensor input(DT_FP32, {1, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        auto output = Tensor(DataType::DT_FP32, {8, 8}, "output");
        FUNCTION("WHERE_FP32_029") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_029");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_030()
{
    PROGRAM("WHERE_FP32_030")
    {
        TileShape::Current().SetVecTile(16, 12, 8);
        Tensor condition(DT_BOOL, {32, 24, 16}, "condition");
        Tensor input(DT_FP32, {1, 24, 16}, "input");
        Tensor other(DT_FP32, {32, 24, 16}, "other");
        auto output = Tensor(DataType::DT_FP32, {32, 24, 16}, "output");
        FUNCTION("WHERE_FP32_030") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_030");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_031()
{
    PROGRAM("WHERE_FP32_031")
    {
        TileShape::Current().SetVecTile(8, 8, 16, 8);
        Tensor condition(DT_BOOL, {16, 32, 32, 16}, "condition");
        Tensor input(DT_FP32, {1, 32, 32, 16}, "input");
        Tensor other(DT_FP32, {16, 32, 32, 16}, "other");
        auto output = Tensor(DataType::DT_FP32, {16, 32, 32, 16}, "output");
        FUNCTION("WHERE_FP32_031") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_031");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_032()
{
    PROGRAM("WHERE_FP32_032")
    {
        TileShape::Current().SetVecTile(1, 3, 32);
        Tensor condition(DT_BOOL, {2, 3, 32}, "condition");
        Tensor input(DT_FP32, {2, 3, 32}, "input");
        Tensor other(DT_FP32, {1, 3, 32}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 3, 32}, "output");
        FUNCTION("WHERE_FP32_032") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_032");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_033()
{
    PROGRAM("WHERE_FP32_033")
    {
        TileShape::Current().SetVecTile(2, 16, 16);
        Tensor condition(DT_BOOL, {2, 32, 16}, "condition");
        Tensor input(DT_FP32, {2, 32, 16}, "input");
        Tensor other(DT_FP32, {2, 1, 16}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 16}, "output");
        FUNCTION("WHERE_FP32_033") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_033");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenWhere::test_where_fp32_034()
{
    PROGRAM("WHERE_FP32_034")
    {
        TileShape::Current().SetVecTile(2, 3, 16, 16);
        Tensor condition(DT_BOOL, {2, 3, 16, 32}, "condition");
        Tensor input(DT_FP32, {2, 3, 16, 32}, "input");
        Tensor other(DT_FP32, {2, 3, 16, 1}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 3, 16, 32}, "output");
        FUNCTION("WHERE_FP32_034") { output = Where(condition, input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_034");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
