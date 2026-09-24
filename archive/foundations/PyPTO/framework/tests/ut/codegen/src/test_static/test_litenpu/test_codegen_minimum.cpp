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
 * \file test_codegen_minimum.cpp
 * \brief
 */

#include "include/test_codegen_minimum.h"

using namespace npu::tile_fwk;

TestCodeGenMinimum::TestCodeGenMinimum() = default;
TestCodeGenMinimum::~TestCodeGenMinimum() = default;

TestCodeGenMinimum& TestCodeGenMinimum::Instance()
{
    static TestCodeGenMinimum instance;
    return instance;
}

void TestCodeGenMinimum::test_minimum_int16_001()
{
    PROGRAM("MINIMUM_INT16_001")
    {
        TileShape::Current().SetVecTile({1, 1, 16, 16});
        Tensor input(DT_INT16, {2, 2, 32, 32}, "input");
        Tensor other(DT_INT16, {2, 2, 32, 32}, "other");
        auto output = Tensor(DataType::DT_INT16, {2, 2, 32, 32}, "output");
        FUNCTION("MINIMUM_INT16_001") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int16_002()
{
    PROGRAM("MINIMUM_INT16_002")
    {
        TileShape::Current().SetVecTile({1, 2, 8, 8});
        Tensor input(DT_INT16, {2, 4, 16, 16}, "input");
        Tensor other(DT_INT16, {2, 1, 16, 16}, "other");
        auto output = Tensor(DataType::DT_INT16, {2, 4, 16, 16}, "output");
        FUNCTION("MINIMUM_INT16_002") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int16_003()
{
    PROGRAM("MINIMUM_INT16_003")
    {
        TileShape::Current().SetVecTile({1, 1, 12, 12});
        Tensor input(DT_INT16, {2, 1, 24, 24}, "input");
        Tensor other(DT_INT16, {2, 3, 24, 24}, "other");
        auto output = Tensor(DataType::DT_INT16, {2, 3, 24, 24}, "output");
        FUNCTION("MINIMUM_INT16_003") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int16_004()
{
    PROGRAM("MINIMUM_INT16_004")
    {
        TileShape::Current().SetVecTile({1, 1, 20, 20});
        Tensor input(DT_INT16, {2, 2, 40, 40}, "input");
        Tensor other(DT_INT16, {2, 2, 1, 40}, "other");
        auto output = Tensor(DataType::DT_INT16, {2, 2, 40, 40}, "output");
        FUNCTION("MINIMUM_INT16_004") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int16_005()
{
    PROGRAM("MINIMUM_INT16_005")
    {
        TileShape::Current().SetVecTile({1, 1, 8, 8});
        Tensor input(DT_INT16, {2, 3, 16, 16}, "input");
        Tensor other(DT_INT16, {2, 3, 16, 16}, "other");
        auto output = Tensor(DataType::DT_INT16, {2, 3, 16, 16}, "output");
        FUNCTION("MINIMUM_INT16_005") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int16_006()
{
    PROGRAM("MINIMUM_INT16_006")
    {
        TileShape::Current().SetVecTile({4, 1});
        Tensor input(DT_INT16, {8, 4}, "input");
        Element other(DT_INT16, 1);
        auto output = Tensor(DataType::DT_INT16, {8, 4}, "output");
        FUNCTION("MINIMUM_INT16_006") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT16_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int16_007()
{
    PROGRAM("MINIMUM_INT16_007")
    {
        TileShape::Current().SetVecTile({2, 1, 32, 32});
        Tensor input(DT_INT16, {4, 1, 32, 32}, "input");
        Tensor other(DT_INT16, {4, 1, 32, 32}, "other");
        auto output = Tensor(DataType::DT_INT16, {4, 1, 32, 32}, "output");
        FUNCTION("MINIMUM_INT16_007") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT16_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int16_008()
{
    PROGRAM("MINIMUM_INT16_008")
    {
        TileShape::Current().SetVecTile({1, 4, 16, 16});
        Tensor input(DT_INT16, {1, 8, 16, 16}, "input");
        Tensor other(DT_INT16, {1, 8, 16, 16}, "other");
        auto output = Tensor(DataType::DT_INT16, {1, 8, 16, 16}, "output");
        FUNCTION("MINIMUM_INT16_008") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT16_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int16_009()
{
    PROGRAM("MINIMUM_INT16_009")
    {
        TileShape::Current().SetVecTile({2, 2, 24, 48});
        Tensor input(DT_INT16, {2, 2, 48, 48}, "input");
        Tensor other(DT_INT16, {2, 2, 48, 48}, "other");
        auto output = Tensor(DataType::DT_INT16, {2, 2, 48, 48}, "output");
        FUNCTION("MINIMUM_INT16_009") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT16_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int16_010()
{
    PROGRAM("MINIMUM_INT16_010")
    {
        TileShape::Current().SetVecTile({1, 4, 32, 32});
        Tensor input(DT_INT16, {1, 4, 32, 64}, "input");
        Tensor other(DT_INT16, {1, 4, 32, 64}, "other");
        auto output = Tensor(DataType::DT_INT16, {1, 4, 32, 64}, "output");
        FUNCTION("MINIMUM_INT16_010") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT16_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int32_001()
{
    PROGRAM("MINIMUM_INT32_001")
    {
        TileShape::Current().SetVecTile({1, 2, 16, 16});
        Tensor input(DT_INT32, {2, 4, 16, 1}, "input");
        Tensor other(DT_INT32, {2, 1, 16, 16}, "other");
        auto output = Tensor(DataType::DT_INT32, {2, 4, 16, 16}, "output");
        FUNCTION("MINIMUM_INT32_001") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int32_002()
{
    PROGRAM("MINIMUM_INT32_002")
    {
        TileShape::Current().SetVecTile({1, 2, 16, 32});
        Tensor input(DT_INT32, {2, 1, 32, 32}, "input");
        Tensor other(DT_INT32, {2, 2, 1, 32}, "other");
        auto output = Tensor(DataType::DT_INT32, {2, 2, 32, 32}, "output");
        FUNCTION("MINIMUM_INT32_002") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int32_003()
{
    PROGRAM("MINIMUM_INT32_003")
    {
        TileShape::Current().SetVecTile({1, 3, 24, 24});
        Tensor input(DT_INT32, {2, 3, 24, 1}, "input");
        Tensor other(DT_INT32, {1, 3, 24, 48}, "other");
        auto output = Tensor(DataType::DT_INT32, {2, 3, 24, 48}, "output");
        FUNCTION("MINIMUM_INT32_003") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int32_004()
{
    PROGRAM("MINIMUM_INT32_004")
    {
        TileShape::Current().SetVecTile({1, 2, 8, 16});
        Tensor input(DT_INT32, {1, 4, 1, 16}, "input");
        Tensor other(DT_INT32, {1, 1, 16, 16}, "other");
        auto output = Tensor(DataType::DT_INT32, {1, 4, 16, 16}, "output");
        FUNCTION("MINIMUM_INT32_004") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int32_005()
{
    PROGRAM("MINIMUM_INT32_005")
    {
        TileShape::Current().SetVecTile({2, 1, 32, 32});
        Tensor input(DT_INT32, {2, 2, 32, 1}, "input");
        Tensor other(DT_INT32, {2, 2, 1, 64}, "other");
        auto output = Tensor(DataType::DT_INT32, {2, 2, 32, 64}, "output");
        FUNCTION("MINIMUM_INT32_005") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int32_006()
{
    PROGRAM("MINIMUM_INT32_006")
    {
        TileShape::Current().SetVecTile({1, 1, 24, 32});
        Tensor input(DT_INT32, {1, 1, 48, 64}, "input");
        Tensor other(DT_INT32, {1, 1, 48, 64}, "other");
        auto output = Tensor(DataType::DT_INT32, {1, 1, 48, 64}, "output");
        FUNCTION("MINIMUM_INT32_006") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int32_007()
{
    PROGRAM("MINIMUM_INT32_007")
    {
        TileShape::Current().SetVecTile({1, 2, 16, 48});
        Tensor input(DT_INT32, {2, 4, 32, 48}, "input");
        Tensor other(DT_INT32, {2, 4, 32, 48}, "other");
        auto output = Tensor(DataType::DT_INT32, {2, 4, 32, 48}, "output");
        FUNCTION("MINIMUM_INT32_007") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT32_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int32_008()
{
    PROGRAM("MINIMUM_INT32_008")
    {
        TileShape::Current().SetVecTile({1, 2, 32, 32});
        Tensor input(DT_INT32, {2, 4, 32, 64}, "input");
        Tensor other(DT_INT32, {2, 4, 32, 64}, "other");
        auto output = Tensor(DataType::DT_INT32, {2, 4, 32, 64}, "output");
        FUNCTION("MINIMUM_INT32_008") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int32_009()
{
    PROGRAM("MINIMUM_INT32_009")
    {
        TileShape::Current().SetVecTile({2, 2, 16, 32});
        Tensor input(DT_INT32, {4, 2, 32, 64}, "input");
        Tensor other(DT_INT32, {4, 2, 32, 64}, "other");
        auto output = Tensor(DataType::DT_INT32, {4, 2, 32, 64}, "output");
        FUNCTION("MINIMUM_INT32_009") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT32_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_int32_010()
{
    PROGRAM("MINIMUM_INT32_010")
    {
        TileShape::Current().SetVecTile({1, 2, 16, 32});
        Tensor input(DT_INT32, {1, 4, 32, 64}, "input");
        Tensor other(DT_INT32, {1, 4, 32, 64}, "other");
        auto output = Tensor(DataType::DT_INT32, {1, 4, 32, 64}, "output");
        FUNCTION("MINIMUM_INT32_010") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_INT32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_001()
{
    PROGRAM("MINIMUM_FP16_001")
    {
        TileShape::Current().SetVecTile({50});
        Tensor input(DT_FP16, {112}, "input");
        Tensor other(DT_FP16, {112}, "other");
        auto output = Tensor(DataType::DT_FP16, {112}, "output");
        FUNCTION("MINIMUM_FP16_001") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_002()
{
    PROGRAM("MINIMUM_FP16_002")
    {
        TileShape::Current().SetVecTile({32});
        Tensor input(DT_FP16, {64}, "input");
        Element other(DT_FP16, 1.0f);
        auto output = Tensor(DataType::DT_FP16, {64}, "output");
        FUNCTION("MINIMUM_FP16_002") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_003()
{
    PROGRAM("MINIMUM_FP16_003")
    {
        TileShape::Current().SetVecTile({16});
        Tensor input(DT_FP16, {32}, "input");
        Tensor other(DT_FP16, {32}, "other");
        auto output = Tensor(DataType::DT_FP16, {32}, "output");
        FUNCTION("MINIMUM_FP16_003") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_004()
{
    PROGRAM("MINIMUM_FP16_004")
    {
        TileShape::Current().SetVecTile({16, 8});
        Tensor input(DT_FP16, {16, 16}, "input");
        Element other(DT_FP16, 1.0f);
        auto output = Tensor(DataType::DT_FP16, {16, 16}, "output");
        FUNCTION("MINIMUM_FP16_004") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_005()
{
    PROGRAM("MINIMUM_FP16_005")
    {
        TileShape::Current().SetVecTile({2, 40});
        Tensor input(DT_FP16, {4, 80}, "input");
        Tensor other(DT_FP16, {4, 80}, "other");
        auto output = Tensor(DataType::DT_FP16, {4, 80}, "output");
        FUNCTION("MINIMUM_FP16_005") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_006()
{
    PROGRAM("MINIMUM_FP16_006")
    {
        TileShape::Current().SetVecTile({1, 48});
        Tensor input(DT_FP16, {2, 96}, "input");
        Tensor other(DT_FP16, {1, 96}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 96}, "output");
        FUNCTION("MINIMUM_FP16_006") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_007()
{
    PROGRAM("MINIMUM_FP16_007")
    {
        TileShape::Current().SetVecTile({2, 16});
        Tensor input(DT_FP16, {5, 1}, "input");
        Tensor other(DT_FP16, {5, 32}, "other");
        auto output = Tensor(DataType::DT_FP16, {5, 32}, "output");
        FUNCTION("MINIMUM_FP16_007") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_008()
{
    PROGRAM("MINIMUM_FP16_008")
    {
        TileShape::Current().SetVecTile({2, 64});
        Tensor input(DT_FP16, {3, 128}, "input");
        Tensor other(DT_FP16, {3, 128}, "other");
        auto output = Tensor(DataType::DT_FP16, {3, 128}, "output");
        FUNCTION("MINIMUM_FP16_008") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_009()
{
    PROGRAM("MINIMUM_FP16_009")
    {
        TileShape::Current().SetVecTile({32, 64});
        Tensor input(DT_FP16, {64, 1}, "input");
        Tensor other(DT_FP16, {64, 64}, "other");
        auto output = Tensor(DataType::DT_FP16, {64, 64}, "output");
        FUNCTION("MINIMUM_FP16_009") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_010()
{
    PROGRAM("MINIMUM_FP16_010")
    {
        TileShape::Current().SetVecTile({64, 32});
        Tensor input(DT_FP16, {1, 64}, "input");
        Tensor other(DT_FP16, {64, 64}, "other");
        auto output = Tensor(DataType::DT_FP16, {64, 64}, "output");
        FUNCTION("MINIMUM_FP16_010") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_011()
{
    PROGRAM("MINIMUM_FP16_011")
    {
        TileShape::Current().SetVecTile({1, 32, 32});
        Tensor input(DT_FP16, {2, 64, 64}, "input");
        Tensor other(DT_FP16, {2, 64, 64}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 64, 64}, "output");
        FUNCTION("MINIMUM_FP16_011") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_012()
{
    PROGRAM("MINIMUM_FP16_012")
    {
        TileShape::Current().SetVecTile({1, 1, 24});
        Tensor input(DT_FP16, {2, 1, 48}, "input");
        Tensor other(DT_FP16, {2, 3, 48}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 3, 48}, "output");
        FUNCTION("MINIMUM_FP16_012") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_013()
{
    PROGRAM("MINIMUM_FP16_013")
    {
        TileShape::Current().SetVecTile({1, 32, 24});
        Tensor input(DT_FP16, {3, 64, 1}, "input");
        Tensor other(DT_FP16, {3, 64, 48}, "other");
        auto output = Tensor(DataType::DT_FP16, {3, 64, 48}, "output");
        FUNCTION("MINIMUM_FP16_013") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_014()
{
    PROGRAM("MINIMUM_FP16_014")
    {
        TileShape::Current().SetVecTile({1, 48, 48});
        Tensor input(DT_FP16, {2, 48, 48}, "input");
        Tensor other(DT_FP16, {2, 1, 48}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 48, 48}, "output");
        FUNCTION("MINIMUM_FP16_014") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_015()
{
    PROGRAM("MINIMUM_FP16_015")
    {
        TileShape::Current().SetVecTile({2, 32, 48});
        Tensor input(DT_FP16, {2, 64, 48}, "input");
        Tensor other(DT_FP16, {2, 64, 48}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 64, 48}, "output");
        FUNCTION("MINIMUM_FP16_015") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_016()
{
    PROGRAM("MINIMUM_FP16_016")
    {
        TileShape::Current().SetVecTile({3, 32, 32});
        Tensor input(DT_FP16, {3, 32, 64}, "input");
        Tensor other(DT_FP16, {3, 32, 64}, "other");
        auto output = Tensor(DataType::DT_FP16, {3, 32, 64}, "output");
        FUNCTION("MINIMUM_FP16_016") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_017()
{
    PROGRAM("MINIMUM_FP16_017")
    {
        TileShape::Current().SetVecTile({1, 16, 64});
        Tensor input(DT_FP16, {2, 32, 1}, "input");
        Tensor other(DT_FP16, {2, 32, 64}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 64}, "output");
        FUNCTION("MINIMUM_FP16_017") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp16_018()
{
    PROGRAM("MINIMUM_FP16_018")
    {
        TileShape::Current().SetVecTile({48, 24, 32});
        Tensor input(DT_FP16, {1, 48, 64}, "input");
        Tensor other(DT_FP16, {48, 48, 64}, "other");
        auto output = Tensor(DataType::DT_FP16, {48, 48, 64}, "output");
        FUNCTION("MINIMUM_FP16_018") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP16_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_001()
{
    PROGRAM("MINIMUM_FP32_001")
    {
        TileShape::Current().SetVecTile({16, 16, 16, 8});
        Tensor input(DT_FP32, {16, 16, 16, 16}, "input");
        Element other(DT_FP32, 1.0f);
        auto output = Tensor(DataType::DT_FP32, {16, 16, 16, 16}, "output");
        FUNCTION("MINIMUM_FP32_001") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_002()
{
    PROGRAM("MINIMUM_FP32_002")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        auto output = Tensor(DataType::DT_FP32, {8, 8}, "output");
        FUNCTION("MINIMUM_FP32_002") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_003()
{
    PROGRAM("MINIMUM_FP32_003")
    {
        TileShape::Current().SetVecTile({8, 4, 4});
        Tensor input(DT_FP32, {8}, "input");
        Tensor other(DT_FP32, {16, 8, 8}, "other");
        auto output = Tensor(DataType::DT_FP32, {16, 8, 8}, "output");
        FUNCTION("MINIMUM_FP32_003") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_004()
{
    PROGRAM("MINIMUM_FP32_004")
    {
        TileShape::Current().SetVecTile({2, 8, 4, 4});
        Tensor input(DT_FP32, {8}, "input");
        Tensor other(DT_FP32, {4, 16, 8, 8}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 16, 8, 8}, "output");
        FUNCTION("MINIMUM_FP32_004") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_005()
{
    PROGRAM("MINIMUM_FP32_005")
    {
        TileShape::Current().SetVecTile({16, 12, 8});
        Tensor input(DT_FP32, {24, 16}, "input");
        Tensor other(DT_FP32, {32, 24, 16}, "other");
        auto output = Tensor(DataType::DT_FP32, {32, 24, 16}, "output");
        FUNCTION("MINIMUM_FP32_005") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_006()
{
    PROGRAM("MINIMUM_FP32_006")
    {
        TileShape::Current().SetVecTile({2, 8, 12, 8});
        Tensor input(DT_FP32, {24, 16}, "input");
        Tensor other(DT_FP32, {4, 32, 24, 16}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 32, 24, 16}, "output");
        FUNCTION("MINIMUM_FP32_006") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_007()
{
    PROGRAM("MINIMUM_FP32_007")
    {
        TileShape::Current().SetVecTile({8, 16, 16, 8});
        Tensor input(DT_FP32, {32, 32, 16}, "input");
        Tensor other(DT_FP32, {16, 32, 32, 16}, "other");
        auto output = Tensor(DataType::DT_FP32, {16, 32, 32, 16}, "output");
        FUNCTION("MINIMUM_FP32_007") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_008()
{
    PROGRAM("MINIMUM_FP32_008")
    {
        TileShape::Current().SetVecTile({2, 2, 8});
        Tensor input(DT_FP32, {1, 1, 16}, "input");
        Tensor other(DT_FP32, {4, 4, 16}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 4, 16}, "output");
        FUNCTION("MINIMUM_FP32_008") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_009()
{
    PROGRAM("MINIMUM_FP32_009")
    {
        TileShape::Current().SetVecTile({2, 2, 2});
        Tensor input(DT_FP32, {1, 1, 1}, "input");
        Tensor other(DT_FP32, {4, 4, 4}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 4, 4}, "output");
        FUNCTION("MINIMUM_FP32_009") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_010()
{
    PROGRAM("MINIMUM_FP32_010")
    {
        TileShape::Current().SetVecTile({8, 2, 2});
        Tensor input(DT_FP32, {16, 1, 1}, "input");
        Tensor other(DT_FP32, {16, 4, 4}, "other");
        auto output = Tensor(DataType::DT_FP32, {16, 4, 4}, "output");
        FUNCTION("MINIMUM_FP32_010") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_011()
{
    PROGRAM("MINIMUM_FP32_011")
    {
        TileShape::Current().SetVecTile({2, 2, 2, 2});
        Tensor input(DT_FP32, {1, 1, 1, 1}, "input");
        Tensor other(DT_FP32, {4, 4, 4, 4}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 4, 4, 4}, "output");
        FUNCTION("MINIMUM_FP32_011") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_012()
{
    PROGRAM("MINIMUM_FP32_012")
    {
        TileShape::Current().SetVecTile({2, 2, 2, 2});
        Tensor input(DT_FP32, {1, 1, 4, 4}, "input");
        Tensor other(DT_FP32, {4, 4, 4, 4}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 4, 4, 4}, "output");
        FUNCTION("MINIMUM_FP32_012") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_013()
{
    PROGRAM("MINIMUM_FP32_013")
    {
        TileShape::Current().SetVecTile({2, 2, 2, 2});
        Tensor input(DT_FP32, {1, 4, 1, 4}, "input");
        Tensor other(DT_FP32, {4, 4, 4, 4}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 4, 4, 4}, "output");
        FUNCTION("MINIMUM_FP32_013") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_014()
{
    PROGRAM("MINIMUM_FP32_014")
    {
        TileShape::Current().SetVecTile({2, 2, 2, 2});
        Tensor input(DT_FP32, {1, 4, 4, 1}, "input");
        Tensor other(DT_FP32, {4, 4, 4, 4}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 4, 4, 4}, "output");
        FUNCTION("MINIMUM_FP32_014") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_015()
{
    PROGRAM("MINIMUM_FP32_015")
    {
        TileShape::Current().SetVecTile({2, 2, 2, 2});
        Tensor input(DT_FP32, {4, 1, 1, 4}, "input");
        Tensor other(DT_FP32, {4, 4, 4, 4}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 4, 4, 4}, "output");
        FUNCTION("MINIMUM_FP32_015") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_016()
{
    PROGRAM("MINIMUM_FP32_016")
    {
        TileShape::Current().SetVecTile({2, 2, 2, 2});
        Tensor input(DT_FP32, {4, 1, 4, 1}, "input");
        Tensor other(DT_FP32, {4, 4, 4, 4}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 4, 4, 4}, "output");
        FUNCTION("MINIMUM_FP32_016") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMinimum::test_minimum_fp32_017()
{
    PROGRAM("MINIMUM_FP32_017")
    {
        TileShape::Current().SetVecTile({2, 2, 2, 2});
        Tensor input(DT_FP32, {4, 4, 1, 1}, "input");
        Tensor other(DT_FP32, {4, 4, 4, 4}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 4, 4, 4}, "output");
        FUNCTION("MINIMUM_FP32_017") { output = Minimum(input, other); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MINIMUM_FP32_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
