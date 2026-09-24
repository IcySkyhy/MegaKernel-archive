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
 * \file test_codegen_exp.cpp
 * \brief
 */

#include "include/test_codegen_exp.h"

using namespace npu::tile_fwk;

TestCodeGenExp::TestCodeGenExp() = default;
TestCodeGenExp::~TestCodeGenExp() = default;

TestCodeGenExp& TestCodeGenExp::Instance()
{
    static TestCodeGenExp instance;
    return instance;
}

void TestCodeGenExp::test_Exp_fp16_001()
{
    PROGRAM("EXP_fp16_001")
    {
        Tensor input(DataType::DT_FP16, {112}, "input");
        auto output = Tensor(DataType::DT_FP16, {112}, "output");
        FUNCTION("EXP_fp16_001")
        {
            TileShape::Current().SetVecTile({50});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_Exp_fp16_002()
{
    PROGRAM("EXP_fp16_002")
    {
        Tensor input(DataType::DT_FP16, {100}, "input");
        auto output = Tensor(DataType::DT_FP16, {100}, "output");
        FUNCTION("EXP_fp16_002")
        {
            TileShape::Current().SetVecTile({100});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_exp_fp32_003()
{
    PROGRAM("EXP_fp32_003")
    {
        Tensor input(DataType::DT_FP32, {4, 128}, "input");
        auto output = Tensor(DataType::DT_FP32, {4, 128}, "output");
        FUNCTION("EXP_fp32_003")
        {
            TileShape::Current().SetVecTile({2, 32});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_exp_fp32_004()
{
    PROGRAM("EXP_fp32_004")
    {
        Tensor input(DataType::DT_FP32, {4, 130}, "input");
        auto output = Tensor(DataType::DT_FP32, {4, 130}, "output");
        FUNCTION("EXP_fp32_004")
        {
            TileShape::Current().SetVecTile({1, 130});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_Exp_fp16_005()
{
    PROGRAM("EXP_fp16_005")
    {
        Tensor input(DataType::DT_FP16, {2, 4, 160}, "input");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 160}, "output");
        FUNCTION("EXP_fp16_005")
        {
            TileShape::Current().SetVecTile({1, 2, 32});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_exp_fp32_006()
{
    PROGRAM("EXP_fp32_006")
    {
        Tensor input(DataType::DT_FP32, {2, 4, 140}, "input");
        auto output = Tensor(DataType::DT_FP32, {2, 4, 140}, "output");
        FUNCTION("EXP_fp32_006")
        {
            TileShape::Current().SetVecTile({1, 2, 140});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_Exp_fp16_007()
{
    PROGRAM("EXP_fp16_007")
    {
        Tensor input(DataType::DT_FP16, {2, 5, 152}, "input");
        auto output = Tensor(DataType::DT_FP16, {2, 5, 152}, "output");
        FUNCTION("EXP_fp16_007")
        {
            TileShape::Current().SetVecTile({1, 5, 32});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp16_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_exp_fp32_008()
{
    PROGRAM("EXP_fp32_008")
    {
        Tensor input(DataType::DT_FP32, {2, 3, 170}, "input");
        auto output = Tensor(DataType::DT_FP32, {2, 3, 170}, "output");
        FUNCTION("EXP_fp32_008")
        {
            TileShape::Current().SetVecTile({1, 3, 170});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_Exp_fp16_009()
{
    PROGRAM("EXP_fp16_009")
    {
        Tensor input(DataType::DT_FP16, {5, 2, 4, 176}, "input");
        auto output = Tensor(DataType::DT_FP16, {5, 2, 4, 176}, "output");
        FUNCTION("EXP_fp16_009")
        {
            TileShape::Current().SetVecTile({2, 1, 2, 16});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp16_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_exp_fp32_010()
{
    PROGRAM("EXP_fp32_010")
    {
        Tensor input(DataType::DT_FP32, {5, 2, 4, 130}, "input");
        auto output = Tensor(DataType::DT_FP32, {5, 2, 4, 130}, "output");
        FUNCTION("EXP_fp32_010")
        {
            TileShape::Current().SetVecTile({1, 1, 1, 130});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_Exp_fp16_011()
{
    PROGRAM("EXP_fp16_011")
    {
        Tensor input(DataType::DT_FP16, {2, 3, 5, 134}, "input");
        auto output = Tensor(DataType::DT_FP16, {2, 3, 5, 134}, "output");
        FUNCTION("EXP_fp16_011")
        {
            TileShape::Current().SetVecTile({1, 1, 5, 32});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp16_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_exp_fp32_012()
{
    PROGRAM("EXP_fp32_012")
    {
        Tensor input(DataType::DT_FP32, {4, 2, 6, 135}, "input");
        auto output = Tensor(DataType::DT_FP32, {4, 2, 6, 135}, "output");
        FUNCTION("EXP_fp32_012")
        {
            TileShape::Current().SetVecTile({2, 2, 3, 32});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp32_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_Exp_fp16_013()
{
    PROGRAM("EXP_fp16_013")
    {
        Tensor input(DataType::DT_FP16, {6, 2, 4, 130}, "input");
        auto output = Tensor(DataType::DT_FP16, {6, 2, 4, 130}, "output");
        FUNCTION("EXP_fp16_013")
        {
            TileShape::Current().SetVecTile({1, 1, 4, 130});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp16_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_exp_fp32_014()
{
    PROGRAM("EXP_fp32_014")
    {
        Tensor input(DataType::DT_FP32, {3, 2, 3, 139}, "input");
        auto output = Tensor(DataType::DT_FP32, {3, 2, 3, 139}, "output");
        FUNCTION("EXP_fp32_014")
        {
            TileShape::Current().SetVecTile({1, 2, 1, 139});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp32_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenExp::test_Exp_fp16_015()
{
    PROGRAM("EXP_fp16_015")
    {
        Tensor input(DataType::DT_FP16, {6, 3, 5, 141}, "input");
        auto output = Tensor(DataType::DT_FP16, {6, 3, 5, 141}, "output");
        FUNCTION("EXP_fp16_015")
        {
            TileShape::Current().SetVecTile({3, 3, 5, 32});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_fp16_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
