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
 * \file test_codegen_mul.cpp
 * \brief
 */

#include "include/test_codegen_mul.h"

using namespace npu::tile_fwk;

TestCodeGenMul::TestCodeGenMul() = default;
TestCodeGenMul::~TestCodeGenMul() = default;

TestCodeGenMul& TestCodeGenMul::Instance()
{
    static TestCodeGenMul instance;
    return instance;
}

void TestCodeGenMul::test_mul_001()
{
    PROGRAM("MUL_001")
    {
        Tensor input0(DataType::DT_INT32, {160}, "input0");
        Tensor input1(DataType::DT_INT32, {160}, "input1");
        auto output = Tensor(DataType::DT_INT32, {160}, "output");
        FUNCTION("MUL_001")
        {
            TileShape::Current().SetVecTile({200});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_002()
{
    PROGRAM("MUL_002")
    {
        Tensor input0(DataType::DT_INT16, {100}, "input0");
        Tensor input1(DataType::DT_INT16, {100}, "input1");
        auto output = Tensor(DataType::DT_INT16, {100}, "output");
        FUNCTION("MUL_002")
        {
            TileShape::Current().SetVecTile({100});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_003()
{
    PROGRAM("MUL_003")
    {
        Tensor input0(DataType::DT_FP32, {112}, "input0");
        Tensor input1(DataType::DT_FP32, {112}, "input1");
        auto output = Tensor(DataType::DT_FP32, {112}, "output");
        FUNCTION("MUL_003")
        {
            TileShape::Current().SetVecTile({100});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_004()
{
    PROGRAM("MUL_004")
    {
        Tensor input0(DataType::DT_FP16, {101}, "input0");
        Tensor input1(DataType::DT_FP16, {101}, "input1");
        auto output = Tensor(DataType::DT_FP16, {101}, "output");
        FUNCTION("MUL_004")
        {
            TileShape::Current().SetVecTile({100});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_005()
{
    PROGRAM("MUL_005")
    {
        Tensor input0(DataType::DT_FP32, {112}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {112}, "output");
        FUNCTION("MUL_005")
        {
            TileShape::Current().SetVecTile({100});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_006()
{
    PROGRAM("MUL_006")
    {
        Tensor input0(DataType::DT_FP16, {101}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {101}, "output");
        FUNCTION("MUL_006")
        {
            TileShape::Current().SetVecTile({100});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_007()
{
    PROGRAM("MUL_007")
    {
        Tensor input0(DataType::DT_INT32, {160}, "input0");
        Tensor input1(DataType::DT_INT32, {1}, "input1");
        auto output = Tensor(DataType::DT_INT32, {160}, "output");
        FUNCTION("MUL_007")
        {
            TileShape::Current().SetVecTile({120});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_008()
{
    PROGRAM("MUL_008")
    {
        Tensor input0(DataType::DT_INT16, {100}, "input0");
        Tensor input1(DataType::DT_INT16, {1}, "input1");
        auto output = Tensor(DataType::DT_INT16, {100}, "output");
        FUNCTION("MUL_008")
        {
            TileShape::Current().SetVecTile({112});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_009()
{
    PROGRAM("MUL_009")
    {
        Tensor input0(DataType::DT_FP32, {32, 20}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {32, 20}, "output");
        FUNCTION("MUL_009")
        {
            TileShape::Current().SetVecTile({64, 32});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_010()
{
    PROGRAM("MUL_010")
    {
        Tensor input0(DataType::DT_FP16, {31, 19}, "input0");
        Tensor input1(DataType::DT_FP16, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_FP16, {31, 19}, "output");
        FUNCTION("MUL_010")
        {
            TileShape::Current().SetVecTile({32, 10});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_011()
{
    PROGRAM("MUL_011")
    {
        Tensor input0(DataType::DT_INT32, {31, 19}, "input0");
        Tensor input1(DataType::DT_INT32, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_INT32, {31, 19}, "output");
        FUNCTION("MUL_011")
        {
            TileShape::Current().SetVecTile({10, 30});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_012()
{
    PROGRAM("MUL_012")
    {
        Tensor input0(DataType::DT_INT16, {31, 19}, "input0");
        Tensor input1(DataType::DT_INT16, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_INT16, {31, 19}, "output");
        FUNCTION("MUL_012")
        {
            TileShape::Current().SetVecTile({10, 14});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_013()
{
    PROGRAM("MUL_013")
    {
        Tensor input0(DataType::DT_INT32, {32, 20}, "input0");
        Tensor input1(DataType::DT_INT32, {1, 20}, "input1");
        auto output = Tensor(DataType::DT_INT32, {32, 20}, "output");
        FUNCTION("MUL_013")
        {
            TileShape::Current().SetVecTile({64, 32});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_014()
{
    PROGRAM("MUL_014")
    {
        Tensor input0(DataType::DT_INT32, {1, 20}, "input0");
        Tensor input1(DataType::DT_INT32, {32, 20}, "input1");
        auto output = Tensor(DataType::DT_INT32, {32, 20}, "output");
        FUNCTION("MUL_014")
        {
            TileShape::Current().SetVecTile({64, 32});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_015()
{
    PROGRAM("MUL_015")
    {
        Tensor input0(DataType::DT_INT16, {31, 21}, "input0");
        Tensor input1(DataType::DT_INT16, {31, 1}, "input1");
        auto output = Tensor(DataType::DT_INT16, {31, 21}, "output");
        FUNCTION("MUL_015")
        {
            TileShape::Current().SetVecTile({32, 10});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_016()
{
    PROGRAM("MUL_016")
    {
        Tensor input0(DataType::DT_FP32, {31, 1}, "input0");
        Tensor input1(DataType::DT_FP32, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_FP32, {31, 19}, "output");
        FUNCTION("MUL_016")
        {
            TileShape::Current().SetVecTile({10, 20});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_017()
{
    PROGRAM("MUL_017")
    {
        Tensor input0(DataType::DT_FP16, {1, 19}, "input0");
        Tensor input1(DataType::DT_FP16, {31, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {31, 19}, "output");
        FUNCTION("MUL_017")
        {
            TileShape::Current().SetVecTile({10, 16});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_018()
{
    PROGRAM("MUL_018")
    {
        Tensor input0(DataType::DT_FP32, {10, 32, 23}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {10, 32, 23}, "output");
        FUNCTION("MUL_018")
        {
            TileShape::Current().SetVecTile({10, 32, 25});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_019()
{
    PROGRAM("MUL_019")
    {
        Tensor input0(DataType::DT_FP16, {10, 32, 19}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {10, 32, 19}, "output");
        FUNCTION("MUL_019")
        {
            TileShape::Current().SetVecTile({10, 32, 20});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_020()
{
    PROGRAM("MUL_020")
    {
        Tensor input0(DataType::DT_INT32, {21, 19, 23}, "input0");
        Tensor input1(DataType::DT_INT32, {21, 19, 23}, "input1");
        auto output = Tensor(DataType::DT_INT32, {21, 19, 23}, "output");
        FUNCTION("MUL_020")
        {
            TileShape::Current().SetVecTile({25, 20, 25});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_020");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_021()
{
    PROGRAM("MUL_021")
    {
        Tensor input0(DataType::DT_INT16, {10, 32, 23}, "input0");
        Tensor input1(DataType::DT_INT16, {10, 32, 23}, "input1");
        auto output = Tensor(DataType::DT_INT16, {10, 32, 23}, "output");
        FUNCTION("MUL_021")
        {
            TileShape::Current().SetVecTile({10, 32, 25});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_021");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_022()
{
    PROGRAM("MUL_022")
    {
        Tensor input0(DataType::DT_INT32, {1, 23, 27}, "input0");
        Tensor input1(DataType::DT_INT32, {13, 23, 27}, "input1");
        auto output = Tensor(DataType::DT_INT32, {13, 23, 27}, "output");
        FUNCTION("MUL_022")
        {
            TileShape::Current().SetVecTile({10, 25, 30});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_022");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_023()
{
    PROGRAM("MUL_023")
    {
        Tensor input0(DataType::DT_FP32, {13, 1, 27}, "input0");
        Tensor input1(DataType::DT_FP32, {13, 23, 27}, "input1");
        auto output = Tensor(DataType::DT_FP32, {13, 23, 27}, "output");
        FUNCTION("MUL_023")
        {
            TileShape::Current().SetVecTile({23, 10, 30});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_023");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_024()
{
    PROGRAM("MUL_024")
    {
        Tensor input0(DataType::DT_INT16, {13, 23, 1}, "input0");
        Tensor input1(DataType::DT_INT16, {13, 23, 27}, "input1");
        auto output = Tensor(DataType::DT_INT16, {13, 23, 27}, "output");
        FUNCTION("MUL_024")
        {
            TileShape::Current().SetVecTile({23, 25, 15});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_024");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_025()
{
    PROGRAM("MUL_025")
    {
        Tensor input0(DataType::DT_FP16, {13, 23, 27}, "input0");
        Tensor input1(DataType::DT_FP16, {13, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {13, 23, 27}, "output");
        FUNCTION("MUL_025")
        {
            TileShape::Current().SetVecTile({10, 10, 30});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_025");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_026()
{
    PROGRAM("MUL_026")
    {
        Tensor input0(DataType::DT_INT16, {13, 23, 27}, "input0");
        Tensor input1(DataType::DT_INT16, {1, 23, 1}, "input1");
        auto output = Tensor(DataType::DT_INT16, {13, 23, 27}, "output");
        FUNCTION("MUL_026")
        {
            TileShape::Current().SetVecTile({10, 25, 10});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_026");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_027()
{
    PROGRAM("MUL_027")
    {
        Tensor input0(DataType::DT_FP16, {13, 23, 27}, "input0");
        Tensor input1(DataType::DT_FP16, {1, 1, 27}, "input1");
        auto output = Tensor(DataType::DT_FP16, {13, 23, 27}, "output");
        FUNCTION("MUL_027")
        {
            TileShape::Current().SetVecTile({23, 10, 10});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_027");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_028()
{
    PROGRAM("MUL_028")
    {
        Tensor input0(DataType::DT_FP32, {63, 1, 1}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 43, 27}, "input1");
        auto output = Tensor(DataType::DT_FP32, {63, 43, 27}, "output");
        FUNCTION("MUL_028")
        {
            TileShape::Current().SetVecTile({23, 20, 17});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_028");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_029()
{
    PROGRAM("MUL_029")
    {
        Tensor input0(DataType::DT_FP32, {5, 16, 11, 12}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {5, 16, 11, 12}, "output");
        FUNCTION("MUL_029")
        {
            TileShape::Current().SetVecTile({5, 20, 15, 12});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_029");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_030()
{
    PROGRAM("MUL_030")
    {
        Tensor input0(DataType::DT_FP16, {5, 5, 6, 7}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {5, 5, 6, 7}, "output");
        FUNCTION("MUL_030")
        {
            TileShape::Current().SetVecTile({5, 5, 10, 10});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_030");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_031()
{
    PROGRAM("MUL_031")
    {
        Tensor input0(DataType::DT_INT32, {21, 12, 15, 16}, "input0");
        Tensor input1(DataType::DT_INT32, {21, 12, 15, 16}, "input1");
        auto output = Tensor(DataType::DT_INT32, {21, 12, 15, 16}, "output");
        FUNCTION("MUL_031")
        {
            TileShape::Current().SetVecTile({5, 12, 16, 16});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_031");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_032()
{
    PROGRAM("MUL_032")
    {
        Tensor input0(DataType::DT_INT16, {11, 19, 13, 11}, "input0");
        Tensor input1(DataType::DT_INT16, {11, 19, 13, 11}, "input1");
        auto output = Tensor(DataType::DT_INT16, {11, 19, 13, 11}, "output");
        FUNCTION("MUL_032")
        {
            TileShape::Current().SetVecTile({12, 5, 15, 12});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_032");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_033()
{
    PROGRAM("MUL_033")
    {
        Tensor input0(DataType::DT_INT32, {1, 11, 13, 17}, "input0");
        Tensor input1(DataType::DT_INT32, {21, 11, 13, 17}, "input1");
        auto output = Tensor(DataType::DT_INT32, {21, 11, 13, 17}, "output");
        FUNCTION("MUL_033")
        {
            TileShape::Current().SetVecTile({21, 12, 5, 20});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_033");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_034()
{
    PROGRAM("MUL_034")
    {
        Tensor input0(DataType::DT_INT16, {11, 1, 15, 17}, "input0");
        Tensor input1(DataType::DT_INT16, {11, 11, 15, 17}, "input1");
        auto output = Tensor(DataType::DT_INT16, {11, 11, 15, 17}, "output");
        FUNCTION("MUL_034")
        {
            TileShape::Current().SetVecTile({11, 12, 15, 2});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_034");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_035()
{
    PROGRAM("MUL_035")
    {
        Tensor input0(DataType::DT_FP32, {21, 11, 1, 17}, "input0");
        Tensor input1(DataType::DT_FP32, {21, 11, 13, 17}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 11, 13, 17}, "output");
        FUNCTION("MUL_035")
        {
            TileShape::Current().SetVecTile({15, 5, 15, 20});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_035");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_036()
{
    PROGRAM("MUL_036")
    {
        Tensor input0(DataType::DT_FP16, {25, 11, 15, 1}, "input0");
        Tensor input1(DataType::DT_FP16, {25, 11, 15, 17}, "input1");
        auto output = Tensor(DataType::DT_FP16, {25, 11, 15, 17}, "output");
        FUNCTION("MUL_036")
        {
            TileShape::Current().SetVecTile({13, 12, 3, 18});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_036");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_037()
{
    PROGRAM("MUL_037")
    {
        Tensor input0(DataType::DT_FP32, {21, 11, 13, 17}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 1, 13, 17}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 11, 13, 17}, "output");
        FUNCTION("MUL_037")
        {
            TileShape::Current().SetVecTile({10, 12, 15, 6});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_037");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_038()
{
    PROGRAM("MUL_038")
    {
        Tensor input0(DataType::DT_FP16, {25, 11, 15, 17}, "input0");
        Tensor input1(DataType::DT_FP16, {1, 11, 1, 17}, "input1");
        auto output = Tensor(DataType::DT_FP16, {25, 11, 15, 17}, "output");
        FUNCTION("MUL_038")
        {
            TileShape::Current().SetVecTile({25, 7, 5, 18});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_038");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_039()
{
    PROGRAM("MUL_039")
    {
        Tensor input0(DataType::DT_FP32, {21, 11, 13, 17}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 11, 13, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 11, 13, 17}, "output");
        FUNCTION("MUL_039")
        {
            TileShape::Current().SetVecTile({21, 3, 13, 6});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_039");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_040()
{
    PROGRAM("MUL_040")
    {
        Tensor input0(DataType::DT_FP16, {25, 1, 1, 17}, "input0");
        Tensor input1(DataType::DT_FP16, {25, 11, 15, 17}, "input1");
        auto output = Tensor(DataType::DT_FP16, {25, 11, 15, 17}, "output");
        FUNCTION("MUL_040")
        {
            TileShape::Current().SetVecTile({25, 11, 5, 3});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_040");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_041()
{
    PROGRAM("MUL_041")
    {
        Tensor input0(DataType::DT_FP32, {22, 1, 13, 1}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 13, 18}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("MUL_041")
        {
            TileShape::Current().SetVecTile({5, 7, 7, 18});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_041");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_042()
{
    PROGRAM("MUL_042")
    {
        Tensor input0(DataType::DT_FP16, {22, 16, 1, 18}, "input0");
        Tensor input1(DataType::DT_FP16, {22, 16, 13, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {22, 16, 13, 18}, "output");
        FUNCTION("MUL_042")
        {
            TileShape::Current().SetVecTile({5, 7, 15, 5});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_042");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_043()
{
    PROGRAM("MUL_043")
    {
        Tensor input0(DataType::DT_FP32, {1, 1, 1, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 13, 18}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("MUL_043")
        {
            TileShape::Current().SetVecTile({5, 16, 7, 5});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_043");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_044()
{
    PROGRAM("MUL_044")
    {
        Tensor input0(DataType::DT_FP16, {1, 1, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP16, {22, 16, 13, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {22, 16, 13, 18}, "output");
        FUNCTION("MUL_044")
        {
            TileShape::Current().SetVecTile({22, 7, 7, 5});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_044");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_045()
{
    PROGRAM("MUL_045")
    {
        Tensor input0(DataType::DT_FP32, {1, 16, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("MUL_045")
        {
            TileShape::Current().SetVecTile({5, 16, 7, 5});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_045");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_046()
{
    PROGRAM("MUL_046")
    {
        Tensor input0(DataType::DT_FP16, {22, 16, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP16, {22, 1, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {22, 16, 13, 18}, "output");
        FUNCTION("MUL_046")
        {
            TileShape::Current().SetVecTile({22, 7, 7, 5});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_046");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_047()
{
    PROGRAM("MUL_047")
    {
        Tensor input0(DataType::DT_FP32, {22, 16, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 1, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("MUL_047")
        {
            TileShape::Current().SetVecTile({5, 16, 7, 5});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_047");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMul::test_mul_048()
{
    PROGRAM("MUL_048")
    {
        Tensor input0(DataType::DT_FP32, {1, 1, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("MUL_048")
        {
            TileShape::Current().SetVecTile({11, 12, 12, 10});
            output = Mul(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MUL_048");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
