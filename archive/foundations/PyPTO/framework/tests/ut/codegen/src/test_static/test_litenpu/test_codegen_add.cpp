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
 * \file test_codegen_add.cpp
 * \brief
 */

#include "include/test_codegen_add.h"

using namespace npu::tile_fwk;

TestCodeGenAdd::TestCodeGenAdd() = default;
TestCodeGenAdd::~TestCodeGenAdd() = default;

TestCodeGenAdd& TestCodeGenAdd::Instance()
{
    static TestCodeGenAdd instance;
    return instance;
}

void TestCodeGenAdd::test_add_001()
{
    PROGRAM("ADD_001")
    {
        Tensor input0(DataType::DT_INT32, {160}, "input0");
        Tensor input1(DataType::DT_INT32, {160}, "input1");
        auto output = Tensor(DataType::DT_INT32, {160}, "output");
        FUNCTION("ADD_001")
        {
            TileShape::Current().SetVecTile({200});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_002()
{
    PROGRAM("ADD_002")
    {
        Tensor input0(DataType::DT_INT16, {100}, "input0");
        Tensor input1(DataType::DT_INT16, {100}, "input1");
        auto output = Tensor(DataType::DT_INT16, {100}, "output");
        FUNCTION("ADD_002")
        {
            TileShape::Current().SetVecTile({100});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_003()
{
    PROGRAM("ADD_003")
    {
        Tensor input0(DataType::DT_FP32, {112}, "input0");
        Tensor input1(DataType::DT_FP32, {112}, "input1");
        auto output = Tensor(DataType::DT_FP32, {112}, "output");
        FUNCTION("ADD_003")
        {
            TileShape::Current().SetVecTile({100});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_004()
{
    PROGRAM("ADD_004")
    {
        Tensor input0(DataType::DT_FP16, {101}, "input0");
        Tensor input1(DataType::DT_FP16, {101}, "input1");
        auto output = Tensor(DataType::DT_FP16, {101}, "output");
        FUNCTION("ADD_004")
        {
            TileShape::Current().SetVecTile({100});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_005()
{
    PROGRAM("ADD_005")
    {
        Tensor input0(DataType::DT_FP32, {112}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {112}, "output");
        FUNCTION("ADD_005")
        {
            TileShape::Current().SetVecTile({100});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_006()
{
    PROGRAM("ADD_006")
    {
        Tensor input0(DataType::DT_FP16, {101}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {101}, "output");
        FUNCTION("ADD_006")
        {
            TileShape::Current().SetVecTile({100});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_007()
{
    PROGRAM("ADD_007")
    {
        Tensor input0(DataType::DT_INT32, {160}, "input0");
        Tensor input1(DataType::DT_INT32, {1}, "input1");
        auto output = Tensor(DataType::DT_INT32, {160}, "output");
        FUNCTION("ADD_007")
        {
            TileShape::Current().SetVecTile({120});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_008()
{
    PROGRAM("ADD_008")
    {
        Tensor input0(DataType::DT_INT16, {100}, "input0");
        Tensor input1(DataType::DT_INT16, {1}, "input1");
        auto output = Tensor(DataType::DT_INT16, {100}, "output");
        FUNCTION("ADD_008")
        {
            TileShape::Current().SetVecTile({112});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_009()
{
    PROGRAM("ADD_009")
    {
        Tensor input0(DataType::DT_FP32, {32, 20}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {32, 20}, "output");
        FUNCTION("ADD_009")
        {
            TileShape::Current().SetVecTile({64, 32});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_010()
{
    PROGRAM("ADD_010")
    {
        Tensor input0(DataType::DT_FP16, {31, 19}, "input0");
        Tensor input1(DataType::DT_FP16, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_FP16, {31, 19}, "output");
        FUNCTION("ADD_010")
        {
            TileShape::Current().SetVecTile({32, 10});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_011()
{
    PROGRAM("ADD_011")
    {
        Tensor input0(DataType::DT_INT32, {31, 19}, "input0");
        Tensor input1(DataType::DT_INT32, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_INT32, {31, 19}, "output");
        FUNCTION("ADD_011")
        {
            TileShape::Current().SetVecTile({10, 30});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_012()
{
    PROGRAM("ADD_012")
    {
        Tensor input0(DataType::DT_INT16, {31, 19}, "input0");
        Tensor input1(DataType::DT_INT16, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_INT16, {31, 19}, "output");
        FUNCTION("ADD_012")
        {
            TileShape::Current().SetVecTile({10, 14});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_013()
{
    PROGRAM("ADD_013")
    {
        Tensor input0(DataType::DT_INT32, {32, 20}, "input0");
        Tensor input1(DataType::DT_INT32, {1, 20}, "input1");
        auto output = Tensor(DataType::DT_INT32, {32, 20}, "output");
        FUNCTION("ADD_013")
        {
            TileShape::Current().SetVecTile({64, 32});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_014()
{
    PROGRAM("ADD_014")
    {
        Tensor input0(DataType::DT_INT32, {1, 20}, "input0");
        Tensor input1(DataType::DT_INT32, {32, 20}, "input1");
        auto output = Tensor(DataType::DT_INT32, {32, 20}, "output");
        FUNCTION("ADD_014")
        {
            TileShape::Current().SetVecTile({64, 32});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_015()
{
    PROGRAM("ADD_015")
    {
        Tensor input0(DataType::DT_INT16, {31, 21}, "input0");
        Tensor input1(DataType::DT_INT16, {31, 1}, "input1");
        auto output = Tensor(DataType::DT_INT16, {31, 21}, "output");
        FUNCTION("ADD_015")
        {
            TileShape::Current().SetVecTile({32, 10});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_016()
{
    PROGRAM("ADD_016")
    {
        Tensor input0(DataType::DT_FP32, {31, 1}, "input0");
        Tensor input1(DataType::DT_FP32, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_FP32, {31, 19}, "output");
        FUNCTION("ADD_016")
        {
            TileShape::Current().SetVecTile({10, 20});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_017()
{
    PROGRAM("ADD_017")
    {
        Tensor input0(DataType::DT_FP16, {1, 19}, "input0");
        Tensor input1(DataType::DT_FP16, {31, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {31, 19}, "output");
        FUNCTION("ADD_017")
        {
            TileShape::Current().SetVecTile({10, 16});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_018()
{
    PROGRAM("ADD_018")
    {
        Tensor input0(DataType::DT_FP32, {10, 32, 23}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {10, 32, 23}, "output");
        FUNCTION("ADD_018")
        {
            TileShape::Current().SetVecTile({10, 32, 25});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_019()
{
    PROGRAM("ADD_019")
    {
        Tensor input0(DataType::DT_FP16, {10, 32, 19}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {10, 32, 19}, "output");
        FUNCTION("ADD_019")
        {
            TileShape::Current().SetVecTile({10, 32, 20});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_020()
{
    PROGRAM("ADD_020")
    {
        Tensor input0(DataType::DT_INT32, {21, 19, 23}, "input0");
        Tensor input1(DataType::DT_INT32, {21, 19, 23}, "input1");
        auto output = Tensor(DataType::DT_INT32, {21, 19, 23}, "output");
        FUNCTION("ADD_020")
        {
            TileShape::Current().SetVecTile({25, 20, 25});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_020");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_021()
{
    PROGRAM("ADD_021")
    {
        Tensor input0(DataType::DT_INT16, {10, 32, 23}, "input0");
        Tensor input1(DataType::DT_INT16, {10, 32, 23}, "input1");
        auto output = Tensor(DataType::DT_INT16, {10, 32, 23}, "output");
        FUNCTION("ADD_021")
        {
            TileShape::Current().SetVecTile({10, 32, 25});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_021");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_022()
{
    PROGRAM("ADD_022")
    {
        Tensor input0(DataType::DT_INT32, {1, 23, 27}, "input0");
        Tensor input1(DataType::DT_INT32, {13, 23, 27}, "input1");
        auto output = Tensor(DataType::DT_INT32, {13, 23, 27}, "output");
        FUNCTION("ADD_022")
        {
            TileShape::Current().SetVecTile({10, 25, 30});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_022");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_023()
{
    PROGRAM("ADD_023")
    {
        Tensor input0(DataType::DT_FP32, {13, 1, 27}, "input0");
        Tensor input1(DataType::DT_FP32, {13, 23, 27}, "input1");
        auto output = Tensor(DataType::DT_FP32, {13, 23, 27}, "output");
        FUNCTION("ADD_023")
        {
            TileShape::Current().SetVecTile({23, 10, 30});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_023");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_024()
{
    PROGRAM("ADD_024")
    {
        Tensor input0(DataType::DT_INT16, {13, 23, 1}, "input0");
        Tensor input1(DataType::DT_INT16, {13, 23, 27}, "input1");
        auto output = Tensor(DataType::DT_INT16, {13, 23, 27}, "output");
        FUNCTION("ADD_024")
        {
            TileShape::Current().SetVecTile({23, 25, 15});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_024");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_025()
{
    PROGRAM("ADD_025")
    {
        Tensor input0(DataType::DT_FP16, {13, 23, 27}, "input0");
        Tensor input1(DataType::DT_FP16, {13, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {13, 23, 27}, "output");
        FUNCTION("ADD_025")
        {
            TileShape::Current().SetVecTile({10, 10, 30});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_025");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_026()
{
    PROGRAM("ADD_026")
    {
        Tensor input0(DataType::DT_INT16, {13, 23, 27}, "input0");
        Tensor input1(DataType::DT_INT16, {1, 23, 1}, "input1");
        auto output = Tensor(DataType::DT_INT16, {13, 23, 27}, "output");
        FUNCTION("ADD_026")
        {
            TileShape::Current().SetVecTile({10, 25, 10});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_026");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_027()
{
    PROGRAM("ADD_027")
    {
        Tensor input0(DataType::DT_FP16, {13, 23, 27}, "input0");
        Tensor input1(DataType::DT_FP16, {1, 1, 27}, "input1");
        auto output = Tensor(DataType::DT_FP16, {13, 23, 27}, "output");
        FUNCTION("ADD_027")
        {
            TileShape::Current().SetVecTile({23, 10, 10});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_027");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_028()
{
    PROGRAM("ADD_028")
    {
        Tensor input0(DataType::DT_FP32, {63, 1, 1}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 43, 27}, "input1");
        auto output = Tensor(DataType::DT_FP32, {63, 43, 27}, "output");
        FUNCTION("ADD_028")
        {
            TileShape::Current().SetVecTile({23, 20, 17});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_028");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_029()
{
    PROGRAM("ADD_029")
    {
        Tensor input0(DataType::DT_FP32, {5, 16, 11, 12}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {5, 16, 11, 12}, "output");
        FUNCTION("ADD_029")
        {
            TileShape::Current().SetVecTile({5, 20, 15, 12});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_029");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_030()
{
    PROGRAM("ADD_030")
    {
        Tensor input0(DataType::DT_FP16, {5, 5, 6, 7}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {5, 5, 6, 7}, "output");
        FUNCTION("ADD_030")
        {
            TileShape::Current().SetVecTile({5, 5, 10, 10});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_030");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_031()
{
    PROGRAM("ADD_031")
    {
        Tensor input0(DataType::DT_INT32, {21, 12, 15, 16}, "input0");
        Tensor input1(DataType::DT_INT32, {21, 12, 15, 16}, "input1");
        auto output = Tensor(DataType::DT_INT32, {21, 12, 15, 16}, "output");
        FUNCTION("ADD_031")
        {
            TileShape::Current().SetVecTile({5, 12, 16, 16});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_031");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_032()
{
    PROGRAM("ADD_032")
    {
        Tensor input0(DataType::DT_INT16, {11, 19, 13, 11}, "input0");
        Tensor input1(DataType::DT_INT16, {11, 19, 13, 11}, "input1");
        auto output = Tensor(DataType::DT_INT16, {11, 19, 13, 11}, "output");
        FUNCTION("ADD_032")
        {
            TileShape::Current().SetVecTile({12, 5, 15, 12});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_032");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_033()
{
    PROGRAM("ADD_033")
    {
        Tensor input0(DataType::DT_INT32, {1, 11, 13, 17}, "input0");
        Tensor input1(DataType::DT_INT32, {21, 11, 13, 17}, "input1");
        auto output = Tensor(DataType::DT_INT32, {21, 11, 13, 17}, "output");
        FUNCTION("ADD_033")
        {
            TileShape::Current().SetVecTile({21, 12, 5, 20});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_033");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_034()
{
    PROGRAM("ADD_034")
    {
        Tensor input0(DataType::DT_INT16, {11, 1, 15, 17}, "input0");
        Tensor input1(DataType::DT_INT16, {11, 11, 15, 17}, "input1");
        auto output = Tensor(DataType::DT_INT16, {11, 11, 15, 17}, "output");
        FUNCTION("ADD_034")
        {
            TileShape::Current().SetVecTile({11, 12, 15, 2});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_034");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_035()
{
    PROGRAM("ADD_035")
    {
        Tensor input0(DataType::DT_FP32, {21, 11, 1, 17}, "input0");
        Tensor input1(DataType::DT_FP32, {21, 11, 13, 17}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 11, 13, 17}, "output");
        FUNCTION("ADD_035")
        {
            TileShape::Current().SetVecTile({15, 5, 15, 20});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_035");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_036()
{
    PROGRAM("ADD_036")
    {
        Tensor input0(DataType::DT_FP16, {25, 11, 15, 1}, "input0");
        Tensor input1(DataType::DT_FP16, {25, 11, 15, 17}, "input1");
        auto output = Tensor(DataType::DT_FP16, {25, 11, 15, 17}, "output");
        FUNCTION("ADD_036")
        {
            TileShape::Current().SetVecTile({13, 12, 3, 18});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_036");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_037()
{
    PROGRAM("ADD_037")
    {
        Tensor input0(DataType::DT_FP32, {21, 11, 13, 17}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 1, 13, 17}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 11, 13, 17}, "output");
        FUNCTION("ADD_037")
        {
            TileShape::Current().SetVecTile({10, 12, 15, 6});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_037");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_038()
{
    PROGRAM("ADD_038")
    {
        Tensor input0(DataType::DT_FP16, {25, 11, 15, 17}, "input0");
        Tensor input1(DataType::DT_FP16, {1, 11, 1, 17}, "input1");
        auto output = Tensor(DataType::DT_FP16, {25, 11, 15, 17}, "output");
        FUNCTION("ADD_038")
        {
            TileShape::Current().SetVecTile({25, 7, 5, 18});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_038");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_039()
{
    PROGRAM("ADD_039")
    {
        Tensor input0(DataType::DT_FP32, {21, 11, 13, 17}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 11, 13, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 11, 13, 17}, "output");
        FUNCTION("ADD_039")
        {
            TileShape::Current().SetVecTile({21, 3, 13, 6});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_039");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_040()
{
    PROGRAM("ADD_040")
    {
        Tensor input0(DataType::DT_FP16, {25, 1, 1, 17}, "input0");
        Tensor input1(DataType::DT_FP16, {25, 11, 15, 17}, "input1");
        auto output = Tensor(DataType::DT_FP16, {25, 11, 15, 17}, "output");
        FUNCTION("ADD_040")
        {
            TileShape::Current().SetVecTile({25, 11, 5, 3});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_040");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_041()
{
    PROGRAM("ADD_041")
    {
        Tensor input0(DataType::DT_FP32, {22, 1, 13, 1}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 13, 18}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("ADD_041")
        {
            TileShape::Current().SetVecTile({5, 7, 7, 18});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_041");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_042()
{
    PROGRAM("ADD_042")
    {
        Tensor input0(DataType::DT_FP16, {22, 16, 1, 18}, "input0");
        Tensor input1(DataType::DT_FP16, {22, 16, 13, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {22, 16, 13, 18}, "output");
        FUNCTION("ADD_042")
        {
            TileShape::Current().SetVecTile({5, 7, 15, 5});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_042");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_043()
{
    PROGRAM("ADD_043")
    {
        Tensor input0(DataType::DT_FP32, {1, 1, 1, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 13, 18}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("ADD_043")
        {
            TileShape::Current().SetVecTile({5, 16, 7, 5});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_043");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_044()
{
    PROGRAM("ADD_044")
    {
        Tensor input0(DataType::DT_FP16, {1, 1, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP16, {22, 16, 13, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {22, 16, 13, 18}, "output");
        FUNCTION("ADD_044")
        {
            TileShape::Current().SetVecTile({22, 7, 7, 5});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_044");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_045()
{
    PROGRAM("ADD_045")
    {
        Tensor input0(DataType::DT_FP32, {1, 16, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("ADD_045")
        {
            TileShape::Current().SetVecTile({5, 16, 7, 5});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_045");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_046()
{
    PROGRAM("ADD_046")
    {
        Tensor input0(DataType::DT_FP16, {22, 16, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP16, {22, 1, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {22, 16, 13, 18}, "output");
        FUNCTION("ADD_046")
        {
            TileShape::Current().SetVecTile({22, 7, 7, 5});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_046");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_047()
{
    PROGRAM("ADD_047")
    {
        Tensor input0(DataType::DT_FP32, {22, 16, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 1, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("ADD_047")
        {
            TileShape::Current().SetVecTile({5, 16, 7, 5});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_047");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAdd::test_add_048()
{
    PROGRAM("ADD_048")
    {
        Tensor input0(DataType::DT_FP32, {1, 1, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("ADD_048")
        {
            TileShape::Current().SetVecTile({11, 12, 12, 10});
            output = Add(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ADD_048");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
