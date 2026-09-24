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
 * \file test_codegen_sub.cpp
 * \brief
 */

#include "include/test_codegen_sub.h"

using namespace npu::tile_fwk;

TestCodeGenSub::TestCodeGenSub() = default;
TestCodeGenSub::~TestCodeGenSub() = default;

TestCodeGenSub& TestCodeGenSub::Instance()
{
    static TestCodeGenSub instance;
    return instance;
}

void TestCodeGenSub::test_sub_001()
{
    PROGRAM("SUB_001")
    {
        Tensor input0(DataType::DT_INT32, {160}, "input0");
        Tensor input1(DataType::DT_INT32, {160}, "input1");
        auto output = Tensor(DataType::DT_INT32, {160}, "output");
        FUNCTION("SUB_001")
        {
            TileShape::Current().SetVecTile({200});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_002()
{
    PROGRAM("SUB_002")
    {
        Tensor input0(DataType::DT_INT16, {100}, "input0");
        Tensor input1(DataType::DT_INT16, {100}, "input1");
        auto output = Tensor(DataType::DT_INT16, {100}, "output");
        FUNCTION("SUB_002")
        {
            TileShape::Current().SetVecTile({100});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_003()
{
    PROGRAM("SUB_003")
    {
        Tensor input0(DataType::DT_FP32, {112}, "input0");
        Tensor input1(DataType::DT_FP32, {112}, "input1");
        auto output = Tensor(DataType::DT_FP32, {112}, "output");
        FUNCTION("SUB_003")
        {
            TileShape::Current().SetVecTile({100});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_004()
{
    PROGRAM("SUB_004")
    {
        Tensor input0(DataType::DT_FP16, {101}, "input0");
        Tensor input1(DataType::DT_FP16, {101}, "input1");
        auto output = Tensor(DataType::DT_FP16, {101}, "output");
        FUNCTION("SUB_004")
        {
            TileShape::Current().SetVecTile({100});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_005()
{
    PROGRAM("SUB_005")
    {
        Tensor input0(DataType::DT_FP32, {112}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {112}, "output");
        FUNCTION("SUB_005")
        {
            TileShape::Current().SetVecTile({100});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_006()
{
    PROGRAM("SUB_006")
    {
        Tensor input0(DataType::DT_FP16, {101}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {101}, "output");
        FUNCTION("SUB_006")
        {
            TileShape::Current().SetVecTile({100});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_007()
{
    PROGRAM("SUB_007")
    {
        Tensor input0(DataType::DT_INT32, {160}, "input0");
        Tensor input1(DataType::DT_INT32, {1}, "input1");
        auto output = Tensor(DataType::DT_INT32, {160}, "output");
        FUNCTION("SUB_007")
        {
            TileShape::Current().SetVecTile({120});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_008()
{
    PROGRAM("SUB_008")
    {
        Tensor input0(DataType::DT_INT16, {100}, "input0");
        Tensor input1(DataType::DT_INT16, {1}, "input1");
        auto output = Tensor(DataType::DT_INT16, {100}, "output");
        FUNCTION("SUB_008")
        {
            TileShape::Current().SetVecTile({112});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_009()
{
    PROGRAM("SUB_009")
    {
        Tensor input0(DataType::DT_FP32, {32, 20}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {32, 20}, "output");
        FUNCTION("SUB_009")
        {
            TileShape::Current().SetVecTile({64, 32});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_010()
{
    PROGRAM("SUB_010")
    {
        Tensor input0(DataType::DT_FP16, {31, 19}, "input0");
        Tensor input1(DataType::DT_FP16, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_FP16, {31, 19}, "output");
        FUNCTION("SUB_010")
        {
            TileShape::Current().SetVecTile({32, 10});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_011()
{
    PROGRAM("SUB_011")
    {
        Tensor input0(DataType::DT_INT32, {31, 19}, "input0");
        Tensor input1(DataType::DT_INT32, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_INT32, {31, 19}, "output");
        FUNCTION("SUB_011")
        {
            TileShape::Current().SetVecTile({10, 30});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_012()
{
    PROGRAM("SUB_012")
    {
        Tensor input0(DataType::DT_INT16, {31, 19}, "input0");
        Tensor input1(DataType::DT_INT16, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_INT16, {31, 19}, "output");
        FUNCTION("SUB_012")
        {
            TileShape::Current().SetVecTile({10, 14});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_013()
{
    PROGRAM("SUB_013")
    {
        Tensor input0(DataType::DT_INT32, {32, 20}, "input0");
        Tensor input1(DataType::DT_INT32, {1, 20}, "input1");
        auto output = Tensor(DataType::DT_INT32, {32, 20}, "output");
        FUNCTION("SUB_013")
        {
            TileShape::Current().SetVecTile({64, 32});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_014()
{
    PROGRAM("SUB_014")
    {
        Tensor input0(DataType::DT_INT32, {1, 20}, "input0");
        Tensor input1(DataType::DT_INT32, {32, 20}, "input1");
        auto output = Tensor(DataType::DT_INT32, {32, 20}, "output");
        FUNCTION("SUB_014")
        {
            TileShape::Current().SetVecTile({64, 32});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_015()
{
    PROGRAM("SUB_015")
    {
        Tensor input0(DataType::DT_INT16, {31, 21}, "input0");
        Tensor input1(DataType::DT_INT16, {31, 1}, "input1");
        auto output = Tensor(DataType::DT_INT16, {31, 21}, "output");
        FUNCTION("SUB_015")
        {
            TileShape::Current().SetVecTile({32, 10});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_016()
{
    PROGRAM("SUB_016")
    {
        Tensor input0(DataType::DT_FP32, {31, 1}, "input0");
        Tensor input1(DataType::DT_FP32, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_FP32, {31, 19}, "output");
        FUNCTION("SUB_016")
        {
            TileShape::Current().SetVecTile({10, 20});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_017()
{
    PROGRAM("SUB_017")
    {
        Tensor input0(DataType::DT_FP16, {1, 19}, "input0");
        Tensor input1(DataType::DT_FP16, {31, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {31, 19}, "output");
        FUNCTION("SUB_017")
        {
            TileShape::Current().SetVecTile({10, 16});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_018()
{
    PROGRAM("SUB_018")
    {
        Tensor input0(DataType::DT_FP32, {10, 32, 23}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {10, 32, 23}, "output");
        FUNCTION("SUB_018")
        {
            TileShape::Current().SetVecTile({10, 32, 25});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_019()
{
    PROGRAM("SUB_019")
    {
        Tensor input0(DataType::DT_FP16, {10, 32, 19}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {10, 32, 19}, "output");
        FUNCTION("SUB_019")
        {
            TileShape::Current().SetVecTile({10, 32, 20});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_020()
{
    PROGRAM("SUB_020")
    {
        Tensor input0(DataType::DT_INT32, {21, 19, 23}, "input0");
        Tensor input1(DataType::DT_INT32, {21, 19, 23}, "input1");
        auto output = Tensor(DataType::DT_INT32, {21, 19, 23}, "output");
        FUNCTION("SUB_020")
        {
            TileShape::Current().SetVecTile({25, 20, 25});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_020");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_021()
{
    PROGRAM("SUB_021")
    {
        Tensor input0(DataType::DT_INT16, {10, 32, 23}, "input0");
        Tensor input1(DataType::DT_INT16, {10, 32, 23}, "input1");
        auto output = Tensor(DataType::DT_INT16, {10, 32, 23}, "output");
        FUNCTION("SUB_021")
        {
            TileShape::Current().SetVecTile({10, 32, 25});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_021");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_022()
{
    PROGRAM("SUB_022")
    {
        Tensor input0(DataType::DT_INT32, {1, 23, 27}, "input0");
        Tensor input1(DataType::DT_INT32, {13, 23, 27}, "input1");
        auto output = Tensor(DataType::DT_INT32, {13, 23, 27}, "output");
        FUNCTION("SUB_022")
        {
            TileShape::Current().SetVecTile({10, 25, 30});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_022");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_023()
{
    PROGRAM("SUB_023")
    {
        Tensor input0(DataType::DT_FP32, {13, 1, 27}, "input0");
        Tensor input1(DataType::DT_FP32, {13, 23, 27}, "input1");
        auto output = Tensor(DataType::DT_FP32, {13, 23, 27}, "output");
        FUNCTION("SUB_023")
        {
            TileShape::Current().SetVecTile({23, 10, 30});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_023");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_024()
{
    PROGRAM("SUB_024")
    {
        Tensor input0(DataType::DT_INT16, {13, 23, 1}, "input0");
        Tensor input1(DataType::DT_INT16, {13, 23, 27}, "input1");
        auto output = Tensor(DataType::DT_INT16, {13, 23, 27}, "output");
        FUNCTION("SUB_024")
        {
            TileShape::Current().SetVecTile({23, 25, 15});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_024");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_025()
{
    PROGRAM("SUB_025")
    {
        Tensor input0(DataType::DT_FP16, {13, 23, 27}, "input0");
        Tensor input1(DataType::DT_FP16, {13, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {13, 23, 27}, "output");
        FUNCTION("SUB_025")
        {
            TileShape::Current().SetVecTile({10, 10, 30});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_025");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_026()
{
    PROGRAM("SUB_026")
    {
        Tensor input0(DataType::DT_INT16, {13, 23, 27}, "input0");
        Tensor input1(DataType::DT_INT16, {1, 23, 1}, "input1");
        auto output = Tensor(DataType::DT_INT16, {13, 23, 27}, "output");
        FUNCTION("SUB_026")
        {
            TileShape::Current().SetVecTile({10, 25, 10});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_026");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_027()
{
    PROGRAM("SUB_027")
    {
        Tensor input0(DataType::DT_FP16, {13, 23, 27}, "input0");
        Tensor input1(DataType::DT_FP16, {1, 1, 27}, "input1");
        auto output = Tensor(DataType::DT_FP16, {13, 23, 27}, "output");
        FUNCTION("SUB_027")
        {
            TileShape::Current().SetVecTile({23, 10, 10});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_027");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_028()
{
    PROGRAM("SUB_028")
    {
        Tensor input0(DataType::DT_FP32, {63, 1, 1}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 43, 27}, "input1");
        auto output = Tensor(DataType::DT_FP32, {63, 43, 27}, "output");
        FUNCTION("SUB_028")
        {
            TileShape::Current().SetVecTile({23, 20, 17});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_028");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_029()
{
    PROGRAM("SUB_029")
    {
        Tensor input0(DataType::DT_FP32, {5, 16, 11, 12}, "input0");
        Element input1(DataType::DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {5, 16, 11, 12}, "output");
        FUNCTION("SUB_029")
        {
            TileShape::Current().SetVecTile({5, 20, 15, 12});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_029");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_030()
{
    PROGRAM("SUB_030")
    {
        Tensor input0(DataType::DT_FP16, {5, 5, 6, 7}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {5, 5, 6, 7}, "output");
        FUNCTION("SUB_030")
        {
            TileShape::Current().SetVecTile({5, 5, 10, 10});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_030");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_031()
{
    PROGRAM("SUB_031")
    {
        Tensor input0(DataType::DT_INT32, {21, 12, 15, 16}, "input0");
        Tensor input1(DataType::DT_INT32, {21, 12, 15, 16}, "input1");
        auto output = Tensor(DataType::DT_INT32, {21, 12, 15, 16}, "output");
        FUNCTION("SUB_031")
        {
            TileShape::Current().SetVecTile({5, 12, 16, 16});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_031");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_032()
{
    PROGRAM("SUB_032")
    {
        Tensor input0(DataType::DT_INT16, {11, 19, 13, 11}, "input0");
        Tensor input1(DataType::DT_INT16, {11, 19, 13, 11}, "input1");
        auto output = Tensor(DataType::DT_INT16, {11, 19, 13, 11}, "output");
        FUNCTION("SUB_032")
        {
            TileShape::Current().SetVecTile({12, 5, 15, 12});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_032");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_033()
{
    PROGRAM("SUB_033")
    {
        Tensor input0(DataType::DT_INT32, {1, 11, 13, 17}, "input0");
        Tensor input1(DataType::DT_INT32, {21, 11, 13, 17}, "input1");
        auto output = Tensor(DataType::DT_INT32, {21, 11, 13, 17}, "output");
        FUNCTION("SUB_033")
        {
            TileShape::Current().SetVecTile({21, 12, 5, 20});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_033");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_034()
{
    PROGRAM("SUB_034")
    {
        Tensor input0(DataType::DT_INT16, {11, 1, 15, 17}, "input0");
        Tensor input1(DataType::DT_INT16, {11, 11, 15, 17}, "input1");
        auto output = Tensor(DataType::DT_INT16, {11, 11, 15, 17}, "output");
        FUNCTION("SUB_034")
        {
            TileShape::Current().SetVecTile({11, 12, 15, 2});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_034");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_035()
{
    PROGRAM("SUB_035")
    {
        Tensor input0(DataType::DT_FP32, {21, 11, 1, 17}, "input0");
        Tensor input1(DataType::DT_FP32, {21, 11, 13, 17}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 11, 13, 17}, "output");
        FUNCTION("SUB_035")
        {
            TileShape::Current().SetVecTile({15, 5, 15, 20});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_035");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_036()
{
    PROGRAM("SUB_036")
    {
        Tensor input0(DataType::DT_FP16, {25, 11, 15, 1}, "input0");
        Tensor input1(DataType::DT_FP16, {25, 11, 15, 17}, "input1");
        auto output = Tensor(DataType::DT_FP16, {25, 11, 15, 17}, "output");
        FUNCTION("SUB_036")
        {
            TileShape::Current().SetVecTile({13, 12, 3, 18});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_036");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_037()
{
    PROGRAM("SUB_037")
    {
        Tensor input0(DataType::DT_FP32, {21, 11, 13, 17}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 1, 13, 17}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 11, 13, 17}, "output");
        FUNCTION("SUB_037")
        {
            TileShape::Current().SetVecTile({10, 12, 15, 6});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_037");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_038()
{
    PROGRAM("SUB_038")
    {
        Tensor input0(DataType::DT_FP16, {25, 11, 15, 17}, "input0");
        Tensor input1(DataType::DT_FP16, {1, 11, 1, 17}, "input1");
        auto output = Tensor(DataType::DT_FP16, {25, 11, 15, 17}, "output");
        FUNCTION("SUB_038")
        {
            TileShape::Current().SetVecTile({25, 7, 5, 18});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_038");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_039()
{
    PROGRAM("SUB_039")
    {
        Tensor input0(DataType::DT_FP32, {21, 11, 13, 17}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 11, 13, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 11, 13, 17}, "output");
        FUNCTION("SUB_039")
        {
            TileShape::Current().SetVecTile({21, 3, 13, 6});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_039");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_040()
{
    PROGRAM("SUB_040")
    {
        Tensor input0(DataType::DT_FP16, {25, 1, 1, 17}, "input0");
        Tensor input1(DataType::DT_FP16, {25, 11, 15, 17}, "input1");
        auto output = Tensor(DataType::DT_FP16, {25, 11, 15, 17}, "output");
        FUNCTION("SUB_040")
        {
            TileShape::Current().SetVecTile({25, 11, 5, 3});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_040");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_041()
{
    PROGRAM("SUB_041")
    {
        Tensor input0(DataType::DT_FP32, {22, 1, 13, 1}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 13, 18}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("SUB_041")
        {
            TileShape::Current().SetVecTile({5, 7, 7, 18});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_041");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_042()
{
    PROGRAM("SUB_042")
    {
        Tensor input0(DataType::DT_FP16, {22, 16, 1, 18}, "input0");
        Tensor input1(DataType::DT_FP16, {22, 16, 13, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {22, 16, 13, 18}, "output");
        FUNCTION("SUB_042")
        {
            TileShape::Current().SetVecTile({5, 7, 15, 5});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_042");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_043()
{
    PROGRAM("SUB_043")
    {
        Tensor input0(DataType::DT_FP32, {1, 1, 1, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 13, 18}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("SUB_043")
        {
            TileShape::Current().SetVecTile({5, 16, 7, 5});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_043");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_044()
{
    PROGRAM("SUB_044")
    {
        Tensor input0(DataType::DT_FP16, {1, 1, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP16, {22, 16, 13, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {22, 16, 13, 18}, "output");
        FUNCTION("SUB_044")
        {
            TileShape::Current().SetVecTile({22, 7, 7, 5});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_044");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_045()
{
    PROGRAM("SUB_045")
    {
        Tensor input0(DataType::DT_FP32, {1, 16, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("SUB_045")
        {
            TileShape::Current().SetVecTile({5, 16, 7, 5});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_045");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_046()
{
    PROGRAM("SUB_046")
    {
        Tensor input0(DataType::DT_FP16, {22, 16, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP16, {22, 1, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {22, 16, 13, 18}, "output");
        FUNCTION("SUB_046")
        {
            TileShape::Current().SetVecTile({22, 7, 7, 5});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_046");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_047()
{
    PROGRAM("SUB_047")
    {
        Tensor input0(DataType::DT_FP32, {22, 16, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 1, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("SUB_047")
        {
            TileShape::Current().SetVecTile({5, 16, 7, 5});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_047");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenSub::test_sub_048()
{
    PROGRAM("SUB_048")
    {
        Tensor input0(DataType::DT_FP32, {1, 1, 13, 18}, "input0");
        Tensor input1(DataType::DT_FP32, {22, 16, 1, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {22, 16, 13, 18}, "output");
        FUNCTION("SUB_048")
        {
            TileShape::Current().SetVecTile({11, 12, 12, 10});
            output = Sub(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SUB_048");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
