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
 * \file test_codegen_cast.cpp
 * \brief
 */

#include "include/test_codegen_cast.h"

using namespace npu::tile_fwk;

TestCodeGenCast::TestCodeGenCast() = default;
TestCodeGenCast::~TestCodeGenCast() = default;

TestCodeGenCast& TestCodeGenCast::Instance()
{
    static TestCodeGenCast instance;
    return instance;
}

void TestCodeGenCast::test_Cast_001()
{
    PROGRAM("CAST_001")
    {
        Tensor input(DataType::DT_FP16, {112}, "input");
        auto output = Tensor(DataType::DT_FP32, {112}, "output");
        FUNCTION("CAST_001")
        {
            TileShape::Current().SetVecTile({50});
            output = Cast(input, DataType::DT_FP32, CAST_NONE, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_002()
{
    PROGRAM("CAST_002")
    {
        Tensor input(DataType::DT_INT32, {100}, "input");
        auto output = Tensor(DataType::DT_FP32, {100}, "output");
        FUNCTION("CAST_002")
        {
            TileShape::Current().SetVecTile({100});
            output = Cast(input, DataType::DT_FP32, CAST_RINT, SaturationMode::ON);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_003()
{
    PROGRAM("CAST_003")
    {
        Tensor input(DataType::DT_INT16, {4, 128}, "input");
        auto output = Tensor(DataType::DT_FP32, {4, 128}, "output");
        FUNCTION("CAST_003")
        {
            TileShape::Current().SetVecTile({2, 32});
            output = Cast(input, DataType::DT_FP32, CAST_ROUND, SaturationMode::ON);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_004()
{
    PROGRAM("CAST_004")
    {
        Tensor input(DataType::DT_FP32, {4, 130}, "input");
        auto output = Tensor(DataType::DT_FP16, {4, 130}, "output");
        FUNCTION("CAST_004")
        {
            TileShape::Current().SetVecTile({1, 130});
            output = Cast(input, DataType::DT_FP16, CAST_ODD, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_005()
{
    PROGRAM("CAST_005")
    {
        Tensor input(DataType::DT_INT8, {2, 4, 160}, "input");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 160}, "output");
        FUNCTION("CAST_005")
        {
            TileShape::Current().SetVecTile({1, 2, 32});
            output = Cast(input, DataType::DT_FP16, CAST_CEIL, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_006()
{
    PROGRAM("CAST_006")
    {
        Tensor input(DataType::DT_UINT8, {2, 4, 140}, "input");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 140}, "output");
        FUNCTION("CAST_006")
        {
            TileShape::Current().SetVecTile({1, 2, 140});
            output = Cast(input, DataType::DT_FP16, CAST_TRUNC, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_007()
{
    PROGRAM("CAST_007")
    {
        Tensor input(DataType::DT_INT16, {2, 5, 152}, "input");
        auto output = Tensor(DataType::DT_FP16, {2, 5, 152}, "output");
        FUNCTION("CAST_007")
        {
            TileShape::Current().SetVecTile({1, 5, 32});
            output = Cast(input, DataType::DT_FP16, CAST_FLOOR, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_008()
{
    PROGRAM("CAST_008")
    {
        Tensor input(DataType::DT_FP32, {2, 3, 170}, "input");
        auto output = Tensor(DataType::DT_INT16, {2, 3, 170}, "output");
        FUNCTION("CAST_008")
        {
            TileShape::Current().SetVecTile({1, 3, 170});
            output = Cast(input, DataType::DT_INT16, CAST_NONE, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_009()
{
    PROGRAM("CAST_009")
    {
        Tensor input(DataType::DT_FP32, {5, 2, 4, 176}, "input");
        auto output = Tensor(DataType::DT_INT32, {5, 2, 4, 176}, "output");
        FUNCTION("CAST_009")
        {
            TileShape::Current().SetVecTile({2, 1, 2, 16});
            output = Cast(input, DataType::DT_INT32, CAST_NONE, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_010()
{
    PROGRAM("CAST_010")
    {
        Tensor input(DataType::DT_FP16, {5, 2, 4, 130}, "input");
        auto output = Tensor(DataType::DT_INT8, {5, 2, 4, 130}, "output");
        FUNCTION("CAST_010")
        {
            TileShape::Current().SetVecTile({1, 1, 1, 130});
            output = Cast(input, DataType::DT_INT8, CAST_NONE, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_011()
{
    PROGRAM("CAST_011")
    {
        Tensor input(DataType::DT_FP16, {2, 3, 5, 134}, "input");
        auto output = Tensor(DataType::DT_UINT8, {2, 3, 5, 134}, "output");
        FUNCTION("CAST_011")
        {
            TileShape::Current().SetVecTile({1, 1, 5, 32});
            output = Cast(input, DataType::DT_UINT8, CAST_NONE, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_012()
{
    PROGRAM("CAST_012")
    {
        Tensor input(DataType::DT_FP16, {4, 2, 6, 135}, "input");
        auto output = Tensor(DataType::DT_INT16, {4, 2, 6, 135}, "output");
        FUNCTION("CAST_012")
        {
            TileShape::Current().SetVecTile({2, 2, 3, 32});
            output = Cast(input, DataType::DT_INT16, CAST_NONE, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_013()
{
    PROGRAM("CAST_013")
    {
        Tensor input(DataType::DT_FP16, {6, 2, 4, 130}, "input");
        auto output = Tensor(DataType::DT_INT32, {6, 2, 4, 130}, "output");
        FUNCTION("CAST_013")
        {
            TileShape::Current().SetVecTile({1, 1, 4, 130});
            output = Cast(input, DataType::DT_INT32, CAST_NONE, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_014()
{
    PROGRAM("CAST_014")
    {
        Tensor input(DataType::DT_FP16, {3, 2, 3, 139}, "input");
        auto output = Tensor(DataType::DT_FP32, {3, 2, 3, 139}, "output");
        FUNCTION("CAST_014")
        {
            TileShape::Current().SetVecTile({1, 2, 1, 139});
            output = Cast(input, DataType::DT_FP32, CAST_NONE, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_015()
{
    PROGRAM("CAST_015")
    {
        Tensor input(DataType::DT_FP16, {6, 3, 5, 141}, "input");
        auto output = Tensor(DataType::DT_FP32, {6, 3, 5, 141}, "output");
        FUNCTION("CAST_015")
        {
            TileShape::Current().SetVecTile({3, 3, 5, 32});
            output = Cast(input, DataType::DT_FP32, CAST_NONE, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_016()
{
    PROGRAM("CAST_016")
    {
        Tensor input(DataType::DT_FP32, {2, 3, 160}, "input");
        auto output = Tensor(DataType::DT_INT16, {2, 3, 160}, "output");
        FUNCTION("CAST_016")
        {
            TileShape::Current().SetVecTile({1, 3, 32});
            output = Cast(input, DataType::DT_INT16, CAST_ROUND, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_017()
{
    PROGRAM("CAST_017")
    {
        Tensor input(DataType::DT_FP32, {3, 4, 128}, "input");
        auto output = Tensor(DataType::DT_INT16, {3, 4, 128}, "output");
        FUNCTION("CAST_017")
        {
            TileShape::Current().SetVecTile({2, 2, 64});
            output = Cast(input, DataType::DT_INT16, CAST_FLOOR, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_018()
{
    PROGRAM("CAST_018")
    {
        Tensor input(DataType::DT_FP32, {2, 5, 144}, "input");
        auto output = Tensor(DataType::DT_INT16, {2, 5, 144}, "output");
        FUNCTION("CAST_018")
        {
            TileShape::Current().SetVecTile({1, 5, 144});
            output = Cast(input, DataType::DT_INT16, CAST_CEIL, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_019()
{
    PROGRAM("CAST_019")
    {
        Tensor input(DataType::DT_FP32, {2, 4, 160}, "input");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 160}, "output");
        FUNCTION("CAST_019")
        {
            TileShape::Current().SetVecTile({1, 2, 160});
            output = Cast(input, DataType::DT_FP16, CAST_ROUND, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_020()
{
    PROGRAM("CAST_020")
    {
        Tensor input(DataType::DT_FP32, {3, 5, 136}, "input");
        auto output = Tensor(DataType::DT_FP16, {3, 5, 136}, "output");
        FUNCTION("CAST_020")
        {
            TileShape::Current().SetVecTile({2, 5, 136});
            output = Cast(input, DataType::DT_FP16, CAST_CEIL, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_020");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenCast::test_Cast_021()
{
    PROGRAM("CAST_021")
    {
        Tensor input(DataType::DT_FP32, {4, 6, 148}, "input");
        auto output = Tensor(DataType::DT_FP16, {4, 6, 148}, "output");
        FUNCTION("CAST_021")
        {
            TileShape::Current().SetVecTile({2, 3, 148});
            output = Cast(input, DataType::DT_FP16, CAST_ODD, SaturationMode::OFF);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CAST_021");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
