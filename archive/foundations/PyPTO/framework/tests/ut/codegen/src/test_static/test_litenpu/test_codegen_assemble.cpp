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
 * \file test_codegen_assemble.cpp
 * \brief
 */

#include "include/test_codegen_assemble.h"

using namespace npu::tile_fwk;

TestCodeGenAssemble::TestCodeGenAssemble() = default;
TestCodeGenAssemble::~TestCodeGenAssemble() = default;

TestCodeGenAssemble& TestCodeGenAssemble::Instance()
{
    static TestCodeGenAssemble instance;
    return instance;
}

void TestCodeGenAssemble::test_assemble_fp16_001()
{
    PROGRAM("ASSEMBLE_FP16_001")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor input(DT_FP16, {2, 2}, "input");
        Tensor out(DT_FP16, {4, 4}, "out");
        FUNCTION("ASSEMBLE_FP16_001") { Assemble({{input, {0, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp16_002()
{
    PROGRAM("ASSEMBLE_FP16_002")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor input(DT_FP16, {3, 3}, "input");
        Tensor out(DT_FP16, {5, 5}, "out");
        FUNCTION("ASSEMBLE_FP16_002") { Assemble({{input, {1, 1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp16_003()
{
    PROGRAM("ASSEMBLE_FP16_003")
    {
        TileShape::Current().SetVecTile({1, 1, 16});
        Tensor input(DT_FP16, {2, 2, 2}, "input");
        Tensor out(DT_FP16, {3, 3, 3}, "out");
        FUNCTION("ASSEMBLE_FP16_003") { Assemble({{input, {0, 0, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp16_004()
{
    PROGRAM("ASSEMBLE_FP16_004")
    {
        TileShape::Current().SetVecTile({16});
        Tensor input(DT_FP16, {4}, "input");
        Tensor out(DT_FP16, {6}, "out");
        FUNCTION("ASSEMBLE_FP16_004") { Assemble({{input, {1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp16_005()
{
    PROGRAM("ASSEMBLE_FP16_005")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor input(DT_FP16, {2, 4}, "input");
        Tensor out(DT_FP16, {3, 6}, "out");
        FUNCTION("ASSEMBLE_FP16_005") { Assemble({{input, {0, 2}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp16_006()
{
    PROGRAM("ASSEMBLE_FP16_006")
    {
        TileShape::Current().SetVecTile({1, 1, 16});
        Tensor input(DT_FP16, {1, 2, 2}, "input");
        Tensor out(DT_FP16, {2, 3, 3}, "out");
        FUNCTION("ASSEMBLE_FP16_006") { Assemble({{input, {0, 1, 1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP16_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp16_007()
{
    PROGRAM("ASSEMBLE_FP16_007")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 16});
        Tensor input(DT_FP16, {2, 2, 2, 2}, "input");
        Tensor out(DT_FP16, {3, 3, 3, 3}, "out");
        FUNCTION("ASSEMBLE_FP16_007") { Assemble({{input, {0, 0, 0, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP16_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp16_008()
{
    PROGRAM("ASSEMBLE_FP16_008")
    {
        TileShape::Current().SetVecTile({16});
        Tensor input(DT_FP16, {3}, "input");
        Tensor out(DT_FP16, {5}, "out");
        FUNCTION("ASSEMBLE_FP16_008") { Assemble({{input, {0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP16_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp16_009()
{
    PROGRAM("ASSEMBLE_FP16_009")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor input(DT_FP16, {1, 3}, "input");
        Tensor out(DT_FP16, {2, 5}, "out");
        FUNCTION("ASSEMBLE_FP16_009") { Assemble({{input, {0, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP16_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp16_010()
{
    PROGRAM("ASSEMBLE_FP16_010")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor input1(DT_FP16, {2, 2}, "input1");
        Tensor input2(DT_FP16, {2, 2}, "input2");
        Tensor out(DT_FP16, {4, 4}, "out");
        FUNCTION("ASSEMBLE_FP16_010") { Assemble({{input1, {0, 0}}, {input2, {2, 2}}}, out, true); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP16_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp32_001()
{
    PROGRAM("ASSEMBLE_FP32_001")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input(DT_FP32, {2, 2}, "input");
        Tensor out(DT_FP32, {4, 4}, "out");
        FUNCTION("ASSEMBLE_FP32_001") { Assemble({{input, {0, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp32_002()
{
    PROGRAM("ASSEMBLE_FP32_002")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input(DT_FP32, {3, 3}, "input");
        Tensor out(DT_FP32, {5, 5}, "out");
        FUNCTION("ASSEMBLE_FP32_002") { Assemble({{input, {1, 1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp32_003()
{
    PROGRAM("ASSEMBLE_FP32_003")
    {
        TileShape::Current().SetVecTile({1, 1, 8});
        Tensor input(DT_FP32, {2, 2, 2}, "input");
        Tensor out(DT_FP32, {3, 3, 3}, "out");
        FUNCTION("ASSEMBLE_FP32_003") { Assemble({{input, {0, 0, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp32_004()
{
    PROGRAM("ASSEMBLE_FP32_004")
    {
        TileShape::Current().SetVecTile({8});
        Tensor input(DT_FP32, {4}, "input");
        Tensor out(DT_FP32, {6}, "out");
        FUNCTION("ASSEMBLE_FP32_004") { Assemble({{input, {1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp32_005()
{
    PROGRAM("ASSEMBLE_FP32_005")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input(DT_FP32, {2, 4}, "input");
        Tensor out(DT_FP32, {3, 6}, "out");
        FUNCTION("ASSEMBLE_FP32_005") { Assemble({{input, {0, 2}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp32_006()
{
    PROGRAM("ASSEMBLE_FP32_006")
    {
        TileShape::Current().SetVecTile({1, 1, 8});
        Tensor input(DT_FP32, {1, 2, 2}, "input");
        Tensor out(DT_FP32, {2, 3, 3}, "out");
        FUNCTION("ASSEMBLE_FP32_006") { Assemble({{input, {0, 1, 1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp32_007()
{
    PROGRAM("ASSEMBLE_FP32_007")
    {
        TileShape::Current().SetVecTile({1, 1, 1, 8});
        Tensor input(DT_FP32, {2, 2, 2, 2}, "input");
        Tensor out(DT_FP32, {3, 3, 3, 3}, "out");
        FUNCTION("ASSEMBLE_FP32_007") { Assemble({{input, {0, 0, 0, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp32_008()
{
    PROGRAM("ASSEMBLE_FP32_008")
    {
        TileShape::Current().SetVecTile({8});
        Tensor input(DT_FP32, {3}, "input");
        Tensor out(DT_FP32, {5}, "out");
        FUNCTION("ASSEMBLE_FP32_008") { Assemble({{input, {0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp32_009()
{
    PROGRAM("ASSEMBLE_FP32_009")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input(DT_FP32, {1, 3}, "input");
        Tensor out(DT_FP32, {2, 5}, "out");
        FUNCTION("ASSEMBLE_FP32_009") { Assemble({{input, {0, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_fp32_010()
{
    PROGRAM("ASSEMBLE_FP32_010")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input1(DT_FP32, {2, 2}, "input1");
        Tensor input2(DT_FP32, {2, 2}, "input2");
        Tensor out(DT_FP32, {4, 4}, "out");
        FUNCTION("ASSEMBLE_FP32_010") { Assemble({{input1, {0, 0}}, {input2, {2, 2}}}, out, true); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int8_001()
{
    PROGRAM("ASSEMBLE_INT8_001")
    {
        TileShape::Current().SetVecTile({1, 32});
        Tensor input(DT_INT8, {2, 2}, "input");
        Tensor out(DT_INT8, {4, 4}, "out");
        FUNCTION("ASSEMBLE_INT8_001") { Assemble({{input, {0, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT8_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int8_002()
{
    PROGRAM("ASSEMBLE_INT8_002")
    {
        TileShape::Current().SetVecTile({1, 32});
        Tensor input(DT_INT8, {3, 3}, "input");
        Tensor out(DT_INT8, {5, 5}, "out");
        FUNCTION("ASSEMBLE_INT8_002") { Assemble({{input, {1, 1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT8_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int8_003()
{
    PROGRAM("ASSEMBLE_INT8_003")
    {
        TileShape::Current().SetVecTile({32});
        Tensor input(DT_INT8, {4}, "input");
        Tensor out(DT_INT8, {6}, "out");
        FUNCTION("ASSEMBLE_INT8_003") { Assemble({{input, {1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT8_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int8_004()
{
    PROGRAM("ASSEMBLE_INT8_004")
    {
        TileShape::Current().SetVecTile({1, 32});
        Tensor input(DT_INT8, {2, 4}, "input");
        Tensor out(DT_INT8, {3, 6}, "out");
        FUNCTION("ASSEMBLE_INT8_004") { Assemble({{input, {0, 2}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT8_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int8_005()
{
    PROGRAM("ASSEMBLE_INT8_005")
    {
        TileShape::Current().SetVecTile({1, 32});
        Tensor input1(DT_INT8, {2, 2}, "input1");
        Tensor input2(DT_INT8, {2, 2}, "input2");
        Tensor out(DT_INT8, {4, 4}, "out");
        FUNCTION("ASSEMBLE_INT8_005") { Assemble({{input1, {0, 0}}, {input2, {2, 2}}}, out, true); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT8_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int16_001()
{
    PROGRAM("ASSEMBLE_INT16_001")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor input(DT_INT16, {2, 2}, "input");
        Tensor out(DT_INT16, {4, 4}, "out");
        FUNCTION("ASSEMBLE_INT16_001") { Assemble({{input, {0, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int16_002()
{
    PROGRAM("ASSEMBLE_INT16_002")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor input(DT_INT16, {3, 3}, "input");
        Tensor out(DT_INT16, {5, 5}, "out");
        FUNCTION("ASSEMBLE_INT16_002") { Assemble({{input, {1, 1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int16_003()
{
    PROGRAM("ASSEMBLE_INT16_003")
    {
        TileShape::Current().SetVecTile({16});
        Tensor input(DT_INT16, {4}, "input");
        Tensor out(DT_INT16, {6}, "out");
        FUNCTION("ASSEMBLE_INT16_003") { Assemble({{input, {1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int16_004()
{
    PROGRAM("ASSEMBLE_INT16_004")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor input(DT_INT16, {2, 4}, "input");
        Tensor out(DT_INT16, {3, 6}, "out");
        FUNCTION("ASSEMBLE_INT16_004") { Assemble({{input, {0, 2}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int16_005()
{
    PROGRAM("ASSEMBLE_INT16_005")
    {
        TileShape::Current().SetVecTile({1, 16});
        Tensor input1(DT_INT16, {2, 2}, "input1");
        Tensor input2(DT_INT16, {2, 2}, "input2");
        Tensor out(DT_INT16, {4, 4}, "out");
        FUNCTION("ASSEMBLE_INT16_005") { Assemble({{input1, {0, 0}}, {input2, {2, 2}}}, out, true); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int32_001()
{
    PROGRAM("ASSEMBLE_INT32_001")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input(DT_INT32, {2, 2}, "input");
        Tensor out(DT_INT32, {4, 4}, "out");
        FUNCTION("ASSEMBLE_INT32_001") { Assemble({{input, {0, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int32_002()
{
    PROGRAM("ASSEMBLE_INT32_002")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input(DT_INT32, {3, 3}, "input");
        Tensor out(DT_INT32, {5, 5}, "out");
        FUNCTION("ASSEMBLE_INT32_002") { Assemble({{input, {1, 1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int32_003()
{
    PROGRAM("ASSEMBLE_INT32_003")
    {
        TileShape::Current().SetVecTile({8});
        Tensor input(DT_INT32, {4}, "input");
        Tensor out(DT_INT32, {6}, "out");
        FUNCTION("ASSEMBLE_INT32_003") { Assemble({{input, {1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int32_004()
{
    PROGRAM("ASSEMBLE_INT32_004")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input(DT_INT32, {2, 4}, "input");
        Tensor out(DT_INT32, {3, 6}, "out");
        FUNCTION("ASSEMBLE_INT32_004") { Assemble({{input, {0, 2}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_int32_005()
{
    PROGRAM("ASSEMBLE_INT32_005")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input1(DT_INT32, {2, 2}, "input1");
        Tensor input2(DT_INT32, {2, 2}, "input2");
        Tensor out(DT_INT32, {4, 4}, "out");
        FUNCTION("ASSEMBLE_INT32_005") { Assemble({{input1, {0, 0}}, {input2, {2, 2}}}, out, true); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_INT32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_list_fp32_001()
{
    PROGRAM("ASSEMBLE_LIST_FP32_001")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input1(DT_FP32, {2, 2}, "input1");
        Tensor input2(DT_FP32, {2, 2}, "input2");
        Tensor out(DT_FP32, {4, 4}, "out");
        FUNCTION("ASSEMBLE_LIST_FP32_001") { Assemble({{input1, {0, 0}}, {input2, {2, 2}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_LIST_FP32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_list_fp32_002()
{
    PROGRAM("ASSEMBLE_LIST_FP32_002")
    {
        TileShape::Current().SetVecTile({8});
        Tensor input1(DT_FP32, {2}, "input1");
        Tensor input2(DT_FP32, {2}, "input2");
        Tensor out(DT_FP32, {8}, "out");
        FUNCTION("ASSEMBLE_LIST_FP32_002") { Assemble({{input1, {1}}, {input2, {3}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_LIST_FP32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_list_fp32_003()
{
    PROGRAM("ASSEMBLE_LIST_FP32_003")
    {
        TileShape::Current().SetVecTile({1, 1, 8});
        Tensor input1(DT_FP32, {2, 2, 2}, "input1");
        Tensor input2(DT_FP32, {2, 2, 2}, "input2");
        Tensor out(DT_FP32, {3, 3, 3}, "out");
        FUNCTION("ASSEMBLE_LIST_FP32_003") { Assemble({{input1, {0, 0, 0}}, {input2, {1, 1, 1}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_LIST_FP32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_list_multi_shape_001()
{
    PROGRAM("ASSEMBLE_LIST_MULTI_SHAPE_001")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input1(DT_FP32, {2, 2}, "input1");
        Tensor input2(DT_FP32, {2, 3}, "input2");
        Tensor out(DT_FP32, {4, 6}, "out");
        FUNCTION("ASSEMBLE_LIST_MULTI_SHAPE_001") { Assemble({{input1, {0, 0}}, {input2, {2, 2}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_LIST_MULTI_SHAPE_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_list_multi_shape_002()
{
    PROGRAM("ASSEMBLE_LIST_MULTI_SHAPE_002")
    {
        TileShape::Current().SetVecTile({1, 8});
        Tensor input1(DT_FP32, {3, 2}, "input1");
        Tensor input2(DT_FP32, {3, 2}, "input2");
        Tensor out(DT_FP32, {5, 4}, "out");
        FUNCTION("ASSEMBLE_LIST_MULTI_SHAPE_002") { Assemble({{input1, {0, 0}}, {input2, {2, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_LIST_MULTI_SHAPE_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenAssemble::test_assemble_list_multi_shape_003()
{
    PROGRAM("ASSEMBLE_LIST_MULTI_SHAPE_003")
    {
        TileShape::Current().SetVecTile({200, 160});
        Tensor input1(DT_FP32, {300, 200}, "input1");
        Tensor input2(DT_FP32, {300, 200}, "input2");
        Tensor out(DT_FP32, {500, 400}, "out");
        FUNCTION("ASSEMBLE_LIST_MULTI_SHAPE_003") { Assemble({{input1, {0, 0}}, {input2, {2, 0}}}, out, false); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_LIST_MULTI_SHAPE_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
