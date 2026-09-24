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
 * \file test_codegen_matmul.cpp
 * \brief
 */

#include "include/test_codegen_matmul.h"

using namespace npu::tile_fwk;

TestCodeGenMatmul::TestCodeGenMatmul() = default;
TestCodeGenMatmul::~TestCodeGenMatmul() = default;

TestCodeGenMatmul& TestCodeGenMatmul::Instance()
{
    static TestCodeGenMatmul instance;
    return instance;
}

void TestCodeGenMatmul::test_matmul_001()
{
    PROGRAM("MATMUL_001")
    {
        Tensor a(DataType::DT_FP16, {16, 16}, "a");
        Tensor b(DataType::DT_FP16, {16, 16}, "b");
        auto c = Tensor(DataType::DT_FP16, {16, 16}, "c");
        FUNCTION("MATMUL_001")
        {
            TileShape::Current().SetCubeTile({16, 16}, {16, 16}, {16, 16});
            c = npu::tile_fwk::Matrix::Matmul(DataType::DT_FP16, a, b, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_002()
{
    PROGRAM("MATMUL_002")
    {
        Tensor a(DataType::DT_FP16, {128, 130}, "a");
        Tensor b(DataType::DT_FP16, {130, 32}, "b");
        Tensor bias(DataType::DT_FP16, {1, 32}, "bias");
        auto c = Tensor(DataType::DT_FP16, {128, 32}, "c");
        FUNCTION("MATMUL_002")
        {
            TileShape::Current().SetCubeTile({16, 16}, {32, 32}, {16, 16});
            npu::tile_fwk::Matrix::MatmulExtendParam extendParam;
            extendParam.biasTensor = bias;
            c = npu::tile_fwk::Matrix::Matmul(DataType::DT_FP16, a, b, extendParam, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_003()
{
    PROGRAM("MATMUL_003")
    {
        Tensor a(DataType::DT_FP16, {30, 150}, "a");
        Tensor b(DataType::DT_FP16, {150, 60}, "b");
        auto c = Tensor(DataType::DT_FP16, {30, 60}, "c");
        FUNCTION("MATMUL_003")
        {
            TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {64, 64});
            c = npu::tile_fwk::Matrix::Matmul(DataType::DT_FP16, a, b, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_004()
{
    PROGRAM("MATMUL_004")
    {
        Tensor a(DataType::DT_FP16, {128, 32}, "a");
        Tensor b(DataType::DT_FP16, {32, 100}, "b");
        Tensor bias(DataType::DT_FP16, {1, 100}, "bias");
        auto c = Tensor(DataType::DT_FP16, {128, 100}, "c");
        FUNCTION("MATMUL_004")
        {
            TileShape::Current().SetCubeTile({64, 64}, {32, 32}, {64, 64});
            npu::tile_fwk::Matrix::MatmulExtendParam extendParam;
            extendParam.biasTensor = bias;
            c = npu::tile_fwk::Matrix::Matmul(DataType::DT_FP16, a, b, extendParam, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_005()
{
    PROGRAM("MATMUL_005")
    {
        Tensor a(DataType::DT_FP16, {40, 130}, "a");
        Tensor b(DataType::DT_FP16, {40, 64}, "b");
        auto c = Tensor(DataType::DT_FP16, {130, 64}, "c");
        FUNCTION("MATMUL_005")
        {
            TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {64, 64});
            c = npu::tile_fwk::Matrix::Matmul(DataType::DT_FP16, a, b, true, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_006()
{
    PROGRAM("MATMUL_006")
    {
        Tensor a(DataType::DT_FP16, {5, 80, 64}, "a");
        Tensor b(DataType::DT_FP16, {5, 64, 1}, "b");
        Tensor bias(DataType::DT_FP16, {5, 1, 1}, "bias");
        auto c = Tensor(DataType::DT_FP16, {5, 80, 1}, "c");
        FUNCTION("MATMUL_006")
        {
            TileShape::Current().SetCubeTile({32, 32}, {64, 64}, {16, 16});
            c = npu::tile_fwk::Matrix::BatchMatmul(DataType::DT_FP16, a, b, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_007()
{
    PROGRAM("MATMUL_007")
    {
        Tensor a(DataType::DT_FP16, {16, 1, 64}, "a");
        Tensor b(DataType::DT_FP16, {16, 64, 64}, "b");
        auto c = Tensor(DataType::DT_FP16, {16, 1, 64}, "c");
        FUNCTION("MATMUL_007")
        {
            TileShape::Current().SetCubeTile({16, 16}, {64, 64}, {16, 16});
            c = npu::tile_fwk::Matrix::BatchMatmul(DataType::DT_FP16, a, b, false, true, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_008()
{
    PROGRAM("MATMUL_008")
    {
        Tensor a(DataType::DT_FP16, {2, 16, 129, 64}, "a");
        Tensor b(DataType::DT_FP16, {2, 16, 64, 35}, "b");
        auto c = Tensor(DataType::DT_FP16, {2, 16, 129, 35}, "c");
        FUNCTION("MATMUL_008")
        {
            TileShape::Current().SetCubeTile({128, 128}, {64, 64}, {32, 32});
            c = npu::tile_fwk::Matrix::BatchMatmul(DataType::DT_FP16, a, b, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_009()
{
    PROGRAM("MATMUL_009")
    {
        Tensor a(DataType::DT_FP16, {2, 8, 80, 160}, "a");
        Tensor b(DataType::DT_FP16, {2, 8, 160, 30}, "b");
        auto c = Tensor(DataType::DT_FP16, {2, 8, 80, 30}, "c");
        FUNCTION("MATMUL_009")
        {
            TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {32, 32});
            c = npu::tile_fwk::Matrix::BatchMatmul(DataType::DT_FP16, a, b, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_010()
{
    PROGRAM("MATMUL_010")
    {
        Tensor a(DataType::DT_FP16, {1, 4, 60, 80}, "a");
        Tensor b(DataType::DT_FP16, {1, 4, 32, 60}, "b");
        auto c = Tensor(DataType::DT_FP16, {1, 4, 80, 60}, "c");
        FUNCTION("MATMUL_010")
        {
            TileShape::Current().SetCubeTile({64, 64}, {32, 32}, {32, 32});
            c = npu::tile_fwk::Matrix::BatchMatmul(DataType::DT_FP16, a, b, true, true, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_s8s8_001()
{
    PROGRAM("MATMUL_S8S8_001")
    {
        Tensor a(DataType::DT_INT8, {16, 32}, "a");
        Tensor b(DataType::DT_INT8, {32, 16}, "b");
        auto c = Tensor(DataType::DT_INT32, {16, 16}, "c");
        FUNCTION("MATMUL_S8S8_001")
        {
            TileShape::Current().SetCubeTile({16, 16}, {32, 32}, {32, 32});
            c = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, a, b, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_S8S8_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_s8s8_002()
{
    PROGRAM("MATMUL_S8S8_002")
    {
        Tensor a(DataType::DT_INT8, {16, 32}, "a");
        Tensor b(DataType::DT_INT8, {32, 16}, "b");
        Tensor bias(DataType::DT_INT32, {1, 16}, "bias");
        auto c = Tensor(DataType::DT_INT32, {16, 16}, "c");
        FUNCTION("MATMUL_S8S8_002")
        {
            TileShape::Current().SetCubeTile({16, 16}, {32, 32}, {32, 32});
            npu::tile_fwk::Matrix::MatmulExtendParam extendParam;
            extendParam.biasTensor = bias;
            c = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, a, b, extendParam, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_S8S8_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_s8s8_003()
{
    PROGRAM("MATMUL_S8S8_003")
    {
        Tensor a(DataType::DT_INT8, {16, 32, 64}, "a");
        Tensor b(DataType::DT_INT8, {16, 64, 16}, "b");
        auto c = Tensor(DataType::DT_INT32, {16, 32, 16}, "c");
        FUNCTION("MATMUL_S8S8_003")
        {
            TileShape::Current().SetCubeTile({32, 32}, {64, 64}, {32, 32});
            c = npu::tile_fwk::Matrix::BatchMatmul(DataType::DT_INT32, a, b, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_S8S8_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_s8s8_004()
{
    PROGRAM("MATMUL_S8S8_004")
    {
        Tensor a(DataType::DT_INT8, {2, 16, 32, 64}, "a");
        Tensor b(DataType::DT_INT8, {2, 16, 64, 32}, "b");
        auto c = Tensor(DataType::DT_INT32, {2, 16, 64, 64}, "c");
        FUNCTION("MATMUL_S8S8_004")
        {
            TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
            c = npu::tile_fwk::Matrix::BatchMatmul(DataType::DT_INT32, a, b, true, true, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_S8S8_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenMatmul::test_matmul_gemv_001()
{
    PROGRAM("MATMUL_GEMV_001")
    {
        Tensor a(DataType::DT_FP16, {1, 256}, "a");
        Tensor b(DataType::DT_FP16, {256, 32}, "b");
        auto c = Tensor(DataType::DT_FP16, {1, 32}, "c");
        FUNCTION("MATMUL_GEMV_001")
        {
            TileShape::Current().SetCubeTile({16, 16}, {64, 128}, {16, 16});
            c = npu::tile_fwk::Matrix::Matmul(DataType::DT_FP16, a, b, false, false, false);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MATMUL_GEMV_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
