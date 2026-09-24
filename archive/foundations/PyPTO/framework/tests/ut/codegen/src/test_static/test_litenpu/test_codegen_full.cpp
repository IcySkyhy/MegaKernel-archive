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
 * \file test_codegen_full.cpp
 * \brief
 */

#include "include/test_codegen_full.h"

using namespace npu::tile_fwk;

TestCodeGenFull::TestCodeGenFull() = default;
TestCodeGenFull::~TestCodeGenFull() = default;

TestCodeGenFull& TestCodeGenFull::Instance()
{
    static TestCodeGenFull instance;
    return instance;
}

void TestCodeGenFull::test_full_001()
{
    PROGRAM("FULL_001")
    {
        Element input(DataType::DT_FP16, 1.0);
        DataType dataType = DataType::DT_FP16;
        std::vector<int64_t> dstShape = {112};

        auto output = Tensor(DataType::DT_FP16, {112}, "output");
        FUNCTION("FULL_001")
        {
            TileShape::Current().SetVecTile({120});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_002()
{
    PROGRAM("FULL_002")
    {
        Element input(DataType::DT_FP32, 1.0f);
        DataType dataType = DataType::DT_FP32;
        std::vector<int64_t> dstShape = {100};

        auto output = Tensor(DataType::DT_FP32, {100}, "output");
        FUNCTION("FULL_002")
        {
            TileShape::Current().SetVecTile({50});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_003()
{
    PROGRAM("FULL_003")
    {
        Element input(DataType::DT_INT8, 1);
        DataType dataType = DataType::DT_INT8;
        std::vector<int64_t> dstShape = {137};

        auto output = Tensor(DataType::DT_INT8, {137}, "output");
        FUNCTION("FULL_003")
        {
            TileShape::Current().SetVecTile({136});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_004()
{
    PROGRAM("FULL_004")
    {
        Element input(DataType::DT_INT16, 1);
        DataType dataType = DataType::DT_INT16;
        std::vector<int64_t> dstShape = {4, 128};

        auto output = Tensor(DataType::DT_INT16, {4, 128}, "output");
        FUNCTION("FULL_004")
        {
            TileShape::Current().SetVecTile({8, 256});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_005()
{
    PROGRAM("FULL_005")
    {
        Element input(DataType::DT_INT32, 1);
        DataType dataType = DataType::DT_INT32;
        std::vector<int64_t> dstShape = {4, 130};

        auto output = Tensor(DataType::DT_INT32, {4, 130}, "output");
        FUNCTION("FULL_005")
        {
            TileShape::Current().SetVecTile({10, 100});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_006()
{
    PROGRAM("FULL_006")
    {
        Element input(DataType::DT_FP16, 1.0);
        DataType dataType = DataType::DT_FP16;
        std::vector<int64_t> dstShape = {15, 31};

        auto output = Tensor(DataType::DT_FP16, {15, 31}, "output");
        FUNCTION("FULL_006")
        {
            TileShape::Current().SetVecTile({5, 32});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_007()
{
    PROGRAM("FULL_007")
    {
        Element input(DataType::DT_FP32, 1.0f);
        DataType dataType = DataType::DT_FP32;
        std::vector<int64_t> dstShape = {4, 140};

        auto output = Tensor(DataType::DT_FP32, {4, 140}, "output");
        FUNCTION("FULL_007")
        {
            TileShape::Current().SetVecTile({2, 70});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_008()
{
    PROGRAM("FULL_008")
    {
        Element input(DataType::DT_INT8, 1);
        DataType dataType = DataType::DT_INT8;
        std::vector<int64_t> dstShape = {10, 5, 12};

        auto output = Tensor(DataType::DT_INT8, {10, 5, 12}, "output");
        FUNCTION("FULL_008")
        {
            TileShape::Current().SetVecTile({5, 5, 32});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_009()
{
    PROGRAM("FULL_009")
    {
        Element input(DataType::DT_INT16, 1);
        DataType dataType = DataType::DT_INT16;
        std::vector<int64_t> dstShape = {7, 3, 170};

        auto output = Tensor(DataType::DT_INT16, {7, 3, 170}, "output");
        FUNCTION("FULL_009")
        {
            TileShape::Current().SetVecTile({5, 5, 100});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_010()
{
    PROGRAM("FULL_010")
    {
        Element input(DataType::DT_INT32, 1);
        DataType dataType = DataType::DT_INT32;
        std::vector<int64_t> dstShape = {9, 8, 100};

        auto output = Tensor(DataType::DT_INT32, {9, 8, 100}, "output");
        FUNCTION("FULL_010")
        {
            TileShape::Current().SetVecTile({5, 4, 120});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_011()
{
    PROGRAM("FULL_011")
    {
        Element input(DataType::DT_FP16, 1.0);
        DataType dataType = DataType::DT_FP16;
        std::vector<int64_t> dstShape = {20, 40, 10};

        auto output = Tensor(DataType::DT_FP16, {20, 40, 10}, "output");
        FUNCTION("FULL_011")
        {
            TileShape::Current().SetVecTile({10, 10, 4});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_012()
{
    PROGRAM("FULL_012")
    {
        Element input(DataType::DT_FP32, 1.0f);
        DataType dataType = DataType::DT_FP32;
        std::vector<int64_t> dstShape = {32, 3, 5, 14};

        auto output = Tensor(DataType::DT_FP32, {32, 3, 5, 14}, "output");
        FUNCTION("FULL_012")
        {
            TileShape::Current().SetVecTile({16, 5, 5, 16});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_013()
{
    PROGRAM("FULL_013")
    {
        Element input(DataType::DT_INT8, 1);
        DataType dataType = DataType::DT_INT8;
        std::vector<int64_t> dstShape = {8, 10, 6, 16};

        auto output = Tensor(DataType::DT_INT8, {8, 10, 6, 16}, "output");
        FUNCTION("FULL_013")
        {
            TileShape::Current().SetVecTile({2, 10, 9, 8});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_014()
{
    PROGRAM("FULL_014")
    {
        Element input(DataType::DT_INT16, 1);
        DataType dataType = DataType::DT_INT16;
        std::vector<int64_t> dstShape = {6, 20, 9, 31};

        auto output = Tensor(DataType::DT_INT16, {6, 20, 9, 31}, "output");
        FUNCTION("FULL_014")
        {
            TileShape::Current().SetVecTile({3, 40, 4, 40});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_015()
{
    PROGRAM("FULL_015")
    {
        Element input(DataType::DT_INT32, 1);
        DataType dataType = DataType::DT_INT32;
        std::vector<int64_t> dstShape = {6, 9, 21, 10};

        auto output = Tensor(DataType::DT_INT32, {6, 9, 21, 10}, "output");
        FUNCTION("FULL_015")
        {
            TileShape::Current().SetVecTile({3, 3, 30, 20});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_016()
{
    PROGRAM("FULL_016")
    {
        Element input(DataType::DT_FP16, 1.0);
        DataType dataType = DataType::DT_FP16;
        std::vector<int64_t> dstShape = {6, 9, 21, 10};

        auto output = Tensor(DataType::DT_FP16, {6, 9, 21, 10}, "output");
        FUNCTION("FULL_016")
        {
            TileShape::Current().SetVecTile({5, 10, 5, 5});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_017()
{
    PROGRAM("FULL_017")
    {
        Element input(DataType::DT_FP32, 1.0f);
        DataType dataType = DataType::DT_FP32;
        std::vector<int64_t> dstShape = {6, 9, 21, 10};

        auto output = Tensor(DataType::DT_FP32, {6, 9, 21, 10}, "output");
        FUNCTION("FULL_017")
        {
            TileShape::Current().SetVecTile({3, 3, 40, 5});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_018()
{
    PROGRAM("FULL_018")
    {
        Element input(DataType::DT_INT8, 1);
        DataType dataType = DataType::DT_INT8;
        std::vector<int64_t> dstShape = {6, 9, 21, 10};

        auto output = Tensor(DataType::DT_INT8, {6, 9, 21, 10}, "output");
        FUNCTION("FULL_018")
        {
            TileShape::Current().SetVecTile({5, 5, 12, 20});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenFull::test_full_019()
{
    PROGRAM("FULL_019")
    {
        Element input(DataType::DT_INT16, 1);
        DataType dataType = DataType::DT_INT16;
        std::vector<int64_t> dstShape = {6, 9, 21, 10};

        auto output = Tensor(DataType::DT_INT16, {6, 9, 21, 10}, "output");
        FUNCTION("FULL_019")
        {
            TileShape::Current().SetVecTile({5, 8, 12, 5});
            output = Full(input, dataType, dstShape);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "FULL_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
