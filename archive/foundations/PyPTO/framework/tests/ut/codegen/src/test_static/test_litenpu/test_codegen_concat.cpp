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
 * \file test_codegen_concat.cpp
 * \brief
 */

#include "include/test_codegen_concat.h"

using namespace npu::tile_fwk;

TestCodeGenConcat::TestCodeGenConcat() = default;
TestCodeGenConcat::~TestCodeGenConcat() = default;

TestCodeGenConcat& TestCodeGenConcat::Instance()
{
    static TestCodeGenConcat instance;
    return instance;
}

void TestCodeGenConcat::test_concat_001()
{
    PROGRAM("CONCAT_001")
    {
        Tensor input0(DataType::DT_INT16, {4, 128}, "input0");
        Tensor input1(DataType::DT_INT16, {7, 128}, "input1");
        std::vector<Tensor> inputs = {input0, input1};
        int32_t axis = 0; // 第一维不同
        auto output = Tensor(DataType::DT_INT16, {11, 128}, "output");
        FUNCTION("CONCAT_001")
        {
            TileShape::Current().SetVecTile({8, 256});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_002()
{
    PROGRAM("CONCAT_002")
    {
        Tensor input0(DataType::DT_INT32, {4, 130}, "input0");
        Tensor input1(DataType::DT_INT32, {4, 90}, "input1");
        std::vector<Tensor> inputs = {input0, input1};
        int32_t axis = 1; // 第二维不同
        auto output = Tensor(DataType::DT_INT32, {4, 220}, "output");
        FUNCTION("CONCAT_002")
        {
            TileShape::Current().SetVecTile({10, 200});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_003()
{
    PROGRAM("CONCAT_003")
    {
        Tensor input0(DataType::DT_FP16, {15, 31}, "input0");
        Tensor input1(DataType::DT_FP16, {20, 31}, "input1");
        Tensor input2(DataType::DT_FP16, {16, 31}, "input2");
        std::vector<Tensor> inputs = {input0, input1, input2};
        int32_t axis = 0; // 第一维不同
        auto output = Tensor(DataType::DT_FP16, {51, 31}, "output");
        FUNCTION("CONCAT_003")
        {
            TileShape::Current().SetVecTile({5, 32});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_004()
{
    PROGRAM("CONCAT_004")
    {
        Tensor input0(DataType::DT_FP32, {4, 140}, "input0");
        Tensor input1(DataType::DT_FP32, {4, 23}, "input1");
        Tensor input2(DataType::DT_FP32, {4, 4}, "input2");
        std::vector<Tensor> inputs = {input0, input1, input2};
        int32_t axis = 1; // 第二维不同
        auto output = Tensor(DataType::DT_FP32, {4, 167}, "output");
        FUNCTION("CONCAT_004")
        {
            TileShape::Current().SetVecTile({2, 280});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_005()
{
    PROGRAM("CONCAT_005")
    {
        Tensor input0(DataType::DT_INT8, {10, 5, 12}, "input0");
        Tensor input1(DataType::DT_INT8, {5, 5, 12}, "input1");
        std::vector<Tensor> inputs = {input0, input1};
        int32_t axis = 0; // 第一维不同
        auto output = Tensor(DataType::DT_INT8, {15, 5, 12}, "output");
        FUNCTION("CONCAT_005")
        {
            TileShape::Current().SetVecTile({5, 5, 32});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_006()
{
    PROGRAM("CONCAT_006")
    {
        Tensor input0(DataType::DT_INT16, {7, 3, 170}, "input0");
        Tensor input1(DataType::DT_INT16, {7, 20, 170}, "input1");
        std::vector<Tensor> inputs = {input0, input1};
        int32_t axis = 1; // 第二维不同
        auto output = Tensor(DataType::DT_INT16, {7, 23, 170}, "output");
        FUNCTION("CONCAT_006")
        {
            TileShape::Current().SetVecTile({5, 5, 400});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_007()
{
    PROGRAM("CONCAT_007")
    {
        Tensor input0(DataType::DT_INT32, {9, 8, 100}, "input0");
        Tensor input1(DataType::DT_INT32, {9, 8, 40}, "input1");
        std::vector<Tensor> inputs = {input0, input1};
        int32_t axis = 2; // 第三维不同
        auto output = Tensor(DataType::DT_INT32, {9, 8, 140}, "output");
        FUNCTION("CONCAT_007")
        {
            TileShape::Current().SetVecTile({5, 4, 120});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_008()
{
    PROGRAM("CONCAT_008")
    {
        Tensor input0(DataType::DT_FP16, {20, 40, 10}, "input0");
        Tensor input1(DataType::DT_FP16, {9, 40, 10}, "input1");
        Tensor input2(DataType::DT_FP16, {12, 40, 10}, "input2");
        std::vector<Tensor> inputs = {input0, input1, input2};
        int32_t axis = 0; // 第一维不同
        auto output = Tensor(DataType::DT_FP16, {41, 40, 10}, "output");
        FUNCTION("CONCAT_008")
        {
            TileShape::Current().SetVecTile({10, 10, 16});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_009()
{
    PROGRAM("CONCAT_009")
    {
        Tensor input0(DataType::DT_FP32, {32, 3, 5, 14}, "input0");
        Tensor input1(DataType::DT_FP32, {21, 3, 5, 14}, "input1");
        std::vector<Tensor> inputs = {input0, input1};
        int32_t axis = 0; // 第一维不同
        auto output = Tensor(DataType::DT_FP32, {53, 3, 5, 14}, "output");
        FUNCTION("CONCAT_009")
        {
            TileShape::Current().SetVecTile({16, 5, 5, 16});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_010()
{
    PROGRAM("CONCAT_010")
    {
        Tensor input0(DataType::DT_INT8, {8, 10, 6, 16}, "input0");
        Tensor input1(DataType::DT_INT8, {8, 4, 6, 16}, "input1");
        std::vector<Tensor> inputs = {input0, input1};
        int32_t axis = 1; // 第二维不同
        auto output = Tensor(DataType::DT_INT8, {8, 14, 6, 16}, "output");
        FUNCTION("CONCAT_010")
        {
            TileShape::Current().SetVecTile({2, 10, 9, 32});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_011()
{
    PROGRAM("CONCAT_011")
    {
        Tensor input0(DataType::DT_INT16, {6, 20, 9, 31}, "input0");
        Tensor input1(DataType::DT_INT16, {6, 20, 23, 31}, "input1");
        std::vector<Tensor> inputs = {input0, input1};
        int32_t axis = 2; // 第三维不同
        auto output = Tensor(DataType::DT_INT16, {6, 20, 32, 31}, "output");
        FUNCTION("CONCAT_011")
        {
            TileShape::Current().SetVecTile({3, 40, 4, 16});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_012()
{
    PROGRAM("CONCAT_012")
    {
        Tensor input0(DataType::DT_INT32, {6, 9, 21, 10}, "input0");
        Tensor input1(DataType::DT_INT32, {6, 9, 21, 13}, "input1");
        std::vector<Tensor> inputs = {input0, input1};
        int32_t axis = 3; // 第四维不同
        auto output = Tensor(DataType::DT_INT32, {6, 9, 21, 23}, "output");
        FUNCTION("CONCAT_012")
        {
            TileShape::Current().SetVecTile({3, 3, 30, 8});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_013()
{
    PROGRAM("CONCAT_013")
    {
        Tensor input0(DataType::DT_FP16, {6, 9, 21, 10}, "input0");
        Tensor input1(DataType::DT_FP16, {9, 9, 21, 10}, "input1");
        Tensor input2(DataType::DT_FP16, {17, 9, 21, 10}, "input2");
        std::vector<Tensor> inputs = {input0, input1, input2};
        int32_t axis = 0; // 第一维不同
        auto output = Tensor(DataType::DT_FP16, {32, 9, 21, 10}, "output");
        FUNCTION("CONCAT_013")
        {
            TileShape::Current().SetVecTile({5, 10, 5, 16});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_014()
{
    PROGRAM("CONCAT_014")
    {
        Tensor input0(DataType::DT_FP32, {6, 9, 21, 10}, "input0");
        Tensor input1(DataType::DT_FP32, {6, 21, 21, 10}, "input1");
        Tensor input2(DataType::DT_FP32, {6, 16, 21, 10}, "input2");
        std::vector<Tensor> inputs = {input0, input1, input2};
        int32_t axis = 1; // 第二维不同
        auto output = Tensor(DataType::DT_FP32, {6, 46, 21, 10}, "output");
        FUNCTION("CONCAT_014")
        {
            TileShape::Current().SetVecTile({3, 3, 40, 8});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_015()
{
    PROGRAM("CONCAT_015")
    {
        Tensor input0(DataType::DT_INT8, {6, 9, 21, 10}, "input0");
        Tensor input1(DataType::DT_INT8, {6, 9, 14, 10}, "input1");
        Tensor input2(DataType::DT_INT8, {6, 9, 19, 10}, "input2");
        std::vector<Tensor> inputs = {input0, input1, input2};
        int32_t axis = 2; // 第三维不同
        auto output = Tensor(DataType::DT_INT8, {6, 9, 54, 10}, "output");
        FUNCTION("CONCAT_015")
        {
            TileShape::Current().SetVecTile({5, 5, 12, 32});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenConcat::test_concat_016()
{
    PROGRAM("CONCAT_016")
    {
        Tensor input0(DataType::DT_INT16, {6, 9, 21, 10}, "input0");
        Tensor input1(DataType::DT_INT16, {6, 9, 21, 40}, "input1");
        Tensor input2(DataType::DT_INT16, {6, 9, 21, 21}, "input2");
        Tensor input3(DataType::DT_INT16, {6, 9, 21, 9}, "input3");
        std::vector<Tensor> inputs = {input0, input1, input2, input3};
        int32_t axis = 3; // 第四维不同
        auto output = Tensor(DataType::DT_INT16, {6, 9, 21, 80}, "output");
        FUNCTION("CONCAT_016")
        {
            TileShape::Current().SetVecTile({5, 8, 12, 16});
            output = Cat(inputs, axis);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "CONCAT_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
