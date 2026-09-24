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
 * \file test_codegen_index_put_.cpp
 * \brief
 */

#include "include/test_codegen_index_put.h"

using namespace npu::tile_fwk;

TestCodeGenIndexPut::TestCodeGenIndexPut() = default;
TestCodeGenIndexPut::~TestCodeGenIndexPut() = default;

TestCodeGenIndexPut& TestCodeGenIndexPut::Instance()
{
    static TestCodeGenIndexPut instance;
    return instance;
}

void TestCodeGenIndexPut::test_index_put_001()
{
    PROGRAM("INDEX_PUT_001")
    {
        Tensor self(DataType::DT_FP16, {60}, "self");
        Tensor values(DataType::DT_FP16, {4}, "values");
        Tensor indice(DataType::DT_INT32, {4}, "indice");
        std::vector<Tensor> indices = {indice};
        bool accumulate = false;

        FUNCTION("INDEX_PUT_001")
        {
            TileShape::Current().SetVecTile({2});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenIndexPut::test_index_put_002()
{
    PROGRAM("INDEX_PUT_002")
    {
        Tensor self(DataType::DT_FP32, {3, 3}, "self");
        Tensor values(DataType::DT_FP32, {2, 3}, "values");
        Tensor indice(DataType::DT_INT64, {2}, "indice");
        std::vector<Tensor> indices = {indice};
        bool accumulate = true;

        FUNCTION("INDEX_PUT_002")
        {
            TileShape::Current().SetVecTile({3});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenIndexPut::test_index_put_003()
{
    PROGRAM("INDEX_PUT_003")
    {
        Tensor self(DataType::DT_INT8, {3, 3}, "self");
        Tensor values(DataType::DT_INT8, {2}, "values");
        std::vector<Tensor> indices;
        Tensor indice0(DataType::DT_INT8, {2}, "indice0");
        Tensor indice1(DataType::DT_INT8, {2}, "indice1");
        indices.push_back(indice0);
        indices.push_back(indice1);
        bool accumulate = false;

        FUNCTION("INDEX_PUT_003")
        {
            TileShape::Current().SetVecTile({3});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenIndexPut::test_index_put_004()
{
    PROGRAM("INDEX_PUT_004")
    {
        Tensor self(DataType::DT_INT16, {64, 128}, "self");
        Tensor values(DataType::DT_INT16, {32, 128}, "values");
        Tensor indice(DataType::DT_UINT8, {32}, "indice");
        std::vector<Tensor> indices = {indice};
        bool accumulate = true;

        FUNCTION("INDEX_PUT_004")
        {
            TileShape::Current().SetVecTile({64});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenIndexPut::test_index_put_005()
{
    PROGRAM("INDEX_PUT_005")
    {
        Tensor self(DataType::DT_INT8, {64, 140}, "self");
        Tensor values(DataType::DT_INT8, {20, 140}, "values");
        Tensor indice(DataType::DT_INT16, {20}, "indice");
        std::vector<Tensor> indices = {indice};
        bool accumulate = false;

        FUNCTION("INDEX_PUT_005")
        {
            TileShape::Current().SetVecTile({80});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenIndexPut::test_index_put_006()
{
    PROGRAM("INDEX_PUT_006")
    {
        Tensor self(DataType::DT_INT32, {16, 32, 120}, "self");
        Tensor values(DataType::DT_INT32, {8, 32, 120}, "values");
        Tensor indice(DataType::DT_UINT16, {8}, "indice");
        std::vector<Tensor> indices = {indice};
        bool accumulate = false;

        FUNCTION("INDEX_PUT_006")
        {
            TileShape::Current().SetVecTile({100});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenIndexPut::test_index_put_007()
{
    PROGRAM("INDEX_PUT_007")
    {
        Tensor self(DataType::DT_INT16, {16, 32, 120}, "self");
        Tensor values(DataType::DT_INT16, {8, 120}, "values");
        std::vector<Tensor> indices;
        Tensor indice0(DataType::DT_UINT32, {8}, "indice0");
        Tensor indice1(DataType::DT_UINT32, {8}, "indice1");
        indices.push_back(indice0);
        indices.push_back(indice1);
        bool accumulate = true;

        FUNCTION("INDEX_PUT_007")
        {
            TileShape::Current().SetVecTile({64});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenIndexPut::test_index_put_008()
{
    PROGRAM("INDEX_PUT_008")
    {
        Tensor self(DataType::DT_INT8, {16, 32, 120}, "self");
        Tensor values(DataType::DT_INT8, {10}, "values");
        std::vector<Tensor> indices;
        Tensor indice0(DataType::DT_UINT32, {10}, "indice0");
        Tensor indice1(DataType::DT_UINT32, {10}, "indice1");
        Tensor indice2(DataType::DT_UINT32, {10}, "indice2");
        indices.push_back(indice0);
        indices.push_back(indice1);
        indices.push_back(indice2);
        bool accumulate = false;

        FUNCTION("INDEX_PUT_008")
        {
            TileShape::Current().SetVecTile({10});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenIndexPut::test_index_put_009()
{
    PROGRAM("INDEX_PUT_009")
    {
        Tensor self(DataType::DT_FP16, {10, 20, 16, 112}, "self");
        Tensor values(DataType::DT_FP16, {2, 20, 16, 112}, "values");
        Tensor indice(DataType::DT_INT32, {2}, "indice");
        std::vector<Tensor> indices = {indice};
        bool accumulate = true;

        FUNCTION("INDEX_PUT_009")
        {
            TileShape::Current().SetVecTile({1});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenIndexPut::test_index_put_010()
{
    PROGRAM("INDEX_PUT_010")
    {
        Tensor self(DataType::DT_FP32, {10, 20, 16, 112}, "self");
        Tensor values(DataType::DT_FP32, {5, 16, 112}, "values");
        std::vector<Tensor> indices;
        Tensor indice0(DataType::DT_INT32, {5}, "indice0");
        Tensor indice1(DataType::DT_INT32, {5}, "indice1");
        indices.push_back(indice0);
        indices.push_back(indice1);
        bool accumulate = false;

        FUNCTION("INDEX_PUT_010")
        {
            TileShape::Current().SetVecTile({70});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenIndexPut::test_index_put_011()
{
    PROGRAM("INDEX_PUT_011")
    {
        Tensor self(DataType::DT_FP16, {10, 20, 16, 112}, "self");
        Tensor values(DataType::DT_FP16, {5, 112}, "values");
        std::vector<Tensor> indices;
        Tensor indice0(DataType::DT_UINT32, {5}, "indice0");
        Tensor indice1(DataType::DT_UINT32, {5}, "indice1");
        Tensor indice2(DataType::DT_UINT32, {5}, "indice2");
        indices.push_back(indice0);
        indices.push_back(indice1);
        indices.push_back(indice2);
        bool accumulate = true;

        FUNCTION("INDEX_PUT_011")
        {
            TileShape::Current().SetVecTile({80});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

void TestCodeGenIndexPut::test_index_put_012()
{
    PROGRAM("INDEX_PUT_012")
    {
        Tensor self(DataType::DT_FP16, {10, 20, 16, 112}, "self");
        Tensor values(DataType::DT_FP16, {5}, "values");
        std::vector<Tensor> indices;
        Tensor indice0(DataType::DT_INT32, {5}, "indice0");
        Tensor indice1(DataType::DT_INT32, {5}, "indice1");
        Tensor indice2(DataType::DT_INT32, {5}, "indice2");
        Tensor indice3(DataType::DT_INT32, {5}, "indice3");
        indices.push_back(indice0);
        indices.push_back(indice1);
        indices.push_back(indice2);
        indices.push_back(indice3);
        bool accumulate = false;

        FUNCTION("INDEX_PUT_012")
        {
            TileShape::Current().SetVecTile({32});
            IndexPut_(self, indices, values, accumulate);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "INDEX_PUT_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function);
}
