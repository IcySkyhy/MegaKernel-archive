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
 * \file test_codegen_dyn_reduce_line.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/function/function.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynRowReduceLine : public CodegenTestBase {
public:
    TestCodegenDynRowReduceLine() : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION}) {}
};

TEST_F(TestCodegenDynRowReduceLine, TestOperationRowSumLine)
{
    int shape0 = 6;
    int shape1 = 1;
    int shape2 = 8;
    int shape3 = 1024;
    std::vector<int64_t> shape = {shape0 * shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0 * shape1, 1, shape3};
    TileShape::Current().SetVecTile({2, 8, 512});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, outshape, "C");

    std::string funcName = "Reduce3dimMoe";
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Sum(input_a, 1, true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TRowSumLine<3>(ubTensor_2, ubTensor_0, ubTensor_3);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynRowReduceLine, TestOperationRowSumSingleTileTensor)
{
    int shape0 = 257;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};

    TileShape::Current().SetVecTile({128, 64});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, outshape, "C");

    std::string funcName = "RowSumSingle_TILETENSOR";
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Sum(input_a, -1, true);
        }
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input_a, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.001f),
    });
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TRowSumSingle<LastUse3Dim<0, 0, 0>>(ubTensor_7, ubTensor_4, ubTensor_8);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynRowReduceLine, TestOperationRowProdLine)
{
    int shape0 = 64;
    int shape1 = 32;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};

    TileShape::Current().SetVecTile({16, 16});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, outshape, "B");

    std::string funcName = "RowProdLine";
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Prod(input_a, 0, true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    const std::string expect = R"(TRowProdLine<3>(ubTensor_17, ubTensor_14);)";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynRowReduceLine, TestOperationRowProdSingleTileTensor)
{
    int shape0 = 64;
    int shape1 = 32;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};

    TileShape::Current().SetVecTile({16, 16});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, outshape, "B");

    std::string funcName = "RowProdSingle_TILETENSOR";
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Prod(input_a, -1, true);
        }
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input_a, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.001f),
    });
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    std::string res = GenCodeByFunction(*function);
    const std::string expect = R"(TRowProdSingle<LastUse3Dim<0, 0, 1>>(ubTensor_7, ubTensor_4, ubTensor_8);)";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynRowReduceLine, TestOperationArgMaxReduceAxisTile)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    int shape0 = 4;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    TileShape::Current().SetVecTile({4, 32});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMaxReduceAxisTile";
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = ArgMax(input_a, -1, true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    const std::string expect = R"(TRowArgMaxWithValueSingle(ubTensor_2, ubTensor_3, ubTensor_0, ubTensor_4);)";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynRowReduceLine, TestOperationArgMinReduceAxisTile)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    int shape0 = 4;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    TileShape::Current().SetVecTile({4, 32});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMinReduceAxisTile";
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = ArgMin(input_a, -1, true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    std::string res = GenCodeByFunction(*function);
    const std::string expect = R"(TRowArgMinWithValueSingle(ubTensor_2, ubTensor_3, ubTensor_0, ubTensor_4);)";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynRowReduceLine, TestOperationArgMaxReduceAxisTileLine)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    int shape0 = 4;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    TileShape::Current().SetVecTile({4, 32});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMaxReduceAxisTileLine";
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = ArgMax(input_a, 0, true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    const std::string expect = R"(TRowArgMaxWithValueLine<3>(ubTensor_2, ubTensor_3, ubTensor_0, ubTensor_4);)";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynRowReduceLine, TestOperationArgMinReduceAxisTileLine)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    int shape0 = 4;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    TileShape::Current().SetVecTile({4, 32});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMinReduceAxisTileLine";
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = ArgMin(input_a, 0, true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    std::string res = GenCodeByFunction(*function);
    const std::string expect = R"(TRowArgMinWithValueLine<3>(ubTensor_2, ubTensor_3, ubTensor_0, ubTensor_4);)";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
