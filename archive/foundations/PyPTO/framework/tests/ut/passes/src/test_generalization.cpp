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
 * \file test_generalization.cpp
 * \brief Unit test for all pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class GeneralizetionTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        dynFunc = std::make_shared<Function>(Program::GetInstance(), "DYN_0", "DYN",
                                             Program::GetInstance().GetCurrentFunction());
        Program::GetInstance().SetCurrentDynamicFunction(dynFunc.get());
    }
    void TearDown() override { Program::GetInstance().SetCurrentDynamicFunction(nullptr); }

    std::shared_ptr<Function> dynFunc;
};

// =======================================================  Single OP Test
// ====================================================================
TEST_F(GeneralizetionTest, TestReshape)
{
    TileShape::Current().SetVecTile({64, 64});
    std::vector<int64_t> shape1{256, 256};
    std::vector<int64_t> shape2{1, 128, 512};

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor1(DT_FP32, shape2, "out_tensor");

    FUNCTION("A") { out_tensor1 = Reshape(in_tensor, {1, 128, 512}); }
}

TEST_F(GeneralizetionTest, TestAssemble)
{
    int N = 2;
    int T = 8;
    std::vector<int64_t> shape{T, T};
    TileShape::Current().SetVecTile({T, T});

    Tensor inputA(DT_FP32, shape, "a");
    Tensor result(DT_FP32, {2 * T, 2 * T}, "result");

    FUNCTION("A")
    {
        std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                aggregation.emplace_back(inputA, std::vector<int64_t>{i * T, j * T});
            }
        auto gatherResult = Assemble(aggregation); // 2*T,2*T
        // 直接赋值报错，需要接一个vector算子
        result = Abs(gatherResult);
    }
}

TEST_F(GeneralizetionTest, TestView)
{
    TileShape::Current().SetVecTile({64, 64});
    std::vector<int64_t> shape1{256, 256};
    std::vector<int64_t> shape2{128, 128};

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor(DT_FP32, shape2, "out_tensor");

    FUNCTION("A")
    {
        auto tmp = View(in_tensor, {128, 128}, {0, 0});
        // 不加abs会报错
        out_tensor = Abs(tmp);
    }
}

TEST_F(GeneralizetionTest, TestScatterUpdate)
{
    TileShape::Current().SetVecTile({16, 32});

    int h = 128, minusTwo = -2;
    Tensor output(DT_INT32, {h, h}, "output");
    Tensor idxs(DT_INT32, {h, h}, "idxs");
    Tensor keyStates(DT_INT32, {h, h}, "keyStates");

    FUNCTION("A") { output = ScatterUpdate(output, idxs, keyStates, minusTwo); }
}

TEST_F(GeneralizetionTest, TestTranspose)
{
    config::GetPassGlobalConfig(KEY_PASS_THREAD_NUM, 2);
    // 若是二维，则报错
    int b = 4;
    int n = 1;
    int s = 32;
    int d = 437;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};

    TileShape::Current().SetVecTile(2, 1, 32, 512);

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, resShape, "res");
    std::string funcName = "DATAMOVE";
    FUNCTION("A") { output = Transpose(input, {1, 2}); }
}

// =======================================================  Same OP Test
// ====================================================================
TEST_F(GeneralizetionTest, TestReshapeReshape)
{
    config::GetPassGlobalConfig(KEY_PASS_THREAD_NUM, 2);
    TileShape::Current().SetVecTile({64, 64});
    std::vector<int64_t> shape1{256, 256};

    std::vector<int64_t> shape4{128, 512};
    std::vector<int64_t> shape5{512, 128};

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor in_tensor1(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor1(DT_FP32, shape5, "out_tensor");

    FUNCTION("B")
    {
        auto tmp = Sub(in_tensor, in_tensor1);
        // 只有reshape会报错
        auto out_tensor = Reshape(tmp, shape4);
        auto tmp1 = Reshape(out_tensor, shape5);
        out_tensor1 = Abs(tmp1);
    }
}

TEST_F(GeneralizetionTest, TestAssembleAssemble)
{
    int N = 2;
    int T = 8;
    std::vector<int64_t> shape{T, T};
    TileShape::Current().SetVecTile({T, T});

    Tensor inputA(DT_FP32, shape, "a");
    Tensor result(DT_FP32, {4 * T, 4 * T}, "result");

    FUNCTION("B")
    {
        std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                aggregation.emplace_back(inputA, std::vector<int64_t>{i * T, j * T});
            }
        auto gatherResult = Assemble(aggregation); // 2*T,2*T

        std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation1;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                aggregation1.emplace_back(gatherResult, std::vector<int64_t>{i * 2 * T, j * 2 * T});
            }
        // 后面接一个abs通过,直接2个Assemble不通过
        auto gatherResult1 = Assemble(aggregation1); // 4*T, 4*T
        result = Abs(gatherResult1);
    }
}

TEST_F(GeneralizetionTest, TestViewView)
{
    std::vector<int64_t> shape1{256, 256};
    std::vector<int64_t> shape2{128, 128};
    std::vector<int64_t> shape3{64, 128};

    std::vector<int64_t> shape4{1, 256, 1, 256};
    std::vector<int64_t> shape5{128, 2, 2, 128};

    TileShape::Current().SetVecTile({64, 64});

    // sub
    Tensor input(DT_FP32, shape1, "input");
    Tensor input1(DT_FP32, shape1, "input");
    // view
    Tensor input_v1(DT_FP32, shape3, "input_b");
    Tensor output_v(DT_FP32, shape3, "output");

    FUNCTION("B")
    {
        auto tmp = Sub(input, input1);
        // 只有view会报错
        auto tmp_v1 = View(tmp, shape2, {0, 0});
        auto tmp_v2 = View(tmp_v1, shape3, {0, 0});
        output_v = Add(tmp_v2, input_v1);
    }
}

TEST_F(GeneralizetionTest, TestScatterUpdateScatterUpdate)
{
    TileShape::Current().SetVecTile({16, 32});

    int h = 128, minusTwo = -2, minusOne = -1;
    Tensor output(DT_INT32, {h, h}, "output");
    Tensor idxs(DT_INT32, {h, h}, "idxs");
    Tensor keyStates(DT_INT32, {h, h}, "keyStates");

    FUNCTION("B")
    {
        output = ScatterUpdate(output, idxs, keyStates, minusTwo);
        output = ScatterUpdate(output, idxs, keyStates, minusOne);
    }
}

TEST_F(GeneralizetionTest, TestTransposeTranspose)
{
    int b = 4;
    int n = 1;
    int s = 32;
    int d = 437;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    std::vector<int64_t> resShape2{s, b, n, d};

    TileShape::Current().SetVecTile(2, 1, 32, 512);

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, resShape2, "res");

    FUNCTION("B")
    {
        auto tmp = Transpose(input, {1, 2});
        output = Transpose(tmp, {0, 2});
    }
}

// =======================================================  Different OP Test
// ====================================================================
TEST_F(GeneralizetionTest, TestReshapeToAll)
{
    int N = 2;
    int T = 64;
    TileShape::Current().SetVecTile({T, T});
    std::vector<int64_t> shape1{64, 1024};
    std::vector<int64_t> shape2{256, 256};

    std::vector<int64_t> shape3{128, 512};
    std::vector<int64_t> shape4{512, 512};
    std::vector<int64_t> shape5{128, 128};
    std::vector<int64_t> shape6{512, 128};

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor1(DT_FP32, shape3, "out_tensor1");
    Tensor out_tensor2(DT_FP32, shape4, "out_tensor2");
    Tensor out_tensor3(DT_FP32, {64, 64}, "out_tensor3");
    Tensor out_tensor4(DT_FP32, shape6, "out_tensor4");

    Tensor idxs(DT_INT32, {256, 256}, "idxs");
    Tensor keyStates(DT_INT32, {256, 256}, "keyStates");
    int minusTwo = -2;

    FUNCTION("C")
    {
        auto tmp = Reshape(in_tensor, shape2); // 256,256
        // to reshape
        out_tensor1 = Reshape(tmp, shape3);

        // to assemble
        std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                aggregation.emplace_back(tmp, std::vector<int64_t>{i * 256, j * 256});
            }
        out_tensor2 = Assemble(aggregation); // 512,512

        // to view 报错
        // to transpose 报错
        // to scatter_update
        tmp = ScatterUpdate(tmp, idxs, keyStates, minusTwo); // 256,256
    }
}

TEST_F(GeneralizetionTest, TestAssembleToAll)
{
    int N = 2;
    int T = 64;
    std::vector<int64_t> shape{T, T};
    TileShape::Current().SetVecTile({T, T});

    Tensor inputA(DT_FP32, shape, "a");
    Tensor result(DT_FP32, {4 * T, 4 * T}, "result");

    Tensor result1(DT_FP32, {256, 64}, "result1");
    Tensor result2(DT_FP32, {64, 64}, "result2");
    Tensor result3(DT_FP32, {128, 128}, "result3");

    Tensor idxs(DT_INT32, {128, 128}, "idxs");
    Tensor keyStates(DT_INT32, {128, 128}, "keyStates");
    int minusTwo = -2;

    FUNCTION("C")
    {
        std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                aggregation.emplace_back(inputA, std::vector<int64_t>{i * T, j * T});
            }
        auto gatherResult = Assemble(aggregation); // 128,128

        // to assemble
        std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation1;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                aggregation1.emplace_back(gatherResult, std::vector<int64_t>{i * 2 * T, j * 2 * T});
            }
        auto gatherResult1 = Assemble(aggregation1); // 256,256
        result = Abs(gatherResult1);

        // to reshape
        auto tmp = Reshape(gatherResult, {256, 64}); // 256,64
        result1 = Abs(tmp);

        // to view
        auto tmp1 = View(gatherResult, {64, 64}, {0, 0}); // 64,64
        result2 = Abs(tmp1);

        // to transpose
        auto tmp2 = Transpose(gatherResult, {1, 0}); // 128,128
        result3 = Abs(tmp2);

        // to scatter_update
        gatherResult = ScatterUpdate(gatherResult, idxs, keyStates, minusTwo); // 128,128
    }
}

TEST_F(GeneralizetionTest, TestViewToAll)
{
    int N = 2;
    int T = 64;
    std::vector<int64_t> shape{256, 256};
    TileShape::Current().SetVecTile({T, T});

    Tensor inputA(DT_FP32, shape, "a");
    Tensor result(DT_FP32, {4 * T, 4 * T}, "result");

    Tensor result1(DT_FP32, {256, 64}, "result1");
    Tensor result2(DT_FP32, {64, 64}, "result2");
    Tensor result3(DT_FP32, {128, 128}, "result3");

    Tensor idxs(DT_INT32, {128, 128}, "idxs");
    Tensor keyStates(DT_INT32, {128, 128}, "keyStates");
    int minusTwo = -2;

    FUNCTION("C")
    {
        auto viewResult = View(inputA, {128, 128}, {0, 0}); // 128,128

        // to assemble
        std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation1;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                aggregation1.emplace_back(viewResult, std::vector<int64_t>{i * 2 * T, j * 2 * T});
            }
        auto gatherResult1 = Assemble(aggregation1); // 256,256
        result = Abs(gatherResult1);

        // to reshape
        auto tmp = Reshape(viewResult, {256, 64}); // 256,64
        result1 = Abs(tmp);

        // to view
        auto tmp1 = View(viewResult, {64, 64}, {0, 0}); // 64,64
        result2 = Abs(tmp1);

        // to transpose
        auto tmp2 = Transpose(viewResult, {1, 0}); // 128,128
        result3 = Abs(tmp2);

        // to scatter_update
        viewResult = ScatterUpdate(viewResult, idxs, keyStates, minusTwo); // 128,128
    }
}

TEST_F(GeneralizetionTest, TestScatterUpdateToAll)
{
    int N = 2;
    int T = 64;
    std::vector<int64_t> shape{128, 128};
    TileShape::Current().SetVecTile({T, T});

    Tensor inputA(DT_FP32, shape, "a");
    Tensor result(DT_FP32, {2 * T, 2 * T}, "result");

    Tensor result1(DT_FP32, {256, 64}, "result1");
    Tensor result2(DT_FP32, {64, 64}, "result2");
    Tensor result3(DT_FP32, {128, 128}, "result3");

    Tensor idxs(DT_INT32, {128, 128}, "idxs");
    Tensor keyStates(DT_INT32, {128, 128}, "keyStates");
    int minusTwo = -2, minusOne = -1;

    FUNCTION("C")
    {
        inputA = ScatterUpdate(inputA, idxs, keyStates, minusOne); // 128,128

        // to assemble
        std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation1;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                aggregation1.emplace_back(inputA, std::vector<int64_t>{i * 2 * T, j * 2 * T});
            }
        auto gatherResult1 = Assemble(aggregation1); // 256,256
        result = Abs(gatherResult1);

        // to reshape
        auto tmp = Reshape(inputA, {256, 64}); // 256,64
        result1 = Abs(tmp);

        // to view
        auto tmp1 = View(inputA, {64, 64}, {0, 0}); // 64,64
        result2 = Abs(tmp1);

        // to transpose
        auto tmp2 = Transpose(inputA, {1, 0}); // 128,128
        result3 = Abs(tmp2);

        // to scatter_update
        inputA = ScatterUpdate(inputA, idxs, keyStates, minusTwo); // 128,128
    }
}

TEST_F(GeneralizetionTest, TestTransposeToAll)
{
    int N = 2;
    int T = 64;
    std::vector<int64_t> shape{128, 128};
    TileShape::Current().SetVecTile({T, T});

    Tensor inputA(DT_FP32, shape, "a");
    Tensor result(DT_FP32, {2 * T, 2 * T}, "result");

    Tensor result1(DT_FP32, {256, 64}, "result1");
    Tensor result2(DT_FP32, {64, 64}, "result2");
    Tensor result3(DT_FP32, {128, 128}, "result3");

    Tensor idxs(DT_INT32, {128, 128}, "idxs");
    Tensor keyStates(DT_INT32, {128, 128}, "keyStates");
    int minusTwo = -2;

    FUNCTION("C")
    {
        auto transposeResult = Transpose(inputA, {1, 0}); // 128,128

        // to assemble
        std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation1;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                aggregation1.emplace_back(transposeResult, std::vector<int64_t>{i * 2 * T, j * 2 * T});
            }
        auto gatherResult1 = Assemble(aggregation1); // 256,256
        result = Abs(gatherResult1);

        // to reshape
        auto tmp = Reshape(transposeResult, {256, 64}); // 256,64
        result1 = Abs(tmp);

        // to view
        auto tmp1 = View(transposeResult, {64, 64}, {0, 0}); // 64,64
        result2 = Abs(tmp1);

        // to transpose
        auto tmp2 = Transpose(transposeResult, {1, 0}); // 128,128
        result3 = Abs(tmp2);

        // to scatter_update
        transposeResult = ScatterUpdate(transposeResult, idxs, keyStates, minusTwo); // 128,128
    }
}
