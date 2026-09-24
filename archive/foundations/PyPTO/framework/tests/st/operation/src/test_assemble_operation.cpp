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
 * \file test_assemble_operation.cpp
 * \brief
 */

#include <random>
#include "interface/configs/config_manager.h"
#include "test_operation.h"
#include "tilefwk/function.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tile_shape.h"
#include "tilefwk/tilefwk_op.h"

using namespace tile_fwk::test_operation;
namespace {
using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class AssembleTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
    void SetUp() override
    {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        // 测试精度工具功能支持时，打开下面的注释
        // config::SetVerifyOption(KEY_VERIFY_TENSOR_GRAPH, true);
        // config::SetVerifyOption(KEY_VERIFY_PASS, true);
        // config::SetVerifyOption(KEY_VERIFY_EXECUTE_GRAPH, true);
        // config::SetVerifyOption(KEY_VERIFY_CHECK_PRECISION, true);
    }

    void TearDown() override {}
};

template <typename T>
auto ShapeSize(const std::vector<T>& shapes)
{
    T res = 1;
    for (auto v : shapes) {
        res *= v;
    }
    return res;
}

template <bool parallel>
void TestAssembleBasic()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    int row = 128;
    auto SimuResult = [&row](const std::vector<float>& a) {
        constexpr int SHAPE0 = 128;
        constexpr int SHAPE1_IN = 512;
        constexpr int SHAPE1_OUT = 8192;
        ASSERT(a.size() == SHAPE0 * SHAPE1_IN);
        std::vector<float> out(SHAPE0 * SHAPE1_OUT, 0);
        std::vector<int> shape1{1, 256};
        std::vector<int> shape2{1, 128};
        std::vector<int> shape3{1, 128};
        for (int i = 0; i < row; i++) {
            std::vector<int> offsets1{i, 0};
            std::vector<int> offsets2{i, 256};
            std::vector<int> offsets3{i, 384};
            for (int j = 0; j < shape1[1]; j++) {
                out[i * SHAPE1_OUT + j] = a[i * SHAPE1_IN + offsets1[1] + j] + 1.14f;
            }
            for (int j = 0; j < shape3[1]; j++) {
                out[i * SHAPE1_OUT + shape1[1] + j] = a[i * SHAPE1_IN + offsets3[1] + j] - 2.33f;
            }
            for (int j = 0; j < shape2[1]; j++) {
                out[i * SHAPE1_OUT + shape1[1] + shape3[1] + j] = a[i * SHAPE1_IN + offsets2[1] + j] * 5.14f;
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {128, 512}, "a");
    Tensor dst(DT_FP32, {128, 8192}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row))
        {
            TileShape::Current().SetVecTile(8, 64);
            Shape shape1{1, 256};
            Shape shape2{1, 128};
            Shape shape3{1, 128};
            std::vector<SymbolicScalar> offsets1{idx, 0};
            std::vector<SymbolicScalar> offsets2{idx, 256};
            std::vector<SymbolicScalar> offsets3{idx, 384};
            auto tmp1 = View(a, shape1, offsets1);
            tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
            auto tmp2 = View(a, shape2, offsets2);
            tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
            auto tmp3 = View(a, shape3, offsets3);
            tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
            std::vector<AssembleItem> items{{tmp1, offsets1}, {tmp3, offsets2}, {tmp2, offsets3}};
            Assemble(items, dst, parallel);
        }
    }
    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
    for (int i = 0; i < ShapeSize(dst.GetShape()); i++) {
        auto actual = ((float*)dstResult->data())[i];
        auto expect = dstGolden[i];
        if (fabs(actual - expect) > eps) {
            std::cout << i << ": actual: " << actual << ", expect: " << expect << std::endl;
        }
    }
}

// 测试单个assemble内串行连接的逻辑正确性
TEST_F(AssembleTest, test_seq_in_assemble) { TestAssembleBasic<false>(); }

// 测试单个assemble内并行连接的逻辑正确性
TEST_F(AssembleTest, test_parallel_in_assemble) { TestAssembleBasic<true>(); }

template <bool parallel>
void TestAssembleOverride()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int N = 32;
    constexpr int M = 48;
    constexpr int T = 1024;
    constexpr int SHIFT = 20;

    auto SimuResult = [](const std::vector<float>& a) {
        ASSERT(a.size() == N * M);
        std::vector<float> out(T, 0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 16; j++) {
                out[i * SHIFT + j] = a[i * M + j] + 1.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * SHIFT + 16 + j] = a[i * M + 24 + j] - 2.33f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * SHIFT + 16 + 8 + j] = a[i * M + 16 + j] * 5.14f;
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {N, M}, "a");
    Tensor dst(DT_FP32, {1, T}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(N))
        {
            TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
            auto tmp1 = View(a, {1, 16}, {idx, 0});
            tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
            auto tmp2 = View(a, {1, 8}, {idx, 16});
            tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
            auto tmp3 = View(a, {1, 8}, {idx, 24});
            tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
            std::vector<AssembleItem> items{
                {tmp1, {0, idx * SHIFT}}, {tmp3, {0, idx * SHIFT + 16}}, {tmp2, {0, idx * SHIFT + 24}}};
            Assemble(items, dst, parallel);
        }
    }
    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

// 测试单个assemble内串行连接，同时assemble间存在覆盖关系，leaf在泳道图上应该串行
TEST_F(AssembleTest, test_seq_in_assemble_and_overwrite_between_assemble) { TestAssembleOverride<false>(); }

// 测试单个assemble内并行连接，同时assemble间存在覆盖关系，leaf在泳道图上应该串行
TEST_F(AssembleTest, test_parallel_in_assemble_and_overwrite_between_assemble) { TestAssembleOverride<true>(); }

// 测试单个assemble内串行连接（存在覆盖关系只能串行），同时assemble间不存在覆盖关系，leaf在泳道图上应该并行
TEST_F(AssembleTest, test_overwrite_in_assemble_and_parallel_between_assemble)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int N = 32;
    constexpr int M = 48;
    constexpr int T = 1024;
    constexpr int SHIFT = 32;

    auto SimuResult = [](const std::vector<float>& a) {
        ASSERT(a.size() == N * M);
        std::vector<float> out(T, 0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 16; j++) {
                out[i * SHIFT + j] = a[i * M + j] + 1.14f;
            }
            for (int j = 0; j < 16; j++) {
                out[i * SHIFT + 8 + j] = a[i * M + 32 + j] - 2.33f;
            }
            for (int j = 0; j < 16; j++) {
                out[i * SHIFT + 8 + 8 + j] = a[i * M + 16 + j] * 5.14f;
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {N, M}, "a");
    Tensor dst(DT_FP32, {1, T}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(N))
        {
            TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
            auto tmp1 = View(a, {1, 16}, {idx, 0});
            tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
            auto tmp2 = View(a, {1, 16}, {idx, 16});
            tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
            auto tmp3 = View(a, {1, 16}, {idx, 32});
            tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
            std::vector<AssembleItem> items{
                {tmp1, {0, idx * SHIFT}}, {tmp3, {0, idx * SHIFT + 8}}, {tmp2, {0, idx * SHIFT + 16}}};
            Assemble(items, dst);
        }
    }
    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

// 测试assemble的结果作为另一个assemble的输入，leaf内串行，leaf间由于无覆盖关系应该为并行
TEST_F(AssembleTest, test_process_after_assemble)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    auto SimuResult = [](const std::vector<float>& a) {
        constexpr int SHAPE0 = 128;
        constexpr int SHAPE1_IN = 32;
        constexpr int SHAPE1_OUT = 128;
        ASSERT(a.size() == SHAPE0 * SHAPE1_IN);
        std::vector<float> out(SHAPE0 * SHAPE1_OUT, 0);
        for (int i = 0; i < 128; i++) {
            std::vector<float> tmpOut(24, 0);
            for (int j = 0; j < 16; j++) {
                tmpOut[j] = a[i * SHAPE1_IN + j] - 1.23f;
            }
            for (int j = 0; j < 8; j++) {
                tmpOut[16 + j] = a[i * SHAPE1_IN + 24 + j] + 1.23f;
            }
            for (auto& v : tmpOut) {
                v *= 1.55f;
            }
            for (int j = 0; j < 24; j++) {
                out[i * SHAPE1_OUT + j] = tmpOut[j];
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {128, 32}, "a");
    Tensor dst(DT_FP32, {128, 128}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(128))
        {
            TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
            Shape shape1{1, 16};
            Shape shape2{1, 8};
            Shape shape3{1, 8};
            std::vector<SymbolicScalar> offsets1{idx, 0};
            std::vector<SymbolicScalar> offsets2{idx, 16};
            std::vector<SymbolicScalar> offsets3{idx, 24};
            auto tmp1 = View(a, shape1, offsets1);
            auto tmp2 = View(a, shape2, offsets2);
            auto tmp3 = View(a, shape3, offsets3);
            Tensor tmpOut = Full(Element(DT_FP32, 0.0f), DT_FP32, {1, 24});
            tmp1 = Add(tmp1, Element(DT_FP32, -1.23f));
            tmp3 = Add(tmp3, Element(DT_FP32, 1.23f));
            Assemble({{tmp1, {0, 0}}, {tmp3, {0, 16}}}, tmpOut, true);
            tmpOut = Mul(tmpOut, Element(DT_FP32, 1.55f));
            Assemble({{tmpOut, {idx, 0}}}, dst);
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

template <bool parallelInLoop1, bool parallelInLoop2>
void TestLoopAfterLoop()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int N = 128;
    constexpr int M = 32;
    constexpr int T = 128;

    int row = 50;
    auto SimuResult = [&row](const std::vector<float>& a) {
        ASSERT(a.size() == N * M);
        std::vector<float> out(N * T, 0);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < 16; j++) {
                out[i * T + j] = a[i * M + j] + 1.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + j] = a[i * M + 24 + j] - 2.33f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + 8 + j] = a[i * M + 16 + j] * 5.14f;
            }
        }
        for (int i = 0; i < row - 1; i++) {
            for (int j = 0; j < 32; j++) {
                out[i * T + 32 + j] = out[i * T + j] * out[(i + 1) * T + j];
            }
            for (int j = 0; j < 32; j++) {
                out[(i + 1) * T + 64 + j] = out[(i + 1) * T + j] - out[i * T + j];
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {N, M}, "a");
    Tensor dst(DT_FP32, {N, T}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row))
        {
            TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
            Shape shape1{1, 16};
            Shape shape2{1, 8};
            Shape shape3{1, 8};
            std::vector<SymbolicScalar> offsets1{idx, 0};
            std::vector<SymbolicScalar> offsets2{idx, 16};
            std::vector<SymbolicScalar> offsets3{idx, 24};
            auto tmp1 = View(a, shape1, offsets1);
            tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
            auto tmp2 = View(a, shape2, offsets2);
            tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
            auto tmp3 = View(a, shape3, offsets3);
            tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
            std::vector<AssembleItem> items{{tmp1, offsets1}, {tmp3, offsets2}, {tmp2, offsets3}};
            Assemble(items, dst, parallelInLoop1);
        }
        LOOP("loop2", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row - 1))
        {
            TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
            auto x = View(dst, {1, 32}, {idx, 0});
            auto y = View(dst, {1, 32}, {idx + 1, 0});
            auto z1 = Mul(x, y);
            auto z2 = Sub(y, x);
            std::vector<AssembleItem> items{{z1, {idx, 32}}, {z2, {idx + 1, 64}}};
            Assemble(items, dst, parallelInLoop2);
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

// 测试leaf间assemble的并行处理，由于各assemble两两无覆盖，应该并行
TEST_F(AssembleTest, test_loop_after_loop_0) { TestLoopAfterLoop<false, false>(); }

// 测试leaf间assemble的并行处理，由于各assemble两两无覆盖，应该并行
TEST_F(AssembleTest, test_loop_after_loop_1) { TestLoopAfterLoop<false, true>(); }

// 测试leaf间assemble的并行处理，由于各assemble两两无覆盖，应该并行
TEST_F(AssembleTest, test_loop_after_loop_2) { TestLoopAfterLoop<true, false>(); }

// 测试leaf间assemble的并行处理，由于各assemble两两无覆盖，应该并行
TEST_F(AssembleTest, test_loop_after_loop_3) { TestLoopAfterLoop<true, true>(); }

// 测试loop内连续assemble多次，loop内多次assemble之间为串行连接，loop间为并行
TEST_F(AssembleTest, test_override_between_assemble)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int N = 128;
    constexpr int M = 32;
    constexpr int T = 128;

    auto SimuResult = [](const std::vector<float>& a) {
        ASSERT(a.size() == N * M);
        std::vector<float> out(N * T, 0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 16; j++) {
                out[i * T + j] = a[i * M + j] + 1.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + j] = a[i * M + 24 + j] - 2.33f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + 8 + j] = a[i * M + 16 + j] * 5.14f;
            }
        }
        std::vector<float> tmp(N * T, 0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 8; j++) {
                tmp[i * T + 2 + j] = out[i * T + 16 + j] * out[i * T + 24 + j];
            }
            for (int j = 0; j < 8; j++) {
                tmp[i * T + 19 + j] = out[i * T + 24 + j] - out[i * T + 16 + j];
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 8; j++) {
                out[i * T + 2 + j] = tmp[i * T + 2 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 19 + j] = tmp[i * T + 19 + j];
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {N, M}, "a");
    Tensor dst(DT_FP32, {N, T}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(N))
        {
            TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
            Shape shape1{1, 16};
            Shape shape2{1, 8};
            Shape shape3{1, 8};
            std::vector<SymbolicScalar> offsets1{idx, 0};
            std::vector<SymbolicScalar> offsets2{idx, 16};
            std::vector<SymbolicScalar> offsets3{idx, 24};
            auto tmp1 = View(a, shape1, offsets1);
            tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
            auto tmp2 = View(a, shape2, offsets2);
            tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
            auto tmp3 = View(a, shape3, offsets3);
            tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
            std::vector<AssembleItem> items{{tmp1, offsets1}, {tmp3, offsets2}, {tmp2, offsets3}};
            Assemble(items, dst, true);

            auto z1 = Mul(tmp2, tmp3);
            auto z2 = Sub(tmp2, tmp3);
            Assemble({{z1, {idx, 2}}, {z2, {idx, 19}}}, dst, true); // 覆盖上一次Assemble的部分结果
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

// 测试loop内连续assemble多次，loop内多次assemble之间为串行连接，loop间由于第一个和最后一个之间存在依赖关系，因此在泳道图上应为串行
TEST_F(AssembleTest, test_mix_assemble_0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int N = 128;
    constexpr int M = 32;
    constexpr int T = 128;

    int row = 50;
    auto SimuResult = [&row](const std::vector<float>& a) {
        ASSERT(a.size() == N * M);
        std::vector<float> out(N * T, 0);
        for (int i = 0; i < row - 1; i++) {
            for (int j = 0; j < 8; j++) {
                out[i * T + 8 + j] = a[i * M + 8 + j] + 1.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + j] = a[i * M + 24 + j] - 2.33f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + 8 + j] = a[i * M + 16 + j] * 5.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 32 + j] = out[i * T + 16 + j] * out[i * T + 24 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 40 + j] = out[i * T + 24 + j] - out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 32 + j] = out[i * T + 8 + j] + out[i * T + 24 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 40 + j] = out[i * T + 8 + j] + out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[(i + 1) * T + j] = out[i * T + 24 + j] - out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[(i + 1) * T + 8 + j] = out[i * T + 16 + j] * out[i * T + 24 + j];
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {N, M}, "a");
    Tensor dst(DT_FP32, {N, T}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row - 1))
        {
            TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
            Shape shape1{1, 8};
            Shape shape2{1, 8};
            Shape shape3{1, 8};
            std::vector<SymbolicScalar> offsets1{idx, 8};
            std::vector<SymbolicScalar> offsets2{idx, 16};
            std::vector<SymbolicScalar> offsets3{idx, 24};
            auto tmp1 = View(a, shape1, offsets1);
            tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
            auto tmp2 = View(a, shape2, offsets2);
            tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
            auto tmp3 = View(a, shape3, offsets3);
            tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
            std::vector<AssembleItem> items{{tmp1, offsets1}, {tmp3, offsets2}, {tmp2, offsets3}};
            Assemble(items, dst, true);

            auto z1 = Mul(tmp2, tmp3);
            auto z2 = Sub(tmp2, tmp3);
            Assemble({{z1, {idx, 32}}, {z2, {idx, 40}}}, dst, true);
            auto z3 = Add(tmp1, tmp2);
            auto z4 = Add(tmp1, tmp3);
            Assemble({{z3, {idx, 32}}, {z4, {idx, 40}}}, dst, true); // 覆盖上一次Assemble的部分结果

            Assemble({{z2, {idx + 1, 0}}, {z1, {idx + 1, 8}}}, dst, true);
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

// 测试多个loop进行不同的assemble（覆盖和可并行的一起执行）
// inner_loop_2的assemble独立无覆盖，在泳道图上应并行
// inner_loop_1覆盖上一个loop1的inner_loop_3数据，存在依赖关系，因此应串行处理
// 任意一组存在依赖关系的inner_loop_1+inner_loop_3和其他的inner_loop_1+inner_loop_3之间无覆盖关系，因此应并行处理
TEST_F(AssembleTest, test_mix_assemble_1)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int N = 128;
    constexpr int M = 32;
    constexpr int T = 128;

    int row = 30;
    auto SimuResult = [&row](const std::vector<float>& a) {
        ASSERT(a.size() == N * M);
        std::vector<float> out(N * T, 0);
        for (int i = 0; i < row - 1; i++) {
            for (int j = 0; j < 8; j++) {
                out[i * T + 8 + j] = a[i * M + 8 + j] + 1.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + j] = a[i * M + 24 + j] - 2.33f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + 8 + j] = a[i * M + 16 + j] * 5.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 32 + j] = out[i * T + 16 + j] * out[i * T + 24 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 40 + j] = out[i * T + 24 + j] - out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 32 + j] = out[i * T + 8 + j] + out[i * T + 24 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 40 + j] = out[i * T + 8 + j] + out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[(i + 1) * T + j] = out[i * T + 24 + j] - out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[(i + 1) * T + 8 + j] = out[i * T + 16 + j] * out[i * T + 24 + j];
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {N, M}, "a");
    Tensor dst(DT_FP32, {N, T}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row - 1))
        {
            TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
            Shape shape1{1, 8};
            Shape shape2{1, 8};
            Shape shape3{1, 8};
            std::vector<SymbolicScalar> offsets1{idx, 8};
            std::vector<SymbolicScalar> offsets2{idx, 16};
            std::vector<SymbolicScalar> offsets3{idx, 24};
            LOOP("inner_loop_1", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                auto tmp1 = View(a, shape1, offsets1);
                tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
                auto tmp2 = View(a, shape2, offsets2);
                tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
                auto tmp3 = View(a, shape3, offsets3);
                tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
                std::vector<AssembleItem> items{{tmp1, offsets1}, {tmp3, offsets2}, {tmp2, offsets3}};
                Assemble(items, dst, true);
            }
            LOOP("inner_loop_2", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                auto tmp1 = View(a, shape1, offsets1);
                tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
                auto tmp2 = View(a, shape2, offsets2);
                tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
                auto tmp3 = View(a, shape3, offsets3);
                tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
                auto z1 = Mul(tmp2, tmp3);
                auto z2 = Sub(tmp2, tmp3);
                Assemble({{z1, {idx, 32}}, {z2, {idx, 40}}}, dst, true);
                auto z3 = Add(tmp1, tmp2);
                auto z4 = Add(tmp1, tmp3);
                Assemble({{z3, {idx, 32}}, {z4, {idx, 40}}}, dst, true); // 覆盖上一次Assemble的部分结果
            }
            LOOP("inner_loop_3", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                auto tmp2 = View(a, shape2, offsets2);
                tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
                auto tmp3 = View(a, shape3, offsets3);
                tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
                auto z1 = Mul(tmp2, tmp3);
                auto z2 = Sub(tmp2, tmp3);
                Assemble({{z2, {idx + 1, 0}}, {z1, {idx + 1, 8}}}, dst, true);
            }
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

// 测试对内部自动分配内存的assemble处理
// inner_loop_2的assemble独立无覆盖，在泳道图上应并行
// inner_loop_1覆盖上一个loop1的inner_loop_3数据，存在依赖关系，因此应串行处理
// 任意一组存在依赖关系的inner_loop_1+inner_loop_3和其他的inner_loop_1+inner_loop_3之间无覆盖关系，因此应并行处理
TEST_F(AssembleTest, test_assemble_to_inner_tensor_0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int N = 128;
    constexpr int M = 32;
    constexpr int T = 128;

    int row = 30;
    auto SimuResult = [&row](const std::vector<float>& a) {
        ASSERT(a.size() == N * M);
        std::vector<float> out(N * T, 0);
        for (int i = 0; i < row - 1; i++) {
            for (int j = 0; j < 8; j++) {
                out[i * T + 8 + j] = a[i * M + 8 + j] + 1.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + j] = a[i * M + 24 + j] - 2.33f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + 8 + j] = a[i * M + 16 + j] * 5.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 32 + j] = out[i * T + 16 + j] * out[i * T + 24 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 40 + j] = out[i * T + 24 + j] - out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 32 + j] = out[i * T + 8 + j] + out[i * T + 24 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 40 + j] = out[i * T + 8 + j] + out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[(i + 1) * T + j] = out[i * T + 24 + j] - out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[(i + 1) * T + 8 + j] = out[i * T + 16 + j] * out[i * T + 24 + j];
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {N, M}, "a");
    Tensor dst(DT_FP32, {N, T}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        Tensor tmpOut(DT_FP32, {N, T}, "tmpOut");
        LOOP("init_data", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
        {
            (void)unusedIdx;
            TileShape::Current().SetVecTile(128, 128);
            tmpOut = Full(Element(DT_FP32, 0.0f), DT_FP32, {N, T});
        }
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row - 1))
        {
            TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
            Shape shape1{1, 8};
            Shape shape2{1, 8};
            Shape shape3{1, 8};
            std::vector<SymbolicScalar> offsets1{idx, 8};
            std::vector<SymbolicScalar> offsets2{idx, 16};
            std::vector<SymbolicScalar> offsets3{idx, 24};
            LOOP("inner_loop_1", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                auto tmp1 = View(a, shape1, offsets1);
                tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
                auto tmp2 = View(a, shape2, offsets2);
                tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
                auto tmp3 = View(a, shape3, offsets3);
                tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
                std::vector<AssembleItem> items{{tmp1, offsets1}, {tmp3, offsets2}, {tmp2, offsets3}};
                Assemble(items, tmpOut, true);
            }
            LOOP("inner_loop_2", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                auto tmp1 = View(a, shape1, offsets1);
                tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
                auto tmp2 = View(a, shape2, offsets2);
                tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
                auto tmp3 = View(a, shape3, offsets3);
                tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
                auto z1 = Mul(tmp2, tmp3);
                auto z2 = Sub(tmp2, tmp3);
                Assemble({{z1, {idx, 32}}, {z2, {idx, 40}}}, tmpOut, true);
                auto z3 = Add(tmp1, tmp2);
                auto z4 = Add(tmp1, tmp3);
                Assemble({{z3, {idx, 32}}, {z4, {idx, 40}}}, tmpOut, true); // 覆盖上一次Assemble的部分结果
            }
            LOOP("inner_loop_3", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                auto tmp2 = View(a, shape2, offsets2);
                tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
                auto tmp3 = View(a, shape3, offsets3);
                tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
                auto z1 = Mul(tmp2, tmp3);
                auto z2 = Sub(tmp2, tmp3);
                Assemble({{z2, {idx + 1, 0}}, {z1, {idx + 1, 8}}}, tmpOut, true);
            }
        }
        LOOP("copy_to_output", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
        {
            (void)unusedIdx;
            TileShape::Current().SetVecTile(128, 128);
            dst = Assign(tmpOut);
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

/* 该用例涉及跨LOOP的写后读后写场景，目前无法支持，先跳过 */
// 测试利用滚动数组逻辑，用临时内存进行循环更新
TEST_F(AssembleTest, test_assemble_to_inner_tensor_1)
{
    return;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int N = 128;
    constexpr int M = 32;
    constexpr int T = 128;

    int row = 20;
    auto SimuResult = [&row](const std::vector<float>& a) {
        ASSERT(a.size() == N * M);
        std::vector<float> out(N * T, 0);
        for (int i = 0; i < row - 1; i++) {
            for (int j = 0; j < 8; j++) {
                out[i * T + 8 + j] = a[i * M + 8 + j] + 1.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + j] = a[i * M + 24 + j] - 2.33f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + 8 + j] = a[i * M + 16 + j] * 5.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 32 + j] = out[i * T + 16 + j] * out[i * T + 24 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 40 + j] = out[i * T + 24 + j] - out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 32 + j] = out[i * T + 8 + j] + out[i * T + 24 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 40 + j] = out[i * T + 8 + j] + out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[(i + 1) * T + j] = out[i * T + 24 + j] - out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[(i + 1) * T + 8 + j] = out[i * T + 16 + j] * out[i * T + 24 + j];
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {N, M}, "a");
    Tensor dst(DT_FP32, {N, T}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        Tensor tmpOut(DT_FP32, {2, T}, "tmpOut");
        Tensor tmpOut2(DT_FP32, {N, T}, "tmpOut2");
        LOOP("init_data", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
        {
            (void)unusedIdx;
            config::SetSemanticLabel("init_data");
            TileShape::Current().SetVecTile(128, 128);
            tmpOut = Full(Element(DT_FP32, 0.0f), DT_FP32, {2, T});
            tmpOut2 = Full(Element(DT_FP32, 0.0f), DT_FP32, {N, T});
        }
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row - 1))
        {
            TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
            Shape shape1{1, 8};
            Shape shape2{1, 8};
            Shape shape3{1, 8};
            std::vector<SymbolicScalar> offsets1{idx, 8};
            std::vector<SymbolicScalar> offsets2{idx, 16};
            std::vector<SymbolicScalar> offsets3{idx, 24};
            LOOP("inner_loop_1", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                config::SetSemanticLabel("inner_loop_1");
                auto tmp1 = Add(View(a, shape1, offsets1), Element(DT_FP32, 1.14f));
                auto tmp2 = Mul(View(a, shape2, offsets2), Element(DT_FP32, 5.14f));
                auto tmp3 = Sub(View(a, shape3, offsets3), Element(DT_FP32, 2.33f));
                std::vector<AssembleItem> items{{tmp1, {idx % 2, 8}}, {tmp3, {idx % 2, 16}}, {tmp2, {idx % 2, 24}}};
                Assemble(items, tmpOut, true);
            }
            LOOP("inner_loop_2", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                config::SetSemanticLabel("inner_loop_2");
                auto tmp1 = Add(View(a, shape1, offsets1), Element(DT_FP32, 1.14f));
                auto tmp2 = Mul(View(a, shape2, offsets2), Element(DT_FP32, 5.14f));
                auto tmp3 = Sub(View(a, shape3, offsets3), Element(DT_FP32, 2.33f));
                Assemble({{Mul(tmp2, tmp3), {idx % 2, 32}}, {Sub(tmp2, tmp3), {idx % 2, 40}}}, tmpOut, true);
                Assemble({{Add(tmp1, tmp2), {idx % 2, 32}}, {Add(tmp1, tmp3), {idx % 2, 40}}}, tmpOut,
                         true); // 覆盖上一次Assemble的部分结果
            }
            LOOP("inner_loop_3", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                config::SetSemanticLabel("inner_loop_3");
                auto tmp2 = Mul(View(a, shape2, offsets2), Element(DT_FP32, 5.14f));
                auto tmp3 = Sub(View(a, shape3, offsets3), Element(DT_FP32, 2.33f));
                Assemble({{Sub(tmp2, tmp3), {(idx + 1) % 2, 0}}, {Mul(tmp2, tmp3), {(idx + 1) % 2, 8}}}, tmpOut, true);
            }
            LOOP("copy_to_tmpout2", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                config::SetSemanticLabel("copy_to_tmpout2");
                TileShape::Current().SetVecTile(128, 128);
                auto x = View(tmpOut, {1, T}, {idx % 2, 0});
                Assemble({{x, {idx, 0}}}, tmpOut2, true);
            }
        }
        LOOP("copy_to_output", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
        {
            (void)unusedIdx;
            config::SetSemanticLabel("copy_to_output");
            TileShape::Current().SetVecTile(128, 128);
            dst = Assign(tmpOut2);
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

// 测试对循环内自动分配内存的assemble处理，应对循环内的assemble对象每次分配新内存
TEST_F(AssembleTest, test_assemble_to_inner_tensor_2)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int N = 128;
    constexpr int M = 32;
    constexpr int T = 128;

    int row = 20;
    auto SimuResult = [&row](const std::vector<float>& a) {
        ASSERT(a.size() == N * M);
        std::vector<float> out(N * T, 0);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < 8; j++) {
                out[i * T + 8 + j] = a[i * M + 8 + j] + 1.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + j] = a[i * M + 24 + j] - 2.33f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 16 + 8 + j] = a[i * M + 16 + j] * 5.14f;
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 32 + j] = out[i * T + 16 + j] * out[i * T + 24 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 40 + j] = out[i * T + 24 + j] - out[i * T + 16 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 32 + j] = out[i * T + 8 + j] + out[i * T + 24 + j];
            }
            for (int j = 0; j < 8; j++) {
                out[i * T + 40 + j] = out[i * T + 8 + j] + out[i * T + 16 + j];
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {N, M}, "a");
    Tensor dst(DT_FP32, {N, T}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        Tensor tmpOut(DT_FP32, {N, T}, "tmpOut2");
        LOOP("init_data_1", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
        {
            (void)unusedIdx;
            config::SetSemanticLabel("init_data");
            TileShape::Current().SetVecTile(128, 128);
            tmpOut = Full(Element(DT_FP32, 0.0f), DT_FP32, {N, T});
        }
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row))
        {
            TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
            Shape shape1{1, 8};
            Shape shape2{1, 8};
            Shape shape3{1, 8};
            std::vector<SymbolicScalar> offsets1{idx, 8};
            std::vector<SymbolicScalar> offsets2{idx, 16};
            std::vector<SymbolicScalar> offsets3{idx, 24};
            Tensor innerTmpOut(DT_FP32, {1, T}, "innerTmpOut");
            LOOP("init_data_2", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                config::SetSemanticLabel("init_data");
                TileShape::Current().SetVecTile(128, 128);
                innerTmpOut = Full(Element(DT_FP32, 0.0f), DT_FP32, {1, T});
            }
            LOOP("inner_loop_1", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                config::SetSemanticLabel("inner_loop_1");
                auto tmp1 = View(a, shape1, offsets1);
                tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
                auto tmp2 = View(a, shape2, offsets2);
                tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
                auto tmp3 = View(a, shape3, offsets3);
                tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
                std::vector<AssembleItem> items{{tmp1, {0, 8}}, {tmp3, {0, 16}}, {tmp2, {0, 24}}};
                Assemble(items, innerTmpOut, true);
            }
            LOOP("inner_loop_2", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                config::SetSemanticLabel("inner_loop_2");
                auto tmp1 = View(a, shape1, offsets1);
                tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
                auto tmp2 = View(a, shape2, offsets2);
                tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
                auto tmp3 = View(a, shape3, offsets3);
                tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
                auto z1 = Mul(tmp2, tmp3);
                auto z2 = Sub(tmp2, tmp3);
                Assemble({{z1, {0, 32}}, {z2, {0, 40}}}, innerTmpOut, true);
                auto z3 = Add(tmp1, tmp2);
                auto z4 = Add(tmp1, tmp3);
                Assemble({{z3, {0, 32}}, {z4, {0, 40}}}, innerTmpOut, true); // 覆盖上一次Assemble的部分结果
            }
            LOOP("copy_to_tmpout", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
            {
                (void)unusedIdx;
                config::SetSemanticLabel("copy_to_tmpout");
                TileShape::Current().SetVecTile(128, 128);
                Assemble({{innerTmpOut, {idx, 0}}}, tmpOut, true);
            }
        }
        LOOP("copy_to_output", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
        {
            (void)unusedIdx;
            config::SetSemanticLabel("copy_to_output");
            TileShape::Current().SetVecTile(128, 128);
            dst = Assign(tmpOut);
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

// 测试对循环内自动分配内存的assemble处理，由于loop间存在依赖关系，泳道图上应该串行
TEST_F(AssembleTest, test_assemble_to_inner_tensor_3)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int N = 24;
    constexpr int M = 48;
    constexpr int T = 1024;
    constexpr int SHIFT = 32; // 下一个LOOP和上一个LOOP之间存在覆盖关系，形成串联
                              // TODO：SHIFT设为20会出现非对齐，导致丢失依赖，等兴旺的cell 链表上库后再试试

    auto SimuResult = [](const std::vector<float>& a) {
        ASSERT(a.size() == N * M);
        std::vector<float> out(T, 0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 16; j++) {
                out[i * SHIFT + j] = a[i * M + j] + 1.14f;
            }
            for (int j = 0; j < 16; j++) {
                out[i * SHIFT + 8 + j] = a[i * M + 32 + j] - 2.33f;
            }
            for (int j = 0; j < 32; j++) {
                out[i * SHIFT + 8 + 8 + j] = a[i * M + 16 + j] * 5.14f;
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {N, M}, "a");
    Tensor dst(DT_FP32, {1, T}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        Tensor tmpOut;
        LOOP("init_data_1", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
        {
            (void)unusedIdx;
            config::SetSemanticLabel("init_data");
            TileShape::Current().SetVecTile(128, 128);
            tmpOut = Full(Element(DT_FP32, 0.0f), DT_FP32, {1, T});
        }
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(N / 8))
        {
            LOOP("inner_loop_1", FunctionType::DYNAMIC_LOOP, idx2, LoopRange(8))
            {
                config::SetSemanticLabel("inner_loop_1");
                auto idx = idx1 * 8 + idx2;
                TileShape::Current().SetVecTile(8, 128); // 暂不让ASSEMBLE_SSA进行自动TILING
                auto tmp1 = View(a, {1, 16}, {idx, 0});
                tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
                auto tmp2 = View(a, {1, 32}, {idx, 16});
                tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
                auto tmp3 = View(a, {1, 16}, {idx, 32});
                tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
                std::vector<AssembleItem> items{
                    {tmp1, {0, idx * SHIFT}},
                    {tmp3, {0, idx * SHIFT + 8}},
                    {tmp2, {0, idx * SHIFT + 16}},
                };
                Assemble(items, tmpOut);
            }
        }
        LOOP("copy_to_output", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1))
        {
            (void)unusedIdx;
            config::SetSemanticLabel("copy_to_output");
            TileShape::Current().SetVecTile(128, 128);
            dst = Assign(tmpOut);
        }
    }
    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
    for (int i = 0; i < ShapeSize(dst.GetShape()); i++) {
        auto actual = ((float*)dstResult->data())[i];
        auto expect = dstGolden[i];
        if (fabs(actual - expect) > eps) {
            std::cout << i << ": actual: " << actual << ", expect: " << expect << std::endl;
        }
    }
}

// 测试tiling下的assemble
TEST_F(AssembleTest, test_mix_assemble_with_tiling)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int N = 128;
    constexpr int M = 512;
    constexpr int T = 1024;

    int row = 50;
    auto SimuResult = [&row](const std::vector<float>& a) {
        ASSERT(a.size() == N * M);
        std::vector<float> out(N * T, 0);
        for (int i = 0; i < row - 1; i++) {
            for (int j = 0; j < 128; j++) {
                out[i * T + 128 + j] = a[i * M + 128 + j] + 1.14f;
            }
            for (int j = 0; j < 128; j++) {
                out[i * T + 256 + j] = a[i * M + 384 + j] - 2.33f;
            }
            for (int j = 0; j < 128; j++) {
                out[i * T + 256 + 128 + j] = a[i * M + 256 + j] * 5.14f;
            }
            for (int j = 0; j < 128; j++) {
                out[i * T + 512 + j] = out[i * T + 256 + j] * out[i * T + 384 + j];
            }
            for (int j = 0; j < 128; j++) {
                out[i * T + 640 + j] = out[i * T + 384 + j] - out[i * T + 256 + j];
            }
            for (int j = 0; j < 128; j++) {
                out[i * T + 512 + j] = out[i * T + 128 + j] + out[i * T + 384 + j];
            }
            for (int j = 0; j < 128; j++) {
                out[i * T + 640 + j] = out[i * T + 128 + j] + out[i * T + 256 + j];
            }
            for (int j = 0; j < 128; j++) {
                out[(i + 1) * T + j] = out[i * T + 384 + j] - out[i * T + 256 + j];
            }
            for (int j = 0; j < 128; j++) {
                out[(i + 1) * T + 128 + j] = out[i * T + 256 + j] * out[i * T + 384 + j];
            }
        }
        return out;
    };

    Tensor a(DT_FP32, {N, M}, "a");
    Tensor dst(DT_FP32, {N, T}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row - 1))
        {
            TileShape::Current().SetVecTile(1, 64);
            Shape shape1{1, 128};
            Shape shape2{1, 128};
            Shape shape3{1, 128};
            std::vector<SymbolicScalar> offsets1{idx, 128};
            std::vector<SymbolicScalar> offsets2{idx, 256};
            std::vector<SymbolicScalar> offsets3{idx, 384};
            auto tmp1 = View(a, shape1, offsets1);
            tmp1 = Add(tmp1, Element(DT_FP32, 1.14f));
            auto tmp2 = View(a, shape2, offsets2);
            tmp2 = Mul(tmp2, Element(DT_FP32, 5.14f));
            auto tmp3 = View(a, shape3, offsets3);
            tmp3 = Sub(tmp3, Element(DT_FP32, 2.33f));
            std::vector<AssembleItem> items{{tmp1, offsets1}, {tmp3, offsets2}, {tmp2, offsets3}};
            Assemble(items, dst, false);

            auto z1 = Mul(tmp2, tmp3);
            auto z2 = Sub(tmp2, tmp3);
            Assemble({{z1, {idx, 512}}, {z2, {idx, 640}}}, dst, true);
            auto z3 = Add(tmp1, tmp2);
            auto z4 = Add(tmp1, tmp3);
            Assemble({{z3, {idx, 512}}, {z4, {idx, 640}}}, dst, true); // 覆盖上一次Assemble的部分结果

            Assemble({{z2, {idx + 1, 0}}, {z1, {idx + 1, 128}}}, dst, true);
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

// 测试Concat转核内Assemble的能力
TEST_F(AssembleTest, test_inner_assemble_with_concat)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    constexpr int X = 509;
    constexpr int M = 1024;

    auto SimuResult = [](const std::vector<float>& a, const std::vector<float>& b) {
        ASSERT(a.size() + b.size() == M);
        std::vector<float> out(M, 0);
        for (size_t i = 0; i < a.size(); i++) {
            out[i] = a[i];
        }
        for (size_t i = 0; i < b.size(); i++) {
            out[a.size() + i] = b[i];
        }
        for (size_t i = 0; i < out.size(); i++) {
            out[i] *= 2;
        }
        return out;
    };

    Tensor a(DT_FP32, {1, X}, "a");
    Tensor b(DT_FP32, {1, M - X}, "b");
    Tensor dst(DT_FP32, {1, M}, "dst");

    std::vector<float> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    std::vector<float> bData(ShapeSize(b.GetShape()), 0);
    for (size_t i = 0; i < bData.size(); i++) {
        bData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData, bData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(a, aData),
        RawTensorData::CreateTensor<float>(b, bData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(dst, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(dst, dstGolden),
    });

    FUNCTION("test", {a, b}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, unused, LoopRange(1))
        {
            (void)unused;
            TileShape::Current().SetVecTile(1, 2048);
            auto c = Cat(std::vector<Tensor>{a, b}, -1);
            dst = Add(c, c);
        }
    }
    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-3f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (float*)dstResult->data(), eps));
}

template <typename T>
void TestInnerAssemble()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    DataType dType;
    if constexpr (std::is_same_v<T, float>) {
        dType = DT_FP32;
    } else if constexpr (std::is_same_v<T, int>) {
        dType = DT_INT32;
    } else if constexpr (std::is_same_v<T, float16>) {
        dType = DT_FP16;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        dType = DT_BF16;
    } else {
        ASSERT(false);
    }

    constexpr int N = 20;
    constexpr int M = 1024;

    int row = N;
    auto SimuResult = [&row](const std::vector<T>& a, const std::vector<int>& lens, T startValue) {
        ASSERT(a.size() == N * M);
        ASSERT(lens.size() == N);
        std::vector<T> out(N * M, -5.0f);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < lens[i]; j++) {
                out[i * M + j] = startValue;
            }
            for (int j = lens[i]; j < M; j++) {
                out[i * M + j] = a[i * M + j];
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < M; j++) {
                out[i * M + j] *= 2;
            }
        }
        return out;
    };

    Tensor a(dType, {N, M}, "a");
    Tensor lens(DT_INT32, {N}, "len");
    Tensor dst(dType, {N, M}, "dst");

    T startValue = uniform(gen);
    std::vector<T> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    std::vector<int> lenData(ShapeSize(lens.GetShape()), 0);
    std::uniform_int_distribution<int> uniformInt(1, M - 1);
    for (size_t i = 0; i < lenData.size(); i++) {
        lenData[i] = uniformInt(gen);
    }
    auto dstGolden = SimuResult(aData, lenData, startValue);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(a, aData),
        RawTensorData::CreateTensor<int>(lens, lenData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(dst, -5.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<T>(dst, dstGolden),
    });

    FUNCTION("test", {a, lens}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row))
        {
            TileShape::Current().SetVecTile(1, 2048);
            Tensor tmp(dType, {1, M}, "tmp");
            auto len = GetTensorData(lens, {idx});
            auto defaultValue = Full(Element(dType, startValue), dType, {1, M}, {1, len}); // full不支持fp16
            auto aView = View(a, {1, M}, {1, M - len}, {idx, len});
            Assemble({{defaultValue, {0, 0}}, {Assign(aView), {0, len}}}, tmp, true);
            tmp = Add(tmp, tmp);
            Assemble({{tmp, {idx, 0}}}, dst, true);
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-6f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (T*)dstResult->data(), eps));
    for (int i = 0; i < ShapeSize(dst.GetShape()); i++) {
        auto actual = ((T*)dstResult->data())[i];
        auto expect = dstGolden[i];
        if (fabs(actual - expect) > eps) {
            std::cout << i << ": actual: " << actual << ", expect: " << expect << std::endl;
        }
    }
}

// 测试核内assemble，尾轴字节对齐，每次view一行
TEST_F(AssembleTest, test_inner_assemble_fp32) { TestInnerAssemble<float>(); }

template <typename T>
void TestInnerAssembleByFrameWork()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    DataType dType;
    if constexpr (std::is_same_v<T, float>) {
        dType = DT_FP32;
    } else if constexpr (std::is_same_v<T, int>) {
        dType = DT_INT32;
    } else if constexpr (std::is_same_v<T, float16>) {
        dType = DT_FP16;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        dType = DT_BF16;
    } else {
        ASSERT(false);
    }

    constexpr int N = 20;
    constexpr int M = 1024;

    int row = N;
    auto SimuResult = [&row](const std::vector<T>& a, const std::vector<int>& lens) {
        ASSERT(a.size() == N * M);
        ASSERT(lens.size() == N);
        std::vector<T> out(N * M, -5.0f);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < lens[i]; j++) {
                out[i * M + j] = a[i * M + j] * 2 * a[i * M + j];
            }
        }
        return out;
    };

    Tensor a(dType, {N, M}, "a");
    Tensor lens(DT_INT32, {N}, "len");
    Tensor dst(dType, {N, M}, "dst");

    std::vector<T> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    std::vector<int> lenData(ShapeSize(lens.GetShape()), 0);
    std::uniform_int_distribution<int> uniformInt(1, M - 1);
    for (size_t i = 0; i < lenData.size(); i++) {
        lenData[i] = uniformInt(gen);
    }
    auto dstGolden = SimuResult(aData, lenData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(a, aData),
        RawTensorData::CreateTensor<int>(lens, lenData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(dst, -5.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<T>(dst, dstGolden),
    });

    FUNCTION("test", {a, lens}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row))
        {
            TileShape::Current().SetVecTile(1, 128);
            auto len = GetTensorData(lens, {idx});
            auto b = View(a, {1, M}, {1, len}, {idx, 0});
            auto c = Add(b, b);
            TileShape::Current().SetVecTile(1, 512);
            auto d = Mul(c, b);
            Assemble({{d, {idx, 0}}}, dst, true);
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-6f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (T*)dstResult->data(), eps));
    for (int i = 0; i < ShapeSize(dst.GetShape()); i++) {
        auto actual = ((T*)dstResult->data())[i];
        auto expect = dstGolden[i];
        if (fabs(actual - expect) > eps) {
            std::cout << i << ": actual: " << actual << ", expect: " << expect << std::endl;
        }
    }
}

// 测试核内assemble，尾轴字节对齐，每次view一行
TEST_F(AssembleTest, test_inner_assemble_by_framework_fp32) { TestInnerAssembleByFrameWork<float>(); }

template <typename T>
void TestInnerAssembleMultiLine()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    DataType dType;
    if constexpr (std::is_same_v<T, float>) {
        dType = DT_FP32;
    } else if constexpr (std::is_same_v<T, int>) {
        dType = DT_INT32;
    } else if constexpr (std::is_same_v<T, float16>) {
        dType = DT_FP16;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        dType = DT_BF16;
    } else {
        ASSERT(false);
    }

    constexpr int N = 20;
    constexpr int M = 1032;

    ASSERT(N % 5 == 0);
    int row = N / 5;
    auto SimuResult = [&row](const std::vector<T>& a, const std::vector<int>& lens, T startValue) {
        ASSERT(a.size() == N * M);
        ASSERT(lens.size() * 5 == N);
        std::vector<T> out(N * M, -5.0f);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < lens[i]; k++) {
                    out[(i * 5 + j) * M + k] = startValue;
                }
                for (int k = lens[i]; k < M; k++) {
                    out[(i * 5 + j) * M + k] = a[(i * 5 + j) * M + k];
                }
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                out[i * M + j] *= 2;
            }
        }
        return out;
    };

    Tensor a(dType, {N, M}, "a");
    Tensor lens(DT_INT32, {row}, "len");
    Tensor dst(dType, {N, M}, "dst");

    T startValue = uniform(gen);
    std::vector<T> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    std::vector<int> lenData(ShapeSize(lens.GetShape()), 0);
    std::uniform_int_distribution<int> uniformInt(1, M - 1);
    for (size_t i = 0; i < lenData.size(); i++) {
        lenData[i] = uniformInt(gen);
    }
    auto dstGolden = SimuResult(aData, lenData, startValue);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(a, aData),
        RawTensorData::CreateTensor<int>(lens, lenData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(dst, -5.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<T>(dst, dstGolden),
    });

    FUNCTION("test", {a, lens}, {dst})
    {
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row))
        {
            TileShape::Current().SetVecTile(5, 2048);
            Tensor tmp(dType, {5, M}, "tmp");
            auto len = GetTensorData(lens, {idx});
            auto defaultValue = Full(Element(dType, startValue), dType, {5, M}, {5, len}); // full不支持fp16
            auto aView = View(a, {5, M}, {5, M - len}, {idx * 5, len});
            Assemble({{defaultValue, {0, 0}}, {Assign(aView), {0, len}}}, tmp, true);
            tmp = Add(tmp, tmp);
            Assemble({{tmp, {idx * 5, 0}}}, dst, true);
        }
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-6f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (T*)dstResult->data(), eps));
    for (int i = 0; i < ShapeSize(dst.GetShape()); i++) {
        auto actual = ((T*)dstResult->data())[i];
        auto expect = dstGolden[i];
        if (fabs(actual - expect) > eps) {
            std::cout << i << ": actual: " << actual << ", expect: " << expect << std::endl;
        }
    }
}

// 测试核内assemble，每次view一个矩形
TEST_F(AssembleTest, test_inner_assemble_multi_line_fp32) { TestInnerAssembleMultiLine<float>(); }

template <typename T>
void TestInnerAssembleMultiView()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(-10, 10);

    DataType dType;
    if constexpr (std::is_same_v<T, float>) {
        dType = DT_FP32;
    } else if constexpr (std::is_same_v<T, int>) {
        dType = DT_INT32;
    } else if constexpr (std::is_same_v<T, float16>) {
        dType = DT_FP16;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        dType = DT_BF16;
    } else {
        ASSERT(false);
    }

    constexpr int N = 20;
    constexpr int M = 1032;

    ASSERT(N % 5 == 0);
    int row = N / 5;
    auto SimuResult = [&row](const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c,
                             const std::vector<int>& lens1, const std::vector<int>& lens2) {
        ASSERT(a.size() == N * M);
        ASSERT(lens1.size() * 5 == N);
        ASSERT(lens2.size() * 5 == N);
        std::vector<T> out(N * M, -5.0f);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < lens1[i]; k++) {
                    out[(i * 5 + j) * M + k] = a[(i * 5 + j) * M + k];
                }
                for (int k = lens1[i]; k < lens2[i]; k++) {
                    out[(i * 5 + j) * M + k] = b[(i * 5 + j) * M + k];
                }
                for (int k = lens2[i]; k < M; k++) {
                    out[(i * 5 + j) * M + k] = c[(i * 5 + j) * M + k];
                }
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                out[i * M + j] *= 2;
            }
        }
        return out;
    };

    Tensor a(dType, {N, M}, "a");
    Tensor b(dType, {N, M}, "b");
    Tensor c(dType, {N, M}, "c");
    Tensor lens1(DT_INT32, {row}, "lens1");
    Tensor lens2(DT_INT32, {row}, "lens2");
    Tensor dst(dType, {N, M}, "dst");

    std::vector<T> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    std::vector<T> bData(ShapeSize(b.GetShape()), 0);
    for (size_t i = 0; i < bData.size(); i++) {
        bData[i] = uniform(gen);
    }
    std::vector<T> cData(ShapeSize(c.GetShape()), 0);
    for (size_t i = 0; i < cData.size(); i++) {
        cData[i] = uniform(gen);
    }
    std::vector<int> lens1Data(ShapeSize(lens1.GetShape()), 0);
    std::uniform_int_distribution<int> uniformInt1(1, M / 2);
    for (size_t i = 0; i < lens1Data.size(); i++) {
        lens1Data[i] = 364; // uniformInt1(gen);
        std::cout << "lens1:" << lens1Data[i] << std::endl;
    }
    std::vector<int> lens2Data(ShapeSize(lens2.GetShape()), 0);
    std::uniform_int_distribution<int> uniformInt2(M / 2, M);
    for (size_t i = 0; i < lens2Data.size(); i++) {
        lens2Data[i] = 644; // uniformInt2(gen);
        std::cout << "lens2:" << lens2Data[i] << std::endl;
    }
    auto dstGolden = SimuResult(aData, bData, cData, lens1Data, lens2Data);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(a, aData),
        RawTensorData::CreateTensor<T>(b, bData),
        RawTensorData::CreateTensor<T>(c, cData),
        RawTensorData::CreateTensor<int>(lens1, lens1Data),
        RawTensorData::CreateTensor<int>(lens2, lens2Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(dst, -5.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<T>(dst, dstGolden),
    });

    FUNCTION("test", {a, b, c, lens1, lens2}, {dst})
    {
        config::SetPassOption(SG_SET_SCOPE, std::vector<int64_t>{1, 1, 0});
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row))
        {
            TileShape::Current().SetVecTile(5, 2048);
            Tensor tmp(dType, {5, M}, "tmp");
            auto len1 = GetTensorData(lens1, {idx});
            auto len2 = GetTensorData(lens2, {idx});
            auto aView = View(a, {5, M}, {5, len1}, {idx * 5, 0});
            auto bView = View(b, {5, M}, {5, len2 - len1}, {idx * 5, len1});
            auto cView = View(c, {5, M}, {5, M - len2}, {idx * 5, len2});
            Assemble({{Assign(aView), {0, 0}}, {Assign(bView), {0, len1}}, {Assign(cView), {0, len2}}}, tmp, true);
            tmp = Add(tmp, tmp);
            Assemble({{tmp, {idx * 5, 0}}}, dst, true);
        }
        config::SetPassOption(SG_SET_SCOPE, std::vector<int64_t>{-1, 0, 0});
    }

    std::cout << "compile finished" << std::endl;

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << "run finished" << std::endl;

    auto dstResult = ProgramData::GetInstance().GetOutputData(0);
    float eps = 1e-6f; // Compare results
    std::cout << "=======================dst===============================" << std::endl;
    EXPECT_TRUE(resultCmp(dstGolden, (T*)dstResult->data(), eps));
    for (int i = 0; i < ShapeSize(dst.GetShape()); i++) {
        auto actual = ((T*)dstResult->data())[i];
        auto expect = dstGolden[i];
        if (fabs(actual - expect) > eps) {
            std::cout << i << ": actual: " << actual << ", expect: " << expect << std::endl;
        }
    }
}

// 测试核内assemble，每次view一个矩形
TEST_F(AssembleTest, test_inner_assemble_multi_view_fp32) { TestInnerAssembleMultiView<float>(); }

// 测试核内assemble，每次view一个矩形
TEST_F(AssembleTest, test_inner_assemble_multi_view_fp16) { TestInnerAssembleMultiView<float16>(); }

// 测试核内assemble，每次view一个矩形
TEST_F(AssembleTest, test_inner_assemble_multi_view_bf16) { TestInnerAssembleMultiView<bfloat16>(); }
} // namespace
