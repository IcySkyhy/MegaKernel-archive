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
 * \file test_view_operation.cpp
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
class ViewTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
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

template <typename T>
void TestInnerView()
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
    auto SimuResult = [&row](const std::vector<T>& a, const std::vector<T>& b) {
        ASSERT(a.size() == N * M);
        std::vector<T> out(N * M, -5.0f);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < M; j++) {
                out[i * 5 * M + j] = 2 * (a[i * 5 * M + j] + b[i * 5 * M + j]);
            }
        }
        return out;
    };

    Tensor a(dType, {N, M}, "a");
    Tensor b(dType, {N, M}, "b");
    Tensor dst(dType, {N, M}, "dst");

    std::vector<T> aData(ShapeSize(a.GetShape()), 0);
    for (size_t i = 0; i < aData.size(); i++) {
        aData[i] = uniform(gen);
    }
    std::vector<T> bData(ShapeSize(b.GetShape()), 0);
    for (size_t i = 0; i < bData.size(); i++) {
        bData[i] = uniform(gen);
    }
    auto dstGolden = SimuResult(aData, bData);
    std::cout << "simu finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(a, aData),
        RawTensorData::CreateTensor<T>(b, bData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(dst, -5.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<T>(dst, dstGolden),
    });

    FUNCTION("test", {a, b}, {dst})
    {
        config::SetPassOption(SG_SET_SCOPE, std::vector<int64_t>{1, 1, 0});
        LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(row))
        {
            TileShape::Current().SetVecTile(5, 2048);
            auto d = Add(a, b);
            TileShape::Current().SetVecTile(1, 2048);
            auto e = View(d, {1, M}, {5 * idx, 0});
            auto f = Add(e, e);
            Assemble({{f, {5 * idx, 0}}}, dst, true);
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

// 测试核内view
TEST_F(ViewTest, test_inner_view_fp32) { TestInnerView<float>(); }

} // namespace
