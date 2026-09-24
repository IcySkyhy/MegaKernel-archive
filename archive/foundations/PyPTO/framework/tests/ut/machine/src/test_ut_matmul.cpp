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
 * \file test_ut_matmul.cpp
 * \brief Unit test for pass manager.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"

using namespace npu::tile_fwk;

namespace {

class DynamicMatmulUTest : public testing::Test {};

template <typename T_inputDtype, typename T_outputDtype, typename T_matrixInputs>
struct MatmulImpl {
    using inputDtype = T_inputDtype;
    using outputDtype = T_outputDtype;
    using cfg = T_matrixInputs;
};

template <bool T_transA, bool T_transB, bool T_isANz, bool T_isBNz, bool T_isCNz>
struct MatrixInputs {
    static constexpr bool transA = T_transA;
    static constexpr bool transB = T_transB;
    static constexpr bool isANz = T_isANz;
    static constexpr bool isBNz = T_isBNz;
    static constexpr bool isCNz = T_isCNz;
};

template <typename T>
DataType GetAstDtype()
{
    DataType astDtype = DataType::DT_BOTTOM;
    if constexpr (std::is_same<T, npu::tile_fwk::float16>::value) {
        astDtype = DataType::DT_FP16;
    }
    if constexpr (std::is_same<T, float>::value) {
        astDtype = DataType::DT_FP32;
    }
    if constexpr (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        astDtype = DataType::DT_BF16;
    }
    if constexpr (std::is_same<T, int8_t>::value) {
        astDtype = DT_INT8;
    }
    if constexpr (std::is_same<T, int32_t>::value) {
        astDtype = DT_INT32;
    }
    EXPECT_NE(astDtype, DT_BOTTOM);
    return astDtype;
}

template <typename MatmulImplType>
void TestDynMatmul(int m, int k, int n, Matrix::MatmulExtendParam param = {})
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    int nb = n;
    int kb = k;
    int ka = k;
    int ma = m;
    if constexpr (MatmulImplType::cfg::transB) {
        std::swap(nb, kb);
    }
    if constexpr (MatmulImplType::cfg::transA) {
        std::swap(ka, ma);
    }
    std::vector<int64_t> shape_c = {m, n};
    std::vector<int64_t> shape_b = {kb, nb};
    std::vector<int64_t> shape_a = {ma, ka};

    auto InputUTDtype = GetAstDtype<typename MatmulImplType::inputDtype>();
    auto OutputUTDtype = GetAstDtype<typename MatmulImplType::outputDtype>();

    auto afmt = MatmulImplType::cfg::isANz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    auto bfmt = MatmulImplType::cfg::isBNz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    auto cfmt = MatmulImplType::cfg::isCNz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor tensor_a(InputUTDtype, shape_a, "tensor_a", afmt);
    Tensor tensor_b(InputUTDtype, shape_b, "tensor_b", bfmt);
    Tensor tensor_c(OutputUTDtype, shape_c, "tensor_c", cfmt);
    FUNCTION("test_dyn_mm", {tensor_a, tensor_b}, {tensor_c})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1))
        {
            Tensor dyn_a = View(tensor_a, {ma, ka}, {ma, ka}, {batchId * ma, 0});
            Tensor dyn_b = View(tensor_b, {kb, nb}, {kb, nb}, {0, 0});
            tensor_c = Matrix::Matmul(OutputUTDtype, dyn_a, dyn_b, param, MatmulImplType::cfg::transA,
                                      MatmulImplType::cfg::transB, MatmulImplType::cfg::isCNz);
        }
    }
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_bf16)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    using TestMatmulType = MatmulImpl<float, float, MatrixInputs<false, false, false, false, false>>;
    Matrix::MatmulExtendParam param;
    TestDynMatmul<TestMatmulType>(m, k, n, param);
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_bf16_C_NZ)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    using TestMatmulType = MatmulImpl<float, float, MatrixInputs<false, false, false, false, true>>;
    Matrix::MatmulExtendParam param;
    TestDynMatmul<TestMatmulType>(m, k, n, param);
}

TEST_F(DynamicMatmulUTest, mm_A_NZ_B_NZ_int8_C_NZ)
{
    TileShape::Current().SetCubeTile({64, 128}, {64, 128}, {64, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    using TestMatmulType = MatmulImpl<int8_t, int32_t, MatrixInputs<true, true, true, true, true>>;
    Matrix::MatmulExtendParam param;
    TestDynMatmul<TestMatmulType>(m, k, n, param);
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_KSplit_bf16)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128}, true);
    int m = 128;
    int k = 256;
    int n = 512;
    using TestMatmulType = MatmulImpl<float, float, MatrixInputs<false, false, false, false, false>>;
    Matrix::MatmulExtendParam param;
    TestDynMatmul<TestMatmulType>(m, k, n, param);
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_pertensor)
{
    int m = 128;
    int n = 512;
    int k = 256;
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Matrix::MatmulExtendParam param;
    param.scaleValue = 2.0f;
    using TestMatmulType = MatmulImpl<int8_t, npu::tile_fwk::float16, MatrixInputs<false, false, false, false, false>>;
    TestDynMatmul<TestMatmulType>(m, k, n, param);
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_perchannel_with_bias)
{
    int m = 128;
    int n = 512;
    int k = 256;
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Matrix::MatmulExtendParam param;
    param.biasTensor = Tensor(DT_INT32, {1, n}, "bias_tensor", TileOpFormat::TILEOP_ND);
    param.scaleTensor = Tensor(DT_UINT64, {1, n}, "scale_tensor", TileOpFormat::TILEOP_ND);
    using TestMatmulType = MatmulImpl<int8_t, npu::tile_fwk::float16, MatrixInputs<false, false, false, false, false>>;
    TestDynMatmul<TestMatmulType>(m, k, n, param);
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_config)
{
    std::shared_ptr<ConfigScope> scope = ConfigManagerNg::GetInstance().CurrentScope();
    ASSERT(scope != nullptr);
    const TileShape& tileScope = scope->GenerateTileShape();
    if (tileScope.GetCubeTile().enableSplitK == false) {
        return;
    }
}

TEST_F(DynamicMatmulUTest, transposed_batchmatmul_test)
{
    int64_t b = 4;
    int64_t m = 16;
    int64_t k = 128;
    int64_t n = 256;
    std::vector<int64_t> shape_a = {m, b, k};
    std::vector<int64_t> shape_b = {b, k, n};
    std::vector<int64_t> shape_c = {m, b, n};

    Tensor tensor_a(DataType::DT_BF16, shape_a, "tensor_a", TileOpFormat::TILEOP_ND);
    Tensor tensor_b(DataType::DT_BF16, shape_b, "tensor_b", TileOpFormat::TILEOP_ND);
    Tensor tensor_c(DataType::DT_BF16, shape_c, "tensor_c", TileOpFormat::TILEOP_ND);

    FUNCTION("test_transposed_batch_mm", {tensor_a, tensor_b}, {tensor_c})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1))
        {
            (void)batchId;
            TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
            tensor_c = Matrix::TransposedBatchMatmul(DataType::DT_BF16, tensor_a, tensor_b);
        }
    }
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_tf32_round)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    using TestMatmulType = MatmulImpl<float, float, MatrixInputs<false, false, false, false, true>>;
    Matrix::MatmulExtendParam param;
    param.transMode = Matrix::TransMode::CAST_ROUND;
    TestDynMatmul<TestMatmulType>(m, k, n, param);
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_tf32_rint)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    using TestMatmulType = MatmulImpl<float, float, MatrixInputs<false, false, false, false, true>>;
    Matrix::MatmulExtendParam param;
    param.transMode = Matrix::TransMode::CAST_RINT;
    TestDynMatmul<TestMatmulType>(m, k, n, param);
}

TEST_F(DynamicMatmulUTest, mm_A_B_ND_KSplit_int8)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128}, true);
    int m = 128;
    int k = 256;
    int n = 512;
    using TestMatmulType = MatmulImpl<int8_t, int32_t, MatrixInputs<false, false, false, false, false>>;
    Matrix::MatmulExtendParam param;
    TestDynMatmul<TestMatmulType>(m, k, n, param);
}

} // namespace
