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
 * \file test_quant.cpp
 * \brief UT for Quant and Matrix::QuantMM operators
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

class QuantUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

TEST_F(QuantUtest, quant_symmetric_no_smooth_fp32)
{
    TileShape::Current().SetVecTile({4, 128});
    std::vector<int64_t> shape = {4, 128};
    Tensor input(DT_FP32, shape, "input");

    FUNCTION("QuantSymNoSmooth_fp32") { auto result = Quant(input); }
}

TEST_F(QuantUtest, quant_symmetric_no_smooth_bf16)
{
    TileShape::Current().SetVecTile({4, 128});
    std::vector<int64_t> shape = {8, 128};
    Tensor input(DT_BF16, shape, "input");

    FUNCTION("QuantSymNoSmooth_bf16") { auto result = Quant(input, true, false, Tensor()); }
}

TEST_F(QuantUtest, quant_symmetric_with_smooth)
{
    TileShape::Current().SetVecTile({4, 128});
    std::vector<int64_t> shape = {4, 128};
    Tensor input(DT_FP32, shape, "input");
    Tensor smoothFactor(DT_FP32, shape, "smoothFactor");

    FUNCTION("QuantSymWithSmooth") { auto result = Quant(input, true, true, smoothFactor); }
}

TEST_F(QuantUtest, quant_asymmetric_no_smooth)
{
    TileShape::Current().SetVecTile({4, 128});
    std::vector<int64_t> shape = {4, 128};
    Tensor input(DT_FP32, shape, "input");

    FUNCTION("QuantAsymNoSmooth") { auto result = Quant(input, false, false, Tensor()); }
}

TEST_F(QuantUtest, quant_asymmetric_with_smooth)
{
    TileShape::Current().SetVecTile({4, 128});
    std::vector<int64_t> shape = {4, 128};
    Tensor input(DT_FP32, shape, "input");
    Tensor smoothFactor(DT_FP32, shape, "smoothFactor");

    FUNCTION("QuantAsymWithSmooth") { auto result = Quant(input, false, true, smoothFactor); }
}

class QuantMMUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

TEST_F(QuantMMUtest, quant_mm_2d)
{
    TileShape::Current().SetVecTile({4, 64});
    TileShape::Current().SetCubeTile({16, 16}, {32, 32, 32}, {32, 32});
    int64_t m = 32;
    int64_t k = 128;
    int64_t n = 64;

    Tensor operand1(DT_BF16, {m, k}, "operand1");
    Tensor operand2(DT_INT8, {k, n}, "operand2");
    Tensor dequantScaleW(DT_FP32, {1, n}, "dequantScaleW");

    FUNCTION("QuantMM_2d") { auto output = Matrix::QuantMM(operand1, operand2, dequantScaleW); }
}

TEST_F(QuantMMUtest, quant_mm_3d)
{
    TileShape::Current().SetVecTile({1, 4, 64});
    TileShape::Current().SetCubeTile({16, 16}, {32, 32, 32}, {32, 32});
    int64_t b = 2;
    int64_t m = 32;
    int64_t k = 128;
    int64_t n = 64;

    Tensor operand1(DT_BF16, {b, m, k}, "operand1");
    Tensor operand2(DT_INT8, {b, k, n}, "operand2");
    Tensor dequantScaleW(DT_FP32, {1, n}, "dequantScaleW");

    FUNCTION("QuantMM_3d") { auto output = Matrix::QuantMM(operand1, operand2, dequantScaleW); }
}

TEST_F(QuantMMUtest, quant_mm_4d_invalid_asserts)
{
    TileShape::Current().SetVecTile({1, 1, 4, 64});
    TileShape::Current().SetCubeTile({16, 16}, {32, 32, 32}, {32, 32});

    Tensor operand1(DT_BF16, {2, 32, 128, 4}, "operand1");
    Tensor operand2(DT_INT8, {128, 64}, "operand2");
    Tensor dequantScaleW(DT_FP32, {1, 64}, "dequantScaleW");

    EXPECT_ANY_THROW({
        FUNCTION("QuantMM_4d_invalid") { auto output = Matrix::QuantMM(operand1, operand2, dequantScaleW);
}
});
}
