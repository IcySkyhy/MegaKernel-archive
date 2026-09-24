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
 * \file test_rms_norm.cpp
 * \brief UT for RmsNorm operator (both overloads)
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

class RmsNormUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

TEST_F(RmsNormUtest, rms_norm_without_gamma_fp32)
{
    TileShape::Current().SetVecTile({1, 128});
    std::vector<int64_t> shape = {4, 128};
    Tensor input(DT_FP32, shape, "input");

    FUNCTION("RmsNormNoGamma_fp32") { auto output = RmsNorm(input); }
}

TEST_F(RmsNormUtest, rms_norm_without_gamma_bf16)
{
    TileShape::Current().SetVecTile({1, 128});
    std::vector<int64_t> shape = {2, 128};
    Tensor input(DT_BF16, shape, "input");

    FUNCTION("RmsNormNoGamma_bf16") { auto output = RmsNorm(input); }
}

TEST_F(RmsNormUtest, rms_norm_with_gamma_fp32)
{
    TileShape::Current().SetVecTile({1, 128});
    std::vector<int64_t> shape = {4, 128};
    Tensor input(DT_FP32, shape, "input");
    Tensor gamma(DT_FP32, {128}, "gamma");
    float epsilon = 1e-6f;

    FUNCTION("RmsNormWithGamma_fp32") { auto output = RmsNorm(input, gamma, epsilon); }
}

TEST_F(RmsNormUtest, rms_norm_with_gamma_bf16)
{
    TileShape::Current().SetVecTile({1, 128});
    std::vector<int64_t> shape = {2, 128};
    Tensor input(DT_BF16, shape, "input");
    Tensor gamma(DT_BF16, {128}, "gamma");
    float epsilon = 1e-5f;

    FUNCTION("RmsNormWithGamma_bf16") { auto output = RmsNorm(input, gamma, epsilon); }
}

TEST_F(RmsNormUtest, rms_norm_with_gamma_custom_epsilon)
{
    TileShape::Current().SetVecTile({1, 64});
    std::vector<int64_t> shape = {8, 64};
    Tensor input(DT_FP32, shape, "input");
    Tensor gamma(DT_FP32, {64}, "gamma");

    FUNCTION("RmsNormWithGamma_customEps") { auto output = RmsNorm(input, gamma, 1e-3f); }
}
