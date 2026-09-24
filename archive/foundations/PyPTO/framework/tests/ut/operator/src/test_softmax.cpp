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
 * \file test_softmax.cpp
 * \brief UT for Softmax and SoftmaxNew operators
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

class SoftmaxUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

TEST_F(SoftmaxUtest, softmax_fp32)
{
    TileShape::Current().SetVecTile({4, 64});
    std::vector<int64_t> shape = {4, 64};
    Tensor input(DT_FP32, shape, "input");

    FUNCTION("Softmax_fp32") { auto output = Softmax(input); }
}

TEST_F(SoftmaxUtest, softmax_bf16)
{
    TileShape::Current().SetVecTile({4, 64});
    std::vector<int64_t> shape = {4, 64};
    Tensor input(DT_BF16, shape, "input");

    FUNCTION("Softmax_bf16") { auto output = Softmax(input); }
}

TEST_F(SoftmaxUtest, softmax_new_fp32_no_cast)
{
    TileShape::Current().SetVecTile({4, 1, 1, 64});
    std::vector<int64_t> shape = {4, 4, 1, 64};
    Tensor input(DT_FP32, shape, "input");

    FUNCTION("SoftmaxNew_fp32") { auto output = SoftmaxNew(input); }
}

TEST_F(SoftmaxUtest, softmax_new_bf16_triggers_cast)
{
    TileShape::Current().SetVecTile({4, 1, 1, 64});
    std::vector<int64_t> shape = {2, 4, 1, 64};
    Tensor input(DT_BF16, shape, "input");

    FUNCTION("SoftmaxNew_bf16") { auto output = SoftmaxNew(input); }
}

TEST_F(SoftmaxUtest, softmax_new_fp16_triggers_cast)
{
    TileShape::Current().SetVecTile({4, 1, 1, 64});
    std::vector<int64_t> shape = {2, 4, 1, 64};
    Tensor input(DT_FP16, shape, "input");

    FUNCTION("SoftmaxNew_fp16") { auto output = SoftmaxNew(input); }
}
