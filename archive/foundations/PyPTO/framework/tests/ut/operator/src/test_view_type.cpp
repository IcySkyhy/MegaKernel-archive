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
 * \file test_view_type.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"
#include "operator/models/nsa/view_type.h"

using namespace npu::tile_fwk;

class ViewTypeUtest : public testing::Test {
public:
    static void SetUpTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false); }

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

TEST_F(ViewTypeUtest, add0_test)
{
    int64_t m = 4;
    int64_t k = 32;
    int64_t n = 1024;

    DataType originDtype = DT_INT8;
    DataType dstDtype = DT_FP32;
    float factor = (float)BytesOf(originDtype) / (float)BytesOf(dstDtype);

    std::vector<int64_t> xShape = {m, k, n};
    std::vector<int64_t> resultShape = {m, k, int(n * factor)};

    Tensor x(originDtype, xShape, "x");
    Tensor result(dstDtype, resultShape, "result");

    ViewTypeFunc(x, result, dstDtype);
}

TEST_F(ViewTypeUtest, quant_test)
{
    int64_t m = 64;
    int64_t k = 1;
    int64_t n = 512;

    DataType originDtype = DT_BF16;
    DataType dstDtype = DT_INT8;
    std::vector<int64_t> xShape = {m, k, n};
    std::vector<int64_t> resultShape = {m, k, n + 16};

    Tensor x(originDtype, xShape, "x");
    Tensor result(dstDtype, resultShape, "result");

    ViewTypeQuantTestFunc(x, result);
}

TEST_F(ViewTypeUtest, dequant_test)
{
    int64_t m = 2048;
    int64_t k = 1;
    int64_t n = 656;

    DataType originDtype = DT_INT8;
    DataType dstDtype = DT_BF16;
    std::vector<int64_t> xShape = {m, k, n};
    std::vector<int64_t> resultShape = {m, k, 512 + 64};

    Tensor x(originDtype, xShape, "x");
    Tensor result(dstDtype, resultShape, "result");

    ViewTypeDequantTestFunc(x, result);
}
