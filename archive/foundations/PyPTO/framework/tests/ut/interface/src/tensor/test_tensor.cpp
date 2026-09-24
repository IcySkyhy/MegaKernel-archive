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
 * \file test_tensor.cpp
 * \brief
 */

#include <cstddef>
#include "gtest/gtest.h"
#include "interface/tensor/tensormap.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

class TestTensor : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestTensor, GetShapeTest)
{
    std::vector<int64_t> tshape = {16, 32, 16, 64};
    std::vector<int64_t> kshape = {};
    Tensor a(DT_FP32, tshape, "A");
    Tensor b(DT_FP32, kshape, "B");

    auto ashape = a.GetShape(1);
    EXPECT_EQ(ashape, 32);
    ashape = a.GetShape(-4);
    EXPECT_EQ(ashape, 16);

    auto validShape = a.GetValidShape();
    for (size_t i = 0; i < tshape.size(); i++) {
        EXPECT_EQ(validShape[i].Concrete(), tshape[i]);
    }
}

TEST_F(TestTensor, GetCachePolicyTest)
{
    std::vector<int64_t> tshape = {4, 4};
    Tensor t1(DT_FP32, tshape, "T1");

    EXPECT_FALSE(t1.GetCachePolicy(CachePolicy::PREFETCH));
    EXPECT_FALSE(t1.GetCachePolicy(CachePolicy::NONE_CACHEABLE));

    t1.SetCachePolicy(CachePolicy::PREFETCH, true);
    EXPECT_TRUE(t1.GetCachePolicy(CachePolicy::PREFETCH));
    EXPECT_FALSE(t1.GetCachePolicy(CachePolicy::NONE_CACHEABLE));

    t1.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    EXPECT_TRUE(t1.GetCachePolicy(CachePolicy::PREFETCH));
    EXPECT_FALSE(t1.GetCachePolicy(CachePolicy::NONE_CACHEABLE));
}

TEST_F(TestTensor, RawTensorNegative)
{
    RawTensor t(DT_INT8, {});
    t.SetRefCount(-2);
    t.AddRefCount(1);
    EXPECT_EQ(t.GetRefCount(), -1);
}

TEST_F(TestTensor, GetShapeNoDimensions)
{
    std::vector<int64_t> shape = {};
    Tensor a(DT_FP32, shape, "A");

    EXPECT_THROW(a.GetShape(0), std::exception);
}

TEST_F(TestTensor, GetShapeAxisOutOfRange)
{
    std::vector<int64_t> shape = {16, 32, 16};
    Tensor a(DT_FP32, shape, "A");

    EXPECT_THROW(a.GetShape(10), std::exception);
    EXPECT_THROW(a.GetShape(-10), std::exception);
}

TEST_F(TestTensor, InvalidShapeValue)
{
    std::vector<int64_t> shape = {-2, 16};
    EXPECT_THROW(Tensor(DT_FP32, shape, "A"), std::exception);
}
