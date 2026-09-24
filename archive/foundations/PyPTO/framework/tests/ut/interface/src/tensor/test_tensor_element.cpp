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
 * \file test_tensor_element.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "tilefwk/element.h"

using namespace npu::tile_fwk;

TEST(TensorElement, DType)
{
    auto elem = Element(DT_INT8, 1L);
    EXPECT_TRUE(elem.IsSigned());
    EXPECT_EQ(elem.GetSignedData(), 1L);

    elem = Element(DT_UINT8, 1UL);
    EXPECT_TRUE(elem.IsUnsigned());
    EXPECT_EQ(elem.GetUnsignedData(), 1UL);

    elem = Element(DT_FP32, 1.0f);
    EXPECT_TRUE(elem.IsFloat());
    EXPECT_EQ(elem.GetFloatData(), 1.0f);
}

TEST(TensorElement, Calculate)
{
    auto lhs = Element(DT_INT8, 1L);
    auto rhs = Element(DT_INT8, 2L);

    EXPECT_EQ(lhs + rhs, Element(DT_INT8, 3L));
    EXPECT_EQ(rhs - lhs, Element(DT_INT8, 1L));
    EXPECT_EQ(lhs * rhs, Element(DT_INT8, 2L));
    EXPECT_EQ(rhs / lhs, Element(DT_INT8, 2L));
    EXPECT_EQ(rhs % lhs, Element(DT_INT8, 0L));

    EXPECT_TRUE(lhs != rhs);
    EXPECT_TRUE(lhs < rhs);
    EXPECT_TRUE(lhs <= rhs);

    EXPECT_FALSE(lhs == rhs);
    EXPECT_FALSE(lhs > rhs);
    EXPECT_FALSE(lhs >= rhs);
}
