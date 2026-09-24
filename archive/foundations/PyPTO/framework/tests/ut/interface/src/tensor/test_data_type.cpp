/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_data_type.cpp
 * \brief Test cases for data type utilities
 */

#include "gtest/gtest.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

TEST(TestDataType, CalculateByteAlignedDataSize)
{
    EXPECT_EQ(DataSizeOf(8, DT_INT8), 8);
    EXPECT_EQ(DataSizeOf(8, DT_FP16), 16);
    EXPECT_EQ(DataSizeOf(8, DT_FP32), 32);
    EXPECT_EQ(DataSizeOf(8, DT_FP4_E2M1X2), 8);
}

TEST(TestDataType, CalculateSubByteDataSize)
{
    EXPECT_EQ(DataSizeOf(2, DT_INT4), 1);
    EXPECT_EQ(DataSizeOf(4, DT_HF4), 2);
    EXPECT_EQ(DataSizeOf(8, DT_FP4_E2M1), 4);
    EXPECT_EQ(DataSizeOf(12, DT_FP4_E1M2), 6);
}
