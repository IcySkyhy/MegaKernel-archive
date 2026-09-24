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
 * \file test_logical_tensor.cpp
 * \brief Test cases for LogicalTensor class with error codes
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/tensormap.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

class TestLogicalTensor : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "TestLogicalTensor SetUpTestCase" << std::endl; }
    static void TearDownTestCase() { std::cout << "TestLogicalTensor TearDownTestCase" << std::endl; }
    void SetUp() override { std::cout << "TestLogicalTensor SetUp" << std::endl; }
    void TearDown() override { std::cout << "TestLogicalTensor TearDown" << std::endl; }
};

TEST_F(TestLogicalTensor, ViewDimensionMismatch)
{
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor input(DT_FP32, shape, "input");

    auto storage = input.GetStorage(false);
    std::vector<int64_t> newShape = {8, 8};
    std::vector<int64_t> newOffset = {0, 0};

    EXPECT_THROW(storage->View(*Program::GetInstance().GetCurrentFunction(), newShape, newOffset), std::exception);
}

TEST_F(TestLogicalTensor, ViewOffsetMismatch)
{
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor input(DT_FP32, shape, "input");

    auto storage = input.GetStorage(false);
    std::vector<int64_t> newShape = {8, 8, 8};
    std::vector<int64_t> newOffset = {0, 0};

    EXPECT_THROW(storage->View(*Program::GetInstance().GetCurrentFunction(), newShape, newOffset), std::exception);
}

TEST_F(TestLogicalTensor, ViewShapeOutOfBounds)
{
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor input(DT_FP32, shape, "input");

    auto storage = input.GetStorage(false);
    std::vector<int64_t> newShape = {20, 8, 8};
    std::vector<int64_t> newOffset = {0, 0, 0};

    EXPECT_THROW(storage->View(*Program::GetInstance().GetCurrentFunction(), newShape, newOffset), std::exception);
}

TEST_F(TestLogicalTensor, PackedDataSize)
{
    Tensor input(DT_FP4_E2M1, {2, 4}, "input");

    auto storage = input.GetStorage(false);
    storage->SetMemoryTypeToBe(MEM_L2);

    EXPECT_EQ(storage->GetDataSize(), 4);
    EXPECT_EQ(storage->GetRawTensor()->GetRawDataSize(), 4);
    EXPECT_EQ(storage->MemorySize(), 4);
}
