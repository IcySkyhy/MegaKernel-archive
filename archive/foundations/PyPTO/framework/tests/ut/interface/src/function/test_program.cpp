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
 * \file test_program.cpp
 * \brief Test cases for Program class with error codes
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

class TestProgram : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "TestProgram SetUpTestCase" << std::endl; }
    static void TearDownTestCase() { std::cout << "TestProgram TearDownTestCase" << std::endl; }
    void SetUp() override { std::cout << "TestProgram SetUp" << std::endl; }
    void TearDown() override { std::cout << "TestProgram TearDown" << std::endl; }
};

TEST_F(TestProgram, AddOperationWithoutActiveFunction)
{
    std::vector<int64_t> shape = {16, 16};
    Tensor input(DT_FP32, shape, "input");

    EXPECT_THROW(Add(input, input), std::exception);
}

TEST_F(TestProgram, GetFunctionByMagicNotFound)
{
    auto func = Program::GetInstance().GetFunctionByMagic(999999);
    EXPECT_EQ(func, nullptr);
}

TEST_F(TestProgram, DumpJsonFileInvalidPath)
{
    EXPECT_THROW(Program::GetInstance().DumpJsonFile("/invalid/path/that/does/not/exist.json"), std::exception);
}
