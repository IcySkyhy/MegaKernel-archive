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
 * \file test_serialization.cpp
 * \brief Unit tests for serialization constants and Kind enum
 */

#include <string>
#include "gtest/gtest.h"
#include "interface/utils/serialization.h"

using namespace npu::tile_fwk;

class TestSerialization : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestSerialization, VersionString) { EXPECT_EQ(T_VERSION, "2.0"); }

TEST_F(TestSerialization, FieldKind) { EXPECT_EQ(T_FIELD_KIND, "kind"); }

TEST_F(TestSerialization, FieldRawtensor) { EXPECT_EQ(T_FIELD_RAWTENSOR, "rawtensor"); }

TEST_F(TestSerialization, KindValues)
{
    EXPECT_EQ(static_cast<int>(Kind::T_KIND_RAW_TENSOR), 0);
    EXPECT_EQ(static_cast<int>(Kind::T_KIND_TENSOR), 1);
    EXPECT_EQ(static_cast<int>(Kind::T_KIND_OPERATION), 2);
    EXPECT_EQ(static_cast<int>(Kind::T_KIND_FUNCTION), 3);
}
