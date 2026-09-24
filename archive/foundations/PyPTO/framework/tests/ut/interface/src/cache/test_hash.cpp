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
 * \file test_hash.cpp
 * \brief Unit tests for FunctionHash class (interface/cache/hash.h)
 */

#include <cstdint>
#include <climits>
#include <unordered_map>
#include "gtest/gtest.h"
#include "interface/cache/hash.h"

using namespace npu::tile_fwk;

class TestFunctionHash : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestFunctionHash, DefaultConstructor)
{
    FunctionHash h;
    EXPECT_TRUE(h.Empty());
    EXPECT_EQ(h.Data(), "");
    EXPECT_EQ(h.GetHash(), 0UL);
}

TEST_F(TestFunctionHash, NumericConstructor)
{
    FunctionHash h(12345UL);
    EXPECT_FALSE(h.Empty());
    EXPECT_EQ(h.Data(), "12345");
    EXPECT_EQ(h.GetHash(), 12345UL);
}

TEST_F(TestFunctionHash, NumericConstructorZero)
{
    FunctionHash h(0UL);
    EXPECT_FALSE(h.Empty());
    EXPECT_EQ(h.Data(), "0");
    EXPECT_EQ(h.GetHash(), 0UL);
}

TEST_F(TestFunctionHash, NumericConstructorMax)
{
    FunctionHash h(ULONG_MAX);
    EXPECT_FALSE(h.Empty());
    EXPECT_EQ(h.GetHash(), ULONG_MAX);
}

TEST_F(TestFunctionHash, CopyConstructor)
{
    FunctionHash src(9999UL);
    FunctionHash copy(src);
    EXPECT_EQ(copy.GetHash(), 9999UL);
    EXPECT_EQ(copy.Data(), "9999");
    EXPECT_FALSE(copy.Empty());
}

TEST_F(TestFunctionHash, AssignmentOperator)
{
    FunctionHash src(7777UL);
    FunctionHash dst;
    dst = src;
    EXPECT_EQ(dst.GetHash(), 7777UL);
    EXPECT_EQ(dst.Data(), "7777");
    EXPECT_FALSE(dst.Empty());
}

TEST_F(TestFunctionHash, OperatorEqualSameValue)
{
    FunctionHash a(100UL);
    FunctionHash b(100UL);
    EXPECT_TRUE(a == b);
}

TEST_F(TestFunctionHash, OperatorEqualDifferentValue)
{
    FunctionHash a(100UL);
    FunctionHash b(200UL);
    EXPECT_FALSE(a == b);
}

TEST_F(TestFunctionHash, StdHashConsistency)
{
    FunctionHash h(42UL);
    std::size_t hash1 = std::hash<FunctionHash>()(h);
    std::size_t hash2 = std::hash<FunctionHash>()(h);
    EXPECT_EQ(hash1, hash2);
}

TEST_F(TestFunctionHash, StdHashDifferentValuesProduceDifferentHashes)
{
    FunctionHash a(1UL);
    FunctionHash b(2UL);
    std::size_t hashA = std::hash<FunctionHash>()(a);
    std::size_t hashB = std::hash<FunctionHash>()(b);
    EXPECT_NE(hashA, hashB);
}

TEST_F(TestFunctionHash, StdHashInUnorderedMap)
{
    std::unordered_map<FunctionHash, int> map;
    FunctionHash k1(10UL);
    FunctionHash k2(20UL);
    map[k1] = 100;
    map[k2] = 200;
    EXPECT_EQ(map.size(), 2UL);
    auto it1 = map.find(k1);
    EXPECT_NE(it1, map.end());
    EXPECT_EQ(it1->second, 100);
    auto it2 = map.find(k2);
    EXPECT_NE(it2, map.end());
    EXPECT_EQ(it2->second, 200);
    FunctionHash k3(30UL);
    auto it3 = map.find(k3);
    EXPECT_EQ(it3, map.end());
}

TEST_F(TestFunctionHash, EmptyHashInUnorderedMap)
{
    std::unordered_map<FunctionHash, int> map;
    FunctionHash emptyKey;
    map[emptyKey] = -1;
    auto it = map.find(emptyKey);
    EXPECT_NE(it, map.end());
    EXPECT_EQ(it->second, -1);
}
