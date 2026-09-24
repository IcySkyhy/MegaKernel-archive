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
 * \file test_string_utils_extended.cpp
 * \brief Extended unit tests for StringUtils (Trim, Split, EndsWith, BaseName, ToLower, ToUpper, AppendUniqueToken,
 * DataCopy, DataSet, ToString)
 */

#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "interface/utils/string_utils.h"

using namespace npu::tile_fwk;

class TestStringUtilsExt : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestStringUtilsExt, TrimLeadingSpaces)
{
    std::string s = "   hello";
    StringUtils::Trim(s);
    EXPECT_EQ(s, "hello");
}

TEST_F(TestStringUtilsExt, TrimTrailingSpaces)
{
    std::string s = "hello   ";
    StringUtils::Trim(s);
    EXPECT_EQ(s, "hello");
}

TEST_F(TestStringUtilsExt, TrimBothSides)
{
    std::string s = "  hello world  ";
    StringUtils::Trim(s);
    EXPECT_EQ(s, "hello world");
}

TEST_F(TestStringUtilsExt, TrimTabs)
{
    std::string s = "\thello\t";
    StringUtils::Trim(s);
    EXPECT_EQ(s, "hello");
}

TEST_F(TestStringUtilsExt, TrimEmptyString)
{
    std::string s = "";
    StringUtils::Trim(s);
    EXPECT_EQ(s, "");
}

TEST_F(TestStringUtilsExt, TrimAllWhitespace)
{
    std::string s = "   \t  ";
    StringUtils::Trim(s);
    EXPECT_EQ(s, "");
}

TEST_F(TestStringUtilsExt, TrimNoWhitespace)
{
    std::string s = "hello";
    StringUtils::Trim(s);
    EXPECT_EQ(s, "hello");
}

TEST_F(TestStringUtilsExt, SplitBasic)
{
    auto result = StringUtils::Split("a,b,c", ",");
    EXPECT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], "a");
    EXPECT_EQ(result[1], "b");
    EXPECT_EQ(result[2], "c");
}

TEST_F(TestStringUtilsExt, SplitEmptyString)
{
    auto result = StringUtils::Split("", ",");
    EXPECT_EQ(result.size(), 0u);
}

TEST_F(TestStringUtilsExt, SplitEmptyPattern)
{
    auto result = StringUtils::Split("abc", "");
    EXPECT_EQ(result.size(), 0u);
}

TEST_F(TestStringUtilsExt, SplitNoMatch)
{
    auto result = StringUtils::Split("abc", ",");
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], "abc");
}

TEST_F(TestStringUtilsExt, SplitSingleChar)
{
    auto result = StringUtils::Split("x", ",");
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], "x");
}

TEST_F(TestStringUtilsExt, SplitMultiCharPattern)
{
    auto result = StringUtils::Split("a::b::c", "::");
    EXPECT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], "a");
    EXPECT_EQ(result[1], "b");
    EXPECT_EQ(result[2], "c");
}

TEST_F(TestStringUtilsExt, EndsWithBasic)
{
    EXPECT_TRUE(StringUtils::EndsWith("hello", "lo"));
    EXPECT_TRUE(StringUtils::EndsWith("hello", "hello"));
    EXPECT_TRUE(StringUtils::EndsWith("hello", ""));
    EXPECT_FALSE(StringUtils::EndsWith("hello", "world"));
    EXPECT_FALSE(StringUtils::EndsWith("hi", "hello"));
}

TEST_F(TestStringUtilsExt, BaseNameWithPath)
{
    std::string result = StringUtils::BaseName("/path/to/file.cpp");
    EXPECT_EQ(result, "file.cpp");
}

TEST_F(TestStringUtilsExt, BaseNameNoPath)
{
    std::string result = StringUtils::BaseName("file.cpp");
    EXPECT_EQ(result, "file.cpp");
}

TEST_F(TestStringUtilsExt, BaseNameJustSlash)
{
    std::string result = StringUtils::BaseName("/file.cpp");
    EXPECT_EQ(result, "file.cpp");
}

TEST_F(TestStringUtilsExt, ToLower)
{
    EXPECT_EQ(StringUtils::ToLower("HELLO"), "hello");
    EXPECT_EQ(StringUtils::ToLower("MiXeD"), "mixed");
    EXPECT_EQ(StringUtils::ToLower("already"), "already");
    EXPECT_EQ(StringUtils::ToLower(""), "");
}

TEST_F(TestStringUtilsExt, ToUpper)
{
    EXPECT_EQ(StringUtils::ToUpper("hello"), "HELLO");
    EXPECT_EQ(StringUtils::ToUpper("MiXeD"), "MIXED");
    EXPECT_EQ(StringUtils::ToUpper("ALREADY"), "ALREADY");
    EXPECT_EQ(StringUtils::ToUpper(""), "");
}

TEST_F(TestStringUtilsExt, DataCopyBasic)
{
    char src[] = "abcdefghij";
    char dest[10] = {};
    StringUtils::DataCopy(dest, 10, src, 10);
    EXPECT_EQ(memcmp(dest, src, 10), 0);
}

TEST_F(TestStringUtilsExt, DataCopyPartial)
{
    char src[] = "hello world";
    char dest[5] = {};
    StringUtils::DataCopy(dest, 5, src, 5);
    EXPECT_EQ(memcmp(dest, "hello", 5), 0);
}

TEST_F(TestStringUtilsExt, DataSetBasic)
{
    char buf[10] = {};
    StringUtils::DataSet(buf, 10, 0xAA, 10);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(buf[i], (char)0xAA);
    }
}

TEST_F(TestStringUtilsExt, DataSetPartial)
{
    char buf[10] = {};
    StringUtils::DataSet(buf, 10, 'X', 5);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(buf[i], 'X');
    }
    for (int i = 5; i < 10; i++) {
        EXPECT_EQ(buf[i], 0);
    }
}

TEST_F(TestStringUtilsExt, ToStringIntVector)
{
    std::vector<int> vec = {1, 2, 3};
    std::string result = StringUtils::ToString(vec);
    EXPECT_NE(result.find("1"), std::string::npos);
    EXPECT_NE(result.find("2"), std::string::npos);
    EXPECT_NE(result.find("3"), std::string::npos);
}

TEST_F(TestStringUtilsExt, AppendUniqueTokenFirst)
{
    std::string acc = "";
    StringUtils::AppendUniqueToken(acc, "alpha");
    EXPECT_EQ(acc, "alpha");
}

TEST_F(TestStringUtilsExt, AppendUniqueTokenSecond)
{
    std::string acc = "alpha";
    StringUtils::AppendUniqueToken(acc, "beta");
    EXPECT_EQ(acc, "alpha;beta");
}

TEST_F(TestStringUtilsExt, AppendUniqueTokenDuplicate)
{
    std::string acc = "alpha;beta";
    StringUtils::AppendUniqueToken(acc, "alpha");
    EXPECT_EQ(acc, "alpha;beta");
}

TEST_F(TestStringUtilsExt, AppendUniqueTokenEmptyToken)
{
    std::string acc = "alpha";
    StringUtils::AppendUniqueToken(acc, "");
    EXPECT_EQ(acc, "alpha");
}

TEST_F(TestStringUtilsExt, AppendUniqueTokenThreeDistinct)
{
    std::string acc = "";
    StringUtils::AppendUniqueToken(acc, "a");
    StringUtils::AppendUniqueToken(acc, "b");
    StringUtils::AppendUniqueToken(acc, "c");
    EXPECT_EQ(acc, "a;b;c");
}

TEST_F(TestStringUtilsExt, StartsWithExtended)
{
    EXPECT_TRUE(StringUtils::StartsWith("hello", "h"));
    EXPECT_TRUE(StringUtils::StartsWith("hello", "he"));
    EXPECT_FALSE(StringUtils::StartsWith("hello", "x"));
    EXPECT_FALSE(StringUtils::StartsWith("short", "longer"));
}

TEST_F(TestStringUtilsExt, VectorOstreamOperator)
{
    std::vector<int> vec = {10, 20, 30};
    std::ostringstream oss;
    oss << vec;
    EXPECT_NE(oss.str().find("10"), std::string::npos);
    EXPECT_NE(oss.str().find("20"), std::string::npos);
    EXPECT_NE(oss.str().find("30"), std::string::npos);
}
