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
 * \file test_monitor_util.cpp
 * \brief Unit tests for monitor_util.h formatting functions (FormatElapsed, PadRight, PadLabel, PadStageName,
 * PadElapsed)
 */

#include <string>
#include "gtest/gtest.h"
#include "interface/compiler_monitor/monitor_util.h"

using namespace npu::tile_fwk;

class TestFormatElapsed : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestFormatElapsed, LessThan60Seconds)
{
    EXPECT_EQ(FormatElapsed(0.0), "0.0s");
    EXPECT_EQ(FormatElapsed(1.5), "1.5s");
    EXPECT_EQ(FormatElapsed(30.0), "30.0s");
    EXPECT_EQ(FormatElapsed(59.9), "59.9s");
}

TEST_F(TestFormatElapsed, Exactly60Seconds)
{
    std::string result = FormatElapsed(60.0);
    EXPECT_NE(result.find("min"), std::string::npos);
    EXPECT_NE(result.find("(60s)"), std::string::npos);
}

TEST_F(TestFormatElapsed, Between60And3600Seconds)
{
    std::string result = FormatElapsed(125.0);
    EXPECT_NE(result.find("2min"), std::string::npos);
    EXPECT_NE(result.find("5s"), std::string::npos);
    EXPECT_NE(result.find("(125s)"), std::string::npos);
}

TEST_F(TestFormatElapsed, Exactly3600Seconds)
{
    std::string result = FormatElapsed(3600.0);
    EXPECT_NE(result.find("1h"), std::string::npos);
    EXPECT_NE(result.find("(3600s)"), std::string::npos);
}

TEST_F(TestFormatElapsed, Over3600Seconds)
{
    std::string result = FormatElapsed(7325.0);
    EXPECT_NE(result.find("2h"), std::string::npos);
    EXPECT_NE(result.find("(7325s)"), std::string::npos);
}

TEST_F(TestFormatElapsed, VerySmallValue) { EXPECT_EQ(FormatElapsed(0.001), "0.0s"); }

class TestPadRight : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestPadRight, ShortStringPadded)
{
    std::string result = PadRight("abc", 6);
    EXPECT_EQ(result, "abc   ");
}

TEST_F(TestPadRight, ExactWidthNoPadding)
{
    std::string result = PadRight("abcdef", 6);
    EXPECT_EQ(result, "abcdef");
}

TEST_F(TestPadRight, LongerThanWidthUnchanged)
{
    std::string result = PadRight("abcdefghij", 6);
    EXPECT_EQ(result, "abcdefghij");
}

TEST_F(TestPadRight, EmptyStringPadded)
{
    std::string result = PadRight("", 4);
    EXPECT_EQ(result, "    ");
}

TEST_F(TestPadRight, ZeroWidth)
{
    std::string result = PadRight("abc", 0);
    EXPECT_EQ(result, "abc");
}

class TestPadSpecialized : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestPadSpecialized, PadElapsed)
{
    std::string result = PadElapsed("1.5s");
    EXPECT_EQ(static_cast<int>(result.size()), MONITOR_ELAPSED_WIDTH);
}

TEST_F(TestPadSpecialized, PadLabel)
{
    std::string result = PadLabel("Function: ");
    EXPECT_EQ(static_cast<int>(result.size()), MONITOR_LABEL_WIDTH);
}

TEST_F(TestPadSpecialized, PadStageName)
{
    std::string result = PadStageName("Pass");
    EXPECT_EQ(static_cast<int>(result.size()), MONITOR_STAGE_NAME_WIDTH);
}

TEST_F(TestPadSpecialized, PadElapsedExactWidth)
{
    std::string eightChars = "12345678";
    std::string result = PadElapsed(eightChars);
    EXPECT_EQ(result, eightChars);
}

TEST_F(TestPadSpecialized, PadLabelExactWidth)
{
    std::string label(MONITOR_LABEL_WIDTH, 'x');
    std::string result = PadLabel(label);
    EXPECT_EQ(result, label);
}

TEST_F(TestPadSpecialized, PadStageNameExactWidth)
{
    std::string name(MONITOR_STAGE_NAME_WIDTH, 'y');
    std::string result = PadStageName(name);
    EXPECT_EQ(result, name);
}
