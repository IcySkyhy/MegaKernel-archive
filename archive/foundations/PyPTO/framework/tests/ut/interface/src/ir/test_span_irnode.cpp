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
 * \file test_span_irnode.cpp
 * \brief Unit tests for IR core classes (Span and IRNode)
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>

#include "ir/scalar_expr.h"

namespace pypto {
namespace ir {

TEST(IRCoreTest, TestSpanBasic)
{
    // Test basic Span functionality
    Span span("test.py", 10, 5, 10, 15);

    ASSERT_EQ(span.Filename(), "test.py");
    ASSERT_EQ(span.BeginLine(), 10);
    ASSERT_EQ(span.BeginColumn(), 5);
    ASSERT_EQ(span.EndLine(), 10);
    ASSERT_EQ(span.EndColumn(), 15);
}

TEST(IRCoreTest, TestSpanToString)
{
    // Test Span to_string() method
    Span span("test.py", 10, 5, 10, 15);
    std::string str = span.ToString();

    // Should contain meaningful location information
    ASSERT_TRUE(!str.empty());
    ASSERT_TRUE(str.find("test.py") != std::string::npos || str.find("10") != std::string::npos);
}

TEST(IRCoreTest, TestIRNodeBasic)
{
    // Test basic IRNode functionality via ConstInt (IRNode is now abstract)
    Span span("test.py", 1, 0, 1, 10);
    auto node = std::make_shared<ConstInt>(42, DataType::INT32, span);

    ASSERT_EQ(node->span_.Filename(), span.Filename());
    ASSERT_EQ(node->span_.BeginLine(), span.BeginLine());
}

TEST(IRCoreTest, TestIRNodeTypeName)
{
    // Test IRNode TypeName() method via ConstInt
    Span span("test.py", 1, 0, 1, 10);
    auto node = std::make_shared<ConstInt>(42, DataType::INT32, span);

    std::string typeName = node->TypeName();
    ASSERT_EQ(typeName, "ConstInt");
}

TEST(IRCoreTest, TestMultipleIRNodes)
{
    // Test creating multiple IRNode instances via ConstInt
    Span span1("file1.py", 1, 0, 1, 10);
    Span span2("file2.py", 5, 5, 5, 15);
    Span span3("file3.py", 10, 0, 12, 0);

    auto node1 = std::make_shared<ConstInt>(1, DataType::INT32, span1);
    auto node2 = std::make_shared<ConstInt>(2, DataType::INT32, span2);
    auto node3 = std::make_shared<ConstInt>(3, DataType::INT32, span3);

    ASSERT_EQ(node1->span_.Filename(), "file1.py");
    ASSERT_EQ(node2->span_.Filename(), "file2.py");
    ASSERT_EQ(node3->span_.Filename(), "file3.py");
}

TEST(IRCoreTest, TestSpanComparison)
{
    // Test comparing Span objects
    Span span1("test.py", 10, 5, 10, 15);
    Span span2("test.py", 10, 5, 10, 15);
    Span span3("test.py", 20, 5, 20, 15);

    // Same location
    ASSERT_EQ(span1.BeginLine(), span2.BeginLine());
    ASSERT_NE(span1.BeginLine(), span3.BeginLine());
}

TEST(IRCoreTest, TestIRNodeSharedPtr)
{
    // Test IRNode with shared_ptr via ConstInt
    Span span("test.py", 1, 0, 1, 10);
    auto node = std::make_shared<ConstInt>(42, DataType::INT32, span);

    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->TypeName(), "ConstInt");
    ASSERT_EQ(node->span_.Filename(), "test.py");
}

} // namespace ir
} // namespace pypto
