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
 * \file test_expr_basic.cpp
 * \brief Unit tests for basic expression classes
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// ============================================================================
// Expr Base Class Tests
// ============================================================================

TEST(ExprBasicTest, TestExprBasicConstructor)
{
    // Test basic Expr construction via ConstInt (Expr is abstract with pure virtual GetKind)
    auto expr = std::make_shared<ConstInt>(0, DataType::INT32, Span::Unknown());
    ASSERT_NE(expr, nullptr);
    ASSERT_EQ(expr->TypeName(), "ConstInt");
}

TEST(ExprBasicTest, TestExprWithType)
{
    // Test Expr with explicit type via ConstInt
    auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::Unknown());
    ASSERT_NE(expr, nullptr);
    ASSERT_EQ(expr->GetType()->TypeName(), "ScalarType");
}

TEST(ExprBasicTest, TestExprGetType)
{
    // Test Expr GetType method via ConstFloat
    auto expr = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::Unknown());
    auto type = expr->GetType();
    ASSERT_NE(type, nullptr);
    ASSERT_EQ(type->TypeName(), "ScalarType");
}

TEST(ExprBasicTest, TestNoneValue)
{
    auto none = NoneValue();

    ASSERT_NE(none, nullptr);
    EXPECT_EQ(none, NoneValue());
    EXPECT_EQ(std::dynamic_pointer_cast<const Var>(none)->name_, "None");
    EXPECT_NE(std::dynamic_pointer_cast<const NoneType>(none->GetType()), nullptr);
}

// ============================================================================
// Op Tests
// ============================================================================

TEST(ExprBasicTest, TestOpBasicConstructor)
{
    // Test basic Op construction
    auto op = std::make_shared<Op>("test_op");
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->name_, "test_op");
}

TEST(ExprBasicTest, TestOpEmptyName)
{
    // Test Op with empty name
    auto op = std::make_shared<Op>("");
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->name_, "");
}

TEST(ExprBasicTest, TestOpSetAttrTypeBool)
{
    // Test Op SetAttrType with bool
    auto op = std::make_shared<Op>("test_op");
    op->SetAttrType<bool>("flag");
    ASSERT_TRUE(op->HasAttr("flag"));
}

TEST(ExprBasicTest, TestOpSetAttrTypeInt)
{
    // Test Op SetAttrType with int
    auto op = std::make_shared<Op>("test_op");
    op->SetAttrType<int>("count");
    ASSERT_TRUE(op->HasAttr("count"));
}

TEST(ExprBasicTest, TestOpSetAttrTypeString)
{
    // Test Op SetAttrType with string
    auto op = std::make_shared<Op>("test_op");
    op->SetAttrType<std::string>("name");
    ASSERT_TRUE(op->HasAttr("name"));
}

TEST(ExprBasicTest, TestOpSetAttrTypeDouble)
{
    // Test Op SetAttrType with double
    auto op = std::make_shared<Op>("test_op");
    op->SetAttrType<double>("value");
    ASSERT_TRUE(op->HasAttr("value"));
}

TEST(ExprBasicTest, TestOpSetAttrTypeDataType)
{
    // Test Op SetAttrType with DataType
    auto op = std::make_shared<Op>("test_op");
    op->SetAttrType<DataType>("dtype");
    ASSERT_TRUE(op->HasAttr("dtype"));
}

TEST(ExprBasicTest, TestOpHasAttr)
{
    // Test Op HasAttr method
    auto op = std::make_shared<Op>("test_op");
    ASSERT_FALSE(op->HasAttr("nonexistent"));

    op->SetAttrType<int>("count");
    ASSERT_TRUE(op->HasAttr("count"));
    ASSERT_FALSE(op->HasAttr("other"));
}

TEST(ExprBasicTest, TestOpGetAttrKeys)
{
    // Test Op GetAttrKeys method
    auto op = std::make_shared<Op>("test_op");
    op->SetAttrType<int>("count");
    op->SetAttrType<bool>("flag");

    auto keys = op->GetAttrKeys();
    ASSERT_EQ(keys.size(), 2);
}

TEST(ExprBasicTest, TestOpGetAttrs)
{
    // Test Op GetAttrs method
    auto op = std::make_shared<Op>("test_op");
    op->SetAttrType<int>("count");
    op->SetAttrType<bool>("flag");

    const auto& attrs = op->GetAttrs();
    ASSERT_EQ(attrs.size(), 2);
}

TEST(ExprBasicTest, TestOpMultipleAttrs)
{
    // Test Op with multiple attributes
    auto op = std::make_shared<Op>("complex_op");
    op->SetAttrType<bool>("flag1");
    op->SetAttrType<int>("count");
    op->SetAttrType<std::string>("name");
    op->SetAttrType<double>("value");
    op->SetAttrType<DataType>("dtype");

    ASSERT_TRUE(op->HasAttr("flag1"));
    ASSERT_TRUE(op->HasAttr("count"));
    ASSERT_TRUE(op->HasAttr("name"));
    ASSERT_TRUE(op->HasAttr("value"));
    ASSERT_TRUE(op->HasAttr("dtype"));
    ASSERT_EQ(op->GetAttrKeys().size(), 5);
}

} // namespace ir
} // namespace pypto
