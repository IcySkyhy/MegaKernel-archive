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
 * \file test_expr.cpp
 * \brief Unit tests for IR expression types (MakeTuple, GetItemExpr)
 */

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class IRExprTest : public testing::Test {};

// ============================================================================
// MakeTuple Tests
// ============================================================================

TEST_F(IRExprTest, TestMakeTupleBasic)
{
    auto elem1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto elem2 = std::make_shared<ConstFloat>(2.0, DataType::FP32, Span::Unknown());
    std::vector<ExprPtr> elements = {elem1, elem2};

    auto tuple = std::make_shared<MakeTuple>(elements, Span::Unknown());

    ASSERT_EQ(tuple->elements_.size(), 2);
    auto tupleType = As<TupleType>(tuple->GetType());
    ASSERT_NE(tupleType, nullptr);
    ASSERT_EQ(tupleType->types_.size(), 2);
}

TEST_F(IRExprTest, TestMakeTupleSingleElement)
{
    auto elem = std::make_shared<ConstInt>(42, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> elements = {elem};

    auto tuple = std::make_shared<MakeTuple>(elements, Span::Unknown());

    ASSERT_EQ(tuple->elements_.size(), 1);
    auto tupleType = As<TupleType>(tuple->GetType());
    ASSERT_NE(tupleType, nullptr);
    ASSERT_EQ(tupleType->types_.size(), 1);
}

// ============================================================================
// GetItemExpr Tests
// ============================================================================

TEST_F(IRExprTest, TestTupleGetItemBasic)
{
    auto elem1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto elem2 = std::make_shared<ConstFloat>(2.0, DataType::FP32, Span::Unknown());
    std::vector<ExprPtr> elements = {elem1, elem2};
    auto tuple = std::make_shared<MakeTuple>(elements, Span::Unknown());

    auto idx0 = std::make_shared<ConstInt>(0, DataType::INDEX, Span::Unknown());
    auto idx1 = std::make_shared<ConstInt>(1, DataType::INDEX, Span::Unknown());
    auto item0 = std::make_shared<GetItemExpr>(tuple, idx0, Span::Unknown());
    auto item1 = std::make_shared<GetItemExpr>(tuple, idx1, Span::Unknown());

    auto type0 = As<ScalarType>(item0->GetType());
    auto type1 = As<ScalarType>(item1->GetType());
    ASSERT_NE(type0, nullptr);
    ASSERT_NE(type1, nullptr);
    ASSERT_EQ(type0->dtype_, DataType::INT32);
    ASSERT_EQ(type1->dtype_, DataType::FP32);
}

TEST_F(IRExprTest, TestTupleGetItemIndex)
{
    auto elem1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto elem2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> elements = {elem1, elem2};
    auto tuple = std::make_shared<MakeTuple>(elements, Span::Unknown());

    auto idx = std::make_shared<ConstInt>(1, DataType::INDEX, Span::Unknown());
    auto item = std::make_shared<GetItemExpr>(tuple, idx, Span::Unknown());
    ASSERT_EQ(As<ConstInt>(item->slice_)->value_, 1);
    ASSERT_EQ(item->value_, tuple);
}

TEST_F(IRExprTest, TestTupleGetItemDynamicIndex)
{
    auto elem1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto elem2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> elements = {elem1, elem2};
    auto tuple = std::make_shared<MakeTuple>(elements, Span::Unknown());

    auto idx = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INDEX), Span::Unknown());
    auto item = std::make_shared<GetItemExpr>(tuple, idx, Span::Unknown());

    auto resultType = As<ScalarType>(item->GetType());
    ASSERT_NE(resultType, nullptr);
    ASSERT_EQ(resultType->dtype_, DataType::INT32);
}

} // namespace ir
} // namespace pypto
