/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "core/dtype.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/reflection/field_traits.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// Forward declaration for visitor pattern
// Implementation in pypto/ir/transform/base/visitor.h
class IRVisitor;

// Forward declaration for Op (defined in expr.h)
class Op;
using OpPtr = std::shared_ptr<const Op>;

/**
 * \brief Constant numeric expression
 *
 * Represents a constant numeric value.
 */
class ConstInt : public Expr {
public:
    const int64_t value_; // Numeric constant value (immutable)

    /**
     * \brief Create a constant expression
     *
     * \param value Numeric value
     * \param span Source location
     */
    ConstInt(int64_t value, DataType dtype, Span span)
        : Expr(std::move(span), std::make_shared<ScalarType>(dtype)), value_(value)
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ConstInt; }
    [[nodiscard]] std::string TypeName() const override { return "ConstInt"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (value as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Expr::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&ConstInt::value_, "value")));
    }

    [[nodiscard]] DataType dtype() const
    {
        // Note: Must use dynamic_pointer_cast here because this header is included before
        // the TypePtr overload of As<> is defined in kind_traits.h
        auto scalarType = std::dynamic_pointer_cast<const ScalarType>(GetType());
        INTERNAL_CHECK(scalarType) << "ConstInt is expected to have ScalarType type, but got " << GetType()->TypeName()
                                   << " at " << span_.ToString();
        return scalarType->dtype_;
    }
};

using ConstIntPtr = std::shared_ptr<const ConstInt>;

/**
 * \brief Constant floating-point expression
 *
 * Represents a constant floating-point value.
 */
class ConstFloat : public Expr {
public:
    const double value_; // Floating-point constant value (immutable)

    /**
     * \brief Create a constant floating-point expression
     *
     * \param value Floating-point value
     * \param dtype Data type
     * \param span Source location
     */
    ConstFloat(double value, DataType dtype, Span span)
        : Expr(std::move(span), std::make_shared<ScalarType>(dtype)), value_(value)
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ConstFloat; }
    [[nodiscard]] std::string TypeName() const override { return "ConstFloat"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (value as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Expr::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&ConstFloat::value_, "value")));
    }

    [[nodiscard]] DataType dtype() const
    {
        // Note: Must use dynamic_pointer_cast here because this header is included before
        // the TypePtr overload of As<> is defined in kind_traits.h
        auto scalarType = std::dynamic_pointer_cast<const ScalarType>(GetType());
        INTERNAL_CHECK(scalarType) << "ConstFloat is expected to have ScalarType type, but got "
                                   << GetType()->TypeName() << " at " << span_.ToString();
        return scalarType->dtype_;
    }
};

using ConstFloatPtr = std::shared_ptr<const ConstFloat>;

/**
 * \brief Constant boolean expression
 *
 * Represents a constant boolean value.
 */
class ConstBool : public Expr {
public:
    const bool value_; // Boolean constant value (immutable)

    /**
     * \brief Create a constant boolean expression
     *
     * \param value Boolean value
     * \param span Source location
     */
    ConstBool(bool value, Span span)
        : Expr(std::move(span), std::make_shared<ScalarType>(DataType::BOOL)), value_(value)
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ConstBool; }
    [[nodiscard]] std::string TypeName() const override { return "ConstBool"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (value as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Expr::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&ConstBool::value_, "value")));
    }

    [[nodiscard]] DataType dtype() const { return DataType::BOOL; }
};

using ConstBoolPtr = std::shared_ptr<const ConstBool>;

/**
 * \brief Base class for binary expressions
 *
 * Abstract base for all operations with two operands.
 */
class BinaryExpr : public Expr {
public:
    ExprPtr left_;  // Left operand
    ExprPtr right_; // Right operand

    BinaryExpr(ExprPtr left, ExprPtr right, DataType dtype, Span span)
        : Expr(std::move(span), std::make_shared<ScalarType>(dtype)), left_(std::move(left)), right_(std::move(right))
    {}

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (left and right as USUAL fields)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Expr::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&BinaryExpr::left_, "left"),
                                              reflection::UsualField(&BinaryExpr::right_, "right")));
    }
};

using BinaryExprPtr = std::shared_ptr<const BinaryExpr>;

// Macro to define binary expression node classes
// Usage: DEFINE_BINARY_EXPR_NODE(Add, "Addition expression (left + right)")
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define DEFINE_BINARY_EXPR_NODE(OpName, Description)                                           \
    /** \brief Description */                                                                  \
    class OpName : public BinaryExpr {                                                         \
    public:                                                                                    \
        OpName(ExprPtr left, ExprPtr right, DataType dtype, Span span)                         \
            : BinaryExpr(std::move(left), std::move(right), std::move(dtype), std::move(span)) \
        {}                                                                                     \
        [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::OpName; }       \
        [[nodiscard]] std::string TypeName() const override { return #OpName; }                \
    };                                                                                         \
                                                                                               \
    using OpName##Ptr = std::shared_ptr<const OpName>

DEFINE_BINARY_EXPR_NODE(Add, "Addition expression (left + right)");
DEFINE_BINARY_EXPR_NODE(Sub, "Subtraction expression (left - right)");
DEFINE_BINARY_EXPR_NODE(Mul, "Multiplication expression (left * right)");
DEFINE_BINARY_EXPR_NODE(FloorDiv, "Floor division expression (left // right)");
DEFINE_BINARY_EXPR_NODE(FloorMod, "Floor modulo expression (left % right)");
DEFINE_BINARY_EXPR_NODE(FloatDiv, "Float division expression (left / right)");
DEFINE_BINARY_EXPR_NODE(Min, "Minimum expression (min(left, right)");
DEFINE_BINARY_EXPR_NODE(Max, "Maximum expression (max(left, right)");
DEFINE_BINARY_EXPR_NODE(Pow, "Power expression (left ** right)");
DEFINE_BINARY_EXPR_NODE(Eq, "Equality expression (left == right)");
DEFINE_BINARY_EXPR_NODE(Ne, "Inequality expression (left != right)");
DEFINE_BINARY_EXPR_NODE(Lt, "Less than expression (left < right)");
DEFINE_BINARY_EXPR_NODE(Le, "Less than or equal to expression (left <= right)");
DEFINE_BINARY_EXPR_NODE(Gt, "Greater than expression (left > right)");
DEFINE_BINARY_EXPR_NODE(Ge, "Greater than or equal to expression (left >= right)");
DEFINE_BINARY_EXPR_NODE(And, "Logical and expression (left and right)");
DEFINE_BINARY_EXPR_NODE(Or, "Logical or expression (left or right)");
DEFINE_BINARY_EXPR_NODE(Xor, "Logical xor expression (left xor right)");
DEFINE_BINARY_EXPR_NODE(BitAnd, "Bitwise and expression (left & right)");
DEFINE_BINARY_EXPR_NODE(BitOr, "Bitwise or expression (left | right)");
DEFINE_BINARY_EXPR_NODE(BitXor, "Bitwise xor expression (left ^ right)");
DEFINE_BINARY_EXPR_NODE(BitShiftLeft, "Bitwise left shift expression (left << right)");
DEFINE_BINARY_EXPR_NODE(BitShiftRight, "Bitwise right shift expression (left >> right)");

#undef DEFINE_BINARY_EXPR_NODE

/**
 * \brief Base class for unary expressions
 *
 * Abstract base for all operations with one operand.
 */
class UnaryExpr : public Expr {
public:
    ExprPtr operand_; // Operand

    UnaryExpr(ExprPtr operand, DataType dtype, Span span)
        : Expr(std::move(span), std::make_shared<ScalarType>(dtype)), operand_(std::move(operand))
    {}

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Expr::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&UnaryExpr::operand_, "operand")));
    }
};

using UnaryExprPtr = std::shared_ptr<const UnaryExpr>;

// Macro to define unary expression node classes
// Usage: DEFINE_UNARY_EXPR_NODE(Neg, "Negation expression (-operand)")
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define DEFINE_UNARY_EXPR_NODE(OpName, Description)                                                                   \
    /** \brief Description */                                                                                         \
    class OpName : public UnaryExpr {                                                                                 \
    public:                                                                                                           \
        OpName(ExprPtr operand, DataType dtype, Span span) : UnaryExpr(std::move(operand), dtype, std::move(span)) {} \
        [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::OpName; }                              \
        [[nodiscard]] std::string TypeName() const override { return #OpName; }                                       \
    };                                                                                                                \
                                                                                                                      \
    using OpName##Ptr = std::shared_ptr<const OpName>

DEFINE_UNARY_EXPR_NODE(Abs, "Absolute value expression (abs(operand))");
DEFINE_UNARY_EXPR_NODE(Neg, "Negation expression (-operand)");
DEFINE_UNARY_EXPR_NODE(Not, "Logical not expression (not operand)");
DEFINE_UNARY_EXPR_NODE(BitNot, "Bitwise not expression (~operand)");
DEFINE_UNARY_EXPR_NODE(Cast, "Cast expression (cast operand to dtype)");

#undef DEFINE_UNARY_EXPR_NODE

class ScalarExpr : public Expr {
public:
    ScalarExpr(DataType dtype, Span span) : Expr(std::move(span), std::make_shared<ScalarType>(dtype)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ScalarExpr; }
    [[nodiscard]] std::string TypeName() const override { return "ScalarExpr"; }

    static constexpr auto GetFieldDescriptors() { return Expr::GetFieldDescriptors(); }
};

using ScalarExprPtr = std::shared_ptr<const ScalarExpr>;

} // namespace ir
} // namespace pypto
