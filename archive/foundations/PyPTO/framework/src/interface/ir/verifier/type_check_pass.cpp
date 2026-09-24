/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/transforms/base/visitor.h"
#include "ir/type.h"
#include "ir/verifier/verification_error.h"
#include "ir/verifier/verifier.h"

namespace pypto {
namespace ir {

// Implement type check error type to string conversion
namespace typecheck {
std::string ErrorTypeToString(ErrorType type)
{
    switch (type) {
        case ErrorType::TYPE_KIND_MISMATCH:
            return "TYPE_KIND_MISMATCH";
        case ErrorType::DTYPE_MISMATCH:
            return "DTYPE_MISMATCH";
        case ErrorType::SHAPE_DIMENSION_MISMATCH:
            return "SHAPE_DIMENSION_MISMATCH";
        case ErrorType::SHAPE_VALUE_MISMATCH:
            return "SHAPE_VALUE_MISMATCH";
        case ErrorType::SIZE_MISMATCH:
            return "SIZE_MISMATCH";
        case ErrorType::IF_CONDITION_MUST_BE_SCALAR:
            return "IF_CONDITION_MUST_BE_SCALAR";
        case ErrorType::FOR_RANGE_MUST_BE_SCALAR:
            return "FOR_RANGE_MUST_BE_SCALAR";
        default:
            return "UNKNOWN";
    }
}
} // namespace typecheck

namespace {

StmtPtr GetLastStmtFromSeq(const StmtPtr& stmt)
{
    auto current = stmt;
    while (auto seq = As<SeqStmts>(current)) {
        if (seq->stmts_.empty()) {
            break;
        }
        current = seq->stmts_.back();
    }
    return current;
}

/**
 * \brief Helper visitor class for type checking
 *
 * Traverses the IR tree and checks type consistency in control flow constructs
 */
class TypeChecker : public IRVisitor {
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

public:
    explicit TypeChecker(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

    void VisitStmt_(const ForStmtPtr& op) override;
    void VisitStmt_(const WhileStmtPtr& op) override;
    void VisitStmt_(const IfStmtPtr& op) override;

    [[nodiscard]] const std::vector<Diagnostic>& GetDiagnostics() const { return diagnostics_; }

private:
    std::vector<Diagnostic>& diagnostics_;

    /**
     * \brief Record an error
     */
    void RecordError(typecheck::ErrorType type, const std::string& message, const Span& span);

    /**
     * \brief Get the last statement in a statement block (recursive for SeqStmts)
     */
    StmtPtr GetLastStmt(const StmtPtr& stmt);

    /**
     * \brief Check type equality including shape for TensorType and TileType
     */
    void CheckTypeEquality(const TypePtr& type1, const TypePtr& type2, const std::string& context,
                           const std::string& desc1, const std::string& desc2, const Span& span);

    /**
     * \brief Check if two ExprPtr represent the same constant value
     */
    [[nodiscard]] bool IsSameConstant(const ExprPtr& expr1, const ExprPtr& expr2) const;

    /**
     * \brief Check if expression type is ScalarType
     */
    void CheckIsScalarType(const ExprPtr& expr, const std::string& context, const Span& span);

    void CheckLoopIterArgYieldTypes(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars,
                                    const YieldStmtPtr& yield_stmt, const std::string& context, const Span& span);
    void CheckLoopIterArgYieldTypeAt(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars,
                                     const YieldStmtPtr& yield_stmt, size_t index, const std::string& context,
                                     const Span& span);
    void CheckIfReturnYieldTypes(const IfStmtPtr& op, const YieldStmtPtr& then_yield, const YieldStmtPtr& else_yield);
    void CheckIfReturnYieldTypeAt(const IfStmtPtr& op, const YieldStmtPtr& then_yield, const YieldStmtPtr& else_yield,
                                  size_t index);
};

// TypeChecker implementation

void TypeChecker::RecordError(typecheck::ErrorType type, const std::string& message, const Span& span)
{
    diagnostics_.emplace_back(DiagnosticSeverity::ERROR, "TypeCheck", static_cast<int>(type), message, span);
}

StmtPtr TypeChecker::GetLastStmt(const StmtPtr& stmt) { return GetLastStmtFromSeq(stmt); }

void TypeChecker::CheckTypeEquality(const TypePtr& type1, const TypePtr& type2, const std::string& context,
                                    const std::string& desc1, const std::string& desc2, const Span& span)
{
    if (!type1 || !type2)
        return;

    // Check ObjectKind first
    if (type1->GetKind() != type2->GetKind()) {
        std::ostringstream msg;
        msg << "Type kind mismatch in " << context << ": " << desc1 << " type '" << type1->TypeName()
            << "' != " << desc2 << " type '" << type2->TypeName() << "'";
        RecordError(typecheck::ErrorType::TYPE_KIND_MISMATCH, msg.str(), span);
        return;
    }

    // For ScalarType, check dtype
    if (type1->GetKind() == ObjectKind::ScalarType) {
        auto scalar1 = std::dynamic_pointer_cast<const ScalarType>(type1);
        auto scalar2 = std::dynamic_pointer_cast<const ScalarType>(type2);
        if (scalar1 && scalar2 && scalar1->dtype_ != scalar2->dtype_) {
            std::ostringstream msg;
            msg << "Dtype mismatch in " << context << ": " << desc1 << " dtype != " << desc2 << " dtype";
            RecordError(typecheck::ErrorType::DTYPE_MISMATCH, msg.str(), span);
        }
        return;
    }

    // For TensorType and TileType, check dtype and shape
    if (type1->GetKind() == ObjectKind::TensorType || type1->GetKind() == ObjectKind::TileType) {
        auto shaped1 = std::dynamic_pointer_cast<const ShapedType>(type1);
        auto shaped2 = std::dynamic_pointer_cast<const ShapedType>(type2);
        if (!shaped1 || !shaped2)
            return;

        // Check dtype
        if (shaped1->dtype_ != shaped2->dtype_) {
            std::ostringstream msg;
            msg << "Dtype mismatch in " << context << ": " << desc1 << " dtype != " << desc2 << " dtype";
            RecordError(typecheck::ErrorType::DTYPE_MISMATCH, msg.str(), span);
        }

        // Check shape dimensions count
        if (shaped1->shape_.size() != shaped2->shape_.size()) {
            std::ostringstream msg;
            msg << "Shape dimension count mismatch in " << context << ": " << desc1 << " has " << shaped1->shape_.size()
                << " dimensions, but " << desc2 << " has " << shaped2->shape_.size() << " dimensions";
            RecordError(typecheck::ErrorType::SHAPE_DIMENSION_MISMATCH, msg.str(), span);
            return;
        }

        // Check each shape dimension
        for (size_t i = 0; i < shaped1->shape_.size(); ++i) {
            const auto& dim1 = shaped1->shape_[i];
            const auto& dim2 = shaped2->shape_[i];

            if (!dim1 || !dim2)
                continue;

            // Try to compare as constants
            if (!IsSameConstant(dim1, dim2)) {
                // Check if both are ConstInt but different values
                auto const_int1 = As<ConstInt>(dim1);
                auto const_int2 = As<ConstInt>(dim2);
                if (const_int1 && const_int2) {
                    std::ostringstream msg;
                    msg << "Shape dimension mismatch in " << context << ": " << desc1 << " dimension[" << i
                        << "] = " << const_int1->value_ << ", but " << desc2 << " dimension[" << i
                        << "] = " << const_int2->value_;
                    RecordError(typecheck::ErrorType::SHAPE_VALUE_MISMATCH, msg.str(), span);
                }
                // For symbolic dimensions, we skip detailed checking
                // A more sophisticated analysis would be needed for symbolic shape verification
            }
        }
    }
}

bool TypeChecker::IsSameConstant(const ExprPtr& expr1, const ExprPtr& expr2) const
{
    if (!expr1 || !expr2)
        return false;

    // Check if both are ConstInt
    auto const_int1 = As<ConstInt>(expr1);
    auto const_int2 = As<ConstInt>(expr2);
    if (const_int1 && const_int2) {
        return const_int1->value_ == const_int2->value_;
    }

    // For symbolic expressions, we consider them potentially equal if they have the same structure
    // A more sophisticated check would require symbolic comparison, but for type checking
    // we primarily care about constant dimensions
    return false;
}

void TypeChecker::CheckIsScalarType(const ExprPtr& expr, const std::string& context, const Span& span)
{
    if (!expr || !expr->GetType())
        return;

    if (!As<ScalarType>(expr->GetType())) {
        std::ostringstream msg;
        msg << context << " must be ScalarType, but got " << expr->GetType()->TypeName();

        // Determine error type based on context
        auto error_type = (context.find("condition") != std::string::npos) ?
                              typecheck::ErrorType::IF_CONDITION_MUST_BE_SCALAR :
                              typecheck::ErrorType::FOR_RANGE_MUST_BE_SCALAR;

        RecordError(error_type, msg.str(), span);
    }
}

void TypeChecker::VisitStmt_(const ForStmtPtr& op)
{
    if (!op)
        return;

    // Check start, stop, step must be ScalarType
    if (op->start_ && op->start_->GetType()) {
        CheckIsScalarType(op->start_, "ForStmt start", op->span_);
    }
    if (op->stop_ && op->stop_->GetType()) {
        CheckIsScalarType(op->stop_, "ForStmt stop", op->span_);
    }
    if (op->step_ && op->step_->GetType()) {
        CheckIsScalarType(op->step_, "ForStmt step", op->span_);
    }

    // Check type consistency between iter_args initValue, yield values, and return_vars
    if (!op->iterArgs_.empty()) {
        StmtPtr last_stmt = GetLastStmt(op->body_);
        auto yield_stmt = As<YieldStmt>(last_stmt);
        if (yield_stmt) {
            CheckLoopIterArgYieldTypes(op->iterArgs_, op->returnVars_, yield_stmt, "ForStmt", op->span_);
        }
    }

    // Continue with default traversal
    IRVisitor::VisitStmt_(op);
}

void TypeChecker::CheckLoopIterArgYieldTypes(const std::vector<IterArgPtr>& iter_args,
                                             const std::vector<VarPtr>& return_vars, const YieldStmtPtr& yield_stmt,
                                             const std::string& context, const Span& span)
{
    const size_t num_iter_args = iter_args.size();
    const size_t num_yield_values = yield_stmt->value_.size();
    const size_t num_return_vars = return_vars.size();
    if (num_iter_args != num_yield_values || num_iter_args != num_return_vars) {
        std::ostringstream msg;
        msg << context << " size mismatch: iter_args=" << num_iter_args << ", yield values=" << num_yield_values
            << ", return_vars=" << num_return_vars;
        RecordError(typecheck::ErrorType::SIZE_MISMATCH, msg.str(), span);
        return;
    }

    for (size_t i = 0; i < num_iter_args; ++i) {
        CheckLoopIterArgYieldTypeAt(iter_args, return_vars, yield_stmt, i, context, span);
    }
}

void TypeChecker::CheckLoopIterArgYieldTypeAt(const std::vector<IterArgPtr>& iter_args,
                                              const std::vector<VarPtr>& return_vars, const YieldStmtPtr& yield_stmt,
                                              size_t index, const std::string& context, const Span& span)
{
    const auto& iter_arg = iter_args[index];
    const auto& yield_value = yield_stmt->value_[index];
    const auto& return_var = return_vars[index];

    if (!iter_arg || !iter_arg->initValue_ || !yield_value || !return_var)
        return;

    auto init_type = iter_arg->initValue_->GetType();
    auto iter_type = iter_arg->iterVar_->GetType();
    auto yield_type = yield_value->GetType();
    auto return_type = return_var->GetType();
    if (!init_type || !iter_type || !yield_type || !return_type)
        return;

    if (!As<UnknownType>(init_type)) {
        CheckTypeEquality(init_type, iter_type, context, "iter_arg[" + std::to_string(index) + "] initValue",
                          "iter_arg[" + std::to_string(index) + "] variable", span);
    }
    CheckTypeEquality(iter_type, yield_type, context, "iter_arg[" + std::to_string(index) + "] variable",
                      "yield value[" + std::to_string(index) + "]", span);
    CheckTypeEquality(iter_type, return_type, context, "iter_arg[" + std::to_string(index) + "] variable",
                      "return_var[" + std::to_string(index) + "]", span);
}

void TypeChecker::VisitStmt_(const WhileStmtPtr& op)
{
    if (!op)
        return;

    // Check condition must be ScalarType (bool)
    if (op->condition_ && op->condition_->GetType()) {
        CheckIsScalarType(op->condition_, "WhileStmt condition", op->span_);
    }

    // Check type consistency between iter_args initValue, yield values, and return_vars
    if (!op->iterArgs_.empty()) {
        StmtPtr last_stmt = GetLastStmt(op->body_);
        auto yield_stmt = As<YieldStmt>(last_stmt);
        if (yield_stmt)
            CheckLoopIterArgYieldTypes(op->iterArgs_, op->returnVars_, yield_stmt, "WhileStmt", op->span_);
    }

    // Continue with default traversal
    IRVisitor::VisitStmt_(op);
}

void TypeChecker::VisitStmt_(const IfStmtPtr& op)
{
    if (!op)
        return;

    // Check condition must be ScalarType
    if (op->condition_ && op->condition_->GetType()) {
        CheckIsScalarType(op->condition_, "IfStmt condition", op->span_);
    }

    // Check type consistency only if return_vars is not empty
    if (!op->returnVars_.empty() && op->elseBody_.has_value()) {
        StmtPtr then_last = GetLastStmt(op->thenBody_);
        StmtPtr else_last = GetLastStmt(op->elseBody_.value());

        auto then_yield = As<YieldStmt>(then_last);
        auto else_yield = As<YieldStmt>(else_last);

        if (then_yield && else_yield) {
            CheckIfReturnYieldTypes(op, then_yield, else_yield);
        }
    }

    // Continue with default traversal
    IRVisitor::VisitStmt_(op);
}

void TypeChecker::CheckIfReturnYieldTypes(const IfStmtPtr& op, const YieldStmtPtr& then_yield,
                                          const YieldStmtPtr& else_yield)
{
    size_t num_then_values = then_yield->value_.size();
    size_t num_else_values = else_yield->value_.size();
    size_t num_return_vars = op->returnVars_.size();
    if (num_then_values != num_else_values || num_then_values != num_return_vars) {
        std::ostringstream msg;
        msg << "IfStmt size mismatch: then yield=" << num_then_values << ", else yield=" << num_else_values
            << ", return_vars=" << num_return_vars;
        RecordError(typecheck::ErrorType::SIZE_MISMATCH, msg.str(), op->span_);
        return;
    }

    for (size_t i = 0; i < num_then_values; ++i) {
        CheckIfReturnYieldTypeAt(op, then_yield, else_yield, i);
    }
}

void TypeChecker::CheckIfReturnYieldTypeAt(const IfStmtPtr& op, const YieldStmtPtr& then_yield,
                                           const YieldStmtPtr& else_yield, size_t index)
{
    const auto& then_value = then_yield->value_[index];
    const auto& else_value = else_yield->value_[index];

    if (!then_value || !else_value)
        return;

    auto then_type = then_value->GetType();
    auto else_type = else_value->GetType();
    if (!then_type || !else_type)
        return;

    if (IsA<NoneType>(then_type) || IsA<NoneType>(else_type))
        return;

    CheckTypeEquality(then_type, else_type, "IfStmt", "then yield value[" + std::to_string(index) + "]",
                      "else yield value[" + std::to_string(index) + "]", op->span_);
}

} // namespace

/**
 * \brief Type check property verifier for use with IRVerifier
 */
class TypeCheckPropertyVerifierImpl : public PropertyVerifier {
public:
    [[nodiscard]] std::string GetName() const override { return "TypeCheck"; }

    void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override
    {
        verifier_detail::VerifyFunctionBodies<TypeChecker>(program, diagnostics);
    }
};

// Factory function for creating TypeCheck property verifier
PropertyVerifierPtr CreateTypeCheckPropertyVerifier() { return std::make_shared<TypeCheckPropertyVerifierImpl>(); }

} // namespace ir
} // namespace pypto
