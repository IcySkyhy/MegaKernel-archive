/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/error.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/program.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/transforms/base/visitor.h"
#include "ir/type.h"
#include "ir/verifier/verification_error.h"
#include "ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace ssa {
std::string ErrorTypeToString(ErrorType type)
{
    switch (type) {
        case ErrorType::MULTIPLE_ASSIGNMENT:
            return "MULTIPLE_ASSIGNMENT";
        case ErrorType::NAME_SHADOWING:
            return "NAME_SHADOWING";
        case ErrorType::MISSING_YIELD:
            return "MISSING_YIELD";
        case ErrorType::USE_BEFORE_DEF:
            return "USE_BEFORE_DEF";
        case ErrorType::ARITY_MISMATCH:
            return "ARITY_MISMATCH";
        default:
            return "UNKNOWN";
    }
}
} // namespace ssa

namespace {

StmtPtr GetLastStmtFromSeq(const StmtPtr& stmt)
{
    auto seq = As<SeqStmts>(stmt);
    if (!seq || seq->stmts_.empty()) {
        return stmt;
    }
    return GetLastStmtFromSeq(seq->stmts_.back());
}

// Number of values carried by a loop/if body terminator (YieldStmt or ContinueStmt). Returns 0 for
// any other stmt (including nullptr), so callers should confirm the stmt is a terminator first.
size_t TerminatorValueCount(const StmtPtr& stmt)
{
    if (auto y = As<YieldStmt>(stmt)) {
        return y->value_.size();
    }
    if (auto c = As<ContinueStmt>(stmt)) {
        return c->value_.size();
    }
    return 0;
}

/**
 * \brief Helper visitor class for SSA verification
 *
 * Traverses the IR tree and collects SSA violations
 */
class SSAVerifier : public IRVisitor {
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

public:
    explicit SSAVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics)
    {
        PushScope(); // function-level base scope
    }

    void VisitExpr_(const VarPtr& op) override;
    void VisitStmt_(const AssignStmtPtr& op) override;
    void VisitStmt_(const TensorOpStmtPtr& op) override;
    void VisitStmt_(const ForStmtPtr& op) override;
    void VisitStmt_(const WhileStmtPtr& op) override;
    void VisitStmt_(const IfStmtPtr& op) override;

    /** Seed function parameters as dominating definitions in the base scope. */
    void SeedParams(const std::vector<VarPtr>& params)
    {
        for (const auto& p : params) {
            Define(p);
        }
    }

    [[nodiscard]] const std::vector<Diagnostic>& GetDiagnostics() const { return diagnostics_; }

private:
    std::vector<Diagnostic>& diagnostics_;
    std::unordered_map<const Var*, int> var_assignment_count_;
    // Scope stack of defined vars (pointer identity). A var is in scope if present in any active scope.
    std::vector<std::unordered_set<const Var*>> scopes_;

    void PushScope() { scopes_.emplace_back(); }
    void PopScope()
    {
        if (!scopes_.empty()) {
            scopes_.pop_back();
        }
    }
    void Define(const VarPtr& v)
    {
        if (v && !scopes_.empty()) {
            scopes_.back().insert(v.get());
        }
    }
    bool IsDefined(const Var* v) const
    {
        if (v == nullptr) {
            return true;
        }
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            if (it->count(v) != 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * \brief Check if a variable has been assigned multiple times
     */
    void CheckVariableAssignment(const VarPtr& var);

    /**
     * \brief Record an error
     */
    void RecordError(ssa::ErrorType type, const std::string& message, const Span& span);

    /**
     * \brief Record a use-before-def (dominance) error for a LogicalTensor used out of scope
     */
    void RecordUseBeforeDef(const VarPtr& var);

    /**
     * \brief Get the last statement in a statement block (recursive for SeqStmts)
     */
    StmtPtr GetLastStmt(const StmtPtr& stmt);

    /**
     * \brief Verify ForStmt specific constraints
     */
    void VerifyForStmt(const ForStmtPtr& for_stmt);

    /**
     * \brief Verify WhileStmt specific constraints
     */
    void VerifyWhileStmt(const WhileStmtPtr& while_stmt);

    /**
     * \brief Verify IfStmt specific constraints
     */
    void VerifyIfStmt(const IfStmtPtr& if_stmt);
};

void SSAVerifier::CheckVariableAssignment(const VarPtr& var)
{
    const Var* key = var.get();
    var_assignment_count_[key]++;

    if (var_assignment_count_[key] > 1) {
        std::ostringstream msg;
        msg << "Variable '" << var->name_ << "' is assigned more than once (" << var_assignment_count_[key]
            << " times), violating SSA form";
        RecordError(ssa::ErrorType::MULTIPLE_ASSIGNMENT, msg.str(), var->span_);
    }
}

void SSAVerifier::RecordError(ssa::ErrorType type, const std::string& message, const Span& span)
{
    diagnostics_.emplace_back(DiagnosticSeverity::ERROR, "SSAVerify", static_cast<int>(type), message, span);
}

void SSAVerifier::RecordUseBeforeDef(const VarPtr& var)
{
    std::ostringstream msg;
    msg << "Variable '" << var->name_
        << "' is used before its dominating definition (out of scope), violating SSA dominance";
    RecordError(ssa::ErrorType::USE_BEFORE_DEF, msg.str(), var->span_);
}

StmtPtr SSAVerifier::GetLastStmt(const StmtPtr& stmt) { return GetLastStmtFromSeq(stmt); }

void SSAVerifier::VerifyForStmt(const ForStmtPtr& for_stmt)
{
    // iter_args and return_vars correspond one-to-one.
    if (for_stmt->iterArgs_.size() != for_stmt->returnVars_.size()) {
        RecordError(ssa::ErrorType::ARITY_MISMATCH, "ForStmt iter_args count must equal return_vars count",
                    for_stmt->span_);
    }

    // If iter_args is non-empty, the body must end with a value-producing terminator
    // (YieldStmt or ContinueStmt) whose value count matches the iter_args count.
    if (!for_stmt->iterArgs_.empty()) {
        StmtPtr last_stmt = GetLastStmt(for_stmt->body_);
        if (!last_stmt || !(As<YieldStmt>(last_stmt) || As<ContinueStmt>(last_stmt))) {
            RecordError(ssa::ErrorType::MISSING_YIELD,
                        "ForStmt with iter_args must end with a YieldStmt or ContinueStmt", for_stmt->span_);
        } else if (TerminatorValueCount(last_stmt) != for_stmt->iterArgs_.size()) {
            RecordError(ssa::ErrorType::ARITY_MISMATCH,
                        "ForStmt body terminator value count must equal iter_args count", for_stmt->span_);
        }
    }
}

void SSAVerifier::VerifyWhileStmt(const WhileStmtPtr& while_stmt)
{
    // iter_args and return_vars correspond one-to-one.
    if (while_stmt->iterArgs_.size() != while_stmt->returnVars_.size()) {
        RecordError(ssa::ErrorType::ARITY_MISMATCH, "WhileStmt iter_args count must equal return_vars count",
                    while_stmt->span_);
    }

    // If iter_args is non-empty, the body must end with a value-producing terminator
    // (YieldStmt or ContinueStmt) whose value count matches the iter_args count.
    if (!while_stmt->iterArgs_.empty()) {
        StmtPtr last_stmt = GetLastStmt(while_stmt->body_);
        if (!last_stmt || !(As<YieldStmt>(last_stmt) || As<ContinueStmt>(last_stmt))) {
            RecordError(ssa::ErrorType::MISSING_YIELD,
                        "WhileStmt with iter_args must end with a YieldStmt or ContinueStmt", while_stmt->span_);
        } else if (TerminatorValueCount(last_stmt) != while_stmt->iterArgs_.size()) {
            RecordError(ssa::ErrorType::ARITY_MISMATCH,
                        "WhileStmt body terminator value count must equal iter_args count", while_stmt->span_);
        }
    }
}

void SSAVerifier::VerifyIfStmt(const IfStmtPtr& if_stmt)
{
    // Check only if return_vars is not empty
    if (if_stmt->returnVars_.empty()) {
        return;
    }

    // Check 1: else_body must exist
    if (!if_stmt->elseBody_.has_value()) {
        RecordError(ssa::ErrorType::MISSING_YIELD, "IfStmt with return_vars must have else branch", if_stmt->span_);
        return;
    }

    // Check 2: Both then_body and else_body must end with YieldStmt
    StmtPtr then_last = GetLastStmt(if_stmt->thenBody_);
    StmtPtr else_last = GetLastStmt(if_stmt->elseBody_.value());

    auto then_yield = As<YieldStmt>(then_last);
    auto else_yield = As<YieldStmt>(else_last);

    if (!then_yield) {
        RecordError(ssa::ErrorType::MISSING_YIELD, "IfStmt then branch must end with YieldStmt when return_vars exist",
                    if_stmt->span_);
    } else if (then_yield->value_.size() != if_stmt->returnVars_.size()) {
        RecordError(ssa::ErrorType::ARITY_MISMATCH, "IfStmt then-branch yield value count must equal return_vars count",
                    if_stmt->span_);
    }

    if (!else_yield) {
        RecordError(ssa::ErrorType::MISSING_YIELD, "IfStmt else branch must end with YieldStmt when return_vars exist",
                    if_stmt->span_);
    } else if (else_yield->value_.size() != if_stmt->returnVars_.size()) {
        RecordError(ssa::ErrorType::ARITY_MISMATCH, "IfStmt else-branch yield value count must equal return_vars count",
                    if_stmt->span_);
    }
}

void SSAVerifier::VisitExpr_(const VarPtr& op)
{
    // Dominance use-check: a LogicalTensor must be defined by an enclosing statement before use.
    // Scalar/symbolic vars (symbolic scalars, shape-dim vars) have no explicit def site in the IR and
    // are intentionally skipped to avoid false positives. Shape dims are not recursed into for the
    // same reason (IRVisitor::VisitExpr_ is deliberately not called).
    if (op && ir::IsA<LogicalTensorType>(op->GetType()) && !IsDefined(op.get())) {
        RecordUseBeforeDef(op);
    }
}

void SSAVerifier::VisitStmt_(const TensorOpStmtPtr& op)
{
    // Operands/tokens are uses; results are defs and must not be visited as uses.
    for (const auto& arg : op->args_) {
        VisitExpr(ir::As<Var>(arg));
    }
    for (const auto& token : op->tokens_) {
        VisitExpr(ir::As<Var>(token));
    }

    for (const auto& result : op->result_) {
        if (op->opcode_ != "ASSEMBLE") {
            // ASSEMBLE currently not generare new variable now
            CheckVariableAssignment(result);
        }
        Define(result);
    }
    for (const auto& token : op->result_token_) {
        auto tokenType = std::dynamic_pointer_cast<const TokenType>(token->GetType());
        if (!tokenType || tokenType->kind_ != TokenKind::READ) {
            CheckVariableAssignment(token);
        }
        Define(token);
    }
}

void SSAVerifier::VisitStmt_(const AssignStmtPtr& op)
{
    // value_ is a use; var_ is a def (defined here, not visited as a use).
    if (op->value_) {
        VisitExpr(op->value_);
    }

    // Check for multiple assignments and define the target.
    CheckVariableAssignment(op->var_);
    Define(op->var_);
}

void SSAVerifier::VisitStmt_(const ForStmtPtr& op)
{
    // Check return_vars for multiple assignments
    for (const auto& return_var : op->returnVars_) {
        if (return_var) {
            CheckVariableAssignment(return_var);
        }
    }

    // start/stop/step and iter_args' initValue are evaluated in the outer scope before the loop.
    if (op->start_)
        VisitExpr(op->start_);
    if (op->stop_)
        VisitExpr(op->stop_);
    if (op->step_)
        VisitExpr(op->step_);

    for (const auto& iter_arg : op->iterArgs_) {
        if (iter_arg && iter_arg->initValue_) {
            VisitExpr(iter_arg->initValue_);
        }
    }

    // Loop body scope: only loopVar and iter_args' iterVar are in scope inside the body.
    PushScope();
    Define(op->loopVar_);
    for (const auto& iter_arg : op->iterArgs_) {
        if (iter_arg) {
            Define(iter_arg->iterVar_);
        }
    }
    if (op->body_) {
        VisitStmt(op->body_);
    }
    PopScope();

    // return_vars are loop outputs: defined in the enclosing scope after the loop.
    for (const auto& return_var : op->returnVars_) {
        Define(return_var);
    }

    // Verify ForStmt specific constraints
    VerifyForStmt(op);
}

void SSAVerifier::VisitStmt_(const WhileStmtPtr& op)
{
    // Check return_vars for multiple assignments
    for (const auto& return_var : op->returnVars_) {
        if (return_var) {
            CheckVariableAssignment(return_var);
        }
    }

    // iter_args' initValue are evaluated in the outer scope before the loop.
    for (const auto& iter_arg : op->iterArgs_) {
        if (iter_arg && iter_arg->initValue_) {
            VisitExpr(iter_arg->initValue_);
        }
    }

    // Loop body scope: iter_args' iterVar are in scope inside the body. The condition references
    // iter_args, so it is evaluated inside the body scope (after defining iter_args).
    PushScope();
    for (const auto& iter_arg : op->iterArgs_) {
        if (iter_arg) {
            Define(iter_arg->iterVar_);
        }
    }
    if (op->condition_) {
        VisitExpr(op->condition_);
    }
    if (op->body_) {
        VisitStmt(op->body_);
    }
    PopScope();

    // return_vars are loop outputs: defined in the enclosing scope after the loop.
    for (const auto& return_var : op->returnVars_) {
        Define(return_var);
    }

    // Verify WhileStmt specific constraints
    VerifyWhileStmt(op);
}

void SSAVerifier::VisitStmt_(const IfStmtPtr& op)
{
    // Check return_vars for multiple assignments
    for (const auto& return_var : op->returnVars_) {
        if (return_var) {
            CheckVariableAssignment(return_var);
        }
    }

    // Visit condition in the current (pre-if) scope.
    if (op->condition_) {
        VisitExpr(op->condition_);
    }

    // Each branch gets its own scope; branch-local defs do not escape.
    if (op->thenBody_) {
        PushScope();
        VisitStmt(op->thenBody_);
        PopScope();
    }
    if (op->elseBody_.has_value() && op->elseBody_.value()) {
        PushScope();
        VisitStmt(op->elseBody_.value());
        PopScope();
    }

    // return_vars are phi outputs: defined in the enclosing scope after the if.
    for (const auto& return_var : op->returnVars_) {
        Define(return_var);
    }

    // Verify IfStmt specific constraints
    VerifyIfStmt(op);
}

} // namespace

/**
 * \brief SSA property verifier for use with IRVerifier
 */
class SSAPropertyVerifierImpl : public PropertyVerifier {
public:
    [[nodiscard]] std::string GetName() const override { return "SSAVerify"; }

    void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override
    {
        for (const auto& entry : program->functions_) {
            const auto& func = entry.second;
            if (!func || !func->body_) {
                continue;
            }
            SSAVerifier verifier(diagnostics);
            verifier.SeedParams(func->params_);
            verifier.VisitStmt(func->body_);
        }
    }
};

PropertyVerifierPtr CreateSSAPropertyVerifier() { return std::make_shared<SSAPropertyVerifierImpl>(); }

} // namespace ir
} // namespace pypto
