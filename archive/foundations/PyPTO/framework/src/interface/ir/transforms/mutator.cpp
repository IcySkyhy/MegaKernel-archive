/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/transforms/base/mutator.h"

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/transforms/base/functor.h"
#include "ir/type.h"

#include "interface/tensor/ir_tensor_op_rebuild.h"

namespace pypto {
namespace ir {

namespace {

/// Reconstruct a binary expression with new children, preserving the concrete type.
/// All binary ops share the constructor signature (ExprPtr, ExprPtr, DataType, Span).
ExprPtr ReconstructBinaryExpr(ObjectKind kind, ExprPtr left, ExprPtr right, DataType dtype, const Span& span)
{
    switch (kind) {
        case ObjectKind::Add:
            return std::make_shared<const Add>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Sub:
            return std::make_shared<const Sub>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Mul:
            return std::make_shared<const Mul>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::FloorDiv:
            return std::make_shared<const FloorDiv>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::FloorMod:
            return std::make_shared<const FloorMod>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::FloatDiv:
            return std::make_shared<const FloatDiv>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Min:
            return std::make_shared<const Min>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Max:
            return std::make_shared<const Max>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Pow:
            return std::make_shared<const Pow>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Eq:
            return std::make_shared<const Eq>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Ne:
            return std::make_shared<const Ne>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Lt:
            return std::make_shared<const Lt>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Le:
            return std::make_shared<const Le>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Gt:
            return std::make_shared<const Gt>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Ge:
            return std::make_shared<const Ge>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::And:
            return std::make_shared<const And>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Or:
            return std::make_shared<const Or>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::Xor:
            return std::make_shared<const Xor>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::BitAnd:
            return std::make_shared<const BitAnd>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::BitOr:
            return std::make_shared<const BitOr>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::BitXor:
            return std::make_shared<const BitXor>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::BitShiftLeft:
            return std::make_shared<const BitShiftLeft>(std::move(left), std::move(right), dtype, span);
        case ObjectKind::BitShiftRight:
            return std::make_shared<const BitShiftRight>(std::move(left), std::move(right), dtype, span);
        default:
            INTERNAL_CHECK_SPAN(false, span) << "Unknown binary expression kind in ReconstructBinaryExpr";
    }
}

/// Reconstruct a unary expression with a new operand, preserving the concrete type.
/// All unary ops share the constructor signature (ExprPtr, DataType, Span).
ExprPtr ReconstructUnaryExpr(ObjectKind kind, ExprPtr operand, DataType dtype, const Span& span)
{
    switch (kind) {
        case ObjectKind::Abs:
            return std::make_shared<const Abs>(std::move(operand), dtype, span);
        case ObjectKind::Neg:
            return std::make_shared<const Neg>(std::move(operand), dtype, span);
        case ObjectKind::Not:
            return std::make_shared<const Not>(std::move(operand), dtype, span);
        case ObjectKind::BitNot:
            return std::make_shared<const BitNot>(std::move(operand), dtype, span);
        case ObjectKind::Cast:
            return std::make_shared<const Cast>(std::move(operand), dtype, span);
        default:
            INTERNAL_CHECK_SPAN(false, span) << "Unknown unary expression kind in ReconstructUnaryExpr";
    }
}

/// Cast an expression back to VarPtr while preserving MemRef instances.
VarPtr AsVar(const ExprPtr& expr, const Span& span)
{
    if (auto var = As<Var>(expr)) {
        return var;
    }
    if (auto memref = As<MemRef>(expr)) {
        return std::static_pointer_cast<const Var>(memref);
    }
    INTERNAL_CHECK_SPAN(false, span) << "invalid var expression";
    return nullptr;
}

/// Rebuild a Call while preserving kwargs_.
ExprPtr ReconstructCallWithKwargs(const std::string& name, std::vector<ExprPtr> args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs, const TypePtr& type,
                                  const Span& span)
{
    return std::make_shared<Call>(name, std::move(args), kwargs, type, span);
}

} // namespace

// Top-level entry points
ProgramPtr IRMutator::VisitProgram(const ProgramPtr& program)
{
    std::vector<FunctionPtr> new_functions;
    bool changed = false;
    for (const auto& entry : program->functions_) {
        auto new_func = VisitFunction(entry.second);
        new_functions.emplace_back(new_func);
        changed = changed || (new_func != entry.second);
    }
    if (!changed) {
        return program;
    }
    return std::make_shared<const Program>(std::move(new_functions), program->name_, program->span_);
}

FunctionPtr IRMutator::VisitFunction(const FunctionPtr& func)
{
    auto new_body = VisitStmt(func->body_);
    if (new_body.get() == func->body_.get()) {
        return func;
    }
    return std::make_shared<const Function>(func->name_, func->params_, func->returnTypes_, std::move(new_body),
                                            func->span_, func->funcType_, func->entry_, func->attrs_);
}

ExprPtr IRMutator::VisitExpr(const ExprPtr& expr) { return ExprFunctor<ExprPtr>::VisitExpr(expr); }

StmtPtr IRMutator::VisitStmt(const StmtPtr& stmt) { return StmtFunctor<StmtPtr>::VisitStmt(stmt); }

std::pair<std::vector<ExprPtr>, bool> IRMutator::VisitExprList(const std::vector<ExprPtr>& exprs)
{
    std::vector<ExprPtr> out;
    bool changed = false;
    out.reserve(exprs.size());
    for (size_t i = 0; i < exprs.size(); ++i) {
        auto ne = ExprFunctor<ExprPtr>::VisitExpr(exprs[i]);
        out.push_back(ne);
        changed = changed || (ne.get() != exprs[i].get());
    }
    return {std::move(out), changed};
}

std::pair<std::vector<VarPtr>, bool> IRMutator::VisitVarList(const std::vector<VarPtr>& vars)
{
    std::vector<VarPtr> out;
    bool changed = false;
    out.reserve(vars.size());
    for (size_t i = 0; i < vars.size(); ++i) {
        auto nv = ExprFunctor<ExprPtr>::VisitExpr(vars[i]);
        out.push_back(AsVar(nv, vars[i]->span_));
        changed = changed || (nv.get() != vars[i].get());
    }
    return {std::move(out), changed};
}

std::pair<std::vector<IterArgPtr>, bool> IRMutator::VisitIterArgList(const std::vector<IterArgPtr>& iterArgs)
{
    std::vector<IterArgPtr> out;
    bool changed = false;
    out.reserve(iterArgs.size());
    for (size_t i = 0; i < iterArgs.size(); ++i) {
        auto newInit = ExprFunctor<ExprPtr>::VisitExpr(iterArgs[i]->initValue_);
        if (newInit.get() != iterArgs[i]->initValue_.get()) {
            out.push_back(std::make_shared<const IterArg>(iterArgs[i]->iterVar_, std::move(newInit)));
            changed = true;
        } else {
            out.push_back(iterArgs[i]);
        }
    }
    // Register old->new IterArg var mappings so subsequent condition/body visits substitute refs.
    for (size_t i = 0; i < iterArgs.size(); ++i) {
        if (out[i].get() != iterArgs[i].get()) {
            var_remap_[iterArgs[i]->iterVar_.get()] = out[i]->iterVar_;
        }
    }
    return {std::move(out), changed};
}

ExprPtr IRMutator::VisitExpr_(const VarPtr& op)
{
    auto it = var_remap_.find(op.get());
    if (it != var_remap_.end()) {
        return it->second;
    }
    return op;
}

ExprPtr IRMutator::VisitExpr_(const MemRefPtr& op) { return op; }

ExprPtr IRMutator::VisitExpr_(const ConstIntPtr& op) { return op; }

ExprPtr IRMutator::VisitExpr_(const ConstFloatPtr& op) { return op; }

ExprPtr IRMutator::VisitExpr_(const ConstBoolPtr& op) { return op; }

ExprPtr IRMutator::VisitExpr_(const CallPtr& op)
{
    auto [new_args, changed] = VisitExprList(op->args_);
    if (changed) {
        return ReconstructCallWithKwargs(op->name_, std::move(new_args), op->kwargs_, op->GetType(), op->span_);
    }
    return op;
}

ExprPtr IRMutator::VisitExpr_(const MakeTuplePtr& op)
{
    auto [new_elements, changed] = VisitExprList(op->elements_);
    if (changed) {
        return std::make_shared<const MakeTuple>(std::move(new_elements), op->span_);
    }
    return op;
}

ExprPtr IRMutator::VisitExpr_(const GetItemExprPtr& op)
{
    INTERNAL_CHECK_SPAN(op->value_, op->span_) << "GetItemExpr has null value";
    INTERNAL_CHECK_SPAN(op->slice_, op->span_) << "GetItemExpr has null slice";
    auto new_value = ExprFunctor<ExprPtr>::VisitExpr(op->value_);
    auto new_slice = ExprFunctor<ExprPtr>::VisitExpr(op->slice_);
    if (new_value.get() != op->value_.get() || new_slice.get() != op->slice_.get()) {
        return std::make_shared<const GetItemExpr>(std::move(new_value), std::move(new_slice), op->span_);
    }
    return op;
}

ExprPtr IRMutator::VisitExpr_(const ScalarExprPtr& op) { return op; }

ExprPtr IRMutator::VisitBinaryExpr_(const BinaryExprPtr& op)
{
    INTERNAL_CHECK_SPAN(op->left_, op->span_) << "BinaryExpr has null left operand";
    INTERNAL_CHECK_SPAN(op->right_, op->span_) << "BinaryExpr has null right operand";
    auto new_left = ExprFunctor<ExprPtr>::VisitExpr(op->left_);
    auto new_right = ExprFunctor<ExprPtr>::VisitExpr(op->right_);
    if (new_left.get() != op->left_.get() || new_right.get() != op->right_.get()) {
        auto scalar_type = As<ScalarType>(op->GetType());
        INTERNAL_CHECK_SPAN(scalar_type, op->span_) << "BinaryExpr has null type";
        return ReconstructBinaryExpr(op->GetKind(), std::move(new_left), std::move(new_right), scalar_type->dtype_,
                                     op->span_);
    }
    return op;
}

ExprPtr IRMutator::VisitUnaryExpr_(const UnaryExprPtr& op)
{
    INTERNAL_CHECK_SPAN(op->operand_, op->span_) << "UnaryExpr has null operand";
    auto new_operand = ExprFunctor<ExprPtr>::VisitExpr(op->operand_);
    if (new_operand.get() != op->operand_.get()) {
        auto scalar_type = As<ScalarType>(op->GetType());
        INTERNAL_CHECK_SPAN(scalar_type, op->span_) << "UnaryExpr has null type";
        return ReconstructUnaryExpr(op->GetKind(), std::move(new_operand), scalar_type->dtype_, op->span_);
    }
    return op;
}

#define DEFINE_BINARY_MUTATOR(OpType) \
    ExprPtr IRMutator::VisitExpr_(const OpType##Ptr& op) { return VisitBinaryExpr_(op); }

DEFINE_BINARY_MUTATOR(Add)
DEFINE_BINARY_MUTATOR(Sub)
DEFINE_BINARY_MUTATOR(Mul)
DEFINE_BINARY_MUTATOR(FloorDiv)
DEFINE_BINARY_MUTATOR(FloorMod)
DEFINE_BINARY_MUTATOR(FloatDiv)
DEFINE_BINARY_MUTATOR(Min)
DEFINE_BINARY_MUTATOR(Max)
DEFINE_BINARY_MUTATOR(Pow)
DEFINE_BINARY_MUTATOR(Eq)
DEFINE_BINARY_MUTATOR(Ne)
DEFINE_BINARY_MUTATOR(Lt)
DEFINE_BINARY_MUTATOR(Le)
DEFINE_BINARY_MUTATOR(Gt)
DEFINE_BINARY_MUTATOR(Ge)
DEFINE_BINARY_MUTATOR(And)
DEFINE_BINARY_MUTATOR(Or)
DEFINE_BINARY_MUTATOR(Xor)
DEFINE_BINARY_MUTATOR(BitAnd)
DEFINE_BINARY_MUTATOR(BitOr)
DEFINE_BINARY_MUTATOR(BitXor)
DEFINE_BINARY_MUTATOR(BitShiftLeft)
DEFINE_BINARY_MUTATOR(BitShiftRight)

#undef DEFINE_BINARY_MUTATOR

#define DEFINE_UNARY_MUTATOR(OpType) \
    ExprPtr IRMutator::VisitExpr_(const OpType##Ptr& op) { return VisitUnaryExpr_(op); }

DEFINE_UNARY_MUTATOR(Abs)
DEFINE_UNARY_MUTATOR(Neg)
DEFINE_UNARY_MUTATOR(Not)
DEFINE_UNARY_MUTATOR(BitNot)
DEFINE_UNARY_MUTATOR(Cast)

#undef DEFINE_UNARY_MUTATOR

StmtPtr IRMutator::VisitStmt_(const AssignStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->var_, op->span_) << "AssignStmt has null var";
    INTERNAL_CHECK_SPAN(op->value_, op->span_) << "AssignStmt has null value";
    auto new_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->var_);
    auto new_value = ExprFunctor<ExprPtr>::VisitExpr(op->value_);

    auto new_var = AsVar(new_var_expr, op->span_);
    if (new_var.get() != op->var_.get() || new_value.get() != op->value_.get()) {
        return std::make_shared<const AssignStmt>(std::move(new_var), std::move(new_value), op->span_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const IfStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->condition_, op->span_) << "IfStmt has null condition";
    auto new_condition = ExprFunctor<ExprPtr>::VisitExpr(op->condition_);
    auto changed = (new_condition.get() != op->condition_.get());

    INTERNAL_CHECK_SPAN(op->thenBody_, op->span_) << "IfStmt has null then_body";
    auto new_then_body = StmtFunctor<StmtPtr>::VisitStmt(op->thenBody_);
    changed = changed || (new_then_body.get() != op->thenBody_.get());

    std::optional<StmtPtr> new_else_body;
    if (op->elseBody_.has_value()) {
        INTERNAL_CHECK_SPAN(*op->elseBody_, op->span_) << "IfStmt has null else_body";
        auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(*op->elseBody_);
        new_else_body = new_stmt;
        changed = changed || (new_stmt.get() != op->elseBody_->get());
    }

    std::vector<VarPtr> new_return_vars;
    new_return_vars.reserve(op->returnVars_.size());
    for (size_t i = 0; i < op->returnVars_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->returnVars_[i], op->span_) << "IfStmt has null return_vars at index " << i;
        auto new_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->returnVars_[i]);
        auto new_var = AsVar(new_var_expr, op->span_);
        new_return_vars.push_back(new_var);
        changed = changed || (new_var.get() != op->returnVars_[i].get());
    }

    if (changed) {
        if (new_else_body.has_value()) {
            return std::make_shared<const IfStmt>(std::move(new_condition), std::move(new_then_body), *new_else_body,
                                                  std::move(new_return_vars), op->span_);
        } else {
            return std::make_shared<const IfStmt>(std::move(new_condition), std::move(new_then_body), std::nullopt,
                                                  std::move(new_return_vars), op->span_);
        }
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const YieldStmtPtr& op)
{
    auto [new_value, changed] = VisitExprList(op->value_);
    if (changed) {
        return std::make_shared<const YieldStmt>(std::move(new_value), op->span_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const ReturnStmtPtr& op)
{
    auto [new_value, changed] = VisitExprList(op->value_);
    if (changed) {
        return std::make_shared<const ReturnStmt>(std::move(new_value), op->span_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const ForStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->loopVar_, op->span_) << "ForStmt has null loop_var";
    INTERNAL_CHECK_SPAN(op->start_, op->span_) << "ForStmt has null start";
    INTERNAL_CHECK_SPAN(op->stop_, op->span_) << "ForStmt has null stop";
    INTERNAL_CHECK_SPAN(op->step_, op->span_) << "ForStmt has null step";

    auto new_loop_var = AsVar(ExprFunctor<ExprPtr>::VisitExpr(op->loopVar_), op->span_);
    auto changed = (new_loop_var.get() != op->loopVar_.get());

    auto new_start = ExprFunctor<ExprPtr>::VisitExpr(op->start_);
    auto new_stop = ExprFunctor<ExprPtr>::VisitExpr(op->stop_);
    auto new_step = ExprFunctor<ExprPtr>::VisitExpr(op->step_);
    changed = changed || (new_start.get() != op->start_.get() || new_stop.get() != op->stop_.get() ||
                          new_step.get() != op->step_.get());

    auto [new_iter_args, iter_changed] = VisitIterArgList(op->iterArgs_);
    changed = changed || iter_changed;

    INTERNAL_CHECK_SPAN(op->body_, op->span_) << "ForStmt has null body";
    auto new_body = StmtFunctor<StmtPtr>::VisitStmt(op->body_);
    changed = changed || (new_body.get() != op->body_.get());

    // Clean up IterArg var remappings.
    for (const auto& old_iter_arg : op->iterArgs_) {
        var_remap_.erase(old_iter_arg->iterVar_.get());
    }

    auto [new_returns, return_changed] = VisitVarList(op->returnVars_);
    changed = changed || return_changed;
    if (changed) {
        return std::make_shared<const ForStmt>(std::move(new_loop_var), std::move(new_start), std::move(new_stop),
                                               std::move(new_step), std::move(new_iter_args), std::move(new_body),
                                               std::move(new_returns), op->span_, op->attrs_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const WhileStmtPtr& op)
{
    // Visit iter_args first (definitions), before condition and body (uses).
    auto [new_iter_args, changed] = VisitIterArgList(op->iterArgs_);

    INTERNAL_CHECK_SPAN(op->condition_, op->span_) << "WhileStmt has null condition";
    auto new_condition = ExprFunctor<ExprPtr>::VisitExpr(op->condition_);
    changed = changed || (new_condition.get() != op->condition_.get());

    INTERNAL_CHECK_SPAN(op->body_, op->span_) << "WhileStmt has null body";
    auto new_body = StmtFunctor<StmtPtr>::VisitStmt(op->body_);
    changed = changed || (new_body.get() != op->body_.get());

    // Clean up IterArg var remappings.
    for (const auto& old_iter_arg : op->iterArgs_) {
        var_remap_.erase(old_iter_arg->iterVar_.get());
    }

    auto [new_returns, return_changed] = VisitVarList(op->returnVars_);
    changed = changed || return_changed;
    if (changed) {
        return std::make_shared<const WhileStmt>(std::move(new_condition), std::move(new_iter_args),
                                                 std::move(new_body), std::move(new_returns), op->span_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const SeqStmtsPtr& op)
{
    std::vector<StmtPtr> new_stmts;
    bool changed = false;
    new_stmts.reserve(op->stmts_.size());
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->stmts_[i], op->span_) << "SeqStmts has null statement at index " << i;
        auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(op->stmts_[i]);
        new_stmts.push_back(new_stmt);
        changed = changed || (new_stmt.get() != op->stmts_[i].get());
    }

    if (changed) {
        return SeqStmts::Flatten(std::move(new_stmts), op->span_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const SectionStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->body_, op->span_) << "SectionStmt has null body";
    auto new_body = StmtFunctor<StmtPtr>::VisitStmt(op->body_);
    if (new_body.get() != op->body_.get()) {
        return std::make_shared<const SectionStmt>(op->sectionKind_, std::move(new_body), op->span_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const EvalStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->expr_, op->span_) << "EvalStmt has null expr";
    auto new_expr = ExprFunctor<ExprPtr>::VisitExpr(op->expr_);
    if (new_expr.get() != op->expr_.get()) {
        return std::make_shared<const EvalStmt>(std::move(new_expr), op->span_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const BreakStmtPtr& op)
{
    auto [new_values, changed] = VisitExprList(op->value_);
    if (changed) {
        return std::make_shared<const BreakStmt>(std::move(new_values), op->span_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const ContinueStmtPtr& op)
{
    auto [new_values, changed] = VisitExprList(op->value_);
    if (changed) {
        return std::make_shared<const ContinueStmt>(std::move(new_values), op->span_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const ScalarOpStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->result_, op->span_) << "ScalarOpStmt has null result";
    auto new_result = As<Var>(ExprFunctor<ExprPtr>::VisitExpr(op->result_));
    auto changed = (new_result.get() != op->result_.get());

    INTERNAL_CHECK_SPAN(op->result_token_, op->span_) << "ScalarOpStmt has null result_token";
    auto new_token = As<Var>(ExprFunctor<ExprPtr>::VisitExpr(op->result_token_));
    changed = changed || (new_token.get() != op->result_token_.get());

    auto [new_args, args_changed] = VisitExprList(op->args_);
    changed = changed || args_changed;
    if (changed) {
        return std::make_shared<const ScalarOpStmt>(std::move(new_result), std::move(new_token), std::move(op->opcode_),
                                                    std::move(new_args), op->span_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const TensorOpStmtPtr& op)
{
    auto [new_results, changed] = VisitVarList(op->result_);
    auto [new_tokens_result, result_tokens_changed] = VisitVarList(op->result_token_);
    changed = changed || result_tokens_changed;

    auto [new_args, args_changed] = VisitExprList(op->args_);
    changed = changed || args_changed;

    auto [new_tokens, tokens_changed] = VisitVarList(op->tokens_);
    changed = changed || tokens_changed;

    if (changed) {
        return npu::tile_fwk::RebuildTensorOpStmt(op, std::move(new_results), std::move(new_tokens_result),
                                                  std::move(new_args), std::move(new_tokens), op->span_);
    }
    return op;
}

StmtPtr IRMutator::VisitStmt_(const StmtPtr& op) { return op; }

} // namespace ir
} // namespace pypto
