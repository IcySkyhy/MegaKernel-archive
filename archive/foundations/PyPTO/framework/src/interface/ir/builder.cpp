/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/builder.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/error.h"
#include "core/logging.h"
#include "tilefwk/error.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr_ops.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {
namespace {

StmtPtr MakeStmtBody(const std::vector<StmtPtr>& stmts, const Span& span)
{
    if (stmts.empty()) {
        return std::make_shared<SeqStmts>(std::vector<StmtPtr>(), span);
    }
    if (stmts.size() == 1) {
        return stmts[0];
    }
    return std::make_shared<SeqStmts>(stmts, span);
}

Span MakeCombinedSpan(const Span& begin_span, const Span& end_span)
{
    return Span(begin_span.Filename(), begin_span.BeginLine(), begin_span.BeginColumn(), end_span.EndLine(),
                end_span.EndColumn());
}

/**
 * \brief Reconcile the branch value types of an if statement's SSA return vars.
 *
 * A ternary expression ``a if cond else b`` lowers to an if statement whose then/else
 * branches each yield their value into a phi return var. When the two branches produce
 * differing but same-category scalar dtypes (e.g. ``INT32`` from a tensor read vs
 * ``INDEX`` from a shape access), promote both yielded values to the wider dtype and
 * retype the phi return var to match, mirroring the promotion binary operators already
 * perform. Non-scalar or cross-category (int vs float) mismatches are left untouched so
 * the type verifier still reports them.
 *
 * \param then_stmts Statements of the then branch (last one is the YieldStmt); mutated in place.
 * \param else_stmts Statements of the else branch (last one is the YieldStmt); mutated in place.
 * \param return_vars Phi return vars; retyped in place when a branch pair is promoted.
 */
void ReconcileIfReturnVarTypes(std::vector<StmtPtr>& then_stmts, std::vector<StmtPtr>& else_stmts,
                               std::vector<VarPtr>& return_vars)
{
    if (return_vars.empty() || then_stmts.empty() || else_stmts.empty()) {
        return;
    }
    auto then_yield = std::dynamic_pointer_cast<const YieldStmt>(then_stmts.back());
    auto else_yield = std::dynamic_pointer_cast<const YieldStmt>(else_stmts.back());
    if (!then_yield || !else_yield) {
        return;
    }
    const size_t count = return_vars.size();
    if (then_yield->value_.size() != count || else_yield->value_.size() != count) {
        return;
    }

    std::vector<ExprPtr> then_values = then_yield->value_;
    std::vector<ExprPtr> else_values = else_yield->value_;
    bool changed = false;
    for (size_t i = 0; i < count; ++i) {
        const ExprPtr& then_value = then_values[i];
        const ExprPtr& else_value = else_values[i];
        if (!then_value || !else_value) {
            continue;
        }
        auto then_scalar = std::dynamic_pointer_cast<const ScalarType>(then_value->GetType());
        auto else_scalar = std::dynamic_pointer_cast<const ScalarType>(else_value->GetType());
        if (!then_scalar || !else_scalar) {
            continue;
        }
        DataType then_dtype = then_scalar->dtype_;
        DataType else_dtype = else_scalar->dtype_;
        if (then_dtype == else_dtype) {
            continue;
        }
        // Only reconcile a matching numeric category (both int or both float); bool and
        // int/float mismatches stay errors for the verifier to surface.
        bool both_int = then_dtype.IsInt() && else_dtype.IsInt();
        bool both_float = then_dtype.IsFloat() && else_dtype.IsFloat();
        if (!both_int && !both_float) {
            continue;
        }
        DataType promoted = PromoteSameCategoryDtype(then_dtype, else_dtype, "ternary", return_vars[i]->span_);
        then_values[i] = MaybeCast(then_value, promoted, then_yield->span_);
        else_values[i] = MaybeCast(else_value, promoted, else_yield->span_);
        return_vars[i] = std::make_shared<Var>(return_vars[i]->name_, std::make_shared<ScalarType>(promoted),
                                               return_vars[i]->span_);
        changed = true;
    }
    if (!changed) {
        return;
    }
    then_stmts.back() = std::make_shared<YieldStmt>(std::move(then_values), then_yield->span_);
    else_stmts.back() = std::make_shared<YieldStmt>(std::move(else_values), else_yield->span_);
}

std::string MakeLoopReturnVarsMismatchMessage(const char* loop_kind, size_t iter_arg_count, size_t return_var_count)
{
    std::ostringstream oss;
    oss << loop_kind << " loop has " << iter_arg_count << " iteration arguments but " << return_var_count
        << " return variables. They must match.";
    return oss.str();
}
} // namespace

// ========== IRBuilder Implementation ==========

IRBuilder::IRBuilder() = default;

// ========== Function Building ==========

void IRBuilder::BeginFunction(const std::string& name, const Span& span, FunctionType type)
{
    if (InFunction()) {
        throw RuntimeError("Cannot begin function '" + name + "': already inside function '" +
                           static_cast<FunctionContext*>(CurrentContext())->GetName() + "' at " +
                           CurrentContext()->GetBeginSpan().ToString());
    }

    context_stack_.push_back(std::make_shared<FunctionContext>(name, span, type));
}

VarPtr IRBuilder::FuncArg(const std::string& name, const TypePtr& type, const Span& span)
{
    ValidateInFunction("FuncArg");

    auto var = std::make_shared<ir::Var>(name, type, span);
    static_cast<FunctionContext*>(CurrentContext())->AddParam(var);
    return var;
}

void IRBuilder::ReturnType(const TypePtr& type)
{
    ValidateInFunction("ReturnType");
    static_cast<FunctionContext*>(CurrentContext())->AddReturnType(type);
}

FunctionPtr IRBuilder::EndFunction(const Span& end_span)
{
    ValidateInFunction("EndFunction");

    auto* func_ctx = static_cast<FunctionContext*>(CurrentContext());

    // Build body from accumulated statements
    StmtPtr body;
    const auto& stmts = func_ctx->GetStmts();
    if (stmts.empty()) {
        // Empty body - create empty SeqStmts
        body = std::make_shared<SeqStmts>(std::vector<StmtPtr>(), end_span);
    } else if (stmts.size() == 1) {
        body = stmts[0];
    } else {
        body = std::make_shared<SeqStmts>(stmts, end_span);
    }

    // Combine begin and end spans
    auto combinedSpan = MakeCombinedSpan(func_ctx->GetBeginSpan(), end_span);

    // Create function
    auto func = std::make_shared<Function>(func_ctx->GetName(), func_ctx->GetParams(), func_ctx->GetReturnTypes(), body,
                                           combinedSpan, func_ctx->GetFuncType(), false, func_ctx->GetAttrs());

    // Pop context
    context_stack_.pop_back();

    return func;
}

// ========== For Loop Building ==========

void IRBuilder::BeginForLoop(const VarPtr& loop_var, const ExprPtr& start, const ExprPtr& stop, const ExprPtr& step,
                             const Span& span)
{
    if (context_stack_.empty()) {
        throw RuntimeError("Cannot begin for loop: not inside a function or another valid context at " +
                           span.ToString());
    }

    context_stack_.push_back(std::make_shared<ForLoopContext>(loop_var, start, stop, step, span));
}

void IRBuilder::AddIterArg(const IterArgPtr& iter_arg)
{
    ValidateInLoop("AddIterArg");
    static_cast<ForLoopContext*>(CurrentContext())->AddIterArg(iter_arg);
}

void IRBuilder::AddReturnVar(const VarPtr& var)
{
    ValidateInLoop("AddReturnVar");
    static_cast<ForLoopContext*>(CurrentContext())->AddReturnVar(var);
}

StmtPtr IRBuilder::EndForLoop(const Span& end_span)
{
    ValidateInLoop("EndForLoop");

    auto* loop_ctx = static_cast<ForLoopContext*>(CurrentContext());

    // Validate iter_args and return_vars match
    if (loop_ctx->GetIterArgs().size() != loop_ctx->GetReturnVars().size()) {
        // Pop context before throwing to maintain stack consistency
        context_stack_.pop_back();

        throw RuntimeError(
            MakeLoopReturnVarsMismatchMessage("For", loop_ctx->GetIterArgs().size(), loop_ctx->GetReturnVars().size()));
    }

    auto body = MakeStmtBody(loop_ctx->GetStmts(), end_span);

    auto combinedSpan = MakeCombinedSpan(loop_ctx->GetBeginSpan(), end_span);

    // Create for statement
    auto for_stmt = std::make_shared<ForStmt>(loop_ctx->GetLoopVar(), loop_ctx->GetStart(), loop_ctx->GetStop(),
                                              loop_ctx->GetStep(), loop_ctx->GetIterArgs(), body,
                                              loop_ctx->GetReturnVars(), combinedSpan);

    // Pop context
    context_stack_.pop_back();

    // Emit to parent context if it exists
    if (!context_stack_.empty()) {
        CurrentContext()->AddStmt(for_stmt);
    }

    return for_stmt;
}

// ========== While Loop Building ==========

void IRBuilder::BeginWhileLoop(const ExprPtr& condition, const Span& span)
{
    if (context_stack_.empty()) {
        throw RuntimeError("Cannot begin while loop: not inside a function or another valid context at " +
                           span.ToString());
    }

    context_stack_.push_back(std::make_shared<WhileLoopContext>(condition, span));
}

void IRBuilder::AddWhileIterArg(const IterArgPtr& iter_arg)
{
    ValidateInWhileLoop("AddWhileIterArg");
    static_cast<WhileLoopContext*>(CurrentContext())->AddIterArg(iter_arg);
}

void IRBuilder::AddWhileReturnVar(const VarPtr& var)
{
    ValidateInWhileLoop("AddWhileReturnVar");
    static_cast<WhileLoopContext*>(CurrentContext())->AddReturnVar(var);
}

void IRBuilder::SetWhileLoopCondition(const ExprPtr& condition)
{
    ValidateInWhileLoop("SetWhileLoopCondition");
    static_cast<WhileLoopContext*>(CurrentContext())->SetCondition(condition);
}

StmtPtr IRBuilder::EndWhileLoop(const Span& end_span)
{
    ValidateInWhileLoop("EndWhileLoop");

    auto* loop_ctx = static_cast<WhileLoopContext*>(CurrentContext());

    // Validate iter_args and return_vars match
    if (loop_ctx->GetIterArgs().size() != loop_ctx->GetReturnVars().size()) {
        // Pop context before throwing to maintain stack consistency
        context_stack_.pop_back();

        throw RuntimeError(MakeLoopReturnVarsMismatchMessage("While", loop_ctx->GetIterArgs().size(),
                                                             loop_ctx->GetReturnVars().size()));
    }

    auto body = MakeStmtBody(loop_ctx->GetStmts(), end_span);
    auto combinedSpan = MakeCombinedSpan(loop_ctx->GetBeginSpan(), end_span);
    // Create while statement
    auto while_stmt = std::make_shared<WhileStmt>(loop_ctx->GetCondition(), loop_ctx->GetIterArgs(), body,
                                                  loop_ctx->GetReturnVars(), combinedSpan);

    // Pop context
    context_stack_.pop_back();

    // Emit to parent context if it exists
    if (!context_stack_.empty()) {
        CurrentContext()->AddStmt(while_stmt);
    }

    return while_stmt;
}

// ========== If Statement Building ==========

void IRBuilder::BeginIf(const ExprPtr& condition, const Span& span)
{
    IRCHECK(!context_stack_.empty()) << "Cannot begin if statement: not inside a function or another valid context at "
                                     << span.ToString();
    context_stack_.push_back(std::make_shared<IfStmtContext>(condition, span));
}

void IRBuilder::BeginElse(const Span& span)
{
    ValidateInIf("BeginElse");

    auto* if_ctx = static_cast<IfStmtContext*>(CurrentContext());
    IRCHECK(!if_ctx->InElseBranch()) << "Cannot begin else branch: already in else branch at " << span.ToString();

    if_ctx->BeginElseBranch();
}

void IRBuilder::AddIfReturnVar(const VarPtr& var)
{
    ValidateInIf("AddIfReturnVar");
    static_cast<IfStmtContext*>(CurrentContext())->AddReturnVar(var);
}

StmtPtr IRBuilder::EndIf(const Span& end_span)
{
    ValidateInIf("EndIf");

    auto* if_ctx = static_cast<IfStmtContext*>(CurrentContext());

    std::vector<StmtPtr> then_stmts = if_ctx->GetStmts();
    std::vector<StmtPtr> else_stmts = if_ctx->GetElseStmts();
    std::vector<VarPtr> return_vars = if_ctx->GetReturnVars();

    // Promote differing but same-category scalar branch types (e.g. ``a if c else b``
    // yielding INT32 vs INDEX) to a common dtype before building the phi.
    if (if_ctx->InElseBranch()) {
        ReconcileIfReturnVarTypes(then_stmts, else_stmts, return_vars);
    }

    // Build then body
    StmtPtr then_body = MakeStmtBody(then_stmts, end_span);

    // Build else body (optional)
    std::optional<StmtPtr> else_body;
    if (if_ctx->InElseBranch() && !else_stmts.empty()) {
        else_body = MakeStmtBody(else_stmts, end_span);
    }

    // Combine begin and end spans
    auto combinedSpan = MakeCombinedSpan(if_ctx->GetBeginSpan(), end_span);

    // Create if statement
    auto if_stmt = std::make_shared<IfStmt>(if_ctx->GetCondition(), then_body, else_body, return_vars, combinedSpan);
    // Pop context
    context_stack_.pop_back();

    // Emit to parent context if it exists
    if (!context_stack_.empty()) {
        CurrentContext()->AddStmt(if_stmt);
    }

    return if_stmt;
}

// ========== Section Building ==========

void IRBuilder::BeginSection(SectionKind section_kind, const Span& span)
{
    CHECK(!context_stack_.empty()) << "Cannot begin section: not inside a function or another valid context at "
                                   << span.ToString();
    context_stack_.push_back(std::make_shared<SectionContext>(section_kind, span));
}

StmtPtr IRBuilder::EndSection(const Span& end_span)
{
    CHECK(!context_stack_.empty() && CurrentContext()->GetType() == BuildContext::Type::SECTION)
        << "Cannot end section: not inside a section context at " << end_span.ToString();

    auto* section_ctx = static_cast<SectionContext*>(CurrentContext());

    // Build body
    StmtPtr body;
    const auto& stmts = section_ctx->GetStmts();
    if (stmts.empty()) {
        body = std::make_shared<SeqStmts>(std::vector<StmtPtr>(), end_span);
    } else if (stmts.size() == 1) {
        body = stmts[0];
    } else {
        body = std::make_shared<SeqStmts>(stmts, end_span);
    }

    // Combine spans
    auto combinedSpan = MakeCombinedSpan(section_ctx->GetBeginSpan(), end_span);

    // Create section statement
    auto section_stmt = std::make_shared<SectionStmt>(section_ctx->GetSectionKind(), body, combinedSpan);

    // Pop context
    context_stack_.pop_back();

    // Emit to parent context if it exists
    if (!context_stack_.empty()) {
        CurrentContext()->AddStmt(section_stmt);
    }

    return section_stmt;
}

// ========== Program Building ==========

void IRBuilder::BeginProgram(const std::string& name, const Span& span)
{
    if (InProgram()) {
        throw RuntimeError("Cannot begin program '" + name + "': already inside program '" +
                           static_cast<ProgramContext*>(CurrentContext())->GetName() + "' at " +
                           CurrentContext()->GetBeginSpan().ToString());
    }

    context_stack_.push_back(std::make_shared<ProgramContext>(name, span));
}

void IRBuilder::AddFunction(const FunctionPtr& func)
{
    ValidateInProgram("AddFunction");
    static_cast<ProgramContext*>(CurrentContext())->AddFunction(func);
}

void IRBuilder::SetInsertPoint(std::shared_ptr<InsertPoint> ctx) { context_stack_.push_back(ctx); }

void IRBuilder::ClearInsertPoint() { context_stack_.pop_back(); }

ProgramPtr IRBuilder::EndProgram(const Span& end_span)
{
    ValidateInProgram("EndProgram");

    auto* prog_ctx = static_cast<ProgramContext*>(CurrentContext());

    // Combine begin and end spans
    auto combinedSpan = MakeCombinedSpan(prog_ctx->GetBeginSpan(), end_span);

    // Create program from functions vector
    auto program = std::make_shared<Program>(prog_ctx->GetFunctions(), prog_ctx->GetName(), combinedSpan);

    // Pop context
    context_stack_.pop_back();

    return program;
}

bool IRBuilder::InProgram() const
{
    for (const auto& ctx : context_stack_) {
        if (ctx->GetType() == BuildContext::Type::PROGRAM) {
            return true;
        }
    }
    return false;
}

std::vector<TypePtr> IRBuilder::GetFunctionReturnTypes(const std::string& func_name) const
{
    // Find the program context in the stack
    for (const auto& ctx : context_stack_) {
        if (ctx->GetType() == BuildContext::Type::PROGRAM) {
            auto* prog_ctx = static_cast<const ProgramContext*>(ctx.get());
            return prog_ctx->GetReturnTypes(func_name);
        }
    }
    return {};
}

// ========== Statement Recording ==========

void IRBuilder::Emit(const StmtPtr& stmt)
{
    if (context_stack_.empty()) {
        throw RuntimeError("Cannot emit statement: not inside any context");
    }

    auto* ctx = CurrentContext();
    ctx->AddStmt(stmt);
}

AssignStmtPtr IRBuilder::Assign(const VarPtr& var, const ExprPtr& value, const Span& span)
{
    auto assign = std::make_shared<AssignStmt>(var, value, span);
    Emit(assign);
    return assign;
}

VarPtr IRBuilder::Var(const std::string& name, const TypePtr& type, const Span& span)
{
    return std::make_shared<ir::Var>(name, type, span);
}

ReturnStmtPtr IRBuilder::Return(const std::vector<ExprPtr>& values, const Span& span)
{
    auto return_stmt = std::make_shared<ReturnStmt>(values, span);
    Emit(return_stmt);
    return return_stmt;
}

ReturnStmtPtr IRBuilder::Return(const Span& span)
{
    auto return_stmt = std::make_shared<ReturnStmt>(span);
    Emit(return_stmt);
    return return_stmt;
}

BreakStmtPtr IRBuilder::Break(const Span& span)
{
    auto break_stmt = std::make_shared<BreakStmt>(span);
    Emit(break_stmt);
    return break_stmt;
}

ContinueStmtPtr IRBuilder::Continue(const Span& span)
{
    auto continue_stmt = std::make_shared<ContinueStmt>(span);
    Emit(continue_stmt);
    return continue_stmt;
}

// ========== Context State Queries ==========

BuildContext* IRBuilder::CurrentContext()
{
    if (context_stack_.empty()) {
        return nullptr;
    }
    return context_stack_.back().get();
}

bool IRBuilder::InFunction() const
{
    for (const auto& ctx : context_stack_) {
        if (ctx->GetType() == BuildContext::Type::FUNCTION) {
            return true;
        }
    }
    return false;
}

bool IRBuilder::InLoop() const
{
    if (context_stack_.empty()) {
        return false;
    }
    return context_stack_.back()->GetType() == BuildContext::Type::FOR_LOOP;
}

bool IRBuilder::InIf() const
{
    if (context_stack_.empty()) {
        return false;
    }
    return context_stack_.back()->GetType() == BuildContext::Type::IF_STMT;
}

bool IRBuilder::InWhileLoop() const
{
    if (context_stack_.empty()) {
        return false;
    }
    return context_stack_.back()->GetType() == BuildContext::Type::WHILE_LOOP;
}

// ========== Private Helpers ==========

template <typename T>
T* IRBuilder::GetCurrentContextAs()
{
    auto* ctx = CurrentContext();
    if (!ctx) {
        return nullptr;
    }
    return dynamic_cast<T*>(ctx);
}

void IRBuilder::ValidateInFunction(const std::string& operation)
{
    IRCHECK(InFunction()) << operation << " can only be called inside a function context";
    IRCHECK(CurrentContext()->GetType() == BuildContext::Type::FUNCTION)
        << operation << " must be called directly in function context, not nested";
}

void IRBuilder::ValidateInLoop(const std::string& operation)
{
    IRCHECK(InLoop()) << operation << " can only be called inside a for loop context";
}

void IRBuilder::ValidateInIf(const std::string& operation)
{
    IRCHECK(InIf()) << operation << " can only be called inside an if statement context";
}

void IRBuilder::ValidateInWhileLoop(const std::string& operation)
{
    IRCHECK(InWhileLoop()) << operation << " can only be called inside a while loop context";
}

void IRBuilder::ValidateInProgram(const std::string& operation)
{
    IRCHECK(InProgram()) << operation << " can only be called inside a program context";
}

// ========== ProgramContext Implementation ==========

void ProgramContext::AddFunction(const FunctionPtr& func)
{
    INTERNAL_CHECK(func) << "Cannot add null function to program";

    // Store return types for this function
    return_types_[func->name_] = func->returnTypes_;

    functions_.push_back(func);
}

std::vector<TypePtr> ProgramContext::GetReturnTypes(const std::string& func_name) const
{
    auto it = return_types_.find(func_name);
    if (it != return_types_.end()) {
        return it->second;
    }
    return {};
}

} // namespace ir
} // namespace pypto
