/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/transforms/base/visitor.h"

#include <cstddef>

#include "core/logging.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/base/functor.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// Top-level entry points
void IRVisitor::VisitProgram(const ProgramPtr& program)
{
    for (const auto& entry : program->functions_) {
        VisitFunction(entry.second);
    }
}

void IRVisitor::VisitFunction(const FunctionPtr& func)
{
    for (auto& param : func->params_) {
        VisitExpr(param);
    }
    if (func->body_) {
        VisitStmt(func->body_);
    }
}

void IRVisitor::VisitExpr(const ExprPtr& expr) { ExprFunctor<void>::VisitExpr(expr); }

void IRVisitor::VisitStmt(const StmtPtr& stmt) { StmtFunctor<void>::VisitStmt(stmt); }

void IRVisitor::VisitType(const TypePtr& type)
{
    if (auto op = As<TensorType>(type)) {
        VisitType_(op);
        return;
    }
    if (auto op = As<TileType>(type)) {
        VisitType_(op);
        return;
    }
    if (auto op = As<ScalarType>(type)) {
        VisitType_(op);
        return;
    }
    if (auto op = As<TupleType>(type)) {
        VisitType_(op);
        return;
    }
    if (auto op = As<PtrType>(type)) {
        VisitType_(op);
        return;
    }
    if (auto op = As<TokenType>(type)) {
        VisitType_(op);
        return;
    }
    if (auto op = As<NoneType>(type)) {
        VisitType_(op);
        return;
    }
    if (auto op = As<LogicalTensorType>(type)) {
        VisitType_(op);
        return;
    }
    if (auto op = As<MemRefType>(type)) {
        VisitType_(op);
        return;
    }
    if (auto op = As<UnknownType>(type)) {
        VisitType_(op);
        return;
    }
}

void IRVisitor::VisitVarLike_(const VarPtr& op)
{
    if (auto tensor_type = As<TensorType>(op->GetType())) {
        for (const auto& dim : tensor_type->shape_) {
            VisitExpr(dim);
        }
    }
}

void IRVisitor::VisitExpr_(const VarPtr& op) { VisitVarLike_(op); }

void IRVisitor::VisitExpr_(const MemRefPtr& op)
{
    INTERNAL_CHECK_SPAN(op->addr_, op->span_) << "MemRef has null offset";
    VisitExpr(op->addr_);
}

void IRVisitor::VisitExpr_(const ConstIntPtr& op) { (void)op; }

void IRVisitor::VisitExpr_(const ConstFloatPtr& op) { (void)op; }

void IRVisitor::VisitExpr_(const ConstBoolPtr& op) { (void)op; }

void IRVisitor::VisitExpr_(const CallPtr& op)
{
    for (size_t i = 0; i < op->args_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->args_[i], op->span_) << "Call has null argument at index " << i;
        VisitExpr(op->args_[i]);
    }
}

void IRVisitor::VisitExpr_(const MakeTuplePtr& op)
{
    for (size_t i = 0; i < op->elements_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->elements_[i], op->span_) << "MakeTuple has null element at index " << i;
        VisitExpr(op->elements_[i]);
    }
}

void IRVisitor::VisitExpr_(const GetItemExprPtr& op)
{
    INTERNAL_CHECK_SPAN(op->value_, op->span_) << "GetItemExpr has null value";
    INTERNAL_CHECK_SPAN(op->slice_, op->span_) << "GetItemExpr has null slice";
    VisitExpr(op->value_);
    VisitExpr(op->slice_);
}

void IRVisitor::VisitExpr_(const ScalarExprPtr& op) { (void)op; }

void IRVisitor::VisitBinaryExpr_(const BinaryExprPtr& op)
{
    INTERNAL_CHECK_SPAN(op->left_, op->span_) << "BinaryExpr has null left operand";
    INTERNAL_CHECK_SPAN(op->right_, op->span_) << "BinaryExpr has null right operand";
    VisitExpr(op->left_);
    VisitExpr(op->right_);
}

void IRVisitor::VisitUnaryExpr_(const UnaryExprPtr& op)
{
    INTERNAL_CHECK_SPAN(op->operand_, op->span_) << "UnaryExpr has null operand";
    VisitExpr(op->operand_);
}

#define DEFINE_BINARY_VISITOR(OpType) \
    void IRVisitor::VisitExpr_(const OpType##Ptr& op) { VisitBinaryExpr_(op); }

DEFINE_BINARY_VISITOR(Add)
DEFINE_BINARY_VISITOR(Sub)
DEFINE_BINARY_VISITOR(Mul)
DEFINE_BINARY_VISITOR(FloorDiv)
DEFINE_BINARY_VISITOR(FloorMod)
DEFINE_BINARY_VISITOR(FloatDiv)
DEFINE_BINARY_VISITOR(Min)
DEFINE_BINARY_VISITOR(Max)
DEFINE_BINARY_VISITOR(Pow)
DEFINE_BINARY_VISITOR(Eq)
DEFINE_BINARY_VISITOR(Ne)
DEFINE_BINARY_VISITOR(Lt)
DEFINE_BINARY_VISITOR(Le)
DEFINE_BINARY_VISITOR(Gt)
DEFINE_BINARY_VISITOR(Ge)
DEFINE_BINARY_VISITOR(And)
DEFINE_BINARY_VISITOR(Or)
DEFINE_BINARY_VISITOR(Xor)
DEFINE_BINARY_VISITOR(BitAnd)
DEFINE_BINARY_VISITOR(BitOr)
DEFINE_BINARY_VISITOR(BitXor)
DEFINE_BINARY_VISITOR(BitShiftLeft)
DEFINE_BINARY_VISITOR(BitShiftRight)

#undef DEFINE_BINARY_VISITOR

#define DEFINE_UNARY_VISITOR(OpType) \
    void IRVisitor::VisitExpr_(const OpType##Ptr& op) { VisitUnaryExpr_(op); }

DEFINE_UNARY_VISITOR(Abs)
DEFINE_UNARY_VISITOR(Neg)
DEFINE_UNARY_VISITOR(Not)
DEFINE_UNARY_VISITOR(BitNot)
DEFINE_UNARY_VISITOR(Cast)

#undef DEFINE_UNARY_VISITOR

void IRVisitor::VisitStmt_(const AssignStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->var_, op->span_) << "AssignStmt has null var";
    INTERNAL_CHECK_SPAN(op->value_, op->span_) << "AssignStmt has null value";
    VisitExpr(op->var_);
    VisitExpr(op->value_);
}

void IRVisitor::VisitStmt_(const IfStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->condition_, op->span_) << "IfStmt has null condition";
    VisitExpr(op->condition_);
    INTERNAL_CHECK_SPAN(op->thenBody_, op->span_) << "IfStmt has null thenBody";
    VisitStmt(op->thenBody_);
    if (op->elseBody_.has_value()) {
        INTERNAL_CHECK_SPAN(*op->elseBody_, op->span_) << "IfStmt has null elseBody";
        VisitStmt(*op->elseBody_);
    }
    for (size_t i = 0; i < op->returnVars_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->returnVars_[i], op->span_) << "IfStmt has null returnVars at index " << i;
        VisitExpr(op->returnVars_[i]);
    }
}

void IRVisitor::VisitStmt_(const YieldStmtPtr& op)
{
    for (size_t i = 0; i < op->value_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->value_[i], op->span_) << "YieldStmt has null value at index " << i;
        VisitExpr(op->value_[i]);
    }
}

void IRVisitor::VisitStmt_(const ReturnStmtPtr& op)
{
    for (size_t i = 0; i < op->value_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->value_[i], op->span_) << "ReturnStmt has null value at index " << i;
        VisitExpr(op->value_[i]);
    }
}

void IRVisitor::VisitStmt_(const ForStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->loopVar_, op->span_) << "ForStmt has null loopVar";
    INTERNAL_CHECK_SPAN(op->start_, op->span_) << "ForStmt has null start";
    INTERNAL_CHECK_SPAN(op->stop_, op->span_) << "ForStmt has null stop";
    INTERNAL_CHECK_SPAN(op->step_, op->span_) << "ForStmt has null step";
    VisitExpr(op->loopVar_);
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    for (size_t i = 0; i < op->iterArgs_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->iterArgs_[i]->initValue_, op->span_)
            << "ForStmt has null iterArgs initValue at index " << i;
        VisitExpr(op->iterArgs_[i]->iterVar_);
        VisitExpr(op->iterArgs_[i]->initValue_);
    }
    INTERNAL_CHECK_SPAN(op->body_, op->span_) << "ForStmt has null body";
    VisitStmt(op->body_);
    for (size_t i = 0; i < op->returnVars_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->returnVars_[i], op->span_) << "ForStmt has null returnVars at index " << i;
        VisitExpr(op->returnVars_[i]);
    }
}

void IRVisitor::VisitStmt_(const WhileStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->condition_, op->span_) << "WhileStmt has null condition";
    VisitExpr(op->condition_);
    for (size_t i = 0; i < op->iterArgs_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->iterArgs_[i]->initValue_, op->span_)
            << "WhileStmt has null iterArgs initValue at index " << i;
        VisitExpr(op->iterArgs_[i]->iterVar_);
        VisitExpr(op->iterArgs_[i]->initValue_);
    }
    INTERNAL_CHECK_SPAN(op->body_, op->span_) << "WhileStmt has null body";
    VisitStmt(op->body_);
    for (size_t i = 0; i < op->returnVars_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->returnVars_[i], op->span_) << "WhileStmt has null returnVars at index " << i;
        VisitExpr(op->returnVars_[i]);
    }
}

void IRVisitor::VisitStmt_(const SeqStmtsPtr& op)
{
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
        INTERNAL_CHECK_SPAN(op->stmts_[i], op->span_) << "SeqStmts has null statement at index " << i;
        VisitStmt(op->stmts_[i]);
    }
}

void IRVisitor::VisitStmt_(const SectionStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->body_, op->span_) << "SectionStmt has null body";
    VisitStmt(op->body_);
}

void IRVisitor::VisitStmt_(const EvalStmtPtr& op)
{
    INTERNAL_CHECK_SPAN(op->expr_, op->span_) << "EvalStmt has null expr";
    VisitExpr(op->expr_);
}

void IRVisitor::VisitStmt_(const BreakStmtPtr& op)
{
    for (auto& v : op->value_) {
        VisitExpr(v);
    }
}

void IRVisitor::VisitStmt_(const ContinueStmtPtr& op)
{
    for (auto& v : op->value_) {
        VisitExpr(v);
    }
}

void IRVisitor::VisitStmt_(const ScalarOpStmtPtr& op)
{
    VisitExpr(op->result_);
    if (op->result_token_) {
        VisitExpr(op->result_token_);
    }
    for (auto v : op->args_) {
        VisitExpr(v);
    }
}

void IRVisitor::VisitStmt_(const TensorOpStmtPtr& op)
{
    for (auto v : op->result_) {
        VisitExpr(v);
    }
    for (auto v : op->args_) {
        VisitExpr(v);
    }
    for (auto v : op->tokens_) {
        VisitExpr(v);
    }
    for (const auto& token : op->result_token_) {
        VisitExpr(token);
    }
}

void IRVisitor::VisitStmt_(const StmtPtr&) {}

// Type handlers - default: no-op
void IRVisitor::VisitType_(const UnknownTypePtr&) {}
void IRVisitor::VisitType_(const ScalarTypePtr&) {}
void IRVisitor::VisitType_(const TensorTypePtr&) {}
void IRVisitor::VisitType_(const TileTypePtr&) {}
void IRVisitor::VisitType_(const TupleTypePtr&) {}
void IRVisitor::VisitType_(const MemRefTypePtr&) {}
void IRVisitor::VisitType_(const PtrTypePtr&) {}
void IRVisitor::VisitType_(const TokenTypePtr&) {}
void IRVisitor::VisitType_(const NoneTypePtr&) {}
void IRVisitor::VisitType_(const LogicalTensorTypePtr&) {}

} // namespace ir
} // namespace pypto
