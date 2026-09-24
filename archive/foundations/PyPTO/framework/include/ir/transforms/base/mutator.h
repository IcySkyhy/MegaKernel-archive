/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_IR_TRANSFORMS_BASE_MUTATOR_H_
#define PYPTO_IR_TRANSFORMS_BASE_MUTATOR_H_

#include <unordered_map>

#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/base/functor.h"

namespace pypto {
namespace ir {

/**
 * \brief IR mutator for immutable transformations
 *
 * Provides default implementations that recursively transform the IR tree.
 * Returns new ExprPtr or StmtPtr for transformed IR nodes, respecting immutability.
 * Uses copy-on-write: if children are unchanged, returns the original shared_ptr.
 */
class IRMutator : public ExprFunctor<ExprPtr>, public StmtFunctor<StmtPtr> {
public:
    ~IRMutator() override = default;

    /// Top-level entry points for mutating a full program or function.
    virtual ProgramPtr VisitProgram(const ProgramPtr& program);
    virtual FunctionPtr VisitFunction(const FunctionPtr& func);

    // Override base class methods
    ExprPtr VisitExpr(const ExprPtr& expr) override;
    StmtPtr VisitStmt(const StmtPtr& stmt) override;

protected:
    // Leaf nodes - return as-is by default
    ExprPtr VisitExpr_(const VarPtr& op) override;
    ExprPtr VisitExpr_(const MemRefPtr& op) override;
    ExprPtr VisitExpr_(const ConstIntPtr& op) override;
    ExprPtr VisitExpr_(const ConstFloatPtr& op) override;
    ExprPtr VisitExpr_(const ConstBoolPtr& op) override;
    ExprPtr VisitExpr_(const CallPtr& op) override;
    ExprPtr VisitExpr_(const MakeTuplePtr& op) override;
    ExprPtr VisitExpr_(const ScalarExprPtr& op) override;
    ExprPtr VisitExpr_(const GetItemExprPtr& op) override;

    // Binary operations - reconstruct with mutated children
    ExprPtr VisitExpr_(const AddPtr& op) override;
    ExprPtr VisitExpr_(const SubPtr& op) override;
    ExprPtr VisitExpr_(const MulPtr& op) override;
    ExprPtr VisitExpr_(const FloorDivPtr& op) override;
    ExprPtr VisitExpr_(const FloorModPtr& op) override;
    ExprPtr VisitExpr_(const FloatDivPtr& op) override;
    ExprPtr VisitExpr_(const MinPtr& op) override;
    ExprPtr VisitExpr_(const MaxPtr& op) override;
    ExprPtr VisitExpr_(const PowPtr& op) override;
    ExprPtr VisitExpr_(const EqPtr& op) override;
    ExprPtr VisitExpr_(const NePtr& op) override;
    ExprPtr VisitExpr_(const LtPtr& op) override;
    ExprPtr VisitExpr_(const LePtr& op) override;
    ExprPtr VisitExpr_(const GtPtr& op) override;
    ExprPtr VisitExpr_(const GePtr& op) override;
    ExprPtr VisitExpr_(const AndPtr& op) override;
    ExprPtr VisitExpr_(const OrPtr& op) override;
    ExprPtr VisitExpr_(const XorPtr& op) override;
    ExprPtr VisitExpr_(const BitAndPtr& op) override;
    ExprPtr VisitExpr_(const BitOrPtr& op) override;
    ExprPtr VisitExpr_(const BitXorPtr& op) override;
    ExprPtr VisitExpr_(const BitShiftLeftPtr& op) override;
    ExprPtr VisitExpr_(const BitShiftRightPtr& op) override;

    // Unary operations - reconstruct with mutated operand
    ExprPtr VisitExpr_(const AbsPtr& op) override;
    ExprPtr VisitExpr_(const NegPtr& op) override;
    ExprPtr VisitExpr_(const NotPtr& op) override;
    ExprPtr VisitExpr_(const BitNotPtr& op) override;
    ExprPtr VisitExpr_(const CastPtr& op) override;

    // Statement types
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override;
    StmtPtr VisitStmt_(const IfStmtPtr& op) override;
    StmtPtr VisitStmt_(const YieldStmtPtr& op) override;
    StmtPtr VisitStmt_(const ReturnStmtPtr& op) override;
    StmtPtr VisitStmt_(const ForStmtPtr& op) override;
    StmtPtr VisitStmt_(const WhileStmtPtr& op) override;
    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override;
    StmtPtr VisitStmt_(const SectionStmtPtr& op) override;
    StmtPtr VisitStmt_(const EvalStmtPtr& op) override;
    StmtPtr VisitStmt_(const BreakStmtPtr& op) override;
    StmtPtr VisitStmt_(const ContinueStmtPtr& op) override;
    StmtPtr VisitStmt_(const ScalarOpStmtPtr& op) override;
    StmtPtr VisitStmt_(const TensorOpStmtPtr& op) override;
    StmtPtr VisitStmt_(const StmtPtr& op) override;

    /// Override to handle ALL binary expressions (Add, Sub, Mul, ...) in one method.
    /// Default: visits children, reconstructs if changed (copy-on-write).
    virtual ExprPtr VisitBinaryExpr_(const BinaryExprPtr& op);

    /// Override to handle ALL unary expressions (Abs, Neg, Not, ...) in one method.
    /// Default: visits operand, reconstructs if changed (copy-on-write).
    virtual ExprPtr VisitUnaryExpr_(const UnaryExprPtr& op);

    /// Visit every expression in `exprs`, returning the new vector if any element changed.
    std::pair<std::vector<ExprPtr>, bool> VisitExprList(const std::vector<ExprPtr>& exprs);

    /// Visit every Var in `vars` (via VisitExpr, cast back to VarPtr), returning the new vector
    /// if any element changed.
    std::pair<std::vector<VarPtr>, bool> VisitVarList(const std::vector<VarPtr>& vars);

    /// Visit every IterArg's initValue_ (via VisitExpr), rebuilding the IterArg when it changed,
    /// and register old->new iterVar mappings in var_remap_ so the following condition/body visits
    /// substitute references. Returns the new vector and whether any IterArg changed. The caller
    /// must erase the registered mappings after visiting the body (IterArg iterVar_ is reused, so
    /// the mappings are currently inert, but kept for correctness).
    std::pair<std::vector<IterArgPtr>, bool> VisitIterArgList(const std::vector<IterArgPtr>& iterArgs);

    /// Pointer remapping for variables whose definitions changed during mutation.
    /// Used to keep body references consistent with new definition pointers
    /// (e.g., when IterArg's initValue_ changes, creating a new IterArg object).
    /// Checked in both VisitExpr_(VarPtr) and VisitExpr_(IterArgPtr) for extensibility.
    std::unordered_map<const Expr*, ExprPtr> var_remap_;
};

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_TRANSFORMS_BASE_MUTATOR_H_
