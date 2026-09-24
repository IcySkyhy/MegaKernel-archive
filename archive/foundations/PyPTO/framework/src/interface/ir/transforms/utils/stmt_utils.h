/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_IR_TRANSFORMS_UTILS_STMT_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_STMT_UTILS_H_

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ir/stmt.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/base/visitor.h"

namespace pypto {
namespace ir {
namespace utils {

/// Collect all Var references from a single expression.
std::unordered_set<const Var*> CollectVarUses(const ExprPtr& expr);

/// Collect all Var references from a single stmt.
std::unordered_set<const Var*> CollectStmtVarRefs(const StmtPtr& stmt);

/// Collect all Var references from a list of stmts.
/// If skip_iter_updates, recursively skip YieldStmt/BreakStmt/ContinueStmt
/// (their values are iter_arg updates, not uses).
std::unordered_set<const Var*> CollectStmtVarRefs(const std::vector<StmtPtr>& stmts, bool skip_iter_updates = false);

/// Flatten a SeqStmtsPtr body to its stmts vector (returns empty if null).
const std::vector<StmtPtr>& FlattenBody(const SeqStmtsPtr& body);

/// Pointer-identity check: true if two stmt lists are identical by pointer.
bool StmtsEqual(const std::vector<StmtPtr>& a, const std::vector<StmtPtr>& b);

/// Build a SeqStmts from a vector of stmts.
SeqStmtsPtr MakeSeqBody(const std::vector<StmtPtr>& stmts, const Span& span);

/// Map from a Var to the expression that should replace it.
using VarExprMap = std::unordered_map<VarPtr, ExprPtr>;

/// Recursively lookup and replace Var references of the `expr`.
ExprPtr LookupVarInExpr(ExprPtr expr, const VarExprMap& varMap);

/// IRMutator that replaces every Var leaf equal to a key in `varMap` with the mapped expression.
/// Copy-on-write: subtrees with no substituted leaf are returned unchanged.
class VarSubstitutor : public IRMutator {
    using IRMutator::VisitExpr_;

public:
    explicit VarSubstitutor(const VarExprMap& varMap) : varMap_(varMap) {}

protected:
    ExprPtr VisitExpr_(const VarPtr& op) override { return LookupVarInExpr(op, varMap_); }

private:
    const VarExprMap& varMap_;
};

/// Deep-substitute variables in a statement tree per `varMap`: every Var leaf equal to a key is
/// replaced by the mapped expression. Copy-on-write (unchanged subtrees returned as-is); no-op
/// when `stmt` is null or `varMap` is empty.
inline StmtPtr SubstituteVars(StmtPtr stmt, const VarExprMap& varMap)
{
    if (!stmt || varMap.empty()) {
        return stmt;
    }
    VarSubstitutor mutator(varMap);
    return mutator.VisitStmt(stmt);
}

class DefVarCollector : public IRVisitor {
public:
    using IRVisitor::VisitStmt_;

    std::unordered_set<VarPtr> defs;

    void VisitStmt_(const TensorOpStmtPtr& op) override
    {
        for (auto& v : op->result_) {
            defs.insert(v);
        }
        for (const auto& token : op->result_token_) {
            defs.insert(token);
        }
        IRVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const IfStmtPtr& op) override
    {
        for (auto& v : op->returnVars_) {
            defs.insert(v);
        }
        IRVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const ForStmtPtr& op) override
    {
        for (auto& v : op->returnVars_) {
            defs.insert(v);
        }
        IRVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const WhileStmtPtr& op) override
    {
        for (auto& v : op->returnVars_) {
            defs.insert(v);
        }
        IRVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const AssignStmtPtr& op) override
    {
        defs.insert(op->var_);
        IRVisitor::VisitStmt_(op);
    }
};

inline std::unordered_set<VarPtr> CollectDefinedVars(StmtPtr stmt)
{
    DefVarCollector collector;
    if (stmt) {
        collector.VisitStmt(stmt);
    }
    return std::move(collector.defs);
}

} // namespace utils
} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_TRANSFORMS_UTILS_STMT_UTILS_H_
