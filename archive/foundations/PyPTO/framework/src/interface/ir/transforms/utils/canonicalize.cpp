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

#include "ir/transforms/utils/canonicalize.h"

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ir/expr.h"
#include "ir/stmt.h"
#include "ir/transforms/base/visitor.h"
#include "stmt_utils.h"

#include "interface/tensor/ir.h"

namespace pypto {
namespace ir {

namespace {

using utils::CollectStmtVarRefs;
using utils::CollectVarUses;
using utils::StmtsEqual;

std::vector<ExprPtr> MakeReturnVars(const std::vector<ExprPtr>& values, const std::vector<size_t>& kept_indices)
{
    std::vector<ExprPtr> result;
    for (size_t idx : kept_indices) {
        INTERNAL_CHECK(idx < values.size());
        result.push_back(values[idx]);
    }
    return result;
}

void FilterTrailingTerminator(std::vector<StmtPtr>& stmts, const std::vector<size_t>& keptIndices)
{
    auto stmt = stmts.back();
    if (auto yield = std::dynamic_pointer_cast<const YieldStmt>(stmt)) {
        stmts.back() = std::make_shared<YieldStmt>(MakeReturnVars(yield->value_, keptIndices), yield->span_);
    } else if (auto brk = std::dynamic_pointer_cast<const BreakStmt>(stmt)) {
        stmts.back() = std::make_shared<BreakStmt>(MakeReturnVars(brk->value_, keptIndices), brk->span_);
    } else if (auto cont = std::dynamic_pointer_cast<const ContinueStmt>(stmt)) {
        stmts.back() = std::make_shared<ContinueStmt>(MakeReturnVars(cont->value_, keptIndices), cont->span_);
    }
}

/// A (defs, uses) pair used for carry-liveness propagation.
struct DefSite {
    std::vector<const Var*> defs;
    std::unordered_set<const Var*> uses;
};

class LivenessCollector : public IRVisitor {
public:
    using IRVisitor::VisitStmt_;

    std::vector<DefSite> sites;
    std::vector<const Var*> liveRoots;

private:
    std::vector<const std::vector<VarPtr>*> rvStack_;

    bool IsSideEffectOp(const std::string& op)
    {
        return op == "ASSEMBLE" || op == "ASSEMBLE_SSA" || op == "ATOMIC_RMW";
    }

    void AddSite(std::vector<const Var*> defs, std::unordered_set<const Var*> uses)
    {
        sites.push_back({std::move(defs), std::move(uses)});
    }

    void AddLiveUses(const ExprPtr& expr)
    {
        auto uses = CollectVarUses(expr);
        liveRoots.insert(liveRoots.end(), uses.begin(), uses.end());
    }

    void AddTerminator(const std::vector<ExprPtr>& values)
    {
        const auto& returnVars = *rvStack_.back();
        for (size_t k = 0; k < values.size(); ++k) {
            AddSite({returnVars[k].get()}, CollectVarUses(values[k]));
        }
    }

    void VisitStmt_(const AssignStmtPtr& op) override { AddSite({op->var_.get()}, CollectVarUses(op->value_)); }

    void VisitStmt_(const TensorOpStmtPtr& op) override
    {
        std::unordered_set<const Var*> uses;

        for (const auto& arg : op->args_) {
            auto r = CollectVarUses(arg);
            uses.insert(r.begin(), r.end());
        }
        for (const auto& token : op->tokens_) {
            uses.insert(token.get());
        }
        CollectScalarVarRefs(op, uses);

        std::vector<const Var*> defs;
        for (const auto& res : op->result_) {
            defs.push_back(res.get());
            if (IsSideEffectOp(op->opcode_)) {
                liveRoots.push_back(res.get());
            }
        }
        for (const auto& token : op->result_token_) {
            defs.push_back(token.get());
        }

        AddSite(std::move(defs), uses);
    }

    void VisitStmt_(const YieldStmtPtr& op) override { AddTerminator(op->value_); }
    void VisitStmt_(const BreakStmtPtr& op) override { AddTerminator(op->value_); }
    void VisitStmt_(const ContinueStmtPtr& op) override { AddTerminator(op->value_); }

    void VisitStmt_(const IfStmtPtr& op) override
    {
        rvStack_.push_back(&op->returnVars_);
        VisitStmt(op->thenBody_);
        if (op->elseBody_.has_value()) {
            VisitStmt(*op->elseBody_);
        }
        rvStack_.pop_back();
    }

    void VisitStmt_(const std::vector<VarPtr>& returnVars, const std::vector<IterArgPtr>& iterArgs, const StmtPtr& body)
    {
        INTERNAL_CHECK(returnVars.size() == iterArgs.size()) << "for stmt must have one return var per iter arg";
        auto n = iterArgs.size();
        for (size_t k = 0; k < n; ++k) {
            if (IsSameRawTensor(iterArgs[k]->iterVar_, returnVars[k])) {
                AddLiveUses(iterArgs[k]->iterVar_);
            }
        }
        rvStack_.push_back(&returnVars);
        VisitStmt(body);
        rvStack_.pop_back();
    }

    void VisitStmt_(const ForStmtPtr& op) override { VisitStmt_(op->returnVars_, op->iterArgs_, op->body_); }

    void VisitStmt_(const WhileStmtPtr& op) override { VisitStmt_(op->returnVars_, op->iterArgs_, op->body_); }
};

std::unordered_set<const Var*> ComputeLiveVars(const StmtPtr& stmt, const std::unordered_set<const Var*>& afterRefs)
{
    LivenessCollector collector;
    collector.VisitStmt(stmt);

    std::unordered_set<const Var*> live(afterRefs);
    for (auto root : collector.liveRoots) {
        live.insert(root);
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& site : collector.sites) {
            bool anyDefLive = false;
            for (auto def : site.defs) {
                if (live.count(def)) {
                    anyDefLive = true;
                    break;
                }
            }
            if (!anyDefLive) {
                continue;
            }
            for (auto use : site.uses) {
                if (live.insert(use).second) {
                    changed = true;
                }
            }
        }
    }
    return live;
}

template <typename T, typename Rebuild>
StmtPtr CanonicalizeLoopImpl(const T& stmt, const std::unordered_set<const Var*>& afterRefs, Rebuild rebuild)
{
    if (stmt->iterArgs_.empty()) {
        auto body = CanonicalizeSeqStmts(stmt->body_, {});
        if (body == stmt->body_)
            return stmt;
        return rebuild(stmt->iterArgs_, body, stmt->returnVars_);
    }

    auto live = ComputeLiveVars(stmt, afterRefs);
    std::vector<size_t> keptIndices;
    for (size_t i = 0; i < stmt->returnVars_.size(); ++i) {
        if (afterRefs.count(stmt->returnVars_[i].get()) || live.count(stmt->iterArgs_[i]->iterVar_.get())) {
            keptIndices.push_back(i);
        }
    }

    // always rebuild the body as yield may carry dead return vars / yield args at deeper levels.
    auto body = CanonicalizeSeqStmts(stmt->body_, keptIndices);
    if (keptIndices.size() == stmt->iterArgs_.size() && body == stmt->body_) {
        return stmt;
    }

    std::vector<IterArgPtr> iterArgs;
    std::vector<VarPtr> returnVars;
    for (size_t i : keptIndices) {
        iterArgs.push_back(stmt->iterArgs_[i]);
        returnVars.push_back(stmt->returnVars_[i]);
    }
    return rebuild(iterArgs, body, returnVars);
}

StmtPtr CanonicalizeIfStmt(IfStmtPtr& ifStmt, const std::unordered_set<const Var*>& suffix)
{
    std::vector<size_t> keptIndices;
    std::vector<VarPtr> returnVars;
    for (size_t i = 0; i < ifStmt->returnVars_.size(); ++i) {
        if (suffix.count(ifStmt->returnVars_[i].get()) > 0) {
            keptIndices.push_back(i);
            returnVars.push_back(ifStmt->returnVars_[i]);
        }
    }

    // Always recurse into then/else body: even when returnVars are unchanged,
    std::optional<SeqStmtsPtr> elseBody;
    auto thenBody = CanonicalizeSeqStmts(ifStmt->thenBody_, keptIndices);
    if (ifStmt->elseBody_) {
        elseBody = CanonicalizeSeqStmts(ifStmt->elseBody_.value(), keptIndices);
    }

    bool bodyChanged = (thenBody.get() != ifStmt->thenBody_.get());
    if (ifStmt->elseBody_ && elseBody.has_value()) {
        bodyChanged = bodyChanged || (elseBody.value().get() != ifStmt->elseBody_.value().get());
    }
    if (returnVars.size() == ifStmt->returnVars_.size() && !bodyChanged) {
        return ifStmt;
    }

    return std::make_shared<const IfStmt>(ifStmt->condition_, thenBody, elseBody, std::move(returnVars), ifStmt->span_);
}

StmtPtr CanonicalizeSectionStmt(SectionStmtPtr& section)
{
    auto newBody = CanonicalizeSeqStmts(section->body_, {});
    if (newBody == section->body_) {
        return section;
    }
    return std::make_shared<const SectionStmt>(section->sectionKind_, newBody, section->span_);
}
} // namespace

SeqStmtsPtr CanonicalizeSeqStmts(const SeqStmtsPtr& seq, const std::vector<size_t>& keptIndices)
{
    if (seq->stmts_.empty())
        return seq;

    auto& stmts = seq->stmts_;

    // the variable used after the block, could only be the returnVars of the block
    // first handle the returnVars in terminate statement
    FilterTrailingTerminator(stmts, keptIndices);

    std::vector<StmtPtr> result(stmts.size());
    std::unordered_set<const Var*> suffix;
    for (size_t idx = stmts.size(); idx-- > 0;) {
        if (auto forStmt = std::dynamic_pointer_cast<const ForStmt>(stmts[idx])) {
            result[idx] = CanonicalizeLoopImpl(
                forStmt, suffix,
                [&forStmt](const std::vector<IterArgPtr>& ia, const SeqStmtsPtr& body, const std::vector<VarPtr>& rv) {
                    return std::make_shared<const ForStmt>(forStmt->loopVar_, forStmt->start_, forStmt->stop_,
                                                           forStmt->step_, ia, body, rv, forStmt->span_,
                                                           forStmt->attrs_);
                });
        } else if (auto whileStmt = std::dynamic_pointer_cast<const WhileStmt>(stmts[idx])) {
            result[idx] = CanonicalizeLoopImpl(
                whileStmt, suffix,
                [whileStmt](const std::vector<IterArgPtr>& ia, const SeqStmtsPtr& body, const std::vector<VarPtr>& rv) {
                    return std::make_shared<const WhileStmt>(whileStmt->condition_, ia, body, rv, whileStmt->span_);
                });
        } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmts[idx])) {
            result[idx] = CanonicalizeIfStmt(if_stmt, suffix);
        } else if (auto section = std::dynamic_pointer_cast<const SectionStmt>(stmts[idx])) {
            result[idx] = CanonicalizeSectionStmt(section);
        } else {
            result[idx] = stmts[idx];
        }
        auto refs = CollectStmtVarRefs(result[idx]);
        suffix.insert(refs.begin(), refs.end());
    }

    if (StmtsEqual(result, seq->stmts_)) {
        return seq;
    }
    return std::make_shared<SeqStmts>(result, seq->span_);
}

} // namespace ir
} // namespace pypto
