/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <unordered_set>

#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/transforms/merge_stmts_pass.h"
#include "ir/transforms/utils/stmt_utils.h"

#include "tilefwk/symbolic_scalar.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/tensor/ir_tensor_op_rebuild.h"

using npu::tile_fwk::SatStatus;
using npu::tile_fwk::SymbolicScalar;

namespace pypto::ir {

namespace {

using npu::tile_fwk::LogicalTensor;
using npu::tile_fwk::Operation;
using utils::LookupVarInExpr;
using utils::SubstituteVars;
using utils::VarExprMap;

constexpr char kLoopCondsAttr[] = "_loop_conds";

struct BranchClassification {
    bool thenDead;
    bool elseDead;
    std::vector<SymbolicScalar> thenConds;
    std::vector<SymbolicScalar> elseConds;
};

StmtPtr SubstituteStmt(StmtPtr stmt, VarExprMap& varMap, std::unordered_set<VarPtr>& clonedVars)
{
    if (!stmt || varMap.empty()) {
        return stmt;
    }

    stmt = SubstituteVars(stmt, varMap);
    auto top = As<TensorOpStmt>(stmt);
    if (!top) {
        return stmt;
    }

    auto op = std::dynamic_pointer_cast<Operation>(std::const_pointer_cast<TensorOpStmt>(top));
    if (!op) {
        return stmt;
    }

    bool changed = false;
    for (auto& attr : op->GetDynamicAttributeList()) {
        auto& s = attr.get();
        if (s.SubstituteVars(varMap).Raw() != s.Raw()) {
            changed = true;
            break;
        }
    }
    if (!changed) {
        return stmt;
    }

    std::vector<VarPtr> newResults;
    newResults.reserve(op->result_.size());
    for (auto& var : op->result_) {
        auto lt = std::dynamic_pointer_cast<const LogicalTensor>(var);
        if (!lt) {
            newResults.push_back(var);
            continue;
        }
        bool needsClone = false;
        auto ltMut = std::const_pointer_cast<LogicalTensor>(lt);
        for (auto& shape : ltMut->GetDynValidShape()) {
            if (shape.SubstituteVars(varMap).Raw() != shape.Raw()) {
                needsClone = true;
                break;
            }
        }
        if (needsClone) {
            auto resultClone = lt->Clone(true);
            varMap[var] = resultClone;
            clonedVars.insert(var);
            newResults.push_back(resultClone);
        } else {
            newResults.push_back(var);
        }
    }

    std::vector<ExprPtr> newArgs;
    newArgs.reserve(op->args_.size());
    for (auto& arg : op->args_) {
        newArgs.push_back(LookupVarInExpr(arg, varMap));
    }

    auto cloned = npu::tile_fwk::RebuildTensorOpStmt(top, newResults, op->result_token_, newArgs, op->tokens_,
                                                     Span::Unknown());
    op = std::dynamic_pointer_cast<Operation>(std::const_pointer_cast<Stmt>(cloned));
    for (auto& attr : op->GetDynamicAttributeList()) {
        auto& s = attr.get();
        s = s.SubstituteVars(varMap).Simplify();
    }
    return cloned;
}

void BuildYieldVarMap(IfStmtPtr ifStmt, SeqStmtsPtr body, VarExprMap& varMap)
{
    if (body->stmts_.empty() || ifStmt->returnVars_.empty()) {
        return;
    }
    auto& lastStmt = body->stmts_.back();
    if (auto yieldStmt = As<YieldStmt>(lastStmt)) {
        for (size_t i = 0; i < ifStmt->returnVars_.size() && i < yieldStmt->value_.size(); ++i) {
            varMap[ifStmt->returnVars_[i]] = yieldStmt->value_[i];
        }
    }
}

void BuildYieldVarMap(ForStmtPtr forStmt, VarExprMap& varMap)
{
    if (forStmt->body_->stmts_.empty() || forStmt->returnVars_.empty()) {
        return;
    }
    auto& lastStmt = forStmt->body_->stmts_.back();
    if (auto c = As<ContinueStmt>(lastStmt)) {
        for (size_t i = 0; i < forStmt->returnVars_.size() && i < c->value_.size(); ++i) {
            varMap[forStmt->returnVars_[i]] = c->value_[i];
        }
    }
    if (auto y = As<YieldStmt>(lastStmt)) {
        for (size_t i = 0; i < forStmt->returnVars_.size() && i < y->value_.size(); ++i) {
            varMap[forStmt->returnVars_[i]] = y->value_[i];
        }
    }
}

static std::vector<VarPtr> CollectOutputVars(const std::vector<StmtPtr>& stmts)
{
    std::vector<VarPtr> outputs;
    for (auto& stmt : stmts) {
        if (auto t = As<TensorOpStmt>(stmt)) {
            for (auto& var : t->result_) {
                outputs.push_back(var);
            }
            outputs.insert(outputs.end(), t->result_token_.begin(), t->result_token_.end());
        } else if (auto f = As<ForStmt>(stmt)) {
            for (auto& var : f->returnVars_) {
                outputs.push_back(var);
            }
        } else if (auto w = As<WhileStmt>(stmt)) {
            for (auto& var : w->returnVars_) {
                outputs.push_back(var);
            }
        } else if (auto i = As<IfStmt>(stmt)) {
            for (auto& var : i->returnVars_) {
                outputs.push_back(var);
            }
        }
    }
    return outputs;
}

std::vector<StmtPtr> RemoveLastYieldStmt(const std::vector<StmtPtr>& stmts)
{
    std::vector<StmtPtr> filteredStmts;
    for (size_t i = 0; i < stmts.size(); ++i) {
        if (i == stmts.size() - 1 && IsA<YieldStmt>(stmts[i])) {
            continue;
        }
        filteredStmts.push_back(stmts[i]);
    }
    return filteredStmts;
}

void ExtendYieldMap(const StmtPtr& s, VarExprMap& yieldMap)
{
    if (auto ifStmt = As<IfStmt>(s)) {
        BuildYieldVarMap(ifStmt, ifStmt->thenBody_, yieldMap);
    } else if (auto forStmt = As<ForStmt>(s)) {
        BuildYieldVarMap(forStmt, yieldMap);
    }
}

std::vector<ExprPtr> ExtractYieldValues(SeqStmtsPtr body)
{
    if (!body || body->stmts_.empty()) {
        return {};
    }
    auto& tailStmt = body->stmts_.back();
    if (!IsA<YieldStmt>(tailStmt)) {
        return {};
    }
    auto yieldStmt = As<YieldStmt>(tailStmt);
    return yieldStmt->value_;
}

// Append `stmts` onto one branch body: substitute each through the running yield-map (seeded from
// this branch's own yield), extend the map as new return vars appear, then re-emit the trailing
// yield with `newVars` appended. A null `branch` (missing else) starts from an empty body.
SeqStmtsPtr BuildAppendedBranch(IfStmtPtr ifStmt, SeqStmtsPtr branch, const std::vector<StmtPtr>& stmts,
                                const std::vector<VarPtr>& newVars, Span span)
{
    std::vector<StmtPtr> out;
    VarExprMap yieldMap;
    std::unordered_set<VarPtr> clonedVars;
    if (branch) {
        out = RemoveLastYieldStmt(branch->stmts_);
        BuildYieldVarMap(ifStmt, branch, yieldMap);
    }
    for (auto& s : stmts) {
        out.push_back(SubstituteStmt(s, yieldMap, clonedVars));
        ExtendYieldMap(s, yieldMap);
    }
    auto values = ExtractYieldValues(branch);
    for (auto& v : newVars) {
        if (clonedVars.count(v)) {
            values.push_back(LookupVarInExpr(v, yieldMap));
        } else {
            values.push_back(v);
        }
    }
    out.push_back(std::make_shared<YieldStmt>(values, span));
    return std::make_shared<SeqStmts>(out, span);
}

// Prepend `stmts` ahead of one branch body: emit each (substituted through the running yield-map,
// extended as new return vars appear), then the original branch stmts minus their trailing yield,
// then a fresh trailing yield with `newVars` followed by the branch's own yield values. A
// disengaged `branch` (missing else) omits the original-body part.
SeqStmtsPtr BuildPrependedBranch(const std::vector<StmtPtr>& stmts, std::optional<SeqStmtsPtr> branch,
                                 const std::vector<VarPtr>& newVars, Span span)
{
    std::vector<StmtPtr> out;
    VarExprMap yieldMap;
    std::unordered_set<VarPtr> clonedVars;
    for (auto& s : stmts) {
        out.push_back(SubstituteStmt(s, yieldMap, clonedVars));
        ExtendYieldMap(s, yieldMap);
    }
    if (branch) {
        for (auto& s : RemoveLastYieldStmt(branch.value()->stmts_)) {
            out.push_back(SubstituteStmt(s, yieldMap, clonedVars));
        }
    }
    std::vector<ExprPtr> values;
    values.reserve(newVars.size());
    for (auto& v : newVars) {
        clonedVars.count(v) ? values.push_back(LookupVarInExpr(v, yieldMap)) : values.push_back(v);
    }
    auto orig = ExtractYieldValues(branch.value_or(nullptr));
    values.insert(values.end(), orig.begin(), orig.end());
    out.push_back(std::make_shared<YieldStmt>(values, span));
    return std::make_shared<SeqStmts>(out, span);
}

// Classify each branch as dead under the accumulated condition path: an empty body, or a branch
// whose conditions (path + cond / !cond) are SAT-unsatisfiable.
BranchClassification ClassifyIfBranches(IfStmtPtr ifStmt, const std::vector<SymbolicScalar>& condPath)
{
    auto cond = SymbolicScalar::FromExpr(ifStmt->condition_);

    std::vector<SymbolicScalar> thenConds = condPath;
    std::vector<SymbolicScalar> elseConds = condPath;
    if (cond.IsValid()) {
        thenConds.push_back(cond);
        elseConds.push_back(!cond);
    }

    bool thenDead = SymbolicScalar::Check(thenConds) == SatStatus::kUnsat;
    bool elseDead = SymbolicScalar::Check(elseConds) == SatStatus::kUnsat;
    return BranchClassification{thenDead, elseDead, std::move(thenConds), std::move(elseConds)};
}

StmtPtr RewriteTerminatorValues(StmtPtr stmt, const VarExprMap& varMap)
{
    std::vector<ExprPtr> newValues;
    auto remapped = [&](const std::vector<ExprPtr>& values) {
        newValues.clear();
        for (auto& v : values) {
            newValues.push_back(utils::LookupVarInExpr(v, varMap));
        }
    };
    if (auto yieldStmt = As<YieldStmt>(stmt)) {
        remapped(yieldStmt->value_);
        return std::make_shared<YieldStmt>(newValues, yieldStmt->span_);
    }
    if (auto contStmt = As<ContinueStmt>(stmt)) {
        remapped(contStmt->value_);
        return std::make_shared<ContinueStmt>(newValues, contStmt->span_);
    }
    if (auto breakStmt = As<BreakStmt>(stmt)) {
        remapped(breakStmt->value_);
        return std::make_shared<BreakStmt>(newValues, breakStmt->span_);
    }
    if (auto retStmt = As<ReturnStmt>(stmt)) {
        remapped(retStmt->value_);
        return std::make_shared<ReturnStmt>(newValues, retStmt->span_);
    }
    return stmt;
}

class MergeStmtImpl {
public:
    explicit MergeStmtImpl(const std::vector<std::string>& extVarNames) : extVarNames_(extVarNames) {}

    SeqStmtsPtr Process(SeqStmtsPtr seq) const
    {
        std::vector<SymbolicScalar> condPath;
        return Process(seq, condPath, {});
    }

private:
    using VarPtrSet = std::unordered_set<const Var*>;
    bool IsExternalVar(const std::string& name) const
    {
        return std::find(extVarNames_.begin(), extVarNames_.end(), name) != extVarNames_.end();
    }

    std::pair<std::vector<VarPtr>, VarExprMap> CloneOrKeepReturnVars(const std::vector<VarPtr>& vars) const
    {
        std::vector<VarPtr> result;
        VarExprMap cloneMap;
        for (auto& var : vars) {
            if (IsExternalVar(var->name_)) {
                result.push_back(var);
            } else {
                auto cloned = var->Clone();
                cloneMap[var] = cloned;
                result.push_back(cloned);
            }
        }
        return {result, cloneMap};
    }

    std::vector<StmtPtr> MergeIfStmts(IfStmtPtr ifStmt, const std::vector<SymbolicScalar>& condPath, VarExprMap& subst,
                                      const VarPtrSet& liveOut) const
    {
        auto cls = ClassifyIfBranches(ifStmt, condPath);
        if (cls.thenDead) {
            if (ifStmt->elseBody_) {
                return SpliceSurvivor(ifStmt, ifStmt->elseBody_.value(), cls.elseConds, subst, liveOut);
            }
            return {};
        }
        if (cls.elseDead) {
            return SpliceSurvivor(ifStmt, ifStmt->thenBody_, cls.thenConds, subst, liveOut);
        }

        // Neither proven dead: merge both branches, clone shared defs in else, rebuild.
        // Vars that stay live after this if (read by later consumers, loop carries or the
        // enclosing scope) must keep the same identity in both branches: the values exit the
        // merged region through the in-place tensor itself, not through return vars, so cloning
        // them in the else branch would orphan the write (e.g. an ASSEMBLE into a buffer that
        // is read right after the if).
        VarPtrSet branchLiveOut = liveOut;
        for (const auto& rv : ifStmt->returnVars_) {
            if (rv) {
                branchLiveOut.insert(rv.get());
            }
        }
        auto thenBody = Process(ifStmt->thenBody_, cls.thenConds, branchLiveOut);
        auto elseBody = Process(ifStmt->elseBody_.value(), cls.elseConds, branchLiveOut);
        auto newElseBody = ResolveDuplicateVars(thenBody, elseBody, liveOut);
        return {
            std::make_shared<IfStmt>(ifStmt->condition_, thenBody, newElseBody, ifStmt->returnVars_, ifStmt->span_)};
    }

    std::optional<SeqStmtsPtr> ResolveDuplicateVars(SeqStmtsPtr thenBody, std::optional<SeqStmtsPtr> optElse,
                                                    const VarPtrSet& liveOut) const
    {
        auto& elseBody = *optElse;
        auto thenDefs = utils::CollectDefinedVars(thenBody);
        auto elseDefs = utils::CollectDefinedVars(elseBody);

        VarExprMap cloneMap;
        std::vector<VarPtr> varList;
        for (auto& v : thenDefs) {
            if (elseDefs.count(v) && !IsExternalVar(v->name_) && !liveOut.count(v.get())) {
                varList.push_back(v);
            }
        }

        // use varList to ensure clone order and naming is deterministic
        std::sort(varList.begin(), varList.end(), [](auto& a, auto& b) { return a->name_ < b->name_; });
        for (auto& v : varList) {
            if (auto lt = std::dynamic_pointer_cast<const LogicalTensor>(v)) {
                cloneMap[v] = lt->Clone(true); // shared raw tensor, distinct LogicalTensor identity
            } else {
                cloneMap[v] = v->Clone();
            }
        }

        std::vector<StmtPtr> newElseStmts;
        for (auto& s : elseBody->stmts_) {
            newElseStmts.push_back(SubstituteVars(s, cloneMap));
        }
        return std::make_shared<SeqStmts>(newElseStmts, elseBody->span_);
    }

    std::vector<StmtPtr> SpliceSurvivor(IfStmtPtr ifStmt, SeqStmtsPtr body, const std::vector<SymbolicScalar>& conds,
                                        VarExprMap& yieldMap, const VarPtrSet& liveOut = {}) const
    {
        auto survivor = Process(body, conds, liveOut);
        if (!survivor) {
            return {};
        }
        BuildYieldVarMap(ifStmt, survivor, yieldMap);
        return RemoveLastYieldStmt(survivor->stmts_);
    }

    // Sink `stmts` (which followed `ifStmt` in source order) into the tail of both branches; their
    // outputs become new if return vars (cloned for non-external vars, recorded in `cloneMap`).
    IfStmtPtr AppendIntoIfStmt(IfStmtPtr ifStmt, const std::vector<StmtPtr>& stmts, VarExprMap& cloneMap) const
    {
        auto newVars = CollectOutputVars(stmts);
        auto [addVars, newClones] = CloneOrKeepReturnVars(newVars);
        for (auto& [k, v] : newClones) {
            cloneMap[k] = v;
        }

        auto newThen = BuildAppendedBranch(ifStmt, ifStmt->thenBody_, stmts, newVars, ifStmt->span_);
        auto newElse = BuildAppendedBranch(ifStmt, ifStmt->elseBody_.value_or(nullptr), stmts, newVars, ifStmt->span_);

        auto retVars = ifStmt->returnVars_;
        retVars.insert(retVars.end(), addVars.begin(), addVars.end());
        return std::make_shared<IfStmt>(ifStmt->condition_, newThen, newElse, retVars, ifStmt->span_);
    }

    // Hoist `stmts` (which preceded `ifStmt`) into the head of both branches, ahead of each branch's
    // own body. New outputs become prepended return vars (clones recorded in `cloneMap`).
    IfStmtPtr PrependIntoIfStmt(IfStmtPtr ifStmt, const std::vector<StmtPtr>& stmts, VarExprMap& cloneMap) const
    {
        auto newVars = CollectOutputVars(stmts);
        auto existing = std::unordered_set<VarPtr>(ifStmt->returnVars_.begin(), ifStmt->returnVars_.end());
        std::vector<VarPtr> toAdd;
        for (auto& var : newVars) {
            if (!existing.count(var)) {
                toAdd.push_back(var);
            }
        }
        auto [prependVars, newClones] = CloneOrKeepReturnVars(toAdd);
        for (auto& [k, v] : newClones) {
            cloneMap[k] = v;
        }

        auto newThen = BuildPrependedBranch(stmts, ifStmt->thenBody_, newVars, ifStmt->span_);
        auto newElse = BuildPrependedBranch(stmts, ifStmt->elseBody_, newVars, ifStmt->span_);

        auto retVars = prependVars;
        retVars.insert(retVars.end(), ifStmt->returnVars_.begin(), ifStmt->returnVars_.end());
        return std::make_shared<IfStmt>(ifStmt->condition_, newThen, newElse, retVars, ifStmt->span_);
    }

    std::vector<StmtPtr> FoldLeadingStmts(std::vector<StmtPtr> stmts, VarExprMap& cloneMap) const
    {
        if (stmts.empty() || !IsA<IfStmt>(stmts.back())) {
            return stmts;
        }
        auto ifStmt = As<IfStmt>(stmts.back());
        size_t ifIdx = stmts.size() - 1;

        // A ForStmt is a barrier: only the trailing run of non-For stmts immediately before the if
        // is hoisted into it. Find where that run starts (just past the last For in the prefix);
        // everything before it stays. Collapse the run + old if in place into the rebuilt if.
        size_t hoist = ifIdx;
        while (hoist > 0 && !IsA<ForStmt>(stmts[hoist - 1])) {
            --hoist;
        }
        if (hoist < ifIdx) {
            std::vector<StmtPtr> toHoist(stmts.begin() + hoist, stmts.begin() + ifIdx);
            auto newIf = PrependIntoIfStmt(ifStmt, std::move(toHoist), cloneMap);
            stmts.resize(hoist);
            stmts.push_back(std::move(newIf));
        }
        return stmts;
    }

    // Right-to-left fold over one barrier-free segment. A live IfStmt absorbs the stmts to its right
    // into both branches (AppendIntoIfStmt); a SAT-proven-dead branch is spliced out (SpliceSurvivor).
    // Results fold into the running `merged`/`cloneMap`/`subst` accumulators.
    void MergeSegment(const std::vector<StmtPtr>& segment, const std::vector<SymbolicScalar>& condPath,
                      std::vector<StmtPtr>& merged, VarExprMap& cloneMap, VarExprMap& subst) const
    {
        std::vector<StmtPtr> collected;

        auto merge_survivor = [&](auto& ifstmt, auto& survivor, auto& conds) {
            VarExprMap local;
            auto ops = SpliceSurvivor(ifstmt, survivor, conds, local);
            // `local` maps this dead-if's return vars to THIS survivor's outputs. Apply it to the
            // already-collected consumers (they execute after the splice, so the outputs dominate)
            if (!local.empty()) {
                for (auto& c : collected) {
                    c = SubstituteVars(c, local);
                }
                for (auto& kv : local) {
                    subst[kv.first] = kv.second;
                }
            }
            // survivor (reverse source order, avoiding O(n^2) front-insertion).
            for (auto rit = ops.rbegin(); rit != ops.rend(); ++rit) {
                collected.push_back(*rit);
            }
        };

        for (size_t i = segment.size(); i > 0; --i) {
            auto& stmt = segment[i - 1];
            if (auto ifStmt = As<IfStmt>(stmt)) {
                auto cls = ClassifyIfBranches(ifStmt, condPath);
                if (cls.thenDead) {
                    if (ifStmt->elseBody_)
                        merge_survivor(ifStmt, *ifStmt->elseBody_, cls.elseConds);
                    continue;
                }
                if (cls.elseDead) {
                    merge_survivor(ifStmt, ifStmt->thenBody_, cls.thenConds);
                    continue;
                }
                std::reverse(collected.begin(), collected.end());
                auto newIf = AppendIntoIfStmt(ifStmt, collected, cloneMap);
                collected.clear();
                collected.push_back(newIf);
            } else {
                collected.push_back(stmt); // O(1) prepend under reverse storage
            }
        }

        std::reverse(collected.begin(), collected.end());
        auto stmts = FoldLeadingStmts(std::move(collected), cloneMap);
        merged.insert(merged.end(), stmts.begin(), stmts.end());
    }

    // Second pass over a merged list: descend into the compound stmts that segment-merge treated as
    // barriers. For/While bodies are re-Processed under strengthened conditions; each IfStmt is
    // finalized via MergeIfStmts, which may add splice substitutions to `subst` for later stmts.
    // `afterUses` carries the vars referenced after this sequence (in enclosing scopes); the
    // per-stmt live-out set is the suffix of the merged list plus it, and is used to keep vars
    // that escape a merged if from being cloned in the else branch.
    std::vector<StmtPtr> RebuildMergedStmts(const std::vector<StmtPtr>& merged,
                                            const std::vector<SymbolicScalar>& condPath,
                                            const VarPtrSet& afterUses) const
    {
        std::vector<VarPtrSet> suffixRefs(merged.size());
        {
            VarPtrSet acc = afterUses;
            for (size_t i = merged.size(); i-- > 0;) {
                suffixRefs[i] = acc;
                auto refs = utils::CollectStmtVarRefs(merged[i]);
                acc.insert(refs.begin(), refs.end());
            }
        }

        std::vector<StmtPtr> result;
        VarExprMap subst;
        for (size_t i = 0; i < merged.size(); ++i) {
            auto& stmt = merged[i];
            auto cur = SubstituteVars(stmt, subst); // no-op when subst is empty
            if (auto ifStmt = AsMut<IfStmt>(cur)) {
                auto stmts = MergeIfStmts(ifStmt, condPath, subst, suffixRefs[i]);
                result.insert(result.end(), stmts.begin(), stmts.end());
            } else if (auto forStmt = As<ForStmt>(cur)) {
                auto conds = condPath;
                auto loopConds = forStmt->GetAttr<std::vector<SymbolicScalar>>(kLoopCondsAttr);
                conds.insert(conds.end(), loopConds.begin(), loopConds.end());
                auto body = Process(forStmt->body_, conds, suffixRefs[i]);
                result.push_back(std::make_shared<ForStmt>(forStmt->loopVar_, forStmt->start_, forStmt->stop_,
                                                           forStmt->step_, forStmt->iterArgs_, body,
                                                           forStmt->returnVars_, forStmt->span_, forStmt->attrs_));
            } else if (auto whileStmt = As<WhileStmt>(cur)) {
                auto body = Process(whileStmt->body_, condPath, suffixRefs[i]);
                result.push_back(std::make_shared<WhileStmt>(whileStmt->condition_, whileStmt->iterArgs_, body,
                                                             whileStmt->returnVars_, whileStmt->span_));
            } else {
                result.push_back(cur);
            }
        }
        return result;
    }

    // Driver: split `seq` into barrier-free segments at Yield/Continue/For/While, segment-merge each,
    // rewrite the trailing terminator's cloned refs and apply survivor substitutions, then rebuild
    // compound stmts recursively (RebuildMergedStmts). For/While are barriers; their bodies merge
    // when RebuildMergedStmts recurses into them. `afterUses` are the vars referenced after this
    // sequence; they (and the sequence's own suffix refs) make up the live-out set that protects
    // escaping vars from being cloned into else branches.
    SeqStmtsPtr Process(SeqStmtsPtr seq, const std::vector<SymbolicScalar>& condPath, const VarPtrSet& afterUses) const
    {
        std::vector<StmtPtr> segment;
        std::vector<StmtPtr> merged;
        VarExprMap cloneMap;
        VarExprMap substMap;

        for (auto& stmt : seq->stmts_) {
            // yield/continue/return are block terminators. while/for is dynamic loop could not be merged also
            if (IsA<YieldStmt>(stmt) || IsA<ContinueStmt>(stmt) || IsA<ReturnStmt>(stmt) || IsA<ForStmt>(stmt) ||
                IsA<WhileStmt>(stmt)) {
                MergeSegment(segment, condPath, merged, cloneMap, substMap);
                merged.push_back(stmt);
                segment.clear();
            } else {
                segment.push_back(stmt);
            }
        }

        MergeSegment(segment, condPath, merged, cloneMap, substMap);

        if (!substMap.empty()) {
            for (auto& s : merged) {
                // substMap rewrites a SAT-dead IfStmt's return vars to its spliced survivor's outputs.
                // That rewrite is only valid for consumers that execute AFTER the survivor. Inside a
                // compound stmt's own body those return vars are input reads (dominated by defs before
                // the splice), so recursing into them would replace an in-scope def with the survivor's
                // own output -- a non-dominating self-reference. Apply only to non-compound siblings.
                if (IsA<IfStmt>(s) || IsA<ForStmt>(s) || IsA<WhileStmt>(s)) {
                    continue;
                }
                s = SubstituteVars(s, substMap);
            }
        }

        if (!cloneMap.empty() && !merged.empty()) {
            merged.back() = RewriteTerminatorValues(merged.back(), cloneMap);
        }
        auto finalResult = RebuildMergedStmts(merged, condPath, afterUses);
        return std::make_shared<SeqStmts>(finalResult, seq->span_);
    }

    const std::vector<std::string>& extVarNames_;
};
} // namespace

SeqStmtsPtr MergeStmtsIntoIfStmt(SeqStmtsPtr seq, const std::vector<std::string>& extVarNames)
{
    return MergeStmtImpl(extVarNames).Process(seq);
}

} // namespace pypto::ir
