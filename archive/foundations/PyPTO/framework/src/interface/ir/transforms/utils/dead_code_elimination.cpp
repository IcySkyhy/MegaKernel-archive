/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "ir/transforms/utils/dead_code_elimination.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "interface/tensor/ir.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/stmt.h"
#include "ir/transforms/base/visitor.h"
#include "ir/type.h"
#include "stmt_utils.h"

namespace pypto {
namespace ir {
namespace dce {

using utils::CollectStmtVarRefs;
using utils::CollectVarUses;
using utils::FlattenBody;
using utils::MakeSeqBody;
using utils::StmtsEqual;

void CollectAllAssignStmts(const std::vector<StmtPtr>& stmts, std::vector<std::shared_ptr<const AssignStmt>>& assigns)
{
    for (const auto& stmt : stmts) {
        if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
            assigns.push_back(assign);
        }
        if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
            CollectAllAssignStmts(FlattenBody(for_stmt->body_), assigns);
        } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
            CollectAllAssignStmts(FlattenBody(if_stmt->thenBody_), assigns);
            if (if_stmt->elseBody_.has_value()) {
                CollectAllAssignStmts(FlattenBody(if_stmt->elseBody_.value()), assigns);
            }
        } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
            CollectAllAssignStmts(FlattenBody(while_stmt->body_), assigns);
        } else if (auto section_stmt = std::dynamic_pointer_cast<const SectionStmt>(stmt)) {
            if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(section_stmt->body_)) {
                CollectAllAssignStmts(seq->stmts_, assigns);
            } else if (section_stmt->body_) {
                std::vector<StmtPtr> single = {section_stmt->body_};
                CollectAllAssignStmts(single, assigns);
            }
        }
    }
}

namespace {

using RemovablePredicate = std::function<bool(const StmtPtr&)>;

/// Collect live-root variables.
/// A statement is a "live root" when it is NOT classified as a removal
/// candidate by `is_removable`. Its own Var references (expressions and
/// direct fields, not nested-body refs) are added to the live set; the
/// nested body, if any, is recursed into separately so its own candidate
/// assignments remain eligible for removal.
void FindLiveRootsRecursiveImpl(const std::vector<StmtPtr>& stmts, const RemovablePredicate& is_removable,
                                std::unordered_set<const Var*>& live)
{
    auto collect_expr_refs = [&](const ExprPtr& expr) {
        if (!expr)
            return;
        auto refs = CollectVarUses(expr);
        live.insert(refs.begin(), refs.end());
    };
    auto collect_iter_arg_refs = [&](const auto& loop_stmt) {
        for (const auto& iter_arg : loop_stmt->iterArgs_) {
            collect_expr_refs(iter_arg->initValue_);
        }
    };

    for (const auto& stmt : stmts) {
        // Live-root: non-candidate leaf statements contribute their refs.
        // ContinueStmt/BreakStmt are always non-removable — their carried vars
        // must be considered live even when no other statement uses them.
        bool is_leaf = std::dynamic_pointer_cast<const AssignStmt>(stmt) ||
                       std::dynamic_pointer_cast<const EvalStmt>(stmt) ||
                       std::dynamic_pointer_cast<const ReturnStmt>(stmt) ||
                       std::dynamic_pointer_cast<const YieldStmt>(stmt) ||
                       std::dynamic_pointer_cast<const TensorOpStmt>(stmt) ||
                       std::dynamic_pointer_cast<const ScalarOpStmt>(stmt) ||
                       std::dynamic_pointer_cast<const ContinueStmt>(stmt) ||
                       std::dynamic_pointer_cast<const BreakStmt>(stmt);
        if (is_leaf && !is_removable(stmt)) {
            auto all_refs = CollectStmtVarRefs(stmt);
            // Do not keep an unconsumed result token alive just because its TensorOp is non-removable.
            if (auto tensor_op = std::dynamic_pointer_cast<const TensorOpStmt>(stmt)) {
                for (const auto& token : tensor_op->result_token_) {
                    all_refs.erase(token.get());
                }
            }
            live.insert(all_refs.begin(), all_refs.end());
        }

        // Control-flow headers: add direct-field refs (bounds, conditions,
        // iter-arg initializers) but defer body traversal to the recursive
        // call so nested candidate assignments remain eligible for removal.
        if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
            collect_expr_refs(for_stmt->start_);
            collect_expr_refs(for_stmt->stop_);
            collect_expr_refs(for_stmt->step_);
            collect_iter_arg_refs(for_stmt);
            FindLiveRootsRecursiveImpl(FlattenBody(for_stmt->body_), is_removable, live);
        } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
            collect_expr_refs(if_stmt->condition_);
            FindLiveRootsRecursiveImpl(FlattenBody(if_stmt->thenBody_), is_removable, live);
            if (if_stmt->elseBody_.has_value()) {
                FindLiveRootsRecursiveImpl(FlattenBody(if_stmt->elseBody_.value()), is_removable, live);
            }
        } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
            collect_expr_refs(while_stmt->condition_);
            collect_iter_arg_refs(while_stmt);
            FindLiveRootsRecursiveImpl(FlattenBody(while_stmt->body_), is_removable, live);
        } else if (auto section_stmt = std::dynamic_pointer_cast<const SectionStmt>(stmt)) {
            if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(section_stmt->body_)) {
                FindLiveRootsRecursiveImpl(seq->stmts_, is_removable, live);
            } else if (section_stmt->body_) {
                std::vector<StmtPtr> single = {section_stmt->body_};
                FindLiveRootsRecursiveImpl(single, is_removable, live);
            }
        }
    }
}

std::vector<StmtPtr> FilterDeadCodeImpl(const std::vector<StmtPtr>& stmts, const RemovablePredicate& is_removable,
                                        const std::unordered_set<const Var*>& live)
{
    std::vector<StmtPtr> result;

    // Check if a body is a single break or continue statement (trivial loop body).
    auto is_trivial_loop_body = [](const std::vector<StmtPtr>& body) -> bool {
        return body.size() == 1 && (ir::As<BreakStmt>(body[0]) || ir::As<ContinueStmt>(body[0]));
    };

    // Check if all statements in a list are YieldStmt.
    auto all_yield = [](const std::vector<StmtPtr>& body) -> bool {
        return body.empty() || (body.size() == 1 && ir::As<YieldStmt>(body[0]));
    };

    for (const auto& stmt : stmts) {
        if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
            if (is_removable(stmt) && !live.count(assign->var_.get()))
                continue;
            result.push_back(stmt);
        } else if (auto tensor_op = std::dynamic_pointer_cast<const TensorOpStmt>(stmt)) {
            if (is_removable(stmt)) {
                bool any_live = false;
                for (const auto& r : tensor_op->result_) {
                    if (live.count(r.get())) {
                        any_live = true;
                        break;
                    }
                }
                for (const auto& token : tensor_op->result_token_) {
                    if (live.count(token.get())) {
                        any_live = true;
                        break;
                    }
                }
                if (!any_live)
                    continue;
            }
            result.push_back(stmt);
        } else if (auto scalar_op = std::dynamic_pointer_cast<const ScalarOpStmt>(stmt)) {
            if (is_removable(stmt) && !live.count(scalar_op->result_.get()))
                continue;
            result.push_back(stmt);
        } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
            auto original = FlattenBody(for_stmt->body_);
            auto filtered = FilterDeadCodeImpl(original, is_removable, live);
            // Drop no-op loops: body is just break/continue, no return vars.
            if (for_stmt->returnVars_.empty() && is_trivial_loop_body(filtered)) {
                continue;
            }
            if (StmtsEqual(filtered, original)) {
                result.push_back(stmt);
                continue;
            }
            auto new_body = MakeSeqBody(filtered, for_stmt->span_);
            auto new_for = std::make_shared<const ForStmt>(for_stmt->loopVar_, for_stmt->start_, for_stmt->stop_,
                                                           for_stmt->step_, for_stmt->iterArgs_, new_body,
                                                           for_stmt->returnVars_, for_stmt->span_, for_stmt->attrs_);
            result.push_back(new_for);
        } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
            auto then_orig = FlattenBody(if_stmt->thenBody_);
            auto filtered_then = FilterDeadCodeImpl(then_orig, is_removable, live);
            bool then_unchanged = StmtsEqual(filtered_then, then_orig);
            bool else_unchanged = true;
            std::optional<SeqStmtsPtr> filtered_else;
            if (if_stmt->elseBody_.has_value()) {
                auto else_orig = FlattenBody(if_stmt->elseBody_.value());
                auto fe = FilterDeadCodeImpl(else_orig, is_removable, live);
                else_unchanged = StmtsEqual(fe, else_orig);
                filtered_else = MakeSeqBody(fe, if_stmt->span_);
            }
            // Drop no-op if-stmts: branches only yield, no return vars.
            bool then_all_yield = all_yield(filtered_then);
            bool else_all_yield = !filtered_else.has_value() || all_yield(FlattenBody(*filtered_else));
            if (if_stmt->returnVars_.empty() && then_all_yield && else_all_yield) {
                continue;
            }
            if (then_unchanged && else_unchanged) {
                result.push_back(stmt);
                continue;
            }
            std::optional<StmtPtr> else_body_stmt;
            if (filtered_else) {
                else_body_stmt = *filtered_else;
            }
            auto new_if = std::make_shared<const IfStmt>(
                if_stmt->condition_,
                filtered_then.empty() ? std::make_shared<const SeqStmts>(std::vector<StmtPtr>{}, if_stmt->span_) :
                                        MakeSeqBody(filtered_then, if_stmt->span_),
                else_body_stmt, if_stmt->returnVars_, if_stmt->span_);
            result.push_back(new_if);
        } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
            auto original = FlattenBody(while_stmt->body_);
            auto filtered = FilterDeadCodeImpl(original, is_removable, live);
            // Drop no-op loops: body is just break/continue, no return vars.
            if (while_stmt->returnVars_.empty() && is_trivial_loop_body(filtered)) {
                continue;
            }
            if (StmtsEqual(filtered, original)) {
                result.push_back(stmt);
                continue;
            }
            auto new_body = MakeSeqBody(filtered, while_stmt->span_);
            auto new_while = std::make_shared<const WhileStmt>(while_stmt->condition_, while_stmt->iterArgs_, new_body,
                                                               while_stmt->returnVars_, while_stmt->span_);
            result.push_back(new_while);
        } else if (auto section_stmt = std::dynamic_pointer_cast<const SectionStmt>(stmt)) {
            std::vector<StmtPtr> original;
            if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(section_stmt->body_)) {
                original = seq->stmts_;
            } else if (section_stmt->body_) {
                original = {section_stmt->body_};
            }
            auto filtered = FilterDeadCodeImpl(original, is_removable, live);
            if (StmtsEqual(filtered, original)) {
                result.push_back(stmt);
                continue;
            }
            auto new_body = filtered.empty() ?
                                std::make_shared<const SeqStmts>(std::vector<StmtPtr>{}, section_stmt->span_) :
                                MakeSeqBody(filtered, section_stmt->span_);
            auto new_section = std::make_shared<const SectionStmt>(section_stmt->sectionKind_, new_body,
                                                                   section_stmt->span_);
            result.push_back(new_section);
        } else {
            result.push_back(stmt);
        }
    }

    return result;
}

/// Predicate for the default `EliminateDeadCode`: any AssignStmt that is not
/// a known side-effect op is a removal candidate.
bool IsRemovableForDefaultDce(const StmtPtr& stmt)
{
    return std::dynamic_pointer_cast<const AssignStmt>(stmt) != nullptr;
}

/// Walk an expression tree and report whether any Call appears.
class CallFinder : public IRVisitor {
public:
    bool found = false;

private:
    using IRVisitor::VisitExpr_;
    void VisitExpr_(const CallPtr& op) override
    {
        found = true;
        IRVisitor::VisitExpr_(op);
    }
};

bool ExprContainsCall(const ExprPtr& expr)
{
    if (!expr)
        return false;
    CallFinder finder;
    finder.VisitExpr(expr);
    return finder.found;
}

/// Predicate for `EliminateDeadScalarAssignments`: an AssignStmt with a
/// scalar-typed LHS whose RHS contains no `Call` anywhere.
bool IsRemovableScalarAssign(const StmtPtr& stmt)
{
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    if (!assign)
        return false;
    if (!As<ScalarType>(assign->var_->GetType()))
        return false;
    if (ExprContainsCall(assign->value_))
        return false;
    return true;
}

} // namespace

std::vector<StmtPtr> EliminateDeadCode(const std::vector<StmtPtr>& stmts, const RemovablePredicate& is_removable)
{
    std::unordered_set<const Var*> live;
    FindLiveRootsRecursiveImpl(stmts, is_removable, live);

    // Collect all definition sites (AssignStmt, TensorOpStmt, ScalarOpStmt)
    // and cache their def/use pairs for the fixed-point propagation.
    struct MemoryDef {
        const Var* var;
        int memory_id;
    };
    struct DefSite {
        std::vector<const Var*> defs;
        std::vector<MemoryDef> memory_defs;
        std::unordered_set<const Var*> uses;
    };

    std::vector<DefSite> def_sites;

    std::function<void(const std::vector<StmtPtr>&)> collect_def_sites = [&](const std::vector<StmtPtr>& stmts_vec) {
        for (const auto& s : stmts_vec) {
            if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(s)) {
                def_sites.push_back({{assign->var_.get()}, {}, CollectVarUses(assign->value_)});
            } else if (auto tensor_op = std::dynamic_pointer_cast<const TensorOpStmt>(s)) {
                std::vector<const Var*> defs;
                std::vector<MemoryDef> memory_defs;
                for (const auto& r : tensor_op->result_) {
                    defs.push_back(r.get());
                    if (auto mid = GetVarMemoryId(r.get())) {
                        memory_defs.push_back({r.get(), *mid});
                    }
                }
                for (const auto& token : tensor_op->result_token_) {
                    defs.push_back(token.get());
                }
                std::unordered_set<const Var*> uses;
                for (const auto& arg : tensor_op->args_) {
                    auto arg_uses = CollectVarUses(arg);
                    uses.insert(arg_uses.begin(), arg_uses.end());
                }
                for (const auto& r : tensor_op->tokens_) {
                    uses.insert(r.get());
                }
                def_sites.push_back({std::move(defs), std::move(memory_defs), std::move(uses)});
            } else if (auto scalar_op = std::dynamic_pointer_cast<const ScalarOpStmt>(s)) {
                std::unordered_set<const Var*> uses;
                for (const auto& arg : scalar_op->args_) {
                    auto arg_uses = CollectVarUses(arg);
                    uses.insert(arg_uses.begin(), arg_uses.end());
                }
                def_sites.push_back({{scalar_op->result_.get()}, {}, std::move(uses)});
            }
            // Recurse into control-flow bodies
            if (auto for_s = std::dynamic_pointer_cast<const ForStmt>(s)) {
                collect_def_sites(FlattenBody(for_s->body_));
            } else if (auto if_s = std::dynamic_pointer_cast<const IfStmt>(s)) {
                collect_def_sites(FlattenBody(if_s->thenBody_));
                if (if_s->elseBody_.has_value()) {
                    collect_def_sites(FlattenBody(if_s->elseBody_.value()));
                }
            } else if (auto while_s = std::dynamic_pointer_cast<const WhileStmt>(s)) {
                collect_def_sites(FlattenBody(while_s->body_));
            } else if (auto sec_s = std::dynamic_pointer_cast<const SectionStmt>(s)) {
                if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(sec_s->body_)) {
                    collect_def_sites(seq->stmts_);
                } else if (sec_s->body_) {
                    collect_def_sites({sec_s->body_});
                }
            }
        }
    };

    collect_def_sites(stmts);

    // Fixed-point: propagate liveness from defs to their uses.
    bool changed = true;
    while (changed) {
        changed = false;
        auto live_memory_ids = CollectMemoryIds(live);
        for (size_t i = def_sites.size(); i-- > 0;) {
            bool any_def_live = false;
            for (const auto* def : def_sites[i].defs) {
                if (live.count(def)) {
                    any_def_live = true;
                    break;
                }
            }
            if (!any_def_live) {
                for (const auto& memory_def : def_sites[i].memory_defs) {
                    if (live_memory_ids.count(memory_def.memory_id)) {
                        if (live.insert(memory_def.var).second)
                            changed = true;
                        any_def_live = true;
                    }
                }
            }
            if (!any_def_live)
                continue;
            for (const Var* ref : def_sites[i].uses) {
                if (live.insert(ref).second)
                    changed = true;
            }
        }
    }

    return FilterDeadCodeImpl(stmts, is_removable, live);
}

std::vector<StmtPtr> EliminateDeadCode(const std::vector<StmtPtr>& stmts)
{
    return EliminateDeadCode(stmts, IsRemovableForDefaultDce);
}

std::vector<StmtPtr> EliminateDeadScalarAssignments(const std::vector<StmtPtr>& stmts)
{
    return EliminateDeadCode(stmts, IsRemovableScalarAssign);
}

} // namespace dce
} // namespace ir
} // namespace pypto
