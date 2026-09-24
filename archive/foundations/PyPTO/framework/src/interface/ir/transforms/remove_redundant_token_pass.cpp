/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir/transforms/remove_redundant_token_pass.h"

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ir/kind_traits.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/utils/stmt_utils.h"
#include "ir/type.h"

#include "interface/operation/attribute.h"
#include "interface/operation/operation.h"
#include "interface/tensor/ir_tensor_op_rebuild.h"
#include "interface/tensor/logical_tensor.h"

namespace pypto::ir {

namespace {

using npu::tile_fwk::AssembleOpAttribute;
using npu::tile_fwk::LogicalTensor;
using npu::tile_fwk::LogicalTensorPtr;
using npu::tile_fwk::Operation;
using npu::tile_fwk::RawTensor;
using npu::tile_fwk::SymbolicScalar;
using npu::tile_fwk::ViewOpAttribute;

LogicalTensorPtr AsLogicalTensor(const ExprPtr& expr)
{
    auto tensor = std::dynamic_pointer_cast<const LogicalTensor>(expr);
    return std::const_pointer_cast<LogicalTensor>(tensor);
}

class RemoveRedundantTokenPass : public IRMutator {
public:
    using IRMutator::VisitStmt_;

    SeqStmtsPtr Apply(const SeqStmtsPtr& seq) { return SeqStmts::AsMut(VisitStmt(seq)); }

private:
    struct AccessRegion {
        RawTensor* raw{nullptr};
        std::vector<SymbolicScalar> offset;
        std::vector<SymbolicScalar> shape;
    };

    struct Node {
        TensorOpStmtPtr op;
        std::vector<VarPtr> tokens;
        bool changed{false};
    };

    using ProducerMap = std::unordered_map<const Var*, size_t>;

    static std::optional<TokenKind> GetTokenKind(const VarPtr& token)
    {
        if (!token) {
            return std::nullopt;
        }
        auto type = As<TokenType>(token->GetType());
        if (!type) {
            return std::nullopt;
        }
        return type->kind_;
    }

    static std::vector<SymbolicScalar> GetOffset(const LogicalTensorPtr& tensor)
    {
        if (!tensor->GetDynOffset().empty()) {
            return tensor->GetDynOffset();
        }
        return SymbolicScalar::FromConcrete(tensor->GetOffset());
    }

    static std::vector<SymbolicScalar> GetShape(const LogicalTensorPtr& tensor)
    {
        if (!tensor->GetDynValidShape().empty()) {
            return tensor->GetDynValidShape();
        }
        return SymbolicScalar::FromConcrete(tensor->GetShape());
    }

    static std::optional<AccessRegion> GetTensorRegion(const LogicalTensorPtr& tensor)
    {
        if (!tensor || !tensor->GetRawTensor()) {
            return std::nullopt;
        }
        auto offset = GetOffset(tensor);
        auto shape = GetShape(tensor);
        if (offset.size() != shape.size()) {
            return std::nullopt;
        }
        return AccessRegion{tensor->GetRawTensor().get(), std::move(offset), std::move(shape)};
    }

    static std::vector<AccessRegion> GetReadRegions(const TensorOpStmtPtr& op)
    {
        std::vector<AccessRegion> regions;
        std::unordered_set<RawTensor*> seenRaws;
        auto operation = std::dynamic_pointer_cast<const Operation>(op);
        auto viewAttr = operation ? std::dynamic_pointer_cast<ViewOpAttribute>(operation->GetOpAttribute()) : nullptr;
        auto viewResult = !op->result_.empty() ? AsLogicalTensor(op->result_.front()) : nullptr;

        for (const auto& arg : op->args_) {
            auto tensor = AsLogicalTensor(arg);
            auto region = GetTensorRegion(tensor);
            if (!region) {
                continue;
            }
            if (!seenRaws.insert(region->raw).second) {
                continue;
            }
            if (viewAttr && viewResult) {
                region->offset = viewAttr->GetFromDynOffset().empty() ?
                                     SymbolicScalar::FromConcrete(viewAttr->GetFromOffset()) :
                                     viewAttr->GetFromDynOffset();
                region->shape = GetShape(viewResult);
            }
            if (region->offset.size() == region->shape.size()) {
                regions.push_back(std::move(*region));
            }
        }
        return regions;
    }

    static std::vector<AccessRegion> GetWriteRegions(const TensorOpStmtPtr& op)
    {
        std::vector<AccessRegion> regions;
        auto operation = std::dynamic_pointer_cast<const Operation>(op);
        auto assembleAttr = operation ? std::dynamic_pointer_cast<AssembleOpAttribute>(operation->GetOpAttribute()) :
                                        nullptr;
        auto source = !op->args_.empty() ? AsLogicalTensor(op->args_.front()) : nullptr;

        for (const auto& result : op->result_) {
            auto tensor = AsLogicalTensor(result);
            auto region = GetTensorRegion(tensor);
            if (!region) {
                continue;
            }
            if (assembleAttr && source) {
                region->offset = assembleAttr->GetToDynOffset().empty() ?
                                     SymbolicScalar::FromConcrete(assembleAttr->GetToOffset()) :
                                     assembleAttr->GetToDynOffset();
                region->shape = GetShape(source);
            }
            if (region->offset.size() == region->shape.size()) {
                regions.push_back(std::move(*region));
            }
        }
        return regions;
    }

    static bool ProveDisjoint(const AccessRegion& lhs, const AccessRegion& rhs)
    {
        if (!lhs.raw || lhs.raw != rhs.raw || lhs.offset.size() != rhs.offset.size() ||
            lhs.shape.size() != rhs.shape.size() || lhs.offset.size() != lhs.shape.size()) {
            return false;
        }
        for (size_t i = 0; i < lhs.shape.size(); ++i) {
            auto lhsBeforeRhs = (lhs.offset[i] + lhs.shape[i] <= rhs.offset[i]).Simplify();
            auto rhsBeforeLhs = (rhs.offset[i] + rhs.shape[i] <= lhs.offset[i]).Simplify();
            if ((lhsBeforeRhs.ConcreteValid() && lhsBeforeRhs.Concrete()) ||
                (rhsBeforeLhs.ConcreteValid() && rhsBeforeLhs.Concrete())) {
                return true;
            }
        }
        return false;
    }

    static std::vector<AccessRegion> GetProducedTokenRegions(const TensorOpStmtPtr& op, const VarPtr& token,
                                                             TokenKind kind)
    {
        std::vector<AccessRegion> regions;
        std::vector<VarPtr> kindTokens;
        for (const auto& resultToken : op->result_token_) {
            if (GetTokenKind(resultToken) == kind) {
                kindTokens.push_back(resultToken);
            }
        }

        auto accesses = kind == TokenKind::READ ? GetReadRegions(op) : GetWriteRegions(op);
        if (kindTokens.size() != accesses.size()) {
            return {};
        }
        for (size_t i = 0; i < kindTokens.size(); ++i) {
            if (kindTokens[i] == token) {
                regions.push_back(std::move(accesses[i]));
            }
        }
        return regions;
    }

    static bool AccessesAreDisjoint(const Node& producer, const Node& consumer, const VarPtr& token, TokenKind kind)
    {
        auto producerRegions = GetProducedTokenRegions(producer.op, token, kind);
        if (producerRegions.empty()) {
            return false;
        }

        auto consumerReads = GetReadRegions(consumer.op);
        auto consumerWrites = GetWriteRegions(consumer.op);
        for (const auto& producerRegion : producerRegions) {
            bool foundRelatedAccess = false;
            if (kind == TokenKind::WRITE) {
                for (const auto& region : consumerReads) {
                    if (region.raw == producerRegion.raw) {
                        foundRelatedAccess = true;
                        if (!ProveDisjoint(producerRegion, region)) {
                            return false;
                        }
                    }
                }
            }
            for (const auto& region : consumerWrites) {
                if (region.raw == producerRegion.raw) {
                    foundRelatedAccess = true;
                    if (!ProveDisjoint(producerRegion, region)) {
                        return false;
                    }
                }
            }
            if (!foundRelatedAccess) {
                return false;
            }
        }
        return true;
    }

    static ProducerMap BuildProducerMap(const std::vector<Node>& nodes)
    {
        ProducerMap producers;
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (const auto& result : nodes[i].op->result_) {
                producers.emplace(result.get(), i);
            }
            for (const auto& token : nodes[i].op->result_token_) {
                producers.emplace(token.get(), i);
            }
        }
        return producers;
    }

    static bool DependsOn(size_t producer, const Node& consumer, const ProducerMap& producers,
                          const VarPtr& ignoredToken = nullptr)
    {
        for (const auto& arg : consumer.op->args_) {
            for (const auto* use : utils::CollectVarUses(arg)) {
                auto it = producers.find(use);
                if (it != producers.end() && it->second == producer) {
                    return true;
                }
            }
        }
        for (const auto& token : consumer.tokens) {
            if (token == ignoredToken) {
                continue;
            }
            auto it = producers.find(token.get());
            if (it != producers.end() && it->second == producer) {
                return true;
            }
        }
        return false;
    }

    static std::optional<size_t> FindTokenProducer(const ProducerMap& producers, const VarPtr& token, size_t consumer)
    {
        auto it = producers.find(token.get());
        if (it == producers.end() || it->second >= consumer) {
            return std::nullopt;
        }
        return it->second;
    }

    static bool IsReachable(const std::vector<Node>& nodes, const ProducerMap& producers, size_t from, size_t to,
                            const VarPtr& ignoredToken)
    {
        std::vector<bool> reachable(to + 1, false);
        reachable[from] = true;
        for (size_t current = from + 1; current <= to; ++current) {
            auto tokenToIgnore = current == to ? ignoredToken : nullptr;
            const auto& node = nodes[current];
            for (const auto& arg : node.op->args_) {
                for (const auto* use : utils::CollectVarUses(arg)) {
                    auto it = producers.find(use);
                    if (it != producers.end() && it->second < current && reachable[it->second]) {
                        reachable[current] = true;
                        break;
                    }
                }
                if (reachable[current]) {
                    break;
                }
            }
            if (reachable[current]) {
                continue;
            }
            for (const auto& token : node.tokens) {
                if (token == tokenToIgnore) {
                    continue;
                }
                auto it = producers.find(token.get());
                if (it != producers.end() && it->second < current && reachable[it->second]) {
                    reachable[current] = true;
                    break;
                }
            }
        }
        return reachable[to];
    }

    static bool HasExternalUse(const Node& node, const std::vector<StmtPtr>& statements)
    {
        std::unordered_set<const Var*> defs;
        for (const auto& result : node.op->result_) {
            defs.insert(result.get());
        }
        for (const auto& token : node.op->result_token_) {
            defs.insert(token.get());
        }

        for (const auto& stmt : statements) {
            if (As<TensorOpStmt>(stmt) || As<ReturnStmt>(stmt)) {
                continue;
            }
            auto uses = utils::CollectStmtVarRefs(stmt);
            for (const auto* def : defs) {
                if (uses.count(def)) {
                    return true;
                }
            }
        }
        return false;
    }

    bool TryRemoveOneToken(std::vector<Node>& nodes, const ProducerMap& producers,
                           const std::vector<StmtPtr>& statements, size_t consumerIndex)
    {
        auto& consumer = nodes[consumerIndex];
        for (size_t tokenIndex = 0; tokenIndex < consumer.tokens.size(); ++tokenIndex) {
            auto token = consumer.tokens[tokenIndex];
            auto kind = GetTokenKind(token);
            if (!kind || (*kind != TokenKind::READ && *kind != TokenKind::WRITE)) {
                continue;
            }
            auto producer = FindTokenProducer(producers, token, consumerIndex);
            if (!producer) {
                continue;
            }

            if (IsReachable(nodes, producers, *producer, consumerIndex, token)) {
                consumer.tokens.erase(consumer.tokens.begin() + static_cast<std::ptrdiff_t>(tokenIndex));
                consumer.changed = true;
                return true;
            }

            if (!AccessesAreDisjoint(nodes[*producer], consumer, token, *kind) ||
                HasExternalUse(consumer, statements)) {
                continue;
            }

            consumer.tokens.erase(consumer.tokens.begin() + static_cast<std::ptrdiff_t>(tokenIndex));
            for (const auto& predecessorToken : nodes[*producer].tokens) {
                if (std::find(consumer.tokens.begin(), consumer.tokens.end(), predecessorToken) ==
                    consumer.tokens.end()) {
                    consumer.tokens.push_back(predecessorToken);
                }
            }
            consumer.changed = true;

            for (size_t successor = consumerIndex + 1; successor < nodes.size(); ++successor) {
                if (!DependsOn(consumerIndex, nodes[successor], producers)) {
                    continue;
                }
                auto& successorNode = nodes[successor];
                auto& successorTokens = successorNode.tokens;
                if (std::find(successorTokens.begin(), successorTokens.end(), token) == successorTokens.end()) {
                    successorTokens.push_back(token);
                    successorNode.changed = true;
                }
            }
            return true;
        }
        return false;
    }

    bool OptimizeStatements(std::vector<StmtPtr>& statements)
    {
        std::vector<Node> nodes;
        for (const auto& stmt : statements) {
            if (auto op = As<TensorOpStmt>(stmt)) {
                nodes.push_back({op, op->tokens_, false});
            }
        }
        auto producers = BuildProducerMap(nodes);

        for (size_t consumerIndex = 0; consumerIndex < nodes.size(); ++consumerIndex) {
            while (TryRemoveOneToken(nodes, producers, statements, consumerIndex)) {
            }
        }

        bool changed = false;
        size_t nodeIndex = 0;
        for (auto& stmt : statements) {
            if (!As<TensorOpStmt>(stmt)) {
                continue;
            }
            auto& node = nodes[nodeIndex++];
            if (!node.changed) {
                continue;
            }
            stmt = npu::tile_fwk::RebuildTensorOpStmt(node.op, node.op->result_, node.op->result_token_, node.op->args_,
                                                      std::move(node.tokens), node.op->span_);
            changed = true;
        }
        return changed;
    }

    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override
    {
        auto seq = As<SeqStmts>(IRMutator::VisitStmt_(op));
        auto statements = seq->stmts_;
        if (!OptimizeStatements(statements)) {
            return seq;
        }
        return std::make_shared<SeqStmts>(std::move(statements), seq->span_);
    }
};

class TokenUseCollector : public IRVisitor {
public:
    using IRVisitor::VisitStmt_;

    std::unordered_set<const Var*> TakeUsedTokens() { return std::move(usedTokens_); }

private:
    void VisitVarLike_(const VarPtr& op) override
    {
        if (As<TokenType>(op->GetType())) {
            usedTokens_.insert(op.get());
        }
        IRVisitor::VisitVarLike_(op);
    }

    void VisitStmt_(const TensorOpStmtPtr& op) override
    {
        for (const auto& arg : op->args_) {
            VisitExpr(arg);
        }
        for (const auto& token : op->tokens_) {
            VisitExpr(token);
        }
    }

    std::unordered_set<const Var*> usedTokens_;
};

class UnusedResultTokenRemover : public IRMutator {
public:
    using IRMutator::VisitStmt_;

    explicit UnusedResultTokenRemover(std::unordered_set<const Var*> usedTokens) : usedTokens_(std::move(usedTokens)) {}

private:
    StmtPtr VisitStmt_(const TensorOpStmtPtr& op) override
    {
        std::vector<VarPtr> resultTokens;
        resultTokens.reserve(op->result_token_.size());
        for (const auto& token : op->result_token_) {
            if (usedTokens_.count(token.get())) {
                resultTokens.push_back(token);
                continue;
            }
            for (const auto& result : op->result_) {
                auto tensor = AsLogicalTensor(result);
                if (tensor && tensor->GetWriteToken() == token) {
                    tensor->SetWriteToken(nullptr);
                }
            }
            for (const auto& arg : op->args_) {
                auto tensor = AsLogicalTensor(arg);
                if (tensor && tensor->GetReadToken() == token) {
                    tensor->SetReadToken(nullptr);
                }
            }
        }
        if (resultTokens.size() == op->result_token_.size()) {
            return op;
        }
        return npu::tile_fwk::RebuildTensorOpStmt(op, op->result_, std::move(resultTokens), op->args_, op->tokens_,
                                                  op->span_);
    }

    std::unordered_set<const Var*> usedTokens_;
};

SeqStmtsPtr RemoveUnusedResultTokens(const SeqStmtsPtr& seq)
{
    TokenUseCollector collector;
    collector.VisitStmt(seq);
    UnusedResultTokenRemover remover(collector.TakeUsedTokens());
    return SeqStmts::AsMut(remover.VisitStmt(seq));
}

} // namespace

SeqStmtsPtr RunRemoveRedundantTokenPass(SeqStmtsPtr seq)
{
    RemoveRedundantTokenPass pass;
    return RemoveUnusedResultTokens(pass.Apply(seq));
}

} // namespace pypto::ir
