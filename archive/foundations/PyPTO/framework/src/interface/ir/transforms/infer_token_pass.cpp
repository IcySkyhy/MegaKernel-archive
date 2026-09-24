/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir/transforms/infer_token_pass.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/base/visitor.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "ir/type.h"

#include "interface/operation/operation.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"

namespace pypto::ir {

using npu::tile_fwk::IRContext;
using npu::tile_fwk::Operation;
using npu::tile_fwk::RawTensor;

class InferTokenPass : public IRMutator {
public:
    using IRMutator::VisitStmt_;

    SeqStmtsPtr Apply(const SeqStmtsPtr& seq)
    {
        BuildStorageClasses(seq);
        RegisterRaws(seq);
        AnalyzeLiveRaws(seq);
        return SeqStmts::AsMut(VisitStmt(seq));
    }

private:
    struct RawTokenState {
        VarPtr latestWrite;
        VarPtr latestRead;
        uint64_t readRevision{0};
    };

    struct TensorAccess {
        RawTensor* raw;
        VarPtr token;
    };

    class StorageClasses {
    public:
        RawTensor* Find(RawTensor* raw)
        {
            auto it = parent_.find(raw);
            if (it == parent_.end()) {
                return raw;
            }
            RawTensor* root = Find(it->second);
            parent_[raw] = root;
            return root;
        }

        void Union(RawTensor* keep, RawTensor* merge)
        {
            keep = Find(keep);
            merge = Find(merge);
            if (keep != merge) {
                parent_[merge] = keep;
            }
        }

    private:
        std::unordered_map<RawTensor*, RawTensor*> parent_;
    };

    using RawTokenStateMap = std::unordered_map<RawTensor*, RawTokenState>;
    using RawNameMap = std::unordered_map<RawTensor*, std::string>;
    using RawSet = std::unordered_set<RawTensor*>;

    static npu::tile_fwk::LogicalTensorPtr AsLogicalTensor(const ExprPtr& expr)
    {
        auto tensor = std::dynamic_pointer_cast<const npu::tile_fwk::LogicalTensor>(expr);
        return std::const_pointer_cast<npu::tile_fwk::LogicalTensor>(tensor);
    }

    static RawTensor* RawOf(const ExprPtr& expr)
    {
        auto tensor = AsLogicalTensor(expr);
        return tensor ? tensor->GetRawTensor().get() : nullptr;
    }

    class StorageAliasCollector : public IRVisitor {
    public:
        explicit StorageAliasCollector(StorageClasses& classes) : classes_(classes) {}

    private:
        using IRVisitor::VisitStmt_;

        void VisitStmt_(const TensorOpStmtPtr& op) override
        {
            auto operation = std::dynamic_pointer_cast<const Operation>(op);
            if (operation) {
                for (size_t i = 0; i < op->result_.size(); ++i) {
                    auto argIndex = operation->GetInplaceIndex(i);
                    if (argIndex < 0 || static_cast<size_t>(argIndex) >= op->args_.size()) {
                        continue;
                    }
                    auto* argRaw = RawOf(op->args_[static_cast<size_t>(argIndex)]);
                    auto* resultRaw = RawOf(op->result_[i]);
                    if (argRaw != nullptr && resultRaw != nullptr) {
                        classes_.Union(argRaw, resultRaw);
                    }
                }
            }
            IRVisitor::VisitStmt_(op);
        }

        StorageClasses& classes_;
    };

    void BuildStorageClasses(const StmtPtr& stmt)
    {
        StorageAliasCollector collector(storage_);
        collector.VisitStmt(stmt);
    }

    class RawTensorCollector : public IRVisitor {
    public:
        RawNameMap TakeRawMap() { return std::move(rawMap_); }

    private:
        using IRVisitor::VisitExpr_;

        void VisitExpr_(const VarPtr& op) override
        {
            auto tensor = AsLogicalTensor(op);
            if (tensor && tensor->GetRawTensor()) {
                rawMap_.emplace(tensor->GetRawTensor().get(), tensor->name_);
            }
            IRVisitor::VisitExpr_(op);
        }

        RawNameMap rawMap_;
    };

    void RegisterRaws(const StmtPtr& stmt)
    {
        RawTensorCollector collector;
        collector.VisitStmt(stmt);
        registeredRaws_.clear();
        for (const auto& [raw, name] : collector.TakeRawMap()) {
            auto* root = storage_.Find(raw);
            if (raw == root) {
                registeredRaws_[root] = name;
            } else {
                registeredRaws_.emplace(root, name);
            }
        }
    }

    void AddTensorRaw(const ExprPtr& expr)
    {
        auto tensor = AsLogicalTensor(expr);
        if (tensor && tensor->GetRawTensor()) {
            liveRaws_.insert(storage_.Find(tensor->GetRawTensor().get()));
        }
    }

    void AnalyzeLiveRaws(const StmtPtr& stmt)
    {
        if (auto seq = As<SeqStmts>(stmt)) {
            for (auto it = seq->stmts_.rbegin(); it != seq->stmts_.rend(); ++it) {
                AnalyzeLiveRaws(*it);
            }
        } else if (auto tensorOp = As<TensorOpStmt>(stmt)) {
            for (const auto& arg : tensorOp->args_) {
                AddTensorRaw(arg);
            }
            for (const auto& result : tensorOp->result_) {
                AddTensorRaw(result);
            }
        } else if (auto ifStmt = As<IfStmt>(stmt)) {
            auto initialLiveRaws = liveRaws_;
            controlLiveOutRaws_[ifStmt.get()] = initialLiveRaws;

            AnalyzeLiveRaws(ifStmt->thenBody_);
            auto thenLiveRaws = liveRaws_;

            liveRaws_ = initialLiveRaws;
            if (ifStmt->elseBody_) {
                AnalyzeLiveRaws(*ifStmt->elseBody_);
            }
            liveRaws_.insert(thenLiveRaws.begin(), thenLiveRaws.end());
        } else if (auto forStmt = As<ForStmt>(stmt)) {
            auto initialLiveRaws = liveRaws_;
            controlLiveOutRaws_[forStmt.get()] = initialLiveRaws;
            AnalyzeLiveRaws(forStmt->body_);
        } else if (auto section = As<SectionStmt>(stmt)) {
            AnalyzeLiveRaws(section->body_);
        }
    }

    VarPtr CreateTokenVar(const npu::tile_fwk::LogicalTensorPtr& tensor, Span span, TokenKind kind)
    {
        const char* suffix = kind == TokenKind::READ ? "_r" : "_w";
        std::string name = tensor->name_ + suffix;
        return IRContext::Get().MakeSemanticToken(name, kind, span);
    }

    VarPtr CreateControlToken(const std::string& base, const char* control, Span span, TokenKind kind)
    {
        const char* suffix = kind == TokenKind::READ ? "_r" : "_w";
        std::string name = base + suffix;
        if (control[0] != '\0') {
            name += "_" + std::string(control);
        }
        return IRContext::Get().MakeSemanticToken(name, kind, span);
    }

    static void AppendToken(const VarPtr& token, std::vector<VarPtr>& tokens, std::unordered_set<VarPtr>& seen)
    {
        if (token && seen.insert(token).second) {
            tokens.push_back(token);
        }
    }

    static RawTokenState GetRawState(const RawTokenStateMap& states, RawTensor* raw)
    {
        auto it = states.find(raw);
        return it == states.end() ? RawTokenState{} : it->second;
    }

    static std::vector<ExprPtr> GetTerminatorValues(const SeqStmtsPtr& body)
    {
        if (!body || body->stmts_.empty()) {
            return {};
        }
        const auto& terminator = body->stmts_.back();
        if (auto yield = As<YieldStmt>(terminator)) {
            return yield->value_;
        }
        if (auto cont = As<ContinueStmt>(terminator)) {
            return cont->value_;
        }
        return {};
    }

    static SeqStmtsPtr AppendTerminatorValues(const SeqStmtsPtr& body, const std::vector<ExprPtr>& values)
    {
        if (values.empty()) {
            return body;
        }

        auto statements = body->stmts_;
        auto newValues = GetTerminatorValues(body);
        newValues.insert(newValues.end(), values.begin(), values.end());
        const auto& terminator = statements.back();
        if (As<YieldStmt>(terminator)) {
            statements.back() = std::make_shared<YieldStmt>(std::move(newValues), terminator->span_);
        } else if (As<ContinueStmt>(terminator)) {
            statements.back() = std::make_shared<ContinueStmt>(std::move(newValues), terminator->span_);
        }
        return std::make_shared<SeqStmts>(std::move(statements), body->span_);
    }

    VarPtr MergeIfToken(const VarPtr& thenToken, const VarPtr& elseToken, const std::string& base, TokenKind kind,
                        Span span, std::vector<ExprPtr>& thenValues, std::vector<ExprPtr>& elseValues,
                        std::vector<VarPtr>& returnVars, bool forceMerge = false)
    {
        if (!thenToken && !elseToken) {
            return nullptr;
        }
        if (!forceMerge && thenToken == elseToken) {
            return thenToken;
        }
        auto token = CreateControlToken(base, "if", span, kind);
        thenValues.push_back(thenToken ? thenToken : NoneValue());
        elseValues.push_back(elseToken ? elseToken : NoneValue());
        returnVars.push_back(token);
        return token;
    }

    VarPtr AddForToken(const VarPtr& initToken, const VarPtr& bodyToken, const std::string& base, TokenKind kind,
                       Span span, std::vector<IterArgPtr>& iterArgs, std::vector<ExprPtr>& bodyValues,
                       std::vector<VarPtr>& returnVars, bool forceMerge = false)
    {
        if (!initToken && !bodyToken) {
            return nullptr;
        }
        if (!forceMerge && initToken == bodyToken) {
            return initToken;
        }
        auto resultToken = CreateControlToken(base, "for", span, kind);
        auto iterToken = IRContext::Get().MakeVar(resultToken->name_ + "_iter", resultToken->GetType(), span);
        iterArgs.push_back(std::make_shared<IterArg>(iterToken, initToken ? initToken : NoneValue()));
        bodyValues.push_back(bodyToken ? bodyToken : NoneValue());
        returnVars.push_back(resultToken);
        return resultToken;
    }

    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override
    {
        std::vector<StmtPtr> statements;
        statements.reserve(op->stmts_.size());
        for (const auto& statement : op->stmts_) {
            statements.push_back(VisitStmt(statement));
        }
        return std::make_shared<SeqStmts>(std::move(statements), op->span_);
    }

    StmtPtr VisitStmt_(const TensorOpStmtPtr& op) override
    {
        std::vector<TensorAccess> writes;
        std::unordered_set<RawTensor*> writtenRaws;
        for (const auto& result : op->result_) {
            auto tensor = AsLogicalTensor(result);
            auto token = CreateTokenVar(tensor, op->span_, TokenKind::WRITE);
            tensor->SetWriteToken(token);
            auto* raw = storage_.Find(tensor->GetRawTensor().get());
            writes.push_back({raw, std::move(token)});
            writtenRaws.insert(raw);
        }

        std::vector<TensorAccess> reads;
        std::unordered_map<RawTensor*, VarPtr> opReadTokens;
        for (const auto& arg : op->args_) {
            auto tensor = AsLogicalTensor(arg);
            auto* raw = storage_.Find(tensor->GetRawTensor().get());
            auto it = opReadTokens.find(raw);
            if (it != opReadTokens.end()) {
                tensor->SetReadToken(it->second);
                continue;
            }

            VarPtr token;
            if (!writtenRaws.count(raw)) {
                token = rawTokenStates_[raw].latestRead;
            }
            if (!token) {
                token = CreateTokenVar(tensor, op->span_, TokenKind::READ);
            }
            tensor->SetReadToken(token);
            opReadTokens.emplace(raw, token);
            reads.push_back({raw, std::move(token)});
        }

        std::vector<VarPtr> tokens = op->tokens_;
        std::unordered_set<VarPtr> seenTokens(tokens.begin(), tokens.end());
        for (const auto& read : reads) {
            if (!writtenRaws.count(read.raw)) {
                AppendToken(rawTokenStates_[read.raw].latestWrite, tokens, seenTokens);
            }
        }
        for (const auto& write : writes) {
            const auto& state = rawTokenStates_[write.raw];
            AppendToken(state.latestRead ? state.latestRead : state.latestWrite, tokens, seenTokens);
        }

        std::vector<VarPtr> resultTokens = op->result_token_;
        for (const auto& write : writes) {
            resultTokens.push_back(write.token);
        }
        for (const auto& read : reads) {
            resultTokens.push_back(read.token);
        }

        auto mutableOp = std::const_pointer_cast<TensorOpStmt>(op);
        mutableOp->tokens_ = std::move(tokens);
        mutableOp->result_token_ = std::move(resultTokens);

        for (const auto& write : writes) {
            rawTokenStates_[write.raw] = RawTokenState{write.token, nullptr, 0};
        }
        for (const auto& read : reads) {
            if (!writtenRaws.count(read.raw)) {
                auto& state = rawTokenStates_[read.raw];
                state.latestRead = read.token;
                state.readRevision = ++readRevision_;
            }
        }
        return op;
    }

    StmtPtr VisitStmt_(const IfStmtPtr& op) override
    {
        auto incomingStates = rawTokenStates_;

        rawTokenStates_ = incomingStates;
        auto thenBody = SeqStmts::AsMut(VisitStmt(op->thenBody_));
        auto thenStates = rawTokenStates_;

        rawTokenStates_ = incomingStates;
        std::optional<SeqStmtsPtr> elseBody;
        RawTokenStateMap elseStates = incomingStates;
        if (op->elseBody_) {
            elseBody = SeqStmts::AsMut(VisitStmt(*op->elseBody_));
            elseStates = rawTokenStates_;
        }

        auto thenTerminatorValues = GetTerminatorValues(thenBody);
        auto elseTerminatorValues = elseBody ? GetTerminatorValues(*elseBody) : std::vector<ExprPtr>{};
        std::vector<ExprPtr> thenTokenValues;
        std::vector<ExprPtr> elseTokenValues;
        std::vector<VarPtr> returnVars = op->returnVars_;
        std::unordered_map<RawTensor*, VarPtr> returnedTensorWrites;

        size_t originalReturnCount = op->returnVars_.size();
        ASSERT(!elseBody || thenTerminatorValues.size() == elseTerminatorValues.size());
        for (size_t i = 0; i < originalReturnCount; ++i) {
            auto returnTensor = AsLogicalTensor(op->returnVars_[i]);
            if (!returnTensor) {
                continue;
            }
            auto thenTensor = AsLogicalTensor(thenTerminatorValues[i]);
            auto elseTensor = elseBody ? AsLogicalTensor(elseTerminatorValues[i]) : nullptr;
            auto thenToken = thenTensor ? thenTensor->GetWriteToken() : nullptr;
            auto elseToken = elseTensor ? elseTensor->GetWriteToken() : nullptr;
            auto token = MergeIfToken(thenToken, elseToken, returnTensor->name_, TokenKind::WRITE, op->span_,
                                      thenTokenValues, elseTokenValues, returnVars);
            if (token) {
                returnTensor->SetWriteToken(token);
                if (returnTensor->GetRawTensor()) {
                    returnedTensorWrites[storage_.Find(returnTensor->GetRawTensor().get())] = token;
                }
            }
        }

        RawTokenStateMap mergedStates = incomingStates;
        const auto& liveOutRaws = controlLiveOutRaws_[op.get()];
        for (const auto& [raw, baseName] : registeredRaws_) {
            if (!liveOutRaws.count(raw)) {
                continue;
            }
            auto thenState = GetRawState(thenStates, raw);
            auto elseState = GetRawState(elseStates, raw);
            RawTokenState mergedState;
            auto returnedWrite = returnedTensorWrites.find(raw);
            if (returnedWrite != returnedTensorWrites.end()) {
                mergedState.latestWrite = returnedWrite->second;
            } else {
                mergedState.latestWrite = MergeIfToken(thenState.latestWrite, elseState.latestWrite, baseName,
                                                       TokenKind::WRITE, op->span_, thenTokenValues, elseTokenValues,
                                                       returnVars);
            }
            bool sameReadState = thenState.latestRead == elseState.latestRead &&
                                 thenState.readRevision == elseState.readRevision;
            mergedState.latestRead = MergeIfToken(thenState.latestRead, elseState.latestRead, baseName, TokenKind::READ,
                                                  op->span_, thenTokenValues, elseTokenValues, returnVars,
                                                  !sameReadState);
            mergedState.readRevision = mergedState.latestRead ?
                                           (sameReadState ? thenState.readRevision : ++readRevision_) :
                                           0;
            if (mergedState.latestWrite || mergedState.latestRead) {
                mergedStates[raw] = std::move(mergedState);
            } else {
                mergedStates.erase(raw);
            }
        }
        for (const auto& [raw, token] : returnedTensorWrites) {
            mergedStates[raw].latestWrite = token;
        }
        rawTokenStates_ = std::move(mergedStates);

        thenBody = AppendTerminatorValues(thenBody, thenTokenValues);
        if (elseBody) {
            elseBody = AppendTerminatorValues(*elseBody, elseTokenValues);
        } else if (!elseTokenValues.empty()) {
            elseBody = std::make_shared<SeqStmts>(
                std::vector<StmtPtr>{std::make_shared<YieldStmt>(elseTokenValues, op->span_)}, op->span_);
        }

        return std::make_shared<IfStmt>(op->condition_, thenBody, elseBody, std::move(returnVars), op->span_);
    }

    StmtPtr VisitStmt_(const ForStmtPtr& op) override
    {
        auto incomingStates = rawTokenStates_;
        rawTokenStates_ = incomingStates;
        auto body = SeqStmts::AsMut(VisitStmt(op->body_));
        auto bodyStates = rawTokenStates_;

        auto bodyTerminatorValues = GetTerminatorValues(body);
        std::vector<IterArgPtr> iterArgs = op->iterArgs_;
        std::vector<ExprPtr> bodyTokenValues;
        std::vector<VarPtr> returnVars = op->returnVars_;
        std::unordered_map<RawTensor*, VarPtr> returnedTensorWrites;

        size_t originalReturnCount = op->returnVars_.size();
        ASSERT(op->iterArgs_.size() == bodyTerminatorValues.size());
        for (size_t i = 0; i < originalReturnCount; ++i) {
            auto returnTensor = AsLogicalTensor(op->returnVars_[i]);
            if (!returnTensor) {
                continue;
            }
            auto initTensor = AsLogicalTensor(op->iterArgs_[i]->initValue_);
            auto bodyTensor = AsLogicalTensor(bodyTerminatorValues[i]);
            auto initToken = initTensor ? initTensor->GetWriteToken() : nullptr;
            auto bodyToken = bodyTensor ? bodyTensor->GetWriteToken() : nullptr;
            auto token = AddForToken(initToken, bodyToken, returnTensor->name_, TokenKind::WRITE, op->span_, iterArgs,
                                     bodyTokenValues, returnVars);
            if (token) {
                returnTensor->SetWriteToken(token);
                if (returnTensor->GetRawTensor()) {
                    returnedTensorWrites[storage_.Find(returnTensor->GetRawTensor().get())] = token;
                }
            }
        }

        RawTokenStateMap mergedStates = incomingStates;
        const auto& liveOutRaws = controlLiveOutRaws_[op.get()];
        for (const auto& [raw, baseName] : registeredRaws_) {
            if (!liveOutRaws.count(raw)) {
                continue;
            }
            auto initState = GetRawState(incomingStates, raw);
            auto bodyState = GetRawState(bodyStates, raw);
            RawTokenState mergedState;
            auto returnedWrite = returnedTensorWrites.find(raw);
            if (returnedWrite != returnedTensorWrites.end()) {
                mergedState.latestWrite = returnedWrite->second;
            } else {
                mergedState.latestWrite = AddForToken(initState.latestWrite, bodyState.latestWrite, baseName,
                                                      TokenKind::WRITE, op->span_, iterArgs, bodyTokenValues,
                                                      returnVars);
            }
            bool sameReadState = initState.latestRead == bodyState.latestRead &&
                                 initState.readRevision == bodyState.readRevision;
            mergedState.latestRead = AddForToken(initState.latestRead, bodyState.latestRead, baseName, TokenKind::READ,
                                                 op->span_, iterArgs, bodyTokenValues, returnVars, !sameReadState);
            mergedState.readRevision = mergedState.latestRead ?
                                           (sameReadState ? initState.readRevision : ++readRevision_) :
                                           0;
            if (mergedState.latestWrite || mergedState.latestRead) {
                mergedStates[raw] = std::move(mergedState);
            } else {
                mergedStates.erase(raw);
            }
        }
        for (const auto& [raw, token] : returnedTensorWrites) {
            mergedStates[raw].latestWrite = token;
        }
        rawTokenStates_ = std::move(mergedStates);

        body = AppendTerminatorValues(body, bodyTokenValues);
        return std::make_shared<ForStmt>(op->loopVar_, op->start_, op->stop_, op->step_, std::move(iterArgs), body,
                                         std::move(returnVars), op->span_, op->attrs_);
    }

    StorageClasses storage_;
    RawTokenStateMap rawTokenStates_;
    uint64_t readRevision_{0};
    RawNameMap registeredRaws_;
    RawSet liveRaws_;
    std::unordered_map<const Stmt*, RawSet> controlLiveOutRaws_;
};

SeqStmtsPtr RunInferTokenPass(SeqStmtsPtr seq)
{
    InferTokenPass inferTokenPass;
    return inferTokenPass.Apply(seq);
}

} // namespace pypto::ir
