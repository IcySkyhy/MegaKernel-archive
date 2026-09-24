/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <algorithm>
#include <vector>

#include "interface/operation/operation.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "ir/type.h"

namespace npu::tile_fwk {

inline constexpr const char* kContractWriteTokenOut = "contract_write_token_out";
inline constexpr const char* kContractWriteTokenIn = "contract_write_token_in";
inline constexpr const char* kContractReadTokenIn = "contract_read_token_in";

inline bool IsMemoryWriteOpcode(Opcode opcode)
{
    return opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_ASSEMBLE_SSA || opcode == Opcode::OP_ATOMIC_RMW;
}

inline bool IsWriteSemanticToken(const ir::VarPtr& token)
{
    if (!token) {
        return false;
    }
    auto tokenType = std::dynamic_pointer_cast<const ir::TokenType>(token->GetType());
    return tokenType != nullptr && tokenType->kind_ == ir::TokenKind::WRITE;
}

inline bool IsReadSemanticToken(const ir::VarPtr& token)
{
    if (!token) {
        return false;
    }
    auto tokenType = std::dynamic_pointer_cast<const ir::TokenType>(token->GetType());
    return tokenType != nullptr && tokenType->kind_ == ir::TokenKind::READ;
}

inline std::vector<ir::VarPtr> CollectWriteSemanticTokens(const std::vector<ir::VarPtr>& tokens)
{
    std::vector<ir::VarPtr> writeTokens;
    for (const auto& token : tokens) {
        if (IsWriteSemanticToken(token)) {
            writeTokens.push_back(token);
        }
    }
    return writeTokens;
}

inline std::vector<ir::VarPtr> CollectReadSemanticTokens(const std::vector<ir::VarPtr>& tokens)
{
    std::vector<ir::VarPtr> readTokens;
    for (const auto& token : tokens) {
        if (IsReadSemanticToken(token)) {
            readTokens.push_back(token);
        }
    }
    return readTokens;
}

inline void SortTokenVars(std::vector<ir::VarPtr>& tokens)
{
    std::sort(tokens.begin(), tokens.end(),
              [](const ir::VarPtr& lhs, const ir::VarPtr& rhs) { return lhs->name_ < rhs->name_; });
}

inline void AppendSemanticNormalTokens(std::vector<ir::VarPtr>& target, const std::vector<ir::VarPtr>& semanticTokens)
{
    for (const auto& semantic : semanticTokens) {
        if (auto normal = IRContext::Get().GetNormalToken(semantic)) {
            target.push_back(normal);
        }
    }
}

inline void AttachContractWriteTokenAttrs(const LogicalTensorPtr& tensor, const Operation& sourceOp)
{
    if (!tensor || !IsMemoryWriteOpcode(sourceOp.GetOpcode())) {
        return;
    }
    auto outTokens = CollectWriteSemanticTokens(sourceOp.result_token_);
    auto writeInTokens = CollectWriteSemanticTokens(sourceOp.tokens_);
    auto readInTokens = CollectReadSemanticTokens(sourceOp.tokens_);
    if (!outTokens.empty()) {
        tensor->SetAttr(kContractWriteTokenOut, std::move(outTokens));
    }
    if (!writeInTokens.empty()) {
        tensor->SetAttr(kContractWriteTokenIn, std::move(writeInTokens));
    }
    if (!readInTokens.empty()) {
        tensor->SetAttr(kContractReadTokenIn, std::move(readInTokens));
    }
}

inline void ApplyContractWriteNormalTokens(Operation& contractOp, const LogicalTensorPtr& tensor)
{
    if (!tensor) {
        return;
    }
    auto deps = IRContext::Get().GetDependToken(std::static_pointer_cast<ir::Expr>(tensor));
    if (!deps.empty()) {
        contractOp.result_token_ = {deps[0]};
    }
    if (tensor->HasAttr(kContractReadTokenIn)) {
        std::vector<ir::VarPtr> readInTokens;
        tensor->GetAttr(kContractReadTokenIn, readInTokens);
        AppendSemanticNormalTokens(contractOp.tokens_, readInTokens);
        SortTokenVars(contractOp.tokens_);
        tensor->RemoveAttr(kContractReadTokenIn);
    }
    if (tensor->HasAttr(kContractWriteTokenOut)) {
        std::vector<ir::VarPtr> outTokens;
        tensor->GetAttr(kContractWriteTokenOut, outTokens);
        AppendSemanticNormalTokens(contractOp.result_token_, outTokens);
        SortTokenVars(contractOp.result_token_);
        tensor->RemoveAttr(kContractWriteTokenOut);
    }
    if (tensor->HasAttr(kContractWriteTokenIn)) {
        std::vector<ir::VarPtr> inTokens;
        tensor->GetAttr(kContractWriteTokenIn, inTokens);
        AppendSemanticNormalTokens(contractOp.tokens_, inTokens);
        SortTokenVars(contractOp.tokens_);
        tensor->RemoveAttr(kContractWriteTokenIn);
    }
}

} // namespace npu::tile_fwk
