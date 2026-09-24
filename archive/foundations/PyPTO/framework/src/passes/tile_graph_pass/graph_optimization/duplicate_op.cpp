/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file duplicate_op.cpp
 * \brief
 */

#include "duplicate_op.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_check/duplicate_op_checker.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/infer_shape_utils.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "DuplicateOp"

namespace npu::tile_fwk {
namespace {
bool IsViewLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_SLICE; }
} // namespace

Status DuplicateOp::PreCheck(Function& function)
{
    DuplicateOpChecker checker;
    return checker.DoPreCheck(function);
}

Status DuplicateOp::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start %s for function [%s].", MODULE_NAME,
                     function.GetRawName().c_str());
    std::vector<Operation*> newOps;
    if (Process(function, newOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Process failed.");
        return FAILED;
    }
    if (!newOps.empty()) {
        if (InferShapeUtils::InferShape(function, newOps) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InferShape for new operations failed.");
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End %s for function [%s].", MODULE_NAME, function.GetRawName().c_str());
    return SUCCESS;
}

Status DuplicateOp::PostCheck(Function& function)
{
    DuplicateOpChecker checker;
    return checker.DoPostCheck(function);
}

Status DuplicateOp::ProcessGatherIn(Function& function, Operation& operation, std::vector<Operation*>& newOps) const
{
    for (const auto& oOperand : operation.GetOOperands()) {
        if (oOperand == nullptr) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "%s[%d]'s oOperand cannot be nullptr; Please check if the oOperand of %s[%d] is nullptr.%s",
                operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), operation.GetOpcodeStr().c_str(),
                operation.GetOpMagic(), GetFormatBacktrace(operation).c_str());
            return FAILED;
        }
        bool isFirst = true;
        auto iOperand = operation.iOperand[0];
        auto consumers = oOperand->GetConsumers(); // copy consumers to avoid erase while iteration
        for (auto& consumer : consumers) {
            if (consumer == nullptr) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor,
                    "OP_GATHER_IN_L1's consumer cannot be nullptr; Please check if the output of OP_GATHER_IN_L1[%d]'s "
                    "consumer is nullptr.",
                    oOperand->GetMagic());
                return FAILED;
            }
            if (consumer->GetOpcode() == Opcode::OP_GATHER_IN_L1) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor,
                    "OP_GATHER_IN_L1's consumer cannot be OP_GATHER_IN_L1; Please check if the type output of "
                    "OP_GATHER_IN_L1[%d]'s consumer is OP_GATHER_IN_L1.",
                    oOperand->GetMagic());
                return FAILED;
            }
            if (isFirst) {
                isFirst = false;
                continue;
            }
            auto dst = oOperand->Clone(function, true);
            if (dst == nullptr) {
                APASS_LOG_ERROR_F(Elements::Tensor,
                                  "Clone OP_GATHER_IN_L1's oOperand[%d] failed; Please check if dst is nullptr.",
                                  oOperand->GetMagic());
                return FAILED;
            }
            consumer->ReplaceInput(dst, oOperand);
            IRBuilder builder;
            auto& newOp = builder.CreateTensorOpStmt(function, Opcode::OP_GATHER_IN_L1, operation.GetIOperands(), {dst},
                                                     operation.GetSpan());
            newOp.SetScopeInfo(operation.GetScopeInfo());
            newOp.SetAttribute(OpAttributeKey::startOffset, operation.GetIntAttribute(OpAttributeKey::startOffset));
            newOps.push_back(&newOp);
        }
    }
    return SUCCESS;
}

Status DuplicateOp::ProcessViewLike(Function& function, Operation& operation, std::vector<Operation*>& newOps) const
{
    auto viewAttr = dynamic_cast<ViewOpAttribute*>(operation.GetOpAttribute().get());
    if (viewAttr != nullptr &&
        (viewAttr->GetTo() == MEM_L1 || viewAttr->GetTo() == MEM_BT || viewAttr->GetTo() == MEM_FIX_QUANT_PRE)) {
        return SUCCESS;
    }
    auto iOperand = operation.iOperand[0];
    for (const auto& oOperand : operation.oOperand) {
        if (oOperand == nullptr) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Null output operand detected while iterating over the output operands of the operation [%d].%s",
                operation.opmagic, GetFormatBacktrace(operation).c_str());
            return FAILED;
        }
        if (oOperand->GetConsumers().size() == 1) {
            continue;
        }
        auto consumers = oOperand->GetConsumers();
        for (auto& consumer : consumers) {
            if (consumer == nullptr) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor,
                    "Null consumer detected while iterating over the consumers of the output operand [%d].",
                    oOperand->magic);
                return FAILED;
            }
            if (IsViewLikeOpcode(consumer->GetOpcode())) {
                continue;
            }
            auto dst = oOperand->Clone(function, true);
            if (dst == nullptr) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Clone failed for output operand [%d].", oOperand->magic);
                return FAILED;
            }
            consumer->ReplaceInput(dst, oOperand);
            IRBuilder builder;
            auto& newOp = builder.CreateTensorOpStmt(function, operation.GetOpcode(), {iOperand}, {dst},
                                                     operation.GetSpan());
            newOp.SetScopeInfo(operation.GetScopeInfo());
            if (operation.GetOpAttribute() != nullptr) {
                newOp.SetOpAttribute(operation.GetOpAttribute()->Clone());
            }
            newOps.push_back(&newOp);
        }
    }
    return SUCCESS;
}

Status DuplicateOp::ProcessOp(Function& function, Operation& operation, std::vector<Operation*>& newOps) const
{
    auto opcode = operation.GetOpcode();
    if (opcode != Opcode::OP_GATHER_IN_L1 && !IsViewLikeOpcode(opcode)) {
        return SUCCESS;
    }
    if (opcode == Opcode::OP_GATHER_IN_L1) {
        if (ProcessGatherIn(function, operation, newOps) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessGatherIn failed.%s", GetFormatBacktrace(operation).c_str());
            return FAILED;
        }
    }
    if (IsViewLikeOpcode(opcode)) {
        if (ProcessViewLike(function, operation, newOps) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessViewLike failed.%s", GetFormatBacktrace(operation).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

void DuplicateOp::PushProducers(const LogicalTensors& operands, std::stack<Operation*>& stack,
                                std::unordered_set<int>& visitedOps, std::unordered_set<int>& visitedTensors) const
{
    for (const auto& iOperand : operands) {
        if (iOperand == nullptr) {
            continue;
        }
        if (!visitedTensors.insert(iOperand->GetMagic()).second) {
            continue;
        }
        for (auto& producer : iOperand->GetProducers()) {
            if (visitedOps.find(producer->GetOpMagic()) == visitedOps.end()) {
                stack.push(producer);
            }
        }
    }
}

Status DuplicateOp::Process(Function& function, std::vector<Operation*>& newOps) const
{
    std::stack<Operation*> stack;
    std::unordered_set<int> visitedOps;
    std::unordered_set<int> visitedTensors;
    visitedOps.reserve(function.GetOperationSize());
    for (const auto& outcast : function.GetOutcast()) {
        for (auto op : outcast->GetProducers()) {
            stack.push(op);
        }
        while (!stack.empty()) {
            Operation* CurrentOp = stack.top();
            stack.pop();
            if (!visitedOps.insert(CurrentOp->GetOpMagic()).second) {
                continue;
            }
            if (ProcessOp(function, *CurrentOp, newOps) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ProcessOp failed.");
                return FAILED;
            }
            PushProducers(CurrentOp->iOperand, stack, visitedOps, visitedTensors);
        }
    }

    // Skip DCE + SortOperations when no duplication occurred.
    if (newOps.empty()) {
        return SUCCESS;
    }

    DeadOperationEliminator eliminator;
    eliminator.EliminateDeadOperationBackward(function);
    return SUCCESS;
}
} // namespace npu::tile_fwk
