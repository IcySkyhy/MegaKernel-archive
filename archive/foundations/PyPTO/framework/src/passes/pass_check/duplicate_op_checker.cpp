/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file duplicate_op_checker.cpp
 * \brief
 */

#include "duplicate_op_checker.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "DuplicateOp"

namespace npu {
namespace tile_fwk {
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;

Status DuplicateOpChecker::PreCheckGatherIn(const Operation& op)
{
    for (const auto& oOperand : op.GetOOperands()) {
        if (oOperand == nullptr) {
            APASS_LOG_ERROR_C(
                OperationErr::OP_NULL_POINTER, Elements::Operation,
                "%s[%d]'s oOperand cannot be nullptr; Please check if the oOperand of %s[%d] is nullptr.%s",
                op.GetOpcodeStr().c_str(), op.GetOpMagic(), op.GetOpcodeStr().c_str(), op.GetOpMagic(),
                GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto consumers = oOperand->GetConsumers();
        for (auto& consumer : consumers) {
            if (consumer == nullptr) {
                APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Tensor,
                                  "OP_GATHER_IN_L1[%d]'s consumer cannot be nullptr.", oOperand->GetMagic());
                return FAILED;
            }
            if (consumer->GetOpcode() == Opcode::OP_GATHER_IN_L1) {
                APASS_LOG_ERROR_C(OperationErr::OP_SPECIAL_CONSTRAINT, Elements::Tensor,
                                  "OP_GATHER_IN_L1[%d]'s consumer cannot be OP_GATHER_IN_L1.", oOperand->GetMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status DuplicateOpChecker::ProcessPreCheck(const Operation& op)
{
    if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
        if (PreCheckGatherIn(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PreCheck for GatherIn failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status DuplicateOpChecker::PostCheckGatherIn(const Operation& op)
{
    for (const auto& oOperand : op.oOperand) {
        if (oOperand == nullptr) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Null output operand detected while iterating over the output operands of the operation [%d].%s",
                op.opmagic, GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if (oOperand->GetConsumers().size() != kNumOne) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "There can not be more than one node among its consumers for op[%d].%s", op.opmagic,
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status DuplicateOpChecker::PostCheckView(const Operation& op)
{
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
    if (viewAttr != nullptr &&
        (viewAttr->GetTo() == MEM_L1 || viewAttr->GetTo() == MEM_BT || viewAttr->GetTo() == MEM_FIX_QUANT_PRE)) {
        return SUCCESS;
    }
    for (const auto& oOperand : op.oOperand) {
        if (oOperand == nullptr) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Null output operand detected while iterating over the output operands of the operation [%d].%s",
                op.opmagic, GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto consumers = oOperand->GetConsumers();
        uint32_t consumerNum = kNumZero;
        for (auto& consumer : consumers) {
            if (consumer == nullptr) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor,
                    "Null consumer detected while iterating over the consumers of the output operand [%d].",
                    oOperand->magic);
                return FAILED;
            }
            if (consumer->GetOpcode() == Opcode::OP_VIEW) {
                continue;
            }
            consumerNum++;
        }
        if (consumerNum == kNumTwo) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "There can not be more than one non-view node among its consumers for op[%d].%s",
                              op.opmagic, GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status DuplicateOpChecker::ProcessPostCheck(const Operation& op)
{
    if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
        if (PostCheckGatherIn(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for GatherIn failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    if (op.GetOpcode() == Opcode::OP_VIEW) {
        if (PostCheckView(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for View failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    return SUCCESS;
}

Status DuplicateOpChecker::DoPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for DuplicateOp");
    if (CheckCompleteness(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckCompleteness for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    for (const auto& op : function.Operations()) {
        if (ProcessPreCheck(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PreCheck for DuplicateOp failed.%s",
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status DuplicateOpChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for DuplicateOp");
    if (CheckCompleteness(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckCompleteness for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    for (const auto& op : function.Operations()) {
        if (ProcessPostCheck(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for DuplicateOp failed.%s",
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
