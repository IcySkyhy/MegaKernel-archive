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
 * \file remove_redundant_reshape_checker.cpp
 * \brief
 */

#include "remove_redundant_reshape_checker.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "RemoveRedundantReshape"

namespace npu {
namespace tile_fwk {
Status RemoveRedundantReshapeChecker::DoDefaultEnabledPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "DoDefaultEnabledPreCheck for RemoveRedundantShape.");
    if (CheckValidOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Found invalid op from the function [%s].",
                          function.GetRawName().c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantReshapeChecker::DoPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "PreCheck for RemoveRedundantShape.");
    if (CheckOpIOValid(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Found invalid input/output in the function [%s].",
                          function.GetRawName().c_str());
        return FAILED;
    }
    if (CheckLocalTensor(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Found invalid tensor in the function [%s].",
                          function.GetRawName().c_str());
        return FAILED;
    }
    for (const auto& op : function.Operations().DuplicatedOpList()) {
        if (ProcessPreCheck(*op)) {
            APASS_LOG_ERROR_F(Elements::Operation, "Precheck RemoveRedundantShape failed. %s",
                              GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantReshapeChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for RemoveRedundantShape.");
    for (const auto& op : function.Operations().DuplicatedOpList()) {
        if (ProcessPostCheck(*op)) {
            APASS_LOG_ERROR_F(Elements::Operation, "Postcheck RemoveRedundantShape failed. %s",
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantReshapeChecker::ProcessPreCheck(const Operation& op)
{
    if (op.GetOpcode() == Opcode::OP_RESHAPE) {
        auto in = op.iOperand.front();
        if (PreCheckReshape(in) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Precheck of reshape op[%d] failed. %s", op.GetOpMagic(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

// PreCheck for reshape
// ..->reshape->out (will be removed regardless of its function)
Status RemoveRedundantReshapeChecker::PreCheckReshape(const LogicalTensorPtr& in)
{
    for (auto& childOp : in->GetConsumers()) {
        if (childOp->GetOpcode() == Opcode::OP_RESHAPE) {
            if (childOp->ConsumerOps().empty()) {
                APASS_LOG_ERROR_C(OperationErr::OP_PRODUCER_CONSUMER, Elements::Operation,
                                  "At least one reshape op without consumer.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status RemoveRedundantReshapeChecker::ProcessPostCheck(const Operation& op)
{
    if (op.GetOpcode() == Opcode::OP_RESHAPE) {
        const auto in = op.iOperand.front();
        if (PostCheckReshape(in) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Postcheck of reshape op[%d] failed. %s", op.GetOpMagic(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

bool CheckForConsecutiveReshape(const Operation* childOp)
{
    for (const auto& consumer : childOp->GetOOperands()[0]->GetConsumers()) {
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE) {
            return true;
        }
    }
    return false;
}

// Postcheck for reshape
Status RemoveRedundantReshapeChecker::PostCheckReshape(const LogicalTensorPtr& in)
{
    for (const auto& childOp : in->GetConsumers()) {
        if (childOp->GetOpcode() == Opcode::OP_RESHAPE) {
            if (CheckForConsecutiveReshape(childOp)) {
                APASS_LOG_ERROR_F(Elements::Operation, "PostCheckReshape failed: Found consecutive reshape ops.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
