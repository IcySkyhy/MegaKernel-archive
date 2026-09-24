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
 * \file remove_redundant_reshape.cpp
 * \brief
 */

#include "remove_redundant_reshape.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_check/remove_redundant_reshape_checker.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/merge_view_assemble_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/view_reshape_assemble_reorder_utils.h"

#define MODULE_NAME "RemoveRedundantReshape"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
namespace {
Status CheckIOOperands(const Operation& op, LogicalTensorPtr& in, LogicalTensorPtr& out)
{
    if (op.GetIOperands().size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op [%d] has invalid input operands. %s", op.GetOpMagic(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op [%d] has invalid input operands. %s", op.GetOpMagic(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    in = op.GetIOperands().front();
    if (in == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op [%d] has null input tensor. %s", op.GetOpMagic(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    out = op.GetOOperands().front();
    if (out == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op [%d] has null output tensor. %s", op.GetOpMagic(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}
} // namespace

Status RemoveRedundantReshape::RunOnFunction(Function& function)
{
    if (ViewReshapeAssembleReorderUtils::ReorderViewReshapeAssemble(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to reorder view/reshape/assemble in function [%s].",
                          function.GetRawName().c_str());
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "Start RemoveRedundantReshape for function [%s].",
                     function.GetRawName().c_str());
    if (RemoveReshape(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to remove redundant reshape in function [%s].",
                          function.GetRawName().c_str());
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "End RemoveRedundantReshape for function [%s].",
                     function.GetRawName().c_str());
    return SUCCESS;
}

Status RemoveRedundantReshape::RemoveReshape(Function& function) const
{
    std::unordered_set<Operation*> redundantResapes;
    LogicalTensorPtr in;
    LogicalTensorPtr out;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        if (CheckIOOperands(op, in, out) != SUCCESS) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Op [%d] has invalid input or output operands; Check if op has valid input and output. %s",
                op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if (CommonUtils::ContainsNegativeOne(in->GetShape()) || CommonUtils::ContainsNegativeOne(out->GetShape())) {
            continue;
        }
        auto consumers = out->GetConsumers();
        bool allConsumersIsReshape = true;
        for (auto& consumerOp : consumers) {
            if (consumerOp == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Consumer of op [%d] is null; Check if consumer is valid. %s",
                                  op.GetOpMagic(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            if (in->shape != out->shape && consumerOp->GetOpcode() != Opcode::OP_RESHAPE) {
                allConsumersIsReshape = false;
                continue;
            }
            consumerOp->ReplaceInput(in, out);
        }
        if (allConsumersIsReshape == true && !function.IsFromOutCast(out)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "All consummers of op [%d] are reshape and the shapes are not -1.",
                              op.GetOpMagic());
            redundantResapes.insert(&op);
        }
    }
    if (!redundantResapes.empty()) {
        for (auto& ele : redundantResapes) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Delete OP_RESHAPE, magic [%d].", ele->GetOpMagic());
            if (ele->IsDeleted()) {
                APASS_LOG_ERROR_F(Elements::Operation, "Op [%d] is already marked as deleted. %s", ele->GetOpMagic(),
                                  GetFormatBacktrace(*ele).c_str());
                return FAILED;
            }
            ele->SetAsDeleted();
        }
        function.EraseOperations(false);
    }
    return SUCCESS;
}

Status RemoveRedundantReshape::DefaultEnabledPreCheck(Function& function)
{
    RemoveRedundantReshapeChecker checker;
    return checker.DoDefaultEnabledPreCheck(function);
}

Status RemoveRedundantReshape::PreCheck(Function& function)
{
    RemoveRedundantReshapeChecker checker;
    return checker.DoPreCheck(function);
}

Status RemoveRedundantReshape::PostCheck(Function& function)
{
    RemoveRedundantReshapeChecker checker;
    return checker.DoPostCheck(function);
}
} // namespace npu::tile_fwk
