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
 * \file remove_redundant_op_checker.cpp
 * \brief
 */

#include "remove_redundant_op_checker.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "RemoveRedundantOp"

namespace npu {
namespace tile_fwk {
Status RemoveRedundantOpChecker::PreCheckAssemble(Function& function, const Operation& op, const LogicalTensorPtr& in)
{
    uint32_t assembleRemoveNum = 0;
    uint32_t otherOpNum = 0;
    for (const auto& childOp : in->GetConsumers()) {
        if (childOp->GetOpcode() != Opcode::OP_ASSEMBLE) {
            ++otherOpNum;
            continue;
        }
        auto childIn = op.iOperand.front();
        auto childOut = op.oOperand.front();
        if (!childOut->GetConsumers().empty()) {
            ++otherOpNum;
            continue;
        }
        if (childIn->shape == childOut->shape && childIn->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
            childOut->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            ++assembleRemoveNum;
        }
    }
    if (assembleRemoveNum > 1 && otherOpNum > 0) {
        APASS_LOG_ERROR_C(OperationErr::OP_PRODUCER_CONSUMER, Elements::Operation,
                          "More than one assemble ddr op without consumer; Please check the num of assemble ddr "
                          "op without consumer.");
        return FAILED;
    }
    auto consumerOps = function.FindConsumers(op);
    if (consumerOps.empty()) {
        auto assembleOut = op.oOperand.front();
        if (assembleOut->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            if (!function.IsFromOutCast(assembleOut)) {
                APASS_LOG_ERROR_C(OperationErr::OP_PRODUCER_CONSUMER, Elements::Operation,
                                  "Op assembleDDR[%d] has no consumer but is not outcast, please check.",
                                  op.GetOpMagic());
                return FAILED;
            }
        } else {
            APASS_LOG_ERROR_C(OperationErr::OP_PRODUCER_CONSUMER, Elements::Operation,
                              "Op assembleUB[%d] has no consumer, please check.", op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PreCheckView(Function& function, const Operation& op, const LogicalTensorPtr& in)
{
    auto out = op.oOperand.front();
    if (in->shape == out->shape && op.ConsumerOps().empty() && in->GetConsumers().size() > 1 &&
        function.IsFromOutCast(out)) {
        APASS_LOG_ERROR_C(OperationErr::OP_PRODUCER_CONSUMER, Elements::Operation,
                          "There is another op consumes the input of a view op[%d] (the output is an outcast) without "
                          "consumer; Please check view op[%d].%s",
                          op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (function.IsFromOutCast(out)) {
        APASS_LOG_ERROR_C(OperationErr::OP_SPECIAL_CONSTRAINT, Elements::Operation,
                          "The output of the op[%d] is an outCast; Please check the type of output from the op[%d].%s",
                          op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

// 要求regcopy一定有后继op
Status RemoveRedundantOpChecker::PreCheckRegCopy(Function& function, const Operation& op)
{
    auto consumerOps = function.FindConsumers(op);
    if (consumerOps.empty()) {
        APASS_LOG_ERROR_C(
            OperationErr::OP_PRODUCER_CONSUMER, Elements::Operation,
            "PreCheck for regcopy op[%d] failed: The output of regcopy has no consumer. Please check regcopy op[%d].%s",
            op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::ProcessPreCheck(Function& function, const Operation& op)
{
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Process preCheck for assemble op[%d].", op.GetOpMagic());
        auto assembleIn = op.iOperand.front();
        if (PreCheckAssemble(function, op, assembleIn) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PreCheck for assemble op[%d] failed.%s", op.GetOpMagic(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    } else if (op.GetOpcode() == Opcode::OP_VIEW) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Process preCheck for view op[%d].", op.GetOpMagic());
        auto viewIn = op.iOperand.front();
        if (PreCheckView(function, op, viewIn) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PreCheck for view op[%d] failed.%s", op.GetOpMagic(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    } else if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Process preCheck for regcopy op[%d].", op.GetOpMagic());
        if (PreCheckRegCopy(function, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PreCheck for regcopy op[%d] failed.%s", op.GetOpMagic(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckAssemble(Function& function, const Operation& op)
{
    auto assembleIn = op.iOperand.front();
    auto assembleOut = op.oOperand.front();
    auto parentOp = *assembleIn->GetProducers().begin();
    if (parentOp == nullptr) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "The input of assemble [%d] has no producer; Please check the input of assemble [%d] to ensure that it has "
            "producers.%s",
            op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (assembleOut->GetProducers().size() > 1) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Assemble's out[%d] has more than one producer, skip checking.",
                          assembleOut->GetMagic());
        return SUCCESS;
    }
    bool hasParallelAssemble = false;
    for (const auto& consumer : assembleIn->GetConsumers()) {
        if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE && consumer->GetOpMagic() != op.GetOpMagic()) {
            hasParallelAssemble = true;
            break;
        }
    }
    bool hasReshapeConsumer = false;
    for (const auto& consumer : assembleOut->GetConsumers()) {
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE) {
            hasReshapeConsumer = true;
            break;
        }
    }
    if (hasParallelAssemble && hasReshapeConsumer) {
        return SUCCESS;
    }
    // 镜像 RemoveRedundantOp 的 outcast 保护：output 为 outcast 且 input 有其他 consumer 的
    // ASSEMBLE 被 pass 有意保留，豁免同 shape/memtype 的冗余检查
    if (function.IsFromOutCast(assembleOut) && assembleIn->GetConsumers().size() > 1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Assemble[%d] kept for outcast protection, skip redundancy check.",
                          op.GetOpMagic());
        return SUCCESS;
    }
    if (assembleIn->shape == assembleOut->shape &&
        assembleIn->GetMemoryTypeOriginal() == assembleOut->GetMemoryTypeOriginal()) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "PostCheck for assemble op[%d] failed: input and output has the same shape and memorytype; Please check "
            "assemble op[%d].%s",
            op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckView(const Operation& op)
{
    auto viewIn = op.iOperand.front();
    auto viewOut = op.oOperand.front();
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
    if (viewOpAttribute != nullptr && viewOpAttribute->GetToDynValidShape().empty() &&
        viewIn->shape == viewOut->shape && viewIn->GetMemoryTypeOriginal() == viewOut->GetMemoryTypeOriginal()) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "PostCheck for view op[%d] failed: DynValidShape empty, same shape and memory type; Please check view "
            "op[%d].%s",
            op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (viewOut->GetConsumers().size() != 1) {
        return SUCCESS;
    }
    auto childOp = *(viewOut->GetConsumers().begin());
    if (childOp == nullptr) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Found null childOp of op[%d]; Please check if the output consumers of the op[%d] are empty.%s",
            op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckRegCopy(const Operation& op)
{
    auto regcopyIn = op.iOperand.front();
    auto regcopyOut = op.oOperand.front();
    if (regcopyIn->shape == regcopyOut->shape &&
        regcopyIn->GetMemoryTypeOriginal() == regcopyOut->GetMemoryTypeOriginal()) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "PostCheck for regcopy op[%d] failed: the shape of input equals to the output; Please check regcopy "
            "op[%d].%s",
            op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckCopyIn(const Operation& op)
{
    auto copyIn = op.iOperand.front();
    auto copyOut = op.oOperand.front();
    if (copyIn->shape == copyOut->shape && copyOut->GetMemoryTypeOriginal() == npu::tile_fwk::MEM_L1) {
        bool isRedundant = true;
        for (const auto& producerOp : op.ProducerOps()) {
            if (producerOp == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Found null producer of op[%d]; Please check the producers of op[%d].%s",
                                  op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            if (producerOp->GetOpcode() != Opcode::OP_VIEW) {
                isRedundant = false;
                break;
            }
        }
        if (isRedundant) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "PostCheck for copyin op[%d] failed: the producers of the op are view; Please check copyin op[%d].%s",
                op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::ProcessPostCheck(Function& function, const Operation& op)
{
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        if (PostCheckAssemble(function, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for Assemble failed.%s", GetFormatBacktrace(op).c_str());
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
    if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
        if (PostCheckRegCopy(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for RegCopy failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    if (op.GetOpcode() == Opcode::OP_COPY_IN) {
        if (PostCheckCopyIn(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for CopyIn failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::DoPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for RemoveRedundantOp");
    if (CheckValidOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Found invalid op in the function.");
        return FAILED;
    }
    if (CheckOpIOValid(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Found invalid input/output from the function.");
        return FAILED;
    }
    for (const auto& op : function.Operations()) {
        if (ProcessPreCheck(function, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PreCheck for RemoveRedundantOp failed.%s",
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for RemoveRedundantOp");
    if (CheckOpIOValid(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Found invalid input or output in the function.");
        return FAILED;
    }
    for (const auto& op : function.Operations()) {
        if (ProcessPostCheck(function, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for RemoveRedundantOp failed.%s",
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
