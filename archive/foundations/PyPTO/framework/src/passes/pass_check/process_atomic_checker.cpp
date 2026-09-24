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
 * \file process_atomic_checker.cpp
 * \brief Checker for ProcessAtomic pass
 */

#include "process_atomic_checker.h"
#include "interface/operation/attribute.h"
#include "interface/operation/operation.h"
#include "tilefwk/tilefwk_op.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "ProcessAtomicChecker"

namespace npu {
namespace tile_fwk {

Status ProcessAtomicChecker::DoPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "PreCheck for ProcessAtomic.");
    if (CheckGraphLoop(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Loopcheck failed; Please check if there is a Loop.");
        return FAILED;
    }
    if (CheckCompleteness(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckCompleteness for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    for (const auto& op : function.Operations()) {
        if (ProcessPreCheck(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PreCheck for ProcessAtomic failed.%s",
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for ProcessAtomic success.");
    return SUCCESS;
}

Status ProcessAtomicChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for ProcessAtomic.");
    if (CheckCompleteness(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckCompleteness for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckNoReduceAcc(function) != SUCCESS) {
        return FAILED;
    }
    if (CheckNoAtomicRMW(function) != SUCCESS) {
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for ProcessAtomic success.");
    return SUCCESS;
}

Status ProcessAtomicChecker::ProcessPreCheck(const Operation& op)
{
    if (op.GetOpcode() == Opcode::OP_A_MUL_B || op.GetOpcode() == Opcode::OP_A_MULACC_B) {
        if (CheckMulOpValidity(op) != SUCCESS) {
            return FAILED;
        }
    }
    if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
        if (CheckReduceAccOpValidity(op) != SUCCESS) {
            return FAILED;
        }
    }
    if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
        if (ValidateAtomicRMW(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Op[%d] validation failed; Please check the atomic rmw operation validity.%s",
                              op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessAtomicChecker::CheckMulOpValidity(const Operation& op)
{
    if (op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Invalid op: [%d] has output num not equal to one; Please check if the output num is one.%s",
                          op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    auto output = op.GetOOperands().front();
    if ((output->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) || (*output->GetConsumers().begin() == nullptr)) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Op[%d] has invalid output tensor[%d]; Please check if the output tensor is vaild.%s",
                          op.GetOpMagic(), output->magic, GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessAtomicChecker::CheckReduceAccOpValidity(const Operation& op)
{
    if (op.GetIOperands().size() < 1) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op[%d] has input num less than 1; Please check the input num.%s",
                          op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Op[%d] has output num not equal to one; Please check if the output num for is one.%s",
                          op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    for (const auto& in : op.GetIOperands()) {
        if (in->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Op[%d] has non-DDR input tensor[%d]; Please check the memory type of the input tensor.%s",
                op.GetOpMagic(), in->magic, GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    for (const auto& out : op.GetOOperands()) {
        if (out->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Op[%d] has non-DDR output tensor[%d]; Please check the memory type of the output tensor.%s",
                op.GetOpMagic(), out->magic, GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessAtomicChecker::ValidateAtomicRMW(const Operation& op)
{
    if (op.GetIOperands().size() < 1) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Op[%d] has input producers num less than 1; Please check the input num.", op.GetOpMagic());
        return FAILED;
    }
    if (CheckAtomicRMWMemoryType(op) != SUCCESS) {
        return FAILED;
    }
    if (CheckAtomicRMWShape(op) != SUCCESS) {
        return FAILED;
    }
    if (CheckAtomicRMWOffset(op) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessAtomicChecker::CheckAtomicRMWMemoryType(const Operation& op)
{
    for (const auto& in : op.GetIOperands()) {
        if (in->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Op[%d] has non-DDR input tensor[%d]; Please check the memory type of the input tensor.",
                              op.GetOpMagic(), in->magic);
            return FAILED;
        }
    }
    for (const auto& out : op.GetOOperands()) {
        if (out->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Op[%d] has non-DDR output tensor[%d]; Please check the memory type of the output tensor.",
                op.GetOpMagic(), out->magic);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessAtomicChecker::CheckAtomicRMWShape(const Operation& op)
{
    auto& inputShape = op.GetIOperands().front()->GetShape();
    auto& outputShape = op.GetOOperands().front()->GetShape();
    if (outputShape.size() < inputShape.size()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Op[%d] output shape size less than input shape size; Please check shape validity.",
                          op.GetOpMagic());
        return FAILED;
    }
    for (size_t i = 0; i < inputShape.size(); ++i) {
        if (outputShape[i] < inputShape[i]) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Op[%d] output shape[%zu]=%ld less than input shape[%zu]=%ld; Please check shape validity.",
                op.GetOpMagic(), i, outputShape[i], i, inputShape[i]);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessAtomicChecker::CheckAtomicRMWOffset(const Operation& op)
{
    auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
    if (assembleAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Op[%d] missing AssembleOpAttribute; Please check if offset attribute is set.",
                          op.GetOpMagic());
        return FAILED;
    }
    auto& toOffset = assembleAttr->GetToOffset();
    for (size_t i = 0; i < toOffset.size(); ++i) {
        if (toOffset[i] < 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "Op[%d] offset[%zu]=%ld is negative; Please check offset validity.",
                              op.GetOpMagic(), i, toOffset[i]);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessAtomicChecker::CheckNoReduceAcc(Function& function)
{
    for (const auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Op[%d] OP_REDUCE_ACC still exists after ProcessAtomic pass; "
                              "Please check if the ReduceAcc was properly eliminated.%s",
                              op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessAtomicChecker::CheckNoAtomicRMW(Function& function)
{
    for (const auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Op[%d] OP_ATOMIC_RMW still exists after ProcessAtomic pass; "
                              "Please check if the AtomicRMW was properly eliminated.%s",
                              op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
