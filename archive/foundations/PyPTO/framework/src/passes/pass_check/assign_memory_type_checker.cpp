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
 * \file assign_memory_type_checker.cpp
 * \brief
 */

#include "assign_memory_type_checker.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"

#define MODULE_NAME "AssignMemoryType"

namespace npu {
namespace tile_fwk {
const std::unordered_set<Opcode> AssignMemoryTypeChecker::kValidProducerOpcodes = {
    Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0_AT, Opcode::OP_L1_TO_L0_BT,
    Opcode::OP_VIEW,      Opcode::OP_SLICE,     Opcode::OP_VEC_DUP};
const std::unordered_set<MemoryType> AssignMemoryTypeChecker::kValidViewToTypes = {
    MemoryType::MEM_BT,    MemoryType::MEM_FIX_QUANT_PRE, MemoryType::MEM_L0A,    MemoryType::MEM_L0B,
    MemoryType::MEM_L0AMX, MemoryType::MEM_L0BMX,         MemoryType::MEM_UNKNOWN};
Status AssignMemoryTypeChecker::DoPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start Precheck for AssignMemoryType.");
    auto operations = function.Operations();
    for (auto& operation : operations) {
        Operation* op_ptr = &operation;
        // A_MUL_B操作输入tensor生产者校验
        if (operation.GetOpcode() == Opcode::OP_A_MUL_B) {
            Status status = CheckAmulBInputProducers(operation);
            if (status != SUCCESS) {
                return status;
            }
        }
        // 创建队列，包含当前操作和嵌套深度
        std::queue<std::pair<Operation*, int>> opQueue;
        std::unordered_set<Operation*> visited;

        opQueue.emplace(op_ptr, 1);
        visited.insert(op_ptr);

        while (!opQueue.empty()) {
            auto [currentOp, depth] = opQueue.front();
            opQueue.pop();

            // 嵌套深度达到3失败
            if (depth > 3) {
                APASS_LOG_WARN_F(Elements::Operation,
                                 "Over three view/assemble/reshape operations in sequence, currently reched %d; "
                                 "Potential suboptimal allocation around operation %d.",
                                 depth, currentOp->GetOpMagic());
                return SUCCESS;
            }
            CheckPattern(currentOp, opQueue, depth, visited);
        }
    }
    return SUCCESS;
}

Status AssignMemoryTypeChecker::CheckAmulBInputProducers(Operation& operation)
{
    auto producerOps = operation.ProducerOps();
    const Opcode expectedViewLikeOpcode = config::EnableSlice() ? Opcode::OP_SLICE : Opcode::OP_VIEW;
    const auto& expectedViewLikeOpcodeStr = OpcodeManager::Inst().GetOpcodeStr(expectedViewLikeOpcode);
    for (auto& producerOp : producerOps) {
        auto producerOpcode = producerOp->GetOpcode();
        if (!kValidProducerOpcodes.count(producerOpcode)) {
            APASS_LOG_ERROR_C(OperationErr::OP_SPECIAL_CONSTRAINT, Elements::Operation,
                              "Memory error, %s[%d] has invalid input producer; "
                              "Please check input producer %s[%d]. %s",
                              operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                              producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic(),
                              GetFormatBacktrace(operation).c_str());
            return FAILED;
        }
        if (producerOpcode == Opcode::OP_VIEW || producerOpcode == Opcode::OP_SLICE) {
            if (producerOpcode != expectedViewLikeOpcode) {
                APASS_LOG_ERROR_C(OperationErr::OP_SPECIAL_CONSTRAINT, Elements::Operation,
                                  "Memory error, %s[%d] has invalid input %s; "
                                  "Please check input %s[%d]. %s",
                                  operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                                  producerOp->GetOpcodeStr().c_str(), expectedViewLikeOpcodeStr.c_str(),
                                  producerOp->GetOpMagic(), GetFormatBacktrace(operation).c_str());
                return FAILED;
            }
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(producerOp->GetOpAttribute().get());
            if (viewOpAttribute == nullptr) {
                APASS_LOG_ERROR_C(OperationErr::OP_SPECIAL_CONSTRAINT, Elements::Operation,
                                  "Slice attribute error, %s[%d] has input %s without ViewOpAttribute; "
                                  "Please check input view %s[%d]. %s",
                                  operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                                  producerOp->GetOpcodeStr().c_str(), producerOp->GetOpcodeStr().c_str(),
                                  producerOp->GetOpMagic(), GetFormatBacktrace(operation).c_str());
                return FAILED;
            }
            MemoryType attrToType = viewOpAttribute->GetTo();
            if (!kValidViewToTypes.count(attrToType)) {
                APASS_LOG_ERROR_C(OperationErr::OP_SPECIAL_CONSTRAINT, Elements::Operation,
                                  "Slice attribute error, %s[%d] has invalid input %s(toType: %d); "
                                  "Please check input view %s[%d]. %s",
                                  operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                                  producerOp->GetOpcodeStr().c_str(), attrToType, producerOp->GetOpcodeStr().c_str(),
                                  producerOp->GetOpMagic(), GetFormatBacktrace(operation).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

// 检查view/assemble/reshape的嵌套深度是否大于等于3
void AssignMemoryTypeChecker::CheckPattern(Operation* operation, std::queue<std::pair<Operation*, int>>& opQueue,
                                           int depth, std::unordered_set<Operation*>& visited)
{
    for (auto& tensor : operation->oOperand) {
        if (tensor == nullptr) {
            continue;
        }
        for (auto& consumerOp : tensor->GetConsumers()) {
            if (consumerOp == nullptr) {
                continue;
            }
            if (IsViewLike(consumerOp->GetOpcode()) || IsAssembleLike(consumerOp->GetOpcode()) ||
                consumerOp->GetOpcode() == Opcode::OP_RESHAPE) {
                Operation* consumerOpPtr = consumerOp;
                if (visited.find(consumerOpPtr) == visited.end()) {
                    opQueue.emplace(consumerOpPtr, depth + 1);
                    visited.insert(consumerOpPtr);
                    break;
                }
            }
        }
    }
}

Status AssignMemoryTypeChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start Postcheck for AssignMemoryType.");
    if (CheckTensorNotMemUnknown(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function,
                          "Postcheck for AssignMemoryType failed since tensor has improper memoryType.");
        return FAILED;
    }
    if (CheckMoveOpReachable(function) == FAILED) {
        APASS_LOG_ERROR_F(
            Elements::Function,
            "Postcheck for AssignMemoryType failed since view/assemble has unreachable input-to-output memoryType.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "End Postcheck for AssignMemoryType!");
    return SUCCESS;
}

Status AssignMemoryTypeChecker::CheckTensorNotMemUnknown(Function& function)
{
    TensorSet allTensors = GraphUtils::GetAllTensors(function);
    for (const auto& tensor : allTensors) {
        if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_UNKNOWN) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d]'s memoryType is still unknown.", tensor->GetMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status AssignMemoryTypeChecker::CheckMoveOpReachable(Function& function)
{
    auto operations = function.Operations();
    for (const auto& operation : operations) {
        if (OpcodeManager::Inst().GetOpCalcType(operation.GetOpcode()) != OpCalcType::MOVE_LOCAL) {
            continue;
        }
        for (const auto& input : operation.GetIOperands()) {
            for (const auto& output : operation.GetOOperands()) {
                auto inMemType = input->GetMemoryTypeOriginal();
                auto outMemType = output->GetMemoryTypeOriginal();
                if (inMemType == outMemType) {
                    continue;
                }
                if (!Platform::Instance().GetDie().HasDirectPath(inMemType, outMemType)) {
                    APASS_LOG_ERROR_F(Elements::Tensor,
                                      "OP[%d] has inputTensor[%d] with memoryType %s and "
                                      "outputTensor[%d] with memoryType %s; The path is not reachable.",
                                      operation.GetOpMagic(), input->GetMagic(),
                                      BriefMemoryTypeToString(inMemType).c_str(), output->GetMagic(),
                                      BriefMemoryTypeToString(outMemType).c_str());
                    return FAILED;
                }
            }
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
