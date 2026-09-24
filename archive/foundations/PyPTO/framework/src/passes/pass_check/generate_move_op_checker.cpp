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
 * \file generate_move_op_checker.cpp
 * \brief
 */

#include "generate_move_op_checker.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "GenerateMoveOp"

namespace npu {
namespace tile_fwk {
Status GenerateMoveOpChecker::DoPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "Start Precheck for GenerateMoveOp.");
    auto operations = function.Operations();
    // Check iOperand and oOperand of OP_CONVERT
    for (auto& operation : operations) {
        const auto opcode = operation.GetOpcode();
        bool isValid = true;
        switch (opcode) {
            case Opcode ::OP_CONVERT:
                isValid = ValidConvertOp(operation);
                break;
            case Opcode ::OP_VIEW:
                isValid = ValidViewOp(operation);
                break;
            case Opcode ::OP_ASSEMBLE:
                isValid = ValidAssembleOp(operation);
                break;
            default:
                continue;
        }
        if (!isValid) {
            APASS_LOG_ERROR_F(Elements::Operation, "Operation validation failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status GenerateMoveOpChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "Start Postcheck for GenerateMoveOp.");
    auto operations = function.Operations();
    for (auto& operation : operations) {
        auto op = operation.GetOpcode();
        if (op == Opcode::OP_DUPLICATE || op == Opcode::OP_CONVERT) {
            APASS_LOG_ERROR_F(Elements::Operation, "Operation validation failed: Operation %s[%d] is invalid here!",
                              operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
            return FAILED;
        }
        if (op == Opcode::OP_ASSEMBLE || op == Opcode::OP_VIEW) {
            if (operation.GetIOperands().size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Operation validation failed: Operation %s[%d] has more than one input.",
                                  operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
                return FAILED;
            }
            if (operation.GetOOperands().size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Operation validation failed: Operation %s[%d] has more than one output.",
                                  operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
                return FAILED;
            }
            auto inputMemType = operation.GetIOperands().front()->GetMemoryTypeOriginal();
            auto outputMemType = operation.GetOOperands().front()->GetMemoryTypeOriginal();
            if (inputMemType != outputMemType) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "Operation validation failed: Operation %s[%d] has dismatched memory type. Input memory type:%s. "
                    "Output memory type:%s",
                    operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                    BriefMemoryTypeToString(inputMemType).c_str(), BriefMemoryTypeToString(outputMemType).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

bool GenerateMoveOpChecker::ValidViewOp(const Operation& op) const
{
    // 校验view单输入单输出，指针非空
    if (op.GetOpAttribute().get() == nullptr) {
        APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                          "View op [%d] check failed : Op attribute is null.", op.GetOpMagic());
        return false;
    }
    if (op.GetIOperands().size() != 1) {
        APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                          "View op [%d] check failed : Found more than one input.", op.GetOpMagic());
        return false;
    }
    if (op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                          "View op [%d] check failed : Found more than one output.", op.GetOpMagic());
        return false;
    }
    if (op.GetIOperands().front() == nullptr) {
        APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                          "View op [%d] check failed : Input is null.", op.GetOpMagic());
        return false;
    }
    if (op.GetOOperands().front() == nullptr) {
        APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                          "View op [%d] check failed : Output is null.", op.GetOpMagic());
        return false;
    }
    if (*(op.oOperand[0]->GetConsumers().begin()) == nullptr) {
        APASS_LOG_ERROR_C(OperationErr::OP_PRODUCER_CONSUMER, Elements::Operation,
                          "View op [%d] check failed : Output has null consumer.", op.GetOpMagic());
        return false;
    }
    bool checkViewOut = CheckViewOutTensorMemType(op);
    if (!checkViewOut) {
        return false;
    }
    return true;
}
bool GenerateMoveOpChecker::CheckViewOutTensorMemType(const Operation& op) const
{
    // 校验view输出tensor内存是否合理
    if (op.GetOOperands().front()->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        return true;
    }
    auto consumerOps = op.oOperand[0]->GetConsumers();
    for (auto childOp : consumerOps) {
        if (childOp == nullptr) {
            APASS_LOG_ERROR_C(OperationErr::OP_PRODUCER_CONSUMER, Elements::Operation,
                              "View op [%d] output has null consumers.", op.GetOpMagic());
            return false;
        }
        auto opcode = childOp->GetOpcode();
        const auto& inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
        if (inputsMemType.empty()) {
            continue;
        }
        bool hasDDRinput = std::find(inputsMemType.begin(), inputsMemType.end(), MemoryType::MEM_DEVICE_DDR) !=
                           inputsMemType.end();
        if (opcode == Opcode::OP_RESHAPE || hasDDRinput) {
            continue;
        }
        if (opcode != Opcode::OP_CONVERT) {
            APASS_LOG_ERROR_C(OperationErr::OP_PRODUCER_CONSUMER, Elements::Operation,
                              "View op [%d] consumer %s[%d] does not support DDR input.", op.GetOpMagic(),
                              childOp->GetOpcodeStr().c_str(), childOp->GetOpMagic());
            return false;
        }
        auto convertOpAttribute = dynamic_cast<ConvertOpAttribute*>(childOp->GetOpAttribute().get());
        if (convertOpAttribute == nullptr) {
            APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                              "View op [%d] consumer %s[%d] has no valid ConvertOpAttribute.", op.GetOpMagic(),
                              childOp->GetOpcodeStr().c_str(), childOp->GetOpMagic());
            return false;
        }
        auto convertPath = convertOpAttribute->GetConvertPath();
        if (convertPath.first != MemoryType::MEM_DEVICE_DDR) {
            APASS_LOG_ERROR_C(OperationErr::OP_PRODUCER_CONSUMER, Elements::Operation,
                              "View op [%d] consumer %s[%d] has invalid convert path.", op.GetOpMagic(),
                              childOp->GetOpcodeStr().c_str(), childOp->GetOpMagic());
            return false;
        }
    }
    return true;
}

bool GenerateMoveOpChecker::ValidAssembleOp(const Operation& op) const
{
    // 校验assemble单输入单输出，指针非空
    if (op.GetOpAttribute().get() == nullptr) {
        APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                          "Assemble op [%d] check failed : Op attribute is null.", op.GetOpMagic());
        return false;
    }
    if (op.GetIOperands().size() != 1) {
        APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                          "Assemble op [%d] check failed : Found more than one input.", op.GetOpMagic());
        return false;
    }
    if (op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                          "Assemble op [%d] check failed : Found more than one output.", op.GetOpMagic());
        return false;
    }
    if (op.GetIOperands().front() == nullptr) {
        APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                          "Assemble op [%d] check failed : Input is null.", op.GetOpMagic());
        return false;
    }
    if (op.GetOOperands().front() == nullptr) {
        APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                          "Assemble op [%d] check failed : Output is null.", op.GetOpMagic());
        return false;
    }
    return true;
}

bool GenerateMoveOpChecker::ValidConvertOp(const Operation& op) const
{
    // 校验convert单输入单输出，指针非空，输入输出内存类型不同，且存在DDR类型
    if (op.GetOpAttribute().get() == nullptr) {
        APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                          "Convert op [%d] check failed : Op attribute is null.", op.GetOpMagic());
        return false;
    }
    if (op.GetIOperands().size() != 1) {
        APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                          "Convert op [%d] check failed : Found more than one input.", op.GetOpMagic());
        return false;
    }
    if (op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                          "Convert op [%d] check failed : Found more than one output.", op.GetOpMagic());
        return false;
    }
    if (op.GetIOperands().front() == nullptr) {
        APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                          "Convert op [%d] check failed : Input is null.", op.GetOpMagic());
        return false;
    }
    if (op.GetOOperands().front() == nullptr) {
        APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                          "Convert op [%d] check failed : Output is null.", op.GetOpMagic());
        return false;
    }
    auto inputMemType = op.GetIOperands().front()->GetMemoryTypeOriginal();
    auto outputMemType = op.GetOOperands().front()->GetMemoryTypeOriginal();
    if (inputMemType == outputMemType) {
        APASS_LOG_ERROR_C(
            TensorErr::TENSOR_INVALID_MEMORY_TYPE, Elements::Operation,
            "Convert op [%d] check failed : Op has dismatched memory type. Input memory type:%s. Output memory type:%s",
            op.GetOpMagic(), BriefMemoryTypeToString(inputMemType).c_str(),
            BriefMemoryTypeToString(outputMemType).c_str());
        return false;
    }
    if (op.GetIOperands().front()->GetShape() != op.GetOOperands().front()->GetShape()) {
        APASS_LOG_ERROR_C(TensorErr::TENSOR_SHAPE_MISMATCH, Elements::Operation,
                          "Convert op [%d] check failed : Input and output tensor has different data shape.",
                          op.GetOpMagic());
        return false;
    }
    return true;
}
} // namespace tile_fwk
} // namespace npu
