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

#define MODULE_NAME "Checker"

namespace npu {
namespace tile_fwk {
Status Checker::DoPreCheck(Function& function)
{
    (void)function;
    return SUCCESS;
}

Status Checker::DoPostCheck(Function& function)
{
    (void)function;
    return SUCCESS;
}

Status Checker::DoDefaultEnabledPreCheck(Function& function)
{
    (void)function;
    return SUCCESS;
}

Status Checker::DoDefaultEnabledPostCheck(Function& function)
{
    (void)function;
    return SUCCESS;
}

Status Checker::CheckConsumerProducer(const LogicalTensorPtr& tensor)
{
    for (const auto& producers : tensor->GetProducers()) {
        if (producers == nullptr) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation, "Found null producer in tensor.");
            return FAILED;
        }
    }
    for (const auto& consumers : tensor->GetConsumers()) {
        if (consumers == nullptr) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation, "Found null consumer in tensor.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Checker::CheckValidOp(Function& function)
{
    for (const auto& op : function.Operations().DuplicatedOpList()) {
        if (op == nullptr) {
            APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_STRUCTURE, Elements::Function,
                              "Found null op in function.Operations().");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Checker::CheckOpIOValid(Function& function)
{
    for (const auto& op : function.Operations().DuplicatedOpList()) {
        for (const auto& input : op->iOperand) {
            if (input == nullptr) {
                APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation, "The input of op[%d] is null",
                                  op->opmagic);
                return FAILED;
            }
            if (CheckConsumerProducer(input) != SUCCESS) {
                APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation,
                                  "CheckConsumerProducer for op[%d]'s input failed!", op->opmagic);
                return FAILED;
            }
        }
        for (const auto& output : op->oOperand) {
            if (output == nullptr) {
                APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation, "The output of op[%d] is null",
                                  op->opmagic);
                return FAILED;
            }
            if (CheckConsumerProducer(output) != SUCCESS) {
                APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation,
                                  "CheckConsumerProducer for op[%d]'s output failed!", op->opmagic);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status Checker::CheckCompleteness(Function& function)
{
    if (function.GetIncast().empty()) {
        APASS_LOG_WARN_F(Elements::Function, "The incast of function[%d] is empty.", function.GetFuncMagic());
        return SUCCESS;
    }
    for (const auto& incast : function.GetIncast()) {
        if (incast == nullptr) {
            APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_STRUCTURE, Elements::Function,
                              "The function[%d] contains incast which is null.", function.GetFuncMagic());
            return FAILED;
        }
        if (incast->GetConsumers().empty()) {
            APASS_LOG_WARN_F(Elements::Operation, "The incast[%d] has no consumer.", incast->GetMagic());
            continue;
        }
    }
    if (function.GetOutcast().empty()) {
        APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_STRUCTURE, Elements::Function,
                          "The outcast of function[%d] is empty.", function.GetFuncMagic());
        return FAILED;
    }
    for (const auto& outcast : function.GetOutcast()) {
        if (outcast == nullptr) {
            APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_STRUCTURE, Elements::Function,
                              "The function[%d] contains outcast which is null.", function.GetFuncMagic());
            return FAILED;
        }
        if (outcast->GetProducers().empty()) {
            APASS_LOG_WARN_F(Elements::Operation, "The outcast[%d] has no producer.", outcast->GetMagic());
            continue;
        }
    }
    return SUCCESS;
}

Status Checker::CheckGraphLoop(Function& function)
{
    if (function.GetTotalSubGraphCount() == 0 && !function.OperationLoopCheck()) {
        APASS_LOG_ERROR_C(GraphErr::GRAPH_LOOP_DETECTION, Elements::Operation,
                          "OperationLoopCheck failed, there is a loop in function[%d].", function.GetFuncMagic());
        return FAILED;
    }
    if (!function.LoopCheck().empty()) {
        APASS_LOG_ERROR_C(GraphErr::GRAPH_LOOP_DETECTION, Elements::Function,
                          "Loopcheck failed, there is a loop in function[%d].", function.GetFuncMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status Checker::PublicCheck(Function& function)
{
    if (CheckValidOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckValidOp for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckOpIOValid(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckOpIOValid for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckCompleteness(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckCompleteness for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckGraphLoop(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckGraphLoop for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckLocalTensor(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckLocalTensor for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    return SUCCESS;
}

inline std::unordered_set<Operation*> GetNeedCheckOps(Function& function, Opcode opcode)
{
    std::unordered_set<Operation*> needCheckOps;
    for (const auto& incast : function.GetIncast()) {
        for (auto& consumer : incast->GetConsumers()) {
            if (consumer->GetOpcode() == opcode) {
                needCheckOps.insert(consumer);
            }
        }
    }
    for (const auto& outcast : function.GetOutcast()) {
        for (auto& producer : outcast->GetProducers()) {
            if (producer->GetOpcode() == opcode) {
                needCheckOps.insert(producer);
            }
        }
    }
    return needCheckOps;
}

Status Checker::CheckDynAttrForView(Function& function)
{
    std::unordered_set<Operation*> needCheckViewOps = GetNeedCheckOps(function, Opcode::OP_VIEW);
    for (const auto& op : needCheckViewOps) {
        const int opMagic = op->GetOpMagic();
        const int funcMagic = function.GetFuncMagic();
        auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
        std::vector<SymbolicScalar>& viewFromDynOffset = viewAttr->GetFromDynOffset();
        if (viewFromDynOffset.empty()) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_DYNAMIC_ATTR, Elements::Operation,
                              "CheckDynAttrForView failed, fromDynOffset_ of op[%d] in function[%d] is empty.", opMagic,
                              funcMagic);
            return FAILED;
        }
        std::vector<SymbolicScalar>& viewToDynValidShape = viewAttr->GetToDynValidShape();
        if (viewToDynValidShape.empty()) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_DYNAMIC_ATTR, Elements::Operation,
                              "CheckDynAttrForView failed, toDynValidShape_ of op[%d] in function[%d] is empty.",
                              opMagic, funcMagic);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Checker::CheckToDynOffsetForAssemble(Function& function)
{
    std::unordered_set<Operation*> needCheckAssembleOps = GetNeedCheckOps(function, Opcode::OP_ASSEMBLE);
    for (const auto& op : needCheckAssembleOps) {
        const int opMagic = op->GetOpMagic();
        const int funcMagic = function.GetFuncMagic();
        auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(op->GetOpAttribute());
        std::vector<SymbolicScalar>& assembleToDynOffset = assembleAttr->GetToDynOffset();
        if (assembleToDynOffset.empty()) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_DYNAMIC_ATTR, Elements::Operation,
                              "CheckToDynOffsetForAssemble failed, toDynOffset_ of op[%d] in function[%d] is empty.",
                              opMagic, funcMagic);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Checker::CheckLocalTensor(Function& function)
{
    auto opList = function.Operations().DuplicatedOpList();
    for (const auto& op : opList) {
        for (auto& iOperand : op->GetIOperands()) {
            if (iOperand == nullptr) {
                APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                                  "The iOperand of op[%d][%s] is null", op->GetOpMagic(), op->GetOpcodeStr().c_str());
                return FAILED;
            }
            if (iOperand->GetProducers().empty() &&
                FunctionUtils::GetNodeType(*iOperand, function) != NodeType::INCAST) {
                APASS_LOG_ERROR_C(
                    TensorErr::TENSOR_NULL_POINTER, Elements::Operation,
                    "A locally defined temporary tensor[%d] cannot be used as an input to an operation[%d].",
                    iOperand->GetMagic(), op->GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
