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
 * \file replace_tensor.cpp
 * \brief
 */

#include "replace_tensor.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "tilefwk/error_code.h"
#include "passes/pass_utils/alignment_utils.h"

#define MODULE_NAME "ReplaceTensor"

namespace npu {
namespace tile_fwk {
bool ReplaceTensor::CheckAddrConflict(const Operation& op)
{
    auto tensorIn = op.GetIOperands().front();
    auto tensorOut = op.GetOOperands().front();
    if (tensorIn->GetRawMagic() != tensorOut->GetRawMagic() &&
        tensorIn->GetRawTensor()->memoryId != tensorOut->GetRawTensor()->memoryId) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "%s op[%d] invalid or conflict. tensorIn magic: %d, rawMagic: %d, tensorOut magic: %d, rawMagic: %d",
            op.GetOpcodeStr().c_str(), op.GetOpMagic(), tensorIn->GetMagic(), tensorIn->GetRawMagic(),
            tensorOut->GetMagic(), tensorOut->GetRawMagic());
        return true;
    }
    return false;
}

/*
用于校验assemble节点的输入输出是否存在冲突
注意，若输入由OP_INDEX_OUTCAST构造，则不会出现冲突
*/
bool ReplaceTensor::CheckIndexProducer(const Operation& op)
{
    for (const auto& producer : op.ProducerOps()) {
        if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            return true;
        }
    }
    return false;
}

bool ReplaceTensor::CheckAssembleConflict(const Operation& op)
{
    if (!CheckIndexProducer(op) && CheckAddrConflict(op)) {
        return true;
    }
    return false;
}

/*
用于校验index_outcast节点的输入输出是否存在冲突
注意，若输出后接assemble节点，则不会出现冲突
*/
bool ReplaceTensor::CheckIndexOutcastConflict(const Operation& op, Function& function)
{
    int index = 2;
    auto indexIn = op.GetInputOperand(index);
    auto indexOut = op.GetOOperands().front();
    if (forwardOps.find(op.GetOpMagic()) != forwardOps.end()) {
        if (function.IsFromInCast(indexIn) && function.IsFromOutCast(indexOut)) {
            return false;
        }
    }
    if (backwardOps.find(op.GetOpMagic()) != backwardOps.end()) {
        if (indexIn->GetRawMagic() != indexOut->GetRawMagic()) {
            return true;
        }
    }
    return false;
}

/*
用于校验reshape节点的输入输出是否存在冲突
需要校验的场景：
    shape输入输出的rawtensor除了首轴之外都一致
*/
bool ReplaceTensor::CheckReshapeConflict(const Operation& op, Function& function)
{
    if (op.GetBoolAttribute(OP_ATTR_PREFIX + "isInplace"))
        return false;
    if (forwardOps.find(op.GetOpMagic()) != forwardOps.end()) {
        auto tensorOut = op.GetOOperands().front();
        if (function.IsFromOutCast(tensorOut)) {
            return false;
        }
    }
    if (backwardOps.find(op.GetOpMagic()) != backwardOps.end()) {
        if (CheckAddrConflict(op)) {
            return true;
        }
    }
    return false;
}

/*
用于校验a_mulacc_b节点的输入输出是否存在冲突
*/
bool ReplaceTensor::CheckAMulAccBConflict(const Operation& op)
{
    int index = 2;
    auto tensorIn = op.GetInputOperand(index);
    auto tensorOut = op.GetOOperands().front();
    auto& inOp = *tensorIn->GetProducers().begin();
    auto& outOp = *tensorOut->GetConsumers().begin();
    if (inOp == nullptr && outOp == nullptr) {
        return false;
    }
    if (tensorIn->GetRawMagic() != tensorOut->GetRawMagic()) {
        return true;
    }
    return false;
}

Status ReplaceTensor::InplaceCheck(Function& function)
{
    struct OpValidator {
        std::function<bool(const Operation&)> validate;
        std::function<bool(const Operation&, Function&)> validateWithFunc;
        std::function<bool(size_t)> inputCountValidator;
        std::function<bool(size_t)> outputCountValidator;
    };

    std::unordered_map<Opcode, OpValidator> opValidators = {
        {Opcode::OP_VIEW,
         {[this](const Operation& op) { return this->CheckAddrConflict(op); }, nullptr,
          [](size_t inputCount) { return inputCount == OperandCount::VIEW_INPUT; },
          [](size_t outputCount) { return outputCount == OperandCount::VIEW_OUTPUT; }}},
        {Opcode::OP_ASSEMBLE,
         {[this](const Operation& op) { return this->CheckAssembleConflict(op); }, nullptr,
          [](size_t inputCount) { return inputCount == OperandCount::ASSEMBLE_INPUT; },
          [](size_t outputCount) { return outputCount == OperandCount::ASSEMBLE_OUTPUT; }}},
        {Opcode::OP_INDEX_OUTCAST,
         {nullptr, [this](const Operation& op, Function& func) { return this->CheckIndexOutcastConflict(op, func); },
          [](size_t inputCount) { return inputCount == OperandCount::INDEX_OUTCAST_INPUTS; },
          [](size_t outputCount) { return outputCount == OperandCount::INDEX_OUTCAST_OUTPUT; }}},
        {Opcode::OP_RESHAPE,
         {nullptr, [this](const Operation& op, Function& func) { return this->CheckReshapeConflict(op, func); },
          [](size_t inputCount) { return inputCount == OperandCount::RESHAPE_INPUT; },
          [](size_t outputCount) { return outputCount == OperandCount::RESHAPE_OUTPUT; }}},
        {Opcode::OP_A_MULACC_B,
         {[this](const Operation& op) { return this->CheckAMulAccBConflict(op); }, nullptr,
          [](size_t inputCount) {
              size_t maxInputs = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 ?
                                     OperandCount::A_MULACC_B_MAX_INPUTS_A5 :
                                     OperandCount::A_MULACC_B_MAX_INPUTS;
              return inputCount >= OperandCount::A_MULACC_B_MIN_INPUTS && inputCount <= maxInputs;
          },
          [](size_t outputCount) { return outputCount == OperandCount::A_MULACC_B_OUTPUT; }}},
    };

    for (const auto& op : function.Operations()) {
        auto it = opValidators.find(op.GetOpcode());
        if (it == opValidators.end())
            continue;
        const auto& validator = it->second;
        if (!validator.inputCountValidator(op.GetInputOperandSize()) ||
            !validator.outputCountValidator(op.GetOutputOperandSize()) ||
            (validator.validate && validator.validate(op)) ||
            (validator.validateWithFunc && validator.validateWithFunc(op, function))) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s op[%d] invalid or conflict.", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

bool ReplaceTensor::CheckInplace(const Operation& op)
{
    if (inplaceOpSet.find(op.GetOpcode()) != inplaceOpSet.end()) {
        return true;
    }
    return false;
}

bool ReplaceTensor::HasSameConsecutive(Operation& op)
{
    for (auto& nextOp : op.ConsumerOps()) {
        if (nextOp->GetOpcode() == op.GetOpcode()) {
            return true;
        }
    }
    return false;
}
Status ReplaceTensor::PreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for ReplaceTensor.");
    if (!function.LoopCheck().empty()) {
        APASS_LOG_ERROR_F(Elements::Function,
                          "Loopcheck failed before PreGraph; Please check whether there is a loop.");
        return FAILED;
    }
    for (auto& op : function.Operations()) {
        if (op.GetSubgraphID() == NOT_IN_SUBGRAPH) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] is not partitioned; Please check subGraphIDs. %s",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if ((op.GetOpcode() != Opcode::OP_ASSEMBLE) && (op.GetOpcode() != Opcode::OP_VIEW) &&
            (op.GetOpcode() != Opcode::OP_RESHAPE)) {
            continue;
        }
        if (HasSameConsecutive(op)) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] has the same Opcode child op; Plese check child ops. %s",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto tensorIn = op.GetIOperands().front();
        auto tensorOut = op.GetOOperands().front();
        if (tensorIn->GetMemoryTypeOriginal() != tensorOut->GetMemoryTypeOriginal()) {
            APASS_LOG_ERROR_F(
                Elements::Tensor,
                "unmatched input output memory type for reshape opmagic: %d, input mem type: %s, output mem type: %s; "
                "Please check the input ans output.",
                op.opmagic, MemoryTypeToString(tensorIn->GetMemoryTypeOriginal()).c_str(),
                MemoryTypeToString(tensorOut->GetMemoryTypeOriginal()).c_str());
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for ReplaceTensor success.");
    return SUCCESS;
}

Status ReplaceTensor::PostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for ReplaceTensor.");
    if (InplaceCheck(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InplaceCheck failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for ReplaceTensor success.");
    return SUCCESS;
}

void ReplaceTensor::UniteTensor(Function& function, UnionFind& uf)
{
    for (const auto& op : function.Operations()) {
        if (CheckInplace(op)) {
            if (inplaceOpMap.find(op.GetOpcode()) != inplaceOpMap.end()) {
                for (const auto& pair : inplaceOpMap.at(op.GetOpcode())) {
                    uf.Unite(op.GetInputOperand(pair.first), op.GetOutputOperand(pair.second));
                    APASS_LOG_INFO_F(Elements::Operation, "Unite %s op[%d] iOperand %d and oOperand %d.",
                                     op.GetOpcodeStr().c_str(), op.GetOpMagic(), op.GetIOperands()[0]->GetMagic(),
                                     op.GetOOperands()[0]->GetMagic());
                }
            } else {
                uf.Unite(op.GetIOperands().front(), op.GetOOperands().front());
                APASS_LOG_INFO_F(Elements::Operation, "Unite %s op[%d] iOperand %d and oOperand %d.",
                                 op.GetOpcodeStr().c_str(), op.GetOpMagic(), op.GetIOperands()[0]->GetMagic(),
                                 op.GetOOperands()[0]->GetMagic());
            }
        }
        if (op.HasAttribute(OpAttributeKey::inplaceIdx)) {
            uf.Unite(op.GetIOperands()[op.GetIntAttribute(OpAttributeKey::inplaceIdx)], op.GetOOperands().front());
        }
    }
}

Status ReplaceTensor::FindBaseTensor(Function& function,
                                     const std::unordered_map<LogicalTensorPtr, int>& tensorToOrderIndex,
                                     LogicalTensors& group, LogicalTensorPtr& baseTensor)
{
    for (const auto& curTensor : group) {
        if (function.IsFromInCast(curTensor) || function.IsFromOutCast(curTensor)) {
            if (baseTensor == nullptr) {
                baseTensor = curTensor;
                APASS_LOG_INFO_F(Elements::Tensor, "Set base Tensor %d", curTensor->GetMagic());
            } else if (baseTensor->Symbol() != curTensor->Symbol() &&
                       baseTensor->GetRawTensor()->memoryId != curTensor->GetRawTensor()->memoryId &&
                       baseTensor->tensor->actualRawmagic != curTensor->tensor->actualRawmagic) {
                APASS_LOG_ERROR_F(Elements::Tensor, "baseTensor %d and curTensor %d has conflict.",
                                  baseTensor->GetMagic(), curTensor->GetMagic());
                return FAILED;
            } else if (function.IsFromInCast(curTensor)) {
                baseTensor = curTensor;
                APASS_LOG_INFO_F(Elements::Tensor, "Set base Tensor %d", curTensor->GetMagic());
            }
        }
    }
    if (baseTensor != nullptr) {
        return SUCCESS;
    }
    LogicalTensors boundTensors;
    for (auto& curTensor : group) {
        if (isBoundTensor(curTensor)) {
            boundTensors.push_back(curTensor);
        }
    }
    LogicalTensors baseGroup = boundTensors.empty() ? group : boundTensors;
    baseTensor = baseGroup.front();
    int64_t baseShape = abs(baseTensor->tensor->GetRawDataSize());
    for (auto& curTensor : baseGroup) {
        int64_t curShape = abs(curTensor->tensor->GetRawDataSize());
        if (curShape > baseShape) {
            APASS_LOG_INFO_F(Elements::Tensor, "Replace curTensor %d size %ld to baseTensor %d size %ld.",
                             curTensor->GetMagic(), curShape, baseTensor->GetMagic(), baseShape);
            baseTensor = curTensor;
            baseShape = curShape;
        } else if (curShape == baseShape && tensorToOrderIndex.at(curTensor) < tensorToOrderIndex.at(baseTensor)) {
            APASS_LOG_INFO_F(Elements::Tensor, "Replace curTensor %d idx %d to baseTensor %d idx %d.",
                             curTensor->GetMagic(), tensorToOrderIndex.at(curTensor), baseTensor->GetMagic(),
                             tensorToOrderIndex.at(baseTensor));
            baseTensor = curTensor;
        }
    }
    return SUCCESS;
}

Status ReplaceTensor::ForwardView(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    if (ForUpdateView(op) == FAILED) {
        return FAILED;
    }
    processedOp.insert(op->GetOpMagic());
    op->GetOOperands()[0]->tensor = rootTensor->tensor;
    forwardOps.insert(op->GetOpMagic());
    forRoots.push(op->GetOOperands()[0]);
    if (!function.IsFromOutCast(op->GetOOperands()[0])) {
        function.UpdateLinkMap(op->GetOOperands()[0], op->GetIOperands()[0]);
    }
    return SUCCESS;
}

Status ReplaceTensor::ForwardReshape(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    processedOp.insert(op->GetOpMagic());
    (void)function;
    if (function.IsFromOutCast(op->GetOOperands()[0])) {
        APASS_LOG_INFO_F(Elements::Operation, "OP_RESHAPE %d oOperand is OutCast, Skip inplace.", op->GetOpMagic());
        return SUCCESS;
    }
    op->GetOOperands()[0]->tensor->actualRawmagic = rootTensor->GetRawMagic();
    forwardOps.insert(op->GetOpMagic());
    forRoots.push(op->GetOOperands()[0]);
    return SUCCESS;
}

Status ReplaceTensor::ForwardInplaceOp(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    auto reusePairs = inplaceOpMap.at(op->GetOpcode());
    for (const auto& reusePair : reusePairs) {
        auto inputIdx = reusePair.first;
        auto outputIdx = reusePair.second;
        auto tensorIn = op->GetIOperands()[inputIdx];
        auto tensorOut = op->GetOOperands()[outputIdx];
        if (tensorIn != rootTensor) {
            APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is not same as rootTensor %d.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic(), tensorIn->GetMagic(),
                             rootTensor->GetMagic());
            return SUCCESS;
        }
        processedOp.insert(op->GetOpMagic());
        if (function.IsFromInCast(tensorIn) && function.IsFromOutCast(tensorOut)) {
            APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is incast, tensorOut %d is outcast.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic(), tensorIn->GetMagic(), tensorOut->GetMagic());
            return SUCCESS;
        }
        tensorOut->tensor = tensorIn->tensor;
        forwardOps.insert(op->GetOpMagic());
        forRoots.push(tensorOut);
        tensorOut->UpdateOffset(tensorIn->GetOffset());
    }
    return SUCCESS;
}

Status ReplaceTensor::ForwardViewType(Operation* op, LogicalTensorPtr& rootTensor)
{
    auto viewTypeIn = op->GetIOperands()[0];
    auto viewTypeOut = op->GetOOperands()[0];
    if (viewTypeIn != rootTensor) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_VIEW_TYPE %d rootTensor %d is not same as viewTypeIn %d.",
                          op->GetOpMagic(), rootTensor->GetMagic(), viewTypeIn->GetMagic());
        return FAILED;
    }
    processedOp.insert(op->GetOpMagic());
    viewTypeOut->tensor->actualRawmagic = viewTypeIn->GetRawMagic();
    forwardOps.insert(op->GetOpMagic());
    if (AdjustOffsetAndRawShape(viewTypeIn, viewTypeOut) == FAILED) {
        return FAILED;
    }
    forRoots.push(viewTypeOut);
    return SUCCESS;
}

bool isInplaceAssemble(Operation* op)
{
    auto assembleIn = op->GetIOperands()[0];
    for (auto inOp : assembleIn->GetProducers()) {
        if (inplaceOpSet.find(inOp->GetOpcode()) != inplaceOpSet.end()) {
            return true;
        }
    }
    return false;
}

bool isMultiAssemble(Operation* op)
{
    auto assembleIn = op->GetIOperands()[0];
    for (auto outOp : assembleIn->GetConsumers()) {
        if (outOp->GetOpMagic() != op->GetOpMagic() && outOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
            return true;
        }
    }
    return false;
}

Status ReplaceTensor::ForwardAssemble(Operation* op, LogicalTensorPtr& rootTensor)
{
    auto assembleIn = op->GetIOperands()[0];
    auto assembleOut = op->GetOOperands()[0];
    if (assembleIn != rootTensor) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_ASSEMBLE %d rootTensor %d is not same as viewTypeIn %d.",
                          op->GetOpMagic(), rootTensor->GetMagic(), assembleIn->GetMagic());
        return FAILED;
    }
    if (isInplaceAssemble(op) || isMultiAssemble(op)) {
        auto& inOp = *(assembleIn)->GetProducers().begin();
        processedOp.insert(op->GetOpMagic());
        forRoots.push(assembleOut);
        if (inOp != nullptr && inOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            APASS_LOG_INFO_F(Elements::Operation,
                             "OP_ASSEMBLE %d parentOp is OP_INDEX_OUTCAST %d, skip replace tensor.", op->GetOpMagic(),
                             inOp->GetOpMagic());
            return SUCCESS;
        }
        assembleOut->tensor = assembleIn->tensor;
        forwardOps.insert(op->GetOpMagic());
        for (auto prodOp : assembleOut->GetProducers()) {
            if (prodOp->GetOpMagic() != op->GetOpMagic()) {
                backRoots.push(assembleOut);
                return SUCCESS;
            }
        }
        return SUCCESS;
    } else {
        forRoots.push(assembleOut);
        backRoots.push(assembleOut);
        return SUCCESS;
    }
}

Status ReplaceTensor::ForwardCopyOut(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    auto index = op->GetIntAttribute(OpAttributeKey::inplaceIdx);
    auto inTensor = op->GetIOperands()[index];
    auto outTensor = op->GetOOperands().front();
    if (inTensor != rootTensor) {
        APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is not same as rootTensor %d.",
                         op->GetOpcodeStr().c_str(), op->GetOpMagic(), inTensor->GetMagic(), rootTensor->GetMagic());
        return SUCCESS;
    }
    processedOp.insert(op->GetOpMagic());
    if (function.IsFromInCast(inTensor) && function.IsFromOutCast(outTensor)) {
        APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] input tensor %d is Incast, output tensor %d is OutCast",
                         op->GetOpcodeStr().c_str(), op->GetOpMagic(), inTensor->GetMagic(), outTensor->GetMagic());
        return SUCCESS;
    }
    if (!function.IsFromOutCast(outTensor)) {
        function.UpdateLinkMap(outTensor, inTensor);
    }
    outTensor->tensor = rootTensor->tensor;
    outTensor->UpdateOffset(rootTensor->GetOffset());
    forRoots.push(outTensor);
    return SUCCESS;
}

Status ReplaceTensor::ForwardInputIdx(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    auto index = op->GetIntAttribute(OpAttributeKey::inplaceIdx);
    auto inTensor = op->GetIOperands()[index];
    auto outTensor = op->GetOOperands().front();
    if (inTensor != rootTensor) {
        APASS_LOG_INFO_F(Elements::Operation, "op %s[%d] tensorIn %d is not same as rootTensor %d.",
                         op->GetOpcodeStr().c_str(), op->GetOpMagic(), inTensor->GetMagic(), rootTensor->GetMagic());
        return SUCCESS;
    }
    processedOp.insert(op->GetOpMagic());
    if (!function.IsFromOutCast(outTensor)) {
        function.UpdateLinkMap(outTensor, inTensor);
    }
    outTensor->tensor = rootTensor->tensor;
    outTensor->UpdateOffset(rootTensor->GetOffset());
    forRoots.push(outTensor);
    return SUCCESS;
}

Status ReplaceTensor::BackwardReshape(Operation* op, LogicalTensorPtr& rootTensor)
{
    processedOp.insert(op->GetOpMagic());
    op->GetIOperands()[0]->tensor->actualRawmagic = rootTensor->GetRawMagic();
    backwardOps.insert(op->GetOpMagic());
    backRoots.push(op->GetIOperands()[0]);
    return SUCCESS;
}

Status ReplaceTensor::BackwardInplaceOp(Operation* op, LogicalTensorPtr& rootTensor)
{
    auto reusePairs = inplaceOpMap.at(op->GetOpcode());
    for (const auto& reusePair : reusePairs) {
        auto inputIdx = reusePair.first;
        auto outputIdx = reusePair.second;
        auto tensorIn = op->GetIOperands()[inputIdx];
        auto tensorOut = op->GetOOperands()[outputIdx];
        if (tensorOut != rootTensor) {
            APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is not same as rootTensor %d.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic(), tensorIn->GetMagic(),
                             rootTensor->GetMagic());
            return SUCCESS;
        }
        processedOp.insert(op->GetOpMagic());
        tensorIn->tensor = tensorOut->tensor;
        backwardOps.insert(op->GetOpMagic());
        backRoots.push(tensorIn);
        tensorOut->UpdateOffset(tensorIn->GetOffset());
    }
    return SUCCESS;
}

Status ReplaceTensor::BackwardView(Operation* op, LogicalTensorPtr& rootTensor)
{
    auto viewIn = op->GetIOperands()[0];
    auto viewOut = op->GetOOperands()[0];
    (void)rootTensor;
    processedOp.insert(op->GetOpMagic());
    backRoots.push(viewIn);
    viewIn->tensor = viewOut->tensor;
    backwardOps.insert(op->GetOpMagic());
    return SUCCESS;
}

Status ReplaceTensor::BackwardViewType(Operation* op, LogicalTensorPtr& rootTensor)
{
    auto viewTypeIn = op->GetIOperands()[0];
    auto viewTypeOut = op->GetOOperands()[0];
    if (viewTypeOut != rootTensor) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_VIEW_TYPE %d rootTensor %d is not same as viewTypeOut %d.",
                          op->GetOpMagic(), rootTensor->GetMagic(), viewTypeOut->GetMagic());
        return FAILED;
    }
    processedOp.insert(op->GetOpMagic());
    viewTypeIn->tensor->actualRawmagic = viewTypeOut->GetRawMagic();
    backwardOps.insert(op->GetOpMagic());
    if (AdjustOffsetAndRawShape(viewTypeOut, viewTypeIn) == FAILED) {
        return FAILED;
    }
    backRoots.push(viewTypeIn);
    return SUCCESS;
}

Status ReplaceTensor::BackwardAssemble(Operation* op, LogicalTensorPtr& rootTensor)
{
    auto& inOp = *(op->GetIOperands()[0])->GetProducers().begin();
    backRoots.push(op->GetIOperands()[0]);
    processedOp.insert(op->GetOpMagic());
    if (inOp != nullptr && inOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
        APASS_LOG_INFO_F(Elements::Operation, "OP_ASSEMBLE %d parent op is OP_INDEX_OUTCAST %d, skip inplace.",
                         op->GetOpMagic(), inOp->GetOpMagic());
        return SUCCESS;
    }
    if (BackUpdateAssemble(op) == FAILED) {
        return FAILED;
    }
    op->GetIOperands()[0]->tensor = rootTensor->tensor;
    if (FoldL0C2UBCopyOffset(op) == FAILED) {
        return FAILED;
    }
    backwardOps.insert(op->GetOpMagic());
    if (op->GetIOperands()[0]->GetConsumers().size() > 1) {
        forRoots.push(op->GetIOperands()[0]);
        for (auto& consumer : op->GetIOperands()[0]->GetConsumers()) {
            if (consumer->GetOpcode() == Opcode::OP_COPY_IN) {
                if (UpdateCopyInAttr(consumer) == FAILED) {
                    APASS_LOG_ERROR_F(Elements::Operation, "Update copyIn[%d] attr failed.", consumer->GetOpMagic());
                    return FAILED;
                }
            }
        }
    }
    return SUCCESS;
}

Status ReplaceTensor::BackwardInputIdx(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    auto index = op->GetIntAttribute(OpAttributeKey::inplaceIdx);
    auto inTensor = op->GetIOperands()[index];
    auto outTensor = op->GetOOperands().front();
    if (outTensor != rootTensor) {
        APASS_LOG_INFO_F(Elements::Operation, "op %s[%d] tensorIn %d is not same as rootTensor %d.",
                         op->GetOpcodeStr().c_str(), op->GetOpMagic(), inTensor->GetMagic(), rootTensor->GetMagic());
        return SUCCESS;
    }
    processedOp.insert(op->GetOpMagic());
    if (!function.IsFromOutCast(outTensor)) {
        function.UpdateLinkMap(outTensor, inTensor);
    }
    inTensor->tensor = rootTensor->tensor;
    inTensor->UpdateOffset(rootTensor->GetOffset());
    forRoots.push(inTensor);
    return SUCCESS;
}

Status ReplaceTensor::ForwardProcess(Function& function)
{
    while (!forRoots.empty()) {
        auto rootTensor = forRoots.front();
        forRoots.pop();
        for (auto& consumerOp : rootTensor->GetConsumers()) {
            if (processedOp.find(consumerOp->GetOpMagic()) != processedOp.end()) {
                continue;
            }
            if (consumerOp->GetOpcode() == Opcode::OP_VIEW) {
                if (ForwardView(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (ForwardAssemble(consumerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_RESHAPE) {
                if (ForwardReshape(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_VIEW_TYPE) {
                if (ForwardViewType(consumerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (inplaceOpMap.find(consumerOp->GetOpcode()) != inplaceOpMap.end()) {
                if (ForwardInplaceOp(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_COPY_OUT &&
                       consumerOp->HasAttribute(OpAttributeKey::inplaceIdx)) {
                if (ForwardCopyOut(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if ((consumerOp->GetOpcode() == Opcode::OP_INDEX_PUT ||
                        consumerOp->GetOpcode() == Opcode::OP_INDEX_ADD) &&
                       consumerOp->HasAttribute(OpAttributeKey::inplaceIdx)) {
                if (ForwardInputIdx(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else {
                continue;
            }
        }
    }
    return SUCCESS;
}

Status ReplaceTensor::BackwardProcess(Function& function)
{
    while (!backRoots.empty()) {
        auto rootTensor = backRoots.front();
        backRoots.pop();
        for (auto& producerOp : rootTensor->GetProducers()) {
            if (processedOp.find(producerOp->GetOpMagic()) != processedOp.end()) {
                continue;
            }
            if (producerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (BackwardAssemble(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (producerOp->GetOpcode() == Opcode::OP_VIEW) {
                if (BackwardView(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (producerOp->GetOpcode() == Opcode::OP_RESHAPE) {
                if (BackwardReshape(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (inplaceOpMap.find(producerOp->GetOpcode()) != inplaceOpMap.end()) {
                if (BackwardInplaceOp(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (producerOp->GetOpcode() == Opcode::OP_VIEW_TYPE) {
                if (BackwardViewType(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if ((producerOp->GetOpcode() == Opcode::OP_INDEX_PUT ||
                        producerOp->GetOpcode() == Opcode::OP_INDEX_ADD) &&
                       producerOp->HasAttribute(OpAttributeKey::inplaceIdx)) {
                if (BackwardInputIdx(producerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else {
                continue;
            }
        }
    }
    return SUCCESS;
}

LogicalTensorPtr ReplaceTensor::FindReplaceSource(Function& function, Operation& op,
                                                  std::unordered_map<Operation*, LogicalTensorPtr>& visited)
{
    if (visited.count(&op) > 0) {
        return visited.at(&op);
    }
    auto inplaceIdx = op.GetIntAttribute(OpAttributeKey::inplaceIdx);
    ASSERT(OperationErr::OP_INVALID_OPERAND_COUNT,
           inplaceIdx >= 0 && inplaceIdx < static_cast<int>(op.GetIOperands().size()));
    auto inplaceIOperand = op.GetInputOperand(inplaceIdx);
    LogicalTensorPtr res = nullptr;
    for (auto producer : inplaceIOperand->GetProducers()) {
        if (!producer->HasAttribute(OpAttributeKey::inplaceIdx)) {
            continue;
        }
        auto tmp = FindReplaceSource(function, *producer, visited);
        if (res == nullptr) {
            res = tmp;
        } else {
            ASSERT(OperationErr::OP_SPECIAL_CONSTRAINT, res == tmp); // inplace路径应总是交汇于同一起点
        }
    }
    if (res == nullptr) {
        res = inplaceIOperand; // 向前没有inplace了，自己就是起点
    }
    visited.emplace(&op, res);
    return res;
}

Status ReplaceTensor::RefactorViewConnectForReplace(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "===> Start RefactorViewConnectForInplace.");
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        if (op.GetInputOperand(0)->GetRawTensor() == op.GetOutputOperand(0)->GetRawTensor()) {
            op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
        }
    }
    std::unordered_map<Operation*, LogicalTensorPtr> visited;
    for (Operation& op : function.Operations()) {
        if (!op.HasAttribute(OpAttributeKey::inplaceIdx) || visited.count(&op) > 0) {
            continue;
        }
        FindReplaceSource(function, op, visited);
    }

    for (auto& [op, srcTensor] : visited) {
        if (op->GetOpcode() != Opcode::OP_VIEW) { // 仅重构View连接
            continue;
        }
        auto inplaceIdx = op->GetIntAttribute(OpAttributeKey::inplaceIdx);
        ASSERT(OperationErr::OP_SPECIAL_CONSTRAINT, inplaceIdx == 0);
        auto iOperand = op->GetInputOperand(inplaceIdx);
        auto oOperand = op->GetOutputOperand(0);
        if (iOperand == srcTensor) { // 开头的VIEW不需要插入NOP来控制顺序
            continue;
        }
        bool hasNonInplaceConsumer = false;
        for (auto consumer : oOperand->GetConsumers()) {
            if (consumer->GetOpcode() != Opcode::OP_NOP && !consumer->HasAttribute(OpAttributeKey::inplaceIdx)) {
                hasNonInplaceConsumer = true;
                break;
            }
        }
        if (hasNonInplaceConsumer) {
            continue;
        }
        ASSERT(TensorErr::TENSOR_SHAPE_MISMATCH, iOperand->GetRawTensor() == srcTensor->GetRawTensor());
        ASSERT(TensorErr::TENSOR_SHAPE_MISMATCH, oOperand->GetRawTensor() == srcTensor->GetRawTensor());
        op->ReplaceIOperand(0, srcTensor);
        auto nopOutput = irBuilder_.CreateTensorVar(srcTensor->GetRawTensor(), Offset(srcTensor->GetOffset().size()),
                                                    srcTensor->GetShape(), std::vector<SymbolicScalar>{});
        nopOutput->SetMemoryTypeBoth(oOperand->GetMemoryTypeOriginal());
        auto& nop = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_NOP, {iOperand, oOperand}, {nopOutput});
        nop.SetAttribute(OpAttributeKey::inplaceIdx, 0);
        nop.UpdateSubgraphID(op->GetSubgraphID());
        auto consumers = oOperand->GetConsumers();
        for (auto consumer : consumers) {
            if (consumer->GetOpcode() == Opcode::OP_NOP || !consumer->HasAttribute(OpAttributeKey::inplaceIdx)) {
                continue;
            }
            consumer->ReplaceIOperand(consumer->GetIntAttribute(OpAttributeKey::inplaceIdx), nopOutput);
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> End RefactorViewConnectForInplace.");
    return SUCCESS;
}

void ReplaceTensor::ProcessHubAssembleOp(Function& function, Operation& hubOp, Operation& assembleOp,
                                         std::shared_ptr<LogicalTensor> hubInput,
                                         std::shared_ptr<LogicalTensor> hubOutput)
{
    auto assembleInput = assembleOp.GetIOperands()[0];
    auto assembleOutput = assembleOp.GetOOperands()[0];
    if (assembleInput.get() != hubOutput.get()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Assemble input[%d] is not HUB output[%d], chain may be broken",
                         assembleInput->GetMagic(), hubOutput->GetMagic());
        return;
    }
    bool isExactOutcast = false;
    auto outcasts = function.GetOutcast();
    for (auto& outcast : outcasts) {
        if (outcast.get() == assembleOutput.get()) {
            isExactOutcast = true;
            break;
        }
    }
    if (!isExactOutcast) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "Assemble[%d] output is not exact outcast, skip HUB memory reuse processing.",
                         assembleOp.GetOpMagic());
        return;
    }
    APASS_LOG_INFO_F(Elements::Operation,
                     "Found exact HUB-ASSEMBLE-OUTCAST chain: HUB[%d] -> ASSEMBLE[%d] -> OUTCAST[%d]",
                     hubOp.GetOpMagic(), assembleOp.GetOpMagic(), assembleOutput->GetMagic());
    auto hubInputMemType = hubInput->GetMemoryTypeOriginal();
    auto hubOutputMemType = hubOutput->GetMemoryTypeOriginal();
    auto assembleOutputMemType = assembleOutput->GetMemoryTypeOriginal();
    if (hubInputMemType != hubOutputMemType || hubInputMemType != assembleOutputMemType) {
        APASS_LOG_WARN_F(Elements::Tensor, "Memory type mismatch: HUB input=%d, HUB output=%d, ASSEMBLE output=%d",
                         hubInputMemType, hubOutputMemType, assembleOutputMemType);
        return;
    }
    hubInput->tensor = assembleOutput->tensor;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(assembleOp.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_WARN_F(Elements::Operation, "HUB assemble op %d attribute is nullptr, skip HUB memory reuse.",
                         assembleOp.GetOpMagic());
        return;
    }
    hubInput->UpdateOffset(assembleOpAttribute->GetToTensorOffset());
    hubOutput->tensor = assembleOutput->tensor;
    hubOutput->UpdateOffset(assembleOpAttribute->GetToTensorOffset());
    APASS_LOG_INFO_F(Elements::Tensor, "Complete memory reuse established: all tensors share HUB input[%d] memory",
                     hubInput->GetMagic());
}

Status ReplaceTensor::ProcessHubOp(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_HUB) {
            continue;
        }
        auto hubInput = op.GetIOperands()[0];  // HUB 的输入 tensor
        auto hubOutput = op.GetOOperands()[0]; // HUB 的输出 tensor
        for (auto consumerOp : hubOutput->GetConsumers()) {
            if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                ProcessHubAssembleOp(function, op, *consumerOp, hubInput, hubOutput);
            }
        }
        for (auto producerOp : hubInput->GetProducers()) {
            if (!OpcodeManager::Inst().IsCopyOut(producerOp->GetOpcode())) {
                continue;
            }
            auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(producerOp->GetOpAttribute());
            if (copyAttr == nullptr) {
                APASS_LOG_INFO_F(Elements::Operation, "Copy Op %d Attribute is nullptr.", producerOp->GetOpMagic());
                continue;
            }
            auto attrOffset = copyAttr->GetToOffset(); // OpImm
            auto tensorOffset = OpImmediate::Specified(hubInput->GetTensorOffset());
            std::vector<OpImmediate> newOffset;
            for (size_t i = 0; i < attrOffset.size(); i++) {
                newOffset.push_back(attrOffset[i] + tensorOffset[i]);
            }
            copyAttr->SetToOffset(newOffset);
            copyAttr->SetRawShape(OpImmediate::Specified(hubInput->tensor->GetDynRawShape()));
        }
    }
    return SUCCESS;
}

std::unordered_map<LogicalTensorPtr, int> ReplaceTensor::BuildTensorOrderIndexMap(Function& function)
{
    std::unordered_map<LogicalTensorPtr, int> tensorToOrderIndex;
    int index = 0;
    for (const auto& op : function.Operations()) {
        for (const auto& inTensor : op.GetIOperands()) {
            if (!tensorToOrderIndex.count(inTensor)) {
                tensorToOrderIndex[inTensor] = index++;
            }
        }
        for (const auto& outTensor : op.GetOOperands()) {
            if (!tensorToOrderIndex.count(outTensor)) {
                tensorToOrderIndex[outTensor] = index++;
            }
        }
    }
    return tensorToOrderIndex;
}

/**
 * @brief 为 UB 内存类型的输入插入拷贝序列 (UB → DDR → UB)
 */
Status ReplaceTensor::InsertCopyUBOp(Function& function, Operation* needInsertCopyAssOp, LogicalTensorPtr& input)
{
    auto copyShape = input->GetShape();
    auto copyRawShape = input->tensor->GetDynRawShape();
    auto copyDynShape = input->GetDynValidShape();
    Offset offset(copyShape.size(), 0);
    auto copyOutOutputPtr = irBuilder_.CreateTensorVar(input->Datatype(), copyShape, std::vector<SymbolicScalar>{});
    copyOutOutputPtr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    auto& copyOutOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_OUT, {input}, {copyOutOutputPtr},
        [&input, &offset, &copyShape, &copyRawShape, &copyDynShape](Operation& op) {
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                input->GetMemoryTypeOriginal(), OpImmediate::Specified(offset), OpImmediate::Specified(copyShape),
                OpImmediate::Specified(copyRawShape), OpImmediate::Specified(copyDynShape)));
        });
    copyOutOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    auto copyInOutputPtr = irBuilder_.CreateTensorVar(input->Datatype(), copyShape, std::vector<SymbolicScalar>{});
    copyInOutputPtr->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto& copyInOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_IN, {copyOutOutputPtr}, {copyInOutputPtr},
        [&offset, &input, &copyShape, &copyRawShape, &copyDynShape](Operation& op) {
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(offset), input->GetMemoryTypeOriginal(), OpImmediate::Specified(copyShape),
                OpImmediate::Specified(copyRawShape), OpImmediate::Specified(copyDynShape)));
        });
    copyInOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    needInsertCopyAssOp->ReplaceInput(copyInOutputPtr, input);
    return SUCCESS;
}

/**
 * @brief 为 DDR 内存类型的输入插入拷贝序列 (DDR → UB → DDR)
 */
Status ReplaceTensor::InsertCopyDDROp(Function& function, Operation* needInsertCopyAssOp, LogicalTensorPtr& input)
{
    auto copyShape = input->GetShape();
    auto copyRawShape = input->tensor->GetDynRawShape();
    auto copyDynShape = input->GetDynValidShape();
    Offset offset(copyShape.size(), 0);
    Offset inOffset = input->GetOffset();
    auto& inOp = *(input)->GetProducers().begin();
    if (needInsertCopyAssOp->GetOpcode() == Opcode::OP_RESHAPE && inOp->GetOpcode() == Opcode::OP_VIEW) {
        auto viewOpAttr = std::dynamic_pointer_cast<ViewOpAttribute>(inOp->GetOpAttribute());
        inOffset = viewOpAttr->GetFrom();
    }
    auto copyInOutputPtr = irBuilder_.CreateTensorVar(input->Datatype(), copyShape, std::vector<SymbolicScalar>{});
    copyInOutputPtr->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    const int UB_SIZE_THRESHOLD = static_cast<int>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB));
    auto memType = copyInOutputPtr->GetMemoryTypeOriginal();
    if ((memType == MemoryType::MEM_UB) && (copyInOutputPtr->GetDataSize() > UB_SIZE_THRESHOLD)) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "Tensor [%d] can not copy to UB, tensor size [%ld] exceeds the UB size [%d] limit.",
                          input->magic, input->GetDataSize(), UB_SIZE_THRESHOLD);
        return FAILED;
    }

    // 为copy到Ub的Tensor进行32B对齐
    AlignmentUtils::ProcessLastDim32BAlignedOnUB(copyInOutputPtr);
    auto& copyInOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_IN, {input}, {copyInOutputPtr},
        [&inOffset, &copyShape, &copyRawShape, &copyDynShape](Operation& op) {
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(inOffset), MemoryType::MEM_UB, OpImmediate::Specified(copyShape),
                OpImmediate::Specified(copyRawShape), OpImmediate::Specified(copyDynShape)));
        });
    copyInOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    auto copyOutOutputPtr = irBuilder_.CreateTensorVar(input->Datatype(), copyShape, std::vector<SymbolicScalar>{});
    copyOutOutputPtr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto& copyOutOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_OUT, {copyInOutputPtr}, {copyOutOutputPtr},
        [&offset, &copyShape, &copyRawShape, &copyDynShape](Operation& op) {
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                MemoryType::MEM_UB, OpImmediate::Specified(offset), OpImmediate::Specified(copyShape),
                OpImmediate::Specified(copyRawShape), OpImmediate::Specified(copyDynShape)));
        });
    copyOutOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    needInsertCopyAssOp->ReplaceInput(copyOutOutputPtr, input);
    return SUCCESS;
}

/**
 * @brief 递归查找需要插入拷贝的 ASSEMBLE 操作
 */
Status ReplaceTensor::FindNeedToCopyAssemble(std::unordered_set<Operation*>& needInsertCopyAssOps,
                                             std::unordered_set<int>& visitedAssOps, Operation& op, Function& function)
{
    visitedAssOps.insert(op.GetOpMagic());
    auto assembleIn = op.GetIOperands()[0];
    auto assembleOut = op.GetOOperands()[0];
    auto producers = assembleIn->GetProducers();
    auto& inOp = *(assembleIn)->GetProducers().begin();
    if ((!producers.empty()) && (((*producers.begin())->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT) ||
                                 (*producers.begin())->GetOpcode() == Opcode::OP_SHMEM_WAIT_UNTIL)) {
        return FAILED;
    }
    const int UB_SIZE_THRESHOLD = static_cast<int>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB));
    auto inProducerOps = inOp->ProducerOps();
    bool allProducerAssemble = !inProducerOps.empty();

    // BMM优化场景：Reshape前驱全为Assemble时不插copy。
    for (const auto& producer : inProducerOps) {
        allProducerAssemble = allProducerAssemble && producer->GetOpcode() == Opcode::OP_ASSEMBLE;
    }

    if (inOp->GetOpcode() == Opcode::OP_RESHAPE && !allProducerAssemble && !function.IsFromOutCast(assembleOut) &&
        !isBoundTensor(assembleOut) && op.GetIOperands()[0]->tensor->GetRawDataSize() <= UB_SIZE_THRESHOLD) {
        needInsertCopyAssOps.insert(&op);
        return SUCCESS;
    }
    auto consumers = assembleIn->GetConsumers();
    bool sameAssembleOut = true;
    for (const auto& con : consumers) {
        if (con->GetOOperands()[0]->GetMagic() != op.GetOOperands()[0]->GetMagic()) {
            sameAssembleOut = false;
            break;
        }
    }
    if (!sameAssembleOut) {
        for (const auto& con : consumers) {
            if (con->GetOpcode() == Opcode::OP_ASSEMBLE) {
                visitedAssOps.insert(con->GetOpMagic());
                needInsertCopyAssOps.insert(con);
            }
        }
    }
    return SUCCESS;
}

bool ReplaceTensor::isBoundTensor(LogicalTensorPtr& curTensor)
{
    std::set<int> boundTensorIDs;
    for (auto& inOp : curTensor->GetProducers()) {
        boundTensorIDs.insert(inOp->GetSubgraphID());
    }
    for (auto& outOp : curTensor->GetConsumers()) {
        boundTensorIDs.insert(outOp->GetSubgraphID());
    }
    if (boundTensorIDs.size() > 1) {
        return true;
    }
    return false;
}

Status ReplaceTensor::FindNeedToCopyReshape(std::unordered_set<Operation*>& needInsertCopyAssOps,
                                            std::unordered_set<int>& visitedReshapeOps, Operation& op,
                                            Function& function)
{
    visitedReshapeOps.insert(op.GetOpMagic());
    if (op.GetIOperands()[0]->tensor->GetRawShapeSize() > 0 && op.GetOOperands()[0]->tensor->GetRawShapeSize() > 0 &&
        op.GetIOperands()[0]->tensor->GetRawShapeSize() != op.GetOOperands()[0]->tensor->GetRawShapeSize()) {
        needInsertCopyAssOps.insert(&op);
        return SUCCESS;
    }
    auto producerOps = op.ProducerOps();
    auto consumerOps = op.ConsumerOps();
    for (auto producesOp : producerOps) {
        if (producesOp->GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        if (isBoundTensor(op.GetOOperands().front()) && !isBoundTensor(producesOp->GetIOperands().front()) &&
            !function.IsFromInCast(producesOp->GetIOperands().front()) &&
            producesOp->GetIOperands().front()->tensor->GetRawShapeSize() !=
                op.GetOOperands().front()->tensor->GetRawShapeSize()) {
            needInsertCopyAssOps.insert(&op);
            continue;
        }
        for (auto consumerOp : consumerOps) {
            if (inplaceOpSet.find(consumerOp->GetOpcode()) != inplaceOpSet.end()) {
                needInsertCopyAssOps.insert(&op);
            }
        }
    }
    return SUCCESS;
}

/**
 * @brief 遍历所有 ASSEMBLE 操作，为需要拷贝的操作插入拷贝序列，避免多个 ASSEMBLE 操作共享同一个输入导致的内存冲突
 * Tensor1 ---> Assemble ---> Tensor2
 *         ---> Assemble ---> Tensor3
 *         ---> Assemble ---> Tensor4

 * Tensor1 ---> View ---> Reshape ---> OP(非CopyIn) ---> Tensor2

 * Tensor1 ---> Reshape ---> Assemble ---> Tensor2(可能造成CopyOut+Reshape+Assemble的一些场景性能损失)
 */
Status ReplaceTensor::InsertNeedCopy(Function& function)
{
    std::unordered_set<int> visitedAssOps;
    std::unordered_set<int> visitedReshapeOps;
    std::unordered_set<Operation*> needInsertCopyAssOps;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && (!visitedAssOps.count(op.GetOpMagic()))) {
            FindNeedToCopyAssemble(needInsertCopyAssOps, visitedAssOps, op, function);
        }
        if (op.GetOpcode() == Opcode::OP_RESHAPE && (!visitedReshapeOps.count(op.GetOpMagic()))) {
            FindNeedToCopyReshape(needInsertCopyAssOps, visitedReshapeOps, op, function);
        }
    }
    std::vector<Operation*> sortedOps(needInsertCopyAssOps.begin(), needInsertCopyAssOps.end());
    std::sort(sortedOps.begin(), sortedOps.end(),
              [](const Operation* a, const Operation* b) { return a->GetOpMagic() < b->GetOpMagic(); });
    for (auto& needInsertCopyAssOp : sortedOps) {
        auto input = needInsertCopyAssOp->GetIOperands()[0];
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            InsertCopyUBOp(function, needInsertCopyAssOp, input);
        } else if (input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            if (InsertCopyDDROp(function, needInsertCopyAssOp, input) == FAILED) {
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status ReplaceTensor::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "===> Start ReplaceTensor.");
    if (InsertNeedCopy(function) == FAILED) {
        return FAILED;
    }
    auto tensorToOrderIndex = BuildTensorOrderIndexMap(function);
    UnionFind uf(tensorToOrderIndex);
    UniteTensor(function, uf);
    std::vector<LogicalTensors> tensorGroups = uf.GetGroups();
    for (auto& group : tensorGroups) {
        LogicalTensorPtr baseTensor = nullptr;
        if (group.size() == 1) {
            continue;
        }
        if (FindBaseTensor(function, tensorToOrderIndex, group, baseTensor) == FAILED || baseTensor == nullptr) {
            return FAILED;
        }
        backRoots.push(baseTensor);
        forRoots.push(baseTensor);
        while (!forRoots.empty() || !backRoots.empty()) {
            if (BackwardProcess(function) == FAILED || ForwardProcess(function) == FAILED) {
                return FAILED;
            }
        }
    }
    if (RefactorViewConnectForReplace(function) == FAILED) {
        return FAILED;
    }
    if (ProcessHubOp(function) == FAILED) {
        return FAILED;
    }
    if (MarkTensorAsPartialMem(function) == FAILED) {
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> End ReplaceTensor.");
    return SUCCESS;
}

Status ReplaceTensor::AdjustOffsetAndRawShape(LogicalTensorPtr& fromView, LogicalTensorPtr& toView) const
{
    auto fromType = fromView->tensor->datatype;
    auto toType = toView->tensor->datatype;
    auto inEntry = viewTypeTable.find(fromType);
    auto outEntry = viewTypeTable.find(toType);
    if (inEntry == viewTypeTable.end() || outEntry == viewTypeTable.end()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "ViewType Input Tensor OR Output Tensor DataType is not in viewType, Please check it!");
        return FAILED;
    }
    int inSize = inEntry->second;
    int outSize = outEntry->second;
    int ratio = inSize > outSize ? inSize / outSize : outSize / inSize;
    bool isExpand = inSize > outSize;
    std::vector<int64_t> fromOffset = fromView->GetOffset();
    std::vector<int64_t> toOffset(fromOffset.size(), 0);
    std::vector<int64_t> inShape = fromView->GetRawTensor()->rawshape;
    std::vector<int64_t> outShape(inShape.size(), 0);
    for (size_t i = 0; i < fromOffset.size(); ++i) {
        if (i != fromOffset.size() - 1) {
            toOffset[i] = fromOffset[i];
            outShape[i] = inShape[i];
            continue;
        }
        if (isExpand) {
            toOffset[i] = fromOffset[i] * ratio;
            outShape[i] = inShape[i] * ratio;
        } else {
            if (fromOffset[i] % ratio != 0 || inShape[i] % ratio != 0) {
                APASS_LOG_ERROR_F(Elements::Operation, "ViewType Offset is not Even.");
                return FAILED;
            }
            toOffset[i] = fromOffset[i] / ratio;
            outShape[i] = inShape[i] / ratio;
        }
    }
    toView->UpdateOffset(toOffset);
    toView->GetRawTensor()->rawshape = outShape;
    return SUCCESS;
}

Status ReplaceTensor::ForUpdateView(Operation* op)
{
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
    auto viewIn = op->GetIOperands()[0];
    auto viewOut = op->GetOOperands()[0];
    std::vector<int64_t> inputOffset = viewIn->GetOffset();
    if (viewAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "ReplaceTensor::ForUpdateView: View op %d Attribute is nullptr.",
                          op->GetOpMagic());
        return FAILED;
    }
    std::vector<int64_t> viewOpOffset = viewAttr->GetFrom();
    auto inputDynOffset = viewIn->GetDynOffset();
    if (inputDynOffset.empty()) {
        inputDynOffset = std::vector<SymbolicScalar>(inputOffset.size(), 0);
    }
    auto attrDynOffset = viewAttr->GetFromDynOffset();
    std::vector<SymbolicScalar> outTensorOffset;
    for (size_t i = 0; i < inputOffset.size(); i++) {
        viewOpOffset[i] = inputOffset[i] + viewOpOffset[i];
        if (attrDynOffset.size() == inputOffset.size()) {
            attrDynOffset[i] = inputDynOffset[i] + attrDynOffset[i];
        }
    }
    viewAttr->SetFromOffset(viewOpOffset, viewAttr->GetFromDynOffset());
    TensorOffset newOffset(viewOpOffset, attrDynOffset);
    viewOut->UpdateOffset(newOffset);
    return SUCCESS;
}

std::vector<OpImmediate> ReplaceTensor::SumOffsetForCopyIn(const std::vector<OpImmediate> offset1,
                                                           const std::vector<OpImmediate> offset2)
{
    std::vector<OpImmediate> res;
    for (size_t i = 0; i < offset1.size(); i++) {
        res.push_back(offset1[i] + offset2[i]);
    }
    return res;
}

Status ReplaceTensor::UpdateCopyInAttr(Operation* copyInOp)
{
    auto input = copyInOp->GetIOperands()[0];
    auto copyInOpAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyInOp->GetOpAttribute());
    if (copyInOpAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "CopyInOp[%d] don not have attr.", copyInOp->GetOpMagic());
        return FAILED;
    } else {
        std::vector<OpImmediate> inputOffset;
        if (input->GetDynOffset().empty()) {
            inputOffset = OpImmediate::Specified(input->GetOffset());
        } else {
            inputOffset = OpImmediate::Specified(input->GetDynOffset());
        }
        std::vector<OpImmediate> oldFromOffset = copyInOpAttr->GetFromOffset();
        if (!inputOffset.empty() && !oldFromOffset.empty() && (inputOffset.size() == oldFromOffset.size())) {
            copyInOpAttr->SetFromOffset(SumOffsetForCopyIn(inputOffset, oldFromOffset));
        }
        copyInOpAttr->SetRawShape(OpImmediate::Specified(input->tensor->GetDynRawShape()));
    }
    return SUCCESS;
}

Status ReplaceTensor::BackUpdateAssemble(Operation* op)
{
    auto assembleIn = op->GetIOperands()[0];
    auto assembleOut = op->GetOOperands()[0];
    auto assAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op->GetOpAttribute());
    if (assAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "ReplaceTensor::BackUpdateAssemble: Assemble op %d Attribute is nullptr.", op->GetOpMagic());
        return FAILED;
    }
    std::vector<int64_t> assOffset = assAttr->GetToOffset();
    std::vector<int64_t> outOffset = assembleOut->GetOffset();
    auto outDynOffset = assembleOut->GetDynOffset();
    auto assDynOffset = assAttr->GetToDynOffset();
    std::vector<SymbolicScalar> inTensorOffset;
    if (outDynOffset.empty()) {
        outDynOffset = std::vector<SymbolicScalar>(outOffset.size(), 0);
    }
    for (size_t i = 0; i < outOffset.size(); i++) {
        assOffset[i] = outOffset[i] + assOffset[i];
        if (assDynOffset.size() == outOffset.size()) {
            assDynOffset[i] = outDynOffset[i] + assDynOffset[i];
        }
    }
    assAttr->SetToOffset(assOffset, assAttr->GetToDynOffset());
    TensorOffset newOffset(assOffset, assDynOffset);
    assembleIn->UpdateOffset(newOffset);
    return SUCCESS;
}

// L0C2UB 直写折叠：BackUpdateAssemble 已把 assemble 目的偏移刷新到输入 tensor 上，
// 若该输入唯一 producer 是 L0C2UB copy：目的偏移携带非零值且 copy 源偏移可证明为立即零时，
// 把目的偏移落入 copy 的 toOffset 并改为 INSERT；源与目的同时携带偏移时报错
Status ReplaceTensor::FoldL0C2UBCopyOffset(Operation* op)
{
    auto assembleIn = op->GetIOperands()[0];
    const auto& producers = assembleIn->GetProducers();
    if (producers.size() != 1) {
        return SUCCESS;
    }
    auto copyOp = *producers.begin();
    if (copyOp->GetOpcode() != Opcode::OP_L0C_COPY_UB) {
        return SUCCESS;
    }
    // copy 输出被多个 consumer 使用时不折叠，避免不同 assemble 的 toOffset 互相覆盖
    if (assembleIn->GetConsumers().size() != 1) {
        return SUCCESS;
    }
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp->GetOpAttribute());
    if (copyAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "ReplaceTensor::FoldL0C2UBCopyOffset: copy op %d attr is nullptr.",
                          copyOp->GetOpMagic());
        return FAILED;
    }
    bool dstHasOffset = false;
    const auto& statOffset = assembleIn->GetOffset();
    const auto& dynOffset = assembleIn->GetDynOffset();
    if (!dynOffset.empty()) {
        dstHasOffset = std::any_of(dynOffset.begin(), dynOffset.end(), [](const SymbolicScalar& scalar) {
            return !scalar.ConcreteValid() || scalar.Concrete() != 0;
        });
    } else {
        dstHasOffset = std::any_of(statOffset.begin(), statOffset.end(), [](int64_t value) { return value != 0; });
    }
    if (!dstHasOffset) {
        return SUCCESS;
    }
    // 源偏移携带判定：非立即零（参数、符号或非零立即数）均视为携带
    bool srcHasOffset = false;
    for (const auto& offset : copyAttr->GetFromOffset()) {
        if (!offset.IsSpecified() || !offset.GetSpecifiedValue().IsImmediate() ||
            offset.GetSpecifiedValue().Raw()->GetImmediateValue() != 0) {
            srcHasOffset = true;
            break;
        }
    }
    if (srcHasOffset) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "ReplaceTensor::FoldL0C2UBCopyOffset: copy op %d carries non-zero fromOffset while its "
                          "assemble consumer op %d carries toOffset, cannot fold offsets.",
                          copyOp->GetOpMagic(), op->GetOpMagic());
        return FAILED;
    }
    copyAttr->SetToOffset(OpImmediate::Specified(TensorOffset(assembleIn->GetOffset(), assembleIn->GetDynOffset())));
    copyOp->SetAttribute(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(Matrix::CopyMode::INSERT));
    return SUCCESS;
}

Status ReplaceTensor::MarkTensorAsPartialMem(Function& func)
{
    for (auto& op : func.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto iOperand = op.GetInputOperand(0);
        auto oOperand = op.GetOutputOperand(0);
        if (iOperand->GetRawTensor() != oOperand->GetRawTensor()) {
            continue;
        }
        iOperand->SetAttr("isPartialMem", true);
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
