/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common_operation_eliminate_utils.cpp
 * \brief utils of common operation elimination
 */

#include "common_operation_eliminate_utils.h"

#include <algorithm>
#include <sstream>
#include <string>

#include "interface/operation/operation.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "CommonOperationEliminateUtils"

namespace npu::tile_fwk {
const std::unordered_set<Opcode>& CommonOperationEliminateUtils::GetSkipEliminateOpcodes()
{
    // Opcodes in this set are excluded from common operation elimination on purpose:
    //   - OP_VIEW: works with GraphPartition processing logic.
    static const std::unordered_set<Opcode> skipOpcodes = {Opcode::OP_VIEW, Opcode::OP_SLICE};
    return skipOpcodes;
}

void CommonOperationEliminateUtils::SortedProducer(std::vector<Operation*>& sortedProducers) const
{
    // Keep the original producer order for ties so hash generation stays deterministic.
    std::stable_sort(sortedProducers.begin(), sortedProducers.end(), [](const Operation* op1, const Operation* op2) {
        const auto& iOp1 = op1->GetIOperands();
        const auto& iOp2 = op2->GetIOperands();
        size_t minLen = std::min(iOp1.size(), iOp2.size());
        for (size_t i = 0; i < minLen; ++i) {
            LogicalTensorPtr ptr1 = iOp1[i];
            LogicalTensorPtr ptr2 = iOp2[i];
            if (ptr1 != ptr2) {
                return ptr1 < ptr2;
            }
        }
        if (iOp1.size() != iOp2.size()) {
            return iOp1.size() < iOp2.size();
        }
        std::stringstream ss1, ss2;
        for (const auto& attr : OpcodeManager::Inst().GetAttrs(op1->GetOpcode())) {
            ss1 << " attr: [" << attr << " : " << op1->DumpAttr(attr) << "]";
        }
        for (const auto& attr : OpcodeManager::Inst().GetAttrs(op2->GetOpcode())) {
            ss2 << " attr: [" << attr << " : " << op2->DumpAttr(attr) << "]";
        }
        return ss1.str() < ss2.str();
    });
}

void CommonOperationEliminateUtils::CollectProducerInfo(const std::vector<Operation*>& sortedProducers,
                                                        const LogicalTensorPtr& curTensor,
                                                        std::vector<std::string>& opStrList,
                                                        std::stringstream& ss) const
{
    for (const auto& op : sortedProducers) {
        if (op == nullptr) {
            continue;
        }
        ss.str("");
        ss.clear();
        ss << op->GetOpcodeStr(true);
        for (const auto& iOperands : op->GetIOperands()) {
            if (iOperands == nullptr || iOperands->tensor == nullptr) {
                continue;
            }
            ss << "[i";
            ss << "$" << iOperands->tensor->DumpSSA(false, false);
            ss << iOperands->DumpType();
            ss << "(";
            for (size_t i = 0; i < iOperands->offset.size(); ++i) {
                ss << iOperands->offset[i];
                if (i != iOperands->offset.size() - 1) {
                    ss << ", ";
                }
            }
            if (curTensor && !curTensor->GetDynValidShape().empty()) {
                std::string shapeStr;
                for (size_t i = 0; i < curTensor->GetDynValidShape().size(); i++) {
                    shapeStr += curTensor->GetDynValidShape()[i].Dump();
                }
                ss << "[" << shapeStr << "]";
                ss << "memoryType: [" << MemoryTypeToString(curTensor->GetMemoryTypeOriginal()) << "]";
            }
            ss << ")]";
        }
        if (op->GetOpAttribute() != nullptr) {
            ss << " " << op->GetOpAttribute()->Dump();
        }
        if (!op->DumpAttr().empty()) {
            ss << " " << op->DumpAttr();
        }
        for (const auto& attr : OpcodeManager::Inst().GetAttrs(op->GetOpcode())) {
            ss << " attr: [" << attr << " : " << op->DumpAttr(attr) << "]";
        }
        ss << "id" << op->GetSubgraphID();
        opStrList.emplace_back(ss.str());
    }
    ss.str("");
    ss.clear();
    for (const auto& str : opStrList) {
        ss << str;
    }
}

unsigned long CommonOperationEliminateUtils::ComputeHash(const std::vector<Operation*>& producers,
                                                         LogicalTensorPtr curTensor) const
{
    std::vector<std::string> opStrList;
    std::stringstream ss;
    std::vector<Operation*> sortedProducers = producers;
    SortedProducer(sortedProducers);
    CollectProducerInfo(sortedProducers, curTensor, opStrList, ss);
    std::hash<std::string> hasher;
    return hasher(ss.str());
}

Status CommonOperationEliminateUtils::EliminateCommonOperation(Function& function)
{
    CommonOperationEliminateUtils commonOperationEliminateUtils;
    return commonOperationEliminateUtils.Process(function);
}

Status CommonOperationEliminateUtils::Process(Function& function)
{
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    std::vector<LogicalTensorPtr> sequence;
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> tensorProducerMap;
    tensorProducerMap = GetTensorProducers(function, sequence);
    mixSubgraphIds_ = GetMixSubgraphIds(function);
    std::unordered_set<Operation*> cacheProducers;
    bool anyDeleted = false;
    for (auto& orderedTensor : sequence) {
        auto& producerGroup = tensorProducerMap[orderedTensor];
        if (producerGroup.empty() ||
            !TensorProducersMerge(function, orderedTensor, cacheProducers, tensorProducerMap)) {
            continue;
        }
        for (auto op : producerGroup) {
            if (op == nullptr) {
                continue;
            }
            if (!cacheProducers.count(op)) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Operation[%d] was set as deleted.", op->GetOpMagic());
                op->SetAsDeleted();
                anyDeleted = true;
            }
        }
    }
    if (anyDeleted) {
        function.EraseOperations(true, false);
        if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Eliminate dead operation failed in CommonOperationEliminateUtils.");
            return FAILED;
        }
    }
    return SUCCESS;
}

std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> CommonOperationEliminateUtils::GetTensorProducers(
    Function& function, std::vector<LogicalTensorPtr>& sequence)
{
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> tensorProducerMap;
    std::unordered_set<int> visitedTensors;
    auto allOps = function.Operations(true).DuplicatedOpList();
    for (const auto& op : allOps) {
        if (op == nullptr) {
            continue;
        }
        auto& outputTensors = op->GetOOperands();
        for (const auto& tensor : outputTensors) {
            if (tensor == nullptr || visitedTensors.count(tensor->GetMagic())) {
                continue;
            }
            visitedTensors.insert(tensor->GetMagic());
            for (const auto& producer : tensor->GetProducers()) {
                if (producer == nullptr) {
                    APASS_LOG_ERROR_F(Elements::Operation, "Producer operation nullptr for Tensor[%d].",
                                      tensor->GetMagic());
                    continue;
                }
                if (tensorProducerMap.count(tensor) == 0) {
                    sequence.push_back(tensor);
                }
                tensorProducerMap[tensor].push_back(producer);
            }
        }
    }
    return tensorProducerMap;
}

std::pair<LogicalTensorPtr, std::vector<Operation*>> CommonOperationEliminateUtils::TensorHashExist(
    const LogicalTensorPtr orderedTensor, std::unordered_set<Operation*>& cacheProducers,
    const std::unordered_map<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerMap)
{
    const std::vector<Operation*>& producers = tensorProducerMap.find(orderedTensor)->second;
    for (auto operation : producers) {
        if (operation == nullptr) {
            continue;
        }
        auto& inputsMemType = OpcodeManager::Inst().GetInputsMemType(operation->GetOpcode());
        auto& outputsMemType = OpcodeManager::Inst().GetOutputsMemType(operation->GetOpcode());
        OpCalcType opCalcType = OpcodeManager::Inst().GetOpCalcType(operation->GetOpcode());
        bool inputCheck = inputsMemType.size() == 1 && inputsMemType[0] == MemoryType::MEM_L1;
        bool calcTypeCheck = opCalcType == OpCalcType::MOVE_LOCAL || opCalcType == OpCalcType::MOVE_IN;
        bool outputCheck = outputsMemType.size() == 1 && outputsMemType[0] != MemoryType::MEM_L1;
        if (inputCheck && calcTypeCheck && outputCheck) { // copy from L1 to L0
            return {nullptr, {}};
        }
        if (GetSkipEliminateOpcodes().count(operation->GetOpcode()) != 0) {
            return {nullptr, {}};
        }
        if (operation->GetBoolAttribute(OpAttributeKey::dontTouch)) {
            return {nullptr, {}};
        }
    }
    uint64_t groupHash = ComputeHash(producers, orderedTensor);
    if (hashCache_.count(groupHash) != 0) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Tensor[%d] are marked as hash already existed tensor.",
                          orderedTensor->GetMagic());
        return hashCache_[groupHash];
    }
    hashCache_.emplace(groupHash, std::make_pair(orderedTensor, producers));
    if (orderedTensor == nullptr) {
        return {nullptr, {}};
    }
    for (auto producer : orderedTensor->GetProducers()) {
        if (producer != nullptr) {
            cacheProducers.insert(producer);
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Tensor[%d] hash already existed.", orderedTensor->GetMagic());
    return {nullptr, {}};
}

void CommonOperationEliminateUtils::UpdateView(ViewOpAttribute* viewOpAttribute,
                                               const std::shared_ptr<LogicalTensor> oldTensor,
                                               const std::shared_ptr<LogicalTensor> newTensor) const
{
    auto& fromOffset = viewOpAttribute->GetFromOffset();
    for (size_t j = 0; j < fromOffset.size(); j++) {
        fromOffset[j] -= oldTensor->offset[j] - newTensor->offset[j];
    }
}

void CommonOperationEliminateUtils::UpdateCopy(CopyOpAttribute* copyOpAttribute,
                                               const std::shared_ptr<LogicalTensor> oldTensor,
                                               const std::shared_ptr<LogicalTensor> newTensor) const
{
    if (!copyOpAttribute->IsCopyOut()) {
        auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
        (void)memType;
        for (size_t j = 0; j < fromOffset.size(); j++) {
            fromOffset[j] -= oldTensor->offset[j] - newTensor->offset[j];
        }
        copyOpAttribute->SetFromOffset(fromOffset);
    }
}

void CommonOperationEliminateUtils::UpdateConnection(LogicalTensorPtr oldTensor, LogicalTensorPtr newTensor)
{
    auto consumers = oldTensor->GetConsumers();
    for (auto& cur : consumers) {
        if (cur == nullptr) {
            continue;
        }
        cur->ReplaceInput(newTensor, oldTensor);
        auto attrPtr = cur->GetOpAttribute().get();
        if (attrPtr == nullptr) {
            continue;
        }
        if (cur->GetOpcode() == Opcode::OP_VIEW) {
            if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(attrPtr)) {
                UpdateView(viewOpAttribute, oldTensor, newTensor);
                continue;
            }
        } else if (cur->GetOpcode() == Opcode::OP_COPY_IN) {
            if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(attrPtr)) {
                UpdateCopy(copyOpAttribute, oldTensor, newTensor);
                continue;
            }
        }
    }
}

uint32_t CommonOperationEliminateUtils::GetTensorCoreFlag(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr) {
        return 0;
    }
    constexpr uint32_t kAicTensorFlag = 1U;
    constexpr uint32_t kAivTensorFlag = 2U;
    switch (tensor->GetMemoryTypeOriginal()) {
        case MemoryType::MEM_L1:
        case MemoryType::MEM_L0A:
        case MemoryType::MEM_L0B:
        case MemoryType::MEM_L0C:
            return kAicTensorFlag;
        case MemoryType::MEM_UB:
            return kAivTensorFlag;
        default:
            return 0;
    }
}

void CommonOperationEliminateUtils::CollectSubgraphIds(const std::set<Operation*, LogicalTensor::CompareOp>& ops,
                                                       std::unordered_set<int>& subgraphIds) const
{
    for (const auto& op : ops) {
        if (op == nullptr) {
            continue;
        }
        if (op->GetSubgraphID() >= 0) {
            subgraphIds.insert(op->GetSubgraphID());
        }
    }
}

void CommonOperationEliminateUtils::UpdateInternalTensorCoreFlag(
    const LogicalTensorPtr& tensor, std::unordered_map<int, uint32_t>& subgraphCoreFlags) const
{
    uint32_t tensorFlag = GetTensorCoreFlag(tensor);
    if (tensorFlag == 0) {
        return;
    }

    std::unordered_set<int> producerSubgraphIds;
    std::unordered_set<int> consumerSubgraphIds;
    CollectSubgraphIds(tensor->GetProducers(), producerSubgraphIds);
    CollectSubgraphIds(tensor->GetConsumers(), consumerSubgraphIds);
    for (const auto producerSubgraphId : producerSubgraphIds) {
        if (consumerSubgraphIds.count(producerSubgraphId) != 0) {
            subgraphCoreFlags[producerSubgraphId] |= tensorFlag;
        }
    }
}

std::unordered_set<int> CommonOperationEliminateUtils::GetMixSubgraphIds(Function& function) const
{
    constexpr uint32_t kAicTensorFlag = 1U;
    constexpr uint32_t kAivTensorFlag = 2U;
    std::unordered_map<int, uint32_t> internalTensorFlagsBySubgraph;
    std::unordered_set<int> handledTensorMagics;
    for (const auto& opPtr : function.Operations(true).DuplicatedOpList()) {
        if (opPtr == nullptr) {
            continue;
        }
        for (const auto& outputTensor : opPtr->GetOOperands()) {
            if (outputTensor == nullptr) {
                continue;
            }
            if (!handledTensorMagics.insert(outputTensor->GetMagic()).second) {
                continue;
            }
            UpdateInternalTensorCoreFlag(outputTensor, internalTensorFlagsBySubgraph);
        }
    }

    std::unordered_set<int> result;
    for (const auto& entry : internalTensorFlagsBySubgraph) {
        if ((entry.second & kAicTensorFlag) == 0 || (entry.second & kAivTensorFlag) == 0) {
            continue;
        }
        result.insert(entry.first);
    }
    return result;
}

MemoryType CommonOperationEliminateUtils::GetAssembleInputMemoryType(Operation* consumer,
                                                                     const LogicalTensorPtr& input) const
{
    if (input != nullptr && input->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        return input->GetMemoryTypeOriginal();
    }
    if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_ASSEMBLE) {
        return MemoryType::MEM_UNKNOWN;
    }
    auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(consumer->GetOpAttribute());
    if (assembleAttr == nullptr) {
        return MemoryType::MEM_UNKNOWN;
    }
    return assembleAttr->GetFrom();
}

MemoryType CommonOperationEliminateUtils::GetAssembleOutputMemoryType(Operation* consumer) const
{
    if (consumer == nullptr) {
        return MemoryType::MEM_UNKNOWN;
    }
    const auto& outputs = consumer->GetOOperands();
    if (!outputs.empty() && outputs.front() != nullptr &&
        outputs.front()->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        return outputs.front()->GetMemoryTypeOriginal();
    }
    return MemoryType::MEM_UNKNOWN;
}

bool CommonOperationEliminateUtils::HasSameMemoryAssembleUse(const LogicalTensorPtr& tensor,
                                                             std::unordered_set<int>& visitedTensorMagics) const
{
    if (tensor == nullptr || !visitedTensorMagics.insert(tensor->GetMagic()).second) {
        return false;
    }
    for (const auto& consumer : tensor->GetConsumers()) {
        if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        MemoryType inputType = GetAssembleInputMemoryType(consumer, tensor);
        MemoryType outputType = GetAssembleOutputMemoryType(consumer);
        if (inputType == MemoryType::MEM_UNKNOWN || outputType == MemoryType::MEM_UNKNOWN || inputType == outputType) {
            return true;
        }
        for (const auto& output : consumer->GetOOperands()) {
            if (HasSameMemoryAssembleUse(output, visitedTensorMagics)) {
                return true;
            }
        }
    }
    return false;
}

bool CommonOperationEliminateUtils::ShouldSkipSameMemoryAssembleMerge(const LogicalTensorPtr& oldTensor) const
{
    std::unordered_set<int> visitedTensorMagics;
    if (HasSameMemoryAssembleUse(oldTensor, visitedTensorMagics)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Skip eliminating Tensor[%d] because it has same-memory assemble consumers.",
                          oldTensor->GetMagic());
        return true;
    }
    return false;
}

bool CommonOperationEliminateUtils::WouldExposeMixInternalTensorAfterMerge(
    const LogicalTensorPtr& oldTensor, const LogicalTensorPtr& newTensor,
    const std::unordered_set<int>& mixSubgraphIds) const
{
    if (oldTensor == nullptr || newTensor == nullptr || mixSubgraphIds.empty()) {
        return false;
    }

    std::unordered_set<int> producerMixSubgraphIds;
    for (const auto& producer : newTensor->GetProducers()) {
        if (producer != nullptr && mixSubgraphIds.count(producer->GetSubgraphID()) != 0) {
            producerMixSubgraphIds.insert(producer->GetSubgraphID());
        }
    }
    for (const auto mixSubgraphId : producerMixSubgraphIds) {
        bool hasConsumerInMix = false;
        bool hasOtherSubgraphOp = false;
        auto updateConsumerSubgraphUse = [&](Operation* op) {
            if (op == nullptr) {
                return;
            }
            if (op->GetSubgraphID() == mixSubgraphId) {
                hasConsumerInMix = true;
            } else {
                hasOtherSubgraphOp = true;
            }
        };
        for (const auto& producer : newTensor->GetProducers()) {
            if (producer != nullptr && producer->GetSubgraphID() != mixSubgraphId) {
                hasOtherSubgraphOp = true;
            }
        }
        for (const auto& consumer : newTensor->GetConsumers()) {
            updateConsumerSubgraphUse(consumer);
        }
        for (const auto& consumer : oldTensor->GetConsumers()) {
            updateConsumerSubgraphUse(consumer);
        }
        if (hasConsumerInMix && hasOtherSubgraphOp) {
            return true;
        }
    }
    return false;
}

bool CommonOperationEliminateUtils::TensorProducersMerge(
    Function& function, const LogicalTensorPtr orderedTensor, std::unordered_set<Operation*>& cacheProducers,
    const std::unordered_map<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerMap)
{
    auto& producers = tensorProducerMap.at(orderedTensor);
    if (producers.empty()) {
        return false;
    }
    auto existOp = TensorHashExist(orderedTensor, cacheProducers, tensorProducerMap);
    if (existOp.first == nullptr || orderedTensor == nullptr || existOp.second.empty()) {
        return false;
    }
    if (orderedTensor->shape != existOp.first->shape) {
        return false;
    }
    if (orderedTensor->tensor->GetDataType() != existOp.first->tensor->GetDataType()) {
        return false;
    }
    LogicalTensorPtr oldTensor = orderedTensor;
    LogicalTensorPtr newTensor = existOp.first;
    if (FunctionUtils::GetNodeType(*oldTensor, function) == NodeType::OUTCAST) {
        return false;
    }
    if (producers.size() == existOp.second.size()) {
        bool allSame = true;
        for (size_t i = 0; i < existOp.second.size() && allSame; i++) {
            allSame = (producers[i] == existOp.second[i]);
        }
        if (allSame) {
            return false;
        }
    }
    if (newTensor->GetConsumers().size() == 0 || oldTensor->GetConsumers().size() == 0) {
        return false;
    }
    if (ShouldSkipSameMemoryAssembleMerge(oldTensor)) {
        return false;
    }
    if (WouldExposeMixInternalTensorAfterMerge(oldTensor, newTensor, mixSubgraphIds_)) {
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "Skip eliminating Tensor[%d] to avoid exposing mix subgraph internal Tensor[%d] to other subgraphs.",
            oldTensor->GetMagic(), newTensor->GetMagic());
        return false;
    }
    UpdateConnection(oldTensor, newTensor);
    oldTensor->GetConsumers().clear();
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "In CommonOperationEliminateUtils, Tensor[%d] and producersgroup are marked as redundant.",
                      oldTensor->GetMagic());
    return true;
}
} // namespace npu::tile_fwk
