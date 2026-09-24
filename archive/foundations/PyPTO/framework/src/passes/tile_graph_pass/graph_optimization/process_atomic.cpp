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
 * \file process_atomic.cpp
 * \brief Process atomic operations including ReduceAcc and AtomicRMW
 */

#include "process_atomic.h"
#include "interface/configs/config_manager_ng.h"
#include "passes/pass_check/process_atomic_checker.h"
#include "interface/operation/attribute.h"
#include "tilefwk/tilefwk_op.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/merge_view_assemble_utils.h"
#include "passes/pass_utils/view_reshape_assemble_reorder_utils.h"
#include "passes/pass_log/pass_log.h"
#include <algorithm>
#include <set>

#define MODULE_NAME "ProcessAtomic"

namespace npu {
namespace tile_fwk {

namespace {
std::vector<SymbolicScalar> GetSymbolicShapeOrStatic(const std::shared_ptr<LogicalTensor>& tensor)
{
    if (tensor == nullptr) {
        return {};
    }
    const auto& dynShape = tensor->GetDynValidShape();
    return dynShape.size() == tensor->GetShape().size() ? dynShape : SymbolicScalar::FromConcrete(tensor->GetShape());
}

bool IsContractOpcode(Opcode opcode)
{
    return opcode == config::GetContractOpcode() || (!config::EnableSlice() && opcode == Opcode::OP_ASSEMBLE_SSA);
}
} // namespace

Status ProcessAtomic::PreCheck(Function& function)
{
    ProcessAtomicChecker checker;
    return checker.DoPreCheck(function);
}

Status ProcessAtomic::PostCheck(Function& function)
{
    ProcessAtomicChecker checker;
    return checker.DoPostCheck(function);
}

Status ProcessAtomic::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start ProcessAtomic.");
    if (CheckAtomicRMWUnsupportedMode(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Unsupported AtomicRMW mode detected.");
        return FAILED;
    }
    bool hasReduceAccCascade = false;
    if (EliminateVecDupBranch(function, hasReduceAccCascade) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate VecDup branch failed.");
        return FAILED;
    }
    if (EliminateReduceAcc(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate ReduceAcc failed.");
        return FAILED;
    }
    if (EliminateAtomicRMW(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate AtomicRMW failed.");
        return FAILED;
    }
    if (hasReduceAccCascade) {
        Status status = MergeViewAssembleUtils::MergeViewAssemble(function);
        if (status != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Merge assemble and view failed.");
            return status;
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End ProcessAtomic.");
    return SUCCESS;
}

Status ProcessAtomic::CheckAtomicRMWUnsupportedMode(Function& function)
{
    for (const auto& op : function.Operations(true, SortOperationsMode::LIGHTWEIGHT)) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            int rmwModeValue = op.GetIntAttribute(OpAttributeKey::rmwMode);
            AtomicRMWMode rmwMode = static_cast<AtomicRMWMode>(rmwModeValue);
            if (rmwMode == AtomicRMWMode::MAX || rmwMode == AtomicRMWMode::MIN) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "Op[%d] AtomicRMW mode '%s' is not supported yet. "
                    "Currently only ADD mode is supported. Please use ADD mode or wait for future support.%s",
                    op.GetOpMagic(), (rmwMode == AtomicRMWMode::MAX ? "MAX" : "MIN"), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status ProcessAtomic::EliminateReduceAcc(Function& function)
{
    bool anyDeleted = false;
    for (auto& op : function.Operations(true, SortOperationsMode::LIGHTWEIGHT)) {
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            APASS_LOG_INFO_F(Elements::Operation, "ATOMIC_ADD, opmagic: %d", op.GetOpMagic());
            auto reduceOut = op.GetOOperands().front();
            reduceOut->GetProducers().clear();

            for (const auto& input : op.GetIOperands()) {
                auto producersBackup = input->GetProducers();
                for (auto& produceCopyOutOp : producersBackup) {
                    produceCopyOutOp->ReplaceOOperand(0, reduceOut);
                    produceCopyOutOp->SetAttribute(RMW_MODE_ATTR_ADD, 1);
                }
            }
            op.SetAsDeleted();
            anyDeleted = true;
            APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] will be deleted.", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic());
        }
    }
    if (anyDeleted) {
        function.EraseOperations(true, true, SortOperationsMode::LIGHTWEIGHT);
        if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function,
                              "Eliminate dead operation failed for ReduceAcc in CommonOperationEliminate.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessAtomic::EliminateAtomicRMW(Function& function)
{
    std::vector<Operation*> atomicRmwOps;
    for (auto& op : function.Operations(true, SortOperationsMode::LIGHTWEIGHT)) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            atomicRmwOps.emplace_back(&op);
        }
    }
    if (atomicRmwOps.empty()) {
        return SUCCESS;
    }
    if (PrepareAtomicRMWSharedInputs(function, atomicRmwOps) != SUCCESS) {
        return FAILED;
    }
    bool anyDeleted = false;
    for (auto* op : atomicRmwOps) {
        if (op == nullptr || op->IsDeleted()) {
            continue;
        }
        if (ProcessSingleAtomicRMW(*op) != SUCCESS) {
            return FAILED;
        }
        anyDeleted = true;
    }
    if (anyDeleted) {
        function.EraseOperations(true, true, SortOperationsMode::LIGHTWEIGHT);
        if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function,
                              "Eliminate dead operation failed for AtomicRMW in CommonOperationEliminate.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessAtomic::ProcessSingleAtomicRMW(Operation& op)
{
    APASS_LOG_INFO_F(Elements::Operation, "ATOMIC_RMW, opmagic: %d", op.GetOpMagic());

    auto rmwOut = op.GetOOperands().front();
    auto contractAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
    if (contractAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op[%d] missing contract op attribute; Cannot eliminate.",
                          op.GetOpMagic());
        return FAILED;
    }

    auto& rmwOffset = contractAttr->GetToOffset();
    auto& rmwDynOffset = contractAttr->GetToDynOffset();

    int rmwModeValue = op.GetIntAttribute(OpAttributeKey::rmwMode);
    AtomicRMWMode rmwMode = static_cast<AtomicRMWMode>(rmwModeValue);
    if (GetRmwAttrKey(rmwMode).empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op[%d] has invalid rmwMode value %d.", op.GetOpMagic(), rmwModeValue);
        return FAILED;
    }
    for (const auto& input : op.GetIOperands()) {
        if (ProcessAtomicInput(op, input, rmwOut, rmwMode, rmwOffset, rmwDynOffset) != SUCCESS) {
            return FAILED;
        }
    }
    op.SetAsDeleted();
    APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] will be deleted.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
    return SUCCESS;
}

Status ProcessAtomic::ProcessAtomicInput(Operation& atomicOp, const std::shared_ptr<LogicalTensor>& input,
                                         const std::shared_ptr<LogicalTensor>& output, AtomicRMWMode rmwMode,
                                         const std::vector<int64_t>& rmwOffset,
                                         const std::vector<SymbolicScalar>& rmwDynOffset)
{
    bool hasContractProducer = false;
    auto producersBackup = input->GetProducers();
    for (auto* producerOp : producersBackup) {
        if (!IsContractOpcode(producerOp->GetOpcode())) {
            continue;
        }
        if (ProcessAtomicContractProducer(atomicOp, *producerOp, output, rmwMode, rmwOffset, rmwDynOffset) != SUCCESS) {
            return FAILED;
        }
        hasContractProducer = true;
    }
    if (hasContractProducer || !HasReshapeProducer(input)) {
        return SUCCESS;
    }
    return ProcessAtomicThroughReshape(atomicOp, input, output, rmwMode, rmwOffset, rmwDynOffset);
}

Status ProcessAtomic::ProcessAtomicContractProducer(Operation& atomicOp, Operation& producerOp,
                                                    const std::shared_ptr<LogicalTensor>& output, AtomicRMWMode rmwMode,
                                                    const std::vector<int64_t>& rmwOffset,
                                                    const std::vector<SymbolicScalar>& rmwDynOffset)
{
    if (producerOp.GetIOperands().size() != 1 || !HasReshapeProducer(producerOp.GetInputOperand(0))) {
        return ProcessContractProducer(producerOp, output, rmwMode, rmwOffset, rmwDynOffset);
    }
    std::vector<int64_t> combinedOffset;
    std::vector<SymbolicScalar> combinedDynOffset;
    if (CombineContractOffset(producerOp, rmwOffset, rmwDynOffset, combinedOffset, combinedDynOffset) != SUCCESS) {
        return FAILED;
    }
    return ProcessAtomicThroughReshape(atomicOp, producerOp.GetInputOperand(0), output, rmwMode, combinedOffset,
                                       combinedDynOffset);
}

Status ProcessAtomic::ProcessAtomicThroughReshape(Operation& atomicOp, const std::shared_ptr<LogicalTensor>& input,
                                                  const std::shared_ptr<LogicalTensor>& output, AtomicRMWMode rmwMode,
                                                  const std::vector<int64_t>& rmwOffset,
                                                  const std::vector<SymbolicScalar>& rmwDynOffset)
{
    ReshapeRemapResult remapResult;
    if (FindUpstreamAssembleAndRemapOffset(input, output, rmwOffset, rmwDynOffset, remapResult) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Op[%d] cannot remap AtomicRMW offset through Reshape; Cannot eliminate.",
                          atomicOp.GetOpMagic());
        return FAILED;
    }
    for (auto* contract : remapResult.assembles) {
        if (MarkContractProducerAtomic(*contract, rmwMode, remapResult.mappedOffset, remapResult.mappedDynOffset) !=
            SUCCESS) {
            return FAILED;
        }
    }
    return RetargetReshapeChain(atomicOp, output, remapResult);
}

bool ProcessAtomic::HasReshapeProducer(const std::shared_ptr<LogicalTensor>& input) const
{
    if (input == nullptr) {
        return false;
    }
    for (auto* producer : input->GetProducers()) {
        if (producer != nullptr && producer->GetOpcode() == Opcode::OP_RESHAPE) {
            return true;
        }
    }
    return false;
}

Status ProcessAtomic::FindUpstreamAssembleAndRemapOffset(const std::shared_ptr<LogicalTensor>& input,
                                                         const std::shared_ptr<LogicalTensor>& outputBase,
                                                         const std::vector<int64_t>& offset,
                                                         const std::vector<SymbolicScalar>& dynOffset,
                                                         ReshapeRemapResult& result) const
{
    auto current = input;
    result = {};
    result.mappedOffset = offset;
    result.mappedDynOffset = dynOffset;
    if (outputBase == nullptr || outputBase->GetShape().size() != offset.size()) {
        return FAILED;
    }
    auto currentBaseShape = outputBase->GetShape();
    auto currentBaseDynShape = GetSymbolicShapeOrStatic(outputBase);
    std::set<int> visited;
    while (current != nullptr) {
        bool found = false;
        if (CollectTerminalAssembles(current, visited, result, found) != SUCCESS) {
            return FAILED;
        }
        if (found) {
            result.assembleOutputShape = std::move(currentBaseShape);
            result.assembleOutputDynShape = std::move(currentBaseDynShape);
            return SUCCESS;
        }
        const auto& producers = current->GetProducers();
        if (producers.size() != 1) {
            return FAILED;
        }
        auto* producer = *producers.begin();
        if (producer == nullptr || !visited.insert(producer->GetOpMagic()).second) {
            return FAILED;
        }
        if (RemapThroughReshape(*producer, current, currentBaseShape, currentBaseDynShape, result) != SUCCESS) {
            return FAILED;
        }
    }
    return FAILED;
}

Status ProcessAtomic::CollectTerminalAssembles(const std::shared_ptr<LogicalTensor>& current, std::set<int>& visited,
                                               ReshapeRemapResult& result, bool& found) const
{
    found = false;
    const auto& producers = current->GetProducers();
    if (producers.empty()) {
        return FAILED;
    }
    bool allAssemble = std::all_of(producers.begin(), producers.end(), [](const Operation* producer) {
        return producer != nullptr && IsContractOpcode(producer->GetOpcode());
    });
    if (!allAssemble) {
        return SUCCESS;
    }
    for (auto* producer : producers) {
        if (!visited.insert(producer->GetOpMagic()).second) {
            return FAILED;
        }
        result.assembles.push_back(producer);
    }
    found = true;
    return SUCCESS;
}

Status ProcessAtomic::RemapThroughReshape(Operation& producer, std::shared_ptr<LogicalTensor>& current,
                                          std::vector<int64_t>& currentBaseShape,
                                          std::vector<SymbolicScalar>& currentBaseDynShape,
                                          ReshapeRemapResult& result) const
{
    if (producer.GetOpcode() != Opcode::OP_RESHAPE || producer.GetIOperands().size() != 1) {
        return FAILED;
    }
    auto next = producer.GetInputOperand(0);
    std::vector<int64_t> nextBaseShape, nextOffset;
    std::vector<SymbolicScalar> nextBaseDynShape, nextDynOffset;
    result.reshapeOps.push_back(&producer);
    result.reshapeOutputShapes.push_back(currentBaseShape);
    result.reshapeOutputDynShapes.push_back(currentBaseDynShape);
    if (!ViewReshapeAssembleReorderUtils::RemapOffsetBackwardThroughReshape(
            next, current, currentBaseShape, currentBaseDynShape, result.mappedOffset, result.mappedDynOffset,
            nextBaseShape, nextBaseDynShape, nextOffset, nextDynOffset)) {
        return FAILED;
    }
    currentBaseShape = std::move(nextBaseShape);
    currentBaseDynShape = std::move(nextBaseDynShape);
    result.mappedOffset = std::move(nextOffset);
    result.mappedDynOffset = std::move(nextDynOffset);
    current = next;
    return SUCCESS;
}

Status ProcessAtomic::RetargetReshapeChain(Operation& atomicOp, const std::shared_ptr<LogicalTensor>& output,
                                           const ReshapeRemapResult& remapResult)
{
    auto* function = atomicOp.BelongTo();
    if (function == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op[%d] does not belong to a function; Cannot retarget Reshape chain.",
                          atomicOp.GetOpMagic());
        return FAILED;
    }
    if (remapResult.assembles.empty()) {
        return FAILED;
    }
    auto current = remapResult.assembles.front()->GetOutputOperand(0);
    if (current == nullptr || current->GetShape() != remapResult.assembleOutputShape) {
        auto original = current;
        current = irBuilder_.CreateTensorVar(*function, output->Datatype(), remapResult.assembleOutputShape,
                                             remapResult.assembleOutputDynShape, output->Format());
        if (original != nullptr) {
            current->CopyMemoryType(original);
        }
        for (auto* assemble : remapResult.assembles) {
            assemble->ReplaceOOperand(0, current);
        }
    }
    for (size_t index = remapResult.reshapeOps.size(); index-- > 0;) {
        auto* reshapeOp = remapResult.reshapeOps[index];
        auto originalReshapeOutput = reshapeOp->GetOutputOperand(0);
        auto reshapeOutput = index == 0 ? output :
                                          irBuilder_.CreateTensorVar(
                                              *function, output->Datatype(), remapResult.reshapeOutputShapes[index],
                                              remapResult.reshapeOutputDynShapes[index], output->Format());
        if (index != 0 && originalReshapeOutput != nullptr) {
            reshapeOutput->CopyMemoryType(originalReshapeOutput);
        }
        reshapeOp->ReplaceIOperand(0, current);
        reshapeOp->ReplaceOOperand(0, reshapeOutput);
        reshapeOp->SetAttribute("reshape", reshapeOutput->GetShape());
        reshapeOp->SetAttribute(OP_ATTR_PREFIX + "validShape", remapResult.reshapeOutputDynShapes[index]);
        reshapeOutput->UpdateDynValidShape(remapResult.reshapeOutputDynShapes[index]);
        current = std::move(reshapeOutput);
    }
    return SUCCESS;
}

Status ProcessAtomic::CombineContractOffset(const Operation& contract, const std::vector<int64_t>& offset,
                                            const std::vector<SymbolicScalar>& dynOffset,
                                            std::vector<int64_t>& combinedOffset,
                                            std::vector<SymbolicScalar>& combinedDynOffset) const
{
    auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(contract.GetOpAttribute());
    if (attr == nullptr || attr->GetToOffset().size() != offset.size()) {
        return FAILED;
    }
    combinedOffset = attr->GetToOffset();
    for (size_t i = 0; i < offset.size(); ++i) {
        combinedOffset[i] += offset[i];
    }
    combinedDynOffset.clear();
    if (!attr->GetToDynOffset().empty() || !dynOffset.empty()) {
        auto lhs = attr->GetToDynOffset().size() == offset.size() ? attr->GetToDynOffset() :
                                                                    SymbolicScalar::FromConcrete(attr->GetToOffset());
        auto rhs = dynOffset.size() == offset.size() ? dynOffset : SymbolicScalar::FromConcrete(offset);
        combinedDynOffset.reserve(offset.size());
        for (size_t i = 0; i < offset.size(); ++i) {
            combinedDynOffset.push_back((lhs[i] + rhs[i]).Simplify());
        }
    }
    return SUCCESS;
}

bool ProcessAtomic::HasContractProducer(const std::shared_ptr<LogicalTensor>& input) const
{
    if (input == nullptr) {
        return false;
    }
    for (auto* producerOp : input->GetProducers()) {
        if (producerOp == nullptr) {
            continue;
        }
        if (IsContractOpcode(producerOp->GetOpcode())) {
            return true;
        }
    }
    return false;
}

bool ProcessAtomic::HasConsumerExcept(const std::shared_ptr<LogicalTensor>& input, const Operation& op) const
{
    if (input == nullptr) {
        return false;
    }
    for (auto* consumerOp : input->GetConsumers()) {
        if (consumerOp != nullptr && consumerOp->GetOpMagic() != op.GetOpMagic()) {
            return true;
        }
    }
    return false;
}

Status ProcessAtomic::PrepareAtomicRMWSharedInputs(Function& function,
                                                   const std::vector<Operation*>& atomicRmwOps) const
{
    for (auto* op : atomicRmwOps) {
        if (op == nullptr || op->IsDeleted()) {
            continue;
        }
        auto inputsBackup = op->GetIOperands();
        for (const auto& input : inputsBackup) {
            if (PrepareExclusiveAtomicInput(function, *op, input) == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Prepare shared input failed for AtomicRMW op[%d].",
                                  op->GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

std::shared_ptr<LogicalTensor> ProcessAtomic::PrepareExclusiveAtomicInput(
    Function& function, Operation& atomicOp, const std::shared_ptr<LogicalTensor>& input) const
{
    if (input == nullptr || !HasConsumerExcept(input, atomicOp) || !HasContractProducer(input)) {
        return input;
    }

    auto producersBackup = input->GetProducers();
    auto clonedInput = input->Clone(function, true);
    if (clonedInput == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Clone atomic input tensor[%d] failed.", input->GetMagic());
        return nullptr;
    }
    atomicOp.ReplaceInput(clonedInput, input);

    for (auto* producerOp : producersBackup) {
        if (producerOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Null producer detected for atomic input tensor[%d].",
                              input->GetMagic());
            return nullptr;
        }
        if (!IsContractOpcode(producerOp->GetOpcode())) {
            continue;
        }
        auto& clonedProducer = producerOp->CloneOperation(function, producerOp->GetIOperands(),
                                                          producerOp->GetOOperands());
        clonedProducer.UpdateSubgraphID(producerOp->GetSubgraphID());
        clonedProducer.SetCoreType(producerOp->GetCoreType());
        clonedProducer.SetSpan(producerOp->GetSpan());
        clonedProducer.SetScopeInfo(producerOp->GetScopeInfo());
        if (producerOp->GetOpAttribute() != nullptr) {
            clonedProducer.SetOpAttribute(producerOp->GetOpAttribute()->Clone());
        }
        clonedProducer.ReplaceOutput(clonedInput, input);
    }
    return clonedInput;
}

std::string ProcessAtomic::GetRmwAttrKey(AtomicRMWMode mode)
{
    switch (mode) {
        case AtomicRMWMode::ADD:
            return RMW_MODE_ATTR_ADD;
        case AtomicRMWMode::MAX:
            return RMW_MODE_ATTR_MAX;
        case AtomicRMWMode::MIN:
            return RMW_MODE_ATTR_MIN;
        default:
            return "";
    }
}

Status ProcessAtomic::ProcessContractProducer(Operation& producerOp, std::shared_ptr<LogicalTensor> rmwOut,
                                              AtomicRMWMode rmwMode, const std::vector<int64_t>& rmwOffset,
                                              const std::vector<SymbolicScalar>& rmwDynOffset)
{
    std::string rmwAttrKey = GetRmwAttrKey(rmwMode);
    if (CheckAndSetRmwAttr(producerOp, rmwMode, rmwAttrKey) != SUCCESS) {
        return FAILED;
    }

    producerOp.ReplaceOOperand(0, rmwOut);

    auto producerContractAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(producerOp.GetOpAttribute());
    if (producerContractAttr != nullptr &&
        AccumulateContractOffset(producerContractAttr, rmwOffset, rmwDynOffset) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessAtomic::MarkContractProducerAtomic(Operation& producerOp, AtomicRMWMode rmwMode,
                                                 const std::vector<int64_t>& rmwOffset,
                                                 const std::vector<SymbolicScalar>& rmwDynOffset)
{
    std::string rmwAttrKey = GetRmwAttrKey(rmwMode);
    if (CheckAndSetRmwAttr(producerOp, rmwMode, rmwAttrKey) != SUCCESS) {
        return FAILED;
    }
    auto producerAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(producerOp.GetOpAttribute());
    if (producerAttr != nullptr && AccumulateContractOffset(producerAttr, rmwOffset, rmwDynOffset) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessAtomic::CheckAndSetRmwAttr(Operation& producerOp, AtomicRMWMode rmwMode, const std::string& rmwAttrKey)
{
    bool hasAdd = producerOp.HasAttr(RMW_MODE_ATTR_ADD);
    bool hasMax = producerOp.HasAttr(RMW_MODE_ATTR_MAX);
    bool hasMin = producerOp.HasAttr(RMW_MODE_ATTR_MIN);
    if (!hasAdd && !hasMax && !hasMin) {
        producerOp.SetAttribute(rmwAttrKey, 1L);
        return SUCCESS;
    }

    bool attrConflict = (rmwMode == AtomicRMWMode::ADD && !hasAdd) || (rmwMode == AtomicRMWMode::MAX && !hasMax) ||
                        (rmwMode == AtomicRMWMode::MIN && !hasMin);

    if (attrConflict) {
        std::string existingAttrType;
        if (hasAdd)
            existingAttrType = "atomic_add";
        else if (hasMax)
            existingAttrType = "atomic_max";
        else if (hasMin)
            existingAttrType = "atomic_min";

        APASS_LOG_ERROR_F(Elements::Operation,
                          "Op[%d] rmwMode conflict: producer contract op already has '%s' attribute, "
                          "but current wants to set '%s'. Cannot set different rmwMode to the same contract op.",
                          producerOp.GetOpMagic(), existingAttrType.c_str(), rmwAttrKey.c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessAtomic::AccumulateContractOffset(std::shared_ptr<AssembleOpAttribute> producerAttr,
                                               const std::vector<int64_t>& rmwOffset,
                                               const std::vector<SymbolicScalar>& rmwDynOffset)
{
    auto& producerOffset = producerAttr->GetToOffset();
    auto& producerDynOffset = producerAttr->GetToDynOffset();
    if (producerOffset.size() != rmwOffset.size() ||
        (!producerDynOffset.empty() && producerDynOffset.size() != producerOffset.size()) ||
        (!rmwDynOffset.empty() && rmwDynOffset.size() != rmwOffset.size())) {
        return FAILED;
    }

    auto originalOffset = producerOffset;
    if (!producerDynOffset.empty() || !rmwDynOffset.empty()) {
        auto lhs = producerDynOffset.empty() ? SymbolicScalar::FromConcrete(originalOffset) : producerDynOffset;
        auto rhs = rmwDynOffset.empty() ? SymbolicScalar::FromConcrete(rmwOffset) : rmwDynOffset;
        producerDynOffset.clear();
        producerDynOffset.reserve(producerOffset.size());
        for (size_t i = 0; i < producerOffset.size(); ++i) {
            producerDynOffset.push_back((lhs[i] + rhs[i]).Simplify());
        }
    }
    for (size_t i = 0; i < producerOffset.size(); ++i) {
        producerOffset[i] += rmwOffset[i];
    }
    return SUCCESS;
}

void ProcessAtomic::CollectReduceAccUpstream(Operation& op, std::set<int>& visited,
                                             std::vector<Operation*>& result) const
{
    if (visited.count(op.GetOpMagic()) > 0 || op.IsDeleted()) {
        return;
    }
    visited.insert(op.GetOpMagic());
    if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
        result.push_back(&op);
        return;
    }
    for (const auto& input : op.GetIOperands()) {
        if (input == nullptr) {
            continue;
        }
        for (auto* producer : input->GetProducers()) {
            if (producer == nullptr || producer->IsDeleted() || producer->GetOpcode() == Opcode::OP_ATOMIC_RMW) {
                continue;
            }
            CollectReduceAccUpstream(*producer, visited, result);
        }
    }
}

Status ProcessAtomic::TraceBackAndRemoveVecDup(Function& function, Operation& op, std::set<int>& visited,
                                               bool& anyRemoved)
{
    if (visited.count(op.GetOpMagic()) > 0 || op.IsDeleted()) {
        return SUCCESS;
    }
    if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
        return SUCCESS;
    }
    visited.insert(op.GetOpMagic());

    if (op.GetOpcode() == Opcode::OP_A_MUL_B || op.GetOpcode() == Opcode::OP_A_MULACC_B) {
        if (RemoveVecDupBranchFromCubeOp(op, anyRemoved) != SUCCESS) {
            return FAILED;
        }
    }

    for (const auto& input : op.GetIOperands()) {
        if (input == nullptr) {
            continue;
        }
        for (auto* producer : input->GetProducers()) {
            if (producer == nullptr || producer->IsDeleted()) {
                continue;
            }
            if (TraceBackAndRemoveVecDup(function, *producer, visited, anyRemoved) != SUCCESS) {
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status ProcessAtomic::RemoveVecDupBranchFromCubeOp(Operation& cubeOp, bool& anyRemoved)
{
    auto inputsBackup = cubeOp.GetIOperands();
    for (const auto& input : inputsBackup) {
        auto producersBackup = input->GetProducers();
        for (auto* producer : producersBackup) {
            if (producer == nullptr || producer->IsDeleted()) {
                continue;
            }
            if (IsContractOpcode(producer->GetOpcode()) && IsVecDupContractInput(*producer)) {
                input->RemoveConsumer(&cubeOp);
                cubeOp.EraseInput(input);
                anyRemoved = true;
                break;
            }
        }
    }
    return SUCCESS;
}

bool ProcessAtomic::IsVecDupContractInput(const Operation& contractOp) const
{
    for (const auto& input : contractOp.GetIOperands()) {
        for (auto* producer : input->GetProducers()) {
            if (producer != nullptr && producer->GetOpcode() == Opcode::OP_VEC_DUP) {
                return true;
            }
        }
    }
    return false;
}

Status ProcessAtomic::EliminateVecDupBranch(Function& function, bool& hasReduceAccCascade)
{
    std::vector<Operation*> reduceAccOps;
    std::set<int> collectVisited;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            CollectReduceAccUpstream(op, collectVisited, reduceAccOps);
        }
    }
    hasReduceAccCascade = !reduceAccOps.empty();
    if (reduceAccOps.empty()) {
        return SUCCESS;
    }

    bool anyRemoved = false;
    std::set<int> traceVisited;
    for (auto* reduceAccOp : reduceAccOps) {
        if (TraceBackAndRemoveVecDup(function, *reduceAccOp, traceVisited, anyRemoved) != SUCCESS) {
            return FAILED;
        }
    }

    if (!anyRemoved) {
        return SUCCESS;
    }
    APASS_LOG_INFO_F(Elements::Function, "EliminateVecDupBranch removed VecDup contract input branch.");
    function.EraseOperations(true);
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate dead operation failed for VecDup branch.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
