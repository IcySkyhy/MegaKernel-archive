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
 * \file dep_manager.cpp
 * \brief Dependency manager implementation
 */

#include "passes/block_graph_pass/schedule_ooo/common/dep_manager.h"
#include "passes/pass_log/pass_log.h"

#ifndef MODULE_NAME
#define MODULE_NAME "DependencyManager"
#endif

namespace npu::tile_fwk {

void DependencyManager::RegisterOp(Operation* op)
{
    if (op == nullptr) {
        return;
    }
    if (inGraph_.find(op) == inGraph_.end()) {
        inGraph_[op] = std::set<Operation*, Operation::OperationComparator>();
    }
    if (outGraph_.find(op) == outGraph_.end()) {
        outGraph_[op] = std::set<Operation*, Operation::OperationComparator>();
    }
}

void DependencyManager::Clear()
{
    inGraph_.clear();
    outGraph_.clear();
}

void DependencyManager::ClearDependencies()
{
    for (auto& [op, preds] : inGraph_) {
        (void)op;
        preds.clear();
    }
    for (auto& [op, succs] : outGraph_) {
        (void)op;
        succs.clear();
    }
}

bool DependencyManager::IsOpAlloc(Operation* op)
{
    if (op == nullptr) {
        return false;
    }
    return op->GetOpcodeStr().find("ALLOC") != std::string::npos;
}

void DependencyManager::AddDependency(Operation* preOp, Operation* postOp)
{
    if (preOp == nullptr || postOp == nullptr) {
        return;
    }
    if (!IsOpAlloc(preOp) && !IsOpAlloc(postOp)) {
        outGraph_[preOp].insert(postOp);
        inGraph_[postOp].insert(preOp);
    }
}

void DependencyManager::AddAllocDependency(Operation* preOp, Operation* postOp)
{
    if (preOp == nullptr || postOp == nullptr) {
        return;
    }
    outGraph_[preOp].insert(postOp);
    inGraph_[postOp].insert(preOp);
}

bool DependencyManager::RemoveDependency(Operation* preOp, Operation* postOp)
{
    if (preOp == nullptr || postOp == nullptr) {
        return false;
    }
    bool removedFromSucc = false;
    bool removedFromPred = false;

    if (outGraph_.find(preOp) != outGraph_.end()) {
        removedFromSucc = outGraph_[preOp].erase(postOp) > 0;
    }
    if (inGraph_.find(postOp) != inGraph_.end()) {
        removedFromPred = inGraph_[postOp].erase(preOp) > 0;
    }

    return removedFromSucc && removedFromPred;
}

int DependencyManager::InsertSuccessor(Operation* op, Operation* succ)
{
    if (op == nullptr || succ == nullptr) {
        return 0;
    }
    auto result = outGraph_[op].insert(succ);
    return result.second ? 1 : 0;
}

int DependencyManager::RemoveSuccessor(Operation* op, Operation* succ)
{
    if (op == nullptr || succ == nullptr) {
        return 0;
    }
    return outGraph_[op].erase(succ);
}

void DependencyManager::RemoveSuccessorOp(Operation* op) { opConsumers.erase(op); }

int DependencyManager::InsertPredecessor(Operation* op, Operation* pred)
{
    if (op == nullptr || pred == nullptr) {
        return 0;
    }
    auto result = inGraph_[op].insert(pred);
    return result.second ? 1 : 0;
}

int DependencyManager::RemovePredecessor(Operation* op, Operation* pred)
{
    if (op == nullptr || pred == nullptr) {
        return 0;
    }
    return inGraph_[op].erase(pred);
}

void DependencyManager::RemovePredecessorOp(Operation* op) { opProducers.erase(op); }

std::set<Operation*, Operation::OperationComparator>& DependencyManager::GetSuccessors(Operation* op)
{
    return outGraph_[op];
}

std::set<Operation*, Operation::OperationComparator>& DependencyManager::GetPredecessors(Operation* op)
{
    return inGraph_[op];
}

bool DependencyManager::HasOp(Operation* op) const { return inGraph_.find(op) != inGraph_.end(); }

std::string DependencyManager::PrintOp(Operation* op)
{
    return op->GetOpcodeStr() + "[" + std::to_string(op->GetOpMagic()) + "]";
}

void DependencyManager::PrintDependencies(const std::vector<Operation*>& ops)
{
    for (const auto& op : ops) {
        if (inGraph_.find(op) == inGraph_.end() || outGraph_.find(op) == outGraph_.end()) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "%s", PrintOp(op).c_str());
        for (const auto& preOp : inGraph_.at(op)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Predecessors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", PrintOp(preOp).c_str());
        }
        for (const auto& succOp : outGraph_.at(op)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Successors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", PrintOp(succOp).c_str());
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "\n");
    }
}

Status DependencyManager::InitAllocDependencies(Operation* op, std::unordered_map<int, Operation*>& tensor2AllocOpMap)
{
    for (auto& tensor : op->GetOOperands()) {
        int memId = tensor->memoryrange.memId;
        if (tensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            if (tensor2AllocOpMap.find(memId) == tensor2AllocOpMap.end()) {
                APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d] must have alloc. magic: %d, op: %s", memId,
                                  tensor->GetMagic(), PrintOp(op).c_str());
                return FAILED;
            }
            AddAllocDependency(tensor2AllocOpMap[memId], op);
        }
    }
    return SUCCESS;
}

void DependencyManager::HandleScaleOpDependency(Operation* op, MemoryType memType)
{
    auto matmulOp = *(op->GetOutputOperand(0))->GetConsumers().begin();
    if (matmulOp == nullptr) {
        return;
    }
    for (auto& input : matmulOp->GetIOperands()) {
        if (input->GetMemoryTypeOriginal() == memType) {
            AddDependency(*input->GetProducers().begin(), op);
        }
    }
}

void DependencyManager::AddProducerDependencies(Operation* op)
{
    for (auto& producer : op->ProducerOps()) {
        Operation* farthestSkip = SkipChain(producer);
        if (farthestSkip == nullptr) {
            AddDependency(producer, op);
            continue;
        }
        for (auto* realProducer : farthestSkip->ProducerOps()) {
            AddDependency(realProducer, op);
        }
    }
}

void DependencyManager::FindDependencies(Operation* op, bool needView)
{
    if (op->GetOpcode() == Opcode::OP_L1_TO_L0A_SCALE) {
        HandleScaleOpDependency(op, MemoryType::MEM_L0A);
    }
    if (op->GetOpcode() == Opcode::OP_L1_TO_L0B_SCALE) {
        HandleScaleOpDependency(op, MemoryType::MEM_L0B);
    }

    if (needView) {
        for (auto& producer : opProducers[op]) {
            AddDependency(producer, op);
        }
        for (auto& consumer : opConsumers[op]) {
            AddDependency(op, consumer);
        }
        return;
    }

    AddProducerDependencies(op);
}

void DependencyManager::InitOpConsumerAndProducer(const std::vector<Operation*>& ops)
{
    opConsumers.clear();
    opProducers.clear();
    std::unordered_set<Operation*> opSet;
    for (auto op : ops) {
        opSet.insert(op);
    }
    for (auto op : ops) {
        for (auto consumer : op->ConsumerOps()) {
            if (opSet.find(consumer) != opSet.end()) {
                opConsumers[op].insert(consumer);
            }
        }
        for (auto producer : op->ProducerOps()) {
            if (opSet.find(producer) != opSet.end()) {
                opProducers[op].insert(producer);
            }
        }
    }
}

Status DependencyManager::InitDependencies(const std::vector<Operation*>& ops, bool needView)
{
    std::unordered_map<int, Operation*> tensor2AllocOpMap;
    InitOpConsumerAndProducer(ops);
    for (const auto& op : ops) {
        if (IsOpAlloc(op)) {
            if (op->GetOOperands().size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] oOperand must be 1.", op->GetOpMagic());
                return FAILED;
            }
            int memId = op->GetOutputOperand(0)->memoryrange.memId;
            tensor2AllocOpMap[memId] = op;
        }
    }

    // === DualDst 兼容: 融合后 OP_L0C_COPY_UB_DUAL_DST 的某个输出失去独立 ALLOC,
    // 把它的 memId 也指向同 op 兄弟输出的 ALLOC, 这样下游 consumer (含 in-place
    // 复用同 memId 的 ADDS 等) 在 InitAllocDependencies 里都能查到。
    for (const auto& op : ops) {
        if (op == nullptr || op->GetOpcode() != Opcode::OP_L0C_COPY_UB_DUAL_DST)
            continue;
        Operation* dualAlloc = nullptr;
        for (auto& outTensor : op->GetOOperands()) {
            auto it = tensor2AllocOpMap.find(outTensor->memoryrange.memId);
            if (it != tensor2AllocOpMap.end()) {
                dualAlloc = it->second;
                break;
            }
        }
        if (dualAlloc == nullptr)
            continue;
        for (auto& outTensor : op->GetOOperands()) {
            int mid = outTensor->memoryrange.memId;
            if (!tensor2AllocOpMap.count(mid)) {
                tensor2AllocOpMap[mid] = dualAlloc;
            }
        }
    }

    Clear();
    for (const auto& op : ops) {
        RegisterOp(op);
    }

    for (const auto& op : ops) {
        if (!IsOpAlloc(op)) {
            FindDependencies(op, needView);
            if (InitAllocDependencies(op, tensor2AllocOpMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "InitAllocDependencies failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
