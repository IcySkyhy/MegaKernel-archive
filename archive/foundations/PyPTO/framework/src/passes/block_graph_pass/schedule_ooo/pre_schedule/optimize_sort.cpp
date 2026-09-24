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
 * \file optimize_sort.cpp
 * \brief
 */

#include "optimize_sort.h"

#include <queue>

#include "prior_dfs_sort.h"
#include "cluster_list_sort.h"
#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {
static constexpr size_t invalidIndex = std::numeric_limits<size_t>::max();

bool OptimizeSort::HasDependency(Operation* rollBackOp, Operation* backOp)
{
    std::map<Operation*, bool> visited;
    for (auto op : operations) {
        visited[op] = false;
    }
    std::function<bool(Operation*)> dfs = [&](Operation* op) -> bool {
        if (op == backOp)
            return true;
        if (visited[op])
            return false;

        visited[op] = true;
        for (auto succOp : state_.depManager.GetSuccessors(op)) {
            if (dfs(succOp)) {
                return true;
            }
        }
        return false;
    };
    return dfs(rollBackOp);
}

std::shared_ptr<std::vector<Operation*>> OptimizeSort::ReplaceIndex(std::shared_ptr<std::vector<Operation*>> curOpList,
                                                                    std::set<size_t>& advanceIndexList,
                                                                    size_t rollBackIndex)
{
    std::vector<Operation*> moveOpList;
    for (auto i : advanceIndexList) {
        APASS_LOG_DEBUG_F(Elements::Operation, "advance index: %zu, op: %s", i,
                          state_.GetOpInfo((*curOpList)[i]).c_str());
        moveOpList.push_back((*curOpList)[i]);
    }
    auto copyCurOpList = std::make_shared<std::vector<Operation*>>(*curOpList);
    for (auto it = advanceIndexList.rbegin(); it != advanceIndexList.rend(); ++it) {
        copyCurOpList->erase(copyCurOpList->begin() + (*it));
    }
    copyCurOpList->insert(copyCurOpList->begin() + rollBackIndex, moveOpList.begin(), moveOpList.end());
    return copyCurOpList;
}

void OptimizeSort::GetPreNode(size_t i, std::shared_ptr<std::vector<Operation*>> curOpList, size_t rollBackIndex,
                              size_t backTraceIndex, std::set<size_t>& dependencyIndexList)
{
    dependencyIndexList.insert(i);
    APASS_LOG_DEBUG_F(Elements::Operation, "dependencyIndexList push index: %zu, Op: %s", i,
                      state_.GetOpInfo((*curOpList)[i]).c_str());
    for (auto preOp : state_.depManager.GetPredecessors((*curOpList)[i])) {
        auto it = std::find(curOpList->begin() + rollBackIndex + 1, curOpList->begin() + backTraceIndex, preOp);
        if (it != curOpList->begin() + backTraceIndex) {
            auto index = std::distance(curOpList->begin(), it);
            GetPreNode(index, curOpList, rollBackIndex, backTraceIndex, dependencyIndexList);
        }
    }
}

void OptimizeSort::GetListToAdvance(size_t rollBackIndex, size_t backTraceIndex,
                                    std::shared_ptr<std::vector<Operation*>> curOpList,
                                    std::set<size_t>& advanceIndexList)
{
    std::set<size_t> dependencyIndexList;
    for (size_t i = rollBackIndex + 1; i <= backTraceIndex; i++) {
        if (HasDependency((*curOpList)[rollBackIndex], (*curOpList)[i])) {
            GetPreNode(i, curOpList, rollBackIndex, backTraceIndex, dependencyIndexList);
        }
    }
    for (size_t i = rollBackIndex + 1; i <= backTraceIndex; i++) {
        if (dependencyIndexList.count(i) == 0) {
            advanceIndexList.insert(i);
            APASS_LOG_DEBUG_F(Elements::Operation, "advanceIndexList push index: %zu, op: %s", i,
                              state_.GetOpInfo((*curOpList)[i]).c_str());
        }
    }
}

// rollBackIndex 位置回退
Status OptimizeSort::RollBack(size_t& startIndex, std::shared_ptr<std::vector<Operation*>>& curOpList,
                              std::map<MemoryType, int64_t>& curMemoryMap)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "=====> Start RollBack.");
    curOpList = backTraceOpList_[backTraceOp_].second;
    MemoryType memType = recordOpBuffer_[backTraceOp_];
    size_t backTraceIndex = backTraceOpList_[backTraceOp_].first + 1;
    backTraceOp_ = (*curOpList)[backTraceIndex];
    size_t rollBackIndex = backTraceIndex;
    APASS_LOG_DEBUG_F(Elements::Operation, "backTraceOp_: %s, backTraceIndex: %zu, memType: %d",
                      state_.GetOpInfo(backTraceOp_).c_str(), backTraceIndex, memType);
    std::set<size_t> advanceIndexList;
    while (rollBackIndex < curOpList->size() && rollBackIndex > 0) {
        rollBackIndex--;
        Operation* rollBackOp = (*curOpList)[rollBackIndex];
        if (recordOpBuffer_[rollBackOp] != memType || !(state_.IsOpAlloc(rollBackOp)) ||
            HasDependency(rollBackOp, backTraceOp_)) {
            continue;
        }
        rollBackNodeOp_ = rollBackOp;
        APASS_LOG_DEBUG_F(Elements::Operation, "Select rollBackOp: %s, rollBackIndex: %zu",
                          state_.GetOpInfo(rollBackOp).c_str(), rollBackIndex);
        recordBufferAllocate_ = backTraceBufferAllocate_;
        recordOpList_ = backTraceOpList_;
        recordBufRefCount_ = backTraceBufRefCount_;
        advanceIndexList.clear();
        GetListToAdvance(rollBackIndex, backTraceIndex, curOpList, advanceIndexList);
        curOpList = ReplaceIndex(curOpList, advanceIndexList, rollBackIndex);
        startIndex = rollBackIndex;
        APASS_LOG_DEBUG_F(Elements::Operation, "RollBack==>change startIndex: %zu", startIndex);
        if (rollBackIndex != 0) {
            curMemoryMap = recordBufferAllocate_[(*curOpList)[rollBackIndex - 1]];
            RecoverSymbol(startIndex - 1, curOpList);
            return SUCCESS;
        }
        curMemoryMap = {{MemoryType::MEM_L0A, 0}, {MemoryType::MEM_L0B, 0}, {MemoryType::MEM_L0C, 0}};
        operations = *curOpList;
        for (auto op : (*curOpList)) {
            visitedOp_[op] = false;
        }
        if (state_.InitBufRefCount(operations) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed at RollBack!");
            return FAILED;
        }
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "RollBack Failed");
    return FAILED;
}

std::shared_ptr<std::vector<Operation*>> OptimizeSort::ReorderOp(std::vector<size_t>& preIdx,
                                                                 std::shared_ptr<std::vector<Operation*>> curOpList,
                                                                 size_t startIndex)
{
    std::sort(preIdx.begin(), preIdx.end());
    std::vector<Operation*> moveOpList;
    APASS_LOG_DEBUG_F(Elements::Operation, "current index: %zu, preIdx size: %zu", startIndex, preIdx.size());
    for (auto i : preIdx) {
        APASS_LOG_DEBUG_F(Elements::Operation, "preidx : %zu, curOp: %s", i, state_.GetOpInfo((*curOpList)[i]).c_str());
        moveOpList.push_back((*curOpList)[i]);
    }
    auto copyCurOpList = std::make_shared<std::vector<Operation*>>(*curOpList);
    for (auto it = preIdx.rbegin(); it != preIdx.rend(); ++it) {
        copyCurOpList->erase(copyCurOpList->begin() + (*it));
    }
    copyCurOpList->insert(copyCurOpList->begin() + startIndex + 1, moveOpList.begin(), moveOpList.end());
    return copyCurOpList;
}

void OptimizeSort::FindIndex(Operation* op, std::shared_ptr<std::vector<Operation*>> curOpList, size_t& index)
{
    for (size_t i = 0; i < curOpList->size(); i++) {
        if ((*curOpList)[i] == op) {
            index = i;
            return;
        }
    }
}

Status OptimizeSort::FindConsumerList(size_t consumerIndex, std::vector<size_t>& preOpList,
                                      std::shared_ptr<std::vector<Operation*>> curOpList)
{
    if ((*curOpList)[consumerIndex] == backTraceOp_) {
        APASS_LOG_WARN_F(Elements::Operation, "backTraceOp_ is one of the predecessor node.");
        return FAILED;
    }
    if ((*curOpList)[consumerIndex] == rollBackNodeOp_) {
        APASS_LOG_WARN_F(Elements::Operation, "rollBackNodeOp_ is one of the predecessor node.");
        return FAILED;
    }
    visitedOp_[(*curOpList)[consumerIndex]] = true;
    preOpList.push_back(consumerIndex);
    APASS_LOG_DEBUG_F(Elements::Operation, "unvisited consumer idx: %zu, op: %s", consumerIndex,
                      state_.GetOpInfo((*curOpList)[consumerIndex]).c_str());
    for (auto op : state_.depManager.GetPredecessors((*curOpList)[consumerIndex])) {
        if (visitedOp_[op] == false) {
            size_t index;
            FindIndex(op, curOpList, index);
            APASS_LOG_DEBUG_F(Elements::Operation, "consumer preIdx: %zu, op: %s", index, state_.GetOpInfo(op).c_str());
            if (FindConsumerList(index, preOpList, curOpList) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "FindConsumerList failed");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OptimizeSort::UpdateOOperandPreDependence(size_t startIndex, std::shared_ptr<std::vector<Operation*>>& curOpList,
                                                 std::vector<Operation*> consumersGroup)
{
    // curOpList 中向后找
    std::vector<size_t> preOpList;
    size_t index = startIndex;
    while (index < curOpList->size()) {
        if (std::find(consumersGroup.begin(), consumersGroup.end(), (*curOpList)[index]) != consumersGroup.end()) {
            APASS_LOG_DEBUG_F(Elements::Operation, "consumer Idx: %zu", index);
            if (FindConsumerList(index, preOpList, curOpList) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "FindConsumerList failed");
                return FAILED;
            }
        }
        index++;
    }
    curOpList = ReorderOp(preOpList, curOpList, startIndex);
    return SUCCESS;
}

Status OptimizeSort::ConsumeOpBuffers(Operation* op)
{
    for (auto memId : state_.GetOpMemIds(op)) {
        if (state_.DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor[%d] failed.", memId);
            return FAILED;
        }
    }
    return SUCCESS;
}

void OptimizeSort::RecoverSymbol(size_t startIndex, std::shared_ptr<std::vector<Operation*>> curOpList)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "RecoverSymbol  startIdx: %zu, curOp: %s", startIndex,
                      state_.GetOpInfo((*curOpList)[startIndex]).c_str());
    Operation* targetOp = (*curOpList)[startIndex];

    bool hasBaseSnapshot = false;
    size_t baseIndex = startIndex;
    auto targetIt = recordBufRefCount_.find(targetOp);
    if (targetIt != recordBufRefCount_.end()) {
        state_.bufRefCount = targetIt->second;
        hasBaseSnapshot = true;
    } else {
        for (size_t i = startIndex; i > 0; --i) {
            size_t idx = i - 1;
            Operation* op = (*curOpList)[idx];
            auto it = recordBufRefCount_.find(op);
            if (it != recordBufRefCount_.end()) {
                state_.bufRefCount = it->second;
                baseIndex = idx;
                hasBaseSnapshot = true;
                break;
            }
        }
    }

    if (!hasBaseSnapshot) {
        state_.bufRefCount = initBufRefCountCache_;
        baseIndex = invalidIndex;
    }

    size_t replayStart = (baseIndex == invalidIndex) ? 0 : baseIndex + 1;
    for (size_t i = replayStart; i <= startIndex && i < curOpList->size(); ++i) {
        if (ConsumeOpBuffers((*curOpList)[i]) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RecoverSymbol replay failed at index %zu.", i);
            return;
        }
    }

    for (size_t i = 0; i < curOpList->size(); ++i) {
        Operation* op = (*curOpList)[i];
        if (i <= startIndex) {
            visitedOp_[op] = true;
        } else {
            visitedOp_[op] = false;
        }
    }
}

// 找未被执行的 consumer
void OptimizeSort::GetConsumerGroup(std::set<Operation*, Operation::OperationComparator>& consumers,
                                    std::vector<Operation*>& consumersGroup)
{
    for (auto op : consumers) {
        APASS_LOG_DEBUG_F(Elements::Operation, "consumer: %s", state_.GetOpInfo(op).c_str());
        if (!visitedOp_[op]) {
            consumersGroup.push_back(op);
            APASS_LOG_DEBUG_F(Elements::Operation, "unvisited consumer: %s", state_.GetOpInfo(op).c_str());
        }
    }
}

void OptimizeSort::GetStackTop(size_t& startIndex, std::shared_ptr<std::vector<Operation*>>& curOpList,
                               std::map<MemoryType, int64_t>& curMemoryMap)
{
    auto topNode = needFreeOpStack_.top();
    needFreeOpStack_.pop();
    curOpList = recordOpList_[topNode.first].second;
    startIndex = recordOpList_[topNode.first].first;
    curMemoryMap = recordBufferAllocate_[topNode.first];
}

Status OptimizeSort::BacktraceOnMemoryExceeded(size_t& startIndex, std::shared_ptr<std::vector<Operation*>>& curOpList,
                                               std::map<MemoryType, int64_t>& curMemoryMap)
{
    APASS_LOG_DEBUG_F(Elements::Tensor, "=====> Start Backtrace.");
    MemoryType memType = (*curOpList)[startIndex]->GetOutputOperand(0)->GetMemoryTypeOriginal();
    std::vector<Operation*> consumersGroup;
    while (startIndex < curOpList->size() && startIndex > 0) {
        startIndex--;
        auto op = (*curOpList)[startIndex];
        if (!needFreeOpStack_.empty() && needFreeOpStack_.top().first == (*curOpList)[startIndex]) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Having traversed %s, the stack needs to be popped",
                              state_.GetOpInfo((*curOpList)[startIndex]).c_str());
            break;
        }
        if (recordOpBuffer_[op] != memType || state_.IsOpAlloc(op)) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "===>start to find unvisited consumer, current index: %zu", startIndex);
        consumersGroup.clear();
        GetConsumerGroup(state_.depManager.GetSuccessors(op), consumersGroup);
        if (consumersGroup.empty()) {
            continue;
        }
        RecoverSymbol(startIndex, curOpList);
        GetConsumerGroup(state_.depManager.GetSuccessors(op), consumersGroup);
        APASS_LOG_DEBUG_F(Elements::Operation, "push %s to stack", state_.GetOpInfo(op).c_str());
        curMemoryMap = recordBufferAllocate_[op];
        recordBufRefCount_[op] = state_.bufRefCount;
        needFreeOpStack_.push(make_pair(op, recordOpBuffer_[op]));
        if (UpdateOOperandPreDependence(startIndex, curOpList, consumersGroup) != SUCCESS) {
            needFreeOpStack_.pop();
            APASS_LOG_DEBUG_F(Elements::Operation, "UpdateOOperandPreDependence failed.");
            continue;
        }
        startIndex++;
        APASS_LOG_DEBUG_F(Elements::Operation, "Backtrace==>change startIndex: %zu", startIndex);
        return SUCCESS;
    }
    if (needFreeOpStack_.empty()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Stack is empty. Start to rollback.");
        return FAILED;
    }
    GetStackTop(startIndex, curOpList, curMemoryMap);
    RecoverSymbol(startIndex, curOpList);
    APASS_LOG_DEBUG_F(Elements::Operation, "pop %s from stack", state_.GetOpInfo((*curOpList)[startIndex]).c_str());
    if (BacktraceOnMemoryExceeded(startIndex, curOpList, curMemoryMap) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "BacktraceOnMemoryExceeded Failed");
        return FAILED;
    }
    return SUCCESS;
}

bool OptimizeSort::IsBufferFull(std::map<MemoryType, int64_t> curMemoryMap, MemoryType memType, int64_t size)
{
    if (memType != MemoryType::MEM_L0A && memType != MemoryType::MEM_L0B && memType != MemoryType::MEM_L0C) {
        APASS_LOG_DEBUG_F(Elements::Operation, "MemoryType is not L0A, L0B, or L0C.");
        return false;
    }
    if (curMemoryMap[memType] + size > state_.localMemSize[memType]) {
        APASS_LOG_DEBUG_F(Elements::Operation, "The %d-memType memory is full, current memory: %ld, memory to add: %ld",
                          memType, static_cast<long>(curMemoryMap[memType]), static_cast<long>(size));
        return true;
    }
    return false;
}

// 修改内存
Status OptimizeSort::ModifyBuffer(std::map<MemoryType, int64_t>& curMemoryMap, MemoryType memType, int64_t size,
                                  bool isAdd)
{
    if (memType != MemoryType::MEM_L0A && memType != MemoryType::MEM_L0B && memType != MemoryType::MEM_L0C) {
        APASS_LOG_DEBUG_F(Elements::Operation, "MemoryType is not L0A, L0B, or L0C.");
        return SUCCESS;
    }
    if (isAdd) {
        if (curMemoryMap[memType] + size > state_.localMemSize[memType]) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Failed to increase memory");
            return FAILED;
        }
        curMemoryMap[memType] = curMemoryMap[memType] + size;
        APASS_LOG_DEBUG_F(Elements::Operation, "Increase %d-memType memory, size: %ld bytes, total memory %ld bytes",
                          memType, static_cast<long>(size), static_cast<long>(curMemoryMap[memType]));
        return SUCCESS;
    }
    if (curMemoryMap[memType] - size < 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Failed to reduce memory");
        return FAILED;
    }
    curMemoryMap[memType] = curMemoryMap[memType] - size;
    APASS_LOG_DEBUG_F(Elements::Operation, "Reduce %d-memType memory, size: %ld bytes, total memory %ld bytes", memType,
                      static_cast<long>(size), static_cast<long>(curMemoryMap[memType]));
    return SUCCESS;
}

// 释放内存 notTaskOp需要减去bufRefCount
Status OptimizeSort::RetireOpBuffer(std::map<MemoryType, int64_t>& curMemoryMap, Operation* op)
{
    for (auto tensor : state_.GetInOutOperandCached(op)) {
        auto memId = tensor->memoryrange.memId;
        if (state_.DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor[%d] failed.", memId);
            return FAILED;
        }
        if (state_.bufRefCount[memId] == 0) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Start to free memory:");
            if (ModifyBuffer(curMemoryMap, tensor->GetMemoryTypeOriginal(),
                             state_.ShapeCeilAlign(tensor->tensor->rawshape, tensor->Datatype()), false) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor[%d] failed.", memId);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

void OptimizeSort::OpMemoryUpdate(Operation* op, size_t startIndex, std::shared_ptr<std::vector<Operation*>> curOpList,
                                  const std::map<MemoryType, int64_t>& curMemoryMap)
{
    recordOpList_[op] = make_pair(startIndex, curOpList);
    recordBufferAllocate_[op] = curMemoryMap;
    recordOpBuffer_[op] = op->GetOutputOperand(0)->GetMemoryTypeOriginal();
}

Status OptimizeSort::AllocExecute(Operation* op, std::shared_ptr<std::vector<Operation*>>& curOpList,
                                  std::map<MemoryType, int64_t>& curMemoryMap, size_t& startIndex, bool& isContinue)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "alloc op: %s", state_.GetOpInfo(op).c_str());
    auto tensor = op->GetOutputOperand(0);
    if (IsBufferFull(curMemoryMap, tensor->GetMemoryTypeOriginal(),
                     state_.ShapeCeilAlign(tensor->GetShape(), tensor->Datatype()))) {
        APASS_LOG_DEBUG_F(Elements::Operation, "The memory of %s needs to be released",
                          std::to_string(tensor->GetMemoryTypeOriginal()).c_str());
        backTraceOp_ = (*curOpList)[startIndex];
        backTraceBufferAllocate_ = recordBufferAllocate_;
        backTraceOpList_ = recordOpList_;
        if (startIndex >= 1) {
            recordBufRefCount_[(*curOpList)[startIndex - 1]] = state_.bufRefCount;
        }
        backTraceBufRefCount_ = recordBufRefCount_;
        APASS_LOG_DEBUG_F(Elements::Operation, "backTraceOp_: %s, backTraceIndex: %zu, memType: %d",
                          state_.GetOpInfo(backTraceOp_).c_str(), backTraceOpList_[backTraceOp_].first,
                          static_cast<int>(recordOpBuffer_[backTraceOp_]));
        APASS_LOG_DEBUG_F(Elements::Operation, "=====> Need backtrace.");
        if (BacktraceOnMemoryExceeded(startIndex, curOpList, curMemoryMap) != SUCCESS) {
            if (RollBack(startIndex, curOpList, curMemoryMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "AllocExecute failed.");
                return FAILED;
            }
            isContinue = true;
            return SUCCESS;
        }
        isContinue = true;
        return SUCCESS;
    }
    return SUCCESS;
}

Status OptimizeSort::OpListExecute(std::shared_ptr<std::vector<Operation*>>& curOpList,
                                   std::map<MemoryType, int64_t>& curMemoryMap, size_t& startIndex)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "===>Start opListExecute, startIndex: %zu", startIndex);
    if (curOpList->empty()) {
        curOpList = std::make_shared<std::vector<Operation*>>(operations);
    }
    while (startIndex < curOpList->size()) {
        auto op = (*curOpList)[startIndex];
        OpMemoryUpdate(op, startIndex, curOpList, curMemoryMap);
        APASS_LOG_DEBUG_F(Elements::Operation, "execute op: %s, index: %zu", state_.GetOpInfo(op).c_str(), startIndex);
        if (state_.IsOpAlloc(op)) {
            bool isContinue = false;
            if (AllocExecute(op, curOpList, curMemoryMap, startIndex, isContinue) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "AllocExecute failed.");
                return FAILED;
            }
            if (isContinue) {
                return SUCCESS;
            }
            auto tensor = op->GetOutputOperand(0);
            if (ModifyBuffer(curMemoryMap, tensor->GetMemoryTypeOriginal(),
                             state_.ShapeCeilAlign(tensor->tensor->rawshape, tensor->Datatype()), true) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Allocate tensor[%d] failed.", tensor->GetMagic());
                return FAILED;
            }
            OpMemoryUpdate(op, startIndex, curOpList, curMemoryMap);
        }
        visitedOp_[op] = true;
        if (RetireOpBuffer(curMemoryMap, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireOp failed! %s", state_.GetOpInfo(op).c_str());
            return FAILED;
        }
        OpMemoryUpdate(op, startIndex, curOpList, curMemoryMap);
        startIndex += 1;
    }
    opFinish_ = true;
    return SUCCESS;
}

Status OptimizeSort::ExecuteOp()
{
    auto curOpList = std::make_shared<std::vector<Operation*>>();
    std::map<MemoryType, int64_t> curMemoryMap = {
        {MemoryType::MEM_L0A, 0}, {MemoryType::MEM_L0B, 0}, {MemoryType::MEM_L0C, 0}};
    size_t startIndex{0};
    for (auto& op : operations) {
        visitedOp_[op] = false;
    }
    initBufRefCountCache_ = state_.bufRefCount;
    while (!opFinish_) {
        if (OpListExecute(curOpList, curMemoryMap, startIndex) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "OpListExecute failed.");
            return FAILED;
        }
    }
    operations = *curOpList;
    return SUCCESS;
}

void OptimizeSort::AllocAhead()
{
    std::vector<Operation*> allocOps;
    std::vector<Operation*> normalOps;
    for (auto& op : operations) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            allocOps.emplace_back(op);
            continue;
        }
        normalOps.emplace_back(op);
    }
    std::vector<Operation*> newOperations;
    std::reverse(allocOps.begin(), allocOps.end());
    newOperations.swap(allocOps);
    newOperations.insert(newOperations.end(), normalOps.begin(), normalOps.end());
    operations = newOperations;
}

const std::unordered_map<std::string, OptimizeSort::Factory> OptimizeSort::SORT_ALGOS = {
    {"PriorDFS", [](std::vector<Operation*> ops, Function& f) { return std::make_unique<PriorDFSSort>(ops, f); }},
    {"ClusterList", [](std::vector<Operation*> ops, Function& f) { return std::make_unique<ClusterListSort>(ops, f); }},
};

std::string OptimizeSort::ResolveOooSortMode(const std::vector<Operation*>& ops, const ParamConfigs& pc)
{
    bool hasAic = false;
    bool hasAiv = false;
    for (auto* op : ops) {
        auto coreType = OpcodeManager::Inst().GetCoreType(op->GetOpcode());
        if (coreType == OpCoreType::AIC) {
            hasAic = true;
        } else if (coreType == OpCoreType::AIV) {
            hasAiv = true;
        }
    }
    if (hasAic && !hasAiv) {
        return pc.oooSortModeAic.empty() ? "PriorDFS" : pc.oooSortModeAic;
    }
    if (hasAiv && !hasAic) {
        return pc.oooSortModeAiv.empty() ? "PriorDFS" : pc.oooSortModeAiv;
    }
    return "PriorDFS";
}

std::unique_ptr<OptimizeSort> OptimizeSort::Create(std::vector<Operation*> ops, Function& func, const std::string& mode)
{
    auto it = SORT_ALGOS.find(mode);
    if (it == SORT_ALGOS.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Unknown sort mode: %s.", mode.c_str());
        return nullptr;
    }
    return it->second(ops, func);
}

Status OptimizeSort::SortOps()
{
    auto mode = ResolveOooSortMode(operations, function_.paramConfigs_);
    auto sorter = Create(operations, function_, mode);
    if (sorter == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to create sorter for mode: %s.", mode.c_str());
        return FAILED;
    }
    LOG_SCOPE_BEGIN(tSortOps, Elements::Function, "SortOps");
    sorter->state_.Init(sorter->operations);
    if (sorter->operations.empty()) {
        return SUCCESS;
    }
    Status result = sorter->DoSortOps();
    LOG_SCOPE_END(tSortOps);
    if (result == SUCCESS) {
        operations = sorter->GetOperations();
    }
    return result;
}

} // namespace npu::tile_fwk
