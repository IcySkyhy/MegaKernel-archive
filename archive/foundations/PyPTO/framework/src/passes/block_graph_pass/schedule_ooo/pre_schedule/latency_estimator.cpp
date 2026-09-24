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
 * \file core_assign.cpp
 * \brief
 */

#include "passes/block_graph_pass/schedule_ooo/pre_schedule/latency_estimator.h"

namespace npu::tile_fwk {

void LatencyEstimator::LaunchReadyIssue()
{
    for (auto& op : taskList) {
        if (state_.IsOpAlloc(op)) {
            auto tensor = op->GetOOperands()[0];
            auto memId = tensor->memoryrange.memId;
            allocIssueQueue[state_.localBufferMap[memId]->memType].Insert(op);
        } else if (state_.depManager.GetPredecessors(op).empty()) {
            auto type = RescheduleUtils::GetOpPipeType(op);
            opQueues[type].Insert(op);
            if (USE_LESS_OPS.find(op->GetOpcode()) == USE_LESS_OPS.end()) {
                APASS_LOG_WARN_F(Elements::Operation, "%s[%d]'s predecessor is empty.", op->GetOpcodeStr().c_str(),
                                 op->GetOpMagic());
            }
        }
    }
}

Status LatencyEstimator::FreeBuffer(Operation* op)
{
    for (auto tensor : state_.GetInOutOperandCached(op)) {
        auto memId = tensor->memoryrange.memId;
        if (state_.DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor [%d] failed.", memId);
            return FAILED;
        }
        if (state_.bufRefCount[memId] == 0) {
            auto freeMemSize = state_.localBufferMap[memId]->size;
            if (spillblockMemIds.find(memId) == spillblockMemIds.end()) {
                state_.localMemoryCurrentSize[state_.localBufferMap[memId]->memType] += freeMemSize;
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "FreeBuffer memType: %d, currentSize %ld, memId: %d, freeMemSize: %lu.",
                    state_.localBufferMap[memId]->memType,
                    static_cast<long>(state_.localMemoryCurrentSize[state_.localBufferMap[memId]->memType]), memId,
                    static_cast<unsigned long>(freeMemSize));
            } else {
                APASS_LOG_DEBUG_F(Elements::Operation, "FreeBuffer memType: %d, memId: %d free in spillblock",
                                  state_.localBufferMap[memId]->memType, memId);
            }

            if (state_.localMemoryCurrentSize[state_.localBufferMap[memId]->memType] >
                    state_.localMemSize[state_.localBufferMap[memId]->memType] ||
                state_.localMemoryCurrentSize[state_.localBufferMap[memId]->memType] < 0) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor [%d] failed.", memId);
                return FAILED;
            }
            APASS_LOG_DEBUG_F(Elements::Tensor, "Free tensor [%d] success.", memId);
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt)
{
    commitCnt++;
    opRetiredInfo[op] = true;
    if (FreeBuffer(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FreeBuffer failed. %s", state_.GetOpInfo(op).c_str());
        return FAILED;
    }

    for (auto succ : state_.depManager.GetSuccessors(op)) {
        if (opRetiredInfo[succ]) {
            continue;
        }
        bool ready = true;
        for (auto pred : state_.depManager.GetPredecessors(succ)) {
            if (!opRetiredInfo[pred]) {
                ready = false;
                break;
            }
        }
        if (ready) {
            opQueues[RescheduleUtils::GetOpPipeType(succ)].Insert(succ);
            APASS_LOG_DEBUG_F(Elements::Operation, "Wakeup: %s", state_.GetOpInfo(succ).c_str());
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::RetireIssueStage(uint64_t& commitCnt, int& nextCycle)
{
    for (auto& [pipeType, pipe] : opQueues) {
        (void)pipeType;
        if (!pipe.busy) {
            continue;
        }
        if (pipe.curOpRetireCycle <= state_.clock) { // 如果该pipe内当前正在执行op，在clock的时刻已经执行完毕。
            Operation* op = pipe.curOp;
            pipe.busy = false;
            pipe.curOp = nullptr;
            APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTE END: %s", state_.GetOpInfo(op).c_str());
            if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed! %s", state_.GetOpInfo(op).c_str());
                return FAILED;
            }
        } else {
            APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTING[%d]: %s", pipe.curOpRetireCycle,
                              state_.GetOpInfo(pipe.curOp).c_str());
            if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                nextCycle = pipe.curOpRetireCycle;
            }
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::ExecuteAllocIssue(uint64_t& commitCnt, MemoryType memType, OpQueue& pipe)
{
    bool canAlloc = true;
    while (canAlloc) {
        if (pipe.Empty()) {
            canAlloc = false;
            break;
        }
        Operation* op = pipe.Front();
        auto memId = state_.GetInOutOperandCached(op)[0]->memoryrange.memId;
        auto needMemSize = state_.localBufferMap[memId]->size;
        if (state_.localMemoryCurrentSize[memType] >= static_cast<long int>(needMemSize)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "ALLOCATE: %s.", state_.GetOpInfo(op).c_str());
            state_.localMemoryCurrentSize[memType] -= needMemSize;
            APASS_LOG_DEBUG_F(Elements::Operation, "ExecuteAllocIssue memType: %d, currentSize %ld, memId: %d.",
                              memType, static_cast<long>(state_.localMemoryCurrentSize[memType]), memId);
            if (state_.localMemoryCurrentSize[memType] > state_.localMemSize[memType] ||
                state_.localMemoryCurrentSize[memType] < 0) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Allocate Tensor[%d] failed.",
                                  state_.GetInOutOperandCached(op)[0]->GetMagic());
                return FAILED;
            }
            pipe.PopFront();
            if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed. %s", state_.GetOpInfo(op).c_str());
                return FAILED;
            }
        } else {
            canAlloc = false;
            APASS_LOG_DEBUG_F(Elements::Tensor, "Cannot alloc Tensor[%d] ",
                              state_.GetInOutOperandCached(op)[0]->GetMagic());
            break;
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::BufferAllocStage(uint64_t& commitCnt)
{
    for (auto& [memoryType, pipe] : allocIssueQueue) {
        if (pipe.Empty()) {
            continue;
        }
        // 不断按顺序执行alloc指令，直到buffer被占满为止。
        if (ExecuteAllocIssue(commitCnt, memoryType, pipe) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::LaunchIssueStage(int& nextCycle)
{
    for (auto& [pipeType, pipe] : opQueues) {
        (void)pipeType;
        if (pipe.Empty() || pipe.busy) {
            continue;
        }
        Operation* op = pipe.PopFront();
        pipe.busy = true;
        pipe.curOp = op;
        pipe.curOpRetireCycle = state_.clock + op->GetLatency();
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }

        APASS_LOG_DEBUG_F(Elements::Operation, "opQueues Insert: %s.", state_.GetOpInfo(op).c_str());
    }
    return SUCCESS;
}

Status LatencyEstimator::SpillOnBlock()
{
    MemoryType spillMemType;
    if (!allocIssueQueue[MemoryType::MEM_UB].Empty()) {
        spillMemType = MemoryType::MEM_UB;
    } else if (!allocIssueQueue[MemoryType::MEM_L1].Empty()) {
        spillMemType = MemoryType::MEM_L1;
    } else {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Buffer[L0A/B/C] is Full. Please check tile shape and OOO spill failed info.");
        return FAILED;
    }

    Operation* op = allocIssueQueue[spillMemType].Front();
    size_t needMemSize = state_.GetInOutOperandCached(op)[0]->tensor->GetRawDataSize();
    spillblockMemIds.insert(state_.GetInOutOperandCached(op)[0]->memoryrange.memId);
    state_.localMemoryCurrentSize[spillMemType] += static_cast<long int>(needMemSize);
    if (state_.localMemoryCurrentSize[spillMemType] < 0 ||
        state_.localMemoryCurrentSize[spillMemType] > state_.localMemSize[spillMemType]) {
        APASS_LOG_ERROR_F(Elements::Operation, "Buffer[%d] is valid. Please check", spillMemType);
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "SpillOnBlock freeBuffer memType: %d, memId: %d, freeMemSize: %lu.",
                      spillMemType, state_.GetInOutOperandCached(op)[0]->memoryrange.memId,
                      static_cast<unsigned long>(needMemSize));
    return SUCCESS;
}

void LatencyEstimator::initLatencyEstimatorOpQueues()
{
    for (size_t i = 0; i <= static_cast<int>(PipeType::PIPE_FIX); i++) {
        opQueues[static_cast<PipeType>(i)] = OpQueue();
    }
}

void LatencyEstimator::InitMemWithoutAlloc()
{
    std::unordered_map<int, Operation*> allocMap;
    for (const auto& op : operations) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            allocMap[op->GetOutputOperand(0)->memoryrange.memId] = op;
        }
    }
    std::unordered_set<int> memIds;
    std::unordered_map<int, Operation*> memIdAllocMap;
    std::unordered_map<Operation*, std::unordered_set<int>> opMemidMap;
    for (const auto& op : taskList) {
        if (state_.IsOpAlloc(op)) {
            memIdAllocMap[op->GetOutputOperand(0)->memoryrange.memId] = op;
        }
        for (auto& iOperand : op->GetIOperands()) {
            if (iOperand->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                memIds.insert(iOperand->memoryrange.memId);
                opMemidMap[op].insert(iOperand->memoryrange.memId);
            }
        }
        for (auto& oOperand : op->GetOOperands()) {
            if (oOperand->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                memIds.insert(oOperand->memoryrange.memId);
                opMemidMap[op].insert(oOperand->memoryrange.memId);
            }
        }
    }
    for (const auto& memId : memIds) {
        if (memIdAllocMap.find(memId) != memIdAllocMap.end()) {
            continue;
        }
        APASS_LOG_INFO_F(Elements::Operation, "The alloc op of memId[%d] in other graph", memId);
        for (size_t i = 0; i < taskList.size(); ++i) {
            if (opMemidMap[taskList[i]].find(memId) == opMemidMap[taskList[i]].end()) {
                continue;
            }
            if (allocMap.find(memId) == allocMap.end()) {
                continue;
            }
            memIdAllocMap[memId] = allocMap[memId];
            taskList.insert(taskList.begin() + i, allocMap[memId]);
            APASS_LOG_INFO_F(Elements::Operation, "Add %s for memId[%d]", state_.GetOpInfo(allocMap[memId]).c_str(),
                             memId);
            break;
        }
    }
}

Status LatencyEstimator::LatencyEstimatorMainLoop() { return RunSchedulerMainLoop(*this); }

Status LatencyEstimator::PreMainLoop()
{
    initLatencyEstimatorOpQueues();
    LaunchReadyIssue();
    state_.numTotalIssues = taskList.size();
    state_.originalNumTotalIssues = state_.numTotalIssues;
    return SUCCESS;
}

Status LatencyEstimator::PostMainLoop()
{
    APASS_LOG_DEBUG_F(Elements::Operation, "\n Estimate Latency: %d", state_.clock);
    return SUCCESS;
}
} // namespace npu::tile_fwk
