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
 * \file ooo_scheduler.cpp
 * \brief
 */

#include "ooo_scheduler.h"

#include "buffer_rearrange.h"
#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {

inline std::string coreTypeToString(CoreLocationType coreLocation)
{
    switch (coreLocation) {
        case CoreLocationType::AIV0:
            return "AIV0";
        case CoreLocationType::AIV1:
            return "AIV1";
        case CoreLocationType::AIC:
            return "AIC";
        default:
            return "MEM_UNKNOWN";
    }
}

inline bool IsMixGraph(const std::vector<Operation*>& operations)
{
    bool hasAIC = false;
    bool hasAIV = false;
    for (auto opPtr : operations) {
        if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIV) {
            hasAIV = true;
        } else if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIC) {
            hasAIC = true;
        }
        if (hasAIC && hasAIV) {
            return true;
        }
    }
    return false;
}

void OoOScheduler::PrintOpList(std::vector<Operation*> opList)
{
    APASS_LOG_INFO_F(Elements::Operation, "====================OoO OP_LIST =====================");
    bool needMark = false;
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
        needMark = true;
    }
    for (auto& op : opList) {
        if (needMark) {
            bool isCubeComponent = op->HasAttribute(OpAttributeKey::isCube) &&
                                   op->GetBoolAttribute(OpAttributeKey::isCube);
            if (!isCubeComponent) {
                op->SetAIVCore(AIVCore::AIV0);
            }
        }
        if (!op->oOperand.empty()) {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d], range[%zu, %zu]", op->GetOpcodeStr().c_str(),
                             op->GetOpMagic(), op->oOperand[0]->memoryrange.start, op->oOperand[0]->memoryrange.end);
        } else {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d]", op->GetOpcodeStr().c_str(), op->GetOpMagic());
        }
    }
}

void OoOScheduler::UpdateIssueExecOrder()
{
    for (size_t idx = 0; idx < state_.orderedOps.size(); idx++) {
        state_.schedInfoMap[state_.orderedOps[idx]].execOrder = idx;
    }
}

Status OoOScheduler::CheckAndUpdateLifecycle()
{
    for (const auto& op : state_.orderedOps) {
        if (!state_.schedInfoMap[op].isRetired) {
            APASS_LOG_ERROR_F(Elements::Operation, "Unexecuted op: %s. %s", state_.GetOpInfo(op).c_str(),
                              GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        if (state_.schedInfoMap[op].isAlloc) {
            op->GetOutputOperand(0)->memoryrange.lifeStart = state_.localBufferMap[state_.GetOpMemIds(op)[0]]
                                                                 ->startCycle;
            op->GetOutputOperand(0)->memoryrange.lifeEnd = state_.localBufferMap[state_.GetOpMemIds(op)[0]]
                                                               ->retireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::SpillOnCoreBlock(std::pair<CoreLocationType, MemoryType> orderFirstPair)
{
    SpillContext ctx;
    if (GenBufferSpill(state_.allocIssueQueue[orderFirstPair.first][orderFirstPair.second].Front(), ctx) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at GenBufferSpill.");
        return FAILED;
    }
    if (ApplySpillContext(ctx, state_.allocIssueQueue[orderFirstPair.first][orderFirstPair.second].Front()) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ApplySpillContext failed.");
        return FAILED;
    }
    return SUCCESS;
}

bool OoOScheduler::IsOpInQueue(Operation* op, const OpQueue& pipe) const
{
    return std::find(pipe.queue.begin(), pipe.queue.end(), op) != pipe.queue.end();
}

Status OoOScheduler::SpillOnGuardBlock()
{
    for (auto* guardedAlloc : guardBlockedAllocs_) {
        auto guardAllocs = dualDstEngine_.GetUnretiredGuardAllocs(guardedAlloc);
        for (auto* guardAlloc : guardAllocs) {
            auto infoIt = state_.schedInfoMap.find(guardAlloc);
            if (infoIt == state_.schedInfoMap.end()) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "DualDst guard state invariant violated: guard alloc %s has no schedule info while guarded alloc "
                    "%s waits for it.",
                    state_.GetOpInfo(guardAlloc).c_str(), state_.GetOpInfo(guardedAlloc).c_str());
                return FAILED;
            }
            auto memIds = state_.GetOpMemIds(guardAlloc);
            if (memIds.empty() || state_.localBufferMap.find(memIds[0]) == state_.localBufferMap.end()) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "DualDst guard state invariant violated: cannot resolve guard alloc buffer for %s while guarded "
                    "alloc %s waits for it.",
                    state_.GetOpInfo(guardAlloc).c_str(), state_.GetOpInfo(guardedAlloc).c_str());
                return FAILED;
            }
            auto coreLocation = infoIt->second.coreLocation;
            auto memType = state_.localBufferMap[memIds[0]]->memType;
            auto& pipe = state_.allocIssueQueue[coreLocation][memType];
            if (pipe.Empty() || !IsOpInQueue(guardAlloc, pipe)) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "DualDst guard state invariant violated: unretired guard alloc %s is not in alloc queue "
                    "(core=%s, memType=%d) while guarded alloc %s waits for it.",
                    state_.GetOpInfo(guardAlloc).c_str(), coreTypeToString(coreLocation).c_str(), memType,
                    state_.GetOpInfo(guardedAlloc).c_str());
                return FAILED;
            }
            if (pipe.Front() == guardedAlloc && guardedAlloc != guardAlloc) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "DualDst guard order invariant violated: guarded alloc %s is before guard alloc %s in the same "
                    "alloc queue.",
                    state_.GetOpInfo(guardedAlloc).c_str(), state_.GetOpInfo(guardAlloc).c_str());
                return FAILED;
            }
            return SpillOnCoreBlock({coreLocation, memType});
        }
    }
    APASS_LOG_ERROR_F(Elements::Operation, "DualDst guard block found no spillable guard alloc.");
    return FAILED;
}

Status OoOScheduler::FindCoreLocationMemoryType(CoreLocationType coreLocation, MemoryType& spillMemType)
{
    bool anyNotEmpty = false;
    for (auto& kv : state_.allocIssueQueue[coreLocation]) {
        if (!kv.second.Empty()) {
            anyNotEmpty = true;
            break;
        }
    }
    if (!anyNotEmpty) {
        return FAILED;
    }
    if (!state_.allocIssueQueue[coreLocation][MemoryType::MEM_UB].Empty()) {
        spillMemType = MemoryType::MEM_UB;
    } else if (!state_.allocIssueQueue[coreLocation][MemoryType::MEM_L1].Empty()) {
        spillMemType = MemoryType::MEM_L1;
    } else if (coreLocation == CoreLocationType::AIC &&
               !state_.allocIssueQueue[coreLocation][MemoryType::MEM_L0C].Empty()) {
        spillMemType = MemoryType::MEM_L0C;
    } else {
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::FindFirstOrder(std::pair<CoreLocationType, MemoryType>& orderFirstPair)
{
    std::unordered_map<CoreLocationType, MemoryType> spillMemTypeMap;
    std::vector<CoreLocationType> coreVec;
    for (auto& coreLocation : state_.coreInitConfigs) {
        MemoryType spillMemType;
        if (FindCoreLocationMemoryType(coreLocation, spillMemType) != SUCCESS) {
            APASS_LOG_INFO_F(Elements::Operation, "FindCoreLocationMemoryType %s failed.",
                             coreTypeToString(coreLocation).c_str());
            continue;
        }
        spillMemTypeMap[coreLocation] = spillMemType;
        coreVec.push_back(coreLocation);
    }
    if (spillMemTypeMap.empty() || coreVec.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "All coreLocation have no spillMemType.");
        return FAILED;
    }
    std::sort(coreVec.begin(), coreVec.end(),
              [&spillMemTypeMap, this](const CoreLocationType& a, const CoreLocationType& b) {
                  return state_.schedInfoMap[state_.allocIssueQueue[a][spillMemTypeMap[a]].Front()].execOrder <
                         state_.schedInfoMap[state_.allocIssueQueue[b][spillMemTypeMap[b]].Front()].execOrder;
              });
    orderFirstPair = std::make_pair(coreVec[0], spillMemTypeMap[coreVec[0]]);
    return SUCCESS;
}

Status OoOScheduler::SpillOnBlock()
{
    if (!guardBlockedAllocs_.empty()) {
        guardBlockedAllocs_.erase(
            std::remove_if(guardBlockedAllocs_.begin(), guardBlockedAllocs_.end(),
                           [this](Operation* g) { return dualDstEngine_.GetUnretiredGuardAllocs(g).empty(); }),
            guardBlockedAllocs_.end());
        if (!guardBlockedAllocs_.empty()) {
            return SpillOnGuardBlock();
        }
    }
    std::pair<CoreLocationType, MemoryType> orderFirstPair;
    if (FindFirstOrder(orderFirstPair) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FindFirstOrder failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "Start to spillOnBlock at %s",
                     coreTypeToString(orderFirstPair.first).c_str());

    int curPreSpillTotalQSize = 0;
    for (auto& corePair : state_.allocIssueQueue) {
        for (auto& memPair : corePair.second) {
            curPreSpillTotalQSize += static_cast<int>(memPair.second.Size());
        }
    }
    if (state_.lastPreSpillTotalQSize >= 0) {
        if (curPreSpillTotalQSize >= state_.lastPreSpillTotalQSize) {
            state_.spillNoProgressCnt++;
            if (state_.spillNoProgressCnt >= MAX_SPILL_NO_PROGRESS_ROUNDS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Spill dead-loop detected: %d consecutive spills without total queue size reduction "
                                  "(current=%d, last=%d).",
                                  state_.spillNoProgressCnt, curPreSpillTotalQSize, state_.lastPreSpillTotalQSize);
                return FAILED;
            }
        } else {
            state_.spillNoProgressCnt = 0;
        }
    }
    state_.lastPreSpillTotalQSize = curPreSpillTotalQSize;

    if (SpillOnCoreBlock(orderFirstPair) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnCoreBlock failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::AllocSkipTensorMemRange(Operation& operation)
{
    auto outTensor = operation.GetOOperands()[0];
    int memId = outTensor->memoryrange.memId;
    if (state_.localBufferMap.find(memId) == state_.localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in state_.localBufferMap.", memId);
        return FAILED;
    }
    outTensor->memoryrange = TileRange(state_.localBufferMap[memId]->start, state_.localBufferMap[memId]->end, memId);
    return SUCCESS;
}

Status OoOScheduler::AllocTensorMemRange(Operation* op)
{
    auto& skipOps = state_.schedInfoMap[op].skipOps;
    for (auto& skipOp : skipOps) {
        if (!IsSkipOp(*skipOp)) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s is not a skip op.", state_.GetOpInfo(skipOp).c_str());
            return FAILED;
        }
        if (AllocSkipTensorMemRange(*skipOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AllocSkipTensorMemRange failed on %s.",
                              state_.GetOpInfo(skipOp).c_str());
            return FAILED;
        }
    }
    for (auto& outTensor : op->GetOOperands()) {
        MemoryType memType = outTensor->GetMemoryTypeOriginal();
        if (memType >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = outTensor->memoryrange.memId;
        if (state_.tensorOccupyMap.find(memId) == state_.tensorOccupyMap.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in tensorOccupyMap.", memId);
            return FAILED;
        }
        if (state_.localBufferMap.find(memId) == state_.localBufferMap.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap.", memId);
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Tensor, "REALLOC Tensor[%d] %s --> %s.", memId,
                          state_.GetOpInfo(state_.tensorOccupyMap[memId]).c_str(), state_.GetOpInfo(op).c_str());
        state_.tensorOccupyMap[memId] = op;
        outTensor->memoryrange = TileRange(state_.localBufferMap[memId]->start, state_.localBufferMap[memId]->end,
                                           memId);
    }
    return SUCCESS;
}

void OoOScheduler::HandleSkipOp(Operation* op)
{
    auto& skipOps = state_.schedInfoMap[op].skipOps;
    for (auto& skipOp : skipOps) {
        if (std::find(state_.newOperations.begin(), state_.newOperations.end(), skipOp) != state_.newOperations.end()) {
            continue;
        }
        state_.newOperations.emplace_back(skipOp);
    }
}

Status OoOScheduler::LaunchIssueStage(int& nextCycle)
{
    for (auto coreLocation : state_.coreInitConfigs) {
        for (auto& pipeEntry : issueQueues[coreLocation]) {
            auto& pipe = pipeEntry.second;
            if (pipe.Empty() || pipe.busy) {
                continue;
            }
            Operation* op = pipe.PopFront();
            // 标注op的生命周期
            op->cycleStart = state_.clock;
            op->cycleEnd = state_.clock + op->GetLatency();
            pipe.busy = true;
            pipe.curOp = op;
            pipe.curOpRetireCycle = state_.clock + op->GetLatency();
            NotifyOpLaunch(op, op->cycleEnd);
            HandleSkipOp(op);
            state_.newOperations.emplace_back(op);
            if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                nextCycle = pipe.curOpRetireCycle;
            }
            if (AllocTensorMemRange(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "AllocTensorMemRangeOp failed at coreType: %s. %s",
                                  coreTypeToString(coreLocation).c_str(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", state_.GetOpInfo(op).c_str());
        }
    }
    return SUCCESS;
}

Status OoOScheduler::TryDualDstAllocOnce(Operation* op, uint64_t& commitCnt, bool& allocated)
{
    if (dualDstEngine_.AllocateDualDstAtCurrent(op, allocated) != SUCCESS)
        return FAILED;
    if (!allocated)
        return SUCCESS; // 当前 Full,由调用方 break 触发 SpillOnBlock
    if (dualDstEngine_.CheckAivUbAllocAlignmentAfterDualDst(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAivUbAllocAlignmentAfterDualDst failed. %s",
                          GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    state_.newOperations.push_back(op);
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert(dualdst): %s.", state_.GetOpInfo(op).c_str());
    if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed for dualdst alloc. %s",
                          GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::TryRegularAllocOnce(Operation* op, MemoryType memType, CoreLocationType coreLocation,
                                         const std::vector<int>& reqMemIds, uint64_t& commitCnt, bool& allocated)
{
    auto& pool = state_.bufferManagerMap[coreLocation][memType];
    auto buf = state_.localBufferMap[reqMemIds[0]];
    bool hasMatchedOffset = false;
    uint64_t matchedOffset = 0;
    if (dualDstEngine_.GetMatchedAivUbAllocOffset(op, hasMatchedOffset, matchedOffset) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetMatchedAivUbAllocOffset failed. %s",
                          GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    if (!hasMatchedOffset && pool.IsFull(buf, true)) {
        allocated = false;
        return SUCCESS;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "ALLOCATE: %s.", state_.GetOpInfo(op).c_str());
    Status allocStatus = hasMatchedOffset ? pool.AllocateAtOffset(buf, matchedOffset) : pool.Allocate(buf);
    if (allocStatus != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Allocate Tensor[%d] failed.", reqMemIds[0]);
        return FAILED;
    }
    // pool-evt 记录单池 alloc 成功时的 clock、memId 和 size，用于重建池占用曲线。
    APASS_LOG_DEBUG_F(Elements::Operation, "[pool-evt] alloc clock=%llu cycles core=%d mt=%d memId=%d size=%llu bytes",
                      (unsigned long long)state_.clock, static_cast<int>(coreLocation), static_cast<int>(memType),
                      reqMemIds[0], (unsigned long long)buf->size);
    NotifyAllocExec(op, reqMemIds[0]);
    state_.tensorOccupyMap[reqMemIds[0]] = op;
    buf->startCycle = state_.clock;
    if (op->GetOutputOperand(0) == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] cannot find oOperand[0]. %s", op->GetOpMagic(),
                          GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    if (dualDstEngine_.RecordAivUbAlloc(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RecordAivUbAlloc failed. %s", GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    state_.newOperations.push_back(op);
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", state_.GetOpInfo(op).c_str());
    if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSuccOp failed. %s", GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    allocated = true;
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(uint64_t& commitCnt, MemoryType memType, OpQueue& pipe)
{
    while (!pipe.Empty()) {
        Operation* op = pipe.Front();
        if (!dualDstEngine_.IsDualDstAllocGuardSatisfied(op)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "DualDst alloc guard blocks: %s.", state_.GetOpInfo(op).c_str());
            if (std::find(guardBlockedAllocs_.begin(), guardBlockedAllocs_.end(), op) == guardBlockedAllocs_.end()) {
                guardBlockedAllocs_.push_back(op);
            }
            break;
        }
        auto& coreLocation = state_.schedInfoMap[op].coreLocation;
        auto& reqMemIds = state_.GetOpMemIds(op);
        bool allocated = false;
        Status st = (memType == MemoryType::MEM_UB && state_.IsDualDstAlloc(op)) ?
                        TryDualDstAllocOnce(op, commitCnt, allocated) :
                        TryRegularAllocOnce(op, memType, coreLocation, reqMemIds, commitCnt, allocated);
        if (st != SUCCESS)
            return FAILED;
        if (!allocated)
            break; // Full;退出循环交给 SpillOnBlock
        pipe.PopFront();
    }
    return SUCCESS;
}

Status OoOScheduler::BufferAllocStage(uint64_t& commitCnt)
{
    guardBlockedAllocs_.clear();
    for (auto coreLocation : state_.coreInitConfigs) {
        for (auto& [memType, pipe] : state_.allocIssueQueue[coreLocation]) {
            if (pipe.Empty()) {
                continue;
            }
            // 不断按顺序执行alloc指令，直到buffer被占满为止。
            if (ExecuteAllocIssue(commitCnt, memType, pipe) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed at coreType: %s.",
                                  coreTypeToString(coreLocation).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::FreeBuffer(Operation* op, std::vector<int>& freedMemIds)
{
    auto& reqMemIds = state_.GetOpMemIds(op);
    for (auto memId : reqMemIds) {
        if (state_.DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor [%d] failed.", memId);
            return FAILED;
        }
        if (state_.bufRefCount[memId] == 0) {
            CoreLocationType coreLocation = state_.ResolveCoreForFree(memId);
            auto memType = state_.localBufferMap[memId]->memType;
            uint64_t freedSize = state_.localBufferMap[memId]->size;
            if (state_.bufferManagerMap[coreLocation][memType].Free(state_.localBufferMap[memId]->id) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor [%d] failed.", memId);
                return FAILED;
            }
            // pool-evt 记录 free 事件。
            APASS_LOG_DEBUG_F(
                Elements::Operation, "[pool-evt] free clock=%llu core=%d mt=%d memId=%d size=%llu freerOp=%s",
                (unsigned long long)state_.clock, static_cast<int>(coreLocation), static_cast<int>(memType), memId,
                (unsigned long long)freedSize, state_.GetOpInfo(op).c_str());
            freedMemIds.push_back(memId);
            if (state_.tensorOccupyMap.erase(memId) == 0) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Erase tensor[%d] failed.", memId);
                return FAILED;
            }
            state_.dualDstMemIdCoreOverride.erase(memId); // free 后清理, 避免后续 spill 重 alloc 复用旧映射
            // 同步清理 dualDstPairedMemId 的双向映射: 这一侧 free 后, 这对 pair 不再"两侧
            // 都活", 后续 spill 选 group 时不能再把它当候选了。
            auto pairIt = state_.dualDstPairedMemId.find(memId);
            if (pairIt != state_.dualDstPairedMemId.end()) {
                state_.dualDstPairedMemId.erase(pairIt->second);
                state_.dualDstPairedMemId.erase(memId);
            }
        }
    }
    return SUCCESS;
}

std::vector<Operation*> OoOScheduler::GetNewOperations()
{
    std::vector<Operation*> uniq;
    uniq.reserve(state_.newOperations.size());
    std::unordered_set<Operation*> seen;
    for (auto* op : state_.newOperations) {
        if (op == nullptr)
            continue;
        if (seen.insert(op).second) {
            uniq.push_back(op);
        } else {
            // 命中即代表上游某条 push 路径漏了去重 (期望随后续根因修复降为 0)。
            APASS_LOG_WARN_F(Elements::Operation, "GetNewOperations dedupe drop duplicate op[%d] %s", op->GetOpMagic(),
                             op->GetOpcodeStr().c_str());
        }
    }
    return uniq;
}

Status OoOScheduler::RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt)
{
    commitCnt++;
    state_.schedInfoMap[op].isRetired = true;
    std::vector<int> freedMemIds;
    if (FreeBuffer(op, freedMemIds) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FreeBufferOp failed. %s", GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    for (auto memId : freedMemIds) {
        state_.localBufferMap[memId]->retireCycle = state_.clock;
    }
    NotifyOpRetire(op, freedMemIds);

    auto& successors = state_.depManager.GetSuccessors(op);
    for (auto succOp : successors) {
        if (state_.schedInfoMap[succOp].isRetired) {
            continue;
        }
        bool ready = true;
        auto& preds = state_.depManager.GetPredecessors(succOp);
        for (auto predOp : preds) {
            if (!state_.schedInfoMap[predOp].isRetired) {
                ready = false;
                break;
            }
        }
        if (ready) {
            issueQueues[state_.schedInfoMap[succOp].coreLocation][state_.schedInfoMap[succOp].pipeType].Insert(succOp);
            APASS_LOG_DEBUG_F(Elements::Operation, "    Wakeup: %s, execOrder: %d", state_.GetOpInfo(succOp).c_str(),
                              state_.schedInfoMap[succOp].execOrder);
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireCoreIssue(CoreLocationType coreLocation, uint64_t& commitCnt, int& nextCycle)
{
    for (auto& [pipeType, pipe] : issueQueues[coreLocation]) {
        if (!pipe.busy) {
            continue;
        }
        if (!state_.pipeEndTime.count(pipeType)) {
            state_.pipeEndTime.emplace(pipeType, pipe.curOpRetireCycle);
        } else {
            auto curEndTime = state_.pipeEndTime[pipeType];
            state_.pipeEndTime[pipeType] = std::max(curEndTime, pipe.curOpRetireCycle);
        }
        if (pipe.curOpRetireCycle <= state_.clock) { // 如果该pipe内当前正在执行op，在clock的时刻已经执行完毕。
            Operation* op = pipe.curOp;
            pipe.busy = false;
            pipe.curOp = nullptr;
            APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTE END: %s", state_.GetOpInfo(op).c_str());
            if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSuccOp failed at coreType: %s! %s",
                                  coreTypeToString(coreLocation).c_str(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTING[%d]: %s", pipe.curOpRetireCycle,
                          state_.GetOpInfo(pipe.curOp).c_str());
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireIssueStage(uint64_t& commitCnt, int& nextCycle)
{
    for (auto coreLocation : state_.coreInitConfigs) {
        if (RetireCoreIssue(coreLocation, commitCnt, nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssueStage failed");
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOScheduler::LaunchReadyIssue()
{
    // 初始化 Queue
    for (auto& op : state_.orderedOps) {
        auto& coreLocation = state_.schedInfoMap[op].coreLocation;
        if (USE_LESS_OPS.find(op->GetOpcode()) != USE_LESS_OPS.end() && state_.depManager.GetPredecessors(op).empty()) {
            issueQueues[coreLocation][state_.schedInfoMap[op].pipeType].Insert(op);
        }
        if (state_.schedInfoMap[op].isAlloc) {
            auto& reqMemIds = state_.GetOpMemIds(op);
            if (!reqMemIds.empty()) {
                auto memType = state_.localBufferMap[reqMemIds[0]]->memType;
                state_.allocIssueQueue[coreLocation][memType].Insert(op);
            }
        }
    }
}

Status OoOScheduler::ScheduleMainLoop()
{
    LOG_SCOPE_BEGIN(tScheduleMainLoop, Elements::Function, "ScheduleMainLoop");
    auto ret = RunSchedulerMainLoop(*this);
    LOG_SCOPE_END(tScheduleMainLoop);
    return ret;
}

Status OoOScheduler::PreMainLoop()
{
    UpdateIssueExecOrder();
    LaunchReadyIssue();
    state_.numTotalIssues = state_.orderedOps.size();
    state_.originalNumTotalIssues = state_.numTotalIssues;
    NotifyMainLoopBegin();
    return SUCCESS;
}

Status OoOScheduler::PostMainLoop()
{
    NotifyMainLoopEnd();
    return SUCCESS;
}

Status OoOScheduler::RetireIssue(Operation* op)
{
    state_.schedInfoMap[op].isRetired = true;
    std::vector<int> freedMemIds;
    return FreeBuffer(op, freedMemIds);
}

Status OoOScheduler::AllocateAfterSpill(Operation* op, LocalBufferPtr allocBuffer, CoreLocationType coreLocation,
                                        bool isDualDst)
{
    if (!isDualDst) {
        if (state_.bufferManagerMap[coreLocation][allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Allocate tensor[%d] failed.", allocBuffer->id);
            return FAILED;
        }
        return SUCCESS;
    }

    bool allocated = false;
    if (dualDstEngine_.AllocateDualDstAtCurrent(op, allocated) != SUCCESS)
        return FAILED;
    if (!allocated) {
        APASS_LOG_ERROR_F(Elements::Operation, "DualDst alloc still infeasible after spill. %s",
                          GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    if (dualDstEngine_.CheckAivUbAllocAlignmentAfterDualDst(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAivUbAllocAlignmentAfterDualDst failed. %s",
                          GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(Operation* op, size_t& pcIdx)
{
    if (state_.localBufferMap.find(state_.GetOpMemIds(op)[0]) == state_.localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap!", state_.GetOpMemIds(op)[0]);
        return FAILED;
    }
    LocalBufferPtr allocBuffer = state_.localBufferMap[state_.GetOpMemIds(op)[0]];
    auto coreLocation = state_.schedInfoMap[op].coreLocation;
    const bool isDualDst = (allocBuffer->memType == MemoryType::MEM_UB) && state_.IsDualDstAlloc(op);

    bool needSpill = false;
    if (isDualDst) {
        bool allocated = false;
        if (dualDstEngine_.AllocateDualDstAtCurrent(op, allocated) != SUCCESS)
            return FAILED;
        if (allocated) {
            if (dualDstEngine_.CheckAivUbAllocAlignmentAfterDualDst(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "CheckAivUbAllocAlignmentAfterDualDst failed. %s",
                                  GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            return SUCCESS;
        }
        needSpill = true;
    } else {
        needSpill = state_.bufferManagerMap[coreLocation][allocBuffer->memType].IsFull(allocBuffer, false);
    }

    if (needSpill) {
        SpillContext ctx;
        if (GenBufferSpill(op, ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GenBufferSpill failed at ExecuteAllocIssue. %s",
                              GetFormatBacktrace(*state_.orderedOps[pcIdx]).c_str());
            return FAILED;
        }
        pcIdx += ctx.newCopyoutOps.size() - ctx.deleteRetiredOpSize;
    }

    return AllocateAfterSpill(op, allocBuffer, coreLocation, isDualDst);
}

Status OoOScheduler::SeqSchedule()
{
    UpdateIssueExecOrder();
    state_.originalNumTotalIssues = state_.orderedOps.size();
    size_t pcIdx = 0;
    LOG_SCOPE_BEGIN(tGenSpillSchedule, Elements::Function, "SeqSchedule");
    while (pcIdx < state_.orderedOps.size()) {
        auto op = state_.orderedOps[pcIdx];
        APASS_LOG_DEBUG_F(Elements::Operation, "Launch %s", state_.GetOpInfo(op).c_str());
        if (state_.schedInfoMap[op].isAlloc) {
            if (ExecuteAllocIssue(op, pcIdx) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed! %s", GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
        }
        for (auto& outTensor : op->GetOOperands()) {
            if (outTensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                state_.tensorOccupyMap[outTensor->memoryrange.memId] = op;
            }
        }
        if (RetireIssue(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssue failed! %s", GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        pcIdx += 1;
        if (state_.orderedOps.size() > state_.originalNumTotalIssues * MAX_NUM_TOTAL_ISSUES_RATIO) {
            APASS_LOG_ERROR_F(Elements::Operation, "Spill dead-loop.");
            return FAILED;
        }
    }
    for (auto bufRef : state_.bufRefCount) {
        if (bufRef.second != 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount not equal to 0!", bufRef.first);
            return FAILED;
        }
    }
    LOG_SCOPE_END(tGenSpillSchedule);
    if (state_.InitBufRefCount(state_.orderedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
    }
    for (const auto& op : state_.orderedOps) {
        state_.schedInfoMap[op].isRetired = false;
    }
    // 更新依赖关系
    if (state_.depManager.InitDependencies(state_.orderedOps, false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    state_.depManager.PrintDependencies(state_.orderedOps);
    return SUCCESS;
}

void OoOScheduler::InitIssueQueuesAndBufferManager()
{
    // 设置比较函数用于OpQueue排序
    auto compareFunc = [this](Operation* a, Operation* b) {
        return state_.schedInfoMap[a].execOrder > state_.schedInfoMap[b].execOrder;
    };

    // 初始化
    for (auto coreLocation : state_.coreInitConfigs) {
        for (size_t i = 0; i <= static_cast<int>(PipeType::PIPE_FIX); i++) {
            OpQueue queue;
            queue.SetCompareFunc(compareFunc);
            issueQueues[coreLocation][static_cast<PipeType>(i)] = queue;
        }
    }

    state_.bufferManagerMap.clear();
    for (auto coreLocation : state_.coreInitConfigs) {
        if (coreLocation == CoreLocationType::AIV0 || coreLocation == CoreLocationType::AIV1) {
            OpQueue queue;
            queue.SetCompareFunc(compareFunc);
            state_.allocIssueQueue[coreLocation][MemoryType::MEM_UB] = queue;
            state_.bufferManagerMap[coreLocation].insert(
                {MemoryType::MEM_UB, BufferPool(MemoryType::MEM_UB, state_.localMemSize[MemoryType::MEM_UB])});
            continue;
        }
        for (size_t i = 1; i < static_cast<int>(MemoryType::MEM_DEVICE_DDR); i++) {
            OpQueue queue;
            queue.SetCompareFunc(compareFunc);
            state_.allocIssueQueue[coreLocation][static_cast<MemoryType>(i)] = queue;
            if (state_.localMemSize.find(static_cast<MemoryType>(i)) != state_.localMemSize.end()) {
                state_.bufferManagerMap[coreLocation].insert(
                    {static_cast<MemoryType>(i),
                     BufferPool(static_cast<MemoryType>(i), state_.localMemSize[static_cast<MemoryType>(i)])});
            }
        }
    }
}

void OoOScheduler::InitTensorCoreMap()
{
    // 不存在 no producer情况
    for (auto op : state_.orderedOps) {
        if (state_.schedInfoMap[op].isAlloc) {
            auto memId = op->GetOutputOperand(0)->memoryrange.memId;
            state_.tensorAllocMap[memId] = op;
        }
    }
}

void OoOScheduler::InitCoreConfig(const std::vector<Operation*>& opList)
{
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
        state_.coreInitConfigs = CORE_INIT_CONFIGS_HARDWARE_ONE;
    } else {
        state_.coreInitConfigs = CORE_INIT_CONFIGS_HARDWARE_TWO;
    }
}

// skipOps 按生产者在前存放: HandleSkipOp 照这个次序写回调度序列, 反了的话消费者会排在生产者之前。
// 向上走出的祖先路径天然是消费者在前, 每条子链单独翻转后追加; 去重命中意味着该 op 连同它的整条
// 祖先前缀已经在更前面放好了, 所以跨 operand 共享祖先时次序依然成立。
void OoOScheduler::InitSkipOps(Operation* op)
{
    if (op == nullptr)
        return;
    std::vector<Operation*> skipOps;
    for (auto iOperand : op->GetIOperands()) {
        for (auto pre : iOperand->GetProducers()) {
            std::vector<Operation*> chain;
            while (pre != nullptr && IsSkipOp(*pre) &&
                   pre->GetOutputOperand(0)->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                chain.push_back(pre);
                const auto& upProducers = pre->GetInputOperand(0)->GetProducers();
                pre = upProducers.empty() ? nullptr : *upProducers.begin();
            }
            for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                if (std::find(skipOps.begin(), skipOps.end(), *it) == skipOps.end()) {
                    skipOps.push_back(*it);
                }
            }
        }
    }
    state_.schedInfoMap[op].skipOps = skipOps;
}

Status OoOScheduler::InitOpCoreType(Operation* op)
{
    if (op == nullptr)
        return FAILED;
    std::unordered_map<int, CoreLocationType> opCoreMap = {
        {0, CoreLocationType::AIC}, {1, CoreLocationType::AIV0}, {2, CoreLocationType::AIV1}};
    CoreLocationType coreLocation;
    int internalSubgraphID = op->GetInternalSubgraphID();
    if (internalSubgraphID != -1) {
        coreLocation = opCoreMap[op->GetInternalSubgraphID()];
    } else if (op->GetCoreType() == CoreType::AIC) {
        coreLocation = CoreLocationType::AIC;
    } else if (op->GetCoreType() == CoreType::AIV) {
        coreLocation = CoreLocationType::AIV0;
    } else {
        // 对 ANY 类型进行处理
        if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            coreLocation = CoreLocationType::AIV0;
        } else if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
            coreLocation = CoreLocationType::AIC;
        } else if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            if (op->GetIOperands().size() == 0 ||
                op->GetInputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
                coreLocation = CoreLocationType::AIC;
            } else if (op->GetInputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                coreLocation = CoreLocationType::AIV0;
            } else if (op->GetInputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
                coreLocation = CoreLocationType::AIC;
            } else {
                APASS_LOG_ERROR_F(Elements::Operation, "%s init coreLocation failed.", state_.GetOpInfo(op).c_str());
                return FAILED;
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "%s init coreLocation failed.", state_.GetOpInfo(op).c_str());
            return FAILED;
        }
    }
    state_.schedInfoMap[op].coreLocation = coreLocation;
    return SUCCESS;
}

Status OoOScheduler::InitOpEntry(Operation* op)
{
    if (op == nullptr)
        return FAILED;

    // skip op 由消费者的 skipOps 带走; 输出在 DDR 上的没有消费者接管, 攒起来等定稿时按拓扑落位。
    if (IsSkipOp(*op)) {
        if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            ddrSkipOps_.push_back(op);
        }
        return SUCCESS;
    }
    if (state_.CheckOpBufferSize(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] CheckOpBufferSize failed! %s", op->GetOpcodeStr().c_str(),
                          op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
        return FAILED;
    }

    // 初始化Operation属性到map
    int order = static_cast<int>(state_.orderedOps.size());
    state_.orderedOps.push_back(op);
    state_.schedInfoMap[op].execOrder = order;
    state_.schedInfoMap[op].pipeType = RescheduleUtils::GetOpPipeType(op);
    state_.schedInfoMap[op].isAlloc = (op->GetOpcodeStr().find("ALLOC") != std::string::npos);
    state_.schedInfoMap[op].isRetired = false;
    state_.SetOpMemIds(op, {});

    InitSkipOps(op);

    // 初始化核属性
    if (InitOpCoreType(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Operation %s init coreType failed!", state_.GetOpInfo(op).c_str());
        return FAILED;
    }

    APASS_LOG_DEBUG_F(Elements::Operation, "issue: %s, coreType: %s", state_.GetOpInfo(op).c_str(),
                      coreTypeToString(state_.schedInfoMap[op].coreLocation).c_str());
    return SUCCESS;
}

Status OoOScheduler::Init(const std::vector<Operation*>& opList,
                          const std::unordered_set<CoreLocationType> fixCoreConfig)
{
    state_.orderedOps.clear();
    state_.schedInfoMap.clear();
    state_.ClearAllOpMemIds();
    state_.localBufferMap.clear();
    ddrSkipOps_.clear();
    LOG_SCOPE_BEGIN(tInit, Elements::Function, "Init");
    // 初始化芯片各buffer大小
    state_.localMemSize = CommonUtils::GetLocalMemorySize();
    if (fixCoreConfig.empty()) {
        InitCoreConfig(opList);
    } else {
        state_.coreInitConfigs = fixCoreConfig;
    }
    // 校验并初始化Operation
    for (const auto& op : opList) {
        if (InitOpEntry(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Operation %s[%d] init issue failed!", op->GetOpcodeStr().c_str(),
                              op->GetOpMagic());
            return FAILED;
        }
    }
    state_.numTotalIssues = state_.orderedOps.size();

    if (state_.InitBufRefCount(state_.orderedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
    }
    // 初始化依赖关系
    if (state_.depManager.InitDependencies(state_.orderedOps, false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    state_.depManager.PrintDependencies(state_.orderedOps);
    InitTensorCoreMap();
    // 初始化内存管理器
    InitIssueQueuesAndBufferManager();
    LOG_SCOPE_END(tInit);
    return SUCCESS;
}

void OoOScheduler::UpdateL0MXMap(const std::vector<Operation*>& opList)
{
    for (size_t i = 0; i < opList.size(); i++) {
        if (opList[i]->GetOpcode() == Opcode::OP_L1_TO_L0B_SCALE) {
            auto l0MxOut = opList[i]->GetOOperands()[0];
            auto consOp = *l0MxOut->GetConsumers().begin();
            LogicalTensorPtr l0ATensor;
            LogicalTensorPtr l0BTensor;
            LogicalTensorPtr l0AMXTensor;
            LogicalTensorPtr l0BMXTensor;
            for (auto& l0Tensor : consOp->GetIOperands()) {
                if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
                    l0ATensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
                    l0BTensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0AMX) {
                    l0AMXTensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0BMX) {
                    l0BMXTensor = l0Tensor;
                }
            }
            dualDstEngine_.GetL02L0MXMap()[l0ATensor] = l0AMXTensor;
            dualDstEngine_.GetL02L0MXMap()[l0BTensor] = l0BMXTensor;
        }
    }
}

void OoOScheduler::UpdateDualDstL0MXRange()
{
    constexpr int kL0mxAddrShiftBits = 4;
    for (auto& entry : dualDstEngine_.GetL02L0MXMap()) {
        auto l0Tensor = entry.first;
        auto l0MXTensor = entry.second;
        int l0MemID = l0Tensor->memoryrange.memId;
        int l0MemMXID = l0MXTensor->memoryrange.memId;
        l0MXTensor->memoryrange = TileRange(state_.localBufferMap[l0MemID]->start >> kL0mxAddrShiftBits,
                                            state_.localBufferMap[l0MemID]->end >> kL0mxAddrShiftBits, l0MemMXID);
    }
}

// 紧跟最后一个生产者插入: 位置唯一确定, 且不会把 skip op 推得比必要更晚。
// 一个生产者都不在序列里的留在头部 —— 无依赖可违背。
void OoOScheduler::PlaceDdrSkipOps()
{
    for (auto* skipOp : ddrSkipOps_) {
        // 已标删的不再落位: 插进 newOperations 就成了"既发射又删除", 数量守恒当场不成立。
        if (skipOp->IsDeleted()) {
            continue;
        }
        size_t insertPos = 0;
        for (auto iOperand : skipOp->GetIOperands()) {
            if (iOperand == nullptr) {
                continue;
            }
            for (auto* prod : iOperand->GetProducers()) {
                // 生产者自己也可能不参与调度 (USE_LESS_OPS), 不在序列里就没有位置约束可言。
                auto it = std::find(state_.newOperations.begin(), state_.newOperations.end(), prod);
                if (it == state_.newOperations.end()) {
                    continue;
                }
                size_t prodPos = static_cast<size_t>(std::distance(state_.newOperations.begin(), it));
                insertPos = std::max(insertPos, prodPos + 1);
            }
        }
        state_.newOperations.insert(state_.newOperations.begin() + static_cast<long>(insertPos), skipOp);
        APASS_LOG_DEBUG_F(Elements::Operation, "Place DDR skip op %s at %zu.", state_.GetOpInfo(skipOp).c_str(),
                          insertPos);
    }
}

Status OoOScheduler::FinalizeScheduleResult(const std::vector<Operation*>& opList)
{
    PlaceDdrSkipOps();
    // 消费者可能在链标删之前就发射过, 那时 HandleSkipOp 已把它写进 newOperations。
    auto& newOps = state_.newOperations;
    newOps.erase(std::remove_if(newOps.begin(), newOps.end(), [](Operation* op) { return op->IsDeleted(); }),
                 newOps.end());
    function_.EraseOperations(false, false);
    UpdateL0MXMap(opList);
    UpdateDualDstL0MXRange();
    PrintOpList(state_.newOperations);
    if (dualDstEngine_.VerifyDualDstSameOffset() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "VerifyDualDstSameOffset failed.");
        return FAILED;
    }
    function_.SetStackWorkespaceSize(state_.workspaceOffset);
    function_.pipeEndTime = state_.pipeEndTime;
    return SUCCESS;
}
Status OoOScheduler::Schedule(const std::vector<Operation*>& opList,
                              const std::unordered_set<CoreLocationType> fixCoreConfig)
{
    struct EndGuard {
        OoOScheduler* self;
        bool success{false};
        ~EndGuard() { self->NotifyScheduleEnd(success); }
    } guard{this};
    if (opList.empty()) {
        guard.success = true;
        return SUCCESS;
    }

    if (Init(opList, fixCoreConfig) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Init failed!");
        return FAILED;
    }
    AllocWorkspaceGM(opList);
    if (dualDstEngine_.RealignAllocByIso(state_.orderedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RealignAllocByIso failed!");
        return FAILED;
    }
    UpdateIssueExecOrder();
    if (dualDstEngine_.RunDualDstFuse() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RunDualDstFuse failed!");
        return FAILED;
    }
    NotifyInitDDRBuffers();
    if (SeqSchedule() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenSpillSchedule failed!");
        return FAILED;
    }
    if (dualDstEngine_.BuildDualDstAllocGuards() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "BuildDualDstAllocGuards failed!");
        return FAILED;
    }
    if (ScheduleMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ScheduleMainLoop failed!");
        return FAILED;
    }
    if (CheckAndUpdateLifecycle() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAndUpdateLifecycle failed!");
        return FAILED;
    }
    if (FinalizeScheduleResult(opList) != SUCCESS)
        return FAILED;
    guard.success = true;
    return SUCCESS;
}

void OoOScheduler::AllocWorkspaceGM(const std::vector<Operation*>& opList)
{
    std::set<int> allocedRawmagic;
    for (auto& inCast : function_.GetIncast()) {
        allocedRawmagic.insert(inCast->tensor->GetRawMagic());
    }
    for (auto& outCast : function_.GetOutcast()) {
        allocedRawmagic.insert(outCast->tensor->GetRawMagic());
    }
    std::map<int, TileRange> rawMagicRange;
    std::map<int, int64_t> rawMagicOffset;
    for (auto& op : opList) {
        for (auto& iOperand : op->GetIOperands()) {
            if (allocedRawmagic.count(iOperand->tensor->GetRawMagic()) &&
                rawMagicRange.count(iOperand->tensor->GetRawMagic())) {
                iOperand->memoryrange = rawMagicRange[iOperand->tensor->GetRawMagic()];
                iOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, rawMagicOffset[iOperand->tensor->GetRawMagic()]);
            } else if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                       !allocedRawmagic.count(iOperand->tensor->GetRawMagic())) {
                allocedRawmagic.insert(iOperand->tensor->GetRawMagic());
                iOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, state_.workspaceOffset);
                iOperand->memoryrange = TileRange(state_.workspaceOffset,
                                                  state_.workspaceOffset + iOperand->tensor->GetRawDataSize(),
                                                  iOperand->tensor->GetRawMagic());
                rawMagicOffset[iOperand->tensor->GetRawMagic()] = state_.workspaceOffset;
                state_.workspaceOffset += iOperand->tensor->GetRawDataSize();
                rawMagicRange[iOperand->tensor->GetRawMagic()] = iOperand->memoryrange;
                ddrKindMap_[iOperand->memoryrange.memId] = DDRBufferKind::FUNCTION_TEMP;
            }
        }
        for (auto& oOperand : op->GetOOperands()) {
            if (allocedRawmagic.count(oOperand->tensor->GetRawMagic()) &&
                rawMagicRange.count(oOperand->tensor->GetRawMagic())) {
                oOperand->memoryrange = rawMagicRange[oOperand->tensor->GetRawMagic()];
                oOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, rawMagicOffset[oOperand->tensor->GetRawMagic()]);
            } else if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                       !allocedRawmagic.count(oOperand->tensor->GetRawMagic())) {
                allocedRawmagic.insert(oOperand->tensor->GetRawMagic());
                oOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, state_.workspaceOffset);
                oOperand->memoryrange = TileRange(state_.workspaceOffset,
                                                  state_.workspaceOffset + oOperand->tensor->GetRawDataSize(),
                                                  oOperand->tensor->GetRawMagic());
                rawMagicOffset[oOperand->tensor->GetRawMagic()] = state_.workspaceOffset;
                state_.workspaceOffset += oOperand->tensor->GetRawDataSize();
                rawMagicRange[oOperand->tensor->GetRawMagic()] = oOperand->memoryrange;
                ddrKindMap_[oOperand->memoryrange.memId] = DDRBufferKind::FUNCTION_TEMP;
            }
        }
    }
}

bool OoOScheduler::HasEnoughBuffer(Operation* allocOp, MemoryType memType)
{
    return !state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType].IsFull(
        state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]], false);
}

Status OoOScheduler::RearrangeBuffer(Operation* allocOp, MemoryType memType)
{
    std::vector<int> memIds = state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType]
                                  .GetAddrSortedBufs();
    for (auto memId : memIds) {
        auto op = spillEngine_.GetSpillOp(memId);
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] lastest write op.", memId);
            return FAILED;
        }
        if (op->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            return SUCCESS;
        }
    }
    std::vector<BufferAddrChange> changes;
    auto status = state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType].CompactBufferSlices(
        state_.localBufferMap, changes);
    if (status == SUCCESS) {
        NotifyBufferRearrange(allocOp, memType, std::move(changes));
    }
    return status;
}

std::vector<std::vector<int>> OoOScheduler::GetSpillGroup(BufferPool& pool, size_t sizeNeedSpill)
{
    return pool.GetSpillGroup(sizeNeedSpill);
}

std::vector<std::vector<int>> OoOScheduler::GetDualSpillGroup(BufferPool& poolA, BufferPool& poolB,
                                                              size_t sizeNeedSpill)
{
    std::vector<std::vector<int>> result;
    auto bufsA = poolA.GetSortedAllocatedBufs();
    auto bufsB = poolB.GetSortedAllocatedBufs();

    size_t iA = 0;
    while (iA < bufsA.size()) {
        size_t startAddrA = poolA.ObtainStartAddr(iA, bufsA);
        if ((poolA.GetMemSize() - startAddrA) < sizeNeedSpill) {
            break;
        }
        size_t jA = poolA.UpdateIdx(iA, sizeNeedSpill, startAddrA, bufsA);
        size_t iB = 0;
        while (iB < bufsB.size()) {
            size_t startAddrB = poolB.ObtainStartAddr(iB, bufsB);
            if (startAddrB != startAddrA) {
                if (startAddrB < startAddrA) {
                    iB++;
                    continue;
                }
                break;
            }
            if ((poolB.GetMemSize() - startAddrB) < sizeNeedSpill) {
                break;
            }
            size_t jB = poolB.UpdateIdx(iB, sizeNeedSpill, startAddrB, bufsB);
            if (jA < iA || jB < iB || (jA == iA && jB == iB)) {
                iB++;
                continue;
            }
            std::vector<int> combined;
            combined.reserve((jA - iA) + (jB - iB));
            for (size_t k = iA; k < jA; k++) {
                combined.push_back(std::get<0>(bufsA[k]));
            }
            for (size_t k = iB; k < jB; k++) {
                combined.push_back(std::get<0>(bufsB[k]));
            }
            result.push_back(std::move(combined));
            iB++;
        }
        iA++;
    }
    return result;
}

Status OoOScheduler::GetGroupNextUseTime(std::vector<int> group, Operation* allocOp, std::vector<int>& groupNextUseTime,
                                         std::unordered_map<int, size_t>& nextUseTimeCache)
{
    size_t minNextUseTime = INT_MAX;
    for (auto& memId : group) {
        Operation* spillOp = spillEngine_.GetSpillOp(memId);
        if (spillOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] occupy op.", memId);
            return FAILED;
        }
        if (spillEngine_.IsBelongSpillBlackList(spillOp, allocOp)) {
            groupNextUseTime.push_back(-1);
            return SUCCESS;
        }
        if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
            minNextUseTime = std::min(minNextUseTime, nextUseTimeCache[memId]);
        } else {
            int nextUseTime = spillEngine_.GetBufNextUseTime(memId);
            if (nextUseTime == -1) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] next used time.", memId);
                return FAILED;
            }
            nextUseTimeCache[memId] = static_cast<size_t>(nextUseTime);
            minNextUseTime = std::min(minNextUseTime, static_cast<size_t>(nextUseTime));
        }
    }
    groupNextUseTime.push_back(minNextUseTime);
    return SUCCESS;
}

bool OoOScheduler::CollectClearableWindowGroup(const std::vector<std::tuple<int, size_t, size_t>>* pools[2],
                                               uint64_t winStart, uint64_t winEnd, Operation* allocOp,
                                               std::unordered_map<int, bool>& spillableCache,
                                               std::unordered_map<int, long long>& nextUseCache,
                                               std::vector<int>& group, long long& winScore)
{
    for (int poolIdx = 0; poolIdx < 2; ++poolIdx) {
        const auto* bufs = pools[poolIdx];
        for (auto& b : *bufs) {
            size_t s = std::get<1>(b);
            size_t e = std::get<2>(b);
            if (e <= winStart || winEnd <= s)
                continue;
            int memId = std::get<0>(b);
            auto spillableIt = spillableCache.find(memId);
            if (spillableIt == spillableCache.end()) {
                Operation* op = spillEngine_.GetSpillOp(memId);
                spillableIt = spillableCache
                                  .emplace(memId, op != nullptr && !spillEngine_.IsBelongSpillBlackList(op, allocOp))
                                  .first;
            }
            if (!spillableIt->second) {
                return false;
            }
            auto nextUseIt = nextUseCache.find(memId);
            if (nextUseIt == nextUseCache.end()) {
                int nextUse = spillEngine_.GetBufNextUseTime(memId);
                long long effectiveNextUse = (nextUse < 0) ? LLONG_MAX : static_cast<long long>(nextUse);
                nextUseIt = nextUseCache.emplace(memId, effectiveNextUse).first;
            }
            winScore = std::min(winScore, nextUseIt->second);
            group.push_back(memId);
        }
    }
    return true;
}

std::vector<std::vector<int>> OoOScheduler::GetCommonOffsetClearGroup(BufferPool& poolA, BufferPool& poolB,
                                                                      size_t sizeNeedSpill, Operation* allocOp)
{
    auto bufsA = poolA.GetSortedAllocatedBufs();
    auto bufsB = poolB.GetSortedAllocatedBufs();
    uint64_t memSize = poolA.GetMemSize();

    std::set<uint64_t> candStarts;
    candStarts.insert(0);
    for (auto& b : bufsA) {
        candStarts.insert(std::get<1>(b));
        candStarts.insert(std::get<2>(b));
    }
    for (auto& b : bufsB) {
        candStarts.insert(std::get<1>(b));
        candStarts.insert(std::get<2>(b));
    }

    std::unordered_map<int, bool> spillableCache;
    std::unordered_map<int, long long> nextUseCache;
    std::vector<int> bestGroup;
    long long bestScore = -1;
    const std::vector<std::tuple<int, size_t, size_t>>* pools[2] = {&bufsA, &bufsB};
    for (uint64_t winStart : candStarts) {
        if (winStart + sizeNeedSpill > memSize)
            continue;
        uint64_t winEnd = winStart + sizeNeedSpill;
        std::vector<int> group;
        long long winScore = LLONG_MAX;
        if (!CollectClearableWindowGroup(pools, winStart, winEnd, allocOp, spillableCache, nextUseCache, group,
                                         winScore) ||
            group.empty())
            continue;
        if (winScore > bestScore) {
            bestScore = winScore;
            bestGroup = std::move(group);
        }
    }
    if (bestGroup.empty()) {
        return {};
    }
    return {std::move(bestGroup)};
}

std::vector<int> OoOScheduler::SelectDualDstSpillBuffers(Operation* allocOp, LocalBufferPtr allocBuffer, bool isDualDst,
                                                         CoreLocationType allocCore)
{
    CoreLocationType coreA = allocCore;
    CoreLocationType coreB = (allocCore == CoreLocationType::AIV0) ? CoreLocationType::AIV1 : CoreLocationType::AIV0;
    size_t needSize = allocBuffer->size;
    if (isDualDst) {
        DualDstAllocCtx ctx;
        if (dualDstEngine_.ResolveDualDstAllocCtx(allocOp, ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "DualDst spill select: ResolveDualDstAllocCtx failed.");
            return {};
        }
        coreA = ctx.coreA;
        coreB = ctx.coreB;
        needSize = ctx.bufA->size;
    }
    auto& poolA = state_.bufferManagerMap[coreA][MemoryType::MEM_UB];
    auto& poolB = state_.bufferManagerMap[coreB][MemoryType::MEM_UB];
    auto clearGroup = GetCommonOffsetClearGroup(poolA, poolB, needSize, allocOp);
    if (clearGroup.empty()) {
        return {};
    }
    return clearGroup.front();
}

std::vector<int> OoOScheduler::SelectSpillBuffers(Operation* allocOp)
{
    LocalBufferPtr allocBuffer = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]];
    std::vector<int> spillGroup;
    std::vector<std::vector<int>> canSpillGroups;

    bool isDualDst = state_.IsDualDstAlloc(allocOp);
    CoreLocationType allocCore = state_.schedInfoMap[allocOp].coreLocation;
    // DualDst 开启时，所有 AIV UB alloc（含非 DualDst）统一走跨池 spill 以维持双池地址对称。
    bool useDualPath = state_.enableDualDst &&
                       (isDualDst || (allocBuffer->memType == MemoryType::MEM_UB &&
                                      (allocCore == CoreLocationType::AIV0 || allocCore == CoreLocationType::AIV1)));

    if (useDualPath) {
        return SelectDualDstSpillBuffers(allocOp, allocBuffer, isDualDst, allocCore);
    }

    auto coreType = state_.schedInfoMap[allocOp].coreLocation;
    auto& pool = state_.bufferManagerMap[coreType][allocBuffer->memType];
    spillGroup = pool.GetAddrSortedBufs();
    canSpillGroups = GetSpillGroup(pool, allocBuffer->size);

    if (canSpillGroups.empty()) {
        return spillGroup;
    }
    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    for (auto& group : canSpillGroups) {
        if (GetGroupNextUseTime(group, allocOp, groupNextUseTime, nextUseTimeCache) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Operation, "Get group next use time failed, begin spill all tensor.");
            return spillGroup;
        }
    }
    size_t groupSel = std::max_element(groupNextUseTime.begin(), groupNextUseTime.end()) - groupNextUseTime.begin();
    if (groupNextUseTime[groupSel] == -1) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill, begin spill all tensor.");
        return spillGroup;
    }
    spillGroup = canSpillGroups[groupSel];
    return spillGroup;
}

Status OoOScheduler::GenBufferSpill(Operation* allocOp, SpillContext& ctx)
{
    auto reqMemType = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]]->memType;
    auto reqSize = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]]->size;
    std::vector<int> spillGroup = SelectSpillBuffers(allocOp);
    if (spillGroup.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Select buffer to spill failed.");
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    ctx.spillMemIds = spillGroup;
    for (auto& memId : spillGroup) {
        if (spillEngine_.SpillBuffer(memId, allocOp, ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Spill tensor[%d] for %s failed!", memId,
                              state_.GetOpInfo(allocOp).c_str());
            NotifyAllocFail(allocOp, reqMemType, reqSize);
            return FAILED;
        }
    }

    if (RearrangeBuffer(allocOp, reqMemType) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffer failed for %s.", GetFormatBacktrace(*allocOp).c_str());
    }
    if (!HasEnoughBuffer(allocOp, reqMemType)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Spill all buffer failed! %s", GetFormatBacktrace(*allocOp).c_str());
        if (PrintSpillFailedInfo(allocOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PrintSpillFailedInfo failed!");
            NotifyAllocFail(allocOp, reqMemType, reqSize);
            return FAILED;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Possible causes: incorrect memory reuse, memory fragmentation, or spill not supported for L0C_COPY_TO_L1."
            "Please check tile shape and OOO spill failed info. Consider avoiding cube-aligned matrix sizes.");
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::ApplySpillContext(SpillContext& ctx, Operation* allocOp)
{
    for (auto* copyoutOp : ctx.newCopyoutOps) {
        int spillMemid = copyoutOp->GetInputOperand(0)->memoryrange.memId;
        Operation* insertOp = nullptr;
        for (auto op : state_.newOperations) {
            auto& reqMemids = state_.GetOpMemIds(op);
            if (std::find(reqMemids.begin(), reqMemids.end(), spillMemid) != reqMemids.end()) {
                insertOp = op;
            }
        }
        if (insertOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Insert %s in newOperations failed.",
                              state_.GetOpInfo(copyoutOp).c_str());
            return FAILED;
        }
        auto it = find(state_.newOperations.begin(), state_.newOperations.end(), insertOp);
        if (it != state_.newOperations.end()) {
            state_.newOperations.insert(it + 1, copyoutOp);
        }
    }
    state_.numTotalIssues += ctx.newNotRetiredCopyOutSize - ctx.deleteNotRetiredOpSize;
    MemoryType memType = allocOp->GetOutputOperand(0)->GetMemoryTypeOriginal();

    for (auto deleteOp : ctx.deleteAllocOps) {
        state_.allocIssueQueue[std::get<2>(deleteOp)][std::get<1>(deleteOp)].DeleteOp(std::get<0>(deleteOp));
    }
    for (auto& newAllocOp : ctx.newAllocOps) {
        state_.allocIssueQueue[state_.schedInfoMap[newAllocOp].coreLocation][memType].Insert(newAllocOp);
    }

    for (auto memId : ctx.spillMemIds) {
        state_.localBufferMap[memId]->retireCycle = state_.clock;
    }
    return SUCCESS;
}

Status OoOScheduler::PrintSpillFailedInfo(Operation* allocOp)
{
    auto memType = state_.localBufferMap[state_.GetOpMemIds(allocOp)[0]]->memType;
    APASS_LOG_ERROR_F(Elements::Operation, "======== OoO Spill failed info ===========");
    APASS_LOG_ERROR_F(Elements::Operation, "Spill failed memoryType: %s. %s", MemoryTypeToString(memType).c_str(),
                      GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- alloc request ----");
    APASS_LOG_ERROR_F(Elements::Operation, "op:%s need buffer size: %lu. %s", state_.GetOpInfo(allocOp).c_str(),
                      state_.localBufferMap[state_.GetOpMemIds(allocOp)[0]]->size,
                      GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- current buffer occupancy ----");
    for (auto& [memId, occupyOp] : state_.tensorOccupyMap) {
        if (state_.localBufferMap[memId]->memType != memType) {
            continue;
        }
        APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d], size:%lu, range[%lu,%lu], last writer: %s. %s", memId,
                          state_.localBufferMap[memId]->size, state_.localBufferMap[memId]->start,
                          state_.localBufferMap[memId]->end, state_.GetOpInfo(occupyOp).c_str(),
                          GetFormatBacktrace(*occupyOp).c_str());
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
