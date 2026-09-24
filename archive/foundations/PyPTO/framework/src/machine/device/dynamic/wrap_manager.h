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
 * \file wrap_manager.h
 * \brief
 */

#pragma once
#include <cstdint>
#include "aicore_constants.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/device/tilefwk/core_func_data.h"
#include "core_status_manager.h"

namespace npu::tile_fwk::dynamic {

struct SchDeviceTaskContext;
using SendTaskToAiCoreFunc = std::function<void(struct SchDeviceTaskContext* devCtx, CoreType type, int coreIdx,
                                                uint64_t newTask)>;

enum class MixResourceType { MIX_UNKNOWN = 0, MIX_1C1V = 1, MIX_1C2V = 2 };
enum class DieId { DIE_0 = 0, DIE_1 = 1, DIE_MIX = 2, DIE_UNKNOWN };

inline void WrapInfoQueueLock(WrapInfoQueue* rq)
{
    while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {
    }
}

inline void WrapInfoQueueUnLock(WrapInfoQueue* rq)
{
    while (!__sync_bool_compare_and_swap(&rq->lock, 1, 0)) {
    }
}

inline uint32_t GetTaskNumByMixResType(uint8_t mixType)
{
    switch (mixType) {
        case static_cast<uint8_t>(MixResourceType::MIX_1C1V):
            return 2;
        case static_cast<uint8_t>(MixResourceType::MIX_1C2V):
            return 3;
        default:
            return 0;
    }
}

inline bool IsWrapTaskReady(const WrapInfo* wrapInfo)
{
    switch (wrapInfo->mixResourceType) {
        case static_cast<uint8_t>(MixResourceType::MIX_1C1V):
            return wrapInfo->tasklist[WRAP_IDX_AIC] != AICORE_TASK_INIT &&
                   wrapInfo->tasklist[WRAP_IDX_AIV0] != AICORE_TASK_INIT;
        case static_cast<uint8_t>(MixResourceType::MIX_1C2V):
            return wrapInfo->tasklist[WRAP_IDX_AIC] != AICORE_TASK_INIT &&
                   wrapInfo->tasklist[WRAP_IDX_AIV0] != AICORE_TASK_INIT &&
                   wrapInfo->tasklist[WRAP_IDX_AIV1] != AICORE_TASK_INIT;
        default:
            return false;
    }
}

#define RETURN_NULL_IF_NOT(val) \
    do {                        \
        if (!(val))             \
            return;             \
    } while (0)

class WrapManager {
public:
    static constexpr uint32_t MAX_DISPATCH_TASK_CNT = 8u;

    struct WrapCoreCandidates {
        uint32_t core1c1vIdx[MAX_DISPATCH_TASK_CNT];
        uint32_t core1c2vIdx[MAX_DISPATCH_TASK_CNT];
        uint32_t core1c1vCnt{0};
        uint32_t core1c2vCnt{0};
    };

    ~WrapManager() {}
    WrapManager() {}

    SchDeviceTaskContext* schDevTaskCtx{nullptr};
    CoreStatusManager* coreStatusMgr_{nullptr};
    DeviceTask* curDevTask_;
    uint32_t* pendingIds_;
    uint32_t* runningIds_;
    uint32_t aicStart_;
    uint32_t aicEnd_;

    int aicValidNum_{0};
    int curDie0MaxCpuId_{0};
    int curDie1StartCpuId_{0};
    DieId dieId_{DieId::DIE_MIX};

    WrapInfoQueue* readyWrapCoreFunctionQue_{nullptr};
    SendTaskToAiCoreFunc SendTaskToAiCore;
    bool isOpenMixSche{false};
    ArchInfo archInfo;
    int schedIdx_;

    // for die-to-die shchedule
    ReadyCoreFunctionQueue* readyDieAicFunctionQue_[DIE_NUM] = {nullptr};
    ReadyCoreFunctionQueue* readyDieAivFunctionQue_[DIE_NUM] = {nullptr};
    ReadyCoreFunctionQueue* selectReadyDieAicFunctionQue_{nullptr};
    ReadyCoreFunctionQueue* selectReadyDieAivFunctionQue_{nullptr};

    inline void InitDeviceInfo(DeviceArgs* deviceArgs, int schedIdx)
    {
        archInfo = deviceArgs->archInfo;
        InitDieMaxCpuId(static_cast<int>(deviceArgs->scheCpuNum));
        InitDieId(schedIdx);
        schedIdx_ = schedIdx;
    }

    inline void InitDieMaxCpuId(int scheCpuNum)
    {
        curDie0MaxCpuId_ = scheCpuNum >> 1;
        // In odd scenes, scheCpuIdx = curDie0MaxCpuId_ is DIE_MIX, else is DIE_1
        curDie1StartCpuId_ = (scheCpuNum & 1) ? curDie0MaxCpuId_ + 1 : curDie0MaxCpuId_;
    }

    inline void InitDieId(int schedIdx)
    {
        if (schedIdx < curDie0MaxCpuId_) {
            dieId_ = DieId::DIE_0;
        } else if (schedIdx >= curDie1StartCpuId_) {
            dieId_ = DieId::DIE_1;
        } else {
            dieId_ = DieId::DIE_MIX;
        }
    }

    inline void GetDieSchedIdRange(int& schedStart, int& schedEnd, int scheCpuNum)
    {
        if (dieId_ == DieId::DIE_0) {
            schedStart = 0;
            schedEnd = curDie0MaxCpuId_;
        } else if (dieId_ == DieId::DIE_1) {
            schedStart = curDie1StartCpuId_;
            schedEnd = scheCpuNum;
        }
    }

    inline DieId GetDieId() { return dieId_; }

    inline void Init(SchDeviceTaskContext* devTaskctx, DeviceTask* curDevTask, CoreStatusManager* coreStatusMgr,
                     uint32_t* pendingIds, uint32_t* runningIds, int aicValidNum, uint32_t aicStart, uint32_t aicEnd,
                     SendTaskToAiCoreFunc func)
    {
        if (archInfo != ArchInfo::DAV_3510)
            return;
        schDevTaskCtx = devTaskctx;
        isOpenMixSche = curDevTask->mixTaskData.wrapIdNum > 0;
        curDevTask_ = curDevTask;
        coreStatusMgr_ = coreStatusMgr;
        pendingIds_ = pendingIds;
        runningIds_ = runningIds;
        aicEnd_ = aicEnd;
        aicStart_ = aicStart;

        aicValidNum_ = aicValidNum;
        SendTaskToAiCore = func;
        readyWrapCoreFunctionQue_ = reinterpret_cast<WrapInfoQueue*>(curDevTask_->mixTaskData.readyWrapCoreFunctionQue);

        SetDieReadyQueue(curDevTask->dieReadyFunctionQue);

        selectReadyDieAicFunctionQue_ = GetDieReadyQueue(
            CoreType::AIC, reinterpret_cast<ReadyCoreFunctionQueue*>(curDevTask->readyAicCoreFunctionQue));
        selectReadyDieAivFunctionQue_ = GetDieReadyQueue(
            CoreType::AIV, reinterpret_cast<ReadyCoreFunctionQueue*>(curDevTask->readyAivCoreFunctionQue));
    }

    inline bool IsMixArch() { return archInfo == ArchInfo::DAV_3510; }

    inline bool GetAvailableWrapCoreIdx(uint8_t mixResourceType, uint32_t aicReadyCnt, uint32_t& coreIdx,
                                        uint32_t& v0Idx)
    {
        for (uint32_t idx = 0; idx < aicReadyCnt; idx++) {
            uint32_t aicIdx = coreStatusMgr_->GetRunReadyCoreIdx(CORE_IDX_AIC, idx);
            uint32_t aivIdx0 = aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
            switch (mixResourceType) {
                case static_cast<uint8_t>(MixResourceType::MIX_1C1V):
                    if (coreStatusMgr_->GetCoreIdxPosition(aivIdx0) != INVALID_COREIDX_POSITION) {
                        coreIdx = aicIdx;
                        v0Idx = aivIdx0;
                        return true;
                    }
                    break;
                case static_cast<uint8_t>(MixResourceType::MIX_1C2V):
                    if (coreStatusMgr_->GetCoreIdxPosition(aivIdx0) != INVALID_COREIDX_POSITION &&
                        coreStatusMgr_->GetCoreIdxPosition(aivIdx0 + 1) != INVALID_COREIDX_POSITION) {
                        coreIdx = aicIdx;
                        v0Idx = aivIdx0;
                        return true;
                    }
                    break;
                default:
                    break;
            }
        }
        return false;
    }

    inline uint32_t GetWrapCoreRunningCnt(WrapCoreCandidates& candidates, uint32_t& core1c1vCnt, uint32_t& core1c2vCnt)
    {
        uint32_t aicReadyCnt = coreStatusMgr_->GetCoreRunReadyCnt(CORE_IDX_AIC);
        for (uint32_t idx = 0; idx < aicReadyCnt && core1c2vCnt < MAX_DISPATCH_TASK_CNT; idx++) {
            uint32_t aicIdx = coreStatusMgr_->GetRunReadyCoreIdx(CORE_IDX_AIC, idx);
            uint32_t aivIdx0 = GetWrapAiv0CoreIdx(aicIdx);
            uint32_t aivIdx1 = GetWrapAiv1CoreIdx(aivIdx0);
            if (coreStatusMgr_->GetCoreIdxPosition(aivIdx0) != INVALID_COREIDX_POSITION) {
                CheckCoreIdxInitStatus(aicIdx);
                CheckCoreIdxInitStatus(aivIdx0);
                if (coreStatusMgr_->GetCoreIdxPosition(aivIdx1) != INVALID_COREIDX_POSITION) {
                    CheckCoreIdxInitStatus(aivIdx1);
                    candidates.core1c2vIdx[core1c2vCnt] = aicIdx;
                    core1c2vCnt++;
                } else if (core1c1vCnt < MAX_DISPATCH_TASK_CNT) {
                    candidates.core1c1vIdx[core1c1vCnt] = aicIdx;
                    core1c1vCnt++;
                }
            }
        }
        candidates.core1c1vCnt = core1c1vCnt;
        candidates.core1c2vCnt = core1c2vCnt;
        return core1c2vCnt + core1c1vCnt;
    }

    inline uint32_t GetWrapCorePendingCnt(WrapCoreCandidates& candidates, uint32_t& core1c1vCnt, uint32_t& core1c2vCnt)
    {
        for (uint32_t idx = aicStart_; idx < aicEnd_ && core1c2vCnt < MAX_DISPATCH_TASK_CNT; idx++) {
            uint32_t aivIdx0 = GetWrapAiv0CoreIdx(idx);
            uint32_t aivIdx1 = GetWrapAiv1CoreIdx(aivIdx0);
            if (pendingIds_[idx] == AICORE_TASK_INIT && pendingIds_[aivIdx0] == AICORE_TASK_INIT) {
                if (pendingIds_[aivIdx1] == AICORE_TASK_INIT) {
                    candidates.core1c2vIdx[core1c2vCnt] = idx;
                    core1c2vCnt++;
                } else if (core1c1vCnt < MAX_DISPATCH_TASK_CNT) {
                    candidates.core1c1vIdx[core1c1vCnt] = idx;
                    core1c1vCnt++;
                }
            }
        }
        candidates.core1c1vCnt = core1c1vCnt;
        candidates.core1c2vCnt = core1c2vCnt;
        return core1c1vCnt + core1c2vCnt;
    }

    // 根据Wrap中任意核找对对应的AIC核
    inline uint16_t GetWrapAicCoreIdx(uint16_t coreIdx)
    {
        return coreIdx < aicValidNum_ ? coreIdx : (coreIdx - aicValidNum_) / AIV_NUM_PER_AI_CORE;
    }

    inline uint16_t GetWrapAiv0CoreIdx(uint16_t aicIdx) { return aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_; }
    inline uint16_t GetWrapAiv1CoreIdx(uint16_t aiv0Idx) { return aiv0Idx + 1; }

    inline void SendMixTasksToCore(const uint32_t* taskIds, uint8_t mixResourceType, uint32_t aicIdx, uint32_t aivIdx0)
    {
        DEV_IF_VERBOSE_DEBUG
        {
            uint32_t wrapId = static_cast<uint32_t>(GetWrapId(taskIds[WRAP_IDX_AIC]));
            DEV_VERBOSE_DEBUG("#trace.mix.send: tid=%d wrapId=%u mixType=%u aicIdx=%u aivIdx0=%u aivIdx1=%u "
                              "taskAic=%u taskAiv0=%u taskAiv1=%u",
                              schedIdx_, wrapId, static_cast<uint32_t>(mixResourceType), aicIdx, aivIdx0, aivIdx0 + 1,
                              taskIds[WRAP_IDX_AIC], taskIds[WRAP_IDX_AIV0], taskIds[WRAP_IDX_AIV1]);
        }
        SendTaskToAiCore(schDevTaskCtx, CoreType::AIC, aicIdx, taskIds[WRAP_IDX_AIC]);
        SendTaskToAiCore(schDevTaskCtx, CoreType::AIV, aivIdx0, taskIds[WRAP_IDX_AIV0]);
        if (mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C2V)) {
            SendTaskToAiCore(schDevTaskCtx, CoreType::AIV, aivIdx0 + 1, taskIds[WRAP_IDX_AIV1]);
        }
    }

    inline void RemoveMixRunAndPendCoreIdx(uint8_t mixResourceType, uint32_t aicIdx, uint32_t aivIdx0)
    {
        coreStatusMgr_->RemoveRunAndPendCoreIdx(aicIdx, CORE_IDX_AIC);
        coreStatusMgr_->RemoveRunAndPendCoreIdx(aivIdx0, CORE_IDX_AIV);
        DEV_VERBOSE_DEBUG("#trace.mix.release: tid=%d core=%u type=%d allFinish=%d", schedIdx_, aicIdx,
                          static_cast<int>(CoreType::AIC), 0);
        DEV_VERBOSE_DEBUG("#trace.mix.release: tid=%d core=%u type=%d allFinish=%d", schedIdx_, aivIdx0,
                          static_cast<int>(CoreType::AIV), 0);
        if (mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C2V)) {
            coreStatusMgr_->RemoveRunAndPendCoreIdx(aivIdx0 + 1, CORE_IDX_AIV);
            DEV_VERBOSE_DEBUG("#trace.mix.release: tid=%d core=%u type=%d allFinish=%d", schedIdx_, aivIdx0 + 1,
                              static_cast<int>(CoreType::AIV), 0);
        }
    }

    inline void TryRemoveCoreIdxFromPending(uint32_t coreIdx, uint32_t coreType)
    {
        if (unlikely(coreStatusMgr_->GetCoreIdxPosition(coreIdx) != INVALID_COREIDX_POSITION)) {
            coreStatusMgr_->RemoveRunAndPendCoreIdx(coreIdx, coreType);
        } else {
            coreStatusMgr_->RemovePendReadyCoreIdx(coreType);
        }
    }

    inline void TryRemoveMixCoreIdxFromPending(uint8_t mixResourceType, uint32_t aicIdx, uint32_t aivIdx0)
    {
        TryRemoveCoreIdxFromPending(aicIdx, CORE_IDX_AIC);
        TryRemoveCoreIdxFromPending(aivIdx0, CORE_IDX_AIV);
        if (mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C2V)) {
            TryRemoveCoreIdxFromPending(aivIdx0 + 1, CORE_IDX_AIV);
        }
    }

    template <typename RemoveMixCoreIdxFunc>
    inline void DispatchWrapTasksImpl(WrapCoreCandidates& candidates, WrapInfo* wrap1c2vTasks[], uint32_t task1c2vCnt,
                                      WrapInfo* wrap1c1vTasks[], uint32_t task1c1vCnt,
                                      RemoveMixCoreIdxFunc removeMixCoreIdx)
    {
        for (uint32_t taskIdx = 0; taskIdx < task1c2vCnt; taskIdx++) {
            WrapInfo* wrapInfo = wrap1c2vTasks[taskIdx];
            uint32_t aicIdx = candidates.core1c2vIdx[taskIdx];
            uint32_t aivIdx0 = GetWrapAiv0CoreIdx(aicIdx);
            DEV_IF_VERBOSE_DEBUG
            {
                uint32_t wrapId = wrapInfo->wrapId;
                DEV_VERBOSE_DEBUG("#trace.mix.wrapresolve: tid=%d wrapId=%u wrapAicoreIdx=%u taskId=%u", schedIdx_,
                                  wrapId, static_cast<uint32_t>(WRAP_IDX_AIC), wrapInfo->tasklist[WRAP_IDX_AIC]);
            }
            SendMixTasksToCore(wrapInfo->tasklist, wrapInfo->mixResourceType, aicIdx, aivIdx0);
            removeMixCoreIdx(wrapInfo->mixResourceType, aicIdx, aivIdx0);
        }

        uint32_t core1c2vIdx = task1c2vCnt;
        uint32_t core1c1vIdx = 0;
        for (uint32_t taskIdx = 0; taskIdx < task1c1vCnt; taskIdx++) {
            WrapInfo* wrapInfo = wrap1c1vTasks[taskIdx];
            uint32_t aicIdx = core1c2vIdx < candidates.core1c2vCnt ? candidates.core1c2vIdx[core1c2vIdx++] :
                                                                     candidates.core1c1vIdx[core1c1vIdx++];
            uint32_t aivIdx0 = GetWrapAiv0CoreIdx(aicIdx);
            DEV_IF_VERBOSE_DEBUG
            {
                uint32_t wrapId = wrapInfo->wrapId;
                DEV_VERBOSE_DEBUG("#trace.mix.wrapresolve: tid=%d wrapId=%u wrapAicoreIdx=%u taskId=%u", schedIdx_,
                                  wrapId, static_cast<uint32_t>(WRAP_IDX_AIC), wrapInfo->tasklist[WRAP_IDX_AIC]);
            }
            SendMixTasksToCore(wrapInfo->tasklist, wrapInfo->mixResourceType, aicIdx, aivIdx0);
            removeMixCoreIdx(wrapInfo->mixResourceType, aicIdx, aivIdx0);
        }
    }

    inline void DispatchWrapTasksFromReadyCore(WrapCoreCandidates& candidates, WrapInfo* wrap1c2vTasks[],
                                               uint32_t task1c2vCnt, WrapInfo* wrap1c1vTasks[], uint32_t task1c1vCnt)
    {
        DispatchWrapTasksImpl(candidates, wrap1c2vTasks, task1c2vCnt, wrap1c1vTasks, task1c1vCnt,
                              [this](uint8_t mixResourceType, uint32_t aicIdx, uint32_t aivIdx0) {
                                  RemoveMixRunAndPendCoreIdx(mixResourceType, aicIdx, aivIdx0);
                              });
    }

    inline void DispatchWrapTasksFromPendingCore(WrapCoreCandidates& candidates, WrapInfo* wrap1c2vTasks[],
                                                 uint32_t task1c2vCnt, WrapInfo* wrap1c1vTasks[], uint32_t task1c1vCnt)
    {
        DispatchWrapTasksImpl(candidates, wrap1c2vTasks, task1c2vCnt, wrap1c1vTasks, task1c1vCnt,
                              [this](uint8_t mixResourceType, uint32_t aicIdx, uint32_t aivIdx0) {
                                  TryRemoveMixCoreIdxFromPending(mixResourceType, aicIdx, aivIdx0);
                              });
    }

    inline void CheckCoreIdxInitStatus(uint32_t coreIdx)
    {
        DEV_IF_VERBOSE_DEBUG
        {
            if (pendingIds_[coreIdx] != AICORE_TASK_INIT || runningIds_[coreIdx] != AICORE_TASK_INIT) {
                DEV_ERROR(CtrlErr::TASK_STATS_ABNORMAL,
                          "#sche.task.run.wrap.stats: core[%u]: pendingId=%x, runningId=%x, is illegal!", coreIdx,
                          pendingIds_[coreIdx], runningIds_[coreIdx]);
            }
        }
    }

    template <typename GetCoreCntFunc, typename PostProcessFunc>
    inline bool DispatchReadyTasksImpl(GetCoreCntFunc getCoreCnt, PostProcessFunc postProcess)
    {
        uint32_t head = __atomic_load_n(&readyWrapCoreFunctionQue_->head, __ATOMIC_RELAXED);
        uint32_t tail = __atomic_load_n(&readyWrapCoreFunctionQue_->tail, __ATOMIC_RELAXED);
        if (tail - head == 0) {
            return false;
        }
        uint32_t core1c1vCnt = 0;
        uint32_t core1c2vCnt = 0;
        uint32_t wrapCoreCnt = getCoreCnt(core1c1vCnt, core1c2vCnt);
        if (wrapCoreCnt == 0)
            return true;

        WrapInfoQueueLock(readyWrapCoreFunctionQue_);
        head = readyWrapCoreFunctionQue_->head;
        uint32_t taskCount = readyWrapCoreFunctionQue_->tail - head;
        if (unlikely(taskCount == 0)) {
            DEV_VERBOSE_DEBUG("mixcore taskCount is zero.");
            WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
            return false;
        }
        WrapInfo* localTasks[MAX_DISPATCH_TASK_CNT];
        uint32_t maxReadyCnt = taskCount > MAX_DISPATCH_TASK_CNT ? MAX_DISPATCH_TASK_CNT : taskCount;
        maxReadyCnt = maxReadyCnt > wrapCoreCnt ? wrapCoreCnt : maxReadyCnt;
        uint32_t taskHead = 0, taskTail = maxReadyCnt;
        while (taskHead < taskTail) {
            WrapInfo* info = &readyWrapCoreFunctionQue_->elem[head++];
            DEV_IF_VERBOSE_DEBUG
            {
                DEV_ASSERT_MSG(CtrlErr::TASK_STATS_ABNORMAL, IsWrapTaskReady(info),
                               "#sche.task.wrap.not_ready: wrapId=%u has unresolved task", info->wrapId);
            }
            if (info->mixResourceType == static_cast<uint32_t>(MixResourceType::MIX_1C2V)) {
                if (core1c2vCnt == 0)
                    break;
                localTasks[taskHead++] = info;
                core1c2vCnt--;
            } else {
                localTasks[--taskTail] = info;
            }
        }
        uint32_t valid1c1vCnt = maxReadyCnt - taskTail;
        uint32_t validReadyCnt = taskHead + valid1c1vCnt;
        readyWrapCoreFunctionQue_->head += validReadyCnt;
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);

        DEV_VERBOSE_DEBUG("#trace.mix.dispatch: tid=%d readyWrapQueHead=%u readyWrapQueTail=%u", schedIdx_,
                          readyWrapCoreFunctionQue_->head, readyWrapCoreFunctionQue_->tail);

        WrapInfo** task1c1vPtr = localTasks + taskTail;
        postProcess(localTasks, taskHead, task1c1vPtr, valid1c1vCnt);
        return taskCount > maxReadyCnt;
    }

    inline bool TryAllocRunningCoreAndSend()
    {
        WrapCoreCandidates candidates;
        return DispatchReadyTasksImpl(
            [this, &candidates](uint32_t& c1c1v, uint32_t& c1c2v) {
                return GetWrapCoreRunningCnt(candidates, c1c1v, c1c2v);
            },
            [this, &candidates](WrapInfo* a[], uint32_t b, WrapInfo* c[], uint32_t d) {
                DispatchWrapTasksFromReadyCore(candidates, a, b, c, d);
            });
    }

    inline void TryAllocPendingCoreAndSend()
    {
        WrapCoreCandidates candidates;
        DispatchReadyTasksImpl(
            [this, &candidates](uint32_t& c1c1v, uint32_t& c1c2v) {
                return GetWrapCorePendingCnt(candidates, c1c1v, c1c2v);
            },
            [this, &candidates](WrapInfo* a[], uint32_t b, WrapInfo* c[], uint32_t d) {
                DispatchWrapTasksFromPendingCore(candidates, a, b, c, d);
            });
    }

    inline void DispatchMixCoreTask()
    {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        bool hasAvailTask = TryAllocRunningCoreAndSend();
        if (hasAvailTask) {
            TryAllocPendingCoreAndSend();
        }
    }

    int32_t GetWrapId(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto opWrapList = reinterpret_cast<int32_t*>(dyntask->devTask.mixTaskData.opWrapList[funcId]);
        if (opWrapList[opIndex] != -1) {
            return MakeMixWrapID(funcId, opWrapList[opIndex]);
        } else {
            return -1;
        }
    }

    inline static int32_t GetWrapAicoreIdx(uint32_t coreType, int32_t wrapVecId)
    {
        if (coreType == static_cast<uint32_t>(CoreType::AIC)) {
            return WRAP_IDX_AIC;
        } else {
            return wrapVecId == 1 ? WRAP_IDX_AIV1 : WRAP_IDX_AIV0;
        }
    }

    inline void PushMixToReadyQueue(const uint32_t* taskIds, uint8_t mixResourceType, uint32_t wrapId)
    {
        WrapInfoQueueLock(readyWrapCoreFunctionQue_);
        WrapInfo* wrapInfo = &readyWrapCoreFunctionQue_->elem[readyWrapCoreFunctionQue_->tail++];
        wrapInfo->mixResourceType = mixResourceType;
        wrapInfo->tasklist[WRAP_IDX_AIC] = taskIds[WRAP_IDX_AIC];
        wrapInfo->tasklist[WRAP_IDX_AIV0] = taskIds[WRAP_IDX_AIV0];
        wrapInfo->tasklist[WRAP_IDX_AIV1] = taskIds[WRAP_IDX_AIV1];
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
        wrapInfo->wrapId = wrapId;
        DEV_VERBOSE_DEBUG("#trace.mix.wrapcreate: tid=%d wrapId=%u mixType=%u taskAic=%u taskAiv0=%u taskAiv1=%u",
                          schedIdx_, wrapId, static_cast<uint32_t>(mixResourceType), taskIds[WRAP_IDX_AIC],
                          taskIds[WRAP_IDX_AIV0], taskIds[WRAP_IDX_AIV1]);
    }

    inline void ResolveDepForOneMix(const uint32_t* taskIds, uint8_t mixResourceType, uint16_t coreIdx)
    {
        uint32_t aicIdx = GetWrapAicCoreIdx(coreIdx);
        uint32_t aivIdx0 = aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
        bool is1c1v = mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C1V);
        bool coreAvailable = coreStatusMgr_->GetCoreIdxPosition(aicIdx) != INVALID_COREIDX_POSITION &&
                             coreStatusMgr_->GetCoreIdxPosition(aivIdx0) != INVALID_COREIDX_POSITION;
        // 1C2V additionally requires the second AIV core.
        if (!is1c1v) {
            coreAvailable = coreAvailable &&
                            coreStatusMgr_->GetCoreIdxPosition(aivIdx0 + 1) != INVALID_COREIDX_POSITION;
        }

        if (unlikely(!coreAvailable)) {
            bool pendingCoreAvailable = pendingIds_[aicIdx] == AICORE_TASK_INIT &&
                                        pendingIds_[aivIdx0] == AICORE_TASK_INIT;
            if (!is1c1v) {
                pendingCoreAvailable = pendingCoreAvailable && pendingIds_[aivIdx0 + 1] == AICORE_TASK_INIT;
            }
            // Dispatch directly when all required pending cores are available.
            if (pendingCoreAvailable) {
                SendMixTasksToCore(taskIds, mixResourceType, aicIdx, aivIdx0);
                TryRemoveMixCoreIdxFromPending(mixResourceType, aicIdx, aivIdx0);
                return;
            }

            uint32_t aicReadyCnt = coreStatusMgr_->GetCoreRunReadyCnt(CORE_IDX_AIC);
            if (!GetAvailableWrapCoreIdx(mixResourceType, aicReadyCnt, aicIdx, aivIdx0)) {
                auto wrapId = static_cast<uint32_t>(GetWrapId(taskIds[WRAP_IDX_AIC]));
                DEV_VERBOSE_DEBUG("#trace.mix.resolve: tid=%d wrapId=%u aicTaskId=%u aiv0TaskId=%u aiv1TaskId=%u "
                                  "mixType=%u directSend=%d",
                                  schedIdx_, wrapId, taskIds[WRAP_IDX_AIC], taskIds[WRAP_IDX_AIV0],
                                  taskIds[WRAP_IDX_AIV1], static_cast<uint32_t>(mixResourceType), 0);
                PushMixToReadyQueue(taskIds, mixResourceType, wrapId);
                return;
            }
        }
        DEV_IF_VERBOSE_DEBUG
        {
            auto wrapIdDirect = static_cast<uint32_t>(GetWrapId(taskIds[WRAP_IDX_AIC]));
            DEV_VERBOSE_DEBUG("#trace.mix.resolve: tid=%d wrapId=%u aicTaskId=%u aiv0TaskId=%u aiv1TaskId=%u "
                              "mixType=%u directSend=%d",
                              schedIdx_, wrapIdDirect, taskIds[WRAP_IDX_AIC], taskIds[WRAP_IDX_AIV0],
                              taskIds[WRAP_IDX_AIV1], static_cast<uint32_t>(mixResourceType), 1);
        }
        SendMixTasksToCore(taskIds, mixResourceType, aicIdx, aivIdx0);
        RemoveMixRunAndPendCoreIdx(mixResourceType, aicIdx, aivIdx0);
    }

    // for die-to-die schedule
    inline void SetDieReadyQueue(const struct DieReadyQueueData dieReadyFunctionQue)
    {
        for (size_t i = 0; i < DIE_NUM; i++) {
            readyDieAivFunctionQue_[i] = reinterpret_cast<ReadyCoreFunctionQueue*>(
                dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
            readyDieAicFunctionQue_[i] = reinterpret_cast<ReadyCoreFunctionQueue*>(
                dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);
        }
    }

    inline ReadyCoreFunctionQueue* GetDieReadyQueue(CoreType type, ReadyCoreFunctionQueue* defaultReadyQue)
    {
        if (!IsMixArch() || dieId_ == DieId::DIE_MIX || dieId_ == DieId::DIE_UNKNOWN) {
            return defaultReadyQue;
        }
        size_t dieIndex = static_cast<size_t>(dieId_);
        ReadyCoreFunctionQueue* dieReadyQueue = nullptr;
        switch (type) {
            case CoreType::AIC:
                dieReadyQueue = readyDieAicFunctionQue_[dieIndex];
                break;
            case CoreType::AIV:
                dieReadyQueue = readyDieAivFunctionQue_[dieIndex];
                break;
            default:
                break;
        }
        return (dieReadyQueue != nullptr) ? dieReadyQueue : defaultReadyQue;
    }

    ReadyCoreFunctionQueue* GetDieReadyAicQue() { return selectReadyDieAicFunctionQue_; }
    ReadyCoreFunctionQueue* GetDieReadyAivQue() { return selectReadyDieAivFunctionQue_; }

    // Index into die-ready queues (DIE_0 / DIE_1). Used when DIE_MIX must drain both.
    inline ReadyCoreFunctionQueue* GetDieReadyQue(CoreType type, size_t dieIndex) const
    {
        if (dieIndex >= DIE_NUM) {
            return nullptr;
        }
        return (type == CoreType::AIC) ? readyDieAicFunctionQue_[dieIndex] : readyDieAivFunctionQue_[dieIndex];
    }
};

} // namespace npu::tile_fwk::dynamic
