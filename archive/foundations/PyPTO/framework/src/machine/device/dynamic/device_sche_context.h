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
 * \file device_machine.h
 * \brief
 */

#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"
#include "aicpu_task_manager.h"
#include "tilefwk/aicpu_common.h"
#include "aicore_constants.h"
#include "core_status_manager.h"
#include "wrap_manager.h"

namespace npu::tile_fwk::dynamic {
const uint32_t READY_ID_FIX_CACHE_NUM = 256;
enum class DevTaskExecStage {
    INIT = 0,
    SEND_CORE_TASK = 1,
    WAIT_TAIL_TASK_FINISH = 2,
    WAIT_ALL_SCH_FINISH = 3,
    FINISH = 4
};

struct SchDeviceTaskContext {
    DeviceTaskCtrl* taskCtrl{nullptr};
    ReadyCoreFunctionQueue* readyAicCoreFunctionQue{nullptr};
    ReadyCoreFunctionQueue* readyAivCoreFunctionQue{nullptr};
    ReadyCoreFunctionQueue* readyAicpuFunctionQue{nullptr};

    uint32_t bindParallelCtxVersion{0};
    DevTaskExecStage curStage{DevTaskExecStage::INIT};
    uint16_t lastSent{0};
    uint16_t allSent{0};
    uint16_t aicpuTaskSendCnt{0};
    uint16_t resolveHubCnt{0};
    uint8_t sendCnt[AICORE_TYPE_NUM]{0, 0};
    uint8_t parallelIdx{0};
    uint8_t isFirstTaskSend{0};
    uint8_t isFree{1};
    uint8_t coreFinishedNum{0};
    std::array<uint8_t, MAX_AICORE_NUM> coreTaskFinished;
    uint16_t readyCount[AICORE_TYPE_NUM]{0, 0};
    uint32_t readyIds[AICORE_TYPE_NUM][READY_ID_FIX_CACHE_NUM];

    WrapManager wrapManager;

    DeviceTaskCtrl* GetDeviceTaskCtrl() { return taskCtrl; }
    DeviceTask* GetDeviceTask() { return taskCtrl->devTask; }
    uint32_t CoreTaskCnt() { return taskCtrl->devTask->coreFunctionCnt; }
    WrapManager& GetWrapManager() { return wrapManager; }
    void BindParallelCtxVersion(uint32_t version) { bindParallelCtxVersion = version; }
    bool IsStage(DevTaskExecStage stage) { return curStage == stage; }
    DevTaskExecStage CurStage() { return curStage; }
    void EntryStage(DevTaskExecStage stage) { curStage = stage; }
    bool IsParallel() { return taskCtrl->ParallelForId() != 0; }
    bool IsRunFinish() { return curStage == DevTaskExecStage::FINISH; }
    void Free(int scheNum)
    {
        isFree = 1;
        taskCtrl->Free(scheNum);
    }
    bool IsFree() { return static_cast<bool>(isFree); }

    void BindTaskCtrl(struct DeviceTaskCtrl* inputTaskCtrl)
    {
        Init();
        taskCtrl = inputTaskCtrl;
        isFirstTaskSend = 0;
        readyAicCoreFunctionQue = reinterpret_cast<ReadyCoreFunctionQueue*>(taskCtrl->devTask->readyAicCoreFunctionQue);
        readyAivCoreFunctionQue = reinterpret_cast<ReadyCoreFunctionQueue*>(taskCtrl->devTask->readyAivCoreFunctionQue);
        readyAicpuFunctionQue = reinterpret_cast<ReadyCoreFunctionQueue*>(taskCtrl->devTask->readyAicpuFunctionQue);
        isFree = 0;
    }

    void Init()
    {
        taskCtrl = nullptr;
        aicpuTaskSendCnt = 0;
        resolveHubCnt = 0;
        lastSent = 0;
        allSent = 0;
        curStage = DevTaskExecStage::INIT;
        coreTaskFinished.fill(0);
        coreFinishedNum = 0;
    }

    void CountCoreTaskSent(uint32_t& cntAic, uint32_t& cntAiv)
    {
        uint8_t sentAic = sendCnt[static_cast<int>(CoreType::AIC)];
        uint8_t sentAiv = sendCnt[static_cast<int>(CoreType::AIV)];

        cntAic += sentAic;
        cntAiv += sentAiv;
        uint16_t curSent = sentAic + sentAiv + aicpuTaskSendCnt;

        aicpuTaskSendCnt = 0;
        sendCnt[static_cast<int>(CoreType::AIC)] = 0;
        sendCnt[static_cast<int>(CoreType::AIV)] = 0;

        if (likely(curSent == 0)) {
            if (lastSent > 0) {
                taskCtrl->finishedFunctionCnt.fetch_add(lastSent, std::memory_order_relaxed);
                lastSent = 0;
            }
        } else {
            lastSent += curSent;
        }

        allSent = taskCtrl->finishedFunctionCnt.load(std::memory_order_relaxed) + lastSent;
    }

    bool NeedSendCoreTask() { return (allSent < taskCtrl->devTask->coreFunctionCnt); }
    uint32_t CurCoreTaskSent(CoreType type) { return sendCnt[static_cast<int>(type)]; }
    void SetAicpuTaskSent(uint32_t taskCnt) { aicpuTaskSendCnt = taskCnt; }
    void SyncAllSchCoreTaskSent()
    {
        if (lastSent > 0) {
            taskCtrl->finishedFunctionCnt.fetch_add(lastSent, std::memory_order_relaxed);
            lastSent = 0;
        }
    }
    bool IsCoreTaskSendFinish() { return (allSent >= CoreTaskCnt()); }
    uint64_t TaskId() { return taskCtrl != nullptr ? taskCtrl->taskId : 0; }
    void Dump()
    {
        DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                  "Devtask:parallelidx=%u, taskid=%lu,ver=%u,forid=%u,iterid=%u,wsid=%u,allsent=%u,total=%lu,stage=%d,"
                  "%s, %s",
                  parallelIdx, taskCtrl->taskId, bindParallelCtxVersion, taskCtrl->ParallelForId(),
                  taskCtrl->ParallelIterId(), taskCtrl->ParallelWsId(), allSent, taskCtrl->devTask->coreFunctionCnt,
                  ToUnderlying(curStage), readyAicCoreFunctionQue->Str().c_str(),
                  readyAivCoreFunctionQue->Str().c_str());
    }
};

// use ring buffer to control parallel multi devtask
struct ParallelSchDeviceTaskContext {
    uint32_t version{0}; // mark ctx version
    uint32_t front{0};
    uint32_t rear{0};
    SchDeviceTaskContext elements[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM];

    void Init(DeviceArgs* deviceArgs, int schedIdx)
    {
        for (uint32_t idx = 0; idx < npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM; idx++) {
            elements[idx].parallelIdx = idx;
            elements[idx].wrapManager.InitDeviceInfo(deviceArgs, schedIdx);
        }
    }
    uint32_t Version() { return version; }
    void UpdateVersion() { version++; }
    bool Empty() { return (front == rear); }
    uint32_t Num() { return (rear - front); }
    bool Full() { return (rear - front) == npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM; }
    SchDeviceTaskContext* RearElement() { return &elements[rear % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM]; }
    SchDeviceTaskContext* Element(uint32_t idx) { return &elements[idx % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM]; }
    SchDeviceTaskContext* FrontElement() { return &elements[front % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM]; }
    void PopFront()
    {
        if (++front == rear) {
            front = 0;
            rear = 0;
        }
    }

    bool EnqueueSchDeviceTask(DeviceTaskCtrl* taskCtrl)
    {
        if (Full()) {
            DEV_ERROR(SchedErr::SCH_DEVTASK_CTX_FULL, "Parallel sch device task ctx is full.");
            return false;
        }
        DEV_INFO("Parallel ctx enque device task %lu, forid=%u, iterid=%u, wsid=%u.", taskCtrl->taskId,
                 taskCtrl->ParallelForId(), taskCtrl->ParallelIterId(), taskCtrl->ParallelWsId());
        auto& ctx = elements[(rear++) % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM];
        ctx.BindTaskCtrl(taskCtrl);
        UpdateVersion();
        ctx.BindParallelCtxVersion(version);
        return true;
    }

    void RecycleFreeContexts()
    {
        while (!Empty()) {
            SchDeviceTaskContext* frontCtx = FrontElement();
            if (frontCtx->IsFree()) {
                PopFront();
                DEV_VERBOSE_DEBUG(
                    "Recycling parallel context parallel_idx=%u, version=%u, devtaskid=%lu, leftCtxNum=%u",
                    frontCtx->parallelIdx, frontCtx->bindParallelCtxVersion, frontCtx->TaskId(), Num());

            } else {
                // 遇到非free的context，停止回收
                DEV_VERBOSE_DEBUG(
                    "Stop recycling at non-free context parallel_idx=%u, stage=%d, devtaskid=%lu, ctxNum=%u",
                    frontCtx->parallelIdx, static_cast<int>(frontCtx->curStage), frontCtx->TaskId(), Num());
                break;
            }
        }
    }

    void GatherParallelDevTaskData(int64_t* localFuncData, uint32_t* localDevTaskIds)
    {
        for (uint32_t idx = front; idx < rear; idx++) {
            uint32_t slot = idx % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM;
            auto dyntask = (DynDeviceTask*)(elements[idx].GetDeviceTask());
            localFuncData[slot] = static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()));
            localDevTaskIds[slot] = dyntask->GetIndex();
        }
    }
};

struct SchduleContext {
    CoreStatusManager coreStatusMgr;
    SchDeviceTaskContext* curSchDevTaskCtx;
    ParallelSchDeviceTaskContext schParallelDevTaskCtx;

    void Init(DeviceArgs* deviceArgs, int schedIdx) { schParallelDevTaskCtx.Init(deviceArgs, schedIdx); }

    SchDeviceTaskContext* ParallelDeviceTaskCtx(uint32_t parallelIdx)
    {
        return schParallelDevTaskCtx.Element(parallelIdx);
    }

    void UpdateParallelVersion() { schParallelDevTaskCtx.UpdateVersion(); }
    uint32_t PrallelVersion() { return schParallelDevTaskCtx.Version(); }

    SchDeviceTaskContext* GetCurSchDevTaskCtx() { return curSchDevTaskCtx; }
    void SetCurSchDevTaskCtx(SchDeviceTaskContext* ctx) { curSchDevTaskCtx = ctx; }

    bool CurSupportParallel()
    {
        if (schParallelDevTaskCtx.Empty()) {
            return true;
        }
        return schParallelDevTaskCtx.FrontElement()->IsParallel();
    }

    bool CanParallelWith(DeviceTaskCtrl* taskCtrl)
    {
        if (schParallelDevTaskCtx.Empty()) {
            return true;
        }

        SchDeviceTaskContext* frontCtx = schParallelDevTaskCtx.FrontElement();
        if (frontCtx->GetDeviceTaskCtrl()->ParallelWsId() == taskCtrl->ParallelWsId()) {
            return false; // workspace conflict
        }

        if (frontCtx->GetDeviceTaskCtrl()->ParallelForId() != taskCtrl->ParallelForId()) {
            return false;
        }
        return true;
    }

    bool EnqueueParallelCtx(DeviceTaskCtrl* taskCtrl) { return schParallelDevTaskCtx.EnqueueSchDeviceTask(taskCtrl); }

    bool DevTaskEmpty() { return schParallelDevTaskCtx.Empty(); }

    SchDeviceTaskContext* FrontDevTaskCtx() { return schParallelDevTaskCtx.FrontElement(); }

    uint32_t DeviceTaskCtxNum() { return schParallelDevTaskCtx.Num(); }
};

struct SchThreadStatus {
    std::atomic<bool> isAicpuIdle[AICORE_TYPE_NUM][MAX_SCHEDULE_AICPU_NUM];

    void Init()
    {
        for (size_t i = 0; i < AICORE_TYPE_NUM; ++i) {
            for (size_t j = 0; j < MAX_SCHEDULE_AICPU_NUM; ++j) {
                isAicpuIdle[i][j].store(true);
            }
        }
    }
};

} // namespace npu::tile_fwk::dynamic
