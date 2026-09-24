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
 * \file device_task_context.h
 * \brief
 */

#pragma once

#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "../device_trace.h"

namespace npu::tile_fwk::dynamic {
struct DeviceTaskContext {
    void InitAllocator(DevAscendProgram* devProg, DeviceWorkspaceAllocator& workspace,
                       npu::tile_fwk::DevStartArgsBase* startArgs);

    DynDeviceTask* BuildDeviceTaskData(DeviceStitchContext& stitchContext, uint32_t taskId, DevAscendProgram* devProg,
                                       bool withoutTail);

    void ReleaseFinishedTasks(int perfEvtReleaseFinishTask, int perfEvtDeallocateTask);

    void AppendFinishTask(DynDeviceTask* dynTask);

    void ShowStats();

    void UpdateReadyTaskNum(uint64_t cnt) { readyTaskNum += cnt; }

private:
    uint64_t stitchedFuncNum{0};
    uint64_t rootFuncNum{0};
    uint64_t leafFuncNum{0};
    uint64_t readyTaskNum{0};
    uint64_t dynFuncDataSize{0};
    uint64_t leafFuncDataSize{0};

private:
    DevAscendProgram* devProg_{nullptr};
    DeviceWorkspaceAllocator* workspace_{nullptr};
    npu::tile_fwk::DevStartArgsBase* startArgs_{nullptr};
    bool enableAicoreResolve_{false};

private:
    int BuildReadyQueue(DynDeviceTask* dyntask, DevAscendProgram* devProg);
    void BuildReadyQueueForFunc(DynDeviceTask* dyntask, size_t funcIndex, bool isNeedWrap, WrapInfoQueue* wrapQueue,
                                int& wrapTaskNum);
    void ProcessAivBatchTasks(ReadyCoreFunctionQueue* aivQueue, size_t totalZeroPredAIVBatchEnd,
                              const predcount_t* dupPredCountList, size_t funcIndex);
    void InitReadyCoreFunctionQueue(ReadyCoreFunctionQueue* q, uint32_t capacity);
    int InitReadyQueues(DynDeviceTask* dyntask, DevAscendProgram* devProg,
                        ReadyCoreFunctionQueue* queue[READY_QUEUE_SIZE]);
    void InitDrcoRootFuncList(DynDeviceTask* dyntask);
    int ProcessZeroPredTask(DynDeviceTask* dyntask, WrapInfoQueue* wrapQueue, bool isNeedWrap);
    void InitDieReadyQueues(DynDeviceTask* dyntask, DevAscendProgram* devProg);
    void DispatchReadyQueueToCores(DynDeviceTask* dyntask, DevAscendProgram* devProg);
    void DispatchDieReadyQueueToCores(DynDeviceTask* dyntask, DevAscendProgram* devProg);
    void UpdateDeviceTaskQueueInfo(DynDeviceTask* dyntask, ReadyCoreFunctionQueue* aicpuQueue,
                                   ReadyCoreFunctionQueue* aivQueue, ReadyCoreFunctionQueue* aicQueue,
                                   WrapInfoQueue* wrapQueue);
    int BuildDynFuncData(DynDeviceTask* dyntask, uint32_t taskId, DevAscendFunctionDupped* stitchedList,
                         uint64_t stitchedSize);
    void BuildDrcoRootFuncData(DynFuncData* dyndata, DevAscendFunctionDupped& stitchedFunc);

    // mix subgraph schedule
    WrapInfoQueue* AllocWrapQueue(DynDeviceTask* dyntask);
    void ProcessWrapQueue(DynDeviceTask* dyntask, uint32_t wrapId, int funcIndex, size_t opIndex,
                          WrapInfoQueue* wrapQueue);
    bool IsMixArch(DevAscendProgram* devProg);
    bool IsMultiDie(DevAscendProgram* devProg);
    bool IsNeedWrapProcess(DynDeviceTask* dyntask, DevAscendProgram* devProg);
    inline void doResolve(DynDeviceTask* dyntask, int coreType, size_t funcIdx, size_t succIdx, predcount_t* predList,
                          bool hasWrap, bool isMultiDie)
    {
        predList[succIdx] -= 1;
        if (predList[succIdx] != 0)
            return;

        if (IsHubType(coreType)) {
            DEV_VERBOSE_DEBUG("doResolve hub func %d succindex %d", static_cast<int>(funcIdx),
                              static_cast<int>(succIdx));
            ResolveEarlyDepends(dyntask, funcIdx, succIdx, hasWrap, isMultiDie);
            return;
        }
        if (hasWrap) {
            int32_t* opWrapList = reinterpret_cast<int32_t*>(dyntask->devTask.mixTaskData.opWrapList[funcIdx]);
            if (opWrapList[succIdx] != -1) {
                ProcessWrapQueue(
                    dyntask, MakeMixWrapID(funcIdx, static_cast<uint32_t>(opWrapList[succIdx])), funcIdx, succIdx,
                    reinterpret_cast<WrapInfoQueue*>(dyntask->devTask.mixTaskData.readyWrapCoreFunctionQue));
                readyTaskNum++;
                return;
            }
        }
        EnqueueReadyTask(dyntask, coreType, funcIdx, succIdx, isMultiDie);
    }

    inline void EnqueueReadyTask(DynDeviceTask* dyntask, int coreType, size_t funcIdx, size_t succIdx, bool isMultiDie)
    {
        if (isMultiDie) {
            auto dieId = GetLoopDieId(dyntask, funcIdx);
            if (dieId >= 0) {
                auto q = reinterpret_cast<ReadyCoreFunctionQueue*>(
                    dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[dieId]);
                if (coreType == static_cast<int>(CoreType::AIV)) {
                    q = reinterpret_cast<ReadyCoreFunctionQueue*>(
                        dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[dieId]);
                }
                q->UnsafeEnqueue(MakeTaskID(funcIdx, succIdx));
                readyTaskNum++;
                return;
            }
        }
        auto q = dyntask->readyQueue[dyntask->GetReadyQueueIndexByCoreType(static_cast<CoreType>(coreType))];
        q->UnsafeEnqueue(MakeTaskID(funcIdx, succIdx));
        readyTaskNum++;
    }

    void ResolveEarlyDepends(DynDeviceTask* dyntask, size_t funcIdx, size_t opIdx, bool hasWrap, bool isMultiDie);

    void ResolveEarlyDepends(DynDeviceTask* dyntask);

public:
    static void DumpReadyQueue(DynDeviceTask* dynTask, const char* prefix);

    static void TraceFirstBatchResolve(DynDeviceTask* dynTask);

    static void DumpDepend(DynDeviceTask* dyntask, DevAscendProgram* devProg, DevStartArgs* startArgs,
                           const char* prefix);

    int BuildDeviceTaskDataAndReadyQueue(DynDeviceTask* dyntask, uint32_t taskId, DevAscendProgram* devProg);

    inline int8_t GetLoopDieId(DynDeviceTask* dyntask, size_t funcIndex)
    {
        DevAscendFunctionDuppedData* duppedData = dyntask->dynFuncDataCacheList[funcIndex].duppedData;
        auto loopDieId = duppedData->loopDieId_;
        if (loopDieId >= static_cast<int8_t>(DIE_NUM) || loopDieId < -1) {
            loopDieId = -1;
        }
        return loopDieId;
    }
};
} // namespace npu::tile_fwk::dynamic
