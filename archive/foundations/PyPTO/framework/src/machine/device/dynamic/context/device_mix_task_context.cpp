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
 * \file device_mix_task_context.cpp
 * \brief
 */

#include "machine/device/dynamic/context/device_task_context.h"

namespace npu::tile_fwk::dynamic {

inline int32_t GetWrapAicoreIdx(uint32_t coreType, int32_t wrapVecId)
{
    if (coreType == static_cast<uint32_t>(CoreType::AIC)) {
        return WRAP_IDX_AIC;
    } else {
        return wrapVecId == 1 ? WRAP_IDX_AIV1 : WRAP_IDX_AIV0;
    }
}

void DeviceTaskContext::ProcessWrapQueue(DynDeviceTask* dyntask, uint32_t wrapId, int funcIndex, size_t opIndex,
                                         WrapInfoQueue* wrapQueue)
{
    DEV_VERBOSE_DEBUG("add task to wrap queue, wrapId = %u, funcIndex = %d, opIndex = %lu", wrapId, funcIndex, opIndex);
    if (wrapQueue == nullptr) {
        DEV_VERBOSE_DEBUG("wrapQueue = nullptr");
        return;
    }

    auto cceBinary = dyntask->cceBinary;
    auto callList = dyntask->dynFuncDataCacheList[funcIndex].calleeList;
    auto wrapLeaf = &cceBinary[callList[opIndex]];
    for (uint32_t idx = wrapQueue->head; idx < wrapQueue->tail; idx++) {
        if (wrapQueue->elem[idx].wrapId == wrapId) {
            uint32_t* tasklist = wrapQueue->elem[idx].tasklist;
            uint32_t wrapAicoreIdx = GetWrapAicoreIdx(wrapLeaf->coreType, wrapLeaf->wrapVecId);
            tasklist[wrapAicoreIdx] = MakeTaskID(funcIndex, opIndex);
            return;
        }
    }

    // add new wrap id to wrapQueue
    WrapInfo* info = &wrapQueue->elem[wrapQueue->tail];
    info->wrapId = wrapId;
    info->mixResourceType = wrapLeaf->mixResourceType;

    uint32_t wrapAicoreIdx = GetWrapAicoreIdx(wrapLeaf->coreType, wrapLeaf->wrapVecId);
    info->tasklist[WRAP_IDX_AIC] = AICORE_TASK_INIT;
    info->tasklist[WRAP_IDX_AIV0] = AICORE_TASK_INIT;
    info->tasklist[WRAP_IDX_AIV1] = AICORE_TASK_INIT;
    info->tasklist[wrapAicoreIdx] = MakeTaskID(funcIndex, opIndex);
    wrapQueue->tail++;
}

WrapInfoQueue* DeviceTaskContext::AllocWrapQueue(DynDeviceTask* dyntask)
{
    uint32_t size = sizeof(WrapInfoQueue) + dyntask->devTask.mixTaskData.wrapIdNum * sizeof(WrapInfo);
    WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size,
                                                  workspace_->SlabAlloc(size, WsAicpuSlabMemType::WRAP_QUEUE));
    WrapInfoQueue* q = qalloc.As<WrapInfoQueue>();
    q->head = 0;
    q->tail = 0;
    q->lock = 0;
    q->capacity = dyntask->devTask.mixTaskData.wrapIdNum;
    q->elem = reinterpret_cast<WrapInfo*>(q + 1);
    return q;
}

bool DeviceTaskContext::IsMixArch(DevAscendProgram* devProg) { return devProg->devArgs.archInfo == ArchInfo::DAV_3510; }

bool DeviceTaskContext::IsMultiDie(DevAscendProgram* devProg)
{
    return devProg->devArgs.archInfo == ArchInfo::DAV_3510;
}
bool DeviceTaskContext::IsNeedWrapProcess(DynDeviceTask* dyntask, DevAscendProgram* devProg)
{
    dyntask->devTask.mixTaskData.wrapIdNum = 0;
    if (!IsMixArch(devProg)) {
        return false;
    }
    for (size_t funcIndex = 0; funcIndex < dyntask->dynFuncDataCacheListSize; ++funcIndex) {
        dyntask->devTask.mixTaskData.wrapIdNum += dyntask->dynFuncDataCacheList[funcIndex].devFunc->wrapIdNum_;
    }
    return dyntask->devTask.mixTaskData.wrapIdNum > 0;
}

void DeviceTaskContext::InitDieReadyQueues(DynDeviceTask* dyntask, DevAscendProgram* devProg)
{
    if (!IsMultiDie(devProg)) {
        return;
    }
    ReadyCoreFunctionQueue* queue[DIE_READY_QUEUE_SIZE * DIE_NUM];
    uint32_t size = sizeof(ReadyCoreFunctionQueue) + dyntask->devTask.coreFunctionCnt * sizeof(taskid_t);
    for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; ++i) {
        WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size,
                                                      workspace_->SlabAlloc(size, WsAicpuSlabMemType::DIE_READY_QUE));
        ReadyCoreFunctionQueue* q = qalloc.As<ReadyCoreFunctionQueue>();
        InitReadyCoreFunctionQueue(q, dyntask->devTask.coreFunctionCnt);
        queue[i] = q;
    }
    for (size_t i = 0; i < DIE_NUM; i++) {
        dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] = PtrToValue(queue[i]);
        dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i] = PtrToValue(queue[DIE_NUM + i]);
    }
}

} // namespace npu::tile_fwk::dynamic
