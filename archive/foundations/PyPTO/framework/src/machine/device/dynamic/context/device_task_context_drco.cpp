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
 * \file device_task_context_drco.cpp
 * \brief AICore dependency resolve (DRCO) helpers of DeviceTaskContext, gated by enableAicoreResolve.
 */

#include "machine/device/dynamic/context/device_task_context.h"

namespace npu::tile_fwk::dynamic {

void DeviceTaskContext::InitDrcoRootFuncList(DynDeviceTask* dyntask)
{
    auto* rootFuncList = reinterpret_cast<npu::tile_fwk::DrcoRootFuncList*>(
        ControlFlowAllocateSlab(
            devProg_, sizeof(npu::tile_fwk::DrcoRootFuncList),
            workspace_->SlabAlloc(sizeof(npu::tile_fwk::DrcoRootFuncList), WsAicpuSlabMemType::DYN_FUNC_DATA))
            .ptr);
    dyntask->drcoRootFuncList = rootFuncList;
    uint32_t queueCapacity = dyntask->devTask.coreFunctionCnt;
    uint32_t globalReadyQueueSize = sizeof(npu::tile_fwk::DrcoGlobalReadyQueue) +
                                    queueCapacity * sizeof(npu::tile_fwk::LeafTaskId);
    for (size_t i = 0; i < npu::tile_fwk::DRCO_QUEUE_MAX; i++) {
        auto* q = workspace_->AllocateDrcoGlobalReadyQueue(globalReadyQueueSize);
        new (q) npu::tile_fwk::DrcoGlobalReadyQueue();
        rootFuncList->globalReadyQueueList[i].ptr = q;
    }
    uint32_t perCoreSize = sizeof(npu::tile_fwk::PerCorePendingQueue) +
                           queueCapacity * sizeof(npu::tile_fwk::LeafTaskId);
    for (uint32_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
        auto* perCoreQueue = workspace_->AllocatePerCorePendingQueue(perCoreSize);
        new (perCoreQueue) npu::tile_fwk::PerCorePendingQueue();
        rootFuncList->perCorePendingQueueArray[i] = perCoreQueue;
    }
    uint32_t localSize = sizeof(npu::tile_fwk::DrcoLocalReadyQueue);
    localSize += queueCapacity * sizeof(npu::tile_fwk::LeafTaskId);
    for (uint32_t ct = 0; ct < npu::tile_fwk::NUM_CORE_TYPES; ct++) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
            auto* localQueue = workspace_->AllocateDrcoLocalReadyQueue(localSize);
            new (localQueue) npu::tile_fwk::DrcoLocalReadyQueue(queueCapacity);
            rootFuncList->localReadyQueueArray[ct][i] = localQueue;
        }
    }
    rootFuncList->totalTaskCount = dyntask->devTask.coreFunctionCnt;
    rootFuncList->executedTaskCount = 0;
    rootFuncList->devTaskFinished = 0;
}

void DeviceTaskContext::DispatchReadyQueueToCores(DynDeviceTask* dyntask, DevAscendProgram* devProg)
{
    ReadyCoreFunctionQueue* aivQueue = dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV)];
    ReadyCoreFunctionQueue* aicQueue = dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC)];

    uint32_t nrValidAic = devProg->devArgs.nrValidAic;
    uint32_t nrAivCores = 2 * nrValidAic;

    // perCorePendingQueue: distribute aic tasks to aicore cores (0..nrValidAic-1)
    uint32_t aicIdx = 0;
    for (const auto* it = aicQueue->begin(); it != aicQueue->end(); ++it) {
        uint32_t coreIdx = aicIdx % nrValidAic;
        dyntask->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
        aicIdx++;
    }

    // perCorePendingQueue: distribute aiv tasks to aiv cores (nrValidAic..3*nrValidAic-1)
    uint32_t aivIdx = 0;
    for (const auto* it = aivQueue->begin(); it != aivQueue->end(); ++it) {
        uint32_t coreIdx = nrValidAic + (aivIdx % nrAivCores);
        dyntask->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
        aivIdx++;
    }
}

void DeviceTaskContext::DispatchDieReadyQueueToCores(DynDeviceTask* dyntask, DevAscendProgram* devProg)
{
    if (!IsMultiDie(devProg)) {
        return;
    }
    uint32_t nrValidAic = devProg->devArgs.nrValidAic;
    uint32_t nrAivCores = 2 * nrValidAic;
    uint32_t halfAic = nrValidAic / 2;
    uint32_t halfAiv = nrAivCores / 2;
    // perCorePendingQueue: distribute die tasks, die0 to first-half cores, die1 to second-half cores
    for (uint32_t dieId = 0; dieId < DIE_NUM; ++dieId) {
        uint32_t aicCoreBase = dieId * halfAic;
        uint32_t aicCoreCnt = (dieId == 0) ? halfAic : (nrValidAic - halfAic);
        ReadyCoreFunctionQueue* dieAicQue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[dieId]);
        uint32_t dieAicIdx = 0;
        for (const auto* it = dieAicQue->begin(); it != dieAicQue->end(); ++it) {
            uint32_t coreIdx = aicCoreBase + (dieAicIdx % aicCoreCnt);
            dyntask->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
            dieAicIdx++;
        }

        uint32_t aivCoreBase = nrValidAic + dieId * halfAiv;
        uint32_t aivCoreCnt = (dieId == 0) ? halfAiv : (nrAivCores - halfAiv);
        ReadyCoreFunctionQueue* dieAivQue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[dieId]);
        uint32_t dieAivIdx = 0;
        for (const auto* it = dieAivQue->begin(); it != dieAivQue->end(); ++it) {
            uint32_t coreIdx = aivCoreBase + (dieAivIdx % aivCoreCnt);
            dyntask->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
            dieAivIdx++;
        }
    }
}

void DeviceTaskContext::BuildDrcoRootFuncData(DynFuncData* dyndata, DevAscendFunctionDupped& stitchedFunc)
{
    DevAscendFunction* source = stitchedFunc.GetSource();
    uint32_t opSize = stitchedFunc.GetOperationSize();
    // AICore drco uses int32_t predCount because aicore atomicAdd only supports 32bit/64bit operands,
    // while the aicpu-side dupped data predCount stays uint16_t (sizeof(predcount_t)).
    uint32_t drcoPredCountSize = opSize * sizeof(int32_t);
    WsAllocation predAlloc = ControlFlowAllocateSlab(
        devProg_, drcoPredCountSize, workspace_->SlabAlloc(drcoPredCountSize, WsAicpuSlabMemType::PRED_COUNT));
    int32_t* aicorePredCount = predAlloc.As<int32_t>();
    for (uint32_t i = 0; i < opSize; ++i) {
        aicorePredCount[i] = static_cast<int32_t>(stitchedFunc.GetOperationCurrPredCount(i));
    }
    auto* rootFuncData = &dyndata->drcoRootFuncData;
    rootFuncData->predCount = aicorePredCount;
    rootFuncData->succStaticList = reinterpret_cast<int32_t*>(&source->GetOperationSucc(0));
    rootFuncData->succStitchList = &stitchedFunc.DupDataForDynFuncData()->GetStitch(0).Head();
    rootFuncData->succInfoList = &source->GetOperationSuccInfo(0);
    dyndata->cceBinaryIndexList = source->GetCalleeIndexAddr();
}

} // namespace npu::tile_fwk::dynamic
