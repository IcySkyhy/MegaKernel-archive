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
 * \file dev_encode_program_ctrlflow_cache.cpp
 * \brief
 */

#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"

namespace npu::tile_fwk::dynamic {

namespace {
template <typename Func>
bool ForEachIncastOutcastAddr(DynDeviceTaskBase* base, Func func)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
        DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
        DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
        DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
        size_t backupSize = sizeof(uint64_t) * (duppedData->GetIncastSize() + duppedData->GetOutcastSize());

        if (!func(dynData, dynDataBackup, backupSize)) {
            return false;
        }
    }
    return true;
}
} // namespace

void DevControlFlowCache::MatchInputOutputDump(DevStartArgsBase* startArgs) const
{
    DEV_VERBOSE_DEBUG("matchio cache input size: %d", (int)inputTensorDataList.size());
    for (size_t k = 0; k < inputTensorDataList.size(); k++) {
        DEV_VERBOSE_DEBUG("matchio cache input %d: %s", (int)k, DumpShape(inputTensorDataList[k].shape).c_str());
    }

    DEV_VERBOSE_DEBUG("matchio cache output size: %d", (int)outputTensorDataList.size());
    for (size_t k = 0; k < outputTensorDataList.size(); k++) {
        DEV_VERBOSE_DEBUG("matchio cache output %d: %s", (int)k, DumpShape(outputTensorDataList[k].shape).c_str());
    }

    DEV_VERBOSE_DEBUG("matchio real input size: %d", (int)startArgs->inputTensorSize);
    for (size_t k = 0; k < startArgs->inputTensorSize; k++) {
        DEV_VERBOSE_DEBUG("matchio real input %d: %s", (int)k, DumpShape(startArgs->GetInputTensor(k).shape).c_str());
    }

    DEV_VERBOSE_DEBUG("matchio real output size: %d", (int)startArgs->outputTensorSize);
    for (size_t k = 0; k < startArgs->outputTensorSize; k++) {
        DEV_VERBOSE_DEBUG("matchio real output %d: %s", (int)k, DumpShape(startArgs->GetOutputTensor(k).shape).c_str());
    }
}

void DevControlFlowCache::PredCountDataBackup(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
        DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
        DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
        size_t backupSize = sizeof(predcount_t) * duppedData->GetOperationSize();

        predcount_t* predCountBackup = reinterpret_cast<predcount_t*>(AllocateCache(backupSize));
        if (predCountBackup == nullptr) {
            return;
        }
        dynDataBackup->predCountBackup = predCountBackup;
        DevMemcpyS(dynDataBackup->predCountBackup, backupSize, &duppedData->GetOperationCurrPredCount(0), backupSize);

        BitmapDataBackup(dynDataCache, dynDataBackup);
    }
}

void DevControlFlowCache::PredCountDataRestore(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
        DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
        DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
        uint32_t opSize = duppedData->GetOperationSize();
        size_t backupSize = sizeof(predcount_t) * opSize;
        DevMemcpyS(&duppedData->GetOperationCurrPredCount(0), backupSize, dynDataBackup->predCountBackup, backupSize);

        DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
        if (dynData->drcoRootFuncData.predCount != nullptr) {
            // drco predCount is int32_t on aicore side, while cache backup is uint16_t(predcount_t)
            int32_t* drcoPredCount = dynData->drcoRootFuncData.predCount;
            for (uint32_t i = 0; i < opSize; ++i) {
                drcoPredCount[i] = static_cast<int32_t>(dynDataBackup->predCountBackup[i]);
            }
        }

        BitmapDataRestore(dynDataCache, dynDataBackup);
    }
}

void DevControlFlowCache::ReadyQueueDataBackup(DynDeviceTaskBase* base)
{
    ReadyQueueCache* readyQueueBackup = reinterpret_cast<ReadyQueueCache*>(AllocateCache(sizeof(ReadyQueueCache)));
    if (readyQueueBackup == nullptr) {
        return;
    }
    readyQueueBackup->coreFunctionCnt = base->devTask.coreFunctionCnt;
    uint32_t readyTaskNum = 0;
    for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
        size_t backupSize = sizeof(uint32_t) * base->readyQueue[i]->Capacity();
        uint32_t* readyQueueBackupElem = reinterpret_cast<uint32_t*>(AllocateCache(backupSize));
        if (readyQueueBackupElem == nullptr) {
            return;
        }

        new (&readyQueueBackup->queueList[i])
            ReadyCoreFunctionQueueUnsafe(base->readyQueue[i]->Capacity(), readyQueueBackupElem);
        readyQueueBackup->queueList[i] = *base->readyQueue[i];
        readyTaskNum += base->readyQueue[i]->UnsafeSize();
    }
    readyQueueBackup->readyTaskNum = readyTaskNum;

    if (base->drcoRootFuncList != nullptr) {
        for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
            auto* src = base->drcoRootFuncList->perCorePendingQueueArray[i];
            if (src == nullptr) {
                continue;
            }
            size_t backupSize = sizeof(npu::tile_fwk::PerCorePendingQueue) +
                                sizeof(npu::tile_fwk::LeafTaskId) * src->size;
            auto* dst = reinterpret_cast<npu::tile_fwk::PerCorePendingQueue*>(AllocateCache(backupSize));
            if (dst == nullptr) {
                continue;
            }
            memcpy_s(dst, backupSize, src, backupSize);
            readyQueueBackup->perCorePendingQueueList[i] = dst;
        }
    }

    base->readyQueueBackup = readyQueueBackup;
}

void DevControlFlowCache::ReadyQueueDataRestore(DynDeviceTaskBase* base, uint32_t nrValidAic)
{
    ReadyQueueCache* readyQueueBackup = base->readyQueueBackup;
    base->devTask.coreFunctionCnt = readyQueueBackup->coreFunctionCnt;
    for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
        *base->readyQueue[i] = readyQueueBackup->queueList[i];
    }

    // for aicore-resolve
    if (base->drcoRootFuncList != nullptr) {
        for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
            auto* dst = base->drcoRootFuncList->perCorePendingQueueArray[i];
            if (dst == nullptr) {
                continue;
            }
            dst->head = 0;
            dst->tail = 0;
            dst->size = 0;
        }

        uint32_t nrAivCores = nrValidAic * 2;
        ReadyCoreFunctionQueue* aivQueue = base->readyQueue[0];
        ReadyCoreFunctionQueue* aicQueue = base->readyQueue[1];
        uint32_t aicIdx = 0;
        for (const auto* it = aicQueue->begin(); it != aicQueue->end(); ++it) {
            uint32_t coreIdx = aicIdx % nrValidAic;
            base->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
            aicIdx++;
        }
        uint32_t aivIdx = 0;
        for (const auto* it = aivQueue->begin(); it != aivQueue->end(); ++it) {
            uint32_t coreIdx = nrValidAic + (aivIdx % nrAivCores);
            base->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
            aivIdx++;
        }
        __sync_synchronize();
        for (uint32_t ct = 0; ct < npu::tile_fwk::NUM_CORE_TYPES; ct++) {
            for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
                auto* dst = base->drcoRootFuncList->localReadyQueueArray[ct][i];
                if (dst == nullptr) {
                    continue;
                }
                dst->head = 0;
                dst->tail = 0;
            }
        }
        base->drcoRootFuncList->executedTaskCount = 0;
        base->drcoRootFuncList->devTaskFinished = 0;
    }
}

void DevControlFlowCache::DieReadyQueueDataBackup(DynDeviceTaskBase* base)
{
    DieReadyQueueCache* dieReadyQueueBackup = reinterpret_cast<DieReadyQueueCache*>(
        AllocateCache(sizeof(DieReadyQueueCache)));
    if (dieReadyQueueBackup == nullptr) {
        return;
    }
    dieReadyQueueBackup->coreFunctionCnt = base->devTask.coreFunctionCnt;
    uint32_t readyTaskNum = 0;
    for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; i++) {
        ReadyCoreFunctionQueue* dieReadyQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            (i < DIE_NUM) ? base->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] :
                            base->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i - DIE_NUM]);
        if (dieReadyQueue == nullptr) {
            new (&dieReadyQueueBackup->queueList[i]) ReadyCoreFunctionQueueUnsafe(0, nullptr);
            continue;
        }
        size_t backupSize = sizeof(uint32_t) * dieReadyQueue->Capacity();
        uint32_t* dieReadyQueueBackupElem = reinterpret_cast<uint32_t*>(AllocateCache(backupSize));
        if (dieReadyQueueBackupElem == nullptr) {
            return;
        }

        new (&dieReadyQueueBackup->queueList[i])
            ReadyCoreFunctionQueueUnsafe(dieReadyQueue->Capacity(), dieReadyQueueBackupElem);
        dieReadyQueueBackup->queueList[i] = *dieReadyQueue;
        readyTaskNum += dieReadyQueue->UnsafeSize();
    }
    dieReadyQueueBackup->readyTaskNum = readyTaskNum;
    base->dieReadyQueueBackup = dieReadyQueueBackup;
}

void DevControlFlowCache::DieReadyQueueDataRestore(DynDeviceTaskBase* base, uint32_t nrValidAic)
{
    DieReadyQueueCache* dieReadyQueueBackup = base->dieReadyQueueBackup;
    if (dieReadyQueueBackup == nullptr) {
        return;
    }
    base->devTask.coreFunctionCnt = dieReadyQueueBackup->coreFunctionCnt;
    for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; i++) {
        ReadyCoreFunctionQueue* dieReadyQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            (i < DIE_NUM) ? base->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] :
                            base->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i - DIE_NUM]);
        if (dieReadyQueue == nullptr) {
            continue;
        }
        *dieReadyQueue = dieReadyQueueBackup->queueList[i];
    }

    if (base->drcoRootFuncList == nullptr) {
        return;
    }
    // for aicore-resolve
    uint32_t nrAivCores = 2 * nrValidAic;
    uint32_t halfAic = nrValidAic / 2;
    uint32_t halfAiv = nrAivCores / 2;
    // perCorePendingQueue: distribute die tasks, die0 to first-half cores, die1 to second-half cores
    for (uint32_t dieId = 0; dieId < DIE_NUM; ++dieId) {
        uint32_t aicCoreBase = dieId * halfAic;
        uint32_t aicCoreCnt = (dieId == 0) ? halfAic : (nrValidAic - halfAic);
        ReadyCoreFunctionQueue* dieAicQue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            base->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[dieId]);
        if (dieAicQue != nullptr) {
            uint32_t dieAicIdx = 0;
            for (const auto* it = dieAicQue->begin(); it != dieAicQue->end(); ++it) {
                uint32_t coreIdx = aicCoreBase + (dieAicIdx % aicCoreCnt);
                base->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
                dieAicIdx++;
            }
        }

        uint32_t aivCoreBase = nrValidAic + dieId * halfAiv;
        uint32_t aivCoreCnt = (dieId == 0) ? halfAiv : (nrAivCores - halfAiv);
        ReadyCoreFunctionQueue* dieAivQue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            base->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[dieId]);
        if (dieAivQue != nullptr) {
            uint32_t dieAivIdx = 0;
            for (const auto* it = dieAivQue->begin(); it != dieAivQue->end(); ++it) {
                uint32_t coreIdx = aivCoreBase + (dieAivIdx % aivCoreCnt);
                base->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
                dieAivIdx++;
            }
        }
    }
}

void DevControlFlowCache::MixTaskDataBackup(DynDeviceTaskBase* base)
{
    if (base->devTask.mixTaskData.wrapIdNum == 0) {
        return;
    }
    MixTaskDataCache* mixTaskDataBackup = reinterpret_cast<MixTaskDataCache*>(AllocateCache(sizeof(MixTaskDataCache)));
    if (mixTaskDataBackup == nullptr) {
        return;
    }
    mixTaskDataBackup->wrapIdNum = base->devTask.mixTaskData.wrapIdNum;
    WrapInfoQueue* wrapInfoQueue = reinterpret_cast<WrapInfoQueue*>(base->devTask.mixTaskData.readyWrapCoreFunctionQue);
    size_t wrapInfoBackupSize = sizeof(WrapInfo) * wrapInfoQueue->capacity;
    WrapInfo* wrapQueueBackupElem = reinterpret_cast<WrapInfo*>(AllocateCache(wrapInfoBackupSize));
    if (wrapQueueBackupElem == nullptr) {
        return;
    }
    mixTaskDataBackup->queue.head = wrapInfoQueue->head;
    mixTaskDataBackup->queue.tail = wrapInfoQueue->tail;
    mixTaskDataBackup->queue.capacity = wrapInfoQueue->capacity;
    mixTaskDataBackup->queue.elem = wrapQueueBackupElem;
    DevMemcpyS(mixTaskDataBackup->queue.elem, wrapInfoBackupSize, wrapInfoQueue->elem, wrapInfoBackupSize);

    constexpr size_t arrSize = sizeof(uint64_t) * MAX_STITCH_FUNC_NUM;
    DevMemcpyS(mixTaskDataBackup->opWrapList, arrSize, base->devTask.mixTaskData.opWrapList, arrSize);

    base->mixTaskDataBackup = mixTaskDataBackup;
}

void DevControlFlowCache::MixTaskDataRestore(DynDeviceTaskBase* base)
{
    if (base->mixTaskDataBackup == nullptr) {
        return;
    }
    MixTaskDataCache* mixTaskDataBackup = base->mixTaskDataBackup;
    base->devTask.mixTaskData.wrapIdNum = mixTaskDataBackup->wrapIdNum;

    WrapInfoQueue* wrapInfoQueue = reinterpret_cast<WrapInfoQueue*>(base->devTask.mixTaskData.readyWrapCoreFunctionQue);
    wrapInfoQueue->head = mixTaskDataBackup->queue.head;
    wrapInfoQueue->tail = mixTaskDataBackup->queue.tail;
    wrapInfoQueue->capacity = mixTaskDataBackup->queue.capacity;

    size_t wrapInfoBackupSize = sizeof(WrapInfo) * wrapInfoQueue->capacity;
    DevMemcpyS(wrapInfoQueue->elem, wrapInfoBackupSize, mixTaskDataBackup->queue.elem, wrapInfoBackupSize);

    constexpr size_t arrSize = sizeof(uint64_t) * MAX_STITCH_FUNC_NUM;
    DevMemcpyS(base->devTask.mixTaskData.opWrapList, arrSize, mixTaskDataBackup->opWrapList, arrSize);
}

void DevControlFlowCache::RelocBuildInputOutputDesc(
    std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict, DevStartArgsBase* devStartArgs)
{
    for (uint64_t i = 0; i < devStartArgs->inputTensorSize; i++) {
        uint64_t addr = devStartArgs->GetInputTensor(i).address;
        cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_INPUT, i);
    }
    for (uint64_t i = 0; i < devStartArgs->outputTensorSize; i++) {
        uint64_t addr = devStartArgs->GetOutputTensor(i).address;
        cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_OUTPUT, i);
    }
}

void DevControlFlowCache::RelocBuildInputOutputDesc(
    std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict,
    DevRelocVector<DevTensorData> inputTensorDataList, DevRelocVector<DevTensorData> outputTensorDataList)
{
    for (uint64_t i = 0; i < inputTensorDataList.size(); i++) {
        uint64_t addr = inputTensorDataList[i].address;
        cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_INPUT, i);
    }
    for (uint64_t i = 0; i < outputTensorDataList.size(); i++) {
        uint64_t addr = outputTensorDataList[i].address;
        cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_OUTPUT, i);
    }
}

void DevControlFlowCache::RelocDescToCache(AddressDescriptor& desc, const RelocRange& relocWorkspace,
                                           std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict)
{
    AddressDescriptor resultDesc;
    uint64_t addr = desc.GetAddressValue();
    if (cacheInputOutputDict.count(addr)) {
        resultDesc = cacheInputOutputDict[addr];
    } else if (addr & (1UL << 58)) {
        resultDesc = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_COMM, addr);
    } else {
        relocWorkspace.Reloc(addr);
        resultDesc = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_WORKSPACE, addr);
    }
    desc = resultDesc;
}

void DevControlFlowCache::RelocDescFromCache(AddressDescriptor& desc, const RelocRange& relocWorkspace,
                                             DevStartArgsBase* devStartArgs)
{
    uint64_t resultAddr = 0;
    switch (desc.cacheKind) {
        case ADDRESS_CACHE_KIND_WORKSPACE:
            resultAddr = desc.cacheValue;
            relocWorkspace.Reloc(resultAddr);
            break;
        case ADDRESS_CACHE_KIND_INPUT:
            resultAddr = devStartArgs->GetInputTensor(desc.cacheValue).address;
            break;
        case ADDRESS_CACHE_KIND_OUTPUT:
            resultAddr = devStartArgs->GetOutputTensor(desc.cacheValue).address;
            break;
        case ADDRESS_CACHE_KIND_COMM:
            resultAddr = desc.cacheValue;
            break;
        default:
            DEV_ERROR(ProgEncodeErr::CACHE_RELOC_KIND_INVALID,
                      "#ctrl.task.pre.cache.reloc: [RelocDescFromCache] Invalid kind: %lu\n",
                      (unsigned long)desc.cacheKind);
            break;
    }
    AddressDescriptor resultDesc = AddressDescriptor::MakeFromAddress(resultAddr);
    desc = resultDesc;
}

void DevControlFlowCache::IncastOutcastAddrBackup(DynDeviceTaskBase* base)
{
    ForEachIncastOutcastAddr(base, [this](DynFuncData* dynData, DynFuncDataBackup* dynDataBackup, size_t backupSize) {
        uint64_t* rawTensorAddrBackup = reinterpret_cast<uint64_t*>(AllocateCache(backupSize));
        if (rawTensorAddrBackup == nullptr) {
            return false;
        }
        dynDataBackup->rawTensorAddrBackup = rawTensorAddrBackup;
        DevMemcpyS(dynDataBackup->rawTensorAddrBackup, backupSize, dynData->rawTensorAddr, backupSize);
        return true;
    });
}

void DevControlFlowCache::IncastOutcastAddrRestore(DynDeviceTaskBase* base)
{
    ForEachIncastOutcastAddr(base, [](DynFuncData* dynData, DynFuncDataBackup* dynDataBackup, size_t backupSize) {
        DevMemcpyS(dynData->rawTensorAddr, backupSize, dynDataBackup->rawTensorAddrBackup, backupSize);
        return true;
    });
}

void DevControlFlowCache::IncastOutcastAddrRestore()
{
    for (size_t i = 0; i < deviceTaskCount; i++) {
        DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[i].dynTaskBase;
        IncastOutcastAddrRestore(dynTaskBase);
    }
}

void DevControlFlowCache::TaskAddrBackupWorkspace(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
        DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
        DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
        DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;

        dynDataBackup->workspaceAddressBackup.runtimeWorkspace = duppedData->runtimeWorkspace_;
        dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace = duppedData->runtimeOutcastWorkspace_;
        dynDataBackup->workspaceAddressBackup.workspaceAddr = dynData->workspaceAddr;
        dynDataBackup->workspaceAddressBackup.stackWorkspaceAddr = dynFuncDataList->stackWorkSpaceAddr;
    }
}

void DevControlFlowCache::TaskAddrRestoreWorkspace(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
        DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
        DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
        DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;

        duppedData->runtimeWorkspace_ = dynDataBackup->workspaceAddressBackup.runtimeWorkspace;
        duppedData->runtimeOutcastWorkspace_ = dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace;
        dynData->workspaceAddr = dynDataBackup->workspaceAddressBackup.workspaceAddr;
    }
    dynFuncDataList->stackWorkSpaceAddr = dynFuncDataBackupList[0].workspaceAddressBackup.stackWorkspaceAddr;
}

void DevControlFlowCache::TaskAddrRestoreWorkspace()
{
    for (size_t i = 0; i < deviceTaskCount; i++) {
        DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[i].dynTaskBase;
        TaskAddrRestoreWorkspace(dynTaskBase);
    }
}

void DevControlFlowCache::TaskAddrRelocWorkspace(uint64_t srcWorkspace, uint64_t dstWorkspace,
                                                 DevStartArgsBase* devStartArgs)
{
    RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
    for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
        DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[deviceIndex].dynTaskBase;

        DynFuncHeader* dynFuncDataList = dynTaskBase->dynFuncDataList;
        DynFuncDataCache* dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
        for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
            DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;

            if (devStartArgs == nullptr) {
                // Host: addr uses backup

                // Reloc Dupped
                relocWorkspace.RelocNullable(dynDataBackup->workspaceAddressBackup.runtimeWorkspace);
                relocWorkspace.RelocNullable(dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace);

                // Reloc DynFuncData
                relocWorkspace.Reloc(dynDataBackup->workspaceAddressBackup.workspaceAddr);
            } else {
                // Device: addr uses actual

                // Reloc Dupped
                relocWorkspace.RelocNullable(duppedData->runtimeWorkspace_);
                relocWorkspace.RelocNullable(duppedData->runtimeOutcastWorkspace_);

                // Reloc DynFuncData
                relocWorkspace.Reloc(dynData->workspaceAddr);
            }
        }
        // Reloc header-level fields (shared by all funcs)
        if (devStartArgs == nullptr) {
            relocWorkspace.Reloc(dynFuncDataBackupList[0].workspaceAddressBackup.stackWorkspaceAddr);
            dynFuncDataList->stackWorkSpaceAddr = dynFuncDataBackupList[0].workspaceAddressBackup.stackWorkspaceAddr;
        } else {
            relocWorkspace.Reloc(dynFuncDataList->stackWorkSpaceAddr);
        }
    }
}

void DevControlFlowCache::IncastOutcastAddrReloc(uint64_t srcWorkspace, uint64_t dstWorkspace,
                                                 DevStartArgsBase* devStartArgs)
{
    RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
    /* empty constructor's overhead should be negligible */
    std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
    if (devStartArgs == nullptr) {
        /* only run on host */
        RelocBuildInputOutputDesc(cacheInputOutputDict, inputTensorDataList, outputTensorDataList);
    }
    for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
        DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[deviceIndex].dynTaskBase;
        DynFuncHeader* dynFuncDataList = dynTaskBase->dynFuncDataList;
        DynFuncDataCache* dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
        for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

            DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
            if (devStartArgs == nullptr) {
                // Host: addr uses backup
                for (uint64_t i = 0; i < duppedData->GetIncastSize(); i++) {
                    AddressDescriptor* addr = reinterpret_cast<AddressDescriptor*>(dynDataBackup->rawTensorAddrBackup +
                                                                                   i);
                    RelocDescToCache(*addr, relocWorkspace, cacheInputOutputDict);
                }
                for (uint64_t i = 0; i < duppedData->GetOutcastSize(); i++) {
                    AddressDescriptor* addr = reinterpret_cast<AddressDescriptor*>(dynDataBackup->rawTensorAddrBackup +
                                                                                   duppedData->GetIncastSize() + i);
                    RelocDescToCache(*addr, relocWorkspace, cacheInputOutputDict);
                }
            } else {
                // Device: addr uses actual
                for (uint64_t i = 0; i < duppedData->GetIncastSize(); i++) {
                    AddressDescriptor* addr = &duppedData->GetIncastAddress(i);
                    RelocDescFromCache(*addr, relocWorkspace, devStartArgs);
                }
                for (uint64_t i = 0; i < duppedData->GetOutcastSize(); i++) {
                    AddressDescriptor* addr = &duppedData->GetOutcastAddress(i);
                    RelocDescFromCache(*addr, relocWorkspace, devStartArgs);
                }
            }
        }
        dynFuncDataList->startArgs = devStartArgs;
    }
}

void DevControlFlowCache::RuntimeAddrBackup(DeviceExecuteSlot* runtimeSlotList,
                                            ItemPool<RuntimeOutcastTensor>* runtimeOutcastTensorPool, uint64_t slotSize,
                                            uint64_t runtimeOutcastTensorSize, TensorAllocator* allocator,
                                            uint32_t parallelism)
{
    uint64_t slotDataSize = sizeof(DeviceExecuteSlot) * slotSize;
    uint64_t runtimeOutcastPoolDataSize = sizeof(ItemPool<RuntimeOutcastTensor>::ItemBlock) * runtimeOutcastTensorSize;
    DevMemcpyS(runtimeBackup.slotContext.slotList.Data(), slotDataSize, runtimeSlotList, slotDataSize);
    auto itemBlockBase = reinterpret_cast<ItemPool<RuntimeOutcastTensor>::ItemBlock*>(&runtimeOutcastTensorPool->At(0));
    DevMemcpyS(runtimeBackup.workspace.runtimeOutcastTensorPool.Data(), runtimeOutcastPoolDataSize, itemBlockBase,
               runtimeOutcastPoolDataSize);
    runtimeBackup.workspace.itemPoolMeta = runtimeOutcastTensorPool->GetMetaData();
    struct Backup {
        static void BackupBlockHeader(WsSlotAllocator::BlockHeader*& ptr, WsSlotAllocator::BlockHeader* base)
        {
            if (ptr == nullptr) {
                ptr = reinterpret_cast<WsSlotAllocator::BlockHeader*>(WsSlotAllocator::kNullBlockHeaderIndex);
                return;
            }
            ptr = reinterpret_cast<WsSlotAllocator::BlockHeader*>(static_cast<uintptr_t>(ptr - base));
        }
        static void BackupPool(WsSlotAllocator& backupAllocator, WsSlotAllocator& runtimePool,
                               WsSlotAllocator::BlockHeader* checkpointList, uint64_t checkpointOffset)
        {
            backupAllocator = runtimePool;
            uint64_t readyCount = runtimePool.initReadyCount_;
            uint64_t backupBytes = sizeof(WsSlotAllocator::BlockHeader) * readyCount;
            WsSlotAllocator::BlockHeader* runtimeBase = runtimePool.GetBlockHeaderBase();
            if (backupBytes > 0) {
                DevMemcpyS(checkpointList + checkpointOffset, backupBytes, runtimeBase, backupBytes);
            }
            BackupBlockHeader(backupAllocator.freeListHeader_, runtimeBase);
            BackupBlockHeader(backupAllocator.notInUseHeaders_, runtimeBase);
            for (uint64_t k = 0; k < readyCount; k++) {
                BackupBlockHeader(checkpointList[checkpointOffset + k].listNext, runtimeBase);
            }
        }
    };

    for (uint32_t i = 0; i < parallelism; i++) {
        runtimeBackup.workspace.tensorAllocators[i].rootInner = allocator[i].rootInner;
        runtimeBackup.workspace.tensorAllocators[i].devTaskInnerExclusiveOutcasts = allocator[i]
                                                                                        .devTaskInnerExclusiveOutcasts;

        auto& checkpointList = runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList;
        uint64_t boundaryOffset = 0;
        uint64_t innerTemporalOffset = allocator[i].devTaskBoundaryOutcasts.slotNum_;
        Backup::BackupPool(runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts,
                           allocator[i].devTaskBoundaryOutcasts, checkpointList.Data(), boundaryOffset);
        Backup::BackupPool(runtimeBackup.workspace.tensorAllocators[i].devTaskInnerTemporalOutcasts,
                           allocator[i].devTaskInnerTemporalOutcasts, checkpointList.Data(), innerTemporalOffset);
    }
}

void DevControlFlowCache::RuntimeAddrRestore(DeviceExecuteSlot* runtimeSlotList,
                                             ItemPool<RuntimeOutcastTensor>* runtimeOutcastTensorPool,
                                             uint64_t slotSize, uint64_t runtimeOutcastTensorSize,
                                             TensorAllocator* allocator, uint32_t parallelism)
{
    uint64_t slotDataSize = sizeof(DeviceExecuteSlot) * slotSize;
    uint64_t runtimeOutcastPoolDataSize = sizeof(ItemPool<RuntimeOutcastTensor>::ItemBlock) * runtimeOutcastTensorSize;
    DevMemcpyS(runtimeSlotList, slotDataSize, runtimeBackup.slotContext.slotList.Data(), slotDataSize);
    auto itemBlockBase = reinterpret_cast<ItemPool<RuntimeOutcastTensor>::ItemBlock*>(&runtimeOutcastTensorPool->At(0));
    DevMemcpyS(itemBlockBase, runtimeOutcastPoolDataSize, runtimeBackup.workspace.runtimeOutcastTensorPool.Data(),
               runtimeOutcastPoolDataSize);
    runtimeOutcastTensorPool->RestoreMetaData(runtimeBackup.workspace.itemPoolMeta);
    struct Restore {
        static void RestoreBlockHeader(WsSlotAllocator::BlockHeader*& ptr, WsSlotAllocator::BlockHeader* base,
                                       WsSlotAllocator::BlockHeader* index)
        {
            if (index == reinterpret_cast<WsSlotAllocator::BlockHeader*>(WsSlotAllocator::kNullBlockHeaderIndex)) {
                ptr = nullptr;
                return;
            }
            ptr = base + (uintptr_t)index;
        }
        static void RestoreSeqAllocator(SeqWsAllocator& dst, SeqWsAllocator& src)
        {
            dst.allocated_ = src.allocated_;
            dst.resetTimes_ = src.resetTimes_;
        }
        static void RestorePool(WsSlotAllocator& runtimePool, WsSlotAllocator& backupAllocator,
                                WsSlotAllocator::BlockHeader* checkpointList, uint64_t checkpointOffset)
        {
            runtimePool.availableSlots_ = backupAllocator.availableSlots_;
            runtimePool.initReadyCount_ = backupAllocator.initReadyCount_;
            WsSlotAllocator::BlockHeader* runtimeBase = runtimePool.GetBlockHeaderBase();
            RestoreBlockHeader(runtimePool.freeListHeader_, runtimeBase, backupAllocator.freeListHeader_);
            RestoreBlockHeader(runtimePool.notInUseHeaders_, runtimeBase, backupAllocator.notInUseHeaders_);
            for (uint64_t k = 0; k < runtimePool.initReadyCount_; k++) {
                RestoreBlockHeader(runtimeBase[k].listNext, runtimeBase, checkpointList[checkpointOffset + k].listNext);
                runtimeBase[k].ptr = checkpointList[checkpointOffset + k].ptr;
            }
        }
    };

    for (uint32_t i = 0; i < parallelism; i++) {
        Restore::RestoreSeqAllocator(allocator[i].rootInner, runtimeBackup.workspace.tensorAllocators[i].rootInner);
        Restore::RestoreSeqAllocator(allocator[i].devTaskInnerExclusiveOutcasts,
                                     runtimeBackup.workspace.tensorAllocators[i].devTaskInnerExclusiveOutcasts);

        auto& checkpointList = runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList;
        uint64_t innerTemporalOffset = allocator[i].devTaskBoundaryOutcasts.slotNum_;
        Restore::RestorePool(allocator[i].devTaskBoundaryOutcasts,
                             runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts, checkpointList.Data(),
                             0);
        Restore::RestorePool(allocator[i].devTaskInnerTemporalOutcasts,
                             runtimeBackup.workspace.tensorAllocators[i].devTaskInnerTemporalOutcasts,
                             checkpointList.Data(), innerTemporalOffset);
    }
}

void DevControlFlowCache::RuntimeAddrRelocProgram(uint64_t srcProgram, uint64_t dstProgram)
{
    RelocRange relocProgram(srcProgram, dstProgram);
    {
        auto& slotList = runtimeBackup.slotContext.slotList;
        DeviceExecuteSlot* base = slotList.Data();
        uint64_t size = slotList.size();
        for (uint64_t k = 0; k < size; k++) {
            relocProgram.RelocNullable(base[k].partialUpdate);
        }
    }
}

void DevControlFlowCache::RuntimeAddrRelocWorkspace(uint64_t srcWorkspace, uint64_t dstWorkspace,
                                                    DevStartArgsBase* devStartArgs, DeviceExecuteSlot* runtimeSlotList,
                                                    ItemPool<RuntimeOutcastTensor>::ItemBlock* runtimeOutcastTensorPool,
                                                    uint32_t parallelism, TensorAllocator* allocator)
{
    RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
    /* empty constructor's overhead should be negligible */
    std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
    if (devStartArgs == nullptr) {
        RelocBuildInputOutputDesc(cacheInputOutputDict, inputTensorDataList, outputTensorDataList);
    }
    {
        for (uint32_t i = 0; i < parallelism; i++) {
            if (devStartArgs == nullptr) {
                // Host: reloc backup only (absolute addr -> workspace offset for CF cache storage).
                auto& slottedOutcastsBlockList = runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList;
                WsSlotAllocator::BlockHeader* base = slottedOutcastsBlockList.Data();
                uint64_t boundaryReady = runtimeBackup.workspace.tensorAllocators[i]
                                             .devTaskBoundaryOutcasts.initReadyCount_;
                uint64_t innerReady = runtimeBackup.workspace.tensorAllocators[i]
                                          .devTaskInnerTemporalOutcasts.initReadyCount_;
                uint64_t innerOffset = runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts.slotNum_;
                for (uint64_t k = 0; k < boundaryReady; k++) {
                    relocWorkspace.RelocNullable(base[k].ptr);
                }
                for (uint64_t k = 0; k < innerReady; k++) {
                    relocWorkspace.RelocNullable(base[innerOffset + k].ptr);
                }
            } else {
                // Device: Restore has copied backup offsets into the live freelist; reloc live only
                // so backup stays in offset form for subsequent cache hits.
                WsSlotAllocator::BlockHeader* liveBoundaryBase = allocator[i]
                                                                     .devTaskBoundaryOutcasts.GetBlockHeaderBase();
                uint64_t liveBoundaryReady = allocator[i].devTaskBoundaryOutcasts.initReadyCount_;
                for (uint64_t k = 0; k < liveBoundaryReady; k++) {
                    relocWorkspace.RelocNullable(liveBoundaryBase[k].ptr);
                }
                WsSlotAllocator::BlockHeader* liveInnerBase = allocator[i]
                                                                  .devTaskInnerTemporalOutcasts.GetBlockHeaderBase();
                uint64_t liveInnerReady = allocator[i].devTaskInnerTemporalOutcasts.initReadyCount_;
                for (uint64_t k = 0; k < liveInnerReady; k++) {
                    relocWorkspace.RelocNullable(liveInnerBase[k].ptr);
                }
            }
        }
    }
    {
        auto& slotList = runtimeBackup.slotContext.slotList;
        DeviceExecuteSlot* base = slotList.Data();
        ItemPool<RuntimeOutcastTensor>::ItemBlock* backupRtOutcastPool = runtimeBackup.workspace
                                                                             .runtimeOutcastTensorPool.Data();
        uint64_t size = slotList.size();
        for (uint64_t k = 0; k < size; k++) {
            static_assert(sizeof(AddressDescriptor) == sizeof(uintdevptr_t),
                          "Please review the following logics when the condition does not hold anymore.");
            if (devStartArgs == nullptr) {
                // Host: addr uses backup
                if (base[k].rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
                    continue;
                }

                auto& rtOutcast = backupRtOutcastPool[base[k].rtOutcastIter].Item();
                if (rtOutcast.isCache) {
                    continue;
                } // To avoid duplicate reloc
                rtOutcast.isCache = true;

                uintdevptr_t addr = rtOutcast.Addr();
                AddressDescriptor* desc = reinterpret_cast<AddressDescriptor*>(&rtOutcast.Addr());
                *desc = AddressDescriptor::MakeFromAddress(addr);
                RelocDescToCache(*desc, relocWorkspace, cacheInputOutputDict);
            } else {
                // Device: addr uses actual
                if (runtimeSlotList[k].rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
                    continue;
                }

                auto& rtOutcast = runtimeOutcastTensorPool[runtimeSlotList[k].rtOutcastIter].Item();
                if (!rtOutcast.isCache) {
                    continue;
                } // To avoid duplicate reloc
                rtOutcast.isCache = false;

                AddressDescriptor* desc = reinterpret_cast<AddressDescriptor*>(&rtOutcast.allocation.ptr);
                RelocDescFromCache(*desc, relocWorkspace, devStartArgs);
                rtOutcast.Addr() = desc->GetAddressValue();
            }
        }
    }
}

void DevControlFlowCache::MixTaskDataReloc(RelocRange& relocCtrlCache, RelocRange& relocProgram,
                                           DynDeviceTaskBase* dynTaskBase, DynFuncHeader* dynFuncDataList)
{
    if (dynTaskBase->devTask.mixTaskData.wrapIdNum == 0) {
        return;
    }

    WrapInfoQueue* tmpWrapInfoQueue = reinterpret_cast<WrapInfoQueue*>(
        dynTaskBase->devTask.mixTaskData.readyWrapCoreFunctionQue);
    WrapInfoQueue*& wrapInfoQueueRef = tmpWrapInfoQueue;
    WrapInfoQueue* wrapInfoQueue = RelocControlFlowCachePointer(wrapInfoQueueRef, relocCtrlCache);
    relocCtrlCache.Reloc(dynTaskBase->devTask.mixTaskData.readyWrapCoreFunctionQue);

    WrapInfo*& wrapInfoElemRef = wrapInfoQueue->elem;
    WrapInfo* wrapInfoElem = RelocControlFlowCachePointer(wrapInfoElemRef, relocCtrlCache);
    (void)wrapInfoElem;

    MixTaskDataCache*& mixTaskDataBackupRef = dynTaskBase->mixTaskDataBackup;
    MixTaskDataCache* mixTaskDataBackup = RelocControlFlowCachePointer(mixTaskDataBackupRef, relocCtrlCache);
    WrapInfo*& wrapInfoBackupElemRef = mixTaskDataBackup->queue.elem;
    WrapInfo* wrapInfoBackupElem = RelocControlFlowCachePointer(wrapInfoBackupElemRef, relocCtrlCache);
    (void)wrapInfoBackupElem;

    for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
        relocProgram.Reloc(dynTaskBase->devTask.mixTaskData.opWrapList[dupIndex]);
        relocProgram.Reloc(mixTaskDataBackup->opWrapList[dupIndex]);
    }
}

void DevControlFlowCache::DieReadyQueueReloc(RelocRange& relocCtrlCache, DynDeviceTaskBase* dynTaskBase)
{
    auto processQueue = [&](uint64_t& queuePtrValue) {
        if (queuePtrValue == 0) {
            return;
        }
        ReadyCoreFunctionQueue* queueRef = reinterpret_cast<ReadyCoreFunctionQueue*>(queuePtrValue);
        ReadyCoreFunctionQueue* queue = RelocControlFlowCachePointer(queueRef, relocCtrlCache);
        relocCtrlCache.Reloc(queuePtrValue);
        if (queue != nullptr) {
            queue->Reloc(relocCtrlCache);
        }
    };
    for (size_t i = 0; i < DIE_NUM; i++) {
        processQueue(dynTaskBase->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
        processQueue(dynTaskBase->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);
    }

    DieReadyQueueCache*& dieReadyQueueBackupRef = dynTaskBase->dieReadyQueueBackup;
    DieReadyQueueCache* dieReadyQueueBackup = RelocControlFlowCachePointer(dieReadyQueueBackupRef, relocCtrlCache);
    if (dieReadyQueueBackup != nullptr) {
        for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; i++) {
            dieReadyQueueBackup->queueList[i].Reloc(relocCtrlCache);
        }
    }
}

void DevControlFlowCache::RelocDuppedDataAndDynFuncData(RelocRange& relocProgram, RelocRange& relocCtrlCache,
                                                        DevAscendFunctionDuppedData* duppedData, DynFuncData* dynData,
                                                        DynFuncDataCache* dynDataCache,
                                                        DynFuncDataBackup* dynDataBackup)
{
    // Reloc Dupped
    relocProgram.Reloc(duppedData->source_);

    // Reloc DynFuncData
    relocProgram.Reloc(dynData->opAttrs);
    relocProgram.Reloc(dynData->opAtrrOffsets);
    relocProgram.Reloc(dynData->rawTensorDesc);

    relocCtrlCache.Reloc(dynData->exprTbl);
    relocCtrlCache.Reloc(dynData->rawTensorAddr);

    relocProgram.Reloc(dynDataCache->devFunc);
    relocProgram.Reloc(dynDataCache->calleeList);
    relocProgram.RelocNullable(dynData->cceBinaryIndexList);

    relocCtrlCache.Reloc(dynDataCache->predCount);
    relocCtrlCache.RelocNullable(dynDataBackup->predCountBackup);
    relocCtrlCache.RelocNullable(dynDataBackup->rawTensorAddrBackup);
    relocCtrlCache.RelocNullable(dynDataBackup->deadEndHubBitmapBackup);
    relocCtrlCache.RelocNullable(dynDataBackup->tailTaskBitmapBackup);

    if (dynData->drcoRootFuncData.predCount != nullptr) {
        relocCtrlCache.Reloc(dynData->drcoRootFuncData.predCount);
        relocProgram.Reloc(dynData->drcoRootFuncData.succStaticList);
        relocCtrlCache.Reloc(dynData->drcoRootFuncData.succStitchList);
        relocProgram.Reloc(dynData->drcoRootFuncData.succInfoList);
    }
}

/* Host-to-cache: devStartArgs should be nullptr. Cache-to-Device: devStartArgs should be filled */
void DevControlFlowCache::TaskAddrRelocProgramAndCtrlCache(uint64_t srcProgram, uint64_t srcCtrlCache,
                                                           uint64_t dstProgram, uint64_t dstCtrlCache)
{
    RelocRange relocCtrlCache(srcCtrlCache, dstCtrlCache);
    RelocRange relocProgram(srcProgram, dstProgram);
    for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
        /* When cached, the pointer is always legal */
        DynDeviceTaskBase*& dynTaskBaseRef = deviceTaskCacheList[deviceIndex].dynTaskBase;
        DynDeviceTaskBase* dynTaskBase = RelocControlFlowCachePointer(dynTaskBaseRef, relocCtrlCache);
        relocCtrlCache.Reloc(dynTaskBase->devTask.readyAivCoreFunctionQue);
        relocCtrlCache.Reloc(dynTaskBase->devTask.readyAicCoreFunctionQue);
        relocCtrlCache.Reloc(dynTaskBase->devTask.readyAicpuFunctionQue);

        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            ReadyCoreFunctionQueue*& readyQueueRef = dynTaskBase->readyQueue[i];
            ReadyCoreFunctionQueue* readyQueue = RelocControlFlowCachePointer(readyQueueRef, relocCtrlCache);
            readyQueue->Reloc(relocCtrlCache);
        }
        relocProgram.Reloc(dynTaskBase->cceBinary);
        relocProgram.Reloc(dynTaskBase->aicpuLeafBinary);

        ReadyQueueCache*& readyQueueBackupRef = dynTaskBase->readyQueueBackup;
        ReadyQueueCache* readyQueueBackup = RelocControlFlowCachePointer(readyQueueBackupRef, relocCtrlCache);
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            readyQueueBackup->queueList[i].Reloc(relocCtrlCache);
        }

        for (size_t i = 0; i < npu::tile_fwk::DRCO_QUEUE_MAX; i++) {
            if (readyQueueBackup->globalReadyQueueList[i].ptr != nullptr) {
                RelocControlFlowCachePointer(readyQueueBackup->globalReadyQueueList[i].ptr, relocCtrlCache);
            }
        }
        for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
            if (readyQueueBackup->perCorePendingQueueList[i] != nullptr) {
                RelocControlFlowCachePointer(readyQueueBackup->perCorePendingQueueList[i], relocCtrlCache);
            }
        }

        DynFuncHeader*& dynFuncDataListRef = dynTaskBase->dynFuncDataList;
        DynFuncHeader* dynFuncDataList = RelocControlFlowCachePointer(dynFuncDataListRef, relocCtrlCache);
        relocProgram.Reloc(dynFuncDataList->cceBinary);

        RelocDrcoRootFuncList(relocCtrlCache, dynTaskBase);

        DynFuncDataCache* dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
        MixTaskDataReloc(relocCtrlCache, relocProgram, dynTaskBase, dynFuncDataList);
        DieReadyQueueReloc(relocCtrlCache, dynTaskBase);
        void*& shmemWaitUntilCacheBackupRef = dynTaskBase->shmemWaitUntilCacheBackup;
        RelocControlFlowCachePointer(shmemWaitUntilCacheBackupRef, relocCtrlCache);
        for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
            DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

            DevAscendFunctionDuppedData*& duppedDataRef = dynDataCache->duppedData;
            DevAscendFunctionDuppedData* duppedData = RelocControlFlowCachePointer(duppedDataRef, relocCtrlCache);

            // Reloc Stitch
            for (uint32_t i = 0; i < duppedData->GetStitchSize(); i++) {
                DevAscendFunctionDuppedStitchList& stitchList = duppedData->GetStitch(i);
                DevAscendFunctionDuppedStitch*& stitchRef = stitchList.Head();
                for (DevAscendFunctionDuppedStitch** nodePtr = &stitchRef; *nodePtr != nullptr;) {
                    DevAscendFunctionDuppedStitch* node = RelocControlFlowCachePointer(*nodePtr, relocCtrlCache);
                    nodePtr = &node->Next();
                }
            }
            RelocDuppedDataAndDynFuncData(relocProgram, relocCtrlCache, duppedData, dynData, dynDataCache,
                                          dynDataBackup);
        }
    }
}

void DevControlFlowCache::RelocDrcoRootFuncList(RelocRange& relocCtrlCache, DynDeviceTaskBase* dynTaskBase)
{
    npu::tile_fwk::DrcoRootFuncList*& drcoRootFuncListRef = dynTaskBase->drcoRootFuncList;
    npu::tile_fwk::DrcoRootFuncList* drcoRootFuncList = nullptr;
    if (drcoRootFuncListRef != nullptr) {
        drcoRootFuncList = RelocControlFlowCachePointer(drcoRootFuncListRef, relocCtrlCache);
    }
    if (drcoRootFuncList != nullptr) {
        for (size_t i = 0; i < npu::tile_fwk::DRCO_QUEUE_MAX; i++) {
            if (drcoRootFuncList->globalReadyQueueList[i].ptr != nullptr) {
                RelocControlFlowCachePointer(drcoRootFuncList->globalReadyQueueList[i].ptr, relocCtrlCache);
            }
        }
        for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
            relocCtrlCache.Reloc(drcoRootFuncList->perCorePendingQueueArray[i]);
        }
        for (uint32_t ct = 0; ct < npu::tile_fwk::NUM_CORE_TYPES; ct++) {
            for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
                relocCtrlCache.Reloc(drcoRootFuncList->localReadyQueueArray[ct][i]);
            }
        }
    }
}

void DevControlFlowCache::RelocMetaCache(uint64_t srcCache, uint64_t dstCache)
{
    intptr_t shift = static_cast<int64_t>(dstCache) - static_cast<int64_t>(srcCache);
    void* offset = data;
    RelocOffset(shift, offset, inputTensorDataList);
    RelocOffset(shift, offset, outputTensorDataList);
    for (uint32_t i = 0; i < SCH_DEVTASK_MAX_PARALLELISM; i++) {
        RelocOffset(shift, offset, runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList);
    }
    RelocOffset(shift, offset, runtimeBackup.slotContext.slotList);
    RelocOffset(shift, offset, runtimeBackup.workspace.runtimeOutcastTensorPool);
    RelocOffset(shift, offset, deviceTaskCacheList);
    RelocOffset(shift, offset, cacheData);
}

} // namespace npu::tile_fwk::dynamic
