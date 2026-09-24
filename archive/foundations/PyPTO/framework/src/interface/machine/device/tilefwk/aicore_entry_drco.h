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
 * \file aicore_entry_drco.h
 * \brief DRCO (Dependency Resolving by aiCOre) scheduling flow for aicore_entry.h.
 */

#ifndef AICORE_ENTRY_DRCO_H
#define AICORE_ENTRY_DRCO_H

// device switch head file begin
namespace npu::tile_fwk {

#if defined(__AIV__)
constexpr CoreType drcoCoreType = CoreType::AIV;
#elif defined(__AIC__)
constexpr CoreType drcoCoreType = CoreType::AIC;
#else
constexpr CoreType drcoCoreType = CoreType::MIX;
#endif

} // namespace npu::tile_fwk
// device switch head file end

#define DRCO_DCCI_SINGLE_CACHE_LINE(ptr) dcci((__gm__ uint8_t*)ptr, SINGLE_CACHE_LINE, CACHELINE_OUT)
#define DRCO_DCCI_ENTIRE_DATA_CACHE() dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT)

#define DRCO_BUSY_BACKOFF_CYC 500

#if ENABLE_AICORE_PRINT
#define DRCO_LOGD(ctx, fmt, ...) AICORE_LOGD((ctx)->logger.Context(), fmt, ##__VA_ARGS__)
#else
#define DRCO_LOGD(ctx, fmt, ...) \
    do {                         \
    } while (0)
#endif

INLINE void DrcoBusyBackOff()
{
    uint64_t t0 = get_sys_cnt();
    while (get_sys_cnt() - t0 < DRCO_BUSY_BACKOFF_CYC) {
    }
}

#define DRCO_LEADER_TIMEOUT_CHECK(t0, loopCount, timelen, lastStatus) \
    ++loopCount;                                                      \
    if ((loopCount % 1000 == 0)) {                                    \
        uint64_t elapsed = get_sys_cnt() - t0;                        \
        if (!warningSet && elapsed > AICORE_WARNING_CYCLES) {         \
            SetWarningStatus(entry.args, lastStatus);                 \
            warningSet = true;                                        \
        }                                                             \
        if (elapsed > (timelen)) {                                    \
            SetLastWordStatus(entry.args, lastStatus);                \
            SyncAllMix();                                             \
            Trap();                                                   \
            return nullptr;                                           \
        }                                                             \
    }

constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

__aicore__ inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_FLAG_ID_MAX = 16;

__aicore__ inline void SyncAllMix()
{
#if IS_AICORE
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG + SYNC_FLAG_ID_MAX);
    ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0, SYNC_AIC_FLAG));
    wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG + SYNC_FLAG_ID_MAX);
#elif defined(__DAV_VEC__)
    set_intra_block(PIPE_MTE3, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

template <typename T>
INLINE T DrcoGmLoad(__gm__ T* ptr)
{
    DRCO_DCCI_SINGLE_CACHE_LINE(ptr);
    return *ptr;
}

template <typename T>
INLINE void DrcoGmStore(__gm__ T* ptr, T value)
{
    *ptr = value;
    DRCO_DCCI_SINGLE_CACHE_LINE(ptr);
}

template <typename T>
INLINE T DrcoGmLoadArray(__gm__ T* ptr, uint32_t idx)
{
    DRCO_DCCI_SINGLE_CACHE_LINE(&ptr[idx]);
    return ptr[idx];
}

template <typename T>
INLINE void DrcoGmStoreArray(__gm__ T* ptr, uint32_t idx, T value)
{
    ptr[idx] = value;
    DRCO_DCCI_SINGLE_CACHE_LINE(&ptr[idx]);
}

using DrcoDeviceTask = npu::tile_fwk::DrcoDeviceTask;
using DrcoDeviceTaskReadyQueue = npu::tile_fwk::DrcoDeviceTaskReadyQueue;
using DrcoGlobalReadyQueue = npu::tile_fwk::DrcoGlobalReadyQueue;
using DrcoGlobalReadyQueuePtr = npu::tile_fwk::DrcoGlobalReadyQueuePtr;
using DrcoLocalReadyQueue = npu::tile_fwk::DrcoLocalReadyQueue;

INLINE __gm__ DrcoDeviceTask* GetCurrentDeviceTask(__gm__ DrcoDeviceTaskReadyQueue* queue)
{
    uint32_t head = DrcoGmLoad(&queue->head);
    uint32_t size = DrcoGmLoad(&queue->size);
    if (head >= size) {
        return nullptr;
    }
    __gm__ DrcoDeviceTask* elem = &queue->dynFuncDataListList[head];
    if (DrcoGmLoad(&elem->dynFuncDataList) == nullptr) {
        return nullptr;
    }
    return elem;
}

INLINE uint32_t DrcoAtomicLoad(__gm__ uint32_t* ptr) { return static_cast<uint32_t>(atomicAdd(ptr, 0)); }

INLINE uint32_t DrcoAtomicAddToSigned(__gm__ int32_t* ptr, int32_t value)
{
    uint32_t result;
    while (true) {
        uint32_t ptrValue = static_cast<uint32_t>(atomicAdd(ptr, 0));
        result = static_cast<uint32_t>(
            atomicCAS(reinterpret_cast<__gm__ uint32_t*>(ptr), ptrValue, static_cast<uint32_t>(ptrValue + value)));
        if (ptrValue == result) {
            break;
        }
    }
    return result;
}

INLINE uint32_t DrcoAtomicAddTo(__gm__ uint32_t* ptr, uint32_t value)
{
    uint32_t result;
    while (true) {
        uint32_t ptrValue = static_cast<uint32_t>(atomicAdd(ptr, 0));
        result = static_cast<uint32_t>(atomicCAS(ptr, ptrValue, ptrValue + value));
        if (ptrValue == result) {
            break;
        }
    }
    return result;
}

INLINE uint32_t DrcoAtomicCasTo(__gm__ uint32_t* ptr, uint32_t compare, uint32_t value)
{
    return static_cast<uint32_t>(atomicCAS(ptr, compare, value));
}

INLINE uint32_t DrcoAtomicExchTo(__gm__ uint32_t* ptr, uint32_t value)
{
    return static_cast<uint32_t>(atomicExch(ptr, value));
}

INLINE int DrcoLocalReadyQueueTryPushTask(__gm__ DrcoLocalReadyQueue* queue, uint32_t readyTask)
{
    __gm__ uint32_t* tailPtr = &queue->tail;
    uint32_t tail = DrcoAtomicLoad(tailPtr);
    if (tail >= DrcoGmLoad(&queue->size)) {
        return -1;
    }
    uint32_t tailPrev = DrcoAtomicCasTo(&queue->tail, tail, tail + 1);
    if (tailPrev == tail) {
        DrcoAtomicExchTo(&queue->taskList[tailPrev], DRCO_ENCODE_TASK(readyTask));
        return 0;
    }
    return 1;
}

INLINE bool DrcoLocalReadyQueuePushTask(__gm__ DrcoLocalReadyQueue* queue, uint32_t readyTask)
{
    int result = DrcoLocalReadyQueueTryPushTask(queue, readyTask);
    while (result == 1) {
        result = DrcoLocalReadyQueueTryPushTask(queue, readyTask);
    }
    return result == 0;
}

INLINE uint32_t DrcoLocalReadyQueueGetFirstTask(__gm__ DrcoLocalReadyQueue* queue)
{
    uint32_t head = DrcoAtomicLoad(&queue->head);
    uint32_t tail = DrcoAtomicLoad(&queue->tail);
    if (head >= tail) {
        return static_cast<uint32_t>(AICORE_TASK_NO_INCOME);
    }
    uint32_t headPrev = DrcoAtomicCasTo(&queue->head, head, head + 1);
    if (headPrev != head) {
        return static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
    }
    uint32_t taskId = DrcoAtomicLoad(&queue->taskList[headPrev]);
    while (taskId == 0) {
        DrcoBusyBackOff();
        taskId = DrcoAtomicLoad(&queue->taskList[headPrev]);
    }
    return DRCO_DECODE_TASK(taskId);
}

INLINE uint32_t DrcoPerCorePendingQueueGetFirstTask(__gm__ npu::tile_fwk::PerCorePendingQueue* queue)
{
    uint32_t head = DrcoGmLoad(&queue->head);
    uint32_t size = DrcoGmLoad(&queue->size);
    if (head >= size) {
        return static_cast<uint32_t>(AICORE_TASK_NO_INCOME);
    }
    uint32_t taskId = DrcoGmLoadArray(queue->taskList, head);
    DrcoGmStore(&queue->head, head + 1);
    return taskId;
}

INLINE int DrcoGlobalReadyQueueTryPushLastTask(__gm__ DrcoGlobalReadyQueue* queue, uint32_t readyTask)
{
    __gm__ uint32_t* tailPtr = &queue->tail;
    uint32_t tail = DrcoAtomicLoad(tailPtr);
    uint32_t tailPrev = DrcoAtomicCasTo(&queue->tail, tail, tail + 1);
    if (tailPrev == tail) {
        DrcoAtomicExchTo(&queue->taskList[tailPrev], DRCO_ENCODE_TASK(readyTask));
        return 0;
    }
    return 1;
}

INLINE void DrcoGlobalReadyQueuePushLastTask(__gm__ DrcoGlobalReadyQueue* queue, uint32_t readyTask)
{
    int result = DrcoGlobalReadyQueueTryPushLastTask(queue, readyTask);
    while (result == 1) {
        result = DrcoGlobalReadyQueueTryPushLastTask(queue, readyTask);
    }
}

INLINE uint32_t DrcoGlobalReadyQueueTryGetFirstTask(__gm__ DrcoGlobalReadyQueue* queue)
{
    uint32_t head = DrcoAtomicLoad(&queue->head);
    uint32_t tail = DrcoAtomicLoad(&queue->tail);
    if (head < tail) {
        uint32_t headPrev = DrcoAtomicCasTo(&queue->head, head, head + 1);
        if (headPrev == head) {
            uint32_t taskId = DrcoAtomicLoad(&queue->taskList[headPrev]);
            while (taskId == 0) {
                taskId = DrcoAtomicLoad(&queue->taskList[headPrev]);
            }
            return DRCO_DECODE_TASK(taskId);
        }
        return static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
    }
    return static_cast<uint32_t>(AICORE_TASK_NO_INCOME);
}

INLINE uint32_t DrcoGlobalReadyQueueGetFirstTask(__gm__ DrcoGlobalReadyQueue* queue)
{
    uint32_t taskId = static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
    while (taskId == static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT)) {
        taskId = DrcoGlobalReadyQueueTryGetFirstTask(queue);
        if (taskId == static_cast<uint32_t>(AICORE_TASK_ALL_FINISH)) {
            break;
        }
    }
    return taskId;
}

INLINE __gm__ DrcoGlobalReadyQueue* GetDrcoGlobalReadyQueue(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                            uint32_t readyQueueCoreType)
{
    if (readyQueueCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::AIV)) {
        return rootFuncList->globalReadyQueueList[npu::tile_fwk::DRCO_QUEUE_AIV].ptr;
    } else if (readyQueueCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::AIC)) {
        return rootFuncList->globalReadyQueueList[npu::tile_fwk::DRCO_QUEUE_AIC].ptr;
    }
    return nullptr;
}

INLINE __gm__ DrcoLocalReadyQueue* GetDrcoLocalReadyQueue(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                          uint32_t readyQueueCoreType)
{
    if (readyQueueCoreType < npu::tile_fwk::NUM_CORE_TYPES) {
        return rootFuncList->localReadyQueueArray[readyQueueCoreType][0];
    }
    return nullptr;
}

struct DrcoDynFuncDataListPush {
    INLINE static void Push(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t succTaskId,
                            uint32_t succCoreType)
    {
        __gm__ DrcoLocalReadyQueue* localQueue = GetDrcoLocalReadyQueue(rootFuncList, succCoreType);
        if (localQueue != nullptr) {
            DRCO_DCCI_SINGLE_CACHE_LINE(localQueue);
            if (DrcoLocalReadyQueuePushTask(localQueue, succTaskId)) {
                return;
            }
        }
        __gm__ DrcoGlobalReadyQueue* globalQueue = GetDrcoGlobalReadyQueue(rootFuncList, succCoreType);
        if (globalQueue != nullptr) {
            DrcoGlobalReadyQueuePushLastTask(globalQueue, succTaskId);
        }
    }
};

template <typename GlobalReadyQueueHandler>
INLINE void ExecDrcoResolve(ExecuteContext* ctx, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t taskId)
{
    constexpr uint32_t HUB_STACK_SIZE = 64;
    uint32_t hubStack[HUB_STACK_SIZE];
    int32_t hubStackTop = -1;
    hubStack[++hubStackTop] = taskId;

    while (hubStackTop >= 0) {
        uint32_t curTaskId = hubStack[hubStackTop--];
        uint32_t funcIdx = npu::tile_fwk::FuncID(curTaskId);
        auto funcData = &ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx].funcDataList[funcIdx];
        DRCO_DCCI_SINGLE_CACHE_LINE(funcData);
        DRCO_DCCI_SINGLE_CACHE_LINE((__gm__ uint8_t*)funcData + 64);
        DRCO_DCCI_SINGLE_CACHE_LINE((__gm__ uint8_t*)funcData + 128);

        __gm__ npu::tile_fwk::DrcoRootFuncData* rootFuncData = &funcData->drcoRootFuncData;
        __gm__ npu::tile_fwk::DevAscendFunctionOperationSuccInfo* succInfoList = rootFuncData->succInfoList;
        __gm__ int32_t* succStaticList = rootFuncData->succStaticList;
        __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode** succStitchList = rootFuncData->succStitchList;
        __gm__ int32_t* predCount = rootFuncData->predCount;

        uint32_t operationIndex = npu::tile_fwk::TaskID(curTaskId);
        volatile __gm__ npu::tile_fwk::DevAscendFunctionOperationSuccInfo* succInfo = &succInfoList[operationIndex];
        uint16_t staticIndex = succInfo->staticIndex;
        uint16_t staticSize = succInfo->staticSize;
        uint32_t stitchIndex = succInfo->stitchIndex;

        for (uint16_t i = staticIndex; i < staticIndex + staticSize; i++) {
            uint32_t succOpIdx = succStaticList[i];
            int32_t old = DrcoAtomicAddToSigned(&predCount[succOpIdx], -1);
            if (old == 1) {
                uint32_t succTaskId = npu::tile_fwk::MakeTaskID(funcIdx, succOpIdx);
                DRCO_LOGD(ctx, "resolve static cur=%u succ=%u", curTaskId, succTaskId);
                int cceBinaryIndex = funcData->cceBinaryIndexList[succOpIdx];
                uint32_t
                    succCoreType = ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx].cceBinary[cceBinaryIndex].coreType;
                if (succCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::HUB)) {
                    if (hubStackTop + 1 < HUB_STACK_SIZE) {
                        hubStack[++hubStackTop] = succTaskId;
                    }
                } else if (succCoreType != static_cast<uint32_t>(npu::tile_fwk::CoreType::HUB_MIX)) {
                    GlobalReadyQueueHandler::Push(rootFuncList, succTaskId, succCoreType);
                }
            }
        }

        if (stitchIndex != 0) {
            for (__gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node = succStitchList[stitchIndex];
                 node != nullptr; node = node->nodeNext) {
                for (uint32_t i = 0; i < node->nodeSize; i++) {
                    uint32_t succTaskId = node->nodeTaskList[i];
                    uint32_t succFuncId = npu::tile_fwk::FuncID(succTaskId);
                    uint32_t succOpIdx = npu::tile_fwk::TaskID(succTaskId);
                    auto* succFuncData = &ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx].funcDataList[succFuncId];
                    DRCO_DCCI_SINGLE_CACHE_LINE(succFuncData);
                    DRCO_DCCI_SINGLE_CACHE_LINE((__gm__ uint8_t*)succFuncData + 64);
                    DRCO_DCCI_SINGLE_CACHE_LINE((__gm__ uint8_t*)succFuncData + 128);
                    __gm__ npu::tile_fwk::DrcoRootFuncData* succRootFuncData = &succFuncData->drcoRootFuncData;
                    int32_t old = DrcoAtomicAddToSigned(&succRootFuncData->predCount[succOpIdx], -1);
                    if (old == 1) {
                        DRCO_LOGD(ctx, "resolve stitch cur=%u succ=%u", curTaskId, succTaskId);
                        int cceBinaryIndex = ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx]
                                                 .funcDataList[succFuncId]
                                                 .cceBinaryIndexList[succOpIdx];
                        uint32_t succCoreType = ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx]
                                                    .cceBinary[cceBinaryIndex]
                                                    .coreType;
                        if (succCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::HUB)) {
                            if (hubStackTop + 1 < HUB_STACK_SIZE) {
                                hubStack[++hubStackTop] = succTaskId;
                            }
                        } else if (succCoreType != static_cast<uint32_t>(npu::tile_fwk::CoreType::HUB_MIX)) {
                            GlobalReadyQueueHandler::Push(rootFuncList, succTaskId, succCoreType);
                        }
                    }
                }
            }
        }
    }
}

INLINE uint32_t DrcoDynFuncDataListGetFirstTask(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList)
{
    uint64_t t0 = get_sys_cnt();
    while (true) {
        if (get_sys_cnt() - t0 > AICORE_LEAF_TASK_RUN_TIMEOUT) {
            Trap();
        }
        __gm__ DrcoLocalReadyQueue* localQueue = GetDrcoLocalReadyQueue(
            rootFuncList, static_cast<uint32_t>(npu::tile_fwk::drcoCoreType));
        if (localQueue != nullptr) {
            uint32_t taskId = static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
            while (taskId == static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT)) {
                taskId = DrcoLocalReadyQueueGetFirstTask(localQueue);
            }
            if (taskId != static_cast<uint32_t>(AICORE_TASK_NO_INCOME)) {
                return taskId;
            }
        }

        __gm__ DrcoGlobalReadyQueue* globalReadyQueue = GetDrcoGlobalReadyQueue(
            rootFuncList, static_cast<uint32_t>(npu::tile_fwk::drcoCoreType));
        if (globalReadyQueue != nullptr) {
            uint32_t taskId = DrcoGlobalReadyQueueGetFirstTask(globalReadyQueue);
            if (taskId != static_cast<uint32_t>(AICORE_TASK_NO_INCOME) &&
                taskId != static_cast<uint32_t>(AICORE_TASK_ALL_FINISH)) {
                return taskId;
            }
        }

        uint32_t execCnt = DrcoAtomicLoad(&rootFuncList->executedTaskCount);
        uint32_t totCnt = DrcoGmLoad(&rootFuncList->totalTaskCount);
        if (execCnt >= totCnt) {
            return static_cast<uint32_t>(AICORE_TASK_ALL_FINISH);
        }
    }
}

struct DrcoEntryState {
    int32_t blockIdx;
    __gm__ KernelArgs* args;
    __gm__ Metrics* metric;
    ExecuteContext ctx;
    __gm__ npu::tile_fwk::RuntimeDataRingBufferHeadData* runtimeDataRingBufferHeadData;
    __gm__ npu::tile_fwk::DevStartArgsBase* base;
    __gm__ DrcoDeviceTaskReadyQueue* deviceTaskReadyQueue;
    uint8_t lastMixResourceType;
};

INLINE void InitDrcoEntry(DrcoEntryState& entry, int64_t cfgdata)
{
    uint64_t start = get_sys_cnt();
#if defined(__AIV__) && defined(__MIX__)
    entry.blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    entry.blockIdx = get_block_idx();
#endif
    auto devArgs = (DeviceArgs*)cfgdata;
    entry.args = (__gm__ KernelArgs*)(devArgs->sharedBuffer + entry.blockIdx * SHARED_BUFFER_SIZE);
    __gm__ Metrics* metric = (__gm__ Metrics*)(entry.args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    entry.metric = metric;
    entry.ctx.args = entry.args;
    entry.ctx.blockIdx = entry.blockIdx;
    __gm__ DevDfxArgs* devDfxAddr = (__gm__ DevDfxArgs*)devArgs->devDfxArgAddr;
    entry.ctx.aicoreDevTaskMetric.devTaskMetricEnable = devDfxAddr->isOpenPerfTrace != 0;
    entry.ctx.profLevel = devDfxAddr->profLevel;
    uint8_t aicoreLogLevel = static_cast<uint8_t>(AicoreLogLevel::NONE);
#if ENABLE_AICORE_PRINT
    if (devDfxAddr->logLevel >= 0) {
        aicoreLogLevel = static_cast<uint8_t>(devDfxAddr->logLevel);
    }
#endif
    if (entry.ctx.aicoreDevTaskMetric.devTaskMetricEnable && metric->turnNum < MAX_ROUND_NUM) {
        uint64_t round = metric->turnNum;
        entry.ctx.aicoreDevTaskMetric.devTaskMetric = &(metric->aicoreDevTaskInfo[round]);
        PerfTraceRecord(INVALID_DEV_TASK_ID, entry.ctx.aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_BEGIN, start);
    }

    set_mask_norm();
    entry.lastMixResourceType = static_cast<uint8_t>(MixResourceType::MIX_UNKNOWN);

    PerfTraceRecord(INVALID_DEV_TASK_ID, entry.ctx.aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_INIT);

    InitCtx(&entry.ctx, metric, nullptr, aicoreLogLevel);

    entry.runtimeDataRingBufferHeadData = (__gm__ npu::tile_fwk::RuntimeDataRingBufferHeadData*)
                                              devArgs->runtimeDataRingBufferAddr;
    DRCO_DCCI_SINGLE_CACHE_LINE(entry.runtimeDataRingBufferHeadData);
    entry.base = npu::tile_fwk::RuntimeDataRingBufferHeadData::GetRuntimeDataCurrent(
        entry.runtimeDataRingBufferHeadData);
    DRCO_DCCI_SINGLE_CACHE_LINE(entry.base);
    entry.deviceTaskReadyQueue = DrcoGmLoad(&entry.base->drcoDeviceTaskReadyQueue);
}

INLINE __gm__ DrcoDeviceTask* GetDrcoDeviceTask(DrcoEntryState& entry, bool& isFirstTask, uint64_t t0,
                                                uint64_t& loopCount, bool& warningSet)
{
    if (entry.blockIdx == 0) {
        if (!isFirstTask) {
            uint32_t oldHead = DrcoAtomicLoad(&entry.deviceTaskReadyQueue->head);
            DrcoAtomicCasTo(&entry.deviceTaskReadyQueue->head, oldHead, oldHead + 1);
        }
        isFirstTask = false;
        __gm__ DrcoDeviceTask* deviceTask = nullptr;
        while (true) {
            DRCO_LEADER_TIMEOUT_CHECK(t0, loopCount, AICORE_LEAF_TASK_RUN_TIMEOUT, STAGE_RUN_LEAFTASK_TIMEOUT);
            if (entry.deviceTaskReadyQueue == nullptr) {
                DRCO_DCCI_SINGLE_CACHE_LINE(entry.runtimeDataRingBufferHeadData);
                entry.base = npu::tile_fwk::RuntimeDataRingBufferHeadData::GetRuntimeDataCurrent(
                    entry.runtimeDataRingBufferHeadData);
                DRCO_DCCI_SINGLE_CACHE_LINE(entry.base);
                entry.deviceTaskReadyQueue = DrcoGmLoad(&entry.base->drcoDeviceTaskReadyQueue);
                if (entry.deviceTaskReadyQueue == nullptr) {
                    DrcoBusyBackOff();
                    continue;
                }
            }
            deviceTask = GetCurrentDeviceTask(entry.deviceTaskReadyQueue);
            if (deviceTask == nullptr) {
                uint32_t qHead = DrcoAtomicLoad(&entry.deviceTaskReadyQueue->head);
                uint32_t qSize = DrcoGmLoad(&entry.deviceTaskReadyQueue->size);
                if (qHead < qSize) {
                    __gm__ DrcoDeviceTask* elem = &entry.deviceTaskReadyQueue->dynFuncDataListList[qHead];
                    if (DrcoGmLoad(&elem->dynFuncDataList) == nullptr) {
                        break;
                    }
                }
                DrcoBusyBackOff();
                continue;
            }
            break;
        }
        SyncAllMix();
        DRCO_LOGD(&entry.ctx, "leader got task=%p head=%u", deviceTask,
                  DrcoAtomicLoad(&entry.deviceTaskReadyQueue->head));
        return deviceTask;
    }

    SyncAllMix();
    if (entry.deviceTaskReadyQueue == nullptr) {
        DRCO_DCCI_SINGLE_CACHE_LINE(entry.runtimeDataRingBufferHeadData);
        entry.base = npu::tile_fwk::RuntimeDataRingBufferHeadData::GetRuntimeDataCurrent(
            entry.runtimeDataRingBufferHeadData);
        DRCO_DCCI_SINGLE_CACHE_LINE(entry.base);
        entry.deviceTaskReadyQueue = DrcoGmLoad(&entry.base->drcoDeviceTaskReadyQueue);
    }
    __gm__ DrcoDeviceTask* deviceTask = GetCurrentDeviceTask(entry.deviceTaskReadyQueue);
    DRCO_LOGD(&entry.ctx, "follower bi=%d got task=%p", entry.blockIdx, deviceTask);
    return deviceTask;
}

template <typename GlobalReadyQueueHandler>
INLINE void ExecDrcoPerCoreTasks(ExecuteContext* ctx, __gm__ npu::tile_fwk::PerCorePendingQueue* myPerCoreQueue,
                                 __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint8_t& lastMixResourceType,
                                 bool& isFirstTask)
{
    while (DrcoGmLoad(&myPerCoreQueue->size) > DrcoGmLoad(&myPerCoreQueue->head)) {
        uint32_t taskId = DrcoPerCorePendingQueueGetFirstTask(myPerCoreQueue);
        if (taskId == static_cast<uint32_t>(AICORE_TASK_NO_INCOME)) {
            break;
        }
        if (isFirstTask) {
            PerfTraceRecord(ctx->SeqNo(), ctx->aicoreDevTaskMetric.devTaskMetric,
                            PERF_TRACE_CORE_DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK);
            isFirstTask = false;
        }
        DRCO_LOGD(ctx, "pc exec=%u", taskId);
        ExecCoreFunctionKernel(ctx, taskId, lastMixResourceType);
#ifdef __HAS_SUB_FUNC__
        ExecDrcoResolve<GlobalReadyQueueHandler>(ctx, rootFuncList, taskId);
#endif
        DrcoAtomicAddTo(&rootFuncList->executedTaskCount, 1);
    }
}

template <typename GlobalReadyQueueHandler>
INLINE void ExecDrcoReadyQueueTasks(ExecuteContext* ctx, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                    uint8_t& lastMixResourceType)
{
    uint32_t taskId = DrcoDynFuncDataListGetFirstTask(rootFuncList);
    while (taskId != static_cast<uint32_t>(AICORE_TASK_ALL_FINISH)) {
        if ((taskId & AICORE_FIN_MASK) == 0) {
            DRCO_LOGD(ctx, "gq exec=%u", taskId);
            ExecCoreFunctionKernel(ctx, taskId, lastMixResourceType);
#ifdef __HAS_SUB_FUNC__
            ExecDrcoResolve<GlobalReadyQueueHandler>(ctx, rootFuncList, taskId);
#endif
            DrcoAtomicAddTo(&rootFuncList->executedTaskCount, 1);
        }
        taskId = DrcoDynFuncDataListGetFirstTask(rootFuncList);
    }
}

INLINE void KernelEntryDrco(int64_t ffts_addr, int64_t inputs, int64_t outputs, int64_t workspace, int64_t tilingdata,
                            int64_t cfgdata)
{
    UNUSED(ffts_addr);
    UNUSED(inputs);
    UNUSED(outputs);
    UNUSED(workspace);
    UNUSED(tilingdata);

    DrcoEntryState entry = {};
    InitDrcoEntry(entry, cfgdata);

    bool isFirstTask = true;
    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    while (true) {
        __gm__ DrcoDeviceTask* deviceTask = GetDrcoDeviceTask(entry, isFirstTask, t0, loop_count, warningSet);
        if (deviceTask == nullptr) {
            break;
        }

        __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList = deviceTask->drcoRootFuncList;
        UpdateCacheDevTask(&entry.ctx, entry.ctx.curLeafTaskParallelIdx, (int64_t)deviceTask->dynFuncDataList);
        entry.ctx.lastTaskFinishCycle = 0;

        __gm__ npu::tile_fwk::PerCorePendingQueue* myPerCoreQueue = DrcoGmLoad(
            &rootFuncList->perCorePendingQueueArray[entry.blockIdx]);
        DRCO_DCCI_SINGLE_CACHE_LINE(myPerCoreQueue);
        if (DrcoGmLoad(&rootFuncList->totalTaskCount) == 0) {
            uint32_t oldHead = DrcoAtomicLoad(&entry.deviceTaskReadyQueue->head);
            DrcoAtomicCasTo(&entry.deviceTaskReadyQueue->head, oldHead, oldHead + 1);
            continue;
        }

        bool devTaskFirstTask = true;
        ExecDrcoPerCoreTasks<DrcoDynFuncDataListPush>(&entry.ctx, myPerCoreQueue, rootFuncList,
                                                      entry.lastMixResourceType, devTaskFirstTask);
        ExecDrcoReadyQueueTasks<DrcoDynFuncDataListPush>(&entry.ctx, rootFuncList, entry.lastMixResourceType);

        SyncAllMix();
        if (entry.blockIdx == 0) {
            DrcoGmStore(&rootFuncList->devTaskFinished, (uint32_t)1);
        }
        DfxProcWhenDevTaskStop(&entry.ctx, entry.args, entry.metric);
    }
    if (entry.blockIdx == 0) {
        uint64_t finished = DrcoGmLoad(&entry.runtimeDataRingBufferHeadData->indexFinished.value) + 1;
        DrcoGmStore(&entry.runtimeDataRingBufferHeadData->indexFinished.value, finished);
    }

    entry.args->taskEntry.reserved[0] = entry.ctx.profLevel;
    DfxProcWhenCoreExit(&entry.ctx, entry.args, entry.metric);
    return;
}

#endif
