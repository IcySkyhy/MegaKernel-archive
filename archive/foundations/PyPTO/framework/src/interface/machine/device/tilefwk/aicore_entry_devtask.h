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
 * \file aicore_entry_devtask.h
 * \brief DevTask management helpers for aicore_entry.h.
 */

#pragma once

INLINE void UpdateCacheDevTask(ExecuteContext* ctx, uint32_t parallelIdx, int64_t devTaskPtr)
{
    __gm__ DynFuncHeader* header = (__gm__ DynFuncHeader*)devTaskPtr;
#if ENABLE_AICORE_RESOLVE
    dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);
#endif
    ctx->cachedDevTasks[parallelIdx].header = header;
    ctx->cachedDevTasks[parallelIdx].seqNo = header->seqNo;
    ctx->cachedDevTasks[parallelIdx].funcDataList = (__gm__ DynFuncData*)(header + 1);
    ctx->cachedDevTasks[parallelIdx].cceBinary = (__gm__ npu::tile_fwk::DynFuncBin*)(header->cceBinary);
}

INLINE volatile __gm__ ParallelDevTask* GetCoreFunctionData(ExecuteContext* ctx)
{
    __gm__ KernelArgs* args = ctx->args;
    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    // kernel start , init prallelDevtask
    volatile __gm__ ParallelDevTask* parallelDevTask = &args->parallelDevTask;
    while (true) {
        if (parallelDevTask->rear - parallelDevTask->front == 0) {
            dcci(parallelDevTask, SINGLE_CACHE_LINE, CACHELINE_OUT);
            AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_DEVICE_TASK_WAIT_TIMEOUT,
                                        STAGE_GET_PARALLEL_DEVTASK_TIMEOUT, nullptr);

            // The aicore received the task stop notification before receiving the leaftask.
            if (GetLeafTaskId() == AICORE_TASK_STOP) {
                SetStatus(args, STAGE_CORE_EXIT);
                WaitWaveSignal(args);
                return nullptr;
            }
            continue;
        }

        // make sure all devtask dcci successfully
        for (uint32_t i = parallelDevTask->front; i < parallelDevTask->rear; ++i) {
            uint32_t idx = i % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM;
            int64_t elemPtr = 0;
            do {
                AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_DEVICE_TASK_WAIT_TIMEOUT,
                                            STAGE_GET_PARALLEL_DEVTASK_TIMEOUT, nullptr);

                // The aicore received the task stop notification before receiving the leaftask.
                if (GetLeafTaskId() == AICORE_TASK_STOP) {
                    SetStatus(args, STAGE_CORE_EXIT);
                    WaitWaveSignal(args);
                    return nullptr;
                }

                dcci(&parallelDevTask->ptrElements[idx], SINGLE_CACHE_LINE, CACHELINE_OUT);
                elemPtr = parallelDevTask->ptrElements[idx];
            } while (elemPtr == 0);
            dcci((__gm__ void*)elemPtr, SINGLE_CACHE_LINE, CACHELINE_OUT);
            UpdateCacheDevTask(ctx, idx, elemPtr);
        }
        return parallelDevTask;
    }

    return nullptr;
}

INLINE uint32_t RefreshParallelDevTaskByModifyFlag(ExecuteContext* ctx, uint32_t highRegValue)
{
    __gm__ KernelArgs* args = ctx->args;
    uint32_t curLeafDevTaskId = npu::tile_fwk::DevTaskId(highRegValue);
    uint32_t mask = npu::tile_fwk::ParallelDevTaskModifyFlag(highRegValue);
    int32_t modifyCnt = __builtin_popcount(mask);
    while (mask) {
        int idx = __builtin_ffs(mask) - 1;

        // dcci read new devtask id
        uint32_t newDevTaskId;
        if (modifyCnt == 1) { // un-parallel devtask scene
            newDevTaskId = curLeafDevTaskId;
        } else {
            uint32_t oldDevTaskId = ctx->cachedDevTasks[idx].seqNo;
            AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
            do {
                AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_GM_DCCI_TIMEOUT,
                                            STAGE_UPDATE_PARALLEL_DEVTASK_ID_TIMEOUT, AICORE_TASK_ABNORMAL_STOP);
                dcci(&ctx->parallelDevTask->idElements[idx], SINGLE_CACHE_LINE, CACHELINE_OUT);
                newDevTaskId = ctx->parallelDevTask->idElements[idx];
            } while (newDevTaskId == oldDevTaskId);
        }

        // dcci read new devtask ptr
        int64_t newElemPtr;
        __gm__ DynFuncHeader* oldHeader = ctx->cachedDevTasks[idx].header;
        AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
        do {
            AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_GM_DCCI_TIMEOUT, STAGE_UPDATE_PARALLEL_DEVTASK_TIMEOUT,
                                        AICORE_TASK_ABNORMAL_STOP);
            dcci(&ctx->parallelDevTask->ptrElements[idx], SINGLE_CACHE_LINE, CACHELINE_OUT);
            newElemPtr = ctx->parallelDevTask->ptrElements[idx];
            if (newElemPtr != (int64_t)oldHeader) {
                dcci((__gm__ void*)newElemPtr, SINGLE_CACHE_LINE, CACHELINE_OUT);
                break;
            }

            if (newElemPtr) {
                dcci((__gm__ void*)newElemPtr, SINGLE_CACHE_LINE, CACHELINE_OUT);
            }
        } while (newElemPtr == 0 || (((__gm__ DynFuncHeader*)newElemPtr)->seqNo != newDevTaskId));
        UpdateCacheDevTask(ctx, idx, newElemPtr);
        mask &= (mask - 1);
    }

    return 0;
}

INLINE uint32_t RefreshParallelDevTask(ExecuteContext* ctx, __gm__ Metrics* metric, uint32_t& lastRegHighVal)
{
    uint32_t newRegHighVal = GetRegHighValue(ctx->args, lastRegHighVal);
    if (newRegHighVal == 0) {
        return AICORE_TASK_ABNORMAL_STOP;
    }

    uint32_t ret = RefreshParallelDevTaskByModifyFlag(ctx, newRegHighVal);
    if (ret == AICORE_TASK_ABNORMAL_STOP) {
        return AICORE_TASK_ABNORMAL_STOP;
    }

    lastRegHighVal = newRegHighVal;
    dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);

    // start new devtask
    PerfTraceRecord(ctx->SeqNo(), ctx->aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_DEV_TASK_RCV_MODEL);
    ctx->lastTaskFinishCycle = 0;
    (void)metric;
    return 0;
}
