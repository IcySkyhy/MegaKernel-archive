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
 * \file aicore_entry_dfx.h
 * \brief DFX/perf helpers for aicore_entry.h (include only after ExecuteContext is defined).
 */

#pragma once

enum DFX_STAGE_STATUS {
    STAGE_HANDSHAKE_START = 1,
    STAGE_HANDSHAKE_END = 2,
    STAGE_CORE_EXIT = 3,
    STAGE_GET_NEXT_TASK_STOP = 4,
    STAGE_PRE_EXEC_COREFUNC_KERNEL = 5,
    STAGE_FINISH_EXEC_COREFUNC_KERNEL = 6,
    STAGE_FINISH_PIPE_SYNC = 7,
    STAGE_FINISH_CUR_TASK = 8,
    STAGE_GET_PARALLEL_DEVTASK_TIMEOUT = 9,
    STAGE_GET_NEXT_TASK_TIMEOUT = 10,
    STAGE_WAVE_TIMEOUT = 11,
    STAGE_GET_HIGH_REG_TIMEOUT = 12,
    STAGE_UPDATE_PARALLEL_DEVTASK_TIMEOUT = 13,
    STAGE_UPDATE_PARALLEL_DEVTASK_ID_TIMEOUT = 14,
    STAGE_RUN_LEAFTASK_TIMEOUT = 15,
    STAGE_GET_FUNCDATA_STOP = 16
};

struct AicoreDevTaskMetric {
    bool devTaskMetricEnable{false};
    __gm__ AicoreMetric* devTaskMetric{nullptr};
};

struct ExecuteContext {
    __gm__ KernelArgs* args;
    int32_t blockIdx;
    volatile __gm__ ParallelDevTask* parallelDevTask{nullptr};
    uint32_t curLeafTaskParallelIdx{0};
    uint32_t seqNo{0};
    uint32_t profLevel{0};
    struct CachedDevTask {
        uint32_t seqNo{0};
        __gm__ DynFuncHeader* header{nullptr};
        __gm__ DynFuncData* funcDataList{nullptr};
        __gm__ DynFuncBin* cceBinary{nullptr};
    } cachedDevTasks[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM];
    uint64_t lastTaskFinishCycle{0};
    AicoreDevTaskMetric aicoreDevTaskMetric;
#if ENABLE_AICORE_PRINT
    AicoreLogger logger;
#endif
    uint32_t SeqNo() { return cachedDevTasks[curLeafTaskParallelIdx].seqNo; }
};

INLINE void Barrier()
{
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    __asm__ __volatile__("" ::: "memory");
#else
    __asm__ __volatile__("");
#endif
}

INLINE void SetStatus(__gm__ KernelArgs* args, int64_t val)
{
    if (!IS_AICORE || DEBUG_SWITCH) {
        Barrier();
        args->dfxBuffer[2] = val;
        dcci(args->dfxBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
}

INLINE void SetLastWordStatus(__gm__ KernelArgs* args, int64_t val)
{
    Barrier();
    args->dfxBuffer[3] = val;
    dcci(args->dfxBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE void SetWarningStatus(__gm__ KernelArgs* args, int64_t val)
{
    Barrier();
    args->dfxBuffer[6] = val;
    dcci(args->dfxBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE void PerfTraceRecord(uint32_t devTaskId, __gm__ AicoreMetric* metric, AicorePerfTrace type, uint64_t cycle = 0)
{
    if (metric != nullptr) {
        uint64_t cnt = metric->cnt;
        if (cnt < PERF_TRACE_CORE_MAX * PERF_TRACE_INST_MAX_NUM_EVERY_TYPE) {
            __gm__ AicoreDevTaskPerf* perf = &(metric->aicoreEveryDevTypeTimeStamp[cnt]);
            perf->type = static_cast<uint64_t>(type);
            perf->aicoreDevTimeStamp = (cycle == 0) ? get_sys_cnt() : cycle;
            perf->devTaskIdx = static_cast<uint64_t>(devTaskId);

            metric->cnt = cnt + 1;
        }
    }
}

INLINE void FlushMetricStatistic(__gm__ volatile KernelArgs* args)
{
    __gm__ volatile Metrics* m = (__gm__ volatile Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m == nullptr) {
        return;
    }

    for (uint32_t i = 0; i < m->taskCount; i++) {
        dcci(&m->tasks[i], SINGLE_CACHE_LINE, CACHELINE_OUT);
    }

    m->isMetricStop = 1;
    dcci(m, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);
}

INLINE void AddMetricStatistic(ExecuteContext* ctx, uint32_t seqNo, uint32_t taskId, int32_t subGraphId, int64_t t1)
{
    auto m = (__gm__ Metrics*)(ctx->args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m && m->taskCount < MAX_DFX_TASK_NUM_PER_CORE) {
        m->tasks[m->taskCount].subGraphId = subGraphId;
        m->tasks[m->taskCount].seqNo = seqNo;
        m->tasks[m->taskCount].taskId = taskId & TASKID_FROM_CTRL_TOPO_MASK;
        m->tasks[m->taskCount].execStart = t1;
        ctx->lastTaskFinishCycle = get_sys_cnt();
        m->tasks[m->taskCount].execEnd = ctx->lastTaskFinishCycle;
        m->taskCount++;
    }
}

INLINE void DfxProcWhenCoreExit(ExecuteContext* ctx, __gm__ KernelArgs* args, __gm__ Metrics* metric)
{
    PerfTraceRecord(INVALID_DEV_TASK_ID, ctx->aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_WAIT_EXIT_NOTIFY);
    if (unlikely(args->taskEntry.reserved[0] == PRO_LEVEL2 || args->taskEntry.reserved[0] == PRO_LEVEL1 ||
                 ctx->aicoreDevTaskMetric.devTaskMetricEnable)) {
        metric->turnNum++;
        FlushMetricStatistic(args);
    }
}

INLINE void DfxProcWhenDevTaskStop(ExecuteContext* ctx, __gm__ KernelArgs* args, __gm__ Metrics* metric)
{
    if (ctx->lastTaskFinishCycle > 0) {
        PerfTraceRecord(ctx->SeqNo(), ctx->aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_DEV_TASK_LEAF_TASK_EXEC,
                        ctx->lastTaskFinishCycle);
    }
    (void)metric;
    (void)args;
}
