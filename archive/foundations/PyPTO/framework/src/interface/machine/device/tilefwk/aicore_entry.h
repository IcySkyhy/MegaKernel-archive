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
 * \file aicore_entry.h
 * \brief
 */

#ifndef AICORE_ENTRY_H
#define AICORE_ENTRY_H

#include <stdint.h>
#include <cstdint>
#include "tilefwk/aicpu_common.h"
#include "tilefwk/aicore_runtime.h"
#include "tilefwk/aicore_print.h"
#include "tilefwk/core_func_data.h"

// device switch head file begin
namespace npu::tile_fwk {
#define DEBUG_SWITCH 0

} // namespace npu::tile_fwk
// device switch head file end

#define TO_STRING_IMPL(str) #str
#define TO_STRING(str) TO_STRING_IMPL(str)

#ifdef __HAS_SUB_FUNC__
#if defined(__MIX__) && defined(__AIV__)
#include TO_STRING(__HEAD_FILE__)
#else
#include TO_STRING(__HEAD_FILE__)
#endif
#endif

INLINE void SetLastWordStatus(__gm__ KernelArgs* args, int64_t val);
INLINE uint32_t WaitWaveSignal(__gm__ KernelArgs* args);
INLINE void Trap()
{
#if IS_AICORE
    pipe_barrier(PIPE_ALL);
    trap();
#endif
}

// AICore timeout must exceed AICPU dispatch timeout. Warning is set at 80% of AICPU dispatch timeout.
#ifdef __DAV_V310
// A5: 1000MHz, AICPU dispatch 超时 10s
#define AICORE_WARNING_CYCLES (AICPU_DISPATCH_TIMEOUT_A5 * 8ULL / 10ULL)
#define AICORE_GM_DCCI_TIMEOUT 13000000000ULL          // 13s
#define AICORE_DEVICE_TASK_WAIT_TIMEOUT 14000000000ULL // 14s
#define AICORE_WAVEFLAG_WAIT_TIMEOUT 15000000000ULL    // 15s
#define AICORE_GET_LEAF_HIGHREG_TIMEOUT 16000000000ULL // 16s
#define AICORE_LEAF_TASK_WAIT_TIMEOUT 120000000000ULL  // 120s
#define AICORE_LEAF_TASK_RUN_TIMEOUT 120000000000ULL   // 120s
#else
// A2A3: 50MHz, AICPU dispatch 超时 10s
#define AICORE_WARNING_CYCLES (AICPU_DISPATCH_TIMEOUT_A2A3 * 8ULL / 10ULL)
#define AICORE_GM_DCCI_TIMEOUT 650000000ULL          // 13s
#define AICORE_DEVICE_TASK_WAIT_TIMEOUT 700000000ULL // 14s
#define AICORE_WAVEFLAG_WAIT_TIMEOUT 750000000ULL    // 15s
#define AICORE_GET_LEAF_HIGHREG_TIMEOUT 800000000ULL // 16s
#define AICORE_LEAF_TASK_WAIT_TIMEOUT 6000000000ULL  // 120s
#define AICORE_LEAF_TASK_RUN_TIMEOUT 6000000000ULL   // 120s
#endif

#define AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count) \
    uint64_t t0 = get_sys_cnt();                   \
    uint64_t loop_count = 0;                       \
    bool warningSet = false;

#define AICORE_TIMEOUT_CHECK_IMPL(t0, loop_count, timelen, lastStatus, action) \
    ++loop_count;                                                              \
    if ((loop_count % 1000 == 0)) {                                            \
        uint64_t elapsed = get_sys_cnt() - t0;                                 \
        if (!warningSet && elapsed > AICORE_WARNING_CYCLES) {                  \
            SetWarningStatus(args, lastStatus);                                \
            warningSet = true;                                                 \
        }                                                                      \
        if (elapsed > (timelen)) {                                             \
            SetLastWordStatus(args, lastStatus);                               \
            Trap();                                                            \
            action;                                                            \
        }                                                                      \
    }

#define AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, timelen, lastStatus, retval) \
    AICORE_TIMEOUT_CHECK_IMPL(t0, loop_count, timelen, lastStatus, return retval)

#define AICORE_TIMEOUT_CHECK_RETURN_VOID(t0, loop_count, timelen, lastStatus) \
    AICORE_TIMEOUT_CHECK_IMPL(t0, loop_count, timelen, lastStatus, return)

using npu::tile_fwk::CoreFunctionData;
using npu::tile_fwk::DevRawTensorDesc;
using npu::tile_fwk::DynFuncBin;
using npu::tile_fwk::DynFuncData;
using npu::tile_fwk::DynFuncHeader;

#include "tilefwk/aicore_entry_dfx.h"

#if IS_AICORE
INLINE uint64_t GetDataMainBase()
{
    uint64_t coreStatus = 0;
    __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "+l"(coreStatus));
    return coreStatus;
}
#endif

INLINE uint32_t GetLeafTaskId() { return ((GetDataMainBase() & 0xFFFFFFFF) - 1); }

INLINE uint32_t GetNextLeafTask(uint32_t lastTaskIdx, __gm__ KernelArgs* args)
{
    uint32_t nextLowIdx = 0;
    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    do {
        nextLowIdx = GetLeafTaskId();
        AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_LEAF_TASK_WAIT_TIMEOUT, STAGE_GET_NEXT_TASK_TIMEOUT,
                                    AICORE_TASK_ABNORMAL_STOP);
    } while (nextLowIdx == lastTaskIdx);

    return nextLowIdx;
}

INLINE uint32_t GetRegHighValue(__gm__ KernelArgs* args, uint32_t lastHighRegValue)
{
    uint32_t highRegVal = 0;
    uint64_t regValue = 0;
    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    do {
        regValue = GetDataMainBase();
        highRegVal = (uint32_t)(regValue >> REG_HIGH_DTASKID_SHIFT);
        AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_GET_LEAF_HIGHREG_TIMEOUT, STAGE_GET_HIGH_REG_TIMEOUT, 0);
    } while (highRegVal == 0 || lastHighRegValue == highRegVal);
    return highRegVal;
}

constexpr uint8_t EVENT_ID15 = 15;
constexpr uint8_t EVENT_NUMS_PER_AIV = 16;
enum class MixResourceType { MIX_UNKNOWN = 0, MIX_1C1V = 1, MIX_1C2V = 2 };

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
INLINE void PipeSyncPre(uint8_t mixResourceType, uint8_t lastMixResourceType)
{
    // only lastTask and curTask are mix should insert sync
    (void)lastMixResourceType;
    if (mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_UNKNOWN)) {
        return;
    }
#if defined(__AIV__)
    set_intra_block(PIPE_S, EVENT_ID15);  // 插在callop开始，这里pipe不重要，前面流水必然已执行完
    wait_intra_block(PIPE_S, EVENT_ID15); // PIPE_S阻塞后续指令下发，避免与后续的CCE指令EventId冲突
#else
    if (mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C2V)) {
        set_intra_block(PIPE_S, EVENT_ID15);                       // 作用于Vec0
        set_intra_block(PIPE_S, EVENT_ID15 + EVENT_NUMS_PER_AIV);  // 作用与Vec1
        wait_intra_block(PIPE_S, EVENT_ID15);                      // 等待Vec0
        wait_intra_block(PIPE_S, EVENT_ID15 + EVENT_NUMS_PER_AIV); // 等待Vec1
    } else {
        set_intra_block(PIPE_S, EVENT_ID15);  // 作用于Vec0
        wait_intra_block(PIPE_S, EVENT_ID15); // 等待Vec0
    }
#endif
}
#endif

INLINE void PipeSync()
{
#if defined(__AIV__)
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
#else
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
#endif
}

INLINE void HandshakeClient(volatile __gm__ int64_t* shakeBuf)
{
    set_cond(AICORE_TASK_INIT);
    volatile __gm__ int64_t* hello = shakeBuf;
    *hello = (int64_t)get_coreid() << 32 | AICORE_SAY_HELLO;
    Barrier();
    dcci(hello, SINGLE_CACHE_LINE, CACHELINE_OUT);
    Barrier();
}

INLINE void SendRegFinish(uint32_t curTaskIdx) { set_cond(curTaskIdx | AICORE_FIN_MASK); }

INLINE void SendRegAck(uint32_t taskIdx) { set_cond(taskIdx); }

INLINE void PmuTestBegin(uint32_t profLevel)
{
    UNUSED(profLevel);
#if IS_AICORE
    // set_ctrl/get_ctrl are CANN CCE intrinsics; unavailable on host AICore model simulation.
    if (unlikely(profLevel == PRO_LEVEL2)) {
        set_ctrl((uint64_t)get_ctrl() | 0x1);
    }
#endif
}

INLINE void PmuTestEnd(uint32_t profLevel)
{
    UNUSED(profLevel);
#if IS_AICORE
    if (unlikely(profLevel == PRO_LEVEL2)) {
        set_ctrl((uint64_t)get_ctrl() - 1);
    }
#endif
}

INLINE __gm__ TaskStat* InitTaskStat(ExecuteContext* ctx)
{
    __gm__ TaskStat* taskStat = nullptr;
    (void)ctx;
#ifdef __DAV_V310
    if (unlikely(ctx->profLevel >= PRO_LEVEL1)) {
        auto m = (__gm__ Metrics*)(ctx->args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
        taskStat = &m->tasks[m->taskCount];
        taskStat->perfDataBaseAddr = reinterpret_cast<uint64_t>(m);
        if (m->taskCount == 0) {
            taskStat->setEventAddr = PERF_DATA_BASE_SIZE;
            taskStat->waitEventAddr = PERF_DATA_BASE_SIZE + SET_WAIT_EVENT_DATA_SIZE;
        } else {
            __gm__ TaskStat* prevTask = &m->tasks[m->taskCount - 1];
            taskStat->setEventAddr = prevTask->setEventAddr + prevTask->setEventNum * sizeof(uint64_t);
            taskStat->waitEventAddr = prevTask->waitEventAddr + prevTask->waitEventNum * sizeof(uint64_t);
        }
        taskStat->setEventNum = 0;
        taskStat->waitEventNum = 0;
    }
#endif
    return taskStat;
}

#define FuncNum(id) TaskID(id)

#ifdef __HAS_SUB_FUNC__
INLINE void ExecDynCoreFunctionKernel(ExecuteContext* ctx, uint32_t taskId, uint8_t& lastMixResourceType)
{
    uint64_t t1 = get_sys_cnt();
    SetStatus(ctx->args, ((uint64_t)taskId << 32) | STAGE_PRE_EXEC_COREFUNC_KERNEL); // high 32 bits used for taskId
    auto& cachedTask = ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx];
    auto funcData = &cachedTask.funcDataList[npu::tile_fwk::FuncID(taskId)];
#if ENABLE_AICORE_RESOLVE
    dcci((__gm__ uint8_t*)funcData, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci((__gm__ uint8_t*)funcData + 64, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci((__gm__ uint8_t*)funcData + 128, SINGLE_CACHE_LINE, CACHELINE_OUT);
#endif
    auto opAttrs = &funcData->opAttrs[funcData->opAtrrOffsets[npu::tile_fwk::TaskID(taskId)]];
#if ENABLE_AICORE_PRINT
    CoreFuncParam param = {funcData, cachedTask.header, opAttrs, funcData->exprTbl, taskId, ctx->logger.Context()};
#else
    CoreFuncParam param = {funcData, cachedTask.header, opAttrs, funcData->exprTbl, taskId, nullptr};
#endif

#ifdef __ENABLE_MAIN_BLOCK
#define __MAIN_BLOCK 1
#else
#define __MAIN_BLOCK 0
#endif

#ifdef __DAV_V310
    // for mix coretasks, use cube's stackworkspace
    int index = __MAIN_BLOCK ? (opAttrs[0] + 1) / 2 : opAttrs[0];
    uint8_t mixResourceType = cachedTask.cceBinary[index].mixResourceType;
    int64_t blockIndex = (mixResourceType != 0) ? get_block_idx() : ctx->blockIdx;
    int64_t gmStackAddr = cachedTask.header->stackWorkSpaceAddr + blockIndex * cachedTask.header->stackWorkSpaceSize;
#else
    int64_t gmStackAddr = cachedTask.header->stackWorkSpaceAddr + ctx->blockIdx * cachedTask.header->stackWorkSpaceSize;
#endif

    __gm__ TaskStat* taskStat = nullptr;
    taskStat = InitTaskStat(ctx);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    PipeSyncPre(mixResourceType, lastMixResourceType);
    lastMixResourceType = mixResourceType;
#else
    (void)lastMixResourceType;
#endif
    CallSubFuncTask(opAttrs[0] + funcData->exprTbl[0], &param, gmStackAddr,
                    (__gm__ int64_t*)cachedTask.header->startArgs->commContexts, taskStat);
    SetStatus(ctx->args, STAGE_FINISH_EXEC_COREFUNC_KERNEL);
    PipeSync();
    SetStatus(ctx->args, STAGE_FINISH_PIPE_SYNC);
    if (unlikely(ctx->profLevel == PRO_LEVEL2 || ctx->profLevel == PRO_LEVEL1)) {
        AddMetricStatistic(ctx, ctx->SeqNo(), taskId, opAttrs[0], t1);
    }
    if (unlikely(ctx->aicoreDevTaskMetric.devTaskMetricEnable) && ctx->lastTaskFinishCycle == 0) {
        ctx->lastTaskFinishCycle = get_sys_cnt();
    }
}
#endif

INLINE void InitCtx(ExecuteContext* ctx, __gm__ Metrics* metric, volatile __gm__ ParallelDevTask* prallelDevTask,
                    uint8_t logLevel = static_cast<uint8_t>(AicoreLogLevel::NONE))
{
    ctx->curLeafTaskParallelIdx = 0; // default init first devtask
    PerfTraceRecord(ctx->SeqNo(), ctx->aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_DEV_TASK_RCV_MODEL);
    ctx->parallelDevTask = prallelDevTask;
#if ENABLE_AICORE_PRINT
    auto buffer = reinterpret_cast<__gm__ uint8_t*>(ctx->args->dfxBuffer[SHAK_BUF_PRINT_BUFFER_INDEX]);
    if (buffer != 0 && ctx->logger.GetBuffer() != buffer) {
        ctx->logger.Init(buffer, PRINT_BUFFER_SIZE, logLevel);
    }
#endif
    (void)metric;
    (void)logLevel;
    dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    return;
}

INLINE void ExecCoreFunctionKernel(ExecuteContext* ctx, uint32_t curTaskIdx, uint8_t& lastMixResourceType)
{
    UNUSED(ctx);
    UNUSED(curTaskIdx);
    UNUSED(lastMixResourceType);
#ifdef __HAS_SUB_FUNC__
    ExecDynCoreFunctionKernel(ctx, curTaskIdx, lastMixResourceType);
    return;
#endif
}

INLINE uint32_t WaitWaveSignal(__gm__ KernelArgs* args)
{
    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    volatile __gm__ int64_t* waveBuffer = args->waveBufferCpuToCore;
    while (true) {
        dcci(waveBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
        if (*waveBuffer == AICORE_SAY_GOODBYE) {
            return 0;
        }
        AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_WAVEFLAG_WAIT_TIMEOUT, STAGE_WAVE_TIMEOUT,
                                    AICORE_TASK_ABNORMAL_STOP);
    }
}

#include "tilefwk/aicore_entry_devtask.h"

#if IS_AICORE
#if ENABLE_AICORE_RESOLVE
#include "tilefwk/aicore_entry_drco.h"
#endif
#endif

INLINE void KernelEntry(int64_t ffts_addr, int64_t inputs, int64_t outputs, int64_t workspace, int64_t tilingdata,
                        int64_t cfgdata)
{
#if IS_AICORE
#if ENABLE_AICORE_RESOLVE
    KernelEntryDrco(ffts_addr, inputs, outputs, workspace, tilingdata, cfgdata);
    return;
#endif
#endif
    uint64_t start = get_sys_cnt();
    UNUSED(ffts_addr);
    UNUSED(inputs);
    UNUSED(outputs);
    UNUSED(workspace);
    UNUSED(tilingdata);
#if defined(__AIV__) and defined(__MIX__)
    int32_t blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    int32_t blockIdx = get_block_idx();
#endif
    auto devArgs = (DeviceArgs*)cfgdata;
    __gm__ KernelArgs* args = (__gm__ KernelArgs*)(devArgs->sharedBuffer + blockIdx * SHARED_BUFFER_SIZE);
    __gm__ Metrics* metric = (__gm__ Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    ExecuteContext ctx = {};
    ctx.args = args;
    ctx.blockIdx = blockIdx;
    __gm__ DevDfxArgs* devDfxAddr = (__gm__ DevDfxArgs*)devArgs->devDfxArgAddr;
    ctx.aicoreDevTaskMetric.devTaskMetricEnable = devDfxAddr->isOpenPerfTrace != 0;
    uint8_t aicoreLogLevel = static_cast<uint8_t>(AicoreLogLevel::NONE);
#if ENABLE_AICORE_PRINT
    if (devDfxAddr->logLevel >= 0) {
        aicoreLogLevel = static_cast<uint8_t>(devDfxAddr->logLevel);
    }
#endif
    if (ctx.aicoreDevTaskMetric.devTaskMetricEnable && metric->turnNum < MAX_ROUND_NUM) {
        uint64_t round = metric->turnNum;
        ctx.aicoreDevTaskMetric.devTaskMetric = &(metric->aicoreDevTaskInfo[round]);
        PerfTraceRecord(INVALID_DEV_TASK_ID, ctx.aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_BEGIN, start);
    }

    bool isFirstTask = true;
    SetStatus(args, STAGE_HANDSHAKE_START);
    HandshakeClient(args->shakeBuffer);
    SetStatus(args, STAGE_HANDSHAKE_END);
    set_mask_norm();
    uint32_t curTaskIdx;
    uint32_t lastTaskIdx = AICORE_TASK_INIT;
    uint32_t lastRegHighVal = 0;
    uint8_t lastMixResourceType = static_cast<uint8_t>(MixResourceType::MIX_UNKNOWN);

    PerfTraceRecord(INVALID_DEV_TASK_ID, ctx.aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_INIT);

    volatile __gm__ ParallelDevTask* parallelDevTask = GetCoreFunctionData(&ctx);
    if (parallelDevTask == nullptr) {
        DfxProcWhenCoreExit(&ctx, args, metric);
        return; // no data exit
    }
    InitCtx(&ctx, metric, parallelDevTask, aicoreLogLevel);
    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    while (true) {
        AICORE_TIMEOUT_CHECK_RETURN_VOID(t0, loop_count, AICORE_LEAF_TASK_RUN_TIMEOUT, STAGE_RUN_LEAFTASK_TIMEOUT);

        curTaskIdx = GetNextLeafTask(lastTaskIdx, args);
        if (curTaskIdx == AICORE_TASK_STOP) {
            DfxProcWhenDevTaskStop(&ctx, args, metric);
            SetStatus(args, STAGE_CORE_EXIT);
            WaitWaveSignal(args); // no data exit
            DfxProcWhenCoreExit(&ctx, args, metric);
            return;
        }
        if (curTaskIdx == AICORE_TASK_ABNORMAL_STOP) {
            SetLastWordStatus(args, STAGE_GET_NEXT_TASK_TIMEOUT);
            Trap();
            return;
        }

        if (npu::tile_fwk::DevTaskDcciFlag(curTaskIdx) == 1) {
            DfxProcWhenDevTaskStop(&ctx, args, metric); // perf trace stop last devtask

            ctx.curLeafTaskParallelIdx = npu::tile_fwk::ParallelIndex(curTaskIdx);

            // need dcci new prallel devtask
            uint32_t ret = RefreshParallelDevTask(&ctx, metric, lastRegHighVal);
            if (ret == AICORE_TASK_ABNORMAL_STOP) {
                return;
            }
            isFirstTask = true;
            t0 = get_sys_cnt(); // reset time out
        }

        // update cur leaftask parallelindex
        ctx.curLeafTaskParallelIdx = npu::tile_fwk::ParallelIndex(curTaskIdx);

        if (isFirstTask) {
            PerfTraceRecord(ctx.SeqNo(), ctx.aicoreDevTaskMetric.devTaskMetric,
                            PERF_TRACE_CORE_DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK);
            isFirstTask = false;
            ctx.profLevel = args->taskEntry.reserved[0];
            ctx.lastTaskFinishCycle = 0;
        }

        SendRegAck(curTaskIdx);
        PmuTestBegin(ctx.profLevel);
        ExecCoreFunctionKernel(&ctx, curTaskIdx, lastMixResourceType);
        PmuTestEnd(ctx.profLevel);
        SendRegFinish(curTaskIdx);
        lastTaskIdx = curTaskIdx;
        SetStatus(args, STAGE_FINISH_CUR_TASK);
    }
}

#endif
