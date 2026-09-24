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
 * \file device_task.h
 * \brief
 */

#pragma once

#include "vector.h"
#include "allocator/allocators.h"
#include "tilefwk/core_func_data.h"
#include "interface/machine/device/tilefwk/aicpu_perf.h"
#include "machine/utils/dynamic/spsc_queue.h"
#include "dev_encode_program_ctrlflow_cache.h"

#ifndef __DEVICE__
#include "interface/configs/config_manager.h"
#endif

namespace npu::tile_fwk::dynamic {
class DeviceWorkspaceAllocator;
struct DynDeviceTask : DynDeviceTaskBase {
    Vector<DevAscendFunctionDupped, WsMemCategory::VECTOR_STITCHED_LIST, DeviceWorkspaceAllocator> stitchedList;
    WsAllocation selfAlloc;
    WsSlabStageAllocMem taskStageAllocMem;

    static uint32_t GetReadyQueueIndexByCoreType(CoreType coreType)
    {
        if (coreType == CoreType::AICPU) {
            return static_cast<uint32_t>(READY_QUEUE_SIZE) - 1;
        }
        return static_cast<uint32_t>(coreType);
    }

    static uint32_t GetGlobalTaskReadyQueueIndexByCoreType(CoreType coreType)
    {
        if (coreType == CoreType::AIC) {
            return DRCO_QUEUE_AIC;
        } else if (coreType == CoreType::AIV) {
            return DRCO_QUEUE_AIV;
        }
        return static_cast<uint32_t>(-1);
    }

    DynDeviceTask(DeviceWorkspaceAllocator& allocator)
    {
        InitDevTaskShell(devTask);
        stitchedList.InitAllocator(allocator);
    }

private:
    // Zero only fields read before explicit assignment; opWrapList is filled per-func in MoveTo.
    static void InitDevTaskShell(DeviceTask& task)
    {
        task.coreFunctionCnt = 0;
        task.coreFunctionReadyStateAddr = 0;
        task.readyAicCoreFunctionQue = 0;
        task.readyAivCoreFunctionQue = 0;
        task.readyAicpuFunctionQue = 0;
        task.dieReadyFunctionQue = DieReadyQueueData{};
        task.mixTaskData.readyWrapCoreFunctionQue = 0;
        task.mixTaskData.wrapIdNum = 0;
        task.coreFuncData = CoreFunctionData{};
        task.l2Info = L2PreInfo{};
        task.costModelData = 0;
        task.aicoreModel = 0;
    }

public:
    predcount_t& GetOperationCurrPredCount(uint32_t id)
    {
        return stitchedList[FuncID(id)].GetOperationCurrPredCount(TaskID(id));
    }

    int GetOperationCoreType(uint32_t id)
    {
        auto callee = stitchedList[FuncID(id)].GetSource()->GetOperationAttrCalleeIndex(TaskID(id));
        return cceBinary[callee].coreType;
    }

    std::string DumpTaskData(uint32_t id)
    {
        auto& funcDup = stitchedList[FuncID(id)];
        return funcDup.DumpDyn(FuncID(id), TaskID(id), cceBinary);
    }

    void DumpTopo(bool enableVFFusion)
    {
        auto header = GetDynFuncDataList();
#ifdef __DEVICE__
        std::string path = "./output/dyn_topo.txt";
#else
        std::string path = config::LogTopFolder() + "/dyn_topo.txt";
#endif
        static std::string lastPath;
        static std::ofstream of;
        if (path != lastPath) {
            if (of.is_open()) {
                of.flush();
                of.close();
            }
            lastPath = path;
            of.open(path);
        }
        if (of.tellp() == 0) {
            of << "seqNo,taskId,rootIndex,rootHash,opmagic,leafIndex,leafHash,coreType,psgId,wrapId,staticSuccCount,"
                  "successors\n";
        }
        for (size_t funcIdx = 0; funcIdx < stitchedList.size(); funcIdx++) {
            stitchedList[funcIdx].DumpTopo(of, header->seqNo, funcIdx, cceBinary, enableVFFusion, &devTask);
        }
        of.flush();
    }

    void DumpLeafs()
    {
        for (size_t funcIdx = 0; funcIdx < stitchedList.size(); funcIdx++) {
            auto lines = stitchedList[funcIdx].DumpLeafs(GetDynFuncDataList()->seqNo, funcIdx);
            for (auto&& line : lines) {
                DEV_DEBUG("[DumpLeafs] %s", line.c_str());
            }
        }
    }

#if DEBUG_INFINITE_LIFETIME
    void DumpTensorAddrInfo(uintdevptr_t dumpTensorWsAddr, uint64_t dumpTensorWsSize)
    {
        UNUSED(dumpTensorWsAddr);
        UNUSED(dumpTensorWsSize);
        std::stringstream oss;
        std::vector<std::string> infos;
        for (uint32_t funcIdx = 0; funcIdx < stitchedList.size(); funcIdx++) {
            stitchedList[funcIdx].DumpTensorAddrInfo(infos, GetDynFuncDataList()->seqNo, funcIdx);
        }
        auto str = std::move(oss).str();
        DEV_INFO("[DumpTensor] seqNo,taskId,rawMagic,address,dtype,bytesOfDtype,(shapes,)");
        DEV_INFO("[DumpTensor] >>>");
        for (auto& info : infos) {
            DEV_INFO("[DumpTensor] %s", info.c_str());
        }
        DEV_INFO("[DumpTensor] <<<");
    }
#endif
};

#define DYN_DEVICE_TASK_EXT_SIZE 0x400
static_assert(sizeof(DynDeviceTask) < sizeof(DynDeviceTaskBase) + DYN_DEVICE_TASK_EXT_SIZE,
              "Invalid dyn device task extension");

struct DeviceTaskCtrl {
    uint64_t taskId{0};
    DeviceTask* devTask{nullptr};

    // 这些原子变量跨线程了，不能sche与ctrl间两边同时写
    std::atomic<uint64_t> finishedFunctionCnt{0};
    std::atomic<bool> runFlag{false};
    std::atomic<int> runCnt{0};

    std::atomic<bool> existNextSameIterTask{false}; // mark have next task
    std::atomic<uint64_t> nextSameIterTaskCtrl{0}; // make sure fetched as soon as possible,ctrl cpu maybe set next task
                                                   // ctrl after this taskctrl fetched by schedule cpu
    std::atomic<bool> notFree{false};
    std::atomic<int> freeCnt{0}; // make sure all sch release this ctrl

    uint32_t ParallelForId() { return (reinterpret_cast<DynDeviceTask*>(devTask))->ParallelForId(); }
    uint32_t ParallelIterId() { return (reinterpret_cast<DynDeviceTask*>(devTask))->ParallelIterId(); }
    bool SupportParallel() { return (reinterpret_cast<DynDeviceTask*>(devTask))->ParallelForId() != 0; }
    uint32_t ParallelWsId() { return (reinterpret_cast<DynDeviceTask*>(devTask))->ParallelWsId(); }
    bool ExistNextSameIterTask() { return existNextSameIterTask.load(std::memory_order_acquire); }
    uint32_t GetMaxC() { return (reinterpret_cast<DynDeviceTask*>(devTask))->GetMaxC(); }
    uint32_t GetMaxV() { return (reinterpret_cast<DynDeviceTask*>(devTask))->GetMaxV(); }
    DeviceTaskCtrl* NextSameIterTaskCtrl()
    {
        return reinterpret_cast<DeviceTaskCtrl*>(nextSameIterTaskCtrl.load(std::memory_order_acquire));
    }

    inline bool IsNotFree() { return notFree.load(std::memory_order_acquire); }
    void SetFree()
    {
        auto* dynTask = reinterpret_cast<DynDeviceTask*>(devTask);
        dynTask->taskStageAllocMem.VerifyAll();
        dynTask->taskStageAllocMem.canFree.store(true);
        notFree.store(false,
                      std::memory_order_release); // parallel device task will reuse this ctrl,so this ctrl cannot free
    }

    void Free(int scheCpuNum)
    {
        if (freeCnt.fetch_add(1, std::memory_order_acquire) == scheCpuNum - 1) {
            SetFree();
        }

        DEV_DEBUG("freecnt : %d", freeCnt.load());
    }

    bool Finish(bool syncWait, int scheCpuNum)
    {
        if (runCnt.fetch_add(1, std::memory_order_acquire) == scheCpuNum - 1) {
            // parallel device task will reuse this ctrl,so this ctrl cannot free, non-parallel task can free
            if (syncWait) {
                SetFree();
            }
            return true;
        } else {
            if (syncWait) {
                // sync point, ensure all aiore_manager threads task finished
                while (runCnt.load(std::memory_order_acquire) != scheCpuNum) {
                }
                return true;
            } else if (runCnt.load(std::memory_order_acquire) == scheCpuNum) {
                return true;
            }
        }
        return false;
    }

    // wait other sch aicpu finish this devtask
    bool TryWaitAllSchFinish(int scheCpuNum)
    {
        if (runCnt.load(std::memory_order_acquire) == scheCpuNum) {
            return true;
        }
        return false;
    }

    void BindTask(DeviceTask* newDevTask)
    {
        devTask = newDevTask;
        taskId = (reinterpret_cast<DynDeviceTask*>(devTask))->GetIndex();

        finishedFunctionCnt.store(0, std::memory_order_relaxed);
        runFlag.store(true, std::memory_order_release);
        notFree.store(true, std::memory_order_release);
        nextSameIterTaskCtrl = 0;

        // parallel devtask  have next same iter task default
        existNextSameIterTask = (reinterpret_cast<DynDeviceTask*>(devTask)->ParallelForId() != 0) ? true : false;
    }

    void InitSchNumCnt()
    {
        runCnt.store(0, std::memory_order_relaxed);
        freeCnt.store(0, std::memory_order_relaxed);
    }
};

constexpr uint32_t DEFAULT_QUEUE_SIZE = 64;
using DeviceTaskCtrlQueue = SPSCQueue<DeviceTaskCtrl*, DEFAULT_QUEUE_SIZE>;

const uint64_t DEV_ARGS_SIZE = 1024; // sizeof(DevStartArgs) is enough, tmp for test GE graph

const uint64_t DEVICE_TASK_CTRL_POOL_SIZE = AlignUp((MAX_DEVICE_TASK_NUM * sizeof(DeviceTaskCtrl)), 512);

const uint64_t DEVICE_TASK_QUEUE_SIZE = sizeof(DeviceTaskCtrlQueue);

const uint64_t DEVICE_SHM_SIZE = DEV_ARGS_SIZE + DEVICE_TASK_CTRL_POOL_SIZE;

static inline uint64_t FillDeviceRuntimeSize(uint64_t& offset, uint64_t size)
{
    uint64_t result = offset;
    offset += size;
    return result;
}

static inline void FillDeviceRuntimeOffset(DevAscendProgram* devProg, uint64_t count)
{
    DeviceRuntimeOffset& offset = devProg->deviceRuntimeOffset;

    uint64_t off = 0;
    offset.startArgsOffset = FillDeviceRuntimeSize(off, DEV_ARGS_SIZE);
    offset.taskCtrlPoolOffset = FillDeviceRuntimeSize(off, DEVICE_TASK_CTRL_POOL_SIZE);
    offset.taskQueueOffset = FillDeviceRuntimeSize(off, DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum);
    offset.deviceTaskReadyQueueOffset = FillDeviceRuntimeSize(off, sizeof(npu::tile_fwk::DrcoDeviceTaskReadyQueue));
    offset.generalOffset = FillDeviceRuntimeSize(off, devProg->memBudget.metadata.general);
    offset.stitchPoolOffset = FillDeviceRuntimeSize(off, devProg->memBudget.metadata.stitchPool);
    offset.size = off;
    offset.count = count;
}

} // namespace npu::tile_fwk::dynamic
