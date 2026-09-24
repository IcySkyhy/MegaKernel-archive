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
 * \file aicpu_task_manager.h
 * \brief
 */

#pragma once

#include <functional>
#include <atomic>
#include <vector>
#include <array>
#include <malloc.h>
#include <queue>

#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/device/distributed/common.h"
#include "machine/device/distributed/shmem_wait_until.h"
#include "machine/utils/machine_ws_intf.h"
#include "interface/operation/opcode.h"
#include "interface/utils/common.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk::dynamic {

class AicpuTaskManager {
public:
    enum TaskType {
        SHMEM_WAIT_UNTIL = 0,
        TASK_TYPE_NUM,
    };

    AicpuTaskManager() {};
    ~AicpuTaskManager() {};

    // 每个AICPU都会调用
    inline void TaskEnqueue(uint64_t taskId, DynDeviceTask* deviceTask)
    {
        auto readyQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(deviceTask->devTask.readyAicpuFunctionQue);
        bool res = readyQueue->TryEnqueue(taskId);
        DEV_ASSERT(SchedErr::READY_QUEUE_OVERFLOW, res);
    }

    inline void InitDeviceArgs(DeviceArgs* deviceArgs)
    {
        sharedBuffer_ = deviceArgs->sharedBuffer;
        aicNum_ = deviceArgs->nrAic;
        aivNum_ = deviceArgs->nrAiv;
        archInfo_ = deviceArgs->archInfo;
    }

    inline int32_t Init(DynDeviceTask* deviceTask, bool profSwitch, uint32_t parallelIdx = 0)
    {
        auto cache = reinterpret_cast<npu::tile_fwk::Distributed::ShmemWaitUntilCache*>(
            deviceTask->shmemWaitUntilCacheBackup);
        if (cache != nullptr) {
            DEV_INFO("MachineFlow: loading cache with pre-built hashTable, taskCount=%u, parallelIdx=%u",
                     cache->taskCount, parallelIdx);
            shmemWaitUntil_.LoadCache(cache, parallelIdx);
            DEV_INFO("MachineFlow: cache loaded successfully for parallelIdx=%u", parallelIdx);

            if (profSwitch) {
                KernelArgs* args = (KernelArgs*)(sharedBuffer_ + (aicNum_ + aivNum_) * SHARED_BUFFER_SIZE);
                aicpuTaskStat_ = (Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
            }
            return DEVICE_MACHINE_OK;
        }
        DEV_ERROR(DevCommonErr::PARAM_INVALID, "MachineFlow: no cache prepared, shmemWaitUntil not initialized");
        return DEVICE_MACHINE_ERROR;
    }

    inline int32_t TaskProcess(uint64_t& taskCount, DynDeviceTask* deviceTask, uint32_t parallelIdx = 0)
    {
        auto readyQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(deviceTask->devTask.readyAicpuFunctionQue);
        if (readyQueue->UnsafeAtomicSize() == 0) {
            return DEVICE_MACHINE_OK;
        }
        auto tasksRange = readyQueue->DequeueAll();
        taskCount = tasksRange.second - tasksRange.first;
        for (auto it = tasksRange.first; it != tasksRange.second; ++it) {
            auto ret = TaskDispatch(*it, deviceTask, parallelIdx);
            if (ret != DEVICE_MACHINE_OK) {
                return ret;
            }
        }
        return DEVICE_MACHINE_OK;
    }

    inline int32_t TaskPoll(AiCoreManager* aiCoreManager, uint32_t parallelIdx = 0)
    {
        return shmemWaitUntil_.PollCompleted(aiCoreManager, parallelIdx);
    }

    inline bool Finished(uint32_t parallelIdx = 0) { return shmemWaitUntil_.runingTaskQueue_[parallelIdx].IsEmpty(); }

    inline int32_t SyncAicpuTaskFinish(AiCoreManager* aiCoreManager, uint32_t parallelIdx = 0)
    {
        TIMEOUT_CHECK_INIT(archInfo_, TIMEOUT_10SEC);
        while (!Finished(parallelIdx)) {
            auto ret = TaskPoll(aiCoreManager, parallelIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            __PYPTO_TIMEOUT_CHECK_EXIT_ONLY(DistributedErrorCode::AICPU_TASK_TIMEOUT,
                                            return DEVICE_MACHINE_TIMEOUT_SYNC_AICPU_FINISH,
                                            "#sche.task.end.sync: SyncAicpuTaskFinish.");
        }
        return DEVICE_MACHINE_OK;
    }

private:
    inline TaskType GetTaskType(uint64_t taskId, DynDeviceTask* deviceTask)
    {
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto callList = deviceTask->dynFuncDataCacheList[funcId].calleeList;
        auto& code = deviceTask->aicpuLeafBinary[callList[opIndex]].aicpuLeafCode;
        auto taskType = TaskType::TASK_TYPE_NUM;
        switch (code[0]) {
            case static_cast<uint32_t>(Opcode::OP_SHMEM_WAIT_UNTIL):
                taskType = TaskType::SHMEM_WAIT_UNTIL;
                break;
            default:
                break;
        }
        return taskType;
    }

    inline int32_t TaskDispatch(uint64_t taskId, DynDeviceTask* deviceTask, uint32_t parallelIdx = 0)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        auto taskType = GetTaskType(taskId, deviceTask);
        DEV_VERBOSE_DEBUG("Dispatch aicpu task %lu.", taskId);
        if (taskType < TaskType::TASK_TYPE_NUM) {
            TaskStat* taskStat = nullptr;
            if (aicpuTaskStat_ != nullptr) {
                taskStat = &(aicpuTaskStat_->tasks[aicpuTaskStat_->taskCount]);
                ++aicpuTaskStat_->taskCount;
            }
            ret = shmemWaitUntil_.EnqueueOp(taskId, parallelIdx, taskStat);
        }
        return ret;
    }

    ArchInfo archInfo_{ArchInfo::DAV_2201};
    npu::tile_fwk::Distributed::ShmemWaitUntilImpl shmemWaitUntil_;
    Metrics* aicpuTaskStat_{nullptr};
    uint64_t sharedBuffer_{0};
    uint32_t aicNum_{0};
    uint32_t aivNum_{0};
};
} // namespace npu::tile_fwk::dynamic
