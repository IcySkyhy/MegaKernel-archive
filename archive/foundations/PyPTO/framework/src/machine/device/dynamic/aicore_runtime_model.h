/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <deque>
#include <array>

#include "machine/device/dynamic/costmodel_utils.h"
#include "machine/device/dynamic/eslmodel_aicore_hal.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/dynamic/aicore_constants.h"
#include "machine/device/dynamic/eslmodel_manager.h"
#include "machine/simulation/aicore_hardware.h"
#include "interface/machine/device/tilefwk/aicpu_common.h"
#ifndef __DEVICE__
#include "adapter/api/runtime_api.h"
#endif

namespace npu::tile_fwk::dynamic {

static inline volatile KernelArgs* GetKernelArgsByCore(int coreIdx, int64_t sharedBuffer)
{
    return reinterpret_cast<KernelArgs*>(static_cast<uint64_t>(sharedBuffer) + SHARED_BUFFER_SIZE * coreIdx);
}

class ModelBase {
public:
    virtual ~ModelBase() = default;

    virtual void Init() {}

    virtual void SetReadyQueue(int coreIdx, int phyId, uint64_t value) = 0;

    virtual uint64_t GetFinishedTask(int coreIdx, int phyId) = 0;

    virtual void ResetShakeBuf(volatile KernelArgs* arg) = 0;

    virtual bool TryHandShakeByGm(volatile KernelArgs* arg, int64_t dotStatus, int32_t& phyId)
    {
        (void)arg;
        (void)dotStatus;
        (void)phyId;
        return false;
    }

    virtual void InitKernelArgs(volatile KernelArgs*& arg, int coreIdx, int64_t sharedBuffer, int64_t buffer)
    {
        (void)buffer;
        if (arg == nullptr) {
            arg = GetKernelArgsByCore(coreIdx, sharedBuffer);
        }
    }

    virtual void SetParallelDevTask(volatile ParallelDevTask* kernelParallDevTask, int parallelIdx, int64_t funcData,
                                    uint32_t devTaskId) = 0;

    virtual void SetParallelDevTaskSize(volatile ParallelDevTask* kernelParallDevTask, uint32_t front,
                                        uint32_t rear) = 0;

    virtual void SetParallelDevTaskCtxVersion(volatile KernelArgs* arg, uint32_t version) = 0;

    virtual void ResetParallelDevTask(volatile KernelArgs* arg) = 0;

    virtual void InitCostModelDevTaskData(int coreIdx, int64_t funcData)
    {
        (void)coreIdx;
        (void)funcData;
    }
};

class AicoreModel : public ModelBase {
public:
    void SetReadyQueue(int coreIdx, int phyId, uint64_t value) override
    {
        (void)coreIdx;
        AicoreHardware::Global().WriteMainBase(static_cast<size_t>(phyId), value);
    }

    uint64_t GetFinishedTask(int coreIdx, int phyId) override
    {
        (void)coreIdx;
        return AicoreHardware::Global().ReadCond(static_cast<size_t>(phyId));
    }

    void ResetShakeBuf(volatile KernelArgs* arg) override
    {
        if (arg == nullptr) {
            return;
        }
        arg->shakeBuffer[0] = 0;
        arg->shakeBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_COREFUNC_DATA_INDEX] = 0;
        arg->waveBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX] = AICORE_SAY_GOODBYE;
    }

    bool TryHandShakeByGm(volatile KernelArgs* arg, int64_t dotStatus, int32_t& phyId) override
    {
        volatile int64_t* shakeBuffer = arg->shakeBuffer;
        if ((*shakeBuffer & 0xFFFFFFFF) != AICORE_SAY_HELLO) {
            return false;
        }
        arg->taskEntry.reserved[0] = static_cast<uint32_t>(dotStatus);
        phyId = (*shakeBuffer >> NUM_THIRTY_TWO) & AICORE_COREID_MASK;
        arg->waveBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX] = 0;
        return true;
    }

    void SetParallelDevTask(volatile ParallelDevTask* kernelParallDevTask, int parallelIdx, int64_t funcData,
                            uint32_t devTaskId) override
    {
        kernelParallDevTask->ptrElements[parallelIdx % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM] = funcData;
        kernelParallDevTask->idElements[parallelIdx % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM] = devTaskId;
    }

    void SetParallelDevTaskSize(volatile ParallelDevTask* kernelParallDevTask, uint32_t front, uint32_t rear) override
    {
        kernelParallDevTask->front = front;
        kernelParallDevTask->rear = rear;
    }

    void SetParallelDevTaskCtxVersion(volatile KernelArgs* arg, uint32_t version) override
    {
        if (arg == nullptr) {
            return;
        }
        arg->parallelDevTask.version = version;
    }

    void ResetParallelDevTask(volatile KernelArgs* arg) override
    {
        if (arg == nullptr) {
            return;
        }
        arg->parallelDevTask.version = 0;
        arg->parallelDevTask.front = 0;
        arg->parallelDevTask.rear = 0;
        for (uint32_t i = 0; i < npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM; ++i) {
            arg->parallelDevTask.ptrElements[i] = 0;
            arg->parallelDevTask.idElements[i] = 0;
        }
    }
};

class EslModel : public ModelBase {
public:
    void Init() override { eslModel_.Init(); }

    void SetReadyQueue(int coreIdx, int phyId, uint64_t value) override
    {
        (void)phyId;
        uint64_t taskId = (value & 0xFFFFFFFF) - 1;
        bool skipReplay = value == 0 || taskId == AICORE_TASK_INIT || taskId == AICORE_TASK_STOP ||
                          taskId == AICORE_FUNC_STOP;
        bool matchReplay = skipReplay || eslModelReplayMgr_->ReplayMatch(taskId);
        if (matchReplay) {
            eslModel_.WriteEslReg(coreIdx, &value);
            replayQueueFlag_[coreIdx] = 0;
        } else {
            taskIds_[coreIdx].push_back(taskId);
            replayQueueFlag_[coreIdx] = 1;
        }
    }

    uint64_t GetFinishedTask(int coreIdx, int phyId) override
    {
        (void)phyId;
        if (replayQueueFlag_[coreIdx] == 0) {
            uint64_t result = eslModel_.ReadEslReg(coreIdx);
            return result;
        } else {
            if (taskIds_[coreIdx].empty())
                return AICORE_FUNC_STOP | AICORE_FIN_MASK;
            uint64_t taskId = 0;
            while (!taskIds_[coreIdx].empty()) {
                taskId = taskIds_[coreIdx].front();
                taskIds_[coreIdx].pop_front();
            }
            return taskId | AICORE_FIN_MASK;
        }
    }

    void ResetShakeBuf(volatile KernelArgs* arg) override
    {
        if (arg == nullptr) {
            return;
        }
        uint64_t valToSend = 0;
        eslModel_.WriteEslMem(reinterpret_cast<uint64_t>(&arg->shakeBuffer[0]), sizeof(uint64_t), &valToSend);
        valToSend = AICORE_SAY_GOODBYE;
        eslModel_.WriteEslMem(reinterpret_cast<uint64_t>(&arg->waveBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX]),
                              sizeof(uint64_t), &valToSend);
    }

    bool TryHandShakeByGm(volatile KernelArgs* arg, int64_t dotStatus, int32_t& phyId) override
    {
#ifndef __DEVICE__
        KernelArgs hostArgs;
        RuntimeMemcpyDirect(&hostArgs, sizeof(KernelArgs),
                            reinterpret_cast<uint8_t*>(PtrToValue((const volatile void*)arg)), sizeof(KernelArgs),
                            RtMemcpyKind::DEVICE_TO_HOST);
        if ((hostArgs.shakeBuffer[0] & 0xFFFFFFFF) != AICORE_SAY_HELLO) {
            return false;
        }
        hostArgs.taskEntry.reserved[0] = static_cast<uint32_t>(dotStatus);
        phyId = static_cast<int32_t>((hostArgs.shakeBuffer[0] >> NUM_THIRTY_TWO) & AICORE_COREID_MASK);
        hostArgs.waveBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX] = 0;
        RuntimeMemcpyDirect(reinterpret_cast<uint8_t*>(PtrToValue(arg)), sizeof(KernelArgs), &hostArgs,
                            sizeof(KernelArgs), RtMemcpyKind::HOST_TO_DEVICE);
        return true;
#else
        (void)arg;
        (void)dotStatus;
        (void)phyId;
        return false;
#endif
    }

    void InitKernelArgs(volatile KernelArgs*& arg, int coreIdx, int64_t sharedBuffer, int64_t buffer) override
    {
        (void)buffer;
        if (arg == nullptr) {
            arg = GetKernelArgsByCore(coreIdx, sharedBuffer);
        }
#ifndef __DEVICE__
        int64_t valToSend = 0;
        RuntimeMemcpyDirect(reinterpret_cast<uint8_t*>(
                                PtrToValue(&arg->shakeBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_COREFUNC_DATA_INDEX])),
                            sizeof(valToSend), &valToSend, sizeof(valToSend), RtMemcpyKind::HOST_TO_DEVICE);
#endif
    }

    void SetParallelDevTask(volatile ParallelDevTask* kernelParallDevTask, int parallelIdx, int64_t funcData,
                            uint32_t devTaskId) override
    {
        eslModel_.WriteEslMem(
            reinterpret_cast<uint64_t>(
                &kernelParallDevTask->ptrElements[parallelIdx % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM]),
            sizeof(funcData), &funcData);
        eslModel_.WriteEslMem(
            reinterpret_cast<uint64_t>(
                &kernelParallDevTask->idElements[parallelIdx % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM]),
            sizeof(devTaskId), &devTaskId);
    }

    void SetParallelDevTaskSize(volatile ParallelDevTask* kernelParallDevTask, uint32_t front, uint32_t rear) override
    {
        eslModel_.WriteEslMem(reinterpret_cast<uint64_t>(&kernelParallDevTask->front), sizeof(front), &front);
        eslModel_.WriteEslMem(reinterpret_cast<uint64_t>(&kernelParallDevTask->rear), sizeof(rear), &rear);
    }

    void SetParallelDevTaskCtxVersion(volatile KernelArgs* arg, uint32_t version) override
    {
        if (arg == nullptr) {
            return;
        }
        eslModel_.WriteEslMem(reinterpret_cast<uint64_t>(&arg->parallelDevTask.version), sizeof(version), &version);
    }

    void ResetParallelDevTask(volatile KernelArgs* arg) override
    {
        if (arg == nullptr) {
            return;
        }
        uint32_t u32Zero = 0;
        eslModel_.WriteEslMem(reinterpret_cast<uint64_t>(&arg->parallelDevTask.version), sizeof(u32Zero), &u32Zero);
        eslModel_.WriteEslMem(reinterpret_cast<uint64_t>(&arg->parallelDevTask.front), sizeof(u32Zero), &u32Zero);
        eslModel_.WriteEslMem(reinterpret_cast<uint64_t>(&arg->parallelDevTask.rear), sizeof(u32Zero), &u32Zero);

        int64_t i64Zero = 0;
        for (uint32_t i = 0; i < npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM; ++i) {
            eslModel_.WriteEslMem(reinterpret_cast<uint64_t>(&arg->parallelDevTask.ptrElements[i]), sizeof(i64Zero),
                                  &i64Zero);
            eslModel_.WriteEslMem(reinterpret_cast<uint64_t>(&arg->parallelDevTask.idElements[i]), sizeof(u32Zero),
                                  &u32Zero);
        }
    }

    void SetEslModelReplayManager(EslModelReplayManager* replayMgr) { eslModelReplayMgr_ = replayMgr; }

private:
    EslAicoreHal eslModel_;
    std::array<std::deque<uint64_t>, MAX_AICORE_NUM> taskIds_;
    EslModelReplayManager* eslModelReplayMgr_{nullptr};
    std::array<int, MAX_AICORE_NUM> replayQueueFlag_;
};

class CostModelAdapter : public ModelBase {
public:
    void SetActualModel(CostModel::AiCoreModel* actualModel) { actualModel_ = actualModel; }

    void SetSendTask(std::function<void(int, uint64_t)> sendTask) { sendTask_ = std::move(sendTask); }

    void SetGetTask(std::function<uint64_t(int)> getTask) { getTask_ = std::move(getTask); }

    void SetReadyQueue(int coreIdx, int phyId, uint64_t value) override
    {
        (void)phyId;
        auto taskId = (value & 0xFFFFFFFF) - 1;
        if (value == 0 || taskId == AICORE_TASK_STOP || (taskId & 0xFFFFFFFF) == AICORE_FUNC_STOP) {
            return;
        }
        if (sendTask_) {
            sendTask_(coreIdx, taskId & 0xFFFFFFFF);
        }
    }

    uint64_t GetFinishedTask(int coreIdx, int phyId) override
    {
        (void)phyId;
        return getTask_ ? getTask_(coreIdx) : 0;
    }

    void ResetShakeBuf(volatile KernelArgs* arg) override { (void)arg; }

    bool TryHandShakeByGm(volatile KernelArgs* arg, int64_t dotStatus, int32_t& phyId) override
    {
        (void)arg;
        (void)dotStatus;
        (void)phyId;
        return false;
    }

    void InitKernelArgs(volatile KernelArgs*& arg, int coreIdx, int64_t sharedBuffer, int64_t buffer) override
    {
        (void)arg;
        (void)coreIdx;
        (void)sharedBuffer;
        (void)buffer;
    }

    void SetParallelDevTask(volatile ParallelDevTask* kernelParallDevTask, int parallelIdx, int64_t funcData,
                            uint32_t devTaskId) override
    {
        (void)kernelParallDevTask;
        (void)parallelIdx;
        (void)funcData;
        (void)devTaskId;
    }

    void SetParallelDevTaskSize(volatile ParallelDevTask* kernelParallDevTask, uint32_t front, uint32_t rear) override
    {
        (void)kernelParallDevTask;
        (void)front;
        (void)rear;
    }

    void SetParallelDevTaskCtxVersion(volatile KernelArgs* arg, uint32_t version) override
    {
        (void)arg;
        (void)version;
    }

    void ResetParallelDevTask(volatile KernelArgs* arg) override { (void)arg; }

    void InitCostModelDevTaskData(int coreIdx, int64_t funcData) override
    {
        if (actualModel_ != nullptr) {
            actualModel_->InitData(coreIdx, funcData);
        }
    }

private:
    CostModel::AiCoreModel* actualModel_{nullptr};
    std::function<void(int, uint64_t)> sendTask_{nullptr};
    std::function<uint64_t(int)> getTask_{nullptr};
};

} // namespace npu::tile_fwk::dynamic
