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
 * \file shmem_wait_until.h
 * \brief
 */

#ifndef SHMEM_WAIT_UNTIL_H
#define SHMEM_WAIT_UNTIL_H

#include <vector>

#include "common.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/device/dynamic/device_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/aikernel_data.h"
#include "interface/tileop/distributed/comm_context.h"

namespace npu::tile_fwk::Distributed {

constexpr uint16_t INVALID_INDEX = 0xFFFF;

struct SignalTileOp {
    void Init(uint64_t taskId, int32_t* addr, int32_t expectedSum, bool resetSignal)
    {
        taskId_ = taskId;
        addr_ = addr;
        expectedSum_ = expectedSum;
        resetSignal_ = resetSignal;
    }
    bool PollCompleted() const;

    uint16_t nextIndex{INVALID_INDEX};
    uint64_t taskId_{0};
    int32_t* addr_{nullptr};
    int32_t expectedSum_{0};
    bool resetSignal_{false};
    TaskStat* profData_{nullptr};
};

struct ShmemWaitUntilCache {
    SignalTileOp taskArray[AICPU_TASK_ARRAY_SIZE];
    uint16_t hashTable[AICPU_TASK_ARRAY_SIZE];
    uint32_t taskCount{0};
};

inline uint32_t HashTaskId(uint32_t taskId) { return taskId & AICPU_TASK_ARRAY_SIZE_MOD; }

inline SignalTileOp* FindTask(uint32_t taskId, ShmemWaitUntilCache* cache)
{
    uint32_t index = HashTaskId(taskId);
    uint16_t currentIndex = cache->hashTable[index];
    uint32_t loopCount = 0;
    while (currentIndex != INVALID_INDEX && loopCount < AICPU_TASK_ARRAY_SIZE) {
        SignalTileOp* task = &cache->taskArray[currentIndex];
        if (task->taskId_ == taskId) {
            return task;
        }
        currentIndex = task->nextIndex;
        loopCount++;
    }
    return nullptr;
}

class CircularQueue {
public:
    CircularQueue() = default;

    inline int32_t Enqueue(SignalTileOp* task)
    {
        queue_[rear_] = task;
        rear_ = (rear_ + 1) & AICPU_TASK_ARRAY_SIZE_MOD;
        if (rear_ == front_) {
            DEV_ERROR(DistributedErrorCode::AICPU_TASK_NUM_EXCEED_LIMIT,
                      "ctrl.task.pre.task.enqueue#: SignalTileOp queue is full, capacity = %lu",
                      AICPU_TASK_ARRAY_SIZE - 1);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        return dynamic::DEVICE_MACHINE_OK;
    }

    inline bool IsEmpty() const { return front_ == rear_; }

    inline int32_t Dequeue()
    {
        if (IsEmpty()) {
            DEV_ERROR(DistributedErrorCode::AICPU_TASK_QUEUE_EMPTY, "sche.task.end.task.dequeue#: Queue is empty.");
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        front_ = (front_ + 1) & AICPU_TASK_ARRAY_SIZE_MOD;
        return dynamic::DEVICE_MACHINE_OK;
    }

    inline const SignalTileOp* operator[](uint16_t index) const { return queue_[index]; }

    inline int32_t Remove(uint16_t index)
    {
        queue_[index] = queue_[front_];
        return Dequeue();
    }

    int32_t PollCompleted(std::function<int32_t(SignalTileOp*)> processor)
    {
        uint16_t current = front_;
        uint16_t end = rear_;
        if (current > end) {
            end += AICPU_TASK_ARRAY_SIZE;
        }
        for (uint16_t i = current; i < end; ++i) {
            uint16_t actualIndex = i & AICPU_TASK_ARRAY_SIZE_MOD;
            SignalTileOp* task = queue_[actualIndex];
            if (task->PollCompleted()) {
                if (task->profData_ != nullptr) {
                    task->profData_->execEnd = dynamic::GetCycles();
                }
                int32_t ret = processor(task);
                if (ret != dynamic::DEVICE_MACHINE_OK) {
                    return ret;
                }
                ret = Remove(actualIndex);
                if (ret != dynamic::DEVICE_MACHINE_OK) {
                    return ret;
                }
            }
        }
        return dynamic::DEVICE_MACHINE_OK;
    }

private:
    SignalTileOp* queue_[AICPU_TASK_ARRAY_SIZE];
    uint16_t front_{0};
    uint16_t rear_{0};
};

class ShmemWaitUntilImpl {
public:
    inline void LoadCache(ShmemWaitUntilCache* cache, uint32_t parallelIdx = 0)
    {
        if (cache == nullptr) {
            DEV_INFO("LoadCache: cache is nullptr, skip");
            return;
        }

        DEV_INFO("LoadCache: cache loaded with %u tasks, parallelIdx=%u", cache->taskCount, parallelIdx);
        cachePtr_[parallelIdx] = cache;
    }

    inline int32_t EnqueueOp(uint64_t taskId, uint32_t parallelIdx = 0, TaskStat* taskStat = nullptr)
    {
        SignalTileOp* task = FindTask(taskId, cachePtr_[parallelIdx]);
        if (task == nullptr) {
            DEV_ERROR(DistributedErrorCode::AICPU_TASKID_NOT_IN_MAP,
                      "ctrl.task.pre.task.enqueue#: taskId=%lu not found, parallelIdx=%u", taskId, parallelIdx);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        if (taskStat != nullptr) {
            task->profData_ = taskStat;
            task->profData_->taskId = static_cast<int32_t>(taskId);
            task->profData_->execStart = dynamic::GetCycles();
        }
        return runingTaskQueue_[parallelIdx].Enqueue(task);
    }

    struct PrepareResult {
        int32_t* addr;
        uint64_t rawAddr;
        uint64_t vaddr;
        uint32_t ownerRankId;
        uint32_t expectedSum;
        uint32_t resetSignal;
        uint32_t totalTileNum;
        uint32_t tileIndex;
        uint32_t bufferStride;
    };

    static inline PrepareResult CalculateTaskAddress(uint64_t taskId,
                                                     const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode,
                                                     npu::tile_fwk::DynFuncData* funcDataList, int64_t* hcclContextAddr)
    {
        AicpuParamInfo paramInfo = DecodeAicpuCode(aicpuCode);
        TensorInfo info = ShmemWaitUntilImpl::GetTensorInfo(taskId, aicpuCode, funcDataList, hcclContextAddr,
                                                            paramInfo);

        uint32_t tileIndex = 0;
        uint32_t viewIndexAccum = 0;

        uint32_t dataDim = paramInfo.dim;
        uint32_t tileShapeDim = paramInfo.tileShape.size();
        uint32_t startDim = dataDim - tileShapeDim;

        for (uint32_t dimIdx = 0; dimIdx < tileShapeDim; ++dimIdx) {
            uint32_t curDim = startDim + dimIdx;
            uint32_t viewShape = paramInfo.viewshapes[dimIdx];
            uint32_t offset = info.offset[curDim];
            uint32_t tileShapeVal = paramInfo.tileShape[dimIdx];

            uint32_t viewIdx = offset / viewShape;
            uint32_t viewOffset = offset % viewShape;

            uint32_t viewTileIdx = viewOffset / tileShapeVal;

            tileIndex += viewTileIdx * paramInfo.viewTileStrides[dimIdx];
            viewIndexAccum += viewIdx * paramInfo.viewIndexStrides[dimIdx];
        }
        tileIndex += viewIndexAccum * paramInfo.viewTileNum;

        int32_t* addr = reinterpret_cast<int32_t*>(info.rawAddr) +
                        CalcLinearOffset(paramInfo.totalTileNum, info.offset[OWNER_RANK_ID_INDEX], tileIndex) *
                            paramInfo.bufferStride;

        return PrepareResult{addr,
                             info.rawAddr,
                             info.vaddr,
                             info.offset[OWNER_RANK_ID_INDEX],
                             static_cast<uint32_t>(info.expectedSum),
                             static_cast<uint32_t>(info.resetSignal),
                             paramInfo.totalTileNum,
                             tileIndex,
                             paramInfo.bufferStride};
    }

    static inline void BuildHashTable(ShmemWaitUntilCache* cache, uint32_t taskCount)
    {
        (void)memset_s(cache->hashTable, sizeof(cache->hashTable), 0xFF, sizeof(cache->hashTable));
        for (uint32_t i = 0; i < taskCount; ++i) {
            SignalTileOp* task = &cache->taskArray[i];
            uint32_t index = HashTaskId(task->taskId_);
            task->nextIndex = cache->hashTable[index];
            cache->hashTable[index] = static_cast<uint16_t>(i);
        }
    }

    static inline int32_t PrepareTask(uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode,
                                      SignalTileOp* targetArray, uint32_t taskIndex,
                                      npu::tile_fwk::DynFuncData* funcDataList, int64_t* hcclContextAddr)
    {
        auto result = CalculateTaskAddress(taskId, aicpuCode, funcDataList, hcclContextAddr);

        targetArray[taskIndex].addr_ = result.addr;
        targetArray[taskIndex].expectedSum_ = result.expectedSum;
        targetArray[taskIndex].resetSignal_ = result.resetSignal;
        targetArray[taskIndex].taskId_ = taskId;

        DEV_DEBUG("PrepareTask taskId=%lu, baseAddr=0x%lx, actualAddr=0x%lx, ownerRank=%u, actual rawShape=[%lu, %u],"
                  "actual offset=[%u, %u], buffer maxTileNum=%lu, bufferStride=%u",
                  taskId, result.rawAddr, reinterpret_cast<uint64_t>(result.addr), result.ownerRankId,
                  GetRankNum(hcclContextAddr, result.vaddr), result.totalTileNum, result.ownerRankId, result.tileIndex,
                  TileOp::Distributed::DecodeShmemAddrMaxTileNum(result.vaddr), result.bufferStride);

        return dynamic::DEVICE_MACHINE_OK;
    }

    int32_t PollCompleted(npu::tile_fwk::dynamic::AiCoreManager* aiCoreManager, uint32_t parallelIdx = 0);

    CircularQueue runingTaskQueue_[SCH_DEVTASK_MAX_PARALLELISM];

private:
    ShmemWaitUntilCache* cachePtr_[SCH_DEVTASK_MAX_PARALLELISM] = {};

    static TensorInfo GetTensorInfo(uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode,
                                    npu::tile_fwk::DynFuncData* funcDataList, int64_t* hcclContextAddr,
                                    const AicpuParamInfo& paramInfo);
};

} // namespace npu::tile_fwk::Distributed
#endif // SHMEM_WAIT_UNTIL_H
