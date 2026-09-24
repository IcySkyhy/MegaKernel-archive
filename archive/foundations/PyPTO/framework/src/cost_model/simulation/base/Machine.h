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
 * \file Machine.h
 * \brief
 */

#pragma once

#include <memory>
#include <string>
#include <climits>
#include <map>

#include "cost_model/simulation/common/Packet.h"
#include "cost_model/simulation/common/BaseQueue.h"

namespace CostModel {
class SimSys;
class Machine : public SimObj {
public:
    /* \brief Unique id of each machine. */
    std::size_t machineId = 0; // Process ID
    std::size_t coreTid = 1;   // Thread ID of the entire view of the machine.
    std::size_t queueSeq = 1;
    std::size_t reversedTidNum = 100;
    std::size_t functionCacheTid = 0;
    uint64_t nextCycles = INT_MAX;
    uint64_t lastCycles = 0;
    bool executingTask = false;
    bool needTerminate = false;
    MachineType machineType = MachineType::UNKNOWN;
    std::shared_ptr<Machine> parentMachine = nullptr;
    std::shared_ptr<Machine> l2cacheMachine = nullptr;
    std::vector<std::shared_ptr<Machine>> subMachines;
    std::shared_ptr<CostModel::SimSys> sim = nullptr;

    // Queues for task and data management
    uint64_t maxRunningTasks = 1;
    SimQueue<TaskPack> submissionQueue;
    SimQueue<CompletedPacket> completionQueue;
    SimQueue<int> outcastReferenceQueue;
    SimQueue<int> incastReferenceQueue;
    SimQueue<int> releaseQueue;
    SimQueue<CachePacket> cacheRespQueue;

    /* \brief Simulate the machine every step */
    void Step() override = 0;
    void Xfer() override = 0;
    /* \brief build machine */
    void Build() override = 0;
    /* \brief reset */
    void Reset() override = 0;
    /* \brief reset */
    std::shared_ptr<SimSys> GetSim() override = 0;
    virtual void Report() = 0;
    virtual void InitQueueDelay() = 0;
    virtual void StepQueue() = 0;
    virtual bool IsTerminate() = 0;
    Machine() : SimObj() {}
    ~Machine() override = default;
    void SetMachineExecuting(bool enable) { executingTask = enable; }
    void LoggerRecordTaskStart(std::string name, std::string hint = "");
    void LoggerRecordTaskEnd();
    void LoggerRecordPipe(std::string name, size_t pipeId);
    void LoggerRecordTileOp(std::string name, size_t pipeId, size_t sTime, size_t eTime);
    virtual void SetQueueCounter(); // Called after the sim pointer is initialized
    void SubmitTask(TaskPack task, uint64_t extraDelay = 0);
    void ResponseData(CachePacket pkt, uint64_t extraDelay = 0);
};
} // namespace CostModel
