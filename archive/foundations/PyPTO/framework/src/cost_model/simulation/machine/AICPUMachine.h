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
 * \file AICPUMachine.h
 * \brief
 */

#pragma once

#include <map>
#include <vector>
#include <random>

#include "cost_model/simulation/base/Machine.h"
#include "cost_model/simulation/machine/DeviceMachine.h"
#include "cost_model/simulation/config/AICPUConfig.h"
#include "cost_model/simulation/statistics/AICPUStats.h"
#include "cost_model/simulation/statistics/TraceLogger.h"

namespace CostModel {

class AICPUThreadState {
public:
    explicit AICPUThreadState(int numThreads)
    {
        batchProcessing.resize(numThreads, false);
        threadDone.resize(numThreads, false);
        completionDone.resize(numThreads, false);
        schedulerDone.resize(numThreads, false);
        currentCompletionEnd.resize(numThreads, 0);
        currentSchedulerEnd.resize(numThreads, 0);
    }

    std::vector<bool> batchProcessing;
    std::vector<bool> threadDone;
    std::vector<bool> completionDone;
    std::vector<bool> schedulerDone;
    std::vector<uint64_t> currentCompletionEnd;
    std::vector<uint64_t> currentSchedulerEnd;
};

class AICPUMachine : public Machine {
public:
    AICPUMachine();
    explicit AICPUMachine(MachineType type);
    uint64_t threadsNum;

    std::shared_ptr<DeviceMachine> top = nullptr;
    AICoreWorkLoadStatus wlStatus; // For dispatch tasks load balace in SMT mode

    void Step() override;
    void Xfer() override;
    void Build() override;
    void Reset() override;
    std::shared_ptr<SimSys> GetSim() override;
    void Report() override;
    void InitQueueDelay() override;
    void StepQueue() override;
    bool IsTerminate() override;

    void RunAtBegin();
    void RunAtEnd();
    void PollingMachine();
    void StatSubmit(std::shared_ptr<Machine> submachine);
    void DispatchPacket();
    void RecordDependency(std::shared_ptr<Task> task);
    void ResolveDependence(const std::shared_ptr<CoreMachine>& core, uint64_t threadId,
                           std::vector<uint64_t>& threadCompletionCycles);
    void WakeupSuccessors(uint64_t threadId, uint64_t& resCycles, std::vector<uint64_t> successors,
                          std::vector<uint64_t>& threadCompletionCycles);
    void CheckDeadlock();

    Task taskInfo;
    std::unique_ptr<AICPUThreadState> threadState;
    MachineType coreMachineType = MachineType::UNKNOWN;
    std::shared_ptr<AICPUStats> stats = nullptr;
    AICPUConfig config;
    SimQueue<uint64_t> localReadyQueues;

    uint64_t pollingTimeAxe = INT_MAX;
    uint64_t dispatchTimeAxe = INT_MAX;
    uint64_t resolveTimeAxe = INT_MAX;
    uint64_t recordCycles = INT_MAX;
    uint64_t globalDelay = 0;
    bool isTerminate = false;

    void UpdatePollingStates(std::vector<bool>& threadGroupActive);
    void UpdateDispatchStates(std::vector<bool>& threadGroupActive, std::vector<uint64_t>& currentCycle);
    uint64_t GetQueueNextCycles();
    void StatTaskType(const MachineType& type, uint64_t& threadId);
    void SendTask(uint64_t taskId, std::shared_ptr<Machine> subMachine, uint64_t delay, uint64_t threadId);
    void DispatchTasksForThread(uint64_t threadId, std::vector<uint64_t>& threadSchedulerCycles,
                                std::vector<uint64_t>& currentCycle);
    void DispatchTasksInReplayMode(uint64_t threadId, std::vector<uint64_t>& threadSchedulerCycles,
                                   std::vector<uint64_t>& currentCycle, uint64_t delayCycle);
    void DispatchTasksInNormalMode(uint64_t threadId, std::vector<uint64_t>& threadSchedulerCycles,
                                   std::vector<uint64_t>& currentCycle, uint64_t delayCycle);

    void DispatchHUBTask();
    void DispatchHUBTaskInReplay();
    void LogHUBTask(std::shared_ptr<Task> task, uint64_t cycle);
    void LoggerDispatch(uint64_t taskId, uint64_t threadId, uint64_t sCycle, uint64_t eCycle);

    uint64_t GetTaskLoad(MachineType type);
};

} // namespace CostModel
