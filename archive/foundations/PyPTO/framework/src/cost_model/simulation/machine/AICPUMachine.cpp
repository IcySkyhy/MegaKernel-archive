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
 * \file AICPUMachine.cpp
 * \brief
 */

#include "cost_model/simulation/machine/AICPUMachine.h"

#include <queue>
#include <mutex>

#include "cost_model/simulation/base/ModelTop.h"
#include "cost_model/simulation/statistics/DeviceStats.h"
#include "tilefwk/pypto_fwk_log.h"

namespace CostModel {
void AICPUMachine::Step()
{
    if (top->taskMap.empty()) {
        return;
    }
    RunAtBegin();
    PollingMachine();
    DispatchPacket();
    RunAtEnd();
}

void AICPUMachine::RunAtBegin()
{
    nextCycles = GetSim()->GetCycles();

    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (!threadState->batchProcessing[threadId]) {
            threadState->batchProcessing[threadId] = true;
            threadState->completionDone[threadId] = false;
            threadState->schedulerDone[threadId] = false;
        }
    }
}

void AICPUMachine::RunAtEnd()
{
    if (!top->taskMap.empty()) {
        CheckDeadlock();
    }

    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (threadState->batchProcessing[threadId] &&
            GetSim()->GetCycles() >= threadState->currentSchedulerEnd[threadId]) {
            threadState->batchProcessing[threadId] = false;
        }
    }
    needTerminate = IsTerminate();
}

std::shared_ptr<SimSys> AICPUMachine::GetSim() { return sim; }

void AICPUMachine::Build()
{
    config.OverrideDefaultConfig(&sim->cfgs);
    threadsNum = config.threadsNum;
    Reset();
    stats = std::make_shared<AICPUStats>(GetSim()->GetReporter());
    LoggerRecordTaskStart("AICPU Init");
    GetSim()->AddCycles();
    LoggerRecordTaskEnd();
    GetSim()->AddCycles();
    InitQueueDelay();
    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        std::string threadName = "AICPU_Thread_" + std::to_string(threadId);
        GetSim()->GetLogger()->SetThreadName(threadName, machineId, threadId + reversedTidNum);
    }
}

AICPUMachine::AICPUMachine() { machineType = MachineType::CPU; }

AICPUMachine::AICPUMachine(CostModel::MachineType type) : AICPUMachine() { machineType = type; }

void AICPUMachine::Reset()
{
    threadState = std::make_unique<AICPUThreadState>(threadsNum);
    pollingTimeAxe = INT_MAX;
    dispatchTimeAxe = INT_MAX;
    resolveTimeAxe = INT_MAX;
    recordCycles = INT_MAX;
}

void AICPUMachine::Xfer()
{
    StepQueue();
    lastCycles = GetSim()->GetCycles();
    nextCycles = INT_MAX;
    nextCycles = std::min(nextCycles, GetQueueNextCycles());
    SIMULATION_LOGI("[Cycle: %lu][AICPUMachine: %lu][Xfer] %lu", GetSim()->GetCycles(), machineId, nextCycles);
    GetSim()->UpdateNextCycles(nextCycles);
}

void AICPUMachine::Report()
{
    int machineSeq = GetMachineSeq(machineId);
    std::string name = std::to_string(machineSeq);
    stats->Report(name);
}

void AICPUMachine::InitQueueDelay()
{
    completionQueue.SetWriteDelay(0);
    completionQueue.SetReadDelay(0);
    outcastReferenceQueue.SetWriteDelay(0);
    outcastReferenceQueue.SetReadDelay(0);
    incastReferenceQueue.SetWriteDelay(0);
    incastReferenceQueue.SetReadDelay(0);
    releaseQueue.SetWriteDelay(0);
    releaseQueue.SetReadDelay(0);
    cacheRespQueue.SetWriteDelay(0);
    cacheRespQueue.SetReadDelay(0);
}

void AICPUMachine::StepQueue()
{
    uint64_t intervalCycles = GetSim()->GetCycles() - lastCycles;
    completionQueue.UpdateIntervalCycles(intervalCycles);
    outcastReferenceQueue.UpdateIntervalCycles(intervalCycles);
    incastReferenceQueue.UpdateIntervalCycles(intervalCycles);
    releaseQueue.UpdateIntervalCycles(intervalCycles);
    cacheRespQueue.UpdateIntervalCycles(intervalCycles);
    localReadyQueues.UpdateIntervalCycles(intervalCycles);
    completionQueue.Step();
    outcastReferenceQueue.Step();
    incastReferenceQueue.Step();
    releaseQueue.Step();
    cacheRespQueue.Step();
    localReadyQueues.Step();
}

uint64_t AICPUMachine::GetQueueNextCycles()
{
    uint64_t res = INT_MAX;
    uint64_t gCycles = GetSim()->GetCycles();
    res = std::min(res, gCycles + completionQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + outcastReferenceQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + incastReferenceQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + releaseQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + cacheRespQueue.GetMinWaitCycles());
    return res;
}

bool AICPUMachine::IsTerminate()
{
    for (auto& readyQ : top->readyQueues) {
        if (!readyQ.second.empty()) {
            return false;
        }
    }
    return (top->taskMap.empty() && completionQueue.IsTerminate() && outcastReferenceQueue.IsTerminate() &&
            incastReferenceQueue.IsTerminate() && releaseQueue.IsTerminate() && cacheRespQueue.IsTerminate());
}

void AICPUMachine::UpdatePollingStates(std::vector<bool>& threadGroupActive)
{
    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (GetSim()->GetCycles() < threadState->currentCompletionEnd[threadId]) {
            SIMULATION_LOGI("[Cycle: %lu][AICPUMachine: %lu][updatepolling] %lu", GetSim()->GetCycles(), machineId,
                            pollingTimeAxe);
            GetSim()->UpdateNextCycles(std::min<uint64_t>(INT_MAX, pollingTimeAxe));
        } else {
            pollingTimeAxe = INT_MAX;
            SIMULATION_LOGI("[Cycle: %lu][AICPUMachine: %lu][updatepolling] %lu", GetSim()->GetCycles(), machineId,
                            pollingTimeAxe);

            GetSim()->UpdateNextCycles(std::min<uint64_t>(INT_MAX, pollingTimeAxe));
        }
        // 检查submission和scheduler状态
        bool threadBusy = (GetSim()->GetCycles() < threadState->currentSchedulerEnd[threadId]) ||
                          (GetSim()->GetCycles() <= threadState->currentCompletionEnd[threadId]);
        bool canProcess = !threadBusy && !top->taskMap.empty();
        if (canProcess && !threadState->completionDone[threadId]) {
            threadGroupActive[threadId] = true;
            threadState->completionDone[threadId] = true;
        }
    }
}

void AICPUMachine::UpdateDispatchStates(std::vector<bool>& threadGroupActive, std::vector<uint64_t>& currentCycle)
{
    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (GetSim()->GetCycles() < threadState->currentSchedulerEnd[threadId]) {
            SIMULATION_LOGI("[Cycle: %lu][AICPUMachine: %lu][updatedispatch] %lu", GetSim()->GetCycles(), machineId,
                            dispatchTimeAxe);
            GetSim()->UpdateNextCycles(std::min<uint64_t>(INT_MAX, dispatchTimeAxe));
        } else {
            dispatchTimeAxe = INT_MAX;
            SIMULATION_LOGI("[Cycle: %lu][AICPUMachine: %lu][updatedispatch] %lu", GetSim()->GetCycles(), machineId,
                            dispatchTimeAxe);
            GetSim()->UpdateNextCycles(std::min<uint64_t>(INT_MAX, dispatchTimeAxe));
        }
        currentCycle.at(threadId) = std::max<uint64_t>(
            std::max<uint64_t>(GetSim()->GetCycles(), threadState->currentCompletionEnd[threadId]),
            threadState->currentSchedulerEnd.at(threadId));
        // 检查submission和completion状态
        bool threadBusy = ((GetSim()->GetCycles() < threadState->currentCompletionEnd[threadId]) ||
                           (GetSim()->GetCycles() <= threadState->currentSchedulerEnd[threadId]));

        bool machineBusy = true;
        uint64_t startIdx = threadId * ((subMachines.size() + threadsNum - 1) / threadsNum);
        uint64_t endIdx = std::min((threadId + 1) * ((subMachines.size() + threadsNum - 1) / threadsNum),
                                   subMachines.size());

        for (uint64_t i = startIdx; i < endIdx; i++) {
            auto& submachine = subMachines[i];
            if (uint64_t(top->executingTaskMap[submachine->machineId] < submachine->maxRunningTasks)) {
                machineBusy = false;
                break;
            }
        }

        if (!threadBusy && !machineBusy && !top->taskMap.empty()) {
            if (!threadState->schedulerDone[threadId]) {
                threadGroupActive[threadId] = true;
                threadState->schedulerDone[threadId] = true;
            }
        }
    }
}

void AICPUMachine::RecordDependency(std::shared_ptr<Task> task)
{
    for (auto& pre : task->predecessors) {
        GetSim()->GetLogger()->AddFlow(pre, task->taskId);
    }
}

void AICPUMachine::ResolveDependence(const std::shared_ptr<CoreMachine>& core, uint64_t threadId,
                                     std::vector<uint64_t>& threadCompletionCycles)
{
    auto resCycles = GetSim()->GetCycles() + threadCompletionCycles[threadId];
    CompletedPacket packet;
    core->completionQueue.Dequeue(packet);
    top->executingTaskMap[core->machineId]--;

    uint64_t taskId = packet.taskId;
    uint64_t taskExeCycle = packet.cycleInfo.taskExecuteEndCycle - packet.cycleInfo.taskExecuteStartCycle;
    stats->totalTaskExecuteCycles += taskExeCycle;
    stats->minTaskExecuteCycles = std::min(stats->minTaskExecuteCycles, taskExeCycle);
    stats->maxTaskExecuteCycles = std::max(stats->maxTaskExecuteCycles, taskExeCycle);

    top->stats->totalTaskExecuteCycles += taskExeCycle;
    top->stats->maxTaskExecuteCycles = std::max(top->stats->maxTaskExecuteCycles, taskExeCycle);
    top->stats->minTaskExecuteCycles = std::min(top->stats->minTaskExecuteCycles, taskExeCycle);

    auto funcHash = top->taskMap[taskId]->functionHash;
    GetSim()->leafFunctionTime[funcHash] = taskExeCycle;
    SIMULATION_LOGI("[Cycle: %lu][AICPUMachine: %lu][AnalysisPacket] Processing completed taskId: %lu from core %lu",
                    GetSim()->GetCycles(), machineId, taskId, core->machineId);

    std::shared_ptr<Task> task;
    std::vector<uint64_t> successors;

    auto taskIt = top->taskMap.find(taskId);
    if (taskIt == top->taskMap.end()) {
        return;
    }
    task = taskIt->second;
    successors = task->successors; // 复制successors以减少持锁时间
    ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, task->status == true)
        << "[SIMULATION]: "
        << "task status is false. taskId=" << task->taskId;
    RecordDependency(task);

    // 如果没有successor，则不需要解依赖耗时
    if (!successors.empty()) {
        GetSim()->GetLogger()->AddEventBegin(
            "Resolving_" + std::to_string(taskId), machineId, threadId + reversedTidNum,
            std::max<uint64_t>(GetSim()->GetCycles(), resCycles), "Device Machine Thread: " + std::to_string(threadId));
    }
    WakeupSuccessors(threadId, resCycles, successors, threadCompletionCycles);

    top->taskMap.erase(taskId);

    resolveTimeAxe = std::min<uint64_t>(INT_MAX, resCycles);
    SIMULATION_LOGI("[Cycle: %lu][AICPUMachine: %lu][resolve] %lu", GetSim()->GetCycles(), machineId, resolveTimeAxe);
    GetSim()->UpdateNextCycles(resolveTimeAxe);

    // 如果没有successor，则不需要解依赖耗时
    if (!successors.empty()) {
        GetSim()->GetLogger()->AddEventEnd(machineId, threadId + reversedTidNum,
                                           std::max<uint64_t>(GetSim()->GetCycles(), resCycles));
    }
}

void AICPUMachine::WakeupSuccessors(uint64_t threadId, uint64_t& resCycles, std::vector<uint64_t> successors,
                                    std::vector<uint64_t>& threadCompletionCycles)
{
    // Process successors
    SIMULATION_LOGI("[Cycle: %lu][AICPUMachine][AnalysisPacket] Found %zu successors", GetSim()->GetCycles(),
                    successors.size());
    for (uint64_t successorTaskId : successors) {
        SIMULATION_LOGI("[Cycle: %lu][AICPUMachine][AnalysisPacket] Processing successor taskId: %lu",
                        GetSim()->GetCycles(), successorTaskId);

        auto& successor = top->taskMap.at(successorTaskId);
        if (successor->remainingPredecessors == 0) {
            continue;
        }
        SIMULATION_LOGI("[Cycle: %lu][AICPUMachine][AnalysisPacket] Remaining before: %d", GetSim()->GetCycles(),
                        successor->remainingPredecessors);

        successor->remainingPredecessors--;
        resCycles += config.resolveCycles;
        stats->threadResolveNum[threadId]++;
        stats->resolveNum++;
        top->stats->resolveNum++;
        threadCompletionCycles[threadId] += config.resolveCycles;
        SIMULATION_LOGI("[Cycle: %lu][AICPUMachine][AnalysisPacket] Remaining after: %d", GetSim()->GetCycles(),
                        successor->remainingPredecessors);

        if (successor->remainingPredecessors == 0) {
            SIMULATION_LOGI("[Cycle: %lu][AICPUMachine][AnalysisPacket] Successor is now READY, pushing to ready queue",
                            GetSim()->GetCycles());

            lastCycles = GetSim()->GetCycles();
            localReadyQueues.Enqueue(successorTaskId, resCycles - GetSim()->GetCycles());
        }
    }
}

void AICPUMachine::PollingMachine()
{
    std::vector<uint64_t> threadCompletionCycles(threadsNum, 0);
    std::vector<bool> threadGroupActive(threadsNum, false);

    // 更新每个线程组的状态
    UpdatePollingStates(threadGroupActive);

    // 按照所有submachine轮询
    for (auto& submachine : subMachines) {
        auto core = std::dynamic_pointer_cast<CoreMachine>(submachine);
        if (!core) {
            continue;
        }

        for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
            if (!threadGroupActive[threadId]) {
                continue;
            }

            GetSim()->GetLogger()->AddEventBegin("PollingCore_" + std::to_string(core->machineId), machineId,
                                                 threadId + reversedTidNum,
                                                 GetSim()->GetCycles() + threadCompletionCycles[threadId],
                                                 "AICPU Machine Thread: " + std::to_string(threadId));

            threadCompletionCycles[threadId] += config.completionCycles;
            stats->pollingNum++;
            stats->threadBatchNum[threadId]++;
            top->stats->pollingNum++;

            while (!core->completionQueue.Empty()) {
                ResolveDependence(core, threadId, threadCompletionCycles);
            }

            GetSim()->GetLogger()->AddEventEnd(machineId, threadId + reversedTidNum,
                                               GetSim()->GetCycles() + threadCompletionCycles[threadId]);
        }
    }

    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (!threadGroupActive[threadId]) {
            continue;
        }
        threadState->currentCompletionEnd[threadId] = GetSim()->GetCycles() + threadCompletionCycles[threadId];

        pollingTimeAxe = std::min<uint64_t>(INT_MAX, threadState->currentCompletionEnd[threadId]);
        SIMULATION_LOGI("[Cycle: %lu][AICPUMachine: %lu][polling] %lu", GetSim()->GetCycles(), machineId,
                        pollingTimeAxe);

        GetSim()->UpdateNextCycles(pollingTimeAxe);
    }
}

void AICPUMachine::StatTaskType(const MachineType& type, uint64_t& threadId)
{
    stats->totalSubmitNum++;
    top->stats->totalSubmitNum++;
    stats->threadSubmitNum[threadId]++;
    if (type == MachineType::AIC) {
        stats->cubeSubmitNum++;
        top->stats->cubeSubmitNum++;
    } else if (type == MachineType::AIV) {
        stats->vectorSubmitNum++;
        top->stats->vectorSubmitNum++;
    }
}

void AICPUMachine::SendTask(uint64_t taskId, std::shared_ptr<Machine> subMachine, uint64_t delay, uint64_t threadId)
{
    auto& task = top->taskMap[taskId];
    TaskPack packet;
    packet.taskId = taskId;
    packet.task.taskPtr = task;
    packet.task.coreMachineType = subMachine->machineType;
    packet.task.functionHash = top->taskMap[taskId]->functionHash;
    packet.cycleInfo.taskGenCycle = GetSim()->GetCycles();
    if (!task->status) {
        task->status = true;
        top->executingTaskMap[subMachine->machineId]++;
        subMachine->SubmitTask(packet, delay);
        SIMULATION_LOGI("[Cycle: %lu][AICPU][DispatchPacket] submit task %lu to Machine %lu", GetSim()->GetCycles(),
                        taskId, subMachine->machineId);

        StatTaskType(subMachine->machineType, threadId);
    }
}

uint64_t AICPUMachine::GetTaskLoad(MachineType type)
{
    uint64_t minTaskNum = UINT64_MAX;
    if (type == MachineType::AIC) {
        for (const auto& pair : top->executingTaskMap) {
            if (GetMachineType(pair.first) == static_cast<int>(CostModel::MachineType::AIC)) {
                minTaskNum = std::min(minTaskNum, pair.second);
            }
        }
    } else if (type == MachineType::AIV) {
        for (const auto& pair : top->executingTaskMap) {
            if (GetMachineType(pair.first) == static_cast<int>(CostModel::MachineType::AIV)) {
                minTaskNum = std::min(minTaskNum, pair.second);
            }
        }
    }

    if (minTaskNum == UINT64_MAX) {
        minTaskNum = 0;
    }

    return minTaskNum;
}

void AICPUMachine::DispatchTasksInNormalMode(uint64_t threadId, std::vector<uint64_t>& threadSchedulerCycles,
                                             std::vector<uint64_t>& currentCycle, uint64_t delayCycle)
{
    for (size_t level = 0; level < wlStatus.maxLevel; level++) {
        for (size_t group = 0; group < wlStatus.groups; group++) {
            size_t subMachineIndex = wlStatus.smtGroupIndexs[group][level];
            auto& submachine = subMachines[subMachineIndex];
            // No task exists in the readQueue of the current machine type.
            if (top->readyQueues[submachine->machineType].empty()) {
                continue;
            }

            // Submit Task To submachine.
            uint64_t minTaskNum = GetTaskLoad(submachine->machineType);
            if (top->executingTaskMap[submachine->machineId] < submachine->maxRunningTasks &&
                top->executingTaskMap[submachine->machineId] <= minTaskNum) {
                uint64_t taskId = top->PopReadyQueue(submachine->machineType);
                GetSim()->GetLogger()->AddEventBegin("Dispatch_Task_" + std::to_string(taskId), machineId,
                                                     threadId + reversedTidNum,
                                                     currentCycle.at(threadId) + threadSchedulerCycles[threadId],
                                                     "Dispatching Machine Thread: " + std::to_string(threadId));
                threadSchedulerCycles[threadId] += config.schedulerCycles;
                delayCycle += config.schedulerCycles;
                SendTask(taskId, submachine, delayCycle, threadId);
                GetSim()->GetLogger()->AddEventEnd(machineId, threadId + reversedTidNum,
                                                   currentCycle.at(threadId) + threadSchedulerCycles[threadId]);
            }
        }
    }
}

void AICPUMachine::DispatchTasksInReplayMode(uint64_t threadId, std::vector<uint64_t>& threadSchedulerCycles,
                                             std::vector<uint64_t>& currentCycle, uint64_t delayCycle)
{
    for (auto& subMachine : subMachines) {
        auto& replayInfoQ = top->replayTasksInfoMap[subMachine->machineId];
        // Has no task to execute
        if (replayInfoQ.empty()) {
            continue;
        }
        // CoreMachine idel
        if (top->executingTaskMap[subMachine->machineId] >= subMachine->maxRunningTasks) {
            continue;
        }
        // Task Ready
        auto& replayTask = replayInfoQ.front();
        if (replayTask.seqNo <= top->currentSeq && top->IsReady(replayTask.taskId)) {
            LoggerDispatch(replayTask.taskId, threadId, currentCycle.at(threadId) + threadSchedulerCycles[threadId],
                           currentCycle.at(threadId) + threadSchedulerCycles[threadId] + config.schedulerCycles);

            threadSchedulerCycles[threadId] += config.schedulerCycles;
            delayCycle += config.schedulerCycles;
            top->ScaleTaskExecuteTime(replayTask);
            SendTask(replayTask.taskId, subMachine, delayCycle, threadId);
            top->EraseReadySet(replayTask.taskId);
            replayInfoQ.pop_front();
        }
    }
}

void AICPUMachine::DispatchTasksForThread(uint64_t threadId, std::vector<uint64_t>& threadSchedulerCycles,
                                          std::vector<uint64_t>& currentCycle)
{
    uint64_t delayCycle = 0;
    if (top->config.replayEnable && !top->replayPreExecute) {
        DispatchHUBTaskInReplay();
        DispatchTasksInReplayMode(threadId, threadSchedulerCycles, currentCycle, delayCycle);
    } else {
        DispatchHUBTask();
        DispatchTasksInNormalMode(threadId, threadSchedulerCycles, currentCycle, delayCycle);
    }
}

void AICPUMachine::LogHUBTask(std::shared_ptr<Task> task, uint64_t cycle)
{
    std::string name = "Virtual:" + std::to_string(task->taskId);
    std::string hint = "Executing TaskId:" + std::to_string(task->taskId);
    size_t hubMachineId = GetSim()->GetHUBCore()->machineId;
    size_t topMachineViewPid = GetSim()->topMachineViewPid;

    GetSim()->GetLogger()->AddEventBegin(name, topMachineViewPid, hubMachineId, cycle, hint);
    GetSim()->GetLogger()->AddEventEnd(topMachineViewPid, hubMachineId, cycle + 1);

    GetSim()->GetLogger()->AddEventBegin(name, hubMachineId, coreTid, cycle, hint);
    GetSim()->GetLogger()->AddEventEnd(hubMachineId, coreTid, cycle + 1);
}

void AICPUMachine::LoggerDispatch(uint64_t taskId, uint64_t threadId, uint64_t sCycle, uint64_t eCycle)
{
    std::string name = "Dispatch_Task_:" + std::to_string(taskId);
    std::string hint = "Dispatching Machine Thread: " + std::to_string(threadId);

    GetSim()->GetLogger()->AddEventBegin(name, machineId, threadId + reversedTidNum, sCycle, hint);
    GetSim()->GetLogger()->AddEventEnd(machineId, threadId + reversedTidNum, eCycle);
}

void AICPUMachine::DispatchHUBTask()
{
    uint64_t hubStartCycle = GetSim()->GetCycles();
    while (!top->readyQueues[MachineType::HUB].empty()) {
        std::shared_ptr<Task> task = nullptr;
        std::vector<uint64_t> successors;
        uint64_t taskId = top->PopReadyQueue(MachineType::HUB);
        auto taskIt = top->taskMap.find(taskId);
        if (taskIt == top->taskMap.end()) {
            continue;
        }
        SIMULATION_LOGI("[Cycle: %lu][AICPU][DispatchHUBTask] process Hub task %lu", GetSim()->GetCycles(), taskId);
        task = taskIt->second;
        successors = task->successors;
        LogHUBTask(task, hubStartCycle);
        hubStartCycle++; // start
        hubStartCycle++; // end
        RecordDependency(task);
        uint64_t resCycles = GetSim()->GetCycles();
        std::vector<uint64_t> threadCompletionCycles = {0};
        WakeupSuccessors(0, resCycles, successors, threadCompletionCycles);
        top->taskMap.erase(taskId);
    }
}

void AICPUMachine::DispatchHUBTaskInReplay()
{
    if (!top->config.replayEnable) {
        return;
    }
    uint64_t hubStartCycle = GetSim()->GetCycles();

    size_t hubMachineId = GetSim()->GetHUBCore()->machineId;
    auto& replayInfoQ = top->replayTasksInfoMap[hubMachineId];
    // Has task to execute
    while (!replayInfoQ.empty()) {
        auto& replayTask = replayInfoQ.front();
        auto taskIt = top->taskMap.find(replayTask.taskId);
        if (taskIt == top->taskMap.end()) {
            replayInfoQ.pop_front();
            continue;
        }
        if (replayTask.seqNo <= top->currentSeq && top->IsReady(replayTask.taskId)) {
            std::shared_ptr<Task> task = nullptr;
            std::vector<uint64_t> successors;
            task = taskIt->second;
            successors = task->successors;

            LogHUBTask(task, hubStartCycle);
            hubStartCycle++;
            hubStartCycle++;
            RecordDependency(task);
            uint64_t resCycles = GetSim()->GetCycles();
            std::vector<uint64_t> threadCompletionCycles = {0};
            WakeupSuccessors(0, resCycles, successors, threadCompletionCycles);
            top->taskMap.erase(replayTask.taskId);
            top->EraseReadySet(replayTask.taskId);
            replayInfoQ.pop_front();
        } else {
            break;
        }
    }
}

void AICPUMachine::DispatchPacket()
{
    std::vector<uint64_t> threadSchedulerCycles(threadsNum, 0);
    std::vector<uint64_t> currentCycle(threadsNum, 0);
    std::vector<bool> threadGroupActive(threadsNum, false);

    bool readyQueueEmpty = true;
    for (auto& readyQ : top->readyQueues) {
        if (!readyQ.second.empty()) {
            readyQueueEmpty = false;
            break;
        }
    }
    if (top->config.replayEnable && !top->readySet.empty()) {
        readyQueueEmpty = false;
    }
    // 更新每个线程组的状态
    UpdateDispatchStates(threadGroupActive, currentCycle);
    // 按照所有submachine轮询
    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (!threadGroupActive[threadId] || readyQueueEmpty) {
            continue;
        }
        DispatchTasksForThread(threadId, threadSchedulerCycles, currentCycle);
    }

    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (!threadGroupActive[threadId]) {
            continue;
        }
        threadState->currentSchedulerEnd[threadId] = std::max<uint64_t>(GetSim()->GetCycles(),
                                                                        currentCycle.at(threadId)) +
                                                     threadSchedulerCycles[threadId];

        dispatchTimeAxe = std::min<uint64_t>(INT_MAX, threadState->currentSchedulerEnd[threadId]);
        if (threadSchedulerCycles[threadId] != 0) {
            SIMULATION_LOGI("[Cycle: %lu][AICPUMachine: %lu][dispatch] %lu", GetSim()->GetCycles(), machineId,
                            dispatchTimeAxe);
            GetSim()->UpdateNextCycles(dispatchTimeAxe);
        } else {
            SIMULATION_LOGI("[Cycle: %lu][AICPUMachine: %lu][dispatch] %lu", GetSim()->GetCycles(), machineId,
                            GetSim()->GetCycles() + 1);
            GetSim()->UpdateNextCycles(GetSim()->GetCycles() + 1);
        }
    }
}

void AICPUMachine::CheckDeadlock()
{
    // shutdown check
    bool execute = false;

    for (auto& submachine : subMachines) {
        if (submachine && IsCoreMachine(submachine->machineType)) {
            if (!submachine->completionQueue.Empty()) {
                execute = true;
                break;
            }
        }
    }

    if (!top->taskMap.empty() && dispatchTimeAxe == INT_MAX && pollingTimeAxe == INT_MAX && execute) {
        GetSim()->UpdateNextCycles(GetSim()->GetCycles() + 1);
    }

    for (const auto& pair : top->readyQueues) {
        if (!pair.second.empty()) {
            execute = true;
        }
    }

    // 检查每个线程的状态
    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (threadState->batchProcessing[threadId] || !threadState->completionDone[threadId] ||
            !threadState->schedulerDone[threadId]) {
            threadState->threadDone[threadId] = true;
        }
    }

    execute |= std::any_of(threadState->threadDone.begin(), threadState->threadDone.end(),
                           [](bool done) { return !done; });

    SetMachineExecuting(execute);
}
} // namespace CostModel
