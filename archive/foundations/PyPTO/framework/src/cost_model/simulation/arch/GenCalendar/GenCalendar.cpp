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
 * \file GenCalendar.cpp
 * \brief
 */

#include "cost_model/simulation/arch/GenCalendar/GenCalendar.h"

#include <fstream>
#include "tilefwk/pypto_fwk_log.h"

namespace CostModel {
void GenCalendar::InitTaskTopoInfo(TaskMap& taskMap)
{
    for (auto& task : taskMap) {
        taskTopoInfo[task.first] = CalendarEntry(task.second);
    }
}

void GenCalendar::InitAICore(uint64_t machineId) { machineStatus[machineId] = CoreInfoStatus(machineId); }

void GenCalendar::LogTaskComplete(uint64_t taskId, uint64_t machineId, uint64_t sTime, uint64_t eTime)
{
    CoreInfoStatus& status = machineStatus[machineId];
    taskTopoInfo[taskId].exeMachineId = machineId;
    taskTopoInfo[taskId].startTime = sTime;
    taskTopoInfo[taskId].endTime = eTime;
    uint64_t cmpSeq = ++status.machineCompleteSeq;
    uint64_t totalCmpSeq = ++totalCompleteSeq;
    taskTopoInfo[taskId].localCompleteSeq = cmpSeq;
    taskTopoInfo[taskId].totalCompleteSeq = totalCmpSeq;
    totalCompleteTasks[totalCmpSeq] = taskId;
    status.completedTasks.push_back(taskId);
    StatWatiTime(taskId, machineId);
}

void GenCalendar::StatWatiTime(uint64_t taskId, uint64_t machineId)
{
    CoreInfoStatus& status = machineStatus[machineId];
    auto& taskEntry = taskTopoInfo[taskId];
    for (auto& srcTaskId : taskEntry.task->predecessors) {
        auto& srcEntry = taskTopoInfo[srcTaskId];
        taskEntry.srcRdyTime = std::max(taskEntry.srcRdyTime, srcEntry.endTime);
    }
    uint64_t maxReadyTime = std::max(taskEntry.srcRdyTime, status.lastTaskEndTime); // predecessor ready and core ready
    uint64_t taskWaitScheduleTime = taskEntry.startTime - maxReadyTime;
    status.coreWaitScheduleTime += taskWaitScheduleTime;

    status.totalWaitTime += taskTopoInfo[taskId].startTime - status.lastTaskEndTime;
    status.lastTaskEndTime = taskTopoInfo[taskId].endTime;
    totalCycles = std::max(totalCycles, taskTopoInfo[taskId].endTime);
}

void GenCalendar::AllocIncCounter()
{
    // Alloc counter for each core.
    for (auto& task : taskTopoInfo) {
        task.second.incCounterId = GetMachineSeq(task.second.exeMachineId);
        for (auto& status : machineStatus) {
            status.second.obtainedCountersVals[task.second.incCounterId] = 0;
        }
    }
}

void GenCalendar::GenTaskDependency(uint64_t taskId)
{
    auto& entry = taskTopoInfo[taskId];
    if (entry.task->predecessors.empty()) {
        return;
    }
    CoreInfoStatus& status = machineStatus[entry.exeMachineId];
    std::map<uint64_t, uint64_t> counterToReqMap;    // Key: counterId, value: expected value.
    std::map<uint64_t, uint64_t> counterToTaskIdMap; // Key: counterId, value: expected value.

    for (auto& preTaskId : entry.task->predecessors) {
        auto& preTask = taskTopoInfo[preTaskId];
        // Current Task and Predecessor Task are in the same machine, no need to wait.
        if (preTask.exeMachineId == entry.exeMachineId) {
            entry.redundantSrcTaskIds.push_back(preTaskId);
            status.totalRedundantDep++;
            status.sameCoreRedundantDep++;
            continue;
        }

        // The obtained counter value is greater than the required value.
        uint64_t expectedCounterId = preTask.incCounterId;
        uint64_t expectedValue = preTask.localCompleteSeq;
        if (status.obtainedCountersVals[expectedCounterId] >= expectedValue) {
            entry.redundantSrcTaskIds.push_back(preTaskId);
            status.totalRedundantDep++;
            status.obtainedCntRedundantDep++;
            continue;
        }

        // The same task sends only one request to the counter(max expected value).
        if (counterToReqMap[expectedCounterId] > expectedValue) {
            entry.redundantSrcTaskIds.push_back(preTaskId);
            status.totalRedundantDep++;
            status.obtainedCntRedundantDep++;
            continue;
        } else if (counterToReqMap[expectedCounterId] != 0) {
            entry.redundantSrcTaskIds.push_back(counterToTaskIdMap[expectedCounterId]);
            status.totalRedundantDep++;
            status.obtainedCntRedundantDep++;
        }
        counterToReqMap[expectedCounterId] = expectedValue;
        counterToTaskIdMap[expectedCounterId] = preTaskId;
    }

    for (auto& req : counterToReqMap) {
        status.obtainedCountersVals[req.first] = req.second;
        entry.waitSrcTaskIds.push_back(counterToTaskIdMap[req.first]);
        status.waitRespCounter++;
    }
    entry.waitSCBVal = status.waitRespCounter;
}

void GenCalendar::GenCounterDepencency()
{
    for (auto& status : machineStatus) {
        for (auto& taskId : status.second.completedTasks) {
            GenTaskDependency(taskId);
        }
    }
}

struct SetHash {
    std::size_t shift1 = 6;
    std::size_t shift2 = 2;
    std::size_t operator()(const std::set<uint64_t>& s) const
    {
        std::size_t seed = 0;
        for (uint64_t val : s) {
            // hash_combine pattern
            seed ^= std::hash<uint64_t>()(val) + 0x9e3779b9 + (seed << shift1) + (seed >> shift2);
        }
        return seed;
    }
};

void GenCalendar::RemoveBarrierCounter()
{
    // WaitSrcId to task map.
    using WaitSrcTaskIdT = std::set<uint64_t>;
    using WaitSrcTaskIdToTaskIdMap = std::unordered_map<WaitSrcTaskIdT, std::vector<uint64_t>, SetHash>;

    // First sort all waiting tasks.
    WaitSrcTaskIdToTaskIdMap waitToTaskMap;
    for (auto& task : taskTopoInfo) {
        WaitSrcTaskIdT x(task.second.waitSrcTaskIds.begin(), task.second.waitSrcTaskIds.end());
        if (x.empty()) {
            continue;
        }
        waitToTaskMap[x].push_back(task.first);
    }

    // Only optimize if we can save more than 100 send_wait.
    uint64_t barrierCounterId = 100;
    uint64_t maxCounterId = 128;
    for (auto& x : waitToTaskMap) {
        for (auto w : x.first) {
            SIMULATION_LOGI("%lu ", w);
        }
        SIMULATION_LOGI("\n -- > ");
        for (auto w : x.second) {
            SIMULATION_LOGI("%lu ", w);
        }
        SIMULATION_LOGI("\n");
        auto total_send_wait = x.first.size() * x.second.size();
        auto total_wait_task = x.second.size();
        auto saved_send_wait = total_send_wait - total_wait_task;
        if (saved_send_wait < barrierCounterId) {
            // This is not our case.
            continue;
        }
        SIMULATION_LOGI("%zu !!!!\n", saved_send_wait);
        if (barrierCounterId == maxCounterId) {
            // I have used all my barrier counters.
            break;
        }
        // Make sure the producer inc the barrier counter.
        for (auto& src_task_id : x.first) {
            taskTopoInfo[src_task_id].barrierCounterIds.push_back(barrierCounterId);
        }
        // Make all the consumer wait for the barrier counter.
        for (auto& dst_task_id : x.second) {
            auto& entry = taskTopoInfo[dst_task_id];
            entry.waitBarrierCounterIds.emplace_back(barrierCounterId, x.first.size());
            // Clear the original
            entry.redundantSrcTaskIds.insert(entry.redundantSrcTaskIds.end(), x.first.begin(), x.first.end());
            entry.waitSrcTaskIds.clear();
        }

        barrierCounterId++;
    }

    // I need to fix the waitScbValue.
    for (auto& status : machineStatus) {
        auto waitValue = 0;
        for (auto& taskId : status.second.completedTasks) {
            auto& entry = taskTopoInfo[taskId];
            waitValue += entry.waitSrcTaskIds.size() + entry.waitBarrierCounterIds.size();
            entry.waitSCBVal = waitValue;
        }
    }
}

void GenCalendar::PrintTask(CalendarEntry& entry, std::ofstream& os)
{
    if (!entry.waitSrcTaskIds.empty() || !entry.waitBarrierCounterIds.empty()) {
        for (auto& srcTaskId : entry.waitSrcTaskIds) {
            auto& srcEntry = taskTopoInfo[srcTaskId];
            os << tab << "send_wait(cnt_" << std::dec << srcEntry.incCounterId;
            os << ", 0x" << std::hex << srcEntry.localCompleteSeq;
            os << ", scb_" << std::dec << entry.scbId << ", 0x" << std::hex << autoIncTag << ");";
            os << " // " << srcEntry.GetInfoLabel() << std::endl;
        }
        for (auto& barrier : entry.waitBarrierCounterIds) {
            os << tab << "send_wait(cnt_" << std::dec << barrier.first;
            os << ", 0x" << std::hex << barrier.second;
            os << ", scb_" << std::dec << entry.scbId << ", 0x" << std::hex << autoIncTag << ");";
            os << " // Barrier " << std::endl;
        }
        os << tab << "wait_spr(scb_" << std::dec << entry.scbId << ", 0x" << std::hex << entry.waitSCBVal << ");";
        os << std::endl;
    }
    os << tab << entry.task->functionName << " // [taskId:" << std::dec << entry.task->taskId << "]";

    if (!entry.redundantSrcTaskIds.empty()) {
        os << " Others Src:";
        for (auto& redundantId : entry.redundantSrcTaskIds) {
            os << taskTopoInfo[redundantId].GetInfoLabel() << ", ";
        }
    }
    os << std::endl;
    os << tab << "inc_cnt(cnt_" << std::dec << entry.incCounterId << ");" << std::endl;
    for (auto barrierCounter : entry.barrierCounterIds) {
        os << tab << "inc_cnt(cnt_" << std::dec << barrierCounter << ");" << std::endl;
    }
}

void GenCalendar::PrintStat(std::ofstream& os)
{
    os << "/*" << std::endl;
    os << "Total Tasks:" << std::dec << totalCompleteTasks.size() << std::endl;
    for (auto& status : machineStatus) {
        auto machineType = static_cast<MachineType>(GetMachineType(status.second.machineId));
        auto machineIdx = GetMachineSeq(status.second.machineId);
        os << MachineName(machineType) << "_Core_" << std::dec << machineIdx << " TasksNum: ";
        os << std::dec << status.second.completedTasks.size() << std::endl;
        os << "    Sent Counter Req Num:" << std::dec << status.second.waitRespCounter << std::endl;
        os << "    Omitted Req Num: " << status.second.totalRedundantDep;
        os << " same core depence: " << status.second.sameCoreRedundantDep;
        os << " + obtained counter dep: " << status.second.obtainedCntRedundantDep << std::endl;
        os << "    Core Last Task End Cycles: " << std::dec << status.second.lastTaskEndTime << std::endl;
        os << "    Core Total Wait Cycles: " << std::dec << status.second.totalWaitTime << std::endl;
        os << "    Core Wait Schedule Cycles: " << std::dec << status.second.coreWaitScheduleTime << std::endl;
        status.second.coreWaitPredecessorTime = status.second.totalWaitTime - status.second.coreWaitScheduleTime;
        os << "    Core Wait Predecessor Cycles: " << std::dec << status.second.coreWaitPredecessorTime << std::endl;
    }
    os << "*/" << std::endl;
}

void GenCalendar::OutputCalendar(std::ofstream& os)
{
    for (auto& status : machineStatus) {
        auto machineType = static_cast<MachineType>(GetMachineType(status.first));
        auto machineIdx = GetMachineSeq(status.first);
        os << "void " << MachineName(machineType) << "_Core_" << machineIdx << "_main()" << std::endl;
        os << "{" << std::endl;
        for (auto& taskId : status.second.completedTasks) {
            PrintTask(taskTopoInfo[taskId], os);
        }
        os << "}\n" << std::endl;
    }
}

void GenCalendar::GenCalendarCpp(std::string& path)
{
    AllocIncCounter();
    GenCounterDepencency();
    RemoveBarrierCounter();

    std::ofstream os(path);
    PrintStat(os);
    OutputCalendar(os);
    os.close();
}

std::string CalendarEntry::GetInfoLabel()
{
    std::stringstream oss;
    oss << "[srcTask:" << task->taskId;
    oss << ", core:" << GetMachineSeq(exeMachineId) << "]";
    return oss.str();
}
} // namespace CostModel
