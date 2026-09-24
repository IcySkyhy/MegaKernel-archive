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
 * \file GenCalendar.h
 * \brief
 */

#pragma once

#include <string>
#include <map>

#include "cost_model/simulation/common/ISA.h"

namespace CostModel {
struct CalendarEntry {
    std::shared_ptr<Task> task;
    uint64_t exeMachineId = 0;
    uint64_t localCompleteSeq = 0;
    uint64_t totalCompleteSeq = 0;
    uint64_t startTime = 0;
    uint64_t endTime = 0;
    uint64_t srcRdyTime = 0;

    uint64_t incCounterId = -1;
    std::vector<uint64_t> barrierCounterIds;
    uint64_t scbId = 4;

    std::vector<uint64_t> waitSrcTaskIds;
    std::vector<std::pair<uint64_t, uint64_t>> waitBarrierCounterIds;
    std::vector<uint64_t> redundantSrcTaskIds;
    uint64_t waitSCBVal = 0;

    explicit CalendarEntry() = default;
    explicit CalendarEntry(std::shared_ptr<Task> t) : task(t) {}
    std::string GetInfoLabel();
};

struct CoreInfoStatus {
    uint64_t machineId = 0;
    uint64_t machineCompleteSeq = 0;
    std::deque<uint64_t> completedTasks;

    std::map<uint64_t, uint64_t> obtainedCountersVals;
    uint64_t waitRespCounter = 0;
    uint64_t lastTaskEndTime = 0;

    // statistics
    uint64_t totalRedundantDep = 0;
    uint64_t sameCoreRedundantDep = 0;
    uint64_t obtainedCntRedundantDep = 0;

    uint64_t totalWaitTime = 0;
    uint64_t coreWaitScheduleTime = 0;
    uint64_t coreWaitPredecessorTime = 0;

    explicit CoreInfoStatus() = default;
    explicit CoreInfoStatus(uint64_t id) : machineId(id) {}
};

class GenCalendar {
public:
    std::map<uint64_t, CalendarEntry> taskTopoInfo;
    std::map<uint64_t, CoreInfoStatus> machineStatus;
    uint64_t totalCompleteSeq = 0;
    std::map<uint64_t, uint64_t> totalCompleteTasks;
    uint64_t totalCycles = 0;

    std::string tab = "    ";
    uint64_t offset = 14;
    uint64_t autoIncTag = 1 << offset;

    GenCalendar() = default;
    void InitTaskTopoInfo(TaskMap& taskMap);
    void InitAICore(uint64_t machineId);
    void LogTaskComplete(uint64_t taskId, uint64_t machineId, uint64_t sTime, uint64_t eTime);
    void StatWatiTime(uint64_t taskId, uint64_t machineId);

    void AllocIncCounter();
    void GenTaskDependency(uint64_t taskId);
    void GenCounterDepencency();
    void CheckRedundantDependency();
    void PrintTask(CalendarEntry& entry, std::ofstream& os);
    void PrintStat(std::ofstream& os);
    void OutputCalendar(std::ofstream& os);
    void GenCalendarCpp(std::string& path);

    void RemoveBarrierCounter();
};
} // namespace CostModel
