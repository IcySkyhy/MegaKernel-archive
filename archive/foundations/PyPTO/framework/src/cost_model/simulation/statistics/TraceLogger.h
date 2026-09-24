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
 * \file TraceLogger.h
 * \brief
 */

// Chrome trace format reference:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0

#pragma once

#include <cstdint>
#include <cstddef>

#include <map>
#include <stack>
#include <tuple>
#include <mutex>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>

#include "cost_model/simulation/common/CommonData.h"
#include "cost_model/simulation/config/TraceConfig.h"

#include <nlohmann/json.hpp>

namespace CostModel {

using Json = nlohmann::json;

struct Event {
    std::string name;
    int id = -1; // invalid -1
    std::string catagory;
    std::string phase;
    std::string bp;
    TimeStamp timestamp;
    Pid pid;
    Tid tid;
    std::string hint;

    Json ToJson();
    Json ToFlowStartJson(int flowId) const;
    Json ToFlowEndJson(int flowId) const;

    EventId GetEventID() const { return EventId{PTid{pid, tid}, id}; }

    std::string GetColor();
    int ExtraHintInfo(std::string& key);
};

enum class CounterType {
    QUEUE_PUSH = 0,
    QUEUE_POP = 1,
    CACHE_HIT = 2,
    CACHE_MISS = 3,
    COUNT_SIZE = 4,
};

struct CounterEvent {
    int id = -1; // invalid -1
    std::string catagory;
    std::string phase;
    CounterType type;
    int size = 0;
    TimeStamp timestamp;
    Pid pid;
    Tid tid;

    Json ToJson() const;

    EventId GetEventID() const { return EventId{PTid{pid, tid}, id}; }
};

struct Thread {
    std::string name;
    Pid pid;
    Tid tid;

    Json ToJson() const;
};

struct Process {
    std::string name;
    Pid pid;
    size_t coreIdx = 0;

    Json ToJson() const;
    Json ToSortIndexJson(int sortIndex) const;
};

struct Duration {
    Event start;
    Event end;
    int width = 30;
    int width2 = 20;
    void OutputContextSwitchTrace(std::ofstream& os, std::map<Pid, Process>& mProcesses,
                                  std::map<PTid, Thread>& mThreads, const uint64_t sysClockTicks);
    void OutputBeginEndTrace(std::ofstream& os, std::map<Pid, Process>& mProcesses, std::map<PTid, Thread>& mThreads,
                             const uint64_t sysClockTicks);
    Json ToJson();
};

struct Flow {
    std::string name;
    EventId from;
    EventId to;
};

struct CoreInfoLog {
    uint64_t idx;
    std::string type;

    std::vector<Json> taskLogs;
    std::map<std::string, std::vector<Json>> pipeLogs;
    explicit CoreInfoLog() {};
    explicit CoreInfoLog(uint64_t id, std::string t) : idx(id), type(t)
    {
        taskLogs.clear();
        pipeLogs.clear();
    }
};

class SimSys;
class TraceLogger {
public:
    TraceConfig config;
    int width = 30;
    int width2 = 20;
    int precision = 6;
    std::map<Pid, Process> mProcesses;
    std::map<PTid, Thread> mThreads;
    std::vector<Event> mEvents;
    std::vector<Flow> mFlows;
    std::vector<CounterEvent> mCounters;
    std::map<PTid, std::vector<CounterEvent>> mCounts;
    std::map<int, Duration> mDurations;
    std::map<int, int> mTaskIDToDurationIndex;
    std::map<Pid, std::map<int, int>> mMachineTileOpMap;
    std::map<PTid, std::stack<Event>> m_eventStacks;

    std::map<PTid, std::vector<CounterEvent>> eachMachineQueueSize;
    std::vector<CounterEvent> totalDeviceMachineQueueSize;
    std::vector<CounterEvent> totalAicpuMachineQueueSize;
    std::vector<CounterEvent> totalCoreMachineQueueSize;
    std::vector<CounterEvent> totalPipeMachineQueueSize;
    std::vector<CounterEvent> totalDeviceMachinePushpopNum;
    std::vector<CounterEvent> totalAicpuMachinePushpopNum;
    std::vector<CounterEvent> totalCoreMachinePushpopNum;
    std::vector<CounterEvent> totalPipeMachinePushpopNum;
    std::map<PTid, std::vector<CounterEvent>> eachPipeMachineWorkload;
    std::vector<CounterEvent> totalPipeMachineWorkload;
    std::vector<CounterEvent> functionCacheTotal;
    std::vector<CounterEvent> functionCacheHit;
    std::vector<CounterEvent> functionCacheMiss;

    std::map<uint64_t, CoreInfoLog> mCoreInfoLogs;
    bool coreInfoBuild = false;

    size_t topMachineViewPid = 1000;
    size_t reversedTidNum = 100; // For Queue Start tid
    size_t coreTid = 1;          // For MachineView tid
    std::shared_ptr<CostModel::SimSys> sim = nullptr;
    int mEventIdPtr = 0;
    int mCounterEventIdPtr = 0;
    int mQSizeIdPtr = 0;
    bool processDeviceReadyQueue = false;
    explicit TraceLogger() {}
    // Logger Trace
    void SetProcessName(std::string name, Pid pid, size_t coreIdx);
    void SetThreadName(std::string name, Pid pid, Tid tid);
    Event AddEventBegin(std::string name, Pid pid, Tid tid, TimeStamp timestamp, std::string hint = "");
    Event AddEventEnd(Pid pid, Tid tid, TimeStamp timestamp);
    void AddDuration(const LogData& data);
    void AddFlow(uint64_t srcTask, uint64_t dstTask);
    void AddTileOpFlow(Pid pid, uint64_t srcMagic, uint64_t dstMagic);
    void AddFlow(std::string name, EventId from, EventId to);
    void AddCounterEvent(Pid pid, Tid tid, CounterType type);
    void LogTaskInfo(Event& start, Event& end);
    void LogPipeInfo(Event& start, Event& end);
    void LogCoreInfo(Duration& duration);

    void EraseLogInfo(uint64_t startCycle);

    // Get Queue Counter based on CountEvents.
    void GetTotalMachineQueueSize(TimeStamp interval);
    void GetFunctionCacheSize(TimeStamp interval,
                              const std::pair<const PTid, std::vector<CounterEvent>>& threadCounter);
    void GetTotalFunctionCacheSize(TimeStamp interval);
    void GetCounters();
    void GetDeviceReadyQ();
    void OutEachMachineQueueSize(std::ofstream& os, const uint64_t sysClockTicks);
    void OutCounters(std::ofstream& os, std::vector<CounterEvent>& counterQ, std::string prefix, std::string suffix,
                     const uint64_t sysClockTicks);
    Json QSizeToJson(std::vector<CounterEvent>& counterQ);
    // Output Trace
    Json ToJson();
    void ToTrace(std::ofstream& os);
    void ToFilterTrace(std::ofstream& os, std::map<int, std::pair<std::string, std::vector<Json>>>& coreTasks);
    void ToPipeTrace(std::ofstream& os);
    void ToCalendarGlobalJson(std::ofstream& osCalendar,
                              std::map<int, std::pair<std::string, std::vector<Json>>> coreTasks);
};
} // namespace CostModel
