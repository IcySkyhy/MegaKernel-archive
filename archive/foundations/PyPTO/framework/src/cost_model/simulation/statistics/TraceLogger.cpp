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
 * \file TraceLogger.cpp
 * \brief
 */

#include "cost_model/simulation/statistics/TraceLogger.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>

#include "cost_model/simulation/base/ModelTop.h"

using namespace std;
namespace CostModel {
Json Event::ToJson()
{
    Json root;

    root["name"] = name;
    if (!catagory.empty()) {
        root["cat"] = catagory;
    }
    root["ph"] = phase;
    if (!bp.empty()) {
        root["bp"] = bp;
    }
    if (id > 0) {
        root["id"] = id;
    }
    root["ts"] = timestamp;
    root["pid"] = pid;
    root["tid"] = tid;

    if (!hint.empty()) {
        Json args;
        args["event-hint"] = hint;
        args["color"] = this->GetColor();
        root["args"] = std::move(args);
    }

    return root;
}

Json Event::ToFlowStartJson(int flowId) const
{
    Json root;
    root["cat"] = "machine-view-last-dep";
    root["id"] = flowId;
    root["name"] = "machine-view-last-dep";
    root["ph"] = "s";
    root["pid"] = pid;
    root["tid"] = tid;
    root["ts"] = timestamp - 1;
    return root;
}

Json Event::ToFlowEndJson(int flowId) const
{
    Json root;
    root["bp"] = "e";
    root["cat"] = "machine-view-last-dep";
    root["id"] = flowId;
    root["name"] = "machine-view-last-dep";
    root["ph"] = "f";
    root["pid"] = pid;
    root["tid"] = tid;
    root["ts"] = timestamp;
    return root;
}

std::string Event::GetColor()
{
    size_t pos1 = name.find('(');
    if (pos1 == std::string::npos) {
        return "";
    }

    size_t pos2 = name.find(')');
    if (pos2 == std::string::npos) {
        return "";
    }
    return name.substr(pos1 + 1, pos2 - pos1 - 1);
}

int Event::ExtraHintInfo(std::string& key)
{
    size_t pos = hint.find(key);
    pos += key.length();
    std::string numStr = hint.substr(pos);
    size_t spacePos = numStr.find(' ');
    numStr = numStr.substr(0, spacePos);
    std::stringstream ss(numStr);
    int value;
    ss >> value;
    return value;
}

Json CounterEvent::ToJson() const
{
    Json jSize;
    jSize["size"] = size;

    Json root;
    root["args"] = jSize;
    root["name"] = catagory;
    root["pid"] = pid;
    root["tid"] = tid;
    root["ph"] = "C";
    root["ts"] = timestamp;

    return root;
}

Json Thread::ToJson() const
{
    Json root;
    Json args;
    args["name"] = name;

    root["args"] = std::move(args);
    root["cat"] = "__metadata";
    root["name"] = "thread_name";
    root["ph"] = "M";
    root["pid"] = pid;
    root["tid"] = tid;

    return root;
}

Json Process::ToJson() const
{
    Json root;

    Json args;
    args["name"] = name;

    root["args"] = std::move(args);
    root["cat"] = "__metadata";
    root["name"] = "process_name";
    root["ph"] = "M";
    root["pid"] = pid;

    return root;
}

Json Process::ToSortIndexJson(int sortIndex) const
{
    Json root;

    Json args;
    args["sort_index"] = sortIndex;

    root["args"] = std::move(args);
    root["cat"] = "__metadata";
    root["name"] = "process_sort_index";
    root["ph"] = "M";
    root["pid"] = pid;

    return root;
}

void Duration::OutputContextSwitchTrace(std::ofstream& os, std::map<Pid, Process>& mProcesses,
                                        std::map<PTid, Thread>& mThreads, const uint64_t sysClockTicks)
{
    string processInfo = to_string(start.tid);
    std::string subgraphName = "SUBGRAPH";
    auto it = mThreads.find(PTid{start.pid, start.tid});
    if (it != mThreads.end()) {
        subgraphName = it->second.name;
    }
    std::ostringstream cpuId;
    int formatOffset = 3;
    cpuId << setfill('0') << setw(formatOffset) << mProcesses[start.tid].coreIdx;
    string cpuIdx = cpuId.str();
    string cpuInfo = "(   0) [" + cpuIdx + "] ....";

    string cpuMgrTile = "cpumgr-idle-" + to_string(start.pid);
    string cpuSwitchTitle = subgraphName + "-" + to_string(start.tid);

    string cpuWakeupInfo = ": sched_wakeup: comm=" + subgraphName + " pid=" + processInfo +
                           " prio=20 target_cpu=" + cpuIdx;
    string cpuSwitchInfo1 =
        ": sched_switch: prev_comm=cpumgr-idle-0 prev_pid=0 prev_prio=-2 prev_state=R+ ==> next_comm=" + subgraphName +
        " next_pid=" + processInfo + " next_prio=5";
    string cpuSwitchInfo2 = ": sched_switch: prev_comm=" + subgraphName + " prev_pid=" + processInfo +
                            " prev_prio=5 prev_state= ==> next_comm=cpumgr-idle-0 next_pid=0 next_prio=-2";
    string cpuIdleInfo = ": cpu_idle: state=0 cpu_id=" + cpuIdx;

    std::ostringstream cyc1;
    int precision = 6;
    cyc1 << std::fixed << std::setprecision(precision) << (double(start.timestamp) / sysClockTicks);
    string sCycle = cyc1.str();

    std::ostringstream cyc2;
    cyc2 << std::fixed << std::setprecision(precision) << (double(end.timestamp) / sysClockTicks);
    string eCycle = cyc2.str();

    os << setw(width) << left << cpuMgrTile << setw(width) << right << cpuInfo << setw(width2) << right << sCycle;
    os << cpuWakeupInfo << std::endl;
    os << setw(width) << left << cpuSwitchTitle << setw(width) << right << cpuInfo << setw(width2) << right << sCycle;
    os << cpuSwitchInfo1 << std::endl;

    os << setw(width) << left << cpuMgrTile << setw(width) << right << cpuInfo << setw(width2) << right << eCycle;
    os << cpuSwitchInfo2 << std::endl;
    os << setw(width) << left << cpuMgrTile << setw(width) << right << cpuInfo << setw(width2) << right << eCycle;
    os << cpuIdleInfo << std::endl;
}

void Duration::OutputBeginEndTrace(std::ofstream& os, std::map<Pid, Process>& mProcesses,
                                   std::map<PTid, Thread>& mThreads, const uint64_t sysClockTicks)
{
    string processInfo = to_string(start.pid);
    std::ostringstream cpuId;
    int formatOffset = 3;
    cpuId << setfill('0') << setw(formatOffset) << mProcesses[start.pid].coreIdx;
    string cpuIdx = cpuId.str();
    string cpuInfo = "(" + processInfo + ") [" + cpuIdx + "] ....";

    string pipeName = "";
    auto it = mThreads.find(PTid{start.pid, start.tid});
    if (it != mThreads.end()) {
        pipeName = it->second.name;
    }

    string title = pipeName + "-" + to_string(start.tid);
    string beginInfo = ": tracing_mark_write: B|" + processInfo + "|" + start.name + " " + start.hint;
    string endInfo = ": tracing_mark_write: E|" + processInfo + "|";

    std::ostringstream cyc1;
    int precision = 6;
    cyc1 << std::fixed << std::setprecision(precision) << (double(start.timestamp) / sysClockTicks);
    string sCycle = cyc1.str();

    std::ostringstream cyc2;
    cyc2 << std::fixed << std::setprecision(precision) << (double(end.timestamp) / sysClockTicks);
    string eCycle = cyc2.str();

    os << setw(width) << left << title << setw(width) << right << cpuInfo << setw(width2) << right << sCycle;
    os << beginInfo << std::endl;
    os << setw(width) << left << title << setw(width) << right << cpuInfo << setw(width2) << right << eCycle;
    os << endInfo << std::endl;
}

Json Duration::ToJson()
{
    Json root;

    if (!start.catagory.empty()) {
        root["cat"] = start.catagory;
    }
    root["ph"] = "X";
    root["id"] = start.id;
    root["name"] = start.name;
    root["pid"] = start.pid;
    root["tid"] = start.tid;
    root["ts"] = start.timestamp;
    root["dur"] = end.timestamp - start.timestamp;

    if (!start.hint.empty()) {
        Json args;
        args["event-hint"] = start.hint;
        args["color"] = start.GetColor();
        root["args"] = std::move(args);
    }

    return root;
}

void TraceLogger::SetProcessName(std::string name, CostModel::Pid pid, size_t coreIdx)
{
    mProcesses[pid] = Process{
        .name = name,
        .pid = pid,
        .coreIdx = coreIdx,
    };
    mMachineTileOpMap[pid] = std::map<int, int>();
}

void TraceLogger::SetThreadName(std::string name, CostModel::Pid pid, CostModel::Tid tid)
{
    mThreads[PTid{pid, tid}] = Thread{
        .name = name,
        .pid = pid,
        .tid = tid,
    };
}

Event TraceLogger::AddEventBegin(std::string name, CostModel::Pid pid, CostModel::Tid tid,
                                 CostModel::TimeStamp timestamp, std::string hint)
{
    mEventIdPtr++;
    auto beginEvent = Event{
        .name = name,
        .id = mEventIdPtr,
        .catagory = "event",
        .phase = "B",
        .bp = "",
        .timestamp = timestamp,
        .pid = pid,
        .tid = tid,
        .hint = hint,
    };
    mEvents.push_back(beginEvent);
    m_eventStacks[PTid{pid, tid}].push(beginEvent);
    return beginEvent;
}

Event TraceLogger::AddEventEnd(CostModel::Pid pid, CostModel::Tid tid, CostModel::TimeStamp timestamp)
{
    auto beginEvent = m_eventStacks[PTid{pid, tid}].top();
    m_eventStacks[PTid{pid, tid}].pop();

    auto endEvent = Event{
        .name = beginEvent.name,
        .id = beginEvent.id,
        .catagory = "event",
        .phase = "E",
        .bp = "",
        .timestamp = timestamp,
        .pid = pid,
        .tid = tid,
        .hint = beginEvent.hint,
    };
    mEvents.push_back(endEvent);
    mDurations.emplace(beginEvent.id, Duration{beginEvent, endEvent});
    auto machineType = GetMachineType(tid);
    if (pid == topMachineViewPid && IsCoreMachine(machineType)) {
        std::string taskKey = "TaskId:";
        auto pos = beginEvent.hint.find(taskKey);
        if (pos != std::string::npos) {
            int taskId = beginEvent.ExtraHintInfo(taskKey);
            mTaskIDToDurationIndex[taskId] = beginEvent.id;
        }
        mMachineTileOpMap[pid].clear();
    }
    return endEvent;
}

void TraceLogger::AddDuration(const LogData& data)
{
    mEventIdPtr++;
    auto beginEvent = Event{
        .name = data.name,
        .id = mEventIdPtr,
        .catagory = "event",
        .phase = "B",
        .bp = "",
        .timestamp = data.sTime,
        .pid = data.pid,
        .tid = data.tid,
        .hint = data.hint,
    };
    auto endEvent = Event{
        .name = data.name,
        .id = mEventIdPtr,
        .catagory = "event",
        .phase = "E",
        .bp = "",
        .timestamp = data.eTime,
        .pid = data.pid,
        .tid = data.tid,
        .hint = beginEvent.hint,
    };
    mEvents.push_back(beginEvent);
    mEvents.push_back(endEvent);
    mDurations.emplace(beginEvent.id, Duration{beginEvent, endEvent});
    if (data.isLogTileOp) {
        std::istringstream iss(data.name);
        int magic;
        iss >> magic;
        mMachineTileOpMap[data.pid][magic] = mEventIdPtr;
    }
}

void TraceLogger::AddFlow(uint64_t srcTask, uint64_t dstTask)
{
    EventId srcId;
    EventId dstId;
    srcId.eid = mTaskIDToDurationIndex[srcTask];
    dstId.eid = mTaskIDToDurationIndex[dstTask];
    AddFlow("flow", srcId, dstId);
}

void TraceLogger::AddTileOpFlow(Pid pid, uint64_t srcMagic, uint64_t dstMagic)
{
    if (mMachineTileOpMap.find(pid) == mMachineTileOpMap.end()) {
        return;
    }
    if (mMachineTileOpMap[pid].find(srcMagic) == mMachineTileOpMap[pid].end() ||
        mMachineTileOpMap[pid].find(dstMagic) == mMachineTileOpMap[pid].end()) {
        return;
    }
    EventId srcId;
    EventId dstId;
    srcId.eid = mMachineTileOpMap[pid][srcMagic];
    dstId.eid = mMachineTileOpMap[pid][dstMagic];
    AddFlow("flow", srcId, dstId);
}

void TraceLogger::AddFlow(std::string name, CostModel::EventId from, CostModel::EventId to)
{
    mFlows.push_back(Flow{name, from, to});
}

void TraceLogger::AddCounterEvent(CostModel::Pid pid, CostModel::Tid tid, CostModel::CounterType type)
{
    mCounterEventIdPtr++;
    auto sizeCount = CounterEvent{
        .id = mCounterEventIdPtr,
        .catagory = "count",
        .phase = "C",
        .type = type,
        .size = 0,
        .timestamp = TimeStamp(sim->GetCycles()),
        .pid = pid,
        .tid = tid,
    };
    mCounters.emplace_back(sizeCount);
    mCounts[PTid{pid, tid}].emplace_back(sizeCount);
}

void TraceLogger::EraseLogInfo(uint64_t startCycle)
{
    auto new_events_end = mEvents.begin();
    for (auto it = mEvents.begin(); it != mEvents.end(); ++it) {
        if (it->timestamp <= startCycle) {
            if (it != new_events_end) {
                *new_events_end = std::move(*it);
            }
            ++new_events_end;
        }
    }
    mEvents.erase(new_events_end, mEvents.end());

    for (auto it = mDurations.begin(); it != mDurations.end();) {
        if (it->second.start.timestamp > startCycle) {
            it = mDurations.erase(it); // map的erase是O(1)摊销时间
        } else {
            ++it;
        }
    }

    auto new_counters_end = mCounters.begin();
    for (auto it = mCounters.begin(); it != mCounters.end(); ++it) {
        if (it->timestamp <= startCycle) {
            if (it != new_counters_end) {
                *new_counters_end = std::move(*it);
            }
            ++new_counters_end;
        }
    }
    mCounters.erase(new_counters_end, mCounters.end());

    for (auto& counts : mCounts) {
        auto new_counts_end = counts.second.begin();
        for (auto it = counts.second.begin(); it != counts.second.end(); ++it) {
            if (it->timestamp <= startCycle) {
                if (it != new_counts_end) {
                    *new_counts_end = std::move(*it);
                }
                ++new_counts_end;
            }
        }
        counts.second.erase(new_counts_end, counts.second.end());
    }

    mTaskIDToDurationIndex.clear();
}

void TraceLogger::GetTotalMachineQueueSize(CostModel::TimeStamp interval)
{
    std::map<int, std::map<int, int>> machinesQueueIntervalCount;
    std::map<int, std::map<int, int>> machinesQueueTotalCount;
    std::map<int, std::map<int, int>> machinesQueuePushpopCount;
    TimeStamp lastTime = 0;
    for (auto& counter : mCounters) {
        if (!sim->IsQueue(counter.tid)) {
            continue;
        }
        int machineType = GetMachineType(counter.pid);
        int qId = counter.tid;
        std::string queueName = mThreads[PTid{counter.pid, counter.tid}].name;
        auto& intervalCount = machinesQueueIntervalCount[machineType][qId];
        auto& totalCount = machinesQueueTotalCount[machineType][qId];
        auto& pushpopCount = machinesQueuePushpopCount[machineType][qId];

        if (counter.type == CounterType::QUEUE_PUSH) {
            intervalCount++;
            totalCount++;
        } else {
            intervalCount--;
            totalCount--;
        }
        pushpopCount++;

        if ((counter.timestamp / interval) != (lastTime / interval)) {
            mCounterEventIdPtr++;
            auto sizeCount = CounterEvent{
                .id = mCounterEventIdPtr,
                .catagory = "count",
                .phase = "C",
                .type = CounterType::COUNT_SIZE,
                .size = totalCount,
                .timestamp = counter.timestamp,
                .pid = counter.pid,
                .tid = counter.tid,
            };
            mCounterEventIdPtr++;
            auto pushpopNum = CounterEvent{
                .id = mCounterEventIdPtr,
                .catagory = "count",
                .phase = "C",
                .type = CounterType::COUNT_SIZE,
                .size = pushpopCount,
                .timestamp = counter.timestamp,
                .pid = counter.pid,
                .tid = counter.tid,
            };
            if (machineType == int(CostModel::MachineType::DEVICE)) {
                totalDeviceMachineQueueSize.emplace_back(sizeCount);
                totalDeviceMachinePushpopNum.emplace_back(pushpopNum);
            } else if (machineType == int(CostModel::MachineType::CPU)) {
                totalAicpuMachineQueueSize.emplace_back(sizeCount);
                totalAicpuMachinePushpopNum.emplace_back(pushpopNum);
            } else if (IsCoreMachine(machineType)) {
                totalCoreMachineQueueSize.emplace_back(sizeCount);
                totalCoreMachinePushpopNum.emplace_back(pushpopNum);
            } else if (machineType == int(CostModel::MachineType::PIPE)) {
                totalPipeMachineQueueSize.emplace_back(sizeCount);
                totalPipeMachinePushpopNum.emplace_back(pushpopNum);
            }
            intervalCount = 0;
            pushpopCount = 0;
        }
        lastTime = counter.timestamp;
    }
}

void TraceLogger::GetFunctionCacheSize(TimeStamp interval,
                                       const std::pair<const PTid, std::vector<CounterEvent>>& threadCounter)
{
    int totalCount = 0;
    int hitCount = 0;
    int missCount = 0;
    TimeStamp lastTime = 0;
    Pid pid = threadCounter.first.pid;
    Tid tid = threadCounter.first.tid;
    for (auto& counter : threadCounter.second) {
        if ((counter.timestamp / interval) != (lastTime / interval)) {
            auto totalNum = CounterEvent{
                .id = ++mCounterEventIdPtr,
                .catagory = "count",
                .phase = "C",
                .type = CounterType::COUNT_SIZE,
                .size = totalCount,
                .timestamp = lastTime,
                .pid = pid,
                .tid = tid,
            };
            auto hitNum = CounterEvent{
                .id = ++mCounterEventIdPtr,
                .catagory = "count",
                .phase = "C",
                .type = CounterType::COUNT_SIZE,
                .size = hitCount,
                .timestamp = lastTime,
                .pid = pid,
                .tid = tid,
            };
            auto missNum = CounterEvent{
                .id = ++mCounterEventIdPtr,
                .catagory = "count",
                .phase = "C",
                .type = CounterType::COUNT_SIZE,
                .size = missCount,
                .timestamp = lastTime,
                .pid = pid,
                .tid = tid,
            };
            functionCacheTotal.emplace_back(totalNum);
            functionCacheHit.emplace_back(hitNum);
            functionCacheMiss.emplace_back(missNum);
            totalCount = 0;
            hitCount = 0;
            missCount = 0;
        }
        if (counter.type == CounterType::CACHE_HIT) {
            hitCount++;
            totalCount++;
        } else if (counter.type == CounterType::CACHE_MISS) {
            missCount++;
            totalCount++;
        } else {
            // unexpected
        }
        lastTime = counter.timestamp;
    }
}

void TraceLogger::GetTotalFunctionCacheSize(TimeStamp interval)
{
    std::vector<CounterEvent> totalCounterVec;
    for (auto& threadCounter : mCounts) {
        if (threadCounter.first.tid == sim->pidToMachineMp[threadCounter.first.pid]->functionCacheTid) {
            std::copy(threadCounter.second.begin(), threadCounter.second.end(), std::back_inserter(totalCounterVec));
        }
    }
    sort(totalCounterVec.begin(), totalCounterVec.end(),
         [&](CounterEvent a, CounterEvent b) { return a.timestamp < b.timestamp; });
    Pid pid = sim->machines[0]->machineId;
    Tid tid = sim->machines[0]->functionCacheTid;

    GetFunctionCacheSize(interval, {{pid, tid}, totalCounterVec});
}

void TraceLogger::GetCounters()
{
    const uint32_t intervalLen = 100;
    TimeStamp interval = TimeStamp(intervalLen);
    for (auto& threadCounter : mCounts) {
        if (threadCounter.first.tid == sim->pidToMachineMp[threadCounter.first.pid]->functionCacheTid) {
            GetFunctionCacheSize(intervalLen, threadCounter);
            continue;
        }
        if (!sim->IsQueue(threadCounter.first.tid)) {
            continue;
        }

        int totalCount = 0;
        TimeStamp lastTime = 0;
        for (auto& count : threadCounter.second) {
            if (count.type == CounterType::QUEUE_PUSH) {
                totalCount++;
            } else {
                totalCount--;
            }
            if ((count.timestamp / interval) != (lastTime / interval)) {
                mCounterEventIdPtr++;
                auto sizeCount = CounterEvent{
                    .id = mCounterEventIdPtr,
                    .catagory = "count",
                    .phase = "C",
                    .type = CounterType::COUNT_SIZE,
                    .size = totalCount,
                    .timestamp = count.timestamp,
                    .pid = threadCounter.first.pid,
                    .tid = threadCounter.first.tid,
                };
                eachMachineQueueSize[threadCounter.first].emplace_back(sizeCount);
            }
            lastTime = count.timestamp;
        }
    }
    GetTotalFunctionCacheSize(intervalLen);
    GetTotalMachineQueueSize(intervalLen);
}

void TraceLogger::GetDeviceReadyQ()
{
    if (processDeviceReadyQueue) {
        return;
    }

    std::map<uint64_t, int> readyQueueCounts; // Key: cycles; value: size;
    int qSize = 0;
    size_t devicePid;
    std::set<uint64_t> readyQueueTidSet;
    sim->GetDeviceReadyQueueInfo(devicePid, readyQueueTidSet);
    for (auto& countEvent : mCounts) {
        if (countEvent.first.pid != devicePid) {
            continue;
        }
        if (readyQueueTidSet.find(countEvent.first.tid) == readyQueueTidSet.end()) {
            continue;
        }
        readyQueueCounts.clear();
        qSize = 0;
        for (auto& event : countEvent.second) {
            if (event.type == CounterType::QUEUE_PUSH) {
                qSize++;
                readyQueueCounts[event.timestamp] = qSize;
            } else {
                qSize--;
                readyQueueCounts[event.timestamp] = qSize;
            }
        }
        for (auto& it : readyQueueCounts) {
            auto sizeCount = CounterEvent{
                .id = mQSizeIdPtr++,
                .catagory = mThreads[countEvent.first].name,
                .phase = "C",
                .type = CounterType::QUEUE_PUSH,
                .size = it.second,
                .timestamp = TimeStamp(it.first),
                .pid = countEvent.first.pid,
                .tid = countEvent.first.tid,
            };
            totalDeviceMachineQueueSize.emplace_back(sizeCount);
        }
    }
    processDeviceReadyQueue = true;
}

void TraceLogger::OutEachMachineQueueSize(std::ofstream& os, const uint64_t sysClockTicks)
{
    std::string title = "queueCounter-0";
    for (auto& machineQCounter : eachMachineQueueSize) {
        auto& ptid = machineQCounter.first;
        std::string queueName = mProcesses[ptid.pid].name + mThreads[ptid].name;
        std::string cpuInfo = "(   0) [000] ....";
        for (auto& count : machineQCounter.second) {
            std::ostringstream cyc1;
            cyc1 << std::fixed << std::setprecision(precision) << (double(count.timestamp) / sysClockTicks);
            std::string cycle = cyc1.str();
            std::string workInfo = ": clock_set_rate: " + queueName + " state=" + std::to_string(count.size) +
                                   " cpu_id=0";
            os << std::setw(width) << std::left << title << std::setw(width) << std::right << cpuInfo;
            os << std::setw(width2) << std::right << cycle << workInfo << std::endl;
        }
    }
}

void TraceLogger::OutCounters(std::ofstream& os, std::vector<CounterEvent>& counterQ, std::string prefix,
                              std::string suffix, const uint64_t sysClockTicks)
{
    std::string title = "queueCounter-0";
    for (auto& counter : counterQ) {
        auto ptid = PTid{counter.pid, counter.tid};
        if (ptid.pid != sim->machines[0]->machineId) {
            std::string queueName = prefix + mThreads[ptid].name + suffix;
            std::string cpuInfo = "(" + to_string(ptid.pid) + ") [" + to_string(ptid.pid % 10000) + "] ....";
            std::ostringstream cyc1;
            cyc1 << std::fixed << std::setprecision(precision) << (double(counter.timestamp) / sysClockTicks);
            std::string cycle = cyc1.str();
            std::string workInfo = ": tracing_mark_write: C|" + to_string(ptid.pid) + "|" + queueName + '|' +
                                   std::to_string(counter.size);
            os << std::setw(width) << std::left << title << std::setw(width) << std::right << cpuInfo;
            os << std::setw(width2) << std::right << cycle << workInfo << std::endl;
        } else {
            std::string queueName = prefix + mThreads[ptid].name + suffix;
            std::string cpuInfo = "(   0) [000] ....";
            std::ostringstream cyc1;
            cyc1 << std::fixed << std::setprecision(precision) << (double(counter.timestamp) / sysClockTicks);
            std::string cycle = cyc1.str();
            std::string workInfo = ": clock_set_rate: " + queueName + " state=" + std::to_string(counter.size) +
                                   " cpu_id=0";
            os << std::setw(width) << std::left << title << std::setw(width) << std::right << cpuInfo;
            os << std::setw(width2) << std::right << cycle << workInfo << std::endl;
        }
    }
}

Json TraceLogger::QSizeToJson(std::vector<CounterEvent>& counterQ)
{
    Json root = Json::array();
    for (auto& count : counterQ) {
        root.emplace_back(count.ToJson());
    }
    return root;
}

Json TraceLogger::ToJson()
{
    Json root;
    auto traceEvents = Json::array();

    int processSortIndex = 0;
    for (auto&& [pid, process] : mProcesses) {
        auto machineType = GetMachineType(pid);
        if (machineType >= int(MachineType::PIPE)) {
            continue;
        };
        traceEvents.emplace_back(process.ToJson());
        traceEvents.emplace_back(process.ToSortIndexJson(processSortIndex++));
    }
    for (auto&& [ptid, thread] : mThreads) {
        if (ptid.tid > coreTid && ptid.tid < reversedTidNum) {
            continue;
        }
        traceEvents.emplace_back(thread.ToJson());
    }

    for (auto& duration : mDurations) {
        traceEvents.emplace_back(duration.second.ToJson());
    }

    int flowIndex = 0;
    for (auto& flow : mFlows) {
        auto& fStart = mDurations[flow.from.eid].end;
        auto& fEnd = mDurations[flow.to.eid].start;
        traceEvents.emplace_back(fStart.ToFlowStartJson(flowIndex));
        traceEvents.emplace_back(fEnd.ToFlowEndJson(flowIndex));
        flowIndex++;
    }
    GetDeviceReadyQ();
    auto readyQJson = QSizeToJson(totalDeviceMachineQueueSize);
    traceEvents.insert(traceEvents.end(), readyQJson.begin(), readyQJson.end());

    root["traceEvents"] = std::move(traceEvents);
    return root;
}

void TraceLogger::ToTrace(std::ofstream& os)
{
    // Context switch
    for (auto& duration : mDurations) {
        if (duration.second.start.pid == topMachineViewPid) {
            duration.second.OutputContextSwitchTrace(os, mProcesses, mThreads, config.sysClockTicks);
        } else {
            duration.second.OutputBeginEndTrace(os, mProcesses, mThreads, config.sysClockTicks);
        }
    }
}

void TraceLogger::LogTaskInfo(Event& start, Event& end)
{
    int coreId = mProcesses[start.pid].coreIdx;

    // Get TaskID
    std::string taskKey = "TaskId:";
    int taskId = start.ExtraHintInfo(taskKey);

    std::string subgraphKey = "pSgId:";
    int pSgId = start.ExtraHintInfo(subgraphKey);

    uint64_t sTime = start.timestamp;
    uint64_t eTime = end.timestamp;

    // 创建任务JSON对象（不再包含coreType）
    Json taskJson;
    taskJson["taskId"] = taskId;
    taskJson["execStart"] = sTime;
    taskJson["execEnd"] = eTime;
    taskJson["subGraphId"] = pSgId;
    taskJson["completeSeq"] = sim->taskCompleteSeq[taskId];

    mCoreInfoLogs[coreId].taskLogs.push_back(taskJson);
}

void TraceLogger::LogPipeInfo(Event& start, Event& end)
{
    std::string name = "";
    if (sim->IsWorkPipe(start.pid, start.tid, name)) {
        int coreId = mProcesses[start.pid].coreIdx;
        uint64_t sTime = start.timestamp;
        uint64_t eTime = end.timestamp;
        Json pipeJson;
        pipeJson["tileOp"] = start.name;
        pipeJson["execStart"] = sTime;
        pipeJson["execEnd"] = eTime;

        mCoreInfoLogs[coreId].pipeLogs[name].emplace_back(pipeJson);
    }
}

void TraceLogger::LogCoreInfo(Duration& duration)
{
    auto& start = duration.start;
    auto& end = duration.end;
    size_t initPos = start.name.find("Init");
    if (initPos != std::string::npos) {
        return;
    }
    if (start.tid == 1) {
        LogTaskInfo(start, end);
    } else {
        LogPipeInfo(start, end);
    }
}

void TraceLogger::ToFilterTrace(std::ofstream& os, std::map<int, std::pair<std::string, std::vector<Json>>>& coreTasks)
{
    for (auto it : mProcesses) {
        auto machineType = GetMachineType(it.first);
        if (IsCoreMachine(machineType)) {
            coreTasks[it.second.coreIdx] = {MachineName(static_cast<MachineType>(machineType)), {}};
            mCoreInfoLogs[it.second.coreIdx] = CoreInfoLog(it.second.coreIdx,
                                                           MachineName(static_cast<MachineType>(machineType)));
        }
    }

    for (auto& duration : mDurations) {
        auto machineType = GetMachineType(duration.second.start.pid);
        if (IsCoreMachine(machineType)) {
            LogCoreInfo(duration.second);
        }
        if (duration.second.start.tid != 1 || !IsCoreMachine(machineType)) {
            continue;
        }
        auto& start = duration.second.start;
        auto& end = duration.second.end;
        size_t initPos = start.name.find("Init");
        if (initPos != std::string::npos) {
            continue;
        }
        // Get CoreID
        int coreId = mProcesses[start.pid].coreIdx;

        std::string seqKey = "SeqNo:";
        int seqNo = start.ExtraHintInfo(seqKey);

        // Get TaskID
        std::string taskKey = "TaskId:";
        int taskId = start.ExtraHintInfo(taskKey);

        std::string subgraphKey = "pSgId:";
        int pSgId = start.ExtraHintInfo(subgraphKey);

        uint64_t sTime = start.timestamp;
        uint64_t eTime = end.timestamp;

        // 创建任务JSON对象（不再包含coreType）
        Json taskJson;
        taskJson["seqNo"] = seqNo;
        taskJson["taskId"] = taskId;
        taskJson["execStart"] = sTime;
        taskJson["execEnd"] = eTime;
        taskJson["subGraphId"] = pSgId;
        taskJson["completeSeq"] = sim->taskCompleteSeq[taskId];

        coreTasks[coreId].second.push_back(taskJson);
    }

    // 输出分组结果
    Json printJson;

    for (auto& it : coreTasks) {
        Json coreJson;
        coreJson["blockIdx"] = it.first;
        coreJson["coreType"] = it.second.first; // 核心类型提升到分组层级
        coreJson["tasks"] = it.second.second;   // 任务列表
        printJson.push_back(coreJson);
    }

    os << printJson.dump(1) << std::endl;
}

void TraceLogger::ToPipeTrace(std::ofstream& os)
{
    Json res;
    for (auto& coreInfo : mCoreInfoLogs) {
        Json coreJson;
        coreJson["blockIdx"] = coreInfo.second.idx;
        coreJson["coreType"] = coreInfo.second.type;
        for (auto& pipe : coreInfo.second.pipeLogs) {
            coreJson["pipeLogs"][pipe.first] = pipe.second;
        }
        res.push_back(coreJson);
    }
    os << res.dump(1) << std::endl;
}

void TraceLogger::ToCalendarGlobalJson(std::ofstream& osCalendar,
                                       std::map<int, std::pair<std::string, std::vector<Json>>> coreTasks)
{
    int numSupportedCounters = 1;
    int counterId = 0;
    Json calendarJson;
    calendarJson["numSupportedCounters"] = numSupportedCounters;
    calendarJson["cores"] = Json::array();
    for (auto& it : coreTasks) {
        Json core;
        core["coreId"] = it.first;
        if (it.second.first.find("HUB") != std::string::npos) {
            continue;
        }
        core["tasks"] = Json::array();
        for (const auto& taskJson : it.second.second) {
            int taskId = taskJson["taskId"];
            Json waitOp;
            waitOp["counterId"] = counterId;
            waitOp["operation"] = "wait";
            waitOp["expectedValue"] = sim->taskToCounter[taskId][0];
            core["tasks"].push_back(waitOp);
            Json taskExec;
            taskExec["functionHash"] = std::to_string(sim->taskToHash[taskId]);
            taskExec["taskId"] = taskId;
            core["tasks"].push_back(taskExec);
            Json setWaitOp;
            setWaitOp["counterId"] = counterId;
            setWaitOp["operation"] = "setWait";
            setWaitOp["expectedValue"] = sim->taskToCounter[taskId][1];
            core["tasks"].push_back(setWaitOp);
            Json setOp;
            setOp["counterId"] = counterId;
            setOp["operation"] = "set";
            setOp["expectedValue"] = sim->taskToCounter[taskId][1] + 1;
            core["tasks"].push_back(setOp);
        }
        calendarJson["cores"].push_back(core);
    }
    osCalendar << calendarJson.dump(1) << std::endl;
}

} // namespace CostModel
