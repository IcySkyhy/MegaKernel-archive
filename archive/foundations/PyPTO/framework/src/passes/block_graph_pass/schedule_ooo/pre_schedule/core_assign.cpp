/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file core_assign.cpp
 * \brief
 */

#include "core_assign.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>

#include "local_search_solver.h"
#include "passes/pass_log/pass_log.h"

#ifndef MODULE_NAME
#define MODULE_NAME "CoreAssign"
#endif

namespace npu::tile_fwk {

void TaskGraph::ApplyCandidate()
{
    int prevTime = makespan;
    int currTime = -1;
    for (auto& task : tasks) {
        currTime = currTime > task.endTimeCandidate ? currTime : task.endTimeCandidate;
    }
    if (prevTime >= 0 && prevTime <= currTime) {
        return;
    }
    for (auto& task : tasks) {
        task.startTime = task.startTimeCandidate;
        task.endTime = task.endTimeCandidate;
        task.targetCoreType = task.targetCoreTypeCandidate;
    }
    makespan = currTime;
    APASS_LOG_INFO_F(Elements::Operation, "Found better schedule, update makespan to %d.", makespan);
}

// 无条件把 Candidate 字段拷贝为 final；makespan 仅作为字段完整性保留，不参与任何决策
void TaskGraph::ApplyCandidateUnconditional()
{
    int currTime = -1;
    for (auto& task : tasks) {
        task.startTime = task.startTimeCandidate;
        task.endTime = task.endTimeCandidate;
        task.targetCoreType = task.targetCoreTypeCandidate;
        currTime = currTime > task.endTime ? currTime : task.endTime;
    }
    makespan = currTime;
}

int TaskGraph::AddTask(const std::string& name, ScheduleCoreType coreType, int latency)
{
    int newTaskIdx = static_cast<int>(tasks.size());
    tasks.emplace_back(name, newTaskIdx, coreType, latency);
    APASS_LOG_DEBUG_F(Elements::Operation, "Create new taskNode with idx=%d and coreType=%s.", newTaskIdx,
                      ScheduleCoreTypeToString(coreType).c_str());
    return newTaskIdx;
}

void TaskGraph::AddDependency(int src, int dst)
{
    if (src == dst) {
        return;
    }
    tasks[src].outTasks.push_back(dst);
    tasks[dst].inTasks.push_back(src);
    APASS_LOG_DEBUG_F(Elements::Operation, "Create dependency from taskNode %d to taskNode %d.", src, dst);
}

inline int UDSFind(std::vector<int>& parent, int i)
{
    if (parent[i] == i) {
        return i;
    }
    return parent[i] = UDSFind(parent, parent[i]);
}

inline void UDSUnion(std::vector<int>& parent, int i, int j)
{
    int rootOfI = UDSFind(parent, i);
    int rootOfJ = UDSFind(parent, j);
    if (rootOfI != rootOfJ) {
        if (rootOfI < rootOfJ) {
            parent[rootOfJ] = rootOfI;
        } else {
            parent[rootOfI] = rootOfJ;
        }
    }
}

void TaskGraph::ClearSchedule()
{
    makespan = -1;
    for (auto& task : tasks) {
        task.targetCoreType = TargetCoreType::UNKNOWN;
        task.startTime = 0;
        task.endTime = 0;
        task.targetCoreTypeCandidate = TargetCoreType::UNKNOWN;
        task.startTimeCandidate = 0;
        task.endTimeCandidate = 0;
    }
}

// 寻找时间槽不重叠情况下的最早执行时间
void CoreScheduler::FindEarliestSlot(std::vector<std::pair<int, int>>& timeSlot, int earliestStart, int latency,
                                     int& currentIdx, std::pair<int, int>& currentInterval)
{
    int currentEarliestStart = INT32_MAX;
    currentIdx = -1;
    currentInterval = std::make_pair(-1, -1);
    for (int i = 0; i < static_cast<int>(timeSlot.size()); i++) {
        int validStart = std::max(timeSlot[i].first, earliestStart);
        if (timeSlot[i].second - validStart < latency) {
            continue;
        }
        if (validStart < currentEarliestStart) {
            currentEarliestStart = validStart;
            currentInterval = std::make_pair(validStart, validStart + latency);
            currentIdx = i;
        }
    }
}

// 更新空闲时间槽
void CoreScheduler::UpdateInterval(std::vector<std::pair<int, int>>& timeSlot, int& insertIdx,
                                   std::pair<int, int>& insertInterval)
{
    auto origInterval = timeSlot[insertIdx];
    timeSlot.erase(timeSlot.begin() + insertIdx);
    if (origInterval.first < insertInterval.first) {
        timeSlot.push_back(std::make_pair(origInterval.first, insertInterval.first));
    }
    if (origInterval.second > insertInterval.second) {
        timeSlot.push_back(std::make_pair(insertInterval.second, origInterval.second));
    }
}

// 使用DFS得到taskNode的一个拓扑排序
std::vector<int> CoreScheduler::GetDFSTopoSeq(TaskGraph& taskGraph)
{
    std::vector<bool> finishedTasks(taskGraph.tasks.size(), false);
    std::vector<int> taskStack;
    std::vector<int> topoSeq;
    for (auto& task : taskGraph.tasks) {
        if (task.outTasks.size() == 0) {
            taskStack.push_back(task.idx);
        }
    }
    std::vector<int> notReadyPrevTaskIds;
    while (taskStack.size() > 0) {
        int taskId = taskStack.back();
        taskStack.pop_back();
        if (finishedTasks[taskId]) {
            continue;
        }
        notReadyPrevTaskIds.clear();
        for (int prevTaskId : taskGraph.tasks[taskId].inTasks) {
            if (!finishedTasks[prevTaskId]) {
                notReadyPrevTaskIds.push_back(prevTaskId);
            }
        }
        if (notReadyPrevTaskIds.size() > 0) {
            taskStack.push_back(taskId);
            taskStack.insert(taskStack.end(), notReadyPrevTaskIds.begin(), notReadyPrevTaskIds.end());
            continue;
        }
        topoSeq.push_back(taskId);
        finishedTasks[taskId] = true;
    }
    return topoSeq;
}

// 基于最早完成时间和空闲时间槽的任务排布
void CoreScheduler::EFTWithInsertSchedule(TaskGraph& taskGraph, std::vector<int>& topoSeq)
{
    std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>> availTime;
    availTime[TargetCoreType::AIC] = {{0, INT32_MAX}};
    availTime[TargetCoreType::AIV0] = {{0, INT32_MAX}};
    availTime[TargetCoreType::AIV1] = {{0, INT32_MAX}};
    std::unordered_map<int, TargetCoreType> vecBranchToCore;
    int currentIdx = -1;
    std::pair<int, int> currentInterval{-1, -1};
    for (int taskId : topoSeq) {
        auto& task = taskGraph.tasks[taskId];
        int evalDepTimeStart = 0;
        for (int prevTaskId : task.inTasks) {
            evalDepTimeStart = std::max(evalDepTimeStart, taskGraph.tasks[prevTaskId].endTimeCandidate);
        }
        TargetCoreType evalCore = TargetCoreType::UNKNOWN;
        if (task.coreType == ScheduleCoreType::AIC) {
            evalCore = TargetCoreType::AIC;
            FindEarliestSlot(availTime[evalCore], evalDepTimeStart, task.latency, currentIdx, currentInterval);
        } else {
            TargetCoreType pinnedCore = LookupPinnedAIVCore(task, vecBranchToCore);
            if (pinnedCore != TargetCoreType::UNKNOWN) {
                evalCore = pinnedCore;
                FindEarliestSlot(availTime[evalCore], evalDepTimeStart, task.latency, currentIdx, currentInterval);
            } else {
                int currentIdxAIV0 = -1;
                std::pair<int, int> currentIntervalAIV0{-1, -1};
                int currentIdxAIV1 = -1;
                std::pair<int, int> currentIntervalAIV1{-1, -1};
                FindEarliestSlot(availTime[TargetCoreType::AIV0], evalDepTimeStart, task.latency, currentIdxAIV0,
                                 currentIntervalAIV0);
                FindEarliestSlot(availTime[TargetCoreType::AIV1], evalDepTimeStart, task.latency, currentIdxAIV1,
                                 currentIntervalAIV1);
                if (currentIntervalAIV0.first <= currentIntervalAIV1.first) {
                    evalCore = TargetCoreType::AIV0;
                    currentIdx = currentIdxAIV0;
                    currentInterval = currentIntervalAIV0;
                } else {
                    evalCore = TargetCoreType::AIV1;
                    currentIdx = currentIdxAIV1;
                    currentInterval = currentIntervalAIV1;
                }
            }
            if (task.vecBranchId >= 0) {
                vecBranchToCore[task.vecBranchId] = evalCore;
            }
        }
        task.targetCoreTypeCandidate = evalCore;
        task.startTimeCandidate = currentInterval.first;
        task.endTimeCandidate = currentInterval.second;
        UpdateInterval(availTime[evalCore], currentIdx, currentInterval);
    }
    taskGraph.ApplyCandidate();
    APASS_LOG_INFO_F(Elements::Operation, "EFTWithInsertSchedule get final makespan %d.", taskGraph.makespan);
}

// 基于最早完成时间的任务排布
void CoreScheduler::EFTSchedule(TaskGraph& taskGraph, std::vector<int>& topoSeq)
{
    std::unordered_map<TargetCoreType, int> currentTime{
        {TargetCoreType::AIC, 0}, {TargetCoreType::AIV0, 0}, {TargetCoreType::AIV1, 0}};
    for (int taskId : topoSeq) {
        int evalDepTimeStart = 0;
        for (int prevTaskId : taskGraph.tasks[taskId].inTasks) {
            evalDepTimeStart = std::max(evalDepTimeStart, taskGraph.tasks[prevTaskId].endTimeCandidate);
        }
        TargetCoreType evalCore = TargetCoreType::UNKNOWN;
        if (taskGraph.tasks[taskId].coreType == ScheduleCoreType::AIC) {
            evalCore = TargetCoreType::AIC;
        } else {
            evalCore = currentTime[TargetCoreType::AIV0] <= currentTime[TargetCoreType::AIV1] ? TargetCoreType::AIV0 :
                                                                                                TargetCoreType::AIV1;
        }
        taskGraph.tasks[taskId].targetCoreTypeCandidate = evalCore;
        taskGraph.tasks[taskId].startTimeCandidate = std::max(evalDepTimeStart, currentTime[evalCore]);
        taskGraph.tasks[taskId].endTimeCandidate = taskGraph.tasks[taskId].startTimeCandidate +
                                                   taskGraph.tasks[taskId].latency;
        currentTime[evalCore] = taskGraph.tasks[taskId].endTimeCandidate;
    }
    taskGraph.ApplyCandidate();
    APASS_LOG_INFO_F(Elements::Operation, "EFTSchedule get final makespan %d.", taskGraph.makespan);
}

// 对所有的拓扑序执行基于最早完成时间的任务排布
void CoreScheduler::BruteForceScheduleRecursiveStep(std::vector<bool>& visited, int recursiveLevel,
                                                    TaskGraph& taskGraph, std::vector<int>& topoList)
{
    if (recursiveLevel >= static_cast<int>(taskGraph.tasks.size())) {
        EFTSchedule(taskGraph, topoList);
    }
    for (auto& task : taskGraph.tasks) {
        if (visited[task.idx]) {
            continue;
        }
        bool canDeploy = true;
        for (int prevTaskIdx : task.inTasks) {
            if (!visited[prevTaskIdx]) {
                canDeploy = false;
                break;
            }
        }
        if (!canDeploy) {
            continue;
        }
        visited[task.idx] = true;
        topoList.push_back(task.idx);
        BruteForceScheduleRecursiveStep(visited, recursiveLevel + 1, taskGraph, topoList);
        topoList.pop_back();
        visited[task.idx] = false;
    }
}

void CoreScheduler::SelectAIVCore(TaskGraph& taskGraph, int taskId,
                                  std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime,
                                  int evalDepTimeStart, int latency, TargetCoreType& evalCore, int& currentIdx,
                                  std::pair<int, int>& currentInterval)
{
    int idxAIV0 = -1;
    std::pair<int, int> intervalAIV0{-1, -1};
    FindEarliestSlot(availTime[TargetCoreType::AIV0], evalDepTimeStart, latency, idxAIV0, intervalAIV0);
    int idxAIV1 = -1;
    std::pair<int, int> intervalAIV1{-1, -1};
    FindEarliestSlot(availTime[TargetCoreType::AIV1], evalDepTimeStart, latency, idxAIV1, intervalAIV1);

    auto getLastFinishBefore = [&](TargetCoreType core, int time) -> int {
        for (auto& slot : availTime[core]) {
            if (slot.second >= time && slot.first <= time) {
                return slot.first;
            }
        }
        return 0;
    };

    bool chooseAIV0 = false;
    if (intervalAIV0.first < intervalAIV1.first) {
        chooseAIV0 = true;
    } else if (intervalAIV0.first == intervalAIV1.first) {
        int depAIVEnd = FindDepAIVLastEnd(taskGraph, taskId);
        int score0 = std::max(getLastFinishBefore(TargetCoreType::AIV0, intervalAIV0.first), depAIVEnd);
        int score1 = std::max(getLastFinishBefore(TargetCoreType::AIV1, intervalAIV1.first), depAIVEnd);
        chooseAIV0 = (score0 <= score1);
    }
    if (chooseAIV0) {
        evalCore = TargetCoreType::AIV0;
        currentIdx = idxAIV0;
        currentInterval = intervalAIV0;
    } else {
        evalCore = TargetCoreType::AIV1;
        currentIdx = idxAIV1;
        currentInterval = intervalAIV1;
    }
}

// 回溯依赖链，找当前 AIV 任务所依赖的所有 AIV 任务中最晚的结束时间
// 对 AIC 前驱则继续回溯其 AIV 前驱
int CoreScheduler::FindDepAIVLastEnd(const TaskGraph& taskGraph, int taskId) const
{
    int result = 0;
    std::queue<int> q;
    std::unordered_set<int> visited;
    q.push(taskId);
    visited.insert(taskId);
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        for (int prevId : taskGraph.tasks[cur].inTasks) {
            if (visited.count(prevId) > 0) {
                continue;
            }
            visited.insert(prevId);
            auto& prev = taskGraph.tasks[prevId];
            bool prevIsAIV = (prev.targetCoreTypeCandidate == TargetCoreType::AIV0 ||
                              prev.targetCoreTypeCandidate == TargetCoreType::AIV1);
            if (prevIsAIV) {
                result = std::max(result, prev.endTimeCandidate);
            } else {
                // AIC 前驱：继续回溯找其 AIV 前驱
                q.push(prevId);
            }
        }
    }
    return result;
}

// --------- GapMin: 启发式最小化相邻依赖边的等待间隔 ---------
// 紧耦合调度单个任务
void CoreScheduler::ScheduleOneTask(TaskGraph& taskGraph, int taskId,
                                    std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime,
                                    std::function<bool(TargetCoreType)> /* isAicCore */,
                                    std::unordered_map<int, TargetCoreType>& vecBranchToCore)
{
    auto& task = taskGraph.tasks[taskId];
    int evalDepTimeStart = 0;
    bool taskIsAic = (task.coreType == ScheduleCoreType::AIC);
    for (int prevTaskId : task.inTasks) {
        auto& prev = taskGraph.tasks[prevTaskId];
        evalDepTimeStart = std::max(evalDepTimeStart, prev.endTimeCandidate);
    }

    TargetCoreType evalCore = TargetCoreType::UNKNOWN;
    int currentIdx = -1;
    std::pair<int, int> currentInterval{-1, -1};

    if (taskIsAic) {
        evalCore = TargetCoreType::AIC;
        FindEarliestSlot(availTime[evalCore], evalDepTimeStart, task.latency, currentIdx, currentInterval);
    } else {
        TargetCoreType pinnedCore = LookupPinnedAIVCore(task, vecBranchToCore);
        if (pinnedCore != TargetCoreType::UNKNOWN) {
            evalCore = pinnedCore;
            FindEarliestSlot(availTime[evalCore], evalDepTimeStart, task.latency, currentIdx, currentInterval);
        } else {
            SelectAIVCore(taskGraph, taskId, availTime, evalDepTimeStart, task.latency, evalCore, currentIdx,
                          currentInterval);
        }
    }
    task.targetCoreTypeCandidate = evalCore;
    task.startTimeCandidate = currentInterval.first;
    task.endTimeCandidate = currentInterval.second;
    UpdateInterval(availTime[evalCore], currentIdx, currentInterval);

    if (!taskIsAic && task.vecBranchId >= 0) {
        vecBranchToCore[task.vecBranchId] = evalCore;
    }
}

// 查找 AIV 任务是否因连通分支约束而必须钉在某个核上
// 遍历已调度的 AIV 任务，若其 vecBranchId 与当前任务相同，则复用其核
TargetCoreType CoreScheduler::LookupPinnedAIVCore(const TaskNode& task,
                                                  const std::unordered_map<int, TargetCoreType>& vecBranchToCore)
{
    if (task.vecBranchId < 0) {
        return TargetCoreType::UNKNOWN;
    }
    auto it = vecBranchToCore.find(task.vecBranchId);
    if (it != vecBranchToCore.end()) {
        return it->second;
    }
    return TargetCoreType::UNKNOWN;
}

// 尝试调度后继
void CoreScheduler::TryScheduleSuccessors(
    TaskGraph& taskGraph, int taskId, std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime,
    std::set<int>& scheduledTasks, std::function<bool(TargetCoreType)> isAicCore,
    std::unordered_map<int, TargetCoreType>& vecBranchToCore)
{
    auto& task = taskGraph.tasks[taskId];

    for (int succId : task.outTasks) {
        if (scheduledTasks.count(succId) > 0) {
            continue;
        }

        auto& succ = taskGraph.tasks[succId];
        bool allPredScheduled = true;
        for (int predId : succ.inTasks) {
            if (scheduledTasks.count(predId) == 0) {
                allPredScheduled = false;
                break;
            }
        }

        if (allPredScheduled) {
            ScheduleOneTask(taskGraph, succId, availTime, isAicCore, vecBranchToCore);
            scheduledTasks.insert(succId);
            TryScheduleSuccessors(taskGraph, succId, availTime, scheduledTasks, isAicCore, vecBranchToCore);
        }
    }
}

// 轮 1：gap-aware 前向排布，紧耦合调度：任务完成后立即调度后继
void CoreScheduler::GapMinForwardPass(TaskGraph& taskGraph, std::vector<int>& topoSeq)
{
    std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>> availTime;
    availTime[TargetCoreType::AIC] = {{0, INT32_MAX}};
    availTime[TargetCoreType::AIV0] = {{0, INT32_MAX}};
    availTime[TargetCoreType::AIV1] = {{0, INT32_MAX}};
    auto isAicCore = [](TargetCoreType c) { return c == TargetCoreType::AIC; };

    std::set<int> scheduledTasks;
    std::unordered_map<int, TargetCoreType> vecBranchToCore;

    for (int taskId : topoSeq) {
        if (scheduledTasks.count(taskId) > 0) {
            continue;
        }

        ScheduleOneTask(taskGraph, taskId, availTime, isAicCore, vecBranchToCore);
        scheduledTasks.insert(taskId);

        TryScheduleSuccessors(taskGraph, taskId, availTime, scheduledTasks, isAicCore, vecBranchToCore);
    }
}

// 轮 2.：按拓扑序反向 ALAP 位移 —— 把每个任务右移到 "同核右邻 start" 与 "所有后继最早 start" 的下界，收敛 outbound gap
void CoreScheduler::GapMinBackwardShift(TaskGraph& taskGraph, const std::vector<int>& topoSeq)
{
    // 1. 按核分组并按 startTimeCandidate 排序，建立同核右邻映射
    std::unordered_map<TargetCoreType, std::vector<int>> coreGroups;
    for (auto& t : taskGraph.tasks) {
        coreGroups[t.targetCoreTypeCandidate].push_back(t.idx);
    }
    std::unordered_map<int, int> sameCoreNext; // taskId -> next taskId on same core (-1 if none)
    for (auto& pr : coreGroups) {
        auto& ids = pr.second;
        std::sort(ids.begin(), ids.end(), [&](int i, int j) {
            return taskGraph.tasks[i].startTimeCandidate < taskGraph.tasks[j].startTimeCandidate;
        });
        for (size_t i = 0; i + 1 < ids.size(); i++) {
            sameCoreNext[ids[i]] = ids[i + 1];
        }
        if (!ids.empty()) {
            sameCoreNext[ids.back()] = -1;
        }
    }

    // 2. 按 topo 逆序 shift
    for (auto it = topoSeq.rbegin(); it != topoSeq.rend(); ++it) {
        int taskId = *it;
        auto& t = taskGraph.tasks[taskId];
        if (t.outTasks.empty()) {
            continue;
        }
        int minSuccStart = INT32_MAX;
        for (int s : t.outTasks) {
            minSuccStart = std::min(minSuccStart, taskGraph.tasks[s].startTimeCandidate);
        }
        int sameCoreCap = INT32_MAX;
        auto sit = sameCoreNext.find(taskId);
        if (sit != sameCoreNext.end() && sit->second >= 0) {
            sameCoreCap = taskGraph.tasks[sit->second].startTimeCandidate;
        }
        int newEnd = std::min(minSuccStart, sameCoreCap);
        if (newEnd <= t.endTimeCandidate) {
            continue;
        }
        int delta = newEnd - t.endTimeCandidate;
        t.startTimeCandidate += delta;
        t.endTimeCandidate = newEnd;
    }
}

// 统计所有 AIC<->AIV 双向跨核边的等待间隔之和
int64_t CoreScheduler::SumCrossCoreGap(const TaskGraph& g) const
{
    auto isAic = [](TargetCoreType c) { return c == TargetCoreType::AIC; };
    int64_t sum = 0;
    for (auto& u : g.tasks) {
        for (int vId : u.outTasks) {
            auto& v = g.tasks[vId];
            bool cross = isAic(u.targetCoreType) != isAic(v.targetCoreType);
            if (!cross) {
                continue;
            }
            sum += std::max(0, v.startTime - u.endTime);
        }
    }
    return sum;
}

// 顶层：两轮 + 无条件应用 + 统计日志
void CoreScheduler::GapMinSchedule(TaskGraph& taskGraph, std::vector<int>& topoSeq)
{
    GapMinForwardPass(taskGraph, topoSeq);
    GapMinBackwardShift(taskGraph, topoSeq);
    taskGraph.ApplyCandidateUnconditional();
    APASS_LOG_INFO_F(Elements::Operation, "GapMinSchedule total cross-core gap=%lld, endTime(max)=%d.",
                     static_cast<long long>(SumCrossCoreGap(taskGraph)), taskGraph.makespan);
}

// 计算 GapMin baseline 的总代价 = makespan + Σ_e (gap(e) * coeff(e))
double CoreScheduler::CalcBaselineCost(const TaskGraph& taskGraph, int n)
{
    double penalty = 0.0;
    static const double gapCoeffTable[2][2] = {{GAP_C_C, GAP_C_V}, {GAP_V_C, GAP_V_V}};
    for (int i = 0; i < n; ++i) {
        auto& taskA = taskGraph.tasks[i];
        bool isAC = (taskA.coreType == ScheduleCoreType::AIC);
        for (int taskBId : taskA.outTasks) {
            auto& taskB = taskGraph.tasks[taskBId];
            int gap = taskB.startTime - taskA.endTime;
            bool isBC = (taskB.coreType == ScheduleCoreType::AIC);
            penalty += gapCoeffTable[isAC ? 0 : 1][isBC ? 0 : 1] * gap;
        }
    }
    return taskGraph.makespan + penalty;
}

// In-process classical local-search refinement on top of the GapMin baseline.
// Returns true if improved cost, false otherwise. Deterministic for identical inputs.
static bool RunLocalSearch(TaskGraph& taskGraph, double baselineTotalCost)
{
    LocalSearchSolver solver;
    return solver.Solve(taskGraph, baselineTotalCost);
}

// Normalize single-AIV branches — if no task is assigned to AIV0, reassign all AIV1 tasks to AIV0
void CoreScheduler::NormalizeSingleAIVBranches(TaskGraph& taskGraph)
{
    bool hasAIV0 = false;
    bool hasAIV1 = false;
    std::vector<int> aiv1TaskIds;
    for (const auto& task : taskGraph.tasks) {
        if (task.coreType != ScheduleCoreType::AIV) {
            continue;
        }
        if (task.targetCoreType == TargetCoreType::AIV0) {
            hasAIV0 = true;
        }
        if (task.targetCoreType == TargetCoreType::AIV1) {
            hasAIV1 = true;
            aiv1TaskIds.push_back(task.idx);
        }
    }
    if ((Platform::Instance().GetSoc().GetAICToAIVCoreRatio() == SoCAICToAIVCoreRatio::ONE_AIC_TO_ONE_AIV_CORE) ||
        (!hasAIV0 && hasAIV1)) {
        for (int taskId : aiv1TaskIds) {
            taskGraph.tasks[taskId].targetCoreType = TargetCoreType::AIV0;
            taskGraph.tasks[taskId].targetCoreTypeCandidate = TargetCoreType::AIV0;
        }
        APASS_LOG_INFO_F(Elements::Operation, "Normalize single-AIV: all %d AIV1 tasks changed to AIV0.",
                         static_cast<int>(aiv1TaskIds.size()));
    }
}

void CoreScheduler::OptimalScheduleWithSearch(TaskGraph& taskGraph, bool enableSearch)
{
    taskGraph.ClearSchedule();
    APASS_LOG_INFO_F(Elements::Operation, "Start hybrid schedule: GapMin baseline + local-search refinement.");

    int n = static_cast<int>(taskGraph.tasks.size());
    if (n == 0) {
        return;
    }

    // Step 1: GapMin baseline
    std::vector<int> topoSeq = GetDFSTopoSeq(taskGraph);
    GapMinSchedule(taskGraph, topoSeq);
    double baselineTotalCost = CalcBaselineCost(taskGraph, n);
    int baselineMakespan = taskGraph.makespan;
    APASS_LOG_INFO_F(Elements::Operation, "GapMin baseline: makespan=%d, total_cost=%.4f.", baselineMakespan,
                     baselineTotalCost);

    // Step 2: in-process local-search refinement (deterministic, no external solver)
    if (enableSearch) {
        RunLocalSearch(taskGraph, baselineTotalCost);
    }
}

bool CoreScheduler::HasOooScopeTasks(const TaskGraph& taskGraph)
{
    for (auto& task : taskGraph.tasks) {
        for (auto* op : task.opList_) {
            if (op->GetOooScopeId() > 0) {
                return true;
            }
        }
    }
    return false;
}

void CoreScheduler::HLFSchedule(TaskGraph& taskGraph)
{
    taskGraph.ClearSchedule();
    int n = static_cast<int>(taskGraph.tasks.size());
    if (n == 0) {
        return;
    }
    APASS_LOG_INFO_F(Elements::Operation, "Start HLF schedule with %d tasks.", n);

    // 计算反向 cycle 深度（从 sink 回推，累加每个 task 的 latency）
    std::vector<int> revDepth(n, 0);
    std::vector<int> revTopoOrder;
    revTopoOrder.reserve(n);
    std::vector<int> outDeg(n, 0);
    for (int i = 0; i < n; i++) {
        outDeg[i] = static_cast<int>(taskGraph.tasks[i].outTasks.size());
    }
    std::queue<int> q;
    for (int i = 0; i < n; i++) {
        if (outDeg[i] == 0) {
            revDepth[i] = std::max(1, taskGraph.tasks[i].latency);
            q.push(i);
        }
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        revTopoOrder.push_back(u);
        for (int pred : taskGraph.tasks[u].inTasks) {
            int newDepth = revDepth[u] + std::max(1, taskGraph.tasks[pred].latency);
            if (newDepth > revDepth[pred]) {
                revDepth[pred] = newDepth;
            }
            outDeg[pred]--;
            if (outDeg[pred] == 0) {
                q.push(pred);
            }
        }
    }

    // 按反向深度降序排列，同深度内 AIC 优先于 AIV
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (revDepth[a] != revDepth[b]) {
            return revDepth[a] > revDepth[b];
        }
        bool aIsAIC = (taskGraph.tasks[a].coreType == ScheduleCoreType::AIC);
        bool bIsAIC = (taskGraph.tasks[b].coreType == ScheduleCoreType::AIC);
        if (aIsAIC != bIsAIC) {
            return aIsAIC > bIsAIC;
        }
        return a < b;
    });

    for (int i = 0; i < n; i++) {
        APASS_LOG_INFO_F(Elements::Operation, "HLF order[%d] = task %d, revDepth=%d, coreType=%s.", i, order[i],
                         revDepth[order[i]],
                         taskGraph.tasks[order[i]].coreType == ScheduleCoreType::AIC ? "AIC" : "AIV");
    }

    EFTWithInsertSchedule(taskGraph, order);
    APASS_LOG_INFO_F(Elements::Operation, "HLF schedule done, makespan=%d.", taskGraph.makespan);
}

// 根据节点数量，判断是否遍历所有拓扑序进行任务排布
void CoreScheduler::Schedule(TaskGraph& taskGraph, const std::string& schedMode)
{
    APASS_LOG_INFO_F(Elements::Operation, "Start schedule, mode=%s.", schedMode.c_str());
    if (schedMode == "HLF") {
        HLFSchedule(taskGraph);
    } else if (schedMode == "GAPMIN") {
        OptimalScheduleWithSearch(taskGraph, /*enableSearch=*/false);
    } else {
        OptimalScheduleWithSearch(taskGraph);
    }
    NormalizeSingleAIVBranches(taskGraph);
}

} // namespace npu::tile_fwk
