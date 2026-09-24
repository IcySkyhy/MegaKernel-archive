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
 * \file core_assign.h
 * \brief
 */

#ifndef PASS_CORE_ASSIGN_H
#define PASS_CORE_ASSIGN_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <queue>
#include <functional>

#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

enum class ScheduleCoreType { AIC = 0, AIV = 1 };

enum class TargetCoreType { AIC = 0, AIV0 = 1, AIV1 = 2, UNKNOWN = 3 };

constexpr int64_t NEGATIVE_ONE = -1;

inline std::string ScheduleCoreTypeToString(ScheduleCoreType coreType)
{
    if (coreType == ScheduleCoreType::AIC) {
        return "AIC";
    }
    if (coreType == ScheduleCoreType::AIV) {
        return "AIV";
    }
    return "UNKNOWN";
}

inline std::string TargetCoreTypeToString(TargetCoreType coreType)
{
    static const std::unordered_map<TargetCoreType, std::string> targetToString{{TargetCoreType::AIC, "AIC"},
                                                                                {TargetCoreType::AIV0, "AIV0"},
                                                                                {TargetCoreType::AIV1, "AIV1"},
                                                                                {TargetCoreType::UNKNOWN, "UNKNOWN"}};
    auto it = targetToString.find(coreType);
    return it != targetToString.end() ? it->second : "UNKNOWN";
}

// 跨核通信延迟惩罚权重（用于调度优化的目标函数, 后续可能需要差异化设置）
constexpr double GAP_C_C = 0.01; // cube -> cube 间隙惩罚
constexpr double GAP_C_V = 0.01; // cube -> vector 间隙惩罚
constexpr double GAP_V_C = 0.01; // vector -> cube 间隙惩罚
constexpr double GAP_V_V = 0.01; // vector -> vector 间隙惩罚

// 切分后的AIC或AIV子图
class TaskNode {
public:
    TaskNode(const std::string& taskName, int index, ScheduleCoreType taskCoreType, int taskLatency)
        : name(taskName), idx(index), coreType(taskCoreType), latency(taskLatency)
    {}
    std::string name;
    int idx;
    ScheduleCoreType coreType;
    int latency;
    std::vector<int> inTasks;
    std::vector<int> outTasks;
    TargetCoreType targetCoreType{TargetCoreType::UNKNOWN};
    int startTime{0};
    int endTime{0};
    TargetCoreType targetCoreTypeCandidate{TargetCoreType::UNKNOWN};
    int startTimeCandidate{0};
    int endTimeCandidate{0};
    std::vector<Operation*> opList_;
    // 该 task 所属的 vec 连通分支 ID（task 级别，分割完 task 后基于 task 间 AIV 连通性计算）
    // -1 表示 AIC task（不参与分支翻转）
    int vecBranchId{-1};
};

// 完整的mix子图
class TaskGraph {
public:
    void ApplyCandidate();
    void ApplyCandidateUnconditional();
    int AddTask(const std::string& name, ScheduleCoreType coreType, int latency);
    void AddDependency(int src, int dst);
    void ClearSchedule();
    std::vector<TaskNode> tasks;
    int makespan{-1};
};

// 用于将切分后的AIC和AIV子图调度到AIC,AIV0和AIV1核心上
class CoreScheduler {
public:
    static void FindEarliestSlot(std::vector<std::pair<int, int>>& timeSlot, int earliestStart, int latency,
                                 int& currentIdx, std::pair<int, int>& currentInterval);
    static void UpdateInterval(std::vector<std::pair<int, int>>& timeSlot, int& insertIdx,
                               std::pair<int, int>& insertInterval);
    std::vector<int> GetDFSTopoSeq(TaskGraph& taskGraph);
    void EFTWithInsertSchedule(TaskGraph& taskGraph, std::vector<int>& topoSeq);
    void EFTSchedule(TaskGraph& taskGraph, std::vector<int>& topoSeq);
    void BruteForceScheduleRecursiveStep(std::vector<bool>& visited, int recursiveLevel, TaskGraph& taskGraph,
                                         std::vector<int>& topoList);
    void Schedule(TaskGraph& taskGraph, const std::string& schedMode = "");
    void OptimalScheduleWithSearch(TaskGraph& taskGraph, bool enableSearch = true);
    double CalcBaselineCost(const TaskGraph& taskGraph, int n);

    void GapMinSchedule(TaskGraph& taskGraph, std::vector<int>& topoSeq);
    void GapMinForwardPass(TaskGraph& taskGraph, std::vector<int>& topoSeq);
    static void GapMinBackwardShift(TaskGraph& taskGraph, const std::vector<int>& topoSeq);
    int64_t SumCrossCoreGap(const TaskGraph& g) const;
    void SelectAIVCore(TaskGraph& taskGraph, int taskId,
                       std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime,
                       int evalDepTimeStart, int latency, TargetCoreType& evalCore, int& currentIdx,
                       std::pair<int, int>& currentInterval);
    int FindDepAIVLastEnd(const TaskGraph& taskGraph, int taskId) const;
    void ScheduleOneTask(TaskGraph& taskGraph, int taskId,
                         std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime,
                         std::function<bool(TargetCoreType)> isAicCore,
                         std::unordered_map<int, TargetCoreType>& vecBranchToCore);
    TargetCoreType LookupPinnedAIVCore(const TaskNode& task,
                                       const std::unordered_map<int, TargetCoreType>& vecBranchToCore);
    void TryScheduleSuccessors(TaskGraph& taskGraph, int taskId,
                               std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime,
                               std::set<int>& scheduledTasks, std::function<bool(TargetCoreType)> isAicCore,
                               std::unordered_map<int, TargetCoreType>& vecBranchToCore);
    void HLFSchedule(TaskGraph& taskGraph);
    static bool HasOooScopeTasks(const TaskGraph& taskGraph);
    void NormalizeSingleAIVBranches(TaskGraph& taskGraph);
};

} // namespace npu::tile_fwk
#endif // PASS_CORE_ASSIGN_H
