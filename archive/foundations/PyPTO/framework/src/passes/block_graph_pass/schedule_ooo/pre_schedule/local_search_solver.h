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
 * \file local_search_solver.h
 * \brief In-process classical local-search refinement of the GapMin schedule.
 *        Fully deterministic (no randomness, no time-based termination).
 */

#ifndef PASS_LOCAL_SEARCH_SOLVER_H
#define PASS_LOCAL_SEARCH_SOLVER_H

#include "core_assign.h"

#include <map>
#include <vector>

namespace npu::tile_fwk {

// 局部搜索浮点比较容差（判断 cost 改善时使用，避免浮点抖动导致误判）
constexpr double LS_EPS = 1e-9;
// 插入位移邻域搜索半径（每个任务在拓扑序中前后最多移动 SHIFT_RADIUS 个位置）
constexpr int LS_SHIFT_RADIUS = 10;
// 禁忌表生存时间（被接受为 fallback 的 move 在后续 TABU_TTL 轮内不再被考虑）
constexpr int LS_TABU_TTL = 7;
// 搜索总操作预算（控制最大迭代次数，防止大图搜索时间过长）
constexpr long long LS_BUDGET_OPS = 1000000LL;

class LocalSearchSolver {
public:
    bool Solve(TaskGraph& taskGraph, double baselineCost);

private:
    struct Snapshot {
        std::vector<TargetCoreType> targetCoreTypeCandidate;
        std::vector<int> startTimeCandidate;
        std::vector<int> endTimeCandidate;
    };

    // 候选邻居（用于恶化接受 fallback）
    struct Fallback {
        double cost = std::numeric_limits<double>::infinity();
        int op = -1;
        int arg = -1;
        std::map<int, TargetCoreType> pin;
        std::vector<int> topo;
        Snapshot snap;
    };

    // 搜索状态：在 Solve 各子函数间共享
    struct SearchState {
        int n;
        double bestCost;
        int bestMakespan;
        Snapshot bestSnap;
        std::vector<int> curTopo;
        std::map<int, TargetCoreType> curPin;
        std::vector<int> branches;
        int maxIters;
        int patience;
        std::map<std::pair<int, int>, int> tabuMap;
        int noImprove = 0;
    };

    // --- 解码器 ---
    double DecodeAndScore(TaskGraph& taskGraph, const std::vector<int>& topoOrder,
                          const std::map<int, TargetCoreType>& pinIn, std::map<int, TargetCoreType>& pinOut,
                          int& makespan);
    void ForwardPassWithPin(TaskGraph& taskGraph, const std::vector<int>& topoOrder,
                            std::map<int, TargetCoreType>& pin);
    void ScheduleOneTask(TaskNode& task, std::map<int, TargetCoreType>& pin,
                         std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime);
    void SelectBestAIVCore(TaskNode& task, std::map<int, TargetCoreType>& pin,
                           std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime,
                           int depStart);
    double CalcCandidateCost(const TaskGraph& taskGraph) const;

    // --- 快照 ---
    void TakeSnapshot(const TaskGraph& taskGraph, Snapshot& snap);
    void RestoreSnapshot(TaskGraph& taskGraph, const Snapshot& snap);
    void TakeBaselineSnapshot(const TaskGraph& taskGraph, Snapshot& snap);

    // --- 工具 ---
    bool CanSwapAdjacent(const TaskGraph& taskGraph, const std::vector<int>& topoOrder, int i) const;
    std::vector<int> CollectBranches(const TaskGraph& taskGraph) const;

    // --- Solve 子步骤 ---
    void InitSearchState(TaskGraph& taskGraph, double baselineCost, SearchState& ss);
    bool RunOneIteration(TaskGraph& taskGraph, SearchState& ss);
    bool RunBranchFlips(TaskGraph& taskGraph, SearchState& ss, Fallback& fb, bool& improved);
    bool RunAdjacentSwaps(TaskGraph& taskGraph, SearchState& ss, Fallback& fb, bool& improved);
    bool RunInsertionShifts(TaskGraph& taskGraph, SearchState& ss, Fallback& fb, bool& improved);
    // 对单个位置 i，在 [lo, hi] 范围内找最优 shift 目标 j
    struct ShiftResult {
        double cost = std::numeric_limits<double>::infinity();
        int j = -1;
        int mks = 0;
        std::map<int, TargetCoreType> pin;
        std::vector<int> topo;
    };
    void FindBestShiftForTask(TaskGraph& taskGraph, SearchState& ss, int i, int lo, int hi, ShiftResult& result);
    void AcceptFallback(TaskGraph& taskGraph, SearchState& ss, Fallback& fb);
    bool ApplyBestResult(TaskGraph& taskGraph, SearchState& ss, double baselineCost);

    // --- Tabu 工具 ---
    static std::pair<int, int> EncodeMove(int op, int arg) { return {op, arg}; }
    bool IsTabu(const SearchState& ss, int op, int arg) const;
    void DecayTabu(SearchState& ss);
};

} // namespace npu::tile_fwk

#endif // PASS_LOCAL_SEARCH_SOLVER_H
