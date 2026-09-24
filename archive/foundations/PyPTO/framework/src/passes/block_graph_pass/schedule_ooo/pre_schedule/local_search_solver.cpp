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
 * \file local_search_solver.cpp
 * \brief 经典局部搜索 schedule 优化器：以 GapMin 调度结果作为初值，通过两类邻域
 *        算子 (分支翻转 / 任务插入位移) 进行滚动基底搜索 + 恶化接受，
 *        全程维护 best-ever，绝不退化 baseline。完全确定、可重入、无外部依赖。
 *        目标函数 makespan + Σ gap*coeff 与 CalcBaselineCost 完全一致。
 */

#include "local_search_solver.h"
#include "passes/pass_log/pass_log.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <set>

#ifndef MODULE_NAME
#define MODULE_NAME "CoreAssign"
#endif

namespace npu::tile_fwk {

// ============================================================
// 工具函数
// ============================================================

constexpr int kMoveAdjacentSwap = 2;
constexpr int kMoveInsertionShift = 3;

void LocalSearchSolver::TakeSnapshot(const TaskGraph& taskGraph, Snapshot& snap)
{
    int n = static_cast<int>(taskGraph.tasks.size());
    snap.targetCoreTypeCandidate.resize(n);
    snap.startTimeCandidate.resize(n);
    snap.endTimeCandidate.resize(n);
    for (int i = 0; i < n; i++) {
        snap.targetCoreTypeCandidate[i] = taskGraph.tasks[i].targetCoreTypeCandidate;
        snap.startTimeCandidate[i] = taskGraph.tasks[i].startTimeCandidate;
        snap.endTimeCandidate[i] = taskGraph.tasks[i].endTimeCandidate;
    }
}

void LocalSearchSolver::RestoreSnapshot(TaskGraph& taskGraph, const Snapshot& snap)
{
    int n = static_cast<int>(taskGraph.tasks.size());
    for (int i = 0; i < n; i++) {
        taskGraph.tasks[i].targetCoreTypeCandidate = snap.targetCoreTypeCandidate[i];
        taskGraph.tasks[i].startTimeCandidate = snap.startTimeCandidate[i];
        taskGraph.tasks[i].endTimeCandidate = snap.endTimeCandidate[i];
    }
}

void LocalSearchSolver::TakeBaselineSnapshot(const TaskGraph& taskGraph, Snapshot& snap)
{
    int n = static_cast<int>(taskGraph.tasks.size());
    snap.targetCoreTypeCandidate.resize(n);
    snap.startTimeCandidate.resize(n);
    snap.endTimeCandidate.resize(n);
    for (int i = 0; i < n; i++) {
        snap.targetCoreTypeCandidate[i] = taskGraph.tasks[i].targetCoreType;
        snap.startTimeCandidate[i] = taskGraph.tasks[i].startTime;
        snap.endTimeCandidate[i] = taskGraph.tasks[i].endTime;
    }
}

bool LocalSearchSolver::CanSwapAdjacent(const TaskGraph& taskGraph, const std::vector<int>& topoOrder, int i) const
{
    int a = topoOrder[i];
    int b = topoOrder[i + 1];
    for (int s : taskGraph.tasks[a].outTasks) {
        if (s == b) {
            return false;
        }
    }
    return true;
}

std::vector<int> LocalSearchSolver::CollectBranches(const TaskGraph& taskGraph) const
{
    std::set<int> bs;
    for (auto& t : taskGraph.tasks) {
        if (t.vecBranchId >= 0) {
            bs.insert(t.vecBranchId);
        }
    }
    return std::vector<int>(bs.begin(), bs.end());
}

bool LocalSearchSolver::IsTabu(const SearchState& ss, int op, int arg) const
{
    auto it = ss.tabuMap.find(EncodeMove(op, arg));
    return it != ss.tabuMap.end() && it->second > 0;
}

void LocalSearchSolver::DecayTabu(SearchState& ss)
{
    for (auto& kv : ss.tabuMap) {
        if (kv.second > 0)
            kv.second--;
    }
}

// ============================================================
// 解码器：ForwardPassWithPin 及其子函数
// ============================================================

void LocalSearchSolver::SelectBestAIVCore(
    TaskNode& task, std::map<int, TargetCoreType>& pin,
    std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime, int depStart)
{
    int idx0 = -1, idx1 = -1;
    std::pair<int, int> iv0{-1, -1}, iv1{-1, -1};
    CoreScheduler::FindEarliestSlot(availTime[TargetCoreType::AIV0], depStart, task.latency, idx0, iv0);
    CoreScheduler::FindEarliestSlot(availTime[TargetCoreType::AIV1], depStart, task.latency, idx1, iv1);
    TargetCoreType core;
    int idx;
    std::pair<int, int> interval;
    if (iv0.first <= iv1.first) {
        core = TargetCoreType::AIV0;
        idx = idx0;
        interval = iv0;
    } else {
        core = TargetCoreType::AIV1;
        idx = idx1;
        interval = iv1;
    }
    if (task.vecBranchId >= 0) {
        pin[task.vecBranchId] = core;
    }
    task.targetCoreTypeCandidate = core;
    task.startTimeCandidate = interval.first;
    task.endTimeCandidate = interval.second;
    CoreScheduler::UpdateInterval(availTime[core], idx, interval);
}

void LocalSearchSolver::ScheduleOneTask(TaskNode& task, std::map<int, TargetCoreType>& pin,
                                        std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime)
{
    // depStart 已由调用方写入 task.endTimeCandidate
    int depStart = task.endTimeCandidate;

    if (task.coreType == ScheduleCoreType::AIC) {
        TargetCoreType core = TargetCoreType::AIC;
        int idx = -1;
        std::pair<int, int> interval{-1, -1};
        CoreScheduler::FindEarliestSlot(availTime[core], depStart, task.latency, idx, interval);
        task.targetCoreTypeCandidate = core;
        task.startTimeCandidate = interval.first;
        task.endTimeCandidate = interval.second;
        CoreScheduler::UpdateInterval(availTime[core], idx, interval);
        return;
    }

    // AIV: 检查 pin
    TargetCoreType forced = TargetCoreType::UNKNOWN;
    if (task.vecBranchId >= 0) {
        auto it = pin.find(task.vecBranchId);
        if (it != pin.end()) {
            forced = it->second;
        }
    }
    if (forced != TargetCoreType::UNKNOWN) {
        int idx = -1;
        std::pair<int, int> interval{-1, -1};
        CoreScheduler::FindEarliestSlot(availTime[forced], depStart, task.latency, idx, interval);
        task.targetCoreTypeCandidate = forced;
        task.startTimeCandidate = interval.first;
        task.endTimeCandidate = interval.second;
        CoreScheduler::UpdateInterval(availTime[forced], idx, interval);
        return;
    }

    SelectBestAIVCore(task, pin, availTime, depStart);
}

void LocalSearchSolver::ForwardPassWithPin(TaskGraph& taskGraph, const std::vector<int>& topoOrder,
                                           std::map<int, TargetCoreType>& pin)
{
    std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>> availTime;
    availTime[TargetCoreType::AIC] = {{0, INT32_MAX}};
    availTime[TargetCoreType::AIV0] = {{0, INT32_MAX}};
    availTime[TargetCoreType::AIV1] = {{0, INT32_MAX}};

    std::set<int> scheduled;

    auto allPredScheduled = [&](int tid) {
        for (int p : taskGraph.tasks[tid].inTasks) {
            if (scheduled.count(p) == 0)
                return false;
        }
        return true;
    };

    auto doSchedule = [&](int tid) {
        auto& task = taskGraph.tasks[tid];
        int depStart = 0;
        for (int p : task.inTasks) {
            depStart = std::max(depStart, taskGraph.tasks[p].endTimeCandidate);
        }
        // 暂存 depStart 到 endTimeCandidate 供 ScheduleOneTask 读取
        task.endTimeCandidate = depStart;
        ScheduleOneTask(task, pin, availTime);
    };

    for (int tid : topoOrder) {
        if (scheduled.count(tid) > 0 || !allPredScheduled(tid))
            continue;
        doSchedule(tid);
        scheduled.insert(tid);
    }

    // 兜底：若 topoOrder 中漏掉某个任务
    int maxRounds = static_cast<int>(taskGraph.tasks.size()) + 1;
    bool progress = true;
    while (progress && maxRounds > 0 && static_cast<int>(scheduled.size()) < static_cast<int>(taskGraph.tasks.size())) {
        progress = false;
        maxRounds--;
        for (int tid = 0; tid < static_cast<int>(taskGraph.tasks.size()); tid++) {
            if (scheduled.count(tid) > 0 || !allPredScheduled(tid))
                continue;
            doSchedule(tid);
            scheduled.insert(tid);
            progress = true;
        }
    }
}

// ============================================================
// DecodeAndScore / CalcCandidateCost
// ============================================================

double LocalSearchSolver::CalcCandidateCost(const TaskGraph& g) const
{
    static const double gapCoeffTable[2][2] = {{GAP_C_C, GAP_C_V}, {GAP_V_C, GAP_V_V}};
    int makespan = 0;
    double penalty = 0.0;
    for (auto& a : g.tasks) {
        makespan = std::max(makespan, a.endTimeCandidate);
        bool isAC = (a.coreType == ScheduleCoreType::AIC);
        for (int bIdx : a.outTasks) {
            auto& b = g.tasks[bIdx];
            int gap = b.startTimeCandidate - a.endTimeCandidate;
            bool isBC = (b.coreType == ScheduleCoreType::AIC);
            penalty += gapCoeffTable[isAC ? 0 : 1][isBC ? 0 : 1] * gap;
        }
    }
    return static_cast<double>(makespan) + penalty;
}

double LocalSearchSolver::DecodeAndScore(TaskGraph& taskGraph, const std::vector<int>& topoOrder,
                                         const std::map<int, TargetCoreType>& pinIn,
                                         std::map<int, TargetCoreType>& pinOut, int& makespan)
{
    for (auto& t : taskGraph.tasks) {
        t.targetCoreTypeCandidate = TargetCoreType::UNKNOWN;
        t.startTimeCandidate = 0;
        t.endTimeCandidate = 0;
    }
    pinOut = pinIn;
    ForwardPassWithPin(taskGraph, topoOrder, pinOut);
    CoreScheduler::GapMinBackwardShift(taskGraph, topoOrder);
    makespan = 0;
    for (auto& t : taskGraph.tasks) {
        makespan = std::max(makespan, t.endTimeCandidate);
    }
    return CalcCandidateCost(taskGraph);
}

// ============================================================
// Solve 子步骤
// ============================================================

void LocalSearchSolver::InitSearchState(TaskGraph& taskGraph, double baselineCost, SearchState& ss)
{
    ss.n = static_cast<int>(taskGraph.tasks.size());
    ss.bestCost = baselineCost;
    ss.bestMakespan = taskGraph.makespan;
    TakeBaselineSnapshot(taskGraph, ss.bestSnap);

    // 初始拓扑序：按 baseline startTime 升序
    ss.curTopo.resize(ss.n);
    for (int i = 0; i < ss.n; i++)
        ss.curTopo[i] = i;
    std::sort(ss.curTopo.begin(), ss.curTopo.end(), [&](int a, int b) {
        if (taskGraph.tasks[a].startTime != taskGraph.tasks[b].startTime)
            return taskGraph.tasks[a].startTime < taskGraph.tasks[b].startTime;
        return a < b;
    });

    // 初始 pin：从 baseline 提取
    for (auto& t : taskGraph.tasks) {
        if (t.targetCoreType == TargetCoreType::AIV0 || t.targetCoreType == TargetCoreType::AIV1) {
            if (t.vecBranchId >= 0)
                ss.curPin[t.vecBranchId] = t.targetCoreType;
        }
    }

    // 解码一次作为搜索起点
    std::map<int, TargetCoreType> tmpPin;
    int tmpMks = 0;
    double curCost = DecodeAndScore(taskGraph, ss.curTopo, ss.curPin, tmpPin, tmpMks);
    ss.curPin = tmpPin;
    if (curCost + LS_EPS < ss.bestCost) {
        TakeSnapshot(taskGraph, ss.bestSnap);
        ss.bestCost = curCost;
        ss.bestMakespan = tmpMks;
    }

    ss.branches = CollectBranches(taskGraph);

    // 预算
    // 预算：每轮 decode 次数 ≈ |B| (OP1) + n (OP2) + n*2*LS_SHIFT_RADIUS (OP3)
    constexpr int kPatienceFloor = 4;   // 最小耐心(迭代数)下限
    constexpr int kPatienceDivisor = 4; // patience = maxIters / 该除数
    long long decodesPerIter = static_cast<long long>(ss.branches.size()) + static_cast<long long>(ss.n) +
                               static_cast<long long>(ss.n) * LS_SHIFT_RADIUS * 2;
    long long opsPerIter = std::max(1LL, decodesPerIter * ss.n);
    ss.maxIters = static_cast<int>(std::min(256LL, std::max(1LL, LS_BUDGET_OPS / opsPerIter)));
    ss.patience = std::max(kPatienceFloor, ss.maxIters / kPatienceDivisor);
    ss.noImprove = 0;

    APASS_LOG_INFO_F(Elements::Operation, "[local_search] n=%d, branches=%zu, maxIters=%d.", ss.n, ss.branches.size(),
                     ss.maxIters);
}

bool LocalSearchSolver::RunBranchFlips(TaskGraph& taskGraph, SearchState& ss, Fallback& fb, bool& improved)
{
    for (int b : ss.branches) {
        auto it = ss.curPin.find(b);
        if (it == ss.curPin.end())
            continue;
        TargetCoreType flipped = (it->second == TargetCoreType::AIV0) ? TargetCoreType::AIV1 : TargetCoreType::AIV0;
        std::map<int, TargetCoreType> candPin = ss.curPin;
        candPin[b] = flipped;
        std::map<int, TargetCoreType> outPin;
        int mks = 0;
        double cost = DecodeAndScore(taskGraph, ss.curTopo, candPin, outPin, mks);
        if (cost + LS_EPS < ss.bestCost) {
            ss.bestCost = cost;
            ss.bestMakespan = mks;
            ss.curPin = outPin;
            TakeSnapshot(taskGraph, ss.bestSnap);
            improved = true;
        } else if (!IsTabu(ss, 1, b) && cost < fb.cost - LS_EPS) {
            fb.cost = cost;
            fb.op = 1;
            fb.arg = b;
            fb.pin = outPin;
            fb.topo = ss.curTopo;
            TakeSnapshot(taskGraph, fb.snap);
        }
    }
    return improved;
}

bool LocalSearchSolver::RunAdjacentSwaps(TaskGraph& taskGraph, SearchState& ss, Fallback& fb, bool& improved)
{
    int n = ss.n;
    for (int i = 0; i + 1 < n; i++) {
        if (!CanSwapAdjacent(taskGraph, ss.curTopo, i))
            continue;
        std::vector<int> candTopo = ss.curTopo;
        std::swap(candTopo[i], candTopo[i + 1]);
        std::map<int, TargetCoreType> outPin;
        int mks = 0;
        double cost = DecodeAndScore(taskGraph, candTopo, ss.curPin, outPin, mks);
        if (cost + LS_EPS < ss.bestCost) {
            ss.bestCost = cost;
            ss.bestMakespan = mks;
            ss.curTopo = candTopo;
            ss.curPin = outPin;
            TakeSnapshot(taskGraph, ss.bestSnap);
            improved = true;
        } else if (!IsTabu(ss, kMoveAdjacentSwap, i) && cost < fb.cost - LS_EPS) {
            fb.cost = cost;
            fb.op = kMoveAdjacentSwap;
            fb.arg = i;
            fb.pin = outPin;
            fb.topo = candTopo;
            TakeSnapshot(taskGraph, fb.snap);
        }
    }
    return improved;
}

// Complexity: Each iteration is O(n² · LS_SHIFT_RADIUS). Total budget capped at LS_BUDGET_OPS ops.
// For n > 1000 InsertionShifts is skipped to avoid excessive overhead.
bool LocalSearchSolver::RunInsertionShifts(TaskGraph& taskGraph, SearchState& ss, Fallback& fb, bool& improved)
{
    constexpr int kInsertionShiftMaxTasks = 1000; // 超过该规模跳过 InsertionShifts 以约束求解耗时
    int n = ss.n;
    // Skip InsertionShifts for large graphs to bound overall solver time.
    if (n > kInsertionShiftMaxTasks) {
        return improved;
    }
    std::vector<int> posOf(n, 0);
    for (int p = 0; p < n; p++)
        posOf[ss.curTopo[p]] = p;

    for (int i = 0; i < n; i++) {
        int tid = ss.curTopo[i];
        int lo = 0, hi = n - 1;
        for (int p : taskGraph.tasks[tid].inTasks)
            lo = std::max(lo, posOf[p] + 1);
        for (int s : taskGraph.tasks[tid].outTasks)
            hi = std::min(hi, posOf[s] - 1);
        lo = std::max(lo, i - LS_SHIFT_RADIUS);
        hi = std::min(hi, i + LS_SHIFT_RADIUS);

        double bestJCost = std::numeric_limits<double>::infinity();
        int bestJ = -1, bestJMks = 0;
        std::map<int, TargetCoreType> bestJPin;
        std::vector<int> bestJTopo;

        for (int j = lo; j <= hi; j++) {
            if (j == i)
                continue;
            std::vector<int> candTopo = ss.curTopo;
            // Use std::rotate instead of erase+insert for O(n) shift instead of O(n) alloc overhead.
            if (j < i) {
                std::rotate(candTopo.begin() + j, candTopo.begin() + i, candTopo.begin() + i + 1);
            } else {
                std::rotate(candTopo.begin() + i, candTopo.begin() + i + 1, candTopo.begin() + j + 1);
            }
            std::map<int, TargetCoreType> outPin;
            int mks = 0;
            double cost = DecodeAndScore(taskGraph, candTopo, ss.curPin, outPin, mks);
            if (cost < bestJCost - LS_EPS) {
                bestJCost = cost;
                bestJMks = mks;
                bestJ = j;
                bestJPin = outPin;
                bestJTopo = candTopo;
            }
        }
        if (bestJ < 0)
            continue;

        if (bestJCost + LS_EPS < ss.bestCost) {
            ss.bestCost = bestJCost;
            ss.bestMakespan = bestJMks;
            ss.curTopo = bestJTopo;
            ss.curPin = bestJPin;
            TakeSnapshot(taskGraph, ss.bestSnap);
            improved = true;
            for (int pp = 0; pp < n; pp++)
                posOf[ss.curTopo[pp]] = pp;
        } else {
            int arg = i * n + bestJ;
            if (!IsTabu(ss, kMoveInsertionShift, arg) && bestJCost < fb.cost - LS_EPS) {
                fb.cost = bestJCost;
                fb.op = kMoveInsertionShift;
                fb.arg = arg;
                fb.pin = bestJPin;
                fb.topo = bestJTopo;
                TakeSnapshot(taskGraph, fb.snap);
            }
        }
    }
    return improved;
}

void LocalSearchSolver::AcceptFallback(TaskGraph& taskGraph, SearchState& ss, Fallback& fb)
{
    ss.curTopo = fb.topo;
    ss.curPin = fb.pin;
    RestoreSnapshot(taskGraph, fb.snap);
    ss.tabuMap[EncodeMove(fb.op, fb.arg)] = LS_TABU_TTL;
}

bool LocalSearchSolver::RunOneIteration(TaskGraph& taskGraph, SearchState& ss)
{
    bool improved = false;
    Fallback fb;

    RunBranchFlips(taskGraph, ss, fb, improved);
    RunAdjacentSwaps(taskGraph, ss, fb, improved);
    RunInsertionShifts(taskGraph, ss, fb, improved);

    if (improved) {
        ss.noImprove = 0;
    } else if (fb.op >= 0) {
        AcceptFallback(taskGraph, ss, fb);
        ss.noImprove++;
        if (ss.noImprove >= ss.patience)
            return false;
    } else {
        return false;
    }

    DecayTabu(ss);
    return true;
}

bool LocalSearchSolver::ApplyBestResult(TaskGraph& taskGraph, SearchState& ss, double baselineCost)
{
    int baselineMakespan = taskGraph.makespan;
    if (ss.bestCost + LS_EPS < baselineCost) {
        RestoreSnapshot(taskGraph, ss.bestSnap);
        taskGraph.ApplyCandidateUnconditional();
        APASS_LOG_INFO_F(Elements::Operation, "[local_search] improved: cost %.4f -> %.4f, makespan %d -> %d.",
                         baselineCost, ss.bestCost, baselineMakespan, ss.bestMakespan);
        return true;
    }
    APASS_LOG_INFO_F(
        Elements::Operation,
        "[local_search] no improvement over baseline (cost %.4f, makespan %d; best cost %.4f), keep baseline.",
        baselineCost, baselineMakespan, ss.bestCost);
    return false;
}

// ============================================================
// Solve 主入口
// ============================================================

bool LocalSearchSolver::Solve(TaskGraph& taskGraph, double baselineCost)
{
    if (taskGraph.tasks.empty())
        return false;

    SearchState ss;
    InitSearchState(taskGraph, baselineCost, ss);

    for (int iter = 0; iter < ss.maxIters; iter++) {
        if (!RunOneIteration(taskGraph, ss))
            break;
    }

    return ApplyBestResult(taskGraph, ss, baselineCost);
}

} // namespace npu::tile_fwk
