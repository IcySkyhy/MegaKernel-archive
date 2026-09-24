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
 * \file grow_local_auto_cores.h
 * \brief
 */

#ifndef OSP_GROW_LOCAL_AUTO_CORES_H
#define OSP_GROW_LOCAL_AUTO_CORES_H

#include <chrono>
#include <climits>
#include <list>
#include <map>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/bsp/model/bsp_schedule.h"
#include "passes/algorithms/osp/bsp/scheduler/scheduler.h"

namespace npu::tile_fwk {
namespace osp {
template <typename WeightT>
struct GrowLocalAutoCoresParams {
    unsigned minSuperstepSize_ = 20;
    WeightT syncCostMultiplierMinSuperstepWeight_ = 1;
    WeightT syncCostMultiplierParallelCheck_ = 4;
};

/**
 * @brief The GreedyBspGrowLocalAutoCores class represents a scheduler that uses a greedy algorithm to compute
 * schedules for BspInstance.
 *
 * This class inherits from the Scheduler class and implements the ComputeSchedule() and methods.
 * The ComputeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 */
template <typename GraphT>
class GrowLocalAutoCores : public Scheduler<GraphT> {
public:
    using VertexIdx = typename GraphT::VertexIdx;

    GrowLocalAutoCores(GrowLocalAutoCoresParams<VWorkwT<GraphT>> params = GrowLocalAutoCoresParams<VWorkwT<GraphT>>())
        : params_(params)
    {}

    virtual ~GrowLocalAutoCores() = default;

    struct ScheduleState {
        std::unordered_set<VertexIdx> ready;
        std::vector<VertexIdx> predec;
        std::vector<std::vector<VertexIdx>> newAssignments;
        std::vector<std::vector<VertexIdx>> bestNewAssignments;
        std::vector<VertexIdx> newReady;
        std::vector<VertexIdx> bestNewReady;
        std::vector<VertexIdx> allReady;
        std::vector<std::vector<VertexIdx>> procReady;
        VWorkwT<GraphT> minWeightParallelCheck;
        VWorkwT<GraphT> minSuperstepWeight;
        double desiredParallelism;
    };

    void InitializeScheduleDataStructures(BspSchedule<GraphT>& schedule, ScheduleState& state)
    {
        const auto& instance = schedule.GetInstance();
        const auto& g = instance.GetComputationalDag();
        const auto n = instance.NumberOfVertices();
        const unsigned p = instance.NumberOfProcessors();

        for (const auto& v : g.Vertices()) {
            schedule.SetAssignedProcessor(v, std::numeric_limits<unsigned>::max());
            schedule.SetAssignedSuperstep(v, std::numeric_limits<unsigned>::max());
        }

        state.predec.resize(n);
        InitializeReadyQueue(g, state.ready, state.predec);

        state.newAssignments.resize(p);
        state.bestNewAssignments.resize(p);
        state.procReady.resize(p);

        state.minWeightParallelCheck = params_.syncCostMultiplierParallelCheck_ * instance.SynchronisationCosts();
        state.minSuperstepWeight = params_.syncCostMultiplierMinSuperstepWeight_ * instance.SynchronisationCosts();
        state.desiredParallelism = static_cast<double>(p);
    }

    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT>& schedule) override
    {
        constexpr unsigned kLimitGrowthDivisor = 2;
        const auto& instance = schedule.GetInstance();
        const auto& g = instance.GetComputationalDag();
        const auto n = instance.NumberOfVertices();
        const unsigned p = instance.NumberOfProcessors();

        auto& nodeToProc = schedule.AssignedProcessors();
        auto& nodeToSupstep = schedule.AssignedSupersteps();

        ScheduleState state;
        InitializeScheduleDataStructures(schedule, state);

        VertexIdx totalAssigned = 0;
        unsigned supstep = 0;

        while (totalAssigned < n) {
            unsigned limit = params_.minSuperstepSize_;
            double bestScore = 0;
            double bestParallelism = 0;
            bool continueSuperstepAttempts = true;

            while (continueSuperstepAttempts) {
                PrepareSuperstepAttempt(state.newAssignments, state.procReady, state.newReady, state.allReady,
                                        state.ready, p);

                VertexIdx newTotalAssigned = 0;
                VWorkwT<GraphT> weightLimit = ScheduleProcessorZero(g, limit, state.allReady, state.procReady[0],
                                                                    state.newAssignments[0], nodeToProc, state.predec,
                                                                    state.newReady, newTotalAssigned, p);
                VWorkwT<GraphT> totalWeightAssigned = ScheduleRemainingProcessors(
                    g, p, weightLimit, state.allReady, state.procReady, state.newAssignments, nodeToProc, state.predec,
                    state.newReady, newTotalAssigned);

                auto result = EvaluateSuperstep(totalWeightAssigned, weightLimit, instance, bestScore, bestParallelism,
                                                state.minWeightParallelCheck, state.minSuperstepWeight,
                                                state.desiredParallelism, totalAssigned, newTotalAssigned, n);

                RollbackAssignments(state.newAssignments, g, nodeToProc, state.predec, p);

                if (result.acceptStep) {
                    state.bestNewAssignments.swap(state.newAssignments);
                    state.bestNewReady.swap(state.newReady);
                    bestScore = result.bestScore;
                    bestParallelism = result.bestParallelism;
                }

                continueSuperstepAttempts = result.continueAttempts;
                limit++;
                limit += (limit / kLimitGrowthDivisor);
            }

            CommitBestAssignments(state.bestNewReady, state.bestNewAssignments, state.ready, nodeToProc, nodeToSupstep,
                                  state.predec, g, supstep, totalAssigned, p);
            state.desiredParallelism = (desiredParallelismHistoryWeight_ * state.desiredParallelism) +
                                       (desiredParallelismCurrentWeight_ * bestParallelism) +
                                       (desiredParallelismProcessorWeight_ * static_cast<double>(p));
            ++supstep;
        }

        schedule.UpdateNumberOfSupersteps();
        return ReturnStatus::OSP_SUCCESS;
    }

private:
    static constexpr double desiredParallelismHistoryWeight_ = 0.3;
    static constexpr double desiredParallelismCurrentWeight_ = 0.6;
    static constexpr double desiredParallelismProcessorWeight_ = 0.1;
    static constexpr double scoreAcceptanceRatio_ = 0.97;
    static constexpr double minAbsoluteParallelism_ = 2.0;
    static constexpr double desiredParallelismFactor_ = 0.8;

    GrowLocalAutoCoresParams<VWorkwT<GraphT>> params_;

    void InitializeReadyQueue(const GraphT& g, std::unordered_set<VertexIdx>& ready, std::vector<VertexIdx>& predec)
    {
        for (const auto& node : g.Vertices()) {
            predec[node] = g.InDegree(node);
            if (predec[node] == 0) {
                ready.insert(node);
            }
        }
    }

    VertexIdx ChooseNode(std::vector<VertexIdx>& procReady, std::vector<VertexIdx>& allReady)
    {
        VertexIdx chosenNode = std::numeric_limits<VertexIdx>::max();

        if (!procReady.empty()) {
            chosenNode = procReady.front();
            std::pop_heap(procReady.begin(), procReady.end(), std::greater<VertexIdx>());
            procReady.pop_back();
        } else if (!allReady.empty()) {
            chosenNode = allReady.front();
            std::pop_heap(allReady.begin(), allReady.end(), std::greater<VertexIdx>());
            allReady.pop_back();
        }

        return chosenNode;
    }

    void UpdateSuccessors(const GraphT& g, VertexIdx node, unsigned proc, unsigned p, std::vector<unsigned>& nodeToProc,
                          std::vector<VertexIdx>& predec, std::vector<VertexIdx>& newReady,
                          std::vector<VertexIdx>& procReady)
    {
        for (const auto& succ : g.Children(node)) {
            if (nodeToProc[succ] == std::numeric_limits<unsigned>::max()) {
                nodeToProc[succ] = proc;
            } else if (nodeToProc[succ] != proc) {
                nodeToProc[succ] = p;
            }

            predec[succ]--;
            if (predec[succ] == 0) {
                newReady.push_back(succ);

                if (nodeToProc[succ] == proc) {
                    procReady.push_back(succ);
                    std::push_heap(procReady.begin(), procReady.end(), std::greater<VertexIdx>());
                }
            }
        }
    }

    void PrepareSuperstepAttempt(std::vector<std::vector<VertexIdx>>& newAssignments,
                                 std::vector<std::vector<VertexIdx>>& procReady, std::vector<VertexIdx>& newReady,
                                 std::vector<VertexIdx>& allReady, const std::unordered_set<VertexIdx>& ready,
                                 unsigned p)
    {
        for (unsigned pIdx = 0; pIdx < p; pIdx++) {
            newAssignments[pIdx].clear();
            procReady[pIdx].clear();
        }
        newReady.clear();
        allReady.assign(ready.begin(), ready.end());
        std::make_heap(allReady.begin(), allReady.end(), std::greater<VertexIdx>());
    }

    VWorkwT<GraphT> ScheduleProcessorZero(const GraphT& g, unsigned limit, std::vector<VertexIdx>& allReady,
                                          std::vector<VertexIdx>& procReady, std::vector<VertexIdx>& assignments,
                                          std::vector<unsigned>& nodeToProc, std::vector<VertexIdx>& predec,
                                          std::vector<VertexIdx>& newReady, VertexIdx& newTotalAssigned, unsigned p)
    {
        VWorkwT<GraphT> weightLimit = 0;

        while (assignments.size() < limit) {
            VertexIdx chosenNode = ChooseNode(procReady, allReady);
            if (chosenNode == std::numeric_limits<VertexIdx>::max()) {
                break;
            }

            assignments.push_back(chosenNode);
            nodeToProc[chosenNode] = 0;
            newTotalAssigned++;
            weightLimit += g.VertexWorkWeight(chosenNode);

            UpdateSuccessors(g, chosenNode, 0, p, nodeToProc, predec, newReady, procReady);
        }

        return weightLimit;
    }

    VWorkwT<GraphT> ScheduleRemainingProcessors(const GraphT& g, unsigned p, VWorkwT<GraphT> weightLimit,
                                                std::vector<VertexIdx>& allReady,
                                                std::vector<std::vector<VertexIdx>>& procReady,
                                                std::vector<std::vector<VertexIdx>>& newAssignments,
                                                std::vector<unsigned>& nodeToProc, std::vector<VertexIdx>& predec,
                                                std::vector<VertexIdx>& newReady, VertexIdx& newTotalAssigned)
    {
        VWorkwT<GraphT> totalWeightAssigned = weightLimit;

        for (unsigned proc = 1; proc < p; ++proc) {
            VWorkwT<GraphT> currentWeightAssigned = 0;

            while (currentWeightAssigned < weightLimit) {
                VertexIdx chosenNode = ChooseNode(procReady[proc], allReady);
                if (chosenNode == std::numeric_limits<VertexIdx>::max()) {
                    break;
                }

                newAssignments[proc].push_back(chosenNode);
                nodeToProc[chosenNode] = proc;
                newTotalAssigned++;
                currentWeightAssigned += g.VertexWorkWeight(chosenNode);

                UpdateSuccessors(g, chosenNode, proc, p, nodeToProc, predec, newReady, procReady[proc]);
            }

            weightLimit = std::max(weightLimit, currentWeightAssigned);
            totalWeightAssigned += currentWeightAssigned;
        }

        return totalWeightAssigned;
    }

    struct SuperstepEvaluation {
        bool acceptStep;
        bool continueAttempts;
        double bestScore;
        double bestParallelism;
    };

    SuperstepEvaluation EvaluateSuperstep(VWorkwT<GraphT> totalWeightAssigned, VWorkwT<GraphT> weightLimit,
                                          const BspInstance<GraphT>& instance, double currentBestScore,
                                          double currentBestParallelism, VWorkwT<GraphT> minWeightParallelCheck,
                                          VWorkwT<GraphT> minSuperstepWeight, double desiredParallelism,
                                          VertexIdx totalAssigned, VertexIdx newTotalAssigned, VertexIdx n)
    {
        SuperstepEvaluation result;
        result.acceptStep = false;
        result.continueAttempts = true;
        result.bestScore = currentBestScore;
        result.bestParallelism = currentBestParallelism;

        double score = static_cast<double>(totalWeightAssigned) /
                       static_cast<double>(weightLimit + instance.SynchronisationCosts());
        double parallelism = 0;
        if (weightLimit > 0) {
            parallelism = static_cast<double>(totalWeightAssigned) / static_cast<double>(weightLimit);
        }

        if (score > scoreAcceptanceRatio_ * currentBestScore) {
            result.bestScore = std::max(currentBestScore, score);
            result.bestParallelism = parallelism;
            result.acceptStep = true;
        } else {
            result.continueAttempts = false;
        }

        if (weightLimit >= minWeightParallelCheck) {
            if (parallelism < std::max(minAbsoluteParallelism_, desiredParallelismFactor_ * desiredParallelism)) {
                result.continueAttempts = false;
            }
        }

        if (weightLimit <= minSuperstepWeight) {
            result.continueAttempts = true;
            if (totalAssigned + newTotalAssigned == n) {
                result.acceptStep = true;
                result.continueAttempts = false;
            }
        }

        if (totalAssigned + newTotalAssigned == n) {
            result.continueAttempts = false;
        }

        return result;
    }

    void RollbackAssignments(const std::vector<std::vector<VertexIdx>>& newAssignments, const GraphT& g,
                             std::vector<unsigned>& nodeToProc, std::vector<VertexIdx>& predec, unsigned p)
    {
        for (unsigned proc = 0; proc < p; ++proc) {
            for (const auto& node : newAssignments[proc]) {
                nodeToProc[node] = std::numeric_limits<unsigned>::max();

                for (const auto& succ : g.Children(node)) {
                    predec[succ]++;
                    nodeToProc[succ] = std::numeric_limits<unsigned>::max();
                }
            }
        }
    }

    void CommitBestAssignments(const std::vector<VertexIdx>& bestNewReady,
                               const std::vector<std::vector<VertexIdx>>& bestNewAssignments,
                               std::unordered_set<VertexIdx>& ready, std::vector<unsigned>& nodeToProc,
                               std::vector<unsigned>& nodeToSupstep, std::vector<VertexIdx>& predec, const GraphT& g,
                               unsigned supstep, VertexIdx& totalAssigned, unsigned p)
    {
        for (const auto& node : bestNewReady) {
            ready.insert(node);
        }

        for (unsigned proc = 0; proc < p; ++proc) {
            for (const auto& node : bestNewAssignments[proc]) {
                nodeToProc[node] = proc;
                nodeToSupstep[node] = supstep;
                ready.erase(node);
                ++totalAssigned;

                for (const auto& succ : g.Children(node)) {
                    predec[succ]--;
                }
            }
        }
    }
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_GROW_LOCAL_AUTO_CORES_HPP
