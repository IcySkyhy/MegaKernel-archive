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
 * \file kl_util.h
 * \brief
 */

#ifndef OSP_KL_UTIL_H
#define OSP_KL_UTIL_H

#include <unordered_set>

#include "kl_active_schedule.h"

namespace npu::tile_fwk {
namespace osp {
template <typename CostT, typename CommCostFunctionT, typename KlActiveScheduleT>
struct RewardPenaltyStrategy {
    KlActiveScheduleT* activeSchedule_;
    CostT maxWeight_;

    unsigned violationsThreshold_ = 0;
    CostT initialPenalty_ = 10.0;
    CostT penalty_ = 0;
    CostT reward_ = 0;

    void Initialize(KlActiveScheduleT& sched, const CostT maxComm, const CostT maxWork)
    {
        maxWeight_ = std::max(maxWork, maxComm * sched.GetInstance().CommunicationCosts());
        activeSchedule_ = &sched;
        initialPenalty_ = static_cast<CostT>(std::sqrt(maxWeight_));
    }

    void InitRewardPenalty(double multiplier = 1.0)
    {
        constexpr double kMaxMultiplier = 10.0;
        multiplier = std::min(multiplier, kMaxMultiplier);
        penalty_ = static_cast<CostT>(initialPenalty_ * multiplier);
        reward_ = static_cast<CostT>(maxWeight_ * multiplier);
    }
};

template <typename VertexType>
struct VectorVertexLockManager {
    std::vector<bool> lockedNodes_;

    void Initialize(size_t numNodes) { lockedNodes_.resize(numNodes); }

    void Lock(VertexType node) { lockedNodes_[node] = true; }

    void Unlock(VertexType node) { lockedNodes_[node] = false; }

    bool IsLocked(VertexType node) { return lockedNodes_[node]; }

    void Clear() { lockedNodes_.assign(lockedNodes_.size(), false); }
};

template <typename GraphT, typename CostT, typename KlActiveScheduleT, unsigned windowSize>
struct AdaptiveAffinityTable {
    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    using VertexType = VertexIdxT<GraphT>;

    void Initialize(const KlActiveScheduleT& sche, const std::size_t initialTableSize)
    {
        activeSchedule_ = &sche;
        graph_ = &(sche.GetInstance().GetComputationalDag());

        lastIdx_ = 0;

        nodeIsSelected_.resize(graph_->NumVertices());
        selectedNodesIdx_.resize(graph_->NumVertices());
        selectedNodes_.resize(initialTableSize);

        nodeIsSelected_.assign(nodeIsSelected_.size(), false);

        affinityTable_.resize(initialTableSize);
        const unsigned numProcs = sche.GetInstance().NumberOfProcessors();
        for (auto& table : affinityTable_) {
            table.resize(numProcs);
            for (auto& row : table) {
                row.resize(windowRange_);
            }
        }
    }

    inline std::vector<VertexType>& GetSelectedNodes() { return selectedNodes_; }
    inline const std::vector<VertexType>& GetSelectedNodes() const { return selectedNodes_; }
    inline size_t size() const { return lastIdx_ - gaps_.size(); }
    inline bool IsSelected(VertexType node) const { return nodeIsSelected_[node]; }
    inline const std::vector<size_t>& GetSelectedNodesIndices() const { return selectedNodesIdx_; }
    inline size_t GetSelectedNodesIdx(VertexType node) const { return selectedNodesIdx_[node]; }
    inline std::vector<std::vector<CostT>>& operator[](VertexType node)
    {
        return affinityTable_[selectedNodesIdx_[node]];
    }
    inline std::vector<std::vector<CostT>>& At(VertexType node) { return affinityTable_[selectedNodesIdx_[node]]; }
    inline const std::vector<std::vector<CostT>>& At(VertexType node) const
    {
        return affinityTable_[selectedNodesIdx_[node]];
    }
    inline std::vector<std::vector<CostT>>& GetAffinityTable(VertexType node)
    {
        return affinityTable_[selectedNodesIdx_[node]];
    }

    bool Insert(VertexType node)
    {
        if (nodeIsSelected_[node]) {
            return false; // Node is already in the table.
        }

        size_t insertLocation;
        if (!gaps_.empty()) {
            insertLocation = gaps_.back();
            gaps_.pop_back();
        } else {
            insertLocation = lastIdx_;

            if (insertLocation >= selectedNodes_.size()) {
                const size_t oldSize = selectedNodes_.size();
                const size_t newSize = std::min(oldSize * 2, static_cast<size_t>(graph_->NumVertices()));

                selectedNodes_.resize(newSize);
                affinityTable_.resize(newSize);

                const unsigned numProcs = activeSchedule_->GetInstance().NumberOfProcessors();
                for (size_t i = oldSize; i < newSize; ++i) {
                    affinityTable_[i].resize(numProcs);
                    for (auto& row : affinityTable_[i]) {
                        row.resize(windowRange_);
                    }
                }
            }
            lastIdx_++;
        }

        nodeIsSelected_[node] = true;
        selectedNodesIdx_[node] = insertLocation;
        selectedNodes_[insertLocation] = node;

        return true;
    }

    void Remove(VertexType node)
    {
        nodeIsSelected_[node] = false;
        gaps_.push_back(selectedNodesIdx_[node]);
    }

    void ResetNodeSelection()
    {
        nodeIsSelected_.assign(nodeIsSelected_.size(), false);
        gaps_.clear();
        lastIdx_ = 0;
    }

    void Clear()
    {
        nodeIsSelected_.clear();
        selectedNodesIdx_.clear();
        affinityTable_.clear();
        selectedNodes_.clear();
        gaps_.clear();
        lastIdx_ = 0;
    }

    void Trim()
    {
        while (!gaps_.empty() && lastIdx_ > 0) {
            size_t lastElementIdx = lastIdx_ - 1;
            if (!nodeIsSelected_[selectedNodes_[lastElementIdx]]) {
                lastIdx_--;
                continue;
            }

            size_t gapIdx = gaps_.back();
            gaps_.pop_back();

            if (gapIdx >= lastIdx_) {
                continue;
            }

            VertexType nodeToMove = selectedNodes_[lastElementIdx];
            std::swap(affinityTable_[gapIdx], affinityTable_[lastElementIdx]);
            std::swap(selectedNodes_[gapIdx], selectedNodes_[lastElementIdx]);
            selectedNodesIdx_[nodeToMove] = gapIdx;
            lastIdx_--;
        }
        gaps_.clear();
    }

private:
    const KlActiveScheduleT* activeSchedule_;
    const GraphT* graph_;

    std::vector<bool> nodeIsSelected_;
    std::vector<size_t> selectedNodesIdx_;

    std::vector<std::vector<std::vector<CostT>>> affinityTable_;
    std::vector<VertexType> selectedNodes_;

    std::vector<size_t> gaps_;
    size_t lastIdx_;
};

template <typename GraphT, typename ContainerT, typename KlActiveScheduleT>
struct VertexSelectionStrategy {
    using EdgeType = EdgeDescT<GraphT>;

    const KlActiveScheduleT* activeSchedule_;
    const GraphT* graph_;
    std::mt19937* gen_;
    std::size_t selectionThreshold_ = 0;
    unsigned strategyCounter_ = 0;

    std::vector<VertexIdxT<GraphT>> permutation_;
    std::size_t permutationIdx_;

    unsigned maxWorkCounter_ = 0;

    inline void Initialize(const KlActiveScheduleT& sche, std::mt19937& gen, const unsigned startStep,
                           const unsigned endStep)
    {
        activeSchedule_ = &sche;
        graph_ = &(sche.GetInstance().GetComputationalDag());
        gen_ = &gen;

        permutation_.reserve(graph_->NumVertices() / activeSchedule_->NumSteps() * (endStep - startStep));
    }

    inline void Setup(const unsigned startStep, const unsigned endStep)
    {
        maxWorkCounter_ = startStep;
        strategyCounter_ = 0;
        permutation_.clear();

        const unsigned numProcs = activeSchedule_->GetInstance().NumberOfProcessors();
        for (unsigned step = startStep; step <= endStep; ++step) {
            const auto& processorVertices = activeSchedule_->GetSetSchedule().GetProcessorStepVertices()[step];
            for (unsigned proc = 0; proc < numProcs; ++proc) {
                for (const auto node : processorVertices[proc]) {
                    permutation_.push_back(node);
                }
            }
        }

        permutationIdx_ = 0;
        std::shuffle(permutation_.begin(), permutation_.end(), *gen_);
    }

    inline void SelectActiveNodes(ContainerT& nodeSelection, const unsigned startStep, const unsigned endStep)
    {
        constexpr unsigned kPermutationStrategyCount = 3;
        constexpr unsigned kMaxWorkStrategyIndex = 4;
        constexpr unsigned kStrategyCycleLength = 5;

        if (strategyCounter_ < kPermutationStrategyCount) {
            SelectNodesPermutationThreshold(selectionThreshold_, nodeSelection);
        } else if (strategyCounter_ == kMaxWorkStrategyIndex) {
            SelectNodesMaxWorkProc(selectionThreshold_, nodeSelection, startStep, endStep);
        }

        strategyCounter_++;
        strategyCounter_ %= kStrategyCycleLength;
    }

    void SelectNodesViolations(ContainerT& nodeSelection, std::unordered_set<EdgeType>& currentViolations,
                               const unsigned startStep, const unsigned endStep)
    {
        for (const auto& edge : currentViolations) {
            const auto sourceV = Source(edge, *graph_);
            const auto targetV = Target(edge, *graph_);

            const unsigned sourceStep = activeSchedule_->AssignedSuperstep(sourceV);
            if (sourceStep >= startStep && sourceStep <= endStep) {
                nodeSelection.Insert(sourceV);
            }

            const unsigned targetStep = activeSchedule_->AssignedSuperstep(targetV);
            if (targetStep >= startStep && targetStep <= endStep) {
                nodeSelection.Insert(targetV);
            }
        }
    }

    void SelectNodesPermutationThreshold(const std::size_t& threshold, ContainerT& nodeSelection)
    {
        const size_t bound = std::min(threshold + permutationIdx_, permutation_.size());
        for (std::size_t i = permutationIdx_; i < bound; i++) {
            nodeSelection.Insert(permutation_[i]);
        }

        permutationIdx_ = bound;
        if (permutationIdx_ + threshold >= permutation_.size()) {
            permutationIdx_ = 0;
            std::shuffle(permutation_.begin(), permutation_.end(), *gen_);
        }
    }

    void SelectNodesMaxWorkProc(const std::size_t& threshold, ContainerT& nodeSelection, const unsigned startStep,
                                const unsigned endStep)
    {
        while (nodeSelection.size() < threshold) {
            if (maxWorkCounter_ > endStep) {
                maxWorkCounter_ = startStep; // wrap around
                break;                       // stop after one full pass
            }

            SelectNodesMaxWorkProcHelper(threshold - nodeSelection.size(), maxWorkCounter_, nodeSelection);
            maxWorkCounter_++;
        }
    }

    void SelectNodesMaxWorkProcHelper(const std::size_t& threshold, unsigned step, ContainerT& nodeSelection)
    {
        const unsigned numMaxWorkProc = activeSchedule_->workDatastructures_.stepMaxWorkProcessorCount_[step];
        for (unsigned idx = 0; idx < numMaxWorkProc; idx++) {
            const unsigned proc = activeSchedule_->workDatastructures_.stepProcessorWork_[step][idx].proc_;
            const std::unordered_set<VertexIdxT<GraphT>> stepProcVert = activeSchedule_->GetSetSchedule()
                                                                            .GetProcessorStepVertices()[step][proc];
            const size_t numInsert = std::min(threshold - nodeSelection.size(), stepProcVert.size());
            auto endIt = stepProcVert.begin();
            std::advance(endIt, numInsert);
            std::for_each(stepProcVert.begin(), endIt, [&](const auto& val) { nodeSelection.Insert(val); });
        }
    }
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_KL_UTIL_HPP
