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
 * \file kl_improver.h
 * \brief
 */

#ifndef OSP_KL_IMPROVER_H
#define OSP_KL_IMPROVER_H

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "kl_active_schedule.h"
#include "kl_util.h"
#include "passes/algorithms/osp/auxiliary/datastructures/heaps/pairing_heap.h"
#include "passes/algorithms/osp/bsp/model/util/compatible_processor_range.h"
#include "passes/algorithms/osp/bsp/scheduler/improvement_scheduler.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_util.h"

namespace npu::tile_fwk {
namespace osp {
struct KlParameter {
    double timeQuality_ = 0.8;
    double superstepRemoveStrength_ = 0.5;
    unsigned numParallelLoops_ = 4;

    unsigned maxInnerIterationsReset_ = 500;
    unsigned maxNoImprovementIterations_ = 50;

    constexpr static unsigned abortScatterNodesViolationThreshold_ = 500;
    constexpr static unsigned initialViolationThreshold_ = 250;

    unsigned maxNoVioaltionsRemovedBacktrackReset_;
    unsigned removeStepEpocs_;
    unsigned nodeMaxStepSelectionEpochs_;
    unsigned maxNoVioaltionsRemovedBacktrackForRemoveStepReset_;
    unsigned maxOuterIterations_;
    unsigned tryRemoveStepAfterNumOuterIterations_;
    unsigned minInnerIterReset_;

    unsigned threadMinRange_ = 8;
    unsigned threadRangeGap_ = 0;
};

template <typename VertexType>
struct KlUpdateInfo;

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize = 1, typename CostT = double>
class KlImprover : public ImprovementScheduler<GraphT> {
public:
    KlImprover(unsigned seed = 42) : ImprovementScheduler<GraphT>() { gen_ = std::mt19937(seed); }

    virtual ~KlImprover() = default;

    virtual ReturnStatus ImproveSchedule(BspSchedule<GraphT>& schedule) override
    {
        constexpr unsigned kMinProcessors = 2;
        if (schedule.GetInstance().NumberOfProcessors() < kMinProcessors) {
            return ReturnStatus::OSP_BEST_FOUND;
        }

        const unsigned numThreads = 1;

        threadDataVec_.resize(numThreads);
        threadFinishedVec_.assign(numThreads, true);

        SetParameters(schedule.GetInstance().NumberOfVertices());
        InitializeDatastructures(schedule);
        const CostT initialCost = activeSchedule_.GetCost();
        const unsigned numSteps = schedule.NumberOfSupersteps();

        SetStartStep(0, threadDataVec_[0]);
        threadDataVec_[0].endStep_ = (numSteps > 0) ? numSteps - 1 : 0;

        auto& threadData = this->threadDataVec_[0];
        threadData.activeScheduleData_.InitializeCost(activeSchedule_.GetCost());
        threadData.selectionStrategy_.Setup(threadData.startStep_, threadData.endStep_);
        RunLocalSearch(threadData);

        SynchronizeActiveSchedule(numThreads);

        if (initialCost > activeSchedule_.GetCost()) {
            activeSchedule_.WriteSchedule(schedule);
            CleanupDatastructures();
            return ReturnStatus::OSP_SUCCESS;
        } else {
            CleanupDatastructures();
            return ReturnStatus::OSP_BEST_FOUND;
        }
    }

    virtual ReturnStatus ImproveScheduleWithTimeLimit(BspSchedule<GraphT>& schedule) override
    {
        computeWithTimeLimit_ = true;
        return ImproveSchedule(schedule);
    }

    virtual void SetTimeQualityParameter(const double timeQuality) { this->parameters_.timeQuality_ = timeQuality; }

    virtual void SetSuperstepRemoveStrengthParameter(const double superstepRemoveStrength)
    {
        this->parameters_.superstepRemoveStrength_ = superstepRemoveStrength;
    }

protected:
    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool enableQuickMoves_ = true;
    constexpr static double epsilon_ = 1e-9;

    using VertexMemWeightT = osp::VMemwT<GraphT>;
    using VertexCommWeightT = osp::VCommwT<GraphT>;
    using VertexWorkWeightT = osp::VWorkwT<GraphT>;
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

    using KlMove = KlMoveStruct<CostT, VertexType>;
    using HeapDatastructure = MaxPairingHeap<VertexType, KlMove>;
    using ActiveScheduleT = KlActiveSchedule<GraphT, CostT>;
    using NodeSelectionContainerT = AdaptiveAffinityTable<GraphT, CostT, ActiveScheduleT, windowSize>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;

    struct ThreadSearchContext {
        unsigned threadId_ = 0;
        unsigned startStep_ = 0;
        unsigned endStep_ = 0;
        unsigned originalEndStep_ = 0;

        VectorVertexLockManager<VertexType> lockManager_;
        HeapDatastructure maxGainHeap_;
        NodeSelectionContainerT affinityTable_;
        std::vector<std::vector<CostT>> localAffinityTable_;
        RewardPenaltyStrategy<CostT, CommCostFunctionT, ActiveScheduleT> rewardPenaltyStrat_;
        VertexSelectionStrategy<GraphT, NodeSelectionContainerT, ActiveScheduleT> selectionStrategy_;
        ThreadLocalActiveScheduleData<GraphT, CostT> activeScheduleData_;

        double averageGain_ = 0.0;
        unsigned maxInnerIterations_ = 0;
        unsigned noImprovementIterationsReducePenalty_ = 0;
        unsigned minInnerIter_ = 0;
        unsigned noImprovementIterationsIncreaseInnerIter_ = 0;
        unsigned stepSelectionEpochCounter_ = 0;
        unsigned stepSelectionCounter_ = 0;
        unsigned stepToRemove_ = 0;
        unsigned localSearchStartStep_ = 0;
        unsigned unlockEdgeBacktrackCounter_ = 0;
        unsigned unlockEdgeBacktrackCounterReset_ = 0;
        unsigned maxNoVioaltionsRemovedBacktrack_ = 0;

        inline unsigned NumSteps() const { return endStep_ - startStep_ + 1; }

        inline unsigned StartIdx(const unsigned nodeStep) const
        {
            return nodeStep < startStep_ + windowSize ? windowSize - (nodeStep - startStep_) : 0;
        }

        inline unsigned EndIdx(unsigned nodeStep) const
        {
            return nodeStep + windowSize <= endStep_ ? windowRange_ : windowRange_ - (nodeStep + windowSize - endStep_);
        }
    };

    bool computeWithTimeLimit_ = false;

    BspSchedule<GraphT>* inputSchedule_;
    const GraphT* graph_;
    const BspInstance<GraphT>* instance_;

    CompatibleProcessorRange<GraphT> procRange_;

    KlParameter parameters_;
    std::mt19937 gen_;

    ActiveScheduleT activeSchedule_;
    CommCostFunctionT commCostF_;
    std::vector<ThreadSearchContext> threadDataVec_;
    std::vector<bool> threadFinishedVec_;

    inline unsigned RelStepIdx(const unsigned nodeStep, const unsigned moveStep) const
    {
        return (moveStep >= nodeStep) ? ((moveStep - nodeStep) + windowSize) : (windowSize - (nodeStep - moveStep));
    }

    inline bool IsCompatible(VertexType node, unsigned proc) const
    {
        return activeSchedule_.GetInstance().IsCompatible(node, proc);
    }

    void SetStartStep(const unsigned step, ThreadSearchContext& threadData)
    {
        constexpr unsigned kNoImprovementIterationsIncreaseInnerIter = 10;
        constexpr unsigned kReducePenaltyDivisor = 5;

        threadData.startStep_ = step;
        threadData.stepToRemove_ = step;
        threadData.stepSelectionCounter_ = step;

        threadData.averageGain_ = 0.0;
        threadData.maxInnerIterations_ = parameters_.maxInnerIterationsReset_;
        threadData.noImprovementIterationsReducePenalty_ = parameters_.maxNoImprovementIterations_ /
                                                           kReducePenaltyDivisor;
        threadData.minInnerIter_ = parameters_.minInnerIterReset_;
        threadData.stepSelectionEpochCounter_ = 0;
        threadData.noImprovementIterationsIncreaseInnerIter_ = kNoImprovementIterationsIncreaseInnerIter;
        threadData.unlockEdgeBacktrackCounterReset_ = 0;
        threadData.unlockEdgeBacktrackCounter_ = threadData.unlockEdgeBacktrackCounterReset_;
        threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackReset_;
    }

    KlMove GetBestMove(NodeSelectionContainerT& affinityTable, VectorVertexLockManager<VertexType>& lockManager,
                       HeapDatastructure& maxGainHeap)
    {
        // To introduce non-determinism and help escape local optima, if there are multiple moves with the same
        // top gain, we randomly select one. We check up to `local_max` ties.
        const unsigned localMax = 50;
        std::vector<VertexType> topGainNodes = maxGainHeap.GetTopKeys(localMax);
        if (topGainNodes.empty()) {
            // This case is guarded by the caller, but for safety:
            topGainNodes.push_back(maxGainHeap.Top());
        }

        std::uniform_int_distribution<size_t> dis(0, topGainNodes.size() - 1);
        const VertexType node = topGainNodes[dis(gen_)];

        KlMove bestMove = maxGainHeap.GetValue(node);
        maxGainHeap.Erase(node);
        lockManager.Lock(node);
        affinityTable.Remove(node);

        return bestMove;
    }

    inline void ProcessOtherStepsBestMove(const unsigned idx, const VertexType& node,
                                          const CostT affinityCurrentProcStep, CostT& maxGain, unsigned& maxProc,
                                          unsigned& maxStep,
                                          const std::vector<std::vector<CostT>>& affinityTableNode) const
    {
        for (const unsigned p : procRange_.CompatibleProcessorsVertex(node)) {
            const CostT gain = affinityCurrentProcStep - affinityTableNode[p][idx];
            if (gain > maxGain) {
                maxGain = gain;
                maxProc = p;
                maxStep = idx;
            }
        }
    }

    template <bool moveToSameSuperStep>
    KlMove ComputeBestMove(VertexType node, const std::vector<std::vector<CostT>>& affinityTableNode,
                           ThreadSearchContext& threadData)
    {
        const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
        const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);

        CostT maxGain = std::numeric_limits<CostT>::lowest();

        unsigned maxProc = std::numeric_limits<unsigned>::max();
        unsigned maxStep = std::numeric_limits<unsigned>::max();

        const CostT affinityCurrentProcStep = affinityTableNode[nodeProc][windowSize];

        unsigned idx = threadData.StartIdx(nodeStep);
        for (; idx < windowSize; idx++) {
            ProcessOtherStepsBestMove(idx, node, affinityCurrentProcStep, maxGain, maxProc, maxStep, affinityTableNode);
        }

        if constexpr (moveToSameSuperStep) {
            for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                if (proc == nodeProc) {
                    continue;
                }

                const CostT gain = affinityCurrentProcStep - affinityTableNode[proc][windowSize];
                if (gain > maxGain) {
                    maxGain = gain;
                    maxProc = proc;
                    maxStep = idx;
                }
            }
        }

        idx++;

        const unsigned bound = threadData.EndIdx(nodeStep);
        for (; idx < bound; idx++) {
            ProcessOtherStepsBestMove(idx, node, affinityCurrentProcStep, maxGain, maxProc, maxStep, affinityTableNode);
        }

        return KlMove(node, maxGain, nodeProc, nodeStep, maxProc, nodeStep + maxStep - windowSize);
    }

    CostT ComputeNodeProcAffinity(VertexWorkWeightT vertexWeight, VertexWorkWeightT maxWork,
                                  VertexWorkWeightT secondMaxWork, VertexWorkWeightT stepProcWork,
                                  unsigned maxWorkProcCount)
    {
        const bool isSoleMaxProcessor = (maxWorkProcCount == 1) && (maxWork == stepProcWork);
        return isSoleMaxProcessor ? std::min(vertexWeight, maxWork - secondMaxWork) : 0.0;
    }

    void UpdateMoveProcAffinity(VertexType node, unsigned nodeStep, unsigned moveProc,
                                VertexWorkWeightT weightAdjustment, VertexWorkWeightT prevMaxWork,
                                VertexWorkWeightT newMaxWeight, CostT prevNodeProcAffinity, CostT newNodeProcAffinity,
                                std::vector<std::vector<CostT>>& affinityTableNode)
    {
        if (activeSchedule_.AssignedProcessor(node) == moveProc || !IsCompatible(node, moveProc)) {
            return;
        }
        const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);
        const VertexWorkWeightT prevNewWeight = vertexWeight +
                                                activeSchedule_.GetStepProcessorWork(nodeStep, moveProc) +
                                                weightAdjustment;
        const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, prevNewWeight, prevNodeProcAffinity);
        const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, moveProc);
        const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
        affinityTableNode[moveProc][windowSize] += (otherAffinity - prevOtherAffinity);
    }

    void HandleSameStepSameNode(VertexType node, const KlMove& move,
                                const PreMoveWorkData<VertexWorkWeightT>& prevWorkData,
                                std::vector<std::vector<CostT>>& affinityTableNode, KlGainUpdateInfo& updateInfo);
    void HandleSameStepDifferentNodeMaxChanged(VertexType node, const KlMove& move, VertexWorkWeightT vertexWeight,
                                               VertexWorkWeightT prevMaxWork, VertexWorkWeightT newMaxWeight,
                                               unsigned idx, std::vector<std::vector<CostT>>& affinityTableNode);
    void HandleSameStepDifferentNodeMaxUnchanged(VertexType node, const KlMove& move, VertexWorkWeightT vertexWeight,
                                                 VertexWorkWeightT prevMaxWork, VertexWorkWeightT newMaxWeight,
                                                 unsigned idx, std::vector<std::vector<CostT>>& affinityTableNode);
    void HandleSameStepMove(VertexType node, const KlMove& move, const PreMoveWorkData<VertexWorkWeightT>& prevWorkData,
                            std::vector<std::vector<CostT>>& affinityTableNode, KlGainUpdateInfo& updateInfo);
    KlGainUpdateInfo UpdateNodeWorkAffinityAfterMove(VertexType node, KlMove move,
                                                     const PreMoveWorkData<VertexWorkWeightT>& prevWorkData,
                                                     std::vector<std::vector<CostT>>& affinityTableNode);
    void ProcessWorkUpdateStep(VertexType node, unsigned nodeStep, unsigned nodeProc, VertexWorkWeightT vertexWeight,
                               unsigned moveStep, unsigned moveProc, VertexWorkWeightT moveCorrectionNodeWeight,
                               const VertexWorkWeightT prevMoveStepMaxWork,
                               const VertexWorkWeightT prevMoveStepSecondMaxWork,
                               unsigned prevMoveStepMaxWorkProcessorCount, bool& updateStep, bool& updateEntireStep,
                               bool& fullUpdate, std::vector<std::vector<CostT>>& affinityTableNode);
    void UpdateNodeWorkAffinity(NodeSelectionContainerT& nodes, KlMove move,
                                const PreMoveWorkData<VertexWorkWeightT>& prevWorkData,
                                std::map<VertexType, KlGainUpdateInfo>& recomputeMaxGain);
    void UpdateBestMove(VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT& affinityTable,
                        ThreadSearchContext& threadData);
    void UpdateBestMove(VertexType node, unsigned step, NodeSelectionContainerT& affinityTable,
                        ThreadSearchContext& threadData);
    void UpdateMaxGain(KlMove move, std::map<VertexType, KlGainUpdateInfo>& recomputeMaxGain,
                       ThreadSearchContext& threadData);
    void ComputeWorkAffinity(VertexType node, std::vector<std::vector<CostT>>& affinityTableNode,
                             ThreadSearchContext& threadData);

    inline void RecomputeNodeMaxGain(VertexType node, NodeSelectionContainerT& affinityTable,
                                     ThreadSearchContext& threadData)
    {
        const auto bestMove = ComputeBestMove<true>(node, affinityTable[node], threadData);
        threadData.maxGainHeap_.Update(node, bestMove);
    }

    inline CostT ComputeSameStepAffinity(const VertexWorkWeightT& maxWorkForStep, const VertexWorkWeightT& newWeight,
                                         const CostT& nodeProcAffinity)
    {
        const CostT maxWorkAfterRemoval = static_cast<CostT>(maxWorkForStep) - nodeProcAffinity;
        if (newWeight > maxWorkAfterRemoval) {
            return newWeight - maxWorkAfterRemoval;
        }
        return 0.0;
    }

    inline CostT ApplyMove(KlMove move, ThreadSearchContext& threadData);

    enum class QuickMoveResult { kContinue, kSkip, kAbort };
    enum class InnerIterResult { kContinue, kBreak, kSkip };

    QuickMoveResult ProcessQuickMoveCandidate(VertexType nextNodeToMove, unsigned& innerIter,
                                              ThreadSearchContext& threadData,
                                              std::unordered_set<VertexType>& localLock,
                                              std::vector<VertexType>& quickMovesStack);

    void RunQuickMoves(unsigned& innerIter, ThreadSearchContext& threadData, const CostT changeInCost,
                       const VertexType bestMoveNode);

    InnerIterResult HandleViolationEscalation(unsigned& violationRemovedCount, unsigned& resetCounter,
                                              unsigned& innerIter, bool iterInitalFeasible,
                                              ThreadSearchContext& threadData);

    InnerIterResult HandleViolations(unsigned& violationRemovedCount, unsigned& resetCounter, unsigned& innerIter,
                                     bool iterInitalFeasible, ThreadSearchContext& threadData);

    bool ProcessInnerIteration(const KlMove& bestMove, std::vector<VertexType>& newNodes,
                               std::vector<VertexType>& unlockNodes,
                               std::map<VertexType, KlGainUpdateInfo>& recomputeMaxGain,
                               const PreMoveWorkData<VertexWorkWeightT>& prevWorkData, ThreadSearchContext& threadData);

    void RunInnerLoop(ThreadSearchContext& threadData, std::vector<VertexType>& newNodes,
                      std::vector<VertexType>& unlockNodes, std::map<VertexType, KlGainUpdateInfo>& recomputeMaxGain);

    bool ShouldTerminateOuterLoop(const std::chrono::time_point<std::chrono::high_resolution_clock>& startTime,
                                  unsigned& noImprovementIterCounter, CostT initialInnerIterCost,
                                  ThreadSearchContext& threadData);

    void RunLocalSearch(ThreadSearchContext& threadData);
    bool OtherThreadsFinished(const unsigned threadId);
    inline void UpdateAffinities(const KlMove& bestMove, ThreadSearchContext& threadData,
                                 std::map<VertexType, KlGainUpdateInfo>& recomputeMaxGain,
                                 std::vector<VertexType>& newNodes,
                                 const PreMoveWorkData<VertexWorkWeightT>& prevWorkData);

    inline bool BlockedEdgeStrategy(VertexType node, std::vector<VertexType>& unlockNodes,
                                    ThreadSearchContext& threadData);
    inline void AdjustLocalSearchParameters(unsigned outerIter, unsigned noImpCounter, ThreadSearchContext& threadData);
    bool IsLocalSearchBlocked(ThreadSearchContext& threadData);
    void SetParameters(VertexIdxT<GraphT> numNodes);
    void ResetInnerSearchStructures(ThreadSearchContext& threadData) const;
    void InitializeDatastructures(BspSchedule<GraphT>& schedule);
    void PrintHeap(HeapDatastructure& maxGainHeap) const;
    void CleanupDatastructures();
    void UpdateAvgGain(const CostT gain, const unsigned numIter, double& averageGain);
    void InsertGainHeap(ThreadSearchContext& threadData);
    void InsertNewNodesGainHeap(std::vector<VertexType>& newNodes, NodeSelectionContainerT& nodes,
                                ThreadSearchContext& threadData);

    inline void ComputeNodeAffinities(VertexType node, std::vector<std::vector<CostT>>& affinityTableNode,
                                      ThreadSearchContext& threadData)
    {
        ComputeWorkAffinity(node, affinityTableNode, threadData);
        commCostF_.ComputeCommAffinity(node, affinityTableNode, threadData.rewardPenaltyStrat_.penalty_,
                                       threadData.rewardPenaltyStrat_.reward_, threadData.startStep_,
                                       threadData.endStep_);
    }

    void SelectActiveNodes(ThreadSearchContext& threadData)
    {
        if (SelectNodesCheckRemoveSuperstep(threadData.stepToRemove_, threadData)) {
            activeSchedule_.SwapEmptyStepFwd(threadData.stepToRemove_, threadData.endStep_);
            threadData.endStep_--;
            threadData.localSearchStartStep_ = static_cast<unsigned>(
                threadData.activeScheduleData_.appliedMoves_.size());
            threadData.activeScheduleData_.UpdateCost(static_cast<CostT>(-1.0 * instance_->SynchronisationCosts()));

            if (threadData.activeScheduleData_.currentViolations_.size() > parameters_.initialViolationThreshold_) {
                activeSchedule_.RevertToBestSchedule(threadData.localSearchStartStep_, threadData.stepToRemove_,
                                                     commCostF_, threadData.activeScheduleData_, threadData.startStep_,
                                                     threadData.endStep_);
            } else {
                threadData.unlockEdgeBacktrackCounter_ = static_cast<unsigned>(
                    threadData.activeScheduleData_.currentViolations_.size());
                threadData.maxInnerIterations_ = std::max(threadData.unlockEdgeBacktrackCounter_ * 5u,
                                                          parameters_.maxInnerIterationsReset_);
                threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_
                                                                  .maxNoVioaltionsRemovedBacktrackForRemoveStepReset_;
                return;
            }
        }
        threadData.localSearchStartStep_ = 0;
        threadData.selectionStrategy_.SelectActiveNodes(threadData.affinityTable_, threadData.startStep_,
                                                        threadData.endStep_);
    }

    bool CheckRemoveSuperstep(unsigned step);
    bool SelectNodesCheckRemoveSuperstep(unsigned& step, ThreadSearchContext& threadData);
    bool ScatterNodesSuperstep(unsigned step, ThreadSearchContext& threadData);
    void SynchronizeActiveSchedule(const unsigned numThreads);
};
} // namespace osp
} // namespace npu::tile_fwk

#include "kl_improver.tpp"
#endif // OSP_KL_IMPROVER_HPP
