/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OSP_KL_IMPROVER_TPP
#define OSP_KL_IMPROVER_TPP

namespace npu::tile_fwk {
namespace osp {
template <typename VertexType>
struct KlUpdateInfo {
    VertexType node_ = 0;

    bool fullUpdate_ = false;
    bool updateFromStep_ = false;
    bool updateToStep_ = false;
    bool updateEntireToStep_ = false;
    bool updateEntireFromStep_ = false;

    KlUpdateInfo() = default;

    KlUpdateInfo(VertexType n)
        : node_(n), fullUpdate_(false), updateEntireToStep_(false), updateEntireFromStep_(false) {}

    KlUpdateInfo(VertexType n, bool full)
        : node_(n), fullUpdate_(full), updateEntireToStep_(false), updateEntireFromStep_(false) {}
};

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::HandleSameStepSameNode(VertexType node,
                            const KlMove &move,
                            const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                            std::vector<std::vector<CostT>> &affinityTableNode,
                            KlGainUpdateInfo &updateInfo)
{
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
    const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);
    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);

    const VertexWorkWeightT prevMaxWork = prevWorkData.fromStepMaxWork_;
    const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(move.fromStep_);
    const VertexWorkWeightT newStepProcWork = activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc);
    const VertexWorkWeightT prevStepProcWork
        = (nodeProc == move.fromProc_) ? newStepProcWork + graph_->VertexWorkWeight(move.node_)
            : (nodeProc == move.toProc_) ? newStepProcWork - graph_->VertexWorkWeight(move.node_)
                                        : newStepProcWork;

    const CostT prevNodeProcAffinity = ComputeNodeProcAffinity(vertexWeight, prevMaxWork,
        prevWorkData.fromStepSecondMaxWork_, prevStepProcWork, prevWorkData.fromStepMaxWorkProcessorCount_);
    const CostT newNodeProcAffinity = ComputeNodeProcAffinity(vertexWeight, newMaxWeight,
        activeSchedule_.GetStepSecondMaxWork(move.fromStep_), newStepProcWork,
        activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep]);

    const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
    if (std::abs(diff) > epsilon_) {
        updateInfo.fullUpdate_ = true;
        affinityTableNode[nodeProc][windowSize] += diff;
    }

    if ((prevMaxWork != newMaxWeight) || updateInfo.fullUpdate_) {
        updateInfo.updateEntireFromStep_ = true;
        for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
            if ((proc == nodeProc) || (proc == move.fromProc_) || (proc == move.toProc_)) {
                continue;
            }
            const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
            const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, newWeight, prevNodeProcAffinity);
            const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
            affinityTableNode[proc][windowSize] += (otherAffinity - prevOtherAffinity);
        }
    }

    const VertexWorkWeightT moveNodeWeight = graph_->VertexWorkWeight(move.node_);
    UpdateMoveProcAffinity(node, nodeStep, move.fromProc_, moveNodeWeight,
                            prevMaxWork, newMaxWeight, prevNodeProcAffinity, newNodeProcAffinity, affinityTableNode);
    UpdateMoveProcAffinity(node, nodeStep, move.toProc_, -moveNodeWeight,
                            prevMaxWork, newMaxWeight, prevNodeProcAffinity, newNodeProcAffinity, affinityTableNode);
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::HandleSameStepDifferentNodeMaxChanged(VertexType node,
                                            const KlMove &move,
                                            VertexWorkWeightT vertexWeight,
                                            VertexWorkWeightT prevMaxWork,
                                            VertexWorkWeightT newMaxWeight,
                                            unsigned idx,
                                            std::vector<std::vector<CostT>> &affinityTableNode)
{
    for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
        const VertexWorkWeightT newWeight
            = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep_, proc);
        if (proc == move.fromProc_) {
            const VertexWorkWeightT prevNewWeight
                = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep_, proc)
                    + graph_->VertexWorkWeight(move.node_);
            const CostT prevAffinity = prevMaxWork < prevNewWeight ? static_cast<CostT>(prevNewWeight)
                                                                            - static_cast<CostT>(prevMaxWork)
                                                                    : 0.0;
            const CostT newAffinity = newMaxWeight < newWeight
                                            ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight)
                                            : 0.0;
            affinityTableNode[proc][idx] += newAffinity - prevAffinity;
        } else if (proc == move.toProc_) {
            const VertexWorkWeightT prevNewWeight = vertexWeight
                                                    + activeSchedule_.GetStepProcessorWork(move.toStep_, proc)
                                                    - graph_->VertexWorkWeight(move.node_);
            const CostT prevAffinity = prevMaxWork < prevNewWeight ? static_cast<CostT>(prevNewWeight)
                                                                            - static_cast<CostT>(prevMaxWork)
                                                                    : 0.0;
            const CostT newAffinity = newMaxWeight < newWeight
                                            ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight)
                                            : 0.0;
            affinityTableNode[proc][idx] += newAffinity - prevAffinity;
        } else {
            const CostT prevAffinity = prevMaxWork < newWeight
                                            ? static_cast<CostT>(newWeight) - static_cast<CostT>(prevMaxWork)
                                            : 0.0;
            const CostT newAffinity = newMaxWeight < newWeight
                                            ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight)
                                            : 0.0;
            affinityTableNode[proc][idx] += newAffinity - prevAffinity;
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::HandleSameStepDifferentNodeMaxUnchanged(VertexType node,
                                                const KlMove &move,
                                                VertexWorkWeightT vertexWeight,
                                                VertexWorkWeightT prevMaxWork,
                                                VertexWorkWeightT newMaxWeight,
                                                unsigned idx,
                                                std::vector<std::vector<CostT>> &affinityTableNode)
{
    if (IsCompatible(node, move.fromProc_)) {
        const VertexWorkWeightT fromNewWeight
            = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep_, move.fromProc_);
        const VertexWorkWeightT fromPrevNewWeight = fromNewWeight + graph_->VertexWorkWeight(move.node_);
        const CostT fromPrevAffinity = prevMaxWork < fromPrevNewWeight ? static_cast<CostT>(fromPrevNewWeight)
                                                                                - static_cast<CostT>(prevMaxWork)
                                                                        : 0.0;

        const CostT fromNewAffinity = newMaxWeight < fromNewWeight ? static_cast<CostT>(fromNewWeight)
                                                                            - static_cast<CostT>(newMaxWeight)
                                                                    : 0.0;
        affinityTableNode[move.fromProc_][idx] += fromNewAffinity - fromPrevAffinity;
    }

    if (IsCompatible(node, move.toProc_)) {
        const VertexWorkWeightT toNewWeight
            = vertexWeight + activeSchedule_.GetStepProcessorWork(move.toStep_, move.toProc_);
        const VertexWorkWeightT toPrevNewWeight = toNewWeight - graph_->VertexWorkWeight(move.node_);
        const CostT toPrevAffinity = prevMaxWork < toPrevNewWeight ? static_cast<CostT>(toPrevNewWeight)
                                                                            - static_cast<CostT>(prevMaxWork)
                                                                    : 0.0;

        const CostT toNewAffinity = newMaxWeight < toNewWeight
                                        ? static_cast<CostT>(toNewWeight) - static_cast<CostT>(newMaxWeight)
                                        : 0.0;
        affinityTableNode[move.toProc_][idx] += toNewAffinity - toPrevAffinity;
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::HandleSameStepMove(VertexType node,
                        const KlMove &move,
                        const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                        std::vector<std::vector<CostT>> &affinityTableNode,
                        KlGainUpdateInfo &updateInfo)
{
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
    const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);

    const unsigned lowerBound = move.fromStep_ > windowSize ? move.fromStep_ - windowSize : 0;
    if (!(lowerBound <= nodeStep && nodeStep <= move.fromStep_ + windowSize)) {
        return;
    }

    updateInfo.updateFromStep_ = true;
    updateInfo.updateToStep_ = true;

    if (nodeStep == move.fromStep_) {
        HandleSameStepSameNode(node, move, prevWorkData, affinityTableNode, updateInfo);
    } else {
        const VertexWorkWeightT prevMaxWork = prevWorkData.fromStepMaxWork_;
        const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(move.fromStep_);
        const unsigned idx = RelStepIdx(nodeStep, move.fromStep_);
        if (prevMaxWork != newMaxWeight) {
            updateInfo.updateEntireFromStep_ = true;
            HandleSameStepDifferentNodeMaxChanged(
                node, move, vertexWeight, prevMaxWork, newMaxWeight, idx, affinityTableNode);
        } else {
            HandleSameStepDifferentNodeMaxUnchanged(
                node, move, vertexWeight, prevMaxWork, newMaxWeight, idx, affinityTableNode);
        }
    }
}


template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
typename KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::KlGainUpdateInfo
KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateNodeWorkAffinityAfterMove(VertexType node,
                                                    KlMove move,
                                                    const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                                                    std::vector<std::vector<CostT>> &affinityTableNode)
{
    KlGainUpdateInfo updateInfo(node);

    if (move.fromStep_ == move.toStep_) {
        HandleSameStepMove(node, move, prevWorkData, affinityTableNode, updateInfo);
    } else {
        const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
        const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
        const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);
        ProcessWorkUpdateStep(
            node, nodeStep, nodeProc, vertexWeight, move.fromStep_, move.fromProc_,
            graph_->VertexWorkWeight(move.node_), prevWorkData.fromStepMaxWork_,
            prevWorkData.fromStepSecondMaxWork_, prevWorkData.fromStepMaxWorkProcessorCount_,
            updateInfo.updateFromStep_, updateInfo.updateEntireFromStep_,
            updateInfo.fullUpdate_, affinityTableNode);
        ProcessWorkUpdateStep(
            node, nodeStep, nodeProc, vertexWeight, move.toStep_, move.toProc_,
            -graph_->VertexWorkWeight(move.node_), prevWorkData.toStepMaxWork_,
            prevWorkData.toStepSecondMaxWork_, prevWorkData.toStepMaxWorkProcessorCount_,
            updateInfo.updateToStep_, updateInfo.updateEntireToStep_,
            updateInfo.fullUpdate_, affinityTableNode);
    }

    return updateInfo;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
CostT KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ApplyMove(KlMove move, ThreadSearchContext &threadData)
{
    activeSchedule_.ApplyMove(move, threadData.activeScheduleData_);
    commCostF_.UpdateDatastructureAfterMove(move, threadData.startStep_, threadData.endStep_);
    CostT changeInCost = -move.gain_;
    changeInCost += static_cast<CostT>(threadData.activeScheduleData_.resolvedViolations_.size())
                    * threadData.rewardPenaltyStrat_.reward_;
    changeInCost -= static_cast<CostT>(threadData.activeScheduleData_.newViolations_.size())
        * threadData.rewardPenaltyStrat_.penalty_;

    threadData.activeScheduleData_.UpdateCost(changeInCost);

    return changeInCost;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
typename KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::QuickMoveResult
KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ProcessQuickMoveCandidate(
    VertexType nextNodeToMove, unsigned &innerIter, ThreadSearchContext &threadData,
    std::unordered_set<VertexType> &localLock, std::vector<VertexType> &quickMovesStack)
{
    threadData.rewardPenaltyStrat_.InitRewardPenalty(
        static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);
    ComputeNodeAffinities(nextNodeToMove, threadData.localAffinityTable_, threadData);
    KlMove bestQuickMove = ComputeBestMove<true>(nextNodeToMove, threadData.localAffinityTable_, threadData);

    localLock.insert(nextNodeToMove);
    if (bestQuickMove.gain_ <= std::numeric_limits<CostT>::lowest()) {
        return QuickMoveResult::kContinue;
    }

    ApplyMove(bestQuickMove, threadData);
    innerIter++;

    if (threadData.activeScheduleData_.newViolations_.size() > 0) {
        for (const auto &keyValuePair : threadData.activeScheduleData_.newViolations_) {
            const auto &key = keyValuePair.first;
            if (localLock.find(key) != localLock.end()) {
                return QuickMoveResult::kAbort;
            }
            quickMovesStack.push_back(key);
        }
        return QuickMoveResult::kContinue;
    }

    if (threadData.activeScheduleData_.feasible_) {
        return QuickMoveResult::kAbort;
    }

    return QuickMoveResult::kContinue;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::RunQuickMoves(unsigned &innerIter,
                    ThreadSearchContext &threadData,
                    const CostT changeInCost,
                    const VertexType bestMoveNode)
{
    innerIter++;

    const size_t numAppliedMoves = threadData.activeScheduleData_.appliedMoves_.size() - 1;
    const CostT savedCost = threadData.activeScheduleData_.cost_ - changeInCost;

    std::unordered_set<VertexType> localLock;
    localLock.insert(bestMoveNode);
    std::vector<VertexType> quickMovesStack;
    quickMovesStack.reserve(10 + threadData.activeScheduleData_.newViolations_.size() * 2);

    for (const auto &keyValuePair : threadData.activeScheduleData_.newViolations_) {
        const auto &key = keyValuePair.first;
        quickMovesStack.push_back(key);
    }

    while (quickMovesStack.size() > 0) {
        auto nextNodeToMove = quickMovesStack.back();
        quickMovesStack.pop_back();

        QuickMoveResult result = ProcessQuickMoveCandidate(
            nextNodeToMove, innerIter, threadData, localLock, quickMovesStack);
        if (result == QuickMoveResult::kAbort) {
            break;
        }
    }

    if (!threadData.activeScheduleData_.feasible_) {
        activeSchedule_.RevertScheduleToBound(
            numAppliedMoves, savedCost, true, commCostF_,
            threadData.activeScheduleData_, threadData.startStep_, threadData.endStep_);
    }

    threadData.affinityTable_.Trim();
    threadData.maxGainHeap_.Clear();
    threadData.rewardPenaltyStrat_.InitRewardPenalty(1.0);
    InsertGainHeap(threadData);    // Re-initialize the heap with the current state
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
typename KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::InnerIterResult
KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::HandleViolationEscalation(
    unsigned &violationRemovedCount, unsigned &resetCounter, unsigned &innerIter,
    bool iterInitalFeasible, ThreadSearchContext &threadData)
{
    violationRemovedCount++;
    if (violationRemovedCount <= 3) {
        return InnerIterResult::kContinue;
    }

    if (resetCounter >= threadData.maxNoVioaltionsRemovedBacktrack_
        || (iterInitalFeasible && threadData.activeScheduleData_.cost_ >= threadData.activeScheduleData_.bestCost_))
    {
        return InnerIterResult::kBreak;
    }

    threadData.affinityTable_.ResetNodeSelection();
    threadData.maxGainHeap_.Clear();
    threadData.lockManager_.Clear();
    threadData.selectionStrategy_.SelectNodesViolations(
        threadData.affinityTable_,
        threadData.activeScheduleData_.currentViolations_,
        threadData.startStep_,
        threadData.endStep_);
    threadData.rewardPenaltyStrat_.InitRewardPenalty(
        static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()));
    InsertGainHeap(threadData);
    resetCounter++;
    innerIter++;
    return InnerIterResult::kSkip;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
typename KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::InnerIterResult
KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::HandleViolations(
    unsigned &violationRemovedCount, unsigned &resetCounter, unsigned &innerIter,
    bool iterInitalFeasible, ThreadSearchContext &threadData)
{
    if (threadData.activeScheduleData_.currentViolations_.size() == 0) {
        return InnerIterResult::kContinue;
    }

    if (threadData.activeScheduleData_.resolvedViolations_.size() > 0) {
        violationRemovedCount = 0;
        return InnerIterResult::kContinue;
    }
    return HandleViolationEscalation(violationRemovedCount, resetCounter, innerIter, iterInitalFeasible, threadData);
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ProcessInnerIteration(const KlMove &bestMove,
                            std::vector<VertexType> &newNodes,
                            std::vector<VertexType> &unlockNodes,
                            std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain,
                            const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                            ThreadSearchContext &threadData)
{
    if (IsLocalSearchBlocked(threadData)) {
        if (not BlockedEdgeStrategy(bestMove.node_, unlockNodes, threadData)) {
            return false;
        }
    }

    threadData.affinityTable_.Trim();
    UpdateAffinities(bestMove, threadData, recomputeMaxGain, newNodes, prevWorkData);

    for (const auto v : unlockNodes) {
        threadData.lockManager_.Unlock(v);
    }
    newNodes.insert(newNodes.end(), unlockNodes.begin(), unlockNodes.end());
    unlockNodes.clear();

    UpdateMaxGain(bestMove, recomputeMaxGain, threadData);
    InsertNewNodesGainHeap(newNodes, threadData.affinityTable_, threadData);

    recomputeMaxGain.clear();
    newNodes.clear();
    return true;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::RunInnerLoop(ThreadSearchContext &threadData,
                    std::vector<VertexType> &newNodes,
                    std::vector<VertexType> &unlockNodes,
                    std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain)
{
    unsigned innerIter = 0;
    unsigned violationRemovedCount = 0;
    unsigned resetCounter = 0;
    bool iterInitalFeasible = threadData.activeScheduleData_.feasible_;

    while (innerIter < threadData.maxInnerIterations_ && threadData.maxGainHeap_.size() > 0) {
        KlMove bestMove = GetBestMove(threadData.affinityTable_, threadData.lockManager_, threadData.maxGainHeap_);
        if (bestMove.gain_ <= std::numeric_limits<CostT>::lowest()) {
            break;
        }
        UpdateAvgGain(bestMove.gain_, innerIter, threadData.averageGain_);

        if (innerIter > threadData.minInnerIter_ && threadData.averageGain_ < 0.0) {
            break;
        }

        const auto prevWorkData = activeSchedule_.GetPreMoveWorkData(bestMove);
        const typename CommCostFunctionT::PreMoveCommDataT prevCommData = commCostF_.GetPreMoveCommData(bestMove);
        const CostT changeInCost = ApplyMove(bestMove, threadData);

        if constexpr (enableQuickMoves_) {
            if (iterInitalFeasible && threadData.activeScheduleData_.newViolations_.size() > 0) {
                RunQuickMoves(innerIter, threadData, changeInCost, bestMove.node_);
                continue;
            }
        }

        InnerIterResult violationResult = HandleViolations(
            violationRemovedCount, resetCounter, innerIter, iterInitalFeasible, threadData);
        if (violationResult == InnerIterResult::kBreak) {
            break;
        }
        if (violationResult == InnerIterResult::kSkip) {
            continue;
        }

        if (!ProcessInnerIteration(bestMove, newNodes, unlockNodes, recomputeMaxGain, prevWorkData, threadData)) {
            break;
        }

        innerIter++;
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ShouldTerminateOuterLoop(
    const std::chrono::time_point<std::chrono::high_resolution_clock> &startTime,
                                unsigned &noImprovementIterCounter,
                                CostT initialInnerIterCost,
                                ThreadSearchContext &threadData)
{
    if (computeWithTimeLimit_) {
        auto finishTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(finishTime - startTime).count();
        if (duration > ImprovementScheduler<GraphT>::timeLimitSeconds_) {
            return true;
        }
    }

    if (OtherThreadsFinished(threadData.threadId_)) {
        return true;
    }

    if (initialInnerIterCost <= threadData.activeScheduleData_.cost_) {
        noImprovementIterCounter++;
        if (noImprovementIterCounter >= parameters_.maxNoImprovementIterations_) {
            return true;
        }
    } else {
        noImprovementIterCounter = 0;
    }

    return false;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::RunLocalSearch(ThreadSearchContext &threadData)
{
    std::vector<VertexType> newNodes;
    std::vector<VertexType> unlockNodes;
    std::map<VertexType, KlGainUpdateInfo> recomputeMaxGain;

    const auto startTime = std::chrono::high_resolution_clock::now();
    unsigned noImprovementIterCounter = 0;

    for (unsigned outerIter = 0; outerIter < parameters_.maxOuterIterations_; outerIter++) {
        CostT initialInnerIterCost = threadData.activeScheduleData_.cost_;

        ResetInnerSearchStructures(threadData);
        SelectActiveNodes(threadData);
        threadData.rewardPenaltyStrat_.InitRewardPenalty(
            static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);
        InsertGainHeap(threadData);

        RunInnerLoop(threadData, newNodes, unlockNodes, recomputeMaxGain);

        activeSchedule_.RevertToBestSchedule(threadData.localSearchStartStep_,
                                                threadData.stepToRemove_,
                                                commCostF_,
                                                threadData.activeScheduleData_,
                                                threadData.startStep_,
                                                threadData.endStep_);

        if (ShouldTerminateOuterLoop(startTime, noImprovementIterCounter, initialInnerIterCost, threadData)) {
            break;
        }

        AdjustLocalSearchParameters(outerIter, noImprovementIterCounter, threadData);
    }

    threadFinishedVec_[threadData.threadId_] = true;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::OtherThreadsFinished(const unsigned threadId)
{
    const size_t numThreads = threadFinishedVec_.size();
    if (numThreads == 1) {
        return false;
    }

    for (size_t i = 0; i < numThreads; i++) {
        if (i != threadId && !threadFinishedVec_[i]) {
            return false;
        }
    }
    return true;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateAffinities(const KlMove &bestMove,
                                ThreadSearchContext &threadData,
                                std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain,
                                std::vector<VertexType> &newNodes,
                                const PreMoveWorkData<VertexWorkWeightT> &prevWorkData)
{
    UpdateNodeWorkAffinity(threadData.affinityTable_, bestMove, prevWorkData, recomputeMaxGain);
    commCostF_.UpdateNodeCommAffinity(bestMove, threadData, threadData.rewardPenaltyStrat_.penalty_,
                                        threadData.rewardPenaltyStrat_.reward_, recomputeMaxGain, newNodes);
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::BlockedEdgeStrategy(
    VertexType node, std::vector<VertexType> &unlockNodes, ThreadSearchContext &threadData)
{
    if (threadData.unlockEdgeBacktrackCounter_ > 1) {
        for (const auto vertexEdgePair : threadData.activeScheduleData_.newViolations_) {
            const auto &e = vertexEdgePair.second;
            const auto sourceV = Source(e, *graph_);
            const auto targetV = Target(e, *graph_);

            if (node == sourceV && threadData.lockManager_.IsLocked(targetV)) {
                unlockNodes.push_back(targetV);
            } else if (node == targetV && threadData.lockManager_.IsLocked(sourceV)) {
                unlockNodes.push_back(sourceV);
            }
        }

        threadData.unlockEdgeBacktrackCounter_--;
        return true;
    } else {
        return false;    // or reset local search and initialize with violating nodes
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::AdjustLocalSearchParameters(
    unsigned outerIter, unsigned noImpCounter, ThreadSearchContext &threadData)
{
    if (noImpCounter >= threadData.noImprovementIterationsReducePenalty_
        && threadData.rewardPenaltyStrat_.initialPenalty_ > 1.0)
{
        threadData.rewardPenaltyStrat_.initialPenalty_
            = static_cast<CostT>(std::floor(std::sqrt(threadData.rewardPenaltyStrat_.initialPenalty_)));
        threadData.unlockEdgeBacktrackCounterReset_ += 1;
        threadData.noImprovementIterationsReducePenalty_ += 15;
    }

    if (parameters_.tryRemoveStepAfterNumOuterIterations_ > 0
        && ((outerIter + 1) % parameters_.tryRemoveStepAfterNumOuterIterations_) == 0)
    {
        threadData.stepSelectionEpochCounter_ = 0;
    }

    if (noImpCounter >= threadData.noImprovementIterationsIncreaseInnerIter_) {
        threadData.minInnerIter_ = static_cast<unsigned>(std::ceil(threadData.minInnerIter_ * 2.2));
        threadData.noImprovementIterationsIncreaseInnerIter_ += 20;
    }
}



template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ScatterNodesSuperstep(
    unsigned step, ThreadSearchContext &threadData)
{
    bool abort = false;

    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
        const std::vector<VertexType> stepProcNodeVec(
            activeSchedule_.GetSetSchedule().GetProcessorStepVertices()[step][proc].begin(),
            activeSchedule_.GetSetSchedule().GetProcessorStepVertices()[step][proc].end());
        for (const auto &node : stepProcNodeVec) {
            threadData.rewardPenaltyStrat_.InitRewardPenalty(
                static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);
            ComputeNodeAffinities(node, threadData.localAffinityTable_, threadData);
            KlMove bestMove = ComputeBestMove<false>(node, threadData.localAffinityTable_, threadData);

            if (bestMove.gain_ <= std::numeric_limits<double>::lowest()) {
                abort = true;
                break;
            }

            ApplyMove(bestMove, threadData);
            if (threadData.activeScheduleData_.currentViolations_.size()
                > parameters_.abortScatterNodesViolationThreshold_)
            {
                abort = true;
                break;
            }

            threadData.affinityTable_.Insert(node);
            if (threadData.activeScheduleData_.newViolations_.size() > 0) {
                for (const auto &vertexEdgePair : threadData.activeScheduleData_.newViolations_) {
                    const auto &vertex = vertexEdgePair.first;
                    threadData.affinityTable_.Insert(vertex);
                }
            }
        }

        if (abort) {
            break;
        }
    }

    if (abort) {
        activeSchedule_.RevertToBestSchedule(
            0, 0, commCostF_, threadData.activeScheduleData_, threadData.startStep_, threadData.endStep_);
        threadData.affinityTable_.ResetNodeSelection();
        return false;
    }
    return true;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::SynchronizeActiveSchedule(const unsigned numThreads)
{
    if (numThreads == 1) {    // single thread case
        activeSchedule_.SetCost(threadDataVec_[0].activeScheduleData_.cost_);
        activeSchedule_.GetVectorSchedule().NumberOfSupersteps() = threadDataVec_[0].NumSteps();
        return;
    }

    unsigned writeCursor = threadDataVec_[0].endStep_ + 1;
    for (unsigned i = 1; i < numThreads; ++i) {
        auto &thread = threadDataVec_[i];
        if (thread.startStep_ <= thread.endStep_) {
            for (unsigned j = thread.startStep_; j <= thread.endStep_; ++j) {
                if (j != writeCursor) {
                    activeSchedule_.SwapSteps(j, writeCursor);
                }
                writeCursor++;
            }
        }
    }
    activeSchedule_.GetVectorSchedule().NumberOfSupersteps() = writeCursor;
    const CostT newCost = commCostF_.ComputeScheduleCost();
    activeSchedule_.SetCost(newCost);
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::SetParameters(VertexIdxT<GraphT> numNodes)
{
    const unsigned logNumNodes = (numNodes > 1) ? static_cast<unsigned>(std::log(numNodes)) : 1;

    // Total number of outer iterations. Proportional to sqrt N.
    parameters_.maxOuterIterations_
        = static_cast<unsigned>(
            std::sqrt(numNodes) * (parameters_.timeQuality_ * 10.0) / parameters_.numParallelLoops_);

    // Number of times to reset the search for violations before giving up.
    parameters_.maxNoVioaltionsRemovedBacktrackReset_ = parameters_.timeQuality_ < 0.75  ? 1
                                                        : parameters_.timeQuality_ < 1.0 ? 2
                                                                                         : 3;

    // Parameters for the superstep removal heuristic.
    parameters_.maxNoVioaltionsRemovedBacktrackForRemoveStepReset_
        = 3 + static_cast<unsigned>(parameters_.superstepRemoveStrength_ * 7);
    parameters_.nodeMaxStepSelectionEpochs_ = parameters_.superstepRemoveStrength_ < 0.75  ? 1
                                              : parameters_.superstepRemoveStrength_ < 1.0 ? 2
                                                                                           : 3;
    parameters_.removeStepEpocs_ = static_cast<unsigned>(parameters_.superstepRemoveStrength_ * 4.0);

    parameters_.minInnerIterReset_
        = static_cast<unsigned>(logNumNodes + logNumNodes * (1.0 + parameters_.timeQuality_));

    if (parameters_.removeStepEpocs_ > 0) {
        parameters_.tryRemoveStepAfterNumOuterIterations_
            = parameters_.maxOuterIterations_ / parameters_.removeStepEpocs_;
    } else {
        // Effectively disable superstep removal if remove_step_epocs is 0.
        parameters_.tryRemoveStepAfterNumOuterIterations_ = parameters_.maxOuterIterations_ + 1;
    }

    unsigned i = 0;
    for (auto &thread : threadDataVec_) {
        thread.threadId_ = i++;
        // The number of nodes to consider in each inner iteration. Proportional to log(N).
        thread.selectionStrategy_.selectionThreshold_
            = static_cast<std::size_t>(std::ceil(parameters_.timeQuality_ * 10 * logNumNodes + logNumNodes));
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateNodeWorkAffinity(
    NodeSelectionContainerT &nodes,
    KlMove move,
    const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
    std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain)
{
    const size_t activeCount = nodes.size();

    for (size_t i = 0; i < activeCount; ++i) {
        const VertexType node = nodes.GetSelectedNodes()[i];

        KlGainUpdateInfo updateInfo = UpdateNodeWorkAffinityAfterMove(node, move, prevWorkData, nodes.At(node));
        if (updateInfo.updateFromStep_ || updateInfo.updateToStep_) {
            recomputeMaxGain[node] = updateInfo;
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateMaxGain(
    KlMove move, std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain, ThreadSearchContext &threadData)
{
    for (auto &pair : recomputeMaxGain) {
        if (pair.second.fullUpdate_) {
            RecomputeNodeMaxGain(pair.first, threadData.affinityTable_, threadData);
        } else {
            if (pair.second.updateEntireFromStep_) {
                UpdateBestMove(pair.first, move.fromStep_, threadData.affinityTable_, threadData);
            } else if (pair.second.updateFromStep_ && IsCompatible(pair.first, move.fromProc_)) {
                UpdateBestMove(pair.first, move.fromStep_, move.fromProc_, threadData.affinityTable_, threadData);
            }

            if (move.fromStep_ != move.toStep_ || not pair.second.updateEntireFromStep_) {
                if (pair.second.updateEntireToStep_) {
                    UpdateBestMove(pair.first, move.toStep_, threadData.affinityTable_, threadData);
                } else if (pair.second.updateToStep_ && IsCompatible(pair.first, move.toProc_)) {
                    UpdateBestMove(pair.first, move.toStep_, move.toProc_, threadData.affinityTable_, threadData);
                }
            }
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ComputeWorkAffinity(
    VertexType node, std::vector<std::vector<CostT>> &affinityTableNode, ThreadSearchContext &threadData)
{
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
    const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);

    unsigned step = (nodeStep > windowSize) ? (nodeStep - windowSize) : 0;
    for (unsigned idx = threadData.StartIdx(nodeStep); idx < threadData.EndIdx(nodeStep); ++idx, ++step) {
        if (idx == windowSize) {
            continue;
        }

        const CostT maxWorkForStep = static_cast<CostT>(activeSchedule_.GetStepMaxWork(step));

        for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
            const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(step, proc);
            const CostT workDiff = static_cast<CostT>(newWeight) - maxWorkForStep;
            affinityTableNode[proc][idx] = std::max(0.0, workDiff);
        }
    }

    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const VertexWorkWeightT maxWorkForStep = activeSchedule_.GetStepMaxWork(nodeStep);
    const bool isSoleMaxProcessor = (activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                    && (maxWorkForStep == activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc));

    const CostT nodeProcAffinity
        = isSoleMaxProcessor
            ? std::min(vertexWeight, maxWorkForStep - activeSchedule_.GetStepSecondMaxWork(nodeStep))
            : 0.0;
    affinityTableNode[nodeProc][windowSize] = nodeProcAffinity;

    for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
        if (proc == nodeProc) {
            continue;
        }

        const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
        affinityTableNode[proc][windowSize] = ComputeSameStepAffinity(maxWorkForStep, newWeight, nodeProcAffinity);
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ProcessWorkUpdateStep(
    VertexType node, unsigned nodeStep, unsigned nodeProc, VertexWorkWeightT vertexWeight,
    unsigned moveStep, unsigned moveProc, VertexWorkWeightT moveCorrectionNodeWeight,
    const VertexWorkWeightT prevMoveStepMaxWork,
    const VertexWorkWeightT prevMoveStepSecondMaxWork,
    unsigned prevMoveStepMaxWorkProcessorCount, bool &updateStep,
    bool &updateEntireStep, bool &fullUpdate,
    std::vector<std::vector<CostT>> &affinityTableNode)
{
    const unsigned lowerBound = moveStep > windowSize ? moveStep - windowSize : 0;
    if (lowerBound <= nodeStep && nodeStep <= moveStep + windowSize) {
        updateStep = true;
        if (nodeStep == moveStep) {
            const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(moveStep);
            const VertexWorkWeightT newSecondMaxWeight = activeSchedule_.GetStepSecondMaxWork(moveStep);
            const VertexWorkWeightT newStepProcWork = activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc);

            const VertexWorkWeightT prevStepProcWork
                = (nodeProc == moveProc) ? newStepProcWork + moveCorrectionNodeWeight
                                         : newStepProcWork;
            const bool prevIsSoleMaxProcessor = (prevMoveStepMaxWorkProcessorCount == 1)
                                                && (prevMoveStepMaxWork == prevStepProcWork);
            const CostT prevNodeProcAffinity
                = prevIsSoleMaxProcessor
                    ? std::min(vertexWeight, prevMoveStepMaxWork - prevMoveStepSecondMaxWork)
                    : 0.0;

            const bool newIsSoleMaxProcessor = (activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                               && (newMaxWeight == newStepProcWork);
            const CostT newNodeProcAffinity = newIsSoleMaxProcessor
                ? std::min(vertexWeight, newMaxWeight - newSecondMaxWeight)
                : 0.0;

            const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
            const bool updateNodeProcAffinity = std::abs(diff) > epsilon_;
            if (updateNodeProcAffinity) {
                fullUpdate = true;
                affinityTableNode[nodeProc][windowSize] += diff;
            }

            if ((prevMoveStepMaxWork != newMaxWeight) || updateNodeProcAffinity) {
                updateEntireStep = true;

                for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                    if ((proc == nodeProc) || (proc == moveProc)) {
                        continue;
                    }

                    const VertexWorkWeightT newWeight
                        = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
                    const CostT prevOtherAffinity
                        = ComputeSameStepAffinity(prevMoveStepMaxWork, newWeight, prevNodeProcAffinity);
                    const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

                    affinityTableNode[proc][windowSize] += (otherAffinity - prevOtherAffinity);
                }
            }

            if (nodeProc != moveProc && IsCompatible(node, moveProc)) {
                const VertexWorkWeightT prevNewWeight
                    = vertexWeight
                        + activeSchedule_.GetStepProcessorWork(nodeStep, moveProc)
                        + moveCorrectionNodeWeight;
                const CostT prevOtherAffinity = ComputeSameStepAffinity(
                    prevMoveStepMaxWork, prevNewWeight, prevNodeProcAffinity);
                const VertexWorkWeightT newWeight
                    = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, moveProc);
                const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

                affinityTableNode[moveProc][windowSize] += (otherAffinity - prevOtherAffinity);
            }
        } else {
            const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(moveStep);
            const unsigned idx = RelStepIdx(nodeStep, moveStep);
            if (prevMoveStepMaxWork != newMaxWeight) {
                updateEntireStep = true;

                // update moving to all procs with special for moveProc
                for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                    const VertexWorkWeightT newWeight
                        = vertexWeight + activeSchedule_.GetStepProcessorWork(moveStep, proc);
                    if (proc != moveProc) {
                        const CostT prevAffinity = prevMoveStepMaxWork < newWeight
                            ? static_cast<CostT>(newWeight)
                                - static_cast<CostT>(prevMoveStepMaxWork)
                            : 0.0;
                        const CostT newAffinity = newMaxWeight < newWeight
                            ? static_cast<CostT>(newWeight)
                                - static_cast<CostT>(newMaxWeight)
                            : 0.0;
                        affinityTableNode[proc][idx] += newAffinity - prevAffinity;
                    } else {
                        const VertexWorkWeightT prevNewWeight
                            = vertexWeight
                                + activeSchedule_.GetStepProcessorWork(moveStep, proc)
                                + moveCorrectionNodeWeight;
                        const CostT prevAffinity = prevMoveStepMaxWork < prevNewWeight
                            ? static_cast<CostT>(prevNewWeight)
                                - static_cast<CostT>(prevMoveStepMaxWork)
                            : 0.0;

                        const CostT newAffinity = newMaxWeight < newWeight
                            ? static_cast<CostT>(newWeight)
                                - static_cast<CostT>(newMaxWeight)
                            : 0.0;
                        affinityTableNode[proc][idx] += newAffinity - prevAffinity;
                    }
                }
            } else {
                // update only moveProc
                if (IsCompatible(node, moveProc)) {
                    const VertexWorkWeightT newWeight
                        = vertexWeight + activeSchedule_.GetStepProcessorWork(moveStep, moveProc);
                    const VertexWorkWeightT prevNewWeight = newWeight + moveCorrectionNodeWeight;
                    const CostT prevAffinity = prevMoveStepMaxWork < prevNewWeight
                        ? static_cast<CostT>(prevNewWeight)
                            - static_cast<CostT>(prevMoveStepMaxWork)
                        : 0.0;

                    const CostT newAffinity = newMaxWeight < newWeight
                        ? static_cast<CostT>(newWeight)
                            - static_cast<CostT>(newMaxWeight)
                        : 0.0;
                    affinityTableNode[moveProc][idx] += newAffinity - prevAffinity;
                }
            }
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::SelectNodesCheckRemoveSuperstep(
    unsigned &stepToRemove, ThreadSearchContext &threadData)
{
    if (threadData.stepSelectionEpochCounter_ >= parameters_.nodeMaxStepSelectionEpochs_ || threadData.NumSteps() < 3) {
        return false;
    }

    for (stepToRemove = threadData.stepSelectionCounter_; stepToRemove <= threadData.endStep_; stepToRemove++) {
        if (CheckRemoveSuperstep(stepToRemove)) {
            if (ScatterNodesSuperstep(stepToRemove, threadData)) {
                threadData.stepSelectionCounter_ = stepToRemove + 1;

                if (threadData.stepSelectionCounter_ > threadData.endStep_) {
                    threadData.stepSelectionCounter_ = threadData.startStep_;
                    threadData.stepSelectionEpochCounter_++;
                }
                return true;
            }
        }
    }

    threadData.stepSelectionEpochCounter_++;
    threadData.stepSelectionCounter_ = threadData.startStep_;
    return false;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::CheckRemoveSuperstep(unsigned step)
{
    if (activeSchedule_.NumSteps() < 2) {
        return false;
    }

    if (activeSchedule_.GetStepMaxWork(step) < instance_->SynchronisationCosts()) {
        return true;
    }

    return false;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ResetInnerSearchStructures(
    ThreadSearchContext &threadData) const
{
    threadData.unlockEdgeBacktrackCounter_ = threadData.unlockEdgeBacktrackCounterReset_;
    threadData.maxInnerIterations_ = parameters_.maxInnerIterationsReset_;
    threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackReset_;
    threadData.averageGain_ = 0.0;
    threadData.affinityTable_.ResetNodeSelection();
    threadData.maxGainHeap_.Clear();
    threadData.lockManager_.Clear();
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::IsLocalSearchBlocked(
    ThreadSearchContext &threadData)
{
    for (const auto &pair : threadData.activeScheduleData_.newViolations_) {
        if (threadData.lockManager_.IsLocked(pair.first)) {
            return true;
        }
    }
    return false;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::InitializeDatastructures(
    BspSchedule<GraphT> &schedule)
{
    inputSchedule_ = &schedule;
    instance_ = &schedule.GetInstance();
    graph_ = &instance_->GetComputationalDag();

    activeSchedule_.Initialize(schedule);

    procRange_.Initialize(*instance_);
    commCostF_.Initialize(activeSchedule_, procRange_);
    const CostT initialCost = commCostF_.ComputeScheduleCost();
    activeSchedule_.SetCost(initialCost);

    for (auto &tData : threadDataVec_) {
        tData.affinityTable_.Initialize(activeSchedule_, tData.selectionStrategy_.selectionThreshold_);
        tData.lockManager_.Initialize(graph_->NumVertices());
        tData.rewardPenaltyStrat_.Initialize(
            activeSchedule_, commCostF_.GetMaxCommWeightMultiplied(), activeSchedule_.GetMaxWorkWeight());
        tData.selectionStrategy_.Initialize(activeSchedule_, gen_, tData.startStep_, tData.endStep_);

        tData.localAffinityTable_.resize(instance_->NumberOfProcessors());
        for (unsigned i = 0; i < instance_->NumberOfProcessors(); ++i) {
            tData.localAffinityTable_[i].resize(windowRange_);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateAvgGain(const CostT gain,
                                                                                                const unsigned numIter,
                                                                                                double &averageGain)
{
    averageGain = static_cast<double>((averageGain * numIter + gain)) / (numIter + 1.0);
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::InsertGainHeap(ThreadSearchContext &threadData)
{
    const size_t activeCount = threadData.affinityTable_.size();

    for (size_t i = 0; i < activeCount; ++i) {
        const VertexType node = threadData.affinityTable_.GetSelectedNodes()[i];
        ComputeNodeAffinities(node, threadData.affinityTable_.At(node), threadData);
        const auto bestMove = ComputeBestMove<true>(node, threadData.affinityTable_[node], threadData);
        threadData.maxGainHeap_.Push(node, bestMove);
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::InsertNewNodesGainHeap(
    std::vector<VertexType> &newNodes, NodeSelectionContainerT &nodes, ThreadSearchContext &threadData)
{
    for (const auto &node : newNodes) {
        nodes.Insert(node);
        ComputeNodeAffinities(node, threadData.affinityTable_.At(node), threadData);
        const auto bestMove = ComputeBestMove<true>(node, threadData.affinityTable_[node], threadData);
        threadData.maxGainHeap_.Push(node, bestMove);
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::CleanupDatastructures()
{
    threadDataVec_.clear();
    activeSchedule_.Clear();
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, unsigned proc,
    NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData)
{
    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);

    if ((nodeProc == proc) && (nodeStep == step)) {
        return;
    }

    KlMove nodeMove = threadData.maxGainHeap_.GetValue(node);
    CostT maxGain = nodeMove.gain_;

    unsigned maxProc = nodeMove.toProc_;
    unsigned maxStep = nodeMove.toStep_;

    if ((maxStep == step) && (maxProc == proc)) {
        RecomputeNodeMaxGain(node, affinityTable, threadData);
    } else {
        const unsigned idx = RelStepIdx(nodeStep, step);
        const CostT gain = affinityTable[node][nodeProc][windowSize] - affinityTable[node][proc][idx];
        if (gain > maxGain) {
            maxGain = gain;
            maxProc = proc;
            maxStep = step;
        }

        const CostT diff = maxGain - nodeMove.gain_;
        if ((std::abs(diff) > epsilon_) || (maxProc != nodeMove.toProc_) || (maxStep != nodeMove.toStep_)) {
            nodeMove.gain_ = maxGain;
            nodeMove.toStep_ = maxStep;
            nodeMove.toProc_ = maxProc;
            threadData.maxGainHeap_.Update(node, nodeMove);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData)
{
    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);

    KlMove nodeMove = threadData.maxGainHeap_.GetValue(node);
    CostT maxGain = nodeMove.gain_;

    unsigned maxProc = nodeMove.toProc_;
    unsigned maxStep = nodeMove.toStep_;

    if (maxStep == step) {
        RecomputeNodeMaxGain(node, affinityTable, threadData);
    } else {
        if (nodeStep != step) {
            const unsigned idx = RelStepIdx(nodeStep, step);
            for (const unsigned p : procRange_.CompatibleProcessorsVertex(node)) {
                const CostT gain = affinityTable[node][nodeProc][windowSize] - affinityTable[node][p][idx];
                if (gain > maxGain) {
                    maxGain = gain;
                    maxProc = p;
                    maxStep = step;
                }
            }
        } else {
            for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                if (proc == nodeProc) {
                    continue;
                }

                const CostT gain = affinityTable[node][nodeProc][windowSize] - affinityTable[node][proc][windowSize];
                if (gain > maxGain) {
                    maxGain = gain;
                    maxProc = proc;
                    maxStep = step;
                }
            }
        }

        const CostT diff = maxGain - nodeMove.gain_;
        if ((std::abs(diff) > epsilon_) || (maxProc != nodeMove.toProc_) || (maxStep != nodeMove.toStep_)) {
            nodeMove.gain_ = maxGain;
            nodeMove.toProc_ = maxProc;
            nodeMove.toStep_ = maxStep;
            threadData.maxGainHeap_.Update(node, nodeMove);
        }
    }
}
}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_KL_IMPROVER_TPP
