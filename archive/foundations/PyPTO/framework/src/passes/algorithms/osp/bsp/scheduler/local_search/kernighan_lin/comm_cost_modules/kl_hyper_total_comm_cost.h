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
 * \file kl_hyper_total_comm_cost.h
 * \brief
 */

#ifndef OSP_KL_HYPER_TOTAL_COMM_COST_H
#define OSP_KL_HYPER_TOTAL_COMM_COST_H

#include "../kl_active_schedule.h"
#include "../kl_improver.h"
#include "lambda_container.h"

namespace npu::tile_fwk {
namespace osp {
template <typename GraphT, typename CostT, unsigned windowSize = 1>
struct KlHyperTotalCommCostFunction {
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;

    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    KlActiveSchedule<GraphT, CostT>* activeSchedule_;
    CompatibleProcessorRange<GraphT>* procRange_;
    const GraphT* graph_;
    const BspInstance<GraphT>* instance_;

    CostT commMultiplier_ = 1;
    CostT maxCommWeight_ = 0;

    LambdaVectorContainer<VertexType> nodeLambdaMap_;

    inline CostT GetCommMultiplier() { return commMultiplier_; }

    inline CostT GetMaxCommWeight() { return maxCommWeight_; }

    inline CostT GetMaxCommWeightMultiplied() { return maxCommWeight_ * commMultiplier_; }

    const std::string Name() const { return "hyper_total_comm_cost"; }

    inline bool IsCompatible(VertexType node, unsigned proc)
    {
        return activeSchedule_->GetInstance().IsCompatible(node, proc);
    }

    void Initialize(KlActiveSchedule<GraphT, CostT>& sched, CompatibleProcessorRange<GraphT>& pRange)
    {
        activeSchedule_ = &sched;
        procRange_ = &pRange;
        instance_ = &sched.GetInstance();
        graph_ = &instance_->GetComputationalDag();
        commMultiplier_ = 1.0 / instance_->NumberOfProcessors();
        nodeLambdaMap_.Initialize(graph_->NumVertices(), instance_->NumberOfProcessors());
    }

    struct EmptyStruct {};

    using PreMoveCommDataT = EmptyStruct;

    inline EmptyStruct GetPreMoveCommData(const KlMove&) { return EmptyStruct(); }

    CostT ComputeScheduleCost()
    {
        CostT workCosts = 0;
        for (unsigned step = 0; step < activeSchedule_->NumSteps(); step++) {
            workCosts += activeSchedule_->GetStepMaxWork(step);
        }

        CostT commCosts = 0;
        for (const auto vertex : graph_->Vertices()) {
            const unsigned vertexProc = activeSchedule_->AssignedProcessor(vertex);
            const CostT vCommCost = graph_->VertexCommWeight(vertex);
            maxCommWeight_ = std::max(maxCommWeight_, vCommCost);

            nodeLambdaMap_.ResetNode(vertex);

            for (const auto& target : instance_->GetComputationalDag().Children(vertex)) {
                const unsigned targetProc = activeSchedule_->AssignedProcessor(target);
                if (nodeLambdaMap_.IncreaseProcCount(vertex, targetProc)) {
                    // is 0 if targetProc == vertexProc
                    commCosts += vCommCost * instance_->CommunicationCosts(vertexProc, targetProc);
                }
            }
        }

        return workCosts + commCosts * commMultiplier_ +
               static_cast<VCommwT<GraphT>>(activeSchedule_->NumSteps() - 1) * instance_->SynchronisationCosts();
    }

    CostT ComputeScheduleCostTest()
    {
        CostT workCosts = 0;
        for (unsigned step = 0; step < activeSchedule_->NumSteps(); step++) {
            workCosts += activeSchedule_->GetStepMaxWork(step);
        }

        CostT commCosts = 0;
        for (const auto vertex : graph_->Vertices()) {
            const unsigned vertexProc = activeSchedule_->AssignedProcessor(vertex);
            const CostT vCommCost = graph_->VertexCommWeight(vertex);
            for (const auto lambdaprocMultPair : nodeLambdaMap_.IterateProcEntries(vertex)) {
                const auto& lambdaProc = lambdaprocMultPair.first;
                commCosts += vCommCost * instance_->CommunicationCosts(vertexProc, lambdaProc);
            }
        }

        return workCosts + commCosts * commMultiplier_ +
               static_cast<VCommwT<GraphT>>(activeSchedule_->NumSteps() - 1) * instance_->SynchronisationCosts();
    }

    inline void UpdateDatastructureAfterMove(const KlMove& move, const unsigned startStep, const unsigned endStep)
    {
        if (move.toProc_ != move.fromProc_) {
            for (const auto& source : instance_->GetComputationalDag().Parents(move.node_)) {
                const unsigned sourceStep = activeSchedule_->AssignedSuperstep(source);
                if (sourceStep < startStep || sourceStep > endStep) {
                    continue;
                }
                UpdateSourceAfterMove(move, source);
            }
        }
    }

    inline void UpdateSourceAfterMove(const KlMove& move, VertexType source)
    {
        nodeLambdaMap_.DecreaseProcCount(source, move.fromProc_);
        nodeLambdaMap_.IncreaseProcCount(source, move.toProc_);
    }

    void MarkForFullRecompute(VertexType target, std::map<VertexType, KlGainUpdateInfo>& maxGainRecompute)
    {
        if (maxGainRecompute.find(target) != maxGainRecompute.end()) {
            maxGainRecompute[target].fullUpdate_ = true;
        } else {
            maxGainRecompute[target] = KlGainUpdateInfo(target, true);
        }
    }

    template <typename ThreadDataT>
    bool IsValidAffinityTarget(VertexType target, unsigned startStep, unsigned endStep, VertexType excludeNode,
                               ThreadDataT& threadData)
    {
        const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
        if (targetStep < startStep || targetStep > endStep)
            return false;
        if (target == excludeNode)
            return false;
        if (not threadData.affinityTable_.IsSelected(target))
            return false;
        if (threadData.lockManager_.IsLocked(target))
            return false;
        return true;
    }

    template <typename AffinityTableT>
    void UpdateChildAffinityFromStep(const KlMove& move, VertexType target, unsigned targetStep, unsigned targetProc,
                                     unsigned targetStartIdx, unsigned endStep, const CostT& penalty,
                                     const CostT& reward, AffinityTableT& affinityTable)
    {
        if (move.fromStep_ < targetStep + (move.fromProc_ == targetProc)) {
            const unsigned diff = targetStep - move.fromStep_;
            const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
            unsigned idx = targetStartIdx;
            for (; idx < bound; idx++) {
                for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                    affinityTable[p][idx] -= penalty;
                }
            }
            if (idx - 1 < bound && IsCompatible(target, move.fromProc_)) {
                affinityTable[move.fromProc_][idx - 1] += penalty;
            }
        } else {
            const unsigned diff = move.fromStep_ - targetStep;
            const unsigned windowBound = EndIdx(targetStep, endStep);
            unsigned idx = std::min(windowSize + diff, windowBound);
            if (idx < windowBound && IsCompatible(target, move.fromProc_)) {
                affinityTable[move.fromProc_][idx] += reward;
            }
            idx++;
            for (; idx < windowBound; idx++) {
                for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                    affinityTable[p][idx] += reward;
                }
            }
        }
    }

    template <typename AffinityTableT>
    void UpdateChildAffinityToStep(const KlMove& move, VertexType target, unsigned targetStep, unsigned targetProc,
                                   unsigned targetStartIdx, unsigned endStep, const CostT& penalty, const CostT& reward,
                                   AffinityTableT& affinityTable);

    template <typename AffinityTableT>
    void UpdateChildLambdaAffinity(const KlMove& move, VertexType target, unsigned targetStep, unsigned targetProc,
                                   unsigned targetStartIdx, unsigned endStep, AffinityTableT& affinityTable);

    template <typename ThreadDataT>
    void ApplyAffinityToRange(VertexType target, unsigned proc, unsigned startStep, unsigned endStep, CostT commAff,
                              ThreadDataT& threadData)
    {
        const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
        auto& affinityRow = threadData.affinityTable_.At(target)[proc];
        for (unsigned idx = StartIdx(targetStep, startStep); idx < EndIdx(targetStep, endStep); idx++) {
            affinityRow[idx] += commAff;
        }
    }

    template <typename ThreadDataT>
    void AdjustLambdaForAllProcs(VertexType target, unsigned sourceProc, CostT commGain, unsigned startStep,
                                 unsigned endStep, CostT sign, std::map<VertexType, KlGainUpdateInfo>& maxGainRecompute,
                                 ThreadDataT& threadData)
    {
        const unsigned targetProc = activeSchedule_->AssignedProcessor(target);
        MarkForFullRecompute(target, maxGainRecompute);
        const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
        const unsigned targetStartIdx = StartIdx(targetStep, startStep);
        const unsigned targetWindowBound = EndIdx(targetStep, endStep);
        auto& affinityTableTarget = threadData.affinityTable_.At(target);
        const CostT commAff = sign * instance_->CommunicationCosts(sourceProc, targetProc) * commGain;
        for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
            if (p == targetProc) {
                continue;
            }
            for (unsigned idx = targetStartIdx; idx < targetWindowBound; idx++) {
                affinityTableTarget[p][idx] += commAff;
            }
        }
    }

    template <typename ThreadDataT>
    void UpdateSourceLambdaFromProc(const KlMove& move, VertexType source, unsigned sourceProc, unsigned startStep,
                                    unsigned endStep, std::map<VertexType, KlGainUpdateInfo>& maxGainRecompute,
                                    ThreadDataT& threadData)
    {
        if (nodeLambdaMap_.HasNoProcEntry(source, move.fromProc_)) {
            const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;
            for (const auto& target : instance_->GetComputationalDag().Children(source)) {
                if (!IsValidAffinityTarget(target, startStep, endStep, move.node_, threadData)) {
                    continue;
                }
                if (sourceProc != move.fromProc_ && IsCompatible(target, move.fromProc_)) {
                    MarkForFullRecompute(target, maxGainRecompute);
                    const CostT commAff = instance_->CommunicationCosts(sourceProc, move.fromProc_) * commGain;
                    ApplyAffinityToRange(target, move.fromProc_, startStep, endStep, commAff, threadData);
                }
            }
        } else if (nodeLambdaMap_.GetProcEntry(source, move.fromProc_) == 1) {
            const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;
            for (const auto& target : instance_->GetComputationalDag().Children(source)) {
                if (!IsValidAffinityTarget(target, startStep, endStep, move.node_, threadData)) {
                    continue;
                }
                if (activeSchedule_->AssignedProcessor(target) == move.fromProc_) {
                    AdjustLambdaForAllProcs(target, sourceProc, commGain, startStep, endStep, -1, maxGainRecompute,
                                            threadData);
                    break;
                }
            }
        }
    }

    template <typename ThreadDataT>
    void UpdateSourceLambdaToProc(const KlMove& move, VertexType source, unsigned sourceProc, unsigned startStep,
                                  unsigned endStep, std::map<VertexType, KlGainUpdateInfo>& maxGainRecompute,
                                  ThreadDataT& threadData);

    template <typename AffinityTableT>
    void UpdateSourceAffinityFromStep(const KlMove& move, VertexType source, unsigned sourceStep, unsigned sourceProc,
                                      unsigned sourceStartIdx, unsigned endStep, const CostT& penalty,
                                      const CostT& reward, AffinityTableT& affinityTableSource);

    template <typename AffinityTableT>
    void UpdateSourceAffinityToStep(const KlMove& move, VertexType source, unsigned sourceStep, unsigned sourceProc,
                                    unsigned sourceStartIdx, unsigned endStep, const CostT& penalty,
                                    const CostT& reward, AffinityTableT& affinityTableSource)
    {
        if (move.toStep_ < sourceStep + (move.toProc_ != sourceProc)) {
            const unsigned diff = sourceStep - move.toStep_;
            const unsigned bound = windowSize > diff ? windowSize - diff : 0;
            unsigned idx = sourceStartIdx;
            for (; idx < bound; idx++) {
                for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                    affinityTableSource[p][idx] -= reward;
                }
            }
            if (windowSize >= diff && IsCompatible(source, move.toProc_)) {
                affinityTableSource[move.toProc_][idx] -= reward;
            }
        } else {
            const unsigned windowBound = EndIdx(sourceStep, endStep);
            const unsigned diff = move.toStep_ - sourceStep;
            unsigned idx = windowSize + diff;
            if (idx < windowBound && IsCompatible(source, move.toProc_)) {
                affinityTableSource[move.toProc_][idx] -= penalty;
            }
            for (; idx < windowBound; idx++) {
                for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                    affinityTableSource[p][idx] += penalty;
                }
            }
        }
    }

    template <typename AffinityTableT>
    void UpdateSourceLambdaCommCost(const KlMove& move, VertexType source, unsigned sourceProc, unsigned sourceStartIdx,
                                    unsigned endStep, unsigned sourceStep, AffinityTableT& affinityTableSource)
    {
        const unsigned windowBound = EndIdx(sourceStep, endStep);

        if (nodeLambdaMap_.HasNoProcEntry(source, move.fromProc_)) {
            const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;
            for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                if (p == sourceProc)
                    continue;
                const CostT commCost = ChangeCommCost(instance_->CommunicationCosts(p, move.fromProc_),
                                                      instance_->CommunicationCosts(sourceProc, move.fromProc_),
                                                      commGain);
                for (unsigned idx = sourceStartIdx; idx < windowBound; idx++) {
                    affinityTableSource[p][idx] -= commCost;
                }
            }
        }

        if (nodeLambdaMap_.GetProcEntry(source, move.toProc_) == 1) {
            const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;
            for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                if (p == sourceProc)
                    continue;
                const CostT commCost = ChangeCommCost(instance_->CommunicationCosts(p, move.toProc_),
                                                      instance_->CommunicationCosts(sourceProc, move.toProc_),
                                                      commGain);
                for (unsigned idx = sourceStartIdx; idx < windowBound; idx++) {
                    affinityTableSource[p][idx] += commCost;
                }
            }
        }
    }

    template <typename ThreadDataT>
    void UpdateChildrenCommAffinity(const KlMove& move, ThreadDataT& threadData, const CostT& penalty,
                                    const CostT& reward, std::map<VertexType, KlGainUpdateInfo>& maxGainRecompute,
                                    std::vector<VertexType>& newNodes)
    {
        const unsigned startStep = threadData.startStep_;
        const unsigned endStep = threadData.endStep_;

        for (const auto& target : instance_->GetComputationalDag().Children(move.node_)) {
            const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
            if (targetStep < startStep || targetStep > endStep)
                continue;
            if (threadData.lockManager_.IsLocked(target))
                continue;

            if (not threadData.affinityTable_.IsSelected(target)) {
                newNodes.push_back(target);
                continue;
            }

            MarkForFullRecompute(target, maxGainRecompute);

            const unsigned targetProc = activeSchedule_->AssignedProcessor(target);
            const unsigned targetStartIdx = StartIdx(targetStep, startStep);
            auto& affinityTable = threadData.affinityTable_.At(target);

            UpdateChildAffinityFromStep(move, target, targetStep, targetProc, targetStartIdx, endStep, penalty, reward,
                                        affinityTable);
            UpdateChildAffinityToStep(move, target, targetStep, targetProc, targetStartIdx, endStep, penalty, reward,
                                      affinityTable);

            if (move.toProc_ != move.fromProc_) {
                UpdateChildLambdaAffinity(move, target, targetStep, targetProc, targetStartIdx, endStep, affinityTable);
            }
        }
    }

    template <typename ThreadDataT>
    void UpdateParentsCommAffinity(const KlMove& move, ThreadDataT& threadData, const CostT& penalty,
                                   const CostT& reward, std::map<VertexType, KlGainUpdateInfo>& maxGainRecompute,
                                   std::vector<VertexType>& newNodes)
    {
        const unsigned startStep = threadData.startStep_;
        const unsigned endStep = threadData.endStep_;

        for (const auto& source : instance_->GetComputationalDag().Parents(move.node_)) {
            if (move.toProc_ != move.fromProc_) {
                const unsigned sourceProc = activeSchedule_->AssignedProcessor(source);
                UpdateSourceLambdaFromProc(move, source, sourceProc, startStep, endStep, maxGainRecompute, threadData);
                UpdateSourceLambdaToProc(move, source, sourceProc, startStep, endStep, maxGainRecompute, threadData);
            }

            const unsigned sourceStep = activeSchedule_->AssignedSuperstep(source);
            if (sourceStep < startStep || sourceStep > endStep)
                continue;
            if (threadData.lockManager_.IsLocked(source))
                continue;

            if (not threadData.affinityTable_.IsSelected(source)) {
                newNodes.push_back(source);
                continue;
            }

            MarkForFullRecompute(source, maxGainRecompute);

            const unsigned sourceProc = activeSchedule_->AssignedProcessor(source);
            const unsigned sourceStartIdx = StartIdx(sourceStep, startStep);
            auto& affinityTableSource = threadData.affinityTable_.At(source);

            UpdateSourceAffinityFromStep(move, source, sourceStep, sourceProc, sourceStartIdx, endStep, penalty, reward,
                                         affinityTableSource);
            UpdateSourceAffinityToStep(move, source, sourceStep, sourceProc, sourceStartIdx, endStep, penalty, reward,
                                       affinityTableSource);

            if (move.toProc_ != move.fromProc_) {
                UpdateSourceLambdaCommCost(move, source, sourceProc, sourceStartIdx, endStep, sourceStep,
                                           affinityTableSource);
            }
        }
    }

    template <typename ThreadDataT>
    void UpdateNodeCommAffinity(const KlMove& move, ThreadDataT& threadData, const CostT& penalty, const CostT& reward,
                                std::map<VertexType, KlGainUpdateInfo>& maxGainRecompute,
                                std::vector<VertexType>& newNodes)
    {
        UpdateChildrenCommAffinity(move, threadData, penalty, reward, maxGainRecompute, newNodes);
        UpdateParentsCommAffinity(move, threadData, penalty, reward, maxGainRecompute, newNodes);
    }

    inline unsigned StartIdx(const unsigned nodeStep, const unsigned startStep)
    {
        return nodeStep < windowSize + startStep ? windowSize - (nodeStep - startStep) : 0;
    }

    inline unsigned EndIdx(const unsigned nodeStep, const unsigned endStep)
    {
        return nodeStep + windowSize <= endStep ? windowRange_ : windowRange_ - (nodeStep + windowSize - endStep);
    }

    inline CostT ChangeCommCost(const VCommwT<GraphT>& pTargetCommCost, const VCommwT<GraphT>& nodeTargetCommCost,
                                const CostT& commGain)
    {
        return pTargetCommCost > nodeTargetCommCost ? (pTargetCommCost - nodeTargetCommCost) * commGain :
                                                      (nodeTargetCommCost - pTargetCommCost) * commGain * -1.0;
    }

    template <typename AffinityTableT>
    void ComputeChildCommAffinity(VertexType node, AffinityTableT& affinityTableNode, const CostT& penalty,
                                  const CostT& reward, unsigned nodeStep, unsigned nodeProc, unsigned nodeStartIdx,
                                  unsigned windowBound);
    template <typename AffinityTableT>
    void ComputeLambdaProcAffinity(VertexType node, AffinityTableT& affinityTableNode, unsigned nodeProc,
                                   unsigned nodeStartIdx, unsigned windowBound);

    template <typename AffinityTableT>
    void ComputeParentStepAffinity(VertexType node, AffinityTableT& affinityTableNode, const CostT& penalty,
                                   const CostT& reward, unsigned nodeStep, unsigned nodeProc, unsigned sourceStep,
                                   unsigned sourceProc, unsigned nodeStartIdx, unsigned windowBound);

    template <typename AffinityTableT>
    void ComputeParentLambdaAffinity(VertexType node, AffinityTableT& affinityTableNode, VertexType source,
                                     unsigned nodeProc, unsigned sourceProc, unsigned nodeStartIdx,
                                     unsigned windowBound);

    template <typename AffinityTableT>
    void ComputeParentCommAffinity(VertexType node, AffinityTableT& affinityTableNode, const CostT& penalty,
                                   const CostT& reward, unsigned nodeStep, unsigned nodeProc, unsigned nodeStartIdx,
                                   unsigned windowBound);

    template <typename AffinityTableT>
    void ComputeCommAffinity(VertexType node, AffinityTableT& affinityTableNode, const CostT& penalty,
                             const CostT& reward, const unsigned startStep, const unsigned endStep);
};
} // namespace osp
} // namespace npu::tile_fwk
#include "kl_hyper_total_comm_cost.tpp"
#endif // OSP_KL_HYPER_TOTAL_COMM_COST_HPP
