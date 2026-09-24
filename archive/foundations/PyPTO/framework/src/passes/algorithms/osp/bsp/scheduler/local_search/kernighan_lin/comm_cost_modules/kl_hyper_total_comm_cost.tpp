/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OSP_KL_HYPER_TOTAL_COMM_COST_TPP
#define OSP_KL_HYPER_TOTAL_COMM_COST_TPP

namespace npu::tile_fwk {
namespace osp {
template <typename GraphT, typename CostT, unsigned windowSize>
template <typename AffinityTableT>
void KlHyperTotalCommCostFunction<GraphT, CostT, windowSize>::UpdateChildAffinityToStep(
    const KlMove &move, VertexType target, unsigned targetStep,
                                unsigned targetProc, unsigned targetStartIdx, unsigned endStep,
                                const CostT &penalty, const CostT &reward, AffinityTableT &affinityTable)
{
    if (move.toStep_ < targetStep + (move.toProc_ == targetProc)) {
        unsigned idx = targetStartIdx;
        const unsigned diff = targetStep - move.toStep_;
        const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
        for (; idx < bound; idx++) {
            for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                affinityTable[p][idx] += penalty;
            }
        }
        if (idx - 1 < bound && IsCompatible(target, move.toProc_)) {
            affinityTable[move.toProc_][idx - 1] -= penalty;
        }
    } else {
        const unsigned diff = move.toStep_ - targetStep;
        const unsigned windowBound = EndIdx(targetStep, endStep);
        unsigned idx = std::min(windowSize + diff, windowBound);
        if (idx < windowBound && IsCompatible(target, move.toProc_)) {
            affinityTable[move.toProc_][idx] -= reward;
        }
        idx++;
        for (; idx < windowBound; idx++) {
            for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                affinityTable[p][idx] -= reward;
            }
        }
    }
}

template <typename GraphT, typename CostT, unsigned windowSize>
template <typename AffinityTableT>
void KlHyperTotalCommCostFunction<GraphT, CostT, windowSize>::UpdateChildLambdaAffinity(
    const KlMove &move, VertexType target, unsigned targetStep,
                                unsigned targetProc, unsigned targetStartIdx, unsigned endStep,
                                AffinityTableT &affinityTable)
{
    const CostT commGain = graph_->VertexCommWeight(move.node_) * commMultiplier_;
    const unsigned windowBound = EndIdx(targetStep, endStep);

    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
        if (p == targetProc) {
            continue;
        }
        if (nodeLambdaMap_.GetProcEntry(move.node_, targetProc) == 1) {
            const CostT x = instance_->CommunicationCosts(move.fromProc_, targetProc) * commGain;
            const CostT y = instance_->CommunicationCosts(move.toProc_, targetProc) * commGain;
            for (unsigned idx = targetStartIdx; idx < windowBound; idx++) {
                affinityTable[p][idx] += x - y;
            }
        }
        if (nodeLambdaMap_.HasNoProcEntry(move.node_, p)) {
            const CostT x = instance_->CommunicationCosts(move.fromProc_, p) * commGain;
            const CostT y = instance_->CommunicationCosts(move.toProc_, p) * commGain;
            for (unsigned idx = targetStartIdx; idx < windowBound; idx++) {
                affinityTable[p][idx] -= x - y;
            }
        }
    }
}

template <typename GraphT, typename CostT, unsigned windowSize>
template <typename AffinityTableT>
void KlHyperTotalCommCostFunction<GraphT, CostT, windowSize>::ComputeChildCommAffinity(
    VertexType node, AffinityTableT &affinityTableNode,
                                const CostT &penalty, const CostT &reward,
                                unsigned nodeStep, unsigned nodeProc,
                                unsigned nodeStartIdx, unsigned windowBound)
{
    for (const auto &target : instance_->GetComputationalDag().Children(node)) {
        const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
        const unsigned targetProc = activeSchedule_->AssignedProcessor(target);

        if (targetStep < nodeStep + (targetProc != nodeProc)) {
            const unsigned diff = nodeStep - targetStep;
            const unsigned bound = windowSize > diff ? windowSize - diff : 0;
            unsigned idx = nodeStartIdx;
            for (; idx < bound; idx++) {
                for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                    affinityTableNode[p][idx] -= reward;
                }
            }
            if (windowSize >= diff && IsCompatible(node, targetProc)) {
                affinityTableNode[targetProc][idx] -= reward;
            }
        } else {
            const unsigned diff = targetStep - nodeStep;
            unsigned idx = windowSize + diff;
            if (idx < windowBound && IsCompatible(node, targetProc)) {
                affinityTableNode[targetProc][idx] -= penalty;
            }
            for (; idx < windowBound; idx++) {
                for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                    affinityTableNode[p][idx] += penalty;
                }
            }
        }
    }
}

template <typename GraphT, typename CostT, unsigned windowSize>
template <typename AffinityTableT>
void KlHyperTotalCommCostFunction<GraphT, CostT, windowSize>::ComputeLambdaProcAffinity(
    VertexType node, AffinityTableT &affinityTableNode,
                                unsigned nodeProc, unsigned nodeStartIdx, unsigned windowBound)
{
    const CostT commGain = graph_->VertexCommWeight(node) * commMultiplier_;
    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
        if (p == nodeProc) {
            continue;
        }
        for (const auto lambdaPair : nodeLambdaMap_.IterateProcEntries(node)) {
            const auto &lambdaProc = lambdaPair.first;
            const CostT commCost = ChangeCommCost(
                instance_->CommunicationCosts(p, lambdaProc),
                instance_->CommunicationCosts(nodeProc, lambdaProc), commGain);
            for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                affinityTableNode[p][idx] += commCost;
            }
        }
    }
}

template <typename GraphT, typename CostT, unsigned windowSize>
template <typename AffinityTableT>
void KlHyperTotalCommCostFunction<GraphT, CostT, windowSize>::ComputeParentStepAffinity(
    VertexType node, AffinityTableT &affinityTableNode,
                                const CostT &penalty, const CostT &reward,
                                unsigned nodeStep, unsigned nodeProc,
                                unsigned sourceStep, unsigned sourceProc,
                                unsigned nodeStartIdx, unsigned windowBound)
{
    if (sourceStep < nodeStep + (sourceProc == nodeProc)) {
        const unsigned diff = nodeStep - sourceStep;
        const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
        unsigned idx = nodeStartIdx;
        for (; idx < bound; idx++) {
            for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                affinityTableNode[p][idx] += penalty;
            }
        }
        if (idx - 1 < bound && IsCompatible(node, sourceProc)) {
            affinityTableNode[sourceProc][idx - 1] -= penalty;
        }
    } else {
        const unsigned diff = sourceStep - nodeStep;
        unsigned idx = std::min(windowSize + diff, windowBound);
        if (idx < windowBound && IsCompatible(node, sourceProc)) {
            affinityTableNode[sourceProc][idx] -= reward;
        }
        idx++;
        for (; idx < windowBound; idx++) {
            for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                affinityTableNode[p][idx] -= reward;
            }
        }
    }
}

template <typename GraphT, typename CostT, unsigned windowSize>
template <typename AffinityTableT>
void KlHyperTotalCommCostFunction<GraphT, CostT, windowSize>::ComputeParentLambdaAffinity(
    VertexType node, AffinityTableT &affinityTableNode,
                                    VertexType source, unsigned nodeProc, unsigned sourceProc,
                                    unsigned nodeStartIdx, unsigned windowBound)
{
    const CostT sourceCommGain = graph_->VertexCommWeight(source) * commMultiplier_;
    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
        if (p == nodeProc) {
            continue;
        }
        if (sourceProc != nodeProc && nodeLambdaMap_.GetProcEntry(source, nodeProc) == 1) {
            for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                affinityTableNode[p][idx] -= instance_->CommunicationCosts(sourceProc, nodeProc) * sourceCommGain;
            }
        }
        if (sourceProc != p && nodeLambdaMap_.HasNoProcEntry(source, p)) {
            for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                affinityTableNode[p][idx] += instance_->CommunicationCosts(sourceProc, p) * sourceCommGain;
            }
        }
    }
}

template <typename GraphT, typename CostT, unsigned windowSize>
template <typename AffinityTableT>
void KlHyperTotalCommCostFunction<GraphT, CostT, windowSize>::ComputeParentCommAffinity(
    VertexType node, AffinityTableT &affinityTableNode,
                                const CostT &penalty, const CostT &reward,
                                unsigned nodeStep, unsigned nodeProc,
                                unsigned nodeStartIdx, unsigned windowBound)
{
    for (const auto &source : instance_->GetComputationalDag().Parents(node)) {
        const unsigned sourceStep = activeSchedule_->AssignedSuperstep(source);
        const unsigned sourceProc = activeSchedule_->AssignedProcessor(source);

        ComputeParentStepAffinity(node, affinityTableNode, penalty, reward,
                                    nodeStep, nodeProc, sourceStep, sourceProc, nodeStartIdx, windowBound);
        ComputeParentLambdaAffinity(node, affinityTableNode, source, nodeProc, sourceProc,
                                    nodeStartIdx, windowBound);
    }
}

template <typename GraphT, typename CostT, unsigned windowSize>
template <typename AffinityTableT>
void KlHyperTotalCommCostFunction<GraphT, CostT, windowSize>::ComputeCommAffinity(VertexType node,
                            AffinityTableT &affinityTableNode,
                            const CostT &penalty,
                            const CostT &reward,
                            const unsigned startStep,
                            const unsigned endStep)
{
    const unsigned nodeStep = activeSchedule_->AssignedSuperstep(node);
    const unsigned nodeProc = activeSchedule_->AssignedProcessor(node);
    const unsigned windowBound = EndIdx(nodeStep, endStep);
    const unsigned nodeStartIdx = StartIdx(nodeStep, startStep);

    ComputeChildCommAffinity(node, affinityTableNode, penalty, reward,
                                nodeStep, nodeProc, nodeStartIdx, windowBound);
    ComputeLambdaProcAffinity(node, affinityTableNode, nodeProc, nodeStartIdx, windowBound);
    ComputeParentCommAffinity(node, affinityTableNode, penalty, reward,
                                nodeStep, nodeProc, nodeStartIdx, windowBound);
}

template <typename GraphT, typename CostT, unsigned windowSize>
template <typename ThreadDataT>
void KlHyperTotalCommCostFunction<GraphT, CostT, windowSize>::UpdateSourceLambdaToProc(
    const KlMove &move, VertexType source, unsigned sourceProc,
                                unsigned startStep, unsigned endStep,
                                std::map<VertexType, KlGainUpdateInfo> &maxGainRecompute,
                                ThreadDataT &threadData)
{
    if (nodeLambdaMap_.GetProcEntry(source, move.toProc_) == 1) {
        const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;
        for (const auto &target : instance_->GetComputationalDag().Children(source)) {
            if (!IsValidAffinityTarget(target, startStep, endStep, move.node_, threadData)) {
                continue;
            }
            if (sourceProc != move.toProc_ && IsCompatible(target, move.toProc_)) {
                MarkForFullRecompute(target, maxGainRecompute);
                const CostT commAff = -(instance_->CommunicationCosts(sourceProc, move.toProc_) * commGain);
                ApplyAffinityToRange(target, move.toProc_, startStep, endStep, commAff, threadData);
            }
        }
    } else if (nodeLambdaMap_.GetProcEntry(source, move.toProc_) == 2) {
        const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;
        for (const auto &target : instance_->GetComputationalDag().Children(source)) {
            if (!IsValidAffinityTarget(target, startStep, endStep, move.node_, threadData)) {
                continue;
            }
            if (activeSchedule_->AssignedProcessor(target) == move.toProc_ && sourceProc != move.toProc_) {
                AdjustLambdaForAllProcs(target, sourceProc, commGain, startStep, endStep, +1,
                                       maxGainRecompute, threadData);
                break;
            }
        }
    }
}

template <typename GraphT, typename CostT, unsigned windowSize>
template <typename AffinityTableT>
void KlHyperTotalCommCostFunction<GraphT, CostT, windowSize>::UpdateSourceAffinityFromStep(
    const KlMove &move, VertexType source, unsigned sourceStep,
                                    unsigned sourceProc, unsigned sourceStartIdx, unsigned endStep,
                                    const CostT &penalty, const CostT &reward, AffinityTableT &affinityTableSource)
{
    if (move.fromStep_ < sourceStep + (move.fromProc_ != sourceProc)) {
        const unsigned diff = sourceStep - move.fromStep_;
        const unsigned bound = windowSize > diff ? windowSize - diff : 0;
        unsigned idx = sourceStartIdx;
        for (; idx < bound; idx++) {
            for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                affinityTableSource[p][idx] += reward;
            }
        }
        if (windowSize >= diff && IsCompatible(source, move.fromProc_)) {
            affinityTableSource[move.fromProc_][idx] += reward;
        }
    } else {
        const unsigned windowBound = EndIdx(sourceStep, endStep);
        const unsigned diff = move.fromStep_ - sourceStep;
        unsigned idx = windowSize + diff;
        if (idx < windowBound && IsCompatible(source, move.fromProc_)) {
            affinityTableSource[move.fromProc_][idx] += penalty;
        }
        for (; idx < windowBound; idx++) {
            for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                affinityTableSource[p][idx] -= penalty;
            }
        }
    }
}
}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_KL_HYPER_TOTAL_COMM_COST_TPP
