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
 * \file kl_active_schedule.h
 * \brief
 */

#ifndef OSP_KL_ACTIVE_SCHEDULE_H
#define OSP_KL_ACTIVE_SCHEDULE_H

#include "passes/algorithms/osp/bsp/scheduler/improvement_scheduler.h"
#include "passes/algorithms/osp/bsp/scheduler/local_search/kernighan_lin/kl_active_schedule_types.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_util.h"

namespace npu::tile_fwk {
namespace osp {
template <typename GraphT, typename CostT>
class KlActiveSchedule {
public:
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using ThreadDataT = ThreadLocalActiveScheduleData<GraphT, CostT>;

    virtual ~KlActiveSchedule() = default;

    inline const BspInstance<GraphT>& GetInstance() const { return *instance_; }

    inline const BspSchedule<GraphT>& GetVectorSchedule() const { return vectorSchedule_; }

    inline BspSchedule<GraphT>& GetVectorSchedule() { return vectorSchedule_; }

    inline const SetSchedule<GraphT>& GetSetSchedule() const { return setSchedule_; }

    inline CostT GetCost() { return cost_; }

    inline bool IsFeasible() { return feasible_; }

    inline unsigned NumSteps() const { return vectorSchedule_.NumberOfSupersteps(); }

    inline unsigned AssignedProcessor(VertexType node) const { return vectorSchedule_.AssignedProcessor(node); }

    inline unsigned AssignedSuperstep(VertexType node) const { return vectorSchedule_.AssignedSuperstep(node); }

    inline VWorkwT<GraphT> GetStepMaxWork(unsigned step) const { return workDatastructures_.StepMaxWork(step); }

    inline VWorkwT<GraphT> GetStepSecondMaxWork(unsigned step) const
    {
        return workDatastructures_.StepSecondMaxWork(step);
    }

    inline std::vector<unsigned>& GetStepMaxWorkProcessorCount()
    {
        return workDatastructures_.stepMaxWorkProcessorCount_;
    }

    inline VWorkwT<GraphT> GetStepProcessorWork(unsigned step, unsigned proc) const
    {
        return workDatastructures_.StepProcWork(step, proc);
    }

    inline PreMoveWorkData<VWorkwT<GraphT>> GetPreMoveWorkData(KlMove move)
    {
        return workDatastructures_.GetPreMoveWorkData(move);
    }

    inline VWorkwT<GraphT> GetMaxWorkWeight() { return workDatastructures_.maxWorkWeight_; }

    inline VWorkwT<GraphT> GetTotalWorkWeight() { return workDatastructures_.totalWorkWeight_; }

    inline void SetCost(CostT cost) { cost_ = cost; }

    KlActiveScheduleWorkDatastructures<GraphT> workDatastructures_;

    inline VWorkwT<GraphT> GetStepTotalWork(unsigned step) const
    {
        VWorkwT<GraphT> totalWork = 0;
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            totalWork += workDatastructures_.StepProcWork(step, proc);
        }
        return totalWork;
    }

    void ApplyMove(KlMove move, ThreadDataT& threadData)
    {
        vectorSchedule_.SetAssignedProcessor(move.node_, move.toProc_);
        vectorSchedule_.SetAssignedSuperstep(move.node_, move.toStep_);

        setSchedule_.GetProcessorStepVertices()[move.fromStep_][move.fromProc_].erase(move.node_);
        setSchedule_.GetProcessorStepVertices()[move.toStep_][move.toProc_].insert(move.node_);

        UpdateViolations(move.node_, threadData);
        threadData.appliedMoves_.push_back(move);

        workDatastructures_.ApplyMove(move, instance_->GetComputationalDag().VertexWorkWeight(move.node_));
    }

    template <typename CommDatastructuresT>
    void RevertToBestSchedule(unsigned startMove, unsigned insertStep, CommDatastructuresT& commDatastructures,
                              ThreadDataT& threadData, unsigned startStep, unsigned& endStep)
    {
        const unsigned bound = std::max(startMove, threadData.bestScheduleIdx_);
        RevertMoves(bound, commDatastructures, threadData, startStep, endStep);

        if (startMove > threadData.bestScheduleIdx_) {
            SwapEmptyStepBwd(++endStep, insertStep);
        }

        RevertMoves(threadData.bestScheduleIdx_, commDatastructures, threadData, startStep, endStep);

        threadData.appliedMoves_.clear();
        threadData.bestScheduleIdx_ = 0;
        threadData.currentViolations_.clear();
        threadData.feasible_ = true;
        threadData.cost_ = threadData.bestCost_;
    }

    template <typename CommDatastructuresT>
    void RevertScheduleToBound(const size_t bound, const CostT newCost, const bool isFeasible,
                               CommDatastructuresT& commDatastructures, ThreadDataT& threadData, unsigned startStep,
                               unsigned endStep)
    {
        RevertMoves(bound, commDatastructures, threadData, startStep, endStep);

        threadData.currentViolations_.clear();
        threadData.feasible_ = isFeasible;
        threadData.cost_ = newCost;
    }

    void ComputeViolations(ThreadDataT& threadData);
    void ComputeWorkMemoryDatastructures(unsigned startStep, unsigned endStep);
    void WriteSchedule(BspSchedule<GraphT>& schedule);
    inline void Initialize(const BspSchedule<GraphT>& schedule);
    inline void Clear();
    void SwapEmptyStepFwd(const unsigned step, const unsigned toStep);
    void SwapEmptyStepBwd(const unsigned toStep, const unsigned emptyStep);
    void SwapSteps(const unsigned step1, const unsigned step2);

private:
    const BspInstance<GraphT>* instance_;

    BspSchedule<GraphT> vectorSchedule_;
    SetSchedule<GraphT> setSchedule_;

    CostT cost_ = 0;
    bool feasible_ = true;

    template <typename CommDatastructuresT>
    void RevertMoves(const size_t bound, CommDatastructuresT& commDatastructures, ThreadDataT& threadData,
                     unsigned startStep, unsigned endStep)
    {
        while (threadData.appliedMoves_.size() > bound) {
            const auto move = threadData.appliedMoves_.back().ReverseMove();
            threadData.appliedMoves_.pop_back();

            vectorSchedule_.SetAssignedProcessor(move.node_, move.toProc_);
            vectorSchedule_.SetAssignedSuperstep(move.node_, move.toStep_);

            setSchedule_.GetProcessorStepVertices()[move.fromStep_][move.fromProc_].erase(move.node_);
            setSchedule_.GetProcessorStepVertices()[move.toStep_][move.toProc_].insert(move.node_);
            workDatastructures_.ApplyMove(move, instance_->GetComputationalDag().VertexWorkWeight(move.node_));
            commDatastructures.UpdateDatastructureAfterMove(move, startStep, endStep);
        }
    }

    bool IsOutEdgeViolation(unsigned nodeStep, unsigned nodeProc, VertexType neighbor) const
    {
        const unsigned neighborStep = vectorSchedule_.AssignedSuperstep(neighbor);
        const unsigned neighborProc = vectorSchedule_.AssignedProcessor(neighbor);
        return (nodeStep > neighborStep) || (nodeStep == neighborStep && nodeProc != neighborProc);
    }

    bool IsInEdgeViolation(unsigned nodeStep, unsigned nodeProc, VertexType neighbor) const
    {
        const unsigned neighborStep = vectorSchedule_.AssignedSuperstep(neighbor);
        const unsigned neighborProc = vectorSchedule_.AssignedProcessor(neighbor);
        return (nodeStep < neighborStep) || (nodeStep == neighborStep && nodeProc != neighborProc);
    }

    template <typename IsViolationFn>
    void ProcessEdgeViolation(const EdgeType& edge, VertexType neighbor, IsViolationFn&& isViolation,
                              ThreadDataT& threadData)
    {
        const bool currentlyViolated = threadData.currentViolations_.find(edge) != threadData.currentViolations_.end();
        if (!currentlyViolated) {
            if (isViolation()) {
                threadData.currentViolations_.insert(edge);
                threadData.newViolations_[neighbor] = edge;
            }
        } else {
            if (!isViolation()) {
                threadData.currentViolations_.erase(edge);
                threadData.resolvedViolations_.insert(edge);
            }
        }
    }

    void UpdateViolations(VertexType node, ThreadDataT& threadData)
    {
        threadData.newViolations_.clear();
        threadData.resolvedViolations_.clear();

        const unsigned nodeStep = vectorSchedule_.AssignedSuperstep(node);
        const unsigned nodeProc = vectorSchedule_.AssignedProcessor(node);

        for (const auto& edge : OutEdges(node, instance_->GetComputationalDag())) {
            const auto& child = Target(edge, instance_->GetComputationalDag());
            ProcessEdgeViolation(
                edge, child, [&]() { return IsOutEdgeViolation(nodeStep, nodeProc, child); }, threadData);
        }

        for (const auto& edge : InEdges(node, instance_->GetComputationalDag())) {
            const auto& parent = Source(edge, instance_->GetComputationalDag());
            ProcessEdgeViolation(
                edge, parent, [&]() { return IsInEdgeViolation(nodeStep, nodeProc, parent); }, threadData);
        }

        threadData.feasible_ = threadData.currentViolations_.empty();
    }
};

template <typename GraphT, typename CostT>
void KlActiveSchedule<GraphT, CostT>::Clear()
{
    workDatastructures_.Clear();
    vectorSchedule_.Clear();
    setSchedule_.Clear();
}

template <typename GraphT, typename CostT>
void KlActiveSchedule<GraphT, CostT>::ComputeViolations(ThreadDataT& threadData)
{
    threadData.currentViolations_.clear();
    threadData.feasible_ = true;

    for (const auto& edge : Edges(instance_->GetComputationalDag())) {
        const auto& sourceV = Source(edge, instance_->GetComputationalDag());
        const auto& targetV = Target(edge, instance_->GetComputationalDag());

        const unsigned sourceProc = AssignedProcessor(sourceV);
        const unsigned targetProc = AssignedProcessor(targetV);
        const unsigned sourceStep = AssignedSuperstep(sourceV);
        const unsigned targetStep = AssignedSuperstep(targetV);
        if (sourceStep > targetStep || (sourceStep == targetStep && sourceProc != targetProc)) {
            threadData.currentViolations_.insert(edge);
            threadData.feasible_ = false;
        }
    }
}

template <typename GraphT, typename CostT>
void KlActiveSchedule<GraphT, CostT>::Initialize(const BspSchedule<GraphT>& schedule)
{
    instance_ = &schedule.GetInstance();
    vectorSchedule_ = BspSchedule(schedule);
    setSchedule_ = SetSchedule(schedule);
    workDatastructures_.Initialize(setSchedule_, *instance_, NumSteps());

    cost_ = 0;
    feasible_ = true;

    ComputeWorkMemoryDatastructures(0, NumSteps() - 1);
}

template <typename GraphT, typename CostT>
void KlActiveSchedule<GraphT, CostT>::ComputeWorkMemoryDatastructures(unsigned startStep, unsigned endStep)
{
    workDatastructures_.ComputeWorkDatastructures(startStep, endStep);
}

template <typename GraphT, typename CostT>
void KlActiveSchedule<GraphT, CostT>::WriteSchedule(BspSchedule<GraphT>& schedule)
{
    for (const auto v : instance_->Vertices()) {
        schedule.SetAssignedProcessor(v, vectorSchedule_.AssignedProcessor(v));
        schedule.SetAssignedSuperstep(v, vectorSchedule_.AssignedSuperstep(v));
    }
    schedule.UpdateNumberOfSupersteps();
}

template <typename GraphT, typename CostT>
void KlActiveSchedule<GraphT, CostT>::SwapEmptyStepFwd(const unsigned step, const unsigned toStep)
{
    for (unsigned i = step; i < toStep; i++) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.GetProcessorStepVertices()[i + 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.GetProcessorStepVertices()[i], setSchedule_.GetProcessorStepVertices()[i + 1]);
        workDatastructures_.SwapSteps(i, i + 1);
    }
}

template <typename GraphT, typename CostT>
void KlActiveSchedule<GraphT, CostT>::SwapEmptyStepBwd(const unsigned toStep, const unsigned emptyStep)
{
    unsigned i = toStep;

    for (; i > emptyStep; i--) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.GetProcessorStepVertices()[i - 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.GetProcessorStepVertices()[i], setSchedule_.GetProcessorStepVertices()[i - 1]);
        workDatastructures_.SwapSteps(i - 1, i);
    }
}

template <typename GraphT, typename CostT>
void KlActiveSchedule<GraphT, CostT>::SwapSteps(const unsigned step1, const unsigned step2)
{
    if (step1 == step2) {
        return;
    }

    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
        for (const auto node : setSchedule_.GetProcessorStepVertices()[step1][proc]) {
            vectorSchedule_.SetAssignedSuperstep(node, step2);
        }
        for (const auto node : setSchedule_.GetProcessorStepVertices()[step2][proc]) {
            vectorSchedule_.SetAssignedSuperstep(node, step1);
        }
    }
    std::swap(setSchedule_.GetProcessorStepVertices()[step1], setSchedule_.GetProcessorStepVertices()[step2]);
    workDatastructures_.SwapSteps(step1, step2);
}
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_KL_ACTIVE_SCHEDULE_HPP
