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
 * \file kl_active_schedule_types.h
 * \brief
 */

#ifndef OSP_KL_ACTIVE_SCHEDULE_TYPES_H
#define OSP_KL_ACTIVE_SCHEDULE_TYPES_H

#include "passes/algorithms/osp/bsp/model/bsp_schedule.h"
#include "passes/algorithms/osp/bsp/model/util/set_schedule.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_util.h"

namespace npu::tile_fwk {
namespace osp {
template <typename CostT, typename VertexIdxT>
struct KlMoveStruct {
    VertexIdxT node_;
    CostT gain_;

    unsigned fromProc_;
    unsigned fromStep_;

    unsigned toProc_;
    unsigned toStep_;

    KlMoveStruct() : node_(0), gain_(0), fromProc_(0), fromStep_(0), toProc_(0), toStep_(0) {}

    KlMoveStruct(VertexIdxT node, CostT gain, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep)
        : node_(node), gain_(gain), fromProc_(fromProc), fromStep_(fromStep), toProc_(toProc), toStep_(toStep)
    {}

    bool operator<(KlMoveStruct<CostT, VertexIdxT> const& rhs) const
    {
        return (gain_ < rhs.gain_) or (gain_ == rhs.gain_ and node_ > rhs.node_);
    }

    bool operator>(KlMoveStruct<CostT, VertexIdxT> const& rhs) const
    {
        return (gain_ > rhs.gain_) or (gain_ >= rhs.gain_ and node_ < rhs.node_);
    }

    KlMoveStruct<CostT, VertexIdxT> ReverseMove() const
    {
        return KlMoveStruct(node_, -gain_, toProc_, toStep_, fromProc_, fromStep_);
    }
};

template <typename WorkWeightT>
struct PreMoveWorkData {
    WorkWeightT fromStepMaxWork_;
    WorkWeightT fromStepSecondMaxWork_;
    unsigned fromStepMaxWorkProcessorCount_;

    WorkWeightT toStepMaxWork_;
    WorkWeightT toStepSecondMaxWork_;
    unsigned toStepMaxWorkProcessorCount_;

    PreMoveWorkData() {}

    PreMoveWorkData(WorkWeightT fromStepMaxWork, WorkWeightT fromStepSecondMaxWork,
                    unsigned fromStepMaxWorkProcessorCount, WorkWeightT toStepMaxWork, WorkWeightT toStepSecondMaxWork,
                    unsigned toStepMaxWorkProcessorCount)
        : fromStepMaxWork_(fromStepMaxWork),
          fromStepSecondMaxWork_(fromStepSecondMaxWork),
          fromStepMaxWorkProcessorCount_(fromStepMaxWorkProcessorCount),
          toStepMaxWork_(toStepMaxWork),
          toStepSecondMaxWork_(toStepSecondMaxWork),
          toStepMaxWorkProcessorCount_(toStepMaxWorkProcessorCount)
    {}
};

template <typename GraphT>
struct KlActiveScheduleWorkDatastructures {
    using WorkWeightT = VWorkwT<GraphT>;

    const BspInstance<GraphT>* instance_;
    const SetSchedule<GraphT>* setSchedule_;

    struct WeightProc {
        WorkWeightT work_;
        unsigned proc_;

        WeightProc() : work_(0), proc_(0) {}

        WeightProc(WorkWeightT work, unsigned proc) : work_(work), proc_(proc) {}

        bool operator<(WeightProc const& rhs) const
        {
            return (work_ > rhs.work_) or (work_ == rhs.work_ and proc_ < rhs.proc_);
        }
    };

    std::vector<std::vector<WeightProc>> stepProcessorWork_;
    std::vector<std::vector<unsigned>> stepProcessorPosition_;
    std::vector<unsigned> stepMaxWorkProcessorCount_;
    WorkWeightT maxWorkWeight_;
    WorkWeightT totalWorkWeight_;

    inline WorkWeightT StepMaxWork(unsigned step) const { return stepProcessorWork_[step][0].work_; }

    inline WorkWeightT StepSecondMaxWork(unsigned step) const
    {
        return stepProcessorWork_[step][stepMaxWorkProcessorCount_[step]].work_;
    }

    inline WorkWeightT StepProcWork(unsigned step, unsigned proc) const
    {
        return stepProcessorWork_[step][stepProcessorPosition_[step][proc]].work_;
    }

    inline WorkWeightT& StepProcWork(unsigned step, unsigned proc)
    {
        return stepProcessorWork_[step][stepProcessorPosition_[step][proc]].work_;
    }

    template <typename CostT, typename VertexIdxT>
    inline PreMoveWorkData<WorkWeightT> GetPreMoveWorkData(KlMoveStruct<CostT, VertexIdxT> move)
    {
        return PreMoveWorkData<WorkWeightT>(StepMaxWork(move.fromStep_), StepSecondMaxWork(move.fromStep_),
                                            stepMaxWorkProcessorCount_[move.fromStep_], StepMaxWork(move.toStep_),
                                            StepSecondMaxWork(move.toStep_), stepMaxWorkProcessorCount_[move.toStep_]);
    }

    inline void Initialize(const SetSchedule<GraphT>& sched, const BspInstance<GraphT>& inst, unsigned numSteps)
    {
        instance_ = &inst;
        setSchedule_ = &sched;
        maxWorkWeight_ = 0;
        totalWorkWeight_ = 0;
        stepProcessorWork_.assign(numSteps, std::vector<WeightProc>(instance_->NumberOfProcessors()));
        stepProcessorPosition_.assign(numSteps, std::vector<unsigned>(instance_->NumberOfProcessors(), 0));
        stepMaxWorkProcessorCount_.assign(numSteps, 0);
    }

    inline void Clear()
    {
        stepProcessorWork_.clear();
        stepProcessorPosition_.clear();
        stepMaxWorkProcessorCount_.clear();
    }

    inline void ArrangeSuperstepData(const unsigned step)
    {
        std::sort(stepProcessorWork_[step].begin(), stepProcessorWork_[step].end());
        unsigned pos = 0;
        const WorkWeightT maxWorkTo = stepProcessorWork_[step][0].work_;

        for (const auto& wp : stepProcessorWork_[step]) {
            stepProcessorPosition_[step][wp.proc_] = pos++;

            if (wp.work_ == maxWorkTo && pos < instance_->NumberOfProcessors()) {
                stepMaxWorkProcessorCount_[step] = pos;
            }
        }
    }

    template <typename CostT, typename VertexIdxT>
    void ApplyMove(KlMoveStruct<CostT, VertexIdxT> move, WorkWeightT workWeight)
    {
        if (workWeight == 0) {
            return;
        }

        if (move.toStep_ != move.fromStep_) {
            StepProcWork(move.toStep_, move.toProc_) += workWeight;
            StepProcWork(move.fromStep_, move.fromProc_) -= workWeight;

            ArrangeSuperstepData(move.toStep_);
            ArrangeSuperstepData(move.fromStep_);
        } else {
            StepProcWork(move.toStep_, move.toProc_) += workWeight;
            StepProcWork(move.fromStep_, move.fromProc_) -= workWeight;
            ArrangeSuperstepData(move.toStep_);
        }
    }

    void SwapSteps(const unsigned step1, const unsigned step2)
    {
        std::swap(stepProcessorWork_[step1], stepProcessorWork_[step2]);
        std::swap(stepProcessorPosition_[step1], stepProcessorPosition_[step2]);
        std::swap(stepMaxWorkProcessorCount_[step1], stepMaxWorkProcessorCount_[step2]);
    }

    void OverrideNextSuperstep(unsigned step)
    {
        const unsigned nextStep = step + 1;
        for (unsigned i = 0; i < instance_->NumberOfProcessors(); i++) {
            stepProcessorWork_[nextStep][i] = stepProcessorWork_[step][i];
            stepProcessorPosition_[nextStep][i] = stepProcessorPosition_[step][i];
        }
        stepMaxWorkProcessorCount_[nextStep] = stepMaxWorkProcessorCount_[step];
    }

    void ResetSuperstep(unsigned step)
    {
        for (unsigned i = 0; i < instance_->NumberOfProcessors(); i++) {
            stepProcessorWork_[step][i] = {0, i};
            stepProcessorPosition_[step][i] = i;
        }
        stepMaxWorkProcessorCount_[step] = instance_->NumberOfProcessors() - 1;
    }

    void ComputeWorkDatastructures(unsigned startStep, unsigned endStep)
    {
        for (unsigned step = startStep; step <= endStep; step++) {
            stepMaxWorkProcessorCount_[step] = 0;
            WorkWeightT maxWork = 0;

            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
                stepProcessorWork_[step][proc].work_ = 0;
                stepProcessorWork_[step][proc].proc_ = proc;

                for (const auto& node : setSchedule_->GetProcessorStepVertices()[step][proc]) {
                    const WorkWeightT vertexWorkWeight = instance_->GetComputationalDag().VertexWorkWeight(node);
                    totalWorkWeight_ += vertexWorkWeight;
                    maxWorkWeight_ = std::max(vertexWorkWeight, maxWorkWeight_);
                    stepProcessorWork_[step][proc].work_ += vertexWorkWeight;
                }

                if (stepProcessorWork_[step][proc].work_ > maxWork) {
                    maxWork = stepProcessorWork_[step][proc].work_;
                    stepMaxWorkProcessorCount_[step] = 1;
                } else if (stepProcessorWork_[step][proc].work_ == maxWork &&
                           stepMaxWorkProcessorCount_[step] < (instance_->NumberOfProcessors() - 1)) {
                    stepMaxWorkProcessorCount_[step]++;
                }
            }

            std::sort(stepProcessorWork_[step].begin(), stepProcessorWork_[step].end());
            unsigned pos = 0;
            for (const auto& wp : stepProcessorWork_[step]) {
                stepProcessorPosition_[step][wp.proc_] = pos++;
            }
        }
    }
};

template <typename GraphT, typename CostT>
struct ThreadLocalActiveScheduleData {
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

    using KlMove = KlMoveStruct<CostT, VertexType>;

    std::unordered_set<EdgeType> currentViolations_;
    std::vector<KlMove> appliedMoves_;

    CostT cost_ = 0;
    CostT initialCost_ = 0;
    bool feasible_ = true;

    CostT bestCost_ = 0;
    unsigned bestScheduleIdx_ = 0;

    std::unordered_map<VertexType, EdgeType> newViolations_;
    std::unordered_set<EdgeType> resolvedViolations_;

    inline void InitializeCost(CostT cost)
    {
        initialCost_ = cost;
        cost_ = cost;
        bestCost_ = cost;
        feasible_ = true;
    }

    inline void UpdateCost(CostT changeInCost)
    {
        cost_ += changeInCost;

        if (cost_ <= bestCost_ && feasible_) {
            bestCost_ = cost_;
            bestScheduleIdx_ = static_cast<unsigned>(appliedMoves_.size());
        }
    }
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_KL_ACTIVE_SCHEDULE_TYPES_H
