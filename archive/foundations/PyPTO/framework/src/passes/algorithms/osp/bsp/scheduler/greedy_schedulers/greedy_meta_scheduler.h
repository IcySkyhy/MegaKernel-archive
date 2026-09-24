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
 * \file greedy_meta_scheduler.h
 * \brief
 */

#ifndef OSP_GREEDY_META_SCHEDULER_H
#define OSP_GREEDY_META_SCHEDULER_H

#include <string>
#include <vector>

#include "passes/algorithms/osp/bsp/model/cost/lazy_communication_cost.h"
#include "passes/algorithms/osp/bsp/scheduler/scheduler.h"
#include "passes/algorithms/osp/bsp/scheduler/serial.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @class GreedyMetaScheduler
 * @brief The GreedyMetaScheduler class represents a meta-scheduler that
 * selects the best schedule produced from a list of added schedulers.
 *
 * This class inherits from the Scheduler class and implements the ComputeSchedule() methods.
 * The ComputeSchedule() method iterates through a list of schedulers, computes a schedule using each one,
 * and returns the schedule with the minimum cost.
 *
 * @tparam GraphT The graph type representing the computational DAG.
 * @tparam CostModel The cost model functor to evaluate schedules. Defaults to LazyCommunicationCost.
 */
template <typename GraphT, typename CostModel = LazyCommunicationCost<GraphT>>
class GreedyMetaScheduler : public Scheduler<GraphT> {
    Serial<GraphT> serialScheduler_;
    std::vector<Scheduler<GraphT>*> schedulers_;

public:
    GreedyMetaScheduler() : Scheduler<GraphT>() {}
    ~GreedyMetaScheduler() override = default;

    void AddSerialScheduler() { schedulers_.push_back(&serialScheduler_); }

    void AddScheduler(Scheduler<GraphT>& s) { schedulers_.push_back(&s); }

    void ResetScheduler() { schedulers_.clear(); }

    ReturnStatus ComputeSchedule(BspSchedule<GraphT>& schedule) override
    {
        if (schedule.GetInstance().GetArchitecture().NumberOfProcessors() == 1) {
            serialScheduler_.ComputeSchedule(schedule);
            return ReturnStatus::OSP_SUCCESS;
        }

        VWorkwT<GraphT> bestScheduleCost = std::numeric_limits<VWorkwT<GraphT>>::max();
        BspSchedule<GraphT> currentSchedule(schedule.GetInstance());

        for (Scheduler<GraphT>* scheduler : schedulers_) {
            scheduler->ComputeSchedule(currentSchedule);
            const VWorkwT<GraphT> scheduleCost = CostModel()(currentSchedule);
            if (scheduleCost < bestScheduleCost) {
                bestScheduleCost = scheduleCost;
                schedule = currentSchedule;
            }
        }

        return ReturnStatus::OSP_SUCCESS;
    }
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_GREEDY_META_SCHEDULER_HPP
