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
 * \file lazy_communication_cost.h
 * \brief
 */

#ifndef OSP_LAZY_COMMUNICATION_COST_H
#define OSP_LAZY_COMMUNICATION_COST_H

#include <algorithm>
#include <vector>

#include "passes/algorithms/osp/bsp/model/cost/cost_model_helpers.h"

namespace npu::tile_fwk {
namespace osp {
template <typename GraphT>
void ComputeLazyCommunicationCosts(const BspInstance<GraphT>& instance, unsigned numberOfSupersteps,
                                   const std::vector<unsigned>& nodeToProcessorAssignment,
                                   const std::vector<unsigned>& nodeToSuperstepAssignment, const unsigned staleness,
                                   std::vector<std::vector<VCommwT<GraphT>>>& rec,
                                   std::vector<std::vector<VCommwT<GraphT>>>& send)
{
    for (const auto& node : instance.Vertices()) {
        std::vector<unsigned> stepNeeded(instance.NumberOfProcessors(), numberOfSupersteps);
        for (const auto& target : instance.GetComputationalDag().Children(node)) {
            if (nodeToProcessorAssignment[node] != nodeToProcessorAssignment[target]) {
                stepNeeded[nodeToProcessorAssignment[target]] = std::min(stepNeeded[nodeToProcessorAssignment[target]],
                                                                         nodeToSuperstepAssignment[target]);
            }
        }

        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
            if (stepNeeded[proc] < numberOfSupersteps) {
                send[nodeToProcessorAssignment[node]]
                    [stepNeeded[proc] - staleness] += instance.SendCosts(nodeToProcessorAssignment[node], proc) *
                                                      instance.GetComputationalDag().VertexCommWeight(node);
                rec[proc][stepNeeded[proc] - staleness] += instance.SendCosts(nodeToProcessorAssignment[node], proc) *
                                                           instance.GetComputationalDag().VertexCommWeight(node);
            }
        }
    }
}

template <typename GraphT>
void ComputeLazyCommunicationCosts(const BspSchedule<GraphT>& schedule, std::vector<std::vector<VCommwT<GraphT>>>& rec,
                                   std::vector<std::vector<VCommwT<GraphT>>>& send)
{
    ComputeLazyCommunicationCosts(schedule.GetInstance(), schedule.NumberOfSupersteps(), schedule.AssignedProcessors(),
                                  schedule.AssignedSupersteps(), schedule.GetStaleness(), rec, send);
}

/**
 * @struct LazyCommunicationCost
 * @brief Implements the lazy communication cost model.
 */
template <typename GraphT>
struct LazyCommunicationCost {
    using CostType = VWorkwT<GraphT>;

    CostType operator()(const BspSchedule<GraphT>& schedule) const
    {
        const auto& numberOfProcessors = schedule.GetInstance().NumberOfProcessors();
        const auto& numberOfSupersteps = schedule.NumberOfSupersteps();

        std::vector<std::vector<VCommwT<GraphT>>> rec(numberOfProcessors,
                                                      std::vector<VCommwT<GraphT>>(numberOfSupersteps, 0));
        std::vector<std::vector<VCommwT<GraphT>>> send(numberOfProcessors,
                                                       std::vector<VCommwT<GraphT>>(numberOfSupersteps, 0));

        ComputeLazyCommunicationCosts(schedule, rec, send);
        const auto maxCommPerStep = cost_helpers::ComputeMaxCommPerStep(schedule, rec, send);

        VCommwT<GraphT> commCosts = 0;
        for (unsigned step = 0; step < numberOfSupersteps; step++) {
            const auto stepCommCost = maxCommPerStep[step];
            commCosts += stepCommCost;

            if (stepCommCost > 0) {
                commCosts += schedule.GetInstance().SynchronisationCosts();
            }
        }

        return commCosts + cost_helpers::ComputeWorkCosts(schedule);
    }
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_LAZY_COMMUNICATION_COST_H
