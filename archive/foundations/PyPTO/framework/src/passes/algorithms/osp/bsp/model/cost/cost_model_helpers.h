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
 * \file cost_model_helpers.h
 * \brief
 */

#ifndef OSP_COST_MODEL_HELPERS_H
#define OSP_COST_MODEL_HELPERS_H

#include <algorithm>
#include <vector>

#include "passes/algorithms/osp/bsp/model/bsp_instance.h"

namespace npu::tile_fwk {
namespace osp {
template <typename GraphT>
class BspSchedule;

namespace cost_helpers {
template <typename GraphT>
std::vector<VCommwT<GraphT>> ComputeMaxCommPerStep(const BspInstance<GraphT>& instance, unsigned numberOfSupersteps,
                                                   const std::vector<std::vector<VCommwT<GraphT>>>& rec,
                                                   const std::vector<std::vector<VCommwT<GraphT>>>& send)
{
    std::vector<VCommwT<GraphT>> maxCommPerStep(numberOfSupersteps, 0);
    for (unsigned step = 0; step < numberOfSupersteps; step++) {
        VCommwT<GraphT> maxSend = 0;
        VCommwT<GraphT> maxRec = 0;

        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
            if (maxSend < send[proc][step]) {
                maxSend = send[proc][step];
            }
            if (maxRec < rec[proc][step]) {
                maxRec = rec[proc][step];
            }
        }
        maxCommPerStep[step] = std::max(maxSend, maxRec) * instance.CommunicationCosts();
    }
    return maxCommPerStep;
}

template <typename GraphT>
std::vector<VCommwT<GraphT>> ComputeMaxCommPerStep(const BspSchedule<GraphT>& schedule,
                                                   const std::vector<std::vector<VCommwT<GraphT>>>& rec,
                                                   const std::vector<std::vector<VCommwT<GraphT>>>& send)
{
    return ComputeMaxCommPerStep(schedule.GetInstance(), schedule.NumberOfSupersteps(), rec, send);
}

template <typename GraphT>
std::vector<VWorkwT<GraphT>> ComputeMaxWorkPerStep(const BspInstance<GraphT>& instance, unsigned numberOfSupersteps,
                                                   const std::vector<unsigned>& nodeToProcessorAssignment,
                                                   const std::vector<unsigned>& nodeToSuperstepAssignment)
{
    std::vector<std::vector<VWorkwT<GraphT>>> work = std::vector<std::vector<VWorkwT<GraphT>>>(
        numberOfSupersteps, std::vector<VWorkwT<GraphT>>(instance.NumberOfProcessors(), 0));
    for (const auto& node : instance.Vertices()) {
        work[nodeToSuperstepAssignment[node]][nodeToProcessorAssignment[node]] += instance.GetComputationalDag()
                                                                                      .VertexWorkWeight(node);
    }

    std::vector<VWorkwT<GraphT>> maxWorkPerStep(numberOfSupersteps, 0);
    for (unsigned step = 0; step < numberOfSupersteps; step++) {
        VWorkwT<GraphT> maxWork = 0;
        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
            if (maxWork < work[step][proc]) {
                maxWork = work[step][proc];
            }
        }

        maxWorkPerStep[step] = maxWork;
    }

    return maxWorkPerStep;
}

template <typename GraphT>
std::vector<VWorkwT<GraphT>> ComputeMaxWorkPerStep(const BspSchedule<GraphT>& schedule)
{
    return ComputeMaxWorkPerStep(schedule.GetInstance(), schedule.NumberOfSupersteps(), schedule.AssignedProcessors(),
                                 schedule.AssignedSupersteps());
}

template <typename GraphT>
VWorkwT<GraphT> ComputeWorkCosts(const BspInstance<GraphT>& instance, unsigned numberOfSupersteps,
                                 const std::vector<unsigned>& nodeToProcessorAssignment,
                                 const std::vector<unsigned>& nodeToSuperstepAssignment)
{
    std::vector<VWorkwT<GraphT>> maxWorkPerStep = ComputeMaxWorkPerStep(
        instance, numberOfSupersteps, nodeToProcessorAssignment, nodeToSuperstepAssignment);

    return std::accumulate(maxWorkPerStep.begin(), maxWorkPerStep.end(), static_cast<VWorkwT<GraphT>>(0));
}

template <typename GraphT>
VWorkwT<GraphT> ComputeWorkCosts(const BspSchedule<GraphT>& schedule)
{
    return ComputeWorkCosts(schedule.GetInstance(), schedule.NumberOfSupersteps(), schedule.AssignedProcessors(),
                            schedule.AssignedSupersteps());
}
} // namespace cost_helpers
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_COST_MODEL_HELPERS_H
