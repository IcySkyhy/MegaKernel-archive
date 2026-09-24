/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OSP_SET_SCHEDULE_H
#define OSP_SET_SCHEDULE_H

#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/bsp/model/bsp_schedule.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @class SetSchedule
 *
 * The SetSchedule stores the assignment of nodes to processors and supersteps in a vector of unordered sets.
 * Each element in the vector represents a superstep and contains a set of nodes for each processor.
 * This class is useful for cases where all nodes of a superstep/processor pair need to be enumerated often.
 *
 * @tparam GraphT The type of the computational DAG.
 */
template <typename GraphT>
class SetSchedule {
public:
    using VertexIdx = VertexIdxT<GraphT>;

    SetSchedule() = default;

    /**
     * @brief Constructs a SetSchedule from another BspSchedule.
     * @param schedule The source schedule to copy from.
     */
    SetSchedule(const BspSchedule<GraphT>& schedule)
        : instance_(&schedule.GetInstance()), numberOfSupersteps_(schedule.NumberOfSupersteps())
    {
        stepProcessorVertices_.resize(schedule.NumberOfSupersteps(), std::vector<std::unordered_set<VertexIdx>>(
                                                                         schedule.GetInstance().NumberOfProcessors()));

        for (const auto v : schedule.GetInstance().Vertices()) {
            const unsigned step = schedule.AssignedSuperstep(v);
            const unsigned proc = schedule.AssignedProcessor(v);
            if (step < numberOfSupersteps_ && proc < instance_->NumberOfProcessors()) {
                stepProcessorVertices_[step][proc].insert(v);
            }
        }
    }

    ~SetSchedule() = default;
    void Clear()
    {
        stepProcessorVertices_.clear();
        numberOfSupersteps_ = 0;
    }

    [[nodiscard]] const BspInstance<GraphT>& GetInstance() const { return *instance_; }
    [[nodiscard]] unsigned NumberOfSupersteps() const { return numberOfSupersteps_; }

    /**
     * @brief Get the internal node assignment structure.
     * @return Reference to the vector of vectors of unordered sets of vertices.
     */
    [[nodiscard]] const std::vector<std::vector<std::unordered_set<VertexIdx>>>& GetProcessorStepVertices() const
    {
        return stepProcessorVertices_;
    }
    [[nodiscard]] std::vector<std::vector<std::unordered_set<VertexIdx>>>& GetProcessorStepVertices()
    {
        return stepProcessorVertices_;
    }

private:
    const BspInstance<GraphT>* instance_ = nullptr;

    unsigned numberOfSupersteps_ = 0;
    std::vector<std::vector<std::unordered_set<VertexIdx>>> stepProcessorVertices_;
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_SET_SCHEDULE_H
