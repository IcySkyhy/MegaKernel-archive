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
 * \file serial.h
 * \brief
 */

#ifndef OSP_SERIAL_H
#define OSP_SERIAL_H

#include <deque>
#include <limits>
#include <string>
#include <vector>

#include "scheduler.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @class Serial
 * @brief The Serial class represents a scheduler that assigns all tasks to a single processor in a serial manner.
 * If the architecture is heterogeneous, it assigns tasks to one processor of each type computing a schedule with the
 * smallest number of supersteps.
 *
 */
template <typename GraphT>
class Serial : public Scheduler<GraphT> {
public:
    /**
     * @brief Default constructor for Serial.
     */
    Serial() : Scheduler<GraphT>() {}

    /**
     * @brief Default destructor for Serial.
     */
    ~Serial() override = default;

    ReturnStatus ComputeSchedule(BspSchedule<GraphT>& schedule) override
    {
        const auto& instance = schedule.GetInstance();
        const auto& dag = instance.GetComputationalDag();
        const auto numVertices = dag.NumVertices();
        if (numVertices == 0) {
            return ReturnStatus::OSP_SUCCESS;
        }

        const auto& arch = instance.GetArchitecture();

        // Select one processor of each type
        const std::vector<unsigned> chosenProcs = SelectProcessor(arch);
        if (chosenProcs.empty()) {
            return ReturnStatus::OSP_ERROR;
        }

        // Build compatibility matrix
        const std::vector<std::vector<unsigned>> nodeTypeCompatibleProcessors = BuildCompatibilityMatrix(instance, dag,
                                                                                                         chosenProcs);

        // Initialize scheduling state
        std::vector<VertexIdxT<GraphT>> inDegree(numVertices);
        std::deque<VertexIdxT<GraphT>> readyNodes;
        std::deque<VertexIdxT<GraphT>> deferredNodes;
        InitializeScheduleState(schedule, dag, inDegree, readyNodes);

        // Main scheduling loop
        VertexIdxT<GraphT> scheduledNodesCount = 0;
        unsigned currentSuperstep = 0;

        while (scheduledNodesCount < numVertices) {
            while (not readyNodes.empty()) {
                VertexIdxT<GraphT> v = readyNodes.front();
                readyNodes.pop_front();

                if (TryScheduleNode(schedule, dag, v, nodeTypeCompatibleProcessors, currentSuperstep, inDegree,
                                    readyNodes, deferredNodes)) {
                    ++scheduledNodesCount;
                }
            }

            if (scheduledNodesCount < numVertices) {
                currentSuperstep++;
                readyNodes.insert(readyNodes.end(), deferredNodes.begin(), deferredNodes.end());
                deferredNodes.clear();
            }
        }

        schedule.SetNumberOfSupersteps(currentSuperstep + 1);
        return ReturnStatus::OSP_SUCCESS;
    }

private:
    std::vector<std::vector<unsigned>> BuildCompatibilityMatrix(const BspInstance<GraphT>& instance, const GraphT& dag,
                                                                const std::vector<unsigned>& chosenProcs)
    {
        const unsigned numNodeTypes = dag.NumVertexTypes();
        std::vector<std::vector<unsigned>> nodeTypeCompatibleProcessors(numNodeTypes);

        for (VTypeT<GraphT> type = 0; type < numNodeTypes; ++type) {
            for (const auto& p : chosenProcs) {
                if (instance.IsCompatibleType(type, instance.ProcessorType(p))) {
                    nodeTypeCompatibleProcessors[type].push_back(p);
                }
            }
        }

        return nodeTypeCompatibleProcessors;
    }

    void InitializeScheduleState(BspSchedule<GraphT>& schedule, const GraphT& dag,
                                 std::vector<VertexIdxT<GraphT>>& inDegree, std::deque<VertexIdxT<GraphT>>& readyNodes)
    {
        for (const auto& v : dag.Vertices()) {
            schedule.SetAssignedProcessor(v, std::numeric_limits<unsigned>::max());
            schedule.SetAssignedSuperstep(v, std::numeric_limits<unsigned>::max());
            inDegree[v] = dag.InDegree(v);
            if (inDegree[v] == 0) {
                readyNodes.push_back(v);
            }
        }
    }

    bool AreParentsCompatible(const BspSchedule<GraphT>& schedule, const GraphT& dag, VertexIdxT<GraphT> v, unsigned p,
                              unsigned currentSuperstep)
    {
        for (const auto& parent : dag.Parents(v)) {
            if (schedule.AssignedSuperstep(parent) == currentSuperstep && schedule.AssignedProcessor(parent) != p) {
                return false;
            }
        }
        return true;
    }

    void UpdateDependencies(const GraphT& dag, VertexIdxT<GraphT> v, std::vector<VertexIdxT<GraphT>>& inDegree,
                            std::deque<VertexIdxT<GraphT>>& readyNodes)
    {
        for (const auto& child : dag.Children(v)) {
            if (--inDegree[child] == 0) {
                readyNodes.push_back(child);
            }
        }
    }

    bool TryScheduleNode(BspSchedule<GraphT>& schedule, const GraphT& dag, VertexIdxT<GraphT> v,
                         const std::vector<std::vector<unsigned>>& nodeTypeCompatibleProcessors,
                         unsigned currentSuperstep, std::vector<VertexIdxT<GraphT>>& inDegree,
                         std::deque<VertexIdxT<GraphT>>& readyNodes, std::deque<VertexIdxT<GraphT>>& deferredNodes)
    {
        unsigned vType = dag.VertexType(v);

        for (const auto& p : nodeTypeCompatibleProcessors[vType]) {
            if (AreParentsCompatible(schedule, dag, v, p, currentSuperstep)) {
                schedule.SetAssignedProcessor(v, p);
                schedule.SetAssignedSuperstep(v, currentSuperstep);
                UpdateDependencies(dag, v, inDegree, readyNodes);
                return true;
            }
        }

        deferredNodes.push_back(v);
        return false;
    }

    std::vector<unsigned> SelectProcessor(const BspArchitecture<GraphT>& arch)
    {
        std::vector<unsigned> chosenProcs;
        if (arch.GetNumberOfProcessorTypes() > 0) {
            std::vector<bool> typeSeen(arch.GetNumberOfProcessorTypes(), false);
            for (unsigned p = 0; p < arch.NumberOfProcessors(); ++p) {
                if (!typeSeen[arch.ProcessorType(p)]) {
                    chosenProcs.push_back(p);
                    typeSeen[arch.ProcessorType(p)] = true;
                }
            }
        }
        return chosenProcs;
    }
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_SERIAL_HPP
