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
 * \file trimmed_group_scheduler.h
 * \brief
 */

#ifndef OSP_TRIMMED_GROUP_SCHEDULER_H
#define OSP_TRIMMED_GROUP_SCHEDULER_H

#include <numeric>
#include <string>
#include <vector>

#include "passes/algorithms/osp/bsp/scheduler/scheduler.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_util.h"
#include "passes/algorithms/osp/graph_algorithms/subgraph_algorithms.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @class TrimmedGroupScheduler
 * @brief A scheduler for a single trimmed group consisting of multiple isomorphic connected components.
 *
 * This scheduler partitions a disconnected subgraph (a pruned group) into its weakly connected components.
 * It assumes these components are isomorphic and distributes them among the available processor groups
 * to balance the load.
 *
 * @tparam ConstrGraphT The type of the graph.
 */
template <typename ConstrGraphT>
class TrimmedGroupScheduler : public Scheduler<ConstrGraphT> {
    Scheduler<ConstrGraphT>* subScheduler_;
    unsigned minNonZeroProcs_;

public:
    /**
     * @brief Constructs a TrimmedGroupScheduler.
     * @param scheduler The sub-scheduler to use for scheduling individual component groups.
     * @param minNonZeroProcs The minimum number of non-zero processors to utilize.
     */
    TrimmedGroupScheduler(Scheduler<ConstrGraphT>& scheduler, unsigned minNonZeroProcs)
        : subScheduler_(&scheduler), minNonZeroProcs_(minNonZeroProcs)
    {}

    ReturnStatus ComputeSchedule(BspSchedule<ConstrGraphT>& schedule) override
    {
        const auto& instance = schedule.GetInstance();
        const ConstrGraphT& dag = instance.GetComputationalDag();

        std::vector<VertexIdxT<ConstrGraphT>> componentMap(dag.NumVertices());
        size_t numComponents = ComputeWeaklyConnectedComponents(dag, componentMap);
        if (numComponents == 0) {
            schedule.SetNumberOfSupersteps(0);
            return ReturnStatus::OSP_SUCCESS;
        }

        std::vector<std::vector<VertexIdxT<ConstrGraphT>>> componentsVertices(numComponents);
        for (VertexIdxT<ConstrGraphT> v = 0; v < dag.NumVertices(); ++v) {
            componentsVertices[componentMap[v]].push_back(v);
        }

        auto componentIndicesPerGroup = DistributeComponents(numComponents);
        auto subArch = BuildSubArchitecture(instance.GetArchitecture());

        return SolveAndMapSubProblems(schedule, componentIndicesPerGroup, componentsVertices, subArch);
    }

private:
    /**
     * @brief Distributes components among the processor groups.
     * @param numComponents Total number of components.
     * @return A vector where each element is a list of component indices assigned to a processor group.
     */
    std::vector<std::vector<unsigned>> DistributeComponents(size_t numComponents)
    {
        const unsigned baseCount = static_cast<unsigned>(numComponents) / minNonZeroProcs_;
        const unsigned remainder = static_cast<unsigned>(numComponents) % minNonZeroProcs_;

        std::vector<std::vector<unsigned>> componentIndicesPerGroup(minNonZeroProcs_);
        unsigned componentCursor = 0;
        for (unsigned i = 0; i < minNonZeroProcs_; ++i) {
            unsigned numToAssign = baseCount + (i < remainder ? 1 : 0);
            for (unsigned j = 0; j < numToAssign; ++j) {
                if (componentCursor < numComponents) {
                    componentIndicesPerGroup[i].push_back(componentCursor++);
                }
            }
        }
        return componentIndicesPerGroup;
    }

    /**
     * @brief Builds the architecture for a single sub-problem (one processor group).
     * @param arch The global architecture.
     * @return The sub-architecture.
     */
    BspArchitecture<ConstrGraphT> BuildSubArchitecture(const BspArchitecture<ConstrGraphT>& arch)
    {
        std::vector<unsigned> subProcCounts(arch.GetNumberOfProcessorTypes());
        std::vector<VMemwT<ConstrGraphT>> memWeights(arch.GetNumberOfProcessorTypes(), 0);

        for (unsigned typeIdx = 0; typeIdx < arch.GetNumberOfProcessorTypes(); ++typeIdx) {
            subProcCounts[typeIdx] = arch.GetProcessorTypeCount()[typeIdx] / minNonZeroProcs_;
            memWeights[typeIdx] = static_cast<VMemwT<ConstrGraphT>>(arch.MaxMemoryBoundProcType(typeIdx));
        }

        BspArchitecture<ConstrGraphT> subArch(arch);
        subArch.SetProcessorsConsequTypes(subProcCounts, memWeights);
        return subArch;
    }

    /**
     * @brief Computes prefix-sum offsets from processor type counts.
     */
    static std::vector<unsigned> ComputeProcTypeOffsets(const BspArchitecture<ConstrGraphT>& arch)
    {
        std::vector<unsigned> offsets(arch.GetNumberOfProcessorTypes(), 0);
        const auto& counts = arch.GetProcessorTypeCount();
        for (unsigned t = 1; t < arch.GetNumberOfProcessorTypes(); ++t) {
            offsets[t] = offsets[t - 1] + counts[t - 1];
        }
        return offsets;
    }

    /**
     * @brief Solves the sub-schedule for each group and maps the results back to the global schedule.
     */
    ReturnStatus SolveAndMapSubProblems(BspSchedule<ConstrGraphT>& schedule,
                                        const std::vector<std::vector<unsigned>>& componentIndicesPerGroup,
                                        const std::vector<std::vector<VertexIdxT<ConstrGraphT>>>& componentsVertices,
                                        const BspArchitecture<ConstrGraphT>& subArch)
    {
        const auto& instance = schedule.GetInstance();
        const auto& dag = instance.GetComputationalDag();

        const auto archProcTypeOffsets = ComputeProcTypeOffsets(instance.GetArchitecture());
        const auto subArchProcTypeOffsets = ComputeProcTypeOffsets(subArch);
        std::vector<unsigned> subProcCounts = subArch.GetProcessorTypeCount();
        unsigned maxSupersteps = 0;

        for (unsigned i = 0; i < minNonZeroProcs_; ++i) {
            if (componentIndicesPerGroup[i].empty())
                continue;

            std::vector<VertexIdxT<ConstrGraphT>> groupVertices;
            for (unsigned compIdx : componentIndicesPerGroup[i]) {
                groupVertices.insert(groupVertices.end(), componentsVertices[compIdx].begin(),
                                     componentsVertices[compIdx].end());
            }
            std::sort(groupVertices.begin(), groupVertices.end());

            BspInstance<ConstrGraphT> subInstance;
            subInstance.GetArchitecture() = subArch;
            subInstance.SetNodeProcessorCompatibility(instance.GetNodeProcessorCompatibilityMatrix());

            auto globalToLocalMap = CreateInducedSubgraphMap(dag, subInstance.GetComputationalDag(), groupVertices);

            BspSchedule<ConstrGraphT> subSchedule(subInstance);
            auto status = subScheduler_->ComputeSchedule(subSchedule);
            if (status != ReturnStatus::OSP_SUCCESS && status != ReturnStatus::OSP_BEST_FOUND) {
                return status;
            }

            for (const auto& vGlobal : groupVertices) {
                const auto vLocal = globalToLocalMap.at(vGlobal);
                const unsigned subProc = subSchedule.AssignedProcessor(vLocal);
                const unsigned subSuperstep = subSchedule.AssignedSuperstep(vLocal);

                const unsigned procType = subArch.ProcessorType(subProc);
                const unsigned localIdxWithinType = subProc - subArchProcTypeOffsets[procType];
                const unsigned globalProc = archProcTypeOffsets[procType] + (i * subProcCounts[procType]) +
                                            localIdxWithinType;

                schedule.SetAssignedProcessor(vGlobal, globalProc);
                schedule.SetAssignedSuperstep(vGlobal, subSuperstep);
            }
            maxSupersteps = std::max(maxSupersteps, subSchedule.NumberOfSupersteps());
        }

        schedule.SetNumberOfSupersteps(maxSupersteps);
        return ReturnStatus::OSP_SUCCESS;
    }
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_TRIMMED_GROUP_SCHEDULER_HPP
