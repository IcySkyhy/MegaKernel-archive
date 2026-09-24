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
 * \file isomorphic_subgraph_scheduler.h
 * \brief
 */

#ifndef OSP_ISOMORPHIC_SUBGRAPH_SCHEDULER_H
#define OSP_ISOMORPHIC_SUBGRAPH_SCHEDULER_H

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

#include "eft_subgraph_scheduler.h"
#include "hash_computer.h"
#include "merkle_hash_computer.h"
#include "orbit_graph_processor.h"
#include "trimmed_group_scheduler.h"
#include "passes/algorithms/osp/bsp/scheduler/scheduler.h"
#include "passes/algorithms/osp/graph_algorithms/subgraph_algorithms.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @brief A scheduler that leverages isomorphic subgraphs to partition a DAG.
 *
 * @class IsomorphicSubgraphScheduler
 *
 * This scheduler first identifies isomorphic subgraphs within the input DAG using a hash-based approach.
 * It then groups these isomorphic subgraphs into "orbits". Each orbit is treated as a single node in a
 * coarser graph. The scheduler then uses an ETF-like approach to schedule these coarse nodes (orbits)
 * onto available processors. Finally, the schedule for each orbit is "unrolled" back to the original
 * DAG, assigning a partition ID to each original vertex.
 *
 * The scheduler supports trimming of isomorphic groups to better fit processor counts, and can
 * dynamically switch between a standard BSP scheduler and a specialized TrimmedGroupScheduler
 * for these trimmed groups.
 *
 * @tparam GraphT The type of the input computational DAG.
 * @tparam ConstrGraphT The type of the constructable computational DAG used for internal representations.
 */
template <typename GraphT, typename ConstrGraphT>
class IsomorphicSubgraphScheduler {
public:
    /**
     * @brief Constructs the scheduler with a reference to a base BSP scheduler.
     * @param bspScheduler The underlying scheduler to use for scheduling individual subgraphs.
     */
    IsomorphicSubgraphScheduler(Scheduler<ConstrGraphT>& bspScheduler)
        : hashComputer_(nullptr), bspScheduler_(&bspScheduler)
    {}

    /**
     * @brief Constructs the scheduler with a base scheduler and an existing hash computer.
     * @param bspScheduler The underlying scheduler.
     * @param hashComputer The pre-computed hash computer for the graph.
     */
    IsomorphicSubgraphScheduler(Scheduler<ConstrGraphT>& bspScheduler,
                                const HashComputer<VertexIdxT<GraphT>>& hashComputer)
        : hashComputer_(&hashComputer), bspScheduler_(&bspScheduler)
    {}

    virtual ~IsomorphicSubgraphScheduler() {}
    void SetWorkThreshold(VWorkwT<ConstrGraphT> workThreshold) { workThreshold_ = workThreshold; }
    void SetMergeDifferentTypes(bool flag) { mergeDifferentNodeTypes_ = flag; }
    void SetCriticalPathThreshold(VWorkwT<ConstrGraphT> criticalPathThreshold)
    {
        criticalPathThreshold_ = criticalPathThreshold;
    }
    void SetOrbitLockRatio(double orbitLockRatio) { orbitLockRatio_ = orbitLockRatio; }
    void SetNaturalBreaksCountPercentage(double naturalBreaksCountPercentage)
    {
        naturalBreaksCountPercentage_ = naturalBreaksCountPercentage;
    }
    void SetAllowTrimmedScheduler(bool flag) { allowUseTrimmedScheduler_ = flag; }
    void DisableUseMaxGroupSize() { useMaxGroupSize_ = false; }
    void SetUseMaxBsp(bool flag) { useMaxBsp_ = flag; }
    void EnableUseMaxGroupSize(const unsigned maxGroupSize)
    {
        useMaxGroupSize_ = true;
        maxGroupSize_ = maxGroupSize;
    }

    /**
     * @brief Computes the partition of the graph.
     *
     * This is the main entry point. It discovers isomorphic groups, potentially trims them,
     * schedules the coarse groups, and then expands the schedule to the original graph elements.
     *
     * @param instance The BSP instance containing the graph and architecture.
     * @return A vector mapping each vertex index to a processor/partition ID.
     */
    std::vector<VertexIdxT<GraphT>> ComputePartition(const BspInstance<GraphT>& instance)
    {
        OrbitGraphProcessor<GraphT, ConstrGraphT> orbitProcessor;
        orbitProcessor.SetWorkThreshold(workThreshold_);
        orbitProcessor.SetMergeDifferentNodeTypes(mergeDifferentNodeTypes_);
        orbitProcessor.SetCriticalPathThreshold(criticalPathThreshold_);
        orbitProcessor.SetLockRatio(orbitLockRatio_);
        orbitProcessor.SetNaturalBreaksCountPercentage(naturalBreaksCountPercentage_);

        std::unique_ptr<HashComputer<VertexIdxT<GraphT>>> localHasher;
        if (!hashComputer_) {
            localHasher = std::make_unique<MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true>>(
                instance.GetComputationalDag(), instance.GetComputationalDag());
            hashComputer_ = localHasher.get();
        }

        orbitProcessor.DiscoverIsomorphicGroups(instance.GetComputationalDag(), *hashComputer_);

        auto isomorphicGroups = orbitProcessor.GetFinalGroups();

        std::vector<bool> wasTrimmed(isomorphicGroups.size(), false);
        // Apply trimming and record which groups were affected
        TrimSubgraphGroups(isomorphicGroups, instance, wasTrimmed);

        auto input = PrepareSubgraphSchedulingInput(instance, isomorphicGroups, wasTrimmed);

        EftSubgraphScheduler<ConstrGraphT> etfScheduler;
        SubgraphSchedule subgraphSchedule = etfScheduler.Run(input.instance_, input.multiplicities_,
                                                             input.requiredProcTypes_, input.maxNumProcessors_);
        // Pass through trimming info
        subgraphSchedule.wasTrimmed_ = std::move(wasTrimmed);

        std::vector<VertexIdxT<GraphT>> partition(instance.NumberOfVertices(), 0);
        ScheduleIsomorphicGroup(instance, isomorphicGroups, subgraphSchedule, partition);

        return partition;
    }

protected:
    template <typename GT, typename CGT>
    struct SubgraphSchedulerInput {
        BspInstance<CGT> instance_;
        std::vector<unsigned> multiplicities_;
        std::vector<unsigned> maxNumProcessors_;
        std::vector<std::vector<VWorkwT<GT>>> requiredProcTypes_;
    };

    std::pair<bool, VTypeT<GraphT>> IsSingleTypeGroup(
        const typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group& group,
        const BspInstance<GraphT>& instance) const
    {
        if (group.subgraphs_.empty() || group.subgraphs_[0].empty()) {
            return {false, 0};
        }

        VTypeT<GraphT> commonNodeType = instance.GetComputationalDag().VertexType(group.subgraphs_[0][0]);
        const auto& repSubgraph = group.subgraphs_[0];

        for (const auto& vertex : repSubgraph) {
            if (instance.GetComputationalDag().VertexType(vertex) != commonNodeType) {
                return {false, 0};
            }
        }

        return {true, commonNodeType};
    }

    unsigned DetermineEffectiveMinProcCount(const typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group& group,
                                            const BspInstance<GraphT>& instance) const
    {
        unsigned effectiveMinProcTypeCount = 0;

        if (useMaxGroupSize_) {
            effectiveMinProcTypeCount = maxGroupSize_;
        } else {
            auto [isSingleType, commonNodeType] = IsSingleTypeGroup(group, instance);

            if (isSingleType) {
                // Dynamically determine min_proc_type_count based on compatible processors for this type
                unsigned minCompatibleProcessors = std::numeric_limits<unsigned>::max();
                const auto& procTypeCounts = instance.GetArchitecture().GetProcessorTypeCount();

                bool foundCompatibleProcessor = false;
                for (unsigned procTypeIdx = 0; procTypeIdx < procTypeCounts.size(); ++procTypeIdx) {
                    if (instance.IsCompatibleType(commonNodeType, procTypeIdx)) {
                        minCompatibleProcessors = std::min(minCompatibleProcessors, procTypeCounts[procTypeIdx]);
                        foundCompatibleProcessor = true;
                    }
                }
                if (foundCompatibleProcessor) {
                    effectiveMinProcTypeCount = minCompatibleProcessors;
                } else {
                    effectiveMinProcTypeCount = 1;
                }
            } else {
                // Fallback to a default min_proc_type_count if not a single-type group or no typed vertices.
                const auto& typeCount = instance.GetArchitecture().GetProcessorTypeCount();
                if (typeCount.empty()) {
                    effectiveMinProcTypeCount = 0;
                }
                effectiveMinProcTypeCount = *std::min_element(typeCount.begin(), typeCount.end());
            }
        }

        // Ensure effective_min_proc_type_count is at least 1 for valid GCD calculation.
        if (effectiveMinProcTypeCount == 0) {
            effectiveMinProcTypeCount = 1;
        }

        return effectiveMinProcTypeCount;
    }

    void PerformGroupTrimming(typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group& group, unsigned gcd,
                              unsigned groupSize, size_t groupIdx, std::vector<bool>& wasTrimmed)
    {
        if (allowUseTrimmedScheduler_) {
            gcd = 1;
        }

        wasTrimmed[groupIdx] = true;
        const unsigned mergeSize = gcd == 0 ? 1 : groupSize / gcd;
        std::vector<std::vector<VertexIdxT<GraphT>>> newSubgraphs;
        newSubgraphs.reserve(gcd);

        size_t originalSgCursor = 0;

        for (unsigned j = 0; j < gcd; ++j) {
            std::vector<VertexIdxT<GraphT>> mergedSgVertices;
            // Estimate capacity for efficiency. Assuming subgraphs have similar sizes.
            if (!group.subgraphs_.empty()) {
                mergedSgVertices.reserve(group.subgraphs_[0].size() * mergeSize);
            }

            for (unsigned k = 0; k < mergeSize; ++k) {
                const auto& sgToMergeVertices = group.subgraphs_[originalSgCursor];
                originalSgCursor++;
                mergedSgVertices.insert(mergedSgVertices.end(), sgToMergeVertices.begin(), sgToMergeVertices.end());
            }
            newSubgraphs.push_back(std::move(mergedSgVertices));
        }
        group.subgraphs_ = std::move(newSubgraphs);
    }

    /**
     * @brief Trims isomorphic subgraph groups to better fit processor availability.
     *
     * Splits large groups into smaller chunks if their size shares a common divisor with
     * the available processor count (or max group size), effectively increasing the number
     * of schedulable tasks ("trimming").
     *
     * @param isomorphicGroups The groups to potentially trim.
     * @param instance The BSP instance.
     * @param wasTrimmed Output vector indicating which groups were trimmed.
     */
    void TrimSubgraphGroups(std::vector<typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group>& isomorphicGroups,
                            const BspInstance<GraphT>& instance, std::vector<bool>& wasTrimmed)
    {
        for (size_t groupIdx = 0; groupIdx < isomorphicGroups.size(); ++groupIdx) {
            auto& group = isomorphicGroups[groupIdx];
            const unsigned groupSize = static_cast<unsigned>(group.size());
            if (groupSize <= 1) {
                continue;
            }

            unsigned effectiveMinProcTypeCount = DetermineEffectiveMinProcCount(group, instance);
            // If effective_min_proc_type_count is 1, no trimming is needed as gcd(X, 1) = 1.
            if (effectiveMinProcTypeCount <= 1) {
                continue;
            }

            unsigned gcd = std::gcd(groupSize, effectiveMinProcTypeCount);
            if (gcd < groupSize) {
                PerformGroupTrimming(group, gcd, groupSize, groupIdx, wasTrimmed);
            } else {
                wasTrimmed[groupIdx] = false;
            }
        }
    }

    void AccumulateGroupProcTypes(const BspInstance<GraphT>& originalInstance,
                                  const typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group& group,
                                  unsigned numProcTypes, std::vector<VWorkwT<GraphT>>& requiredProcTypes,
                                  std::vector<VertexIdxT<ConstrGraphT>>& contractionMap, size_t coarseNodeIdx)
    {
        for (const auto& subgraph : group.subgraphs_) {
            for (const auto& vertex : subgraph) {
                contractionMap[vertex] = static_cast<VertexIdxT<ConstrGraphT>>(coarseNodeIdx);
                const auto vertexWork = originalInstance.GetComputationalDag().VertexWorkWeight(vertex);
                const auto vertexType = originalInstance.GetComputationalDag().VertexType(vertex);
                for (unsigned j = 0; j < numProcTypes; ++j) {
                    if (originalInstance.IsCompatibleType(vertexType, j)) {
                        requiredProcTypes[j] += vertexWork;
                    }
                }
            }
        }
    }

    SubgraphSchedulerInput<GraphT, ConstrGraphT> PrepareSubgraphSchedulingInput(
        const BspInstance<GraphT>& originalInstance,
        const std::vector<typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group>& isomorphicGroups,
        const std::vector<bool>& wasTrimmed)
    {
        SubgraphSchedulerInput<GraphT, ConstrGraphT> result;
        result.instance_.GetArchitecture() = originalInstance.GetArchitecture();
        const unsigned numProcTypes = originalInstance.GetArchitecture().GetNumberOfProcessorTypes();

        result.multiplicities_.resize(isomorphicGroups.size());
        result.maxNumProcessors_.resize(isomorphicGroups.size());
        result.requiredProcTypes_.resize(isomorphicGroups.size());
        std::vector<VertexIdxT<ConstrGraphT>> contractionMap(originalInstance.NumberOfVertices());

        size_t coarseNodeIdx = 0;
        for (const auto& group : isomorphicGroups) {
            result.maxNumProcessors_[coarseNodeIdx] = static_cast<unsigned>(group.size() * group.subgraphs_[0].size());
            result.multiplicities_[coarseNodeIdx] = (wasTrimmed[coarseNodeIdx] && allowUseTrimmedScheduler_) ?
                                                        1 :
                                                        static_cast<unsigned>(group.subgraphs_.size());
            result.requiredProcTypes_[coarseNodeIdx].assign(numProcTypes, 0);

            AccumulateGroupProcTypes(originalInstance, group, numProcTypes, result.requiredProcTypes_[coarseNodeIdx],
                                     contractionMap, coarseNodeIdx);
            ++coarseNodeIdx;
        }
        coarser_util::ConstructCoarseDag(originalInstance.GetComputationalDag(), result.instance_.GetComputationalDag(),
                                         contractionMap);
        return result;
    }

    std::pair<std::map<std::pair<unsigned, unsigned>, VertexIdxT<GraphT>>, VertexIdxT<GraphT>>
    BuildRelativePartitionMap(const BspSchedule<ConstrGraphT>& bspSchedule, VertexIdxT<GraphT> numRepVertices,
                              bool maxBsp)
    {
        std::map<std::pair<unsigned, unsigned>, VertexIdxT<GraphT>> spProcToRelativePartition;
        VertexIdxT<GraphT> numPartitionsPerSubgraph = 0;
        for (VertexIdxT<GraphT> j = 0; j < numRepVertices; ++j) {
            auto spPair = maxBsp ? std::make_pair(static_cast<unsigned>(j), 0U) :
                                   std::make_pair(bspSchedule.AssignedSuperstep(j), bspSchedule.AssignedProcessor(j));
            if (spProcToRelativePartition.find(spPair) == spProcToRelativePartition.end()) {
                spProcToRelativePartition[spPair] = numPartitionsPerSubgraph++;
            }
        }
        return {std::move(spProcToRelativePartition), numPartitionsPerSubgraph};
    }

    std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<ConstrGraphT>> BuildIsomorphicVertexMapping(
        const BspInstance<GraphT>& instance, const std::vector<VertexIdxT<GraphT>>& subgraphVerticesSorted,
        const MerkleHashComputer<ConstrGraphT>& repHasher)
    {
        ConstrGraphT currentSubgraphGraph;
        auto originalToLocalMap = CreateInducedSubgraphMap(instance.GetComputationalDag(), currentSubgraphGraph,
                                                           subgraphVerticesSorted);

        std::vector<VertexIdxT<GraphT>> localToOriginal(currentSubgraphGraph.NumVertices());
        for (const auto& [orig, local] : originalToLocalMap) {
            localToOriginal[local] = orig;
        }

        MerkleHashComputer<ConstrGraphT> currentHasher(currentSubgraphGraph);
        std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<ConstrGraphT>> mapping;
        for (const auto& [hash, repOrbitNodes] : repHasher.GetOrbits()) {
            const auto& currentOrbitNodes = currentHasher.GetOrbitFromHash(hash);
            for (size_t k = 0; k < repOrbitNodes.size(); ++k) {
                mapping[localToOriginal[currentOrbitNodes[k]]] = static_cast<VertexIdxT<ConstrGraphT>>(
                    repOrbitNodes[k]);
            }
        }
        return mapping;
    }

    void ApplyPartitionPattern(
        const std::vector<VertexIdxT<GraphT>>& subgraphVerticesSorted,
        const std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<ConstrGraphT>>& vertexToRepLocalIdx,
        const BspSchedule<ConstrGraphT>& bspSchedule, bool maxBsp,
        const std::map<std::pair<unsigned, unsigned>, VertexIdxT<GraphT>>& spProcToRelativePartition,
        VertexIdxT<GraphT> partitionOffset, std::vector<VertexIdxT<GraphT>>& partition)
    {
        for (const auto& currentVertex : subgraphVerticesSorted) {
            const auto repLocalIdx = vertexToRepLocalIdx.at(currentVertex);
            auto spPair = maxBsp ? std::make_pair(static_cast<unsigned>(repLocalIdx), 0U) :
                                   std::make_pair(bspSchedule.AssignedSuperstep(repLocalIdx),
                                                  bspSchedule.AssignedProcessor(repLocalIdx));
            partition[currentVertex] = partitionOffset + spProcToRelativePartition.at(spPair);
        }
    }

    void ScheduleIsomorphicGroup(
        const BspInstance<GraphT>& instance,
        const std::vector<typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group>& isomorphicGroups,
        const SubgraphSchedule& subSched, std::vector<VertexIdxT<GraphT>>& partition)
    {
        VertexIdxT<GraphT> currentPartitionIdx = 0;

        for (size_t groupIdx = 0; groupIdx < isomorphicGroups.size(); ++groupIdx) {
            const auto& group = isomorphicGroups[groupIdx];
            if (group.subgraphs_.empty())
                continue;

            auto repSubgraphVertices = group.subgraphs_[0];
            BspInstance<ConstrGraphT> representativeInstance;
            auto repGlobalToLocalMap = CreateInducedSubgraphMap(
                instance.GetComputationalDag(), representativeInstance.GetComputationalDag(), repSubgraphVertices);

            representativeInstance.GetArchitecture() = instance.GetArchitecture();
            const auto& procsForGroup = subSched.nodeAssignedWorkerPerType_[groupIdx];
            std::vector<VMemwT<ConstrGraphT>> memWeights(procsForGroup.size(), 0);
            for (unsigned procType = 0; procType < procsForGroup.size(); ++procType) {
                memWeights[procType] = static_cast<VMemwT<ConstrGraphT>>(
                    instance.GetArchitecture().MaxMemoryBoundProcType(procType));
            }
            representativeInstance.GetArchitecture().SetProcessorsConsequTypes(procsForGroup, memWeights);
            representativeInstance.SetNodeProcessorCompatibility(instance.GetProcessorCompatibilityMatrix());

            unsigned minNonZeroProcs = std::numeric_limits<unsigned>::max();
            for (const auto& procCount : procsForGroup) {
                if (procCount > 0)
                    minNonZeroProcs = std::min(minNonZeroProcs, procCount);
            }

            Scheduler<ConstrGraphT>* schedulerForGroupPtr = bspScheduler_;
            std::unique_ptr<Scheduler<ConstrGraphT>> trimmedSchedulerOwner;
            if (subSched.wasTrimmed_[groupIdx] && minNonZeroProcs > 1 && allowUseTrimmedScheduler_) {
                trimmedSchedulerOwner = std::make_unique<TrimmedGroupScheduler<ConstrGraphT>>(*bspScheduler_,
                                                                                              minNonZeroProcs);
                schedulerForGroupPtr = trimmedSchedulerOwner.get();
            }

            BspSchedule<ConstrGraphT> bspSchedule(representativeInstance);
            schedulerForGroupPtr->ComputeSchedule(bspSchedule);
            const bool maxBsp = useMaxBsp_ && (representativeInstance.GetComputationalDag().NumEdges() == 0) &&
                                (representativeInstance.GetComputationalDag().VertexType(0) == 0);

            auto [spProcToRelativePartition, numPartitionsPerSubgraph] = BuildRelativePartitionMap(
                bspSchedule, static_cast<VertexIdxT<GraphT>>(repSubgraphVertices.size()), maxBsp);

            MerkleHashComputer<ConstrGraphT> repHasher(representativeInstance.GetComputationalDag());

            for (VertexIdxT<GraphT> i = 0; i < static_cast<VertexIdxT<GraphT>>(group.subgraphs_.size()); ++i) {
                auto currentSubgraphVerticesSorted = group.subgraphs_[i];
                std::sort(currentSubgraphVerticesSorted.begin(), currentSubgraphVerticesSorted.end());

                auto currentVertexToRepLocalIdx = (i == 0) ? std::move(repGlobalToLocalMap) :
                                                             BuildIsomorphicVertexMapping(
                                                                 instance, currentSubgraphVerticesSorted, repHasher);

                ApplyPartitionPattern(currentSubgraphVerticesSorted, currentVertexToRepLocalIdx, bspSchedule, maxBsp,
                                      spProcToRelativePartition, currentPartitionIdx, partition);
                currentPartitionIdx += numPartitionsPerSubgraph;
            }
        }
    }

private:
    const HashComputer<VertexIdxT<GraphT>>* hashComputer_;

    Scheduler<ConstrGraphT>* bspScheduler_;
    bool useMaxGroupSize_ = false;
    unsigned maxGroupSize_ = 0;
    VWorkwT<ConstrGraphT> workThreshold_ = 10;
    VWorkwT<ConstrGraphT> criticalPathThreshold_ = 10;
    double orbitLockRatio_ = 0.5;
    double naturalBreaksCountPercentage_ = 0.1;
    bool mergeDifferentNodeTypes_ = false;
    bool allowUseTrimmedScheduler_ = false;
    bool useMaxBsp_ = false;
};
} // namespace osp
} // namespace npu::tile_fwk

#endif // OSP_ISOMORPHIC_SUBGRAPH_SCHEDULER_HPP
