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
 * \file orbit_graph_processor.h
 * \brief
 */

#ifndef OSP_ORBIT_GRAPH_PROCESSOR_H
#define OSP_ORBIT_GRAPH_PROCESSOR_H

#include <algorithm>
#include <map>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/coarser/coarser_util.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/hash_computer.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/merkle_hash_computer.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_path_util.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_util.h"
#include "passes/algorithms/osp/graph_algorithms/subgraph_algorithms.h"

#include "interface/utils/common.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @class OrbitGraphProcessor
 * @brief A simple processor that groups nodes of a DAG based on their Merkle hash.
 *
 * This class uses a MerkleHashComputer to assign a structural hash to each node.
 * It then partitions the DAG by grouping all nodes with the same hash into an "orbit".
 * A coarse graph is constructed where each node represents one such orbit.
 */
template <typename GraphT, typename ConstrGraphT>
class OrbitGraphProcessor {
public:
    using VertexType = VertexIdxT<GraphT>;

    // Represents a group of isomorphic subgraphs, corresponding to a single node in a coarse graph.
    struct Group {
        // Each vector of vertices represents one of the isomorphic subgraphs in this group.
        std::vector<std::vector<VertexType>> subgraphs_;

        inline size_t size() const { return subgraphs_.size(); }
    };

    explicit OrbitGraphProcessor() {}

    void SetMergeDifferentNodeTypes(bool flag) { mergeDifferentNodeTypes_ = flag; }
    void SetWorkThreshold(VWorkwT<ConstrGraphT> workThreshold) { workThreshold_ = workThreshold; }
    void SetCriticalPathThreshold(VWorkwT<ConstrGraphT> criticalPathThreshold)
    {
        criticalPathThreshold_ = criticalPathThreshold;
    }
    void SetLockRatio(double lockRatio) { lockOrbitRatio_ = lockRatio; }
    void SetNaturalBreaksCountPercentage(double percentage) { naturalBreaksCountPercentage_ = percentage; }

    /**
     * @brief Discovers isomorphic groups (orbits) in the DAG and constructs an initial coarse graph.
     *
     * Uses a HashComputer to identify symmetric nodes (orbits) and groups them.
     * Then performs coarsening (either adaptive or static) to merge these groups further.
     *
     * @param dag The input computational DAG.
     * @param hasher The hash computer providing orbit information.
     */
    void DiscoverIsomorphicGroups(const GraphT& dag, const HashComputer<VertexType>& hasher);

    const ConstrGraphT& GetCoarseGraph() const { return coarseGraph_; }

    const std::vector<VertexType>& GetContractionMap() const { return contractionMap_; }

    const ConstrGraphT& GetFinalCoarseGraph() const { return finalCoarseGraph_; }

    const std::vector<VertexType>& GetFinalContractionMap() const { return finalContractionMap_; }

    const std::vector<Group>& GetFinalGroups() const { return finalGroups_; }

private:
    // Results from the first (orbit) coarsening step
    ConstrGraphT coarseGraph_;
    std::vector<VertexType> contractionMap_;

    // Results from the second (custom) coarsening step
    ConstrGraphT finalCoarseGraph_;
    std::vector<VertexType> finalContractionMap_;
    std::vector<Group> finalGroups_;
    size_t currentSymmetry_;

    size_t minSymmetry_ = 2; // min symmetry threshold
    VWorkwT<ConstrGraphT> workThreshold_ = 0;
    VWorkwT<ConstrGraphT> criticalPathThreshold_ = 0;
    bool mergeDifferentNodeTypes_ = true;
    double lockOrbitRatio_ = 0.5;

    double naturalBreaksCountPercentage_ = 0.2;

    struct PairHasher {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2>& p) const
        {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            HashCombine(h1, h2);
            return h1;
        }
    };

    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nonViableEdgesCache_;
    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nonViableCritPathEdgesCache_;

    std::pair<ConstrGraphT, std::vector<VertexType>> SimulateMerge(VertexType u, VertexType v,
                                                                   const ConstrGraphT& currentCoarseGraph) const;

    void CommitMerge(VertexType u, VertexType v, ConstrGraphT&& nextCoarseGraph,
                     const std::vector<VertexType>& groupRemap, std::vector<std::vector<VertexType>>&& newSubgraphs,
                     ConstrGraphT& currentCoarseGraph, std::vector<Group>& currentGroups);

    bool ShouldSkipEdge(VertexType u, VertexType v, const ConstrGraphT& currentCoarseGraph,
                        const std::vector<Group>& currentGroups,
                        const std::vector<VertexIdxT<ConstrGraphT>>& vertexPoset,
                        const std::vector<VertexIdxT<ConstrGraphT>>& vertexBotPoset,
                        const VWorkwT<ConstrGraphT> workThreshold) const;

    bool TryMergeEdge(VertexType u, VertexType v, const GraphT& originalDag, const ConstrGraphT& currentCoarseGraph,
                      const std::vector<Group>& currentGroups, const VWorkwT<ConstrGraphT> pathThreshold,
                      std::vector<std::vector<VertexType>>& outNewSubgraphs, ConstrGraphT& outTempGraph,
                      std::vector<VertexType>& outTempContractionMap);

    void MergeSmallOrbits(const GraphT& originalDag, ConstrGraphT& currentCoarseGraph,
                          std::vector<Group>& currentGroups, const VWorkwT<ConstrGraphT> workThreshold,
                          const VWorkwT<ConstrGraphT> pathThreshold = 0);

    bool IsEdgeMergeCandidate(VertexType u, VertexType v, const ConstrGraphT& currentCoarseGraph,
                              const std::vector<VertexIdxT<ConstrGraphT>>& vertexPoset,
                              const std::vector<VertexIdxT<ConstrGraphT>>& vertexBotPoset,
                              const bool mergeDifferentNodeTypes);

    bool IsSignificanceMergeBlocked(VertexType u, VertexType v, const ConstrGraphT& currentCoarseGraph,
                                    const std::vector<Group>& currentGroups,
                                    const std::vector<VWorkwT<GraphT>>& lockThresholdPerType,
                                    const bool mergeDifferentNodeTypes, std::size_t newSize);

    bool TryContractEdge(VertexType u, VertexType v, const GraphT& originalDag, ConstrGraphT& currentCoarseGraph,
                         std::vector<Group>& currentGroups, const bool mergeBelowThreshold,
                         const std::vector<VWorkwT<GraphT>>& lockThresholdPerType, const bool mergeDifferentNodeTypes,
                         const VWorkwT<ConstrGraphT> pathThreshold);

    void ContractEdgesAdpativeSym(const GraphT& originalDag, ConstrGraphT& currentCoarseGraph,
                                  std::vector<Group>& currentGroups, const bool mergeDifferentNodeTypes,
                                  const bool mergeBelowThreshold,
                                  const std::vector<VWorkwT<GraphT>>& lockThresholdPerType,
                                  const VWorkwT<ConstrGraphT> pathThreshold = 0);

    std::vector<size_t> FindSignificantSymmetryLevels(const std::map<size_t, size_t>& orbitSizeCounts,
                                                      size_t countThreshold);

    size_t FindFallbackSymmetryLevel(const std::map<size_t, size_t>& orbitSizeCounts);

    std::vector<size_t> ComputeSymmetryLevels(const std::map<size_t, size_t> orbitSizeCounts);

    void PerformCoarseningAdaptiveSymmetry(const GraphT& originalDag, const ConstrGraphT& initialCoarseGraph,
                                           const std::vector<VWorkwT<GraphT>>& lockThresholdPerType,
                                           const std::vector<size_t>& symmetryLevelsToTest);

    bool IsMergeViable(const GraphT& originalDag, const Group& groupU, const Group& groupV,
                       std::vector<std::vector<VertexType>>& outNewSubgraphs) const;
};
} // namespace osp
} // namespace npu::tile_fwk

#include "passes/algorithms/osp/dag_divider/isomorphism_divider/orbit_graph_processor.tpp"

#endif // OSP_ORBIT_GRAPH_PROCESSOR_H
