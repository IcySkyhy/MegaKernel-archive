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
 * \file orbit_graph_processor.tpp
 * \brief Out-of-class method definitions for OrbitGraphProcessor.
 */

namespace npu::tile_fwk {
namespace osp {
template <typename GraphT, typename ConstrGraphT>
void OrbitGraphProcessor<GraphT, ConstrGraphT>::DiscoverIsomorphicGroups(
    const GraphT &dag, const HashComputer<VertexType> &hasher)
{
    coarseGraph_ = ConstrGraphT();
    contractionMap_.clear();
    finalCoarseGraph_ = ConstrGraphT();
    finalContractionMap_.clear();
    finalGroups_.clear();
    nonViableEdgesCache_.clear();
    nonViableCritPathEdgesCache_.clear();

    if (dag.NumVertices() == 0) return;
    const auto &orbits = hasher.GetOrbits();

    contractionMap_.assign(dag.NumVertices(), 0);
    VertexType coarseNodeIdx = 0;

    for (const auto &hashVerticesPair : orbits) {
        const auto &vertices = hashVerticesPair.second;
        for (const auto v : vertices) {
            contractionMap_[v] = coarseNodeIdx;
        }
        coarseNodeIdx++;
    }

    std::vector<VWorkwT<GraphT>> workPerVertexType;
    workPerVertexType.resize(mergeDifferentNodeTypes_ ? 1U : dag.NumVertexTypes(), 0);

    std::map<size_t, size_t> orbitSizeCounts;
    for (const auto &orbit : orbits) {
        const auto &vertices = orbit.second;
        const size_t orbitSize = vertices.size();
        if (orbitSize == 1U) continue;

        orbitSizeCounts[orbitSize]++;

        VWorkwT<GraphT> orbitWork = 0;
        for (const auto v : vertices) {
            orbitWork += dag.VertexWorkWeight(v);
        }

        if (not mergeDifferentNodeTypes_) {
            workPerVertexType[dag.VertexType(vertices[0])] += orbitWork;
        } else {
            workPerVertexType[0] += orbitWork;
        }
    }

    std::vector<VWorkwT<GraphT>> lockThresholdPerType(workPerVertexType.size());
    for (size_t i = 0; i < workPerVertexType.size(); ++i) {
        lockThresholdPerType[i]
            = static_cast<VWorkwT<GraphT>>(lockOrbitRatio_ * workPerVertexType[i]);
    }

    std::vector<size_t> symmetryLevelsToTest = ComputeSymmetryLevels(orbitSizeCounts);
    coarser_util::ConstructCoarseDag(dag, coarseGraph_, contractionMap_);
    PerformCoarseningAdaptiveSymmetry(
        dag, coarseGraph_, lockThresholdPerType, symmetryLevelsToTest);
}

template <typename GraphT, typename ConstrGraphT>
std::pair<ConstrGraphT, std::vector<typename OrbitGraphProcessor<GraphT, ConstrGraphT>::VertexType>>
OrbitGraphProcessor<GraphT, ConstrGraphT>::SimulateMerge(
    VertexType u, VertexType v,
    const ConstrGraphT &currentCoarseGraph) const
{
    std::vector<VertexType> tempContractionMap(currentCoarseGraph.NumVertices());
    VertexType newIdx = 0;
    for (VertexType i = 0; i < static_cast<VertexType>(tempContractionMap.size()); ++i) {
        if (i != v) {
            tempContractionMap[i] = newIdx++;
        }
    }
    tempContractionMap[v] = tempContractionMap[u];

    ConstrGraphT tempCoarseGraph;
    coarser_util::ConstructCoarseDag(currentCoarseGraph, tempCoarseGraph, tempContractionMap);

    return {std::move(tempCoarseGraph), std::move(tempContractionMap)};
}

template <typename GraphT, typename ConstrGraphT>
void OrbitGraphProcessor<GraphT, ConstrGraphT>::CommitMerge(
    VertexType u, VertexType v, ConstrGraphT &&nextCoarseGraph,
    const std::vector<VertexType> &groupRemap,
    std::vector<std::vector<VertexType>> &&newSubgraphs,
    ConstrGraphT &currentCoarseGraph,
    std::vector<Group> &currentGroups)
{
    currentCoarseGraph = std::move(nextCoarseGraph);

    // Update caches for new vertex indices
    auto UpdateCache = [&](auto &cache)
    {
        std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nextCache;
        for (const auto &[oldU, oldV] : cache) {
            const VertexType newU = groupRemap[oldU];
            const VertexType newV = groupRemap[oldV];
            if (oldU != v && oldV != v && newU != newV) {
                nextCache.insert({newU, newV});
            }
        }
        cache = std::move(nextCache);
    };
    UpdateCache(nonViableEdgesCache_);
    UpdateCache(nonViableCritPathEdgesCache_);

    // Update groups
    std::vector<Group> nextGroups(currentCoarseGraph.NumVertices());
    for (VertexType i = 0; i < static_cast<VertexType>(currentGroups.size()); ++i) {
        if (i != u && i != v) {
            nextGroups[groupRemap[i]] = std::move(currentGroups[i]);
        }
    }
    nextGroups[groupRemap[u]].subgraphs_ = std::move(newSubgraphs);
    currentGroups = std::move(nextGroups);
}

template <typename GraphT, typename ConstrGraphT>
bool OrbitGraphProcessor<GraphT, ConstrGraphT>::ShouldSkipEdge(
    VertexType u, VertexType v,
    const ConstrGraphT &currentCoarseGraph,
    const std::vector<Group> &currentGroups,
    const std::vector<VertexIdxT<ConstrGraphT>> &vertexPoset,
    const std::vector<VertexIdxT<ConstrGraphT>> &vertexBotPoset,
    const VWorkwT<ConstrGraphT> workThreshold) const
{
    // Check node type compatibility
    if (not mergeDifferentNodeTypes_) {
        if (currentCoarseGraph.VertexType(u) != currentCoarseGraph.VertexType(v)) {
            return true;
        }
    }

    // Check if edge is in non-viable cache
    if (nonViableEdgesCache_.count({u, v})
        || nonViableCritPathEdgesCache_.count({u, v}))
    {
        return true;
    }

    // Check work thresholds
    const VWorkwT<ConstrGraphT> uWorkWeight = currentCoarseGraph.VertexWorkWeight(u);
    const VWorkwT<ConstrGraphT> vWorkWeight = currentCoarseGraph.VertexWorkWeight(v);
    const VWorkwT<ConstrGraphT> vThreshold
        = workThreshold * static_cast<VWorkwT<ConstrGraphT>>(currentGroups[v].size());
    const VWorkwT<ConstrGraphT> uThreshold
        = workThreshold * static_cast<VWorkwT<ConstrGraphT>>(currentGroups[u].size());

    if (uWorkWeight > uThreshold && vWorkWeight > vThreshold) {
        return true;
    }

    // Check poset constraints
    if ((vertexPoset[u] + 1 != vertexPoset[v])
        && (vertexBotPoset[u] != 1 + vertexBotPoset[v]))
    {
        return true;
    }

    return false;
}

template <typename GraphT, typename ConstrGraphT>
bool OrbitGraphProcessor<GraphT, ConstrGraphT>::TryMergeEdge(
    VertexType u, VertexType v, const GraphT &originalDag,
    const ConstrGraphT &currentCoarseGraph,
    const std::vector<Group> &currentGroups,
    const VWorkwT<ConstrGraphT> pathThreshold,
    std::vector<std::vector<VertexType>> &outNewSubgraphs,
    ConstrGraphT &outTempGraph,
    std::vector<VertexType> &outTempContractionMap)
{
    // Check merge structural viability
    const bool mergeIsValid
        = IsMergeViable(originalDag, currentGroups[u], currentGroups[v], outNewSubgraphs);
    if (!mergeIsValid) {
        nonViableEdgesCache_.insert({u, v});
        return false;
    }

    // Simulate merge and check critical path
    auto [tempCoarseGraph, tempContractionMap]
        = SimulateMerge(u, v, currentCoarseGraph);

    if (CriticalPathWeight(tempCoarseGraph)
        > (pathThreshold
               * static_cast<VWorkwT<ConstrGraphT>>(outNewSubgraphs.size())
           + CriticalPathWeight(currentCoarseGraph)))
    {
        nonViableCritPathEdgesCache_.insert({u, v});
        return false;
    }

    outTempGraph = std::move(tempCoarseGraph);
    outTempContractionMap = std::move(tempContractionMap);
    return true;
}

template <typename GraphT, typename ConstrGraphT>
void OrbitGraphProcessor<GraphT, ConstrGraphT>::MergeSmallOrbits(
    const GraphT &originalDag,
    ConstrGraphT &currentCoarseGraph,
    std::vector<Group> &currentGroups,
    const VWorkwT<ConstrGraphT> workThreshold,
    const VWorkwT<ConstrGraphT> pathThreshold)
{
    bool changed = true;
    while (changed) {
        const std::vector<VertexIdxT<ConstrGraphT>> vertexPoset
            = GetTopNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(
                currentCoarseGraph);
        const std::vector<VertexIdxT<ConstrGraphT>> vertexBotPoset
            = GetBottomNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(
                currentCoarseGraph);

        changed = false;
        for (const auto u : currentCoarseGraph.Vertices()) {
            for (const auto v : currentCoarseGraph.Children(u)) {
                if (ShouldSkipEdge(u, v, currentCoarseGraph, currentGroups,
                                   vertexPoset, vertexBotPoset, workThreshold))
                {
                    continue;
                }

                std::vector<std::vector<VertexType>> newSubgraphs;
                ConstrGraphT tempCoarseGraph;
                std::vector<VertexType> tempContractionMap;

                if (!TryMergeEdge(u, v, originalDag, currentCoarseGraph,
                                  currentGroups, pathThreshold,
                                  newSubgraphs, tempCoarseGraph,
                                  tempContractionMap))
                {
                    continue;
                }

                CommitMerge(u, v, std::move(tempCoarseGraph),
                            tempContractionMap, std::move(newSubgraphs),
                            currentCoarseGraph, currentGroups);

                changed = true;
                break;
            }
            if (changed) break;
        }
    }
}

template <typename GraphT, typename ConstrGraphT>
bool OrbitGraphProcessor<GraphT, ConstrGraphT>::IsEdgeMergeCandidate(
    VertexType u, VertexType v,
    const ConstrGraphT &currentCoarseGraph,
    const std::vector<VertexIdxT<ConstrGraphT>> &vertexPoset,
    const std::vector<VertexIdxT<ConstrGraphT>> &vertexBotPoset,
    const bool mergeDifferentNodeTypes)
{
    if (nonViableEdgesCache_.count({u, v})
        || nonViableCritPathEdgesCache_.count({u, v}))
    {
        return false;
    }
    if (not mergeDifferentNodeTypes
        && currentCoarseGraph.VertexType(u) != currentCoarseGraph.VertexType(v))
    {
        return false;
    }
    if ((vertexPoset[u] + 1 != vertexPoset[v])
        && (vertexBotPoset[u] != 1 + vertexBotPoset[v]))
    {
        return false;
    }
    return true;
}

template <typename GraphT, typename ConstrGraphT>
bool OrbitGraphProcessor<GraphT, ConstrGraphT>::IsSignificanceMergeBlocked(
    VertexType u, VertexType v,
    const ConstrGraphT &currentCoarseGraph,
    const std::vector<Group> &currentGroups,
    const std::vector<VWorkwT<GraphT>> &lockThresholdPerType,
    const bool mergeDifferentNodeTypes,
    std::size_t newSize)
{
    VTypeT<GraphT> uType = 0;
    VTypeT<GraphT> vType = 0;
    if (not mergeDifferentNodeTypes) {
        uType = currentCoarseGraph.VertexType(u);
        vType = currentCoarseGraph.VertexType(v);
    }

    const std::size_t uSize = currentGroups[u].size();
    const std::size_t vSize = currentGroups[v].size();
    const bool uSig = (uSize >= minSymmetry_)
        && (currentCoarseGraph.VertexWorkWeight(u) > lockThresholdPerType[uType]);
    const bool vSig = (vSize >= minSymmetry_)
        && (currentCoarseGraph.VertexWorkWeight(v) > lockThresholdPerType[vType]);

    return (uSig && vSig && newSize < std::min(uSize, vSize))
        || ((uSig ^ vSig) && newSize < (uSig ? uSize : vSize));
}

template <typename GraphT, typename ConstrGraphT>
bool OrbitGraphProcessor<GraphT, ConstrGraphT>::TryContractEdge(
    VertexType u, VertexType v,
    const GraphT &originalDag,
    ConstrGraphT &currentCoarseGraph,
    std::vector<Group> &currentGroups,
    const bool mergeBelowThreshold,
    const std::vector<VWorkwT<GraphT>> &lockThresholdPerType,
    const bool mergeDifferentNodeTypes,
    const VWorkwT<ConstrGraphT> pathThreshold)
{
    std::vector<std::vector<VertexType>> newSubgraphs;
    if (!IsMergeViable(originalDag, currentGroups[u], currentGroups[v], newSubgraphs)) {
        nonViableEdgesCache_.insert({u, v});
        return false;
    }

    const std::size_t newSize = newSubgraphs.size();
    const bool mergeViable = (newSize >= currentSymmetry_);
    const bool bothBelowMinimalThreshold = mergeBelowThreshold
        && (currentGroups[u].size() < minSymmetry_)
        && (currentGroups[v].size() < minSymmetry_);

    if (!mergeViable && !bothBelowMinimalThreshold) {
        nonViableEdgesCache_.insert({u, v});
        return false;
    }

    if (IsSignificanceMergeBlocked(u, v, currentCoarseGraph, currentGroups,
                                   lockThresholdPerType, mergeDifferentNodeTypes,
                                   newSize))
    {
        nonViableEdgesCache_.insert({u, v});
        return false;
    }

    auto [tempCoarseGraph, tempContractionMap]
        = SimulateMerge(u, v, currentCoarseGraph);
    if (CriticalPathWeight(tempCoarseGraph)
        > (pathThreshold
               * static_cast<VWorkwT<ConstrGraphT>>(newSubgraphs.size())
           + CriticalPathWeight(currentCoarseGraph)))
    {
        nonViableCritPathEdgesCache_.insert({u, v});
        return false;
    }

    CommitMerge(u, v, std::move(tempCoarseGraph), tempContractionMap,
                std::move(newSubgraphs), currentCoarseGraph, currentGroups);
    return true;
}

template <typename GraphT, typename ConstrGraphT>
void OrbitGraphProcessor<GraphT, ConstrGraphT>::ContractEdgesAdpativeSym(
    const GraphT &originalDag,
    ConstrGraphT &currentCoarseGraph,
    std::vector<Group> &currentGroups,
    const bool mergeDifferentNodeTypes,
    const bool mergeBelowThreshold,
    const std::vector<VWorkwT<GraphT>> &lockThresholdPerType,
    const VWorkwT<ConstrGraphT> pathThreshold)
{
    bool changed = true;
    while (changed) {
        const auto vertexPoset
            = GetTopNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(
                currentCoarseGraph);
        const auto vertexBotPoset
            = GetBottomNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(
                currentCoarseGraph);

        changed = false;
        for (const auto &edge : Edges(currentCoarseGraph)) {
            VertexType u = Source(edge, currentCoarseGraph);
            VertexType v = Target(edge, currentCoarseGraph);
            if (!IsEdgeMergeCandidate(u, v, currentCoarseGraph,
                                      vertexPoset, vertexBotPoset,
                                      mergeDifferentNodeTypes))
            {
                continue;
            }

            if (TryContractEdge(u, v, originalDag, currentCoarseGraph,
                                currentGroups, mergeBelowThreshold,
                                lockThresholdPerType,
                                mergeDifferentNodeTypes, pathThreshold))
            {
                changed = true;
                break;
            }
        }
    }
}

template <typename GraphT, typename ConstrGraphT>
std::vector<size_t>
OrbitGraphProcessor<GraphT, ConstrGraphT>::FindSignificantSymmetryLevels(
    const std::map<size_t, size_t> &orbitSizeCounts,
    size_t countThreshold)
{
    std::vector<size_t> sortedSizes;
    sortedSizes.reserve(orbitSizeCounts.size());
    for (const auto &pair: orbitSizeCounts) {
        sortedSizes.push_back(pair.first);
    }
    std::sort(sortedSizes.rbegin(), sortedSizes.rend());

    std::vector<size_t> levels;
    for (const size_t currentSize : sortedSizes) {
        if (currentSize >= minSymmetry_
            && orbitSizeCounts.at(currentSize) >= countThreshold)
        {
            levels.push_back(currentSize);
        }
    }
    return levels;
}

template <typename GraphT, typename ConstrGraphT>
size_t OrbitGraphProcessor<GraphT, ConstrGraphT>::FindFallbackSymmetryLevel(
    const std::map<size_t, size_t> &orbitSizeCounts)
{
    size_t maxCount = 0;
    size_t sizeWithMaxCount = 0;
    for (const auto &[size, count] : orbitSizeCounts) {
        if (count > maxCount) {
            maxCount = count;
            sizeWithMaxCount = size;
        }
    }
    return sizeWithMaxCount;
}

template <typename GraphT, typename ConstrGraphT>
std::vector<size_t>
OrbitGraphProcessor<GraphT, ConstrGraphT>::ComputeSymmetryLevels(
    const std::map<size_t, size_t> orbitSizeCounts)
{
    minSymmetry_ = 2;

    size_t totalOrbitGroups = 0;
    for (const auto &pair: orbitSizeCounts) {
        totalOrbitGroups += pair.second;
    }
    size_t countThreshold = static_cast<size_t>(
        static_cast<double>(totalOrbitGroups) * naturalBreaksCountPercentage_);
    if (countThreshold == 0 && totalOrbitGroups > 0) {
        countThreshold = 1;
    }

    std::vector<size_t> symmetryLevelsToTest
        = FindSignificantSymmetryLevels(orbitSizeCounts, countThreshold);

    if (symmetryLevelsToTest.empty()) {
        const size_t fallback = FindFallbackSymmetryLevel(orbitSizeCounts);
        if (fallback > 0) {
            symmetryLevelsToTest.push_back(fallback);
        }
    }
    if (symmetryLevelsToTest.empty()) {
        symmetryLevelsToTest.push_back(2);
    }

    minSymmetry_ = symmetryLevelsToTest.back();

    std::sort(symmetryLevelsToTest.rbegin(), symmetryLevelsToTest.rend());
    auto last = std::unique(
        symmetryLevelsToTest.begin(), symmetryLevelsToTest.end());
    symmetryLevelsToTest.erase(last, symmetryLevelsToTest.end());

    return symmetryLevelsToTest;
}

template <typename GraphT, typename ConstrGraphT>
void OrbitGraphProcessor<GraphT, ConstrGraphT>::PerformCoarseningAdaptiveSymmetry(
    const GraphT &originalDag,
    const ConstrGraphT &initialCoarseGraph,
    const std::vector<VWorkwT<GraphT>> &lockThresholdPerType,
    const std::vector<size_t> &symmetryLevelsToTest)
{
    finalCoarseGraph_ = ConstrGraphT();
    finalContractionMap_.clear();

    if (initialCoarseGraph.NumVertices() == 0) return;

    ConstrGraphT currentCoarseGraph = initialCoarseGraph;
    std::vector<Group> currentGroups(initialCoarseGraph.NumVertices());
    std::vector<VertexType> currentContractionMap = contractionMap_;

    for (VertexType i = 0; i < originalDag.NumVertices(); ++i) {
        const VertexType coarseNode = contractionMap_[i];
        currentGroups[coarseNode].subgraphs_.push_back({i});
    }

    for (const auto sym : symmetryLevelsToTest) {
        currentSymmetry_ = sym;
        const bool isLastLoop = (sym == symmetryLevelsToTest.back());

        nonViableEdgesCache_.clear();
        ContractEdgesAdpativeSym(originalDag, currentCoarseGraph,
                                currentGroups, false, isLastLoop,
                                lockThresholdPerType);

        if (mergeDifferentNodeTypes_) {
            ContractEdgesAdpativeSym(originalDag, currentCoarseGraph,
                                    currentGroups, mergeDifferentNodeTypes_,
                                    isLastLoop, lockThresholdPerType);
        }

        nonViableCritPathEdgesCache_.clear();
        ContractEdgesAdpativeSym(originalDag, currentCoarseGraph,
                                currentGroups, mergeDifferentNodeTypes_,
                                isLastLoop, lockThresholdPerType,
                                criticalPathThreshold_);
    }

    nonViableEdgesCache_.clear();
    MergeSmallOrbits(originalDag, currentCoarseGraph, currentGroups,
                     workThreshold_);

    // Rebuild contraction map from currentGroups
    currentContractionMap.assign(originalDag.NumVertices(), 0);
    for (VertexType coarseIdx = 0;
         coarseIdx < static_cast<VertexType>(currentGroups.size());
         ++coarseIdx)
    {
        for (const auto &subgraph : currentGroups[coarseIdx].subgraphs_) {
            for (const auto v : subgraph) {
                currentContractionMap[v] = coarseIdx;
            }
        }
    }

    finalCoarseGraph_ = std::move(currentCoarseGraph);
    finalContractionMap_ = std::move(currentContractionMap);
    finalGroups_ = std::move(currentGroups);
}

template <typename GraphT, typename ConstrGraphT>
bool OrbitGraphProcessor<GraphT, ConstrGraphT>::IsMergeViable(
    const GraphT &originalDag,
    const Group &groupU,
    const Group &groupV,
    std::vector<std::vector<VertexType>> &outNewSubgraphs) const
{
    std::vector<VertexType> allNodes;
    const size_t uNodes
        = groupU.subgraphs_.empty()
              ? 0
              : groupU.subgraphs_.size() * groupU.subgraphs_[0].size();
    const size_t vNodes
        = groupV.subgraphs_.empty()
              ? 0
              : groupV.subgraphs_.size() * groupV.subgraphs_[0].size();
    allNodes.reserve(uNodes + vNodes);
    for (const auto &sg : groupU.subgraphs_) {
        allNodes.insert(allNodes.end(), sg.begin(), sg.end());
    }
    for (const auto &sg : groupV.subgraphs_) {
        allNodes.insert(allNodes.end(), sg.begin(), sg.end());
    }

    std::sort(allNodes.begin(), allNodes.end());
    ConstrGraphT inducedSubgraph;

    auto map = CreateInducedSubgraphMap(originalDag, inducedSubgraph, allNodes);
    std::vector<VertexType> components;    // local -> component_id
    size_t numComponents
        = ComputeWeaklyConnectedComponents(inducedSubgraph, components);
    outNewSubgraphs.assign(numComponents, std::vector<VertexType>());
    if (allNodes.empty()) return true;

    for (const auto &node : allNodes) {
        outNewSubgraphs[components[map[node]]].push_back(node);
    }

    if (numComponents > 1) {
        const size_t firstSgSize = outNewSubgraphs[0].size();
        ConstrGraphT repSg;
        CreateInducedSubgraphMap(originalDag, repSg, outNewSubgraphs[0]);

        for (size_t i = 1; i < numComponents; ++i) {
            if (outNewSubgraphs[i].size() != firstSgSize) return false;

            ConstrGraphT currentSg;
            CreateInducedSubgraphMap(
                originalDag, currentSg, outNewSubgraphs[i]);
            if (!AreIsomorphicByMerkleHash(repSg, currentSg)) return false;
        }
    }
    return true;
}
}    // namespace osp
} // namespace npu::tile_fwk
