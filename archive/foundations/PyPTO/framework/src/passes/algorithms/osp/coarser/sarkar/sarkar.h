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
 * \file sarkar.h
 * \brief
 */

#ifndef PASS_OSP_SARKAR_H
#define PASS_OSP_SARKAR_H

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>
#include <type_traits>

#include "passes/algorithms/osp/auxiliary/datastructures/union_find_universe.h"
#include "passes/algorithms/osp/coarser/coarser.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_path_util.h"

#include "interface/utils/common.h"

namespace npu::tile_fwk {
namespace osp {
namespace sarkar_params {
enum class Mode {
    LINES,
    FAN_IN_FULL,
    FAN_IN_PARTIAL,
    FAN_OUT_FULL,
    FAN_OUT_PARTIAL,
    LEVEL_EVEN,
    LEVEL_ODD,
    FAN_IN_BUFFER,
    FAN_OUT_BUFFER,
    HOMOGENEOUS_BUFFER
};

template <typename CommCostType>
struct Parameters {
    double geomDecay_{0.875};
    double leniency_{0.0};
    Mode mode_{Mode::LINES};
    CommCostType commCost_{static_cast<CommCostType>(0)};
    CommCostType maxWeight_{std::numeric_limits<CommCostType>::max()};
    CommCostType smallWeightThreshold_{std::numeric_limits<CommCostType>::lowest()};
    bool useTopPoset_{true};
};
} // namespace sarkar_params

template <typename GraphTIn, typename GraphTOut>
class Sarkar : public CoarserGenExpansionMap<GraphTIn, GraphTOut> {
public:
    virtual std::vector<std::vector<VertexIdxT<GraphTIn>>> GenerateVertexExpansionMap(const GraphTIn& dagIn) override;
    std::vector<std::vector<VertexIdxT<GraphTIn>>> GenerateVertexExpansionMap(const GraphTIn& dagIn,
                                                                              VertexIdxT<GraphTIn>& diff);

    inline void SetParameters(const sarkar_params::Parameters<VWorkwT<GraphTIn>>& params) { params_ = params; }

    inline sarkar_params::Parameters<VWorkwT<GraphTIn>>& GetParameters() { return params_; }

    inline const sarkar_params::Parameters<VWorkwT<GraphTIn>>& GetParameters() const { return params_; }

    Sarkar(sarkar_params::Parameters<VWorkwT<GraphTIn>> params = sarkar_params::Parameters<VWorkwT<GraphTIn>>())
        : params_(params) {};

    Sarkar(const Sarkar&) = default;
    Sarkar(Sarkar&&) = default;
    Sarkar& operator=(const Sarkar&) = default;
    Sarkar& operator=(Sarkar&&) = default;
    virtual ~Sarkar() override = default;

private:
    sarkar_params::Parameters<VWorkwT<GraphTIn>> params_;

    std::vector<VertexIdxT<GraphTIn>> GetBotPosetMap(const GraphTIn& graph) const;
    std::vector<VWorkwT<GraphTIn>> GetTopDistance(VWorkwT<GraphTIn> commCost, const GraphTIn& graph) const;
    std::vector<VWorkwT<GraphTIn>> GetBotDistance(VWorkwT<GraphTIn> commCost, const GraphTIn& graph) const;

    VertexIdxT<GraphTIn> SingleContraction(VWorkwT<GraphTIn> commCost, const GraphTIn& graph,
                                           std::vector<std::vector<VertexIdxT<GraphTIn>>>& expansionMapOutput) const;
    VertexIdxT<GraphTIn> AllChildrenContraction(
        VWorkwT<GraphTIn> commCost, const GraphTIn& graph,
        std::vector<std::vector<VertexIdxT<GraphTIn>>>& expansionMapOutput) const;
    VertexIdxT<GraphTIn> SomeChildrenContraction(
        VWorkwT<GraphTIn> commCost, const GraphTIn& graph,
        std::vector<std::vector<VertexIdxT<GraphTIn>>>& expansionMapOutput) const;
    VertexIdxT<GraphTIn> AllParentsContraction(
        VWorkwT<GraphTIn> commCost, const GraphTIn& graph,
        std::vector<std::vector<VertexIdxT<GraphTIn>>>& expansionMapOutput) const;
    VertexIdxT<GraphTIn> SomeParentsContraction(
        VWorkwT<GraphTIn> commCost, const GraphTIn& graph,
        std::vector<std::vector<VertexIdxT<GraphTIn>>>& expansionMapOutput) const;
    VertexIdxT<GraphTIn> LevelContraction(VWorkwT<GraphTIn> commCost, const GraphTIn& graph,
                                          std::vector<std::vector<VertexIdxT<GraphTIn>>>& expansionMapOutput) const;

    VertexIdxT<GraphTIn> HomogeneousBufferMerge(
        VWorkwT<GraphTIn> commCost, const GraphTIn& graph,
        std::vector<std::vector<VertexIdxT<GraphTIn>>>& expansionMapOutput) const;
    std::vector<std::size_t> HomogeneousMerge(const std::size_t number, const std::size_t minSize,
                                              const std::size_t maxSize) const;

    std::vector<std::size_t> ComputeNodeHashes(const GraphTIn& graph,
                                               const std::vector<VertexIdxT<GraphTIn>>& vertexPoset,
                                               const std::vector<VWorkwT<GraphTIn>>& dist) const;
};

template <typename GraphTIn, typename GraphTOut>
std::vector<VertexIdxT<GraphTIn>> Sarkar<GraphTIn, GraphTOut>::GetBotPosetMap(const GraphTIn& graph) const
{
    std::vector<VertexIdxT<GraphTIn>> botPosetMap = GetBottomNodeDistance<GraphTIn, VertexIdxT<GraphTIn>>(graph);

    const VertexIdxT<GraphTIn> max = botPosetMap.size() == 0U ?
                                         0 :
                                         *std::max_element(botPosetMap.begin(), botPosetMap.end()) + 1;

    for (std::size_t i = 0; i < botPosetMap.size(); i++) {
        botPosetMap[i] = max - botPosetMap[i];
    }

    return botPosetMap;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<VWorkwT<GraphTIn>> Sarkar<GraphTIn, GraphTOut>::GetTopDistance(VWorkwT<GraphTIn> commCost,
                                                                           const GraphTIn& graph) const
{
    std::vector<VWorkwT<GraphTIn>> topDist(graph.NumVertices(), 0);

    for (const auto& vertex : GetTopOrder<GraphTIn>(graph)) {
        VWorkwT<GraphTIn> maxTemp = 0;

        for (const auto& j : graph.Parents(vertex)) {
            maxTemp = std::max(maxTemp, topDist[j]);
        }
        if (graph.InDegree(vertex) > 0) {
            maxTemp += commCost;
        }

        topDist[vertex] = maxTemp + graph.VertexWorkWeight(vertex);
    }

    return topDist;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<VWorkwT<GraphTIn>> Sarkar<GraphTIn, GraphTOut>::GetBotDistance(VWorkwT<GraphTIn> commCost,
                                                                           const GraphTIn& graph) const
{
    std::vector<VWorkwT<GraphTIn>> botDist(graph.NumVertices(), 0);

    const auto topOrder = GetTopOrder<GraphTIn>(graph);
    for (auto revTopIt = topOrder.crbegin(); revTopIt != topOrder.crend(); ++revTopIt) {
        const auto& vertex = *revTopIt;

        VWorkwT<GraphTIn> maxTemp = 0;

        for (const auto& j : graph.Children(vertex)) {
            maxTemp = std::max(maxTemp, botDist[j]);
        }
        if (graph.OutDegree(vertex) > 0) {
            maxTemp += commCost;
        }

        botDist[vertex] = maxTemp + graph.VertexWorkWeight(vertex);
    }

    return botDist;
}

template <typename GraphTIn, typename GraphTOut>
VertexIdxT<GraphTIn> Sarkar<GraphTIn, GraphTOut>::SingleContraction(
    VWorkwT<GraphTIn> commCost, const GraphTIn& graph,
    std::vector<std::vector<VertexIdxT<GraphTIn>>>& expansionMapOutput) const
{
    using VertexType = VertexIdxT<GraphTIn>;
    expansionMapOutput.clear();

    const std::vector<VertexIdxT<GraphTIn>> vertexPoset = params_.useTopPoset_ ?
                                                              GetTopNodeDistance<GraphTIn, VertexIdxT<GraphTIn>>(
                                                                  graph) :
                                                              GetBotPosetMap(graph);
    const std::vector<VWorkwT<GraphTIn>> topDist = GetTopDistance(commCost, graph);
    const std::vector<VWorkwT<GraphTIn>> botDist = GetBotDistance(commCost, graph);

    auto cmp = [](const std::tuple<long, VertexType, VertexType>& lhs,
                  const std::tuple<long, VertexType, VertexType>& rhs) {
        return (std::get<0>(lhs) > std::get<0>(rhs)) ||
               ((std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) < std::get<1>(rhs))) ||
               ((std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) == std::get<1>(rhs)) &&
                (std::get<2>(lhs) < std::get<2>(rhs)));
    };
    std::set<std::tuple<long, VertexType, VertexType>, decltype(cmp)> edgePriority(cmp);

    for (const VertexType& edgeSrc : graph.Vertices()) {
        for (const VertexType& edgeTgt : graph.Children(edgeSrc)) {
            if (graph.VertexType(edgeSrc) != graph.VertexType(edgeTgt)) {
                continue;
            }

            if (vertexPoset[edgeSrc] + 1 != vertexPoset[edgeTgt]) {
                continue;
            }
            if (topDist[edgeSrc] + commCost + graph.VertexWorkWeight(edgeTgt) != topDist[edgeTgt]) {
                continue;
            }
            if (botDist[edgeTgt] + commCost + graph.VertexWorkWeight(edgeSrc) != botDist[edgeSrc]) {
                continue;
            }
            if (graph.VertexWorkWeight(edgeSrc) + graph.VertexWorkWeight(edgeTgt) > params_.maxWeight_) {
                continue;
            }

            VWorkwT<GraphTIn> maxPath = topDist[edgeSrc] + botDist[edgeTgt] + commCost;
            VWorkwT<GraphTIn> maxParentDist = 0;
            VWorkwT<GraphTIn> maxChildDist = 0;

            for (const auto& par : graph.Parents(edgeSrc)) {
                maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
            }
            for (const auto& par : graph.Parents(edgeTgt)) {
                if (par == edgeSrc) {
                    continue;
                }
                maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
            }

            for (const auto& chld : graph.Children(edgeSrc)) {
                if (chld == edgeTgt) {
                    continue;
                }
                maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
            }
            for (const auto& chld : graph.Children(edgeTgt)) {
                maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
            }

            VWorkwT<GraphTIn> newMaxPath = maxParentDist + maxChildDist + graph.VertexWorkWeight(edgeSrc) +
                                           graph.VertexWorkWeight(edgeTgt);
            long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);

            // cannot have leniency here as it may destroy symmetries
            if (savings >= 0) {
                edgePriority.emplace(savings, edgeSrc, edgeTgt);
            }
        }
    }

    std::vector<bool> partitionedSourceFlag(graph.NumVertices(), false);
    std::vector<bool> partitionedTargetFlag(graph.NumVertices(), false);

    VertexIdxT<GraphTIn> maxCorseningNum = graph.NumVertices() -
                                           static_cast<VertexIdxT<GraphTIn>>(static_cast<double>(graph.NumVertices()) *
                                                                             params_.geomDecay_);

    VertexIdxT<GraphTIn> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = edgePriority.begin(); prioIter != edgePriority.end(); prioIter++) {
        const long& edgeSave = std::get<0>(*prioIter);
        const VertexType& edgeSrc = std::get<1>(*prioIter);
        const VertexType& edgeTgt = std::get<2>(*prioIter);

        // Iterations halt
        if (edgeSave < minSave) {
            break;
        }

        // Check whether we can glue
        if (partitionedSourceFlag[edgeSrc]) {
            continue;
        }
        if (partitionedSourceFlag[edgeTgt]) {
            continue;
        }
        if (partitionedTargetFlag[edgeSrc]) {
            continue;
        }
        if (partitionedTargetFlag[edgeTgt]) {
            continue;
        }

        bool shouldSkipSrc = false;
        for (const VertexType& chld : graph.Children(edgeSrc)) {
            if ((vertexPoset[chld] == vertexPoset[edgeSrc] + 1) && partitionedTargetFlag[chld]) {
                shouldSkipSrc = true;
                break;
            }
        }
        bool shouldSkipTgt = false;
        for (const VertexType& par : graph.Parents(edgeTgt)) {
            if ((vertexPoset[par] + 1 == vertexPoset[edgeTgt]) && partitionedSourceFlag[par]) {
                shouldSkipTgt = true;
                break;
            }
        }
        if (shouldSkipSrc && shouldSkipTgt) {
            continue;
        }

        // Adding to partition
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{edgeSrc, edgeTgt});
        counter++;
        if (counter > maxCorseningNum) {
            minSave = edgeSave;
        }
        partitionedSourceFlag[edgeSrc] = true;
        partitionedTargetFlag[edgeTgt] = true;
    }

    expansionMapOutput.reserve(graph.NumVertices() - counter);
    for (const VertexType& vert : graph.Vertices()) {
        if (partitionedSourceFlag[vert])
            continue;
        if (partitionedTargetFlag[vert])
            continue;

        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}

template <typename GraphTIn, typename GraphTOut>
VertexIdxT<GraphTIn> Sarkar<GraphTIn, GraphTOut>::AllChildrenContraction(
    VWorkwT<GraphTIn> commCost, const GraphTIn& graph,
    std::vector<std::vector<VertexIdxT<GraphTIn>>>& expansionMapOutput) const
{
    using VertexType = VertexIdxT<GraphTIn>;
    constexpr int kMinOutDegreeForContraction = 2;

    expansionMapOutput.clear();

    const std::vector<VertexIdxT<GraphTIn>> vertexPoset = GetTopNodeDistance<GraphTIn, VertexIdxT<GraphTIn>>(graph);
    const std::vector<VWorkwT<GraphTIn>> topDist = GetTopDistance(commCost, graph);
    const std::vector<VWorkwT<GraphTIn>> botDist = GetBotDistance(commCost, graph);

    auto cmp = [](const std::pair<long, VertexType>& lhs, const std::pair<long, VertexType>& rhs) {
        return (lhs.first > rhs.first) || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<long, VertexType>, decltype(cmp)> vertPriority(cmp);

    for (const VertexType& groupHead : graph.Vertices()) {
        if (graph.OutDegree(groupHead) < kMinOutDegreeForContraction) {
            continue;
        }

        bool shouldSkip = false;
        for (const VertexType& groupFoot : graph.Children(groupHead)) {
            if (graph.VertexType(groupHead) != graph.VertexType(groupFoot)) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) {
            continue;
        }
        for (const VertexType& groupFoot : graph.Children(groupHead)) {
            if (vertexPoset[groupFoot] != vertexPoset[groupHead] + 1) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) {
            continue;
        }
        VWorkwT<GraphTIn> combinedWeight = graph.VertexWorkWeight(groupHead);
        for (const VertexType& groupFoot : graph.Children(groupHead)) {
            combinedWeight += graph.VertexWorkWeight(groupFoot);
        }
        if (combinedWeight > params_.maxWeight_) {
            continue;
        }

        VWorkwT<GraphTIn> maxPath = topDist[groupHead] + botDist[groupHead] - graph.VertexWorkWeight(groupHead);
        for (const VertexType& chld : graph.Children(groupHead)) {
            maxPath = std::max(maxPath, topDist[chld] + botDist[chld] - graph.VertexWorkWeight(chld));
        }

        VWorkwT<GraphTIn> maxParentDist = 0;
        VWorkwT<GraphTIn> maxChildDist = 0;

        for (const VertexType& par : graph.Parents(groupHead)) {
            maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
        }
        for (const VertexType& groupFoot : graph.Children(groupHead)) {
            for (const VertexType& par : graph.Parents(groupFoot)) {
                if (par == groupHead) {
                    continue;
                }
                maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
            }
        }

        for (const VertexType& groupFoot : graph.Children(groupHead)) {
            for (const VertexType& chld : graph.Children(groupFoot)) {
                maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
            }
        }

        VWorkwT<GraphTIn> newMaxPath = maxParentDist + maxChildDist + graph.VertexWorkWeight(groupHead);
        for (const VertexType& groupFoot : graph.Children(groupHead)) {
            newMaxPath += graph.VertexWorkWeight(groupFoot);
        }

        long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);
        if (savings + static_cast<long>(params_.leniency_ * static_cast<double>(maxPath)) >= 0) {
            vertPriority.emplace(savings, groupHead);
        }
    }

    std::vector<bool> partitionedFlag(graph.NumVertices(), false);

    VertexIdxT<GraphTIn> maxCorseningNum = graph.NumVertices() -
                                           static_cast<VertexIdxT<GraphTIn>>(static_cast<double>(graph.NumVertices()) *
                                                                             params_.geomDecay_);

    VertexIdxT<GraphTIn> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = vertPriority.begin(); prioIter != vertPriority.end(); prioIter++) {
        const long& vertSave = prioIter->first;
        const VertexType& groupHead = prioIter->second;

        // Iterations halt
        if (vertSave < minSave) {
            break;
        }

        // Check whether we can glue
        if (partitionedFlag[groupHead]) {
            continue;
        }
        bool shouldSkip = false;
        for (const VertexType& groupFoot : graph.Children(groupHead)) {
            if (partitionedFlag[groupFoot]) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) {
            continue;
        }

        // Adding to partition
        std::vector<VertexType> part;
        part.reserve(1 + graph.OutDegree(groupHead));
        part.emplace_back(groupHead);
        for (const VertexType& groupFoot : graph.Children(groupHead)) {
            part.emplace_back(groupFoot);
        }

        expansionMapOutput.emplace_back(std::move(part));
        counter += static_cast<VertexIdxT<GraphTIn>>(graph.OutDegree(groupHead));
        if (counter > maxCorseningNum) {
            minSave = vertSave;
        }
        partitionedFlag[groupHead] = true;
        for (const VertexType& groupFoot : graph.Children(groupHead)) {
            partitionedFlag[groupFoot] = true;
        }
    }

    for (const VertexType& vert : graph.Vertices()) {
        if (partitionedFlag[vert])
            continue;
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}
} // end namespace osp
} // namespace npu::tile_fwk
#include "sarkar.tpp"
#endif // PASS_OSP_SARKAR_HPP
