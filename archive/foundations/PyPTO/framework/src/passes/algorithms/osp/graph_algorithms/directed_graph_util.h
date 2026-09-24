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
 * \file directed_graph_util.h
 * \brief
 */

#ifndef OSP_DIRECTED_GRAPH_UTIL_H
#define OSP_DIRECTED_GRAPH_UTIL_H

#include <limits>
#include <queue>
#include <unordered_set>
#include <vector>

/**
 * @file directed_graph_util.hpp
 * @brief Utility functions and classes for working with directed graphs.
 *
 * This file provides a collection of utility functions, iterators, and views
 * for performing operations on directed graphs. These utilities include
 * functions for checking graph properties, retrieving specific vertices,
 * and traversing the graph using BFS and DFS.
 */

namespace npu::tile_fwk {
namespace osp {
/**
 * @brief Checks if there is an edge between two vertices in the graph.
 *
 * @tparam GraphT The type of the graph.
 * @param src The source vertex.
 * @param dest The destination vertex.
 * @param graph The graph to check.
 * @return true if there is an edge from src to dest, false otherwise.
 */
template <typename GraphT>
bool Edge(const VertexIdxT<GraphT>& src, const VertexIdxT<GraphT>& dest, const GraphT& graph)
{
    for (const auto& child : graph.Children(src)) {
        if (child == dest) {
            return true;
        }
    }
    return false;
}

template <typename GraphT>
VWorkwT<GraphT> CriticalPathWeight(const GraphT& graph)
{
    if (graph.NumVertices() == 0) {
        return 0;
    }

    std::vector<VWorkwT<GraphT>> topLength(graph.NumVertices(), 0);
    VWorkwT<GraphT> criticalPathWeight = 0;

    // calculating length of longest path
    for (const auto& node : GetTopOrder(graph)) {
        VWorkwT<GraphT> maxTemp = 0;
        for (const auto& parent : graph.Parents(node)) {
            maxTemp = std::max(maxTemp, topLength[parent]);
        }

        topLength[node] = maxTemp + graph.VertexWorkWeight(node);
        if (topLength[node] > criticalPathWeight) {
            criticalPathWeight = topLength[node];
        }
    }

    return criticalPathWeight;
}

template <typename GraphT>
std::pair<EdgeDescT<GraphT>, bool> EdgeDesc(const VertexIdxT<GraphT>& src, const VertexIdxT<GraphT>& dest,
                                            const GraphT& graph)
{
    for (const auto& edge : OutEdges(src, graph)) {
        if (Target(edge, graph) == dest) {
            return {edge, true};
        }
    }
    return {EdgeDescT<GraphT>(), false};
}

/**
 * @brief Computes the weakly connected components of a directed graph.
 *
 * A weakly connected component is a maximal subgraph where for any two vertices
 * u, v in the subgraph, there is a path between u and v in the underlying
 * undirected graph.
 *
 * @tparam GraphT The type of the graph, which must satisfy the `directed_graph` concept.
 * @param graph The input directed graph.
 * @param[out] components A vector where `components[i]` will be the component ID for vertex `i`.
 * @return The total number of weakly connected components.
 */
template <typename GraphT>
std::size_t ComputeWeaklyConnectedComponents(const GraphT& graph, std::vector<VertexIdxT<GraphT>>& components)
{
    using VertexType = VertexIdxT<GraphT>;

    if (graph.NumVertices() == 0) {
        components.clear();
        return 0;
    }

    components.assign(graph.NumVertices(), std::numeric_limits<VertexType>::max());
    VertexType componentId = 0;

    for (const auto& v : graph.Vertices()) {
        if (components[v] != std::numeric_limits<VertexType>::max()) {
            continue;
        }

        std::vector<VertexType> q;
        q.push_back(v);
        components[v] = componentId;
        size_t head = 0;

        while (head < q.size()) {
            VertexType u = q[head++];
            for (const auto& neighbor : graph.Parents(u)) {
                if (components[neighbor] == std::numeric_limits<VertexType>::max()) {
                    components[neighbor] = componentId;
                    q.push_back(neighbor);
                }
            }
            for (const auto& neighbor : graph.Children(u)) {
                if (components[neighbor] == std::numeric_limits<VertexType>::max()) {
                    components[neighbor] = componentId;
                    q.push_back(neighbor);
                }
            }
        }
        componentId++;
    }
    return componentId;
}

/**
 * @brief Counts the number of weakly connected components in a directed graph.
 * @param graph The input directed graph.
 * @return The number of weakly connected components.
 */
template <typename GraphT>
std::size_t CountWeaklyConnectedComponents(const GraphT& graph)
{
    std::vector<VertexIdxT<GraphT>> components;
    return ComputeWeaklyConnectedComponents(graph, components);
}
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_DIRECTED_GRAPH_UTIL_HPP
