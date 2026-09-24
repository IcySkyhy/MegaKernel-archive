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
 * \file directed_graph_path_util.h
 * \brief
 */

#ifndef PASS_OSP_DIRECTED_GRAPH_PATH_UTIL_H
#define PASS_OSP_DIRECTED_GRAPH_PATH_UTIL_H

#include <map>
#include <queue>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/graph_algorithms/directed_graph_top_sort.h"

namespace npu::tile_fwk {
namespace osp {
template <typename T = unsigned, typename GraphT, typename NeighborFunc, typename IterFunc>
std::vector<T> GetNodeDistanceImpl(const GraphT& graph, NeighborFunc getNeighbors, IterFunc iterate)
{
    static_assert(std::is_integral_v<T>, "T must be of integral type");
    std::vector<T> distance(graph.NumVertices(), 0);
    iterate([&](auto v) {
        T maxDist = 0;
        for (const auto& n : getNeighbors(v))
            maxDist = std::max(maxDist, distance[n] + 1);
        distance[v] = maxDist;
    });
    return distance;
}

template <typename GraphT, typename T = unsigned>
std::vector<T> GetBottomNodeDistance(const GraphT& graph)
{
    const auto topOrder = GetTopOrder(graph);
    return GetNodeDistanceImpl<T>(
        graph, [&](auto v) { return graph.Children(v); },
        [&](auto f) {
            for (auto it = topOrder.crbegin(); it != topOrder.crend(); ++it)
                f(*it);
        });
}

template <typename GraphT, typename T = unsigned>
std::vector<T> GetTopNodeDistance(const GraphT& graph)
{
    return GetNodeDistanceImpl<T>(
        graph, [&](auto v) { return graph.Parents(v); },
        [&](auto f) {
            for (const auto& v : GetTopOrder(graph))
                f(v);
        });
}
} // namespace osp
} // namespace npu::tile_fwk
#endif // PASS_OSP_DIRECTED_GRAPH_PATH_UTIL_HPP
