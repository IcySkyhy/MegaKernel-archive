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
 * \file directed_graph_top_sort.h
 * \brief
 */

#ifndef PASS_OSP_DIRECTED_GRAPH_TOP_SORT_H
#define PASS_OSP_DIRECTED_GRAPH_TOP_SORT_H

#include <limits>
#include <queue>
#include <random>
#include <vector>

#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {
namespace osp {
template <typename GraphT>
std::vector<VertexIdxT<GraphT>> GetTopOrder(const GraphT& graph)
{
    if constexpr (hasVerticesInTopOrderV<GraphT>) {
        std::vector<VertexIdxT<GraphT>> topOrd(graph.NumVertices());
        std::iota(topOrd.begin(), topOrd.end(), static_cast<VertexIdxT<GraphT>>(0));
        return topOrd;
    } else {
        using VertexType = VertexIdxT<GraphT>;

        std::vector<VertexType> predecessorsCount(graph.NumVertices(), 0);
        std::vector<VertexType> topOrder;
        topOrder.reserve(graph.NumVertices());

        std::queue<VertexType> next;

        // Find source nodes
        for (const VertexType& v : graph.Vertices()) {
            if (graph.InDegree(v) == 0) {
                next.push(v);
            }
        }

        // Execute BFS
        while (!next.empty()) {
            const VertexType node = next.front();
            next.pop();
            topOrder.push_back(node);

            for (const VertexType& current : graph.Children(node)) {
                ++predecessorsCount[current];
                if (predecessorsCount[current] == graph.InDegree(current)) {
                    next.push(current);
                }
            }
        }

        if (static_cast<VertexType>(topOrder.size()) != graph.NumVertices()) {
            APASS_LOG_ERROR_F(Elements::Config,
                              "Error during topological ordering: "
                              "TopOrder.size() != graph.NumVertices() [%d != %d]",
                              static_cast<int>(topOrder.size()), static_cast<int>(graph.NumVertices()));
            throw std::runtime_error("Error during topological ordering: TopOrder.size() != graph.NumVertices() [" +
                                     std::to_string(topOrder.size()) + " != " + std::to_string(graph.NumVertices()) +
                                     "]");
        }

        return topOrder;
    }
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> GetTopOrderReverse(const GraphT& graph)
{
    std::vector<VertexIdxT<GraphT>> topOrder = GetTopOrder(graph);
    std::reverse(topOrder.begin(), topOrder.end());
    return topOrder;
}
} // namespace osp
} // namespace npu::tile_fwk
#endif // PASS_OSP_DIRECTED_GRAPH_TOP_SORT_HPP
