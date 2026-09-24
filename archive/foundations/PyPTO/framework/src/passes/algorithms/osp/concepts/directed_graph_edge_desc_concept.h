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
 * \file directed_graph_edge_desc_concept.h
 * \brief
 */

#ifndef OSP_DIRECTED_GRAPH_EDGE_DESC_CONCEPT_H
#define OSP_DIRECTED_GRAPH_EDGE_DESC_CONCEPT_H

#include "graph_traits.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_edge_view.h"

/**
 * @file directed_graph_edge_desc_concept.hpp
 * @brief Default implementations for edge descriptors in directed graphs.
 */
namespace npu::tile_fwk {
namespace osp {
/**
 * @brief Default implementation to get the source vertex of an edge.
 */
template <typename GraphT>
inline VertexIdxT<GraphT> Source(const DirectedEdge<GraphT>& edge, const GraphT&)
{
    return edge.source_;
}

/**
 * @brief Default implementation to get the target vertex of an edge.
 */
template <typename GraphT>
inline VertexIdxT<GraphT> Target(const DirectedEdge<GraphT>& edge, const GraphT&)
{
    return edge.target_;
}

/**
 * @brief Get a view of all edges in the graph.
 */
template <typename GraphT>
inline EdgeView<GraphT> Edges(const GraphT& graph)
{
    return EdgeView<GraphT>(graph);
}

/**
 * @brief Get a view of outgoing edges from a vertex.
 */
template <typename GraphT>
inline OutEdgeView<GraphT> OutEdges(VertexIdxT<GraphT> u, const GraphT& graph)
{
    return OutEdgeView<GraphT>(graph, u);
}

/**
 * @brief Get a view of incoming edges to a vertex.
 */
template <typename GraphT>
inline InEdgeView<GraphT> InEdges(VertexIdxT<GraphT> v, const GraphT& graph)
{
    return InEdgeView<GraphT>(graph, v);
}
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_DIRECTED_GRAPH_EDGE_DESC_CONCEPT_HPP
