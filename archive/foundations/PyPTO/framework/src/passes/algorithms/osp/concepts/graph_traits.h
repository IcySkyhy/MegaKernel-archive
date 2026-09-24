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
* \file graph_traits.h
* \brief

*/

#ifndef OSP_GRAPH_TRAITS_H
#define OSP_GRAPH_TRAITS_H

#include "interface/utils/common.h"

/**
 * @file graph_traits.hpp
 * @brief Type traits and concepts for graph structures in OneStopParallel.
 *
 * This file defines the core requirements for types used by graph implementations in the library,
 * specifically for computational DAGs. It provides mechanisms
 * to extract types for vertex indices, edge descriptors, and weights,
 * ensuring that graph implementations conform to the expected interfaces.
 */

namespace npu::tile_fwk {
namespace osp {
template <typename T>
using VertexIdxT = typename T::VertexIdx;

template <typename T>
using VWorkwT = typename T::VertexWorkWeightType;

template <typename T>
using VCommwT = typename T::VertexCommWeightType;

template <typename T>
using VMemwT = typename T::VertexMemWeightType;

template <typename T>
using VTypeT = typename T::VertexTypeType;

/**
 * @brief Traits to check for the existence of specific type members.
 *
 * These structs inherit from `std::true_type` if the specified member type exists in `T`,
 * otherwise they inherit from `std::false_type`.
 */

template <typename T, typename = void>
struct HasEdgeDescTmember : std::false_type {};

template <typename T>
struct HasEdgeDescTmember<T, std::void_t<typename T::DirectedEdgeDescriptor>> : std::true_type {};

/**
 * @brief A default edge descriptor for directed graphs.
 *
 * This struct is used when the graph type does not provide its own edge descriptor.
 * It simply holds the source and target vertex indices.
 *
 * @tparam GraphT The graph type.
 */
template <typename GraphT>
struct DirectedEdge {
    VertexIdxT<GraphT> source_;
    VertexIdxT<GraphT> target_;

    bool operator==(const DirectedEdge& other) const { return source_ == other.source_ && target_ == other.target_; }

    bool operator!=(const DirectedEdge& other) const { return !(*this == other); }

    DirectedEdge() : source_(0), target_(0) {}

    DirectedEdge(const DirectedEdge& other) = default;
    DirectedEdge(DirectedEdge&& other) = default;
    DirectedEdge& operator=(const DirectedEdge& other) = default;
    DirectedEdge& operator=(DirectedEdge&& other) = default;
    ~DirectedEdge() = default;

    DirectedEdge(VertexIdxT<GraphT> src, VertexIdxT<GraphT> tgt) : source_(src), target_(tgt) {}
};

/**
 * @brief Helper struct to extract the edge descriptor type of a directed graph.
 *
 * If the graph defines `directed_edge_descriptor`, it is extracted;
 * otherwise, `directed_edge` is used as a default implementation.
 */
template <typename T, bool hasEdge>
struct DirectedGraphEdgeDescTraitsHelper {
    using DirectedEdgeDescriptor = DirectedEdge<T>;
};

template <typename T>
struct DirectedGraphEdgeDescTraits {
    using DirectedEdgeDescriptor = typename DirectedGraphEdgeDescTraitsHelper<
        T, HasEdgeDescTmember<T>::value>::DirectedEdgeDescriptor;
};

template <typename T>
using EdgeDescT = typename DirectedGraphEdgeDescTraits<T>::DirectedEdgeDescriptor;

// -----------------------------------------------------------------------------
// Property Traits
// -----------------------------------------------------------------------------

/**
 * @brief Check if a graph guarantees vertices are stored/iterated in topological order.
 * It allows a graph implementation to notify algorithms that vertices are stored/iterated
 * in topological order which can be used to optimize the algorithm.
 */
template <typename T, typename = void>
struct HasVerticesInTopOrderTrait : std::false_type {};

template <typename T>
struct HasVerticesInTopOrderTrait<T, std::void_t<decltype(T::verticesInTopOrder_)>>
    : std::bool_constant<std::is_same_v<decltype(T::verticesInTopOrder_), const bool> && T::verticesInTopOrder_> {};

template <typename T>
inline constexpr bool hasVerticesInTopOrderV = HasVerticesInTopOrderTrait<T>::value;
} // namespace osp
} // namespace npu::tile_fwk

/**
 * @brief Specialization of std::hash for osp::directed_edge.
 *
 * This specialization provides a hash function for osp::directed_edge, which is used in hash-based containers like
 * std::unordered_set and std::unordered_map.
 */
template <typename GraphT>
struct std::hash<npu::tile_fwk::osp::DirectedEdge<GraphT>> {
    std::size_t operator()(const npu::tile_fwk::osp::DirectedEdge<GraphT>& p) const noexcept
    {
        // Combine hashes of source and target
        std::size_t h1 = std::hash<npu::tile_fwk::osp::VertexIdxT<GraphT>>{}(p.source_);
        std::size_t h2 = std::hash<npu::tile_fwk::osp::VertexIdxT<GraphT>>{}(p.target_);
        npu::tile_fwk::HashCombine(h1, h2);
        return h1;
    }
};
#endif // OSP_GRAPH_TRAITS_HPP
