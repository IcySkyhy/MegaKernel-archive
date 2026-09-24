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
 * \file computational_dag_construction_util.h
 * \brief
 */

#ifndef OSP_COMPUTATIONAL_DAG_CONSTRUCTION_UTIL_H
#define OSP_COMPUTATIONAL_DAG_CONSTRUCTION_UTIL_H

#include "passes/algorithms/osp/concepts/constructable_computational_dag_concept.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @brief Constructs a computational DAG from another graph.
 *
 * This function copies the structure and properties of a source graph into a target graph structure.
 * Assumes that the vertices of the source graph are indexed from 0 to N-1. If the target
 * graph is empty, indices are sequentially assigned starting from 0. If the target graph
 * is not empty, new vertices will be added to the target graph and their indices will be
 * sequentially assigned starting from the index N.
 *
 * @tparam GraphFrom The type of the source graph. Must satisfy `is_computational_dag`.
 * @tparam GraphTo The type of the target graph. Must satisfy `is_constructable_cdag_vertex`.
 * @param from The source graph.
 * @param to The target graph.
 */
template <typename GraphFrom, typename GraphTo>
void ConstructComputationalDag(const GraphFrom& from, GraphTo& to)
{
    std::vector<VertexIdxT<GraphTo>> vertexMap;
    vertexMap.reserve(from.NumVertices());

    for (const auto& vIdx : from.Vertices()) {
        vertexMap.push_back(to.AddVertex(from.VertexWorkWeight(vIdx), from.VertexCommWeight(vIdx),
                                         from.VertexMemWeight(vIdx), from.VertexType(vIdx)));
    }

    for (const auto& v : from.Vertices()) {
        for (const auto& child : from.Children(v)) {
            to.AddEdge(vertexMap[v], vertexMap[child]);
        }
    }
}
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_COMPUTATIONAL_DAG_CONSTRUCTION_UTIL_HPP
