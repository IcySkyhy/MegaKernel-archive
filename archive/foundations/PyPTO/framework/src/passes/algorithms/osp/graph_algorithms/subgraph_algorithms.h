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
 * \file subgraph_algorithms.h
 * \brief
 */

#ifndef OSP_SUBGRAPH_ALGORITHMS_H
#define OSP_SUBGRAPH_ALGORITHMS_H

#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "passes/algorithms/osp/concepts/graph_traits.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_top_sort.h"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/compact_sparse_graph.h"
#include "passes/algorithms/osp/concepts/constructable_computational_dag_concept.h"

namespace npu::tile_fwk {
namespace osp {
template <typename GraphTIn, typename GraphTOut>
std::unordered_map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>> CreateInducedSubgraphMap(
    const GraphTIn& dag, GraphTOut& dagOut, const std::vector<VertexIdxT<GraphTIn>>& selectedNodes)
{
    std::unordered_map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>> localIdx;
    localIdx.reserve(selectedNodes.size());

    for (const auto& node : selectedNodes) {
        localIdx[node] = dagOut.NumVertices();
        dagOut.AddVertex(dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node),
                         dag.VertexType(node));
    }

    for (const auto& node : selectedNodes) {
        for (const auto& pred : dag.Parents(node)) {
            if (localIdx.count(pred)) {
                dagOut.AddEdge(localIdx[pred], localIdx[node]);
            }
        }
    }

    return localIdx;
}
} // end namespace osp
} // namespace npu::tile_fwk
#endif // OSP_SUBGRAPH_ALGORITHMS_HPP
