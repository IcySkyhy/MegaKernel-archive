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
 * \file coarser_util.h
 * \brief
 */

#ifndef PASS_OSP_COARSER_UTIL_H
#define PASS_OSP_COARSER_UTIL_H

#include <algorithm>
#include <queue>
#include <set>
#include <vector>

#include "passes/algorithms/osp/concepts/constructable_computational_dag_concept.h"
#include "passes/algorithms/osp/concepts/graph_traits.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_util.h"

namespace npu::tile_fwk {
namespace osp {
template <typename T, typename Ind>
void InversePermuteInplace(std::vector<T>& vec, std::vector<Ind>& perm)
{
    static_assert(std::is_integral_v<Ind>);
    static_assert(std::is_unsigned_v<Ind>);

    for (Ind i = 0; i < perm.size(); ++i) {
        Ind j = i;
        while (i != perm[i]) {
            std::swap(vec[j], vec[perm[i]]);
            j = perm[i];
            std::swap(perm[j], perm[i]);
        }
    }
}

namespace coarser_util {
template <typename GraphTOut>
bool CheckValidContractionMap(const std::vector<VertexIdxT<GraphTOut>>& vertexContractionMap)
{
    std::set<VertexIdxT<GraphTOut>> image(vertexContractionMap.cbegin(), vertexContractionMap.cend());
    const VertexIdxT<GraphTOut> imageSize = static_cast<VertexIdxT<GraphTOut>>(image.size());

    return std::all_of(image.cbegin(), image.cend(), [imageSize](const VertexIdxT<GraphTOut>& vert) {
        return (vert >= static_cast<VertexIdxT<GraphTOut>>(0)) && (vert < imageSize);
    });
}

template <typename T>
struct AccSum {
    T operator()(const T& a, const T& b) { return a + b; }
};

template <typename T>
struct AccMax {
    T operator()(const T& a, const T& b) { return std::max(a, b); }
};

/**
 * @brief Coarsens the input computational DAG into a simplified version.
 *
 * @param dagIn The input computational DAG to be coarsened. It is expected to be a valid graph structure.
 * @param coarsenedDag The output computational DAG after coarsening. It will be populated by this method.
 * @param vertexContractionMap Output mapping from dagIn to coarsenedDag.
 * @return A status code indicating the success or failure of the coarsening operation.
 */
template <typename GraphTIn, class GraphTOut>
bool InitializeCoarseGraph(const GraphTIn& dagIn, GraphTOut& coarsenedDag,
                           const std::vector<VertexIdxT<GraphTOut>>& vertexContractionMap)
{
    static_assert(isDirectConstructableCdagV<GraphTOut> || isConstructableCdagV<GraphTOut>,
                  "Out-Graph must be (directly) constructable.");

    const VertexIdxT<GraphTOut> numVertQuotient = (*std::max_element(vertexContractionMap.cbegin(),
                                                                     vertexContractionMap.cend())) +
                                                  1;

    if constexpr (isDirectConstructableCdagV<GraphTOut>) {
        std::set<std::pair<VertexIdxT<GraphTOut>, VertexIdxT<GraphTOut>>> quotientEdges;
        for (const VertexIdxT<GraphTIn>& vert : dagIn.Vertices()) {
            for (const VertexIdxT<GraphTIn>& chld : dagIn.Children(vert)) {
                if (vertexContractionMap[vert] != vertexContractionMap[chld]) {
                    quotientEdges.emplace(vertexContractionMap[vert], vertexContractionMap[chld]);
                }
            }
        }
        coarsenedDag = GraphTOut(numVertQuotient, quotientEdges);
        for (const VertexIdxT<GraphTIn>& vert : coarsenedDag.Vertices()) {
            coarsenedDag.SetVertexWorkWeight(vert, 0);
            coarsenedDag.SetVertexCommWeight(vert, 0);
            coarsenedDag.SetVertexMemWeight(vert, 0);
        }
    } else if constexpr (isConstructableCdagV<GraphTOut>) {
        coarsenedDag = GraphTOut();
        for (VertexIdxT<GraphTOut> vert = 0; vert < numVertQuotient; ++vert) {
            coarsenedDag.AddVertex(0, 0, 0);
        }
        for (const VertexIdxT<GraphTIn>& vert : dagIn.Vertices()) {
            for (const VertexIdxT<GraphTIn>& chld : dagIn.Children(vert)) {
                if (vertexContractionMap[vert] == vertexContractionMap[chld])
                    continue;

                if (not Edge(vertexContractionMap[vert], vertexContractionMap[chld], coarsenedDag)) {
                    coarsenedDag.AddEdge(vertexContractionMap[vert], vertexContractionMap[chld]);
                }
            }
        }
    } else {
        return false;
    }
    return true;
}

template <typename GraphTIn, class GraphTOut, typename VWorkAccMethod = AccSum<VWorkwT<GraphTIn>>,
          typename VCommAccMethod = AccSum<VCommwT<GraphTIn>>, typename VMemAccMethod = AccSum<VMemwT<GraphTIn>>>
void AccumulateVertexWeights(const GraphTIn& dagIn, GraphTOut& coarsenedDag,
                             const std::vector<VertexIdxT<GraphTOut>>& vertexContractionMap)
{
    for (const VertexIdxT<GraphTIn>& vert : dagIn.Vertices()) {
        coarsenedDag.SetVertexWorkWeight(
            vertexContractionMap[vert],
            VWorkAccMethod()(coarsenedDag.VertexWorkWeight(vertexContractionMap[vert]), dagIn.VertexWorkWeight(vert)));
        coarsenedDag.SetVertexCommWeight(
            vertexContractionMap[vert],
            VCommAccMethod()(coarsenedDag.VertexCommWeight(vertexContractionMap[vert]), dagIn.VertexCommWeight(vert)));
        coarsenedDag.SetVertexMemWeight(
            vertexContractionMap[vert],
            VMemAccMethod()(coarsenedDag.VertexMemWeight(vertexContractionMap[vert]), dagIn.VertexMemWeight(vert)));
    }

    for (const VertexIdxT<GraphTIn>& vert : dagIn.Vertices()) {
        coarsenedDag.SetVertexType(vertexContractionMap[vert], dagIn.VertexType(vert));
    }
}

template <typename GraphTIn, class GraphTOut, typename VWorkAccMethod = AccSum<VWorkwT<GraphTIn>>,
          typename VCommAccMethod = AccSum<VCommwT<GraphTIn>>, typename VMemAccMethod = AccSum<VMemwT<GraphTIn>>>
bool ConstructCoarseDag(const GraphTIn& dagIn, GraphTOut& coarsenedDag,
                        const std::vector<VertexIdxT<GraphTOut>>& vertexContractionMap)
{
    if (vertexContractionMap.size() == 0) {
        coarsenedDag = GraphTOut();
        return true;
    }

    if (!InitializeCoarseGraph(dagIn, coarsenedDag, vertexContractionMap))
        return false;
    AccumulateVertexWeights<GraphTIn, GraphTOut, VWorkAccMethod, VCommAccMethod, VMemAccMethod>(dagIn, coarsenedDag,
                                                                                                vertexContractionMap);
    return true;
}

template <typename GraphTIn>
bool CheckValidExpansionMap(const std::vector<std::vector<VertexIdxT<GraphTIn>>>& vertexExpansionMap)
{
    std::size_t cntr = 0;

    std::vector<bool> preImage;
    for (const std::vector<VertexIdxT<GraphTIn>>& group : vertexExpansionMap) {
        if (group.size() == 0)
            return false;

        for (const VertexIdxT<GraphTIn> vert : group) {
            if (vert < static_cast<VertexIdxT<GraphTIn>>(0))
                return false;
            if (static_cast<std::size_t>(vert) >= preImage.size()) {
                preImage.resize(vert + 1, false);
            }

            if (preImage[vert])
                return false;
            preImage[vert] = true;
            cntr++;
        }
    }

    return (cntr == preImage.size());
}

template <typename GraphTIn, typename GraphTOut>
std::vector<VertexIdxT<GraphTOut>> InvertVertexExpansionMap(
    const std::vector<std::vector<VertexIdxT<GraphTIn>>>& vertexExpansionMap)
{
    VertexIdxT<GraphTIn> numVert = 0;
    for (const auto& group : vertexExpansionMap) {
        for (const VertexIdxT<GraphTIn>& vert : group) {
            numVert = std::max(numVert, vert + 1);
        }
    }

    std::vector<VertexIdxT<GraphTOut>> vertexContractionMap(numVert);
    for (std::size_t i = 0; i < vertexExpansionMap.size(); i++) {
        for (const VertexIdxT<GraphTIn>& vert : vertexExpansionMap[i]) {
            vertexContractionMap[vert] = static_cast<VertexIdxT<GraphTOut>>(i);
        }
    }

    return vertexContractionMap;
}

template <typename GraphTIn>
void ReorderExpansionMap(const GraphTIn& graph, std::vector<std::vector<VertexIdxT<GraphTIn>>>& vertexExpansionMap)
{
    std::vector<std::size_t> vertexContractionMap(graph.NumVertices());
    for (std::size_t i = 0; i < vertexExpansionMap.size(); i++) {
        for (const VertexIdxT<GraphTIn>& vert : vertexExpansionMap[i]) {
            vertexContractionMap[vert] = i;
        }
    }

    std::vector<std::size_t> prec(vertexExpansionMap.size(), 0);
    for (const auto& vert : graph.Vertices()) {
        for (const auto& par : graph.Parents(vert)) {
            if (vertexContractionMap.at(par) != vertexContractionMap.at(vert)) {
                prec[vertexContractionMap.at(vert)] += 1;
            }
        }
    }

    for (auto& comp : vertexExpansionMap) {
        std::nth_element(comp.begin(), comp.begin(), comp.end());
    }

    auto cmp = [&vertexExpansionMap](const std::size_t& lhs, const std::size_t& rhs) {
        return vertexExpansionMap[lhs] > vertexExpansionMap[rhs]; // because priority queue is a max_priority queue
    };

    std::priority_queue<std::size_t, std::vector<std::size_t>, decltype(cmp)> ready(cmp);
    std::vector<std::size_t> topOrder;
    topOrder.reserve(vertexExpansionMap.size());
    for (std::size_t i = 0; i < vertexExpansionMap.size(); ++i) {
        if (prec[i] == 0) {
            ready.emplace(i);
        }
    }

    while (!ready.empty()) {
        const std::size_t nextGroup = ready.top();
        ready.pop();
        topOrder.emplace_back(nextGroup);

        for (const auto& vert : vertexExpansionMap[nextGroup]) {
            for (const auto& chld : graph.Children(vert)) {
                if ((vertexContractionMap.at(vert) != vertexContractionMap.at(chld)) &&
                    (--prec[vertexContractionMap.at(chld)] == 0)) {
                    ready.emplace(vertexContractionMap.at(chld));
                }
            }
        }
    }

    InversePermuteInplace(vertexExpansionMap, topOrder);
    return;
}
} // end namespace coarser_util
} // end namespace osp
} // namespace npu::tile_fwk
#endif // PASS_OSP_COARSER_UTIL_HPP
