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
 * \file coarser.h
 * \brief
 */

#ifndef PASS_OSP_COARSER_H
#define PASS_OSP_COARSER_H

#include <algorithm>
#include <set>
#include <vector>

#include "passes/algorithms/osp/coarser/coarser_util.h"
#include "passes/algorithms/osp/concepts/constructable_computational_dag_concept.h"
#include "passes/algorithms/osp/concepts/graph_traits.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template <typename GraphTIn, typename GraphTOut>
class Coarser {
    // probably too strict, need to be refined.
    // maybe add concept for when Gtaph_t2 is constructable/coarseable from GraphTIn
    static_assert(std::is_same_v<VWorkwT<GraphTIn>, VWorkwT<GraphTOut>>,
                  "GraphTIn and GraphTOut must have the same work weight type");
    static_assert(std::is_same_v<VMemwT<GraphTIn>, VMemwT<GraphTOut>>,
                  "GraphTIn and GraphTOut must have the same memory weight type");
    static_assert(std::is_same_v<VCommwT<GraphTIn>, VCommwT<GraphTOut>>,
                  "GraphTIn and GraphTOut must have the same communication weight type");

public:
    /**
     * @brief Coarsens the input computational DAG into a simplified version.
     *
     * @param dag_in The input computational DAG to be coarsened. It is expected to be a valid graph structure.
     * @param coarsened_dag The output computational DAG after coarsening. It will be populated by this method.
     * @param vertex_contraction_map Output mapping from dag_in to coarsened_dag.
     * @return A status code indicating the success or failure of the coarsening operation.
     */
    virtual bool CoarsenDag(const GraphTIn& dagIn, GraphTOut& coarsenedDag,
                            std::vector<VertexIdxT<GraphTOut>>& vertexContractionMap) = 0;

    /**
     * @brief Destructor for the Coarser class.
     */
    virtual ~Coarser() = default;
};

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template <typename GraphTIn, typename GraphTOut>
class CoarserGenExpansionMap : public Coarser<GraphTIn, GraphTOut> {
public:
    virtual std::vector<std::vector<VertexIdxT<GraphTIn>>> GenerateVertexExpansionMap(const GraphTIn& dagIn) = 0;

    virtual bool CoarsenDag(const GraphTIn& dagIn, GraphTOut& coarsenedDag,
                            std::vector<VertexIdxT<GraphTOut>>& vertexContractionMap) override
    {
        if (dagIn.NumVertices() == 0) {
            vertexContractionMap = std::vector<VertexIdxT<GraphTOut>>();
            return true;
        }

        std::vector<std::vector<VertexIdxT<GraphTIn>>> vertexExpansionMap = GenerateVertexExpansionMap(dagIn);
        coarser_util::ReorderExpansionMap<GraphTIn>(dagIn, vertexExpansionMap);

        vertexContractionMap = coarser_util::InvertVertexExpansionMap<GraphTIn, GraphTOut>(vertexExpansionMap);
        return coarser_util::ConstructCoarseDag(dagIn, coarsenedDag, vertexContractionMap);
    }

    /**
     * @brief Destructor for the CoarserGenExpansionMap class.
     */
    virtual ~CoarserGenExpansionMap() = default;
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // PASS_OSP_COARSER_HPP
