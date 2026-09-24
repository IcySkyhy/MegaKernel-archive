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
 * \file dag_vector_adapter.h
 * \brief
 */

#ifndef PASS_OSP_DAG_VECTOR_ADAPTER_H
#define PASS_OSP_DAG_VECTOR_ADAPTER_H

#include <vector>

#include "passes/algorithms/osp/graph_implementations/adj_list_impl/cdag_vertex_impl.h"
#include "passes/algorithms/osp/graph_implementations/integral_range.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @brief Adapter to view a pair of adjacency lists (out-neighbors and in-neighbors) as a computational DAG.
 *
 * This class adapts raw adjacency lists (vectors of vectors) into a graph interface compatible with
 * the OSP computational DAG concepts. It stores pointers to the external adjacency lists, so the
 * lifetime of these lists must exceed the lifetime of this adapter.
 *
 * This class satisfies the following concepts:
 * - `is_computational_dag_typed_vertices`
 * - `is_directed_graph`
 * - `has_vertex_weights`
 * - `is_directed_graph_edge_desc`
 *
 * @tparam v_impl The vertex implementation type. This type must satisfy the following requirements:
 * - It must define the following member types:
 *   - `vertex_idx_type`: The type used for vertex indices (e.g., `size_t`).
 *   - `work_weight_type`: The type used for computational work weights.
 *   - `comm_weight_type`: The type used for communication weights.
 *   - `mem_weight_type`: The type used for memory weights.
 *   - `cdag_VertexTypeType`: The type used for vertex types.
 * - It must have the following public data members:
 *   - `id`: Of type `vertex_idx_type`.
 *   - `work_weight`: Of type `work_weight_type`.
 *   - `comm_weight`: Of type `comm_weight_type`.
 *   - `mem_weight`: Of type `mem_weight_type`.
 *   - `vertex_type`: Of type `cdag_VertexTypeType`.
 * - It must be constructible with the signature:
 *   `v_impl(vertex_idx_type id, work_weight_type work_weight, comm_weight_type comm_weight, mem_weight_type mem_weight,
 * cdag_VertexTypeType vertex_type)`
 *
 * @tparam index_t The type used for vertex indices in the adjacency lists.
 */
template <typename VImpl>
class DagVectorAdapter {
public:
    using VertexIdx = typename VImpl::VertexIdxType;
    using IndexT = VertexIdx;

    using VertexWorkWeightType = typename VImpl::WorkWeightType;
    using VertexCommWeightType = typename VImpl::CommWeightType;
    using VertexMemWeightType = typename VImpl::MemWeightType;
    using VertexTypeType = typename VImpl::CDagVertexTypeType;

    DagVectorAdapter() = default;

    /**
     * @brief Constructs a dag_vector_adapter from adjacency lists.
     *
     * @param out_neigbors_ Vector of vectors representing out-neighbors for each vertex.
     * @param in_neigbors_ Vector of vectors representing in-neighbors for each vertex.
     *
     * @warning The adapter stores pointers to these vectors. They must remain valid for the lifetime of the adapter.
     */
    DagVectorAdapter(const std::vector<std::vector<IndexT>>& outNeigbors,
                     const std::vector<std::vector<IndexT>>& inNeigbors)
        : vertices_(outNeigbors.size()),
          outNeigbors_(&outNeigbors),
          inNeigbors_(&inNeigbors),
          numEdges_(0),
          numVertexTypes_(1)
    {
        for (VertexIdx i = 0; i < static_cast<VertexIdx>(outNeigbors.size()); ++i) {
            vertices_[i].id_ = i;
            numEdges_ += outNeigbors[i].size();
        }
    }

    DagVectorAdapter(const DagVectorAdapter& other) = default;
    DagVectorAdapter& operator=(const DagVectorAdapter& other) = default;
    DagVectorAdapter(DagVectorAdapter&& other) noexcept = default;
    DagVectorAdapter& operator=(DagVectorAdapter&& other) noexcept = default;

    ~DagVectorAdapter() = default;

    void SetInOutNeighbors(const std::vector<std::vector<IndexT>>& inNeigbors,
                           const std::vector<std::vector<IndexT>>& outNeigbors)
    {
        outNeigbors_ = &outNeigbors;
        inNeigbors_ = &inNeigbors;

        vertices_.resize(outNeigbors_->size());

        numEdges_ = 0;
        for (VertexIdx i = 0; i < static_cast<VertexIdx>(outNeigbors_->size()); ++i) {
            vertices_[i].id_ = i;
            numEdges_ += outNeigbors[i].size();
        }

        numVertexTypes_ = 1;
    }

    [[nodiscard]] auto Vertices() const { return IntegralRange<VertexIdx>(static_cast<VertexIdx>(vertices_.size())); }
    [[nodiscard]] VertexIdx NumVertices() const { return static_cast<VertexIdx>(vertices_.size()); }
    [[nodiscard]] VertexIdx NumEdges() const { return static_cast<VertexIdx>(numEdges_); }
    [[nodiscard]] auto Parents(const VertexIdx v) const { return (*inNeigbors_)[v]; }
    [[nodiscard]] auto Children(const VertexIdx v) const { return (*outNeigbors_)[v]; }
    [[nodiscard]] VertexIdx InDegree(const VertexIdx v) const
    {
        return static_cast<VertexIdx>((*inNeigbors_)[v].size());
    }
    [[nodiscard]] VertexIdx OutDegree(const VertexIdx v) const
    {
        return static_cast<VertexIdx>((*outNeigbors_)[v].size());
    }
    [[nodiscard]] VertexWorkWeightType VertexWorkWeight(const VertexIdx v) const { return vertices_[v].workWeight_; }
    [[nodiscard]] VertexCommWeightType VertexCommWeight(const VertexIdx v) const { return vertices_[v].commWeight_; }
    [[nodiscard]] VertexMemWeightType VertexMemWeight(const VertexIdx v) const { return vertices_[v].memWeight_; }
    [[nodiscard]] VertexTypeType NumVertexTypes() const { return numVertexTypes_; }
    [[nodiscard]] VertexTypeType VertexType(const VertexIdx v) const { return vertices_[v].vertexType_; }

    void SetVertexWorkWeight(const VertexIdx v, const VertexWorkWeightType workWeight)
    {
        vertices_.at(v).workWeight_ = workWeight;
    }
    void SetVertexMemWeight(const VertexIdx v, const VertexMemWeightType memWeight)
    {
        vertices_.at(v).memWeight_ = memWeight;
    }
    void SetVertexCommWeight(const VertexIdx v, const VertexCommWeightType commWeight)
    {
        vertices_.at(v).commWeight_ = commWeight;
    }
    void SetVertexType(const VertexIdx v, const VertexTypeType vertexType)
    {
        vertices_.at(v).vertexType_ = vertexType;
        numVertexTypes_ = std::max(numVertexTypes_, vertexType + 1);
    }

private:
    std::vector<VImpl> vertices_;

    const std::vector<std::vector<IndexT>>* outNeigbors_;
    const std::vector<std::vector<IndexT>>* inNeigbors_;

    std::size_t numEdges_ = 0;
    unsigned numVertexTypes_ = 0;
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // PASS_OSP_DAG_VECTOR_ADAPTER_HPP
