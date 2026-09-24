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
 * \file computational_dag_vector_impl.h
 * \brief
 */

#ifndef OSP_COMPUTATIONAL_DAG_VECTOR_IMPL_H
#define OSP_COMPUTATIONAL_DAG_VECTOR_IMPL_H

#include <algorithm>
#include <vector>

#include "cdag_vertex_impl.h"
#include "passes/algorithms/osp/concepts/directed_graph_edge_desc_concept.h"
#include "passes/algorithms/osp/graph_algorithms/computational_dag_construction_util.h"
#include "passes/algorithms/osp/graph_implementations/integral_range.h"

namespace npu::tile_fwk {
namespace osp {
/**
 * @brief A vector-based implementation of a computational DAG.
 *
 * This class implements a computational DAG using adjacency lists stored in two std::vectors.
 * It manages the storage of vertices and edges, and provides an interface to query and modify the graph.
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
 * @see CDagVertexImpl for a reference implementation of the vertex type.
 */
template <typename VImpl>
class ComputationalDagVectorImpl {
public:
    using VertexIdx = typename VImpl::VertexIdxType;

    using VertexWorkWeightType = typename VImpl::WorkWeightType;
    using VertexCommWeightType = typename VImpl::CommWeightType;
    using VertexMemWeightType = typename VImpl::MemWeightType;
    using VertexTypeType = typename VImpl::CDagVertexTypeType;

    ComputationalDagVectorImpl() = default;

    /**
     * @brief Constructs a graph with a specified number of vertices.
     *
     * @param NumVertices The number of vertices to initialize.
     */
    explicit ComputationalDagVectorImpl(const VertexIdx numVertices)
        : vertices_(numVertices), outNeigbors_(numVertices), inNeigbors_(numVertices), numEdges_(0), numVertexTypes_(0)
    {
        for (VertexIdx i = 0; i < numVertices; ++i) {
            vertices_[i].id_ = i;
        }
    }

    ComputationalDagVectorImpl(const ComputationalDagVectorImpl& other) = default;
    ComputationalDagVectorImpl& operator=(const ComputationalDagVectorImpl& other) = default;

    /**
     * @brief Constructs a graph from another graph type.
     *
     * This constructor initializes the graph by copying the structure and properties from another graph `other`.
     * The source graph `GraphT` must satisfy the `is_computational_dag` concept.
     *
     * @tparam GraphT The type of the source graph. Must satisfy `is_computational_dag_v`.
     * @param other The source graph to copy from.
     */
    template <typename GraphT>
    explicit ComputationalDagVectorImpl(const GraphT& other)
    {
        ConstructComputationalDag(other, *this);
    }

    ComputationalDagVectorImpl(ComputationalDagVectorImpl&& other) noexcept
        : vertices_(std::move(other.vertices_)),
          outNeigbors_(std::move(other.outNeigbors_)),
          inNeigbors_(std::move(other.inNeigbors_)),
          numEdges_(other.numEdges_),
          numVertexTypes_(other.numVertexTypes_)
    {
        other.numEdges_ = 0;
        other.numVertexTypes_ = 0;
    };

    ComputationalDagVectorImpl& operator=(ComputationalDagVectorImpl&& other) noexcept
    {
        if (this != &other) {
            vertices_ = std::move(other.vertices_);
            outNeigbors_ = std::move(other.outNeigbors_);
            inNeigbors_ = std::move(other.inNeigbors_);
            numEdges_ = other.numEdges_;
            numVertexTypes_ = other.numVertexTypes_;

            other.numEdges_ = 0;
            other.numVertexTypes_ = 0;
        }
        return *this;
    }

    ~ComputationalDagVectorImpl() = default;

    [[nodiscard]] auto Vertices() const { return IntegralRange<VertexIdx>(static_cast<VertexIdx>(vertices_.size())); }
    [[nodiscard]] VertexIdx NumVertices() const { return static_cast<VertexIdx>(vertices_.size()); }
    [[nodiscard]] bool empty() const { return vertices_.empty(); }
    [[nodiscard]] VertexIdx NumEdges() const { return numEdges_; }
    [[nodiscard]] const std::vector<VertexIdx>& Parents(const VertexIdx v) const { return inNeigbors_[v]; }
    [[nodiscard]] const std::vector<VertexIdx>& Children(const VertexIdx v) const { return outNeigbors_[v]; }
    [[nodiscard]] VertexIdx InDegree(const VertexIdx v) const { return static_cast<VertexIdx>(inNeigbors_[v].size()); }
    [[nodiscard]] VertexIdx OutDegree(const VertexIdx v) const
    {
        return static_cast<VertexIdx>(outNeigbors_[v].size());
    }
    [[nodiscard]] VertexCommWeightType VertexCommWeight(const VertexIdx v) const { return vertices_[v].commWeight_; }
    [[nodiscard]] VertexWorkWeightType VertexWorkWeight(const VertexIdx v) const { return vertices_[v].workWeight_; }
    [[nodiscard]] VertexMemWeightType VertexMemWeight(const VertexIdx v) const { return vertices_[v].memWeight_; }
    [[nodiscard]] VertexTypeType VertexType(const VertexIdx v) const { return vertices_[v].vertexType_; }
    VertexIdx AddVertex(const VertexWorkWeightType workWeight, const VertexCommWeightType commWeight,
                        const VertexMemWeightType memWeight, const VertexTypeType vertexType = 0)
    {
        vertices_.emplace_back(vertices_.size(), workWeight, commWeight, memWeight, vertexType);
        outNeigbors_.push_back({});
        inNeigbors_.push_back({});

        numVertexTypes_ = std::max(numVertexTypes_, vertexType + 1);

        return vertices_.back().id_;
    }

    [[nodiscard]] VertexTypeType NumVertexTypes() const { return numVertexTypes_; }

    void SetVertexWorkWeight(const VertexIdx v, const VertexWorkWeightType workWeight)
    {
        vertices_.at(v).workWeight_ = workWeight;
    }

    void SetVertexCommWeight(const VertexIdx v, const VertexCommWeightType commWeight)
    {
        vertices_.at(v).commWeight_ = commWeight;
    }

    void SetVertexMemWeight(const VertexIdx v, const VertexMemWeightType memWeight)
    {
        vertices_.at(v).memWeight_ = memWeight;
    }

    void SetVertexType(const VertexIdx v, const VertexTypeType vertexType)
    {
        vertices_.at(v).vertexType_ = vertexType;
        numVertexTypes_ = std::max(numVertexTypes_, vertexType + 1);
    }

    bool AddEdge(const VertexIdx source, const VertexIdx target)
    {
        if (source >= static_cast<VertexIdx>(vertices_.size()) || target >= static_cast<VertexIdx>(vertices_.size()) ||
            source == target) {
            return false;
        }

        const auto& out = outNeigbors_.at(source);
        if (std::find(out.begin(), out.end(), target) != out.end()) {
            return false;
        }

        outNeigbors_[source].push_back(target);
        inNeigbors_.at(target).push_back(source);
        numEdges_++;

        return true;
    }

private:
    std::vector<VImpl> vertices_;

    std::vector<std::vector<VertexIdx>> outNeigbors_;
    std::vector<std::vector<VertexIdx>> inNeigbors_;

    VertexIdx numEdges_ = 0;
    unsigned numVertexTypes_ = 0;
};

/**
 * @brief Default implementation of a computational DAG using unsigned integer weights.
 */
using ComputationalDagVectorImplDefUnsignedT = ComputationalDagVectorImpl<CDagVertexImplUnsigned>;

/**
 * @brief Default implementation of a computational DAG using signed integer weights.
 */
using ComputationalDagVectorImplDefIntT = ComputationalDagVectorImpl<CDagVertexImplInt>;
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_COMPUTATIONAL_DAG_VECTOR_IMPL_HPP
