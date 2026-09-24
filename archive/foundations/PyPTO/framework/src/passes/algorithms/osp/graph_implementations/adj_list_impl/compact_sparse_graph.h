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
 * \file compact_sparse_graph.h
 * \brief
 */

#ifndef PASS_OSP_COMPACT_SPARSE_GRAPH_H
#define PASS_OSP_COMPACT_SPARSE_GRAPH_H

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <type_traits>
#include <vector>

#include "passes/algorithms/osp/graph_implementations/integral_range.h"

namespace npu::tile_fwk {
namespace osp {
template <typename VertT = std::size_t, typename EdgeT = std::size_t, typename WorkWeightType = unsigned,
          typename CommWeightType = unsigned, typename MemWeightType = unsigned,
          typename VertexTypeTemplateType = unsigned>
class CompactSparseGraph {
    static_assert(std::is_integral<VertT>::value && std::is_integral<EdgeT>::value,
                  "Vertex and edge type must be of integral nature.");
    static_assert(std::is_arithmetic_v<WorkWeightType> && "Work weight must be of arithmetic type.");
    static_assert(std::is_arithmetic_v<CommWeightType> && "Communication weight must be of arithmetic type.");
    static_assert(std::is_arithmetic_v<MemWeightType> && "Memory weight must be of arithmetic type.");
    static_assert(std::is_integral_v<VertexTypeTemplateType> && "Vertex type type must be of integral type.");

public:
    using VertexIdx = VertT;

    using VertexWorkWeightType = WorkWeightType;
    using VertexCommWeightType = CommWeightType;
    using VertexMemWeightType = MemWeightType;
    using VertexTypeType = VertexTypeTemplateType;

    static bool constexpr verticesInTopOrder_ = true;
    static bool constexpr childrenInTopOrder_ = true;
    static bool constexpr childrenInVertexOrder_ = true;
    static bool constexpr parentsInTopOrder_ = true;
    static bool constexpr parentsInVertexOrder_ = true;

    CompactSparseGraph() = default;
    CompactSparseGraph(const CompactSparseGraph& other) = default;
    CompactSparseGraph(CompactSparseGraph&& other) = default;
    CompactSparseGraph& operator=(const CompactSparseGraph& other) = default;
    CompactSparseGraph& operator=(CompactSparseGraph&& other) = default;
    ~CompactSparseGraph() = default;

    template <template <typename, typename...> class Container>
    CompactSparseGraph(VertexIdx numVertices, const Container<std::pair<VertexIdx, VertexIdx>>& edges)
        : numberOfVertices_(numVertices), numberOfEdges_(static_cast<EdgeT>(edges.size()))
    {
        vertWorkWeights_ = std::vector<VertexWorkWeightType>(NumVertices(), 1);
        vertCommWeights_ = std::vector<VertexCommWeightType>(NumVertices(), 0);
        vertMemWeights_ = std::vector<VertexMemWeightType>(NumVertices(), 0);
        numberOfVertexTypes_ = 1;
        vertTypes_ = std::vector<VertexTypeType>(NumVertices(), 0);

        // Construction
        std::vector<std::vector<VertexIdx>> childrenTmp(NumVertices());
        std::vector<EdgeT> numParentsTmp(NumVertices(), 0);

        for (const auto& edge : edges) {
            childrenTmp[edge.first].push_back(edge.second);
            numParentsTmp[edge.second]++;
        }

        std::vector<VertexIdx> cscEdgeChildren;
        cscEdgeChildren.reserve(NumEdges());
        std::vector<EdgeT> cscSourcePtr(NumVertices() + 1);
        std::vector<VertexIdx> csrEdgeParents(NumEdges());
        std::vector<EdgeT> csrTargetPtr;
        csrTargetPtr.reserve(NumVertices() + 1);

        for (VertexIdx vert = 0; vert < NumVertices(); ++vert) {
            cscSourcePtr[vert] = static_cast<EdgeT>(cscEdgeChildren.size());

            std::sort(childrenTmp[vert].begin(), childrenTmp[vert].end());
            for (const auto& chld : childrenTmp[vert]) {
                cscEdgeChildren.emplace_back(chld);
            }
        }
        cscSourcePtr[NumVertices()] = static_cast<EdgeT>(cscEdgeChildren.size());

        csrTargetPtr = std::vector<EdgeT>(NumVertices() + 1, 0);
        for (std::size_t i = 0U; i < numParentsTmp.size(); ++i) {
            csrTargetPtr[i + 1] = csrTargetPtr[i] + numParentsTmp[i];
        }

        std::vector<EdgeT> offset = csrTargetPtr;
        for (VertexIdx vert = 0; vert < NumVertices(); ++vert) {
            for (const auto& chld : childrenTmp[vert]) {
                csrEdgeParents[offset[chld]++] = vert;
            }
        }

        cscOutEdges_ = CompactChildrenEdges(std::move(cscEdgeChildren), std::move(cscSourcePtr));
        csrInEdges_ = CompactParentEdges(std::move(csrEdgeParents), std::move(csrTargetPtr));
    }

    inline auto Vertices() const { return IntegralRange<VertexIdx>(numberOfVertices_); }

    inline VertT NumVertices() const { return numberOfVertices_; }
    inline EdgeT NumEdges() const { return numberOfEdges_; }

    inline auto Parents(const VertexIdx& v) const { return csrInEdges_.Parents(v); }
    inline auto Children(const VertexIdx& v) const { return cscOutEdges_.Children(v); }

    inline EdgeT InDegree(const VertexIdx& v) const { return csrInEdges_.NumberOfParents(v); }
    inline EdgeT OutDegree(const VertexIdx& v) const { return cscOutEdges_.NumberOfChildren(v); }

    inline VertexWorkWeightType VertexWorkWeight(const VertexIdx& v) const { return vertWorkWeights_[v]; }
    inline VertexCommWeightType VertexCommWeight(const VertexIdx& v) const { return vertCommWeights_[v]; }
    inline VertexMemWeightType VertexMemWeight(const VertexIdx& v) const { return vertMemWeights_[v]; }
    inline VertexTypeType VertexType(const VertexIdx& v) const { return vertTypes_[v]; }

    inline VertexTypeType NumVertexTypes() const { return numberOfVertexTypes_; }

    inline void SetVertexWorkWeight(const VertexIdx& v, const VertexWorkWeightType workWeight)
    {
        vertWorkWeights_[v] = workWeight;
    }
    inline void SetVertexCommWeight(const VertexIdx& v, const VertexCommWeightType commWeight)
    {
        vertCommWeights_[v] = commWeight;
    }
    inline void SetVertexMemWeight(const VertexIdx& v, const VertexMemWeightType memWeight)
    {
        vertMemWeights_[v] = memWeight;
    }
    inline void SetVertexType(const VertexIdx& v, const VertexTypeType vertexType)
    {
        vertTypes_[v] = vertexType;
        numberOfVertexTypes_ = std::max(numberOfVertexTypes_, vertexType);
    }

protected:
    class CompactParentEdges {
    public:
        CompactParentEdges() = default;
        CompactParentEdges(const CompactParentEdges& other) = default;
        CompactParentEdges(CompactParentEdges&& other) = default;
        CompactParentEdges& operator=(const CompactParentEdges& other) = default;
        CompactParentEdges& operator=(CompactParentEdges&& other) = default;
        ~CompactParentEdges() = default;

        CompactParentEdges(std::vector<VertexIdx>&& csrEdgeParents, std::vector<EdgeT>&& csrTargetPtr)
            : csrEdgeParents_(std::move(csrEdgeParents)), csrTargetPtr_(std::move(csrTargetPtr)) {};

        inline EdgeT NumberOfParents(const VertexIdx v) const { return csrTargetPtr_[v + 1] - csrTargetPtr_[v]; }

        class ParentRange {
        public:
            ParentRange(const std::vector<VertexIdx>& csrEdgeParents, const std::vector<EdgeT>& csrTargetPtr,
                        const VertexIdx vert)
                : csrEdgeParents_(csrEdgeParents), csrTargetPtr_(csrTargetPtr), vert_(vert) {};

            inline auto cbegin() const
            {
                auto it = csrEdgeParents_.cbegin();
                std::advance(it, csrTargetPtr_[vert_]);
                return it;
            }

            inline auto cend() const
            {
                auto it = csrEdgeParents_.cbegin();
                std::advance(it, csrTargetPtr_[vert_ + 1]);
                return it;
            }

            inline auto end() const { return cend(); }
            inline auto begin() const { return cbegin(); }

            inline auto crbegin() const
            {
                auto it = csrEdgeParents_.crbegin();
                std::advance(it, csrTargetPtr_[csrTargetPtr_.size() - 1] - csrTargetPtr_[vert_ + 1]);
                return it;
            };

            inline auto crend() const
            {
                auto it = csrEdgeParents_.crbegin();
                std::advance(it, csrTargetPtr_[csrTargetPtr_.size() - 1] - csrTargetPtr_[vert_]);
                return it;
            };

            inline auto rend() const { return crend(); }
            inline auto rbegin() const { return crbegin(); }

        private:
            const std::vector<VertexIdx>& csrEdgeParents_;
            const std::vector<EdgeT>& csrTargetPtr_;
            const VertexIdx vert_;
        };

        inline ParentRange Parents(const VertexIdx vert) const
        {
            return ParentRange(csrEdgeParents_, csrTargetPtr_, vert);
        }

    private:
        // Compressed Sparse Row (CSR)
        std::vector<VertexIdx> csrEdgeParents_;
        std::vector<EdgeT> csrTargetPtr_;
    };

    class CompactChildrenEdges {
    public:
        CompactChildrenEdges() = default;
        CompactChildrenEdges(const CompactChildrenEdges& other) = default;
        CompactChildrenEdges(CompactChildrenEdges&& other) = default;
        CompactChildrenEdges& operator=(const CompactChildrenEdges& other) = default;
        CompactChildrenEdges& operator=(CompactChildrenEdges&& other) = default;
        ~CompactChildrenEdges() = default;

        CompactChildrenEdges(std::vector<VertexIdx>&& cscEdgeChildren, std::vector<EdgeT>&& cscSourcePtr)
            : cscEdgeChildren_(std::move(cscEdgeChildren)), cscSourcePtr_(std::move(cscSourcePtr)) {};

        inline EdgeT NumberOfChildren(const VertexIdx v) const { return cscSourcePtr_[v + 1] - cscSourcePtr_[v]; }

        inline VertexIdx Source(const EdgeT& indx) const
        {
            auto it = std::upper_bound(cscSourcePtr_.cbegin(), cscSourcePtr_.cend(), indx);
            VertexIdx src = static_cast<VertexIdx>(std::distance(cscSourcePtr_.cbegin(), it) - 1);
            return src;
        };

        inline VertexIdx Target(const EdgeT& indx) const { return cscEdgeChildren_[indx]; }

        inline EdgeT ChildrenIndxBegin(const VertexIdx& vert) const { return cscSourcePtr_[vert]; }

        class ChildrenRange {
        public:
            ChildrenRange(const std::vector<VertexIdx>& cscEdgeChildren, const std::vector<EdgeT>& cscSourcePtr,
                          const VertexIdx vert)
                : cscEdgeChildren_(cscEdgeChildren), cscSourcePtr_(cscSourcePtr), vert_(vert) {};

            inline auto cbegin() const
            {
                auto it = cscEdgeChildren_.cbegin();
                std::advance(it, cscSourcePtr_[vert_]);
                return it;
            };

            inline auto cend() const
            {
                auto it = cscEdgeChildren_.cbegin();
                std::advance(it, cscSourcePtr_[vert_ + 1]);
                return it;
            };

            inline auto begin() const { return cbegin(); }
            inline auto end() const { return cend(); }

            inline auto crbegin() const
            {
                auto it = cscEdgeChildren_.crbegin();
                std::advance(it, cscSourcePtr_[cscSourcePtr_.size() - 1] - cscSourcePtr_[vert_ + 1]);
                return it;
            };

            inline auto crend() const
            {
                auto it = cscEdgeChildren_.crbegin();
                std::advance(it, cscSourcePtr_[cscSourcePtr_.size() - 1] - cscSourcePtr_[vert_]);
                return it;
            };

            inline auto rbegin() const { return crbegin(); }
            inline auto rend() const { return crend(); }

        private:
            const std::vector<VertexIdx>& cscEdgeChildren_;
            const std::vector<EdgeT>& cscSourcePtr_;
            const VertexIdx vert_;
        };

        inline ChildrenRange Children(const VertexIdx vert) const
        {
            return ChildrenRange(cscEdgeChildren_, cscSourcePtr_, vert);
        }

    private:
        // Compressed Sparse Column (CSC)
        std::vector<VertexIdx> cscEdgeChildren_;
        std::vector<EdgeT> cscSourcePtr_;
    };

    VertexIdx numberOfVertices_ = static_cast<VertT>(0);
    EdgeT numberOfEdges_ = static_cast<EdgeT>(0);

    CompactParentEdges csrInEdges_;
    CompactChildrenEdges cscOutEdges_;

    VertexTypeType numberOfVertexTypes_ = static_cast<VertexTypeType>(1);

    std::vector<VertexWorkWeightType> vertWorkWeights_;
    std::vector<VertexCommWeightType> vertCommWeights_;
    std::vector<VertexMemWeightType> vertMemWeights_;
    std::vector<VertexTypeType> vertTypes_;

private:
    using ThisT = CompactSparseGraph<VertT, EdgeT, WorkWeightType, CommWeightType, MemWeightType,
                                     VertexTypeTemplateType>;
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // PASS_OSP_COMPACT_SPARSE_GRAPH_HPP
