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
 * \file osp_partitioner.h
 * \brief This file declares the OspPartitioner class, which is a tile graph pass that fuses
 * single operations into kernels using OSP algorithms.
 */

#ifndef PASS_OSP_PARTITIONER_H
#define PASS_OSP_PARTITIONER_H

#include "passes/algorithms/osp/graph_implementations/adj_list_impl/dag_vector_adapter.h"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/compact_sparse_graph.h"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.h"
#include "passes/pass_utils/graph_utils.h"

#include "tilefwk/platform.h"
#include "supernode_graph_builder.h"
#include "passes/pass_interface/pass.h"

#include <unordered_map>

namespace npu::tile_fwk {
namespace osp {
template <typename GraphT>
class BspArchitecture;
template <typename GraphT>
class BspInstance;
} // namespace osp

template <typename WorkType>
struct ArchParameters {
    WorkType commCost_ = 1;
    WorkType synchCost_ = 8000;
    double commCorrectionFactor_ = 0.01;
    WorkType partitionWorkUpperBound_ = std::numeric_limits<WorkType>::max();
    WorkType partitionWorkLowerBound_ = std::numeric_limits<WorkType>::lowest();
};

enum class OspMode { SARKAR = 1, MERKLEBSP = 2 };

class OspPartitioner : public SuperNodeGraphBuilder {
public:
    OspPartitioner(OspMode mode) : SuperNodeGraphBuilder(GraphUtils::IsCVMixPlatform()), ospMode_(mode) {};
    OspPartitioner(OspMode mode, bool useCVMixPartition) : SuperNodeGraphBuilder(useCVMixPartition), ospMode_(mode) {};
    ~OspPartitioner() = default;

    Status SetParameter(const Function& function);
    Status PartitionGraph(Function& function);

private:
    using VertType = int32_t;
    using WorkType = int32_t;
    using VTypeType = unsigned;
    using VertexImpl = osp::CDagVertexImpl<VertType, WorkType, WorkType, WorkType, VTypeType>;
    using GraphType = osp::DagVectorAdapter<VertexImpl>;
    using ConstrGraphType = osp::ComputationalDagVectorImpl<VertexImpl>;
    using CoarseGraphType = osp::CompactSparseGraph<VertType, VertType, WorkType, WorkType, WorkType, VTypeType>;

    // Core/Vertex type translation maps
    const std::unordered_map<OpCoreType, VTypeType> ospCoreTypeMapSplit{
        {OpCoreType::AIC, 0U}, {OpCoreType::AIV, 1U}, {OpCoreType::AICPU, 2U},
        {OpCoreType::ANY, 3U}, {OpCoreType::HUB, 4U}, {OpCoreType::GMATOMIC, 5U}};
    const std::unordered_map<OpCoreType, VTypeType> ospCoreTypeMapMix{
        {OpCoreType::AIC, 0U}, {OpCoreType::AIV, 0U}, {OpCoreType::AICPU, 1U},
        {OpCoreType::ANY, 2U}, {OpCoreType::HUB, 3U}, {OpCoreType::GMATOMIC, 4U}};

    // Parameters
    ArchParameters<WorkType> archParameters_;
    OspMode ospMode_;

    // Init
    Status BuildSuperNodeGraph() override;

    // Construction of OSP instance
    Status ConstructDagCVSplit(GraphType& graph);
    Status ConstructDagCVMix(GraphType& graph);
    Status ConstructDag(GraphType& graph);
    void ConstructBspArchCVSplit(osp::BspArchitecture<GraphType>& bspArch);
    Status ConstructBspArchCVMix(osp::BspArchitecture<GraphType>& bspArch);
    Status ConstructBspInstance(osp::BspInstance<GraphType>& bspInst);

    // Construction Helpers
    void SetVertexCommMemWeight(GraphType& graph, int32_t vertex);
    VTypeType GetOspCoreTypeSplit(OpCoreType coreType);
    VTypeType GetOspCoreTypeMix(OpCoreType coreType);

    // Run OSP Partition
    Status RunOspPartition(Function& function);
    Status UpdatePartitionResult(Function& function, std::vector<VertType>& vertexContractionMap);

    // Algorithms
    Status RunSarkar(const GraphType& graph, CoarseGraphType& coarseGraph, std::vector<VertType>& vertexContractionMap);
    Status RunMerkleBsp(const osp::BspInstance<GraphType>& bspInst, std::vector<VertType>& vertexContractionMap);

    // Helpers
    uint64_t CombineHash(const uint64_t h1, const uint64_t h2) const override;
    uint64_t CombineNeighborHashes(uint64_t baseHash, const std::vector<int32_t>& neighbors,
                                   const std::vector<uint64_t>& hashSource);
    void BuildNodeHashValues(const std::vector<uint64_t>& opHashList);
    Status BuildHashValues() override;
};

} // namespace npu::tile_fwk
#endif // PASS_OSP_PARTITIONER_H
