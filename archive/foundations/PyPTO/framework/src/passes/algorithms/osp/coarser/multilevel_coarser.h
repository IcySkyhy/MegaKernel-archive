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
 * \file multilevel_coarser.h
 * \brief
 */

#ifndef PASS_OSP_MULTILEVEL_COARSER_H
#define PASS_OSP_MULTILEVEL_COARSER_H

#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "passes/algorithms/osp/bsp/scheduler/scheduler.h" // for ReturnStatus
#include "passes/algorithms/osp/coarser/coarser.h"

namespace npu::tile_fwk {
namespace osp {
template <typename GraphT, typename GraphTCoarse>
class MultilevelCoarser : public Coarser<GraphT, GraphTCoarse> {
public:
    MultilevelCoarser() : originalGraph_(nullptr) {};
    MultilevelCoarser(const GraphT& graph) : originalGraph_(&graph) {};
    virtual ~MultilevelCoarser() = default;

    bool CoarsenDag(const GraphT& dagIn, GraphTCoarse& coarsenedDag,
                    std::vector<VertexIdxT<GraphTCoarse>>& vertexContractionMap) override;

    ReturnStatus Run(const GraphT& graph);

protected:
    inline const GraphT* GetOriginalGraph() const { return originalGraph_; }

    std::vector<std::unique_ptr<GraphTCoarse>> dagHistory_;
    std::vector<std::unique_ptr<std::vector<VertexIdxT<GraphTCoarse>>>> contractionMaps_;

    ReturnStatus AddContraction(std::vector<VertexIdxT<GraphTCoarse>>&& contractionMap);
    ReturnStatus AddContraction(std::vector<VertexIdxT<GraphTCoarse>>&& contractionMap, GraphTCoarse&& contractedGraph);
    void AddIdentityContraction();

    std::vector<VertexIdxT<GraphTCoarse>> GetCombinedContractionMap() const;

    virtual ReturnStatus RunContractions() = 0;
    void CompactifyDagHistory();

    void ClearComputationData();

private:
    const GraphT* originalGraph_;
};

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarser<GraphT, GraphTCoarse>::Run(const GraphT& graph)
{
    ClearComputationData();
    originalGraph_ = &graph;

    ReturnStatus status = ReturnStatus::OSP_SUCCESS;
    status = std::max(status, RunContractions());

    if (dagHistory_.size() == 0) {
        AddIdentityContraction();
    }

    return status;
}

template <typename GraphT, typename GraphTCoarse>
void MultilevelCoarser<GraphT, GraphTCoarse>::ClearComputationData()
{
    dagHistory_.clear();
    dagHistory_.shrink_to_fit();

    contractionMaps_.clear();
    contractionMaps_.shrink_to_fit();
}

template <typename GraphT, typename GraphTCoarse>
void MultilevelCoarser<GraphT, GraphTCoarse>::CompactifyDagHistory()
{
    constexpr size_t kMinHistorySizeForCompactify = 3;
    constexpr double kCompactifyRatioThreshold = 1.25;

    if (dagHistory_.size() < kMinHistorySizeForCompactify) {
        return;
    }

    size_t dagIndxFirst = dagHistory_.size() - 2;
    size_t mapIndxFirst = contractionMaps_.size() - 2;

    size_t dagIndxSecond = dagHistory_.size() - 1;
    size_t mapIndxSecond = contractionMaps_.size() - 1;

    if ((static_cast<double>(dagHistory_[dagIndxFirst - 1]->NumVertices()) /
         static_cast<double>(dagHistory_[dagIndxSecond - 1]->NumVertices())) > kCompactifyRatioThreshold) {
        return;
    }

    // Compute combined contraction_map
    std::unique_ptr<std::vector<VertexIdxT<GraphTCoarse>>>
        combiContractionMap = std::make_unique<std::vector<VertexIdxT<GraphTCoarse>>>(
            contractionMaps_[mapIndxFirst]->size());
    for (std::size_t vert = 0; vert < contractionMaps_[mapIndxFirst]->size(); ++vert) {
        combiContractionMap->at(vert) = contractionMaps_[mapIndxSecond]->at(contractionMaps_[mapIndxFirst]->at(vert));
    }

    // Delete ComputationalDag
    auto dagIt = dagHistory_.begin();
    std::advance(dagIt, dagIndxFirst);
    dagHistory_.erase(dagIt);

    // Delete contraction map
    auto contrMapIt = contractionMaps_.begin();
    std::advance(contrMapIt, mapIndxSecond);
    contractionMaps_.erase(contrMapIt);

    // Replace contraction map
    contractionMaps_[mapIndxFirst] = std::move(combiContractionMap);
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(
    std::vector<VertexIdxT<GraphTCoarse>>&& contractionMap)
{
    std::unique_ptr<GraphTCoarse> newGraph = std::make_unique<GraphTCoarse>();

    auto contrMapPtr = std::make_unique<std::vector<VertexIdxT<GraphTCoarse>>>(
        std::forward<std::vector<VertexIdxT<GraphTCoarse>>>(contractionMap));
    contractionMaps_.emplace_back(std::move(contrMapPtr));

    bool success = false;

    if (dagHistory_.size() == 0) {
        success = coarser_util::ConstructCoarseDag<GraphT, GraphTCoarse>(*(GetOriginalGraph()), *newGraph,
                                                                         *(contractionMaps_.back()));
    } else {
        success = coarser_util::ConstructCoarseDag<GraphTCoarse, GraphTCoarse>(*(dagHistory_.back()), *newGraph,
                                                                               *(contractionMaps_.back()));
    }

    dagHistory_.emplace_back(std::move(newGraph));

    if (success) {
        CompactifyDagHistory();
        return ReturnStatus::OSP_SUCCESS;
    } else {
        return ReturnStatus::OSP_ERROR;
    }
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(
    std::vector<VertexIdxT<GraphTCoarse>>&& contractionMap, GraphTCoarse&& contractedGraph)
{
    std::unique_ptr<GraphTCoarse> graphPtr = std::make_unique<GraphTCoarse>(
        std::forward<GraphTCoarse>(contractedGraph));
    dagHistory_.emplace_back(std::move(graphPtr));

    auto contrMapPtr = std::make_unique<std::vector<VertexIdxT<GraphTCoarse>>>(
        std::forward<std::vector<VertexIdxT<GraphTCoarse>>>(contractionMap));
    contractionMaps_.emplace_back(std::move(contrMapPtr));

    CompactifyDagHistory();
    return ReturnStatus::OSP_SUCCESS;
}

template <typename GraphT, typename GraphTCoarse>
std::vector<VertexIdxT<GraphTCoarse>> MultilevelCoarser<GraphT, GraphTCoarse>::GetCombinedContractionMap() const
{
    std::vector<VertexIdxT<GraphTCoarse>> combinedContractionMap(originalGraph_->NumVertices());
    std::iota(combinedContractionMap.begin(), combinedContractionMap.end(), 0);

    for (std::size_t j = 0; j < contractionMaps_.size(); ++j) {
        for (std::size_t i = 0; i < combinedContractionMap.size(); ++i) {
            combinedContractionMap[i] = contractionMaps_[j]->at(combinedContractionMap[i]);
        }
    }

    return combinedContractionMap;
}

template <typename GraphT, typename GraphTCoarse>
bool MultilevelCoarser<GraphT, GraphTCoarse>::CoarsenDag(const GraphT& dagIn, GraphTCoarse& coarsenedDag,
                                                         std::vector<VertexIdxT<GraphTCoarse>>& vertexContractionMap)
{
    ClearComputationData();

    ReturnStatus status = Run(dagIn);
    if (status != ReturnStatus::OSP_SUCCESS && status != ReturnStatus::OSP_BEST_FOUND) {
        return false;
    }

    if (dagHistory_.size() == 0) {
        return false;
    };
    coarsenedDag = *(dagHistory_.back());

    vertexContractionMap = GetCombinedContractionMap();

    return true;
}

template <typename GraphT, typename GraphTCoarse>
void MultilevelCoarser<GraphT, GraphTCoarse>::AddIdentityContraction()
{
    std::size_t nVert;
    if (dagHistory_.size() == 0) {
        nVert = static_cast<std::size_t>(originalGraph_->NumVertices());
    } else {
        nVert = static_cast<std::size_t>(dagHistory_.back()->NumVertices());
    }

    std::vector<VertexIdxT<GraphTCoarse>> contractionMap(nVert);
    std::iota(contractionMap.begin(), contractionMap.end(), 0);

    AddContraction(std::move(contractionMap));
    CompactifyDagHistory();
}
} // end namespace osp
} // namespace npu::tile_fwk
#endif // PASS_OSP_MULTILEVEL_COARSER_HPP
