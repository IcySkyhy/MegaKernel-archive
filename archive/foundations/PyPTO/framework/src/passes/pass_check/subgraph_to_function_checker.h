/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file subgraph_to_function_checker.h
 * \brief
 */

#ifndef SUBGRAPH_TO_FUNCTION_CHECKER_H
#define SUBGRAPH_TO_FUNCTION_CHECKER_H

#include "checker.h"
#include "interface/utils/common.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"

namespace npu {
namespace tile_fwk {

class SubGraphToFuncChecker : Checker {
public:
    Status DoPreCheck(Function& function) override;
    Status DoPostCheck(Function& function) override;

    void SetInOutGraph(const std::vector<std::vector<size_t>>& inGraph,
                       const std::vector<std::vector<size_t>>& outGraph);
    void SetColorGraph(const std::vector<std::vector<int>>& colorInGraph,
                       const std::vector<std::vector<int>>& colorOutGraph);

private:
    Status NOPCheck(const Operation& op) const;
    Status CheckSubGraphTopo(Function& function) const;

    template <typename eType>
    Status InAndOutGraphConsistencyCheck(const std::vector<std::vector<eType>>& inEdgeGraph,
                                         const std::vector<std::vector<eType>>& outEdgeGraph);

    bool foundNodeInNeighbor(const int dstNode, const std::vector<int>& searchGraph) const;
    Status BuildInGraph(Function& function);
    Status BuildOutGraph(Function& function);
    Status EdgeIndexCheck(const bool found, const int newIndex, const size_t graphSize) const;
    Status CheckInAndOutGraphMatch(Function& function);
    bool HasOnlyViewProducers(const std::set<Operation*, LogicalTensor::CompareOp>& producers);
    Status CheckSubGraphBoundary(Function& function);
    Status VerifyRedundantEdge(const int srcNode, const int dstNode) const;
    Status ColorOutGraphCheck(Function& function) const;
    Status VerifySingleOpTopology(Function& function, size_t opIndex);
    Status CheckReadyStateConsistency(Function& function, size_t opIndex);

private:
    std::vector<std::vector<size_t>> inGraph_;
    std::vector<std::vector<size_t>> outGraph_;
    std::vector<std::vector<int>> colorInGraph_;
    std::vector<std::vector<int>> colorOutGraph_;
    const int kShapePlaceholderForParameterized = -2;
};

} // namespace tile_fwk
} // namespace npu

#endif // SUBGRAPH_TO_FUNCTION_CHECKER_H
