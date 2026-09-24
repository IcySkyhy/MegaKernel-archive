/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PASS_PRIOR_DFS_SORT_H
#define PASS_PRIOR_DFS_SORT_H

#include "optimize_sort.h"

namespace npu::tile_fwk {

enum class PromotePriority {
    Assemble = 0,
    Skip = 1,
    DdrCopyOut = 2,
    Normal = 3,
};

class PriorDFSSort : public OptimizeSort {
public:
    using OptimizeSort::OptimizeSort;

private:
    Status DoSortOps() override;
    void PromoteOps();
    PromotePriority ClassifyPromoteOp(Operation* op) const;

    Status PriorDFS(const std::unordered_map<Opcode, int>& preNodePriority);
    void SortOutNodeQueue(std::vector<Operation*>& outNodeQueue);
    Status DFSFromOutNode(const std::vector<Operation*>& outNodeQueue,
                          const std::unordered_map<Opcode, int>& preNodePriority, std::map<Operation*, bool>& visited);
    void DFSFromSingleNode(Operation* op, std::map<Operation*, bool>& visited, std::vector<Operation*>& newOpList,
                           const std::unordered_map<Opcode, int>& preNodePriority);
    void ForwardDfs(Operation* curOp, std::vector<Operation*>& newOpList, std::map<Operation*, bool>& visited,
                    const std::unordered_map<Opcode, int>& preNodePriority, std::deque<Operation*>& queue);
    void QueueNotReadyPreNode(Operation* curOp, std::map<Operation*, bool>& visited,
                              const std::unordered_map<Opcode, int>& preNodePriority, std::deque<Operation*>& queue);
    int GetMaxDepthSimple(Operation* op);
    int GetNodePriority(const std::unordered_map<Opcode, int>& preNodePriority, Operation* op);
    Operation* FindNodeMinNumUnvisitedPreNode(std::map<Operation*, bool> visited,
                                              const std::vector<Operation*>& outNodeQueue);
    int GetNumUnvisitPreNode(Operation* op, std::map<Operation*, bool>& visited);
    void UpdatePreNodeQueue(std::unordered_set<Operation*>& curr, std::unordered_set<Operation*>& preNodeTotal,
                            std::map<Operation*, bool>& visited);

    std::unordered_map<Operation*, int> depthCache_;
};

} // namespace npu::tile_fwk
#endif // PASS_PRIOR_DFS_SORT_H
