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
 * \file iso_matcher.h
 * \brief Task-level isomorphism matching + ordered pairing.
 *        FindTaskEntryOps locates entry points (ops with no in-task producer),
 *        then IsoMatchChains BFSes over ConsumerOps() data-dependency successors
 *        layer by layer from all entry points simultaneously. A node pair is
 *        isomorphic iff same opcode and, for every output operand, same
 *        MemoryType and same GetShape(). On ambiguous (>1 candidate) matches,
 *        pair the min-opmagic one at current layer but do NOT descend deeper
 *        for that node (per-node truncation). Result pairs are ordered by
 *        layer then A-side opmagic, giving a 1:1 ordering correspondence.
 */

#ifndef PASS_ISO_MATCHER_H_
#define PASS_ISO_MATCHER_H_

#include <cstddef>
#include <unordered_set>
#include <vector>

namespace npu::tile_fwk {

class Operation;

struct IsoPair {
    Operation* opA{nullptr};
    Operation* opB{nullptr};
    int depth{0}; // BFS depth, entry points are depth 0.
};

struct IsoMatchResult {
    bool rootIsomorphic{false};
    std::vector<IsoPair> pairs;      // Non-alloc pairs, emitted in BFS order.
    std::vector<IsoPair> allocPairs; // Alloc pairs collected during BFS.
    int maxMatchedDepth{0};
    std::size_t truncatedCount{0}; // Number of ambiguous nodes that were not expanded further.
};

// Find task entry points: ProducerOps is empty, or all producers are outside the task.
// If no entry point is found, return an empty list and let the caller treat it as non-isomorphic.
std::vector<Operation*> FindTaskEntryOps(const std::vector<Operation*>& taskOps,
                                         const std::unordered_set<Operation*>& taskSet);

// Multi-entry isomorphism matching: run BFS from all rootsA/rootsB entry points at the same time.
// BFS only follows ConsumerOps inside taskOpsA/taskOpsB.
// rootIsomorphic means the entry-point signature multisets are identical.
IsoMatchResult IsoMatchChains(const std::vector<Operation*>& rootsA, const std::vector<Operation*>& rootsB,
                              const std::unordered_set<Operation*>& taskOpsA,
                              const std::unordered_set<Operation*>& taskOpsB);

} // namespace npu::tile_fwk

#endif // PASS_ISO_MATCHER_H_
