/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "prior_dfs_sort.h"
#include "passes/pass_log/pass_log.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <set>
#include <unordered_set>

namespace npu::tile_fwk {

void PriorDFSSort::UpdatePreNodeQueue(std::unordered_set<Operation*>& curr,
                                      std::unordered_set<Operation*>& preNodeTotal, std::map<Operation*, bool>& visited)
{
    std::unordered_set<Operation*> next;
    for (auto& curOp : curr) {
        for (auto& preOp : state_.depManager.GetPredecessors(curOp)) {
            if (!visited[preOp] && preNodeTotal.find(preOp) == preNodeTotal.end()) {
                next.insert(preOp);
            }
        }
    }
    for (auto& nextOp : next) {
        preNodeTotal.insert(nextOp);
    }
    curr.swap(next);
}

int PriorDFSSort::GetNumUnvisitPreNode(Operation* op, std::map<Operation*, bool>& visited)
{
    std::unordered_set<Operation*> preNodeTotal;
    std::unordered_set<Operation*> curr;
    for (auto& preOp : state_.depManager.GetPredecessors(op)) {
        if (!visited[preOp]) {
            curr.insert(preOp);
            preNodeTotal.insert(preOp);
        }
    }
    while (!curr.empty()) {
        UpdatePreNodeQueue(curr, preNodeTotal, visited);
    }
    return preNodeTotal.size();
}

Operation* PriorDFSSort::FindNodeMinNumUnvisitedPreNode(std::map<Operation*, bool> visited,
                                                        const std::vector<Operation*>& outNodeQueue)
{
    Operation* res = nullptr;
    int minUnvisitedNode = INT_MAX;
    for (auto& outNode : outNodeQueue) {
        if (visited[outNode]) {
            continue;
        }
        int curUnvisitedNode = GetNumUnvisitPreNode(outNode, visited);
        if (curUnvisitedNode < minUnvisitedNode) {
            res = outNode;
            minUnvisitedNode = curUnvisitedNode;
        }
    }
    return res;
}

int PriorDFSSort::GetNodePriority(const std::unordered_map<Opcode, int>& preNodePriority, Operation* op)
{
    int prior = 10;
    auto it = preNodePriority.find(op->GetOpcode());
    if (it != preNodePriority.end()) {
        prior = it->second;
    }
    return prior;
}

int PriorDFSSort::GetMaxDepthSimple(Operation* op)
{
    auto it = depthCache_.find(op);
    if (it != depthCache_.end()) {
        return it->second;
    }

    int maxDepth = 0;
    for (const auto& pre : state_.depManager.GetPredecessors(op)) {
        maxDepth = std::max(maxDepth, GetMaxDepthSimple(pre));
    }

    int depth = maxDepth + 1;
    depthCache_[op] = depth;
    return depth;
}

void PriorDFSSort::QueueNotReadyPreNode(Operation* curOp, std::map<Operation*, bool>& visited,
                                        const std::unordered_map<Opcode, int>& preNodePriority,
                                        std::deque<Operation*>& queue)
{
    std::vector<Operation*> notReadyPreNode;
    for (auto& preOp : state_.depManager.GetPredecessors(curOp)) {
        if (!visited[preOp]) {
            notReadyPreNode.push_back(preOp);
        }
    }
    std::sort(notReadyPreNode.begin(), notReadyPreNode.end(), [&](Operation* a, Operation* b) {
        int priorA = GetNodePriority(preNodePriority, a);
        int priorB = GetNodePriority(preNodePriority, b);
        if (priorA != priorB) {
            return priorA < priorB;
        } else {
            int depA = GetMaxDepthSimple(a);
            int depB = GetMaxDepthSimple(b);
            if (depA == depB) {
                int aIdx = std::find(operations.begin(), operations.end(), a) - operations.begin();
                int bIdx = std::find(operations.begin(), operations.end(), b) - operations.begin();
                return aIdx < bIdx;
            }
            return depA < depB;
        }
    });
    for (auto& preOp : notReadyPreNode) {
        queue.push_front(preOp);
    }
}

void PriorDFSSort::ForwardDfs(Operation* curOp, std::vector<Operation*>& newOpList, std::map<Operation*, bool>& visited,
                              const std::unordered_map<Opcode, int>& preNodePriority, std::deque<Operation*>& queue)
{
    bool ready = true;
    for (auto& preOp : state_.depManager.GetPredecessors(curOp)) {
        if (!visited[preOp]) {
            ready = false;
            break;
        }
    }

    if (ready) {
        visited[curOp] = true;
        queue.pop_front();
        newOpList.push_back(curOp);
    } else {
        QueueNotReadyPreNode(curOp, visited, preNodePriority, queue);
    }
}

void PriorDFSSort::DFSFromSingleNode(Operation* op, std::map<Operation*, bool>& visited,
                                     std::vector<Operation*>& newOpList,
                                     const std::unordered_map<Opcode, int>& preNodePriority)
{
    if (visited[op]) {
        return;
    }

    std::deque<Operation*> queue = {op};
    while (!queue.empty()) {
        auto curOp = queue.front();
        if (visited[curOp]) {
            queue.pop_front();
            continue;
        }

        ForwardDfs(curOp, newOpList, visited, preNodePriority, queue);
    }
}

Status PriorDFSSort::DFSFromOutNode(const std::vector<Operation*>& outNodeQueue,
                                    const std::unordered_map<Opcode, int>& preNodePriority,
                                    std::map<Operation*, bool>& visited)
{
    std::vector<Operation*> newOpList;
    if (outNodeQueue.size() != 0) {
        DFSFromSingleNode(outNodeQueue[0], visited, newOpList, preNodePriority);
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "Subgraph must have operation with outdegree 0.");
        return FAILED;
    }

    for (size_t i = 1; i < outNodeQueue.size(); i++) {
        while (!visited[outNodeQueue[i]]) {
            auto node = FindNodeMinNumUnvisitedPreNode(visited, outNodeQueue);
            if (node == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "FindNodeMinNumUnvisitedPreNode failed.");
                return FAILED;
            }
            DFSFromSingleNode(node, visited, newOpList, preNodePriority);
        }
    }
    operations = newOpList;
    return SUCCESS;
}

Status PriorDFSSort::PriorDFS(const std::unordered_map<Opcode, int>& preNodePriority)
{
    std::map<Operation*, bool> visited;
    std::vector<Operation*> outNodeQueue;
    depthCache_.clear();
    for (size_t i = 0; i < operations.size(); i++) {
        visited[operations[i]] = false;
        if (state_.depManager.GetSuccessors(operations[i]).empty()) {
            outNodeQueue.emplace_back(operations[i]);
        }
    }

    SortOutNodeQueue(outNodeQueue);

    if (DFSFromOutNode(outNodeQueue, preNodePriority, visited) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "DFSFromOutNode failed.");
        return FAILED;
    }
    return SUCCESS;
}

void PriorDFSSort::SortOutNodeQueue(std::vector<Operation*>& outNodeQueue)
{
    // 全量反向回溯每个出口节点到所有入度=0 的源头，用源头集合做 groupId。
    // 同源的出口节点归为一组，不同源的各自成组；组间按最大深度降序，组内按深度降序。
    // 递归 DFS 后序 + memoization，每个节点的源点集只算一次，整体 O(V+E)。
    // ALLOC 类操作不参与同源判断（它们是内存分配，不是数据依赖）。
    std::unordered_map<Operation*, std::set<Operation*>> sourceSetCache;
    std::function<std::set<Operation*>(Operation*)> collectSources = [&](Operation* op) -> std::set<Operation*> {
        auto it = sourceSetCache.find(op);
        if (it != sourceSetCache.end()) {
            return it->second;
        }
        const auto& preds = state_.depManager.GetPredecessors(op);
        std::vector<Operation*> dataPreds;
        for (auto* pred : preds) {
            if (!IsAllocOpCode(pred->GetOpcode())) {
                dataPreds.push_back(pred);
            }
        }
        if (dataPreds.empty()) {
            sourceSetCache[op] = {op};
            return sourceSetCache[op];
        }
        std::set<Operation*> sources;
        for (auto* pred : dataPreds) {
            auto predSources = collectSources(pred);
            sources.insert(predSources.begin(), predSources.end());
        }
        sourceSetCache[op] = sources;
        return sources;
    };

    std::unordered_map<Operation*, std::set<Operation*>> outNodeSources;
    std::map<std::set<Operation*>, int> groupMaxDepth;
    for (auto* outNode : outNodeQueue) {
        auto sources = collectSources(outNode);
        outNodeSources[outNode] = sources;
        groupMaxDepth[sources] = std::max(groupMaxDepth[sources], GetMaxDepthSimple(outNode));
    }

    // 组间按组内最大深度降序，组内按深度降序（长链先走）
    std::stable_sort(outNodeQueue.begin(), outNodeQueue.end(), [&](Operation* a, Operation* b) {
        const auto& sa = outNodeSources[a];
        const auto& sb = outNodeSources[b];
        if (sa != sb) {
            return groupMaxDepth[sa] > groupMaxDepth[sb];
        }
        return GetMaxDepthSimple(a) > GetMaxDepthSimple(b);
    });
}

Status PriorDFSSort::DoSortOps()
{
    AllocAhead();
    std::unordered_map<Opcode, int> preNodePriority = {
        {Opcode::OP_UB_ALLOC, 0},
        {Opcode::OP_L1_ALLOC, 0},
        {Opcode::OP_L0A_ALLOC, 0},
        {Opcode::OP_L0B_ALLOC, 0},
        {Opcode::OP_L0C_ALLOC, 0},
        {Opcode::OP_BT_ALLOC, 0},
        {Opcode::OP_FIX_ALLOC, 0},
        {Opcode::OP_L1_TO_L0A, 1},
        {Opcode::OP_L1_TO_L0B, 1},
        {Opcode::OP_L1_TO_L0_AT, 1},
        {Opcode::OP_L1_TO_L0_BT, 1},
        {Opcode::OP_L1_TO_FIX, 1},
        {Opcode::OP_L1_TO_FIX_QUANT_PRE, 1},
        {Opcode::OP_L1_TO_FIX_RELU_PRE, 1},
        {Opcode::OP_L1_TO_FIX_RELU_POST, 1},
        {Opcode::OP_L1_TO_FIX_QUANT_POST, 1},
        {Opcode::OP_L1_TO_FIX_ELT_ANTIQ, 1},
        {Opcode::OP_L1_TO_FIX_MTE2_ANTIQ, 1},
        {Opcode::OP_L1_TO_BT, 1},
        {Opcode::OP_COPY_IN, 2},
        {Opcode::OP_UB_COPY_IN, 2},
        {Opcode::OP_L1_COPY_IN, 2},
        {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2},
        {Opcode::OP_L1_COPY_UB, 2},
        {Opcode::OP_L0C_COPY_UB, 2},
        {Opcode::OP_UB_COPY_L1, 2},
    };
    if (PriorDFS(preNodePriority) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "PriorDFS failed.");
        return FAILED;
    }
    PromoteOps();
    if (ExecuteOp() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ExecuteOp failed.");
        return FAILED;
    }
    return SUCCESS;
}

PromotePriority PriorDFSSort::ClassifyPromoteOp(Operation* op) const
{
    if (!op) {
        return PromotePriority::Normal;
    }
    if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
        return PromotePriority::Assemble;
    }
    if (IsSkipOp(*op)) {
        return PromotePriority::Skip;
    }
    if (OpcodeManager::Inst().IsCopyOut(op->GetOpcode()) &&
        op->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        return PromotePriority::DdrCopyOut;
    }
    return PromotePriority::Normal;
}

struct PromoteCmp {
    const std::unordered_map<Operation*, size_t>& pos;
    const std::unordered_map<Operation*, PromotePriority>& cls;

    bool operator()(Operation* a, Operation* b) const
    {
        if (cls.at(a) != cls.at(b))
            return cls.at(a) > cls.at(b);

        return pos.at(a) > pos.at(b);
    }
};

void PriorDFSSort::PromoteOps()
{
    if (operations.empty())
        return;

    std::unordered_map<Operation*, int> indegree;
    std::unordered_map<Operation*, size_t> pos;
    std::unordered_map<Operation*, PromotePriority> cls;

    indegree.reserve(operations.size());
    pos.reserve(operations.size());
    cls.reserve(operations.size());

    for (size_t i = 0; i < operations.size(); ++i) {
        auto* op = operations[i];
        pos[op] = i;
        cls[op] = ClassifyPromoteOp(op);
        indegree[op] = state_.depManager.HasOp(op) ? state_.depManager.GetPredecessors(op).size() : 0;
    }

    std::priority_queue<Operation*, std::vector<Operation*>, PromoteCmp> ready(PromoteCmp{pos, cls});

    for (auto* op : operations) {
        if (indegree[op] == 0)
            ready.push(op);
    }

    std::vector<Operation*> reordered;
    reordered.reserve(operations.size());

    while (!ready.empty()) {
        auto* cur = ready.top();
        ready.pop();
        reordered.push_back(cur);
        if (!state_.depManager.HasOp(cur))
            continue;

        for (auto* succ : state_.depManager.GetSuccessors(cur)) {
            if (--indegree[succ] == 0)
                ready.push(succ);
        }
    }

    if (reordered.size() == operations.size())
        operations.swap(reordered);
}

} // namespace npu::tile_fwk
