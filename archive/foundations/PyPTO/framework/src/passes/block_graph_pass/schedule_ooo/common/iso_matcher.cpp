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
 * \file iso_matcher.cpp
 * \brief Implementation of IsoMatchChains and FindTaskEntryOps.
 */

#include "passes/block_graph_pass/schedule_ooo/common/iso_matcher.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface/operation/operation.h"
#include "interface/operation/opcode.h"
#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "IsoMatcher"

namespace npu::tile_fwk {

namespace {

// 签名由 opcode、输出数量、每个输出 tensor 的 MemoryType 和 shape 组成。
// alloc 节点额外追加 (基于唯一输出 tensor 的 GetProducers/GetConsumers):
//   || prod:<count> | <prodOpcode>:<outIdx> | ...   (生产者: opcode + tensor 是生产者第几输出)
//   || cons:<count> | <consOpcode>:<inIdx> | ...     (消费者: opcode + tensor 是消费者第几输入)
// 非 alloc 节点额外追加:
//   || pred:<count> | <predOpcode> | ...   (前驱 opcode 多集)
//   || succ:<count> | <succOpcode> | ...   (后继 opcode 多集)
void AppendOutputTensorSig(const Operation* op, std::ostringstream& oss)
{
    for (const auto& out : op->GetOOperands()) {
        oss << '|';
        if (out == nullptr) {
            oss << "null";
            continue;
        }
        oss << static_cast<int>(out->GetMemoryTypeOriginal()) << ':';
        for (auto dim : out->GetShape()) {
            oss << dim << ',';
        }
    }
}

void AppendSortedOpcodes(const std::string& tag, const std::vector<int>& opcodes, std::ostringstream& oss)
{
    auto sortedOpcodes = opcodes;
    std::sort(sortedOpcodes.begin(), sortedOpcodes.end());
    oss << tag << sortedOpcodes.size();
    for (int code : sortedOpcodes) {
        oss << '|' << code;
    }
}

void AppendNonAllocNeighborSig(const Operation* op, std::ostringstream& oss)
{
    std::vector<int> predOpcodes;
    for (Operation* pred : op->ProducerOps()) {
        if (pred == nullptr)
            continue;
        predOpcodes.push_back(static_cast<int>(pred->GetOpcode()));
    }
    AppendSortedOpcodes("||pred:", predOpcodes, oss);

    std::vector<int> succOpcodes;
    for (Operation* succ : op->ConsumerOps()) {
        if (succ == nullptr)
            continue;
        succOpcodes.push_back(static_cast<int>(succ->GetOpcode()));
    }
    AppendSortedOpcodes("||succ:", succOpcodes, oss);
}

std::vector<std::string> BuildProducerEntries(const LogicalTensorPtr& tensor)
{
    std::vector<std::string> entries;
    for (Operation* prod : tensor->GetProducers()) {
        if (prod == nullptr)
            continue;
        const auto& prodOutputs = prod->GetOOperands();
        for (size_t j = 0; j < prodOutputs.size(); ++j) {
            if (prodOutputs[j] == tensor) {
                entries.push_back(std::to_string(static_cast<int>(prod->GetOpcode())) + ":" + std::to_string(j));
                break;
            }
        }
    }
    std::sort(entries.begin(), entries.end());
    return entries;
}

std::vector<std::string> BuildConsumerEntries(const LogicalTensorPtr& tensor)
{
    std::vector<std::string> entries;
    for (Operation* cons : tensor->GetConsumers()) {
        if (cons == nullptr)
            continue;
        const auto& consInputs = cons->GetIOperands();
        for (size_t i = 0; i < consInputs.size(); ++i) {
            if (consInputs[i] == tensor) {
                entries.push_back(std::to_string(static_cast<int>(cons->GetOpcode())) + ":" + std::to_string(i));
                break;
            }
        }
    }
    std::sort(entries.begin(), entries.end());
    return entries;
}

void AppendEntries(const std::string& tag, const std::vector<std::string>& entries, std::ostringstream& oss)
{
    oss << tag << entries.size();
    for (const auto& entry : entries) {
        oss << '|' << entry;
    }
}

void AppendAllocTensorRelationSig(const Operation* op, std::ostringstream& oss)
{
    const auto& outputs = op->GetOOperands();
    if (outputs.empty() || outputs[0] == nullptr) {
        return;
    }
    const auto& tensor = outputs[0];
    AppendEntries("||prod:", BuildProducerEntries(tensor), oss);
    AppendEntries("||cons:", BuildConsumerEntries(tensor), oss);
}

std::string SigOf(const Operation* op)
{
    if (op == nullptr)
        return std::string();
    std::ostringstream oss;
    oss << static_cast<int>(op->GetOpcode()) << '|' << op->GetOOperands().size();
    AppendOutputTensorSig(op, oss);
    if (IsAllocOpCode(op->GetOpcode())) {
        AppendAllocTensorRelationSig(op, oss);
    } else {
        AppendNonAllocNeighborSig(op, oss);
    }
    return oss.str();
}

// Append direct data-dependency successors (ConsumerOps) to `out`, skipping
// visited, alloc, and any successor outside the task boundary (taskOps).
void CollectSuccessors(Operation* op, std::unordered_set<int>& visited, const std::unordered_set<Operation*>& taskOps,
                       std::vector<Operation*>& out)
{
    if (op == nullptr)
        return;
    for (Operation* succ : op->ConsumerOps()) {
        if (succ == nullptr)
            continue;
        if (taskOps.find(succ) == taskOps.end())
            continue; // stop at task boundary
        int magic = succ->GetOpMagic();
        if (visited.count(magic) > 0)
            continue;
        visited.insert(magic);
        out.push_back(succ);
    }
}

bool CheckRootSignatures(const std::vector<Operation*>& rootsA, const std::vector<Operation*>& rootsB)
{
    std::vector<std::string> sigA;
    std::vector<std::string> sigB;
    for (auto* op : rootsA)
        sigA.push_back(SigOf(op));
    for (auto* op : rootsB)
        sigB.push_back(SigOf(op));
    std::sort(sigA.begin(), sigA.end());
    std::sort(sigB.begin(), sigB.end());
    if (sigA.size() != sigB.size())
        return false;
    for (size_t i = 0; i < sigA.size(); i++) {
        if (sigA[i] != sigB[i])
            return false;
    }
    return true;
}

using DepthSigGroups = std::map<int, std::unordered_map<std::string, std::vector<Operation*>>>;

DepthSigGroups BuildDepthSigGroups(const std::vector<Operation*>& roots, const std::unordered_set<Operation*>& taskOps)
{
    std::unordered_set<int> visited;
    for (auto* root : roots)
        visited.insert(root->GetOpMagic());

    DepthSigGroups depthSigGroups;
    std::vector<Operation*> cur = roots;
    int depth = 0;
    while (!cur.empty()) {
        std::vector<Operation*> next;
        for (Operation* op : cur) {
            depthSigGroups[depth][SigOf(op)].push_back(op);
            CollectSuccessors(op, visited, taskOps, next);
        }
        cur = std::move(next);
        ++depth;
    }
    return depthSigGroups;
}

void MatchIsoDepth(int depth, const std::vector<Operation*>& curA, const DepthSigGroups& depthSigGroupsB,
                   const std::unordered_set<Operation*>& taskOpsA, std::unordered_set<int>& visitedA,
                   std::unordered_set<int>& pairedB, std::vector<Operation*>& nextA, IsoMatchResult& res)
{
    auto byMagic = [](Operation* a, Operation* b) { return a->GetOpMagic() < b->GetOpMagic(); };
    auto layerIt = depthSigGroupsB.find(depth);
    if (layerIt == depthSigGroupsB.end())
        return;
    auto& groupB = layerIt->second;
    for (Operation* a : curA) {
        auto it = groupB.find(SigOf(a));
        if (it == groupB.end())
            continue;
        std::vector<Operation*> cands;
        for (Operation* cand : it->second) {
            if (pairedB.count(cand->GetOpMagic()) == 0)
                cands.push_back(cand);
        }
        if (cands.empty())
            continue;
        std::sort(cands.begin(), cands.end(), byMagic);

        Operation* chosen = cands[0];
        pairedB.insert(chosen->GetOpMagic());
        if (IsAllocOpCode(a->GetOpcode())) {
            res.allocPairs.push_back({a, chosen, depth});
        } else {
            res.pairs.push_back({a, chosen, depth});
            if (depth > res.maxMatchedDepth)
                res.maxMatchedDepth = depth;
        }

        if (cands.size() > 1) {
            res.truncatedCount++;
            APASS_LOG_INFO_F(Elements::Operation, "IsoMatch truncated A-side depth=%d opmagic=%d opcode=%d sig=[%s]",
                             depth, a->GetOpMagic(), static_cast<int>(a->GetOpcode()), SigOf(a).c_str());
            for (size_t ci = 0; ci < cands.size(); ++ci) {
                Operation* cand = cands[ci];
                APASS_LOG_INFO_F(Elements::Operation, "IsoMatch truncated B-cand[%zu] opmagic=%d opcode=%d sig=[%s]",
                                 ci, cand->GetOpMagic(), static_cast<int>(cand->GetOpcode()), SigOf(cand).c_str());
            }
        } else {
            CollectSuccessors(a, visitedA, taskOpsA, nextA);
        }
    }
}

} // namespace

std::vector<Operation*> FindTaskEntryOps(const std::vector<Operation*>& taskOps,
                                         const std::unordered_set<Operation*>& taskSet)
{
    std::vector<Operation*> entries;
    for (Operation* op : taskOps) {
        if (op == nullptr)
            continue;
        bool hasInTaskProducer = false;
        for (Operation* prod : op->ProducerOps()) {
            if (prod != nullptr && taskSet.count(prod) > 0) {
                hasInTaskProducer = true;
                break;
            }
        }
        if (!hasInTaskProducer) {
            entries.push_back(op);
        }
    }
    return entries;
}

IsoMatchResult IsoMatchChains(const std::vector<Operation*>& rootsA, const std::vector<Operation*>& rootsB,
                              const std::unordered_set<Operation*>& taskOpsA,
                              const std::unordered_set<Operation*>& taskOpsB)
{
    IsoMatchResult res;
    if (rootsA.empty() || rootsB.empty())
        return res;

    auto byMagic = [](Operation* a, Operation* b) { return a->GetOpMagic() < b->GetOpMagic(); };

    if (!CheckRootSignatures(rootsA, rootsB))
        return res;

    res.rootIsomorphic = true;

    // BFS 前先标记全部入口点，避免被后继重复加入。
    std::unordered_set<int> visitedA;
    for (auto* a : rootsA)
        visitedA.insert(a->GetOpMagic());

    auto depthSigGroupsB = BuildDepthSigGroups(rootsB, taskOpsB);

    // === 阶段 2: A 侧 BFS, depth=d 的 A 节点只在 B 侧 depth=d 的节点中找同构候选 ===
    std::unordered_set<int> pairedB; // 全局已配对 B 节点 opmagic
    std::vector<Operation*> curA = rootsA;
    int depth = 0;
    while (!curA.empty()) {
        std::sort(curA.begin(), curA.end(), byMagic);
        std::vector<Operation*> nextA;

        MatchIsoDepth(depth, curA, depthSigGroupsB, taskOpsA, visitedA, pairedB, nextA, res);
        curA = std::move(nextA);
        ++depth;
    }
    return res;
}

} // namespace npu::tile_fwk
