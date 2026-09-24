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
 * \file iso_partitioner.h
 * \brief
 */

#ifndef PASS_ISO_PARTITIONER_H
#define PASS_ISO_PARTITIONER_H
#include <unordered_set>
#include "supernode_graph_builder.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {

struct IntPairHash {
    std::size_t operator()(const std::pair<int32_t, int32_t>& p) const
    {
        auto h1 = std::hash<int32_t>{}(p.first);
        auto h2 = std::hash<int32_t>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

constexpr int64_t LATENCY_THRESHOLD_LEVEL1 = 10000000;
constexpr int64_t LATENCY_THRESHOLD_LEVEL2 = 290000;
constexpr int64_t LATENCY_THRESHOLD_LEVEL3 = 50000;

constexpr int32_t CYCLE_UB_LEVEL1 = 5000000; // 超大型计算场景
constexpr int32_t CYCLE_UB_LEVEL2 = 150000;  // 大型计算场景
constexpr int32_t CYCLE_UB_LEVEL3 = 10000;   // 中型计算场景（默认值）
constexpr int32_t CYCLE_UB_LEVEL4 = 1536;    // 小型计算场景

enum class GraphExtendResult { EXTEND_SUCCESS, EXTEND_LINK_EXHAUST, EXTEND_NODE_EXHAUST };

class SubGraph {
public:
    SubGraph(std::shared_ptr<OperationGraphInfo> operationInfo, std::shared_ptr<NodeGraphInfo> superNodeInfo)
        : operationInfo_(operationInfo), superNodeInfo_(superNodeInfo)
    {}
    int32_t GetExpandCandidate(size_t expandNodeIdx, size_t expandLinkIdx, GraphExtendResult& res);
    void AddNode(int32_t nodeIdx);
    void Merge(SubGraph& sg);
    bool HasNode(int32_t nodeIdx) const;
    void BuildInOutSet();
    int32_t GetLatency() const;
    void Clear();
    std::string DumpStr();
    const std::vector<int32_t>& GetNodeList();
    std::vector<Operation*> GetOpList();
    void SetAllowCrossScopeMerge(bool allow) { allowCrossScopeMerge_ = allow; }
    bool GetAllowCrossScopeMerge() const { return allowCrossScopeMerge_; }

    std::shared_ptr<OperationGraphInfo> operationInfo_;
    std::shared_ptr<NodeGraphInfo> superNodeInfo_;
    std::vector<int32_t> nodeList_;
    std::unordered_set<int32_t> nodeSet_;
    std::vector<int32_t> inNodes_;
    std::vector<int32_t> outNodes_;
    std::unordered_set<std::pair<int32_t, int32_t>, IntPairHash> mergeHistoryIsoSub_;
    int32_t cycle_{0};
    OpCoreType coreType_{OpCoreType::ANY};
    bool mergeable_{true};
    int32_t scopeId_{-1};
    bool allowCrossScopeMerge_{false};
};

class IsomorphismGraphGroup {
public:
    Status BuildGraphGroup(std::shared_ptr<OperationGraphInfo> operationInfo,
                           std::shared_ptr<NodeGraphInfo> superNodeInfo, std::vector<int32_t>& expandCandidate,
                           std::unordered_set<int32_t>& currentNodeSet, std::vector<int32_t>& idxInLinkNum,
                           std::deque<int32_t>& zeroInQueue);
    Status ExpandIsoGraphs(std::unordered_set<int32_t>& currentNodeSet, std::vector<int32_t>& idxInLinkNum,
                           std::deque<int32_t>& zeroInQueue, int32_t pgUpperBound);
    static bool IsoGraphMerge(std::shared_ptr<IsomorphismGraphGroup>& currGraph,
                              std::shared_ptr<IsomorphismGraphGroup>& mergeGraph,
                              std::vector<std::pair<int32_t, int32_t>>& isoSubIdxs);
    size_t Size() const;
    void Clear();
    bool GetMergeable();
    int32_t GetLatency() const;
    Status InLinkCountDelete(int32_t nodeIdx, std::vector<int32_t>& idxInLinkNum, std::deque<int32_t>& zeroInQueue);
    bool IsLegalIsoGraphExtender(std::vector<int32_t>& expandCandidate, std::unordered_set<int32_t>& currentNodeSet,
                                 std::vector<int32_t>& idxInLinkNum, int32_t pgUpperBound);
    bool IsLegalSubGraphMerge(SubGraph* sg1, SubGraph* sg2);
    std::shared_ptr<SubGraph> GetSubGraph(int32_t idx);
    std::vector<std::shared_ptr<SubGraph>> isoGraphs_;
    std::unordered_set<int32_t> subVisitedNodeSet_;
    bool mergeable_;
    std::shared_ptr<OperationGraphInfo> operationInfo_;
    std::shared_ptr<NodeGraphInfo> superNodeInfo_;
};

class IsoPartitioner : public SuperNodeGraphBuilder {
public:
    Status PartitionGraph(Function& function);
    Status SetParameter(int32_t parallelNum, int32_t pgLowerBound, bool useReduceBalanceHash = true);

private:
    Status BuildIsomorphismGroups();
    Status IsomorphismGroupMergeStep(bool nonIsoGraphsMerge);
    Status IsomorphismGroupMergeProcess(bool nonIsoGraphsMerge);
    Status UpdatePartitionResult(Function& function);
    Status EstimateCycleUB(Function& function);
    Status RunPartitionSteps(Function& function);
    bool HandleLiteNPU(Function& function);
    Status IsomorphismGroupMergePrepare(std::vector<std::pair<int32_t, int32_t>>& isoSubIdxs,
                                        std::vector<std::unordered_set<int32_t>>& isoInGraph,
                                        std::vector<std::unordered_set<int32_t>>& isoOutGraph,
                                        std::vector<std::vector<int32_t>>& isoNodeList,
                                        std::vector<int32_t>& isoIdx2color);
    std::vector<int32_t> GetCandidateMergeColors(int32_t currColor,
                                                 std::vector<std::unordered_set<int32_t>>& isoInGraph,
                                                 std::vector<std::unordered_set<int32_t>>& isoOutGraph,
                                                 std::vector<std::vector<int32_t>>& isoNodeList,
                                                 std::vector<int32_t>& isoIdx2color, bool nonIsoGraphsMerge);
    bool SuitableForMergeCheck(int32_t currColor, int32_t mergeColor, bool nonIsoGraphsMerge) const;
    bool CanMergeScopes(int32_t currColor, int32_t mergeColor) const;
    int32_t CalculateMergedLatency(int32_t currColor, int32_t mergeColor) const;
    bool CheckIsoMergeConditions(int32_t currColorSize, int32_t mergeColorSize) const;
    std::vector<std::shared_ptr<IsomorphismGraphGroup>> isoSubGroups_;
    int32_t tryMergeLoopNum_ = 100;
    int32_t cycleUB_ = -1;
    int32_t parallelNum_ = -1;
    int32_t cycleLB_ = -1;
};

} // namespace npu::tile_fwk
#endif // PASS_ISO_PARTITIONER_H
