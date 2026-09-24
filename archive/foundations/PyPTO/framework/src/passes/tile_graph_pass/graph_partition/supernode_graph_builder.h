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
 * \file supernode_graph_builder.h
 * \brief
 */

#ifndef SUPERNODE_GRAPH_BUILDER_H
#define SUPERNODE_GRAPH_BUILDER_H
#include <unordered_set>
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {
class OperationGraphInfo {
public:
    uint64_t GetHash(const Operation* op) const;
    bool CoreTypeMergeable(const std::set<OpCoreType>& coreTypes) const;
    std::vector<int32_t> GetSameLevelOpIdx(int32_t opIdx, Opcode opLabel) const;
    std::vector<Operation*> opList_;
    std::unordered_map<int32_t, int32_t> magic2Idx_;
    std::vector<std::vector<int32_t>> inGraph_;
    std::vector<std::vector<int32_t>> outGraph_;
    std::vector<uint64_t> opHashList_;
    std::vector<OpCoreType> opCoreType_;
    bool useCVMixPartition_ = false;
};

class NodeGraphInfo {
public:
    Status Build(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                 const std::vector<std::pair<int32_t, int32_t>>& mergePair, bool markIsCube);
    Status AvoidLoop(const std::shared_ptr<OperationGraphInfo> operationGraphInfo, std::vector<int32_t>& parent,
                     std::vector<std::vector<int32_t>>& node2Op, bool& updated);
    Status BuildInOutGraph(const std::shared_ptr<OperationGraphInfo> operationGraphInfo);
    void SetNodeCoreTypeAndMergeable(const std::shared_ptr<OperationGraphInfo> operationGraphInfo, bool markIsCube);
    void BuildNodeMapping(const std::shared_ptr<OperationGraphInfo> operationGraphInfo);
    bool CheckScopeNotMergeable(Operation& op);
    bool CheckUbToUbWithDynOffset(Operation& op);
    bool CheckViewAssembleOffset(Operation& op);
    Status MergeSrcToDstIsland(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                               std::vector<int32_t>& parent, int32_t src, int32_t dst);
    int32_t GetNodeCycle(int32_t nodeIdx) const;
    bool GetNodeMergeable(const std::shared_ptr<OperationGraphInfo> operationGraphInfo, int32_t nodeIdx);
    std::vector<std::vector<int32_t>> node2Op_;
    std::vector<Operation::ScopeInfo> nodeScope_;
    std::vector<int32_t> op2Node_;
    std::vector<std::vector<int32_t>> nodeInGraphList_;
    std::vector<std::vector<int32_t>> nodeOutGraphList_;
    std::vector<OpCoreType> nodeCoreType_;
    std::vector<int32_t> nodeCycles_;
    std::vector<bool> nodeMergeable_;
    std::vector<uint64_t> nodeHashList_;
    std::unordered_map<uint64_t, std::vector<int32_t>> hash2NodeMap_;
};

class SuperNodeGraphBuilder {
public:
    SuperNodeGraphBuilder() = default;
    SuperNodeGraphBuilder(bool useCVMixPartition) : useCVMixPartition_(useCVMixPartition) {};
    virtual ~SuperNodeGraphBuilder() = default;

protected:
    Status BuildOpGraph(const std::vector<Operation*>& opList);
    virtual Status BuildSuperNodeGraph();
    /*!
     * \brief 按 scope 对 SuperNode 合并。
     *
     * - CVMix 场景：scope 内存在 AIC+AIV 混合时，分配唯一 cvFuseId，供下游融合调度使用；
     * - 非 CVMix 场景：allowParallelMerge 控制同 scope 下 node 是全量合并还是仅合并
     *   有直连边的相邻 node。
     */
    Status ProcessScopeMerge();

    struct ScopeCoreTypeInfo {
        bool hasCube = false;
        bool hasVector = false;
    };
    struct ScopeCollectResult {
        std::map<int32_t, ScopeCoreTypeInfo> scopeCoreTypes;
        std::map<int32_t, bool> scopeAllowParallel;
        std::map<int32_t, std::vector<int32_t>> scope2Nodes;
    };
    ScopeCollectResult CollectScopeInfo(int32_t numNodes);
    Status ValidateScopeCoreTypes(int32_t scopeId, const ScopeCoreTypeInfo& coreTypeInfo, bool isCVMix,
                                  std::map<int32_t, int32_t>& scopeToCvFuseId);
    Status CheckAndMergeScopes(const ScopeCollectResult& scopeInfo, std::vector<int32_t>& snParent, bool& needRebuild,
                               std::map<int32_t, int32_t>& scopeToCvFuseId);
    void MergeScopeNodesParallel(const std::vector<int32_t>& nodes, int32_t scopeId, std::vector<int32_t>& snParent,
                                 bool& needRebuild);
    void MergeScopeNodesSequential(const std::vector<int32_t>& nodes, int32_t scopeId, std::vector<int32_t>& snParent,
                                   bool& needRebuild);
    Status RebuildSuperNodes(std::vector<int32_t>& snParent, int32_t numNodes);
    void ApplyCvFuseIds(const std::map<int32_t, int32_t>& scopeToCvFuseId);

    virtual Status BuildHashValues();

    // BuildSuperNodeGraph helpers
    bool L1CopyInCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList,
                         int32_t i, std::vector<std::pair<int32_t, int32_t>>& mergePair);
    bool ConvertCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList,
                        int32_t i, std::vector<std::pair<int32_t, int32_t>>& mergePair);
    bool AssembleCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList,
                         int32_t i, std::vector<std::pair<int32_t, int32_t>>& mergePair);
    bool CopyOutCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList,
                        int32_t i, std::vector<std::pair<int32_t, int32_t>>& mergePair, bool assembleScene);
    bool CopyInCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList,
                       int32_t i, std::vector<std::pair<int32_t, int32_t>>& mergePair);
    bool MulAccCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList,
                       int32_t i, std::vector<std::pair<int32_t, int32_t>>& mergePair);
    bool AssembleToCopyoutScene(Operation* op);
    bool ExpandCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList,
                       int32_t i, std::vector<std::pair<int32_t, int32_t>>& mergePair);

    // BuildHashValues helpers
    virtual uint64_t CombineHash(const uint64_t h1, const uint64_t h2) const;
    std::vector<std::pair<int32_t, int32_t>> GetReduceNodeMergePair() const;
    Status BuildReduceNodeHash(std::shared_ptr<NodeGraphInfo> reduceNodeInfo);
    Status BuildBalanceOpHash(std::vector<uint64_t>& opHashList);
    void ComputeDirectionalNodeHash(std::shared_ptr<NodeGraphInfo> reduceNodeInfo,
                                    const std::vector<uint64_t>& reduceNodeHashList, std::vector<uint64_t>& hashList,
                                    bool reverse);

    // Parameters
    bool useReduceBalanceHash_ = true;
    bool useCVMixPartition_ = false;
    int nextCvFuseId_ = 0;

    // Data
    std::shared_ptr<OperationGraphInfo> operationInfo_;
    std::shared_ptr<NodeGraphInfo> superNodeInfo_;
};
static constexpr int DEFAULT_SCOPE_ID = -1;
} // namespace npu::tile_fwk
#endif // SUPERNODE_GRAPH_BUILDER_H
