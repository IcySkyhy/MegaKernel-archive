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
 * \file task_splitter.h
 * \brief TaskSplitter and helper classes for DAG task graph splitting, clustering, and merging.
 */

#ifndef PASS_TASK_SPLITTER_H
#define PASS_TASK_SPLITTER_H

#include "core_assign.h"

namespace npu::tile_fwk {

// OoO 碎子图合并阈值：cycle 低于此值的 vector cluster 视为碎子图
constexpr int OOO_CYCLE_LB = 512;
// 碎子图合并最大遍历轮次
constexpr int OOO_SMALL_CLUSTER_MERGE_MAX_ITER = 3;

// 并查集
class DSUWithOrder {
public:
    DSUWithOrder(int num);
    int Find(int i);
    void Union(int i, int j);
    std::vector<int> parent;
};

// 用于进行子图切分，任务排布和internalSubgraphId写回
class TaskSplitter {
public:
    void SplitGraph(const std::vector<Operation*>& opList); // 此处opList必须符合拓扑序
    void BuildOpGraph();
    void BuildInOutGraph(std::vector<std::set<int>>& inGraph, std::vector<std::set<int>>& outGraph,
                         std::vector<int>& clusterIds, int clusterNum);
    TaskGraph BuildTaskGraph();
    void BuildSameLayerConnectionWithBack();
    void BuildSameLayerConnectionWithFront();
    void UnionSameCoreOps(DSUWithOrder& dsu);
    void UnionSameLayerConnections(DSUWithOrder& dsu);
    void UnionCrossCoreAICToAIV(DSUWithOrder& dsu);
    void UnionL0CToL1CopyIn(DSUWithOrder& dsu);
    void UnionCombineOps(DSUWithOrder& dsu);
    void UnionVecClustersByDep(DSUWithOrder& dsu);
    void MergeSmallVecClusters(DSUWithOrder& dsu);
    void UnionCubeClustersByDep(DSUWithOrder& dsu);
    void UnionByOooScope(DSUWithOrder& dsu);
    void PropagateOooScopeToReshape();
    std::unordered_set<int> CollectOooScopeProtectedClusters(DSUWithOrder& dsu);
    std::unordered_set<int> CollectProtectedClusterIds(const std::vector<int>& opCluster) const;
    int BuildCluster(std::vector<int>& clusterIds, std::vector<ScheduleCoreType>& clusterCoreTypes);
    void ReverseDFSFindByOutputMemType(int opIdx, MemoryType targetMemType, std::vector<int>& result,
                                       std::vector<bool>& visited);
    std::vector<std::vector<int>> FindMergeableTaskNodes();
    void MergeTask();
    void MergeTaskByTargetCoreType();
    Status MarkInternalSubgraphID();
    Status SetAllocInternalSubgraphID(Operation* op);
    void CombineSCC(std::vector<int>& clusterIds, std::vector<ScheduleCoreType>& clusterCoreTypes,
                    std::vector<std::set<int>>& inGraph, std::vector<std::set<int>>& outGraph,
                    std::vector<std::vector<int>>& sccResult);
    void RecordIDMap(std::unordered_map<int, int>& oldClusterToNewCluster,
                     std::vector<ScheduleCoreType>& clusterCoreTypes);
    TaskGraph& GetTaskGraph() { return taskGraph_; }
    std::vector<Operation*> GetMergedOperations();

    // BuildCluster 辅助函数
    inline bool IsL1MultiConsumerSkip(size_t idx) const;
    inline bool IsL1ToL0Op(int opIdx) const;
    int CollectClusters(DSUWithOrder& dsu, std::unordered_map<int, int>& rootToCluster,
                        std::vector<ScheduleCoreType>& coreTypes) const;
    void GetOpClusterIds(DSUWithOrder& dsu, const std::unordered_map<int, int>& rootToCluster,
                         std::vector<int>& opCluster) const;
    void BuildClusterDepGraph(const std::vector<int>& opCluster, int clusterNum,
                              std::vector<std::set<int>>& clusterIn) const;
    void BuildClusterInOutGraph(const std::vector<std::set<int>>& clusterIn, int clusterNum,
                                std::vector<std::set<int>>& clusterOut) const;
    void GroupAIVClustersByDep(const std::vector<std::set<int>>& clusterIn,
                               const std::vector<std::set<int>>& clusterOut,
                               const std::vector<ScheduleCoreType>& tmpCoreTypes, int clusterNum,
                               std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>>& vecGroups) const;
    void MergeVecClusterGroup(DSUWithOrder& dsu, const std::vector<int>& opCluster,
                              const std::vector<int>& vecClusters);
    void GroupAICClustersByDep(const std::vector<std::set<int>>& clusterIn,
                               const std::vector<std::set<int>>& clusterOut,
                               const std::vector<ScheduleCoreType>& tmpCoreTypes, int clusterNum,
                               std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>>& cubeGroups) const;
    void MergeCubeClusterGroup(DSUWithOrder& dsu, const std::vector<int>& opCluster,
                               const std::vector<int>& cubeClusters);
    void BuildClusterGraph(const std::vector<int>& opCluster, int clusterNum, std::vector<std::set<int>>& clIn,
                           std::vector<std::set<int>>& clOut) const;
    void EstimateClusterCycles(const std::vector<int>& opCluster, int clusterNum, std::vector<int>& clusterCycle) const;
    std::vector<int> TopoSortClusters(int clusterNum, const std::vector<std::set<int>>& clIn,
                                      const std::vector<std::set<int>>& clOut) const;
    int FindMergeTarget(int clusterId, const std::vector<ScheduleCoreType>& coreTypes, const std::vector<int>& cycle,
                        const std::vector<std::set<int>>& clIn, const std::vector<std::set<int>>& clOut) const;
    void UnionTwoClusters(DSUWithOrder& dsu, const std::vector<int>& opCluster, int srcCluster, int dstCluster);
    void UpdateClusterGraphAfterMerge(int src, int dst, std::vector<int>& cycle, std::vector<std::set<int>>& clIn,
                                      std::vector<std::set<int>>& clOut) const;
    void RunSmallClusterMerge(DSUWithOrder& dsu, const std::vector<int>& topoOrder,
                              std::vector<ScheduleCoreType>& coreTypes, std::vector<int>& cycle,
                              std::vector<std::set<int>>& clIn, std::vector<std::set<int>>& clOut, int clusterNum,
                              const std::unordered_set<int>& protectedClusterIds = {});

    // 基于 task 间 AIV 连通性计算分支 ID
    void ComputeTaskLevelBranches();

    std::vector<Operation*> opList_;
    std::vector<ScheduleCoreType> opCoreTypes_;
    std::vector<std::set<int>> opInGraph_;
    std::vector<std::set<int>> opOutGraph_;
    std::vector<std::pair<int, int>> sameLayerConnection_;
    std::vector<std::vector<int>> taskIdToOps_;
    std::vector<int> opIdxToTaskId_;
    std::vector<ScheduleCoreType> taskCoreTypes_;
    std::vector<std::set<int>> inGraph_;
    std::vector<std::set<int>> outGraph_;
    std::unordered_map<int, int> opMagicToIdx_;
    TaskGraph taskGraph_;
    std::unordered_set<int> oooScopeProtectedClusters_;
    std::vector<std::vector<int>> cycledSCCClusters_;
    std::vector<std::pair<int, int>> cycledTaskNodePairs_;
    void RecordCycledClusters(const std::vector<ScheduleCoreType>& clusterCoreTypes,
                              const std::vector<std::vector<int>>& sccResult);
    const std::vector<std::pair<int, int>>& GetCycledTaskNodePairs() const { return cycledTaskNodePairs_; }
};

// 使用TarJan算法寻找强连通分量
class StrongConnectionComponentFinder {
public:
    void Find(std::vector<std::set<int>>& inGraph, std::vector<std::set<int>>& outGraph,
              std::vector<std::vector<int>>& sccResult);
    void TarJanAlg(int idx, std::vector<std::set<int>>& outGraph, std::vector<std::vector<int>>& sccResult);
    std::vector<std::vector<int>> strongConnectionComponent_;
    int index_;
    std::vector<int> dfn_;
    std::vector<int> low_;
    std::vector<int> stack_;
    std::vector<bool> instack_;
    std::unordered_set<int> visited_;
};

// 使用传递闭包判断有向无环图中节点可达性
class DAGReachableJudger {
public:
    void Build(const std::vector<std::set<int>>& inGraph, const std::vector<std::set<int>>& outGraph);
    void SetReachable(const int src, const int dst);
    void MergeReachable(int src, int dst);
    bool IsReachable(int src, int dst);
    std::vector<std::vector<uint32_t>> reachableSet;
};

} // namespace npu::tile_fwk
#endif // PASS_TASK_SPLITTER_H
