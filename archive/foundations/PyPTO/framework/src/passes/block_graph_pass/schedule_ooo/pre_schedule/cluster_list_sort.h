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
 * \file cluster_list_sort.h
 * \brief Cluster-based List Schedule: static clustering + ready-cluster queue + three-way compare,
 *        targeting vec, not online-memory aware. Splits the op graph into maximal single-in-single-out
 *        chains (clusters), maintains a ready-cluster queue, and picks one op per step via comparator
 *        (peak -> end -> net-release -> secondaryId). Ops with net memory growth are deferred via
 *        break_op_queue with three-way comparison. Allocs are excluded and reinserted at first use;
 *        L0 memory is handled by the base class ExecuteOp().
 */

#ifndef PASS_CLUSTER_LIST_SORT_H
#define PASS_CLUSTER_LIST_SORT_H

#include "optimize_sort.h"

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace npu::tile_fwk {

class ClusterListSort : public OptimizeSort {
public:
    using OptimizeSort::OptimizeSort;

private:
    Status DoSortOps() override;

    // === 静态量 / 内存量 ===
    int64_t TensorBytes(int memId) const;
    int64_t ReleaseBytes(Operation* op) const; // 该 op 执行后 refcount 归零的输入字节
    int64_t OccupyBytes(Operation* op) const;  // 该 op 新写出的本地输出字节
    int64_t NetOf(Operation* op) const { return OccupyBytes(op) - ReleaseBytes(op); }
    void DecRemainCons(Operation* op); // op 输入 memId 的本 list 内剩余消费者数各 −1（同 op 内去重）

    // === 建簇：极大单入单出链 ===
    Status BuildClusters();
    void ComputeDegrees();            // 填 producersByMem_/consumersByMem_（按 memId 聚合）
    bool IsChainStart(Operation* op); // 链起点判据（查 depManager 非 const）
    bool IsChainEnd(Operation* op);   // 链终点判据（同上）
    void LinkAdjacent(std::unordered_map<Operation*, Operation*>& nextOf,
                      std::unordered_map<Operation*, Operation*>& prevOf);
    Status WalkChains(const std::unordered_map<Operation*, Operation*>& nextOf,
                      const std::unordered_map<Operation*, Operation*>& prevOf); // 走链建簇 + 对称性校验
    void AssignSecondaryIds();                                                   // 分支簇并查集合并
    int64_t ComputeBasePeak(const std::vector<Operation*>& clusterOps) const;    // 簇内静态峰值

    // === 主排序 / 调度 ===
    bool ClusterStaticLess(int a, int b) const;       // 簇静态键：(basePeak, end, secondaryId, id)
    void TryAddReady(int cid);                        // 簇头前驱已齐 → 入 readyClusters_（去重）
    void Commit(Operation* op);                       // 执行一个 op：入序 / refcount− / 游标+ / 解锁后继
    std::vector<Operation*> SelectReadyHeads() const; // 取可执行的 ready 簇头
    std::vector<Operation*> SortedReadyHeads();       // 取 ready 簇头（插入时已有序）
    std::pair<Operation*, int64_t> TryStickyCluster(); // 黏簇尝试，失败清 currentCid_
    std::pair<Operation*, int64_t> PickTiedCandidate(const std::vector<Operation*>& heads) const; // 打平组选净涨最小
    void ThreeWayChoose(const std::vector<Operation*>& heads, Operation* cand, int64_t candNet); // 三方净涨比较
    Status RunScheduleLoop();

    // === alloc 回插 ===
    std::vector<Operation*> CollectLeftoverAllocs(const std::vector<Operation*>& allocOps,
                                                  const std::unordered_set<int>& allocated) const;
    std::vector<Operation*> ReinsertAllocs(const std::vector<Operation*>& allocOps) const;

    // === DoSortOps 子步骤 ===
    std::vector<Operation*> InitRunState(); // 分离 alloc/normals + 重置工作态 + tensorRemainCons_
    void InitUnmetPreds();                  // unmetPreds_ 初始化（只算 normals_ 内依赖）

    // === 簇（簇级状态收进一个对象，不写进 IR） ===
    struct Cluster {
        int id = -1;
        std::vector<Operation*> ops; // 建链序 == 簇内执行序（intraOrder 即下标）
        int64_t basePeak = 0;        // 簇内静态峰值
        int64_t end = 0;             // 簇末保留的本地输出字节
        int secondaryId = -1;        // 喂同一聚点的分支簇共享
        size_t cursor = 0;           // 下一个待执行 op 的下标
        bool inReady = false;        // 是否已入 readyClusters_（去重，防重复 push）
        Operation* CurOp() const { return cursor < ops.size() ? ops[cursor] : nullptr; }
        bool Done() const { return cursor >= ops.size(); }
    };

    // === 簇结构 ===
    std::vector<Cluster> clusters_;               // 按 clusterId 下标索引
    std::unordered_map<Operation*, int> cluster_; // op -> clusterId（反查）

    // === 建簇工作态（ComputeDegrees 内填，只统计 normals_，按 memId 聚合 view） ===
    // memId -> 本 list 内产/读该 buffer 的 op 们。度数=size()；唯一者=size()==1 时的 [0]。
    std::unordered_map<int, std::vector<Operation*>> producersByMem_;
    std::unordered_map<int, std::vector<Operation*>> consumersByMem_;

    // === DoSortOps 单次运行的工作态（开头重置） ===
    std::vector<Operation*> normals_;          // 非 alloc op
    std::unordered_set<Operation*> normalSet_; // normals_ 的集合视图
    std::unordered_map<int, int>
        tensorRemainCons_; // memId -> 本 list 内剩余未执行消费者 op 数（非全局 refcount，不含 alloc）
    std::unordered_map<Operation*, int> unmetPreds_; // op -> 未满足前驱数（只算 normals_ 内）
    std::deque<Operation*> breakQ_;                  // 押后的 op，尾进头出
    std::unordered_set<Operation*> inBreakQ_;        // 快速判定 + 冻结其簇
    std::vector<int> readyClusters_;                 // ready 簇 id，按 ClusterStaticLess 有序
    std::unordered_map<int, size_t> readyPos_;       // clusterId → readyClusters_ 下标（O(1) 定位删除位置）
    int lastSecondaryId_ = -1;               // 上一个 Commit 的簇的 secondaryId（net 打平时黏在同组）
    int currentCid_ = -1;                    // 黏簇：当前正在推进的簇，-1 表示需重选
    std::vector<Operation*> orderedNormals_; // 最终执行序（不含 alloc）
};

} // namespace npu::tile_fwk
#endif // PASS_CLUSTER_LIST_SORT_H
