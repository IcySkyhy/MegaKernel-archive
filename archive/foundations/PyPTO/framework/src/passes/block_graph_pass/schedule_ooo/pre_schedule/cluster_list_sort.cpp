/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cluster_list_sort.h"
#include "passes/pass_log/pass_log.h"

#include <algorithm>
#include <deque>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace npu::tile_fwk {

// === 静态量查询 === //

int64_t ClusterListSort::TensorBytes(int memId) const
{
    auto it = state_.localBufferMap.find(memId);
    if (it == state_.localBufferMap.end() || it->second == nullptr) {
        return 0;
    }
    return static_cast<int64_t>(it->second->size);
}

// === occupy / release === //

// release = 该 op 是本 list 内末位读者（tensorRemainCons_==1）的输入 tensor 字节（同 memId 去重）。
int64_t ClusterListSort::ReleaseBytes(Operation* op) const
{
    int64_t bytes = 0;
    std::unordered_set<int> counted;
    for (const auto& t : op->GetIOperands()) {
        if (t == nullptr || !counted.insert(t->memoryrange.memId).second) {
            continue;
        }
        auto rit = tensorRemainCons_.find(t->memoryrange.memId);
        if (rit != tensorRemainCons_.end() && rit->second == 1) {
            bytes += TensorBytes(t->memoryrange.memId);
        }
    }
    return bytes;
}

// occupy = 该 op 写出的新本地输出字节（落 DDR 的输出排除；同 memId 去重）。
int64_t ClusterListSort::OccupyBytes(Operation* op) const
{
    int64_t bytes = 0;
    std::unordered_set<int> counted;
    for (const auto& t : op->GetOOperands()) {
        if (t == nullptr) {
            continue;
        }
        if (t->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            continue; // 落 DDR 不占 local
        }
        int memId = t->memoryrange.memId;
        if (!counted.insert(memId).second) {
            continue;
        }
        bytes += TensorBytes(memId);
    }
    return bytes;
}

// op 执行 → 它读的每个 memId 在本 list 内的剩余消费者数各 −1（同 op 内去重）。
void ClusterListSort::DecRemainCons(Operation* op)
{
    std::unordered_set<int> seen;
    for (const auto& t : op->GetIOperands()) {
        if (t == nullptr || !seen.insert(t->memoryrange.memId).second) {
            continue;
        }
        auto rit = tensorRemainCons_.find(t->memoryrange.memId);
        if (rit != tensorRemainCons_.end() && rit->second > 0) {
            --rit->second;
        }
    }
}

// 簇主排序键：静态 (basePeak, end, secondaryId)，全不依赖剩余消费者数（键值建簇时算好，调度期不变）。
// net 不在这里——它在取候选时对 peak/end 打平的簇即时动态算。
bool ClusterListSort::ClusterStaticLess(int a, int b) const
{
    const Cluster& ca = clusters_[static_cast<size_t>(a)];
    const Cluster& cb = clusters_[static_cast<size_t>(b)];
    if (ca.basePeak != cb.basePeak) {
        return ca.basePeak < cb.basePeak; // 1) peak 最小
    }
    if (ca.end != cb.end) {
        return ca.end < cb.end; // 2) tie → end 最小
    }
    // 3) tie → 同一 secondaryId（喂同一聚点的分支）聚拢：同组相邻，缩短其产物同挂内存窗口
    if (ca.secondaryId != cb.secondaryId) {
        return ca.secondaryId < cb.secondaryId;
    }
    return a < b; // 4) 稳定兜底
}

// === 建簇：极大单入单出链，fan-out / join / 多产 / 多消 处断簇 === //

// 一个 tensor 算不算 "本地"（片上）：与 OccupyBytes / dep_manager 一致，`< MEM_DEVICE_DDR`。
// copy_in（读 DDR 写本地）→ 无本地输入 → 天然簇起点；copy_out（读本地写 DDR）→ 无本地输出 →
// 天然簇终点。故 copy 边界由 "本地有效度" 归一覆盖，无需按 opcode 枚举。
static bool IsLocalTensor(const LogicalTensorPtr& t)
{
    return t != nullptr && t->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR;
}

// 取 op 的 distinct 本地输入 / 输出 memId 集合（同 op 内按 memId 去重）。
static std::vector<int> LocalInMemIds(Operation* op)
{
    std::vector<int> ids;
    std::unordered_set<int> seen;
    for (const auto& t : op->GetIOperands()) {
        if (IsLocalTensor(t) && seen.insert(t->memoryrange.memId).second) {
            ids.push_back(t->memoryrange.memId);
        }
    }
    return ids;
}
static std::vector<int> LocalOutMemIds(Operation* op)
{
    std::vector<int> ids;
    std::unordered_set<int> seen;
    for (const auto& t : op->GetOOperands()) {
        if (IsLocalTensor(t) && seen.insert(t->memoryrange.memId).second) {
            ids.push_back(t->memoryrange.memId);
        }
    }
    return ids;
}

int64_t ClusterListSort::ComputeBasePeak(const std::vector<Operation*>& clusterOps) const
{
    // 有序遍历，每个 op 的 live = 其 输入∪输出 的 distinct-memId bytes 之和。
    // 复用（输出与输入同 memId）→ 同一 memId 在集合中只出现一次 → 天然取 max（同 memId bytes 相同），
    // 不去重整簇、不分组。
    int64_t peak = 0;
    for (auto* op : clusterOps) {
        std::unordered_set<int> live;
        int64_t cur = 0;
        auto add = [&](const LogicalTensorPtr& t) {
            if (!IsLocalTensor(t)) {
                return;
            }
            int memId = t->memoryrange.memId;
            if (live.insert(memId).second) {
                cur += TensorBytes(memId);
            }
        };
        for (const auto& t : op->GetIOperands()) {
            add(t);
        }
        for (const auto& t : op->GetOOperands()) {
            add(t);
        }
        peak = std::max(peak, cur);
    }
    return peak;
}

// 本地 buffer 的生产者 / 消费者列表（只统计 normals_，按 memId 聚合 view）。
// 一张 vector map 同时给度数(size)和唯一 op(size==1 时的 [0])。
void ClusterListSort::ComputeDegrees()
{
    producersByMem_.clear();
    consumersByMem_.clear();
    for (auto* op : normals_) {
        for (int memId : LocalOutMemIds(op)) {
            producersByMem_[memId].push_back(op);
        }
        for (int memId : LocalInMemIds(op)) {
            consumersByMem_[memId].push_back(op);
        }
    }
}

// 链起点：0 本地输入(copy_in) / 多本地输入(join) / 唯一输入被多方产(join) / 生产者在 list 外(边界) /
//         有 depManager 前驱在 list 内但非唯一本地生产者（DDR/反依赖等链外数据边 → 中段 op 会被卡 unmetPreds_）。
bool ClusterListSort::IsChainStart(Operation* op)
{
    auto in = LocalInMemIds(op);
    if (in.size() != 1) {
        return true;
    }
    auto it = producersByMem_.find(in[0]);
    if (it == producersByMem_.end() || it->second.size() > 1) {
        return true; // 无 list 内生产者 → 边界起点
    }
    Operation* soleLocalProd = it->second[0];
    for (auto* pred : state_.depManager.GetPredecessors(op)) {
        if (pred != soleLocalProd && normalSet_.count(pred) != 0U) {
            return true; // 链外数据依赖 → 断簇为头，让 TryAddReady 正确 gate
        }
    }
    return false;
}

// 链终点：0 本地输出(copy_out) / 多本地输出(fan-out) / 唯一输出被多方读(fan-out) / 消费者在 list 外(边界) /
//         有 depManager 后继在 list 内但非唯一本地消费者（对称断簇，保证簇末释放点不被链外后继遮蔽）。
bool ClusterListSort::IsChainEnd(Operation* op)
{
    auto out = LocalOutMemIds(op);
    if (out.size() != 1) {
        return true;
    }
    auto it = consumersByMem_.find(out[0]);
    if (it == consumersByMem_.end() || it->second.size() > 1) {
        return true; // 无 list 内消费者 → 边界终点
    }
    Operation* soleLocalCons = it->second[0];
    for (auto* succ : state_.depManager.GetSuccessors(op)) {
        if (succ != soleLocalCons && normalSet_.count(succ) != 0U) {
            return true;
        }
    }
    return false;
}

// 相邻可并：W->X 当且仅当 W 非 END、X 非 START，且连接 tensor 是 W 唯一本地输出 == X 唯一本地输入。
void ClusterListSort::LinkAdjacent(std::unordered_map<Operation*, Operation*>& nextOf,
                                   std::unordered_map<Operation*, Operation*>& prevOf)
{
    for (auto* w : normals_) {
        if (IsChainEnd(w)) {
            continue;
        }
        int memId = LocalOutMemIds(w)[0]; // IsChainEnd==false 保证恰 1 个本地输出且消费者数==1
        Operation* x = consumersByMem_.at(memId).front();
        if (IsChainStart(x)) {
            continue;
        }
        if (LocalInMemIds(x)[0] != memId) { // X 唯一本地输入必须正是 memId
            continue;
        }
        nextOf[w] = x;
        prevOf[x] = w;
    }
}

// 从链头（无 prev）走链建簇（ops 即簇内执行序，孤立 op 自成一簇），并做对称性校验。
// 对称性：inDeg>1 必是簇首、outDeg>1 必是簇末。违反即建链有 bug，后续 peak/end/排序全基于错误结构 → fail-fast。
Status ClusterListSort::WalkChains(const std::unordered_map<Operation*, Operation*>& nextOf,
                                   const std::unordered_map<Operation*, Operation*>& prevOf)
{
    for (auto* op : normals_) {
        if (prevOf.count(op) != 0U) {
            continue; // 非链头
        }
        int cid = static_cast<int>(clusters_.size());
        Cluster cl;
        cl.id = cid;
        for (Operation* cur = op; cur != nullptr;) {
            cluster_[cur] = cid;
            cl.ops.push_back(cur);
            auto nit = nextOf.find(cur);
            cur = (nit == nextOf.end()) ? nullptr : nit->second;
        }
        cl.basePeak = ComputeBasePeak(cl.ops);
        cl.end = cl.ops.empty() ? 0 : OccupyBytes(cl.ops.back()); // 簇末保留的本地输出字节
        for (size_t i = 0; i < cl.ops.size(); ++i) {
            bool badStart = IsChainStart(cl.ops[i]) && i != 0;
            bool badEnd = IsChainEnd(cl.ops[i]) && i != cl.ops.size() - 1;
            if (badStart || badEnd) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "ClusterListSort: op not at expected chain %s, cluster=%d idx=%zu op=%s.",
                                  badStart ? "head" : "tail", cid, i, state_.GetOpInfo(cl.ops[i]).c_str());
                return FAILED;
            }
        }
        clusters_.push_back(std::move(cl));
    }
    return SUCCESS;
}

// 喂同一聚点（inDeg>1 op）的分支簇并查集合并，secondaryId = 簇的并查集根。排序键用它把同组分支聚拢。
// 让它们在 peak/end/net 全打平时相邻执行，缩短各分支产物同挂内存的窗口。
void ClusterListSort::AssignSecondaryIds()
{
    int n = static_cast<int>(clusters_.size());
    std::vector<int> parent(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        parent[static_cast<size_t>(i)] = i;
    }
    std::function<int(int)> find = [&](int x) {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            x = parent[static_cast<size_t>(x)];
        }
        return x;
    };
    for (auto* j : normals_) {
        auto in = LocalInMemIds(j);
        if (in.size() < 2) {
            continue; // 非聚点
        }
        int firstCid = -1;
        for (int memId : in) {
            auto pit = producersByMem_.find(memId);
            if (pit == producersByMem_.end() || pit->second.size() != 1) {
                continue; // 输入无唯一本地生产者（多产/外部）→ 跳过
            }
            int cid = cluster_[pit->second[0]];
            if (firstCid < 0) {
                firstCid = cid;
            } else if (find(firstCid) != find(cid)) {
                parent[static_cast<size_t>(std::max(find(firstCid), find(cid)))] = std::min(
                    find(firstCid), find(cid)); // 归到较小根，id 稳定
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        clusters_[static_cast<size_t>(i)].secondaryId = find(i);
    }
}

Status ClusterListSort::BuildClusters()
{
    clusters_.clear();
    cluster_.clear();
    ComputeDegrees();
    std::unordered_map<Operation*, Operation*> nextOf;
    std::unordered_map<Operation*, Operation*> prevOf;
    LinkAdjacent(nextOf, prevOf);
    if (WalkChains(nextOf, prevOf) != SUCCESS) {
        return FAILED;
    }
    AssignSecondaryIds();
    APASS_LOG_INFO_F(Elements::Operation, "ClusterListSort: built %zu clusters.", clusters_.size());
    for (size_t i = 0; i < clusters_.size(); ++i) {
        const Cluster& cl = clusters_[i];
        std::string opsStr;
        for (size_t j = 0; j < cl.ops.size(); ++j) {
            if (j != 0) {
                opsStr += " -> ";
            }
            opsStr += state_.GetOpInfo(cl.ops[j]);
        }
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "ClusterListSort: cluster[%zu] basePeak=%lld end=%lld secondaryId=%d ops(%zu): %s", i,
                          static_cast<long long>(cl.basePeak), static_cast<long long>(cl.end), cl.secondaryId,
                          cl.ops.size(), opsStr.c_str());
    }
    return SUCCESS;
}

// === alloc 按首次引用回插 === //

std::vector<Operation*> ClusterListSort::CollectLeftoverAllocs(const std::vector<Operation*>& allocOps,
                                                               const std::unordered_set<int>& allocated) const
{
    std::vector<Operation*> leftover;
    for (auto* a : allocOps) {
        bool placed = false;
        for (const auto& t : a->GetOOperands()) {
            if (t != nullptr && allocated.count(t->memoryrange.memId) != 0U) {
                placed = true;
                break;
            }
        }
        if (!placed) {
            leftover.push_back(a);
        }
    }
    return leftover;
}

std::vector<Operation*> ClusterListSort::ReinsertAllocs(const std::vector<Operation*>& allocOps) const
{
    // memId -> 产它的 alloc op（一个 memId 唯一一个 alloc）
    std::unordered_map<int, Operation*> allocByMem;
    for (auto* a : allocOps) {
        for (const auto& t : a->GetOOperands()) {
            if (t != nullptr) {
                allocByMem[t->memoryrange.memId] = a;
            }
        }
    }

    std::vector<Operation*> newOps;
    newOps.reserve(orderedNormals_.size() + allocOps.size());
    std::unordered_set<int> allocated;
    auto emitAllocFor = [&](int memId) {
        if (memId < 0 || allocated.count(memId) != 0U) {
            return;
        }
        auto ait = allocByMem.find(memId);
        if (ait == allocByMem.end()) {
            return;
        }
        newOps.push_back(ait->second);
        allocated.insert(memId);
    };

    for (auto* op : orderedNormals_) {
        for (const auto& t : op->GetOOperands()) {
            if (t != nullptr) {
                emitAllocFor(t->memoryrange.memId);
            }
        }
        for (const auto& t : op->GetIOperands()) {
            if (t != nullptr) {
                emitAllocFor(t->memoryrange.memId);
            }
        }
        newOps.push_back(op);
    }

    auto leftover = CollectLeftoverAllocs(allocOps, allocated);
    if (!leftover.empty()) {
        newOps.insert(newOps.begin(), leftover.begin(), leftover.end());
    }
    return newOps;
}

// === 主流程 === //

// 簇头前驱已齐 → 按静态序增量插入 readyClusters_（去重，一簇只入一次）。
void ClusterListSort::TryAddReady(int cid)
{
    Cluster& cl = clusters_[static_cast<size_t>(cid)];
    if (cl.inReady) {
        return;
    }
    Operation* head = cl.CurOp();
    if (head != nullptr && unmetPreds_[head] == 0) {
        auto it = std::lower_bound(readyClusters_.begin(), readyClusters_.end(), cid,
                                   [this](int a, int b) { return ClusterStaticLess(a, b); });
        size_t pos = static_cast<size_t>(it - readyClusters_.begin());
        readyClusters_.insert(it, cid);
        readyPos_[cid] = pos;
        for (size_t i = pos + 1; i < readyClusters_.size(); ++i) {
            readyPos_[readyClusters_[i]] = i;
        }
        cl.inReady = true;
    }
}

// 执行一个 op：入序列、剩余消费者数递减、簇游标推进；解锁 normals_ 内后继前驱，跨簇边落在簇头则该簇可能变 ready。
void ClusterListSort::Commit(Operation* op)
{
    orderedNormals_.push_back(op);
    DecRemainCons(op);
    int cid = cluster_[op];
    Cluster& cl = clusters_[static_cast<size_t>(cid)];
    lastSecondaryId_ = cl.secondaryId; // net 打平时黏同组
    ++cl.cursor;
    if (cl.Done()) { // 簇排完即移出 ready 队，避免死 id 累积拖慢后续 SelectReadyHeads
        auto pit = readyPos_.find(cid);
        if (pit != readyPos_.end()) {
            size_t pos = pit->second;
            readyPos_.erase(pit);
            readyClusters_.erase(readyClusters_.begin() + static_cast<ptrdiff_t>(pos));
            for (size_t i = pos; i < readyClusters_.size(); ++i) {
                readyPos_[readyClusters_[i]] = i;
            }
        }
        cl.inReady = false;
        if (currentCid_ == cid) {
            currentCid_ = -1; // 当前黏簇完成 → 下一轮重选
        }
    }
    for (auto* succ : state_.depManager.GetSuccessors(op)) {
        if (normalSet_.count(succ) == 0U) {
            continue;
        }
        auto uit = unmetPreds_.find(succ);
        if (uit != unmetPreds_.end() && uit->second > 0) {
            --uit->second;
        }
        int scid = cluster_[succ]; // 跨簇边只落在簇头；满足即该簇可入 ready
        if (clusters_[static_cast<size_t>(scid)].ops.front() == succ && unmetPreds_[succ] == 0) {
            TryAddReady(scid);
        }
    }
}

// 取未完成、未冻结、前驱已齐的 ready 簇头，静态序保持。unmetPreds_ 复检：簇中段 op 可能带
//     depManager 的反依赖(WAR/WAW)外部前驱，不止链上数据边，游标走到它时未必已齐。
std::vector<Operation*> ClusterListSort::SelectReadyHeads() const
{
    std::vector<Operation*> heads;
    for (int cid : readyClusters_) {
        Operation* c = clusters_[static_cast<size_t>(cid)].CurOp();
        if (c != nullptr && inBreakQ_.count(c) == 0U && unmetPreds_.at(c) == 0) {
            heads.push_back(c);
        }
    }
    return heads;
}

// 候选 = 静态序头部 (basePeak, end) 打平组里 net 最小者（net = occupy−release，动态现算）。
// net 再打平 → 黏在与上一个 Commit 同 secondaryId 的簇（把同一 join 分支组连续做完），
// 都不同再落静态序（secondaryId→id，heads 已按此排好，严格<保留靠前者）。
std::pair<Operation*, int64_t> ClusterListSort::PickTiedCandidate(const std::vector<Operation*>& heads) const
{
    const Cluster& front = clusters_[static_cast<size_t>(cluster_.at(heads.front()))];
    Operation* cand = heads.front();
    int64_t candNet = NetOf(cand);
    bool candSticky = clusters_[static_cast<size_t>(cluster_.at(cand))].secondaryId == lastSecondaryId_;
    for (size_t i = 1; i < heads.size(); ++i) {
        const Cluster& cl = clusters_[static_cast<size_t>(cluster_.at(heads[i]))];
        if (cl.basePeak != front.basePeak || cl.end != front.end) {
            break; // 超出打平组
        }
        int64_t n = NetOf(heads[i]);
        bool sticky = cl.secondaryId == lastSecondaryId_;
        // net 更小则换；net 平且它黏同组而当前候选不黏 → 换（黏性作为 net 的次级键）。
        if (n < candNet || (n == candNet && sticky && !candSticky)) {
            cand = heads[i];
            candNet = n;
            candSticky = sticky;
        }
    }
    return {cand, candNet};
}

// 三方比较：cand（打平组净涨最小的簇头）、breakQ_ 队头、下一个 ready 簇头，取 net 最小者执行；
//     若 cand 落败则此刻才押后进 breakQ_（尾进）。调用前已知 candNet>0。
void ClusterListSort::ThreeWayChoose(const std::vector<Operation*>& heads, Operation* cand, int64_t candNet)
{
    Operation* best = cand;
    int64_t bestNet = candNet;
    if (!breakQ_.empty()) {
        int64_t bqNet = NetOf(breakQ_.front());
        if (bqNet < bestNet) {
            best = breakQ_.front();
            bestNet = bqNet;
        }
    }
    for (auto* h : heads) { // "下一个 ready 簇头" = 静态序里第一个非 cand 的簇头
        if (h != cand) {
            if (NetOf(h) < bestNet) {
                best = h;
            }
            break;
        }
    }
    if (best == cand) {
        Commit(cand); // 候选净涨最小 → 簇内继续，currentCid_ 不变
        return;
    }
    breakQ_.push_back(cand);
    inBreakQ_.insert(cand);
    if (best == breakQ_.front()) {
        breakQ_.pop_front();
        inBreakQ_.erase(best);
    }
    Commit(best);
    currentCid_ = cluster_.at(best); // 切到胜出簇
}

// readyClusters_ 已在插入时按 ClusterStaticLess 有序，直接取可执行簇头。
std::vector<Operation*> ClusterListSort::SortedReadyHeads() { return SelectReadyHeads(); }

// 黏簇尝试：当前簇可执行则返回其 op 和 net，否则清空 currentCid_ 返回 nullptr。
std::pair<Operation*, int64_t> ClusterListSort::TryStickyCluster()
{
    if (currentCid_ < 0) {
        return {nullptr, 0};
    }
    Operation* head = clusters_[static_cast<size_t>(currentCid_)].CurOp();
    if (head != nullptr && inBreakQ_.count(head) == 0U && unmetPreds_.at(head) == 0) {
        return {head, NetOf(head)};
    }
    currentCid_ = -1; // 完成 / 冻结 / 卡前驱 → 重选
    return {nullptr, 0};
}

// 主循环：黏簇 + 静态有序 ready 簇 + 三方比较 break_op_queue。
// 黏簇：优先推进 currentCid_ 直到其 op 净涨内存（candNet>0）才进三方比较考虑让位；
//       当前簇完成/冻结/卡前驱时清空 currentCid_，下一轮 PickTiedCandidate 重选。
Status ClusterListSort::RunScheduleLoop()
{
    for (const auto& cl : clusters_) { // 初始 ready：源簇（簇头无未满足前驱）
        TryAddReady(cl.id);
    }
    while (orderedNormals_.size() < normals_.size()) {
        // 1. 黏簇尝试
        auto [cand, candNet] = TryStickyCluster();
        // 2. 无黏簇候选 → 取 heads 重选（或 breakQ 兜底）
        std::vector<Operation*> heads;
        bool newCluster = false;
        if (cand == nullptr) {
            heads = SortedReadyHeads();
            if (heads.empty()) { // 无 ready 簇头 → 强制排 breakQ_ 队头（保证收敛、不饿死）
                if (breakQ_.empty()) {
                    APASS_LOG_ERROR_F(Elements::Operation,
                                      "ClusterListSort: no ready cluster and empty breakQ (cycle?).");
                    return FAILED;
                }
                Operation* head = breakQ_.front();
                breakQ_.pop_front();
                inBreakQ_.erase(head);
                Commit(head);
                currentCid_ = cluster_[head];
                continue;
            }
            auto [c, n] = PickTiedCandidate(heads);
            cand = c;
            candNet = n;
            currentCid_ = cluster_.at(cand);
            newCluster = true;
        }
        // 3. 刚选定簇首个 op 无条件执行 / 黏簇候选净涨内存 → 取 heads 供三方比较
        if (!newCluster && candNet > 0 && heads.empty()) {
            heads = SortedReadyHeads();
        }
        // 4. 执行
        if (newCluster || candNet <= 0) {
            Commit(cand); // 新簇首 op 无条件直行 / net≤0 黏簇直行
        } else {
            ThreeWayChoose(heads, cand, candNet); // net>0 三方比较，内部更新 currentCid_
        }
    }
    return SUCCESS;
}

// 分离 alloc / normals，重置工作态，初始化 tensorRemainCons_。返回 allocOps。
std::vector<Operation*> ClusterListSort::InitRunState()
{
    std::vector<Operation*> allocOps;
    normals_.clear();
    for (auto* op : operations) {
        (state_.IsOpAlloc(op) ? allocOps : normals_).push_back(op);
    }
    normalSet_ = std::unordered_set<Operation*>(normals_.begin(), normals_.end());
    breakQ_.clear();
    inBreakQ_.clear();
    readyClusters_.clear();
    readyPos_.clear();
    lastSecondaryId_ = -1;
    currentCid_ = -1;
    orderedNormals_.clear();
    orderedNormals_.reserve(normals_.size());

    // 本 list 内剩余消费者计数：只扫 normals_ 输入，同 op 内 memId 去重。减到 1 即末位读者 → 触发 ReleaseBytes。
    tensorRemainCons_.clear();
    for (auto* op : normals_) {
        std::unordered_set<int> seen;
        for (const auto& t : op->GetIOperands()) {
            if (t != nullptr && seen.insert(t->memoryrange.memId).second) {
                ++tensorRemainCons_[t->memoryrange.memId];
            }
        }
    }
    return allocOps;
}

// 未满足前驱数（只算 normals_ 内部依赖）。
void ClusterListSort::InitUnmetPreds()
{
    unmetPreds_.clear();
    for (auto* op : normals_) {
        int cnt = 0;
        for (auto* pred : state_.depManager.GetPredecessors(op)) {
            cnt += static_cast<int>(normalSet_.count(pred) != 0U);
        }
        unmetPreds_[op] = cnt;
    }
}

Status ClusterListSort::DoSortOps()
{
    auto allocOps = InitRunState();

    if (BuildClusters() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "ClusterListSort: BuildClusters failed (cluster shape invariant violated).");
        return FAILED;
    }

    InitUnmetPreds();

    if (RunScheduleLoop() != SUCCESS) {
        return FAILED;
    }
    if (orderedNormals_.size() != normals_.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "ClusterListSort: ordered %zu vs %zu normals (cycle?).",
                          orderedNormals_.size(), normals_.size());
        return FAILED;
    }

    operations = ReinsertAllocs(allocOps);
    APASS_LOG_INFO_F(Elements::Operation, "ClusterListSort: %zu allocs + %zu normals scheduled.", allocOps.size(),
                     orderedNormals_.size());

    if (ExecuteOp() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ExecuteOp failed.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
