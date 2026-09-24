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
 * \file greedy_children.h
 * \brief
 */

#ifndef OSP_GREEDY_CHILDREN_H
#define OSP_GREEDY_CHILDREN_H

#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/bsp/scheduler/scheduler.h"

namespace npu::tile_fwk {
namespace osp {
template <typename GraphT>
class GreedyChildren : public Scheduler<GraphT> {
public:
    GreedyChildren(bool ensureEnoughSources = true) : Scheduler<GraphT>(), ensureEnoughSources_(ensureEnoughSources) {}

    ReturnStatus ComputeSchedule(BspSchedule<GraphT>& sched) override
    {
        using VertexType = VertexIdxT<GraphT>;
        const auto& instance = sched.GetInstance();
        const auto& graph = instance.GetComputationalDag();

        // Initialize state
        std::multiset<std::pair<unsigned, VertexType>, std::greater<>> next;
        std::vector<VertexType> predecessorsCount(instance.NumberOfVertices(), 0);
        InitializeScheduleState(sched, graph, next);

        unsigned superstepCounter = 0;

        while (!next.empty()) {
            std::unordered_set<VertexType> nodesAssignedThisSuperstep;
            std::vector<VWorkwT<GraphT>> processorWeights(instance.NumberOfProcessors(), 0);

            bool fewSources = next.size() < instance.NumberOfProcessors();
            bool nodeAdded = true;

            while (!next.empty() && nodeAdded) {
                nodeAdded = false;
                for (auto iter = next.begin(); iter != next.cend(); iter++) {
                    if (TryScheduleNode(sched, instance, graph, iter, nodesAssignedThisSuperstep, processorWeights,
                                        predecessorsCount, next, superstepCounter)) {
                        nodeAdded = true;
                        break;
                    }
                }
                if (ensureEnoughSources_ && fewSources && next.size() >= instance.NumberOfProcessors()) {
                    break;
                }
            }
            superstepCounter++;
        }

        return ReturnStatus::OSP_SUCCESS;
    }

private:
    bool ensureEnoughSources_;

    void InitializeScheduleState(BspSchedule<GraphT>& sched, const GraphT& graph,
                                 std::multiset<std::pair<unsigned, VertexIdxT<GraphT>>, std::greater<>>& next)
    {
        for (const auto& v : graph.Vertices()) {
            sched.SetAssignedProcessor(v, std::numeric_limits<unsigned>::max());
            if (graph.InDegree(v) == 0) {
                next.emplace(graph.OutDegree(v), v);
            }
        }
    }

    struct ParentCompatibilityResult {
        bool compatible;
        bool processorSet;
        unsigned processor;
    };

    bool ProcessSuperstepParent(const BspSchedule<GraphT>& sched, const BspInstance<GraphT>& instance,
                                VertexIdxT<GraphT> node, VertexIdxT<GraphT> parent, bool& processorSet,
                                unsigned& processorToBeAllocated)
    {
        const unsigned parProc = sched.AssignedProcessor(parent);

        if (!processorSet) {
            if (!instance.IsCompatible(node, parProc)) {
                return false;
            }
            processorSet = true;
            processorToBeAllocated = parProc;
            return true;
        }

        return parProc == processorToBeAllocated;
    }

    ParentCompatibilityResult CheckParentCompatibility(
        const GraphT& graph, const BspSchedule<GraphT>& sched, const BspInstance<GraphT>& instance,
        VertexIdxT<GraphT> node, const std::unordered_set<VertexIdxT<GraphT>>& nodesAssignedThisSuperstep)
    {
        bool processorSet = false;
        unsigned processorToBeAllocated = 0;

        for (const auto& par : graph.Parents(node)) {
            if (nodesAssignedThisSuperstep.count(par)) {
                if (!ProcessSuperstepParent(sched, instance, node, par, processorSet, processorToBeAllocated)) {
                    return {false, false, 0};
                }
            }
        }

        return {true, processorSet, processorToBeAllocated};
    }

    unsigned FindBestProcessor(const BspInstance<GraphT>& instance, VertexIdxT<GraphT> node,
                               const std::vector<VWorkwT<GraphT>>& processorWeights)
    {
        VWorkwT<GraphT> minWeight = std::numeric_limits<VWorkwT<GraphT>>::max();
        unsigned bestProc = std::numeric_limits<unsigned>::max();

        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
            if (instance.IsCompatible(node, p)) {
                if (processorWeights[p] < minWeight) {
                    minWeight = processorWeights[p];
                    bestProc = p;
                }
            }
        }

        return bestProc;
    }

    void UpdateReadyQueue(const GraphT& graph, VertexIdxT<GraphT> node,
                          std::vector<VertexIdxT<GraphT>>& predecessorsCount,
                          std::multiset<std::pair<unsigned, VertexIdxT<GraphT>>, std::greater<>>& next)
    {
        std::vector<VertexIdxT<GraphT>> newNodes;
        for (const auto& chld : graph.Children(node)) {
            predecessorsCount[chld]++;
            if (predecessorsCount[chld] == graph.InDegree(chld)) {
                newNodes.emplace_back(chld);
            }
        }
        for (const auto& vrt : newNodes) {
            next.emplace(graph.OutDegree(vrt), vrt);
        }
    }

    bool TryScheduleNode(BspSchedule<GraphT>& sched, const BspInstance<GraphT>& instance, const GraphT& graph,
                         typename std::multiset<std::pair<unsigned, VertexIdxT<GraphT>>, std::greater<>>::iterator iter,
                         std::unordered_set<VertexIdxT<GraphT>>& nodesAssignedThisSuperstep,
                         std::vector<VWorkwT<GraphT>>& processorWeights,
                         std::vector<VertexIdxT<GraphT>>& predecessorsCount,
                         std::multiset<std::pair<unsigned, VertexIdxT<GraphT>>, std::greater<>>& next,
                         unsigned superstepCounter)
    {
        const auto& node = iter->second;

        auto result = CheckParentCompatibility(graph, sched, instance, node, nodesAssignedThisSuperstep);
        if (!result.compatible) {
            return false;
        }

        sched.SetAssignedSuperstep(node, superstepCounter);

        if (result.processorSet) {
            sched.SetAssignedProcessor(node, result.processor);
        } else {
            sched.SetAssignedProcessor(node, FindBestProcessor(instance, node, processorWeights));
        }

        nodesAssignedThisSuperstep.emplace(node);
        processorWeights[sched.AssignedProcessor(node)] += graph.VertexWorkWeight(node);

        UpdateReadyQueue(graph, node, predecessorsCount, next);
        next.erase(iter);

        return true;
    }
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_GREEDY_CHILDREN_HPP
