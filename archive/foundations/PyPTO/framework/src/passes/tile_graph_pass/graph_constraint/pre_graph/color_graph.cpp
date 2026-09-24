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
 * \file color_graph.cpp
 * \brief
 */

#include "color_graph.h"

#include <unordered_set>

namespace npu::tile_fwk {
void ColorGraph::BuildSubgraphDependencyGraph(Function& function, std::vector<std::set<int>>& subgraphInGraph,
                                              std::vector<std::set<int>>& subgraphOutGraph)
{
    std::unordered_set<const LogicalTensor*> visitedTensors;
    visitedTensors.reserve(function.Operations(false).size());
    std::vector<int> producerSubgraphIds;
    std::vector<int> consumerSubgraphIds;
    std::vector<bool> producerSubgraphSeen(function.GetTotalSubGraphCount(), false);
    std::vector<bool> consumerSubgraphSeen(function.GetTotalSubGraphCount(), false);

    for (const auto& op : function.Operations(false)) {
        for (const auto& output : op.GetOOperands()) {
            if (!visitedTensors.insert(output.get()).second) {
                continue;
            }
            producerSubgraphIds.clear();
            consumerSubgraphIds.clear();
            for (const auto* producer : output->GetProducers()) {
                if (producer->BelongTo() != &function) {
                    continue;
                }
                int subgraphId = producer->GetSubgraphID();
                if (subgraphId < 0 || subgraphId >= static_cast<int>(function.GetTotalSubGraphCount())) {
                    continue;
                }
                if (!producerSubgraphSeen[subgraphId]) {
                    producerSubgraphSeen[subgraphId] = true;
                    producerSubgraphIds.emplace_back(subgraphId);
                }
            }
            for (const auto* consumer : output->GetConsumers()) {
                if (consumer->BelongTo() != &function) {
                    continue;
                }
                int subgraphId = consumer->GetSubgraphID();
                if (subgraphId < 0 || subgraphId >= static_cast<int>(function.GetTotalSubGraphCount())) {
                    continue;
                }
                if (!consumerSubgraphSeen[subgraphId]) {
                    consumerSubgraphSeen[subgraphId] = true;
                    consumerSubgraphIds.emplace_back(subgraphId);
                }
            }
            for (int producerSubgraphId : producerSubgraphIds) {
                for (int consumerSubgraphId : consumerSubgraphIds) {
                    if (producerSubgraphId != consumerSubgraphId) {
                        subgraphInGraph[consumerSubgraphId].insert(producerSubgraphId);
                        subgraphOutGraph[producerSubgraphId].insert(consumerSubgraphId);
                    }
                }
            }
            for (int producerSubgraphId : producerSubgraphIds) {
                producerSubgraphSeen[producerSubgraphId] = false;
            }
            for (int consumerSubgraphId : consumerSubgraphIds) {
                consumerSubgraphSeen[consumerSubgraphId] = false;
            }
        }
    }
}

Status DFSVisit(std::unordered_set<int>& visited, int preColor, std::unordered_map<int, int>& newColorMap,
                std::vector<std::set<int>>& colorInGraph, std::vector<std::set<int>>& colorOutGraph)
{
    std::vector<int> visitStack{preColor};
    std::unordered_set<int> inStack;
    while (visitStack.size() > 0) {
        int currColor = visitStack.back();
        if (visited.count(currColor) > 0) {
            visitStack.pop_back();
            continue;
        }
        bool allVisited = true;
        for (int pred : colorInGraph[currColor]) {
            if (visited.count(pred) == 0) {
                visitStack.push_back(pred);
                inStack.insert(pred);
                allVisited = false;
            }
        }
        if (!allVisited) {
            continue;
        }
        int currColorNum = newColorMap.size();
        newColorMap[currColor] = currColorNum;
        visited.insert(currColor);
        visitStack.pop_back();
        for (int succ : colorOutGraph[currColor]) {
            if (visited.count(succ) == 0 && inStack.count(succ) == 0) {
                visitStack.push_back(succ);
                inStack.insert(succ);
            }
        }
    }
    return SUCCESS;
}

Status ColorGraph::PreColorSort(Function& function)
{
    int colorNum = function.GetTotalSubGraphCount();
    std::vector<std::set<int>> colorInGraph(colorNum);
    std::vector<std::set<int>> colorOutGraph(colorNum);
    ColorGraph::BuildSubgraphDependencyGraph(function, colorInGraph, colorOutGraph);
    std::unordered_map<int, int> newColorMap;
    std::unordered_set<int> visited;
    for (int preColor = 0; preColor < colorNum; preColor++) {
        if (visited.count(preColor) > 0) {
            continue;
        }
        DFSVisit(visited, preColor, newColorMap, colorInGraph, colorOutGraph);
    }
    std::set<int> subgraphSet;
    for (auto& op : function.Operations()) {
        int opColor = op.GetSubgraphID();
        subgraphSet.insert(opColor);
        op.UpdateSubgraphID(newColorMap[opColor]);
    }
    function.SetTotalSubGraphCount(subgraphSet.size());
    return SUCCESS;
}

void ColorGraph::InitializeTensorMem(Operation& op) const
{
    for (auto& output : op.GetOOperands()) {
        TileRange range;
        range.memId = output->tensor->GetRawMagic();
        output->memoryrange = range;
    }
}
} // namespace npu::tile_fwk
