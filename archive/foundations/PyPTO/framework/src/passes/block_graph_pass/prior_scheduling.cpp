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
 * \file prior_scheduling.cpp
 * \brief
 */

#include "prior_scheduling.h"

#include <queue>
#include <climits>
#include "interface/function/function.h"
#include "tilefwk/error_code.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
Status PriorScheduling::RunOnFunction(Function& function)
{
#ifndef PRIOR_SCHEDULING
    return SUCCESS;
#endif
    PriorSchedulingFunc(function);
    return SUCCESS;
}

std::vector<int> FordBellman(int wholeGraphTopologySize, const std::vector<std::pair<int, int>>& edges)
{
    std::vector<int> subgrDepthVector(wholeGraphTopologySize + 1, INT_MAX);
    subgrDepthVector[wholeGraphTopologySize] = 0;

    bool any = true;
    while (any == true) {
        any = false;
        for (auto elem : edges) {
            if (subgrDepthVector[elem.second] > subgrDepthVector[elem.first] - 1) {
                subgrDepthVector[elem.second] = subgrDepthVector[elem.first] - 1;
                any = true;
            }
        }
    }
    return subgrDepthVector;
}

void FillGraphTopologyInfo(std::map<int, std::set<int>>& subgrChild, std::map<int, std::set<int>>& subgrParent,
                           std::map<int, int>& subgrChildNum, std::map<int, int>& subgrParentNum, Function& function)
{
    auto wholeGraphTopology = function.rootFunc_->topoInfo_.topology_;
    int wholeGraphTopologySize = int(wholeGraphTopology.size());
    std::cout << "!Topology size: " << wholeGraphTopologySize << std::endl;

    // Fill graph topology childs & parents
    for (auto& subg : wholeGraphTopology) {
        for (auto child : subg.outGraph) {
            subgrChild[subg.esgId].insert(child);
            subgrParent[child].insert(subg.esgId);
        }
    }

    // Define number of childs for each node
    for (auto& [vertex, childs] : subgrChild) {
        subgrChildNum[vertex] = childs.size();
    }

    // Define number of parents for each node
    for (auto& [vertex, parents] : subgrParent) {
        subgrParentNum[vertex] = parents.size();
    }
}

void FindSubgrDepth(std::map<int, int>& subgrChildNum, std::map<int, int>& subgrParentNum,
                    std::vector<int>& subgrDepthVector, Function& function)
{
    auto wholeGraphTopology = function.rootFunc_->topoInfo_.topology_;
    int wholeGraphTopologySize = int(wholeGraphTopology.size());

    // Create edges
    std::vector<std::pair<int, int>> edges;
    for (auto& subg : wholeGraphTopology) {
        for (auto child : subg.outGraph) {
            edges.push_back({child, subg.esgId});
        }
    }

    // Create lastVertices
    std::vector<int> lastVertices;
    for (auto& subg : wholeGraphTopology) {
        if (subgrChildNum[subg.esgId] == 0 && subgrParentNum[subg.esgId] != 0) {
            lastVertices.push_back(subg.esgId);
        }
    }

    // Define depth for each node using FordBellman algorithm
    for (size_t i = 0; i < lastVertices.size(); i++) {
        edges.push_back({wholeGraphTopologySize, lastVertices[i]});
    }
    auto d = FordBellman(wholeGraphTopologySize, edges);
    for (size_t idx = 0; idx < subgrDepthVector.size(); idx++) {
        subgrDepthVector[idx] = std::max(subgrDepthVector[idx], -d[idx] - 1);
    }
    edges.clear();
    lastVertices.clear();
}

void DefineLevels(std::map<int, int>& subgrChildNum, std::map<int, int>& subgrParentNum,
                  std::vector<int>& subgrDepthVector,
                  std::map<int, std::vector<std::pair<int, std::pair<int, int>>>>& levels, int& levelSize)
{
    // Define levels and deps inside level
    for (size_t idx = 0; idx < subgrDepthVector.size(); idx++) {
        if (subgrParentNum[idx] != 0) {
            levels[subgrDepthVector[idx]].push_back(
                std::make_pair(idx, std::make_pair(subgrChildNum[idx], subgrParentNum[idx])));
        }
    }

    // Processing ready-to-run last level nodes
    levelSize = levels.size();
    for (size_t idx = 0; idx < subgrDepthVector.size(); idx++) {
        if ((subgrParentNum[idx] == 0) && (subgrDepthVector[idx] == levelSize)) {
            levels[levelSize].push_back(std::make_pair(idx, std::make_pair(subgrChildNum[idx], subgrParentNum[idx])));
        }
    }
    subgrChildNum.clear();
    levelSize = levels.size();
}

void SortPriorities(std::map<int, std::set<int>>& subgrChild, std::map<int, std::set<int>>& subgrParent,
                    std::map<int, std::vector<std::pair<int, std::pair<int, int>>>>& levels,
                    std::map<int, std::map<int, std::pair<int, int>>>& levelsMap,
                    std::map<int, std::vector<std::pair<int, int>>>& priorities)
{
    std::map<int, std::vector<std::pair<int, std::pair<int, int>>>> levelsTmp;
    for (auto& level : levels) {
        if (level.first == 0) {
            continue;
        }
        int prior = 0;
        for (auto& priorTmp : priorities[level.first - 1]) {
            for (auto& elem : subgrParent[priorTmp.first]) {
                if (levelsMap[level.first].count(elem) > 0 && subgrChild.count(elem) > 0) {
                    levelsTmp[level.first].push_back(std::make_pair(
                        elem, std::make_pair(levelsMap[level.first][elem].first, levelsMap[level.first][elem].second)));
                    subgrChild.erase(elem);
                }
            }
            std::sort(levelsTmp[level.first].begin(), levelsTmp[level.first].end(),
                      [](const std::pair<int, std::pair<int, int>>& x, const std::pair<int, std::pair<int, int>>& y) {
                          return x.second.second > y.second.second;
                      }); // Sort by parents
            std::sort(levelsTmp[level.first].begin(), levelsTmp[level.first].end(),
                      [](const std::pair<int, std::pair<int, int>>& x, const std::pair<int, std::pair<int, int>>& y) {
                          return x.second.first > y.second.first;
                      }); // Sort by childs

            for (auto& elemPrior : levelsTmp[level.first]) {
                priorities[level.first].push_back(std::make_pair(elemPrior.first, prior));
                prior++;
            }
            levelsTmp.clear();
        }
        std::sort(priorities[level.first].begin(), priorities[level.first].end(),
                  [](const std::pair<int, int>& x, const std::pair<int, int>& y) {
                      return x.second < y.second;
                  }); // Sort by priorities
    }
}

void DefinePriorities(std::map<int, std::set<int>>& subgrChild, std::map<int, std::set<int>>& subgrParent,
                      std::map<int, std::vector<std::pair<int, std::pair<int, int>>>>& levels,
                      std::map<int, std::vector<std::pair<int, int>>>& priorities)
{
    // Define priority of nodes inside level-0
    std::sort(levels[0].begin(), levels[0].end(),
              [](const std::pair<int, std::pair<int, int>>& x, const std::pair<int, std::pair<int, int>>& y) {
                  return x.second.second > y.second.second;
              });
    int zeroLevelIndex = 0;
    for (auto& elem : levels[0]) {
        priorities[0].push_back(std::make_pair(elem.first, zeroLevelIndex));
        zeroLevelIndex++;
    }

    // Find priors for subgraphs on all levels
    std::map<int, std::map<int, std::pair<int, int>>> levelsMap;
    for (auto& [level, instance] : levels) {
        for (auto& elem : instance) {
            levelsMap[level][elem.first] = elem.second;
        }
    }
    SortPriorities(subgrChild, subgrParent, levels, levelsMap, priorities);
    subgrChild.clear();
    subgrParent.clear();
    levels.clear();
}

void SetAbsolutePrioritiesAndChangeOutGraphs(std::map<int, std::vector<std::pair<int, int>>>& priorities,
                                             Function& function)
{
    // Set absolute prioritties
    std::map<int, int> prioritiesNew;
    uint32_t factor = 100000;
    for (auto iter = priorities.begin(); iter != priorities.end(); ++iter) {
        for (auto& pair : iter->second) {
            prioritiesNew[pair.first] = iter->first * factor - pair.second;
        }
    }

    // Define additional structure with priorities
    std::map<int, std::vector<std::pair<int, int>>> subgOutGraphTmp;
    for (auto& subg : function.rootFunc_->topoInfo_.topology_) {
        for (auto& child : subg.outGraph) {
            subgOutGraphTmp[subg.esgId].push_back(std::make_pair(child, prioritiesNew[child]));
        }
    }
    prioritiesNew.clear();

    for (auto& elem : subgOutGraphTmp) {
        std::sort(subgOutGraphTmp[elem.first].begin(), subgOutGraphTmp[elem.first].end(),
                  [](const std::pair<int, int>& x, const std::pair<int, int>& y) {
                      return x.second < y.second;
                  }); // Sort by priorities
    }

    // Change outGrapgs of subgraphs
    setType outGraphTmp;
#ifdef PRIOR_SCHEDULING
    outGraphTmp.reserve(function.rootFunc_->topoInfo_.topology_.size());
#endif
    for (auto& subg : function.rootFunc_->topoInfo_.topology_) {
        if (subg.outGraph.size() <= 1) {
            continue;
        }
        outGraphTmp.clear();
        for (auto& child : subgOutGraphTmp[subg.esgId]) {
            outGraphTmp.insert(child.first);
        }
        subg.outGraph = outGraphTmp;
    }
    outGraphTmp.clear();
}

std::map<int, int> GetLastLevelMap(Function& function)
{
    std::map<int, int> LastLevelMap;
    const size_t aicValue = 0;
    const size_t aivValue = 1;
    const size_t aicpuValue = 2;
    for (size_t aic = 0; aic < function.rootFunc_->GetReadySubGraphCount(CoreType::AIC); aic++) {
        LastLevelMap[function.rootFunc_->GetReadySubGraphId(CoreType::AIC, aic)] = aicValue;
    }
    for (size_t aiv = 0; aiv < function.rootFunc_->GetReadySubGraphCount(CoreType::AIV); aiv++) {
        LastLevelMap[function.rootFunc_->GetReadySubGraphId(CoreType::AIV, aiv)] = aivValue;
    }
    for (size_t aicpu = 0; aicpu < function.rootFunc_->GetReadySubGraphCount(CoreType::AICPU); aicpu++) {
        LastLevelMap[function.rootFunc_->GetReadySubGraphId(CoreType::AICPU, aicpu)] = aicpuValue;
    }
    return LastLevelMap;
}

void UpdateReadySubGraphId(Function& function, const std::vector<std::pair<int, int>>& LastLevelVector)
{
    size_t aic = 0;
    size_t aiv = 0;
    size_t aicpu = 0;
    const size_t aicValue = 0;
    const size_t aivValue = 1;
    const size_t aicpuValue = 2;
    for (int i = 0; i < function.rootFunc_->GetAllReadySubGraphCount(); i++) {
        if (LastLevelVector[i].second == aicValue) {
            function.rootFunc_->ReplaceReadySubGraphIds(CoreType::AIC, aic, LastLevelVector[i].first);
            aic++;
        }
        if (LastLevelVector[i].second == aivValue) {
            function.rootFunc_->ReplaceReadySubGraphIds(CoreType::AIV, aiv, LastLevelVector[i].first);
            aiv++;
        }
        if (LastLevelVector[i].second == aicpuValue) {
            function.rootFunc_->ReplaceReadySubGraphIds(CoreType::AICPU, aicpu, LastLevelVector[i].first);
            aicpu++;
        }
    }
}

void ChangeReadyTasksPriorities(std::map<int, int>& subgrParentNum, std::vector<int>& subgrDepthVector,
                                std::map<int, std::vector<std::pair<int, int>>>& priorities, int levelSize,
                                Function& function)
{
    // Processing ready-to-run not last level nodes
    int lastLevelIntermediateSize = priorities[priorities.size() - 1].size();

    for (size_t idx = 0; idx < subgrDepthVector.size(); idx++) {
        if ((subgrParentNum[idx] == 0) && (subgrDepthVector[idx] != (levelSize - 1))) {
            priorities[priorities.size() - 1].push_back(std::make_pair(idx, lastLevelIntermediateSize));
            lastLevelIntermediateSize++;
        }
    }
    subgrParentNum.clear();
    subgrDepthVector.clear();

    // Change readyAic & readyAiv & readyAicpu priorities
    std::map<int, int> LastLevelMap = GetLastLevelMap(function);
    ASSERT(FunctionErr::FUNCTION_GRAPH_STRUCTURE, priorities[priorities.size() - 1].size() == LastLevelMap.size());

    size_t lastLevelSize = LastLevelMap.size();
    std::vector<std::pair<int, int>> LastLevelVector;
    for (size_t i = 0; i < lastLevelSize; i++) {
        LastLevelVector.push_back(std::make_pair(priorities[priorities.size() - 1][i].first,
                                                 LastLevelMap[priorities[priorities.size() - 1][i].first]));
    }
    UpdateReadySubGraphId(function, LastLevelVector);
}

void PriorScheduling::PriorSchedulingFunc(Function& function) const
{
    // Define necessary variables & data structures
    auto wholeGraphTopology = function.rootFunc_->topoInfo_.topology_;
    int wholeGraphTopologySize = int(wholeGraphTopology.size());
    int levelSize = 0;
    std::map<int, std::set<int>> subgrChild;
    std::map<int, std::set<int>> subgrParent;
    std::map<int, int> subgrChildNum;
    std::map<int, int> subgrParentNum;
    std::vector<int> subgrDepthVector(wholeGraphTopologySize, 0);
    std::map<int, std::vector<std::pair<int, std::pair<int, int>>>> levels;
    std::map<int, std::vector<std::pair<int, int>>> priorities;

    // Call functions
    FillGraphTopologyInfo(subgrChild, subgrParent, subgrChildNum, subgrParentNum, function);
    FindSubgrDepth(subgrChildNum, subgrParentNum, subgrDepthVector, function);
    DefineLevels(subgrChildNum, subgrParentNum, subgrDepthVector, levels, levelSize);
    DefinePriorities(subgrChild, subgrParent, levels, priorities);
    SetAbsolutePrioritiesAndChangeOutGraphs(priorities, function);
    ChangeReadyTasksPriorities(subgrParentNum, subgrDepthVector, priorities, levelSize, function);
}
} // namespace npu::tile_fwk
