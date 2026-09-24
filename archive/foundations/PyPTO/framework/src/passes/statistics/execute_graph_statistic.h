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
 * \file execute_graph_statistic.h
 * \brief Execution graph analysis and statistic reporting
 */
#ifndef EXECUTE_GRAPH_STATISTIC_H
#define EXECUTE_GRAPH_STATISTIC_H

#include <climits>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "interface/function/function.h"

using json = nlohmann::json;

namespace npu::tile_fwk {

struct PathResult {
    int maxLength = 0; // Maximum path length found
};

struct ConcurrencyStats {
    int maxConcurrency = 0; // Maximum concurrent operations found
};

struct DependencyStats {
    size_t total_predecessors = 0;
    size_t total_successors = 0;
    int min_predecessors = INT_MAX;
    int max_predecessors = 0;
    int min_successors = INT_MAX;
    int max_successors = 0;
    size_t valid_entries = 0;
    std::vector<int> min_pred_nodes;
    std::vector<int> max_pred_nodes;
    std::vector<int> min_succ_nodes;
    std::vector<int> max_succ_nodes;
};

struct MinMaxStats {
    int min_value;
    int max_value;
    std::vector<int> min_nodes;
    std::vector<int> max_nodes;
};

class ExecutionGraphStatistic {
public:
    json AnalyzeExecutionGraph(Function& func, const std::multimap<int, int>& psgToESgMap,
                               const std::vector<std::vector<OperationPtr>>& subgraphGroups);
    void AnalyzeIsomorphism(json& report, const std::multimap<int, int>& psgToESgMap,
                            const std::vector<std::vector<OperationPtr>>& subgraphGroups);

private:
    uint64_t AnalyzePeakMemoryUsage(Function* rootFunc, std::vector<int>& peakMemoryUsageSubgraphs);
    uint64_t CalculateOperationMemory(const Operation& op);
    std::unordered_map<CoreType, int> CountCoreTypes(Function* rootFunc);
    uint64_t AnalyzeSubgraphLatencies(Function& func, uint64_t& maxLatency, uint64_t& minLatency,
                                      std::vector<int>& maxLatencySubgraphs, std::vector<int>& minLatencySubgraphs);
    PathResult FindLongestPath(Function& func);
    ConcurrencyStats CalculateConcurrency(Function& func);
    json AnalyzeGraphDependencies(Function& func);
    void UpdateMinMaxStats(int count, int esgId, MinMaxStats& stats);
    json FormatDependencyStats(const DependencyStats& stats);
    double FormatUsageRate(double value);
};
} // namespace npu::tile_fwk

#endif // EXECUTE_GRAPH_STATISTIC_H
