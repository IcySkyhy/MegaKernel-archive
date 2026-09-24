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
 * \file static_subgraph_processor.h
 * \brief
 */

#ifndef PASS_STATIC_SUBGRAPH_PROCESSOR_H_
#define PASS_STATIC_SUBGRAPH_PROCESSOR_H_

#include <vector>
#include <map>
#include <memory>
#include <string>
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/platform.h"
#include "tilefwk/data_type.h"
#include "passes/pass_utils/subfunc_utils.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/statistics/execute_graph_statistic.h"
#include "passes/pass_log/pass_log.h"

#undef MODULE_NAME
#define MODULE_NAME "StaticSubgraphProcessor"

namespace npu::tile_fwk {

class StaticSubgraphProcessor {
public:
    StaticSubgraphProcessor() = default;
    ~StaticSubgraphProcessor() = default;

    // 静态流程专用函数
    // ESGGraphType相关方法
    Status CalOpCnt(size_t i, int32_t& cubeOpCnt, int32_t& vecOpCnt, int32_t& aicpuOpCnt);
    Status SetESGGraphType(int32_t cubeOpCnt, int32_t vecOpCnt, int32_t aicpuOpCnt, CoreType& esgGraphType);
    Status DetermineGraphType(size_t i, CoreType& esgGraphType);
    Status SetCallAttrGraphType(Function* rootFunc, size_t i, const CoreType& esgGraphType);

    Status HandleReadyStates(Function* rootFunc);
    Status BuildGraph(Function& function);
    Status BuildInGraph(Function& function);
    Status EdgeIndexCheck(const bool found, const int newIndex, const size_t graphSize) const;

    SubfuncTopologyInfoTy ConstructSubgraphTopologyInfo(Function& function,
                                                        std::vector<SubfuncInvokeInfoTy>& esgInvokeInfoMap);
    void UpdateTopoEntry(size_t i, int eSgId, int realOutDegree, const setType& succESgs, SubfuncTopologyInfoTy& topo);

    void SetColorGraph(size_t i, const OperationsViewer& list);
    void BuildColorGraph(Function& function);
    void PrintColorGraph(const Function& function);
    void ProcessColorGraph(Function& function);
    void FindRedundantEdges(int colorNum, std::vector<std::vector<int>>& redundantColorInGraph,
                            std::vector<std::vector<int>>& redundantColorOutGraph);
    void EraseRedundantColorEdges(const Function& function);
    Status SetReadySubGraphType(Function* rootFunc, size_t i, const CoreType& esgGraphType);
    void SetNList(std::vector<std::vector<OperationPtr>>& nList) { nLIST_ = &nList; }

    std::vector<std::vector<OperationPtr>>& GetNList()
    {
        if (nLIST_ == nullptr) {
            APASS_LOG_ERROR_F(Elements::Function, "nLIST is not initialized in StaticSubgraphProcessor");
        }
        return *nLIST_;
    }
    // 静态流程专有参数
    std::vector<std::vector<size_t>> inGraph;
    std::vector<std::vector<size_t>> outGraph;
    std::vector<bool> isReshape;
    std::vector<std::vector<int>> colorInGraph;
    std::vector<std::vector<int>> colorOutGraph;
    std::vector<int64_t> subgTopoParamOffsets;
    std::vector<std::vector<OperationPtr>>* nLIST_ = nullptr;
};
} // namespace npu::tile_fwk

#endif // PASS_STATIC_SUBGRAPH_PROCESSOR_H_
