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
 * \file static_subgraph_processor.cpp
 * \brief
 */

#include "passes/tile_graph_pass/static_subgraph_processor.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "StaticSubgraphProcessor"

namespace npu::tile_fwk {

Status StaticSubgraphProcessor::BuildGraph(Function& function)
{
    auto operationViewer = function.Operations();
    inGraph.resize(operationViewer.size());
    outGraph.resize(operationViewer.size());
    if (BuildInGraph(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Build failed, Please check above for detailed inforamtion.");
        return FAILED;
    }
    for (size_t i = 0; i < operationViewer.size(); i++) {
        for (auto parentSeqNo : inGraph[i]) {
            outGraph[parentSeqNo].push_back(i);
        }
    }
    return SUCCESS;
}

Status StaticSubgraphProcessor::BuildInGraph(Function& function)
{
    auto operationViewer = function.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        inGraph[i].clear();
        outGraph[i].clear();
        // inGraph
        for (auto& inOperand : operationViewer[i].GetIOperands()) {
            for (auto& parentOp : inOperand->GetProducers()) {
                auto [parentSeqNo, found] = operationViewer.FindOpPosition(*parentOp);
                if (EdgeIndexCheck(found, parentSeqNo, inGraph.size()) != SUCCESS) {
                    APASS_LOG_ERROR_F(Elements::Operation,
                                      "Error inserting op magic %d in function %d %s to inGraph. %s",
                                      parentOp->GetOpMagic(), function.GetFuncMagic(), function.GetRawName().c_str(),
                                      GetFormatBacktrace(parentOp).c_str());
                    return FAILED;
                }
                inGraph[i].push_back(parentSeqNo);
            }
        }

        for (const auto& inControlOp : operationViewer[i].GetInCtrlOperations()) {
            auto [parentSeqNo, found] = operationViewer.FindOpPosition(*inControlOp);
            if (EdgeIndexCheck(found, parentSeqNo, inGraph.size()) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Error inserting op magic %d in function %d %s to inGraph. %s",
                                  inControlOp->GetOpMagic(), function.GetFuncMagic(), function.GetRawName().c_str(),
                                  GetFormatBacktrace(inControlOp).c_str());
                return FAILED;
            }
            inGraph[i].push_back(parentSeqNo);
        }
    }
    return SUCCESS;
}

Status StaticSubgraphProcessor::EdgeIndexCheck(const bool found, const int newIndex, const size_t graphSize) const
{
    if (!found) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op magic not found; please check if the input is correct.");
        return FAILED;
    }
    if (static_cast<size_t>(newIndex) >= graphSize) {
        APASS_LOG_ERROR_F(Elements::Operation, "Parent index %d is larger than operations_ size %zu.", newIndex,
                          graphSize);
        return FAILED;
    }
    return SUCCESS;
}

SubfuncTopologyInfoTy StaticSubgraphProcessor::ConstructSubgraphTopologyInfo(
    Function& function, std::vector<SubfuncInvokeInfoTy>& esgInvokeInfoMap)
{
    int maxOutDegree = 0;
    SubfuncTopologyInfoTy topo;
    topo.SetTableSize(esgInvokeInfoMap.size());
    subgTopoParamOffsets.emplace_back(0);
    BuildColorGraph(function);
    PrintColorGraph(function);
    EraseRedundantColorEdges(function);
    PrintColorGraph(function);
    APASS_LOG_INFO_F(Elements::Operation, "ColorOutGraph size %zu.", colorOutGraph.size());
    for (size_t i = 0; i < colorOutGraph.size(); i++) {
        setType succESgs;
        int eSgId = i;
        bool skip = isReshape[i];
        succESgs.clear();
        if (!skip) {
            for (size_t j = 0; j < colorOutGraph[i].size(); j++) {
                succESgs.insert(colorOutGraph[i][j]);
            }
        }
        maxOutDegree = static_cast<int>(succESgs.size()) > maxOutDegree ? static_cast<int>(succESgs.size()) :
                                                                          maxOutDegree;
        int realOutDegree = 0;
        for (auto& item : colorInGraph[i]) {
            if (!isReshape[item]) {
                realOutDegree++;
            }
        }
        UpdateTopoEntry(i, eSgId, realOutDegree, succESgs, topo);
        // Add subgTopoParamOffsets for simcpu
        int64_t offSize = sizeof(int64_t) +                                          // ProgramSubgraph Id size
                          sizeof(int64_t) +                                          // ReadyOrNot size
                          sizeof(int64_t) +                                          // outDependEsgIdList.size
                          sizeof(int64_t) * (static_cast<int64_t>(succESgs.size())); // outDependEsgIdList
        subgTopoParamOffsets.emplace_back(subgTopoParamOffsets.back() + offSize);
    }
    topo.SetMaxM(maxOutDegree);
    return topo;
}

void StaticSubgraphProcessor::BuildColorGraph(Function& function)
{
    colorInGraph = std::vector<std::vector<int>>(function.GetTotalSubGraphCount());
    colorOutGraph = std::vector<std::vector<int>>(function.GetTotalSubGraphCount());
    isReshape = std::vector<bool>(function.GetTotalSubGraphCount(), false);
    auto list = function.Operations();
    for (size_t i = 0; i < list.size(); i++) {
        if (list[i].GetSubgraphID() < 0) {
            continue;
        }
        SetColorGraph(i, list);
    }
    ProcessColorGraph(function);
}

void StaticSubgraphProcessor::PrintColorGraph(const Function& function)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Color Graph: ");
    for (size_t i = 0; i < function.GetTotalSubGraphCount(); i++) {
        APASS_LOG_DEBUG_F(Elements::Graph, "%zu: %zu, %zu.", i, colorInGraph[i].size(), colorOutGraph[i].size());
        APASS_LOG_DEBUG_F(Elements::Graph, "ColorInGraph: %s.", IntVecToStr(colorInGraph[i]).c_str());
        APASS_LOG_DEBUG_F(Elements::Graph, "ColorOutGraph: %s.", IntVecToStr(colorOutGraph[i]).c_str());
    }
    int inCount = 0, outCount = 0;
    for (size_t i = 0; i < function.GetTotalSubGraphCount(); i++) {
        inCount += colorInGraph[i].size();
        outCount += colorOutGraph[i].size();
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Total in: %d, total out: %d.", inCount, outCount);
}

inline void findAllReachableNodes(int start_node, std::vector<std::vector<int>>& outGraph,
                                  std::vector<std::unordered_set<int>>& reachable, std::vector<int>& visited)
{
    visited[start_node] = 1;
    reachable[start_node].insert(start_node);
    for (int v : outGraph[start_node]) {
        if (visited[v] == 0) {
            findAllReachableNodes(v, outGraph, reachable, visited);
        }
        reachable[start_node].insert(reachable[v].begin(), reachable[v].end());
    }
}

void StaticSubgraphProcessor::FindRedundantEdges(int colorNum, std::vector<std::vector<int>>& redundantColorInGraph,
                                                 std::vector<std::vector<int>>& redundantColorOutGraph)
{
    std::vector<std::unordered_set<int>> reachable(colorNum);
    std::vector<int> visited(colorNum, 0);
    for (int i = 0; i < colorNum; ++i) {
        if (visited[i] == 0) {
            findAllReachableNodes(i, colorOutGraph, reachable, visited); // DFS记忆化计算
        }
    }
    for (int u = 0; u < colorNum; ++u) {
        for (int v : colorOutGraph[u]) {
            bool is_redundant = false;
            for (int w : colorOutGraph[u]) {
                if (w == v) {
                    continue;
                }
                if (reachable[w].count(v)) {
                    is_redundant = true;
                    break;
                }
            }
            if (is_redundant) {
                redundantColorOutGraph[u].push_back(v);
                redundantColorInGraph[v].push_back(u);
            }
        }
    }
}

void StaticSubgraphProcessor::EraseRedundantColorEdges(const Function& function)
{
    size_t colorNum = function.GetTotalSubGraphCount();
    std::vector<std::vector<int>> redundantColorInGraph(colorNum), redundantColorOutGraph(colorNum);
    // Find redundant edges
    FindRedundantEdges(colorNum, redundantColorInGraph, redundantColorOutGraph);
    // Erase redundant edges
    for (size_t i = 0; i < colorNum; i++) {
        std::sort(redundantColorOutGraph[i].begin(), redundantColorOutGraph[i].end());
        APASS_LOG_INFO_F(Elements::Operation, "Redundant outgraph of %zu is %s.", i,
                         IntVecToStr(redundantColorOutGraph[i]).c_str());
        std::vector<int> newGraph;
        // update color_in_graph
        size_t j = 0U;
        for (int k : redundantColorInGraph[i]) {
            while (colorInGraph[i][j] != k) {
                newGraph.push_back(colorInGraph[i][j]);
                j++;
            }
            j++;
        }
        while (j < colorInGraph[i].size()) {
            newGraph.push_back(colorInGraph[i][j]);
            j++;
        }
        colorInGraph[i] = newGraph;
        // update color_out_graph
        newGraph.clear();
        j = 0;
        for (int k : redundantColorOutGraph[i]) {
            while (colorOutGraph[i][j] != k) {
                newGraph.push_back(colorOutGraph[i][j]);
                j++;
            }
            j++;
        }
        while (j < colorOutGraph[i].size()) {
            newGraph.push_back(colorOutGraph[i][j]);
            j++;
        }
        colorOutGraph[i] = newGraph;
    }
}

void StaticSubgraphProcessor::SetColorGraph(size_t i, const OperationsViewer& list)
{
    for (int j : outGraph[i]) {
        if (list[i].GetSubgraphID() != list[j].GetSubgraphID()) {
            if (list[j].GetSubgraphID() < 0) {
                continue;
            }
            colorOutGraph[list[i].GetSubgraphID()].push_back(list[j].GetSubgraphID());
            colorInGraph[list[j].GetSubgraphID()].push_back(list[i].GetSubgraphID());
        }
    }
}

void StaticSubgraphProcessor::ProcessColorGraph(Function& function)
{
    for (size_t i = 0; i < function.GetTotalSubGraphCount(); i++) {
        auto& nList = GetNList();
        if (nList[i].size() == 1UL && colorInGraph[i].size() == 0 && nList[i][0]->GetOpcode() == Opcode::OP_RESHAPE) {
            isReshape[i] = true;
        }
        std::sort(colorInGraph[i].begin(), colorInGraph[i].end());
        colorInGraph[i].resize(std::unique(colorInGraph[i].begin(), colorInGraph[i].end()) - colorInGraph[i].begin());

        std::sort(colorOutGraph[i].begin(), colorOutGraph[i].end());
        colorOutGraph[i].resize(std::unique(colorOutGraph[i].begin(), colorOutGraph[i].end()) -
                                colorOutGraph[i].begin());
    }
}

Status StaticSubgraphProcessor::SetReadySubGraphType(Function* rootFunc, size_t i, const CoreType& esgGraphType)
{
    // Verify topology index is valid
    if (i >= rootFunc->topoInfo_.topology_.size()) {
        APASS_LOG_ERROR_F(Elements::Function, "Topology index %zu out of bounds (total topology entries: %zu).", i,
                          rootFunc->topoInfo_.topology_.size());
        return FAILED;
    }
    if (rootFunc->topoInfo_.topology_[i].readyState != 0) {
        return SUCCESS;
    }
    if (esgGraphType == CoreType::AIC) {
        rootFunc->EmplaceReadySubGraphIds(CoreType::AIC, i);
        APASS_LOG_DEBUG_F(Elements::Graph, "Esg %zu is ready aic sub graph.", i);
        return SUCCESS;
    }
    if (esgGraphType == CoreType::AIV) {
        rootFunc->EmplaceReadySubGraphIds(CoreType::AIV, i);
        APASS_LOG_DEBUG_F(Elements::Graph, "Esg %zu is ready aiv sub graph.", i);
        return SUCCESS;
    }
    if (esgGraphType == CoreType::AICPU) {
        rootFunc->EmplaceReadySubGraphIds(CoreType::AICPU, i);
        APASS_LOG_DEBUG_F(Elements::Graph, "Esg %zu is ready aicpu sub graph.", i);
        return SUCCESS;
    }
    if (esgGraphType == CoreType::MIX) {
        APASS_LOG_DEBUG_F(Elements::Graph, "Esg %zu is ready mix sub graph.", i);
    }
    return SUCCESS;
}

void StaticSubgraphProcessor::UpdateTopoEntry(size_t i, int eSgId, int realOutDegree, const setType& succESgs,
                                              SubfuncTopologyInfoTy& topo)
{
    int readyOrNot = -1 * realOutDegree;
    topo.AddEntry(eSgId, readyOrNot, succESgs);
    auto& nList = GetNList();
    if ((nList[i].size() == 1UL) && (nList[i][0]->GetCoreType() == CoreType::AICPU)) {
        auto& op = nList[i][0];
        const std::string extParamKey = OP_ATTR_PREFIX + "distributed";
        if (op->HasAttr(extParamKey)) {
            std::vector<int64_t> extParams = op->GetVectorIntAttribute(extParamKey);
            topo.UpdateEntry(static_cast<uint32_t>(op->GetOpcode()), extParams.size(), extParams);
            APASS_LOG_DEBUG_F(Elements::Graph, "UpdateEntry size=%lu.", extParams.size());
        }
    }
    APASS_LOG_DEBUG_F(Elements::Graph, "AddEntry ESgId %d ReadyOrNot %d %zu %d.", eSgId, readyOrNot,
                      topo.readyIds_.size(), topo.readyIds_[0]);
}

Status StaticSubgraphProcessor::SetESGGraphType(int32_t cubeOpCnt, int32_t vecOpCnt, int32_t aicpuOpCnt,
                                                CoreType& esgGraphType)
{
    if (aicpuOpCnt > 0) {
        esgGraphType = CoreType::AICPU;
        return SUCCESS;
    }
    if (cubeOpCnt == 0 && vecOpCnt > 0) {
        esgGraphType = CoreType::AIV;
        return SUCCESS;
    }
    if (cubeOpCnt > 0 && vecOpCnt == 0) {
        esgGraphType = CoreType::AIC;
        return SUCCESS;
    }
    if (cubeOpCnt <= 0 || vecOpCnt <= 0) {
        return SUCCESS;
    }
    esgGraphType = CoreType::MIX;
    if (!GraphUtils::IsCVMixPlatform()) {
        APASS_LOG_ERROR_F(Elements::Graph, "Get CoreType::MIX in C-V separate platform.");
        return FAILED;
    }
    return SUCCESS;
}

Status StaticSubgraphProcessor::DetermineGraphType(size_t i, CoreType& esgGraphType)
{
    int32_t cubeOpCnt = 0;
    int32_t vecOpCnt = 0;
    int32_t aicpuOpCnt = 0;
    if (CalOpCnt(i, cubeOpCnt, vecOpCnt, aicpuOpCnt) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Graph, "CalOpCnt failed.");
        return FAILED;
    }
    if (SetESGGraphType(cubeOpCnt, vecOpCnt, aicpuOpCnt, esgGraphType) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Graph, "SetESGGraphType failed.");
        return FAILED;
    }
    if (GetNList()[i].size() == 1 && GetNList()[i][0]->GetOpcode() == Opcode::OP_RESHAPE &&
        colorInGraph[i].size() != 0) {
        esgGraphType = CoreType::HUB;
    }
    return SUCCESS;
}

Status StaticSubgraphProcessor::SetCallAttrGraphType(Function* rootFunc, size_t i, const CoreType& esgGraphType)
{
    // Get the operation and verify it exists
    if (i >= rootFunc->Operations().size()) {
        APASS_LOG_ERROR_F(Elements::Function, "Operation index %zu out of bounds (total operations: %zu).", i,
                          rootFunc->Operations().size());
        return FAILED;
    }
    auto& op = rootFunc->Operations()[i];
    auto callAttr = dynamic_cast<CallOpAttribute*>(op.GetOpAttribute().get());
    if (callAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to get CallOpAttribute for operation %zu (opcode: %s). %s", i,
                          op.GetOpcodeStr().c_str(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    callAttr->invokeInfo_->SetGraphType(esgGraphType);
    return SUCCESS;
}

bool IsCubeOp(Operation& op)
{
    if (op.GetBoolAttribute(OpAttributeKey::isCube)) {
        return true;
    }
    if ((op.GetOpcode() == Opcode::OP_L0C_COPY_OUT) || (op.GetOpcode() == Opcode::OP_L1_COPY_IN)) {
        return true;
    }
    return false;
}

bool IsAICPUOp(Operation& op)
{
    if ((op.GetCoreType() == CoreType::AICPU)) {
        return true;
    }
    return false;
}

Status StaticSubgraphProcessor::CalOpCnt(size_t i, int32_t& cubeOpCnt, int32_t& vecOpCnt, int32_t& aicpuOpCnt)
{
    auto& nList = GetNList();
    for (size_t j = 0; j < nList[i].size(); j++) {
        if (IsCubeOp(*nList[i][j])) {
            cubeOpCnt += 1;
            continue;
        }
        if (IsAICPUOp(*nList[i][j])) {
            aicpuOpCnt += 1;
            continue;
        }
        vecOpCnt += 1;
    }
    return SUCCESS;
}

Status StaticSubgraphProcessor::HandleReadyStates(Function* rootFunc)
{
    if (rootFunc == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Root function is nullptr.");
        return FAILED;
    }
    auto& nList = GetNList();
    for (size_t i = 0; i < nList.size(); i++) {
        CoreType esgGraphType = CoreType::AIV;
        if (DetermineGraphType(i, esgGraphType) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "DetermineGraphType failed.");
            return FAILED;
        }
        if (SetCallAttrGraphType(rootFunc, i, esgGraphType) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "SetCallAttrGraphType failed.");
            return FAILED;
        }
        if (SetReadySubGraphType(rootFunc, i, esgGraphType) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "SetReadySubGraphType failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
