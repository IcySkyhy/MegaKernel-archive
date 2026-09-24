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
 * \file pre_graph.cpp
 * \brief
 */

#include "pre_graph.h"
#include "passes/pass_check/pre_graph_checker.h"
#include "passes/pass_utils/merge_view_assemble_utils.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "PreGraphProcess"

namespace npu::tile_fwk {
void PreGraphProcess::UpdateCopyOpIsCube(Operation& op) const
{
    /*
    后续考虑移到InsertCopyOp
    */
    if (IsCopyIn(op.GetOpcode())) {
        if (op.GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            op.SetAttribute(OpAttributeKey::isCube, false);
        } else {
            op.SetAttribute(OpAttributeKey::isCube, true);
        }
    }

    if ((IsCopyOut(op.GetOpcode())) && (!OpcodeManager::Inst().IsSharedMemory(op.GetOpcode()))) {
        bool hasUbInput = false;
        for (const auto& ioperand : op.GetIOperands()) {
            if (ioperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                hasUbInput = true;
                break;
            }
        }
        if (hasUbInput) {
            op.SetAttribute(OpAttributeKey::isCube, false);
        } else {
            op.SetAttribute(OpAttributeKey::isCube, true);
        }
    }
}

enum class HubMergeType { AIV, AIC, HUB, NOMERGE };

static void MakeSubgraphIdContinue(Function& function)
{
    std::set<int> subgraphIds;
    for (auto& op : function.Operations()) {
        subgraphIds.insert(op.GetSubgraphID());
    }
    function.SetTotalSubGraphCount(subgraphIds.size());
    std::unordered_map<int, int> newIdMap;
    int newId = 0;
    for (auto id : subgraphIds) {
        newIdMap[id] = newId;
        newId++;
    }
    for (auto& op : function.Operations()) {
        op.UpdateSubgraphID(newIdMap[op.GetSubgraphID()]);
    }
}

static void DSUinit(std::vector<int>& parent, int num)
{
    parent.resize(num);
    for (int i = 0; i < num; i++) {
        parent[i] = i;
    }
}

static int DSUfind(std::vector<int>& parent, int i)
{
    if (parent[i] != i) {
        parent[i] = DSUfind(parent, parent[i]);
    }
    return parent[i];
}

static void DSUunite(std::vector<int>& parent, int i, int j)
{
    i = DSUfind(parent, i);
    j = DSUfind(parent, j);
    if (i < j) {
        parent[j] = i;
    } else if (j < i) {
        parent[i] = j;
    }
}

static std::vector<HubMergeType> MarkSubgraphType(Function& function)
{
    std::vector<bool> hasAIC(function.GetTotalSubGraphCount(), false);
    std::vector<bool> hasAIV(function.GetTotalSubGraphCount(), false);
    std::vector<bool> hasView(function.GetTotalSubGraphCount(), false);
    std::vector<bool> hasAssemble(function.GetTotalSubGraphCount(), false);
    std::vector<bool> hasOthers(function.GetTotalSubGraphCount(), false);
    std::vector<HubMergeType> subgraphTypes(function.GetTotalSubGraphCount());
    for (auto& op : function.Operations()) {
        int currSubgraphID = op.GetSubgraphID();
        if (op.HasAttr(OpAttributeKey::isCube) && op.GetBoolAttribute(OpAttributeKey::isCube)) {
            hasAIC[currSubgraphID] = true;
        } else if (op.HasAttr(OpAttributeKey::isCube) && !op.GetBoolAttribute(OpAttributeKey::isCube)) {
            hasAIV[currSubgraphID] = true;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            hasView[currSubgraphID] = true;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            hasAssemble[currSubgraphID] = true;
        }
    }
    for (int subgraphId = 0; subgraphId < static_cast<int>(function.GetTotalSubGraphCount()); subgraphId++) {
        if (hasAIC[subgraphId] && hasAIV[subgraphId]) {
            subgraphTypes[subgraphId] = HubMergeType::NOMERGE;
        } else if (hasAIC[subgraphId]) {
            subgraphTypes[subgraphId] = HubMergeType::AIC;
        } else {
            subgraphTypes[subgraphId] = HubMergeType::AIV;
        }
        if (hasOthers[subgraphId]) {
            continue;
        }
        if (hasView[subgraphId] && hasAssemble[subgraphId]) {
            subgraphTypes[subgraphId] = HubMergeType::NOMERGE;
        } else {
            subgraphTypes[subgraphId] = HubMergeType::HUB;
        }
    }
    return subgraphTypes;
}

static Status HubSpecialProcess(Function& function)
{
    std::vector<HubMergeType> subgraphTypes = MarkSubgraphType(function);
    std::unordered_map<int, bool> updateIsCube;

    std::vector<std::set<int>> subgraphInGraph(function.GetTotalSubGraphCount());
    std::vector<std::set<int>> subgraphOutGraph(function.GetTotalSubGraphCount());
    ColorGraph::BuildSubgraphDependencyGraph(function, subgraphInGraph, subgraphOutGraph);
    std::vector<int> parent;
    DSUinit(parent, function.GetTotalSubGraphCount());
    for (int subgraphId = 0; subgraphId < static_cast<int>(function.GetTotalSubGraphCount()); subgraphId++) {
        if (subgraphTypes[subgraphId] != HubMergeType::HUB) {
            continue;
        }
        int mergeCandidate = -1;
        if (subgraphInGraph[subgraphId].size() == 1) {
            mergeCandidate = *(subgraphInGraph[subgraphId].begin());
        }
        if (subgraphOutGraph[subgraphId].size() == 1) {
            mergeCandidate = *(subgraphOutGraph[subgraphId].begin());
        }
        if (mergeCandidate == -1 || subgraphTypes[mergeCandidate] == HubMergeType::HUB ||
            subgraphTypes[mergeCandidate] == HubMergeType::NOMERGE) {
            continue;
        }
        DSUunite(parent, subgraphId, mergeCandidate);
        updateIsCube[subgraphId] = subgraphTypes[mergeCandidate] == HubMergeType::AIC;
    }
    for (auto& op : function.Operations()) {
        int currSubgraphID = op.GetSubgraphID();
        op.UpdateSubgraphID(DSUfind(parent, currSubgraphID));
        if (updateIsCube.count(currSubgraphID) > 0) {
            op.SetAttribute(OpAttributeKey::isCube, updateIsCube[currSubgraphID]);
        }
    }
    MakeSubgraphIdContinue(function);
    return SUCCESS;
}

Status PreGraphProcess::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "===> combine the hub subgraph to a nearby subgraph.");
    if (function.GetTotalSubGraphCount() > 1) {
        HubSpecialProcess(function);
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> start PreGraph.");
    ColorGraph colorGraph;
    colorGraph.PreColorSort(function);
    auto opList = function.Operations();
    for (auto& op : opList) {
        colorGraph.InitializeTensorMem(op);
        UpdateCopyOpIsCube(op);
    }
    SetBoundary setBoundary;
    setBoundary.SetTensorBoundary(function);
    // Processing Special Ops
    SetCopyAttr setCopyAttr;
    RemoveRedundantAssemble removeRedundantAssemble;
    if (removeRedundantAssemble.ProcessViewToCopyIn(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessViewToCopyIn failed.");
        return FAILED;
    }
    for (auto& op : opList) {
        bool* distCopyType = op.GetAttr<bool>(OpAttributeKey::isDistCopyOut);
        if (IsCopyOut(op.GetOpcode()) && (op.GetOpAttribute() == nullptr) &&
            op.GetOpcode() != Opcode::OP_RESHAPE_COPY_OUT && !(distCopyType && *distCopyType)) {
            setCopyAttr.ProcessSpecialMTEOperation(op);
        }
        if (IsCopyIn(op.GetOpcode()) && (op.GetOpAttribute() == nullptr) &&
            op.GetOpcode() != Opcode::OP_RESHAPE_COPY_IN && !(distCopyType && !*distCopyType)) {
            setCopyAttr.ProcessMoveInOperation(op);
        }
    }
    if (removeRedundantAssemble.DeleteRedundantAssemble(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "DeleteRedundantAssemble failed.");
        return FAILED;
    }
    CubeProcess cubeProcess;
    if (cubeProcess.UpdateCubeOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Update Cube attr failed.");
        return FAILED;
    }
    Status status = MergeViewAssembleUtils::MergeViewAssemble(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Merge assemble and view failed.");
        return status;
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> End PreGraph.");
    return SUCCESS;
}

Status PreGraphProcess::PreCheck(Function& function)
{
    PreGraphProcessChecker checker;
    return checker.DoPreCheck(function);
}

Status PreGraphProcess::PostCheck(Function& function)
{
    PreGraphProcessChecker checker;
    return checker.DoPostCheck(function);
}
} // namespace npu::tile_fwk
