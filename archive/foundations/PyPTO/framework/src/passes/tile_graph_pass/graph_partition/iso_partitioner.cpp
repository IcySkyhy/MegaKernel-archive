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
 * \file iso_partitioner.cpp
 * \brief
 */

#include "iso_partitioner.h"
#include <iostream>
#include <deque>
#include <algorithm>
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/parallel_tool.h"
#include "passes/pass_check/iso_partitioner_checker.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "GraphPartition"

namespace npu::tile_fwk {

bool IsoPartitioner::HandleLiteNPU(Function& function)
{
    if (!IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
        return false;
    }
    for (auto& op : function.Operations(false)) {
        op.UpdateSubgraphID(0);
    }
    function.SetTotalSubGraphCount(1);
    APASS_LOG_INFO_F(Elements::Operation, "Graph Partition is skipped.");
    return true;
}

Status IsoPartitioner::RunPartitionSteps(Function& function)
{
    if (EstimateCycleUB(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Config, "Estimate and refresh cycleUB_ failed.");
        return FAILED;
    }
    if (cycleUB_ == -1 || parallelNum_ == -1 || cycleLB_ == -1) {
        APASS_LOG_ERROR_F(Elements::Config, "Partition parameters not initialized.");
        return FAILED;
    }
    if (BuildOpGraph(function.Operations().DuplicatedOpList()) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Partition the computational graph failed in building operation graph.");
        return FAILED;
    }
    if (BuildSuperNodeGraph() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Partition the computational graph failed in building SuperNode graph.");
        return FAILED;
    }
    if (BuildHashValues() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function,
                          "Partition the computational graph failed in building SuperNode hash values.");
        return FAILED;
    }
    if (BuildIsomorphismGroups() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Partition the computational graph failed in partitioning the graph.");
        return FAILED;
    }
    if (IsomorphismGroupMergeProcess(true) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function,
                          "Partition the computational graph failed in merging non-isomorphism groups.");
        return FAILED;
    }
    if (IsomorphismGroupMergeProcess(false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function,
                          "Partition the computational graph failed in merging isomorphism groups.");
        return FAILED;
    }
    if (UpdatePartitionResult(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Partition the computational graph failed in updating the Function.");
        return FAILED;
    }
    return SUCCESS;
}

Status IsoPartitioner::PartitionGraph(Function& function)
{
    if (HandleLiteNPU(function)) {
        return SUCCESS;
    }
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    return RunPartitionSteps(function);
}

Status IsoPartitioner::BuildIsomorphismGroups()
{
    std::vector<int32_t> idxInLinkNum;
    std::deque<int32_t> zeroInQueue;
    std::unordered_set<int32_t> currentNodeSet;
    for (size_t i = 0; i < superNodeInfo_->nodeInGraphList_.size(); i++) {
        idxInLinkNum.push_back(superNodeInfo_->nodeInGraphList_[i].size());
        if (superNodeInfo_->nodeInGraphList_[i].size() == 0) {
            zeroInQueue.push_front(i);
        }
    }
    currentNodeSet.clear();
    while (zeroInQueue.size() > 0) {
        int32_t currIdx = zeroInQueue[0];
        zeroInQueue.pop_front();
        uint64_t hs = superNodeInfo_->nodeHashList_[currIdx];
        std::vector<int32_t>& expandCandidate = superNodeInfo_->hash2NodeMap_[hs];
        bool isLegalStart = true;
        for (size_t i = 0; i < expandCandidate.size(); i++) {
            if (idxInLinkNum[expandCandidate[i]] != 0 || currentNodeSet.count(expandCandidate[i]) > 0) {
                isLegalStart = false;
                break;
            }
        }
        if (!isLegalStart) {
            continue;
        }
        std::shared_ptr<IsomorphismGraphGroup> currentGraphGroup = std::make_shared<IsomorphismGraphGroup>();
        if (currentGraphGroup == nullptr) {
            APASS_LOG_ERROR_F(Elements::Function, "Create current IsomorphismGraphGroup failed.");
            return FAILED;
        }
        if (currentGraphGroup->BuildGraphGroup(operationInfo_, superNodeInfo_, expandCandidate, currentNodeSet,
                                               idxInLinkNum, zeroInQueue) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Build initial IsomorphismGraphGroup failed.");
            return FAILED;
        }
        if (currentGraphGroup->GetMergeable()) {
            if (currentGraphGroup->ExpandIsoGraphs(currentNodeSet, idxInLinkNum, zeroInQueue, cycleUB_) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "Expand the isomorphism group failed.");
                return FAILED;
            }
        }
        isoSubGroups_.push_back(currentGraphGroup);
    }
    return SUCCESS;
}

Status IsomorphismGraphGroup::BuildGraphGroup(std::shared_ptr<OperationGraphInfo> operationInfo,
                                              std::shared_ptr<NodeGraphInfo> superNodeInfo,
                                              std::vector<int32_t>& expandCandidate,
                                              std::unordered_set<int32_t>& currentNodeSet,
                                              std::vector<int32_t>& idxInLinkNum, std::deque<int32_t>& zeroInQueue)
{
    operationInfo_ = operationInfo;
    superNodeInfo_ = superNodeInfo;
    subVisitedNodeSet_.clear();
    subVisitedNodeSet_.insert(expandCandidate.begin(), expandCandidate.end());
    currentNodeSet.insert(expandCandidate.begin(), expandCandidate.end());
    for (int32_t nodeIdx : expandCandidate) {
        if (InLinkCountDelete(nodeIdx, idxInLinkNum, zeroInQueue) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "In-link count delete failed.");
            return FAILED;
        }
        std::shared_ptr<SubGraph> sgPtr = std::make_shared<SubGraph>(operationInfo, superNodeInfo);
        if (sgPtr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Function, "Create SubGraph failed.");
            return FAILED;
        }
        sgPtr->AddNode(nodeIdx);
        sgPtr->scopeId_ = superNodeInfo->nodeScope_[nodeIdx].scopeId;
        sgPtr->SetAllowCrossScopeMerge(superNodeInfo->nodeScope_[nodeIdx].allowCrossScopeMerge);
        isoGraphs_.push_back(sgPtr);
    }
    mergeable_ = superNodeInfo_->nodeMergeable_[expandCandidate[0]];
    return SUCCESS;
}

Status IsomorphismGraphGroup::InLinkCountDelete(int32_t nodeIdx, std::vector<int32_t>& idxInLinkNum,
                                                std::deque<int32_t>& zeroInQueue)
{
    for (int32_t consumer : superNodeInfo_->nodeOutGraphList_[nodeIdx]) {
        if (consumer < 0 || consumer >= static_cast<int32_t>(idxInLinkNum.size())) {
            APASS_LOG_ERROR_F(Elements::Operation, "Consumer index(%d) illegal in InLinkCountDelete.", consumer);
            return FAILED;
        }
        idxInLinkNum[consumer] -= 1;
        if (idxInLinkNum[consumer] == 0) {
            zeroInQueue.push_back(consumer);
        }
        if (idxInLinkNum[consumer] < 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "Negative in-link count in InLinkCountDelete.");
            return FAILED;
        }
    }
    return SUCCESS;
}

size_t IsomorphismGraphGroup::Size() const { return isoGraphs_.size(); }

bool IsomorphismGraphGroup::GetMergeable() { return mergeable_ != 0; }

void SubGraph::AddNode(int32_t nodeIdx)
{
    if (coreType_ == OpCoreType::ANY) {
        coreType_ = superNodeInfo_->nodeCoreType_[nodeIdx];
    }
    nodeList_.push_back(nodeIdx);
    nodeSet_.insert(nodeIdx);
    cycle_ += superNodeInfo_->GetNodeCycle(nodeIdx);
}

Status IsomorphismGraphGroup::ExpandIsoGraphs(std::unordered_set<int32_t>& currentNodeSet,
                                              std::vector<int32_t>& idxInLinkNum, std::deque<int32_t>& zeroInQueue,
                                              int32_t pgUpperBound)
{
    size_t expandNodeIdx = 0;
    size_t expandLinkIdx = 0;
    GraphExtendResult extendStatus = GraphExtendResult::EXTEND_SUCCESS;
    while (extendStatus != GraphExtendResult::EXTEND_NODE_EXHAUST) {
        std::vector<int32_t> expandCandidate;
        for (size_t i = 0; i < isoGraphs_.size(); i++) {
            int32_t newLeaf = isoGraphs_[i]->GetExpandCandidate(expandNodeIdx, expandLinkIdx, extendStatus);
            expandCandidate.push_back(newLeaf);
        }
        if (extendStatus == GraphExtendResult::EXTEND_LINK_EXHAUST) {
            expandNodeIdx += 1;
            expandLinkIdx = 0;
            continue;
        }
        if (extendStatus == GraphExtendResult::EXTEND_NODE_EXHAUST) {
            break;
        }
        if (!IsLegalIsoGraphExtender(expandCandidate, currentNodeSet, idxInLinkNum, pgUpperBound)) {
            expandLinkIdx += 1;
            continue;
        }
        expandLinkIdx += 1;
        for (size_t i = 0; i < expandCandidate.size(); i++) {
            isoGraphs_[i]->AddNode(expandCandidate[i]);
            currentNodeSet.insert(expandCandidate[i]);
            subVisitedNodeSet_.insert(expandCandidate[i]);
            if (InLinkCountDelete(expandCandidate[i], idxInLinkNum, zeroInQueue) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "In-link count delete failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

bool SubGraph::HasNode(int32_t nodeIdx) const
{
    if (nodeSet_.count(nodeIdx) > 0) {
        return true;
    }
    return false;
}

bool IsomorphismGraphGroup::IsLegalIsoGraphExtender(std::vector<int32_t>& expandCandidate,
                                                    std::unordered_set<int32_t>& currentNodeSet,
                                                    std::vector<int32_t>& idxInLinkNum, int32_t pgUpperBound)
{
    if (!superNodeInfo_->nodeMergeable_[expandCandidate[0]]) {
        return false;
    }
    if (superNodeInfo_->hash2NodeMap_[superNodeInfo_->nodeHashList_[expandCandidate[0]]].size() != isoGraphs_.size()) {
        return false;
    }
    for (size_t i = 0; i < expandCandidate.size(); i++) {
        int32_t candidate = expandCandidate[i];
        if (idxInLinkNum[candidate] != 0) {
            return false;
        }
        if (superNodeInfo_->nodeHashList_[candidate] != superNodeInfo_->nodeHashList_[expandCandidate[0]]) {
            return false;
        }
        if (currentNodeSet.count(candidate) > 0) {
            return false;
        }
        for (int32_t fromNode : superNodeInfo_->nodeInGraphList_[candidate]) {
            if (subVisitedNodeSet_.count(fromNode) > 0 && !isoGraphs_[i]->HasNode(fromNode)) {
                return false;
            }
        }
    }
    int32_t newLatency = isoGraphs_[0]->GetLatency() + superNodeInfo_->GetNodeCycle(expandCandidate[0]);
    if (newLatency > pgUpperBound) {
        return false;
    }
    for (size_t i = 0; i < expandCandidate.size(); i++) {
        int origScopeId = isoGraphs_[i]->scopeId_;
        int mergeScopeId = superNodeInfo_->nodeScope_[expandCandidate[i]].scopeId;
        if (origScopeId != mergeScopeId) {
            APASS_LOG_INFO_F(Elements::Operation, "Cannot merge supernodes with different scopeId %d and %d.",
                             origScopeId, mergeScopeId);
            return false;
        }
    }
    std::unordered_set<int32_t> candSet(expandCandidate.begin(), expandCandidate.end());
    if (candSet.size() != expandCandidate.size()) {
        return false;
    }
    OpCoreType graphIsCube = superNodeInfo_->nodeCoreType_[isoGraphs_[0]->GetNodeList()[0]];
    OpCoreType candidateIsCube = superNodeInfo_->nodeCoreType_[expandCandidate[0]];
    std::set<OpCoreType> coreTypes{graphIsCube, candidateIsCube};
    return operationInfo_->CoreTypeMergeable(coreTypes);
}

int32_t SubGraph::GetExpandCandidate(size_t expandNodeIdx, size_t expandLinkIdx, GraphExtendResult& res)
{
    if (expandNodeIdx >= nodeList_.size()) {
        res = GraphExtendResult::EXTEND_NODE_EXHAUST;
        return 0;
    }
    if (expandLinkIdx >= superNodeInfo_->nodeOutGraphList_[nodeList_[expandNodeIdx]].size()) {
        res = GraphExtendResult::EXTEND_LINK_EXHAUST;
        return 0;
    }
    res = GraphExtendResult::EXTEND_SUCCESS;
    int32_t value = superNodeInfo_->nodeOutGraphList_[nodeList_[expandNodeIdx]][expandLinkIdx];
    return value;
}

const std::vector<int32_t>& SubGraph::GetNodeList() { return nodeList_; }

void SubGraph::BuildInOutSet()
{
    inNodes_.clear();
    outNodes_.clear();
    for (int32_t nodeIdx : nodeList_) {
        for (int32_t inIdx : superNodeInfo_->nodeInGraphList_[nodeIdx]) {
            if (nodeSet_.count(inIdx) == 0) {
                inNodes_.push_back(inIdx);
            }
        }
        for (int32_t outIdx : superNodeInfo_->nodeOutGraphList_[nodeIdx]) {
            if (nodeSet_.count(outIdx) == 0) {
                outNodes_.push_back(outIdx);
            }
        }
    }
    std::sort(inNodes_.begin(), inNodes_.end());
    inNodes_.erase(std::unique(inNodes_.begin(), inNodes_.end()), inNodes_.end());
    std::sort(outNodes_.begin(), outNodes_.end());
    outNodes_.erase(std::unique(outNodes_.begin(), outNodes_.end()), outNodes_.end());
}

std::shared_ptr<SubGraph> IsomorphismGraphGroup::GetSubGraph(int32_t idx)
{
    if (idx < 0 || idx >= static_cast<int32_t>(isoGraphs_.size())) {
        return nullptr;
    }
    return isoGraphs_[idx];
}

int32_t SubGraph::GetLatency() const { return cycle_; }

int32_t IsomorphismGraphGroup::GetLatency() const
{
    if (isoGraphs_.size() == 0) {
        return 0;
    }
    return isoGraphs_[0]->GetLatency();
}

void IsomorphismGraphGroup::Clear()
{
    isoGraphs_.clear();
    mergeable_ = true;
    operationInfo_ = nullptr;
    superNodeInfo_ = nullptr;
}

Status IsoPartitioner::IsomorphismGroupMergePrepare(std::vector<std::pair<int32_t, int32_t>>& isoSubIdxs,
                                                    std::vector<std::unordered_set<int32_t>>& isoInGraph,
                                                    std::vector<std::unordered_set<int32_t>>& isoOutGraph,
                                                    std::vector<std::vector<int32_t>>& isoNodeList,
                                                    std::vector<int32_t>& isoIdx2color)
{
    isoSubIdxs.resize(superNodeInfo_->nodeInGraphList_.size());
    isoInGraph.resize(isoSubGroups_.size());
    isoOutGraph.resize(isoSubGroups_.size());
    for (size_t i = 0; i < isoSubGroups_.size(); i++) {
        for (size_t j = 0; j < isoSubGroups_[i]->Size(); j++) {
            for (int32_t nodeIdx : isoSubGroups_[i]->GetSubGraph(j)->GetNodeList()) {
                if (nodeIdx < 0 || static_cast<size_t>(nodeIdx) >= superNodeInfo_->nodeInGraphList_.size()) {
                    APASS_LOG_ERROR_F(Elements::Operation, "NodeIdx illegal in IsomorphismGroupMergePrepare.");
                    return FAILED;
                }
                isoSubIdxs[nodeIdx] = std::pair<int32_t, int32_t>{i, j};
            }
            isoSubGroups_[i]->GetSubGraph(j)->BuildInOutSet();
        }
    }
    for (size_t i = 0; i < isoSubGroups_.size(); i++) {
        for (size_t j = 0; j < isoSubGroups_[i]->Size(); j++) {
            isoSubGroups_[i]->GetSubGraph(j)->mergeHistoryIsoSub_.insert(std::pair<int32_t, int32_t>{i, j});
            for (int32_t nodeIdx : isoSubGroups_[i]->GetSubGraph(j)->inNodes_) {
                if (nodeIdx < 0 || static_cast<size_t>(nodeIdx) >= superNodeInfo_->nodeInGraphList_.size()) {
                    APASS_LOG_ERROR_F(Elements::Operation, "NodeIdx illegal in IsomorphismGroupMergePrepare.");
                    return FAILED;
                }
                if (isoSubIdxs[nodeIdx].first != static_cast<int32_t>(i)) {
                    isoInGraph[i].insert(isoSubIdxs[nodeIdx].first);
                }
            }
            for (int32_t nodeIdx : isoSubGroups_[i]->GetSubGraph(j)->outNodes_) {
                if (nodeIdx < 0 || static_cast<size_t>(nodeIdx) >= superNodeInfo_->nodeInGraphList_.size()) {
                    APASS_LOG_ERROR_F(Elements::Operation, "NodeIdx illegal in IsomorphismGroupMergePrepare.");
                    return FAILED;
                }
                if (isoSubIdxs[nodeIdx].first != static_cast<int32_t>(i)) {
                    isoOutGraph[i].insert(isoSubIdxs[nodeIdx].first);
                }
            }
        }
    }
    isoNodeList.resize(isoSubGroups_.size());
    isoIdx2color.resize(isoSubGroups_.size());
    for (int32_t i = 0; i < static_cast<int32_t>(isoSubGroups_.size()); i++) {
        isoIdx2color[i] = i;
        isoNodeList[i].push_back(i);
    }
    return SUCCESS;
}

std::vector<int32_t> IsoPartitioner::GetCandidateMergeColors(int32_t currColor,
                                                             std::vector<std::unordered_set<int32_t>>& isoInGraph,
                                                             std::vector<std::unordered_set<int32_t>>& isoOutGraph,
                                                             std::vector<std::vector<int32_t>>& isoNodeList,
                                                             std::vector<int32_t>& isoIdx2color, bool nonIsoGraphsMerge)
{
    std::unordered_set<int32_t> inputColors;
    std::unordered_set<int32_t> selfNodes;
    std::unordered_set<int32_t> outputColors;
    if (!isoSubGroups_[currColor]->mergeable_) {
        return {};
    }
    if (nonIsoGraphsMerge && isoSubGroups_[currColor]->Size() != 1) {
        return {};
    }
    if (!nonIsoGraphsMerge && isoSubGroups_[currColor]->Size() <= 1) {
        return {};
    }
    selfNodes.insert(isoNodeList[currColor].begin(), isoNodeList[currColor].end());
    for (size_t idx : isoNodeList[currColor]) {
        for (size_t inIdx : isoInGraph[idx]) {
            inputColors.insert(isoIdx2color[inIdx]);
        }
        for (size_t outIdx : isoOutGraph[idx]) {
            outputColors.insert(isoIdx2color[outIdx]);
        }
    }
    inputColors.erase(currColor);
    outputColors.erase(currColor);
    std::vector<int32_t> candidateMergeColors;
    if (inputColors.size() == 1 && isoSubGroups_[*inputColors.begin()]->mergeable_) {
        candidateMergeColors.push_back(*inputColors.begin());
    }
    if (outputColors.size() == 1 && isoSubGroups_[*outputColors.begin()]->mergeable_) {
        candidateMergeColors.push_back(*outputColors.begin());
    }
    std::vector<int32_t> mergeColors;
    for (int32_t candidate : candidateMergeColors) {
        if (nonIsoGraphsMerge && isoSubGroups_[candidate]->Size() == 1) {
            mergeColors.push_back(candidate);
            continue;
        }
        if (!nonIsoGraphsMerge && isoSubGroups_[candidate]->Size() > 1) {
            mergeColors.push_back(candidate);
        }
    }
    return mergeColors;
}

bool IsoPartitioner::CanMergeScopes(int32_t currColor, int32_t mergeColor) const
{
    auto canMergeFrom = [this](const std::shared_ptr<IsomorphismGraphGroup>& fromGroup,
                               const std::shared_ptr<IsomorphismGraphGroup>& toGroup) -> bool {
        for (auto& g : fromGroup->isoGraphs_) {
            if (g->scopeId_ == -1)
                continue;
            if (!g->GetAllowCrossScopeMerge()) {
                APASS_LOG_INFO_F(Elements::Operation,
                                 "Cannot merge: subgraph scopeId=%d with allowCrossScopeMerge=false.", g->scopeId_);
                return false;
            }
            for (auto& tg : toGroup->isoGraphs_) {
                if (tg->scopeId_ != -1) {
                    APASS_LOG_INFO_F(Elements::Operation,
                                     "Cannot merge: allowCrossScopeMerge=true requires target scope=-1, got %d.",
                                     tg->scopeId_);
                    return false;
                }
            }
        }
        return true;
    };

    return canMergeFrom(isoSubGroups_[currColor], isoSubGroups_[mergeColor]) &&
           canMergeFrom(isoSubGroups_[mergeColor], isoSubGroups_[currColor]);
}

int32_t IsoPartitioner::CalculateMergedLatency(int32_t currColor, int32_t mergeColor) const
{
    int32_t currColorSize = static_cast<int32_t>(isoSubGroups_[currColor]->Size());
    int32_t mergeColorSize = static_cast<int32_t>(isoSubGroups_[mergeColor]->Size());
    if (currColorSize <= mergeColorSize) {
        return isoSubGroups_[currColor]->GetLatency() +
               isoSubGroups_[mergeColor]->GetLatency() * (mergeColorSize / currColorSize);
    }
    return isoSubGroups_[currColor]->GetLatency() * (currColorSize / mergeColorSize) +
           isoSubGroups_[mergeColor]->GetLatency();
}

bool IsoPartitioner::CheckIsoMergeConditions(int32_t currColorSize, int32_t mergeColorSize) const
{
    bool isSuitableForMerge = (currColorSize == mergeColorSize);
    isSuitableForMerge = isSuitableForMerge || (std::min(currColorSize, mergeColorSize) >= parallelNum_);
    return isSuitableForMerge;
}

bool IsoPartitioner::SuitableForMergeCheck(int32_t currColor, int32_t mergeColor, bool nonIsoGraphsMerge) const
{
    if (!CanMergeScopes(currColor, mergeColor)) {
        return false;
    }
    std::set<OpCoreType> opcoreTypes{isoSubGroups_[currColor]->GetSubGraph(0)->coreType_,
                                     isoSubGroups_[mergeColor]->GetSubGraph(0)->coreType_};
    bool coreTypeMergable = operationInfo_->CoreTypeMergeable(opcoreTypes);
    int32_t currColorSize = static_cast<int32_t>(isoSubGroups_[currColor]->Size());
    int32_t mergeColorSize = static_cast<int32_t>(isoSubGroups_[mergeColor]->Size());
    if (currColorSize == 0 || mergeColorSize == 0) {
        return false;
    }
    int32_t latencyMerged = CalculateMergedLatency(currColor, mergeColor);
    bool cycleMergable = latencyMerged <= cycleUB_;
    if (nonIsoGraphsMerge) {
        bool shouldMerge = coreTypeMergable && cycleMergable;
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Try merge current group: %d [%s]\n\t with: %d [%s], is suitable for merge: %d.", currColor,
                          isoSubGroups_[currColor]->GetSubGraph(0)->DumpStr().c_str(), mergeColor,
                          isoSubGroups_[mergeColor]->GetSubGraph(0)->DumpStr().c_str(), shouldMerge);
        return shouldMerge;
    }
    bool isSuitableForMerge = CheckIsoMergeConditions(currColorSize, mergeColorSize);
    isSuitableForMerge = isSuitableForMerge || (std::min(isoSubGroups_[currColor]->GetLatency(),
                                                         isoSubGroups_[mergeColor]->GetLatency()) <= cycleLB_);
    isSuitableForMerge = coreTypeMergable && isSuitableForMerge && cycleMergable;
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "Try merge current group: %d [%s]\n\t with: %d [%s], is suitable for merge: %d.", currColor,
                      isoSubGroups_[currColor]->GetSubGraph(0)->DumpStr().c_str(), mergeColor,
                      isoSubGroups_[mergeColor]->GetSubGraph(0)->DumpStr().c_str(), isSuitableForMerge);
    return isSuitableForMerge;
}

Status IsoPartitioner::IsomorphismGroupMergeStep(bool nonIsoGraphsMerge)
{
    std::vector<std::pair<int32_t, int32_t>> isoSubIdxs;
    std::vector<std::unordered_set<int32_t>> isoInGraph;
    std::vector<std::unordered_set<int32_t>> isoOutGraph;
    std::vector<std::vector<int32_t>> isoNodeList;
    std::vector<int32_t> isoIdx2color;
    if (IsomorphismGroupMergePrepare(isoSubIdxs, isoInGraph, isoOutGraph, isoNodeList, isoIdx2color) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "IsomorphismGroupMergePrepare failed.");
        return FAILED;
    }
    size_t currColor = 0;
    while (currColor < isoSubGroups_.size()) {
        std::vector<int32_t> mergeColors = GetCandidateMergeColors(currColor, isoInGraph, isoOutGraph, isoNodeList,
                                                                   isoIdx2color, nonIsoGraphsMerge);
        bool updated = false;
        for (int32_t mergeColor : mergeColors) {
            if (SuitableForMergeCheck(currColor, mergeColor, nonIsoGraphsMerge) &&
                IsomorphismGraphGroup::IsoGraphMerge(isoSubGroups_[currColor], isoSubGroups_[mergeColor], isoSubIdxs)) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Merge current group %zu with %d succeed.", currColor,
                                  mergeColor);
                for (int32_t mergeNodeIdx : isoNodeList[mergeColor]) {
                    isoIdx2color[mergeNodeIdx] = currColor;
                }
                isoNodeList[currColor].insert(isoNodeList[currColor].end(), isoNodeList[mergeColor].begin(),
                                              isoNodeList[mergeColor].end());
                isoNodeList[mergeColor].clear();
                updated = true;
                break;
            }
        }
        if (!updated) {
            currColor++;
        }
    }
    return SUCCESS;
}

Status IsoPartitioner::IsomorphismGroupMergeProcess(bool nonIsoGraphsMerge)
{
    for (int32_t loopCount = 0; loopCount < tryMergeLoopNum_; loopCount++) {
        if (IsomorphismGroupMergeStep(nonIsoGraphsMerge) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "IsomorphismGroupMergeStep failed.");
            return FAILED;
        }
        size_t originalColor = isoSubGroups_.size();
        isoSubGroups_.erase(std::remove_if(isoSubGroups_.begin(), isoSubGroups_.end(),
                                           [](auto& groupPtr) { return groupPtr->Size() == 0; }),
                            isoSubGroups_.end());
        if (originalColor == isoSubGroups_.size()) {
            break;
        }
    }
    return SUCCESS;
}

std::string SubGraph::DumpStr()
{
    std::stringstream ss;
    ss << "  NodeList: {";
    for (auto op : GetOpList()) {
        ss << op->GetOpcodeStr() << "(" << op->GetOpMagic() << "), ";
    }
    ss << "}; cycles: {" << cycle_ << "};";
    ss << " core type: {" << static_cast<int32_t>(coreType_) << "};" << std::endl;
    return ss.str();
}

std::vector<Operation*> SubGraph::GetOpList()
{
    std::vector<Operation*> subOpList;
    if (cycle_ == 0) {
        return subOpList;
    }
    for (size_t i = 0; i < nodeList_.size(); i++) {
        for (int32_t opIdx : superNodeInfo_->node2Op_[nodeList_[i]]) {
            subOpList.push_back(operationInfo_->opList_[opIdx]);
        }
    }
    return subOpList;
}

bool IsomorphismGraphGroup::IsoGraphMerge(std::shared_ptr<IsomorphismGraphGroup>& currGraph,
                                          std::shared_ptr<IsomorphismGraphGroup>& mergeGraph,
                                          std::vector<std::pair<int32_t, int32_t>>& isoSubIdxs)
{
    bool swapped = currGraph->Size() > mergeGraph->Size();
    if (swapped) {
        currGraph.swap(mergeGraph);
    }
    size_t currSize = currGraph->Size();
    size_t mergeSize = mergeGraph->Size();
    std::vector<std::unordered_set<int32_t>> connection(currSize, std::unordered_set<int32_t>{});
    std::unordered_map<std::pair<int32_t, int32_t>, int32_t, IntPairHash> mergeHistory2MergeIdx;
    for (size_t j = 0; j < mergeSize; j++) {
        for (const std::pair<int32_t, int32_t>& mHist : mergeGraph->GetSubGraph(j)->mergeHistoryIsoSub_) {
            mergeHistory2MergeIdx[mHist] = j;
        }
    }
    for (size_t i = 0; i < currSize; i++) {
        for (int32_t nodeIdx : currGraph->GetSubGraph(i)->inNodes_) {
            std::pair<int32_t, int32_t>& nodeBelong = isoSubIdxs[nodeIdx];
            if (mergeHistory2MergeIdx.count(nodeBelong) > 0) {
                connection[i].insert(mergeHistory2MergeIdx[nodeBelong]);
            }
        }
        for (int32_t nodeIdx : currGraph->GetSubGraph(i)->outNodes_) {
            std::pair<int32_t, int32_t>& nodeBelong = isoSubIdxs[nodeIdx];
            if (mergeHistory2MergeIdx.count(nodeBelong) > 0) {
                connection[i].insert(mergeHistory2MergeIdx[nodeBelong]);
            }
        }
    }
    size_t mergeTimes = 0;
    std::unordered_set<int32_t> mergeSet;
    for (auto& conn : connection) {
        mergeTimes += conn.size();
        mergeSet.insert(conn.begin(), conn.end());
    }
    if (mergeTimes != mergeSize || mergeSet.size() != mergeSize) {
        if (swapped) {
            currGraph.swap(mergeGraph);
        }
        return false;
    }
    for (int32_t i = 0; i < static_cast<int32_t>(currSize); i++) {
        for (int32_t j : connection[i]) {
            currGraph->GetSubGraph(i)->Merge(*mergeGraph->GetSubGraph(j));
        }
    }
    mergeGraph->Clear();
    return true;
}

void SubGraph::Merge(SubGraph& sg)
{
    nodeList_.insert(nodeList_.end(), sg.nodeList_.begin(), sg.nodeList_.end());
    if (nodeSet_.size() < sg.nodeSet_.size()) {
        nodeSet_.swap(sg.nodeSet_);
    }
    nodeSet_.insert(sg.nodeSet_.begin(), sg.nodeSet_.end());
    cycle_ += sg.cycle_;
    mergeHistoryIsoSub_.insert(sg.mergeHistoryIsoSub_.begin(), sg.mergeHistoryIsoSub_.end());

    std::vector<int32_t> inNodesTmp;
    std::vector<int32_t> outNodesTmp;
    for (int32_t nodeIdx : inNodes_) {
        if (nodeSet_.count(nodeIdx) == 0) {
            inNodesTmp.push_back(nodeIdx);
        }
    }
    for (int32_t nodeIdx : sg.inNodes_) {
        if (nodeSet_.count(nodeIdx) == 0) {
            inNodesTmp.push_back(nodeIdx);
        }
    }
    for (int32_t nodeIdx : outNodes_) {
        if (nodeSet_.count(nodeIdx) == 0) {
            outNodesTmp.push_back(nodeIdx);
        }
    }
    for (int32_t nodeIdx : sg.outNodes_) {
        if (nodeSet_.count(nodeIdx) == 0) {
            outNodesTmp.push_back(nodeIdx);
        }
    }
    std::sort(inNodesTmp.begin(), inNodesTmp.end());
    inNodesTmp.erase(std::unique(inNodesTmp.begin(), inNodesTmp.end()), inNodesTmp.end());
    std::sort(outNodesTmp.begin(), outNodesTmp.end());
    outNodesTmp.erase(std::unique(outNodesTmp.begin(), outNodesTmp.end()), outNodesTmp.end());
    inNodes_.swap(inNodesTmp);
    outNodes_.swap(outNodesTmp);
}

Status IsoPartitioner::UpdatePartitionResult(Function& function)
{
    int32_t colorIdx = 0;
    for (size_t i = 0; i < isoSubGroups_.size(); i++) {
        for (size_t j = 0; j < isoSubGroups_[i]->Size(); j++) {
            for (auto op : isoSubGroups_[i]->GetSubGraph(j)->GetOpList()) {
                op->UpdateSubgraphID(colorIdx);
            }
            colorIdx++;
        }
    }
    for (const auto& op : function.Operations()) {
        if (op.GetSubgraphID() < 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "GraphPartition left op %s[%d] without a valid SubgraphID.",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
    }
    function.SetTotalSubGraphCount(colorIdx);
    return SUCCESS;
}

Status IsoPartitioner::SetParameter(int32_t parallelNum, int32_t pgLowerBound, bool useReduceBalanceHash)
{
    if (parallelNum < 0) {
        APASS_LOG_ERROR_F(Elements::Config, "Illegal parallelNum: %d; Parameter parallelNum must be non-negative.",
                          parallelNum);
        return FAILED;
    }
    if (pgLowerBound < 0) {
        APASS_LOG_ERROR_F(Elements::Config, "Illegal pgLowerBound: %d; Parameter pgLowerBound must be non-negative.",
                          pgLowerBound);
        return FAILED;
    }
    parallelNum_ = parallelNum;
    cycleLB_ = pgLowerBound;
    useReduceBalanceHash_ = useReduceBalanceHash;
    return SUCCESS;
}

Status IsoPartitioner::EstimateCycleUB(Function& function)
{
    // 计算总cycle
    int64_t totalLatency = 0;
    for (const auto& op : function.Operations()) {
        int32_t latency = op.GetLatency();
        if (latency < 0) {
            APASS_LOG_WARN_F(Elements::Config, "Detected op: %d negative latency: %d, ignoring.", op.GetOpMagic(),
                             latency);
            continue;
        }
        totalLatency += latency;
    }

    // 根据阈值推导cycleUB_
    int32_t estimatedCycleUB = CYCLE_UB_LEVEL3;
    if (totalLatency >= LATENCY_THRESHOLD_LEVEL1) {
        estimatedCycleUB = CYCLE_UB_LEVEL1;
    } else if (totalLatency >= LATENCY_THRESHOLD_LEVEL2) {
        estimatedCycleUB = CYCLE_UB_LEVEL2;
    } else if (totalLatency >= LATENCY_THRESHOLD_LEVEL3) {
        estimatedCycleUB = CYCLE_UB_LEVEL3;
    } else {
        estimatedCycleUB = CYCLE_UB_LEVEL4;
    }

    // 更新成员变量并记录日志
    cycleUB_ = estimatedCycleUB;
    APASS_LOG_INFO_F(Elements::Config, "Estimated cycleUB_: %d based on total latency: %ld", cycleUB_, totalLatency);

    return SUCCESS;
}
} // namespace npu::tile_fwk
