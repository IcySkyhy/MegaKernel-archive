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
 * \file supernode_graph_builder.cpp
 * \brief
 */

#include "supernode_graph_builder.h"
#include <iostream>
#include <algorithm>
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "SuperNodeGraphBuilder"

namespace npu::tile_fwk {

const std::unordered_set<Opcode> nodeScopeSkipCode{Opcode::OP_CONTRACT, Opcode::OP_SLICE, Opcode::OP_ASSEMBLE,
                                                   Opcode::OP_VIEW};

uint64_t OperationGraphInfo::GetHash(const Operation* op) const
{
    // Numeric hash replaces string concatenation to avoid malloc at 200K+ scale.
    // Tags distinguish IOperand/OOperand sections (equivalent to original "IOperand-"/"OOperand-" prefixes).
    // hashCombine follows the boost::hash_combine formula:
    //   - 0x9e3779b9ULL is the fractional part of the golden ratio (floor(2^32 / phi)),
    //     used as a constant to ensure well-distributed bit mixing.
    //   - Shifts 6 (left) and 2 (right) spread high/low bits across the 64-bit word,
    //     preventing collision patterns when seeds share low-order bits.
    auto hashCombine = [](uint64_t seed, uint64_t val) -> uint64_t {
        return seed ^ (val + 0x9e3779b9ULL + (seed << 6) + (seed >> 2));
    };
    constexpr uint64_t IOPERAND_TAG = 0x49504F5045524E44ULL;
    constexpr uint64_t OOPERAND_TAG = 0x4F4F5045524E44ULL;
    uint64_t hashValue = static_cast<uint64_t>(op->GetOpcode());
    hashValue = hashCombine(hashValue, IOPERAND_TAG);
    for (const auto& tensor : op->GetIOperands()) {
        hashValue = hashCombine(hashValue, static_cast<uint64_t>(tensor->GetMemoryTypeOriginal()));
        hashValue = hashCombine(hashValue, static_cast<uint64_t>(tensor->tensor->datatype));
    }
    hashValue = hashCombine(hashValue, OOPERAND_TAG);
    for (const auto& tensor : op->GetOOperands()) {
        hashValue = hashCombine(hashValue, static_cast<uint64_t>(tensor->GetMemoryTypeOriginal()));
        hashValue = hashCombine(hashValue, static_cast<uint64_t>(tensor->tensor->datatype));
    }
    return hashValue;
}

std::vector<int32_t> OperationGraphInfo::GetSameLevelOpIdx(int32_t opIdx, Opcode opLabel) const
{
    if (opIdx < 0 || opIdx >= static_cast<int32_t>(opList_.size()) || opList_[opIdx]->GetOOperands().empty()) {
        return {};
    }
    std::vector<int32_t> res;
    std::shared_ptr<LogicalTensor> output = opList_[opIdx]->GetOOperands()[0];
    for (const auto& parentOpPtr : output->GetProducers()) {
        if (parentOpPtr->GetOpcode() == opLabel) {
            int32_t parentOpMagic = parentOpPtr->GetOpMagic();
            if (magic2Idx_.count(parentOpMagic) > 0) {
                int32_t targetIdx = magic2Idx_.at(parentOpPtr->GetOpMagic());
                res.push_back(targetIdx);
            }
        }
    }
    return res;
}

inline bool IsValidOpIndex(size_t opSize, int32_t opIdx) { return opIdx >= 0 && static_cast<size_t>(opIdx) < opSize; }

inline bool IsL0CAccumMatmulEdge(const std::shared_ptr<OperationGraphInfo>& operationInfo, int32_t consumerIdx,
                                 int32_t producerIdx)
{
    if (operationInfo == nullptr) {
        return false;
    }
    if (!IsValidOpIndex(operationInfo->opList_.size(), consumerIdx) ||
        !IsValidOpIndex(operationInfo->opList_.size(), producerIdx)) {
        return false;
    }
    const Operation* consumerOp = operationInfo->opList_[consumerIdx];
    const Operation* producerOp = operationInfo->opList_[producerIdx];
    if (consumerOp == nullptr || producerOp == nullptr) {
        return false;
    }
    if (OpcodeManager::Inst().GetOpCalcType(consumerOp->GetOpcode()) != OpCalcType::MATMUL ||
        OpcodeManager::Inst().GetOpCalcType(producerOp->GetOpcode()) != OpCalcType::MATMUL) {
        return false;
    }
    // A_MULACC_B 的第 3 个输入是 L0C 累加输入，用它判断 matmul 之间是否为真实累加依赖。
    constexpr size_t kMatmulAccumInputIndex = 2;
    auto accumTensor = consumerOp->GetInputOperand(kMatmulAccumInputIndex);
    if (accumTensor == nullptr || accumTensor->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        return false;
    }
    for (const auto* accumProducer : accumTensor->GetProducers()) {
        if (accumProducer == producerOp) {
            return true;
        }
    }
    return false;
}

bool OperationGraphInfo::CoreTypeMergeable(const std::set<OpCoreType>& coreTypes) const
{
    if (coreTypes.size() == 1 && (*coreTypes.begin() == OpCoreType::AICPU || *coreTypes.begin() == OpCoreType::HUB)) {
        return false;
    }
    if (useCVMixPartition_ || coreTypes.size() == 1) {
        return true;
    }
    const size_t maxSeperateCoreNum = 2;
    if (coreTypes.size() > maxSeperateCoreNum) {
        return false;
    }
    if (coreTypes.size() == maxSeperateCoreNum) {
        auto firstType = *coreTypes.begin();
        auto secondType = *(++coreTypes.begin());
        if (firstType == OpCoreType::AICPU || secondType == OpCoreType::AICPU) {
            return false;
        }
        if (firstType == OpCoreType::HUB || secondType == OpCoreType::HUB) {
            return false;
        }
        if (firstType == OpCoreType::ANY || secondType == OpCoreType::ANY) {
            return true;
        }
    }
    return false;
}

inline int32_t FindParent(std::vector<int32_t>& parent, int32_t i)
{
    if (i < 0 || i >= static_cast<int32_t>(parent.size())) {
        APASS_LOG_ERROR_F(Elements::Operation, "Call FindParent with illegal parameter %d.", i);
        return -1;
    }
    if (parent[i] == i) {
        return i;
    }
    std::vector<int32_t> searchPath;
    int32_t currIdx = i;
    while (parent[currIdx] != currIdx) {
        searchPath.push_back(currIdx);
        currIdx = parent[currIdx];
        if (currIdx < 0 || currIdx >= static_cast<int32_t>(parent.size())) {
            APASS_LOG_ERROR_F(Elements::Operation, "Find illegal parameter %d in FindParent.", currIdx);
            return -1;
        }
        if (searchPath.size() > (parent.size() + 1)) {
            APASS_LOG_ERROR_F(Elements::Operation, "Find loop in FindParent.");
            return -1;
        }
    }
    for (auto parentIdx : searchPath) {
        parent[parentIdx] = currIdx;
    }
    return currIdx;
}

inline std::string GetOpCoreTypeStr(OpCoreType coreType)
{
    std::map<OpCoreType, std::string> coreTypeStr{{OpCoreType::AIC, "AIC"},
                                                  {OpCoreType::AIV, "AIV"},
                                                  {OpCoreType::AICPU, "AICPU"},
                                                  {OpCoreType::HUB, "HUB"},
                                                  {OpCoreType::GMATOMIC, "GMATOMIC"}};
    if (coreTypeStr.count(coreType) == 0) {
        return "UNKNOWN";
    } else {
        return coreTypeStr[coreType];
    }
}

Status NodeGraphInfo::MergeSrcToDstIsland(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                                          std::vector<int32_t>& parent, int32_t src, int32_t dst)
{
    int32_t srcParent = FindParent(parent, src);
    int32_t dstParent = FindParent(parent, dst);
    if (srcParent == -1 || dstParent == -1) {
        APASS_LOG_ERROR_F(Elements::Operation, "Merge node in the disjoint set failed.%s",
                          GetFormatBacktrace(*(operationGraphInfo->opList_[src])).c_str());
        return FAILED;
    }
    std::set<OpCoreType> coreTypes{operationGraphInfo->opCoreType_[src], operationGraphInfo->opCoreType_[dst],
                                   operationGraphInfo->opCoreType_[srcParent],
                                   operationGraphInfo->opCoreType_[dstParent]};
    bool isAICPUandVIEW = false;
    isAICPUandVIEW = isAICPUandVIEW || (operationGraphInfo->opCoreType_[src] == OpCoreType::AICPU &&
                                        IsViewLike(operationGraphInfo->opList_[dst]->GetOpcode()));
    isAICPUandVIEW = isAICPUandVIEW || (operationGraphInfo->opCoreType_[dst] == OpCoreType::AICPU &&
                                        IsViewLike(operationGraphInfo->opList_[src]->GetOpcode()));
    bool isAICPUandAssemble = false;
    // HUB只能和View/Assemble在一张子图中
    bool hubWithViewAssemble = false;
    const size_t maxCoreTypesNum = 2;
    if (coreTypes.size() == maxCoreTypesNum && coreTypes.count(OpCoreType::HUB) > 0) {
        auto srcOpCoreType = operationGraphInfo->opCoreType_[src];
        auto dstOpCoreType = operationGraphInfo->opCoreType_[dst];
        auto srcOpCode = operationGraphInfo->opList_[src]->GetOpcode();
        auto dstOpCode = operationGraphInfo->opList_[dst]->GetOpcode();
        if (srcOpCoreType != OpCoreType::HUB) {
            hubWithViewAssemble = IsViewLike(srcOpCode) || IsAssembleLike(srcOpCode);
        }
        if (dstOpCoreType != OpCoreType::HUB) {
            hubWithViewAssemble = IsViewLike(dstOpCode) || IsAssembleLike(dstOpCode);
        }
    }
    isAICPUandAssemble = isAICPUandAssemble || (operationGraphInfo->opCoreType_[src] == OpCoreType::AICPU &&
                                                IsAssembleLike(operationGraphInfo->opList_[dst]->GetOpcode()));
    isAICPUandAssemble = isAICPUandAssemble || (operationGraphInfo->opCoreType_[dst] == OpCoreType::AICPU &&
                                                IsAssembleLike(operationGraphInfo->opList_[src]->GetOpcode()));
    if ((!hubWithViewAssemble) && (!isAICPUandVIEW) && (!isAICPUandAssemble) &&
        (!operationGraphInfo->CoreTypeMergeable(coreTypes))) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Try to merge operations with different OpCoreType in building SuperNode.");
        std::vector<int> mergeIdxs{src, srcParent, dst, dstParent};
        for (int mergeIdx : mergeIdxs) {
            auto& mergeOp = operationGraphInfo->opList_[mergeIdx];
            APASS_LOG_ERROR_F(Elements::Operation, "%s [opMagic: %d] [opCoreType: %s].%s",
                              mergeOp->GetOpcodeStr().c_str(), mergeOp->GetOpMagic(),
                              GetOpCoreTypeStr(operationGraphInfo->opCoreType_[mergeIdx]).c_str(),
                              GetFormatBacktrace(*mergeOp).c_str());
        }
        return FAILED;
    }
    parent[srcParent] = dstParent;
    return SUCCESS;
}

std::vector<int32_t> NodeInnerExpand(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                                     std::vector<int32_t>& nodeOps)
{
    std::vector<int32_t> frontBackVisitedOp;
    int32_t minOpIdx = static_cast<int32_t>(operationGraphInfo->opList_.size());
    int32_t maxOpIdx = -1;
    for (int32_t opIdx : nodeOps) {
        minOpIdx = opIdx < minOpIdx ? opIdx : minOpIdx;
        maxOpIdx = opIdx > maxOpIdx ? opIdx : maxOpIdx;
    }
    std::unordered_set<int32_t> frontVisitedOp;
    std::vector<int32_t> frontVisitStack(nodeOps);
    while (frontVisitStack.size() > 0) {
        int32_t opIdx = frontVisitStack.back();
        frontVisitStack.pop_back();
        if (frontVisitedOp.count(opIdx) > 0) {
            continue;
        }
        frontVisitedOp.insert(opIdx);
        for (int32_t nextOpIdx : operationGraphInfo->outGraph_[opIdx]) {
            if (nextOpIdx <= maxOpIdx) {
                frontVisitStack.push_back(nextOpIdx);
            }
        }
    }
    std::unordered_set<int32_t> backVisitedOp;
    std::vector<int32_t> backVisitStack(nodeOps);
    while (backVisitStack.size() > 0) {
        int32_t opIdx = backVisitStack.back();
        backVisitStack.pop_back();
        if (backVisitedOp.count(opIdx) > 0) {
            continue;
        }
        if (frontVisitedOp.count(opIdx) > 0) {
            frontBackVisitedOp.push_back(opIdx);
        }
        backVisitedOp.insert(opIdx);
        for (int32_t prevOpIdx : operationGraphInfo->inGraph_[opIdx]) {
            if (prevOpIdx >= minOpIdx) {
                backVisitStack.push_back(prevOpIdx);
            }
        }
    }
    return frontBackVisitedOp;
}

Status NodeGraphInfo::AvoidLoop(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                                std::vector<int32_t>& parent, std::vector<std::vector<int32_t>>& node2Op, bool& updated)
{
    std::vector<Operation*>& opList = operationGraphInfo->opList_;
    std::vector<int32_t> parentToNodes(opList.size(), -1);
    updated = false;
    node2Op.clear();
    for (int32_t i = 0; i < static_cast<int32_t>(opList.size()); i++) {
        int32_t currParent = FindParent(parent, i);
        if (currParent == -1) {
            APASS_LOG_ERROR_F(Elements::Operation, "Find parent in the union set failed.%s",
                              GetFormatBacktrace(*(operationGraphInfo->opList_[i])).c_str());
            return FAILED;
        }
        if (currParent == i) {
            parentToNodes[i] = node2Op.size();
            node2Op.push_back(std::vector<int32_t>());
        }
    }
    for (int32_t i = 0; i < static_cast<int32_t>(operationGraphInfo->opList_.size()); i++) {
        int32_t currParent = FindParent(parent, i);
        if (currParent == -1) {
            APASS_LOG_ERROR_F(Elements::Operation, "Find parent in the union set failed.%s",
                              GetFormatBacktrace(*(operationGraphInfo->opList_[i])).c_str());
            return FAILED;
        }
        int32_t nodeIdx = parentToNodes[currParent];
        node2Op[nodeIdx].push_back(i);
    }
    for (size_t nodeIdx = 0; nodeIdx < node2Op.size(); nodeIdx++) {
        if (node2Op[nodeIdx].size() <= 1) {
            continue; // single-op node: BFS would only find itself
        }
        std::vector<int32_t> expandNode = NodeInnerExpand(operationGraphInfo, node2Op[nodeIdx]);
        if (expandNode.size() == node2Op[nodeIdx].size() || expandNode.empty()) {
            continue;
        }
        updated = true;
        for (size_t opIdx = 1; opIdx < expandNode.size(); opIdx++) {
            if (MergeSrcToDstIsland(operationGraphInfo, parent, expandNode[0], expandNode[opIdx]) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "Build the disjoint set failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status NodeGraphInfo::Build(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                            const std::vector<std::pair<int32_t, int32_t>>& mergePair, bool markIsCube)
{
    std::vector<Operation*>& opList = operationGraphInfo->opList_;
    std::vector<int32_t> parent(opList.size());
    for (size_t i = 0; i < opList.size(); i++) {
        parent[i] = i;
    }
    for (auto& pr : mergePair) {
        if (MergeSrcToDstIsland(operationGraphInfo, parent, pr.first, pr.second) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Build the disjoint set failed.");
            return FAILED;
        }
    }
    bool updated = true;
    while (updated) {
        updated = false;
        if (AvoidLoop(operationGraphInfo, parent, node2Op_, updated) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Avoid loop in building node failed");
            return FAILED;
        }
    }
    BuildNodeMapping(operationGraphInfo);
    BuildInOutGraph(operationGraphInfo);
    SetNodeCoreTypeAndMergeable(operationGraphInfo, markIsCube);
    return SUCCESS;
}

bool NodeGraphInfo::CheckScopeNotMergeable(Operation& op)
{
    return op.GetScopeId() != -1 && !op.GetAllowCrossScopeMerge();
}

bool NodeGraphInfo::CheckUbToUbWithDynOffset(Operation& op)
{
    if (!IsViewLike(op.GetOpcode()) && !IsAssembleLike(op.GetOpcode())) {
        return false;
    }

    auto input = op.GetIOperands().empty() ? nullptr : op.GetIOperands()[0];
    auto output = op.GetOOperands().empty() ? nullptr : op.GetOOperands()[0];
    if (!input || !output) {
        return false;
    }

    bool isUbToUb = input->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
                    output->GetMemoryTypeOriginal() == MemoryType::MEM_UB;
    if (!isUbToUb) {
        return false;
    }

    bool hasDynOffset = false;
    if (IsViewLike(op.GetOpcode())) {
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (viewAttr && !viewAttr->GetFromDynOffset().empty()) {
            hasDynOffset = true;
        }
    } else if (IsAssembleLike(op.GetOpcode())) {
        auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
        if (assembleAttr && !assembleAttr->GetToDynOffset().empty()) {
            hasDynOffset = true;
        }
    }

    if (!hasDynOffset) {
        return false;
    }

    APASS_LOG_INFO_F(Elements::Operation,
                     "Node contains UB-to-UB %s[%d] with dynamic offset, marking as not mergeable.",
                     op.GetOpcodeStr().c_str(), op.GetOpMagic());
    return true;
}

bool NodeGraphInfo::CheckViewAssembleOffset(Operation& op)
{
    if (!IsViewLike(op.GetOpcode()) && !IsAssembleLike(op.GetOpcode())) {
        return false;
    }

    std::string offsetStr;
    std::string dynOffsetStr;

    if (IsViewLike(op.GetOpcode())) {
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (!viewAttr) {
            return false;
        }
        auto& fromOffset = viewAttr->GetFromOffset();
        for (auto val : fromOffset) {
            offsetStr += std::to_string(val) + ",";
        }
        auto& fromDynOffset = viewAttr->GetFromDynOffset();
        for (auto& val : fromDynOffset) {
            dynOffsetStr += val.Dump() + ",";
        }
    } else if (IsAssembleLike(op.GetOpcode())) {
        auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
        if (!assembleAttr) {
            return false;
        }
        auto& toOffset = assembleAttr->GetToOffset();
        for (auto val : toOffset) {
            offsetStr += std::to_string(val) + ",";
        }
        auto& toDynOffset = assembleAttr->GetToDynOffset();
        for (auto& val : toDynOffset) {
            dynOffsetStr += val.Dump() + ",";
        }
    }

    if (offsetStr.empty() && dynOffsetStr.empty()) {
        return false;
    }

    if (offsetStr != dynOffsetStr) {
        APASS_LOG_INFO_F(Elements::Operation, "%s[%d] offset[%s] != dynOffset[%s], marking as not mergeable.",
                         op.GetOpcodeStr().c_str(), op.GetOpMagic(), offsetStr.c_str(), dynOffsetStr.c_str());
        return true;
    }

    return false;
}

bool NodeGraphInfo::GetNodeMergeable(const std::shared_ptr<OperationGraphInfo> operationGraphInfo, int32_t nodeIdx)
{
    bool isMergeable = !(node2Op_[nodeIdx].size() == 1 &&
                         operationGraphInfo->opList_[node2Op_[nodeIdx][0]]->GetOpcode() == Opcode::OP_RESHAPE &&
                         ((nodeInGraphList_[nodeIdx].size() > 1 && nodeOutGraphList_[nodeIdx].size() > 1) ||
                          (nodeInGraphList_[nodeIdx].size() > 1 && nodeOutGraphList_[nodeIdx].empty()) ||
                          (nodeInGraphList_[nodeIdx].empty() && nodeOutGraphList_[nodeIdx].size() > 1)));

    for (auto opIdx : node2Op_[nodeIdx]) {
        auto& op = operationGraphInfo->opList_[opIdx];

        if (CheckScopeNotMergeable(*op)) {
            isMergeable = false;
        }

        if (CheckUbToUbWithDynOffset(*op) && CheckViewAssembleOffset(*op)) {
            isMergeable = false;
        }
    }

    return isMergeable;
}

Status NodeGraphInfo::BuildInOutGraph(const std::shared_ptr<OperationGraphInfo> operationGraphInfo)
{
    size_t numNodes = node2Op_.size();
    nodeInGraphList_.assign(numNodes, std::vector<int32_t>());
    nodeOutGraphList_.assign(numNodes, std::vector<int32_t>());
    // INVALID_NODE_IDX marks unseen neighbour slots during edge deduplication.
    constexpr int32_t INVALID_NODE_IDX = -1;
    std::vector<int32_t> inSeen(numNodes, INVALID_NODE_IDX);
    std::vector<int32_t> outSeen(numNodes, INVALID_NODE_IDX);
    for (size_t i = 0; i < numNodes; i++) {
        int32_t iIdx = static_cast<int32_t>(i);
        for (int32_t opIdx : node2Op_[i]) {
            for (int32_t publisherOpIdx : operationGraphInfo->inGraph_[opIdx]) {
                int32_t publisherNodeIdx = op2Node_[publisherOpIdx];
                if (publisherNodeIdx == iIdx) {
                    continue;
                }
                if (inSeen[publisherNodeIdx] != iIdx) {
                    inSeen[publisherNodeIdx] = iIdx;
                    nodeInGraphList_[i].push_back(publisherNodeIdx);
                }
                if (outSeen[publisherNodeIdx] != iIdx) {
                    outSeen[publisherNodeIdx] = iIdx;
                    nodeOutGraphList_[publisherNodeIdx].push_back(iIdx);
                }
            }
        }
    }
    return SUCCESS;
}

void NodeGraphInfo::SetNodeCoreTypeAndMergeable(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                                                bool markIsCube)
{
    nodeCoreType_.resize(node2Op_.size());
    nodeMergeable_.resize(node2Op_.size());
    for (size_t i = 0; i < node2Op_.size(); i++) {
        nodeCoreType_[i] = OpCoreType::AIV;
        for (int32_t opIdx : node2Op_[i]) {
            if (operationGraphInfo->opCoreType_[opIdx] != OpCoreType::ANY) {
                nodeCoreType_[i] = operationGraphInfo->opCoreType_[opIdx];
                break;
            }
        }
        nodeMergeable_[i] = GetNodeMergeable(operationGraphInfo, i);
        if (!markIsCube) {
            continue;
        }
        bool isCube = false;
        for (auto j : node2Op_[i]) {
            if (operationGraphInfo->opCoreType_[j] == OpCoreType::AIC) {
                isCube = true;
                break;
            }
        }
        for (auto j : node2Op_[i]) {
            operationGraphInfo->opList_[j]->SetAttribute(OpAttributeKey::isCube, isCube);
        }
    }
}

int32_t NodeGraphInfo::GetNodeCycle(int32_t nodeIdx) const
{
    if (nodeIdx < 0 || nodeIdx >= static_cast<int32_t>(nodeCycles_.size())) {
        return 0;
    }
    return nodeCycles_[nodeIdx];
}

Status SuperNodeGraphBuilder::BuildOpGraph(const std::vector<Operation*>& opList)
{
    operationInfo_ = std::make_shared<OperationGraphInfo>();
    if (operationInfo_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Create OperationInfo failed.");
        return FAILED;
    }
    operationInfo_->opList_ = opList;
    operationInfo_->opHashList_.resize(opList.size());
    operationInfo_->opCoreType_.resize(opList.size());
    operationInfo_->useCVMixPartition_ = useCVMixPartition_;
    for (size_t i = 0; i < opList.size(); i++) {
        operationInfo_->magic2Idx_[opList[i]->GetOpMagic()] = i;
    }
    // Pre-build tensor → producer op indices map to avoid red-black tree traversal (GetProducers())
    std::unordered_map<LogicalTensor*, std::vector<int32_t>> tensorToProducers;
    tensorToProducers.reserve(opList.size());
    for (size_t i = 0; i < opList.size(); i++) {
        for (const auto& output : opList[i]->GetOOperands()) {
            tensorToProducers[output.get()].push_back(static_cast<int32_t>(i));
        }
    }
    operationInfo_->inGraph_.resize(opList.size());
    operationInfo_->outGraph_.resize(opList.size());
    // INVALID_OP_IDX marks unseen producer/consumer slots during edge deduplication.
    constexpr int32_t INVALID_OP_IDX = -1;
    std::vector<int32_t> inSeen(opList.size(), INVALID_OP_IDX);
    std::vector<int32_t> outSeen(opList.size(), INVALID_OP_IDX);
    for (size_t i = 0; i < opList.size(); i++) {
        int32_t iIdx = static_cast<int32_t>(i);
        for (const auto& input : opList[i]->GetIOperands()) {
            auto it = tensorToProducers.find(input.get());
            if (it == tensorToProducers.end()) {
                continue;
            }
            for (int32_t operationInIdx : it->second) {
                if (inSeen[operationInIdx] != iIdx) {
                    inSeen[operationInIdx] = iIdx;
                    operationInfo_->inGraph_[i].push_back(operationInIdx);
                }
                if (outSeen[operationInIdx] != iIdx) {
                    outSeen[operationInIdx] = iIdx;
                    operationInfo_->outGraph_[operationInIdx].push_back(iIdx);
                }
            }
        }
    }
    for (size_t i = 0; i < opList.size(); i++) {
        operationInfo_->opHashList_[i] = operationInfo_->GetHash(opList[i]);
        operationInfo_->opCoreType_[i] = OpcodeManager::Inst().GetCoreType(opList[i]->GetOpcode());
    }
    return SUCCESS;
}

inline bool IsL0cToL1MoveOp(Operation* op)
{
    return (IsViewLike(op->GetOpcode()) || IsAssembleLike(op->GetOpcode())) && op->GetOOperands().size() > 0 &&
           op->GetIOperands().size() > 0 && op->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L0C &&
           op->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1;
}

inline bool HasL0CDataEdge(Operation* producer, Operation* consumer)
{
    if (producer == nullptr || consumer == nullptr) {
        return false;
    }
    for (const auto& consumerOperand : consumer->GetOOperands()) {
        if (consumerOperand != nullptr && consumerOperand->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
            return false;
        }
    }
    for (const auto& output : producer->GetOOperands()) {
        if (output == nullptr || output->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
            continue;
        }
        for (const auto& input : consumer->GetIOperands()) {
            if (input == output) {
                return true;
            }
        }
    }
    return false;
}

bool IsCubeLocalLoad(Operation* op)
{
    if (op == nullptr || op->GetOOperands().empty()) {
        return false;
    }
    static const std::unordered_set<MemoryType> cubeLocalLoadMemoryTypes{
        MemoryType::MEM_L0A,   MemoryType::MEM_L0B, MemoryType::MEM_L0AMX,
        MemoryType::MEM_L0BMX, MemoryType::MEM_BT,  MemoryType::MEM_FIX_QUANT_PRE};
    MemoryType outputMemType = op->GetOOperands()[0]->GetMemoryTypeOriginal();
    return cubeLocalLoadMemoryTypes.count(outputMemType) > 0;
}

bool SuperNodeGraphBuilder::L1CopyInCombine(const std::shared_ptr<OperationGraphInfo> operationInfo,
                                            std::vector<Operation*>& opList, int32_t i,
                                            std::vector<std::pair<int32_t, int32_t>>& mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    if (IsL0cToL1MoveOp(opList[i])) {
        for (auto outNode : operationInfo->outGraph_[i]) {
            if (IsAssembleLike(opList[i]->GetOpcode()) && IsCubeLocalLoad(opList[outNode])) {
                continue;
            }
            mergePair.emplace_back(outNode, i);
            APASS_LOG_DEBUG_F(Elements::Operation, "Combine %d and %d(outNode) for L1 CopyIn in building SuperNode.",
                              opList[i]->GetOpMagic(), opList[outNode]->GetOpMagic());
        }
        for (auto inNode : operationInfo->inGraph_[i]) {
            mergePair.emplace_back(inNode, i);
            APASS_LOG_DEBUG_F(Elements::Operation, "Combine %d and %d(inNode) for L1 CopyIn in building SuperNode.",
                              opList[i]->GetOpMagic(), opList[inNode]->GetOpMagic());
        }
        return true;
    }
    if (opList[i]->GetOOperands().size() > 0 && opList[i]->GetIOperands().size() > 0 &&
        opList[i]->GetIOperands()[0]->GetMemoryTypeOriginal() != MemoryType::MEM_UB &&
        (opList[i]->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1 ||
         opList[i]->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_BT ||
         opList[i]->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_FIX_QUANT_PRE)) {
        for (auto outNode : operationInfo->outGraph_[i]) {
            mergePair.emplace_back(outNode, i);
            APASS_LOG_DEBUG_F(Elements::Operation, "Combine %d and %d for L1 CopyIn in building SuperNode.",
                              opList[i]->GetOpMagic(), opList[outNode]->GetOpMagic());
        }
        return true;
    }
    return false;
}

bool SuperNodeGraphBuilder::ConvertCombine(const std::shared_ptr<OperationGraphInfo> operationInfo,
                                           std::vector<Operation*>& opList, int32_t i,
                                           std::vector<std::pair<int32_t, int32_t>>& mergePair)
{
    const std::unordered_set<MemoryType> AICmem{MemoryType::MEM_L0C, MemoryType::MEM_L1, MemoryType::MEM_L0A,
                                                MemoryType::MEM_L0B};
    const std::unordered_set<MemoryType> AIVmem{MemoryType::MEM_UB};
    if (i < 0 || i >= static_cast<int32_t>(opList.size())) {
        return false;
    }
    if (opList[i]->GetOpcode() != Opcode::OP_CONVERT) {
        return false;
    }
    std::shared_ptr<ConvertOpAttribute> attr = std::static_pointer_cast<ConvertOpAttribute>(
        opList[i]->GetOpAttribute());
    if (attr == nullptr) {
        APASS_LOG_WARN_F(Elements::Operation, "Convert Op %d has no ConvertOpAttribute.", opList[i]->GetOpMagic());
        return true;
    }
    std::pair<MemoryType, MemoryType> convertPath = attr->GetConvertPath();
    bool isAICtoAIV = (AICmem.count(convertPath.first) > 0 && AIVmem.count(convertPath.second) > 0);
    bool isAIVtoAIC = (AIVmem.count(convertPath.first) > 0 && AICmem.count(convertPath.second) > 0);
    if (isAICtoAIV || isAIVtoAIC) {
        for (auto inNode : operationInfo->inGraph_[i]) {
            mergePair.emplace_back(inNode, i);
        }
        return true;
    }
    for (auto inNode : operationInfo->inGraph_[i]) {
        mergePair.emplace_back(inNode, i);
    }
    for (auto outNode : operationInfo->outGraph_[i]) {
        mergePair.emplace_back(outNode, i);
    }
    return true;
}

bool SuperNodeGraphBuilder::AssembleCombine(const std::shared_ptr<OperationGraphInfo> operationInfo,
                                            std::vector<Operation*>& opList, int32_t i,
                                            std::vector<std::pair<int32_t, int32_t>>& mergePair)
{
    if (i < 0 || i >= static_cast<int32_t>(opList.size())) {
        return false;
    }
    // assemble 特殊处理, assemble到local tensor，需要将这些assemble统一island
    if (IsAssembleLike(opList[i]->GetOpcode())) {
        if (opList[i]->GetOOperands().empty()) {
            return false;
        }
        if (AssembleToCopyoutScene(opList[i])) {
            // 在GenerateMoveOp中需要转换为CopyOut的Assemble, 参考CopyOutCombine处理
            return CopyOutCombine(operationInfo, opList, i, mergePair, true);
        }
        // assmemble和其输入绑定
        if (operationInfo->inGraph_[i].size() > 0) {
            mergePair.emplace_back(i, *(operationInfo->inGraph_[i].begin()));
            APASS_LOG_DEBUG_F(Elements::Operation, "Combine %d and %d for Assemble in building SuperNode.",
                              opList[i]->GetOpMagic(), opList[*(operationInfo->inGraph_[i].begin())]->GetOpMagic());
        }
        return true;
    }
    return false;
}

bool SuperNodeGraphBuilder::CopyOutCombine(const std::shared_ptr<OperationGraphInfo> operationInfo,
                                           std::vector<Operation*>& opList, int32_t i,
                                           std::vector<std::pair<int32_t, int32_t>>& mergePair, bool assembleScene)
{
    if (i < 0 || i >= static_cast<int32_t>(opList.size())) {
        return false;
    }
    std::vector<int32_t> candidateOpMagic;
    // 所有的copyout操作与其输入绑定
    if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_OUT || assembleScene) {
        for (auto inNode : operationInfo->inGraph_[i]) {
            mergePair.emplace_back(inNode, i);
            APASS_LOG_DEBUG_F(Elements::Operation, "Combine %d and %d for CopyOut in building SuperNode.",
                              opList[inNode]->GetOpMagic(), opList[i]->GetOpMagic());
        }
        return true;
    }
    return false;
}

bool SuperNodeGraphBuilder::CopyInCombine(const std::shared_ptr<OperationGraphInfo> operationInfo,
                                          std::vector<Operation*>& opList, int32_t i,
                                          std::vector<std::pair<int32_t, int32_t>>& mergePair)
{
    if (i < 0 || i >= static_cast<int32_t>(opList.size())) {
        return false;
    }
    // 所有的copyin操作与其输出绑定
    // OP_TRANSPOSE_VNCHWCONV 虽然归类为 MOVE_LOCAL，但其输入输出均在 UB，
    // 将其与下游 VIEW 合并会拉入含 matmul 的混合 supernode，导致 isCube 污染，
    // 进而使后续 AddAlloc 插入的 UB_ALLOC 与该 V 算子 core type 不一致而报错。
    if ((OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_IN ||
         OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_LOCAL) &&
        opList[i]->GetOpcode() != Opcode::OP_TRANSPOSE_VNCHWCONV && operationInfo->outGraph_[i].size() > 0) {
        int32_t outNode = *(operationInfo->outGraph_[i].begin());
        mergePair.emplace_back(i, outNode);
        APASS_LOG_DEBUG_F(Elements::Operation, "Combine %d and %d for CopyIn in building SuperNode.",
                          opList[i]->GetOpMagic(), opList[outNode]->GetOpMagic());
        return true;
    }
    return false;
}

bool SuperNodeGraphBuilder::MulAccCombine(const std::shared_ptr<OperationGraphInfo> operationInfo,
                                          std::vector<Operation*>& opList, int32_t i,
                                          std::vector<std::pair<int32_t, int32_t>>& mergePair)
{
    if (i < 0 || i >= static_cast<int32_t>(opList.size())) {
        return false;
    }
    // 只要求 A_MULACC_B 的 L0C 累加依赖保持在同一个子图中。
    // 其他 matmul-to-matmul 依赖由普通数据通路规则处理；在这里合并会把多个独立 tile 塌缩成过大的后端编译单元。
    if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MATMUL) {
        for (auto inOp : operationInfo->inGraph_[i]) {
            if (OpcodeManager::Inst().GetOpCalcType(opList[inOp]->GetOpcode()) == OpCalcType::MATMUL) {
                bool isAccumEdge = IsL0CAccumMatmulEdge(operationInfo, i, inOp);
                if (!isAccumEdge) {
                    continue;
                }
                mergePair.emplace_back(i, inOp);
                APASS_LOG_DEBUG_F(Elements::Operation, "Combine %d and %d for MulAcc accum edge in building SuperNode.",
                                  opList[i]->GetOpMagic(), opList[inOp]->GetOpMagic());
            }
        }
        // 沿 L0C 数据边递归合并非 MatMul 后继；MatMul 累加依赖已由上面的 MulAcc 逻辑处理。
        std::vector<std::pair<int32_t, int32_t>> pending;
        for (int32_t outOp : operationInfo->outGraph_[i]) {
            pending.emplace_back(i, outOp);
        }
        while (!pending.empty()) {
            auto [inOp, outOp] = pending.back();
            pending.pop_back();
            if (HasL0CDataEdge(opList[inOp], opList[outOp]) &&
                OpcodeManager::Inst().GetOpCalcType(opList[outOp]->GetOpcode()) != OpCalcType::MATMUL) {
                mergePair.emplace_back(i, outOp);
                APASS_LOG_DEBUG_F(Elements::Operation, "Combine MatMul %d and L0C successor %d in building SuperNode.",
                                  opList[i]->GetOpMagic(), opList[outOp]->GetOpMagic());
                for (int32_t nextOp : operationInfo->outGraph_[outOp]) {
                    pending.emplace_back(outOp, nextOp);
                }
            }
        }
        return true;
    }
    return false;
}

bool SuperNodeGraphBuilder::ExpandCombine(const std::shared_ptr<OperationGraphInfo> operationInfo,
                                          std::vector<Operation*>& opList, int32_t i,
                                          std::vector<std::pair<int32_t, int32_t>>& mergePair)
{
    if (i < 0 || i >= static_cast<int32_t>(opList.size())) {
        return false;
    }
    // Expand operation with only one child
    if (opList[i]->GetOpcode() == Opcode::OP_EXPAND) {
        if (operationInfo->outGraph_[i].size() == 1U) {
            mergePair.emplace_back(i, *(operationInfo->outGraph_[i].begin()));
            APASS_LOG_DEBUG_F(Elements::Operation, "Combine %d and %d for Expand in building SuperNode.",
                              opList[i]->GetOpMagic(), opList[*(operationInfo->outGraph_[i].begin())]->GetOpMagic());
            return true;
        }
    }
    return false;
}

bool SuperNodeGraphBuilder::AssembleToCopyoutScene(Operation* op)
{
    auto assembleIn = op->iOperand.front();
    auto parentOp = *assembleIn->GetProducers().begin();
    if (op->iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR ||
        op->oOperand.front()->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR ||
        parentOp->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT || parentOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
        return false;
    }
    return true;
}

inline bool FindScopeFromProducers(Operation* op, Operation::ScopeInfo& foundScope)
{
    for (auto& input : op->GetIOperands()) {
        for (auto* producer : input->GetProducers()) {
            if (producer->BelongTo() != op->BelongTo()) {
                continue;
            }
            if (producer->GetScopeId() != -1) {
                foundScope = producer->GetScopeInfo();
                return true;
            }
        }
    }
    return false;
}

inline bool FindScopeFromConsumers(Operation* op, Operation::ScopeInfo& foundScope)
{
    for (auto& output : op->GetOOperands()) {
        for (auto* consumer : output->GetConsumers()) {
            if (consumer->BelongTo() != op->BelongTo()) {
                continue;
            }
            if (consumer->GetScopeId() != -1) {
                foundScope = consumer->GetScopeInfo();
                return true;
            }
        }
    }
    return false;
}

inline void PropagateScopeInfo(std::vector<Operation*>& opList)
{
    // Assemble-like ops inherit scope from producers, view-like ops from consumers.
    // Pending list is in topological order; reversing deferred before retry
    // resolves view-like chains (tail-first) in one additional pass.
    std::vector<Operation*> pending;
    for (Operation* op : opList) {
        if (!IsAssembleLike(op->GetOpcode()) && !IsViewLike(op->GetOpcode())) {
            continue;
        }
        if (op->GetScopeId() == -1) {
            pending.push_back(op);
            continue;
        }
        Operation::ScopeInfo neighbourScope{};
        bool found = IsAssembleLike(op->GetOpcode()) ? FindScopeFromProducers(op, neighbourScope) :
                                                       FindScopeFromConsumers(op, neighbourScope);
        if (found && neighbourScope.scopeId != op->GetScopeId()) {
            pending.push_back(op);
        }
    }

    while (!pending.empty()) {
        std::vector<Operation*> deferred;
        for (Operation* op : pending) {
            Operation::ScopeInfo foundScope{};
            bool found = IsAssembleLike(op->GetOpcode()) ? FindScopeFromProducers(op, foundScope) :
                                                           FindScopeFromConsumers(op, foundScope);
            if (found && foundScope.scopeId != op->GetScopeId()) {
                op->SetScopeInfo(foundScope);
            } else if (!found) {
                deferred.push_back(op);
            }
        }
        if (deferred.size() == pending.size()) {
            break; // no progress
        }
        std::reverse(deferred.begin(), deferred.end());
        pending = std::move(deferred);
    }
}

Status SuperNodeGraphBuilder::BuildSuperNodeGraph()
{
    std::vector<Operation*>& opList = operationInfo_->opList_;
    if (opList.size() != operationInfo_->inGraph_.size() || opList.size() != operationInfo_->outGraph_.size()) {
        APASS_LOG_ERROR_F(Elements::Function, "Operation inGraph and outGraph have not been initialized.");
        return FAILED;
    }
    std::vector<std::pair<int32_t, int32_t>> mergePair;
    PropagateScopeInfo(opList);
    for (size_t i = 0; i < opList.size(); i++) {
        if (ConvertCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (L1CopyInCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (AssembleCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (CopyOutCombine(operationInfo_, opList, i, mergePair, false)) {
            continue;
        }
        if (CopyInCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (MulAccCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
    }
    superNodeInfo_ = std::make_shared<NodeGraphInfo>();
    if (superNodeInfo_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Create SuperNodeInfo failed.");
        return FAILED;
    }
    if (superNodeInfo_->Build(operationInfo_, mergePair, true) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Build SuperNodeInfo Failed.");
        return FAILED;
    }
    return ProcessScopeMerge();
}

void NodeGraphInfo::BuildNodeMapping(const std::shared_ptr<OperationGraphInfo> operationGraphInfo)
{
    int32_t numNodes = static_cast<int32_t>(node2Op_.size());
    op2Node_.resize(operationGraphInfo->opList_.size());
    nodeScope_.assign(numNodes, Operation::ScopeInfo());
    nodeCycles_.assign(numNodes, 0);
    for (int32_t nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
        bool allSkipCode = std::all_of(node2Op_[nodeIdx].begin(), node2Op_[nodeIdx].end(), [&](int32_t opIdx) {
            return nodeScopeSkipCode.count(operationGraphInfo->opList_[opIdx]->GetOpcode()) > 0;
        });
        for (int32_t opIdx : node2Op_[nodeIdx]) {
            op2Node_[opIdx] = nodeIdx;
            const auto& op = operationGraphInfo->opList_[opIdx];
            const auto& scopeInfo = op->GetScopeInfo();
            if (scopeInfo.scopeId != -1) {
                bool isSkipCode = nodeScopeSkipCode.count(op->GetOpcode()) > 0;
                if (!isSkipCode || allSkipCode) {
                    nodeScope_[nodeIdx] = scopeInfo;
                }
            }
            nodeCycles_[nodeIdx] += op->GetLatency();
        }
    }
}

SuperNodeGraphBuilder::ScopeCollectResult SuperNodeGraphBuilder::CollectScopeInfo(int32_t numNodes)
{
    ScopeCollectResult result;
    for (int32_t nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
        const auto& scopeInfo = superNodeInfo_->nodeScope_[nodeIdx];
        if (scopeInfo.scopeId == -1) {
            continue;
        }
        result.scope2Nodes[scopeInfo.scopeId].push_back(nodeIdx);
        if (scopeInfo.allowParallelMerge) {
            result.scopeAllowParallel[scopeInfo.scopeId] = true;
        }
    }
    for (size_t opIdx = 0; opIdx < operationInfo_->opList_.size(); opIdx++) {
        const auto& scopeInfo = operationInfo_->opList_[opIdx]->GetScopeInfo();
        if (scopeInfo.scopeId == -1) {
            continue;
        }
        if (nodeScopeSkipCode.count(operationInfo_->opList_[opIdx]->GetOpcode()) > 0) {
            continue;
        }
        bool isCube = operationInfo_->opList_[opIdx]->HasAttr(OpAttributeKey::isCube) &&
                      operationInfo_->opList_[opIdx]->GetBoolAttribute(OpAttributeKey::isCube);
        if (isCube) {
            result.scopeCoreTypes[scopeInfo.scopeId].hasCube = true;
        } else {
            result.scopeCoreTypes[scopeInfo.scopeId].hasVector = true;
        }
    }
    return result;
}

Status SuperNodeGraphBuilder::ValidateScopeCoreTypes(int32_t scopeId, const ScopeCoreTypeInfo& coreTypeInfo,
                                                     bool isCVMix, std::map<int32_t, int32_t>& scopeToCvFuseId)
{
    if (!coreTypeInfo.hasCube || !coreTypeInfo.hasVector) {
        return SUCCESS;
    }
    if (isCVMix) {
        scopeToCvFuseId[scopeId] = nextCvFuseId_++;
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Function, "Cannot mix cube and vector op on a CV separate platform, scopeId=%d",
                      scopeId);
    return FAILED;
}

void SuperNodeGraphBuilder::MergeScopeNodesParallel(const std::vector<int32_t>& nodes, int32_t scopeId,
                                                    std::vector<int32_t>& snParent, bool& needRebuild)
{
    int32_t firstNode = -1;
    int32_t p1 = -1;
    for (int32_t nodeIdx : nodes) {
        if (firstNode == -1) {
            firstNode = nodeIdx;
            p1 = FindParent(snParent, firstNode);
        } else {
            int32_t p2 = FindParent(snParent, nodeIdx);
            snParent[p2] = p1;
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Combine %d and %d for ScopeMerge(parallel) scopeId=%d in building SuperNode.",
                              operationInfo_->opList_[superNodeInfo_->node2Op_[nodeIdx][0]]->GetOpMagic(),
                              operationInfo_->opList_[superNodeInfo_->node2Op_[firstNode][0]]->GetOpMagic(), scopeId);
            needRebuild = true;
        }
    }
}

void SuperNodeGraphBuilder::MergeScopeNodesSequential(const std::vector<int32_t>& nodes, int32_t scopeId,
                                                      std::vector<int32_t>& snParent, bool& needRebuild)
{
    for (int32_t nodeIdx : nodes) {
        int32_t p1 = FindParent(snParent, nodeIdx);
        for (int32_t outNodeIdx : superNodeInfo_->nodeOutGraphList_[nodeIdx]) {
            if (superNodeInfo_->nodeScope_[outNodeIdx].scopeId == scopeId) {
                int32_t p2 = FindParent(snParent, outNodeIdx);
                snParent[p2] = p1;
                APASS_LOG_DEBUG_F(Elements::Operation,
                                  "Combine %d and %d for ScopeMerge scopeId=%d in building SuperNode.",
                                  operationInfo_->opList_[superNodeInfo_->node2Op_[outNodeIdx][0]]->GetOpMagic(),
                                  operationInfo_->opList_[superNodeInfo_->node2Op_[nodeIdx][0]]->GetOpMagic(), scopeId);
                needRebuild = true;
            }
        }
    }
}

Status SuperNodeGraphBuilder::CheckAndMergeScopes(const ScopeCollectResult& scopeInfo, std::vector<int32_t>& snParent,
                                                  bool& needRebuild, std::map<int32_t, int32_t>& scopeToCvFuseId)
{
    bool isCVMix = GraphUtils::IsCVMixPlatform();
    for (auto& [scopeId, coreTypeInfo] : scopeInfo.scopeCoreTypes) {
        if (ValidateScopeCoreTypes(scopeId, coreTypeInfo, isCVMix, scopeToCvFuseId) != SUCCESS) {
            return FAILED;
        }
        if (coreTypeInfo.hasCube && coreTypeInfo.hasVector) {
            continue;
        }
        auto nodesIt = scopeInfo.scope2Nodes.find(scopeId);
        if (nodesIt == scopeInfo.scope2Nodes.end()) {
            APASS_LOG_DEBUG_F(Elements::Operation, "ScopeId=%d has no supernode to merge, skip.", scopeId);
            continue;
        }
        bool allowParallel = scopeInfo.scopeAllowParallel.count(scopeId) > 0 &&
                             scopeInfo.scopeAllowParallel.at(scopeId);
        if (allowParallel) {
            MergeScopeNodesParallel(nodesIt->second, scopeId, snParent, needRebuild);
        } else {
            MergeScopeNodesSequential(nodesIt->second, scopeId, snParent, needRebuild);
        }
    }
    return SUCCESS;
}

Status SuperNodeGraphBuilder::RebuildSuperNodes(std::vector<int32_t>& snParent, int32_t numNodes)
{
    std::vector<int32_t> parentToNewNode(numNodes, -1);
    std::vector<std::vector<int32_t>> newNode2Op;

    for (int32_t nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
        int32_t p = FindParent(snParent, nodeIdx);
        if (parentToNewNode[p] == -1) {
            parentToNewNode[p] = static_cast<int32_t>(newNode2Op.size());
            newNode2Op.push_back({});
        }
        for (int32_t opIdx : superNodeInfo_->node2Op_[nodeIdx]) {
            newNode2Op[parentToNewNode[p]].push_back(opIdx);
        }
    }

    std::vector<int32_t> opParent(operationInfo_->opList_.size());
    for (int32_t i = 0; i < static_cast<int32_t>(operationInfo_->opList_.size()); i++) {
        opParent[i] = i;
    }
    for (const auto& nodeOps : newNode2Op) {
        if (nodeOps.empty()) {
            continue;
        }
        int32_t firstOp = nodeOps[0];
        for (int32_t opIdx : nodeOps) {
            opParent[opIdx] = firstOp;
        }
    }
    bool updated = true;
    while (updated) {
        updated = false;
        if (superNodeInfo_->AvoidLoop(operationInfo_, opParent, newNode2Op, updated) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Avoid loop after scope merge failed.");
            return FAILED;
        }
    }

    superNodeInfo_->node2Op_ = std::move(newNode2Op);
    superNodeInfo_->BuildNodeMapping(operationInfo_);
    superNodeInfo_->BuildInOutGraph(operationInfo_);
    superNodeInfo_->SetNodeCoreTypeAndMergeable(operationInfo_, false);
    return SUCCESS;
}

void SuperNodeGraphBuilder::ApplyCvFuseIds(const std::map<int32_t, int32_t>& scopeToCvFuseId)
{
    // 遍历所有supernode，若supernode中存在任一op的scopeId在scopeToCvFuseId中，
    // 则将该supernode中所有op标记为该scope对应的cvFuseId
    for (size_t nodeIdx = 0; nodeIdx < superNodeInfo_->node2Op_.size(); nodeIdx++) {
        int32_t cvFuseId = -1;
        for (int32_t opIdx : superNodeInfo_->node2Op_[nodeIdx]) {
            int32_t scopeId = operationInfo_->opList_[opIdx]->GetScopeId();
            auto it = scopeToCvFuseId.find(scopeId);
            if (it != scopeToCvFuseId.end()) {
                cvFuseId = it->second;
                break;
            }
        }
        if (cvFuseId != -1) {
            for (int32_t opIdx : superNodeInfo_->node2Op_[nodeIdx]) {
                operationInfo_->opList_[opIdx]->scopeInfo_.SetCvFuseId(cvFuseId);
            }
        }
    }
}

Status SuperNodeGraphBuilder::ProcessScopeMerge()
{
    int32_t numNodes = static_cast<int32_t>(superNodeInfo_->node2Op_.size());
    auto scopeInfo = CollectScopeInfo(numNodes);

    std::vector<int32_t> snParent(numNodes);
    for (int32_t i = 0; i < numNodes; i++) {
        snParent[i] = i;
    }

    bool needRebuild = false;
    std::map<int32_t, int32_t> scopeToCvFuseId;
    Status ret = CheckAndMergeScopes(scopeInfo, snParent, needRebuild, scopeToCvFuseId);
    if (ret != SUCCESS) {
        return ret;
    }

    if (needRebuild) {
        if (RebuildSuperNodes(snParent, numNodes) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Rebuild supernodes after scope merge failed.");
            return FAILED;
        }
        scopeInfo = CollectScopeInfo(static_cast<int32_t>(superNodeInfo_->node2Op_.size()));
    }

    if (GraphUtils::IsCVMixPlatform()) {
        ApplyCvFuseIds(scopeToCvFuseId);
    }
    return SUCCESS;
}

uint64_t SuperNodeGraphBuilder::CombineHash(const uint64_t h1, const uint64_t h2) const
{
    const uint64_t kMul = 0x9ddfea08eb382d69ULL;
    constexpr int kHashShiftBits = 47; // CityHash 风格 hash mixing 的位移位数
    uint64_t a = (h1 ^ h2) * kMul;
    a ^= (a >> kHashShiftBits);
    uint64_t b = (h2 ^ a) * kMul;
    b ^= (b >> kHashShiftBits);
    b *= kMul;
    return b;
}

std::vector<std::pair<int32_t, int32_t>> SuperNodeGraphBuilder::GetReduceNodeMergePair() const
{
    std::unordered_set<Opcode> reduceType{Opcode::OP_PAIRMAX, Opcode::OP_PAIRMIN, Opcode::OP_PAIRSUM};
    std::vector<Operation*>& opList = operationInfo_->opList_;
    std::vector<std::pair<int32_t, int32_t>> mergePair;
    for (size_t i = 0; i < opList.size(); i++) {
        if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MATMUL) {
            for (auto inOp : operationInfo_->inGraph_[i]) {
                if (OpcodeManager::Inst().GetOpCalcType(opList[inOp]->GetOpcode()) == OpCalcType::MATMUL) {
                    mergePair.emplace_back(i, inOp);
                    APASS_LOG_DEBUG_F(Elements::Operation, "Combine %d and %d for MulAcc in building ReduceNode.",
                                      opList[i]->GetOpMagic(), opList[inOp]->GetOpMagic());
                }
            }
            continue;
        }
        if (reduceType.count(opList[i]->GetOpcode()) > 0 && operationInfo_->outGraph_[i].size() == 1 &&
            opList[i]->GetOpcode() == opList[*(operationInfo_->outGraph_[i].begin())]->GetOpcode()) {
            mergePair.emplace_back(i, *(operationInfo_->outGraph_[i].begin()));
            APASS_LOG_DEBUG_F(Elements::Operation, "Combine %d and %d for Reduce AIV Operation in building ReduceNode.",
                              opList[i]->GetOpMagic(), opList[*(operationInfo_->outGraph_[i].begin())]->GetOpMagic());
        }
    }
    return mergePair;
}

void SuperNodeGraphBuilder::ComputeDirectionalNodeHash(std::shared_ptr<NodeGraphInfo> reduceNodeInfo,
                                                       const std::vector<uint64_t>& reduceNodeHashList,
                                                       std::vector<uint64_t>& hashList, bool reverse)
{
    const auto& neighbourGraph = reverse ? reduceNodeInfo->nodeOutGraphList_ : reduceNodeInfo->nodeInGraphList_;
    int32_t start = reverse ? static_cast<int32_t>(operationInfo_->opList_.size()) - 1 : 0;
    int32_t end = reverse ? -1 : static_cast<int32_t>(operationInfo_->opList_.size());
    int32_t step = reverse ? -1 : 1;
    for (int32_t i = start; i != end; i += step) {
        int32_t nodeIdx = reduceNodeInfo->op2Node_[i];
        if (hashList[nodeIdx] != 0) {
            continue;
        }
        hashList[nodeIdx] = reduceNodeHashList[nodeIdx];
        uint64_t xorHash = 0;
        uint64_t sumHash = 0;
        for (int32_t j : neighbourGraph[nodeIdx]) {
            uint64_t neighbourHash = hashList[j];
            xorHash ^= neighbourHash;
            sumHash += neighbourHash;
        }
        // HASH_SUM_MIXER is an odd 64-bit constant from murmur3-style finalizers
        // that disperses sumHash bits so the XOR with xorHash is order-independent
        // yet less likely to collide.
        constexpr uint64_t HASH_SUM_MIXER = 0x9ddfea08eb382d69ULL;
        hashList[nodeIdx] = CombineHash(hashList[nodeIdx], xorHash ^ (sumHash * HASH_SUM_MIXER));
    }
}

Status SuperNodeGraphBuilder::BuildReduceNodeHash(std::shared_ptr<NodeGraphInfo> reduceNodeInfo)
{
    std::vector<uint64_t> reduceNodeHashListFront(reduceNodeInfo->node2Op_.size(), 0);
    std::vector<uint64_t> reduceNodeHashListBack(reduceNodeInfo->node2Op_.size(), 0);
    std::vector<uint64_t> reduceNodeHashList(reduceNodeInfo->node2Op_.size(), 0);
    if (operationInfo_->opHashList_.size() != reduceNodeInfo->op2Node_.size()) {
        APASS_LOG_ERROR_F(Elements::Function, "Operation number mismatch in OperationInfo and ReduceNodeInfo.");
        return FAILED;
    }
    for (size_t i = 0; i < reduceNodeInfo->node2Op_.size(); i++) {
        reduceNodeHashList[i] = 0;
        for (int32_t opInNode : reduceNodeInfo->node2Op_[i]) {
            reduceNodeHashList[i] = CombineHash(reduceNodeHashList[i], operationInfo_->opHashList_[opInNode]);
        }
    }
    ComputeDirectionalNodeHash(reduceNodeInfo, reduceNodeHashList, reduceNodeHashListFront, false);
    ComputeDirectionalNodeHash(reduceNodeInfo, reduceNodeHashList, reduceNodeHashListBack, true);
    for (size_t i = 0; i < reduceNodeInfo->node2Op_.size(); i++) {
        reduceNodeHashList[i] = CombineHash(reduceNodeHashListFront[i], reduceNodeHashListBack[i]);
    }
    reduceNodeInfo->nodeHashList_ = reduceNodeHashList;
    return SUCCESS;
}

Status SuperNodeGraphBuilder::BuildBalanceOpHash(std::vector<uint64_t>& opHashList)
{
    std::vector<std::pair<int32_t, int32_t>> mergePair = GetReduceNodeMergePair();
    std::shared_ptr<NodeGraphInfo> reduceNodeInfo = std::make_shared<NodeGraphInfo>();
    if (reduceNodeInfo == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Create ReduceNodeInfo failed.");
        return FAILED;
    }
    reduceNodeInfo->Build(operationInfo_, mergePair, false);
    BuildReduceNodeHash(reduceNodeInfo);
    std::vector<uint64_t> opHashListFrontBack(operationInfo_->opList_.size(), 0);
    for (size_t i = 0; i < reduceNodeInfo->node2Op_.size(); i++) {
        if (reduceNodeInfo->node2Op_[i].size() == 1) {
            opHashListFrontBack[reduceNodeInfo->node2Op_[i][0]] = reduceNodeInfo->nodeHashList_[i];
            continue;
        }
        std::vector<int32_t>& localOps = reduceNodeInfo->node2Op_[i];
        std::unordered_map<int32_t, uint64_t> localFront;
        std::unordered_map<int32_t, uint64_t> localBack;
        for (size_t localIdx = 0; localIdx < localOps.size(); localIdx++) {
            int32_t localOpIdx = localOps[localIdx];
            localFront[localOpIdx] = operationInfo_->opHashList_[localOpIdx];
            for (int32_t publisherOpIdx : operationInfo_->inGraph_[localOpIdx]) {
                if (localFront.count(publisherOpIdx) > 0) {
                    localFront[localOpIdx] = CombineHash(localFront[localOpIdx], localFront[publisherOpIdx]);
                }
            }
        }
        for (int32_t localIdx = static_cast<int32_t>(localOps.size()) - 1; localIdx >= 0; localIdx--) {
            int32_t localOpIdx = localOps[localIdx];
            localBack[localOpIdx] = operationInfo_->opHashList_[localOpIdx];
            for (int32_t consumerOpIdx : operationInfo_->outGraph_[localOpIdx]) {
                if (localBack.count(consumerOpIdx) > 0) {
                    localBack[localOpIdx] = CombineHash(localBack[localOpIdx], localBack[consumerOpIdx]);
                }
            }
        }
        for (size_t localIdx = 0; localIdx < localOps.size(); localIdx++) {
            int32_t localOpIdx = localOps[localIdx];
            uint64_t localHash = CombineHash(localFront[localOpIdx], localBack[localOpIdx]);
            opHashListFrontBack[localOpIdx] = CombineHash(reduceNodeInfo->nodeHashList_[i], localHash);
        }
    }
    opHashList = opHashListFrontBack;
    return SUCCESS;
}

Status SuperNodeGraphBuilder::BuildHashValues()
{
    std::vector<uint64_t> opHashList;
    if (useReduceBalanceHash_) {
        if (BuildBalanceOpHash(opHashList) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "BuildBalanceOpHash failed.");
            return FAILED;
        }
    } else {
        std::vector<uint64_t> opHashListFront(operationInfo_->opList_.size(), 0);
        std::vector<uint64_t> opHashListBack(operationInfo_->opList_.size(), 0);
        std::vector<uint64_t> opHashListFrontBack(operationInfo_->opList_.size(), 0);
        for (size_t i = 0; i < operationInfo_->opList_.size(); i++) {
            opHashListFront[i] = operationInfo_->opHashList_[i];
            for (int32_t j : operationInfo_->inGraph_[i]) {
                opHashListFront[i] = CombineHash(opHashListFront[i], opHashListFront[j]);
            }
        }
        for (int32_t i = static_cast<int32_t>(operationInfo_->opList_.size() - 1); i >= 0; i--) {
            opHashListBack[i] = operationInfo_->opHashList_[i];
            for (int32_t j : operationInfo_->outGraph_[i]) {
                std::set<OpCoreType> coreTypes{operationInfo_->opCoreType_[i], operationInfo_->opCoreType_[j]};
                if (!operationInfo_->CoreTypeMergeable(coreTypes)) {
                    continue;
                }
                opHashListBack[i] = CombineHash(opHashListBack[i], opHashListBack[j]);
            }
        }
        for (size_t i = 0; i < operationInfo_->opList_.size(); i++) {
            opHashListFrontBack[i] = CombineHash(opHashListFront[i], opHashListBack[i]);
        }
        opHashList.swap(opHashListFrontBack);
    }
    if (superNodeInfo_->op2Node_.size() != operationInfo_->opList_.size()) {
        APASS_LOG_ERROR_F(Elements::Function, "Operation number mismatch in SuperNodeInfo and OperationInfo.");
        return FAILED;
    }
    int32_t numNode = superNodeInfo_->node2Op_.size();
    superNodeInfo_->nodeHashList_.resize(numNode);
    for (int32_t i = 0; i < numNode; i++) {
        superNodeInfo_->nodeHashList_[i] = 0;
        for (int32_t opIdx : superNodeInfo_->node2Op_[i]) {
            superNodeInfo_->nodeHashList_[i] = CombineHash(superNodeInfo_->nodeHashList_[i], opHashList[opIdx]);
        }
    }
    for (int32_t i = 0; i < numNode; i++) {
        superNodeInfo_->hash2NodeMap_[superNodeInfo_->nodeHashList_[i]].push_back(i);
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
