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
 * \file mix_internal_components_analyzer.cpp
 * \brief
 */

#include "passes/pass_utils/pass_utils.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_internal_components_analyzer.h"

namespace npu {
namespace tile_fwk {
Status MixInternalComponentsAnalyzer::AnalyzeInternalComponents(
    Function& mixSubgraphFunc, std::vector<InternalComponentInfo>& internalComponents) const
{
    std::map<int, std::vector<Operation*>> componentsByInternalID;

    // step1:处理InternalSubgraphIDs的传播继承和传播
    auto status = ProcessInternalSubgraphIDs(mixSubgraphFunc, componentsByInternalID);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ProcessInternalSubgraphIDs Failed.");
        return status;
    }

    for (const auto& [internalID, operations] : componentsByInternalID) {
        std::string suffix = "_internal_" + std::to_string(internalID);
        // 向返回容器中添加component信息
        internalComponents.emplace_back(internalID, suffix, AIVCore::UNSPECIFIED);
        auto& curComponent = internalComponents.back();
        curComponent.operations = operations;

        // step2:处理 componentType 属性的传播
        ComponentType componentType = ComponentType::UNKNOWN;
        auto determineComponent = DetermineComponentType(curComponent, componentType);
        if (determineComponent != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "DetermineComponentType Failed.");
            return determineComponent;
        }
        curComponent.componentType = componentType;

        // step3:处理 aivCore 属性的传播（AIV0/AIV1/UNSPECIFIED，UNSPECIFIED表示Cube或其他类型）
        AIVCore aivCore = AIVCore::UNSPECIFIED;
        auto determineAivCore = DetermineComponentAIVCore(operations, curComponent.componentType, aivCore);
        if (determineAivCore != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "DetermineComponentAIVCore Failed.");
            return determineAivCore;
        }
        curComponent.aivCore = aivCore;

        APASS_LOG_DEBUG_F(Elements::Operation, "Internal component: internalSubgraphID=%d, operationCount=%zu.",
                          internalID, operations.size());
    }
    APASS_LOG_INFO_F(Elements::Operation, "ProcessPassDependencies success! Analyzed %zu internal components.",
                     internalComponents.size());
    return SUCCESS;
}

// 处理InternalSubgraphIDs的传播继承和传播
Status MixInternalComponentsAnalyzer::ProcessInternalSubgraphIDs(
    Function& mixSubgraphFunc, std::map<int, std::vector<Operation*>>& componentsByInternalID) const
{
    // step1:前校验
    auto precheck = PreCheckSubGraphIDs(mixSubgraphFunc);
    if (precheck != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Precheck ProcessInternalSubgraphIDs Failed.");
        return precheck;
    }

    std::vector<Operation*> unassignedOps;
    // step2: 按算子已有的internalSubgraphID做分组，收集无ID的未分配算子
    componentsByInternalID = GroupOperationsByExistingInternalID(mixSubgraphFunc, unassignedOps);

    // step3: 如果存在未分配的算子（非同步），执行同步算子合并逻辑
    if (!unassignedOps.empty()) {
        APASS_LOG_INFO_F(Elements::Operation,
                         "Found %zu operations without internalSubgraphID, using heuristic analysis",
                         unassignedOps.size());
        ProcessUnassignedOperations(unassignedOps, componentsByInternalID, mixSubgraphFunc);
    }

    // step4:后校验
    auto postcheck = PostCheckSubGraphIDs(mixSubgraphFunc);
    if (postcheck != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Postcheck ProcessInternalSubgraphIDs Failed.");
        return postcheck;
    }

    APASS_LOG_INFO_F(Elements::Operation, "Success to group operations into %zu internal components",
                     componentsByInternalID.size());
    return SUCCESS;
}

// 前校验：所有非同步op均已被标记subgraphID
Status MixInternalComponentsAnalyzer::PreCheckSubGraphIDs(Function& mixSubgraphFunc) const
{
    auto operationViewer = mixSubgraphFunc.Operations(false);
    for (size_t idx = 0; idx < operationViewer.size(); ++idx) {
        const auto& op = operationViewer[idx];
        // 跳过空操作，无需校验
        if (op.IsNOP()) {
            continue;
        }
        // 核心规则：校验所有非同步算子必须有合法的SubgraphID(>=0)
        if (!IsSyncOperation(const_cast<Operation*>(&op))) {
            int internalSubgraphID = op.GetInternalSubgraphID();
            if (internalSubgraphID < 0) {
                APASS_LOG_ERROR_F(Elements::Operation, "[PreCheck]:Invalid non-sync operation %s[%d] found!",
                                  op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "[PreCheck] Success: All non-sync operations have valid internalSubgraphID.");
    return SUCCESS;
}

// 后校验：所有op均已被标记internalSubgraphID
Status MixInternalComponentsAnalyzer::PostCheckSubGraphIDs(Function& mixSubgraphFunc) const
{
    auto operationViewer = mixSubgraphFunc.Operations(false);
    for (size_t idx = 0; idx < operationViewer.size(); ++idx) {
        const auto& op = operationViewer[idx];
        // 跳过空操作，无需校验
        if (op.IsNOP()) {
            continue;
        }
        // 核心规则：校验所有算子必须有合法的internalSubgraphID(>=0)
        int internalSubgraphID = op.GetInternalSubgraphID();
        if (internalSubgraphID < 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "[PostCheck]: Invalid operation %s[%d] found!",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "[PostCheck] Success: All operations have valid internalSubgraphID.");
    return SUCCESS;
}

std::map<int, std::vector<Operation*>> MixInternalComponentsAnalyzer::GroupOperationsByExistingInternalID(
    Function& mixSubgraphFunc, std::vector<Operation*>& unassignedOps) const
{
    std::map<int, std::vector<Operation*>> internalIDToOperations;
    auto operationViewer = mixSubgraphFunc.Operations(false);
    for (size_t idx = 0; idx < operationViewer.size(); idx++) {
        auto& op = operationViewer[idx];
        if (op.IsNOP())
            continue;
        int internalSubgraphID = op.GetInternalSubgraphID();
        if (internalSubgraphID >= 0) {
            internalIDToOperations[internalSubgraphID].push_back(&op);
            APASS_LOG_DEBUG_F(Elements::Operation, "Operation %s assigned to internalID=%d", op.GetOpcodeStr().c_str(),
                              internalSubgraphID);
        } else {
            // 没有有效的internalSubgraphID，收集到未分配列表
            unassignedOps.push_back(&op);
        }
    }
    APASS_LOG_INFO_F(Elements::Operation,
                     "Grouped operations by existing internalSubgraphID into %zu groups, %zu unassigned",
                     internalIDToOperations.size(), unassignedOps.size());
    return internalIDToOperations;
}

void MixInternalComponentsAnalyzer::AssignSyncOpByAIVCore(
    Operation* syncOp, std::map<int, std::vector<Operation*>>& componentsByInternalID,
    std::unordered_map<Operation*, int>& opToComponentMap,
    std::map<AIVCore, std::vector<Operation*>>& unmatchedSyncOpsByAIVCore) const
{
    AIVCore opAIVCore = syncOp->GetAIVCore();
    bool assignedByAIVCore = false;
    for (const auto& [compID, ops] : componentsByInternalID) {
        if (GetComponentAIVCore(ops) == opAIVCore) {
            syncOp->UpdateInternalSubgraphID(compID);
            componentsByInternalID[compID].push_back(syncOp);
            opToComponentMap[syncOp] = compID;
            assignedByAIVCore = true;
            APASS_LOG_DEBUG_F(Elements::Operation, "Sync op %s %d assigned to component %d by AIVCore (AIVCore=%d)",
                              syncOp->GetOpcodeStr().c_str(), syncOp->GetOpMagic(), compID,
                              static_cast<int>(opAIVCore));
            break;
        }
    }
    if (!assignedByAIVCore) {
        unmatchedSyncOpsByAIVCore[opAIVCore].push_back(syncOp);
    }
}

void MixInternalComponentsAnalyzer::CreateComponentsForUnmatchedSyncOps(
    std::map<int, std::vector<Operation*>>& componentsByInternalID,
    std::unordered_map<Operation*, int>& opToComponentMap,
    const std::map<AIVCore, std::vector<Operation*>>& unmatchedSyncOpsByAIVCore) const
{
    if (unmatchedSyncOpsByAIVCore.empty()) {
        return;
    }
    int maxID = -1;
    for (const auto& [id, ops] : componentsByInternalID) {
        (void)ops;
        if (id > maxID) {
            maxID = id;
        }
    }
    for (auto& [aivCore, ops] : unmatchedSyncOpsByAIVCore) {
        int newID = ++maxID;
        for (auto* op : ops) {
            op->UpdateInternalSubgraphID(newID);
            opToComponentMap[op] = newID;
        }
        componentsByInternalID[newID] = ops;
        APASS_LOG_INFO_F(Elements::Operation, "Created new component %d for %zu sync ops with AIVCore=%d", newID,
                         ops.size(), static_cast<int>(aivCore));
    }
}

void MixInternalComponentsAnalyzer::ProcessUnassignedOperations(
    std::vector<Operation*>& unassignedOps, std::map<int, std::vector<Operation*>>& componentsByInternalID,
    Function& mixSubgraphFunc) const
{
    // 预先构建op到scope的映射表，并分析现有scope的coreType
    std::unordered_map<Operation*, int> opToComponentMap;
    for (const auto& [internalID, operations] : componentsByInternalID) {
        for (auto* op : operations) {
            opToComponentMap[op] = internalID;
        }
    }
    std::vector<Operation*> remainingOps;
    std::vector<Operation*> syncOps;
    // 从未分配算子中筛选出同步算子，填充syncOps容器
    for (auto* op : unassignedOps) {
        if (IsSyncOperation(op)) {
            syncOps.push_back(op);
        } else {
            remainingOps.push_back(op);
        }
    }
    // 处理同步op
    std::map<AIVCore, std::vector<Operation*>> unmatchedSyncOpsByAIVCore;
    for (auto* syncOp : syncOps) {
        bool merged = MergeSyncOperation(syncOp, componentsByInternalID, opToComponentMap, mixSubgraphFunc);
        if (!merged) {
            AssignSyncOpByAIVCore(syncOp, componentsByInternalID, opToComponentMap, unmatchedSyncOpsByAIVCore);
        }
    }
    // 为无匹配component的sync op按AIVCore创建新component
    CreateComponentsForUnmatchedSyncOps(componentsByInternalID, opToComponentMap, unmatchedSyncOpsByAIVCore);
    // 报告未分配的op
    if (!remainingOps.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Found %zu unexpected unassigned operations after first step:", remainingOps.size());
        for (auto* op : remainingOps) {
            APASS_LOG_DEBUG_F(Elements::Operation, "  Unassigned: %s %d", op->GetOpcodeStr().c_str(), op->GetOpMagic());
        }
    }
}

AIVCore MixInternalComponentsAnalyzer::GetComponentAIVCore(const std::vector<Operation*>& operations) const
{
    for (auto* op : operations) {
        if (!IsSyncOperation(op) && op->GetAIVCore() != AIVCore::UNSPECIFIED) {
            return op->GetAIVCore();
        }
    }
    return AIVCore::UNSPECIFIED;
}

AIVCore MixInternalComponentsAnalyzer::FindFirstAIVCore(const std::vector<Operation*>& operations) const
{
    for (auto* op : operations) {
        if (op->GetAIVCore() != AIVCore::UNSPECIFIED) {
            return op->GetAIVCore();
        }
    }
    return AIVCore::UNSPECIFIED;
}

bool MixInternalComponentsAnalyzer::IsSyncOperation(Operation* op) const
{
    if (!op) {
        return false;
    }

    Opcode opcode = op->GetOpcode();

    // 同步操作类型列表
    return opcode == Opcode::OP_SYNC_SRC || opcode == Opcode::OP_SYNC_DST || opcode == Opcode::OP_CV_SYNC_SRC ||
           opcode == Opcode::OP_CV_SYNC_DST || opcode == Opcode::OP_PHASE1 || opcode == Opcode::OP_PHASE2 ||
           opcode == Opcode::OP_BAR_V || opcode == Opcode::OP_BAR_M || opcode == Opcode::OP_BAR_ALL ||
           opcode == Opcode::OP_FFTS_CROSS_CORE_SYNC || opcode == Opcode::OP_WAIT_FLAG_DEV;
}

bool MixInternalComponentsAnalyzer::MergeSyncOperation(Operation* op,
                                                       std::map<int, std::vector<Operation*>>& componentsByInternalID,
                                                       std::unordered_map<Operation*, int>& opToComponentMap,
                                                       Function& mixSubgraphFunc) const
{
    Opcode opcode = op->GetOpcode();
    // OP_PHASE2: 往前找到第一个COPY_IN放到COPY_IN所在组
    if (opcode == Opcode::OP_PHASE2) {
        return MergeSyncPhase2(op, mixSubgraphFunc, componentsByInternalID, opToComponentMap);
    }

    // OP_PHASE1: 往后找到第一个COPY_IN放到COPY_IN所在组
    if (opcode == Opcode::OP_PHASE1) {
        return MergeSyncPhase1(op, mixSubgraphFunc, componentsByInternalID, opToComponentMap);
    }

    // OP_SYNC_SRC、OP_CV_SYNC_SRC: 往前找到第一个非同步op放到该op所在分组
    if (opcode == Opcode::OP_SYNC_SRC) {
        Operation* targetSrcOp = FindFirstOpBackward(
            op, mixSubgraphFunc, [this](Operation* candidate) { return !IsSyncOperation(candidate); });
        return MergeSyncSrcDst(op, targetSrcOp, componentsByInternalID, opToComponentMap);
    }

    // BAR类、OP_SYNC_DST、OP_CV_SYNC_DST: 往后找到第一个非同步op放到该op所在分组
    if (opcode == Opcode::OP_BAR_V || opcode == Opcode::OP_BAR_M || opcode == Opcode::OP_SYNC_DST) {
        Operation* targetDstOp = FindFirstOpForward(
            op, mixSubgraphFunc, [this](Operation* candidate) { return !IsSyncOperation(candidate); });
        return MergeSyncSrcDst(op, targetDstOp, componentsByInternalID, opToComponentMap);
    }

    APASS_LOG_WARN_F(Elements::Operation, "Unhandled sync operation type: %s %d", op->GetOpcodeStr().c_str(),
                     op->GetOpMagic());
    return false;
}

bool MixInternalComponentsAnalyzer::MergeSyncPhase2(Operation* op, Function& mixSubgraphFunc,
                                                    std::map<int, std::vector<Operation*>>& componentsByInternalID,
                                                    std::unordered_map<Operation*, int>& opToComponentMap) const
{
    Operation* targetOp = FindFirstOpBackward(
        op, mixSubgraphFunc, [](Operation* candidate) { return candidate != nullptr && !candidate->IsNOP(); });
    if (targetOp) {
        auto it = opToComponentMap.find(targetOp);
        if (it != opToComponentMap.end()) {
            op->UpdateInternalSubgraphID(it->second);
            componentsByInternalID[it->second].push_back(op);
            opToComponentMap[op] = it->second;
            APASS_LOG_DEBUG_F(Elements::Operation, "Merged PHASE2 %d to component %d via previous op %d",
                              op->GetOpMagic(), it->second, targetOp->GetOpMagic());
            return true;
        }
    }
    APASS_LOG_WARN_F(Elements::Operation, "Failed to merge PHASE2 %d: no valid previous op found backward",
                     op->GetOpMagic());
    return false;
}

bool MixInternalComponentsAnalyzer::MergeSyncPhase1(Operation* op, Function& mixSubgraphFunc,
                                                    std::map<int, std::vector<Operation*>>& componentsByInternalID,
                                                    std::unordered_map<Operation*, int>& opToComponentMap) const
{
    Operation* targetOp = FindFirstOpForward(
        op, mixSubgraphFunc, [](Operation* candidate) { return candidate != nullptr && !candidate->IsNOP(); });
    if (targetOp) {
        auto it = opToComponentMap.find(targetOp);
        if (it != opToComponentMap.end()) {
            op->UpdateInternalSubgraphID(it->second);
            componentsByInternalID[it->second].push_back(op);
            opToComponentMap[op] = it->second;
            APASS_LOG_DEBUG_F(Elements::Operation, "Merged PHASE1 %d to component %d via next op %d", op->GetOpMagic(),
                              it->second, targetOp->GetOpMagic());
            return true;
        }
    }
    APASS_LOG_WARN_F(Elements::Operation, "Failed to merge PHASE1 %d: no valid next op found forward",
                     op->GetOpMagic());
    return false;
}

bool MixInternalComponentsAnalyzer::MergeSyncSrcDst(Operation* op, Operation* targetOp,
                                                    std::map<int, std::vector<Operation*>>& componentsByInternalID,
                                                    std::unordered_map<Operation*, int>& opToComponentMap) const
{
    if (targetOp) {
        auto it = opToComponentMap.find(targetOp);
        if (it != opToComponentMap.end()) {
            op->UpdateInternalSubgraphID(it->second);
            componentsByInternalID[it->second].push_back(op);
            opToComponentMap[op] = it->second;
            APASS_LOG_DEBUG_F(Elements::Operation, "Merged %s %d to component %d via non-sync op %d",
                              op->GetOpcodeStr().c_str(), op->GetOpMagic(), it->second, targetOp->GetOpMagic());
            return true;
        } else {
            // 目标op存在但尚未分配
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Cannot merge %s %d: target op %d exists but not yet assigned (will retry later)",
                              op->GetOpcodeStr().c_str(), op->GetOpMagic(), targetOp->GetOpMagic());
            return false;
        }
    }
    APASS_LOG_WARN_F(Elements::Operation, "Failed to merge %s %d: no non-sync operation found in search direction",
                     op->GetOpcodeStr().c_str(), op->GetOpMagic());
    return false;
}

Operation* MixInternalComponentsAnalyzer::FindFirstOpBackward(Operation* startOp, Function& mixSubgraphFunc,
                                                              std::function<bool(Operation*)> predicate) const
{
    const auto& opList = mixSubgraphFunc.Operations(false).DuplicatedOpList();
    int startIndex = GetStartIndex(opList, startOp);
    if (startIndex == -1) {
        return nullptr;
    }

    // 向前搜索（向序列开始方向）
    for (int i = startIndex - 1; i >= 0; --i) {
        Operation* candidate = opList[i];
        if (predicate(candidate)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Found target op %d at index %d (searching backward from %d)",
                              candidate->GetOpMagic(), i, startIndex);
            return candidate;
        }
    }

    APASS_LOG_DEBUG_F(Elements::Operation, "No matching op found for op %d in backward direction",
                      startOp->GetOpMagic());
    return nullptr;
}

Operation* MixInternalComponentsAnalyzer::FindFirstOpForward(Operation* startOp, Function& mixSubgraphFunc,
                                                             std::function<bool(Operation*)> predicate) const
{
    const auto& opList = mixSubgraphFunc.Operations(false).DuplicatedOpList();
    int startIndex = GetStartIndex(opList, startOp);
    if (startIndex == -1) {
        return nullptr;
    }

    // 向后搜索（向序列结束方向）
    for (int i = startIndex + 1; i < static_cast<int>(opList.size()); ++i) {
        Operation* candidate = opList[i];
        if (predicate(candidate)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Found target op %d at index %d (searching forward from %d)",
                              candidate->GetOpMagic(), i, startIndex);
            return candidate;
        }
    }

    APASS_LOG_DEBUG_F(Elements::Operation, "No matching op found for op %d in forward direction",
                      startOp->GetOpMagic());
    return nullptr;
}

// 处理 componentType 属性
Status MixInternalComponentsAnalyzer::DetermineComponentType(const InternalComponentInfo& component,
                                                             ComponentType& componentType) const
{
    componentType = ComponentType::UNKNOWN;
    if (component.operations.empty()) {
        APASS_LOG_WARN_F(Elements::Operation, "Empty component, cannot determine type");
        return SUCCESS;
    }
    // 增加 iscube 属性一致性校验
    bool isConsistent = CheckAllCubeAttrConsistent(component);
    if (!isConsistent) {
        APASS_LOG_ERROR_F(Elements::Operation, "[IsCubeAttr_CHECK] Component %s has inconsistent isCube attribute!",
                          component.suffix.c_str());
        return FAILED;
    }
    // 遍历所有非同步op，查找isCube属性
    bool isAllSyncOp{true};
    for (auto* op : component.operations) {
        // 跳过同步op
        if (IsSyncOperation(op)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Skipping sync op %d (opcode=%s)", op->GetOpMagic(),
                              op->GetOpcodeStr().c_str());
            continue;
        }
        isAllSyncOp = false;
        // 检查isCube属性
        if (op->HasAttribute(OpAttributeKey::isCube)) {
            bool isCube = op->GetBoolAttribute(OpAttributeKey::isCube);
            if (isCube) {
                APASS_LOG_DEBUG_F(Elements::Operation,
                                  "Component %s determined as C_SCOPE (non-sync op %d has isCube=true)",
                                  component.suffix.c_str(), op->GetOpMagic());
                componentType = ComponentType::C_SCOPE;
                return SUCCESS;
            }
        }
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Component %s determined as V_SCOPE (non-sync op %d has isCube=false or no isCube attr)",
                          component.suffix.c_str(), op->GetOpMagic());
        componentType = ComponentType::V_SCOPE;
        return SUCCESS;
    }
    if (isAllSyncOp) {
        return DetermineAllSyncComponentType(component, componentType);
    }
    return FAILED;
}

Status MixInternalComponentsAnalyzer::DetermineAllSyncComponentType(const InternalComponentInfo& component,
                                                                    ComponentType& componentType) const
{
    AIVCore aivCore = FindFirstAIVCore(component.operations);
    if (aivCore == AIVCore::UNSPECIFIED) {
        componentType = ComponentType::C_SCOPE;
    } else {
        componentType = ComponentType::V_SCOPE;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Component %s has only sync operations, determined as %s by AIVCore=%d",
                      component.suffix.c_str(), componentType == ComponentType::C_SCOPE ? "C_SCOPE" : "V_SCOPE",
                      static_cast<int>(aivCore));
    return SUCCESS;
}

// 校验当前component内部的iscube属性是否一致
bool MixInternalComponentsAnalyzer::CheckAllCubeAttrConsistent(const InternalComponentInfo& component) const
{
    bool refIsCube = false;
    bool firstNonSyncOpFound = false;
    for (auto* op : component.operations) {
        if (IsSyncOperation(op)) {
            continue;
        }
        bool curHasCube = op->HasAttribute(OpAttributeKey::isCube);
        bool curIsCube = curHasCube ? op->GetBoolAttribute(OpAttributeKey::isCube) : false;

        if (!firstNonSyncOpFound) {
            // 获取第一个非同步op的isCube属性作为基准
            refIsCube = curIsCube;
            firstNonSyncOpFound = true;
            continue;
        }
        if (refIsCube != curIsCube) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Component %s has inconsistent isCube attribute! Error op magic=%d, opcode=%s.",
                              component.suffix.c_str(), op->GetOpMagic(), op->GetOpcodeStr().c_str());
            return false;
        }
    }
    return true;
}

// 处理 AIVCore 属性
Status MixInternalComponentsAnalyzer::DetermineComponentAIVCore(const std::vector<Operation*>& operations,
                                                                ComponentType componentType, AIVCore& outAivCore) const
{
    outAivCore = AIVCore::UNSPECIFIED;
    // 空scope的AIVcore属性设置为UNSPECIFIED
    if (operations.empty()) {
        return SUCCESS;
    }
    int componentID = operations[0]->GetInternalSubgraphID();

    switch (componentType) {
        case ComponentType::C_SCOPE:
            return ProcessCubeScope(operations, componentID);
        case ComponentType::V_SCOPE:
            return ProcessVecScope(operations, componentID, outAivCore);
        default:
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Cannot determine AIVCore for component %d: all ops are sync or UNKNOWN scope",
                              operations[0]->GetInternalSubgraphID());
            return FAILED;
    }
}

Status MixInternalComponentsAnalyzer::ProcessCubeScope(const std::vector<Operation*>& operations, int componentID) const
{
    // CUBE SCOPE: 处理L0C_COPY_UB OP的subBlockIdx属性
    APASS_LOG_DEBUG_F(Elements::Operation, "Component %d is cube scope, start process L0C_COPY_UB subBlockIdx Attr.",
                      componentID);
    AIVCore targetAIVCore = AIVCore::UNSPECIFIED;
    for (auto* op : operations) {
        if (op->GetOpcode() == Opcode::OP_L0C_COPY_UB) {
            // 1. 校验L0C_COPY_UB的消费者v_scope的AIVCore属性一致性
            auto checkRet = CheckL0CCopyUBConsumerAIVCoreConsistency(op, componentID);
            if (checkRet != SUCCESS) {
                return checkRet;
            }
            // 2. 获取目标AIVCore并设置subBlockIdx属性
            targetAIVCore = FindConsumerVectorAIVCore(op);
            if (targetAIVCore != AIVCore::UNSPECIFIED) {
                int64_t subBlockIdx = (targetAIVCore == AIVCore::AIV0) ? 0 : 1;
                op->SetAttr(OpAttributeKey::subBlockIdx, subBlockIdx);
                APASS_LOG_DEBUG_F(Elements::Operation, "Set SUB_BLOCK_IDX=%ld for L0C_COPY_UB op %d", subBlockIdx,
                                  op->GetOpMagic());
            }
        }
    }
    return SUCCESS;
}

Status MixInternalComponentsAnalyzer::ProcessVecScope(const std::vector<Operation*>& operations, int componentID,
                                                      AIVCore& outAivCore) const
{
    // VEC SCOPE: 基于第一个非同步op确定AIVCore属性
    APASS_LOG_DEBUG_F(Elements::Operation, "Component %d is vec scope. Start process AIVCore", componentID);
    for (auto* op : operations) {
        if (!IsSyncOperation(op) && op->GetAIVCore() != AIVCore::UNSPECIFIED) {
            AIVCore refAIVCore = op->GetAIVCore();
            // 校验所有非同步算子的AIVCore属性一致
            Status checkRet = CheckVecScopeAivCoreConsistant(operations, componentID, refAIVCore);
            if (checkRet != SUCCESS) {
                return checkRet;
            }
            // 校验通过，设置输出并返回
            APASS_LOG_DEBUG_F(Elements::Operation, "Component AIVCore determined by op %s: AIV%d",
                              op->GetOpcodeStr().c_str(), (refAIVCore == AIVCore::AIV0 ? 0 : 1));
            outAivCore = refAIVCore;
            return SUCCESS;
        }
    }
    // 无非同步op时，从同步op的AIVCore推断
    AIVCore syncAIVCore = FindFirstAIVCore(operations);
    if (syncAIVCore != AIVCore::UNSPECIFIED) {
        outAivCore = syncAIVCore;
        APASS_LOG_DEBUG_F(Elements::Operation, "Component %d AIVCore determined by sync op: AIV%d", componentID,
                          (syncAIVCore == AIVCore::AIV0 ? 0 : 1));
    }
    return SUCCESS;
}

Status MixInternalComponentsAnalyzer::CheckVecScopeAivCoreConsistant(const std::vector<Operation*>& operations,
                                                                     int componentID, AIVCore refAIVCore) const
{
    // 校验所有非同步算子的AIVCore属性一致
    for (auto* check_op : operations) {
        // 只校验非同步算子
        if (IsSyncOperation(check_op)) {
            continue;
        }
        AIVCore check_core = check_op->GetAIVCore();
        // 非UNSPECIFIED的AIVCore必须与基准值一致
        if (check_core != AIVCore::UNSPECIFIED && check_core != refAIVCore) {
            APASS_LOG_ERROR_F(Elements::Operation, "[AIVCore_CHECK] Component %d has inconsistent AIVCore!",
                              componentID);
            return FAILED;
        }
    }
    return SUCCESS;
}

// 校验函数：校验L0C_COPY_UB的消费者v_scope的AIVCore属性一致性
Status MixInternalComponentsAnalyzer::CheckL0CCopyUBConsumerAIVCoreConsistency(Operation* copyOp, int componentID) const
{
    if (copyOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Component %d, L0C_COPY_UB op is nullptr.", componentID);
        return FAILED;
    }
    std::vector<AIVCore> consumerAivCores;
    // 收集Vector消费者的AIVCore属性
    CollectConsumerAivCores(copyOp, consumerAivCores);
    if (!consumerAivCores.empty()) {
        // 校验所有消费者AIVCore属性是否具有一致性
        AIVCore refConsumerAivCore = consumerAivCores.front();
        for (size_t i = 1; i < consumerAivCores.size(); i++) {
            if (consumerAivCores[i] != refConsumerAivCore) {
                APASS_LOG_ERROR_F(Elements::Operation, "Component %d, L0C_COPY_UB op %d has inconsistent AIVore.",
                                  componentID, copyOp->GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

// 获取L0C_COPY_UB输出tensor的第一个V_SCOPE消费者的AIVCore属性
AIVCore MixInternalComponentsAnalyzer::FindConsumerVectorAIVCore(Operation* copyOp) const
{
    std::vector<AIVCore> consumerAivCores;
    // 收集Vector消费者的AIVCore属性
    CollectConsumerAivCores(copyOp, consumerAivCores);
    if (!consumerAivCores.empty()) {
        // 直接返回第一个消费者的AIVCore属性
        return consumerAivCores.front();
    }
    return AIVCore::UNSPECIFIED;
}

// 辅助函数：收集L0C_COPY_UB的所有下游V_SCOPE的AivCore到输出容器consumerAivCores
void MixInternalComponentsAnalyzer::CollectConsumerAivCores(Operation* copyOp,
                                                            std::vector<AIVCore>& consumerAivCores) const
{
    consumerAivCores.clear();
    if (copyOp == nullptr) {
        return;
    }
    auto outputTensors = copyOp->GetOOperands();
    for (auto tensor : outputTensors) {
        if (tensor == nullptr) {
            continue;
        }
        for (auto* consumer : tensor->GetConsumers()) {
            if (consumer == nullptr) {
                continue;
            }
            AIVCore consumerCore = consumer->GetAIVCore();
            if (consumerCore != AIVCore::UNSPECIFIED) {
                consumerAivCores.push_back(consumerCore);
            }
        }
    }
}

int GetStartIndex(const std::vector<Operation*>& opList, Operation* startOp)
{
    // 找到起始op的索引
    for (size_t i = 0; i < opList.size(); ++i) {
        if (opList[i] == startOp) {
            return i;
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Start op %d not found in sequence", startOp->GetOpMagic());
    return -1;
}
} // namespace tile_fwk
} // namespace npu
