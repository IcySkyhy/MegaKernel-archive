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
 * \file mix_internal_components_analyzer.h
 * \brief 用于PASS间继承和传播mix子图内scope属性
 */

#ifndef MIX_INTERNAL_COMPONENTS_ANALYZER_H
#define MIX_INTERNAL_COMPONENTS_ANALYZER_H

#include "passes/block_graph_pass/mix_subgraph_split/mix_subgraph_split_utils.h"

namespace npu {
namespace tile_fwk {

class MixInternalComponentsAnalyzer {
public:
    // 处理internalSubgraphIDs的继承和传播
    Status AnalyzeInternalComponents(Function& mixSubgraphFunc,
                                     std::vector<InternalComponentInfo>& internalComponents) const;

private:
    // 前校验：校验所有非同步OP的subgraphId属性非空
    Status PreCheckSubGraphIDs(Function& mixSubgraphFunc) const;
    // 后校验：校验所有OP的subgraphId属性非空
    Status PostCheckSubGraphIDs(Function& mixSubgraphFunc) const;

    bool IsSyncOperation(Operation* op) const;

    // 处理InternalSubgraphIDs的传播继承和传播
    Status ProcessInternalSubgraphIDs(Function& mixSubgraphFunc,
                                      std::map<int, std::vector<Operation*>>& componentsByInternalID) const;
    // 处理已分组op
    std::map<int, std::vector<Operation*>> GroupOperationsByExistingInternalID(
        Function& mixSubgraphFunc, std::vector<Operation*>& unassignedOps) const;
    // 处理未分组的op
    void ProcessUnassignedOperations(std::vector<Operation*>& unassignedOps,
                                     std::map<int, std::vector<Operation*>>& componentsByInternalID,
                                     Function& mixSubgraphFunc) const;
    void AssignSyncOpByAIVCore(Operation* syncOp, std::map<int, std::vector<Operation*>>& componentsByInternalID,
                               std::unordered_map<Operation*, int>& opToComponentMap,
                               std::map<AIVCore, std::vector<Operation*>>& unmatchedSyncOpsByAIVCore) const;
    void CreateComponentsForUnmatchedSyncOps(
        std::map<int, std::vector<Operation*>>& componentsByInternalID,
        std::unordered_map<Operation*, int>& opToComponentMap,
        const std::map<AIVCore, std::vector<Operation*>>& unmatchedSyncOpsByAIVCore) const;
    // 获取组件的有效AIVCore（基于组内非同步op的AIVCore）
    AIVCore GetComponentAIVCore(const std::vector<Operation*>& operations) const;
    // 合并同步op
    bool MergeSyncOperation(Operation* op, std::map<int, std::vector<Operation*>>& componentsByInternalID,
                            std::unordered_map<Operation*, int>& opToComponentMap, Function& mixSubgraphFunc) const;

    bool MergeSyncPhase2(Operation* op, Function& mixSubgraphFunc,
                         std::map<int, std::vector<Operation*>>& componentsByInternalID,
                         std::unordered_map<Operation*, int>& opToComponentMap) const;
    bool MergeSyncPhase1(Operation* op, Function& mixSubgraphFunc,
                         std::map<int, std::vector<Operation*>>& componentsByInternalID,
                         std::unordered_map<Operation*, int>& opToComponentMap) const;
    bool MergeSyncSrcDst(Operation* op, Operation* targetOp,
                         std::map<int, std::vector<Operation*>>& componentsByInternalID,
                         std::unordered_map<Operation*, int>& opToComponentMap) const;

    // 搜索函数
    Operation* FindFirstOpForward(Operation* startOp, Function& mixSubgraphFunc,
                                  std::function<bool(Operation*)> predicate) const;
    Operation* FindFirstOpBackward(Operation* startOp, Function& mixSubgraphFunc,
                                   std::function<bool(Operation*)> predicate) const;

    // 处理 componentType 属性
    Status DetermineComponentType(const InternalComponentInfo& component, ComponentType& componentType) const;
    Status DetermineAllSyncComponentType(const InternalComponentInfo& component, ComponentType& componentType) const;
    // 校验当前component内部的iscube属性是否一致
    bool CheckAllCubeAttrConsistent(const InternalComponentInfo& component) const;

    // 处理 AIVCore 属性
    Status DetermineComponentAIVCore(const std::vector<Operation*>& operations, ComponentType componentType,
                                     AIVCore& outAivCore) const;
    Status ProcessCubeScope(const std::vector<Operation*>& operations, int componentID) const;
    Status ProcessVecScope(const std::vector<Operation*>& operations, int componentID, AIVCore& outAivCore) const;
    Status CheckVecScopeAivCoreConsistant(const std::vector<Operation*>& operations, int componentID,
                                          AIVCore refAIVCore) const;
    // 校验函数：校验L0C_COPY_UB的消费者v_scope的AIVCore属性一致性
    Status CheckL0CCopyUBConsumerAIVCoreConsistency(Operation* copyOp, int componentID) const;
    // 获取L0C_COPY_UB输出tensor的第一个V_SCOPE消费者的AIVCore属性
    AIVCore FindConsumerVectorAIVCore(Operation* copyOp) const;
    // 辅助函数：收集L0C_COPY_UB的所有下游V_SCOPE的AivCore到输出容器consumerAivCores
    void CollectConsumerAivCores(Operation* copyOp, std::vector<AIVCore>& consumerAivCores) const;
    // 辅助函数：从所有op（含同步）中查找第一个非UNSPECIFIED的AIVCore
    AIVCore FindFirstAIVCore(const std::vector<Operation*>& operations) const;
};
} // namespace tile_fwk
} // namespace npu

#endif // MIX_INTERNAL_COMPONENTS_ANALYZER_H
