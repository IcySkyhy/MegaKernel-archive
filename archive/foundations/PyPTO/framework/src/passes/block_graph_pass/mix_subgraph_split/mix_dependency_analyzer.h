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
 * \file mix_dependency_analyzer.h
 * \brief 用于提供所需的接口
 */

#ifndef MIX_DEPENDENCY_ANALYZER_H
#define MIX_DEPENDENCY_ANALYZER_H

#include "passes/block_graph_pass/mix_subgraph_split/mix_subgraph_split_utils.h"

namespace npu {
namespace tile_fwk {
struct SimpleTensorParam {
    LogicalTensorPtr tensor;
    int opMagic;
    int operandIdx;

    SimpleTensorParam(LogicalTensorPtr t, int magic, int idx) : tensor(t), opMagic(magic), operandIdx(idx) {}
};

struct AnalyzerInput {
    std::vector<InternalComponentInfo> components;
    Function* originalMixFunc;
    AnalyzerInput(const std::vector<InternalComponentInfo>& comp, Function* func)
        : components(comp), originalMixFunc(func)
    {}
};

struct AnalyzerOutput {
    SubgraphToFunction subgraphToFunction;
    std::vector<InternalDependencyInfo> internalDeps;
    std::unordered_map<int, std::vector<SimpleTensorParam>> allIncasts;
    std::unordered_map<int, std::vector<SimpleTensorParam>> allOutcasts;
    AnalyzerOutput(const SubgraphToFunction& subFunc, const std::vector<InternalDependencyInfo>& deps,
                   const std::unordered_map<int, std::vector<SimpleTensorParam>> incasts,
                   const std::unordered_map<int, std::vector<SimpleTensorParam>> outcasts)
        : subgraphToFunction(subFunc), internalDeps(deps), allIncasts(incasts), allOutcasts(outcasts)
    {}
};

class MixDependencyAnalyzer {
public:
    // 记录直接外部依赖
    void InitSubgraphToFunction(const std::vector<InternalComponentInfo>& components);
    void InOutCastRecord(Function* originalMixFunc);
    // 1.分析组件间直接依赖
    std::unordered_map<int, std::set<int>> AnalyzeComponentDependencies(
        Function& mixFunc, std::map<std::pair<int, int>, std::vector<LogicalTensorPtr>>& crossComponentTensors);
    Status ValidateCrossComponentDependencies(
        const AnalyzerInput& input, const std::unordered_map<int, std::set<int>>& directDeps,
        const std::map<std::pair<int, int>, std::vector<LogicalTensorPtr>>& crossComponentTensors);
    // 2.计算依赖传递闭包
    void ComputeDependencyClosure(std::unordered_map<int, std::set<int>>& dependencies);
    // 3.提取外部依赖
    void ExtractExternalDependencies(const std::vector<SubfuncInvokeInfoTy>& subFuncInvokeInfos);
    // 4.传播外部依赖(基于传递闭包)
    // 看SRS-2依赖重建
    // 传递结果记录在allIncast上
    // key->component， value: 哪些incast
    void PropagateExternalDependenciesWithClosure(const std::unordered_map<int, std::set<int>>& dependencyClosure);
    // 5.收集内部依赖(C-C, V-V)
    // 只用同类型的依赖需要记录，其他的需要消除
    // 需要分析同类型的依赖有哪些
    // 最后转成控制边的依赖internalDeps
    // 先识别cube/vector, component先标上
    void CollectInternalDependencies(const std::unordered_map<int, std::set<int>>& dependencyClosure,
                                     const std::vector<InternalComponentInfo>& components);
    // 6.消除冗余依赖
    // 将多余的依赖转换成普通的依赖
    // 可以优化一下，只消除了外部的
    // 本质上就是看是否存在冗余依赖
    void EliminateRedundantDependencies();

    // 基于可达性移除冗余的外部依赖
    void EliminateRedundantOuterDeps(const std::vector<std::vector<bool>>& innerDeps,
                                     std::unordered_map<int, std::vector<SimpleTensorParam>>& allTensors);
    // 基于可达性移除冗余的内部依赖
    void EliminateRedundantInnerDeps(std::vector<std::vector<bool>>& innerDeps);
    // 外部接口
    Status ProcessDependencyAnalyzer(const AnalyzerInput& input, AnalyzerOutput& output);

private:
    // 完成闭包信息的初始化处理
    void InitDependencies(std::unordered_map<int, std::set<int>>& dependencies);
    // 使用Warshall算法
    void WarshallAlgorithm(std::vector<std::vector<bool>>& matrix);
    // 使用Warshall算法更新依赖关系
    void UpdateDependencies(std::unordered_map<int, std::set<int>>& dependencies);
    // 将可达阵退化为最小邻接阵
    void ObtainMinAdjMatrix(std::vector<std::vector<bool>>& matrix);
    // 判断是否包含对应tensor
    bool ContainsTensor(const std::vector<SimpleTensorParam>& tensors, const LogicalTensorPtr& tensor) const;
    // 构建可达阵的转置（即反向的可达阵）
    void PropagateIncastDependencies(const std::set<int>& targets, const std::vector<SimpleTensorParam>& tensorParams);
    void PropagateOutcastDependencies(int targetComp, int sourceComp);

    void Reset();
    std::vector<std::vector<bool>> Transpose(const std::vector<std::vector<bool>>& matrix);
    bool IsTensorInComponentIncasts(int compId, const LogicalTensorPtr& tensor) const;
    bool CheckDirectionAndCollectValid(const std::vector<LogicalTensorPtr>& tensors, int src, int dst,
                                       bool& hasValid) const;
    void LogIllegalBidirectionalDependency(int comp1, int comp2, const AnalyzerInput& input) const;
    void FilterNonBoundaryTensors(Function* originalMixFunc);
    int maxComponent;
    SubgraphToFunction subgraphToFunction;
    std::vector<InternalDependencyInfo> internalDeps;
    std::unordered_map<int, std::vector<SimpleTensorParam>> allIncasts;
    std::unordered_map<int, std::vector<SimpleTensorParam>> allOutcasts;
};
} // namespace tile_fwk
} // namespace npu

#endif // MIX_DEPENDENCY_ANALYZER_H
