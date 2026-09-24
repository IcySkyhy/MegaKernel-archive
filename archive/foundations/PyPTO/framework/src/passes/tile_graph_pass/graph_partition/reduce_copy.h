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
 * \file reduce_copy.h
 * \brief
 */

#ifndef PASS_REDUCE_COPY_H_
#define PASS_REDUCE_COPY_H_

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/tensor/logical_tensor.h"
#include <vector>
#include <set>
#include <unordered_set>
#include <queue>
#include <utility>

namespace npu::tile_fwk {
struct BoundaryTensorInfo {
    int tensorMagic;
    std::vector<int> producerSubgraphs;
    std::vector<int> consumerSubgraphs;
    bool isDDR{false}; // 目的: 仅 DDR tensor 参与 inner-external-use 检查
};

struct MergeInput {
    int numSubgraph;
    int maxLatency;
    std::pair<double, double> aivRatio;
    std::vector<int> subgraphAICLatency;
    std::vector<int> subgraphAIVLatency;
    std::vector<std::set<int>> subGraphInGraph;
    std::vector<std::set<int>> subGraphOutGraph;
    std::vector<std::vector<int>> mergeGroup;
    std::vector<bool> isEnforceMergeGroup;
    std::vector<bool> isValidMergeGroup;
    std::vector<BoundaryTensorInfo> boundaryTensors;
    std::vector<std::vector<int>> subgraphToBoundaryTensorIds;
};

struct MergeOutput {
    int numSubgraphUpdated;
    std::vector<int> subgraphIdUpdated;
};

class MixGraphMerger {
public:
    MixGraphMerger() = default;
    ~MixGraphMerger() = default;
    MergeOutput Merge(const MergeInput& input);
    bool enableAutoMix{true};

private:
    MergeInput mInput;
    MergeOutput mOutput;
    std::vector<int> mParent;
    std::vector<int> mRank;
    std::vector<std::vector<int>> mRootToBoundaryTensorIds;
    std::vector<int> mTensorVisitStamp;
    int mVisitStamp{0};
    std::unordered_set<int> mGlobalOutputSinks; // 出度0子图: 不作为任何 boundary tensor producer, 即最终输出端点
    // cached merged graph (avoid redundant rebuild in CanMergeWithoutCycle)
    std::vector<std::set<int>> mCachedOutGraph;
    std::vector<std::set<int>> mCachedInGraph;
    // HasCycle scratch buffers (avoid per-call reallocation)
    std::vector<int> mInDegreeBuf;
    std::vector<bool> mIsRootBuf;
    std::queue<int> mQueueBuf;

    void Initialize(const MergeInput& input);
    void InitBoundaryTensorIndex();
    int FindParent(int x);
    void UnionSets(int x, int y);
    bool CanMergeWithoutCycle(const std::vector<int>& actualGroup);
    bool CanMergeWithConstraints(const std::vector<int>& actualGroup);
    void PerformMerge(const std::vector<int>& actualGroup);
    void UpdateBoundaryTensorIndex(const std::vector<int>& actualGroup);
    void ApplyMergeToGraph(const std::vector<int>& actualGroup);
    void MergeNodesInCachedGraph(int root, const std::set<int>& del);
    void UpdateOutput();
    bool CheckLatencyConstraint(const std::vector<int>& actualGroup);
    bool CheckMergeBenefitByStructuralPattern(const std::vector<int>& actualGroup);
    bool CheckNoExternalUseOfMergedInnerTensor(const std::vector<int>& actualGroup);
    bool IsInvalidMergedInnerTensor(int tensorId, const std::unordered_set<int>& mergedRoots);
    std::vector<int> GetActualGroup(const std::vector<int>& group);
    void BuildMergedGraph(std::vector<std::set<int>>& outGraph, std::vector<std::set<int>>& inGraph);
    bool HasCycle(const std::vector<std::set<int>>& outGraph, const std::vector<std::set<int>>& inGraph);
};

class ReduceCopyMerge : public Pass {
public:
    ReduceCopyMerge() : Pass("ReduceCopyMerge") { SetSupportedArches({NPUArch::DAV_3510}); }
    ~ReduceCopyMerge() override = default;

private:
    Status BuildGraph(Function& function, MergeInput& mergeInput);
    Status BuildMergeGroup(Function& function, MergeInput& mergeInput);
    void CombineForkSubgraph(Function& function, MergeInput& mergeInput);
    Status MarkNoMergeSubgraph(Function& function);
    void UpdateConnectRecord(Function& function, MergeInput& mergeInput);
    void UpdateBoundaryTensorSize(LogicalTensorPtr& tensor, int tensorSize);
    void RecordBoundaryTensorInfo(LogicalTensorPtr& tensor, MergeInput& mergeInput, const std::set<int>& connectGraphs);
    void UpdateMergeInput(MergeInput& mergeInput, std::multimap<int, std::vector<int>>& sortedMergeGroup);
    bool IsEnforceMergeBoundary(LogicalTensorPtr& tensor);
    Status RunOnFunction(Function& function) override;
    Status PostCheck(Function& function) override;
    std::unordered_map<int, std::vector<int>> subgraphToOutputTensors;
    std::unordered_map<int, std::vector<int>> subgraphToInputTensors;
    std::unordered_map<int, std::vector<int>> tensorToMergeGroup;
    std::unordered_set<int> noMergeSubgraph;
    std::unordered_set<int> noMergeSubgraphEnforce;
    std::map<std::vector<int>, int> mergeGroupToPriority;
    std::set<std::vector<int>> enforceMergeGroup;
    std::vector<int> subgraphInputSize;
    std::vector<int> subgraphOutputSize;
};

} // namespace npu::tile_fwk
#endif
