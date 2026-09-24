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
 * \file n_buffer_merge.h
 * \brief
 */

#ifndef PASS_N_BUFFER_MERGE_H_
#define PASS_N_BUFFER_MERGE_H_

#include <functional>
#include <queue>

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/tile_graph_pass/graph_partition/hash_order_utils.h"
namespace npu::tile_fwk {

constexpr int64_t
    VEC_NBUFFER_SETTING_DEFAULT_MERGE_NUM_KEY = -1; // manualMerge模式配置默认合并粒度的key值，n个子图合并为一个

constexpr int64_t MULITY_IN_OUT_MERGE_KEY = -2; // 多输入输出子图合并配置，{-2, 0} 自动合并，{-2, 1} 手动合并

class NBufferMerge : public Pass {
public:
    NBufferMerge() : Pass("NBufferMerge") {}
    ~NBufferMerge() override = default;

private:
    // A tensor-level barrier compresses all producerColor -> consumerColor edges for one tensor.
    // This avoids expanding high fanin/fanout tensors into producer * consumer color edges.
    struct ColorTensorDependency {
        std::vector<int> consumerColors;
        std::vector<int> nonProducerConsumerColors;
        int remainingProducerCount{0};
        int remainingProducerXor{0};
    };
    static int GetMinDifferentColor(const std::vector<int>& colors, int color);
    static void AddConsumerBarrier(const std::vector<int>& producerColors, const std::vector<int>& consumerColors,
                                   ColorTensorDependency& colorDep, std::vector<int>& pendingBarrierCount,
                                   bool& hasInterColorDependency);
    static void BuildColorTensorDependencies(const RescheduleUtils::TensorDependencyMap& tensorDeps, int colorNum,
                                             std::vector<ColorTensorDependency>& colorTensorDeps,
                                             std::vector<std::vector<int>>& producerColorToDeps,
                                             std::vector<int>& pendingBarrierCount);
    using ColorQueue = std::priority_queue<int, std::vector<int>, std::greater<int>>;
    static void ReleaseColor(int color, std::vector<int>& pendingBarrierCount, ColorQueue& colorQueue);
    static void UpdateDependencyAfterProducerReady(ColorTensorDependency& colorDep, int producerColor,
                                                   std::vector<int>& pendingBarrierCount, ColorQueue& colorQueue);
    static Status RunColorBarrierTopo(int colorNum, const std::vector<std::vector<int>>& producerColorToDeps,
                                      std::vector<ColorTensorDependency>& colorTensorDeps,
                                      std::vector<int>& pendingBarrierCount, std::vector<int>& colorOrder);
    static void BuildColorDependencySummary(const OperationsViewer& opList, int colorNum,
                                            std::vector<std::vector<int>>& inColor,
                                            std::vector<std::vector<int>>& outColor);
    static Status BuildColorTopoOrder(const OperationsViewer& opList, int colorNum, std::vector<int>& colorOrder);
    static Status ApplyColorTopoOrder(OperationsViewer& opList, int colorNum);
    Status RunOnFunction(Function& function) override;
    Status NBufferMergeProcess(Function& func);
    Status Init(Function& func);
    void InitParam(OperationsViewer& opOriList);
    void GetColorHash(const Function& func, const OperationsViewer& opOriList, std::vector<uint64_t>& hashColor,
                      std::map<uint64_t, std::vector<int>>& hashMap);
    void SetHashOrderInfoOnOps(int funcMagic, const OperationsViewer& opOriList,
                               const std::map<uint64_t, std::vector<int>>& hashMap,
                               const std::set<int32_t>& mulaccGraph);
    Status CheckAndFixColorOrder(OperationsViewer& opOriList, int& color1, std::vector<int>& colorCycles1,
                                 std::vector<std::vector<int>>& colorNode1);
    void UpdateOpColor(OperationsViewer& opOriList, int& color, std::vector<int>& colorCycles,
                       std::vector<std::vector<int>>& colorNode);
    std::map<uint64_t, size_t> GetIsoColorMergeNum(const std::map<uint64_t, std::vector<int>>& hashMap) const;
    std::vector<std::vector<int>> SortColorWithInput(std::vector<int>& colorValues) const;
    Status MergeProcess(const OperationsViewer& opOriList, std::map<uint64_t, std::vector<int>>& hashMap,
                        std::map<uint64_t, size_t>& hashMergeNum, std::vector<uint64_t>& hashColor);
    void MergePingPong(std::vector<std::vector<int>>& sortedColors, const OperationsViewer& opOriList,
                       std::vector<uint64_t>& hashColor, size_t& numDBmerge, int hashOrder);
    std::map<uint64_t, size_t> SetNumDB(const Function& func, std::map<uint64_t, std::vector<int>>& hashMap);
    Status BuildHashMergeNum(const Function& func, std::map<uint64_t, std::vector<int>>& hashMap,
                             std::map<uint64_t, size_t>& hashMergeNum);
    Status RunMergeByMode(const OperationsViewer& opOriList, std::map<uint64_t, std::vector<int>>& hashMap,
                          std::map<uint64_t, size_t>& hashMergeNum, std::vector<uint64_t>& hashColor);
    void LogHashOverview(const Function& func, const std::map<uint64_t, std::vector<int>>& hashMap) const;
    void ApplyByFuncNumDB(int currentFuncMagic, std::map<uint64_t, size_t>& numDBMap,
                          std::map<uint64_t, std::vector<int>>& hashMap);
    void ApplyGlobalNumDB(std::map<uint64_t, size_t>& numDBMap, std::map<uint64_t, std::vector<int>>& hashMap);
    Status CheckVecNBufferSettingForManualMerge();
    Status MergeProcessForMulityInOut(const OperationsViewer& opOriList,
                                      const std::map<uint64_t, std::vector<int>>& hashMap,
                                      const std::map<uint64_t, size_t>& hashMergeNum, std::vector<uint64_t>& hashColor);
    Status InitVecNBufferModeBySetting();
    Status CollectSemanticLabelOverrides(const OperationsViewer& opOriList, const std::vector<uint64_t>& hashColor,
                                         std::map<uint64_t, size_t>& labelOverrides) const;
    Status ApplySemanticLabelSettings(const OperationsViewer& opOriList, std::map<uint64_t, size_t>& hashMergeNum,
                                      const std::map<uint64_t, std::vector<int>>& hashMap,
                                      const std::vector<uint64_t>& hashColor);

private:
    int colorNum_{0};
    std::vector<std::vector<int>> inColor_;
    std::vector<std::vector<int>> outColor_;
    std::vector<std::vector<int>> colorNode_;
    std::unordered_map<int, int> colorTopoOrder_;
    std::vector<int> colorCycles_;
    int vecNBuffermode_;
    int mgVecParallelLb_;
    std::map<int64_t, int64_t> vecNBufferSetting_;
    std::map<std::string, int64_t> vecNBufferSettingByFunc_;
    std::map<std::string, int64_t> vecNBufferSettingByLabel_;
    std::map<uint64_t, int> hashOrder_; // Deterministic ordered map for compilation repeatability
    enum ModeType { noMerge = 0, autoMerge = 1, manualMerge = 2, autoMulityInOutMerge = 3, manualMulityInOutMerge = 4 };
};
} // namespace npu::tile_fwk
#endif // PASS_N_BUFFER_MERGE_H_
