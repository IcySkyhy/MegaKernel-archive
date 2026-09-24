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
 * \file n_buffer_merge.cpp
 * \brief
 */

#include "n_buffer_merge.h"
#include <algorithm>
#include <climits>
#include <limits>
#include <unordered_map>

#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/parallel_tool.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/reschedule_utils.h"

#define MODULE_NAME "NBufferMerge"

namespace npu::tile_fwk {
namespace {
constexpr int INVALID_COLOR = -1;
} // namespace

int NBufferMerge::GetMinDifferentColor(const std::vector<int>& colors, int color)
{
    if (colors.empty()) {
        return INVALID_COLOR;
    }
    if (colors[0] != color) {
        return colors[0];
    }
    return colors.size() > 1 ? colors[1] : INVALID_COLOR;
}

void NBufferMerge::AddConsumerBarrier(const std::vector<int>& producerColors, const std::vector<int>& consumerColors,
                                      ColorTensorDependency& colorDep, std::vector<int>& pendingBarrierCount,
                                      bool& hasInterColorDependency)
{
    for (int consumerColor : consumerColors) {
        PASS_ASSERT(consumerColor >= 0 && consumerColor < static_cast<int>(pendingBarrierCount.size()))
            << "Consumer color is out of range: " << consumerColor;
        bool isProducer = std::binary_search(producerColors.begin(), producerColors.end(), consumerColor);
        if (!isProducer) {
            colorDep.nonProducerConsumerColors.emplace_back(consumerColor);
            pendingBarrierCount[consumerColor]++;
            hasInterColorDependency = true;
            continue;
        }
        if (producerColors.size() > 1) {
            pendingBarrierCount[consumerColor]++;
            hasInterColorDependency = true;
        }
    }
}

void NBufferMerge::BuildColorTensorDependencies(const RescheduleUtils::TensorDependencyMap& tensorDeps, int colorNum,
                                                std::vector<ColorTensorDependency>& colorTensorDeps,
                                                std::vector<std::vector<int>>& producerColorToDeps,
                                                std::vector<int>& pendingBarrierCount)
{
    producerColorToDeps.assign(colorNum, {});
    pendingBarrierCount.assign(colorNum, 0);
    colorTensorDeps.clear();
    colorTensorDeps.reserve(tensorDeps.size());
    for (const auto& [tensor, depInfo] : tensorDeps) {
        (void)tensor;
        const auto& producerColors = depInfo.producerColors;
        const auto& consumerColors = depInfo.consumerColors;
        if (producerColors.empty() || consumerColors.empty()) {
            continue;
        }
        ColorTensorDependency colorDep;
        colorDep.consumerColors = consumerColors;
        colorDep.remainingProducerCount = static_cast<int>(producerColors.size());
        for (int producerColor : producerColors) {
            colorDep.remainingProducerXor ^= producerColor;
        }
        bool hasInterColorDependency = false;
        AddConsumerBarrier(producerColors, consumerColors, colorDep, pendingBarrierCount, hasInterColorDependency);
        if (!hasInterColorDependency) {
            continue;
        }
        int depIndex = static_cast<int>(colorTensorDeps.size());
        colorTensorDeps.emplace_back(std::move(colorDep));
        for (int producerColor : producerColors) {
            PASS_ASSERT(producerColor >= 0 && producerColor < colorNum)
                << "Producer color is out of range: " << producerColor;
            producerColorToDeps[producerColor].emplace_back(depIndex);
        }
    }
}

void NBufferMerge::ReleaseColor(int color, std::vector<int>& pendingBarrierCount, ColorQueue& colorQueue)
{
    if (color < 0 || color >= static_cast<int>(pendingBarrierCount.size()) || pendingBarrierCount[color] <= 0) {
        return;
    }
    pendingBarrierCount[color]--;
    if (pendingBarrierCount[color] == 0) {
        colorQueue.push(color);
    }
}

void NBufferMerge::UpdateDependencyAfterProducerReady(ColorTensorDependency& colorDep, int producerColor,
                                                      std::vector<int>& pendingBarrierCount, ColorQueue& colorQueue)
{
    if (colorDep.remainingProducerCount <= 0) {
        return;
    }
    colorDep.remainingProducerCount--;
    // producerColors are unique; XOR removes ready producers and identifies the last remaining one in O(1).
    colorDep.remainingProducerXor ^= producerColor;
    if (colorDep.remainingProducerCount == 1) {
        int remainingColor = colorDep.remainingProducerXor;
        if (std::binary_search(colorDep.consumerColors.begin(), colorDep.consumerColors.end(), remainingColor)) {
            ReleaseColor(remainingColor, pendingBarrierCount, colorQueue);
        }
        return;
    }
    if (colorDep.remainingProducerCount == 0) {
        for (int consumerColor : colorDep.nonProducerConsumerColors) {
            ReleaseColor(consumerColor, pendingBarrierCount, colorQueue);
        }
    }
}

Status NBufferMerge::RunColorBarrierTopo(int colorNum, const std::vector<std::vector<int>>& producerColorToDeps,
                                         std::vector<ColorTensorDependency>& colorTensorDeps,
                                         std::vector<int>& pendingBarrierCount, std::vector<int>& colorOrder)
{
    colorOrder.clear();
    colorOrder.reserve(colorNum);
    ColorQueue colorQueue;
    // tensorDeps is an unordered_map, so dependency release order is not stable. Use a min-heap
    // to keep the resulting topological order deterministic across compiler invocations.
    for (int color = 0; color < colorNum; color++) {
        if (pendingBarrierCount[color] == 0) {
            colorQueue.push(color);
        }
    }
    while (!colorQueue.empty()) {
        int color = colorQueue.top();
        colorQueue.pop();
        colorOrder.emplace_back(color);
        for (int depIndex : producerColorToDeps[color]) {
            UpdateDependencyAfterProducerReady(colorTensorDeps[depIndex], color, pendingBarrierCount, colorQueue);
        }
    }
    if (static_cast<int>(colorOrder.size()) == colorNum) {
        return SUCCESS;
    }
    for (int color = 0; color < colorNum; color++) {
        if (pendingBarrierCount[color] != 0) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Color [%d] has cycle in graph; Please check and adjust the merge method.", color);
        }
    }
    return FAILED;
}

void NBufferMerge::BuildColorDependencySummary(const OperationsViewer& opList, int colorNum,
                                               std::vector<std::vector<int>>& inColor,
                                               std::vector<std::vector<int>>& outColor)
{
    inColor.assign(colorNum, {});
    outColor.assign(colorNum, {});
    auto tensorDeps = RescheduleUtils::BuildTensorDependencyInfo(opList);
    std::vector<int> minInputColor(colorNum, std::numeric_limits<int>::max());
    std::vector<int> minOutputColor(colorNum, std::numeric_limits<int>::max());
    for (const auto& [tensor, depInfo] : tensorDeps) {
        (void)tensor;
        const auto& producerColors = depInfo.producerColors;
        const auto& consumerColors = depInfo.consumerColors;
        if (producerColors.empty() || consumerColors.empty()) {
            continue;
        }
        for (int consumerColor : consumerColors) {
            int predColor = GetMinDifferentColor(producerColors, consumerColor);
            if (predColor >= 0) {
                minInputColor[consumerColor] = std::min(minInputColor[consumerColor], predColor);
            }
        }
        for (int producerColor : producerColors) {
            int succColor = GetMinDifferentColor(consumerColors, producerColor);
            if (succColor >= 0) {
                minOutputColor[producerColor] = std::min(minOutputColor[producerColor], succColor);
            }
        }
    }
    for (int color = 0; color < colorNum; color++) {
        if (minInputColor[color] != std::numeric_limits<int>::max()) {
            inColor[color].emplace_back(minInputColor[color]);
        }
        if (minOutputColor[color] != std::numeric_limits<int>::max()) {
            outColor[color].emplace_back(minOutputColor[color]);
        }
    }
}

Status NBufferMerge::BuildColorTopoOrder(const OperationsViewer& opList, int colorNum, std::vector<int>& colorOrder)
{
    auto tensorDeps = RescheduleUtils::BuildTensorDependencyInfo(opList);
    std::vector<ColorTensorDependency> colorTensorDeps;
    std::vector<std::vector<int>> producerColorToDeps(colorNum);
    std::vector<int> pendingBarrierCount(colorNum, 0);
    BuildColorTensorDependencies(tensorDeps, colorNum, colorTensorDeps, producerColorToDeps, pendingBarrierCount);
    return RunColorBarrierTopo(colorNum, producerColorToDeps, colorTensorDeps, pendingBarrierCount, colorOrder);
}

Status NBufferMerge::ApplyColorTopoOrder(OperationsViewer& opList, int colorNum)
{
    std::vector<int> colorOrder;
    if (BuildColorTopoOrder(opList, colorNum, colorOrder) == FAILED) {
        return FAILED;
    }
    std::vector<int> newSubgraphId(colorNum, INVALID_COLOR);
    for (size_t i = 0; i < colorOrder.size(); i++) {
        newSubgraphId[colorOrder[i]] = static_cast<int>(i);
    }
    for (size_t i = 0; i < opList.size(); i++) {
        int oldSubgraphId = opList[i].GetSubgraphID();
        if (oldSubgraphId < 0) {
            continue;
        }
        if (oldSubgraphId >= colorNum || newSubgraphId[oldSubgraphId] < 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "Invalid SubGraph ID %d while applying color topo order.",
                              oldSubgraphId);
            return FAILED;
        }
        opList[i].UpdateSubgraphID(newSubgraphId[oldSubgraphId]);
    }
    return SUCCESS;
}
void NBufferMerge::UpdateOpColor(OperationsViewer& opOriList, int& color, std::vector<int>& colorCycles,
                                 std::vector<std::vector<int>>& colorNode)
{
    std::vector<int> oriColor2NewColor(color);
    int colorCount = 0;
    for (int i = 0; i < color; i++) {
        if (colorCycles[i] != 0) {
            oriColor2NewColor[i] = colorCount;
            colorCount++;
        }
        colorCycles[i] = 0;
        colorNode[i].clear();
    }
    color = colorCount;
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        opOriList[i].UpdateSubgraphID(oriColor2NewColor[opOriList[i].GetSubgraphID()]);
    }
}

Status NBufferMerge::CheckAndFixColorOrder(OperationsViewer& opOriList, int& colorNum, std::vector<int>& colorCycles,
                                           std::vector<std::vector<int>>& colorNode)
{
    UpdateOpColor(opOriList, colorNum, colorCycles, colorNode);
    if (ApplyColorTopoOrder(opOriList, colorNum) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Operation, "ColorTopo failed; Please check the ColorTopo method.");
        return FAILED;
    }
    // 重新统计colorNode等
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        colorCycles[opOriList[i].GetSubgraphID()] += opOriList[i].GetLatency();
        colorNode[opOriList[i].GetSubgraphID()].push_back(i);
    }
    return SUCCESS;
}

void NBufferMerge::InitParam(OperationsViewer& opOriList)
{
    for (size_t i = 0; i < opOriList.size(); i++) {
        // 过滤FromInCast节点和NOP节点
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        int subgraphId = opOriList[i].GetSubgraphID();
        colorCycles_[subgraphId] += opOriList[i].GetLatency();
        colorNode_[subgraphId].push_back(static_cast<int>(i));
    }
    BuildColorDependencySummary(opOriList, colorNum_, inColor_, outColor_);
}

Status NBufferMerge::Init(Function& func)
{
    size_t colorMax{0U};
    std::set<int> colorSet;
    auto opOriList = func.Operations(true, SortOperationsMode::LIGHTWEIGHT);
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        colorSet.insert(opOriList[i].GetSubgraphID());
        if (opOriList[i].GetSubgraphID() > static_cast<int>(colorMax)) {
            colorMax = opOriList[i].GetSubgraphID();
        }
    }
    if (colorSet.size() == 0) {
        APASS_LOG_INFO_F(Elements::Operation, "Color size is 0, skip nbuffer merge.");
        return SUCCESS;
    }
    if (colorSet.size() != colorMax + 1) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Colors are not continously numbered from 0, func magic : %d; Please check whether the subgraph IDs are "
            "correct.",
            func.GetFuncMagic());
        return FAILED;
    }
    colorNum_ = colorMax + 1;
    colorNode_.resize(colorNum_);
    colorCycles_.resize(colorNum_, 0);
    inColor_.resize(colorNum_);
    outColor_.resize(colorNum_);
    InitParam(opOriList);
    APASS_LOG_INFO_F(Elements::Operation, "Before Nbuffer merge.");
    RescheduleUtils::PrintColorNode(func);
    return SUCCESS;
}

std::map<uint64_t, size_t> NBufferMerge::GetIsoColorMergeNum(const std::map<uint64_t, std::vector<int>>& hashMap) const
{
    std::map<uint64_t, size_t> hashCoreNum;
    for (const auto& entry : hashMap) {
        if (entry.first == 0 || entry.second.empty()) {
            continue;
        }
        if (hashCoreNum.find(entry.first) == hashCoreNum.end()) {
            hashCoreNum[entry.first] = mgVecParallelLb_;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph hash: %lu, size %zu, core num: %zu.", entry.first,
                          entry.second.size(), hashCoreNum[entry.first]);
        if (entry.second.size() <= hashCoreNum[entry.first]) {
            hashCoreNum[entry.first] = 1U;
            continue;
        }
        auto initNum = (entry.second.size() + hashCoreNum[entry.first] - 1) / hashCoreNum[entry.first];
        auto usedCore = (entry.second.size() + initNum - 1) / initNum;
        while ((usedCore < hashCoreNum[entry.first]) && (initNum > 1)) {
            initNum--;
            usedCore = (entry.second.size() + initNum - 1) / initNum;
        }
        hashCoreNum[entry.first] = initNum;
        APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph hash: %lu, merge num: %zu.", entry.first,
                          hashCoreNum[entry.first]);
    }
    return hashCoreNum;
}

void NBufferMerge::GetColorHash(const Function& func, const OperationsViewer& opOriList,
                                std::vector<uint64_t>& hashColor, std::map<uint64_t, std::vector<int>>& hashMap)
{
    std::vector<uint64_t> hashTileOp;
    RescheduleUtils::BuildInputTopoHashList(opOriList, hashTileOp);
    uint64_t a = 0x12345678;
    uint64_t p = 23;
    const uint64_t mod = 0xFFFFFFFFFFFFF;
    std::set<int32_t> mulaccGraph;
    std::unordered_map<int, int> reshapeCount;
    std::unordered_map<int, int> subgraphOpCount;
    for (size_t i = 0; i < opOriList.size(); i++) {
        int subGraphID = opOriList[i].GetSubgraphID();
        if (subGraphID < 0) {
            continue;
        }
        // 单独的reshape不用合并
        subgraphOpCount[subGraphID]++;
        if (opOriList[i].GetOpcode() == Opcode::OP_RESHAPE)
            reshapeCount[subGraphID]++;
        if (OpcodeManager::Inst().GetCoreType(opOriList[i].GetOpcode()) == OpCoreType::AIC) {
            mulaccGraph.insert(subGraphID);
            continue;
        }
        hashColor[subGraphID] = (hashColor[subGraphID] * p + (hashTileOp[i] ^ a)) % mod;
    }
    for (auto& [id, count] : reshapeCount) {
        if (count == subgraphOpCount[id])
            hashColor[id] = 0;
    }
    for (auto subgraphId : mulaccGraph) {
        hashColor[subgraphId] = 0;
    }
    int order = 0;
    for (int i = 0; i < colorNum_; i++) {
        if (mulaccGraph.count(i))
            continue;
        hashMap[hashColor[i]].push_back(i);
        if (hashMap[hashColor[i]].size() == 1) {
            hashOrder_[hashColor[i]] = order;
            order++;
        }
    }
    SetHashOrderInfoOnOps(func.GetFuncMagic(), opOriList, hashMap, mulaccGraph);
}

void NBufferMerge::SetHashOrderInfoOnOps(int funcMagic, const OperationsViewer& opOriList,
                                         const std::map<uint64_t, std::vector<int>>& hashMap,
                                         const std::set<int32_t>& mulaccGraph)
{
    for (auto& entry : hashMap) {
        int hashOrderVal = hashOrder_[entry.first];
        std::string fullHashOrder = "func" + std::to_string(funcMagic) + "_" + std::to_string(hashOrderVal);
        size_t subgraphCount = entry.second.size();
        for (auto subgraphId : entry.second) {
            if (mulaccGraph.count(subgraphId))
                continue;
            for (auto opIdx : colorNode_[subgraphId]) {
                opOriList[opIdx].SetHashOrderInfo(OpAttributeKey::vecMergeHashOrder,
                                                  OpAttributeKey::vecMergeSubgraphCount, fullHashOrder, subgraphCount);
            }
        }
    }
}

std::vector<std::vector<int>> NBufferMerge::SortColorWithInput(std::vector<int>& colorValues) const
{
    std::map<int, std::vector<int>> inColorToOutColor;
    int inCount = -1;
    for (auto color : colorValues) {
        if (inColor_[color].empty()) {
            inColorToOutColor[inCount--].push_back(color);
            continue;
        }
        for (auto inColor : inColor_[color]) {
            inColorToOutColor[inColor].push_back(color);
        }
    }
    std::map<int, std::vector<int>> outColorToInColor;
    int outCount = -1;
    for (auto color : colorValues) {
        if (outColor_[color].empty()) {
            outColorToInColor[outCount--].push_back(color);
            continue;
        }
        for (auto outColor : outColor_[color]) {
            outColorToInColor[outColor].push_back(color);
        }
    }
    std::vector<std::vector<int>> res;
    std::map<int, std::vector<int>> colorWithSameInOut = (inColorToOutColor.size() <= outColorToInColor.size()) ?
                                                             inColorToOutColor :
                                                             outColorToInColor;
    std::set<int> visitedColorSet;
    for (auto& entry : colorWithSameInOut) {
        std::vector<int> sortedColor;
        for (auto subgraphColor : entry.second) {
            if (visitedColorSet.count(subgraphColor) == 0) {
                visitedColorSet.insert(subgraphColor);
                sortedColor.push_back(subgraphColor);
            }
        }
        if (!sortedColor.empty()) {
            res.push_back(sortedColor);
        }
    }
    return res;
}

void NBufferMerge::MergePingPong(std::vector<std::vector<int>>& sortedColors, const OperationsViewer& opOriList,
                                 std::vector<uint64_t>& hashColor, size_t& numDBmerge, int hashOrder)
{
    int pingColor = -1;
    for (auto& input2Color : sortedColors) {
        APASS_LOG_DEBUG_F(Elements::Operation, "HashOrder %d, NBuffer %zu, Number of subgraphs %zu, SubGraphIDs %s",
                          hashOrder, numDBmerge, input2Color.size(), IntVecToStr(input2Color).c_str());
        if (vecNBuffermode_ == autoMulityInOutMerge || vecNBuffermode_ == manualMulityInOutMerge) {
            std::sort(input2Color.begin(), input2Color.end(),
                      [&](int x, int y) { return colorTopoOrder_[x] < colorTopoOrder_[y]; });
        }
        for (size_t i = 0; i < input2Color.size(); i++) {
            if (numDBmerge == 0) {
                continue;
            }
            if (i % numDBmerge == 0) {
                pingColor = input2Color[i];
                continue;
            }
            int pongColor = input2Color[i];
            for (auto opIdxMergedDB : colorNode_[pongColor]) {
                opOriList[opIdxMergedDB].UpdateSubgraphID(pingColor);
                colorNode_[pingColor].push_back(opIdxMergedDB);
            }
            colorCycles_[pingColor] += colorCycles_[pongColor];
            hashColor[pingColor] += hashColor[pongColor];
            colorCycles_[pongColor] = 0;
            colorNode_[pongColor].clear();
            hashColor[pongColor] = 0;
            APASS_LOG_DEBUG_F(Elements::Operation, "HashOrder %d, SubGraph Merge: %d -> %d.", hashOrder, pongColor,
                              pingColor);
        }
    }
}

Status NBufferMerge::MergeProcessForMulityInOut(const OperationsViewer& opOriList,
                                                const std::map<uint64_t, std::vector<int>>& hashMap,
                                                const std::map<uint64_t, size_t>& hashMergeNum,
                                                std::vector<uint64_t>& hashColor)
{
    std::vector<uint64_t> hashMapKeys;
    for (const auto& entry : hashMap) {
        hashMapKeys.push_back(entry.first);
    }
    std::vector<int> colorOrder;
    if (BuildColorTopoOrder(opOriList, colorNum_, colorOrder) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Operation, "MergeProcessForMulityInOut: BuildColorTopoOrder failed.");
        return FAILED;
    }
    colorTopoOrder_.clear();
    for (size_t i = 0; i < colorOrder.size(); i++) {
        colorTopoOrder_[colorOrder[i]] = static_cast<int>(i);
    }
    ParallelTool::Instance().Parallel_for(0, hashMapKeys.size(), 1, [&](int st, int et, int tid) {
        (void)tid;
        for (int hashMapKeyIdx = st; hashMapKeyIdx < et; hashMapKeyIdx++) {
            uint64_t colorHashValue = hashMapKeys[hashMapKeyIdx];
            if (colorHashValue == 0)
                continue;
            auto it = hashMap.find(colorHashValue);
            if (it == hashMap.end())
                continue;
            std::vector<int> colorValues = it->second;
            if (colorValues.empty())
                continue;
            std::vector<std::vector<int>> sortedColors;
            sortedColors.push_back(colorValues);
            APASS_LOG_INFO_F(Elements::Operation, "HashOrder %d, HashValue %lu (MulityInOut mode): subgraphs %s.",
                             hashOrder_[colorHashValue], colorHashValue, IntVecToStr(colorValues).c_str());
            size_t numDBMerge = (vecNBuffermode_ == autoMulityInOutMerge) ? hashMergeNum.at(colorHashValue) :
                                                                            hashMergeNum.at(hashOrder_[colorHashValue]);
            MergePingPong(sortedColors, opOriList, hashColor, numDBMerge, hashOrder_[colorHashValue]);
        }
    });
    return SUCCESS;
}

Status NBufferMerge::MergeProcess(const OperationsViewer& opOriList, std::map<uint64_t, std::vector<int>>& hashMap,
                                  std::map<uint64_t, size_t>& hashMergeNum, std::vector<uint64_t>& hashColor)
{
    std::vector<uint64_t> hashMapKeys;
    for (const auto& entry : hashMap) {
        hashMapKeys.push_back(entry.first);
    }
    ParallelTool::Instance().Parallel_for(0, hashMapKeys.size(), 1, [&](int st, int et, int tid) {
        (void)tid;
        for (int hashMapKeyIdx = st; hashMapKeyIdx < et; hashMapKeyIdx++) {
            uint64_t colorHashValue = hashMapKeys[hashMapKeyIdx];
            if (colorHashValue == 0)
                continue;
            std::vector<int>& colorValues = hashMap[colorHashValue];
            auto sortedColors = SortColorWithInput(colorValues);
            if (sortedColors.empty())
                continue;
            APASS_LOG_INFO_F(
                Elements::Operation, "HashOrder %d, HashValue %lu: total %zu groups, subgraphs before grouping: %s.",
                hashOrder_[colorHashValue], colorHashValue, sortedColors.size(), IntVecToStr(colorValues).c_str());
            for (size_t groupIdx = 0; groupIdx < sortedColors.size(); groupIdx++) {
                APASS_LOG_INFO_F(Elements::Operation, "  Group %zu: subgraphs %s.", groupIdx,
                                 IntVecToStr(sortedColors[groupIdx]).c_str());
            }
            size_t numDBMerge = (vecNBuffermode_ == autoMerge) ? hashMergeNum[colorHashValue] :
                                                                 hashMergeNum[hashOrder_[colorHashValue]];
            MergePingPong(sortedColors, opOriList, hashColor, numDBMerge, hashOrder_[colorHashValue]);
        }
    });
    return SUCCESS;
}

void NBufferMerge::ApplyByFuncNumDB(int currentFuncMagic, std::map<uint64_t, size_t>& numDBList,
                                    std::map<uint64_t, std::vector<int>>& hashMap)
{
    auto hashMergeNum = GetIsoColorMergeNum(hashMap);
    for (const auto& entry : hashMergeNum) {
        numDBList[hashOrder_[entry.first]] = entry.second;
    }

    auto defaultIt = vecNBufferSettingByFunc_.find(FUNC_HASH_ORDER_DEFAULT_KEY);
    if (defaultIt != vecNBufferSettingByFunc_.end()) {
        int defaultVal = defaultIt->second;
        for (const auto& [hashVal, order] : hashOrder_) {
            (void)hashVal;
            numDBList[order] = defaultVal;
        }
        APASS_LOG_INFO_F(Elements::Config, "Applied DEFAULT config: %d for all hashOrders in function magic %d",
                         defaultVal, currentFuncMagic);
    }

    for (const auto& entry : vecNBufferSettingByFunc_) {
        if (entry.first == FUNC_HASH_ORDER_DEFAULT_KEY) {
            continue;
        }
        int funcMagic, localOrder;
        if (!ParseFuncHashOrder(entry.first, funcMagic, localOrder)) {
            APASS_LOG_WARN_F(Elements::Config, "Invalid func hashOrder format: %s, ignored.", entry.first.c_str());
            continue;
        }
        if (funcMagic == currentFuncMagic) {
            numDBList[localOrder] = entry.second;
        }
    }
}

void NBufferMerge::ApplyGlobalNumDB(std::map<uint64_t, size_t>& numDBList,
                                    std::map<uint64_t, std::vector<int>>& hashMap)
{
    auto it = vecNBufferSetting_.find(VEC_NBUFFER_SETTING_DEFAULT_MERGE_NUM_KEY);
    if (it != vecNBufferSetting_.end()) {
        int defaultVal = it->second;
        for (const auto& [hashVal, order] : hashOrder_) {
            (void)hashVal;
            numDBList[order] = defaultVal;
        }
    } else {
        auto hashMergeNum = GetIsoColorMergeNum(hashMap);
        for (const auto& entry : hashMergeNum) {
            numDBList[hashOrder_[entry.first]] = entry.second;
        }
    }

    for (const auto& entry : vecNBufferSetting_) {
        bool found = false;
        for (const auto& [hashVal, order] : hashOrder_) {
            (void)hashVal;
            if (order == entry.first) {
                found = true;
                break;
            }
        }
        if (found) {
            numDBList[entry.first] = entry.second;
        }
    }
}

std::map<uint64_t, size_t> NBufferMerge::SetNumDB(const Function& func, std::map<uint64_t, std::vector<int>>& hashMap)
{
    std::map<uint64_t, size_t> numDBList;
    if (!vecNBufferSettingByFunc_.empty()) {
        ApplyByFuncNumDB(func.GetFuncMagic(), numDBList, hashMap);
    } else {
        ApplyGlobalNumDB(numDBList, hashMap);
    }
    return numDBList;
}

Status NBufferMerge::CollectSemanticLabelOverrides(const OperationsViewer& opOriList,
                                                   const std::vector<uint64_t>& hashColor,
                                                   std::map<uint64_t, size_t>& labelOverrides) const
{
    auto labelToColors = BuildLabelToColorsMap(opOriList);
    // hashMergeNum is keyed by colorHashValue in auto modes, by hashOrder in manual modes.
    bool useHashValueAsKey = (vecNBuffermode_ == autoMerge || vecNBuffermode_ == autoMulityInOutMerge);
    for (const auto& [label, mergeNum] : vecNBufferSettingByLabel_) {
        auto it = labelToColors.find(label);
        if (it == labelToColors.end()) {
            APASS_LOG_WARN_F(
                Elements::Config,
                "Semantic label '%s' specified in vec_nbuffer_setting not found in any operation. "
                "This label is ignored. Please check that the label matches an operation's semantic_label.",
                label.c_str());
            continue;
        }

        for (int color : it->second) {
            uint64_t colorHash = hashColor[color];
            uint64_t target = colorHash;
            if (!useHashValueAsKey) {
                auto hashOrderIt = hashOrder_.find(colorHash);
                if (hashOrderIt == hashOrder_.end()) {
                    APASS_LOG_WARN_F(Elements::Config,
                                     "Could not find hash order for subgraph color %d with semantic label '%s'.", color,
                                     label.c_str());
                    continue;
                }
                target = static_cast<uint64_t>(hashOrderIt->second);
            }
            auto overIt = labelOverrides.find(target);
            if (overIt != labelOverrides.end()) {
                overIt->second = std::max(overIt->second, static_cast<size_t>(mergeNum));
            } else {
                labelOverrides[target] = static_cast<size_t>(mergeNum);
            }
        }
    }
    return SUCCESS;
}

Status NBufferMerge::ApplySemanticLabelSettings(const OperationsViewer& opOriList,
                                                std::map<uint64_t, size_t>& hashMergeNum,
                                                const std::map<uint64_t, std::vector<int>>& /* hashMap */,
                                                const std::vector<uint64_t>& hashColor)
{
    if (vecNBufferSettingByLabel_.empty()) {
        return SUCCESS;
    }

    std::map<uint64_t, size_t> labelOverrides;
    if (CollectSemanticLabelOverrides(opOriList, hashColor, labelOverrides) == FAILED) {
        return FAILED;
    }
    for (const auto& [target, val] : labelOverrides) {
        hashMergeNum[target] = val;
        APASS_LOG_INFO_F(Elements::Config, "Applied semantic label override: target=%lu, merge_num=%zu", target, val);
    }

    return SUCCESS;
}

void NBufferMerge::LogHashOverview(const Function& func, const std::map<uint64_t, std::vector<int>>& hashMap) const
{
    int funcMagic = func.GetFuncMagic();
    APASS_LOG_INFO_F(Elements::Function, "Computation graph [%s] overview.", func.GetMagicName().c_str());
    for (auto& entry : hashMap) {
        std::string fullHashOrder = "func" + std::to_string(funcMagic) + "_" +
                                    std::to_string(hashOrder_.at(entry.first));
        APASS_LOG_INFO_F(Elements::Function, "Vec merge hashOrder: %s, Subgraph count: %zu, , Subgraph IDs: %s",
                         fullHashOrder.c_str(), entry.second.size(), IntVecToStr(entry.second).c_str());
    }
    APASS_LOG_INFO_F(Elements::Function, "Computation graph [%s] overview end.", func.GetMagicName().c_str());
}

Status NBufferMerge::BuildHashMergeNum(const Function& func, std::map<uint64_t, std::vector<int>>& hashMap,
                                       std::map<uint64_t, size_t>& hashMergeNum)
{
    if (vecNBuffermode_ == autoMerge || vecNBuffermode_ == autoMulityInOutMerge) {
        APASS_LOG_INFO_F(Elements::Config, "Manually set mode to %d, automatically calculate mergeNum.",
                         vecNBuffermode_);
        hashMergeNum = GetIsoColorMergeNum(hashMap);
        return SUCCESS;
    }
    if (CheckVecNBufferSettingForManualMerge() == FAILED) {
        APASS_LOG_ERROR_F(
            Elements::Config,
            "Check VEC_NBUFFER_SETTING for manualMerge failed; Please check the VEC_NBUFFER_SETTING config.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Config, "Manually set mode to %d.", vecNBuffermode_);
    hashMergeNum = SetNumDB(func, hashMap);
    return SUCCESS;
}

Status NBufferMerge::RunMergeByMode(const OperationsViewer& opOriList, std::map<uint64_t, std::vector<int>>& hashMap,
                                    std::map<uint64_t, size_t>& hashMergeNum, std::vector<uint64_t>& hashColor)
{
    if (vecNBuffermode_ == autoMulityInOutMerge || vecNBuffermode_ == manualMulityInOutMerge) {
        if (MergeProcessForMulityInOut(opOriList, hashMap, hashMergeNum, hashColor) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "MergeProcessForMulityInOut failed; Please check the MergeProcessForMulityInOut method.");
            return FAILED;
        }
    } else {
        if (MergeProcess(opOriList, hashMap, hashMergeNum, hashColor) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Operation, "MergeProcess failed; Please check the MergeProcess method.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status NBufferMerge::NBufferMergeProcess(Function& func)
{
    if (Init(func) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Operation, "Init Failed; Please check the Init method.");
        return FAILED;
    }
    if (colorNum_ == 0) {
        return SUCCESS;
    }
    APASS_LOG_INFO_F(Elements::Operation, "User set nbuffer mode: %d", vecNBuffermode_);
    auto opOriList = func.Operations(true, SortOperationsMode::LIGHTWEIGHT);
    std::vector<uint64_t> hashColor(colorNum_, 0);
    std::map<uint64_t, std::vector<int>> hashMap;
    hashOrder_.clear();
    GetColorHash(func, opOriList, hashColor, hashMap);
    LogHashOverview(func, hashMap);

    std::map<uint64_t, size_t> hashMergeNum;
    if (BuildHashMergeNum(func, hashMap, hashMergeNum) == FAILED) {
        return FAILED;
    }
    if (ApplySemanticLabelSettings(opOriList, hashMergeNum, hashMap, hashColor) == FAILED) {
        APASS_LOG_ERROR_F(
            Elements::Config,
            "ApplySemanticLabelSettings failed; Please check the semantic labels in vec_nbuffer_setting.");
        return FAILED;
    }
    if (RunMergeByMode(opOriList, hashMap, hashMergeNum, hashColor) == FAILED) {
        return FAILED;
    }
    if (CheckAndFixColorOrder(opOriList, colorNum_, colorCycles_, colorNode_) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "CheckAndFixColorOrder failed; Please check the CheckAndFixColorOrder method.");
        return FAILED;
    }
    func.SetTotalSubGraphCount(colorNum_);
    APASS_LOG_DEBUG_F(Elements::Operation, "After Nbuffer merge.");
    RescheduleUtils::PrintColorNode(func);
    return SUCCESS;
}

Status NBufferMerge::CheckVecNBufferSettingForManualMerge()
{
    if (!vecNBufferSettingByFunc_.empty()) {
        for (const auto& pair : vecNBufferSettingByFunc_) {
            if (pair.second <= 0 || pair.second > static_cast<int64_t>(INT_MAX)) {
                APASS_LOG_ERROR_F(Elements::Config,
                                  "The value %ld of the key '%s' in VEC_NBUFFER_SETTING is incorrect; "
                                  "Please set values more than 0 and not exceeding the INT_MAX %d.",
                                  pair.second, pair.first.c_str(), INT_MAX);
                return FAILED;
            }
        }
        return SUCCESS;
    }

    if (vecNBufferSetting_.size() == 0) {
        APASS_LOG_ERROR_F(Elements::Config, "Mode is set to %d; Please set VEC_NBUFFER_SETTING to non-empty.",
                          vecNBuffermode_);
        return FAILED;
    }
    for (const auto& pair : vecNBufferSetting_) {
        if (pair.first == VEC_NBUFFER_SETTING_DEFAULT_MERGE_NUM_KEY) {
            continue;
        }
        bool found = false;
        for (const auto& [hashVal, order] : hashOrder_) {
            (void)hashVal;
            if (order == pair.first) {
                found = true;
                break;
            }
        }
        if (!found) {
            APASS_LOG_WARN_F(
                Elements::Config,
                "The VEC_NBUFFER_SETTING key %ld is invalid; This hashOrder does not exist in current graph.",
                pair.first);
        }
        if (pair.second <= 0 || pair.second > static_cast<int64_t>(INT_MAX)) {
            APASS_LOG_ERROR_F(Elements::Config,
                              "The value %ld of the key %ld in VEC_NBUFFER_SETTING is incorrect; Please set values of "
                              "VEC_NBUFFER_SETTING more than 0 and not exceeding the INT_MAX %d.",
                              pair.second, pair.first, INT_MAX);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status NBufferMerge::InitVecNBufferModeBySetting()
{
    if (!vecNBufferSettingByFunc_.empty()) {
        if (vecNBufferSettingByFunc_.size() == 1) {
            auto defaultIt = vecNBufferSettingByFunc_.find(FUNC_HASH_ORDER_DEFAULT_KEY);
            if (defaultIt != vecNBufferSettingByFunc_.end() && defaultIt->second == 1 &&
                vecNBufferSettingByLabel_.empty()) {
                vecNBuffermode_ = noMerge;
                APASS_LOG_INFO_F(Elements::Config, "Mode is noMerge by DEFAULT setting, skip NBufferMerge.");
                return SUCCESS;
            }
        }
        vecNBuffermode_ = manualMulityInOutMerge;
        return SUCCESS;
    }

    if (vecNBufferSetting_.size() == 0) {
        vecNBuffermode_ = autoMerge;
        return SUCCESS;
    }
    std::map<int64_t, int64_t> skipSetting = {{-1, 1}}; // 仅配置{{-1, 1}} 跳过合并
    if (vecNBufferSetting_ == skipSetting && vecNBufferSettingByLabel_.empty()) {
        vecNBuffermode_ = noMerge;
        return SUCCESS;
    }
    std::map<int64_t, int64_t> autoMulityInOutSetting = {{-2, 0}};
    if (vecNBufferSetting_ == autoMulityInOutSetting) {
        vecNBuffermode_ = autoMulityInOutMerge;
        return SUCCESS;
    }
    auto it = vecNBufferSetting_.find(MULITY_IN_OUT_MERGE_KEY);
    if (it != vecNBufferSetting_.end()) {
        if (it->second != 1) {
            APASS_LOG_ERROR_F(
                Elements::Config,
                "key=-2 is the multi-input/output merge control: use {-2: 0} for auto multi-in/out merge, or {-2: 1} "
                "for manual multi-in/out merge. Got invalid value=%ld for key=-2.",
                it->second);
            return FAILED;
        }
        vecNBufferSetting_.erase(it);
        vecNBuffermode_ = manualMulityInOutMerge;
        return SUCCESS;
    }
    vecNBuffermode_ = manualMerge;
    return SUCCESS;
}

Status NBufferMerge::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "===> Start NBufferMerge.");
    vecNBufferSetting_ = function.paramConfigs_.vecNBufferSetting;
    vecNBufferSettingByFunc_ = function.paramConfigs_.vecNBufferSettingByFunc;
    vecNBufferSettingByLabel_ = function.paramConfigs_.vecNBufferSettingByLabel;
    mgVecParallelLb_ = function.paramConfigs_.mgVecParallelLb;
    if (InitVecNBufferModeBySetting() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Config, "InitVecNBufferModeBySetting failed.");
        return FAILED;
    }
    if (vecNBuffermode_ == noMerge) {
        APASS_LOG_INFO_F(Elements::Config, "Mode is noMerge, skip NBufferMerge.");
        return SUCCESS;
    }
    if (NBufferMergeProcess(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "NBufferMergeProcess failed; Please check the NBufferMergeProcess method.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> Finish NBufferMerge.");
    return SUCCESS;
}
} // namespace npu::tile_fwk
