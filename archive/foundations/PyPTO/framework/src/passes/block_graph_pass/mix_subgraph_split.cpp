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
 * \file mix_subgraph_split.cpp
 * \brief
 */

#include "passes/block_graph_pass/mix_subgraph_split.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/utils/id_gen.h"

namespace npu {
namespace tile_fwk {
// 初始化静态成员
std::unordered_map<FunctionHash, GlobalSplitRecord> MixSubgraphSplit::globalSplitRecords_;
std::atomic<uint64_t> MixSubgraphSplit::globalNextMixId_{0};

Status MixSubgraphSplit::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function,
                     "===============================================================> Start MixSubgraphSplit.");
    // 获取rootFunc和programs
    auto rootFunc = function.rootFunc_;
    if (rootFunc == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Get root function failed.");
        return FAILED;
    }

    auto& programs = rootFunc->programs_;
    // 收集所有需要拆分的Mix子图及其内部组件信息
    std::vector<MixSubgraphInfo> mixSubgraphs;
    std::set<uint64_t> mixSubgraphIDsToDelete;
    std::vector<Operation*> callOpsToDelete;
    if (GatherSubGraphInfo(function, mixSubgraphs, mixSubgraphIDsToDelete, callOpsToDelete) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "GatherSubGraphInfo failed");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "Found %zu leaf function to process", programs.size());

    if (mixSubgraphs.empty()) {
        APASS_LOG_INFO_F(Elements::Function, "No mix subgraph found, jump MixSubgraphSplit.");
        return SUCCESS;
    }
    std::unordered_map<uint64_t, std::vector<uint64_t>> mixSubgraphNewIDs;
    std::unordered_map<uint64_t, uint64_t> programIDRemap;
    if (CalculateSplit(function, mixSubgraphs, mixSubgraphIDsToDelete, mixSubgraphNewIDs, programIDRemap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CalculateSplit failed");
        return FAILED;
    }
    if (ExecuteSplit(function, mixSubgraphs, callOpsToDelete, mixSubgraphNewIDs, programIDRemap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ExecuteSplit failed");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function,
                     "===============================================================> Finish MixSubgraphSplit.");
    return SUCCESS;
}

Status MixSubgraphSplit::GatherSubGraphInfo(Function& function, std::vector<MixSubgraphInfo>& mixSubgraphs,
                                            std::set<uint64_t>& mixSubgraphIDsToDelete,
                                            std::vector<Operation*>& callOpsToDelete)
{
    auto rootFunc = function.rootFunc_;
    auto callOps = rootFunc->GetCallopList();

    // 按哈希值分组callOp
    std::unordered_map<FunctionHash, std::vector<Operation*>> hashToCallOps;
    for (auto* callOp : callOps) {
        auto callAttr = dynamic_cast<CallOpAttribute*>(callOp->GetOpAttribute().get());
        if (callAttr != nullptr) {
            FunctionHash calleeHash = callAttr->GetCalleeHash();
            hashToCallOps[calleeHash].push_back(callOp);
        }
    }
    // 为每个哈希值查找对应的cacheFunction并判断是否为Mix子图
    for (auto& pair : hashToCallOps) {
        const FunctionHash& calleeHash = pair.first;
        std::vector<Operation*>& callOpList = pair.second;
        if (callOpList.empty()) {
            continue;
        }
        // 从全局缓存中获取function
        auto cacheValue = Program::GetInstance().TryHitCahce(calleeHash);
        Function* cacheFunc = cacheValue->GetFunction();
        // 检查是否是Mix子图
        if (!IsMixSubgraph(*cacheFunc)) {
            continue;
        }
        // 分析内部scope
        std::vector<InternalComponentInfo> components;
        auto status = componentsAnalyzer_.AnalyzeInternalComponents(*cacheFunc, components);
        if (status != SUCCESS || components.size() <= 1) {
            APASS_LOG_ERROR_F(Elements::Function, "AnalyzeInternalComponents failed for mix subgraph");
            return FAILED;
        }
        // 确定programID（仅在当前function中查找）
        uint64_t localProgramID = INVALID_PROGRAM_ID;
        bool isInCurrentFunc = false;
        // 检查这个function是否在当前function的programs中
        for (const auto& program : rootFunc->programs_) {
            if (program.second == cacheFunc) {
                localProgramID = program.first;
                isInCurrentFunc = true;
                break;
            }
        }
        // 添加到待处理列表
        mixSubgraphs.push_back(
            MixSubgraphInfo(localProgramID, cacheFunc, components, callOpList, calleeHash, isInCurrentFunc));
        // 如果需要删除当前function中的原leaffunction
        if (isInCurrentFunc) {
            mixSubgraphIDsToDelete.insert(localProgramID);
        }
        callOpsToDelete.insert(callOpsToDelete.end(), callOpList.begin(), callOpList.end());
        APASS_LOG_INFO_F(Elements::Function, "Found mix subgraph: local=%d, programID=%lu, callOps=%zu, components=%zu",
                         isInCurrentFunc, localProgramID, callOpList.size(), components.size());
    }
    return SUCCESS;
}

Status MixSubgraphSplit::CalculateSplit(Function& function, std::vector<MixSubgraphInfo>& mixSubgraphs,
                                        std::set<uint64_t>& mixSubgraphIDsToDelete,
                                        std::unordered_map<uint64_t, std::vector<uint64_t>>& mixSubgraphNewIDs,
                                        std::unordered_map<uint64_t, uint64_t>& programIDRemap)
{
    // 计算最终的programID分配
    auto rootFunc = function.rootFunc_;
    size_t originalCount = rootFunc->programs_.size();
    size_t deleteCount = mixSubgraphIDsToDelete.size();
    size_t newSubgraphCount = 0;
    for (const auto& mixInfo : mixSubgraphs) {
        if (mixInfo.isLocalFunction) {
            // 只有本地function的拆分才会在当前function中添加新program
            newSubgraphCount += mixInfo.components.size();
        }
    }
    size_t finalCount = originalCount - deleteCount + newSubgraphCount;
    APASS_LOG_INFO_F(Elements::Function, "Program count: original= %zu, delete=%zu, new=%zu, final=%zu", originalCount,
                     deleteCount, newSubgraphCount, finalCount);
    // 构建programID重映射表
    uint64_t nextProgramID = 0; // 从0开始重新分配连续ID

    // 首先映射保留的子图（非Mix子图）
    for (auto& program : rootFunc->programs_) {
        if (mixSubgraphIDsToDelete.find(program.first) == mixSubgraphIDsToDelete.end()) {
            programIDRemap[program.first] = nextProgramID++;
            APASS_LOG_DEBUG_F(Elements::Function, "Remap preserved program: %lu ->  %lu", program.first,
                              programIDRemap[program.first]);
        }
    }
    // 为新创建的子图分配连续的ID
    for (const auto& mixInfo : mixSubgraphs) {
        if (!mixInfo.isLocalFunction) {
            continue; // 跳过跨function
        }
        std::vector<uint64_t> newProgramIDs;
        for (size_t i = 0; i < mixInfo.components.size(); ++i) {
            newProgramIDs.push_back(nextProgramID++);
        }
        mixSubgraphNewIDs[mixInfo.programID] = newProgramIDs;
        APASS_LOG_INFO_F(Elements::Function, "Allocated %zu new programIDs for local mix subgraph %lu",
                         newProgramIDs.size(), mixInfo.programID);
    }
    return SUCCESS;
}

Status MixSubgraphSplit::ExecuteSplit(Function& function, std::vector<MixSubgraphInfo>& mixSubgraphs,
                                      std::vector<Operation*> callOpsToDelete,
                                      std::unordered_map<uint64_t, std::vector<uint64_t>>& mixSubgraphNewIDs,
                                      std::unordered_map<uint64_t, uint64_t>& programIDRemap)
{
    // 执行实际的拆分
    std::vector<MixSubgraphSplitResult> splitResults;
    auto rootFunc = function.rootFunc_;
    for (const auto& mixInfo : mixSubgraphs) {
        std::vector<uint64_t> newProgramIDs;
        if (mixInfo.isLocalFunction) {
            // 本地function：使用预分配的programID
            auto it = mixSubgraphNewIDs.find(mixInfo.programID);
            if (it != mixSubgraphNewIDs.end()) {
                newProgramIDs = it->second;
            } else {
                APASS_LOG_ERROR_F(Elements::Function, "No programIDs allocated for local mix subgraph %lu",
                                  mixInfo.programID);
                return FAILED;
            }
        } else {
            // 跨function：创建虚拟的programID（不添加到programs中）
            static uint64_t tempIDBase = 0xFFFFFFFF00000000ULL;
            for (size_t i = 0; i < mixInfo.components.size(); ++i) {
                newProgramIDs.push_back(tempIDBase++);
            }
        }
        // 统一调用ProcessLeafFunction
        if (ProcessLeafFunction(*rootFunc, mixInfo, newProgramIDs, splitResults) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "ProcessLeafFunction failed for function %s",
                              mixInfo.function->GetRawName().c_str());
            return FAILED;
        }
    }
    // 删除原始Mix子图的callOp
    DeleteOriginalMixCallOps(*rootFunc, callOpsToDelete);
    APASS_LOG_INFO_F(Elements::Function, "Found %zu mix subgraphs to split", mixSubgraphs.size());

    // 应用拆分结果并重新映射所有programID
    auto status = ApplySplitResultsWithRemap(function, splitResults, programIDRemap, mixSubgraphNewIDs);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ApplySplitResultsWithRemap failed.");
        return status;
    }
    return SUCCESS;
}

bool MixSubgraphSplit::IsMixSubgraph(Function& function) const
{
    auto operations = function.Operations(false);
    for (size_t idx = 0; idx < operations.size(); idx++) {
        auto& op = operations[idx];
        if (op.IsNOP())
            continue;
        // 只要有一个op有有效的internalSubgraphID，就认为是Mix子图
        int internalSubgraphID = op.GetInternalSubgraphID();
        if (internalSubgraphID > 0) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Function %s identified as mix subgraph: op %s has internalSubgraphID=%d",
                              function.GetRawName().c_str(), op.GetOpcodeStr().c_str(), internalSubgraphID);
            return true;
        }
    }
    APASS_LOG_DEBUG_F(Elements::Function, "Function %s is not a mix subgraph: no ops with internalSubgraphID",
                      function.GetRawName().c_str());
    return false;
}

MixResourceType MixSubgraphSplit::GetMixResourceType(Function& mixFunc) const
{
    bool hasAIV0 = false;
    bool hasAIV1 = false;
    auto operations = mixFunc.Operations(false);
    for (size_t idx = 0; idx < operations.size(); idx++) {
        auto& op = operations[idx];
        if (op.IsNOP())
            continue;
        auto aivCore = op.GetAIVCore();
        if (aivCore == AIVCore::AIV0) {
            hasAIV0 = true;
        } else if (aivCore == AIVCore::AIV1) {
            hasAIV1 = true;
        }
        if (hasAIV0 && hasAIV1) {
            return MixResourceType::ONE_CUBE_TWO_VECTOR;
        }
    }
    if (hasAIV0 || hasAIV1) {
        return MixResourceType::ONE_CUBE_ONE_VECTOR;
    }
    return MixResourceType::UNKNOWN;
}

Status MixSubgraphSplit::ApplySplitResultsWithRemap(
    Function& function, const std::vector<MixSubgraphSplitResult>& splitResults,
    const std::unordered_map<uint64_t, uint64_t>& programIDRemap,
    const std::unordered_map<uint64_t, std::vector<uint64_t>>& mixSubgraphNewIDs)
{
    auto rootFunc = function.rootFunc_;
    if (rootFunc == nullptr) {
        return FAILED;
    }
    size_t originalCount = rootFunc->programs_.size();
    // 构建新的programs映射
    std::map<uint64_t, Function*> newPrograms;
    // 添加保留的子图
    for (auto& program : rootFunc->programs_) {
        if (programIDRemap.find(program.first) != programIDRemap.end()) {
            uint64_t newID = programIDRemap.at(program.first);
            newPrograms[newID] = program.second;
            // 更新function的programID
            if (program.second != nullptr) {
                program.second->SetProgramId(newID);
                APASS_LOG_DEBUG_F(Elements::Function, "Updated preserved program: oldID=%lu -> newID=%lu",
                                  program.first, newID);
            }
        }
    }
    // 添加新创建的子图
    for (const auto& result : splitResults) {
        if (result.originalProgramID == INVALID_PROGRAM_ID) {
            APASS_LOG_DEBUG_F(Elements::Function, "Skip cross-function result: originalProgramID=INVALID");
            continue;
        }
        auto it = mixSubgraphNewIDs.find(result.originalProgramID);
        if (it == mixSubgraphNewIDs.end()) {
            APASS_LOG_WARN_F(Elements::Function, "No programIDs found for result with originalProgramID=%lu",
                             result.originalProgramID);
            continue;
        }
        const auto& newProgramIDs = it->second;
        for (size_t i = 0; i < result.newFunctions.size(); i++) {
            uint64_t newProgramID = newProgramIDs[i];
            Function* newFunc = result.newFunctions[i];
            if (newFunc != nullptr) {
                newPrograms[newProgramID] = newFunc;
                newFunc->SetProgramId(newProgramID);
                APASS_LOG_DEBUG_F(Elements::Function, "Added new subgraph: programID=%lu, function=%s", newProgramID,
                                  newFunc->GetRawName().c_str());
            }
        }
    }
    // 更新rootFunc的programs
    rootFunc->programs_ = std::move(newPrograms);
    APASS_LOG_INFO_F(Elements::Function, "Program mapping completed: original count=%lu, new count=%zu", originalCount,
                     rootFunc->programs_.size());
    return SUCCESS;
}

void MixSubgraphSplit::DisplayComponents(const std::vector<InternalComponentInfo>& components)
{
    for (size_t i = 0; i < components.size(); i++) {
        const auto& component = components[i];
        APASS_LOG_DEBUG_F(Elements::Function, "Component[%zu]: internalID=%d, suffix=%s, aivCore=%d, operations=%zu", i,
                          component.internalSubgraphID, component.suffix.c_str(), static_cast<int>(component.aivCore),
                          component.operations.size());
        // 打印component中的operations
        for (size_t j = 0; j < component.operations.size(); j++) {
            auto* op = component.operations[j];
            if (op != nullptr) {
                APASS_LOG_DEBUG_F(Elements::Operation, "  Operation[%zu]: magic=%d, opcode=%s, internalSubgraphID=%d",
                                  j, op->GetOpMagic(), op->GetOpcodeStr().c_str(), op->GetInternalSubgraphID());
            }
        }
    }
}

Status MixSubgraphSplit::GenNewFunctions(Function& rootFunc, Function* originalMixFunc,
                                         const std::vector<InternalComponentInfo>& components,
                                         const std::vector<uint64_t>& newProgramIDs,
                                         SubgraphToFunction& subgraphToFunction, std::vector<Function*>& newFunctions,
                                         uint64_t mixId, MixResourceType resourceType)
{
    for (size_t i = 0; i < components.size(); i++) {
        FunctionClone functionClone(rootFunc, originalMixFunc);
        auto newFunc = functionClone.CloneFunctionByComponent(components[i], newProgramIDs[i], i);
        if (newFunc == nullptr) {
            APASS_LOG_ERROR_F(Elements::Function, "CloneFunctionByComponent failed for function: %s",
                              originalMixFunc->GetRawName().c_str());
            return FAILED; // 或者适当的错误处理
        }
        subgraphToFunction.InsertParameter(i, *newFunc);
        // 在ComputeHash之前设置mixId和resourceType
        auto leafAttr = newFunc->GetLeafFuncAttribute();
        if (leafAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Function, "LeafFuncAttribute not set for new function");
            return FAILED;
        }
        leafAttr->mixId = mixId;
        leafAttr->mixResourceType = resourceType;
        APASS_LOG_DEBUG_F(Elements::Function, "Set mixId=%lu to leaf function %s (component %zu)", mixId,
                          newFunc->GetRawName().c_str(), i);
        newFunc->ComputeHash();
        FunctionHash funcHash = newFunc->GetFunctionHash();
        APASS_LOG_DEBUG_F(Elements::Function, "Function %s computed hash: %lu (mixId=%lu)",
                          newFunc->GetMagicName().c_str(), funcHash.GetHash(), mixId);
        Program::GetInstance().GetFunctionCache().Insert(funcHash, *newFunc);
        Program::GetInstance().InsertFuncToFunctionMap(newFunc->GetMagicName(), functionClone.cloneFunc);
        newFunctions.push_back(newFunc);
    }
    return SUCCESS;
}

// 每个mixLeafFunction的处理
Status MixSubgraphSplit::ProcessLeafFunction(Function& rootFunc, const MixSubgraphInfo& mixInfo,
                                             const std::vector<uint64_t>& newProgramIDs,
                                             std::vector<MixSubgraphSplitResult>& splitResults)
{
    auto* originalMixFunc = mixInfo.function;
    const auto& components = mixInfo.components;
    const auto& originalCallOps = mixInfo.originalCallOps;
    bool isLocalFunction = mixInfo.isLocalFunction;
    uint64_t programID = mixInfo.programID;
    APASS_LOG_INFO_F(Elements::Function, "Processing %s function %s (programID=%lu, components=%zu)",
                     isLocalFunction ? "local" : "non-local", originalMixFunc->GetRawName().c_str(), programID,
                     components.size());
    APASS_LOG_DEBUG_F(Elements::Function, "=== Component Details ===");
    DisplayComponents(components);
    // 1. 准备分析器输出和子leafFunction列表
    std::shared_ptr<AnalyzerOutput> analyzerOutput = nullptr;
    std::vector<Function*> newFunctions;
    SubgraphToFunction subgraphToFunction;
    if (isLocalFunction) {
        AnalyzerInput analyzerInput(components, originalMixFunc);
        analyzerOutput = std::make_shared<AnalyzerOutput>(SubgraphToFunction{}, std::vector<InternalDependencyInfo>(),
                                                          std::unordered_map<int, std::vector<SimpleTensorParam>>(),
                                                          std::unordered_map<int, std::vector<SimpleTensorParam>>());
        Status depStatus = dependencyAnalyzer_.ProcessDependencyAnalyzer(analyzerInput, *analyzerOutput);
        if (depStatus != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Dependency analyzer failed for function %s",
                              originalMixFunc->GetRawName().c_str());
            return FAILED;
        }
        uint64_t mixId = globalNextMixId_++;
        APASS_LOG_DEBUG_F(Elements::Function, "Assigning mixId=%lu for original mix function programID=%lu", mixId,
                          programID);
        MixResourceType resourceType = GetMixResourceType(*originalMixFunc);
        APASS_LOG_DEBUG_F(Elements::Function, "Mix resource type: %d for programID=%lu", static_cast<int>(resourceType),
                          programID);
        // 为每个scope创建leaf function
        if (GenNewFunctions(rootFunc, originalMixFunc, components, newProgramIDs, analyzerOutput->subgraphToFunction,
                            newFunctions, mixId, resourceType) != SUCCESS) {
            return FAILED;
        }

        // 应用最终的依赖到leaf functions（外部依赖）
        APASS_LOG_INFO_F(Elements::Function, "Applying final dependencies to leaf functions...");
        ApplyFinalDependencies(newFunctions, analyzerOutput->allIncasts, analyzerOutput->allOutcasts);
        // 记录到全局（仅本地function）
        RecordSplitResult(originalMixFunc, newFunctions, newProgramIDs, components, mixId, analyzerOutput);
        subgraphToFunction = analyzerOutput->subgraphToFunction;
    } else {
        // 非本地function：从全局记录获取
        // 2.1 从全局记录获取
        auto it = globalSplitRecords_.find(mixInfo.hashValue);
        if (it == globalSplitRecords_.end()) {
            APASS_LOG_ERROR_F(Elements::Function, "No global split record found for non-local function %s",
                              originalMixFunc->GetRawName().c_str());
            return FAILED;
        }
        const auto& splitRecord = it->second;
        newFunctions = splitRecord.splitFunctions;
        analyzerOutput = splitRecord.analyzerOutput;
        if (analyzerOutput) {
            subgraphToFunction = analyzerOutput->subgraphToFunction;
        }
    }
    // 为每个原始CallOp创建一组新的callOp, 每个原始callOp使用不同的wrapId（包含dummyTensor依赖）
    APASS_LOG_DEBUG_F(Elements::Operation, "Creating call operations for %zu components", components.size());
    if (callOpBuilder_.CreateCallOps(rootFunc, originalCallOps, originalMixFunc, components, newProgramIDs,
                                     subgraphToFunction, newFunctions) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to create call ops for function %s",
                          originalMixFunc->GetRawName().c_str());
        return FAILED;
    }
    // 4. 保存结果
    MixSubgraphSplitResult result;
    result.originalProgramID = isLocalFunction ? programID : INVALID_PROGRAM_ID;
    result.originalFunction = originalMixFunc;
    result.newProgramIDs = newProgramIDs;
    result.newFunctions = newFunctions;
    result.components = components;
    result.originalCallOps = originalCallOps;
    splitResults.push_back(result);

    APASS_LOG_INFO_F(Elements::Function, "Successfully processed %s function %s: %zu sub-functions",
                     isLocalFunction ? "local" : "non-local", originalMixFunc->GetRawName().c_str(),
                     newFunctions.size());
    return SUCCESS;
}

void MixSubgraphSplit::RecordSplitResult(Function* leafFunc, const std::vector<Function*>& newFunctions,
                                         const std::vector<uint64_t>& newProgramIDs,
                                         const std::vector<InternalComponentInfo>& components, uint64_t mixId,
                                         const std::shared_ptr<AnalyzerOutput>& analyzerOutput)
{
    // 检查输入参数的有效性
    if (leafFunc == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Cannot record split result: leafFunc is null");
        return;
    }
    // 计算函数的哈希值
    FunctionHash funcHash = leafFunc->ComputeHash();
    auto existingIt = globalSplitRecords_.find(funcHash);
    if (existingIt != globalSplitRecords_.end()) {
        // 已经存在记录，检查是否是同一个函数
        if (existingIt->second.originalLeafFunc == leafFunc) {
            APASS_LOG_WARN_F(Elements::Function,
                             "Split result already recorded for function %s (hash: %lu), overwriting",
                             leafFunc->GetRawName().c_str(), funcHash.GetHash());
        } else {
            // 哈希冲突：不同函数有相同哈希值
            APASS_LOG_ERROR_F(
                Elements::Function, "Hash collision for function %s (hash: %lu) with existing function %s",
                leafFunc->GetRawName().c_str(), funcHash.GetHash(),
                existingIt->second.originalLeafFunc ? existingIt->second.originalLeafFunc->GetRawName().c_str() :
                                                      "null");
        }
    }
    // 创建记录
    GlobalSplitRecord record;
    record.originalLeafFunc = leafFunc;
    record.splitFunctions = newFunctions;
    record.programIDs = newProgramIDs;
    record.components = components;
    record.mixId = mixId;
    record.analyzerOutput = analyzerOutput;
    // 记录到全局映射
    globalSplitRecords_[funcHash] = record;
    // 打印详细信息
    APASS_LOG_INFO_F(Elements::Function, "Recorded split result for leaf function %s", leafFunc->GetRawName().c_str());
    APASS_LOG_INFO_F(Elements::Function, "  - Hash: %lu", funcHash.GetHash());
    APASS_LOG_INFO_F(Elements::Function, "  - MixId: %lu", mixId);
    APASS_LOG_INFO_F(Elements::Function, "  - Sub-functions: %zu", newFunctions.size());
    APASS_LOG_INFO_F(Elements::Function, "  - Components: %zu", components.size());
    APASS_LOG_INFO_F(Elements::Function, "  - Has analyzer output: %s", analyzerOutput ? "yes" : "no");
    // 记录子函数的详细信息
    for (size_t i = 0; i < newFunctions.size(); i++) {
        Function* subFunc = newFunctions[i];
        if (subFunc) {
            uint64_t programID = (i < newProgramIDs.size()) ? newProgramIDs[i] : INVALID_PROGRAM_ID;
            APASS_LOG_DEBUG_F(Elements::Function, "  Sub-function[%zu]: %s (programID=%lu)", i,
                              subFunc->GetRawName().c_str(), programID);
        }
    }
    // 记录scope信息
    for (size_t i = 0; i < components.size(); i++) {
        const auto& component = components[i];
        APASS_LOG_DEBUG_F(Elements::Function, "  Component[%zu]: internalID=%d, aivCore=%d, ops=%zu", i,
                          component.internalSubgraphID, static_cast<int>(component.aivCore),
                          component.operations.size());
    }
}

void MixSubgraphSplit::ApplyFinalDependencies(
    const std::vector<Function*>& newFunctions,
    const std::unordered_map<int, std::vector<SimpleTensorParam>>& allIncasts,
    const std::unordered_map<int, std::vector<SimpleTensorParam>>& allOutcasts) const
{
    APASS_LOG_INFO_F(Elements::Function, "Applying final dependencies to %zu leaf functions", newFunctions.size());
    for (size_t i = 0; i < newFunctions.size(); i++) {
        Function* leafFunc = newFunctions[i];
        if (!leafFunc)
            continue;
        // 应用incast依赖
        auto incastIt = allIncasts.find(i);
        if (incastIt != allIncasts.end()) {
            ApplyIncastDependencies(leafFunc, i, incastIt->second);
        }
        // 应用outcast依赖
        auto outcastIt = allOutcasts.find(i);
        if (outcastIt != allOutcasts.end()) {
            ApplyOutcastDependencies(leafFunc, i, outcastIt->second);
        }
    }
}

// 应用incast依赖
void MixSubgraphSplit::ApplyIncastDependencies(Function* leafFunc, int componentId,
                                               const std::vector<SimpleTensorParam>& incastParams) const
{
    if (!leafFunc)
        return;
    // 获取当前已有的incast，用于去重
    const auto& existingIncasts = leafFunc->GetIncast();
    std::unordered_set<uint32_t> existingMagicSet;

    for (const auto& tensor : existingIncasts) {
        if (tensor) {
            existingMagicSet.insert(tensor->magic);
        }
    }
    for (const auto& param : incastParams) {
        if (!param.tensor) {
            APASS_LOG_WARN_F(Elements::Tensor, "Component %d: Null tensor in incast params, skipping", componentId);
            continue;
        }
        // 检查是否已经存在相同tensor（按magic）
        if (existingMagicSet.find(param.tensor->magic) != existingMagicSet.end()) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Component %d: Tensor %d already in incast list, skipping", componentId,
                              param.tensor->GetRawMagic());
            continue;
        }
        // 添加新的incast
        leafFunc->AppendIncast(param.tensor, param.opMagic, param.operandIdx);
        existingMagicSet.insert(param.tensor->magic);
        APASS_LOG_DEBUG_F(Elements::Tensor, "Component %d: Added incast - tensor %d (opMagic=%d, operandIdx=%d)",
                          componentId, param.tensor->GetRawMagic(), param.opMagic, param.operandIdx);
    }
}

// 应用outcast依赖
void MixSubgraphSplit::ApplyOutcastDependencies(Function* leafFunc, int componentId,
                                                const std::vector<SimpleTensorParam>& outcastParams) const
{
    if (!leafFunc)
        return;
    // 获取当前已有的outcast，用于去重
    const auto& existingOutcasts = leafFunc->GetOutcast();
    std::unordered_set<uint32_t> existingMagicSet;

    for (const auto& tensor : existingOutcasts) {
        if (tensor) {
            existingMagicSet.insert(tensor->magic);
        }
    }
    for (const auto& param : outcastParams) {
        if (!param.tensor) {
            APASS_LOG_WARN_F(Elements::Tensor, "Component %d: Null tensor in outcast params, skipping", componentId);
            continue;
        }

        // 检查是否已经存在相同tensor（按magic）
        if (existingMagicSet.find(param.tensor->magic) != existingMagicSet.end()) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Component %d: Tensor %d already in outcast list, skipping",
                              componentId, param.tensor->GetRawMagic());
            continue;
        }

        // 添加新的outcast
        leafFunc->AppendOutcast(param.tensor, param.opMagic, param.operandIdx);
        existingMagicSet.insert(param.tensor->magic);
        APASS_LOG_DEBUG_F(Elements::Tensor, "Component %d: Added outcast - tensor %d (opMagic=%d, operandIdx=%d)",
                          componentId, param.tensor->GetRawMagic(), param.opMagic, param.operandIdx);
    }
}

void MixSubgraphSplit::DeleteOriginalMixCallOps(Function& rootFunc, const std::vector<Operation*>& callOpsToDelete)
{
    if (callOpsToDelete.empty()) {
        return;
    }
    for (auto* callOp : callOpsToDelete) {
        if (callOp != nullptr && !callOp->IsDeleted()) {
            callOp->SetAsDeleted();
            APASS_LOG_DEBUG_F(Elements::Operation, "Deleted callOp with magic=%d", callOp->GetOpMagic());
        }
    }
    // 执行实际删除
    rootFunc.EraseOperations(false);
    APASS_LOG_INFO_F(Elements::Operation, "Deleted %zu original mix subgraph callOps", callOpsToDelete.size());
}

void MixSubgraphSplit::Reset()
{
    APASS_LOG_INFO_F(Elements::Function, "MixSubgraphSplit::Reset() called, clearing cache");
    // 清理所有缓存记录
    globalSplitRecords_.clear();
    globalNextMixId_ = 0;
}
} // namespace tile_fwk
} // namespace npu
