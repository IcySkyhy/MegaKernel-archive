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
 * \file pass_manager.cpp
 * \brief
 */

#include "pass_manager.h"

#include <algorithm>
#include <initializer_list>
#include <unistd.h>
#include "interface/configs/config_manager.h"
#include "interface/compiler_monitor/monitor_pass_scope.h"
#include "interface/utils/string_utils.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_interface/pass_type.h"
#include "pass_registry.h"
#include "tilefwk/error.h"
#include "tilefwk/platform.h"
#include "pass_dependency.h"
// tensor graph pass
#include "passes/tensor_graph_pass/infer_tensor_format.h"
#include "passes/tensor_graph_pass/remove_redundant_reshape.h"
#include "passes/tensor_graph_pass/auto_cast.h"
#include "passes/tensor_graph_pass/infer_memory_conflict.h"
#include "passes/tensor_graph_pass/remove_undriven_view.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "passes/tensor_graph_pass/loop_unroll.h"
//  tile graph pass
#include "passes/tile_graph_pass/graph_partition/common_operation_eliminate.h"
#include "passes/tile_graph_pass/graph_partition/graph_partition.h"
#include "passes/tile_graph_pass/graph_partition/l1_copy_reuse.h"
#include "passes/tile_graph_pass/graph_partition/n_buffer_merge.h"
#include "passes/tile_graph_pass/graph_optimization/graph_optimization.h"
#include "passes/tile_graph_pass/graph_constraint/graph_constraint.h"
#include "passes/tile_graph_pass/data_path/data_path.h"
#include "passes/tile_graph_pass/subgraph_to_function.h"
// execute graph pass
#include "passes/block_graph_pass/memory_reuse/memory_reuse.h"
#include "passes/block_graph_pass/insert_sync.h"
#include "passes/block_graph_pass/schedule_ooo/schedule_ooo.h"
#include "passes/block_graph_pass/codegen_preproc.h"
#include "passes/block_graph_pass/infer_param_index.h"
#include "passes/block_graph_pass/copy_out_resolve.h"
#include "passes/block_graph_pass/dyn_attr_to_static.h"
#include "passes/block_graph_pass/mix_subgraph_split.h"
#include "passes/block_graph_pass/last_use_mark.h"
#include "passes/block_graph_pass/loopaxes_proc.h"
#include "passes/block_graph_pass/tune_tileopseq_for_vf.h"
#include "passes/block_graph_pass/tune_sync_for_vf.h"
#include "passes/pass_log/pass_log.h"

#undef MODULE_NAME
#define MODULE_NAME "PassManager"

namespace npu::tile_fwk {

namespace {
using PassEntry = PassManager::PassEntry;

std::vector<PassEntry> BuildPassEntries(std::initializer_list<PassName> passNames)
{
    std::vector<PassEntry> entries;
    entries.reserve(passNames.size());
    for (auto passName : passNames) {
        entries.emplace_back(PassNameStr(passName), passName);
    }
    return entries;
}

std::vector<PassEntry> BuildPvc2OooPassEntries()
{
    return BuildPassEntries({
        PassName::INFER_TENSOR_FORMAT,
        PassName::REMOVE_REDUNDANT_RESHAPE,
        PassName::AUTO_CAST,
        PassName::INFER_MEMORY_CONFLICT,
        PassName::REMOVE_UNDRIVEN_VIEW,
        PassName::EXPAND_FUNCTION,
        PassName::MERGE_VIEW_ASSEMBLE,
        PassName::SPLIT_RESHAPE,
        PassName::SPLIT_RAW_TENSOR,
        PassName::SPLIT_LARGE_FANOUT_TENSOR,
        PassName::DUPLICATE_OP,
        PassName::ASSIGN_MEMORY_TYPE,
        PassName::INFER_DISCONTINUOUS_INPUT,
        PassName::REMOVE_REDUNDANT_OP,
        PassName::INSERT_OP_FOR_VIEWASSEMBLE,
        PassName::PROCESS_ATOMIC,
        PassName::GRAPH_PARTITION,
        PassName::N_BUFFER_MERGE,
        PassName::L1_COPY_IN_REUSE_MERGE,
        PassName::REDUCE_COPY_MERGE,
        PassName::INTRA_SUBGRAPH_ADAPTER,
        PassName::GENERATE_MOVE_OP,
        PassName::COMMON_OPERATION_ELIMINATE,
        PassName::AXIS_COMBINE,
        PassName::PAD_LOCAL_BUFFER,
        PassName::REMOVE_UNALIGNED_RESHAPE,
        PassName::REPLACE_TENSOR,
        PassName::PRE_GRAPH_PROCESS,
        PassName::INFER_DYN_SHAPE,
        PassName::SUBGRAPH_TO_FUNCTION,
        PassName::INFER_PARAM_INDEX,
        PassName::SRC_DST_BUFFER_MERGE,
        PassName::ADD_ALLOC,
        PassName::OOO_SCHEDULE,
        PassName::TUNE_TILEOP_SEQ_FOR_VF,
        PassName::REMOVE_ALLOC,
        PassName::COPY_OUT_RESOLVE,
        PassName::INSERT_SYNC,
        PassName::TUNE_SYNC_FOR_VF,
        PassName::MIX_SUBGRAPH_SPLIT,
        PassName::LAST_USE_MARK,
        PassName::GLOBAL_MEMORY_REUSE,
        PassName::CODEGEN_PREPROC,
    });
}
} // namespace

PassManager& PassManager::Instance()
{
    static PassManager instance;
    return instance;
}

void RegPass()
{
    REG_PASS(InferTensorFormat);
    REG_PASS(GlobalMemoryReuse);
    REG_PASS(SubgraphToFunction);
    REG_PASS(GraphPartition);
    REG_PASS(ReduceCopyMerge);
    REG_PASS(InsertSync);
    REG_PASS(OoOSchedule);
    REG_PASS(RemoveUndrivenView);
    REG_PASS(ExpandFunction);
    REG_PASS(CommonOperationEliminate);
    REG_PASS(GenerateMoveOp);
    REG_PASS(AssignMemoryType);
    REG_PASS(RemoveRedundantReshape);
    REG_PASS(AutoCast);
    REG_PASS(InferMemoryConflict);
    REG_PASS(NBufferMerge);
    REG_PASS(L1CopyInReuseMerge);
    REG_PASS(MergeViewAssemble);
    REG_PASS(IntraSubgraphAdapter);
    REG_PASS(PadLocalBuffer);
    REG_PASS(ReplaceTensor);
    REG_PASS(PreGraphProcess);
    REG_PASS(RemoveRedundantOp);
    REG_PASS(SplitRawTensor);
    REG_PASS(SplitReshape);
    REG_PASS(RemoveUnalignedReshape);
    REG_PASS(CodegenPreproc);
    REG_PASS(SplitLargeFanoutTensor);
    REG_PASS(ProcessAtomic);
    REG_PASS(InferDynShape);
    REG_PASS(InferParamIndex);
    REG_PASS(AddAlloc);
    REG_PASS(RemoveAlloc);
    REG_PASS(CopyOutResolve);
    REG_PASS(SrcDstBufferMerge);
    REG_PASS(LoopUnroll);
    REG_PASS(DynAttrToStatic);
    REG_PASS(InferDiscontinuousInput);
    REG_PASS(MixSubgraphSplit);
    REG_PASS(LastUseMark);
    REG_PASS(DuplicateOp);
    REG_PASS(AxisCombine);
    REG_PASS(InsertOpForViewAssemble);
    REG_PASS(LoopaxesProc);
    REG_PASS(TuneTileOpSeqForVF);
    REG_PASS(TuneSyncForVF);
}

void PassManager::RegDefaultStrategy()
{
    RegisterStrategy("PVC2_OOO", BuildPvc2OooPassEntries());
    RegisterStrategy("FunctionUnroll", BuildPassEntries({PassName::LOOP_UNROLL}));
    RegisterStrategy("ExecuteGraph", BuildPassEntries({PassName::DYN_ATTR_TO_STATIC, PassName::LOOPAXES_PROC}));
}

PassManager::PassManager()
{
    RegPass();
    // Register strategies
    RegDefaultStrategy();
}

void PassManager::RegisterStrategy(const std::string& strategy, const std::vector<PassEntry>& passEntries)
{
    // check pass dependency
    std::vector<PassName> passes;
    for (const auto& passEntry : passEntries) {
        passes.emplace_back(passEntry.passName);
    }
    PassDependency::Instance().CheckStrategyDependency(strategy, passes);

    // check identifiers duplication
    std::vector<PassEntry> newPassEntries;
    std::set<std::string> identifiers;
    for (auto& pass : passEntries) {
        if (!(identifiers.insert(pass.identifier).second)) {
            APASS_LOG_WARN_F(Elements::Function, "Duplicated identifier: %s.", pass.identifier.c_str());
            continue;
        }
        newPassEntries.push_back(pass);
    }
    auto strategyPasses = strategies_.find(strategy);
    if (strategyPasses == strategies_.end()) {
        strategies_.emplace(strategy, newPassEntries);
        return;
    }
    strategyPasses->second = newPassEntries;
    APASS_LOG_WARN_F(Elements::Function, "Strategy %s has been changed.", strategy.c_str());
}

std::vector<PassManager::PassEntry> PassManager::GetStrategyPasses(const std::string& strategy) const
{
    auto it = strategies_.find(strategy);
    if (it == strategies_.end()) {
        APASS_LOG_WARN_F(Elements::Function, "Strategy %s does not exist.", strategy.c_str());
        auto emptyPass = std::vector<PassManager::PassEntry>();
        return emptyPass;
    }
    NPUArch currArch = Platform::Instance().GetSoc().GetNPUArch();
    auto selectedPass = std::vector<PassManager::PassEntry>();
    for (auto& currPassEntry : it->second) {
        const auto& passName = PassNameStr(currPassEntry.passName);
        auto pass = PassRegistry::GetInstance().CreatePass(passName);
        if (pass == nullptr) {
            APASS_LOG_WARN_F(Elements::Function, "Pass %s does not exist.", passName);
            continue;
        }
        std::vector<NPUArch>& arches = pass->GetSupportedArches();
        if ((!arches.empty()) && (std::find(arches.begin(), arches.end(), currArch) == arches.end())) {
            continue;
        }
        selectedPass.push_back(currPassEntry);
    }
    return selectedPass;
}

std::string PassManager::GetStrategyLogFolderName(const std::string& strategy) const
{
    if (strategy.empty()) {
        return "";
    }
    std::lock_guard<std::mutex> lock(strategyLogMutex_);
    auto it = strategyLogIndices_.find(strategy);
    if (it == strategyLogIndices_.end()) {
        it = strategyLogIndices_.emplace(strategy, strategyLogIndices_.size()).first;
    }
    return BuildStrategyLogFolderName(strategy, it->second);
}

std::string PassManager::GetResumePath(const std::string& strategy)
{
    auto strategyPasses = GetStrategyPasses(strategy);
    for (size_t i = 0; i < strategyPasses.size(); i++) {
        const auto& identifier = strategyPasses[i].identifier;
        auto passDfxCfg = ConfigManager::Instance().GetPassConfigs(strategy, identifier);
        if (passDfxCfg.resumePath != "") {
            if (access(passDfxCfg.resumePath.c_str(), F_OK) == 0) {
                startIdx = i;
            }
            return passDfxCfg.resumePath;
        }
    }
    startIdx = static_cast<size_t>(0);
    return "";
}

static bool ShouldTerminateAtStage(const std::string& identifier)
{
    static const std::unordered_map<std::string, int64_t> kPassToStageMap = {
        {"ExpandFunction", CS_TENSOR_GRAPH},
        {"SubgraphToFunction", CS_TILE_GRAPH},
    };
    auto it = kPassToStageMap.find(identifier);
    if (it != kPassToStageMap.end() && it->second == config::GetHostOption<int64_t>(COMPILE_STAGE)) {
        APASS_LOG_INFO_F(Elements::Function, "Compile stage terminates after %s.", identifier.c_str());
        return true;
    }
    return false;
}

static void WarnUnmatchedDumpPassGraph(const std::string& strategy, const std::vector<std::string>& dumpPassGraph,
                                       const std::vector<std::string>& identifiers)
{
    if (dumpPassGraph.empty()) {
        return;
    }
    std::vector<std::string> unmatchedPasses;
    for (const auto& passIdentifier : dumpPassGraph) {
        const bool matched = std::find(identifiers.begin(), identifiers.end(), passIdentifier) != identifiers.end();
        const bool alreadyRecorded = std::find(unmatchedPasses.begin(), unmatchedPasses.end(), passIdentifier) !=
                                     unmatchedPasses.end();
        if (!matched && !alreadyRecorded) {
            unmatchedPasses.push_back(passIdentifier);
        }
    }
    if (unmatchedPasses.empty()) {
        return;
    }
    APASS_LOG_WARN_F(Elements::Function,
                     "debug.dump_pass_graph contains pass identifier(s) %s that do not exist in strategy %s.",
                     StringUtils::ToString(unmatchedPasses).c_str(), strategy.c_str());
}

Status PassManager::RunPass(Program& program, Function& function, const std::string& strategy) const
{
    auto strategyPasses = GetStrategyPasses(strategy);
    const std::string strategyLogFolder = GetStrategyLogFolderName(strategy);
    std::unique_ptr<Pass> pass = nullptr;
    std::vector<std::string> identifiers;
    std::transform(strategyPasses.begin(), strategyPasses.end(), std::back_inserter(identifiers),
                   [](const PassEntry& elem) { return elem.identifier; });
    ConfigManager::Instance().PassConfigsDebugInfo(strategy, identifiers);
    const auto dumpPassGraph = config::GetDebugOption<std::vector<std::string>>(CFG_DUMP_PASS_GRAPH);
    if (strategy == config::GetPassStrategy()) {
        WarnUnmatchedDumpPassGraph(strategy, dumpPassGraph, identifiers);
    }
    for (size_t i = startIdx; i < strategyPasses.size(); i++) {
        const auto& identifier = strategyPasses[i].identifier;
        if (ShouldTerminateAtStage(identifier)) {
            return SUCCESS;
        }
        const auto& passName = strategyPasses[i].passName;
        pass = PassRegistry::GetInstance().CreatePass(PassNameStr(passName));
        if (pass == nullptr) {
            APASS_LOG_ERROR_F(Elements::Function, "Pass [%s] does not exist.", PassNameStr(passName));
            return FAILED;
        }
        PassLogUtil logUtil(*pass, function, strategyLogFolder, i);
        auto passDfxCfg = ConfigManager::Instance().GetPassConfigs(strategy, identifier);
        if (!dumpPassGraph.empty()) {
            const bool matched = std::find(dumpPassGraph.begin(), dumpPassGraph.end(), identifier) !=
                                 dumpPassGraph.end();
            passDfxCfg.dumpGraph = matched;
            passDfxCfg.printGraph = matched;
        }
        if (config::GetDebugOption<int64_t>(CFG_COMPILE_DBEUG_MODE) == CFG_DEBUG_ALL) {
            passDfxCfg.printGraph = true;
            passDfxCfg.dumpGraph = true;
            passDfxCfg.printProgram = true;
        }
        pass->SetPassConfigs(passDfxCfg);
        APASS_LOG_INFO_F(Elements::Function, "Apply pass <%s> on function: %s.", identifier.c_str(),
                         function.GetMagicName().c_str());
        MonitorPassCompileScope passCompileScope(strategy, identifier, i, function.GetMagicName(),
                                                 MonitorManager::Instance().GetCurrentFunctionIndex(),
                                                 MonitorManager::Instance().GetCurrentFuncOpSize());
        Status status = pass->Run(function, strategy, identifier, i, strategyLogFolder);
        auto passEnd = std::chrono::high_resolution_clock::now();
        passCompileScope.FinishAt(status == SUCCESS, passEnd);
        if (status != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Run pass <%s> failed.", identifier.c_str());
            return FAILED;
        }
        if (identifier == "ExpandFunction") {
            // ExpandFunction 之后的图规模，作为后续 pass 的阈值基线。
            MonitorManager::Instance().SetCurrentFuncOpSize(static_cast<int>(function.GetOperationSize()), true);
        }
        if (passDfxCfg.dumpPassTimeCost) {
            LogPassRuntime(identifier, program, function, passCompileScope.GetStartTime());
        }
        if (config::GetVerifyOption<bool>(KEY_ENABLE_PASS_VERIFY)) {
            Program::GetInstance().VerifyPass(&function, i, identifier);
        }
    }
    if (config::GetDebugOption<int64_t>(CFG_COMPILE_DBEUG_MODE) == CFG_DEBUG_ALL && pass != nullptr) {
        ExtractPassLogByFunction(function, strategyLogFolder);
    }
    if (strategy == config::GetPassStrategy() && !strategyLogFolder.empty()) {
        const std::string graphPath = config::LogTopFolder() + "/" + kComputationGraphFolder + "/" + strategyLogFolder;
        const std::string realGraphPath = RealPath(graphPath);
        if (!realGraphPath.empty()) {
            config::SetRunDataOption(KEY_COMPUTE_GRAPH_PATH, realGraphPath);
        }
    }
    return SUCCESS;
}

void PassManager::ResetAllPasses()
{
    APASS_LOG_INFO_F(Elements::Function, "ResetAllPasses: resetting all passes from all strategies");
    // 用于记录已经 reset 过的 Pass，避免重复 reset
    std::unordered_set<PassName> resetPasses;
    // 遍历所有策略
    for (auto& strategyPair : strategies_) {
        const auto& strategyName = strategyPair.first;
        const auto& passEntries = strategyPair.second;

        APASS_LOG_DEBUG_F(Elements::Function, "ResetAllPasses: processing strategy: %s", strategyName.c_str());

        // 遍历策略中的所有 Pass
        for (const auto& passEntry : passEntries) {
            PassName passName = passEntry.passName;

            // 避免重复 reset 同一个 Pass
            if (resetPasses.find(passName) != resetPasses.end()) {
                continue;
            }

            auto pass = PassRegistry::GetInstance().CreatePass(PassNameStr(passName));
            if (pass == nullptr) {
                APASS_LOG_WARN_F(Elements::Function, "ResetAllPasses: Pass [%s] does not exist.",
                                 PassNameStr(passName));
                continue;
            }

            APASS_LOG_DEBUG_F(Elements::Function, "ResetAllPasses: resetting pass: %s", PassNameStr(passName));

            // 调用 Pass 的 Reset 方法
            pass->Reset();

            // 记录已 reset 的 Pass
            resetPasses.insert(passName);
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "ResetAllPasses: reset %zu unique passes", resetPasses.size());
}
} // namespace npu::tile_fwk
