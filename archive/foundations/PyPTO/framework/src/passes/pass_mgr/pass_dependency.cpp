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
 * \file pass_dependency.cpp
 * \brief
 */

#include "pass_dependency.h"
#include "pass_manager.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "PassDependency"

namespace npu::tile_fwk {
PassDependency& PassDependency::Instance()
{
    static PassDependency instance;
    return instance;
}

PassDependency::PassDependency()
{
    RegisterPreDependencies();
    RegisterSequenceDependencies();
    APASS_LOG_DEBUG_F(Elements::Manager, "Strategy dependency checker initialized.");
}

void PassDependency::RegisterPreDependencies()
{
    auto registerDependency = [this](const PassName& name, const std::vector<PassName>& dependencies) {
        preDependencies_[name] = std::move(dependencies);
    };

    registerDependency(PassName::EXPAND_FUNCTION, {PassName::AUTO_CAST, PassName::INFER_MEMORY_CONFLICT});
    registerDependency(PassName::GRAPH_PARTITION, {PassName::DUPLICATE_OP, PassName::SPLIT_LARGE_FANOUT_TENSOR,
                                                   PassName::SPLIT_RESHAPE, PassName::PROCESS_ATOMIC});
    registerDependency(PassName::SUBGRAPH_TO_FUNCTION, {PassName::GRAPH_PARTITION, PassName::REPLACE_TENSOR,
                                                        PassName::PRE_GRAPH_PROCESS, PassName::INFER_DYN_SHAPE});
    registerDependency(PassName::INSERT_SYNC, {PassName::OOO_SCHEDULE, PassName::COPY_OUT_RESOLVE});
    registerDependency(PassName::REDUCE_COPY_MERGE, {PassName::GRAPH_PARTITION});
    registerDependency(PassName::N_BUFFER_MERGE, {PassName::GRAPH_PARTITION});
    registerDependency(PassName::L1_COPY_IN_REUSE_MERGE, {PassName::GRAPH_PARTITION});
    registerDependency(PassName::INTRA_SUBGRAPH_ADAPTER, {PassName::GRAPH_PARTITION});
    registerDependency(PassName::GENERATE_MOVE_OP, {PassName::GRAPH_PARTITION});
}

void PassDependency::RegisterSequenceDependencies()
{
    auto registerDependency = [this](const PassName& name, const std::vector<PassName>& dependencies) {
        sequenceDependencies_[name] = std::move(dependencies);
    };

    registerDependency(PassName::GRAPH_PARTITION, {PassName::GRAPH_PARTITION, PassName::REDUCE_COPY_MERGE,
                                                   PassName::N_BUFFER_MERGE, PassName::L1_COPY_IN_REUSE_MERGE,
                                                   PassName::INTRA_SUBGRAPH_ADAPTER, PassName::GENERATE_MOVE_OP});
}

Status PassDependency::CheckStrategyDependency(const std::string& strategyName, const std::vector<PassName>& passes)
{
    APASS_LOG_DEBUG_F(Elements::Manager, "Start dependency check for strategy %s.", strategyName.c_str());
    bool needWarn = false;
    std::unordered_set<PassName> processedPasses;
    std::unordered_set<PassName> duplicates;
    std::optional<PassName> prePass;

    for (size_t i = 0; i < passes.size(); i++) {
        const PassName pName = passes[i];
        if (prePass.has_value() && prePass.value() == pName) {
            needWarn = true;
            duplicates.emplace(pName);
            prePass = pName;
            continue;
        }
        prePass = pName;

        if (CheckSequenceDependency(i, strategyName, passes) != SUCCESS) {
            needWarn = true;
        }

        processedPasses.emplace(pName);
        auto it = preDependencies_.find(pName);
        if (it == preDependencies_.end()) {
            continue;
        }
        std::vector<PassName> missingDeps;
        for (const auto& dependentPass : it->second) {
            if (processedPasses.find(dependentPass) == processedPasses.end()) {
                missingDeps.push_back(dependentPass);
            }
        }
        if (missingDeps.empty()) {
            continue;
        }
        needWarn = true;
        APASS_LOG_WARN_F(Elements::Manager,
                         "In strategy %s, %s is missing dependencies, %s are required; Please insert %s before %s.",
                         strategyName.c_str(), PassNameStr(pName),
                         CommonUtils::ContainerToStr<std::vector<PassName>>(it->second).c_str(),
                         CommonUtils::ContainerToStr<std::vector<PassName>>(missingDeps).c_str(), PassNameStr(pName));
    }
    if (duplicates.size() != 0) {
        APASS_LOG_WARN_F(
            Elements::Manager,
            "In strategy %s, %s are each arranged at least twice in a row; Please make sure all are needed.",
            strategyName.c_str(), CommonUtils::ContainerToStr<std::unordered_set<PassName>>(duplicates).c_str());
    }
    APASS_LOG_DEBUG_F(Elements::Manager, "Finish dependency check for strategy %s.", strategyName.c_str());
    return needWarn ? WARNING : SUCCESS;
}

Status PassDependency::CheckSequenceDependency(size_t index, const std::string& strategyName,
                                               const std::vector<PassName>& passes)
{
    auto it = sequenceDependencies_.find(passes[index]);
    if (it == sequenceDependencies_.end()) {
        return SUCCESS;
    }
    std::vector<PassName>& expectedPasses = it->second;
    std::vector<PassName> actualPasses;

    for (size_t i = 0; i < expectedPasses.size(); i++) {
        if (index + i >= passes.size() || passes[index + i] != expectedPasses[i]) {
            actualPasses.assign(passes.begin() + index, passes.end());
            APASS_LOG_WARN_F(
                Elements::Manager,
                "In strategy %s, %s has mismatched sequence dependencies. Expected ordered prefix starting at this "
                "pass: %s, but actual sequence from this pass onward "
                "is: %s.",
                strategyName.c_str(), PassNameStr(passes[index]),
                CommonUtils::ContainerToStr<std::vector<PassName>>(expectedPasses).c_str(),
                CommonUtils::ContainerToStr<std::vector<PassName>>(actualPasses).c_str());
            return WARNING;
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
