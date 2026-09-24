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
 * \file plugin.cpp
 * \brief
 */

#include "plugin.h"

namespace npu::tile_fwk {

PluginManager& PluginManager::GetInstance()
{
    static PluginManager pluginManager;
    return pluginManager;
}

bool PluginManager::AddPlugin(const std::shared_ptr<PluginBase>& plugin)
{
    if (pluginDict_.count(plugin->GetName())) {
        return false;
    }
    pluginDict_[plugin->GetName()] = plugin;
    pluginListDict_[plugin->GetKind()].push_back(plugin);
    return true;
}

void PluginManager::ClearPlugin()
{
    pluginListDict_.clear();
    pluginDict_.clear();
}

bool PluginManager::AddPluginCodegenSrc(const std::string& name, const PluginCodegenSrc::EntryType& rawEntry)
{
    auto entry = std::make_shared<PluginCodegenSrc::EntryType>(rawEntry);
    std::shared_ptr<PluginCodegenSrc> plugin = std::make_shared<PluginCodegenSrc>(name, entry);
    std::shared_ptr<PluginBase> pluginBase = std::static_pointer_cast<PluginBase>(plugin);
    return AddPlugin(pluginBase);
}

std::string PluginManager::RunPluginCodegenSrc(const std::string& filepath, const std::string& source)
{
    std::string sourceResult = source;
    std::vector<std::shared_ptr<PluginCodegenSrc>> pluginList = GetPlugin<PluginCodegenSrc>();
    if (pluginList.size() != 0) {
        for (size_t k = 0; k < pluginList.size(); k++) {
            auto plugin = pluginList[k];
            sourceResult = plugin->Call(filepath, sourceResult);
        }
    }
    return sourceResult;
}

} // namespace npu::tile_fwk
