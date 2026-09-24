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
 * \file plugin.h
 * \brief
 */

#include <functional>
#include <string>
#include <memory>

namespace npu::tile_fwk {

enum class PluginKind : int {
    CODEGEN_SRC,
};

class PluginBase {
public:
    PluginBase(PluginKind kind, const std::string& name) : kind_(kind), name_(name) {}
    PluginKind GetKind() { return kind_; }
    const std::string& GetName() const { return name_; }

private:
    PluginKind kind_;
    std::string name_;
};

class PluginCodegenSrc : public PluginBase {
public:
    static constexpr PluginKind kind = PluginKind::CODEGEN_SRC;
    typedef std::function<std::string(const std::string& filepath, const std::string& source)> EntryType;

    PluginCodegenSrc(const std::string& name, std::shared_ptr<EntryType> entryHandler)
        : PluginBase(PluginKind::CODEGEN_SRC, name), entryHandler_(entryHandler)
    {}

    std::string Call(const std::string& filepath, const std::string& source)
    {
        return (*entryHandler_)(filepath, source);
    }

private:
    std::shared_ptr<EntryType> entryHandler_;
};

class PluginManager {
public:
    static PluginManager& GetInstance();

    void ClearPlugin();

    template <typename TPlugin>
    std::vector<std::shared_ptr<TPlugin>> GetPlugin()
    {
        std::vector<std::shared_ptr<PluginBase>>& pluginBaseList = pluginListDict_[TPlugin::kind];

        std::vector<std::shared_ptr<TPlugin>> pluginList;
        for (auto& pluginBase : pluginBaseList) {
            auto plugin = std::static_pointer_cast<TPlugin>(pluginBase);
            pluginList.push_back(plugin);
        }
        return pluginList;
    }

    bool AddPluginCodegenSrc(const std::string& name, const PluginCodegenSrc::EntryType& rawEntry);
    std::string RunPluginCodegenSrc(const std::string& filepath, const std::string& source);

private:
    bool AddPlugin(const std::shared_ptr<PluginBase>& plugin);

    std::unordered_map<PluginKind, std::vector<std::shared_ptr<PluginBase>>> pluginListDict_;
    std::unordered_map<std::string, std::shared_ptr<PluginBase>> pluginDict_;
};

} // namespace npu::tile_fwk
