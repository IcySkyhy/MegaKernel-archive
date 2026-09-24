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
 * \file cann_adapter.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <map>
#include "adapter/manager/plugin_handler.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
template <typename EnumType>
class CannAdapter {
public:
    CannAdapter() : isInit_(false) { functions_.fill(nullptr); }
    ~CannAdapter()
    {
        libHandler_.CloseHandler();
        functions_.fill(nullptr);
        isInit_ = false;
    }
    bool Initialize(const std::string& libName, const std::map<EnumType, std::string>& funcNameMap)
    {
        if (isInit_) {
            return true;
        }
        if (!libHandler_.OpenHandler(libName)) {
            return false;
        }
        ADAPTER_LOGD("Library[%s] has been load.", libName.c_str());
        InitFunctions(funcNameMap);
        isInit_ = true;
        return true;
    }
    void* GetFunction(const EnumType func) const
    {
        if (func < static_cast<EnumType>(0) || func >= EnumType::Bottom) {
            return nullptr;
        }
        return functions_[static_cast<size_t>(func)];
    }

private:
    void InitFunctions(const std::map<EnumType, std::string>& funcNameMap)
    {
        functions_.fill(nullptr);
        for (const std::pair<const EnumType, std::string>& item : funcNameMap) {
            void* func = libHandler_.GetFunction(item.second);
            if (func == nullptr) {
                ADAPTER_LOGI("Fail to load function[%s]", item.second.c_str());
                continue;
            }
            ADAPTER_LOGD("Function[%s] has been load successfully.", item.second.c_str());
            functions_[static_cast<size_t>(item.first)] = func;
        }
    }
    bool isInit_;
    PluginHandler libHandler_;
    std::array<void*, static_cast<size_t>(EnumType::Bottom)> functions_;
};
} // namespace npu::tile_fwk
