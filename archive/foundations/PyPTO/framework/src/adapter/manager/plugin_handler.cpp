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
 * \file plugin_handler.cpp
 * \brief
 */

#include "adapter/manager/plugin_handler.h"
#include <dlfcn.h>
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
PluginHandler::PluginHandler() : handler_(nullptr) {}

PluginHandler::~PluginHandler() { CloseHandler(); }

bool PluginHandler::OpenHandler(const std::string& libName)
{
    handler_ = dlopen(libName.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handler_ == nullptr) {
        ADAPTER_LOGW("Failed to load library[%s], error: %s", libName.c_str(), dlerror());
    }
    return handler_ != nullptr;
}

void PluginHandler::CloseHandler()
{
    if (handler_ == nullptr) {
        return;
    }
    (void)dlclose(handler_);
    handler_ = nullptr;
}

void* PluginHandler::GetFunction(const std::string& funcName) const
{
    void* func = dlsym(handler_, funcName.c_str());
    if (func == nullptr) {
        ADAPTER_LOGI("Failed to load symbol[%s] from library, error: %s", funcName.c_str(), dlerror());
    }
    return func;
}
} // namespace npu::tile_fwk
