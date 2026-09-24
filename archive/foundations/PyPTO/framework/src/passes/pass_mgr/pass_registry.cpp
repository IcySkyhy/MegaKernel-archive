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
 * \file pass_registry.cpp
 * \brief
 */

#include "pass_registry.h"

#include "passes/pass_log/pass_log.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "PassRegistry"

namespace npu::tile_fwk {
// PassRegistry
PassRegistry& PassRegistry::GetInstance()
{
    static PassRegistry instance;
    return instance;
}

void PassRegistry::RegisterPass(const std::string& passName, CreateFn createFn)
{
    std::lock_guard lock(mtx_);
    passCreators_.emplace(passName, std::move(createFn));
    APASS_LOG_INFO_F(Elements::Function, "Register pass: %s", passName.c_str());
}

std::unique_ptr<Pass> PassRegistry::CreatePass(const std::string& passName) const
{
    std::lock_guard lock(mtx_);
    if (auto it = passCreators_.find(passName); it != passCreators_.end()) {
        return it->second();
    }
    return nullptr;
}

// PassRegistrar
PassRegistrar::PassRegistrar(const std::string& passName, PassRegistry::CreateFn createFn,
                             std::function<void()> typeCheck)
{
    ASSERT(FunctionErr::FUNCTION_GRAPH_STRUCTURE, !passName.empty())
        << "[PassRegistry][Manager][ERROR]: PassName can not be empty.";
    typeCheck();
    PassRegistry::GetInstance().RegisterPass(passName, std::move(createFn));
}
} // namespace npu::tile_fwk
