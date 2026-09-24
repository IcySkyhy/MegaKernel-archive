/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "backend/common/backend_registry.h"

#include <memory>
#include <string>
#include <utility>

#include "backend/common/backend.h"
#include "backend/common/soc.h"
#include "core/error.h"
#include "core/logging.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {

BackendRegistry& BackendRegistry::Instance()
{
    static BackendRegistry instance;
    return instance;
}

void BackendRegistry::Register(const std::string& type_name, CreateFunc func)
{
    CHECK(registry_.find(type_name) == registry_.end()) << "Backend type already registered: " << type_name;
    registry_[type_name] = std::move(func);
}

std::unique_ptr<Backend> BackendRegistry::Create(const std::string& type_name, const std::shared_ptr<const SoC>& soc)
{
    (void)type_name;
    (void)soc;
    throw ir::ValueError("Cannot create backend instances via registry - backends are singletons. "
                         "Use BackendCCE::Instance() instead.");
}

bool BackendRegistry::IsRegistered(const std::string& type_name) const
{
    return registry_.find(type_name) != registry_.end();
}

std::unique_ptr<Backend> CreateBackendFromRegistry(const std::string& type_name, const std::shared_ptr<const SoC>& soc)
{
    (void)type_name;
    (void)soc;
    throw ir::ValueError("Cannot create backend instances via registry - backends are singletons. "
                         "Use BackendCCE::Instance() instead.");
}

namespace {
bool RegisterBackendCCE()
{
    BackendRegistry::Instance().Register("CCE", [](const std::shared_ptr<const SoC>& /*unused*/) {
        throw ir::ValueError("Cannot create BackendCCE via registry - use BackendCCE::Instance()");
        return std::unique_ptr<Backend>(nullptr);
    });
    return true;
}

static bool backend_cce_registered = RegisterBackendCCE();
} // namespace

} // namespace backend
} // namespace pypto
