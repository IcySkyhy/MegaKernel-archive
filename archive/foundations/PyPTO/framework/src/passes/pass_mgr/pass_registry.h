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
 * \file pass_registry.h
 * \brief
 */

#ifndef PASSES_PASS_REG_H_
#define PASSES_PASS_REG_H_

#include <map>
#include <memory>
#include <string>
#include <functional>
#include <type_traits>
#include <mutex>

#include "passes/pass_interface/pass.h"
#include "tilefwk/error.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
class PassRegistry {
public:
    using CreateFn = std::function<std::unique_ptr<Pass>()>;
    ~PassRegistry() = default;
    static PassRegistry& GetInstance();
    void RegisterPass(const std::string& passName, CreateFn createFn);
    std::unique_ptr<Pass> CreatePass(const std::string& passName) const;

private:
    PassRegistry() = default;

private:
    mutable std::mutex mtx_;
    std::map<std::string, CreateFn> passCreators_;
};

class PassRegistrar {
public:
    PassRegistrar(const std::string& passName, PassRegistry::CreateFn createFn, std::function<void()> typeCheck);

    ~PassRegistrar() = default;
};

#define REG_PASS(DerivedPass)                                                                                     \
    [[maybe_unused]] static auto tile_fwk_passRegister##DerivedPass = ::npu::tile_fwk::PassRegistrar(             \
        #DerivedPass, []() -> std::unique_ptr<::npu::tile_fwk::Pass> { return std::make_unique<DerivedPass>(); }, \
        []() {                                                                                                    \
            static_assert(std::is_base_of_v<::npu::tile_fwk::Pass, DerivedPass>);                                 \
            std::string passName = DerivedPass().GetName();                                                       \
            ASSERT(FunctionErr::FUNCTION_GRAPH_STRUCTURE, passName == #DerivedPass)                               \
                << "[PassRegistry][Manager][ERROR]: Pass class " << #DerivedPass                                  \
                << " has incompatible name: " << passName;                                                        \
        })
} // namespace npu::tile_fwk
#endif // PASSES_PASS_REG_H_
