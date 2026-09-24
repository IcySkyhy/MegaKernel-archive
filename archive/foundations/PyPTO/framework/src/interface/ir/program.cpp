/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/program.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/span.h"

namespace pypto {
namespace ir {

// Vector-based constructor: creates string-keyed map from function names
Program::Program(const std::vector<FunctionPtr>& functions, std::string name, Span span, IRDebugInfoPtr debugInfo)
    : IRNode(std::move(span)), name_(std::move(name)), debugInfo_(std::move(debugInfo))
{
    std::set<std::string> function_names;
    for (const auto& func : functions) {
        INTERNAL_CHECK(func) << "Program constructor encountered null function";
        auto func_name = func->name_;
        INTERNAL_CHECK(!func_name.empty()) << "Program constructor encountered empty function name";
        CHECK(function_names.find(func_name) == function_names.end())
            << "Duplicate function name \"" << func_name << "\"";
        function_names.insert(func_name);
        functions_.emplace(func_name, func);
    }
}

FunctionPtr Program::GetFunction(const std::string& name) const
{
    auto it = functions_.find(name);
    if (it != functions_.end()) {
        return it->second;
    }
    return nullptr;
}

} // namespace ir
} // namespace pypto
