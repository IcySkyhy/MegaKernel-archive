/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/cce/code_context.h"

#include <cctype>
#include <cstddef>
#include <string>

#include "core/logging.h"
#include "tilefwk/error.h"
#include "ir/expr.h"

namespace pypto {

namespace codegen {

std::string CodeContext::GetVarName(const ir::VarPtr& var)
{
    CHECK(var != nullptr) << "Cannot get name for null variable";
    auto it = name_to_cpp_.find(var->name_);
    if (it != name_to_cpp_.end()) {
        return it->second;
    }
    // Fallback: when a Var references a pre-SSA base name (e.g. `_buf_tile_0`)
    // that was versioned by the SSA pass to `<name>_0` but the reference site
    // was not updated, look up the first versioned form.
    auto v0 = name_to_cpp_.find(var->name_ + "_0");
    if (v0 != name_to_cpp_.end()) {
        return v0->second;
    }
    // Fallback: for variables first defined inside a loop body, the post-loop
    // reference may use the raw base name while SSA registered a versioned form
    // (e.g. 'reg_dst' vs 'reg_dst_1'). Search for the highest-versioned match.
    std::string best_match;
    int best_version = -1;
    std::string prefix = var->name_ + "_";
    for (const auto& [key, value] : name_to_cpp_) {
        if (key.size() > prefix.size() && key.compare(0, prefix.size(), prefix) == 0) {
            const char* suffix = key.c_str() + prefix.size();
            bool all_digits = true;
            for (const char* p = suffix; *p; ++p) {
                if (*p < '0' || *p > '9') {
                    all_digits = false;
                    break;
                }
            }
            if (all_digits && *suffix != '\0') {
                int ver = std::atoi(suffix);
                if (ver > best_version) {
                    best_version = ver;
                    best_match = value;
                }
            }
        }
    }
    if (!best_match.empty()) {
        name_to_cpp_[var->name_] = best_match;
        return best_match;
    }
    // Auto-register: variable may originate from a different section (Cube/Vector)
    // that was cleared on section boundary. Use the sanitized IR name.
    std::string cpp_name = SanitizeName(var);
    name_to_cpp_[var->name_] = cpp_name;
    auto_registered_.insert(cpp_name);
    return cpp_name;
}

bool CodeContext::IsAutoRegistered(const std::string& cpp_name) const { return auto_registered_.count(cpp_name) > 0; }

void CodeContext::RegisterVar(const ir::VarPtr& var, const std::string& cpp_name)
{
    CHECK(var != nullptr) << "Cannot register null variable";
    CHECK(!cpp_name.empty()) << "Cannot register variable with empty name";

    // Check if this name is already registered (suppress for array access and alias optimizations)
    auto it = name_to_cpp_.find(var->name_);
    if (it != name_to_cpp_.end() && it->second != cpp_name && cpp_name.find('[') == std::string::npos) {
        IR_LOGW() << "Variable " << var->name_ << " re-registered with different C++ name: " << cpp_name << " vs "
                  << it->second;
    }

    // Register name-based mapping
    name_to_cpp_[var->name_] = cpp_name;
}

void CodeContext::Clear() { name_to_cpp_.clear(); }

std::string CodeContext::SanitizeName(const ir::VarPtr& var) const
{
    CHECK(var != nullptr) << "Cannot sanitize null variable";
    auto ir_name = var->name_;
    if (ir_name.empty()) {
        return "var";
    }

    std::string result;
    result.reserve(ir_name.size());

    // First character must be letter or underscore
    if (std::isalpha(static_cast<unsigned char>(ir_name[0])) || ir_name[0] == '_') {
        result += ir_name[0];
    } else {
        result += '_';
    }

    // Subsequent characters can be alphanumeric or underscore
    for (size_t i = 1; i < ir_name.size(); ++i) {
        char c = ir_name[i];
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            result += c;
        } else {
            result += '_';
        }
    }

    return result;
}

} // namespace codegen

} // namespace pypto
