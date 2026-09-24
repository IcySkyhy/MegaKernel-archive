/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ir/core.h"
#include "ir/debug_info.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

/**
 * \brief Program definition
 *
 * Represents a complete program with functions.
 * Programs are immutable IR nodes.
 *
 * Functions are stored in an ordered map by name to ensure deterministic
 * ordering for structural equality and hashing.
 *
 * \note The function name must be unique within the program.
 */
class Program : public IRNode {
public:
    Program(std::map<std::string, FunctionPtr> functions, std::string name, Span span,
            IRDebugInfoPtr debugInfo = nullptr)
        : IRNode(std::move(span)),
          name_(std::move(name)),
          functions_(std::move(functions)),
          debugInfo_(std::move(debugInfo))
    {}

    Program(const std::vector<FunctionPtr>& functions, std::string name, Span span, IRDebugInfoPtr debugInfo = nullptr);

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Program; }
    [[nodiscard]] std::string TypeName() const override { return "Program"; }

    [[nodiscard]] FunctionPtr GetFunction(const std::string& name) const;

    /// Tuple/struct field-name side table; owned by the Program, supplied by the parser.
    /// May be null when a Program is built without one (e.g. rebuilt by a pass). Excluded
    /// from GetFieldDescriptors -> not part of structural equality / hash / serialization.
    [[nodiscard]] const IRDebugInfo* GetDebugInfo() const { return debugInfo_.get(); }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (name as IGNORE field, functions as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(IRNode::GetFieldDescriptors(),
                              std::make_tuple(reflection::IgnoreField(&Program::name_, "name"),
                                              reflection::UsualField(&Program::functions_, "functions")));
    }

public:
    std::string name_;
    std::map<std::string, FunctionPtr> functions_;
    /// Codegen-level annotation; excluded from reflection (see GetDebugInfo).
    IRDebugInfoPtr debugInfo_;
};

using ProgramPtr = std::shared_ptr<const Program>;

} // namespace ir
} // namespace pypto
