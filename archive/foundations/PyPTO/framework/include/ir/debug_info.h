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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pypto {
namespace ir {

class TupleType;
using TupleTypePtr = std::shared_ptr<const TupleType>;

/**
 * \brief Compilation-session side table for tuple field names.
 *
 * Field names of named tuples / structs are *not* part of the tuple type's core
 * semantics (positional element types and order are). They are only needed by
 * codegen during the struct-compat transition (to emit `a.field`) and for dump /
 * error display. This side table keeps that metadata out of both the IR nodes and
 * the TupleType structural identity.
 *
 * The key is the `TupleType` pointer. TupleType is never structurally interned --
 * each `struct.create` / `MakeTuple` allocates a fresh instance (`make_shared`),
 * so same-shape-different-name structs get distinct keys. IR passes copy `TypePtr`
 * by shared_ptr (types are immutable and never rebuilt via reflection), so the
 * pointer stays valid and identical while IR transformation passes rebuild the
 * surrounding nodes.
 *
 * The parser populates this (via `makeNamedTuple` / struct.create lowering); the
 * instance rides on `Program` so codegen can capture it at its entry point.
 */
class IRDebugInfo {
public:
    /// Record the ordered field-name list for a named tuple / struct type.
    void RegisterTupleFields(const TupleTypePtr& type, std::vector<std::string> fields)
    {
        if (type == nullptr) {
            return;
        }
        tupleFields_[type.get()] = std::move(fields);
    }

    /// Look up field names by type pointer. Returns nullptr if not registered.
    [[nodiscard]] const std::vector<std::string>* GetTupleFields(const TupleType* type) const
    {
        auto it = tupleFields_.find(type);
        return it == tupleFields_.end() ? nullptr : &it->second;
    }

    /// Record the C++ struct type name for a named tuple / struct type (e.g. the tiling
    /// Python class name), so codegen emits `struct <name>` instead of a fixed default.
    void RegisterTupleName(const TupleTypePtr& type, std::string name)
    {
        if (type == nullptr) {
            return;
        }
        tupleNames_[type.get()] = std::move(name);
    }

    /// Look up the struct type name by type pointer. Returns nullptr if not registered.
    [[nodiscard]] const std::string* GetTupleName(const TupleType* type) const
    {
        auto it = tupleNames_.find(type);
        return it == tupleNames_.end() ? nullptr : &it->second;
    }

    /// Merge another table's entries into this one (later registrations win).
    void Merge(const IRDebugInfo& other)
    {
        for (const auto& [type, fields] : other.tupleFields_) {
            tupleFields_[type] = fields;
        }
        for (const auto& [type, name] : other.tupleNames_) {
            tupleNames_[type] = name;
        }
    }

private:
    std::unordered_map<const TupleType*, std::vector<std::string>> tupleFields_;
    std::unordered_map<const TupleType*, std::string> tupleNames_;
};

using IRDebugInfoPtr = std::shared_ptr<IRDebugInfo>;

} // namespace ir
} // namespace pypto
