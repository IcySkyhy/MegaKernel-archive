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
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

#include "ir/core.h"
#include "ir/expr.h"
#include "ir/memory_space.h"
#include "ir/reflection/field_traits.h"
#include "ir/span.h"

namespace pypto {
namespace ir {

/**
 * \brief Memory reference variable for shaped types (tensor and tile)
 *
 * Represents a memory allocation with metadata (space, address, size, id).
 * Inherits from Var, making it a first-class IR expression that can be
 * declared and referenced like other variables.
 *
 * Memory references have auto-generated names based on their ID (e.g., "mem_ddr_0")
 * and MemRefType as their type.
 *
 * Aliasing analysis: use MayAlias() and SameAllocation() static methods to
 * determine if two MemRefs may reference overlapping memory regions.
 */
class MemRef : public Var {
public:
    MemorySpace memorySpace_; ///< Memory space (DDR, Vec, Mat, etc.)
    ExprPtr addr_;            ///< Starting address expression
    uint64_t size_;           ///< Size in bytes (64-bit unsigned)

    /**
     * \brief Constructor with all parameters including explicit ID
     *
     * Generates a variable name from the ID (e.g., "mem_ddr_0") and creates
     * a MemRefType for the type. Calls Var constructor with these values.
     *
     * \param memory_space Memory space (DDR, Vec, Mat, etc.)
     * \param addr Starting address expression
     * \param size Size in bytes
     * \param id Unique identifier (used to generate variable name)
     * \param span Source location (defaults to Span::Unknown())
     */
    MemRef(MemorySpace memory_space, ExprPtr addr, uint64_t size, uint64_t id, Span span = Span::Unknown());

    /**
     * \brief Constructor without explicit ID (id defaults to 0)
     *
     * Backwards-compatible 4-arg constructor for call sites that do not
     * track an allocation counter.
     */
    MemRef(MemorySpace memory_space, ExprPtr addr, uint64_t size, Span span = Span::Unknown());

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::MemRef; }
    [[nodiscard]] std::string TypeName() const override { return "MemRef"; }

    /// Are two MemRefs from the same allocation? (same space, size, and address)
    static bool SameAllocation(const MemRefPtr& a, const MemRefPtr& b);

    /// Do two MemRefs potentially alias? (same space + overlapping byte ranges)
    static bool MayAlias(const MemRefPtr& a, const MemRefPtr& b);

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * Note: id_ is an IgnoreField because it is used only for name generation
     * and should not affect structural equality/hashing between two MemRefs
     * that share identical memory space, address, and size.
     *
     * \return Tuple of field descriptors
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Var::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&MemRef::memorySpace_, "memory_space"),
                                              reflection::UsualField(&MemRef::addr_, "addr"),
                                              reflection::UsualField(&MemRef::size_, "size")));
    }
};

using MemRefPtr = std::shared_ptr<const MemRef>;

} // namespace ir
} // namespace pypto
