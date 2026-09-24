/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/memref.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <utility>

#include "core/error.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

std::string MemorySpaceToString(MemorySpace space)
{
    switch (space) {
        case MemorySpace::DDR:
            return "DDR";
        case MemorySpace::Vec:
            return "Vec";
        case MemorySpace::Mat:
            return "Mat";
        case MemorySpace::Left:
            return "Left";
        case MemorySpace::Right:
            return "Right";
        case MemorySpace::Scaling:
            return "Scaling";
        case MemorySpace::Acc:
            return "Acc";
        case MemorySpace::Bias:
            return "Bias";
        case MemorySpace::ScaleLeft:
            return "ScaleLeft";
        case MemorySpace::ScaleRight:
            return "ScaleRight";
        default:
            return "Unknown";
    }
}

MemorySpace StringToMemorySpace(const std::string& str)
{
    if (str == "DDR")
        return MemorySpace::DDR;
    if (str == "Vec")
        return MemorySpace::Vec;
    if (str == "Mat")
        return MemorySpace::Mat;
    if (str == "Left")
        return MemorySpace::Left;
    if (str == "Right")
        return MemorySpace::Right;
    if (str == "Scaling")
        return MemorySpace::Scaling;
    if (str == "Acc")
        return MemorySpace::Acc;
    if (str == "Bias")
        return MemorySpace::Bias;
    if (str == "ScaleLeft")
        return MemorySpace::ScaleLeft;
    if (str == "ScaleRight")
        return MemorySpace::ScaleRight;
    throw ValueError("Unknown MemorySpace: " + str);
}

// Helper for auto-generated variable names
static std::string ToLowerCase(const std::string& str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) { return std::tolower(c); });
    return result;
}

// 5-arg constructor (block semantics, with explicit id)
MemRef::MemRef(MemorySpace memory_space, ExprPtr addr, uint64_t size, uint64_t id, Span span)
    : Var("mem_" + ToLowerCase(MemorySpaceToString(memory_space)) + "_" + std::to_string(id), GetMemRefType(),
          std::move(span)),
      memorySpace_(memory_space),
      addr_(std::move(addr)),
      size_(size)
{}

// 4-arg constructor (id_ defaults to 0, for callers that do not track allocation counter)
MemRef::MemRef(MemorySpace memory_space, ExprPtr addr, uint64_t size, Span span)
    : MemRef(memory_space, std::move(addr), size, 0, std::move(span))
{}

// Aliasing analysis (merged from outer version)
bool MemRef::MayAlias(const MemRefPtr& a, const MemRefPtr& b)
{
    if (a->memorySpace_ != b->memorySpace_)
        return false;

    auto off_a = As<ConstInt>(a->addr_);
    auto off_b = As<ConstInt>(b->addr_);
    if (off_a && off_b) {
        int64_t end_a = off_a->value_ + static_cast<int64_t>(a->size_);
        int64_t end_b = off_b->value_ + static_cast<int64_t>(b->size_);
        return off_a->value_ < end_b && off_b->value_ < end_a;
    }
    return true; // same memory space, symbolic offsets → conservatively alias
}

bool MemRef::SameAllocation(const MemRefPtr& a, const MemRefPtr& b)
{
    if (a->memorySpace_ != b->memorySpace_ || a->size_ != b->size_)
        return false;
    auto off_a = As<ConstInt>(a->addr_);
    auto off_b = As<ConstInt>(b->addr_);
    if (off_a && off_b) {
        return off_a->value_ == off_b->value_;
    }
    return a->addr_ == b->addr_;
}

} // namespace ir
} // namespace pypto
