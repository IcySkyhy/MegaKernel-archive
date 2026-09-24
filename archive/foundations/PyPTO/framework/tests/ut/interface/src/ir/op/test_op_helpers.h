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

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {
namespace test_helpers {

inline Span Sp(const char* file = "test") { return Span(file, 1, 1); }

inline ExprPtr MakeScalarVar(const std::string& name, DataType dt, Span sp = Span("test", 1, 1))
{
    return std::make_shared<Var>(name, std::make_shared<ScalarType>(dt), sp);
}

inline ExprPtr MakeTileVar(const std::string& name, std::vector<int64_t> dims, DataType dt,
                           Span sp = Span("test", 1, 1))
{
    std::vector<ExprPtr> shape;
    shape.reserve(dims.size());
    for (auto d : dims) {
        shape.push_back(std::make_shared<ConstInt>(d, DataType::INT64, sp));
    }
    return std::make_shared<Var>(name, std::make_shared<TileType>(shape, dt), sp);
}

inline ExprPtr MakeTensorVar(const std::string& name, std::vector<int64_t> dims, DataType dt,
                             Span sp = Span("test", 1, 1))
{
    std::vector<ExprPtr> shape;
    shape.reserve(dims.size());
    for (auto d : dims) {
        shape.push_back(std::make_shared<ConstInt>(d, DataType::INT64, sp));
    }
    return std::make_shared<Var>(name, std::make_shared<TensorType>(shape, dt), sp);
}

inline ExprPtr MakeOffsetsTuple(std::vector<int64_t> offsets, Span sp = Span("test", 1, 1))
{
    std::vector<ExprPtr> elems;
    elems.reserve(offsets.size());
    for (auto o : offsets) {
        elems.push_back(std::make_shared<ConstInt>(o, DataType::INT64, sp));
    }
    return std::make_shared<MakeTuple>(elems, sp);
}

inline ExprPtr MakeTileVarWithHwInfo(const std::string& name, std::vector<int64_t> dims, DataType dt, TilePad pad,
                                     Span sp = Span("test", 1, 1))
{
    HardwareInfo hw(TileLayout::row_major, TileLayout::none_box, 512, pad, CompactMode::null);
    return std::make_shared<Var>(name,
                                 std::make_shared<TileType>(dims, dt, std::optional<MemRefPtr>(std::nullopt),
                                                            std::nullopt, std::optional<HardwareInfo>(hw)),
                                 sp);
}

inline ExprPtr MakeTileVarWithMemRef(const std::string& name, std::vector<int64_t> dims, DataType dt, MemorySpace ms,
                                     int64_t addr, uint64_t size, Span sp = Span("test", 1, 1))
{
    std::vector<ExprPtr> shape;
    shape.reserve(dims.size());
    for (auto d : dims) {
        shape.push_back(std::make_shared<ConstInt>(d, DataType::INT64, sp));
    }
    auto addr_expr = std::make_shared<ConstInt>(addr, DataType::INDEX, sp);
    auto memref = std::make_shared<MemRef>(ms, addr_expr, size);
    return std::make_shared<Var>(name, std::make_shared<TileType>(shape, dt, std::optional<MemRefPtr>(memref)), sp);
}

inline ExprPtr MakePtrVar(const std::string& name, DataType dt, Span sp = Span("test", 1, 1))
{
    return std::make_shared<Var>(name, std::make_shared<PtrType>(dt), sp);
}

inline ExprPtr MakeIntTuple(std::vector<int64_t> vals) { return MakeOffsetsTuple(vals); }

inline TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }

inline IRNodePtr Node(const IRNodePtr& p) { return p; }

} // namespace test_helpers
} // namespace ir
} // namespace pypto
