/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/cce/type_converter.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/logging.h"
#include "ir/op_attr_types.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/pipe.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {

namespace codegen {

namespace {

// Resolve a valid_shape ExprPtr to a string for use in the Tile<> type template.
// Only ConstInt values are supported (Var �?compile-time unknown, skip).
// Returns "" if the expression cannot be resolved to a constant.
// ConstInt(-1) means "use full shape at runtime" �?emits "-1" in template (dynamic valid_shape).
std::string ResolveValidShapeDim(const ir::ExprPtr& expr, int64_t /*fallback*/)
{
    if (!expr)
        return "-1";
    if (auto c = ir::As<ir::ConstInt>(expr)) {
        return (c->value_ == -1) ? "-1" : std::to_string(c->value_);
    }
    // Var or other dynamic expression: cannot embed in template parameter.
    return "";
}

std::string ConvertTilePadToPTOValue(ir::TilePad pad)
{
    switch (pad) {
        case ir::TilePad::null:
            return "PadValue::Null";
        case ir::TilePad::zero:
            return "PadValue::Zero";
        case ir::TilePad::max:
            return "PadValue::Max";
        case ir::TilePad::min:
            return "PadValue::Min";
        default:
            throw pypto::ir::ValueError("Invalid TilePad value");
    }
}

std::string ConvertCompactModeToPTOValue(ir::CompactMode compact)
{
    switch (compact) {
        case ir::CompactMode::null:
            return "";
        case ir::CompactMode::normal:
            return "CompactMode::Normal";
        case ir::CompactMode::row_plus_one:
            return "CompactMode::RowPlusOne";
        default:
            throw pypto::ir::ValueError("Invalid CompactMode value");
    }
}

void ApplyTileViewValidShape(const ir::TileTypePtr& tile_type, int64_t rows, int64_t cols, std::string& type_rows,
                             std::string& type_cols)
{
    if (!tile_type->tileView_.has_value()) {
        return;
    }
    const auto& tv = tile_type->tileView_.value();
    if (tv.validShape.empty()) {
        return;
    }
    std::string resolved_rows = tv.validShape.size() >= 1 ? ResolveValidShapeDim(tv.validShape[0], rows) : "";
    std::string resolved_cols = tv.validShape.size() >= 2 ? ResolveValidShapeDim(tv.validShape[1], cols) : "";
    if (!resolved_rows.empty()) {
        type_rows = resolved_rows;
    }
    if (!resolved_cols.empty()) {
        type_cols = resolved_cols;
    }
}

} // namespace

std::string TypeConverter::ConvertTileType(const ir::TileTypePtr& tile_type, int64_t rows, int64_t cols) const
{
    std::ostringstream type_alias;
    if (!tile_type->memref_.has_value()) {
        // No memref: IfStmt return variable or intermediate tile.
        std::string tile_type_str = "TileType::Vec";
        std::string BLayout = "RowMajor";
        std::string SLayout = "NoneBox";
        std::string fractal = "512";
        std::string pad_value;
        if (cols == 1) {
            BLayout = "ColMajor";
        }
        // Default valid_shape template params to -1 (dynamic); overridden when user provides explicit values.
        std::string type_rows = "-1";
        std::string type_cols = "-1";
        std::string compact_value;
        if (tile_type->hardwareInfo_.has_value()) {
            const auto& hw = tile_type->hardwareInfo_.value();
            BLayout = ConvertTileLayout(hw.blayout);
            SLayout = ConvertTileLayout(hw.slayout);
            fractal = std::to_string(hw.fractal);
            pad_value = ConvertTilePadToPTOValue(hw.pad);
            compact_value = ConvertCompactModeToPTOValue(hw.compact);
            using TL = ir::TileLayout;
            if (hw.blayout == TL::col_major && hw.slayout == TL::row_major)
                tile_type_str = "TileType::Mat";
            else if (hw.blayout == TL::row_major && hw.slayout == TL::row_major)
                tile_type_str = "TileType::Left";
            else if (hw.blayout == TL::row_major && hw.slayout == TL::col_major)
                tile_type_str = "TileType::Right";
        }
        ApplyTileViewValidShape(tile_type, rows, cols, type_rows, type_cols);
        type_alias << "Tile<" << tile_type_str << ", " << tile_type->dtype_.ToCTypeString() << ", " << rows << ", "
                   << cols << ", BLayout::" << BLayout << ", " << type_rows << ", " << type_cols
                   << ", SLayout::" << SLayout << ", " << fractal;
        if (!pad_value.empty()) {
            type_alias << ", " << pad_value;
        }
        if (!compact_value.empty()) {
            type_alias << ", " << compact_value;
        }
        type_alias << ">";
        return type_alias.str();
    }
    ir::MemorySpace space = (*tile_type->memref_)->memorySpace_; // NOLINT(bugprone-unchecked-optional-access)
    std::string tile_type_str = ConvertMemorySpaceToTileType(space);

    // TODO(YunjiQin): BLayout and SLayout should be determined by the tile format
    std::string BLayout = "RowMajor";
    std::string SLayout = "NoneBox";
    std::string fractal = "512";
    std::string pad_value;
    std::string compact_value;
    // Default valid_shape template params to -1 (dynamic); overridden when user provides explicit values.
    std::string type_rows = "-1";
    std::string type_cols = "-1";

    if (cols == 1) {
        BLayout = "ColMajor";
    } else if (tile_type->hardwareInfo_.has_value()) {
        const auto& hw = tile_type->hardwareInfo_.value();
        BLayout = ConvertTileLayout(hw.blayout);
        SLayout = ConvertTileLayout(hw.slayout);
        fractal = std::to_string(hw.fractal);
        pad_value = ConvertTilePadToPTOValue(hw.pad);
        compact_value = ConvertCompactModeToPTOValue(hw.compact);
    }
    ApplyTileViewValidShape(tile_type, rows, cols, type_rows, type_cols);
    type_alias << "Tile<" << tile_type_str << ", " << tile_type->dtype_.ToCTypeString() << ", " << rows << ", " << cols
               << ", BLayout::" << BLayout << ", " << type_rows << ", " << type_cols << ", SLayout::" << SLayout << ", "
               << fractal;
    if (!pad_value.empty()) {
        type_alias << ", " << pad_value;
    }
    if (!compact_value.empty()) {
        type_alias << ", " << compact_value;
    }
    type_alias << ">";

    return type_alias.str();
}

std::string TypeConverter::ConvertMemorySpaceToTileType(ir::MemorySpace space) const
{
    switch (space) {
        case ir::MemorySpace::Left:
            return "TileType::Left";
        case ir::MemorySpace::Right:
            return "TileType::Right";
        case ir::MemorySpace::Acc:
            return "TileType::Acc";
        case ir::MemorySpace::Mat:
            return "TileType::Mat";
        case ir::MemorySpace::Vec:
            return "TileType::Vec";
        case ir::MemorySpace::Scaling:
            return "TileType::Scaling";
        case ir::MemorySpace::Bias:
            return "TileType::Bias";
        case ir::MemorySpace::ScaleLeft:
            return "TileType::ScaleLeft";
        case ir::MemorySpace::ScaleRight:
            return "TileType::ScaleRight";
        case ir::MemorySpace::DDR:
            // DDR is for GlobalTensor, not Tile - should not reach here
            throw pypto::ir::ValueError("DDR is for GlobalTensor, not Tile");
        default:
            throw pypto::ir::ValueError("Invalid MemorySpace value");
    }
}

std::string TypeConverter::ConvertPipeType(ir::PipeType pipe) const
{
    if (pipe == ir::PipeType::MTE1) {
        return "PIPE_MTE1";
    }
    if (pipe == ir::PipeType::MTE2) {
        return "PIPE_MTE2";
    }
    if (pipe == ir::PipeType::MTE3) {
        return "PIPE_MTE3";
    }
    if (pipe == ir::PipeType::M) {
        return "PIPE_M";
    }
    if (pipe == ir::PipeType::V) {
        return "PIPE_V";
    }
    if (pipe == ir::PipeType::S) {
        return "PIPE_S";
    }
    if (pipe == ir::PipeType::FIX) {
        return "PIPE_FIX";
    }
    if (pipe == ir::PipeType::ALL) {
        return "PIPE_ALL";
    }
    throw pypto::ir::ValueError("Invalid PipeType value");
}

std::string TypeConverter::ConvertEventId(int event_id) const
{
    CHECK(event_id >= 0 && event_id <= 7) << "Event ID must be in range [0, 7], got " << event_id;
    return "EVENT_ID" + std::to_string(event_id);
}

std::string TypeConverter::ConvertCastRoundMode(int mode) const
{
    auto m = static_cast<ir::RoundMode>(mode);
    return ir::EnumToString(m);
}

std::string TypeConverter::ConvertTileLayout(ir::TileLayout layout) const
{
    switch (layout) {
        case ir::TileLayout::none_box:
            return "NoneBox";
        case ir::TileLayout::row_major:
            return "RowMajor";
        case ir::TileLayout::col_major:
            return "ColMajor";
        default:
            throw pypto::ir::ValueError("Invalid TileLayout value");
    }
}

std::string TypeConverter::GenerateShapeType(const std::vector<int64_t>& dims) const
{
    CHECK(!dims.empty()) << "Cannot generate Shape type for empty dimensions";

    std::ostringstream oss;
    oss << "pto::Shape<";

    // Pad to 5 dimensions with leading 1s
    const size_t target_dims = 5;
    CHECK(dims.size() <= target_dims) << "Cannot generate Shape with more than " << target_dims << " dimensions, got "
                                      << dims.size();

    // Add leading 1s for padding
    for (size_t i = 0; i < target_dims - dims.size(); ++i) {
        oss << "1, ";
    }

    // Add actual dimensions
    for (size_t i = 0; i < dims.size(); ++i) {
        oss << dims[i];
        if (i < dims.size() - 1) {
            oss << ", ";
        }
    }

    oss << ">";
    return oss.str();
}

std::string TypeConverter::GenerateStrideType(const std::vector<int64_t>& shape) const
{
    CHECK(!shape.empty()) << "Cannot generate Stride type for empty shape";

    std::ostringstream oss;
    oss << "pto::Stride<";

    // Pad to 5 dimensions with leading 1s
    const size_t target_dims = 5;
    CHECK(shape.size() <= target_dims) << "Cannot generate Stride with more than " << target_dims << " dimensions, got "
                                       << shape.size();

    // Add leading 1s for padding
    for (size_t i = 0; i < target_dims - shape.size(); ++i) {
        oss << "1, ";
    }

    // set dynamic strides, will get from runtime
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << "-1";
        if (i < shape.size() - 1) {
            oss << ", ";
        }
    }

    oss << ">";
    return oss.str();
}

} // namespace codegen

} // namespace pypto
