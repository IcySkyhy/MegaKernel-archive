/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_IR_TYPE_H_
#define PYPTO_IR_TYPE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/core.h"
#include "ir/op_attr_types.h"
#include "ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

// Forward declarations
class Expr;
using ExprPtr = std::shared_ptr<const Expr>;

class MemRef;
using MemRefPtr = std::shared_ptr<const MemRef>;

/**
 * \brief Base class for type representations in the IR
 *
 * Types represent the structure and properties of values in the IR.
 * All types are immutable.
 */
class Type {
public:
    virtual ~Type() = default;

    /**
     * \brief Get the Kind of this type
     *
     * \return The ObjectKind enum value identifying the concrete type
     */
    [[nodiscard]] virtual ObjectKind GetKind() const = 0;

    /**
     * \brief Get the type name of this type
     *
     * \return Human-readable type name (e.g., "ScalarType", "TensorType")
     */
    [[nodiscard]] virtual std::string TypeName() const { return "Type"; }

    static constexpr auto GetFieldDescriptors() { return std::make_tuple(); }
};

using TypePtr = std::shared_ptr<const Type>;

/**
 * \brief Unknown type representation
 *
 * Represents an unknown or unspecified type.
 * Used as the default type for expressions when type information is not available.
 */
class UnknownType : public Type {
public:
    /**
     * \brief Create an unknown type
     */
    UnknownType() = default;

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::UnknownType; }
    [[nodiscard]] std::string TypeName() const override { return "UnknownType"; }

    static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using UnknownTypePtr = std::shared_ptr<const UnknownType>;

/**
 * \brief Get a shared pointer to the singleton UnknownType instance
 *
 * \return Shared pointer to UnknownType
 */
inline UnknownTypePtr GetUnknownType()
{
    static const auto unknownType = std::make_shared<UnknownType>();
    return unknownType;
}

/**
 * \brief Scalar type representation
 *
 * Represents a scalar value type with a data type.
 */
class ScalarType : public Type {
public:
    DataType dtype_;

    /**
     * \brief Create a scalar type
     *
     * \param dtype Data type
     */
    explicit ScalarType(DataType dtype) : dtype_(dtype) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ScalarType; }
    [[nodiscard]] std::string TypeName() const override { return "ScalarType"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Type::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&ScalarType::dtype_, "dtype")));
    }
};

using ScalarTypePtr = std::shared_ptr<const ScalarType>;

/**
 * \brief Tensor layout enumeration
 *
 * Used by both Tensor (global memory) and Tile (on-chip buffer) layers.
 *
 * Tensor layer (ND/DN/NZ only; ZN/NN/ZZ have no Tensor-layer meaning):
 * - ND: Row-major, contiguous in the last dimension
 * - DN: Column-major, contiguous in the first dimension
 * - NZ: Fractal — N-dimension outer block, Z-order inner block
 *
 * Tile layer — maps to (blayout, slayout) of HardwareInfo:
 *   blayout/slayout values: row_major, col_major, none_box
 * - ND: (row_major,  none_box)   — no fractal
 * - DN: (col_major,  none_box)   — no fractal
 * - NZ: (col_major,  row_major)  — standard fractal
 * - ZN: (row_major,  col_major)  — transposed fractal
 * - NN: (col_major,  col_major)  — both levels col_major
 * - ZZ: (row_major,  row_major)  — both levels row_major
 *
 * \see Defined via PYPTO_DECLARE_ENUM in ir/op_attr_types.h
 */

/**
 * \brief Tensor view representation
 *
 * Represents the view information for a tensor, including stride and layout.
 * The shape is stored in TensorType itself, so TensorView only needs
 * stride and layout information.
 */
struct TensorView {
    std::vector<ExprPtr> validShape; ///< Valid shape dimensions (symbolic or constant)
    std::vector<ExprPtr> stride;     ///< Stride for each dimension
    TensorLayout layout;             ///< Tensor layout type
    std::optional<ExprPtr> ptr;      ///< Source pointer (set for ptr.make_tensor-created views)

    TensorView() : layout(TensorLayout::ND) {}

    TensorView(std::vector<ExprPtr> strideIn, TensorLayout layoutIn) : stride(std::move(strideIn)), layout(layoutIn) {}

    TensorView(std::vector<ExprPtr> strideIn, TensorLayout layoutIn, ExprPtr ptrExpr)
        : stride(std::move(strideIn)), layout(layoutIn), ptr(std::move(ptrExpr))
    {}

    TensorView(std::vector<ExprPtr> validShapeIn, std::vector<ExprPtr> strideIn, TensorLayout layoutIn)
        : validShape(std::move(validShapeIn)), stride(std::move(strideIn)), layout(layoutIn)
    {}

    TensorView(std::vector<ExprPtr> validShapeIn, std::vector<ExprPtr> strideIn, TensorLayout layoutIn, ExprPtr ptrExpr)
        : validShape(std::move(validShapeIn)), stride(std::move(strideIn)), layout(layoutIn), ptr(std::move(ptrExpr))
    {}

    static constexpr auto GetFieldDescriptors()
    {
        return std::make_tuple(reflection::UsualField(&TensorView::validShape, "valid_shape"),
                               reflection::UsualField(&TensorView::stride, "stride"),
                               reflection::UsualField(&TensorView::layout, "layout"));
    }
};

/**
 * \brief Tile layout enumeration
 *
 * Shared by blayout and slayout fields in HardwareInfo:
 * - none_box: No layout constraint
 * - row_major: Row-major layout
 * - col_major: Column-major layout
 */
enum class TileLayout {
    none_box,  ///< No layout constraint
    row_major, ///< Row-major layout
    col_major  ///< Column-major layout
};

/**
 * \brief Tile pad enumeration
 *
 * Defines the padding mode for out-of-bound tile accesses:
 * - null: No padding
 * - zero: Pad with zero
 * - max: Pad with maximum value of the element type
 * - min: Pad with minimum value of the element type
 */
enum class TilePad {
    null, ///< No padding
    zero, ///< Zero padding
    max,  ///< Max value padding
    min   ///< Min value padding
};

/**
 * \brief Compact mode enumeration for tile buffer
 *
 * Defines the compact mode for tile buffers, used to handle
 * tail-block scenarios
 * where L0A/L0B split layouts are non-contiguous:
 * - null: No compact mode (default)
 * - normal: Normal compact mode
 * - row_plus_one: Row plus one compact mode
 */
enum class CompactMode {
    null,        ///< No compact mode (default)
    normal,      ///< Normal compact mode
    row_plus_one ///< Row plus one compact mode
};

/**
 * \brief Hardware-specific tile information
 *
 * Contains layout, fractal, padding, and compact mode parameters
 * that describe how a tile is physically stored in hardware memory.
 */
struct HardwareInfo {
    static constexpr uint64_t kDefaultFractal = 512;
    TileLayout blayout = TileLayout::row_major; ///< Block layout
    TileLayout slayout = TileLayout::none_box;  ///< Scatter layout
    uint64_t fractal = kDefaultFractal;         ///< Fractal size
    TilePad pad = TilePad::null;                ///< Pad mode
    CompactMode compact = CompactMode::null;    ///< Compact mode

    HardwareInfo() = default;

    HardwareInfo(TileLayout blayoutIn, TileLayout slayoutIn = TileLayout::none_box,
                 uint64_t fractalIn = kDefaultFractal, TilePad padIn = TilePad::null,
                 CompactMode compactIn = CompactMode::null)
        : blayout(blayoutIn), slayout(slayoutIn), fractal(fractalIn), pad(padIn), compact(compactIn)
    {}

    static constexpr auto GetFieldDescriptors()
    {
        return std::make_tuple(reflection::UsualField(&HardwareInfo::blayout, "blayout"),
                               reflection::UsualField(&HardwareInfo::slayout, "slayout"),
                               reflection::UsualField(&HardwareInfo::fractal, "fractal"),
                               reflection::UsualField(&HardwareInfo::pad, "pad"),
                               reflection::UsualField(&HardwareInfo::compact, "compact"));
    }
};

/**
 * \brief Tile view representation
 *
 * Represents the view information for a tile, including valid shape,
 * stride, and start offset. This is used by TileType to track how
 * a tile views its underlying memory.
 */
struct TileView {
    std::vector<ExprPtr> validShape; ///< Valid shape dimensions
    std::vector<ExprPtr> stride;     ///< Stride for each dimension
    ExprPtr startOffset;             ///< Starting offset

    TileView() = default;

    TileView(std::vector<ExprPtr> validShapeIn, std::vector<ExprPtr> strideIn, ExprPtr startOffsetIn)
        : validShape(std::move(validShapeIn)), stride(std::move(strideIn)), startOffset(std::move(startOffsetIn))
    {}

    static constexpr auto GetFieldDescriptors()
    {
        return std::make_tuple(reflection::UsualField(&TileView::validShape, "valid_shape"),
                               reflection::UsualField(&TileView::stride, "stride"),
                               reflection::UsualField(&TileView::startOffset, "start_offset"));
    }
};

/**
 * \brief Base class for shaped types (tensors and tiles)
 *
 * Represents types that have shape dimensions and optional memory references.
 * Both TensorType and TileType inherit from this class.
 */
class ShapedType : public Type {
public:
    DataType dtype_;                  ///< Element data type
    std::vector<ExprPtr> shape_;      ///< Shape dimensions (symbolic or constant)
    std::optional<MemRefPtr> memref_; ///< Optional memory reference (shared pointer)

    /**
     * \brief Create a shaped type without memory reference
     *
     * \param dtype Element data type
     * \param shape Shape dimensions
     */
    ShapedType(DataType dtype, std::vector<ExprPtr> shape)
        : dtype_(dtype), shape_(std::move(shape)), memref_(std::nullopt)
    {}

    /**
     * \brief Create a shaped type with constant shape
     *
     * \param dtype Element data type
     * \param shape Shape dimensions
     */
    ShapedType(DataType dtype, const std::vector<int64_t>& shape, std::optional<MemRefPtr> memref);

    /**
     * \brief Create a shaped type with memory reference (shared_ptr)
     *
     * \param dtype Element data type
     * \param shape Shape dimensions
     * \param memref Memory reference (shared pointer)
     */
    ShapedType(DataType dtype, std::vector<ExprPtr> shape, MemRefPtr memref)
        : dtype_(dtype), shape_(std::move(shape)), memref_(std::move(memref))
    {}

    /**
     * \brief Create a shaped type with optional memory reference (shared_ptr)
     *
     * \param dtype Element data type
     * \param shape Shape dimensions
     * \param memref Optional memory reference (shared pointer)
     */
    ShapedType(DataType dtype, std::vector<ExprPtr> shape, std::optional<MemRefPtr> memref)
        : dtype_(dtype), shape_(std::move(shape)), memref_(std::move(memref))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ShapedType; }
    [[nodiscard]] std::string TypeName() const override { return "ShapedType"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Type::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&ShapedType::dtype_, "dtype"),
                                              reflection::UsualField(&ShapedType::shape_, "shape"),
                                              reflection::UsualField(&ShapedType::memref_, "memref")));
    }
};

using ShapedTypePtr = std::shared_ptr<const ShapedType>;

/**
 * \brief Tensor type representation
 *
 * Represents a tensor type with a data type and shape dimensions.
 */
class TensorType : public ShapedType {
public:
    std::optional<TensorView> tensor_view_; ///< Optional tensor view information

    TensorType(std::vector<ExprPtr> shape, DataType dtype)
        : ShapedType(dtype, std::move(shape)), tensor_view_(std::nullopt)
    {}

    TensorType(std::vector<ExprPtr> shape, DataType dtype, MemRefPtr memref)
        : ShapedType(dtype, std::move(shape), std::move(memref)), tensor_view_(std::nullopt)
    {}

    /**
     * \brief Create a tensor type with optional memory reference (shared_ptr)
     *
     * \param shape Shape dimensions
     * \param dtype Element data type
     * \param memref Optional memory reference (shared pointer)
     */
    TensorType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref)
        : ShapedType(dtype, std::move(shape), std::move(memref)), tensor_view_(std::nullopt)
    {}

    TensorType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref)
        : ShapedType(dtype, shape, std::move(memref)), tensor_view_(std::nullopt)
    {}

    TensorType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref,
               std::optional<TensorView> tensorView)
        : ShapedType(dtype, std::move(shape), std::move(memref)), tensor_view_(std::move(tensorView))
    {}

    TensorType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref,
               std::optional<TensorView> tensorView)
        : ShapedType(dtype, shape, std::move(memref)), tensor_view_(std::move(tensorView))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TensorType; }
    [[nodiscard]] std::string TypeName() const override { return "TensorType"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(ShapedType::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&TensorType::tensor_view_, "tensor_view")));
    }
};

using TensorTypePtr = std::shared_ptr<const TensorType>;

/**
 * \brief Tile type representation
 *
 * Represents a tile type (multi-dimensional tensor).
 * Tiles are used for hardware-optimized operations on multi-dimensional data structures.
 * Note: Code generation currently only supports up to 2D tiles.
 */
class TileType : public ShapedType {
public:
    std::optional<TileView> tileView_;         ///< Optional tile view information
    std::optional<HardwareInfo> hardwareInfo_; ///< Optional hardware-specific tile information

    TileType(std::vector<ExprPtr> shape, DataType dtype)
        : ShapedType(dtype, std::move(shape)), tileView_(std::nullopt), hardwareInfo_(std::nullopt)
    {}

    TileType(std::vector<ExprPtr> shape, DataType dtype, MemRefPtr memref)
        : ShapedType(dtype, std::move(shape), std::move(memref)), tileView_(std::nullopt), hardwareInfo_(std::nullopt)
    {}

    TileType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref)
        : ShapedType(dtype, std::move(shape), std::move(memref)), tileView_(std::nullopt), hardwareInfo_(std::nullopt)
    {}

    TileType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref,
             std::optional<TileView> tileView, std::optional<HardwareInfo> hardwareInfo = std::nullopt)
        : ShapedType(dtype, shape, std::move(memref)),
          tileView_(std::move(tileView)),
          hardwareInfo_(std::move(hardwareInfo))
    {}

    TileType(std::vector<ExprPtr> shape, DataType dtype, MemRefPtr memref, std::optional<TileView> tileView,
             std::optional<HardwareInfo> hardwareInfo = std::nullopt)
        : ShapedType(dtype, std::move(shape), std::move(memref)),
          tileView_(std::move(tileView)),
          hardwareInfo_(std::move(hardwareInfo))
    {}

    TileType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref,
             std::optional<TileView> tileView, std::optional<HardwareInfo> hardwareInfo = std::nullopt)
        : ShapedType(dtype, std::move(shape), std::move(memref)),
          tileView_(std::move(tileView)),
          hardwareInfo_(std::move(hardwareInfo))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TileType; }
    [[nodiscard]] std::string TypeName() const override { return "TileType"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(ShapedType::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&TileType::tileView_, "tile_view"),
                                              reflection::UsualField(&TileType::hardwareInfo_, "hardware_info")));
    }
};

using TileTypePtr = std::shared_ptr<const TileType>;

/**
 * \brief Tuple type representation
 *
 * Represents a tuple type containing multiple types.
 * Tuples are used for multiple return values and structured data.
 */
class TupleType : public Type {
public:
    std::vector<TypePtr> types_; // Types in the tuple

    /**
     * \brief Create a (positional) tuple type
     *
     * Field names of named tuples / structs are NOT part of the tuple type. They are
     * carried by the parser-populated IRDebugInfo side table (see ir/debug_info.h),
     * keyed by the TupleType pointer, so they do not affect structural identity.
     *
     * \param types List of types in the tuple
     */
    explicit TupleType(std::vector<TypePtr> types);

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TupleType; }
    [[nodiscard]] std::string TypeName() const override { return "TupleType"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Type::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&TupleType::types_, "types")));
    }
};

using TupleTypePtr = std::shared_ptr<const TupleType>;

/**
 * \brief Memory reference type representation
 *
 * Represents a memory reference type for shaped data (tensors and tiles).
 * Used as the type for MemRef variables.
 */
class MemRefType : public Type {
public:
    /**
     * \brief Create a memory reference type
     */
    MemRefType() = default;

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::MemRefType; }
    [[nodiscard]] std::string TypeName() const override { return "MemRefType"; }

    static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using MemRefTypePtr = std::shared_ptr<const MemRefType>;

/**
 * \brief Get a shared pointer to the singleton MemRefType instance
 *
 * \return Shared pointer to MemRefType
 */
inline MemRefTypePtr GetMemRefType()
{
    static const auto memrefType = std::make_shared<MemRefType>();
    return memrefType;
}

/**
 * \brief Pointer type representation
 *
 * Represents a raw pointer to global memory of a specific element type.
 * Corresponds to `!pto.ptr<dtype>` in PTO MLIR.
 * Used as the base-pointer argument for `pl.make_tensor` body ops.
 *
 * `base_ptr` and `offset` are codegen-level annotations (excluded from
 * structural equality) that track the decomposition of chained addptr calls.
 */
class PtrType : public Type {
public:
    DataType dtype_; ///< Element type pointed to

    /// Original ptr function param expr (codegen-level, excluded from reflection)
    std::optional<ExprPtr> base_ptr;
    /// Accumulated element offset expr (codegen-level, excluded from reflection)
    std::optional<ExprPtr> offset;

    /**
     * \brief Create a pointer type
     *
     * \param dtype Element data type
     */
    explicit PtrType(DataType dtype) : dtype_(dtype) {}

    /**
     * \brief Create a pointer type with addptr decomposition
     *
     * \param dtype Element data type
     * \param base Original pointer parameter expression
     * \param off  Accumulated element offset expression
     */
    PtrType(DataType dtype, ExprPtr base, ExprPtr off)
        : dtype_(dtype), base_ptr(std::move(base)), offset(std::move(off))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::PtrType; }
    [[nodiscard]] std::string TypeName() const override { return "PtrType"; }

    // Only dtype_ participates in structural equality; base_ptr/offset are codegen annotations.
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Type::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&PtrType::dtype_, "dtype")));
    }
};

using PtrTypePtr = std::shared_ptr<const PtrType>;

enum class TokenKind : uint8_t {
    NORMAL,
    READ,
    WRITE,
};

/**
 * \brief Token type representation
 *
 * Represents an opaque token value used for side-effect ordering.
 * Tokens carry no data and are only used to establish dependencies
 * between operations.READ and WRITE tokens are semantic tokens used
 * by the tensor graph; NORMAL tokens are used by tile graph.
 */
class TokenType : public Type {
public:
    TokenKind kind_;

    explicit TokenType(TokenKind kind = TokenKind::NORMAL) : kind_(kind) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TokenType; }
    [[nodiscard]] std::string TypeName() const override { return "Token"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(Type::GetFieldDescriptors(),
                              std::make_tuple(reflection::UsualField(&TokenType::kind_, "kind")));
    }
};

using TokenTypePtr = std::shared_ptr<const TokenType>;

/**
 * \brief Get a shared pointer to the singleton TokenType instance
 *
 * \return Shared pointer to TokenType
 */
inline TokenTypePtr GetTokenType(TokenKind kind = TokenKind::NORMAL)
{
    static const auto normalTokenType = std::make_shared<TokenType>(TokenKind::NORMAL);
    static const auto readTokenType = std::make_shared<TokenType>(TokenKind::READ);
    static const auto writeTokenType = std::make_shared<TokenType>(TokenKind::WRITE);
    switch (kind) {
        case TokenKind::READ:
            return readTokenType;
        case TokenKind::WRITE:
            return writeTokenType;
        case TokenKind::NORMAL:
        default:
            return normalTokenType;
    }
}

inline TokenTypePtr GetReadTokenType() { return GetTokenType(TokenKind::READ); }

inline TokenTypePtr GetWriteTokenType() { return GetTokenType(TokenKind::WRITE); }

/**
 * \brief None type representation
 *
 * Represents the absence of a value (void-like). Carries no data.
 */
class NoneType : public Type {
public:
    NoneType() = default;

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::NoneType; }
    [[nodiscard]] std::string TypeName() const override { return "None"; }

    static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using NoneTypePtr = std::shared_ptr<const NoneType>;

/**
 * \brief Get a shared pointer to the singleton NoneType instance
 *
 * \return Shared pointer to NoneType
 */
inline NoneTypePtr GetNoneType()
{
    static const auto noneType = std::make_shared<NoneType>();
    return noneType;
}

/**
 * \brief Logical tensor type representation
 *
 * Represents a logical tensor with dtype and shape, without memory allocation info.
 * Used for tensor values in the PIL/IR layer before memory planning.
 */
class LogicalTensorType : public Type {
public:
    LogicalTensorType() = default;

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::LogicalTensorType; }
    [[nodiscard]] std::string TypeName() const override { return "LogicalTensorType"; }

    static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using LogicalTensorTypePtr = std::shared_ptr<const LogicalTensorType>;

inline LogicalTensorTypePtr GetLogicalTensorType()
{
    static const auto type = std::make_shared<LogicalTensorType>();
    return type;
}

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_TYPE_H_
