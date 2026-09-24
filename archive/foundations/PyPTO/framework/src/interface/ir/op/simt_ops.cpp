/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file simt_ops.cpp
 * \brief SIMT context and launch operations used by PyPTO Pro's A5 direct-CCE path.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_attr_types.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceSimtContextComponentType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    CHECK(args.empty()) << "SIMT context operations do not accept positional arguments";
    int axis = GetOpKwarg<int>(kwargs, "axis", 0);
    CHECK(axis >= 0 && axis <= 2) << "SIMT context axis must be in [0, 2]";
    return std::make_shared<ScalarType>(DataType(DataType::UINT32));
}

TypePtr DeduceSimtLinearThreadIdxType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    CHECK(args.empty()) << "SIMT builtin scalar operations do not accept positional arguments";
    CHECK(kwargs.empty()) << "SIMT builtin scalar operations do not accept keyword arguments";
    return std::make_shared<ScalarType>(DataType(DataType::UINT32));
}

TypePtr DeduceSimtWarpSizeType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    CHECK(args.empty()) << "simt.warp_size does not accept positional arguments";
    CHECK(kwargs.empty()) << "simt.warp_size does not accept keyword arguments";
    return std::make_shared<ScalarType>(DataType(DataType::INT32));
}

bool IsSimtCastIntegerDtype(DataType dtype)
{
    return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
           dtype == DataType::INT64 || dtype == DataType::UINT8 || dtype == DataType::UINT16 ||
           dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

bool IsSimtCastWideIntegerDtype(DataType dtype)
{
    return dtype == DataType::INT32 || dtype == DataType::INT64 || dtype == DataType::UINT32 ||
           dtype == DataType::UINT64;
}

bool IsSimtCastSupportedDtype(DataType dtype)
{
    return IsSimtCastIntegerDtype(dtype) || dtype == DataType::FP16 || dtype == DataType::BF16 ||
           dtype == DataType::FP32;
}

bool IsSimtCastStandardRoundMode(RoundMode mode)
{
    return mode == RoundMode::CAST_NONE || mode == RoundMode::CAST_RINT || mode == RoundMode::CAST_ROUND ||
           mode == RoundMode::CAST_FLOOR || mode == RoundMode::CAST_CEIL || mode == RoundMode::CAST_TRUNC;
}

bool IsSimtCastSupported(DataType source_dtype, DataType target_dtype, RoundMode mode)
{
    if (source_dtype == target_dtype) {
        return IsSimtCastSupportedDtype(source_dtype) && mode == RoundMode::CAST_NONE;
    }
    if (IsSimtCastIntegerDtype(source_dtype) && IsSimtCastIntegerDtype(target_dtype)) {
        return mode == RoundMode::CAST_NONE;
    }
    if ((source_dtype == DataType::FP16 || source_dtype == DataType::BF16) && target_dtype == DataType::FP32) {
        return mode == RoundMode::CAST_NONE;
    }
    if (source_dtype == DataType::FP32 && target_dtype == DataType::FP16) {
        return IsSimtCastStandardRoundMode(mode) || mode == RoundMode::CAST_ODD;
    }
    if (source_dtype == DataType::FP32 && target_dtype == DataType::BF16) {
        return IsSimtCastStandardRoundMode(mode);
    }
    if ((source_dtype == DataType::FP32 && IsSimtCastWideIntegerDtype(target_dtype)) ||
        (IsSimtCastWideIntegerDtype(source_dtype) && target_dtype == DataType::FP32)) {
        return IsSimtCastStandardRoundMode(mode);
    }
    return false;
}

TypePtr DeduceSimtCastType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    CHECK(args.size() == 1) << "simt.cast requires one scalar argument";
    auto source_type = As<ScalarType>(args[0]->GetType());
    CHECK(source_type) << "simt.cast value must be a scalar";

    DataType target_dtype = GetOpKwarg<DataType>(kwargs, "target_type");
    auto mode = static_cast<RoundMode>(GetOpKwarg<int>(kwargs, "mode"));
    CHECK(IsSimtCastSupported(source_type->dtype_, target_dtype, mode))
        << "simt.cast does not support " << source_type->dtype_.ToString() << " -> " << target_dtype.ToString()
        << " with mode " << EnumToString(mode);
    return std::make_shared<ScalarType>(target_dtype);
}

bool IsSimtAtomicDtype(const DataType& dtype, std::initializer_list<DataType> supported_dtypes)
{
    for (const auto& supported_dtype : supported_dtypes) {
        if (dtype == supported_dtype) {
            return true;
        }
    }
    return false;
}

TypePtr DeduceSimtAtomicType(const std::string& op_name, size_t operand_count,
                             std::initializer_list<DataType> ub_dtypes, std::initializer_list<DataType> gm_dtypes,
                             const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             std::initializer_list<DataType> no_result_dtypes = {})
{
    CHECK(args.size() == operand_count + 2)
        << op_name << " requires container, offset, and " << operand_count << " scalar operand(s)";
    CHECK(kwargs.empty()) << op_name << " does not accept keyword arguments";

    auto tile_type = As<TileType>(args[0]->GetType());
    auto tensor_type = As<TensorType>(args[0]->GetType());
    CHECK(tile_type || tensor_type) << op_name << " container must be a Tile or Tensor";
    DataType dtype = tile_type ? tile_type->dtype_ : tensor_type->dtype_;
    CHECK(IsSimtAtomicDtype(dtype, tile_type ? ub_dtypes : gm_dtypes))
        << op_name << " does not support dtype " << dtype.ToString() << " on " << (tile_type ? "UB Tile" : "GM Tensor");

    auto offset_type = As<ScalarType>(args[1]->GetType());
    CHECK(offset_type && offset_type->dtype_ != DataType::BOOL &&
          (offset_type->dtype_.IsInt() || offset_type->dtype_ == DataType::INDEX))
        << op_name << " offset must be a non-bool integer scalar";

    for (size_t i = 0; i < operand_count; ++i) {
        auto operand_type = As<ScalarType>(args[i + 2]->GetType());
        CHECK(operand_type) << op_name << " operand " << i << " must be a scalar";
        CHECK(operand_type->dtype_ == dtype) << op_name << " operand " << i << " dtype must match target dtype "
                                             << dtype.ToString() << ", but got " << operand_type->dtype_.ToString();
    }
    if (IsSimtAtomicDtype(dtype, no_result_dtypes)) {
        return GetNoneType();
    }
    return std::make_shared<ScalarType>(dtype);
}

std::string FormatSupportedDtypes(std::initializer_list<DataType> dtypes)
{
    std::string result;
    for (const auto& dtype : dtypes) {
        if (!result.empty()) {
            result += ", ";
        }
        result += dtype.ToString();
    }
    return result;
}

TypePtr DeduceSimtMathType(const std::string& op_name, size_t operand_count,
                           std::initializer_list<DataType> supported_dtypes, bool returns_bool,
                           const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    CHECK(args.size() == operand_count) << op_name << " requires exactly " << operand_count << " scalar operand(s)";
    CHECK(kwargs.empty()) << op_name << " does not accept keyword arguments";

    DataType dtype = DataType::BOOL;
    for (size_t i = 0; i < args.size(); ++i) {
        auto scalar_type = As<ScalarType>(args[i]->GetType());
        CHECK(scalar_type) << op_name << " operand " << i << " must be a scalar";
        if (i == 0) {
            dtype = scalar_type->dtype_;
            CHECK(IsSimtAtomicDtype(dtype, supported_dtypes))
                << op_name << " supports only " << FormatSupportedDtypes(supported_dtypes) << ", got "
                << dtype.ToString();
        } else {
            CHECK(scalar_type->dtype_ == dtype) << op_name << " requires operands with the same dtype, got "
                                                << dtype.ToString() << " and " << scalar_type->dtype_.ToString();
        }
    }
    return std::make_shared<ScalarType>(returns_bool ? DataType::BOOL : dtype);
}

TypePtr DeduceSimtLaunchType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    CHECK(args.size() >= 3) << "simt.launch requires three launch dimensions";
    int64_t total_threads = 1;
    for (std::size_t i = 0; i < 3; ++i) {
        auto dim_type = As<ScalarType>(args[i]->GetType());
        CHECK(dim_type && dim_type->dtype_.IsInt() && dim_type->dtype_ != DataType(DataType::BOOL))
            << "simt.launch dimensions must be non-bool integer scalars";
        auto dim = As<ConstInt>(args[i]);
        CHECK(dim && dim->value_ > 0 && dim->value_ <= 2048)
            << "simt.launch dimensions must be compile-time integers in [1, 2048]";
        total_threads *= dim->value_;
    }
    CHECK(total_threads <= 2048) << "simt.launch total thread count must not exceed 2048";
    int max_threads = GetOpKwarg<int>(kwargs, "max_threads");
    CHECK(max_threads >= 1 && max_threads <= 2048) << "simt.launch max_threads must be in [1, 2048]";
    CHECK(total_threads <= max_threads) << "simt.launch threads " << total_threads << " exceed callee max_threads "
                                        << max_threads;
    for (std::size_t i = 3; i < args.size(); ++i) {
        auto arg_type = args[i]->GetType();
        if (auto tile_type = As<TileType>(arg_type)) {
            CHECK(tile_type->shape_.size() == 2) << "simt.launch Tile argument must have a two-dimensional shape";
            CHECK(As<Var>(args[i]) != nullptr) << "simt.launch Tile argument must be a Var";
            for (const auto& dim : tile_type->shape_) {
                CHECK(As<ConstInt>(dim) != nullptr) << "simt.launch Tile argument must have a static shape";
            }
            CHECK(tile_type->dtype_.GetBit() >= 8) << "simt.launch Tile argument has a sub-byte element dtype";
            CHECK(tile_type->memref_.has_value()) << "simt.launch Tile argument has no memory reference";
            CHECK((*tile_type->memref_)->memorySpace_ == MemorySpace::Vec)
                << "simt.launch Tile argument must be a Vec-memory Tile";
            CHECK(tile_type->hardwareInfo_.has_value() && tile_type->hardwareInfo_->blayout == TileLayout::row_major &&
                  tile_type->hardwareInfo_->slayout == TileLayout::none_box)
                << "simt.launch Tile argument must be an ND Vec Tile";
        } else if (auto tensor_type = As<TensorType>(arg_type)) {
            CHECK(!tensor_type->shape_.empty()) << "simt.launch Tensor argument must have a non-empty shape";
            for (const auto& dim : tensor_type->shape_) {
                CHECK(As<ConstInt>(dim) != nullptr) << "simt.launch Tensor argument must have a static shape";
            }
            CHECK(tensor_type->dtype_.GetBit() >= 8) << "simt.launch Tensor argument has a sub-byte element dtype";
            CHECK(tensor_type->tensor_view_.has_value() && tensor_type->tensor_view_->layout == TensorLayout::ND)
                << "simt.launch Tensor argument must use ND layout";
            CHECK(As<Var>(args[i]) != nullptr) << "simt.launch Tensor argument must be a Var";
        } else {
            CHECK(As<ScalarType>(arg_type)) << "simt.launch supports only Tile, Tensor, and scalar arguments";
        }
    }
    return GetUnknownType();
}

} // namespace

REGISTER_OP("simt.thread_idx")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT thread index")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.block_dim")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT block dimensions")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.block_idx")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT block index")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.grid_dim")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT grid dimensions")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.linear_thread_idx")
    .set_op_category("SimtOp")
    .set_description("Read the x-major flattened thread index within the current block")
    .no_argument()
    .f_deduce_type(DeduceSimtLinearThreadIdxType);

REGISTER_OP("simt.warp_size")
    .set_op_category("SimtOp")
    .set_description("Read the native SIMT warp size")
    .no_argument()
    .f_deduce_type(DeduceSimtWarpSizeType);

REGISTER_OP("simt.cast")
    .set_op_category("SimtOp")
    .set_description("Convert one SIMT scalar value to a supported target dtype")
    .add_argument("value", "Source scalar value")
    .set_attr<DataType>("target_type")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceSimtCastType);

#define REGISTER_SIMT_MATH_UNARY_OP(OpName, Description, ReturnsBool, ...)                                      \
    REGISTER_OP("simt." OpName)                                                                                 \
        .set_op_category("SimtOp")                                                                              \
        .set_description(Description)                                                                           \
        .add_argument("value", "Scalar operand")                                                                \
        .f_deduce_type(                                                                                         \
            [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) { \
                return DeduceSimtMathType("simt." OpName, 1, {__VA_ARGS__}, ReturnsBool, args, kwargs);         \
            })

REGISTER_SIMT_MATH_UNARY_OP("abs", "Compute the absolute value of one supported scalar", false, DataType::FP16,
                            DataType::BF16, DataType::FP32, DataType::INT64);
REGISTER_SIMT_MATH_UNARY_OP("sqrt", "Compute the square root of one floating-point scalar", false, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("rsqrt", "Compute the reciprocal square root of one floating-point scalar", false,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("exp", "Compute the natural exponential of one floating-point scalar", false,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("exp2", "Compute the base-two exponential of one floating-point scalar", false,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("log", "Compute the natural logarithm of one floating-point scalar", false, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("log2", "Compute the base-two logarithm of one floating-point scalar", false,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("log1p", "Compute log(1 + value) for one FP32 scalar", false, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("sin", "Compute the sine of one floating-point scalar", false, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("cos", "Compute the cosine of one floating-point scalar", false, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("tanh", "Compute the hyperbolic tangent of one floating-point scalar", false,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("rint", "Round one floating-point scalar to the nearest integer value", false,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("round", "Round one floating-point scalar halfway away from zero", false, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("floor", "Round one floating-point scalar downward", false, DataType::FP16, DataType::BF16,
                            DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("ceil", "Round one floating-point scalar upward", false, DataType::FP16, DataType::BF16,
                            DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("trunc", "Round one floating-point scalar toward zero", false, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("isnan", "Test whether one floating-point scalar is NaN", true, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("isinf", "Test whether one floating-point scalar is infinite", true, DataType::FP16,
                            DataType::BF16, DataType::FP32);

#undef REGISTER_SIMT_MATH_UNARY_OP

REGISTER_OP("simt.min")
    .set_op_category("SimtOp")
    .set_description("Compute the minimum of two same-dtype supported scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType(
            "simt.min", 2,
            {DataType::FP16, DataType::BF16, DataType::FP32, DataType::INT8, DataType::INT16, DataType::INT32,
             DataType::INT64, DataType::UINT8, DataType::UINT16, DataType::UINT32, DataType::UINT64},
            false, args, kwargs);
    });

REGISTER_OP("simt.max")
    .set_op_category("SimtOp")
    .set_description("Compute the maximum of two same-dtype supported scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType(
            "simt.max", 2,
            {DataType::FP16, DataType::BF16, DataType::FP32, DataType::INT8, DataType::INT16, DataType::INT32,
             DataType::INT64, DataType::UINT8, DataType::UINT16, DataType::UINT32, DataType::UINT64},
            false, args, kwargs);
    });

REGISTER_OP("simt.fma")
    .set_op_category("SimtOp")
    .set_description("Compute a fused multiply-add of three same-dtype floating-point scalars")
    .add_argument("lhs", "Left multiplication operand")
    .add_argument("rhs", "Right multiplication operand")
    .add_argument("addend", "Scalar addend")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType("simt.fma", 3, {DataType::FP16, DataType::BF16, DataType::FP32}, false, args, kwargs);
    });

REGISTER_OP("simt.atomic_add")
    .set_op_category("SimtOp")
    .set_description("Atomically add to one UB Tile or GM Tensor element; FP16/BF16 return no value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Scalar addend")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_add", 1,
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32},
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32,
                                     DataType::INT64, DataType::UINT64},
                                    args, kwargs, {DataType::FP16, DataType::BF16});
    });

REGISTER_OP("simt.atomic_sub")
    .set_op_category("SimtOp")
    .set_description("Atomically subtract from one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Scalar subtrahend")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType(
            "simt.atomic_sub", 1, {DataType::INT32, DataType::UINT32, DataType::FP32},
            {DataType::INT32, DataType::UINT32, DataType::FP32, DataType::INT64, DataType::UINT64}, args, kwargs);
    });

REGISTER_OP("simt.atomic_exch")
    .set_op_category("SimtOp")
    .set_description("Atomically exchange one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Replacement scalar")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType(
            "simt.atomic_exch", 1, {DataType::INT32, DataType::UINT32, DataType::FP32},
            {DataType::INT32, DataType::UINT32, DataType::FP32, DataType::INT64, DataType::UINT64}, args, kwargs);
    });

REGISTER_OP("simt.atomic_max")
    .set_op_category("SimtOp")
    .set_description("Atomically maximize one UB Tile or GM Tensor element; FP16/BF16 return no value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Candidate scalar")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_max", 1,
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32},
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32,
                                     DataType::INT64, DataType::UINT64},
                                    args, kwargs, {DataType::FP16, DataType::BF16});
    });

REGISTER_OP("simt.atomic_min")
    .set_op_category("SimtOp")
    .set_description("Atomically minimize one UB Tile or GM Tensor element; FP16/BF16 return no value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Candidate scalar")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_min", 1,
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32},
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32,
                                     DataType::INT64, DataType::UINT64},
                                    args, kwargs, {DataType::FP16, DataType::BF16});
    });

REGISTER_OP("simt.atomic_inc")
    .set_op_category("SimtOp")
    .set_description("Atomically increment one wrapping counter element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("limit", "Inclusive wrap limit")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_inc", 1, {DataType::UINT32}, {DataType::UINT32, DataType::UINT64},
                                    args, kwargs);
    });

REGISTER_OP("simt.atomic_dec")
    .set_op_category("SimtOp")
    .set_description("Atomically decrement one wrapping counter element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("limit", "Inclusive wrap limit")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_dec", 1, {DataType::UINT32}, {DataType::UINT32, DataType::UINT64},
                                    args, kwargs);
    });

REGISTER_OP("simt.atomic_cas")
    .set_op_category("SimtOp")
    .set_description("Atomically compare and exchange one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("compare", "Expected scalar")
    .add_argument("value", "Replacement scalar")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType(
            "simt.atomic_cas", 2, {DataType::INT32, DataType::UINT32, DataType::FP32},
            {DataType::INT32, DataType::UINT32, DataType::FP32, DataType::INT64, DataType::UINT64}, args, kwargs);
    });

REGISTER_OP("simt.atomic_and")
    .set_op_category("SimtOp")
    .set_description("Atomically apply bitwise AND to one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Scalar bit mask")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_and", 1, {DataType::INT32, DataType::UINT32},
                                    {DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64}, args,
                                    kwargs);
    });

REGISTER_OP("simt.atomic_or")
    .set_op_category("SimtOp")
    .set_description("Atomically apply bitwise OR to one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Scalar bit mask")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_or", 1, {DataType::INT32, DataType::UINT32},
                                    {DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64}, args,
                                    kwargs);
    });

REGISTER_OP("simt.atomic_xor")
    .set_op_category("SimtOp")
    .set_description("Atomically apply bitwise XOR to one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Scalar bit mask")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_xor", 1, {DataType::INT32, DataType::UINT32},
                                    {DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64}, args,
                                    kwargs);
    });

REGISTER_OP("simt.launch")
    .set_op_category("SimtOp")
    .set_description("Launch a SIMT vector function from an AIV kernel")
    .add_argument("threads_x", "Compile-time X dimension")
    .add_argument("threads_y", "Compile-time Y dimension")
    .add_argument("threads_z", "Compile-time Z dimension")
    .set_attr<std::string>("callee")
    .set_attr<int>("max_threads")
    .f_deduce_type(DeduceSimtLaunchType);

} // namespace ir
} // namespace pypto
