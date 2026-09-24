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
 * @file ptr_ops.cpp
 * \brief Pointer operations for ptoas IR scene (ptr.addptr, ptr.make_tensor)
 *
 * These ops emit PTO MLIR instructions (pto.addptr, pto.make_tensor_view)
 */

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceAddPtrType([[maybe_unused]] const std::vector<ExprPtr>& args,
                         [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    // ptr.addptr: Advance a pointer by an integer offset
    // Args: (ptr, offset)
    // Returns: same PtrType as input (pointer bumped but same element dtype)
    CHECK(args.size() == 0x2) << "ptr.addptr requires exactly 2 arguments (ptr, offset), but got " << args.size();

    // First argument must be PtrType
    auto ptr_type = As<PtrType>(args[0]->GetType());
    CHECK(ptr_type) << "ptr.addptr requires first argument to be a PtrType, but got " << args[0]->GetType()->TypeName()
                    << ". Use pl.Ptr[dtype] to annotate pointer parameters.";

    // Second argument must be ScalarType with integer or index dtype
    auto offset_type = As<ScalarType>(args[1]->GetType());
    CHECK(offset_type) << "ptr.addptr requires second argument (offset) to be a ScalarType, but got "
                       << args[1]->GetType()->TypeName();
    CHECK(offset_type->dtype_.IsInt() || offset_type->dtype_ == DataType(DataType::INDEX))
        << "ptr.addptr offset must have integer or index dtype, but got " << offset_type->dtype_.ToString();

    // Return the same PtrType (pointer is advanced but still points to same element type),
    // with base_ptr/offset annotations for codegen indirect-select support.
    ExprPtr new_base_ptr;
    ExprPtr new_offset;

    if (ptr_type->base_ptr.has_value()) {
        // Chained addptr: propagate base from the input ptr, fold offsets if possible.
        new_base_ptr = *ptr_type->base_ptr;
        if (auto c1 = As<ConstInt>(*ptr_type->offset)) {
            if (auto c2 = As<ConstInt>(args[1])) {
                new_offset = std::make_shared<ConstInt>(c1->value_ + c2->value_, DataType(DataType::INDEX),
                                                        args[1]->span_);
            } else {
                new_offset = std::make_shared<Add>(*ptr_type->offset, args[1], DataType(DataType::INDEX),
                                                   args[1]->span_);
            }
        } else {
            new_offset = std::make_shared<Add>(*ptr_type->offset, args[1], DataType(DataType::INDEX), args[1]->span_);
        }
    } else {
        // Direct addptr on a function parameter — record base and offset directly.
        new_base_ptr = args[0];
        new_offset = args[1];
    }

    return std::make_shared<PtrType>(ptr_type->dtype_, new_base_ptr, new_offset);
}

REGISTER_OP("ptr.addptr")
    .set_op_category("PtrOp")
    .set_description("Advance a pointer by an integer offset (emits pto.addptr)")
    .add_argument("ptr", "Input raw pointer (PtrType)")
    .add_argument("offset", "Integer byte offset (ScalarType with integer/index dtype)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceAddPtrType(args, kwargs);
    });

TypePtr DeduceMakePtrType([[maybe_unused]] const std::vector<ExprPtr>& args,
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    // ptr.make_ptr: Reinterpret a pointer as a (usually different) element dtype, or extract
    // a raw pointer from a tensor. Args: (ptr_or_tensor); kwarg: dtype.
    // Returns: a PtrType with the new element dtype, reusing the same underlying address.
    CHECK(args.size() == 1) << "ptr.make_ptr requires exactly 1 argument (ptr), but got " << args.size();

    auto ptr_type = As<PtrType>(args[0]->GetType());
    auto tensor_type = As<TensorType>(args[0]->GetType());
    CHECK(ptr_type || tensor_type)
        << "ptr.make_ptr requires first argument to be a PtrType or TensorType, but got "
        << args[0]->GetType()->TypeName()
        << ". Use pl.Ptr[dtype] for pointer params or pl.Tensor[[shape], dtype] for tensor params.";

    // The new element dtype: explicit 'dtype' kwarg if given, otherwise keep the source dtype
    // (an identity reinterpret / address extraction).
    DataType source_dtype = ptr_type ? ptr_type->dtype_ : tensor_type->dtype_;
    DataType result_dtype = GetOpKwarg<DataType>(kwargs, "dtype", std::optional<DataType>(source_dtype));

    // Preserve the base_ptr/offset codegen annotations from a PtrType source so that a
    // make_ptr derived from an addptr chain still carries its indirect-select metadata.
    if (ptr_type && ptr_type->base_ptr.has_value() && ptr_type->offset.has_value()) {
        return std::make_shared<PtrType>(result_dtype, *ptr_type->base_ptr, *ptr_type->offset);
    }
    return std::make_shared<PtrType>(result_dtype);
}

REGISTER_OP("ptr.make_ptr")
    .set_op_category("PtrOp")
    .set_description("Reinterpret a pointer as a different element dtype (emits a pointer cast)")
    .add_argument("ptr", "Input raw pointer (PtrType)")
    .set_attr<DataType>("dtype")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceMakePtrType(args, kwargs);
    });

TypePtr DeduceMakeTensorType([[maybe_unused]] const std::vector<ExprPtr>& args,
                             [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    // ptr.make_tensor: Create a tensor view from a pointer (or an existing tensor) with
    // shape and optional strides. An empty stride tuple means an implicit contiguous
    // row-major layout. Args: (ptr_or_tensor, shape_tuple, stride_tuple)
    CHECK(args.size() == 0x3) << "ptr.make_tensor requires exactly 3 arguments (ptr, shape, stride), but got "
                              << args.size();

    // First argument is either:
    //   - a PtrType: a raw pointer to typed global memory, or
    //   - a TensorType: re-view an existing tensor, reusing its underlying data pointer with a
    //     new shape/stride (and optionally a new dtype). args[0] itself is stored as the view's
    //     source expr; the CCE codegen resolves the tensor's base pointer from it.
    DataType source_dtype;
    if (auto ptr_type = As<PtrType>(args[0]->GetType())) {
        source_dtype = ptr_type->dtype_;
    } else if (auto src_tensor_type = As<TensorType>(args[0]->GetType())) {
        source_dtype = src_tensor_type->dtype_;
    } else {
        CHECK(false) << "ptr.make_tensor requires first argument to be a PtrType or TensorType, but got "
                     << args[0]->GetType()->TypeName()
                     << ". Use pl.Ptr[dtype] to annotate pointer parameters, or pass an existing pl.Tensor.";
    }

    // Second argument must be MakeTuple (shape)
    auto shape_tuple = As<MakeTuple>(args[1]);
    CHECK(shape_tuple) << "ptr.make_tensor requires shape to be a MakeTuple";

    // Third argument must be MakeTuple (stride); an empty tuple means implicit contiguous stride.
    auto stride_tuple = As<MakeTuple>(args[2]);
    CHECK(stride_tuple) << "ptr.make_tensor requires stride to be a MakeTuple";

    const auto& stride = stride_tuple->elements_;
    if (!stride.empty()) {
        CHECK(shape_tuple->elements_.size() == stride.size())
            << "ptr.make_tensor shape rank (" << shape_tuple->elements_.size() << ") must match stride rank ("
            << stride.size() << ")";
    }

    // Element dtype: use the explicit 'dtype' kwarg if provided, otherwise derive it from the
    // source pointer/tensor's element type. This lets callers reinterpret a raw byte pointer
    // (e.g. uint8) as a typed view (e.g. fp16) without baking the dtype into the pointer parameter.
    DataType result_dtype = GetOpKwarg<DataType>(kwargs, "dtype", std::optional<DataType>(source_dtype));

    TensorView tv(stride, TensorLayout::ND, args[0]);
    return std::make_shared<TensorType>(shape_tuple->elements_, result_dtype, std::nullopt, tv);
}

REGISTER_OP("ptr.make_tensor")
    .set_op_category("PtrOp")
    .set_description("Create a tensor view from a pointer or an existing tensor with shape and optional strides"
                     " (an empty stride means contiguous row-major layout; emits pto.make_tensor_view)")
    .add_argument("ptr", "Input raw pointer (PtrType) or source tensor (TensorType)")
    .add_argument("shape", "New shape dimensions (MakeTuple of ConstInt)")
    .add_argument("stride", "Stride per dimension (MakeTuple of ConstInt or Var; empty means contiguous)")
    .set_attr<DataType>("dtype")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceMakeTensorType(args, kwargs);
    });

} // namespace ir
} // namespace pypto
