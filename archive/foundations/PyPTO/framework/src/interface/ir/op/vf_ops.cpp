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
 * @file vf_ops.cpp
 * \brief VF API IR op registration.
 *
 * Registers VF ops with type deduction for the direct VF instruction path.
 * These ops are part of the merged shared IR surface and remain consumed by
 * the block VF frontend/backend path.
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceVFUnknownType([[maybe_unused]] const std::vector<ExprPtr>& args,
                            [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    return GetUnknownType();
}

TypePtr DeduceVFScalarType([[maybe_unused]] const std::vector<ExprPtr>& args,
                           [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    for (const auto& [key, value] : kwargs) {
        if (key == "dtype") {
            return std::make_shared<ScalarType>(AnyCast<DataType>(value, "kwarg: dtype"));
        }
    }
    return std::make_shared<ScalarType>(DataType::FP32);
}

TypePtr DeduceVFMaskType([[maybe_unused]] const std::vector<ExprPtr>& args,
                         const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    for (const auto& [key, value] : kwargs) {
        if (key == "dtype") {
            return std::make_shared<ScalarType>(AnyCast<DataType>(value, "kwarg: dtype"));
        }
    }
    return std::make_shared<ScalarType>(DataType::UINT16);
}

TypePtr DeduceVFFromDstArg([[maybe_unused]] const std::vector<ExprPtr>& args,
                           [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    if (!args.empty()) {
        return args[0]->GetType();
    }
    return std::make_shared<ScalarType>(DataType::FP32);
}
} // namespace

// Register declaration
REGISTER_OP("vf.reg_tensor")
    .set_op_category("VFOp")
    .set_description("Declare a vector register")
    .no_argument()
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFScalarType);

REGISTER_OP("vf.mask_reg")
    .set_op_category("VFOp")
    .set_description("Declare a mask register (without initialization)")
    .no_argument()
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFMaskType);

// Mask
REGISTER_OP("vf.create_mask")
    .set_op_category("VFOp")
    .set_description("Declare and initialize a mask register")
    .no_argument()
    .set_attr<int>("pattern")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFMaskType);

// Initialization
REGISTER_OP("vf.full")
    .set_op_category("VFOp")
    .set_description("Scalar broadcast to register")
    .add_argument("dst", "Destination register")
    .add_argument("scalar", "Scalar value to broadcast")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .set_attr<int>("pos")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFFromDstArg);

// Data transfer
REGISTER_OP("vf.load_align")
    .set_op_category("VFOp")
    .set_description("Unified vlds/plds load (MaskReg dst via dist kwarg, RegTensor dst via dist kwarg)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source UB pointer")
    .add_argument("offset", "Element offset or post-update stride")
    .set_attr<int>("data_copy_mode")
    .set_attr<int>("block_stride")
    .set_attr<int>("repeat_stride")
    .set_attr<bool>("post_update")
    .set_attr<int>("dist")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.store_align")
    .set_op_category("VFOp")
    .set_description(
        "Store aligned data from register to UB (MaskReg src via dist kwarg → psts/pst, RegTensor src → vsts/vst)")
    .add_argument("dst", "Destination UB pointer")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("dist")
    .set_attr<int>("data_copy_mode")
    .set_attr<int>("block_stride")
    .set_attr<int>("repeat_stride")
    .set_attr<bool>("post_update")
    .f_deduce_type(DeduceVFUnknownType);

// Compute
REGISTER_OP("vf.max")
    .set_op_category("VFOp")
    .set_description("Vector maximum")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.add")
    .set_op_category("VFOp")
    .set_description("Vector addition")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.sub")
    .set_op_category("VFOp")
    .set_description("Vector subtraction")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.and_")
    .set_op_category("VFOp")
    .set_description("Bitwise AND")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.xor")
    .set_op_category("VFOp")
    .set_description("Bitwise XOR")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.or_")
    .set_op_category("VFOp")
    .set_description("Bitwise OR")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

// Unified reduction: supports mode=SUM/MAX/MIN (new) and reduce_type=ADD/MAX (legacy)
REGISTER_OP("vf.reduce_sum")
    .set_op_category("VFOp")
    .set_description("Sum reduction across all lanes (vcadd/vcgadd)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<bool>("datablock")
    .set_attr<int>("merge_mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.reduce_max")
    .set_op_category("VFOp")
    .set_description("Max reduction across all lanes (vcmax/vcgmax)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<bool>("datablock")
    .set_attr<int>("merge_mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.reduce_min")
    .set_op_category("VFOp")
    .set_description("Min reduction across all lanes (vcmin/vcgmin)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<bool>("datablock")
    .set_attr<int>("merge_mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.mul")
    .set_op_category("VFOp")
    .set_description("Vector multiplication")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.mul_add_dst")
    .set_op_category("VFOp")
    .set_description("Fused multiply-add (FMA): dst = src0 * src1 + dst (hardware vmadd)")
    .add_argument("dst", "Destination/accumulator register (read+write)")
    .add_argument("src0", "Source register 0 (multiplicand)")
    .add_argument("src1", "Source register 1 (multiplicand)")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.div")
    .set_op_category("VFOp")
    .set_description("Vector division")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.muls")
    .set_op_category("VFOp")
    .set_description("Scalar multiplication")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar multiplier")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.ln")
    .set_op_category("VFOp")
    .set_description("Natural logarithm (vln instruction)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.log")
    .set_op_category("VFOp")
    .set_description("Natural logarithm (alias of Ln, same vln instruction on A5)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.min")
    .set_op_category("VFOp")
    .set_description("Vector minimum")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.exp")
    .set_op_category("VFOp")
    .set_description("Exponential function")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.abs")
    .set_op_category("VFOp")
    .set_description("Absolute value")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.not_")
    .set_op_category("VFOp")
    .set_description("Bitwise NOT")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.sqrt")
    .set_op_category("VFOp")
    .set_description("Square root")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.relu")
    .set_op_category("VFOp")
    .set_description("ReLU activation")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.neg")
    .set_op_category("VFOp")
    .set_description("Negate")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.adds")
    .set_op_category("VFOp")
    .set_description("Scalar addition")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar addend")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.subs")
    .set_op_category("VFOp")
    .set_description("Scalar subtraction")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar subtrahend")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.mins")
    .set_op_category("VFOp")
    .set_description("Scalar minimum")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar value")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.maxs")
    .set_op_category("VFOp")
    .set_description("Scalar maximum")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar value")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.leaky_relu")
    .set_op_category("VFOp")
    .set_description("Leaky ReLU activation")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("alpha", "Negative slope scalar")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.interleave")
    .set_op_category("VFOp")
    .set_description("Interleave two registers (vintlv for RegTensor, pintlv for MaskReg)")
    .add_argument("dst0", "Destination register 0")
    .add_argument("dst1", "Destination register 1")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.pair_reduce_sum")
    .set_op_category("VFOp")
    .set_description("Pairwise reduction sum")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.abs_sub")
    .set_op_category("VFOp")
    .set_description("Absolute difference |src0 - src1|")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.axpy")
    .set_op_category("VFOp")
    .set_description("Accumulate: dst = src * scalar + dst")
    .add_argument("dst", "Destination/accumulator register (read+write)")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar multiplier")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.bit_cast")
    .set_op_category("VFOp")
    .set_description("Bitwise type reinterpretation (no value conversion)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFScalarType);

REGISTER_OP("vf.mul_dst_add")
    .set_op_category("VFOp")
    .set_description("Multiply-dst-add: dst = dst + src0 * src1 (vmadd)")
    .add_argument("dst", "Destination/accumulator register (read+write)")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.pack")
    .set_op_category("VFOp")
    .set_description("Pack/narrow data type (e.g. u16->u8). vpack for RegTensor, ppack for MaskReg.")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .set_attr<int>("part")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.unpack")
    .set_op_category("VFOp")
    .set_description("Unpack/widen data type (e.g. u8->u16). vunpack for RegTensor, punpack for MaskReg.")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .set_attr<int>("part")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.prelu")
    .set_op_category("VFOp")
    .set_description("Parametric ReLU with per-element slope")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("slope", "Slope register (per-element)")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.shift_left")
    .set_op_category("VFOp")
    .set_description("Vector-vector shift left")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("shift", "Shift amount register (signed)")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.shift_right")
    .set_op_category("VFOp")
    .set_description("Vector-vector shift right")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("shift", "Shift amount register (signed)")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.mull")
    .set_op_category("VFOp")
    .set_description("Long multiply: 32x32->64, output split into lo/hi registers")
    .add_argument("dst_lo", "Low 32 bits of product")
    .add_argument("dst_hi", "High 32 bits of product")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.addc")
    .set_op_category("VFOp")
    .set_description("Add with carry: carry_out, dst = src0 + src1 + carry_in")
    .add_argument("carry_out", "Output carry flag register")
    .add_argument("dst", "Destination register (sum)")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("carry_in", "Input carry flag register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.subc")
    .set_op_category("VFOp")
    .set_description("Subtract with borrow: borrow_out, dst = src0 - src1 - borrow_in")
    .add_argument("borrow_out", "Output borrow flag register")
    .add_argument("dst", "Destination register (difference)")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("borrow_in", "Input borrow flag register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.exp_sub")
    .set_op_category("VFOp")
    .set_description("Fused exp(src - max)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("max", "Max register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("layout")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.astype")
    .set_op_category("VFOp")
    .set_description("Type conversion")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("layout")
    .set_attr<int>("round_mode")
    .set_attr<int>("saturate")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.de_interleave")
    .set_op_category("VFOp")
    .set_description("De-interleave (vdintlv for RegTensor, pdintlv for MaskReg)")
    .add_argument("dst0", "Destination register 0")
    .add_argument("dst1", "Destination register 1")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.select")
    .set_op_category("VFOp")
    .set_description("Conditional select")
    .add_argument("dst", "Destination register")
    .add_argument("src_true", "True branch register")
    .add_argument("src_false", "False branch register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.update_mask")
    .set_op_category("VFOp")
    .set_description("Update mask with scalar value")
    .add_argument("scalar", "Scalar value for mask")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFMaskType);

REGISTER_OP("vf.mem_bar")
    .set_op_category("VFOp")
    .set_description("Memory barrier")
    .no_argument()
    .f_deduce_type(DeduceVFUnknownType);

// TopK VF ops
REGISTER_OP("vf.histograms")
    .set_op_category("VFOp")
    .set_description("Histogram accumulation")
    .add_argument("dst", "Destination histogram register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("bin_type")
    .set_attr<int>("hist_type")
    .f_deduce_type(DeduceVFFromDstArg);
REGISTER_OP("vf.eq")
    .set_op_category("VFOp")
    .set_description("Equality compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .set_attr<DataType>("cmp_dtype")
    .f_deduce_type(DeduceVFMaskType);

REGISTER_OP("vf.ne")
    .set_op_category("VFOp")
    .set_description("Not-equal compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .set_attr<DataType>("cmp_dtype")
    .f_deduce_type(DeduceVFMaskType);

REGISTER_OP("vf.lt")
    .set_op_category("VFOp")
    .set_description("Less-than compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .set_attr<DataType>("cmp_dtype")
    .f_deduce_type(DeduceVFMaskType);

REGISTER_OP("vf.gt")
    .set_op_category("VFOp")
    .set_description("Greater-than compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .set_attr<DataType>("cmp_dtype")
    .f_deduce_type(DeduceVFMaskType);

REGISTER_OP("vf.le")
    .set_op_category("VFOp")
    .set_description("Less-or-equal compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .set_attr<DataType>("cmp_dtype")
    .f_deduce_type(DeduceVFMaskType);

REGISTER_OP("vf.ge")
    .set_op_category("VFOp")
    .set_description("Greater-or-equal compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .set_attr<DataType>("cmp_dtype")
    .f_deduce_type(DeduceVFMaskType);
REGISTER_OP("vf.squeeze")

    .set_op_category("VFOp")
    .set_description("Squeeze mask to indices")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("gather_mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.arange")
    .set_op_category("VFOp")
    .set_description("Generate index sequence (index_order=INCREASE_ORDER: vci INC_ORDER, DECREASE_ORDER: DEC_ORDER)")
    .add_argument("dst", "Destination register")
    .set_attr<int>("index_order")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.gather")
    .set_op_category("VFOp")
    .set_description("Gather elements by indices. Two forms: "
                     "(1) UB-to-Reg: gather(src_ub, indices, mask[, data_copy_mode]). "
                     "b8 sources are zero-extended to b16. "
                     "b16 sources support both uint16 and uint32 index. "
                     "(2) Reg-to-Reg: gather(src_reg, indices). "
                     "Source and destination must have the same type, supports INT8->INT8. "
                     "Backend dispatches based on src argument type (Tile vs RegTensor).")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source UB pointer or source register")
    .add_argument("indices", "Index register")
    .add_argument("mask", "Mask register (required for UB form, omitted for Reg form)")
    .set_attr<int>("data_copy_mode")
    .f_deduce_type(DeduceVFFromDstArg);

// Two calling conventions (codegen branches on arg count):
//   store_unalign:      3 args [dst, src, align_reg] -> vstur (strideless)
//                       4 args [dst, vreg, ureg, stride] -> vstus (strided)
//   store_unalign_post: 2 args [dst, align_reg] -> vstar (strideless)
//                       3 args [dst, ureg, stride] -> vstas (strided)
// Declared at the strideless minimum; the extra strided arg is passed through.
REGISTER_OP("vf.store_unalign")
    .set_op_category("VFOp")
    .set_description("Store unaligned data with post update")
    .add_argument("dst_ptr", "Destination UB pointer")
    .add_argument("src", "Source register")
    .add_argument("align_reg", "Alignment register")
    .set_attr<bool>("post_update")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.store_unalign_post")
    .set_op_category("VFOp")
    .set_description("Complete unaligned store")
    .add_argument("dst_ptr", "Destination UB pointer")
    .add_argument("align_reg", "Alignment register")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.unalign_reg_for_store")
    .set_op_category("VFOp")
    .set_description("Declare unaligned register for store")
    .no_argument()
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.clear_spr")
    .set_op_category("VFOp")
    .set_description("Clear special purpose register AR")
    .no_argument()
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.load_unalign_init")
    .set_op_category("VFOp")
    .set_description("Declare unaligned register for load")
    .no_argument()
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.load_unalign_pre")
    .set_op_category("VFOp")
    .set_description("Setup unaligned load (vldas)")
    .add_argument("ureg", "UnalignRegForLoad register")
    .add_argument("src_ptr", "Source UB pointer")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.load_unalign")
    .set_op_category("VFOp")
    .set_description("Load unaligned data from UB to register (vldus, optional stride for POST_UPDATE)")
    .add_argument("dst", "Destination register")
    .add_argument("ureg", "UnalignRegForLoad register")
    .add_argument("src_ptr", "Source UB pointer")
    .add_argument("stride", "Optional post-update stride in bytes")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.scatter")
    .set_op_category("VFOp")
    .set_description("Scatter store by indices (vscatter)")
    .add_argument("base_ptr", "Base UB pointer")
    .add_argument("src", "Source register")
    .add_argument("index", "Index register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.unsqueeze")
    .set_op_category("VFOp")
    .set_description("Unsqueeze mask bits to register (vusqz)")
    .add_argument("dst", "Destination register")
    .add_argument("mask", "Mask register to unsqueeze")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.truncate")
    .set_op_category("VFOp")
    .set_description("Truncate to integer (vtrc with ROUND_Z, alias of Round with round_mode=TRUNC)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.mask_gen_with_reg_tensor")
    .set_op_category("VFOp")
    .set_description("Generate MaskReg from RegTensor bit at offset (movvp)")
    .add_argument("src", "Source register (uint16/uint32)")
    .set_attr<int>("offset")
    .f_deduce_type(DeduceVFMaskType);

REGISTER_OP("vf.get_mask_spr")
    .set_op_category("VFOp")
    .set_description("Get mask from special purpose register (movp_b32/b16 via kwarg width)")
    .no_argument()
    .set_attr<int>("width")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.log2")
    .set_op_category("VFOp")
    .set_description("Base-2 logarithm: dst = log2(src) = ln(src) * 1/ln(2)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.log10")
    .set_op_category("VFOp")
    .set_description("Base-10 logarithm: dst = log10(src) = ln(src) * 1/ln(10)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.muls_cast")
    .set_op_category("VFOp")
    .set_description("Multiply by scalar then cast to half: dst(fp16) = cast(src(fp32) * scalar)")
    .add_argument("dst", "Destination register (fp16)")
    .add_argument("src", "Source register (fp32)")
    .add_argument("scalar", "Scalar multiplier")
    .add_argument("mask", "Mask register")
    .set_attr<int>("layout")
    .f_deduce_type(DeduceVFFromDstArg);

// mask_and/or/xor/not/mov/sel/pack/unpack/interleave/deinterleave have been
// merged into their unified counterparts (and_, or_, ..., move, select,
// pack, unpack, interleave, de_interleave). The Python parser redirects
// vf.mask_* calls to the unified IR op via _VF_OP_NAME_MAP.

// vf.mask_load/mask_store/mask_store_unalign IR registrations removed —
// the parser redirects these to vf.load_align/vf.store_align/vf.store_unalign
// which dispatch via IsMaskRegVar (matching AscendC's function-overloading model).

REGISTER_OP("vf.load")
    .set_op_category("VFOp")
    .set_description("Unified unaligned load (vldas+vldus, matches AscendC Load)")
    .add_argument("dst", "Destination register")
    .add_argument("src_ptr", "Source UB pointer")
    .add_argument("stride", "Post-update stride (optional, triggers POST_UPDATE mode)")
    .set_attr<bool>("post_update")
    .set_attr<int>("repeat_stride")
    .set_attr<int>("count")
    .f_deduce_type(DeduceVFFromDstArg);

REGISTER_OP("vf.store")
    .set_op_category("VFOp")
    .set_description("Unified unaligned store (vstus+vstas, matches AscendC Store)")
    .add_argument("dst_ptr", "Destination UB pointer")
    .add_argument("src", "Source register")
    .add_argument("count", "Element count (optional, defaults to 256/elem_bytes)")
    .set_attr<bool>("post_update")
    .set_attr<int>("repeat_stride")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.create_addr_reg")
    .set_op_category("VFOp")
    .set_description("Create an address offset register for aligned load/store")
    .add_argument("index0", "Loop axis 0 index")
    .add_argument("stride0", "Loop axis 0 stride (element count)")
    .add_argument("index1", "Loop axis 1 index (optional)")
    .add_argument("stride1", "Loop axis 1 stride (optional)")
    .add_argument("index2", "Loop axis 2 index (optional)")
    .add_argument("stride2", "Loop axis 2 stride (optional)")
    .add_argument("index3", "Loop axis 3 index (optional)")
    .add_argument("stride3", "Loop axis 3 stride (optional)")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.move")
    .set_op_category("VFOp")
    .set_description("Move/copy register elements (RegTensor or MaskReg). "
                     "RegTensor: vmov with MODE_MERGING (masked) or vmov (unmasked). "
                     "MaskReg: pmov (masked) or pmov (unmasked). "
                     "Unified op: dst register kind is inferred from source operands.")
    .add_argument("dst", "Destination register (RegTensor or MaskReg)")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register (optional)")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFFromDstArg);

} // namespace ir
} // namespace pypto
