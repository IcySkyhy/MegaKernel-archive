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
 * @file backend_950_cce_vf_ops.cpp
 * \brief CCE backend op registration for VF API operations (A5 target).
 *
 * VF ops directly emit CCE intrinsics (vlds, vmax, vdup, etc.) without
 * going through the PTO-ISA intermediate layer.
 */

#include <string>
#include <unordered_map>

#include "backend/backend_cce.h"
#include "backend/common/backend.h"
#include "codegen/cce/cce_codegen.h"
#include "codegen/codegen_base.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/pipe.h"
#include "ir/op_attr_types.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using ir::DataType;

static std::string VFEnumValueName(const char* full_name)
{
    const char* sep = std::strrchr(full_name, ':');
    return sep ? std::string(sep + 1) : std::string(full_name);
}

// Format a DataType for log messages in the frontend DT_XXX style.
static std::string DTypeStr(const DataType& dt) { return "DT_" + ir::DTypeToString(dt); }

// Returns true when the dst argument (args_[0], or args_[0]/args_[1] for 2-dst
// ops) is a MaskReg variable. Used by unified emitters to dispatch between
// v* (RegTensor) and p* (MaskReg) CCE intrinsics.
static bool IsDstMaskReg(const ir::CallPtr& op, codegen::CCECodegen& codegen, size_t idx = 0)
{
    if (idx >= op->args_.size())
        return false;
    auto dst_var = ir::As<ir::Var>(op->args_[idx]);
    if (dst_var) {
        return codegen.IsMaskRegVar(codegen.GetVarName(dst_var));
    }
    return false;
}

// ============================================================================
// RegTensor declaration
// ============================================================================

static std::string EmitVFRegTensor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto dtype = op->GetKwarg<DataType>("dtype");
    std::string reg_name = codegen.GetCurrentResultTarget();
    std::string decl;
    if (dtype == DataType::INT4) {
        decl = "vector_s4x2 " + reg_name + ";";
    } else if (dtype == DataType::UINT4) {
        decl = "vector_u4x2 " + reg_name + ";";
    } else {
        decl = "RegTensor<" + dtype.ToCTypeString() + "> " + reg_name + ";";
    }
    codegen.HoistRegTensorDecl(decl);
    codegen.RegisterRegTensorVar(reg_name);
    return "";
}

// ============================================================================
// MaskReg declaration (no initialization — unlike create_mask which emits pset)
// ============================================================================

static std::string EmitVFMaskReg(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();
    codegen.HoistRegTensorDecl("MaskReg " + reg_name + ";");
    codegen.RegisterMaskRegVar(reg_name);
    return "";
}

// Best-effort dtype extraction from an Expr's deduced type. Handles both
// ScalarType (RegTensor outputs) and ShapedType (Tile / Tensor expressions).
static DataType GetExprDtype(const ir::ExprPtr& expr, DataType fallback = DataType::UINT32)
{
    auto type = expr->GetType();
    if (auto st = ir::As<ir::ScalarType>(type))
        return st->dtype_;
    if (auto sh = ir::As<ir::ShapedType>(type))
        return sh->dtype_;
    return fallback;
}

// For ops that only support ZEROING: return "MODE_ZEROING", reject MERGING.
static std::string VFZeroingOnly(const ir::CallPtr& op, const std::string& op_name)
{
    if (op->HasKwarg("mode")) {
        auto mode = static_cast<ir::MergeMode>(op->GetKwarg<int>("mode"));
        CHECK(mode == ir::MergeMode::ZEROING) << op_name << " only supports ZEROING mode on current device";
    }
    return "MODE_ZEROING";
}

// For ops that support both ZEROING and MERGING: default ZEROING, use user value if provided.
static std::string VFAnyMode(const ir::CallPtr& op)
{
    if (!op->HasKwarg("mode")) {
        return "MODE_ZEROING";
    }
    auto mode = static_cast<ir::MergeMode>(op->GetKwarg<int>("mode"));
    return mode == ir::MergeMode::MERGING ? "MODE_MERGING" : "MODE_ZEROING";
}

// Check if a DataType is a b8-width type (8-bit storage).
// Includes INT8, UINT8, BOOL, and all FP8 types (FP8E4M3FN, FP8E5M2, HF8).
// FP4 types (FP4E2M1, FP4E1M2, FP4) are b4 but stored as b8 (packed 2-per-byte),
// so they are also treated as b8 for load/store mode selection.
static bool IsB8Type(DataType dt) { return dt.GetBit() <= 8; }

// Check if a DataType is a b16-width type (16-bit storage).
static bool IsB16Type(DataType dt) { return dt.GetBit() == 16; }

// Check if a DataType lacks a direct vdup/vlds/vsts intrinsic overload and
// must be reinterpreted as uint8_t. The bisheng __VF_VDUP/__VF_VLDS/__VF_VSTS
// macros only instantiate overloads for u8/s8/u16/s16/f16/u32/s32/f32/bf16/
// f8e4m3/f8e5m2/f8e8m0/f4e2m1x2/f4e1m2x2 and b64 types. FP4/HF4 (b4 packed as
// b8) and HF8 (b8) have no overload, but share the same physical b8 register
// layout as u8.
static bool NeedsB8Reinterpret(DataType dt)
{
    return dt == DataType::FP4 || dt == DataType::FP4E2M1 || dt == DataType::FP4E1M2 || dt == DataType::HF4 ||
           dt == DataType::HF8 || dt == DataType::INT4 || dt == DataType::UINT4;
}

// Map a DataType to the correct C pointer type for __ubuf__ load/store.
// VF load/store intrinsics (vlds/vsts/vld/vst/vsldb/vsstb) accept both
// signed and unsigned pointer types, so we use the native C type to match
// the RegTensor element type and avoid type-mismatch errors.
// B64 types (INT64/UINT64) are stored as pairs of 32-bit halves, so the
// pointer type is the 32-bit half type for non-B64-overload paths.
// (B64 vlds/vsts overloads override the pointer type via GetB64PtrType.)
// Types lacking a direct bisheng overload (FP4/HF4/HF8/INT4/UINT4) are
// reinterpreted as uint8_t.
static std::string DtypeToPtrType(DataType dt)
{
    if (NeedsB8Reinterpret(dt))
        return "uint8_t";
    if (dt == DataType::INT64)
        return "int32_t";
    if (dt == DataType::UINT64)
        return "uint32_t";
    return dt.ToCTypeString();
}

// Return the native 64-bit pointer type for B64 vlds/vsts overloads
// (__VF_VLDS_B64/__VF_VSTS_B64), which expect __ubuf__ int64_t*/uint64_t*.
static std::string GetB64PtrType(DataType dt)
{
    if (dt == DataType::INT64)
        return "int64_t";
    if (dt == DataType::UINT64)
        return "uint64_t";
    return dt.ToCTypeString();
}

// Return the (RegTensor<uint8_t>&) cast prefix for types that need B8 reinterpret.
static std::string GetB8Cast(DataType dt) { return NeedsB8Reinterpret(dt) ? "(RegTensor<uint8_t>&)" : ""; }

// Return the cast prefix for B64 types (INT64/UINT64).
// B64 vlds/vsts overloads (__VF_VLDS_B64/__VF_VSTS_B64) expect vector_2xvl_s64/u64,
// but the register is declared as RegTensor<int64_t>/RegTensor<uint64_t>.
static std::string GetB64Cast(DataType dt)
{
    if (dt == DataType::INT64)
        return "(vector_2xvl_s64&)";
    if (dt == DataType::UINT64)
        return "(vector_2xvl_u64&)";
    return "";
}

// Resolve an offset argument to a C++ code string.
// If the argument is a 2-element MakeTuple [row, col], compute the linear
// offset `row * cols + col` using the tile's shape[1] (number of columns).
// Otherwise, emit the expression directly (integer offset, AddrReg, etc.).
// Returns empty string if the argument is not an offset (caller should skip).
static std::string ResolveOffsetArg(codegen::CCECodegen& codegen, const ir::ExprPtr& offset_expr,
                                    const ir::ExprPtr& tile_expr)
{
    if (auto tuple = ir::As<ir::MakeTuple>(offset_expr)) {
        if (tuple->elements_.size() == 2) {
            std::string row_str = codegen.GetExprAsCode(tuple->elements_[0]);
            std::string col_str = codegen.GetExprAsCode(tuple->elements_[1]);
            // Get cols from tile shape[1]
            std::string cols_str = "1";
            if (auto tile_type = ir::As<ir::TileType>(tile_expr->GetType())) {
                if (tile_type->shape_.size() >= 2) {
                    cols_str = codegen.GetExprAsCode(tile_type->shape_[1]);
                }
            }
            return "((" + row_str + ") * (" + cols_str + ") + (" + col_str + "))";
        }
    }
    return codegen.GetExprAsCode(offset_expr);
}

// ============================================================================
// CreateMask — declares MaskReg + emits VF init instruction
// ============================================================================

static std::string EmitVFCreateMask(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // pattern defaults to ALL, dtype defaults to FP32 — either kwarg may be omitted.
    auto pattern = op->HasKwarg("pattern") ? static_cast<ir::MaskPattern>(op->GetKwarg<int>("pattern")) :
                                             ir::MaskPattern::ALL;
    auto dtype = op->HasKwarg("dtype") ? op->GetKwarg<DataType>("dtype") : DataType::FP32;
    // MaskReg dtype must be b8/b16/b32/b64 — determines mask granularity
    // FP4 types (GetBit()==4) are b8 storage (packed 2-per-byte), treated as b8
    CHECK(IsB8Type(dtype) || dtype.GetBit() == 16 || dtype.GetBit() == 32 || dtype.GetBit() == 64)
        << "vf.create_mask dtype must be b8/b16/b32/b64, got " << DTypeStr(dtype);
    std::string reg_name = codegen.GetCurrentResultTarget();
    codegen.Emit("MaskReg " + reg_name + ";");
    codegen.RegisterMaskRegVar(reg_name);
    // Map pypto pattern enum to CCE PAT_* constant
    std::string pat;
    switch (pattern) {
        case ir::MaskPattern::ALL:
            pat = "PAT_ALL";
            break;
        case ir::MaskPattern::ALLF:
            pat = "PAT_ALLF";
            break;
        case ir::MaskPattern::VL1:
            pat = "PAT_VL1";
            break;
        case ir::MaskPattern::VL2:
            pat = "PAT_VL2";
            break;
        case ir::MaskPattern::VL3:
            pat = "PAT_VL3";
            break;
        case ir::MaskPattern::VL4:
            pat = "PAT_VL4";
            break;
        case ir::MaskPattern::VL8:
            pat = "PAT_VL8";
            break;
        case ir::MaskPattern::VL16:
            pat = "PAT_VL16";
            break;
        case ir::MaskPattern::VL32:
            pat = "PAT_VL32";
            break;
        case ir::MaskPattern::VL64:
            pat = "PAT_VL64";
            break;
        case ir::MaskPattern::VL128:
            pat = "PAT_VL128";
            break;
        case ir::MaskPattern::M3:
            pat = "PAT_M3";
            break;
        case ir::MaskPattern::M4:
            pat = "PAT_M4";
            break;
        case ir::MaskPattern::H:
            pat = "PAT_H";
            break;
        case ir::MaskPattern::Q:
            pat = "PAT_Q";
            break;
        default:
            pat = "PAT_ALL";
            break;
    }
    // Select pset instruction based on data element size (not mask type)
    // float/int32 (4 bytes) → pset_b32, half/bf16 (2 bytes) → pset_b16, int8 (1 byte) → pset_b8
    // FP8/FP4 types are b8 storage → pset_b8
    if (IsB8Type(dtype)) {
        codegen.Emit(reg_name + " = pset_b8(" + pat + ");");
    } else if (dtype.GetBit() == 32) {
        codegen.Emit(reg_name + " = pset_b32(" + pat + ");");
    } else {
        // FP16, BF16, UINT16, INT16 etc. (2 bytes)
        codegen.Emit(reg_name + " = pset_b16(" + pat + ");");
    }
    return "";
}

// ============================================================================
// Duplicate — scalar broadcast
// ============================================================================

static std::string EmitVFDuplicate(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src, (optional) mask]
    CHECK(op->args_.size() >= 2 && op->args_.size() <= 3) << "vf.full requires 2-3 args (dst, src[, mask])";
    // vdup supports b8/b16/b32/b64 element widths (bool, int, float, FP8/FP4/HF8 types)
    DataType src_dt = GetExprDtype(op->args_[1], DataType::FP32);
    CHECK((IsB8Type(src_dt) || src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64))
        << "vf.full src only supports b8/b16/b32/b64 types, got " << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src_str = codegen.GetExprAsCode(op->args_[1]);
    // Detect vector-source broadcast: either explicit pos kwarg, or src is a RegTensor variable.
    std::string pos = "";
    if (op->HasKwarg("pos")) {
        pos = VFEnumValueName(ir::EnumToString(static_cast<ir::DuplicatePos>(op->GetKwarg<int>("pos"))));
    }
    bool is_vector_src = !pos.empty();
    if (!is_vector_src) {
        auto src_var = ir::As<ir::Var>(op->args_[1]);
        if (src_var) {
            std::string src_name = codegen.GetVarName(src_var);
            is_vector_src = codegen.IsRegTensorVar(src_name);
        }
    }
    if (is_vector_src) {
        // Vector-source broadcast (Tensor mode): vdup(dst, src_vec, mask, POS_xxx, MODE)
        // pos kwarg: "LOWEST" -> POS_LOWEST, "HIGHEST" -> POS_HIGHEST
        if (pos.empty() || pos == "LOWEST")
            pos = "POS_LOWEST";
        else if (pos == "HIGHEST")
            pos = "POS_HIGHEST";
        std::string mode = VFAnyMode(op);
        // FP4/HF8/HF4/INT4/UINT4 lack a vdup overload — reinterpret as uint8_t
        std::string cast = GetB8Cast(src_dt);
        // Tensor mode always requires a mask (vdup(dstReg, srcReg, mask))
        if (op->args_.size() >= 3) {
            std::string mask = codegen.GetExprAsCode(op->args_[2]);
            codegen.Emit("vdup(" + cast + dst + ", " + cast + src_str + ", " + mask + ", " + pos + ", " + mode + ");");
        } else {
            // No mask provided — create an ALL mask inline for Tensor mode
            static int dup_mask_counter = 0;
            std::string mask_var = "__dup_mask_" + std::to_string(dup_mask_counter++);
            std::string pat = "PAT_ALL";
            std::string pset_fn = "pset_b32";
            if (IsB8Type(src_dt))
                pset_fn = "pset_b8";
            else if (src_dt.GetBit() == 16)
                pset_fn = "pset_b16";
            codegen.Emit("MaskReg " + mask_var + " = " + pset_fn + "(" + pat + ");");
            codegen.RegisterMaskRegVar(mask_var);
            codegen.Emit("vdup(" + cast + dst + ", " + cast + src_str + ", " + mask_var + ", " + pos + ", " + mode +
                         ");");
        }
    } else if (op->args_.size() >= 3) {
        // Scalar broadcast with mask: vdup(dst, scalar, preg, MODE_ZEROING/MERGING)
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        std::string mode = VFAnyMode(op);
        codegen.Emit("vdup(" + dst + ", " + src_str + ", " + mask + ", " + mode + ");");
    } else {
        // Scalar broadcast without mask: vbr(dst, scalar)
        codegen.Emit("vbr(" + dst + ", " + src_str + ");");
    }
    return "";
}

// ============================================================================
// Helper: get __ubuf__ pointer from tile or tile-flavored GetItemExpr
// ============================================================================

static std::string GetUBufPtr(codegen::CCECodegen& codegen, const ir::ExprPtr& expr,
                              const std::string& cast_type = "float", bool is_post_update = false)
{
    std::string ptr = codegen.GetOrCreateVFTilePtr(expr, is_post_update);
    std::string tile_ctype = GetExprDtype(expr, DataType::FP32).ToCTypeString();
    if (cast_type == tile_ctype)
        return ptr;
    return "(__ubuf__ " + cast_type + " *)" + ptr;
}

// ============================================================================
// LoadAlign (unified) — vlds / plds with dist & post_update kwargs
// Replaces: LoadAlign, LoadAlignMode, LoadAlignPostUpdate, LoadAlignPostupdate, LoadAlignUnpackV2
// ============================================================================

static std::string EmitVFLoadAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // 4-arg form: deinterleave load_align(dst0, dst1, ptr, offset, dist="DINTLV_Bxx")
    if (op->args_.size() == 4) {
        CHECK(!op->HasKwarg("data_copy_mode"))
            << "vf.load_align 4-arg (de-interleave) form does not support data_copy_mode";
        CHECK(!op->HasKwarg("block_stride"))
            << "vf.load_align 4-arg (de-interleave) form does not support block_stride";
        CHECK(!op->HasKwarg("repeat_stride"))
            << "vf.load_align 4-arg (de-interleave) form does not support repeat_stride";
        std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
        std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
        std::string offset_str = ResolveOffsetArg(codegen, op->args_[3], op->args_[2]);
        DataType dst_dt = GetExprDtype(op->args_[0]);
        // Pointer type follows the register's declared dtype. When it differs
        // from the tile's dtype, adjust the offset by the element-width ratio so
        // the byte address stays correct (mirrors b64→b32 stride*2).
        std::string ptr_type = DtypeToPtrType(dst_dt);
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[2], ptr_type);
        DataType tile_dt = GetExprDtype(op->args_[2], dst_dt);
        std::string cast = GetB8Cast(dst_dt);
        std::string effective_offset = offset_str;
        if (tile_dt.GetBit() != dst_dt.GetBit() && !offset_str.empty()) {
            effective_offset = "(" + offset_str + ") * " + std::to_string(tile_dt.GetBit()) + " / " +
                               std::to_string(dst_dt.GetBit());
        }
        std::string dintlv_mode;
        if (op->HasKwarg("dist")) {
            dintlv_mode = VFEnumValueName(ir::EnumToString(static_cast<ir::LoadDist>(op->GetKwarg<int>("dist"))));
        } else {
            if (IsB8Type(dst_dt))
                dintlv_mode = "DINTLV_B8";
            else if (IsB16Type(dst_dt))
                dintlv_mode = "DINTLV_B16";
            else
                dintlv_mode = "DINTLV_B32";
        }
        bool post_update = false;
        if (op->HasKwarg("post_update")) {
            post_update = op->GetKwarg<bool>("post_update");
        }
        if (post_update) {
            codegen.Emit("vlds(" + cast + dst0 + ", " + cast + dst1 + ", " + ub_ptr + ", " + effective_offset + ", " +
                         dintlv_mode + ", POST_UPDATE);");
        } else {
            codegen.Emit("vlds(" + cast + dst0 + ", " + cast + dst1 + ", " + ub_ptr + ", " + effective_offset + ", " +
                         dintlv_mode + ");");
        }
        return "";
    }
    // 2-arg form: load_align(dst, ptr) — MaskReg dst → plds, RegTensor dst → vlds
    if (op->args_.size() == 2) {
        CHECK(!op->HasKwarg("data_copy_mode")) << "vf.load_align 2-arg form does not support data_copy_mode";
        CHECK(!op->HasKwarg("block_stride")) << "vf.load_align 2-arg form does not support block_stride";
        CHECK(!op->HasKwarg("repeat_stride")) << "vf.load_align 2-arg form does not support repeat_stride";
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        DataType dst_dt = GetExprDtype(op->args_[0]);
        bool dst_is_mask = false;
        if (auto dst_v = ir::As<ir::Var>(op->args_[0])) {
            dst_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(dst_v));
        }
        if (dst_is_mask) {
            std::string mode = "NORM";
            if (op->HasKwarg("dist")) {
                mode = VFEnumValueName(ir::EnumToString(static_cast<ir::LoadDist>(op->GetKwarg<int>("dist"))));
                CHECK(mode == "NORM" || mode == "US" || mode == "DS")
                    << "vf.load_align 2-arg (MaskReg) only supports NORM/US/DS dist, got " << mode;
            }
            std::string plds_ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("plds(" + dst + ", " + plds_ptr + ", 0, " + mode + ");");
        } else {
            // RegTensor 2-arg form: vlds with hardcoded NORM, dist kwarg not supported
            if (op->HasKwarg("dist")) {
                auto dist_val = VFEnumValueName(ir::EnumToString(static_cast<ir::LoadDist>(op->GetKwarg<int>("dist"))));
                CHECK(dist_val == "NORM")
                    << "vf.load_align 2-arg (RegTensor) only supports NORM dist, got " << dist_val;
            }
            std::string ptr_type = DtypeToPtrType(dst_dt);
            std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
            std::string cast = GetB8Cast(dst_dt);
            // B64 types (INT64/UINT64) use __VF_VLDS_B64 which takes 3 args (no NORM mode)
            if (dst_dt.GetBit() == 64) {
                std::string b64_cast = GetB64Cast(dst_dt);
                std::string b64_ptr = GetUBufPtr(codegen, op->args_[1], GetB64PtrType(dst_dt));
                codegen.Emit("vlds(" + b64_cast + dst + ", " + b64_ptr + ", 0);");
            } else {
                codegen.Emit("vlds(" + cast + dst + ", " + ub_ptr + ", 0, NORM);");
            }
        }
        return "";
    }
    CHECK(op->args_.size() == 3)
        << "vf.load_align requires 2, 3, or 4 args (dst, src_ptr[, offset]) or (dst0, dst1, src_ptr, offset)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string offset_str = ResolveOffsetArg(codegen, op->args_[2], op->args_[1]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    // Pointer type follows the register's declared dtype. When it differs from
    // the tile's dtype, adjust the offset by the element-width ratio so the byte
    // address stays correct (mirrors b64→b32 stride*2).
    std::string ptr_type = DtypeToPtrType(dst_dt);
    DataType tile_dt = GetExprDtype(op->args_[1], dst_dt);
    // Is the destination a MaskReg? (routes to pld/plds instead of vld/vlds)
    bool dst_is_mask = false;
    if (auto dst_v = ir::As<ir::Var>(op->args_[0])) {
        dst_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(dst_v));
    }
    // Offset adjustment only for RegTensor path (MaskReg always uses uint32_t ptr)
    std::string effective_offset = offset_str;
    if (!dst_is_mask && tile_dt.GetBit() != dst_dt.GetBit()) {
        effective_offset = "(" + offset_str + ") * " + std::to_string(tile_dt.GetBit()) + " / " +
                           std::to_string(dst_dt.GetBit());
    }
    // Determine mode from kwargs (dist kwarg is legacy alias for mode).
    // MaskReg (plds) and RegTensor (vlds) paths share the same LoadDist enum:
    // LoadDist includes NORM/US/DS/BRC/... and EnumToString yields the bare
    // name (e.g. "DS"), which plds accepts directly and vlds maps to Bxx suffix.
    std::string mode = "NORM";
    if (op->HasKwarg("dist"))
        mode = VFEnumValueName(ir::EnumToString(static_cast<ir::LoadDist>(op->GetKwarg<int>("dist"))));
    // AddrReg offset path: MaskReg dst -> pld, RegTensor dst -> vld
    if (codegen.IsAddrRegVar(offset_str)) {
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], dst_is_mask ? "uint32_t" : ptr_type);
        if (dst_is_mask) {
            codegen.Emit("pld(" + dst + ", " + ub_ptr + ", " + offset_str + ", " + mode + ");");
        } else {
            codegen.Emit("vld(" + dst + ", " + ub_ptr + ", " + effective_offset + ", " + mode + ");");
        }
        return "";
    }
    // Check for DataBlock load path (vsldb)
    std::string data_copy_mode = "NORM";
    if (op->HasKwarg("data_copy_mode")) {
        data_copy_mode = VFEnumValueName(
            ir::EnumToString(static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"))));
    }
    bool post_update = false;
    if (op->HasKwarg("post_update")) {
        post_update = op->GetKwarg<bool>("post_update");
    }
    // vsldb path (non-contiguous datablock load). Accept both DATA_BLOCK_LOAD
    // (pypto legacy name) and DATA_BLOCK_COPY so
    // code written against either naming does not silently fall back to vlds.
    if (data_copy_mode == "DATA_BLOCK_LOAD" || data_copy_mode == "DATA_BLOCK_COPY") {
        // In DataBlock mode, args[2] is a mask register, not an offset.
        // Verify the user didn't pass an integer/AddrReg offset by mistake.
        auto mask_var = ir::As<ir::Var>(op->args_[2]);
        CHECK(mask_var != nullptr)
            << "vf.load_align with data_copy_mode=DATA_BLOCK_COPY requires args[2] to be a "
            << "mask register, but got an offset value (offset is not supported in DataBlock mode)";
        // vsldb only supports b8/b16/b32 (not b64)
        // vsldb path: load_align(dst, ptr, mask, data_copy_mode=..., block_stride=N, ...)
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string block_stride = "0";
        std::string repeat_stride = "0";
        if (op->HasKwarg("block_stride")) {
            block_stride = std::to_string(op->GetKwarg<int>("block_stride"));
        }
        if (op->HasKwarg("repeat_stride")) {
            repeat_stride = std::to_string(op->GetKwarg<int>("repeat_stride"));
        }
        if (post_update) {
            std::string ub_ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
            codegen.Emit("vsldb(" + dst + ", " + ub_ptr + ", (" + block_stride + " << 16u) | (" + repeat_stride +
                         " & 0xFFFFU), " + mask_reg + ", POST_UPDATE);");
        } else {
            std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
            codegen.Emit("vsldb(" + dst + ", " + ub_ptr + ", (" + block_stride + " << 16u), " + mask_reg + ");");
        }
        return "";
    }
    // Route by dst variable type: MaskReg → plds, RegTensor → vlds
    if (dst_is_mask) {
        // plds path: dst is MaskReg, pointer is always uint32_t*
        // pld/plds: only supports NORM/US/DS dist
        CHECK(mode == "NORM" || mode == "US" || mode == "DS")
            << "vf.load_align (MaskReg) only supports NORM/US/DS dist, got " << mode;
        std::string plds_mode = mode; // pass through NORM/US/DS directly
        std::string plds_ptr;
        if (post_update) {
            plds_ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
            codegen.Emit("plds(" + dst + ", " + plds_ptr + ", " + offset_str + ", " + plds_mode + ", POST_UPDATE);");
        } else {
            plds_ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("plds(" + dst + ", " + plds_ptr + ", " + offset_str + ", " + plds_mode + ");");
        }
        return "";
    }
    // vlds path: dst is RegTensor
    // Get UB pointer (post_update uses reference-based ptr)
    std::string ub_ptr;
    if (post_update) {
        ub_ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
    } else {
        ub_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
    }
    // Determine vlds mode string
    std::string vlds_mode;
    if (mode == "NORM") {
        vlds_mode = "NORM";
    } else if (mode == "BRC") {
        if (IsB8Type(dst_dt))
            vlds_mode = "BRC_B8";
        else if (IsB16Type(dst_dt))
            vlds_mode = "BRC_B16";
        else
            vlds_mode = "BRC_B32";
    } else if (mode == "US") {
        if (IsB8Type(dst_dt))
            vlds_mode = "US_B8";
        else
            vlds_mode = "US_B16";
    } else if (mode == "DS") {
        if (IsB8Type(dst_dt))
            vlds_mode = "DS_B8";
        else
            vlds_mode = "DS_B16";
    } else if (mode == "UNPK") {
        if (IsB8Type(dst_dt))
            vlds_mode = "UNPK_B8";
        else if (IsB16Type(dst_dt))
            vlds_mode = "UNPK_B16";
        else
            vlds_mode = "UNPK_B32";
    } else if (mode == "UNPK4") {
        vlds_mode = "UNPK4_B8";
    } else if (mode == "BLK") {
        vlds_mode = "BLK";
    } else if (mode == "E2B") {
        if (IsB16Type(dst_dt))
            vlds_mode = "E2B_B16";
        else
            vlds_mode = "E2B_B32";
    } else {
        // Fallback: pass through directly (e.g. BRC_B32, DS_B16, E2B_B32, etc.)
        vlds_mode = mode;
    }
    std::string cast = GetB8Cast(dst_dt);
    // B64 types (INT64/UINT64) use __VF_VLDS_B64 which takes 3 args (no mode)
    if (dst_dt.GetBit() == 64) {
        std::string b64_cast = GetB64Cast(dst_dt);
        std::string b64_ptr;
        if (post_update) {
            b64_ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
        } else {
            b64_ptr = GetUBufPtr(codegen, op->args_[1], GetB64PtrType(dst_dt));
        }
        if (post_update) {
            std::string post_offset = "(" + effective_offset + ") * 2";
            codegen.Emit("vlds(" + b64_cast + dst + ", " + b64_ptr + ", " + post_offset + ", POST_UPDATE);");
        } else {
            codegen.Emit("vlds(" + b64_cast + dst + ", " + b64_ptr + ", " + effective_offset + ");");
        }
    } else if (post_update) {
        // B64 register types need stride doubled (postUpdateStride * 2 for 8-byte elements)
        std::string post_offset = effective_offset;
        if (dst_dt.GetBit() == 64) {
            post_offset = "(" + effective_offset + ") * 2";
        }
        codegen.Emit("vlds(" + cast + dst + ", " + ub_ptr + ", " + post_offset + ", " + vlds_mode + ", POST_UPDATE);");
    } else {
        codegen.Emit("vlds(" + cast + dst + ", " + ub_ptr + ", " + effective_offset + ", " + vlds_mode + ");");
    }
    return "";
}

// ============================================================================
// StoreAlign — vsts
// ============================================================================

static std::string EmitVFStoreAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst_ptr, src_reg, mask, (optional) block_stride, (optional) repeat_stride]
    DataType src_dt = GetExprDtype(op->args_[1]);
    // vsts supports b8/b16/b32/b64 element widths
    CHECK(IsB8Type(src_dt) || src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64)
        << "vf.store_align only supports b8/b16/b32/b64 types, got " << DTypeStr(src_dt);
    std::string src_reg = codegen.GetExprAsCode(op->args_[1]);
    std::string cast = GetB8Cast(src_dt);
    // MaskReg src path: when args[1] is a MaskReg, dispatch to psts/pst (mask store)
    bool src_is_mask = false;
    if (auto src_v = ir::As<ir::Var>(op->args_[1])) {
        src_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(src_v));
    }
    if (src_is_mask) {
        CHECK(!op->HasKwarg("data_copy_mode")) << "vf.store_align (MaskReg src) does not support data_copy_mode";
        CHECK(!op->HasKwarg("block_stride")) << "vf.store_align (MaskReg src) does not support block_stride";
        CHECK(!op->HasKwarg("repeat_stride")) << "vf.store_align (MaskReg src) does not support repeat_stride";
        std::string dist = "NORM";
        if (op->HasKwarg("dist")) {
            dist = VFEnumValueName(ir::EnumToString(static_cast<ir::StoreDist>(op->GetKwarg<int>("dist"))));
            // pst/psts: only supports NORM and PACK dist
            CHECK(dist == "NORM" || dist == "PACK" || dist == "PK")
                << "vf.store_align (MaskReg) only supports NORM/PACK dist, got " << dist;
            // psts/pst use PK (not PACK) for packed mode
            if (dist == "PACK")
                dist = "PK";
        }
        // AddrReg offset path: 3rd arg is AddrReg -> pst(mask, ptr, areg, dist)
        if (op->args_.size() >= 3) {
            std::string third_arg = codegen.GetExprAsCode(op->args_[2]);
            if (codegen.IsAddrRegVar(third_arg)) {
                std::string ub_ptr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
                codegen.Emit("pst(" + src_reg + ", " + ub_ptr + ", " + third_arg + ", " + dist + ");");
                return "";
            }
            // Post-update path: int offset with post_update kwarg
            bool post_update = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
            if (post_update) {
                std::string ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
                codegen.Emit("psts(" + src_reg + ", " + ptr_var + ", " + third_arg + ", " + dist + ", POST_UPDATE);");
                return "";
            }
        }
        // Default: psts with offset=0
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
        codegen.Emit("psts(" + src_reg + ", " + ub_ptr + ", 0, " + dist + ");");
        return "";
    }
    // AddrReg offset path: when 4th arg is an AddrReg variable,
    // emit vst(src, ptr, areg, dist, mask) — 5 args (note: vst, not vsts)
    if (op->args_.size() >= 4) {
        std::string addr_reg = codegen.GetExprAsCode(op->args_[3]);
        if (codegen.IsAddrRegVar(addr_reg)) {
            std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
            std::string ptr_type = "float";
            if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
                ptr_type = DtypeToPtrType(scalar_type->dtype_);
            }
            std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
            // Auto-select dist based on src dtype
            std::string dist = "NORM_B32";
            if (IsB8Type(src_dt))
                dist = "NORM_B8";
            else if (IsB16Type(src_dt))
                dist = "NORM_B16";
            codegen.Emit("vst(" + cast + src_reg + ", " + dst_ptr + ", " + addr_reg + ", " + dist + ", " + mask_reg +
                         ");");
            return "";
        }
    }
    // Get kwargs with defaults
    std::string dist = "";
    if (op->HasKwarg("dist")) {
        dist = VFEnumValueName(ir::EnumToString(static_cast<ir::StoreDist>(op->GetKwarg<int>("dist"))));
    }
    if (dist.empty()) {
        // Auto-select default dist based on src dtype (uses NORM_B8/B16/B32 by element width)
        DataType src_dtype = DataType::FP32;
        if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
            src_dtype = scalar_type->dtype_;
        } else if (auto tile_type_tmp = ir::As<ir::TileType>(op->args_[0]->GetType())) {
            src_dtype = tile_type_tmp->dtype_;
        }
        if (IsB8Type(src_dtype))
            dist = "NORM_B8";
        else if (IsB16Type(src_dtype))
            dist = "NORM_B16";
        else
            dist = "NORM_B32";
    }
    // Auto-expand shorthand dist names to element-width-qualified CCE constants
    if (dist == "FIRST_ELEMENT" || dist == "FIRST_ELE") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (IsB8Type(sd))
            dist = "ONEPT_B8";
        else if (IsB16Type(sd))
            dist = "ONEPT_B16";
        else
            dist = "ONEPT_B32";
    } else if (dist == "PACK") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (IsB8Type(sd) || IsB16Type(sd))
            dist = "PK_B16";
        else if (sd.GetBit() == 32)
            dist = "PK_B32";
        else
            dist = "PK_B64";
    } else if (dist == "PACK4") {
        dist = "PK4_B32";
    } else if (dist == "INTLV") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (IsB8Type(sd))
            dist = "INTLV_B8";
        else if (IsB16Type(sd))
            dist = "INTLV_B16";
        else
            dist = "INTLV_B32";
    }
    bool post_update = false;
    if (op->HasKwarg("post_update")) {
        post_update = op->GetKwarg<bool>("post_update");
    }
    std::string data_copy_mode = "NORM";
    if (op->HasKwarg("data_copy_mode")) {
        data_copy_mode = VFEnumValueName(
            ir::EnumToString(static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"))));
    }
    // Pointer type follows the register's declared dtype. When it differs from
    // the tile's dtype, the offset/stride must be adjusted by the element-width
    // ratio so the byte address stays correct (mirrors b64→b32
    // stride*2). No register cast needed — the register is already src_dt.
    std::string ptr_type = DtypeToPtrType(src_dt);
    DataType tile_dt = DataType::FP32;
    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    if (tile_type) {
        tile_dt = tile_type->dtype_;
    } else if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
        tile_dt = scalar_type->dtype_;
    } else {
        tile_dt = src_dt;
    }
    std::string width_op;
    if (tile_dt.GetBit() != src_dt.GetBit()) {
        width_op = " * " + std::to_string(tile_dt.GetBit()) + " / " + std::to_string(src_dt.GetBit());
    }
    // INTLV modes need two src registers: args = [dst_ptr, src_reg, src1, mask]
    bool is_intlv = (dist == "INTLV_B8" || dist == "INTLV_B16" || dist == "INTLV_B32");
    if (is_intlv) {
        CHECK(op->args_.size() == 4) << "vf.store_align INTLV requires 4 args (dst_ptr, src_reg, src1, mask)";
        CHECK(!op->HasKwarg("data_copy_mode")) << "vf.store_align INTLV mode is incompatible with data_copy_mode";
        CHECK(!op->HasKwarg("post_update")) << "vf.store_align INTLV mode does not support post_update";
        CHECK(!op->HasKwarg("block_stride")) << "vf.store_align INTLV mode does not support block_stride";
        CHECK(!op->HasKwarg("repeat_stride")) << "vf.store_align INTLV mode does not support repeat_stride";
        std::string src1 = codegen.GetExprAsCode(op->args_[2]);
        std::string mask_reg = codegen.GetExprAsCode(op->args_[3]);
        // vsts 2-source overload (__VF_VSTSX2) does not exist for FP32;
        // cast to UINT32 (same 32-bit width) to use the UINT32 overload.
        std::string intlv_ptr_type = ptr_type;
        if (src_dt == DataType::FP32) {
            src_reg = "(RegTensor<uint32_t> &)" + src_reg;
            src1 = "(RegTensor<uint32_t> &)" + src1;
            intlv_ptr_type = "uint32_t";
        }
        std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], intlv_ptr_type);
        codegen.Emit("vsts(" + cast + src_reg + ", " + cast + src1 + ", " + dst_ptr + ", 0, " + dist + ", " + mask_reg +
                     ");");
    } else if (data_copy_mode == "DATA_BLOCK_LOAD" || data_copy_mode == "DATA_BLOCK_COPY") {
        // Accept both DATA_BLOCK_LOAD (pypto legacy name) and DATA_BLOCK_COPY
        //, consistent with load_align behavior.
        // vsstb only supports b8/b16/b32 (not b64)
        DataType dc_src_dt = GetExprDtype(op->args_[1]);
        CHECK(dc_src_dt.GetBit() == 8 || dc_src_dt.GetBit() == 16 || dc_src_dt.GetBit() == 32)
            << "vf.store_align (DATA_BLOCK_COPY) only supports b8/b16/b32, got " << DTypeStr(dc_src_dt);
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string block_stride = "0";
        std::string repeat_stride = "0";
        if (op->args_.size() >= 4) {
            block_stride = codegen.GetExprAsCode(op->args_[3]);
        } else if (op->HasKwarg("block_stride")) {
            block_stride = std::to_string(op->GetKwarg<int>("block_stride"));
        }
        if (op->args_.size() >= 5) {
            repeat_stride = codegen.GetExprAsCode(op->args_[4]);
        } else if (op->HasKwarg("repeat_stride")) {
            repeat_stride = std::to_string(op->GetKwarg<int>("repeat_stride"));
        }
        if (post_update) {
            std::string ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
            codegen.Emit("vsstb(" + cast + src_reg + ", " + ptr_var + ", " + "(" + block_stride + " << 16u) | (" +
                         repeat_stride + " & 0xFFFFU), " + mask_reg + ", POST_UPDATE);");
        } else {
            std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
            codegen.Emit("vsstb(" + cast + src_reg + ", " + dst_ptr + ", " + "(" + block_stride + " << 16u) | (" +
                         repeat_stride + " & 0xFFFFU), " + mask_reg + ");");
        }
    } else if (post_update) {
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string stride = (op->args_.size() >= 4) ? codegen.GetExprAsCode(op->args_[3]) : "0";
        // B64 register types need stride doubled (postUpdateStride * 2)
        std::string effective_stride = stride;
        if (src_dt.GetBit() == 64) {
            effective_stride = "(" + stride + ") * 2";
        } else if (!width_op.empty()) {
            effective_stride = "(" + stride + ")" + width_op;
        }
        std::string ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
        // B64 types use __VF_VSTS_B64 which takes 4 args (no dist mode)
        if (src_dt.GetBit() == 64) {
            std::string b64_cast = GetB64Cast(src_dt);
            codegen.Emit("vsts(" + b64_cast + src_reg + ", " + ptr_var + ", " + effective_stride + ", " + mask_reg +
                         ", POST_UPDATE);");
        } else {
            codegen.Emit("vsts(" + cast + src_reg + ", " + ptr_var + ", " + effective_stride + ", " + dist + ", " +
                         mask_reg + ", POST_UPDATE);");
        }
    } else {
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
        std::string offset_str = "0";
        if (op->args_.size() >= 4) {
            offset_str = ResolveOffsetArg(codegen, op->args_[3], op->args_[0]);
        }
        std::string effective_offset = offset_str;
        if (!width_op.empty()) {
            effective_offset = "(" + offset_str + ")" + width_op;
        }
        // B64 types (INT64/UINT64) use __VF_VSTS_B64 which takes 4 args (no dist mode)
        if (src_dt.GetBit() == 64) {
            std::string b64_cast = GetB64Cast(src_dt);
            std::string b64_ptr = GetUBufPtr(codegen, op->args_[0], GetB64PtrType(src_dt));
            codegen.Emit("vsts(" + b64_cast + src_reg + ", " + b64_ptr + ", " + effective_offset + ", " + mask_reg +
                         ");");
        } else {
            codegen.Emit("vsts(" + cast + src_reg + ", " + dst_ptr + ", " + effective_offset + ", " + dist + ", " +
                         mask_reg + ");");
        }
    }
    return "";
}

// ============================================================================
// MemBar — mem_bar
// ============================================================================

static std::string EmitVFMemBar(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string mode = "VST_VLD";
    if (op->HasKwarg("mode")) {
        mode = VFEnumValueName(ir::EnumToString(static_cast<ir::MemBarMode>(op->GetKwarg<int>("mode"))));
    }
    codegen.Emit("mem_bar(" + mode + ");");
    return "";
}

// ============================================================================
// Max — vmax
// ============================================================================

static std::string EmitVFMax(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src0, src1, mask]
    CHECK(op->args_.size() == 4) << "vf.max requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK((s0_dt.IsInt() || s0_dt == DataType::FP16 || s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << "vf.max src only supports supported types, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    CHECK((s1_dt.IsInt() || s1_dt == DataType::FP16 || s1_dt == DataType::FP32 || s1_dt == DataType::BF16))
        << "vf.max src only supports supported types, got " << DTypeStr(s1_dt);
    DataType vf_max_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(s0_dt == vf_max_dst_dt && s1_dt == vf_max_dst_dt)
        << "vf.max requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_max_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFAnyMode(op);
    codegen.Emit("vmax(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Add — vadd
// ============================================================================

static std::string EmitVFAdd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.add requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK((s0_dt.IsInt() || s0_dt == DataType::FP16 || s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << "vf.add src0 only supports INT/UINT/FP16/FP32/BF16, got " << DTypeStr(s0_dt);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    CHECK(dst_dt == s0_dt && dst_dt == s1_dt)
        << "vf.add requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string mode = VFAnyMode(op);
    codegen.Emit("vadd(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Sub — vsub
// ============================================================================

static std::string EmitVFSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.sub requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK((s0_dt.IsInt() || s0_dt == DataType::FP16 || s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << "vf.sub src0 only supports INT/UINT/FP16/FP32/BF16, got " << DTypeStr(s0_dt);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    CHECK(dst_dt == s0_dt && dst_dt == s1_dt)
        << "vf.sub requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string mode = VFZeroingOnly(op, "vf.sub");
    codegen.Emit("vsub(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// And — vand
// ============================================================================

// ============================================================================
// And — vand, Xor — vxor, Or — vor
// Bitwise operations: type-agnostic, only requires same bit width across
// dst/src0/src1.  Any b8/b16/b32/b64 type (including FP16/BF16/FP32) is valid
// because vand/vxor/vor operate on raw bits.
// ============================================================================

static std::string EmitVFAnd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.and_ requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        codegen.Emit("pand(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ");");
        return "";
    }
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK(s0_dt == DataType::INT8 || s0_dt == DataType::UINT8 || s0_dt == DataType::BOOL ||
          s0_dt == DataType::FP8E4M3FN || s0_dt == DataType::FP8E5M2 || s0_dt == DataType::FP8E8M0 ||
          s0_dt == DataType::HF8 || s0_dt == DataType::INT16 || s0_dt == DataType::UINT16 || s0_dt == DataType::FP16 ||
          s0_dt == DataType::BF16 || s0_dt == DataType::INT32 || s0_dt == DataType::UINT32 || s0_dt == DataType::FP32 ||
          s0_dt == DataType::INT64 || s0_dt == DataType::UINT64)
        << "vf.and_ src0 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    CHECK(s1_dt == DataType::INT8 || s1_dt == DataType::UINT8 || s1_dt == DataType::BOOL ||
          s1_dt == DataType::FP8E4M3FN || s1_dt == DataType::FP8E5M2 || s1_dt == DataType::FP8E8M0 ||
          s1_dt == DataType::HF8 || s1_dt == DataType::INT16 || s1_dt == DataType::UINT16 || s1_dt == DataType::FP16 ||
          s1_dt == DataType::BF16 || s1_dt == DataType::INT32 || s1_dt == DataType::UINT32 || s1_dt == DataType::FP32 ||
          s1_dt == DataType::INT64 || s1_dt == DataType::UINT64)
        << "vf.and_ src1 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s1_dt);
    CHECK(dst_dt == s0_dt && dst_dt == s1_dt)
        << "vf.and_ requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string cast_prefix = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)";
    std::string s0_expr = (s0_dt == dst_dt) ? src0 : (cast_prefix + src0);
    std::string s1_expr = (s1_dt == dst_dt) ? src1 : (cast_prefix + src1);
    std::string mode = VFZeroingOnly(op, "vf.and_");
    codegen.Emit("vand(" + dst + ", " + s0_expr + ", " + s1_expr + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Xor — vxor, Or — vor
// Bitwise operations: type-agnostic, only requires same bit width across
// dst/src0/src1.  Any b8/b16/b32/b64 type (including FP16/BF16/FP32) is valid
// because vxor/vor operate on raw bits.
// ============================================================================

static std::string EmitVFBinaryBitwise(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                       const std::string& op_name, const std::string& instruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << op_name << " requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        std::string p_instr = "p" + instruction.substr(1);
        codegen.Emit(p_instr + "(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ");");
        return "";
    }
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK(s0_dt == DataType::INT8 || s0_dt == DataType::UINT8 || s0_dt == DataType::BOOL ||
          s0_dt == DataType::FP8E4M3FN || s0_dt == DataType::FP8E5M2 || s0_dt == DataType::FP8E8M0 ||
          s0_dt == DataType::HF8 || s0_dt == DataType::INT16 || s0_dt == DataType::UINT16 || s0_dt == DataType::FP16 ||
          s0_dt == DataType::BF16 || s0_dt == DataType::INT32 || s0_dt == DataType::UINT32 || s0_dt == DataType::FP32 ||
          s0_dt == DataType::INT64 || s0_dt == DataType::UINT64)
        << op_name
        << " src0 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    CHECK(s1_dt == DataType::INT8 || s1_dt == DataType::UINT8 || s1_dt == DataType::BOOL ||
          s1_dt == DataType::FP8E4M3FN || s1_dt == DataType::FP8E5M2 || s1_dt == DataType::FP8E8M0 ||
          s1_dt == DataType::HF8 || s1_dt == DataType::INT16 || s1_dt == DataType::UINT16 || s1_dt == DataType::FP16 ||
          s1_dt == DataType::BF16 || s1_dt == DataType::INT32 || s1_dt == DataType::UINT32 || s1_dt == DataType::FP32 ||
          s1_dt == DataType::INT64 || s1_dt == DataType::UINT64)
        << op_name
        << " src1 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s1_dt);
    CHECK(dst_dt == s0_dt && dst_dt == s1_dt)
        << op_name << " requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string cast_prefix = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)";
    std::string s0_expr = (s0_dt == dst_dt) ? src0 : (cast_prefix + src0);
    std::string s1_expr = (s1_dt == dst_dt) ? src1 : (cast_prefix + src1);
    std::string mode = VFZeroingOnly(op, op_name);
    codegen.Emit(instruction + "(" + dst + ", " + s0_expr + ", " + s1_expr + ", " + mask + ", " + mode + ");");
    return "";
}

static std::string EmitVFXor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFBinaryBitwise(op, codegen_base, "vf.xor", "vxor");
}

// ============================================================================
// Or — vor
// ============================================================================

static std::string EmitVFOr(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFBinaryBitwise(op, codegen_base, "vf.or_", "vor");
}

// ============================================================================
// Reduce — vcadd/vcmax/vcmin + vcgadd/vcgmax/vcgmin (unified)
// Supports both new-style (mode=SUM/MAX/MIN, datablock) and legacy (reduce_type=ADD/MAX, merge_mode)
// ============================================================================

// Shared reduction emitter. `reduce_mode` is one of "SUM"/"MAX"/"MIN".
static std::string EmitVFReduceImpl(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                    const std::string& reduce_mode)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << op->name_ << " requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    bool datablock = false;
    if (op->HasKwarg("datablock")) {
        datablock = op->GetKwarg<bool>("datablock");
    }
    if (datablock) {
        // Datablock reduce only supports b16/b32 (no b64, no BF16)
        CHECK((src_dt.GetBit() == 16 || src_dt.GetBit() == 32) &&
              (src_dt.IsInt() || src_dt == DataType::FP16 || src_dt == DataType::FP32))
            << op->name_ << " (datablock) src only supports b16/b32 INT/UINT/FP16/FP32, got " << DTypeStr(src_dt);
    } else {
        // Non-datablock reduce supports b16/b32/b64 (no BF16)
        CHECK((src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64) &&
              (src_dt.IsInt() || src_dt == DataType::FP16 || src_dt == DataType::FP32))
            << op->name_ << " src only supports b16/b32/b64 INT/UINT/FP16/FP32, got " << DTypeStr(src_dt);
        DataType reduce_dst_dt = GetExprDtype(op->args_[0]);
        CHECK(src_dt == reduce_dst_dt) << op->name_ << " requires src and dst to have the same type, got dst="
                                       << DTypeStr(reduce_dst_dt) << " src=" << DTypeStr(src_dt);
    }
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string intrinsic;
    if (reduce_mode == "SUM")
        intrinsic = datablock ? "vcgadd" : "vcadd";
    else if (reduce_mode == "MAX")
        intrinsic = datablock ? "vcgmax" : "vcmax";
    else
        intrinsic = datablock ? "vcgmin" : "vcmin";
    // reduce ops use "merge_mode" kwarg (not "mode")
    std::string mode = "MODE_ZEROING";
    if (op->HasKwarg("merge_mode")) {
        auto merge_mode = static_cast<ir::MergeMode>(op->GetKwarg<int>("merge_mode"));
        CHECK(merge_mode == ir::MergeMode::ZEROING)
            << op->name_ << " only supports ZEROING mode on current device, but got MERGING";
        mode = merge_mode == ir::MergeMode::MERGING ? "MODE_MERGING" : "MODE_ZEROING";
    }
    codegen.Emit(intrinsic + "(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

static std::string EmitVFReduceSum(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFReduceImpl(op, codegen_base, "SUM");
}

static std::string EmitVFReduceMax(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFReduceImpl(op, codegen_base, "MAX");
}

static std::string EmitVFReduceMin(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFReduceImpl(op, codegen_base, "MIN");
}

// ============================================================================
// Mul — vmul
// ============================================================================

static std::string EmitVFMul(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.mul requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK((s0_dt.GetBit() == 16 || s0_dt.GetBit() == 32 || s0_dt.GetBit() == 64))
        << "vf.mul src only supports supported types, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    CHECK((s1_dt.GetBit() == 16 || s1_dt.GetBit() == 32 || s1_dt.GetBit() == 64))
        << "vf.mul src only supports supported types, got " << DTypeStr(s1_dt);
    DataType vf_mul_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(s0_dt == vf_mul_dst_dt && s1_dt == vf_mul_dst_dt)
        << "vf.mul requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_mul_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.mul");
    codegen.Emit("vmul(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// MulAddDst — vmula (hardware FMA: dst = src0 * src1 + dst)
// ============================================================================

static std::string EmitVFMulAddDst(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.mul_add_dst requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK((s0_dt.GetBit() == 16 || s0_dt.GetBit() == 32 || s0_dt.GetBit() == 64))
        << "vf.mul_add_dst src only supports supported types, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    CHECK((s1_dt.GetBit() == 16 || s1_dt.GetBit() == 32 || s1_dt.GetBit() == 64))
        << "vf.mul_add_dst src only supports supported types, got " << DTypeStr(s1_dt);
    DataType vf_mul_add_dst_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(s0_dt == vf_mul_add_dst_dst_dt && s1_dt == vf_mul_add_dst_dst_dt)
        << "vf.mul_add_dst requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_mul_add_dst_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.mul_add_dst");
    codegen.Emit("vmula(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Div — vdiv
// ============================================================================

static std::string EmitVFDiv(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.div requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK((s0_dt.IsInt() || s0_dt == DataType::FP16 || s0_dt == DataType::FP32))
        << "vf.div src0 only supports INT/UINT/FP16/FP32, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    CHECK((s1_dt.IsInt() || s1_dt == DataType::FP16 || s1_dt == DataType::FP32))
        << "vf.div src1 only supports INT/UINT/FP16/FP32, got " << DTypeStr(s1_dt);
    DataType vf_div_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(s0_dt == vf_div_dst_dt && s1_dt == vf_div_dst_dt)
        << "vf.div requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_div_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.div");
    codegen.Emit("vdiv(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Muls — vmuls
// ============================================================================

static std::string EmitVFMuls(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, scalar, mask]
    CHECK(op->args_.size() == 4) << "vf.muls requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt.IsInt() || src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.muls src only supports INT/UINT/FP16/FP32, got " << DTypeStr(src_dt);
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    CHECK((scalar_dt == DataType::INT16 || scalar_dt == DataType::UINT16 || scalar_dt == DataType::INT32 ||
           scalar_dt == DataType::UINT32 || scalar_dt == DataType::INT64 || scalar_dt == DataType::UINT64 ||
           scalar_dt == DataType::FP16 || scalar_dt == DataType::FP32))
        << "vf.muls scalar only supports INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32, got " << DTypeStr(scalar_dt);
    DataType vf_muls_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_muls_dst_dt) << "vf.muls requires src and dst to have the same type, got dst="
                                    << DTypeStr(vf_muls_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.muls");
    codegen.Emit("vmuls(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Ln — vln (natural logarithm, basic precision)
// ============================================================================

static std::string EmitVFUnary(const ir::CallPtr& op, codegen::CodegenBase& codegen_base, const std::string& op_name,
                               const std::string& instruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << op_name << " requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, op_name);
    codegen.Emit(instruction + "(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

static std::string EmitVFLn(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.ln src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_ln_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_ln_dst_dt) << "vf.ln requires src and dst to have the same type, got dst="
                                  << DTypeStr(vf_ln_dst_dt) << " src=" << DTypeStr(src_dt);
    return EmitVFUnary(op, codegen_base, "vf.ln", "vln");
}

// ============================================================================
// Log — vln (natural logarithm, same as Ln on A5/dav_3510)
// ============================================================================

static std::string EmitVFLog(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.log src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_log_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_log_dst_dt) << "vf.log requires src and dst to have the same type, got dst="
                                   << DTypeStr(vf_log_dst_dt) << " src=" << DTypeStr(src_dt);
    return EmitVFUnary(op, codegen_base, "vf.log", "vln");
}

// ============================================================================
// Min — vmin
// ============================================================================

static std::string EmitVFMin(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.min requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK((s0_dt.IsInt() || s0_dt == DataType::FP16 || s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << "vf.min src only supports supported types, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    CHECK((s1_dt.IsInt() || s1_dt == DataType::FP16 || s1_dt == DataType::FP32 || s1_dt == DataType::BF16))
        << "vf.min src only supports supported types, got " << DTypeStr(s1_dt);
    DataType vf_min_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(s0_dt == vf_min_dst_dt && s1_dt == vf_min_dst_dt)
        << "vf.min requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_min_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFAnyMode(op);
    codegen.Emit("vmin(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Exp — vexp
// ============================================================================

static std::string EmitVFExp(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.exp requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.exp src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_exp_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_exp_dst_dt) << "vf.exp requires src and dst to have the same type, got dst="
                                   << DTypeStr(vf_exp_dst_dt) << " src=" << DTypeStr(src_dt);
    DataType exp_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == exp_dst_dt) << "vf.exp requires src and dst to have the same type, got dst=" << DTypeStr(exp_dst_dt)
                                << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.exp");
    codegen.Emit("vexp(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Abs — vabs
// ============================================================================

static std::string EmitVFAbs(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.abs requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt.IsSignedInt() || src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.abs src only supports INT8/INT16/INT32/INT64/FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_abs_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_abs_dst_dt) << "vf.abs requires src and dst to have the same type, got dst="
                                   << DTypeStr(vf_abs_dst_dt) << " src=" << DTypeStr(src_dt);
    DataType abs_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == abs_dst_dt) << "vf.abs requires src and dst to have the same type, got dst=" << DTypeStr(abs_dst_dt)
                                << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.abs");
    codegen.Emit("vabs(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Not — vnot
// ============================================================================

static std::string EmitVFNot(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.not_ requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK(src_dt == DataType::INT8 || src_dt == DataType::UINT8 || src_dt == DataType::BOOL ||
          src_dt == DataType::INT16 || src_dt == DataType::UINT16 || src_dt == DataType::INT32 ||
          src_dt == DataType::UINT32 || src_dt == DataType::FP16 || src_dt == DataType::FP32 ||
          src_dt == DataType::INT64 || src_dt == DataType::UINT64)
        << "vf.not_ src only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/FP16/FP32/INT64/UINT64, got "
        << DTypeStr(src_dt);
    DataType not_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == not_dst_dt) << "vf.not_ requires src and dst to have the same type, got dst="
                                << DTypeStr(not_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    if (IsDstMaskReg(op, codegen)) {
        codegen.Emit("pnot(" + dst + ", " + src + ", " + mask + ");");
        return "";
    }
    std::string mode = VFZeroingOnly(op, "vf.not_");
    codegen.Emit("vnot(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Sqrt — vsqrt
// ============================================================================

static std::string EmitVFSqrt(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.sqrt requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.sqrt src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_sqrt_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_sqrt_dst_dt) << "vf.sqrt requires src and dst to have the same type, got dst="
                                    << DTypeStr(vf_sqrt_dst_dt) << " src=" << DTypeStr(src_dt);
    DataType sqrt_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == sqrt_dst_dt) << "vf.sqrt requires src and dst to have the same type, got dst="
                                 << DTypeStr(sqrt_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.sqrt");
    // High-precision mode: when precision=True, emit vsqrt with the
    // SqrtSpecificMode struct to enable 0-ulp fast-inverse algorithm.
    // The struct is emitted as a static constexpr local so the template
    // can take a pointer to it.
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        codegen.Emit("vsqrt<float, true>(" + dst + ", " + src + ", " + mask + ");");
    } else {
        codegen.Emit("vsqrt(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Relu — vrelu
// ============================================================================

static std::string EmitVFRelu(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.relu requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::INT32 || src_dt == DataType::INT64 || src_dt == DataType::FP16 ||
           src_dt == DataType::FP32))
        << "vf.relu src only supports INT32/INT64/FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_relu_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_relu_dst_dt) << "vf.relu requires src and dst to have the same type, got dst="
                                    << DTypeStr(vf_relu_dst_dt) << " src=" << DTypeStr(src_dt);
    DataType relu_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == relu_dst_dt) << "vf.relu requires src and dst to have the same type, got dst="
                                 << DTypeStr(relu_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.relu");
    codegen.Emit("vrelu(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Neg — vneg
// ============================================================================

static std::string EmitVFNeg(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.neg requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt.IsSignedInt() || src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.neg src only supports INT8/INT16/INT32/INT64/FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_neg_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_neg_dst_dt) << "vf.neg requires src and dst to have the same type, got dst="
                                   << DTypeStr(vf_neg_dst_dt) << " src=" << DTypeStr(src_dt);
    DataType neg_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == neg_dst_dt) << "vf.neg requires src and dst to have the same type, got dst=" << DTypeStr(neg_dst_dt)
                                << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.neg");
    codegen.Emit("vneg(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Adds — vadds (scalar addition)
// ============================================================================

static std::string EmitVFAdds(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.adds requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt.IsInt() || src_dt == DataType::FP16 || src_dt == DataType::FP32 || src_dt == DataType::BF16))
        << "vf.adds src only supports supported types, got " << DTypeStr(src_dt);
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    CHECK((scalar_dt.IsInt() || scalar_dt == DataType::FP16 || scalar_dt == DataType::FP32 ||
           scalar_dt == DataType::BF16))
        << "vf.adds scalar only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32/BF16, got "
        << DTypeStr(scalar_dt);
    DataType vf_adds_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_adds_dst_dt) << "vf.adds requires src and dst to have the same type, got dst="
                                    << DTypeStr(vf_adds_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.adds");
    codegen.Emit("vadds(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Subs — vadds with negated scalar (scalar subtraction: dst = src - scalar)
// ============================================================================

static std::string EmitVFSubs(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.subs requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt.IsInt() || src_dt == DataType::FP16 || src_dt == DataType::FP32 || src_dt == DataType::BF16))
        << "vf.subs src only supports INT/UINT/FP16/FP32/BF16, got " << DTypeStr(src_dt);
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    CHECK((scalar_dt.IsInt() || scalar_dt == DataType::FP16 || scalar_dt == DataType::FP32 ||
           scalar_dt == DataType::BF16))
        << "vf.subs scalar only supports INT/UINT/FP16/FP32/BF16, got " << DTypeStr(scalar_dt);
    DataType vf_subs_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_subs_dst_dt) << "vf.subs requires src and dst to have the same type, got dst="
                                    << DTypeStr(vf_subs_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.subs");
    codegen.Emit("vadds(" + dst + ", " + src + ", -(" + scalar_str + "), " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Mins — vmins (scalar minimum)
// ============================================================================

static std::string EmitVFMins(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.mins requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt.IsInt() || src_dt == DataType::FP16 || src_dt == DataType::FP32 || src_dt == DataType::BF16))
        << "vf.mins src only supports supported types, got " << DTypeStr(src_dt);
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    CHECK((scalar_dt.IsInt() || scalar_dt == DataType::FP16 || scalar_dt == DataType::FP32 ||
           scalar_dt == DataType::BF16))
        << "vf.mins scalar only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32/BF16, got "
        << DTypeStr(scalar_dt);
    DataType vf_mins_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_mins_dst_dt) << "vf.mins requires src and dst to have the same type, got dst="
                                    << DTypeStr(vf_mins_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.mins");
    codegen.Emit("vmins(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Maxs — vmaxs (scalar maximum)
// ============================================================================

static std::string EmitVFMaxs(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.maxs requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt.IsInt() || src_dt == DataType::FP16 || src_dt == DataType::FP32 || src_dt == DataType::BF16))
        << "vf.maxs src only supports supported types, got " << DTypeStr(src_dt);
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    CHECK((scalar_dt.IsInt() || scalar_dt == DataType::FP16 || scalar_dt == DataType::FP32 ||
           scalar_dt == DataType::BF16))
        << "vf.maxs scalar only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32/BF16, got "
        << DTypeStr(scalar_dt);
    DataType vf_maxs_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_maxs_dst_dt) << "vf.maxs requires src and dst to have the same type, got dst="
                                    << DTypeStr(vf_maxs_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.maxs");
    codegen.Emit("vmaxs(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// LeakyRelu — vlrelu
// ============================================================================

static std::string EmitVFLeakyRelu(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.leaky_relu requires 4 args (dst, src, alpha, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.leaky_relu src only supports supported types, got " << DTypeStr(src_dt);
    DataType alpha_dt = GetExprDtype(op->args_[2]);
    CHECK((alpha_dt == DataType::FP16 || alpha_dt == DataType::FP32))
        << "vf.leaky_relu scalar only supports FP16/FP32, got " << DTypeStr(alpha_dt);
    DataType vf_leaky_relu_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_leaky_relu_dst_dt)
        << "vf.leaky_relu requires src and dst to have the same type, got dst=" << DTypeStr(vf_leaky_relu_dst_dt)
        << " src=" << DTypeStr(src_dt);
    DataType lrelu_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == lrelu_dst_dt) << "vf.leaky_relu requires src and dst to have the same type, got dst="
                                  << DTypeStr(lrelu_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string alpha = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.leaky_relu");
    codegen.Emit("vlrelu(" + dst + ", " + src + ", " + alpha + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Interleave — vintlv
// ============================================================================

static std::string EmitVFInterleave(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.interleave requires 4 args (dst0, dst1, src0, src1)";
    std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
    std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        DataType dtype = DataType::FP32;
        if (op->HasKwarg("dtype")) {
            dtype = op->GetKwarg<DataType>("dtype");
        }
        // MaskInterleave: only supports b8/b16/b32 (not b64)
        CHECK(dtype.GetBit() == 8 || dtype.GetBit() == 16 || dtype.GetBit() == 32)
            << "vf.interleave (MaskReg) only supports b8/b16/b32, got " << DTypeStr(dtype);
        std::string pintlv_op;
        if (dtype == DataType::UINT8 || dtype == DataType::INT8) {
            pintlv_op = "pintlv_b8";
        } else if (dtype.GetBit() == 16) {
            pintlv_op = "pintlv_b16";
        } else {
            pintlv_op = "pintlv_b32";
        }
        codegen.Emit(pintlv_op + "(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
        return "";
    }
    // vintlv requires src0/src1 to be b8/b16/b32/b64
    DataType src0_dt = GetExprDtype(op->args_[2]);
    DataType src1_dt = GetExprDtype(op->args_[3]);
    CHECK((src0_dt.GetBit() == 8 || src0_dt.GetBit() == 16 || src0_dt.GetBit() == 32 || src0_dt.GetBit() == 64))
        << "vf.interleave only supports b8/b16/b32/b64 types, got " << DTypeStr(src0_dt);
    CHECK((src0_dt.GetBit() == src1_dt.GetBit()))
        << "vf.interleave requires src0 and src1 to have the same bit width, got src0=" << DTypeStr(src0_dt)
        << " src1=" << DTypeStr(src1_dt);
    codegen.Emit("vintlv(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
    return "";
}

// ============================================================================
// PairReduceSum — vcpadd
// ============================================================================

static std::string EmitVFPairReduceSum(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.pair_reduce_sum requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.pair_reduce_sum src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType prs_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == prs_dst_dt) << "vf.pair_reduce_sum requires src and dst to have the same type, got dst="
                                << DTypeStr(prs_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.pair_reduce_sum");
    codegen.Emit("vcpadd(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// AbsSub — vabsdif (absolute difference: |src0 - src1|)
// ============================================================================

static std::string EmitVFAbsSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.abs_sub requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK((s0_dt == DataType::FP16 || s0_dt == DataType::FP32 || s0_dt == DataType::INT64))
        << "vf.abs_sub src0 only supports FP16/FP32/INT64, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    CHECK((s1_dt == DataType::FP16 || s1_dt == DataType::FP32 || s1_dt == DataType::INT64))
        << "vf.abs_sub src1 only supports FP16/FP32/INT64, got " << DTypeStr(s1_dt);
    DataType vf_abs_sub_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(s0_dt == vf_abs_sub_dst_dt && s1_dt == vf_abs_sub_dst_dt)
        << "vf.abs_sub requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_abs_sub_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.abs_sub");
    codegen.Emit("vabsdif(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Axpy — vaxpy (accumulate: dst = src * scalar + dst)
// ============================================================================

static std::string EmitVFAxpy(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.axpy requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::FP16 || src_dt == DataType::FP32 || src_dt == DataType::INT64 ||
           src_dt == DataType::UINT64))
        << "vf.axpy src only supports FP16/FP32/INT64/UINT64, got " << DTypeStr(src_dt);
    // vaxpy supports half/float/uint64_t/int64_t
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    CHECK(scalar_dt == DataType::FP16 || scalar_dt == DataType::FP32 || scalar_dt == DataType::UINT64 ||
          scalar_dt == DataType::INT64)
        << "vf.axpy scalar only supports FP16/FP32/UINT64/INT64, got " << DTypeStr(scalar_dt);
    DataType vf_axpy_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_axpy_dst_dt) << "vf.axpy requires src and dst to have the same type, got dst="
                                    << DTypeStr(vf_axpy_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.axpy");
    codegen.Emit("vaxpy(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Madd — vmadd (multiply-accumulate: dst = src0 * src1 + dst)
// ============================================================================

static std::string EmitVFMulDstAdd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.mul_dst_add requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK((s0_dt == DataType::FP16 || s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << "vf.mul_dst_add src0 only supports FP16/FP32/BF16, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    CHECK((s1_dt == DataType::FP16 || s1_dt == DataType::FP32 || s1_dt == DataType::BF16))
        << "vf.mul_dst_add src1 only supports FP16/FP32/BF16, got " << DTypeStr(s1_dt);
    DataType vf_mul_dst_add_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(s0_dt == vf_mul_dst_add_dst_dt && s1_dt == vf_mul_dst_add_dst_dt)
        << "vf.mul_dst_add requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_mul_dst_add_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    // vmadd supports u16/i16/u32/i32/half/float/bf/i64/u64
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.mul_dst_add");
    codegen.Emit("vmadd(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Pack — vpack (narrow data type)
// ============================================================================

static std::string EmitVFPack(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "vf.pack requires 2 args (dst, src)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string part = "LOWER";
    if (op->HasKwarg("part")) {
        part = VFEnumValueName(ir::EnumToString(static_cast<ir::PackPart>(op->GetKwarg<int>("part"))));
    }
    if (IsDstMaskReg(op, codegen)) {
        std::string cce_half = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("ppack(" + dst + ", " + src + ", " + cce_half + ");");
        return "";
    }

    DataType src_dt = GetExprDtype(op->args_[1]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    CHECK((IsB8Type(src_dt) || src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64))
        << "vf.pack src only supports b8/b16/b32/b64 types, got " << DTypeStr(src_dt);
    // vpack: dst bit width must be half of src
    CHECK(dst_dt.GetBit() == src_dt.GetBit() / 2)
        << "vf.pack: dst bit width must be half of src (dst=" << DTypeStr(dst_dt) << " " << dst_dt.GetBit()
        << "-bit, src=" << DTypeStr(src_dt) << " " << src_dt.GetBit() << "-bit)";

    if (src_dt.GetBit() == 64) {
        // 64-bit source → 32-bit dst: use DeInterleave with a zero register,
        // mirroring PackImpl<..., part>(dst, src) for 8-byte src.
        std::string part_check = (part == "LOWER" || part == "LOWEST") ? "LOWEST" : "HIGHEST";
        std::string zero_var = dst + "_pack_zero_";
        std::string dump_var = dst + "_pack_dump_";
        std::string mask_var = dst + "_pack_mask_";
        codegen.Emit("RegTensor<uint32_t> " + zero_var + ";");
        codegen.Emit("RegTensor<uint32_t> " + dump_var + ";");
        codegen.Emit("MaskReg " + mask_var + " = pset_b32(PAT_ALL);");
        codegen.Emit("vdup(" + zero_var + ", 0, " + mask_var + ", MODE_ZEROING);");
        if (part_check == "LOWEST") {
            codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + dst + ", " + dump_var + ", " + "(RegTensor<uint32_t>&)" +
                         src + ", " + zero_var + ");");
        } else {
            codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + dst + ", " + dump_var + ", " + zero_var +
                         ", (RegTensor<uint32_t>&)" + src + ");");
        }
    } else {
        std::string cce_part = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("vpack(" + dst + ", " + src + ", " + cce_part + ", MODE_UNKNOWN);");
    }
    return "";
}

// ============================================================================
// Unpack — vunpack (widen data type)
// ============================================================================

static std::string EmitVFUnpack(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "vf.unpack requires 2 args (dst, src)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string part = "LOWER";
    if (op->HasKwarg("part")) {
        part = VFEnumValueName(ir::EnumToString(static_cast<ir::PackPart>(op->GetKwarg<int>("part"))));
    }

    if (IsDstMaskReg(op, codegen)) {
        std::string cce_half = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("punpack(" + dst + ", " + src + ", " + cce_half + ");");
        return "";
    }

    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((IsB8Type(src_dt) || src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64))
        << "vf.unpack src only supports b8/b16/b32/b64 types, got " << DTypeStr(src_dt);
    // vunpack: dst bit width must be double of src
    CHECK(dst_dt.GetBit() == src_dt.GetBit() * 2)
        << "vf.unpack: dst bit width must be double of src (dst=" << DTypeStr(dst_dt) << " " << dst_dt.GetBit()
        << "-bit, src=" << DTypeStr(src_dt) << " " << src_dt.GetBit() << "-bit)";

    if (dst_dt.GetBit() == 64) {
        std::string src_ctype = src_dt.ToCTypeString();
        std::string part_check = (part == "LOWER" || part == "LOWEST") ? "LOWEST" : "HIGHEST";
        std::string pad_var = dst + "_unpack_pad_";
        std::string dump_var = dst + "_unpack_dump_";
        std::string mask_var = dst + "_unpack_mask_";
        codegen.Emit("RegTensor<" + src_ctype + "> " + pad_var + ";");
        codegen.Emit("RegTensor<" + src_ctype + "> " + dump_var + ";");
        codegen.Emit("MaskReg " + mask_var + " = pset_b32(PAT_ALL);");
        if (src_dt == DataType::INT32) {
            codegen.Emit("vshrs(" + pad_var + ", " + src + ", 31, " + mask_var + ", MODE_ZEROING);");
        } else {
            codegen.Emit("vdup(" + pad_var + ", 0, " + mask_var + ", MODE_ZEROING);");
        }
        if (part_check == "LOWEST") {
            codegen.Emit("vintlv((RegTensor<" + src_ctype + ">&)" + dst + ", " + dump_var + ", " + "(RegTensor<" +
                         src_ctype + ">&)" + src + ", " + pad_var + ");");
        } else {
            codegen.Emit("vintlv(" + dump_var + ", (RegTensor<" + src_ctype + ">&)" + dst + ", " + "(RegTensor<" +
                         src_ctype + ">&)" + src + ", " + pad_var + ");");
        }
    } else {
        std::string cce_part = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("vunpack(" + dst + ", " + src + ", " + cce_part + ");");
    }
    return "";
}

// ============================================================================
// PRelu — vprelu (parametric ReLU with per-element slope)
// ============================================================================

static std::string EmitVFPRelu(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.prelu requires 4 args (dst, src, slope, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.prelu src only supports supported types, got " << DTypeStr(src_dt);
    DataType slope_dt = GetExprDtype(op->args_[2]);
    DataType prelu_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == prelu_dst_dt && slope_dt == prelu_dst_dt)
        << "vf.prelu requires dst, src, slope to have the same type, got dst=" << DTypeStr(prelu_dst_dt)
        << " src=" << DTypeStr(src_dt) << " slope=" << DTypeStr(slope_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string slope = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.prelu");
    codegen.Emit("vprelu(" + dst + ", " + src + ", " + slope + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// ShiftLeft — unified left shift: vshl (per-lane, shift is a RegTensor) or
// vshls (uniform, shift is a scalar). The former standalone vf.shift_lefts
// (scalar) op is merged in here; the register-vs-scalar decision is made from
// the codegen RegTensor registry (mirrors EmitVFDuplicate's IsRegTensorVar
// dispatch), not from the op name.
// ============================================================================

// Returns true when the shift-amount arg (op->args_[2]) is a per-lane vector
// register; false for a uniform scalar shift (integer literal or plain scalar).
static bool ShiftAmountIsRegister(const ir::CallPtr& op, codegen::CCECodegen& codegen)
{
    // Integer literals are always scalar shifts — no need to consult the registry.
    if (ir::As<ir::ConstInt>(op->args_[2]))
        return false;
    auto shift_var = ir::As<ir::Var>(op->args_[2]);
    if (!shift_var)
        return false;
    return codegen.IsRegTensorVar(codegen.GetVarName(shift_var));
}

static std::string EmitVFShift(const ir::CallPtr& op, codegen::CodegenBase& codegen_base, const std::string& op_name,
                               const std::string& vector_instruction, const std::string& scalar_instruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << op_name << " requires 4 args (dst, src, shift, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK(src_dt == DataType::INT8 || src_dt == DataType::UINT8 || src_dt == DataType::INT16 ||
          src_dt == DataType::UINT16 || src_dt == DataType::INT32 || src_dt == DataType::UINT32 ||
          src_dt == DataType::INT64 || src_dt == DataType::UINT64)
        << op_name << " src only supports integer types, got " << DTypeStr(src_dt);
    DataType shift_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == shift_dst_dt) << op_name
                                  << " requires src and dst to have the same type, got dst=" << DTypeStr(shift_dst_dt)
                                  << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string shift = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, op_name);
    if (ShiftAmountIsRegister(op, codegen)) {
        // vshl/vshr (vector): shift reg must be signed int (int8/int16/int32/int64)
        DataType shift_dt = GetExprDtype(op->args_[2]);
        CHECK(shift_dt == DataType::INT8 || shift_dt == DataType::INT16 || shift_dt == DataType::INT32 ||
              shift_dt == DataType::INT64)
            << op_name << " (vector) shift register only supports INT8/INT16/INT32/INT64, got " << DTypeStr(shift_dt);
        codegen.Emit(vector_instruction + "(" + dst + ", " + src + ", *(vector_s32*)&(" + shift + "), " + mask + ", " +
                     mode + ");");
    } else {
        codegen.Emit(scalar_instruction + "(" + dst + ", " + src + ", (int16_t)(" + shift + "), " + mask + ", " + mode +
                     ");");
    }
    return "";
}

static std::string EmitVFShiftLeft(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFShift(op, codegen_base, "vf.shift_left", "vshl", "vshls");
}

// ============================================================================
// ShiftRight — unified right shift: vshr (per-lane, shift is a RegTensor) or
// vshrs (uniform, shift is a scalar). The former standalone vf.shift_rights
// (scalar) op is merged in here; dispatch uses ShiftAmountIsRegister (codegen
// RegTensor registry), not the op name.
// ============================================================================

static std::string EmitVFShiftRight(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFShift(op, codegen_base, "vf.shift_right", "vshr", "vshrs");
}

// ============================================================================
// Mull — vmull (long multiply: 32x32->64, lo/hi split)
// ============================================================================

static std::string EmitVFMull(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 5) << "vf.mull requires 5 args (dst_lo, dst_hi, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[2]);
    CHECK((s0_dt == DataType::INT32 || s0_dt == DataType::UINT32))
        << "vf.mull src only supports supported types, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[3]);
    CHECK((s1_dt == DataType::INT32 || s1_dt == DataType::UINT32))
        << "vf.mull src only supports supported types, got " << DTypeStr(s1_dt);
    DataType dst_lo_dt = GetExprDtype(op->args_[0]);
    DataType dst_hi_dt = GetExprDtype(op->args_[1]);
    CHECK(dst_lo_dt == s0_dt && dst_hi_dt == s0_dt && s0_dt == s1_dt)
        << "vf.mull requires dst_lo, dst_hi, src0, src1 to have the same type, got dst_lo=" << DTypeStr(dst_lo_dt)
        << " dst_hi=" << DTypeStr(dst_hi_dt) << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst_lo = codegen.GetExprAsCode(op->args_[0]);
    std::string dst_hi = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string mask = codegen.GetExprAsCode(op->args_[4]);
    codegen.Emit("vmull(" + dst_lo + ", " + dst_hi + ", " + src0 + ", " + src1 + ", " + mask + ");");
    return "";
}

// ============================================================================
// Addc — vaddcs (add with carry)
// ============================================================================

static std::string EmitVFAddc(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 6) << "vf.addc requires 6 args (carry_out, dst, src0, src1, carry_in, mask)";
    DataType s0_dt = GetExprDtype(op->args_[2]);
    CHECK(s0_dt == DataType::INT32 || s0_dt == DataType::UINT32)
        << "vf.addc src0 only supports INT32/UINT32, got " << DTypeStr(s0_dt);
    DataType dst_dt = GetExprDtype(op->args_[1]);
    DataType s1_dt = GetExprDtype(op->args_[3]);
    CHECK(dst_dt == s0_dt && dst_dt == s1_dt)
        << "vf.addc requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string carry_out = codegen.GetExprAsCode(op->args_[0]);
    std::string dst = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string carry_in = codegen.GetExprAsCode(op->args_[4]);
    std::string mask = codegen.GetExprAsCode(op->args_[5]);
    codegen.Emit("vaddcs(" + carry_out + ", " + dst + ", " + src0 + ", " + src1 + ", " + carry_in + ", " + mask + ");");
    return "";
}

// ============================================================================
// Subc — vsubcs (subtract with borrow)
// ============================================================================

static std::string EmitVFSubc(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 6) << "vf.subc requires 6 args (borrow_out, dst, src0, src1, borrow_in, mask)";
    DataType s0_dt = GetExprDtype(op->args_[2]);
    CHECK(s0_dt == DataType::INT32 || s0_dt == DataType::UINT32)
        << "vf.subc src0 only supports INT32/UINT32, got " << DTypeStr(s0_dt);
    DataType dst_dt = GetExprDtype(op->args_[1]);
    DataType s1_dt = GetExprDtype(op->args_[3]);
    CHECK(dst_dt == s0_dt && dst_dt == s1_dt)
        << "vf.subc requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string borrow_out = codegen.GetExprAsCode(op->args_[0]);
    std::string dst = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string borrow_in = codegen.GetExprAsCode(op->args_[4]);
    std::string mask = codegen.GetExprAsCode(op->args_[5]);
    codegen.Emit("vsubcs(" + borrow_out + ", " + dst + ", " + src0 + ", " + src1 + ", " + borrow_in + ", " + mask +
                 ");");
    return "";
}

// ============================================================================
// ExpSub — vexpdif
// ============================================================================

// Helpers for multi-step int→int conversion via float intermediate.
// Available int→float vcvt overloads (bisheng intrinsics):
//   b8→f16 (widening), s16→f16 (same-width), s16→f32 (widening), s32→f32 (same-width)
// NOT available: u16→f16, u16→f32, u32→f32, s32→f16

// Emit int→float vcvt step. Returns the float dtype used (FP16 or FP32).
// For unavailable src→f16 paths (u16, u32, s32), routes through f32 instead.
static DataType EmitIntToFloatVcvt(codegen::CCECodegen& codegen, const std::string& dst_reg, const std::string& src_reg,
                                   const std::string& mask, const std::string& part, const std::string& round,
                                   const std::string& mode, DataType src_dt)
{
    if (src_dt.GetBit() == 8) {
        // b8→f16: widening → vcvt(dst, src, mask, PART, MODE) — 5 args
        codegen.Emit("vcvt(" + dst_reg + ", " + src_reg + ", " + mask + ", " + part + ", " + mode + ");");
        return DataType::FP16;
    }
    if (src_dt == DataType::INT16) {
        // s16→f16: same-width → vcvt(dst, src, mask, ROUND, MODE) — 5 args
        codegen.Emit("vcvt(" + dst_reg + ", " + src_reg + ", " + mask + ", " + round + ", " + mode + ");");
        return DataType::FP16;
    }
    // b32 (s32/u32): s32→f32 same-width. u32 needs reinterpret as s32 first.
    std::string effective_src = src_reg;
    if (src_dt == DataType::UINT32)
        effective_src = "(RegTensor<int32_t> &)" + src_reg;
    // s32→f32: same-width → vcvt(dst, src, mask, ROUND, MODE) — 5 args
    codegen.Emit("vcvt(" + dst_reg + ", " + effective_src + ", " + mask + ", " + round + ", " + mode + ");");
    return DataType::FP32;
}

// Emit float→int vcvt step. float_dt is always FP16 (from EmitIntToFloatVcvt).
static void EmitFloatToIntVcvt(codegen::CCECodegen& codegen, const std::string& dst_reg, const std::string& src_reg,
                               const std::string& mask, const std::string& part, const std::string& round,
                               const std::string& sat, const std::string& mode, DataType dst_dt)
{
    if (dst_dt.GetBit() == 8) {
        // float→b8: narrower-int → vcvt(dst, src, mask, ROUND, SAT, PART, MODE) — 7 args
        codegen.Emit("vcvt(" + dst_reg + ", " + src_reg + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
                     mode + ");");
    } else if (dst_dt.GetBit() == 16) {
        // f16→b16: same-width → vcvt(dst, src, mask, ROUND, RS, MODE) — 5 args
        codegen.Emit("vcvt(" + dst_reg + ", " + src_reg + ", " + mask + ", " + round + ", " + sat + ", " + mode + ");");
    } else {
        // f16→b32: wider-int → vcvt(dst, src, mask, ROUND, PART, MODE) — 6 args
        codegen.Emit("vcvt(" + dst_reg + ", " + src_reg + ", " + mask + ", " + round + ", " + part + ", " + mode +
                     ");");
    }
}

static std::string EmitVFExpSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, max, mask]
    CHECK(op->args_.size() == 4) << "vf.exp_sub requires 4 args (dst, src, max, mask)";
    // vexpdiff: dst must be FP32, src can be FP32 or FP16
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK(src_dt == DataType::FP32 || src_dt == DataType::FP16)
        << "vf.exp_sub only supports FP32/FP16 src, got " << DTypeStr(src_dt);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    CHECK(dst_dt == DataType::FP32) << "vf.exp_sub destination only supports FP32, got " << DTypeStr(dst_dt);
    DataType max_dt = GetExprDtype(op->args_[2]);
    // vexpdiff: src0 and src1 must be the same type (both float or both half)
    CHECK(src_dt == max_dt) << "vf.exp_sub requires src and max to have the same type, got src=" << DTypeStr(src_dt)
                            << " max=" << DTypeStr(max_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string max_reg = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // layout kwarg selects the result half: ZERO -> PART_EVEN (default), ONE -> PART_ODD.
    // vexpdiff: only supports RegLayout ZERO/ONE
    std::string part = "PART_EVEN";
    if (op->HasKwarg("layout")) {
        auto layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
        CHECK(layout == "ZERO" || layout == "ONE") << "vf.exp_sub only supports layout ZERO/ONE, got " << layout;
        if (layout == "ONE")
            part = "PART_ODD";
    }
    codegen.Emit("vexpdif(" + dst + ", " + src + ", " + max_reg + ", " + mask + ", " + part + ");");
    return "";
}

// ============================================================================
// Cast — vcvt
// ============================================================================

static std::string EmitVFCast(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, mask]
    CHECK(op->args_.size() == 3) << "vf.astype requires 3 args (dst, src, mask)";
    // vcvt: src and dst must have different types
    DataType src_dtype = GetExprDtype(op->args_[1], DataType::FP32);
    DataType dst_dtype = GetExprDtype(op->args_[0]);
    CHECK(src_dtype != dst_dtype) << "vf.astype: src and dst must have different types (both are "
                                  << DTypeStr(src_dtype) << ")";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    // Get layout and round_mode with defaults
    std::string layout = "ZERO";
    if (op->HasKwarg("layout")) {
        layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
        // vcvt: layout must be ZERO/ONE/TWO/THREE
        CHECK(layout == "ZERO" || layout == "ONE" || layout == "TWO" || layout == "THREE")
            << "vf.astype only supports layout ZERO/ONE/TWO/THREE, got " << layout;
    }
    std::string round_mode = "CAST_RINT";
    if (op->HasKwarg("round_mode")) {
        round_mode = VFEnumValueName(ir::EnumToString(static_cast<ir::VFRoundMode>(op->GetKwarg<int>("round_mode"))));
    }
    // A5 vcvt only supports MODE_ZEROING at the instruction level.
    std::string mode_value = VFZeroingOnly(op, "vf.astype");
    std::string part;
    if (layout == "ZERO")
        part = "PART_EVEN";
    else if (layout == "ONE")
        part = "PART_ODD";
    else if (layout == "TWO")
        part = "PART_TWO";
    else if (layout == "THREE")
        part = "PART_THREE";
    else
        part = "PART_EVEN";
    std::string part_pp;
    if (layout == "ZERO")
        part_pp = "PART_P0";
    else if (layout == "ONE")
        part_pp = "PART_P1";
    else if (layout == "TWO")
        part_pp = "PART_P2";
    else if (layout == "THREE")
        part_pp = "PART_P3";
    else
        part_pp = "PART_P0";
    // Map round_mode to ::ROUND constants (dav_3510 GetRound mapping):
    //   CAST_RINT  → ROUND_R (round to nearest even)
    //   CAST_ROUND → ROUND_A (round to nearest, away from zero)
    //   CAST_FLOOR → ROUND_F
    //   CAST_CEIL  → ROUND_C
    //   CAST_TRUNC → ROUND_Z
    //   CAST_ODD   → ROUND_O (Von Neumann rounding)
    //   CAST_HYBRID→ ROUND_H (3510 only, not supported by all vcvt overloads)
    //   unknown    → ROUND_R (safe fallback, universally supported)
    std::string round;
    if (round_mode == "CAST_RINT")
        round = "ROUND_R";
    else if (round_mode == "CAST_ROUND")
        round = "ROUND_A";
    else if (round_mode == "CAST_FLOOR")
        round = "ROUND_F";
    else if (round_mode == "CAST_CEIL")
        round = "ROUND_C";
    else if (round_mode == "CAST_TRUNC")
        round = "ROUND_Z";
    else if (round_mode == "CAST_ODD")
        round = "ROUND_O"; // only supported by widening float→float vcvt overloads
    else if (round_mode == "CAST_HYBRID")
        round = "ROUND_H"; // only supported by widening float→float vcvt overloads
    else
        round = "ROUND_R"; // unknown → ROUND_R (safe fallback)
    // Get saturation kwarg (default: disabled)
    std::string sat = "RS_DISABLE";
    if (op->HasKwarg("saturate")) {
        std::string sat_val = VFEnumValueName(
            ir::EnumToString(static_cast<ir::SaturateMode>(op->GetKwarg<int>("saturate"))));
        if (sat_val == "ON" || sat_val == "ENABLE")
            sat = "RS_ENABLE";
    }
    // Determine if narrowing or widening conversion based on src/dst dtype.
    // Widening: dst wider than src — no ROUND/RS needed, just PART + MODE
    // Narrowing: dst narrower — needs ROUND + RS + PART + MODE
    // Float→same-width-int (FP32→S32): ROUND + RS + MODE (no PART)
    // INT→FLOAT same-width (S32→FP32): ROUND + MODE (no RS, no PART)
    // FLOAT→FLOAT same-width (FP16→BF16): ROUND + MODE (no RS, no PART)
    // INT→INT narrowing: RS + PART + MODE (no ROUND)
    // Cross-width INT↔FLOAT (S16→FP32, S32→FP16, FP32→S64, S64→FP32): ROUND + PART + MODE
    bool is_widening = false;
    bool is_int_to_float = false;
    bool is_float_to_same_int = false;     // FP32→S32, FP16→S16 (same-width float→int)
    bool is_float_to_narrower_int = false; // FP16→INT8/UINT8, BF16→INT8/UINT8 (2x narrowing float→int)
    bool is_float_to_wider_int = false;    // FP16→S32, BF16→S32 (widening float→int)
    bool is_int_2x_narrowing = false;      // 2x narrowing with RS: b32→b16, b16→u8, b64→b32
    bool is_int_4x_narrowing = false;      // 4x narrowing with RS: b32→u8
    bool is_int_4x_widening = false;       // 4x int widening: b8→b32 (PART_P0/P1/P2/P3 only)
    bool is_cross_width = false;           // cross-width INT↔FLOAT with ROUND + PART + MODE
    bool is_s4_widening = false;           // INT4→FP16/BF16/INT16 (vcvt_s42f16/bf16/s16)
    bool is_s4_narrowing = false;          // FP16→INT4 (vcvt_f162s4)
    bool is_s16_to_s4 = false;             // INT16→INT4 (two-step: s16→f16→s4, mirroring AscendC Cast)
    bool is_int_int_two_step = false;      // int→int via f16 intermediate (cross-sign widening, int→INT8 narrowing)
    // FP8/FP4 low-precision conversions
    bool is_fp_widen_pp = false;          // 4x widening PP: FP8→FP32, FP4→BF16 → vcvt(dst,src,mask,PART_PP,MODE)
    bool is_fp_narrow_rnd_sat_pp = false; // 4x narrowing RND_SAT_PP: FP32→FP8 → vcvt(dst,src,mask,ROUND,SAT)
    bool is_fp_narrow_rnd_pp = false;     // 4x narrowing RND_PP: BF16→FP4 → vcvt(dst,src,mask,ROUND,PART_PP,MODE)
    // S4 (INT4) special instructions: vcvt_s42f16, vcvt_s42bf16, vcvt_s42s16, vcvt_f162s4
    if (src_dtype == DataType::INT4 &&
        (dst_dtype == DataType::FP16 || dst_dtype == DataType::BF16 || dst_dtype == DataType::INT16)) {
        is_s4_widening = true;
    } else if (dst_dtype == DataType::INT4 && src_dtype == DataType::FP16) {
        is_s4_narrowing = true;
    } else if (dst_dtype == DataType::INT4 && src_dtype == DataType::INT16) {
        is_s16_to_s4 = true;
    }
    // Widening cases: vcvt(dst, src, mask, PART, MODE) — no ROUND/RS
    // FP16→FP32, BF16→FP32, INT16→INT32, UINT16→UINT32, INT8→INT16, UINT8→UINT16,
    // INT8/UINT8→FP16, S32→S64, INT16/UINT16→FP32 (__VF_VCVTIF_PART)
    // Cross-sign s→u 2x widening: INT16→UINT32 (AscendC layoutMerge partCondition supports this)
    // NOTE: Cross-sign u→s widening (u8→s16, u16→s32) and s8→u16 are NOT supported by hardware.
    // NOTE: u32→s64 is NOT supported (only s32→s64 is).
    else if ((src_dtype == DataType::FP16 && dst_dtype == DataType::FP32) ||
             (src_dtype == DataType::BF16 && dst_dtype == DataType::FP32) ||
             (src_dtype == DataType::UINT16 && dst_dtype == DataType::UINT32) ||
             (src_dtype == DataType::INT16 && dst_dtype == DataType::INT32) ||
             (src_dtype == DataType::INT16 && dst_dtype == DataType::UINT32) ||
             (src_dtype == DataType::INT8 && dst_dtype == DataType::INT16) ||
             (src_dtype == DataType::UINT8 && dst_dtype == DataType::UINT16) ||
             (src_dtype == DataType::INT8 && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::UINT8 && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::INT32 && dst_dtype == DataType::INT64) ||
             (src_dtype == DataType::INT16 && dst_dtype == DataType::FP32) ||
             (src_dtype == DataType::UINT16 && dst_dtype == DataType::FP32) ||
             (src_dtype == DataType::HF8 && dst_dtype == DataType::FP16)) {
        is_widening = true;
    }
    // INT→FLOAT same-width or FLOAT→FLOAT same-width: vcvt(dst, src, mask, ROUND, MODE_ZEROING)
    // S32/U32→FP32, S16/U16→FP16, FP16→BF16, BF16→FP16
    else if (((src_dtype == DataType::INT32 || src_dtype == DataType::UINT32) && dst_dtype == DataType::FP32) ||
             ((src_dtype == DataType::INT16 || src_dtype == DataType::UINT16) && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::FP16 && dst_dtype == DataType::BF16) ||
             (src_dtype == DataType::BF16 && dst_dtype == DataType::FP16)) {
        is_int_to_float = true;
    }
    // FLOAT→same-width-INT: vcvt(dst, src, mask, ROUND, RS, MODE_ZEROING) — no PART
    // FP32→S32/U32, FP16→S16/U16 (same element width)
    else if ((src_dtype == DataType::FP32 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32)) ||
             (src_dtype == DataType::FP16 && (dst_dtype == DataType::INT16 || dst_dtype == DataType::UINT16))) {
        is_float_to_same_int = true;
    }
    // FLOAT→narrower-INT (2x narrowing): vcvt(dst, src, mask, ROUND, RS, PART, MODE_ZEROING)
    // FP16→INT8/UINT8, BF16→INT8/UINT8 (b16→b8, uses __VF_VCVTFI_SAT_PART with PART_EVEN/ODD)
    else if ((src_dtype == DataType::FP16 || src_dtype == DataType::BF16) &&
             (dst_dtype == DataType::INT8 || dst_dtype == DataType::UINT8)) {
        is_float_to_narrower_int = true;
    }
    // FLOAT→wider-INT: vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING) — no RS
    // FP16→S32/U32, BF16→S32/U32 (half-width float to full-width int)
    else if ((src_dtype == DataType::FP16 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32)) ||
             (src_dtype == DataType::BF16 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32))) {
        is_float_to_wider_int = true;
    }
    // Cross-width INT↔FLOAT: vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING)
    // S32/U32→FP16, FP32→S64, S64→FP32
    // (S16/U16→FP32 moved to is_widening — uses __VF_VCVTIF_PART with PART+MODE only)
    else if (((src_dtype == DataType::INT32 || src_dtype == DataType::UINT32) && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::FP32 && dst_dtype == DataType::INT64) ||
             (src_dtype == DataType::INT64 && dst_dtype == DataType::FP32)) {
        is_cross_width = true;
    }
    // INT→INT 4x narrowing to UINT8 (b32→u8): vcvt(dst, src, mask, RS, PART_PP, MODE) — 6 args
    // __VF_VCVTII_SAT_PP: uses PART_P0/P1/P2/P3, supports RS parameter
    else if ((src_dtype == DataType::INT32 || src_dtype == DataType::UINT32) && dst_dtype == DataType::UINT8) {
        is_int_4x_narrowing = true;
    }
    // INT→INT 2x narrowing to UINT8/INT16/UINT16 (b16→u8, b32→b16, b64→b32): vcvt(dst, src, mask, RS, PART, MODE) — 6
    // args
    // __VF_VCVTII_SAT_PART: uses PART_EVEN/ODD, supports RS parameter
    else if (((src_dtype == DataType::INT32 || src_dtype == DataType::UINT32) &&
              (dst_dtype == DataType::INT16 || dst_dtype == DataType::UINT16)) ||
             ((src_dtype == DataType::INT16 || src_dtype == DataType::UINT16) && dst_dtype == DataType::UINT8) ||
             (src_dtype == DataType::INT64 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32))) {
        is_int_2x_narrowing = true;
    }
    // INT→INT 4x widening (b8→b32): vcvt(dst, src, mask, PART_PP, MODE_ZEROING)
    // __VF_VCVTII_PP: uses PART_P0/P1/P2/P3
    // NOTE: Cross-sign 4x widening (s8→u32, u8→s32) is NOT supported by hardware.
    else if ((src_dtype == DataType::INT8 && dst_dtype == DataType::INT32) ||
             (src_dtype == DataType::UINT8 && dst_dtype == DataType::UINT32)) {
        is_int_4x_widening = true;
    }
    // FP8/FP4 4x widening (PP mode): FP8→FP32, FP4→BF16
    // __VF_VCVTFF_PP: vcvt(dst, src, mask, PART_PP, MODE_ZEROING) — 5 args
    else if (((src_dtype == DataType::FP8E4M3FN || src_dtype == DataType::FP8E5M2 || src_dtype == DataType::HF8) &&
              dst_dtype == DataType::FP32) ||
             ((src_dtype == DataType::FP4E2M1 || src_dtype == DataType::FP4E1M2) && dst_dtype == DataType::BF16)) {
        is_fp_widen_pp = true;
    }
    // FP8/FP4 4x narrowing with SAT (RND_SAT_PP): FP32→FP8
    // __VF_VCVTFF_RND_SAT_PP: vcvt(dst, src, mask, ROUND, SAT) — 5 args
    else if (src_dtype == DataType::FP32 &&
             (dst_dtype == DataType::FP8E4M3FN || dst_dtype == DataType::FP8E5M2 || dst_dtype == DataType::HF8)) {
        is_fp_narrow_rnd_sat_pp = true;
    }
    // FP4 4x narrowing without SAT (RND_PP): BF16→FP4
    // __VF_VCVTFF_RND_PP: vcvt(dst, src, mask, ROUND, PART_PP, MODE_ZEROING) — 6 args
    else if (src_dtype == DataType::BF16 && (dst_dtype == DataType::FP4E2M1 || dst_dtype == DataType::FP4E1M2)) {
        is_fp_narrow_rnd_pp = true;
    }
    // Remaining int→int paths not directly supported by hardware vcvt overloads:
    // cross-sign widening (u→s, s→u) and int→INT8 narrowing.
    // Route through f16 intermediate (two-step), mirroring AscendC's s16→f16→s4 approach.
    if (src_dtype.IsInt() && dst_dtype.IsInt() && !is_s4_widening && !is_s4_narrowing && !is_s16_to_s4 &&
        !is_widening && !is_int_4x_widening && !is_int_4x_narrowing && !is_int_2x_narrowing) {
        is_int_int_two_step = true;
    }

    // Validate round_mode against the specific vcvt overload constraints:
    // - Widening (no precision loss): round_mode is ignored (UNKNOWN), warn if set
    // - is_int_to_float / is_float_to_same_int: R/A/F/C/Z only (no O/H)
    // - is_fp_narrow_rnd_sat_pp (FP32→FP8E4M3FN/FP8E5M2): only ROUND_R
    // - is_fp_narrow_rnd_sat_pp (FP32→HF8): only ROUND_A/ROUND_H
    // - is_fp_narrow_rnd_pp (BF16→FP4): R/A/F/C/Z only (no O/H)
    // - FP16→HF8 (else/fallback path): only ROUND_A/ROUND_H
    // - is_fp_widen_pp (FP8→FP32, FP4→BF16): no round_mode (UNKNOWN)
    // - is_s4_widening (INT4→FP16/BF16/INT16): no round_mode (UNKNOWN)
    // - is_int_2x_narrowing / is_int_4x_narrowing / is_int_4x_widening / is_s16_to_s4: no round_mode
    if (is_widening || is_fp_widen_pp || is_s4_widening || is_s16_to_s4 || is_int_2x_narrowing || is_int_4x_narrowing ||
        is_int_4x_widening) {
        CHECK(!op->HasKwarg("round_mode") || round_mode == "CAST_RINT")
            << "vf.astype: round_mode is not applicable for this widening/no-precision-loss "
            << "conversion path (src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype)
            << "), only default CAST_RINT is accepted";
    }
    if (is_int_to_float || is_float_to_same_int || is_float_to_narrower_int || is_fp_narrow_rnd_pp ||
        is_float_to_wider_int || is_cross_width || is_int_int_two_step) {
        CHECK(round != "ROUND_O" && round != "ROUND_H")
            << "vf.astype: round_mode CAST_ODD/CAST_HYBRID is not supported for this conversion path "
            << "(src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << "), "
            << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC";
    }
    if (is_fp_narrow_rnd_sat_pp) {
        if (dst_dtype == DataType::HF8) {
            // FP32→HF8: only CAST_ROUND/CAST_HYBRID. Default CAST_RINT is invalid,
            // so if user didn't specify, silently use CAST_ROUND (the hardware default).
            if (!op->HasKwarg("round_mode")) {
                round = "ROUND_A";
            } else {
                CHECK(round == "ROUND_A" || round == "ROUND_H")
                    << "vf.astype: FP32→HF8 only supports round_mode CAST_ROUND/CAST_HYBRID, got " << round_mode;
            }
        } else {
            // FP32→FP8E4M3FN/FP8E5M2: only CAST_RINT. Default is CAST_RINT, so
            // not specifying round_mode is fine.
            if (op->HasKwarg("round_mode")) {
                CHECK(round == "ROUND_R")
                    << "vf.astype: FP32→FP8E4M3FN/FP8E5M2 only supports round_mode CAST_RINT, got " << round_mode;
            }
        }
    }
    if (src_dtype == DataType::FP16 && dst_dtype == DataType::HF8) {
        // FP16→HF8: only CAST_ROUND/CAST_HYBRID. Default CAST_RINT is invalid,
        // so if user didn't specify, silently use CAST_ROUND.
        if (!op->HasKwarg("round_mode")) {
            round = "ROUND_A";
        } else {
            CHECK(round == "ROUND_A" || round == "ROUND_H")
                << "vf.astype: FP16→HF8 only supports round_mode CAST_ROUND/CAST_HYBRID, got " << round_mode;
        }
    }
    // FP16→INT4 (is_s4_narrowing): R/A/F/C/Z only (no O/H)
    if (is_s4_narrowing) {
        CHECK(round != "ROUND_O" && round != "ROUND_H")
            << "vf.astype: round_mode CAST_ODD/CAST_HYBRID is not supported for FP16→INT4, "
            << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC";
    }
    // Validate saturate against conversion path:
    // - Widening / is_fp_widen_pp / is_s4_widening / is_int_4x_widening: saturate not applicable
    // - is_int_to_float (INT→FLOAT): saturate not applicable (default saturated, no choice)
    // - is_int_4x_narrowing (b32→u8): saturate is mandatory (always RS_ENABLE)
    // - is_int_2x_narrowing (b32→b16, b16→u8, b64→b32): saturate OFF/ON both supported
    // - is_s16_to_s4 (INT16→INT4): saturate OFF/ON both supported (like is_s4_narrowing)
    // - is_float_to_narrower_int (FP16→INT8/UINT8): saturate OFF/ON both supported
    // - is_float_to_same_int (FP32→S32): saturate OFF/ON both supported
    // - BF16→FP16 (float→float same-width): saturate OFF/ON both supported
    // - is_fp_narrow_rnd_sat_pp: saturate is mandatory (always RS_ENABLE)
    // - is_fp_narrow_rnd_pp (BF16→FP4): saturate not applicable (UNKNOWN)
    // - FP→FP32 (widening float): only OFF (non-saturated)
    if (is_widening || is_fp_widen_pp || is_s4_widening || is_int_4x_widening || is_fp_narrow_rnd_pp ||
        (is_int_to_float && !(src_dtype == DataType::BF16 && dst_dtype == DataType::FP16))) {
        CHECK(!op->HasKwarg("saturate") || sat == "RS_DISABLE")
            << "vf.astype: saturate is not applicable for this conversion path "
            << "(src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << ")";
    }
    if (is_int_4x_narrowing) {
        // b32→b8 (4x narrowing): saturate is always enabled (RS_ENABLE). Default is RS_DISABLE,
        // so if user didn't specify, silently enable it.
        if (!op->HasKwarg("saturate")) {
            sat = "RS_ENABLE";
        } else {
            CHECK(sat == "RS_ENABLE") << "vf.astype: 4x int narrowing (b32→b8) requires saturate=ON (RS_ENABLE), "
                                      << "OFF is not supported for this path";
        }
    }
    if (is_fp_narrow_rnd_sat_pp) {
        // FP32→FP8: saturate is always enabled (RS_ENABLE). Default is RS_DISABLE,
        // so if user didn't specify, silently enable it.
        if (!op->HasKwarg("saturate")) {
            sat = "RS_ENABLE";
        } else {
            CHECK(sat == "RS_ENABLE") << "vf.astype: FP32→FP8 conversion requires saturate=ON (RS_ENABLE), "
                                      << "OFF is not supported for this path";
        }
    }
    if (dst_dtype == DataType::FP32 && src_dtype.GetBit() < 32 && (src_dtype.IsFloat() || src_dtype == DataType::HF8)) {
        // FP→FP32 widening: only non-saturated mode
        CHECK(sat == "RS_DISABLE") << "vf.astype: conversion to FP32 only supports saturate=OFF (non-saturated mode)";
    }
    // Validate layout against conversion path:
    // - Same-width conversions (is_int_to_float, is_float_to_same_int): layout not applicable
    // - FP16→BF16 and BF16→FP16 (same-width float→float): layout not applicable
    // - FP32→INT64 and INT64→FP32: same 64-bit width, layout not applicable
    if (is_int_to_float || is_float_to_same_int || (src_dtype == DataType::FP16 && dst_dtype == DataType::BF16) ||
        (src_dtype == DataType::BF16 && dst_dtype == DataType::FP16) ||
        (src_dtype == DataType::FP32 && dst_dtype == DataType::INT64) ||
        (src_dtype == DataType::INT64 && dst_dtype == DataType::FP32)) {
        CHECK(!op->HasKwarg("layout") || layout == "ZERO")
            << "vf.astype: layout is not applicable for this same-width conversion path "
            << "(src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << ")";
    }
    // FP8/FP4 widening: layout supports ZERO/ONE/TWO/THREE (4x expansion)
    // FP8/FP4 2x widening (HF8→FP16): layout supports ZERO/ONE only
    if (is_fp_widen_pp) {
        if (src_dtype == DataType::HF8 && dst_dtype == DataType::FP16) {
            CHECK(layout == "ZERO" || layout == "ONE")
                << "vf.astype: HF8→FP16 only supports layout ZERO/ONE, got " << layout;
        } else if (src_dtype == DataType::FP8E8M0 && dst_dtype == DataType::BF16) {
            CHECK(layout == "ZERO" || layout == "ONE")
                << "vf.astype: FP8E8M0→BF16 only supports layout ZERO/ONE, got " << layout;
        }
        // FP8E4M3FN/FP8E5M2→FP32 and FP4→BF16 support all four layouts
    }
    // 2x widening (is_widening): layout supports ZERO/ONE only
    if (is_widening) {
        CHECK(layout == "ZERO" || layout == "ONE")
            << "vf.astype: 2x widening conversion only supports layout ZERO/ONE, got " << layout
            << " (src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << ")";
    }
    // 4x int widening (b8→b32): layout supports ZERO/ONE/TWO/THREE
    if (is_int_4x_widening) {
        CHECK(layout == "ZERO" || layout == "ONE" || layout == "TWO" || layout == "THREE")
            << "vf.astype: 4x int widening conversion only supports layout ZERO/ONE/TWO/THREE, got " << layout
            << " (src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << ")";
    }
    // 4x int narrowing (b32→u8) and INT16→INT4: layout supports ZERO/ONE/TWO/THREE
    if (is_int_4x_narrowing || is_s16_to_s4) {
        CHECK(layout == "ZERO" || layout == "ONE" || layout == "TWO" || layout == "THREE")
            << "vf.astype: 4x int narrowing conversion only supports layout ZERO/ONE/TWO/THREE, got " << layout
            << " (src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << ")";
    }
    // Float narrowing with PART (partCondition in AscendC): layout supports ZERO/ONE only.
    // This covers FP32→FP16, FP32→BF16, FP16→HF8, FP16→UINT8, FP16→INT8, BF16→INT32, FP32→INT16,
    // FP32→INT64 (all use vcvt(dst,src,mask,ROUND,SAT,PART,MODE) — 7 args).
    // AscendC: static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>());
    if (!is_s4_widening && !is_s4_narrowing && !is_s16_to_s4 && !is_widening && !is_fp_widen_pp && !is_int_to_float &&
        !is_float_to_same_int && !is_float_to_narrower_int && !is_float_to_wider_int && !is_cross_width &&
        !is_int_2x_narrowing && !is_int_int_two_step && !is_int_4x_narrowing && !is_int_4x_widening &&
        !is_fp_narrow_rnd_sat_pp && !is_fp_narrow_rnd_pp &&
        !(src_dtype == DataType::FP16 && dst_dtype == DataType::BF16) &&
        !(src_dtype == DataType::BF16 && dst_dtype == DataType::FP16) &&
        !(src_dtype == DataType::FP32 && dst_dtype == DataType::INT64) &&
        !(src_dtype == DataType::INT64 && dst_dtype == DataType::FP32)) {
        CHECK(layout == "ZERO" || layout == "ONE")
            << "vf.astype: layout only supports ZERO/ONE for this conversion path "
            << "(src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << "), got " << layout;
        // Round mode validation for fallback (partCondition) paths:
        // FP32→FP16: supports CAST_ODD but NOT CAST_HYBRID
        if (src_dtype == DataType::FP32 && dst_dtype == DataType::FP16) {
            CHECK(round != "ROUND_H")
                << "vf.astype: FP32→FP16 does not support round_mode CAST_HYBRID, "
                << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC/CAST_ODD";
        } else {
            // All other fallback paths (FP32→BF16, FP16→INT8, FP16→UINT8, BF16→FP16, FP32→INT16, etc.):
            // support CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC only (no CAST_ODD/CAST_HYBRID)
            CHECK(round != "ROUND_O" && round != "ROUND_H")
                << "vf.astype: round_mode CAST_ODD/CAST_HYBRID is not supported for this conversion path "
                << "(src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << "), "
                << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC";
        }
    }
    if (is_s4_widening) {
        // INT4→FP16/BF16/INT16: specialized vcvt_s42*
        if (dst_dtype == DataType::FP16) {
            codegen.Emit("vcvt_s42f16(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
        } else if (dst_dtype == DataType::BF16) {
            codegen.Emit("vcvt_s42bf16(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
        } else {
            codegen.Emit("vcvt_s42s16(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
        }
    } else if (is_s4_narrowing) {
        // FP16→INT4: vcvt_f162s4
        codegen.Emit("vcvt_f162s4(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part_pp +
                     ", " + mode_value + ");");
    } else if (is_widening) {
        // vcvt(dst, src, mask, PART, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + part + ", " + mode_value + ");");
    } else if (is_int_to_float) {
        // vcvt(dst, src, mask, ROUND, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + mode_value + ");");
    } else if (is_float_to_same_int) {
        // vcvt(dst, src, mask, ROUND, RS, MODE_ZEROING) — no PART
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + mode_value + ");");
    } else if (is_float_to_wider_int || is_cross_width) {
        // vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING) — no RS
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + part + ", " + mode_value + ");");
    } else if (is_float_to_narrower_int) {
        // FLOAT→narrower-INT (2x narrowing): FP16→INT8/UINT8, BF16→INT8/UINT8
        // AscendC CastOperator<T,U,RoundMode,SatMode,RegLayout,MaskMergeMode> emits:
        //   vcvt(dstReg, srcReg, mask, roundModeValue, satModeValue, partModeValue, modeValue)
        // satMode is passed through directly (NO_SAT → RS_DISABLE, SAT → RS_ENABLE).
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
                     mode_value + ");");
    } else if (is_int_4x_widening) {
        // 4x int widening (b8→b32): vcvt(dst, src, mask, PART_PP, MODE_ZEROING) — 5 args
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
    } else if (is_int_4x_narrowing) {
        // 4x int narrowing to UINT8 (b32→u8): vcvt(dst, src, mask, RS, PART_PP, MODE) — 6 args
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + sat + ", " + part_pp + ", " + mode_value + ");");
    } else if (is_int_2x_narrowing) {
        // 2x int narrowing with RS (b32→b16, b16→u8, b64→b32): vcvt(dst, src, mask, RS, PART, MODE) — 6 args
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + sat + ", " + part + ", " + mode_value + ");");
    } else if (is_s16_to_s4) {
        // INT16→INT4: two-step conversion mirroring AscendC CastOperator (s16→f16→s4).
        // AscendC hardcodes RoundMode::CAST_RINT for both steps (round_mode not in trait).
        // Step 1: vcvt(tmp_f16, src_s16, mask, ROUND_R, MODE) — s16→f16 (5 args, rndMergeCast)
        // Step 2: vcvt_f162s4(dst_s4, tmp_f16, mask, ROUND_R, SAT, PART_PP, MODE) — f16→s4 (7 args)
        std::string tmp = dst + "_f16_tmp";
        codegen.HoistRegTensorDecl("RegTensor<half> " + tmp + ";");
        codegen.Emit("vcvt(" + tmp + ", " + src + ", " + mask + ", ROUND_R, " + mode_value + ");");
        codegen.Emit("vcvt_f162s4(" + dst + ", " + tmp + ", " + mask + ", ROUND_R, " + sat + ", " + part_pp + ", " +
                     mode_value + ");");
    } else if (is_int_int_two_step) {
        // Multi-step int→int for paths not directly supported by hardware vcvt overloads.
        // Available int→float overloads: b8→f16, s16→f16, s16→f32, s32→f32
        // NOT available: u16→f16, u16→f32, u32→f32, s32→f16
        //
        // Strategy by src/dst type combination:
        // 1. src→f16 available (b8, s16): two-step src→f16→dst
        // 2. s32/u32 src + b8 dst: three-step src→f32→f16→dst (f32→f16 narrowing needed)
        // 3. u16 src + b8 dst: four-step u16→u32→f32→f16→dst (u16→f16 not available)
        // 4. u16 src + b32 dst: u16→u32 widening + dst cast (zero-extend, semantically correct)
        // 5. same-width cross-sign (u16→s16, s32→u32, u32→s32): reinterpret cast, no instruction
        std::string dst_c_type = dst_dtype.ToCTypeString();
        bool src_to_f16_ok = (src_dtype.GetBit() == 8) || (src_dtype == DataType::INT16);
        bool src_is_b32 = (src_dtype == DataType::INT32 || src_dtype == DataType::UINT32);
        bool dst_is_b8 = (dst_dtype.GetBit() == 8);
        bool dst_is_b32 = (dst_dtype.GetBit() == 32);
        bool same_width = (src_dtype.GetBit() == dst_dtype.GetBit());

        if (src_to_f16_ok) {
            // Two-step: src→f16→dst
            std::string tmp = dst + "_f16_tmp";
            codegen.HoistRegTensorDecl("RegTensor<half> " + tmp + ";");
            EmitIntToFloatVcvt(codegen, tmp, src, mask, part, round, mode_value, src_dtype);
            EmitFloatToIntVcvt(codegen, dst, tmp, mask, part, round, sat, mode_value, dst_dtype);
        } else if (src_is_b32 && dst_is_b8) {
            // Three-step: src→f32→f16→dst (s32→f32→f16→s8, u32 reinterpreted as s32 first)
            std::string tmp_f32 = dst + "_f32_tmp";
            std::string tmp_f16 = dst + "_f16_tmp";
            codegen.HoistRegTensorDecl("RegTensor<float> " + tmp_f32 + ";");
            codegen.HoistRegTensorDecl("RegTensor<half> " + tmp_f16 + ";");
            EmitIntToFloatVcvt(codegen, tmp_f32, src, mask, part, round, mode_value, src_dtype);
            // f32→f16: float narrowing → vcvt(tmp_f16, tmp_f32, mask, ROUND, SAT, PART, MODE) — 7 args
            codegen.Emit("vcvt(" + tmp_f16 + ", " + tmp_f32 + ", " + mask + ", " + round + ", " + sat + ", " + part +
                         ", " + mode_value + ");");
            EmitFloatToIntVcvt(codegen, dst, tmp_f16, mask, part, round, sat, mode_value, dst_dtype);
        } else if (src_dtype == DataType::UINT16 && dst_is_b8) {
            // Four-step: u16→u32→f32→f16→dst (u16→f16 not available, so widen to u32 first)
            std::string tmp_u32 = dst + "_u32_tmp";
            std::string tmp_f32 = dst + "_f32_tmp";
            std::string tmp_f16 = dst + "_f16_tmp";
            codegen.HoistRegTensorDecl("RegTensor<uint32_t> " + tmp_u32 + ";");
            codegen.HoistRegTensorDecl("RegTensor<float> " + tmp_f32 + ";");
            codegen.HoistRegTensorDecl("RegTensor<half> " + tmp_f16 + ";");
            // u16→u32: widening → vcvt(tmp_u32, src, mask, PART, MODE) — 5 args
            codegen.Emit("vcvt(" + tmp_u32 + ", " + src + ", " + mask + ", " + part + ", " + mode_value + ");");
            EmitIntToFloatVcvt(codegen, tmp_f32, tmp_u32, mask, part, round, mode_value, DataType::UINT32);
            // f32→f16: float narrowing → vcvt(tmp_f16, tmp_f32, mask, ROUND, SAT, PART, MODE) — 7 args
            codegen.Emit("vcvt(" + tmp_f16 + ", " + tmp_f32 + ", " + mask + ", " + round + ", " + sat + ", " + part +
                         ", " + mode_value + ");");
            EmitFloatToIntVcvt(codegen, dst, tmp_f16, mask, part, round, sat, mode_value, dst_dtype);
        } else if (src_dtype == DataType::UINT16 && dst_is_b32) {
            // u16→s32: u16→u32 widening with dst cast (zero-extend preserves value)
            std::string u32_c_type = DataType::UINT32.ToCTypeString();
            codegen.Emit("vcvt((RegTensor<" + u32_c_type + "> &)" + dst + ", " + src + ", " + mask + ", " + part +
                         ", " + mode_value + ");");
        } else if (same_width && src_dtype != dst_dtype) {
            // Same-width cross-sign (u16→s16, s32→u32, u32→s32): reinterpret cast
            codegen.Emit(dst + " = (RegTensor<" + dst_c_type + "> &)" + src + ";");
        }
    } else if (is_fp_widen_pp) {
        // FP8/FP4 4x widening PP: vcvt(dst, src, mask, PART_PP, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
    } else if (is_fp_narrow_rnd_sat_pp) {
        // FP32→FP8 4x narrowing RND_SAT_PP: vcvt(dst, src, mask, ROUND, SAT, PART_PP, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part_pp + ", " +
                     mode_value + ");");
    } else if (is_fp_narrow_rnd_pp) {
        // BF16→FP4 4x narrowing RND_PP: vcvt(dst, src, mask, ROUND, PART_PP, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + part_pp + ", " + mode_value +
                     ");");
    } else {
        // Float narrowing (FP32→FP16, FP32→BF16, FP16→INT8, etc.)
        // vcvt(dst, src, mask, ROUND, RS, PART, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
                     mode_value + ");");
    }
    return "";
}

// ============================================================================
// DeInterleave — vdintlv
// ============================================================================

static std::string EmitVFDeInterleave(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst0, dst1, src0, src1]
    CHECK(op->args_.size() == 4) << "vf.de_interleave requires 4 args (dst0, dst1, src0, src1)";
    std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
    std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        DataType dtype = DataType::FP32;
        if (op->HasKwarg("dtype")) {
            dtype = op->GetKwarg<DataType>("dtype");
        }
        // MaskDeInterleave: only supports b8/b16/b32 (not b64)
        CHECK(dtype.GetBit() == 8 || dtype.GetBit() == 16 || dtype.GetBit() == 32)
            << "vf.de_interleave (MaskReg) only supports b8/b16/b32, got " << DTypeStr(dtype);
        std::string pdintlv_op;
        if (dtype == DataType::UINT8 || dtype == DataType::INT8) {
            pdintlv_op = "pdintlv_b8";
        } else if (dtype.GetBit() == 16) {
            pdintlv_op = "pdintlv_b16";
        } else {
            pdintlv_op = "pdintlv_b32";
        }
        codegen.Emit(pdintlv_op + "(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
        return "";
    }
    // vdintlv overloads are keyed on the dst element type; if src dtype differs
    // (e.g. u16 reg reinterpreted as u8 to split into byte streams, mirroring
    // `(RegTensor<uint8_t>&)vreg0U16` in vf_topk.h:60), emit a reinterpret cast.
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s0_dt = GetExprDtype(op->args_[2]);
    DataType s1_dt = GetExprDtype(op->args_[3]);
    std::string cast_prefix = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)";
    std::string s0_expr = (s0_dt == dst_dt) ? src0 : (cast_prefix + src0);
    std::string s1_expr = (s1_dt == dst_dt) ? src1 : (cast_prefix + src1);
    codegen.Emit("vdintlv(" + dst0 + ", " + dst1 + ", " + s0_expr + ", " + s1_expr + ");");
    return "";
}

// ============================================================================
// Select — vsel
// ============================================================================

static std::string EmitVFSelect(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src_true, src_false, mask]
    CHECK(op->args_.size() == 4) << "vf.select requires 4 args (dst, src_true, src_false, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src_true = codegen.GetExprAsCode(op->args_[1]);
    std::string src_false = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        codegen.Emit("psel(" + dst + ", " + src_true + ", " + src_false + ", " + mask + ");");
        return "";
    }
    // Doc: select supports BOOL/INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32
    DataType dst_dt = GetExprDtype(op->args_[0]);
    CHECK(dst_dt == DataType::INT8 || dst_dt == DataType::UINT8 || dst_dt == DataType::BOOL ||
          dst_dt == DataType::INT16 || dst_dt == DataType::UINT16 || dst_dt == DataType::FP16 ||
          dst_dt == DataType::BF16 || dst_dt == DataType::INT32 || dst_dt == DataType::UINT32 ||
          dst_dt == DataType::FP32)
        << "vf.select only supports BOOL/INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32, got " << DTypeStr(dst_dt);
    DataType st_dt = GetExprDtype(op->args_[1]);
    DataType sf_dt = GetExprDtype(op->args_[2]);
    CHECK(dst_dt.GetBit() == st_dt.GetBit() && dst_dt.GetBit() == sf_dt.GetBit())
        << "vf.select requires dst, src_true, src_false to have the same bit width, got dst=" << dst_dt.GetBit()
        << "-bit src_true=" << st_dt.GetBit() << "-bit src_false=" << sf_dt.GetBit() << "-bit";
    std::string cast_prefix = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)";
    std::string st_expr = (st_dt == dst_dt) ? src_true : (cast_prefix + src_true);
    std::string sf_expr = (sf_dt == dst_dt) ? src_false : (cast_prefix + src_false);
    codegen.Emit("vsel(" + dst + ", " + st_expr + ", " + sf_expr + ", " + mask + ");");
    return "";
}

// ============================================================================
// UpdateMask — plt_b32/plt_b16
// ============================================================================

static std::string EmitVFUpdateMask(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 1) << "vf.update_mask requires 1 arg (scalar)";
    std::string scalar = codegen.GetExprAsCode(op->args_[0]);
    std::string reg_name = codegen.GetCurrentResultTarget();
    // Default to b32 (float), use dtype kwarg to select b16 or b8
    bool use_b8 = false;
    bool use_b16 = false;
    if (op->HasKwarg("dtype")) {
        auto dtype = op->GetKwarg<DataType>("dtype");
        CHECK(IsB8Type(dtype) || dtype.GetBit() == 16 || dtype.GetBit() == 32 || dtype.GetBit() == 64)
            << "vf.update_mask dtype must be b8/b16/b32/b64, got " << DTypeStr(dtype);
        use_b8 = (dtype == DataType::UINT8 || dtype == DataType::INT8);
        use_b16 = (dtype.GetBit() == 16);
    }
    // plt_b32/plt_b16 requires uint32_t& reference, so declare a variable first
    std::string scalar_var = "_vf_mask_scalar_" + std::to_string(codegen.GetTileOffsetCounter());
    codegen.Emit("uint32_t " + scalar_var + " = (uint32_t)" + scalar + ";");
    codegen.Emit("MaskReg " + reg_name + ";");
    codegen.RegisterMaskRegVar(reg_name);
    if (use_b8) {
        codegen.Emit(reg_name + " = plt_b8(" + scalar_var + ", POST_UPDATE);");
    } else if (use_b16) {
        codegen.Emit(reg_name + " = plt_b16(" + scalar_var + ", POST_UPDATE);");
    } else {
        codegen.Emit(reg_name + " = plt_b32(" + scalar_var + ", POST_UPDATE);");
    }
    return "";
}

static std::string EmitVFHistograms(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src, mask]
    CHECK(op->args_.size() == 3) << "vf.histograms requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    // vhist: dst must be uint16_t
    DataType src_dt = GetExprDtype(op->args_[1]);
    // vhist: src must be uint8_t
    CHECK(src_dt == DataType::UINT8) << "vf.histograms source only supports UINT8, got " << DTypeStr(src_dt);
    // Reinterpret src as RegTensor<uint8_t>& if its dtype isn't u8
    std::string bin_type = VFEnumValueName(ir::EnumToString(static_cast<ir::BinType>(op->GetKwarg<int>("bin_type"))));
    std::string bin_const = (bin_type == "BIN1") ? "Bin_N1" : "Bin_N0";
    std::string src_expr = (src_dt == DataType::UINT8) ? src : ("(RegTensor<uint8_t> &)" + src);
    // hist_type: ACCUMULATE (chistv2, default) or FREQUENCY (dhistv2)
    std::string hist_type = "ACCUMULATE";
    if (op->HasKwarg("hist_type")) {
        hist_type = VFEnumValueName(ir::EnumToString(static_cast<ir::HistType>(op->GetKwarg<int>("hist_type"))));
    }
    if (hist_type == "FREQUENCY") {
        codegen.Emit("dhistv2(" + dst + ", " + src_expr + ", " + mask + ", " + bin_const + ");");
    } else {
        codegen.Emit("chistv2(" + dst + ", " + src_expr + ", " + mask + ", " + bin_const + ");");
    }
    return "";
}

static std::string EmitVFCompareImpl(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                     const std::string& cmp_mode)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src0, src1, mask_src]
    CHECK(op->args_.size() == 4) << op->name_ << " requires 4 args (dst, src0, src1, mask)";
    std::string mask_dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask_src = codegen.GetExprAsCode(op->args_[3]);
    // vcmp supports: u8,s8,u16,s16,u32,s32,half,float,bf16,u64,s64 (no bool, no FP8)
    DataType s0_dt = GetExprDtype(op->args_[1]);
    CHECK((s0_dt.IsInt() || s0_dt == DataType::FP16 || s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << op->name_ << " source only supports INT/UINT/FP16/FP32/BF16, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    // Bit width check: in scalar path, uses is_convertible (allows int64 scalar -> int32 reg)
    // so we defer the strict check to the vector-vector path only.
    // Here we just ensure s1 is not a wider type that can't convert (e.g. float vs int).
    bool is_scalar_src = true;
    auto src1_var = ir::As<ir::Var>(op->args_[2]);
    if (src1_var) {
        std::string src1_name = codegen.GetVarName(src1_var);
        is_scalar_src = !codegen.IsRegTensorVar(src1_name);
    }
    if (!is_scalar_src) {
        if (op->HasKwarg("cmp_dtype")) {
            DataType cmp_dt = op->GetKwarg<DataType>("cmp_dtype");
            CHECK((cmp_dt.GetBit() == 8 || cmp_dt.GetBit() == 16 || cmp_dt.GetBit() == 32 || cmp_dt.GetBit() == 64))
                << op->name_ << " cmp_dtype only supports b8/b16/b32/b64 types, got " << DTypeStr(cmp_dt);
        } else {
            CHECK((s0_dt.GetBit() == s1_dt.GetBit()))
                << op->name_ << " requires src0 and src1 to have the same bit width, got src0=" << DTypeStr(s0_dt)
                << " src1=" << DTypeStr(s1_dt) << ". Pass cmp_dtype to compare different-width regs.";
        }
    } else {
        // Scalar compare: is_convertible allows scalar to be wider than reg
        CHECK(s1_dt.GetBit() >= s0_dt.GetBit())
            << op->name_ << " scalar must be convertible to reg type, got src0=" << DTypeStr(s0_dt)
            << " scalar=" << DTypeStr(s1_dt);
    }
    std::string suffix = "eq";
    if (cmp_mode == "NE")
        suffix = "ne";
    else if (cmp_mode == "LT")
        suffix = "lt";
    else if (cmp_mode == "GT")
        suffix = "gt";
    else if (cmp_mode == "GE")
        suffix = "ge";
    else if (cmp_mode == "LE")
        suffix = "le";
    if (is_scalar_src) {
        codegen.Emit("vcmps_" + suffix + "(" + mask_dst + ", " + src0 + ", " + src1 + ", " + mask_src + ");");
    } else {
        DataType canonical = GetExprDtype(op->args_[1]);
        if (op->HasKwarg("cmp_dtype")) {
            canonical = op->GetKwarg<DataType>("cmp_dtype");
        }
        std::string cast_prefix = "(RegTensor<" + canonical.ToCTypeString() + "> &)";
        std::string s0_expr = (s0_dt == canonical) ? src0 : (cast_prefix + src0);
        std::string s1_expr = (s1_dt == canonical) ? src1 : (cast_prefix + src1);
        codegen.Emit("vcmp_" + suffix + "(" + mask_dst + ", " + s0_expr + ", " + s1_expr + ", " + mask_src + ");");
    }
    return "";
}

static std::string EmitVFSqueeze(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, mask]
    CHECK(op->args_.size() == 3) << "vf.squeeze requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    // vsqz requires dst & src to share the same vector element type. Reinterpret
    // src as RegTensor<dst-dtype>& if necessary (mirrors vf_topk.h pattern of
    // `(RegTensor<u32>&)idxC` before passing to Squeeze).
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::INT8 || src_dt == DataType::UINT8 || src_dt == DataType::INT16 ||
           src_dt == DataType::UINT16 || src_dt == DataType::INT32 || src_dt == DataType::UINT32 ||
           src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.squeeze src only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/FP16/FP32, got " << DTypeStr(src_dt);
    std::string dst_expr = dst;
    std::string src_expr = src;
    if (dst_dt != src_dt) {
        src_expr = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)" + src;
    }
    // mode kwarg: "STORED" (default) or "NO_STORED"
    // backward compat: gather_mode="STORE_REG" / "NO_STORE_REG"
    std::string mode = "MODE_STORED";
    if (op->HasKwarg("gather_mode")) {
        auto gm = static_cast<ir::SqueezeMode>(op->GetKwarg<int>("gather_mode"));
        if (gm == ir::SqueezeMode::NO_STORE_REG)
            mode = "MODE_NO_STORED";
        else
            mode = "MODE_STORED";
    }
    codegen.Emit("vsqz(" + dst_expr + ", " + src_expr + ", " + mask + ", " + mode + ");");
    return "";
}

static std::string EmitVFArange(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, start]
    CHECK(op->args_.size() == 2) << "vf.arange requires 2 args (dst, start)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string start = codegen.GetExprAsCode(op->args_[1]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    // vci: supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32
    CHECK(dst_dt == DataType::INT8 || dst_dt == DataType::UINT8 || dst_dt == DataType::INT16 ||
          dst_dt == DataType::UINT16 || dst_dt == DataType::INT32 || dst_dt == DataType::UINT32 ||
          dst_dt == DataType::INT64 || dst_dt == DataType::UINT64 || dst_dt == DataType::FP16 ||
          dst_dt == DataType::FP32)
        << "vf.arange only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32, got "
        << DTypeStr(dst_dt);
    // vci: scalarValue must be convertible to RegTensor data type
    // is_convertible<U, ActualT>() — start must be convertible to dst type.
    // This allows e.g. int64(index) start with int32 dst (narrowing conversion).
    // Only reject if start is smaller than dst (would lose data).
    // index_order kwarg selects the direction: INCREASE_ORDER (default) ->
    // dst[i] = start + i; DECREASE_ORDER -> dst[i] = start - i.
    bool is_decrease = false;
    if (op->HasKwarg("index_order")) {
        auto o = static_cast<ir::IndexOrder>(op->GetKwarg<int>("index_order"));
        if (o == ir::IndexOrder::DECREASE_ORDER)
            is_decrease = true;
    }
    // vci accepts signed integer types (int8/int16/int32) and float types (half/float).
    // Unsigned types (uint8/uint16/uint32) have no vci overload — cast to signed.
    // Signed and float types are passed directly (no cast needed).
    std::string elem_type = dst_dt.ToCTypeString();
    if (dst_dt == DataType::UINT8)
        elem_type = "int8_t";
    else if (dst_dt == DataType::UINT16)
        elem_type = "int16_t";
    else if (dst_dt == DataType::UINT32)
        elem_type = "int32_t";
    // b64 (INT64/UINT64): single vci does not support 8-byte elements.
    // Replicate ArangeB64 using pure bisheng intrinsics:
    //   1. vci int32 low-half (0,1,2,...) into a temp RegTensor<int32_t>
    //   2. vneg if DECREASE_ORDER (produces 0,-1,-2,...)
    //   3. vdup int32 high-half = 0
    //   4. vintlv to interleave low/high into dst (RegTensor<int64_t>)
    //   5. vadds to add the scalar start offset (as b64)
    if (dst_dt == DataType::INT64 || dst_dt == DataType::UINT64) {
        std::string lo = dst + "_b64_lo_";
        std::string hi = dst + "_b64_hi_";
        std::string dump = dst + "_b64_dump_";
        std::string m = dst + "_b64_m_";
        codegen.Emit("RegTensor<int32_t> " + lo + ";");
        codegen.Emit("RegTensor<int32_t> " + hi + ";");
        codegen.Emit("RegTensor<int32_t> " + dump + ";");
        codegen.Emit("MaskReg " + m + " = pset_b32(PAT_ALL);");
        codegen.Emit("vci(" + lo + ", 0, INC_ORDER);");
        if (is_decrease) {
            codegen.Emit("vneg(" + lo + ", " + lo + ", " + m + ", MODE_ZEROING);");
        }
        codegen.Emit("vdup(" + hi + ", 0, " + m + ", MODE_ZEROING);");
        codegen.Emit("vintlv((RegTensor<uint32_t> &)" + dst + ", (RegTensor<uint32_t> &)" + dump +
                     ", (RegTensor<uint32_t> &)" + lo + ", (RegTensor<uint32_t> &)" + hi + ");");
        codegen.Emit("vadds(" + dst + ", " + dst + ", (int64_t)(" + start + "), " + m + ", MODE_ZEROING);");
        return "";
    }
    // Non-b64: vci INC_ORDER generates value, value+1, ..., value+VL-1.
    // vci DEC_ORDER generates value+VL-1, value+VL-2, ..., value.
    std::string order_str = is_decrease ? "DEC_ORDER" : "INC_ORDER";
    if (elem_type != dst_dt.ToCTypeString()) {
        codegen.Emit("vci((RegTensor<" + elem_type + "> &)" + dst + ", " + start + ", " + order_str + ");");
    } else {
        codegen.Emit("vci(" + dst + ", " + start + ", " + order_str + ");");
    }
    return "";
}

static std::string EmitVFGather(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Two forms, dispatched by src argument type:
    //   Reg-to-Reg: args = [dst, src_reg, indices]  — no mask, src/dst same type
    //   UB-to-Reg:  args = [dst, src_ub, indices, mask] — with mask, b8 zero-extends to b16
    auto src_tile_type = ir::As<ir::TileType>(op->args_[1]->GetType());
    if (!src_tile_type) {
        // Reg-to-Reg form
        CHECK(op->args_.size() == 3) << "vf.gather (reg→reg) requires exactly 3 args (dst, src, indices), "
                                     << "mask is not supported in reg→reg form; got " << op->args_.size() << " args";
        CHECK(!op->HasKwarg("data_copy_mode")) << "vf.gather (reg→reg) does not support data_copy_mode "
                                               << "(only Tile→Reg form supports it)";
        // Supports b8/b16/b32; src and dst must have the same type; index bit width
        // must match src. No b64 support (vselr limitation).
        DataType dst_dt = GetExprDtype(op->args_[0]);
        DataType src_dt = GetExprDtype(op->args_[1]);
        DataType idx_dt = GetExprDtype(op->args_[2]);
        CHECK((src_dt.GetBit() == idx_dt.GetBit()))
            << "vf.gather (reg→reg) requires index bit width to match src, got src=" << DTypeStr(src_dt)
            << " index=" << DTypeStr(idx_dt);
        DataType gather_dst_dt = GetExprDtype(op->args_[0]);
        CHECK(src_dt.GetBit() == gather_dst_dt.GetBit())
            << "vf.gather (reg→reg) requires src and dst to have the same bit width, got dst="
            << DTypeStr(gather_dst_dt) << " src=" << DTypeStr(src_dt);
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        std::string src = codegen.GetExprAsCode(op->args_[1]);
        std::string indices = codegen.GetExprAsCode(op->args_[2]);
        std::string cast_type = "uint32_t";
        if (dst_dt.GetBit() <= 8) {
            cast_type = "uint8_t";
        } else if (dst_dt.GetBit() == 16) {
            cast_type = "uint16_t";
        }
        codegen.Emit("vselr((RegTensor<" + cast_type + ">&)" + dst + ", (RegTensor<" + cast_type + ">&)" + src +
                     ", (RegTensor<" + cast_type + ">&)" + indices + ");");
        return "";
    }

    // UB-to-Reg form
    // args: [dst, src_ub, indices, mask]
    CHECK(op->args_.size() == 4) << "vf.gather requires 4 args (dst, src, indices, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType src_dt = GetExprDtype(op->args_[1]);
    DataType idx_dt = GetExprDtype(op->args_[2]);
    std::string indices = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    // Check mode: DATA_BLOCK_LOAD -> block gather, otherwise -> per-element gather
    bool is_datablock = false;
    if (op->HasKwarg("data_copy_mode")) {
        auto mode = static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"));
        CHECK(mode == ir::DataCopyMode::NORM || mode == ir::DataCopyMode::DATA_BLOCK_LOAD)
            << "vf.gather only supports data_copy_mode=NORM or DATA_BLOCK_LOAD, got "
            << VFEnumValueName(ir::EnumToString(mode));
        is_datablock = (mode == ir::DataCopyMode::DATA_BLOCK_LOAD);
        if (!is_datablock) {
            CHECK(!op->HasKwarg("block_stride")) << "vf.gather (NORM mode) does not support block_stride";
        }
    }

    if (is_datablock) {
        // DataCopyGatherB: dst b8/b16/b32/b64, index must be uint32_t
        CHECK(idx_dt == DataType::UINT32)
            << "vf.gather (DATA_BLOCK_LOAD) index must be UINT32, got " << DTypeStr(idx_dt);
        // Block gather: always casts to signed types (s8/s16/s32/s64)
        // and uses int8_t*/int16_t*/int32_t*/int64_t* for the UB pointer.
        std::string signed_c_type;
        if (dst_dt.GetBit() <= 8) {
            signed_c_type = "int8_t";
        } else if (dst_dt.GetBit() == 16) {
            signed_c_type = "int16_t";
        } else if (dst_dt.GetBit() == 32) {
            signed_c_type = "int32_t";
        } else {
            signed_c_type = "int64_t";
        }
        std::string gb_ub_ptr = GetUBufPtr(codegen, op->args_[1], signed_c_type);
        codegen.Emit("vgatherb((RegTensor<" + signed_c_type + ">&)" + dst + ", " + gb_ub_ptr +
                     ", (RegTensor<uint32_t> &)" + indices + ", " + mask + ");");
    } else {
        // DataCopyGather: specific src-dst-index type combinations
        // src b8 -> dst b16 + idx u16; src b16 -> dst b16 + idx u16/u32;
        // src b32 -> dst b32 + idx u32; src b64 -> dst b64 + idx u32/u64
        bool is_b16_src = (dst_dt == DataType::INT16 || dst_dt == DataType::UINT16 || dst_dt == DataType::FP16 ||
                           dst_dt == DataType::BF16);
        bool use_vgather2_bc = is_b16_src && (idx_dt.GetBit() >= 32);
        if (dst_dt.GetBit() == 16) {
            CHECK(src_dt.GetBit() == 8 || src_dt.GetBit() == 16)
                << "vf.gather (NORM) b16 dst requires b8/b16 src, got src=" << DTypeStr(src_dt);
            if (!use_vgather2_bc) {
                CHECK(idx_dt == DataType::UINT16)
                    << "vf.gather (NORM) b16 dst requires UINT16 index (or UINT32 for vgather2_bc), got "
                    << DTypeStr(idx_dt);
            } else {
                CHECK(idx_dt == DataType::UINT32)
                    << "vf.gather (NORM) b16 dst with vgather2_bc requires UINT32 index, got " << DTypeStr(idx_dt);
            }
        } else if (dst_dt.GetBit() == 32) {
            CHECK(src_dt.GetBit() == 32) << "vf.gather (NORM) b32 requires b32 src, got src=" << DTypeStr(src_dt);
            CHECK(idx_dt == DataType::UINT32)
                << "vf.gather (NORM) b32 dst requires UINT32 index, got " << DTypeStr(idx_dt);
        }
        std::string idx_c_type = use_vgather2_bc ? "uint32_t" : ((dst_dt.GetBit() >= 32) ? "uint32_t" : "uint16_t");
        std::string dst_expr = dst;
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], src_dt.ToCTypeString());
        if (dst_dt == DataType::INT8 || dst_dt == DataType::UINT8) {
            dst_expr = "(RegTensor<int16_t>&)" + dst;
            ub_ptr = GetUBufPtr(codegen, op->args_[1], "int8_t");
        } else if (is_b16_src) {
            // vgather2_bc b16 overload expects vector_s16& (RegTensor<int16_t>&).
            // The b16 element occupies the lower 16 bits of each 32-bit slot;
            // the upper 16 bits are zero. Both vgather2 and vgather2_bc use the
            // same dst cast for b16 source types.
            dst_expr = "(RegTensor<int16_t>&)" + dst;
            ub_ptr = GetUBufPtr(codegen, op->args_[1], "int16_t");
        } else if (dst_dt.GetBit() == 32) {
            dst_expr = "(RegTensor<int32_t>&)" + dst;
            ub_ptr = GetUBufPtr(codegen, op->args_[1], "int32_t");
        }
        std::string instr = use_vgather2_bc ? "vgather2_bc" : "vgather2";
        codegen.Emit(instr + "(" + dst_expr + ", " + ub_ptr + ", (RegTensor<" + idx_c_type + "> &)" + indices + ", " +
                     mask + ");");
    }
    return "";
}

static std::string EmitVFStoreUnAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // MaskReg src path: when args[1] is a MaskReg, dispatch to pstu
    bool src_is_mask = false;
    if (auto src_v = ir::As<ir::Var>(op->args_[1])) {
        src_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(src_v));
    }
    if (src_is_mask) {
        CHECK(op->args_.size() == 3) << "vf.store_unalign mask path requires 3 args (ptr, mask, ureg)";
        // DataCopyUnAlign: only supports b16/b32 (SupportBytes<T, 2, 4>)
        DataType tile_dt = GetExprDtype(op->args_[0]);
        CHECK(tile_dt.GetBit() == 16 || tile_dt.GetBit() == 32)
            << "vf.store_unalign (mask path) only supports b16/b32 tile types, got " << DTypeStr(tile_dt);
        std::string vreg = codegen.GetExprAsCode(op->args_[1]);
        DataType mask_dt = GetExprDtype(op->args_[1], DataType::UINT16);
        int elem_bytes = static_cast<int>(mask_dt.GetBit() / 8);
        if (elem_bytes <= 0)
            elem_bytes = 4;
        // pstu only accepts uint16_t* or uint32_t* (DataCopyUnAlignImpl
        // casts to unsigned regardless of template T). b16→uint16_t, b32→uint32_t.
        std::string ptr_type = (elem_bytes <= 2) ? "uint16_t" : "uint32_t";
        // pstu modifies the pointer in-place (*&), so use post-update ref.
        // AscendC signature: pstu(ureg, mask, (__ubuf__ uint32_t*&)dstAddr)
        // The & in the cast is required so pstu advances the cursor, otherwise
        // vstar in store_unalign_post would overwrite pstu's output at the same address.
        std::string tile_ptr = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
        std::string ureg = codegen.GetExprAsCode(op->args_[2]);
        codegen.Emit("pstu(" + ureg + ", " + vreg + ", (__ubuf__ " + ptr_type + " *&)" + tile_ptr + ");");
        return "";
    }
    // Two calling conventions, distinguished by arg count:
    //   3 args [dst, src, align_reg]                    -> vstur (strideless, legacy)
    //   4 args [dst, vreg, ureg, stride] (+post_update) -> vstus (strided)
    // Store-unalign on A5 (dav_3510) only supports signed types for 4/8-byte data
    // (see asc-devkit dav_3510/kernel_reg_compute_datacopy_store_impl.h:495-498).
    // Cast unsigned 32/64-bit src reg + dst ptr to signed int32_t/int64_t.
    CHECK(op->args_.size() >= 3) << "vf.store_unalign requires >=3 args (dst, vreg, ureg[, stride])";
    DataType src_dt = GetExprDtype(op->args_[1]);
    // vstur/vstus support b8/b16/b32/b64 element widths
    CHECK(src_dt.GetBit() == 8 || src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64)
        << "vf.store_unalign source only supports b8/b16/b32/b64 types, got " << DTypeStr(src_dt);
    DataType cast_dt = src_dt;
    if (src_dt == DataType::UINT32) {
        cast_dt = DataType::INT32;
    } else if (src_dt == DataType::UINT16) {
        cast_dt = DataType::INT16;
    } else if (src_dt == DataType::UINT8) {
        cast_dt = DataType::INT8;
    }
    std::string base_c_type = cast_dt.ToCTypeString();
    // Unalign-store cursor: keep separate from a base-pointer load of the same tile.
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], base_c_type, /*is_post_update=*/true);
    std::string vreg = codegen.GetExprAsCode(op->args_[1]);
    std::string ureg = codegen.GetExprAsCode(op->args_[2]);
    // Reinterpret src reg to signed type when needed.
    std::string vreg_expr = (cast_dt == src_dt) ? vreg : ("(RegTensor<" + base_c_type + "> &)" + vreg);

    if (op->args_.size() >= 4) {
        // Strided form -> vstus(ureg, stride, vreg, dst, POST_UPDATE|NORM)
        std::string stride = codegen.GetExprAsCode(op->args_[3]);
        bool post_update = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
        std::string pu = post_update ? "POST_UPDATE" : "NORM";
        codegen.Emit("vstus(" + ureg + ", " + stride + ", " + vreg_expr + ", " + dst_ptr + ", " + pu + ");");
    } else {
        // Strideless legacy form -> vstur(align_reg, src, dst, POST_UPDATE)
        codegen.Emit("vstur(" + ureg + ", " + vreg_expr + ", " + dst_ptr + ", POST_UPDATE);");
    }
    return "";
}

static std::string EmitVFStoreUnAlignPost(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Two calling conventions, distinguished by arg count:
    //   2 args [dst, align_reg]                  -> vstar (strideless, legacy)
    //   3 args [dst, ureg, stride] (+post_update) -> vstas (strided)
    // Match the paired store's pointer type — A5 only supports signed types.
    CHECK(op->args_.size() >= 2) << "vf.store_unalign_post requires >=2 args (dst, ureg[, stride])";
    DataType tile_dt = GetExprDtype(op->args_[0]);
    // DataCopyUnAlignPost: supports b8/b16/b32/b64
    CHECK(IsB8Type(tile_dt) || tile_dt.GetBit() == 16 || tile_dt.GetBit() == 32 || tile_dt.GetBit() == 64)
        << "vf.store_unalign_post only supports b8/b16/b32/b64 types, got " << DTypeStr(tile_dt);
    DataType cast_dt = tile_dt;
    if (tile_dt == DataType::UINT32) {
        cast_dt = DataType::INT32;
    } else if (tile_dt == DataType::UINT16) {
        cast_dt = DataType::INT16;
    } else if (tile_dt == DataType::UINT8) {
        cast_dt = DataType::INT8;
    }
    std::string base_c_type = cast_dt.ToCTypeString();
    // Share the same cursor as the paired store_unalign (is_post_update=true).
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], base_c_type, /*is_post_update=*/true);
    std::string ureg = codegen.GetExprAsCode(op->args_[1]);

    if (op->args_.size() >= 3) {
        // Strided form -> vstas(ureg, dst, stride, POST_UPDATE|NORM)
        std::string stride = codegen.GetExprAsCode(op->args_[2]);
        bool post_update = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
        std::string pu = post_update ? "POST_UPDATE" : "NORM";
        codegen.Emit("vstas(" + ureg + ", " + dst_ptr + ", " + stride + ", " + pu + ");");
    } else {
        // Strideless legacy form -> vstar(align_reg, dst)
        codegen.Emit("vstar(" + ureg + ", " + dst_ptr + ");");
    }
    return "";
}

static std::string EmitVFUnalignRegForStore(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();
    codegen.Emit("UnalignReg " + reg_name + ";");
    return "";
}

static std::string EmitVFClearSpr(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    codegen.Emit("sprclr(SPR_AR);");
    return "";
}

// ============================================================================
// UnalignRegForLoad — declare unaligned load register
// ============================================================================

static std::string EmitVFUnalignRegForLoad(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();
    codegen.Emit("UnalignReg " + reg_name + ";");
    return "";
}

// ============================================================================
// LoadUnalignPre — vldas (setup unaligned load)
// ============================================================================

static std::string EmitVFLoadUnalignPre(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "vf.load_unalign_pre requires 2 args (ureg, src_ptr)";
    std::string ureg = codegen.GetExprAsCode(op->args_[0]);
    DataType dt = GetExprDtype(op->args_[1], DataType::FP32);
    // vldas supports b8/b16/b32/b64 element widths
    CHECK(IsB8Type(dt) || dt.GetBit() == 16 || dt.GetBit() == 32 || dt.GetBit() == 64)
        << "vf.load_unalign_pre only supports b8/b16/b32/b64 types, got " << DTypeStr(dt);
    int elem_bytes = static_cast<int>(dt.GetBit() / 8);
    if (elem_bytes <= 0)
        elem_bytes = 4;
    std::string ptr_type;
    if (elem_bytes == 1) {
        ptr_type = "uint8_t";
    } else if (elem_bytes == 8) {
        ptr_type = "uint32_t";
    } else if (elem_bytes == 4) {
        ptr_type = "int32_t";
    } else {
        if (dt == DataType::FP16 || dt == DataType::BF16)
            ptr_type = "half";
        else
            ptr_type = "uint16_t";
    }
    std::string src_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
    codegen.Emit("vldas(" + ureg + ", " + src_ptr + ");");
    return "";
}

// ============================================================================
// LoadUnalign — vldus (unaligned load body, supports 3-arg and 4-arg strided)
// ============================================================================

static std::string EmitVFLoadUnalign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() >= 3) << "vf.load_unalign requires 3-4 args (dst, ureg, src_ptr [, stride])";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string ureg = codegen.GetExprAsCode(op->args_[1]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    CHECK((IsB8Type(dst_dt) || dst_dt.GetBit() == 16 || dst_dt.GetBit() == 32 || dst_dt.GetBit() == 64))
        << "vf.load_unalign only supports b8/b16/b32/b64 types, got " << DTypeStr(dst_dt);
    // vldus supports b8/b16/b32/b64 element widths
    std::string ptr_type = dst_dt.ToCTypeString();
    if (op->args_.size() >= 4) {
        std::string stride = codegen.GetExprAsCode(op->args_[3]);
        std::string src_ptr = codegen.GetOrCreateVFTilePtr(op->args_[2], /*is_post_update=*/true);
        if (ptr_type != "float") {
            src_ptr = "(__ubuf__ " + ptr_type + " *&)" + src_ptr;
        }
        int elem_bytes = static_cast<int>(dst_dt.GetBit() / 8);
        if (elem_bytes <= 0)
            elem_bytes = 4;
        std::string effective_stride = (elem_bytes == 8) ? ("(" + stride + ") * 2") : stride;
        codegen.Emit("vldus(" + dst + ", " + ureg + ", " + src_ptr + ", " + effective_stride + ", POST_UPDATE);");
    } else {
        std::string src_ptr = GetUBufPtr(codegen, op->args_[2], ptr_type);
        codegen.Emit("vldus(" + dst + ", " + ureg + ", " + src_ptr + ");");
    }
    return "";
}

// ============================================================================
// Scatter — vscatter (scatter store by indices)
// ============================================================================

static std::string EmitVFScatter(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.scatter requires 4 args (base_ptr, src, index, mask)";
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string index = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    DataType src_dt = GetExprDtype(op->args_[1]);
    DataType idx_dt = GetExprDtype(op->args_[2]);
    // Doc: src supports DT_INT8,DT_UINT8,DT_INT16,DT_UINT16,DT_FP16,DT_BF16,
    // DT_INT32,DT_UINT32,DT_FP32,DT_INT64,DT_UINT64
    CHECK(src_dt == DataType::INT8 || src_dt == DataType::UINT8 || src_dt == DataType::BOOL ||
          src_dt == DataType::INT16 || src_dt == DataType::UINT16 || src_dt == DataType::FP16 ||
          src_dt == DataType::BF16 || src_dt == DataType::INT32 || src_dt == DataType::UINT32 ||
          src_dt == DataType::FP32 || src_dt == DataType::INT64 || src_dt == DataType::UINT64)
        << "vf.scatter only supports INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/INT64/UINT64, got "
        << DTypeStr(src_dt);
    if (src_dt.GetBit() == 8 || src_dt.GetBit() == 16) {
        CHECK(idx_dt == DataType::UINT16) << "vf.scatter b8/b16 src requires UINT16 index, got " << DTypeStr(idx_dt);
    } else if (src_dt.GetBit() == 32) {
        CHECK(idx_dt == DataType::UINT32) << "vf.scatter b32 src requires UINT32 index, got " << DTypeStr(idx_dt);
    } else {
        CHECK(idx_dt == DataType::UINT32 || idx_dt == DataType::UINT64)
            << "vf.scatter b64 src requires UINT32/UINT64 index, got " << DTypeStr(idx_dt);
    }
    std::string base_c_type = src_dt.ToCTypeString();
    std::string base_ptr = GetUBufPtr(codegen, op->args_[0], base_c_type);
    std::string idx_c_type = (src_dt == DataType::UINT32) ? "uint32_t" :
                             (src_dt == DataType::INT32)  ? "int32_t" :
                             (src_dt == DataType::FP32)   ? "uint32_t" :
                                                            "uint16_t";
    codegen.Emit("vscatter(" + src + ", " + base_ptr + ", (RegTensor<" + idx_c_type + "> &)" + index + ", " + mask +
                 ");");
    return "";
}

// ============================================================================
// Unsqueeze — vusqz (expand mask bits into register)
// ============================================================================

static std::string EmitVFUnsqueeze(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "vf.unsqueeze requires 2 args (dst, mask)";
    // PrefixSum (vusqz): int8/uint8/int16/uint16/int32/uint32 only
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string mask = codegen.GetExprAsCode(op->args_[1]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    CHECK((dst_dt == DataType::INT8 || dst_dt == DataType::UINT8 || dst_dt == DataType::INT16 ||
           dst_dt == DataType::UINT16 || dst_dt == DataType::INT32 || dst_dt == DataType::UINT32))
        << "vf.unsqueeze dst only supports INT8/UINT8/INT16/UINT16/INT32/UINT32, got " << DTypeStr(dst_dt);
    codegen.Emit("vusqz(" + dst + ", " + mask + ");");
    return "";
}

// ============================================================================
// Truncate — vtrc with ROUND_Z (alias for Round with round_mode=TRUNC)
// ============================================================================

static std::string EmitVFTruncate(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.truncate requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK(src_dt == DataType::FP16 || src_dt == DataType::BF16 || src_dt == DataType::FP32)
        << "vf.truncate src only supports FP16/BF16/FP32, got " << DTypeStr(src_dt);
    DataType vf_truncate_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == vf_truncate_dst_dt) << "vf.truncate requires src and dst to have the same type, got dst="
                                        << DTypeStr(vf_truncate_dst_dt) << " src=" << DTypeStr(src_dt);
    DataType trc_dst_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt == trc_dst_dt) << "vf.truncate requires src and dst to have the same type, got dst="
                                << DTypeStr(trc_dst_dt) << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string round_const = "ROUND_Z";
    if (op->HasKwarg("round_mode")) {
        auto rm = static_cast<ir::VFRoundMode>(op->GetKwarg<int>("round_mode"));
        if (rm == ir::VFRoundMode::CAST_RINT)
            round_const = "ROUND_R";
        else if (rm == ir::VFRoundMode::CAST_CEIL)
            round_const = "ROUND_C";
        else if (rm == ir::VFRoundMode::CAST_FLOOR)
            round_const = "ROUND_F";
        else if (rm == ir::VFRoundMode::CAST_TRUNC)
            round_const = "ROUND_Z";
        else
            CHECK(false) << "vf.truncate only supports round_mode CAST_RINT/CAST_CEIL/CAST_FLOOR/CAST_TRUNC, got "
                         << VFEnumValueName(ir::EnumToString(rm));
    }
    std::string mode = VFZeroingOnly(op, "vf.truncate");
    codegen.Emit("vtrc(" + dst + ", " + src + ", " + round_const + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// MaskGenWithRegTensor — movvp (generate MaskReg from RegTensor bit offset)
// ============================================================================

static std::string EmitVFMaskGenWithRegTensor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 1) << "vf.mask_gen_with_reg_tensor requires 1 arg (src)";
    std::string src = codegen.GetExprAsCode(op->args_[0]);
    std::string mask_dst = codegen.GetCurrentResultTarget();
    codegen.Emit("MaskReg " + mask_dst + ";");
    codegen.RegisterMaskRegVar(mask_dst);
    std::string offset = "0";
    if (op->HasKwarg("offset")) {
        offset = std::to_string(op->GetKwarg<int>("offset"));
    }
    DataType src_dt = GetExprDtype(op->args_[0]);
    CHECK(src_dt.GetBit() == 16 || src_dt.GetBit() == 32)
        << "vf.mask_gen_with_reg_tensor source only supports b16/b32 types, got " << DTypeStr(src_dt);
    // offset must be 0~15 for b16, 0~31 for b32
    int offset_val = 0;
    if (op->HasKwarg("offset")) {
        offset_val = op->GetKwarg<int>("offset");
    }
    if (src_dt.GetBit() == 16) {
        CHECK(offset_val >= 0 && offset_val <= 15)
            << "vf.mask_gen_with_reg_tensor offset must be 0~15 for b16, got " << offset_val;
    } else {
        CHECK(offset_val >= 0 && offset_val <= 31)
            << "vf.mask_gen_with_reg_tensor offset must be 0~31 for b32, got " << offset_val;
    }
    if (src_dt.GetBit() == 16) {
        codegen.Emit("movvp(" + mask_dst + ", (RegTensor<uint16_t> &)" + src + ", " + offset + ");");
    } else {
        codegen.Emit("movvp(" + mask_dst + ", (RegTensor<uint32_t> &)" + src + ", " + offset + ");");
    }
    return "";
}

// ============================================================================
// GetMaskSpr (unified) — movp_b32/movp_b16 with width kwarg
// Replaces: GetMaskSprB32, GetMaskSprB16
// ============================================================================

static std::string EmitVFGetMaskSpr(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();
    std::string width = "B32";
    if (op->HasKwarg("width"))
        width = VFEnumValueName(ir::EnumToString(static_cast<ir::MaskWidth>(op->GetKwarg<int>("width"))));
    // MoveMask: only supports b16/b32 (SupportBytes<T, 2, 4>)
    CHECK(width == "B16" || width == "B32") << "vf.get_mask_spr only supports B16/B32 width, got " << width;
    if (width == "B16")
        codegen.Emit("MaskReg " + reg_name + " = movp_b16();");
    else
        codegen.Emit("MaskReg " + reg_name + " = movp_b32();");
    codegen.RegisterMaskRegVar(reg_name);
    return "";
}

// ============================================================================
// Registration
// ============================================================================

REGISTER_BACKEND_OP(BackendCCE, "vf.reg_tensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFRegTensor(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mask_reg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMaskReg(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.create_mask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCreateMask(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDuplicate(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_align")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoadAlign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_align")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFStoreAlign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMax(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAdd(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSub(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.and_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAnd(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.xor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFXor(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.or_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFOr(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFReduceSum(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFReduceMax(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFReduceMin(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMul(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul_add_dst")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMulAddDst(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDiv(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMuls(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.ln")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLn(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.log")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLog(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMin(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.exp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFExp(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.abs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAbs(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.not_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFNot(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.sqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSqrt(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFRelu(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.neg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFNeg(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAdds(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.subs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSubs(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mins")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMins(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.maxs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMaxs(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.leaky_relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLeakyRelu(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.interleave")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFInterleave(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.pair_reduce_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFPairReduceSum(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.abs_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAbsSub(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.axpy")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAxpy(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul_dst_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMulDstAdd(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.pack")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFPack(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.unpack")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFUnpack(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.prelu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFPRelu(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.shift_left")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFShiftLeft(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.shift_right")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFShiftRight(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mull")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMull(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.addc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAddc(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.subc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSubc(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.exp_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFExpSub(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.astype")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCast(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.de_interleave")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDeInterleave(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.select")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSelect(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.update_mask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFUpdateMask(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mem_bar")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMemBar(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.histograms")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFHistograms(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.eq")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "EQ");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.ne")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "NE");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.lt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "LT");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.gt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "GT");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.le")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "LE");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.ge")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "GE");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.squeeze")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSqueeze(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.arange")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFArange(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.gather")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFGather(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_unalign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFStoreUnAlign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_unalign_post")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFStoreUnAlignPost(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.unalign_reg_for_store")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFUnalignRegForStore(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.clear_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFClearSpr(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign_init")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFUnalignRegForLoad(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign_pre")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoadUnalignPre(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoadUnalign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.scatter")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFScatter(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.unsqueeze")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFUnsqueeze(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.truncate")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFTruncate(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mask_gen_with_reg_tensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFMaskGenWithRegTensor(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.get_mask_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFGetMaskSpr(op, codegen); });

// ============================================================================
// Log2 — composite: vln + vmuls(1/ln2) = ln(x) * 1.4426950408889634
// ============================================================================

static std::string EmitVFLog2(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.log2 requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.log2 src only supports FP16/FP32, got " << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.log2");
    codegen.Emit("vln(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    codegen.Emit("vmuls(" + dst + ", " + dst + ", 1.4426950408889634f, " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Log10 — composite: vln + vmuls(1/ln10) = ln(x) * 0.4342944819032518
// ============================================================================

static std::string EmitVFLog10(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "vf.log10 requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK((src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.log10 src only supports FP16/FP32, got " << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.log10");
    codegen.Emit("vln(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    codegen.Emit("vmuls(" + dst + ", " + dst + ", 0.4342944819032518f, " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// MulsCast — vmulscvt (fused multiply-scalar-cast: dst(fp16) = cast(src(fp32) * scalar))
// ============================================================================

static std::string EmitVFMulsCast(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 4) << "vf.muls_cast requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    CHECK(src_dt == DataType::FP32) << "vf.muls_cast source only supports FP32, got " << DTypeStr(src_dt);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    CHECK(dst_dt == DataType::FP16) << "vf.muls_cast destination only supports FP16, got " << DTypeStr(dst_dt);
    // FusedMulsCast: scalar must be float (Tuple<half, float, float>)
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    CHECK(scalar_dt == DataType::FP32) << "vf.muls_cast scalar only supports FP32, got " << DTypeStr(scalar_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // layout kwarg selects the result half: ZERO -> PART_EVEN (default), ONE -> PART_ODD.
    // FusedMulsCast: only supports RegLayout ZERO/ONE
    std::string part = "PART_EVEN";
    if (op->HasKwarg("layout")) {
        auto layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
        CHECK(layout == "ZERO" || layout == "ONE") << "vf.muls_cast only supports layout ZERO/ONE, got " << layout;
        if (layout == "ONE")
            part = "PART_ODD";
    }
    codegen.Emit("vmulscvt(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + part + ");");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "vf.log2")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLog2(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.log10")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLog10(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.muls_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMulsCast(op, codegen); });

// ============================================================================
// Load (unified) — vldas + vldus all-in-one unaligned load
// ============================================================================

static std::string EmitVFLoad(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() >= 2 && op->args_.size() <= 3) << "vf.load requires 2-3 args (dst, src_ptr[, stride])";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    CHECK((IsB8Type(dst_dt) || dst_dt.GetBit() == 16 || dst_dt.GetBit() == 32 || dst_dt.GetBit() == 64))
        << "vf.load only supports b8/b16/b32/b64 types, got " << DTypeStr(dst_dt);
    // vldas/vldus support b8/b16/b32/b64 element widths
    int elem_bytes = static_cast<int>(dst_dt.GetBit() / 8);
    if (elem_bytes <= 0)
        elem_bytes = 4;
    // vldas requires integer pointer types; vldus requires dst reg type == ptr type.
    // Strategy: use int-type ptr for vldas, native-type ptr for vldus, cast dst for vldus if needed.
    std::string vldas_ptr_type;
    if (elem_bytes == 1) {
        vldas_ptr_type = "uint8_t";
    } else if (elem_bytes == 8) {
        vldas_ptr_type = "uint32_t";
    } else if (elem_bytes == 4) {
        vldas_ptr_type = "int32_t";
    } else {
        if (dst_dt == DataType::FP16 || dst_dt == DataType::BF16)
            vldas_ptr_type = "half";
        else
            vldas_ptr_type = "uint16_t";
    }
    // vldus needs matching dst/ptr types; for float data use float ptr directly.
    std::string vldus_ptr_type = dst_dt.ToCTypeString();
    static int load_counter = 0;
    std::string ureg_name = "__ureg_ld_" + std::to_string(load_counter++);
    codegen.Emit("UnalignReg " + ureg_name + ";");
    if (op->args_.size() == 3) {
        std::string vldas_ptr = GetUBufPtr(codegen, op->args_[1], vldas_ptr_type, /*is_post_update=*/true);
        std::string vldus_ptr = GetUBufPtr(codegen, op->args_[1], vldus_ptr_type, /*is_post_update=*/true);
        std::string stride = codegen.GetExprAsCode(op->args_[2]);
        std::string effective_stride = (elem_bytes == 8) ? ("(" + stride + ") * 2") : stride;
        std::string post_mode = "POST_UPDATE";
        if (op->HasKwarg("post_mode")) {
            post_mode = op->GetKwarg<std::string>("post_mode");
        }
        codegen.Emit("vldas(" + ureg_name + ", " + vldas_ptr + ");");
        codegen.Emit("vldus(" + dst + ", " + ureg_name + ", " + vldus_ptr + ", " + effective_stride + ", " + post_mode +
                     ");");
    } else {
        std::string vldas_ptr = GetUBufPtr(codegen, op->args_[1], vldas_ptr_type);
        std::string vldus_ptr = GetUBufPtr(codegen, op->args_[1], vldus_ptr_type);
        codegen.Emit("vldas(" + ureg_name + ", " + vldas_ptr + ");");
        codegen.Emit("vldus(" + dst + ", " + ureg_name + ", " + vldus_ptr + ");");
    }
    return "";
}

// ============================================================================
// Store (unified) — vstus + vstas all-in-one unaligned store
// ============================================================================

static std::string EmitVFStore(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() >= 2 && op->args_.size() <= 3) << "vf.store requires 2-3 args (dst_ptr, src[, count])";
    DataType src_dt = GetExprDtype(op->args_[1]);
    // vstus/vstas support b8/b16/b32/b64 element widths
    CHECK(IsB8Type(src_dt) || src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64)
        << "vf.store only supports b8/b16/b32/b64 types, got " << DTypeStr(src_dt);
    int elem_bytes = static_cast<int>(src_dt.GetBit() / 8);
    if (elem_bytes <= 0)
        elem_bytes = 4;
    // vstus requires matching src reg type and dst pointer type (same as vldus).
    // Use native type for the pointer; no cast on the src register.
    std::string ptr_type = src_dt.ToCTypeString();
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type, /*is_post_update=*/true);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string count;
    if (op->args_.size() == 3) {
        count = codegen.GetExprAsCode(op->args_[2]);
    } else if (op->HasKwarg("count")) {
        count = std::to_string(op->GetKwarg<int>("count"));
    } else {
        count = std::to_string(256 / elem_bytes);
    }
    std::string effective_count = (elem_bytes == 8) ? ("(" + count + ") * 2") : count;
    std::string post_mode = "POST_UPDATE";
    if (op->HasKwarg("post_mode")) {
        post_mode = op->GetKwarg<std::string>("post_mode");
    }
    static int store_counter = 0;
    std::string ureg_name = "__ureg_st_" + std::to_string(store_counter++);
    codegen.Emit("UnalignReg " + ureg_name + ";");
    codegen.Emit("vstus(" + ureg_name + ", " + effective_count + ", " + src + ", " + dst_ptr + ", " + post_mode + ");");
    codegen.Emit("vstas(" + ureg_name + ", " + dst_ptr + ", 0, " + post_mode + ");");
    return "";
}

// EmitVFMaskLoad/Store/StoreUnalign have been removed — their logic is now
// unified into EmitVFLoadAlign/EmitVFStoreAlign/EmitVFStoreUnAlign via
// IsMaskRegVar dispatch, matching function-overloading model.

REGISTER_BACKEND_OP(BackendCCE, "vf.load")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoad(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFStore(op, codegen); });

// mask_load/mask_store/mask_store_unalign backend registrations removed —
// the parser redirects these to vf.load_align/vf.store_align/vf.store_unalign
// which dispatch via IsMaskRegVar.

// ============================================================================
// CreateAddrReg — AddrReg declaration + vag_b8/b16/b32 intrinsic
// ============================================================================

static std::string EmitVFCreateAddrReg(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() >= 2 && op->args_.size() <= 8 && op->args_.size() % 2 == 0)
        << "vf.create_addr_reg requires 2-8 args (1-4 index/stride pairs)";
    std::string reg_name = codegen.GetCurrentResultTarget();
    // Determine element width from dtype kwarg (default b32)
    DataType dt = DataType::FP32;
    if (op->HasKwarg("dtype")) {
        dt = op->GetKwarg<DataType>("dtype");
    }
    // vag_b8/b16/b32 support b8/b16/b32/b64 element widths (b64 uses vag_b32 with doubled stride)
    CHECK(IsB8Type(dt) || dt.GetBit() == 16 || dt.GetBit() == 32 || dt.GetBit() == 64)
        << "vf.create_addr_reg dtype must be b8/b16/b32/b64, got " << DTypeStr(dt);
    std::string vag_fn;
    if (dt == DataType::UINT8 || dt == DataType::INT8)
        vag_fn = "vag_b8";
    else if (dt.GetBit() == 16)
        vag_fn = "vag_b16";
    else
        vag_fn = "vag_b32";
    // Collect stride args (every 2nd arg). For b64, each stride is doubled.
    std::string stride_args;
    for (size_t i = 1; i < op->args_.size(); i += 2) {
        std::string stride = codegen.GetExprAsCode(op->args_[i]);
        if (dt == DataType::UINT64 || dt == DataType::INT64) {
            stride = "(" + stride + ") * 2";
        }
        if (!stride_args.empty())
            stride_args += ", ";
        stride_args += stride;
    }
    // AddrReg (vector_address) must be declared and initialized in a single
    // statement (bisheng rejects a separate declaration + assignment). Emit the
    // declaration and vag_* initializer together, matching the
    // `AddrReg x = CreateAddrReg<T>(...)` usage. The vag_* must sit inside the
    // physical loop it is bound to (HardwareLoop).
    codegen.Emit("AddrReg " + reg_name + " = " + vag_fn + "(" + stride_args + ");");
    codegen.RegisterAddrRegVar(reg_name);
    return "";
}

// ============================================================================
// Move — vmov (RegTensor) / pmov (MaskReg), with or without mask
// ============================================================================

static std::string EmitVFMove(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2 || op->args_.size() == 3)
        << "vf.move requires 2 args (dst, src) or 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    // Detect dst type: MaskReg vs RegTensor
    bool is_mask_dst = false;
    auto dst_var = ir::As<ir::Var>(op->args_[0]);
    if (dst_var) {
        is_mask_dst = codegen.IsMaskRegVar(codegen.GetVarName(dst_var));
    }
    if (!is_mask_dst) {
        DataType src_dt = GetExprDtype(op->args_[1]);
        CHECK(src_dt == DataType::INT8 || src_dt == DataType::UINT8 || src_dt == DataType::BOOL ||
              src_dt == DataType::INT16 || src_dt == DataType::UINT16 || src_dt == DataType::FP16 ||
              src_dt == DataType::BF16 || src_dt == DataType::INT32 || src_dt == DataType::UINT32 ||
              src_dt == DataType::FP32)
            << "vf.move src only supports BOOL/INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32, got "
            << DTypeStr(src_dt);
        DataType vf_move_dst_dt = GetExprDtype(op->args_[0]);
        CHECK(src_dt == vf_move_dst_dt) << "vf.move requires src and dst to have the same type, got dst="
                                        << DTypeStr(vf_move_dst_dt) << " src=" << DTypeStr(src_dt);
    }
    if (op->args_.size() == 3) {
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        if (is_mask_dst) {
            codegen.Emit("pmov(" + dst + ", " + src + ", " + mask + ");");
        } else {
            // vf.move only supports MERGING mode (AscendC Copy/Move default).
            if (op->HasKwarg("mode")) {
                auto mode_val = static_cast<ir::MergeMode>(op->GetKwarg<int>("mode"));
                CHECK(mode_val == ir::MergeMode::MERGING) << "vf.move only supports MERGING mode on current device";
            }
            std::string mode = "MODE_MERGING";
            codegen.Emit("vmov(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
        }
    } else {
        if (is_mask_dst) {
            codegen.Emit("pmov(" + dst + ", " + src + ");");
        } else {
            codegen.Emit("vmov(" + dst + ", " + src + ");");
        }
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "vf.create_addr_reg")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCreateAddrReg(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.move")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMove(op, codegen); });

// ============================================================================
// BitCast — type reinterpretation (no instruction, just C++ reference cast)
// ============================================================================

static std::string EmitVFBitCast(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 1 || op->args_.size() == 2) << "vf.bit_cast requires 1 arg (src) or 2 args (dst, src)";
    DataType target_dt = op->GetKwarg<DataType>("dtype");
    std::string src = codegen.GetExprAsCode(op->args_.back());
    if (op->args_.size() == 2) {
        // Assignment form: dst = vf.bit_cast(src, dtype=xxx)
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        codegen.Emit(dst + " = (RegTensor<" + target_dt.ToCTypeString() + ">&)" + src + ";");
    } else {
        // Nested form: vf.bit_cast(src, dtype=xxx) used as expression argument.
        // Return the cast expression directly so it inlines into the parent op's
        // instruction call, matching the documented behaviour:
        //   vor(reg_c, (RegTensor<uint8_t>&)reg_a, (RegTensor<uint8_t>&)reg_b, preg, MODE_ZEROING);
        // When the parser materializes the result into a temp variable (_expr_tmp_N),
        // VisitStmt_(AssignStmtPtr) emits "auto _expr_tmp_N = (RegTensor<T>&)src;"
        // which properly declares the variable.
        return "(RegTensor<" + target_dt.ToCTypeString() + "> &)" + src;
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "vf.bit_cast")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFBitCast(op, codegen); });

} // namespace backend
} // namespace pypto
