/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backend/backend_cce.h"
#include "codegen/cce/cce_codegen.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace backend {
namespace {

using Kwargs = std::vector<std::pair<std::string, std::any>>;

class CapturingCCECodegen final : public codegen::CCECodegen {
public:
    using codegen::CCECodegen::CCECodegen;
    void SetTarget(std::string target) { target_ = std::move(target); }

    [[nodiscard]] std::string GetCurrentResultTarget() const override { return target_; }

    void Emit(const std::string& line) override
    {
        emitted_ += line;
        emitted_ += '\n';
    }

    std::string GetExprAsCode(const ir::ExprPtr& expr) override
    {
        if (auto var = ir::As<ir::Var>(expr)) {
            return var->name_;
        }
        if (auto value = ir::As<ir::ConstInt>(expr)) {
            return std::to_string(value->value_);
        }
        if (auto value = ir::As<ir::ConstFloat>(expr)) {
            return std::to_string(value->value_);
        }
        return codegen::CCECodegen::GetExprAsCode(expr);
    }

    std::string GetVarName(const ir::VarPtr& var) override { return var->name_; }

    [[nodiscard]] const std::string& Emitted() const { return emitted_; }

private:
    std::string target_{"result"};
    std::string emitted_;
};

ir::TypePtr ScalarType(ir::DataType dtype) { return std::make_shared<const ir::ScalarType>(dtype); }

ir::VarPtr MakeVar(const std::string& name, ir::DataType dtype = ir::DataType::FP32)
{
    return std::make_shared<const ir::Var>(name, ScalarType(dtype), ir::Span::Unknown());
}

ir::VarPtr MakeTile(const std::string& name, ir::DataType dtype = ir::DataType::FP32)
{
    auto type = std::make_shared<const ir::TileType>(std::vector<int64_t>{16, 16}, dtype,
                                                     std::optional<ir::MemRefPtr>(std::nullopt),
                                                     std::optional<ir::TileView>(std::nullopt));
    return std::make_shared<const ir::Var>(name, type, ir::Span::Unknown());
}

ir::ExprPtr Int(int64_t value)
{
    return std::make_shared<const ir::ConstInt>(value, ir::DataType::INT64, ir::Span::Unknown());
}

ir::ExprPtr Float(double value)
{
    return std::make_shared<const ir::ConstFloat>(value, ir::DataType::FP32, ir::Span::Unknown());
}

int EnumValue(ir::MergeMode value) { return static_cast<int>(value); }

template <typename Enum>
int EnumValue(Enum value)
{
    return static_cast<int>(value);
}

ir::CallPtr MakeCall(const std::string& name, std::vector<ir::ExprPtr> args = {}, Kwargs kwargs = {})
{
    return std::make_shared<const ir::Call>(name, std::move(args), std::move(kwargs), ir::Span::Unknown());
}

std::string Invoke(CapturingCCECodegen& codegen, const std::string& name, std::vector<ir::ExprPtr> args = {},
                   Kwargs kwargs = {}, const std::string& target = "result")
{
    const std::size_t emitted_size = codegen.Emitted().size();
    const auto* info = BackendCCE::Instance().GetOpInfo(name);
    EXPECT_NE(info, nullptr) << name;
    if (info == nullptr) {
        return "";
    }
    EXPECT_EQ(info->pipe, ir::PipeType::V) << name;
    codegen.SetTarget(target);
    EXPECT_TRUE(info->codegen_func(MakeCall(name, std::move(args), std::move(kwargs)), codegen).empty());
    return codegen.Emitted().substr(emitted_size);
}

void ExpectContains(const std::string& generated, const std::vector<std::string>& fragments)
{
    for (const auto& fragment : fragments) {
        EXPECT_NE(generated.find(fragment), std::string::npos) << fragment << "\n" << generated;
    }
}

void ExpectInvoke(CapturingCCECodegen& codegen, const std::string& name, const std::vector<std::string>& expected,
                  std::vector<ir::ExprPtr> args = {}, Kwargs kwargs = {}, const std::string& target = "result")
{
    SCOPED_TRACE(name);
    ExpectContains(Invoke(codegen, name, std::move(args), std::move(kwargs), target), expected);
}

TEST(BackendCCEVFOpsTest, RegistersExpectedVectorFunctionOperations)
{
    const std::vector<std::string> names = {"vf.reg_tensor",
                                            "vf.create_mask",
                                            "vf.full",
                                            "vf.load_align",
                                            "vf.store_align",
                                            "vf.max",
                                            "vf.add",
                                            "vf.sub",
                                            "vf.and_",
                                            "vf.xor",
                                            "vf.or_",
                                            "vf.reduce_sum",
                                            "vf.reduce_max",
                                            "vf.reduce_min",
                                            "vf.mul",
                                            "vf.mul_add_dst",
                                            "vf.div",
                                            "vf.muls",
                                            "vf.ln",
                                            "vf.log",
                                            "vf.min",
                                            "vf.exp",
                                            "vf.abs",
                                            "vf.not_",
                                            "vf.sqrt",
                                            "vf.relu",
                                            "vf.neg",
                                            "vf.adds",
                                            "vf.subs",
                                            "vf.mins",
                                            "vf.maxs",
                                            "vf.leaky_relu",
                                            "vf.interleave",
                                            "vf.pair_reduce_sum",
                                            "vf.abs_sub",
                                            "vf.axpy",
                                            "vf.mul_dst_add",
                                            "vf.pack",
                                            "vf.unpack",
                                            "vf.prelu",
                                            "vf.shift_left",
                                            "vf.shift_right",
                                            "vf.mull",
                                            "vf.addc",
                                            "vf.subc",
                                            "vf.exp_sub",
                                            "vf.astype",
                                            "vf.de_interleave",
                                            "vf.select",
                                            "vf.update_mask",
                                            "vf.mem_bar",
                                            "vf.histograms",
                                            "vf.eq",
                                            "vf.ne",
                                            "vf.lt",
                                            "vf.gt",
                                            "vf.le",
                                            "vf.ge",
                                            "vf.squeeze",
                                            "vf.arange",
                                            "vf.gather",
                                            "vf.store_unalign",
                                            "vf.store_unalign_post",
                                            "vf.unalign_reg_for_store",
                                            "vf.clear_spr",
                                            "vf.load_unalign_init",
                                            "vf.load_unalign_pre",
                                            "vf.load_unalign",
                                            "vf.scatter",
                                            "vf.unsqueeze",
                                            "vf.truncate",
                                            "vf.mask_gen_with_reg_tensor",
                                            "vf.get_mask_spr",
                                            "vf.log2",
                                            "vf.log10",
                                            "vf.muls_cast",
                                            "vf.load",
                                            "vf.store",
                                            "vf.create_addr_reg",
                                            "vf.move"};

    for (const auto& name : names) {
        const auto* info = BackendCCE::Instance().GetOpInfo(name);
        ASSERT_NE(info, nullptr) << name;
        EXPECT_EQ(info->pipe, ir::PipeType::V) << name;
        EXPECT_TRUE(static_cast<bool>(info->codegen_func)) << name;
    }
}

TEST(BackendCCEVFOpsTest, EmitsDeclarationsMasksBroadcastsAndMoves)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp = MakeVar("fp");
    auto s4 = MakeVar("s4", ir::DataType::INT4);
    auto u4 = MakeVar("u4", ir::DataType::UINT4);
    auto i32_reg = MakeVar("i32_reg", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto mask2 = MakeVar("mask2", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);

    Invoke(codegen, "vf.reg_tensor", {}, {{"dtype", ir::DataType::FP32}}, "fp");
    Invoke(codegen, "vf.reg_tensor", {}, {{"dtype", ir::DataType::INT4}}, "s4");
    Invoke(codegen, "vf.reg_tensor", {}, {{"dtype", ir::DataType::UINT4}}, "u4");
    Invoke(codegen, "vf.reg_tensor", {}, {{"dtype", ir::DataType::INT32}}, "i32_reg");
    EXPECT_TRUE(codegen.IsRegTensorVar("fp"));
    EXPECT_TRUE(codegen.IsRegTensorVar("s4"));
    EXPECT_TRUE(codegen.IsRegTensorVar("u4"));

    ExpectInvoke(codegen, "vf.create_mask", {"MaskReg mask;", "mask = pset_b8(PAT_VL16);"}, {},
                 {{"pattern", EnumValue(ir::MaskPattern::VL16)}, {"dtype", ir::DataType::INT8}}, "mask");
    ExpectInvoke(codegen, "vf.create_mask", {"MaskReg mask2;", "mask2 = pset_b16(PAT_ALL);"}, {},
                 {{"dtype", ir::DataType::FP16}}, "mask2");
    ExpectInvoke(codegen, "vf.create_mask", {"MaskReg mask_default;", "mask_default = pset_b32(PAT_ALL);"}, {}, {},
                 "mask_default");
    ExpectInvoke(codegen, "vf.create_mask", {"pset_b8("}, {},
                 {{"pattern", EnumValue(ir::MaskPattern::ALL)}, {"dtype", ir::DataType::FP8E4M3FN}}, "mask_fp8");
    ExpectInvoke(codegen, "vf.create_mask", {"pset_b8("}, {},
                 {{"pattern", EnumValue(ir::MaskPattern::ALL)}, {"dtype", ir::DataType::FP4E2M1}}, "mask_fp4");
    EXPECT_TRUE(codegen.IsMaskRegVar("mask"));

    ExpectInvoke(codegen, "vf.full", {"vbr(fp, 2.500000);"}, {fp, Float(2.5)});
    ExpectInvoke(codegen, "vf.full", {"vdup(fp, 1.000000, mask, MODE_MERGING);"}, {fp, Float(1.0), mask},
                 {{"mode", EnumValue(ir::MergeMode::MERGING)}});
    codegen.RegisterRegTensorVar("s4");
    ExpectInvoke(codegen, "vf.full", {"POS_HIGHEST", "MODE_ZEROING"}, {fp, s4},
                 {{"pos", EnumValue(ir::DuplicatePos::HIGHEST)}});

    ExpectInvoke(codegen, "vf.create_addr_reg", {"AddrReg addr = vag_b32((4) * 2, (8) * 2);"},
                 {Int(0), Int(4), Int(1), Int(8)}, {{"dtype", ir::DataType::INT64}}, "addr");
    EXPECT_TRUE(codegen.IsAddrRegVar("addr"));
    ExpectInvoke(codegen, "vf.move", {"vmov(fp, fp);"}, {fp, fp});
    ExpectInvoke(codegen, "vf.move", {"vmov(fp, fp, mask, MODE_MERGING);"}, {fp, fp, mask});
    ExpectInvoke(codegen, "vf.move", {"pmov(mask2, mask);"}, {mask2, mask});
    ExpectInvoke(codegen, "vf.move", {"pmov(mask2, mask, mask);"}, {mask2, mask, mask});
}

TEST(BackendCCEVFOpsTest, EmitsArithmeticIntrinsics)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst");
    auto src0 = MakeVar("src0");
    auto src1 = MakeVar("src1");
    auto int_dst = MakeVar("int_dst", ir::DataType::INT32);
    auto int_src0 = MakeVar("int_src0", ir::DataType::INT32);
    auto int_src1 = MakeVar("int_src1", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto carry = MakeVar("carry", ir::DataType::UINT32);

    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};
    const auto expect_binary = [&](const std::string& name, const std::string& intrinsic) {
        ExpectInvoke(codegen, name, {intrinsic}, {dst, src0, src1, mask}, zeroing);
    };
    expect_binary("vf.max", "vmax(");
    expect_binary("vf.add", "vadd(");
    expect_binary("vf.sub", "vsub(");
    expect_binary("vf.and_", "vand(");
    expect_binary("vf.xor", "vxor(");
    expect_binary("vf.or_", "vor(");
    expect_binary("vf.mul", "vmul(");
    expect_binary("vf.mul_add_dst", "vmula(");
    expect_binary("vf.div", "vdiv(");
    expect_binary("vf.min", "vmin(");
    expect_binary("vf.abs_sub", "vabsdif(");
    expect_binary("vf.mul_dst_add", "vmadd(");
    expect_binary("vf.prelu", "vprelu(");

    const auto expect_unary = [&](const std::string& name, const std::vector<std::string>& expected) {
        ExpectInvoke(codegen, name, expected, {dst, src0, mask}, zeroing);
    };
    expect_unary("vf.ln", {"vln("});
    expect_unary("vf.log", {"vln("});
    expect_unary("vf.exp", {"vexp("});
    expect_unary("vf.abs", {"vabs("});
    expect_unary("vf.not_", {"vnot("});
    expect_unary("vf.sqrt", {"vsqrt("});
    ExpectInvoke(codegen, "vf.sqrt", {"vsqrt<float, true>("}, {dst, src0, mask}, {{"precision", true}});
    expect_unary("vf.relu", {"vrelu("});
    expect_unary("vf.neg", {"vneg("});
    expect_unary("vf.log2", {"vln(", "1.4426950408889634f"});
    expect_unary("vf.log10", {"vln(", "0.4342944819032518f"});

    const auto expect_scalar = [&](const std::string& name, const std::string& intrinsic) {
        ExpectInvoke(codegen, name, {intrinsic}, {dst, src0, Float(0.5), mask}, zeroing);
    };
    expect_scalar("vf.muls", "vmuls(");
    expect_scalar("vf.adds", "vadds(");
    expect_scalar("vf.subs", "vadds(");
    expect_scalar("vf.mins", "vmins(");
    expect_scalar("vf.maxs", "vmaxs(");
    expect_scalar("vf.leaky_relu", "vlrelu(");
    auto fp32_src = MakeVar("fp32_src", ir::DataType::FP32);
    auto fp16_dst = MakeVar("fp16_dst", ir::DataType::FP16);
    codegen.RegisterRegTensorVar("fp32_src");
    codegen.RegisterRegTensorVar("fp16_dst");
    ExpectInvoke(codegen, "vf.muls_cast", {"vmulscvt("}, {fp16_dst, fp32_src, Float(0.5), mask},
                 {{"mode", EnumValue(ir::MergeMode::ZEROING)}, {"dtype", ir::DataType::FP16}});
    auto f8_src = MakeVar("f8_src", ir::DataType::FP8E4M3FN);
    codegen.RegisterRegTensorVar("f8_src");
    ExpectInvoke(codegen, "vf.full", {"vdup("}, {dst, f8_src, mask},
                 {{"mode", EnumValue(ir::MergeMode::MERGING)}, {"pos", EnumValue(ir::DuplicatePos::LOWEST)}});
}

TEST(BackendCCEVFOpsTest, EmitsReductionAndPermutationIntrinsics)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst");
    auto dst2 = MakeVar("dst2");
    auto src0 = MakeVar("src0");
    auto src1 = MakeVar("src1");
    auto int_dst = MakeVar("int_dst", ir::DataType::INT32);
    auto int_src0 = MakeVar("int_src0", ir::DataType::INT32);
    auto int_src1 = MakeVar("int_src1", ir::DataType::INT32);
    auto int_src = MakeVar("int_src", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto carry = MakeVar("carry", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("int_src");
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};

    ExpectInvoke(codegen, "vf.reduce_sum", {"vcgadd("}, {dst, src0, mask},
                 {{"datablock", true}, {"merge_mode", EnumValue(ir::MergeMode::ZEROING)}});
    ExpectInvoke(codegen, "vf.reduce_max", {"vcmax("}, {dst, src0, mask});
    ExpectInvoke(codegen, "vf.reduce_min", {"vcgmin("}, {dst, src0, mask}, {{"datablock", true}});
    ExpectInvoke(codegen, "vf.interleave", {"vintlv("}, {dst, dst2, src0, src1});
    ExpectInvoke(codegen, "vf.de_interleave", {"vdintlv("}, {dst, dst2, src0, int_src});
    ExpectInvoke(codegen, "vf.pair_reduce_sum", {"vcpadd("}, {dst, src0, mask}, zeroing);
    ExpectInvoke(codegen, "vf.axpy", {"vaxpy("}, {dst, src0, Float(0.25), mask}, zeroing);
    ExpectInvoke(codegen, "vf.shift_left", {"vshls("}, {int_dst, int_src0, Int(2), mask}, zeroing);
    ExpectInvoke(codegen, "vf.shift_right", {"vshr("}, {int_dst, int_src0, int_src, mask}, zeroing);
    ExpectInvoke(codegen, "vf.mull", {"vmull("}, {int_dst, int_dst, int_src0, int_src1, mask});
    ExpectInvoke(codegen, "vf.addc", {"vaddcs("}, {carry, int_dst, int_src0, int_src1, mask, mask});
    ExpectInvoke(codegen, "vf.subc", {"vsubcs("}, {carry, int_dst, int_src0, int_src1, mask, mask});
    ExpectInvoke(codegen, "vf.exp_sub", {"vexpdif(", "PART_ODD"}, {dst, src0, src1, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)}, {"dtype", ir::DataType::FP32}});
    ExpectInvoke(codegen, "vf.select", {"vsel("}, {dst, src0, int_src, mask});
    ExpectInvoke(codegen, "vf.mem_bar", {"mem_bar(VV_ALL)"}, {}, {{"mode", EnumValue(ir::MemBarMode::VV_ALL)}});
}

TEST(BackendCCEVFOpsTest, EmitsPackAndCastIntrinsics)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto bf16 = MakeVar("bf16", ir::DataType::BF16);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto i32 = MakeVar("i32", ir::DataType::INT32);
    auto i16 = MakeVar("i16", ir::DataType::INT16);
    auto i8 = MakeVar("i8", ir::DataType::INT8);
    auto u16 = MakeVar("u16", ir::DataType::UINT16);
    auto u32 = MakeVar("u32", ir::DataType::UINT32);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto s4 = MakeVar("s4", ir::DataType::INT4);
    auto f8e4m3 = MakeVar("f8e4m3", ir::DataType::FP8E4M3FN);
    auto f8e5m2 = MakeVar("f8e5m2", ir::DataType::FP8E5M2);
    auto hf8 = MakeVar("hf8", ir::DataType::HF8);
    auto f4e2m1 = MakeVar("f4e2m1", ir::DataType::FP4E2M1);
    auto f4e1m2 = MakeVar("f4e1m2", ir::DataType::FP4E1M2);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    for (const auto& var :
         {fp32, fp16, bf16, i64, i32, i16, i8, u16, u32, u8, s4, f8e4m3, f8e5m2, hf8, f4e2m1, f4e1m2}) {
        codegen.RegisterRegTensorVar(var->name_);
    }

    ExpectInvoke(codegen, "vf.pack", {"vpack(", "HIGHER"}, {u16, u32},
                 {{"part", EnumValue(ir::PackPart::UPPER)}, {"dtype", ir::DataType::UINT16}});
    ExpectInvoke(codegen, "vf.pack", {"vdintlv("}, {i32, i64}, {{"dtype", ir::DataType::INT32}});
    ExpectInvoke(codegen, "vf.unpack", {"vunpack(", "HIGHER"}, {u32, u16},
                 {{"part", EnumValue(ir::PackPart::UPPER)}, {"dtype", ir::DataType::UINT32}});
    ExpectInvoke(codegen, "vf.unpack", {"vintlv("}, {i64, i32}, {{"dtype", ir::DataType::INT64}});

    const Kwargs cast_options = {{"layout", EnumValue(ir::CastLayout::ONE)},
                                 {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                                 {"saturate", EnumValue(ir::SaturateMode::ON)}};
    auto with_dtype = [](const Kwargs& base, ir::DataType dt) {
        Kwargs result = base;
        result.emplace_back("dtype", dt);
        return result;
    };
    // FP16→FP32 (2x widening): no round_mode/saturate
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp32, fp16, mask, PART_ODD, MODE_ZEROING);"}, {fp32, fp16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)}, {"dtype", ir::DataType::FP32}});
    // INT32→FP32 (same-width int→float): layout=ZERO, no saturate
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp32, i32, mask, ROUND_F, MODE_ZEROING);"}, {fp32, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"dtype", ir::DataType::FP32}});
    // FP32→INT32 (same-width float→int): layout=ZERO
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, fp32, mask, ROUND_F, RS_ENABLE, MODE_ZEROING);"}, {i32, fp32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT32}});
    // FP16→INT32 (wider int): uses ROUND + PART
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, fp16, mask, ROUND_F, PART_ODD, MODE_ZEROING);"}, {i32, fp16, mask},
                 with_dtype(cast_options, ir::DataType::INT32));
    // INT32→FP16 (cross-width): uses ROUND + PART
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp16, i32, mask, ROUND_F, PART_ODD, MODE_ZEROING);"}, {fp16, i32, mask},
                 with_dtype(cast_options, ir::DataType::FP16));
    // INT32→INT16 (int narrowing): no round_mode, uses RS + PART
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i16, i32, mask, RS_ENABLE, PART_ODD, MODE_ZEROING);"}, {i16, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT16}});
    // INT4→FP16 (s4 widening): no round_mode/saturate
    ExpectInvoke(codegen, "vf.astype", {"vcvt_s42f16(fp16, s4, mask, PART_P1, MODE_ZEROING);"}, {fp16, s4, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)}, {"dtype", ir::DataType::FP16}});
    ExpectInvoke(codegen, "vf.astype", {"vcvt_f162s4(s4, fp16, mask, ROUND_F, RS_ENABLE, PART_P1, MODE_ZEROING);"},
                 {s4, fp16, mask}, with_dtype(cast_options, ir::DataType::INT4));
    // INT16→INT4 (two-step: s16→f16→s4, mirroring AscendC Cast)
    // Default layout=ZERO → PART_P0, default saturate=OFF → RS_DISABLE
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(s4_f16_tmp, i16, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt_f162s4(s4, s4_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);"},
                 {s4, i16, mask}, {{"dtype", ir::DataType::INT4}});
    // int→int two-step through f16 (cross-sign widening, int→INT8 narrowing)
    // UINT8→INT16: u8→f16 (widening, 5 args) → f16→s16 (same-int, 5 args)
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i16_f16_tmp, u8, mask, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i16, i16_f16_tmp, mask, ROUND_R, RS_DISABLE, MODE_ZEROING);"},
                 {i16, u8, mask}, {{"dtype", ir::DataType::INT16}});
    // INT32→INT8: three-step s32→f32→f16→s8
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i8_f32_tmp, i32, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt(i8_f16_tmp, i8_f32_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i8, i8_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, i32, mask}, {{"dtype", ir::DataType::INT8}});
    // UINT16→INT32: u16→u32 widening with dst cast (zero-extend, semantically correct)
    ExpectInvoke(codegen, "vf.astype", {"vcvt((RegTensor<uint32_t> &)i32, u16, mask, PART_EVEN, MODE_ZEROING);"},
                 {i32, u16, mask}, {{"dtype", ir::DataType::INT32}});
    // FP8/FP4 low-precision conversions
    // 4x widening PP (5-arg): FP8→FP32, FP4→BF16 → vcvt(dst,src,mask,PART_PP,MODE)
    // Widening paths: round_mode and saturate are not applicable
    const Kwargs widen_options = {{"layout", EnumValue(ir::CastLayout::ONE)}};
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp32, f8e4m3, mask, PART_P1, MODE_ZEROING);"}, {fp32, f8e4m3, mask},
                 widen_options);
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp32, f8e5m2, mask, PART_P1, MODE_ZEROING);"}, {fp32, f8e5m2, mask},
                 widen_options);
    ExpectInvoke(codegen, "vf.astype", {"vcvt(bf16, f4e2m1, mask, PART_P1, MODE_ZEROING);"}, {bf16, f4e2m1, mask},
                 widen_options);
    // 2x widening PART (5-arg): HF8→FP16 → vcvt(dst,src,mask,PART,MODE)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp16, hf8, mask, PART_ODD, MODE_ZEROING);"}, {fp16, hf8, mask},
                 widen_options);
    // 4x narrowing RND_SAT_PP (7-arg): FP32→FP8 → vcvt(dst,src,mask,ROUND,SAT,PART_PP,MODE)
    // FP32→FP8E4M3FN: round_mode must be CAST_RINT
    ExpectInvoke(codegen, "vf.astype", {"vcvt(f8e4m3, fp32, mask, ROUND_R, RS_ENABLE, PART_P1, MODE_ZEROING);"},
                 {f8e4m3, fp32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)}});
    // 4x narrowing RND_PP (6-arg): BF16→FP4 → vcvt(dst,src,mask,ROUND,PART_PP,MODE)
    // BF16→FP4: saturate not applicable
    ExpectInvoke(codegen, "vf.astype", {"vcvt(f4e2m1, bf16, mask, ROUND_F, PART_P1, MODE_ZEROING);"},
                 {f4e2m1, bf16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)}, {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)}});
    // 2x narrowing RND_SAT_PART (7-arg): FP16→HF8 → vcvt(dst,src,mask,ROUND,SAT,PART,MODE)
    // FP16→HF8: round_mode must be CAST_ROUND/CAST_HYBRID
    ExpectInvoke(codegen, "vf.astype", {"vcvt(hf8, fp16, mask, ROUND_A, RS_ENABLE, PART_ODD, MODE_ZEROING);"},
                 {hf8, fp16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_ROUND)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)}});
    // FP8E5M2 / FP4E1M2 narrowing casts with dtype
    // FP32→FP8E5M2: round_mode must be CAST_RINT
    ExpectInvoke(codegen, "vf.astype", {"vcvt(f8e5m2, fp32, mask, ROUND_R, RS_ENABLE, PART_P1, MODE_ZEROING);"},
                 {f8e5m2, fp32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::FP8E5M2}});
    // BF16→FP4E1M2: saturate not applicable
    ExpectInvoke(codegen, "vf.astype", {"vcvt(f4e1m2, bf16, mask, ROUND_F, PART_P1, MODE_ZEROING);"},
                 {f4e1m2, bf16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"dtype", ir::DataType::FP4E1M2}});
    // layout TWO/THREE → PART_P2/PART_P3 (only for 4x narrowing: FP32→FP8)
    const Kwargs layout_two = {{"layout", EnumValue(ir::CastLayout::TWO)},
                               {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                               {"saturate", EnumValue(ir::SaturateMode::ON)},
                               {"dtype", ir::DataType::FP8E4M3FN}};
    ExpectInvoke(codegen, "vf.astype", {"PART_P2"}, {f8e4m3, fp32, mask}, layout_two);
    const Kwargs layout_three = {{"layout", EnumValue(ir::CastLayout::THREE)},
                                 {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                                 {"saturate", EnumValue(ir::SaturateMode::ON)},
                                 {"dtype", ir::DataType::FP8E4M3FN}};
    ExpectInvoke(codegen, "vf.astype", {"PART_P3"}, {f8e4m3, fp32, mask}, layout_three);
}

// ============================================================================
// Tests for new validation CHECKs added for data type consistency and
// parameter combination constraints.
// ============================================================================

TEST(BackendCCEVFOpsTest, RejectsMismatchedSrcDstTypes)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp32_dst = MakeVar("fp32_dst", ir::DataType::FP32);
    auto fp16_src = MakeVar("fp16_src", ir::DataType::FP16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // vf.add: dst=FP32, src0=FP16, src1=FP16 → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.add", {fp32_dst, fp16_src, fp16_src, mask}));

    // vf.sub: dst=FP32, src0=FP16, src1=FP16 → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.sub", {fp32_dst, fp16_src, fp16_src, mask}));

    // vf.move: dst=FP32, src=FP16 → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.move", {fp32_dst, fp16_src, mask}));

    // vf.abs: dst=FP32, src=FP16 → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.abs", {fp32_dst, fp16_src, mask}));

    // vf.truncate: dst=INT32, src=FP32 → should reject
    auto i32_dst = MakeVar("i32_dst", ir::DataType::INT32);
    auto fp32_src = MakeVar("fp32_src", ir::DataType::FP32);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.truncate", {i32_dst, fp32_src, mask}));

    // vf.xor: dst=32-bit, src0=16-bit → should reject (bit width mismatch)
    auto fp16_dst = MakeVar("fp16_dst", ir::DataType::FP16);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.xor", {fp32_dst, fp16_src, fp16_dst, mask}));
}

TEST(BackendCCEVFOpsTest, RejectsMergingForZeroingOnlyOps)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst", ir::DataType::FP32);
    auto src0 = MakeVar("src0", ir::DataType::FP32);
    auto src1 = MakeVar("src1", ir::DataType::FP32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    const Kwargs merging = {{"mode", EnumValue(ir::MergeMode::MERGING)}};

    // Ops that only support ZEROING should reject MERGING
    EXPECT_ANY_THROW(Invoke(codegen, "vf.sub", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.mul", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.div", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.and_", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.xor", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.adds", {dst, src0, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.abs", {dst, src0, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.sqrt", {dst, src0, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.truncate", {dst, src0, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {dst, src0, mask},
                            {{"mode", EnumValue(ir::MergeMode::MERGING)}, {"dtype", ir::DataType::FP16}}));

    // vf.move only supports MERGING, should reject ZEROING
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};
    EXPECT_ANY_THROW(Invoke(codegen, "vf.move", {dst, src0, mask}, zeroing));
}

TEST(BackendCCEVFOpsTest, RejectsInvalidParameterCombinations)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto tile8 = MakeTile("tile8", ir::DataType::UINT8);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    codegen.RegisterAddrRegVar("addr");

    // load_align 2-arg with data_copy_mode → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.load_align", {fp16, tile},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));

    // load_align 4-arg (de-interleave) with data_copy_mode → should reject
    auto fp16b = MakeVar("fp16b", ir::DataType::FP16);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.load_align", {fp16, fp16b, tile, Int(0)},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));

    // store_align MaskReg with data_copy_mode → should reject
    codegen.RegisterMaskRegVar("mask");
    EXPECT_ANY_THROW(Invoke(codegen, "vf.store_align", {tile, mask},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));

    // store_align INTLV with data_copy_mode → should reject
    auto fp16c = MakeVar("fp16c", ir::DataType::FP16);
    EXPECT_ANY_THROW(Invoke(
        codegen, "vf.store_align", {tile, fp16, fp16c, mask},
        {{"dist", EnumValue(ir::StoreDist::INTLV)}, {"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));

    // gather reg→reg with 4 args (mask) → should reject
    auto u16_idx = MakeVar("u16_idx", ir::DataType::UINT16);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.gather", {fp16, fp16b, u16_idx, mask}));

    // gather reg→reg with data_copy_mode → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.gather", {fp16, fp16b, u16_idx},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_LOAD)}}));
}

TEST(BackendCCEVFOpsTest, RejectsInvalidAstypeRoundMode)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // FP16→FP32 widening with round_mode=CAST_ROUND → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {fp32, fp16, mask},
                            {{"layout", EnumValue(ir::CastLayout::ZERO)},
                             {"round_mode", EnumValue(ir::VFRoundMode::CAST_ROUND)},
                             {"saturate", EnumValue(ir::SaturateMode::ON)},
                             {"dtype", ir::DataType::FP32}}));

    // FP32→FP8E4M3FN with round_mode=CAST_ROUND (only CAST_RINT allowed) → should reject
    auto f8_dst = MakeVar("f8_dst", ir::DataType::FP8E4M3FN);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {f8_dst, fp32, mask},
                            {{"layout", EnumValue(ir::CastLayout::ZERO)},
                             {"round_mode", EnumValue(ir::VFRoundMode::CAST_ROUND)},
                             {"saturate", EnumValue(ir::SaturateMode::ON)},
                             {"dtype", ir::DataType::FP8E4M3FN}}));

    // INT32→FP32 (is_int_to_float) with round_mode=CAST_ODD → should reject
    auto i32 = MakeVar("i32", ir::DataType::INT32);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {fp32, i32, mask},
                            {{"layout", EnumValue(ir::CastLayout::ZERO)},
                             {"round_mode", EnumValue(ir::VFRoundMode::CAST_ODD)},
                             {"dtype", ir::DataType::FP32}}));
}

TEST(BackendCCEVFOpsTest, RejectsInvalidAstypeSaturateAndLayout)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto bf16 = MakeVar("bf16", ir::DataType::BF16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // FP32→FP8 widening path with saturate=ON → should reject (not applicable for narrowing)
    // Actually FP32→FP8 is narrowing RND_SAT_PP, saturate must be ON.
    // Test the reverse: FP8→FP32 widening with saturate=ON → should reject
    auto f8_src = MakeVar("f8_src", ir::DataType::FP8E4M3FN);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {fp32, f8_src, mask},
                            {{"layout", EnumValue(ir::CastLayout::ZERO)},
                             {"saturate", EnumValue(ir::SaturateMode::ON)},
                             {"dtype", ir::DataType::FP32}}));

    // FP32→INT32 (is_float_to_same_int) with layout=ONE → should reject (layout not applicable)
    auto i32_dst = MakeVar("i32_dst", ir::DataType::INT32);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {i32_dst, fp32, mask},
                            {{"layout", EnumValue(ir::CastLayout::ONE)},
                             {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                             {"saturate", EnumValue(ir::SaturateMode::ON)},
                             {"dtype", ir::DataType::INT32}}));

    // HF8→FP16 widening with layout=TWO → should reject (only ZERO/ONE for 2x widening)
    auto hf8 = MakeVar("hf8", ir::DataType::HF8);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {fp16, hf8, mask},
                            {{"layout", EnumValue(ir::CastLayout::TWO)}, {"dtype", ir::DataType::FP16}}));
}

TEST(BackendCCEVFOpsTest, EmitsCompareHistogramAndMaskConversions)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto i32 = MakeVar("i32", ir::DataType::INT32);
    auto u16 = MakeVar("u16", ir::DataType::UINT16);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    for (const auto& var : {fp32, fp16, i64, i32, u16, u8}) {
        codegen.RegisterRegTensorVar(var->name_);
    }

    ExpectInvoke(codegen, "vf.update_mask", {"plt_b8("}, {Int(17)}, {{"dtype", ir::DataType::UINT8}}, "mask8");
    ExpectInvoke(codegen, "vf.update_mask", {"plt_b16("}, {Int(17)}, {{"dtype", ir::DataType::FP16}}, "mask16");
    ExpectInvoke(codegen, "vf.update_mask", {"plt_b32("}, {Int(17)}, {}, "mask32");
    ExpectInvoke(codegen, "vf.histograms", {"dhistv2("}, {u16, u8, mask},
                 {{"bin_type", EnumValue(ir::BinType::BIN1)}, {"hist_type", EnumValue(ir::HistType::FREQUENCY)}});
    ExpectInvoke(codegen, "vf.histograms", {"chistv2("}, {u16, u8, mask}, {{"bin_type", EnumValue(ir::BinType::BIN0)}});

    ExpectInvoke(codegen, "vf.eq", {"vcmps_eq("}, {mask, fp32, Float(1.0), mask});
    ExpectInvoke(codegen, "vf.ne", {"vcmp_ne("}, {mask, fp32, fp16, mask}, {{"cmp_dtype", ir::DataType::UINT8}});
    ExpectInvoke(codegen, "vf.lt", {"vcmp_lt("}, {mask, fp32, fp16, mask}, {{"cmp_dtype", ir::DataType::UINT8}});
    ExpectInvoke(codegen, "vf.gt", {"vcmp_gt("}, {mask, fp32, fp16, mask}, {{"cmp_dtype", ir::DataType::UINT8}});
    ExpectInvoke(codegen, "vf.le", {"vcmp_le("}, {mask, fp32, fp16, mask}, {{"cmp_dtype", ir::DataType::UINT8}});
    ExpectInvoke(codegen, "vf.ge", {"vcmp_ge("}, {mask, fp32, fp16, mask}, {{"cmp_dtype", ir::DataType::UINT8}});
    ExpectInvoke(codegen, "vf.squeeze", {"vsqz(", "MODE_NO_STORED"}, {i32, fp16, mask},
                 {{"gather_mode", EnumValue(ir::SqueezeMode::NO_STORE_REG)}});
    ExpectInvoke(codegen, "vf.arange", {"vci(i32, 3, DEC_ORDER)"}, {i32, Int(3)},
                 {{"index_order", EnumValue(ir::IndexOrder::DECREASE_ORDER)}, {"dtype", ir::DataType::INT32}});
    ExpectInvoke(codegen, "vf.arange", {"vci(i64_b64_lo_"}, {i64, Int(5)}, {{"dtype", ir::DataType::INT64}});
    ExpectInvoke(codegen, "vf.unsqueeze", {"vusqz("}, {i32, mask});
    ExpectInvoke(codegen, "vf.truncate", {"vtrc(fp32, fp32, ROUND_C, mask, MODE_ZEROING)"}, {fp32, fp32, mask},
                 {{"round_mode", EnumValue(ir::VFRoundMode::CAST_CEIL)}, {"mode", EnumValue(ir::MergeMode::ZEROING)}});
}

TEST(BackendCCEVFOpsTest, EmitsAlignedDataMovement)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto tile8 = MakeTile("tile8", ir::DataType::UINT8);
    auto tile64 = MakeTile("tile64", ir::DataType::INT64);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp16b = MakeVar("fp16b", ir::DataType::FP16);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    auto f8 = MakeVar("f8", ir::DataType::FP8E4M3FN);
    codegen.RegisterAddrRegVar("addr");
    codegen.RegisterMaskRegVar("mask");
    codegen.RegisterRegTensorVar("f8");

    ExpectInvoke(codegen, "vf.load_align", {"vlds(fp16"}, {fp16, tile, Int(0)});
    ExpectInvoke(codegen, "vf.load_align", {"BRC_B8"}, {u8, tile8, Int(1)}, {{"dist", EnumValue(ir::LoadDist::BRC)}});
    ExpectInvoke(codegen, "vf.load_align", {"(2) * 2", "POST_UPDATE"}, {i64, tile64, Int(2)}, {{"post_update", true}});
    ExpectInvoke(codegen, "vf.load_align", {"vld(fp16"}, {fp16, tile, addr});
    ExpectInvoke(codegen, "vf.load_align", {"plds(mask"}, {mask, tile, Int(0)},
                 {{"dist", EnumValue(ir::LoadDist::US)}});
    ExpectInvoke(codegen, "vf.load_align", {"DINTLV_B16", "POST_UPDATE"}, {fp16, fp16b, tile, Int(3)},
                 {{"dist", EnumValue(ir::LoadDist::DINTLV_B16)}, {"post_update", true}});
    ExpectInvoke(codegen, "vf.load_align", {"vsldb(", "POST_UPDATE"}, {u8, tile8, mask},
                 {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)},
                  {"block_stride", 2},
                  {"repeat_stride", 3},
                  {"post_update", true}});
    ExpectInvoke(codegen, "vf.load_align", {"BRC_B8"}, {f8, tile8, Int(1)}, {{"dist", EnumValue(ir::LoadDist::BRC)}});

    ExpectInvoke(codegen, "vf.store_align", {"vsts("}, {tile, fp16, mask});
    ExpectInvoke(codegen, "vf.store_align", {"ONEPT_B8"}, {tile8, u8, mask},
                 {{"dist", EnumValue(ir::StoreDist::FIRST_ELEMENT)}});
    ExpectInvoke(codegen, "vf.store_align", {"INTLV_B16"}, {tile, fp16, fp16b, mask},
                 {{"dist", EnumValue(ir::StoreDist::INTLV)}});
    ExpectInvoke(codegen, "vf.store_align", {"vsstb(", "POST_UPDATE"}, {tile, fp16, mask, Int(2), Int(3)},
                 {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}, {"post_update", true}});
    ExpectInvoke(codegen, "vf.store_align", {"vst(fp16"}, {tile, fp16, mask, addr});
    ExpectInvoke(codegen, "vf.store_align", {"NORM_B8"}, {tile8, f8, mask});
}

TEST(BackendCCEVFOpsTest, EmitsGatherAndUnalignedDataMovement)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto tile8 = MakeTile("tile8", ir::DataType::UINT8);
    auto tile64 = MakeTile("tile64", ir::DataType::INT64);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto index = MakeVar("index", ir::DataType::UINT32);
    auto index_u16 = MakeVar("index_u16", ir::DataType::UINT16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto ureg = MakeVar("ureg", ir::DataType::INT64);

    // b16 src + uint16 index -> vgather2
    ExpectInvoke(codegen, "vf.gather", {"vgather2("}, {fp16, tile, index_u16, mask});
    // b16 src + uint32 index -> vgather2_bc
    ExpectInvoke(codegen, "vf.gather", {"vgather2_bc("}, {fp16, tile, index, mask});
    ExpectInvoke(codegen, "vf.gather", {"vgatherb("}, {fp16, tile, index, mask},
                 {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_LOAD)}});
    ExpectInvoke(codegen, "vf.gather", {"vselr("}, {fp16, fp16, index_u16});
    ExpectInvoke(codegen, "vf.scatter", {"vscatter("}, {tile, fp16, index_u16, mask});
    ExpectInvoke(codegen, "vf.load", {"UnalignReg __ureg_ld_", "vldas(", "vldus("}, {fp16, tile});
    ExpectInvoke(codegen, "vf.load", {"UnalignReg __ureg_ld_", "vldas(", "vldus(", "(4) * 2", "NORM"},
                 {i64, tile64, Int(4)}, {{"post_mode", std::string("NORM")}});
    ExpectInvoke(codegen, "vf.store", {"UnalignReg __ureg_st_", "vstus(", "vstas("}, {tile, fp16});
    ExpectInvoke(codegen, "vf.store", {"UnalignReg __ureg_st_", "vstus(", "vstas(", "(7) * 2", "NORM"},
                 {tile64, i64, Int(7)}, {{"post_mode", std::string("NORM")}});

    ExpectInvoke(codegen, "vf.unalign_reg_for_store", {"UnalignReg store_ureg;"}, {}, {}, "store_ureg");
    ExpectInvoke(codegen, "vf.load_unalign_init", {"UnalignReg load_ureg;"}, {}, {}, "load_ureg");
    ExpectInvoke(codegen, "vf.load_unalign_pre", {"vldas("}, {ureg, tile});
    ExpectInvoke(codegen, "vf.load_unalign", {"vldus("}, {fp16, ureg, tile});
    ExpectInvoke(codegen, "vf.load_unalign", {"vldus(", "(4) * 2", "POST_UPDATE"}, {i64, ureg, tile64, Int(4)});
    ExpectInvoke(codegen, "vf.store_unalign", {"vstur("}, {tile8, u8, ureg});
    ExpectInvoke(codegen, "vf.store_unalign", {"vstus(", "POST_UPDATE"}, {tile8, u8, ureg, Int(2)},
                 {{"post_update", true}});
    ExpectInvoke(codegen, "vf.store_unalign_post", {"vstar("}, {tile8, ureg});
    ExpectInvoke(codegen, "vf.store_unalign_post", {"vstas(", "POST_UPDATE"}, {tile8, ureg, Int(2)},
                 {{"post_update", true}});
    ExpectInvoke(codegen, "vf.clear_spr", {"sprclr(SPR_AR)"});
}

TEST(BackendCCEVFOpsTest, EmitsMaskLogicOperations)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto mask0 = MakeVar("mask0", ir::DataType::UINT32);
    auto mask1 = MakeVar("mask1", ir::DataType::UINT32);
    auto mask2 = MakeVar("mask2", ir::DataType::UINT32);
    codegen.RegisterMaskRegVar("mask0");
    codegen.RegisterMaskRegVar("mask1");
    codegen.RegisterMaskRegVar("mask2");

    ExpectInvoke(codegen, "vf.and_", {"pand("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.or_", {"por("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.xor", {"pxor("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.not_", {"pnot("}, {mask0, mask1, mask2});
    ExpectInvoke(codegen, "vf.move", {"pmov("}, {mask0, mask1, mask2});
    ExpectInvoke(codegen, "vf.select", {"psel("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.pack", {"ppack(mask0, mask1, HIGHER)"}, {mask0, mask1},
                 {{"part", EnumValue(ir::PackPart::UPPER)}});
    ExpectInvoke(codegen, "vf.unpack", {"punpack(mask0, mask1, LOWER)"}, {mask0, mask1});
    ExpectInvoke(codegen, "vf.interleave", {"pintlv_b8("}, {mask0, mask1, mask1, mask2},
                 {{"dtype", ir::DataType::UINT8}});
    ExpectInvoke(codegen, "vf.de_interleave", {"pdintlv_b16("}, {mask0, mask1, mask1, mask2},
                 {{"dtype", ir::DataType::FP16}});
}

TEST(BackendCCEVFOpsTest, EmitsMaskMemoryAndSpecialRegisterOperations)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::UINT32);
    auto reg = MakeVar("reg", ir::DataType::UINT16);
    auto mask0 = MakeVar("mask0", ir::DataType::UINT32);
    auto mask1 = MakeVar("mask1", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    codegen.RegisterMaskRegVar("mask0");
    codegen.RegisterMaskRegVar("mask1");
    codegen.RegisterAddrRegVar("addr");

    ExpectInvoke(codegen, "vf.load_align", {"plds(mask0"}, {mask0, tile, Int(0)},
                 {{"dist", EnumValue(ir::LoadDist::DS)}});
    ExpectInvoke(codegen, "vf.load_align", {"pld(mask1"}, {mask1, tile, addr});
    ExpectInvoke(codegen, "vf.store_align", {"psts(mask0"}, {tile, mask0, mask1},
                 {{"dist", EnumValue(ir::StoreDist::NORM)}});
    ExpectInvoke(codegen, "vf.store_align", {"pst(mask0"}, {tile, mask0, addr});
    ExpectInvoke(codegen, "vf.store_unalign", {"pstu("}, {tile, mask0, ureg});
    ExpectInvoke(codegen, "vf.mask_gen_with_reg_tensor", {"movvp(generated_mask, (RegTensor<uint16_t> &)reg, 4)"},
                 {reg}, {{"offset", 4}}, "generated_mask");
    ExpectInvoke(codegen, "vf.get_mask_spr", {"movp_b16()"}, {}, {{"width", EnumValue(ir::MaskWidth::B16)}},
                 "spr_mask16");
    ExpectInvoke(codegen, "vf.get_mask_spr", {"movp_b32()"}, {}, {}, "spr_mask32");
}

TEST(BackendCCEVFOpsTest, BitCastEmitInlineReferenceCast)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto src = MakeVar("src", ir::DataType::FP8E4M3FN);

    // Create a vf.bit_cast Call with proper type so GetExprDtype returns FP32
    auto fp32_type = std::make_shared<const ir::ScalarType>(ir::DataType::FP32);
    auto bit_cast_call = std::make_shared<const ir::Call>(
        "vf.bit_cast", std::vector<ir::ExprPtr>{src},
        std::vector<std::pair<std::string, std::any>>{{"dtype", ir::DataType::FP32}}, fp32_type, ir::Span::Unknown());

    // GetExprAsCode on bit_cast should return the cast expression directly
    EXPECT_EQ(codegen.GetExprAsCode(bit_cast_call), "(RegTensor<float> &)src");

    // vf.xor with both args as bit_cast: vxor(dst, (RegTensor<float>&)src, ...)
    auto dst = MakeVar("dst", ir::DataType::FP32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    ExpectInvoke(codegen, "vf.xor", {"vxor(", "(RegTensor<float> &)src", "(RegTensor<float> &)src"},
                 {dst, bit_cast_call, bit_cast_call, mask});

    // vf.xor with one bit_cast arg and one plain FP32 arg (src2 must match bit_cast's 32-bit width)
    auto src2 = MakeVar("src2", ir::DataType::FP32);
    ExpectInvoke(codegen, "vf.xor", {"vxor(", "(RegTensor<float> &)src", "src2"}, {dst, bit_cast_call, src2, mask});
}

TEST(BackendCCEVFOpsTest, EmitsB64LoadStoreAndNewCastPaths)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile64 = MakeTile("tile64", ir::DataType::INT64);
    auto tile64u = MakeTile("tile64u", ir::DataType::UINT64);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto u64 = MakeVar("u64", ir::DataType::UINT64);
    auto i32 = MakeVar("i32", ir::DataType::INT32);
    auto u32 = MakeVar("u32", ir::DataType::UINT32);
    auto u16 = MakeVar("u16", ir::DataType::UINT16);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto i16 = MakeVar("i16", ir::DataType::INT16);
    auto i8 = MakeVar("i8", ir::DataType::INT8);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto bf16 = MakeVar("bf16", ir::DataType::BF16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    for (const auto& var : {i64, u64, i32, u32, u16, u8, i16, i8, fp16, fp32, bf16}) {
        codegen.RegisterRegTensorVar(var->name_);
    }

    // B64 load_align with post_update: __VF_VLDS_B64 POST_UPDATE path
    ExpectInvoke(codegen, "vf.load_align", {"vector_2xvl_s64", "POST_UPDATE"}, {i64, tile64, Int(2)},
                 {{"post_update", true}});

    // B64 store_align with post_update: __VF_VSTS_B64 POST_UPDATE path
    ExpectInvoke(codegen, "vf.store_align", {"vector_2xvl_s64", "POST_UPDATE"}, {tile64, i64, mask, Int(2)},
                 {{"post_update", true}});

    // INT32→UINT8 (4x int narrowing): vcvt(dst, src, mask, RS, PART_PP, MODE) — 6 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(u8, i32, mask, RS_ENABLE, PART_P0, MODE_ZEROING);"}, {u8, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::UINT8}});

    // UINT32→UINT8 (4x int narrowing)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(u8, u32, mask, RS_ENABLE, PART_P0, MODE_ZEROING);"}, {u8, u32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::UINT8}});

    // INT8→INT32 (4x int widening): vcvt(dst, src, mask, PART_PP, MODE_ZEROING) — 5 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, i8, mask, PART_P0, MODE_ZEROING);"}, {i32, i8, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::INT32}});

    // FP16→INT8 (float_to_narrower_int): vcvt(dst, src, mask, ROUND, SAT, PART, MODE) — 7 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i8, fp16, mask, ROUND_F, RS_ENABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, fp16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT8}});

    // BF16→INT8 (float_to_narrower_int)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i8, bf16, mask, ROUND_F, RS_ENABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, bf16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT8}});

    // INT16→UINT8 (2x int narrowing): vcvt(dst, src, mask, RS, PART, MODE) — 6 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(u8, i16, mask, RS_DISABLE, PART_EVEN, MODE_ZEROING);"}, {u8, i16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::UINT8}});

    // INT32→INT16 (2x int narrowing)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i16, i32, mask, RS_DISABLE, PART_EVEN, MODE_ZEROING);"}, {i16, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::INT16}});

    // INT64→INT32 (2x int narrowing)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, i64, mask, RS_DISABLE, PART_EVEN, MODE_ZEROING);"}, {i32, i64, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::INT32}});

    // INT16→INT4 (s16_to_s4 two-step)
    auto s4 = MakeVar("s4", ir::DataType::INT4);
    codegen.RegisterRegTensorVar("s4");
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(s4_f16_tmp, i16, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt_f162s4(s4, s4_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);"},
                 {s4, i16, mask}, {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::INT4}});

    // UINT32→INT8 (int_int_two_step: u32→s32 reinterpret → s32→f32→f16→s8)
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i8_f32_tmp, (RegTensor<int32_t> &)u32, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt(i8_f16_tmp, i8_f32_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i8, i8_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, u32, mask}, {{"dtype", ir::DataType::INT8}});

    // UINT16→INT8 (int_int_two_step: u16→u32→f32→f16→s8, 4-step)
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i8_u32_tmp, u16, mask, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i8_f32_tmp, (RegTensor<int32_t> &)i8_u32_tmp, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt(i8_f16_tmp, i8_f32_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i8, i8_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, u16, mask}, {{"dtype", ir::DataType::INT8}});

    // UINT16→INT16 (int_int_two_step: same-width cross-sign reinterpret cast)
    ExpectInvoke(codegen, "vf.astype", {"i16 = (RegTensor<int16_t> &)u16;"}, {i16, u16, mask},
                 {{"dtype", ir::DataType::INT16}});

    // UINT32→INT32 (int_int_two_step: same-width cross-sign reinterpret cast)
    ExpectInvoke(codegen, "vf.astype", {"i32 = (RegTensor<int32_t> &)u32;"}, {i32, u32, mask},
                 {{"dtype", ir::DataType::INT32}});

    // FP16→INT8 with saturate=OFF (float_to_narrower_int, no saturation)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i8, fp16, mask, ROUND_F, RS_DISABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, fp16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::OFF)},
                  {"dtype", ir::DataType::INT8}});

    // INT32→UINT8 (4x int narrowing) without explicit saturate → implicit RS_ENABLE
    ExpectInvoke(codegen, "vf.astype", {"vcvt(u8, i32, mask, RS_ENABLE, PART_P1, MODE_ZEROING);"}, {u8, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)}, {"dtype", ir::DataType::UINT8}});

    // INT8→INT32 (4x int widening) with layout TWO
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, i8, mask, PART_P2, MODE_ZEROING);"}, {i32, i8, mask},
                 {{"layout", EnumValue(ir::CastLayout::TWO)}, {"dtype", ir::DataType::INT32}});

    // INT8→INT32 (4x int widening) with layout THREE
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, i8, mask, PART_P3, MODE_ZEROING);"}, {i32, i8, mask},
                 {{"layout", EnumValue(ir::CastLayout::THREE)}, {"dtype", ir::DataType::INT32}});

    // FP32→INT16 (float narrowing): vcvt(dst, src, mask, ROUND, SAT, PART, MODE) — 7 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i16, fp32, mask, ROUND_F, RS_ENABLE, PART_EVEN, MODE_ZEROING);"},
                 {i16, fp32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT16}});

    // FP32→INT32 (float_to_same_int with saturate=OFF): vcvt(dst, src, mask, ROUND, RS, MODE) — 5 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, fp32, mask, ROUND_F, RS_DISABLE, MODE_ZEROING);"}, {i32, fp32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::OFF)},
                  {"dtype", ir::DataType::INT32}});

    // INT32→UINT8 (4x int narrowing) with explicit saturate=OFF → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {u8, i32, mask},
                            {{"layout", EnumValue(ir::CastLayout::ZERO)},
                             {"saturate", EnumValue(ir::SaturateMode::OFF)},
                             {"dtype", ir::DataType::UINT8}}));

    // INT16→INT8 (int_int_two_step src_to_f16_ok: s16→f16→s8)
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i8_f16_tmp, i16, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt(i8, i8_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, i16, mask}, {{"dtype", ir::DataType::INT8}});

    // INT32→INT8 with saturate=ON (int_int_two_step src_is_b32 && dst_is_b8: s32→f32→f16→s8)
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i8_f32_tmp, i32, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt(i8_f16_tmp, i8_f32_tmp, mask, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i8, i8_f16_tmp, mask, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT8}});
}

} // namespace
} // namespace backend
} // namespace pypto
