/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_vf_ops.cpp
 * \brief Coverage tests for vf_ops.cpp type deduction
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

class VFOpsTest : public testing::Test {};

// ============================================================================
// DeduceVFScalarType: vf.reg_tensor
// ============================================================================

TEST_F(VFOpsTest, RegTensor_NoArgs_ReturnsFP32Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("vf.reg_tensor", {}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

// ============================================================================
// DeduceVFMaskType: vf.create_mask
// ============================================================================

TEST_F(VFOpsTest, CreateMask_NoArgs_ReturnsUINT16Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("vf.create_mask", {}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::UINT16);
}

// ============================================================================
// DeduceVFFromDstArg: vf.full
// ============================================================================

TEST_F(VFOpsTest, Full_WithArgs_ReturnsDstArgType)
{
    auto& reg = OpRegistry::GetInstance();
    auto dst = MakeScalarVar("dst", DataType::FP16);
    auto scalar = MakeScalarVar("scalar", DataType::FP32);
    auto mask = MakeScalarVar("mask", DataType::UINT16);
    auto call = reg.Create("vf.full", {dst, scalar, mask}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

// ============================================================================
// DeduceVFFromDstArg: vf.muls, vf.add
// ============================================================================

TEST_F(VFOpsTest, Muls_WithArgs_ReturnsDstArgType)
{
    auto& reg = OpRegistry::GetInstance();
    auto dst = MakeScalarVar("dst", DataType::FP32);
    auto src = MakeScalarVar("src", DataType::FP16);
    auto scalar = MakeScalarVar("scalar", DataType::FP16);
    auto mask = MakeScalarVar("mask", DataType::UINT16);
    auto call = reg.Create("vf.muls", {dst, src, scalar, mask}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(VFOpsTest, Add_WithArgs_ReturnsDstArgType)
{
    auto& reg = OpRegistry::GetInstance();
    auto dst = MakeScalarVar("dst", DataType::FP32);
    auto src0 = MakeScalarVar("src0", DataType::FP16);
    auto src1 = MakeScalarVar("src1", DataType::FP16);
    auto mask = MakeScalarVar("mask", DataType::UINT16);
    auto call = reg.Create("vf.add", {dst, src0, src1, mask}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

// ============================================================================
// DeduceVFScalarType: vf.bit_cast
// ============================================================================

TEST_F(VFOpsTest, BitCast_WithDtypeKwarg_ReturnsTargetScalarType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeScalarVar("src", DataType::FP8E4M3FN);
    auto call = reg.Create("vf.bit_cast", {src}, {{"dtype", DataType::FP32}}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(VFOpsTest, BitCast_NoDtypeKwarg_DefaultsToFP32)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeScalarVar("src", DataType::UINT16);
    auto call = reg.Create("vf.bit_cast", {src}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

} // namespace ir
} // namespace pypto
