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

#include <memory>
#include <string>

#include "codegen/cce/code_context.h"
#include "ir/scalar_expr.h"

namespace pypto {
namespace codegen {
namespace {

ir::VarPtr MakeScalarVar(const std::string& name)
{
    auto type = std::make_shared<const ir::ScalarType>(ir::DataType::INT32);
    return std::make_shared<const ir::Var>(name, type, ir::Span::Unknown());
}

} // namespace

TEST(CodeContextTest, ResolvesHighestNumericSSAVersion)
{
    CodeContext context;
    context.RegisterVar(MakeScalarVar("reg_dst_1"), "reg_dst_v1");
    context.RegisterVar(MakeScalarVar("reg_dst_12"), "reg_dst_v12");
    context.RegisterVar(MakeScalarVar("reg_dst_suffix"), "not_a_version");

    EXPECT_EQ(context.GetVarName(MakeScalarVar("reg_dst")), "reg_dst_v12");
    EXPECT_EQ(context.GetVarName(MakeScalarVar("reg_dst")), "reg_dst_v12");
}

TEST(CodeContextTest, KeepsVersionZeroFallbackPrecedence)
{
    CodeContext context;
    context.RegisterVar(MakeScalarVar("value_0"), "value_v0");
    context.RegisterVar(MakeScalarVar("value_9"), "value_v9");

    EXPECT_EQ(context.GetVarName(MakeScalarVar("value")), "value_v0");
}

} // namespace codegen
} // namespace pypto
