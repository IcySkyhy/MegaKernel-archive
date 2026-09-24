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
 * \file test_debug_ops.cpp
 * \brief Coverage tests for debug_ops.cpp type deduction
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

// ============================================================================
// debug.dump_tensor
// ============================================================================

class DebugOpsTest : public testing::Test {};

TEST_F(DebugOpsTest, DumpTensor_FullWindow_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("t", {16, 32}, DataType::FP16);
    auto offsets = MakeIntTuple({0, 0});
    auto shapes = MakeIntTuple({16, 32});
    auto call = reg.Create("debug.dump_tensor", {tensor, offsets, shapes}, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(DebugOpsTest, DumpTensor_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("debug.dump_tensor", {MakeTensorVar("t", {16}, DataType::FP16)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(DebugOpsTest, DumpTensor_NonTensorFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("debug.dump_tensor",
                                  {MakeScalarVar("s", DataType::FP32), MakeIntTuple({0}), MakeIntTuple({16})}, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// debug.dump_tile
// ============================================================================

TEST_F(DebugOpsTest, DumpTile_1Arg_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("t", {16, 32}, DataType::FP16);
    auto call = reg.Create("debug.dump_tile", {tile}, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(DebugOpsTest, DumpTile_3Args_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("t", {16, 32}, DataType::FP16);
    auto offsets = MakeIntTuple({0, 0});
    auto shapes = MakeIntTuple({8, 16});
    auto call = reg.Create("debug.dump_tile", {tile, offsets, shapes}, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(DebugOpsTest, DumpTile_NonTileFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("debug.dump_tile", {MakeScalarVar("s", DataType::FP32)}, Sp()), npu::tile_fwk::Error);
}

// ============================================================================
// debug.printf
// ============================================================================

TEST_F(DebugOpsTest, Printf_IntConversion_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"format", std::string("val=%d")}};
    auto call = reg.Create("debug.printf", {MakeScalarVar("x", DataType::INT32)}, kwargs, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(DebugOpsTest, Printf_FloatConversion_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"format", std::string("val=%f")}};
    auto call = reg.Create("debug.printf", {MakeScalarVar("x", DataType::FP32)}, kwargs, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(DebugOpsTest, Printf_UnsignedConversion_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"format", std::string("val=%u")}};
    auto call = reg.Create("debug.printf", {MakeScalarVar("x", DataType::UINT32)}, kwargs, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(DebugOpsTest, Printf_HexConversion_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"format", std::string("val=%x")}};
    auto call = reg.Create("debug.printf", {MakeScalarVar("x", DataType::UINT32)}, kwargs, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(DebugOpsTest, Printf_MissingFormat_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("debug.printf", {MakeScalarVar("x", DataType::INT32)}, Sp()), npu::tile_fwk::Error);
}

TEST_F(DebugOpsTest, Printf_ArgCountMismatch_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"format", std::string("%d %d")}};
    EXPECT_THROW((void)reg.Create("debug.printf", {MakeScalarVar("x", DataType::INT32)}, kwargs, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(DebugOpsTest, Printf_FloatConversionWithInt_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"format", std::string("%f")}};
    EXPECT_THROW((void)reg.Create("debug.printf", {MakeScalarVar("x", DataType::INT32)}, kwargs, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// debug.assert
// ============================================================================

TEST_F(DebugOpsTest, Assert_BoolCondition_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"condition_text", std::string("x > 0")},
                                                            {"format", std::string("")}};
    auto call = reg.Create("debug.assert", {MakeScalarVar("c", DataType::BOOL)}, kwargs, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(DebugOpsTest, Assert_WithFormatArgs_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"condition_text", std::string("x > 0")},
                                                            {"format", std::string("x=%d")}};
    auto call = reg.Create("debug.assert", {MakeScalarVar("c", DataType::BOOL), MakeScalarVar("x", DataType::INT32)},
                           kwargs, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(DebugOpsTest, Assert_NonBoolCondition_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"condition_text", std::string("x > 0")},
                                                            {"format", std::string("")}};
    EXPECT_THROW((void)reg.Create("debug.assert", {MakeScalarVar("c", DataType::INT32)}, kwargs, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(DebugOpsTest, Assert_MissingConditionText_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"format", std::string("")}};
    EXPECT_THROW((void)reg.Create("debug.assert", {MakeScalarVar("c", DataType::BOOL)}, kwargs, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// debug.trap
// ============================================================================

TEST_F(DebugOpsTest, Trap_NoArgs_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("debug.trap", {}, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

} // namespace ir
} // namespace pypto
