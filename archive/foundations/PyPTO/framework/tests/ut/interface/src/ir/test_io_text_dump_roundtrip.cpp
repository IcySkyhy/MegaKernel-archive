/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 * \file test_text_roundtrip.cpp
 * \brief Round-trip tests for IRTextDumper / IRTextLoader.
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/transforms/io_text.h"
#include "ir/type.h"
#include "test_ir.h"

namespace pypto {
namespace ir {

static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }
static Span Sp() { return Span("test", 1, 1); }

static VarPtr Var_(const std::string& name, DataType dt = DataType::INT32)
{
    return std::make_shared<Var>(name, Scalar(dt), Sp());
}

class IoTextDumpRoundTripTest : public testing::Test {};

// ---- Expression round-trip ----
TEST_F(IoTextDumpRoundTripTest, TestExprRoundTrip)
{
    std::string error;
    auto a = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    EXPECT_EQ(TextDump(a), "42");
    EXPECT_EQ(TextDump(TextLoadExpr("42", error)), "42");
    EXPECT_TRUE(error.empty());

    auto b = std::make_shared<ConstFloat>(3.5, DataType::FP32, Sp());
    EXPECT_EQ(TextDump(b), "3.5");

    EXPECT_EQ(TextDump(std::make_shared<ConstBool>(true, Sp())), "true");
    EXPECT_EQ(TextDump(std::make_shared<ConstBool>(false, Sp())), "false");

    // Binary: (1 add 2)
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto two = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto add = std::make_shared<Add>(one, two, DataType::INT32, Sp());
    EXPECT_EQ(TextDump(add), "(1 add 2)");
    auto loaded = TextLoadExpr("(1 add 2)", error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(loaded), "(1 add 2)");

    // Unary: (neg 5)
    auto five = std::make_shared<ConstInt>(5, DataType::INT32, Sp());
    auto neg = std::make_shared<Neg>(five, DataType::INT32, Sp());
    EXPECT_EQ(TextDump(neg), "(neg 5)");
    EXPECT_EQ(TextDump(TextLoadExpr("(neg 5)", error)), "(neg 5)");
    EXPECT_TRUE(error.empty());

    // Cast: (cast float 1)
    auto cast = std::make_shared<Cast>(one, DataType::FP32, Sp());
    EXPECT_EQ(TextDump(cast), "(cast float 1)");
    EXPECT_EQ(TextDump(TextLoadExpr("(cast float 1)", error)), "(cast float 1)");
    EXPECT_TRUE(error.empty());

    // MakeTuple: tuple(1, 2)
    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{one, two}, Sp());
    EXPECT_EQ(TextDump(tup), "tuple(1, 2)");
    EXPECT_EQ(TextDump(TextLoadExpr("tuple(1, 2)", error)), "tuple(1, 2)");
    EXPECT_TRUE(error.empty());

    // GetItemExpr: getitem(tuple(1, 2), 0)
    auto idx = std::make_shared<ConstInt>(0, DataType::INDEX, Sp());
    auto gi = std::make_shared<GetItemExpr>(tup, idx, Sp());
    EXPECT_EQ(TextDump(gi), "getitem(tuple(1, 2), 0)");
}

// ---- Type round-trip ----
TEST_F(IoTextDumpRoundTripTest, TestTypeRoundTrip)
{
    std::string error;
    EXPECT_EQ(TextDumpType(Scalar(DataType::FP32)), "float");
    EXPECT_EQ(TextDumpType(TextLoadType("float", error)), "float");
    EXPECT_TRUE(error.empty());

    EXPECT_EQ(TextDumpType(GetUnknownType()), "unknown");
    EXPECT_EQ(TextDumpType(GetTokenType()), "token");
    EXPECT_EQ(TextDumpType(GetNoneType()), "none");

    EXPECT_EQ(TextDumpType(std::make_shared<PtrType>(DataType::FP16)), "ptr<half>");
    EXPECT_EQ(TextDumpType(TextLoadType("ptr<half>", error)), "ptr<half>");
    EXPECT_TRUE(error.empty());

    auto tensor = std::make_shared<TensorType>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, Sp()),
                             std::make_shared<ConstInt>(8, DataType::INT64, Sp())},
        DataType::FP32);
    EXPECT_EQ(TextDumpType(tensor), "tensor<4 x 8, float, tensor_view<>>");
    EXPECT_EQ(TextDumpType(TextLoadType("tensor<4 x 8, float, tensor_view<>>", error)),
              "tensor<4 x 8, float, tensor_view<>>");
    EXPECT_TRUE(error.empty());
}

// ---- Statement round-trip ----
TEST_F(IoTextDumpRoundTripTest, TestStmtRoundTrip)
{
    std::string error;
    // AssignStmt: int32_t %x = 42;
    auto x = Var_("x");
    auto assign = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, Sp()), Sp());
    std::string dumped = TextDump(assign);
    EXPECT_NE(dumped.find("int32_t %x = 42;"), std::string::npos);

    // Round-trip
    auto loaded = TextLoadStmt(dumped, error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(loaded), dumped);

    // ReturnStmt
    auto ret = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, Sp())},
                                            Sp());
    std::string retDumped = TextDump(ret);
    EXPECT_NE(retDumped.find("return"), std::string::npos);
    auto retLoaded = TextLoadStmt(retDumped, error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(retLoaded), retDumped);

    // YieldStmt
    auto yld = std::make_shared<YieldStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, Sp())},
                                           Sp());
    std::string yldDumped = TextDump(yld);
    EXPECT_NE(yldDumped.find("yield"), std::string::npos);

    // BreakStmt / ContinueStmt
    EXPECT_EQ(TextDump(std::make_shared<BreakStmt>(Sp())), "break;");
    EXPECT_EQ(TextDump(std::make_shared<ContinueStmt>(Sp())), "continue;");
    EXPECT_EQ(TextDump(TextLoadStmt("break;", error)), "break;");
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(TextLoadStmt("continue;", error)), "continue;");
    EXPECT_TRUE(error.empty());
}

// ---- For loop round-trip ----
TEST_F(IoTextDumpRoundTripTest, TestForLoopRoundTrip)
{
    std::string error;
    auto i = Var_("i");
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto body = std::make_shared<AssignStmt>(Var_("x"), one, Sp());
    auto forStmt = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, body, std::vector<VarPtr>{},
                                             Sp());
    std::string dumped = TextDump(forStmt);
    EXPECT_NE(dumped.find("for"), std::string::npos);
    EXPECT_NE(dumped.find("inrange"), std::string::npos);

    auto loaded = TextLoadStmt(dumped, error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(loaded), dumped);
}

// ---- Function round-trip ----
TEST_F(IoTextDumpRoundTripTest, TestFunctionRoundTrip)
{
    std::string error;
    auto x = Var_("x");
    auto body = std::make_shared<ReturnStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(42, DataType::INT32, Sp())}, Sp());
    auto func = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                           body, Sp());
    std::string dumped = TextDump(func);
    EXPECT_NE(dumped.find("function f"), std::string::npos);
    EXPECT_NE(dumped.find("incast("), std::string::npos);
    EXPECT_NE(dumped.find("outcast("), std::string::npos);

    auto loaded = TextLoadFunction(dumped, error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(loaded), dumped);
}

} // namespace ir
} // namespace pypto
