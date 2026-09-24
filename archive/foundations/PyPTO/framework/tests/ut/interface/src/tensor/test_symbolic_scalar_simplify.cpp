/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include "gtest/gtest.h"
#include "tilefwk/symbolic_scalar.h"

namespace npu::tile_fwk {
SymbolicScalar _sym(const std::string& name) { return SymbolicScalar(name); }

// ============================================================================
// Unary
// ============================================================================

TEST(TestSimplify, Neg)
{
    auto x = _sym("x");
    EXPECT_EQ((-(-x)).Simplify().Dump(), "x");
}

TEST(TestSimplify, NegSub)
{
    auto x = _sym("x");
    auto y = _sym("y");
    EXPECT_EQ((-(x - y)).Simplify().Dump(), "(y-x)");
}

TEST(TestSimplify, Pos)
{
    auto x = _sym("x");
    EXPECT_EQ((+x).Simplify().Dump(), "x");
}

TEST(TestSimplify, Not)
{
    auto x = _sym("x");
    EXPECT_EQ((!(!x)).Simplify().Dump(), "x");
}

// ============================================================================
// Add
// ============================================================================

TEST(TestSimplify, Add)
{
    auto x = _sym("x");
    EXPECT_EQ(((x + 3) + 5).Simplify().Dump(), "(x+8)");
}

// ============================================================================
// Sub
// ============================================================================

TEST(TestSimplify, Sub)
{
    auto x = _sym("x");
    EXPECT_EQ((x - x).Simplify().Dump(), "0");
}

// ============================================================================
// Mul
// ============================================================================

TEST(TestSimplify, Mul)
{
    auto x = _sym("x");
    EXPECT_EQ(((x * 3) * 5).Simplify().Dump(), "(x*15)");
}

// ============================================================================
// Div
// ============================================================================

TEST(TestSimplify, Div)
{
    auto x = _sym("x");
    EXPECT_EQ(((x * 6) / 3).Simplify().Dump(), "(x*2)");
}

// ============================================================================
// Mod
// ============================================================================

TEST(TestSimplify, Mod)
{
    auto x = _sym("x");
    EXPECT_EQ(((x * 6) % 3).Simplify().Dump(), "0");
}

// ============================================================================
// Min / Max
// ============================================================================

TEST(TestSimplify, Min)
{
    auto x = _sym("x");
    EXPECT_EQ(x.Min(x).Simplify().Dump(), "x");
}

TEST(TestSimplify, Max)
{
    auto x = _sym("x");
    EXPECT_EQ(x.Max(x).Simplify().Dump(), "x");
}

// ============================================================================
// Comparisons
// ============================================================================

TEST(TestSimplify, Eq)
{
    auto x = _sym("x");
    EXPECT_EQ((x == x).Simplify().Dump(), "1");
}

TEST(TestSimplify, Ne)
{
    auto x = _sym("x");
    EXPECT_EQ((x != x).Simplify().Dump(), "0");
}

TEST(TestSimplify, Lt)
{
    auto x = _sym("x");
    EXPECT_EQ((x < x).Simplify().Dump(), "0");
}

TEST(TestSimplify, Le)
{
    auto x = _sym("x");
    EXPECT_EQ((x <= x).Simplify().Dump(), "1");
}

TEST(TestSimplify, Gt)
{
    auto x = _sym("x");
    EXPECT_EQ((x > x).Simplify().Dump(), "0");
}

TEST(TestSimplify, Ge)
{
    auto x = _sym("x");
    EXPECT_EQ((x >= x).Simplify().Dump(), "1");
}

// ============================================================================
// Mixed / Complex
// ============================================================================

TEST(TestSimplify, PreserveImmediate)
{
    auto x = SymbolicScalar(10);
    auto result = x.Simplify();
    EXPECT_TRUE(result.IsImmediate());
    EXPECT_EQ(result.Concrete(), 10);
}

TEST(TestSimplify, PreserveSymbol)
{
    auto x = _sym("x");
    auto result = x.Simplify();
    EXPECT_TRUE(result.IsSymbol());
    EXPECT_EQ(result.Dump(), "x");
}

TEST(TestSimplify, NestedSimplify)
{
    auto x = _sym("x");
    auto y = _sym("y");
    EXPECT_EQ(((x - y) + y).Simplify().Dump(), "x");
}

TEST(TestSimplify, ComplexCancellation)
{
    auto x = _sym("x");
    auto y = _sym("y");
    auto z = _sym("z");
    EXPECT_EQ((x * y + x * z).Simplify().Dump(), "((y+z)*x)");
}

// ============================================================================
// Logical and/or
// ============================================================================

TEST(TestSimplify, LogicOp)
{
    auto x = _sym("x");
    auto y = _sym("y");

    EXPECT_EQ((x && 0).Simplify().Dump(), "0");
    EXPECT_EQ((x && 1).Simplify().Dump(), "x");
    EXPECT_EQ((x && x).Simplify().Dump(), "x");

    EXPECT_EQ((x || 0).Simplify().Dump(), "x");
    EXPECT_EQ((x || 1).Simplify().Dump(), "1");
    EXPECT_EQ((x || x).Simplify().Dump(), "x");

    EXPECT_EQ((x && (x || y)).Simplify().Dump(), "x");
    EXPECT_EQ((x || (x && y)).Simplify().Dump(), "x");

    EXPECT_EQ((!(x && y)).Simplify().Dump(), "((!x)||(!y))");
    EXPECT_EQ((!(x || y)).Simplify().Dump(), "((!x)&&(!y))");
}

} // namespace npu::tile_fwk
