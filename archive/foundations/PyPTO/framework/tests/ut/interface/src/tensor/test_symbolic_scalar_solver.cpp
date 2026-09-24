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
#include <vector>

#include "gtest/gtest.h"
#include "tilefwk/symbolic_scalar.h"
#include "interface/tensor/symbolic_scalar_solver.h"

namespace npu::tile_fwk {

// File-local: a sibling TU in this binary already defines a namespace-scope _sym.
static SymbolicScalar Sym(const std::string& name) { return SymbolicScalar(name); }

// equality lhs == rhs -> affine row (lhs - rhs)
static AffineForm EqRow(const SymbolicScalar& lhs, const SymbolicScalar& rhs)
{
    return AffineForm::Sub(lhs.Raw(), rhs.Raw());
}

TEST(TestAffineForm, InvalidWhenNonlinearOrNonAffine)
{
    auto x = Sym("x"), y = Sym("y");
    EXPECT_TRUE(AffineForm(x.Raw()).ok);
    EXPECT_TRUE(AffineForm((x + y).Raw()).ok);
    EXPECT_TRUE(AffineForm((x * 3 - y).Raw()).ok);
    EXPECT_FALSE(AffineForm((x * y).Raw()).ok); // symbol * symbol
    EXPECT_TRUE(AffineForm((x / y).Raw()).ok);  // treate a symbol
    EXPECT_TRUE(AffineForm((x % y).Raw()).ok);  // treate a symbol
}

TEST(TestGaussCheck, EmptySystemIsConsistent)
{
    std::vector<AffineForm> rows;
    EXPECT_FALSE(AffineForm::GaussCheck(rows));
}

TEST(TestGaussCheck, GroundEqualityTautology)
{
    std::vector<AffineForm> rows{EqRow(SymbolicScalar(5), SymbolicScalar(5))};
    EXPECT_FALSE(AffineForm::GaussCheck(rows));
}

TEST(TestGaussCheck, GroundEqualityContradiction)
{
    std::vector<AffineForm> rows{EqRow(SymbolicScalar(7), SymbolicScalar(2))};
    EXPECT_TRUE(AffineForm::GaussCheck(rows));
}

TEST(TestGaussCheck, UniqueSolutionThreeVar)
{
    auto x = Sym("x"), y = Sym("y"), z = Sym("z");
    std::vector<AffineForm> rows{
        EqRow(x + y, 5),
        EqRow(y + z, 7),
        EqRow(x + z, 6),
    };
    EXPECT_FALSE(AffineForm::GaussCheck(rows));
}

// (x+y)+(z+w)=10 but (y+z)+(x+w)=7: hidden contradiction.
TEST(TestGaussCheck, FourVarCycleSumContradiction)
{
    auto x = Sym("x"), y = Sym("y"), z = Sym("z"), w = Sym("w");
    std::vector<AffineForm> rows{
        EqRow(x + y, 3),
        EqRow(y + z, 5),
        EqRow(z + w, 7),
        EqRow(x + w, 2),
    };
    EXPECT_TRUE(AffineForm::GaussCheck(rows));
}

TEST(TestGaussCheck, FourVarCycleSumConsistent)
{
    auto x = Sym("x"), y = Sym("y"), z = Sym("z"), w = Sym("w");
    std::vector<AffineForm> rows{
        EqRow(x + y, 3),
        EqRow(y + z, 5),
        EqRow(z + w, 7),
        EqRow(x + w, 5),
    };
    EXPECT_FALSE(AffineForm::GaussCheck(rows));
}

TEST(TestGaussCheck, EqualityChainContradiction)
{
    auto a = Sym("a"), b = Sym("b"), c = Sym("c"), d = Sym("d");
    std::vector<AffineForm> rows{
        EqRow(a, b), EqRow(b, c), EqRow(c, d), EqRow(d, 1), EqRow(a, 2),
    };
    EXPECT_TRUE(AffineForm::GaussCheck(rows));
}

TEST(TestGaussCheck, EqualityChainConsistent)
{
    auto a = Sym("a"), b = Sym("b"), c = Sym("c"), d = Sym("d");
    std::vector<AffineForm> rows{
        EqRow(a, b), EqRow(b, c), EqRow(c, d), EqRow(d, 1), EqRow(a, 1),
    };
    EXPECT_FALSE(AffineForm::GaussCheck(rows));
}

TEST(TestGaussCheck, RedundantScaledEquationsConsistent)
{
    auto x = Sym("x"), y = Sym("y");
    std::vector<AffineForm> rows{
        EqRow(x * 3 - y * 2, 1),
        EqRow(x * 6 - y * 4, 2),
    };
    EXPECT_FALSE(AffineForm::GaussCheck(rows));
}

TEST(TestGaussCheck, RedundantScaledEquationsContradiction)
{
    auto x = Sym("x"), y = Sym("y");
    std::vector<AffineForm> rows{
        EqRow(x * 3 - y * 2, 1),
        EqRow(x * 6 - y * 4, 3),
    };
    EXPECT_TRUE(AffineForm::GaussCheck(rows));
}

// row3 = row1 + row2 in coefficients; rhs 11 == 7 + 4.
TEST(TestGaussCheck, SingularThreeByThreeConsistent)
{
    auto x = Sym("x"), y = Sym("y"), z = Sym("z");
    std::vector<AffineForm> rows{
        EqRow(x * 2 + y * 3 - z, 7),
        EqRow(x - y + z * 2, 4),
        EqRow(x * 3 + y * 2 + z, 11),
    };
    EXPECT_FALSE(AffineForm::GaussCheck(rows));
}

// Same LHS dependency, but rhs 9 != 7 + 4.
TEST(TestGaussCheck, SingularThreeByThreeContradiction)
{
    auto x = Sym("x"), y = Sym("y"), z = Sym("z");
    std::vector<AffineForm> rows{
        EqRow(x * 2 + y * 3 - z, 7),
        EqRow(x - y + z * 2, 4),
        EqRow(x * 3 + y * 2 + z, 9),
    };
    EXPECT_TRUE(AffineForm::GaussCheck(rows));
}

TEST(TestGaussCheck, TilingArithmeticConsistent)
{
    auto i = Sym("i"), j = Sym("j");
    std::vector<AffineForm> rows{
        EqRow(i * 4 + j, 10),
        EqRow(i, j),
        EqRow(j * 2, 4),
    };
    EXPECT_FALSE(AffineForm::GaussCheck(rows));
}

TEST(TestGaussCheck, TilingArithmeticContradiction)
{
    auto i = Sym("i"), j = Sym("j");
    std::vector<AffineForm> rows{
        EqRow(i * 4 + j, 10),
        EqRow(i, j),
        EqRow(j * 2, 5),
    };
    EXPECT_TRUE(AffineForm::GaussCheck(rows));
}

} // namespace npu::tile_fwk
