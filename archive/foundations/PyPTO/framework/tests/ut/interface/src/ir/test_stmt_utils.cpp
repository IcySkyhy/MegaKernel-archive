/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file test_stmt_utils.cpp
 * \brief Unit tests for ir::utils helpers (LookupVarInExpr, ...).
 */

#include "gtest/gtest.h"

#include <memory>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/transforms/utils/stmt_utils.h"
#include "ir/type.h"

namespace pypto {
namespace ir {
namespace {

// A single map + inputs that drive LookupVarInExpr through every branch:
//   - non-Var input           -> passthrough (early return of `expr`)
//   - Var absent from map     -> returns the var itself
//   - transitive Var->Var chain landing on a non-Var value -> returns that value
//   - self-cycle (d->d)       -> cycle guard exits, returns the revisited var
//   - mutual cycle (e<->f)    -> cycle guard exits, returns the revisited var
TEST(StmtUtilsTest, LookupVarInExprCoversAllPaths)
{
    auto sp = Span::Unknown();
    auto st = std::make_shared<ScalarType>(DataType::INT32);

    VarPtr a = std::make_shared<const Var>("a", st, sp);
    VarPtr b = std::make_shared<const Var>("b", st, sp);
    VarPtr c = std::make_shared<const Var>("c", st, sp);
    VarPtr d = std::make_shared<const Var>("d", st, sp); // self-cycle
    VarPtr e = std::make_shared<const Var>("e", st, sp); // mutual cycle e <-> f
    VarPtr f = std::make_shared<const Var>("f", st, sp);
    VarPtr g = std::make_shared<const Var>("g", st, sp); // not in map

    ExprPtr one = std::make_shared<ConstInt>(1, DataType::INT32, sp); // non-Var map value
    ExprPtr two = std::make_shared<ConstInt>(2, DataType::INT32, sp); // non-Var input

    utils::VarExprMap vm;
    vm[a] = b;   // Var -> Var
    vm[b] = c;   // Var -> Var (transitive)
    vm[c] = one; // Var -> non-Var expr
    vm[d] = d;   // self-cycle
    vm[e] = f;   // mutual cycle
    vm[f] = e;

    // Non-Var input: dynamic_pointer_cast<Var> fails -> return expr unchanged.
    EXPECT_EQ(utils::LookupVarInExpr(two, vm), two);

    // Var not present in the map: loop runs once, find() == end -> return the var itself.
    EXPECT_EQ(utils::LookupVarInExpr(g, vm), g);

    // Transitive chain a -> b -> c -> one: walks two Var->Var edges, then the cast of `one`
    // to Var fails -> return the mapped non-Var value.
    EXPECT_EQ(utils::LookupVarInExpr(a, vm), one);

    // Self-cycle d -> d: second visit to d fails the seen.insert() guard -> exit, return d.
    EXPECT_EQ(utils::LookupVarInExpr(d, vm), d);

    // Mutual cycle e -> f -> e: third visit to e fails the guard -> exit, return e.
    EXPECT_EQ(utils::LookupVarInExpr(e, vm), e);
}

} // namespace
} // namespace ir
} // namespace pypto
