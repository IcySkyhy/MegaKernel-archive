/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root directory of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"

#include "tilefwk/tilefwk_op.h"
#include "ir/transforms/passes.h"

#include "program_builder.h"

using namespace npu::tile_fwk;

TEST(MergeStmtPass, TestMergeStmtsIntoIf)
{
    ProgramBuilder p;

    auto a = Tensor(DT_FP32, {32, 32}, "a");
    auto b = Tensor(DT_FP32, {32, 32}, "b");

    p.BeginFunction("TestMergeStmtsIntoIf", {a, b});

    auto x0 = p.Alloc(DT_FP32, {32, 32}, "oi_update");
    auto x1 = p.Alloc(DT_FP32, {32, 32}, "oi_update1");

    auto n = SymbolicScalar("n");
    p.For(0, n, 1, {{"oi_update", x0}, {"oi_update1", x1}}, [&](SymbolicScalar i, const std::vector<ir::VarPtr>& c) {
        auto v0 = c[0];
        auto v1 = c[1];

        // ---- top-level if #1 ----
        auto top1 = p.If(
            i == 0,
            [&] {
                p.Yield(p.If(
                    i == n - 1, [&] { p.Yield({v0, v1}); },
                    [&] {
                        auto x = Add(a, Element(DT_FP32, 0));
                        p.Yield(x, v1);
                    }));
            },
            [&] {
                auto x = Add(a, p.AsTensor(v0));
                p.Yield(p.If(
                    i == n - 1,
                    [&] {
                        Assemble(x, {0, 0}, b);
                        p.Yield(v0, v1);
                    },
                    [&] { p.Yield(x, v1); }));
            });

        // ---- top-level if #2: feeds on if #1's return vars ----
        auto t0 = top1[0];
        auto t1 = top1[1];
        auto top2 = p.If(
            i == 0,
            [&] {
                p.Yield(p.If(
                    i == n - 1, [&] { p.Yield(t0, t1); },
                    [&] {
                        auto x = Add(a, a);
                        p.Yield(t0, x);
                    }));
            },
            [&] {
                auto x = Add(a, p.AsTensor(t1));
                p.Yield(p.If(
                    i == n - 1,
                    [&] {
                        Assemble(x, {0, 0}, b);
                        p.Yield(t0, x);
                    },
                    [&] { p.Yield(t0, x); }));
            });

        p.Continue(top2);
    });

    auto prog = p.EndFunction();

    // Code-coverage smoke test: run MergeStmtsIntoIf on the if-tree and confirm the loop survives.
    auto out = pypto::ir::pass::MergeStmtsIntoIf()(prog);
    ASSERT_NE(out, nullptr);
}

TEST(MergeStmtPass, TestDynValidShapeCloneThenElse)
{
    ProgramBuilder p;

    auto x = Tensor(DT_FP32, {32, 32}, "x");
    auto y = Tensor(DT_FP32, {32, 32}, "y");
    auto z = Tensor(DT_FP32, {32, 32}, "z");

    p.BeginFunction("TestDynValidShapeClone", {x, y, z});

    auto n = SymbolicScalar("n");
    auto m = SymbolicScalar("m");
    p.For(0, 2, 1, {}, [&](SymbolicScalar i, const std::vector<ir::VarPtr>&) {
        auto xv = View(x, {16, 16}, {SymbolicScalar(0), SymbolicScalar(0)});
        auto yv = View(y, {16, 16}, {SymbolicScalar(0), SymbolicScalar(0)});

        auto rets = p.If(
            i == 0,
            [&] {
                auto t = Add(xv, yv);
                p.Yield(t, n.Min(m));
            },
            [&] {
                auto t = Add(xv, yv);
                p.Yield(t, n.Min(16));
            });

        auto t = p.AsTensor(rets[0]);
        auto vs = p.AsSymbol(rets[1]);

        auto outView = View(t, {16, 16}, {vs, SymbolicScalar(16)}, {SymbolicScalar(0), SymbolicScalar(0)});
        auto outView2 = View(outView, {16, 16}, {SymbolicScalar(16), SymbolicScalar(16)}, {vs, SymbolicScalar(0)});
        Assemble(outView2, {0, 0}, z);
        p.Continue();
    });

    auto prog = p.EndFunction();

    auto out = pypto::ir::pass::MergeStmtsIntoIf()(prog);
    ASSERT_NE(out, nullptr);
}
