/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root directory of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"

#include "ir/transforms/passes.h"

#include "program_builder.h"

using namespace npu::tile_fwk;

// Mirrors of python/tests/ut/ir/test_merge_pass/test_oi_update1.py
TEST(CanonicalizePassTest, OiUpdate1)
{
    auto a = Tensor(DT_FP32, {32, 32}, "a");
    auto b = Tensor(DT_FP32, {32, 32}, "b");

    ProgramBuilder p;

    p.BeginFunction("OiUpdate1", {a, b});
    auto oi0 = p.Alloc(DT_FP32, {32, 32}, "oi_update_init");  // carry 0 initial value
    auto oi1 = p.Alloc(DT_FP32, {32, 32}, "oi_update1_init"); // carry 1 initial value

    p.For(0, 10, 1, {{"oi_update", oi0}, {"oi_update1", oi1}}, [&](SymbolicScalar i, const std::vector<ir::VarPtr>& c) {
        auto v0 = c[0]; // oi_update
        auto v1 = c[1]; // oi_update1
        auto tmp = Add(a, p.AsTensor(v0));
        Assemble(tmp, {0, 0}, b);
        auto fwd1 = p.If(i == 0, [&] { p.Yield(v1); }, [&] { p.Yield(v1); });
        auto fwd = p.AsTensor(fwd1[0]);
        Assemble(fwd, {0, 0}, b); // side effect keeps fwd1 (and via the yield, oi_update1)
        p.Yield(tmp, fwd);        // loop-body carry-forward
    });

    auto prog = p.EndFunction();
    ir::pass::Canonicalize()(prog);
}
