/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"

#include "ir/transforms/passes.h"
#include "interface/tensor/irbuilder.h"

#include "program_builder.h"

using namespace npu::tile_fwk;

class TestRemoveRedundantTokenPass : public testing::Test {
protected:
    void SetUp() override
    {
        savedFlag_ = IRContext::Get().AssembleNewLogicalTensor();
        IRContext::Get().SetAssembleNewLogicalTensor(true);
    }

    void TearDown() override { IRContext::Get().SetAssembleNewLogicalTensor(savedFlag_); }

private:
    bool savedFlag_{false};
};

TEST_F(TestRemoveRedundantTokenPass, TensorOps)
{
    auto a = Tensor(DT_FP32, {16, 16}, "a");
    auto out = Tensor(DT_FP32, {16, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("TensorOps", {a, out});

    auto lhs = Add(a, Element(DT_FP32, 1));
    auto rhs = Sub(a, Element(DT_FP32, 1));
    Assemble(Add(lhs, rhs), {0, 0}, out);

    auto prog = p.EndFunction();

    auto inferred = pypto::ir::pass::InferTokenPass()(prog);
    auto result = pypto::ir::pass::RemoveRedundantTokenPass()(inferred);

    ASSERT_NE(result, nullptr);
}

TEST_F(TestRemoveRedundantTokenPass, DisjointWrites)
{
    auto a = Tensor(DT_FP32, {8, 16}, "a");
    auto out = Tensor(DT_FP32, {16, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("DisjointWrites", {a, out});

    auto v1 = Add(a, Element(DT_FP32, 1));
    Assemble(v1, {0, 0}, out);

    auto v2 = Sub(a, Element(DT_FP32, 1));
    Assemble(v2, {8, 0}, out);

    auto prog = p.EndFunction();

    auto inferred = pypto::ir::pass::InferTokenPass()(prog);
    auto result = pypto::ir::pass::RemoveRedundantTokenPass()(inferred);

    ASSERT_NE(result, nullptr);
}

TEST_F(TestRemoveRedundantTokenPass, IfStmt)
{
    auto a = Tensor(DT_FP32, {16, 16}, "a");
    auto out = Tensor(DT_FP32, {16, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("IfStmt", {a, out});

    auto result = p.If(
        SymbolicScalar("condition") > 0, [&] { p.Yield(Add(a, Element(DT_FP32, 1))); },
        [&] { p.Yield(Sub(a, Element(DT_FP32, 1))); });

    Assemble(p.AsTensor(result[0]), {0, 0}, out);

    auto prog = p.EndFunction();

    auto inferred = pypto::ir::pass::InferTokenPass()(prog);
    auto removed = pypto::ir::pass::RemoveRedundantTokenPass()(inferred);

    ASSERT_NE(removed, nullptr);
}

TEST_F(TestRemoveRedundantTokenPass, ForStmt)
{
    auto a = Tensor(DT_FP32, {16, 16}, "a");
    auto out = Tensor(DT_FP32, {16, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("ForStmt", {a, out});

    auto init = Add(a, Element(DT_FP32, 1));
    auto result = p.For(0, 2, 1, {{"carry", init}}, [&](SymbolicScalar, const std::vector<ir::VarPtr>& carries) {
        p.Continue(Add(p.AsTensor(carries[0]), Element(DT_FP32, 1)));
    });

    Assemble(p.AsTensor(result[0]), {0, 0}, out);

    auto prog = p.EndFunction();

    auto inferred = pypto::ir::pass::InferTokenPass()(prog);
    auto removed = pypto::ir::pass::RemoveRedundantTokenPass()(inferred);

    ASSERT_NE(removed, nullptr);
}
