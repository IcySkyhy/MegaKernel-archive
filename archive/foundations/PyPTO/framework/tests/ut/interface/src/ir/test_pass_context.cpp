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
 * \file test_pass_context.cpp
 * \brief Coverage tests for PassContext and CallbackInstrument (pass_context.cpp)
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/pass_context.h"
#include "ir/transforms/passes.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

static Span Sp() { return Span("test", 1, 1); }

class ScopedPassContext {
public:
    explicit ScopedPassContext(PassContext& ctx) : ctx_(ctx) { ctx_.EnterContext(); }
    ~ScopedPassContext() { ctx_.ExitContext(); }

private:
    PassContext& ctx_;
};

class PassContextTest : public testing::Test {
protected:
    ProgramPtr MakeSimpleProgram()
    {
        auto value = std::make_shared<ConstInt>(7, DataType::INT32, Sp());
        auto body = std::make_shared<EvalStmt>(value, Sp());
        auto func = std::make_shared<Function>("ctx_func", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp());
        return std::make_shared<Program>(std::vector<FunctionPtr>{func}, "ctx_prog", Sp());
    }
};

// ============================================================================
// PassContext: thread-local stack
// ============================================================================

TEST_F(PassContextTest, TestNestedContexts)
{
    {
        PassContext ctx1({});
        PassContext ctx2({});
        ScopedPassContext guard1(ctx1);
        EXPECT_EQ(PassContext::Current(), &ctx1);

        {
            ScopedPassContext guard2(ctx2);
            EXPECT_EQ(PassContext::Current(), &ctx2);
        }

        EXPECT_EQ(PassContext::Current(), &ctx1);
    }
    EXPECT_EQ(PassContext::Current(), nullptr);
}

// ============================================================================
// PassContext: instruments
// ============================================================================

TEST_F(PassContextTest, TestRunBeforePassCallsAllInstruments)
{
    int count = 0;
    auto inst1 = std::make_shared<CallbackInstrument>([&count](const Pass&, const ProgramPtr&) { count++; }, nullptr,
                                                      "I1");
    auto inst2 = std::make_shared<CallbackInstrument>([&count](const Pass&, const ProgramPtr&) { count++; }, nullptr,
                                                      "I2");
    PassContext ctx({inst1, inst2});
    auto p = pass::CreateProgramPass([](const ProgramPtr& prog) { return prog; }, "BeforeInstrumentedPass");
    auto prog = MakeSimpleProgram();
    ctx.RunBeforePass(p, prog);
    EXPECT_EQ(count, 2);
}

TEST_F(PassContextTest, TestRunAfterPassCallsAllInstruments)
{
    int count = 0;
    auto inst1 = std::make_shared<CallbackInstrument>(
        nullptr, [&count](const Pass&, const ProgramPtr&) { count++; }, "I1");
    auto inst2 = std::make_shared<CallbackInstrument>(
        nullptr, [&count](const Pass&, const ProgramPtr&) { count++; }, "I2");
    PassContext ctx({inst1, inst2});
    auto p = pass::CreateProgramPass([](const ProgramPtr& prog) { return prog; }, "AfterInstrumentedPass");
    auto prog = MakeSimpleProgram();
    ctx.RunAfterPass(p, prog);
    EXPECT_EQ(count, 2);
}

// ============================================================================
// PassContext: integration with Pass execution
// ============================================================================

TEST_F(PassContextTest, TestPassCallsContextInstruments)
{
    std::string before_pass_name;
    std::string after_pass_name;
    ProgramPtr before_program;
    ProgramPtr after_program;
    auto inst = std::make_shared<CallbackInstrument>(
        [&before_pass_name, &before_program](const Pass& pass, const ProgramPtr& program) {
            before_pass_name = pass.GetName();
            before_program = program;
        },
        [&after_pass_name, &after_program](const Pass& pass, const ProgramPtr& program) {
            after_pass_name = pass.GetName();
            after_program = program;
        },
        "TrackInst");
    PassContext ctx({inst});
    ScopedPassContext guard(ctx);

    auto p = pass::CreateProgramPass([](const ProgramPtr& prog) { return prog; }, "TrackedPass");
    auto prog = MakeSimpleProgram();
    auto result = p(prog);

    EXPECT_EQ(before_pass_name, "TrackedPass");
    EXPECT_EQ(after_pass_name, "TrackedPass");
    EXPECT_EQ(before_program.get(), prog.get());
    EXPECT_EQ(after_program.get(), result.get());
}

} // namespace ir
} // namespace pypto
