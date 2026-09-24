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
 * \file test_passes.cpp
 * \brief Coverage tests for pass infrastructure (passes.cpp)
 */

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/passes.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }
static Span Sp() { return Span("test", 1, 1); }

class PassesTest : public testing::Test {
protected:
    FunctionPtr MakeFunction(const std::string& name, int64_t value, FunctionType func_type = FunctionType::OPAQUE,
                             std::optional<int> max_threads = std::nullopt)
    {
        auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
        auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(value, DataType::INT32, Sp()), Sp());
        std::vector<std::pair<std::string, std::any>> attrs;
        if (max_threads.has_value()) {
            attrs.emplace_back(kMaxThreadsAttr, *max_threads);
        }
        return std::make_shared<Function>(name, std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                          body, Sp(), func_type, false, std::move(attrs));
    }

    ProgramPtr MakeSimpleProgram()
    {
        return std::make_shared<Program>(std::vector<FunctionPtr>{MakeFunction("f", 1)}, "prog", Sp());
    }

    ProgramPtr MakeTwoFunctionProgram()
    {
        return std::make_shared<Program>(std::vector<FunctionPtr>{MakeFunction("f", 1), MakeFunction("g", 2)}, "prog",
                                         Sp());
    }
};

TEST_F(PassesTest, TestNullPassIntrospection)
{
    Pass pass;

    EXPECT_EQ(pass.GetName(), "NullPass");
    EXPECT_TRUE(pass.GetRequiredProperties().Empty());
    EXPECT_TRUE(pass.GetProducedProperties().Empty());
    EXPECT_TRUE(pass.GetInvalidatedProperties().Empty());
}

TEST_F(PassesTest, TestCreateFunctionPassTransform)
{
    auto p = pass::CreateFunctionPass(
        [](const FunctionPtr& f) {
            auto x = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span("test", 1, 1));
            auto body = std::make_shared<AssignStmt>(
                x, std::make_shared<ConstInt>(99, DataType::INT32, Span("test", 1, 1)), Span("test", 1, 1));
            return std::make_shared<Function>(f->name_, f->params_, f->returnTypes_, body, f->span_, f->funcType_);
        },
        "RewriteFunc");
    auto prog = MakeSimpleProgram();
    auto result = p(prog);
    auto func = result->GetFunction("f");
    ASSERT_NE(func, nullptr);
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(func->body_->stmts_[0]);
    ASSERT_NE(assign, nullptr);
    auto ci = std::dynamic_pointer_cast<const ConstInt>(assign->value_);
    ASSERT_NE(ci, nullptr);
    EXPECT_EQ(ci->value_, 99);
}

TEST_F(PassesTest, TestCreateFunctionPassProperties)
{
    PassProperties properties{{IRProperty::TypeChecked}, {IRProperty::SSAForm}, {IRProperty::NoNestedCalls}};
    auto p = pass::CreateFunctionPass([](const FunctionPtr& f) { return f; }, "FunctionWithProperties", properties);

    EXPECT_TRUE(p.GetRequiredProperties().Contains(IRProperty::TypeChecked));
    EXPECT_TRUE(p.GetProducedProperties().Contains(IRProperty::SSAForm));
    EXPECT_TRUE(p.GetInvalidatedProperties().Contains(IRProperty::NoNestedCalls));
}

TEST_F(PassesTest, TestFunctionPassTransformsEveryFunction)
{
    int call_count = 0;
    auto p = pass::CreateFunctionPass(
        [&call_count](const FunctionPtr& f) {
            ++call_count;
            return f;
        },
        "CountFunctions");

    auto result = p(MakeTwoFunctionProgram());

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->name_, "prog");
    EXPECT_NE(result->GetFunction("f"), nullptr);
    EXPECT_NE(result->GetFunction("g"), nullptr);
    EXPECT_EQ(call_count, 2);
}

TEST_F(PassesTest, TestUnnamedFunctionPassUsesDefaultName)
{
    auto p = pass::CreateFunctionPass([](const FunctionPtr& f) { return f; });

    EXPECT_EQ(p.GetName(), "FunctionPass");
    EXPECT_TRUE(p.GetRequiredProperties().Empty());
    EXPECT_TRUE(p.GetProducedProperties().Empty());
    EXPECT_TRUE(p.GetInvalidatedProperties().Empty());
}

// ============================================================================
// CreateProgramPass
// ============================================================================

TEST_F(PassesTest, TestCreateProgramPassTransformAndProperties)
{
    PassProperties properties{{IRProperty::SSAForm}, {IRProperty::NoNestedCalls}, {IRProperty::TypeChecked}};
    auto p = pass::CreateProgramPass(
        [](const ProgramPtr& prog) {
            auto x = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span("test", 1, 1));
            auto body = std::make_shared<AssignStmt>(
                x, std::make_shared<ConstInt>(7, DataType::INT32, Span("test", 1, 1)), Span("test", 1, 1));
            auto func = std::make_shared<Function>("rewritten", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body,
                                                   Span("test", 1, 1));
            return std::make_shared<Program>(std::vector<FunctionPtr>{func}, prog->name_, prog->span_);
        },
        "RewriteProgram", properties);

    auto result = p(MakeSimpleProgram());
    ASSERT_NE(result, nullptr);
    EXPECT_NE(result->GetFunction("rewritten"), nullptr);
    EXPECT_TRUE(p.GetRequiredProperties().Contains(IRProperty::SSAForm));
    EXPECT_TRUE(p.GetProducedProperties().Contains(IRProperty::NoNestedCalls));
    EXPECT_TRUE(p.GetInvalidatedProperties().Contains(IRProperty::TypeChecked));
}

TEST_F(PassesTest, TestUnnamedProgramPassUsesDefaultName)
{
    auto p = pass::CreateProgramPass([](const ProgramPtr& prog) { return prog; });

    EXPECT_EQ(p.GetName(), "ProgramPass");
    EXPECT_TRUE(p.GetRequiredProperties().Empty());
    EXPECT_TRUE(p.GetProducedProperties().Empty());
    EXPECT_TRUE(p.GetInvalidatedProperties().Empty());
}

TEST_F(PassesTest, TestPassCopyMovePreserveImplAndRunAlias)
{
    auto original = pass::CreateProgramPass([](const ProgramPtr& prog) { return prog; }, "CopyablePass");
    Pass copied(original);
    Pass copy_assigned;
    copy_assigned = original;
    Pass moved(std::move(copied));
    Pass move_assigned;
    move_assigned = std::move(copy_assigned);
    auto prog = MakeSimpleProgram();

    EXPECT_EQ(original.GetName(), "CopyablePass");
    EXPECT_EQ(moved.GetName(), "CopyablePass");
    EXPECT_EQ(move_assigned.GetName(), "CopyablePass");
    EXPECT_EQ(move_assigned.run(prog).get(), prog.get());
}

TEST_F(PassesTest, TestRunVerifierFactoryContract)
{
    auto verifier = pass::RunVerifier({"SSAVerify", "TypeCheck", "NoNestedCall"});
    auto prog = MakeSimpleProgram();

    EXPECT_EQ(verifier.GetName(), "IRVerifier");
    EXPECT_EQ(verifier.run(prog).get(), prog.get());
}

TEST_F(PassesTest, TestLegacyShimPassFactoryNames)
{
    const std::vector<std::pair<std::string, Pass (*)()>> factories{
        {"InitMemRef", &pass::InitMemRef},
        {"BasicMemoryReuse", &pass::BasicMemoryReuse},
        {"AllocateMemoryAddr", &pass::AllocateMemoryAddr},
        {"OutlineIncoreScopes", &pass::OutlineIncoreScopes},
        {"ConvertTensorToBlockOps", &pass::ConvertTensorToBlockOps},
        {"FlattenCallExpr", &pass::FlattenCallExpr},
        {"NormalizeStmtStructure", &pass::NormalizeStmtStructure},
        {"FlattenSingleStmt", &pass::FlattenSingleStmt},
    };

    for (const auto& factory_case : factories) {
        EXPECT_EQ(factory_case.second().GetName(), factory_case.first);
    }
}

// ============================================================================
// PassPipeline
// ============================================================================

TEST_F(PassesTest, TestPassPipelineEmpty)
{
    PassPipeline pipeline;
    EXPECT_TRUE(pipeline.GetPassNames().empty());
    auto prog = MakeSimpleProgram();
    auto result = pipeline.Run(prog);
    EXPECT_EQ(result.get(), prog.get());
}

TEST_F(PassesTest, TestPassPipelineAddPass)
{
    PassPipeline pipeline;
    pipeline.AddPass(pass::CreateProgramPass([](const ProgramPtr& prog) { return prog; }, "FirstPass"));
    pipeline.AddPass(pass::CreateProgramPass([](const ProgramPtr& prog) { return prog; }, "SecondPass"));
    auto names = pipeline.GetPassNames();
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "FirstPass");
    EXPECT_EQ(names[1], "SecondPass");
}

TEST_F(PassesTest, TestPassPipelineRun)
{
    int call_count = 0;
    PassPipeline pipeline;
    pipeline.AddPass(pass::CreateProgramPass(
        [&call_count](const ProgramPtr& prog) {
            call_count++;
            return prog;
        },
        "Counter1"));
    pipeline.AddPass(pass::CreateProgramPass(
        [&call_count](const ProgramPtr& prog) {
            call_count++;
            return prog;
        },
        "Counter2"));
    auto function = MakeFunction("f", 1, FunctionType::SIMT_VF, 256);
    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{function}, "prog", Sp());
    auto result = pipeline.Run(prog);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(call_count, 2);
    auto result_function = result->GetFunction("f");
    ASSERT_NE(result_function, nullptr);
    ASSERT_TRUE(result_function->HasAttr(kMaxThreadsAttr));
    EXPECT_EQ(result_function->GetAttr<int>(kMaxThreadsAttr), 256);
}

} // namespace ir
} // namespace pypto
