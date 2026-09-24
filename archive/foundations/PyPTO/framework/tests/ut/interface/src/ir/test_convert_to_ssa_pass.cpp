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
 * \file test_convert_to_ssa_pass.cpp
 * \brief Coverage tests for ConvertToSSA pass (convert_to_ssa_pass.cpp)
 */

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/passes.h"
#include "ir/type.h"
#include "test_ir.h"

namespace pypto {
namespace ir {

static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }
static Span Sp() { return Span("test", 1, 1); }

class ConvertToSSAPassTest : public testing::Test {
protected:
    VarPtr MakeVar(const std::string& name, TypePtr type = Scalar(DataType::INT32))
    {
        return std::make_shared<Var>(name, type, Sp());
    }

    ExprPtr Int(int64_t value) { return std::make_shared<ConstInt>(value, DataType::INT32, Sp()); }

    FunctionPtr MakeFunc(const std::string& name, const std::vector<VarPtr>& params, const StmtPtr& body,
                         const std::vector<TypePtr>& ret_types = {})
    {
        return std::make_shared<Function>(name, params, ret_types, body, Sp());
    }

    ProgramPtr MakeProg(const FunctionPtr& func)
    {
        return std::make_shared<Program>(std::vector<FunctionPtr>{func}, "prog", Sp());
    }

    ProgramPtr RunPass(const ProgramPtr& prog)
    {
        auto p = pass::ConvertToSSA();
        return p(prog);
    }

    FunctionPtr RunOnFunc(const FunctionPtr& func)
    {
        auto result = RunPass(MakeProg(func));
        return result->GetFunction(func->name_);
    }
};

// ============================================================================
// For-loop with assignment to outer variable creates iter_arg
// ============================================================================

TEST_F(ConvertToSSAPassTest, TestForLoopOuterVarBecomesIterArg)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto init_x = std::make_shared<AssignStmt>(x, zero, Sp());
    auto add = std::make_shared<Add>(x, i, DataType::INT32, Sp());
    auto assign_x = std::make_shared<AssignStmt>(x, add, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto forBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign_x, yield}, Sp());
    auto forStmt = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, forBody,
                                             std::vector<VarPtr>{}, Sp());
    auto outerBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{init_x, forStmt}, Sp());
    auto func = MakeFunc("f", {}, outerBody);
    auto result = RunPass(MakeProg(func));
    auto result_func = result->GetFunction("f");
    auto seq = result_func->body_;
    ASSERT_NE(seq, nullptr);
    ASSERT_EQ(seq->stmts_.size(), 2u);
    auto init_assign = std::dynamic_pointer_cast<const AssignStmt>(seq->stmts_[0]);
    ASSERT_NE(init_assign, nullptr);
    auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(seq->stmts_[1]);
    ASSERT_NE(for_stmt, nullptr);
    ASSERT_EQ(for_stmt->iterArgs_.size(), 1u);
    ASSERT_EQ(for_stmt->returnVars_.size(), 1u);
    auto init_var = std::dynamic_pointer_cast<const Var>(for_stmt->iterArgs_[0]->initValue_);
    ASSERT_NE(init_var, nullptr);
    EXPECT_EQ(init_var->name_, init_assign->var_->name_);
}

TEST_F(ConvertToSSAPassTest, TestNoneInitLoopCarryUsesMergedValueType)
{
    // A slot with no initial value is seeded with `None`, which the builder types as NoneType.
    auto none_type = GetNoneType();
    auto none = MakeVar("None", none_type);
    auto result_seed = MakeVar("result", none_type);
    auto init_result = std::make_shared<AssignStmt>(result_seed, none, Sp());

    auto result_then = MakeVar("result", Scalar(DataType::INT32));
    auto assign_then = std::make_shared<AssignStmt>(result_then, Int(1), Sp());
    auto break_then = std::make_shared<BreakStmt>(Sp());
    auto then_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign_then, break_then}, Sp());
    auto branch = std::make_shared<IfStmt>(std::make_shared<ConstBool>(true, Sp()), then_body, std::nullopt,
                                           std::vector<VarPtr>{}, Sp());

    auto result_transient = MakeVar("result", Scalar(DataType::FP32));
    auto assign_transient = std::make_shared<AssignStmt>(result_transient,
                                                         std::make_shared<ConstFloat>(1.0, DataType::FP32, Sp()), Sp());
    auto result_tail = MakeVar("result", Scalar(DataType::INT32));
    auto assign_tail = std::make_shared<AssignStmt>(result_tail, Int(2), Sp());
    auto break_tail = std::make_shared<BreakStmt>(Sp());
    auto loop_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{branch, assign_transient, assign_tail, break_tail},
                                                Sp());
    auto loop = std::make_shared<WhileStmt>(std::make_shared<ConstBool>(true, Sp()), std::vector<IterArgPtr>{},
                                            loop_body, std::vector<VarPtr>{}, Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{init_result, loop}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {}, body));
    auto result_seq = result_func->body_;
    auto result_loop = std::dynamic_pointer_cast<const WhileStmt>(result_seq->stmts_[1]);
    ASSERT_NE(result_loop, nullptr);
    ASSERT_EQ(result_loop->iterArgs_.size(), 1u);
    ASSERT_EQ(result_loop->returnVars_.size(), 1u);
    EXPECT_NE(As<NoneType>(result_loop->iterArgs_[0]->initValue_->GetType()), nullptr);
    auto result_iter_type = As<ScalarType>(result_loop->iterArgs_[0]->iterVar_->GetType());
    ASSERT_NE(result_iter_type, nullptr);
    EXPECT_EQ(result_iter_type->dtype_, DataType::INT32);
    auto result_return_type = As<ScalarType>(result_loop->returnVars_[0]->GetType());
    ASSERT_NE(result_return_type, nullptr);
    EXPECT_EQ(result_return_type->dtype_, DataType::INT32);

    auto result_loop_body = std::dynamic_pointer_cast<const SeqStmts>(result_loop->body_);
    auto result_branch = std::dynamic_pointer_cast<const IfStmt>(result_loop_body->stmts_[0]);
    ASSERT_NE(result_branch, nullptr);
    auto result_then_body = std::dynamic_pointer_cast<const SeqStmts>(result_branch->thenBody_);
    ASSERT_NE(result_then_body, nullptr);
    ASSERT_GE(result_then_body->stmts_.size(), 2u);
    auto result_break = std::dynamic_pointer_cast<const BreakStmt>(result_then_body->stmts_[1]);
    ASSERT_NE(result_break, nullptr);
    ASSERT_EQ(result_break->value_.size(), 1u);
    auto result_break_type = As<ScalarType>(result_break->value_[0]->GetType());
    ASSERT_NE(result_break_type, nullptr);
    EXPECT_EQ(result_break_type->dtype_, DataType::INT32);

    ASSERT_TRUE(result_branch->elseBody_.has_value());
    auto result_else_body = std::dynamic_pointer_cast<const SeqStmts>(*result_branch->elseBody_);
    ASSERT_NE(result_else_body, nullptr);
    ASSERT_FALSE(result_else_body->stmts_.empty());
    auto result_else_yield = std::dynamic_pointer_cast<const YieldStmt>(result_else_body->stmts_.back());
    ASSERT_NE(result_else_yield, nullptr);
    ASSERT_EQ(result_else_yield->value_.size(), 1u);
    auto result_else_value = As<Var>(result_else_yield->value_[0]);
    ASSERT_NE(result_else_value, nullptr);
    EXPECT_NE(As<NoneType>(result_else_value->GetType()), nullptr);
}

TEST_F(ConvertToSSAPassTest, TestNestedLoopKeepsNoneInitUninitialized)
{
    auto none_type = GetNoneType();
    auto none = MakeVar("None", none_type);
    auto result_seed = MakeVar("result", none_type);
    auto init_result = std::make_shared<AssignStmt>(result_seed, none, Sp());

    auto result_value = MakeVar("result", Scalar(DataType::INT32));
    auto assign_result = std::make_shared<AssignStmt>(result_value, Int(1), Sp());
    auto inner_break = std::make_shared<BreakStmt>(Sp());
    auto inner_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign_result, inner_break}, Sp());
    auto inner_loop = std::make_shared<WhileStmt>(std::make_shared<ConstBool>(true, Sp()), std::vector<IterArgPtr>{},
                                                  inner_body, std::vector<VarPtr>{}, Sp());
    auto outer_break = std::make_shared<BreakStmt>(Sp());
    auto outer_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{inner_loop, outer_break}, Sp());
    auto outer_loop = std::make_shared<WhileStmt>(std::make_shared<ConstBool>(true, Sp()), std::vector<IterArgPtr>{},
                                                  outer_body, std::vector<VarPtr>{}, Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{init_result, outer_loop}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {}, body));
    auto result_outer = std::dynamic_pointer_cast<const WhileStmt>(result_func->body_->stmts_[1]);
    ASSERT_NE(result_outer, nullptr);
    ASSERT_EQ(result_outer->iterArgs_.size(), 1u);
    EXPECT_NE(As<NoneType>(result_outer->iterArgs_[0]->initValue_->GetType()), nullptr);

    auto result_outer_body = std::dynamic_pointer_cast<const SeqStmts>(result_outer->body_);
    auto result_inner = std::dynamic_pointer_cast<const WhileStmt>(result_outer_body->stmts_[0]);
    ASSERT_NE(result_inner, nullptr);
    ASSERT_EQ(result_inner->iterArgs_.size(), 1u);
    EXPECT_NE(As<NoneType>(result_inner->iterArgs_[0]->initValue_->GetType()), nullptr);
    EXPECT_NE(As<ScalarType>(result_inner->iterArgs_[0]->iterVar_->GetType()), nullptr);
}

// ============================================================================
// Variable use references current version
// ============================================================================

TEST_F(ConvertToSSAPassTest, TestVarUseRefsCurrentVersion)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto assign1 = std::make_shared<AssignStmt>(x, one, Sp());
    auto two = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto add = std::make_shared<Add>(x, two, DataType::INT32, Sp());
    auto assign2 = std::make_shared<AssignStmt>(x, add, Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign1, assign2}, Sp());
    auto func = MakeFunc("f", {x}, body);
    auto result = RunPass(MakeProg(func));
    auto result_func = result->GetFunction("f");
    auto seq = result_func->body_;
    ASSERT_NE(seq, nullptr);
    ASSERT_GE(seq->stmts_.size(), 2u);
    auto a2 = std::dynamic_pointer_cast<const AssignStmt>(seq->stmts_[1]);
    ASSERT_NE(a2, nullptr);
    auto result_add = std::dynamic_pointer_cast<const Add>(a2->value_);
    ASSERT_NE(result_add, nullptr);
    auto lhs_var = std::dynamic_pointer_cast<const Var>(result_add->left_);
    ASSERT_NE(lhs_var, nullptr);
    auto a1 = std::dynamic_pointer_cast<const AssignStmt>(seq->stmts_[0]);
    ASSERT_NE(a1, nullptr);
    EXPECT_EQ(lhs_var->name_, a1->var_->name_);
}

TEST_F(ConvertToSSAPassTest, TestSharedExpressionKeepsIdentity)
{
    auto x = MakeVar("x");
    auto idx = MakeVar("idx");
    auto tuple_anchor = MakeVar("tuple_anchor", std::make_shared<TupleType>(std::vector<TypePtr>{
                                                    Scalar(DataType::INT32), Scalar(DataType::INT32)}));
    auto selected = MakeVar("selected");
    auto tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{x, x}, Sp());
    auto anchor_assign = std::make_shared<AssignStmt>(tuple_anchor, tuple, Sp());
    auto get_item = std::make_shared<GetItemExpr>(tuple, idx, Sp());
    auto selected_assign = std::make_shared<AssignStmt>(selected, get_item, Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{anchor_assign, selected_assign}, Sp());

    auto result = RunPass(MakeProg(MakeFunc("f", {x, idx}, body)));
    auto seq = result->GetFunction("f")->body_;
    auto ssa_anchor = std::dynamic_pointer_cast<const AssignStmt>(seq->stmts_[0]);
    auto ssa_selected = std::dynamic_pointer_cast<const AssignStmt>(seq->stmts_[1]);
    ASSERT_NE(ssa_anchor, nullptr);
    ASSERT_NE(ssa_selected, nullptr);
    auto ssa_get_item = std::dynamic_pointer_cast<const GetItemExpr>(ssa_selected->value_);
    ASSERT_NE(ssa_get_item, nullptr);
    EXPECT_EQ(ssa_anchor->value_.get(), ssa_get_item->value_.get());
}

// ============================================================================
// If-stmt without else — creates phi for modified var
// ============================================================================

TEST_F(ConvertToSSAPassTest, TestIfNoElseCreatesPhiForModifiedVar)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto assign = std::make_shared<AssignStmt>(x, one, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, assign, std::nullopt, std::vector<VarPtr>{}, Sp());
    auto func = MakeFunc("f", {x}, ifStmt);
    auto result = RunPass(MakeProg(func));
    auto result_func = result->GetFunction("f");
    ASSERT_EQ(result_func->body_->stmts_.size(), 1u);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    EXPECT_FALSE(result_if->returnVars_.empty());
}

TEST_F(ConvertToSSAPassTest, TestUnknownVarUseIsKeptAsOriginal)
{
    auto external = MakeVar("external");
    auto y = MakeVar("y");
    auto body = std::make_shared<AssignStmt>(y, external, Sp());
    auto result_func = RunOnFunc(MakeFunc("f", {}, body));
    ASSERT_NE(result_func, nullptr);

    ASSERT_EQ(result_func->body_->stmts_.size(), 1u);
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(assign, nullptr);
    auto rhs = std::dynamic_pointer_cast<const Var>(assign->value_);
    ASSERT_NE(rhs, nullptr);
    EXPECT_EQ(rhs->name_, "external");
}

TEST_F(ConvertToSSAPassTest, TestIfWithoutPhiKeepsReturnVarsEmpty)
{
    auto x = MakeVar("x");
    auto local = MakeVar("local");
    auto assign = std::make_shared<AssignStmt>(local, Int(1), Sp());
    auto if_stmt = std::make_shared<IfStmt>(std::make_shared<ConstBool>(true, Sp()), assign, std::nullopt,
                                            std::vector<VarPtr>{}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {x}, if_stmt));
    ASSERT_NE(result_func, nullptr);
    ASSERT_EQ(result_func->body_->stmts_.size(), 1u);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    EXPECT_TRUE(result_if->returnVars_.empty());
    EXPECT_FALSE(result_if->elseBody_.has_value());
}

TEST_F(ConvertToSSAPassTest, TestIfExistingReturnVarWithoutPhiIsVersioned)
{
    auto y = MakeVar("y");
    auto noop = std::make_shared<EvalStmt>(Int(0), Sp());
    auto if_stmt = std::make_shared<IfStmt>(std::make_shared<ConstBool>(true, Sp()), noop, std::nullopt,
                                            std::vector<VarPtr>{y}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {y}, if_stmt));
    ASSERT_NE(result_func, nullptr);
    ASSERT_EQ(result_func->body_->stmts_.size(), 1u);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    ASSERT_EQ(result_if->returnVars_.size(), 1u);
    EXPECT_NE(result_if->returnVars_[0]->name_.find("y_"), std::string::npos);
    ASSERT_TRUE(result_if->elseBody_.has_value());
    ASSERT_EQ((*result_if->elseBody_)->stmts_.size(), 1u);
    auto else_yield = std::dynamic_pointer_cast<const YieldStmt>((*result_if->elseBody_)->stmts_[0]);
    ASSERT_NE(else_yield, nullptr);
    EXPECT_TRUE(else_yield->value_.empty());
}

TEST_F(ConvertToSSAPassTest, TestIfExistingReturnVarNotDuplicatedWhenPhiAlreadyCreated)
{
    auto x = MakeVar("x");
    auto then_assign = std::make_shared<AssignStmt>(x, Int(1), Sp());
    auto else_assign = std::make_shared<AssignStmt>(x, Int(2), Sp());
    auto if_stmt = std::make_shared<IfStmt>(std::make_shared<ConstBool>(true, Sp()), then_assign,
                                            std::optional<StmtPtr>(else_assign), std::vector<VarPtr>{x}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {x}, if_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    ASSERT_EQ(result_if->returnVars_.size(), 1u);
    EXPECT_NE(result_if->returnVars_[0]->name_.find("x_"), std::string::npos);
}

TEST_F(ConvertToSSAPassTest, TestIfElseOnlyAssignmentUsesBeforeVersionForThenYield)
{
    auto x = MakeVar("x");
    auto then_noop = std::make_shared<EvalStmt>(Int(0), Sp());
    auto else_assign = std::make_shared<AssignStmt>(x, Int(2), Sp());
    auto if_stmt = std::make_shared<IfStmt>(std::make_shared<ConstBool>(true, Sp()), then_noop,
                                            std::optional<StmtPtr>(else_assign), std::vector<VarPtr>{}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {x}, if_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    ASSERT_EQ(result_if->returnVars_.size(), 1u);
    ASSERT_FALSE(result_if->thenBody_->stmts_.empty());
    auto then_yield = std::dynamic_pointer_cast<const YieldStmt>(result_if->thenBody_->stmts_.back());
    ASSERT_NE(then_yield, nullptr);
    ASSERT_EQ(then_yield->value_.size(), 1u);
    auto yielded_x = std::dynamic_pointer_cast<const Var>(then_yield->value_[0]);
    ASSERT_NE(yielded_x, nullptr);
    EXPECT_EQ(yielded_x->name_, result_func->params_[0]->name_);
}

TEST_F(ConvertToSSAPassTest, TestIfMergesPhiValuesWithExistingBranchYields)
{
    auto x = MakeVar("x");
    auto z = MakeVar("z");
    auto then_body = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(x, Int(1), Sp()),
                             std::make_shared<YieldStmt>(std::vector<ExprPtr>{z}, Sp())},
        Sp());
    auto else_body = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(x, Int(2), Sp()),
                             std::make_shared<YieldStmt>(std::vector<ExprPtr>{z}, Sp())},
        Sp());
    auto if_stmt = std::make_shared<IfStmt>(std::make_shared<ConstBool>(true, Sp()), then_body,
                                            std::optional<StmtPtr>(else_body), std::vector<VarPtr>{z}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {x, z}, if_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    ASSERT_EQ(result_if->returnVars_.size(), 2u);
    auto then_yield = std::dynamic_pointer_cast<const YieldStmt>(result_if->thenBody_->stmts_.back());
    ASSERT_NE(then_yield, nullptr);
    ASSERT_EQ(then_yield->value_.size(), 2u);
    auto phi_value = std::dynamic_pointer_cast<const Var>(then_yield->value_[0]);
    auto existing_value = std::dynamic_pointer_cast<const Var>(then_yield->value_[1]);
    ASSERT_NE(phi_value, nullptr);
    ASSERT_NE(existing_value, nullptr);
    EXPECT_NE(phi_value->name_.find("x_"), std::string::npos);
    EXPECT_NE(existing_value->name_.find("z_"), std::string::npos);
}

TEST_F(ConvertToSSAPassTest, TestRejectsUnprojectedVectorSection)
{
    auto x = MakeVar("x");
    auto section_assign = std::make_shared<AssignStmt>(x, Int(1), Sp());
    auto section = std::make_shared<SectionStmt>(SectionKind::Vector, section_assign, Sp());

    EXPECT_THROW(RunOnFunc(MakeFunc("f", {x}, section)), InternalError);
}

TEST_F(ConvertToSSAPassTest, TestPreservesVFSectionScope)
{
    auto x = MakeVar("x");
    auto section_assign = std::make_shared<AssignStmt>(x, Int(1), Sp());
    auto section = std::make_shared<SectionStmt>(SectionKind::VF, section_assign, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {x}, section));
    ASSERT_NE(result_func, nullptr);
    auto result_section = std::dynamic_pointer_cast<const SectionStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_section, nullptr);
    EXPECT_EQ(result_section->sectionKind_, SectionKind::VF);
    ASSERT_FALSE(result_section->body_->stmts_.empty());
}

TEST_F(ConvertToSSAPassTest, TestForLoopSkipsExistingIterArgAndPreservesReturnVar)
{
    auto i = MakeVar("i");
    auto x = MakeVar("x");
    auto acc_iter = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), Int(0), Sp());
    auto acc_out = MakeVar("acc_out");
    auto add = std::make_shared<Add>(acc_iter->iterVar_, i, DataType::INT32, Sp());
    auto assign_acc = std::make_shared<AssignStmt>(acc_iter->iterVar_, add, Sp());
    auto assign_x = std::make_shared<AssignStmt>(x, i, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{acc_iter->iterVar_}, Sp());
    auto for_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign_acc, assign_x, yield}, Sp());
    auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(10), Int(1), std::vector<IterArgPtr>{acc_iter}, for_body,
                                              std::vector<VarPtr>{acc_out}, Sp());
    auto body = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(x, Int(0), Sp()), for_stmt}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {}, body));
    ASSERT_NE(result_func, nullptr);
    auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[1]);
    ASSERT_NE(result_for, nullptr);
    ASSERT_EQ(result_for->iterArgs_.size(), 2u);
    ASSERT_EQ(result_for->returnVars_.size(), 2u);
    EXPECT_EQ(result_for->returnVars_[0]->name_, "acc_out");
    EXPECT_NE(result_for->returnVars_[1]->name_.find("x_"), std::string::npos);
    ASSERT_FALSE(result_for->body_->stmts_.empty());
    auto result_yield = std::dynamic_pointer_cast<const YieldStmt>(result_for->body_->stmts_.back());
    ASSERT_NE(result_yield, nullptr);
    ASSERT_EQ(result_yield->value_.size(), 2u);
}

TEST_F(ConvertToSSAPassTest, TestForLoopAppendsYieldToSeqBodyWithoutTrailingYield)
{
    auto x = MakeVar("x");
    auto y = MakeVar("y");
    auto i = MakeVar("i");
    auto init_x = std::make_shared<AssignStmt>(x, Int(0), Sp());
    auto assign_x = std::make_shared<AssignStmt>(x, i, Sp());
    auto assign_y = std::make_shared<AssignStmt>(y, Int(1), Sp());
    auto for_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign_x, assign_y}, Sp());
    auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(10), Int(1), std::vector<IterArgPtr>{}, for_body,
                                              std::vector<VarPtr>{}, Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{init_x, for_stmt}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {}, body));
    ASSERT_NE(result_func, nullptr);
    ASSERT_EQ(result_func->body_->stmts_.size(), 2u);
    auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[1]);
    ASSERT_NE(result_for, nullptr);
    ASSERT_FALSE(result_for->iterArgs_.empty());
    ASSERT_FALSE(result_for->body_->stmts_.empty());
    auto result_yield = std::dynamic_pointer_cast<const YieldStmt>(result_for->body_->stmts_.back());
    ASSERT_NE(result_yield, nullptr);
    ASSERT_EQ(result_yield->value_.size(), 1u);
}

TEST_F(ConvertToSSAPassTest, TestForLoopSkipsLoopVarAndLocalAssignments)
{
    auto i = MakeVar("i");
    auto local = MakeVar("local");
    auto assign_i = std::make_shared<AssignStmt>(i, Int(1), Sp());
    auto assign_local = std::make_shared<AssignStmt>(local, i, Sp());
    auto for_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign_i, assign_local}, Sp());
    auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(10), Int(1), std::vector<IterArgPtr>{}, for_body,
                                              std::vector<VarPtr>{}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {}, for_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_for, nullptr);
    EXPECT_TRUE(result_for->iterArgs_.empty());
    EXPECT_TRUE(result_for->returnVars_.empty());
}

TEST_F(ConvertToSSAPassTest, TestForLoopAppendsYieldToSingleStmtBody)
{
    auto x = MakeVar("x");
    auto i = MakeVar("i");
    auto init_x = std::make_shared<AssignStmt>(x, Int(0), Sp());
    auto assign_x = std::make_shared<AssignStmt>(x, i, Sp());
    auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(10), Int(1), std::vector<IterArgPtr>{}, assign_x,
                                              std::vector<VarPtr>{}, Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{init_x, for_stmt}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {}, body));
    ASSERT_NE(result_func, nullptr);
    auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[1]);
    ASSERT_NE(result_for, nullptr);
    ASSERT_EQ(result_for->body_->stmts_.size(), 2u);
    auto result_yield = std::dynamic_pointer_cast<const YieldStmt>(result_for->body_->stmts_.back());
    ASSERT_NE(result_yield, nullptr);
    ASSERT_EQ(result_yield->value_.size(), 1u);
}

TEST_F(ConvertToSSAPassTest, TestWhileLoopPreservesExistingIterArgsAndReturnVars)
{
    auto count_iter = std::make_shared<IterArg>("count", Scalar(DataType::INT32), Int(0), Sp());
    auto count_out = MakeVar("count_out");
    auto cond = std::make_shared<Lt>(count_iter->iterVar_, Int(10), DataType::BOOL, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{count_iter->iterVar_}, Sp());
    auto while_stmt = std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{count_iter}, yield,
                                                  std::vector<VarPtr>{count_out}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {}, while_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_while = std::dynamic_pointer_cast<const WhileStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_while, nullptr);
    ASSERT_EQ(result_while->iterArgs_.size(), 1u);
    ASSERT_EQ(result_while->returnVars_.size(), 1u);
    EXPECT_EQ(result_while->returnVars_[0]->name_, "count_out");
    auto result_cond = std::dynamic_pointer_cast<const Lt>(result_while->condition_);
    ASSERT_NE(result_cond, nullptr);
    auto cond_lhs = std::dynamic_pointer_cast<const Var>(result_cond->left_);
    ASSERT_NE(cond_lhs, nullptr);
    EXPECT_EQ(cond_lhs->name_, result_while->iterArgs_[0]->iterVar_->name_);
}

TEST_F(ConvertToSSAPassTest, TestTileTypeValidShapeVarsAreSubstituted)
{
    auto m = MakeVar("m");
    auto stride = Int(1);
    auto start = Int(0);
    TileView tile_view(std::vector<ExprPtr>{m}, std::vector<ExprPtr>{stride}, start);
    auto tile_type = std::make_shared<TileType>(std::vector<ExprPtr>{m}, DataType::FP32, std::optional<MemRefPtr>{},
                                                std::make_optional(tile_view));
    auto tile = MakeVar("tile", tile_type);
    auto body = std::make_shared<AssignStmt>(tile, m, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {m}, body));
    ASSERT_NE(result_func, nullptr);
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(assign, nullptr);
    auto result_type = std::dynamic_pointer_cast<const TileType>(assign->var_->GetType());
    ASSERT_NE(result_type, nullptr);
    ASSERT_TRUE(result_type->tileView_.has_value());
    ASSERT_FALSE(result_type->tileView_->validShape.empty());
    auto substituted_shape = std::dynamic_pointer_cast<const Var>(result_type->tileView_->validShape[0]);
    ASSERT_NE(substituted_shape, nullptr);
    EXPECT_EQ(substituted_shape->name_, result_func->params_[0]->name_);
}

TEST_F(ConvertToSSAPassTest, TestPtrTypeBaseAndOffsetVarsAreSubstituted)
{
    auto ptr = MakeVar("ptr", std::make_shared<PtrType>(DataType::FP32));
    auto offset = MakeVar("offset");
    auto derived_ptr_type = std::make_shared<PtrType>(DataType::FP32, ptr, offset);
    auto derived = MakeVar("derived", derived_ptr_type);
    auto body = std::make_shared<AssignStmt>(derived, ptr, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {ptr, offset}, body));
    ASSERT_NE(result_func, nullptr);
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(assign, nullptr);
    auto result_type = std::dynamic_pointer_cast<const PtrType>(assign->var_->GetType());
    ASSERT_NE(result_type, nullptr);
    ASSERT_TRUE(result_type->base_ptr.has_value());
    ASSERT_TRUE(result_type->offset.has_value());
    auto base_ptr = std::dynamic_pointer_cast<const Var>(*result_type->base_ptr);
    auto base_offset = std::dynamic_pointer_cast<const Var>(*result_type->offset);
    ASSERT_NE(base_ptr, nullptr);
    ASSERT_NE(base_offset, nullptr);
    EXPECT_EQ(base_ptr->name_, result_func->params_[0]->name_);
    EXPECT_EQ(base_offset->name_, result_func->params_[1]->name_);
}

TEST_F(ConvertToSSAPassTest, TestTensorViewPtrVarIsSubstituted)
{
    auto ptr = MakeVar("ptr", std::make_shared<PtrType>(DataType::FP32));
    TensorView tensor_view(std::vector<ExprPtr>{Int(1)}, TensorLayout::ND, ptr);
    auto tensor_type = std::make_shared<TensorType>(std::vector<ExprPtr>{Int(4)}, DataType::FP32,
                                                    std::optional<MemRefPtr>{}, std::make_optional(tensor_view));
    auto tensor = MakeVar("tensor", tensor_type);
    auto body = std::make_shared<AssignStmt>(tensor, ptr, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", {ptr}, body));
    ASSERT_NE(result_func, nullptr);
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(assign, nullptr);
    auto result_type = std::dynamic_pointer_cast<const TensorType>(assign->var_->GetType());
    ASSERT_NE(result_type, nullptr);
    ASSERT_TRUE(result_type->tensor_view_.has_value());
    ASSERT_TRUE(result_type->tensor_view_->ptr.has_value());
    auto view_ptr = std::dynamic_pointer_cast<const Var>(*result_type->tensor_view_->ptr);
    ASSERT_NE(view_ptr, nullptr);
    EXPECT_EQ(view_ptr->name_, result_func->params_[0]->name_);
}

} // namespace ir
} // namespace pypto
