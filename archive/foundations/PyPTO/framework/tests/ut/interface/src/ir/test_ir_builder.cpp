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
 * \file test_ir_builder.cpp
 * \brief Unit tests for IRBuilder ported from Python test_ir_builder.py
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/builder.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/printer.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// Helper: shorthand ScalarType
static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }

static Span Sp() { return Span("test", 1, 1); }

// ============================================================================
// Context State Queries
// ============================================================================

class IRBuilderTest : public testing::Test {};

TEST_F(IRBuilderTest, TestContextState)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    ASSERT_FALSE(b.InFunction());
    ASSERT_FALSE(b.InLoop());
    ASSERT_FALSE(b.InIf());
    ASSERT_FALSE(b.InProgram());

    b.BeginFunction("f", sp);
    ASSERT_TRUE(b.InFunction());
    ASSERT_FALSE(b.InLoop());
    ASSERT_FALSE(b.InIf());
    ASSERT_FALSE(b.InProgram());

    auto i = b.Var("i", st, sp);
    b.BeginForLoop(i, std::make_shared<ConstInt>(0, DataType::INT32, sp),
                   std::make_shared<ConstInt>(10, DataType::INT32, sp),
                   std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    ASSERT_TRUE(b.InFunction());
    ASSERT_TRUE(b.InLoop());
    ASSERT_FALSE(b.InIf());
    b.EndForLoop(sp);

    b.BeginIf(std::make_shared<ConstBool>(true, sp), sp);
    ASSERT_TRUE(b.InFunction());
    ASSERT_TRUE(b.InIf());
    b.EndIf(sp);

    b.EndFunction(sp);

    b.BeginProgram("prog", sp);
    ASSERT_TRUE(b.InProgram());
    b.EndProgram(sp);
}

// ============================================================================
// Function Building
// ============================================================================

TEST_F(IRBuilderTest, TestEmptyFunction)
{
    IRBuilder b;
    auto sp = Sp();

    b.BeginFunction("empty_func", sp);
    auto func = b.EndFunction(sp);

    ASSERT_NE(func, nullptr);
    ASSERT_EQ(func->name_, "empty_func");
    ASSERT_EQ(func->params_.size(), 0u);
    ASSERT_EQ(func->returnTypes_.size(), 0u);
}

TEST_F(IRBuilderTest, TestFunctionWithParamsAndReturns)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("add_func", sp);
    auto x = b.FuncArg("x", st, sp);
    auto y = b.FuncArg("y", st, sp);
    b.ReturnType(st);
    b.Assign(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    auto func = b.EndFunction(sp);

    ASSERT_EQ(func->name_, "add_func");
    ASSERT_EQ(func->params_.size(), 2u);
    ASSERT_EQ(func->params_[0]->name_, "x");
    ASSERT_EQ(func->params_[1]->name_, "y");
    ASSERT_EQ(func->returnTypes_.size(), 1u);
}

TEST_F(IRBuilderTest, TestFunctionStrMatchesManual)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("test_func", sp);
    auto x = b.FuncArg("x", st, sp);
    b.ReturnType(st);
    b.Assign(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    auto builtFunc = b.EndFunction(sp);

    auto manualX = std::make_shared<Var>("x", st, sp);
    auto manualAssign = std::make_shared<AssignStmt>(manualX, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    auto manualFunc = std::make_shared<Function>("test_func", std::vector<VarPtr>{manualX}, std::vector<TypePtr>{st},
                                                 manualAssign, sp);

    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(builtFunc)),
              PythonPrint(std::static_pointer_cast<const IRNode>(manualFunc)));
}

// ============================================================================
// Statement Helpers
// ============================================================================

TEST_F(IRBuilderTest, TestVar)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    auto v = b.Var("tmp", st, sp);
    ASSERT_NE(v, nullptr);
    ASSERT_EQ(v->name_, "tmp");
}

TEST_F(IRBuilderTest, TestAssign)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);
    auto stmt = b.Assign(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    b.EndFunction(sp);

    ASSERT_NE(stmt, nullptr);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(stmt)), "x: ir.Scalar[ir.INT32] = 42");
}

TEST_F(IRBuilderTest, TestReturnWithValues)
{
    IRBuilder b;
    auto sp = Sp();

    b.BeginFunction("f", sp);
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto stmt = b.Return(std::vector<ExprPtr>{val}, sp);
    ASSERT_NE(stmt, nullptr);
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestReturnEmpty)
{
    IRBuilder b;
    auto sp = Sp();

    b.BeginFunction("g", sp);
    auto stmt = b.Return(sp);
    ASSERT_NE(stmt, nullptr);
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestEmit)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    b.FuncArg("x", st, sp);
    auto call = std::make_shared<Call>("some_op",
                                       std::vector<ExprPtr>{std::make_shared<ConstInt>(42, DataType::INT32, sp)}, sp);
    b.Emit(std::make_shared<EvalStmt>(call, sp));
    auto func = b.EndFunction(sp);

    auto evalStmt = std::dynamic_pointer_cast<const EvalStmt>(func->body_->stmts_[0]);
    ASSERT_NE(evalStmt, nullptr);
}

// ============================================================================
// For Loop Building
// ============================================================================

TEST_F(IRBuilderTest, TestForLoop)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);
    auto i = b.Var("i", st, sp);

    b.BeginForLoop(i, std::make_shared<ConstInt>(0, DataType::INT32, sp),
                   std::make_shared<ConstInt>(10, DataType::INT32, sp),
                   std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    b.Assign(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto forStmtBase = b.EndForLoop(sp);

    ASSERT_NE(forStmtBase, nullptr);
    ASSERT_EQ(forStmtBase->GetKind(), ObjectKind::ForStmt);
    auto forStmt = std::dynamic_pointer_cast<const ForStmt>(forStmtBase);
    ASSERT_EQ(forStmt->loopVar_->name_, "i");
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestForLoopWithIterArgs)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto i = b.Var("i", st, sp);

    b.BeginForLoop(i, std::make_shared<ConstInt>(0, DataType::INT32, sp),
                   std::make_shared<ConstInt>(10, DataType::INT32, sp),
                   std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);

    auto initVal = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto iterArg = std::make_shared<IterArg>("sum", st, initVal, sp);
    b.AddIterArg(iterArg);

    auto retVar = b.Var("sum_out", st, sp);
    b.AddReturnVar(retVar);

    b.Emit(std::make_shared<YieldStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, sp)}, sp));
    auto forStmtBase = b.EndForLoop(sp);

    ASSERT_EQ(forStmtBase->GetKind(), ObjectKind::ForStmt);
    auto forStmt = std::dynamic_pointer_cast<const ForStmt>(forStmtBase);
    ASSERT_EQ(forStmt->iterArgs_.size(), 1u);
    ASSERT_EQ(forStmt->returnVars_.size(), 1u);
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestForLoopStrMatchesManual)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto i = b.Var("i", st, sp);

    b.BeginForLoop(i, std::make_shared<ConstInt>(0, DataType::INT32, sp),
                   std::make_shared<ConstInt>(10, DataType::INT32, sp),
                   std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto initVal = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto iterArg = std::make_shared<IterArg>("sum", st, initVal, sp);
    b.AddIterArg(iterArg);
    auto retVar = std::make_shared<Var>("sum_out", st, sp);
    b.AddReturnVar(retVar);
    b.Emit(std::make_shared<YieldStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, sp)}, sp));
    auto builtFor = b.EndForLoop(sp);
    b.EndFunction(sp);

    auto manualI = std::make_shared<Var>("i", st, sp);
    auto manualIterArg = std::make_shared<IterArg>("sum", st, std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
    auto manualRetVar = std::make_shared<Var>("sum_out", st, sp);
    auto manualFor = std::make_shared<ForStmt>(
        manualI, std::make_shared<ConstInt>(0, DataType::INT32, sp),
        std::make_shared<ConstInt>(10, DataType::INT32, sp), std::make_shared<ConstInt>(1, DataType::INT32, sp),
        std::vector<IterArgPtr>{manualIterArg},
        std::make_shared<YieldStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, sp)}, sp),
        std::vector<VarPtr>{manualRetVar}, sp);

    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(builtFor)),
              PythonPrint(std::static_pointer_cast<const IRNode>(manualFor)));
}

// ============================================================================
// While Loop Building
// ============================================================================

TEST_F(IRBuilderTest, TestWhileLoop)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);

    b.BeginWhileLoop(std::make_shared<ConstBool>(true, sp), sp);
    b.Assign(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto whileStmtBase = b.EndWhileLoop(sp);

    ASSERT_EQ(whileStmtBase->GetKind(), ObjectKind::WhileStmt);
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestWhileLoopWithIterArgs)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);

    b.BeginWhileLoop(std::make_shared<ConstBool>(true, sp), sp);

    auto initVal = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto iterArg = std::make_shared<IterArg>("sum", st, initVal, sp);
    b.AddWhileIterArg(iterArg);

    auto retVar = b.Var("sum_out", st, sp);
    b.AddWhileReturnVar(retVar);

    b.Assign(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto whileStmtBase = b.EndWhileLoop(sp);

    ASSERT_EQ(whileStmtBase->GetKind(), ObjectKind::WhileStmt);
    auto whileStmt = std::dynamic_pointer_cast<const WhileStmt>(whileStmtBase);
    ASSERT_EQ(whileStmt->iterArgs_.size(), 1u);
    ASSERT_EQ(whileStmt->returnVars_.size(), 1u);
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestWhileLoopSetCondition)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);

    b.BeginWhileLoop(std::make_shared<ConstBool>(true, sp), sp);

    auto initVal = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto iterArg = std::make_shared<IterArg>("cnt", st, initVal, sp);
    b.AddWhileIterArg(iterArg);

    auto newCond = std::make_shared<ConstBool>(false, sp);
    b.SetWhileLoopCondition(newCond);

    auto retVar = b.Var("cnt_out", st, sp);
    b.AddWhileReturnVar(retVar);

    b.Assign(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto whileStmtBase = b.EndWhileLoop(sp);

    ASSERT_EQ(whileStmtBase->GetKind(), ObjectKind::WhileStmt);
    auto whileStmt = std::dynamic_pointer_cast<const WhileStmt>(whileStmtBase);
    ASSERT_EQ(whileStmt->iterArgs_.size(), 1u);
    ASSERT_EQ(whileStmt->returnVars_.size(), 1u);
    b.EndFunction(sp);
}

// ============================================================================
// If Statement Building
// ============================================================================

TEST_F(IRBuilderTest, TestIf)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);

    b.BeginIf(std::make_shared<ConstBool>(true, sp), sp);
    b.Assign(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    auto ifStmtBase = b.EndIf(sp);

    ASSERT_EQ(ifStmtBase->GetKind(), ObjectKind::IfStmt);
    auto ifStmt = std::dynamic_pointer_cast<const IfStmt>(ifStmtBase);
    ASSERT_FALSE(ifStmt->elseBody_.has_value());
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestIfElse)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);

    b.BeginIf(std::make_shared<ConstBool>(true, sp), sp);
    b.Assign(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    b.BeginElse(sp);
    b.Assign(x, std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
    auto ifStmtBase = b.EndIf(sp);

    ASSERT_EQ(ifStmtBase->GetKind(), ObjectKind::IfStmt);
    auto ifStmt = std::dynamic_pointer_cast<const IfStmt>(ifStmtBase);
    ASSERT_TRUE(ifStmt->elseBody_.has_value());
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestIfWithReturnVars)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);
    auto retVar = b.Var("out", st, sp);

    b.BeginIf(std::make_shared<ConstBool>(true, sp), sp);
    b.AddIfReturnVar(retVar);
    b.Assign(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    auto ifStmtBase = b.EndIf(sp);

    ASSERT_EQ(ifStmtBase->GetKind(), ObjectKind::IfStmt);
    auto ifStmt = std::dynamic_pointer_cast<const IfStmt>(ifStmtBase);
    ASSERT_EQ(ifStmt->returnVars_.size(), 1u);
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestIfElseStrMatchesManual)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);
    auto val42 = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto val0 = std::make_shared<ConstInt>(0, DataType::INT32, sp);

    b.BeginIf(std::make_shared<ConstBool>(true, sp), sp);
    b.Assign(x, val42, sp);
    b.BeginElse(sp);
    b.Assign(x, val0, sp);
    auto builtIf = b.EndIf(sp);
    b.EndFunction(sp);

    auto manualX = std::make_shared<Var>("x", st, sp);
    auto manualThen = std::make_shared<AssignStmt>(manualX, val42, sp);
    auto manualElse = std::make_shared<AssignStmt>(manualX, val0, sp);
    auto manualIf = std::make_shared<IfStmt>(std::make_shared<ConstBool>(true, sp), manualThen, manualElse,
                                             std::vector<VarPtr>{}, sp);

    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(builtIf)),
              PythonPrint(std::static_pointer_cast<const IRNode>(manualIf)));
}

// ============================================================================
// Program Building
// ============================================================================

TEST_F(IRBuilderTest, TestProgram)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("func_a", sp);
    auto x = b.FuncArg("x", st, sp);
    b.ReturnType(st);
    b.Assign(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto funcA = b.EndFunction(sp);

    b.BeginFunction("func_b", sp);
    auto y = b.FuncArg("y", st, sp);
    b.ReturnType(st);
    b.Assign(y, std::make_shared<ConstInt>(2, DataType::INT32, sp), sp);
    auto funcB = b.EndFunction(sp);

    b.BeginProgram("test_prog", sp);
    b.AddFunction(funcA);
    b.AddFunction(funcB);
    auto prog = b.EndProgram(sp);

    ASSERT_NE(prog, nullptr);
    ASSERT_EQ(prog->name_, "test_prog");
    ASSERT_EQ(prog->functions_.size(), 2u);
    ASSERT_NE(prog->GetFunction("func_a"), nullptr);
    ASSERT_NE(prog->GetFunction("func_b"), nullptr);
}

TEST_F(IRBuilderTest, TestProgramFunctionsInsertionOrder)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("zebra", sp);
    auto z = b.FuncArg("z", st, sp);
    b.Assign(z, std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
    auto funcZ = b.EndFunction(sp);

    b.BeginFunction("alpha", sp);
    auto a = b.FuncArg("a", st, sp);
    b.Assign(a, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto funcA = b.EndFunction(sp);

    b.BeginProgram("prog", sp);
    b.AddFunction(funcZ);
    b.AddFunction(funcA);
    auto prog = b.EndProgram(sp);

    ASSERT_TRUE(prog->functions_.count("zebra"));
    ASSERT_TRUE(prog->functions_.count("alpha"));
    ASSERT_EQ(prog->functions_.at("zebra")->name_, "zebra");
    ASSERT_EQ(prog->functions_.at("alpha")->name_, "alpha");
}

TEST_F(IRBuilderTest, TestGetFunctionReturnTypes)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginProgram("prog", sp);

    b.BeginFunction("foo", sp);
    b.FuncArg("x", st, sp);
    b.ReturnType(st);
    b.ReturnType(st);
    auto func = b.EndFunction(sp);
    b.AddFunction(func);

    ASSERT_EQ(func->returnTypes_.size(), 2u);

    b.EndProgram(sp);
}

// ============================================================================
// Nested Constructs
// ============================================================================

TEST_F(IRBuilderTest, TestNestedForLoops)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);
    auto i = b.Var("i", st, sp);
    auto j = b.Var("j", st, sp);

    b.BeginForLoop(i, std::make_shared<ConstInt>(0, DataType::INT32, sp),
                   std::make_shared<ConstInt>(10, DataType::INT32, sp),
                   std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    b.BeginForLoop(j, std::make_shared<ConstInt>(0, DataType::INT32, sp),
                   std::make_shared<ConstInt>(5, DataType::INT32, sp),
                   std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    b.Assign(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto innerBase = b.EndForLoop(sp);
    auto outerBase = b.EndForLoop(sp);

    ASSERT_EQ(innerBase->GetKind(), ObjectKind::ForStmt);
    ASSERT_EQ(outerBase->GetKind(), ObjectKind::ForStmt);
    auto inner = std::dynamic_pointer_cast<const ForStmt>(innerBase);
    auto outer = std::dynamic_pointer_cast<const ForStmt>(outerBase);
    ASSERT_EQ(outer->loopVar_->name_, "i");
    ASSERT_EQ(inner->loopVar_->name_, "j");

    auto func = b.EndFunction(sp);
    auto funcBody = std::dynamic_pointer_cast<const ForStmt>(func->body_->stmts_[0]);
    ASSERT_NE(funcBody, nullptr);
    auto innerBody = std::dynamic_pointer_cast<const ForStmt>(funcBody->body_->stmts_[0]);
    ASSERT_NE(innerBody, nullptr);
}

TEST_F(IRBuilderTest, TestForWithIf)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);
    auto i = b.Var("i", st, sp);

    b.BeginForLoop(i, std::make_shared<ConstInt>(0, DataType::INT32, sp),
                   std::make_shared<ConstInt>(10, DataType::INT32, sp),
                   std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    b.BeginIf(std::make_shared<ConstBool>(true, sp), sp);
    b.Assign(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    b.EndIf(sp);
    b.EndForLoop(sp);

    auto func = b.EndFunction(sp);
    auto forStmt = std::dynamic_pointer_cast<const ForStmt>(func->body_->stmts_[0]);
    ASSERT_NE(forStmt, nullptr);
    auto ifStmt = std::dynamic_pointer_cast<const IfStmt>(forStmt->body_->stmts_[0]);
    ASSERT_NE(ifStmt, nullptr);
}

TEST_F(IRBuilderTest, TestIfWithNestedFor)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);
    auto i = b.Var("i", st, sp);

    b.BeginIf(std::make_shared<ConstBool>(true, sp), sp);
    b.BeginForLoop(i, std::make_shared<ConstInt>(0, DataType::INT32, sp),
                   std::make_shared<ConstInt>(5, DataType::INT32, sp),
                   std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    b.Assign(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    b.EndForLoop(sp);
    b.EndIf(sp);

    auto func = b.EndFunction(sp);
    auto ifStmt = std::dynamic_pointer_cast<const IfStmt>(func->body_->stmts_[0]);
    ASSERT_NE(ifStmt, nullptr);
    auto forStmt = std::dynamic_pointer_cast<const ForStmt>(ifStmt->thenBody_->stmts_[0]);
    ASSERT_NE(forStmt, nullptr);
}

// ============================================================================
// Complex End-to-End
// ============================================================================

TEST_F(IRBuilderTest, TestComplexProgram)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    // Build: def compute(x: int32) -> int32:
    //   for i in range(0, 10, 1):
    //     if i < 5:
    //       x = x + 1
    //     else:
    //       x = x - 1
    //   return x
    b.BeginFunction("compute", sp);
    auto x = b.FuncArg("x", st, sp);
    auto i = b.Var("i", st, sp);
    b.ReturnType(st);

    b.BeginForLoop(i, std::make_shared<ConstInt>(0, DataType::INT32, sp),
                   std::make_shared<ConstInt>(10, DataType::INT32, sp),
                   std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);

    auto cond = std::make_shared<Lt>(i, std::make_shared<ConstInt>(5, DataType::INT32, sp), DataType::INT32, sp);
    b.BeginIf(cond, sp);
    b.Assign(x, std::make_shared<Add>(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), DataType::INT32, sp), sp);
    b.BeginElse(sp);
    b.Assign(x, std::make_shared<Sub>(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), DataType::INT32, sp), sp);
    b.EndIf(sp);
    b.EndForLoop(sp);

    b.Return(std::vector<ExprPtr>{x}, sp);
    auto compute = b.EndFunction(sp);

    b.BeginProgram("my_prog", sp);
    b.AddFunction(compute);
    auto prog = b.EndProgram(sp);

    ASSERT_NE(prog, nullptr);
    ASSERT_EQ(prog->name_, "my_prog");
    ASSERT_EQ(prog->functions_.size(), 1u);
    ASSERT_NE(prog->GetFunction("compute"), nullptr);

    auto func = prog->GetFunction("compute");
    ASSERT_NE(func, nullptr);

    // Body should be SeqStmts (for + return)
    auto seqBody = func->body_;
    ASSERT_NE(seqBody, nullptr);
    ASSERT_EQ(seqBody->stmts_.size(), 2u);

    auto forStmt = std::dynamic_pointer_cast<const ForStmt>(seqBody->stmts_[0]);
    ASSERT_NE(forStmt, nullptr);

    auto retStmt = std::dynamic_pointer_cast<const ReturnStmt>(seqBody->stmts_[1]);
    ASSERT_NE(retStmt, nullptr);

    // For body should be IfStmt with else
    auto ifBody = std::dynamic_pointer_cast<const IfStmt>(forStmt->body_->stmts_[0]);
    ASSERT_NE(ifBody, nullptr);
    ASSERT_TRUE(ifBody->elseBody_.has_value());
}

// ============================================================================
// Section Building
// ============================================================================

TEST_F(IRBuilderTest, TestSectionVector)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);

    b.BeginSection(SectionKind::Vector, sp);
    b.Assign(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto sectionStmtBase = b.EndSection(sp);

    ASSERT_NE(sectionStmtBase, nullptr);
    ASSERT_EQ(sectionStmtBase->GetKind(), ObjectKind::SectionStmt);
    auto sectionStmt = std::dynamic_pointer_cast<const SectionStmt>(sectionStmtBase);
    ASSERT_EQ(sectionStmt->sectionKind_, SectionKind::Vector);
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestSectionCube)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);

    b.BeginSection(SectionKind::Cube, sp);
    b.Assign(x, std::make_shared<ConstInt>(2, DataType::INT32, sp), sp);
    auto sectionStmtBase = b.EndSection(sp);

    ASSERT_NE(sectionStmtBase, nullptr);
    ASSERT_EQ(sectionStmtBase->GetKind(), ObjectKind::SectionStmt);
    auto sectionStmt = std::dynamic_pointer_cast<const SectionStmt>(sectionStmtBase);
    ASSERT_EQ(sectionStmt->sectionKind_, SectionKind::Cube);
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestSectionEmpty)
{
    IRBuilder b;
    auto sp = Sp();

    b.BeginFunction("f", sp);
    b.BeginSection(SectionKind::Vector, sp);
    auto sectionStmtBase = b.EndSection(sp);

    ASSERT_NE(sectionStmtBase, nullptr);
    ASSERT_EQ(sectionStmtBase->GetKind(), ObjectKind::SectionStmt);
    auto sectionStmt = std::dynamic_pointer_cast<const SectionStmt>(sectionStmtBase);
    auto body = std::dynamic_pointer_cast<const SeqStmts>(sectionStmt->body_);
    ASSERT_NE(body, nullptr);
    ASSERT_EQ(body->stmts_.size(), 0u);
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestSectionMultipleStmts)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);
    auto y = b.Var("y", st, sp);

    b.BeginSection(SectionKind::Vector, sp);
    b.Assign(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    b.Assign(y, std::make_shared<ConstInt>(2, DataType::INT32, sp), sp);
    auto sectionStmtBase = b.EndSection(sp);

    ASSERT_NE(sectionStmtBase, nullptr);
    auto sectionStmt = std::dynamic_pointer_cast<const SectionStmt>(sectionStmtBase);
    auto body = std::dynamic_pointer_cast<const SeqStmts>(sectionStmt->body_);
    ASSERT_NE(body, nullptr);
    ASSERT_EQ(body->stmts_.size(), 2u);
    b.EndFunction(sp);
}

TEST_F(IRBuilderTest, TestSectionNestedInFor)
{
    IRBuilder b;
    auto sp = Sp();
    auto st = Scalar(DataType::INT32);

    b.BeginFunction("f", sp);
    auto x = b.FuncArg("x", st, sp);
    auto i = b.Var("i", st, sp);

    b.BeginForLoop(i, std::make_shared<ConstInt>(0, DataType::INT32, sp),
                   std::make_shared<ConstInt>(10, DataType::INT32, sp),
                   std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    b.BeginSection(SectionKind::Cube, sp);
    b.Assign(x, std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    b.EndSection(sp);
    b.EndForLoop(sp);

    auto func = b.EndFunction(sp);
    auto forStmt = std::dynamic_pointer_cast<const ForStmt>(func->body_->stmts_[0]);
    ASSERT_NE(forStmt, nullptr);
    auto sectionStmt = std::dynamic_pointer_cast<const SectionStmt>(forStmt->body_->stmts_[0]);
    ASSERT_NE(sectionStmt, nullptr);
    ASSERT_EQ(sectionStmt->sectionKind_, SectionKind::Cube);
}

} // namespace ir
} // namespace pypto
