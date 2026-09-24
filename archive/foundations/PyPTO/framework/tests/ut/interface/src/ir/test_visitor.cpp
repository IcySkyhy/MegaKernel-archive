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
 * \file test_visitor.cpp
 * \brief Coverage tests for IRVisitor traversal (visitor.cpp)
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
#include "ir/transforms/base/visitor.h"
#include "ir/type.h"
#include "test_ir.h"

namespace pypto {
namespace ir {

static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }
static Span Sp() { return Span("test", 1, 1); }

// Simple visitor that just ensures traversal doesn't crash and visits all nodes
class TestVisitor : public IRVisitor {
public:
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

    std::vector<std::string> visited;

    void VisitExpr_(const ConstIntPtr&) override { visited.push_back("ConstInt"); }
    void VisitExpr_(const ConstFloatPtr&) override { visited.push_back("ConstFloat"); }
    void VisitExpr_(const ConstBoolPtr&) override { visited.push_back("ConstBool"); }
    void VisitExpr_(const VarPtr&) override { visited.push_back("Var"); }
    void VisitExpr_(const MemRefPtr&) override { visited.push_back("MemRef"); }
    void VisitStmt_(const BreakStmtPtr&) override { visited.push_back("BreakStmt"); }
    void VisitStmt_(const ContinueStmtPtr&) override { visited.push_back("ContinueStmt"); }
    void VisitStmt_(const TensorOpStmtPtr&) override { visited.push_back("TensorOpStmt"); }
    void VisitStmt_(const ScalarOpStmtPtr&) override { visited.push_back("ScalarOpStmt"); }
};

// ============================================================================
// Leaf expression visitors
// ============================================================================

class IRVisitorTest : public testing::Test {};

TEST_F(IRVisitorTest, TestVisitConstInt)
{
    TestVisitor v;
    v.VisitExpr(std::make_shared<ConstInt>(42, DataType::INT32, Sp()));
    ASSERT_EQ(v.visited.back(), "ConstInt");
}

TEST_F(IRVisitorTest, TestVisitConstFloat)
{
    TestVisitor v;
    v.VisitExpr(std::make_shared<ConstFloat>(3.14, DataType::FP32, Sp()));
    ASSERT_EQ(v.visited.back(), "ConstFloat");
}

TEST_F(IRVisitorTest, TestVisitConstBool)
{
    TestVisitor v;
    v.VisitExpr(std::make_shared<ConstBool>(true, Sp()));
    ASSERT_EQ(v.visited.back(), "ConstBool");
}

TEST_F(IRVisitorTest, TestVisitVar)
{
    TestVisitor v;
    v.VisitExpr(std::make_shared<Var>("x", Scalar(DataType::INT32), Sp()));
    ASSERT_EQ(v.visited.back(), "Var");
}

TEST_F(IRVisitorTest, TestVisitVarWithTensorShape)
{
    TestVisitor v;
    auto d1 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto d2 = std::make_shared<ConstInt>(32, DataType::INT64, Sp());
    auto tt = std::make_shared<TensorType>(std::vector<ExprPtr>{d1, d2}, DataType::FP32);
    v.VisitExpr(std::make_shared<Var>("t", tt, Sp()));
    // VisitVarLike_ traverses shape — at least Var is visited
    ASSERT_GE(v.visited.size(), 1u);
}

TEST_F(IRVisitorTest, TestVisitMemRef)
{
    TestVisitor v;
    auto off = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    v.VisitExpr(std::make_shared<MemRef>(MemorySpace::DDR, off, 1024, Sp()));
    ASSERT_EQ(v.visited.back(), "MemRef");
}

TEST_F(IRVisitorTest, TestVisitCall)
{
    TestVisitor v;
    auto arg = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    v.VisitExpr(std::make_shared<Call>("op", std::vector<ExprPtr>{arg}, Sp()));
    // Call visits args
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitMakeTuple)
{
    TestVisitor v;
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    v.VisitExpr(std::make_shared<MakeTuple>(std::vector<ExprPtr>{a, b}, Sp()));
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitTupleGetItem)
{
    TestVisitor v;
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a}, Sp());
    auto idx = std::make_shared<ConstInt>(0, DataType::INDEX, Sp());
    v.VisitExpr(std::make_shared<GetItemExpr>(tup, idx, Sp()));
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitBinaryExpr)
{
    TestVisitor v;
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
#define DEFINE_BINARY_EXPR(name)                                      \
    v.VisitExpr(std::make_shared<name>(a, b, DataType::INT32, Sp())); \
    ASSERT_FALSE(v.visited.empty());
    DEFINE_BINARY_EXPR_ALL()
#undef DEFINE_BINARY_EXPR
}

TEST_F(IRVisitorTest, TestVisitUnaryExpr)
{
    TestVisitor v;
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
#define DEFINE_UNARY_EXPR(name)                                    \
    v.VisitExpr(std::make_shared<name>(a, DataType::INT32, Sp())); \
    ASSERT_FALSE(v.visited.empty());
    DEFINE_UNARY_EXPR_ALL()
#undef DEFINE_UNARY_EXPR
}

// ============================================================================
// Statement visitors
// ============================================================================

TEST_F(IRVisitorTest, TestVisitAssignStmt)
{
    TestVisitor v;
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    v.VisitStmt(std::make_shared<AssignStmt>(x, val, Sp()));
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitIfStmt)
{
    TestVisitor v;
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto thenBody = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto elseBody = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(2, DataType::INT32, Sp()), Sp());
    v.VisitStmt(std::make_shared<IfStmt>(cond, thenBody, elseBody, std::vector<VarPtr>{}, Sp()));
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitForStmt)
{
    TestVisitor v;
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto init = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto iterArg = std::make_shared<IterArg>("sum", Scalar(DataType::INT32), init, Sp());
    auto retVar = std::make_shared<Var>("sum_out", Scalar(DataType::INT32), Sp());
    auto body = std::make_shared<YieldStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, Sp())},
                                            Sp());
    v.VisitStmt(std::make_shared<ForStmt>(i, start, stop, step, std::vector<IterArgPtr>{iterArg}, body,
                                          std::vector<VarPtr>{retVar}, Sp()));
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitWhileStmt)
{
    TestVisitor v;
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    v.VisitStmt(std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, body, std::vector<VarPtr>{}, Sp()));
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitSeqStmts)
{
    TestVisitor v;
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto a1 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto a2 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(2, DataType::INT32, Sp()), Sp());
    v.VisitStmt(std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, a2}, Sp()));
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitYieldStmt)
{
    TestVisitor v;
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    v.VisitStmt(std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, Sp()));
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitReturnStmt)
{
    TestVisitor v;
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    v.VisitStmt(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{val}, Sp()));
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitEvalStmt)
{
    TestVisitor v;
    auto arg = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto call = std::make_shared<Call>("op", std::vector<ExprPtr>{arg}, Sp());
    v.VisitStmt(std::make_shared<EvalStmt>(call, Sp()));
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitBreakStmt)
{
    TestVisitor v;
    v.VisitStmt(std::make_shared<BreakStmt>(Sp()));
    ASSERT_EQ(v.visited.back(), "BreakStmt");
}

TEST_F(IRVisitorTest, TestVisitContinueStmt)
{
    TestVisitor v;
    v.VisitStmt(std::make_shared<ContinueStmt>(Sp()));
    ASSERT_EQ(v.visited.back(), "ContinueStmt");
}

TEST_F(IRVisitorTest, TestVisitScalarOpStmt)
{
    TestVisitor v;
    auto result = std::make_shared<Var>("res", Scalar(DataType::INT32), Sp());
    auto token = std::make_shared<Var>("tok", Scalar(DataType::INT32), Sp());
    auto arg = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto stmt = std::make_shared<ScalarOpStmt>(result, token, "add", std::vector<ExprPtr>{arg}, Sp());
    v.VisitStmt(stmt);
    ASSERT_EQ(v.visited.back(), "ScalarOpStmt");
}

TEST_F(IRVisitorTest, TestVisitTensorOpStmt)
{
    TestVisitor v;
    auto result = std::make_shared<Var>("res", Scalar(DataType::INT32), Sp());
    auto token = std::make_shared<Var>("tok", Scalar(DataType::INT32), Sp());
    auto arg = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto stmt = std::make_shared<TensorOpStmt>(std::vector<VarPtr>{result}, token, "matmul", std::vector<ExprPtr>{arg},
                                               std::vector<VarPtr>{}, std::vector<std::pair<std::string, std::any>>{},
                                               Sp());
    v.VisitStmt(stmt);
    ASSERT_EQ(v.visited.back(), "TensorOpStmt");
}

// ============================================================================
// Program and Function visitors
// ============================================================================

TEST_F(IRVisitorTest, TestVisitFunction)
{
    TestVisitor v;
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, Sp()), Sp());
    auto func = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                           body, Sp());
    v.VisitFunction(func);
    ASSERT_FALSE(v.visited.empty());
}

TEST_F(IRVisitorTest, TestVisitProgram)
{
    TestVisitor v;
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto f1 = std::make_shared<Function>("f1", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body, Sp());
    auto f2 = std::make_shared<Function>("f2", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body, Sp());
    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{f1, f2}, "prog", Sp());
    v.VisitProgram(prog);
    ASSERT_FALSE(v.visited.empty());
}

} // namespace ir
} // namespace pypto
