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
 * \file test_mutator.cpp
 * \brief Coverage tests for IRMutator transformation (mutator.cpp)
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
#include "ir/transforms/base/mutator.h"
#include "ir/type.h"
#include "test_ir.h"

namespace pypto {
namespace ir {

static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }
static Span Sp() { return Span("test", 1, 1); }

class IdentityMutator : public IRMutator {};

class IRMutatorTest : public testing::Test {};

// ============================================================================
// Identity Mutator — copy-on-write returns originals
// ============================================================================

TEST_F(IRMutatorTest, TestIdentityAllExprs)
{
    IdentityMutator m;
    auto ci = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    EXPECT_EQ(m.VisitExpr(ci).get(), ci.get());
    auto cf = std::make_shared<ConstFloat>(3.14, DataType::FP32, Sp());
    EXPECT_EQ(m.VisitExpr(cf).get(), cf.get());
    auto cb = std::make_shared<ConstBool>(true, Sp());
    EXPECT_EQ(m.VisitExpr(cb).get(), cb.get());
    auto v = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    EXPECT_EQ(m.VisitExpr(v).get(), v.get());

    auto off = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    auto mr = std::make_shared<MemRef>(MemorySpace::DDR, off, 1024, Sp());
    EXPECT_EQ(m.VisitExpr(mr).get(), mr.get());

    auto call = std::make_shared<Call>("op", std::vector<ExprPtr>{ci}, Sp());
    EXPECT_EQ(m.VisitExpr(call).get(), call.get());

    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto mt = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a, b}, Sp());
    EXPECT_EQ(m.VisitExpr(mt).get(), mt.get());

    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a}, Sp());
    auto idx = std::make_shared<ConstInt>(0, DataType::INDEX, Sp());
    auto tgi = std::make_shared<GetItemExpr>(tup, idx, Sp());
    EXPECT_EQ(m.VisitExpr(tgi).get(), tgi.get());

// Binary expressions
#define DEFINE_BINARY_EXPR(name)                                         \
    {                                                                    \
        auto expr = std::make_shared<name>(a, b, DataType::INT32, Sp()); \
        EXPECT_EQ(m.VisitExpr(expr).get(), expr.get());                  \
    }
    DEFINE_BINARY_EXPR_ALL()
#undef DEFINE_BINARY_EXPR

// Unary expressions
#define DEFINE_UNARY_EXPR(name)                                       \
    {                                                                 \
        auto expr = std::make_shared<name>(a, DataType::INT32, Sp()); \
        EXPECT_EQ(m.VisitExpr(expr).get(), expr.get());               \
    }
    DEFINE_UNARY_EXPR_ALL()
#undef DEFINE_UNARY_EXPR
}

TEST_F(IRMutatorTest, TestIdentityAllStmts)
{
    IdentityMutator m;
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto iterArg = std::make_shared<IterArg>("sum", Scalar(DataType::INT32), zero, Sp());
    auto retVar = std::make_shared<Var>("sum_out", Scalar(DataType::INT32), Sp());

    auto assign = std::make_shared<AssignStmt>(x, val, Sp());
    EXPECT_EQ(m.VisitStmt(assign).get(), assign.get());

    auto ifStmt = std::make_shared<IfStmt>(cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp());
    EXPECT_EQ(m.VisitStmt(ifStmt).get(), ifStmt.get());

    auto ifElse = std::make_shared<IfStmt>(cond, yield, yield, std::vector<VarPtr>{retVar}, Sp());
    EXPECT_EQ(m.VisitStmt(ifElse).get(), ifElse.get());

    auto yieldS = std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, Sp());
    EXPECT_EQ(m.VisitStmt(yieldS).get(), yieldS.get());

    auto ret = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{val}, Sp());
    EXPECT_EQ(m.VisitStmt(ret).get(), ret.get());

    auto forS = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{iterArg}, yield,
                                          std::vector<VarPtr>{retVar}, Sp());
    EXPECT_EQ(m.VisitStmt(forS).get(), forS.get());

    auto whileS = std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{}, Sp());
    EXPECT_EQ(m.VisitStmt(whileS).get(), whileS.get());

    auto a1 = std::make_shared<AssignStmt>(x, one, Sp());
    auto a2 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(2, DataType::INT32, Sp()), Sp());
    auto seq = std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, a2}, Sp());
    EXPECT_EQ(m.VisitStmt(seq).get(), seq.get());

    auto call = std::make_shared<Call>("op", std::vector<ExprPtr>{}, Sp());
    auto eval = std::make_shared<EvalStmt>(call, Sp());
    EXPECT_EQ(m.VisitStmt(eval).get(), eval.get());

    auto brk = std::make_shared<BreakStmt>(Sp());
    EXPECT_EQ(m.VisitStmt(brk).get(), brk.get());

    auto cont = std::make_shared<ContinueStmt>(Sp());
    EXPECT_EQ(m.VisitStmt(cont).get(), cont.get());

    // ScalarOpStmt
    auto scalarRes = std::make_shared<Var>("res", Scalar(DataType::INT32), Sp());
    auto scalarTok = std::make_shared<Var>("tok", Scalar(DataType::INT32), Sp());
    auto scalarArg = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto scalarOp = std::make_shared<ScalarOpStmt>(scalarRes, scalarTok, "add", std::vector<ExprPtr>{scalarArg}, Sp());
    EXPECT_EQ(m.VisitStmt(scalarOp).get(), scalarOp.get());

    // TensorOpStmt
    auto tensorRes = std::make_shared<Var>("res", Scalar(DataType::INT32), Sp());
    auto tensorTok = std::make_shared<Var>("tok", Scalar(DataType::INT32), Sp());
    auto tensorArg = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto tensorOp = std::make_shared<TensorOpStmt>(std::vector<VarPtr>{tensorRes}, tensorTok, "matmul",
                                                   std::vector<ExprPtr>{tensorArg}, std::vector<VarPtr>{},
                                                   std::vector<std::pair<std::string, std::any>>{}, Sp());
    EXPECT_EQ(m.VisitStmt(tensorOp).get(), tensorOp.get());
}

TEST_F(IRMutatorTest, TestIdentityFunctionAndProgram)
{
    IdentityMutator m;
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, Sp()), Sp());
    auto func = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                           body, Sp());
    EXPECT_EQ(m.VisitFunction(func).get(), func.get());

    auto f1 = std::make_shared<Function>("f1", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body, Sp());
    auto f2 = std::make_shared<Function>("f2", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body, Sp());
    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{f1, f2}, "prog", Sp());
    EXPECT_EQ(m.VisitProgram(prog).get(), prog.get());
}

// ============================================================================
// Mutator that changes ConstInt(42) → ConstInt(99)
// ============================================================================

class ConstRewriter : public IRMutator {
public:
    using IRMutator::VisitExpr_;
    ExprPtr VisitExpr_(const ConstIntPtr& op) override
    {
        if (op->value_ == 42) {
            return std::make_shared<ConstInt>(99, std::dynamic_pointer_cast<const ScalarType>(op->GetType())->dtype_,
                                              op->span_);
        }
        return op;
    }
};

TEST_F(IRMutatorTest, TestRewriteConstIntAndPropagation)
{
    ConstRewriter m;
    auto val42 = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    // Direct rewrite
    auto result = m.VisitExpr(val42);
    EXPECT_NE(result.get(), val42.get());
    EXPECT_EQ(std::dynamic_pointer_cast<const ConstInt>(result)->value_, 99);

    // Propagation through binary expr
    auto add = std::make_shared<Add>(val42, one, DataType::INT32, Sp());
    EXPECT_NE(m.VisitExpr(add).get(), add.get());

    // Propagation through assign stmt
    auto assign = std::make_shared<AssignStmt>(x, val42, Sp());
    EXPECT_NE(m.VisitStmt(assign).get(), assign.get());

    // Propagation through function
    auto body = std::make_shared<AssignStmt>(x, val42, Sp());
    auto func = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                           body, Sp());
    EXPECT_NE(m.VisitFunction(func).get(), func.get());

    // Propagation through program
    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "prog", Sp());
    EXPECT_NE(m.VisitProgram(prog).get(), prog.get());
}

TEST_F(IRMutatorTest, TestRewriteAllBinaryExpr)
{
    ConstRewriter m;
    auto a = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
#define DEFINE_BINARY_EXPR(name)                                                        \
    {                                                                                   \
        auto expr = std::make_shared<name>(a, b, DataType::INT32, Sp());                \
        auto result = m.VisitExpr(expr);                                                \
        EXPECT_NE(result.get(), expr.get()) << "Expected reconstruction for " #name;    \
        auto reconstructed = std::dynamic_pointer_cast<const name>(result);             \
        EXPECT_NE(reconstructed, nullptr) << "Expected type " #name " after rebuild";   \
        auto newLeft = std::dynamic_pointer_cast<const ConstInt>(reconstructed->left_); \
        EXPECT_NE(newLeft, nullptr);                                                    \
        EXPECT_EQ(newLeft->value_, 99);                                                 \
    }
    DEFINE_BINARY_EXPR_ALL()
#undef DEFINE_BINARY_EXPR
}

TEST_F(IRMutatorTest, TestRewriteAllUnaryExpr)
{
    ConstRewriter m;
    auto a = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
#define DEFINE_UNARY_EXPR(name)                                                          \
    {                                                                                    \
        auto expr = std::make_shared<name>(a, DataType::INT32, Sp());                    \
        auto result = m.VisitExpr(expr);                                                 \
        EXPECT_NE(result.get(), expr.get()) << "Expected reconstruction for " #name;     \
        auto reconstructed = std::dynamic_pointer_cast<const name>(result);              \
        EXPECT_NE(reconstructed, nullptr) << "Expected type " #name " after rebuild";    \
        auto newOp = std::dynamic_pointer_cast<const ConstInt>(reconstructed->operand_); \
        EXPECT_NE(newOp, nullptr);                                                       \
        EXPECT_EQ(newOp->value_, 99);                                                    \
    }
    DEFINE_UNARY_EXPR_ALL()
#undef DEFINE_UNARY_EXPR
}

TEST_F(IRMutatorTest, TestRewriteScalarOpStmt)
{
    ConstRewriter m;
    auto res = std::make_shared<Var>("res", Scalar(DataType::INT32), Sp());
    auto tok = std::make_shared<Var>("tok", Scalar(DataType::INT32), Sp());
    auto val42 = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto stmt = std::make_shared<ScalarOpStmt>(res, tok, "add", std::vector<ExprPtr>{val42}, Sp());
    auto result = m.VisitStmt(stmt);
    EXPECT_NE(result.get(), stmt.get());
    auto mutated = std::dynamic_pointer_cast<const ScalarOpStmt>(result);
    EXPECT_NE(mutated, nullptr);
    auto newArg = std::dynamic_pointer_cast<const ConstInt>(mutated->args_[0]);
    EXPECT_EQ(newArg->value_, 99);
}

TEST_F(IRMutatorTest, TestRewriteTensorOpStmt)
{
    ConstRewriter m;
    auto res = std::make_shared<Var>("res", Scalar(DataType::INT32), Sp());
    auto tok = std::make_shared<Var>("tok", Scalar(DataType::INT32), Sp());
    auto val42 = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto stmt = std::make_shared<TensorOpStmt>(std::vector<VarPtr>{res}, tok, "matmul", std::vector<ExprPtr>{val42},
                                               std::vector<VarPtr>{}, std::vector<std::pair<std::string, std::any>>{},
                                               Sp());
    auto result = m.VisitStmt(stmt);
    EXPECT_NE(result.get(), stmt.get());
    auto mutated = std::dynamic_pointer_cast<const TensorOpStmt>(result);
    EXPECT_NE(mutated, nullptr);
    auto newArg = std::dynamic_pointer_cast<const ConstInt>(mutated->args_[0]);
    EXPECT_EQ(newArg->value_, 99);
}

} // namespace ir
} // namespace pypto
