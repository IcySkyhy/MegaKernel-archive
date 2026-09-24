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
 * \file test_ir.cpp
 * \brief Unit tests ported from Python test_ir.py — string representations, IRCHECK macros, and structural comparison
 */

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/printer.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

// Helper: shorthand for constructing shared ScalarType
static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }

// ============================================================================
// IRCHECK / INTERNAL_CHECK Tests
// ============================================================================

class IRCheckTest : public testing::Test {};

TEST_F(IRCheckTest, TestCheckPass) { IRCHECK(true) << "should not throw"; }

TEST_F(IRCheckTest, TestCheckFail) { ASSERT_THROW(IRCHECK(false) << "test check message", ValueError); }

TEST_F(IRCheckTest, TestInternalCheckPass) { INTERNAL_CHECK(true) << "should not throw"; }

TEST_F(IRCheckTest, TestInternalCheckFail)
{
    ASSERT_THROW(INTERNAL_CHECK(false) << "test internal check message", InternalError);
}

// ============================================================================
// String Representation — Types
// ============================================================================

class IRTypeStrTest : public testing::Test {};

TEST_F(IRTypeStrTest, TestUnknownTypeStr)
{
    auto ut = std::make_shared<UnknownType>();
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const Type>(ut)), "ir.Unknown");
}

TEST_F(IRTypeStrTest, TestScalarTypeStr)
{
    auto st = std::make_shared<ScalarType>(DataType::INT32);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const Type>(st)), "ir.Scalar[ir.INT32]");
}

TEST_F(IRTypeStrTest, TestTensorTypeStr)
{
    auto dim1 = std::make_shared<ConstInt>(16, DataType::INT64, Span::Unknown());
    auto dim2 = std::make_shared<ConstInt>(32, DataType::INT64, Span::Unknown());
    auto tt = std::make_shared<TensorType>(std::vector<ExprPtr>{dim1, dim2}, DataType::FP32);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const Type>(tt)), "ir.Tensor[[16, 32], ir.FP32]");
}

TEST_F(IRTypeStrTest, TestTensorTypeWithMemRefStr)
{
    auto dim1 = std::make_shared<ConstInt>(16, DataType::INT64, Span::Unknown());
    auto dim2 = std::make_shared<ConstInt>(32, DataType::INT64, Span::Unknown());
    auto offset = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRefPtr memref = std::make_shared<MemRef>(MemorySpace::DDR, offset, 1024);
    auto tt = std::make_shared<TensorType>(std::vector<ExprPtr>{dim1, dim2}, DataType::FP16, memref);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const Type>(tt)),
              "ir.Tensor[[16, 32], ir.FP16, ir.MemRef(ir.MemorySpace.DDR, 0, 1024)]");
}

TEST_F(IRTypeStrTest, TestTupleTypeStr)
{
    auto t1 = std::make_shared<ScalarType>(DataType::INT32);
    auto t2 = std::make_shared<ScalarType>(DataType::FP32);
    auto tup = std::make_shared<TupleType>(std::vector<TypePtr>{t1, t2});
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const Type>(tup)),
              "ir.Tuple[ir.Scalar[ir.INT32], ir.Scalar[ir.FP32]]");
}

TEST_F(IRTypeStrTest, TestPtrTypeStr)
{
    auto pt = std::make_shared<PtrType>(DataType::FP32);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const Type>(pt)), "ir.Ptr");
}

// ============================================================================
// String Representation — Expressions
// ============================================================================

class IRExprStrTest : public testing::Test {
protected:
    Span sp = Span("test", 1, 1);
};

TEST_F(IRExprStrTest, TestConstIntStr)
{
    auto ci = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(ci)), "42");
}

TEST_F(IRExprStrTest, TestConstFloatStr)
{
    auto cf = std::make_shared<ConstFloat>(3.14, DataType::FP32, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(cf)), "3.14");
}

TEST_F(IRExprStrTest, TestConstBoolStr)
{
    auto cb = std::make_shared<ConstBool>(true, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(cb)), "True");
}

TEST_F(IRExprStrTest, TestVarStr)
{
    auto var = std::make_shared<Var>("x", Scalar(DataType::INT32), sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(var)), "x");
}

TEST_F(IRExprStrTest, TestBinaryOpsStr)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, sp);

    struct BinOpTest {
        std::string opName;
        ExprPtr expr;
        std::string expected;
    };

    std::vector<BinOpTest> tests = {
        {"Add", std::make_shared<Add>(a, b, DataType::INT32, sp), "1 + 2"},
        {"Sub", std::make_shared<Sub>(a, b, DataType::INT32, sp), "1 - 2"},
        {"Mul", std::make_shared<Mul>(a, b, DataType::INT32, sp), "1 * 2"},
        {"FloorDiv", std::make_shared<FloorDiv>(a, b, DataType::INT32, sp), "1 // 2"},
        {"FloatDiv", std::make_shared<FloatDiv>(a, b, DataType::INT32, sp), "1 / 2"},
        {"FloorMod", std::make_shared<FloorMod>(a, b, DataType::INT32, sp), "1 % 2"},
        {"Pow", std::make_shared<Pow>(a, b, DataType::INT32, sp), "1 ** 2"},
        {"Eq", std::make_shared<Eq>(a, b, DataType::INT32, sp), "1 == 2"},
        {"Ne", std::make_shared<Ne>(a, b, DataType::INT32, sp), "1 != 2"},
        {"Lt", std::make_shared<Lt>(a, b, DataType::INT32, sp), "1 < 2"},
        {"Le", std::make_shared<Le>(a, b, DataType::INT32, sp), "1 <= 2"},
        {"Gt", std::make_shared<Gt>(a, b, DataType::INT32, sp), "1 > 2"},
        {"Ge", std::make_shared<Ge>(a, b, DataType::INT32, sp), "1 >= 2"},
        {"And", std::make_shared<And>(a, b, DataType::INT32, sp), "1 and 2"},
        {"Or", std::make_shared<Or>(a, b, DataType::INT32, sp), "1 or 2"},
        {"Xor", std::make_shared<Xor>(a, b, DataType::INT32, sp), "1 xor 2"},
        {"BitAnd", std::make_shared<BitAnd>(a, b, DataType::INT32, sp), "1 & 2"},
        {"BitOr", std::make_shared<BitOr>(a, b, DataType::INT32, sp), "1 | 2"},
        {"BitXor", std::make_shared<BitXor>(a, b, DataType::INT32, sp), "1 ^ 2"},
        {"BitShiftLeft", std::make_shared<BitShiftLeft>(a, b, DataType::INT32, sp), "1 << 2"},
        {"BitShiftRight", std::make_shared<BitShiftRight>(a, b, DataType::INT32, sp), "1 >> 2"},
    };

    for (const auto& t : tests) {
        auto node = std::static_pointer_cast<const IRNode>(t.expr);
        ASSERT_EQ(PythonPrint(node), t.expected) << "Failed for op: " << t.opName;
    }
}

TEST_F(IRExprStrTest, TestMinMaxStr)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, sp);

    auto minExpr = std::make_shared<Min>(a, b, DataType::INT32, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(minExpr)), "ir.min(1, 2)");

    auto maxExpr = std::make_shared<Max>(a, b, DataType::INT32, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(maxExpr)), "ir.max(1, 2)");
}

TEST_F(IRExprStrTest, TestUnaryOpsStr)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto cb = std::make_shared<ConstBool>(true, sp);

    auto neg = std::make_shared<Neg>(a, DataType::INT32, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(neg)), "-1");

    auto notExpr = std::make_shared<Not>(cb, DataType::BOOL, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(notExpr)), "not True");

    auto absExpr = std::make_shared<Abs>(a, DataType::INT32, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(absExpr)), "ir.abs(1)");

    auto cast = std::make_shared<Cast>(a, DataType::FP32, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(cast)), "ir.cast(1, ir.FP32)");
}

TEST_F(IRExprStrTest, TestMakeTupleAndGetItemStr)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, sp);

    auto mt = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a, b}, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(mt)), "[1, 2]");

    auto idx = std::make_shared<ConstInt>(0, DataType::INDEX, sp);
    auto tgi = std::make_shared<GetItemExpr>(mt, idx, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(tgi)), "[1, 2][0]");
}

TEST_F(IRExprStrTest, TestCallStr)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, sp);
    auto call = std::make_shared<Call>("my_op", std::vector<ExprPtr>{a, b}, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(call)), "ir.call @my_op(1, 2)");
}

TEST_F(IRExprStrTest, TestMemRefStr)
{
    auto offset = std::make_shared<ConstInt>(0, DataType::INT64, sp);
    auto memref = std::make_shared<MemRef>(MemorySpace::Vec, offset, 2048, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(memref)), "ir.MemRef(ir.MemorySpace.Vec, 0, 2048)");
}

// ============================================================================
// String Representation — Statements
// ============================================================================

class IRStmtStrTest : public testing::Test {
protected:
    Span sp = Span("test", 1, 1);
    TypePtr st = Scalar(DataType::INT32);
};

TEST_F(IRStmtStrTest, TestAssignStmtStr)
{
    auto x = std::make_shared<Var>("x", st, sp);
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto assign = std::make_shared<AssignStmt>(x, val, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(assign)), "x: ir.Scalar[ir.INT32] = 42");
}

TEST_F(IRStmtStrTest, TestSeqStmtsStr)
{
    auto x = std::make_shared<Var>("x", st, sp);
    auto y = std::make_shared<Var>("y", st, sp);
    auto assign_x = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    auto assign_y = std::make_shared<AssignStmt>(y, std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
    auto seq = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign_x, assign_y}, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(seq)),
              "x: ir.Scalar[ir.INT32] = 42\ny: ir.Scalar[ir.INT32] = 0");
}

TEST_F(IRStmtStrTest, TestIfStmtStr)
{
    auto x = std::make_shared<Var>("x", st, sp);
    auto cond = std::make_shared<ConstBool>(true, sp);
    auto assign = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);

    auto ifStmt = std::make_shared<IfStmt>(cond, assign, std::nullopt, std::vector<VarPtr>{}, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(ifStmt)), "if True:\n    x: ir.Scalar[ir.INT32] = 42");
}

TEST_F(IRStmtStrTest, TestIfElseStmtStr)
{
    auto x = std::make_shared<Var>("x", st, sp);
    auto y = std::make_shared<Var>("y", st, sp);
    auto cond = std::make_shared<ConstBool>(true, sp);
    auto thenBody = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    auto elseBody = std::make_shared<AssignStmt>(y, std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);

    auto ifElse = std::make_shared<IfStmt>(cond, thenBody, elseBody, std::vector<VarPtr>{}, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(ifElse)),
              "if True:\n    x: ir.Scalar[ir.INT32] = 42\nelse:\n    y: ir.Scalar[ir.INT32] = 0");
}

TEST_F(IRStmtStrTest, TestForStmtStr)
{
    auto i = std::make_shared<Var>("i", st, sp);
    auto init = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto iterArg = std::make_shared<IterArg>("sum", st, init, sp);
    auto retVar = std::make_shared<Var>("sum_out", st, sp);
    auto body = std::make_shared<YieldStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, sp)},
                                            sp);

    auto forStmt = std::make_shared<ForStmt>(i, std::make_shared<ConstInt>(0, DataType::INT32, sp),
                                             std::make_shared<ConstInt>(10, DataType::INT32, sp),
                                             std::make_shared<ConstInt>(1, DataType::INT32, sp),
                                             std::vector<IterArgPtr>{iterArg}, body, std::vector<VarPtr>{retVar}, sp);

    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(forStmt)),
              "for i, (sum,) in ir.range(0, 10, 1, init_values=(0,)):\n"
              "    sum_out = ir.yield_(1)");
}

TEST_F(IRStmtStrTest, TestWhileStmtStr)
{
    auto x = std::make_shared<Var>("x", st, sp);
    auto cond = std::make_shared<ConstBool>(true, sp);
    auto assign = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);

    auto whileStmt = std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, assign, std::vector<VarPtr>{}, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(whileStmt)),
              "while True:\n    x: ir.Scalar[ir.INT32] = 42");
}

TEST_F(IRStmtStrTest, TestYieldStmtStr)
{
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(yield)), "ir.yield_(42)");

    auto emptyYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(emptyYield)), "ir.yield_()");
}

TEST_F(IRStmtStrTest, TestReturnStmtStr)
{
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto ret = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{val}, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(ret)), "return 42");

    auto emptyRet = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(emptyRet)), "return");
}

TEST_F(IRStmtStrTest, TestBreakContinueStmtStr)
{
    auto brk = std::make_shared<BreakStmt>(sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(brk)), "break");

    auto cont = std::make_shared<ContinueStmt>(sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(cont)), "continue");
}

TEST_F(IRStmtStrTest, TestEvalStmtStr)
{
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto call = std::make_shared<Call>("some_op", std::vector<ExprPtr>{val}, sp);
    auto eval = std::make_shared<EvalStmt>(call, sp);
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(eval)), "ir.eval(ir.call @some_op(42))");
}

TEST_F(IRStmtStrTest, TestFunctionStr)
{
    auto x = std::make_shared<Var>("x", st, sp);
    auto assign = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    auto func = std::make_shared<Function>("test_func", std::vector<VarPtr>{x}, std::vector<TypePtr>{st}, assign, sp);

    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(func)),
              "@ir.function\ndef test_func(x: ir.Scalar[ir.INT32]) -> ir.Scalar[ir.INT32]:\n"
              "    x: ir.Scalar[ir.INT32] = 42");
}

TEST_F(IRStmtStrTest, TestProgramStr)
{
    auto x = std::make_shared<Var>("x", st, sp);
    auto assign = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    auto func1 = std::make_shared<Function>("test_func", std::vector<VarPtr>{x}, std::vector<TypePtr>{st}, assign, sp);
    auto func2 = std::make_shared<Function>("test_func2", std::vector<VarPtr>{x}, std::vector<TypePtr>{st}, assign, sp);

    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func1, func2}, "test_prog", sp);

    std::string expected = "# ir.program: test_prog\n"
                           "@ir.function\n"
                           "def test_func(x: ir.Scalar[ir.INT32]) -> ir.Scalar[ir.INT32]:\n"
                           "    x: ir.Scalar[ir.INT32] = 42\n"
                           "@ir.function\n"
                           "def test_func2(x: ir.Scalar[ir.INT32]) -> ir.Scalar[ir.INT32]:\n"
                           "    x: ir.Scalar[ir.INT32] = 42";
    ASSERT_EQ(PythonPrint(std::static_pointer_cast<const IRNode>(prog)), expected);

    ASSERT_NE(prog->GetFunction("test_func"), nullptr);
}

// ============================================================================
// Structural Hash / Equal Tests
// ============================================================================

class IRStructuralTest : public testing::Test {
protected:
    Span sp = Span("test", 1, 1);
};

TEST_F(IRStructuralTest, TestHashIgnoresSpan)
{
    auto a = std::make_shared<Var>("x", Scalar(DataType::INT32), Span("file_a", 1, 1));
    auto b = std::make_shared<Var>("x", Scalar(DataType::INT32), Span("file_b", 99, 99));
    ASSERT_EQ(structural_hash(std::static_pointer_cast<const IRNode>(a)),
              structural_hash(std::static_pointer_cast<const IRNode>(b)));
}

TEST_F(IRStructuralTest, TestEqualIdenticalNodes)
{
    auto a = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto b = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    ASSERT_TRUE(structural_equal(std::static_pointer_cast<const IRNode>(a), std::static_pointer_cast<const IRNode>(b)));
}

TEST_F(IRStructuralTest, TestEqualDifferentNodes)
{
    auto a = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto b = std::make_shared<ConstInt>(99, DataType::INT32, sp);
    ASSERT_FALSE(
        structural_equal(std::static_pointer_cast<const IRNode>(a), std::static_pointer_cast<const IRNode>(b)));
}

TEST_F(IRStructuralTest, TestEqualIgnoresSpan)
{
    auto a = std::make_shared<ConstInt>(42, DataType::INT32, Span("f1", 1, 1));
    auto b = std::make_shared<ConstInt>(42, DataType::INT32, Span("f2", 99, 99));
    ASSERT_TRUE(structural_equal(std::static_pointer_cast<const IRNode>(a), std::static_pointer_cast<const IRNode>(b)));
}

TEST_F(IRStructuralTest, TestEqualAutoMapping)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), sp);
    auto y = std::make_shared<Var>("y", Scalar(DataType::INT32), sp);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, sp);

    auto exprX = std::make_shared<Add>(x, one, DataType::INT32, sp);
    auto exprY = std::make_shared<Add>(y, one, DataType::INT32, sp);

    ASSERT_FALSE(structural_equal(std::static_pointer_cast<const IRNode>(exprX),
                                  std::static_pointer_cast<const IRNode>(exprY), false));
    ASSERT_TRUE(structural_equal(std::static_pointer_cast<const IRNode>(exprX),
                                 std::static_pointer_cast<const IRNode>(exprY), true));
}

TEST_F(IRStructuralTest, TestHashAutoMapping)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), sp);
    auto y = std::make_shared<Var>("y", Scalar(DataType::INT32), sp);

    ASSERT_NE(structural_hash(std::static_pointer_cast<const IRNode>(x), false),
              structural_hash(std::static_pointer_cast<const IRNode>(y), false));
    ASSERT_EQ(structural_hash(std::static_pointer_cast<const IRNode>(x), true),
              structural_hash(std::static_pointer_cast<const IRNode>(y), true));
}

TEST_F(IRStructuralTest, TestAssertStructuralEqualPass)
{
    auto a = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto b = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    ASSERT_NO_THROW(
        assert_structural_equal(std::static_pointer_cast<const IRNode>(a), std::static_pointer_cast<const IRNode>(b)));
}

TEST_F(IRStructuralTest, TestAssertStructuralEqualFail)
{
    auto a = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto b = std::make_shared<ConstInt>(99, DataType::INT32, sp);
    ASSERT_THROW(
        assert_structural_equal(std::static_pointer_cast<const IRNode>(a), std::static_pointer_cast<const IRNode>(b)),
        npu::tile_fwk::Error);
}

TEST_F(IRStructuralTest, TestHashType)
{
    auto t1 = std::make_shared<ScalarType>(DataType::INT32);
    auto t2 = std::make_shared<ScalarType>(DataType::INT32);
    auto t3 = std::make_shared<ScalarType>(DataType::FP32);

    ASSERT_EQ(structural_hash(t1), structural_hash(t2));
    ASSERT_NE(structural_hash(t1), structural_hash(t3));
}

TEST_F(IRStructuralTest, TestEqualType)
{
    auto t1 = std::make_shared<ScalarType>(DataType::INT32);
    auto t2 = std::make_shared<ScalarType>(DataType::INT32);
    auto t3 = std::make_shared<ScalarType>(DataType::FP32);

    ASSERT_TRUE(structural_equal(t1, t2));
    ASSERT_FALSE(structural_equal(t1, t3));
}

TEST_F(IRStructuralTest, TestAssertStructuralEqualTypePass)
{
    auto t1 = std::make_shared<ScalarType>(DataType::INT32);
    auto t2 = std::make_shared<ScalarType>(DataType::INT32);
    ASSERT_NO_THROW(assert_structural_equal(t1, t2));
}

TEST_F(IRStructuralTest, TestAssertStructuralEqualTypeFail)
{
    auto t1 = std::make_shared<ScalarType>(DataType::INT32);
    auto t2 = std::make_shared<ScalarType>(DataType::FP32);
    ASSERT_THROW(assert_structural_equal(t1, t2), npu::tile_fwk::Error);
}

TEST_F(IRStructuralTest, TestEqualComplexExpressions)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, sp);
    auto add1 = std::make_shared<Add>(a, b, DataType::INT32, sp);
    auto add2 = std::make_shared<Add>(a, b, DataType::INT32, sp);
    auto sub1 = std::make_shared<Sub>(a, b, DataType::INT32, sp);

    ASSERT_TRUE(
        structural_equal(std::static_pointer_cast<const IRNode>(add1), std::static_pointer_cast<const IRNode>(add2)));
    ASSERT_FALSE(
        structural_equal(std::static_pointer_cast<const IRNode>(add1), std::static_pointer_cast<const IRNode>(sub1)));
}

TEST_F(IRStructuralTest, TestEqualNestedStatements)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), sp);
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto assign = std::make_shared<AssignStmt>(x, val, sp);

    auto seq1 = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign}, sp);
    auto seq2 = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign}, sp);

    ASSERT_TRUE(
        structural_equal(std::static_pointer_cast<const IRNode>(seq1), std::static_pointer_cast<const IRNode>(seq2)));
}

// ============================================================================
// FunctionType conversion tests
// ============================================================================

TEST_F(IRCheckTest, TestFunctionTypeToString)
{
    ASSERT_EQ(FunctionTypeToString(FunctionType::OPAQUE), "Opaque");
    ASSERT_EQ(FunctionTypeToString(FunctionType::ORCHESTRATION), "Orchestration");
    ASSERT_EQ(FunctionTypeToString(FunctionType::IN_CORE), "InCore");
    ASSERT_EQ(FunctionTypeToString(FunctionType::HELPER), "Helper");
    ASSERT_EQ(FunctionTypeToString(static_cast<FunctionType>(999)), "Unknown");
}

TEST_F(IRCheckTest, TestStringToFunctionType)
{
    ASSERT_EQ(StringToFunctionType("Opaque"), FunctionType::OPAQUE);
    ASSERT_EQ(StringToFunctionType("Orchestration"), FunctionType::ORCHESTRATION);
    ASSERT_EQ(StringToFunctionType("InCore"), FunctionType::IN_CORE);
    ASSERT_EQ(StringToFunctionType("Helper"), FunctionType::HELPER);
    ASSERT_THROW(StringToFunctionType("Invalid"), std::invalid_argument);
}

TEST_F(IRCheckTest, TestPipeTypeToString)
{
    ASSERT_EQ(PipeTypeToString(PipeType::MTE1), "MTE1");
    ASSERT_EQ(PipeTypeToString(PipeType::MTE2), "MTE2");
    ASSERT_EQ(PipeTypeToString(PipeType::MTE3), "MTE3");
    ASSERT_EQ(PipeTypeToString(PipeType::M), "M");
    ASSERT_EQ(PipeTypeToString(PipeType::V), "V");
    ASSERT_EQ(PipeTypeToString(PipeType::S), "S");
    ASSERT_EQ(PipeTypeToString(PipeType::FIX), "FIX");
    ASSERT_EQ(PipeTypeToString(PipeType::ALL), "ALL");
    ASSERT_THROW(PipeTypeToString(static_cast<PipeType>(999)), TypeError);
}

} // namespace ir
} // namespace pypto
