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
 * \file test_printer.cpp
 * \brief Coverage tests for IRPrinter (printer.cpp)
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/printer.h"
#include "ir/type.h"
#include "test_ir.h"
#include "tilefwk/symbolic_scalar.h"

using npu::tile_fwk::SymbolicScalar;

namespace pypto {
namespace ir {

static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }
static Span Sp() { return Span("test", 1, 1); }

static VarPtr Var_(const std::string& name, DataType dt = DataType::INT32)
{
    return std::make_shared<Var>(name, Scalar(dt), Sp());
}

class IRPrinterTest : public testing::Test {
protected:
    std::string Print(const ExprPtr& expr) { return PythonPrint(expr); }
    std::string Print(const StmtPtr& stmt) { return PythonPrint(stmt); }
    std::string Print(const FunctionPtr& func) { return PythonPrint(func); }
    std::string Print(const ProgramPtr& prog) { return PythonPrint(prog); }
    std::string PrintType(const TypePtr& type) { return PythonPrint(type); }
};

// ============================================================================
// Expression printing — constants, vars, memref, call, tuple
// ============================================================================

TEST_F(IRPrinterTest, TestPrintConstantsAndBasicExprs)
{
    // ConstInt
    EXPECT_EQ(Print(std::make_shared<ConstInt>(42, DataType::INT32, Sp())), "42");

    // ConstFloat
    EXPECT_FALSE(Print(std::make_shared<ConstFloat>(3.14, DataType::FP32, Sp())).empty());
    EXPECT_EQ(Print(std::make_shared<ConstFloat>(5.0, DataType::FP32, Sp())), "5.0");

    // ConstBool
    EXPECT_EQ(Print(std::make_shared<ConstBool>(true, Sp())), "True");
    EXPECT_EQ(Print(std::make_shared<ConstBool>(false, Sp())), "False");

    // Var
    EXPECT_EQ(Print(Var_("x")), "x");

    // MemRef
    auto off = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    EXPECT_FALSE(Print(std::make_shared<MemRef>(MemorySpace::DDR, off, 1024, Sp())).empty());

    // Call
    auto arg1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto arg2 = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto callResult = Print(std::make_shared<Call>("op", std::vector<ExprPtr>{arg1, arg2}, Sp()));
    EXPECT_FALSE(callResult.empty());
    EXPECT_NE(callResult.find("call @op"), std::string::npos);

    // MakeTuple
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    EXPECT_EQ(Print(std::make_shared<MakeTuple>(std::vector<ExprPtr>{a, b}, Sp())), "[1, 2]");

    // GetItemExpr
    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a}, Sp());
    auto idx = std::make_shared<ConstInt>(0, DataType::INDEX, Sp());
    EXPECT_EQ(Print(std::make_shared<GetItemExpr>(tup, idx, Sp())), "[1][0]");
}

// ============================================================================
// Binary and Unary expression printing — all operators
// ============================================================================

TEST_F(IRPrinterTest, TestPrintBinaryExpr)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
#define DEFINE_BINARY_EXPR(name)                                         \
    {                                                                    \
        auto expr = std::make_shared<name>(a, b, DataType::INT32, Sp()); \
        EXPECT_FALSE(Print(expr).empty());                               \
    }
    DEFINE_BINARY_EXPR_ALL()
#undef DEFINE_BINARY_EXPR
}

TEST_F(IRPrinterTest, TestPrintUnaryExpr)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
#define DEFINE_UNARY_EXPR(name)                                       \
    {                                                                 \
        auto expr = std::make_shared<name>(a, DataType::INT32, Sp()); \
        EXPECT_FALSE(Print(expr).empty());                            \
    }
    DEFINE_UNARY_EXPR_ALL()
#undef DEFINE_UNARY_EXPR
}

// ============================================================================
// Precedence / parenthesis tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrecedenceAddSubChildNeedsParens)
{
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto two = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto three = std::make_shared<ConstInt>(3, DataType::INT32, Sp());
    auto mul = std::make_shared<Mul>(std::make_shared<Add>(one, two, DataType::INT32, Sp()), three, DataType::INT32,
                                     Sp());
    EXPECT_NE(Print(mul).find("("), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrecedencePowRightAssoc)
{
    auto two = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto three = std::make_shared<ConstInt>(3, DataType::INT32, Sp());
    auto four = std::make_shared<ConstInt>(4, DataType::INT32, Sp());
    auto outer = std::make_shared<Pow>(two, std::make_shared<Pow>(three, four, DataType::INT32, Sp()), DataType::INT32,
                                       Sp());
    EXPECT_EQ(Print(outer), "2 ** 3 ** 4");
}

TEST_F(IRPrinterTest, TestPrecedencePowLeftNeedsParens)
{
    auto two = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto three = std::make_shared<ConstInt>(3, DataType::INT32, Sp());
    auto four = std::make_shared<ConstInt>(4, DataType::INT32, Sp());
    auto outer = std::make_shared<Pow>(std::make_shared<Pow>(two, three, DataType::INT32, Sp()), four, DataType::INT32,
                                       Sp());
    EXPECT_NE(Print(outer).find("("), std::string::npos);
}

TEST_F(IRPrinterTest, TestUnaryWithLowerPrecedenceOperand)
{
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto two = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto add = std::make_shared<Add>(one, two, DataType::INT32, Sp());

    EXPECT_NE(Print(std::make_shared<Neg>(add, DataType::INT32, Sp())).find("("), std::string::npos);
    EXPECT_NE(
        Print(std::make_shared<Not>(std::make_shared<And>(one, two, DataType::INT32, Sp()), DataType::INT32, Sp()))
            .find("("),
        std::string::npos);
    EXPECT_NE(Print(std::make_shared<BitNot>(add, DataType::INT32, Sp())).find("("), std::string::npos);
}

TEST_F(IRPrinterTest, TestBitNotSimple)
{
    EXPECT_EQ(
        Print(std::make_shared<BitNot>(std::make_shared<ConstInt>(5, DataType::INT32, Sp()), DataType::INT32, Sp())),
        "~5");
}

TEST_F(IRPrinterTest, TestSamePrecedenceLeftAssocNeedsParens)
{
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto two = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto three = std::make_shared<ConstInt>(3, DataType::INT32, Sp());
    auto outer = std::make_shared<Sub>(one, std::make_shared<Sub>(two, three, DataType::INT32, Sp()), DataType::INT32,
                                       Sp());
    EXPECT_NE(Print(outer).find("("), std::string::npos);
}

// ============================================================================
// Type printing
// ============================================================================

TEST_F(IRPrinterTest, TestPrintTypes)
{
    // ScalarType
    EXPECT_NE(PrintType(Scalar(DataType::INT32)).find("Scalar"), std::string::npos);

    // TensorType
    auto off = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    MemRefPtr memref = std::make_shared<MemRef>(MemorySpace::DDR, off, 128, Sp());
    auto tensor = std::make_shared<TensorType>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, Sp()),
                             std::make_shared<ConstInt>(8, DataType::INT64, Sp())},
        DataType::FP32, memref);
    auto tensorStr = PrintType(tensor);
    EXPECT_NE(tensorStr.find("Tensor"), std::string::npos);
    EXPECT_NE(tensorStr.find("FP32"), std::string::npos);

    // TileType without memref
    auto tile = std::make_shared<TileType>(std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, Sp())},
                                           DataType::FP32);
    EXPECT_NE(PrintType(tile).find("Tile"), std::string::npos);

    // TileType with memref
    auto tileMr = std::make_shared<TileType>(std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, Sp())},
                                             DataType::FP32, std::optional<MemRefPtr>(memref));
    auto tileMrStr = PrintType(tileMr);
    EXPECT_NE(tileMrStr.find("Tile"), std::string::npos);
    EXPECT_NE(tileMrStr.find("MemRef"), std::string::npos);

    // TupleType empty
    EXPECT_NE(PrintType(std::make_shared<TupleType>(std::vector<TypePtr>{})).find("Tuple[()]"), std::string::npos);

    // TupleType non-empty
    EXPECT_NE(
        PrintType(std::make_shared<TupleType>(std::vector<TypePtr>{Scalar(DataType::INT32), Scalar(DataType::FP32)}))
            .find("Tuple"),
        std::string::npos);

    // MemRefType
    EXPECT_NE(PrintType(GetMemRefType()).find("MemRefType"), std::string::npos);

    // PtrType
    EXPECT_NE(PrintType(std::make_shared<PtrType>(DataType::FP32)).find("Ptr"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintTensorTypeWithVarShape)
{
    auto dim = Var_("N", DataType::INT64);
    auto type = std::make_shared<TensorType>(std::vector<ExprPtr>{dim}, DataType::FP32);
    EXPECT_NE(PrintType(type).find("N"), std::string::npos);
}

// ============================================================================
// Statement printing
// ============================================================================

TEST_F(IRPrinterTest, TestPrintBasicStmts)
{
    // AssignStmt
    auto x = Var_("x");
    auto assign = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, Sp()), Sp());
    auto assignStr = Print(assign);
    EXPECT_NE(assignStr.find("x"), std::string::npos);
    EXPECT_NE(assignStr.find("42"), std::string::npos);

    // AssignStmt concise
    EXPECT_NE(PythonPrint(assign, "ir", true).find("x = 42"), std::string::npos);

    // YieldStmt multi-value
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto yieldStr = Print(std::make_shared<YieldStmt>(std::vector<ExprPtr>{a, b}, Sp()));
    EXPECT_NE(yieldStr.find("yield_"), std::string::npos);
    EXPECT_NE(yieldStr.find(", "), std::string::npos);

    // ReturnStmt multi-value
    auto retStr = Print(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{a, b}, Sp()));
    EXPECT_NE(retStr.find("return"), std::string::npos);
    EXPECT_NE(retStr.find(", "), std::string::npos);

    // ReturnStmt empty
    EXPECT_EQ(Print(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, Sp())), "return");

    // SeqStmts
    auto s1 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto s2 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(2, DataType::INT32, Sp()), Sp());
    auto seqStr = Print(std::make_shared<SeqStmts>(std::vector<StmtPtr>{s1, s2}, Sp()));
    EXPECT_NE(seqStr.find("1"), std::string::npos);
    EXPECT_NE(seqStr.find("2"), std::string::npos);

    // EvalStmt
    auto call = std::make_shared<Call>("op", std::vector<ExprPtr>{}, Sp());
    EXPECT_NE(Print(std::make_shared<EvalStmt>(call, Sp())).find("eval"), std::string::npos);

    // BreakStmt / ContinueStmt
    EXPECT_EQ(Print(std::make_shared<BreakStmt>(Sp())), "break");
    EXPECT_EQ(Print(std::make_shared<ContinueStmt>(Sp())), "continue");
}

TEST_F(IRPrinterTest, TestPrintIfStmt)
{
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto x = Var_("x");

    // If-else with return vars
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto retVar = Var_("out");
    auto ifElse = std::make_shared<IfStmt>(cond, yield, yield, std::vector<VarPtr>{retVar}, Sp());
    auto ifElseStr = Print(ifElse);
    EXPECT_NE(ifElseStr.find("if"), std::string::npos);
    EXPECT_NE(ifElseStr.find("else"), std::string::npos);

    // If without else
    auto assign = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto ifNoElse = std::make_shared<IfStmt>(cond, assign, std::nullopt, std::vector<VarPtr>{}, Sp());
    auto ifNoElseStr = Print(ifNoElse);
    EXPECT_NE(ifNoElseStr.find("if"), std::string::npos);
    EXPECT_EQ(ifNoElseStr.find("else"), std::string::npos);

    // Yield without return vars
    auto yieldBody = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    EXPECT_NE(
        Print(std::make_shared<IfStmt>(cond, yieldBody, std::nullopt, std::vector<VarPtr>{}, Sp())).find("yield_"),
        std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintForStmt)
{
    auto i = Var_("i");
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto x = Var_("x");

    // For with multiple iterArgs
    auto init1 = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto init2 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto ia1 = std::make_shared<IterArg>("acc1", Scalar(DataType::INT32), init1, Sp());
    auto ia2 = std::make_shared<IterArg>("acc2", Scalar(DataType::INT32), init2, Sp());
    auto rv1 = Var_("out1");
    auto rv2 = Var_("out2");
    auto body = std::make_shared<YieldStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, Sp()),
                                                                 std::make_shared<ConstInt>(2, DataType::INT32, Sp())},
                                            Sp());
    auto forStr = Print(std::make_shared<ForStmt>(i, start, stop, step, std::vector<IterArgPtr>{ia1, ia2}, body,
                                                  std::vector<VarPtr>{rv1, rv2}, Sp()));
    EXPECT_NE(forStr.find("for"), std::string::npos);
    EXPECT_NE(forStr.find("acc1"), std::string::npos);
    EXPECT_NE(forStr.find("acc2"), std::string::npos);

    // For without iterArgs
    auto simpleBody = std::make_shared<AssignStmt>(Var_("x"), std::make_shared<ConstInt>(1, DataType::INT32, Sp()),
                                                   Sp());
    auto simpleFor = Print(std::make_shared<ForStmt>(i, start, stop, step, std::vector<IterArgPtr>{}, simpleBody,
                                                     std::vector<VarPtr>{}, Sp()));
    EXPECT_NE(simpleFor.find("for"), std::string::npos);
    EXPECT_EQ(simpleFor.find("init_values"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintWhileStmt)
{
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto x = Var_("x");

    // While without iterArgs
    auto simpleBody = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    EXPECT_NE(
        Print(std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, simpleBody, std::vector<VarPtr>{}, Sp()))
            .find("while"),
        std::string::npos);

    // While with single iterArg
    auto init = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ia = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), init, Sp());
    auto rv = Var_("out");
    auto yieldBody = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, Sp())}, Sp());
    auto whileStr = Print(
        std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{ia}, yieldBody, std::vector<VarPtr>{rv}, Sp()));
    EXPECT_NE(whileStr.find("while_"), std::string::npos);
    EXPECT_NE(whileStr.find("acc"), std::string::npos);

    // While with multiple iterArgs
    auto init2 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto ia2 = std::make_shared<IterArg>("b", Scalar(DataType::INT32), init2, Sp());
    auto rv2 = Var_("out2");
    auto multiBody = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, Sp()),
                             std::make_shared<ConstInt>(2, DataType::INT32, Sp())},
        Sp());
    auto multiStr = Print(std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{ia, ia2}, multiBody,
                                                      std::vector<VarPtr>{rv, rv2}, Sp()));
    EXPECT_NE(multiStr.find("while_"), std::string::npos);
    EXPECT_NE(multiStr.find("b"), std::string::npos);
}

// ============================================================================
// Function and Program printing
// ============================================================================

TEST_F(IRPrinterTest, TestPrintFunction)
{
    auto x = Var_("x");
    auto y = Var_("y", DataType::FP32);

    // Basic function
    auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, Sp()), Sp());
    auto funcStr = Print(std::make_shared<Function>("f", std::vector<VarPtr>{x},
                                                    std::vector<TypePtr>{Scalar(DataType::INT32)}, body, Sp()));
    EXPECT_NE(funcStr.find("def f"), std::string::npos);
    EXPECT_NE(funcStr.find("@ir.function"), std::string::npos);

    // Multi params
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto multiParam = Print(
        std::make_shared<Function>("g", std::vector<VarPtr>{x, y}, std::vector<TypePtr>{}, yield, Sp()));
    EXPECT_NE(multiParam.find("x"), std::string::npos);
    EXPECT_NE(multiParam.find("y"), std::string::npos);

    // Multi return types
    auto val = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto multiRet = Print(std::make_shared<Function>(
        "h", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32), Scalar(DataType::FP32)},
        std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, Sp()), Sp()));
    EXPECT_NE(multiRet.find("tuple["), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintFunctionBodies)
{
    auto x = Var_("x");

    // Yield body → return
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto yieldFunc = Print(
        std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                   std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, Sp()), Sp()));
    EXPECT_NE(yieldFunc.find("return 42"), std::string::npos);

    // SeqStmts body with yield at end → return
    auto a1 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto a2 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(2, DataType::INT32, Sp()), Sp());
    auto yieldStmt = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(3, DataType::INT32, Sp())}, Sp());
    auto seqFunc = Print(
        std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                   std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, a2, yieldStmt}, Sp()), Sp()));
    EXPECT_NE(seqFunc.find("return 3"), std::string::npos);

    // Empty SeqStmts body → pass
    auto emptyFunc = Print(std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{},
                                                      std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, Sp()), Sp()));
    EXPECT_NE(emptyFunc.find("pass"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintProgram)
{
    auto x = Var_("x");
    auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto f1 = std::make_shared<Function>("f1", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body, Sp());
    auto f2 = std::make_shared<Function>("f2", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body, Sp());
    auto progStr = Print(std::make_shared<Program>(std::vector<FunctionPtr>{f1, f2}, "prog", Sp()));
    EXPECT_NE(progStr.find("ir.program"), std::string::npos);
    EXPECT_NE(progStr.find("f1"), std::string::npos);
    EXPECT_NE(progStr.find("f2"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintSimtFunctionDecorators)
{
    auto body = std::make_shared<ReturnStmt>(Sp());
    auto simt_function = std::make_shared<Function>(
        "entry", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp(), FunctionType::SIMT_VF, false,
        std::vector<std::pair<std::string, std::any>>{{kMaxThreadsAttr, 256}});
    auto unbounded_simt_function = std::make_shared<Function>(
        "unbounded", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp(), FunctionType::SIMT_VF);
    auto simt_callee = std::make_shared<Function>("helper", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp(),
                                                  FunctionType::SIMT_CALLEE);

    auto function_str = Print(simt_function);
    auto unbounded_function_str = Print(unbounded_simt_function);
    auto callee_str = Print(simt_callee);

    EXPECT_NE(function_str.find("@ir.simt.function"), std::string::npos);
    EXPECT_NE(function_str.find("ir.FunctionType.SimtVF"), std::string::npos);
    EXPECT_EQ(function_str.find("max_threads"), std::string::npos);
    EXPECT_NE(function_str.find("def entry()"), std::string::npos);
    EXPECT_NE(unbounded_function_str.find("@ir.simt.function"), std::string::npos);
    EXPECT_NE(unbounded_function_str.find("ir.FunctionType.SimtVF"), std::string::npos);
    EXPECT_EQ(unbounded_function_str.find("max_threads"), std::string::npos);
    EXPECT_NE(callee_str.find("@ir.simt.function"), std::string::npos);
    EXPECT_NE(callee_str.find("ir.FunctionType.SimtCallee"), std::string::npos);
    EXPECT_NE(callee_str.find("def helper()"), std::string::npos);
}

// ============================================================================
// Control flow with SeqStmts body
// ============================================================================

TEST_F(IRPrinterTest, TestPrintIfWithSeqStmtsBody)
{
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto x = Var_("x");
    auto retVar = Var_("out");
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(2, DataType::INT32, Sp())},
                                             Sp());

    // If with SeqStmts body (no else)
    auto a1 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto thenBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, yield}, Sp());
    auto ifStr = Print(std::make_shared<IfStmt>(cond, thenBody, std::nullopt, std::vector<VarPtr>{retVar}, Sp()));
    EXPECT_NE(ifStr.find("if"), std::string::npos);
    EXPECT_NE(ifStr.find("yield_"), std::string::npos);

    // If with SeqStmts then + else
    auto a2 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(3, DataType::INT32, Sp()), Sp());
    auto yield2 = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT32, Sp())}, Sp());
    auto elseBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{a2, yield2}, Sp());
    EXPECT_NE(Print(std::make_shared<IfStmt>(cond, thenBody, elseBody, std::vector<VarPtr>{retVar}, Sp())).find("else"),
              std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintForWithSeqStmtsBody)
{
    auto i = Var_("i");
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto init = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ia = std::make_shared<IterArg>("sum", Scalar(DataType::INT32), init, Sp());
    auto rv = Var_("sum_out");
    auto x = Var_("x");
    auto a1 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(2, DataType::INT32, Sp())},
                                             Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, yield}, Sp());
    auto forStr = Print(std::make_shared<ForStmt>(i, start, stop, step, std::vector<IterArgPtr>{ia}, body,
                                                  std::vector<VarPtr>{rv}, Sp()));
    EXPECT_NE(forStr.find("for"), std::string::npos);
    EXPECT_NE(forStr.find("yield_"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintWhileWithSeqStmtsBody)
{
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto x = Var_("x");
    auto a1 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(2, DataType::INT32, Sp())},
                                             Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, yield}, Sp());

    // Without iterArgs
    EXPECT_NE(Print(std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, body, std::vector<VarPtr>{}, Sp()))
                  .find("while"),
              std::string::npos);

    // With iterArgs
    auto init = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ia = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), init, Sp());
    auto rv = Var_("out");
    auto whileStr = Print(
        std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{ia}, body, std::vector<VarPtr>{rv}, Sp()));
    EXPECT_NE(whileStr.find("while_"), std::string::npos);
}

// ============================================================================
// Nested SeqStmts and multi return vars
// ============================================================================

TEST_F(IRPrinterTest, TestPrintNestedSeqStmtsAndMultiReturnVars)
{
    auto x = Var_("x");
    auto a1 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto a2 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(2, DataType::INT32, Sp()), Sp());
    auto a3 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(3, DataType::INT32, Sp()), Sp());

    // Nested SeqStmts
    auto inner = std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, a2}, Sp());
    auto outer = std::make_shared<SeqStmts>(std::vector<StmtPtr>{inner, a3}, Sp());
    auto nestedStr = Print(outer);
    EXPECT_NE(nestedStr.find("1"), std::string::npos);
    EXPECT_NE(nestedStr.find("2"), std::string::npos);
    EXPECT_NE(nestedStr.find("3"), std::string::npos);

    // If with multi return vars
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto yieldBody = std::make_shared<YieldStmt>(std::vector<ExprPtr>{a, b}, Sp());
    auto elseBody = std::make_shared<YieldStmt>(std::vector<ExprPtr>{a, b}, Sp());
    auto rv1 = Var_("out1");
    auto rv2 = Var_("out2");
    auto multiStr = Print(std::make_shared<IfStmt>(cond, yieldBody, elseBody, std::vector<VarPtr>{rv1, rv2}, Sp()));
    EXPECT_NE(multiStr.find("out1"), std::string::npos);
    EXPECT_NE(multiStr.find("out2"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintIfWithEmptySeqStmtsBody)
{
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto emptyBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, Sp());
    EXPECT_NE(Print(std::make_shared<IfStmt>(cond, emptyBody, std::nullopt, std::vector<VarPtr>{}, Sp())).find("pass"),
              std::string::npos);
}

// ============================================================================
// ScalarOpStmt and TensorOpStmt printing
// ============================================================================

TEST_F(IRPrinterTest, TestPrintScalarOpStmt)
{
    auto res = Var_("res");
    auto tok = Var_("tok");
    auto arg = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto stmt = std::make_shared<ScalarOpStmt>(res, tok, "add", std::vector<ExprPtr>{arg}, Sp());
    auto str = Print(stmt);
    EXPECT_EQ(str, "res, tok = add(1)");
}

TEST_F(IRPrinterTest, TestPrintTensorOpStmt)
{
    auto res = Var_("res");
    auto tok = Var_("tok");
    auto arg = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto stmt = std::make_shared<TensorOpStmt>(std::vector<VarPtr>{res}, tok, "matmul", std::vector<ExprPtr>{arg},
                                               std::vector<VarPtr>{}, std::vector<std::pair<std::string, std::any>>{},
                                               Sp());
    auto str = Print(stmt);
    EXPECT_EQ(str, "res, tok = matmul(1)");
}

TEST_F(IRPrinterTest, TestPrintTensorOpStmtMultiResult)
{
    auto r1 = Var_("r1");
    auto r2 = Var_("r2");
    auto tok = Var_("tok");
    auto arg = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto stmt = std::make_shared<TensorOpStmt>(std::vector<VarPtr>{r1, r2}, tok, "split", std::vector<ExprPtr>{arg},
                                               std::vector<VarPtr>{}, std::vector<std::pair<std::string, std::any>>{},
                                               Sp());
    auto str = Print(stmt);
    EXPECT_EQ(str, "[r1, r2], tok = split(1)");
}

TEST_F(IRPrinterTest, TestPrintAttributes)
{
    auto res = Var_("res");
    auto tok = Var_("tok");
    auto arg = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto t1 = Var_("t1");
    auto x = SymbolicScalar("x");

    std::vector<std::pair<std::string, std::any>> attrs = {
        {"axis", 0},
        {"set_pipe", 1},
        {"wait_pipe", 1},
        {"uint64_val", 1UL},
        {"double_val", 1.0},
        {"float_val", 1.0f},
        {"bool_val", true},
        {"str_val", "hello"},
        {"datatype", DataType::INT32},
        {"memoryspace", MemorySpace::DDR},
        {"SymbolicScalar", x},
        {"vector_int", std::vector<int>{1, 2, 3}},
        {"vector_symbolic_scalar", std::vector<SymbolicScalar>{x, x, x}},
        {"unknown", t1}};
    auto stmt = std::make_shared<TensorOpStmt>(std::vector<VarPtr>{res}, tok, "op", std::vector<ExprPtr>{arg},
                                               std::vector<VarPtr>{t1}, attrs, Sp());
    auto str = Print(stmt);
    EXPECT_EQ(
        str,
        "res, tok = op(1, tokens=[t1], attrs=[\"axis\": 0, \"set_pipe\": ir.PipeType.MTE2, \"wait_pipe\": "
        "ir.PipeType.MTE2, \"uint64_val\": 1, \"double_val\": 1.0, \"float_val\": 1.0, \"bool_val\": True, "
        "\"str_val\": Unsupported, \"datatype\": ir.INT32, \"memoryspace\": ir.MemorySpace.DDR, \"SymbolicScalar\": "
        "x, \"vector_int\": [1, 2, 3], \"vector_symbolic_scalar\": [x, x, x], \"unknown\": Unsupported])");
}

} // namespace ir
} // namespace pypto
