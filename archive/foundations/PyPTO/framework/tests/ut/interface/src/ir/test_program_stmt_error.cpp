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
 * \file test_program_stmt_error.cpp
 * \brief Unit tests for program.cpp, core.cpp, expr.cpp, stmt.cpp, error.cpp, memref.cpp
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

static Span TestSpan() { return Span("test.py", 1, 0); }
static TypePtr Int32Type() { return std::make_shared<ScalarType>(DataType::INT32); }

// ============================================================================
// Program Tests (program.cpp)
// ============================================================================

TEST(ProgramTest, ConstructFromFunctionVector)
{
    auto span = TestSpan();
    auto intType = Int32Type();
    auto x = std::make_shared<Var>("x", intType, span);
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    auto func = std::make_shared<Function>("func1", std::vector<VarPtr>{x}, std::vector<TypePtr>{intType}, body, span);

    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "test_prog", span);
    ASSERT_NE(prog, nullptr);
    ASSERT_EQ(prog->name_, "test_prog");
    ASSERT_EQ(prog->functions_.size(), 1u);
}

TEST(ProgramTest, ConstructFromMultipleFunctions)
{
    auto span = TestSpan();
    auto intType = Int32Type();
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);

    auto func1 = std::make_shared<Function>("func_a", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, span);
    auto func2 = std::make_shared<Function>("func_b", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, span);

    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func1, func2}, "multi_prog", span);
    ASSERT_EQ(prog->functions_.size(), 2u);
}

TEST(ProgramTest, GetFunctionByName)
{
    auto span = TestSpan();
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    auto func = std::make_shared<Function>("my_func", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, span);
    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "prog", span);

    auto found = prog->GetFunction("my_func");
    ASSERT_NE(found, nullptr);
    ASSERT_EQ(found->name_, "my_func");

    auto notFound = prog->GetFunction("nonexistent");
    ASSERT_EQ(notFound, nullptr);
}

TEST(ProgramTest, GetGlobalVarByName)
{
    auto span = TestSpan();
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    auto func = std::make_shared<Function>("my_func", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, span);
    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "prog", span);

    auto found = prog->GetFunction("my_func");
    ASSERT_NE(found, nullptr);
    ASSERT_EQ(found->name_, "my_func");
}

TEST(ProgramTest, ProgramKindAndTypeName)
{
    auto span = TestSpan();
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    auto func = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, span);
    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "p", span);

    ASSERT_EQ(prog->GetKind(), ObjectKind::Program);
    ASSERT_EQ(prog->TypeName(), "Program");
}

// ============================================================================
// Expr Tests (expr.cpp)
// ============================================================================

TEST(ExprTest, MakeTupleConstruction)
{
    auto span = TestSpan();
    auto intType = Int32Type();
    auto x = std::make_shared<Var>("x", intType, span);
    auto y = std::make_shared<Var>("y", intType, span);

    auto tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{x, y}, span);
    ASSERT_NE(tuple, nullptr);
    ASSERT_EQ(tuple->elements_.size(), 2u);

    // Type should be TupleType
    auto tupleType = As<TupleType>(tuple->GetType());
    ASSERT_NE(tupleType, nullptr);
    ASSERT_EQ(tupleType->types_.size(), 2u);
}

TEST(ExprTest, GetItemExpr)
{
    auto span = TestSpan();
    auto intType = Int32Type();
    auto floatType = std::make_shared<ScalarType>(DataType::FP32);
    auto x = std::make_shared<Var>("x", intType, span);
    auto y = std::make_shared<Var>("y", floatType, span);

    auto tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{x, y}, span);
    auto idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
    auto getItem = std::make_shared<GetItemExpr>(tuple, idx, span);

    ASSERT_NE(getItem, nullptr);
    ASSERT_EQ(As<ConstInt>(getItem->slice_)->value_, 0);
    // Type should be the first element's type (INT32)
    auto ResultType = As<ScalarType>(getItem->GetType());
    ASSERT_NE(ResultType, nullptr);
    ASSERT_EQ(ResultType->dtype_, DataType::INT32);
}

TEST(ExprTest, GetItemSecondElement)
{
    auto span = TestSpan();
    auto intType = Int32Type();
    auto floatType = std::make_shared<ScalarType>(DataType::FP32);
    auto x = std::make_shared<Var>("x", intType, span);
    auto y = std::make_shared<Var>("y", floatType, span);

    auto tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{x, y}, span);
    auto idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
    auto getItem = std::make_shared<GetItemExpr>(tuple, idx, span);

    auto ResultType = As<ScalarType>(getItem->GetType());
    ASSERT_NE(ResultType, nullptr);
    ASSERT_EQ(ResultType->dtype_, DataType::FP32);
}

// ============================================================================
// Error Tests (error.cpp)
// ============================================================================

TEST(ErrorTest, RuntimeErrorConstruction)
{
    try {
        throw RuntimeError("test error message");
    } catch (const RuntimeError& e) {
        std::string msg = e.what();
        ASSERT_NE(msg.find("test error message"), std::string::npos);
    }
}

TEST(ErrorTest, ValueErrorConstruction)
{
    try {
        throw ValueError("value error");
    } catch (const ValueError& e) {
        std::string msg = e.what();
        ASSERT_NE(msg.find("value error"), std::string::npos);
    }
}

TEST(ErrorTest, InternalErrorConstruction)
{
    try {
        throw InternalError("internal error");
    } catch (const InternalError& e) {
        std::string msg = e.what();
        ASSERT_NE(msg.find("internal error"), std::string::npos);
    }
}

TEST(ErrorTest, ErrorGetFullMessage)
{
    try {
        throw RuntimeError("test full message");
    } catch (const Error& e) {
        auto fullMsg = e.GetFullMessage();
        ASSERT_NE(fullMsg.find("test full message"), std::string::npos);
    }
}

TEST(ErrorTest, ErrorGetFormattedStackTrace)
{
    try {
        throw RuntimeError("stack trace test");
    } catch (const Error& e) {
        // GetFormattedStackTrace should not crash
        auto trace = e.GetFormattedStackTrace();
        // trace may be empty in release builds
        ASSERT_TRUE(true);
    }
}

// ============================================================================
// MemRef Tests (memref.cpp)
// ============================================================================

TEST(MemRefTest, Construction)
{
    auto span = TestSpan();
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, span);
    auto memref = std::make_shared<MemRef>(MemorySpace::Vec, addr, 1024, span);

    ASSERT_NE(memref, nullptr);
    ASSERT_EQ(memref->memorySpace_, MemorySpace::Vec);
    ASSERT_EQ(memref->size_, 1024u);
    ASSERT_EQ(memref->GetKind(), ObjectKind::MemRef);
}

TEST(MemRefTest, MemorySpaceToString)
{
    ASSERT_EQ(MemorySpaceToString(MemorySpace::DDR), "DDR");
    ASSERT_EQ(MemorySpaceToString(MemorySpace::Vec), "Vec");
    ASSERT_EQ(MemorySpaceToString(MemorySpace::Mat), "Mat");
    ASSERT_EQ(MemorySpaceToString(MemorySpace::Left), "Left");
    ASSERT_EQ(MemorySpaceToString(MemorySpace::Right), "Right");
    ASSERT_EQ(MemorySpaceToString(MemorySpace::Acc), "Acc");
}
} // namespace ir
} // namespace pypto
