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
 * \file test_program.cpp
 * \brief Unit tests for IR Program construction and lookup
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

// Helper to create a simple function
static FunctionPtr MakeSimpleFunction(const std::string& name)
{
    auto bodyVal = std::make_shared<ConstInt>(0, DataType::INT32, Span::Unknown());
    auto body = std::make_shared<EvalStmt>(bodyVal, Span::Unknown());
    std::vector<VarPtr> params;
    std::vector<TypePtr> returnTypes;
    return std::make_shared<Function>(name, params, returnTypes, body, Span::Unknown());
}

class IRProgramTest : public testing::Test {};

// ============================================================================
// Program Constructor Tests
// ============================================================================

TEST_F(IRProgramTest, TestProgramBasicConstruction)
{
    auto func = MakeSimpleFunction("main");
    std::vector<FunctionPtr> funcs = {func};
    auto program = std::make_shared<Program>(funcs, "test_program", Span::Unknown());

    ASSERT_EQ(program->name_, "test_program");
    ASSERT_EQ(program->functions_.size(), 1);
}

TEST_F(IRProgramTest, TestProgramMultipleFunctions)
{
    auto func1 = MakeSimpleFunction("func_a");
    auto func2 = MakeSimpleFunction("func_b");
    std::vector<FunctionPtr> funcs = {func1, func2};
    auto program = std::make_shared<Program>(funcs, "multi_func", Span::Unknown());

    ASSERT_EQ(program->functions_.size(), 2);
}

TEST_F(IRProgramTest, TestProgramEmptyFunctions)
{
    std::vector<FunctionPtr> funcs;
    auto program = std::make_shared<Program>(funcs, "empty", Span::Unknown());

    ASSERT_EQ(program->functions_.size(), 0);
}

// ============================================================================
// Program GetFunction Tests
// ============================================================================

TEST_F(IRProgramTest, TestGetFunctionByName)
{
    auto func = MakeSimpleFunction("my_func");
    std::vector<FunctionPtr> funcs = {func};
    auto program = std::make_shared<Program>(funcs, "test", Span::Unknown());

    auto found = program->GetFunction("my_func");
    ASSERT_NE(found, nullptr);
    ASSERT_EQ(found->name_, "my_func");
}

TEST_F(IRProgramTest, TestGetFunctionNotFound)
{
    auto func = MakeSimpleFunction("existing");
    std::vector<FunctionPtr> funcs = {func};
    auto program = std::make_shared<Program>(funcs, "test", Span::Unknown());

    auto found = program->GetFunction("nonexistent");
    ASSERT_EQ(found, nullptr);
}

// ============================================================================
// Program Validation Tests
// ============================================================================

TEST_F(IRProgramTest, TestProgramDuplicateFunctionNameThrows)
{
    auto func1 = MakeSimpleFunction("dup_name");
    auto func2 = MakeSimpleFunction("dup_name");
    std::vector<FunctionPtr> funcs = {func1, func2};

    ASSERT_THROW(std::make_shared<Program>(funcs, "test", Span::Unknown()), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
