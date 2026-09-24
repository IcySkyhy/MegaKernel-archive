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
 * \file test_verifier.cpp
 * \brief Coverage tests for ir/verifier: IRVerifier, TypeCheck, SSA, NoNestedCall passes
 */

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/builder.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/verifier/verification_error.h"
#include "ir/stmt.h"
#include "ir/type.h"
#include "ir/verifier/verification_error.h"
#include "ir/verifier/verifier.h"

namespace pypto {
namespace ir {

static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }
static TypePtr Tensor2D(DataType dt)
{
    return std::make_shared<TensorType>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(int64_t(16), DataType::INT64, Span::Unknown()),
                             std::make_shared<ConstInt>(int64_t(32), DataType::INT64, Span::Unknown())},
        dt);
}
static Span Sp() { return Span("test_verifier", 1, 1); }

// Helper to build a minimal program with one function containing given body
static ProgramPtr MakeProgram(const std::string& func_name, const StmtPtr& body)
{
    auto func = std::make_shared<Function>(func_name, std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp());
    auto program = std::make_shared<Program>(std::map<std::string, FunctionPtr>{}, "test_prog", Sp());
    program->functions_[func_name] = func;
    return program;
}

static std::vector<Diagnostic> VerifyWithTypeCheck(const StmtPtr& body)
{
    IRVerifier v;
    v.AddRule(CreateTypeCheckPropertyVerifier());
    return v.Verify(MakeProgram("f", body));
}

static std::vector<Diagnostic> VerifyWithNoNestedCall(const StmtPtr& body)
{
    IRVerifier v;
    v.AddRule(CreateNoNestedCallPropertyVerifier());
    return v.Verify(MakeProgram("f", body));
}

static bool HasErrorCode(const std::vector<Diagnostic>& diags, int code)
{
    for (const auto& diag : diags) {
        if (diag.errorCode == code)
            return true;
    }
    return false;
}

// ============================================================================
// verifier.cpp: IRVerifier basic API
// ============================================================================

class VerifierApiTest : public testing::Test {};

TEST_F(VerifierApiTest, Verify_NullProgram_ReturnsEmpty)
{
    IRVerifier v;
    v.AddRule(CreateTypeCheckPropertyVerifier());
    auto diags = v.Verify(nullptr);
    EXPECT_TRUE(diags.empty());
}

TEST_F(VerifierApiTest, Verify_EmptyProgram_ReturnsEmpty)
{
    IRVerifier v;
    v.AddRule(CreateTypeCheckPropertyVerifier());
    v.AddRule(CreateSSAPropertyVerifier());
    v.AddRule(CreateNoNestedCallPropertyVerifier());
    auto program = std::make_shared<Program>(std::map<std::string, FunctionPtr>{}, "empty", Sp());
    auto diags = v.Verify(program);
    EXPECT_TRUE(diags.empty());
}

TEST_F(VerifierApiTest, AddRule_Duplicate_IgnoresSecond)
{
    IRVerifier v;
    v.AddRule(CreateTypeCheckPropertyVerifier());
    v.AddRule(CreateTypeCheckPropertyVerifier());
    auto program = std::make_shared<Program>(std::map<std::string, FunctionPtr>{}, "empty", Sp());
    auto diags = v.Verify(program);
    EXPECT_TRUE(diags.empty());
}

TEST_F(VerifierApiTest, AddRule_Null_DoesNotCrash)
{
    IRVerifier v;
    v.AddRule(nullptr);
    auto program = std::make_shared<Program>(std::map<std::string, FunctionPtr>{}, "empty", Sp());
    auto diags = v.Verify(program);
    EXPECT_TRUE(diags.empty());
}

TEST_F(VerifierApiTest, DisableRule_SkipsVerification)
{
    IRVerifier v;
    v.AddRule(CreateSSAPropertyVerifier());
    v.DisableRule("SSAVerify");
    EXPECT_FALSE(v.IsRuleEnabled("SSAVerify"));

    // Build a program with SSA violation — should not be reported since rule is disabled
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto val = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());
    auto a1 = std::make_shared<AssignStmt>(x, val, Sp());
    auto a2 = std::make_shared<AssignStmt>(x, val, Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, a2}, Sp());
    auto program = MakeProgram("f", body);
    auto diags = v.Verify(program);
    EXPECT_TRUE(diags.empty());
}

TEST_F(VerifierApiTest, EnableRule_ReEnables)
{
    IRVerifier v;
    v.AddRule(CreateSSAPropertyVerifier());
    v.DisableRule("SSAVerify");
    v.EnableRule("SSAVerify");
    EXPECT_TRUE(v.IsRuleEnabled("SSAVerify"));
}

TEST_F(VerifierApiTest, VerifyOrThrow_ThrowsOnError)
{
    IRVerifier v;
    v.AddRule(CreateSSAPropertyVerifier());

    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto val = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());
    auto a1 = std::make_shared<AssignStmt>(x, val, Sp());
    auto a2 = std::make_shared<AssignStmt>(x, val, Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, a2}, Sp());
    auto program = MakeProgram("f", body);
    EXPECT_THROW(v.VerifyOrThrow(program), VerificationError);
}

TEST_F(VerifierApiTest, VerifyOrThrow_NoThrowOnClean)
{
    IRVerifier v;
    v.AddRule(CreateSSAPropertyVerifier());
    v.AddRule(CreateTypeCheckPropertyVerifier());

    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto val = std::make_shared<ConstInt>(int64_t(42), DataType::INT32, Sp());
    auto body = std::make_shared<AssignStmt>(x, val, Sp());
    auto program = MakeProgram("f", body);
    EXPECT_NO_THROW(v.VerifyOrThrow(program));
}

TEST_F(VerifierApiTest, GenerateReport_EmptyDiagnostics_ShowsPassed)
{
    auto report = IRVerifier::GenerateReport({});
    EXPECT_NE(report.find("PASSED"), std::string::npos);
}

// ============================================================================
// verify_ssa_pass.cpp: SSA violations
// ============================================================================

class VerifierSSATest : public testing::Test {};

TEST_F(VerifierSSATest, MultipleAssignment_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateSSAPropertyVerifier());

    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto val = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());
    auto a1 = std::make_shared<AssignStmt>(x, val, Sp());
    auto a2 = std::make_shared<AssignStmt>(x, val, Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, a2}, Sp());
    auto program = MakeProgram("f", body);
    auto diags = v.Verify(program);
    EXPECT_GE(diags.size(), 1u);
}

TEST_F(VerifierSSATest, TensorOpResultTokens)
{
    IRVerifier v;
    v.AddRule(CreateSSAPropertyVerifier());

    auto write_token = std::make_shared<Var>("write_token", GetWriteTokenType(), Sp());
    auto read_token = std::make_shared<Var>("read_token", GetReadTokenType(), Sp());
    auto op = std::make_shared<TensorOpStmt>(std::vector<VarPtr>{}, std::vector<VarPtr>{write_token, read_token}, "EXP",
                                             std::vector<ExprPtr>{}, std::vector<VarPtr>{},
                                             decltype(TensorOpStmt::attrs_){}, Sp());

    EXPECT_TRUE(v.Verify(MakeProgram("f", op)).empty());
}

TEST_F(VerifierSSATest, ForStmt_MissingYield_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateSSAPropertyVerifier());

    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(int64_t(0), DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(int64_t(10), DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());
    auto iter_arg = std::make_shared<IterArg>("sum", Scalar(DataType::INT32), zero, Sp());
    auto ret_var = std::make_shared<Var>("sum_out", Scalar(DataType::INT32), Sp());

    // Body without YieldStmt
    auto noop = std::make_shared<AssignStmt>(std::make_shared<Var>("tmp", Scalar(DataType::INT32), Sp()), zero, Sp());

    auto for_stmt = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{iter_arg}, noop,
                                              std::vector<VarPtr>{ret_var}, Sp());
    auto program = MakeProgram("f", for_stmt);
    auto diags = v.Verify(program);
    EXPECT_GE(diags.size(), 1u);
}

TEST_F(VerifierSSATest, IfStmt_MissingElseYield_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateSSAPropertyVerifier());

    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto ret_var = std::make_shared<Var>("result", Scalar(DataType::INT32), Sp());
    auto val = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());

    auto then_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, Sp());
    // else body without yield
    auto else_noop = std::make_shared<AssignStmt>(std::make_shared<Var>("tmp", Scalar(DataType::INT32), Sp()), val,
                                                  Sp());

    auto if_stmt = std::make_shared<IfStmt>(cond, then_yield, else_noop, std::vector<VarPtr>{ret_var}, Sp());
    auto program = MakeProgram("f", if_stmt);
    auto diags = v.Verify(program);
    EXPECT_GE(diags.size(), 1u);
}

TEST_F(VerifierSSATest, WhileStmt_MissingYield_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateSSAPropertyVerifier());

    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto zero = std::make_shared<ConstInt>(int64_t(0), DataType::INT32, Sp());
    auto iter_arg = std::make_shared<IterArg>("cnt", Scalar(DataType::INT32), zero, Sp());
    auto ret_var = std::make_shared<Var>("cnt_out", Scalar(DataType::INT32), Sp());

    auto noop = std::make_shared<AssignStmt>(std::make_shared<Var>("tmp", Scalar(DataType::INT32), Sp()), zero, Sp());

    auto while_stmt = std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{iter_arg}, noop,
                                                  std::vector<VarPtr>{ret_var}, Sp());
    auto program = MakeProgram("f", while_stmt);
    auto diags = v.Verify(program);
    EXPECT_GE(diags.size(), 1u);
}

// ============================================================================
// type_check_pass.cpp: Type checking errors
// ============================================================================

class VerifierTypeCheckTest : public testing::Test {};

TEST_F(VerifierTypeCheckTest, ForStmt_NonScalarStart_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateTypeCheckPropertyVerifier());

    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto tensor_val = std::make_shared<Var>("t", Tensor2D(DataType::FP32), Sp());
    auto ten = std::make_shared<ConstInt>(int64_t(10), DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());

    // start is a tensor — should be scalar
    auto for_stmt = std::make_shared<ForStmt>(i, tensor_val, ten, one, std::vector<IterArgPtr>{}, yield,
                                              std::vector<VarPtr>{}, Sp());
    auto program = MakeProgram("f", for_stmt);
    auto diags = v.Verify(program);
    EXPECT_GE(diags.size(), 1u);
}

TEST_F(VerifierTypeCheckTest, IfStmt_NonScalarCondition_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateTypeCheckPropertyVerifier());

    auto tensor_cond = std::make_shared<Var>("c", Tensor2D(DataType::FP32), Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());

    auto if_stmt = std::make_shared<IfStmt>(tensor_cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp());
    auto program = MakeProgram("f", if_stmt);
    auto diags = v.Verify(program);
    EXPECT_GE(diags.size(), 1u);
}

TEST_F(VerifierTypeCheckTest, WhileStmt_NonScalarCondition_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateTypeCheckPropertyVerifier());

    auto tensor_cond = std::make_shared<Var>("c", Tensor2D(DataType::FP32), Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());

    auto while_stmt = std::make_shared<WhileStmt>(tensor_cond, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{},
                                                  Sp());
    auto program = MakeProgram("f", while_stmt);
    auto diags = v.Verify(program);
    EXPECT_GE(diags.size(), 1u);
}

TEST_F(VerifierTypeCheckTest, ForStmt_SizeMismatch_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateTypeCheckPropertyVerifier());

    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(int64_t(0), DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(int64_t(10), DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());
    auto iter_arg = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), zero, Sp());
    auto ret_var = std::make_shared<Var>("acc_out", Scalar(DataType::INT32), Sp());

    // Yield with 0 values but iter_args has 1
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());

    auto for_stmt = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{iter_arg}, yield,
                                              std::vector<VarPtr>{ret_var}, Sp());
    auto program = MakeProgram("f", for_stmt);
    auto diags = v.Verify(program);
    EXPECT_GE(diags.size(), 1u);
}

TEST_F(VerifierTypeCheckTest, ErrorTypeToString_ReturnsCorrectString)
{
    using namespace typecheck;
    EXPECT_EQ(ErrorTypeToString(ErrorType::TYPE_KIND_MISMATCH), "TYPE_KIND_MISMATCH");
    EXPECT_EQ(ErrorTypeToString(ErrorType::DTYPE_MISMATCH), "DTYPE_MISMATCH");
    EXPECT_EQ(ErrorTypeToString(ErrorType::SHAPE_DIMENSION_MISMATCH), "SHAPE_DIMENSION_MISMATCH");
    EXPECT_EQ(ErrorTypeToString(ErrorType::SHAPE_VALUE_MISMATCH), "SHAPE_VALUE_MISMATCH");
    EXPECT_EQ(ErrorTypeToString(ErrorType::SIZE_MISMATCH), "SIZE_MISMATCH");
    EXPECT_EQ(ErrorTypeToString(ErrorType::IF_CONDITION_MUST_BE_SCALAR), "IF_CONDITION_MUST_BE_SCALAR");
    EXPECT_EQ(ErrorTypeToString(ErrorType::FOR_RANGE_MUST_BE_SCALAR), "FOR_RANGE_MUST_BE_SCALAR");
}

TEST_F(VerifierTypeCheckTest, ScalarType_DtypeMismatch_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateTypeCheckPropertyVerifier());

    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(int64_t(0), DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(int64_t(10), DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());

    // iter_arg is INT32, but yield is FP32 - both ScalarType but different dtype
    auto iter_arg = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), zero, Sp());
    auto ret_var = std::make_shared<Var>("acc_out", Scalar(DataType::INT32), Sp());
    auto float_val = std::make_shared<ConstFloat>(1.0, DataType::FP32, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{float_val}, Sp());

    auto for_stmt = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{iter_arg}, yield,
                                              std::vector<VarPtr>{ret_var}, Sp());
    auto program = MakeProgram("f", for_stmt);
    auto diags = v.Verify(program);

    // Should report DTYPE_MISMATCH error
    EXPECT_GE(diags.size(), 1u);
    bool found_dtype_error = false;
    for (const auto& diag : diags) {
        if (diag.errorCode == static_cast<int>(typecheck::ErrorType::DTYPE_MISMATCH)) {
            found_dtype_error = true;
            break;
        }
    }
    EXPECT_TRUE(found_dtype_error);
}

TEST_F(VerifierTypeCheckTest, IfStmt_YieldTypeMismatch_ReportsError)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto ret_var = std::make_shared<Var>("result", Scalar(DataType::INT32), Sp());

    auto then_yield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(int64_t(42), DataType::INT32, Sp())}, Sp());
    auto else_yield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstFloat>(3.14, DataType::FP32, Sp())}, Sp());

    auto if_stmt = std::make_shared<IfStmt>(cond, then_yield, else_yield, std::vector<VarPtr>{ret_var}, Sp());
    auto diags = VerifyWithTypeCheck(if_stmt);
    EXPECT_TRUE(HasErrorCode(diags, static_cast<int>(typecheck::ErrorType::DTYPE_MISMATCH)));
}

TEST_F(VerifierTypeCheckTest, IfStmt_YieldSizeMismatch_ReportsError)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto ret_var1 = std::make_shared<Var>("result1", Scalar(DataType::INT32), Sp());
    auto ret_var2 = std::make_shared<Var>("result2", Scalar(DataType::INT32), Sp());

    auto then_yield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp()),
                             std::make_shared<ConstInt>(int64_t(2), DataType::INT32, Sp())},
        Sp());
    auto else_yield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(int64_t(3), DataType::INT32, Sp())}, Sp());

    auto if_stmt = std::make_shared<IfStmt>(cond, then_yield, else_yield, std::vector<VarPtr>{ret_var1, ret_var2},
                                            Sp());
    auto diags = VerifyWithTypeCheck(if_stmt);
    EXPECT_TRUE(HasErrorCode(diags, static_cast<int>(typecheck::ErrorType::SIZE_MISMATCH)));
}

TEST_F(VerifierTypeCheckTest, TensorType_ShapeValueMismatch_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateTypeCheckPropertyVerifier());

    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(int64_t(0), DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(int64_t(10), DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());

    // iter_arg is TensorType with shape [16, 32]
    auto dim16 = std::make_shared<ConstInt>(int64_t(16), DataType::INDEX, Sp());
    auto dim32 = std::make_shared<ConstInt>(int64_t(32), DataType::INDEX, Sp());
    auto tensor_type1 = std::make_shared<TensorType>(std::vector<ExprPtr>{dim16, dim32}, DataType::FP32);
    auto iter_arg = std::make_shared<IterArg>("tensor", tensor_type1,
                                              std::make_shared<ConstFloat>(0.0, DataType::FP32, Sp()), Sp());

    // yield is TensorType with shape [16, 64] - different second dimension
    auto dim64 = std::make_shared<ConstInt>(int64_t(64), DataType::INDEX, Sp());
    auto tensor_type2 = std::make_shared<TensorType>(std::vector<ExprPtr>{dim16, dim64}, DataType::FP32);
    auto yield_val = std::make_shared<Var>("yield_tensor", tensor_type2, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{yield_val}, Sp());

    auto ret_var = std::make_shared<Var>("result", tensor_type1, Sp());

    auto for_stmt = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{iter_arg}, yield,
                                              std::vector<VarPtr>{ret_var}, Sp());
    auto program = MakeProgram("f", for_stmt);
    auto diags = v.Verify(program);

    // Should report SHAPE_VALUE_MISMATCH error
    EXPECT_GE(diags.size(), 1u);
    bool found_shape_error = false;
    for (const auto& diag : diags) {
        if (diag.errorCode == static_cast<int>(typecheck::ErrorType::SHAPE_VALUE_MISMATCH)) {
            found_shape_error = true;
            break;
        }
    }
    EXPECT_TRUE(found_shape_error);
}

TEST_F(VerifierTypeCheckTest, ForStmt_NonScalarRange_ReportsError)
{
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(int64_t(0), DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(int64_t(10), DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());
    auto tensor_val = std::make_shared<Var>("t", Tensor2D(DataType::FP32), Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());

    EXPECT_TRUE(
        HasErrorCode(VerifyWithTypeCheck(std::make_shared<ForStmt>(i, zero, tensor_val, one, std::vector<IterArgPtr>{},
                                                                   yield, std::vector<VarPtr>{}, Sp())),
                     static_cast<int>(typecheck::ErrorType::FOR_RANGE_MUST_BE_SCALAR)));

    EXPECT_TRUE(
        HasErrorCode(VerifyWithTypeCheck(std::make_shared<ForStmt>(i, zero, ten, tensor_val, std::vector<IterArgPtr>{},
                                                                   yield, std::vector<VarPtr>{}, Sp())),
                     static_cast<int>(typecheck::ErrorType::FOR_RANGE_MUST_BE_SCALAR)));
}

TEST_F(VerifierTypeCheckTest, WhileStmt_IterArgYieldTypeMismatch_ReportsError)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto zero = std::make_shared<ConstInt>(int64_t(0), DataType::INT32, Sp());
    auto iter_arg = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), zero, Sp());
    auto ret_var = std::make_shared<Var>("acc_out", Scalar(DataType::INT32), Sp());
    auto yield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstFloat>(1.0, DataType::FP32, Sp())}, Sp());

    auto while_stmt = std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{iter_arg}, yield,
                                                  std::vector<VarPtr>{ret_var}, Sp());
    auto diags = VerifyWithTypeCheck(while_stmt);
    EXPECT_TRUE(HasErrorCode(diags, static_cast<int>(typecheck::ErrorType::DTYPE_MISMATCH)));
}

TEST_F(VerifierTypeCheckTest, WhileStmt_UnknownInitWithConcreteCarry_NoError)
{
    auto concrete_type = Scalar(DataType::INT32);
    auto none = std::make_shared<Var>("None", GetUnknownType(), Sp());
    auto iter_var = std::make_shared<Var>("value_iter", concrete_type, Sp());
    auto iter_arg = std::make_shared<IterArg>(iter_var, none);
    auto yielded = std::make_shared<Var>("value_next", concrete_type, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{yielded}, Sp());
    auto return_var = std::make_shared<Var>("value_result", concrete_type, Sp());
    auto while_stmt = std::make_shared<WhileStmt>(std::make_shared<ConstBool>(true, Sp()),
                                                  std::vector<IterArgPtr>{iter_arg}, yield,
                                                  std::vector<VarPtr>{return_var}, Sp());

    EXPECT_TRUE(VerifyWithTypeCheck(while_stmt).empty());
}

TEST_F(VerifierTypeCheckTest, IfStmt_ValidBranches_NoError)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());

    EXPECT_EQ(
        VerifyWithTypeCheck(std::make_shared<IfStmt>(cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp())).size(),
        0u);

    EXPECT_EQ(VerifyWithTypeCheck(std::make_shared<IfStmt>(cond, yield, yield, std::vector<VarPtr>{}, Sp())).size(),
              0u);
}

// ============================================================================
// verify_no_nested_call_pass.cpp: Nested call detection
// ============================================================================

class VerifierNoNestedCallTest : public testing::Test {};

TEST_F(VerifierNoNestedCallTest, CallInCallArgs_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateNoNestedCallPropertyVerifier());

    auto inner_call = std::make_shared<Call>("inner_op", std::vector<ExprPtr>{}, Sp());
    auto outer_call = std::make_shared<Call>("outer_op", std::vector<ExprPtr>{inner_call}, Sp());
    auto eval = std::make_shared<EvalStmt>(outer_call, Sp());
    auto program = MakeProgram("f", eval);
    auto diags = v.Verify(program);
    EXPECT_GE(diags.size(), 1u);
}

TEST_F(VerifierNoNestedCallTest, CallInIfCondition_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateNoNestedCallPropertyVerifier());

    auto call_cond = std::make_shared<Call>("cond_op", std::vector<ExprPtr>{}, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto if_stmt = std::make_shared<IfStmt>(call_cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp());
    auto program = MakeProgram("f", if_stmt);
    auto diags = v.Verify(program);
    EXPECT_GE(diags.size(), 1u);
}

TEST_F(VerifierNoNestedCallTest, CallInForRange_ReportsError)
{
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(int64_t(0), DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(int64_t(10), DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto call_expr = std::make_shared<Call>("op", std::vector<ExprPtr>{}, Sp());

    auto verify_for = [&](const ExprPtr& start, const ExprPtr& stop, const ExprPtr& step) {
        auto for_stmt = std::make_shared<ForStmt>(i, start, stop, step, std::vector<IterArgPtr>{}, yield,
                                                  std::vector<VarPtr>{}, Sp());
        auto diags = VerifyWithNoNestedCall(for_stmt);
        EXPECT_FALSE(diags.empty());
    };

    verify_for(call_expr, ten, one);
    verify_for(zero, call_expr, one);
    verify_for(zero, ten, call_expr);
}

TEST_F(VerifierNoNestedCallTest, CallInWhileCondition_ReportsError)
{
    IRVerifier v;
    v.AddRule(CreateNoNestedCallPropertyVerifier());

    auto call_cond = std::make_shared<Call>("cond_op", std::vector<ExprPtr>{}, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto while_stmt = std::make_shared<WhileStmt>(call_cond, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{},
                                                  Sp());
    auto program = MakeProgram("f", while_stmt);
    auto diags = v.Verify(program);
    EXPECT_GE(diags.size(), 1u);
}

TEST_F(VerifierNoNestedCallTest, CallInBinaryExpr_ReportsError)
{
    auto call_expr = std::make_shared<Call>("op", std::vector<ExprPtr>{}, Sp());
    auto one = std::make_shared<ConstInt>(int64_t(1), DataType::INT32, Sp());
    auto true_val = std::make_shared<ConstBool>(true, Sp());

    auto verify_binary = [&](const ExprPtr& expr) {
        auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
        auto assign = std::make_shared<AssignStmt>(x, expr, Sp());
        auto diags = VerifyWithNoNestedCall(assign);
        EXPECT_FALSE(diags.empty());
    };

    verify_binary(std::make_shared<Add>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<Sub>(one, call_expr, DataType::INT32, Sp()));
    verify_binary(std::make_shared<Mul>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<Eq>(call_expr, one, DataType::BOOL, Sp()));
    verify_binary(std::make_shared<And>(call_expr, true_val, DataType::BOOL, Sp()));
    verify_binary(std::make_shared<BitAnd>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<FloorDiv>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<FloorMod>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<FloatDiv>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<Min>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<Max>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<Pow>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<Ne>(call_expr, one, DataType::BOOL, Sp()));
    verify_binary(std::make_shared<Lt>(call_expr, one, DataType::BOOL, Sp()));
    verify_binary(std::make_shared<Le>(call_expr, one, DataType::BOOL, Sp()));
    verify_binary(std::make_shared<Gt>(call_expr, one, DataType::BOOL, Sp()));
    verify_binary(std::make_shared<Ge>(call_expr, one, DataType::BOOL, Sp()));
    verify_binary(std::make_shared<Or>(call_expr, true_val, DataType::BOOL, Sp()));
    verify_binary(std::make_shared<Xor>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<BitOr>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<BitXor>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<BitShiftLeft>(call_expr, one, DataType::INT32, Sp()));
    verify_binary(std::make_shared<BitShiftRight>(call_expr, one, DataType::INT32, Sp()));
}

TEST_F(VerifierNoNestedCallTest, CallInUnaryExpr_ReportsError)
{
    auto call_expr = std::make_shared<Call>("op", std::vector<ExprPtr>{}, Sp());

    auto verify_unary = [&](const ExprPtr& expr) {
        auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
        auto assign = std::make_shared<AssignStmt>(x, expr, Sp());
        auto diags = VerifyWithNoNestedCall(assign);
        EXPECT_FALSE(diags.empty());
    };

    verify_unary(std::make_shared<Neg>(call_expr, DataType::INT32, Sp()));
    verify_unary(std::make_shared<Abs>(call_expr, DataType::FP32, Sp()));
    verify_unary(std::make_shared<Not>(call_expr, DataType::BOOL, Sp()));
    verify_unary(std::make_shared<BitNot>(call_expr, DataType::INT32, Sp()));
    verify_unary(std::make_shared<Cast>(call_expr, DataType::FP32, Sp()));
}

TEST_F(VerifierNoNestedCallTest, NoNesting_Clean)
{
    IRVerifier v;
    v.AddRule(CreateNoNestedCallPropertyVerifier());

    auto val = std::make_shared<ConstInt>(int64_t(42), DataType::INT32, Sp());
    auto call = std::make_shared<Call>("op", std::vector<ExprPtr>{val}, Sp());
    auto eval = std::make_shared<EvalStmt>(call, Sp());
    auto program = MakeProgram("f", eval);
    auto diags = v.Verify(program);
    EXPECT_TRUE(diags.empty());
}

TEST_F(VerifierNoNestedCallTest, BitCastAsNestedCall_NoError)
{
    IRVerifier v;
    v.AddRule(CreateNoNestedCallPropertyVerifier());

    auto bit_cast_call = std::make_shared<Call>("vf.bit_cast", std::vector<ExprPtr>{}, Sp());
    auto outer_call = std::make_shared<Call>("vf.xor", std::vector<ExprPtr>{bit_cast_call}, Sp());
    auto eval = std::make_shared<EvalStmt>(outer_call, Sp());
    auto program = MakeProgram("f", eval);
    auto diags = v.Verify(program);
    EXPECT_TRUE(diags.empty());
}

TEST_F(VerifierNoNestedCallTest, ErrorTypeToString_ReturnsCorrectString)
{
    using namespace nested_call;
    EXPECT_EQ(ErrorTypeToString(ErrorType::CALL_IN_CALL_ARGS), "CALL_IN_CALL_ARGS");
    EXPECT_EQ(ErrorTypeToString(ErrorType::CALL_IN_IF_CONDITION), "CALL_IN_IF_CONDITION");
    EXPECT_EQ(ErrorTypeToString(ErrorType::CALL_IN_FOR_RANGE), "CALL_IN_FOR_RANGE");
    EXPECT_EQ(ErrorTypeToString(ErrorType::CALL_IN_BINARY_EXPR), "CALL_IN_BINARY_EXPR");
    EXPECT_EQ(ErrorTypeToString(ErrorType::CALL_IN_UNARY_EXPR), "CALL_IN_UNARY_EXPR");
    EXPECT_EQ(ErrorTypeToString(ErrorType::CALL_IN_WHILE_CONDITION), "CALL_IN_WHILE_CONDITION");
}

} // namespace ir
} // namespace pypto
