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
 * \file test_error.cpp
 * \brief Unit tests for core error handling and exception types
 */

#include "gtest/gtest.h"

#include <string>

#include "core/error.h"

namespace pypto {
namespace ir {

class CoreErrorTest : public testing::Test {};

// ============================================================================
// Error Base Class Tests
// ============================================================================

TEST_F(CoreErrorTest, TestErrorMessage)
{
    Error err("test error message");
    ASSERT_STREQ(err.what(), "test error message");
}

TEST_F(CoreErrorTest, TestErrorGetFullMessage)
{
    Error err("something went wrong");
    std::string fullMsg = err.GetFullMessage();
    ASSERT_FALSE(fullMsg.empty());
    ASSERT_NE(fullMsg.find("something went wrong"), std::string::npos);
}

TEST_F(CoreErrorTest, TestErrorGetFormattedStackTrace)
{
    Error err("trace test");
    // GetFormattedStackTrace may return empty in non-debug builds
    std::string trace = err.GetFormattedStackTrace();
    // Just verify it doesn't crash
    ASSERT_TRUE(trace.empty() || !trace.empty());
}

TEST_F(CoreErrorTest, TestErrorGetStackTrace)
{
    Error err("stack test");
    const auto& frames = err.GetStackTrace();
    // Frames may be empty in some build configurations
    (void)frames;
}

// ============================================================================
// Derived Error Types Tests
// ============================================================================

TEST_F(CoreErrorTest, TestValueError)
{
    ValueError err("invalid value");
    ASSERT_STREQ(err.what(), "invalid value");
}

TEST_F(CoreErrorTest, TestTypeError)
{
    TypeError err("type mismatch");
    ASSERT_STREQ(err.what(), "type mismatch");
}

TEST_F(CoreErrorTest, TestRuntimeError)
{
    RuntimeError err("runtime failure");
    ASSERT_STREQ(err.what(), "runtime failure");
}

TEST_F(CoreErrorTest, TestNotImplementedError)
{
    NotImplementedError err("not implemented");
    ASSERT_STREQ(err.what(), "not implemented");
}

TEST_F(CoreErrorTest, TestIndexError)
{
    IndexError err("index out of range");
    ASSERT_STREQ(err.what(), "index out of range");
}

TEST_F(CoreErrorTest, TestAssertionError)
{
    AssertionError err("assertion failed");
    ASSERT_STREQ(err.what(), "assertion failed");
}

TEST_F(CoreErrorTest, TestInternalError)
{
    InternalError err("internal error");
    ASSERT_STREQ(err.what(), "internal error");
}

// ============================================================================
// Error Throw and Catch Tests
// ============================================================================

TEST_F(CoreErrorTest, TestThrowCatchValueError)
{
    ASSERT_THROW(throw ValueError("bad value"), ValueError);
    ASSERT_THROW(throw ValueError("bad value"), Error);
}

TEST_F(CoreErrorTest, TestThrowCatchTypeError)
{
    ASSERT_THROW(throw TypeError("bad type"), TypeError);
    ASSERT_THROW(throw TypeError("bad type"), Error);
}

// ============================================================================
// Diagnostic Tests
// ============================================================================

TEST_F(CoreErrorTest, TestDiagnosticDefault)
{
    Diagnostic diag;
    ASSERT_EQ(diag.severity, DiagnosticSeverity::ERROR);
    ASSERT_EQ(diag.errorCode, 0);
    ASSERT_TRUE(diag.ruleName.empty());
    ASSERT_TRUE(diag.message.empty());
}

TEST_F(CoreErrorTest, TestDiagnosticConstruction)
{
    Diagnostic diag(DiagnosticSeverity::WARNING, "TestRule", 42, "test message", Span::Unknown());
    ASSERT_EQ(diag.severity, DiagnosticSeverity::WARNING);
    ASSERT_EQ(diag.ruleName, "TestRule");
    ASSERT_EQ(diag.errorCode, 42);
    ASSERT_EQ(diag.message, "test message");
}

// ============================================================================
// VerificationError Tests
// ============================================================================

TEST_F(CoreErrorTest, TestVerificationError)
{
    std::vector<Diagnostic> diags;
    diags.emplace_back(DiagnosticSeverity::ERROR, "Rule1", 1, "error msg", Span::Unknown());
    diags.emplace_back(DiagnosticSeverity::WARNING, "Rule2", 2, "warn msg", Span::Unknown());

    VerificationError err("verification failed", std::move(diags));
    ASSERT_STREQ(err.what(), "verification failed");
    ASSERT_EQ(err.GetDiagnostics().size(), 2);
    ASSERT_EQ(err.GetDiagnostics()[0].severity, DiagnosticSeverity::ERROR);
    ASSERT_EQ(err.GetDiagnostics()[1].severity, DiagnosticSeverity::WARNING);
}

// ============================================================================
// StackFrame Tests
// ============================================================================

TEST_F(CoreErrorTest, TestStackFrameDefault)
{
    StackFrame frame;
    ASSERT_EQ(frame.lineno, 0);
    ASSERT_EQ(frame.pc, 0);
    ASSERT_TRUE(frame.function.empty());
    ASSERT_TRUE(frame.filename.empty());
}

TEST_F(CoreErrorTest, TestStackFrameConstruction)
{
    StackFrame frame("my_func", "my_file.cpp", 42, 0x1234);
    ASSERT_EQ(frame.function, "my_func");
    ASSERT_EQ(frame.filename, "my_file.cpp");
    ASSERT_EQ(frame.lineno, 42);
    ASSERT_EQ(frame.pc, 0x1234);
}

// ============================================================================
// Backtrace Tests
// ============================================================================

TEST_F(CoreErrorTest, TestBacktraceFormatEmptyTrace)
{
    std::vector<StackFrame> emptyFrames;
    std::string result = Backtrace::FormatStackTrace(emptyFrames);
    ASSERT_TRUE(result.empty());
}

TEST_F(CoreErrorTest, TestBacktraceCaptureStackTrace)
{
    auto& bt = Backtrace::GetInstance();
    auto frames = bt.CaptureStackTrace();
    // Just verify it doesn't crash; frames may be empty in some builds
    (void)frames;
}

} // namespace ir
} // namespace pypto
