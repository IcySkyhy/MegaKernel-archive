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
 * \file test_core.cpp
 * \brief Unit tests for IR core (Span, Error, Backtrace)
 */

#include "gtest/gtest.h"

#include <string>

#include "core/error.h"
#include "ir/core.h"

namespace pypto {
namespace ir {

class IRCoreExtTest : public testing::Test {};

// ============================================================================
// Span::to_string Tests
// ============================================================================

TEST_F(IRCoreExtTest, TestSpanToString)
{
    Span sp("test.py", 10, 5, 20, 15);
    std::string result = sp.ToString();
    ASSERT_NE(result.find("test.py"), std::string::npos);
    ASSERT_NE(result.find("10"), std::string::npos);
    ASSERT_NE(result.find("5"), std::string::npos);
}

TEST_F(IRCoreExtTest, TestSpanToStringUnknown)
{
    Span sp = Span::Unknown();
    std::string result = sp.ToString();
    ASSERT_FALSE(result.empty());
}

// ============================================================================
// Error Tests
// ============================================================================

TEST_F(IRCoreExtTest, TestInternalErrorThrow)
{
    ASSERT_THROW({ throw InternalError("test error"); }, InternalError);
}

TEST_F(IRCoreExtTest, TestRuntimeErrorThrow)
{
    ASSERT_THROW({ throw RuntimeError("test runtime error"); }, RuntimeError);
}

TEST_F(IRCoreExtTest, TestTypeErrorThrow)
{
    ASSERT_THROW({ throw TypeError("test type error"); }, TypeError);
}

TEST_F(IRCoreExtTest, TestValueErrorThrow)
{
    ASSERT_THROW({ throw ValueError("test value error"); }, ValueError);
}

TEST_F(IRCoreExtTest, TestErrorGetFullMessage)
{
    try {
        throw InternalError("test message");
    } catch (const Error& e) {
        std::string msg = e.GetFullMessage();
        ASSERT_NE(msg.find("test message"), std::string::npos);
    }
}

TEST_F(IRCoreExtTest, TestErrorGetFormattedStackTrace)
{
    try {
        throw InternalError("test");
    } catch (const Error& e) {
        // Just verify it doesn't crash
        std::string trace = e.GetFormattedStackTrace();
        // trace may or may not be empty depending on build mode
        (void)trace;
    }
}

// ============================================================================
// Backtrace Tests
// ============================================================================

TEST_F(IRCoreExtTest, TestBacktraceGetInstance)
{
    auto& bt = Backtrace::GetInstance();
    (void)bt;
}

TEST_F(IRCoreExtTest, TestBacktraceCaptureStackTrace)
{
    auto& bt = Backtrace::GetInstance();
    auto frames = bt.CaptureStackTrace(0);
    // Should capture at least some frames
    ASSERT_FALSE(frames.empty());
}

TEST_F(IRCoreExtTest, TestBacktraceFormatStackTrace)
{
    auto& bt = Backtrace::GetInstance();
    auto frames = bt.CaptureStackTrace(0);
    std::string formatted = Backtrace::FormatStackTrace(frames);
    // May or may not have content depending on debug info
    (void)formatted;
}

TEST_F(IRCoreExtTest, TestBacktraceFormatEmptyTrace)
{
    std::vector<StackFrame> emptyFrames;
    std::string formatted = Backtrace::FormatStackTrace(emptyFrames);
    ASSERT_TRUE(formatted.empty());
}

TEST_F(IRCoreExtTest, TestStackFrameToString)
{
    StackFrame frame("test_func", "test.cpp", 42, 0x1234);
    std::string result = frame.ToString();
    ASSERT_NE(result.find("test_func"), std::string::npos);
    ASSERT_NE(result.find("test.cpp"), std::string::npos);
}

TEST_F(IRCoreExtTest, TestStackFrameToStringNoFile)
{
    StackFrame frame("test_func", "", 0, 0x1234);
    std::string result = frame.ToString();
    ASSERT_NE(result.find("test_func"), std::string::npos);
}

TEST_F(IRCoreExtTest, TestStackFrameToStringNoFunction)
{
    StackFrame frame("", "test.cpp", 42, 0x1234);
    std::string result = frame.ToString();
    // Should show hex address when no function name
    ASSERT_NE(result.find("0x"), std::string::npos);
}

} // namespace ir
} // namespace pypto
