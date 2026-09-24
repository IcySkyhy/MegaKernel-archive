/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_backtrace.cpp
 * \brief Backtrace test using new IR error mechanism (pypto::ir::Error)
 *
 * This test demonstrates the optimized backtrace implementation in the new IR.
 * Key features:
 * - Python-style traceback header
 * - Stack reversal (most recent call last)
 * - File/Line format in Debug mode
 * - Source code display in Debug mode
 * - Graceful fallback in Release mode
 *
 * Usage:
 *   # Release mode (traditional format with Python-style header and stack reversal)
 *   python3 build_ci.py -u=BacktraceTest.NestedCallTest -d=11 -f cpp
 *
 *   # Debug mode (full optimization with File/Line and source code)
 *   python3 build_ci.py -u=BacktraceTest.NestedCallTest -d=11 -f cpp --build_type=Debug
 */

#include "gtest/gtest.h"
#include "core/error.h"
#include <iostream>

using namespace pypto::ir;

class BacktraceTest : public testing::Test {
protected:
    void SetUp() override
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BACKTRACE OUTPUT TEST (New IR Error Mechanism)\n";
        std::cout << std::string(80, '=') << "\n";
    }

    void TearDown() override { std::cout << std::string(80, '=') << "\n\n"; }
};

/**
 * Test 1: Simple Error Exception
 *
 * This test throws a simple error and prints the backtrace using new IR error mechanism.
 * Expected output:
 * - Release mode: Python-style header + stack reversal + traditional format
 * - Debug mode: Python-style header + stack reversal + File/Line format + source code
 */
TEST_F(BacktraceTest, SimpleErrorTest)
{
    std::cout << "\nTest: Simple Error Exception\n";
    std::cout << std::string(40, '-') << "\n\n";

    try {
        // Throw an error with backtrace (new IR error mechanism)
        throw Error("Test error: Invalid operation");
    } catch (const Error& e) {
        // Print the complete error message with backtrace
        std::cout << "Caught Error Exception:\n";
        std::cout << e.GetFullMessage() << "\n";
    }
}

/**
 * Test 2: Nested Function Calls
 *
 * This test demonstrates backtrace with multiple function call levels.
 * The output will show the full call stack with proper formatting.
 */

// Helper functions to create nested call stack
void Level3Function() { throw Error("Error occurred at Level 3"); }

void Level2Function()
{
    Level3Function(); // Call next level
}

void Level1Function()
{
    Level2Function(); // Call next level
}

TEST_F(BacktraceTest, NestedCallTest)
{
    std::cout << "\nTest: Nested Function Calls (3 levels)\n";
    std::cout << std::string(40, '-') << "\n\n";

    try {
        Level1Function(); // Start the call chain
    } catch (const Error& e) {
        std::cout << "Caught Error from Nested Calls:\n";
        std::cout << e.GetFullMessage() << "\n";
    }
}

// ============================================================================
// StackFrame::to_string Tests
// ============================================================================

TEST_F(BacktraceTest, StackFrameToStringWithFunction)
{
    StackFrame frame("myFunction", "", 0, 0);
    std::string result = frame.ToString();
    ASSERT_NE(result.find("myFunction"), std::string::npos);
}

TEST_F(BacktraceTest, StackFrameToStringWithoutFunction)
{
    StackFrame frame("", "", 0, 0xDEADBEEF);
    std::string result = frame.ToString();
    ASSERT_NE(result.find("0x"), std::string::npos);
    ASSERT_NE(result.find("deadbeef"), std::string::npos);
}

TEST_F(BacktraceTest, StackFrameToStringWithFileAndLine)
{
    StackFrame frame("myFunc", "/path/to/file.cpp", 42, 0x1234);
    std::string result = frame.ToString();
    ASSERT_NE(result.find("myFunc"), std::string::npos);
    ASSERT_NE(result.find("/path/to/file.cpp"), std::string::npos);
    ASSERT_NE(result.find("42"), std::string::npos);
}

TEST_F(BacktraceTest, StackFrameToStringWithFileNoLine)
{
    StackFrame frame("myFunc", "/path/to/file.cpp", 0, 0x1234);
    std::string result = frame.ToString();
    ASSERT_NE(result.find("myFunc"), std::string::npos);
    ASSERT_NE(result.find("/path/to/file.cpp"), std::string::npos);
    // lineno == 0, so no ":0" in the output
    ASSERT_EQ(result.find(":0"), std::string::npos);
}

TEST_F(BacktraceTest, StackFrameDefaultConstructor)
{
    StackFrame frame;
    ASSERT_EQ(frame.lineno, 0);
    ASSERT_EQ(frame.pc, 0u);
    ASSERT_TRUE(frame.function.empty());
    ASSERT_TRUE(frame.filename.empty());
}

// ============================================================================
// FormatStackTrace Tests with manually constructed frames
// ============================================================================

TEST_F(BacktraceTest, FormatStackTraceEmpty)
{
    auto& bt = Backtrace::GetInstance();
    std::vector<StackFrame> frames;
    std::string result = bt.FormatStackTrace(frames);
    ASSERT_TRUE(result.empty());
}

TEST_F(BacktraceTest, FormatStackTraceWithFileAndLine)
{
    auto& bt = Backtrace::GetInstance();

    StackFrame frame1("funcA", "/src/a.cpp", 10, 0x1000);
    StackFrame frame2("funcB", "/src/b.cpp", 20, 0x2000);
    std::vector<StackFrame> frames = {frame1, frame2};

    std::string result = bt.FormatStackTrace(frames);
    // Frames are reversed (most recent last)
    ASSERT_NE(result.find("/src/a.cpp"), std::string::npos);
    ASSERT_NE(result.find("/src/b.cpp"), std::string::npos);
}

TEST_F(BacktraceTest, FormatStackTraceReleaseModeFormat)
{
    auto& bt = Backtrace::GetInstance();

    // Frame without filename but with function, libname, offset
    StackFrame frame;
    frame.function = "testFunc";
    frame.libname = "libtest.so";
    frame.offset = "+0x42";
    frame.pc = 0xABCD;
    std::vector<StackFrame> frames = {frame};

    std::string result = bt.FormatStackTrace(frames);
    ASSERT_NE(result.find("libtest.so"), std::string::npos);
    ASSERT_NE(result.find("testFunc"), std::string::npos);
    ASSERT_NE(result.find("+0x42"), std::string::npos);
}

TEST_F(BacktraceTest, FormatStackTraceReleaseModeNoOffset)
{
    auto& bt = Backtrace::GetInstance();

    StackFrame frame;
    frame.function = "testFunc";
    frame.libname = "libtest.so";
    frame.pc = 0xABCD;
    std::vector<StackFrame> frames = {frame};

    std::string result = bt.FormatStackTrace(frames);
    ASSERT_NE(result.find("libtest.so(testFunc)"), std::string::npos);
}

TEST_F(BacktraceTest, FormatStackTraceReleaseModeNoLibname)
{
    auto& bt = Backtrace::GetInstance();

    StackFrame frame;
    frame.function = "testFunc";
    frame.pc = 0xABCD;
    std::vector<StackFrame> frames = {frame};

    std::string result = bt.FormatStackTrace(frames);
    ASSERT_NE(result.find("testFunc"), std::string::npos);
}

TEST_F(BacktraceTest, FormatStackTraceFiltering)
{
    auto& bt = Backtrace::GetInstance();

    // Frame with filtered filename should be excluded
    StackFrame frameFiltered;
    frameFiltered.function = "filtered_func";
    frameFiltered.filename = "/path/to/nanobind/module.cpp";
    frameFiltered.lineno = 10;
    frameFiltered.pc = 0x1000;

    StackFrame frameKept;
    frameKept.function = "kept_func";
    frameKept.filename = "/path/to/my_code.cpp";
    frameKept.lineno = 20;
    frameKept.pc = 0x2000;

    std::vector<StackFrame> frames = {frameFiltered, frameKept};
    std::string result = bt.FormatStackTrace(frames);
    ASSERT_EQ(result.find("nanobind"), std::string::npos);
    ASSERT_NE(result.find("my_code.cpp"), std::string::npos);
}

TEST_F(BacktraceTest, FormatStackTraceDeduplication)
{
    auto& bt = Backtrace::GetInstance();

    // Two frames with same PC address - only first should be kept
    StackFrame frame1("funcA", "/src/a.cpp", 10, 0x1000);
    StackFrame frame2("funcA_inline", "/src/a.cpp", 11, 0x1000);
    StackFrame frame3("funcB", "/src/b.cpp", 20, 0x2000);

    std::vector<StackFrame> frames = {frame1, frame2, frame3};
    std::string result = bt.FormatStackTrace(frames);
    // After reversal and dedup, frame3 comes first, then one of frame1/frame2
    ASSERT_NE(result.find("/src/b.cpp"), std::string::npos);
    ASSERT_NE(result.find("/src/a.cpp"), std::string::npos);
}

// ============================================================================
// CaptureStackTrace Tests
// ============================================================================

TEST_F(BacktraceTest, CaptureStackTraceBasic)
{
    auto& bt = Backtrace::GetInstance();
    auto frames = bt.CaptureStackTrace(0);
    // We should get at least a few frames
    ASSERT_GT(frames.size(), 0u);
}

TEST_F(BacktraceTest, CaptureStackTraceSkipAll)
{
    auto& bt = Backtrace::GetInstance();
    // Skip more frames than available - should return empty
    auto frames = bt.CaptureStackTrace(9999);
    ASSERT_TRUE(frames.empty());
}

// ============================================================================
// GetInstance singleton
// ============================================================================

TEST_F(BacktraceTest, GetInstanceSingleton)
{
    auto& bt1 = Backtrace::GetInstance();
    auto& bt2 = Backtrace::GetInstance();
    ASSERT_EQ(&bt1, &bt2);
}
