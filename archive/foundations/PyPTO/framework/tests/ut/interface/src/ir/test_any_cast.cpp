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
 * \file test_any_cast.cpp
 * \brief Unit tests for AnyCast type conversion utilities
 */

#include "gtest/gtest.h"

#include <any>
#include <string>

#include "core/any_cast.h"

namespace pypto {

TEST(IR_CoreAnyCastTest, TestAnyCastInt)
{
    // Test casting to int
    std::any value = 42;
    int result = AnyCast<int>(value);
    ASSERT_EQ(result, 42);
}

TEST(CoreAnyCastTest, TestAnyCastString)
{
    // Test casting to string
    std::any value = std::string("hello");
    std::string result = AnyCast<std::string>(value);
    ASSERT_EQ(result, "hello");
}

TEST(CoreAnyCastTest, TestAnyCastDouble)
{
    // Test casting to double
    std::any value = 3.14;
    double result = AnyCast<double>(value);
    ASSERT_DOUBLE_EQ(result, 3.14);
}

TEST(CoreAnyCastTest, TestAnyCastBool)
{
    // Test casting to bool
    std::any value = true;
    bool result = AnyCast<bool>(value);
    ASSERT_EQ(result, true);
}

// Temporarily commented out - depends on core/error.h
/*
TEST(CoreAnyCastTest, TestAnyCastTypeMismatch) {
    // Test that type mismatch throws TypeError
    std::any value = 42;  // int

    try {
        AnyCast<std::string>(value);  // Try to cast to string
        FAIL() << "Expected TypeError to be thrown";
    } catch (const TypeError& e) {
        // Expected
        std::string msg = e.what();
        ASSERT_TRUE(msg.find("type") != std::string::npos ||
                    msg.find("Type") != std::string::npos ||
                    msg.find("mismatch") != std::string::npos);
    } catch (...) {
        FAIL() << "Expected TypeError but caught different exception";
    }
}

TEST(CoreAnyCastTest, TestAnyCastEmpty) {
    // Test casting from empty any
    std::any value;  // Empty

    try {
        AnyCast<int>(value);
        FAIL() << "Expected exception for empty any";
    } catch (const Error& e) {
        // Expected - should throw some error for empty any
        ASSERT_NE(e.what(), nullptr);
    }
}
*/

TEST(CoreAnyCastTest, TestAnyCastRef)
{
    // Test AnyCastRef - reference casting
    std::any value = 42;

    try {
        const int result = AnyCastRef<int>(value);
        ASSERT_EQ(result, 42);
    } catch (...) {
        FAIL() << "AnyCastRef should succeed for matching type";
    }
}

// Temporarily commented out - depends on core/error.h
/*
TEST(CoreAnyCastTest, TestAnyCastRefTypeMismatch) {
    // Test that AnyCastRef type mismatch throws TypeError
    std::any value = 42;  // int

    try {
        AnyCastRef<double>(value);  // Try to cast to double
        FAIL() << "Expected TypeError to be thrown";
    } catch (const TypeError& e) {
        // Expected
        ASSERT_NE(e.what(), nullptr);
    } catch (...) {
        FAIL() << "Expected TypeError but caught different exception";
    }
}
*/

TEST(CoreAnyCastTest, TestAnyCastMultipleTypes)
{
    // Test casting between different types in sequence
    std::any value1 = 100;
    std::any value2 = std::string("world");
    std::any value3 = 2.71;

    ASSERT_EQ(AnyCast<int>(value1), 100);
    ASSERT_EQ(AnyCast<std::string>(value2), "world");
    ASSERT_DOUBLE_EQ(AnyCast<double>(value3), 2.71);
}

TEST(CoreAnyCastTest, TestAnyCastPointer)
{
    // Test casting pointers
    int original = 42;
    std::any value = &original;

    try {
        int* result = AnyCast<int*>(value);
        ASSERT_EQ(*result, 42);
    } catch (...) {
        FAIL() << "AnyCast should succeed for pointer type";
    }
}

TEST(CoreAnyCastTest, TestAnyCastConstRef)
{
    // Test const reference casting
    std::any value = std::string("test");

    try {
        const std::string result = AnyCastRef<std::string>(value);
        ASSERT_EQ(result, "test");
    } catch (...) {
        FAIL() << "AnyCastRef should succeed for const reference";
    }
}

// Temporarily commented out - depends on core/error.h
/*
TEST(CoreAnyCastTest, TestAnyCastErrorMessage) {
    // Test that error message contains useful information
    std::any value = 42;

    try {
        AnyCast<std::string>(value);
        FAIL() << "Expected TypeError";
    } catch (const TypeError& e) {
        std::string msg = e.GetFullMessage();
        // Message should contain type information
        ASSERT_TRUE(!msg.empty());
    }
}
*/

} // namespace pypto
