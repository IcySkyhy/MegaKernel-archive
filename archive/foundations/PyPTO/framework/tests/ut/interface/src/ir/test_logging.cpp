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
 * \file test_logging.cpp
 * \brief Unit tests for logging framework
 */

#include "gtest/gtest.h"

#include <fstream>
#include <string>

#include "core/logging.h"

namespace pypto {

TEST(CoreLoggingTest, TestStdLogger)
{
    // Test basic StdLogger functionality
    auto logger = std::make_unique<StdLogger>();

    // Should not crash when logging
    logger->Log("Test message").Log("\n");
    logger->Log("Debug message").Log("\n");
    logger->Log("Warning message").Log("\n");
    logger->Log("Error message").Log("\n");
}

TEST(CoreLoggingTest, TestFileLogger)
{
    // Test FileLogger functionality
    std::string testFile = "/tmp/pypto_test_log.txt";

    {
        auto logger = std::make_unique<FileLogger>(testFile, false);
        logger->Log("File test message").Log("\n");
    } // Logger should flush and close on destruction

    // Read back and verify
    std::ifstream ifs(testFile);
    ASSERT_TRUE(ifs.is_open());

    std::string content;
    std::string line;
    while (std::getline(ifs, line)) {
        content += line + "\n";
    }
    ifs.close();

    // Should contain the message
    ASSERT_TRUE(content.find("File test message") != std::string::npos);

    // Clean up
    std::remove(testFile.c_str());
}

TEST(CoreLoggingTest, TestLineLogger)
{
    // Test LineLogger functionality (in-memory logging)
    auto logger = LoggerManager::LineLoggerRegister("test_logger");
    ASSERT_NE(logger, nullptr);

    logger->Log("Test line 1");
    logger->Log("Test line 2");
    logger->Log("Test line 3");

    // Should be able to retrieve the logger
    auto logger2 = LoggerManager::LineLoggerRegister("test_logger");
    ASSERT_EQ(logger, logger2); // Same instance
}

TEST(CoreLoggingTest, TestLogLevels)
{
    // Test different log levels exist
    ASSERT_EQ(static_cast<int>(LogLevel::DEBUG), 0);
    ASSERT_EQ(static_cast<int>(LogLevel::INFO), 1);
    ASSERT_EQ(static_cast<int>(LogLevel::WARN), 2);
    ASSERT_EQ(static_cast<int>(LogLevel::ERROR), 3);
    ASSERT_EQ(static_cast<int>(LogLevel::FATAL), 4);
    ASSERT_EQ(static_cast<int>(LogLevel::EVENT), 5);
}

TEST(CoreLoggingTest, TestStreamOperators)
{
    // Test that various types can be logged
    auto logger = std::make_unique<StdLogger>();

    logger->Log("String: ").Log("test").Log("\n");
    logger->Log("Int: ").Log(42).Log("\n");
    logger->Log("Float: ").Log(3.14).Log("\n");
    logger->Log("Bool: ").Log(true).Log("\n");
}

TEST(CoreLoggingTest, TestLoggerManagerRegister)
{
    // Test LoggerManager registration
    auto logger1 = LoggerManager::LineLoggerRegister("logger1");
    auto logger2 = LoggerManager::LineLoggerRegister("logger2");

    ASSERT_NE(logger1, nullptr);
    ASSERT_NE(logger2, nullptr);
    ASSERT_NE(logger1, logger2); // Different loggers

    // Registering same name should return same instance
    auto logger1Again = LoggerManager::LineLoggerRegister("logger1");
    ASSERT_EQ(logger1, logger1Again);
}

TEST(CoreLoggingTest, TestFileLoggerMultipleWrites)
{
    // Test multiple writes to file logger
    std::string testFile = "/tmp/pypto_test_log_multi.txt";

    {
        auto logger = std::make_unique<FileLogger>(testFile, false);
        logger->Log("Message 1").Log("\n");
        logger->Log("Message 2").Log("\n");
        logger->Log("Message 3").Log("\n");
    }

    std::ifstream ifs(testFile);
    ASSERT_TRUE(ifs.is_open());

    std::string content;
    std::string line;
    int lineCount = 0;
    while (std::getline(ifs, line)) {
        content += line + "\n";
        lineCount++;
    }
    ifs.close();

    // Should have all three messages
    ASSERT_TRUE(content.find("Message 1") != std::string::npos);
    ASSERT_TRUE(content.find("Message 2") != std::string::npos);
    ASSERT_TRUE(content.find("Message 3") != std::string::npos);
    ASSERT_GE(lineCount, 3);

    std::remove(testFile.c_str());
}

TEST(CoreLoggingTest, TestLoggerNotNull)
{
    // Test that loggers are not null
    auto stdLogger = std::make_unique<StdLogger>();
    ASSERT_NE(stdLogger, nullptr);

    auto lineLogger = LoggerManager::LineLoggerRegister("test");
    ASSERT_NE(lineLogger, nullptr);
}

} // namespace pypto
