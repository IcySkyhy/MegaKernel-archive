/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_device_trace.cpp
 * \brief Unit tests for device_trace.cpp covering DeviceTrace class functionality
 */

#include <gtest/gtest.h>
#include <string>
#include <cstring>
#include <thread>
#include <vector>

#include "machine/device/dynamic/device_trace.h"

using namespace npu::tile_fwk::dynamic;

class TestDeviceTrace : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestDeviceTrace, GetInstance_ReturnsValidSingleton)
{
    DeviceTrace& instance1 = DeviceTrace::GetInstance();
    DeviceTrace& instance2 = DeviceTrace::GetInstance();
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(TestDeviceTrace, GetInstance_MultipleCallsReturnSameInstance)
{
    for (int i = 0; i < 10; i++) {
        DeviceTrace& instance = DeviceTrace::GetInstance();
        DeviceTrace& refInstance = DeviceTrace::GetInstance();
        EXPECT_EQ(&instance, &refInstance);
    }
}

TEST_F(TestDeviceTrace, SubmitTraceMsg_WithValidMessage)
{
    DeviceTrace& instance = DeviceTrace::GetInstance();
    std::string testMsg = "Test trace message for unit test";
    instance.SubmitTraceMsg(testMsg);
    SUCCEED();
}

TEST_F(TestDeviceTrace, SubmitTraceMsg_WithEmptyMessage)
{
    DeviceTrace& instance = DeviceTrace::GetInstance();
    std::string emptyMsg = "";
    instance.SubmitTraceMsg(emptyMsg);
    SUCCEED();
}

TEST_F(TestDeviceTrace, SubmitTraceMsg_WithLongMessage)
{
    DeviceTrace& instance = DeviceTrace::GetInstance();
    std::string longMsg;
    for (size_t i = 0; i < 500; i++) {
        longMsg += "abcdefghij";
    }
    instance.SubmitTraceMsg(longMsg);
    SUCCEED();
}

TEST_F(TestDeviceTrace, SubmitTraceMsg_WithSpecialCharacters)
{
    DeviceTrace& instance = DeviceTrace::GetInstance();
    std::string specialMsg = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?\n\t\r";
    instance.SubmitTraceMsg(specialMsg);
    SUCCEED();
}

TEST_F(TestDeviceTrace, SubmitTraceMsg_WithUnicodeCharacters)
{
    DeviceTrace& instance = DeviceTrace::GetInstance();
    std::string unicodeMsg = "Unicode test: 中文测试";
    instance.SubmitTraceMsg(unicodeMsg);
    SUCCEED();
}

TEST_F(TestDeviceTrace, ReportTraceMsg_MultipleCalls)
{
    DeviceTrace& instance = DeviceTrace::GetInstance();
    for (int i = 0; i < 5; i++) {
        instance.ReportTraceMsg();
    }
    SUCCEED();
}

TEST_F(TestDeviceTrace, SubmitAndReport_TraceWorkflow)
{
    DeviceTrace& instance = DeviceTrace::GetInstance();
    instance.SubmitTraceMsg("First trace message");
    instance.SubmitTraceMsg("Second trace message");
    instance.SubmitTraceMsg("Third trace message");
    instance.ReportTraceMsg();
    SUCCEED();
}

TEST_F(TestDeviceTrace, MAX_MSG_LEN_ValueCheck) { EXPECT_EQ(MAX_MSG_LEN, 112); }

TEST_F(TestDeviceTrace, Filename_Macro_BasicPath)
{
    const char* path = "/path/to/test_file.cpp";
    const char* filename = strrchr(path, '/') ? strrchr(path, '/') + 1 : path;
    EXPECT_STREQ(filename, "test_file.cpp");
}

TEST_F(TestDeviceTrace, Filename_Macro_NoSlash)
{
    const char* path = "simple_file.cpp";
    const char* filename = strrchr(path, '/') ? strrchr(path, '/') + 1 : path;
    EXPECT_STREQ(filename, "simple_file.cpp");
}

TEST_F(TestDeviceTrace, Filename_Macro_EmptyPath)
{
    const char* path = "";
    const char* filename = strrchr(path, '/') ? strrchr(path, '/') + 1 : path;
    EXPECT_STREQ(filename, "");
}

TEST_F(TestDeviceTrace, ConcurrentAccess_ThreadSafety)
{
    DeviceTrace& instance = DeviceTrace::GetInstance();

    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&instance, i]() { instance.SubmitTraceMsg("Thread message " + std::to_string(i)); });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    SUCCEED();
}

TEST_F(TestDeviceTrace, ConcurrentAccess_MixedOperations)
{
    DeviceTrace& instance = DeviceTrace::GetInstance();

    std::vector<std::thread> threads;
    for (int i = 0; i < 20; i++) {
        if (i % 3 == 0) {
            threads.emplace_back(
                [&instance, i]() { instance.SubmitTraceMsg("Submit from thread " + std::to_string(i)); });
        } else if (i % 3 == 1) {
            threads.emplace_back([&instance]() { instance.ReportTraceMsg(); });
        } else {
            threads.emplace_back([&instance]() {
                DeviceTrace& localInstance = DeviceTrace::GetInstance();
                EXPECT_EQ(&localInstance, &instance);
            });
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    SUCCEED();
}

TEST_F(TestDeviceTrace, StressTest_ManyMessages)
{
    DeviceTrace& instance = DeviceTrace::GetInstance();
    for (int i = 0; i < 1000; i++) {
        instance.SubmitTraceMsg("Stress test message " + std::to_string(i));
    }
    instance.ReportTraceMsg();
    SUCCEED();
}

TEST_F(TestDeviceTrace, Integration_SubmitReportSequence)
{
    DeviceTrace& instance = DeviceTrace::GetInstance();

    std::vector<std::string> messages = {"Initialization started", "Loading configuration", "Processing data",
                                         "Executing kernel",       "Finalizing results",    "Cleanup completed"};

    for (const auto& msg : messages) {
        instance.SubmitTraceMsg(msg);
    }

    instance.ReportTraceMsg();
    SUCCEED();
}
