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
 * \file test_device_error_tracking.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <functional>
#include "securec.h"
#include "tilefwk/device_error_code.h"
#include "machine/runtime/runner/device_error_tracking.h"

using namespace npu::tile_fwk;

std::string CaptureStdout(std::function<void()> func)
{
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return "";
    }

    int old_stdout = dup(STDOUT_FILENO);
    if (old_stdout == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        return "";
    }

    if (dup2(pipefd[1], STDOUT_FILENO) == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        close(old_stdout);
        return "";
    }

    close(pipefd[1]);
    func();
    fflush(stdout);

    if (dup2(old_stdout, STDOUT_FILENO) == -1) {
        close(pipefd[0]);
        close(old_stdout);
        return "";
    }

    char buffer[4096] = {0};
    ssize_t len = read(pipefd[0], buffer, sizeof(buffer) - 1);
    close(pipefd[0]);
    close(old_stdout);

    return std::string(buffer, len);
}

TEST(DeviceErrorTrackingTest, GetRetcodeMessageCoversAllRanges)
{
    EXPECT_STREQ(GetRetcodeMessage(PYPTO_DEVICE_ERROR_PARAM_INVALID), "param invalid");
    EXPECT_STREQ(GetRetcodeMessage(PYPTO_DEVICE_ERROR_SYMBOL_NOT_FOUND), "symbol not found");
    EXPECT_STREQ(GetRetcodeMessage(PYPTO_DEVICE_ERROR_FEATURE_NOT_SUPPORT), "feature not support");
    EXPECT_STREQ(GetRetcodeMessage(PYPTO_DEVICE_ERROR_TIMEOUT), "driver timeout");
    EXPECT_STREQ(GetRetcodeMessage(PYPTO_DEVICE_ERROR_INTERNAL_ERROR), "runtime internal error");
    EXPECT_STREQ(GetRetcodeMessage(PYPTO_DEVICE_ERROR_AICORE_EXCEPTION), "aicore exception");
    EXPECT_STREQ(GetRetcodeMessage(PYPTO_DEVICE_ERROR_AICPU_EXCEPTION), "aicpu exception");
}

TEST(DeviceErrorTrackingTest, GetRetcodeMessageUnknown)
{
    EXPECT_STREQ(GetRetcodeMessage(-1), "unknown error");
    EXPECT_STREQ(GetRetcodeMessage(0), "unknown error");
    EXPECT_STREQ(GetRetcodeMessage(999999), "unknown error");
}

TEST(DeviceErrorTrackingTest, PyPTOExceptionInfoCallBackOutputsCorrectInfo)
{
    AclRtExceptionInfo exceptionInfo{};
    memset_s(&exceptionInfo, sizeof(AclRtExceptionInfo), 0, sizeof(AclRtExceptionInfo));
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.taskid = 123;
    exceptionInfo.streamid = 456;
    exceptionInfo.deviceid = 0;
    exceptionInfo.retcode = PYPTO_DEVICE_ERROR_AICORE_EXCEPTION;

    char kernelName[] = "test_kernel";
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName = kernelName;

    std::string output = CaptureStdout([&]() { PyPTOExceptionInfoCallBack(&exceptionInfo); });

    EXPECT_NE(output.find("aicore exception"), std::string::npos);
    EXPECT_NE(output.find("device_id: 0"), std::string::npos);
    EXPECT_NE(output.find("stream_id: 456"), std::string::npos);
    EXPECT_NE(output.find("task_id: 123"), std::string::npos);
    EXPECT_NE(output.find("retcode: 507015"), std::string::npos);
    EXPECT_NE(output.find("kernelName: test_kernel"), std::string::npos);
    EXPECT_NE(output.find("PyPTO Internal Error"), std::string::npos);
}

TEST(DeviceErrorTrackingTest, InitializeErrorCallbackExecutesNormally)
{
    std::string output = CaptureStdout([&]() { InitializeErrorCallback(); });
    SUCCEED() << "InitializeErrorCallback executed normally";
}
