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
 * \file interpreter_log_test_utils.h
 * \brief Common helpers for interpreter/log related unit tests:
 *        - CaptureLogFileAndEcho: capture log output from log files (落盘形式，与 LogManager 路径一致)
 *        - VerifyLogContainsFailed: check VERIFY log failures
 *        - VerifyLogContainsIndex0Failed: check VERIFY index 0 failures
 */

#pragma once

#include <functional>
#include <string>
#include <cstdio>
#include <regex>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <unistd.h>

#include "interface/interpreter/interpreter_log.h"

static constexpr const char* kInterpLogTestOutputRoot = "build/output/bin/output";

static inline std::string InterpLogTestFindLatestInterpreterLog()
{
    std::string latestPath;
    time_t latestMTime = 0;

    DIR* rootDir = opendir(kInterpLogTestOutputRoot);
    if (rootDir == nullptr) {
        return "";
    }

    struct dirent* outputEntry = nullptr;
    while ((outputEntry = readdir(rootDir)) != nullptr) {
        std::string outputName = outputEntry->d_name;
        if (outputName == "." || outputName == ".." || outputName.find("output_") != 0) {
            continue;
        }

        std::string outputPath = std::string(kInterpLogTestOutputRoot) + "/" + outputName;
        struct stat outputSt;
        if (stat(outputPath.c_str(), &outputSt) != 0 || !S_ISDIR(outputSt.st_mode)) {
            continue;
        }

        DIR* verifyDir = opendir(outputPath.c_str());
        if (verifyDir == nullptr) {
            continue;
        }

        struct dirent* verifyEntry = nullptr;
        while ((verifyEntry = readdir(verifyDir)) != nullptr) {
            std::string verifyName = verifyEntry->d_name;
            if (verifyName == "." || verifyName == ".." || verifyName.find("verify_") != 0) {
                continue;
            }
            std::string logPath = outputPath + "/" + verifyName + "/interpreter.log";
            struct stat logSt;
            if (stat(logPath.c_str(), &logSt) != 0 || !S_ISREG(logSt.st_mode)) {
                continue;
            }
            if (logSt.st_mtime >= latestMTime) {
                latestMTime = logSt.st_mtime;
                latestPath = logPath;
            }
        }
        closedir(verifyDir);
    }
    closedir(rootDir);

    return latestPath;
}

// 读取最新 verify 目录中的 interpreter.log（整文件内容）
static inline std::string CaptureLogFileAndEcho(std::function<void()> func)
{
    func();

    // 优先读取 interpreter_log 当前配置路径，避免依赖 UT 执行时 cwd
    const std::string configuredPath = npu::tile_fwk::interpreter::LogFilePath();
    {
        std::ifstream ifs(configuredPath, std::ios::binary);
        if (ifs) {
            std::string captured;
            captured.append(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
            return captured;
        }
    }

    const std::string fallbackPath = InterpLogTestFindLatestInterpreterLog();
    if (fallbackPath.empty()) {
        return "";
    }
    std::ifstream ifs(fallbackPath, std::ios::binary);
    if (!ifs) {
        return "";
    }
    std::string captured;
    captured.append(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    return captured;
}

// 捕获 stdout 输出，用于校验日志内容
static inline std::string CaptureStdoutAndEcho(std::function<void()> func)
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

    char buffer[8192] = {0};
    ssize_t len = read(pipefd[0], buffer, sizeof(buffer) - 1);
    close(pipefd[0]);

    std::string captured(len > 0 ? static_cast<size_t>(len) : 0, '\0');
    if (len > 0) {
        captured.assign(buffer, static_cast<size_t>(len));
        // 同时打印到控制台，便于调试查看
        ssize_t written = write(old_stdout, buffer, static_cast<size_t>(len));
        (void)written;
    }
    close(old_stdout);

    return captured;
}

// 检查日志中是否出现 [ERROR] 级别记录，任何 ERROR 均视为用例失败
inline bool VerifyLogContainsFailed(const std::string& logOutput)
{
    static const std::regex kErrorPattern(R"(\[ERROR\])");
    return std::regex_search(logOutput, kErrorPattern);
}

// 仅检查 [VERIFY] 日志行中 index 0 是否出现 FAILED，用于 Topk 用例
inline bool VerifyLogContainsIndex0Failed(const std::string& logOutput)
{
    // 只关心 flow_verifier 打印的 index 0 结果行：
    // "... [VERIFY]: ... Verify for ... index 0 result FAILED"
    static const std::regex kVerifyIndex0FailedPattern(R"(\[VERIFY][^\n]*index 0 result FAILED)");
    return std::regex_search(logOutput, kVerifyIndex0FailedPattern);
}
