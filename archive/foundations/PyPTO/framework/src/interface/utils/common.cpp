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
 * \file common.cpp
 * \brief
 */

#include "interface/utils/common.h"

#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <spawn.h>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <cstdlib>

namespace npu::tile_fwk {

std::string OperandTypeToStr(OperandType t)
{
    static std::map<OperandType, std::string> strMap = {
        {BUF_UB, "UB"},     {BUF_L1, "L1"},        {BUF_L0A, "L0A"},      {BUF_L0B, "L0B"},
        {BUF_L0C, "L0C"},   {BUF_FIX, "FIX"},      {BUF_DDR, "DDR"},      {BUF_REG, "REG"},
        {SCALAR, "SCALAR"}, {BUF_BT, "BiasTable"}, {BUF_L0AMX, "L0A_MX"}, {BUF_L0BMX, "L0B_MX"},
    };

    if (strMap.count(t)) {
        return strMap[t];
    }

    return "INVALID";
}

std::string SymbolicVecToStr(const std::vector<SymbolicScalar>& a)
{
    std::stringstream ss;
    ss << "[";

    if (!a.empty()) {
        ss << a[0].Dump();
        for (size_t i = 1; i < a.size(); ++i) {
            ss << ", " << a[i].Dump();
        }
    }

    ss << "]";
    return ss.str();
}

std::string ParamLocToStr(uint32_t loc)
{
    std::stringstream ss;
    auto type = loc >> ParamLocOffset;
    auto index = loc & ((1 << ParamLocOffset) - 1);
    ss << type << ":" << index << "_" << loc;
    return ss.str();
}

std::string Trim(const std::string& str)
{
    size_t start = str.find_first_not_of(" \t\n\r");
    size_t end = str.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

std::vector<std::string> SplitString(const std::string& str)
{
    std::vector<std::string> args;
    std::stringstream ss(str);
    std::string arg;
    while (ss >> arg) {
        args.push_back(arg);
    }
    return args;
}

std::string GetEnvVar(const std::string& varName, bool trim, bool toLower)
{
    const char* rawValue = std::getenv(varName.c_str());
    const size_t envVarMaxSize = 1024UL * 1024UL;
    if (rawValue == nullptr || (strnlen(rawValue, envVarMaxSize) >= envVarMaxSize)) {
        return "";
    }
    std::string value = rawValue;
    if (trim) {
        value = Trim(value);
    }
    if (toLower && !value.empty()) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    }
    return value;
}

uint64_t TimeStamp::CurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000000 + tv.tv_usec; // 1000000 is us per sec
}

std::string SafeExecCommandWithOutput(const std::vector<std::string>& args)
{
    if (args.empty()) {
        return "";
    }

    std::vector<char*> argv;
    for (const auto& a : args) {
        argv.push_back(const_cast<char*>(a.c_str()));
    }
    argv.push_back(nullptr);

    int pipefd[2];
    if (pipe(pipefd) == -1) {
        return "";
    }

    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return "";
    }

    auto cleanupOnError = [&]() {
        posix_spawn_file_actions_destroy(&actions);
        close(pipefd[0]);
        close(pipefd[1]);
    };

    if (posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO) != 0 ||
        posix_spawn_file_actions_addclose(&actions, pipefd[0]) != 0 ||
        posix_spawn_file_actions_addclose(&actions, pipefd[1]) != 0) {
        cleanupOnError();
        return "";
    }

    pid_t pid;
    int spawnRet = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), ::environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipefd[1]);

    if (spawnRet != 0) {
        close(pipefd[0]);
        return "";
    }

    std::string output;
    char buffer[4096];
    ssize_t bytesRead;
    while ((bytesRead = read(pipefd[0], buffer, sizeof(buffer))) > 0) {
        output.append(buffer, bytesRead);
    }
    close(pipefd[0]);

    int status;
    waitpid(pid, &status, 0);
    return output;
}

int SafeExecCommand(const std::vector<std::string>& args)
{
    if (args.empty()) {
        return -1;
    }

    std::vector<char*> argv;
    for (const auto& a : args) {
        argv.push_back(const_cast<char*>(a.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid;
    int spawnRet = posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), ::environ);
    if (spawnRet != 0) {
        return -1;
    }

    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
}

} // namespace npu::tile_fwk
