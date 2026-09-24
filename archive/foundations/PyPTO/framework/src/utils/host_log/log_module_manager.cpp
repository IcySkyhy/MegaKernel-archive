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
 * \file log_module_manager.cpp
 * \brief
 */

#include "host_log/log_module_manager.h"

#include <string>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace npu::tile_fwk {
namespace {
constexpr int32_t kInvalidModuleLogLevel = -1;
constexpr const char* kEnvModuleLogLevel = "ASCEND_MODULE_LOG_LEVEL";
const std::map<std::string, LogModule> kLogModuleMap = {
    {"FUNCTION", LogModule::FUNCTION},       {"PASS", LogModule::PASS},
    {"CODEGEN", LogModule::CODEGEN},         {"MACHINE", LogModule::MACHINE},
    {"DISTRIBUTED", LogModule::DISTRIBUTED}, {"SIMULATION", LogModule::SIMULATION},
    {"VERIFY", LogModule::VERIFY},           {"COMPILER_MONITOR", LogModule::COMPILER_MONITOR},
    {"ADAPTER", LogModule::ADAPTER},         {"PLATFORM", LogModule::PLATFORM}};

inline bool IsLogLevelValid(const int32_t logLevel) { return logLevel >= DLOG_DEBUG && logLevel <= DLOG_ERROR; }

inline bool IsLogModuleValid(const LogModule logModule)
{
    return logModule >= LogModule::FUNCTION && logModule < LogModule::BOTTOM;
}

bool GetEnvStr(const char* envName, std::string& envValue)
{
    const size_t envValueMaxLen = 1024UL;
    const char* envStr = std::getenv(envName);
    if (envStr == nullptr || strnlen(envStr, envValueMaxLen) >= envValueMaxLen) {
        return false;
    }
    envValue = envStr;
    return true;
}

int ParseStrToInt(const std::string& str)
{
    try {
        return std::stoi(str);
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << std::endl;
    } catch (const std::out_of_range& oor) {
        std::cerr << "Out of Range error: " << oor.what() << std::endl;
    }
    return -1;
}

void Trim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isspace(c); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int c) { return !std::isspace(c); }).base(), s.end());
}

void ParseModuleLogLevel(const std::string& levelStr, std::map<std::string, int>& moduleLogLevel)
{
    if (levelStr.empty()) {
        return;
    }
    size_t lastPos = 0;
    size_t pos = levelStr.find(":");
    while (pos != std::string::npos) {
        std::string subLevelStr = levelStr.substr(lastPos, pos - lastPos);
        size_t subPos = subLevelStr.find("=");
        if (subPos != std::string::npos) {
            std::string subModuleName = subLevelStr.substr(0, subPos);
            Trim(subModuleName);
            moduleLogLevel.emplace(subModuleName, ParseStrToInt(subLevelStr.substr(subPos + 1)));
        }
        lastPos = pos + 1;
        pos = levelStr.find(":", lastPos);
    }
    std::string subLevelStr = levelStr.substr(lastPos);
    size_t subPos = subLevelStr.find("=");
    if (subPos != std::string::npos) {
        std::string subModuleName = subLevelStr.substr(0, subPos);
        Trim(subModuleName);
        moduleLogLevel.emplace(subModuleName, ParseStrToInt(subLevelStr.substr(subPos + 1)));
    }
}
} // namespace
LogModuleManager& LogModuleManager::Instance()
{
    static LogModuleManager logModuleManager;
    return logModuleManager;
}

LogModuleManager::LogModuleManager()
{
    moduleLogLevel_.fill(kInvalidModuleLogLevel);
    std::string moduleLogLevelStr;
    if (!GetEnvStr(kEnvModuleLogLevel, moduleLogLevelStr)) {
        return;
    }
    std::map<std::string, int> moduleLogLevelMap;
    ParseModuleLogLevel(moduleLogLevelStr, moduleLogLevelMap);
    for (const auto& item : moduleLogLevelMap) {
        auto iter = kLogModuleMap.find(item.first);
        if (iter == kLogModuleMap.end()) {
            continue;
        }
        if (!IsLogLevelValid(item.second)) {
            continue;
        }
        moduleLogLevel_[static_cast<size_t>(iter->second)] = item.second;
    }
}

LogModuleManager::~LogModuleManager() { moduleLogLevel_.fill(kInvalidModuleLogLevel); }

int32_t LogModuleManager::GetModuleLogLevel(const LogModule logModule) const
{
    return IsLogModuleValid(logModule) ? moduleLogLevel_[static_cast<size_t>(logModule)] : kInvalidModuleLogLevel;
}

int32_t LogModuleManager::GetLowestLogLevel() const
{
    int32_t lowestLogLevel = kInvalidModuleLogLevel;
    for (size_t i = 0; i < moduleLogLevel_.size(); ++i) {
        if (!IsLogLevelValid(moduleLogLevel_[i])) {
            continue;
        }
        if (lowestLogLevel < 0 || moduleLogLevel_[i] < lowestLogLevel) {
            lowestLogLevel = moduleLogLevel_[i];
        }
    }
    return lowestLogLevel;
}
} // namespace npu::tile_fwk
