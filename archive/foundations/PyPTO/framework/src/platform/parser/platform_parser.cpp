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
 * \file platform_parser.cpp
 * \brief
 */
#include <fstream>

#include "platform_parser.h"
#include "tilefwk/error_code.h"
#include "simulation_platform/simulation_platform.h"

namespace npu {
namespace tile_fwk {
const std::string platformConfigEnv = "PLATFORM_CONFIG_PATH";
const std::string version = "version";
const std::string aic = "AIC";
const std::string aiv = "AIV";
const std::string ccecAicVersion = "CCEC_AIC_version";
const std::string ccecAivVersion = "CCEC_AIV_version";
const std::string ccecCubeVersion = "CCEC_CUBE_version";
const std::string ccecVectorVersion = "CCEC_VECTOR_version";

bool PlatformParser::FilterCCECVersion(const std::string& key, std::string& coreType) const
{
    const std::string prefix = "CCEC_";
    const std::string suffix = "_version";
    const size_t prefixLen = prefix.length();
    const size_t suffixLen = suffix.length();
    if (key.length() >= (prefixLen + suffixLen) && key.substr(0, prefixLen) == prefix &&
        key.substr(key.length() - suffixLen) == suffix) {
        coreType = key.substr(prefixLen, key.length() - prefixLen - suffixLen);
        return true;
    } else {
        return false;
    }
}

bool PlatformParser::GetSizeVal(const std::string& column, const std::string& key, size_t& val) const
{
    std::string valStr;
    const size_t max_size_t = std::numeric_limits<size_t>::max();
    if (!GetStringVal(column, key, valStr)) {
        return false;
    }
    val = 0UL;

    constexpr int kRadix10 = 10;
    constexpr int kMaxDigit10 = kRadix10 - 1;

    for (const char& c : valStr) {
        int digit = c - '0';
        if (digit < 0 || digit > kMaxDigit10) {
            return false;
        }
        if (val > (max_size_t - digit) / kRadix10) {
            return false;
        }
        val = val * kRadix10 + digit;
    }
    return true;
}

bool PlatformParser::GetCCECVersion(std::unordered_map<std::string, std::string>& ccecVersion) const
{
    const std::vector<std::string> ccecVersions = {ccecAicVersion, ccecAivVersion, ccecCubeVersion, ccecVectorVersion};
    ccecVersion.clear();
    std::string coreType;
    std::string versionVal;
    for (const auto& curVersion : ccecVersions) {
        if (FilterCCECVersion(curVersion, coreType) && GetStringVal(version, curVersion, versionVal)) {
            ccecVersion[coreType] = versionVal;
        }
    }
    return !ccecVersion.empty();
}

INIParser::INIParser() { INIParserInit(""); }

INIParser::INIParser(const std::string& socVersion) { INIParserInit(socVersion); }

void INIParser::INIParserInit(const std::string& socVersion)
{
    std::string srcPath;
    SimulationPlatform simulationPlatform;
    simulationPlatform.GetCostModelPlatformRealPath(socVersion, srcPath);
    PLATFORM_LOGD("Try to initiate the ini parser.");
    if (!Initialize(srcPath)) {
        throw std::runtime_error("can not open simulation file: " + srcPath);
    }
}

bool INIParser::Initialize(const std::string& iniFilePath)
{
    PLATFORM_LOGI("Start to parse ini_file %s.", iniFilePath.c_str());
    if (!ReadINIFile(iniFilePath)) {
        PLATFORM_LOGE(FeError::NOT_EXIST, "ReadINIFile failed.");
        return false;
    }
    PLATFORM_LOGD("Parse ini_file %s successfully.", iniFilePath.c_str());
    return true;
}

bool INIParser::ReadINIFile(const std::string& filepath)
{
    data_.clear();
    std::ifstream file(filepath);
    PLATFORM_LOGD("Try to open ini file: %s.", filepath.c_str());
    if (!file.is_open()) {
        PLATFORM_LOGE(FeError::BAD_FD, "Failed to open ini file: %s.", filepath.c_str());
        return false;
    }
    std::string line;
    std::string section;
    while (std::getline(file, line)) {
        TrimLine(line);
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            constexpr size_t kLeftBracketLen = std::char_traits<char>::length("[");
            constexpr size_t kRightBracketLen = std::char_traits<char>::length("]");
            if (line.size() <= kLeftBracketLen + kRightBracketLen) {
                continue;
            }
            section = line.substr(kLeftBracketLen, line.size() - kLeftBracketLen - kRightBracketLen);
            continue;
        }
        size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) {
            PLATFORM_LOGW("Illegal ini format[%s].", line.c_str());
            continue;
        }
        std::string key = line.substr(0, equalPos);
        std::string value = line.substr(equalPos + 1);
        if (key.empty()) {
            PLATFORM_LOGW("Empty attribute[%s].", line.c_str());
            continue;
        }
        data_[section][key] = value;
    }
    file.close();
    return true;
}

bool INIParser::GetStringVal(const std::string& column, const std::string& key, std::string& val) const
{
    PLATFORM_LOGD("Try to obtain value from column[%s] and key[%s] throughs ini file.", column.c_str(), key.c_str());
    val.clear();
    auto iter = data_.find(column);
    if (iter == data_.end()) {
        PLATFORM_LOGE(FeError::INVALID_TYPE, "Cannot find attr '%s' from the ini file.", column.c_str());
        return false;
    }
    auto value = iter->second;
    auto iter2 = value.find(key);
    if (iter2 == value.end()) {
        PLATFORM_LOGE(FeError::INVALID_VAL, "Cannot find attr '%s' from the [%s] tab.", key.c_str(), column.c_str());
        return false;
    }
    val = iter2->second;
    PLATFORM_LOGD("Value[%s][%s] = %s.", column.c_str(), key.c_str(), val.c_str());
    return true;
}

bool CmdParser::GetStringVal(const std::string& column, const std::string& key, std::string& val) const
{
    val.clear();
    if (!CannHostRuntime::Instance().GetSocSpec(column, key, val)) {
        PLATFORM_LOGE(FeError::UNKNOWN, "Cannot find soc spec '%s' from the [%s] column.", key.c_str(), column.c_str());
        return false;
    }
    return true;
}
} // namespace tile_fwk
} // namespace npu
