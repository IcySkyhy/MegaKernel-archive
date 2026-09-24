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
 * \file simulation_platform.cpp
 * \brief
 */

#include <climits>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include "simulation_platform.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {
const std::string PLATFORM_INFO_RELATIVE_PATH = "/configs/";
const std::string FWK_CONFIG_RELATIVE_PATH = "tile_fwk_config.json";
const std::string DEFAULT_SOC_VERSION = "A2A3";
const std::string INI_EXTENSION = ".ini";
const uint32_t PLATFORM_FAILED = 0xFFFFFFFF;
const uint32_t PLATFORM_SUCCESS = 0;

namespace {
std::string DPlatformToSocVersion(DPlatform platform)
{
    static const std::unordered_map<DPlatform, std::string> mappings = {
        {DPlatform::ASCEND_910B1, DEFAULT_SOC_VERSION},
        {DPlatform::ASCEND_910B2, DEFAULT_SOC_VERSION},
        {DPlatform::ASCEND_910B3, DEFAULT_SOC_VERSION},
        {DPlatform::ASCEND_910B4, DEFAULT_SOC_VERSION},
        {DPlatform::ASCEND_910C4, "910PR_9363"},
        {DPlatform::ASCEND_950PR_9579, "950PR_957x"},
        {DPlatform::ASCEND_950DT_9581, "950DT_9581"},
        {DPlatform::ASCEND_950DT_9581X, "950DT_9581x"},
        {DPlatform::ASCEND_950DT_9582, "950DT_958x"},
        {DPlatform::ASCEND_950PR_9582, "950PR_958x"},
        {DPlatform::ASCEND_950DT_9579, "950DT_957x"},
        {DPlatform::KIRIN_9030, "Kirin9030"},
        {DPlatform::KIRIN_X90, "KirinX90"},
    };
    auto it = mappings.find(platform);
    if (it != mappings.end()) {
        return it->second;
    }
    return DEFAULT_SOC_VERSION;
}
} // namespace

std::string SimulationPlatform::GetDevicePlatform()
{
    const std::string configPath = RealPath(GetPyptoLibPath() + PLATFORM_INFO_RELATIVE_PATH + FWK_CONFIG_RELATIVE_PATH);
    if (configPath.empty()) {
        PLATFORM_LOGW("Failed to open tile_fwk_config.json, use default platform.");
        return DEFAULT_SOC_VERSION;
    }
    std::ifstream jsonFile(configPath);
    nlohmann::json jsonData = nlohmann::json::parse(jsonFile);
    jsonFile.close();

    std::string platformStr;
    if (jsonData.contains("global") && jsonData["global"].contains("platform") &&
        jsonData["global"]["platform"].contains("device_platform") &&
        jsonData["global"]["platform"]["device_platform"].is_string()) {
        platformStr = jsonData["global"]["platform"]["device_platform"].get<std::string>();
        PLATFORM_LOGD("Key 'global.platform.device_platform' specified SoC version:%s.", platformStr.c_str());
    } else {
        PLATFORM_LOGW("Key 'global.platform.device_platform' not found in %s, use default platform.",
                      configPath.c_str());
    }

    DPlatform platform = StringToDpaltform(platformStr);
    return DPlatformToSocVersion(platform);
}

bool SimulationPlatform::GetCostModelPlatformRealPath(const std::string& socVersion, std::string& realPath)
{
    std::string platformSocVersion = socVersion;
    if (platformSocVersion.empty()) {
        platformSocVersion = GetDevicePlatform();
        PLATFORM_LOGD("Config specified SoC version:%s.", platformSocVersion.c_str());
    }
    realPath = RealPath(GetPyptoLibPath() + PLATFORM_INFO_RELATIVE_PATH + platformSocVersion + INI_EXTENSION);
    if (realPath.empty()) {
        return false;
    }
    return true;
}
} // namespace tile_fwk
} // namespace npu
