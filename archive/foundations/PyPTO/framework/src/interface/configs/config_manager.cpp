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
 * \file config_manager.cpp
 * \brief
 */

#include "config_manager.h"
#include <map>
#include <fstream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <shared_mutex>
#include <limits.h>

#include "tilefwk/pypto_fwk_log.h"
#include "interface/utils/common.h"
#include "utils/file_utils.h"
#include "tilefwk/comm_group_recorder.h"
#include <unistd.h>
namespace npu::tile_fwk {

const std::string tilefwkConfigEnvName = "TILEFWK_CONFIG_PATH";
static PassConfigs InternalGetPassConfigs(const nlohmann::json& root, const GlobalPassConfigs* globalConfigs);
static GlobalPassConfigs InternalGetGlobalConfigs(const nlohmann::json& globalCfg);

static const nlohmann::json* GetJsonNode(const nlohmann::json& root, const std::vector<std::string>& keys)
{
    auto* curr = &root;
    for (auto&& key : keys) {
        if (auto it = curr->find(key); it != curr->end()) {
            curr = &*it;
        } else {
            return nullptr;
        }
    }
    return curr;
}

static const nlohmann::json* GetJsonChild(const nlohmann::json& root, const std::string& key)
{
    return GetJsonNode(root, {key});
}

ConfigManager::ConfigManager() { Initialize(); }

ConfigManager& ConfigManager::Instance()
{
    static ConfigManager instance;
    return instance;
}

const nlohmann::json* ConfigManager::GetJsonNode(const nlohmann::json& root, const std::vector<std::string>& keys)
{
    return ::npu::tile_fwk::GetJsonNode(root, keys);
}

Status ConfigManager::Initialize()
{
    /* 环境变量优先生效 */
    std::string jsonFilePath = GetEnvVar(tilefwkConfigEnvName);
    if (jsonFilePath.empty()) {
        jsonFilePath = RealPath(GetPyptoLibPath() + "/configs/tile_fwk_config.json");
    }

    config::SetRunDataOption(KEY_RUNTYPE, "npu");
    config::SetRunDataOption(KEY_PTO_CONFIG_FILE, jsonFilePath);
    FE_LOGI("Start to parse json_file %s", jsonFilePath.c_str());
    std::ifstream ifs(jsonFilePath);
    if (!ifs.is_open()) {
        FE_LOGE(FeError::INVALID_FILE, "Open file %s failed.", jsonFilePath.c_str());
        return FAILED;
    }
    ifs >> json_;
    ifs.close();

#ifdef SRCPATH
    constexpr const char* SRC_PATH = SRCPATH;
    // update Json_ through genJson
    std::string genJsonPath = std::string(SRC_PATH) + "/framework/src/cost_model/simulation/scripts/";
    if (IsPathExist(genJsonPath)) {
        auto files = GetFiles(genJsonPath, "json");
        if (!files.empty()) {
            for (const auto& file : files) {
                std::ifstream genJsonFile(genJsonPath + file);
                nlohmann::json jsonConfig = nlohmann::json::parse(genJsonFile);
                genJsonFile.close();

                if (jsonConfig.contains("global_configs")) {
                    const auto& genGlobal = jsonConfig["global_configs"];
                    if (genGlobal.contains("platform_configs") && !genGlobal["platform_configs"].empty()) {
                        json_["global"]["platform"].update(genGlobal["platform_configs"]);
                    }
                    if (genGlobal.contains("simulation_configs") && !genGlobal["simulation_configs"].empty()) {
                        json_["global"]["simulation"].update(genGlobal["simulation_configs"]);
                    }
                }
            }
        }
    }
#endif

    originJson_ = json_;

    if (auto* node = GetJsonNode(json_, {"global", "pass"})) {
        globalPassConfigs_ = InternalGetGlobalConfigs(*node);
    }

    return SUCCESS;
}

void ConfigManager::RefreshGlobalPassCfg()
{
    if (auto* node = GetJsonNode(json_, {"global", "pass"})) {
        globalPassConfigs_ = InternalGetGlobalConfigs(*node);
    }
}

static std::string GetHostName()
{
    char host[HOST_NAME_MAX] = {0};
    if (gethostname(host, sizeof(host)) == 0) {
        return std::string(host);
    }
    return "unknown-host";
}

constexpr int LIMIT_DIR_NUM_BEFORE_CREATE = 127;
constexpr const char* PREFIX_RUNDATA = "rundata_";
constexpr const char* PREFIX_OUTPUT = "output_";
constexpr const char* ENV_VAR_PYPTO_HOME = "PYPTO_HOME";
constexpr const char* ENV_VAR_HOME = "HOME";

static std::string CreateLogTopFolder()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;
    auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    constexpr int NUM_SIX = 6;
    timestamp << "_" << std::setw(NUM_SIX) << std::setfill('0') << us;

    std::string folderPath = config::OutputBaseDir();
    bool ret = CreateDir(folderPath);
    CHECK(FeError::BAD_FD, ret) << "Failed to create dir: " << folderPath << ", ensure its parent dir exists.";

    folderPath = folderPath + "/output_" + timestamp.str() + "_" + std::to_string(getpid()) + "_" + GetHostName();
    ret = CreateDir(folderPath);
    FE_ASSERT(FeError::BAD_FD, ret) << "Failed to create dir: " << folderPath << ", ensure its parent dir exists.";

    return folderPath;
}

const std::string& ConfigManager::LogTopFolder()
{
    if (globalConfigs_.logTopFolder.empty()) {
        globalConfigs_.logTopFolder = CreateLogTopFolder();
    }
    return globalConfigs_.logTopFolder;
}

const std::string& ConfigManager::LogTensorGraphFolder()
{
    if (globalConfigs_.logTensorGraphFolder.empty()) {
        globalConfigs_.logTensorGraphFolder = LogTopFolder() + "/TensorGraph";
        CreateDir(globalConfigs_.logTensorGraphFolder);
    }
    return globalConfigs_.logTensorGraphFolder;
}

const std::string& ConfigManager::LogFile()
{
    if (globalConfigs_.logFile.empty()) {
        globalConfigs_.logFile = LogTopFolder() + "/run.log";
    }
    return globalConfigs_.logFile;
}

void ConfigManager::ResetLog(const std::string& path)
{
    if (!logRotationEnabled_) {
        return;
    }
    std::string newLogFile;
    if (path.empty()) {
        globalConfigs_.logTopFolder = CreateLogTopFolder();
        newLogFile = globalConfigs_.logTopFolder + "/run.log";
    } else {
        newLogFile = path + "/run.log";
    }
    globalConfigs_.logFile = std::move(newLogFile);
    globalConfigs_.logTensorGraphFolder.clear();
}

PassConfigs ConfigManager::GetPassConfigs(const std::string& strategy, const std::string& identifier) const
{
    auto* node = GetJsonNode(json_, {"global", "pass_strategies", strategy, identifier});
    if (!node) {
        return globalPassConfigs_.defaultPassConfigs;
    }
    return InternalGetPassConfigs(*node, &globalPassConfigs_);
}

void ConfigManager::PassConfigsDebugInfo(const std::string& strategy, const std::vector<std::string>& identifiers) const
{
    auto* node = GetJsonNode(json_, {"global", "pass_strategies", strategy});
    if (!node) {
        FE_LOGI("[ConfigManager] Missing custom pass strategy < %s > configs. %s", strategy.c_str(),
                "You may add your own custom strategy configs in 'tile_fwk_config.json'.");
        return;
    }

    size_t maxLength = 0;
    for (auto&& identifier : identifiers) {
        maxLength = std::max(maxLength, identifier.size());
    }

    FE_LOGI("[ConfigManager] Strategy < %s > is found. Custom pass strategy will be used.", strategy.c_str());
    for (auto&& identifier : identifiers) {
        std::string spaces(maxLength - identifier.size(), ' ');
        if (node->find(identifier) != node->end()) {
            FE_LOGI("[ConfigManager] Pass instance %s<%s> configs loaded.", spaces.c_str(), identifier.c_str());
        } else {
            FE_LOGI("[ConfigManager] Pass instance %s<%s> configs for pass strategy <%s> is missing. \
            You may add your own custom strategy configs in 'tile_fwk_config.json'.",
                    spaces.c_str(), identifier.c_str(), strategy.c_str());
        }
    }
}

/* Helper Functions */
static std::map<std::string, std::function<void(PassConfigs&, const nlohmann::json&)>> g_assignPassConfigFns = {
    {KEY_PRINT_GRAPH, [](PassConfigs& configs, const nlohmann::json& node) { configs.printGraph = node.get<bool>(); }},
    {KEY_PRINT_PROGRAM,
     [](PassConfigs& configs, const nlohmann::json& node) { configs.printProgram = node.get<bool>(); }},
    {KEY_DUMP_GRAPH, [](PassConfigs& configs, const nlohmann::json& node) { configs.dumpGraph = node.get<bool>(); }},
    {KEY_DUMP_PASS_TIME_COST,
     [](PassConfigs& configs, const nlohmann::json& node) { configs.dumpPassTimeCost = node.get<bool>(); }},
    {KEY_PRE_CHECK, [](PassConfigs& configs, const nlohmann::json& node) { configs.preCheck = node.get<bool>(); }},
    {KEY_POST_CHECK, [](PassConfigs& configs, const nlohmann::json& node) { configs.postCheck = node.get<bool>(); }},
    {KEY_EXPECTED_VALUE_CHECK,
     [](PassConfigs& configs, const nlohmann::json& node) { configs.expectedValueCheck = node.get<bool>(); }},
    {KEY_DISABLE_PASS,
     [](PassConfigs& configs, const nlohmann::json& node) { configs.disablePass = node.get<bool>(); }},
    {KEY_HEALTH_CHECK,
     [](PassConfigs& configs, const nlohmann::json& node) { configs.healthCheck = node.get<bool>(); }},
    {KEY_ADDR_CHECK, [](PassConfigs& configs, const nlohmann::json& node) { configs.addrCheck = node.get<bool>(); }},
    {KEY_RESUME_PARH,
     [](PassConfigs& configs, const nlohmann::json& node) { configs.resumePath = node.get<std::string>(); }},
};

static PassConfigs InternalGetPassConfigs(const nlohmann::json& root, const GlobalPassConfigs* globalConfigs)
{
    PassConfigs configs;
    if (globalConfigs != nullptr) {
        if (!globalConfigs->enablePassConfigs) {
            return {};
        }
        configs = globalConfigs->defaultPassConfigs;
    }

    for (auto&& [key, assignFn] : g_assignPassConfigFns) {
        if (auto* node = GetJsonChild(root, key)) {
            assignFn(configs, *node);
        }
    }
    return configs;
}

static std::map<std::string, std::function<void(GlobalPassConfigs&, const nlohmann::json&)>> g_assignGlobalConfigFns = {
    {"enable_pass_configs",
     [](GlobalPassConfigs& configs, const nlohmann::json& node) { configs.enablePassConfigs = node.get<bool>(); }},
    {"default_pass_configs",
     [](GlobalPassConfigs& configs, const nlohmann::json& node) {
         configs.defaultPassConfigs = InternalGetPassConfigs(node, nullptr);
     }},
};

static GlobalPassConfigs InternalGetGlobalConfigs(const nlohmann::json& globalCfg)
{
    GlobalPassConfigs configs;
    for (auto&& [key, assignFn] : g_assignGlobalConfigFns) {
        if (auto* node = GetJsonChild(globalCfg, key)) {
            assignFn(configs, *node);
        }
    }
    return configs;
}

struct RunDataDir {
    std::string path;
    std::string dName;

    std::string montage() { return path + "/" + dName; }

    bool empty() { return (path.empty() || dName.empty()); }

    void Reset()
    {
        path.clear();
        dName.clear();
    }
};

struct ConfigStorage {
    ConfigStorage() { Init(); }

    void Init()
    {
        auto res = ConfigManager::Instance().GetPrintOptions();
        if (res != nullptr && res->is_object()) {
            printOption.edgeItems = res->value("edgeitems", printOption.edgeItems);
            printOption.precision = res->value("precision", printOption.precision);
            printOption.threshold = res->value("threshold", printOption.threshold);
            printOption.linewidth = res->value("linewidth", printOption.linewidth);
        }
        Reset();
    }

    void Reset()
    {
        funcType = FunctionType::DYNAMIC;
        semanticLabel = nullptr;
        rundataDir.Reset();
    }

    FunctionType funcType;
    std::shared_ptr<SemanticLabel> semanticLabel;
    RunDataDir rundataDir;
    PrintOptions printOption;
};

namespace config {

static ConfigStorage g_config;
std::shared_mutex g_rwlock;

void Reset()
{
    g_config.Reset();
    ConfigManagerNg::CurrentScope()->Clear();
}

void SetBuildStatic(bool isStatic)
{
    g_config.funcType = isStatic ? FunctionType::STATIC : FunctionType::DYNAMIC;
    FE_LOGD("Set functionType[%s] successfully.", (isStatic ? "STATIC" : "DYNAMIC"));
}

FunctionType GetFunctionType() { return g_config.funcType; }

void SetSemanticLabel(const std::string& label, const char* filename, int lineno)
{
    g_config.semanticLabel = std::make_shared<SemanticLabel>(label, filename, lineno);
    FE_LOGD("Set semanticLabel[%s] successfully.", label.c_str());
}

void SetSemanticLabel(std::shared_ptr<SemanticLabel> label) { g_config.semanticLabel = label; }

std::shared_ptr<SemanticLabel> GetSemanticLabel() { return g_config.semanticLabel; }

void CreateRunDataDir()
{
    std::string envStr = GetEnvVar(ENV_VAR_PYPTO_HOME);
    std::string dir = envStr.empty() ? (GetEnvVar(ENV_VAR_HOME) + "/.pypto") : envStr;
    g_config.rundataDir.path = dir + "/run";
    RemoveOldDirectories(g_config.rundataDir.path, PREFIX_RUNDATA, LIMIT_DIR_NUM_BEFORE_CREATE);
    auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y%m%d%H%M%S");
    g_config.rundataDir.dName = PREFIX_RUNDATA + timestamp.str();
    bool res = CreateDir(g_config.rundataDir.montage(), true);
    FE_ASSERT(FeError::BAD_FD, res) << "Failed to create directory: " << g_config.rundataDir.montage();
}

void SetRunDataOption(const std::string& key, const std::string& value)
{
    static nlohmann::json j;
    std::shared_lock lock(g_rwlock);
    j[key] = value;
    auto dumpValue = j.dump(2);
    if (g_config.rundataDir.empty()) {
        CreateRunDataDir();
    }
    auto filename = g_config.rundataDir.montage() + "/rundata.json";
    SaveFileSafe(filename, dumpValue);
}

void SetPrintOptions(int edgeItems, int precision, int threshold, int linewidth)
{
    g_config.printOption.edgeItems = edgeItems;
    g_config.printOption.precision = precision;
    g_config.printOption.threshold = threshold;
    g_config.printOption.linewidth = linewidth;
    FE_LOGD("Set print option [%d %d %d %d] successfully.", edgeItems, precision, threshold, linewidth);
}

PrintOptions& GetPrintOptions() { return g_config.printOption; }

const std::string& OutputBaseDir()
{
    static std::string baseDir;
    static std::once_flag flag;
    std::call_once(flag, []() {
        const char* tileFwkDir = std::getenv("TILE_FWK_OUTPUT_DIR");
        if (tileFwkDir != nullptr && strlen(tileFwkDir) > 0) {
            FE_LOGD("Get env TILE_FWK_OUTPUT_DIR[%s] successfully.", tileFwkDir);
            baseDir = tileFwkDir;
        } else {
            const char* ascendWorkPath = std::getenv("ASCEND_WORK_PATH");
            if (ascendWorkPath != nullptr && strlen(ascendWorkPath) > 0) {
                baseDir = std::string(ascendWorkPath) + "/pypto";
            } else {
                baseDir = "output";
            }
        }
    });
    return baseDir;
}

std::string GetEmitPath(const std::string& name)
{
    std::string dirPath;
    if (ConfigManager::Instance().GetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, false) || IsFixedCceMode()) {
        std::string rootDir;
        const char* ascendWorkPath = std::getenv("ASCEND_WORK_PATH");
        if (ascendWorkPath != nullptr && strlen(ascendWorkPath) > 0) {
            rootDir = std::string(ascendWorkPath) + "/pypto";
        }
        std::vector<std::string> groupNames = Distributed::CommGroupRecorder::GetInstance().Output();
        if (groupNames.size() == 0) {
            dirPath = rootDir.empty() ? name : (rootDir + "/" + name);
        } else {
            const char* rankId = std::getenv("TILE_FWK_DEVICE_ID");
            if (rootDir.empty()) {
                dirPath = std::string(rankId) + "/" + name;
            } else {
                dirPath = rootDir + "/" + std::string(rankId) + "/" + name;
            }
        }
    } else {
        dirPath = LogTopFolder() + "/" + name;
    }
    return dirPath;
}

} // namespace config
} // namespace npu::tile_fwk
