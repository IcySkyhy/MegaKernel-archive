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
 * \file config_manager.h
 * \brief
 */

#pragma once

#include <nlohmann/json.hpp>
#include <sys/file.h>
#include <string>
#include <unistd.h>
#include <type_traits>
#include <set>
#include "utils/file_utils.h"
#include "interface/configs/config_manager_ng.h"
#include "tilefwk/config.h"
#include "tilefwk/function.h"
#include "tilefwk/platform.h"
#include "interface/utils/common.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
using JsonExpcetion = nlohmann::json::exception;

/* Platform KEYs */
const std::string KEY_STD_LOG_LEVEL = "STD_LOG_LEVEL";
const std::string KEY_FILE_LOG_LEVEL = "FILE_LOG_LEVEL";
const std::string KEY_GRAPH_FILE_TYPE = "GRAPH_FILE_TYPE";
const std::string KEY_GRAPH_ONLY_DOT = "GRAPH_ONLY_DOT";
const std::string KEY_ENABLE_COST_MODEL = "enable_cost_model";
const std::string KEY_ENABLE_DYN_FULL_COST_MODEL = "ENABLE_DYN_FULL_COST_MODEL";
const std::string KEY_ENABLE_AIHAC_BACKEND = "enable_aihac_backend";
const std::string KEY_ENABLE_CHECKER = "enable_checker";
const std::string KEY_DUMP_SOURCE_LOCATION = "dump_source_location";
const std::string KEY_ENABLE_PROF_FUNC = "enable_prof_func";
const std::string KEY_ENABLE_PROF_AICORE_TIME = "enable_prof_aicore_time";
const std::string KEY_ENABLE_PROF_AICORE_PMU = "enable_prof_aicore_pmu";
const std::string KEY_ENABLE_DYN_COST_MODEL = "enable_dyn_cost_model";
const std::string KEY_TEST_IS_TIG = "test_is_tig";

/* Simulation KEYs */
const std::string KEY_BUILD_TASK_BASED_TOPO = "build_task_based_topo";
const std::string KEY_SIM_MODE = "sim_mode";
const std::string KEY_ACCURACY_LEVEL = "accuracy_level";
const std::string KEY_DEBUG_SINGLE_FUNC = "debug_single_func";
const std::string KEY_DEBUG_SINGLE_FUNCNAME = "leaf_function";
const std::string KEY_DRAW_FUNCTION_GRAPH = "draw_function_graph";
const std::string KEY_LOG_LEVEL = "log_level";
const std::string KEY_EXECUTE_CYCLE_THRESHOLD = "timeout_threshold";
const std::string KEY_JSON_PATH = "json_path";
const std::string KEY_AGENT_JSON_PATH = "agent_json_path";
const std::string KEY_ARGS = "args";

/* Host KEYs */
const std::string KEY_STRATEGY = "strategy";
const std::string KEY_ENABLE_BINARY_CACHE = "enable_binary_cache";

/* Pass KEYs */

const std::string KEY_PRINT_GRAPH = "print_graph";
const std::string KEY_PRINT_PROGRAM = "print_program";
const std::string KEY_DUMP_GRAPH = "dump_graph";
const std::string KEY_DUMP_PASS_TIME_COST = "dump_pass_time_cost";
const std::string KEY_PRE_CHECK = "pre_check";
const std::string KEY_POST_CHECK = "post_check";
const std::string KEY_EXPECTED_VALUE_CHECK = "expected_value_check";
const std::string KEY_DISABLE_PASS = "disable_pass";
const std::string KEY_HEALTH_CHECK = "health_check";
const std::string KEY_ADDR_CHECK = "addr_check";
const std::string KEY_RESUME_PARH = "RESUME_PATH";
const std::string KEY_EXEC_VERIFIER = "EXEC_VERIFIER";
const std::string KEY_SET_SCOPE = "SCOPE";
const std::string KEY_ENABLE_CV_FUSE = "enable_cv_fuse";
const std::string KEY_PASS_THREAD_NUM = "pass_thread_num";
const std::string KEY_VF_OPT_MARK_FOR = "vf_opt_mark_for";
const std::string KEY_ENABLE_VF = "enable_vf";

/* CodeGen KEYs */
const std::string KEY_PARALLEL_COMPILE = "parallel_compile";
const std::string KEY_FIXED_OUTPUT_PATH = "fixed_output_path"; // if true, dump cce to fixed directory (respects
                                                               // ASCEND_WORK_PATH/pypto)
const std::string KEY_FORCE_OVERWRITE = "force_overwrite";     // if true, don't dump cce when file exists
const std::string KEY_CODEGEN_SUPPORT_TILE_TENSOR = "codegen_support_tile_tensor"; // if true, gen code with layout mode
const std::string KEY_ENABLE_PMU_TRACE = "enable_pmu_trace";                       // if true, enable pmu trace

struct PassConfigs {
    bool printGraph{false};
    bool printProgram{false};
    bool dumpGraph{false};
    bool dumpPassTimeCost{false};
    bool preCheck{false};
    bool postCheck{false};
    bool expectedValueCheck{false};
    bool disablePass{false};
    bool healthCheck{false};
    bool addrCheck{false};
    std::string resumePath{""};
};

struct GlobalPassConfigs {
    bool enablePassConfigs{false};
    PassConfigs defaultPassConfigs;
};

template <typename T>
using ConvertedConfigType = std::conditional_t<std::is_constructible_v<std::string, T>, std::string, std::decay_t<T>>;

struct InternalGlobalConfig {
    std::string logTopFolder;
    std::string logTensorGraphFolder;
    std::string logFile;
};

class ConfigManager {
public:
    static ConfigManager& Instance();
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    Status Initialize();

    const GlobalPassConfigs& GetGlobalConfigs() const { return globalPassConfigs_; }
    PassConfigs GetPassConfigs(const std::string& strategy, const std::string& identifier) const;
    void PassConfigsDebugInfo(const std::string& strategy, const std::vector<std::string>& identifiers) const;

    const InternalGlobalConfig& GetInternalConfig() const { return globalConfigs_; }
    void SetInternalConfig(const InternalGlobalConfig& globalConfig) { globalConfigs_ = globalConfig; }

    template <typename T>
    auto GetPlatformConfig(const std::string& key, const T& defaultValue)
    {
        return GetConfig(json_, {"global", "platform", key}, defaultValue);
    }

    template <typename T>
    auto GetHostConfig(const std::string& key, const T& defaultValue)
    {
        return GetConfig(json_, {"global", "host", key}, defaultValue);
    }

    template <typename T>
    auto GetSimConfig(const std::string& key, const T& defaultValue)
    {
        return GetConfig(json_, {"global", "simulation", key}, defaultValue);
    }

    template <typename T>
    auto GetCodeGenConfig(const std::string& key, const T& defaultValue)
    {
        return GetConfig(json_, {"global", "codegen", key}, defaultValue);
    }

    template <typename T>
    auto GetPassConfig(const std::string& strategy, const std::string& identifier, const std::string& key,
                       const T& defaultValue)
    {
        return GetConfig(json_, {"global", "pass_strategies", strategy, identifier, key}, defaultValue);
    }

    template <typename T>
    auto GetPassDefaultConfig(const std::string& key, const T& defaultValue)
    {
        return GetConfig(json_, {"global", "pass", "default_pass_configs", key}, defaultValue);
    }

    template <typename T>
    auto SetPassDefaultConfig(const std::string& key, const T& value)
    {
        SetConfig(json_, {"global", "pass", "default_pass_configs", key}, value);
        RefreshGlobalPassCfg();
    }

    template <typename T>
    auto GetPassGlobalConfig(const std::string& key, const T& defaultValue)
    {
        return GetConfig(json_, {"global", "pass", key}, defaultValue);
    }

    template <typename T>
    auto SetPassGlobalConfig(const std::string& key, const T& value)
    {
        SetConfig(json_, {"global", "pass", key}, value);
        RefreshGlobalPassCfg();
    }

    template <typename T>
    void SetPlatformConfig(const std::string& key, const T& value)
    {
        SetConfig(json_, {"global", "platform", key}, value);
    }

    template <typename T>
    void SetHostConfig(const std::string& key, const T& value)
    {
        SetConfig(json_, {"global", "host", key}, value);
    }

    template <typename T>
    void SetSimConfig(const std::string& key, const T& value)
    {
        SetConfig(json_, {"global", "simulation", key}, value);
    }

    template <typename T>
    void SetPassConfig(const std::string& strategy, const std::string& identifier, const std::string& key,
                       const T& value)
    {
        SetConfig(json_, {"global", "pass_strategies", strategy, identifier, key}, value);
    }

    template <typename T>
    void SetCodeGenConfig(const std::string& key, const T& value)
    {
        SetConfig(json_, {"global", "codegen", key}, value);
    }

    const nlohmann::json* GetPrintOptions() { return GetJsonNode(json_, {"global", "tensor_print"}); }

    void Reset() { json_ = originJson_; }

    const nlohmann::json& GetJsonData() const { return json_; }
    void SetJsonData(const nlohmann::json& json) { json_ = json; }

    const std::string& LogTopFolder();
    const std::string& LogTensorGraphFolder();
    const std::string& LogFile();
    void ResetLog(const std::string& path = "");
    void SetLogRotationEnabled(bool enabled) { logRotationEnabled_ = enabled; }

private:
    GlobalPassConfigs globalPassConfigs_;
    InternalGlobalConfig globalConfigs_;
    nlohmann::json json_;
    nlohmann::json originJson_;
    bool logRotationEnabled_ = true;

    ConfigManager();

    ~ConfigManager() = default;

    static const nlohmann::json* GetJsonNode(const nlohmann::json& root, const std::vector<std::string>& keys);
    void RefreshGlobalPassCfg();

    template <typename T>
    static void SetConfig(nlohmann::json& root, const std::vector<std::string>& keys, const T& value)
    {
        auto* node = &root;
        for (auto&& key : keys) {
            node = &(*node)[key];
        }
        *node = value;
    }

    template <typename T>
    static ConvertedConfigType<T> GetConfig(const nlohmann::json& root, const std::vector<std::string>& keys,
                                            const T& defaultValue)
    {
        if (auto* node = GetJsonNode(root, keys)) {
            return node->get<ConvertedConfigType<T>>();
        }
        return defaultValue;
    }

    template <typename T>
    static auto GetChildConfig(const nlohmann::json& root, const std::string& key, const T& defaultValue)
    {
        return GetConfig(root, {key}, defaultValue);
    }
};

// config.h

/* Rundata KEYS */
constexpr const char* KEY_RUNTYPE = "runtype";
constexpr const char* KEY_PTO_CONFIG_FILE = "pto_config_file";
constexpr const char* KEY_COMPUTE_GRAPH_PATH = "compute_graph_path";
constexpr const char* KEY_AICPU_PERF_GRAPH_PATH = "aicpu_perf_path";
constexpr const char* KEY_SWIM_GRAPH_PATH = "swim_graph_path";
constexpr const char* KEY_VERIFY_PATH = "verify_path";
constexpr const char* KEY_PROGRAM_PATH = "program_file";

struct ConfigStorage;

struct PrintOptions {
    int edgeItems;
    int precision;
    int threshold;
    int linewidth;
};

struct SemanticLabel {
    std::string label;
    std::string filename;
    int lineno;

    SemanticLabel(const std::string& tlabel, const char* tfilename, int tlineno)
        : label(tlabel), filename(tfilename), lineno(tlineno)
    {}
    SemanticLabel(const std::string& tlabel, const std::string& tfilename, int tlineno)
        : label(tlabel), filename(tfilename), lineno(tlineno)
    {}
};

namespace config {
template <typename T>
auto GetPlatformConfig(const std::string& key, const T& defaultValue)
{
    return ConfigManager::Instance().GetPlatformConfig(key, defaultValue);
}

template <typename T>
auto GetHostConfig(const std::string& key, const T& defaultValue)
{
    return ConfigManager::Instance().GetHostConfig(key, defaultValue);
}

template <typename T>
auto GetPassConfig(const std::string& strategy, const std::string& identifier, const std::string& key,
                   const T& defaultValue)
{
    return ConfigManager::Instance().GetPassConfig(strategy, identifier, key, defaultValue);
}

template <typename T>
auto GetSimConfig(const std::string& key, const T& defaultValue)
{
    return ConfigManager::Instance().GetSimConfig(key, defaultValue);
}

template <typename T>
auto GetPassGlobalConfig(const std::string& key, const T& defaultValue)
{
    return ConfigManager::Instance().GetPassGlobalConfig(key, defaultValue);
}

template <typename T>
auto SetPassGlobalConfig(const std::string& key, const T& value)
{
    ConfigManager::Instance().SetPassGlobalConfig(key, value);
}

template <typename T>
auto GetPassDefaultConfig(const std::string& key, const T& defaultValue)
{
    return ConfigManager::Instance().GetPassDefaultConfig(key, defaultValue);
}

template <typename T>
auto SetPassDefaultConfig(const std::string& key, const T& value)
{
    ConfigManager::Instance().SetPassDefaultConfig(key, value);
}

template <typename T>
auto GetCodeGenConfig(const std::string& key, const T& value)
{
    return ConfigManager::Instance().GetCodeGenConfig(key, value);
}

template <typename T>
void SetPlatformConfig(const std::string& key, const T& value)
{
    ConfigManager::Instance().SetPlatformConfig(key, value);
}

template <typename T>
void SetHostConfig(const std::string& key, const T& value)
{
    ConfigManager::Instance().SetHostConfig(key, value);
}

template <typename T>
void SetPassConfig(const std::string& strategy, const std::string& identifier, const std::string& key, const T& value)
{
    ConfigManager::Instance().SetPassConfig(strategy, identifier, key, value);
}

template <typename T>
void SetSimConfig(const std::string& key, const T& value)
{
    ConfigManager::Instance().SetSimConfig(key, value);
}

template <typename T>
void SetCodeGenConfig(const std::string& key, const T& value)
{
    ConfigManager::Instance().SetCodeGenConfig(key, value);
}

inline DPlatform GetDevicePlatform()
{
    auto platform = ConfigManager::Instance().GetPlatformConfig("device_platform", "ASCEND_910B2");
    return StringToDpaltform(platform);
}

inline const std::string GetAbsoluteTopFolder() { return RealPath(ConfigManager::Instance().LogTopFolder()); }

inline const std::string& LogTopFolder() { return ConfigManager::Instance().LogTopFolder(); }

inline const std::string& LogTensorGraphFolder() { return ConfigManager::Instance().LogTensorGraphFolder(); }

inline const std::string& LogFile() { return ConfigManager::Instance().LogFile(); }

const std::string& OutputBaseDir();

std::string GetEmitPath(const std::string& name);

inline bool IsFixedCceMode()
{
    auto scope = ConfigManagerNg::CurrentScope();
    if (scope == nullptr) {
        return false;
    }
    std::string fullKey = std::string("debug.") + CFG_COMPILE_DBEUG_MODE;
    if (!scope->HasConfig(fullKey)) {
        return false;
    }
    return GetDebugOption<int64_t>(CFG_COMPILE_DBEUG_MODE) == CFG_COMPILE_FIXED_CCE;
}

inline Status SetPassStrategy(const std::string s)
{
    SetHostConfig(KEY_STRATEGY, s);
    return SUCCESS;
}

inline std::string GetPassStrategy() { return GetHostConfig(KEY_STRATEGY, "OOO"); }

inline bool UseTIG()
{
    return GetPassStrategy() == "TIG" || GetPassStrategy() == "PVC2_OOO" || GetPlatformConfig(KEY_TEST_IS_TIG, false) ||
           GetPassStrategy() == "DFS_OOO" || GetPassStrategy() == "BFS_DFS_OOO";
}

// config.h

FunctionType GetFunctionType();

std::shared_ptr<SemanticLabel> GetSemanticLabel();
void SetSemanticLabel(std::shared_ptr<SemanticLabel> label);

PrintOptions& GetPrintOptions();

void SetRunDataOption(const std::string& key, const std::string& value);

} // namespace config

} // namespace npu::tile_fwk
