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
 * \file config_manager_ng.h
 * \brief
 */

#ifndef CONFIG_MANAGER_NG_H
#define CONFIG_MANAGER_NG_H

#include <map>
#include <memory>
#include <list>
#include <string>

#include "tilefwk/tilefwk.h"
#include "tilefwk/tile_shape.h"
#include "interface/operation/opcode.h"

namespace npu::tile_fwk {

// pass
constexpr const char* SG_PARALLEL_NUM = "pg_parallel_lower_bound";
constexpr const char* SG_PG_LOWER_BOUND = "pg_lower_bound";
constexpr const char* SG_PARTITION_ALGORITHM = "pg_partition_algorithm";
constexpr const char* SG_SET_SCOPE = "sg_set_scope";
constexpr const char* SG_SET_OOO_SCOPE = "sg_set_ooo_scope";
constexpr const char* OOO_SCHED_MODE = "ooo_sched_mode";
constexpr const char* OOO_SORT_MODE_AIC = "ooo_sort_mode_aic";
constexpr const char* OOO_SORT_MODE_AIV = "ooo_sort_mode_aiv";
constexpr const char* CUBE_L1_REUSE_SETTING = "cube_l1_reuse_setting";
constexpr const char* CUBE_NBUFFER_SETTING = "cube_nbuffer_setting";
constexpr const char* MG_COPYIN_UPPER_BOUND = "mg_copyin_upper_bound";
constexpr const char* OOO_PRESCHEDULE_METHOD = "ooo_preschedule_method";
constexpr const char* VEC_NBUFFER_SETTING = "vec_nbuffer_setting";
constexpr const char* SG_CUBE_PARALLEL_NUM = "sg_cube_parallel_num";
constexpr const char* MG_VEC_PARALLEL_LB = "mg_vec_parallel_lb";
constexpr const char* DB_TYPE = "db_type";
constexpr const char* COPYOUT_RESOLVE_COALESCING = "copyout_resolve_coalescing";
constexpr const char* AUTO_MIX_PARTITION = "auto_mix_partition";
constexpr const char* SG_SET_TUNEVF_MODE = "sg_set_tunevf_mode";
constexpr const char* ENABLE_SLICE = "enable_slice";

// runtime
constexpr const char* DEVICE_SCHED_MODE = "device_sched_mode";
constexpr const char* STITCH_FUNCTION_MAX_NUM = "stitch_function_max_num";
constexpr const char* MAX_WORKSPACE_KB = "max_workspace_kb";

constexpr const char* CFG_RUN_MODE = "run_mode";
constexpr const char* CFG_VALID_SHAPE_OPTIMIZE = "valid_shape_optimize";
constexpr int64_t CFG_RUN_MODE_NPU = 0;
constexpr int64_t CFG_RUN_MODE_SIM = 1;
constexpr const char* READY_ON_HOST_TENSORS = "ready_on_host_tensors";
constexpr const char* DEVICE_SCHED_PARALLELISM = "device_sched_parallelism";
constexpr const char* LAUNCH_SCHED_AICPU_NUM = "launch_sched_aicpu_num";
constexpr const char* LAUNCH_EARLY_MODE = "launch_early_mode";

// host
constexpr const char* COMPILE_STAGE = "compile_stage";
constexpr const char* COMPILE_MONITOR_ENABLE = "compile_monitor_enable";
constexpr const char* INTERVAL_SEC = "compile_monitor_print_interval";
constexpr const char* TIMEOUT_SEC = "compile_timeout_stage";
constexpr const char* TOTAL_TIMEOUT_SEC = "compile_timeout";
constexpr int64_t CS_ALL_COMPLETE = 0;
constexpr int64_t CS_TENSOR_GRAPH = 1;
constexpr int64_t CS_TILE_GRAPH = 2;
constexpr int64_t CS_EXECUTE_GRAPH = 3;
constexpr int64_t CS_CODEGEN_INSTRUCTION = 4;
constexpr int64_t CS_CODEGEN_BINARY = 5;

// codegen
constexpr const char* SUPPORT_DYNAMIC_ALIGNED = "support_dynamic_aligned";
constexpr const char* PLATFORM_SOC_VERSION = "soc_version";
constexpr const char* ENABLE_PMU_TRACE = "enable_pmu_trace";
constexpr const char* VF_OPTIONS = "vf_options";
constexpr const char* VF_COMPILER = "vf_compiler";

/* flow virifer tools KEYs */
const std::string KEY_ENABLE_PASS_VERIFY = "enable_pass_verify";
const std::string KEY_PASS_VERIFY_SAVE_TENSOR = "pass_verify_save_tensor";
const std::string KEY_PASS_VERIFY_SAVE_TENSOR_DIR = "pass_verify_save_tensor_dir";
const std::string KEY_PASS_VERIFY_FILTER = "pass_verify_pass_filter";
const std::string KEY_PASS_VERIFY_ERROR_TOL = "pass_verify_error_tol";

// debug
constexpr const char* CFG_COMPILE_DBEUG_MODE = "compile_debug_mode";
constexpr const char* CFG_RUNTIME_DBEUG_MODE = "runtime_debug_mode";
constexpr const char* CFG_DUMP_PASS_GRAPH = "dump_pass_graph";
constexpr int64_t CFG_DEBUG_NONE = 0;
constexpr int64_t CFG_DEBUG_ALL = 1;
constexpr int64_t CFG_COMPILE_FIXED_CCE = 2;
constexpr int64_t CFG_RUINTIME_DEBUG_AICORE_MODEL = 2;
constexpr int64_t CFG_RUNTIME_DEBUG_VERIFY = 3;
constexpr int64_t CFG_RUNTIME_DEBUG_GM_OUT_OF_BOUNDS = 4;

// operation
const std::string KEY_COMBINE_AXIS = "combine_axis";

class ConfigScope;
struct ConfigManagerImpl;
using ConfigScopePtr = std::shared_ptr<ConfigScope>;

class ConfigScope {
public:
    /**
     * \brief Get the config value with the specific key. throw runtime_error if
     * the key is not found.
     */
    const std::any& GetAnyConfig(const std::string& key) const;

    /**
     * \brief Returns a map of all configuration key-value pairs.
     */
    const std::map<std::string, std::any> GetAllConfig() const;

    /**
     * \brief Get the typed config value with the specific key.
     *
     */
    template <typename T>
    const T GetConfig(const std::string& key) const
    {
        return GetConfigAllType<T>(key);
    }

    /**
     * \brief Check if the config with the specific key exists.
     */
    bool HasConfig(const std::string& key) const;

    /**
     * \brief Get pass config (prefix: "pass.")
     */
    template <typename T>
    T GetPassConfig(const std::string& key) const
    {
        return GetConfigAllType<T>("pass." + key);
    }

    /**
     * \brief Get runtime config (prefix: "runtime.")
     */
    template <typename T>
    T GetRuntimeConfig(const std::string& key) const
    {
        return GetConfigAllType<T>("runtime." + key);
    }

    /**
     * \brief Get codegen config (prefix: "codegen.")
     */
    template <typename T>
    T GetCodegenConfig(const std::string& key) const
    {
        return GetConfigAllType<T>("codegen." + key);
    }

    /**
     * \brief Get host config (prefix: "host.")
     */
    template <typename T>
    T GetHostConfig(const std::string& key) const
    {
        return GetConfigAllType<T>("host." + key);
    }

    /**
     * \brief Get verify config (prefix: "verify.")
     */
    template <typename T>
    T GetVerifyConfig(const std::string& key) const
    {
        return GetConfigAllType<T>("verify." + key);
    }

    /**
     * \brief Get operation config (prefix: "operation.")
     */
    template <typename T>
    T GetOperationConfig(const std::string& key) const
    {
        return GetConfigAllType<T>("operation." + key);
    }

    /**
     * \brief Retrieves the CubeTile configuration.
     */
    CubeTile GetCubeTile() const
    {
        const std::any& value = GetAnyConfig("cube_tile_shapes");
        return pypto::AnyCast<CubeTile>(value);
    }

    /**
     * \brief Retrieves the ConvTile configuration.
     */
    ConvTile GetConvTile() const { return pypto::AnyCast<ConvTile>(GetAnyConfig("conv_tile_shapes")); }

    /**
     * \brief Retrieves the VecTile configuration as a VecTile structure.
     */
    VecTile GetVecTile() const
    {
        auto& value = GetAnyConfig("vec_tile_shapes");
        return {pypto::AnyCastRef<std::vector<int64_t>>(value)};
    }

    /**
     * \brief Retrieves the matrix size configuration as a vector of integers.
     */
    std::vector<int64_t> GetMatrixSize() const
    {
        auto& value = GetAnyConfig("matrix_size");
        return pypto::AnyCast<std::vector<int64_t>>(value);
    }

    /**
     * \brief Generate a TileShape object from the current configuration scope.
     */
    TileShape GenerateTileShape() const;

    /**
     * \brief Return the type of the config value with the specific key. type void
     * if the key is not found.
     */
    const std::type_info& Type(const std::string& key) const;

    /**
     * \brief Return all configures in current scope
     *
     * \return std::string
     */
    std::string ToString() const;

    /**
     * \brief Add or update a config value for the given key.
     * \param key The config key.
     * \param value The config value to set.
     */
    void AddValue(const std::string& key, std::any value);

    void UpdateValueWithAny(const std::string& key, std::any value);

    /**
     * \brief update a config value for the given key.
     * \param key The config key.
     * \param value The config value to set.
     */
    template <typename T>
    void UpdateValue(const std::string& key, T RawValue)
    {
        std::any value = ConvertToAny(RawValue);
        UpdateValueWithAny(key, value);
    }

    /**
     * \brief clear the config in Scope
     */
    void Clear();

    template <typename T>
    T GetConfigAllType(const std::string& key) const
    {
        if constexpr (std::is_same_v<T, bool>) {
            return AnyCast<bool>(GetAnyConfig(key));
        } else if constexpr (std::is_integral_v<T>) {
            int64_t tmp = AnyCast<int64_t>(GetAnyConfig(key));
            return static_cast<T>(tmp);
        } else {
            return AnyCast<T>(GetAnyConfig(key));
        }
    }

    ConfigScope(ConfigScopePtr parent);
    ~ConfigScope();

private:
    friend struct ConfigManagerImpl;
    std::shared_ptr<ConfigScope> Clone();

    std::shared_ptr<ConfigScope> parent_;
    std::list<ConfigScope*> children_;
    std::map<std::string, std::any> values_;

    std::string name_;
    std::string begin_file_;
    int begin_lino_{0};
    std::string end_file_;
    int end_lino_{0};

    template <typename T>
    std::any ConvertToAny(T value)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return value;
        } else if constexpr (std::is_integral_v<T>) {
            return static_cast<int64_t>(value);
        } else if constexpr (std::is_same_v<T, const char*>) {
            return std::string(value);
        } else {
            return value;
        }
    }
};

class ConfigManagerNg {
public:
    /**
     * \brief Begin a new scope with the given config values.
     *
     * \param values
     */
    void BeginScope(const std::string& name, std::map<std::string, std::any>&& values,
                    const char* file = __builtin_FILE(), int line = __builtin_LINE());

    /**
     * \brief End the current scope.
     *
     */
    void EndScope(const char* file = __builtin_FILE(), int line = __builtin_LINE());

    /**
     * @brief RAII guard: BeginScope on construction, EndScope on destruction.
     * Use instead of manual BeginScope + EndScope pair for exception safety.
     */
    class JitScopeGuard {
    public:
        JitScopeGuard(const std::string& name, std::map<std::string, std::any>&& values = {},
                      const char* file = __builtin_FILE(), int line = __builtin_LINE());
        ~JitScopeGuard();
        JitScopeGuard(const JitScopeGuard&) = delete;
        JitScopeGuard& operator=(const JitScopeGuard&) = delete;

    private:
        // RAII: ctor does BeginScope, dtor does EndScope
    };

    /**
     * @brief RAII guard: PushScope on construction, EndScope on destruction.
     * Use instead of manual Restore + EndScope pair for exception safety.
     */
    class ScopedRestore {
    public:
        explicit ScopedRestore(std::shared_ptr<ConfigScope> scope);
        ~ScopedRestore();
        ScopedRestore(const ScopedRestore&) = delete;
        ScopedRestore& operator=(const ScopedRestore&) = delete;

    private:
        // RAII: ctor does PushScope, dtor does EndScope
    };

    /**
     * @brief Set the Scope object
     * \brief Scope is not modifiable after it's begin, SetScope is just a syntax sugar for:
     * \code {.c}
     *   auto oldScope = CurrentScope();
     *   EndScope();
     *   BeginScope(oldScope.values + values);
     * \endcode
     *
     * @param values
     */
    void SetScope(std::map<std::string, std::any>&& values, const char* file = __builtin_FILE(),
                  int line = __builtin_LINE());

    /**
     * @brief Get the Current Scope object
     *
     * @return std::shared_ptr<ConfigScope>
     */
    static std::shared_ptr<ConfigScope> CurrentScope();

    /**
     * @brief Get the Global Scope object
     *
     * @return std::shared_ptr<ConfigScope>
     */
    static std::shared_ptr<ConfigScope> GlobalScope();

    /**
     * @brief change the currentScope
     */
    void PushScope(ConfigScopePtr scope);

    /**
     * @brief Get the type of the config value with the specific key. type void
     * if the key is not found.
     */
    const std::type_info& Type(const std::string& key) const;

    /**
     * @brief Get the range of the config value with the specific key.
     */
    const std::map<std::string, std::pair<int64_t, int64_t>>& Range() const;

    /**
     * \brief Check if the value is within the specified range.
     */
    bool IsWithinRange(const std::string& properties, std::any& value) const;

    static ConfigManagerNg& GetInstance();

    std::string GetOptionsTree();

    /**
     * @brief Get Global Config for cpp frontend
     *
     */
    template <typename T>
    static T GetGlobalConfig(const std::string& key)
    {
        return GetInstance().globalScope->GetConfig<T>("global." + key);
    }

    /**
     * @brief Set Global Config for cpp frontend
     *
     */
    template <typename T>
    static void SetGlobalConfig(const std::string& key, T value)
    {
        return GetInstance().globalScope->UpdateValue("global." + key, value);
    }

    /**
     * @brief Set Global Config for python frontend
     *
     */
    void SetGlobalConfig(std::map<std::string, std::any>&& values, const char* file, int lino);

    ~ConfigManagerNg();

private:
    ConfigManagerNg();
    ConfigManagerNg(const ConfigManagerNg&) = delete;
    ConfigManagerNg& operator=(const ConfigManagerNg&) = delete;

private:
    std::unique_ptr<ConfigManagerImpl> impl_;
    ConfigScopePtr globalScope;
};

namespace config {

std::shared_ptr<ConfigScope> Duplicate();

/**
 * @brief Get code generation configuration option
 */
template <typename T>
inline T GetCodeGenOption(const std::string& key)
{
    return ConfigManagerNg::CurrentScope()->GetConfigAllType<T>("codegen." + key);
}

/**
 * @brief Get pass configuration option
 */
template <typename T>
inline T GetPassOption(const std::string& key)
{
    return ConfigManagerNg::CurrentScope()->GetConfigAllType<T>("pass." + key);
}

inline bool EnableSlice() { return GetPassOption<bool>(ENABLE_SLICE); }

inline Opcode GetSliceOpcode() { return EnableSlice() ? Opcode::OP_SLICE : Opcode::OP_VIEW; }

inline Opcode GetContractOpcode() { return EnableSlice() ? Opcode::OP_CONTRACT : Opcode::OP_ASSEMBLE; }

/**
 * @brief Get runtime configuration option
 */
template <typename T>
inline T GetRuntimeOption(const std::string& key)
{
    return ConfigManagerNg::CurrentScope()->GetConfigAllType<T>("runtime." + key);
}

/**
 * @brief Get host configuration option
 */
template <typename T>
inline T GetHostOption(const std::string& key)
{
    return ConfigManagerNg::CurrentScope()->GetConfigAllType<T>("host." + key);
}

/**
 * @brief Get verification configuration option
 */
template <typename T>
inline T GetVerifyOption(const std::string& key)
{
    return ConfigManagerNg::CurrentScope()->GetConfigAllType<T>("verify." + key);
}

/**
 * @brief Get debug configuration option
 */
template <typename T>
inline T GetDebugOption(const std::string& key)
{
    return ConfigManagerNg::CurrentScope()->GetConfigAllType<T>("debug." + key);
}

inline bool IsRuntimeDebugAllEnabled() { return GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL; }

/**
 * @brief Get operation configuration option
 */
template <typename T>
inline T GetOperationOption(const std::string& key)
{
    return ConfigManagerNg::CurrentScope()->GetConfigAllType<T>("operation." + key);
}

} // namespace config

} // namespace npu::tile_fwk
#endif // CONFIG_MANAGER_NG_H
