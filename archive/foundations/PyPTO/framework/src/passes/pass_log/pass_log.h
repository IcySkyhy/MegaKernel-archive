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
 * \file pass_log.h
 * \brief
 */

#ifndef PASS_LOG_H
#define PASS_LOG_H

#include <string>
#include <chrono>
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {

constexpr const char* kComputationGraphFolder = "computation_graph";

std::string GetFormatBacktrace(const Operation& op);

std::string GetFormatBacktrace(const OperationPtr& op);

std::string GetFormatBacktrace(const Operation* op);

std::string EscapeShellArg(const std::string& arg);

std::string BuildStrategyLogFolderName(const std::string& strategy, size_t strategyIndex);

void LogPassRuntime(const std::string& identifier, Program& program, Function& function,
                    const std::chrono::time_point<std::chrono::high_resolution_clock>& start);

void ExtractPassLogByFunction(const Function& function, const std::string& strategy);

enum class Elements { Operation, Tensor, Function, Graph, Config, Manager };

inline const char* toString(Elements elem)
{
    static const std::unordered_map<Elements, const char*> passElementName = {
        {Elements::Operation, "Operation"}, {Elements::Tensor, "Tensor"}, {Elements::Function, "Function"},
        {Elements::Graph, "Graph"},         {Elements::Config, "Config"}, {Elements::Manager, "Manager"}};

    auto it = passElementName.find(elem);
    return (it != passElementName.end()) ? it->second : "Unknown";
}

class ScopeTimer {
public:
    ScopeTimer(const char* moduleName, Elements opEnum, const char* tag)
        : module_(moduleName), opEnum_(opEnum), tag_(tag)
    {}

    void Start()
    {
        started_ = true;
        ended_ = false;
        start_ = std::chrono::steady_clock::now();
        PASS_LOGI("[%s][%s]: ==========> start %s.", module_, toString(opEnum_), tag_);
    }

    void End()
    {
        if (!started_ || ended_) {
            return;
        }
        ended_ = true;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_)
                      .count();
        PASS_LOGI("[%s][%s]: <========== end %s, cost time=%lld us.", module_, toString(opEnum_), tag_, (long long)us);
    }

    ~ScopeTimer()
    {
        if (started_ && !ended_) {
            End();
        }
    }

private:
    const char* module_;
    Elements opEnum_;
    const char* tag_;

    bool started_{false};
    bool ended_{false};
    std::chrono::steady_clock::time_point start_;
};

/**
 * @brief RAII utility for managing pass-specific log file redirection
 *
 * Redirects log output to a pass-specific file during construction and
 * restores the original log output on destruction. Automatically cleans
 * up empty log directories.
 */
class PassLogUtil {
public:
    PassLogUtil(Pass& pass, Function& function, size_t passIndex);
    PassLogUtil(Pass& pass, Function& function, const std::string& strategy, size_t passIndex);
    ~PassLogUtil();

    PassLogUtil(const PassLogUtil&) = delete;
    PassLogUtil& operator=(const PassLogUtil&) = delete;

private:
    std::string originLogOutPath_;
    std::string logFilePath_;
    std::string logFolder_;
    std::string strategyFolder_;
    std::string computationGraphFolder_;
};
} // namespace npu::tile_fwk

#define LOG_SCOPE_BEGIN(timerVar, opEnum, tag)     \
    ScopeTimer timerVar(MODULE_NAME, opEnum, tag); \
    timerVar.Start()

#define LOG_SCOPE_END(timerVar) timerVar.End()

#define APASS_LOG_DEBUG_F(opEnum, fmt, ...) \
    PYPTO_HOST_LOG(DLOG_DEBUG, PASS, "[%s.%s]:" fmt, MODULE_NAME, toString(opEnum), ##__VA_ARGS__)
#define APASS_LOG_INFO_F(opEnum, fmt, ...) \
    PYPTO_HOST_LOG(DLOG_INFO, PASS, "[%s.%s]:" fmt, MODULE_NAME, toString(opEnum), ##__VA_ARGS__)
#define APASS_LOG_WARN_F(opEnum, fmt, ...) \
    PYPTO_HOST_LOG(DLOG_WARN, PASS, "[%s.%s]:" fmt, MODULE_NAME, toString(opEnum), ##__VA_ARGS__)
#define APASS_LOG_ERROR_F(opEnum, fmt, ...)                                                                            \
    PYPTO_HOST_LOGE_WITH_ERRCODE(PASS, InternalError::PASS_INNER_ERROR, "[%s.%s]:" fmt, MODULE_NAME, toString(opEnum), \
                                 ##__VA_ARGS__)
#define APASS_LOG_ERROR_C(errCode, opEnum, fmt, ...) \
    PYPTO_HOST_LOGE_WITH_ERRCODE(PASS, errCode, "[%s.%s]:" fmt, MODULE_NAME, toString(opEnum), ##__VA_ARGS__)
#define APASS_LOG_EVENT_F(opEnum, fmt, ...) \
    PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, PASS, "[%s.%s]:" fmt, MODULE_NAME, toString(opEnum), ##__VA_ARGS__)

#endif // PASSES_LOG_H
