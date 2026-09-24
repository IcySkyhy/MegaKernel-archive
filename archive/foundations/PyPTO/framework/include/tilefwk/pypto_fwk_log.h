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
 * \file pypto_fwk_log.h
 * \brief
 */

#pragma once

#include "tilefwk/tilefwk_log.h"
#include "tilefwk/error_code.h"
#include "tilefwk/error_manager.h"
#include "tilefwk/error.h"

#ifndef __DEVICE__

#define PYPTO_HOST_LOGE_WITH_ERRCODE(module, errCode, fmt, ...)                                                   \
    PYPTO_HOST_LOG(DLOG_ERROR, module, "ErrCode: F%05X! Enum: %s " fmt, static_cast<uint32_t>(errCode) & 0xFFFFF, \
                   #errCode, ##__VA_ARGS__);                                                                      \
    REPORT_ERROR_MSG(errCode, fmt, ##__VA_ARGS__)

#define PYPTO_SIM_LOGE_WITH_ERRCODE(module, errCode, fmt, ...)                                                   \
    PYPTO_SIM_LOG(DLOG_ERROR, module, "ErrCode: F%05X! Enum: %s " fmt, static_cast<uint32_t>(errCode) & 0xFFFFF, \
                  #errCode, ##__VA_ARGS__);                                                                      \
    REPORT_ERROR_MSG(errCode, fmt, ##__VA_ARGS__)

#define FE_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, FUNCTION, __VA_ARGS__)
#define FE_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, FUNCTION, __VA_ARGS__)
#define FE_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, FUNCTION, __VA_ARGS__)
#define FE_LOGE(errCode, fmt, ...) PYPTO_HOST_LOGE_WITH_ERRCODE(FUNCTION, errCode, fmt, ##__VA_ARGS__)
#define FE_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, FUNCTION, __VA_ARGS__)
#define FE_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, FUNCTION, __VA_ARGS__)

#define PASS_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, PASS, __VA_ARGS__)
#define PASS_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, PASS, __VA_ARGS__)
#define PASS_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, PASS, __VA_ARGS__)
#define PASS_LOGE(errCode, fmt, ...) PYPTO_HOST_LOGE_WITH_ERRCODE(PASS, errCode, fmt, ##__VA_ARGS__)
#define PASS_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, PASS, __VA_ARGS__)
#define PASS_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, PASS, __VA_ARGS__)

#define CODEGEN_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGE(errCode, fmt, ...) PYPTO_HOST_LOGE_WITH_ERRCODE(CODEGEN, errCode, fmt, ##__VA_ARGS__)
#define CODEGEN_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGI_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_INFO, CODEGEN, __VA_ARGS__)

#define MACHINE_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, MACHINE, __VA_ARGS__)
#define MACHINE_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, MACHINE, __VA_ARGS__)
#define MACHINE_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, MACHINE, __VA_ARGS__)
#define MACHINE_LOGE(errCode, fmt, ...)                                     \
    do {                                                                    \
        PYPTO_HOST_LOGE_WITH_ERRCODE(MACHINE, errCode, fmt, ##__VA_ARGS__); \
        MACHINE_ASSERT(false);                                              \
    } while (0)
#define MACHINE_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, MACHINE, __VA_ARGS__)
#define MACHINE_LOGI_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_INFO, MACHINE, __VA_ARGS__)
#define MACHINE_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, MACHINE, __VA_ARGS__)

#define DISTRIBUTED_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGE(errCode, fmt, ...) PYPTO_HOST_LOGE_WITH_ERRCODE(DISTRIBUTED, errCode, fmt, ##__VA_ARGS__)
#define DISTRIBUTED_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, DISTRIBUTED, __VA_ARGS__)

#define SIMULATION_LOGD(...) PYPTO_SIM_LOG(DLOG_DEBUG, SIMULATION, __VA_ARGS__)
#define SIMULATION_LOGI(...) PYPTO_SIM_LOG(DLOG_INFO, SIMULATION, __VA_ARGS__)
#define SIMULATION_LOGW(...) PYPTO_SIM_LOG(DLOG_WARN, SIMULATION, __VA_ARGS__)
#define SIMULATION_LOGE(errCode, fmt, ...) PYPTO_SIM_LOGE_WITH_ERRCODE(SIMULATION, errCode, fmt, ##__VA_ARGS__)

#define VERIFY_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, VERIFY, __VA_ARGS__)
#define VERIFY_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, VERIFY, __VA_ARGS__)
#define VERIFY_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, VERIFY, __VA_ARGS__)
#define VERIFY_LOGE(errCode, fmt, ...) PYPTO_HOST_LOGE_WITH_ERRCODE(VERIFY, errCode, fmt, ##__VA_ARGS__)
#define VERIFY_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, VERIFY, __VA_ARGS__)
#define VERIFY_LOGE_FULL(errCode, fmt, ...)                                   \
    PYPTO_HOST_SPLIT_LOG(DLOG_ERROR, VERIFY, "ErrCode: F%05X! Enum: %s " fmt, \
                         static_cast<uint32_t>(errCode) & 0xFFFFF, #errCode, ##__VA_ARGS__)

#define COMPILER_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, COMPILER_MONITOR, __VA_ARGS__)
#define COMPILER_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, COMPILER_MONITOR, __VA_ARGS__)
#define COMPILER_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, COMPILER_MONITOR, __VA_ARGS__)
#define COMPILER_LOGE(errCode, fmt, ...) PYPTO_HOST_LOGE_WITH_ERRCODE(COMPILER_MONITOR, errCode, fmt, ##__VA_ARGS__)
#define COMPILER_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, COMPILER_MONITOR, __VA_ARGS__)
#define COMPILER_WARN_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_WARN, COMPILER_MONITOR, __VA_ARGS__)
#define COMPILER_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, COMPILER_MONITOR, __VA_ARGS__)

#define ADAPTER_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, ADAPTER, __VA_ARGS__)
#define ADAPTER_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, ADAPTER, __VA_ARGS__)
#define ADAPTER_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, ADAPTER, __VA_ARGS__)
#define ADAPTER_LOGE(errCode, fmt, ...) PYPTO_HOST_LOGE_WITH_ERRCODE(ADAPTER, errCode, fmt, ##__VA_ARGS__)

#define PLATFORM_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, PLATFORM, __VA_ARGS__)
#define PLATFORM_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, PLATFORM, __VA_ARGS__)
#define PLATFORM_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, PLATFORM, __VA_ARGS__)
#define PLATFORM_LOGE(errCode, fmt, ...) PYPTO_HOST_LOGE_WITH_ERRCODE(PLATFORM, errCode, fmt, ##__VA_ARGS__)

#define MATMUL_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, MATMUL, __VA_ARGS__)
#define MATMUL_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, MATMUL, __VA_ARGS__)
#define MATMUL_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, MATMUL, __VA_ARGS__)
#define MATMUL_LOGE(errCode, fmt, ...) PYPTO_HOST_LOGE_WITH_ERRCODE(MATMUL, errCode, fmt, ##__VA_ARGS__)

#define CONV_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, CONV, __VA_ARGS__)
#define CONV_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, CONV, __VA_ARGS__)
#define CONV_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, CONV, __VA_ARGS__)
#define CONV_LOGE(errCode, fmt, ...) PYPTO_HOST_LOGE_WITH_ERRCODE(CONV, errCode, fmt, ##__VA_ARGS__)

#define VECTOR_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, VECTOR, __VA_ARGS__)
#define VECTOR_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, VECTOR, __VA_ARGS__)
#define VECTOR_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, VECTOR, __VA_ARGS__)
#define VECTOR_LOGE(errCode, fmt, ...) PYPTO_HOST_LOGE_WITH_ERRCODE(VECTOR, errCode, fmt, ##__VA_ARGS__)

#define PYPTO_LOGD(...) PYPTO_HOST_LOG_WITHOUT_MODULE(DLOG_DEBUG, __VA_ARGS__)
#define PYPTO_LOGI(...) PYPTO_HOST_LOG_WITHOUT_MODULE(DLOG_INFO, __VA_ARGS__)
#define PYPTO_LOGW(...) PYPTO_HOST_LOG_WITHOUT_MODULE(DLOG_WARN, __VA_ARGS__)
#define PYPTO_LOGE(...) PYPTO_HOST_LOG_WITHOUT_MODULE(DLOG_ERROR, __VA_ARGS__)
#define PYPTO_LOGE_FULL(fmt, ...) PYPTO_HOST_SPLIT_LOG_WITHOUT_MODULE(DLOG_ERROR, fmt, ##__VA_ARGS__)

#define AICORE_HOST_LOG_INNER(level, fmt, ...)                                                                       \
    do {                                                                                                             \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                             \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(false);                                                   \
        }                                                                                                            \
        if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel != nullptr &&                                          \
            npu::tile_fwk::LogFuncInfo::Instance().record != nullptr) {                                              \
            if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel(PYPTO, level, npu::tile_fwk::LogModule::AICORE)) { \
                npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, fmt, ##__VA_ARGS__);                     \
            }                                                                                                        \
        }                                                                                                            \
    } while (0)

#define AICORE_HOST_LOGD(fmt, ...) AICORE_HOST_LOG_INNER(DLOG_DEBUG, fmt, ##__VA_ARGS__)
#define AICORE_HOST_LOGI(fmt, ...) AICORE_HOST_LOG_INNER(DLOG_INFO, fmt, ##__VA_ARGS__)
#define AICORE_HOST_LOGW(fmt, ...) AICORE_HOST_LOG_INNER(DLOG_WARN, fmt, ##__VA_ARGS__)
#define AICORE_HOST_LOGE(fmt, ...) AICORE_HOST_LOG_INNER(DLOG_ERROR, fmt, ##__VA_ARGS__)

#endif
