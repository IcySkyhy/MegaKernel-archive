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

#include <cstdint>
#include <cstring>
#include <cmath>

#define DLOG_DEBUG 0x0
#define DLOG_INFO 0x1
#define DLOG_WARN 0x2
#define DLOG_ERROR 0x3

#define PYPTO 59

#ifndef __DEVICE__
#ifndef __FILE_NAME__
#define __FILE_NAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

namespace npu::tile_fwk {
enum class LogModule {
    FUNCTION = 0,
    PASS,
    CODEGEN,
    MACHINE,
    DISTRIBUTED,
    SIMULATION,
    VERIFY,
    COMPILER_MONITOR,
    ADAPTER,
    PLATFORM,
    CONV,
    MATMUL,
    VECTOR,
    AICORE,
    BOTTOM
};

class LogFuncInfo {
public:
    static LogFuncInfo& Instance();
    int32_t (*checkLevel)(int32_t, int32_t, LogModule);
    void (*record)(int32_t, int32_t, const char*, ...) __attribute__((format(printf, 3, 4)));
    void (*pyptoRecord)(int32_t, int32_t, const char*, ...) __attribute__((format(printf, 3, 4)));
    void (*setAttr)(bool);

private:
    LogFuncInfo();
    ~LogFuncInfo();
};
} // namespace npu::tile_fwk

#define PYPTO_HOST_LOG_INNER(level, module, has_module, fmt, ...)                                                    \
    do {                                                                                                             \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                             \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(false);                                                   \
        }                                                                                                            \
        if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel != nullptr &&                                          \
            npu::tile_fwk::LogFuncInfo::Instance().record != nullptr) {                                              \
            if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel(PYPTO, level, npu::tile_fwk::LogModule::module)) { \
                if (has_module) {                                                                                    \
                    npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:" fmt, __FILE_NAME__,   \
                                                                  __LINE__, #module, ##__VA_ARGS__);                 \
                } else {                                                                                             \
                    npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d]:" fmt, __FILE_NAME__,       \
                                                                  __LINE__, ##__VA_ARGS__);                          \
                }                                                                                                    \
            }                                                                                                        \
        }                                                                                                            \
    } while (0)

#define PYPTO_HOST_LOG(level, module, fmt, ...) PYPTO_HOST_LOG_INNER(level, module, true, fmt, ##__VA_ARGS__)

#define PYPTO_HOST_LOG_WITHOUT_MODULE(level, fmt, ...) PYPTO_HOST_LOG_INNER(level, BOTTOM, false, fmt, ##__VA_ARGS__)

#define MAX_LOG_LENGTH 880

#define PYPTO_HOST_SPLIT_LOG_INNER(level, module, has_module, fmt, ...)                                                \
    do {                                                                                                               \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                               \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(false);                                                     \
        }                                                                                                              \
        if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel == nullptr ||                                            \
            npu::tile_fwk::LogFuncInfo::Instance().record == nullptr) {                                                \
            break;                                                                                                     \
        }                                                                                                              \
        if (!npu::tile_fwk::LogFuncInfo::Instance().checkLevel(PYPTO, level, npu::tile_fwk::LogModule::module)) {      \
            break;                                                                                                     \
        }                                                                                                              \
        char* formatStr = nullptr;                                                                                     \
        int len = asprintf(&formatStr, fmt, ##__VA_ARGS__);                                                            \
        if (len <= 0 || formatStr == nullptr) {                                                                        \
            break;                                                                                                     \
        }                                                                                                              \
        if (len < MAX_LOG_LENGTH) {                                                                                    \
            if (has_module) {                                                                                          \
                npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:%s", __FILE_NAME__, __LINE__, \
                                                              #module, formatStr);                                     \
            } else {                                                                                                   \
                npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d]:%s", __FILE_NAME__, __LINE__,     \
                                                              formatStr);                                              \
            }                                                                                                          \
        } else {                                                                                                       \
            char* msgBegin = formatStr;                                                                                \
            char* msgEnd = formatStr + len;                                                                            \
            while (msgBegin < msgEnd) {                                                                                \
                if (has_module) {                                                                                      \
                    npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:%.880s", __FILE_NAME__,   \
                                                                  __LINE__, #module, msgBegin);                        \
                } else {                                                                                               \
                    npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d]:%.880s", __FILE_NAME__,       \
                                                                  __LINE__, msgBegin);                                 \
                }                                                                                                      \
                msgBegin += MAX_LOG_LENGTH;                                                                            \
            }                                                                                                          \
        }                                                                                                              \
        free(formatStr);                                                                                               \
    } while (0)

#define PYPTO_HOST_SPLIT_LOG(level, module, fmt, ...) \
    PYPTO_HOST_SPLIT_LOG_INNER(level, module, true, fmt, ##__VA_ARGS__)

#define PYPTO_HOST_SPLIT_LOG_WITHOUT_MODULE(level, fmt, ...) \
    PYPTO_HOST_SPLIT_LOG_INNER(level, BOTTOM, false, fmt, ##__VA_ARGS__)

#define PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(level, module, fmt, ...)                                                  \
    do {                                                                                                             \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                             \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(false);                                                   \
        }                                                                                                            \
        if (npu::tile_fwk::LogFuncInfo::Instance().record != nullptr) {                                              \
            npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:" fmt, __FILE_NAME__, __LINE__, \
                                                          #module, ##__VA_ARGS__);                                   \
        }                                                                                                            \
    } while (0)

#define PYPTO_SIM_LOG(level, module, fmt, ...)                                                                       \
    do {                                                                                                             \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                             \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(true);                                                    \
        }                                                                                                            \
        if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel != nullptr &&                                          \
            npu::tile_fwk::LogFuncInfo::Instance().pyptoRecord != nullptr) {                                         \
            if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel(PYPTO, level, npu::tile_fwk::LogModule::module)) { \
                npu::tile_fwk::LogFuncInfo::Instance().pyptoRecord(PYPTO, level, "[%s:%d][%s]:" fmt, __FILE_NAME__,  \
                                                                   __LINE__, #module, ##__VA_ARGS__);                \
            }                                                                                                        \
        }                                                                                                            \
    } while (0)

#define PYPTO_SIM_LOG_WITHOUT_LEVEL_CHECK(level, module, fmt, ...)                                              \
    do {                                                                                                        \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                        \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(true);                                               \
        }                                                                                                       \
        if (npu::tile_fwk::LogFuncInfo::Instance().pyptoRecord != nullptr) {                                    \
            npu::tile_fwk::LogFuncInfo::Instance().pyptoRecord(PYPTO, level, "[%s:%d][%s]:" fmt, __FILE_NAME__, \
                                                               __LINE__, #module, ##__VA_ARGS__);               \
        }                                                                                                       \
    } while (0)
#endif
