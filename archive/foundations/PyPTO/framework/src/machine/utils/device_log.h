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
 * \file device_log.h
 * \brief
 */

#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <ctime>
#include <cassert>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <execinfo.h>
#include "securec.h"
#include "tilefwk/aikernel_define.h"
#include "machine/utils/device_switch.h"
#include "tilefwk/error_code.h"
#include "machine/device/dynamic/device_trace.h"

#ifdef __DEVICE__
#include "dlog_pub.h"
#include "machine/device/dynamic/aicpu_instrumentation.h"
#else
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error.h"
#endif

namespace npu::tile_fwk {
#define DEV_IF_NONDEVICE if constexpr (!IsDeviceMode())

#define DEV_IF_DEVICE if constexpr (IsDeviceMode())

#define DEV_IF_VERBOSE_DEBUG if constexpr (IsCompileVerboseLog())

#define DEV_IF_VERBOSE_LOG if constexpr (IsCompileVerboseLog())

inline constexpr bool IsCompileVerboseLog()
{
#if ENABLE_COMPILE_VERBOSE_LOG
    return true;
#else
    return false;
#endif
}

#ifdef __DEVICE__
#define GET_TID() syscall(__NR_gettid)
#define LOG_MOD_ID AICPU

#define DEV_IF_DEBUG if (unlikely(!HardBranchTrue(verboseDebug) || IsLogEnableDebug()))
#define DEV_IF_INFO if (unlikely(!HardBranchTrue(verboseInfo) || IsLogEnableInfo()))

inline bool g_isLogEnableDebug = false;
inline bool g_isLogEnableInfo = false;
inline bool g_isLogEnableWarn = false;
inline bool g_isLogEnableError = false;

HardBranchGroupDefine(verboseInfo);
HardBranchGroupDefine(verboseDebug);

inline void InitLogSwitch()
{
    g_isLogEnableDebug = CheckLogLevel(LOG_MOD_ID, DLOG_DEBUG);
    g_isLogEnableInfo = CheckLogLevel(LOG_MOD_ID, DLOG_INFO);
    g_isLogEnableWarn = CheckLogLevel(LOG_MOD_ID, DLOG_WARN);
    g_isLogEnableError = CheckLogLevel(LOG_MOD_ID, DLOG_ERROR);
    if (g_isLogEnableDebug) {
        npu::tile_fwk::dynamic::HardBranchManager::GetInstance().AddGroup(HardBranchGroupCreate(verboseDebug));
    }
    if (g_isLogEnableInfo) {
        npu::tile_fwk::dynamic::HardBranchManager::GetInstance().AddGroup(HardBranchGroupCreate(verboseInfo));
    }
    npu::tile_fwk::dynamic::HardBranchManager::GetInstance().SwitchToJump();
    npu::tile_fwk::dynamic::HardBranchManager::GetInstance().Clear();
}

inline bool IsLogEnableDebug() { return g_isLogEnableDebug; }
inline bool IsLogEnableInfo() { return g_isLogEnableInfo; }
inline bool IsLogEnableWarn() { return g_isLogEnableWarn; }
inline bool IsLogEnableError() { return g_isLogEnableError; }

constexpr int MAX_LOG_CHUNK = 824;
template <typename... Args>
inline void DeviceLogSplitDebug(const char* func, const char* format, Args... args)
{
    char* formatStr = nullptr;
    int len = asprintf(&formatStr, format, args...);
    if (len <= 0 || formatStr == nullptr) {
        return;
    }
    // 分段输出
    if (len <= MAX_LOG_CHUNK) {
        dlog_debug(LOG_MOD_ID, "%lu %s\n%s", GET_TID(), func, formatStr);
    } else {
        char* msgBegin = formatStr;
        char* msgEnd = formatStr + len;
        int index = 1;
        int total = (len + MAX_LOG_CHUNK - 1) / MAX_LOG_CHUNK;
        while (msgBegin < msgEnd) {
            dlog_debug(LOG_MOD_ID, "%lu %s [Segment %d/%d]\n%.824s", GET_TID(), func, index, total, msgBegin);
            msgBegin += MAX_LOG_CHUNK;
            index++;
        }
    }
    free(formatStr);
}

#define DEV_DEBUG_SPLIT(fmt, ...)                                            \
    do {                                                                     \
        if (unlikely(!HardBranchTrue(verboseDebug) || IsLogEnableDebug())) { \
            DeviceLogSplitDebug(__FUNCTION__, fmt, ##__VA_ARGS__);           \
        }                                                                    \
    } while (false)

#define DEV_DEBUG(fmt, ...)                                                                  \
    do {                                                                                     \
        if (unlikely(!HardBranchTrue(verboseDebug) || IsLogEnableDebug())) {                 \
            dlog_debug(LOG_MOD_ID, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                                                    \
    } while (false)

#define DEV_INFO(fmt, ...)                                                                  \
    do {                                                                                    \
        if (unlikely(!HardBranchTrue(verboseInfo) || IsLogEnableInfo())) {                  \
            dlog_info(LOG_MOD_ID, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                                                   \
    } while (false)

#define DEV_WARN(fmt, ...)                                                                  \
    do {                                                                                    \
        if (IsLogEnableWarn()) {                                                            \
            dlog_warn(LOG_MOD_ID, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                                                   \
    } while (false)

#define DEV_ERROR(errCode, fmt, ...)                                                         \
    do {                                                                                     \
        if (IsLogEnableError()) {                                                            \
            dlog_error(LOG_MOD_ID, "%lu %s\nErrCode: F%05X! " #fmt, GET_TID(), __FUNCTION__, \
                       static_cast<uint32_t>(errCode) & 0xFFFFF, ##__VA_ARGS__);             \
            DEV_ATRACE(fmt, ##__VA_ARGS__);                                                  \
        }                                                                                    \
    } while (false)

#define DEV_EVENT(fmt, ...)                                                                            \
    do {                                                                                               \
        dlog_info(LOG_MOD_ID | RUN_LOG_MASK, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__); \
    } while (false)

#define DEV_TRACE_LOG_ERROR(errCode, fmt, ...)                                               \
    do {                                                                                     \
        if (IsLogEnableError()) {                                                            \
            dlog_error(LOG_MOD_ID, "%lu %s\nErrCode: F%05X! " #fmt, GET_TID(), __FUNCTION__, \
                       static_cast<uint32_t>(errCode) & 0xFFFFF, ##__VA_ARGS__);             \
        }                                                                                    \
    } while (false)

#define DEV_VERBOSE_DEBUG(fmt, ...)                           \
    do {                                                      \
        DEV_IF_VERBOSE_LOG { DEV_DEBUG(fmt, ##__VA_ARGS__); } \
    } while (0)

#define DEV_VERBOSE_DEBUG_SPLIT(fmt, ...)                           \
    do {                                                            \
        DEV_IF_VERBOSE_LOG { DEV_DEBUG_SPLIT(fmt, ##__VA_ARGS__); } \
    } while (0)

#define DEV_VERBOSE_INFO(fmt, ...)                           \
    do {                                                     \
        DEV_IF_VERBOSE_LOG { DEV_INFO(fmt, ##__VA_ARGS__); } \
    } while (0)

#define DEV_ASSERT_MSG(errCode, expr, fmt, args...)                           \
    do {                                                                      \
        if (!(expr)) {                                                        \
            DEV_ERROR(errCode, "Assertion failed (%s): " fmt, #expr, ##args); \
            abort();                                                          \
        }                                                                     \
    } while (0)

#define DEV_ASSERT(errCode, expr)                               \
    do {                                                        \
        if (!(expr)) {                                          \
            DEV_ERROR(errCode, "Assertion failed (%s)", #expr); \
            abort();                                            \
        }                                                       \
    } while (0)

#define DEV_MEM_DUMP(fmt, args...)

#else // none device
#define DEV_IF_DEBUG if (true)
#define DEV_IF_INFO if (true)

#define DEV_VERBOSE_DEBUG_SPLIT(fmt, args...) PYPTO_SIM_LOG(DLOG_DEBUG, MACHINE, fmt, ##args)
#define DEV_VERBOSE_DEBUG(fmt, args...) PYPTO_SIM_LOG(DLOG_DEBUG, MACHINE, fmt, ##args)
#define DEV_VERBOSE_INFO(fmt, args...) PYPTO_SIM_LOG(DLOG_INFO, MACHINE, fmt, ##args)
#define DEV_DEBUG_SPLIT(fmt, args...) PYPTO_SIM_LOG(DLOG_DEBUG, MACHINE, fmt, ##args)
#define DEV_DEBUG(fmt, args...) PYPTO_SIM_LOG(DLOG_DEBUG, MACHINE, fmt, ##args)
#define DEV_INFO(fmt, args...) PYPTO_SIM_LOG(DLOG_INFO, MACHINE, fmt, ##args)
#define DEV_WARN(fmt, args...) PYPTO_SIM_LOG(DLOG_WARN, MACHINE, fmt, ##args)
#define DEV_ERROR(errCode, fmt, args...) PYPTO_SIM_LOGE_WITH_ERRCODE(MACHINE, errCode, fmt, ##args)
#define DEV_EVENT(fmt, args...) PYPTO_SIM_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, MACHINE, fmt, ##args)

#if DEBUG_MEM_DUMP_LEVEL != DEBUG_MEM_DUMP_DISABLE
#define DEV_MEM_DUMP(fmt, args...) MACHINE_LOGD("[WsMem Statistics] " fmt, ##args)
#else
#define DEV_MEM_DUMP(fmt, args...)
#endif // DEBUG_MEM_DUMP_LEVEL != DEBUG_MEM_DUMP_DISABLE

#define DEV_ASSERT_MSG(errCode, expr, fmt, args...)        \
    do {                                                   \
        if (!(expr)) {                                     \
            DEV_ERROR(errCode, "%s :" fmt, #expr, ##args); \
            MACHINE_ASSERT(false);                         \
        }                                                  \
    } while (0)

#define DEV_ASSERT(errCode, expr)            \
    do {                                     \
        if (!(expr)) {                       \
            DEV_ERROR(errCode, "%s", #expr); \
            MACHINE_ASSERT(false);           \
        }                                    \
    } while (0)

#endif

#define DevMemcpyS(dest, destMax, src, count) \
    npu::tile_fwk::DevMemcpySWithCheck((dest), (destMax), (src), (count), __func__, __FILE__, __LINE__)

inline void DevMemcpySWithCheck(void* dest, size_t destMax, const void* src, size_t count, const char* func,
                                const char* file, int line)
{
    if (count > 0) {
        const errno_t ret = memcpy_s(dest, destMax, src, count);
        if (ret != EOK) {
            DEV_ERROR(DevCommonErr::MEMCPY_FAILED,
                      "memcpy_s failed: func=%s, file=%s:%d, ret=%d, dest=%p, destMax=%zu, src=%p, count=%zu", func,
                      file, line, ret, dest, destMax, src, count);
            DEV_ASSERT(DevCommonErr::MEMCPY_FAILED, false);
        }
    }
}

#define BACKTRACE_STACK_COUNT 64

static inline void PrintBacktrace [[maybe_unused]] (const npu::tile_fwk::ThreadErr errcode,
                                                    const std::string& prefix = "", int count = BACKTRACE_STACK_COUNT)
{
    std::vector<void*> backtraceStack(count);
    int backtraceStackCount = backtrace(backtraceStack.data(), static_cast<int>(backtraceStack.size()));
    DEV_ERROR(errcode, "backtrace %s count:%d", prefix.c_str(), backtraceStackCount);
    char** backtraceSymbolList = backtrace_symbols(backtraceStack.data(), backtraceStackCount);
    for (int i = 0; i < backtraceStackCount; i++) {
        DEV_ERROR(errcode, "backtrace %s frame[%d]: %s", prefix.c_str(), i, backtraceSymbolList[i]);
    }
    free(backtraceSymbolList);
}
} // namespace npu::tile_fwk
