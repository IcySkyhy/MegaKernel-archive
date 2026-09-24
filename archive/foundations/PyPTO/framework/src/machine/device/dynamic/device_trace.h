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
 * \file device_trace.h
 * \brief Device trace functionality with dynamic library loading
 *
 * This header provides device trace capabilities by dynamically loading
 * trace libraries (libascend_trace.so or libutrace.so) and providing
 * unified interfaces for trace operations.
 *
 * Key features:
 * - Automatic library detection and loading
 * - Thread-safe singleton pattern
 * - Support for both Atrace and Utrace backends
 * - PyPTO-style error handling and logging
 */

#pragma once

#include <functional>
#include <mutex>
#include <map>
#include <string>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <atomic>
#define MAX_MSG_LEN 112
static constexpr int MAX_AICPU_NUM = 8;
static constexpr int MAX_EVENT_NUM = 2;

#define DEV_ATRACE(fmt, ...)                                                    \
    do {                                                                        \
        char buf[MAX_MSG_LEN];                                                  \
        sprintf_s(buf, MAX_MSG_LEN, fmt, ##__VA_ARGS__);                        \
        npu::tile_fwk::dynamic::DeviceTrace::GetInstance().SubmitTraceMsg(buf); \
    } while (false)
#ifdef __DEVICE__
#include "trace/atrace_types.h"
#include "trace/atrace_pub.h"

namespace npu::tile_fwk::dynamic {

/**
 * \brief Device trace manager with dynamic library loading
 *
 * This singleton class manages device trace functionality by:
 * - Dynamically loading trace libraries at runtime
 * - Providing unified interfaces for trace operations
 * - Supporting both Atrace and Utrace backends
 * - Ensuring thread-safe initialization and access
 *
 * Thread-safety: All public methods are thread-safe through mutex protection.
 */
class DeviceTrace {
public:
    ~DeviceTrace();
    TraHandle CreateTraceHandle();
    void SubmitTraceMsg(const std::string& traceMsg);
    void ReportTraceMsg();
    /**
     * \brief Get the singleton instance of DeviceTrace
     * \return Reference to the singleton DeviceTrace instance
     */
    static DeviceTrace& GetInstance();

    /**
     * \brief Initialize the device trace manager
     *
     * Attempts to load trace libraries in the following order:
     * 1. libascend_trace.so (preferred)
     * 2. libutrace.so (fallback)
     *
     * \return SUCCESS if initialization succeeds, error code otherwise
     */
    int32_t Initialize(void* targ);

    int32_t ConnectTraceD2H(void* targ);

    int32_t BindHandleToEventHandle(TraHandle handle, uint8_t threadIdx);

private:
    /**
     * \brief Destroy a trace handle
     * \param handle Trace handle to destroy
     */
    std::function<void(TraHandle handle)> TraceDestroy{};

    /**
     * \brief Submit trace data
     * \param handle Trace handle
     * \param buffer Pointer to trace data buffer
     * \param bufSize Size of the trace data buffer
     * \return SUCCESS if submission succeeds, error code otherwise
     */
    std::function<TraStatus(TraHandle handle, const void* buffer, uint32_t bufSize)> TraceSubmit{};

    /**
     * \brief Create a trace handle
     * \param tracerType Type of tracer to create
     * \param objName Name of the trace object
     * \return Trace handle on success, nullptr on failure
     */
    std::function<TraHandle(TracerType tracerType, const char* objName)> TraceCreate{};

    /**
     * \brief Save trace data
     * \param tracerType Type of tracer to save
     * \param syncFlag Whether to save synchronously
     * \return Trace status
     */
    std::function<TraStatus(TracerType tracerType, bool syncFlag)> TraceSave{};

    /**
     * \brief Create a trace event handle
     * \param eventName Name of the event
     * \return Event handle on success, nullptr on failure
     */
    std::function<TraEventHandle(const char* eventName)> TraceEventCreate{};

    /**
     * \brief Bind a trace event to a trace handle
     * \param eventHandle Event handle to bind
     * \param handle Trace handle to bind to
     * \return Trace status
     */
    std::function<TraStatus(TraEventHandle eventHandle, TraHandle handle)> TraceEventBindTrace{};

    /**
     * \brief Report a trace event
     * \param eventHandle Event handle to report
     * \return Trace status
     */
    std::function<TraStatus(TraEventHandle eventHandle)> TraceEventReport{};

    /**
     * \brief Report a trace event synchronously
     * \param eventHandle Event handle to report
     * \return Trace status
     */
    std::function<TraStatus(TraEventHandle eventHandle)> TraceEventReportSync{};

    /**
     * \brief Destroy a trace event handle
     * \param eventHandle Event handle to destroy
     */
    std::function<void(TraEventHandle eventHandle)> TraceEventDestroy{};

    std::function<TraStatus(const TraceGlobalAttr* attr)> TraceSetGlobalAttr{};

private:
    void* handle_{nullptr};
    TraHandle pyptoHandleArray_[MAX_AICPU_NUM] = {-1};
    TraEventHandle eventHandleArry_[MAX_EVENT_NUM] = {-1};
    std::atomic<int> threadIdx;

    DeviceTrace(const DeviceTrace&) = delete;
    DeviceTrace& operator=(const DeviceTrace&) = delete;
    DeviceTrace() = default;
    int32_t UtraceInitialize();
    int32_t AtraceInitialize();

    /**
     * \brief Initialize Atrace function pointers
     * \return SUCCESS if successful, error code otherwise
     */
    int32_t InitializeAtraceFunctions();

    /**
     * \brief Initialize Utrace function pointers
     * \return SUCCESS if successful, error code otherwise
     */
    int32_t InitializeUtraceFunctions();

    /**
     * \brief Load a dynamic library
     * \param libraryName Name of the library to load
     * \return Handle to the loaded library, nullptr on failure
     */
    void* LoadLibrary(const std::string& libraryName);

    /**
     * \brief Get a symbol from the loaded library
     * \param symbolName Name of the symbol to retrieve
     * \return Pointer to the symbol, nullptr on failure
     */
    void* GetSymbol(const std::string& symbolName);
};
} // namespace npu::tile_fwk::dynamic

#else
namespace npu::tile_fwk::dynamic {
class DeviceTrace {
public:
    static DeviceTrace& GetInstance();
    void SubmitTraceMsg(const std::string& traceMsg);
    void ReportTraceMsg();
};
} // namespace npu::tile_fwk::dynamic
#endif
