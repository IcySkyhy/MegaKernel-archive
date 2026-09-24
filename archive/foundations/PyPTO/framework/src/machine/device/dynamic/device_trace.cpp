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
 * \file device_trace.cpp
 * \brief Device trace implementation with dynamic library loading
 */

#include <dlfcn.h>
#include <string>
#include <cstdarg>
#include <cstdio>
#include "tilefwk/error_code.h"
#include "machine/utils/device_log.h"
#include "device_trace.h"
#ifdef __DEVICE__
#include "machine/device/tilefwk/aicpu_common.h"
#include "machine/utils/machine_ws_intf.h"
#include "driver/ascend_hal_base.h"
#include "driver/ascend_hal.h"

namespace npu::tile_fwk::dynamic {

constexpr uint8_t MAX_HANDLE_NUM = 5;
std::atomic<int8_t> g_thread_idx{0};

DeviceTrace& DeviceTrace::GetInstance()
{
    static DeviceTrace deviceTrace;
    return deviceTrace;
}

DeviceTrace::~DeviceTrace()
{
    if (handle_ != nullptr) {
        dlclose(handle_);
        handle_ = nullptr;
    }
    for (auto pyptoHandle : pyptoHandleArray_) {
        if (pyptoHandle < 0) {
            continue;
        }
        TraceDestroy(pyptoHandle);
    }
    for (auto enventHandle : eventHandleArry_) {
        TraceEventDestroy(enventHandle);
    }
}

void* DeviceTrace::LoadLibrary(const std::string& libraryName)
{
    void* handle = dlopen(libraryName.c_str(), RTLD_NOW);
    if (handle == nullptr) {
        const char* error = dlerror();
        DEV_WARN("Failed to load library %s: %s", libraryName.c_str(), error ? error : "unknown error");
    }
    return handle;
}

void* DeviceTrace::GetSymbol(const std::string& symbolName)
{
    if (handle_ == nullptr) {
        DEV_WARN("Cannot get symbol %s from null handle", symbolName.c_str());
        return nullptr;
    }

    void* symbol = dlsym(handle_, symbolName.c_str());
    if (symbol == nullptr) {
        const char* error = dlerror();
        DEV_WARN("Can not get symbol %s: {%s}", symbolName.c_str(), error ? error : "unknown error");
    }
    return symbol;
}

int32_t DeviceTrace::InitializeAtraceFunctions()
{
    TraceCreate = reinterpret_cast<TraHandle (*)(TracerType, const char*)>(GetSymbol("AtraceCreate"));
    if (TraceCreate == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceDestroy = reinterpret_cast<void (*)(TraHandle)>(GetSymbol("AtraceDestroy"));
    if (TraceDestroy == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceSubmit = reinterpret_cast<TraStatus (*)(TraHandle, const void*, uint32_t)>(GetSymbol("AtraceSubmit"));
    if (TraceSubmit == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceSave = reinterpret_cast<TraStatus (*)(TracerType, bool)>(GetSymbol("AtraceSave"));
    if (TraceSave == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceEventCreate = reinterpret_cast<TraEventHandle (*)(const char*)>(GetSymbol("AtraceEventCreate"));
    if (TraceEventCreate == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceEventBindTrace = reinterpret_cast<TraStatus (*)(TraEventHandle, TraHandle)>(GetSymbol("AtraceEventBindTrace"));
    if (TraceEventBindTrace == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceEventReport = reinterpret_cast<TraStatus (*)(TraEventHandle)>(GetSymbol("AtraceEventReport"));
    if (TraceEventReport == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceEventReportSync = reinterpret_cast<TraStatus (*)(TraEventHandle)>(GetSymbol("AtraceEventReportSync"));
    if (TraceEventReportSync == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceEventDestroy = reinterpret_cast<void (*)(TraEventHandle)>(GetSymbol("AtraceEventDestroy"));
    if (TraceEventDestroy == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceSetGlobalAttr = reinterpret_cast<TraStatus (*)(const TraceGlobalAttr* attr)>(GetSymbol("AtraceSetGlobalAttr"));
    if (TraceSetGlobalAttr == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    return 0;
}

int32_t DeviceTrace::InitializeUtraceFunctions()
{
    TraceCreate = reinterpret_cast<TraHandle (*)(TracerType, const char*)>(GetSymbol("UtraceCreate"));
    if (TraceCreate == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceDestroy = reinterpret_cast<void (*)(TraHandle)>(GetSymbol("UtraceDestroy"));
    if (TraceDestroy == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceSubmit = reinterpret_cast<TraStatus (*)(TraHandle, const void*, uint32_t)>(GetSymbol("UtraceSubmit"));
    if (TraceSubmit == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceSave = reinterpret_cast<TraStatus (*)(TracerType, bool)>(GetSymbol("UtraceSave"));
    if (TraceSave == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceEventCreate = reinterpret_cast<TraEventHandle (*)(const char*)>(GetSymbol("UtraceEventCreate"));
    if (TraceEventCreate == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceEventBindTrace = reinterpret_cast<TraStatus (*)(TraEventHandle, TraHandle)>(GetSymbol("UtraceEventBindTrace"));
    if (TraceEventBindTrace == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceEventReport = reinterpret_cast<TraStatus (*)(TraEventHandle)>(GetSymbol("UtraceEventReport"));
    if (TraceEventReport == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceEventDestroy = reinterpret_cast<void (*)(TraEventHandle)>(GetSymbol("UtraceEventDestroy"));
    if (TraceEventDestroy == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    TraceSetGlobalAttr = reinterpret_cast<TraStatus (*)(const TraceGlobalAttr* attr)>(GetSymbol("UtraceSetGlobalAttr"));
    if (TraceSetGlobalAttr == nullptr) {
        return static_cast<int32_t>(DevCommonErr::CANN_API_NOT_FOUND);
    }

    return 0;
}

int32_t DeviceTrace::AtraceInitialize()
{
    handle_ = LoadLibrary("libascend_trace.so");
    if (handle_ != nullptr) {
        DEV_INFO("Successfully loaded libascend_trace.so");
        int32_t ret = InitializeAtraceFunctions();
        if (ret != 0) {
            DEV_TRACE_LOG_ERROR(DevCommonErr::INIT_FAILED, "Failed to initialize Atrace functions, error code: %d",
                                static_cast<int>(ret));
            dlclose(handle_);
            handle_ = nullptr;
            return ret;
        }
        DEV_INFO("Device trace initialized with Atrace backend");
        return 0;
    }
    return static_cast<int32_t>(DevCommonErr::LOAD_LIBRARY_FAILED);
}

int32_t DeviceTrace::UtraceInitialize()
{
    handle_ = LoadLibrary("libutrace.so");
    if (handle_ != nullptr) {
        DEV_INFO("Successfully loaded libutrace.so");
        int32_t ret = InitializeUtraceFunctions();
        if (ret != 0) {
            DEV_TRACE_LOG_ERROR(DevCommonErr::INIT_FAILED, "Failed to initialize Utrace functions, error code: %d",
                                static_cast<int>(ret));
            dlclose(handle_);
            handle_ = nullptr;
            return ret;
        }

        DEV_INFO("Device trace initialized with Utrace backend");
        return 0;
    }
    return static_cast<int32_t>(DevCommonErr::LOAD_LIBRARY_FAILED);
}

int32_t DeviceTrace::Initialize(void* targ)
{
    if (handle_ != nullptr) {
        DEV_DEBUG("Device trace already initialized");
        return 0;
    }

    DEV_INFO("Initializing device trace...");
    if (AtraceInitialize() == 0) {
        DEV_INFO("Current using so is libascend_trace.so");
        return ConnectTraceD2H(targ);
    }

    DEV_WARN("Failed to load libascend_trace.so, trying libutrace.so as fallback");
    if (UtraceInitialize() == 0) {
        DEV_INFO("Current using so is libutrace.so");
        return ConnectTraceD2H(targ);
    }
    DEV_TRACE_LOG_ERROR(DevCommonErr::LOAD_LIBRARY_FAILED,
                        "Failed to initialize device trace: no trace library available");
    return static_cast<int32_t>(DevCommonErr::LOAD_LIBRARY_FAILED);
}

int32_t DeviceTrace::ConnectTraceD2H(void* targ)
{
    DeviceKernelArgs* kargs = (DeviceKernelArgs*)targ;
    DeviceArgs* devArgs = reinterpret_cast<DeviceArgs*>(kargs->cfgdata);
    if (devArgs->devDfxArgAddr != 0) {
        DevDfxArgs* devDfxArgs = reinterpret_cast<DevDfxArgs*>(devArgs->devDfxArgAddr);
        TraceGlobalAttr traceAttr;
        traceAttr.saveMode = 1;
        uint32_t localDevId = 0;
        // This is primarily used to convert the actual phy device Id to the host-side deviceId
        // for locating the communication channel established by trace.
        drvGetLocalDevIDByHostDevID(devDfxArgs->deviceId, &localDevId);
        uint32_t hostPid = 0;
        drvQueryProcessHostPid(getpid(), nullptr, nullptr, &hostPid, nullptr);
        traceAttr.deviceId = static_cast<uint8_t>(localDevId);
        traceAttr.pid = hostPid;
        DEV_INFO("ArgsAddr: %lu, Set deviceId: %u.", devArgs->devDfxArgAddr, devDfxArgs->deviceId);

        if (TraceSetGlobalAttr(&traceAttr) != 0) {
            return static_cast<int32_t>(DevCommonErr::LOAD_LIBRARY_FAILED);
        }
        DEV_INFO("Set Global Attr success");
    }
    // create two event handle
    for (int i = 0; i < MAX_EVENT_NUM; i++) {
        std::string eventTraceHandleName = "PYPTO_Event_Trace_" + std::to_string(i);
        auto eventHandle = TraceEventCreate(eventTraceHandleName.c_str());
        if (eventHandle < 0) {
            DEV_WARN("Create pypto event trace failed");
            return static_cast<int32_t>(DevCommonErr::LOAD_LIBRARY_FAILED);
        }
        DEV_INFO("Create pypto eventHandle_ successful");
        eventHandleArry_[i] = eventHandle;
    }
    return 0;
}

int32_t DeviceTrace::BindHandleToEventHandle(TraHandle handle, uint8_t threadIdx)
{
    // Primarily, each AICPU creates one obj handle.
    //  The trace module limits each event handle to binding at most 5 obj handles.
    auto status = TraceEventBindTrace(eventHandleArry_[threadIdx / MAX_HANDLE_NUM], handle);
    if (status < 0) {
        DEV_WARN("Bind pypto trace handle to pypto event trace failed, error status: %d", status);
        return static_cast<int32_t>(DevCommonErr::LOAD_LIBRARY_FAILED);
    }
    return 0;
}

TraHandle DeviceTrace::CreateTraceHandle()
{
    thread_local TraHandle pyptoHandle = -1;
    if (pyptoHandle >= 0) {
        return pyptoHandle;
    }
    auto threadIdx = g_thread_idx.fetch_add(1, std::memory_order_relaxed);
    std::string handleName = "PyPTO_Aicpu_" + std::to_string(threadIdx) + "_Trace";
    pyptoHandle = TraceCreate(TracerType::TRACER_TYPE_SCHEDULE, handleName.c_str());
    if (pyptoHandle < 0) {
        DEV_WARN("Create pypto trace failed");
        return -1;
    }
    if (threadIdx < MAX_AICPU_NUM) {
        pyptoHandleArray_[threadIdx] = pyptoHandle;
    } else {
        pyptoHandle = -1;
        DEV_WARN("Create pypto trace failed because aicpu more than %d", MAX_AICPU_NUM);
        return pyptoHandle;
    }
    DEV_INFO("Create pypto trace Handle successful");
    auto status = BindHandleToEventHandle(pyptoHandle, threadIdx);
    if (status != 0) {
        return -1;
    }
    DEV_INFO("Bind pypto eventHandle_ successful");
    return pyptoHandle;
}

void DeviceTrace::SubmitTraceMsg(const std::string& traceMsg)
{
    auto pyptoHandle = CreateTraceHandle();
    if (pyptoHandle < 0 || traceMsg.empty()) {
        DEV_WARN("pypto Handle is null or traceMsg is empty, cann't to submit");
        return;
    }
    uint32_t msgSize = static_cast<uint32_t>(traceMsg.size());
    const void* buffer = reinterpret_cast<const void*>(traceMsg.c_str());
    uint32_t bufSize = msgSize > MAX_MSG_LEN ? MAX_MSG_LEN : msgSize;
    auto ret = TraceSubmit(pyptoHandle, buffer, bufSize);
    if (ret < 0) {
        DEV_WARN("Submit pyptoHandle buffer info failed, ret: %d", ret);
    }
}

void DeviceTrace::ReportTraceMsg()
{
    for (auto enventHandle : eventHandleArry_) {
        if (enventHandle < 0) {
            continue;
        }
        auto ret = TraceEventReport(enventHandle);
        if (ret < 0) {
            DEV_WARN("Report pypto event Handle buffer info failed, ret: %d", ret);
        }
        DEV_INFO("Event Handle %ld, Report Submit Msg success", enventHandle);
    }
}
} // namespace npu::tile_fwk::dynamic
#else
namespace npu::tile_fwk::dynamic {
DeviceTrace& DeviceTrace::GetInstance()
{
    static DeviceTrace deviceTrace;
    return deviceTrace;
}
void DeviceTrace::SubmitTraceMsg(const std::string& traceMsg) { (void)traceMsg; }
void DeviceTrace::ReportTraceMsg() {}
} // namespace npu::tile_fwk::dynamic
#endif
