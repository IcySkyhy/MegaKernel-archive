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
 * \file acl_api.cpp
 * \brief
 */

#include "adapter/api/acl_api.h"

#ifdef BUILD_WITH_CANN
#include "adapter/manager/adapter_manager.h"
#include <type_traits>
#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#include "runtime/base.h"
#endif
#include "adapter/stubs/acl_stubs.h"

namespace npu::tile_fwk {
AclError AclInit(const char* configPath)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Init);
    if (func != nullptr) {
        aclError (*aclFunc)(const char*) = reinterpret_cast<aclError (*)(const char*)>(func);
        return aclFunc(configPath);
    }
#endif
    return StubAclInit(configPath);
}

AclError AclFinalize()
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Finalize);
    if (func != nullptr) {
        aclError (*aclFunc)(void) = reinterpret_cast<aclError (*)(void)>(func);
        return aclFunc();
    }
#endif
    return StubAclFinalize();
}

AclError AclRtMemcpy(void* dst, size_t destMax, const void* src, size_t count, AclRtMemcpyKind kind)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtMemcpy);
    if (func != nullptr) {
        aclError (*aclFunc)(void*, size_t, const void*, size_t, aclrtMemcpyKind) = reinterpret_cast<aclError (*)(
            void*, size_t, const void*, size_t, aclrtMemcpyKind)>(func);
        return aclFunc(dst, destMax, src, count, static_cast<aclrtMemcpyKind>(kind));
    }
#endif
    return StubRtMemcpy(dst, destMax, src, count, kind);
}

AclError AclRtMalloc(void** devPtr, size_t size, AclRtMemMallocPolicy policy)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtMalloc);
    if (func != nullptr) {
        aclError (*aclFunc)(
            void**, size_t,
            aclrtMemMallocPolicy) = reinterpret_cast<aclError (*)(void**, size_t, aclrtMemMallocPolicy)>(func);
        return aclFunc(devPtr, size, static_cast<aclrtMemMallocPolicy>(policy));
    }
#endif
    return StubRtMalloc(devPtr, size, policy);
}

AclError AclRtFree(void* devPtr)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtFree);
    if (func != nullptr) {
        aclError (*aclFunc)(void*) = reinterpret_cast<aclError (*)(void*)>(func);
        return aclFunc(devPtr);
    }
#endif
    return StubRtFree(devPtr);
}

AclError AclRtSetDevice(int32_t deviceId)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSetDevice);
    if (func != nullptr) {
        aclError (*aclFunc)(int32_t) = reinterpret_cast<aclError (*)(int32_t)>(func);
        return aclFunc(deviceId);
    }
#endif
    return StubRtSetDevice(deviceId);
}

AclError AclRtResetDevice(int32_t deviceId)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtResetDevice);
    if (func != nullptr) {
        aclError (*aclFunc)(int32_t) = reinterpret_cast<aclError (*)(int32_t)>(func);
        return aclFunc(deviceId);
    }
#endif
    return StubRtResetDevice(deviceId);
}

AclError AclRtCreateEvent(AclRtEvent* event)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateEvent);
    if (func != nullptr) {
        aclError (*aclFunc)(aclrtEvent*) = reinterpret_cast<aclError (*)(aclrtEvent*)>(func);
        return aclFunc(event);
    }
#endif
    return StubRtCreateEvent(event);
}

AclError AclRtRecordEvent(AclRtEvent event, AclRtStream stream)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtRecordEvent);
    if (func != nullptr) {
        aclError (*aclFunc)(aclrtEvent, aclrtStream) = reinterpret_cast<aclError (*)(aclrtEvent, aclrtStream)>(func);
        return aclFunc(event, stream);
    }
#endif
    return StubRtRecordEvent(event, stream);
}

AclError AclRtCreateEventExWithFlag(AclRtEvent* event, uint32_t flag)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateEventExWithFlag);
    if (func != nullptr) {
        aclError (*aclFunc)(aclrtEvent*, uint32_t) = reinterpret_cast<aclError (*)(aclrtEvent*, uint32_t)>(func);
        return aclFunc(event, flag);
    }
#endif
    return StubRtCreateEventExWithFlag(event, flag);
}

AclError AclRtStreamWaitEvent(AclRtStream stream, AclRtEvent event)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtStreamWaitEvent);
    if (func != nullptr) {
        aclError (*aclFunc)(aclrtStream, aclrtEvent) = reinterpret_cast<aclError (*)(aclrtStream, aclrtEvent)>(func);
        return aclFunc(stream, event);
    }
#endif
    return StubRtStreamWaitEvent(stream, event);
}

AclError AclRtGetStreamResLimit(AclRtStream stream, AclRtDevResLimitType type, uint32_t* value)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtGetStreamResLimit);
    if (func != nullptr) {
        aclError (*aclFunc)(
            aclrtStream, aclrtDevResLimitType,
            uint32_t*) = reinterpret_cast<aclError (*)(aclrtStream, aclrtDevResLimitType, uint32_t*)>(func);
        return aclFunc(stream, static_cast<aclrtDevResLimitType>(type), value);
    }
#endif
    return StubRtGetStreamResLimit(stream, type, value);
}

AclError AclRtGetStreamAttribute(AclRtStream stream, AclRtStreamAttr stmAttrType, AclRtStreamAttrValue* value)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtGetStreamAttribute);
    if (func != nullptr) {
        aclError (*aclFunc)(aclrtStream, aclrtStreamAttr, aclrtStreamAttrValue*) = reinterpret_cast<aclError (*)(
            aclrtStream, aclrtStreamAttr, aclrtStreamAttrValue*)>(func);
        return aclFunc(stream, static_cast<aclrtStreamAttr>(stmAttrType),
                       reinterpret_cast<aclrtStreamAttrValue*>(value));
    }
#endif
    return StubRtGetStreamAttribute(stream, stmAttrType, value);
}

AclError AclRtCacheLastTaskOpInfo(const void* const infoPtr, size_t infoSize)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCacheLastTaskOpInfo);
    if (func != nullptr) {
        aclError (*aclFunc)(const void* const,
                            size_t) = reinterpret_cast<aclError (*)(const void* const, size_t)>(func);
        return aclFunc(infoPtr, infoSize);
    }
#endif
    return StubRtCacheLastTaskOpInfo(infoPtr, infoSize);
}

AclError AclRtSetExceptionInfoCallback(AclRtExceptionInfoCallback callback)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSetExceptionInfoCallback);
    if (func != nullptr) {
        aclError (*aclFunc)(aclrtExceptionInfoCallback) = reinterpret_cast<aclError (*)(aclrtExceptionInfoCallback)>(
            func);
        return aclFunc(reinterpret_cast<aclrtExceptionInfoCallback>(callback));
    }
#endif
    return StubRtSetExceptionInfoCallback(callback);
}

AclError AclRtCreateStream(AclRtStream* stream)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateStream);
    if (func != nullptr) {
        aclError (*aclFunc)(aclrtStream*) = reinterpret_cast<aclError (*)(aclrtStream*)>(func);
        return aclFunc(stream);
    }
#endif
    return StubRtCreateStream(stream);
}

AclError AclRtSynchronizeStream(AclRtStream stream)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSynchronizeStream);
    if (func != nullptr) {
        aclError (*aclFunc)(aclrtStream) = reinterpret_cast<aclError (*)(aclrtStream)>(func);
        return aclFunc(stream);
    }
#endif
    return StubRtSynchronizeStream(stream);
}

AclError AclRtDestroyStream(AclRtStream stream)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtDestroyStream);
    if (func != nullptr) {
        aclError (*aclFunc)(aclrtStream) = reinterpret_cast<aclError (*)(aclrtStream)>(func);
        return aclFunc(stream);
    }
#endif
    return StubRtDestroyStream(stream);
}

AclError AclRtBinaryLoadFromFile(const char* binPath, AclRtBinaryLoadOptions* options, AclRtBinHandle* binHandle)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::BinaryLoadFromFile);
    if (func != nullptr) {
        aclError (*aclFunc)(const char*, aclrtBinaryLoadOptions*, aclrtBinHandle*) = reinterpret_cast<aclError (*)(
            const char*, aclrtBinaryLoadOptions*, aclrtBinHandle*)>(func);
        return aclFunc(binPath, reinterpret_cast<aclrtBinaryLoadOptions*>(options),
                       reinterpret_cast<aclrtBinHandle*>(binHandle));
    }
#endif
    return StubRtBinaryLoadFromFile(binPath, options, binHandle);
}

AclError AclRtBinaryGetFunction(AclRtBinHandle binHandle, const char* kernelName, AclRtFuncHandle* funcHandle)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::BinaryGetFunction);
    if (func != nullptr) {
        aclError (*aclFunc)(
            aclrtBinHandle, const char*,
            aclrtFuncHandle*) = reinterpret_cast<aclError (*)(aclrtBinHandle, const char*, aclrtFuncHandle*)>(func);
        return aclFunc(reinterpret_cast<aclrtBinHandle>(binHandle), kernelName,
                       reinterpret_cast<aclrtFuncHandle*>(funcHandle));
    }
#endif
    return StubRtBinaryGetFunction(binHandle, kernelName, funcHandle);
}

AclError AclMdlRICaptureGetInfo(AclRtStream stream, AclMdlRICaptureStatus* status, AclMdlRI* modelRI)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::MdlRICaptureGetInfo);
    if (func != nullptr) {
        aclError (*aclFunc)(
            aclrtStream, aclmdlRICaptureStatus*,
            aclmdlRI*) = reinterpret_cast<aclError (*)(aclrtStream, aclmdlRICaptureStatus*, aclmdlRI*)>(func);
        return aclFunc(stream, reinterpret_cast<aclmdlRICaptureStatus*>(status), reinterpret_cast<aclmdlRI*>(modelRI));
    }
#endif
    return StubMdlRICaptureGetInfo(stream, status, modelRI);
}

AclError AclMdlRICaptureThreadExchangeMode(AclMdlRICaptureMode* mode)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::MdlRICaptureThreadExchangeMode);
    if (func != nullptr) {
        aclError (*aclFunc)(aclmdlRICaptureMode*) = reinterpret_cast<aclError (*)(aclmdlRICaptureMode*)>(func);
        return aclFunc(reinterpret_cast<aclmdlRICaptureMode*>(mode));
    }
#endif
    return StubMdlRICaptureThreadExchangeMode(mode);
}

AclError AclSysGetVersionStr(const char* pkgName, char* versionStr)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::SysGetVersionStr);
    if (func != nullptr) {
        // Match acl_rt.h: aclsysGetVersionStr(char *pkgName, char *versionStr)
        aclError (*aclFunc)(char*, char*) = reinterpret_cast<aclError (*)(char*, char*)>(func);
        return aclFunc(const_cast<char*>(pkgName), versionStr);
    }
#endif
    return StubSysGetVersionStr(pkgName, versionStr);
}
#ifdef BUILD_WITH_CANN
static_assert(std::is_same<AclError, aclError>::value);
static_assert(std::is_same<AclRtStream, aclrtStream>::value);
static_assert(std::is_same<AclRtEvent, aclrtEvent>::value);
static_assert(std::is_same<AclRtBinHandle, aclrtBinHandle>::value);
static_assert(std::is_same<AclRtFuncHandle, aclrtFuncHandle>::value);
static_assert(sizeof(AclRtBinaryLoadOption) == sizeof(aclrtBinaryLoadOption));
static_assert(sizeof(AclRtBinaryLoadOptions) == sizeof(aclrtBinaryLoadOptions));
static_assert(std::is_same<AclMdlRI, aclmdlRI>::value);
static_assert(ACLRT_SUCCESS == ACL_SUCCESS);
static_assert(ACLRT_ERROR_REPEAT_INITIALIZE == ACL_ERROR_REPEAT_INITIALIZE);
static_assert(sizeof(AclRtStreamAttrValue) == sizeof(aclrtStreamAttrValue));
#endif
} // namespace npu::tile_fwk
