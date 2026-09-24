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
 * \file acl_stubs.cpp
 * \brief
 */

#include "adapter/stubs/acl_stubs.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
AclError StubAclInit(const char* configPath)
{
    ADAPTER_LOGD("Enter stub function of AclInit.");
    (void)configPath;
    return ACLRT_SUCCESS;
}

AclError StubAclFinalize()
{
    ADAPTER_LOGD("Enter stub function of AclFinalize.");
    return ACLRT_SUCCESS;
}

AclError StubRtMemcpy(void* dst, size_t destMax, const void* src, size_t count, AclRtMemcpyKind kind)
{
    ADAPTER_LOGD("Enter stub function of AclRtMemcpy.");
    (void)dst;
    (void)destMax;
    (void)src;
    (void)count;
    (void)kind;
    return ACLRT_SUCCESS;
}

AclError StubRtMalloc(void** devPtr, size_t size, AclRtMemMallocPolicy policy)
{
    ADAPTER_LOGD("Enter stub function of AclRtMalloc.");
    (void)devPtr;
    (void)size;
    (void)policy;
    return ACLRT_SUCCESS;
}

AclError StubRtFree(void* devPtr)
{
    ADAPTER_LOGD("Enter stub function of AclRtFree.");
    (void)devPtr;
    return ACLRT_SUCCESS;
}

AclError StubRtSetDevice(int32_t deviceId)
{
    ADAPTER_LOGD("Enter stub function of AclRtSetDevice.");
    (void)deviceId;
    return ACLRT_SUCCESS;
}

AclError StubRtResetDevice(int32_t deviceId)
{
    ADAPTER_LOGD("Enter stub function of AclRtResetDevice.");
    (void)deviceId;
    return ACLRT_SUCCESS;
}

AclError StubRtCreateEvent(AclRtEvent* event)
{
    ADAPTER_LOGD("Enter stub function of AclRtCreateEvent.");
    (void)event;
    return ACLRT_SUCCESS;
}

AclError StubRtRecordEvent(AclRtEvent event, AclRtStream stream)
{
    ADAPTER_LOGD("Enter stub function of AclRtRecordEvent.");
    (void)event;
    (void)stream;
    return ACLRT_SUCCESS;
}

AclError StubRtCreateEventExWithFlag(AclRtEvent* event, uint32_t flag)
{
    ADAPTER_LOGD("Enter stub function of AclRtCreateEventExWithFlag.");
    (void)event;
    (void)flag;
    return ACLRT_SUCCESS;
}

AclError StubRtStreamWaitEvent(AclRtStream stream, AclRtEvent event)
{
    ADAPTER_LOGD("Enter stub function of AclRtStreamWaitEvent.");
    (void)stream;
    (void)event;
    return ACLRT_SUCCESS;
}

AclError StubRtGetStreamResLimit(AclRtStream stream, AclRtDevResLimitType type, uint32_t* value)
{
    ADAPTER_LOGD("Enter stub function of AclRtGetStreamResLimit.");
    (void)stream;
    (void)type;
    if (value != nullptr) {
        *value = 20;
    }
    return ACLRT_SUCCESS;
}

AclError StubRtGetStreamAttribute(AclRtStream stream, AclRtStreamAttr stmAttrType, AclRtStreamAttrValue* value)
{
    ADAPTER_LOGD("Enter stub function of AclRtGetStreamAttribute.");
    (void)stream;
    (void)stmAttrType;
    if (value != nullptr) {
        value->cacheOpInfoSwitch = 1;
    }
    return ACLRT_SUCCESS;
}

AclError StubRtCacheLastTaskOpInfo(const void* const infoPtr, size_t infoSize)
{
    ADAPTER_LOGD("Enter stub function of AclRtCacheLastTaskOpInfo.");
    (void)infoPtr;
    (void)infoSize;
    return ACLRT_SUCCESS;
}

AclError StubRtSetExceptionInfoCallback(AclRtExceptionInfoCallback callback)
{
    ADAPTER_LOGD("Enter stub function of AclRtSetExceptionInfoCallback.");
    (void)callback;
    return ACLRT_SUCCESS;
}

AclError StubRtCreateStream(AclRtStream* stream)
{
    ADAPTER_LOGD("Enter stub function of AclRtCreateStream.");
    (void)stream;
    return ACLRT_SUCCESS;
}

AclError StubRtSynchronizeStream(AclRtStream stream)
{
    ADAPTER_LOGD("Enter stub function of AclRtSynchronizeStream.");
    (void)stream;
    return ACLRT_SUCCESS;
}

AclError StubRtDestroyStream(AclRtStream stream)
{
    ADAPTER_LOGD("Enter stub function of AclRtDestroyStream.");
    (void)stream;
    return ACLRT_SUCCESS;
}

AclError StubRtBinaryLoadFromFile(const char* binPath, AclRtBinaryLoadOptions* options, AclRtBinHandle* binHandle)
{
    ADAPTER_LOGD("Enter stub function of AclRtBinaryLoadFromFile.");
    (void)binPath;
    (void)options;
    if (binHandle != nullptr) {
        *binHandle = nullptr;
    }
    return ACLRT_SUCCESS;
}

AclError StubRtBinaryGetFunction(AclRtBinHandle binHandle, const char* kernelName, AclRtFuncHandle* funcHandle)
{
    ADAPTER_LOGD("Enter stub function of AclRtBinaryGetFunction.");
    (void)binHandle;
    (void)kernelName;
    if (funcHandle != nullptr) {
        *funcHandle = nullptr;
    }
    return ACLRT_SUCCESS;
}

AclError StubMdlRICaptureGetInfo(AclRtStream stream, AclMdlRICaptureStatus* status, AclMdlRI* modelRI)
{
    ADAPTER_LOGD("Enter stub function of AclMdlRICaptureGetInfo.");
    (void)stream;
    (void)status;
    (void)modelRI;
    return ACLRT_SUCCESS;
}

AclError StubMdlRICaptureThreadExchangeMode(AclMdlRICaptureMode* mode)
{
    ADAPTER_LOGD("Enter stub function of AclMdlRICaptureThreadExchangeMode.");
    (void)mode;
    return ACLRT_SUCCESS;
}

AclError StubSysGetVersionStr(const char* pkgName, char* versionStr)
{
    ADAPTER_LOGD("Enter stub function of AclSysGetVersionStr.");
    (void)pkgName;
    if (versionStr != nullptr) {
        versionStr[0] = '\0';
    }
    return ACLRT_SUCCESS;
}
} // namespace npu::tile_fwk
