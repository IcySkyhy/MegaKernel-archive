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
 * \file acl_stubs.h
 * \brief
 */

#pragma once

#include "adapter/api/acl_define.h"

namespace npu::tile_fwk {
AclError StubAclInit(const char* configPath);
AclError StubAclFinalize();
AclError StubRtMemcpy(void* dst, size_t destMax, const void* src, size_t count, AclRtMemcpyKind kind);
AclError StubRtMalloc(void** devPtr, size_t size, AclRtMemMallocPolicy policy);
AclError StubRtFree(void* devPtr);
AclError StubRtSetDevice(int32_t deviceId);
AclError StubRtResetDevice(int32_t deviceId);
AclError StubRtCreateEvent(AclRtEvent* event);
AclError StubRtRecordEvent(AclRtEvent event, AclRtStream stream);
AclError StubRtCreateEventExWithFlag(AclRtEvent* event, uint32_t flag);
AclError StubRtStreamWaitEvent(AclRtStream stream, AclRtEvent event);
AclError StubRtGetStreamResLimit(AclRtStream stream, AclRtDevResLimitType type, uint32_t* value);
AclError StubRtGetStreamAttribute(AclRtStream stream, AclRtStreamAttr stmAttrType, AclRtStreamAttrValue* value);
AclError StubRtCacheLastTaskOpInfo(const void* const infoPtr, size_t infoSize);
AclError StubRtSetExceptionInfoCallback(AclRtExceptionInfoCallback callback);
AclError StubRtCreateStream(AclRtStream* stream);
AclError StubRtSynchronizeStream(AclRtStream stream);
AclError StubRtDestroyStream(AclRtStream stream);
AclError StubRtBinaryLoadFromFile(const char* binPath, AclRtBinaryLoadOptions* options, AclRtBinHandle* binHandle);
AclError StubRtBinaryGetFunction(AclRtBinHandle binHandle, const char* kernelName, AclRtFuncHandle* funcHandle);
AclError StubMdlRICaptureGetInfo(AclRtStream stream, AclMdlRICaptureStatus* status, AclMdlRI* modelRI);
AclError StubMdlRICaptureThreadExchangeMode(AclMdlRICaptureMode* mode);
AclError StubSysGetVersionStr(const char* pkgName, char* versionStr);
} // namespace npu::tile_fwk
