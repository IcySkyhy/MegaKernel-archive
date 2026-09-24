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
 * \file acl_api.h
 * \brief
 */

#pragma once

#include "adapter/api/acl_define.h"

namespace npu::tile_fwk {
AclError AclInit(const char* configPath);
AclError AclFinalize();
AclError AclRtMemcpy(void* dst, size_t destMax, const void* src, size_t count, AclRtMemcpyKind kind);
AclError AclRtSetDevice(int32_t deviceId);
AclError AclRtResetDevice(int32_t deviceId);
AclError AclRtCreateEvent(AclRtEvent* event);
AclError AclRtRecordEvent(AclRtEvent event, AclRtStream stream);
AclError AclRtCreateEventExWithFlag(AclRtEvent* event, uint32_t flag);
AclError AclRtStreamWaitEvent(AclRtStream stream, AclRtEvent event);
AclError AclRtGetStreamResLimit(AclRtStream stream, AclRtDevResLimitType type, uint32_t* value);
AclError AclRtGetStreamAttribute(AclRtStream stream, AclRtStreamAttr stmAttrType, AclRtStreamAttrValue* value);
AclError AclRtCacheLastTaskOpInfo(const void* const infoPtr, size_t infoSize);
AclError AclRtSetExceptionInfoCallback(AclRtExceptionInfoCallback callback);
AclError AclMdlRICaptureGetInfo(AclRtStream stream, AclMdlRICaptureStatus* status, AclMdlRI* modelRI);
AclError AclMdlRICaptureThreadExchangeMode(AclMdlRICaptureMode* mode);
AclError AclRtMalloc(void** devPtr, size_t size, AclRtMemMallocPolicy policy);
AclError AclRtFree(void* devPtr);
AclError AclRtCreateStream(AclRtStream* stream);
AclError AclRtSynchronizeStream(AclRtStream stream);
AclError AclRtDestroyStream(AclRtStream stream);
AclError AclRtBinaryLoadFromFile(const char* binPath, AclRtBinaryLoadOptions* options, AclRtBinHandle* binHandle);
AclError AclRtBinaryGetFunction(AclRtBinHandle binHandle, const char* kernelName, AclRtFuncHandle* funcHandle);
AclError AclSysGetVersionStr(const char* pkgName, char* versionStr);
} // namespace npu::tile_fwk
