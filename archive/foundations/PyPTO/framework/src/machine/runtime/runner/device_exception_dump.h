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
 * \file device_exception_dump.h
 * \brief
 */

#ifndef DEVICE_EXCEPTION_DUMP_H
#define DEVICE_EXCEPTION_DUMP_H

#include "adapter/api/adump_api.h"
#include "adapter/api/acl_define.h"
#include "adapter/api/runtime_api.h"

namespace npu::tile_fwk::dynamic {
int32_t AdumpRegExceptionDump();
int32_t ExceptionDumpCallBack(AclRtExceptionInfo* exceptionInfo, AdxExceptionDumpInfo* exceptionDumpInfo,
                              uint32_t exceptionDumpSize, uint32_t* exceptionDumpRealSize, AdxExceptionDumpMode* mode);
} // namespace npu::tile_fwk::dynamic
#endif // exception Dump h
