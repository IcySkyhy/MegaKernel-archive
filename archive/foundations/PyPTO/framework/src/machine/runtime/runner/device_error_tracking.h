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
 * \file device_error_tracking.h
 * \brief
 */

#ifndef DEVICE_ERROR_TRACKING_H
#define DEVICE_ERROR_TRACKING_H

#include "adapter/api/acl_define.h"

namespace npu::tile_fwk {
const char* GetRetcodeMessage(int32_t retcode);
void PyPTOExceptionInfoCallBack(AclRtExceptionInfo* exceptionInfo);
void InitializeErrorCallback();
} // namespace npu::tile_fwk
#endif
