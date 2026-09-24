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
 * \file device_launcher_context.cpp
 * \brief
 */

#include "machine/runtime/context/device_launcher_context.h"
#include "adapter/api/runtime_capture_context.h"

namespace npu::tile_fwk {
DeviceLauncherContext& DeviceLauncherContext::Get()
{
    static DeviceLauncherContext context;
    return context;
}

void DeviceLauncherContext::SetCaptureMode(const bool captureMode)
{
    captureMode_ = captureMode;
    RuntimeCaptureContext::SetCaptureMode(captureMode);
}
} // namespace npu::tile_fwk
