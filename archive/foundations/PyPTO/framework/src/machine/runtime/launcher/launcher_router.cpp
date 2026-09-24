/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file launcher_router.cpp
 * \brief runtime launch mode routing helpers
 */

#include "machine/runtime/launcher/launcher_router.h"

namespace npu::tile_fwk::dynamic {
LaunchMode LauncherRouter::ResolveByDebugMode(int64_t debugMode)
{
    switch (debugMode) {
        case CFG_DEBUG_ALL:
        case CFG_RUNTIME_DEBUG_VERIFY:
            return LaunchMode::EMULATION;
        case CFG_RUINTIME_DEBUG_AICORE_MODEL:
            return LaunchMode::AICORE_MODEL;
        default:
            return LaunchMode::DEVICE_RT;
    }
}

LaunchMode LauncherRouter::ResolveCurrent()
{
    auto debugMode = config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE);
    return ResolveByDebugMode(debugMode);
}

} // namespace npu::tile_fwk::dynamic
