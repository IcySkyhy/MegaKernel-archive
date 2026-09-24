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
 * \file launcher_router.h
 * \brief runtime launch mode routing helpers
 */

#pragma once

#include <cstdint>
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk::dynamic {
enum class LaunchMode {
    DEVICE_RT = 0,
    EMULATION = 1,
    AICORE_MODEL = 2,
};

class LauncherRouter {
public:
    static LaunchMode ResolveByDebugMode(int64_t debugMode);
    static LaunchMode ResolveCurrent();
};
} // namespace npu::tile_fwk::dynamic
