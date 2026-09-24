/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "machine/runtime/launcher/launcher_router.h"
#include "interface/configs/config_manager_ng.h"

using namespace npu::tile_fwk::dynamic;
using namespace npu::tile_fwk;

TEST(LauncherRouterTest, ResolveByDebugMode_AllBranches)
{
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(CFG_DEBUG_ALL), LaunchMode::EMULATION);
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(CFG_RUNTIME_DEBUG_VERIFY), LaunchMode::EMULATION);
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(CFG_RUINTIME_DEBUG_AICORE_MODEL), LaunchMode::AICORE_MODEL);
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(0), LaunchMode::DEVICE_RT);
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(-1), LaunchMode::DEVICE_RT);
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(99), LaunchMode::DEVICE_RT);

    auto mode = LauncherRouter::ResolveCurrent();
    EXPECT_TRUE(mode == LaunchMode::DEVICE_RT || mode == LaunchMode::EMULATION || mode == LaunchMode::AICORE_MODEL);
}
