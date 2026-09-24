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
 * \file test_runtime_context.cpp
 * \brief UT for machine/runtime/context (device_launcher_context) and launcher_router
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/context/device_launcher_context.h"
#undef private
#include "machine/runtime/launcher/launcher_router.h"
#include "interface/configs/config_manager.h"
#include "interface/program/program.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(RuntimeContextTest, DeviceLauncherContextAndLauncherRouter)
{
    auto& ctx1 = DeviceLauncherContext::Get();
    auto& ctx2 = DeviceLauncherContext::Get();
    EXPECT_EQ(&ctx1, &ctx2);

    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, false);
    ctx1.Initialize();
    EXPECT_TRUE(config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, false));
    ctx1.Finalize();
    EXPECT_FALSE(config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, false));

    ctx1.SetCaptureMode(true);
    EXPECT_TRUE(ctx1.IsCaptureMode());
    ctx1.SetCaptureMode(false);
    EXPECT_FALSE(ctx1.IsCaptureMode());

    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(CFG_DEBUG_ALL), LaunchMode::EMULATION);
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(CFG_RUNTIME_DEBUG_VERIFY), LaunchMode::EMULATION);
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(CFG_RUINTIME_DEBUG_AICORE_MODEL), LaunchMode::AICORE_MODEL);
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(0), LaunchMode::DEVICE_RT);
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(999), LaunchMode::DEVICE_RT);

    config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, CFG_DEBUG_ALL);
    EXPECT_EQ(LauncherRouter::ResolveCurrent(), LaunchMode::EMULATION);
    config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, CFG_RUINTIME_DEBUG_AICORE_MODEL);
    EXPECT_EQ(LauncherRouter::ResolveCurrent(), LaunchMode::AICORE_MODEL);
    config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, static_cast<int64_t>(0));
    EXPECT_EQ(LauncherRouter::ResolveCurrent(), LaunchMode::DEVICE_RT);
}
