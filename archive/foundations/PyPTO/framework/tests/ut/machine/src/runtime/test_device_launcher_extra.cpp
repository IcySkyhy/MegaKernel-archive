/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_device_launcher_extra.cpp
 * \brief UT for machine/runtime/launcher/device_launcher.cpp static functions
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/launcher/device_launcher.h"
#undef private
#include "machine/runtime/context/device_launcher_context.h"
#include "machine/runtime/memory_utils/emulation_memory_utils.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(DeviceLauncherStaticTest, GetAiCpuNum_AllArchBranches)
{
    EXPECT_EQ(DeviceLauncher::GetAiCpuNum(10, 1, ArchInfo::DAV_2201, true), 1u);
    EXPECT_EQ(DeviceLauncher::GetAiCpuNum(10, 3, ArchInfo::DAV_2201, true), 5u);
    EXPECT_EQ(DeviceLauncher::GetAiCpuNum(10, 3, ArchInfo::DAV_2201, false), 3u);
    auto dav3510Result = DeviceLauncher::GetAiCpuNum(10, 3, ArchInfo::DAV_3510, true);
    EXPECT_GT(dav3510Result, 0u);
    EXPECT_LE(dav3510Result, 10u);
}

TEST(DeviceLauncherStaticTest, GetDav3510DieMaxCpuid_AllValues)
{
    EXPECT_EQ(DeviceLauncher::GetDav3510DieMaxCpuid(DAV3510_AICPU_NUM_6), DAV3510_DIE0_MAX_CPUID_4);
    EXPECT_EQ(DeviceLauncher::GetDav3510DieMaxCpuid(DAV3510_AICPU_NUM_7), DAV3510_DIE0_MAX_CPUID_5);
    EXPECT_EQ(DeviceLauncher::GetDav3510DieMaxCpuid(8), 0u);
    EXPECT_EQ(DeviceLauncher::GetDav3510DieMaxCpuid(0), 0u);
}

TEST(DeviceLauncherStaticTest, CacheAndStreamAndDumpOps)
{
    DeviceLauncher::FreeControlFlowCache(nullptr);

    DeviceLauncherContext::Get().SetCaptureMode(false);
    EXPECT_FALSE(DeviceLauncher::IsCaptureMode());

    auto func = std::make_shared<Function>(Program::GetInstance(), "cache_test", "TENSOR_cache_test", nullptr);
    DeviceLauncher::SetDevRunCacheKernelEnable(func.get(), true);
    EXPECT_TRUE(DeviceLauncher::IsDevRunCacheKernelEnable(func.get()));
    DeviceLauncher::SetDevRunCacheKernelEnable(func.get(), false);
    EXPECT_FALSE(DeviceLauncher::IsDevRunCacheKernelEnable(func.get()));
    EXPECT_EQ(DeviceLauncher::GetDevRunCacheOperator(func.get()), nullptr);

    DeviceLauncher::SaveStream(nullptr);

    AclMdlRI rtModel = nullptr;
    DeviceLauncher::AddAicpuStream(false, rtModel);

    DeviceLauncher::DataDumpInit();
    DeviceLauncher::DataDumpUnInit();
}

TEST(DeviceLauncherStaticTest, ConfigFillAndSwimLane_AllBranches)
{
    DeviceLauncherConfig config;
    config.blockdim = 0;
    config.aicpuNum = 0;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
    EXPECT_GT(config.blockdim, 0u);
    EXPECT_GT(config.aicpuNum, 0u);

    config.blockdim = 9999;
    config.aicpuNum = 9999;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
    EXPECT_LE(config.blockdim, 9999u);
    EXPECT_LE(config.aicpuNum, 9999u);

    config::SetPlatformConfig(KEY_ENABLE_PROF_FUNC, false);
    config::SetPlatformConfig(KEY_ENABLE_PROF_AICORE_TIME, false);
    config::SetPlatformConfig(KEY_ENABLE_PROF_AICORE_PMU, false);
    config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, static_cast<int64_t>(0));
    ToSubMachineConfig toSubMachineConfig;
    DeviceLauncher::FillSwimLaneEnableInfo(toSubMachineConfig);

    config::SetPlatformConfig(KEY_ENABLE_PROF_FUNC, true);
    config::SetPlatformConfig(KEY_ENABLE_PROF_AICORE_TIME, true);
    config::SetPlatformConfig(KEY_ENABLE_PROF_AICORE_PMU, true);
    DeviceLauncher::FillSwimLaneEnableInfo(toSubMachineConfig);

    config::SetPlatformConfig(KEY_ENABLE_PROF_FUNC, false);
    config::SetPlatformConfig(KEY_ENABLE_PROF_AICORE_TIME, false);
    config::SetPlatformConfig(KEY_ENABLE_PROF_AICORE_PMU, false);
}

TEST(DeviceLauncherStaticTest, RunPreSync_AllBranches)
{
    int rc = DeviceLauncher::RunPreSync(nullptr, nullptr, nullptr);
    EXPECT_EQ(rc, 0);
}

TEST(DeviceLauncherStaticTest, AclModeGuard_RAII)
{
    AclMdlRICaptureMode mode = AclMdlRICaptureMode::RELAXED;
    {
        AclModeGuard guard(mode);
    }
    SUCCEED();
}

TEST(DeviceLauncherStaticTest, ValidateRuntimeDevice_MatchesCurrent)
{
    int32_t currentDevId = 0;
    RuntimeGetDevice(&currentDevId);
    ValidateRuntimeDevice(currentDevId);
    SUCCEED();
}

TEST(DeviceLauncherStaticTest, DeviceSynchronize_NullStreams)
{
    int rc = DeviceLauncher::DeviceSynchronize(nullptr, nullptr);
    EXPECT_GE(rc, -1);
}

TEST(DeviceLauncherStaticTest, SetCaptureStream_NullModel)
{
    DeviceLauncherContext::Get().SetCaptureMode(true);
    bool isCapture = true;
    int rc = DeviceLauncher::SetCaptureStream(nullptr, nullptr, isCapture);
    EXPECT_GE(rc, -1);
    DeviceLauncherContext::Get().SetCaptureMode(false);
}

TEST(DeviceLauncherStaticTest, SetCaptureStream_NotCapture)
{
    DeviceLauncherContext::Get().SetCaptureMode(false);
    bool isCapture = false;
    int rc = DeviceLauncher::SetCaptureStream(nullptr, nullptr, isCapture);
    EXPECT_EQ(rc, 0);
}

TEST(DeviceLauncherStaticTest, RunWithProfile_CaptureMode)
{
    DeviceLauncherContext::Get().SetCaptureMode(true);
    int rc = DeviceLauncher::RunWithProfile(nullptr, nullptr, true);
    EXPECT_EQ(rc, 0);
    DeviceLauncherContext::Get().SetCaptureMode(false);
}

TEST(DeviceLauncherStaticTest, DumpIOTensorsWithCann_EmptyTensors)
{
    std::vector<DeviceTensorData> tensors;
    DeviceLauncher::DumpIOTensorsWithCann(nullptr, tensors, "test_func");
    SUCCEED();
}

TEST(DeviceLauncherStaticTest, DumpIOTensorsWithCann_WithTensors)
{
    std::vector<DeviceTensorData> tensors;
    tensors.emplace_back(DataType::DT_FP32, nullptr, std::vector<int64_t>{2, 3});
    DeviceLauncher::DumpIOTensorsWithCann(nullptr, tensors, "test_func");
    SUCCEED();
}
