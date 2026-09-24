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
 * \file test_device_perf.cpp
 * \brief UT for machine/runtime/runner/device_perf.cpp
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/runner/device_perf.h"
#undef private
#include "interface/configs/config_manager.h"
#include "tilefwk/aicpu_common.h"

using namespace npu::tile_fwk;

TEST(DevicePerfTest, AllPerfOps_NoCrash)
{
    {
        DevicePerf perf;
    }

    DevicePerf perf;
    DeviceArgs args;
    args.nrAic = 2;
    args.nrAiv = 4;
    perf.args_ = args;
    EXPECT_EQ(perf.GetPerfDataSize(), 2u + 4u + 1u);

    perf.SyncProfData(false);
    perf.ResetMetrics(0);
    perf.ResetMetrics(999);
    perf.ReleasePerfData();
    EXPECT_TRUE(perf.perfData_.empty());

    perf.StopMachinePerfTraceDumpThread();
    perf.StartMachinePerfTraceDumpThread();
    EXPECT_FALSE(perf.dumpThread_.joinable());
}

TEST(DevicePerfTest, ResetMetrics_EmptyPerfData)
{
    DevicePerf perf;
    perf.ResetMetrics(0);
}

TEST(DevicePerfTest, ResetMetrics_IndexOutOfRange)
{
    DevicePerf perf;
    perf.perfData_.push_back(nullptr);
    perf.ResetMetrics(5);
}

TEST(DevicePerfTest, SyncProfData_DebugDisabled)
{
    DevicePerf perf;
    perf.SyncProfData(false);
}

TEST(DevicePerfTest, ReleasePerfData_NullPtrs)
{
    DevicePerf perf;
    perf.perfData_.push_back(nullptr);
    perf.perfData_.push_back(nullptr);
    perf.ReleasePerfData();
    EXPECT_TRUE(perf.perfData_.empty());
}

TEST(DevicePerfTest, GetPerfDataSize_Default)
{
    DevicePerf perf;
    DeviceArgs args{};
    args.nrAic = 0;
    args.nrAiv = 0;
    perf.args_ = args;
    EXPECT_EQ(perf.GetPerfDataSize(), 1u);
}

TEST(DevicePerfTest, StartMachinePerfTraceDumpThread_ZeroAddr)
{
    DevicePerf perf;
    perf.args_.aicpuPerfAddr = 0;
    perf.StartMachinePerfTraceDumpThread();
    EXPECT_FALSE(perf.dumpThread_.joinable());
}

TEST(DevicePerfTest, ResetPerData_WithPerfData)
{
    DevicePerf perf;
    DeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    perf.args_ = args;
    perf.perfData_.push_back(nullptr);
    perf.perfData_.push_back(nullptr);
    perf.ResetPerData();
}

TEST(DevicePerfTest, RunPrepare_DebugMode)
{
    DevicePerf perf;
    DeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    args.sharedBuffer = 0x1000;
    perf.args_ = args;
    perf.perfData_.push_back(nullptr);
    bool result = perf.RunPrepare();
    EXPECT_TRUE(result);
}

TEST(DevicePerfTest, ResetMetrics_WithAicpuPerfAddr)
{
    DevicePerf perf;
    DeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    args.aicpuPerfAddr = 0x2000;
    perf.args_ = args;
    perf.perfData_.push_back(nullptr);
    perf.ResetMetrics(0);
    EXPECT_TRUE(perf.isPerfDataInited_);
}

TEST(DevicePerfTest, ResetMetrics_WithAicpuPerfAddr_AlreadyInited)
{
    DevicePerf perf;
    DeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    args.aicpuPerfAddr = 0x2000;
    perf.args_ = args;
    perf.perfData_.push_back(nullptr);
    perf.isPerfDataInited_ = true;
    perf.ResetMetrics(0);
}

TEST(DevicePerfTest, ResetMetrics_WithoutAicpuPerfAddr)
{
    DevicePerf perf;
    DeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    args.aicpuPerfAddr = 0;
    perf.args_ = args;
    perf.perfData_.push_back(nullptr);
    perf.ResetMetrics(0);
}
