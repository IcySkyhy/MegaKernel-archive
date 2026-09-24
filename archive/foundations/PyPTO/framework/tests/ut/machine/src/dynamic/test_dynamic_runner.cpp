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
 * \file test_dynamic_runner.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include <cstdlib>
#include "machine/runtime/runner/kernel_binary.h"
#include <memory>
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/runner/host_prof.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "machine/device/machine_interface/pypto_aicpu_interface.h"
#include "machine/utils/machine_ws_intf.h"
#include "interface/program/program.h"
#include "utils/file_utils.h"
#include "tilefwk/aicpu_common.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/runtime/runner/dump_device_perf.h"
#define private public
using namespace npu::tile_fwk;

extern "C" uint32_t DynPyptoKernelServerNull(void* targ);
extern "C" uint32_t DynTileFwkBackendKernelServer(void* targ);
extern "C" uint32_t StaticTileFwkBackendKernelServer(void* targ);
extern "C" int DynTileFwkBackendKernelServerInit(void* targ);
class TestDynamicDeviceRunner : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

// 必须在加载 pypto server .so 的用例之前执行：ExecuteFunc 在符号未就绪时返回非 0，覆盖 pypto_aicpu_interface.cpp 中
// DEV_ERROR 分支。
TEST_F(TestDynamicDeviceRunner, DynPyptoKernelServer_ReturnsErrorWhenKernelNotLoaded)
{
    EXPECT_EQ(DynPyptoKernelServer(nullptr), 1U);
}

TEST_F(TestDynamicDeviceRunner, DynPyptoKernelServerInit_ReturnsErrorWhenKernelNotLoaded)
{
    EXPECT_EQ(DynPyptoKernelServerInit(nullptr), 1U);
}

TEST_F(TestDynamicDeviceRunner, TestInitArgs)
{
    [[maybe_unused]] DeviceArgs args;
    args.nrAic = 2;
    args.nrAiv = 2;
    args.nrValidAic = args.nrAic;
    npu::tile_fwk::DevicePerf::GetInstance().SyncProfData(false);
}

TEST_F(TestDynamicDeviceRunner, test_pypto_kernel_server_null)
{
    DeviceKernelArgs pyptoKernelArgs;
    DeviceArgs devKernelArgs;
    devKernelArgs.aicpuSoLen = 2;
    pyptoKernelArgs.cfgdata = static_cast<int64_t*>(static_cast<void*>(&devKernelArgs));
    auto ret = DynPyptoKernelServerNull(&pyptoKernelArgs);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestDynamicDeviceRunner, test_dump_device_perf)
{
    setenv("DUMP_DEVICE_PERF", "true", 1);
    DeviceArgs devKernelArgs;
    config::SetOptionsNg<int64_t>("debug.runtime_debug_mode", 1);
    npu::tile_fwk::dynamic::KernelBinary::InitMetaData(devKernelArgs);
    devKernelArgs.nrAic = 1;
    devKernelArgs.nrAiv = 2;
    devKernelArgs.nrValidAic = 1;
    devKernelArgs.nrAicpu = 3;
    std::vector<void*> perfData;
    constexpr uint64_t AICPU_NUM_OF_RUN_AICPU_TASKS = 1;
    const uint64_t perfDataNum = devKernelArgs.nrAic + devKernelArgs.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS;
    std::vector<std::vector<uint8_t>> metricStorage(perfDataNum, std::vector<uint8_t>(PERF_DATA_TOTAL_SIZE, 0));
    Metrics* metr = reinterpret_cast<Metrics*>(metricStorage[0].data());
    TaskStat taskStat{};
    taskStat.execEnd = 1;
    metr->taskCount = 1;
    metr->tasks[0] = taskStat;
    metr->aicoreDevTaskInfo[0].cnt = 1;
    metr->aicoreDevTaskInfo[0].aicoreEveryDevTypeTimeStamp[0].type = 0;
    metr->aicoreDevTaskInfo[0].aicoreEveryDevTypeTimeStamp[0].devTaskIdx = 0;
    metr->aicoreDevTaskInfo[0].aicoreEveryDevTypeTimeStamp[0].aicoreDevTimeStamp = 1;
    metr->turnNum = 1;

    auto aicpuMetPer = std::make_unique<MetricPerf>();
    aicpuMetPer->devTaskPerfs[0][0].timeStamp[0] = 1;
    aicpuMetPer->devTaskPerfs[1][0].timeStamp[0] = 2;
    aicpuMetPer->devTaskPerfs[2][0].timeStamp[0] = 3;
    devKernelArgs.aicpuPerfAddr = npu::tile_fwk::dynamic::PtrToValue(static_cast<void*>(aicpuMetPer.get()));

    for (uint64_t i = 0; i < perfDataNum; i++) {
        perfData.push_back(static_cast<void*>(metricStorage[i].data()));
    }
    npu::tile_fwk::dynamic::DumpAicoreTaskExectInfo(devKernelArgs, perfData);
    std::string jsonPath = npu::tile_fwk::config::LogTopFolder() + "/tilefwk_L1_prof_data.json";
    EXPECT_EQ(IsPathExist(jsonPath), true);
    setenv("DUMP_DEVICE_PERF", "true", 1);
    npu::tile_fwk::dynamic::DumpDevTaskPerfData(devKernelArgs, perfData, true);
    jsonPath = npu::tile_fwk::config::LogTopFolder() + "/machine_runtime_operator_trace.json";
    unsetenv("DUMP_DEVICE_PERF");
    EXPECT_EQ(IsPathExist(jsonPath), false);
}

TEST_F(TestDynamicDeviceRunner, test_launch_init)
{
    DeviceKernelArgs pyptoKernelArgs;
    DeviceArgs devKernelArgs;
    devKernelArgs.aicpuPerfAddr = 1;
    pyptoKernelArgs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
    pyptoKernelArgs.cfgdata = static_cast<int64_t*>(static_cast<void*>(&devKernelArgs));
    auto ret = DynTileFwkBackendKernelServer(&pyptoKernelArgs);
    EXPECT_NE(ret, 0);
}

TEST_F(TestDynamicDeviceRunner, test_launch_init_server)
{
    auto ret = DynTileFwkBackendKernelServerInit(nullptr);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestDynamicDeviceRunner, test_static) { EXPECT_EQ(StaticTileFwkBackendKernelServer(nullptr), 0); }

TEST_F(TestDynamicDeviceRunner, DynPyptoKernelServerNull_RejectsNullArgs)
{
    EXPECT_EQ(DynPyptoKernelServerNull(nullptr), 1U);
}
