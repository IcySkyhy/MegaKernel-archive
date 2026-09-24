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
 * \file test_device_runner.cpp
 * \brief
 */

#include <regex>
#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "machine/runtime/runner/host_prof.h"
#include "machine/runtime/runner/device_perf.h"
#include "machine/runtime/runner/pmu_common.h"
#include "machine/utils/machine_ws_intf.h"
#define private public
#include "machine/device/dynamic/aicore_prof.h"
#ifdef private
#undef private
#endif
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "machine/device/dynamic/aicore_manager.h"
#include "machine/device/tilefwk/aicpu_common.h"
#include "machine/device/dynamic/device_sche_context.h"
#include "machine/runtime/launcher/device_launcher_types.h"

using namespace npu::tile_fwk;
class TestDeviceRunner : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestDeviceRunner, test_set_pmu_event)
{
    std::vector<int64_t> pmuEvtType;
    for (int i = 0; i < 9; i++) {
        setenv("PYPTO_PROF_PMU_EVENT_TYPE", std::to_string(i).c_str(), 1);
        npu::tile_fwk::PmuCommon::InitPmuEventType(ArchInfo::DAV_2201, pmuEvtType);
    }
    for (int i = 0; i < 9; i++) {
        setenv("PYPTO_PROF_PMU_EVENT_TYPE", std::to_string(i).c_str(), 1);
        npu::tile_fwk::PmuCommon::InitPmuEventType(ArchInfo::DAV_3510, pmuEvtType);
    }
}

TEST_F(TestDeviceRunner, test_ini_proflevel)
{
    npu::tile_fwk::dynamic::SchThreadStatus status;
    std::unique_ptr<npu::tile_fwk::dynamic::AiCoreManager>
        AiCoreManagerPtr = std::make_unique<npu::tile_fwk::dynamic::AiCoreManager>(status);
    AiCoreManagerPtr->aicNum_ = 0;
    AiCoreManagerPtr->aivNum_ = 1;
    AiCoreManagerPtr->aivStart_ = 0;
    AiCoreManagerPtr->aivEnd_ = 1;
    AiCoreManagerPtr->aicStart_ = 0;
    AiCoreManagerPtr->aicEnd_ = 0;
    AiCoreManagerPtr->aicpuIdx_ = 0;
    npu::tile_fwk::dynamic::AiCoreProf prof(*AiCoreManagerPtr);
    int64_t* oriRegAddrs_ = (int64_t*)malloc(sizeof(int64_t) * 1024 * 2);
    int64_t* regAddrs_ = oriRegAddrs_ + 1024;
    regAddrs_[0] = (int64_t)&regAddrs_[0];
    ToSubMachineConfig toSubMachineConfig;
    if (AdprofReportAdditionalInfo != nullptr) {
        AdprofReportAdditionalInfo(0, 0, 0);
    }
    toSubMachineConfig.profConfig.Add(ProfConfig::OFF);
    std::unique_ptr<DeviceArgs> devArgs = std::make_unique<DeviceArgs>();
    devArgs->toSubMachineConfig = toSubMachineConfig;
    prof.ProfInit(devArgs.get());
    toSubMachineConfig.profConfig.Remove(ProfConfig::OFF);
    toSubMachineConfig.profConfig.Add(ProfConfig::AICPU_FUNC);
    devArgs->toSubMachineConfig = toSubMachineConfig;
    prof.ProfInit(devArgs.get());
    toSubMachineConfig.profConfig.Remove(ProfConfig::AICPU_FUNC);
    toSubMachineConfig.profConfig.Add(ProfConfig::AICORE_TIME);
    devArgs->toSubMachineConfig = toSubMachineConfig;
    prof.ProfInit(devArgs.get());
    toSubMachineConfig.profConfig.Remove(ProfConfig::AICORE_TIME);
    toSubMachineConfig.profConfig.Add(ProfConfig::AICORE_PMU);
    devArgs->toSubMachineConfig = toSubMachineConfig;
    int64_t* pmuEventAddrs = (int64_t*)calloc(10, sizeof(int64_t));
    if (pmuEventAddrs == nullptr) {
        return;
    }
    devArgs->corePmuRegAddr = reinterpret_cast<uint64_t>(regAddrs_);
    devArgs->pmuEventAddr = reinterpret_cast<uint64_t>(pmuEventAddrs);
    prof.ProfInit(devArgs.get());
    prof.ProfStart();
    int32_t aicoreId = 0;
    TaskStat* taskStat = new TaskStat();
    taskStat->seqNo = 1;
    taskStat->subGraphId = 0;
    taskStat->taskId = 0;
    taskStat->execStart = 0;
    taskStat->execEnd = 1;
    taskStat->waitEventNum = 0;
    taskStat->setEventNum = 0;
    taskStat->perfDataBaseAddr = 0;
    taskStat->setEventAddr = 0;
    taskStat->waitEventAddr = 0;
    prof.ProfGetLog(aicoreId, taskStat);
    prof.profLevel_ = npu::tile_fwk::dynamic::PROF_LEVEL_FUNC_LOG_PMU;
    uint32_t ctrl0val = 0;
    prof.addrs_.ctrl0Addr = &ctrl0val;
    prof.ProfStop();
    delete taskStat;
    free(pmuEventAddrs);
    free(oriRegAddrs_);
}

TEST_F(TestDeviceRunner, test_create_proflevel)
{
    npu::tile_fwk::dynamic::SchThreadStatus status;
    std::unique_ptr<npu::tile_fwk::dynamic::AiCoreManager>
        AiCoreManagerPtr = std::make_unique<npu::tile_fwk::dynamic::AiCoreManager>(status);
    npu::tile_fwk::dynamic::AiCoreProf prof(*AiCoreManagerPtr);

    ProfConfig config0;
    EXPECT_TRUE(config0.Empty());
    EXPECT_EQ(npu::tile_fwk::dynamic::CreateProfLevel(config0), npu::tile_fwk::dynamic::PROF_LEVEL_OFF);
    ProfConfig config1;
    config1.Add(ProfConfig::AICORE_PMU);
    EXPECT_EQ(npu::tile_fwk::dynamic::CreateProfLevel(config1), npu::tile_fwk::dynamic::PROF_LEVEL_FUNC_LOG_PMU);
    EXPECT_FALSE(config1.Empty());
    ProfConfig config2;
    config2.Add(ProfConfig::AICORE_TIME);
    EXPECT_EQ(npu::tile_fwk::dynamic::CreateProfLevel(config2), npu::tile_fwk::dynamic::PROF_LEVEL_FUNC_LOG);
    ProfConfig config3;
    config3.Add(ProfConfig::AICPU_FUNC);
    EXPECT_EQ(npu::tile_fwk::dynamic::CreateProfLevel(config3), npu::tile_fwk::dynamic::PROF_LEVEL_FUNC);
    EXPECT_TRUE(config3.Contains(ProfConfig::AICPU_FUNC));

    ProfConfig config4 = config2 | config3;
    EXPECT_EQ(config4.value, 0x3);
    ProfConfig config5 = ProfConfig::AICPU_FUNC | ProfConfig::AICORE_TIME;
    EXPECT_TRUE(config5.Overlaps(ProfConfig::AICPU_FUNC));
    EXPECT_TRUE(config5.Overlaps(ProfConfig::AICORE_TIME));
    EXPECT_FALSE(config5.Overlaps(ProfConfig::AICORE_PMU));

    ProfConfig config6 = ProfConfig::AICPU_FUNC | ProfConfig::AICORE_TIME;
    config6.Remove(ProfConfig::AICPU_FUNC);
    EXPECT_EQ(config6.value, ProfConfig::AICORE_TIME);

    config6.Remove(ProfConfig::AICORE_TIME);
    EXPECT_EQ(config6.value, 0);

    ToSubMachineConfig config7;
    EXPECT_EQ(config7.profConfig.value, ProfConfig::OFF);
}

TEST_F(TestDeviceRunner, KernelLaunchInfo_Constructor)
{
    npu::tile_fwk::dynamic::KernelLaunchInfo info(nullptr, nullptr, nullptr, 24, 6);
    EXPECT_EQ(info.schedStream, nullptr);
    EXPECT_EQ(info.ctrlStream, nullptr);
    EXPECT_EQ(info.aicoreStream, nullptr);
    EXPECT_EQ(info.blockDim, 24u);
    EXPECT_EQ(info.aicpuNum, 6u);
    EXPECT_EQ(info.binHandle, nullptr);
    EXPECT_FALSE(info.isCaptureActivate);
}

TEST_F(TestDeviceRunner, HostProf_GetHostProfType_Default) { EXPECT_EQ(HostProf::GetInstance().GetHostProfType(), 0u); }

TEST_F(TestDeviceRunner, HostProf_SetProfFunction_Null) { HostProf::GetInstance().SetProfFunction(nullptr); }

TEST_F(TestDeviceRunner, DevicePerf_SetDebugEnable)
{
    setenv("ENABLE_AICORE_RESOLVE", "true", 1);
    ToSubMachineConfig machinConfig;
    DevicePerf::GetInstance().SetDebugEnable(machinConfig);
    unsetenv("ENABLE_AICORE_RESOLVE");
}

TEST_F(TestDeviceRunner, HostProf_ReportHostProfInfo_NoProf)
{
    HostProf::GetInstance().ReportHostProfInfo(nullptr, 100, 24, 1, false);
}

TEST_F(TestDeviceRunner, HostProf_ReportHostProfInfo_WithProf)
{
    HostProf::profSwitch_ = MSPF_TASK_TIME_L1_MASK | MSPF_TASK_TIME_L2_MASK;
    HostProf::profType_ = MSPF_COMMANDHANDLE_TYPE_START;
    HostProf::GetInstance().ReportHostProfInfo(nullptr, 100, 24, 1, false);
    HostProf::profSwitch_ = 0;
    HostProf::profType_ = 0;
}

TEST_F(TestDeviceRunner, HostProf_ReportHostProfInfo_WithCore)
{
    HostProf::profSwitch_ = MSPF_TASK_TIME_L1_MASK | MSPF_TASK_TIME_L2_MASK;
    HostProf::profType_ = MSPF_COMMANDHANDLE_TYPE_START;
    HostProf::GetInstance().ReportHostProfInfo(nullptr, 100, 24, 1, true);
    HostProf::profSwitch_ = 0;
    HostProf::profType_ = 0;
}

TEST_F(TestDeviceRunner, HostProf_ReportHostProfInfo_MixAic)
{
    HostProf::profSwitch_ = MSPF_TASK_TIME_L1_MASK | MSPF_TASK_TIME_L2_MASK;
    HostProf::profType_ = MSPF_COMMANDHANDLE_TYPE_START;
    HostProf::GetInstance().ReportHostProfInfo(nullptr, 100, 24, MSPF_GE_TASK_TYPE_MIX_AIC, false);
    HostProf::profSwitch_ = 0;
    HostProf::profType_ = 0;
}
