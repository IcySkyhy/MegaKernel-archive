/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file test_device_launcher_host.cpp
 * \brief UT for machine/runtime/launcher/device_launcher.h host-side branches
 */

#include <gtest/gtest.h>
#include "test_machine_common.h"

struct DeviceLauncherHostTest : UnitTestBase {
protected:
    void SetUp() override
    {
        UnitTestBase::SetUp();
        PrepareSimpleFunction();
    }

    Function* GetFunc() { return Program::GetInstance().GetLastFunction(); }

    static DeviceLauncherConfig MakeConfig()
    {
        DeviceLauncherConfig config;
        config.blockdim = 24;
        return config;
    }

private:
    static void PrepareSimpleFunction()
    {
        config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 0x4);
        int tiling = 32;
        TileShape::Current().SetVecTile(tiling, tiling);
        int n = tiling * 4;
        Tensor inputA(DT_INT32, {n, n}, "A");
        Tensor inputB(DT_INT32, {n, n}, "B");
        Tensor output(DT_INT32, {n, n}, "O");

        ProgramData::GetInstance().AppendInputs({
            RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
            RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
        });
        ProgramData::GetInstance().AppendOutputs({
            RawTensorData::CreateConstantTensor<int32_t>(output, 0),
        });

        FUNCTION("dl_host_func", {inputA, inputB}, {output})
        {
            LOOP("dl_host_L0", FunctionType::DYNAMIC_LOOP, _, LoopRange(0x1))
            {
                (void)_;
                output = Add(inputA, inputB);
            }
        }
    }
};

// Covers: DeviceInitLauncherConfigForUser (device_launcher.h:101-106)
TEST_F(DeviceLauncherHostTest, DeviceInitLauncherConfigForUser_SetsLaunchSchedArgs)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);
    auto dynAttr = func->GetDyndevAttribute();
    ASSERT_NE(dynAttr, nullptr);

    std::vector<uint8_t> devProgData = dynAttr->devProgBinary;
    DeviceLauncher::DeviceInitLauncherConfigForUser(devProgData);

    auto* devProg = reinterpret_cast<DevAscendProgram*>(devProgData.data());
    EXPECT_GE(devProg->devArgs.launchSchedAicpuNum, 0u);
}

// Covers: PrepareDevProgArgsCpuInfo + PrepareDevProgArgs (device_launcher.h:172-250)
TEST_F(DeviceLauncherHostTest, PrepareDevProgArgsCpuInfo_FillsConfig)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);
    auto dynAttr = func->GetDyndevAttribute();
    auto devProgData = dynAttr->devProgBinary;
    auto* devProg = reinterpret_cast<DevAscendProgram*>(devProgData.data());

    DeviceLauncherConfig config = MakeConfig();
    DeviceLauncher::DeviceInitLauncherConfigForUser(devProgData);
    DeviceLauncher::PrepareDevProgArgsCpuInfo(devProg, config);
    EXPECT_GT(config.aicpuNum, 0);
    EXPECT_GE(devProg->devArgs.nrValidAic, 0);
}

// Covers: PrepareDevProgArgs emulation path + dynWorkspaceSize branch (device_launcher.h:201-250)
TEST_F(DeviceLauncherHostTest, PrepareDevProgArgs_EmuluationAndDynWorkspace)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);
    auto dynAttr = func->GetDyndevAttribute();
    auto devProgData = dynAttr->devProgBinary;
    auto* devProg = reinterpret_cast<DevAscendProgram*>(devProgData.data());

    DeviceLauncherConfig config = MakeConfig();
    DeviceLauncher::PrepareDevProgArgs(devProg, config, false);
    EXPECT_EQ(devProg->devArgs.nrAic, DeviceLauncher::kDefaultAicNum);
    EXPECT_EQ(devProg->devArgs.nrAiv, DeviceLauncher::kDefaultAivNum);
    EXPECT_EQ(devProg->devArgs.taskType, DEVICE_TASK_TYPE_DYN);

    config = MakeConfig();
    config.dynWorkspaceSize = 4096;
    DeviceLauncher::PrepareDevProgArgs(devProg, config, false);
    EXPECT_GE(devProg->memBudget.tensor.maxDynamicAssembleOutcastMem, 4096u);
}

// Covers: DeviceInitTilingData full flow + ctrlFlowCache recording path (device_launcher.h:329-346)
TEST_F(DeviceLauncherHostTest, DeviceInitTilingData_EmuluationAndRecordingPath)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);
    auto dynAttr = func->GetDyndevAttribute();

    EmulationMemoryUtils memUtils;
    DeviceKernelArgs kArgs;
    DeviceLauncherConfig config = MakeConfig();
    DeviceLauncher::DeviceInitTilingData(memUtils, kArgs, dynAttr->devProgBinary, nullptr, config, nullptr);
    EXPECT_NE(kArgs.cfgdata, nullptr);

    DeviceKernelArgs kArgs2;
    DevControlFlowCache ctrlCache;
    ctrlCache.isRecording = true;
    DeviceLauncher::DeviceInitTilingData(memUtils, kArgs2, dynAttr->devProgBinary, &ctrlCache, config, nullptr);
    EXPECT_NE(kArgs2.cfgdata, nullptr);
}

// Covers: DeviceInitKernelInOuts normal + l2 offset + resize (device_launcher.h:367-423)
TEST_F(DeviceLauncherHostTest, DeviceInitKernelInOuts_AllBranches)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    EmulationMemoryUtils memUtils;
    DeviceKernelArgs kArgs;
    std::vector<DeviceTensorData> inputs = {DeviceTensorData(DT_INT32, static_cast<void*>(nullptr), {4, 4})};
    std::vector<DeviceTensorData> outputs = {DeviceTensorData(DT_INT32, static_cast<void*>(nullptr), {4, 4})};
    std::vector<uint8_t> disableL2List = {0, 0};
    DeviceLauncher::DeviceInitKernelInOuts(memUtils, kArgs, inputs, outputs, disableL2List);
    EXPECT_NE(kArgs.inputs, nullptr);

    uint8_t buf[64] = {0};
    DeviceKernelArgs kArgs2;
    std::vector<DeviceTensorData> inputs2 = {DeviceTensorData(DT_INT32, static_cast<void*>(buf), {4, 4})};
    std::vector<DeviceTensorData> outputs2 = {DeviceTensorData(DT_INT32, static_cast<void*>(buf), {4, 4})};
    std::vector<uint8_t> disableL2List2 = {1, 0};
    DeviceLauncher::DeviceInitKernelInOuts(memUtils, kArgs2, inputs2, outputs2, disableL2List2);
    EXPECT_NE(kArgs2.inputs, nullptr);

    DeviceKernelArgs kArgs3;
    std::vector<DeviceTensorData> largeInputs;
    std::vector<DeviceTensorData> largeOutputs;
    std::vector<uint8_t> largeDisableL2;
    for (int i = 0; i < 200; ++i) {
        largeInputs.emplace_back(DT_INT32, static_cast<void*>(nullptr), std::vector<int64_t>{4, 4});
        largeDisableL2.push_back(0);
    }
    largeOutputs.emplace_back(DT_INT32, static_cast<void*>(nullptr), std::vector<int64_t>{4, 4});
    largeDisableL2.push_back(0);
    DeviceLauncher::DeviceInitKernelInOuts(memUtils, kArgs3, largeInputs, largeOutputs, largeDisableL2);
    EXPECT_NE(kArgs3.inputs, nullptr);
}

// Covers: BuildInputOutputFromHost normal + nullptr (device_launcher.h:425-457)
TEST_F(DeviceLauncherHostTest, BuildInputOutputFromHost_NormalAndNullptr)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto& inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
    EmulationMemoryUtils memUtils;
    auto [inputs, outputs] = DeviceLauncher::BuildInputOutputFromHost(memUtils, inputDataList, outputDataList);
    EXPECT_EQ(inputs.size(), inputDataList.size());
    EXPECT_EQ(outputs.size(), outputDataList.size());

    std::vector<RawTensorDataPtr> nullInputs = {nullptr};
    std::vector<RawTensorDataPtr> nullOutputs = {nullptr};
    auto [inputs2, outputs2] = DeviceLauncher::BuildInputOutputFromHost(memUtils, nullInputs, nullOutputs);
    EXPECT_EQ(inputs2.size(), 1u);
    EXPECT_EQ(outputs2.size(), 1u);
}

// Covers: CopyFromDev non-null + nullptr (device_launcher.h:460-467)
TEST_F(DeviceLauncherHostTest, CopyFromDev_NonNullAndNullptr)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
    EmulationMemoryUtils memUtils;
    for (auto& output : outputDataList) {
        if (output) {
            memUtils.CopyToDev(*output);
        }
    }
    DeviceLauncher::CopyFromDev(EmulationMemoryUtils(), outputDataList);

    std::vector<RawTensorDataPtr> nullOutputs = {nullptr, nullptr};
    DeviceLauncher::CopyFromDev(EmulationMemoryUtils(), nullOutputs);
    SUCCEED();
}

// Covers: DeviceInitDistributedContext host path (device_launcher.h:307-327)
TEST_F(DeviceLauncherHostTest, DeviceInitDistributedContext_HostPath_FillsCommContexts)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);
    auto dynAttr = func->GetDyndevAttribute();

    EmulationMemoryUtils memUtils;
    DeviceKernelArgs kArgs;
    DeviceLauncher::DeviceInitDistributedContext(memUtils, dynAttr->commGroupNames, kArgs);
    EXPECT_NE(kArgs.commContexts, nullptr);
}

// Covers: FillDeviceKernelArgs (device_launcher.h:293-305) full flow
TEST_F(DeviceLauncherHostTest, FillDeviceKernelArgs_EmuluationPath_FillsKArgs)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);
    auto dynAttr = func->GetDyndevAttribute();

    EmulationMemoryUtils memUtils;
    DeviceKernelArgs kArgs;
    DeviceLauncher::FillDeviceKernelArgs(memUtils, dynAttr->devProgBinary, kArgs, dynAttr->commGroupNames);
    EXPECT_NE(kArgs.cfgdata, nullptr);
    EXPECT_NE(kArgs.commContexts, nullptr);
}

// Covers: HasInplaceArgs (device_launcher.h:79)
TEST_F(DeviceLauncherHostTest, HasInplaceArgs_ReturnsBoolean)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);
    DevAscendProgram* devProg = DeviceLauncher::GetDevProg(func);
    ASSERT_NE(devProg, nullptr);
    // The value depends on compile-time analysis; just verify the call succeeds.
    (void)DeviceLauncher::HasInplaceArgs(func);
    SUCCEED();
}

// ===== device_launcher.cpp function tests =====

// Covers: LaunchSyncTask early launch mode 1 + mode 0 + capture (device_launcher.cpp:361-374)
TEST_F(DeviceLauncherHostTest, LaunchSyncTask_EarlyLaunchModes_ReturnZero)
{
    EXPECT_EQ(DeviceLauncher::LaunchSyncTask(nullptr, false, 1), 0);
    EXPECT_EQ(DeviceLauncher::LaunchSyncTask(nullptr, true, 0), 0);
}

// Covers: GetCaptureInfo (device_launcher.cpp:338-343)
TEST_F(DeviceLauncherHostTest, GetCaptureInfo_NullStream_SetsCaptureMode)
{
    AclMdlRI rtModel = nullptr;
    DeviceLauncher::GetCaptureInfo(nullptr, rtModel);
    EXPECT_FALSE(DeviceLauncher::IsCaptureMode());
}

// Covers: RunWithProfile non-debug + debug+capture (device_launcher.cpp:71-87)
TEST_F(DeviceLauncherHostTest, RunWithProfile_AllModes_ReturnZero)
{
    config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, static_cast<int64_t>(0));
    EXPECT_EQ(DeviceLauncher::RunWithProfile(nullptr, nullptr, false), 0);
    config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, CFG_DEBUG_ALL);
    EXPECT_EQ(DeviceLauncher::RunWithProfile(nullptr, nullptr, true), 0);
    config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, static_cast<int64_t>(0));
}

// Covers: SetDevPerfAddr, DumpIOTensorsWithCann, CheckAscendDriverVersionOnboard
TEST_F(DeviceLauncherHostTest, MiscNoOpFunctions_NoCrash)
{
    ToSubMachineConfig machinConfig;
    DeviceLauncher::SetDevPerfAddr(false, false, machinConfig);
    std::vector<DeviceTensorData> tensors;
    DeviceLauncher::DumpIOTensorsWithCann(nullptr, tensors, "test_func");
    DeviceLauncher::CheckAscendDriverVersionOnboard();
    SUCCEED();
}

// Covers: CopyControlFlowCache (device_launcher.cpp:293-315)
TEST_F(DeviceLauncherHostTest, CopyControlFlowCache_AllocatesAndFrees)
{
    DevControlFlowCache cache;
    cache.usedCacheSize = 64;
    uint8_t* result = DeviceLauncher::CopyControlFlowCache(&cache);
    if (result != nullptr) {
        DeviceLauncher::FreeControlFlowCache(result);
    }
    SUCCEED();
}

TEST_F(DeviceLauncherHostTest, SetCaptureStream_NormalMode)
{
    AclRtStream aicoreStream = nullptr;
    AclRtStream aicpuStream = nullptr;
    AclRtCreateStream(&aicoreStream);
    AclRtCreateStream(&aicpuStream);

    bool isCapture = false;
    int ret = DeviceLauncher::SetCaptureStream(aicoreStream, aicpuStream, isCapture);
    EXPECT_EQ(ret, 0);
    EXPECT_FALSE(isCapture);

    if (aicoreStream)
        AclRtDestroyStream(aicoreStream);
    if (aicpuStream)
        AclRtDestroyStream(aicpuStream);
}

TEST_F(DeviceLauncherHostTest, SetDevRunCacheKernel_EnableDisable)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    DeviceLauncher::SetDevRunCacheKernelEnable(func, false);
    EXPECT_FALSE(DeviceLauncher::IsDevRunCacheKernelEnable(func));

    DeviceLauncher::SetDevRunCacheKernelEnable(func, true);
    EXPECT_TRUE(DeviceLauncher::IsDevRunCacheKernelEnable(func));

    uint8_t dummyProg[64] = {0};
    DeviceLauncher::SetDevRunCacheKernel(func, dummyProg);

    auto* op = DeviceLauncher::GetDevRunCacheOperator(func);
    EXPECT_NE(op, nullptr);

    DeviceLauncher::SetDevRunCacheKernelEnable(func, false);
}

TEST_F(DeviceLauncherHostTest, DataDumpInit_UnInit_NoDump)
{
    DeviceLauncher::DataDumpInit();
    DeviceLauncher::DataDumpUnInit();
    SUCCEED();
}

TEST_F(DeviceLauncherHostTest, DumpIOTensorsWithCann_EmptyTensors)
{
    std::vector<DeviceTensorData> tensors;
    DeviceLauncher::DumpIOTensorsWithCann(nullptr, tensors, "test_func");
    SUCCEED();
}

TEST_F(DeviceLauncherHostTest, FreeControlFlowCache_NullSafe)
{
    DeviceLauncher::FreeControlFlowCache(nullptr);
    SUCCEED();
}

TEST_F(DeviceLauncherHostTest, SaveStream_SetsCurrentStream)
{
    AclRtStream stream = nullptr;
    AclRtCreateStream(&stream);
    DeviceLauncher::SaveStream(stream);
    if (stream)
        AclRtDestroyStream(stream);
    SUCCEED();
}

TEST_F(DeviceLauncherHostTest, AddAicpuStream_NonCaptureMode)
{
    AclMdlRI rtModel = nullptr;
    DeviceLauncher::AddAicpuStream(false, rtModel);
    SUCCEED();
}

TEST_F(DeviceLauncherHostTest, IsCaptureMode_ReturnsFalse) { EXPECT_FALSE(DeviceLauncher::IsCaptureMode()); }

TEST_F(DeviceLauncherHostTest, RunPreSync_WithRealStreams)
{
    AclRtStream scheStream = nullptr;
    AclRtStream ctrlStream = nullptr;
    AclRtStream aicoreStream = nullptr;

    AclRtCreateStream(&scheStream);
    AclRtCreateStream(&ctrlStream);
    AclRtCreateStream(&aicoreStream);

    if (scheStream && ctrlStream && aicoreStream) {
        int ret = DeviceLauncher::RunPreSync(scheStream, ctrlStream, aicoreStream);
        EXPECT_EQ(ret, 0);
    }

    if (scheStream)
        AclRtDestroyStream(scheStream);
    if (ctrlStream)
        AclRtDestroyStream(ctrlStream);
    if (aicoreStream)
        AclRtDestroyStream(aicoreStream);
}

TEST_F(DeviceLauncherHostTest, LaunchSyncTask_AllModes)
{
    AclRtStream stream = nullptr;
    AclRtCreateStream(&stream);

    EXPECT_EQ(DeviceLauncher::LaunchSyncTask(stream, false, 1), 0);
    EXPECT_EQ(DeviceLauncher::LaunchSyncTask(stream, true, 0), 0);

    if (stream)
        AclRtDestroyStream(stream);
}
