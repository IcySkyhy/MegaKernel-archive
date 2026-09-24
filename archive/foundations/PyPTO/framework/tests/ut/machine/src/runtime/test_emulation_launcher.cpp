/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_emulation_launcher.cpp
 * \brief Covers functions in emulation_launcher.cpp NOT already covered by test_control_flow.cpp
 *        New coverage: EmulationLaunchDeviceTensorData, EmulationLaunchOnceWithHostTensorData,
 *                      EmulationBuildControlFlowCache, CreateHostCtrlFlowCache,
 *                      BuildControlFlowCacheWithEmulationTensorData
 */

#include <gtest/gtest.h>
#include "test_machine_common.h"

struct EmulationLauncherTest : UnitTestBase {
protected:
    void SetUp() override
    {
        UnitTestBase::SetUp();
        PrepareSimpleFunction();
    }

    Function* GetFunc() { return Program::GetInstance().GetLastFunction(); }

    std::tuple<std::vector<DeviceTensorData>, std::vector<DeviceTensorData>, EmulationMemoryUtils> BuildIO()
    {
        auto& inputDataList = ProgramData::GetInstance().GetInputDataList();
        auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
        EmulationMemoryUtils memUtils;
        std::vector<DeviceTensorData> inputDeviceDataList;
        std::vector<DeviceTensorData> outputDeviceDataList;
        std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(
            memUtils, inputDataList, outputDataList);
        return {std::move(inputDeviceDataList), std::move(outputDeviceDataList), std::move(memUtils)};
    }

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

        FUNCTION("emu_extra_func", {inputA, inputB}, {output})
        {
            LOOP("emu_extra_L0", FunctionType::DYNAMIC_LOOP, _, LoopRange(0x1))
            {
                (void)_;
                output = Add(inputA, inputB);
            }
        }
    }
};

// Covers: EmulationLaunchDeviceTensorData -> toHostTensorData -> FreeHostTensorData
//        EmulationLaunchOnceWithHostTensorData (called internally)
TEST_F(EmulationLauncherTest, EmulationLaunchDeviceTensorData_WithDeviceTensorData)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto [inputDeviceDataList, outputDeviceDataList, memUtils] = BuildIO();
    EXPECT_EQ(0, EmulationLauncher::EmulationLaunchDeviceTensorData(func, inputDeviceDataList, outputDeviceDataList,
                                                                    MakeConfig(), nullptr));
}

// Covers: EmulationLaunchOnceWithHostTensorData directly
TEST_F(EmulationLauncherTest, EmulationLaunchOnceWithHostTensorData_DirectCall)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto [inputDeviceDataList, outputDeviceDataList, memUtils] = BuildIO();
    EXPECT_EQ(0, EmulationLauncher::EmulationLaunchOnceWithHostTensorData(
                     func, inputDeviceDataList, outputDeviceDataList, nullptr, memUtils, MakeConfig()));
}

// Covers: EmulationBuildControlFlowCache (standalone, called by BuildControlFlowCacheWithEmulationTensorData)
//        CreateHostCtrlFlowCache
//        BuildControlFlowCacheWithEmulationTensorData
TEST_F(EmulationLauncherTest, BuildControlFlowCacheWithEmulationTensorData_CoversCreateAndBuild)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto [inputDeviceDataList, outputDeviceDataList, memUtils] = BuildIO();
    DevControlFlowCache* outCtrlFlowCache = nullptr;
    EXPECT_EQ(0,
              EmulationLauncher::BuildControlFlowCacheWithEmulationTensorData(
                  func, inputDeviceDataList, outputDeviceDataList, nullptr, &outCtrlFlowCache, memUtils, MakeConfig()));
    EXPECT_NE(outCtrlFlowCache, nullptr);
}

// Covers: CreateHostCtrlFlowCache directly
TEST_F(EmulationLauncherTest, CreateHostCtrlFlowCache_DirectCall)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    EmulationMemoryUtils memUtils;
    DevAscendProgram* devProg = DeviceLauncher::GetDevProg(func);
    ASSERT_NE(devProg, nullptr);

    DevControlFlowCache* cache = EmulationLauncher::CreateHostCtrlFlowCache(devProg, func, memUtils);
    EXPECT_NE(cache, nullptr);
}

// Covers: EmulationBuildControlFlowCache standalone with real IO data
TEST_F(EmulationLauncherTest, EmulationBuildControlFlowCache_WithRealIO)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto [inputDeviceDataList, outputDeviceDataList, memUtils] = BuildIO();
    DeviceKernelArgs kArgs;
    auto dynAttr = func->GetDyndevAttribute();
    auto config = MakeConfig();
    DeviceLauncher::DeviceInitDistributedContext(memUtils, dynAttr->commGroupNames, kArgs);
    DeviceLauncher::DeviceInitTilingData(memUtils, kArgs, dynAttr->devProgBinary, nullptr, config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(memUtils, kArgs, inputDeviceDataList, outputDeviceDataList,
                                           dynAttr->disableL2List);

    EXPECT_EQ(0, EmulationLauncher::EmulationBuildControlFlowCache(kArgs));
}

// Covers: EmulationLaunchDeviceTensorData with real IO (toHostTensorData non-empty path)
TEST_F(EmulationLauncherTest, EmulationLaunchDeviceTensorData_WithRealIO)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto [inputDeviceDataList, outputDeviceDataList, memUtils] = BuildIO();
    EXPECT_EQ(0, EmulationLauncher::EmulationLaunchDeviceTensorData(func, inputDeviceDataList, outputDeviceDataList,
                                                                    MakeConfig(), nullptr));
}

TEST_F(EmulationLauncherTest, EmulationRunOnce_WithInputCtrlCache)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto [inputDeviceDataList, outputDeviceDataList, memUtils] = BuildIO();
    DevControlFlowCache* outCtrlFlowCache = nullptr;
    EXPECT_EQ(0,
              EmulationLauncher::BuildControlFlowCacheWithEmulationTensorData(
                  func, inputDeviceDataList, outputDeviceDataList, nullptr, &outCtrlFlowCache, memUtils, MakeConfig()));
    ASSERT_NE(outCtrlFlowCache, nullptr);

    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(func, outCtrlFlowCache, MakeConfig()));
}

TEST_F(EmulationLauncherTest, EmulationRunOnce_WithoutInputCtrlCache)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(func, nullptr, MakeConfig()));
}

TEST_F(EmulationLauncherTest, BuildControlFlowCache_WithOutCtrlFlowCache)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    EmulationMemoryUtils memUtils;
    DevControlFlowCache* outCtrlFlowCache = nullptr;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(func, &outCtrlFlowCache, memUtils, MakeConfig()));
    EXPECT_NE(outCtrlFlowCache, nullptr);
}

TEST_F(EmulationLauncherTest, BuildControlFlowCache_WithInputOutputList)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto [inputDeviceDataList, outputDeviceDataList, memUtils] = BuildIO();
    DevControlFlowCache* outCtrlFlowCache = nullptr;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(func, memUtils, inputDeviceDataList, outputDeviceDataList,
                                                          &outCtrlFlowCache, MakeConfig()));
    EXPECT_NE(outCtrlFlowCache, nullptr);
}

TEST_F(EmulationLauncherTest, BuildControlFlowCache_WithEmptyInputOutputList)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    EmulationMemoryUtils memUtils;
    std::vector<DeviceTensorData> emptyInputList;
    std::vector<DeviceTensorData> emptyOutputList;
    DevControlFlowCache* outCtrlFlowCache = nullptr;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(func, memUtils, emptyInputList, emptyOutputList,
                                                          &outCtrlFlowCache, MakeConfig()));
    EXPECT_NE(outCtrlFlowCache, nullptr);
}

TEST_F(EmulationLauncherTest, BuildControlFlowCacheWithEmulationTensorData_WithDisableCtrlFlowCache)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto dynAttr = func->GetDyndevAttribute();
    auto* devProg = DeviceLauncher::GetDevProg(func);
    ASSERT_NE(devProg, nullptr);

    devProg->disableCtrlFlowCache = 1;

    auto [inputDeviceDataList, outputDeviceDataList, memUtils] = BuildIO();
    DevControlFlowCache* outCtrlFlowCache = nullptr;
    EXPECT_EQ(0,
              EmulationLauncher::BuildControlFlowCacheWithEmulationTensorData(
                  func, inputDeviceDataList, outputDeviceDataList, nullptr, &outCtrlFlowCache, memUtils, MakeConfig()));
    EXPECT_EQ(outCtrlFlowCache, nullptr);

    devProg->disableCtrlFlowCache = 0;
}

TEST_F(EmulationLauncherTest, BuildControlFlowCache_WithDisableCtrlFlowCache)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto dynAttr = func->GetDyndevAttribute();
    auto* devProg = DeviceLauncher::GetDevProg(func);
    ASSERT_NE(devProg, nullptr);

    devProg->disableCtrlFlowCache = 1;

    EmulationMemoryUtils memUtils;
    DevControlFlowCache* outCtrlFlowCache = nullptr;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(func, &outCtrlFlowCache, memUtils, MakeConfig()));
    EXPECT_EQ(outCtrlFlowCache, nullptr);

    devProg->disableCtrlFlowCache = 0;
}

TEST_F(EmulationLauncherTest, BuildControlFlowCache_WithInputOutputListAndDisableCtrlFlowCache)
{
    auto* func = GetFunc();
    ASSERT_NE(func, nullptr);

    auto dynAttr = func->GetDyndevAttribute();
    auto* devProg = DeviceLauncher::GetDevProg(func);
    ASSERT_NE(devProg, nullptr);

    devProg->disableCtrlFlowCache = 1;

    auto [inputDeviceDataList, outputDeviceDataList, memUtils] = BuildIO();
    DevControlFlowCache* outCtrlFlowCache = nullptr;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(func, memUtils, inputDeviceDataList, outputDeviceDataList,
                                                          &outCtrlFlowCache, MakeConfig()));
    EXPECT_EQ(outCtrlFlowCache, nullptr);

    devProg->disableCtrlFlowCache = 0;
}
