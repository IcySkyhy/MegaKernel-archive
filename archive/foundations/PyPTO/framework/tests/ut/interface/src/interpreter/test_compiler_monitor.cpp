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
 * \file test_compiler_monitor.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_stage_scope.h"
#include "interface/compiler_monitor/monitor_util.h"

namespace npu::tile_fwk {
class CompilerMonitor : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {}

    void TearDown() override
    {
        MonitorManager::Instance().NotifyCompilationFinished();
        MonitorManager::Instance().SetProcessingThresholdSec(60);
    }
};

TEST_F(CompilerMonitor, CompilerMonitorInitial)
{
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetStageTimeoutFlag("Pass");
    MonitorManager::Instance().GetStageTimeoutFlag("Pass");
    MonitorManager::Instance().GetStageStartTime();
    MonitorManager::Instance().GetStageElapsedTotals();
}

TEST_F(CompilerMonitor, FormatUnknownOpCountAsNotAvailable)
{
    EXPECT_EQ(FormatOpCount(-1), "N/A");
    EXPECT_EQ(FormatOpCount(-10), "N/A");
    EXPECT_EQ(FormatOpCount(0), "0");
    EXPECT_EQ(FormatOpCount(587), "587");
}

TEST_F(CompilerMonitor, CompilerMonitorImpl)
{
    MonitorManager::Instance().Initialize(true, 1, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(5);
    MonitorManager::Instance().SetCurrentFunctionIndex(3);
    sleep(2);
}

TEST_F(CompilerMonitor, CompilerMonitorTestPrint)
{
    MonitorManager::Instance().Initialize(true, 1, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(5);
    MonitorManager::Instance().SetCurrentFunctionIndex(3);
    MonitorManager::Instance().StartStage("Pass");
    sleep(1);
    MonitorManager::Instance().EndStage("Pass");
    sleep(1);
}

TEST_F(CompilerMonitor, CompilerMonitorRootFuncBasic)
{
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetRootFuncCount(4);

    EXPECT_EQ(MonitorManager::Instance().GetRootFuncCount(), 4);

    int idx1 = MonitorManager::Instance().PrepareNextRootFunc();
    EXPECT_EQ(idx1, 1);

    int idx2 = MonitorManager::Instance().PrepareNextRootFunc();
    EXPECT_EQ(idx2, 2);

    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, idx1, "func_A");
    auto stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 1u);
    EXPECT_EQ(stages[0].rootFuncIndex, 1);
    EXPECT_EQ(stages[0].rootFuncName, "func_A");
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, idx1, "func_A");

    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, idx2, "func_B");
    stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 1u);
    EXPECT_EQ(stages[0].rootFuncIndex, 2);
    EXPECT_EQ(stages[0].rootFuncName, "func_B");
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, idx2, "func_B");

    stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 0u);
}

TEST_F(CompilerMonitor, CompilerMonitorFuncToBinStage)
{
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetRootFuncCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("leaf_func_1");

    int rootFuncIdx = MonitorManager::Instance().PrepareNextRootFunc();
    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, rootFuncIdx, "root_func_1");
    sleep(1);
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, rootFuncIdx, "root_func_1");

    int rootFuncIdx2 = MonitorManager::Instance().PrepareNextRootFunc();
    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, rootFuncIdx2, "root_func_2");
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, rootFuncIdx2, "root_func_2");
}

TEST_F(CompilerMonitor, CompilerMonitorStageScopeRootFunc)
{
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetRootFuncCount(2);

    {
        MonitorStageScope scope("Pass");
        sleep(1);
    }

    {
        MonitorStageScope scope(STAGE_FUNC_TO_BIN, 1, "root_func_A", 10);
        sleep(1);
    }
}

TEST_F(CompilerMonitor, CompilerMonitorFuncToBinProcessing)
{
    MonitorManager::Instance().SetProcessingThresholdSec(1);
    MonitorManager::Instance().Initialize(true, 1, 0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(4);
    MonitorManager::Instance().SetRootFuncCount(3);

    int idx1 = MonitorManager::Instance().PrepareNextRootFunc();
    int idx2 = MonitorManager::Instance().PrepareNextRootFunc();
    int idx3 = MonitorManager::Instance().PrepareNextRootFunc();

    EXPECT_EQ(idx1, 1);
    EXPECT_EQ(idx2, 2);
    EXPECT_EQ(idx3, 3);

    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, idx1, "func_Alpha");
    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, idx2, "func_Beta");
    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, idx3, "func_Gamma");

    auto stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 3u);
    EXPECT_EQ(stages[0].rootFuncIndex, 1);
    EXPECT_EQ(stages[0].rootFuncName, "func_Alpha");
    EXPECT_EQ(stages[1].rootFuncIndex, 2);
    EXPECT_EQ(stages[1].rootFuncName, "func_Beta");
    EXPECT_EQ(stages[2].rootFuncIndex, 3);
    EXPECT_EQ(stages[2].rootFuncName, "func_Gamma");

    sleep(2);

    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, idx1, "func_Alpha");
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, idx2, "func_Beta");
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, idx3, "func_Gamma");

    stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 0u);

    MonitorManager::Instance().SetCurrentFunctionIndex(4);
}

TEST_F(CompilerMonitor, CompilerMonitorAllStageProcessingPaths)
{
    MonitorManager::Instance().SetProcessingThresholdSec(1);
    MonitorManager::Instance().Initialize(true, 1, 2, 600);

    MonitorManager::Instance().SetTotalFunctionCount(4);
    MonitorManager::Instance().SetCurrentFunctionIndex(2);
    MonitorManager::Instance().SetCurrentFunctionName("leaf_func_2");

    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().StartStage("CodeGen");
    sleep(2);
    MonitorManager::Instance().EndStage("Pass");
    MonitorManager::Instance().EndStage("CodeGen");

    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().StartStage("CodeGen");
    sleep(2);
    MonitorManager::Instance().EndStage("CodeGen");
}

TEST_F(CompilerMonitor, CompilerMonitorTimeoutSecZero)
{
    // compile_timeout_stage == 0 disables stage timeout; stages should complete without timeout.
    MonitorManager::Instance().SetProcessingThresholdSec(1);
    MonitorManager::Instance().Initialize(true, 1, 0, 600);
    EXPECT_EQ(MonitorManager::Instance().GetTimeoutSec(), 0.0);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetRootFuncCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    sleep(1);
    MonitorManager::Instance().EndStage("Pass");

    int idx = MonitorManager::Instance().PrepareNextRootFunc();
    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, idx, "root_func_1");
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, idx, "root_func_1");
}

TEST_F(CompilerMonitor, CompilerMonitorHostMachineStepGroup)
{
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().BeginHostMachineCompileGroup(4);
    EXPECT_EQ(MonitorManager::Instance().GetHostMachineTotalSteps(), 4);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 1);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 2);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 3);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 4);

    MonitorManager::Instance().SetCompilerMonitorOptions(false, 2, 4, 5);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), -1);
}

} // namespace npu::tile_fwk
