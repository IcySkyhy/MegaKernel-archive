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
 * \file test_monitor_manager.cpp
 * \brief Unit tests for MonitorManager state management, static functions, getters/setters (no sleep)
 */

#include <cstdlib>
#include <string>
#include <map>
#include <thread>
#include <chrono>
#include "gtest/gtest.h"
#define private public
#include "interface/compiler_monitor/monitor_manager.h"
#undef private
#include "monitor_test_fixture.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_util.h"

using namespace npu::tile_fwk;

class TestMonitorManagerState : public MonitorTestFixtureBase {};

TEST_F(TestMonitorManagerState, CalcPassStageTimeoutSecZeroOpSize)
{
    double result = MonitorManager::CalcPassStageTimeoutSec(0);
    double expected = 90.0 * 1.0 / 200000.0;
    EXPECT_NEAR(result, expected, 0.001);
}

TEST_F(TestMonitorManagerState, CalcPassStageTimeoutSecSmallOpSize)
{
    double result = MonitorManager::CalcPassStageTimeoutSec(100);
    double expected = 90.0 * 100.0 / 200000.0;
    EXPECT_NEAR(result, expected, 0.001);
}

TEST_F(TestMonitorManagerState, CalcPassStageTimeoutSecExactlyBaseOpSize)
{
    double result = MonitorManager::CalcPassStageTimeoutSec(200000);
    EXPECT_NEAR(result, 90.0, 0.001);
}

TEST_F(TestMonitorManagerState, CalcPassStageTimeoutSecLargeOpSize)
{
    double result = MonitorManager::CalcPassStageTimeoutSec(1000000);
    double expected = 90.0 * 1000000.0 / 200000.0;
    EXPECT_NEAR(result, expected, 0.01);
}

TEST_F(TestMonitorManagerState, FormatPassDurationForLogNormal)
{
    std::string result = MonitorManager::FormatPassDurationForLog(1.5);
    EXPECT_NE(result.find("1500.000ms"), std::string::npos);
}

TEST_F(TestMonitorManagerState, FormatPassDurationForLogZero)
{
    std::string result = MonitorManager::FormatPassDurationForLog(0.0);
    EXPECT_NE(result.find("0.000ms"), std::string::npos);
}

TEST_F(TestMonitorManagerState, FormatPassDurationForLogNegative)
{
    std::string result = MonitorManager::FormatPassDurationForLog(-1.0);
    EXPECT_EQ(result, "-1");
}

TEST_F(TestMonitorManagerState, FormatPassDurationForLogSmallValue)
{
    std::string result = MonitorManager::FormatPassDurationForLog(0.001);
    EXPECT_NE(result.find("1.000ms"), std::string::npos);
}

TEST_F(TestMonitorManagerState, SetCompilerMonitorOptionsEnvVars)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(true, 5, 10.0, 120, false);
    EXPECT_EQ(MonitorManager::Instance().IsEnabled(), true);
    EXPECT_EQ(MonitorManager::Instance().GetIntervalSec(), 5);
    EXPECT_EQ(MonitorManager::Instance().GetTimeoutSec(), 10.0);
    EXPECT_EQ(MonitorManager::Instance().GetTotalTimeoutSec(), 120);
    EXPECT_EQ(MonitorManager::Instance().IsPassDetailEnabled(), false);

    const char* intervalEnv = std::getenv("PYPTO_COMPILER_MONITOR_INTERVAL_SEC");
    EXPECT_NE(intervalEnv, nullptr);
    const char* timeoutEnv = std::getenv("PYPTO_COMPILER_MONITOR_TIMEOUT_SEC");
    EXPECT_NE(timeoutEnv, nullptr);
    const char* totalEnv = std::getenv("PYPTO_COMPILER_MONITOR_TOTAL_TIMEOUT_SEC");
    EXPECT_NE(totalEnv, nullptr);
}

TEST_F(TestMonitorManagerState, SetCompilerMonitorOptionsZeroInterval)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(true, 0, -1.0, -1, false);
    EXPECT_EQ(MonitorManager::Instance().GetIntervalSec(), 60);
}

TEST_F(TestMonitorManagerState, SetCompilerMonitorOptionsNegativeTimeout)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(true, 60, -2.0, 600, false);
    EXPECT_EQ(MonitorManager::Instance().GetTimeoutSec(), 0.0);
    MonitorManager::Instance().SetCompilerMonitorOptions(true, 60, -1.0, 600, false);
    EXPECT_EQ(MonitorManager::Instance().GetTimeoutSec(), 0.0);
    MonitorManager::Instance().SetCompilerMonitorOptions(true, 60, 0.0, 600, false);
    EXPECT_EQ(MonitorManager::Instance().GetTimeoutSec(), 0.0);
}

TEST_F(TestMonitorManagerState, SetCompilerMonitorOptionsNegativeTotalTimeout)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(true, 60, -1.0, -1, false);
    EXPECT_EQ(MonitorManager::Instance().GetTotalTimeoutSec(), 600);
}

TEST_F(TestMonitorManagerState, SetCompilerMonitorOptionsPassDetailEnable)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(true, 60, -1.0, 600, true);
    EXPECT_EQ(MonitorManager::Instance().IsPassDetailEnabled(), true);
}

TEST_F(TestMonitorManagerState, SetCompilerMonitorOptionsDisable)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(true, 60, -1.0, 600, false);
    EXPECT_EQ(MonitorManager::Instance().IsEnabled(), true);
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 60, -1.0, 600, false);
    EXPECT_EQ(MonitorManager::Instance().IsEnabled(), false);
}

TEST_F(TestMonitorManagerState, InitializeAndShutdownLifecycle)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentStageName(), "Prepare");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, InitializeNotEnabledNoThread)
{
    MonitorManager::Instance().Initialize(false, 60, -1.0, 600);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, SetTotalFunctionCountAndIndex)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(10);
    EXPECT_EQ(MonitorManager::Instance().GetTotalFunctionCount(), 10);

    int idx1 = MonitorManager::Instance().GetAndIncrementNextFunctionIndex();
    EXPECT_EQ(idx1, 1);
    int idx2 = MonitorManager::Instance().GetAndIncrementNextFunctionIndex();
    EXPECT_EQ(idx2, 2);

    MonitorManager::Instance().SetCurrentFunctionIndex(5);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentFunctionIndex(), 5);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetAndIncrementNextFunctionIndexDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 60, -1.0, 600);
    int idx = MonitorManager::Instance().GetAndIncrementNextFunctionIndex();
    EXPECT_EQ(idx, 0);
}

TEST_F(TestMonitorManagerState, SetCurrentFunctionIndexDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 60, -1.0, 600);
    MonitorManager::Instance().SetCurrentFunctionIndex(5);
}

TEST_F(TestMonitorManagerState, SetCurrentFunctionNameAndOpSize)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetCurrentFunctionName("test_func_1");
    EXPECT_EQ(MonitorManager::Instance().GetCurrentFunctionName(), "test_func_1");

    MonitorManager::Instance().SetCurrentFuncOpSize(5000);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentFuncOpSize(), 5000);

    MonitorManager::Instance().SetFuncSumOpSize(1000, false);
    EXPECT_EQ(MonitorManager::Instance().GetFuncSumOpSize(), 1000);
    MonitorManager::Instance().SetFuncSumOpSize(2000, false);
    EXPECT_EQ(MonitorManager::Instance().GetFuncSumOpSize(), 3000);
    MonitorManager::Instance().SetFuncSumOpSize(0, true);
    EXPECT_EQ(MonitorManager::Instance().GetFuncSumOpSize(), 0);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, SetCurrentFuncOpSizeDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 60, -1.0, 600);
    MonitorManager::Instance().SetCurrentFuncOpSize(100);
}

TEST_F(TestMonitorManagerState, SetFuncSumOpSizeDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 60, -1.0, 600);
    MonitorManager::Instance().SetFuncSumOpSize(100, false);
}

TEST_F(TestMonitorManagerState, SetCurrentFunctionNameDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 60, -1.0, 600);
    MonitorManager::Instance().SetCurrentFunctionName("name");
}

TEST_F(TestMonitorManagerState, SwitchStageReset)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetCurrentFunctionName("func_before_reset");
    MonitorManager::Instance().SetCurrentFuncOpSize(9999);
    MonitorManager::Instance().SetFuncSumOpSize(5000, false);

    MonitorManager::Instance().SwitchStageReset();

    EXPECT_EQ(MonitorManager::Instance().GetCurrentFuncOpSize(), 0);
    EXPECT_EQ(MonitorManager::Instance().GetFuncSumOpSize(), 0);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, SetCurrentFuncOpSizeUpdateActiveStage)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetCurrentFunctionIndex(2);
    MonitorManager::Instance().SetCurrentFunctionName("func_2");

    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().SetCurrentFuncOpSize(12345, true);

    auto stages = MonitorManager::Instance().GetActiveStages();
    bool found = false;
    for (const auto& s : stages) {
        if (s.stageName == "Pass" && s.functionIndex == 2 && s.functionOpSize == 12345) {
            found = true;
        }
    }
    EXPECT_TRUE(found);

    MonitorManager::Instance().EndStage("Pass");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetProgressWidthSingleFunction)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    int pw = MonitorManager::Instance().GetProgressWidth();
    EXPECT_EQ(pw, 3);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetProgressWidthMultipleFunctions)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(100);
    int pw = MonitorManager::Instance().GetProgressWidth();
    EXPECT_GE(pw, 5);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetProgressWidthWithRootFuncs)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(10);
    MonitorManager::Instance().SetRootFuncCount(50);
    int pw = MonitorManager::Instance().GetProgressWidth();
    EXPECT_GE(pw, 5);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, StageTimeoutFlagSetAndGet)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    EXPECT_EQ(MonitorManager::Instance().GetStageTimeoutFlag("Pass"), false);
    MonitorManager::Instance().SetStageTimeoutFlag("Pass");
    EXPECT_EQ(MonitorManager::Instance().GetStageTimeoutFlag("Pass"), true);
    EXPECT_EQ(MonitorManager::Instance().GetStageTimeoutFlag("Unknown"), false);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, SetActiveStageWarningPrinted)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetRootFuncCount(2);
    int idx = MonitorManager::Instance().PrepareNextRootFunc();
    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, idx, "root_func_1");

    auto stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages[0].warningPrinted, false);

    MonitorManager::Instance().SetActiveStageWarningPrinted(STAGE_FUNC_TO_BIN, idx);
    stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages[0].warningPrinted, true);

    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, idx, "root_func_1");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetCurrentStageElapsedNotInitialized)
{
    double elapsed = MonitorManager::Instance().GetCurrentStageElapsed("Pass");
    EXPECT_EQ(elapsed, 0.0);
}

TEST_F(TestMonitorManagerState, StartStageAndEndStagePass)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    EXPECT_EQ(MonitorManager::Instance().GetCurrentStageName(), "Pass");
    auto stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 1u);
    EXPECT_EQ(stages[0].stageName, "Pass");

    MonitorManager::Instance().EndStage("Pass");
    stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 0u);

    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("Pass"), totals.end());
    EXPECT_GT(totals["Pass"], 0.0);

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, StartStageAndEndStageCodeGen)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);

    MonitorManager::Instance().StartStage("CodeGen");
    EXPECT_EQ(MonitorManager::Instance().GetCurrentStageName(), "CodeGen");
    MonitorManager::Instance().EndStage("CodeGen");

    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("CodeGen"), totals.end());
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, StartStageFuncToBinRemovesCodeGen)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetRootFuncCount(1);

    MonitorManager::Instance().StartStage("CodeGen");
    auto stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 1u);
    EXPECT_EQ(stages[0].stageName, "CodeGen");

    int idx = MonitorManager::Instance().PrepareNextRootFunc();
    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, idx, "root_func_1");
    stages = MonitorManager::Instance().GetActiveStages();
    bool hasCodeGen = false;
    bool hasFuncToBin = false;
    for (const auto& s : stages) {
        if (s.stageName == "CodeGen")
            hasCodeGen = true;
        if (s.stageName == STAGE_FUNC_TO_BIN)
            hasFuncToBin = true;
    }
    EXPECT_FALSE(hasCodeGen);
    EXPECT_TRUE(hasFuncToBin);
    EXPECT_TRUE(MonitorManager::Instance().hasSuspendedCodeGen_);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, idx, "root_func_1");
    MonitorManager::Instance().EndStage("CodeGen");

    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    ASSERT_NE(totals.find("CodeGen"), totals.end());
    // FuncToBin is folded into CodeGen wall-clock and not listed separately.
    EXPECT_TRUE(totals.find(STAGE_FUNC_TO_BIN) == totals.end());
    EXPECT_GE(totals.at("CodeGen"), 0.15);
    EXPECT_FALSE(MonitorManager::Instance().hasSuspendedCodeGen_);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, StartStageHostMachineRemovesCodeGen)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().BeginHostMachineCompileGroup(3);

    MonitorManager::Instance().StartStage("CodeGen");
    int stepIdx = MonitorManager::Instance().AllocHostMachineStepIndex();
    MonitorManager::Instance().StartStage(STAGE_HOST_MACHINE, stepIdx, "host_step_1");

    auto stages = MonitorManager::Instance().GetActiveStages();
    bool hasCodeGen = false;
    bool hasHostMachine = false;
    for (const auto& s : stages) {
        if (s.stageName == "CodeGen")
            hasCodeGen = true;
        if (s.stageName == STAGE_HOST_MACHINE)
            hasHostMachine = true;
    }
    EXPECT_FALSE(hasCodeGen);
    EXPECT_TRUE(hasHostMachine);

    MonitorManager::Instance().EndStage(STAGE_HOST_MACHINE, stepIdx, "host_step_1");
    MonitorManager::Instance().EndStage("CodeGen");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, TryEndPrepareStage)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentStageName(), "Prepare");

    MonitorManager::Instance().TryEndPrepareStage();

    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("Prepare"), totals.end());
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, PrintCurrentTotalElapsed)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().PrintCurrentTotalElapsed("Test checkpoint");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, PrintCurrentTotalElapsedDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 60, -1.0, 600);
    MonitorManager::Instance().PrintCurrentTotalElapsed("Should not print");
}

TEST_F(TestMonitorManagerState, ProcessingThresholdSec)
{
    EXPECT_EQ(MonitorManager::Instance().GetProcessingThresholdSec(), 60);
    MonitorManager::Instance().SetProcessingThresholdSec(30);
    EXPECT_EQ(MonitorManager::Instance().GetProcessingThresholdSec(), 30);
    MonitorManager::Instance().SetProcessingThresholdSec(60);
}

TEST_F(TestMonitorManagerState, HostMachineStepGroup)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().BeginHostMachineCompileGroup(4);
    EXPECT_EQ(MonitorManager::Instance().GetHostMachineTotalSteps(), 4);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 1);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 2);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 3);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 4);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, HostMachineStepGroupDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 60, -1.0, 600);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), -1);
}

TEST_F(TestMonitorManagerState, BeginHostMachineCompileGroupDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 60, -1.0, 600);
    MonitorManager::Instance().BeginHostMachineCompileGroup(3);
}

TEST_F(TestMonitorManagerState, GetHostMachineTotalStepsDefault)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().BeginHostMachineCompileGroup(0);
    EXPECT_EQ(MonitorManager::Instance().GetHostMachineTotalSteps(), 1);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, RootFuncTracking)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetRootFuncCount(3);
    EXPECT_EQ(MonitorManager::Instance().GetRootFuncCount(), 3);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentRootFuncIndex(), 0);

    int idx1 = MonitorManager::Instance().PrepareNextRootFunc();
    EXPECT_EQ(idx1, 1);

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, RootFuncTrackingDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 60, -1.0, 600);
    EXPECT_EQ(MonitorManager::Instance().PrepareNextRootFunc(), 0);
}

TEST_F(TestMonitorManagerState, SetRootFuncCountDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 60, -1.0, 600);
    MonitorManager::Instance().SetRootFuncCount(5);
}

TEST_F(TestMonitorManagerState, StartPassCompileAndEnd)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartPassCompile("StrategyA", "Pass1", 0, "func_1", 1, 1000);
    std::string desc = MonitorManager::Instance().GetCurrentPassDescription();
    EXPECT_NE(desc.find("Pass1"), std::string::npos);
    EXPECT_NE(desc.find("StrategyA"), std::string::npos);

    MonitorManager::Instance().EndPassCompile("StrategyA", "Pass1", 0, "func_1", 1);
    desc = MonitorManager::Instance().GetCurrentPassDescription();
    EXPECT_EQ(desc, "");

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetCurrentPassDescriptionInactive)
{
    std::string desc = MonitorManager::Instance().GetCurrentPassDescription();
    EXPECT_EQ(desc, "");
}

TEST_F(TestMonitorManagerState, RecordPassCompileTimeAndRetrieve)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartPassCompile("StrategyA", "Pass1", 0, "func_1", 1, 5000);
    MonitorManager::Instance().RecordPassCompileTime("StrategyA", "Pass1", 0, "func_1", 1, 5000, 0.5, true);
    MonitorManager::Instance().EndPassCompile("StrategyA", "Pass1", 0, "func_1", 1);

    auto timings = MonitorManager::Instance().GetPassCompileTimings();
    EXPECT_EQ(timings.size(), 1u);
    EXPECT_EQ(timings[0].strategy, "StrategyA");
    EXPECT_EQ(timings[0].passIdentifier, "Pass1");
    EXPECT_EQ(timings[0].functionIndex, 1);
    EXPECT_EQ(timings[0].functionOpSize, 5000);
    EXPECT_NEAR(timings[0].elapsedSec, 0.5, 0.01);
    EXPECT_EQ(timings[0].success, true);

    auto elapsedTotals = MonitorManager::Instance().GetPassElapsedTotals();
    EXPECT_NE(elapsedTotals.find("StrategyA::Pass1"), elapsedTotals.end());

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, RecordPassCompileTimeNotInitialized)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().NotifyCompilationFinished();
    size_t sizeBefore = MonitorManager::Instance().GetPassCompileTimings().size();
    MonitorManager::Instance().RecordPassCompileTime("S", "P", 0, "f", 1, 100, 1.0, true);
    size_t sizeAfter = MonitorManager::Instance().GetPassCompileTimings().size();
    EXPECT_EQ(sizeAfter, sizeBefore);
}

TEST_F(TestMonitorManagerState, StartPassCompileNotInitialized)
{
    MonitorManager::Instance().StartPassCompile("S", "P", 0, "f", 1, 100);
}

TEST_F(TestMonitorManagerState, MultiplePassCompileRecords)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);

    MonitorManager::Instance().StartPassCompile("StratA", "Pass1", 0, "f1", 1, 1000);
    MonitorManager::Instance().RecordPassCompileTime("StratA", "Pass1", 0, "f1", 1, 1000, 0.3, true);
    MonitorManager::Instance().EndPassCompile("StratA", "Pass1", 0, "f1", 1);

    MonitorManager::Instance().StartPassCompile("StratA", "Pass2", 1, "f1", 1, 1000);
    MonitorManager::Instance().RecordPassCompileTime("StratA", "Pass2", 1, "f1", 1, 1000, 0.7, true);
    MonitorManager::Instance().EndPassCompile("StratA", "Pass2", 1, "f1", 1);

    MonitorManager::Instance().StartPassCompile("StratB", "Pass1", 0, "f2", 2, 2000);
    MonitorManager::Instance().RecordPassCompileTime("StratB", "Pass1", 0, "f2", 2, 2000, 0.4, false);
    MonitorManager::Instance().EndPassCompile("StratB", "Pass1", 0, "f2", 2);

    auto timings = MonitorManager::Instance().GetPassCompileTimings();
    EXPECT_EQ(timings.size(), 3u);

    auto totals = MonitorManager::Instance().GetPassElapsedTotals();
    EXPECT_NE(totals.find("StratA::Pass1"), totals.end());
    EXPECT_NE(totals.find("StratA::Pass2"), totals.end());
    EXPECT_NE(totals.find("StratB::Pass1"), totals.end());

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, PassDetailEnableMode)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600, true);
    EXPECT_EQ(MonitorManager::Instance().IsPassDetailEnabled(), true);

    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().StartPassCompile("StratA", "Pass1", 0, "func_1", 1, 1000);
    MonitorManager::Instance().RecordPassCompileTime("StratA", "Pass1", 0, "func_1", 1, 1000, 0.1, true);
    MonitorManager::Instance().EndPassCompile("StratA", "Pass1", 0, "func_1", 1);
    MonitorManager::Instance().EndStage("Pass");

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetTotalElapsedBeforeFinish)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    double elapsed = MonitorManager::Instance().GetTotalElapsed();
    EXPECT_GE(elapsed, 0.0);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, NotifyCompilationFinishedPrintsSummary)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetCurrentFunctionIndex(3);

    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().EndStage("Pass");

    MonitorManager::Instance().NotifyCompilationFinished();
    double totalElapsed = MonitorManager::Instance().GetTotalElapsed();
    EXPECT_GT(totalElapsed, 0.0);
}

TEST_F(TestMonitorManagerState, NotifyCompilationFinishedNotInitialized)
{
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, ShutdownNotInitialized) { MonitorManager::Instance().Shutdown(); }

TEST_F(TestMonitorManagerState, EndStageNoMatchStillCompletes)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);

    MonitorManager::Instance().EndStage("NonExistentStage");
    MonitorManager::Instance().NotifyCompilationFinished();
}
