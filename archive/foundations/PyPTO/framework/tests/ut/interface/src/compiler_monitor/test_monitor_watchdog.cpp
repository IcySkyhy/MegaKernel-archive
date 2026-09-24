/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_monitor_watchdog.cpp
 * \brief Unit tests for Watchdog mode (compile_monitor_enable=1): banner, summary, output suppression
 */

#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#define private public
#include "interface/compiler_monitor/monitor_manager.h"
#undef private
#include "gtest/gtest.h"
#include "monitor_test_fixture.h"
#include "interface/compiler_monitor/monitor_impl.h"

using namespace npu::tile_fwk;

class TestMonitorWatchdog : public MonitorTestFixtureBase {};

// --- RecordTimedOutStage: safe, idempotent, stores elapsed ---

TEST_F(TestMonitorWatchdog, RecordTimedOutStageSafeAndIdempotent)
{
    MonitorManager::Instance().Initialize(true, 60, 0.0, 600);
    MonitorManager::Instance().RecordTimedOutStage("Pass", 100.0);
    MonitorManager::Instance().RecordTimedOutStage("CodeGen", 15.0);
    MonitorManager::Instance().RecordTimedOutStage("Pass", 200.0);

    const auto& stages = MonitorManager::Instance().summaryTimedOutStages_;
    EXPECT_EQ(stages.size(), 2u);
    EXPECT_EQ(stages.count("Pass"), 1u);
    EXPECT_EQ(stages.count("CodeGen"), 1u);
    EXPECT_NEAR(stages.at("Pass"), 100.0, 0.01);
}

// --- Watchdog mode: passDetailEnable_ is false when mode=1, true when mode=2 ---

TEST_F(TestMonitorWatchdog, PassDetailEnableFlag)
{
    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, false);
    EXPECT_FALSE(MonitorManager::Instance().IsPassDetailEnabled());
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, true);
    EXPECT_TRUE(MonitorManager::Instance().IsPassDetailEnabled());
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(false, 60, 0.0, 600, false);
    EXPECT_FALSE(MonitorManager::Instance().IsEnabled());
}

// --- summaryTimedOutStages_ cleared on re-init ---

TEST_F(TestMonitorWatchdog, SummaryClearedOnReinit)
{
    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, false);
    MonitorManager::Instance().RecordTimedOutStage("Pass", 100.0);
    EXPECT_EQ(MonitorManager::Instance().summaryTimedOutStages_.size(), 1u);
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, false);
    EXPECT_EQ(MonitorManager::Instance().summaryTimedOutStages_.size(), 0u);
}

// --- PrintCurrentTotalElapsed: no crash in both modes ---

TEST_F(TestMonitorWatchdog, PrintCurrentTotalElapsedNoCrash)
{
    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, false);
    MonitorManager::Instance().PrintCurrentTotalElapsed("test");
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, true);
    MonitorManager::Instance().PrintCurrentTotalElapsed("test");
}

// --- RecordPassCompileTime in non-detail: records timing, detects timeout ---

TEST_F(TestMonitorWatchdog, RecordPassCompileTimeNonDetailMode)
{
    // timeoutSec_ > 0 enables stage timeout; Pass uses ops-scaled threshold (1000 ops ≈ 0.1s).
    MonitorManager::Instance().Initialize(true, 60, 1.0, 600, false);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().StartPassCompile("StratA", "Pass1", 0, "func_1", 1, 1000);
    MonitorManager::Instance().RecordPassCompileTime("StratA", "Pass1", 0, "func_1", 1, 1000, 0.5, true);
    MonitorManager::Instance().EndPassCompile("StratA", "Pass1", 0, "func_1", 1);
    MonitorManager::Instance().EndStage("Pass");

    auto timings = MonitorManager::Instance().GetPassCompileTimings();
    EXPECT_EQ(timings.size(), 1u);
    EXPECT_EQ(timings[0].strategy, "StratA");
    EXPECT_NEAR(timings[0].elapsedSec, 0.5, 0.01);

    EXPECT_EQ(MonitorManager::Instance().summaryTimedOutStages_.count("Pass"), 1u);
}

// --- RecordPassCompileTime in non-detail: no timeout when timeoutSec_ disabled ---

TEST_F(TestMonitorWatchdog, RecordPassCompileTimeNoTimeoutWhenUnderThreshold)
{
    // compile_timeout_stage <= 0 disables stage timeout detection.
    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, false);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().StartPassCompile("StratA", "Pass1", 0, "func_1", 1, 1000);
    MonitorManager::Instance().RecordPassCompileTime("StratA", "Pass1", 0, "func_1", 1, 1000, 0.5, true);
    MonitorManager::Instance().EndPassCompile("StratA", "Pass1", 0, "func_1", 1);
    MonitorManager::Instance().EndStage("Pass");

    EXPECT_EQ(MonitorManager::Instance().summaryTimedOutStages_.size(), 0u);
}

// --- TryEndPrepareStage: no crash in both modes ---

TEST_F(TestMonitorWatchdog, TryEndPrepareStageNoCrash)
{
    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, false);
    MonitorManager::Instance().TryEndPrepareStage();
    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("Prepare"), totals.end());
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, true);
    MonitorManager::Instance().TryEndPrepareStage();
}

// --- EndStage: no crash, records elapsed in both modes ---

TEST_F(TestMonitorWatchdog, EndStageRecordsElapsed)
{
    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, false);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");
    MonitorManager::Instance().StartStage("CodeGen");
    MonitorManager::Instance().EndStage("CodeGen");
    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("CodeGen"), totals.end());
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, true);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");
    MonitorManager::Instance().StartStage("CodeGen");
    MonitorManager::Instance().EndStage("CodeGen");
}

// --- Stage timeout: CodeGen timeout, no false positive, disabled when <= 0 ---

TEST_F(TestMonitorWatchdog, StageTimeoutFallback)
{
    MonitorManager::Instance().Initialize(true, 60, 0.001, 600, false);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");
    MonitorManager::Instance().StartStage("CodeGen");
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    MonitorManager::Instance().EndStage("CodeGen");
    EXPECT_EQ(MonitorManager::Instance().summaryTimedOutStages_.count("CodeGen"), 1u);
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(true, 60, 10.0, 600, false);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");
    MonitorManager::Instance().StartStage("CodeGen");
    MonitorManager::Instance().EndStage("CodeGen");
    EXPECT_EQ(MonitorManager::Instance().summaryTimedOutStages_.size(), 0u);
    MonitorManager::Instance().NotifyCompilationFinished();

    // compile_timeout_stage <= 0 disables stage timeout.
    MonitorManager::Instance().Initialize(true, 60, 0.0, 600, false);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");
    MonitorManager::Instance().StartStage("CodeGen");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    MonitorManager::Instance().EndStage("CodeGen");
    EXPECT_EQ(MonitorManager::Instance().summaryTimedOutStages_.size(), 0u);
}

// --- Prepare stage timeout ---

TEST_F(TestMonitorWatchdog, PrepareStageTimeout)
{
    MonitorManager::Instance().Initialize(true, 60, 0.001, 600, false);
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    MonitorManager::Instance().TryEndPrepareStage();
    EXPECT_EQ(MonitorManager::Instance().summaryTimedOutStages_.count("Prepare"), 1u);
}

// --- Detail mode (mode=2): Pass stage timeout triggers WARNING print via CheckStageTimeoutOnEnd / tick ---

TEST_F(TestMonitorWatchdog, DetailModePassStageTimeoutPrintsWarning)
{
    // timeoutSec_ > 0 enables Pass stage timeout (ops-scaled; 1 op ≈ 0.45ms with base 90s).
    // Detail mode prints WARNING but does not record into summaryTimedOutStages_ (Watchdog-only map).
    MonitorManager::Instance().Initialize(true, 60, 1.0, 600, true);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");
    MonitorManager::Instance().SetCurrentFuncOpSize(1);

    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().StartPassCompile("StratA", "Pass1", 0, "func_1", 1, 1);
    MonitorManager::Instance().RecordPassCompileTime("StratA", "Pass1", 0, "func_1", 1, 1, 0.5, true);
    MonitorManager::Instance().EndPassCompile("StratA", "Pass1", 0, "func_1", 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    MonitorManager::Instance().EndStage("Pass");
    // Smoke: path completes; WARNING is emitted on stdout (see CI log), not via summaryTimedOutStages_.
}

// --- Detail mode: background thread tick triggers WARNING for generic stage (HandleGenericStage) ---

TEST_F(TestMonitorWatchdog, DetailModeGenericStageTimeoutViaTick)
{
    MonitorManager::Instance().Initialize(true, 60, 0.001, 600, true);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("CodeGen");
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    MonitorManager::Instance().EndStage("CodeGen");
}

// --- Detail mode: background thread tick triggers WARNING for Pass stage (EmitPassStageTimeoutWarningIfNeeded) ---

TEST_F(TestMonitorWatchdog, DetailModePassTimeoutViaTick)
{
    MonitorManager::Instance().Initialize(true, 60, 0.001, 600, true);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().StartPassCompile("StratA", "Pass1", 0, "func_1", 1, 1000);
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    MonitorManager::Instance().EndPassCompile("StratA", "Pass1", 0, "func_1", 1);
    MonitorManager::Instance().EndStage("Pass");
}

// --- Detail mode: background thread tick triggers total timeout (PrintTotalTimeOut) ---

TEST_F(TestMonitorWatchdog, DetailModeTotalTimeoutViaTick)
{
    MonitorManager::Instance().Initialize(true, 60, 0.0, 1, true);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");

    MonitorManager::Instance().StartStage("CodeGen");
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    MonitorManager::Instance().EndStage("CodeGen");
    EXPECT_EQ(MonitorManager::Instance().summaryTimedOutStages_.count("Total"), 1u);
}

// --- Detail mode: background thread tick triggers WARNING for FuncToBin stage (HandleFuncToBin) ---

TEST_F(TestMonitorWatchdog, DetailModeFuncToBinTimeoutViaTick)
{
    MonitorManager::Instance().Initialize(true, 60, 0.001, 600, true);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetRootFuncCount(1);

    int idx = MonitorManager::Instance().PrepareNextRootFunc();
    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, idx, "root_func_1", 100);
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, idx, "root_func_1", 100);
}

// --- Detail mode: background thread tick triggers WARNING for HostMachine stage (HandleHostMachine) ---

TEST_F(TestMonitorWatchdog, DetailModeHostMachineTimeoutViaTick)
{
    MonitorManager::Instance().Initialize(true, 60, 0.001, 600, true);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().BeginHostMachineCompileGroup(1);

    int step = MonitorManager::Instance().AllocHostMachineStepIndex();
    MonitorManager::Instance().StartStage(STAGE_HOST_MACHINE, step, "hm_step_1", 100);
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    MonitorManager::Instance().EndStage(STAGE_HOST_MACHINE, step, "hm_step_1", 100);
}

// --- Detail mode: background thread tick triggers processing heartbeat for generic stage ---

TEST_F(TestMonitorWatchdog, DetailModeGenericStageProcessingHeartbeat)
{
    MonitorManager::Instance().SetProcessingThresholdSec(0);
    MonitorManager::Instance().Initialize(true, 1, 0.001, 600, true);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");

    MonitorManager::Instance().StartStage("CodeGen");
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    MonitorManager::Instance().EndStage("CodeGen");
    MonitorManager::Instance().SetProcessingThresholdSec(60);
}

// --- Watchdog summary: prints DetectedStage line when total timeout ---

TEST_F(TestMonitorWatchdog, WatchdogSummaryPrintsDetectedStageOnTotalTimeout)
{
    MonitorManager::Instance().Initialize(true, 60, 0.0, 10, false);
    MonitorManager::Instance().SetTotalFunctionCount(2);

    // Total overtime but no single stage exceeds avg threshold (10/4=2.5s).
    MonitorManager::Instance().stageElapsedTotals_["Prepare"] = 1.0;
    MonitorManager::Instance().stageElapsedTotals_["Pass"] = 1.0;
    MonitorManager::Instance().stageElapsedTotals_["HostMachine"] = 1.0;
    MonitorManager::Instance().stageElapsedTotals_["CodeGen"] = 1.0;
    MonitorManager::Instance().totalStart_ = std::chrono::steady_clock::now() - std::chrono::seconds(12);
    MonitorManager::Instance().PrintWatchdogSummary(12.0);
    EXPECT_NEAR(MonitorManager::Instance().GetTotalElapsed(), 12.0, 0.01);

    // CodeGen exceeds avg threshold: summary still prints DetectedStage only (no function list).
    MonitorManager::Instance().stageElapsedTotals_["CodeGen"] = 8.0;
    MonitorManager::Instance().PrintWatchdogSummary(12.0);
    EXPECT_NEAR(MonitorManager::Instance().GetTotalElapsed(), 12.0, 0.01);
}
