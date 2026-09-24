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
 * \file test_monitor_pass_scope.cpp
 * \brief Unit tests for MonitorPassCompileScope RAII guard (no sleep)
 */

#include <chrono>
#include <string>
#include "gtest/gtest.h"
#include "monitor_test_fixture.h"
#include "interface/compiler_monitor/monitor_pass_scope.h"

using namespace npu::tile_fwk;

class TestMonitorPassScope : public MonitorTestFixtureBase {};

TEST_F(TestMonitorPassScope, BasicScopeFinishSuccess)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    {
        MonitorPassCompileScope scope("StratA", "Pass1", 0, "func_1", 1, 1000);
        scope.Finish(true);
    }

    auto timings = MonitorManager::Instance().GetPassCompileTimings();
    EXPECT_EQ(timings.size(), 1u);
    EXPECT_EQ(timings[0].success, true);
    EXPECT_EQ(timings[0].strategy, "StratA");

    MonitorManager::Instance().EndStage("Pass");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorPassScope, ScopeFinishFailure)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    {
        MonitorPassCompileScope scope("StratB", "Pass2", 1, "func_1", 1, 500);
        scope.Finish(false);
    }

    auto timings = MonitorManager::Instance().GetPassCompileTimings();
    EXPECT_EQ(timings.size(), 1u);
    EXPECT_EQ(timings[0].success, false);
    EXPECT_EQ(timings[0].strategy, "StratB");

    MonitorManager::Instance().EndStage("Pass");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorPassScope, DestructorCallsFinishFalseWhenNotFinished)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    {
        MonitorPassCompileScope scope("StratC", "Pass3", 2, "func_1", 1, 2000);
    }

    auto timings = MonitorManager::Instance().GetPassCompileTimings();
    EXPECT_EQ(timings.size(), 1u);
    EXPECT_EQ(timings[0].success, false);

    MonitorManager::Instance().EndStage("Pass");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorPassScope, FinishAtWithExplicitEndTime)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    {
        MonitorPassCompileScope scope("StratD", "Pass4", 0, "func_1", 1, 3000);
        auto endTime = scope.GetStartTime() + std::chrono::milliseconds(100);
        scope.FinishAt(true, endTime);
    }

    auto timings = MonitorManager::Instance().GetPassCompileTimings();
    EXPECT_EQ(timings.size(), 1u);
    EXPECT_EQ(timings[0].success, true);
    EXPECT_NEAR(timings[0].elapsedSec, 0.1, 0.05);

    MonitorManager::Instance().EndStage("Pass");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorPassScope, DoubleFinishIgnored)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    {
        MonitorPassCompileScope scope("StratE", "Pass5", 0, "func_1", 1, 1000);
        scope.Finish(true);
        scope.Finish(false);
    }

    auto timings = MonitorManager::Instance().GetPassCompileTimings();
    EXPECT_EQ(timings.size(), 1u);
    EXPECT_EQ(timings[0].success, true);

    MonitorManager::Instance().EndStage("Pass");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorPassScope, MultipleScopesSequential)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    {
        MonitorPassCompileScope scope1("Strat1", "Pass1", 0, "func_1", 1, 1000);
        scope1.Finish(true);
    }
    {
        MonitorPassCompileScope scope2("Strat1", "Pass2", 1, "func_1", 1, 1000);
        scope2.Finish(true);
    }
    {
        MonitorPassCompileScope scope3("Strat2", "Pass1", 0, "func_1", 1, 2000);
        scope3.Finish(false);
    }

    auto timings = MonitorManager::Instance().GetPassCompileTimings();
    EXPECT_EQ(timings.size(), 3u);

    auto totals = MonitorManager::Instance().GetPassElapsedTotals();
    EXPECT_NE(totals.find("Strat1::Pass1"), totals.end());
    EXPECT_NE(totals.find("Strat1::Pass2"), totals.end());
    EXPECT_NE(totals.find("Strat2::Pass1"), totals.end());

    MonitorManager::Instance().EndStage("Pass");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorPassScope, GetStartTimeReturnsConstructionTime)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");

    auto before = std::chrono::high_resolution_clock::now();
    MonitorPassCompileScope scope("StratX", "Pass0", 0, "func_0", 0, 500);
    auto after = std::chrono::high_resolution_clock::now();

    auto startTime = scope.GetStartTime();
    EXPECT_GE(startTime, before);
    EXPECT_LE(startTime, after);

    scope.Finish(true);
    MonitorManager::Instance().NotifyCompilationFinished();
}
