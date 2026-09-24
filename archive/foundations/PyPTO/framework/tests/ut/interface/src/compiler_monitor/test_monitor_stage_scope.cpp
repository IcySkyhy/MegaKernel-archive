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
 * \file test_monitor_stage_scope.cpp
 * \brief Unit tests for MonitorStageScope RAII guard (no sleep)
 */

#include <string>
#include "gtest/gtest.h"
#include "monitor_test_fixture.h"
#include "interface/compiler_monitor/monitor_stage_scope.h"

using namespace npu::tile_fwk;

class TestMonitorStageScope : public MonitorTestFixtureBase {};

TEST_F(TestMonitorStageScope, SimpleStageScopePass)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    {
        MonitorStageScope scope("Pass");
        auto stages = MonitorManager::Instance().GetActiveStages();
        EXPECT_EQ(stages.size(), 1u);
        EXPECT_EQ(stages[0].stageName, "Pass");
    }

    auto stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 0u);

    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("Pass"), totals.end());
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorStageScope, SimpleStageScopeCodeGen)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(1);

    {
        MonitorStageScope scope("CodeGen");
        EXPECT_EQ(MonitorManager::Instance().GetCurrentStageName(), "CodeGen");
    }

    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("CodeGen"), totals.end());
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorStageScope, RootFuncStageScope)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetRootFuncCount(2);

    int idx = MonitorManager::Instance().PrepareNextRootFunc();
    {
        MonitorStageScope scope(STAGE_FUNC_TO_BIN, idx, "root_func_1", 10);
        auto stages = MonitorManager::Instance().GetActiveStages();
        EXPECT_EQ(stages.size(), 1u);
        EXPECT_EQ(stages[0].stageName, STAGE_FUNC_TO_BIN);
        EXPECT_EQ(stages[0].rootFuncIndex, idx);
        EXPECT_EQ(stages[0].rootFuncName, "root_func_1");
    }

    auto stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 0u);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorStageScope, NestedStageScopeCodeGenAndFuncToBin)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetRootFuncCount(1);

    int idx = MonitorManager::Instance().PrepareNextRootFunc();

    {
        MonitorStageScope codeGenScope("CodeGen");
        {
            MonitorStageScope funcToBinScope(STAGE_FUNC_TO_BIN, idx, "root_func_A", 50);
            auto stages = MonitorManager::Instance().GetActiveStages();
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
        }

        auto stages = MonitorManager::Instance().GetActiveStages();
        EXPECT_EQ(stages.size(), 0u);
    }

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorStageScope, MultipleRootFuncScopesConcurrent)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(4);
    MonitorManager::Instance().SetRootFuncCount(2);

    int idx1 = MonitorManager::Instance().PrepareNextRootFunc();
    int idx2 = MonitorManager::Instance().PrepareNextRootFunc();

    {
        MonitorStageScope scope1(STAGE_FUNC_TO_BIN, idx1, "root_A", 20);
        MonitorStageScope scope2(STAGE_FUNC_TO_BIN, idx2, "root_B", 30);

        auto stages = MonitorManager::Instance().GetActiveStages();
        EXPECT_EQ(stages.size(), 2u);
    }

    auto stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 0u);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorStageScope, HostMachineStageScope)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().BeginHostMachineCompileGroup(3);

    int step = MonitorManager::Instance().AllocHostMachineStepIndex();
    {
        MonitorStageScope scope(STAGE_HOST_MACHINE, step, "HM_Step1", 100);
        auto stages = MonitorManager::Instance().GetActiveStages();
        EXPECT_EQ(stages[0].stageName, STAGE_HOST_MACHINE);
        EXPECT_EQ(stages[0].rootFuncIndex, step);
    }

    auto stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 0u);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorStageScope, StageScopeAccumulatesElapsedTotals)
{
    MonitorManager::Instance().Initialize(true, 60, -1.0, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);

    {
        MonitorStageScope scope1("Pass");
    }
    {
        MonitorStageScope scope2("Pass");
    }

    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("Pass"), totals.end());
    EXPECT_GT(totals["Pass"], 0.0);

    MonitorManager::Instance().NotifyCompilationFinished();
}
