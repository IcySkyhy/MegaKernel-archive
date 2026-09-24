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
 * \file test_perf_event_sampler.cpp
 * \brief Unit tests for perf_event_sampler.cpp covering default-compiled metric helpers
 */

#include <cstdint>
#include <gtest/gtest.h>

#include "machine/device/dynamic/perf_event_sampler.h"

using namespace npu::tile_fwk;

class TestPerfEventSampler : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestPerfEventSampler, PerfCacheMetrics_DefaultValues)
{
    PerfCacheMetrics metrics;
    EXPECT_FALSE(metrics.valid);
    EXPECT_DOUBLE_EQ(metrics.missRate, 0.0);
}

TEST_F(TestPerfEventSampler, PerfDerivedMetrics_DefaultValues)
{
    PerfDerivedMetrics metrics;
    EXPECT_DOUBLE_EQ(metrics.ipc, 0.0);
    EXPECT_DOUBLE_EQ(metrics.cpi, 0.0);
    EXPECT_DOUBLE_EQ(metrics.branchMissRate, 0.0);
    EXPECT_FALSE(metrics.l1dCache.valid);
    EXPECT_FALSE(metrics.llCache.valid);
}

TEST_F(TestPerfEventSampler, DividePerfCounter_ZeroDivisorReturnsZero)
{
    EXPECT_DOUBLE_EQ(DividePerfCounter(10, 0), 0.0);
}

TEST_F(TestPerfEventSampler, DividePerfCounter_ValidDivisorReturnsQuotient)
{
    EXPECT_DOUBLE_EQ(DividePerfCounter(10, 4), 2.5);
}

TEST_F(TestPerfEventSampler, PercentPerfCounter_ZeroDivisorReturnsZero)
{
    EXPECT_DOUBLE_EQ(PercentPerfCounter(10, 0), 0.0);
}

TEST_F(TestPerfEventSampler, PercentPerfCounter_ValidDivisorReturnsPercent)
{
    EXPECT_DOUBLE_EQ(PercentPerfCounter(1, 4), 25.0);
}

TEST_F(TestPerfEventSampler, BuildPerfCacheMetrics_ZeroRefsReturnsInvalid)
{
    PerfCacheMetrics metrics = BuildPerfCacheMetrics(0, 1);

    EXPECT_FALSE(metrics.valid);
    EXPECT_DOUBLE_EQ(metrics.missRate, 0.0);
}

TEST_F(TestPerfEventSampler, BuildPerfCacheMetrics_ValidRefsReturnsMissRate)
{
    PerfCacheMetrics metrics = BuildPerfCacheMetrics(80, 8);

    EXPECT_TRUE(metrics.valid);
    EXPECT_DOUBLE_EQ(metrics.missRate, 10.0);
}

TEST_F(TestPerfEventSampler, BuildPerfDerivedMetrics_CalculatesExpectedValues)
{
    uint64_t counts[MAX_PERF_EVENT_NUM] = {0};
    counts[IDX_CPU_CYCLES] = 200;
    counts[IDX_INSTRUCTIONS] = 100;
    counts[IDX_BRANCH_INST] = 40;
    counts[IDX_BRANCH_MISS] = 2;
    counts[IDX_L1D_CACHE_REFS] = 80;
    counts[IDX_L1D_CACHE_MISSES] = 8;
    counts[IDX_LL_CACHE_REFS] = 60;
    counts[IDX_LL_CACHE_MISSES] = 3;

    PerfDerivedMetrics metrics = BuildPerfDerivedMetrics(counts);

    EXPECT_DOUBLE_EQ(metrics.ipc, 0.5);
    EXPECT_DOUBLE_EQ(metrics.cpi, 2.0);
    EXPECT_DOUBLE_EQ(metrics.branchMissRate, 5.0);
    EXPECT_TRUE(metrics.l1dCache.valid);
    EXPECT_DOUBLE_EQ(metrics.l1dCache.missRate, 10.0);
    EXPECT_TRUE(metrics.llCache.valid);
    EXPECT_DOUBLE_EQ(metrics.llCache.missRate, 5.0);
}

TEST_F(TestPerfEventSampler, BuildPerfDerivedMetrics_ZeroDivisorsReturnsZeroMetrics)
{
    uint64_t counts[MAX_PERF_EVENT_NUM] = {0};

    PerfDerivedMetrics metrics = BuildPerfDerivedMetrics(counts);

    EXPECT_DOUBLE_EQ(metrics.ipc, 0.0);
    EXPECT_DOUBLE_EQ(metrics.cpi, 0.0);
    EXPECT_DOUBLE_EQ(metrics.branchMissRate, 0.0);
    EXPECT_FALSE(metrics.l1dCache.valid);
    EXPECT_FALSE(metrics.llCache.valid);
}

TEST_F(TestPerfEventSampler, AicpuPerfEventSampler_DefaultSwitchNoopMethods)
{
    AicpuPerfEventSampler sampler;

    sampler.Begin();
    sampler.End();
    sampler.Dump();

    SUCCEED();
}

TEST_F(TestPerfEventSampler, GetAicpuPerfEventSampler_ReturnsSameThreadLocalInstance)
{
    AicpuPerfEventSampler& sampler1 = GetAicpuPerfEventSampler();
    AicpuPerfEventSampler& sampler2 = GetAicpuPerfEventSampler();

    EXPECT_EQ(&sampler1, &sampler2);
}

TEST_F(TestPerfEventSampler, AicpuPerfScopedSampler_DefaultSwitchConstructs)
{
    {
        AicpuPerfScopedSampler sampler("ut_perf_event_sampler");
    }

    SUCCEED();
}
