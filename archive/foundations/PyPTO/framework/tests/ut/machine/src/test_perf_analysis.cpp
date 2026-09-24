/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <chrono>
#include "machine/host/perf_analysis.h"

namespace npu::tile_fwk {
class PerfAnalysisTest : public ::testing::Test {
protected:
    void SetUp() override { PerfAnalysis::Get().Reset(); }

    void TearDown() override {}

    void BusyWait(uint64_t microseconds)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        while ((uint64_t)elapsed < microseconds) {
            now = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        }
    }
};

TEST_F(PerfAnalysisTest, SingletonPattern)
{
    PerfAnalysis& instance1 = PerfAnalysis::Get();
    PerfAnalysis& instance2 = PerfAnalysis::Get();
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(PerfAnalysisTest, TraceBasicFunctionality)
{
    auto& perf = PerfAnalysis::Get();
    perf.Trace(TracePhase::RunDeviceInit);
    BusyWait(1000);
    perf.Trace(TracePhase::RunDevEnvReady);
    uint64_t totalTimeUs = perf.GetTraceTotalTimeUs();
    EXPECT_GT(totalTimeUs, 0);
    for (int i = 0; i < 5; ++i) {
        BusyWait(500);
        perf.Trace(TracePhase::RunDeviceInit);
    }
    totalTimeUs = perf.GetTraceTotalTimeUs();
    EXPECT_GT(totalTimeUs, 0);
}

TEST_F(PerfAnalysisTest, EventBasicFunctionality)
{
    auto& perf = PerfAnalysis::Get();
    perf.EventBegin(EventPhase::BuildCtrlFlowCache);
    BusyWait(2000);
    perf.EventEnd(EventPhase::BuildCtrlFlowCache);
    uint64_t eventTimeUs = perf.GetEventTotalTimeUs();
    EXPECT_GE(eventTimeUs, 2000);
}
} // namespace npu::tile_fwk
