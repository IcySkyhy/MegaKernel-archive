/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/dynamic/device_common.h"

using namespace npu::tile_fwk::dynamic;

TEST(DeviceUtilsTest, IsDeviceMode_AndTimeFunctions)
{
    EXPECT_FALSE(IsDeviceMode());

    uint64_t t1 = GetTimeMonotonic();
    EXPECT_GT(t1, 0u);
    uint64_t t2 = GetTimeMonotonic();
    EXPECT_GE(t2, t1);

    EXPECT_GT(GetCycles(), 0u);
    EXPECT_GT(GetFreq(), 0u);
    EXPECT_GT(CurrentTime(), 0u);
}

TEST(DeviceUtilsTest, PtrToPtr_ConvertsPointerTypes)
{
    int32_t val = 42;
    int32_t* p = &val;
    uint8_t* bp = PtrToPtr<int32_t, uint8_t>(p);
    EXPECT_EQ(reinterpret_cast<void*>(bp), reinterpret_cast<void*>(p));

    const int32_t* cp = &val;
    const uint8_t* cbp = PtrToPtr<int32_t, uint8_t>(cp);
    EXPECT_EQ(reinterpret_cast<const void*>(cbp), reinterpret_cast<const void*>(cp));
}

TEST(DeviceUtilsTest, PtrToValue_AndValueToPtr_Roundtrip)
{
    int x = 10;
    uint64_t v = PtrToValue(&x);
    EXPECT_EQ(v, reinterpret_cast<uintptr_t>(&x));
    EXPECT_EQ(ValueToPtr(v), &x);
    EXPECT_EQ(ValueToPtr(0u), nullptr);

    volatile int vy = 20;
    uint64_t vy2 = PtrToValue(&vy);
    EXPECT_EQ(vy2, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&vy)));

    EXPECT_EQ(PtrToValue(static_cast<const void*>(nullptr)), 0u);
}

TEST(DeviceUtilsTest, VPtrToValue_ConvertsVectorOfPointers)
{
    int a = 1, b = 2, c = 3;
    std::vector<void*> ptrs = {&a, &b, &c};
    auto vals = VPtrToValue(ptrs);
    EXPECT_EQ(vals.size(), 3u);
    EXPECT_EQ(vals[0], PtrToValue(&a));
    EXPECT_EQ(vals[1], PtrToValue(&b));
    EXPECT_EQ(vals[2], PtrToValue(&c));

    std::vector<void*> empty;
    auto emptyVals = VPtrToValue(empty);
    EXPECT_TRUE(emptyVals.empty());
}

TEST(DeviceUtilsTest, PerfEventNames_AndEnableArrays)
{
    EXPECT_NE(std::string(PerfEventName[PERF_EVT_EXEC_DYN]), "");
    EXPECT_NE(std::string(PerfEventName[PERF_EVT_INIT]), "");
    EXPECT_NE(std::string(PerfEventName[PERF_EVT_MAX]), "");

    EXPECT_TRUE(PerfEvtEnable[PERF_EVT_EXEC_DYN]);
    EXPECT_TRUE(PerfEvtEnable[PERF_EVT_MAX]);
}

TEST(DeviceUtilsTest, TimeoutConstants_AndMaps)
{
    EXPECT_EQ(TIMEOUT_A2A3_50US, 2500ULL);
    EXPECT_EQ(TIMEOUT_A2A3_1SEC, 50000000ULL);
    EXPECT_EQ(TIMEOUT_A5_50US, 50000ULL);
    EXPECT_EQ(TIMEOUT_A5_1SEC, 1000000000ULL);
    EXPECT_EQ(HAND_SHAKE_TIMEOUT_A2A3_CYCLES, 48000000000ULL);
    EXPECT_EQ(HAND_SHAKE_TIMEOUT_A5_CYCLES, 960000000000ULL);

    EXPECT_EQ(TIMEOUT_MAP_A2A3[TIMEOUT_INDEX_50US], TIMEOUT_A2A3_50US);
    EXPECT_EQ(TIMEOUT_MAP_A2A3[TIMEOUT_INDEX_1SEC], TIMEOUT_A2A3_1SEC);
    EXPECT_EQ(TIMEOUT_MAP_A2A3[TIMEOUT_INDEX_10SEC], TIMEOUT_A2A3_10SEC);
    EXPECT_EQ(TIMEOUT_MAP_A2A3[TIMEOUT_INDEX_1MIN], TIMEOUT_A2A3_1MIN);
    EXPECT_EQ(TIMEOUT_MAP_A2A3[TIMEOUT_INDEX_20MIN], TIMEOUT_A2A3_20MIN);
    EXPECT_EQ(TIMEOUT_MAP_A2A3[TIMEOUT_INDEX_HAND_SHAKE], HAND_SHAKE_TIMEOUT_A2A3_CYCLES);

    EXPECT_EQ(TIMEOUT_MAP_A5[TIMEOUT_INDEX_50US], TIMEOUT_A5_50US);
    EXPECT_EQ(TIMEOUT_MAP_A5[TIMEOUT_INDEX_1SEC], TIMEOUT_A5_1SEC);
    EXPECT_EQ(TIMEOUT_MAP_A5[TIMEOUT_INDEX_10SEC], TIMEOUT_A5_10SEC);
    EXPECT_EQ(TIMEOUT_MAP_A5[TIMEOUT_INDEX_1MIN], TIMEOUT_A5_1MIN);
    EXPECT_EQ(TIMEOUT_MAP_A5[TIMEOUT_INDEX_20MIN], TIMEOUT_A5_20MIN);
    EXPECT_EQ(TIMEOUT_MAP_A5[TIMEOUT_INDEX_HAND_SHAKE], HAND_SHAKE_TIMEOUT_A5_CYCLES);
}

TEST(DeviceUtilsTest, NumericConstants)
{
    EXPECT_EQ(CTRL_CPU_THREAD_IDX, 0u);
    EXPECT_EQ(START_AICPU_NUM, 3);
    EXPECT_EQ(NUM_FIFTY, 50u);
    EXPECT_EQ(US_PER_SEC, 1000000u);
    EXPECT_EQ(NSEC_PER_USEC, 1000u);
    EXPECT_EQ(NSEC_PER_SEC, 1000000000u);
    EXPECT_EQ(MAX_MNG_AICORE_AVG_NUM, 8);
    EXPECT_EQ(CORE_IDX_AIV, 0u);
    EXPECT_EQ(CORE_IDX_AIC, 1u);
    EXPECT_EQ(AIV_NUM_PER_AI_CORE, 2u);
    EXPECT_EQ(INVALID_CORE_IDX, 0xFFu);
}

TEST(DeviceCommonTest, CalcSchAicpuNumByBlockDim_AllBranches)
{
    auto dav2201 = ArchInfo::DAV_2201;
    EXPECT_EQ(CalcSchAicpuNumByBlockDim(8, 5, dav2201), 1u);
    EXPECT_EQ(CalcSchAicpuNumByBlockDim(16, 5, dav2201), 2u);
    EXPECT_EQ(CalcSchAicpuNumByBlockDim(24, 5, dav2201), 3u);
    EXPECT_EQ(CalcSchAicpuNumByBlockDim(100, 5, dav2201), 3u);
    EXPECT_EQ(CalcSchAicpuNumByBlockDim(7, 5, dav2201), 1u);
    EXPECT_EQ(CalcSchAicpuNumByBlockDim(9, 5, dav2201), 2u);

    auto dav3510 = ArchInfo::DAV_3510;
    EXPECT_EQ(CalcSchAicpuNumByBlockDim(8, 7, dav3510), 1u);
    EXPECT_EQ(CalcSchAicpuNumByBlockDim(16, 7, dav3510), 2u);
    EXPECT_EQ(CalcSchAicpuNumByBlockDim(48, 7, dav3510), 6u);
    EXPECT_EQ(CalcSchAicpuNumByBlockDim(100, 7, dav3510), 6u);
    EXPECT_EQ(CalcSchAicpuNumByBlockDim(7, 7, dav3510), 1u);
}
