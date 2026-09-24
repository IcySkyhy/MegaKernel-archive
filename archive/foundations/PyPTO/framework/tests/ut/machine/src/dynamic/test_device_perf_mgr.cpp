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
#include <fstream>
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/dynamic/device_perf.h"

using namespace npu::tile_fwk::dynamic;

TEST(PerfettoMgrTest, SingletonAndAllocRecord)
{
    auto& inst1 = PerfettoMgr::Instance();
    auto& inst2 = PerfettoMgr::Instance();
    EXPECT_EQ(&inst1, &inst2);

    auto* rec = inst1.allocRecord(0);
    EXPECT_NE(rec, nullptr);
}

TEST(PerfettoMgrTest, PerfBeginEnd_AndPerfEvent)
{
    auto& mgr = PerfettoMgr::Instance();
    mgr.PerfBegin(PERF_EVT_INIT, 0);
    mgr.PerfEnd(PERF_EVT_INIT, 0);

    uint64_t start = GetCycles();
    uint64_t end = start + 1000;
    mgr.PerfEvent(PERF_EVT_INIT, 0, start, end, "test_event");
    SUCCEED();
}

TEST(PerfettoMgrTest, Dump_WritesFile)
{
    auto& mgr = PerfettoMgr::Instance();
    mgr.allocRecord(1);
    std::string tmpFile = "/tmp/test_perf_dump.txt";
    mgr.Dump(tmpFile);
    std::ifstream f(tmpFile);
    EXPECT_TRUE(f.good());
    std::remove(tmpFile.c_str());
}

TEST(PerfettoMgrTest, Array_PushPopFullEmpty_AndAlloc)
{
    PerfettoMgr::Array<int, 4> arr;
    EXPECT_TRUE(arr.Empty());
    EXPECT_FALSE(arr.Full());

    arr.Push(1);
    arr.Push(2);
    arr.Push(3);
    arr.Push(4);
    EXPECT_TRUE(arr.Full());
    EXPECT_EQ(arr.Top(), 4);

    arr.Pop();
    EXPECT_FALSE(arr.Full());
    EXPECT_EQ(arr.Top(), 3);

    int* p = arr.Alloc();
    EXPECT_NE(p, nullptr);
    *p = 42;
    EXPECT_EQ(arr.Top(), 42);
}

TEST(PerfEvtMgrTest, SingletonAndProfControl)
{
    auto& inst1 = PerfEvtMgr::Instance();
    auto& inst2 = PerfEvtMgr::Instance();
    EXPECT_EQ(&inst1, &inst2);

    EXPECT_FALSE(inst1.GetIsOpenProf());
    inst1.SetIsOpenProf(true, 0x1000);
    EXPECT_TRUE(inst1.GetIsOpenProf());
    inst1.SetIsOpenProf(false);
    EXPECT_FALSE(inst1.GetIsOpenProf());

    inst1.AddCtrlTurn();
    inst1.AddScheduleTurn();
}

TEST(PerfEvtMgrTest, PerfBeginEnd_AndDump)
{
    auto& mgr = PerfEvtMgr::Instance();
    mgr.PerfBegin(PERF_EVT_INIT);
    mgr.PerfEnd(PERF_EVT_INIT);
    mgr.PerfBegin(PERF_EVT_EXEC_DYN);
    mgr.PerfEnd(PERF_EVT_EXEC_DYN);
    mgr.Dump();
    SUCCEED();
}

TEST(PerfEvtMgrTest, RepeatPuts_AndPerfTrace)
{
    PerfEvtMgr::RepeatPuts('=', 40);
    PerfEvtMgr::RepeatPuts('-', 10);

    auto& mgr = PerfEvtMgr::Instance();
    mgr.PerfTrace(0, MAX_USED_AICPU_NUM + 1, 0);
    mgr.PerfTrace(0, 0, 0);
    SUCCEED();
}

TEST(PerfEvtMgrTest, SetIsOpenProf_ExceedMaxTurn)
{
    auto& mgr = PerfEvtMgr::Instance();
    for (uint32_t i = 0; i < MAX_ROUND_NUM + 1; ++i) {
        mgr.AddCtrlTurn();
    }
    mgr.SetIsOpenProf(true, 0x2000);
    EXPECT_FALSE(mgr.GetIsOpenProf());
}

TEST(PerfFreeFunctionsTest, AllPerfFunctions_NoCrash)
{
    AutoScopedPerf scoped(PERF_EVT_INIT);

    PerfBegin(PERF_EVT_INIT);
    PerfEnd(PERF_EVT_INIT);

    PerfMtBegin(PERF_EVT_INIT, 0);
    PerfMtEnd(PERF_EVT_INIT, 0);

    uint64_t start = GetCycles();
    PerfMtEvent(PERF_EVT_INIT, 0, start, start + 100, "test");

    PerfMtTrace(0, 0, 0);
}
