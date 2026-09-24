/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_determinism.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/determinism.h"

namespace npu::tile_fwk {
class DeterminismTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(DeterminismTest, Verify)
{
    constexpr int dtsize = 4;
    constexpr int dim = 32;
    constexpr int write = 16;
    auto tensor = std::make_shared<TraceRawTensorMemory>(TraceMemoryRange(0, dim * dim * dtsize),
                                                         std::vector<int64_t>({dim, dim}));

    TraceCopy copy(true, tensor, {0, 0}, {write, write}, false);
    auto leafCopyOutList = std::vector<TraceCopy>({copy});
    auto l0 = std::make_shared<TraceLeafTask>(TraceLeafTaskUid(0, 0, 0, 0, 0));
    auto l1 = std::make_shared<TraceLeafTask>(TraceLeafTaskUid(0, 0, 0, 1, 0));
    auto l2 = std::make_shared<TraceLeafTask>(TraceLeafTaskUid(0, 0, 0, 2, 0));
    auto l3 = std::make_shared<TraceLeafTask>(TraceLeafTaskUid(0, 0, 0, 3, 0));
    l0->GetCopyOutList() = leafCopyOutList;
    l1->GetCopyOutList() = leafCopyOutList;
    l2->GetCopyOutList() = leafCopyOutList;
    l3->GetCopyOutList() = leafCopyOutList;
    l0->AddSucc(l1->GetUid());
    l0->AddSucc(l2->GetUid());
    l1->AddSucc(l3->GetUid());
    l2->AddSucc(l3->GetUid());

    auto leafDict = std::map<TraceLeafTaskUid, std::shared_ptr<TraceLeafTask>>({
        {l0->GetUid(), l0},
        {l1->GetUid(), l1},
        {l2->GetUid(), l2},
        {l3->GetUid(), l3},
    });

    auto rootInList = std::vector<std::shared_ptr<TraceRawTensorMemory>>();
    auto rootOutList = std::vector<std::shared_ptr<TraceRawTensorMemory>>();
    auto root = std::make_shared<TraceRootTask>(TraceRootTaskUid(0, 0, 0));
    root->GetLeafTaskDict() = leafDict;
    root->GetIncastList() = rootInList;
    root->GetOutcastList() = rootOutList;
    auto rootDict = std::map<TraceRootTaskUid, std::shared_ptr<TraceRootTask>>({
        {root->GetUid(), root},
    });
    auto dev = std::make_shared<TraceDeviceTask>(TraceDeviceTaskUid(0));
    dev->GetRootTaskDict() = rootDict;

    auto depGraph = dev->BuildDependGraph();
    EXPECT_EQ(depGraph.GetLeafTaskDependIndexDict().find(l0->GetUid())->second, 0x0);
    EXPECT_EQ(depGraph.GetLeafTaskDependIndexDict().find(l1->GetUid())->second, 0x1);
    EXPECT_EQ(depGraph.GetLeafTaskDependIndexDict().find(l2->GetUid())->second, 0x2);
    EXPECT_EQ(depGraph.GetLeafTaskDependIndexDict().find(l3->GetUid())->second, 0x3);
    EXPECT_EQ(depGraph.GetReachDict()[0x1][0x2], npu::tile_fwk::INVALID_TRACE_TASK_DEPEND_INDEX);
    EXPECT_EQ(depGraph.GetReachDict()[0x2][0x1], npu::tile_fwk::INVALID_TRACE_TASK_DEPEND_INDEX);
    {
        auto raceList = dev->CheckRace(depGraph);
        EXPECT_EQ(raceList.size(), 1);
        EXPECT_EQ(raceList[0].GetKind(), TraceRaceKind::RACE_WRITE_WRITE);
        EXPECT_EQ(raceList[0].GetSrc().GetLeafTask(), l1);
        EXPECT_EQ(raceList[0].GetDst().GetLeafTask(), l2);
    }
    {
        l1->GetCopyOutList()[0].SetIsAtomicAdd(true);
        l2->GetCopyOutList()[0].SetIsAtomicAdd(true);
        auto raceList = dev->CheckRace(depGraph);
        EXPECT_EQ(raceList.size(), 1);
        EXPECT_EQ(raceList[0].GetKind(), TraceRaceKind::RACE_ATOMIC_ADD);
        EXPECT_EQ(raceList[0].GetSrc().GetLeafTask(), l1);
        EXPECT_EQ(raceList[0].GetDst().GetLeafTask(), l2);
    }
    {
        l2->GetCopyOutList()[0].SetOffset({0, write});
        auto raceList = dev->CheckRace(depGraph);
        EXPECT_EQ(raceList.size(), 0);
    }
}

template <typename T>
void LoadEvent(TraceExecution& exec, const T& event)
{
    std::string trace = "#trace:" + event.Dump();
    exec.LoadTrace(trace);
}

TEST_F(DeterminismTest, LoadTrace)
{
    using namespace npu::tile_fwk::schema;

    TraceExecution exec;
    LoadEvent(exec, REvent(RUid(0, 0x7, 1), RActWorkspace(Range(0x1734bc90, 0x17368290))));
    LoadEvent(exec, REvent(RUid(0, 0x7, 1), expr(std::vector<int>({0x6, 0x5, 0x0, 0x0, 0x3, 0x1}))));
    LoadEvent(exec, REvent(RUid(0, 0x7, 1), RActIncastCount(0x8)));
    LoadEvent(exec, REvent(RUid(0, 0x7, 1), RActIncast(incast(0x0), Range(0x17912d60, 0x17932d60))));
    LoadEvent(exec, REvent(RUid(0, 0x7, 1), RActIncast(incast(0x1), Range(0x17712d40, 0x17812d40))));
    LoadEvent(exec, REvent(RUid(0, 0x7, 1), RActOutcastCount(0x4)));
    LoadEvent(exec, REvent(RUid(0, 0x7, 1), RActOutcast(outcast(0x0), Range(0x17255290, 0x17255310))));
    LoadEvent(exec, REvent(RUid(0, 0x7, 1), RActOutcast(outcast(0x1), Range(0x17265290, 0x17265310))));
    LoadEvent(exec, REvent(RUid(0, 0x7, 1), RActRawTensorCount(0x18)));
    LoadEvent(exec, REvent(RUid(0, 0x7, 1), RActRawTensor(0x0, rawDesc(0x1, 0x1, 0x100000))));
    LoadEvent(exec, REvent(RUid(0, 0x7, 1), RActRawTensor(0x1, rawDesc(0x1, 0x3, 0x20000))));

    LoadEvent(exec, LEvent(LUid(0, 0x7, 1, 1, 2), coa(std::vector<std::string>({"1", "2", "?3", "4"}))));
    LoadEvent(exec, LEvent(LUid(0, 0x7, 1, 2, 2), coa(std::vector<std::string>({"5", "6", "?7", "8"}))));

    auto rootTask = exec.GetRootTask(TraceRootTaskUid(0, 0x7, 1));
    auto deviceTask = exec.GetDeviceTask(TraceDeviceTaskUid(0));
    EXPECT_EQ(deviceTask->GetRootTaskDict().begin()->second, rootTask);
    EXPECT_EQ(rootTask->GetWorkspaceMemoryRange().GetBegin(), 0x1734bc90);
    EXPECT_EQ(rootTask->GetExprList(), std::vector<int64_t>({0x6, 0x5, 0x0, 0x0, 0x3, 0x1}));
    EXPECT_EQ(rootTask->GetIncastList().size(), 0x8);
    EXPECT_EQ(rootTask->GetIncastList()[0]->GetMemoryRange().GetBegin(), 0x17912d60);
    EXPECT_EQ(rootTask->GetIncastList()[1]->GetMemoryRange().GetBegin(), 0x17712d40);
    EXPECT_EQ(rootTask->GetOutcastList().size(), 0x4);
    EXPECT_EQ(rootTask->GetOutcastList()[0]->GetMemoryRange().GetBegin(), 0x17255290);
    EXPECT_EQ(rootTask->GetOutcastList()[1]->GetMemoryRange().GetBegin(), 0x17265290);
    EXPECT_EQ(rootTask->GetRawTensorDescList().size(), 0x18);
    EXPECT_EQ(rootTask->GetRawTensorDescList()[0].GetSize(), 0x100000);
    EXPECT_EQ(rootTask->GetRawTensorDescList()[1].GetSize(), 0x20000);

    auto leafTask1 = exec.GetLeafTask(TraceLeafTaskUid(0, 0x7, 1, 1, 2));
    auto leafTask2 = exec.GetLeafTask(TraceLeafTaskUid(0, 0x7, 1, 2, 2));
    EXPECT_EQ(rootTask->GetLeafTaskDict()[leafTask1->GetUid()], leafTask1);
    EXPECT_EQ(rootTask->GetLeafTaskDict()[leafTask2->GetUid()], leafTask2);
    EXPECT_EQ(leafTask1->GetCoaList(),
              std::vector<TraceCoa>({TraceCoa(1), TraceCoa(2), TraceCoa(3, true), TraceCoa(4)}));
    EXPECT_EQ(leafTask2->GetCoaList(),
              std::vector<TraceCoa>({TraceCoa(5), TraceCoa(6), TraceCoa(7, true), TraceCoa(8)}));
}

} // namespace npu::tile_fwk
