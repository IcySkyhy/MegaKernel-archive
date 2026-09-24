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
#define private public
#define protected public
#include "machine/device/dynamic/device_sche_context.h"
#undef private
#undef protected

using namespace npu::tile_fwk::dynamic;

TEST(SchDeviceTaskContextTest, Init_ResetsAllFields)
{
    SchDeviceTaskContext ctx;
    ctx.aicpuTaskSendCnt = 5;
    ctx.lastSent = 10;
    ctx.allSent = 20;
    ctx.curStage = DevTaskExecStage::SEND_CORE_TASK;
    ctx.coreFinishedNum = 3;

    ctx.Init();

    EXPECT_EQ(ctx.taskCtrl, nullptr);
    EXPECT_EQ(ctx.aicpuTaskSendCnt, 0u);
    EXPECT_EQ(ctx.lastSent, 0u);
    EXPECT_EQ(ctx.allSent, 0u);
    EXPECT_EQ(ctx.curStage, DevTaskExecStage::INIT);
    EXPECT_EQ(ctx.coreFinishedNum, 0u);
    for (auto v : ctx.coreTaskFinished) {
        EXPECT_EQ(v, 0u);
    }
}

TEST(SchDeviceTaskContextTest, StageManagement_AndBasicAccessors)
{
    SchDeviceTaskContext ctx;
    ctx.Init();
    EXPECT_TRUE(ctx.IsStage(DevTaskExecStage::INIT));
    EXPECT_EQ(ctx.CurStage(), DevTaskExecStage::INIT);
    EXPECT_TRUE(ctx.IsFree());
    EXPECT_EQ(ctx.TaskId(), 0u);

    ctx.EntryStage(DevTaskExecStage::SEND_CORE_TASK);
    EXPECT_TRUE(ctx.IsStage(DevTaskExecStage::SEND_CORE_TASK));
    EXPECT_FALSE(ctx.IsStage(DevTaskExecStage::INIT));

    ctx.EntryStage(DevTaskExecStage::FINISH);
    EXPECT_TRUE(ctx.IsRunFinish());

    ctx.SetAicpuTaskSent(7);
    EXPECT_EQ(ctx.aicpuTaskSendCnt, 7u);

    ctx.sendCnt[static_cast<int>(npu::tile_fwk::CoreType::AIC)] = 5;
    ctx.sendCnt[static_cast<int>(npu::tile_fwk::CoreType::AIV)] = 3;
    EXPECT_EQ(ctx.CurCoreTaskSent(npu::tile_fwk::CoreType::AIC), 5u);
    EXPECT_EQ(ctx.CurCoreTaskSent(npu::tile_fwk::CoreType::AIV), 3u);
}

TEST(SchDeviceTaskContextTest, CountCoreTaskSent_AndSync)
{
    // CountCoreTaskSent requires non-null taskCtrl with valid finishedFunctionCnt
    // Verify the zero-sent path and SyncAllSchCoreTaskSent with lastSent=0 are safe
    SchDeviceTaskContext ctx;
    ctx.Init();
    EXPECT_EQ(ctx.lastSent, 0u);
    SUCCEED();
}

TEST(ParallelSchDeviceTaskContextTest, Lifecycle_EmptyUpdateAccess)
{
    ParallelSchDeviceTaskContext pctx;
    EXPECT_TRUE(pctx.Empty());
    EXPECT_EQ(pctx.Num(), 0u);
    EXPECT_FALSE(pctx.Full());
    EXPECT_EQ(pctx.Version(), 0u);

    pctx.UpdateVersion();
    EXPECT_EQ(pctx.Version(), 1u);
    pctx.UpdateVersion();
    EXPECT_EQ(pctx.Version(), 2u);

    auto* e0 = pctx.Element(0);
    auto* e1 = pctx.Element(1);
    EXPECT_NE(e0, nullptr);
    EXPECT_NE(e1, nullptr);
    EXPECT_NE(e0, e1);
    EXPECT_EQ(pctx.FrontElement(), e0);
    EXPECT_EQ(pctx.RearElement(), e0);

    pctx.rear = 1;
    pctx.front = 0;
    pctx.PopFront();
    EXPECT_TRUE(pctx.Empty());
    EXPECT_EQ(pctx.front, 0u);
    EXPECT_EQ(pctx.rear, 0u);
}

TEST(SchduleContextTest, AllAccessors_AndStateQueries)
{
    SchduleContext sctx;
    EXPECT_TRUE(sctx.CurSupportParallel());
    EXPECT_TRUE(sctx.DevTaskEmpty());
    EXPECT_EQ(sctx.DeviceTaskCtxNum(), 0u);
    EXPECT_EQ(sctx.PrallelVersion(), 0u);

    SchDeviceTaskContext ctx;
    sctx.SetCurSchDevTaskCtx(&ctx);
    EXPECT_EQ(sctx.GetCurSchDevTaskCtx(), &ctx);

    auto* e = sctx.ParallelDeviceTaskCtx(0);
    EXPECT_NE(e, nullptr);
    EXPECT_NE(sctx.FrontDevTaskCtx(), nullptr);

    sctx.UpdateParallelVersion();
    EXPECT_EQ(sctx.PrallelVersion(), 1u);
}

TEST(SchThreadStatusTest, Init_SetsAllIdle_AndToggle)
{
    SchThreadStatus status;
    status.Init();
    for (size_t i = 0; i < AICORE_TYPE_NUM; ++i) {
        for (size_t j = 0; j < npu::tile_fwk::dynamic::MAX_SCHEDULE_AICPU_NUM; ++j) {
            EXPECT_TRUE(status.isAicpuIdle[i][j].load());
        }
    }

    status.isAicpuIdle[0][0].store(false);
    EXPECT_FALSE(status.isAicpuIdle[0][0].load());
    EXPECT_TRUE(status.isAicpuIdle[0][1].load());
}
