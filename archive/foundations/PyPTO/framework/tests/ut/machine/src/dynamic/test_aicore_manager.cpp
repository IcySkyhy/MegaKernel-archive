/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstring>

#define private public
#define protected public
#include "machine/device/dynamic/aicore_manager.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
struct MgrTestEnv {
    SchThreadStatus status;
    std::unique_ptr<AiCoreManager> mgr;
    std::unique_ptr<SchduleContext> ctxPtr;
    uint8_t* sharedBuf{nullptr};
    int64_t* regAddrsArr{nullptr};
    uint32_t* regMem{nullptr};
    static constexpr int AIC_NUM = 2;
    static constexpr int AIV_NUM = 0;

    MgrTestEnv() : ctxPtr(std::make_unique<SchduleContext>())
    {
        status.Init();
        mgr = std::make_unique<AiCoreManager>(status);
        mgr->aicNum_ = AIC_NUM;
        mgr->aivNum_ = AIV_NUM;
        mgr->aicStart_ = 0;
        mgr->aicEnd_ = AIC_NUM;
        mgr->aivStart_ = AIC_NUM;
        mgr->aivEnd_ = AIC_NUM;
        mgr->adjAicEnd_ = AIC_NUM;
        mgr->adjAivEnd_ = AIC_NUM;
        mgr->aicpuIdx_ = 0;
        mgr->schedIdx_ = 0;
        mgr->aicpuNum_ = 1;
        mgr->aicValidNum_ = AIC_NUM;
        mgr->archInfo_ = ArchInfo::DAV_2201;
        mgr->enableEslModel_ = false;
        mgr->hasAicpuTask_ = false;
        mgr->enableFairSch_ = false;
        mgr->enableL2CacheSch_ = false;
        mgr->validGetPgMask_ = false;
        mgr->releaseCoreByRegValFn_ = &AiCoreManager::ReleaseCoreByRegValByAsyncMode;
        mgr->pendingIds_.fill(AICORE_STATUS_INIT);
        mgr->runningIds_.fill(AICORE_STATUS_INIT);
        mgr->runningResolveIndexList_.fill(0);
        mgr->pendingResolveIndexList_.fill(0);
        mgr->pingPongFlag_.fill(0);
        mgr->context_ = ctxPtr.get();

        const size_t rawSize = SHARED_BUFFER_SIZE * (AIC_NUM + AIV_NUM);
        const size_t sharedSize = ((rawSize + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
        sharedBuf = reinterpret_cast<uint8_t*>(aligned_alloc(PAGE_SIZE, sharedSize));
        memset(sharedBuf, 0, sharedSize);
        regMem = new uint32_t[256]();
        regAddrsArr = new int64_t[AIC_NUM + AIV_NUM]();
        for (int i = 0; i < AIC_NUM; i++) {
            regAddrsArr[i] = reinterpret_cast<int64_t>(&regMem[i * 64]);
        }

        DeviceArgs devArgs{};
        devArgs.sharedBuffer = reinterpret_cast<int64_t>(sharedBuf);
        devArgs.coreRegAddr = reinterpret_cast<int64_t>(regAddrsArr);
        devArgs.nrAic = AIC_NUM;
        devArgs.nrAiv = AIV_NUM;
        devArgs.archInfo = ArchInfo::DAV_2201;
        devArgs.enableEslModel = false;
        mgr->aicoreHal_.Init(&devArgs, &mgr->aicoreProf_);
        for (int i = 0; i < AIC_NUM; i++) {
            mgr->aicoreHal_.GetPhyIdByBlockId(i) = i;
        }

        auto& ctx = *ctxPtr;
        ctx.coreStatusMgr.coreRunReadyCnt_[0] = 0;
        ctx.coreStatusMgr.coreRunReadyCnt_[1] = 0;
        ctx.coreStatusMgr.corePendReadyCnt_[0] = 0;
        ctx.coreStatusMgr.corePendReadyCnt_[1] = 0;
        ctx.coreStatusMgr.waitTaskCnt[0] = 0;
        ctx.coreStatusMgr.waitTaskCnt[1] = 0;
        for (uint32_t i = 0; i < MAX_AICORE_NUM; i++) {
            ctx.coreStatusMgr.coreIdxPosition_[i] = INVALID_COREIDX_POSITION;
        }
    }

    ~MgrTestEnv()
    {
        if (sharedBuf)
            free(sharedBuf);
        delete[] regMem;
        delete[] regAddrsArr;
    }

    SchduleContext& ctx() { return *ctxPtr; }
};
} // namespace

TEST(AicoreManagerTest, AddAndRemoveRunReadyCoreIdx)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);

    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(0, type);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreRunReadyCnt_[type], 1u);
    EXPECT_EQ(env.ctx().coreStatusMgr.runReadyCoreIdx_[type][0], 0u);

    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(1, type);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreRunReadyCnt_[type], 2u);

    env.ctx().coreStatusMgr.RemoveRunReadyCoreIdx(0, type);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreRunReadyCnt_[type], 1u);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreIdxPosition_[0], INVALID_COREIDX_POSITION);
}

TEST(AicoreManagerTest, RemoveRunReadyCoreIdx_InvalidPos)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.coreIdxPosition_[0] = INVALID_COREIDX_POSITION;
    env.ctx().coreStatusMgr.RemoveRunReadyCoreIdx(0, type);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreRunReadyCnt_[type], 0u);
}

TEST(AicoreManagerTest, RemoveReadyCoreIdxTail)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);

    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(0, type);
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(1, type);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreRunReadyCnt_[type], 2u);

    env.ctx().coreStatusMgr.RemoveReadyCoreIdxTail(1, type);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreRunReadyCnt_[type], 1u);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreIdxPosition_[1], INVALID_COREIDX_POSITION);
}

TEST(AicoreManagerTest, RemoveReadyCoreIdxTail_InvalidPos)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.coreIdxPosition_[0] = INVALID_COREIDX_POSITION;
    env.ctx().coreStatusMgr.RemoveReadyCoreIdxTail(0, type);
}

TEST(AicoreManagerTest, AicpuIsBusyAndIdle)
{
    MgrTestEnv env;
    env.mgr->AicpuIsBusy(CoreType::AIC);
    EXPECT_FALSE(env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].load());

    env.mgr->AicpuIsIdle(CoreType::AIC);
    EXPECT_TRUE(env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].load());
}

TEST(AicoreManagerTest, AicpuIsBusyAlreadyBusy)
{
    MgrTestEnv env;
    env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].store(false);
    env.mgr->AicpuIsBusy(CoreType::AIC);
    EXPECT_FALSE(env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].load());
}

TEST(AicoreManagerTest, AicpuIsIdleAlreadyIdle)
{
    MgrTestEnv env;
    env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].store(true);
    env.mgr->AicpuIsIdle(CoreType::AIC);
    EXPECT_TRUE(env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].load());
}

TEST(AicoreManagerTest, IsExistOtherAicpuIdle_SingleCpu)
{
    MgrTestEnv env;
    env.mgr->aicpuNum_ = 1;
    env.mgr->schedIdx_ = 0;
    EXPECT_FALSE(env.mgr->IsExistOtherAicpuIdle(CoreType::AIC));
}

TEST(AicoreManagerTest, IsExistOtherAicpuIdle_OtherIdle)
{
    MgrTestEnv env;
    env.mgr->aicpuNum_ = 2;
    env.mgr->schedIdx_ = 0;
    env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][1].store(true);
    EXPECT_TRUE(env.mgr->IsExistOtherAicpuIdle(CoreType::AIC));
}

TEST(AicoreManagerTest, IsExistOtherAicpuIdle_OtherBusy)
{
    MgrTestEnv env;
    env.mgr->aicpuNum_ = 2;
    env.mgr->schedIdx_ = 0;
    env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][1].store(false);
    EXPECT_FALSE(env.mgr->IsExistOtherAicpuIdle(CoreType::AIC));
}

TEST(AicoreManagerTest, UpdateAicoreEnd_Normal)
{
    int end = 0;
    AiCoreManager::UpdateAicoreEnd(4, 0, 2, 1, 0, end);
    EXPECT_EQ(end, 2);

    AiCoreManager::UpdateAicoreEnd(5, 0, 2, 1, 0, end);
    EXPECT_EQ(end, 3);

    AiCoreManager::UpdateAicoreEnd(5, 1, 2, 1, 0, end);
    EXPECT_EQ(end, 2);
}

TEST(AicoreManagerTest, UpdateAicoreEnd_ZeroPart)
{
    int end = 99;
    AiCoreManager::UpdateAicoreEnd(4, 0, 0, 1, 5, end);
    EXPECT_EQ(end, 5);
}

TEST(AicoreManagerTest, GetPhyIdByBlockId)
{
    MgrTestEnv env;
    env.mgr->aicoreHal_.GetPhyIdByBlockId(0) = 5;
    EXPECT_EQ(env.mgr->GetPhyIdByBlockId(0), 5);
}

TEST(AicoreManagerTest, SyncAicpuTaskFinish_NoAicpuTask)
{
    MgrTestEnv env;
    env.mgr->hasAicpuTask_ = false;
    SchDeviceTaskContext devTaskCtx{};
    EXPECT_EQ(env.mgr->SyncAicpuTaskFinish(&devTaskCtx), DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, IsNeedProcAicpuTask)
{
    MgrTestEnv env;
    env.mgr->hasAicpuTask_ = false;
    EXPECT_FALSE(env.mgr->IsNeedProcAicpuTask());
    env.mgr->hasAicpuTask_ = true;
    EXPECT_TRUE(env.mgr->IsNeedProcAicpuTask());
}

TEST(AicoreManagerTest, GetAllAiCoreNum)
{
    MgrTestEnv env;
    EXPECT_EQ(env.mgr->GetAllAiCoreNum(), MgrTestEnv::AIC_NUM + MgrTestEnv::AIV_NUM);
}

TEST(AicoreManagerTest, AicoreType)
{
    MgrTestEnv env;
    EXPECT_EQ(env.mgr->AicoreType(0), CoreType::AIC);
    EXPECT_EQ(env.mgr->AicoreType(1), CoreType::AIC);
    env.mgr->aivStart_ = 2;
    env.mgr->aivEnd_ = 4;
    env.mgr->aicEnd_ = 2;
    EXPECT_EQ(env.mgr->AicoreType(2), CoreType::AIV);
}

TEST(AicoreManagerTest, SetDotStatus)
{
    MgrTestEnv env;
    env.mgr->SetDotStatus(42);
    EXPECT_EQ(env.mgr->dotStatus_, 42);
}

TEST(AicoreManagerTest, SetSchedSyncMode)
{
    MgrTestEnv env;
    env.mgr->SetSchedSyncMode(1);
    EXPECT_EQ(env.mgr->releaseCoreByRegValFn_, &AiCoreManager::ReleaseCoreByRegValBySyncMode);
    env.mgr->SetSchedSyncMode(0);
    EXPECT_EQ(env.mgr->releaseCoreByRegValFn_, &AiCoreManager::ReleaseCoreByRegValByAsyncMode);
}

TEST(AicoreManagerTest, CheckAndResetReg_InvalidPgMask)
{
    MgrTestEnv env;
    env.mgr->validGetPgMask_ = false;
    EXPECT_TRUE(env.mgr->CheckAndResetReg());
}

TEST(AicoreManagerTest, ForEachManageAicore)
{
    MgrTestEnv env;
    int count = 0;
    env.mgr->ForEachManageAicore([&count](int) { count++; });
    EXPECT_EQ(count, MgrTestEnv::AIC_NUM);
}

TEST(AicoreManagerTest, ForEachManageAicoreReverse)
{
    MgrTestEnv env;
    std::vector<int> order;
    env.mgr->ForEachManageAicoreReverse([&order](int i) { order.push_back(i); });
    EXPECT_EQ(order.size(), static_cast<size_t>(MgrTestEnv::AIC_NUM));
    if (order.size() >= 2) {
        EXPECT_GT(order[0], order[1]);
    }
}

TEST(AicoreManagerTest, ForEachManageAicoreWithRet_Success)
{
    MgrTestEnv env;
    int ret = env.mgr->ForEachManageAicoreWithRet([](int) -> int { return DEVICE_MACHINE_OK; });
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, ForEachManageAicoreWithRet_Error)
{
    MgrTestEnv env;
    int ret = env.mgr->ForEachManageAicoreWithRet([](int) -> int { return DEVICE_MACHINE_ERROR; });
    EXPECT_NE(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, NeedsHwStopOnLastDevTask_NonDevice)
{
    MgrTestEnv env;
    EXPECT_FALSE(env.mgr->NeedsHwStopOnLastDevTask());
}

TEST(AicoreManagerTest, RuntimeCopyOutResolveCounterDecode)
{
    EXPECT_EQ(AiCoreManager::RuntimeCopyOutResolveCounterDecode(0x00050000), 0u);
    EXPECT_EQ(AiCoreManager::RuntimeCopyOutResolveCounterDecode(0x00050003), 3u);
    EXPECT_EQ(AiCoreManager::RuntimeCopyOutResolveCounterDecode(0xFFFF), 0xFFFFu);
}

TEST(AicoreManagerTest, RecordResolveTask_Normal)
{
    MgrTestEnv env;
    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    env.mgr->RecordResolveTask(ctx, finishCnt, 0, 0x1234, 0);
    EXPECT_EQ(finishCnt, 1u);
    EXPECT_EQ(ctx[0].finishIds, 0x1234u);
    EXPECT_EQ(ctx[0].finishCoreIdx, 0);
}

TEST(AicoreManagerTest, RecordResolveTask_Overflow)
{
    MgrTestEnv env;
    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = MAX_RESOLVE_TASK_NUM;
    env.mgr->RecordResolveTask(ctx, finishCnt, 0, 0x1234, 0);
    EXPECT_EQ(finishCnt, MAX_RESOLVE_TASK_NUM);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValBySyncMode_NoMatch)
{
    MgrTestEnv env;
    int coreIdx = 0;
    env.mgr->pendingIds_[coreIdx] = 0x999;
    env.mgr->runningIds_[coreIdx] = AICORE_TASK_INIT;

    uint32_t resloveParallelIdx = 0;
    int32_t ret = env.mgr->ResolveWhenSyncMode(CoreType::AIC, 0x100, TASK_FIN_STATE, coreIdx, resloveParallelIdx);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->pendingIds_[coreIdx], 0x999u);
}

TEST(AicoreManagerTest, DumpDfxWhenCoreNotStop)
{
    MgrTestEnv env;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.coreTaskFinished.fill(1);
    env.mgr->DumpDfxWhenCoreNotStop(&devTaskCtx);
}

TEST(AicoreManagerTest, DumpDfxWhenCoreNotStop_NotFinished)
{
    MgrTestEnv env;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.coreTaskFinished.fill(0);
    env.mgr->pendingIds_[0] = 0x100;
    env.mgr->runningIds_[0] = 0x200;
    env.mgr->DumpDfxWhenCoreNotStop(&devTaskCtx);
}

TEST(AicoreManagerTest, DumpLastWord)
{
    MgrTestEnv env;
    env.mgr->pendingIds_[0] = 0x100;
    env.mgr->runningIds_[0] = 0x200;
    env.mgr->DumpLastWord(0);
}

TEST(AicoreManagerTest, DumpLastWord_InitState)
{
    MgrTestEnv env;
    env.mgr->pendingIds_[0] = AICORE_TASK_INIT;
    env.mgr->runningIds_[0] = AICORE_TASK_INIT;
    env.mgr->DumpLastWord(0);
}

TEST(AicoreManagerTest, MarkCoreStoped)
{
    MgrTestEnv env;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.coreTaskFinished.fill(0);
    devTaskCtx.coreFinishedNum = 0;
    env.mgr->MarkCoreStoped(&devTaskCtx, 0);
    EXPECT_EQ(devTaskCtx.coreTaskFinished[0], 1);
    EXPECT_EQ(devTaskCtx.coreFinishedNum, 1u);
}

TEST(AicoreManagerTest, GetRunReadyCoreNum)
{
    MgrTestEnv env;
    env.ctx().coreStatusMgr.coreRunReadyCnt_[static_cast<int>(CoreType::AIC)] = 3;
    EXPECT_EQ(env.mgr->GetRunReadyCoreNum(CoreType::AIC), 3u);
}

TEST(AicoreManagerTest, GetReadyCoreNum_NoTail)
{
    MgrTestEnv env;
    env.ctx().coreStatusMgr.corePendReadyCnt_[static_cast<int>(CoreType::AIC)] = 2;
    EXPECT_EQ(env.mgr->GetReadyCoreNum(CoreType::AIC, false), 2u);
}

TEST(AicoreManagerTest, CheckIsTailBatch_SingleCpu)
{
    MgrTestEnv env;
    env.mgr->aicpuNum_ = 1;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    uint64_t remaining = 0;
    EXPECT_FALSE(env.mgr->CheckIsTailBatch(&devTaskCtx, CoreType::AIC, remaining));
}

TEST(AicoreManagerTest, SetSchduleContext)
{
    MgrTestEnv env;
    auto newCtx = std::make_unique<SchduleContext>();
    env.mgr->SetSchduleContext(newCtx.get());
    EXPECT_EQ(env.mgr->context_, newCtx.get());
}

TEST(AicoreManagerTest, HandShakeCorrectReadyCore)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.mgr->adjAicEnd_ = 1;
    env.mgr->aicEnd_ = 2;
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(0, type);
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(1, type);
    env.ctx().coreStatusMgr.corePendReadyCnt_[type] = 2;

    env.mgr->HandShakeCorrectReadyCore(CoreType::AIC);
    EXPECT_EQ(env.ctx().coreStatusMgr.corePendReadyCnt_[type], 1);
}

TEST(AicoreManagerTest, CalcAdjAicoreEnd_NonDeviceNonEsl)
{
    MgrTestEnv env;
    env.mgr->enableEslModel_ = false;
    env.mgr->CalcAdjAicoreEnd(nullptr, true);
    EXPECT_EQ(env.mgr->adjAicEnd_, env.mgr->aicEnd_);
    EXPECT_EQ(env.mgr->adjAivEnd_, env.mgr->aivEnd_);
}

TEST(AicoreManagerTest, PostRun_Success)
{
    MgrTestEnv env;
    env.mgr->PostRun(DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, BatchStopAllManagedCores_NonDeviceNonSim)
{
    MgrTestEnv env;
    env.mgr->BatchStopAllManagedCores();
}

TEST(AicoreManagerTest, ResetRegAll)
{
    MgrTestEnv env;
    env.mgr->ResetRegAll();
}

TEST(AicoreManagerTest, NormalStop)
{
    MgrTestEnv env;
    env.mgr->NormalStop();
}

TEST(AicoreManagerTest, AbnormalStop)
{
    MgrTestEnv env;
    env.mgr->AbnormalStop();
}

TEST(AicoreManagerTest, SendAllCoreStop_NonDeviceNonSim)
{
    MgrTestEnv env;
    env.mgr->SendAllCoreStop();
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_PendingFinished)
{
    MgrTestEnv env;
    int coreIdx = 0;
    int type = static_cast<int>(CoreType::AIC);
    uint32_t taskId = 0x100;
    env.mgr->pendingIds_[coreIdx] = taskId;
    env.mgr->runningIds_[coreIdx] = AICORE_TASK_INIT;
    env.ctx().coreStatusMgr.corePendReadyCnt_[type] = 0;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(taskId) | (static_cast<uint64_t>(TASK_FIN_STATE) << 31);
    uint32_t aicpuCallCode = 0;

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, aicpuCallCode, taskId, TASK_FIN_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->pendingIds_[coreIdx], AICORE_TASK_INIT);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], AICORE_TASK_INIT);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_RunningFinished)
{
    MgrTestEnv env;
    int coreIdx = 0;
    uint32_t runningTaskId = 0x200;
    env.mgr->pendingIds_[coreIdx] = AICORE_TASK_INIT;
    env.mgr->runningIds_[coreIdx] = runningTaskId;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(runningTaskId) | (static_cast<uint64_t>(TASK_FIN_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, runningTaskId, TASK_FIN_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], AICORE_TASK_INIT);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_PendingAck)
{
    MgrTestEnv env;
    int coreIdx = 0;
    uint32_t taskId = 0x300;
    env.mgr->pendingIds_[coreIdx] = taskId;
    env.mgr->runningIds_[coreIdx] = AICORE_TASK_INIT;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(taskId) | (static_cast<uint64_t>(TASK_ACK_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, taskId, TASK_ACK_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], taskId);
    EXPECT_EQ(env.mgr->pendingIds_[coreIdx], AICORE_TASK_INIT);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_InconsistentState)
{
    MgrTestEnv env;
    int coreIdx = 0;
    env.mgr->pendingIds_[coreIdx] = 0x100;
    env.mgr->runningIds_[coreIdx] = 0x200;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(0x999) | (static_cast<uint64_t>(TASK_FIN_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, 0x999, TASK_FIN_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, CalcAdjAicoreEnd_NullTaskCtrl)
{
    MgrTestEnv env;
    env.mgr->enableEslModel_ = true;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.taskCtrl = nullptr;
    env.mgr->CalcAdjAicoreEnd(&devTaskCtx, true);
    EXPECT_EQ(env.mgr->adjAicEnd_, env.mgr->aicEnd_);
    EXPECT_EQ(env.mgr->adjAivEnd_, env.mgr->aivEnd_);
}

TEST(AicoreManagerTest, CalcAdjAicoreEnd_MaxExceeds)
{
    MgrTestEnv env;
    env.mgr->enableEslModel_ = true;
    env.mgr->aicValidNum_ = 2;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    dynTask->maxC_ = 100;
    dynTask->maxV_ = 100;
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    env.mgr->CalcAdjAicoreEnd(&devTaskCtx, true);
    EXPECT_EQ(env.mgr->adjAicEnd_, env.mgr->aicEnd_);
    EXPECT_EQ(env.mgr->adjAivEnd_, env.mgr->aivEnd_);
}

TEST(AicoreManagerTest, CalcAdjAicoreEnd_MaxZero)
{
    MgrTestEnv env;
    env.mgr->enableEslModel_ = true;
    env.mgr->aicValidNum_ = 2;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    dynTask->maxC_ = 0;
    dynTask->maxV_ = 0;
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    env.mgr->CalcAdjAicoreEnd(&devTaskCtx, true);
    EXPECT_EQ(env.mgr->adjAicEnd_, env.mgr->aicEnd_);
}

TEST(AicoreManagerTest, CalcAdjAicoreEnd_MaxLessThanValid)
{
    MgrTestEnv env;
    env.mgr->enableEslModel_ = true;
    env.mgr->aicValidNum_ = 4;
    env.mgr->aicStart_ = 0;
    env.mgr->aicEnd_ = 4;
    env.mgr->aivStart_ = 4;
    env.mgr->aivEnd_ = 12;
    env.mgr->adjAicEnd_ = 4;
    env.mgr->adjAivEnd_ = 12;
    env.mgr->aicpuNum_ = 1;
    env.mgr->schedIdx_ = 0;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    dynTask->maxC_ = 2;
    dynTask->maxV_ = 4;
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    env.mgr->CalcAdjAicoreEnd(&devTaskCtx, true);
    EXPECT_LE(env.mgr->adjAicEnd_, env.mgr->aicEnd_);
}

TEST(AicoreManagerTest, CalcAdjAicoreEnd_NoUpdateNeeded)
{
    MgrTestEnv env;
    env.mgr->enableEslModel_ = true;
    env.mgr->aicValidNum_ = 2;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    dynTask->maxC_ = 100;
    dynTask->maxV_ = 100;
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    env.mgr->CalcAdjAicoreEnd(&devTaskCtx, false);
}

TEST(AicoreManagerTest, ProcessParallelDevTasks_Empty)
{
    MgrTestEnv env;
    env.ctx().schParallelDevTaskCtx.front = 0;
    env.ctx().schParallelDevTaskCtx.rear = 0;
    int32_t ret = env.mgr->ProcessParallelDevTasks();
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, ProcessParallelDevTasks_FreeCtx)
{
    MgrTestEnv env;
    auto& pctx = env.ctx().schParallelDevTaskCtx;
    pctx.front = 0;
    pctx.rear = 1;
    pctx.elements[0].isFree = 1;
    int32_t ret = env.mgr->ProcessParallelDevTasks();
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, UpdateParallelCtxAndCalcModifyFlag_Empty)
{
    MgrTestEnv env;
    env.ctx().schParallelDevTaskCtx.front = 0;
    env.ctx().schParallelDevTaskCtx.rear = 0;
    uint64_t flag = env.mgr->UpdateParallelCtxAndCalcModifyFlag(0, 0);
    EXPECT_EQ(flag, 0u);
}

TEST(AicoreManagerTest, UpdateParallelCtxAndCalcModifyFlag_FreeCtx)
{
    MgrTestEnv env;
    auto& pctx = env.ctx().schParallelDevTaskCtx;
    pctx.front = 0;
    pctx.rear = 1;
    pctx.elements[0].isFree = 1;
    uint64_t flag = env.mgr->UpdateParallelCtxAndCalcModifyFlag(0, 0);
    EXPECT_EQ(flag, 0u);
}

TEST(AicoreManagerTest, FillKernelArgsParallexDevTaskPreGathered)
{
    MgrTestEnv env;
    ParallelSchDeviceTaskContext pctx{};
    pctx.front = 0;
    pctx.rear = 0;
    int64_t localFuncData[SCH_DEVTASK_MAX_PARALLELISM] = {};
    uint32_t localDevTaskIds[SCH_DEVTASK_MAX_PARALLELISM] = {};
    env.mgr->FillKernelArgsParallexDevTaskPreGathered(&pctx, 0, localFuncData, localDevTaskIds);
}

TEST(AicoreManagerTest, RunTask_FinishStage)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.curStage = DevTaskExecStage::FINISH;
    int32_t ret = env.mgr->RunTask(&devTaskCtx);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, RunTask_WaitAllSchFinish)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    taskCtrl.finishedFunctionCnt.store(100);
    taskCtrl.runFlag.store(false);
    taskCtrl.runCnt.store(1);
    dynTask->devTask.coreFunctionCnt = 100;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.curStage = DevTaskExecStage::WAIT_ALL_SCH_FINISH;
    devTaskCtx.allSent = 100;
    int32_t ret = env.mgr->RunTask(&devTaskCtx);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, RunTask_InvalidStage)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.curStage = static_cast<DevTaskExecStage>(99);
    int32_t ret = env.mgr->RunTask(&devTaskCtx);
    EXPECT_NE(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, ProcessTaskLoop_Parallel)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    dynTask->devTask.coreFunctionCnt = 0;
    dynTask->parallelInfo.forId = 1;
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    taskCtrl.finishedFunctionCnt.store(0);
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.allSent = 0;
    bool isFinish = false;
    int32_t ret = env.mgr->ProcessTaskLoop(&devTaskCtx, isFinish);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, TrySendTaskDirectly_RunReadyAvailable)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(0, type);
    env.ctx().coreStatusMgr.corePendReadyCnt_[type] = 1;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    bool result = env.mgr->TrySendTaskDirectly(&devTaskCtx, type, 0x100);
    EXPECT_TRUE(result);
}

TEST(AicoreManagerTest, TrySendTaskDirectly_NoPendReady)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.coreRunReadyCnt_[type] = 0;
    env.ctx().coreStatusMgr.corePendReadyCnt_[type] = 0;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    bool result = env.mgr->TrySendTaskDirectly(&devTaskCtx, type, 0x100);
    EXPECT_FALSE(result);
}

TEST(AicoreManagerTest, TrySendTaskDirectly_PendReadyAvailable)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.coreRunReadyCnt_[type] = 0;
    env.ctx().coreStatusMgr.corePendReadyCnt_[type] = 1;
    env.ctx().coreStatusMgr.SetLastPendReadyCoreIdx(type, 0);
    env.mgr->pendingIds_[0] = AICORE_TASK_INIT;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    bool result = env.mgr->TrySendTaskDirectly(&devTaskCtx, type, 0x100);
    EXPECT_TRUE(result);
}

TEST(AicoreManagerTest, PushReadyTask_Normal)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.readyCount[CORE_IDX_AIC] = 0;

    int32_t ret = env.mgr->PushReadyTask(&devTaskCtx, CORE_IDX_AIC, 0x100);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(devTaskCtx.readyCount[CORE_IDX_AIC], 1);
}

TEST(AicoreManagerTest, PushReadyTask_L2CacheDirectSend)
{
    MgrTestEnv env;
    env.mgr->enableL2CacheSch_ = true;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(0, type);

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.readyCount[type] = 0;

    int32_t ret = env.mgr->PushReadyTask(&devTaskCtx, type, 0x100);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, BatchPushReadyQueue_EmptyReady)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.readyCount[CORE_IDX_AIC] = 0;
    devTaskCtx.readyCount[CORE_IDX_AIV] = 0;

    int32_t ret = env.mgr->BatchPushReadyQueue(&devTaskCtx);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, PushAicpuTaskQueue)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    uint32_t aicpuQueueElem[64];
    LockableQueueGeneric<uint32_t> aicpuQueue(64, aicpuQueueElem);
    devTaskCtx.readyAicpuFunctionQue = &aicpuQueue;

    env.mgr->PushAicpuTaskQueue(&devTaskCtx, 0x100);
    EXPECT_EQ(aicpuQueue.Size(), 1u);
}

TEST(AicoreManagerTest, AicpuIsBusyAndIdle_AIV)
{
    MgrTestEnv env;
    env.mgr->AicpuIsBusy(CoreType::AIV);
    EXPECT_FALSE(env.status.isAicpuIdle[static_cast<int>(CoreType::AIV)][0].load());
    env.mgr->AicpuIsIdle(CoreType::AIV);
    EXPECT_TRUE(env.status.isAicpuIdle[static_cast<int>(CoreType::AIV)][0].load());
}

TEST(AicoreManagerTest, IsExistOtherAicpuIdle_AIV)
{
    MgrTestEnv env;
    env.mgr->aicpuNum_ = 2;
    env.mgr->schedIdx_ = 0;
    env.status.isAicpuIdle[static_cast<int>(CoreType::AIV)][1].store(true);
    EXPECT_TRUE(env.mgr->IsExistOtherAicpuIdle(CoreType::AIV));
}

TEST(AicoreManagerTest, UpdateAiCoreBlockIndexSection)
{
    MgrTestEnv env;
    env.mgr->aicValidNum_ = 4;
    env.mgr->aicpuNum_ = 2;
    env.mgr->schedIdx_ = 0;
    env.mgr->archInfo_ = ArchInfo::DAV_2201;
    env.mgr->UpdateAiCoreBlockIndexSection();
    EXPECT_EQ(env.mgr->aicStart_, 0);
    EXPECT_EQ(env.mgr->aicEnd_, 2);
}

TEST(AicoreManagerTest, FillKernelArgsParallexDevTaskPreGathered_WithData)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    auto& pctx = env.ctx().schParallelDevTaskCtx;
    pctx.front = 0;
    pctx.rear = 2;
    pctx.elements[0].BindTaskCtrl(&taskCtrl);
    pctx.elements[1].BindTaskCtrl(&taskCtrl);

    int64_t localFuncData[SCH_DEVTASK_MAX_PARALLELISM] = {0x1000, 0x2000};
    uint32_t localDevTaskIds[SCH_DEVTASK_MAX_PARALLELISM] = {0, 1};
    env.mgr->FillKernelArgsParallexDevTaskPreGathered(&pctx, 0, localFuncData, localDevTaskIds);
}

TEST(AicoreManagerTest, ProcessTaskLoop_ParallelFinish)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    dynTask->devTask.coreFunctionCnt = 10;
    dynTask->parallelInfo.forId = 1;
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    taskCtrl.finishedFunctionCnt.store(10);
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.allSent = 10;
    bool isFinish = false;
    int32_t ret = env.mgr->ProcessTaskLoop(&devTaskCtx, isFinish);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_TRUE(isFinish);
}

TEST(AicoreManagerTest, TryBatchSendTask_WithPendReady)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.corePendReadyCnt_[type] = 2;
    env.ctx().coreStatusMgr.SetLastPendReadyCoreIdx(type, 0);
    env.mgr->pendingIds_[0] = AICORE_TASK_INIT;
    env.mgr->pendingIds_[1] = AICORE_TASK_INIT;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    uint32_t queueElem[64];
    LockableQueueGeneric<uint32_t> readyQueue(64, queueElem);
    readyQueue.UnsafeEnqueue(0x100);
    readyQueue.UnsafeEnqueue(0x200);

    uint64_t sent = env.mgr->TryBatchSendTask(&devTaskCtx, CoreType::AIC, &readyQueue, 0, 2);
    EXPECT_EQ(sent, 2u);
}

TEST(AicoreManagerTest, DispatchAiCoreTask_WithWaitTask)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.waitTaskCnt[type] = 1;
    env.ctx().coreStatusMgr.coreRunReadyCnt_[type] = 1;
    env.ctx().coreStatusMgr.runReadyCoreIdx_[type][0] = 0;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    uint32_t queueElem[64];
    LockableQueueGeneric<uint32_t> readyQueue(64, queueElem);
    devTaskCtx.readyAicCoreFunctionQue = &readyQueue;

    int32_t ret = env.mgr->DispatchAiCoreTask(&devTaskCtx, CoreType::AIC, &readyQueue, 0, 2);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, DispatchAiCoreTask_FairSchBusy)
{
    MgrTestEnv env;
    env.mgr->enableFairSch_ = true;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.coreRunReadyCnt_[type] = 0;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    uint32_t queueElem[64];
    LockableQueueGeneric<uint32_t> readyQueue(64, queueElem);
    devTaskCtx.readyAicCoreFunctionQue = &readyQueue;

    int32_t ret = env.mgr->DispatchAiCoreTask(&devTaskCtx, CoreType::AIC, &readyQueue, 0, 2);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, BatchPushReadyQueue_AivWithMixarch)
{
    MgrTestEnv env;
    env.mgr->aivStart_ = 2;
    env.mgr->aivEnd_ = 4;
    env.mgr->adjAivEnd_ = 4;
    env.mgr->archInfo_ = ArchInfo::DAV_3510;
    int aivType = static_cast<int>(CoreType::AIV);
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(2, aivType);

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.wrapManager.archInfo = ArchInfo::DAV_3510;
    devTaskCtx.readyCount[CORE_IDX_AIC] = 0;
    devTaskCtx.readyCount[CORE_IDX_AIV] = 1;
    devTaskCtx.readyIds[CORE_IDX_AIV][0] = 0x100;

    uint32_t queueElem[64];
    LockableQueueGeneric<uint32_t> readyQueue(64, queueElem);
    devTaskCtx.readyAivCoreFunctionQue = &readyQueue;

    int32_t ret = env.mgr->BatchPushReadyQueue(&devTaskCtx);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, TrySendTaskDirectly_AivType)
{
    MgrTestEnv env;
    env.mgr->aivStart_ = 2;
    env.mgr->aivEnd_ = 4;
    env.mgr->adjAivEnd_ = 4;
    int type = static_cast<int>(CoreType::AIV);
    env.ctx().coreStatusMgr.coreRunReadyCnt_[type] = 0;
    env.ctx().coreStatusMgr.corePendReadyCnt_[type] = 1;
    env.ctx().coreStatusMgr.SetLastPendReadyCoreIdx(type, 2);
    env.mgr->pendingIds_[2] = AICORE_TASK_INIT;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    bool result = env.mgr->TrySendTaskDirectly(&devTaskCtx, type, 0x100);
    EXPECT_TRUE(result);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_PendingFinishedWithRunning)
{
    MgrTestEnv env;
    int coreIdx = 0;
    uint32_t pendingTaskId = 0x100;
    uint32_t runningTaskId = 0x200;
    env.mgr->pendingIds_[coreIdx] = pendingTaskId;
    env.mgr->runningIds_[coreIdx] = runningTaskId;
    env.mgr->runningResolveIndexList_[coreIdx] = 0;
    env.mgr->pendingResolveIndexList_[coreIdx] = 0;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(pendingTaskId) | (static_cast<uint64_t>(TASK_FIN_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, pendingTaskId, TASK_FIN_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->pendingIds_[coreIdx], AICORE_TASK_INIT);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], AICORE_TASK_INIT);
}

TEST(AicoreManagerTest, UpdateAiCoreBlockIndexSection_NonDeviceNonSim)
{
    MgrTestEnv env;
    env.mgr->aicValidNum_ = 4;
    env.mgr->aicpuNum_ = 2;
    env.mgr->schedIdx_ = 0;
    env.mgr->archInfo_ = ArchInfo::DAV_2201;
    env.mgr->enableEslModel_ = false;
    env.ctx().coreStatusMgr.corePendReadyCnt_[static_cast<int>(CoreType::AIC)] = 0;
    env.ctx().coreStatusMgr.corePendReadyCnt_[static_cast<int>(CoreType::AIV)] = 0;
    env.mgr->UpdateAiCoreBlockIndexSection();
    EXPECT_EQ(env.mgr->aicStart_, 0);
    EXPECT_EQ(env.mgr->aicEnd_, 2);
    EXPECT_EQ(env.ctx().coreStatusMgr.corePendReadyCnt_[static_cast<int>(CoreType::AIC)], 2);
}

TEST(AicoreManagerTest, ProcessParallelDevTasks_WithNonFreeCtx)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    taskCtrl.notFree.store(true);
    taskCtrl.freeCnt.store(0);
    auto& pctx = env.ctx().schParallelDevTaskCtx;
    pctx.front = 0;
    pctx.rear = 1;
    pctx.elements[0].BindTaskCtrl(&taskCtrl);
    pctx.elements[0].curStage = DevTaskExecStage::FINISH;

    int32_t ret = env.mgr->ProcessParallelDevTasks();
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, CalcAdjAicoreEnd_Dav3510)
{
    MgrTestEnv env;
    env.mgr->enableEslModel_ = true;
    env.mgr->aicValidNum_ = 2;
    env.mgr->archInfo_ = ArchInfo::DAV_3510;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    dynTask->maxC_ = 1;
    dynTask->maxV_ = 2;
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    env.mgr->CalcAdjAicoreEnd(&devTaskCtx, true);
    EXPECT_LE(env.mgr->adjAicEnd_, env.mgr->aicEnd_);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_PendingAckWithRunning_Dav3510)
{
    MgrTestEnv env;
    env.mgr->archInfo_ = ArchInfo::DAV_3510;
    int coreIdx = 0;
    uint32_t pendingTaskId = 0x300;
    uint32_t runningTaskId = 0x400;
    env.mgr->pendingIds_[coreIdx] = pendingTaskId;
    env.mgr->runningIds_[coreIdx] = runningTaskId;
    env.mgr->runningResolveIndexList_[coreIdx] = 0;
    env.mgr->pendingResolveIndexList_[coreIdx] = 0;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(pendingTaskId) | (static_cast<uint64_t>(TASK_ACK_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, pendingTaskId, TASK_ACK_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], pendingTaskId);
    EXPECT_EQ(env.mgr->pendingIds_[coreIdx], AICORE_TASK_INIT);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_RunningFinished_Dav3510)
{
    MgrTestEnv env;
    env.mgr->archInfo_ = ArchInfo::DAV_3510;
    int coreIdx = 0;
    uint32_t runningTaskId = 0x500;
    env.mgr->pendingIds_[coreIdx] = AICORE_TASK_INIT;
    env.mgr->runningIds_[coreIdx] = runningTaskId;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(runningTaskId) | (static_cast<uint64_t>(TASK_FIN_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, runningTaskId, TASK_FIN_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], AICORE_TASK_INIT);
}

TEST(AicoreManagerTest, PostRun_Error)
{
    MgrTestEnv env;
    SPSCQueue<DeviceTaskCtrl*, DEFAULT_QUEUE_SIZE> taskQueue;
    env.mgr->taskQueue_ = &taskQueue;
    env.mgr->PostRun(DEVICE_MACHINE_ERROR);
}

TEST(AicoreManagerTest, BatchPushReadyQueue_AivReady)
{
    MgrTestEnv env;
    env.mgr->aivStart_ = 2;
    env.mgr->aivEnd_ = 4;
    env.mgr->adjAivEnd_ = 4;
    int aivType = static_cast<int>(CoreType::AIV);
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(2, aivType);

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.readyCount[CORE_IDX_AIC] = 0;
    devTaskCtx.readyCount[CORE_IDX_AIV] = 1;
    devTaskCtx.readyIds[CORE_IDX_AIV][0] = 0x100;

    int32_t ret = env.mgr->BatchPushReadyQueue(&devTaskCtx);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, UpdateRunReadyCoreNum_AivExpand)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIV);
    env.mgr->aivStart_ = 2;
    env.mgr->adjAivEnd_ = 3;
    env.ctx().coreStatusMgr.coreRunReadyCnt_[type] = 0;
    env.mgr->UpdateRunReadyCoreNum(env.mgr->aicEnd_, 2);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreRunReadyCnt_[type], 1);
}

TEST(AicoreManagerTest, UpdateRunReadyCoreNum_AivShrink)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIV);
    env.mgr->aivStart_ = 2;
    env.mgr->adjAivEnd_ = 2;
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(2, type);
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(3, type);
    env.mgr->UpdateRunReadyCoreNum(env.mgr->aicEnd_, 4);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreRunReadyCnt_[type], 0);
}

TEST(AicoreManagerTest, UpdateRunReadyCoreNum_AicExpand)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.mgr->aicStart_ = 0;
    env.mgr->adjAicEnd_ = 3;
    env.mgr->aicEnd_ = 2;
    env.ctx().coreStatusMgr.coreRunReadyCnt_[type] = 0;
    env.mgr->UpdateRunReadyCoreNum(2, env.mgr->adjAivEnd_);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreRunReadyCnt_[type], 1);
}

TEST(AicoreManagerTest, UpdateRunReadyCoreNum_AicShrink)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.mgr->aicStart_ = 0;
    env.mgr->adjAicEnd_ = 1;
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(0, type);
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(1, type);
    env.mgr->UpdateRunReadyCoreNum(2, env.mgr->adjAivEnd_);
    EXPECT_EQ(env.ctx().coreStatusMgr.coreRunReadyCnt_[type], 1);
}

TEST(AicoreManagerTest, DumpAicoreStatusWhenTimeout)
{
    MgrTestEnv env;
    bool handFlag[MAX_AICORE_NUM] = {false};
    handFlag[0] = true;
    handFlag[1] = false;
    env.mgr->DumpAicoreStatusWhenTimeout(handFlag);
}

TEST(AicoreManagerTest, HandShakeByGmForAic)
{
    MgrTestEnv env;
    bool aicAllSuccess = false;
    bool handFlag[MAX_AICORE_NUM] = {false};
    int handShakeNum = 0;
    int aicSucessCnt = 0;
    env.mgr->HandShakeByGmForAic(aicAllSuccess, handFlag, handShakeNum, aicSucessCnt);
}

TEST(AicoreManagerTest, HandShakeByGmForAiv)
{
    MgrTestEnv env;
    env.mgr->aivStart_ = 2;
    env.mgr->aivEnd_ = 4;
    env.mgr->adjAivEnd_ = 4;
    bool aivAllSuccess = false;
    bool handFlag[MAX_AICORE_NUM] = {false};
    int handShakeNum = 0;
    int aivSucessCnt = 0;
    env.mgr->HandShakeByGmForAiv(aivAllSuccess, handFlag, handShakeNum, aivSucessCnt);
}

TEST(AicoreManagerTest, HandShakeCorrectReadyCore_AIV)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIV);
    env.mgr->aivStart_ = 2;
    env.mgr->aivEnd_ = 4;
    env.mgr->adjAivEnd_ = 3;
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(2, type);
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(3, type);
    env.ctx().coreStatusMgr.corePendReadyCnt_[type] = 2;
    env.mgr->HandShakeCorrectReadyCore(CoreType::AIV);
    EXPECT_EQ(env.ctx().coreStatusMgr.corePendReadyCnt_[type], 1);
}

TEST(AicoreManagerTest, HandShakePostProc_NullCtx)
{
    MgrTestEnv env;
    env.mgr->HandShakePostProc(nullptr, false, false);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_LastDevTaskAck)
{
    MgrTestEnv env;
    int coreIdx = 0;
    uint32_t pendingTaskId = 0x600;
    env.mgr->pendingIds_[coreIdx] = pendingTaskId;
    env.mgr->runningIds_[coreIdx] = 0x700;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(pendingTaskId) | (static_cast<uint64_t>(TASK_ACK_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, pendingTaskId, TASK_ACK_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->pendingIds_[coreIdx], AICORE_TASK_INIT);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], pendingTaskId);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_PendingAckNoRunning)
{
    MgrTestEnv env;
    int coreIdx = 0;
    uint32_t pendingTaskId = 0x300;
    env.mgr->pendingIds_[coreIdx] = pendingTaskId;
    env.mgr->runningIds_[coreIdx] = AICORE_TASK_INIT;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(pendingTaskId) | (static_cast<uint64_t>(TASK_ACK_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, pendingTaskId, TASK_ACK_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], pendingTaskId);
    EXPECT_EQ(env.mgr->pendingIds_[coreIdx], AICORE_TASK_INIT);
    EXPECT_EQ(finishCnt, 0u);
}

TEST(AicoreManagerTest, FillKernelArgsParallexDevTaskPreGathered_WithMultipleIterations)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    auto& pctx = env.ctx().schParallelDevTaskCtx;
    pctx.front = 0;
    pctx.rear = 3;
    pctx.elements[0].BindTaskCtrl(&taskCtrl);
    pctx.elements[1].BindTaskCtrl(&taskCtrl);
    pctx.elements[2].BindTaskCtrl(&taskCtrl);

    int64_t localFuncData[SCH_DEVTASK_MAX_PARALLELISM] = {0x1000, 0x2000, 0x3000};
    uint32_t localDevTaskIds[SCH_DEVTASK_MAX_PARALLELISM] = {0, 1, 2};
    env.mgr->FillKernelArgsParallexDevTaskPreGathered(&pctx, 0, localFuncData, localDevTaskIds);
}

TEST(AicoreManagerTest, TryBatchSendTask_WithPendReadyCores)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.corePendReadyCnt_[type] = 2;
    env.ctx().coreStatusMgr.SetLastPendReadyCoreIdx(type, 0);
    env.mgr->pendingIds_[0] = AICORE_TASK_INIT;
    env.mgr->pendingIds_[1] = AICORE_TASK_INIT;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    uint32_t queueElem[64];
    LockableQueueGeneric<uint32_t> readyQueue(64, queueElem);
    readyQueue.UnsafeEnqueue(0x100);
    readyQueue.UnsafeEnqueue(0x200);

    uint64_t sent = env.mgr->TryBatchSendTask(&devTaskCtx, CoreType::AIC, &readyQueue, 0, 2);
    EXPECT_EQ(sent, 2u);
}

TEST(AicoreManagerTest, DispatchAiCoreTask_WithFairSchBusy)
{
    MgrTestEnv env;
    env.mgr->enableFairSch_ = true;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.coreRunReadyCnt_[type] = 0;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    uint32_t queueElem[64];
    LockableQueueGeneric<uint32_t> readyQueue(64, queueElem);
    devTaskCtx.readyAicCoreFunctionQue = &readyQueue;

    int32_t ret = env.mgr->DispatchAiCoreTask(&devTaskCtx, CoreType::AIC, &readyQueue, 0, 2);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, BatchPushReadyQueue_WithAivReady)
{
    MgrTestEnv env;
    env.mgr->aivStart_ = 2;
    env.mgr->aivEnd_ = 4;
    env.mgr->adjAivEnd_ = 4;
    env.mgr->archInfo_ = ArchInfo::DAV_3510;
    int aivType = static_cast<int>(CoreType::AIV);
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(2, aivType);

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.wrapManager.archInfo = ArchInfo::DAV_3510;
    devTaskCtx.readyCount[CORE_IDX_AIC] = 0;
    devTaskCtx.readyCount[CORE_IDX_AIV] = 1;
    devTaskCtx.readyIds[CORE_IDX_AIV][0] = 0x100;

    uint32_t queueElem[64];
    LockableQueueGeneric<uint32_t> readyQueue(64, queueElem);
    devTaskCtx.readyAivCoreFunctionQue = &readyQueue;

    int32_t ret = env.mgr->BatchPushReadyQueue(&devTaskCtx);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_RunningFinishedWithPendingInit)
{
    MgrTestEnv env;
    int coreIdx = 0;
    uint32_t runningTaskId = 0x500;
    env.mgr->pendingIds_[coreIdx] = AICORE_TASK_INIT;
    env.mgr->runningIds_[coreIdx] = runningTaskId;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(runningTaskId) | (static_cast<uint64_t>(TASK_FIN_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, runningTaskId, TASK_FIN_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], AICORE_TASK_INIT);
}

TEST(AicoreManagerTest, TrySendTaskDirectly_WithAivPendReady)
{
    MgrTestEnv env;
    env.mgr->aivStart_ = 2;
    env.mgr->aivEnd_ = 4;
    env.mgr->adjAivEnd_ = 4;
    int type = static_cast<int>(CoreType::AIV);
    env.ctx().coreStatusMgr.coreRunReadyCnt_[type] = 0;
    env.ctx().coreStatusMgr.corePendReadyCnt_[type] = 1;
    env.ctx().coreStatusMgr.SetLastPendReadyCoreIdx(type, 2);
    env.mgr->pendingIds_[2] = AICORE_TASK_INIT;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    bool result = env.mgr->TrySendTaskDirectly(&devTaskCtx, type, 0x100);
    EXPECT_TRUE(result);
}

TEST(AicoreManagerTest, ResolveDepForAicpuTask_EmptyQueue)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;

    uint32_t aicpuQueueElem[64];
    LockableQueueGeneric<uint32_t> aicpuQueue(64, aicpuQueueElem);
    dynTask->devTask.readyAicpuFunctionQue = reinterpret_cast<uint64_t>(&aicpuQueue);

    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    env.ctx().SetCurSchDevTaskCtx(&devTaskCtx);

    uint64_t taskCount = 0;
    int32_t ret = env.mgr->ResolveDepForAicpuTask(taskCount);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValBySyncMode_DifferentPending)
{
    MgrTestEnv env;
    int coreIdx = 0;
    env.mgr->pendingIds_[coreIdx] = 0x999;
    env.mgr->runningIds_[coreIdx] = AICORE_TASK_INIT;

    uint32_t resloveParallelIdx = 0;
    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(0x100) | (static_cast<uint64_t>(TASK_FIN_STATE) << 31);
    int32_t ret = env.mgr->ReleaseCoreByRegValBySyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                         finTaskRegVal, 0, 0x100, TASK_FIN_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->pendingIds_[coreIdx], 0x999u);
}

TEST(AicoreManagerTest, EnableDieScheduling_NegativeLoopDieId)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    DevAscendFunctionDuppedData duppedData{};
    duppedData.loopDieId_ = -1;
    dynTask->dynFuncDataCacheList[0].duppedData = &duppedData;

    bool result = env.mgr->EnableDieScheduling(&devTaskCtx, CoreType::AIC, MakeTaskID(0, 0));
    EXPECT_FALSE(result);
}

TEST(AicoreManagerTest, EnableDieScheduling_NoFairSch)
{
    MgrTestEnv env;
    env.mgr->enableFairSch_ = false;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.wrapManager.archInfo = ArchInfo::DAV_3510;
    devTaskCtx.wrapManager.dieId_ = DieId::DIE_0;

    DevAscendFunctionDuppedData duppedData{};
    duppedData.loopDieId_ = 0;
    dynTask->dynFuncDataCacheList[0].duppedData = &duppedData;

    bool result = env.mgr->EnableDieScheduling(&devTaskCtx, CoreType::AIC, MakeTaskID(0, 0));
    EXPECT_TRUE(result);
}

TEST(AicoreManagerTest, EnableDieScheduling_FairSchIdle)
{
    MgrTestEnv env;
    env.mgr->enableFairSch_ = true;
    env.mgr->aicpuNum_ = 2;
    env.mgr->schedIdx_ = 0;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.wrapManager.archInfo = ArchInfo::DAV_3510;
    devTaskCtx.wrapManager.dieId_ = DieId::DIE_0;
    devTaskCtx.wrapManager.curDie0MaxCpuId_ = 1;
    devTaskCtx.wrapManager.curDie1StartCpuId_ = 1;

    DevAscendFunctionDuppedData duppedData{};
    duppedData.loopDieId_ = 0;
    dynTask->dynFuncDataCacheList[0].duppedData = &duppedData;

    env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].store(true);

    bool result = env.mgr->EnableDieScheduling(&devTaskCtx, CoreType::AIC, MakeTaskID(0, 0));
    EXPECT_TRUE(result);
}

TEST(AicoreManagerTest, EnableDieScheduling_FairSchBusy)
{
    MgrTestEnv env;
    env.mgr->enableFairSch_ = true;
    env.mgr->aicpuNum_ = 2;
    env.mgr->schedIdx_ = 0;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.wrapManager.archInfo = ArchInfo::DAV_3510;
    devTaskCtx.wrapManager.dieId_ = DieId::DIE_0;
    devTaskCtx.wrapManager.curDie0MaxCpuId_ = 1;
    devTaskCtx.wrapManager.curDie1StartCpuId_ = 1;

    DevAscendFunctionDuppedData duppedData{};
    duppedData.loopDieId_ = 0;
    dynTask->dynFuncDataCacheList[0].duppedData = &duppedData;

    env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].store(false);

    bool result = env.mgr->EnableDieScheduling(&devTaskCtx, CoreType::AIC, MakeTaskID(0, 0));
    EXPECT_FALSE(result);
}

TEST(AicoreManagerTest, DispatchAiCoreTask_FairSchIdle)
{
    MgrTestEnv env;
    env.mgr->enableFairSch_ = true;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.coreRunReadyCnt_[type] = 1;
    env.ctx().coreStatusMgr.runReadyCoreIdx_[type][0] = 0;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    uint32_t queueElem[64];
    LockableQueueGeneric<uint32_t> readyQueue(64, queueElem);
    devTaskCtx.readyAicCoreFunctionQue = &readyQueue;

    int32_t ret = env.mgr->DispatchAiCoreTask(&devTaskCtx, CoreType::AIC, &readyQueue, 0, 2);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, UpdateParallelCtxAndCalcModifyFlag_WithNewerVersion)
{
    MgrTestEnv env;
    auto& pctx = env.ctx().schParallelDevTaskCtx;
    pctx.front = 0;
    pctx.rear = 1;
    pctx.elements[0].isFree = 0;
    pctx.elements[0].bindParallelCtxVersion = 5;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    pctx.elements[0].BindTaskCtrl(&taskCtrl);

    DynFuncHeader header{};
    header.seqNo = 0;
    dynTask->dynFuncDataList = &header;

    uint64_t flag = env.mgr->UpdateParallelCtxAndCalcModifyFlag(0, 3);
    EXPECT_NE(flag, 0u);
}

TEST(AicoreManagerTest, HandShakePostProc_WithCtx)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    env.ctx().coreStatusMgr.waitTaskCnt[0] = 2;
    env.ctx().coreStatusMgr.waitTaskCnt[1] = 3;
    env.mgr->HandShakePostProc(&devTaskCtx, false, false);
}

TEST(AicoreManagerTest, DumpAicoreStatusWhenTimeout_AllFlags)
{
    MgrTestEnv env;
    env.mgr->aivStart_ = 2;
    env.mgr->aivEnd_ = 4;
    bool handFlag[MAX_AICORE_NUM] = {false};
    handFlag[0] = true;
    handFlag[1] = true;
    handFlag[2] = true;
    handFlag[3] = false;
    env.mgr->DumpAicoreStatusWhenTimeout(handFlag);
}

TEST(AicoreManagerTest, BatchPushReadyQueue_AicReadyWithMixarch)
{
    MgrTestEnv env;
    env.mgr->archInfo_ = ArchInfo::DAV_3510;
    int aicType = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(0, aicType);

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.wrapManager.archInfo = ArchInfo::DAV_3510;
    devTaskCtx.readyCount[CORE_IDX_AIC] = 1;
    devTaskCtx.readyIds[CORE_IDX_AIC][0] = 0x100;
    devTaskCtx.readyCount[CORE_IDX_AIV] = 0;

    uint32_t queueElem[64];
    LockableQueueGeneric<uint32_t> readyQueue(64, queueElem);
    devTaskCtx.readyAicCoreFunctionQue = &readyQueue;

    int32_t ret = env.mgr->BatchPushReadyQueue(&devTaskCtx);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, PushReadyTask_ReadyCountFull)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    devTaskCtx.readyCount[CORE_IDX_AIC] = READY_ID_FIX_CACHE_NUM;
    for (uint32_t i = 0; i < READY_ID_FIX_CACHE_NUM; i++) {
        devTaskCtx.readyIds[CORE_IDX_AIC][i] = i + 1;
    }

    uint32_t queueElem[256];
    LockableQueueGeneric<uint32_t> readyQueue(256, queueElem);
    devTaskCtx.readyAicCoreFunctionQue = &readyQueue;
    devTaskCtx.wrapManager.archInfo = ArchInfo::DAV_2201;

    int32_t ret = env.mgr->PushReadyTask(&devTaskCtx, CORE_IDX_AIC, 0x200);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(devTaskCtx.readyCount[CORE_IDX_AIC], 1u);
}

TEST(AicoreManagerTest, TrySendTaskDirectly_FairSchOtherIdle)
{
    MgrTestEnv env;
    env.mgr->enableFairSch_ = true;
    env.mgr->aicpuNum_ = 2;
    env.mgr->schedIdx_ = 0;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreStatusMgr.coreRunReadyCnt_[type] = 0;
    env.ctx().coreStatusMgr.corePendReadyCnt_[type] = 1;
    env.status.isAicpuIdle[type][1].store(true);

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    bool result = env.mgr->TrySendTaskDirectly(&devTaskCtx, type, 0x100);
    EXPECT_FALSE(result);
}

TEST(AicoreManagerTest, CalcAdjAicoreEnd_UpdateCoreNum)
{
    MgrTestEnv env;
    env.mgr->enableEslModel_ = true;
    env.mgr->aicValidNum_ = 4;
    env.mgr->aicStart_ = 0;
    env.mgr->aicEnd_ = 4;
    env.mgr->aivStart_ = 4;
    env.mgr->aivEnd_ = 12;
    env.mgr->adjAicEnd_ = 4;
    env.mgr->adjAivEnd_ = 12;
    env.mgr->aicpuNum_ = 1;
    env.mgr->schedIdx_ = 0;

    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    dynTask->maxC_ = 2;
    dynTask->maxV_ = 4;
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);

    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(0, static_cast<int>(CoreType::AIC));
    env.ctx().coreStatusMgr.AddRunReadyCoreIdx(1, static_cast<int>(CoreType::AIC));

    env.mgr->CalcAdjAicoreEnd(&devTaskCtx, true);
    EXPECT_LE(env.mgr->adjAicEnd_, env.mgr->aicEnd_);
}

TEST(AicoreManagerTest, ProcessParallellDevTasksFinish_WaitNext)
{
    MgrTestEnv env;
    auto dynTask = std::make_unique<DynDeviceTaskBase>();
    DeviceTaskCtrl taskCtrl{};
    taskCtrl.devTask = &dynTask->devTask;
    taskCtrl.existNextSameIterTask.store(true);
    taskCtrl.nextSameIterTaskCtrl.store(0);
    taskCtrl.notFree.store(true);
    taskCtrl.freeCnt.store(0);

    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.BindTaskCtrl(&taskCtrl);
    env.mgr->ProcessParallellDevTasksFinish(&devTaskCtx);
}
