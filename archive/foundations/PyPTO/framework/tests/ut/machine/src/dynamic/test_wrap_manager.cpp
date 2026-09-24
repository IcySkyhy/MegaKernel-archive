/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 */

#include <gtest/gtest.h>
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/device/dynamic/wrap_manager.h"
#include "machine/device/dynamic/aicore_manager.h"
#include "machine/utils/machine_ws_intf.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class WrapManagerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        wm_.coreStatusMgr_ = &coreStatusMgr_;
        wm_.pendingIds_ = pendingIds_;
        wm_.runningIds_ = runningIds_;
        wm_.aicStart_ = 0;
        wm_.aicEnd_ = 2;
        wm_.aicValidNum_ = 2;
        wm_.readyWrapCoreFunctionQue_ = &wrapInfoQueue_;
        wm_.archInfo = ArchInfo::DAV_3510;
        wm_.curDevTask_ = &devTask_;

        for (int i = 0; i < 10; i++) {
            pendingIds_[i] = AICORE_TASK_INIT;
            runningIds_[i] = AICORE_TASK_INIT;
        }

        wrapInfoQueue_.head = 0;
        wrapInfoQueue_.tail = 0;
        wrapInfoQueue_.capacity = 10;
        wrapInfoQueue_.elem = wrapInfoElems_;
        wrapInfoQueue_.lock = 0;
    }

    WrapManager wm_;
    CoreStatusManager coreStatusMgr_;
    uint32_t pendingIds_[10];
    uint32_t runningIds_[10];
    WrapInfoQueue wrapInfoQueue_;
    WrapInfo wrapInfoElems_[10];
    DeviceTask devTask_;
};

TEST_F(WrapManagerTest, IsMixArch_Dav3510_ReturnsTrue)
{
    wm_.archInfo = ArchInfo::DAV_3510;
    EXPECT_TRUE(wm_.IsMixArch());
}

TEST_F(WrapManagerTest, IsMixArch_Dav2201_ReturnsFalse)
{
    wm_.archInfo = ArchInfo::DAV_2201;
    EXPECT_FALSE(wm_.IsMixArch());
}

TEST_F(WrapManagerTest, GetWrapAicCoreIdx_AicCore_ReturnsSameIdx)
{
    uint16_t result = wm_.GetWrapAicCoreIdx(0);
    EXPECT_EQ(result, 0);

    result = wm_.GetWrapAicCoreIdx(1);
    EXPECT_EQ(result, 1);
}

TEST_F(WrapManagerTest, GetWrapAicCoreIdx_AivCore_ReturnsAicIdx)
{
    uint16_t result = wm_.GetWrapAicCoreIdx(2);
    EXPECT_EQ(result, 0);

    result = wm_.GetWrapAicCoreIdx(4);
    EXPECT_EQ(result, 1);
}

TEST_F(WrapManagerTest, GetWrapAiv0CoreIdx_ReturnsCorrectIdx)
{
    uint16_t result = wm_.GetWrapAiv0CoreIdx(0);
    EXPECT_EQ(result, 2);

    result = wm_.GetWrapAiv0CoreIdx(1);
    EXPECT_EQ(result, 4);
}

TEST_F(WrapManagerTest, GetWrapAiv1CoreIdx_ReturnsNextIdx)
{
    uint16_t result = wm_.GetWrapAiv1CoreIdx(2);
    EXPECT_EQ(result, 3);

    result = wm_.GetWrapAiv1CoreIdx(4);
    EXPECT_EQ(result, 5);
}

TEST_F(WrapManagerTest, GetWrapAicoreIdx_AicCore_ReturnsWrapIdxAic)
{
    int32_t result = WrapManager::GetWrapAicoreIdx(static_cast<uint32_t>(CoreType::AIC), 0);
    EXPECT_EQ(result, WRAP_IDX_AIC);
}

TEST_F(WrapManagerTest, GetWrapAicoreIdx_AivCoreWrapVec0_ReturnsWrapIdxAiv0)
{
    int32_t result = WrapManager::GetWrapAicoreIdx(static_cast<uint32_t>(CoreType::AIV), 0);
    EXPECT_EQ(result, WRAP_IDX_AIV0);
}

TEST_F(WrapManagerTest, GetWrapAicoreIdx_AivCoreWrapVec1_ReturnsWrapIdxAiv1)
{
    int32_t result = WrapManager::GetWrapAicoreIdx(static_cast<uint32_t>(CoreType::AIV), 1);
    EXPECT_EQ(result, WRAP_IDX_AIV1);
}

TEST_F(WrapManagerTest, GetAvailableWrapCoreIdx_NoReadyCores_ReturnsFalse)
{
    uint32_t coreIdx = 0;
    uint32_t v0Idx = 0;
    bool result = wm_.GetAvailableWrapCoreIdx(static_cast<uint8_t>(MixResourceType::MIX_1C1V), 0, coreIdx, v0Idx);
    EXPECT_FALSE(result);
}

TEST_F(WrapManagerTest, GetAvailableWrapCoreIdx_WithReadyCores1C1V_ReturnsTrue)
{
    coreStatusMgr_.AddRunReadyCoreIdx(0, static_cast<int>(CoreType::AIC));
    coreStatusMgr_.AddRunReadyCoreIdx(2, static_cast<int>(CoreType::AIV));

    uint32_t coreIdx = 0;
    uint32_t v0Idx = 0;
    bool result = wm_.GetAvailableWrapCoreIdx(static_cast<uint8_t>(MixResourceType::MIX_1C1V), 1, coreIdx, v0Idx);
    EXPECT_TRUE(result);
    EXPECT_EQ(coreIdx, 0);
    EXPECT_EQ(v0Idx, 2);
}

TEST_F(WrapManagerTest, GetAvailableWrapCoreIdx_WithReadyCores1C2V_ReturnsTrue)
{
    coreStatusMgr_.AddRunReadyCoreIdx(0, static_cast<int>(CoreType::AIC));
    coreStatusMgr_.AddRunReadyCoreIdx(2, static_cast<int>(CoreType::AIV));
    coreStatusMgr_.AddRunReadyCoreIdx(3, static_cast<int>(CoreType::AIV));

    uint32_t coreIdx = 0;
    uint32_t v0Idx = 0;
    bool result = wm_.GetAvailableWrapCoreIdx(static_cast<uint8_t>(MixResourceType::MIX_1C2V), 1, coreIdx, v0Idx);
    EXPECT_TRUE(result);
    EXPECT_EQ(coreIdx, 0);
    EXPECT_EQ(v0Idx, 2);
}

TEST_F(WrapManagerTest, GetWrapCoreRunningCnt_NoReadyCores_ReturnsZero)
{
    WrapManager::WrapCoreCandidates candidates;
    uint32_t core1c1vCnt = 0;
    uint32_t core1c2vCnt = 0;
    uint32_t result = wm_.GetWrapCoreRunningCnt(candidates, core1c1vCnt, core1c2vCnt);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(core1c1vCnt, 0);
    EXPECT_EQ(core1c2vCnt, 0);
}

TEST_F(WrapManagerTest, GetWrapCoreRunningCnt_WithReadyCores1C1V_ReturnsCount)
{
    coreStatusMgr_.AddRunReadyCoreIdx(0, static_cast<int>(CoreType::AIC));
    coreStatusMgr_.AddRunReadyCoreIdx(2, static_cast<int>(CoreType::AIV));

    WrapManager::WrapCoreCandidates candidates;
    uint32_t core1c1vCnt = 0;
    uint32_t core1c2vCnt = 0;
    uint32_t result = wm_.GetWrapCoreRunningCnt(candidates, core1c1vCnt, core1c2vCnt);
    EXPECT_EQ(result, 1);
    EXPECT_EQ(core1c1vCnt, 1);
    EXPECT_EQ(core1c2vCnt, 0);
}

TEST_F(WrapManagerTest, GetWrapCoreRunningCnt_WithReadyCores1C2V_ReturnsCount)
{
    coreStatusMgr_.AddRunReadyCoreIdx(0, static_cast<int>(CoreType::AIC));
    coreStatusMgr_.AddRunReadyCoreIdx(2, static_cast<int>(CoreType::AIV));
    coreStatusMgr_.AddRunReadyCoreIdx(3, static_cast<int>(CoreType::AIV));

    WrapManager::WrapCoreCandidates candidates;
    uint32_t core1c1vCnt = 0;
    uint32_t core1c2vCnt = 0;
    uint32_t result = wm_.GetWrapCoreRunningCnt(candidates, core1c1vCnt, core1c2vCnt);
    EXPECT_EQ(result, 1);
    EXPECT_EQ(core1c1vCnt, 0);
    EXPECT_EQ(core1c2vCnt, 1);
}

TEST_F(WrapManagerTest, GetWrapCorePendingCnt_AllPending_ReturnsCount)
{
    WrapManager::WrapCoreCandidates candidates;
    uint32_t core1c1vCnt = 0;
    uint32_t core1c2vCnt = 0;
    uint32_t result = wm_.GetWrapCorePendingCnt(candidates, core1c1vCnt, core1c2vCnt);
    EXPECT_EQ(result, 2);
}

TEST_F(WrapManagerTest, GetWrapId_NoWrapList_ReturnsMinusOne)
{
    int32_t opWrapList[4] = {-1, -1, -1, -1};
    devTask_.mixTaskData.opWrapList[0] = reinterpret_cast<uint64_t>(opWrapList);
    int32_t result = wm_.GetWrapId(MakeTaskID(0, 0));
    EXPECT_EQ(result, -1);
}

TEST_F(WrapManagerTest, GetWrapId_WithWrapList_ReturnsWrapId)
{
    int32_t opWrapList[4] = {5, -1, -1, -1};
    devTask_.mixTaskData.opWrapList[0] = reinterpret_cast<uint64_t>(opWrapList);

    int32_t result = wm_.GetWrapId(MakeTaskID(0, 0));
    EXPECT_EQ(result, MakeMixWrapID(0, 5));
}

TEST_F(WrapManagerTest, PushMixToReadyQueue_AddsToQueue)
{
    uint32_t taskIds[MAX_WRAP_TASK_NUM] = {0x100, 0x200, 0x300};
    wm_.PushMixToReadyQueue(taskIds, static_cast<uint8_t>(MixResourceType::MIX_1C2V), 1);

    EXPECT_EQ(wrapInfoQueue_.tail, 1);
    EXPECT_EQ(wrapInfoElems_[0].tasklist[WRAP_IDX_AIC], 0x100);
    EXPECT_EQ(wrapInfoElems_[0].tasklist[WRAP_IDX_AIV0], 0x200);
    EXPECT_EQ(wrapInfoElems_[0].tasklist[WRAP_IDX_AIV1], 0x300);
    EXPECT_EQ(wrapInfoElems_[0].mixResourceType, static_cast<uint8_t>(MixResourceType::MIX_1C2V));
    EXPECT_EQ(wrapInfoElems_[0].wrapId, 1);
}

TEST_F(WrapManagerTest, ResolveDepForOneMix_CoreAvailable_SendsTasks)
{
    coreStatusMgr_.AddRunReadyCoreIdx(0, static_cast<int>(CoreType::AIC));
    coreStatusMgr_.AddRunReadyCoreIdx(2, static_cast<int>(CoreType::AIV));
    coreStatusMgr_.AddRunReadyCoreIdx(3, static_cast<int>(CoreType::AIV));

    int sendCount = 0;
    wm_.SendTaskToAiCore = [&sendCount](SchDeviceTaskContext*, CoreType, int, uint64_t) { sendCount++; };

    uint32_t taskIds[MAX_WRAP_TASK_NUM] = {0x100, 0x200, 0x300};
    wm_.ResolveDepForOneMix(taskIds, static_cast<uint8_t>(MixResourceType::MIX_1C2V), 0);

    EXPECT_EQ(sendCount, 3);
}

TEST_F(WrapManagerTest, ResolveDepForOneMix_CoreNotAvailable_PushesToQueue)
{
    int sendCount = 0;
    wm_.SendTaskToAiCore = [&sendCount](SchDeviceTaskContext*, CoreType, int, uint64_t) { sendCount++; };

    int32_t opWrapList[4] = {5, -1, -1, -1};
    devTask_.mixTaskData.opWrapList[0] = reinterpret_cast<uint64_t>(opWrapList);

    uint32_t aicIdx = wm_.GetWrapAicCoreIdx(0);
    uint32_t aivIdx0 = wm_.GetWrapAiv0CoreIdx(aicIdx);
    uint32_t aivIdx1 = wm_.GetWrapAiv1CoreIdx(aivIdx0);
    pendingIds_[aicIdx] = 0x999;
    pendingIds_[aivIdx0] = 0x999;
    pendingIds_[aivIdx1] = 0x999;

    uint32_t taskIds[MAX_WRAP_TASK_NUM] = {0x100, 0x200, 0x300};
    wm_.ResolveDepForOneMix(taskIds, static_cast<uint8_t>(MixResourceType::MIX_1C2V), 0);

    EXPECT_EQ(sendCount, 0);
    EXPECT_EQ(wrapInfoQueue_.tail, 1);
}

TEST_F(WrapManagerTest, InitDieMaxCpuId_EvenCpuNum_SetsCorrectValues)
{
    wm_.InitDieMaxCpuId(4);
    EXPECT_EQ(wm_.curDie0MaxCpuId_, 2);
    EXPECT_EQ(wm_.curDie1StartCpuId_, 2);
}

TEST_F(WrapManagerTest, InitDieMaxCpuId_OddCpuNum_SetsCorrectValues)
{
    wm_.InitDieMaxCpuId(5);
    EXPECT_EQ(wm_.curDie0MaxCpuId_, 2);
    EXPECT_EQ(wm_.curDie1StartCpuId_, 3);
}

TEST_F(WrapManagerTest, InitDieId_LessThanDie0Max_SetsDie0)
{
    wm_.curDie0MaxCpuId_ = 2;
    wm_.curDie1StartCpuId_ = 2;
    wm_.InitDieId(0);
    EXPECT_EQ(wm_.dieId_, DieId::DIE_0);
}

TEST_F(WrapManagerTest, InitDieId_GreaterThanOrEqualDie1Start_SetsDie1)
{
    wm_.curDie0MaxCpuId_ = 2;
    wm_.curDie1StartCpuId_ = 2;
    wm_.InitDieId(2);
    EXPECT_EQ(wm_.dieId_, DieId::DIE_1);
}

TEST_F(WrapManagerTest, InitDieId_BetweenDie0AndDie1_SetsDieMix)
{
    wm_.curDie0MaxCpuId_ = 2;
    wm_.curDie1StartCpuId_ = 3;
    wm_.InitDieId(2);
    EXPECT_EQ(wm_.dieId_, DieId::DIE_MIX);
}

TEST_F(WrapManagerTest, GetDieSchedIdRange_Die0_ReturnsCorrectRange)
{
    wm_.dieId_ = DieId::DIE_0;
    wm_.curDie0MaxCpuId_ = 2;

    int schedStart = 0;
    int schedEnd = 0;
    wm_.GetDieSchedIdRange(schedStart, schedEnd, 4);

    EXPECT_EQ(schedStart, 0);
    EXPECT_EQ(schedEnd, 2);
}

TEST_F(WrapManagerTest, GetDieSchedIdRange_Die1_ReturnsCorrectRange)
{
    wm_.dieId_ = DieId::DIE_1;
    wm_.curDie1StartCpuId_ = 2;

    int schedStart = 0;
    int schedEnd = 0;
    wm_.GetDieSchedIdRange(schedStart, schedEnd, 4);

    EXPECT_EQ(schedStart, 2);
    EXPECT_EQ(schedEnd, 4);
}

TEST_F(WrapManagerTest, GetDieId_ReturnsCurrentDieId)
{
    wm_.dieId_ = DieId::DIE_0;
    EXPECT_EQ(wm_.GetDieId(), DieId::DIE_0);

    wm_.dieId_ = DieId::DIE_1;
    EXPECT_EQ(wm_.GetDieId(), DieId::DIE_1);

    wm_.dieId_ = DieId::DIE_MIX;
    EXPECT_EQ(wm_.GetDieId(), DieId::DIE_MIX);
}

TEST_F(WrapManagerTest, SetDieReadyQueue_SetsQueuesCorrectly)
{
    DieReadyQueueData dieData;
    ReadyCoreFunctionQueue aicQueue0, aicQueue1, aivQueue0, aivQueue1;
    dieData.readyDieAicCoreFunctionQue[0] = reinterpret_cast<uint64_t>(&aicQueue0);
    dieData.readyDieAicCoreFunctionQue[1] = reinterpret_cast<uint64_t>(&aicQueue1);
    dieData.readyDieAivCoreFunctionQue[0] = reinterpret_cast<uint64_t>(&aivQueue0);
    dieData.readyDieAivCoreFunctionQue[1] = reinterpret_cast<uint64_t>(&aivQueue1);

    wm_.SetDieReadyQueue(dieData);

    EXPECT_EQ(wm_.readyDieAicFunctionQue_[0], &aicQueue0);
    EXPECT_EQ(wm_.readyDieAicFunctionQue_[1], &aicQueue1);
    EXPECT_EQ(wm_.readyDieAivFunctionQue_[0], &aivQueue0);
    EXPECT_EQ(wm_.readyDieAivFunctionQue_[1], &aivQueue1);
}

TEST_F(WrapManagerTest, GetDieReadyQueue_NotMixArch_ReturnsDefault)
{
    wm_.archInfo = ArchInfo::DAV_2201;
    ReadyCoreFunctionQueue defaultQueue;

    ReadyCoreFunctionQueue* result = wm_.GetDieReadyQueue(CoreType::AIC, &defaultQueue);
    EXPECT_EQ(result, &defaultQueue);
}

TEST_F(WrapManagerTest, GetDieReadyQueue_DieMix_ReturnsDefault)
{
    wm_.archInfo = ArchInfo::DAV_3510;
    wm_.dieId_ = DieId::DIE_MIX;
    ReadyCoreFunctionQueue defaultQueue;

    ReadyCoreFunctionQueue* result = wm_.GetDieReadyQueue(CoreType::AIC, &defaultQueue);
    EXPECT_EQ(result, &defaultQueue);
}

TEST_F(WrapManagerTest, GetDieReadyQueue_Die0_ReturnsDie0Queue)
{
    wm_.archInfo = ArchInfo::DAV_3510;
    wm_.dieId_ = DieId::DIE_0;

    DieReadyQueueData dieData{};
    ReadyCoreFunctionQueue aicQueue0, aivQueue0;
    dieData.readyDieAicCoreFunctionQue[0] = reinterpret_cast<uint64_t>(&aicQueue0);
    dieData.readyDieAivCoreFunctionQue[0] = reinterpret_cast<uint64_t>(&aivQueue0);
    wm_.SetDieReadyQueue(dieData);

    ReadyCoreFunctionQueue defaultQueue;
    ReadyCoreFunctionQueue* result = wm_.GetDieReadyQueue(CoreType::AIC, &defaultQueue);
    EXPECT_EQ(result, &aicQueue0);
}

TEST(IsWrapTaskReadyTest, Mix1C1V_AllReady_ReturnsTrue)
{
    npu::tile_fwk::WrapInfo wrapInfo;
    wrapInfo.mixResourceType = static_cast<uint8_t>(MixResourceType::MIX_1C1V);
    wrapInfo.tasklist[WRAP_IDX_AIC] = 0x100;
    wrapInfo.tasklist[WRAP_IDX_AIV0] = 0x200;

    EXPECT_TRUE(IsWrapTaskReady(&wrapInfo));
}

TEST(IsWrapTaskReadyTest, Mix1C1V_NotAllReady_ReturnsFalse)
{
    npu::tile_fwk::WrapInfo wrapInfo;
    wrapInfo.mixResourceType = static_cast<uint8_t>(MixResourceType::MIX_1C1V);
    wrapInfo.tasklist[WRAP_IDX_AIC] = 0x100;
    wrapInfo.tasklist[WRAP_IDX_AIV0] = AICORE_TASK_INIT;

    EXPECT_FALSE(IsWrapTaskReady(&wrapInfo));
}

TEST(IsWrapTaskReadyTest, Mix1C2V_AllReady_ReturnsTrue)
{
    npu::tile_fwk::WrapInfo wrapInfo;
    wrapInfo.mixResourceType = static_cast<uint8_t>(MixResourceType::MIX_1C2V);
    wrapInfo.tasklist[WRAP_IDX_AIC] = 0x100;
    wrapInfo.tasklist[WRAP_IDX_AIV0] = 0x200;
    wrapInfo.tasklist[WRAP_IDX_AIV1] = 0x300;

    EXPECT_TRUE(IsWrapTaskReady(&wrapInfo));
}

TEST(IsWrapTaskReadyTest, Mix1C2V_NotAllReady_ReturnsFalse)
{
    npu::tile_fwk::WrapInfo wrapInfo;
    wrapInfo.mixResourceType = static_cast<uint8_t>(MixResourceType::MIX_1C2V);
    wrapInfo.tasklist[WRAP_IDX_AIC] = 0x100;
    wrapInfo.tasklist[WRAP_IDX_AIV0] = 0x200;
    wrapInfo.tasklist[WRAP_IDX_AIV1] = AICORE_TASK_INIT;

    EXPECT_FALSE(IsWrapTaskReady(&wrapInfo));
}

TEST(GetTaskNumByMixResTypeTest, Mix1C1V_Returns2)
{
    EXPECT_EQ(GetTaskNumByMixResType(static_cast<uint8_t>(MixResourceType::MIX_1C1V)), 2);
}

TEST(GetTaskNumByMixResTypeTest, Mix1C2V_Returns3)
{
    EXPECT_EQ(GetTaskNumByMixResType(static_cast<uint8_t>(MixResourceType::MIX_1C2V)), 3);
}

TEST(GetTaskNumByMixResTypeTest, Unknown_Returns0)
{
    EXPECT_EQ(GetTaskNumByMixResType(static_cast<uint8_t>(MixResourceType::MIX_UNKNOWN)), 0);
}
