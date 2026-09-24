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
 * \file test_dev_encode_program_ctrlflow_cache.cpp
 * \brief Unit tests for the aicore-resolve (drco) branches in DevControlFlowCache.
 */
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <memory>
#include <vector>

#define private public
#define protected public
#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"
#include "machine/utils/queues.h"
#include "interface/machine/device/tilefwk/aikernel_device_task.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {

constexpr size_t kCtrlCacheSize = 512 * 1024;

// A host-side DrcoRootFuncList that mimics what InitDrcoRootFuncList builds:
// all per-core/local queues are backed by real memory, except the last entry of
// each array which stays null to exercise the null-continue branches.
struct DrcoQueueFixture {
    DrcoRootFuncList root{};
    std::vector<uint8_t> perCoreStorage;
    std::vector<uint8_t> localStorage;

    void Build(uint32_t coreFunctionCnt = 8)
    {
        const size_t perCoreBytes = sizeof(PerCorePendingQueue) + 16 * sizeof(LeafTaskId);
        const size_t localBytes = sizeof(DrcoLocalReadyQueue);
        constexpr size_t localAlign = alignof(DrcoLocalReadyQueue);
        perCoreStorage.assign(MAX_AICORE_NUM_FOR_QUEUE * perCoreBytes, 0);
        localStorage.assign(NUM_CORE_TYPES * NUM_LOCAL_GROUPS * localBytes + localAlign, 0);
        uintptr_t localBase = reinterpret_cast<uintptr_t>(localStorage.data());
        uint8_t* localAlignedBase = reinterpret_cast<uint8_t*>((localBase + localAlign - 1) &
                                                               ~static_cast<uintptr_t>(localAlign - 1));
        for (uint32_t i = 0; i < MAX_AICORE_NUM_FOR_QUEUE; ++i) {
            if (i == MAX_AICORE_NUM_FOR_QUEUE - 1) {
                root.perCorePendingQueueArray[i] = nullptr;
                continue;
            }
            root.perCorePendingQueueArray[i] = reinterpret_cast<PerCorePendingQueue*>(perCoreStorage.data() +
                                                                                      i * perCoreBytes);
        }
        for (uint32_t ct = 0; ct < NUM_CORE_TYPES; ++ct) {
            for (uint32_t i = 0; i < NUM_LOCAL_GROUPS; ++i) {
                if (ct == NUM_CORE_TYPES - 1 && i == NUM_LOCAL_GROUPS - 1) {
                    root.localReadyQueueArray[ct][i] = nullptr;
                    continue;
                }
                root.localReadyQueueArray[ct][i] = reinterpret_cast<DrcoLocalReadyQueue*>(
                    localAlignedBase + (ct * NUM_LOCAL_GROUPS + i) * localBytes);
            }
        }
        root.totalTaskCount = coreFunctionCnt;
        root.executedTaskCount = 0;
        root.devTaskFinished = 0;
    }
};

// Sets up the three ready queues (AIV/AIC/AICPU) on a DynDeviceTask.
void SetupReadyQueues(DynDeviceTask* dyntask, std::array<std::array<uint32_t, 16>, READY_QUEUE_SIZE>& elemBuf,
                      std::array<std::unique_ptr<ReadyCoreFunctionQueue>, READY_QUEUE_SIZE>& queue)
{
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        queue[i] = std::make_unique<ReadyCoreFunctionQueue>(16, elemBuf[i].data());
        dyntask->readyQueue[i] = queue[i].get();
    }
}

// Builds a DynFuncHeader with funcNum entries followed by the DynFuncData array.
void SetupDynFuncHeader(DynDeviceTask* dyntask,
                        std::array<uint8_t, sizeof(DynFuncHeader) + 8 * sizeof(DynFuncData)>& hdrBuf,
                        uint32_t funcNum = 1)
{
    auto* header = reinterpret_cast<DynFuncHeader*>(hdrBuf.data());
    header->seqNo = 0;
    header->funcNum = funcNum;
    header->funcSize = static_cast<uint32_t>(sizeof(DynFuncHeader) + funcNum * sizeof(DynFuncData));
    dyntask->dynFuncDataList = header;
}

// Builds a minimal DevAscendFunctionDuppedData with one leaf op and no stitches.
DevAscendFunctionDuppedData* SetupDuppedData(std::array<uint8_t, 1024>& dupBuf, uint32_t opCount = 1)
{
    auto* duppedData = reinterpret_cast<DevAscendFunctionDuppedData*>(dupBuf.data());
    duppedData->source_ = nullptr;
    duppedData->operationList_.size = opCount;
    duppedData->operationList_.predCountBase = static_cast<uint32_t>(sizeof(DevAscendFunctionDuppedData));
    duppedData->operationList_.stitchCount = 0;
    return duppedData;
}

void SetupCtrlCache(DevControlFlowCache& ctrl, std::vector<uint8_t>& cacheBuf, size_t cacheSize = kCtrlCacheSize)
{
    cacheBuf.assign(cacheSize, 0);
    ctrl.cacheData = DevRelocVector<uint8_t>(static_cast<int>(cacheSize), cacheBuf.data());
    ctrl.cacheDataOffset = 0;
}

} // namespace

TEST(CtrlFlowCacheDrcoUt, PredCountDataRestore_CoversDrcoPredCount)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);

    std::array<uint8_t, sizeof(DynFuncHeader) + 8 * sizeof(DynFuncData)> hdrBuf{};
    SetupDynFuncHeader(dyntask.get(), hdrBuf, 1);

    std::array<uint8_t, 1024> dupBuf{};
    DynFuncDataCache& cache = dyntask->dynFuncDataCacheList[0];
    cache.duppedData = SetupDuppedData(dupBuf, 1);
    cache.predCount = nullptr;
    cache.calleeList = nullptr;
    cache.devFunc = nullptr;

    dyntask->dynFuncDataBackupList[0] = DynFuncDataBackup{};
    std::array<uint8_t, 64> predCountBackup{};
    dyntask->dynFuncDataBackupList[0].predCountBackup = reinterpret_cast<predcount_t*>(predCountBackup.data());

    std::array<uint8_t, 64> predCountDest{};
    dyntask->dynFuncDataList->At(0).drcoRootFuncData.predCount = reinterpret_cast<int32_t*>(predCountDest.data());

    std::vector<uint8_t> cacheBuf;
    DevControlFlowCache ctrl;
    SetupCtrlCache(ctrl, cacheBuf);

    ctrl.PredCountDataRestore(dyntask.get());
    SUCCEED();
}

TEST(CtrlFlowCacheDrcoUt, ReadyQueueDataBackupRestore_WithDrco)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 8;

    DrcoQueueFixture drco;
    drco.Build();
    // mark one per-core queue non-empty so the backup memcpy path uses real size
    drco.root.perCorePendingQueueArray[3]->size = 1;
    drco.root.perCorePendingQueueArray[3]->taskList[0] = MakeTaskID(2, 0);
    dyntask->drcoRootFuncList = &drco.root;

    std::array<std::array<uint32_t, 16>, READY_QUEUE_SIZE> elemBuf{};
    std::array<std::unique_ptr<ReadyCoreFunctionQueue>, READY_QUEUE_SIZE> queue{};
    SetupReadyQueues(dyntask.get(), elemBuf, queue);

    const int aivIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV);
    const int aicIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC);
    queue[aivIdx]->UnsafeEnqueue(MakeTaskID(0, 1));
    queue[aivIdx]->UnsafeEnqueue(MakeTaskID(0, 2));
    queue[aicIdx]->UnsafeEnqueue(MakeTaskID(1, 1));

    std::vector<uint8_t> cacheBuf;
    DevControlFlowCache ctrl;
    SetupCtrlCache(ctrl, cacheBuf);

    ctrl.ReadyQueueDataBackup(dyntask.get());
    ASSERT_NE(dyntask->readyQueueBackup, nullptr);

    ctrl.ReadyQueueDataRestore(dyntask.get(), 4);
    SUCCEED();

    EXPECT_EQ(dyntask->devTask.coreFunctionCnt, 8U);
    EXPECT_EQ(drco.root.perCorePendingQueueArray[0]->size, 1U);
    uint32_t aivRouted = 0;
    for (uint32_t i = 4; i < 4 + 8; ++i) {
        aivRouted += drco.root.perCorePendingQueueArray[i]->size;
    }
    EXPECT_EQ(aivRouted, 2U);
    EXPECT_EQ(drco.root.executedTaskCount, 0U);
}

TEST(CtrlFlowCacheDrcoUt, DieReadyQueueDataBackupRestore_WithDrco)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 8;

    DrcoQueueFixture drco;
    drco.Build();
    dyntask->drcoRootFuncList = &drco.root;

    std::array<std::array<uint32_t, 16>, DIE_NUM> bufAiv{};
    std::array<std::array<uint32_t, 16>, DIE_NUM> bufAic{};
    std::array<std::unique_ptr<ReadyCoreFunctionQueue>, DIE_NUM> qAiv;
    std::array<std::unique_ptr<ReadyCoreFunctionQueue>, DIE_NUM> qAic;
    for (uint32_t i = 0; i < DIE_NUM; ++i) {
        qAiv[i] = std::make_unique<ReadyCoreFunctionQueue>(16, bufAiv[i].data());
        qAic[i] = std::make_unique<ReadyCoreFunctionQueue>(16, bufAic[i].data());
        dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] = reinterpret_cast<uint64_t>(qAiv[i].get());
        dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i] = reinterpret_cast<uint64_t>(qAic[i].get());
        qAiv[i]->UnsafeEnqueue(MakeTaskID(i, 1));
        qAic[i]->UnsafeEnqueue(MakeTaskID(i, 2));
    }

    std::vector<uint8_t> cacheBuf;
    DevControlFlowCache ctrl;
    SetupCtrlCache(ctrl, cacheBuf);

    ctrl.DieReadyQueueDataBackup(dyntask.get());
    ASSERT_NE(dyntask->dieReadyQueueBackup, nullptr);

    ctrl.DieReadyQueueDataRestore(dyntask.get(), 4);
    SUCCEED();

    EXPECT_EQ(drco.root.perCorePendingQueueArray[0]->size, 1U);
    EXPECT_EQ(drco.root.perCorePendingQueueArray[2]->size, 1U);
    EXPECT_EQ(drco.root.perCorePendingQueueArray[4]->size, 1U);
    EXPECT_EQ(drco.root.perCorePendingQueueArray[8]->size, 1U);
}

TEST(CtrlFlowCacheDrcoUt, TaskAddrRelocProgramAndCtrlCache_WithDrco)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 8;

    DrcoQueueFixture drco;
    drco.Build();
    drco.root.globalReadyQueueList[DRCO_QUEUE_AIV].ptr = reinterpret_cast<DrcoGlobalReadyQueue*>(0x8);
    dyntask->drcoRootFuncList = &drco.root;

    std::array<std::array<uint32_t, 16>, READY_QUEUE_SIZE> elemBuf{};
    std::array<std::unique_ptr<ReadyCoreFunctionQueue>, READY_QUEUE_SIZE> queue{};
    SetupReadyQueues(dyntask.get(), elemBuf, queue);

    std::array<uint8_t, sizeof(DynFuncHeader) + 8 * sizeof(DynFuncData)> hdrBuf{};
    SetupDynFuncHeader(dyntask.get(), hdrBuf, 1);

    std::array<uint8_t, 1024> dupBuf{};
    DynFuncDataCache& cache = dyntask->dynFuncDataCacheList[0];
    cache.duppedData = SetupDuppedData(dupBuf, 1);
    cache.predCount = nullptr;
    cache.calleeList = nullptr;
    cache.devFunc = nullptr;
    dyntask->dynFuncDataBackupList[0] = DynFuncDataBackup{};

    std::vector<uint8_t> cacheBuf;
    DevControlFlowCache ctrl;
    SetupCtrlCache(ctrl, cacheBuf);

    ctrl.ReadyQueueDataBackup(dyntask.get());
    ASSERT_NE(dyntask->readyQueueBackup, nullptr);
    // Force the non-null reloc branch for the global ready queue backup.
    dyntask->readyQueueBackup->globalReadyQueueList[0].ptr = reinterpret_cast<DrcoGlobalReadyQueue*>(0x10);

    DeviceTaskCache entry;
    entry.dynTaskBase = dyntask.get();
    ctrl.deviceTaskCacheList = DevRelocVector<DeviceTaskCache>(1, &entry);
    ctrl.deviceTaskCount = 1;

    ctrl.TaskAddrRelocProgramAndCtrlCache(0, 0, 0, 0);
    SUCCEED();
}
