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
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "machine/device/dynamic/aicore_manager.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
struct AicpuMgrTestEnv {
    AicpuTaskManager mgr;
    uint32_t queueElem[256];
    LockableQueueGeneric<uint32_t> readyQueue;
    DynDeviceTaskBase dynTask;

    AicpuMgrTestEnv() : readyQueue(256, queueElem)
    {
        dynTask.devTask.readyAicpuFunctionQue = reinterpret_cast<uint64_t>(&readyQueue);
    }
};
} // namespace

TEST(AicpuTaskManagerTest, BasicLifecycle)
{
    auto mgr = std::make_unique<AicpuTaskManager>();
    EXPECT_TRUE(mgr->Finished(0));
    EXPECT_EQ(static_cast<int>(AicpuTaskManager::SHMEM_WAIT_UNTIL), 0);
    EXPECT_EQ(static_cast<int>(AicpuTaskManager::TASK_TYPE_NUM), 1);

    DeviceArgs args{};
    args.sharedBuffer = 0x1000;
    args.nrAic = 4;
    args.nrAiv = 8;
    args.archInfo = ArchInfo::DAV_3510;
    mgr->InitDeviceArgs(&args);
    EXPECT_EQ(mgr->sharedBuffer_, 0x1000u);
    EXPECT_EQ(mgr->aicNum_, 4u);
    EXPECT_EQ(mgr->aivNum_, 8u);
    EXPECT_EQ(mgr->archInfo_, ArchInfo::DAV_3510);
}

TEST(AicpuTaskManagerTest, TaskEnqueue)
{
    auto envPtr = std::make_unique<AicpuMgrTestEnv>();
    auto& env = *envPtr;
    env.mgr.TaskEnqueue(42, reinterpret_cast<DynDeviceTask*>(&env.dynTask));
    EXPECT_EQ(env.readyQueue.Size(), 1u);
    for (uint32_t i = 0; i < 10; i++) {
        env.mgr.TaskEnqueue(i, reinterpret_cast<DynDeviceTask*>(&env.dynTask));
    }
    EXPECT_EQ(env.readyQueue.Size(), 11u);
}

TEST(AicpuTaskManagerTest, Init_NullCache)
{
    auto envPtr = std::make_unique<AicpuMgrTestEnv>();
    auto& env = *envPtr;
    env.dynTask.shmemWaitUntilCacheBackup = nullptr;
    int32_t ret = env.mgr.Init(reinterpret_cast<DynDeviceTask*>(&env.dynTask), false, 0);
    EXPECT_NE(ret, DEVICE_MACHINE_OK);
}

TEST(AicpuTaskManagerTest, TaskProcess_EmptyQueue)
{
    auto envPtr = std::make_unique<AicpuMgrTestEnv>();
    auto& env = *envPtr;
    uint64_t taskCount = 0;
    int32_t ret = env.mgr.TaskProcess(taskCount, reinterpret_cast<DynDeviceTask*>(&env.dynTask), 0);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(taskCount, 0u);
}

TEST(AicpuTaskManagerTest, Init_WithCache)
{
    auto envPtr = std::make_unique<AicpuMgrTestEnv>();
    auto& env = *envPtr;

    auto cache = std::make_unique<npu::tile_fwk::Distributed::ShmemWaitUntilCache>();
    cache->taskCount = 5;
    env.dynTask.shmemWaitUntilCacheBackup = cache.get();

    int32_t ret = env.mgr.Init(reinterpret_cast<DynDeviceTask*>(&env.dynTask), false, 0);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicpuTaskManagerTest, Init_WithCacheAndProf)
{
    auto envPtr = std::make_unique<AicpuMgrTestEnv>();
    auto& env = *envPtr;

    uint8_t sharedBuf[4096] = {0};
    env.mgr.sharedBuffer_ = reinterpret_cast<uint64_t>(sharedBuf);
    env.mgr.aicNum_ = 2;
    env.mgr.aivNum_ = 0;

    auto cache = std::make_unique<npu::tile_fwk::Distributed::ShmemWaitUntilCache>();
    cache->taskCount = 5;
    env.dynTask.shmemWaitUntilCacheBackup = cache.get();

    int32_t ret = env.mgr.Init(reinterpret_cast<DynDeviceTask*>(&env.dynTask), true, 0);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicpuTaskManagerTest, TaskPoll_EmptyQueue)
{
    auto envPtr = std::make_unique<AicpuMgrTestEnv>();
    auto& env = *envPtr;
    SchThreadStatus status;
    status.Init();
    auto aicoreMng = std::make_unique<AiCoreManager>(status);
    int32_t ret = env.mgr.TaskPoll(aicoreMng.get(), 0);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicpuTaskManagerTest, SyncAicpuTaskFinish_AlreadyFinished)
{
    auto envPtr = std::make_unique<AicpuMgrTestEnv>();
    auto& env = *envPtr;
    SchThreadStatus status;
    status.Init();
    auto aicoreMng = std::make_unique<AiCoreManager>(status);
    int32_t ret = env.mgr.SyncAicpuTaskFinish(aicoreMng.get(), 0);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicpuTaskManagerTest, Finished_Default)
{
    auto mgr = std::make_unique<AicpuTaskManager>();
    EXPECT_TRUE(mgr->Finished(0));
}
