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
 * \file test_core_status_manager.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/machine/device/tilefwk/core_func_data.h"
#include "machine/device/dynamic/core_status_manager.h"

using namespace npu::tile_fwk::dynamic;

class TestCoreStatusManager : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestCoreStatusManager, Constructor_InitializesCorrectly)
{
    CoreStatusManager manager;

    EXPECT_EQ(manager.waitTaskCnt[0], 0u);
    EXPECT_EQ(manager.waitTaskCnt[1], 0u);
    EXPECT_EQ(manager.corePendReadyCnt_[0], 0u);
    EXPECT_EQ(manager.corePendReadyCnt_[1], 0u);
    EXPECT_EQ(manager.coreRunReadyCnt_[0], 0u);
    EXPECT_EQ(manager.coreRunReadyCnt_[1], 0u);
    EXPECT_EQ(manager.lastPendReadyCoreIdx_[0], 0u);
    EXPECT_EQ(manager.lastPendReadyCoreIdx_[1], 0u);

    for (uint32_t i = 0; i < MAX_AICORE_NUM; i++) {
        EXPECT_EQ(manager.coreIdxPosition_[i], INVALID_COREIDX_POSITION);
    }
}

TEST_F(TestCoreStatusManager, AddRunReadyCoreIdx)
{
    CoreStatusManager manager;
    int coreIdx = 0;
    int type = 0;

    manager.AddRunReadyCoreIdx(coreIdx, type);

    EXPECT_EQ(manager.coreRunReadyCnt_[type], 1u);
    EXPECT_EQ(manager.runReadyCoreIdx_[type][0], coreIdx);
    EXPECT_EQ(manager.coreIdxPosition_[coreIdx], 0u);
}

TEST_F(TestCoreStatusManager, AddPendReadyCoreIdx)
{
    CoreStatusManager manager;
    int type = 0;

    manager.AddPendReadyCoreIdx(type);

    EXPECT_EQ(manager.corePendReadyCnt_[type], 1u);
}

TEST_F(TestCoreStatusManager, AddRunAndPendCoreIdx)
{
    CoreStatusManager manager;
    int coreIdx = 5;
    int type = 1;

    manager.AddRunAndPendCoreIdx(coreIdx, type);

    EXPECT_EQ(manager.coreRunReadyCnt_[type], 1u);
    EXPECT_EQ(manager.runReadyCoreIdx_[type][0], coreIdx);
    EXPECT_EQ(manager.coreIdxPosition_[coreIdx], 0u);
    EXPECT_EQ(manager.corePendReadyCnt_[type], 1u);
}

TEST_F(TestCoreStatusManager, RemoveRunReadyCoreIdx)
{
    CoreStatusManager manager;
    int coreIdx = 3;
    int type = 0;

    manager.AddRunReadyCoreIdx(coreIdx, type);
    EXPECT_EQ(manager.coreRunReadyCnt_[type], 1u);

    manager.RemoveRunReadyCoreIdx(coreIdx, type);
    EXPECT_EQ(manager.coreRunReadyCnt_[type], 0u);
    EXPECT_EQ(manager.coreIdxPosition_[coreIdx], INVALID_COREIDX_POSITION);
}

TEST_F(TestCoreStatusManager, RemoveRunReadyCoreIdx_NotInList)
{
    CoreStatusManager manager;
    int coreIdx = 10;
    int type = 0;

    manager.RemoveRunReadyCoreIdx(coreIdx, type);
    EXPECT_EQ(manager.coreRunReadyCnt_[type], 0u);
}

TEST_F(TestCoreStatusManager, RemovePendReadyCoreIdx)
{
    CoreStatusManager manager;
    int type = 1;

    manager.AddPendReadyCoreIdx(type);
    EXPECT_EQ(manager.corePendReadyCnt_[type], 1u);

    manager.RemovePendReadyCoreIdx(type);
    EXPECT_EQ(manager.corePendReadyCnt_[type], 0u);
}

TEST_F(TestCoreStatusManager, RemoveRunAndPendCoreIdx)
{
    CoreStatusManager manager;
    int coreIdx = 7;
    int type = 0;

    manager.AddRunAndPendCoreIdx(coreIdx, type);
    EXPECT_EQ(manager.coreRunReadyCnt_[type], 1u);
    EXPECT_EQ(manager.corePendReadyCnt_[type], 1u);

    manager.RemoveRunAndPendCoreIdx(coreIdx, type);
    EXPECT_EQ(manager.coreRunReadyCnt_[type], 0u);
    EXPECT_EQ(manager.corePendReadyCnt_[type], 0u);
    EXPECT_EQ(manager.coreIdxPosition_[coreIdx], INVALID_COREIDX_POSITION);
}

TEST_F(TestCoreStatusManager, BatchRemovePendReadyCoreIdx)
{
    CoreStatusManager manager;
    int type = 0;

    manager.AddPendReadyCoreIdx(type);
    manager.AddPendReadyCoreIdx(type);
    manager.AddPendReadyCoreIdx(type);
    EXPECT_EQ(manager.corePendReadyCnt_[type], 3u);

    manager.BatchRemovePendReadyCoreIdx(type, 2);
    EXPECT_EQ(manager.corePendReadyCnt_[type], 1u);
}

TEST_F(TestCoreStatusManager, RemoveReadyCoreIdxTail)
{
    CoreStatusManager manager;
    int coreIdx = 4;
    int type = 1;

    manager.AddRunReadyCoreIdx(coreIdx, type);
    EXPECT_EQ(manager.coreRunReadyCnt_[type], 1u);

    manager.RemoveReadyCoreIdxTail(coreIdx, type);
    EXPECT_EQ(manager.coreRunReadyCnt_[type], 0u);
    EXPECT_EQ(manager.coreIdxPosition_[coreIdx], INVALID_COREIDX_POSITION);
}

TEST_F(TestCoreStatusManager, RemoveReadyCoreIdxTail_NotInList)
{
    CoreStatusManager manager;
    int coreIdx = 15;
    int type = 0;

    manager.RemoveReadyCoreIdxTail(coreIdx, type);
    EXPECT_EQ(manager.coreRunReadyCnt_[type], 0u);
}

TEST_F(TestCoreStatusManager, MultipleOperations)
{
    CoreStatusManager manager;

    for (int i = 0; i < 5; i++) {
        manager.AddRunAndPendCoreIdx(i, 0);
    }
    EXPECT_EQ(manager.coreRunReadyCnt_[0], 5u);
    EXPECT_EQ(manager.corePendReadyCnt_[0], 5u);

    manager.RemoveRunAndPendCoreIdx(2, 0);
    EXPECT_EQ(manager.coreRunReadyCnt_[0], 4u);
    EXPECT_EQ(manager.corePendReadyCnt_[0], 4u);
}
