/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_instrumentation.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "machine/device/dynamic/aicpu_instrumentation.h"
#include "machine/utils/device_log.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class InstrumentationTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

HardBranchGroupDefine(oneCase);

static int GetData(int n)
{
    if (HardBranchTrue(oneCase)) {
        return n + 1;
    } else {
        return n - 1;
    }
}

TEST_F(InstrumentationTest, HardBranch)
{
    HardBranchManager manager;
    manager.AddGroup(HardBranchGroupCreate(oneCase));
    EXPECT_EQ(1, manager.Size());

    static int data = 0x10;
    int v0 = GetData(data);
    EXPECT_EQ(0x11, v0);
    manager.SwitchToJump();
    int v1 = GetData(data);
    EXPECT_EQ(0x0f, v1);
    manager.SwitchToNop();
    int v2 = GetData(data);
    EXPECT_EQ(0x11, v2);
}

TEST_F(InstrumentationTest, HardBranchManager)
{
    HardBranchManager& manager = HardBranchManager::GetInstance();
    manager.Placeholder();
    manager.AddGroup(HardBranchGroupCreate(oneCase));
    EXPECT_TRUE(manager.Size() == 1);
}
