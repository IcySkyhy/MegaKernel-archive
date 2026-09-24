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
 * \file test_item_pool.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "machine/utils/dynamic/item_pool.h"

using namespace npu::tile_fwk::dynamic;
class ItemPoolTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

#define BASE 10000
struct Object {
    static int& GetDestructorCount()
    {
        static int destructorCount = 0;
        return destructorCount;
    }
    int value;
    Object(int n) { value = BASE + n; }
    ~Object() { GetDestructorCount()++; }
};

TEST_F(ItemPoolTest, FreeList)
{
    uint64_t workspaceSize = 0x1000 * 0x100;
    std::vector<uint8_t> workspace(workspaceSize, 0x0);
    SeqWsAllocator aicpuCoherentAllocator;
    aicpuCoherentAllocator.InitMetadataAllocator((uint64_t)workspace.data(), workspace.size());

    {
        ItemPool<Object> pool(aicpuCoherentAllocator, 100);
        auto v1 = pool.Create(20);
        EXPECT_EQ(BASE + 20, v1->value);
        pool.Destroy(v1);
        auto v2 = pool.Create(30);
        EXPECT_EQ(v1, v2);
        EXPECT_EQ(1, Object::GetDestructorCount());
    }
    EXPECT_EQ(2, Object::GetDestructorCount());
}

TEST_F(ItemPoolTest, BumpVirginSlotsWithoutFreelist)
{
    uint64_t workspaceSize = 0x1000 * 0x100;
    std::vector<uint8_t> workspace(workspaceSize, 0x0);
    SeqWsAllocator aicpuCoherentAllocator;
    aicpuCoherentAllocator.InitMetadataAllocator((uint64_t)workspace.data(), workspace.size());

    Object::GetDestructorCount() = 0;
    constexpr size_t poolSize = 64;
    {
        ItemPool<Object> pool(aicpuCoherentAllocator, poolSize);
        EXPECT_EQ(pool.InitReadyCount(), 0u);
        EXPECT_EQ(pool.FreeItemNum(), poolSize);
        EXPECT_EQ(pool.Size(), poolSize);

        auto v0 = pool.Create(0);
        auto v1 = pool.Create(1);
        EXPECT_EQ(pool.InitReadyCount(), 2u);
        EXPECT_NE(v0, v1);

        pool.Destroy(v0);
        auto v2 = pool.Create(2);
        // Recycled freelist slot should reuse v0's address; high-water stays 2.
        EXPECT_EQ(v2, v0);
        EXPECT_EQ(pool.InitReadyCount(), 2u);
        pool.Destroy(v1);
        pool.Destroy(v2);
    }
    // Destroy(v0)+Destroy(v1)+Destroy(v2); ~ItemPool skips freelist slots (no double-dtor).
    EXPECT_EQ(3, Object::GetDestructorCount());
}
