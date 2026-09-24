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
 * \file test_slab_allocator.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "machine/utils/dynamic/allocator/slab_ws_allocator.h"

using namespace npu::tile_fwk::dynamic;
class SlabWsAllocatorTest : public ::testing::Test {
protected:
    static constexpr size_t TEST_MEM_SIZE = 1024 * 1024; // 1MB test memory
    static constexpr uint32_t SLAB_ALIGN_SIZE = 4096;    // 4KB slab size

    void SetUp() override
    {
        testMemory = malloc(TEST_MEM_SIZE);
        allocator.Init(testMemory, TEST_MEM_SIZE, SLAB_ALIGN_SIZE);
    }

    void TearDown() override
    {
        free(testMemory);
        testMemory = nullptr;
    }

    void* testMemory;
    SlabWsAllocator allocator;
};

TEST_F(SlabWsAllocatorTest, Initialization)
{
    EXPECT_NE(testMemory, nullptr);
    EXPECT_TRUE(allocator.RegistCache(0, 64));
    EXPECT_TRUE(allocator.RegistCache(1, 128));
    EXPECT_TRUE(allocator.RegistCache(2, 256));
}

TEST_F(SlabWsAllocatorTest, CacheRegistration)
{
    EXPECT_TRUE(allocator.RegistCache(0, 64));
    EXPECT_TRUE(allocator.RegistCache(1, 128));
    EXPECT_TRUE(allocator.RegistCache(2, 256));

    EXPECT_TRUE(allocator.RegistCache(0, 64));

    EXPECT_TRUE(allocator.RegistCache(1, 128));

    EXPECT_FALSE(allocator.RegistCache(2, 512));

    EXPECT_FALSE(allocator.RegistCache(SLAB_ALLOCATOR_MAX_CACHES, 64));
}

TEST_F(SlabWsAllocatorTest, BasicAllocation)
{
    allocator.RegistCache(0, 64);
    allocator.RegistCache(1, 128);

    void* obj1 = allocator.Alloc(0);
    void* obj2 = allocator.Alloc(1);

    EXPECT_NE(obj1, nullptr);
    EXPECT_NE(obj2, nullptr);
    EXPECT_NE(obj1, obj2);
}

TEST_F(SlabWsAllocatorTest, MultipleAllocationsSameCache)
{
    const int NUM_ALLOCS = 10;
    allocator.RegistCache(0, 64);

    void* objects[NUM_ALLOCS];
    for (int i = 0; i < NUM_ALLOCS; ++i) {
        objects[i] = allocator.Alloc(0);
        EXPECT_NE(objects[i], nullptr);
    }

    for (int i = 0; i < NUM_ALLOCS; ++i) {
        for (int j = i + 1; j < NUM_ALLOCS; ++j) {
            EXPECT_NE(objects[i], objects[j]);
        }
    }
}

TEST_F(SlabWsAllocatorTest, StageAllocationTracking)
{
    allocator.RegistCache(0, 64);
    allocator.RegistCache(1, 128);

    (void)allocator.Alloc(0);
    (void)allocator.Alloc(0);
    (void)allocator.Alloc(1);

    auto allocInfo = allocator.PopStageAllocMem(false, 0);

    void* cache0_head = allocInfo.heads[0];
    EXPECT_NE(cache0_head, nullptr);

    void* current = cache0_head;
    int count = 0;
    while (current) {
        count++;
        current = *static_cast<void**>(current);
    }
    EXPECT_EQ(count, 2);

    void* cache1_head = allocInfo.heads[1];
    EXPECT_NE(cache1_head, nullptr);
    EXPECT_EQ(*static_cast<void**>(cache1_head), nullptr);

    (void)allocator.Alloc(0);
    (void)allocator.Alloc(0);
    void* tailalloc = allocator.Alloc(0);
    (void)allocator.Alloc(1);

    allocInfo = allocator.PopStageAllocMem(true, 0);
    cache0_head = allocInfo.heads[0];
    EXPECT_NE(cache0_head, nullptr);

    current = cache0_head;
    count = 0;
    while (current) {
        count++;
        current = *static_cast<void**>(current);
    }
    EXPECT_EQ(count, 2);
    cache1_head = allocInfo.heads[1];
    EXPECT_NE(cache1_head, nullptr);
    EXPECT_EQ(*static_cast<void**>(cache1_head), nullptr);

    allocInfo = allocator.PopStageAllocMem(true, 0);
    EXPECT_EQ(allocInfo.heads[0], nullptr);
    EXPECT_EQ(allocInfo.tails[0], nullptr);

    uint32_t offset = SLAB_OBJ_META_SIZE;
    allocInfo = allocator.PopStageAllocMem(false, 0);
    EXPECT_EQ(allocInfo.heads[0], (uint8_t*)tailalloc - offset);
    EXPECT_EQ(allocInfo.tails[0], (uint8_t*)tailalloc - offset);

    allocInfo = allocator.PopStageAllocMem(false, 0);
    EXPECT_EQ(allocInfo.heads[0], nullptr);
    EXPECT_EQ(allocInfo.tails[0], nullptr);
}

TEST_F(SlabWsAllocatorTest, BatchFreeing)
{
    allocator.RegistCache(0, 64);

    void* obj1 = allocator.Alloc(0);
    void* obj2 = allocator.Alloc(0);
    auto allocInfo1 = allocator.PopStageAllocMem(false, 0);

    allocator.FreeStageAllocMem(allocInfo1);

    void* obj3 = allocator.Alloc(0);
    void* obj4 = allocator.Alloc(0);
    (void)allocator.PopStageAllocMem(false, 0);

    EXPECT_EQ(obj1, obj3);
    EXPECT_EQ(obj2, obj4);
}

TEST_F(SlabWsAllocatorTest, MixedCacheAllocations)
{
    allocator.RegistCache(0, 64);
    allocator.RegistCache(1, 128);
    allocator.RegistCache(2, 256);

    void* small1 = allocator.Alloc(0);
    void* medium1 = allocator.Alloc(1);
    void* large1 = allocator.Alloc(2);

    EXPECT_NE(small1, nullptr);
    EXPECT_NE(medium1, nullptr);
    EXPECT_NE(large1, nullptr);

    auto allocInfo = allocator.PopStageAllocMem(false, 0);
    allocator.FreeStageAllocMem(allocInfo);

    // Allocate again
    void* small2 = allocator.Alloc(0);
    void* medium2 = allocator.Alloc(1);
    void* large2 = allocator.Alloc(2);

    EXPECT_NE(small2, nullptr);
    EXPECT_NE(medium2, nullptr);
    EXPECT_NE(large2, nullptr);
}

TEST_F(SlabWsAllocatorTest, ExistCacheCheck)
{
    allocator.RegistCache(0, 64);
    allocator.RegistCache(1, 128);

    EXPECT_TRUE(allocator.ExistCache(0, 64));
    EXPECT_TRUE(allocator.ExistCache(1, 128));
    EXPECT_FALSE(allocator.ExistCache(0, 128));
    EXPECT_FALSE(allocator.ExistCache(2, 64));
    EXPECT_FALSE(allocator.ExistCache(SLAB_ALLOCATOR_MAX_CACHES, 64));
}

TEST_F(SlabWsAllocatorTest, InvalidAllocation)
{
    EXPECT_EQ(allocator.Alloc(0), nullptr);

    allocator.RegistCache(0, 64);
    EXPECT_EQ(allocator.Alloc(SLAB_ALLOCATOR_MAX_CACHES), nullptr);
}

TEST_F(SlabWsAllocatorTest, MemoryExhaustion)
{
    allocator.RegistCache(0, 64);

    auto stats = allocator.GetAllocatorStats();
    const int OBJS_PER_SLAB = SlabWsAllocator::CalcObjsPerSlab(SLAB_ALIGN_SIZE, 64);
    const int TOTAL_SLABS = stats.totalSlabCount;
    const int MAX_ALLOCS = OBJS_PER_SLAB * TOTAL_SLABS;

    int successfulAllocs = 0;
    for (int i = 0; i < MAX_ALLOCS + 10; ++i) {
        void* obj = allocator.Alloc(0);
        if (obj) {
            successfulAllocs++;
        }
    }

    EXPECT_EQ(successfulAllocs, MAX_ALLOCS);
}

TEST_F(SlabWsAllocatorTest, PartialBatchFreeing)
{
    allocator.RegistCache(0, 64);
    allocator.RegistCache(1, 128);

    void* obj1 = allocator.Alloc(0);
    void* obj2 = allocator.Alloc(1);
    (void)allocator.Alloc(0);

    auto allocInfo = allocator.PopStageAllocMem(false, 0);
    allocInfo.heads[1] = nullptr;

    allocator.FreeStageAllocMem(allocInfo);

    void* obj4 = allocator.Alloc(0);
    EXPECT_EQ(obj1, obj4);

    void* obj5 = allocator.Alloc(1);
    EXPECT_NE(obj2, obj5);
}

TEST_F(SlabWsAllocatorTest, EmptyBatchFreeing)
{
    auto emptyInfo = allocator.PopStageAllocMem(false, 0);

    allocator.FreeStageAllocMem(emptyInfo);

    allocator.RegistCache(0, 64);
    void* obj = allocator.Alloc(0);
    auto allocInfo = allocator.PopStageAllocMem(false, 0);
    allocator.FreeStageAllocMem(allocInfo);

    EXPECT_NE(obj, nullptr);
}

TEST_F(SlabWsAllocatorTest, LazySlabAllocation)
{
    allocator.RegistCache(0, 64);

    void* obj1 = allocator.Alloc(0);
    EXPECT_NE(obj1, nullptr);

    void* obj2 = allocator.Alloc(0);
    EXPECT_NE(obj2, nullptr);
}

TEST_F(SlabWsAllocatorTest, SlabReuseAfterFree)
{
    allocator.RegistCache(0, 64);

    void* obj1 = allocator.Alloc(0);
    void* obj2 = allocator.Alloc(0);
    auto allocInfo1 = allocator.PopStageAllocMem(false, 0);
    allocator.FreeStageAllocMem(allocInfo1);

    void* obj3 = allocator.Alloc(0);
    void* obj4 = allocator.Alloc(0);
    (void)allocator.PopStageAllocMem(false, 0);

    EXPECT_EQ(obj1, obj3);
    EXPECT_EQ(obj2, obj4);
}

TEST_F(SlabWsAllocatorTest, MultipleAllocationBatches)
{
    allocator.RegistCache(0, 64);

    const int NUM_BATCHES = 5;
    const int ALLOCS_PER_BATCH = 10;

    for (int batch = 0; batch < NUM_BATCHES; ++batch) {
        for (int i = 0; i < ALLOCS_PER_BATCH; ++i) {
            void* obj = allocator.Alloc(0);
            EXPECT_NE(obj, nullptr);
        }

        auto allocInfo = allocator.PopStageAllocMem(false, 0);
        allocator.FreeStageAllocMem(allocInfo);
    }

    void* finalObj = allocator.Alloc(0);
    EXPECT_NE(finalObj, nullptr);
}

TEST_F(SlabWsAllocatorTest, LargeObjectAllocation)
{
    const uint32_t largeObjSize = SLAB_ALIGN_SIZE / 2;
    allocator.RegistCache(0, largeObjSize);

    void* obj1 = allocator.Alloc(0);
    EXPECT_NE(obj1, nullptr);

    void* obj2 = allocator.Alloc(0);
    EXPECT_NE(obj2, nullptr);

    uintptr_t slab1 = reinterpret_cast<uintptr_t>(obj1) & ~(SLAB_ALIGN_SIZE - 1);
    uintptr_t slab2 = reinterpret_cast<uintptr_t>(obj2) & ~(SLAB_ALIGN_SIZE - 1);
    EXPECT_NE(slab1, slab2);
}

TEST_F(SlabWsAllocatorTest, CacheWithZeroSize)
{
    EXPECT_FALSE(allocator.RegistCache(0, 0));

    EXPECT_TRUE(allocator.RegistCache(0, 64));
}

TEST_F(SlabWsAllocatorTest, AlignmentHandling)
{
    const uint32_t nonPowerOfTwoAlign = 3000;
    void* altMemory = malloc(TEST_MEM_SIZE);
    SlabWsAllocator altAllocator;
    altAllocator.Init(altMemory, TEST_MEM_SIZE, nonPowerOfTwoAlign);

    altAllocator.RegistCache(0, 64);
    void* obj = altAllocator.Alloc(0);
    EXPECT_NE(obj, nullptr);

    free(altMemory);
}

TEST_F(SlabWsAllocatorTest, MagicNumberIntact)
{
    allocator.RegistCache(0, 64);
    allocator.RegistCache(1, 128);

    (void)allocator.Alloc(0);
    (void)allocator.Alloc(0);
    (void)allocator.Alloc(1);

    auto allocInfo = allocator.PopStageAllocMem(false, 0);
    SlabWsAllocator::VerifyStageAllocMem(allocInfo, "test");
    allocator.FreeStageAllocMem(allocInfo);
}

TEST_F(SlabWsAllocatorTest, ObjAddressAlignment)
{
    allocator.RegistCache(0, 64);
    allocator.RegistCache(1, 100);
    allocator.RegistCache(2, 200);

    for (int i = 0; i < 200; ++i) {
        void* obj = allocator.Alloc(i % 3);
        ASSERT_NE(obj, nullptr);
        uintptr_t addr = reinterpret_cast<uintptr_t>(obj);
        EXPECT_EQ(addr % SLAB_ALIGN_GRANULARITY, 0u) << "obj " << i << " addr=" << addr << " not 64-byte aligned";
    }
}
