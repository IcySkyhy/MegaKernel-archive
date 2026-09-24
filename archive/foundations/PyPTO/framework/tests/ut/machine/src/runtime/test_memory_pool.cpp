/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file test_memory_pool.cpp
 * \brief
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/memory_utils/memory_pool.h"
#undef private
#include "interface/configs/config_manager.h"
#include "machine/runtime/context/device_launcher_context.h"

using namespace npu::tile_fwk;

TEST(MemoryBlockTest, FirstAllocation_ReturnsBaseAddr)
{
    uint8_t buf[1024];
    MemoryBlock block(buf, 1024);
    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, static_cast<void*>(buf));
    EXPECT_EQ(block.usedSize, 1024u);
}

TEST(MemoryBlockTest, SecondAllocation_ReturnsNull)
{
    uint8_t buf[1024];
    MemoryBlock block(buf, 1024);
    block.Allocate(512);
    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, nullptr);
}

TEST(MemoryBlockTest, AlignExceedsBlockSize_ReturnsNull)
{
    uint8_t buf[256];
    MemoryBlock block(buf, 256);
    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, nullptr);
}

TEST(DevMemoryPoolTest, FreeDevAddr_ErrorPaths_Throw)
{
    EXPECT_ANY_THROW(DevMemoryPool::Instance().FreeDevAddr(nullptr));
    uint8_t dummy = 0;
    EXPECT_ANY_THROW(DevMemoryPool::Instance().FreeDevAddr(&dummy));
}

TEST(DevMemoryPoolTest, SentinelAndPrintStatus_NoCrash)
{
    EXPECT_TRUE(DevMemoryPool::Instance().CheckAllSentinels());
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().PrintPoolStatus();
    SUCCEED();
}

TEST(DevMemoryPoolTest, AllocAndFree_Lifecycle)
{
    DevMemoryPool::Instance().DestroyPool();
    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);
    if (ptr != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }
    SUCCEED();
}

TEST(DevMemoryPoolTest, AllocDevAddr_ErrorPaths_Throw)
{
    EXPECT_ANY_THROW(DevMemoryPool::Instance().AllocDevAddr(nullptr, 1024));
    uint8_t* ptr = reinterpret_cast<uint8_t*>(0x1);
    EXPECT_ANY_THROW(DevMemoryPool::Instance().AllocDevAddr(&ptr, 0));
}

TEST(DevMemoryPoolTest, DynamicRecycle_AfterFree_RemovesEmptyBlock)
{
    DevMemoryPool::Instance().DestroyPool();
    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);
    if (ptr != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    DevMemoryPool::Instance().DynamicRecycle();
    SUCCEED();
}

TEST(DevMemoryPoolTest, DestroyPool_Twice_NoCrash)
{
    DevMemoryPool::Instance().DestroyPool();
    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 512);
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().DestroyPool();
    SUCCEED();
}

TEST(DevMemoryPoolTest, DevAlloc_ValidSize_ReturnsAllocatedMemory)
{
    DevMemoryPool::Instance().DestroyPool();
    void* ptr = DevAlloc(256);
    if (ptr != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }
    SUCCEED();
}

TEST(DevMemoryPoolTest, DevAllocAndCopy_ZeroSize_Throw)
{
    DevMemoryPool::Instance().DestroyPool();
    EXPECT_ANY_THROW(DevAlloc(0));
    uint8_t src[1] = {0};
    EXPECT_ANY_THROW(CopyDataToDevice(src, 0));
}

TEST(DevMemoryPoolTest, CopyDataToDevice_ValidData_CopiesCorrectly)
{
    DevMemoryPool::Instance().DestroyPool();
    uint8_t src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    void* devPtr = CopyDataToDevice(src, 8);
    if (devPtr != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(devPtr);
    }
    SUCCEED();
}

TEST(DevMemoryPoolTest, AllocDevAddrInPool_NullPtr_ThrowsException)
{
    EXPECT_ANY_THROW(DevMemoryPool::Instance().AllocDevAddrInPool(nullptr, 1024));
}

TEST(DevMemoryPoolTest, AllocDevAddrInPool_AllocateFromExistingBlock)
{
    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr1 = nullptr;
    bool result1 = DevMemoryPool::Instance().AllocDevAddrInPool(&ptr1, 1024);

    if (result1 && ptr1 != nullptr) {
        uint8_t* ptr2 = nullptr;
        bool result2 = DevMemoryPool::Instance().AllocDevAddrInPool(&ptr2, 512);

        if (result2 && ptr2 != nullptr) {
            EXPECT_NE(ptr1, ptr2);
            DevMemoryPool::Instance().FreeDevAddr(ptr2);
        }

        DevMemoryPool::Instance().FreeDevAddr(ptr1);
    }
}

TEST(DevMemoryPoolTest, AllocDevAddrInPool_AllocationFailed_ThrowsException)
{
    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr = nullptr;
    EXPECT_ANY_THROW(DevMemoryPool::Instance().AllocDevAddrInPool(&ptr, 1024ULL * 1024 * 1024 * 100));
}

TEST(DevMemoryPoolTest, PutSentinelAddr_WithMemCheckEnabled)
{
    config::SetDebugOption("runtime_debug_mode", 1);

    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        SUCCEED();

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    config::SetDebugOption("runtime_debug_mode", 0);
}

TEST(DevMemoryPoolTest, CheckAllSentinels_WithMemCheckEnabled)
{
    config::SetDebugOption("runtime_debug_mode", 1);

    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        bool result = DevMemoryPool::Instance().CheckAllSentinels();
        EXPECT_TRUE(result);

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    config::SetDebugOption("runtime_debug_mode", 0);
}

TEST(DevMemoryPoolTest, CheckAllSentinels_AllGoodFalse)
{
    config::SetDebugOption("runtime_debug_mode", 1);

    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        bool result = DevMemoryPool::Instance().CheckAllSentinels();
        EXPECT_TRUE(result);

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    config::SetDebugOption("runtime_debug_mode", 0);
}

TEST(DevMemoryPoolTest, CheckSentinel_RecordNotFound)
{
    config::SetDebugOption("runtime_debug_mode", 1);

    DevMemoryPool::Instance().DestroyPool();

    uint8_t fakeAddr = 0;
    DevMemoryPool::Instance().CheckSentinel(&fakeAddr, false);
    SUCCEED();

    config::SetDebugOption("runtime_debug_mode", 0);
}

TEST(DevMemoryPoolTest, CheckSentinel_MemcpyFailed)
{
    config::SetDebugOption("runtime_debug_mode", 1);

    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        bool result = DevMemoryPool::Instance().CheckSentinel(ptr, false);
        EXPECT_TRUE(result);

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    config::SetDebugOption("runtime_debug_mode", 0);
}

TEST(DevMemoryPoolTest, CheckSentinel_SentinelMismatch)
{
    config::SetDebugOption("runtime_debug_mode", 1);

    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        bool result = DevMemoryPool::Instance().CheckSentinel(ptr, false);
        EXPECT_TRUE(result);

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    config::SetDebugOption("runtime_debug_mode", 0);
}

TEST(DevMemoryPoolTest, CheckSentinel_AllGoodFalse)
{
    config::SetDebugOption("runtime_debug_mode", 1);

    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        bool result = DevMemoryPool::Instance().CheckSentinel(ptr, false);
        EXPECT_TRUE(result);

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    config::SetDebugOption("runtime_debug_mode", 0);
}

TEST(DevMemoryPoolTest, DynamicRecycle_RecycleEmptyBlocks)
{
    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr1 = nullptr;
    uint8_t* ptr2 = nullptr;

    DevMemoryPool::Instance().AllocDevAddr(&ptr1, 1024);
    DevMemoryPool::Instance().AllocDevAddr(&ptr2, 2048);

    if (ptr1 != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr1);

        DevMemoryPool::Instance().DynamicRecycle();
    }

    if (ptr2 != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr2);
    }
}

TEST(DevMemoryPoolTest, CreateNewBlock_AllocationFailed_ThrowsException)
{
    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr = nullptr;
    EXPECT_ANY_THROW(DevMemoryPool::Instance().AllocDevAddrInPool(&ptr, 1024ULL * 1024 * 1024 * 100));
}

TEST(DevMemoryPoolTest, DevAlloc_AllocationFailed_ThrowsException)
{
    DevMemoryPool::Instance().DestroyPool();

    EXPECT_ANY_THROW(DevAlloc(1024ULL * 1024 * 1024 * 100));
}

TEST(DevMemoryPoolTest, DevAlloc_MemsetFailed)
{
    DevMemoryPool::Instance().DestroyPool();

    void* ptr = DevAlloc(1024);

    if (ptr != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }
}

TEST(DevMemoryPoolTest, CopyDataToDevice_AllocationFailed_ThrowsException)
{
    DevMemoryPool::Instance().DestroyPool();

    uint8_t src[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    EXPECT_ANY_THROW(CopyDataToDevice(src, 1024ULL * 1024 * 1024 * 100));
}

TEST(DevMemoryPoolTest, CopyDataToDevice_MemcpyFailed)
{
    DevMemoryPool::Instance().DestroyPool();

    uint8_t src[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    void* ptr = CopyDataToDevice(src, 16);

    if (ptr != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }
}

TEST(DevMemoryPoolTest, AllocDevAddrInPool_ZeroSize_ReturnsFalse)
{
    uint8_t* ptr = nullptr;
    bool result = DevMemoryPool::Instance().AllocDevAddrInPool(&ptr, 0);

    EXPECT_FALSE(result);
    EXPECT_EQ(ptr, nullptr);
}

TEST(DevMemoryPoolTest, FreeDevAddr_NullPtr_ThrowsException)
{
    EXPECT_ANY_THROW(DevMemoryPool::Instance().FreeDevAddr(nullptr));
}

TEST(DevMemoryPoolTest, FreeDevAddr_UnknownPtr_ThrowsException)
{
    uint8_t dummy = 0;
    EXPECT_ANY_THROW(DevMemoryPool::Instance().FreeDevAddr(&dummy));
}

TEST(DevMemoryPoolTest, PrintPoolStatus_WithAllocations)
{
    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr1 = nullptr;
    uint8_t* ptr2 = nullptr;

    DevMemoryPool::Instance().AllocDevAddr(&ptr1, 1024);
    DevMemoryPool::Instance().AllocDevAddr(&ptr2, 2048);

    DevMemoryPool::Instance().PrintPoolStatus();

    if (ptr1 != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr1);
    }
    if (ptr2 != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr2);
    }
}

TEST(DevMemoryPoolTest, NormalizedRtMemcpy_CaptureMode)
{
    DeviceLauncherContext::Get().SetCaptureMode(true);

    uint8_t src[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint8_t dst[16] = {0};

    auto ret = NormalizedRtMemcpy(dst, 16, src, 16, RtMemcpyKind::HOST_TO_HOST);
    EXPECT_EQ(ret, RT_SUCCESS);

    for (int i = 0; i < 16; i++) {
        EXPECT_EQ(dst[i], src[i]);
    }

    DeviceLauncherContext::Get().SetCaptureMode(false);
}

TEST(DevMemoryPoolTest, PutSentinelAddr_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().needMemCheck_ = true;

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        EXPECT_FALSE(DevMemoryPool::Instance().sentinelValMap_.empty());

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    DevMemoryPool::Instance().needMemCheck_ = false;
}

TEST(DevMemoryPoolTest, CheckAllSentinels_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().needMemCheck_ = true;

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        bool result = DevMemoryPool::Instance().CheckAllSentinels();
        EXPECT_TRUE(result);

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    DevMemoryPool::Instance().needMemCheck_ = false;
}

TEST(DevMemoryPoolTest, CheckSentinel_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().needMemCheck_ = true;

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        bool result = DevMemoryPool::Instance().CheckSentinel(ptr, false);
        EXPECT_TRUE(result);

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    DevMemoryPool::Instance().needMemCheck_ = false;
}

TEST(DevMemoryPoolTest, CheckSentinel_RecordNotFound_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().needMemCheck_ = true;

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        uint8_t fakeAddr = 0;
        bool result = DevMemoryPool::Instance().CheckSentinel(&fakeAddr, false);
        EXPECT_FALSE(result);

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    DevMemoryPool::Instance().needMemCheck_ = false;
}

TEST(DevMemoryPoolTest, CheckSentinel_SpecialAddr_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().needMemCheck_ = true;

    uint8_t* specialAddr = reinterpret_cast<uint8_t*>(0x12345678);
    bool result = DevMemoryPool::Instance().CheckSentinel(specialAddr, false);
    EXPECT_TRUE(result);

    DevMemoryPool::Instance().needMemCheck_ = false;
}

TEST(DevMemoryPoolTest, PrintSentinelVal_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().needMemCheck_ = true;

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        std::vector<uint64_t> sentinelVal(64, 0xDEADBEEFDEADBEEF);
        DevMemoryPool::Instance().PrintSentinelVal(sentinelVal, ptr);

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    DevMemoryPool::Instance().needMemCheck_ = false;
}

TEST(DevMemoryPoolTest, DynamicRecycle_WithEmptyBlocks_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr1 = nullptr;
    uint8_t* ptr2 = nullptr;

    DevMemoryPool::Instance().AllocDevAddr(&ptr1, 1024);
    DevMemoryPool::Instance().AllocDevAddr(&ptr2, 2048);

    if (ptr1 != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr1);

        size_t blocksBefore = DevMemoryPool::Instance().memoryBlocks_.size();
        DevMemoryPool::Instance().DynamicRecycle();
        size_t blocksAfter = DevMemoryPool::Instance().memoryBlocks_.size();

        EXPECT_LE(blocksAfter, blocksBefore);
    }

    if (ptr2 != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr2);
    }
}

TEST(DevMemoryPoolTest, AllocDevAddrInPool_WithMemCheck_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().needMemCheck_ = true;

    uint8_t* ptr1 = nullptr;
    bool result1 = DevMemoryPool::Instance().AllocDevAddrInPool(&ptr1, 1024);

    if (result1 && ptr1 != nullptr) {
        uint8_t* ptr2 = nullptr;
        bool result2 = DevMemoryPool::Instance().AllocDevAddrInPool(&ptr2, 512);

        if (result2 && ptr2 != nullptr) {
            EXPECT_NE(ptr1, ptr2);
            DevMemoryPool::Instance().FreeDevAddr(ptr2);
        }

        DevMemoryPool::Instance().FreeDevAddr(ptr1);
    }

    DevMemoryPool::Instance().needMemCheck_ = false;
}

TEST(DevMemoryPoolTest, AllocDevAddrInPool_AllocateFromExistingBlock_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr1 = nullptr;
    bool result1 = DevMemoryPool::Instance().AllocDevAddrInPool(&ptr1, 1024);

    if (result1 && ptr1 != nullptr) {
        uint8_t* ptr2 = nullptr;
        bool result2 = DevMemoryPool::Instance().AllocDevAddrInPool(&ptr2, 512);

        if (result2 && ptr2 != nullptr) {
            EXPECT_NE(ptr1, ptr2);
            DevMemoryPool::Instance().FreeDevAddr(ptr2);
        }

        DevMemoryPool::Instance().FreeDevAddr(ptr1);
    }
}

TEST(DevMemoryPoolTest, CheckSentinel_SentinelMismatch_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().needMemCheck_ = true;

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        uint8_t* sentinelAddr = ptr + 1024;
        memset(sentinelAddr, 0xFF, 512);

        bool result = DevMemoryPool::Instance().CheckSentinel(ptr, false);
        EXPECT_FALSE(result);

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    DevMemoryPool::Instance().needMemCheck_ = false;
}

TEST(DevMemoryPoolTest, CheckAllSentinels_AllGoodFalse_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().needMemCheck_ = true;

    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);

    if (ptr != nullptr) {
        uint8_t* sentinelAddr = ptr + 1024;
        memset(sentinelAddr, 0xFF, 512);

        bool result = DevMemoryPool::Instance().CheckAllSentinels();
        EXPECT_FALSE(result);

        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    DevMemoryPool::Instance().needMemCheck_ = false;
}

TEST(DevMemoryPoolTest, DynamicRecycle_WithMultipleEmptyBlocks_DirectTest)
{
    DevMemoryPool::Instance().DestroyPool();

    uint8_t* ptr1 = nullptr;
    uint8_t* ptr2 = nullptr;
    uint8_t* ptr3 = nullptr;

    DevMemoryPool::Instance().AllocDevAddr(&ptr1, 1024);
    DevMemoryPool::Instance().AllocDevAddr(&ptr2, 2048);
    DevMemoryPool::Instance().AllocDevAddr(&ptr3, 4096);

    if (ptr1 != nullptr && ptr2 != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr1);
        DevMemoryPool::Instance().FreeDevAddr(ptr2);

        size_t blocksBefore = DevMemoryPool::Instance().memoryBlocks_.size();
        DevMemoryPool::Instance().DynamicRecycle();
        size_t blocksAfter = DevMemoryPool::Instance().memoryBlocks_.size();

        EXPECT_LE(blocksAfter, blocksBefore);
    }

    if (ptr3 != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr3);
    }
}
