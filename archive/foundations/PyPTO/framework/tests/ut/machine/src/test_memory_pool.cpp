/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include <cstring>

#define private public
#include "machine/runtime/memory_utils/memory_pool.h"
#undef private

#include "adapter/api/acl_api.h"
#include "adapter/api/runtime_api.h"

using namespace npu::tile_fwk;

class MemoryPoolTest : public testing::Test {};

class MemoryPoolNpuTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto ret = AclInit(nullptr);
        if (ret != ACLRT_SUCCESS && ret != ACLRT_ERROR_REPEAT_INITIALIZE) {
            GTEST_SKIP() << "AclInit failed";
        }
        ret = AclRtSetDevice(0);
        if (ret != ACLRT_SUCCESS) {
            GTEST_SKIP() << "AclRtSetDevice failed";
        }
    }

    void TearDown() override
    {
        AclRtResetDevice(0);
        AclFinalize();
    }
};

TEST_F(MemoryPoolTest, MemoryBlock_AllCases)
{
    uint8_t buffer[1024];
    MemoryBlock block(buffer, sizeof(buffer));
    EXPECT_EQ(block.baseAddr, buffer);
    EXPECT_EQ(block.blockSize, sizeof(buffer));
    EXPECT_EQ(block.usedSize, 0u);

    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, buffer);
    EXPECT_EQ(block.usedSize, sizeof(buffer));

    uint8_t buffer2[512];
    MemoryBlock block2(buffer2, sizeof(buffer2));
    void* ptr2 = block2.Allocate(512);
    EXPECT_EQ(ptr2, buffer2);

    uint8_t buffer3[256];
    MemoryBlock block3(buffer3, sizeof(buffer3));
    void* ptr3 = block3.Allocate(512);
    EXPECT_EQ(ptr3, nullptr);
    EXPECT_EQ(block3.usedSize, 0u);

    uint8_t buffer4[1024];
    MemoryBlock block4(buffer4, sizeof(buffer4));
    void* ptr4a = block4.Allocate(512);
    EXPECT_NE(ptr4a, nullptr);
    void* ptr4b = block4.Allocate(512);
    EXPECT_EQ(ptr4b, nullptr);

    uint8_t buffer5[1024];
    MemoryBlock block5(buffer5, sizeof(buffer5));
    void* ptr5 = block5.Allocate(0);
    EXPECT_NE(ptr5, nullptr);
}

TEST_F(MemoryPoolTest, DevMemoryPool_AllCases)
{
    auto& pool1 = DevMemoryPool::Instance();
    auto& pool2 = DevMemoryPool::Instance();
    EXPECT_EQ(&pool1, &pool2);

    EXPECT_ANY_THROW(pool1.FreeDevAddr(nullptr));

    EXPECT_TRUE(pool1.CheckAllSentinels());

    pool1.DestroyPool();
}

TEST_F(MemoryPoolNpuTest, AllocDevAddr_Success)
{
    auto& pool = DevMemoryPool::Instance();
    uint8_t* devAddr = nullptr;
    pool.AllocDevAddr(&devAddr, 1024);
    ASSERT_NE(devAddr, nullptr);

    uint8_t hostBuf[1024] = {0};
    auto ret = RuntimeMemcpyDirect(hostBuf, 1024, devAddr, 1024, RtMemcpyKind::DEVICE_TO_HOST);
    EXPECT_EQ(ret, RT_SUCCESS);

    pool.FreeDevAddr(devAddr);
}

TEST_F(MemoryPoolNpuTest, AllocDevAddr_MultipleAllocations)
{
    auto& pool = DevMemoryPool::Instance();
    std::vector<uint8_t*> addrs;

    for (int i = 0; i < 5; ++i) {
        uint8_t* devAddr = nullptr;
        pool.AllocDevAddr(&devAddr, 2048);
        ASSERT_NE(devAddr, nullptr);
        addrs.push_back(devAddr);
    }

    for (auto addr : addrs) {
        pool.FreeDevAddr(addr);
    }
}

TEST_F(MemoryPoolNpuTest, AllocDevAddr_ZeroSize)
{
    auto& pool = DevMemoryPool::Instance();
    uint8_t* devAddr = nullptr;
    EXPECT_ANY_THROW(pool.AllocDevAddr(&devAddr, 0));
}

TEST_F(MemoryPoolNpuTest, AllocDevAddr_NullPtr)
{
    auto& pool = DevMemoryPool::Instance();
    EXPECT_ANY_THROW(pool.AllocDevAddr(nullptr, 1024));
}

TEST_F(MemoryPoolNpuTest, CheckAllSentinels_NoAllocations)
{
    auto& pool = DevMemoryPool::Instance();
    EXPECT_TRUE(pool.CheckAllSentinels());
}

TEST_F(MemoryPoolNpuTest, DestroyPool_Safe)
{
    auto& pool = DevMemoryPool::Instance();
    uint8_t* devAddr = nullptr;
    pool.AllocDevAddr(&devAddr, 512);
    if (devAddr != nullptr) {
        pool.FreeDevAddr(devAddr);
    }
    pool.DestroyPool();
    SUCCEED();
}

TEST_F(MemoryPoolNpuTest, FreeDevAddr_UnknownPointer)
{
    auto& pool = DevMemoryPool::Instance();
    uint8_t fakeAddr[100];
    EXPECT_ANY_THROW(pool.FreeDevAddr(fakeAddr));
}

TEST_F(MemoryPoolNpuTest, AllocDevAddr_LargeSize)
{
    auto& pool = DevMemoryPool::Instance();
    uint8_t* devAddr = nullptr;
    pool.AllocDevAddr(&devAddr, 10 * 1024 * 1024);
    if (devAddr != nullptr) {
        pool.FreeDevAddr(devAddr);
    }
    SUCCEED();
}

TEST_F(MemoryPoolNpuTest, MultipleAllocFreeCycles)
{
    auto& pool = DevMemoryPool::Instance();
    for (int i = 0; i < 10; ++i) {
        uint8_t* devAddr = nullptr;
        pool.AllocDevAddr(&devAddr, 1024 * (i + 1));
        if (devAddr != nullptr) {
            pool.FreeDevAddr(devAddr);
        }
    }
    SUCCEED();
}

TEST_F(MemoryPoolNpuTest, NormalizedRtMemcpy_HostToDevice)
{
    auto& pool = DevMemoryPool::Instance();
    uint8_t* devDst = nullptr;
    pool.AllocDevAddr(&devDst, 256);
    if (devDst == nullptr) {
        GTEST_SKIP() << "Failed to allocate device memory";
    }

    uint8_t hostSrc[256];
    for (int i = 0; i < 256; ++i) {
        hostSrc[i] = static_cast<uint8_t>(i);
    }

    auto copyRet = NormalizedRtMemcpy(devDst, 256, hostSrc, 256, RtMemcpyKind::HOST_TO_DEVICE);
    EXPECT_EQ(copyRet, RT_SUCCESS);

    uint8_t hostDst[256];
    copyRet = NormalizedRtMemcpy(hostDst, 256, devDst, 256, RtMemcpyKind::DEVICE_TO_HOST);
    EXPECT_EQ(copyRet, RT_SUCCESS);

    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(hostDst[i], static_cast<uint8_t>(i));
    }

    pool.FreeDevAddr(devDst);
}

TEST_F(MemoryPoolNpuTest, NormalizedRtMemcpy_DeviceToDevice)
{
    auto& pool = DevMemoryPool::Instance();
    uint8_t* devSrc = nullptr;
    uint8_t* devDst = nullptr;

    pool.AllocDevAddr(&devSrc, 128);
    pool.AllocDevAddr(&devDst, 128);

    if (devSrc == nullptr || devDst == nullptr) {
        if (devSrc)
            pool.FreeDevAddr(devSrc);
        if (devDst)
            pool.FreeDevAddr(devDst);
        GTEST_SKIP() << "Failed to allocate device memory";
    }

    uint8_t hostData[128];
    for (int i = 0; i < 128; ++i) {
        hostData[i] = static_cast<uint8_t>(i * 2);
    }

    NormalizedRtMemcpy(devSrc, 128, hostData, 128, RtMemcpyKind::HOST_TO_DEVICE);
    auto copyRet = NormalizedRtMemcpy(devDst, 128, devSrc, 128, RtMemcpyKind::DEVICE_TO_DEVICE);
    EXPECT_EQ(copyRet, RT_SUCCESS);

    uint8_t hostResult[128];
    NormalizedRtMemcpy(hostResult, 128, devDst, 128, RtMemcpyKind::DEVICE_TO_HOST);

    for (int i = 0; i < 128; ++i) {
        EXPECT_EQ(hostResult[i], static_cast<uint8_t>(i * 2));
    }

    pool.FreeDevAddr(devSrc);
    pool.FreeDevAddr(devDst);
}

TEST_F(MemoryPoolNpuTest, DynamicRecycle_EmptyPool)
{
    auto& pool = DevMemoryPool::Instance();
    pool.DynamicRecycle();
    EXPECT_TRUE(pool.memoryBlocks_.empty());
}

TEST_F(MemoryPoolNpuTest, DynamicRecycle_WithEmptyBlocks)
{
    auto& pool = DevMemoryPool::Instance();

    uint8_t* addr1 = nullptr;
    uint8_t* addr2 = nullptr;
    pool.AllocDevAddr(&addr1, 1024);
    pool.AllocDevAddr(&addr2, 2048);

    if (addr1 && addr2) {
        pool.FreeDevAddr(addr1);
        pool.FreeDevAddr(addr2);
        pool.DynamicRecycle();
    }
}

TEST_F(MemoryPoolNpuTest, PrintPoolStatus_EmptyPool)
{
    auto& pool = DevMemoryPool::Instance();
    pool.DestroyPool();
    pool.PrintPoolStatus();
}

TEST_F(MemoryPoolNpuTest, PrintPoolStatus_WithAllocations)
{
    auto& pool = DevMemoryPool::Instance();

    uint8_t* addr = nullptr;
    pool.AllocDevAddr(&addr, 4096);

    if (addr) {
        pool.PrintPoolStatus();
        pool.FreeDevAddr(addr);
    }
}

TEST_F(MemoryPoolNpuTest, AllocDevAddrInPool_MultipleAllocationsSameBlock)
{
    auto& pool = DevMemoryPool::Instance();
    pool.DestroyPool();

    uint8_t* addr1 = nullptr;
    uint8_t* addr2 = nullptr;

    pool.AllocDevAddr(&addr1, 512);
    ASSERT_NE(addr1, nullptr);

    pool.AllocDevAddr(&addr2, 512);
    ASSERT_NE(addr2, nullptr);

    EXPECT_NE(addr1, addr2);

    pool.FreeDevAddr(addr1);
    pool.FreeDevAddr(addr2);
}

TEST_F(MemoryPoolNpuTest, AllocDevAddrInPool_ReuseBlockAfterFree)
{
    auto& pool = DevMemoryPool::Instance();
    pool.DestroyPool();

    uint8_t* addr1 = nullptr;
    pool.AllocDevAddr(&addr1, 1024);
    ASSERT_NE(addr1, nullptr);

    pool.FreeDevAddr(addr1);

    uint8_t* addr2 = nullptr;
    pool.AllocDevAddr(&addr2, 1024);
    ASSERT_NE(addr2, nullptr);

    pool.FreeDevAddr(addr2);
}

TEST_F(MemoryPoolNpuTest, PrintSentinelVal_IndirectCall)
{
    auto& pool = DevMemoryPool::Instance();
    pool.DestroyPool();

    pool.needMemCheck_ = true;

    uint8_t* addr = nullptr;
    pool.AllocDevAddr(&addr, 256);

    if (addr) {
        pool.FreeDevAddr(addr);
    }

    pool.needMemCheck_ = false;
}
