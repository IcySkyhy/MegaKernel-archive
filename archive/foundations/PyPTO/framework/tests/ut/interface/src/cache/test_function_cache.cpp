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
 * \file test_function_cache.cpp
 * \brief Unit tests for FunctionCache, CacheValue, CacheHeader (interface/cache/function_cache.h/.cpp)
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include "gtest/gtest.h"
#include "interface/cache/function_cache.h"
#include "tilefwk/core_func_data.h"

using namespace npu::tile_fwk;

class TestCacheValue : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestCacheValue, CreateCacheTopoAllocation)
{
    size_t allocSize = sizeof(uint64_t) + sizeof(CoreFunctionTopo) * 3;
    auto cache = CacheValue::CreateCache<CoreFunctionTopoCache>(allocSize);
    EXPECT_NE(cache.get(), nullptr);
    memset_s(cache.get(), allocSize, 0, allocSize);
    cache->dataSize = allocSize;
    EXPECT_EQ(cache->dataSize, allocSize);
}

TEST_F(TestCacheValue, CreateCacheBinAllocation)
{
    size_t allocSize = sizeof(uint64_t) + sizeof(CoreFunctionBin) * 2;
    auto cache = CacheValue::CreateCache<CoreFunctionBinCache>(allocSize);
    EXPECT_NE(cache.get(), nullptr);
    memset_s(cache.get(), allocSize, 0, allocSize);
    cache->dataSize = allocSize;
    EXPECT_EQ(cache->dataSize, allocSize);
}

TEST_F(TestCacheValue, CreateCacheReadyListAllocation)
{
    size_t allocSize = sizeof(uint64_t) + sizeof(ReadyCoreFunction) * 4;
    auto cache = CacheValue::CreateCache<ReadyCoreFunctionCache>(allocSize);
    EXPECT_NE(cache.get(), nullptr);
    memset_s(cache.get(), allocSize, 0, allocSize);
    cache->dataSize = allocSize;
    EXPECT_EQ(cache->dataSize, allocSize);
}

TEST_F(TestCacheValue, CreateCacheSharedPtrOwnership)
{
    size_t allocSize = 128;
    auto cache = CacheValue::CreateCache<CoreFunctionTopoCache>(allocSize);
    auto copy = cache;
    EXPECT_EQ(copy.get(), cache.get());
    cache.reset();
    EXPECT_NE(copy.get(), nullptr);
    copy.reset();
}

TEST_F(TestCacheValue, DefaultInitNullptrs)
{
    CacheValue val;
    EXPECT_EQ(val.topoCache.get(), nullptr);
    EXPECT_EQ(val.binCache.get(), nullptr);
    EXPECT_EQ(val.readyListCache.get(), nullptr);
    EXPECT_EQ(val.GetFunction(), nullptr);
}

TEST_F(TestCacheValue, SetAndGetCacheFunction)
{
    CacheValue val;
    int dummy = 0;
    Function* fakePtr = reinterpret_cast<Function*>(&dummy);
    val.SetCacheFunction(fakePtr);
    EXPECT_EQ(val.GetFunction(), fakePtr);
    val.SetCacheFunction(nullptr);
    EXPECT_EQ(val.GetFunction(), nullptr);
}

class TestCacheHeader : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestCacheHeader, DefaultInit)
{
    CacheHeader hdr;
    EXPECT_EQ(hdr.virtualFunctionNum, 0UL);
}

TEST_F(TestCacheHeader, FieldAssignment)
{
    CacheHeader hdr{};
    hdr.coreFunctionNum = 5;
    hdr.virtualFunctionNum = 2;
    hdr.readyCoreFunctionNum = 3;
    hdr.programFuncionNum = 4;
    EXPECT_EQ(hdr.coreFunctionNum, 5UL);
    EXPECT_EQ(hdr.virtualFunctionNum, 2UL);
    EXPECT_EQ(hdr.readyCoreFunctionNum, 3UL);
    EXPECT_EQ(hdr.programFuncionNum, 4UL);
}

class TestFunctionCache : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestFunctionCache, GetEmptyCacheReturnsNullopt)
{
    FunctionCache fc;
    HashKey key(1UL);
    auto result = fc.Get(key);
    EXPECT_FALSE(result.has_value());
}

TEST_F(TestFunctionCache, GetCacheFunctionEmptyReturnsNullptr)
{
    FunctionCache fc;
    HashKey key(1UL);
    EXPECT_EQ(fc.GetCacheFunction(key), nullptr);
}

TEST_F(TestFunctionCache, SizeEmptyCache)
{
    FunctionCache fc;
    EXPECT_EQ(fc.Size(), 0UL);
}

TEST_F(TestFunctionCache, GetHitRateEmptyCache)
{
    FunctionCache fc;
    HashKey key(1UL);
    fc.Get(key);
    std::string rate = fc.GetHitRate();
    EXPECT_NE(rate.find("0"), std::string::npos);
}

TEST_F(TestFunctionCache, ResetClearsEmptyCache)
{
    FunctionCache fc;
    fc.Reset();
    EXPECT_EQ(fc.Size(), 0UL);
}

TEST_F(TestFunctionCache, MultipleMissGetHitRate)
{
    FunctionCache fc;
    HashKey k1(1UL);
    HashKey k2(2UL);
    HashKey k3(3UL);
    fc.Get(k1);
    fc.Get(k2);
    fc.Get(k3);
    std::string rate = fc.GetHitRate();
    EXPECT_NE(rate.find("0/3"), std::string::npos);
}

TEST_F(TestFunctionCache, PackedStructTopoCacheLayout)
{
    size_t topoEntrySize = sizeof(CoreFunctionTopo) + sizeof(uint64_t) * 2;
    size_t topoNum = 3;
    size_t totalSize = sizeof(uint64_t) + topoNum * sizeof(uint64_t) + topoNum * topoEntrySize;

    auto cache = CacheValue::CreateCache<CoreFunctionTopoCache>(totalSize);
    memset_s(cache.get(), totalSize, 0, totalSize);
    cache->dataSize = totalSize;

    uint8_t* base = reinterpret_cast<uint8_t*>(cache.get());
    uint64_t* offsets = reinterpret_cast<uint64_t*>(base + sizeof(uint64_t));
    offsets[0] = sizeof(uint64_t) + topoNum * sizeof(uint64_t);
    offsets[1] = offsets[0] + topoEntrySize;
    offsets[2] = offsets[1] + topoEntrySize;

    CoreFunctionTopo* topo0 = reinterpret_cast<CoreFunctionTopo*>(base + offsets[0]);
    topo0->coreType = static_cast<uint64_t>(MachineType::AIC);
    topo0->psgId = 0;
    topo0->readyCount = 0;
    topo0->depNum = 2;
    topo0->depIds[0] = 1;
    topo0->depIds[1] = 2;

    EXPECT_EQ(topo0->coreType, static_cast<uint64_t>(MachineType::AIC));
    EXPECT_EQ(topo0->depNum, 2UL);
    EXPECT_EQ(topo0->depIds[0], 1UL);
    EXPECT_EQ(topo0->depIds[1], 2UL);
}

TEST_F(TestFunctionCache, PackedStructBinCacheLayout)
{
    size_t binDataLen = 16;
    size_t binEntrySize = sizeof(uint64_t) + binDataLen;
    size_t binNum = 2;
    size_t totalSize = sizeof(uint64_t) + binNum * sizeof(uint64_t) + binNum * binEntrySize;

    auto cache = CacheValue::CreateCache<CoreFunctionBinCache>(totalSize);
    memset_s(cache.get(), totalSize, 0, totalSize);
    cache->dataSize = totalSize;

    uint8_t* base = reinterpret_cast<uint8_t*>(cache.get());
    uint64_t* offsets = reinterpret_cast<uint64_t*>(base + sizeof(uint64_t));

    offsets[0] = sizeof(uint64_t) + binNum * sizeof(uint64_t);
    offsets[1] = offsets[0] + binEntrySize;

    CoreFunctionBin* bin0 = reinterpret_cast<CoreFunctionBin*>(base + offsets[0]);
    bin0->size = binDataLen;
    memset_s(bin0->data, binDataLen, 0xAA, binDataLen);

    EXPECT_EQ(bin0->size, binDataLen);
    EXPECT_EQ(bin0->data[0], 0xAA);
}

TEST_F(TestFunctionCache, PackedStructReadyListCacheLayout)
{
    size_t readyNum = 3;
    size_t totalSize = sizeof(uint64_t) + readyNum * sizeof(ReadyCoreFunction);

    auto cache = CacheValue::CreateCache<ReadyCoreFunctionCache>(totalSize);
    memset_s(cache.get(), totalSize, 0, totalSize);
    cache->dataSize = totalSize;

    cache->readyCoreFunction[0].id = 0;
    cache->readyCoreFunction[0].coreType = static_cast<uint64_t>(MachineType::AIC);
    cache->readyCoreFunction[1].id = 1;
    cache->readyCoreFunction[1].coreType = static_cast<uint64_t>(MachineType::AIV);
    cache->readyCoreFunction[2].id = 5;
    cache->readyCoreFunction[2].coreType = static_cast<uint64_t>(MachineType::AICPU);

    EXPECT_EQ(cache->readyCoreFunction[0].id, 0UL);
    EXPECT_EQ(cache->readyCoreFunction[0].coreType, static_cast<uint64_t>(MachineType::AIC));
    EXPECT_EQ(cache->readyCoreFunction[1].coreType, static_cast<uint64_t>(MachineType::AIV));
    EXPECT_EQ(cache->readyCoreFunction[2].coreType, static_cast<uint64_t>(MachineType::AICPU));
    EXPECT_EQ(cache->dataSize, totalSize);
}

TEST_F(TestFunctionCache, CreateCacheWithLargeSize)
{
    size_t largeSize = 1024 * 1024;
    auto cache = CacheValue::CreateCache<CoreFunctionTopoCache>(largeSize);
    EXPECT_NE(cache.get(), nullptr);
    memset_s(cache.get(), largeSize, 0, largeSize);
}

TEST_F(TestFunctionCache, CacheValueFullComposition)
{
    CacheValue val;
    val.header.coreFunctionNum = 10;
    val.header.virtualFunctionNum = 2;
    val.header.readyCoreFunctionNum = 3;
    val.header.programFuncionNum = 8;

    size_t topoSize = 256;
    val.topoCache = CacheValue::CreateCache<CoreFunctionTopoCache>(topoSize);
    memset_s(val.topoCache.get(), topoSize, 0, topoSize);
    val.topoCache->dataSize = topoSize;

    size_t binSize = 128;
    val.binCache = CacheValue::CreateCache<CoreFunctionBinCache>(binSize);
    memset_s(val.binCache.get(), binSize, 0, binSize);
    val.binCache->dataSize = binSize;

    size_t readySize = 64;
    val.readyListCache = CacheValue::CreateCache<ReadyCoreFunctionCache>(readySize);
    memset_s(val.readyListCache.get(), readySize, 0, readySize);
    val.readyListCache->dataSize = readySize;

    EXPECT_EQ(val.header.coreFunctionNum, 10UL);
    EXPECT_EQ(val.header.virtualFunctionNum, 2UL);
    EXPECT_NE(val.topoCache.get(), nullptr);
    EXPECT_NE(val.binCache.get(), nullptr);
    EXPECT_NE(val.readyListCache.get(), nullptr);
}
