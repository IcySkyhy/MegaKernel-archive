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
 * \file test_topo_processor.cpp
 * \brief Unit tests for TopoProcessor, IdList, IdListKey (interface/cache/topo_processor.h)
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <tuple>
#include <unordered_map>
#include "gtest/gtest.h"
#include "interface/cache/topo_processor.h"
#include "interface/cache/function_cache.h"
#include "tilefwk/core_func_data.h"

using namespace npu::tile_fwk;

static size_t TopoEntrySize(uint64_t depNum, uint64_t extParamNum)
{
    return sizeof(CoreFunctionTopo) + sizeof(uint64_t) * (depNum + extParamNum);
}

static std::shared_ptr<CoreFunctionTopoCache> BuildTopoCache(
    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int64_t, uint64_t, uint64_t, std::vector<uint64_t>>>&
        entries)
{
    uint64_t topoNum = entries.size();
    size_t offsetsSize = (1 + topoNum) * sizeof(uint64_t);
    size_t topoDataSize = 0;
    for (const auto& e : entries) {
        uint64_t depNum = std::get<4>(e);
        uint64_t extParamNum = std::get<5>(e);
        topoDataSize += TopoEntrySize(depNum, extParamNum);
    }
    size_t totalSize = offsetsSize + topoDataSize;

    auto cache = CacheValue::CreateCache<CoreFunctionTopoCache>(totalSize);
    memset_s(cache.get(), totalSize, 0, totalSize);
    cache->dataSize = totalSize;

    uint8_t* base = reinterpret_cast<uint8_t*>(cache.get());
    uint64_t* header = reinterpret_cast<uint64_t*>(base);
    header[0] = totalSize;

    uint64_t curOffset = offsetsSize;
    for (uint64_t i = 0; i < topoNum; i++) {
        header[i + 1] = curOffset;
        const auto& e = entries[i];
        uint64_t coreType = std::get<0>(e);
        uint32_t extType = static_cast<uint32_t>(std::get<1>(e));
        uint64_t psgId = std::get<2>(e);
        int64_t readyCount = std::get<3>(e);
        uint64_t depNum = std::get<4>(e);
        uint64_t extParamNum = std::get<5>(e);
        const std::vector<uint64_t>& depIds = std::get<6>(e);

        CoreFunctionTopo* topo = reinterpret_cast<CoreFunctionTopo*>(base + curOffset);
        topo->coreType = coreType;
        topo->extType = extType;
        topo->psgId = psgId;
        topo->readyCount = readyCount;
        topo->depNum = depNum;
        topo->extParamNum = extParamNum;
        for (uint64_t j = 0; j < depIds.size(); j++) {
            topo->depIds[j] = depIds[j];
        }

        curOffset += TopoEntrySize(depNum, extParamNum);
    }
    return cache;
}

class TestIdListKey : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestIdListKey, EqualitySameHashAndLen)
{
    IdListKey a{0x1234ABCD, 5};
    IdListKey b{0x1234ABCD, 5};
    EXPECT_TRUE(a == b);
}

TEST_F(TestIdListKey, InequalityDifferentHash)
{
    IdListKey a{0x1234ABCD, 5};
    IdListKey b{0x5678EF01, 5};
    EXPECT_FALSE(a == b);
}

TEST_F(TestIdListKey, InequalityDifferentLen)
{
    IdListKey a{0x1234ABCD, 5};
    IdListKey b{0x1234ABCD, 3};
    EXPECT_FALSE(a == b);
}

TEST_F(TestIdListKey, HashValueConsistency)
{
    IdListKey k{0x1234ABCD, 5};
    uint32_t h1 = HashValue(k);
    uint32_t h2 = HashValue(k);
    EXPECT_EQ(h1, h2);
}

TEST_F(TestIdListKey, HashValueDifferentKeysDifferentHashes)
{
    IdListKey a{0x1234ABCD, 5};
    IdListKey b{0x5678EF01, 5};
    EXPECT_NE(HashValue(a), HashValue(b));
}

TEST_F(TestIdListKey, IdListKeyHashFunctor)
{
    IdListKey k{0xABCD, 3};
    IdListKeyHash hasher;
    uint32_t h1 = hasher(k);
    uint32_t h2 = hasher(k);
    EXPECT_EQ(h1, h2);
    EXPECT_EQ(h1, HashValue(k));
}

TEST_F(TestIdListKey, IdListKeyInUnorderedMap)
{
    std::unordered_map<IdListKey, int, IdListKeyHash> map;
    IdListKey k1{0x1111, 2};
    IdListKey k2{0x2222, 2};
    map[k1] = 10;
    map[k2] = 20;
    EXPECT_EQ(map.size(), 2UL);
    auto it = map.find(k1);
    EXPECT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

class TestIdList : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestIdList, ConstructionCopiesData)
{
    uint64_t src[] = {10, 20, 30};
    IdList list(src, 3, 99);
    EXPECT_EQ(list.len, 3U);
    EXPECT_EQ(list.keyId, 99UL);
    EXPECT_EQ(list.data[0], 10UL);
    EXPECT_EQ(list.data[1], 20UL);
    EXPECT_EQ(list.data[2], 30UL);
}

TEST_F(TestIdList, ConstructionSingleElement)
{
    uint64_t src[] = {42};
    IdList list(src, 1, 1);
    EXPECT_EQ(list.len, 1U);
    EXPECT_EQ(list.keyId, 1UL);
    EXPECT_EQ(list.data[0], 42UL);
}

class TestTopoProcessor : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestTopoProcessor, NoBatchDependReturnsOriginal)
{
    auto cache = BuildTopoCache({
        {static_cast<uint64_t>(MachineType::AIC), 0, 0, 0, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIV), 0, 1, 0, 1, 0, {0}},
    });
    uint64_t topoNum = 2;

    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(10, 1);

    EXPECT_EQ(virtualNum, 0UL);
    EXPECT_EQ(resultCache.get(), cache.get());
}

TEST_F(TestTopoProcessor, NoTopoNodesReturnsOriginal)
{
    auto cache = BuildTopoCache({});
    uint64_t topoNum = 0;

    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(1, 1);

    EXPECT_EQ(virtualNum, 0UL);
}

static auto BuildPureBatchDependCache()
{
    auto cache = BuildTopoCache({
        {static_cast<uint64_t>(MachineType::AIC), 0, 0, -2, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 1, -2, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 2, 0, 2, 0, {0, 1}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 3, 0, 2, 0, {0, 1}},
    });
    uint64_t topoNum = 4;
    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(2, 1);
    EXPECT_EQ(virtualNum, 1UL);
    EXPECT_NE(resultCache.get(), nullptr);
    return std::make_tuple(resultCache, virtualNum, topoNum);
}

TEST_F(TestTopoProcessor, PureBatchDependCreatesVirtualPure)
{
    auto [resultCache, virtualNum, topoNum] = BuildPureBatchDependCache();

    uint8_t* base = reinterpret_cast<uint8_t*>(resultCache.get());
    uint64_t* offsets = reinterpret_cast<uint64_t*>(base);

    CoreFunctionTopo* virtualTopo = reinterpret_cast<CoreFunctionTopo*>(base + offsets[topoNum + 1]);
    EXPECT_EQ(virtualTopo->coreType, static_cast<uint64_t>(MachineType::VIRTUAL_PURE));
    EXPECT_EQ(virtualTopo->psgId, 0xFFFFFFFFUL);
    EXPECT_EQ(virtualTopo->readyCount, -2);
}

TEST_F(TestTopoProcessor, PureBatchDependOldTopoRedirected)
{
    auto cache = BuildTopoCache({
        {static_cast<uint64_t>(MachineType::AIC), 0, 0, -3, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 1, -3, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 2, 0, 2, 0, {0, 1}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 3, 0, 2, 0, {0, 1}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 4, 0, 2, 0, {0, 1}},
    });
    uint64_t topoNum = 5;

    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(2, 1);

    EXPECT_EQ(virtualNum, 1UL);

    uint8_t* base = reinterpret_cast<uint8_t*>(resultCache.get());
    uint64_t* offsets = reinterpret_cast<uint64_t*>(base);

    uint64_t virtualTopoId = topoNum;
    for (uint64_t i = 2; i < topoNum; i++) {
        CoreFunctionTopo* consumerTopo = reinterpret_cast<CoreFunctionTopo*>(base + offsets[i + 1]);
        EXPECT_EQ(consumerTopo->depNum, 1UL);
        EXPECT_EQ(consumerTopo->depIds[0], virtualTopoId);
    }
}

TEST_F(TestTopoProcessor, MixBatchDependCreatesVirtualMix)
{
    auto cache = BuildTopoCache({
        {static_cast<uint64_t>(MachineType::AIC), 0, 0, -1, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 1, 0, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 2, 0, 1, 0, {0}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 3, 0, 1, 0, {0}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 4, 0, 1, 0, {0}},
    });
    uint64_t topoNum = 5;

    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(1, 3);

    EXPECT_EQ(virtualNum, 1UL);

    uint8_t* base = reinterpret_cast<uint8_t*>(resultCache.get());
    uint64_t* offsets = reinterpret_cast<uint64_t*>(base);
    CoreFunctionTopo* virtualTopo = reinterpret_cast<CoreFunctionTopo*>(base + offsets[topoNum + 1]);
    EXPECT_EQ(virtualTopo->coreType, static_cast<uint64_t>(MachineType::VIRTUAL_MIX));
}

TEST_F(TestTopoProcessor, BelowMergeNumThresholdNoVirtualNode)
{
    auto cache = BuildTopoCache({
        {static_cast<uint64_t>(MachineType::AIC), 0, 0, 0, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 1, 0, 1, 0, {0}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 2, 0, 1, 0, {0}},
    });
    uint64_t topoNum = 3;

    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(1, 10);

    EXPECT_EQ(virtualNum, 0UL);
    EXPECT_EQ(resultCache.get(), cache.get());
}

TEST_F(TestTopoProcessor, SingleTopoNoDependencies)
{
    auto cache = BuildTopoCache({
        {static_cast<uint64_t>(MachineType::AIC), 0, 0, 0, 0, 0, {}},
    });
    uint64_t topoNum = 1;

    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(1, 1);

    EXPECT_EQ(virtualNum, 0UL);
}

TEST_F(TestTopoProcessor, DifferentDepIdListsCreateSeparateVirtualNodes)
{
    auto cache = BuildTopoCache({
        {static_cast<uint64_t>(MachineType::AIC), 0, 0, -2, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 1, -2, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 2, 0, 2, 0, {0, 1}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 3, 0, 2, 0, {0, 1}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 4, -2, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 5, 0, 1, 0, {4}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 6, 0, 1, 0, {4}},
    });
    uint64_t topoNum = 7;

    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(1, 1);

    EXPECT_EQ(virtualNum, 2UL);

    uint8_t* base = reinterpret_cast<uint8_t*>(resultCache.get());
    uint64_t* offsets = reinterpret_cast<uint64_t*>(base);

    CoreFunctionTopo* v0 = reinterpret_cast<CoreFunctionTopo*>(base + offsets[topoNum + 1]);
    EXPECT_EQ(v0->coreType, static_cast<uint64_t>(MachineType::VIRTUAL_PURE));
    EXPECT_EQ(v0->depNum, 2UL);

    CoreFunctionTopo* v1 = reinterpret_cast<CoreFunctionTopo*>(base + offsets[topoNum + 2]);
    EXPECT_EQ(v1->coreType, static_cast<uint64_t>(MachineType::VIRTUAL_PURE));
    EXPECT_EQ(v1->depNum, 1UL);
}

TEST_F(TestTopoProcessor, FinalTopoContainsAllOriginalAndVirtualTopos)
{
    auto [resultCache, virtualNum, topoNum] = BuildPureBatchDependCache();

    uint8_t* base = reinterpret_cast<uint8_t*>(resultCache.get());

    uint64_t totalTopoNum = topoNum + virtualNum;
    uint64_t* offsets = reinterpret_cast<uint64_t*>(base);
    for (uint64_t i = 0; i < totalTopoNum; i++) {
        CoreFunctionTopo* topo = reinterpret_cast<CoreFunctionTopo*>(base + offsets[i + 1]);
        EXPECT_NE(topo, nullptr);
    }

    CoreFunctionTopo* originalLeaf0 = reinterpret_cast<CoreFunctionTopo*>(base + offsets[1]);
    EXPECT_EQ(originalLeaf0->coreType, static_cast<uint64_t>(MachineType::AIC));
}

TEST_F(TestTopoProcessor, ExtParamNumPreservedInFinalTopo)
{
    auto cache = BuildTopoCache({
        {static_cast<uint64_t>(MachineType::AICPU), 1, 0, 0, 2, 1, {1, 2}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 1, 0, 1, 0, {0}},
    });
    uint64_t topoNum = 2;

    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(10, 1);

    EXPECT_EQ(virtualNum, 0UL);

    uint8_t* base = reinterpret_cast<uint8_t*>(resultCache.get());
    uint64_t* offsets = reinterpret_cast<uint64_t*>(base);
    CoreFunctionTopo* topo0 = reinterpret_cast<CoreFunctionTopo*>(base + offsets[1]);
    EXPECT_EQ(topo0->extParamNum, 1UL);
}

TEST_F(TestTopoProcessor, VirtualTopoReadyCountNegated)
{
    auto cache = BuildTopoCache({
        {static_cast<uint64_t>(MachineType::AIC), 0, 0, -4, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 1, -4, 0, 0, {}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 2, 0, 2, 0, {0, 1}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 3, 0, 2, 0, {0, 1}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 4, 0, 2, 0, {0, 1}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 5, 0, 2, 0, {0, 1}},
    });
    uint64_t topoNum = 6;

    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(2, 1);

    EXPECT_EQ(virtualNum, 1UL);

    uint8_t* base = reinterpret_cast<uint8_t*>(resultCache.get());
    uint64_t* offsets = reinterpret_cast<uint64_t*>(base);
    CoreFunctionTopo* vTopo = reinterpret_cast<CoreFunctionTopo*>(base + offsets[topoNum + 1]);
    EXPECT_EQ(vTopo->readyCount, -4);
}

TEST_F(TestTopoProcessor, MergeBatchDependWithHighThresholdFiltersAll)
{
    auto cache = BuildTopoCache({
        {static_cast<uint64_t>(MachineType::AIC), 0, 0, 0, 1, 0, {1}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 1, 0, 2, 0, {0, 2}},
        {static_cast<uint64_t>(MachineType::AIC), 0, 2, 0, 1, 0, {1}},
    });
    uint64_t topoNum = 3;

    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(100, 1);

    EXPECT_EQ(virtualNum, 0UL);
    EXPECT_EQ(resultCache.get(), cache.get());
}

TEST_F(TestTopoProcessor, TopoWithAICPUCoreType)
{
    auto cache = BuildTopoCache({
        {static_cast<uint64_t>(MachineType::AICPU), 5, 0, 0, 1, 2, {1}},
    });
    uint64_t topoNum = 1;

    TopoProcessor proc(cache, topoNum);
    auto [resultCache, virtualNum] = proc.MergeBatchDepend(100, 1);

    EXPECT_EQ(virtualNum, 0UL);

    uint8_t* base = reinterpret_cast<uint8_t*>(resultCache.get());
    uint64_t* offsets = reinterpret_cast<uint64_t*>(base);
    CoreFunctionTopo* topo0 = reinterpret_cast<CoreFunctionTopo*>(base + offsets[1]);
    EXPECT_EQ(topo0->coreType, static_cast<uint64_t>(MachineType::AICPU));
    EXPECT_EQ(topo0->extType, 5U);
    EXPECT_EQ(topo0->extParamNum, 2UL);
}
