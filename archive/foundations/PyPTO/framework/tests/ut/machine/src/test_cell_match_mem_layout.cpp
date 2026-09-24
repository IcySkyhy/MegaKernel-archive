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
 * \file test_cell_match_mem_layout.cpp
 * \brief Unit tests for dynamic CellMatch functionality
 *
 * This test file verifies the correctness of:
 * 1. Dynamic memory layout calculations
 * 2. Metadata bit manipulation functions
 * 3. Mutex operation detection logic
 * 4. Op-id management (add and get with desc)
 * 5. Capacity configuration logic
 */

#include <gtest/gtest.h>
#include <cstring>
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "interface/machine/device/tilefwk/aikernel_data.h"

using namespace npu::tile_fwk::dynamic;
using npu::tile_fwk::MakeTaskID;

class CellMatchDynamicTest : public testing::Test {
protected:
    DevCellMatchTableDesc desc;

    void SetUp() override
    {
        desc.cacheOpMaxCount[CELL_MATCH_OP_TYPE_ATOMIC_WRITE] = CELL_MATCH_MAX_ATOMIC_WRITE_COUNT;
        desc.cacheOpMaxCount[CELL_MATCH_OP_TYPE_NORMAL_WRITE] = CELL_MATCH_MAX_NORMAL_WRITE_COUNT;
        desc.cacheOpMaxCount[CELL_MATCH_OP_TYPE_READ] = CELL_MATCH_MAX_READ_COUNT;
        desc.UpdateCellMemLayOut();
    }

    void TearDown() override {}
};

TEST_F(CellMatchDynamicTest, CellMemLayOutCalculation)
{
    DevCellMatchTableDesc testDesc;
    testDesc.cacheOpMaxCount[CELL_MATCH_OP_TYPE_ATOMIC_WRITE] = CELL_MATCH_MAX_ATOMIC_WRITE_COUNT;
    testDesc.cacheOpMaxCount[CELL_MATCH_OP_TYPE_NORMAL_WRITE] = 1;
    testDesc.cacheOpMaxCount[CELL_MATCH_OP_TYPE_READ] = CELL_MATCH_MAX_READ_COUNT;
    testDesc.UpdateCellMemLayOut();

    EXPECT_EQ(testDesc.cellUint64Size, 1 + 1 + CELL_MATCH_MAX_ATOMIC_WRITE_COUNT + CELL_MATCH_MAX_READ_COUNT);
    EXPECT_EQ(testDesc.opMemLayOutIndex[CELL_MATCH_OP_TYPE_NORMAL_WRITE], 1);
    EXPECT_EQ(testDesc.opMemLayOutIndex[CELL_MATCH_OP_TYPE_ATOMIC_WRITE], 2);
    EXPECT_EQ(testDesc.opMemLayOutIndex[CELL_MATCH_OP_TYPE_READ], 2 + CELL_MATCH_MAX_ATOMIC_WRITE_COUNT);

    testDesc.cacheOpMaxCount[CELL_MATCH_OP_TYPE_ATOMIC_WRITE] = 0;
    testDesc.cacheOpMaxCount[CELL_MATCH_OP_TYPE_READ] = 0;
    testDesc.UpdateCellMemLayOut();
    EXPECT_EQ(testDesc.cellUint64Size, 1 + 1);
    EXPECT_EQ(testDesc.opMemLayOutIndex[CELL_MATCH_OP_TYPE_NORMAL_WRITE], 1);
    EXPECT_EQ(testDesc.opMemLayOutIndex[CELL_MATCH_OP_TYPE_ATOMIC_WRITE], 2);
    EXPECT_EQ(testDesc.opMemLayOutIndex[CELL_MATCH_OP_TYPE_READ], 2);
}

TEST_F(CellMatchDynamicTest, CellIndexCalculation)
{
    EXPECT_EQ(CellMatchCellIndexToMemBase(0, desc), 0);
    EXPECT_EQ(CellMatchCellIndexToMemBase(1, desc), desc.cellUint64Size);
    EXPECT_EQ(CellMatchCellIndexToMemBase(5, desc), 5 * desc.cellUint64Size);
}

TEST_F(CellMatchDynamicTest, MetadataBitManipulation)
{
    uint64_t meta = 0;

    CellMatchSetCurrentOpType(meta, CELL_MATCH_OP_TYPE_NORMAL_WRITE);
    EXPECT_EQ(CellMatchGetCurrentOpType(meta), CELL_MATCH_OP_TYPE_NORMAL_WRITE);

    CellMatchSetCurrentOpType(meta, CELL_MATCH_OP_TYPE_ATOMIC_WRITE);
    EXPECT_EQ(CellMatchGetCurrentOpType(meta), CELL_MATCH_OP_TYPE_ATOMIC_WRITE);

    CellMatchSetCurrentOpType(meta, CELL_MATCH_OP_TYPE_READ);
    EXPECT_EQ(CellMatchGetCurrentOpType(meta), CELL_MATCH_OP_TYPE_READ);

    CellMatchSetPrevMutexOpType(meta, CELL_MATCH_OP_TYPE_NORMAL_WRITE);
    EXPECT_EQ(CellMatchGetPrevMutexOpType(meta), CELL_MATCH_OP_TYPE_NORMAL_WRITE);

    CellMatchSetCurrentOpCount(meta, 5);
    EXPECT_EQ(CellMatchGetCurrentOpCount(meta), 5);

    CellMatchSetCurrentOpCount(meta, 127);
    EXPECT_EQ(CellMatchGetCurrentOpCount(meta), 127);

    CellMatchSetCurrentOpCount(meta, CELL_MATCH_METADATA_OP_COUNT_MAX);
    EXPECT_EQ(CellMatchGetCurrentOpCount(meta), CELL_MATCH_METADATA_OP_COUNT_MAX);

    CellMatchSetPrevMutexOpCount(meta, 10);
    EXPECT_EQ(CellMatchGetPrevMutexOpCount(meta), 10);

    CellMatchSetTagId(meta, 12345);
    EXPECT_EQ(CellMatchGetTagId(meta), 12345);

    meta = 0;
    CellMatchSetCurrentOpType(meta, CELL_MATCH_OP_TYPE_ATOMIC_WRITE);
    CellMatchSetCurrentOpCount(meta, 50);
    CellMatchSetPrevMutexOpType(meta, CELL_MATCH_OP_TYPE_READ);
    CellMatchSetPrevMutexOpCount(meta, 20);
    CellMatchSetTagId(meta, 999);

    EXPECT_EQ(CellMatchGetCurrentOpType(meta), CELL_MATCH_OP_TYPE_ATOMIC_WRITE);
    EXPECT_EQ(CellMatchGetCurrentOpCount(meta), 50);
    EXPECT_EQ(CellMatchGetPrevMutexOpType(meta), CELL_MATCH_OP_TYPE_READ);
    EXPECT_EQ(CellMatchGetPrevMutexOpCount(meta), 20);
    EXPECT_EQ(CellMatchGetTagId(meta), 999);
}

TEST_F(CellMatchDynamicTest, MutexOperationDetection)
{
    EXPECT_FALSE(CellMatchIsMutexOp(CELL_MATCH_OP_TYPE_ATOMIC_WRITE, CELL_MATCH_OP_TYPE_ATOMIC_WRITE));
    EXPECT_FALSE(CellMatchIsMutexOp(CELL_MATCH_OP_TYPE_READ, CELL_MATCH_OP_TYPE_READ));

    EXPECT_TRUE(CellMatchIsMutexOp(CELL_MATCH_OP_TYPE_READ, CELL_MATCH_OP_TYPE_NORMAL_WRITE));
    EXPECT_TRUE(CellMatchIsMutexOp(CELL_MATCH_OP_TYPE_NORMAL_WRITE, CELL_MATCH_OP_TYPE_READ));

    EXPECT_TRUE(CellMatchIsMutexOp(CELL_MATCH_OP_TYPE_READ, CELL_MATCH_OP_TYPE_ATOMIC_WRITE));
    EXPECT_TRUE(CellMatchIsMutexOp(CELL_MATCH_OP_TYPE_ATOMIC_WRITE, CELL_MATCH_OP_TYPE_READ));

    EXPECT_TRUE(CellMatchIsMutexOp(CELL_MATCH_OP_TYPE_ATOMIC_WRITE, CELL_MATCH_OP_TYPE_NORMAL_WRITE));
    EXPECT_TRUE(CellMatchIsMutexOp(CELL_MATCH_OP_TYPE_NORMAL_WRITE, CELL_MATCH_OP_TYPE_ATOMIC_WRITE));

    EXPECT_TRUE(CellMatchIsMutexOp(CELL_MATCH_OP_TYPE_NORMAL_WRITE, CELL_MATCH_OP_TYPE_NORMAL_WRITE));

    EXPECT_TRUE(CellMatchIsMutexOp(CELL_MATCH_OP_TYPE_NONE, CELL_MATCH_OP_TYPE_READ));
}

TEST_F(CellMatchDynamicTest, OpIdManagement)
{
    uint64_t cellMatchTable[512];
    (void)memset_s(cellMatchTable, sizeof(cellMatchTable), 0, sizeof(cellMatchTable));

    for (int i = 0; i < 512; i++) {
        cellMatchTable[i] = AICORE_TASK_INIT;
    }

    uint64_t cellMemBase = CellMatchCellIndexToMemBase(0, desc);

    uint64_t atomicTaskId1 = (1ULL << CELL_MATCH_META_TAGID_SHIFT32) | MakeTaskID(2, 10);
    CellMatchAddOpId(cellMatchTable, cellMemBase, atomicTaskId1, 0, CELL_MATCH_OP_TYPE_ATOMIC_WRITE, desc);

    uint64_t retrievedAtomic1 = CellMatchGetOpId(cellMatchTable, cellMemBase, CELL_MATCH_OP_TYPE_ATOMIC_WRITE, 0, desc);
    EXPECT_EQ(retrievedAtomic1, atomicTaskId1);

    uint64_t readTaskId1 = (1ULL << CELL_MATCH_META_TAGID_SHIFT32) | MakeTaskID(3, 15);
    CellMatchAddOpId(cellMatchTable, cellMemBase, readTaskId1, 0, CELL_MATCH_OP_TYPE_READ, desc);

    uint64_t retrievedRead1 = CellMatchGetOpId(cellMatchTable, cellMemBase, CELL_MATCH_OP_TYPE_READ, 0, desc);
    EXPECT_EQ(retrievedRead1, readTaskId1);

    uint64_t normalTaskId1 = (1ULL << CELL_MATCH_META_TAGID_SHIFT32) | MakeTaskID(4, 20);
    CellMatchAddOpId(cellMatchTable, cellMemBase, normalTaskId1, 0, CELL_MATCH_OP_TYPE_NORMAL_WRITE, desc);

    uint64_t retrievedNormal1 = CellMatchGetOpId(cellMatchTable, cellMemBase, CELL_MATCH_OP_TYPE_NORMAL_WRITE, 0, desc);
    EXPECT_EQ(retrievedNormal1, normalTaskId1);

    uint64_t atomicTaskId2 = (1ULL << CELL_MATCH_META_TAGID_SHIFT32) | MakeTaskID(5, 25);
    CellMatchAddOpId(cellMatchTable, cellMemBase, atomicTaskId2, 1, CELL_MATCH_OP_TYPE_ATOMIC_WRITE, desc);

    uint64_t retrievedAtomic2 = CellMatchGetOpId(cellMatchTable, cellMemBase, CELL_MATCH_OP_TYPE_ATOMIC_WRITE, 1, desc);
    EXPECT_NE(retrievedAtomic2, AICORE_TASK_INIT);
}

TEST_F(CellMatchDynamicTest, CapacityConfiguration)
{
    EXPECT_EQ(CELL_MATCH_METADATA_OP_COUNT_MAX, 255);
    EXPECT_EQ(CELL_MATCH_MAX_ATOMIC_WRITE_COUNT, 254);
    EXPECT_EQ(CELL_MATCH_MAX_READ_COUNT, 128);
    EXPECT_EQ(CELL_MATCH_MAX_NORMAL_WRITE_COUNT, 1);
    EXPECT_EQ(CELL_MATCH_INVALID_OP_COUNT, 0xFF);
    EXPECT_LT(CELL_MATCH_MAX_ATOMIC_WRITE_COUNT, CELL_MATCH_INVALID_OP_COUNT);
}

TEST_F(CellMatchDynamicTest, PrevMutexCountMaxNotTreatedAsInvalid)
{
    uint64_t meta = 0;
    CellMatchSetPrevMutexOpType(meta, CELL_MATCH_OP_TYPE_ATOMIC_WRITE);
    CellMatchSetPrevMutexOpCount(meta, CELL_MATCH_MAX_ATOMIC_WRITE_COUNT);
    EXPECT_EQ(CellMatchGetPrevMutexOpCount(meta), CELL_MATCH_MAX_ATOMIC_WRITE_COUNT);
    EXPECT_NE(CellMatchGetPrevMutexOpCount(meta), CELL_MATCH_INVALID_OP_COUNT);

    CellMatchSetPrevMutexOpType(meta, CELL_MATCH_OP_TYPE_NONE);
    CellMatchSetPrevMutexOpCount(meta, CELL_MATCH_INVALID_OP_COUNT);
    EXPECT_EQ(CellMatchGetPrevMutexOpCount(meta), CELL_MATCH_INVALID_OP_COUNT);
}

TEST_F(CellMatchDynamicTest, OpIdBoundaryCheck)
{
    uint64_t cellMatchTable[1024];
    (void)memset_s(cellMatchTable, sizeof(cellMatchTable), 0, sizeof(cellMatchTable));

    uint64_t cellMemBase = 0;

    uint64_t taskId = (1ULL << CELL_MATCH_META_TAGID_SHIFT32) | MakeTaskID(2, 10);

    CellMatchAddOpId(cellMatchTable, cellMemBase, taskId, desc.GetCacheOpMaxCount(CELL_MATCH_OP_TYPE_ATOMIC_WRITE),
                     CELL_MATCH_OP_TYPE_ATOMIC_WRITE, desc);
    uint64_t retrieved = CellMatchGetOpId(cellMatchTable, cellMemBase, CELL_MATCH_OP_TYPE_ATOMIC_WRITE,
                                          desc.GetCacheOpMaxCount(CELL_MATCH_OP_TYPE_ATOMIC_WRITE), desc);
    EXPECT_EQ(retrieved, taskId);

    CellMatchAddOpId(cellMatchTable, cellMemBase, taskId, desc.GetCacheOpMaxCount(CELL_MATCH_OP_TYPE_READ),
                     CELL_MATCH_OP_TYPE_READ, desc);
    retrieved = CellMatchGetOpId(cellMatchTable, cellMemBase, CELL_MATCH_OP_TYPE_READ,
                                 desc.GetCacheOpMaxCount(CELL_MATCH_OP_TYPE_READ), desc);
    EXPECT_EQ(retrieved, taskId);
}

TEST_F(CellMatchDynamicTest, ZeroCapacityDesc)
{
    DevCellMatchTableDesc zeroDesc;
    zeroDesc.cacheOpMaxCount[CELL_MATCH_OP_TYPE_ATOMIC_WRITE] = 0;
    zeroDesc.cacheOpMaxCount[CELL_MATCH_OP_TYPE_NORMAL_WRITE] = 1;
    zeroDesc.cacheOpMaxCount[CELL_MATCH_OP_TYPE_READ] = 0;
    zeroDesc.UpdateCellMemLayOut();

    EXPECT_EQ(zeroDesc.cellUint64Size, 2);

    uint64_t cellMatchTable[100];
    (void)memset_s(cellMatchTable, sizeof(cellMatchTable), 0, sizeof(cellMatchTable));
    uint64_t cellMemBase = 0;

    uint64_t taskId = (1ULL << CELL_MATCH_META_TAGID_SHIFT32) | MakeTaskID(2, 10);

    CellMatchAddOpId(cellMatchTable, cellMemBase, taskId, 0, CELL_MATCH_OP_TYPE_ATOMIC_WRITE, zeroDesc);
    uint64_t retrieved = CellMatchGetOpId(cellMatchTable, cellMemBase, CELL_MATCH_OP_TYPE_ATOMIC_WRITE, 0, zeroDesc);
    EXPECT_EQ(retrieved, taskId);

    CellMatchAddOpId(cellMatchTable, cellMemBase, taskId, 0, CELL_MATCH_OP_TYPE_READ, zeroDesc);
    retrieved = CellMatchGetOpId(cellMatchTable, cellMemBase, CELL_MATCH_OP_TYPE_READ, 0, zeroDesc);
    EXPECT_EQ(retrieved, taskId);

    CellMatchAddOpId(cellMatchTable, cellMemBase, taskId, 0, CELL_MATCH_OP_TYPE_NORMAL_WRITE, zeroDesc);
    retrieved = CellMatchGetOpId(cellMatchTable, cellMemBase, CELL_MATCH_OP_TYPE_NORMAL_WRITE, 0, zeroDesc);
    EXPECT_NE(retrieved, AICORE_TASK_INIT);
}
