/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dev_cell_match_dump.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include "machine/utils/dynamic/dev_cell_match_dump.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"

using namespace npu::tile_fwk::dynamic;

namespace {
DevCellMatchTableDesc MakeDesc(uint32_t normalWriteCnt = 1, uint32_t atomicWriteCnt = 0, uint32_t readCnt = 0)
{
    DevCellMatchTableDesc desc;
    desc.SetCellShape({1});
    desc.SetStrideShape({1});
    desc.SetCacheOpMaxCount({normalWriteCnt, atomicWriteCnt, readCnt});
    return desc;
}

uint64_t BuildMeta(uint32_t curType, uint32_t curCnt, uint32_t prevType, uint32_t prevCnt, uint64_t tagId)
{
    uint64_t meta = 0;
    CellMatchSetCurrentOpType(meta, curType);
    CellMatchSetCurrentOpCount(meta, curCnt);
    CellMatchSetPrevMutexOpType(meta, prevType);
    CellMatchSetPrevMutexOpCount(meta, prevCnt);
    CellMatchSetTagId(meta, tagId);
    return meta;
}
} // namespace

TEST(DevCellMatchDumpTest, NullData_ReturnsEmpty)
{
    DevCellMatchTableDesc desc = MakeDesc();
    auto result = DumpCellMatchPartialUpdateTable(nullptr, 100, desc);
    EXPECT_TRUE(result.empty());
}

TEST(DevCellMatchDumpTest, ZeroCellUint64Size_ReturnsEmpty)
{
    DevCellMatchTableDesc desc = MakeDesc();
    desc.cellUint64Size = 0;
    std::vector<uint64_t> data(10, 0);
    auto result = DumpCellMatchPartialUpdateTable(data.data(), data.size(), desc);
    EXPECT_TRUE(result.empty());
}

TEST(DevCellMatchDumpTest, ZeroDataSize_ReturnsEmpty)
{
    DevCellMatchTableDesc desc = MakeDesc();
    std::vector<uint64_t> data(10, 0);
    auto result = DumpCellMatchPartialUpdateTable(data.data(), 0, desc);
    EXPECT_TRUE(result.empty());
}

TEST(DevCellMatchDumpTest, SingleCellNormalWrite)
{
    DevCellMatchTableDesc desc = MakeDesc(1, 0, 0);
    std::vector<uint64_t> data(desc.cellUint64Size, 0);
    data[0] = BuildMeta(CELL_MATCH_OP_TYPE_NORMAL_WRITE, 1, CELL_MATCH_OP_TYPE_NONE, CELL_MATCH_INVALID_OP_COUNT, 42);
    data[1] = 100;
    auto result = DumpCellMatchPartialUpdateTable(data.data(), data.size(), desc);
    EXPECT_FALSE(result.empty());
    EXPECT_NE(result.find("cell[0]"), std::string::npos);
    EXPECT_NE(result.find("NW"), std::string::npos);
}

TEST(DevCellMatchDumpTest, MultipleOpsWithAtomicAndRead)
{
    DevCellMatchTableDesc desc = MakeDesc(1, 2, 1);
    std::vector<uint64_t> data(desc.cellUint64Size * 1, 0);
    data[0] = BuildMeta(CELL_MATCH_OP_TYPE_ATOMIC_WRITE, 2, CELL_MATCH_OP_TYPE_READ, 1, 5);
    data[1] = 0;
    data[2] = 200;
    data[3] = 201;
    data[4] = 300;
    auto result = DumpCellMatchPartialUpdateTable(data.data(), data.size(), desc);
    EXPECT_FALSE(result.empty());
    EXPECT_NE(result.find("AW"), std::string::npos);
    EXPECT_NE(result.find("RD"), std::string::npos);
}

TEST(DevCellMatchDumpTest, InvalidOpType_ShowsQuestionMark)
{
    DevCellMatchTableDesc desc = MakeDesc(1, 0, 0);
    std::vector<uint64_t> data(desc.cellUint64Size, 0);
    data[0] = BuildMeta(99, 0, 88, 0, 0);
    auto result = DumpCellMatchPartialUpdateTable(data.data(), data.size(), desc);
    EXPECT_FALSE(result.empty());
    EXPECT_NE(result.find("?"), std::string::npos);
}

TEST(DevCellMatchDumpTest, PrevCntInvalid_ShowsInvalid)
{
    DevCellMatchTableDesc desc = MakeDesc(1, 0, 0);
    std::vector<uint64_t> data(desc.cellUint64Size, 0);
    data[0] = BuildMeta(CELL_MATCH_OP_TYPE_NORMAL_WRITE, 1, CELL_MATCH_OP_TYPE_NONE, CELL_MATCH_INVALID_OP_COUNT, 0);
    data[1] = 1;
    auto result = DumpCellMatchPartialUpdateTable(data.data(), data.size(), desc);
    EXPECT_NE(result.find("prevCnt=INVALID"), std::string::npos);
}
