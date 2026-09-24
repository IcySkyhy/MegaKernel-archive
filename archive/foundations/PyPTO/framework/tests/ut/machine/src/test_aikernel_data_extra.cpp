/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include "interface/machine/device/tilefwk/aikernel_data.h"

using namespace npu::tile_fwk;

class AikernelDataExtraTest : public testing::Test {};

TEST_F(AikernelDataExtraTest, DevShape_Equal_AllCases)
{
    DevShape a{};
    a.dimSize = 3;
    a.dim[0] = 2;
    a.dim[1] = 3;
    a.dim[2] = 4;

    DevShape b{};
    b.dimSize = 3;
    b.dim[0] = 2;
    b.dim[1] = 3;
    b.dim[2] = 4;
    EXPECT_TRUE(a.Equal(b));

    DevShape c{};
    c.dimSize = 2;
    c.dim[0] = 2;
    c.dim[1] = 3;
    EXPECT_FALSE(a.Equal(c));

    DevShape d{};
    d.dimSize = 3;
    d.dim[0] = 2;
    d.dim[1] = 5;
    d.dim[2] = 4;
    EXPECT_FALSE(a.Equal(d));

    DevShape e{};
    e.dimSize = 0;
    DevShape f{};
    f.dimSize = 0;
    EXPECT_TRUE(e.Equal(f));
}

TEST_F(AikernelDataExtraTest, DevShape_GetSize_AllCases)
{
    DevShape s{};
    s.dimSize = 3;
    s.dim[0] = 2;
    s.dim[1] = 3;
    s.dim[2] = 4;
    EXPECT_EQ(s.GetSize(), 24);

    DevShape empty{};
    empty.dimSize = 0;
    EXPECT_EQ(empty.GetSize(), 1);
}

TEST_F(AikernelDataExtraTest, TaskIdBitFields_AllCases)
{
    uint32_t taskId = (5u << TASKID_TASK_BITS) | 10u;
    EXPECT_EQ(FuncID(taskId), 5u);
    EXPECT_EQ(TaskID(taskId), 10u);

    uint32_t combined = MakeTaskID(7, 42);
    EXPECT_EQ(TaskID(combined), 42u);
    EXPECT_EQ(FuncID(combined), 7u);

    uint32_t parallelTask = (3u << (TASKID_TASK_BITS + TASKID_FUNC_BITS)) | (5u << TASKID_TASK_BITS) | 10u;
    EXPECT_EQ(ParallelIndex(parallelTask), 3u);

    uint32_t dcciTask = (1u << (TASKID_TASK_BITS + TASKID_FUNC_BITS + TASKID_PARALLEL_INDEX_BITS));
    EXPECT_EQ(DevTaskDcciFlag(dcciTask), 1u);
    EXPECT_EQ(DevTaskDcciFlag(0u), 0u);
}

TEST_F(AikernelDataExtraTest, HighRegBitFields_AllCases)
{
    uint64_t highRegValue = 0xABCD123456ULL;
    uint32_t devTaskId = DevTaskId(highRegValue);
    EXPECT_EQ(devTaskId, highRegValue & REG_VAL_DEVTASK_ID_MASK);

    uint64_t highRegWithFlag = (0xFFULL << REG_VAL_DEVTASK_ID_BITS) | 0x123456ULL;
    uint32_t flag = ParallelDevTaskModifyFlag(highRegWithFlag);
    EXPECT_EQ(flag, 0xFFu);

    uint64_t highRegNoFlag = 0x123456ULL;
    EXPECT_EQ(ParallelDevTaskModifyFlag(highRegNoFlag), 0u);
}

TEST_F(AikernelDataExtraTest, Constants_AllValues)
{
    EXPECT_EQ(SCH_DEVTASK_MAX_PARALLELISM, (1u << TASKID_PARALLEL_INDEX_BITS));
    EXPECT_EQ(SCH_DEVTASK_MAX_PARALLELISM, 8u);

    uint32_t expected = (1u << (TASKID_TASK_BITS + TASKID_FUNC_BITS)) - 1;
    EXPECT_EQ(TASKID_FROM_CTRL_TOPO_MASK, expected);
}

TEST_F(AikernelDataExtraTest, DynFuncStructs_AllCases)
{
    DynFuncBin bin{};
    EXPECT_EQ(bin.coreType, 0u);
    EXPECT_EQ(bin.psgId, 0u);
    EXPECT_EQ(bin.funcHash, 0u);
    EXPECT_EQ(bin.wrapVecId, -1);
    EXPECT_EQ(bin.mixResourceType, 0);

    DynFuncHeader header{};
    header.seqNo = 42;
    header.funcNum = 5;
    EXPECT_EQ(header.GetIndex(), 42u);
    EXPECT_EQ(header.Size(), 5u);
}
