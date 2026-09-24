/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <string>

#include "machine/utils/dynamic/dev_encode_function_stitch.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DevEncodeFunctionStitchTest : public testing::Test {};

TEST_F(DevEncodeFunctionStitchTest, Stitch_InitWithNext_SetsSizeZero)
{
    DevAscendFunctionDuppedStitch stitch;
    stitch.InitWithNext(nullptr);
    EXPECT_EQ(stitch.Size(), 0u);
    EXPECT_EQ(stitch.Next(), nullptr);
}

TEST_F(DevEncodeFunctionStitchTest, Stitch_PushBack_IncreasesSize)
{
    DevAscendFunctionDuppedStitch stitch;
    stitch.InitWithNext(nullptr);
    stitch.SafePushBack(100);
    EXPECT_EQ(stitch.Size(), 1u);
    EXPECT_EQ(stitch.At(0), 100u);
    stitch.SafePushBack(200);
    EXPECT_EQ(stitch.Size(), 2u);
    EXPECT_EQ(stitch.At(1), 200u);
}

TEST_F(DevEncodeFunctionStitchTest, Stitch_ForEach_VisitsAll)
{
    DevAscendFunctionDuppedStitch stitch;
    stitch.InitWithNext(nullptr);
    stitch.SafePushBack(10);
    stitch.SafePushBack(20);
    stitch.SafePushBack(30);
    std::vector<uint32_t> visited;
    stitch.ForEach([&](uint32_t id) { visited.push_back(id); });
    EXPECT_EQ(visited.size(), 3u);
    EXPECT_EQ(visited[0], 10u);
    EXPECT_EQ(visited[1], 20u);
    EXPECT_EQ(visited[2], 30u);
}

TEST_F(DevEncodeFunctionStitchTest, StitchList_IsNull_WhenEmpty)
{
    DevAscendFunctionDuppedStitchList list;
    EXPECT_TRUE(list.IsNull());
}

TEST_F(DevEncodeFunctionStitchTest, StitchList_PushBack_CreatesHead)
{
    DevAscendFunctionDuppedStitchList list;
    DevAscendFunctionDuppedStitch node;
    node.InitWithNext(nullptr);
    list.PushBack(42, [&]() -> DevAscendFunctionDuppedStitch* { return &node; });
    EXPECT_FALSE(list.IsNull());
    EXPECT_EQ(list.Head()->Size(), 1u);
    EXPECT_EQ(list.Head()->At(0), 42u);
}

TEST_F(DevEncodeFunctionStitchTest, StitchList_ForEach_VisitsAllNodes)
{
    DevAscendFunctionDuppedStitch node1, node2;
    node1.InitWithNext(nullptr);
    node2.InitWithNext(nullptr);

    DevAscendFunctionDuppedStitchList list;
    int allocIdx = 0;
    DevAscendFunctionDuppedStitch* nodes[] = {&node1, &node2};
    list.PushBack(10, [&]() -> DevAscendFunctionDuppedStitch* { return nodes[allocIdx++]; });
    list.PushBack(20, [&]() -> DevAscendFunctionDuppedStitch* { return nodes[allocIdx++]; });

    std::vector<uint32_t> visited;
    list.ForEach([&](uint32_t id) { visited.push_back(id); });
    EXPECT_EQ(visited.size(), 2u);
}

TEST_F(DevEncodeFunctionStitchTest, StitchList_Dump_EmptyList)
{
    DevAscendFunctionDuppedStitchList list;
    std::string dump = list.Dump();
    EXPECT_EQ(dump, "[]");
}

TEST_F(DevEncodeFunctionStitchTest, StitchList_Dump_WithElements)
{
    DevAscendFunctionDuppedStitch node;
    node.InitWithNext(nullptr);

    DevAscendFunctionDuppedStitchList list;
    list.PushBack(MakeTaskID(1, 2), [&]() -> DevAscendFunctionDuppedStitch* { return &node; });
    std::string dump = list.Dump();
    EXPECT_FALSE(dump.empty());
    EXPECT_NE(dump.find("["), std::string::npos);
}

TEST_F(DevEncodeFunctionStitchTest, DumpTask_SingleId)
{
    uint32_t taskId = MakeTaskID(3, 5);
    std::string result = DevAscendFunctionDuppedStitchList::DumpTask(taskId);
    EXPECT_NE(result.find("3"), std::string::npos);
    EXPECT_NE(result.find("5"), std::string::npos);
}

TEST_F(DevEncodeFunctionStitchTest, DumpTask_ArrayWithInit)
{
    uint32_t idx[4] = {MakeTaskID(1, 2), AICORE_TASK_INIT, MakeTaskID(3, 4), AICORE_TASK_INIT};
    std::string result = DevAscendFunctionDuppedStitchList::DumpTask(idx, 4);
    EXPECT_NE(result.find("size = 4"), std::string::npos);
    EXPECT_NE(result.find("[0]="), std::string::npos);
    EXPECT_NE(result.find("[2]="), std::string::npos);
    EXPECT_EQ(result.find("[1]="), std::string::npos);
}

TEST_F(DevEncodeFunctionStitchTest, CellMatchProcessByDim_1D)
{
    DevCellMatchTableDesc desc;
    desc.cellShape.dimSize = 1;
    desc.cellShape.dim[0] = 4;
    desc.stride.dimStride[0] = 1;
    desc.SetStrideShape({4});

    uint64_t rangeBegin[DEV_SHAPE_DIM_MAX] = {0};
    uint64_t rangeEnd[DEV_SHAPE_DIM_MAX] = {3};

    struct TestHandler {
        static uint32_t Process([[maybe_unused]] int index, [[maybe_unused]] const DevCellMatchTableDesc& desc)
        {
            return 0;
        }
    };

    uint32_t ret = CellMatchProcessByDim<TestHandler>(desc.GetDimensionSize(), desc, rangeBegin, rangeEnd);
    EXPECT_EQ(ret, 0u);
}

TEST_F(DevEncodeFunctionStitchTest, CellMatchProcessByDim_2D)
{
    DevCellMatchTableDesc desc;
    desc.cellShape.dimSize = 2;
    desc.cellShape.dim[0] = 2;
    desc.cellShape.dim[1] = 2;
    desc.stride.dimStride[0] = 2;
    desc.stride.dimStride[1] = 1;
    desc.SetStrideShape({2, 2});

    uint64_t rangeBegin[DEV_SHAPE_DIM_MAX] = {0, 0};
    uint64_t rangeEnd[DEV_SHAPE_DIM_MAX] = {1, 1};

    struct TestHandler {
        static uint32_t Process([[maybe_unused]] int index, [[maybe_unused]] const DevCellMatchTableDesc& desc)
        {
            return 0;
        }
    };

    uint32_t ret = CellMatchProcessByDim<TestHandler>(desc.GetDimensionSize(), desc, rangeBegin, rangeEnd);
    EXPECT_EQ(ret, 0u);
}

TEST_F(DevEncodeFunctionStitchTest, CellMatchProcessByDim_3D)
{
    DevCellMatchTableDesc desc;
    desc.cellShape.dimSize = 3;
    desc.cellShape.dim[0] = 2;
    desc.cellShape.dim[1] = 2;
    desc.cellShape.dim[2] = 2;
    desc.stride.dimStride[0] = 4;
    desc.stride.dimStride[1] = 2;
    desc.stride.dimStride[2] = 1;
    desc.SetStrideShape({2, 2, 2});

    uint64_t rangeBegin[DEV_SHAPE_DIM_MAX] = {0, 0, 0};
    uint64_t rangeEnd[DEV_SHAPE_DIM_MAX] = {1, 1, 1};

    struct TestHandler {
        static uint32_t Process([[maybe_unused]] int index, [[maybe_unused]] const DevCellMatchTableDesc& desc)
        {
            return 0;
        }
    };

    uint32_t ret = CellMatchProcessByDim<TestHandler>(desc.GetDimensionSize(), desc, rangeBegin, rangeEnd);
    EXPECT_EQ(ret, 0u);
}

TEST_F(DevEncodeFunctionStitchTest, CellMatchProcessByDim_4D)
{
    DevCellMatchTableDesc desc;
    desc.cellShape.dimSize = 4;
    desc.cellShape.dim[0] = 2;
    desc.cellShape.dim[1] = 2;
    desc.cellShape.dim[2] = 2;
    desc.cellShape.dim[3] = 2;
    desc.stride.dimStride[0] = 8;
    desc.stride.dimStride[1] = 4;
    desc.stride.dimStride[2] = 2;
    desc.stride.dimStride[3] = 1;
    desc.SetStrideShape({2, 2, 2, 2});

    uint64_t rangeBegin[DEV_SHAPE_DIM_MAX] = {0, 0, 0, 0};
    uint64_t rangeEnd[DEV_SHAPE_DIM_MAX] = {1, 1, 1, 1};

    struct TestHandler {
        static uint32_t Process([[maybe_unused]] int index, [[maybe_unused]] const DevCellMatchTableDesc& desc)
        {
            return 0;
        }
    };

    uint32_t ret = CellMatchProcessByDim<TestHandler>(desc.GetDimensionSize(), desc, rangeBegin, rangeEnd);
    EXPECT_EQ(ret, 0u);
}

TEST_F(DevEncodeFunctionStitchTest, CellMatchProcessByDim_5D)
{
    DevCellMatchTableDesc desc;
    desc.cellShape.dimSize = 5;
    desc.cellShape.dim[0] = 2;
    desc.cellShape.dim[1] = 2;
    desc.cellShape.dim[2] = 2;
    desc.cellShape.dim[3] = 2;
    desc.cellShape.dim[4] = 2;
    desc.stride.dimStride[0] = 16;
    desc.stride.dimStride[1] = 8;
    desc.stride.dimStride[2] = 4;
    desc.stride.dimStride[3] = 2;
    desc.stride.dimStride[4] = 1;
    desc.SetStrideShape({2, 2, 2, 2, 2});

    uint64_t rangeBegin[DEV_SHAPE_DIM_MAX] = {0, 0, 0, 0, 0};
    uint64_t rangeEnd[DEV_SHAPE_DIM_MAX] = {1, 1, 1, 1, 1};

    struct TestHandler {
        static uint32_t Process([[maybe_unused]] int index, [[maybe_unused]] const DevCellMatchTableDesc& desc)
        {
            return 0;
        }
    };

    uint32_t ret = CellMatchProcessByDim<TestHandler>(desc.GetDimensionSize(), desc, rangeBegin, rangeEnd);
    EXPECT_EQ(ret, 0u);
}

TEST_F(DevEncodeFunctionStitchTest, CellMatchProcessByDim_ErrorOnNonZeroReturn)
{
    DevCellMatchTableDesc desc;
    desc.cellShape.dimSize = 1;
    desc.cellShape.dim[0] = 4;
    desc.stride.dimStride[0] = 1;
    desc.SetStrideShape({4});

    uint64_t rangeBegin[DEV_SHAPE_DIM_MAX] = {0};
    uint64_t rangeEnd[DEV_SHAPE_DIM_MAX] = {3};

    struct ErrorHandler {
        static uint32_t Process([[maybe_unused]] int index, [[maybe_unused]] const DevCellMatchTableDesc& desc)
        {
            return 42;
        }
    };

    uint32_t ret = CellMatchProcessByDim<ErrorHandler>(desc.GetDimensionSize(), desc, rangeBegin, rangeEnd);
    EXPECT_EQ(ret, 42u);
}

TEST_F(DevEncodeFunctionStitchTest, DevAscendProgramUpdate_Empty)
{
    DevAscendProgramUpdate update;
    EXPECT_TRUE(update.Empty());
}

TEST_F(DevEncodeFunctionStitchTest, DUPPED_STITCH_SIZE_CorrectValue)
{
    EXPECT_EQ(DUPPED_STITCH_SIZE, 0x10u - (sizeof(void*) / sizeof(uint32_t)) - 0x1u);
}
