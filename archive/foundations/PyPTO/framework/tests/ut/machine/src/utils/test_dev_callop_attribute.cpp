/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#define private public
#include "machine/utils/dynamic/dev_callop_attribute.h"
#undef private

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(DevCallOpAttributeTest, CheckOffsetAndValidShapeInRawShape_NoClamp)
{
    uint64_t offset[DEV_SHAPE_DIM_MAX] = {0, 0, 0, 0, 0};
    uint64_t validShape[DEV_SHAPE_DIM_MAX] = {4, 8, 0, 0, 0};
    uint64_t rawShape[DEV_SHAPE_DIM_MAX] = {16, 32, 0, 0, 0};
    bool result = CheckOffsetAndValidShapeInRawShape(offset, validShape, rawShape, 2);
    EXPECT_FALSE(result);
    EXPECT_EQ(offset[0], 0u);
    EXPECT_EQ(validShape[0], 4u);
}

TEST(DevCallOpAttributeTest, CheckOffsetAndValidShapeInRawShape_OffsetOutOfRange)
{
    uint64_t offset[DEV_SHAPE_DIM_MAX] = {20, 0, 0, 0, 0};
    uint64_t validShape[DEV_SHAPE_DIM_MAX] = {4, 8, 0, 0, 0};
    uint64_t rawShape[DEV_SHAPE_DIM_MAX] = {16, 32, 0, 0, 0};
    bool result = CheckOffsetAndValidShapeInRawShape(offset, validShape, rawShape, 2);
    EXPECT_TRUE(result);
    EXPECT_EQ(offset[0], 16u);
    EXPECT_EQ(validShape[0], 0u);
}

TEST(DevCallOpAttributeTest, CheckOffsetAndValidShapeInRawShape_ValidShapeOutOfRange)
{
    uint64_t offset[DEV_SHAPE_DIM_MAX] = {10, 0, 0, 0, 0};
    uint64_t validShape[DEV_SHAPE_DIM_MAX] = {20, 8, 0, 0, 0};
    uint64_t rawShape[DEV_SHAPE_DIM_MAX] = {16, 32, 0, 0, 0};
    bool result = CheckOffsetAndValidShapeInRawShape(offset, validShape, rawShape, 2);
    EXPECT_TRUE(result);
    EXPECT_EQ(validShape[0], 6u);
}

TEST(DevCallOpAttributeTest, CheckOffsetAndValidShapeInRawShape_ZeroValidShape)
{
    uint64_t offset[DEV_SHAPE_DIM_MAX] = {0, 0, 0, 0, 0};
    uint64_t validShape[DEV_SHAPE_DIM_MAX] = {0, 8, 0, 0, 0};
    uint64_t rawShape[DEV_SHAPE_DIM_MAX] = {16, 32, 0, 0, 0};
    bool result = CheckOffsetAndValidShapeInRawShape(offset, validShape, rawShape, 2);
    EXPECT_FALSE(result);
}

TEST(DevCallOpAttributeTest, DumpCellMatchAccessRange_NoCrash)
{
    DevCellMatchTableDesc desc{};
    desc.SetCellShape({4, 8});
    desc.SetStrideShape({4, 8});
    uint64_t offset[DEV_SHAPE_DIM_MAX] = {0, 0, 0, 0, 0};
    uint64_t validShape[DEV_SHAPE_DIM_MAX] = {4, 8, 0, 0, 0};
    uint64_t rawShape[DEV_SHAPE_DIM_MAX] = {16, 32, 0, 0, 0};
    DumpCellMatchAccessRange(1, 0, offset, validShape, rawShape, desc);
}

TEST(DevCallOpAttributeTest, DumpCellMatchAccessRange_LargeDim)
{
    DevCellMatchTableDesc desc{};
    desc.SetCellShape({4, 8, 16, 32, 64});
    desc.SetStrideShape({4, 8, 16, 32, 64});
    uint64_t offset[DEV_SHAPE_DIM_MAX] = {1, 2, 3, 4, 5};
    uint64_t validShape[DEV_SHAPE_DIM_MAX] = {10, 20, 30, 40, 50};
    uint64_t rawShape[DEV_SHAPE_DIM_MAX] = {100, 200, 300, 400, 500};
    DumpCellMatchAccessRange(2, 3, offset, validShape, rawShape, desc);
}
