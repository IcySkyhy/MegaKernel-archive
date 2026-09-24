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
 * \file test_conv_vec_tile_inference.cpp
 * \brief Unit test for ConvVecTileInference.
 */

#include <gtest/gtest.h>
#include <vector>
#include "tilefwk/tile_shape.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/conv/conv_vec_tile_inference.h"

namespace npu::tile_fwk {

class TestConvVecTileInference : public ::testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(TestConvVecTileInference, InferConv1d_FP16)
{
    ConvTile convTile;
    convTile.tileL1Info = Conv::TileL1Info(1, 1, 64, 64, 16, 16, 16, 1);
    convTile.tileL0Info = Conv::TileL0Info(1, 64, 16, 16);
    convTile.setL0Tile = true;

    std::vector<int64_t> oriFmapShape = {1, 32, 64};
    std::vector<int64_t> oriWeightShape = {32, 32, 3};
    auto result = Conv::InferConvVecTileShapes(convTile, DT_FP16, oriFmapShape, oriWeightShape, false, true, 1);

    EXPECT_FALSE(result.fmapVecTile.tile.empty());
    EXPECT_FALSE(result.weightVecTile.tile.empty());
    EXPECT_FALSE(result.outVecTile.tile.empty());

    int64_t c0 = 16;
    EXPECT_EQ(result.fmapVecTile.tile[0], 1);
    EXPECT_EQ(result.fmapVecTile.tile[1], c0);
    EXPECT_EQ(result.weightVecTile.tile[0], 16);
    EXPECT_EQ(result.weightVecTile.tile[1], c0);
    EXPECT_EQ(result.outVecTile.tile.back(), c0);
}

TEST_F(TestConvVecTileInference, InferConv2d_FP16)
{
    ConvTile convTile;
    convTile.tileL1Info = Conv::TileL1Info(2, 2, 64, 64, 16, 16, 16, 1);
    convTile.tileL0Info = Conv::TileL0Info(2, 64, 16, 16);
    convTile.setL0Tile = true;

    std::vector<int64_t> oriFmapShape = {1, 16, 2, 64};
    std::vector<int64_t> oriWeightShape = {32, 16, 3, 3};
    auto result = Conv::InferConvVecTileShapes(convTile, DT_FP16, oriFmapShape, oriWeightShape, false, false, 1);

    EXPECT_FALSE(result.fmapVecTile.tile.empty());
    EXPECT_FALSE(result.weightVecTile.tile.empty());
    EXPECT_FALSE(result.outVecTile.tile.empty());

    int64_t c0 = 16;
    EXPECT_EQ(result.fmapVecTile.tile.size(), 4u);
    EXPECT_EQ(result.fmapVecTile.tile[0], 1);
    EXPECT_EQ(result.fmapVecTile.tile[1], c0);
    EXPECT_EQ(result.weightVecTile.tile.size(), 4u);
    EXPECT_EQ(result.weightVecTile.tile[0], 16);
    EXPECT_EQ(result.outVecTile.tile.size(), 5u);
    EXPECT_EQ(result.outVecTile.tile.back(), c0);
}

TEST_F(TestConvVecTileInference, InferConv2d_FP32)
{
    ConvTile convTile;
    convTile.tileL1Info = Conv::TileL1Info(2, 2, 64, 64, 8, 8, 16, 1);
    convTile.tileL0Info = Conv::TileL0Info(2, 64, 8, 16);
    convTile.setL0Tile = true;

    std::vector<int64_t> oriFmapShape = {1, 8, 2, 64};
    std::vector<int64_t> oriWeightShape = {32, 8, 3, 3};
    auto result = Conv::InferConvVecTileShapes(convTile, DT_FP32, oriFmapShape, oriWeightShape, false, false, 1);

    EXPECT_FALSE(result.fmapVecTile.tile.empty());
    EXPECT_FALSE(result.weightVecTile.tile.empty());
    EXPECT_FALSE(result.outVecTile.tile.empty());

    int64_t c0 = 8;
    EXPECT_EQ(result.fmapVecTile.tile[0], 1);
    EXPECT_EQ(result.fmapVecTile.tile[1], c0);
    EXPECT_EQ(result.outVecTile.tile.back(), c0);
}

TEST_F(TestConvVecTileInference, InferConv2d_BF16_Groups)
{
    ConvTile convTile;
    convTile.tileL1Info = Conv::TileL1Info(2, 2, 64, 64, 16, 16, 16, 1);
    convTile.tileL0Info = Conv::TileL0Info(2, 64, 16, 16);
    convTile.setL0Tile = true;

    std::vector<int64_t> oriFmapShape = {1, 32, 2, 64};
    std::vector<int64_t> oriWeightShape = {32, 16, 3, 3};
    auto result = Conv::InferConvVecTileShapes(convTile, DT_BF16, oriFmapShape, oriWeightShape, false, false, 2);

    EXPECT_FALSE(result.fmapVecTile.tile.empty());
    EXPECT_FALSE(result.weightVecTile.tile.empty());
    EXPECT_FALSE(result.outVecTile.tile.empty());

    int64_t c0 = 16;
    EXPECT_EQ(result.fmapVecTile.tile[0], 1);
    EXPECT_EQ(result.fmapVecTile.tile[1], c0);
    int64_t cinPerGroup = oriFmapShape[1] / 2;
    EXPECT_LE(result.fmapVecTile.tile[1], cinPerGroup);
}

TEST_F(TestConvVecTileInference, InferConv3d_FP16)
{
    ConvTile convTile;
    convTile.tileL1Info = Conv::TileL1Info(2, 2, 64, 64, 16, 16, 16, 1);
    convTile.tileL0Info = Conv::TileL0Info(2, 64, 16, 16);
    convTile.setL0Tile = true;

    std::vector<int64_t> oriFmapShape = {1, 32, 2, 2, 64};
    std::vector<int64_t> oriWeightShape = {32, 32, 2, 3, 3};
    auto result = Conv::InferConvVecTileShapes(convTile, DT_FP16, oriFmapShape, oriWeightShape, true, false, 1);

    EXPECT_FALSE(result.fmapVecTile.tile.empty());
    EXPECT_FALSE(result.weightVecTile.tile.empty());
    EXPECT_FALSE(result.outVecTile.tile.empty());

    int64_t c0 = 16;
    EXPECT_EQ(result.fmapVecTile.tile.size(), 5u);
    EXPECT_EQ(result.fmapVecTile.tile[0], 1);
    EXPECT_EQ(result.fmapVecTile.tile[1], c0);
    EXPECT_EQ(result.weightVecTile.tile.size(), 5u);
    EXPECT_EQ(result.outVecTile.tile.size(), 6u);
    EXPECT_EQ(result.outVecTile.tile.back(), c0);
}

TEST_F(TestConvVecTileInference, InferEmptyTileL1Wout)
{
    ConvTile convTile;
    convTile.tileL1Info = Conv::TileL1Info(0, 0, 0, 0, 0, 0, 0, 0);
    convTile.tileL0Info = Conv::TileL0Info(0, 0, 0, 0);
    convTile.setL0Tile = true;

    std::vector<int64_t> oriFmapShape = {1, 16, 2, 64};
    std::vector<int64_t> oriWeightShape = {32, 16, 3, 3};
    auto result = Conv::InferConvVecTileShapes(convTile, DT_FP16, oriFmapShape, oriWeightShape, false, false, 1);

    EXPECT_TRUE(result.fmapVecTile.tile.empty());
    EXPECT_TRUE(result.weightVecTile.tile.empty());
    EXPECT_TRUE(result.outVecTile.tile.empty());
}

TEST_F(TestConvVecTileInference, InferEmptyShapes)
{
    ConvTile convTile;
    convTile.tileL1Info = Conv::TileL1Info(2, 2, 64, 64, 16, 16, 16, 1);
    convTile.tileL0Info = Conv::TileL0Info(2, 64, 16, 16);
    convTile.setL0Tile = true;

    std::vector<int64_t> emptyShape;
    std::vector<int64_t> oriWeightShape = {32, 16, 3, 3};
    auto result = Conv::InferConvVecTileShapes(convTile, DT_FP16, emptyShape, oriWeightShape, false, false, 1);

    EXPECT_TRUE(result.fmapVecTile.tile.empty());
    EXPECT_TRUE(result.weightVecTile.tile.empty());
    EXPECT_TRUE(result.outVecTile.tile.empty());
}

TEST_F(TestConvVecTileInference, SelectConvVecTile)
{
    Conv::ConvVecTileShapes vecTiles;
    vecTiles.fmapVecTile.tile = {1, 16, 2, 16};
    vecTiles.weightVecTile.tile = {16, 16, 3, 16};
    vecTiles.outVecTile.tile = {1, 1, 2, 64, 16};

    auto ndTile = Conv::SelectConvVecTile(vecTiles, TileOpFormat::TILEOP_ND);
    EXPECT_EQ(ndTile.tile, vecTiles.outVecTile.tile);

    auto nc1hwc0Tile = Conv::SelectConvVecTile(vecTiles, TileOpFormat::TILEOP_NC1HWC0);
    EXPECT_EQ(nc1hwc0Tile.tile, vecTiles.fmapVecTile.tile);

    auto ndc1hwc0Tile = Conv::SelectConvVecTile(vecTiles, TileOpFormat::TILEOP_NDC1HWC0);
    EXPECT_EQ(ndc1hwc0Tile.tile, vecTiles.fmapVecTile.tile);

    auto fractalZTile = Conv::SelectConvVecTile(vecTiles, TileOpFormat::TILEOP_FRACTAL_Z);
    EXPECT_EQ(fractalZTile.tile, vecTiles.weightVecTile.tile);
}

TEST_F(TestConvVecTileInference, GetReshapeVecTile_5D_to_4D)
{
    VecTile srcTile;
    srcTile.tile = {1, 16, 2, 4, 16};
    auto result = Conv::GetReshapeVecTile(srcTile, false);

    EXPECT_EQ(result.tile.size(), 4u);
    EXPECT_EQ(result.tile[0], 1);
    EXPECT_EQ(result.tile[1], 256);
    EXPECT_EQ(result.tile[2], 2);
    EXPECT_EQ(result.tile[3], 4);
}

TEST_F(TestConvVecTileInference, GetReshapeVecTile_Conv1d)
{
    VecTile srcTile;
    srcTile.tile = {1, 256, 2, 4};
    auto result = Conv::GetReshapeVecTile(srcTile, true);

    EXPECT_EQ(result.tile.size(), 3u);
    EXPECT_EQ(result.tile[0], 1);
    EXPECT_EQ(result.tile[1], 256);
    EXPECT_EQ(result.tile[2], 4);
}

TEST_F(TestConvVecTileInference, GetReshapeVecTile_4D_NoChange)
{
    VecTile srcTile;
    srcTile.tile = {1, 16, 2, 16};
    auto result = Conv::GetReshapeVecTile(srcTile, false);

    EXPECT_EQ(result.tile.size(), 4u);
    EXPECT_EQ(result.tile, srcTile.tile);
}

TEST_F(TestConvVecTileInference, InferConv2d_FP16_VecTileBytesWithinUbLimit)
{
    ConvTile convTile;
    convTile.tileL1Info = Conv::TileL1Info(2, 2, 64, 64, 16, 16, 16, 1);
    convTile.tileL0Info = Conv::TileL0Info(2, 64, 16, 16);
    convTile.setL0Tile = true;

    std::vector<int64_t> oriFmapShape = {1, 16, 2, 64};
    std::vector<int64_t> oriWeightShape = {32, 16, 3, 3};
    auto result = Conv::InferConvVecTileShapes(convTile, DT_FP16, oriFmapShape, oriWeightShape, false, false, 1);

    int64_t dtypeSize = 2;
    auto checkUbLimit = [&](const VecTile& tile) {
        int64_t total = dtypeSize;
        for (int64_t v : tile.tile) {
            total *= v;
        }
        size_t ubLimit = Platform::Instance().GetAIVCore().GetMemorySize(MemoryType::MEM_UB) / 4;
        EXPECT_LE(total, static_cast<int64_t>(ubLimit));
    };

    checkUbLimit(result.fmapVecTile);
    checkUbLimit(result.weightVecTile);
    checkUbLimit(result.outVecTile);
}

} // namespace npu::tile_fwk
