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
 * \file test_parallel_sort.cpp
 * \brief
 */
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/inner/tilefwk.h"

using namespace npu::tile_fwk;

class ParallelSortUTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

struct SortParams {
    int length;
    int tileSize;
    int descending;
};

template <typename T = float, typename idxT = int>
void SortTest(const SortParams& params)
{
    int length = params.length;
    int tileSize = params.tileSize;
    int descending = params.descending;

    DataType dType = DT_FP32;
    DataType idxDType = DT_INT32;
    std::vector<int64_t> shape = {1, length};

    Tensor x(dType, shape, "x");
    Tensor y(dType, shape, "y");
    Tensor yIdx(idxDType, shape, "yIdx");

    config::SetBuildStatic(true);
    FUNCTION("Sort", {x, y, yIdx})
    {
        TileShape::Current().SetVecTile({1, tileSize});
        std::tie(y, yIdx) = Sort(x, descending);
    }
}

template <typename T = float, typename idxT = int>
void SortWithIndexTest(const SortParams& params)
{
    int length = params.length;
    int tileSize = params.tileSize;
    int descending = params.descending;

    DataType dType = DT_FP32;
    DataType idxDType = DT_INT32;
    std::vector<int64_t> shape = {1, length};

    Tensor x(dType, shape, "x");
    Tensor idx(idxDType, shape, "idx");
    Tensor y(dType, shape, "y");
    Tensor yIdx(idxDType, shape, "yIdx");

    config::SetBuildStatic(true);
    FUNCTION("Sort", {x, idx, y, yIdx})
    {
        TileShape::Current().SetVecTile({1, tileSize});
        std::tie(y, yIdx) = SortWithIndex(x, idx, descending);
    }
}

TEST_F(ParallelSortUTest, fp32_64k_8k)
{
    SortParams params = {1024 * 64, 1024 * 8, true};
    SortTest(params);
}

TEST_F(ParallelSortUTest, withindex_fp32_64k_8k)
{
    SortParams params = {1024 * 64, 1024 * 8, true};
    SortWithIndexTest(params);
}
