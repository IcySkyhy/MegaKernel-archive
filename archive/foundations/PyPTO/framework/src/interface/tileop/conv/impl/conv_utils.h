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
 * \file conv_utils.h
 * \brief Common constants, structs and helper functions for conv tile operations
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_UTILS__H
#define TILEOP_TILE_OPERATOR_CONV_UTILS__H
#include <limits.h>
#include "utils/layout.h"
#include "utils/tile_tensor.h"

constexpr int16_t SHAPE_DIM5 = 5;
constexpr int16_t CONV_IDX_0 = 0;
constexpr int16_t CONV_IDX_1 = 1;
constexpr int16_t CONV_IDX_2 = 2;
constexpr int16_t CONV_IDX_3 = 3;
constexpr int16_t CONV_IDX_4 = 4;
constexpr int16_t MKN_N_VALUE = 16;
constexpr uint16_t NUM0 = 0;
constexpr uint16_t NUM1 = 1;

struct ShapeInfo {
    int64_t shape0 = 0;
    int64_t shape1 = 0;
    int64_t shape2 = 0;
    int64_t shape3 = 0;
    int64_t shape4 = 0;
};

struct OffsetInfo {
    int64_t offset0 = 0;
    int64_t offset1 = 0;
    int64_t offset2 = 0;
    int64_t offset3 = 0;
    int64_t offset4 = 0;
};

template <int16_t idx, typename U>
INLINE int64_t GetConvShape(const U& tileTensor)
{
    static_assert(idx < SHAPE_DIM5, "Idx should be less than 5");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetShapeDim<idx>();
}

template <int16_t idx, typename U>
INLINE int64_t GetConvStride(const U& tileTensor)
{
    static_assert(idx < SHAPE_DIM5, "Idx should be less than 5");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetStrideDim<idx>();
}

template <bool isConv3D, typename U, int64_t elements, int64_t c0Size>
using select_srcTensor = std::conditional_t<
    isConv3D,
    pto::ConvTile<pto::TileType::Mat, typename U::Type, elements * c0Size * sizeof(typename U::Type),
                  pto::Layout::NDC1HWC0, pto::ConvTileShape<-1, -1, -1, -1, -1, c0Size>>,
    pto::ConvTile<pto::TileType::Mat, typename U::Type, elements * sizeof(typename U::Type), pto::Layout::NC1HWC0,
                  pto::ConvTileShape<-1, -1, -1, -1, -1>>>;

#endif // TILEOP_TILE_OPERATOR_CONV_UTILS__H
