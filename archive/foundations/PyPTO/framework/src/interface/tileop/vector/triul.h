/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file triul.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_TRIUL__H
#define TILEOP_TILE_OPERATOR_TRIUL__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_TRIUL TTriUL
template <int isUpper, typename DstTensor, typename SrcTensor>
TILEOP void TTriUL(DstTensor dst, SrcTensor src, int diagonal)
{
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<DstTensor, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<DstTensor, 4, 5>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<SrcTensor, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<SrcTensor, 4, 5>();
    auto srcShape3 = srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto srcShape4 = srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    using dstTileDefine = pto::Tile<pto::TileType::Vec, typename DstTensor::Type, dstTileH, dstTileW,
                                    pto::BLayout::RowMajor, -1, -1>;
    using srcTileDefine = pto::Tile<pto::TileType::Vec, typename SrcTensor::Type, srcTileH, srcTileW,
                                    pto::BLayout::RowMajor, -1, -1>;
    dstTileDefine dstTile(shape3, shape4);
    srcTileDefine srcTile(srcShape3, srcShape4);
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() +
                                                 GenTileOffset(dst, tileOffsets) * sizeof(typename DstTensor::Type)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() +
                                                 GenTileOffset(src, tileOffsets) * sizeof(typename SrcTensor::Type)));
                pto::TTRI<dstTileDefine, isUpper>(dstTile, diagonal);
#ifdef __DAV_V220
                pipe_barrier(PIPE_V);
#endif
                pto::TMUL(dstTile, dstTile, srcTile);
            }
        }
    }
}

#endif
