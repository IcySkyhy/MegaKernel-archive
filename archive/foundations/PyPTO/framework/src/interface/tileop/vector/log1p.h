/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILEOP_TILE_OPERATOR_LOG1P__H
#define TILEOP_TILE_OPERATOR_LOG1P__H

#include <type_traits>

#include "pto_tile.h"
#include "unary.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename TDst, typename TSrc, typename TTmp0, typename TTmp1, typename TTmp2, typename TMask>
TILEOP void TLog1pTileImpl(TDst dstTile, TSrc srcTile, TTmp0 tmp0Tile, TTmp1 tmp1Tile, TTmp2 tmp2Tile, TMask maskTile)
{
    constexpr float scalarOne = 1.0f;
    constexpr float scalarNegOne = -1.0f;
    const float scalarInf = INFINITY;

    // tmp0 = x + 1
    pto::TADDS(tmp0Tile, srcTile, scalarOne);
    SyncV();
    // tmp1 = (x + 1) - 1
    pto::TADDS(tmp1Tile, tmp0Tile, scalarNegOne);
    SyncV();
    // tmp2 = x / ((x + 1) - 1)
    pto::TDIV(tmp2Tile, srcTile, tmp1Tile);
    // dst = log(x + 1)
    pto::TLOG(dstTile, tmp0Tile);
    SyncV();
    // dst = log(x + 1) * x / ((x + 1) - 1)
    pto::TMUL(dstTile, dstTile, tmp2Tile);
    // If x + 1 == 1, return x.
    pto::TCMPS(maskTile, tmp0Tile, scalarOne, pto::CmpMode::EQ);
    SyncV();
    pto::TSEL(dstTile, maskTile, srcTile, dstTile, tmp1Tile);
    SyncV();

    // If x + 1 == inf, return inf.
    pto::TCMPS(maskTile, tmp0Tile, scalarInf, pto::CmpMode::EQ);
    SyncV();
    pto::TSEL(dstTile, maskTile, tmp0Tile, dstTile, tmp1Tile);
    SyncV();
}

#define OP_TILE_OP_LOG1P TLog1p
template <typename T0, typename T1, typename T2>
TILEOP void TLog1p(T0 dst, T1 src, T2 tmp)
{
    static_assert(std::is_same_v<typename T0::Type, float> && std::is_same_v<typename T1::Type, float>,
                  "TLog1p only supports fp32 tile input and output.");

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dataTypeSize = sizeof(typename T0::Type);

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW * 4, pto::BLayout::RowMajor, -1, -1>;

    DataTileDefine dstTile(shape3, shape4);
    DataTileDefine srcTile(shape3, shape4);
    DataTileDefine tmp0Tile(shape3, shape4);
    DataTileDefine tmp1Tile(shape3, shape4);
    DataTileDefine tmp2Tile(shape3, shape4);
    MaskTileDefine maskTile(shape3, shape4);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto srcOffset = GenTileOffset(src, tileOffsets);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dataTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * dataTypeSize));
                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + 0 * tileH * tileW * dataTypeSize));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + 1 * tileH * tileW * dataTypeSize));
                pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + 2 * tileH * tileW * dataTypeSize));
                pto::TASSIGN(maskTile, (uint64_t)(tmp.GetAddr() + 3 * tileH * tileW * dataTypeSize));
                TLog1pTileImpl(dstTile, srcTile, tmp0Tile, tmp1Tile, tmp2Tile, maskTile);
            }
        }
    }
}

#endif
