/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file atanh.h
 * \brief atanh(x) = 0.5 * ln((1 + x) / (1 - x))
 */

#ifndef TILEOP_TILE_OPERATOR_ATANH__H
#define TILEOP_TILE_OPERATOR_ATANH__H
#include "unary.h"
#include <type_traits>

constexpr float ATANH_LIMIT = 1e34f;

#define OP_TILE_OP_ATANH TAtanh
template <typename T0, typename T1, typename T2>
TILEOP void TAtanh(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    constexpr auto tileShapeSize = TileOp::GetAnyAxisMergeResult<
        DIM_1ST, Std::tuple_size<typename T0::TileShape>::value, typename T0::TileShape>();

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW * 4, pto::BLayout::RowMajor, -1, -1>;

    DataTileDefine dstTile(dstShape3, dstShape4);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    DataTileDefine tmp0Tile(dstShape3, dstShape4);
    DataTileDefine tmp1Tile(dstShape3, dstShape4);
    DataTileDefine tmp2Tile(dstShape3, dstShape4);
    DataTileDefine tmp3Tile(dstShape3, dstShape4);
    MaskTileDefine tmpMaskTile(dstShape3, dstShape4);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T2::Type>(GenTileOffset(tmp, tileOffsets));
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + 2 * tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp3Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + 3 * tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmpMaskTile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + 4 * tileShapeSize * dstTypeSize));

                // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
                pto::TABS(tmp3Tile, srcExecTile);
                SyncV();

                pto::TMULS(tmp1Tile, tmp3Tile, -1.0f);
                SyncV();

                pto::TADDS(tmp1Tile, tmp1Tile, 1.0f);
                SyncV();

                pto::TDIV(tmp2Tile, tmp3Tile, tmp1Tile);
                SyncV();

                pto::TMULS(tmp1Tile, tmp2Tile, 2.0f);
                SyncV();

                pto::TADDS(tmp0Tile, tmp1Tile, 1.0f);
                SyncV();

                pto::TADDS(tmp1Tile, tmp0Tile, -1.0f);
                SyncV();

                pto::TMINS(tmp1Tile, tmp1Tile, ATANH_LIMIT);
                SyncV();

                pto::TLOG(dstTile, tmp0Tile);
                SyncV();

                pto::TMUL(dstTile, dstTile, tmp2Tile);
                SyncV();

                pto::TDIV(tmp2Tile, dstTile, tmp1Tile);
                SyncV();

                // Handle x = 0 case
                pto::TCMPS(tmpMaskTile, tmp0Tile, 1.0f, pto::CmpMode::EQ);
                SyncV();
                pto::TSEL(tmp1Tile, tmpMaskTile, tmp3Tile, tmp2Tile, tmp0Tile);
                SyncV();

                // Handle sign: atanh(-x) = -atanh(x)
                pto::TCMPS(tmpMaskTile, srcExecTile, 0.0f, pto::CmpMode::LT);
                SyncV();
                pto::TMULS(tmp0Tile, tmp1Tile, -1.0f);
                SyncV();
                pto::TSEL(dstTile, tmpMaskTile, tmp0Tile, tmp1Tile, tmp2Tile);
                SyncV();
            }
        }
    }
}

#endif
