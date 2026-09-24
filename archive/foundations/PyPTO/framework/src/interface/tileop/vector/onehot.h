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
 * \file one_hot.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_ONEHOT__H
#define TILEOP_TILE_OPERATOR_ONEHOT__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename TileData>
PTO_INST void TZEROS(TileData& dst, int shape3, int shape4)
{
    static_assert(TileData::isRowMajor, "layout of dst must be pto::BLayout::RowMajor.");
    if constexpr (std::is_same_v<typename TileData::DType, int64_t>) {
        using DstTileType = pto::Tile<TileData::Loc, int32_t, TileData::Rows, TileData::Cols * 2,
                                      pto::BLayout::RowMajor, -1, -1>;
        DstTileType dstTile(shape3, shape4 * 2);
        pto::TASSIGN(dstTile, (uint64_t)dst.data());
        pto::TEXPANDS(dstTile, 0);
    } else {
        pto::TEXPANDS(dst, 0);
    }
}

#define OP_TILE_OP_ONEHOT TOneHot
template <typename DST, typename SRC>
TILEOP void TOneHot(DST dst, SRC src)
{
    constexpr auto dstShapeSize = Std::tuple_size<typename DST::Shape>::value;
    constexpr auto srcShapeSize = Std::tuple_size<typename SRC::Shape>::value;
    static_assert(srcShapeSize + 1 == dstShapeSize, "dst Shape Size must be src Shape Size + 1.");
    auto dstTile = PtoTile<DST>(dst);
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    const auto srcLayout = src.GetLayout();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    using SrcDtype = typename SRC::Type;
    using DstDtype = typename DST::Type;
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<DST, DIM_5TH, MAX_DIMS>();
    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, dstOffset);
                TZEROS(dstTile.Data(), dstShape3, dstShape4);
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                auto srcPtr = (__ubuf__ SrcDtype*)(src.GetAddr() + srcOffset * sizeof(SrcDtype));
                for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    auto dstPtr = dstTile.Data().data() + n3Index * dstTileW;
                    SrcDtype onePos = srcPtr[n3Index];
                    dstPtr[onePos] = 1;
                }
                set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            }
        }
    }
}
#endif
