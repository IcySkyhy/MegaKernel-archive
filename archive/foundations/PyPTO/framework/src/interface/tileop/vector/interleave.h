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
 * \file interleave.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_INTERLEAVE__H
#define TILEOP_TILE_OPERATOR_INTERLEAVE__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#if defined PTO_NPU_ARCH_A5

enum class InterleaveOp {
    INTERLEAVE,
    DEINTERLEAVE,
};

template <InterleaveOp op, typename T0, typename T1, typename T2, typename T3>
TILEOP void InterleaveComputeImpl(T0 dst0, T1 dst1, T2 src0, T3 src1)
{
    if constexpr (op == InterleaveOp::INTERLEAVE) {
        pto::TINTERLEAVE(dst1, dst0, src1, src0);
        return;
    }
    if constexpr (op == InterleaveOp::DEINTERLEAVE) {
        pto::TDEINTERLEAVE(dst1, dst0, src1, src0);
        return;
    }
}

template <InterleaveOp op, typename T0, typename T1, typename T2, typename T3>
TILEOP void InterleaveCompute(T0 dst0, T1 dst1, T2 src0, T3 src1)
{
    const auto dstLayout = dst0.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dst0Tile = PtoTile<T0>(dst0);
    auto dst1Tile = PtoTile<T1>(dst1);
    auto src0Tile = PtoTile<T2>(src0);
    auto src1Tile = PtoTile<T3>(src1);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dst0Tile.Assign(dst0, tileOffsets);
                dst1Tile.Assign(dst1, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                src1Tile.Assign(src1, tileOffsets);
                InterleaveComputeImpl<op>(dst0Tile.Data(), dst1Tile.Data(), src0Tile.Data(), src1Tile.Data());
            }
        }
    }
}

#define OP_TILE_OP_INTERLEAVE TInterleave
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TInterleave(T0 dst0, T1 dst1, T2 src0, T3 src1)
{
    InterleaveCompute<InterleaveOp::INTERLEAVE>(dst0, dst1, src0, src1);
}

#define OP_TILE_OP_DEINTERLEAVE TDeInterleave
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TDeInterleave(T0 dst0, T1 dst1, T2 src0, T3 src1)
{
    InterleaveCompute<InterleaveOp::DEINTERLEAVE>(dst0, dst1, src0, src1);
}

#define OP_TILE_OP_DEINTERLEAVE_SINGLE TDeInterleave
template <typename T0, typename T1, typename T2>
TILEOP void TDeInterleave(T0 dst0, T1 dst1, T2 src)
{
    const auto dstLayout = dst0.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dst0Tile = PtoTile<T0>(dst0);
    auto dst1Tile = PtoTile<T1>(dst1);
    auto srcTile = PtoTile<T2>(src);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dst0Tile.Assign(dst0, tileOffsets);
                dst1Tile.Assign(dst1, tileOffsets);
                srcTile.Assign(src, tileOffsets);
                pto::TDEINTERLEAVE(dst1Tile.Data(), dst0Tile.Data(), srcTile.Data());
            }
        }
    }
}

#endif // defined PTO_NPU_ARCH_A5
#endif // TILEOP_TILE_OPERATOR_INTERLEAVE__H
