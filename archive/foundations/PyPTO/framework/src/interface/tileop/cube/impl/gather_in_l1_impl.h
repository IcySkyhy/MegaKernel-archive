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
 * \file gather_in_l1_impl.h
 * \brief gatherInL1 Interface Implementation, only used in deepseek model (Atlas A3, Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_GATHER_IN_L1_IMPL__H
#define TILEOP_TILE_OPERATOR_GATHER_IN_L1_IMPL__H

#include "cube_utils.h"

template <int64_t blockSize, typename globalData, typename tileData, typename TileData, typename GlobalData,
          typename BlockT, typename OffsetT>
INLINE void GatherExecute(TileData dst, GlobalData src, BlockT block, OffsetT offset, uint64_t offsetsStartOffset,
                          uint64_t srcColumnStartOffset, uint64_t GMBlockTableOffset, int64_t srcShape0,
                          int64_t srcShape1, int64_t srcStride0, int64_t srcStride1, int64_t dstShape0,
                          int64_t dstShape1, int64_t srcCol, int64_t loop, int64_t c0Size)
{
    for (int64_t i = 0; i < loop; i++) {
        uint64_t gatherOffset = offset.GetAddr()[i + offsetsStartOffset];
        gatherOffset = CalaOffset2PageAttention<uint64_t, typename BlockT::Type, blockSize>(
            block.GetAddr() + GMBlockTableOffset, gatherOffset);
        globalData src0Global(
            (__gm__ typename GlobalData::Type*)(src.GetAddr() + gatherOffset * srcCol + srcColumnStartOffset),
            pto::Shape<1, 1, 1, -1, -1>(srcShape0, srcShape1), pto::Stride<1, 1, 1, -1, -1>(srcStride0, srcStride1));
        tileData dstL1(dstShape0, dstShape1);
        pto::TASSIGN(dstL1, (uint64_t)((__cbuf__ typename TileData::Type*)dst.GetAddr() + i * c0Size));
        pto::TLOAD(dstL1, src0Global);
    }
}

template <int64_t blockSize, typename TileData, typename GlobalData, typename BlockT, typename OffsetT,
          typename SrcCoord, typename OffsetCoord, typename BlockCoord>
INLINE void TGatherInL1Impl(TileData dst, GlobalData src, BlockT block, OffsetT offset, SrcCoord srcCoord,
                            OffsetCoord offsetCoord, BlockCoord blockCoord)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename TileData::Type);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t blockShape1 = GetShape<1>(block);
    int64_t offsetShape1 = GetShape<1>(offset);
    uint64_t srcColumnStartOffset = TileOp::GetTupleElement<SrcCoord, DIM_1ST, SHAPE_DIM2, 0>(srcCoord);
    uint64_t offsetsRowStartOffset = TileOp::GetTupleElement<OffsetCoord, DIM_1ST, SHAPE_DIM2, 0>(offsetCoord);
    uint64_t offsetsColumnStartOffset = TileOp::GetTupleElement<OffsetCoord, DIM_2ND, SHAPE_DIM2, 0>(offsetCoord);
    uint64_t offsetsStartOffset = offsetsRowStartOffset * offsetShape1 + offsetsColumnStartOffset;
    uint64_t GMBlockTableOffset0 = TileOp::GetTupleElement<BlockCoord, DIM_1ST, SHAPE_DIM2, 0>(blockCoord);
    uint64_t GMBlockTableOffset1 = TileOp::GetTupleElement<BlockCoord, DIM_2ND, SHAPE_DIM2, 0>(blockCoord);
    uint64_t GMBlockTableOffset = GMBlockTableOffset0 * blockShape1 + GMBlockTableOffset1;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename TileData::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    if (dstShape1 % c0Size > 0) {
        using tileData = pto::Tile<pto::TileType::Mat, typename TileData::Type, staticL1H, staticL1W,
                                   pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
        GatherExecute<blockSize, globalData, tileData>(
            dst, src, block, offset, offsetsStartOffset, srcColumnStartOffset, GMBlockTableOffset, 1, staticL1W,
            srcStride0, srcStride1, dstShape0, dstShape1, srcShape1, dstShape0, c0Size);
    } else {
        using tileData = pto::Tile<pto::TileType::Mat, typename TileData::Type, staticL1W / c0Size, staticL1H * c0Size,
                                   pto::BLayout::RowMajor, -1, -1>;
        // 这里需要采用性能更高的ND2ND的搬运方式。
        // GM上将(1, dstShape1)的数据，转变为(dstShape1 / c0Size, c0Size)。
        // L1上将(staticL1W / c0Size, staticL1H / 16, 16, c0Size)的NZ数据，对每c0Size列做展平为一行，得到(staticL1W /
        // c0Size, staticL1H * c0Size)的ND格式。 那么L1上有效数据就是(dstShape1 / c0Size, c0Size)。
        GatherExecute<blockSize, globalData, tileData>(
            dst, src, block, offset, offsetsStartOffset, srcColumnStartOffset, GMBlockTableOffset, dstShape1 / c0Size,
            c0Size, c0Size, 1, dstShape1 / c0Size, c0Size, srcShape1, dstShape0, c0Size);
    }
}

#endif // TILEOP_TILE_OPERATOR_GATHER_IN_L1_IMPL__H
