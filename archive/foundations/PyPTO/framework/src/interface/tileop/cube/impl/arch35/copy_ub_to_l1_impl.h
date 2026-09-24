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
 * \file copy_ub_to_l1_impl.h
 * \brief UB to L1 Data Movement Interface Implementation (Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_ARCH35_COPY_UB_TO_L1_IMPL__H
#define TILEOP_TILE_OPERATOR_ARCH35_COPY_UB_TO_L1_IMPL__H

#include "../cube_utils.h"

template <CopyMode mode, bool isGemv = false, typename Coord, typename DstTileData, typename SrcTileData>
INLINE void TCopyUB2L1Impl(DstTileData& dst, SrcTileData& src, const Coord& dstCoord, const Coord& srcCoord)
{
    static_assert(mode != CopyMode::UNKNOWN,
                  "[TCopyUB2L1 Error]: Current CopyMode is UNKNOWN. CopyMode only support EXTRACT, INSERT and MOVE");
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    constexpr bool isB4 = CheckIsB4<DstTileData>();
    constexpr int64_t c0Size = isB4 ? FP4_BLOCK_ALIGN_BYTE : BLOCK_ALIGN_BYTE / sizeof(typename SrcTileData::Type);
    int64_t srcOffset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(srcCoord);
    int64_t srcOffset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(srcCoord);
    int64_t dstOffset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(dstCoord);
    int64_t dstOffset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(dstCoord);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    constexpr int64_t
        staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename SrcTileData::TileShape>::type::value;
    constexpr int64_t staticUBW = Std::tuple_element<shapeSize - 1, typename SrcTileData::TileShape>::type::value;
    constexpr int64_t
        staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstTileData::TileShape>::type::value;
    constexpr int64_t staticL1W = Std::tuple_element<shapeSize - 1, typename DstTileData::TileShape>::type::value;
    // GEMV A 来自 UB 时保持 ND：tile 声明 RowMajor + NoneBox，命中 TExtractVecToMat 的 ND 分支
    constexpr pto::BLayout ubBLayout = isGemv ? pto::BLayout::RowMajor : pto::BLayout::ColMajor;
    constexpr pto::SLayout ubSLayout = isGemv ? pto::SLayout::NoneBox : pto::SLayout::RowMajor;
    constexpr pto::BLayout l1BLayout = isGemv ? pto::BLayout::RowMajor : pto::BLayout::ColMajor;
    constexpr pto::SLayout l1SLayout = isGemv ? pto::SLayout::NoneBox : pto::SLayout::RowMajor;
    using tileL1Tensor = pto::Tile<pto::TileType::Mat, typename DstTileData::Type, staticL1H, staticL1W, l1BLayout,
                                   staticL1H, staticL1W, l1SLayout>;
    tileL1Tensor l1Tile;
    pto::TASSIGN(l1Tile, (uint64_t)dst.GetAddr());
    if constexpr (mode == CopyMode::EXTRACT || mode == CopyMode::MOVE) {
        using tileUBTensor = pto::Tile<pto::TileType::Vec, typename SrcTileData::Type, staticUBH, staticUBW, ubBLayout,
                                       staticUBH, staticUBW, ubSLayout>;
        tileUBTensor UBTile;
        pto::TASSIGN(UBTile, (uint64_t)src.GetAddr());
        pto::TEXTRACT(l1Tile, UBTile, srcOffset0, srcOffset1);
    } else if (mode == CopyMode::INSERT) {
        using tileUBTensor = pto::Tile<pto::TileType::Vec, typename SrcTileData::Type, staticUBH, staticUBW,
                                       pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
        tileUBTensor UBTile(srcShape0, srcShape1);
        pto::TASSIGN(UBTile, (uint64_t)src.GetAddr());
        pto::TINSERT(l1Tile, UBTile, dstOffset0, dstOffset1);
    }
#ifndef __LITE_NPU
    // litenpu does not need set_l1_2d after ub2l1 dma, matmul mmad can be calculated by setting corresponding m,k,n
    // parameters
    using tileL1PadTensor = pto::Tile<pto::TileType::Mat, typename DstTileData::Type, staticL1H, staticL1W,
                                      pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
    tileL1PadTensor l1PadTile(dstShape0, dstShape1);
    pto::TFILLPAD(l1PadTile, l1PadTile);
#endif
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_COPY_UB_TO_L1_IMPL__H
