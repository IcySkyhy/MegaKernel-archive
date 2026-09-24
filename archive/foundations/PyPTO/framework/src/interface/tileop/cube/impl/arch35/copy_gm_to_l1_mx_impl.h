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
 * \file copy_gm_to_l1_mx_impl.h
 * \brief MX Matmul Scale Data DDR->L1 Load Internal Implementation (Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_ARCH35_COPY_GM_TO_L1_MX_IMPL__H
#define TILEOP_TILE_OPERATOR_ARCH35_COPY_GM_TO_L1_MX_IMPL__H

#include "../cube_utils.h"

template <CopyInMode mode, typename Coord, typename TileData, typename GlobalData>
INLINE void TLoadAMXImpl(TileData& dst, GlobalData& src, const Coord& coord)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM3, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM3, 0>(coord);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM3, typename TileData::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value *
                               SHAPE_DIM2;
    using shapeDim2 = pto::Shape<1, 1, -1, -1, SHAPE_DIM2>;
    using strideDim3 = pto::Stride<-1, -1, -1, SHAPE_DIM2, 1>;
    using tileData = pto::Tile<pto::TileType::Mat, typename TileData::Type, staticL1H, staticL1W,
                               pto::BLayout::RowMajor, -1, -1, pto::SLayout::RowMajor, pto::TileConfig::alignedSize>;
    static_assert(mode == CopyInMode::ND2NZ || mode == CopyInMode::DN2NZ,
                  "ScaleA GM->L1 data movement only supports ND2NZ or DN2NZ mode.");
    if constexpr (mode == CopyInMode::ND2NZ) {
        int64_t gmOffset = offset1 * SHAPE_DIM2 + offset0 * srcShape1 * SHAPE_DIM2;
        using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim3, pto::Layout::MX_A_ND>;
        globalData src0Global(
            (__gm__ typename GlobalData::Type*)(src.GetAddr() + gmOffset), shapeDim2(dstShape0, dstShape1),
            strideDim3(srcShape0 * srcShape1 * SHAPE_DIM2, srcShape0 * srcShape1 * SHAPE_DIM2, srcShape1 * SHAPE_DIM2));
        tileData dstL1(dstShape0, dstShape1 * SHAPE_DIM2);
        pto::TASSIGN(dstL1, static_cast<uint64_t>(dst.GetAddr()));
        pto::TLOAD(dstL1, src0Global);
    } else if constexpr (mode == CopyInMode::DN2NZ) {
        int64_t gmOffset = offset1 * SHAPE_DIM2 + offset0 * SHAPE_DIM2 * srcShape1;
        using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim3, pto::Layout::MX_A_DN>;
        globalData src0Global(
            (__gm__ typename GlobalData::Type*)(src.GetAddr() + gmOffset), shapeDim2(dstShape0, dstShape1),
            strideDim3(srcShape0 * srcShape1 * SHAPE_DIM2, srcShape0 * srcShape1 * SHAPE_DIM2, srcShape1 * SHAPE_DIM2));
        tileData dstL1(dstShape0, dstShape1 * SHAPE_DIM2);
        pto::TASSIGN(dstL1, static_cast<uint64_t>(dst.GetAddr()));
        pto::TLOAD(dstL1, src0Global);
    }
}

template <CopyInMode mode, typename Coord, typename TileData, typename GlobalData>
INLINE void TLoadBMXImpl(TileData& dst, GlobalData& src, const Coord& coord)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM3, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM3, 0>(coord);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM3, typename TileData::TileShape>::type::value *
                               SHAPE_DIM2;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, 1, -1, -1, SHAPE_DIM2>;
    using strideDim3 = pto::Stride<-1, -1, -1, SHAPE_DIM2, 1>;
    using tileData = pto::Tile<pto::TileType::Mat, typename TileData::Type, staticL1H, staticL1W,
                               pto::BLayout::ColMajor, -1, -1, pto::SLayout::ColMajor, pto::TileConfig::alignedSize>;
    static_assert(mode == CopyInMode::ND2NZ || mode == CopyInMode::DN2NZ,
                  "ScaleB GM->L1 data movement only supports ND2NZ or DN2NZ mode.");
    if constexpr (mode == CopyInMode::ND2NZ) {
        int64_t gmOffset = offset1 * SHAPE_DIM2 + offset0 * SHAPE_DIM2 * srcShape1;
        using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim3, pto::Layout::MX_B_ND>;
        globalData src0Global(
            (__gm__ typename GlobalData::Type*)(src.GetAddr() + gmOffset), shapeDim2(dstShape0, dstShape1),
            strideDim3(srcShape0 * srcShape1 * SHAPE_DIM2, srcShape0 * srcShape1 * SHAPE_DIM2, srcShape1 * SHAPE_DIM2));
        tileData dstL1(dstShape0 * SHAPE_DIM2, dstShape1);
        pto::TASSIGN(dstL1, static_cast<uint64_t>(dst.GetAddr()));
        pto::TLOAD(dstL1, src0Global);
    } else if constexpr (mode == CopyInMode::DN2NZ) {
        int64_t gmOffset = offset1 * SHAPE_DIM2 + offset0 * srcShape1 * SHAPE_DIM2;
        using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim3, pto::Layout::MX_B_DN>;
        globalData src0Global(
            (__gm__ typename GlobalData::Type*)(src.GetAddr() + gmOffset), shapeDim2(dstShape0, dstShape1),
            strideDim3(srcShape0 * srcShape1 * SHAPE_DIM2, srcShape0 * srcShape1 * SHAPE_DIM2, srcShape1 * SHAPE_DIM2));
        tileData dstL1(dstShape0 * SHAPE_DIM2, dstShape1);
        pto::TASSIGN(dstL1, static_cast<uint64_t>(dst.GetAddr()));
        pto::TLOAD(dstL1, src0Global);
    }
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_COPY_GM_TO_L1_MX_IMPL__H
