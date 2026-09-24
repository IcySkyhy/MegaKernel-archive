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
 * \file copy_gm_to_l1_impl.h
 * \brief DDR to L1 Data Movement Interface Implementation (Atlas A3, Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_COPY_GM_TO_L1_IMPL__H
#define TILEOP_TILE_OPERATOR_COPY_GM_TO_L1_IMPL__H

#include "cube_utils.h"

// validK=0场景：将L1 tensor填充为零，确保matmul计算产生正确的全零输出
template <typename TileData>
INLINE void TFillPadZeroK(TileData& dst)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    // validShape设为0，表示无有效数据
    int64_t dstShape0 = 0;
    int64_t dstShape1 = 0;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename TileData::TileShape>::type::value;
    using tileData = pto::Tile<pto::TileType::Mat, typename TileData::Type, staticL1H, staticL1W,
                               pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
    // 构造时传入validShape=(0,0)，TFILLPAD会填充整个静态分配的空间
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, static_cast<uint64_t>(dst.GetAddr()));
    pto::TFILLPAD(dstL1, dstL1);
}

template <PaddingMode padMode, typename l1TileData>
INLINE void TLoadFillPad(l1TileData& l1Tile, int64_t l1Shape0, int64_t l1Shape1)
{
    // NORMAL_PADDING_MODE场景只对外轴进行补零操作，补齐到ceilalign(l1Shape0, 16)
    if constexpr (padMode == PaddingMode::NORMAL_PADDING_MODE) {
        if ((l1Shape0 & 0xF) != 0) {
            pto::TFILLPAD(l1Tile, l1Tile);
        }
        // MX_PADDING_MODE场景为MXMatmul场景下，对内外轴同时补零，补齐到对应的tileshape大小
    } else if constexpr (padMode == PaddingMode::MX_PADDING_MODE) {
        pto::TFILLPAD(l1Tile, l1Tile);
    }
}

// Copy data from DDR to L1 with ND -> NZ format
template <PaddingMode padMode, typename TileData, typename GlobalData>
INLINE void TLoadND2NZ(TileData& dst, GlobalData& src, const int64_t& offset0, const int64_t& offset1)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename TileData::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    using tileData = pto::Tile<
        pto::TileType::Mat, typename TileData::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1, -1,
        pto::SLayout::RowMajor, pto::TileConfig::fractalABSize, pto::PadValue::Null,
        padMode == PaddingMode::NORMAL_PADDING_MODE ? pto::CompactMode::RowAlignedPadding : pto::CompactMode::Null>;
    int64_t gmOffset = offset1 + offset0 * srcShape1;
    // FP4数据类型数据宽度减半，地址偏移需要右移1位
    if constexpr (CheckIsB4<TileData>()) {
        gmOffset = gmOffset >> 1;
    }
    globalData src0Global((__gm__ typename GlobalData::Type*)(src.GetAddr() + gmOffset),
                          shapeDim2(dstShape0, dstShape1), strideDim2(srcStride0, srcStride1));
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, static_cast<uint64_t>(dst.GetAddr()));
    pto::TLOAD(dstL1, src0Global);
    // L1数据为NZ时，外轴非16元素对齐需要pad到16对齐
#ifndef __LITE_NPU
    TLoadFillPad<padMode>(dstL1, dstShape0, dstShape1);
#endif
}

// Copy data from DDR to L1 with NZ -> NZ format
template <PaddingMode padMode, typename TileData, typename GlobalData>
INLINE void TLoadNZ2NZ(TileData& dst, GlobalData& src, const int64_t& dstOffset0, const int64_t& dstOffset1,
                       const int64_t& srcOffset0, const int64_t& srcOffset1, const int64_t& curH, const int64_t& curW)
{
    constexpr bool isB4 = CheckIsB4<TileData>();
    constexpr int64_t c0Size = isB4 ? FP4_BLOCK_ALIGN_BYTE : BLOCK_ALIGN_BYTE / sizeof(typename GlobalData::Type);
    constexpr auto shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    int64_t srcShape0 = curH;
    int64_t srcShape1 = curW;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename TileData::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, -1, -1, BLOCK_CUBE_M_N, c0Size>;
    using strideDim2 = pto::Stride<-1, -1, -1, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim2, pto::Layout::NZ>;
    using tileData = pto::Tile<pto::TileType::Mat, typename TileData::Type, staticL1H, staticL1W,
                               pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
    int64_t gmOffset = CalNZOffset(srcShape0, srcShape1, srcOffset0, srcOffset1, c0Size);
    int64_t l1Offset = CalNZOffset(dstShape0, dstShape1, dstOffset0, dstOffset1, c0Size);
    // FP4数据类型数据宽度减半，地址偏移需要右移1位
    if constexpr (isB4) {
        gmOffset = gmOffset >> 1;
    }
    globalData src0Global((__gm__ typename GlobalData::Type*)(src.GetAddr() + gmOffset),
                          shapeDim2(dstShape1 / c0Size, dstShape0 / BLOCK_CUBE_M_N),
                          strideDim2(srcShape0 * srcShape1, srcShape0 * c0Size, BLOCK_CUBE_M_N * c0Size));
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)((typename TileData::Type*)dst.GetAddr() + l1Offset));
    pto::TLOAD(dstL1, src0Global);
    // L1数据为NZ时，外轴非16元素对齐需要pad到16对齐
#ifndef __LITE_NPU
    TLoadFillPad<padMode>(dstL1, dstShape0, dstShape1);
#endif
}

// Copy data from DDR to L1 with ND -> ND format
template <typename TileData, typename GlobalData>
INLINE void TLoadND2ND(TileData& dst, GlobalData& src, const int64_t& offset0, const int64_t& offset1)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename TileData::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    // 目前场景,ND2ND只搬运bias和fixpipe，大小均为1 * N，offset0默认均为0
    int64_t gmOffset = offset1 + offset0 * srcShape1;
    globalData src0Global((__gm__ typename GlobalData::Type*)(src.GetAddr() + gmOffset),
                          shapeDim2(dstShape0, dstShape1), strideDim2(srcStride0, srcStride1));
    using tileData = pto::Tile<pto::TileType::Mat, typename TileData::Type, staticL1H, staticL1W,
                               pto::BLayout::RowMajor, -1, -1>;
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, static_cast<uint64_t>(dst.GetAddr()));
    pto::TLOAD(dstL1, src0Global);
}

// Copy data from DDR to L1 with ND -> NZ format after reshape
template <PaddingMode padMode, typename TileTensor, typename GlobalTensor>
INLINE void TReshapeLoadND2NZ(TileTensor& dst, GlobalTensor& src, const int64_t srcOffset0, const int64_t srcOffset1,
                              const int64_t gShape0, const int64_t gShape1)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileTensor::Shape>::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileTensor::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename TileTensor::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename GlobalTensor::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    using tileData = pto::Tile<pto::TileType::Mat, typename TileTensor::Type, staticL1H, staticL1W,
                               pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, pto::TileConfig::fractalABSize,
                               pto::PadValue::Null, pto::CompactMode::RowAlignedPadding>;
    int64_t gmOffset = srcOffset1 + srcOffset0 * gShape1;
    // FP4数据类型数据宽度减半，地址偏移需要右移1位
    if constexpr (CheckIsB4<TileTensor>()) {
        gmOffset = gmOffset >> 1;
    }
    globalData srcGlobal((__gm__ typename GlobalTensor::Type*)(src.GetAddr() + gmOffset),
                         shapeDim2(dstShape0, dstShape1), strideDim2(gShape1, 1));
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, static_cast<uint64_t>(dst.GetAddr()));
    pto::TLOAD(dstL1, srcGlobal);
    // L1数据为NZ时，外轴非16元素对齐需要pad到16对齐
#ifndef __LITE_NPU
    if ((dstShape0 & 0xF) != 0) {
        pto::TFILLPAD(dstL1, dstL1);
    }
#endif
}

#endif // TILEOP_TILE_OPERATOR_COPY_GM_TO_L1_IMPL__H
