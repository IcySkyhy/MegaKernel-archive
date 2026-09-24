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
 * \file copy_l1_to_l0_impl.h
 * \brief L1 to L0 Data Movement Interface Implementation (Atlas A3, Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_COPY_L1_TO_L0_IMPL__H
#define TILEOP_TILE_OPERATOR_COPY_L1_TO_L0_IMPL__H

#include "cube_utils.h"

template <bool isTrans, bool isMX, bool isGemv = false, typename DstTileData, typename SrcTileData>
INLINE void TExtractL1ToL0Impl(DstTileData& dst, SrcTileData& src, const int64_t& offset0, const int64_t& offset1)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename SrcTileData::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename SrcTileData::TileShape>::type::value;
    constexpr auto staticL0H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstTileData::TileShape>::type::value;
    constexpr auto staticL0W = Std::tuple_element<shapeSize - 1, typename DstTileData::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    // MX Matmul场景，L0的k轴需要保证64对齐，非对齐输入场景需要对validshapeK对齐到64
    if constexpr (DstTileData::FORMAT == Hardware::L0A) {
        if constexpr (isMX) {
            dstShape1 = (dstShape1 + MX_BLOCK_ALIGN_BYTE - 1) / MX_BLOCK_ALIGN_BYTE * MX_BLOCK_ALIGN_BYTE;
        }
        // validK=0场景下，L1已预填充零数据，L0A使用staticL0W（静态tile大小）保证计算结果为全零
        dstShape1 = dstShape1 == 0 ? staticL0W : dstShape1;
    } else {
        if constexpr (isMX) {
            dstShape0 = (dstShape0 + MX_BLOCK_ALIGN_BYTE - 1) / MX_BLOCK_ALIGN_BYTE * MX_BLOCK_ALIGN_BYTE;
        }
        // L0B同理，validK=0场景下使用staticL0W保证计算结果为全零
        dstShape0 = dstShape0 == 0 ? staticL0H : dstShape0;
    }
    // L1 Tile内模板参数对应含义：
    // Tile类型为Cube用于Matmul，矩阵数据类型，TileShape0，TileShape1，大分型RowMajor表明Z，ColMajor表明N,
    // validShape0, validShape1, 小分型
    using tileL1Tensor = pto::Tile<pto::TileType::Mat, typename SrcTileData::Type, isTrans ? staticL1W : staticL1H,
                                   isTrans ? staticL1H : staticL1W,
                                   isGemv ? pto::BLayout::RowMajor :
                                            (isTrans ? pto::BLayout::RowMajor : pto::BLayout::ColMajor),
                                   -1, -1, isTrans ? pto::SLayout::ColMajor : pto::SLayout::RowMajor>;
    // L0 TileLeft为L0A的Tile，TileRight为L0B的Tile，传入的值分别为：
    // 矩阵数据类型，tileShape0，tileShape1，validShape0，validShape0（-1表明传递动态值，在声明时传入）
    using tileL0Tensor = std::conditional_t<
        DstTileData::FORMAT == Hardware::L0A,
        pto::TileLeftCompact<typename DstTileData::Type, staticL0H, staticL0W, -1, -1>,
        pto::TileRightCompact<typename DstTileData::Type, staticL0H, staticL0W, -1, -1>>;
    tileL1Tensor l1Tile(srcShape0, srcShape1);
    tileL0Tensor l0Tile(dstShape0, dstShape1);
    // FP32场景时，当输入shape的内轴不满足c0size=32对齐，会向16元素补齐，需设置SetKAligned表示跳过填充的数据
    if (std::is_same<typename tileL0Tensor::DType, float>::value && DstTileData::FORMAT == Hardware::L0A) {
        l0Tile.SetKAligned(true);
    }
    pto::TASSIGN(l1Tile, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l0Tile, static_cast<uint64_t>(dst.GetAddr()));
    pto::TEXTRACT(l0Tile, l1Tile, isTrans ? offset1 : offset0, isTrans ? offset0 : offset1);
}

#endif // TILEOP_TILE_OPERATOR_COPY_L1_TO_L0_IMPL__H
