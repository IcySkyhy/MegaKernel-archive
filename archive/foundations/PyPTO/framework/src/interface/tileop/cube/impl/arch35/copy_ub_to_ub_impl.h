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
 * \file copy_ub_to_ub_impl.h
 * \brief UB to UB Data Movement Interface Implementation (Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_ARCH35_COPY_UB_TO_UB_IMPL__H
#define TILEOP_TILE_OPERATOR_ARCH35_COPY_UB_TO_UB_IMPL__H

#include "../cube_utils.h"

template <typename DstTileData, typename SrcTileData>
INLINE void TMoveND2NZImpl(DstTileData& dst, SrcTileData& src)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr int64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    constexpr int64_t
        staticNDH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename SrcTileData::TileShape>::type::value;
    constexpr int64_t staticNDW = Std::tuple_element<shapeSize - 1, typename SrcTileData::TileShape>::type::value;
    constexpr int64_t
        staticNZH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstTileData::TileShape>::type::value;
    constexpr int64_t staticNZW = Std::tuple_element<shapeSize - 1, typename DstTileData::TileShape>::type::value;
    using tileNDTensor = pto::Tile<pto::TileType::Vec, typename SrcTileData::Type, staticNDH, staticNDW,
                                   pto::BLayout::RowMajor, staticNDH, staticNDW>;
    // staticNZH - 1 is for resolving bank conflicts
    using tileNZTensor = pto::Tile<pto::TileType::Vec, typename DstTileData::Type, staticNZH, staticNZW,
                                   pto::BLayout::ColMajor, staticNZH - 1, staticNZW, pto::SLayout::RowMajor,
                                   pto::TileConfig::fractalABSize, pto::PadValue::Null, pto::CompactMode::RowPlusOne>;
    tileNDTensor srcTile;
    tileNZTensor dstTile;
    pto::TASSIGN(srcTile, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(dstTile, static_cast<uint64_t>(dst.GetAddr()));
    pto::TMOV(dstTile, srcTile);
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_COPY_UB_TO_UB_IMPL__H
