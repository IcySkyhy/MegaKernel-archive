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
 * \file copy_l1_to_bt_fb_impl.h
 * \brief L1 to BiasTable and L1 to FixPipeBuffer Data Movement Interface Implementation (Atlas A3, Ascend 950PR/Ascend
 * 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_COPY_L1_TO_BT_FB_IMPL__H
#define TILEOP_TILE_OPERATOR_COPY_L1_TO_BT_FB_IMPL__H

#include "cube_utils.h"

template <bool isTrans, typename DstTileData, typename SrcTileData>
INLINE void TExtractL1ToBTOrFBImpl(DstTileData& dst, SrcTileData& src)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename SrcTileData::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSize - 1, typename DstTileData::TileShape>::type::value;
    int64_t nL1 = GetShape<1>(src);
    int64_t nL0 = GetShape<1>(dst);
    using tileL1Tensor = pto::Tile<pto::TileType::Mat, typename SrcTileData::Type, 1, staticL1W, pto::BLayout::RowMajor,
                                   -1, -1>;
    using tileBiasOrFbTensor = pto::Tile<DstTileData::FORMAT == Hardware::BIAS ? pto::TileType::Bias :
                                                                                 pto::TileType::Scaling,
                                         typename DstTileData::Type, 1, staticL0BW, pto::BLayout::RowMajor, -1, -1>;
    tileL1Tensor l1Tensor(1, nL1);
    tileBiasOrFbTensor biasOrFbTensor(1, nL0);
    pto::TASSIGN<tileL1Tensor>(l1Tensor, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN<tileBiasOrFbTensor>(biasOrFbTensor, static_cast<uint64_t>(dst.GetAddr()));
    pto::TMOV(biasOrFbTensor, l1Tensor);
}

#endif // TILEOP_TILE_OPERATOR_COPY_L1_TO_BT_FB_IMPL__H
