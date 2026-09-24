/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_store_nz2nz_impl.h
 * \brief Copy data from L0C to DDR with NZ2NZ (chip-independent)
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_STORE_NZ2NZ_IMPL__H
#define TILEOP_TILE_OPERATOR_CONV_STORE_NZ2NZ_IMPL__H
#include "conv_utils.h"

/**
 * Copy data from L0C to DDR with NZ -> NC1HWC0 format.
 *
 * dst: GM [N, C1, H, W, C0]
 * src: l0c(NZ)
 * offset0: N
 * offset1: C1
 * offset2: H
 * offset3: W
 * offset4: 0
 */
template <int64_t reluType, typename T, typename U>
INLINE void TStoreConv2DNZ2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN,
                              const int64_t& realCutW, const int64_t& cutW)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename T::Type);
    int64_t dstShapeC1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstShapeH = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstShapeW = GetConvShape<CONV_IDX_3>(dst);
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideC1 = GetConvStride<CONV_IDX_1>(dst);
    int64_t dstStrideH = GetConvStride<CONV_IDX_2>(dst);
    int64_t dstStrideW = GetConvStride<CONV_IDX_3>(dst);
    int64_t dstStrideC0 = GetConvStride<CONV_IDX_4>(dst);
    int64_t gmOffset = offsetInfo.offset0 * dstStrideN + offsetInfo.offset1 * dstStrideC1 +
                       offsetInfo.offset2 * dstStrideH + offsetInfo.offset3 * dstStrideW;

    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, srcM, srcN, pto::BLayout::ColMajor, -1, -1,
                               pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null,
                               pto::CompactMode::Null>;
    // Shape: N=1, C1, H, W, C0=c0Size
    using shapeDim = pto::Shape<1, -1, -1, -1, c0Size>;
    using strideDim = pto::Stride<-1, -1, -1, -1, -1>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim, strideDim, pto::Layout::NC1HWC0>;
    // 分块搬出，每次搬出cutW大小的数据
    for (int64_t loopH = 0; loopH < (realM / realCutW); loopH++) {
        globalData dstGlobal((__gm__ typename T::Type*)(dst.GetAddr() + gmOffset),
                             shapeDim(dstShapeC1, dstShapeH, dstShapeW),
                             strideDim(dstStrideN, dstStrideC1, dstStrideH, dstStrideW, dstStrideC0));
        tileData srcL0C(realCutW, realN);
        pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr() + loopH * cutW * BLOCK_CUBE_M_N * sizeof(typename U::Type));
        pto::TSTORE<tileData, globalData, pto::AtomicType::AtomicNone,
                    reluType == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(dstGlobal, srcL0C);
        gmOffset += dstStrideH;
    }
}

/**
 * Copy data from L0C to DDR with NZ -> NDC1HWC0 format.
 *
 * dst: GM [N, D, C1, H, W, C0], C1 = CeilDiv(cout, C0)
 * src: l0c(NZ)
 * offset0: N
 * offset1: D
 * offset2: C1
 * offset3: H
 * offset4: W
 */
template <int64_t reluType, typename T, typename U>
INLINE void TStoreConv3DNZ2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN,
                              const int64_t& realCutW, const int64_t& cutW)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    int64_t dstShapeD = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstShapeC1 = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstShapeH = GetConvShape<CONV_IDX_3>(dst);
    int64_t dstShapeW = GetConvShape<CONV_IDX_4>(dst);
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideD = GetConvStride<CONV_IDX_1>(dst);
    int64_t dstStrideC1 = GetConvStride<CONV_IDX_2>(dst);
    int64_t dstStrideH = GetConvStride<CONV_IDX_3>(dst);
    int64_t dstStrideW = GetConvStride<CONV_IDX_4>(dst);
    int64_t gmOffset = offsetInfo.offset0 * dstStrideN + offsetInfo.offset1 * dstStrideD +
                       offsetInfo.offset2 * dstStrideC1 + offsetInfo.offset3 * dstStrideH +
                       offsetInfo.offset4 * dstStrideW;

    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, srcM, srcN, pto::BLayout::ColMajor, -1, -1,
                               pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null,
                               pto::CompactMode::Null>;
    // Shape: N=1, D, C1, H, W, C0(由于Shape只支持5维，且C0可以根据dtype计算得出，所以这里不配置C0)
    using shapeDim = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim = pto::Stride<-1, -1, -1, -1, -1>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim, strideDim, pto::Layout::NDC1HWC0>;
    for (int64_t loopH = 0; loopH < (realM / realCutW); loopH++) {
        globalData dstGlobal((__gm__ typename T::Type*)(dst.GetAddr() + gmOffset),
                             shapeDim(dstShapeD, dstShapeC1, dstShapeH, dstShapeW),
                             strideDim(dstStrideN, dstStrideD, dstStrideC1, dstStrideH, dstStrideW));
        tileData srcL0C(realCutW, realN);
        pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr() + loopH * cutW * BLOCK_CUBE_M_N * sizeof(typename U::Type));
        pto::TSTORE<tileData, globalData, pto::AtomicType::AtomicNone,
                    reluType == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(dstGlobal, srcL0C);
        gmOffset += dstStrideH;
    }
}

template <bool isConv3D, int64_t reluType, typename T, typename U>
INLINE void TStoreConvNZ2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN,
                            const int64_t& realCutW, const int64_t& cutW)
{
    if constexpr (isConv3D) {
        TStoreConv3DNZ2NZ<reluType>(dst, src, offsetInfo, realM, realN, realCutW, cutW);
    } else {
        TStoreConv2DNZ2NZ<reluType>(dst, src, offsetInfo, realM, realN, realCutW, cutW);
    }
}

#endif // TILEOP_TILE_OPERATOR_CONV_STORE_NZ2NZ_IMPL__H
