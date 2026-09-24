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
 * \file conv_store_nz2dn_impl.h
 * \brief Copy data from L0C to DDR with NZ2DN, NZ -> NCHW/NCDHW (Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_STORE_NZ2DN_IMPL__H
#define TILEOP_TILE_OPERATOR_CONV_STORE_NZ2DN_IMPL__H
#include "conv_offset_utils.h"

/**
 * Copy data from L0C to DDR with NZ -> NCHW format.
 * dst: GM(NCHW)
 * src: l0c(NZ)
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_h_offset (conv2d 参数顺序调整：n, c, h, w, 0)
 * offset3: dst_w_offset
 * offset4: 0 (占位)
 */
template <int64_t reluType, typename T, typename U>
INLINE void TStoreConv2DNZ2DN(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN,
                              const int64_t& realCutW, const int64_t& cutW)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    int64_t dstN = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstC = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstH = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_3>(dst);
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideC = GetConvStride<CONV_IDX_1>(dst);
    int64_t dstStrideH = GetConvStride<CONV_IDX_2>(dst);
    int64_t dstStrideW = GetConvStride<CONV_IDX_3>(dst);
    int64_t gmOffset = offsetInfo.offset0 * dstStrideN + offsetInfo.offset1 * dstStrideC + offsetInfo.offset2 * dstW +
                       offsetInfo.offset3;

    using shapeDim4 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim4 = pto::Stride<1, -1, -1, -1, -1>;
    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, srcM, srcN, pto::BLayout::ColMajor, -1, -1,
                               pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null,
                               pto::CompactMode::Null>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim4, strideDim4, pto::Layout::NCHW>;
    // 分块搬出，每次搬出cutW大小的数据
    for (int64_t loopH = 0; loopH < (realM / realCutW); loopH++) {
        globalData dstGlobal((__gm__ typename T::Type*)(dst.GetAddr() + gmOffset), shapeDim4(dstN, dstC, dstH, dstW),
                             strideDim4(dstStrideN, dstStrideC, dstStrideH, dstStrideW));
        tileData srcL0C(realCutW, realN);
        pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr() + loopH * cutW * BLOCK_CUBE_M_N * sizeof(typename U::Type));
        pto::TSTORE<tileData, globalData, pto::AtomicType::AtomicNone,
                    reluType == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(dstGlobal, srcL0C);
        gmOffset += dstW;
    }
}

/**
 * Copy data from L0C to DDR with NZ -> NCDHW format.
 * dst: GM(NCDHW)
 * src: l0c(NZ)
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_d_offset
 * offset3: dst_h_offset
 * offset4: dst_w_offset
 */
template <int64_t reluType, typename T, typename U>
INLINE void TStoreConv3DNZ2DN(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN,
                              const int64_t& realCutW, const int64_t& cutW)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    int64_t dstN = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstC = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstD = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstH = GetConvShape<CONV_IDX_3>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_4>(dst);
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideC = GetConvStride<CONV_IDX_1>(dst);
    int64_t dstStrideD = GetConvStride<CONV_IDX_2>(dst);
    int64_t dstStrideH = GetConvStride<CONV_IDX_3>(dst);
    int64_t dstStrideW = GetConvStride<CONV_IDX_4>(dst);
    ShapeInfo shapeInfo{dstC, dstD, dstH, dstW};
    using shapeDim5 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, srcM, srcN, pto::BLayout::ColMajor, -1, -1,
                               pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null,
                               pto::CompactMode::Null>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim5, strideDim5, pto::Layout::NCDHW>;
    // 分块搬出，每次搬出cutW大小的数据
    for (int64_t loopH = 0; loopH < (realM / realCutW); loopH++) {
        int64_t gmOffset = CalStoreOffsetNCDHW(shapeInfo, offsetInfo, loopH);
        globalData dstGlobal((__gm__ typename T::Type*)(dst.GetAddr() + gmOffset),
                             shapeDim5(dstN, dstC, dstD, dstH, dstW),
                             strideDim5(dstStrideN, dstStrideC, dstStrideD, dstStrideH, dstStrideW));
        tileData srcL0C(realCutW, realN);
        pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr() + loopH * cutW * BLOCK_CUBE_M_N * sizeof(typename U::Type));
        pto::TSTORE<tileData, globalData, pto::AtomicType::AtomicNone,
                    reluType == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(dstGlobal, srcL0C);
    }
}

template <bool isConv3D, int64_t reluType, typename T, typename U>
INLINE void TStoreConvNZ2DN(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN,
                            const int64_t& realCutW, const int64_t& cutW)
{
    if constexpr (isConv3D) {
        TStoreConv3DNZ2DN<reluType>(dst, src, offsetInfo, realM, realN, realCutW, cutW);
    } else {
        TStoreConv2DNZ2DN<reluType>(dst, src, offsetInfo, realM, realN, realCutW, cutW);
    }
}

#endif // TILEOP_TILE_OPERATOR_CONV_STORE_NZ2DN_IMPL__H
