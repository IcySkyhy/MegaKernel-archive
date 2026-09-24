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
 * \file conv_load_dn2nz_impl.h
 * \brief Copy input data from DDR to L1 with DN2NZ, NCHW/NCDHW -> NZ (Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_LOAD_DN2NZ_IMPL__H
#define TILEOP_TILE_OPERATOR_CONV_LOAD_DN2NZ_IMPL__H
#include "conv_offset_utils.h"

/**
 * Copy input data from DDR to L1 with DN2NZ, input NCHW -> NC1HWC0, weight NCHW -> FZ.
 * dst: input -> AL1(NC1HWC0), weight -> BL1(FZ)
 * src: input -> GM(NCHW), weigh -> GM(NCHW)
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_h_offset, weight -> 0 (conv2d 参数顺序调整：n, c, h, w, 0)
 * offset3: input -> src_w_offset, weight -> 0
 * offset4: input -> 0 (占位), weight -> 0
 * isFmap: true -> input, false -> weight
 */
template <bool isFmap, typename T, typename U>
INLINE void TLoadConv2DDN2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const ShapeInfo& srcShapeInfo)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t srcC = GetConvShape<CONV_IDX_1>(src);
    int64_t srcH = GetConvShape<CONV_IDX_2>(src);
    int64_t srcW = GetConvShape<CONV_IDX_3>(src);
    int64_t dstShape0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstShape1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstShape2 = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstShape3 = GetConvShape<CONV_IDX_3>(dst);
    int64_t srcStrideN = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStrideC = GetConvStride<CONV_IDX_1>(src);
    int64_t srcStrideH = GetConvStride<CONV_IDX_2>(src);
    int64_t srcStrideW = GetConvStride<CONV_IDX_3>(src);
    constexpr auto stcDstShape0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto stcDstShape1 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    constexpr auto stcDstShape2 = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;
    constexpr auto stcDstShape3 = Std::tuple_element<CONV_IDX_3, typename T::TileShape>::type::value;
    ShapeInfo shapeInfo = {srcC, srcH, srcW};
    using shapeDim4 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim4 = pto::Stride<1, -1, -1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim4, strideDim4, pto::Layout::NCHW>;
    int64_t gmOffset = CalLoadOffsetNCHW<isFmap>(shapeInfo, offsetInfo);
    // conv2d 参数顺序调整：n, c, h, w, 0 (最后的0占位)
    // srcShapeInfo: {shape0(n), shape1(c), shape2(h), shape3(w), shape4(0)}
    globalData srcGlobal((__gm__ typename U::Type*)(src.GetAddr() + gmOffset),
                         shapeDim4(srcShapeInfo.shape0, srcShapeInfo.shape1, srcShapeInfo.shape2, srcShapeInfo.shape3),
                         strideDim4(srcStrideN, srcStrideC, srcStrideH, srcStrideW));
    if constexpr (isFmap) {
        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * stcDstShape3 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::NC1HWC0,
                                       pto::ConvTileShape<-1, -1, -1, -1, c0Size>>;
        tileData dstL1(dstShape0, dstShape1, dstShape2, dstShape3);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    } else {
        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::FRACTAL_Z,
                                       pto::ConvTileShape<-1, -1, -1, -1, 1>>;
        tileData dstL1(dstShape0, dstShape1, dstShape2, dstShape3);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    }
}

/**
 * Copy input data from DDR to L1 with DN2NZ, input NCDHW -> NDC1HWC0, weight NCDHW -> FZ_3D.
 * dst: input -> AL1(NC1HWC0), weight -> BL1(FZ)
 * src: input -> GM(NCHW), weigh -> GM(NCHW)
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_d_offset, weight -> src_d_offset
 * offset3: input -> src_h_offset, weight -> 0
 * offset4: input -> src_w_offset, weight -> 0
 * isFmap: true -> input, false -> weight
 */
template <bool isFmap, typename T, typename U>
INLINE void TLoadConv3DDN2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const ShapeInfo& srcShapeInfo)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t srcC = GetConvShape<CONV_IDX_1>(src);
    int64_t srcD = GetConvShape<CONV_IDX_2>(src);
    int64_t srcH = GetConvShape<CONV_IDX_3>(src);
    int64_t srcW = GetConvShape<CONV_IDX_4>(src);
    int64_t dstShape0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstShape1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstShape2 = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstShape3 = GetConvShape<CONV_IDX_3>(dst);
    int64_t srcStrideN = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStrideC = GetConvStride<CONV_IDX_1>(src);
    int64_t srcStrideD = GetConvStride<CONV_IDX_2>(src);
    int64_t srcStrideH = GetConvStride<CONV_IDX_3>(src);
    int64_t srcStrideW = GetConvStride<CONV_IDX_4>(src);
    constexpr auto stcDstShape0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto stcDstShape1 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    constexpr auto stcDstShape2 = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;
    constexpr auto stcDstShape3 = Std::tuple_element<CONV_IDX_3, typename T::TileShape>::type::value;
    ShapeInfo shapeInfo = {srcC, srcD, srcH, srcW};
    using shapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    using strideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim5, strideDim5, pto::Layout::NCDHW>;
    int64_t gmOffset = CalLoadOffsetNCDHW<isFmap>(shapeInfo, offsetInfo);
    globalData srcGlobal((__gm__ typename U::Type*)(src.GetAddr() + gmOffset),
                         shapeDim5(srcShapeInfo.shape0, srcShapeInfo.shape1, srcShapeInfo.shape2, srcShapeInfo.shape3,
                                   srcShapeInfo.shape4),
                         strideDim5(srcStrideN, srcStrideC, srcStrideD, srcStrideH, srcStrideW));
    if constexpr (isFmap) {
        int64_t dstShape4 = GetConvShape<CONV_IDX_4>(dst);
        constexpr auto stcDstShape4 = Std::tuple_element<CONV_IDX_4, typename T::TileShape>::type::value;
        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * stcDstShape3 * stcDstShape4 *
                                    BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::NDC1HWC0,
                                       pto::ConvTileShape<-1, -1, -1, -1, -1, c0Size>>;
        tileData dstL1(dstShape0, dstShape1, dstShape2, dstShape3, dstShape4);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    } else {
        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::FRACTAL_Z_3D,
                                       pto::ConvTileShape<-1, -1, -1, -1, 1>>;
        tileData dstL1(dstShape0, dstShape1, dstShape2, dstShape3);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    }
}

template <bool isConv3D, bool isFmap, typename T, typename U>
INLINE void TLoadConvDN2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const ShapeInfo& srcShapeInfo)
{
    if constexpr (isConv3D) {
        TLoadConv3DDN2NZ<isFmap>(dst, src, offsetInfo, srcShapeInfo);
    } else {
        TLoadConv2DDN2NZ<isFmap>(dst, src, offsetInfo, srcShapeInfo);
    }
}

#endif // TILEOP_TILE_OPERATOR_CONV_LOAD_DN2NZ_IMPL__H
