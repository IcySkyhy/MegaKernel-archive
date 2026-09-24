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
 * \file conv_load_nz2nz_impl.h
 * \brief Copy input data from DDR to L1 with NZ2NZ (chip-independent)
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_LOAD_NZ2NZ_IMPL__H
#define TILEOP_TILE_OPERATOR_CONV_LOAD_NZ2NZ_IMPL__H
#include "conv_utils.h"

/**
 * Copy input data from DDR to L1 with NZ2NZ
 *
 * dst: fmap -> AL1(NC1HWC0: [N, C1, H, W, C0]), weight -> BL1(FractalZ: [c1hw, cout1, n0, c0])
 * src: fmap -> GM(NC1HWC0), weigh -> GM(FractalZ)
 * offset0: fmap -> N, weight -> c1hw
 * offset1: fmap -> C1,  weight -> cout1
 * offset2: fmap -> H, weight -> 0
 * offset3: fmap -> W, weight -> 0
 * offset4: fmap -> 0, weight -> 0
 * isFmap: true -> fmap, false -> weight
 */
template <bool isFmap, typename T, typename U>
INLINE void TLoadConv2DNZ2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const ShapeInfo& srcShapeInfo)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t dstShape0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstShape1 = GetConvShape<CONV_IDX_1>(dst);
    constexpr auto stcDstShape0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto stcDstShape1 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    constexpr auto stcDstShape2 = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;
    int64_t srcStride0 = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStride1 = GetConvStride<CONV_IDX_1>(src);
    int64_t srcStride2 = GetConvStride<CONV_IDX_2>(src);
    int64_t srcStride3 = GetConvStride<CONV_IDX_3>(src);

    if constexpr (isFmap) {
        int64_t dstShape2 = GetConvShape<CONV_IDX_2>(dst);
        int64_t dstShape3 = GetConvShape<CONV_IDX_3>(dst);
        constexpr auto stcDstShape3 = Std::tuple_element<CONV_IDX_3, typename T::TileShape>::type::value;
        int64_t gmOffset = offsetInfo.offset0 * srcStride0 + offsetInfo.offset1 * srcStride1 +
                           offsetInfo.offset2 * srcStride2 + offsetInfo.offset3 * srcStride3;

        // srcShapeInfo: {shape0(n), shape1(c1), shape2(h), shape3(w), shape4(c0)}
        using shapeDim = pto::Shape<1, -1, -1, -1, -1>;
        using strideDim = pto::Stride<-1, -1, -1, -1, 1>;
        using globalData = pto::GlobalTensor<typename U::Type, shapeDim, strideDim, pto::Layout::NC1HWC0>;
        globalData srcGlobal((__gm__ typename U::Type*)(src.GetAddr() + gmOffset),
                             shapeDim(srcShapeInfo.shape1, srcShapeInfo.shape2, srcShapeInfo.shape3, c0Size),
                             strideDim(srcStride0, srcStride1, srcStride2, srcStride3));

        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * stcDstShape3 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::NC1HWC0,
                                       pto::ConvTileShape<-1, -1, -1, -1, c0Size>>;
        tileData dstL1(dstShape0, dstShape1, dstShape2, dstShape3);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    } else {
        // srcShapeInfo: {shape0(c1hw), shape1(n1), shape2(n0), shape3(c0), shape4(0)}
        int64_t gmOffset = offsetInfo.offset0 * srcStride0 + offsetInfo.offset1 * srcStride1;
        using shapeDim = pto::Shape<1, -1, -1, MKN_N_VALUE, c0Size>;
        using strideDim = pto::Stride<1, -1, -1, -1, -1>;
        using globalData = pto::GlobalTensor<typename U::Type, shapeDim, strideDim, pto::Layout::FRACTAL_Z>;
        globalData srcGlobal((__gm__ typename U::Type*)(src.GetAddr() + gmOffset),
                             shapeDim(srcShapeInfo.shape0, srcShapeInfo.shape1),
                             strideDim(srcStride0, srcStride1, srcStride2, srcStride3));
        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::FRACTAL_Z,
                                       pto::ConvTileShape<-1, -1, MKN_N_VALUE, c0Size>>;
        tileData dstL1(dstShape0, dstShape1);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    }
}

/**
 * Copy input data from DDR to L1 with NZ2NZ
 *
 * dst: fmap -> AL1(NDC1HWC0: [N, D, C1, H, W, C0]), weight -> BL1(FractalZ_3D: [c1dhw, cout1, n0, c0])
 * src: fmap -> GM(NDC1HWC0), weigh -> GM(FractalZ_3D)
 * offset0: fmap -> N, weight -> c1dhw
 * offset1: fmap -> D,  weight -> cout1
 * offset2: fmap -> C1,  weight -> 0
 * offset3: fmap -> H, weight -> 0
 * offset4: fmap -> W, weight -> 0
 * isFmap: true -> fmap, false -> weight
 */
template <bool isFmap, typename T, typename U>
INLINE void TLoadConv3DNZ2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const ShapeInfo& srcShapeInfo)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    constexpr auto stcDstShape0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto stcDstShape1 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    constexpr auto stcDstShape2 = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;
    int64_t dstShape0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstShape1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t srcStride0 = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStride1 = GetConvStride<CONV_IDX_1>(src);
    int64_t srcStride2 = GetConvStride<CONV_IDX_2>(src);
    int64_t srcStride3 = GetConvStride<CONV_IDX_3>(src);

    if constexpr (isFmap) {
        constexpr auto stcDstShape3 = Std::tuple_element<CONV_IDX_3, typename T::TileShape>::type::value;
        constexpr auto stcDstShape4 = Std::tuple_element<CONV_IDX_4, typename T::TileShape>::type::value;
        int64_t dstShape2 = GetConvShape<CONV_IDX_2>(dst);
        int64_t dstShape3 = GetConvShape<CONV_IDX_3>(dst);
        int64_t dstShape4 = GetConvShape<CONV_IDX_4>(dst);
        int64_t srcStride4 = GetConvStride<CONV_IDX_4>(src);
        int64_t gmOffset = offsetInfo.offset0 * srcStride0 + offsetInfo.offset1 * srcStride1 +
                           offsetInfo.offset2 * srcStride2 + offsetInfo.offset3 * srcStride3 +
                           offsetInfo.offset4 * srcStride4;
        // Shape: N=1, D, C1, H, W, C0(由于Shape只支持5维，且C0可以根据dtype计算得出，所以这里不配置C0)
        using shapeDim = pto::Shape<1, -1, -1, -1, -1>;
        using strideDim = pto::Stride<-1, -1, -1, -1, -1>;
        using globalData = pto::GlobalTensor<typename U::Type, shapeDim, strideDim, pto::Layout::NDC1HWC0>;
        globalData srcGlobal(
            (__gm__ typename U::Type*)(src.GetAddr() + gmOffset),
            shapeDim(srcShapeInfo.shape1, srcShapeInfo.shape2, srcShapeInfo.shape3, srcShapeInfo.shape4),
            strideDim(srcStride0, srcStride1, srcStride2, srcStride3, srcStride4));

        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * stcDstShape3 * stcDstShape4 *
                                    BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::NDC1HWC0,
                                       pto::ConvTileShape<-1, -1, -1, -1, -1, c0Size>>;
        tileData dstL1(dstShape0, dstShape1, dstShape2, dstShape3, dstShape4);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    } else {
        int64_t gmOffset = offsetInfo.offset0 * srcStride0 + offsetInfo.offset1 * srcStride1;
        // Shape: 1(占位), dc1hw, cout1, n0=16, c0
        using shapeDim = pto::Shape<1, -1, -1, MKN_N_VALUE, c0Size>;
        using strideDim = pto::Stride<1, -1, -1, -1, -1>;
        using globalData = pto::GlobalTensor<typename U::Type, shapeDim, strideDim, pto::Layout::FRACTAL_Z_3D>;
        globalData srcGlobal((__gm__ typename U::Type*)(src.GetAddr() + gmOffset),
                             shapeDim(srcShapeInfo.shape0, srcShapeInfo.shape1),
                             strideDim(srcStride0, srcStride1, srcStride2, srcStride3));
        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::FRACTAL_Z_3D,
                                       pto::ConvTileShape<-1, -1, MKN_N_VALUE, c0Size>>;
        tileData dstL1(dstShape0, dstShape1);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    }
}

template <bool isConv3D, bool isFmap, typename T, typename U>
INLINE void TLoadConvNZ2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const ShapeInfo& srcShapeInfo)
{
    if constexpr (isConv3D) {
        TLoadConv3DNZ2NZ<isFmap>(dst, src, offsetInfo, srcShapeInfo);
    } else {
        TLoadConv2DNZ2NZ<isFmap>(dst, src, offsetInfo, srcShapeInfo);
    }
}

#endif // TILEOP_TILE_OPERATOR_CONV_LOAD_NZ2NZ_IMPL__H
