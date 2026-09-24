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
 * \file conv_load2d_load3d_impl.h
 * \brief Copy data from L1 to L0A (img2col for fmap) / L0B (extract for weight)
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_LOAD2D_LOAD3D_IMPL__H
#define TILEOP_TILE_OPERATOR_CONV_LOAD2D_LOAD3D_IMPL__H
#include "conv_utils.h"

template <bool isConv3D, typename srcTensorType>
INLINE void SetConvTileParams(srcTensorType& l1, const uint8_t* padValues, const int64_t& filterH,
                              const int64_t& filterW, const int64_t& dilationH, const int64_t& dilationW,
                              const int64_t& strideH, const int64_t& strideW, const int64_t& padValue,
                              const int64_t& mL0, const int64_t& kL0, const int64_t& shape1, const int64_t& shape2,
                              const int64_t& shape3, const int64_t& shape4, const int64_t& c0Size,
                              const int64_t& repeatStride, const int64_t& repeatTime, const int64_t& dstStride)
{
    l1.SetPadListArray(padValues);
    l1.SetFilterH(filterH);
    l1.SetFilterW(filterW);
    l1.SetDilationH(dilationH);
    l1.SetDilationW(dilationW);
    l1.SetStrideH(strideH);
    l1.SetStrideW(strideW);
    l1.SetPadValue(padValue);
    l1.SetRepeatMode(NUM0);
    l1.SetRepeatTime(repeatTime);
    l1.SetRepeatStride(repeatStride / BLOCK_CUBE_M_N);
#if defined PTO_NPU_ARCH_A5
    l1.SetDstStride(mL0 / BLOCK_CUBE_M_N);
    l1.SetDstMposition(NUM0);
#endif

    if constexpr (isConv3D) {
        l1.SetFmapH(static_cast<uint16_t>(shape3));
        l1.SetFmapW(static_cast<uint16_t>(shape4));
        l1.SetChannelSize(shape2 * c0Size);
    } else {
        l1.SetFmapH(static_cast<uint16_t>(shape2));
        l1.SetFmapW(static_cast<uint16_t>(shape3));
        l1.SetChannelSize(shape1 * shape4); // c1 * c0
    }
}

template <bool isConv3D, typename T, typename U>
TILEOP void TLoad3D(T& dst, U& src, const int64_t& mPos, const int64_t& kPos, const int64_t& padLeft,
                    const int64_t& padRight, const int64_t& padTop, const int64_t& padBottom, const int64_t& padValue,
                    const int64_t& filterH, const int64_t& filterW, const int64_t& dilationH, const int64_t& dilationW,
                    const int64_t& strideH, const int64_t& strideW, const int64_t& repeatStride,
                    const int64_t& repeatTime, const int64_t& dstStride)
{
    // 2D： n c1 h w c0
    // 3D： n d c1 h w
    constexpr auto static0 = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto static1 = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr auto static2 = Std::tuple_element<CONV_IDX_2, typename U::TileShape>::type::value;
    constexpr auto static3 = Std::tuple_element<CONV_IDX_3, typename U::TileShape>::type::value;
    constexpr auto static4 = Std::tuple_element<CONV_IDX_4, typename U::TileShape>::type::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    constexpr auto elements = static0 * static1 * static2 * static3 * static4;
    int64_t shape0 = GetConvShape<CONV_IDX_0>(src);
    int64_t shape1 = GetConvShape<CONV_IDX_1>(src);
    int64_t shape2 = GetConvShape<CONV_IDX_2>(src);
    int64_t shape3 = GetConvShape<CONV_IDX_3>(src);
    int64_t shape4 = GetConvShape<CONV_IDX_4>(src);
    using srcTensor = select_srcTensor<isConv3D, U, elements, c0Size>;
    srcTensor l1(shape0, shape1, shape2, shape3, shape4);

    constexpr auto staticML0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto staticKL0 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    int64_t mL0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t kL0 = GetConvShape<CONV_IDX_1>(dst);
    using dstTensor = pto::TileLeft<typename T::Type, staticML0, staticKL0, -1, -1>;
    dstTensor l0(dstStride, kL0);

    uint8_t values[4] = {static_cast<uint8_t>(padLeft), static_cast<uint8_t>(padRight), static_cast<uint8_t>(padTop),
                         static_cast<uint8_t>(padBottom)};
    SetConvTileParams<isConv3D>(l1, values, filterH, filterW, dilationH, dilationW, strideH, strideW, padValue, mL0,
                                kL0, shape1, shape2, shape3, shape4, c0Size, repeatStride, repeatTime, dstStride);

    pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()));
    pto::TIMG2COL<dstTensor, srcTensor, pto::SetFmatrixMode::FMATRIX_A_AUTO>(l0, l1, mPos, kPos);
}

template <typename T, typename U>
TILEOP void TLoad2D(T& dst, U& src, const int64_t& indexRow, const int64_t& indexCol)
{
    constexpr auto staticC1HW = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto staticN1 = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr auto staticN0 = Std::tuple_element<CONV_IDX_2, typename U::TileShape>::type::value;
    constexpr auto staticC0 = Std::tuple_element<CONV_IDX_3, typename U::TileShape>::type::value;
    constexpr auto bufferSize = staticC1HW * staticN1 * staticN0 * staticC0;
    int64_t c1hw = GetConvShape<CONV_IDX_0>(src);
    int64_t n1 = GetConvShape<CONV_IDX_1>(src);
    int64_t n0 = GetConvShape<CONV_IDX_2>(src);
    int64_t c0 = GetConvShape<CONV_IDX_3>(src);
    using srcTensor = pto::ConvTile<pto::TileType::Mat, typename U::Type, bufferSize, pto::Layout::FRACTAL_Z,
                                    pto::ConvTileShape<-1, -1, staticN0, staticC0>>;
    srcTensor l1(c1hw, n1);

    constexpr auto staticKL0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto staticNL0 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    int64_t kL0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t nL0 = GetConvShape<CONV_IDX_1>(dst);
    using dstTensor = pto::TileRight<typename T::Type, staticKL0, staticNL0, -1, -1>;
    dstTensor l0(kL0, nL0);

    pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()));
    pto::TEXTRACT<dstTensor, srcTensor>(l0, l1, indexRow, indexCol);
}
#endif // TILEOP_TILE_OPERATOR_CONV_LOAD2D_LOAD3D_IMPL__H
