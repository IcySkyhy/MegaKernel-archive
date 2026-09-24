/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sign.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_SIGN__H
#define TILEOP_TILE_OPERATOR_SIGN__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <type_traits>

template <typename LastUse, typename T, typename DstTile, typename SrcTile>
TILEOP void SignInt(DstTile dstTile, SrcTile srcTile)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    pto::TMINS(dstTile, srcTile, static_cast<T>(1));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMAXS(dstTile, dstTile, static_cast<T>(-1));
}

template <typename LastUse, typename T, typename DstTile, typename SrcTile, typename TmpTile>
TILEOP void SignIntCast(DstTile dstTile, SrcTile srcTile, TmpTile tmpTile)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    pto::TCVT(tmpTile, srcTile, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMINS(tmpTile, tmpTile, static_cast<half>(5.960464e-08f));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMAXS(tmpTile, tmpTile, static_cast<half>(-5.960464e-08f));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMULS(tmpTile, tmpTile, static_cast<half>(4.096000e+03f));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMULS(tmpTile, tmpTile, static_cast<half>(4.096000e+03f));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCVT(dstTile, tmpTile, pto::RoundMode::CAST_NONE);
}

template <typename LastUse, typename T, typename DstTile, typename SrcTile, typename RightTile, typename MaskTile,
          typename ScalarTmpTile>
TILEOP void SignFloat(DstTile dstTile, SrcTile srcTile, RightTile rightTile, MaskTile maskTile,
                      ScalarTmpTile scalarTmpTile)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;

    pto::TCMPS(maskTile, srcTile, static_cast<T>(0.0f), pto::CmpMode::LT);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TEXPANDS(rightTile, static_cast<T>(1.0f));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSELS(rightTile, maskTile, rightTile, scalarTmpTile, static_cast<T>(0.0f));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCMPS(maskTile, srcTile, static_cast<T>(0.0f), pto::CmpMode::GT);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TEXPANDS(dstTile, static_cast<T>(1.0f));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSELS(dstTile, maskTile, dstTile, scalarTmpTile, static_cast<T>(0.0f));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(dstTile, dstTile, rightTile);
}

template <typename LastUse, typename T, typename DstTile, typename SrcTile, typename WorkTile, typename MaskTile,
          typename ScalarTmpTile>
TILEOP void SignImpl(DstTile dstTile, SrcTile srcTile, WorkTile workTile, MaskTile maskTile,
                     ScalarTmpTile scalarTmpTile)
{
    if constexpr (std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value) {
        SignInt<LastUse, T, DstTile, SrcTile>(dstTile, srcTile);
    } else if constexpr (std::is_same<T, half>::value || std::is_same<T, float>::value) {
        SignFloat<LastUse, T>(dstTile, srcTile, workTile, maskTile, scalarTmpTile);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        static_assert(!std::is_same<T, bfloat16_t>::value, "BF16 Sign must be converted to FP32 by AutoCast.");
    } else if constexpr (std::is_same<T, int8_t>::value) {
        SignIntCast<LastUse, T>(dstTile, srcTile, workTile);
    }
    return;
}

#define OP_TILE_OP_SIGN TSign
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1, typename T3>
TILEOP void TSign(T0 dst, T1 src, T3 tmp)
{
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto tmpLayout = tmp.GetLayout();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);

    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    auto srcExecShape3 = GetElementwiseOperandExecShapeDim<3, expectSize>(dst, src);
    auto srcExecShape4 = GetElementwiseOperandExecShapeDim<4, expectSize>(dst, src);

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();

    using SrcExecConfig = ElementwiseOperandExecConfig<T0, T1>;
    constexpr auto srcTileH = SrcExecConfig::tileH;
    constexpr auto srcTileW = SrcExecConfig::tileW;

    using DstType = typename T0::Type;
    using WorkType = std::conditional_t<std::is_same<DstType, int8_t>::value, half, DstType>;
    constexpr auto align32 = TileOp::BLOCK_SIZE / sizeof(WorkType);
    constexpr auto tmpTileW = (srcTileW + align32 - 1) / align32 * align32;
    constexpr size_t workBlockBytes = srcTileH * tmpTileW * sizeof(WorkType);

    using DstTile = pto::Tile<pto::TileType::Vec, DstType, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcTile = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1,
                              -1>;
    using WorkTile = pto::Tile<pto::TileType::Vec, WorkType, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTile = pto::Tile<pto::TileType::Vec, uint8_t, srcTileH, tmpTileW * sizeof(WorkType),
                               pto::BLayout::RowMajor, -1, -1>;
    using ScalarTmpTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, TileOp::BLOCK_SIZE, pto::BLayout::RowMajor, -1, -1>;

    DstTile dstTile(dstShape3, dstShape4);
    SrcTile srcTile(srcExecShape3, srcExecShape4);
    WorkTile workTile(srcExecShape3, srcExecShape4);
    MaskTile maskTile(srcExecShape3, srcExecShape4);
    ScalarTmpTile scalarTmpTile(1, TileOp::BLOCK_SIZE);

    pto::TASSIGN(workTile, (uint64_t)(tmp.GetAddr()));
    pto::TASSIGN(maskTile, (uint64_t)(tmp.GetAddr() + workBlockBytes));
    pto::TASSIGN(scalarTmpTile, (uint64_t)(tmp.GetAddr() + 2 * workBlockBytes));

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                SignImpl<LastUse, DstType>(dstTile, srcTile, workTile, maskTile, scalarTmpTile);
            }
        }
    }
}

#endif
