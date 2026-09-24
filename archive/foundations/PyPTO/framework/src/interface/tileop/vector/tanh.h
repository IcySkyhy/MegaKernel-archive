/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tanh.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_TANH__H
#define TILEOP_TILE_OPERATOR_TANH__H
#include "unary.h"
#include <type_traits>

constexpr float TANH_POLY_015 = 0.0157396831f;
constexpr float TANH_POLY_NEG_052 = -0.0523039624f;
constexpr float TANH_POLY_133 = 0.133152977f;
constexpr float TANH_POLY_NEG_0333 = -0.333327681f;
constexpr float TANH_THRESHOLD = 0.55f;
constexpr float TANH_CLIP_VALUE = 20.0f;
constexpr float TANH_TWO = 2.0f;

template <typename LastUse, typename T, typename DstTile, typename SrcTile, typename TmpTile, typename CmpTile,
          typename AddrUBTile>
TILEOP void TanhFP32(DstTile dstTile, SrcTile srcTile, TmpTile tmpTile, TmpTile tmpTile2, CmpTile cmpTile,
                     AddrUBTile startAddrUBTile)
{
    pto::TMUL(tmpTile, srcTile, srcTile);
    SyncV();

    pto::TMULS(dstTile, tmpTile, static_cast<T>(TANH_POLY_015));
    SyncV();

    pto::TADDS(dstTile, dstTile, static_cast<T>(TANH_POLY_NEG_052));
    SyncV();

    pto::TMUL(dstTile, dstTile, tmpTile);
    SyncV();
    pto::TADDS(dstTile, dstTile, static_cast<T>(TANH_POLY_133));
    SyncV();

    pto::TMUL(dstTile, dstTile, tmpTile);
    SyncV();
    pto::TADDS(dstTile, dstTile, static_cast<T>(TANH_POLY_NEG_0333));
    SyncV();

    pto::TMUL(dstTile, dstTile, tmpTile);
    SyncV();

    pto::TMUL(dstTile, dstTile, srcTile);
    SyncV();
    pto::TADD(dstTile, dstTile, srcTile);
    SyncV();

    pto::TABS(tmpTile, srcTile);
    SyncV();

    pto::TMINS(srcTile, srcTile, static_cast<T>(TANH_CLIP_VALUE));
    SyncV();

    pto::TMULS(srcTile, srcTile, static_cast<T>(TANH_TWO));
    SyncV();

    pto::TEXP(srcTile, srcTile);
    SyncV();

    pto::TADDS(tmpTile2, srcTile, static_cast<T>(-1.0f));
    SyncV();

    pto::TADDS(srcTile, srcTile, static_cast<T>(1.0f));
    SyncV();

    pto::TDIV(tmpTile2, tmpTile2, srcTile);
    SyncV();

    pto::TCMPS(cmpTile, tmpTile, static_cast<T>(TANH_THRESHOLD), pto::CmpMode::LT);
    SyncV();

    pto::TSEL(dstTile, cmpTile, dstTile, tmpTile2, startAddrUBTile);
}

template <typename LastUse, typename T, typename DstTile, typename SrcTile, typename TmpTile, typename CmpTile,
          typename AddrUBTile>
TILEOP void TanhCast(DstTile dstTile, SrcTile srcTile, TmpTile tmpTile1, TmpTile tmpTile2, TmpTile tmpTile3,
                     TmpTile tmpTile4, CmpTile cmpTile, AddrUBTile startAddrUBTile)
{
    pto::TCVT(tmpTile1, srcTile, pto::RoundMode::CAST_NONE);
    SyncV();

    pto::TMUL(tmpTile2, tmpTile1, tmpTile1);
    SyncV();

    pto::TMULS(tmpTile3, tmpTile2, static_cast<float>(TANH_POLY_015));
    SyncV();

    pto::TADDS(tmpTile3, tmpTile3, static_cast<float>(TANH_POLY_NEG_052));
    SyncV();

    pto::TMUL(tmpTile3, tmpTile3, tmpTile2);
    SyncV();
    pto::TADDS(tmpTile3, tmpTile3, static_cast<float>(TANH_POLY_133));
    SyncV();

    pto::TMUL(tmpTile3, tmpTile3, tmpTile2);
    SyncV();
    pto::TADDS(tmpTile3, tmpTile3, static_cast<float>(TANH_POLY_NEG_0333));
    SyncV();

    pto::TMUL(tmpTile3, tmpTile3, tmpTile2);
    SyncV();

    pto::TMUL(tmpTile3, tmpTile3, tmpTile1);
    SyncV();
    pto::TADD(tmpTile3, tmpTile3, tmpTile1);
    SyncV();

    pto::TABS(tmpTile2, tmpTile1);
    SyncV();

    pto::TMINS(tmpTile1, tmpTile1, static_cast<float>(TANH_CLIP_VALUE));
    SyncV();

    pto::TMULS(tmpTile1, tmpTile1, static_cast<float>(TANH_TWO));
    SyncV();

    pto::TEXP(tmpTile1, tmpTile1);
    SyncV();

    pto::TADDS(tmpTile4, tmpTile1, static_cast<float>(-1.0f));
    SyncV();

    pto::TADDS(tmpTile1, tmpTile1, static_cast<float>(1.0f));
    SyncV();

    pto::TDIV(tmpTile4, tmpTile4, tmpTile1);
    SyncV();

    pto::TCMPS(cmpTile, tmpTile2, static_cast<float>(TANH_THRESHOLD), pto::CmpMode::LT);
    SyncV();

    pto::TSEL(tmpTile3, cmpTile, tmpTile3, tmpTile4, startAddrUBTile);
    SyncV();

    pto::TCVT(dstTile, tmpTile3, pto::RoundMode::CAST_NONE);
}

#define OP_TILE_OP_TANH Ttanh
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1, typename T3>
TILEOP void TTanh(T0 dst, T1 src, T3 tmp)
{
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
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

    using DstTile = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1,
                              -1>;
    using SrcTile = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1,
                              -1>;

    DstTile dstTile(dstShape3, dstShape4);
    SrcTile srcTile(srcExecShape3, srcExecShape4);

    constexpr unsigned alignUint8 = 32;
    constexpr unsigned addressUsed = 4;
    using AddrUBTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, alignUint8, pto::BLayout::RowMajor, -1, -1>;
    AddrUBTile startAddrUBTile(1, addressUsed);

    if constexpr (std::is_same<typename T0::Type, float>::value) {
        constexpr auto ALIGN32FP32 = 8;
        constexpr auto tmpTileW = (dstTileW + ALIGN32FP32 - 1) / ALIGN32FP32 * ALIGN32FP32;
        constexpr auto cmpTileW = (dstTileW / 8 + alignUint8 - 1) / alignUint8 * alignUint8;
        using TmpTile = pto::Tile<pto::TileType::Vec, float, dstTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
        using CmpTile = pto::Tile<pto::TileType::Vec, uint8_t, dstTileH, cmpTileW, pto::BLayout::RowMajor, -1, -1>;

        auto tmpOffset = dstTileH * tmpTileW;
        auto cmpOffset = dstTileH * cmpTileW;
        TmpTile tmpTile1(dstShape3, dstShape4);
        TmpTile tmpTile2(dstShape3, dstShape4);
        CmpTile cmpTile(dstTileH, dstShape4 / 8);
        pto::TASSIGN(tmpTile1, (uint64_t)(tmp.GetAddr()));
        pto::TASSIGN(tmpTile2, (uint64_t)(tmp.GetAddr() + tmpOffset * sizeof(float)));
        pto::TASSIGN(cmpTile, (uint64_t)(tmp.GetAddr() + 2 * tmpOffset * sizeof(float)));
        pto::TASSIGN(startAddrUBTile,
                     (uint64_t)(tmp.GetAddr() + 2 * tmpOffset * sizeof(float) + cmpOffset * sizeof(uint8_t)));

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    TanhFP32<LastUse, typename T0::Type, DstTile, SrcTile, TmpTile, CmpTile, AddrUBTile>(
                        dstTile, srcTile, tmpTile1, tmpTile2, cmpTile, startAddrUBTile);
                }
            }
        }
    } else if constexpr (std::is_same<typename T0::Type, half>::value ||
                         std::is_same<typename T0::Type, bfloat16_t>::value) {
        constexpr auto ALIGN32FP32 = 8;
        constexpr auto tmpTileW = (srcTileW + ALIGN32FP32 - 1) / ALIGN32FP32 * ALIGN32FP32;
        constexpr auto cmpTileW = (srcTileW / 8 + alignUint8 - 1) / alignUint8 * alignUint8;
        using TmpTile = pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
        using CmpTile = pto::Tile<pto::TileType::Vec, uint8_t, srcTileH, cmpTileW, pto::BLayout::RowMajor, -1, -1>;

        auto tmpOffset = srcTileH * tmpTileW;
        auto cmpOffset = srcTileH * cmpTileW;
        TmpTile tmpTile1(srcExecShape3, srcExecShape4);
        TmpTile tmpTile2(srcExecShape3, srcExecShape4);
        TmpTile tmpTile3(srcExecShape3, srcExecShape4);
        TmpTile tmpTile4(srcExecShape3, srcExecShape4);
        CmpTile cmpTile(srcTileH, srcExecShape4 / 8);
        pto::TASSIGN(tmpTile2, (uint64_t)(tmp.GetAddr()));
        pto::TASSIGN(tmpTile1, (uint64_t)(tmp.GetAddr() + tmpOffset * sizeof(float)));
        pto::TASSIGN(tmpTile3, (uint64_t)(tmp.GetAddr() + 2 * tmpOffset * sizeof(float)));
        pto::TASSIGN(tmpTile4, (uint64_t)(tmp.GetAddr() + 3 * tmpOffset * sizeof(float)));
        pto::TASSIGN(cmpTile, (uint64_t)(tmp.GetAddr() + 4 * tmpOffset * sizeof(float)));
        pto::TASSIGN(startAddrUBTile,
                     (uint64_t)(tmp.GetAddr() + 4 * tmpOffset * sizeof(float) + cmpOffset * sizeof(uint8_t)));

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    TanhCast<LastUse, typename T0::Type, DstTile, SrcTile, TmpTile, CmpTile, AddrUBTile>(
                        dstTile, srcTile, tmpTile1, tmpTile2, tmpTile3, tmpTile4, cmpTile, startAddrUBTile);
                }
            }
        }
    }
}

#endif
