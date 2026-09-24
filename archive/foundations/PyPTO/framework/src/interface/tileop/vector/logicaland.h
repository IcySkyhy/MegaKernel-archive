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
 * \file logicaland.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_LOGICALAND__H
#define TILEOP_TILE_OPERATOR_LOGICALAND__H
#include <type_traits>

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename Type>
using CVT_TYPE = std::conditional_t<
    std::is_same_v<Type, float> || std::is_same_v<Type, int32_t> || std::is_same_v<Type, int64_t>, float, half>;

template <typename T, typename U>
using FINAL_TYPE = std::conditional_t<std::is_same_v<CVT_TYPE<T>, CVT_TYPE<U>>, CVT_TYPE<T>, float>;

template <typename T, typename U, typename Cvt>
TILEOP T& StandardizedSrcTile([[maybe_unused]] T& dst, U& src, Cvt& cvt)
{
    if constexpr (std::is_same_v<typename T::DType, typename U::DType>) {
        return src;
    } else {
        if constexpr (!std::is_same_v<typename T::DType, float> || sizeof(typename U::DType) > 1) {
            pto::TCVT(dst, src, pto::RoundMode::CAST_NONE);
        } else {
            pto::TCVT(cvt, src, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
            pipe_barrier(PIPE_V);
#endif
            pto::TCVT(dst, cvt, pto::RoundMode::CAST_NONE);
        }
        return dst;
    }
}

template <typename T, typename U, typename R, typename V>
TILEOP void CalculateLogicalTile(T& dst, U& src, U& zeros, U& ones, R& res, V& startAddrUBTile)
{
    pto::TCMP(res, src, zeros, pto::CmpMode::EQ);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSEL(dst, res, zeros, ones, startAddrUBTile);
}

template <typename T, typename U, typename R, typename Cvt, typename V>
TILEOP void CalculateSrcLogicalTile(T& dst, U& src, T& zeros, T& ones, R& res, Cvt& cvt, V& startAddrUBTile)
{
    auto tile = StandardizedSrcTile(dst, src, cvt);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    CalculateLogicalTile(dst, tile, zeros, ones, res, startAddrUBTile);
}

template <typename T, typename U, typename TMP>
TILEOP void SelectLogicalResult(T& dst, U& src1, U& src2, TMP& tmp)
{
    pto::TMIN(src1, src1, src2);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    if constexpr (std::is_same_v<typename TMP::DType, typename U::DType>) {
        pto::TCVT(dst, src1, pto::RoundMode::CAST_NONE);
    } else {
        pto::TCVT(tmp, src1, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        pto::TCVT(dst, tmp, pto::RoundMode::CAST_NONE);
    }
}

template <typename T, typename U1, typename U2, typename L, typename Res, typename Cvt, typename V>
TILEOP void LogicalAndImpl(T& dst, U1& src1, U2& src2, L& tmp1, L& tmp2, L& ones, L& zeros, Res& res1, Res& res2,
                           Cvt& cvt, V& startAddrUBTile)
{
    CalculateSrcLogicalTile(tmp1, src1, zeros, ones, res1, cvt, startAddrUBTile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    CalculateSrcLogicalTile(tmp2, src2, zeros, ones, res2, cvt, startAddrUBTile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    SelectLogicalResult(dst, tmp1, tmp2, cvt);
}

TILEOP uint64_t GenTmpTileAddr(uint64_t addr, uint64_t offset)
{
    constexpr uint32_t ALIGN_SIZE = 32;
    uintptr_t start = reinterpret_cast<uintptr_t>(addr + offset);
    return (start + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TLogicalAnd(T0 dst, T1 src1, T2 src2, T3 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    using DType = FINAL_TYPE<typename T1::Type, typename T2::Type>;
    constexpr auto COUNT_MAX = 64;
    constexpr auto ALIGN_SIZE = 32;
    constexpr auto BITS_PER_BYTE = 8;
    constexpr auto CMP_BYTE_SIZE = (COUNT_MAX + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
    constexpr auto TMP_OFFSET = sizeof(DType) * COUNT_MAX;

    using DstTileTensor = TileTensor<typename T0::Type, LocalLayout2Dim<1, COUNT_MAX>, Hardware::UB>;
    using Src1TileTensor = TileTensor<typename T1::Type, LocalLayout2Dim<1, COUNT_MAX>, Hardware::UB>;
    using Src2TileTensor = TileTensor<typename T2::Type, LocalLayout2Dim<1, COUNT_MAX>, Hardware::UB>;
    using TmpTileTensor = TileTensor<DType, LocalLayout2Dim<1, COUNT_MAX>, Hardware::UB>;
    using CvtTileTensor = TileTensor<half, LocalLayout2Dim<1, COUNT_MAX>, Hardware::UB>;
    using StartAddrUBTile = TileTensor<uint8_t, LocalLayout2Dim<1, ALIGN_SIZE>, Hardware::UB>;
    // Tile shape of pto tile must 32 byte align
    constexpr auto CMP_TILE = ((CMP_BYTE_SIZE + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    using CmpBitsTileTensor = TileTensor<uint8_t, StaticLayout2Dim<1, CMP_BYTE_SIZE, 1, CMP_TILE>, Hardware::UB>;

    using DstTile = PtoTile<DstTileTensor>;
    using Src1Tile = PtoTile<Src1TileTensor>;
    using Src2Tile = PtoTile<Src2TileTensor>;
    using TmpTile = PtoTile<TmpTileTensor>;
    using CvtTile = PtoTile<CvtTileTensor>;
    using SauTile = PtoTile<StartAddrUBTile>;

    auto cmp1Addr = (uint64_t)tmp.GetAddr();
    auto cmp2Addr = GenTmpTileAddr(cmp1Addr, CMP_BYTE_SIZE);
    auto zeroAddr = GenTmpTileAddr(cmp2Addr, CMP_BYTE_SIZE);
    auto oneAddr = GenTmpTileAddr(zeroAddr, TMP_OFFSET);
    auto tmp1Addr = GenTmpTileAddr(oneAddr, TMP_OFFSET);
    auto tmp2Addr = GenTmpTileAddr(tmp1Addr, TMP_OFFSET);
    auto sauAddr = GenTmpTileAddr(tmp2Addr, TMP_OFFSET);
    auto cvtAddr = GenTmpTileAddr(sauAddr, ALIGN_SIZE);

    auto cmp1Tile = PtoTile<CmpBitsTileTensor>(cmp1Addr);
    auto cmp2Tile = PtoTile<CmpBitsTileTensor>(cmp2Addr);

    auto numLoop = dstShape4 / COUNT_MAX;
    auto remainAfterLoop = dstShape4 % COUNT_MAX;

    auto validCols = numLoop > 0 ? COUNT_MAX : remainAfterLoop;
    auto loopDstTile = DstTile(1, validCols);
    auto loopSrc1Tile = Src1Tile(1, validCols);
    auto loopSrc2Tile = Src2Tile(1, validCols);
    auto loopTmp1Tile = TmpTile(1, validCols, tmp1Addr);
    auto loopTmp2Tile = TmpTile(1, validCols, tmp2Addr);
    auto loopOneTile = TmpTile(1, validCols, oneAddr);
    auto loopZeroTile = TmpTile(1, validCols, zeroAddr);
    auto loopCvtTile = CvtTile(1, validCols, cvtAddr);
    auto startAddrUBTile = SauTile(1, ALIGN_SIZE, sauAddr);

    auto remainDstTile = loopDstTile;
    auto remainSrc1Tile = loopSrc1Tile;
    auto remainSrc2Tile = loopSrc2Tile;
    auto remainOneTile = loopOneTile;
    auto remainZeroTile = loopZeroTile;
    auto remainTmp1Tile = loopTmp1Tile;
    auto remainTmp2Tile = loopTmp2Tile;
    auto remainCvtTile = loopCvtTile;

    if (numLoop > 0 && remainAfterLoop > 0) {
        remainDstTile = DstTile(1, remainAfterLoop);
        remainSrc1Tile = Src1Tile(1, remainAfterLoop);
        remainSrc2Tile = Src2Tile(1, remainAfterLoop);
        remainOneTile = TmpTile(1, remainAfterLoop, oneAddr);
        remainZeroTile = TmpTile(1, remainAfterLoop, zeroAddr);
        remainTmp1Tile = TmpTile(1, remainAfterLoop, tmp1Addr);
        remainTmp2Tile = TmpTile(1, remainAfterLoop, tmp2Addr);
        remainCvtTile = CvtTile(1, remainAfterLoop, cvtAddr);
    }

    if (numLoop > 0) {
        pto::TEXPANDS(loopOneTile.Data(), 1.0);
        pto::TEXPANDS(loopZeroTile.Data(), 0.0);
    } else if (remainAfterLoop > 0) {
        pto::TEXPANDS(remainOneTile.Data(), 1.0);
        pto::TEXPANDS(remainZeroTile.Data(), 0.0);
    }

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    auto tileOffsets = TileOffset4Dim(n0Index, n1Index, n2Index, n3Index);
                    auto dstOffset = GenTileOffset(dst, tileOffsets);
                    auto src1Offset = GenTileOffset(src1, tileOffsets);
                    auto src2Offset = GenTileOffset(src2, tileOffsets);
                    for (LoopVar j = 0; j < numLoop; j++) {
                        loopDstTile.Assign(dst.GetAddr(), dstOffset + j * COUNT_MAX);
                        loopSrc1Tile.Assign(src1.GetAddr(), src1Offset + j * COUNT_MAX);
                        loopSrc2Tile.Assign(src2.GetAddr(), src2Offset + j * COUNT_MAX);
                        LogicalAndImpl(loopDstTile.Data(), loopSrc1Tile.Data(), loopSrc2Tile.Data(),
                                       loopTmp1Tile.Data(), loopTmp2Tile.Data(), loopOneTile.Data(),
                                       loopZeroTile.Data(), cmp1Tile.Data(), cmp2Tile.Data(), loopCvtTile.Data(),
                                       startAddrUBTile.Data());
                    }
                    if (remainAfterLoop > 0) {
                        remainDstTile.Assign(dst.GetAddr(), dstOffset + numLoop * COUNT_MAX);
                        remainSrc1Tile.Assign(src1.GetAddr(), src1Offset + numLoop * COUNT_MAX);
                        remainSrc2Tile.Assign(src2.GetAddr(), src2Offset + numLoop * COUNT_MAX);
                        LogicalAndImpl(remainDstTile.Data(), remainSrc1Tile.Data(), remainSrc2Tile.Data(),
                                       remainTmp1Tile.Data(), remainTmp2Tile.Data(), remainOneTile.Data(),
                                       remainZeroTile.Data(), cmp1Tile.Data(), cmp2Tile.Data(), remainCvtTile.Data(),
                                       startAddrUBTile.Data());
                    }
                }
            }
        }
    }
}
#endif // TILEOP_TILE_OPERATOR_LOGICALAND__H
