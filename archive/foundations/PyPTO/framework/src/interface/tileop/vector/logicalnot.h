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
 * \file logicalnot.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_LOGICALNOT__H
#define TILEOP_TILE_OPERATOR_LOGICALNOT__H
#include <type_traits>

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename U>
using select_type = std::conditional_t<std::is_same_v<typename U::Type, float>, float, half>;

template <typename U>
using select_type_bool = std::conditional_t<std::is_same_v<typename U::Type, bool>, uint8_t, typename U::Type>;

// 大整数类型的简化版实现
template <typename T, typename DstTile, typename SrcTile, typename TmpTile>
TILEOP void LogicalNotImplInt(DstTile dstTile, SrcTile srcTile, TmpTile tmpTile)
{
    using UnsignedT = std::make_unsigned_t<T>;
    constexpr int64_t COUNT_MAX_BYTES = 2048 * sizeof(UnsignedT);
    using USrcTile = pto::Tile<pto::TileType::Vec, UnsignedT, 1, 2048, pto::BLayout::RowMajor, -1, -1>;

    unsigned dstValid = dstTile.GetValidCol();

    USrcTile unsignedSrcTile(1, dstValid);
    pto::TASSIGN(unsignedSrcTile, (uint64_t)(tmpTile.data()));

    USrcTile unsignedSrcView(1, dstValid);
    pto::TASSIGN(unsignedSrcView, (uint64_t)(srcTile.data()));

    pto::TMINS(unsignedSrcTile, unsignedSrcView, static_cast<UnsignedT>(1));
    using SignedT = std::make_signed_t<UnsignedT>;
    using SSrcTile = pto::Tile<pto::TileType::Vec, SignedT, 1, 2048, pto::BLayout::RowMajor, -1, -1>;
    SSrcTile signedTile(1, dstValid);
    pto::TASSIGN(signedTile, (uint64_t)(unsignedSrcTile.data()));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUBS(signedTile, signedTile, static_cast<SignedT>(1));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMULS(signedTile, signedTile, static_cast<SignedT>(-1));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    // Extract low bytes using TGATHER with mask pattern
    if constexpr (sizeof(UnsignedT) == 2) {
        using U8ViewTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, COUNT_MAX_BYTES, pto::BLayout::RowMajor, -1, -1>;
        using U8DstTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, 2048, pto::BLayout::RowMajor, -1, -1>;
        U8ViewTile srcView(1, dstValid * sizeof(UnsignedT));
        U8DstTile gatherDst(1, dstValid);
        pto::TASSIGN(srcView, (uint64_t)(unsignedSrcTile.data()));
        pto::TASSIGN(gatherDst, (uint64_t)(dstTile.data()));
        pto::TGATHER<U8DstTile, U8ViewTile, pto::MaskPattern::P0101>(gatherDst, srcView);
    } else if constexpr (sizeof(UnsignedT) == 4) {
        using U8ViewTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, COUNT_MAX_BYTES, pto::BLayout::RowMajor, -1, -1>;
        using U8DstTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, 2048, pto::BLayout::RowMajor, -1, -1>;
        U8ViewTile srcView(1, dstValid * sizeof(UnsignedT));
        U8DstTile gatherDst(1, dstValid);
        pto::TASSIGN(srcView, (uint64_t)(unsignedSrcTile.data()));
        pto::TASSIGN(gatherDst, (uint64_t)(dstTile.data()));
        pto::TGATHER<U8DstTile, U8ViewTile, pto::MaskPattern::P0001>(gatherDst, srcView);
    }
}

template <typename T, typename DstTile, typename SrcTile, typename CastTile, typename ExpTile, typename VcmpResTile,
          typename TileStartAddrUB>
TILEOP void LogicalNotImpl(DstTile dstTile, SrcTile srcTile, CastTile castTile, ExpTile oneTile, ExpTile zeroTile,
                           VcmpResTile vcmpResTile, TileStartAddrUB startAddrUBTile)
{
    // Existing path for bool/uint8/int8/half/float
    if constexpr (std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value) {
        pto::TCVT(castTile, srcTile, pto::RoundMode::CAST_NONE);
    }
    pto::TEXPANDS(oneTile, 1.0);
    pto::TEXPANDS(zeroTile, 0.0);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    if constexpr (std::is_same<T, half>::value || std::is_same<T, float>::value) {
        pto::TCMP(vcmpResTile, srcTile, zeroTile, pto::CmpMode::EQ);
    } else if (std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value) {
        pto::TCMP(vcmpResTile, castTile, zeroTile, pto::CmpMode::EQ);
    }

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSEL(oneTile, vcmpResTile, oneTile, zeroTile, startAddrUBTile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    if constexpr (std::is_same<T, float>::value) {
        pto::TCVT(castTile, oneTile, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        pto::TCVT(dstTile, castTile, pto::RoundMode::CAST_NONE);
    } else {
        pto::TCVT(dstTile, oneTile, pto::RoundMode::CAST_NONE);
    }
}

template <typename T0, typename T1>
TILEOP void LogicalNotProcessTileInt(T0 dst, T1 src, __ubuf__ int8_t* tmpBuffer, unsigned count, uint64_t dstOff,
                                     uint64_t srcOff)
{
    using T = select_type_bool<T1>;
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    constexpr int64_t COUNT_MAX = 2048;
    using SrcTile = pto::Tile<pto::TileType::Vec, T, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using DstTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using TmpTile = pto::Tile<pto::TileType::Vec, half, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    DstTile dstTile(1, count);
    SrcTile srcTile(1, count);
    TmpTile tmpTile(1, count);
    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOff * dstTypeSize));
    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOff * srcTypeSize));
    pto::TASSIGN(tmpTile, (uint64_t)(tmpBuffer));
    LogicalNotImplInt<T, DstTile, SrcTile, TmpTile>(dstTile, srcTile, tmpTile);
}

template <typename T0, typename T1, typename VcmpResTileT, typename TileStartAddrUBT>
TILEOP void LogicalNotProcessTileGeneric(T0 dst, T1 src, __ubuf__ int8_t* compareCondition,
                                         __ubuf__ int8_t* oneCondition, __ubuf__ half* castCondition,
                                         VcmpResTileT vcmpResTile, TileStartAddrUBT startAddrUBTile, unsigned count,
                                         uint64_t dstOff, uint64_t srcOff)
{
    using U = select_type<T1>;
    using T = select_type_bool<T1>;
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    constexpr int64_t COUNT_MAX = 2048;
    using DstTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using SrcTile = pto::Tile<pto::TileType::Vec, T, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using CastTile = pto::Tile<pto::TileType::Vec, half, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using ExpTile = pto::Tile<pto::TileType::Vec, U, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    DstTile dstTile(1, count);
    SrcTile srcTile(1, count);
    CastTile castTile(1, count);
    ExpTile oneTile(1, count);
    ExpTile zeroTile(1, count);
    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOff * dstTypeSize));
    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOff * srcTypeSize));
    pto::TASSIGN(castTile, (uint64_t)(castCondition));
    pto::TASSIGN(oneTile, (uint64_t)(oneCondition));
    pto::TASSIGN(zeroTile, (uint64_t)(compareCondition));
    LogicalNotImpl<T, DstTile, SrcTile, CastTile, ExpTile, VcmpResTileT, TileStartAddrUBT>(
        dstTile, srcTile, castTile, oneTile, zeroTile, vcmpResTile, startAddrUBTile);
}

template <typename T1, typename T2, typename VcmpResTileT, typename TileStartAddrUBT>
TILEOP void LogicalNotGenericSetup(T2 tmp, __ubuf__ int8_t*& compareCondition, __ubuf__ int8_t*& oneCondition,
                                   __ubuf__ half*& castCondition, VcmpResTileT& vcmpResTile,
                                   TileStartAddrUBT& startAddrUBTile)
{
    constexpr int64_t COUNT_MAX = 2048;
    constexpr uint32_t ALIGN_SIZE = 32;
    constexpr int64_t TYPE_SIZE = std::is_same_v<typename T1::Type, float> ? 4 : 2;
    uint32_t vcmpBitSize = (COUNT_MAX + 7) / 8;
    __ubuf__ int8_t* vcmpBitResult = reinterpret_cast<__ubuf__ int8_t*>(tmp.GetAddr());
    uintptr_t zeroCondAddr = reinterpret_cast<uintptr_t>(vcmpBitResult + vcmpBitSize);
    zeroCondAddr = (zeroCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    compareCondition = reinterpret_cast<__ubuf__ int8_t*>(zeroCondAddr);
    uintptr_t oneCondAddr = reinterpret_cast<uintptr_t>(compareCondition + COUNT_MAX * TYPE_SIZE);
    oneCondAddr = (oneCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    oneCondition = reinterpret_cast<__ubuf__ int8_t*>(oneCondAddr);
    uintptr_t castAddr = reinterpret_cast<uintptr_t>(oneCondition + COUNT_MAX * TYPE_SIZE);
    castAddr = (castAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    castCondition = reinterpret_cast<__ubuf__ half*>(castAddr);
    uintptr_t startAddrAddr = reinterpret_cast<uintptr_t>(castCondition + COUNT_MAX);
    startAddrAddr = (startAddrAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ uint8_t* startAddrUB = reinterpret_cast<__ubuf__ uint8_t*>(startAddrAddr);
    pto::TASSIGN(vcmpResTile, (uint64_t)(vcmpBitResult));
    pto::TASSIGN(startAddrUBTile, (uint64_t)(startAddrUB));
}

template <typename T0, typename T1, typename VcmpResTileT, typename TileStartAddrUBT>
TILEOP void LogicalNotProcessTile(T0 dst, T1 src, __ubuf__ int8_t* tmpBuffer, __ubuf__ int8_t* compareCondition,
                                  __ubuf__ int8_t* oneCondition, __ubuf__ half* castCondition, VcmpResTileT vcmpResTile,
                                  TileStartAddrUBT startAddrUBTile, unsigned count, uint64_t dstOff, uint64_t srcOff)
{
    if constexpr (std::is_same_v<typename T1::Type, int16_t> || std::is_same_v<typename T1::Type, uint16_t> ||
                  std::is_same_v<typename T1::Type, int32_t> || std::is_same_v<typename T1::Type, uint32_t>) {
        LogicalNotProcessTileInt(dst, src, tmpBuffer, count, dstOff, srcOff);
    } else {
        LogicalNotProcessTileGeneric(dst, src, compareCondition, oneCondition, castCondition, vcmpResTile,
                                     startAddrUBTile, count, dstOff, srcOff);
    }
}

template <typename T1, typename T2, typename VcmpResTileT, typename TileStartAddrUBT>
TILEOP void LogicalNotPrepareState(T2 tmp, __ubuf__ int8_t*& tmpBuffer, __ubuf__ int8_t*& compareCondition,
                                   __ubuf__ int8_t*& oneCondition, __ubuf__ half*& castCondition,
                                   VcmpResTileT& vcmpResTile, TileStartAddrUBT& startAddrUBTile)
{
    if constexpr (std::is_same_v<typename T1::Type, int16_t> || std::is_same_v<typename T1::Type, uint16_t> ||
                  std::is_same_v<typename T1::Type, int32_t> || std::is_same_v<typename T1::Type, uint32_t>) {
        tmpBuffer = reinterpret_cast<__ubuf__ int8_t*>(tmp.GetAddr());
    } else {
        LogicalNotGenericSetup<T1>(tmp, compareCondition, oneCondition, castCondition, vcmpResTile, startAddrUBTile);
    }
}

template <typename T0, typename T1, typename T2, typename LayoutD, typename LayoutS>
TILEOP void LogicalNotIterateTiles(T0 dst, T1 src, T2 tmp, LayoutD dstLayout, LayoutS srcLayout)
{
    constexpr int64_t COUNT_MAX = 2048;
    constexpr uint32_t ALIGN_SIZE = 32;
    auto dstShape0 = dstLayout.template GetShapeDim<0, 5>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, 5>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, 5>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, 5>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, 5>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, 5>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, 5>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, 5>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, 5>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, 5>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, 5>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, 5>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, 5>();
    __ubuf__ int8_t* tmpBuffer = nullptr;
    __ubuf__ int8_t* compareCondition = nullptr;
    __ubuf__ int8_t* oneCondition = nullptr;
    __ubuf__ half* castCondition = nullptr;
    using VcmpResTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, COUNT_MAX / 8, pto::BLayout::RowMajor, 1,
                                  COUNT_MAX / 8>;
    using TileStartAddrUB = pto::Tile<pto::TileType::Vec, uint8_t, 1, ALIGN_SIZE, pto::BLayout::RowMajor, -1, -1>;
    VcmpResTile vcmpResTile;
    TileStartAddrUB startAddrUBTile(1, ALIGN_SIZE / 8);
    LogicalNotPrepareState<T1>(tmp, tmpBuffer, compareCondition, oneCondition, castCondition, vcmpResTile,
                               startAddrUBTile);
    unsigned numLoop = dstShape4 / COUNT_MAX;
    unsigned remainAfterLoop = dstShape4 % COUNT_MAX;
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 +
                                     n3Index * dstStride3;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2 +
                                     n3Index * srcStride3;
                    for (LoopVar j = 0; j < numLoop; j++) {
                        LogicalNotProcessTile(dst, src, tmpBuffer, compareCondition, oneCondition, castCondition,
                                              vcmpResTile, startAddrUBTile, static_cast<unsigned>(COUNT_MAX),
                                              static_cast<uint64_t>(dstOffset + j * COUNT_MAX),
                                              static_cast<uint64_t>(srcOffset + j * COUNT_MAX));
                    }
                    if (remainAfterLoop > 0) {
                        LogicalNotProcessTile(dst, src, tmpBuffer, compareCondition, oneCondition, castCondition,
                                              vcmpResTile, startAddrUBTile, remainAfterLoop,
                                              static_cast<uint64_t>(dstOffset + numLoop * COUNT_MAX),
                                              static_cast<uint64_t>(srcOffset + numLoop * COUNT_MAX));
                    }
                }
            }
        }
    }
}

#define OP_TILE_OP_LOGICALNOT TLogicalNot
template <typename T0, typename T1, typename T2>
TILEOP void TLogicalNot(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    LogicalNotIterateTiles<T0, T1, T2>(dst, src, tmp, dstLayout, srcLayout);
}
#endif
