
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
 * \file compare.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_COMPARE__H
#define TILEOP_TILE_OPERATOR_COMPARE__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename T>
struct CompareTmpBuffers {
    __ubuf__ uint8_t* vcmpBitResult;
    __ubuf__ uint8_t* startAddrUB;
    __ubuf__ T* zeroCondition;
    __ubuf__ T* oneCondition;
    __ubuf__ T* vselResult;
};

template <typename T0, typename T1 = T0>
struct CompareTileTypes {
    static constexpr int64_t COUNT_MAX = 1024;
    using SrcTile0 = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using SrcTile1 = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using SrcTile = SrcTile0;
    using DstTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using CmpTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using TmpTile = pto::Tile<pto::TileType::Vec, half, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
};

template <typename T, typename TTmp>
TILEOP CompareTmpBuffers<T> InitCompareTmpBuffers(TTmp tmpbuf)
{
    constexpr uint64_t countNum = 4096 / sizeof(typename T::Type);
    const uint32_t ALIGNMENT = 32;
    const uint32_t vcmpBitsSize = (countNum + 7) / 8;

    uint64_t tmpbufAddr = tmpbuf.GetAddr();
    __ubuf__ uint8_t* currentPtr = reinterpret_cast<__ubuf__ uint8_t*>(tmpbufAddr);

    CompareTmpBuffers<T> buffers;
    buffers.vcmpBitResult = currentPtr;
    currentPtr += vcmpBitsSize;

    currentPtr = reinterpret_cast<__ubuf__ uint8_t*>((reinterpret_cast<uintptr_t>(currentPtr) + ALIGNMENT - 1) &
                                                     ~(ALIGNMENT - 1));
    buffers.startAddrUB = reinterpret_cast<__ubuf__ uint8_t*>(currentPtr);
    currentPtr += ALIGNMENT;

    currentPtr = reinterpret_cast<__ubuf__ uint8_t*>((reinterpret_cast<uintptr_t>(currentPtr) + ALIGNMENT - 1) &
                                                     ~(ALIGNMENT - 1));
    buffers.zeroCondition = reinterpret_cast<__ubuf__ T*>(currentPtr);
    currentPtr += countNum * sizeof(typename T::Type);

    currentPtr = reinterpret_cast<__ubuf__ uint8_t*>((reinterpret_cast<uintptr_t>(currentPtr) + ALIGNMENT - 1) &
                                                     ~(ALIGNMENT - 1));
    buffers.oneCondition = reinterpret_cast<__ubuf__ T*>(currentPtr);
    currentPtr += countNum * sizeof(typename T::Type);

    currentPtr = reinterpret_cast<__ubuf__ uint8_t*>((reinterpret_cast<uintptr_t>(currentPtr) + ALIGNMENT - 1) &
                                                     ~(ALIGNMENT - 1));
    buffers.vselResult = reinterpret_cast<__ubuf__ T*>(currentPtr);

    return buffers;
}

struct CompareLayoutInfo {
    size_t shape0, shape1, shape2, shape3, shape4, dstShape;
    size_t stride0, stride1, stride2, stride3;
    size_t dstStride0, dstStride1, dstStride2, dstStride3;
};

template <typename T, typename TDst>
TILEOP CompareLayoutInfo ExtractLayoutInfo(const T& src, const TDst& dst)
{
    constexpr size_t expectSize = 5;
    const auto srcLayout = src.GetLayout();
    const auto dstLayout = dst.GetLayout();

    CompareLayoutInfo info;
    info.shape0 = srcLayout.template GetShapeDim<0, expectSize>();
    info.shape1 = srcLayout.template GetShapeDim<1, expectSize>();
    info.shape2 = srcLayout.template GetShapeDim<2, expectSize>();
    info.shape3 = srcLayout.template GetShapeDim<3, expectSize>();
    info.shape4 = srcLayout.template GetShapeDim<4, expectSize>();
    info.dstShape = dstLayout.template GetShapeDim<4, expectSize>();

    info.stride0 = srcLayout.template GetStrideDim<0, expectSize>();
    info.stride1 = srcLayout.template GetStrideDim<1, expectSize>();
    info.stride2 = srcLayout.template GetStrideDim<2, expectSize>();
    info.stride3 = srcLayout.template GetStrideDim<3, expectSize>();

    info.dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    info.dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    info.dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    info.dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();

    return info;
}

template <int64_t cmpOp, typename DstTileType, typename SrcTileType0, typename SrcTileType1>
TILEOP void ExecuteCompare(DstTileType& dst0, SrcTileType0& src0Tile, SrcTileType1& src1Tile)
{
    switch (static_cast<pto::CmpMode>(cmpOp)) {
        case pto::CmpMode::EQ:
            pto::TCMP(dst0, src0Tile, src1Tile, pto::CmpMode::EQ);
            break;
        case pto::CmpMode::NE:
            pto::TCMP(dst0, src0Tile, src1Tile, pto::CmpMode::NE);
            break;
        case pto::CmpMode::LT:
            pto::TCMP(dst0, src0Tile, src1Tile, pto::CmpMode::LT);
            break;
        case pto::CmpMode::LE:
            pto::TCMP(dst0, src0Tile, src1Tile, pto::CmpMode::LE);
            break;
        case pto::CmpMode::GT:
            pto::TCMP(dst0, src0Tile, src1Tile, pto::CmpMode::GT);
            break;
        case pto::CmpMode::GE:
            pto::TCMP(dst0, src0Tile, src1Tile, pto::CmpMode::GE);
            break;
    }
}

template <int64_t cmpOp, typename DstTileType, typename SrcTileType, typename TVal>
TILEOP void ExecuteCompareScalar(DstTileType& dst0, SrcTileType& srcTile, TVal scalarVal)
{
    switch (static_cast<pto::CmpMode>(cmpOp)) {
        case pto::CmpMode::EQ:
            pto::TCMPS(dst0, srcTile, scalarVal, pto::CmpMode::EQ);
            break;
        case pto::CmpMode::NE:
            pto::TCMPS(dst0, srcTile, scalarVal, pto::CmpMode::NE);
            break;
        case pto::CmpMode::LT:
            pto::TCMPS(dst0, srcTile, scalarVal, pto::CmpMode::LT);
            break;
        case pto::CmpMode::LE:
            pto::TCMPS(dst0, srcTile, scalarVal, pto::CmpMode::LE);
            break;
        case pto::CmpMode::GT:
            pto::TCMPS(dst0, srcTile, scalarVal, pto::CmpMode::GT);
            break;
        case pto::CmpMode::GE:
            pto::TCMPS(dst0, srcTile, scalarVal, pto::CmpMode::GE);
            break;
    }
}

template <typename T, typename SrcTileType, typename DstTileType, typename CmpTileType, typename TmpTileType,
          typename AddrUBTileType>
TILEOP void PostProcessMode0(DstTileType& bitResTile, CmpTileType& cmpResTile, SrcTileType& vselResultTile,
                             SrcTileType& oneConditionTile, SrcTileType& zeroConditionTile,
                             AddrUBTileType& startAddrUBTile, TmpTileType& tmpTile, __ubuf__ T* zeroCondition)
{
    pto::TEXPANDS(zeroConditionTile, 0.000000e+00f);
    pto::TEXPANDS(oneConditionTile, 1.000000e+00f);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSEL(vselResultTile, cmpResTile, oneConditionTile, zeroConditionTile, startAddrUBTile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    if constexpr (sizeof(typename T::Type) == 2) {
        pto::TCVT(bitResTile, vselResultTile, pto::RoundMode::CAST_NONE);
    } else if constexpr (sizeof(typename T::Type) == 4) {
        pto::TASSIGN(tmpTile, reinterpret_cast<uint64_t>(zeroCondition));
        pto::TCVT(tmpTile, vselResultTile, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        pto::TCVT(bitResTile, tmpTile, pto::RoundMode::CAST_NONE);
    }
}

TILEOP void CalcOffsets(const CompareLayoutInfo& info, size_t n0Index, size_t n1Index, size_t n2Index, size_t n3Index,
                        size_t& srcOffset, size_t& dstOffset)
{
    dstOffset = n0Index * info.dstStride0 + n1Index * info.dstStride1 + n2Index * info.dstStride2 +
                n3Index * info.dstStride3;
    srcOffset = n0Index * info.stride0 + n1Index * info.stride1 + n2Index * info.stride2 + n3Index * info.stride3;
}

template <typename T, typename Types, typename TileStartAddrUB>
TILEOP void InitCommonTiles(typename Types::SrcTile& vselResultTile, TileStartAddrUB& startAddrUBTile,
                            typename Types::SrcTile& oneConditionTile, typename Types::SrcTile& zeroConditionTile,
                            typename Types::DstTile& bitResTile, typename Types::CmpTile& cmpResTile,
                            const CompareTmpBuffers<T>& buffers, uint64_t dstAddr, size_t shape4, size_t dstShape)
{
    vselResultTile = typename Types::SrcTile(1, shape4);
    oneConditionTile = typename Types::SrcTile(1, shape4);
    zeroConditionTile = typename Types::SrcTile(1, shape4);
    bitResTile = typename Types::DstTile(1, dstShape);
    cmpResTile = typename Types::CmpTile(1, dstShape);

    pto::TASSIGN(bitResTile, dstAddr);
    pto::TASSIGN(vselResultTile, reinterpret_cast<uint64_t>(buffers.vselResult));
    pto::TASSIGN(startAddrUBTile, reinterpret_cast<uint64_t>(buffers.startAddrUB));
    pto::TASSIGN(oneConditionTile, reinterpret_cast<uint64_t>(buffers.oneCondition));
    pto::TASSIGN(zeroConditionTile, reinterpret_cast<uint64_t>(buffers.zeroCondition));
    pto::TASSIGN(cmpResTile, reinterpret_cast<uint64_t>(buffers.vcmpBitResult));
}

template <int64_t cmpOp, int64_t mode, typename T0, typename T1, typename TDst, typename TTmp>
TILEOP void TCompare(TDst dst, T0 src0, T1 src1, TTmp tmpbuf)
{
    auto info = ExtractLayoutInfo(src0, dst);
    auto buffers = InitCompareTmpBuffers<T0>(tmpbuf);
    constexpr auto dstTypeSize = sizeof(typename TDst::Type);
    constexpr auto src0TypeSize = sizeof(typename T0::Type);
    constexpr auto src1TypeSize = sizeof(typename T1::Type);
    constexpr unsigned alignUint8 = 32;
    constexpr unsigned addressUsed = 4;
    using Types = CompareTileTypes<T0, T1>;
    using SrcTile0 = typename Types::SrcTile0;
    using SrcTile1 = typename Types::SrcTile1;
    using SrcTile = typename Types::SrcTile;
    using DstTile = typename Types::DstTile;
    using CmpTile = typename Types::CmpTile;
    using TmpTile = typename Types::TmpTile;
    using TileStartAddrUB = pto::Tile<pto::TileType::Vec, uint8_t, 1, alignUint8, pto::BLayout::RowMajor, -1, -1>;

    constexpr uint64_t countBy4096 = 4096 / sizeof(typename T0::Type);
    constexpr uint64_t elementsPerCount = (countBy4096 < static_cast<uint64_t>(Types::COUNT_MAX)) ?
                                              countBy4096 :
                                              static_cast<uint64_t>(Types::COUNT_MAX);
    constexpr uint64_t dstElementsPerCount = (mode == 0) ? elementsPerCount : ((elementsPerCount + 7) / 8);

    uint64_t numCountPerLine = info.shape4 / elementsPerCount;
    uint64_t elementsRemainPerLine = info.shape4 % elementsPerCount;

    for (LoopVar n0Index = 0; n0Index < info.shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < info.shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < info.shape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < info.shape3; ++n3Index) {
                    size_t srcOffset, dstOffset;
                    CalcOffsets(info, n0Index, n1Index, n2Index, n3Index, srcOffset, dstOffset);
                    auto src1Offset = GenTileOffset(src1, TileOffset4Dim(n0Index, n1Index, n2Index, n3Index));

                    for (LoopVar j = 0; j < numCountPerLine; ++j) {
                        size_t curShape4 = elementsPerCount;
                        size_t curDstShape = (mode == 0) ? curShape4 : ((curShape4 + 7) / 8);

                        size_t curSrcOffset = srcOffset + j * elementsPerCount;
                        size_t curSrc1Offset = src1Offset + j * elementsPerCount;
                        size_t curDstOffset = dstOffset + j * dstElementsPerCount;

                        uint64_t dstAddr = dst.GetAddr() + curDstOffset * dstTypeSize;

                        SrcTile0 src0Tile(1, curShape4);
                        SrcTile1 src1Tile(1, curShape4);
                        SrcTile vselResultTile, oneConditionTile, zeroConditionTile;
                        DstTile bitResTile;
                        CmpTile cmpResTile;
                        TileStartAddrUB startAddrUBTile(1, addressUsed);
                        TmpTile tmpTile(1, curShape4);

                        InitCommonTiles<T0, Types>(vselResultTile, startAddrUBTile, oneConditionTile, zeroConditionTile,
                                                   bitResTile, cmpResTile, buffers, dstAddr, curShape4, curDstShape);

                        pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + curSrcOffset * src0TypeSize));
                        pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + curSrc1Offset * src1TypeSize));

                        auto& dst0 = (mode == 0) ? cmpResTile : bitResTile;
                        ExecuteCompare<cmpOp>(dst0, src0Tile, src1Tile);

                        if constexpr (mode == 0) {
                            PostProcessMode0<T0>(bitResTile, cmpResTile, vselResultTile, oneConditionTile,
                                                 zeroConditionTile, startAddrUBTile, tmpTile, buffers.zeroCondition);
                        }
                    }

                    if (elementsRemainPerLine) {
                        size_t curShape4 = elementsRemainPerLine;
                        size_t curDstShape = (mode == 0) ? curShape4 : ((curShape4 + 7) / 8);

                        size_t curSrcOffset = srcOffset + numCountPerLine * elementsPerCount;
                        size_t curSrc1Offset = src1Offset + numCountPerLine * elementsPerCount;
                        size_t curDstOffset = dstOffset + numCountPerLine * dstElementsPerCount;

                        uint64_t dstAddr = dst.GetAddr() + curDstOffset * dstTypeSize;

                        SrcTile0 src0Tile(1, curShape4);
                        SrcTile1 src1Tile(1, curShape4);
                        SrcTile vselResultTile, oneConditionTile, zeroConditionTile;
                        DstTile bitResTile;
                        CmpTile cmpResTile;
                        TileStartAddrUB startAddrUBTile(1, addressUsed);
                        TmpTile tmpTile(1, curShape4);

                        InitCommonTiles<T0, Types>(vselResultTile, startAddrUBTile, oneConditionTile, zeroConditionTile,
                                                   bitResTile, cmpResTile, buffers, dstAddr, curShape4, curDstShape);

                        pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + curSrcOffset * src0TypeSize));
                        pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + curSrc1Offset * src1TypeSize));

                        auto& dst0 = (mode == 0) ? cmpResTile : bitResTile;
                        ExecuteCompare<cmpOp>(dst0, src0Tile, src1Tile);

                        if constexpr (mode == 0) {
                            PostProcessMode0<T0>(bitResTile, cmpResTile, vselResultTile, oneConditionTile,
                                                 zeroConditionTile, startAddrUBTile, tmpTile, buffers.zeroCondition);
                        }
                    }
                }
            }
        }
    }
}

template <int64_t cmpOp, int64_t mode, typename TVal, typename T, typename TDst, typename TTmp>
TILEOP void TCompareScalar(TDst dst, T src, TTmp tmpbuf, TVal scalarVal)
{
    auto buffers = InitCompareTmpBuffers<T>(tmpbuf);
    auto info = ExtractLayoutInfo(src, dst);
    constexpr auto dstTypeSize = sizeof(typename TDst::Type);
    constexpr auto srcTypeSize = sizeof(typename T::Type);
    constexpr unsigned alignUint8 = 32;
    constexpr unsigned addressUsed = 4;
    using Types = CompareTileTypes<T>;
    using DstTile = typename Types::DstTile;
    using CmpTile = typename Types::CmpTile;
    using SrcTile = typename Types::SrcTile;
    using TmpTile = typename Types::TmpTile;
    using TileStartAddrUB = pto::Tile<pto::TileType::Vec, uint8_t, 1, alignUint8, pto::BLayout::RowMajor, -1, -1>;

    constexpr uint64_t countBy4096 = 4096 / sizeof(typename T::Type);
    constexpr uint64_t elementsPerCount = (countBy4096 < static_cast<uint64_t>(Types::COUNT_MAX)) ?
                                              countBy4096 :
                                              static_cast<uint64_t>(Types::COUNT_MAX);
    constexpr uint64_t dstElementsPerCount = (mode == 0) ? elementsPerCount : ((elementsPerCount + 7) / 8);

    uint64_t numCountPerLine = info.shape4 / elementsPerCount;
    uint64_t elementsRemainPerLine = info.shape4 % elementsPerCount;

    for (LoopVar n0Index = 0; n0Index < info.shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < info.shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < info.shape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < info.shape3; ++n3Index) {
                    size_t srcOffset, dstOffset;
                    CalcOffsets(info, n0Index, n1Index, n2Index, n3Index, srcOffset, dstOffset);

                    for (LoopVar j = 0; j < numCountPerLine; ++j) {
                        size_t curShape4 = elementsPerCount;
                        size_t curDstShape = (mode == 0) ? curShape4 : ((curShape4 + 7) / 8);

                        size_t curSrcOffset = srcOffset + j * elementsPerCount;
                        size_t curDstOffset = dstOffset + j * dstElementsPerCount;

                        uint64_t dstAddr = dst.GetAddr() + curDstOffset * dstTypeSize;

                        SrcTile srcTile(1, curShape4);
                        SrcTile vselResultTile, oneConditionTile, zeroConditionTile;
                        DstTile bitResTile;
                        CmpTile cmpResTile;
                        TileStartAddrUB startAddrUBTile(1, addressUsed);
                        TmpTile tmpTile(1, curShape4);

                        InitCommonTiles<T, Types>(vselResultTile, startAddrUBTile, oneConditionTile, zeroConditionTile,
                                                  bitResTile, cmpResTile, buffers, dstAddr, curShape4, curDstShape);

                        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + curSrcOffset * srcTypeSize));

                        auto& dst0 = (mode == 0) ? cmpResTile : bitResTile;
                        ExecuteCompareScalar<cmpOp>(dst0, srcTile, scalarVal);

                        if constexpr (mode == 0) {
                            PostProcessMode0<T>(bitResTile, cmpResTile, vselResultTile, oneConditionTile,
                                                zeroConditionTile, startAddrUBTile, tmpTile, buffers.zeroCondition);
                        }
                    }

                    if (elementsRemainPerLine) {
                        size_t curShape4 = elementsRemainPerLine;
                        size_t curDstShape = (mode == 0) ? curShape4 : ((curShape4 + 7) / 8);

                        size_t curSrcOffset = srcOffset + numCountPerLine * elementsPerCount;
                        size_t curDstOffset = dstOffset + numCountPerLine * dstElementsPerCount;

                        uint64_t dstAddr = dst.GetAddr() + curDstOffset * dstTypeSize;

                        SrcTile srcTile(1, curShape4);
                        SrcTile vselResultTile, oneConditionTile, zeroConditionTile;
                        DstTile bitResTile;
                        CmpTile cmpResTile;
                        TileStartAddrUB startAddrUBTile(1, addressUsed);
                        TmpTile tmpTile(1, curShape4);

                        InitCommonTiles<T, Types>(vselResultTile, startAddrUBTile, oneConditionTile, zeroConditionTile,
                                                  bitResTile, cmpResTile, buffers, dstAddr, curShape4, curDstShape);

                        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + curSrcOffset * srcTypeSize));

                        auto& dst0 = (mode == 0) ? cmpResTile : bitResTile;
                        ExecuteCompareScalar<cmpOp>(dst0, srcTile, scalarVal);

                        if constexpr (mode == 0) {
                            PostProcessMode0<T>(bitResTile, cmpResTile, vselResultTile, oneConditionTile,
                                                zeroConditionTile, startAddrUBTile, tmpTile, buffers.zeroCondition);
                        }
                    }
                }
            }
        }
    }
}
#endif
