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
 * \file radix_select_b2.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_RADIX_SELECT_B2__H
#define TILEOP_TILE_OPERATOR_RADIX_SELECT_B2__H
#if defined(PTO_NPU_ARCH_A5)

#include "radix_select_util.h"

namespace RadixSelectUtil {
/*
Memory Usage:
srcMaskAddr       | srcMaskShape    uint16  |
srcTwiddleInAddr  | srcTileW        srcType |
cmpAddr           | cmpAlign        uint8   |
highAddr          | 32              uint8   |
selectCountGTAddr | 8               uint32  | and rowMinAddr
selectCountEQAddr | 8               uint32  | and gatherAddr
uselessAddr       | 8               uint32  |
histogramAddr     | 256             uint32  | selectGTAddr     | kAlign uint32 |
histogramTmpAddr  | 256             uint32  | selectEQAddr     | kAlign uint32 |
tciAddr           | 256             uint32  |
*/
template <int k, bool isLargest, typename VAL, typename IDX, typename TMP, typename SRC>
TILEOP void RadixSelectCalcB2(VAL value, IDX index, TMP tmp, SRC src)
{
    PTO_RS_PREPARE;
    constexpr bool isUInt = std::is_same_v<SrcDType, uint16_t>;
    constexpr bool isFloat = !(std::is_same_v<SrcDType, uint16_t> || std::is_same_v<SrcDType, int16_t>);
    // Define memory address
    size_t point = tmp.GetAddr();
    size_t srcMaskAddr = DefineWorkSpace<uint16_t, srcTileH * srcMaskShape>(point);
    size_t sortTmpAddr_ = point;
    size_t srcTwiddleInAddr = DefineWorkSpace<SrcDType, srcTileH * srcTileW>(point);
    size_t cmpAddr = DefineWorkSpace<uint8_t, srcTileH * cmpAlign>(point);
    size_t highAddr = DefineWorkSpace<uint8_t, srcTileH * 32>(point);
    size_t selectCountGTAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t rowMinAddr = selectCountGTAddr;
    size_t selectCountEQAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t gatherAddr = selectCountEQAddr;
    size_t uselessAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t histogramAddr = DefineWorkSpace<uint32_t, srcTileH * 256>(point);
    size_t histogramTmpAddr = DefineWorkSpace<uint32_t, srcTileH * 256>(point);
    size_t tciAddr = DefineWorkSpace<uint32_t, srcTileH * 256>(point);
    point = histogramAddr;
    size_t selectGTAddr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point);
    size_t selectEQAddr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point);
    PTO_RS_SORT_ADDR_DEFINE(16);
    // Define tile
    PTO_RS_COMMON_TILE_DEFINE;
    auto twiddleIntKTile = DefineTile<ConvIntType, srcTileH, srcTileW>(srcShape[3], k, srcTwiddleInAddr);
    auto srcMaskInt16Tile = DefineTile<int16_t, srcTileH, srcMaskShape>(srcShape[3], srcShape[4], srcMaskAddr);
    auto srcMaskInt16MaxTile = DefineTile<int16_t, srcTileH, srcMaskShape>(srcShape[3], srcMaskShape, srcMaskAddr);
    auto srcMaskInt16KTile = DefineTile<int16_t, srcTileH, srcMaskShape>(srcShape[3], k, srcMaskAddr);
    auto highTile = DefineTile<uint8_t, srcTileH, 32, true>(1, srcShape[3], highAddr);
    auto highIntTile = DefineTile<int32_t, srcTileH>(srcShape[3], 8, highAddr);
    auto highUInt16Tile = DefineTile<uint16_t, srcTileH>(srcShape[3], 16, highAddr);
    auto histogramUInt32Tile = DefineTile<uint32_t, srcTileH, 256>(srcShape[3], 256, histogramAddr);
    auto histogramInt32Tile = DefineTile<int32_t, srcTileH, 256>(srcShape[3], 256, histogramAddr);
    auto histogramInt32PreTile = DefineTile<int32_t, srcTileH, 256>(srcShape[3], 256, histogramAddr - 32);
    auto histogramTmpUInt32Tile = DefineTile<uint32_t, srcTileH, 256>(srcShape[3], 256, histogramTmpAddr);
    auto histogramTmpInt32Tile = DefineTile<int32_t, srcTileH, 256>(srcShape[3], 256, histogramTmpAddr);
    auto tciTile = DefineTile<uint32_t, srcTileH, 256>(srcShape[3], 256, tciAddr);
    PTO_RS_SORT_TILE_DEFINE(16);

    for (LoopVar n0Index = 0; n0Index < srcShape[0]; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < srcShape[1]; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < srcShape[2]; ++n2Index) {
                RadixSelectBatchAssign(src, value, index, n0Index, n1Index, n2Index, srcStride, valStride, idxStride,
                                       srcTypeSize, idxTypeSize, srcIntTile, valIntTile, idxTile);
                RadixSelectTCI(tciTile);
                RadixSelectTwiddle<isLargest, true, isUInt, isFloat, SrcDType>(twiddleIntTile, srcIntTile,
                                                                               srcMaskInt16Tile, cmpTile, uselessTile);
                RadixSelectHistogram<pto::HistByte::BYTE_1>(histogramUInt32Tile, twiddleUIntTile, highTile);
                pto::TCMPS(cmpTile, histogramUInt32Tile, static_cast<uint32_t>(srcShape[4] - k), pto::CmpMode::GT);
                pto::TMOV(histogramTmpUInt32Tile, tciTile);
                pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                pto::TEXPANDS(highIntTile, 0);
                pto::TROWMIN(highIntTile, histogramTmpInt32Tile, uselessTile);
                pto::TCMPS(cmpTile, highIntTile, 0, pto::CmpMode::NE);
                pto::TADDS(rowMinTile, highIntTile, 7);
                RadixSelectGather(gatherIntTile, histogramInt32PreTile, rowMinTile, uselessTile);
                pto::TSELS(gatherIntTile, cmpTile, gatherIntTile, uselessTile, 0);
                pto::TEXPANDS(rowMinTile, static_cast<int32_t>(srcShape[4] - k));
                pto::TSUB(rowMinTile, rowMinTile, gatherIntTile);
                RadixSelectHistogram<pto::HistByte::BYTE_0>(histogramUInt32Tile, twiddleUIntTile, highTile);
                RadixSelectCmps<pto::CmpMode::GT>(cmpTile, histogramInt32Tile, rowMinTile);
                pto::TMOV(histogramTmpUInt32Tile, tciTile);
                pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                pto::TEXPANDS(rowMinTile, 0);
                pto::TROWMIN(rowMinTile, histogramTmpInt32Tile, uselessTile);
                pto::TSHLS(highIntTile, highIntTile, 8);
                pto::TADD(highIntTile, highIntTile, rowMinTile);
                pto::TEXPANDS(srcMaskInt16MaxTile, static_cast<int16_t>(0));
                pto::TMOV(srcMaskInt16Tile, twiddleIntTile);
                RadixSelectFinalSelect(selectInt32GTTile, selectInt32EQTile, srcMaskInt16Tile, highUInt16Tile,
                                       selectCountUInt32GTTile, selectCountUInt32EQTile, idxTile, uselessTile);
                pto::TEXPANDS(srcMaskInt16MaxTile, 0x7fff);
                RadixSelectGather(srcMaskInt16KTile, twiddleIntTile, idxTile, uselessTile);
                RadixSelectSortPrepare(sortTempInt16MaxTile, number0UInt16Tile, number1UInt16Tile, number2UInt16Tile,
                                       number3UInt16Tile);
                RadixSelectSortTwoBit<0, 16>(
                    srcMaskInt16KTile, srcMaskInt16KTile, indexInt32Tile, sortTempInt16KTile, sortTempInt16KTile,
                    uselessTile, select1Int32Tile, select2Int32Tile, select3Int32Tile, cnt1UInt32Tile, cnt2UInt32Tile,
                    cnt3UInt32Tile, number0UInt16Tile, number1UInt16Tile, number2UInt16Tile, number3UInt16Tile);
                RadixSelectGather(select1Int32Tile, idxTile, indexInt32Tile, uselessTile);
                pto::TMOV(idxTile, select1Int32Tile);
                RadixSelectTwiddle<isLargest, false, isUInt, isFloat, SrcDType>(valIntTile, srcMaskInt16KTile,
                                                                                twiddleIntKTile, cmpTile, uselessTile);
            }
        }
    }
}

} // namespace RadixSelectUtil

#endif // defined(PTO_NPU_ARCH_A5)
#endif // TILEOP_TILE_OPERATOR_RADIX_SELECT_B2__H
