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
 * \file radix_select_b1.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_RADIX_SELECT_B1__H
#define TILEOP_TILE_OPERATOR_RADIX_SELECT_B1__H
#if defined(PTO_NPU_ARCH_A5)

#include "radix_select_util.h"

namespace RadixSelectUtil {
/*
Memory Usage:
srcMaskAddr       | srcMaskShape    uint16  |
srcTwiddleInAddr  | srcTileW        srcType |
cmpAddr           | cmpAlign        uint8   | srcTmpAddr       | srcMaskShape uint16  |
highAddr          | 32              uint8   |
selectCountGTAddr | 8               uint32  | and rowMinAddr
selectCountEQAddr | 8               uint32  | and gatherAddr
uselessAddr       | 8               uint32  |
histogramAddr     | 256             uint32  | selectGTAddr     | kAlign uint32 |
histogramTmpAddr  | 256             uint32  | selectEQAddr     | kAlign uint32 |
*/
template <int k, bool isLargest, typename VAL, typename IDX, typename TMP, typename SRC>
TILEOP void RadixSelectCalcB1(VAL value, IDX index, TMP tmp, SRC src)
{
    PTO_RS_PREPARE;
    constexpr bool isUInt = std::is_same_v<SrcDType, uint8_t>;
    constexpr bool isFloat = !(std::is_same_v<SrcDType, uint8_t> || std::is_same_v<SrcDType, int8_t>);
    // Define memory address
    size_t point = tmp.GetAddr();
    size_t srcMaskAddr = DefineWorkSpace<uint16_t, srcTileH * srcMaskShape>(point);
    size_t sortTmpAddr_ = point;
    size_t srcTwiddleInAddr = DefineWorkSpace<SrcDType, srcTileH * srcTileW>(point);
    size_t cmpAddr = DefineWorkSpace<uint8_t, srcTileH * cmpAlign>(point);
    size_t srcTmpAddr = cmpAddr;
    size_t highAddr = DefineWorkSpace<uint8_t, srcTileH * 32>(point);
    size_t selectCountGTAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t rowMinAddr = selectCountGTAddr;
    size_t selectCountEQAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t gatherAddr = selectCountEQAddr;
    size_t uselessAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t histogramAddr = DefineWorkSpace<uint32_t, srcTileH * 256>(point);
    size_t histogramTmpAddr = DefineWorkSpace<uint32_t, srcTileH * 256>(point);
    point = histogramAddr;
    size_t selectGTAddr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point);
    size_t selectEQAddr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point);
    PTO_RS_SORT_ADDR_DEFINE(16);
    // Define tile
    PTO_RS_COMMON_TILE_DEFINE;
    auto srcMaskUInt8KTile = DefineTile<uint8_t, srcTileH, srcMaskShape>(srcShape[3], k, srcMaskAddr);
    auto srcMaskInt8KTile = DefineTile<int8_t, srcTileH, srcMaskShape>(srcShape[3], k, srcMaskAddr);
    auto srcMaskUInt16Tile = DefineTile<uint16_t, srcTileH, srcMaskShape>(srcShape[3], srcShape[4], srcMaskAddr);
    auto srcMaskInt16Tile = DefineTile<int16_t, srcTileH, srcMaskShape>(srcShape[3], srcShape[4], srcMaskAddr);
    auto srcMaskInt16KTile = DefineTile<int16_t, srcTileH, srcMaskShape>(srcShape[3], k, srcMaskAddr);
    auto srcMaskInt16MaxTile = DefineTile<int16_t, srcTileH, srcMaskShape>(srcShape[3], srcMaskShape, srcMaskAddr);
    auto srcTmpInt16KTile = DefineTile<int16_t, srcTileH, srcMaskShape>(srcShape[3], k, srcTmpAddr);
    auto highTile = DefineTile<uint8_t, srcTileH, 32, true>(1, srcShape[3], highAddr);
    auto highIntTile = DefineTile<int32_t, srcTileH>(srcShape[3], 8, highAddr);
    auto highUInt16Tile = DefineTile<uint16_t, srcTileH>(srcShape[3], 16, highAddr);
    auto histogramUInt32Tile = DefineTile<uint32_t, srcTileH, 256>(srcShape[3], 256, histogramAddr);
    auto histogramTmpUInt32Tile = DefineTile<uint32_t, srcTileH, 256>(srcShape[3], 256, histogramTmpAddr);
    auto histogramTmpInt32Tile = DefineTile<int32_t, srcTileH, 256>(srcShape[3], 256, histogramTmpAddr);
    PTO_RS_SORT_TILE_DEFINE(16);

    for (LoopVar n0Index = 0; n0Index < srcShape[0]; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < srcShape[1]; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < srcShape[2]; ++n2Index) {
                RadixSelectBatchAssign(src, value, index, n0Index, n1Index, n2Index, srcStride, valStride, idxStride,
                                       srcTypeSize, idxTypeSize, srcIntTile, valIntTile, idxTile);
                RadixSelectTwiddle<isLargest, true, isUInt, isFloat, SrcDType>(twiddleIntTile, srcIntTile, uselessTile,
                                                                               uselessTile, uselessTile);
                pto::TEXPANDS(srcMaskInt16MaxTile, static_cast<int16_t>(0));
                pto::TEXPANDS(highUInt16Tile, 0);
                pto::TCVT(srcMaskUInt16Tile, twiddleUIntTile, pto::RoundMode::CAST_TRUNC);
                RadixSelectHistogram<pto::HistByte::BYTE_0>(histogramUInt32Tile, srcMaskUInt16Tile, highTile);
                pto::TCMPS(cmpTile, histogramUInt32Tile, static_cast<uint32_t>(srcShape[4] - k), pto::CmpMode::GT);
                RadixSelectTCI(histogramTmpUInt32Tile);
                pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                pto::TROWMIN(highIntTile, histogramTmpInt32Tile, uselessTile);
                RadixSelectFinalSelect(selectInt32GTTile, selectInt32EQTile, srcMaskInt16Tile, highUInt16Tile,
                                       selectCountUInt32GTTile, selectCountUInt32EQTile, idxTile, uselessTile);
                RadixSelectGather(srcTmpInt16KTile, srcMaskInt16Tile, idxTile, uselessTile);
                pto::TEXPANDS(srcMaskInt16MaxTile, 0x7fff);
                pto::TMOV(srcMaskInt16KTile, srcTmpInt16KTile);
                RadixSelectSortPrepare(sortTempInt16MaxTile, number0UInt16Tile, number1UInt16Tile, number2UInt16Tile,
                                       number3UInt16Tile);
                RadixSelectSortTwoBit<0, 8>(srcMaskInt16KTile, srcMaskInt16KTile, indexInt32Tile, sortTempInt16KTile,
                                            sortTempInt16KTile, uselessTile, select1Int32Tile, select2Int32Tile,
                                            select3Int32Tile, cnt1UInt32Tile, cnt2UInt32Tile, cnt3UInt32Tile,
                                            number0UInt16Tile, number1UInt16Tile, number2UInt16Tile, number3UInt16Tile);
                RadixSelectGather(select1Int32Tile, idxTile, indexInt32Tile, uselessTile);
                pto::TMOV(idxTile, select1Int32Tile);
                pto::TCVT(srcMaskUInt8KTile, srcMaskInt16KTile, pto::RoundMode::CAST_TRUNC);
                RadixSelectTwiddle<isLargest, false, isUInt, isFloat, SrcDType>(valIntTile, srcMaskInt8KTile,
                                                                                uselessTile, uselessTile, uselessTile);
            }
        }
    }
}

} // namespace RadixSelectUtil

#endif // defined(PTO_NPU_ARCH_A5)
#endif // TILEOP_TILE_OPERATOR_RADIX_SELECT_B1__H
