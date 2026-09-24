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
 * \file radix_select_b4.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_RADIX_SELECT_B4__H
#define TILEOP_TILE_OPERATOR_RADIX_SELECT_B4__H
#if defined(PTO_NPU_ARCH_A5)

#include "radix_select_util.h"

namespace RadixSelectUtil {
/*
Memory Usage:
srcMaskAddr       | srcMaskShape    uint32  |
srcTwiddleInAddr  | srcTileW        srcType |
cmpAddr           | cmpAlign        uint8   |
high1Addr         | 32              uint8   |
high2Addr         | 32              uint8   |
high3Addr         | 32              uint8   |
selectCountGTAddr | 8               uint32  | and rowMinAddr
selectCountEQAddr | 8               uint32  | and gatherAddr
remindKAddr       | 8               uint32  |
kthValueAddr      | 8               uint32  |
tmpAddr           | 8               uint32  |
uselessAddr       | 8               uint32  |
histogramAddr     | 256             uint32  | selectGTAddr     | kAlign uint32 |
histogramTmpAddr  | 256             uint32  | selectEQAddr     | kAlign uint32 |
tciAddr           | 256             uint32  |
*/
template <int k, bool isLargest, typename VAL, typename IDX, typename TMP, typename SRC>
TILEOP void RadixSelectCalcB4(VAL value, IDX index, TMP tmp, SRC src)
{
    PTO_RS_PREPARE;
    constexpr bool isUInt = std::is_same_v<SrcDType, uint32_t>;
    constexpr bool isFloat = !(std::is_same_v<SrcDType, uint32_t> || std::is_same_v<SrcDType, int32_t>);
    // Define memory address
    size_t point = tmp.GetAddr();
    size_t srcMaskAddr = DefineWorkSpace<uint32_t, srcTileH * srcMaskShape>(point);
    size_t sortTmpAddr_ = point;
    size_t srcTwiddleInAddr = DefineWorkSpace<SrcDType, srcTileH * srcTileW>(point);
    size_t cmpAddr = DefineWorkSpace<uint8_t, srcTileH * cmpAlign>(point);
    size_t high1Addr = DefineWorkSpace<uint8_t, srcTileH * 32>(point);
    size_t high2Addr = DefineWorkSpace<uint8_t, srcTileH * 32>(point);
    size_t high3Addr = DefineWorkSpace<uint8_t, srcTileH * 32>(point);
    size_t selectCountGTAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t rowMinAddr = selectCountGTAddr;
    size_t selectCountEQAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t gatherAddr = selectCountEQAddr;
    size_t remindKAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t kthValueAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t tmpAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t uselessAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t histogramAddr = DefineWorkSpace<uint32_t, srcTileH * 256>(point);
    size_t histogramTmpAddr = DefineWorkSpace<uint32_t, srcTileH * 256>(point);
    size_t tciAddr = DefineWorkSpace<uint32_t, srcTileH * 256>(point);
    point = histogramAddr;
    size_t selectGTAddr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point);
    size_t selectEQAddr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point);
    PTO_RS_SORT_ADDR_DEFINE(32);
    // Define tile
    PTO_RS_COMMON_TILE_DEFINE;
    auto twiddleIntKTile = DefineTile<ConvIntType, srcTileH, srcTileW>(srcShape[3], k, srcTwiddleInAddr);
    auto srcMaskInt32Tile = DefineTile<int32_t, srcTileH, srcMaskShape>(srcShape[3], srcShape[4], srcMaskAddr);
    auto srcMaskInt32MaxTile = DefineTile<int32_t, srcTileH, srcMaskShape>(srcShape[3], srcMaskShape, srcMaskAddr);
    auto srcMaskInt32KTile = DefineTile<int32_t, srcTileH, srcMaskShape>(srcShape[3], k, srcMaskAddr);
    auto srcMaskUInt32KTile = DefineTile<uint32_t, srcTileH, srcMaskShape>(srcShape[3], k, srcMaskAddr);
    auto high1Tile = DefineTile<uint8_t, 1>(1, 1, high1Addr);
    auto high2Tile = DefineTile<uint8_t, 2>(2, 1, high1Addr);
    auto high3Tile = DefineTile<uint8_t, 3>(3, 1, high1Addr);
    auto high1IntTile = DefineTile<int32_t, srcTileH>(srcShape[3], 8, high1Addr);
    auto high2IntTile = DefineTile<int32_t, srcTileH>(srcShape[3], 8, high2Addr);
    auto high3IntTile = DefineTile<int32_t, srcTileH>(srcShape[3], 8, high3Addr);
    auto hignMaxIntTile = DefineTile<int32_t, srcTileH * 3>(srcShape[3] * 3, 8, high1Addr);
    auto histogramUInt32Tile = DefineTile<uint32_t, srcTileH, 256>(srcShape[3], 256, histogramAddr);
    auto histogramInt32Tile = DefineTile<int32_t, srcTileH, 256>(srcShape[3], 256, histogramAddr);
    auto histogramInt32PreTile = DefineTile<int32_t, srcTileH, 256>(srcShape[3], 256, histogramAddr - 32);
    auto histogramTmpUInt32Tile = DefineTile<uint32_t, srcTileH, 256>(srcShape[3], 256, histogramTmpAddr);
    auto histogramTmpInt32Tile = DefineTile<int32_t, srcTileH, 256>(srcShape[3], 256, histogramTmpAddr);
    auto remindKTile = DefineTile<int32_t, srcTileH>(srcShape[3], 8, remindKAddr);
    auto kthValueIntTile = DefineTile<int32_t, srcTileH>(srcShape[3], 8, kthValueAddr);
    auto kthValueUInt32Tile = DefineTile<uint32_t, srcTileH>(srcShape[3], 8, kthValueAddr);
    auto tmpIntTile = DefineTile<int32_t, srcTileH>(srcShape[3], 8, tmpAddr);
    auto sortTempUInt32KTile = DefineTile<uint32_t, srcTileH, kAlign>(srcShape[3], k, sortTmpAddr);
    auto tciTile = DefineTile<uint32_t, srcTileH, 256>(srcShape[3], 256, tciAddr);
    PTO_RS_SORT_TILE_DEFINE(32);

    for (LoopVar n0Index = 0; n0Index < srcShape[0]; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < srcShape[1]; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < srcShape[2]; ++n2Index) {
                RadixSelectBatchAssign(src, value, index, n0Index, n1Index, n2Index, srcStride, valStride, idxStride,
                                       srcTypeSize, idxTypeSize, srcIntTile, valIntTile, idxTile);
                RadixSelectTCI(tciTile);
                RadixSelectTwiddle<isLargest, true, isUInt, isFloat, SrcDType>(twiddleIntTile, srcIntTile,
                                                                               srcMaskInt32Tile, cmpTile, uselessTile);
                pto::TEXPANDS(kthValueIntTile, static_cast<int32_t>(0));
                pto::TEXPANDS(hignMaxIntTile, 0);
                RadixSelectHistogramB4<pto::HistByte::BYTE_3>(histogramUInt32Tile, twiddleUIntTile, high1Tile);
                pto::TEXPANDS(remindKTile, static_cast<int32_t>(srcShape[4] - k));
                pto::TCMPS(cmpTile, histogramInt32Tile, static_cast<uint32_t>(srcShape[4] - k), pto::CmpMode::GT);
                pto::TMOV(histogramTmpUInt32Tile, tciTile);
                pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                pto::TROWMIN(high1IntTile, histogramTmpInt32Tile, uselessTile);
                pto::TSHLS(kthValueIntTile, high1IntTile, 24);
                pto::TCMPS(cmpTile, high1IntTile, 0, pto::CmpMode::NE);
                pto::TADDS(rowMinTile, high1IntTile, 7);
                RadixSelectGather(gatherIntTile, histogramInt32PreTile, rowMinTile, uselessTile);
                pto::TSELS(gatherIntTile, cmpTile, gatherIntTile, uselessTile, 0);
                pto::TSUB(remindKTile, remindKTile, gatherIntTile);
                RadixSelectHistogramB4<pto::HistByte::BYTE_2>(histogramUInt32Tile, twiddleUIntTile, high1Tile);
                RadixSelectCmps<pto::CmpMode::GT>(cmpTile, histogramInt32Tile, remindKTile);
                pto::TMOV(histogramTmpUInt32Tile, tciTile);
                pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                pto::TROWMIN(high2IntTile, histogramTmpInt32Tile, uselessTile);
                pto::TSHLS(tmpIntTile, high2IntTile, 16);
                pto::TOR(kthValueIntTile, kthValueIntTile, tmpIntTile);
                pto::TCMPS(cmpTile, high2IntTile, 0, pto::CmpMode::NE);
                pto::TADDS(rowMinTile, high2IntTile, 7);
                RadixSelectGather(gatherIntTile, histogramInt32PreTile, rowMinTile, uselessTile);
                pto::TSELS(gatherIntTile, cmpTile, gatherIntTile, uselessTile, 0);
                pto::TSUB(remindKTile, remindKTile, gatherIntTile);
                RadixSelectHistogramB4<pto::HistByte::BYTE_1>(histogramUInt32Tile, twiddleUIntTile, high2Tile);
                RadixSelectCmps<pto::CmpMode::GT>(cmpTile, histogramInt32Tile, remindKTile);
                pto::TMOV(histogramTmpUInt32Tile, tciTile);
                pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                pto::TROWMIN(high3IntTile, histogramTmpInt32Tile, uselessTile);
                pto::TSHLS(tmpIntTile, high3IntTile, 8);
                pto::TOR(kthValueIntTile, kthValueIntTile, tmpIntTile);
                pto::TCMPS(cmpTile, high3IntTile, 0, pto::CmpMode::NE);
                pto::TADDS(rowMinTile, high3IntTile, 7);
                RadixSelectGather(gatherIntTile, histogramInt32PreTile, rowMinTile, uselessTile);
                pto::TSELS(gatherIntTile, cmpTile, gatherIntTile, uselessTile, 0);
                pto::TSUB(remindKTile, remindKTile, gatherIntTile);
                RadixSelectHistogramB4<pto::HistByte::BYTE_0>(histogramUInt32Tile, twiddleUIntTile, high3Tile);
                RadixSelectCmps<pto::CmpMode::GT>(cmpTile, histogramInt32Tile, remindKTile);
                pto::TMOV(histogramTmpUInt32Tile, tciTile);
                pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                pto::TROWMIN(rowMinTile, histogramTmpInt32Tile, uselessTile);
                pto::TOR(kthValueIntTile, kthValueIntTile, rowMinTile);
                pto::TEXPANDS(srcMaskInt32MaxTile, static_cast<int32_t>(0));
                pto::TMOV(srcMaskInt32Tile, twiddleIntTile);
                RadixSelectFinalSelect(selectInt32GTTile, selectInt32EQTile, twiddleUIntTile, kthValueUInt32Tile,
                                       selectCountUInt32GTTile, selectCountUInt32EQTile, idxTile, uselessTile);
                pto::TEXPANDS(srcMaskInt32MaxTile, 0x7fff);
                RadixSelectGather(srcMaskInt32KTile, twiddleIntTile, idxTile, uselessTile);
                RadixSelectSortPrepare(sortTempInt32MaxTile, number0UInt32Tile, number1UInt32Tile, number2UInt32Tile,
                                       number3UInt32Tile);
                RadixSelectSortTwoBit<0, 32>(
                    srcMaskInt32KTile, srcMaskUInt32KTile, indexInt32Tile, sortTempInt32KTile, sortTempUInt32KTile,
                    uselessTile, select1Int32Tile, select2Int32Tile, select3Int32Tile, cnt1UInt32Tile, cnt2UInt32Tile,
                    cnt3UInt32Tile, number0UInt32Tile, number1UInt32Tile, number2UInt32Tile, number3UInt32Tile);
                RadixSelectGather(select1Int32Tile, idxTile, indexInt32Tile, uselessTile);
                pto::TMOV(idxTile, select1Int32Tile);
                RadixSelectTwiddle<isLargest, false, isUInt, isFloat, SrcDType>(valIntTile, srcMaskInt32KTile,
                                                                                twiddleIntKTile, cmpTile, uselessTile);
            }
        }
    }
}

} // namespace RadixSelectUtil

#endif // defined(PTO_NPU_ARCH_A5)
#endif // TILEOP_TILE_OPERATOR_RADIX_SELECT_B4__H
