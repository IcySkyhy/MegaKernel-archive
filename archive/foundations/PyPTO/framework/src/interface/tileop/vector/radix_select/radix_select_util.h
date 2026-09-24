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
 * \file radix_select_util.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_RADIX_SELECT_UTIL__H
#define TILEOP_TILE_OPERATOR_RADIX_SELECT_UTIL__H
#if defined(PTO_NPU_ARCH_A5)
#include "utils/layout.h"
#include "utils/tile_tensor.h"

namespace RadixSelectUtil {
template <size_t size = sizeof(uint8_t)>
struct IntBySize {
    using T = int8_t;
};
template <>
struct IntBySize<sizeof(uint16_t)> {
    using T = int16_t;
};
template <>
struct IntBySize<sizeof(uint32_t)> {
    using T = int32_t;
};
template <>
struct IntBySize<sizeof(uint64_t)> {
    using T = int64_t;
};
template <size_t size = sizeof(uint8_t)>
struct UIntBySize {
    using T = uint8_t;
};
template <>
struct UIntBySize<sizeof(uint16_t)> {
    using T = uint16_t;
};
template <>
struct UIntBySize<sizeof(uint32_t)> {
    using T = uint32_t;
};
template <>
struct UIntBySize<sizeof(uint64_t)> {
    using T = uint64_t;
};
template <typename T>
struct SignByType {
    using U = typename IntBySize<sizeof(T)>::T;
    static constexpr U value = U(1) << (sizeof(U) * 8 - 1);
};
template <typename T>
struct RSExchangeTile {
    using U = typename pto::Tile<T::Loc, typename T::DType, T::ColStride, T::RowStride, T::BFractal, -1, -1>;
};
template <typename T, size_t other = 1, size_t tile = 32ULL / sizeof(T), bool isCol = false>
struct RSTile {
    using U = typename std::conditional_t<
        isCol, typename pto::Tile<pto::TileType::Vec, T, tile, other, pto::BLayout::ColMajor, -1, -1>,
        typename pto::Tile<pto::TileType::Vec, T, other, tile, pto::BLayout::RowMajor, -1, -1> >;
};
template <typename T, size_t other = 1, size_t tile = 32ULL / sizeof(T), bool isCol = false>
__aicore__ inline auto DefineTile(size_t row, size_t col, size_t addr)
{
    typename RSTile<T, other, tile, isCol>::U t(row, col);
    pto::TASSIGN(t, addr);
    return t;
}
template <typename T, size_t other = 1, size_t tile = 32ULL / sizeof(T), bool isCol = false>
__aicore__ inline auto DefineTile(size_t row, size_t col)
{
    return typename RSTile<T, other, tile, isCol>::U(row, col);
}
template <typename T, size_t size>
__aicore__ inline size_t DefineWorkSpace(size_t& point)
{
    size_t result = point;
    point += size * sizeof(T);
    return result;
}
__aicore__ inline constexpr size_t AlignUp(size_t size, size_t align)
{
    if (align == 0) {
        return size;
    }
    return (size + align - 1) / align * align;
}
#define PTO_RS_GET_STRIDE(valName, layout)                                                                  \
    size_t valName[] = {static_cast<size_t>(layout.GetLayout().template GetStrideDim<DIM_1ST, MAX_DIMS>()), \
                        static_cast<size_t>(layout.GetLayout().template GetStrideDim<DIM_2ND, MAX_DIMS>()), \
                        static_cast<size_t>(layout.GetLayout().template GetStrideDim<DIM_3RD, MAX_DIMS>()), \
                        static_cast<size_t>(layout.GetLayout().template GetStrideDim<DIM_4TH, MAX_DIMS>())}
#define PTO_RS_GET_SHAPE(valName, layout)                                                                  \
    size_t valName[] = {static_cast<size_t>(layout.GetLayout().template GetShapeDim<DIM_1ST, MAX_DIMS>()), \
                        static_cast<size_t>(layout.GetLayout().template GetShapeDim<DIM_2ND, MAX_DIMS>()), \
                        static_cast<size_t>(layout.GetLayout().template GetShapeDim<DIM_3RD, MAX_DIMS>()), \
                        static_cast<size_t>(layout.GetLayout().template GetShapeDim<DIM_4TH, MAX_DIMS>()), \
                        static_cast<size_t>(layout.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>())}
#define PTO_RS_PREPARE                                                                 \
    using SrcDType = typename SRC::Type;                                               \
    using IdxDType = typename IDX::Type;                                               \
    constexpr auto srcTypeSize = sizeof(SrcDType);                                     \
    constexpr auto idxTypeSize = sizeof(IdxDType);                                     \
    using ConvUIntType = typename UIntBySize<srcTypeSize>::T;                          \
    using ConvIntType = typename IntBySize<srcTypeSize>::T;                            \
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<SRC, DIM_5TH, MAX_DIMS>(); \
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<SRC, DIM_4TH, MAX_DIMS>(); \
    constexpr auto valTileW = TileOp::GetTensorTileShapeDim<VAL, DIM_5TH, MAX_DIMS>(); \
    constexpr auto idxTileW = TileOp::GetTensorTileShapeDim<IDX, DIM_5TH, MAX_DIMS>(); \
    constexpr auto srcMaskShape = AlignUp(srcTileW, 128);                              \
    constexpr int64_t cmpSize = (srcTileW > 256 ? srcTileW : 256) / 8;                 \
    constexpr auto cmpAlign = AlignUp(cmpSize, 32);                                    \
    constexpr auto kAlign = AlignUp(k, 128);                                           \
    PTO_RS_GET_SHAPE(srcShape, src);                                                   \
    PTO_RS_GET_STRIDE(srcStride, src);                                                 \
    PTO_RS_GET_STRIDE(valStride, value);                                               \
    PTO_RS_GET_STRIDE(idxStride, index)
#define PTO_RS_SORT_ADDR_DEFINE(type)                                               \
    point = sortTmpAddr_;                                                           \
    size_t sortTmpAddr = DefineWorkSpace<uint##type##_t, srcTileH * kAlign>(point); \
    size_t number0Addr = DefineWorkSpace<uint16_t, srcTileH * 16>(point);           \
    size_t number1Addr = DefineWorkSpace<uint16_t, srcTileH * 16>(point);           \
    size_t number2Addr = DefineWorkSpace<uint16_t, srcTileH * 16>(point);           \
    size_t number3Addr = DefineWorkSpace<uint16_t, srcTileH * 16>(point);           \
    size_t cnt1Addr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);               \
    size_t cnt2Addr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);               \
    size_t cnt3Addr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);               \
    size_t select1Addr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point);       \
    size_t select2Addr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point);       \
    size_t select3Addr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point);       \
    size_t indexAddr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point)
#define PTO_RS_SORT_TILE_DEFINE(type)                                                                                \
    auto sortTempInt##type##KTile = DefineTile<int##type##_t, srcTileH, kAlign>(srcShape[3], k, sortTmpAddr);        \
    auto sortTempInt##type##MaxTile = DefineTile<int##type##_t, srcTileH, kAlign>(srcShape[3], kAlign, sortTmpAddr); \
    auto number0UInt##type##Tile = DefineTile<uint##type##_t>(srcShape[3], 256 / type, number0Addr);                 \
    auto number1UInt##type##Tile = DefineTile<uint##type##_t>(srcShape[3], 256 / type, number1Addr);                 \
    auto number2UInt##type##Tile = DefineTile<uint##type##_t>(srcShape[3], 256 / type, number2Addr);                 \
    auto number3UInt##type##Tile = DefineTile<uint##type##_t>(srcShape[3], 256 / type, number3Addr);                 \
    auto cnt1UInt32Tile = DefineTile<uint32_t, srcTileH>(srcShape[3], 8, cnt1Addr);                                  \
    auto cnt2UInt32Tile = DefineTile<uint32_t, srcTileH>(srcShape[3], 8, cnt2Addr);                                  \
    auto cnt3UInt32Tile = DefineTile<uint32_t, srcTileH>(srcShape[3], 8, cnt3Addr);                                  \
    auto select1Int32Tile = DefineTile<int32_t, srcTileH, kAlign>(srcShape[3], k, select1Addr);                      \
    auto select2Int32Tile = DefineTile<int32_t, srcTileH, kAlign>(srcShape[3], k, select2Addr);                      \
    auto select3Int32Tile = DefineTile<int32_t, srcTileH, kAlign>(srcShape[3], k, select3Addr);                      \
    auto indexInt32Tile = DefineTile<int32_t, srcTileH, kAlign>(srcShape[3], k, indexAddr)
#define PTO_RS_COMMON_TILE_DEFINE                                                                                  \
    auto srcIntTile = DefineTile<ConvIntType, srcTileH, srcTileW>(srcShape[3], srcShape[4]);                       \
    auto valIntTile = DefineTile<ConvIntType, srcTileH, valTileW>(srcShape[3], k);                                 \
    auto idxTile = DefineTile<IdxDType, srcTileH, idxTileW>(srcShape[3], k);                                       \
    auto cmpTile = DefineTile<uint8_t, srcTileH, cmpAlign>(srcShape[3], cmpSize, cmpAddr);                         \
    auto selectInt32GTTile = DefineTile<int32_t, srcTileH, kAlign>(srcShape[3], k, selectGTAddr);                  \
    auto selectInt32EQTile = DefineTile<int32_t, srcTileH, kAlign>(srcShape[3], k, selectEQAddr);                  \
    auto selectCountUInt32GTTile = DefineTile<uint32_t, srcTileH>(srcShape[3], 1, selectCountGTAddr);              \
    auto selectCountUInt32EQTile = DefineTile<uint32_t, srcTileH>(srcShape[3], 1, selectCountEQAddr);              \
    auto rowMinTile = DefineTile<int32_t, srcTileH>(srcShape[3], 8, rowMinAddr);                                   \
    auto gatherIntTile = DefineTile<int32_t, srcTileH>(srcShape[3], 8, gatherAddr);                                \
    auto uselessTile = DefineTile<uint32_t>(1, 1, uselessAddr);                                                    \
    auto twiddleIntTile = DefineTile<ConvIntType, srcTileH, srcTileW>(srcShape[3], srcShape[4], srcTwiddleInAddr); \
    auto twiddleUIntTile = DefineTile<ConvUIntType, srcTileH, srcTileW>(srcShape[3], srcShape[4], srcTwiddleInAddr)

template <pto::CmpMode cmpMode, typename SELECT, typename SRC, typename K, typename COUNT, typename USELESS>
TILEOP void RadixSelectGather(SELECT select, SRC src, K k, COUNT count, USELESS useless)
{
    auto validRow = select.GetValidRow();
    using T1 = typename RSExchangeTile<SELECT>::U;
    T1 selectOneRow(1, select.GetValidCol());
    using T2 = typename RSExchangeTile<SRC>::U;
    T2 srcOneRow(1, src.GetValidCol());
    using T3 = typename RSExchangeTile<K>::U;
    T3 kOneRow(1, k.GetValidCol());
    using T4 = typename RSExchangeTile<COUNT>::U;
    T4 countOneRow(1, count.GetValidCol());
    for (LoopVar n3Index = 0; n3Index < validRow; ++n3Index) {
        pto::TASSIGN(selectOneRow, (int64_t)(select.data() + n3Index * SELECT::RowStride));
        pto::TASSIGN(srcOneRow, (int64_t)(src.data() + n3Index * SRC::RowStride));
        pto::TASSIGN(kOneRow, (int64_t)(k.data() + n3Index * K::RowStride));
        pto::TASSIGN(countOneRow, (int64_t)(count.data() + n3Index * COUNT::RowStride));
        pto::TGATHER<T1, T2, T3, T4, USELESS, cmpMode>(selectOneRow, srcOneRow, kOneRow, countOneRow, useless, 0);
    }
}

template <typename DST, typename SRC, typename IDX, typename USELESS>
TILEOP void RadixSelectGather(DST dst, SRC src, IDX idx, USELESS useless)
{
    auto validRow = dst.GetValidRow();
    typename RSExchangeTile<DST>::U dstOneRow(1, dst.GetValidCol());
    typename RSExchangeTile<SRC>::U srcOneRow(1, src.GetValidCol());
    typename RSExchangeTile<IDX>::U idxOneRow(1, idx.GetValidCol());
    for (LoopVar n3Index = 0; n3Index < validRow; ++n3Index) {
        pto::TASSIGN(dstOneRow, (int64_t)(dst.data() + n3Index * DST::RowStride));
        pto::TASSIGN(srcOneRow, (int64_t)(src.data() + n3Index * SRC::RowStride));
        pto::TASSIGN(idxOneRow, (int64_t)(idx.data() + n3Index * IDX::RowStride));
        pto::TGATHER(dstOneRow, srcOneRow, idxOneRow, useless);
    }
}

template <pto::CmpMode cmpMode, typename DST, typename SRC, typename VAL>
TILEOP void RadixSelectCmps(DST dst, SRC src, VAL val)
{
    auto validRow = dst.GetValidRow();
    typename RSExchangeTile<DST>::U dstOneRow(1, dst.GetValidCol());
    typename RSExchangeTile<SRC>::U srcOneRow(1, src.GetValidCol());
    typename RSExchangeTile<VAL>::U valOneRow(1, val.GetValidCol());
    for (LoopVar n3Index = 0; n3Index < validRow; ++n3Index) {
        pto::TASSIGN(dstOneRow, (int64_t)(dst.data() + n3Index * DST::RowStride));
        pto::TASSIGN(srcOneRow, (int64_t)(src.data() + n3Index * SRC::RowStride));
        pto::TASSIGN(valOneRow, (int64_t)(val.data() + n3Index * VAL::RowStride));
        pto::TCMPS(dstOneRow, srcOneRow, valOneRow, cmpMode);
    }
}

template <pto::HistByte histByte, typename DST, typename SRC, typename IDX>
TILEOP void RadixSelectHistogram(DST dst, SRC src, IDX idx)
{
    auto validRow = dst.GetValidRow();
    typename RSExchangeTile<DST>::U dstOneRow(1, dst.GetValidCol());
    typename RSExchangeTile<SRC>::U srcOneRow(1, src.GetValidCol());
    typename RSExchangeTile<IDX>::U idxOneRow(1, 1);
    for (LoopVar n3Index = 0; n3Index < validRow; ++n3Index) {
        pto::TASSIGN(dstOneRow, (int64_t)(dst.data() + n3Index * DST::RowStride));
        pto::TASSIGN(srcOneRow, (int64_t)(src.data() + n3Index * SRC::RowStride));
        pto::TASSIGN(idxOneRow, (int64_t)(idx.data() + n3Index * IDX::ColStride));
        pto::THISTOGRAM<histByte>(dstOneRow, srcOneRow, idxOneRow);
    }
}

template <pto::HistByte histByte, typename DST, typename SRC, typename IDX>
TILEOP void RadixSelectHistogramB4(DST dst, SRC src, IDX idx)
{
    auto validRow = dst.GetValidRow();
    typename RSExchangeTile<DST>::U dstOneRow(1, dst.GetValidCol());
    using SrcTile = typename pto::Tile<SRC::Loc, typename SRC::DType, SRC::Rows, SRC::Rows * 32, SRC::BFractal, -1, -1>;
    SrcTile srcOneRow(1, src.GetValidCol());
    using IdxTile = typename pto::Tile<IDX::Loc, typename IDX::DType, IDX::Rows, SRC::Rows * 32, IDX::BFractal, -1, -1>;
    IdxTile idxOneRow(idx.GetValidRow(), 1);
    for (LoopVar n3Index = 0; n3Index < validRow; ++n3Index) {
        pto::TASSIGN(dstOneRow, (int64_t)(dst.data() + n3Index * DST::RowStride));
        pto::TASSIGN(srcOneRow, (int64_t)(src.data() + n3Index * SRC::RowStride));
        pto::TASSIGN(idxOneRow, (int64_t)(idx.data() + n3Index * IDX::RowStride));
        pto::THISTOGRAM<histByte>(dstOneRow, srcOneRow, idxOneRow);
    }
}

template <typename DST>
TILEOP void RadixSelectTCI(DST dst)
{
    auto validRow = dst.GetValidRow();
    using T = typename RSExchangeTile<DST>::U;
    T dstOneRowOrigin(1, dst.GetValidCol());
    pto::TASSIGN(dstOneRowOrigin, (int64_t)dst.data());
    pto::TCI<T, T, uint32_t, 0>(dstOneRowOrigin, static_cast<uint32_t>(0), dstOneRowOrigin);
    T dstOneRow(1, dst.GetValidCol());
    for (LoopVar n3Index = 1; n3Index < validRow; ++n3Index) {
        pto::TASSIGN(dstOneRow, (int64_t)(dst.data() + n3Index * DST::RowStride));
        pto::TMOV(dstOneRow, dstOneRowOrigin);
    }
}

template <typename DST, typename CMP, typename SRC, typename IDX, typename USELESS>
TILEOP void RadixSelectSel(DST dst, CMP cmp, SRC src, IDX idx, USELESS useless)
{
    auto validRow = dst.GetValidRow();
    using TileDefine = typename pto::Tile<pto::TileType::Vec, typename DST::DType, 1, DST::RowStride,
                                          pto::BLayout::RowMajor, -1, -1>;
    TileDefine dstOneRow(1, dst.GetValidCol());
    typename RSExchangeTile<CMP>::U cmpOneRow(1, cmp.GetValidCol());
    TileDefine srcOneRow(1, src.GetValidCol());
    TileDefine idxOneRow(1, idx.GetValidCol());
    for (LoopVar n3Index = 0; n3Index < validRow; ++n3Index) {
        pto::TASSIGN(dstOneRow, (int64_t)(dst.data() + n3Index * DST::RowStride));
        pto::TASSIGN(cmpOneRow, (int64_t)(cmp.data() + n3Index * CMP::RowStride));
        pto::TASSIGN(srcOneRow, (int64_t)(src.data() + n3Index * SRC::RowStride));
        pto::TASSIGN(idxOneRow, (int64_t)(idx.data() + n3Index * IDX::RowStride));
        pto::TSEL(dstOneRow, cmpOneRow, srcOneRow, idxOneRow, useless);
    }
}

template <typename SRC, typename VAL, typename IDX, typename SRCTILE, typename VALTILE, typename IDXTILE>
TILEOP void RadixSelectBatchAssign(SRC src, VAL value, IDX index, LoopVar n0Index, LoopVar n1Index, LoopVar n2Index,
                                   const size_t (&srcStride)[4], const size_t (&valStride)[4],
                                   const size_t (&idxStride)[4], size_t srcTypeSize, size_t idxTypeSize,
                                   SRCTILE& srcIntTile, VALTILE& valIntTile, IDXTILE& idxTile)
{
    uint64_t srcAddr = src.GetAddr() +
                       (n0Index * srcStride[0] + n1Index * srcStride[1] + n2Index * srcStride[2]) * srcTypeSize;
    uint64_t valAddr = value.GetAddr() +
                       (n0Index * valStride[0] + n1Index * valStride[1] + n2Index * valStride[2]) * srcTypeSize;
    uint64_t idxAddr = index.GetAddr() +
                       (n0Index * idxStride[0] + n1Index * idxStride[1] + n2Index * idxStride[2]) * idxTypeSize;
    pto::TASSIGN(srcIntTile, srcAddr);
    pto::TASSIGN(valIntTile, valAddr);
    pto::TASSIGN(idxTile, idxAddr);
}

template <typename GT, typename EQ, typename SRC, typename K, typename CGT, typename CEQ, typename IDX,
          typename USELESS>
TILEOP void RadixSelectFinalSelect(GT selectGT, EQ selectEQ, SRC srcMask, K kth, CGT countGT, CEQ countEQ, IDX idx,
                                   USELESS useless)
{
    RadixSelectGather<pto::CmpMode::GT>(selectGT, srcMask, kth, countGT, useless);
    RadixSelectGather<pto::CmpMode::EQ>(selectEQ, srcMask, kth, countEQ, useless);
    pto::TCONCAT(idx, selectGT, selectEQ, countGT, countEQ);
}

/*
Memory Usage:
srcMaskAddr       | srcMaskShape    uint16/uint32 |
sortTmpAddr       | kAlign          uint16/uint32 |
number0Addr       | 16              uint16        |
number1Addr       | 16              uint16        |
number2Addr       | 16              uint16        |
number3Addr       | 16              uint16        |
cnt1Addr          | 8               uint32        |
cnt2Addr          | 8               uint32        |
cnt3Addr          | 8               uint32        |
select1Addr       | kAlign          uint32        |
select2Addr       | kAlign          uint32        |
select3Addr       | kAlign          uint32        |
indexAddr         | kAlign          uint32        |
*/
template <int64_t bit, typename SRC, typename SRC2, typename IDX, typename TMP, typename TMP2, typename SELECT,
          typename COUNT, typename NUM, typename USELESS>
TILEOP void RadixSelectSortTwoBitCalc(SRC src, SRC2 src2, IDX index, TMP tmp, TMP2 tmp2, USELESS useless,
                                      SELECT select1, SELECT select2, SELECT select3, COUNT count1, COUNT count2,
                                      COUNT count3, NUM num0, NUM num1, NUM num2, NUM num3)
{
    pto::TSHRS(tmp, src, bit);
    pto::TANDS(tmp, tmp, 0x3);
    RadixSelectGather<pto::CmpMode::EQ>(select2, tmp2, num3, count2, useless);
    RadixSelectGather<pto::CmpMode::EQ>(select1, tmp2, num2, count1, useless);
    pto::TADD(count3, count1, count2);
    pto::TCONCAT(select3, select2, select1, count2, count1);
    RadixSelectGather<pto::CmpMode::EQ>(select1, tmp2, num1, count1, useless);
    pto::TADD(count2, count1, count3);
    pto::TCONCAT(select2, select3, select1, count3, count1);
    if constexpr (bit == 0) {
        RadixSelectGather<pto::CmpMode::EQ>(select3, tmp2, num0, count3, useless);
        pto::TCONCAT(select1, select2, select3, count2, count3);
        RadixSelectGather(tmp, src, select1, useless);
    } else {
        RadixSelectGather<pto::CmpMode::EQ>(select1, tmp2, num0, count1, useless);
        pto::TCONCAT(select3, select2, select1, count2, count1);
        RadixSelectGather(select1, index, select3, useless);
        RadixSelectGather(tmp, src, select3, useless);
    }
}

template <int64_t bit, int64_t lastBit, typename SRC, typename SRC2, typename IDX, typename TMP, typename TMP2,
          typename SELECT, typename COUNT, typename NUM, typename USELESS>
TILEOP void RadixSelectSortTwoBit(SRC src, SRC2 src2, IDX index, TMP tmp, TMP2 tmp2, USELESS useless, SELECT select1,
                                  SELECT select2, SELECT select3, COUNT count1, COUNT count2, COUNT count3, NUM num0,
                                  NUM num1, NUM num2, NUM num3)
{
    if constexpr (bit < lastBit) {
        if constexpr (bit % 4 == 0) {
            RadixSelectSortTwoBitCalc<bit>(src, src2, index, tmp, tmp2, useless, select1, select2, select3, count1,
                                           count2, count3, num0, num1, num2, num3);
        } else {
            RadixSelectSortTwoBitCalc<bit>(tmp, tmp2, select1, src, src2, useless, index, select2, select3, count1,
                                           count2, count3, num0, num1, num2, num3);
        }
        RadixSelectSortTwoBit<bit + 2, lastBit>(src, src2, index, tmp, tmp2, useless, select1, select2, select3, count1,
                                                count2, count3, num0, num1, num2, num3);
    }
}

template <typename TMP, typename NUM>
TILEOP void RadixSelectSortPrepare(TMP tmp, NUM num0, NUM num1, NUM num2, NUM num3)
{
    pto::TEXPANDS(tmp, 0x7fff);
    pto::TEXPANDS(num0, 0);
    pto::TEXPANDS(num1, 1);
    pto::TEXPANDS(num2, 2);
    pto::TEXPANDS(num3, 3);
}

template <bool isLargest, bool in, bool isUInt, bool isFloat, typename SrcDType, typename TWI, typename SRC,
          typename TMP, typename CMP, typename USELESS>
TILEOP void RadixSelectTwiddle(TWI twi, SRC src, TMP tmp, CMP cmp, USELESS useless)
{
    if constexpr (!isLargest && !in) {
        pto::TNOT(src, src);
    }
    constexpr auto SIGN = SignByType<SrcDType>::value;
    if constexpr (isFloat) {
        constexpr pto::CmpMode cmpMode = in ? pto::CmpMode::LT : pto::CmpMode::GE;
        pto::TCMPS(cmp, src, 0, cmpMode);
        pto::TXORS(tmp, src, SIGN, useless);
        pto::TNOT(twi, src);
        RadixSelectSel(twi, cmp, twi, tmp, useless);
    } else if constexpr (isUInt) {
        pto::TMOV(twi, src);
    } else {
        pto::TXORS(twi, src, SIGN, useless);
    }
    if constexpr (!isLargest && in) {
        pto::TNOT(twi, twi);
    }
}

} // namespace RadixSelectUtil

#endif // defined(PTO_NPU_ARCH_A5)
#endif // TILEOP_TILE_OPERATOR_RADIX_SELECT_UTIL__H
