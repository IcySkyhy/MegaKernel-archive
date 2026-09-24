/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file where.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_WHERE__H
#define TILEOP_TILE_OPERATOR_WHERE__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <unsigned elementsCount>
TILEOP void CaculateMask(uint64_t condition, __ubuf__ half* castCondition, __ubuf__ half* compareCondition,
                         __ubuf__ uint8_t* vcmpBitResult, const unsigned curCount)
{
    constexpr unsigned bitsOfByte = 8;
    using TileCondition = pto::Tile<pto::TileType::Vec, uint8_t, 1, elementsCount, pto::BLayout::RowMajor, -1, -1>;
    using TileConditionHalf = pto::Tile<pto::TileType::Vec, half, 1, elementsCount, pto::BLayout::RowMajor, -1, -1>;
    using TileVCmpBitResult = pto::Tile<pto::TileType::Vec, uint8_t, 1, elementsCount / bitsOfByte,
                                        pto::BLayout::RowMajor, -1, -1>;

    TileCondition conditionTile(1, curCount);
    TileConditionHalf castConditionTile(1, curCount);
    TileConditionHalf compareConditionTile(1, curCount);
    TileVCmpBitResult vcmpBitResultTile(1, curCount / bitsOfByte);

    pto::TASSIGN(vcmpBitResultTile, (uint64_t)(vcmpBitResult));
    pto::TASSIGN(conditionTile, (uint64_t)condition);
    pto::TASSIGN(castConditionTile, (uint64_t)(castCondition));
    pto::TASSIGN(compareConditionTile, (uint64_t)(compareCondition));

    pto::TCVT(castConditionTile, conditionTile, pto::RoundMode::CAST_NONE);
    pto::TEXPANDS(compareConditionTile, (half)1.000000e+00f);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCMP(vcmpBitResultTile, castConditionTile, compareConditionTile, pto::CmpMode::EQ);
}

template <typename T, unsigned elementsCount>
TILEOP void ProcessWhere(uint64_t dst, uint64_t vcmpBitResult, uint64_t src0, uint64_t src1, uint64_t startAddrUB,
                         const unsigned curCount)
{
    constexpr unsigned bitsOfByte = 8;
    constexpr unsigned addressUsed = 4;
    constexpr unsigned alignUint8 = 32;
    using TileVCmpBitResult = pto::Tile<pto::TileType::Vec, uint8_t, 1, elementsCount / bitsOfByte,
                                        pto::BLayout::RowMajor, -1, -1>;
    TileVCmpBitResult vcmpBitResultTile(1, curCount / bitsOfByte);
    pto::TASSIGN(vcmpBitResultTile, (uint64_t)(vcmpBitResult));
    using TileStartAddrUB = pto::Tile<pto::TileType::Vec, uint8_t, 1, alignUint8, pto::BLayout::RowMajor, -1, -1>;
    TileStartAddrUB startAddrUBTile(1, addressUsed);
    pto::TASSIGN(startAddrUBTile, (uint64_t)(startAddrUB));

    using TileDst = pto::Tile<pto::TileType::Vec, T, 1, elementsCount, pto::BLayout::RowMajor, -1, -1>;
    TileDst dstTile(1, curCount);
    TileDst src0Tile(1, curCount);
    TileDst src1Tile(1, curCount);

    pto::TASSIGN(dstTile, (uint64_t)dst);
    pto::TASSIGN(src0Tile, (uint64_t)src0);
    pto::TASSIGN(src1Tile, (uint64_t)src1);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSEL(dstTile, vcmpBitResultTile, src0Tile, src1Tile, startAddrUBTile);
}

#define OP_TILE_OP_WHERETT TWhereTT
template <typename TDst, typename TTmp, typename TCond, typename TSrc0, typename TSrc1>
TILEOP void TWhereTT(TDst dst, TTmp tmpbuf, TCond condition, TSrc0 src0, TSrc1 src1)
{
    using ShapeValueType = typename Std::tuple_element<0, typename TDst::Shape>::type;
    constexpr auto shapeSize = Std::tuple_size<typename TDst::Shape>::value;

    constexpr unsigned elementsPerCount = 1024;
    constexpr unsigned bitsOfByte = 8;
    uint64_t tmpbufAddr = tmpbuf.GetAddr();
    __ubuf__ half* castCondition = reinterpret_cast<__ubuf__ half*>(tmpbufAddr);
    __ubuf__ half* compareCondition = castCondition + elementsPerCount;
    __ubuf__ uint8_t* vcmpBitResult = reinterpret_cast<__ubuf__ uint8_t*>(compareCondition + elementsPerCount);
    __ubuf__ uint8_t* startAddrUB = reinterpret_cast<__ubuf__ uint8_t*>(vcmpBitResult + elementsPerCount / bitsOfByte);

    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto conditionLayout = condition.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto shape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto shape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto shape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto shape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto conditionShape = condition.GetLayout().template GetShapeDim<4, expectSize>();

    auto stride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto stride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto stride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto stride3 = dstLayout.template GetStrideDim<3, expectSize>();
    auto conditionStride0 = conditionLayout.template GetStrideDim<0, expectSize>();
    auto conditionStride1 = conditionLayout.template GetStrideDim<1, expectSize>();
    auto conditionStride2 = conditionLayout.template GetStrideDim<2, expectSize>();
    auto conditionStride3 = conditionLayout.template GetStrideDim<3, expectSize>();

    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename TDst::TileShape>::type::value;
    constexpr auto conditionTileW = Std::tuple_element<shapeSize - 1, typename TCond::TileShape>::type::value;
    constexpr auto dstTypeSize = sizeof(typename TDst::Type);
    constexpr auto conditionTypeSize = sizeof(typename TCond::Type);
    constexpr auto src0TypeSize = sizeof(typename TSrc0::Type);
    constexpr auto src1TypeSize = sizeof(typename TSrc1::Type);

    unsigned numCountPerLine = shape4 / elementsPerCount;
    unsigned elementsRemainPerLine = shape4 % elementsPerCount;

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < shape3; ++n3Index) {
                    auto tileOffsets = TileOffset4Dim(n0Index, n1Index, n2Index, n3Index);
                    auto src0BaseOffset = GenTileOffset(src0, tileOffsets);
                    auto src1BaseOffset = GenTileOffset(src1, tileOffsets);
                    if constexpr (std::is_same_v<typename TCond::Type, bool>) {
                        for (LoopVar j = 0; j < numCountPerLine; j++) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   j * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + j * elementsPerCount;
                            auto src0Offset = src0BaseOffset + j * elementsPerCount;
                            auto src1Offset = src1BaseOffset + j * elementsPerCount;

                            CaculateMask<elementsPerCount>(
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize), castCondition,
                                compareCondition, vcmpBitResult, elementsPerCount);

                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize), (uint64_t)(vcmpBitResult),
                                (uint64_t)(src0.GetAddr() + src0Offset * src0TypeSize),
                                (uint64_t)(src1.GetAddr() + src1Offset * src1TypeSize), (uint64_t)(startAddrUB),
                                elementsPerCount);
                        }
                        if (elementsRemainPerLine) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   numCountPerLine * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + numCountPerLine * elementsPerCount;
                            auto src0Offset = src0BaseOffset + numCountPerLine * elementsPerCount;
                            auto src1Offset = src1BaseOffset + numCountPerLine * elementsPerCount;
                            CaculateMask<elementsPerCount>(
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize), castCondition,
                                compareCondition, vcmpBitResult, elementsRemainPerLine);

                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize), (uint64_t)(vcmpBitResult),
                                (uint64_t)(src0.GetAddr() + src0Offset * src0TypeSize),
                                (uint64_t)(src1.GetAddr() + src1Offset * src1TypeSize), (uint64_t)(startAddrUB),
                                elementsRemainPerLine);
                        }
                    } else {
                        constexpr unsigned addressUsed = 4;
                        constexpr unsigned alignUint8 = 32;
                        using TileStartAddrUB = pto::Tile<pto::TileType::Vec, uint8_t, 1, alignUint8,
                                                          pto::BLayout::RowMajor, -1, -1>;
                        TileStartAddrUB startAddrUBTile(1, addressUsed);
                        pto::TASSIGN(startAddrUBTile, (uint64_t)(startAddrUB));

                        using TileDst = pto::Tile<pto::TileType::Vec, typename TDst::Type, 1, tileW,
                                                  pto::BLayout::RowMajor, -1, -1>;
                        using TileMask = pto::Tile<pto::TileType::Vec, typename TCond::Type, 1, conditionTileW,
                                                   pto::BLayout::RowMajor, -1, -1>;
                        TileDst dstTile(1, shape4);
                        TileMask maskTile(1, conditionShape);
                        TileDst src0Tile(1, shape4);
                        TileDst src1Tile(1, shape4);

                        auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                               n2Index * conditionStride2 + n3Index * conditionStride3;
                        auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 + n3Index * stride3;

                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + offset * dstTypeSize));
                        pto::TASSIGN(maskTile, (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize));
                        pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0BaseOffset * src0TypeSize));
                        pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1BaseOffset * src1TypeSize));
                        pto::TSEL(dstTile, maskTile, src0Tile, src1Tile, startAddrUBTile);
                    }
                }
            }
        }
    }
}

#define OP_TILE_OP_WHERE_TS TWhereTS
template <typename TDst, typename TTmp, typename TCond, typename TSrc0, typename TSrc1>
TILEOP void TWhereTS(TDst dst, TTmp tmpbuf, TCond condition, TSrc0 src0, TSrc1 src1)
{
    using ShapeValueType = typename Std::tuple_element<0, typename TDst::Shape>::type;
    constexpr auto shapeSize = Std::tuple_size<typename TDst::Shape>::value;

    constexpr unsigned elementsPerCount = 1024;
    constexpr unsigned bitsOfByte = 8;
    uint64_t tmpbufAddr = tmpbuf.GetAddr();
    __ubuf__ half* castCondition = reinterpret_cast<__ubuf__ half*>(tmpbufAddr);
    __ubuf__ half* compareCondition = castCondition + elementsPerCount;
    __ubuf__ uint8_t* vcmpBitResult = reinterpret_cast<__ubuf__ uint8_t*>(compareCondition + elementsPerCount);
    __ubuf__ typename TDst::Type* otherTempTensor = (__ubuf__ typename TDst::Type*)(vcmpBitResult +
                                                                                    elementsPerCount / bitsOfByte);
    __ubuf__ uint8_t* startAddrUB = reinterpret_cast<__ubuf__ uint8_t*>(otherTempTensor + elementsPerCount);

    using TileTSrc1 = pto::Tile<pto::TileType::Vec, TSrc1, 1, elementsPerCount, pto::BLayout::RowMajor, -1, -1>;
    TileTSrc1 src1Tile(1, elementsPerCount);
    pto::TASSIGN(src1Tile, (uint64_t)(otherTempTensor));
    pto::TEXPANDS(src1Tile, src1);

    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto conditionLayout = condition.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto shape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto shape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto shape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto shape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto conditionShape = condition.GetLayout().template GetShapeDim<4, expectSize>();

    auto stride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto stride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto stride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto stride3 = dstLayout.template GetStrideDim<3, expectSize>();
    auto conditionStride0 = conditionLayout.template GetStrideDim<0, expectSize>();
    auto conditionStride1 = conditionLayout.template GetStrideDim<1, expectSize>();
    auto conditionStride2 = conditionLayout.template GetStrideDim<2, expectSize>();
    auto conditionStride3 = conditionLayout.template GetStrideDim<3, expectSize>();

    constexpr auto dstTypeSize = sizeof(typename TDst::Type);
    constexpr auto conditionTypeSize = sizeof(typename TCond::Type);
    constexpr auto src0TypeSize = sizeof(typename TSrc0::Type);

    unsigned numCountPerLine = shape4 / elementsPerCount;
    unsigned elementsRemainPerLine = shape4 % elementsPerCount;

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < shape3; ++n3Index) {
                    auto src0BaseOffset = GenTileOffset(src0, TileOffset4Dim(n0Index, n1Index, n2Index, n3Index));
                    if constexpr (std::is_same_v<typename TCond::Type, bool>) {
                        for (LoopVar j = 0; j < numCountPerLine; j++) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   j * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + j * elementsPerCount;
                            auto src0Offset = src0BaseOffset + j * elementsPerCount;

                            CaculateMask<elementsPerCount>(
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize), castCondition,
                                compareCondition, vcmpBitResult, elementsPerCount);

                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize), (uint64_t)(vcmpBitResult),
                                (uint64_t)(src0.GetAddr() + src0Offset * src0TypeSize), (uint64_t)(otherTempTensor),
                                (uint64_t)(startAddrUB), elementsPerCount);
                        }
                        if (elementsRemainPerLine) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   numCountPerLine * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + numCountPerLine * elementsPerCount;
                            auto src0Offset = src0BaseOffset + numCountPerLine * elementsPerCount;
                            CaculateMask<elementsPerCount>(
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize), castCondition,
                                compareCondition, vcmpBitResult, elementsRemainPerLine);

                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize), (uint64_t)(vcmpBitResult),
                                (uint64_t)(src0.GetAddr() + src0Offset * src0TypeSize), (uint64_t)(otherTempTensor),
                                (uint64_t)(startAddrUB), elementsRemainPerLine);
                        }
                    } else {
                        for (LoopVar j = 0; j < numCountPerLine; j++) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   j * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + j * elementsPerCount;
                            auto src0Offset = src0BaseOffset + j * elementsPerCount;
                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize),
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize),
                                (uint64_t)(src0.GetAddr() + src0Offset * src0TypeSize), (uint64_t)(otherTempTensor),
                                (uint64_t)(startAddrUB), elementsPerCount);
                        }
                        if (elementsRemainPerLine) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   numCountPerLine * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + numCountPerLine * elementsPerCount;
                            auto src0Offset = src0BaseOffset + numCountPerLine * elementsPerCount;
                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize),
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize),
                                (uint64_t)(src0.GetAddr() + src0Offset * src0TypeSize), (uint64_t)(otherTempTensor),
                                (uint64_t)(startAddrUB), elementsRemainPerLine);
                        }
                    }
                }
            }
        }
    }
}

#define OP_TILE_OP_WHERE_ST TWhereST
template <typename TDst, typename TTmp, typename TCond, typename TSrc0, typename TSrc1>
TILEOP void TWhereST(TDst dst, TTmp tmpbuf, TCond condition, TSrc0 src0, TSrc1 src1)
{
    using ShapeValueType = typename Std::tuple_element<0, typename TDst::Shape>::type;
    constexpr auto shapeSize = Std::tuple_size<typename TDst::Shape>::value;

    constexpr unsigned elementsPerCount = 1024;
    constexpr unsigned bitsOfByte = 8;
    uint64_t tmpbufAddr = tmpbuf.GetAddr();
    __ubuf__ half* castCondition = reinterpret_cast<__ubuf__ half*>(tmpbufAddr);
    __ubuf__ half* compareCondition = castCondition + elementsPerCount;
    __ubuf__ uint8_t* vcmpBitResult = reinterpret_cast<__ubuf__ uint8_t*>(compareCondition + elementsPerCount);
    __ubuf__ typename TDst::Type* inputTempTensor = (__ubuf__ typename TDst::Type*)(vcmpBitResult +
                                                                                    elementsPerCount / bitsOfByte);
    __ubuf__ uint8_t* startAddrUB = reinterpret_cast<__ubuf__ uint8_t*>(inputTempTensor + elementsPerCount);

    using TileTSrc0 = pto::Tile<pto::TileType::Vec, TSrc0, 1, elementsPerCount, pto::BLayout::RowMajor, -1, -1>;
    TileTSrc0 src0Tile(1, elementsPerCount);
    pto::TASSIGN(src0Tile, (uint64_t)(inputTempTensor));
    pto::TEXPANDS(src0Tile, src0);

    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto conditionLayout = condition.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto shape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto shape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto shape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto shape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto conditionShape = condition.GetLayout().template GetShapeDim<4, expectSize>();

    auto stride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto stride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto stride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto stride3 = dstLayout.template GetStrideDim<3, expectSize>();
    auto conditionStride0 = conditionLayout.template GetStrideDim<0, expectSize>();
    auto conditionStride1 = conditionLayout.template GetStrideDim<1, expectSize>();
    auto conditionStride2 = conditionLayout.template GetStrideDim<2, expectSize>();
    auto conditionStride3 = conditionLayout.template GetStrideDim<3, expectSize>();

    constexpr auto dstTypeSize = sizeof(typename TDst::Type);
    constexpr auto conditionTypeSize = sizeof(typename TCond::Type);
    constexpr auto src1TypeSize = sizeof(typename TSrc1::Type);

    unsigned numCountPerLine = shape4 / elementsPerCount;
    unsigned elementsRemainPerLine = shape4 % elementsPerCount;

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < shape3; ++n3Index) {
                    auto src1BaseOffset = GenTileOffset(src1, TileOffset4Dim(n0Index, n1Index, n2Index, n3Index));
                    if constexpr (std::is_same_v<typename TCond::Type, bool>) {
                        for (LoopVar j = 0; j < numCountPerLine; j++) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   j * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + j * elementsPerCount;
                            auto src1Offset = src1BaseOffset + j * elementsPerCount;

                            CaculateMask<elementsPerCount>(
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize), castCondition,
                                compareCondition, vcmpBitResult, elementsPerCount);

                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize), (uint64_t)(vcmpBitResult),
                                (uint64_t)(inputTempTensor), (uint64_t)(src1.GetAddr() + src1Offset * src1TypeSize),
                                (uint64_t)(startAddrUB), elementsPerCount);
                        }
                        if (elementsRemainPerLine) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   numCountPerLine * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + numCountPerLine * elementsPerCount;
                            auto src1Offset = src1BaseOffset + numCountPerLine * elementsPerCount;
                            CaculateMask<elementsPerCount>(
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize), castCondition,
                                compareCondition, vcmpBitResult, elementsRemainPerLine);

                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize), (uint64_t)(vcmpBitResult),
                                (uint64_t)(inputTempTensor), (uint64_t)(src1.GetAddr() + src1Offset * src1TypeSize),
                                (uint64_t)(startAddrUB), elementsRemainPerLine);
                        }
                    } else {
                        for (LoopVar j = 0; j < numCountPerLine; j++) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   j * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + j * elementsPerCount;
                            auto src1Offset = src1BaseOffset + j * elementsPerCount;
                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize),
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize),
                                (uint64_t)(inputTempTensor), (uint64_t)(src1.GetAddr() + src1Offset * src1TypeSize),
                                (uint64_t)(startAddrUB), elementsPerCount);
                        }
                        if (elementsRemainPerLine) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   numCountPerLine * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + numCountPerLine * elementsPerCount;
                            auto src1Offset = src1BaseOffset + numCountPerLine * elementsPerCount;
                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize),
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize),
                                (uint64_t)(inputTempTensor), (uint64_t)(src1.GetAddr() + src1Offset * src1TypeSize),
                                (uint64_t)(startAddrUB), elementsRemainPerLine);
                        }
                    }
                }
            }
        }
    }
}

#define OP_TILE_OP_WHERE_SS TWhereSS
template <typename TDst, typename TTmp, typename TCond, typename TSrc0, typename TSrc1>
TILEOP void TWhereSS(TDst dst, TTmp tmpbuf, TCond condition, TSrc0 src0, TSrc1 src1)
{
    using ShapeValueType = typename Std::tuple_element<0, typename TDst::Shape>::type;
    constexpr auto shapeSize = Std::tuple_size<typename TDst::Shape>::value;

    constexpr unsigned elementsPerCount = 1024;
    constexpr unsigned bitsOfByte = 8;
    uint64_t tmpbufAddr = tmpbuf.GetAddr();
    __ubuf__ half* castCondition = reinterpret_cast<__ubuf__ half*>(tmpbufAddr);
    __ubuf__ half* compareCondition = castCondition + elementsPerCount;
    __ubuf__ uint8_t* vcmpBitResult = reinterpret_cast<__ubuf__ uint8_t*>(compareCondition + elementsPerCount);
    __ubuf__ typename TDst::Type* inputTempTensor = (__ubuf__ typename TDst::Type*)(vcmpBitResult +
                                                                                    elementsPerCount / bitsOfByte);
    __ubuf__ typename TDst::Type* otherTempTensor = (__ubuf__ typename TDst::Type*)(inputTempTensor + elementsPerCount);
    __ubuf__ uint8_t* startAddrUB = reinterpret_cast<__ubuf__ uint8_t*>(otherTempTensor + elementsPerCount);

    using TileTSrc0 = pto::Tile<pto::TileType::Vec, TSrc0, 1, elementsPerCount, pto::BLayout::RowMajor, -1, -1>;
    TileTSrc0 src0Tile(1, elementsPerCount);
    TileTSrc0 src1Tile(1, elementsPerCount);
    pto::TASSIGN(src0Tile, (uint64_t)(inputTempTensor));
    pto::TASSIGN(src1Tile, (uint64_t)(otherTempTensor));
    pto::TEXPANDS(src0Tile, src0);
    pto::TEXPANDS(src1Tile, src1);

    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto conditionLayout = condition.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto shape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto shape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto shape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto shape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto conditionShape = condition.GetLayout().template GetShapeDim<4, expectSize>();

    auto stride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto stride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto stride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto stride3 = dstLayout.template GetStrideDim<3, expectSize>();
    auto conditionStride0 = conditionLayout.template GetStrideDim<0, expectSize>();
    auto conditionStride1 = conditionLayout.template GetStrideDim<1, expectSize>();
    auto conditionStride2 = conditionLayout.template GetStrideDim<2, expectSize>();
    auto conditionStride3 = conditionLayout.template GetStrideDim<3, expectSize>();

    constexpr auto dstTypeSize = sizeof(typename TDst::Type);
    constexpr auto conditionTypeSize = sizeof(typename TCond::Type);

    unsigned numCountPerLine = shape4 / elementsPerCount;
    unsigned elementsRemainPerLine = shape4 % elementsPerCount;

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < shape3; ++n3Index) {
                    if constexpr (std::is_same_v<typename TCond::Type, bool>) {
                        for (LoopVar j = 0; j < numCountPerLine; j++) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   j * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + j * elementsPerCount;

                            CaculateMask<elementsPerCount>(
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize), castCondition,
                                compareCondition, vcmpBitResult, elementsPerCount);

                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize), (uint64_t)(vcmpBitResult),
                                (uint64_t)(inputTempTensor), (uint64_t)(otherTempTensor), (uint64_t)(startAddrUB),
                                elementsPerCount);
                        }
                        if (elementsRemainPerLine) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   numCountPerLine * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + numCountPerLine * elementsPerCount;
                            CaculateMask<elementsPerCount>(
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize), castCondition,
                                compareCondition, vcmpBitResult, elementsRemainPerLine);

                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize), (uint64_t)(vcmpBitResult),
                                (uint64_t)(inputTempTensor), (uint64_t)(otherTempTensor), (uint64_t)(startAddrUB),
                                elementsRemainPerLine);
                        }
                    } else {
                        for (LoopVar j = 0; j < numCountPerLine; j++) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   j * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + j * elementsPerCount;
                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize),
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize),
                                (uint64_t)(inputTempTensor), (uint64_t)(otherTempTensor), (uint64_t)(startAddrUB),
                                elementsPerCount);
                        }
                        if (elementsRemainPerLine) {
                            auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 +
                                                   n2Index * conditionStride2 + n3Index * conditionStride3 +
                                                   numCountPerLine * elementsPerCount;
                            auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 +
                                          n3Index * stride3 + numCountPerLine * elementsPerCount;
                            ProcessWhere<typename TDst::Type, elementsPerCount>(
                                (uint64_t)(dst.GetAddr() + offset * dstTypeSize),
                                (uint64_t)(condition.GetAddr() + conditionOffset * conditionTypeSize),
                                (uint64_t)(inputTempTensor), (uint64_t)(otherTempTensor), (uint64_t)(startAddrUB),
                                elementsRemainPerLine);
                        }
                    }
                }
            }
        }
    }
}

#endif
