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
 * \file uniform.h
 * \brief Uniform random number generator implementation
 */

#ifndef TILEOP_TILE_OPERATOR_UNIFORM__H
#define TILEOP_TILE_OPERATOR_UNIFORM__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#if defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
#define OP_TILE_OP_UNIFORM TUniform
template <typename TDst, typename TTmp>
TILEOP void TUniform(TDst dst, TTmp tmpbuf, uint64_t key, uint64_t counter0, uint64_t counter1, uint16_t rounds)
{
    using ShapeValueType = typename Std::tuple_element<0, typename TDst::Shape>::type;
    constexpr auto shapeSize = Std::tuple_size<typename TDst::Shape>::value;
    constexpr int Size = Std::tuple_element<shapeSize - 1, typename TDst::TileShape>::type::value;
    constexpr int tileW = (Size + 7) / 8 * 8;
    constexpr size_t ALIGN_SIZE = 32;

    uint64_t tileCounter[2] = {counter0, counter1};

    uint64_t tmpbufAddr = tmpbuf.GetAddr();
    __ubuf__ uint32_t* uint32Buffer = reinterpret_cast<__ubuf__ uint32_t*>(tmpbufAddr);

    using TileUint32 = pto::Tile<pto::TileType::Vec, uint32_t, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    TileUint32 uint32Tile(1, Size);
    TileUint32 dstUint32Tile(1, Size);
    pto::TASSIGN(uint32Tile, (uint64_t)uint32Buffer);

    pto::TRandomKey uniformKey = {static_cast<uint32_t>(key & 0xFFFFFFFF), static_cast<uint32_t>(key >> 32)};
    pto::TRandomCounter uniformCounter = {
        static_cast<uint32_t>(tileCounter[0] & 0xFFFFFFFF), static_cast<uint32_t>(tileCounter[0] >> 32),
        static_cast<uint32_t>(tileCounter[1] & 0xFFFFFFFF), static_cast<uint32_t>(tileCounter[1] >> 32)};

    if (rounds == 7) {
        pto::TRANDOM<7>(uint32Tile, uniformKey, uniformCounter);
    } else {
        pto::TRANDOM<10>(uint32Tile, uniformKey, uniformCounter);
    }

    using DstType = typename TDst::Type;
    constexpr bool isFloat = std::is_same_v<DstType, float>;
    constexpr bool isHalf = std::is_same_v<DstType, half>;
    constexpr bool isBfloat16 = std::is_same_v<DstType, bfloat16_t>;

    if constexpr (isFloat) {
        pto::TASSIGN(dstUint32Tile, (uint64_t)dst.GetAddr());

        pto::TANDS(dstUint32Tile, uint32Tile, 0x7fffff);
        pto::TORS(uint32Tile, dstUint32Tile, 0x3f800000);

        using TileFloat = pto::Tile<pto::TileType::Vec, float, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileFloat floatTile(1, Size);
        pto::TASSIGN(floatTile, (uint64_t)uint32Buffer);

        using TileDst = pto::Tile<pto::TileType::Vec, DstType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileDst dstTile(1, Size);
        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr()));
        pto::TSUBS(dstTile, floatTile, 1.0f);
    } else if constexpr (isHalf || isBfloat16) {
        constexpr int64_t uint32BufferBytes = ((Size * sizeof(uint32_t) + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        __ubuf__ uint32_t* uint32BufferLow = reinterpret_cast<__ubuf__ uint32_t*>(tmpbufAddr + uint32BufferBytes);

        TileUint32 uint32TileLow(1, Size);
        pto::TASSIGN(uint32TileLow, (uint64_t)uint32BufferLow);

        pto::TANDS(uint32TileLow, uint32Tile, 0xFFFF);

        __ubuf__ uint16_t* uint16Buffer = reinterpret_cast<__ubuf__ uint16_t*>(tmpbufAddr);

        using TileUint16 = pto::Tile<pto::TileType::Vec, uint16_t, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileUint16 uint16Tile(1, Size);
        TileUint16 dstUint16Tile(1, Size);
        pto::TASSIGN(uint16Tile, (uint64_t)uint16Buffer);
        pto::TASSIGN(dstUint16Tile, (uint64_t)dst.GetAddr());

        pto::TCVT(uint16Tile, uint32TileLow, pto::RoundMode::CAST_NONE);

        if constexpr (isHalf) {
            pto::TANDS(dstUint16Tile, uint16Tile, 0x3ff);
            pto::TORS(uint16Tile, dstUint16Tile, 0x3c00);
        } else {
            pto::TANDS(dstUint16Tile, uint16Tile, 0x7f);
            pto::TORS(uint16Tile, dstUint16Tile, 0x3f80);
        }

        __ubuf__ DstType* resultBuffer = reinterpret_cast<__ubuf__ DstType*>(uint16Buffer);

        using TileResult = pto::Tile<pto::TileType::Vec, DstType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileResult resultTile(1, Size);
        pto::TASSIGN(resultTile, (uint64_t)resultBuffer);

        using TileDst = pto::Tile<pto::TileType::Vec, DstType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileDst dstTile(1, Size);
        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr()));

        if constexpr (isHalf) {
            pto::TSUBS(dstTile, resultTile, static_cast<half>(1.0));
        } else {
            pto::TSUBS(dstTile, resultTile, static_cast<bfloat16_t>(1.0));
        }
    }
}
#endif

#endif
