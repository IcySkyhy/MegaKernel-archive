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
 * \file hypot.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_HYPOT__H
#define TILEOP_TILE_OPERATOR_HYPOT__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <type_traits>

template <typename T>
struct HypotTmpBuffers {
    __ubuf__ void* buf0;
    __ubuf__ void* buf1;

    __ubuf__ float* fp32Buf0;
    __ubuf__ float* fp32Buf1;
};

template <typename T, typename TTmp>
TILEOP HypotTmpBuffers<T> InitHypotTmpBuffers(TTmp tmpbuf, size_t elementCount)
{
    uint64_t dataSizeBytes = elementCount * sizeof(float);
    const uint32_t ALIGNMENT = 32;
    uint64_t alignedSizeBytes = (dataSizeBytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    uint64_t tmpbufAddr = tmpbuf.GetAddr();
    __ubuf__ uint8_t* basePtr = reinterpret_cast<__ubuf__ uint8_t*>(tmpbufAddr);

    HypotTmpBuffers<T> buffers;
    buffers.fp32Buf0 = reinterpret_cast<__ubuf__ float*>(basePtr);
    buffers.buf0 = reinterpret_cast<__ubuf__ void*>(basePtr);
    __ubuf__ uint8_t* ptrBuf1 = basePtr + alignedSizeBytes;

    buffers.fp32Buf1 = reinterpret_cast<__ubuf__ float*>(ptrBuf1);
    buffers.buf1 = reinterpret_cast<__ubuf__ void*>(ptrBuf1);

    return buffers;
}

struct HypotLayoutInfo {
    size_t shape0, shape1, shape2, shape3, shape4, dstShape;
    size_t stride0, stride1, stride2, stride3;
    size_t stride1_0, stride1_1, stride1_2, stride1_3;
    size_t dstStride0, dstStride1, dstStride2, dstStride3;
};

template <typename T, typename TDst>
TILEOP HypotLayoutInfo ExtractHypotLayoutInfo(const T& src0, const T& src1, const TDst& dst)
{
    constexpr size_t expectSize = 5;
    const auto src0Layout = src0.GetLayout();
    const auto src1Layout = src1.GetLayout();
    const auto dstLayout = dst.GetLayout();

    HypotLayoutInfo info;
    info.shape0 = src0Layout.template GetShapeDim<0, expectSize>();
    info.shape1 = src0Layout.template GetShapeDim<1, expectSize>();
    info.shape2 = src0Layout.template GetShapeDim<2, expectSize>();
    info.shape3 = src0Layout.template GetShapeDim<3, expectSize>();
    info.shape4 = src0Layout.template GetShapeDim<4, expectSize>();
    info.dstShape = dstLayout.template GetShapeDim<4, expectSize>();

    info.stride0 = src0Layout.template GetStrideDim<0, expectSize>();
    info.stride1 = src0Layout.template GetStrideDim<1, expectSize>();
    info.stride2 = src0Layout.template GetStrideDim<2, expectSize>();
    info.stride3 = src0Layout.template GetStrideDim<3, expectSize>();

    info.stride1_0 = src1Layout.template GetStrideDim<0, expectSize>();
    info.stride1_1 = src1Layout.template GetStrideDim<1, expectSize>();
    info.stride1_2 = src1Layout.template GetStrideDim<2, expectSize>();
    info.stride1_3 = src1Layout.template GetStrideDim<3, expectSize>();

    info.dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    info.dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    info.dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    info.dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();

    return info;
}

// result = max * sqrt(1 + (min/max)^2)
template <typename TileType>
TILEOP void ExecuteHypotRobust(TileType& dstTile, TileType& src0Tile, TileType& src1Tile, TileType& maxTile,
                               TileType& minTile)
{
    pto::TABS(src0Tile, src0Tile);
    pto::TABS(src1Tile, src1Tile);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    pto::TMAX(maxTile, src0Tile, src1Tile);
    pto::TMIN(minTile, src0Tile, src1Tile);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    pto::TADDS(maxTile, maxTile, 1e-9f);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    pto::TDIV(minTile, minTile, maxTile);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    pto::TMUL(minTile, minTile, minTile);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    pto::TADDS(minTile, minTile, 1.0f);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    pto::TSQRT(minTile, minTile);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    pto::TMUL(dstTile, maxTile, minTile);
}

// result = (half) sqrt( (float)src0^2 + (float)src1^2 )
template <typename DstTileType, typename SrcTileType, typename Fp32TileType>
TILEOP void ExecuteHypotFp16(DstTileType& dstTile, SrcTileType& src0Tile, SrcTileType& src1Tile, Fp32TileType& tmp0Fp32,
                             Fp32TileType& tmp1Fp32)
{
    pto::TCVT(tmp0Fp32, src0Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp1Fp32, src1Tile, pto::RoundMode::CAST_NONE);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    pto::TMUL(tmp0Fp32, tmp0Fp32, tmp0Fp32);
    pto::TMUL(tmp1Fp32, tmp1Fp32, tmp1Fp32);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    pto::TADD(tmp0Fp32, tmp0Fp32, tmp1Fp32);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    pto::TSQRT(tmp0Fp32, tmp0Fp32);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif

    pto::TCVT(dstTile, tmp0Fp32, pto::RoundMode::CAST_NONE);
}

TILEOP void CalcHypotOffsets(const HypotLayoutInfo& info, size_t n0Index, size_t n1Index, size_t n2Index,
                             size_t n3Index, size_t& src0Offset, size_t& src1Offset, size_t& dstOffset)
{
    dstOffset = n0Index * info.dstStride0 + n1Index * info.dstStride1 + n2Index * info.dstStride2 +
                n3Index * info.dstStride3;
    src0Offset = n0Index * info.stride0 + n1Index * info.stride1 + n2Index * info.stride2 + n3Index * info.stride3;
    src1Offset = n0Index * info.stride1_0 + n1Index * info.stride1_1 + n2Index * info.stride1_2 +
                 n3Index * info.stride1_3;
}

template <typename T, typename TDst, typename TTmp>
TILEOP void THypot(TDst dst, T src0, T src1, TTmp tmpbuf)
{
    auto info = ExtractHypotLayoutInfo(src0, src1, dst);
    auto buffers = InitHypotTmpBuffers<T>(tmpbuf, info.shape4);

    constexpr auto dataTypeSize = sizeof(typename T::Type);
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T, 4, 5>();
    using DataTile = pto::Tile<pto::TileType::Vec, typename T::Type, 1, srcTileW, pto::BLayout::RowMajor, -1, -1>;
    using Fp32Tile = pto::Tile<pto::TileType::Vec, float, 1, srcTileW, pto::BLayout::RowMajor, -1, -1>;

    for (LoopVar n0Index = 0; n0Index < info.shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < info.shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < info.shape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < info.shape3; ++n3Index) {
                    size_t src0Offset, src1Offset, dstOffset;
                    CalcHypotOffsets(info, n0Index, n1Index, n2Index, n3Index, src0Offset, src1Offset, dstOffset);

                    uint64_t dstAddr = dst.GetAddr() + dstOffset * dataTypeSize;
                    uint64_t src0Addr = src0.GetAddr() + src0Offset * dataTypeSize;
                    uint64_t src1Addr = src1.GetAddr() + src1Offset * dataTypeSize;

                    DataTile dstTile(1, info.dstShape);
                    DataTile src0Tile(1, info.shape4);
                    DataTile src1Tile(1, info.shape4);

                    pto::TASSIGN(dstTile, dstAddr);
                    pto::TASSIGN(src0Tile, src0Addr);
                    pto::TASSIGN(src1Tile, src1Addr);

                    if constexpr (std::is_same_v<typename T::Type, float>) {
                        DataTile maxTile(1, info.shape4);
                        DataTile minTile(1, info.shape4);
                        pto::TASSIGN(maxTile, reinterpret_cast<uint64_t>(buffers.buf0));
                        pto::TASSIGN(minTile, reinterpret_cast<uint64_t>(buffers.buf1));
                        ExecuteHypotRobust(dstTile, src0Tile, src1Tile, maxTile, minTile);
                    } else {
                        Fp32Tile tmp0Fp32(1, info.shape4);
                        Fp32Tile tmp1Fp32(1, info.shape4);
                        pto::TASSIGN(tmp0Fp32, reinterpret_cast<uint64_t>(buffers.fp32Buf0));
                        pto::TASSIGN(tmp1Fp32, reinterpret_cast<uint64_t>(buffers.fp32Buf1));
                        ExecuteHypotFp16(dstTile, src0Tile, src1Tile, tmp0Fp32, tmp1Fp32);
                    }

#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                }
            }
        }
    }
}

#define OP_TILE_OP_HYPOT THypot

#endif
