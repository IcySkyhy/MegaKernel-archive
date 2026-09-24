/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <
    typename T, int SrcRows, int SrcCols, int W0Rows, int W0Cols, int W1Rows, int W1Cols, int V0Rows, int V0Cols,
    int V1Rows, int V1Cols, bool ZeroDst = false>
__global__ AICORE void runTExtractNd2xNz(
    __gm__ T __out__* out0, __gm__ T __out__* out1, __gm__ T __in__* src, uint16_t ir0, uint16_t ic0, uint16_t ir1,
    uint16_t ic1)
{
    constexpr int c0 = BLOCK_BYTE_SIZE / sizeof(T);

    using SrcShape = Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStride = pto::Stride<1, 1, 1, SrcCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    using Dst0Shape = Shape<1, W0Cols / c0, W0Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0>;
    using Dst0Stride = pto::Stride<W0Cols / c0 * W0Rows * c0, W0Rows * c0, FRACTAL_NZ_ROW * c0, c0, 1>;
    using Dst0Global = GlobalTensor<T, Dst0Shape, Dst0Stride, Layout::NZ>;
    using Dst1Shape = Shape<1, W1Cols / c0, W1Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0>;
    using Dst1Stride = pto::Stride<W1Cols / c0 * W1Rows * c0, W1Rows * c0, FRACTAL_NZ_ROW * c0, c0, 1>;
    using Dst1Global = GlobalTensor<T, Dst1Shape, Dst1Stride, Layout::NZ>;

    using SrcTile = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor, -1, -1>;
    using Dst0Tile = Tile<
        TileType::Vec, T, W0Rows, W0Cols, BLayout::ColMajor, V0Rows, V0Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>;
    using Dst1Tile = Tile<
        TileType::Vec, T, W1Rows, W1Cols, BLayout::ColMajor, V1Rows, V1Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>;

    SrcTile srcTile(SrcRows, SrcCols);
    Dst0Tile dst0Tile;
    Dst1Tile dst1Tile;
    constexpr uint32_t kAlign = 0x100;
    constexpr uint32_t srcBytes = static_cast<uint32_t>(SrcRows) * SrcCols * sizeof(T);
    constexpr uint32_t dst0Bytes = static_cast<uint32_t>(W0Rows) * W0Cols * sizeof(T);
    constexpr uint32_t dst1Bytes = static_cast<uint32_t>(W1Rows) * W1Cols * sizeof(T);
    constexpr uint32_t dst0Addr = ((srcBytes + kAlign - 1) / kAlign) * kAlign;
    constexpr uint32_t dst1Addr = dst0Addr + ((dst0Bytes + kAlign - 1) / kAlign) * kAlign;
    static_assert(dst1Addr + dst1Bytes <= TMP_UB_OFFSET);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dst0Tile, dst0Addr);
    TASSIGN(dst1Tile, dst1Addr);

    SrcGlobal srcGlobal(src);
    Dst0Global dst0Global(out0);
    Dst1Global dst1Global(out1);

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#endif

    if constexpr (ZeroDst) {
        __ubuf__ uint16_t* z0 = (__ubuf__ uint16_t*)(uintptr_t)dst0Addr;
        __ubuf__ uint16_t* z1 = (__ubuf__ uint16_t*)(uintptr_t)dst1Addr;
        constexpr uint32_t n0 = dst0Bytes / sizeof(uint16_t);
        constexpr uint32_t n1 = dst1Bytes / sizeof(uint16_t);
        set_mask_count();
        set_vector_mask(0, n0);
        vector_dup(z0, (uint16_t)0, 0, 1, 1, 8, 8);
        set_vector_mask(0, n1);
        vector_dup(z1, (uint16_t)0, 0, 1, 1, 8, 8);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }

    TEXTRACT(dst0Tile, dst1Tile, srcTile, ir0, ic0, ir1, ic1);

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dst0Global, dst0Tile);
    TSTORE(dst1Global, dst1Tile);
    out0 = dst0Global.data();
    out1 = dst1Global.data();
}

template <typename T>
static void dispatchTExtractNd2xNz(
    uint8_t* out0, uint8_t* out1, uint8_t* src, uint16_t ir0, uint16_t ic0, uint16_t ir1, uint16_t ic1, void* stream)
{
    runTExtractNd2xNz<T, 64, 128, 32, 64, 32, 64, 32, 64, 32, 64>
        <<<1, nullptr, stream>>>((T*)out0, (T*)out1, (T*)src, ir0, ic0, ir1, ic1);
}

template <typename T>
static void dispatchTExtractNd2xNzOddValid(
    uint8_t* out0, uint8_t* out1, uint8_t* src, uint16_t ir0, uint16_t ic0, uint16_t ir1, uint16_t ic1, void* stream)
{
    runTExtractNd2xNz<T, 64, 128, 32, 64, 32, 64, 32, 63, 32, 61, true>
        <<<1, nullptr, stream>>>((T*)out0, (T*)out1, (T*)src, ir0, ic0, ir1, ic1);
}

template <typename T>
static void dispatchTExtractNd2xNz1x1(
    uint8_t* out0, uint8_t* out1, uint8_t* src, uint16_t ir0, uint16_t ic0, uint16_t ir1, uint16_t ic1, void* stream)
{
    constexpr int c0 = BLOCK_BYTE_SIZE / sizeof(T);
    runTExtractNd2xNz<T, 64, 128, FRACTAL_NZ_ROW, c0, FRACTAL_NZ_ROW, c0, 1, 1, 1, 1>
        <<<1, nullptr, stream>>>((T*)out0, (T*)out1, (T*)src, ir0, ic0, ir1, ic1);
}

void launchTExtractNd2xNz(
    int key, uint8_t* out0, uint8_t* out1, uint8_t* src, uint16_t ir0, uint16_t ic0, uint16_t ir1, uint16_t ic1,
    void* stream)
{
    switch (key) {
        case 0:
            dispatchTExtractNd2xNz<half>(out0, out1, src, ir0, ic0, ir1, ic1, stream);
            break;
        case 1:
            dispatchTExtractNd2xNz<float>(out0, out1, src, ir0, ic0, ir1, ic1, stream);
            break;
        case 2:
            dispatchTExtractNd2xNz<bfloat16_t>(out0, out1, src, ir0, ic0, ir1, ic1, stream);
            break;
        case 3:
            dispatchTExtractNd2xNz<int8_t>(out0, out1, src, ir0, ic0, ir1, ic1, stream);
            break;
        case 4:
            dispatchTExtractNd2xNz<int32_t>(out0, out1, src, ir0, ic0, ir1, ic1, stream);
            break;
        default:
            break;
    }
}

void launchTExtractNd2xNzOddValid(
    int key, uint8_t* out0, uint8_t* out1, uint8_t* src, uint16_t ir0, uint16_t ic0, uint16_t ir1, uint16_t ic1,
    void* stream)
{
    switch (key) {
        case 3:
            dispatchTExtractNd2xNzOddValid<int8_t>(out0, out1, src, ir0, ic0, ir1, ic1, stream);
            break;
        default:
            break;
    }
}

void launchTExtractNd2xNz1x1(
    int key, uint8_t* out0, uint8_t* out1, uint8_t* src, uint16_t ir0, uint16_t ic0, uint16_t ir1, uint16_t ic1,
    void* stream)
{
    switch (key) {
        case 0:
            dispatchTExtractNd2xNz1x1<half>(out0, out1, src, ir0, ic0, ir1, ic1, stream);
            break;
        case 1:
            dispatchTExtractNd2xNz1x1<float>(out0, out1, src, ir0, ic0, ir1, ic1, stream);
            break;
        case 3:
            dispatchTExtractNd2xNz1x1<int8_t>(out0, out1, src, ir0, ic0, ir1, ic1, stream);
            break;
        default:
            break;
    }
}
