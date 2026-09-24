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
#include <pto/common/constants.hpp>
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTSubS(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x28000);

    Event<Op::TLOAD, Op::TSUBS> event0;
    Event<Op::TSUBS, Op::TSTORE_VEC> event1;

    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TSUBS(dstTile, srcTile, scalar, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <
    typename T, int srcGmRow, int srcGmCol, int dstGmRow, int dstGmCol, int tileRow, int tileCol, int validRow,
    int validCol>
PTO_INTERNAL void runTSubSByColTile(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using SrcTileData = Tile<TileType::Vec, T, tileRow, tileCol, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, tileRow, tileCol, BLayout::RowMajor, -1, -1>;

    for (int colOffset = 0; colOffset < validCol; colOffset += tileCol) {
        int curCol = (validCol - colOffset) < tileCol ? (validCol - colOffset) : tileCol;
        GlobalData srcGlobal(src + colOffset, DynDim2Shape(validRow, curCol), DynDim2Stride(srcGmRow, srcGmCol));
        GlobalData dstGlobal(out + colOffset, DynDim2Shape(validRow, curCol), DynDim2Stride(dstGmRow, dstGmCol));
        SrcTileData srcTile(validRow, curCol);
        DstTileData dstTile(validRow, curCol);
        TASSIGN(srcTile, 0x0);
        TASSIGN(dstTile, 0x28000);

        Event<Op::TLOAD, Op::TSUBS> event0;
        Event<Op::TSUBS, Op::TSTORE_VEC> event1;

        event0 = TLOAD(srcTile, srcGlobal);
        event1 = TSUBS(dstTile, srcTile, scalar, event0);
        TSTORE(dstGlobal, dstTile, event1);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
#endif
    }
}

extern "C" __global__ AICORE void launchTSUBSCase1(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTSubS<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase2(__gm__ aclFloat16* out, __gm__ aclFloat16* src, float scalar)
{
    runTSubS<half, 63, 128, 63, 63, 64, 64>((__gm__ half*)out, (__gm__ half*)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase3(__gm__ int32_t* out, __gm__ int32_t* src, int32_t scalar)
{
    runTSubS<int32_t, 31, 256, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase4(__gm__ int16_t* out, __gm__ int16_t* src, int16_t scalar)
{
    runTSubS<int16_t, 15, 192, 15, 15, 192, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase5(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTSubS<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase6(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTSubS<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase7(__gm__ uint32_t* out, __gm__ uint32_t* src, uint32_t scalar)
{
    runTSubS<uint32_t, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase8(__gm__ uint16_t* out, __gm__ uint16_t* src, uint16_t scalar)
{
    runTSubS<uint16_t, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase9(__gm__ int8_t* out, __gm__ int8_t* src, int8_t scalar)
{
    runTSubS<int8_t, 256, 64, 256, 256, 32, 32>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase10(__gm__ uint8_t* out, __gm__ uint8_t* src, uint8_t scalar)
{
    runTSubS<uint8_t, 256, 64, 256, 256, 32, 32>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase11(__gm__ uint8_t* out, __gm__ uint8_t* src, uint8_t scalar)
{
    runTSubS<uint8_t, 1, 64, 1, 1, 32, 32>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase12(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTSubS<int64_t, 4, 16, 4, 4, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase13(__gm__ uint64_t* out, __gm__ uint64_t* src, uint64_t scalar)
{
    runTSubS<uint64_t, 4, 16, 4, 4, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase14(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTSubSByColTile<int64_t, 96, 32768, 32, 1024, 32, 128, 32, 1024>(out, src, scalar);
}

template <typename T, int cols>
PTO_INTERNAL void runTSubSWideInt64(__gm__ T* out, __gm__ T* src, T scalar)
{
    constexpr int tileRows = 1;
    constexpr int tileCols = 64;

    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;

    for (int col = 0; col < cols; col += tileCols) {
        int validCols = (col + tileCols <= cols) ? tileCols : (cols - col);
        GlobalData srcGlobal(src + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        GlobalData dstGlobal(out + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        TileData srcTile(tileRows, validCols);
        TileData dstTile(tileRows, validCols);
        TASSIGN(srcTile, 0x0);
        TASSIGN(dstTile, 0x2000);

        Event<Op::TLOAD, Op::TSUBS> event0 = TLOAD(srcTile, srcGlobal);
        Event<Op::TSUBS, Op::TSTORE_VEC> event1 = TSUBS(dstTile, srcTile, scalar, event0);
        TSTORE(dstGlobal, dstTile, event1);
        pipe_barrier(PIPE_ALL);
    }
}

extern "C" __global__ AICORE void launchTSUBSCase15(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTSubSWideInt64<int64_t, 16364>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase16(__gm__ uint64_t* out, __gm__ uint64_t* src, uint64_t scalar)
{
    runTSubSWideInt64<uint64_t, 16364>(out, src, scalar);
}

template <uint32_t caseId, typename T>
void launchTSUBSTestCase(void* out, void* src, T scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTSUBSCase1<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 2: {
            launchTSUBSCase2<<<1, nullptr, stream>>>((aclFloat16*)out, (aclFloat16*)src, scalar);
            break;
        }
        case 3: {
            launchTSUBSCase3<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src, scalar);
            break;
        }
        case 4: {
            launchTSUBSCase4<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src, scalar);
            break;
        }
        case 5: {
            launchTSUBSCase5<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 6: {
            launchTSUBSCase6<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 7: {
            launchTSUBSCase7<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src, scalar);
            break;
        }
        case 8: {
            launchTSUBSCase8<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src, scalar);
            break;
        }
        case 9: {
            launchTSUBSCase9<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src, scalar);
            break;
        }
        case 10: {
            launchTSUBSCase10<<<1, nullptr, stream>>>((uint8_t*)out, (uint8_t*)src, scalar);
            break;
        }
        case 11: {
            launchTSUBSCase11<<<1, nullptr, stream>>>((uint8_t*)out, (uint8_t*)src, scalar);
            break;
        }
        case 12: {
            launchTSUBSCase12<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 13: {
            launchTSUBSCase13<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src, (uint64_t)scalar);
            break;
        }
        case 14: {
            launchTSUBSCase14<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 15: {
            launchTSUBSCase15<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 16: {
            launchTSUBSCase16<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src, (uint64_t)scalar);
            break;
        }
        default: {
        }
    }
}

template <uint32_t caseId>
void launchTSUBSTestCase(void* out, void* src, float scalar, aclrtStream stream)
{
    launchTSUBSTestCase<caseId, float>(out, src, scalar, stream);
}

template void launchTSUBSTestCase<1, float>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<2, aclFloat16>(void*, void*, aclFloat16, aclrtStream);
template void launchTSUBSTestCase<3, int32_t>(void*, void*, int32_t, aclrtStream);
template void launchTSUBSTestCase<4, int16_t>(void*, void*, int16_t, aclrtStream);
template void launchTSUBSTestCase<5, float>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<6, float>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<7, uint32_t>(void*, void*, uint32_t, aclrtStream);
template void launchTSUBSTestCase<8, uint16_t>(void*, void*, uint16_t, aclrtStream);
template void launchTSUBSTestCase<9, int8_t>(void*, void*, int8_t, aclrtStream);
template void launchTSUBSTestCase<10, uint8_t>(void*, void*, uint8_t, aclrtStream);
template void launchTSUBSTestCase<11, uint8_t>(void*, void*, uint8_t, aclrtStream);
template void launchTSUBSTestCase<12, int64_t>(void*, void*, int64_t, aclrtStream);
template void launchTSUBSTestCase<13, uint64_t>(void*, void*, uint64_t, aclrtStream);
template void launchTSUBSTestCase<14, int64_t>(void*, void*, int64_t, aclrtStream);
template void launchTSUBSTestCase<15, int64_t>(void*, void*, int64_t, aclrtStream);
template void launchTSUBSTestCase<16, uint64_t>(void*, void*, uint64_t, aclrtStream);
template void launchTSUBSTestCase<1>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<2>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<3>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<4>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<5>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<6>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<7>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<8>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<9>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<10>(void*, void*, float, aclrtStream);
template void launchTSUBSTestCase<11>(void*, void*, float, aclrtStream);
