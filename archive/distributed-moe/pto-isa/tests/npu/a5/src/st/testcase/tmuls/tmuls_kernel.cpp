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
PTO_INTERNAL void runTMuls(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));

    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x26000);

    Event<Op::TLOAD, Op::TMULS> event0;
    Event<Op::TMULS, Op::TSTORE_VEC> event1;
    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TMULS(dstTile, srcTile, scalar, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <
    typename T, int srcGmRow, int srcGmCol, int dstGmRow, int dstGmCol, int tileRow, int tileCol, int validRow,
    int validCol>
PTO_INTERNAL void runTMulsByColTile(__gm__ T* out, __gm__ T* src, T scalar)
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
        TASSIGN(dstTile, 0x26000);

        Event<Op::TLOAD, Op::TMULS> event0;
        Event<Op::TMULS, Op::TSTORE_VEC> event1;
        event0 = TLOAD(srcTile, srcGlobal);
        event1 = TMULS(dstTile, srcTile, scalar, event0);
        TSTORE(dstGlobal, dstTile, event1);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
#endif
    }
}

extern "C" __global__ AICORE void launchTMULSCase1(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTMuls<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase2(__gm__ aclFloat16* out, __gm__ aclFloat16* src, float scalar)
{
    runTMuls<half, 63, 128, 63, 63, 64, 64>((__gm__ half*)out, (__gm__ half*)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTMULSCase3(__gm__ int32_t* out, __gm__ int32_t* src, int32_t scalar)
{
    runTMuls<int32_t, 31, 256, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase4(__gm__ int16_t* out, __gm__ int16_t* src, int16_t scalar)
{
    runTMuls<int16_t, 15, 192, 15, 15, 192, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase5(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTMuls<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase6(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTMuls<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase7(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTMuls<float, 1, 32, 1, 1, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase8(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTMuls<int64_t, 4, 16, 4, 4, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase9(__gm__ uint64_t* out, __gm__ uint64_t* src, uint64_t scalar)
{
    runTMuls<uint64_t, 4, 16, 4, 4, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase10(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTMulsByColTile<int64_t, 96, 32768, 32, 1024, 32, 128, 32, 1024>(out, src, scalar);
}

template <typename T, int cols>
PTO_INTERNAL void runTMulsWideInt64(__gm__ T* out, __gm__ T* src, T scalar)
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

        Event<Op::TLOAD, Op::TMULS> event0 = TLOAD(srcTile, srcGlobal);
        Event<Op::TMULS, Op::TSTORE_VEC> event1 = TMULS(dstTile, srcTile, scalar, event0);
        TSTORE(dstGlobal, dstTile, event1);
        pipe_barrier(PIPE_ALL);
    }
}

extern "C" __global__ AICORE void launchTMULSCase11(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTMulsWideInt64<int64_t, 16364>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase12(__gm__ uint64_t* out, __gm__ uint64_t* src, uint64_t scalar)
{
    runTMulsWideInt64<uint64_t, 16364>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase13(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTMulsWideInt64<int64_t, 16368>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase14(__gm__ uint64_t* out, __gm__ uint64_t* src, uint64_t scalar)
{
    runTMulsWideInt64<uint64_t, 16368>(out, src, scalar);
}

template <uint32_t caseId, typename T>
struct TMulsCaseLauncher {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream) {}
};

template <typename T>
struct TMulsCaseLauncher<1, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase1<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<2, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase2<<<1, nullptr, stream>>>((aclFloat16*)out, (aclFloat16*)src, scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<3, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase3<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src, scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<4, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase4<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src, scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<5, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase5<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<6, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase6<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<7, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase7<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<8, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase8<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<9, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase9<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src, (uint64_t)scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<10, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase10<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<11, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase11<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<12, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase12<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src, (uint64_t)scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<13, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase13<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
    }
};

template <typename T>
struct TMulsCaseLauncher<14, T> {
    static void Launch(void* out, void* src, T scalar, aclrtStream stream)
    {
        launchTMULSCase14<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src, (uint64_t)scalar);
    }
};

template <uint32_t caseId, typename T>
void launchTMULSTestCase(void* out, void* src, T scalar, aclrtStream stream)
{
    TMulsCaseLauncher<caseId, T>::Launch(out, src, scalar, stream);
}

template <uint32_t caseId>
void launchTMULSTestCase(void* out, void* src, float scalar, aclrtStream stream)
{
    launchTMULSTestCase<caseId, float>(out, src, scalar, stream);
}

template void launchTMULSTestCase<1, float>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<2, aclFloat16>(void*, void*, aclFloat16, aclrtStream);
template void launchTMULSTestCase<3, int32_t>(void*, void*, int32_t, aclrtStream);
template void launchTMULSTestCase<4, int16_t>(void*, void*, int16_t, aclrtStream);
template void launchTMULSTestCase<5, float>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<6, float>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<7, float>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<8, int64_t>(void*, void*, int64_t, aclrtStream);
template void launchTMULSTestCase<9, uint64_t>(void*, void*, uint64_t, aclrtStream);
template void launchTMULSTestCase<10, int64_t>(void*, void*, int64_t, aclrtStream);
template void launchTMULSTestCase<11, int64_t>(void*, void*, int64_t, aclrtStream);
template void launchTMULSTestCase<12, uint64_t>(void*, void*, uint64_t, aclrtStream);
template void launchTMULSTestCase<13, int64_t>(void*, void*, int64_t, aclrtStream);
template void launchTMULSTestCase<14, uint64_t>(void*, void*, uint64_t, aclrtStream);
template void launchTMULSTestCase<1>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<2>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<3>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<4>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<5>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<6>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<7>(void*, void*, float, aclrtStream);
