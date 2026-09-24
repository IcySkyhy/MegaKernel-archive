/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <pto/npu/a5/TInsert.hpp>

using namespace pto;

AICORE inline void ReadbackCbufToUbuf(
    __ubuf__ void* dstUb, __cbuf__ void* srcCbuf, uint16_t burstNum, uint16_t burstLen, uint16_t srcGap, uint8_t syncId,
    uint8_t eventIdNum)
{
    wait_intra_block(PIPE_MTE1, syncId);
    wait_intra_block(PIPE_MTE1, syncId + eventIdNum);
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(dstUb, srcCbuf, 0, burstNum, burstLen, srcGap, 0);
    copy_cbuf_to_ubuf(dstUb, srcCbuf, 1, burstNum, burstLen, srcGap, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncId);
    set_intra_block(PIPE_MTE1, syncId + eventIdNum);
}

template <typename GlobalData, typename VecTile>
AICORE inline void WaitAndStore(GlobalData& dstGlobal, VecTile& dstTile, uint8_t syncId)
{
    wait_intra_block(PIPE_MTE3, syncId);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertZN(__gm__ T* out, __gm__ T* src)
{
    using FlatShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using FlatStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, FlatShapeDim5, FlatStridDim5>;
    using OutGlobalData = GlobalTensor<T, FlatShapeDim5, FlatStridDim5>;

    using SrcNdTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using SrcZnTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols, SLayout::ColMajor>;
    using DstVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using MatTile = Tile<TileType::Mat, T, Rows, Cols, BLayout::RowMajor, -1, -1, SLayout::ColMajor>;

    SrcNdTile srcNdTile(Rows, Cols);
    SrcZnTile srcZnTile;
    DstVecTile dstTile(Rows, Cols);
    MatTile matTile(Rows, Cols);

    TASSIGN(srcNdTile, 0x0);
    TASSIGN(srcZnTile, 0x0);
    TASSIGN(dstTile, 0x10000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint16_t burstNum = 1;
    constexpr uint16_t burstLen = (Rows * Cols * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcNdTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, srcZnTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertZNFp4(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t byteRows = Rows / 2;

    using FlatShapeDim5 = pto::Shape<1, 1, 1, byteRows, Cols>;
    using FlatStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<uint8_t, FlatShapeDim5, FlatStridDim5>;
    using OutGlobalData = GlobalTensor<uint8_t, FlatShapeDim5, FlatStridDim5>;

    using SrcNdTile = Tile<TileType::Vec, uint8_t, byteRows, Cols, BLayout::RowMajor, -1, -1>;
    using SrcZnTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols, SLayout::ColMajor>;
    using DstVecTile = Tile<TileType::Vec, uint8_t, byteRows, Cols, BLayout::RowMajor, -1, -1>;
    using MatTile = Tile<TileType::Mat, T, Rows, Cols, BLayout::RowMajor, -1, -1, SLayout::ColMajor>;

    SrcNdTile srcNdTile(byteRows, Cols);
    SrcZnTile srcZnTile;
    DstVecTile dstTile(byteRows, Cols);
    MatTile matTile(Rows, Cols);

    TASSIGN(srcNdTile, 0x0);
    TASSIGN(srcZnTile, 0x0);
    TASSIGN(dstTile, 0x10000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(reinterpret_cast<__gm__ uint8_t*>(src));
    OutGlobalData dstGlobal(reinterpret_cast<__gm__ uint8_t*>(out));

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint16_t burstNum = 1;
    constexpr uint16_t burstLen = (byteRows * Cols) / BLOCK_BYTE_SIZE;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ uint8_t* dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcNdTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, srcZnTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <
    typename T, uint32_t ValidRow, uint32_t ValidCol, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols,
    uint32_t IdxRow, uint32_t IdxCol>
AICORE void runTInsertZNOffset(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t bgElems = DstRows * DstCols;

    using BgShapeDim5 = pto::Shape<1, 1, 1, DstRows, DstCols>;
    using BgStridDim5 = pto::Stride<1, 1, 1, DstCols, 1>;
    using BgGlobalData = GlobalTensor<T, BgShapeDim5, BgStridDim5>;
    using OutGlobalData = GlobalTensor<T, BgShapeDim5, BgStridDim5>;

    using SrcShapeDim5 = pto::Shape<1, 1, 1, ValidRow, SrcCols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, SrcCols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    using BgNdTile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::RowMajor, -1, -1>;
    using SrcNdTile = Tile<TileType::Vec, T, ValidRow, SrcCols, BLayout::RowMajor, -1, -1>;
    using SrcZnTile =
        Tile<TileType::Vec, T, ValidRow, SrcCols, BLayout::RowMajor, ValidRow, ValidCol, SLayout::ColMajor>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::RowMajor, -1, -1>;
    using MatTile = Tile<TileType::Mat, T, DstRows, DstCols, BLayout::RowMajor, -1, -1, SLayout::ColMajor>;

    BgNdTile bgNdTile(DstRows, DstCols);
    SrcNdTile srcNdTile(ValidRow, SrcCols);
    SrcZnTile srcZnTile;
    DstVecTile dstTile(DstRows, DstCols);
    MatTile matTile(DstRows, DstCols);

    TASSIGN(bgNdTile, 0x0);
    TASSIGN(srcNdTile, 0x10000);
    TASSIGN(srcZnTile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    BgGlobalData bgGlobal(src);
    SrcGlobalData srcGlobal(src + bgElems);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint16_t fullBurstLen = (DstRows * DstCols * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* bgUbAddr = bgNdTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(bgNdTile, bgGlobal);
    TLOAD(srcNdTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    copy_ubuf_to_cbuf((__cbuf__ void*)matAddr, (__ubuf__ void*)bgUbAddr, 0, 1, fullBurstLen, 0, 0);
    TINSERT(matTile, srcZnTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf(
        (__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, static_cast<uint16_t>(1), fullBurstLen, 0, syncId,
        eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertZNKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertZN<T, Rows, Cols>(reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <
    typename T, uint32_t ValidRow, uint32_t ValidCol, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols,
    uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void launchTInsertZNOffsetKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertZNOffset<T, ValidRow, ValidCol, SrcCols, DstRows, DstCols, IdxRow, IdxCol>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertZNFp4Kernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertZNFp4<T, Rows, Cols>(reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <int32_t testKey>
void launchTInsertZNFp4(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertZNFp4Kernel<float4_e2m1x2_t, 64, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertZNFp4Kernel<float4_e1m2x2_t, 64, 32><<<1, nullptr, stream>>>(out, src);
    }
}

template <int32_t testKey>
void launchTInsertZN(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertZNKernel<half, 16, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertZNKernel<half, 16, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertZNKernel<float, 8, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertZNKernel<float, 16, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTInsertZNKernel<int32_t, 8, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTInsertZNKernel<int8_t, 32, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 7) {
        launchTInsertZNKernel<half, 32, 64><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 8) {
        launchTInsertZNKernel<bfloat16_t, 16, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 9) {
        launchTInsertZNKernel<float8_e4m3_t, 32, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 10) {
        launchTInsertZNKernel<float8_e5m2_t, 32, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 11) {
        launchTInsertZNKernel<hifloat8_t, 32, 64><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 12) {
        launchTInsertZNKernel<float8_e8m0_t, 32, 32><<<1, nullptr, stream>>>(out, src);
    }
}

template <int32_t testKey>
void launchTInsertZNOffset(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertZNOffsetKernel<half, 16, 16, 16, 32, 32, 0, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertZNOffsetKernel<half, 16, 16, 16, 32, 32, 16, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertZNOffsetKernel<float, 8, 16, 16, 16, 32, 8, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertZNOffsetKernel<half, 16, 15, 16, 16, 32, 0, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTInsertZNOffsetKernel<int8_t, 32, 32, 32, 64, 64, 32, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTInsertZNOffsetKernel<half, 16, 32, 32, 32, 64, 0, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 7) {
        launchTInsertZNOffsetKernel<half, 16, 16, 16, 32, 32, 16, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 8) {
        launchTInsertZNOffsetKernel<float, 8, 15, 16, 16, 32, 8, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 9) {
        launchTInsertZNOffsetKernel<half, 32, 32, 32, 64, 64, 0, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 10) {
        launchTInsertZNOffsetKernel<half, 16, 16, 16, 16, 48, 0, 8><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 11) {
        launchTInsertZNOffsetKernel<float, 8, 16, 16, 8, 48, 0, 24><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 12) {
        launchTInsertZNOffsetKernel<half, 16, 16, 16, 32, 48, 16, 8><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 13) {
        launchTInsertZNOffsetKernel<float, 8, 13, 16, 8, 48, 0, 24><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTInsertZN<1>(uint64_t*, uint64_t*, void*);
template void launchTInsertZN<2>(uint64_t*, uint64_t*, void*);
template void launchTInsertZN<3>(uint64_t*, uint64_t*, void*);
template void launchTInsertZN<4>(uint64_t*, uint64_t*, void*);
template void launchTInsertZN<5>(uint64_t*, uint64_t*, void*);
template void launchTInsertZN<6>(uint64_t*, uint64_t*, void*);
template void launchTInsertZN<7>(uint64_t*, uint64_t*, void*);
template void launchTInsertZN<8>(uint64_t*, uint64_t*, void*);
template void launchTInsertZN<9>(uint64_t*, uint64_t*, void*);
template void launchTInsertZN<10>(uint64_t*, uint64_t*, void*);
template void launchTInsertZN<11>(uint64_t*, uint64_t*, void*);
template void launchTInsertZN<12>(uint64_t*, uint64_t*, void*);

template void launchTInsertZNFp4<1>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNFp4<2>(uint64_t*, uint64_t*, void*);

template void launchTInsertZNOffset<1>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<2>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<3>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<4>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<5>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<6>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<7>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<8>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<9>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<10>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<11>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<12>(uint64_t*, uint64_t*, void*);
template void launchTInsertZNOffset<13>(uint64_t*, uint64_t*, void*);
