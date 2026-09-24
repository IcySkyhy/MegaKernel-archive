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

using namespace pto;

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T>
AICORE constexpr inline T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

template <typename T>
using CType = typename std::conditional_t<std::is_same_v<T, int8_t>, int32_t, half>;

template <typename AType, typename BType, int M, int K, int N, int validM, int validK, int validN>
AICORE inline void RunMATMUL(__gm__ AType* src0, __gm__ BType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<
        AType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        BType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN<0x20000>(aMatTile);
    TASSIGN<0x10000>(bMatTile);

    using RightTile = TileRight<BType, K, N, validK, validN>;
    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;
    RightTile bTile;
    AccTile cTile;
    TASSIGN<0x0>(bTile);
    TASSIGN<0x0>(cTile);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aMatTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
}

template <typename AType, typename BType, int M, int K, int N, int validM, int validK, int validN>
AICORE inline void RunMATMUL_NZUNALIGN(__gm__ AType* src0, __gm__ BType* src1)
{
    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN<0x20000>(aMatTile);
    TASSIGN<0x10000>(bMatTile);

    using RightTile = TileRight<BType, K, N, K, N>;
    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;
    RightTile bTile;
    AccTile cTile;
    TASSIGN<0x0>(bTile);
    TASSIGN<0x0>(cTile);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aMatTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
}

template <
    typename OutType, typename SrcTileData, int validM, int validN, Layout layoutType = Layout::ND,
    int sfractalSize = 512>
AICORE inline void RunTSTORE(__gm__ OutType* out, SrcTileData& srcTile)
{
    if constexpr (layoutType == Layout::ND) {
        using GlobalDataOut = GlobalTensor<
            OutType, pto::Shape<1, 1, 1, validM, validN>,
            pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, srcTile);
    } else if constexpr (layoutType == Layout::NZ) {
        constexpr uint16_t sGRows_ = 16;
        constexpr uint16_t sGCols_ = BLOCK_BYTE_SIZE / sizeof(OutType);
        constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(validM, sGRows_);
        constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(validN, sGCols_);
        using DynShapeDim5 = Shape<1, kGCols_, kGRows_, sGRows_, sGCols_>;
        using DynStridDim5 = pto::Stride<
            kGCols_ * kGRows_ * sGCols_ * sGRows_, kGRows_ * sGCols_ * sGRows_, sGCols_ * sGRows_, sGCols_, 1>;

        using GlobalDataOut = GlobalTensor<OutType, DynShapeDim5, DynStridDim5, layoutType>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, srcTile);
    }
}

template <Layout layoutType>
AICORE inline constexpr BLayout GetTileBLayout()
{
    if constexpr (layoutType == Layout::NZ) {
        return BLayout::ColMajor;
    } else {
        return BLayout::RowMajor;
    }
}

template <Layout layoutType>
AICORE inline constexpr SLayout GetTileSLayout()
{
    if constexpr (layoutType == Layout::NZ) {
        return SLayout::RowMajor;
    } else {
        return SLayout::NoneBox;
    }
}

template <
    typename OutType, typename AType, typename BType, int validM, int validK, int validN, int row, int col,
    bool isNZUnalign = false, Layout layoutType = Layout::ND, int sfractalSize = 512>
__global__ AICORE void RunTMOV(__gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1)
{
    constexpr int blockAlign = std::is_same_v<AType, int8_t> ? 32 : 16;
    constexpr int M = CeilAlign<int>(validM, blockAlign);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    if constexpr (!isNZUnalign) {
        RunMATMUL<AType, BType, M, K, N, validM, validK, validN>(src0, src1);
    } else {
        RunMATMUL_NZUNALIGN<AType, BType, M, K, N, validM, validK, validN>(src0, src1);
    }
    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN<0x0>(cTile);

    using DstTileData = std::conditional_t<
        isNZUnalign,
        Tile<
            TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), row, col, GetTileSLayout<layoutType>(),
            sfractalSize>,
        Tile<
            TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), validM, validN,
            GetTileSLayout<layoutType>(), sfractalSize>>;
    DstTileData dstTileData;
    TASSIGN<0x0>(dstTileData);

    TMOV(dstTileData, cTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    RunTSTORE<OutType, DstTileData, validM, validN, layoutType, sfractalSize>(out, dstTileData);
}

template <int32_t tilingKey>
void LaunchTMOVAcc2VecNZ2ND(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOV<half, half, half, 60, 127, 120, 64, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (tilingKey == 2) {
        RunTMOV<half, half, half, 110, 100, 80, 112, 80><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (tilingKey == 3) {
        RunTMOV<half, half, half, 6, 7, 8, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (tilingKey == 4) {
        RunTMOV<half, half, half, 111, 47, 96, 112, 96><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    }
}

template void LaunchTMOVAcc2VecNZ2ND<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMOVAcc2VecNZ2ND<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMOVAcc2VecNZ2ND<3>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMOVAcc2VecNZ2ND<4>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecNZ2NZ(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOV<half, half, half, 96, 80, 112, 96, 112, false, Layout::NZ, 1024><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (tilingKey == 2) {
        RunTMOV<half, half, half, 80, 112, 96, 80, 96, false, Layout::NZ, 1024><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (tilingKey == 3) {
        RunTMOV<half, half, half, 13, 16, 9, 16, 16, true, Layout::NZ, 1024><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (tilingKey == 4) {
        RunTMOV<half, half, half, 45, 112, 43, 48, 48, true, Layout::NZ, 1024><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    }
}

template void LaunchTMOVAcc2VecNZ2NZ<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMOVAcc2VecNZ2NZ<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMOVAcc2VecNZ2NZ<3>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMOVAcc2VecNZ2NZ<4>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
