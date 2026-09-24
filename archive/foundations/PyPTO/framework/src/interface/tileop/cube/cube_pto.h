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
 * \file cube_pto.h
 * \brief TileOp Interface Definition
 */

#ifndef TILEOP_TILE_OPERATOR_CUBE_PTO__H
#define TILEOP_TILE_OPERATOR_CUBE_PTO__H

// Common Operator Definitions (Shared by Atlas A3, Ascend 950PR/Ascend 950DT)
#include "impl/copy_gm_to_l1_impl.h"
#include "impl/copy_l0c_to_gm_impl.h"
#include "impl/copy_l0c_to_l1_impl.h"
#include "impl/copy_l1_to_bt_fb_impl.h"
#include "impl/copy_l1_to_l0_impl.h"
#include "impl/cube_utils.h"
#include "impl/gather_in_l1_impl.h"
#include "impl/mmad_impl.h"

// Operator Header File for Ascend 950PR/Ascend 950DT Architectures, Enabled Only When PTO_NPU_ARCH_A5 Macro is Defined.
#if defined(PTO_NPU_ARCH_A5) || defined(__LITE_NPU)
#include "impl/arch35/copy_gm_to_l1_mx_impl.h"
#include "impl/arch35/copy_l0c_to_ub_impl.h"
#include "impl/arch35/copy_l1_to_l0_mx_impl.h"
#include "impl/arch35/copy_ub_to_l1_impl.h"
#include "impl/arch35/copy_ub_to_ub_impl.h"
#include "impl/arch35/mmad_mx_impl.h"
#endif

#ifdef __LITE_NPU
template <bool initMatrixC, TransMode transMode, bool kAlignFlag, typename TileAcc, typename TileLeft,
          typename TileRight>
TILEOP void TGEMV(TileAcc& c, TileLeft& a, TileRight& b)
{
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
                  "[TGEMV ERROR]: Shape dim size should be 2");
    TMatmulImpl<initMatrixC, transMode, kAlignFlag, true>(c, a, b);
}

template <TransMode transMode, bool kAlignFlag, typename TileAcc, typename TileLeft, typename TileRight,
          typename TileBias>
TILEOP void TGEMV(TileAcc& c, TileLeft& a, TileRight& b, TileBias& bias)
{
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
                  "[TGEMV ERROR]: Shape dim size should be 2");
    TMatmulImpl<transMode, kAlignFlag, true>(c, a, b, bias);
}
#endif

// TileOp Definitions for Matrix Multiplication & Data Movement on Ascend 950PR/Ascend 950DT Architectures
#if defined(PTO_NPU_ARCH_A5) || defined(__LITE_NPU)
// Copy Scale A data from DDR to L1 for MX matmul
template <CopyInMode mode, typename Coord, typename TileData, typename GlobalData>
TILEOP void TLoadAMX(TileData& dst, GlobalData& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
                  "[TLoadAMX Error]: MXMatmul A Scale Shape Size should be 3 Dim");
    static_assert(TileData::FORMAT == Hardware::L1 && GlobalData::FORMAT == Hardware::GM,
                  "[TLoadAMX Error]: Dst format should be L1 and Src format should be GM");
    TLoadAMXImpl<mode, Coord, TileData, GlobalData>(dst, src, coord);
}

// Copy Scale B data from DDR to L1 for MX matmul
template <CopyInMode mode, typename Coord, typename TileData, typename GlobalData>
TILEOP void TLoadBMX(TileData& dst, GlobalData& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
                  "[TLoadBMX Error]: MXMatmul B Scale Shape Size should be 3 Dim");
    static_assert(TileData::FORMAT == Hardware::L1 && GlobalData::FORMAT == Hardware::GM,
                  "[TLoadBMX Error]: Dst format should be L1 and Src format should be GM");
    TLoadBMXImpl<mode, Coord, TileData, GlobalData>(dst, src, coord);
}

// Copy data from UB to UB with ND -> NZ format
template <typename DstTileData, typename SrcTileData>
TILEOP void TMoveND2NZ(DstTileData& dst, SrcTileData& src)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(DstTileData::FORMAT == Hardware::UB && SrcTileData::FORMAT == Hardware::UB);
    TMoveND2NZImpl<DstTileData, SrcTileData>(dst, src);
}

// Copy data from UB to L1 with NZ -> NZ format
template <CopyMode mode, typename Coord, typename DstTileData, typename SrcTileData>
TILEOP void TCopyUB2L1(DstTileData& dst, SrcTileData& src, const Coord& dstCoord, const Coord& srcCoord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(DstTileData::FORMAT == Hardware::L1 && SrcTileData::FORMAT == Hardware::UB);
    TCopyUB2L1Impl<mode, false, Coord, DstTileData, SrcTileData>(dst, src, dstCoord, srcCoord);
}

#ifdef __LITE_NPU
// GEMV A 来自 UB 时走 ND 数据路径（tile 声明为 RowMajor + NoneBox，命中 TExtractVecToMat ND 分支）
template <CopyMode mode, typename Coord, typename DstTileData, typename SrcTileData>
TILEOP void TCopyUB2L1ND2ND(DstTileData& dst, SrcTileData& src, const Coord& dstCoord, const Coord& srcCoord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(DstTileData::FORMAT == Hardware::L1 && SrcTileData::FORMAT == Hardware::UB);
    TCopyUB2L1Impl<mode, true, Coord, DstTileData, SrcTileData>(dst, src, dstCoord, srcCoord);
}
#endif

// Copy data from L1 to L0A_MX scale or L0B_MX scale
template <typename Coord, typename DstTileData, typename SrcTileData>
TILEOP void TExtractMX(DstTileData& dst, SrcTileData& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
                  "[TExtractMX Error]: L0A_MX scale or L0B_MX scale Shape Size should be 3 Dim");
    static_assert((DstTileData::FORMAT == Hardware::L0A_MX || DstTileData::FORMAT == Hardware::L0B_MX) &&
                  SrcTileData::FORMAT == Hardware::L1);
    TExtractMXImpl<Coord, DstTileData, SrcTileData>(dst, src, coord);
}

// Copy data from L0C to UB
template <typename config, CopyMode mode, DualDstMode dualDstMode, typename Coord, typename DstTileTensor,
          typename SrcTileTensor, typename FbTileTensor>
TILEOP void TCopyL0C2UB(DstTileTensor& dst, SrcTileTensor& src, FbTileTensor& fixbuf, const Coord& dstCoord,
                        const Coord& srcCoord, int16_t subblockId, uint64_t scaleValue = 0)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileTensor::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(DstTileTensor::FORMAT == Hardware::UB && SrcTileTensor::FORMAT == Hardware::L0C);
    TCopyL0C2UBImpl<config, mode, dualDstMode, Coord, DstTileTensor, SrcTileTensor, FbTileTensor>(
        dst, src, fixbuf, dstCoord, srcCoord, subblockId, scaleValue);
}

template <bool initMatrixC, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale>
TILEOP void MatmulMX(TileRes& c, TileLeft& a, TileLeftScale& aScale, TileRight& b, TileRightScale& bScale)
{
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileRes::Shape>::value;
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeAScale = Std::tuple_size<typename TileLeftScale::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeBScale = Std::tuple_size<typename TileRightScale::Shape>::value;
    static_assert(shapeSizeC == SHAPE_DIM2 && shapeSizeA == SHAPE_DIM2 && shapeSizeAScale == SHAPE_DIM3 &&
                      shapeSizeB == SHAPE_DIM2 && shapeSizeBScale == SHAPE_DIM3,
                  "[MatmulMX ERROR]: Tensor Shape dim size should be 2 and Scale Shape dim size should be 3");
    MatmulMXImpl<initMatrixC>(c, a, aScale, b, bScale);
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias>
TILEOP void MatmulMX(TileRes& c, TileLeft& a, TileLeftScale& aScale, TileRight& b, TileRightScale& bScale,
                     TileBias& bias)
{
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileRes::Shape>::value;
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeAScale = Std::tuple_size<typename TileLeftScale::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeBScale = Std::tuple_size<typename TileRightScale::Shape>::value;
    static_assert(shapeSizeC == SHAPE_DIM2 && shapeSizeA == SHAPE_DIM2 && shapeSizeAScale == SHAPE_DIM3 &&
                      shapeSizeB == SHAPE_DIM2 && shapeSizeBScale == SHAPE_DIM3,
                  "[MatmulMX ERROR]: Shape dim size should be 2 and Scale Shape dim size should be 3");
    MatmulMXImpl(c, a, aScale, b, bScale, bias);
}
// End of TileOp Interface Definitions for Ascend 950PR/Ascend 950DT Architecture
#endif

// Common Operator TileOp Interface Definitions

// Copy data from DDR to L1
template <CopyInMode copyMode, PaddingMode padMode, int64_t kIndex, typename Coord, typename TileData,
          typename GlobalData>
TILEOP void TLoad(TileData& dst, GlobalData& src, const Coord& dstCoord, const Coord& srcCoord, const int64_t& curH,
                  const int64_t& curW)
{
    constexpr auto shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    // Handle K=0 scenario: fill L1 with zero and return early if needed
    if (HandleZeroKScenario<copyMode, kIndex>(dst)) {
        return;
    }
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    int64_t dstOffset0 = TileOp::GetTupleElement<Coord, DIM_1ST, shapeSize, 0>(dstCoord);
    int64_t dstOffset1 = TileOp::GetTupleElement<Coord, DIM_2ND, shapeSize, 0>(dstCoord);
    int64_t srcOffset0 = TileOp::GetTupleElement<Coord, DIM_1ST, shapeSize, 0>(srcCoord);
    int64_t srcOffset1 = TileOp::GetTupleElement<Coord, DIM_2ND, shapeSize, 0>(srcCoord);
    static_assert(TileData::FORMAT == Hardware::L1 && GlobalData::FORMAT == Hardware::GM,
                  "[TLoad Error]: Dst format shoulde be L1 and Src format shoulde be GM");
    if constexpr (copyMode == CopyInMode::ND2NZ) {
        TLoadND2NZ<padMode>(dst, src, srcOffset0, srcOffset1);
    } else if constexpr (copyMode == CopyInMode::NZ2NZ) {
        TLoadNZ2NZ<padMode>(dst, src, dstOffset0, dstOffset1, srcOffset0, srcOffset1, curH, curW);
    } else if constexpr (copyMode == CopyInMode::ND2ND) {
        TLoadND2ND(dst, src, srcOffset0, srcOffset1);
    }
    return;
}

// Copy data from DDR to L1
template <CopyInMode copyMode, PaddingMode padMode, int64_t kIndex, typename Coord, typename TileTensor,
          typename GlobalTensor>
TILEOP void TReshapeLoad(TileTensor& dst, GlobalTensor& src, const Coord& srcCoord, const int64_t& gShape0,
                         const int64_t& gShape1)
{
    constexpr auto shapeSize = Std::tuple_size<typename TileTensor::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    if (HandleZeroKScenario<copyMode, kIndex>(dst)) {
        return;
    }
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    if (gShape0 == 0 || gShape1 == 0) {
        return;
    }
    int64_t srcOffset0 = TileOp::GetTupleElement<Coord, DIM_1ST, shapeSize, 0>(srcCoord);
    int64_t srcOffset1 = TileOp::GetTupleElement<Coord, DIM_2ND, shapeSize, 0>(srcCoord);
    static_assert(TileTensor::FORMAT == Hardware::L1 && GlobalTensor::FORMAT == Hardware::GM,
                  "[TReshapeLoad Error]: Dst format shoulde be L1 and Src format shoulde be GM");
    static_assert(copyMode == CopyInMode::ND2NZ, "[TReshapeLoad Error]: Reshape CopyIn L1 just support ND2NZ");
    if constexpr (copyMode == CopyInMode::ND2NZ) {
        TReshapeLoadND2NZ<padMode>(dst, src, srcOffset0, srcOffset1, gShape0, gShape1);
    }
    return;
}

// Copy data from L0C to L1 with quantization ability
template <typename config, typename Coord, typename DstTileData, typename SrcTileData, typename FpTileData>
TILEOP void TExtract(DstTileData& dst, SrcTileData& src, FpTileData& fixbuf, const Coord& l1Coord,
                     const Coord& l0cCoord, uint64_t scaleValue = 0)
{
    if (!CheckShapeValid(dst, src) || !CheckShapeValid(dst, fixbuf)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(SrcTileData::FORMAT == Hardware::L0C && DstTileData::FORMAT == Hardware::L1);
    TExtractL0C2L1Impl<config>(dst, src, fixbuf, l1Coord, l0cCoord, scaleValue);
}

// Copy data from L1 to L0A/L0B
template <bool isTrans, bool isMX, typename Coord, typename DstTileData, typename SrcTileData>
TILEOP void TExtractL1ToL0(DstTileData& dst, SrcTileData& src, const Coord& coord)
{
    if (!CheckBASEMNValid(dst)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2,
                  "[TExtractL1ToL0 Error]: Shape Size should be 2 Dim");
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(coord);
    static_assert(SrcTileData::FORMAT == Hardware::L1 &&
                  (DstTileData::FORMAT == Hardware::L0A || DstTileData::FORMAT == Hardware::L0B));
    TExtractL1ToL0Impl<isTrans, isMX>(dst, src, offset0, offset1);
}

#ifdef __LITE_NPU
// GEMV A 的 L1→L0A 走 ND 数据路径（L1 tile 声明 RowMajor，命中 TExtractToAVector ND 分支）
template <bool isTrans, bool isMX, typename Coord, typename DstTileData, typename SrcTileData>
TILEOP void TExtractL1ToL0ND2ND(DstTileData& dst, SrcTileData& src, const Coord& coord)
{
    if (!CheckBASEMNValid(dst)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2,
                  "[TExtractL1ToL0ND2ND Error]: Shape Size should be 2 Dim");
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(coord);
    static_assert(SrcTileData::FORMAT == Hardware::L1 &&
                  (DstTileData::FORMAT == Hardware::L0A || DstTileData::FORMAT == Hardware::L0B));
    TExtractL1ToL0Impl<isTrans, isMX, true>(dst, src, offset0, offset1);
}
#endif

// Copy data from L1 to BT
template <bool isTrans, typename Coord, typename DstTileData, typename SrcTileData>
TILEOP void TExtractL1ToBT(DstTileData& dst, SrcTileData& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2,
                  "[TExtractL1ToBT Error]: Shape Size should be 2 Dim");
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(coord);
    static_assert(SrcTileData::FORMAT == Hardware::L1 && DstTileData::FORMAT == Hardware::BIAS);
    TExtractL1ToBTOrFBImpl<isTrans>(dst, src);
}

// Copy data from L1 to FB
template <bool isTrans, typename Coord, typename DstTileData, typename SrcTileData>
TILEOP void TExtractL1ToFB(DstTileData& dst, SrcTileData& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2,
                  "[TExtractL1ToFB Error]: Shape Size should be 2 Dim");
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(coord);
    static_assert(SrcTileData::FORMAT == Hardware::L1 && DstTileData::FORMAT == Hardware::FIXBUF);
    TExtractL1ToBTOrFBImpl<isTrans>(dst, src);
}

template <bool initMatrixC, TransMode transMode, bool kAlignFlag, typename TileAcc, typename TileLeft,
          typename TileRight>
TILEOP void TMatmul(TileAcc& c, TileLeft& a, TileRight& b)
{
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
                  "[Matmul ERROR]: Shape dim size should be 2");
    TMatmulImpl<initMatrixC, transMode, kAlignFlag>(c, a, b);
}

template <TransMode transMode, bool kAlignFlag, typename TileAcc, typename TileLeft, typename TileRight,
          typename TileBias>
TILEOP void TMatmul(TileAcc& c, TileLeft& a, TileRight& b, TileBias& bias)
{
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
                  "[Matmul ERROR]: Shape dim size should be 2");
    TMatmulImpl<transMode, kAlignFlag>(c, a, b, bias);
}

// Copy data from L0C to DDR with quantization ability
template <typename config, typename Coord, typename GlobalData, typename TileData, typename FpTileData>
TILEOP void TStore(GlobalData& dst, TileData& src, FpTileData& fixbuf, const Coord& coord, const int64_t& curH,
                   const int64_t& curW, uint64_t scaleValue = 0)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename GlobalData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(coord);
    if constexpr (TileData::FORMAT == Hardware::L0C && GlobalData::FORMAT == Hardware::GM) {
        if constexpr (config::kMode == CopyOutMode::NZ2ND) {
            TStoreNZ2ND<config>(dst, src, fixbuf, offset0, offset1, scaleValue);
        } else {
            TStoreNZ2NZ<config>(dst, src, fixbuf, offset0, offset1, curH, curW, scaleValue);
        }
    }
}

// L1 spill(Only used in deepseek model)
// When L1 space is insufficient, spill to GM. (Supported on A2/A3 only.)
template <typename config, typename Coord, typename GlobalData, typename TileData>
TILEOP void TStore(GlobalData& dst, TileData& src, const Coord& coord)
{
    TStoreL1SpillImpl<config>(dst, src, coord);
}

template <int64_t blockSize, typename TileData, typename GlobalData, typename BlockT, typename OffsetT,
          typename SrcCoord, typename OffsetCoord, typename BlockCoord>
TILEOP void TGatherInL1(TileData dst, GlobalData src, BlockT block, OffsetT offset, SrcCoord srcCoord,
                        OffsetCoord offsetCoord, BlockCoord blockCoord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    TGatherInL1Impl<blockSize>(dst, src, block, offset, srcCoord, offsetCoord, blockCoord);
}

// Copy data from L0C to DDR with quantization ability
template <typename config, typename Coord, typename GlobalTensor, typename TileTensor, typename FpTileTensor>
TILEOP void TReshapeStore(GlobalTensor& dst, TileTensor& src, FpTileTensor& fixbuf, const Coord& coord,
                          const int64_t& gShape0, const int64_t& gShape1, uint64_t scaleValue = 0)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename GlobalTensor::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(config::kMode == CopyOutMode::NZ2ND, "OutPut format only support NZ2ND when using ReshapeCopyout.");
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(coord);
    if constexpr (TileTensor::FORMAT == Hardware::L0C && GlobalTensor::FORMAT == Hardware::GM) {
        if constexpr (config::kMode == CopyOutMode::NZ2ND) {
            TReshapeStoreNZ2ND<config>(dst, src, fixbuf, offset0, offset1, gShape0, gShape1, scaleValue);
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_CUBE_PTO__H
