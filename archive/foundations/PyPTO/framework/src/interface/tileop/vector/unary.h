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
 * \file vec_unary.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_UNARY__H
#define TILEOP_TILE_OPERATOR_VEC_UNARY__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#include <cmath>
#include <limits>

TILEOP void SyncV()
{
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
}

template <typename DType>
TILEOP constexpr bool IsIntegralType()
{
    return std::is_same_v<DType, int32_t> || std::is_same_v<DType, uint32_t> || std::is_same_v<DType, int8_t> ||
           std::is_same_v<DType, uint8_t> || std::is_same_v<DType, int16_t> || std::is_same_v<DType, uint16_t> ||
           std::is_same_v<DType, int64_t> || std::is_same_v<DType, uint64_t>;
}

template <UnaryOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1>
TILEOP void UnaryComputeImpl(T0 dst, T1 src)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    if constexpr (op == UnaryOp::EXP) {
        PTO_WITH_LAST_USE(pto::TEXP<PrecisionType>(dst, src), n1, n2);
        return;
    }
    if constexpr (op == UnaryOp::RSQRT) {
        PTO_WITH_LAST_USE(pto::TRSQRT(dst, src), n1, n2);
        return;
    }
    if constexpr (op == UnaryOp::SQRT) {
        PTO_WITH_LAST_USE(pto::TSQRT<PrecisionType>(dst, src), n1, n2);
        return;
    }
    if constexpr (op == UnaryOp::BRCB) {
        PTO_WITH_LAST_USE(pto::TROWEXPAND(dst, src), n1, n2);
        return;
    }
    if constexpr (op == UnaryOp::ABS) {
        PTO_WITH_LAST_USE(pto::TABS(dst, src), n1, n2);
        return;
    }
    if constexpr (op == UnaryOp::RECIPROCAL) {
        PTO_WITH_LAST_USE(pto::TRECIP<PrecisionType>(dst, src), n1, n2);
        return;
    }
    if constexpr (op == UnaryOp::BITWISENOT) {
        PTO_WITH_LAST_USE(pto::TNOT(dst, src), n1, n2);
        return;
    }
    if constexpr (op == UnaryOp::RELU) {
        pto::TMAXS(dst, src, 0.0f);
        return;
    }
    if constexpr (op == UnaryOp::LN) {
        pto::TLOG<PrecisionType>(dst, src);
        return;
    }
}

template <typename T, typename HalfTileDefineSrc, typename TileDefineDst, typename B16TileDefineSrc>
TILEOP void IsFiniteCalcImpl(TileDefineDst dst, B16TileDefineSrc src, B16TileDefineSrc bufferB16,
                             HalfTileDefineSrc bufferFP16)
{
    int16_t mask = 0;
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        mask = 0x7F80;
    } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float>) {
        mask = 0x7C00;
    }
    pto::TANDS(bufferB16, src, mask);
    SyncV();
    pto::TSUBS(bufferB16, bufferB16, mask);
    SyncV();
    pto::TMAXS(bufferB16, bufferB16, (int16_t)-1);
    SyncV();
    pto::TMULS(bufferB16, bufferB16, (int16_t)-1);
    SyncV();
    pto::TCVT(dst, bufferFP16, pto::RoundMode::CAST_CEIL);
    SyncV();
}

template <typename T, typename HalfTileDefineSrc, bool CombineAxis, typename TileDefineDst, typename B16TileDefineSrc>
TILEOP void IsFiniteComputeImpl(TileDefineDst dst, B16TileDefineSrc src, HalfTileDefineSrc buffer)
{
    if constexpr (!CombineAxis) {
        HalfTileDefineSrc bufferFP16(src.GetValidRow(), src.GetValidCol());
        pto::TASSIGN(bufferFP16, reinterpret_cast<std::uintptr_t>(buffer.data()));
        B16TileDefineSrc bufferB16(src.GetValidRow(), src.GetValidCol());
        pto::TASSIGN(bufferB16, reinterpret_cast<std::uintptr_t>(buffer.data()));
        IsFiniteCalcImpl<T>(dst, src, bufferB16, bufferFP16);
    } else {
        HalfTileDefineSrc bufferFP16;
        pto::TASSIGN(bufferFP16, reinterpret_cast<std::uintptr_t>(buffer.data()));
        B16TileDefineSrc bufferB16;
        pto::TASSIGN(bufferB16, reinterpret_cast<std::uintptr_t>(buffer.data()));
        IsFiniteCalcImpl<T>(dst, src, bufferB16, bufferFP16);
    }
}

template <UnaryOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1>
TILEOP void UnaryCompute(T0 dst, T1 src)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }

    using SrcExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
    if constexpr (TileOp::IsConstContinous<T0, T1>()) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto srcTile = PtoTile<T0, pto::BLayout::RowMajor, true, SrcExecDtype>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        UnaryComputeImpl<op, PrecisionType, LastUse>(dstTile, srcTile);
        return;
    }

    auto dstTile = PtoTile<T0>(dst);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                UnaryComputeImpl<op, PrecisionType, LastUse>(dstTile.Data(), srcExecTile);
            }
        }
    }
}

#define OP_TILE_OP_EXP TExp
template <typename LastUse, typename T0, typename T1>
TILEOP void BrcbCompute(T0 dst, T1 src)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    const auto srcLayout = src.GetLayout();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();

    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor>;
    using SrcTileDefine = typename std::conditional<
        (srcTileW == 1), pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::ColMajor>,
        pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileW, srcTileH, pto::BLayout::ColMajor>>::type;

    SrcTileDefine srcTile;
    DstTileDefine dstTile;
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto dstTileOffsets = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcTileOffsets = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstTileOffsets * sizeof(typename T0::Type)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcTileOffsets * sizeof(typename T1::Type)));
                UnaryComputeImpl<UnaryOp::BRCB, 0, LastUse>(dstTile, srcTile);
            }
        }
    }
}

#define OP_TILE_OP_EXP TExp
template <auto PrecisionType = pto::ExpAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename T0,
          typename T1>
TILEOP void TExp(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::EXP, PrecisionType, LastUse>(dst, src);
}

#define OP_TILE_OP_RSQRT TRsqrt
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TRsqrt(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::RSQRT, 0, LastUse>(dst, src);
}

#define OP_TILE_OP_SQRT TSqrt
template <auto PrecisionType = pto::SqrtAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename T0,
          typename T1>
TILEOP void TSqrt(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::SQRT, PrecisionType, LastUse>(dst, src);
}

template <typename DstTileTensor, typename SrcTileTensor, typename BufferTileTensor>
TILEOP void TIsFiniteCombineAxis(DstTileTensor dst, SrcTileTensor src, BufferTileTensor buffer)
{
    using DstType = std::conditional_t<std::is_same_v<typename DstTileTensor::Type, bool>, uint8_t,
                                       typename DstTileTensor::Type>;
    using SrcType = typename SrcTileTensor::Type;

    constexpr size_t tileSrcH = GetMergedAxisIfNeed<SrcTileTensor, true>();
    constexpr size_t tileSrcW = TileOp::GetTensorTileShapeDim<SrcTileTensor, DIM_5TH, MAX_DIMS>();
    constexpr size_t tileDstH = GetMergedAxisIfNeed<DstTileTensor, true>();
    constexpr size_t tileDstW = TileOp::GetTensorTileShapeDim<DstTileTensor, DIM_5TH, MAX_DIMS>();

    constexpr int validH = GetValidHeight<SrcTileTensor, true>();
    constexpr int validW = GetValidWidth<SrcTileTensor>();

    if constexpr (IsIntegralType<SrcType>()) {
        using TileDefineDst = pto::Tile<pto::TileType::Vec, int16_t, tileDstH, (tileDstW + 1) / 2,
                                        pto::BLayout::RowMajor, validH, (validW + 1) / 2>;
        TileDefineDst dstTile;
        pto::TASSIGN(dstTile, dst.GetAddr());
        int16_t mask = 0x0101;
        TANDS(dstTile, dstTile, 0);
        SyncV();
        TORS(dstTile, dstTile, mask);
        SyncV();
        return;
    } else {
        using TileDefineDst = pto::Tile<pto::TileType::Vec, DstType, tileDstH, tileDstW, pto::BLayout::RowMajor, validH,
                                        validW>;
        using HalfTileDefineSrc = pto::Tile<pto::TileType::Vec, half, tileSrcH,
                                            tileSrcW * sizeof(SrcType) / sizeof(half), pto::BLayout::RowMajor, validH,
                                            validW>;
        using B16TileDefineSrc = pto::Tile<pto::TileType::Vec, int16_t, tileSrcH,
                                           tileSrcW * sizeof(SrcType) / sizeof(int16_t), pto::BLayout::RowMajor, validH,
                                           validW>;

        HalfTileDefineSrc bufferTile;
        TileDefineDst dstTile;
        B16TileDefineSrc srcTile;
        pto::TASSIGN(bufferTile, buffer.GetAddr());
        pto::TASSIGN(dstTile, dst.GetAddr());
        pto::TASSIGN(srcTile, src.GetAddr());

        if constexpr (std::is_same_v<SrcType, float>) {
            using FP32TileDefineSrc = pto::Tile<pto::TileType::Vec, float, tileSrcH, tileSrcW, pto::BLayout::RowMajor,
                                                validH, validW>;
            FP32TileDefineSrc srcFP32;
            HalfTileDefineSrc srcFP16;
            pto::TASSIGN(srcFP32, src.GetAddr());
            pto::TASSIGN(srcFP16, src.GetAddr());
            pto::TCVT(srcFP16, srcFP32, pto::RoundMode::CAST_NONE);
            SyncV();
        }

        IsFiniteComputeImpl<SrcType, HalfTileDefineSrc, true>(dstTile, srcTile, bufferTile);
    }
}

template <typename DstTileTensor, typename SrcTileTensor>
TILEOP void TIsFinite4Integral(DstTileTensor dst, SrcTileTensor src)
{
    using DstType = std::conditional_t<std::is_same_v<typename DstTileTensor::Type, bool>, uint8_t,
                                       typename DstTileTensor::Type>;
    using SrcType = typename SrcTileTensor::Type;
    constexpr size_t tileSrcH = GetMergedAxisIfNeed<SrcTileTensor, false>();
    constexpr size_t tileSrcW = TileOp::GetTensorTileShapeDim<SrcTileTensor, DIM_5TH, MAX_DIMS>();
    constexpr size_t tileDstH = GetMergedAxisIfNeed<DstTileTensor, false>();
    constexpr size_t tileDstW = TileOp::GetTensorTileShapeDim<DstTileTensor, DIM_5TH, MAX_DIMS>();

    static_assert(IsElementwiseDstLayoutCoveredByOperand<DstTileTensor, SrcTileTensor>());
    int validH = dst.GetLayout().template GetShapeDim<DIM_4TH, MAX_DIMS>();
    int validW = dst.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>();

    using TileDefineDst = pto::Tile<pto::TileType::Vec, int16_t, tileDstH, tileDstW / 2, pto::BLayout::RowMajor, -1,
                                    -1>;
    TileDefineDst dstTile(validH, (validW + 1) / 2);
    pto::TASSIGN(dstTile, dst.GetAddr());
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    int16_t mask = 0x0101;

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, dst.GetAddr() + GenTileOffset(dst, tileOffsets) * sizeof(DstType));
                TANDS(dstTile, dstTile, 0);
                SyncV();
                TORS(dstTile, dstTile, mask);
                SyncV();
            }
        }
    }
}

template <typename DstTileTensor, typename SrcTileTensor, typename BufferTileTensor>
TILEOP void TIsFinite4Floats(DstTileTensor dst, SrcTileTensor src, BufferTileTensor buffer)
{
    using SrcType = typename SrcTileTensor::Type;
    using DstType = std::conditional_t<std::is_same_v<typename DstTileTensor::Type, bool>, uint8_t,
                                       typename DstTileTensor::Type>;
    static_assert(IsElementwiseDstLayoutCoveredByOperand<DstTileTensor, SrcTileTensor>());
    constexpr size_t tileSrcH = ElementwiseOperandExecConfig<DstTileTensor, SrcTileTensor>::tileH;
    constexpr size_t tileSrcW = TileOp::GetTensorTileShapeDim<SrcTileTensor, DIM_5TH, MAX_DIMS>();
    constexpr size_t tileDstH = GetMergedAxisIfNeed<DstTileTensor, false>();
    constexpr size_t tileDstW = TileOp::GetTensorTileShapeDim<DstTileTensor, DIM_5TH, MAX_DIMS>();

    int validH = dst.GetLayout().template GetShapeDim<DIM_4TH, MAX_DIMS>();
    int validW = dst.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>();

    using TileDefineDst = pto::Tile<pto::TileType::Vec, DstType, tileDstH, tileDstW, pto::BLayout::RowMajor, -1, -1>;
    using HalfTileDefineSrc = pto::Tile<pto::TileType::Vec, half, tileSrcH, tileSrcW * sizeof(SrcType) / sizeof(half),
                                        pto::BLayout::RowMajor, -1, -1>;
    using B16TileDefineSrc = pto::Tile<pto::TileType::Vec, int16_t, tileSrcH,
                                       tileSrcW * sizeof(SrcType) / sizeof(int16_t), pto::BLayout::RowMajor, -1, -1>;

    HalfTileDefineSrc bufferTile(validH, validW);
    pto::TASSIGN(bufferTile, buffer.GetAddr());

    TileDefineDst dstTile(validH, validW);
    B16TileDefineSrc srcTile(validH, validW);

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, dst.GetAddr() + GenTileOffset(dst, tileOffsets) * sizeof(DstType));
                pto::TASSIGN(srcTile, src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(int16_t));
                if constexpr (std::is_same_v<SrcType, float>) {
                    using FP32TileDefineSrc = pto::Tile<pto::TileType::Vec, float, tileSrcH, tileSrcW,
                                                        pto::BLayout::RowMajor, -1, -1>;
                    FP32TileDefineSrc srcFP32(validH, validW);
                    HalfTileDefineSrc srcFP16(validH, validW);
                    pto::TASSIGN(srcFP32, src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(float));
                    pto::TASSIGN(srcFP16, src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(half));
                    pto::TCVT(srcFP16, srcFP32, pto::RoundMode::CAST_NONE);
                    SyncV();
                }
                IsFiniteComputeImpl<SrcType, HalfTileDefineSrc, false>(dstTile, srcTile, bufferTile);
            }
        }
    }
}

#define OP_TILE_OP_ISFINITE TIsFinite
template <typename DstTileTensor, typename SrcTileTensor, typename BufferTileTensor>
TILEOP void TIsFinite(DstTileTensor dst, SrcTileTensor src, BufferTileTensor buffer)
{
    constexpr bool sameExecSize = GetAllAxisTileProduct<DstTileTensor>() == GetAllAxisTileProduct<SrcTileTensor>();
    if constexpr (TileOp::IsConstContinous<DstTileTensor, SrcTileTensor>() && sameExecSize) {
        TIsFiniteCombineAxis(dst, src, buffer);
        return;
    }

    using SrcType = typename SrcTileTensor::Type;
    if constexpr (IsIntegralType<SrcType>()) {
        TIsFinite4Integral(dst, src);
    } else {
        TIsFinite4Floats(dst, src, buffer);
    }
}

#define OP_TILE_OP_BRCB Tbrcb
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void Tbrcb(T0 dst, T1 src)
{
    BrcbCompute<LastUse>(dst, src);
}

#define OP_TILE_OP_ABS TAbs
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TAbs(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::ABS, 0, LastUse>(dst, src);
}

#define OP_TILE_OP_BITWISENOT TBitwiseNot
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TBitwiseNot(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::BITWISENOT, 0, LastUse>(dst, src);
}

#define OP_TILE_OP_LOG TLog
template <auto PrecisionType = pto::LogAlgorithm::DEFAULT, typename T0, typename T1>
TILEOP void TLog(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::LN, PrecisionType, LastUse2Dim<0, 0>>(dst, src);
}

template <typename Ttemp, typename T0, typename T1>
TILEOP void CeilComputeImpl(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::DType>) {
        pto::TMOV(dst, src);
    } else {
        pto::TCVT(dst, src, pto::RoundMode::CAST_CEIL);
    }
}
#define OP_TILE_OP_CEIL TCEIL
template <typename T0, typename T1>
TILEOP void TCeil(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::Type>) {
        if ((uint64_t)dst.GetAddr() == (uint64_t)src.GetAddr()) {
            return;
        }
    }

    using SrcExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
    if constexpr (TileOp::IsConstContinous<T0, T1>()) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto srcTile = PtoTile<T0, pto::BLayout::RowMajor, true, SrcExecDtype>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        CeilComputeImpl<float>(dstTile, srcTile);
        return;
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                CeilComputeImpl<float>(dstTile.Data(), srcExecTile);
            }
        }
    }
}

template <typename Ttemp, typename T0, typename T1>
TILEOP void FloorComputeImpl(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::DType>) {
        pto::TMOV(dst, src);
    } else {
        pto::TCVT(dst, src, pto::RoundMode::CAST_FLOOR);
    }
}
#define OP_TILE_OP_FLOOR TFLOOR
template <typename T0, typename T1>
TILEOP void TFloor(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::Type>) {
        if ((uint64_t)dst.GetAddr() == (uint64_t)src.GetAddr()) {
            return;
        }
    }

    using SrcExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
    if constexpr (TileOp::IsConstContinous<T0, T1>()) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto srcTile = PtoTile<T0, pto::BLayout::RowMajor, true, SrcExecDtype>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        FloorComputeImpl<float>(dstTile, srcTile);
        return;
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                FloorComputeImpl<float>(dstTile.Data(), srcExecTile);
            }
        }
    }
}

template <typename Ttemp, typename T0, typename T1>
TILEOP void TruncComputeImpl(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::DType>) {
        pto::TMOV(dst, src);
    } else {
        pto::TCVT(dst, src, pto::RoundMode::CAST_TRUNC);
    }
}
#define OP_TILE_OP_TRUNC TTRUNC
template <typename T0, typename T1>
TILEOP void TTrunc(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::Type>) {
        if ((uint64_t)dst.GetAddr() == (uint64_t)src.GetAddr()) {
            return;
        }
    }

    using SrcExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
    if constexpr (TileOp::IsConstContinous<T0, T1>()) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto srcTile = PtoTile<T0, pto::BLayout::RowMajor, true, SrcExecDtype>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        TruncComputeImpl<float>(dstTile, srcTile);
        return;
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                TruncComputeImpl<float>(dstTile.Data(), srcExecTile);
            }
        }
    }
}

#define OP_TILE_OP_EXP2 TExp2
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TExp2(T0 dst, T1 tmp, T2 tmp2, T3 src)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto tmpTile = PtoTile<T1>(tmp);
    auto tmpTile2 = PtoTile<T2>(tmp2);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                tmpTile2.Assign(tmp2, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

                if constexpr (std::is_same_v<typename T3::Type, float>) {
                    pto::TEXPANDS(tmpTile2.Data(), 2.0f);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TLOG(tmpTile2.Data(), tmpTile2.Data());
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TMUL(tmpTile2.Data(), srcExecTile, tmpTile2.Data());
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TEXP(dstTile.Data(), tmpTile2.Data());
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                } else {
                    pto::TCVT(tmpTile.Data(), srcExecTile, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TEXPANDS(tmpTile2.Data(), 2.0f);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TLOG(tmpTile2.Data(), tmpTile2.Data());
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TMUL(tmpTile.Data(), tmpTile.Data(), tmpTile2.Data());
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    if constexpr (std::is_same_v<typename T3::Type, half> ||
                                  std::is_same_v<typename T3::Type, bfloat16_t>) {
                        pto::TEXP(tmpTile2.Data(), tmpTile.Data());
#ifdef __DAV_V220
                        pipe_barrier(PIPE_V);
#endif
                        pto::TCVT(dstTile.Data(), tmpTile2.Data(), pto::RoundMode::CAST_RINT);
                    } else {
                        pto::TEXP(dstTile.Data(), tmpTile.Data());
                    }
                }
            }
        }
    }
}

#define OP_TILE_OP_ROUND TRound
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TRound(T0 dst, T1 tmp, T2 src, Scalar powDecimals)
{
    if constexpr (std::is_integral_v<typename T2::Type>) {
        if ((uint64_t)dst.GetAddr() == (uint64_t)src.GetAddr() && powDecimals >= 1.0f) {
            return;
        }
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto tmpTile = PtoTile<T1>(tmp);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

                if constexpr (std::is_integral_v<typename T2::Type>) {
                    if (powDecimals >= 1.0f) {
                        pto::TMOV(dstTile.Data(), srcExecTile);
                        continue;
                    }
                }

                if constexpr (std::is_same_v<typename T2::Type, float>) {
                    pto::TMULS(srcExecTile, srcExecTile, powDecimals);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TCVT(srcExecTile, srcExecTile, pto::RoundMode::CAST_RINT);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TDIVS(dstTile.Data(), srcExecTile, powDecimals);
                } else {
                    pto::TCVT(tmpTile.Data(), srcExecTile, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TMULS(tmpTile.Data(), tmpTile.Data(), powDecimals);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TCVT(tmpTile.Data(), tmpTile.Data(), pto::RoundMode::CAST_RINT);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TMULS(tmpTile.Data(), tmpTile.Data(), 1.0f / powDecimals);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TCVT(dstTile.Data(), tmpTile.Data(), pto::RoundMode::CAST_RINT);
                }
            }
        }
    }
}

#define OP_TILE_OP_EXPM1 TExpm1
template <typename T0, typename T1, typename T2>
TILEOP void TExpm1(T0 dst, T1 tmp, T2 src)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto tmpTile = PtoTile<T1>(tmp);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

                if constexpr (std::is_same_v<typename T2::Type, float>) {
                    pto::TEXP(dstTile.Data(), srcExecTile);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TADDS(dstTile.Data(), dstTile.Data(), -1.0f);
                } else {
                    pto::TCVT(tmpTile.Data(), srcExecTile, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TEXP(tmpTile.Data(), tmpTile.Data());
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    if constexpr (std::is_same_v<typename T2::Type, half> ||
                                  std::is_same_v<typename T2::Type, bfloat16_t>) {
                        pto::TADDS(tmpTile.Data(), tmpTile.Data(), -1.0f);
#ifdef __DAV_V220
                        pipe_barrier(PIPE_V);
#endif
                        pto::TCVT(dstTile.Data(), tmpTile.Data(), pto::RoundMode::CAST_RINT);
                    } else {
                        pto::TADDS(dstTile.Data(), tmpTile.Data(), -1.0f);
                    }
                }
            }
        }
    }
}

#define OP_TILE_OP_RECIPROCAL TReciprocal
template <auto PrecisionType = pto::RecipAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename T0,
          typename T1>
TILEOP void TReciprocal(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::RECIPROCAL, PrecisionType, LastUse>(dst, src);
}

#define OP_TILE_OP_RELU TRelu
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TRelu(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::RELU, 0, LastUse>(dst, src);
}

#define OP_TILE_OP_SINH TSinh
template <typename T0, typename T1, typename T2>
TILEOP void TSinh(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr float SCALAR_ZERO_0199 = 0.0001998459335617813754003f;
    constexpr float SCALAR_ZERO_0833 = 0.00833308538698833f;
    constexpr float SCALAR_ZERO_166 = 0.16666668254541f;
    constexpr float SCALAR_ZERO_48 = 0.48f;
    constexpr float SCALAR_ONE = 1.0f;
    constexpr float SCALAR_ZERO_POINT_FIVE = 0.5f;
    constexpr float SCALAR_NEGATIVE_15 = -1.5f;
    constexpr float SCALAR_NEGATIVE_ONE = -1.0f;
    constexpr float SCALAR_ZERO = 0.0f;

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    constexpr auto tileShapeSize = TileOp::GetAnyAxisMergeResult<
        DIM_1ST, Std::tuple_size<typename T0::TileShape>::value, typename T0::TileShape>();

    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                    -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW * 4, pto::BLayout::RowMajor, -1, -1>;
    DstTileDefine dstTile(dstShape3, dstShape4);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    DstTileDefine tmp0Tile(dstShape3, dstShape4);
    DstTileDefine tmp1Tile(dstShape3, dstShape4);
    DstTileDefine tmp2Tile(dstShape3, dstShape4);
    DstTileDefine tmp3Tile(dstShape3, dstShape4);
    MaskTileDefine tmp1MaskTile(dstShape3, dstShape4);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T2::Type>(GenTileOffset(tmp, tileOffsets));
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + 2 * tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp3Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + 3 * tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp1MaskTile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + tileShapeSize * dstTypeSize));

                // sinh(x) = x + x^3 / 3! + x^5 / 5! + x^7 / 7! for small x
                pto::TABS(tmp0Tile, srcExecTile);
                SyncV();
                pto::TMUL(tmp1Tile, tmp0Tile, tmp0Tile);
                SyncV();
                pto::TMULS(tmp2Tile, tmp1Tile, SCALAR_ZERO_0199);
                SyncV();
                pto::TADDS(tmp2Tile, tmp2Tile, SCALAR_ZERO_0833);
                SyncV();
                pto::TMUL(tmp2Tile, tmp2Tile, tmp1Tile);
                SyncV();
                pto::TADDS(tmp2Tile, tmp2Tile, SCALAR_ZERO_166);
                SyncV();
                pto::TMUL(tmp2Tile, tmp2Tile, tmp1Tile);
                SyncV();
                pto::TADDS(tmp2Tile, tmp2Tile, SCALAR_ONE);
                SyncV();
                pto::TMUL(tmp2Tile, tmp2Tile, tmp0Tile);
                SyncV();

                // sinh(x) = 1/2 * (e^{x/2} - e^{-3x/2}) * e^{x/2} for large x
                pto::TMULS(tmp1Tile, tmp0Tile, SCALAR_ZERO_POINT_FIVE);
                SyncV();
                pto::TEXP<pto::ExpAlgorithm::HIGH_PRECISION>(tmp1Tile, tmp1Tile);
                SyncV();
                pto::TMULS(tmp3Tile, tmp0Tile, SCALAR_NEGATIVE_15);
                SyncV();
                pto::TEXP<pto::ExpAlgorithm::HIGH_PRECISION>(tmp3Tile, tmp3Tile);
                SyncV();
                pto::TSUB(tmp3Tile, tmp1Tile, tmp3Tile);
                SyncV();
                pto::TMULS(tmp3Tile, tmp3Tile, SCALAR_ZERO_POINT_FIVE);
                SyncV();
                pto::TMUL(tmp3Tile, tmp3Tile, tmp1Tile);
                SyncV();

                pto::TCMPS(tmp1MaskTile, tmp0Tile, SCALAR_ZERO_48, pto::CmpMode::LT);
                SyncV();
                pto::TSEL(dstTile, tmp1MaskTile, tmp2Tile, tmp3Tile, tmp0Tile);
                SyncV();

                pto::TMULS(tmp2Tile, dstTile, SCALAR_NEGATIVE_ONE);
                SyncV();
                pto::TCMPS(tmp1MaskTile, srcExecTile, SCALAR_ZERO, pto::CmpMode::GE);
                SyncV();
                pto::TSEL(dstTile, tmp1MaskTile, dstTile, tmp2Tile, tmp0Tile);
                SyncV();
            }
        }
    }
}

#ifdef __DAV_V220
template <typename T0, typename T1, typename T2>
TILEOP void TCoshCompute(T0 dstTile, T1 srcTile, T2 tmpTile)
{
    constexpr float SCALAR_ZERO_POINT_FIVE = 0.5f;
    constexpr float SCALAR_NEGATIVE_ONE_POINT_FIVE = -1.5f;

    // cosh(x) = 0.5 * (exp(x / 2) + exp(-3 * x / 2)) * exp(x / 2)
    pto::TABS(tmpTile, srcTile);
    SyncV();
    pto::TMULS(dstTile, tmpTile, SCALAR_NEGATIVE_ONE_POINT_FIVE);
    SyncV();
    pto::TMULS(tmpTile, tmpTile, SCALAR_ZERO_POINT_FIVE);
    SyncV();
    pto::TEXP<pto::ExpAlgorithm::HIGH_PRECISION>(tmpTile, tmpTile);
    SyncV();
    pto::TEXP<pto::ExpAlgorithm::HIGH_PRECISION>(dstTile, dstTile);
    SyncV();
    pto::TADD(dstTile, dstTile, tmpTile);
    SyncV();
    pto::TMULS(dstTile, dstTile, SCALAR_ZERO_POINT_FIVE);
    SyncV();
    pto::TMUL(dstTile, dstTile, tmpTile);
    SyncV();
}
#else
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
TILEOP void TCoshCompute(T0 dstTile, T1 srcTile, T2 tmp0Tile, T3 tmp1Tile, T4 tmp2Tile, T5 tmp0IntTile, T6 tmp2MaskTile)
{
    constexpr float kLog2e = 1.442695041f;
    constexpr float kNegLn2Hi = -0.6931471825f;
    constexpr float kLn2Lo = 1.9046542e-9f;
    constexpr float kExpMagic = 12583037.0f;
    constexpr int32_t kFp32MantissaBits = 23;
    constexpr float kNClamp = 126.0f;
    constexpr float kHalfInv8 = 0.125f;
    constexpr float kTwo = 2.0f;
    constexpr float kOvfThreshold = 90.0f;
    constexpr float kInf = std::numeric_limits<float>::infinity();

    // ax = abs(x), temporarily stored in dstTile.
    pto::TABS(dstTile, srcTile);
    // n = trunc(ax * log2(e))
    pto::TMULS(tmp0Tile, dstTile, kLog2e);
    pto::TCVT(tmp0Tile, tmp0Tile, pto::RoundMode::CAST_TRUNC);
    // n = min(n, 126)
    pto::TMINS(tmp0Tile, tmp0Tile, kNClamp);
    // r = ax
    pto::TADDS(tmp1Tile, dstTile, 0.0f);
    // r += n * (-ln2_hi)
    pto::TMULS(tmp2Tile, tmp0Tile, 0.0f);
    pto::TADDS(tmp2Tile, tmp2Tile, kNegLn2Hi);
    pto::TMULADDDST(tmp1Tile, tmp0Tile, tmp2Tile);
    // r += n * (+ln2_lo)
    pto::TMULS(tmp2Tile, tmp2Tile, 0.0f);
    pto::TADDS(tmp2Tile, tmp2Tile, kLn2Lo);
    pto::TMULADDDST(tmp1Tile, tmp0Tile, tmp2Tile);
    // p2 = 2^(n - 2)
    pto::TADDS(tmp0Tile, tmp0Tile, kExpMagic);
    pto::TSHLS(tmp0IntTile, tmp0IntTile, kFp32MantissaBits);
    // er = exp(r) * p2 = exp(ax) / 4
    pto::TEXP(tmp1Tile, tmp1Tile);
    pto::TMUL(tmp1Tile, tmp1Tile, tmp0Tile);
    // Generate the overflow mask before overwriting ax in dstTile.
    // NaN >= 90 is false, so the naturally computed NaN is preserved.
    pto::TCMPS(tmp2MaskTile, dstTile, kOvfThreshold, pto::CmpMode::GE);
    // dst = 0.125 / er
    pto::TRECIP(dstTile, tmp1Tile);
    pto::TMULS(dstTile, dstTile, kHalfInv8);
    // tmp0 = 2 * er
    pto::TMULS(tmp0Tile, tmp1Tile, kTwo);
    // dst = 2 * er + 0.125 / er
    pto::TADD(dstTile, tmp0Tile, dstTile);
    pto::TADDS(tmp0Tile, tmp0Tile, kInf);
    // Select +inf on overflow; otherwise preserve the regular result.
    // er in tmp1Tile is dead and can be used as TSEL scratch.
    pto::TSEL(dstTile, tmp2MaskTile, tmp0Tile, dstTile, tmp1Tile);
}
#endif

#define OP_TILE_OP_COSH TCosh
template <typename T0, typename T1, typename T2>
TILEOP void TCosh(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto tileShapeSize = TileOp::GetAnyAxisMergeResult<
        DIM_1ST, Std::tuple_size<typename T0::TileShape>::value, typename T0::TileShape>();

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    DataTileDefine dstTile(dstShape3, dstShape4);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);

#ifdef __DAV_V220
    DataTileDefine tmpTile(dstShape3, dstShape4);
#else
    using IntTileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW * sizeof(float), pto::BLayout::RowMajor,
                                     -1, -1>;

    DataTileDefine tmp0Tile(dstShape3, dstShape4);
    DataTileDefine tmp1Tile(dstShape3, dstShape4);
    DataTileDefine tmp2Tile(dstShape3, dstShape4);
    IntTileDefine tmp0IntTile(dstShape3, dstShape4);
    MaskTileDefine tmp2MaskTile(dstShape3, dstShape4);
#endif

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T2::Type>(GenTileOffset(tmp, tileOffsets));
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

#ifdef __DAV_V220
                pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr() + tmpByteOffset));
                TCoshCompute(dstTile, srcExecTile, tmpTile);
#else
                auto tmp0Addr = static_cast<uint64_t>(tmp.GetAddr() + tmpByteOffset);
                auto tmp1Addr = static_cast<uint64_t>(tmp.GetAddr() + tmpByteOffset + tileShapeSize * dstTypeSize);
                auto tmp2Addr = static_cast<uint64_t>(tmp.GetAddr() + tmpByteOffset + 2 * tileShapeSize * dstTypeSize);

                pto::TASSIGN(tmp0Tile, tmp0Addr);
                pto::TASSIGN(tmp1Tile, tmp1Addr);
                pto::TASSIGN(tmp2Tile, tmp2Addr);
                pto::TASSIGN(tmp0IntTile, tmp0Addr);
                pto::TASSIGN(tmp2MaskTile, tmp2Addr);

                TCoshCompute(dstTile, srcExecTile, tmp0Tile, tmp1Tile, tmp2Tile, tmp0IntTile, tmp2MaskTile);
#endif
            }
        }
    }
}

template <UnaryOp op, typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void reduceKCompute(T0 dst, T1 tmp0, T2 tmp1, T3 tmp2, T4 src0)
{
    // define the number of x div pi
    constexpr float PI_FOR_X_TODIV = 0.3183098733425140380859375;
    // define the PI for compute
    constexpr float PI_V2 = 3.140625;
    constexpr float KPI_FIRS_PI_MULS = 0.0009670257568359375;
    constexpr float KPI_TWI_PI_MULS = 6.2771141529083251953125e-7;
    constexpr float KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10;
    constexpr float KPI_FOR_PI_MULS = -1.0290623200529979163359041220560e-13;
    constexpr float POINT_FIVE = 0.5;
    constexpr float K2_SCA = -2.0;
    constexpr float M4_SCA = 4.0;
    constexpr float TRIG_ZERO = 0.0;
    constexpr float TRIG_ONE = 1.0;
    // define the number of down of pi_div
    constexpr float PI_DOWN = 1.57079637050628662109375;
    // kpi_2
    constexpr float PI_RESDOWN_ADDS_NEG = -0.00000004371139000189375;

    pto::TMULS(tmp0, src0, TRIG_ZERO);
    SyncV();
    pto::TADD(tmp2, src0, tmp0);
    SyncV();
    //  k=round(x/π), x0=x-kπ, x0 belongs to [-π/2, π/2]
    //  cos(x) = (-1)^k * sin(x0 + π/2)
    pto::TMULS(tmp0, tmp2, PI_FOR_X_TODIV);
    SyncV();
    if constexpr (op == UnaryOp::SIN) {
        pto::TCVT(tmp1, tmp0, pto::RoundMode::CAST_ROUND);
        SyncV();
    }
    if constexpr (op == UnaryOp::COS) {
        pto::TADDS(tmp0, tmp0, POINT_FIVE);
        SyncV();
        pto::TCVT(tmp1, tmp0, pto::RoundMode::CAST_RINT);
        SyncV();
    }
    pto::TCVT(tmp0, tmp1, pto::RoundMode::CAST_NONE);
    SyncV();

    // x -= k * pi_0
    pto::TMULS(dst, tmp0, PI_V2);
    SyncV();
    pto::TSUB(tmp2, tmp2, dst);
    SyncV();
    // x -= k * pi_1
    pto::TMULS(dst, tmp0, KPI_FIRS_PI_MULS);
    SyncV();
    pto::TSUB(tmp2, tmp2, dst);
    SyncV();
    // x = x + PI_DOWN
    if constexpr (op == UnaryOp::COS) {
        pto::TADDS(tmp2, tmp2, PI_DOWN);
        SyncV();
    }
    // x -= k * pi_2
    pto::TMULS(dst, tmp0, KPI_TWI_PI_MULS);
    SyncV();
    pto::TSUB(tmp2, tmp2, dst);
    SyncV();
    // x -= k * pi_3
    pto::TMULS(dst, tmp0, KPI_THIR_PI_MULS);
    SyncV();
    pto::TSUB(tmp2, tmp2, dst);
    SyncV();
    // x -= k * pi_4
    pto::TMULS(dst, tmp0, KPI_FOR_PI_MULS);
    SyncV();
    pto::TSUB(tmp2, tmp2, dst);
    SyncV();

    if constexpr (op == UnaryOp::COS) {
        // x = x + PI_RESDOWN_ADDS_NEG
        pto::TADDS(tmp2, tmp2, PI_RESDOWN_ADDS_NEG);
        SyncV();
    }
    // kover2
    pto::TMULS(dst, tmp0, POINT_FIVE);
    SyncV();
    pto::TCVT(tmp1, dst, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TCVT(dst, tmp1, pto::RoundMode::CAST_NONE);
    SyncV();
    // kover2floorm4
    pto::TMULS(dst, dst, M4_SCA);
    SyncV();
    // k2
    pto::TMULS(tmp0, tmp0, K2_SCA);
    SyncV();
    // sign
    pto::TADD(dst, dst, tmp0);
    SyncV();
    pto::TADDS(dst, dst, TRIG_ONE);
    SyncV();
}

template <UnaryOp op, typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void SinCosCompute(T0 dst, T1 tmp0, T2 tmp1, T3 tmp2, T4 src0)
{
    constexpr float RES_MULTI_SCA = 2.604926501e-6;
    constexpr float RES_ADDICT_UP = -0.0001980894471;
    constexpr float ADD2S = 0.008333049340;
    constexpr float ADD3S = -0.1666665792;
    constexpr float TRIG_ONE = 1.0;

    // x^2
    pto::TMUL(tmp0, tmp2, tmp2);
    SyncV();
    // sin(x) = x * P(x)
    // P(x) = (((x^2 * R0 + R1) * x^2 + R2) * x^2 + R3) * x^2 + 1.0
    // roundTensor = mul(x^2, 2.604926501e-6)
    pto::TMULS(tmp1, tmp0, RES_MULTI_SCA);
    SyncV();
    pto::TADDS(tmp1, tmp1, RES_ADDICT_UP);
    SyncV();
    // roundTensor = mul(roundTensor, x^2)
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, ADD2S);
    SyncV();
    // roundTensor = mul(roundTensor, x^2)
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, ADD3S);
    SyncV();
    // roundTensor = mul(roundTensor, x^2)
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, TRIG_ONE);
    SyncV();
    // sin(x) = x * P(x)
    pto::TMUL(tmp1, tmp2, tmp1);
    SyncV();
    pto::TMUL(dst, dst, tmp1);
    SyncV();
    return;
}
// P(x) = (((((0.053443748819x^2+0.75517016694e1)x^2+0.10162808918e3)x^2
//          +0.13938061484e4)x^2+0.50637915060e4)x^2+0.29639384698e5)x
template <typename T0, typename T1, typename T2>
TILEOP void ErfComputeP(T0 dst, T1 tmp0, T2 tmp1)
{
    constexpr float SCALAR_P0 = 0.29639384698e5;
    constexpr float SCALAR_P1 = 0.50637915060e4;
    constexpr float SCALAR_P2 = 0.13938061484e4;
    constexpr float SCALAR_P3 = 0.10162808918e3;
    constexpr float SCALAR_P4 = 0.75517016694e1;
    constexpr float SCALAR_P5 = 0.053443748819;
    // x^2
    pto::TMULS(tmp1, tmp0, SCALAR_P5);
    SyncV();
    pto::TADDS(tmp1, tmp1, SCALAR_P4);
    SyncV();
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, SCALAR_P3);
    SyncV();
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, SCALAR_P2);
    SyncV();
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, SCALAR_P1);
    SyncV();
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, SCALAR_P0);
    SyncV();
    pto::TMUL(tmp1, dst, tmp1);
    return;
}
// Q(x) = ((((x^2+0.31212858877e2)x^2+0.39856963806e3)x^2+0.30231248150e4)x^2+0.13243365831e5)x^2+0.26267224157e5
template <typename T0, typename T1>
TILEOP void ErfComputeQ(T0 tmp0, T1 tmp2)
{
    constexpr float SCALAR_Q0 = 0.26267224157e5;
    constexpr float SCALAR_Q1 = 0.13243365831e5;
    constexpr float SCALAR_Q2 = 0.30231248150e4;
    constexpr float SCALAR_Q3 = 0.39856963806e3;
    constexpr float SCALAR_Q4 = 0.31212858877e2;

    pto::TADDS(tmp2, tmp0, SCALAR_Q4);
    SyncV();
    pto::TMUL(tmp2, tmp0, tmp2);
    SyncV();
    pto::TADDS(tmp2, tmp2, SCALAR_Q3);
    SyncV();
    pto::TMUL(tmp2, tmp0, tmp2);
    SyncV();
    pto::TADDS(tmp2, tmp2, SCALAR_Q2);
    SyncV();
    pto::TMUL(tmp2, tmp0, tmp2);
    SyncV();
    pto::TADDS(tmp2, tmp2, SCALAR_Q1);
    SyncV();
    pto::TMUL(tmp2, tmp0, tmp2);
    SyncV();
    pto::TADDS(tmp2, tmp2, SCALAR_Q0);
    return;
}
// Erf(x) = P(x) / Q(x)
template <typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void ErfPadeCompute(T0 dst, T1 tmp0, T2 tmp1, T3 tmp2, T4 src)
{
    constexpr float ERF_BOUNDARY_MAX = 3.92;

    pto::TMINS(dst, src, ERF_BOUNDARY_MAX);
    SyncV();
    pto::TMAXS(dst, dst, -ERF_BOUNDARY_MAX);
    SyncV();
    // x^2
    pto::TMUL(tmp0, dst, dst);
    SyncV();
    ErfComputeP(dst, tmp0, tmp1);
    SyncV();
    ErfComputeQ(tmp0, tmp2);
    SyncV();
    pto::TDIV(dst, tmp1, tmp2);
    SyncV();
    return;
}

template <typename T0, typename T1, typename T2>
TILEOP void ErfSubsectionSmallCompute(T0 dst, T1 tmp2, T2 src)
{
    using FloatIntUnion = union {
        uint32_t i;
        float f;
    };
    pto::TMUL(dst, src, src);
    pto::TMULS(tmp2, dst, FloatIntUnion{.i = 0x38B1E96A}.f);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0xBA574D20}.f);
    pto::TMUL(tmp2, dst, tmp2);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0x3BAAD5EA}.f);
    pto::TMUL(tmp2, dst, tmp2);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0xBCDC1BE7}.f);
    pto::TMUL(tmp2, dst, tmp2);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0x3DE718AF}.f);
    pto::TMUL(tmp2, dst, tmp2);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0xBEC093AC}.f);
    pto::TMUL(tmp2, dst, tmp2);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0x3E0375D3}.f);
    pto::TMUL(tmp2, src, tmp2);
    pto::TADD(tmp2, tmp2, src);
    return;
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void ErfSubsectionLargeCompute(T0 dst, T1 tmp0, T2 tmp1, T3 src)
{
    using FloatIntUnion = union {
        uint32_t i;
        float f;
    };
    constexpr float LOG2_VALUE = 2.0f;
    constexpr float ZERO_VALUE = 0.0f;

    pto::TABS(tmp1, src);
    pto::TMULS(dst, tmp1, FloatIntUnion{0x38EB4C3A}.f);
    pto::TADDS(dst, dst, FloatIntUnion{0xBAAE005B}.f);
    pto::TMUL(dst, tmp1, dst);
    pto::TADDS(dst, dst, FloatIntUnion{0x3C09919F}.f);
    pto::TMUL(dst, tmp1, dst);
    pto::TADDS(dst, dst, FloatIntUnion{0xBD24D99A}.f);
    pto::TMUL(dst, tmp1, dst);
    pto::TADDS(dst, dst, FloatIntUnion{0x3E235519}.f);
    pto::TMUL(dst, tmp1, dst);
    pto::TADDS(dst, dst, FloatIntUnion{0x3F69B4F9}.f);
    pto::TMUL(dst, tmp1, dst);
    pto::TADDS(dst, dst, FloatIntUnion{0x3F210A14}.f);
    pto::TNEG(tmp1, tmp1);
    pto::TMUL(dst, tmp1, dst);
    pto::TADD(dst, dst, tmp1);

    pto::TEXPANDS(tmp1, LOG2_VALUE);
    pto::TLOG<pto::LogAlgorithm::DEFAULT>(tmp1, tmp1);
    pto::TMUL(dst, tmp1, dst);
    pto::TEXP<pto::ExpAlgorithm::DEFAULT>(dst, dst);
    pto::TEXPANDS(tmp1, FloatIntUnion{0x3F800000}.f);
    pto::TSUB(dst, tmp1, dst);
    pto::TCMPS(tmp0, src, ZERO_VALUE, pto::CmpMode::GE);
    pto::TNEG(tmp1, dst);
    // tmp0=1取正值，tmp0=0取负值
    pto::TSEL(dst, tmp0, dst, tmp1, tmp1);
    return;
}

template <typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void ErfSubsectionCompute(T0 dst, T1 tmp0, T2 tmp1, T3 tmp2, T4 src)
{
    using FloatIntUnion = union {
        uint32_t i;
        float f;
    };
    // tmp2
    ErfSubsectionSmallCompute(dst, tmp2, src);
    // dst
    ErfSubsectionLargeCompute(dst, tmp0, tmp1, src);

    pto::TABS(tmp1, src);
    pto::TCMPS(tmp0, tmp1, FloatIntUnion{0x3F8060FE}.f, pto::CmpMode::GE);
    // A5 TSEL的tmp未使用
    pto::TSEL(dst, tmp0, dst, tmp2, tmp2);
    return;
}

template <UnaryOp op, typename T0, typename T1, typename T2>
TILEOP void TrigErfCompute(T0 dst, T1 tmp, T2 src)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    using TmpFP32Tile = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using TmpINT32Tile = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using TmpMaskTile = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW * 4, pto::BLayout::RowMajor, -1, -1>;

    TmpFP32Tile dstTile(shape3, shape4);
    TmpFP32Tile tmp0Tile(shape3, shape4);
    TmpINT32Tile tmp1Tile(shape3, shape4);
    TmpFP32Tile tmp2Tile(shape3, shape4);
    TmpFP32Tile tmp3Tile(shape3, shape4);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + GenTileOffset(dst, tileOffsets) * dstTypeSize));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr()));
                pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + 2 * tileW * tileH * dstTypeSize));
                if constexpr (op == UnaryOp::ERF) {
                    pto::TASSIGN(tmp3Tile, (uint64_t)(tmp.GetAddr() + tileW * tileH * dstTypeSize));
#ifdef __DAV_V220
                    ErfPadeCompute(dstTile, tmp0Tile, tmp3Tile, tmp2Tile, srcExecTile);
#else
                    TmpMaskTile tmpmaskTile(shape3, shape4);
                    pto::TASSIGN(tmpmaskTile, (uint64_t)(tmp.GetAddr()));
                    ErfSubsectionCompute(dstTile, tmpmaskTile, tmp3Tile, tmp2Tile, srcExecTile);
#endif
                } else {
                    pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + tileW * tileH * dstTypeSize));

                    reduceKCompute<op>(dstTile, tmp0Tile, tmp1Tile, tmp2Tile, srcExecTile);
                    SyncV();
                    TmpFP32Tile tmp3Tile(shape3, shape4);
                    pto::TASSIGN(tmp3Tile, (uint64_t)(tmp.GetAddr() + tileW * tileH * dstTypeSize));
                    SinCosCompute<op>(dstTile, tmp0Tile, tmp3Tile, tmp2Tile, srcExecTile);
                }
            }
        }
    }
}

template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void PackCompute(T0 dst, T1 src)
{
    auto shape = dst.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    if constexpr (std::is_same_v<typename T0::Type, int64_t> || std::is_same_v<typename T0::Type, uint64_t>) {
        using TileDef = pto::Tile<pto::TileType::Vec, int32_t, 1, tileW * 2, pto::BLayout::RowMajor, -1, -1>;
        TileDef dstTile(1, shape * 2);
        TileDef srcTile(1, shape * 2);
        pto::TASSIGN(dstTile, dst.GetAddr());
        pto::TASSIGN(srcTile, src.GetAddr());
        pto::TMOV(dstTile, srcTile);
    } else {
        using TileDef = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileDef dstTile(1, shape);
        TileDef srcTile(1, shape);
        pto::TASSIGN(dstTile, dst.GetAddr());
        pto::TASSIGN(srcTile, src.GetAddr());
        pto::TMOV(dstTile, srcTile);
    }
}

#define OP_TILE_OP_PACK TPack
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TPack(T0 dst, T1 src)
{
    PackCompute(dst, src);
}

#define OP_TILE_OP_UNPACK TUnPack
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TUnPack(T0 dst, T1 src)
{
    PackCompute(dst, src);
}

#define OP_TILE_OP_SIN TSin
template <typename T0, typename T1, typename T2>
TILEOP void TSin(T0 dst, T1 tmp, T2 src)
{
    TrigErfCompute<UnaryOp::SIN>(dst, tmp, src);
}

#define OP_TILE_OP_COS TCos
template <typename T0, typename T1, typename T2>
TILEOP void TCos(T0 dst, T1 tmp, T2 src)
{
    TrigErfCompute<UnaryOp::COS>(dst, tmp, src);
}

#define OP_TILE_OP_ERF TErf
template <typename T0, typename T1, typename T2>
TILEOP void TErf(T0 dst, T1 tmp, T2 src)
{
    TrigErfCompute<UnaryOp::ERF>(dst, tmp, src);
}

constexpr float ERFC_FP32_MIN = 2.168404344971009e-19f;
constexpr float ERFC_BOUNDARY_MAX = 10.0f;
constexpr float ERFC_NEG_BOUNDARY_MAX = -10.0f;
constexpr float ERFC_NEG_ONE = -1.0f;
constexpr float ERFC_ONE = 1.0f;

constexpr float ERFC_R0 = 0.1735313680e-7f;
constexpr float ERFC_R1 = -0.9856738394e-6f;
constexpr float ERFC_R2 = 0.2517003236e-4f;
constexpr float ERFC_R3 = -0.3848015171e-3f;
constexpr float ERFC_R4 = 0.5681528564e0f;
constexpr float ERFC_R5 = 0.5245623129e1f;
constexpr float ERFC_R6 = 0.2107740710e2f;
constexpr float ERFC_R7 = 0.4212761755e2f;
constexpr float ERFC_R8 = 0.4380524149e2f;

constexpr float ERFC_S1 = 0.9349684299e1f;
constexpr float ERFC_S2 = 0.3756930664e2f;
constexpr float ERFC_S3 = 0.8058268949e2f;
constexpr float ERFC_S4 = 0.9155653738e2f;
constexpr float ERFC_S5 = 0.4380524152e2f;

template <typename DstTileType, typename SrcTileType>
TILEOP inline void ErfcClip(DstTileType& dst, const SrcTileType& src)
{
    pto::TMINS(dst, src, ERFC_BOUNDARY_MAX);
    SyncV();
    pto::TMAXS(dst, dst, ERFC_NEG_BOUNDARY_MAX);
    SyncV();
}

template <typename TileType>
TILEOP inline void ErfcPreCompute(TileType& xb, const TileType& clipped_x, TileType& xa)
{
    pto::TABS(xa, clipped_x);
    SyncV();
    pto::TADDS(xa, xa, ERFC_FP32_MIN);
    SyncV();
    pto::TDIV(xb, clipped_x, xa);
    SyncV();
}

template <typename TileType>
TILEOP inline void ErfcComputeR(TileType& tmpCompBuf2, TileType& tmpCompBuf3, const TileType& z)
{
    pto::TMULS(tmpCompBuf2, z, ERFC_R0);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R1);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R2);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R3);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R4);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R5);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R6);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R7);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R8);
    SyncV();
}

template <typename TileType>
TILEOP inline void ErfcComputeS(TileType& tmpCompBuf2, TileType& tmpCompBuf4, const TileType& z)
{
    pto::TADDS(tmpCompBuf2, z, ERFC_S1);
    SyncV();
    pto::TMUL(tmpCompBuf4, z, tmpCompBuf2);
    SyncV();
    pto::TADDS(tmpCompBuf2, tmpCompBuf4, ERFC_S2);
    SyncV();
    pto::TMUL(tmpCompBuf4, z, tmpCompBuf2);
    SyncV();
    pto::TADDS(tmpCompBuf2, tmpCompBuf4, ERFC_S3);
    SyncV();
    pto::TMUL(tmpCompBuf4, z, tmpCompBuf2);
    SyncV();
    pto::TADDS(tmpCompBuf2, tmpCompBuf4, ERFC_S4);
    SyncV();
    pto::TMUL(tmpCompBuf4, z, tmpCompBuf2);
    SyncV();
    pto::TADDS(tmpCompBuf2, tmpCompBuf4, ERFC_S5);
    SyncV();
}

template <typename TileType>
TILEOP inline void ErfcPublicSteps(TileType& tmpCompBuf1, TileType& tmpCompBuf2, TileType& tmpCompBuf3,
                                   TileType& tmpCompBuf4)
{
    ErfcComputeR(tmpCompBuf2, tmpCompBuf3, tmpCompBuf1);
    ErfcComputeS(tmpCompBuf2, tmpCompBuf4, tmpCompBuf1);

    pto::TDIV(tmpCompBuf2, tmpCompBuf3, tmpCompBuf2);
    SyncV();
    pto::TMUL(tmpCompBuf1, tmpCompBuf1, tmpCompBuf1);
    SyncV();
    pto::TMULS(tmpCompBuf1, tmpCompBuf1, ERFC_NEG_ONE);
    SyncV();
    pto::TEXP(tmpCompBuf1, tmpCompBuf1);
    SyncV();
    pto::TMUL(tmpCompBuf2, tmpCompBuf1, tmpCompBuf2);
    SyncV();
}

template <typename TileType>
TILEOP inline void ErfcPostCompute(TileType& dst, const TileType& xb, TileType& tmpCompBuf2, TileType& tmpCompBuf3)
{
    pto::TMULS(tmpCompBuf3, xb, ERFC_NEG_ONE);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf3, ERFC_ONE);
    SyncV();
    pto::TMUL(tmpCompBuf2, tmpCompBuf2, xb);
    SyncV();
    pto::TADD(dst, tmpCompBuf2, tmpCompBuf3);
    SyncV();
}

#define OP_TILE_OP_ERFC TErfc
template <typename T0, typename T1, typename T2>
TILEOP void TErfc(T0 dst, T1 tmp, T2 src)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    if (shape0 == 0 || shape1 == 0 || shape2 == 0 || shape3 == 0 || shape4 == 0) {
        return;
    }

    using TmpFP32Tile = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    TmpFP32Tile dstTile(shape3, shape4);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    TmpFP32Tile tmpCompBuf1(shape3, shape4);
    TmpFP32Tile tmpCompBuf2(shape3, shape4);
    TmpFP32Tile tmpCompBuf3(shape3, shape4);
    TmpFP32Tile tmpCompBuf4(shape3, shape4);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + GenTileOffset(dst, tileOffsets) * dstTypeSize));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

                pto::TASSIGN(tmpCompBuf1, (uint64_t)(tmp.GetAddr()));
                pto::TASSIGN(tmpCompBuf2, (uint64_t)(tmp.GetAddr() + 1 * tileW * tileH * dstTypeSize));
                pto::TASSIGN(tmpCompBuf3, (uint64_t)(tmp.GetAddr() + 2 * tileW * tileH * dstTypeSize));
                pto::TASSIGN(tmpCompBuf4, (uint64_t)(tmp.GetAddr() + 3 * tileW * tileH * dstTypeSize));

                ErfcClip(dstTile, srcExecTile);
                ErfcPreCompute(dstTile, dstTile, tmpCompBuf1);
                ErfcPublicSteps(tmpCompBuf1, tmpCompBuf2, tmpCompBuf3, tmpCompBuf4);
                ErfcPostCompute(dstTile, dstTile, tmpCompBuf2, tmpCompBuf3);
            }
        }
    }
}

// Horner evaluation of arcsin Taylor on t in [0, 1/sqrt(2)]:
//   arcsin(t) = t * (c0 + c1*s + c2*s^2 + ... + c7*s^7),  s = t^2
template <typename TOut, typename TIn, typename TScratch>
TILEOP void ArcsinPolyHorner(TOut outTile, TIn tTile, TScratch sScratch)
{
    constexpr float ASIN_C0 = 1.0f;        // 1
    constexpr float ASIN_C1 = 0.16666667f; // 1/6
    constexpr float ASIN_C2 = 0.075f;      // 3/40
    constexpr float ASIN_C3 = 0.04464286f; // 5/112
    constexpr float ASIN_C4 = 0.03038194f; // 35/1152
    constexpr float ASIN_C5 = 0.02237216f; // 63/2816
    constexpr float ASIN_C6 = 0.01735276f; // 231/13312
    constexpr float ASIN_C7 = 0.01396484f; // 143/10240

    // s = t^2
    pto::TMUL(sScratch, tTile, tTile);
    SyncV();
    // acc = c7*s + c6
    pto::TMULS(outTile, sScratch, ASIN_C7);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C6);
    SyncV();
    // acc = acc*s + c5
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C5);
    SyncV();
    // acc = acc*s + c4
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C4);
    SyncV();
    // acc = acc*s + c3
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C3);
    SyncV();
    // acc = acc*s + c2
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C2);
    SyncV();
    // acc = acc*s + c1
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C1);
    SyncV();
    // acc = acc*s + c0
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C0);
    SyncV();
    // result = acc * t
    pto::TMUL(outTile, outTile, tTile);
    SyncV();
}

template <bool IsAsin, typename TDst, typename TSrc, typename TTmp0, typename TTmp1, typename TTmp2, typename TMask>
TILEOP void TAsinAcosTileImpl(TDst dstTile, TSrc srcTile, TTmp0 tmp0Tile, TTmp1 tmp1Tile, TTmp2 tmp2Tile,
                              TMask maskTile)
{
    constexpr float ASIN_THRESHOLD = 0.70710678f; // 1/sqrt(2)
    constexpr float PI_HALF = 1.57079633f;
    constexpr float SCALAR_ONE = 1.0f;
    constexpr float SCALAR_NEGATIVE_ONE = -1.0f;
    constexpr float SCALAR_ZERO = 0.0f;

    // ---- 1) tmp0 = |x| ----
    pto::TABS(tmp0Tile, srcTile);
    SyncV();

    // ---- 2) Reduce both branches to t in [0, 1/sqrt(2)] ----
    // tmp1 = sqrt(1 - x^2), the reduced argument for the large branch
    pto::TMUL(tmp1Tile, tmp0Tile, tmp0Tile);
    SyncV();
    pto::TMULS(tmp1Tile, tmp1Tile, SCALAR_NEGATIVE_ONE);
    SyncV();
    pto::TADDS(tmp1Tile, tmp1Tile, SCALAR_ONE);
    SyncV();
    pto::TSQRT(tmp1Tile, tmp1Tile);
    SyncV();

    // tmp2 = |x| for the small branch, sqrt(1 - x^2) for the large branch
    pto::TCMPS(maskTile, tmp0Tile, ASIN_THRESHOLD, pto::CmpMode::LE);
    SyncV();
    // dstTile is still unused and serves as TSEL scratch on A2/A3.
    pto::TSEL(tmp2Tile, maskTile, tmp0Tile, tmp1Tile, dstTile);
    SyncV();

    // ---- 3) Evaluate the shared polynomial once ----
    // tmp1 is no longer needed and is reused as Horner scratch.
    ArcsinPolyHorner(dstTile, tmp2Tile, tmp1Tile);

    // tmp0 = pi/2 - poly(t), used only by the large branch
    pto::TMULS(tmp0Tile, dstTile, SCALAR_NEGATIVE_ONE);
    SyncV();
    pto::TADDS(tmp0Tile, tmp0Tile, PI_HALF);
    SyncV();
    pto::TSEL(dstTile, maskTile, dstTile, tmp0Tile, tmp1Tile);
    SyncV();
    // dst now == arcsin(|x|), >= 0

    // ---- 4) Sign restore ----
    if constexpr (IsAsin) {
        // arcsin is odd: dst = src >= 0 ? dst : -dst
        pto::TMULS(tmp0Tile, dstTile, SCALAR_NEGATIVE_ONE);
        SyncV();
        pto::TCMPS(maskTile, srcTile, SCALAR_ZERO, pto::CmpMode::GE);
        SyncV();
        pto::TSEL(dstTile, maskTile, dstTile, tmp0Tile, tmp1Tile);
        SyncV();
    } else {
        // arccos(x) = pi/2 - sign(src)*arcsin(|x|)
        //   src >= 0: pi/2 - dst
        //   src <  0: pi/2 + dst
        pto::TMULS(tmp0Tile, dstTile, SCALAR_NEGATIVE_ONE);
        SyncV();
        pto::TCMPS(maskTile, srcTile, SCALAR_ZERO, pto::CmpMode::GE);
        SyncV();
        pto::TSEL(dstTile, maskTile, tmp0Tile, dstTile, tmp1Tile);
        SyncV();
        pto::TADDS(dstTile, dstTile, PI_HALF);
        SyncV();
    }
}

// Unified body for TAsin / TAcos.
//   |x| <= 1/sqrt(2):  arcsin(|x|) via 8-term Taylor on |x|
//   |x| >  1/sqrt(2):  arcsin(|x|) = pi/2 - arcsin(sqrt(1 - x^2))
template <bool IsAsin, typename T0, typename T1, typename T2>
TILEOP void TAsinAcosImpl(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW * 4, pto::BLayout::RowMajor, -1, -1>;

    DataTileDefine dstTile(shape3, shape4);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    DataTileDefine tmp0Tile(shape3, shape4); // |x|, then large-branch result / negated result
    DataTileDefine tmp1Tile(shape3, shape4); // large-branch argument, then Horner / TSEL scratch
    DataTileDefine tmp2Tile(shape3, shape4); // shared reduced argument
    MaskTileDefine maskTile(shape3, shape4);

    constexpr size_t tmpStride = tileH * tileW * dstTypeSize;
    pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + 0 * tmpStride));
    pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + 1 * tmpStride));
    pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + 2 * tmpStride));
    pto::TASSIGN(maskTile, (uint64_t)(tmp.GetAddr() + 3 * tmpStride));

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + GenTileOffset(dst, tileOffsets) * dstTypeSize));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                TAsinAcosTileImpl<IsAsin>(dstTile, srcExecTile, tmp0Tile, tmp1Tile, tmp2Tile, maskTile);
            }
        }
    }
}

#define OP_TILE_OP_ASIN TAsin
template <typename T0, typename T1, typename T2>
TILEOP void TAsin(T0 dst, T1 src, T2 tmp)
{
    TAsinAcosImpl<true>(dst, src, tmp);
}

#define OP_TILE_OP_ACOS TAcos
template <typename T0, typename T1, typename T2>
TILEOP void TAcos(T0 dst, T1 src, T2 tmp)
{
    TAsinAcosImpl<false>(dst, src, tmp);
}

#define OP_TILE_OP_ASINH TASinh
template <typename T0, typename T1, typename T2>
TILEOP void TASinh(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    constexpr float CONST_BRANCH_CONDITION = 0.00024414063;
    constexpr float CONST_ZERO = 0.0f;
    constexpr float CONST_ONE = 1.0f;
    constexpr float CONST_NEG_ONE = -1.0f;
    constexpr float CONST_COMPARE_VALUE_MIN = 1e-45f;
    constexpr float CONST_COMPARE_VALUE_MAX = 3.4028235e34f;
    constexpr float CONST_LOG_TWO_VALUE = 6.93147180559945286227e-01f;

    constexpr auto tileShapeSize = TileOp::GetAnyAxisMergeResult<
        DIM_1ST, Std::tuple_size<typename T0::TileShape>::value, typename T0::TileShape>();

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW, pto::BLayout::RowMajor, -1, -1>;
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    DataTileDefine dstTile(dstShape3, dstShape4);
    DataTileDefine tmp0Tile(dstShape3, dstShape4);
    DataTileDefine tmp1Tile(dstShape3, dstShape4);
    DataTileDefine tmp2Tile(dstShape3, dstShape4);
    DataTileDefine tmp3Tile(dstShape3, dstShape4);
    MaskTileDefine tmp2MaskTile(dstShape3, dstShape4);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T2::Type>(GenTileOffset(tmp, tileOffsets));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));

                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + 2 * tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp3Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + 3 * tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp2MaskTile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + 2 * tileShapeSize * dstTypeSize));

                pto::TABS(tmp0Tile, srcExecTile); // |x|
                SyncV();
                pto::TDIVS(tmp1Tile, CONST_ONE, tmp0Tile); // 1/|x|
                SyncV();
                pto::TMUL(tmp2Tile, tmp1Tile, tmp1Tile); // 1/(|x|)^2
                SyncV();

                pto::TADDS(tmp3Tile, tmp2Tile, CONST_ONE); // 1 + 1/(|x|)^2
                SyncV();
                pto::TSQRT(tmp3Tile, tmp3Tile); // sqrt(1 + 1/(|x|)^2)
                SyncV();
                pto::TADD(tmp1Tile, tmp3Tile, tmp1Tile); // sqrt(1 + 1/(|x|)^2) + 1/|x|
                SyncV();
                pto::TDIV(tmp1Tile, tmp0Tile, tmp1Tile); // |x| / (sqrt(1 + 1/(|x|)^2) + 1/|x|)
                SyncV();
                pto::TADD(tmp1Tile, tmp0Tile, tmp1Tile); // r = |x| + |x| / (sqrt(1 + 1/(|x|)^2) + 1/|x|)
                SyncV();
                pto::TADDS(tmp3Tile, tmp1Tile, CONST_ONE); // r + 1
                SyncV();

                pto::TADDS(dstTile, tmp3Tile, CONST_NEG_ONE); // clamp(r, s_min, s_max)
                SyncV();
                pto::TMAXS(dstTile, dstTile, CONST_COMPARE_VALUE_MIN);
                SyncV();
                pto::TMINS(dstTile, dstTile, CONST_COMPARE_VALUE_MAX);
                SyncV();

                pto::TLOG(tmp3Tile, tmp3Tile); // log(r + 1)
                SyncV();
                pto::TMUL(tmp1Tile, tmp1Tile, tmp3Tile); // r * log(r + 1)
                SyncV();
                pto::TDIV(tmp1Tile, tmp1Tile, dstTile); // r * log(r + 1) / clamp(r, s_min, s_max)
                SyncV();

                pto::TLOG(tmp3Tile, tmp0Tile); // log(|x|)
                SyncV();
                pto::TADDS(tmp3Tile, tmp3Tile, CONST_LOG_TWO_VALUE); // log(|x|) + log2
                SyncV();
                pto::TADD(tmp2Tile, tmp3Tile, tmp2Tile); // log(|x|) + log2 + 1/(|x|)^2
                SyncV();
                pto::TMIN(tmp1Tile, tmp1Tile, tmp2Tile); // min
                SyncV();

                pto::TCMPS(tmp2MaskTile, tmp0Tile, CONST_BRANCH_CONDITION, pto::CmpMode::LT);
                SyncV();
                pto::TSEL(tmp0Tile, tmp2MaskTile, tmp0Tile, tmp1Tile, tmp3Tile);
                SyncV();
                pto::TMULS(tmp1Tile, tmp0Tile, CONST_NEG_ONE);
                SyncV();

                pto::TCMPS(tmp2MaskTile, srcExecTile, CONST_ZERO, pto::CmpMode::GE);
                SyncV();
                pto::TSEL(dstTile, tmp2MaskTile, tmp0Tile, tmp1Tile, tmp3Tile);
                SyncV();
            }
        }
    }
}

#define OP_TILE_OP_ACOSH TACosh
template <typename T0, typename T1, typename T2>
TILEOP void TACosh(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr float CONST_ONE = 1.0f;
    constexpr float CONST_NEG_ONE = -1.0f;
    constexpr float CONST_COMPARE_VALUE_MIN = 1e-45f;
    constexpr float CONST_COMPARE_VALUE_MAX = 3.4028235e34f;
    constexpr float CONST_LOG_TWO_VALUE = 6.93147180559945286227e-01f;

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    constexpr auto tileShapeSize = TileOp::GetAnyAxisMergeResult<
        DIM_1ST, Std::tuple_size<typename T0::TileShape>::value, typename T0::TileShape>();

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    DataTileDefine dstTile(dstShape3, dstShape4);
    DataTileDefine tmp0Tile(dstShape3, dstShape4);
    DataTileDefine tmp1Tile(dstShape3, dstShape4);
    DataTileDefine tmp2Tile(dstShape3, dstShape4);
    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T2::Type>(GenTileOffset(tmp, tileOffsets));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));

                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + 2 * tileShapeSize * dstTypeSize));

                pto::TADDS(tmp0Tile, srcExecTile, CONST_NEG_ONE); // t
                SyncV();
                pto::TADD(tmp1Tile, tmp0Tile, tmp0Tile); // 2t
                SyncV();
                pto::TMUL(tmp2Tile, tmp0Tile, tmp0Tile); // t^2
                SyncV();
                pto::TADD(tmp1Tile, tmp1Tile, tmp2Tile); // t^2 + 2t
                SyncV();
                pto::TSQRT(tmp1Tile, tmp1Tile); // sqrt(t^2 + 2t)
                SyncV();
                pto::TADD(tmp1Tile, tmp1Tile, tmp0Tile); // t + sqrt(t^2 + 2t) = r
                SyncV();
                pto::TADDS(tmp2Tile, tmp1Tile, CONST_ONE); // r + 1
                SyncV();

                pto::TADDS(tmp0Tile, tmp2Tile, CONST_NEG_ONE); // clamp(r, s_min, s_max)
                SyncV();
                pto::TMAXS(tmp0Tile, tmp0Tile, CONST_COMPARE_VALUE_MIN);
                SyncV();
                pto::TMINS(tmp0Tile, tmp0Tile, CONST_COMPARE_VALUE_MAX);
                SyncV();

                pto::TLOG(dstTile, tmp2Tile); // log(r + 1)
                SyncV();
                pto::TMUL(dstTile, dstTile, tmp1Tile); // r * log(r + 1)
                SyncV();
                pto::TDIV(dstTile, dstTile, tmp0Tile); // r * log(r + 1) / clamp(r, s_min, s_max)
                SyncV();

                pto::TLOG<pto::LogAlgorithm::HIGH_PRECISION>(tmp0Tile, srcExecTile); // log(x)
                SyncV();
                pto::TADDS(tmp0Tile, tmp0Tile, CONST_LOG_TWO_VALUE); // log(x) + log(2)
                SyncV();
                pto::TMIN(dstTile, dstTile, tmp0Tile);
                SyncV();
            }
        }
    }
}
#endif
