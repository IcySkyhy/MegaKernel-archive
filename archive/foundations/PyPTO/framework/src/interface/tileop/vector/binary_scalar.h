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
 * \file binary_scalar.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BINARY_SCALAR__H
#define TILEOP_TILE_OPERATOR_BINARY_SCALAR__H
#include "binary.h"

template <BinaryScalarOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarComputeImpl(T0 dst, T1 src0, Scalar src1)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    if constexpr (op == BinaryScalarOp::ADD) {
        PTO_WITH_LAST_USE(pto::TADDS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::SUB) {
        if constexpr (std::is_same<Scalar, half>::value) {
            PTO_WITH_LAST_USE(
                pto::TADDS(dst, src0, static_cast<half>(static_cast<float>(-1) * static_cast<float>(src1))), n1, n2);
        } else {
            PTO_WITH_LAST_USE(pto::TADDS(dst, src0, -src1), n1, n2);
        }
        return;
    }

    if constexpr (op == BinaryScalarOp::MUL) {
        PTO_WITH_LAST_USE(pto::TMULS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::DIV) {
        PTO_WITH_LAST_USE(pto::TDIVS<PrecisionType>(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::MAX) {
        PTO_WITH_LAST_USE(pto::TMAXS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::MIN) {
        PTO_WITH_LAST_USE(pto::TMINS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::BITWISEAND) {
        pto::TANDS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::BITWISEOR) {
        pto::TORS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::MOD) {
        pto::TFMODS<PrecisionType>(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::LRELU) {
        pto::TLRELU(dst, src0, src1);
        return;
    }
}

template <BinaryScalarOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarCompute(T0 dst, T1 src0, Scalar src1)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }

    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                BinaryScalarComputeImpl<op, PrecisionType, LastUse>(dstTile.Data(), src0ExecTile, src1);
            }
        }
    }
}

#define OP_TILE_OP_ADDS TAddS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TAddS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::ADD, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_SUBS TSubS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TSubS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::SUB, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MULS TMulS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMulS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MUL, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_DIVS TDivS
template <auto PrecisionType = pto::DivAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename Scalar,
          typename T0, typename T1>
TILEOP void TDivS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::DIV, PrecisionType, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MAXS TMaxS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMaxS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MAX, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MINS TMinS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMinS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MIN, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_LRELU TLReLU
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TLReLU(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::LRELU, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEANDS TBitwiseAndS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseAndS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::BITWISEAND, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEORS TBitwiseOrS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseOrS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::BITWISEOR, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_GCDS TGcdS
template <typename Scalar, typename T0, typename T1>
TILEOP void TGcdS(T0 dst, T1 src0, Scalar src1)
{
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src0.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto srcStride3 = srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    auto src0Addr = (__ubuf__ typename T1::Type*)((uint64_t)(src0.GetAddr()));
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    using inputType = typename T1::Type;

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (LoopVar n = 0; n < shape0; n++) {
        for (LoopVar j = 0; j < shape1; j++) {
            for (LoopVar k = 0; k < shape2; k++) {
                for (LoopVar m = 0; m < shape3; m++) {
                    for (LoopVar i = 0; i < shape4; i++) {
                        int dstOffset = n * dstStride0 + j * dstStride1 + k * dstStride2 + m * dstStride3 + i;
                        int srcOffset = n * srcStride0 + j * srcStride1 + k * srcStride2 + m * srcStride3 + i;
                        inputType a = src0Addr[srcOffset];
                        inputType b = src1;
                        if (a < 0) {
                            a = 0 - a;
                        }
                        if (b < 0) {
                            b = 0 - b;
                        }
                        if (b == 0) {
                            dstAddr[dstOffset] = a;
                            continue;
                        }
                        while (a % b != 0) {
                            inputType c = a % b;
                            a = b;
                            b = c;
                        }
                        dstAddr[dstOffset] = b;
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

#define OP_TILE_OP_MODS TModS
template <auto PrecisionType = pto::FmodSAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename Scalar,
          typename T0, typename T1>
TILEOP void TModS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MOD, PrecisionType, LastUse>(dst, src0, src1);
}

template <BinaryScalarOp op, auto PrecisionType = 0, typename T0, typename T1, typename Scalar, typename T2>
TILEOP void BinaryScalarTmpComputeImpl(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    if constexpr (op == BinaryScalarOp::BITWISEXOR) {
        pto::TXORS(dst, src0, src1, tmp);
        return;
    }
    if constexpr (op == BinaryScalarOp::REM) {
        pto::TREMS<PrecisionType>(dst, src0, src1, tmp);
        return;
    }
    if constexpr (op == BinaryScalarOp::POW) {
        pto::TPOWS<PrecisionType>(dst, src0, src1, tmp);
        return;
    }
}

template <BinaryScalarOp op, auto PrecisionType = 0, typename T0, typename T1, typename Scalar, typename T2>
TILEOP void BinaryScalarTmpCompute(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    if constexpr (op == BinaryScalarOp::REM) {
        auto tmpTile = PtoTile<T2>(tmp);
        for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                    dstTile.Assign(dst, tileOffsets);
                    AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                    tmpTile.Assign(tmp, tileOffsets);
                    BinaryScalarTmpComputeImpl<op, PrecisionType>(dstTile.Data(), src0ExecTile, src1, tmpTile.Data());
                }
            }
        }
    } else {
        auto tmpExecTile = MakeElementwiseOperandExecTile(dst, tmp);
        for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                    dstTile.Assign(dst, tileOffsets);
                    AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                    AssignElementwiseOperandExecTile(tmpExecTile, tmp, tileOffsets);
                    BinaryScalarTmpComputeImpl<op, PrecisionType>(dstTile.Data(), src0ExecTile, src1, tmpExecTile);
                }
            }
        }
    }
}

#define OP_TILE_OP_BITWISEXORS TBitwiseXorS
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TBitwiseXorS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    BinaryScalarTmpCompute<BinaryScalarOp::BITWISEXOR, 0>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_REMS TRemainderS
template <typename Scalar, auto PrecisionType = pto::RemSAlgorithm::DEFAULT, typename T0, typename T1, typename T2>
TILEOP void TRemainderS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    BinaryScalarTmpCompute<BinaryScalarOp::REM, PrecisionType>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_POWS TPowS
template <auto PrecisionType = pto::PowAlgorithm::DEFAULT, typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TPowS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    BinaryScalarTmpCompute<BinaryScalarOp::POW, PrecisionType>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_REMRS TRemainderRS
template <typename Scalar, auto PrecisionType = pto::RemAlgorithm::DEFAULT, typename T0, typename T1, typename T2>
TILEOP void TRemainderRS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();
    using tmp0TileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor,
                                     -1, -1>;
    using tmp1TileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 2, tmpTileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    tmp0TileDefine tmp0Tile(shape3, shape4);
    tmp1TileDefine tmp1Tile(2, shape4);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr()));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + shape3 * tmpTileW * sizeof(typename T2::Type)));
                pto::TEXPANDS(tmp0Tile, src1);
#ifdef __DAV_V220
                pipe_barrier(PIPE_V);
#endif
                pto::TREM<PrecisionType>(dstTile.Data(), tmp0Tile, src0ExecTile, tmp1Tile);
            }
        }
    }
}

#define OP_TILE_OP_FLOORDIVS TFloorDivS
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TFloorDivS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto tileShapeSize = TileOp::GetAnyAxisMergeResult<
        DIM_1ST, Std::tuple_size<typename T0::TileShape>::value, typename T0::TileShape>();

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    DataTileDefine src0Tile(dstShape3, dstShape4);
    DataTileDefine dstTile(dstShape3, dstShape4);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto srcOffset = GenTileOffset(src0, tileOffsets);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + srcOffset * dstTypeSize));
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));

                if constexpr (std::is_same_v<typename T0::Type, half> ||
                              std::is_same_v<typename T0::Type, bfloat16_t>) {
                    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor,
                                                     -1, -1>;
                    Fp32TileDefine tmp0Tile(dstShape3, dstShape4);
                    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
                    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Tile, tmp0Tile, static_cast<float>(src1));
                    SyncV();
                    pto::TCVT(tmp0Tile, tmp0Tile, pto::RoundMode::CAST_FLOOR);
                    SyncV();
                    pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_RINT);
                    SyncV();
                } else if constexpr (std::is_same_v<typename T0::Type, float>) {
                    pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(dstTile, src0Tile, static_cast<float>(src1));
                    SyncV();
                    pto::TCVT(dstTile, dstTile, pto::RoundMode::CAST_FLOOR);
                    SyncV();
                }

#ifdef __DAV_V220
                if constexpr (std::is_same_v<typename T0::Type, int32_t>) {
                    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor,
                                                     -1, -1>;
                    using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor,
                                                      -1, -1>;
                    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW,
                                                     pto::BLayout::RowMajor, -1, -1>;

                    Fp32TileDefine tmp0Fp32Tile(dstShape3, dstShape4);
                    Fp32TileDefine tmp2Fp32Tile(dstShape3, dstShape4);
                    Int32TileDefine tmp0I32Tile(dstShape3, dstShape4);
                    Int32TileDefine tmp2I32Tile(dstShape3, dstShape4);
                    Int32TileDefine tmp3I32Tile(dstShape3, dstShape4);
                    Int32TileDefine tmp4I32Tile(dstShape3, dstShape4);
                    Int32TileDefine tmp5I32Tile(dstShape3, dstShape4);
                    MaskTileDefine tmp1MaskTile(dstShape3, dstShape4);

                    pto::TASSIGN(tmp0Fp32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(int32_t)));
                    pto::TASSIGN(tmp2Fp32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 2, sizeof(int32_t)));
                    pto::TASSIGN(tmp0I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(int32_t)));
                    pto::TASSIGN(tmp2I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 2, sizeof(int32_t)));
                    pto::TASSIGN(tmp3I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 3, sizeof(int32_t)));
                    pto::TASSIGN(tmp4I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 4, sizeof(int32_t)));
                    pto::TASSIGN(tmp5I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 5, sizeof(int32_t)));
                    pto::TASSIGN(tmp1MaskTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(int32_t)));
                    auto divisor = static_cast<int32_t>(src1);

                    // Step 1: approximate quotient by float32 division, then floor and cast to int32.
                    // q = floor(float32(x1) / float32(x2))
                    pto::TCVT(tmp0Fp32Tile, src0Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    SyncV();
                    pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Fp32Tile, tmp0Fp32Tile,
                                                                  static_cast<float>(divisor));
                    SyncV();
                    pto::TCVT(dstTile, tmp0Fp32Tile, pto::RoundMode::CAST_FLOOR);
                    SyncV();

                    // Step 2: compute exact int32 remainder: r = x1 - q * x2.
                    pto::TMULS(tmp0I32Tile, dstTile, divisor);
                    SyncV();
                    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile);
                    SyncV();

                    // Step 3: refine q with floor(float32(r) / float32(x2)).
                    pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Fp32Tile, tmp2Fp32Tile,
                                                                  static_cast<float>(divisor));
                    SyncV();
                    pto::TCVT(tmp0I32Tile, tmp2Fp32Tile, pto::RoundMode::CAST_FLOOR);
                    SyncV();

                    // Step 4: apply the remainder-based correction.
                    // q_corrected = q + correction
                    pto::TADD(dstTile, dstTile, tmp0I32Tile);
                    SyncV();

                    // Step 5: recompute r2 with q_corrected.
                    pto::TMULS(tmp0I32Tile, dstTile, divisor);
                    SyncV();
                    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile); // r2
                    SyncV();

                    // Step 6: final +/-1 correction. A valid floor-div remainder must satisfy
                    // 0 <= r2 * sign(x2) < abs(x2).
                    auto absSrc1 = divisor;
                    if (divisor < 0) {
                        pto::TMULS(tmp0I32Tile, tmp0I32Tile, -1); // r2_adj = -r2
                        SyncV();
                        absSrc1 = -divisor;
                    }

                    pto::TADDS(tmp3I32Tile, tmp0I32Tile, -absSrc1); // diff = r2_adj - abs(x2)
                    SyncV();

                    // Build tensor constants and use TSEL instead of TSELS to avoid the A2/A3
                    // tensor-scalar select path, whose first lane can be unstable across calls.
                    pto::TSUB(tmp4I32Tile, tmp0I32Tile, tmp0I32Tile); // zero
                    SyncV();

                    // If r2_adj < 0, q_corrected is too large: final_corr = -1; otherwise 0.
                    pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    SyncV();
                    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::LT);
                    SyncV();
                    pto::TADDS(tmp2I32Tile, tmp4I32Tile, -1);
                    SyncV();
                    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp4I32Tile, tmp5I32Tile);
                    SyncV();

                    // If diff >= 0, q_corrected is too small: final_corr = 1.
                    pto::TCVT(tmp2Fp32Tile, tmp3I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    SyncV();
                    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::GE);
                    SyncV();
                    pto::TADDS(tmp2I32Tile, tmp4I32Tile, 1);
                    SyncV();
                    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp0I32Tile, tmp5I32Tile);
                    SyncV();

                    // res = q_corrected + final_corr
                    pto::TADD(dstTile, dstTile, tmp0I32Tile);
                    SyncV();
                } else if constexpr (std::is_same_v<typename T0::Type, int8_t> ||
                                     std::is_same_v<typename T0::Type, uint8_t>) {
                    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1,
                                                     -1>;
                    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor,
                                                     -1, -1>;
                    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
                    Fp32TileDefine tmp1Tile(dstShape3, dstShape4);
                    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
                    pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));
                    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(tmp1Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp1Tile, tmp1Tile, static_cast<float>(src1));
                    SyncV();
                    pto::TCVT(tmp0Tile, tmp1Tile, pto::RoundMode::CAST_FLOOR);
                    SyncV();
                    pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_FLOOR, pto::SaturationMode::ON);
                    SyncV();
                }
#else
                if constexpr (std::is_same_v<typename T0::Type, uint8_t>) {
                    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1,
                                                     -1>;
                    using Int16TileDefine = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW, pto::BLayout::RowMajor,
                                                      -1, -1>;
                    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
                    Int16TileDefine tmp1Tile(dstShape3, dstShape4);
                    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
                    pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));
                    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
                    pto::TCVT(tmp1Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
                    pto::TDIVS(tmp1Tile, tmp1Tile, static_cast<int16_t>(src1));
                    pto::TCVT(dstTile, tmp1Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::ON);
                } else if constexpr (std::is_same_v<typename T0::Type, int8_t>) {
                    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1,
                                                     -1>;
                    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
                    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
                    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
                    if (src1 == 0) {
                        pto::TEXPANDS(tmp0Tile, static_cast<half>(0.0f));
                    } else {
                        pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Tile, tmp0Tile,
                                                                      static_cast<half>(static_cast<float>(src1)));
                    }
                    pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_FLOOR);
                } else if constexpr (std::is_same_v<typename T0::Type, int32_t>) {
                    using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor,
                                                      -1, -1>;
                    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW,
                                                     pto::BLayout::RowMajor, -1, -1>;
                    Int32TileDefine tmp0DataTile(dstShape3, dstShape4);
                    Int32TileDefine tmp1DataTile(dstShape3, dstShape4);
                    MaskTileDefine tmp2MaskTile(dstShape3, dstShape4);
                    MaskTileDefine tmp3MaskTile(dstShape3, dstShape4);
                    pto::TASSIGN(tmp0DataTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
                    pto::TASSIGN(tmp1DataTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));
                    pto::TASSIGN(tmp2MaskTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 2, sizeof(float)));
                    pto::TASSIGN(tmp3MaskTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));

                    if (src1 == 0) {
                        constexpr int32_t pos = 0x7FFF7F7F;
                        constexpr int32_t neg = 0x80008080;
                        pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::LT);
                        pto::TSELS(dstTile, tmp2MaskTile, dstTile, tmp0DataTile, pos);
                        pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::GE);
                        pto::TSELS(dstTile, tmp2MaskTile, dstTile, tmp0DataTile, neg);
                    } else {
                        if (src1 < 0) {
                            pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::GE);
                        } else {
                            pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::LT);
                        }
                        pto::TDIVS(dstTile, src0Tile, static_cast<int32_t>(src1));
                        pto::TMULS(tmp0DataTile, dstTile, static_cast<int32_t>(src1));
                        pto::TSUB(tmp0DataTile, src0Tile, tmp0DataTile);
                        pto::TCMPS(tmp3MaskTile, tmp0DataTile, 0, pto::CmpMode::NE);
                        pto::TAND(tmp2MaskTile, tmp2MaskTile, tmp3MaskTile);
                        pto::TADDS(tmp1DataTile, dstTile, -1);
                        pto::TSEL(dstTile, tmp2MaskTile, tmp1DataTile, dstTile, tmp0DataTile);
                    }
                }
#endif
            }
        }
    }
}
#endif
