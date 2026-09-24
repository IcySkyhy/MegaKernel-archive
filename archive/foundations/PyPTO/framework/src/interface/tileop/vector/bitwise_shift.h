/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bitwise_shift.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BITWISE_SHIFT__H
#define TILEOP_TILE_OPERATOR_BITWISE_SHIFT__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "tileop_common.h"
#include "unary.h"

template <BitwiseShiftOp op, typename T0, typename T1, typename T2>
TILEOP void BitwiseShiftComputeImpl(T0 dst, T1 src0, T2 src1)
{
    if constexpr (op == BitwiseShiftOp::BITWISERIGHTSHIFT) {
        pto::TSHR(dst, src0, src1);
        return;
    }

    if constexpr (op == BitwiseShiftOp::BITWISELEFTSHIFT) {
        pto::TSHL(dst, src0, src1);
        return;
    }
}

template <BitwiseShiftOp op, typename T0, typename T1, typename Scalar>
TILEOP void BitwiseShiftScalarComputeImpl(T0 dst, T1 src0, Scalar src1)
{
    if constexpr (op == BitwiseShiftOp::BITWISERIGHTSHIFT) {
        pto::TSHRS(dst, src0, src1);
        return;
    }

    if constexpr (op == BitwiseShiftOp::BITWISELEFTSHIFT) {
        pto::TSHLS(dst, src0, src1);
        return;
    }
}

template <size_t MAX_SHIFT_NUM, typename T, typename U, typename V, typename USigned, typename VSigned>
TILEOP void GetValidShiftTile(T& dst, U& src1, V& tmp, USigned& src1Signed, VSigned& tmpSigned)
{
    pto::TEXPANDS(tmp, MAX_SHIFT_NUM);
    SyncV();
    pto::TSUB(tmpSigned, tmpSigned, src1Signed);
    SyncV();
    pto::TOR(tmp, tmp, src1);
    SyncV();
    pto::TSHRS(tmpSigned, tmpSigned, MAX_SHIFT_NUM);
    SyncV();
    pto::TNOT(dst, tmp);
    SyncV();
    pto::TAND(src1, src1, dst);
    SyncV();
    pto::TEXPANDS(dst, MAX_SHIFT_NUM);
    SyncV();
    pto::TAND(tmp, tmp, dst);
    SyncV();
    pto::TOR(src1, src1, tmp);
}

template <BitwiseShiftOp op, size_t MAX_SHIFT_NUM, typename T0, typename T1, typename T2, typename T3,
          typename T2Signed, typename T3Signed>
TILEOP void BitwiseShiftImpl(T0& dst, T1& src0, T2& src1, T3& tmp, T2Signed& src1Signed, T3Signed& tmpSigned)
{
    GetValidShiftTile<MAX_SHIFT_NUM>(dst, src1, tmp, src1Signed, tmpSigned);
    SyncV();
    BitwiseShiftComputeImpl<op>(dst, src0, src1);
}

template <BitwiseShiftOp op, typename T0, typename T1, typename T2, typename T3>
TILEOP void BitwiseShiftCompute(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    constexpr auto MAX_SHIFT_NUM = sizeof(typename T0::Type) * TileOp::BLOCK_NELEM_B32;
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    auto src1ExecTile = MakeElementwiseOperandExecTile(dst, src1);
    auto tmpExecTile = MakeElementwiseOperandExecTile(dst, tmp);
    auto src1SignedExecTile = MakeElementwiseOperandExecTile<std::make_signed_t<typename PtoTile<T2>::Dtype>>(dst,
                                                                                                              src1);
    auto tmpSignedExecTile = MakeElementwiseOperandExecTile<std::make_signed_t<typename PtoTile<T3>::Dtype>>(dst, tmp);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                AssignElementwiseOperandExecTile(src1ExecTile, src1, tileOffsets);
                AssignElementwiseOperandExecTile(tmpExecTile, tmp, tileOffsets);
                AssignElementwiseOperandExecTile(src1SignedExecTile, src1, tileOffsets);
                AssignElementwiseOperandExecTile(tmpSignedExecTile, tmp, tileOffsets);
                BitwiseShiftImpl<op, MAX_SHIFT_NUM>(dstTile.Data(), src0ExecTile, src1ExecTile, tmpExecTile,
                                                    src1SignedExecTile, tmpSignedExecTile);
            }
        }
    }
}

template <BitwiseShiftOp op, typename T0, typename T1, typename Scalar>
TILEOP void BitwiseShiftScalarCompute(T0 dst, T1 src0, Scalar src1)
{
    constexpr auto MAX_SHIFT_NUM = sizeof(typename T0::Type) * TileOp::BLOCK_NELEM_B32;
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    if (src1 < 0 || src1 > MAX_SHIFT_NUM) {
        src1 = MAX_SHIFT_NUM;
    }
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                BitwiseShiftScalarComputeImpl<op>(dstTile.Data(), src0ExecTile, src1);
            }
        }
    }
}

template <BitwiseShiftOp op, size_t MAX_SHIFT_NUM, typename T0, typename Scalar, typename T1, typename T2,
          typename T1Signed, typename T2Signed>
TILEOP void ScalarBitwiseShiftImpl(T0& dst, Scalar& src0, T1& src1, T2& tmp, T1Signed& src1Signed, T2Signed& tmpSigned)
{
    GetValidShiftTile<MAX_SHIFT_NUM>(dst, src1, tmp, src1Signed, tmpSigned);
    SyncV();
    pto::TEXPANDS(dst, src0);
    SyncV();
    BitwiseShiftComputeImpl<op>(dst, dst, src1);
}

template <BitwiseShiftOp op, typename T0, typename Scalar, typename T1, typename T2>
TILEOP void ScalarBitwiseShiftCompute(T0 dst, Scalar src0, T1 src1, T2 tmp)
{
    constexpr auto MAX_SHIFT_NUM = sizeof(typename T0::Type) * TileOp::BLOCK_NELEM_B32;
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstTile = PtoTile<T0>(dst);
    auto src1ExecTile = MakeElementwiseOperandExecTile(dst, src1);
    auto tmpExecTile = MakeElementwiseOperandExecTile(dst, tmp);
    auto src1SignedExecTile = MakeElementwiseOperandExecTile<std::make_signed_t<typename PtoTile<T1>::Dtype>>(dst,
                                                                                                              src1);
    auto tmpSignedExecTile = MakeElementwiseOperandExecTile<std::make_signed_t<typename PtoTile<T2>::Dtype>>(dst, tmp);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(src1ExecTile, src1, tileOffsets);
                AssignElementwiseOperandExecTile(tmpExecTile, tmp, tileOffsets);
                AssignElementwiseOperandExecTile(src1SignedExecTile, src1, tileOffsets);
                AssignElementwiseOperandExecTile(tmpSignedExecTile, tmp, tileOffsets);
                ScalarBitwiseShiftImpl<op, MAX_SHIFT_NUM>(dstTile.Data(), src0, src1ExecTile, tmpExecTile,
                                                          src1SignedExecTile, tmpSignedExecTile);
            }
        }
    }
}

#define OP_TILE_OP_BITWISERIGHTSHIFT TBitrshift
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TBitrshift(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    BitwiseShiftCompute<BitwiseShiftOp::BITWISERIGHTSHIFT>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_BITWISELEFTSHIFT TBitlshift
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TBitlshift(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    BitwiseShiftCompute<BitwiseShiftOp::BITWISELEFTSHIFT>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_BITWISERIGHTSHIFTS TBitrshiftS
template <typename Scalar, typename T0, typename T1>
TILEOP void TBitrshiftS(T0 dst, T1 src0, Scalar src1)
{
    BitwiseShiftScalarCompute<BitwiseShiftOp::BITWISERIGHTSHIFT>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISELEFTSHIFTS TBitlshiftS
template <typename Scalar, typename T0, typename T1>
TILEOP void TBitlshiftS(T0 dst, T1 src0, Scalar src1)
{
    BitwiseShiftScalarCompute<BitwiseShiftOp::BITWISELEFTSHIFT>(dst, src0, src1);
}

#define OP_TILE_OP_SBITWISERIGHTSHIFT TSBitrshift
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TSBitrshift(T0 dst, T1 src1, Scalar src0, T2 tmp)
{
    ScalarBitwiseShiftCompute<BitwiseShiftOp::BITWISERIGHTSHIFT>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_SBITWISELEFTSHIFT TSBitlshift
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TSBitlshift(T0 dst, T1 src1, Scalar src0, T2 tmp)
{
    ScalarBitwiseShiftCompute<BitwiseShiftOp::BITWISELEFTSHIFT>(dst, src0, src1, tmp);
}
#endif
