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
 * \file expand_exp_dif.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_EXPAND_EXP_DIF__H
#define TILEOP_TILE_OPERATOR_EXPAND_EXP_DIF__H
#include "binary.h"
#include "unary.h"

#define OP_TILE_OP_EXPANDEXPDIF TExpandExpDif
template <int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TExpandExpDif(T0 dst, T1 src0, T2 src1)
{
    if constexpr (GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_NONE) {
        const auto src0Layout = src0.GetLayout();
        auto src0Shape3 = src0Layout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
        auto src0Shape4 = src0Layout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
        const auto src1Layout = src1.GetLayout();
        auto src1Shape3 = src1Layout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
        auto src1Shape4 = src1Layout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
        // (m, n) & (m, n) -> (m, n) use SUB + EXP
        if ((src0Shape3 == src1Shape3) && (src0Shape4 == src1Shape4)) {
            BinaryCompute<BinaryOp::SUB, 0, LastUse3Dim<0, 0, 0>, BrcOperands...>(dst, src0, src1);
#ifdef __DAV_V220
            pipe_barrier(PIPE_V);
#endif
            UnaryCompute<UnaryOp::EXP, pto::ExpAlgorithm::DEFAULT, LastUse2Dim<0, 0>>(dst, dst);
            return;
        }
    }
    BinaryCompute<BinaryOp::EXPANDEXPDIF, 0, LastUse3Dim<0, 0, 0>, BrcOperands...>(dst, src0, src1);
}

#endif // TILEOP_TILE_OPERATOR_EXPAND_EXP_DIF__H
