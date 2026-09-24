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
 * \file mmad_impl.h
 * \brief General Matrix Multiplication Interface Implementation (Atlas A3, Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_MMAD_IMPL__H
#define TILEOP_TILE_OPERATOR_MMAD_IMPL__H

#include "cube_utils.h"

template <bool initMatrixC, TransMode transMode, bool kAlignFlag, bool isGemv = false, typename TileAcc,
          typename TileLeft, typename TileRight>
INLINE void TMatmulImpl(TileAcc& c, TileLeft& a, TileRight& b)
{
    int64_t validM = GetShape<0>(a);
    int64_t validN = GetShape<1>(b);
    int64_t validK = GetShape<1>(a);
    // validK=0场景特殊处理：
    // AMULB需要做MAD产生全0矩阵，AMULACCB跳过MAD
    if (validM == 0 || validN == 0 || (validK == 0 && !initMatrixC)) {
        return;
    }
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename TileAcc::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename TileAcc::TileShape>::type::value;
    // validK=0且A_MUL_B模式：使用静态K大小，配合全零L1数据产生正确输出
    if (validK == 0 && initMatrixC) {
        validK = staticL0AW;
    }
    using tileL0ATensor = pto::TileLeft<typename TileLeft::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename TileRight::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename TileAcc::Type, staticL0CH, staticL0CW, -1, -1>;
    if constexpr (isGemv) {
        validM = 1; // GEMV M=1，不打 pad 到 16
    } else {
        validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    }
    tileL0ATensor l0a(validM, validK);
    tileL0BTensor l0b(validK, validN);
    tileL0CTensor l0c(validM, validN);
    if constexpr (std::is_same<typename tileL0ATensor::DType, float>::value) {
        l0a.ResetMadMode();
        l0a.SetKAligned(kAlignFlag);
    }
    if constexpr (transMode != TransMode::CAST_NONE) {
        l0a.SetMadTF32Mode(static_cast<pto::RoundMode>(transMode));
    }
    pto::TASSIGN(l0a, static_cast<uint64_t>(a.GetAddr()));
    pto::TASSIGN(l0b, static_cast<uint64_t>(b.GetAddr()));
    pto::TASSIGN(l0c, static_cast<uint64_t>(c.GetAddr()));
    if constexpr (initMatrixC) {
        if constexpr (isGemv) {
            pto::TGEMV(l0c, l0a, l0b);
        } else {
            pto::TMATMUL(l0c, l0a, l0b);
        }
    } else {
        if constexpr (isGemv) {
            pto::TGEMV_ACC(l0c, l0c, l0a, l0b);
        } else {
            pto::TMATMUL_ACC(l0c, l0c, l0a, l0b);
        }
    }
    if constexpr (transMode != TransMode::CAST_NONE) {
        l0a.ResetMadMode();
    }
}

template <TransMode transMode, bool kAlignFlag, bool isGemv = false, typename TileAcc, typename TileLeft,
          typename TileRight, typename TileBias>
INLINE void TMatmulImpl(TileAcc& c, TileLeft& a, TileRight& b, TileBias& bias)
{
    int64_t validM = GetShape<0>(a);
    int64_t validN = GetShape<1>(b);
    int64_t validK = GetShape<1>(a);
    if (validM == 0 || validN == 0 || validK == 0) {
        return;
    }
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename TileAcc::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename TileAcc::TileShape>::type::value;
    using tileL0ATensor = pto::TileLeft<typename TileLeft::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename TileRight::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename TileAcc::Type, staticL0CH, staticL0CW, -1, -1>;
    using tileBiasTensor = pto::Tile<pto::TileType::Bias, typename TileBias::Type, 1, staticL0BW,
                                     pto::BLayout::RowMajor, -1, -1>;
    if constexpr (isGemv) {
        validM = 1; // GEMV M=1，不打 pad 到 16
    } else {
        validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    }
    tileL0ATensor l0a(validM, validK);
    tileL0BTensor l0b(validK, validN);
    tileL0CTensor l0c(validM, validN);
    tileBiasTensor biasT(1, validN);
    if constexpr (std::is_same<typename tileL0ATensor::DType, float>::value) {
        l0a.SetKAligned(kAlignFlag);
        l0a.ResetMadMode();
    }
    if constexpr (transMode != TransMode::CAST_NONE) {
        l0a.SetMadTF32Mode(static_cast<pto::RoundMode>(transMode));
    }
    pto::TASSIGN(l0a, static_cast<uint64_t>(a.GetAddr()));
    pto::TASSIGN(l0b, static_cast<uint64_t>(b.GetAddr()));
    pto::TASSIGN(l0c, static_cast<uint64_t>(c.GetAddr()));
    pto::TASSIGN(biasT, static_cast<uint64_t>(bias.GetAddr()));
    if constexpr (isGemv) {
        pto::TGEMV_BIAS(l0c, l0a, l0b, biasT);
    } else {
        pto::TMATMUL_BIAS(l0c, l0a, l0b, biasT);
    }
    if constexpr (transMode != TransMode::CAST_NONE) {
        l0a.ResetMadMode();
    }
}

#endif // TILEOP_TILE_OPERATOR_MMAD_IMPL__H
