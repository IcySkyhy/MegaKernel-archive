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
 * \file mmad_mx_impl.h
 * \brief MX Matmul Interface Implementation (Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_ARCH35_MMAD_MX_IMPL__H
#define TILEOP_TILE_OPERATOR_ARCH35_MMAD_MX_IMPL__H

#include "../cube_utils.h"

namespace {
template <typename TileL0C, typename TileL0A, typename TileL0AScale, typename TileL0B, typename TileL0BScale,
          typename TileC, typename TileA, typename TileAScale, typename TileB, typename TileBScale>
INLINE void AssignTensorAddresses(TileL0C& l0c, TileL0A& l0a, TileL0AScale& l0aScale, TileL0B& l0b,
                                  TileL0BScale& l0bScale, TileC& c, TileA& a, TileAScale& aScale, TileB& b,
                                  TileBScale& bScale)
{
    pto::TASSIGN(l0a, static_cast<uint64_t>(a.GetAddr()));
    pto::TASSIGN(l0aScale, static_cast<uint64_t>(aScale.GetAddr()));
    pto::TASSIGN(l0b, static_cast<uint64_t>(b.GetAddr()));
    pto::TASSIGN(l0bScale, static_cast<uint64_t>(bScale.GetAddr()));
    pto::TASSIGN(l0c, static_cast<uint64_t>(c.GetAddr()));
}
} // namespace

template <bool initMatrixC, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale>
INLINE void MatmulMXImpl(TileRes& c, TileLeft& a, TileLeftScale& aScale, TileRight& b, TileRightScale& bScale)
{
    int64_t validM = GetShape<0>(a);
    int64_t validK = GetShape<1>(a);
    int64_t validN = GetShape<1>(b);
    int64_t validScaleK = GetShape<1>(aScale) * SHAPE_DIM2;
    // validK=0场景特殊处理：
    // AMULB需要做MAD产生全0矩阵，AMULACCB跳过MAD
    if (validM == 0 || validN == 0 || (validK == 0 && !initMatrixC)) {
        return;
    }
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileRes::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeBScale = Std::tuple_size<typename TileRightScale::Shape>::value;
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeAScale = Std::tuple_size<typename TileLeftScale::Shape>::value;
    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0AScaleW = Std::tuple_element<shapeSizeAScale - SHAPE_DIM2,
                                                        typename TileLeftScale::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0BScaleH = Std::tuple_element<shapeSizeBScale - SHAPE_DIM3,
                                                        typename TileRightScale::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename TileRes::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename TileRes::TileShape>::type::value;
    // validK=0或者validScaleK=0且A_MUL_B模式：使用静态K大小，配合全零L1数据产生正确输出
    if (validK == 0 && initMatrixC) {
        validK = staticL0AW;
        validScaleK = staticL0AScaleW * SHAPE_DIM2;
    }
    using tileL0CTensor = pto::TileAcc<typename TileRes::Type, staticL0CH, staticL0CW, -1, -1>;
    using tileL0ATensor = pto::TileLeft<typename TileLeft::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0AScaleTensor = pto::TileLeftScale<typename TileLeft::Type, staticL0AH, staticL0AScaleW * SHAPE_DIM2, -1,
                                                  -1>;
    using tileL0BTensor = pto::TileRight<typename TileRight::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0BScaleTensor = pto::TileRightScale<typename TileRight::Type, staticL0BScaleH * SHAPE_DIM2, staticL0BW,
                                                   -1, -1>;
    validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    validK = (validK + MX_BLOCK_ALIGN_BYTE - 1) / MX_BLOCK_ALIGN_BYTE * MX_BLOCK_ALIGN_BYTE;
    tileL0ATensor l0a(validM, validK);
    tileL0AScaleTensor l0aScale(validM, validScaleK);
    tileL0BTensor l0b(validK, validN);
    tileL0BScaleTensor l0bScale(validScaleK, validN);
    tileL0CTensor l0c(validM, validN);
    AssignTensorAddresses(l0c, l0a, l0aScale, l0b, l0bScale, c, a, aScale, b, bScale);
#ifndef __LITE_NPU
    if constexpr (initMatrixC) {
        pto::TMATMUL_MX(l0c, l0a, l0aScale, l0b, l0bScale);
    } else {
        pto::TMATMUL_MX(l0c, l0c, l0a, l0aScale, l0b, l0bScale);
    }
#endif
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias>
INLINE void MatmulMXImpl(TileRes& c, TileLeft& a, TileLeftScale& aScale, TileRight& b, TileRightScale& bScale,
                         TileBias& bias)
{
    int64_t validM = GetShape<0>(a);
    int64_t validK = GetShape<1>(a);
    int64_t validScaleK = GetShape<1>(aScale) * SHAPE_DIM2;
    int64_t validN = GetShape<1>(b);
    if (validM == 0 || validN == 0 || validK == 0) {
        return;
    }
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileRes::Shape>::value;
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeAScale = Std::tuple_size<typename TileLeftScale::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeBScale = Std::tuple_size<typename TileRightScale::Shape>::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename TileRes::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename TileRes::TileShape>::type::value;
    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0AScaleW = Std::tuple_element<shapeSizeAScale - SHAPE_DIM2,
                                                        typename TileLeftScale::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0BScaleH = Std::tuple_element<shapeSizeBScale - SHAPE_DIM3,
                                                        typename TileRightScale::TileShape>::type::value;
    using tileL0CTensor = pto::TileAcc<typename TileRes::Type, staticL0CH, staticL0CW, -1, -1>;
    using tileL0ATensor = pto::TileLeft<typename TileLeft::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0AScaleTensor = pto::TileLeftScale<typename TileLeft::Type, staticL0AH, staticL0AScaleW * SHAPE_DIM2, -1,
                                                  -1>;
    using tileL0BTensor = pto::TileRight<typename TileRight::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0BScaleTensor = pto::TileRightScale<typename TileRight::Type, staticL0BScaleH * SHAPE_DIM2, staticL0BW,
                                                   -1, -1>;
    using tileBiasTensor = pto::Tile<pto::TileType::Bias, typename TileBias::Type, 1, staticL0BW,
                                     pto::BLayout::RowMajor, -1, -1>;
    validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    validK = (validK + MX_BLOCK_ALIGN_BYTE - 1) / MX_BLOCK_ALIGN_BYTE * MX_BLOCK_ALIGN_BYTE;
    tileL0ATensor l0a(validM, validK);
    tileL0AScaleTensor l0aScale(validM, validScaleK);
    tileL0BTensor l0b(validK, validN);
    tileL0BScaleTensor l0bScale(validScaleK, validN);
    tileL0CTensor l0c(validM, validN);
    tileBiasTensor biasT(1, validM);
    AssignTensorAddresses(l0c, l0a, l0aScale, l0b, l0bScale, c, a, aScale, b, bScale);
    pto::TASSIGN(biasT, static_cast<uint64_t>(bias.GetAddr()));
#ifndef __LITE_NPU
    pto::TMATMUL_MX(l0c, l0a, l0aScale, l0b, l0bScale, biasT);
#endif
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_MMAD_MX_IMPL__H
