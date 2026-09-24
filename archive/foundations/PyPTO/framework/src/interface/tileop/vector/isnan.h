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
 * \file isnan.h
 * \brief isnan operator: identify NaN, NaN returns true, otherwise false
 */

#ifndef TILEOP_TILE_OPERATOR_ISNAN__H
#define TILEOP_TILE_OPERATOR_ISNAN__H
#include "pto_tile.h"
#include "tileop_common.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <type_traits>

#define OP_TILE_OP_ISNAN TIsNan

template <bool ResultAliasesCompare, typename TDst, typename TCompare, typename TMask, typename TResult,
          typename TScalarTmp>
TILEOP void TIsNanCompareImpl(TDst dstTile, TCompare compareTile, TMask maskTile, TResult resultTile,
                              TScalarTmp scalarTmpTile)
{
    // EQ is false only for NaN. All 0/1 materialization occurs in tmp.
    pto::TCMP(maskTile, compareTile, compareTile, pto::CmpMode::EQ);
    if constexpr (ResultAliasesCompare) {
        SyncV();
    }
    pto::TEXPANDS(resultTile, static_cast<half>(0.0f));
    SyncV();
    pto::TSELS(resultTile, maskTile, resultTile, scalarTmpTile, static_cast<half>(1.0f));
    SyncV();
    pto::TCVT(dstTile, resultTile, pto::RoundMode::CAST_NONE);
    SyncV();
}

template <typename TDst, typename TSrc, typename TCast, typename TMask, typename TResult, typename TScalarTmp>
TILEOP void TIsNanTileImpl(TDst dstTile, TSrc srcTile, TCast castTile, TMask maskTile, TResult resultTile,
                           TScalarTmp scalarTmpTile)
{
    if constexpr (std::is_same_v<typename TSrc::DType, bfloat16_t>) {
        // The original srcTile is read-only; castTile is backed by tmp.
        pto::TCVT(castTile, srcTile, pto::RoundMode::CAST_NONE);
        SyncV();
        TIsNanCompareImpl<true>(dstTile, castTile, maskTile, resultTile, scalarTmpTile);
    } else {
        TIsNanCompareImpl<false>(dstTile, srcTile, maskTile, resultTile, scalarTmpTile);
    }
}

template <typename T0, typename T1, typename T2>
TILEOP void TIsNan(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto srcExecShape3 = GetElementwiseOperandExecShapeDim<DIM_4TH, MAX_DIMS>(dst, src);
    auto srcExecShape4 = GetElementwiseOperandExecShapeDim<DIM_5TH, MAX_DIMS>(dst, src);

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    using SrcExecConfig = ElementwiseOperandExecConfig<T0, T1>;
    constexpr auto srcTileH = SrcExecConfig::tileH;
    constexpr auto srcTileW = SrcExecConfig::tileW;
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);

    // Scratch tiles need their width aligned to 32 bytes (a pto::Tile requirement for UB buffers),
    // so use tmpTileW instead of the raw srcTileW.
    constexpr auto align32Half = TileOp::BLOCK_SIZE / sizeof(half);
    constexpr auto tmpTileW = (srcTileW + align32Half - 1) / align32Half * align32Half;

    // The operation layer always reserves three FP32-sized blocks plus one 32-byte scalar block,
    // laid out back to back. We mirror that fixed layout here regardless of dtype: the cast block is
    // simply left unused for fp16/fp32. Keeping the layout dtype-independent keeps both sides in sync.
    constexpr size_t blockBytes = srcTileH * tmpTileW * sizeof(float);
    constexpr size_t castOffset = 0;
    constexpr size_t maskOffset = blockBytes;
    constexpr size_t resultOffset = 2 * blockBytes;
    constexpr size_t scalarTmpOffset = 3 * blockBytes;

    using DstTile = pto::Tile<pto::TileType::Vec, uint8_t, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcTile = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1,
                              -1>;
    using CastTile = pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTile = pto::Tile<pto::TileType::Vec, uint8_t, srcTileH, tmpTileW * sizeof(float), pto::BLayout::RowMajor,
                               -1, -1>;
    using ResultTile = pto::Tile<pto::TileType::Vec, half, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
    using ScalarTmpTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, TileOp::BLOCK_SIZE, pto::BLayout::RowMajor, -1, -1>;

    DstTile dstTile(shape3, shape4);
    SrcTile srcTile(srcExecShape3, srcExecShape4);
    CastTile castTile(srcExecShape3, srcExecShape4);
    MaskTile maskTile(srcExecShape3, srcExecShape4);
    ResultTile resultTile(srcExecShape3, srcExecShape4);
    ScalarTmpTile scalarTmpTile(1, TileOp::BLOCK_SIZE);
    pto::TASSIGN(castTile, (uint64_t)(tmp.GetAddr() + castOffset));
    pto::TASSIGN(maskTile, (uint64_t)(tmp.GetAddr() + maskOffset));
    pto::TASSIGN(resultTile, (uint64_t)(tmp.GetAddr() + resultOffset));
    pto::TASSIGN(scalarTmpTile, (uint64_t)(tmp.GetAddr() + scalarTmpOffset));

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + GenTileOffset(dst, tileOffsets) * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + GenTileOffset(src, tileOffsets) * srcTypeSize));
                TIsNanTileImpl(dstTile, srcTile, castTile, maskTile, resultTile, scalarTmpTile);
            }
        }
    }
}

#endif
