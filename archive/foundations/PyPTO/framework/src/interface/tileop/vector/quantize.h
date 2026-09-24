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
 * \file quantize.h
 * \brief INT8 对称/非对称量化 Tile 算子
 *
 * - INT8_SYM:  FP32 -> INT8,  范围 [-128, 127]
 * - INT8_ASYM: FP32 -> UINT8, 范围 [0, 255]
 *
 * 只支持逐行量化 (axis=-1)，逐列量化在 Operation 层通过 Transpose 实现
 */

#ifndef TILEOP_TILE_OPERATOR_QUANTIZE__H
#define TILEOP_TILE_OPERATOR_QUANTIZE__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <type_traits>

/// Tile 内存对齐字节数
constexpr size_t TILE_ALIGNMENT_BYTES = 32;

/// 向上取整到对齐边界
#ifndef PTO_CEIL
#define PTO_CEIL(x, y) ((((x) + (y) - 1) / (y)) * (y))
#endif

// =============================================================================
// INT8 对称量化
// =============================================================================
#define OP_TILE_OP_TQUANT_INT8_SYM TQuantInt8Sym

/**
 * @brief INT8 对称量化 (逐行)
 * @param dst   输出 INT8 张量, 形状 [..., H, W]
 * @param src   输入 FP32 张量, 形状 [..., H, W]
 * @param scale 缩放因子, 形状 [..., H]
 * @param tmp   临时缓冲区
 */
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuantInt8Sym(T0 dst, T1 src, T2 scale, T3 tmp)
{
    constexpr size_t expectSize = 5;

    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();

    // 获取形状
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    // 获取步长
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    auto scaleStride1 = scaleLayout.template GetStrideDim<1, expectSize>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<2, expectSize>();
    auto scaleStride3 = scaleLayout.template GetStrideDim<3, expectSize>();

    // 获取 Tile 形状并计算对齐
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr int paddedCol_dst = PTO_CEIL(dstTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(int8_t)));

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr int paddedCol_src = PTO_CEIL(srcTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));
    constexpr int paddedRow_src = PTO_CEIL(
        srcTileH, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float))); // vbrcb processes in groups of 8 rows

    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, 4, expectSize>();
    constexpr int paddedRow_scale = PTO_CEIL(scaleTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // 计算一个 tile 的大小（元素个数）
    constexpr auto tileShapeSize = TileOp::GetAnyAxisMergeResult<
        DIM_1ST, Std::tuple_size<typename T0::TileShape>::value, typename T0::TileShape>();

    // 数据类型
    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using ScaleDtype = typename T2::Type;
    using TmpDtype = int32_t;

    // 定义 Tile 类型
    using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, paddedCol_dst, pto::BLayout::RowMajor, -1,
                                    -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, paddedCol_src, pto::BLayout::RowMajor, -1,
                                    -1>;
    using ParaTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, 1, pto::BLayout::ColMajor, -1,
                                     -1>;
    // tmpbuf: aligned size with int32_t type
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, TmpDtype, paddedRow_src, paddedCol_src, pto::BLayout::RowMajor,
                                    -1, -1>;

    // 遍历所有 Tile
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                ParaTileDefine scaleTile(srcShape3, 1);
                TmpTileDefine tmpTile(srcShape3, srcShape4);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                // scaleOffset should need to be shifted
                auto scaleOffset = n0Index * scaleStride1 + n1Index * scaleStride2 + n2Index * scaleStride3;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                // tmpbuf 从单独的 tmp 参数获取地址
                pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));

                pto::TQUANT<pto::QuantType::INT8_SYM>(dstTile, srcTile, scaleTile, tmpTile);
            }
        }
    }
}

// =============================================================================
// INT8 非对称量化
// =============================================================================
#define OP_TILE_OP_TQUANT_INT8_ASYM TQuantInt8Asym

/**
 * @brief INT8 非对称量化 (逐行)
 * @param dst    输出 UINT8 张量, 形状 [..., H, W]
 * @param src    输入 FP32 张量, 形状 [..., H, W]
 * @param scale  缩放因子, 形状 [..., H]
 * @param offset 零点偏移, 形状 [..., H]
 * @param tmp    临时缓冲区
 */
template <typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TQuantInt8Asym(T0 dst, T1 src, T2 scale, T3 offset, T4 tmp)
{
    constexpr size_t expectSize = 5;

    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();
    const auto offsetLayout = offset.GetLayout();

    // 获取形状
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    // 获取步长
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    auto scaleStride1 = scaleLayout.template GetStrideDim<1, expectSize>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<2, expectSize>();
    auto scaleStride3 = scaleLayout.template GetStrideDim<3, expectSize>();

    auto offsetStride1 = offsetLayout.template GetStrideDim<1, expectSize>();
    auto offsetStride2 = offsetLayout.template GetStrideDim<2, expectSize>();
    auto offsetStride3 = offsetLayout.template GetStrideDim<3, expectSize>();

    // 获取 Tile 形状
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr int paddedCol_dst = PTO_CEIL(dstTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(int8_t)));

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr int paddedCol_src = PTO_CEIL(srcTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));
    constexpr int paddedRow_src = PTO_CEIL(
        srcTileH, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float))); // vbrcb processes in groups of 8 rows

    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, 4, expectSize>();
    constexpr int paddedRow_scale = PTO_CEIL(scaleTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));
    constexpr auto offsetTileW = TileOp::GetTensorTileShapeDim<T3, 4, expectSize>();
    constexpr int paddedRow_offset = PTO_CEIL(offsetTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // 计算一个 tile 的大小（元素个数）
    constexpr auto tileShapeSize = TileOp::GetAnyAxisMergeResult<
        DIM_1ST, Std::tuple_size<typename T0::TileShape>::value, typename T0::TileShape>();

    // 数据类型
    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using ScaleDtype = typename T2::Type;
    using OffsetDtype = typename T3::Type;
    using TmpDtype = int32_t;

    // 定义 Tile 类型
    using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
    using ParaTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, 1, pto::BLayout::ColMajor, -1,
                                     -1>;
    // tmpbuf: aligned size with int32_t type
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, TmpDtype, paddedRow_src, paddedCol_src, pto::BLayout::RowMajor,
                                    -1, -1>;

    // 遍历所有 Tile
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                ParaTileDefine scaleTile(srcShape3, 1);
                ParaTileDefine offsetTile(srcShape3, 1);
                TmpTileDefine tmpTile(srcShape3, srcShape4);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                // scaleOffset & offsetOffset should need to be shifted
                auto scaleOffset = n0Index * scaleStride1 + n1Index * scaleStride2 + n2Index * scaleStride3;
                auto offsetOffset = n0Index * offsetStride1 + n1Index * offsetStride2 + n2Index * offsetStride3;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));
                // tmpbuf 从单独的 tmp 参数获取地址
                pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));

                pto::TQUANT<pto::QuantType::INT8_ASYM>(dstTile, srcTile, scaleTile, tmpTile, &offsetTile);
            }
        }
    }
}

// =============================================================================
// 统一量化接口
// =============================================================================
#define OP_TILE_OP_TQUANT TQuant

/**
 * @brief 统一量化接口
 * @tparam quantType INT8_SYM 或 INT8_ASYM
 */
template <pto::QuantType quantType, typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuant(T0 dst, T1 src, T2 scale, T3 tmp)
{
    static_assert(quantType == pto::QuantType::INT8_SYM,
                  "TQuant with 4 parameters only supports INT8_SYM. "
                  "TQuant only supports INT8_SYM(4 parameters) and INT8_ASYM(5 parameters).");
    TQuantInt8Sym(dst, src, scale, tmp);
}

template <pto::QuantType quantType, typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TQuant(T0 dst, T1 src, T2 scale, T3 offset, T4 tmp)
{
    static_assert(quantType == pto::QuantType::INT8_ASYM,
                  "TQuant with 5 parameters only supports INT8_ASYM."
                  "TQuant only supports INT8_SYM(4 parameters) and INT8_ASYM(5 parameters).");
    TQuantInt8Asym(dst, src, scale, offset, tmp);
}

#if defined PTO_NPU_ARCH_A5
// =============================================================================
// MX 量化
// =============================================================================
#define OP_TILE_OP_QUANT_MX TQuantMX
constexpr int kDequantScaleRoundingModeRoundUp = 0;
constexpr int kDequantScaleRoundingModeRoundDown = 1;
constexpr int kQuantMXPerformanceModeOn = 1;

template <int DEQUANT_SCALE_ROUNDING_MODE, typename DstTile, typename SrcTile, typename ExpTile, typename MaxTile,
          typename ScalingTile>
__aicore__ inline void QuantMXDispatch(DstTile& dstTile, SrcTile& srcTile, ExpTile& expTile, MaxTile& maxTile,
                                       ScalingTile& scalingTile)
{
    if constexpr (std::is_same_v<typename DstTile::DType, float4_e2m1x2_t>) {
        if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
            pto::TQUANT<pto::QuantType::MXFP4_E2M1, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile,
                        pto::QuantScaleAlg::NV>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        } else {
            pto::TQUANT<pto::QuantType::MXFP4_E2M1, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile,
                        pto::QuantScaleAlg::OCP>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        }
    } else {
        if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
            pto::TQUANT<pto::QuantType::MXFP8, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile, pto::QuantScaleAlg::NV>(
                dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        } else {
            pto::TQUANT<pto::QuantType::MXFP8, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile,
                        pto::QuantScaleAlg::OCP>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        }
    }
}

template <int DEQUANT_SCALE_ROUNDING_MODE, typename DstTile, typename SrcTile, typename ExpTile, typename MaxTile,
          typename ScalingTile>
__aicore__ inline void QuantMXDispatchDn(DstTile& dstTile, SrcTile& srcTile, ExpTile& expTile, MaxTile& maxTile,
                                         ScalingTile& scalingTile)
{
    if constexpr (std::is_same_v<typename DstTile::DType, float4_e2m1x2_t>) {
        if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
            pto::TQUANT<0, pto::MxQuantAlg::NvMxFp4E2M1, true>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        } else {
            pto::TQUANT<0, pto::MxQuantAlg::OcpMxFp4E2M1, true>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        }
    } else {
        if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
            pto::TQUANT<0, pto::MxQuantAlg::NvMxFp8E4M3, true>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        } else {
            pto::TQUANT<0, pto::MxQuantAlg::OcpMxFp8E4M3, true>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        }
    }
}

template <typename T, typename Layout>
__aicore__ inline size_t GetQuantMXPerformanceGroupedOffset(const Layout& layout, LoopVar n0Index, LoopVar n1Index,
                                                            LoopVar n2Index)
{
    (void)n0Index;
    constexpr auto srcRank = Std::tuple_size<typename T::Shape>::value;
    static_assert(srcRank >= 1 && srcRank <= 4, "TQuantMX only supports 1D to 4D input.");
    if constexpr (srcRank <= 2) {
        return 0;
    } else if constexpr (srcRank == 3) {
        return n2Index * layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    } else {
        return n1Index * layout.template GetStrideDim<DIM_3RD, MAX_DIMS>() +
               n2Index * layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    }
}

template <int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, typename T0, typename T1,
          typename T2, typename T3, typename T4>
TILEOP void TQuantMXGeneral(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    (void)AXIS;
    const auto dstLayout = dst.GetLayout();
    const auto expLayout = exp.GetLayout();
    const auto maxLayout = maxScratch.GetLayout();
    const auto scalingLayout = scalingScratch.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto expStride0 = expLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto expStride1 = expLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto expStride2 = expLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto expTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto expTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    using ExpByteTile = pto::Tile<pto::TileType::Vec, uint8_t, expTileH, expTileW, pto::BLayout::RowMajor, -1, -1>;

    auto dstTile = PtoTile<T0>(dst);
    auto maxTile = PtoTile<T2>(maxScratch);
    auto scalingTile = PtoTile<T3>(scalingScratch);
    auto srcTile = PtoTile<T4>(src);
    ExpByteTile expByteTile(expLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(),
                            expLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());

    (void)srcLayout;
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto expTileOffset = n0Index * expStride0 + n1Index * expStride1 + n2Index * expStride2;
                auto maxTileOffset = GetQuantMXPerformanceGroupedOffset<T4>(maxLayout, n0Index, n1Index, n2Index);
                auto scalingTileOffset = GetQuantMXPerformanceGroupedOffset<T4>(scalingLayout, n0Index, n1Index,
                                                                                n2Index);
                auto srcTileAddr = (uint64_t)(src.GetAddr() +
                                              GenTileOffset(src, tileOffsets) * sizeof(typename T4::Type));
                dstTile.Assign(dst, tileOffsets);
                maxTile.Assign((uint64_t)(maxScratch.GetAddr() + maxTileOffset * sizeof(typename T2::Type)));
                scalingTile.Assign(
                    (uint64_t)(scalingScratch.GetAddr() + scalingTileOffset * sizeof(typename T3::Type)));
                srcTile.Assign(srcTileAddr);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                              DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
                    QuantMXDispatch<DEQUANT_SCALE_ROUNDING_MODE>(dstTile.Data(), srcTile.Data(), expByteTile,
                                                                 maxTile.Data(), scalingTile.Data());
                } else {
                    static_assert(DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                                      DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp,
                                  "TQuantMX only supports ROUND_DOWN (OCP) and ROUND_UP (NV) modes currently.");
                }
            }
        }
    }
}

template <int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, typename T0, typename T1,
          typename T2, typename T3, typename T4>
TILEOP void TQuantMXDn(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    (void)AXIS;
    const auto dstLayout = dst.GetLayout();
    const auto expLayout = exp.GetLayout();
    const auto maxLayout = maxScratch.GetLayout();
    const auto scalingLayout = scalingScratch.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto expStride0 = expLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto expStride1 = expLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto expStride2 = expLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto expTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto expTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    using ExpByteTile = pto::Tile<pto::TileType::Vec, uint8_t, expTileH, expTileW, pto::BLayout::RowMajor, -1, -1>;

    auto dstTile = PtoTile<T0>(dst);
    auto maxTile = PtoTile<T2>(maxScratch);
    auto scalingTile = PtoTile<T3>(scalingScratch);
    auto srcTile = PtoTile<T4>(src);
    ExpByteTile expByteTile(expLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(),
                            expLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());

    (void)maxLayout;
    (void)scalingLayout;
    (void)srcLayout;
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto expTileOffset = n0Index * expStride0 + n1Index * expStride1 + n2Index * expStride2;
                auto srcTileAddr = (uint64_t)(src.GetAddr() +
                                              GenTileOffset(src, tileOffsets) * sizeof(typename T4::Type));
                dstTile.Assign(dst, tileOffsets);
                maxTile.Assign(maxScratch, tileOffsets);
                scalingTile.Assign(scalingScratch, tileOffsets);
                srcTile.Assign(srcTileAddr);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                              DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
                    QuantMXDispatchDn<DEQUANT_SCALE_ROUNDING_MODE>(dstTile.Data(), srcTile.Data(), expByteTile,
                                                                   maxTile.Data(), scalingTile.Data());
                } else {
                    static_assert(DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                                      DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp,
                                  "TQuantMX only supports ROUND_DOWN (OCP) and ROUND_UP (NV) modes currently.");
                }
            }
        }
    }
}

template <int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, typename T0, typename T1,
          typename T2, typename T3, typename T4>
TILEOP void TQuantMXPerformance(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    (void)AXIS;
    const auto dstLayout = dst.GetLayout();
    const auto expLayout = exp.GetLayout();
    const auto maxLayout = maxScratch.GetLayout();
    const auto scalingLayout = scalingScratch.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto scalingTile = PtoTile<T3>(scalingScratch);
    auto srcTile = PtoTile<T4>(src);
    constexpr int kMxQuantGroupSize = 32;
    constexpr int kExpTileAlignElems = 32;
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T4, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T4, DIM_5TH, MAX_DIMS>();
    constexpr auto expFlatTileW = srcTileH * ((srcTileW + kMxQuantGroupSize - 1) / kMxQuantGroupSize);
    constexpr auto expFlatTileWAligned = ((expFlatTileW + kExpTileAlignElems - 1) / kExpTileAlignElems) *
                                         kExpTileAlignElems;
    constexpr auto maxTileH = TileOp::GetTensorTileShapeDim<T2, DIM_4TH, MAX_DIMS>();
    constexpr auto maxTileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();
    using ExpByteTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, expFlatTileWAligned, pto::BLayout::RowMajor, -1, -1>;
    using MaxDtype = std::conditional_t<std::is_same_v<typename T2::Type, bool>, uint8_t, typename T2::Type>;
    using MaxTile = pto::Tile<pto::TileType::Vec, MaxDtype, maxTileH, maxTileW, pto::BLayout::RowMajor, -1, -1>;
    ExpByteTile expByteTile(
        1, srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>() *
               ((srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>() + kMxQuantGroupSize - 1) / kMxQuantGroupSize));
    MaxTile maxTile(maxLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(),
                    maxLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto expTileOffset = GetQuantMXPerformanceGroupedOffset<T4>(expLayout, n0Index, n1Index, n2Index);
                auto maxTileOffset = GetQuantMXPerformanceGroupedOffset<T4>(maxLayout, n0Index, n1Index, n2Index);
                auto scalingTileOffset = GetQuantMXPerformanceGroupedOffset<T4>(scalingLayout, n0Index, n1Index,
                                                                                n2Index);
                auto srcTileAddr = (uint64_t)(src.GetAddr() +
                                              GenTileOffset(src, tileOffsets) * sizeof(typename T4::Type));
                dstTile.Assign(dst, tileOffsets);
                scalingTile.Assign(
                    (uint64_t)(scalingScratch.GetAddr() + scalingTileOffset * sizeof(typename T3::Type)));
                srcTile.Assign(srcTileAddr);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                pto::TASSIGN(maxTile, (uint64_t)(maxScratch.GetAddr() + maxTileOffset * sizeof(typename T2::Type)));
                if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                              DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
                    QuantMXDispatch<DEQUANT_SCALE_ROUNDING_MODE>(dstTile.Data(), srcTile.Data(), expByteTile, maxTile,
                                                                 scalingTile.Data());
                } else {
                    static_assert(DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                                      DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp,
                                  "TQuantMX only supports ROUND_DOWN (OCP) and ROUND_UP (NV) modes currently.");
                }
            }
        }
    }
}

template <int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, int PERFORMANCE_MODE = 0,
          typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TQuantMX(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    constexpr auto srcRank = Std::tuple_size<typename T4::Shape>::value;
    if constexpr (srcRank >= 2 && AXIS == static_cast<int>(srcRank) - 2) {
        TQuantMXDn<DEQUANT_SCALE_ROUNDING_MODE, AXIS>(dst, exp, maxScratch, scalingScratch, src);
    } else if constexpr (PERFORMANCE_MODE == kQuantMXPerformanceModeOn) {
        TQuantMXPerformance<DEQUANT_SCALE_ROUNDING_MODE, AXIS>(dst, exp, maxScratch, scalingScratch, src);
    } else {
        TQuantMXGeneral<DEQUANT_SCALE_ROUNDING_MODE, AXIS>(dst, exp, maxScratch, scalingScratch, src);
    }
}
#endif // defined PTO_NPU_ARCH_A5
#endif // TILEOP_TILE_OPERATOR_QUANTIZE__H
