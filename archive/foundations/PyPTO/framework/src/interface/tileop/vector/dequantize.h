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
 * \file dequantize.h
 * \brief INT8/INT16 反量化 Tile 算子
 *
 * - INT8 -> FP32:  int8 -> half -> float, 然后 (src - offset) * scale
 * - INT16 -> FP32: int16 -> float, 然后 (src - offset) * scale
 *
 * 只支持逐行反量化 (axis=-1)，逐列反量化在 Operation 层通过 Transpose 实现
 */

#ifndef TILEOP_TILE_OPERATOR_DEQUANTIZE__H
#define TILEOP_TILE_OPERATOR_DEQUANTIZE__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

namespace pto {
enum class DequantType {
    INT8 = 0, // INT8 -> FP32
    INT16 = 1 // INT16 -> FP32
};
} // namespace pto

/// 向上取整到对齐边界
#ifndef PTO_CEIL
#define PTO_CEIL(x, y) ((((x) + (y) - 1) / (y)) * (y))
#endif

// =============================================================================
// INT8 反量化 (INT8 -> FP32)
// =============================================================================
#define OP_TILE_OP_TDEQUANT_INT8 TDequantInt8

/**
 * @brief INT8 反量化 (逐行)
 * @param dst     输出 FP32 张量, 形状 [..., H, W]
 * @param src     输入 INT8 张量, 形状 [..., H, W]
 * @param scale   缩放因子, 形状 [..., H]
 * @param offset  零点偏移, 形状 [..., H] (对称量化时传全0)
 *
 * 公式: dst = (src - offset) * scale
 * 转换路径: int8 -> half -> float
 */
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TDequantInt8(T0 dst, T1 src, T2 scale, T3 offset)
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
    // dst: FP32, 需要 32 字节对齐
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr int paddedCol_dst = PTO_CEIL(dstTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // src: INT8, 需要 32 字节对齐
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr int paddedCol_src = PTO_CEIL(srcTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(int8_t)));

    // scale/offset: FP32, ColMajor 布局
    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, 4, expectSize>();
    constexpr int paddedRow_scale = PTO_CEIL(scaleTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));
    constexpr auto offsetTileW = TileOp::GetTensorTileShapeDim<T3, 4, expectSize>();
    constexpr int paddedRow_offset = PTO_CEIL(offsetTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // 数据类型
    using DstDtype = typename T0::Type;    // float
    using SrcDtype = typename T1::Type;    // int8_t
    using ScaleDtype = typename T2::Type;  // float
    using OffsetDtype = typename T3::Type; // float

    // 定义 Tile 类型
    using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, paddedCol_dst, pto::BLayout::RowMajor, -1,
                                    -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, paddedCol_src, pto::BLayout::RowMajor, -1,
                                    -1>;
    using ScaleTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, 1, pto::BLayout::ColMajor, -1,
                                      -1>;
    using OffsetTileDefine = pto::Tile<pto::TileType::Vec, OffsetDtype, paddedRow_offset, 1, pto::BLayout::ColMajor, -1,
                                       -1>;

    // 遍历所有 Tile
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                ScaleTileDefine scaleTile(srcShape3, 1);
                OffsetTileDefine offsetTile(srcShape3, 1);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                auto scaleOffset = n0Index * scaleStride1 + n1Index * scaleStride2 + n2Index * scaleStride3;
                auto offsetOffset = n0Index * offsetStride1 + n1Index * offsetStride2 + n2Index * offsetStride3;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));

                pto::TDEQUANT(dstTile, srcTile, scaleTile, offsetTile);
            }
        }
    }
}

// =============================================================================
// INT16 反量化 (INT16 -> FP32)
// =============================================================================
#define OP_TILE_OP_TDEQUANT_INT16 TDequantInt16

/**
 * @brief INT16 反量化 (逐行)
 * @param dst     输出 FP32 张量, 形状 [..., H, W]
 * @param src     输入 INT16 张量, 形状 [..., H, W]
 * @param scale   缩放因子, 形状 [..., H]
 * @param offset  零点偏移, 形状 [..., H] (对称量化时传全0)
 *
 * 公式: dst = (src - offset) * scale
 * 转换路径: int16 -> float
 */
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TDequantInt16(T0 dst, T1 src, T2 scale, T3 offset)
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
    // dst: FP32
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr int paddedCol_dst = PTO_CEIL(dstTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // src: INT16
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr int paddedCol_src = PTO_CEIL(srcTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(int16_t)));

    // scale/offset: FP32, ColMajor
    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, 4, expectSize>();
    constexpr int paddedRow_scale = PTO_CEIL(scaleTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));
    constexpr auto offsetTileW = TileOp::GetTensorTileShapeDim<T3, 4, expectSize>();
    constexpr int paddedRow_offset = PTO_CEIL(offsetTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // 数据类型
    using DstDtype = typename T0::Type;    // float
    using SrcDtype = typename T1::Type;    // int16_t
    using ScaleDtype = typename T2::Type;  // float
    using OffsetDtype = typename T3::Type; // float

    // 定义 Tile 类型
    using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, paddedCol_dst, pto::BLayout::RowMajor, -1,
                                    -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, paddedCol_src, pto::BLayout::RowMajor, -1,
                                    -1>;
    using ScaleTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, 1, pto::BLayout::ColMajor, -1,
                                      -1>;
    using OffsetTileDefine = pto::Tile<pto::TileType::Vec, OffsetDtype, paddedRow_offset, 1, pto::BLayout::ColMajor, -1,
                                       -1>;

    // 遍历所有 Tile
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                ScaleTileDefine scaleTile(srcShape3, 1);
                OffsetTileDefine offsetTile(srcShape3, 1);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                auto scaleOffset = n0Index * scaleStride1 + n1Index * scaleStride2 + n2Index * scaleStride3;
                auto offsetOffset = n0Index * offsetStride1 + n1Index * offsetStride2 + n2Index * offsetStride3;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));

                pto::TDEQUANT(dstTile, srcTile, scaleTile, offsetTile);
            }
        }
    }
}

// =============================================================================
// 统一反量化接口
// =============================================================================
#define OP_TILE_OP_TDEQUANT TDequant

/**
 * @brief 统一反量化接口
 * @tparam dequantType INT8 或 INT16
 *
 * 注意: TDequant 总是需要 4 个参数 (dst, src, scale, offset)
 * 对称量化时，offset 传全 0 的张量
 */
template <pto::DequantType dequantType, typename T0, typename T1, typename T2, typename T3>
TILEOP void TDequant(T0 dst, T1 src, T2 scale, T3 offset)
{
    if constexpr (dequantType == pto::DequantType::INT8) {
        TDequantInt8(dst, src, scale, offset);
    } else if constexpr (dequantType == pto::DequantType::INT16) {
        TDequantInt16(dst, src, scale, offset);
    } else {
        static_assert(dequantType == pto::DequantType::INT8 || dequantType == pto::DequantType::INT16,
                      "TDequant only supports INT8 or INT16 type.");
    }
}

#endif // TILEOP_TILE_OPERATOR_DEQUANTIZE__H
