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
 * \file prelu.h
 * \brief
 */

#ifndef INTERFACE_TILEOP_VECTOR_PRELU_H_
#define INTERFACE_TILEOP_VECTOR_PRELU_H_

#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_PRELU TPrelu
template <unsigned axis, typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1, typename T2, typename T3>
TILEOP void TPRelu(T0 dst, T1 src, T2 weight, T3 tmp)
{
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    constexpr auto weightTypeSize = sizeof(typename T2::Type);

    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto srcShape0 = srcLayout.template GetShapeDim<0, expectSize>();
    auto srcShape1 = srcLayout.template GetShapeDim<1, expectSize>();
    auto srcShape2 = srcLayout.template GetShapeDim<2, expectSize>();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();

    if constexpr (axis == 5) {
        // For 1D input (N), weight is (1,)
        using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor, -1,
                                        -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW, pto::BLayout::RowMajor, -1,
                                        -1>;
        DstTileDefine dstTile(1, dstShape4);
        SrcTileDefine srcTile(1, srcShape4);
        auto weightAddr = (__ubuf__ typename T2::Type*)((uint64_t)(weight.GetAddr()));
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        auto negative_slope = *weightAddr;

        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr()));
        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr()));

        pto::TLRELU(dstTile, srcTile, negative_slope);
    } else if constexpr (axis == 4) {
        // For 2D input (N, C), weight is (C,)
        constexpr size_t ALIGN_SIZE = 32;
        constexpr size_t SIZEOFBYTE = 8;
        constexpr int64_t tmpSize = (srcTileW + SIZEOFBYTE - 1) / SIZEOFBYTE;
        constexpr auto tmpTileW = (tmpSize + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
        auto tmpShape = (srcShape4 + SIZEOFBYTE - 1) / SIZEOFBYTE;

        using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor, -1,
                                        -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW, pto::BLayout::RowMajor, -1,
                                        -1>;
        using TmpTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, 1, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
        DstTileDefine dstTile(1, dstShape4);
        SrcTileDefine srcTile(1, srcShape4);
        TmpTileDefine tmpTile(1, tmpShape);
        SrcTileDefine weightTile(1, srcShape4);
        pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
        pto::TASSIGN(weightTile, (uint64_t)(weight.GetAddr()));

        for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
            auto dstOffset = n3Index * dstStride3;
            auto srcOffset = n3Index * srcStride3;

            pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
            pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));

            pto::TPRELU(dstTile, srcTile, weightTile, tmpTile);
        }
    } else if constexpr (axis == 3) {
        // For 3D input (N, C, L), weight is (C,)
        using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor, -1,
                                        -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW, pto::BLayout::RowMajor, -1,
                                        -1>;
        DstTileDefine dstTile(1, dstShape4);
        SrcTileDefine srcTile(1, srcShape4);
        auto weightAddr = (__ubuf__ typename T2::Type*)((uint64_t)(weight.GetAddr()));
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
            auto negative_slope = *(weightAddr + n3Index);
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n2Index * dstStride2 + n3Index * dstStride3;
                auto srcOffset = n2Index * srcStride2 + n3Index * srcStride3;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));

                pto::TLRELU(dstTile, srcTile, negative_slope);
            }
        }
    } else if constexpr (axis == 2) {
        // For 4D input (N, C, H, W), weight is (C,)
        using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW,
                                        pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW,
                                        pto::BLayout::RowMajor, -1, -1>;

        DstTileDefine dstTile(dstShape3, dstShape4);
        SrcTileDefine srcTile(srcShape3, srcShape4);
        auto weightAddr = (__ubuf__ typename T2::Type*)((uint64_t)(weight.GetAddr()));
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
            auto negative_slope = *(weightAddr + n2Index);
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                auto dstOffset = n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n1Index * srcStride1 + n2Index * srcStride2;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));

                pto::TLRELU(dstTile, srcTile, negative_slope);
            }
        }
    }
}

#endif // INTERFACE_TILEOP_VECTOR_PRELU_H_
