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
 * \file extract.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_EXTRACT__H
#define TILEOP_TILE_OPERATOR_EXTRACT__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_EXTRACT TExtract
template <int k, int extractMode, int isLargest, typename T0, typename T1>
TILEOP void TExtract(T0 dst, T1 src)
{
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW,
                                                pto::BLayout::RowMajor, -1, -1>;
                using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW,
                                                pto::BLayout::RowMajor, -1, -1>;
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcTileW);
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                constexpr auto pattern = (extractMode == 0) ? pto::MaskPattern::P0101 : pto::MaskPattern::P1010;
                pto::TGATHER<DstTileDefine, SrcTileDefine, pattern>(dstTile, srcTile);
#ifdef __DAV_V220
                pipe_barrier(PIPE_V);
#endif
                if constexpr (extractMode == 0 && isLargest == 0) {
                    using DstAddTileDefine = pto::Tile<pto::TileType::Vec, int32_t, dstTileH, dstTileW,
                                                       pto::BLayout::RowMajor, -1, -1>;
                    DstAddTileDefine dstAddTile(dstShape3, dstTileW);
                    pto::TASSIGN(dstAddTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    int32_t scalar = -2147483648;
                    pto::TADDS(dstAddTile, dstAddTile, scalar);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                }
            }
        }
    }
}

#define OP_TILE_OP_EXTRACTSINGLE TExtractSingle
template <int extractMode, int isLargest, typename T0, typename T1>
TILEOP void TExtractSingle(T0 dst, T1 src)
{
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();

    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();

    if (srcShape4 == 0) {
        return;
    }

    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; n3Index++) {
                    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    DstTileDefine dstTile(1, dstShape4);
                    SrcTileDefine srcTile(1, srcShape4);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 +
                                     n3Index * dstStride3;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2 +
                                     n3Index * srcStride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    constexpr auto pattern = (extractMode == 0) ? pto::MaskPattern::P0101 : pto::MaskPattern::P1010;
                    pto::TGATHER<DstTileDefine, SrcTileDefine, pattern>(dstTile, srcTile);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif

                    if constexpr (extractMode == 0 && isLargest == 0) {
                        int32_t scalar = -2147483648;
                        using DstAddTileDefine = pto::Tile<pto::TileType::Vec, int32_t, 1, dstTileW,
                                                           pto::BLayout::RowMajor, -1, -1>;
                        DstAddTileDefine dstAddTile(1, dstShape4);
                        pto::TASSIGN(dstAddTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                        pto::TADDS(dstAddTile, dstAddTile, scalar);
#ifdef __DAV_V220
                        pipe_barrier(PIPE_V);
                        ;
#endif
                    }
                }
            }
        }
    }
}

#endif
