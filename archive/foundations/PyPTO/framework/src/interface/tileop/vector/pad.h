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
 * \file pad.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_PAD__H
#define TILEOP_TILE_OPERATOR_PAD__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <pto::PadValue padValue, typename DstTensor, typename SrcTensor>
TILEOP void TPad(DstTensor dst, SrcTensor src)
{
    constexpr auto dstShapeSize = Std::tuple_size<typename DstTensor::Shape>::value;
    constexpr auto srcShapeSize = Std::tuple_size<typename SrcTensor::Shape>::value;
    static_assert(srcShapeSize == dstShapeSize, "Pad: Src and Dst rank mismatch");

    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    const auto srcLayout = src.GetLayout();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    using SrcDtype = typename SrcTensor::Type;
    using DstDtype = typename DstTensor::Type;
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<DstTensor, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<DstTensor, 4, 5>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<SrcTensor, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<SrcTensor, 4, 5>();
    using DstTileType = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1,
                                  pto::SLayout::NoneBox, 512, padValue>;
    using SrcTileType = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileType dstTile(dstShape3, dstShape4);
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto dstAddr = dst.GetAddr() + dstOffset * sizeof(DstDtype);
                pto::TASSIGN(dstTile, dstAddr);
                SrcTileType srcTile(srcShape3, srcShape4);
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                auto srcAddr = src.GetAddr() + srcOffset * sizeof(SrcDtype);
                pto::TASSIGN(srcTile, srcAddr);
                pto::TFILLPAD_EXPAND(dstTile, srcTile);
            }
        }
    }
}

#endif
