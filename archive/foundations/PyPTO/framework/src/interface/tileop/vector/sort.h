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
 * \file sort.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_SORT__H
#define TILEOP_TILE_OPERATOR_SORT__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_BITSORT TBitSort
template <int axis, int offset, int isLargest, typename T0, typename T1, typename T2>
TILEOP void TBitSort(T0 dst, T1 src, T2 tmp)
{
    constexpr size_t expectSize = 5;
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr auto tmpTileW = dstTileW / 2;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();

    const auto srcLayout = src.GetLayout();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                using IdxTileDefine = pto::Tile<pto::TileType::Vec, uint32_t, 1, tmpTileW, pto::BLayout::RowMajor, -1,
                                                -1>;
                IdxTileDefine idxTile(1, srcShape4);
                pto::TASSIGN(idxTile, (uint64_t)(tmp.GetAddr()));
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                pto::TCI<IdxTileDefine, uint32_t, 0>(idxTile, offset);
                set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, tmpTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    DstTileDefine dstTile(1, dstShape4);
                    SrcTileDefine srcTile(1, srcShape4);
                    TmpTileDefine tmpTile(1, tmpTileW);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 +
                                     n3Index * dstStride3;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2 +
                                     n3Index * srcStride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * srcTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr() + tmpTileW * srcTypeSize));
                    if constexpr (isLargest == 0) {
                        using SrcAddTileDefine = pto::Tile<pto::TileType::Vec, int32_t, 1, srcTileW,
                                                           pto::BLayout::RowMajor, -1, -1>;
                        SrcAddTileDefine srcAddTile(1, srcShape4);
                        pto::TASSIGN(srcAddTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                        int32_t scalar = -2147483648;
                        pto::TADDS(srcAddTile, srcAddTile, scalar);
#ifdef __DAV_V220
                        pipe_barrier(PIPE_V);
#endif
                    }
                    pto::TSORT32(dstTile, srcTile, idxTile, tmpTile);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                }
            }
        }
    }
}

#define OP_TILE_OP_MRGSORT TMrgSort
template <int axis, int k, int mergeSize, typename T0, typename T1, typename T2>
TILEOP void TMrgSort(T0 dst, T1 src, T2 tmp)
{
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();

    const auto srcLayout = src.GetLayout();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    uint32_t totalNum = srcTileW / 2;
    if (srcShape4 == 0) {
        return;
    }
    srcShape4 = srcShape4 / 2;
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    DstTileDefine dstTile(1, dstShape4);
                    SrcTileDefine srcTile(1, srcShape4 * 2);
                    TmpTileDefine tmpTile(1, srcTileW);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 +
                                     n3Index * dstStride3;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2 +
                                     n3Index * srcStride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * srcTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                    LoopVar z = mergeSize;
                    for (; z * 4 <= srcShape4; z *= 4) {
                        uint32_t repeat_mrg = srcShape4 / (z * 4);
                        pto::TMRGSORT(tmpTile, srcTile, z * 2);
#ifdef __DAV_V220
                        pipe_barrier(PIPE_V);
#endif
                        using SrcMovTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW,
                                                           pto::BLayout::RowMajor, -1, -1>;
                        using TmpMovTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW,
                                                           pto::BLayout::RowMajor, -1, -1>;
                        SrcMovTileDefine srcMovTile(1, z * repeat_mrg * 8);
                        TmpMovTileDefine tmpMovTile(1, z * repeat_mrg * 8);
                        pto::TASSIGN(srcMovTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                        pto::TASSIGN(tmpMovTile, (uint64_t)(tmp.GetAddr()));
                        pto::TMOV(srcMovTile, tmpMovTile);
#ifdef __DAV_V220
                        pipe_barrier(PIPE_V);
#endif
                    }
                    if (z < srcShape4) {
                        int32_t arrayCount = 0;
                        int32_t mrgArray[15] = {0};
                        int32_t tmpInner = srcShape4;
                        for (LoopVar i = z; i >= mergeSize; i /= 4) {
                            int32_t count;
                            for (count = 0; count < tmpInner / i; count++) {
                                mrgArray[arrayCount++] = i;
                            }
                            tmpInner -= count * i;
                        }
                        if (tmpInner > 0) {
                            mrgArray[arrayCount++] = tmpInner;
                        }
                        uint16_t mrgSortedLen = 0;
                        for (LoopVar i = 0; i < arrayCount - 1; ++i) {
                            mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
                            uint64_t tmpMrgSortedLen = mrgSortedLen;
                            uint64_t tmpMrgArray = mrgArray[i + 1];
                            if (mrgSortedLen > k) {
                                tmpMrgSortedLen = k;
                            }
                            if (mrgArray[i + 1] > k) {
                                tmpMrgArray = k;
                            }
                            SrcTileDefine src1Tile(1, tmpMrgSortedLen * 2), src2Tile(1, tmpMrgArray * 2);
                            TmpTileDefine tmp1Tile(1, (tmpMrgSortedLen + tmpMrgArray) * 2);
                            DstTileDefine dst1Tile(1, (tmpMrgSortedLen + tmpMrgArray) * 2);
                            pto::TASSIGN(src1Tile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                            pto::TASSIGN(src2Tile,
                                         (uint64_t)(src.GetAddr() + (srcOffset + mrgSortedLen * 2) * srcTypeSize));
                            pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr()));
                            pto::TASSIGN(dst1Tile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                            pto::MrgSortExecutedNumList executedNumList;
                            pto::TMRGSORT<DstTileDefine, TmpTileDefine, SrcTileDefine, SrcTileDefine, false>(
                                dst1Tile, executedNumList, tmp1Tile, src1Tile, src2Tile);
#ifdef __DAV_V220
                            pipe_barrier(PIPE_V);
#endif
                        }
                    }
                    constexpr int64_t TileW = ((k + 7) / 8) * 16;
                    using DstTileMovDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, TileW,
                                                       pto::BLayout::RowMajor, -1, -1>;
                    using SrcTileMovDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, TileW,
                                                       pto::BLayout::RowMajor, -1, -1>;
                    DstTileMovDefine dstTileMov(1, ((k + 7) / 8) * 16);
                    SrcTileMovDefine srcTileMov(1, ((k + 7) / 8) * 16);
                    pto::TASSIGN(dstTileMov, (uint64_t)(dst.GetAddr() + dstOffset * srcTypeSize));
                    pto::TASSIGN(srcTileMov, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TMOV(dstTileMov, srcTileMov);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                }
            }
        }
    }
}

#define OP_TILE_OP_TILEDMEGSORT TTiledMrgSort
template <int k, int validBit, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
TILEOP void TTiledMrgSort(T0 dst, T1 src1, T2 src2, T3 src3, T4 src4, T5 tmp)
{
    constexpr size_t expectSize = 5;
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T5, 3, expectSize>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T5, 4, expectSize>();
    const auto dstTiledSortLayout = dst.GetLayout();
    auto dstShape0 = dstTiledSortLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstTiledSortLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstTiledSortLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstTiledSortLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstTiledSortLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstTiledSortLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstTiledSortLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstTiledSortLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstTiledSortLayout.template GetStrideDim<3, expectSize>();

    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();

    const auto src1Layout = src1.GetLayout();

    auto src1Shape4 = src1Layout.template GetShapeDim<4, expectSize>();
    auto src1Stride0 = src1Layout.template GetStrideDim<0, expectSize>();
    auto src1Stride1 = src1Layout.template GetStrideDim<1, expectSize>();
    auto src1Stride2 = src1Layout.template GetStrideDim<2, expectSize>();
    auto src1Stride3 = src1Layout.template GetStrideDim<3, expectSize>();
    constexpr auto src1TileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();

    const auto src2Layout = src2.GetLayout();
    auto src2Shape4 = src2Layout.template GetShapeDim<4, expectSize>();
    auto src2Stride0 = src2Layout.template GetStrideDim<0, expectSize>();
    auto src2Stride1 = src2Layout.template GetStrideDim<1, expectSize>();
    auto src2Stride2 = src2Layout.template GetStrideDim<2, expectSize>();
    auto src2Stride3 = src2Layout.template GetStrideDim<3, expectSize>();
    constexpr auto src2TileW = TileOp::GetTensorTileShapeDim<T2, 4, expectSize>();

    const auto src3Layout = src3.GetLayout();
    auto src3Shape4 = src3Layout.template GetShapeDim<4, expectSize>();
    auto src3Stride0 = src3Layout.template GetStrideDim<0, expectSize>();
    auto src3Stride1 = src3Layout.template GetStrideDim<1, expectSize>();
    auto src3Stride2 = src3Layout.template GetStrideDim<2, expectSize>();
    auto src3Stride3 = src3Layout.template GetStrideDim<3, expectSize>();
    constexpr auto src3TileW = TileOp::GetTensorTileShapeDim<T3, 4, expectSize>();

    const auto src4Layout = src4.GetLayout();

    auto src4Shape4 = src4Layout.template GetShapeDim<4, expectSize>();
    auto src4Stride0 = src4Layout.template GetStrideDim<0, expectSize>();
    auto src4Stride1 = src4Layout.template GetStrideDim<1, expectSize>();
    auto src4Stride2 = src4Layout.template GetStrideDim<2, expectSize>();
    auto src4Stride3 = src4Layout.template GetStrideDim<3, expectSize>();
    constexpr auto src4TileW = TileOp::GetTensorTileShapeDim<T4, 4, expectSize>();

    constexpr auto srcTypeSize = sizeof(typename T1::Type);

    int validBitNew = validBit;
    if (src1Shape4 == 0 || src2Shape4 == 0) {
        return;
    } else if (src3Shape4 == 0) {
        validBitNew = 2;
        src4Shape4 = src2Shape4;
    } else if (src4Shape4 == 0) {
        validBitNew = 3;
        src4Shape4 = src3Shape4;
    }
    int32_t kLast = k * 2;
    if (k * 2 > src4Shape4) {
        kLast = src4Shape4;
    }
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    using Src1TileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, src1TileW,
                                                     pto::BLayout::RowMajor, -1, -1>;
                    using Src2TileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, src2TileW,
                                                     pto::BLayout::RowMajor, -1, -1>;
                    using Src3TileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, src3TileW,
                                                     pto::BLayout::RowMajor, -1, -1>;
                    using Src4TileDefine = pto::Tile<pto::TileType::Vec, typename T4::Type, 1, src4TileW,
                                                     pto::BLayout::RowMajor, -1, -1>;
                    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T5::Type, 1, tmpTileW,
                                                    pto::BLayout::RowMajor, 1, tmpTileW>;
                    DstTileDefine dstTile(1, dstShape4);
                    TmpTileDefine tmpTile;
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 +
                                     n3Index * dstStride3;
                    auto src1Offset = n0Index * src1Stride0 + n1Index * src1Stride1 + n2Index * src1Stride2 +
                                      n3Index * src1Stride3;
                    auto src2Offset = n0Index * src2Stride0 + n1Index * src2Stride1 + n2Index * src2Stride2 +
                                      n3Index * src2Stride3;
                    auto src3Offset = n0Index * src3Stride0 + n1Index * src3Stride1 + n2Index * src3Stride2 +
                                      n3Index * src3Stride3;
                    auto src4Offset = n0Index * src4Stride0 + n1Index * src4Stride1 + n2Index * src4Stride2 +
                                      n3Index * src4Stride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * srcTypeSize));
                    pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                    pto::MrgSortExecutedNumList executedNumList;
                    if (validBitNew == 2) {
                        Src1TileDefine src1Tile(1, k * 2);
                        Src2TileDefine src2Tile(1, kLast);
                        pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * srcTypeSize));
                        pto::TASSIGN(src2Tile, (uint64_t)(src2.GetAddr() + src2Offset * srcTypeSize));
                        pto::TMRGSORT<DstTileDefine, TmpTileDefine, Src1TileDefine, Src2TileDefine, false>(
                            dstTile, executedNumList, tmpTile, src1Tile, src2Tile);
                    } else if (validBitNew == 3) {
                        Src1TileDefine src1Tile(1, k * 2);
                        Src2TileDefine src2Tile(1, k * 2);
                        Src3TileDefine src3Tile(1, kLast);
                        pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * srcTypeSize));
                        pto::TASSIGN(src2Tile, (uint64_t)(src2.GetAddr() + src2Offset * srcTypeSize));
                        pto::TASSIGN(src3Tile, (uint64_t)(src3.GetAddr() + src3Offset * srcTypeSize));
                        pto::TMRGSORT<DstTileDefine, TmpTileDefine, Src1TileDefine, Src2TileDefine, Src3TileDefine,
                                      false>(dstTile, executedNumList, tmpTile, src1Tile, src2Tile, src3Tile);
                    } else if (validBitNew == 4) {
                        Src1TileDefine src1Tile(1, k * 2);
                        Src2TileDefine src2Tile(1, k * 2);
                        Src3TileDefine src3Tile(1, k * 2);
                        Src4TileDefine src4Tile(1, kLast);
                        pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * srcTypeSize));
                        pto::TASSIGN(src2Tile, (uint64_t)(src2.GetAddr() + src2Offset * srcTypeSize));
                        pto::TASSIGN(src3Tile, (uint64_t)(src3.GetAddr() + src3Offset * srcTypeSize));
                        pto::TASSIGN(src4Tile, (uint64_t)(src4.GetAddr() + src4Offset * srcTypeSize));
                        pto::TMRGSORT<DstTileDefine, TmpTileDefine, Src1TileDefine, Src2TileDefine, Src3TileDefine,
                                      Src4TileDefine, false>(dstTile, executedNumList, tmpTile, src1Tile, src2Tile,
                                                             src3Tile, src4Tile);
                    }
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                }
            }
        }
    }
}

#define OP_TILE_OP_TWOTILEMRGSORT TTwoTileMrgSort
template <unsigned firstShape, typename T0, typename T1>
TILEOP void TTwoTileMrgSort(T0 dst, T1 src)
{
    constexpr size_t expectSize = 5;

    const auto dstLayout = dst.GetLayout();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();

    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    const auto srcLayout = src.GetLayout();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    if (srcShape4 == 0) {
        return;
    }

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();
    auto srcStride4 = srcLayout.template GetStrideDim<4, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();

    constexpr auto tileW = srcTileW;

    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; n3Index++) {
                    if (srcShape4 <= firstShape) {
                        using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW,
                                                        pto::BLayout::RowMajor, -1, -1>;
                        using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, tileW,
                                                        pto::BLayout::RowMajor, -1, -1>;
                        DstTileDefine dstTile(1, dstShape4);
                        SrcTileDefine srcTile(1, srcShape4);
                        auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 +
                                         n3Index * dstStride3;
                        auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2 +
                                         n3Index * srcStride3;
                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));

                        pto::TMOV(dstTile, srcTile);
                    } else {
                        using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW,
                                                        pto::BLayout::RowMajor, -1, -1>;
                        DstTileDefine dstTile(1, dstShape4);
                        auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 +
                                         n3Index * dstStride3;
                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                        using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, firstShape,
                                                        pto::BLayout::RowMajor, -1, -1>;
                        SrcTileDefine src0Tile(1, firstShape);
                        SrcTileDefine src1Tile(1, srcShape4 - firstShape);
                        auto src0Offset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2 +
                                          n3Index * srcStride3;
                        auto src1Offset = src0Offset + firstShape;
                        pto::TASSIGN(src0Tile, (uint64_t)(src.GetAddr() + src0Offset * srcTypeSize));
                        pto::TASSIGN(src1Tile, (uint64_t)(src.GetAddr() + src1Offset * srcTypeSize));

                        pto::MrgSortExecutedNumList executedNumList;
                        // 直接将dst用作tmp, 节省空间
                        pto::TMRGSORT<DstTileDefine, DstTileDefine, SrcTileDefine, SrcTileDefine, false>(
                            dstTile, executedNumList, dstTile, src0Tile, src1Tile);
                    }
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                }
            }
        }
    }
}

#endif
