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
 * \file reduce.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_REDUCE__H
#define TILEOP_TILE_OPERATOR_REDUCE__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <ReduceOp op, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void ReduceComputeImpl(T0 dst, T1 src, T2 tmp)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;
    if constexpr (op == ReduceOp::SUM) {
        PTO_WITH_LAST_USE(pto::TROWSUM(dst, src, tmp), n1, n2, n3);
    } else if constexpr (op == ReduceOp::MAX) {
        PTO_WITH_LAST_USE(pto::TROWMAX(dst, src, tmp), n1, n2, n3);
    } else if constexpr (op == ReduceOp::MIN) {
        PTO_WITH_LAST_USE(pto::TROWMIN(dst, src, tmp), n1, n2, n3);
    } else if constexpr (op == ReduceOp::PROD) {
        PTO_WITH_LAST_USE(pto::TROWPROD(dst, src, tmp), n1, n2, n3);
    } else if constexpr (op == ReduceOp::ARGMAX) {
        pto::TROWARGMAX(dst, src, tmp);
    } else if constexpr (op == ReduceOp::ARGMIN) {
        pto::TROWARGMIN(dst, src, tmp);
    }
}

template <ReduceOp op, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void ReduceLastAxisCompute(T0 dst, T1 src, T2 tmp)
{
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tmpShapeSize = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T2, 3, 5>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor,
                                    tmpTileH, tmpTileW>;
    TmpTileDefine tmpTile;

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
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();

    const auto srcLayout = src.GetLayout();
    auto srcShape0 = srcLayout.template GetShapeDim<0, expectSize>();
    auto srcShape1 = srcLayout.template GetShapeDim<1, expectSize>();
    auto srcShape2 = srcLayout.template GetShapeDim<2, expectSize>();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    if (srcShape0 == 0 || srcShape1 == 0 || srcShape2 == 0 || srcShape3 == 0 || srcShape4 == 0) {
        return;
    }
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                using DstTileDefine = typename std::conditional<
                    (dstTileW == 1),
                    pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::ColMajor, -1,
                              -1>,
                    pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1,
                              -1> >::type;
                using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW,
                                                pto::BLayout::RowMajor, -1, -1>;
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                ReduceComputeImpl<op, LastUse>(dstTile, srcTile, tmpTile);
            }
        }
    }
}

#define OP_TILE_OP_ROWSUMSINGLE TRowSumSingle
template <typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2>
TILEOP void TRowSumSingle(T0 dst, T1 src, T2 tmp)
{
    ReduceLastAxisCompute<ReduceOp::SUM, LastUse>(dst, src, tmp);
}

template <ReduceOp op, typename T0, typename T1, typename T2, typename T3>
TILEOP void ArgReduceLastAxisCompute(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T3, 3, 5>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T3, 4, 5>();
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor,
                                    tmpTileH, tmpTileW>;
    TmpTileDefine tmpTile;

    constexpr size_t expectSize = 5;
    const auto dstLayout = dstValue.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();

    const auto srcLayout = src.GetLayout();
    auto srcShape0 = srcLayout.template GetShapeDim<0, expectSize>();
    auto srcShape1 = srcLayout.template GetShapeDim<1, expectSize>();
    auto srcShape2 = srcLayout.template GetShapeDim<2, expectSize>();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    if (srcShape0 == 0 || srcShape1 == 0 || srcShape2 == 0 || srcShape3 == 0 || srcShape4 == 0) {
        return;
    }
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T2, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();
    constexpr auto srcTypeSize = sizeof(typename T2::Type);
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                using DstValueTileDefine = typename std::conditional<
                    (dstTileW == 1),
                    pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::ColMajor, -1,
                              -1>,
                    pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1,
                              -1> >::type;
                using DstIndexTileDefine = typename std::conditional<
                    (dstTileW == 1),
                    pto::Tile<pto::TileType::Vec, typename T1::Type, dstTileH, dstTileW, pto::BLayout::ColMajor, -1,
                              -1>,
                    pto::Tile<pto::TileType::Vec, typename T1::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1,
                              -1> >::type;
                using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, srcTileH, srcTileW,
                                                pto::BLayout::RowMajor, -1, -1>;
                DstValueTileDefine dstValueTile(dstShape3, dstShape4);
                DstIndexTileDefine dstIndexTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstValueTile, (uint64_t)(dstValue.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(dstIndexTile, (uint64_t)(dstIndex.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                if constexpr (op == ReduceOp::ROWARGMAXWITHVALUE) {
                    pto::TROWARGMAX(dstValueTile, dstIndexTile, srcTile, tmpTile);
                } else if constexpr (op == ReduceOp::ROWARGMINWITHVALUE) {
                    pto::TROWARGMIN(dstValueTile, dstIndexTile, srcTile, tmpTile);
                }
            }
        }
    }
}

#define OP_TILE_OP_ROWARGMAXWITHVALUESINGLE TRowArgMaxWithValueSingle
template <typename LastUse = LastUse4Dim<0, 0, 0, 0>, typename T0, typename T1, typename T2, typename T3>
TILEOP void TRowArgMaxWithValueSingle(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    ArgReduceLastAxisCompute<ReduceOp::ROWARGMAXWITHVALUE>(dstValue, dstIndex, src, tmp);
}

#define OP_TILE_OP_ROWARGMINWITHVALUESINGLE TRowArgMinWithValueSingle
template <typename LastUse = LastUse4Dim<0, 0, 0, 0>, typename T0, typename T1, typename T2, typename T3>
TILEOP void TRowArgMinWithValueSingle(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    ArgReduceLastAxisCompute<ReduceOp::ROWARGMINWITHVALUE>(dstValue, dstIndex, src, tmp);
}

#define OP_TILE_OP_ROWMAXSINGLE TRowMaxSingle
template <typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2>
TILEOP void TRowMaxSingle(T0 dst, T1 src, T2 tmp)
{
    ReduceLastAxisCompute<ReduceOp::MAX, LastUse>(dst, src, tmp);
}

#define OP_TILE_OP_ROWMINSINGLE TRowMinSingle
template <typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2>
TILEOP void TRowMinSingle(T0 dst, T1 src, T2 tmp)
{
    ReduceLastAxisCompute<ReduceOp::MIN, LastUse>(dst, src, tmp);
}

#define OP_TILE_OP_ROWPRODSINGLE TRowProdSingle
template <typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2>
TILEOP void TRowProdSingle(T0 dst, T1 src, T2 tmp)
{
    ReduceLastAxisCompute<ReduceOp::PROD, LastUse>(dst, src, tmp);
}

template <ReduceOp op, int axis, typename T0, typename T1>
TILEOP void TRowMaxMinProdLineDynamic(T0 dst, T1 src)
{
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, axis + dstShapeSize - 5>();
    constexpr auto
        dstTileW = TileOp::GetAnyAxisMergeResult<axis + dstShapeSize - 3, dstShapeSize, typename T0::TileShape>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, axis + srcShapeSize - 5>();
    constexpr auto
        srcTileW = TileOp::GetAnyAxisMergeResult<axis + srcShapeSize - 3, srcShapeSize, typename T1::TileShape>();
    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    constexpr size_t expectSize = 5;
    constexpr auto typeSize = sizeof(typename T1::Type);
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    size_t dstShape[] = {static_cast<size_t>(dstLayout.template GetShapeDim<0, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<1, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<2, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<3, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<4, expectSize>())};
    size_t dstStride[] = {static_cast<size_t>(dstLayout.template GetStrideDim<0, expectSize>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<1, expectSize>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<2, expectSize>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<3, expectSize>())};
    size_t srcShape[] = {static_cast<size_t>(srcLayout.template GetShapeDim<0, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<1, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<2, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<3, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<4, expectSize>())};
    size_t srcStride[] = {static_cast<size_t>(srcLayout.template GetStrideDim<0, expectSize>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<1, expectSize>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<2, expectSize>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<3, expectSize>())};
    for (LoopVar n0Index = 0, n0Size = (axis == 0 ? (size_t)1 : dstShape[0]); n0Index < n0Size; ++n0Index) {
        for (LoopVar n1Index = 0, n1Size = (axis == 1 ? (size_t)1 : dstShape[1]); n1Index < n1Size; ++n1Index) {
            for (LoopVar n2Index = 0, n2Size = (axis == 2 ? (size_t)1 : dstShape[2]); n2Index < n2Size; ++n2Index) {
                for (LoopVar n3Index = 0, n3Size = (axis == 3 ? (size_t)1 : dstShape[3]); n3Index < n3Size; ++n3Index) {
                    DstTileDefine dstTile(dstShape[axis], dstShape[4]);
                    SrcTileDefine srcTile(srcShape[axis], srcShape[4]);
                    auto dstOffset = n0Index * dstStride[0] + n1Index * dstStride[1] + n2Index * dstStride[2] +
                                     n3Index * dstStride[3];
                    auto srcOffset = n0Index * srcStride[0] + n1Index * srcStride[1] + n2Index * srcStride[2] +
                                     n3Index * srcStride[3];
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * typeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * typeSize));
                    if constexpr (op == ReduceOp::MAX) {
                        pto::TCOLMAX(dstTile, srcTile);
                    } else if constexpr (op == ReduceOp::MIN) {
                        pto::TCOLMIN(dstTile, srcTile);
                    } else if constexpr (op == ReduceOp::PROD) {
                        pto::TCOLPROD(dstTile, srcTile);
                    }
                }
            }
        }
    }
}

#define OP_TILE_OP_ROWMAXLINE TRowMaxLine
template <int axis, typename T0, typename T1>
TILEOP void TRowMaxLine(T0 dst, T1 src)
{
    TRowMaxMinProdLineDynamic<ReduceOp::MAX, axis>(dst, src);
}

#define OP_TILE_OP_ROWMINLINE TRowMinLine
template <int axis, typename T0, typename T1>
TILEOP void TRowMinLine(T0 dst, T1 src)
{
    TRowMaxMinProdLineDynamic<ReduceOp::MIN, axis>(dst, src);
}

#define OP_TILE_OP_ROWPRODLINE TRowProdLine
template <int axis, typename T0, typename T1>
TILEOP void TRowProdLine(T0 dst, T1 src)
{
    TRowMaxMinProdLineDynamic<ReduceOp::PROD, axis>(dst, src);
}

template <int axis, ReduceOp op, typename DstTileDefine, typename SrcTileDefine, typename TmpTileDefine, typename T0,
          typename T1, typename T2>
TILEOP void ColReduceWithTmpImp(T0 dst, T1 src, T2 tmp)
{
    constexpr size_t expectSize = 5;
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto tmpLayout = tmp.GetLayout();
    size_t dstShape[] = {static_cast<size_t>(dstLayout.template GetShapeDim<0, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<1, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<2, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<3, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<4, expectSize>())};
    size_t dstStride[] = {static_cast<size_t>(dstLayout.template GetStrideDim<0, expectSize>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<1, expectSize>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<2, expectSize>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<3, expectSize>())};
    size_t srcShape[] = {static_cast<size_t>(srcLayout.template GetShapeDim<0, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<1, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<2, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<3, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<4, expectSize>())};
    size_t srcStride[] = {static_cast<size_t>(srcLayout.template GetStrideDim<0, expectSize>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<1, expectSize>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<2, expectSize>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<3, expectSize>())};
    for (LoopVar n0Index = 0, n0Size = (axis == 0 ? (size_t)1 : dstShape[0]); n0Index < n0Size; ++n0Index) {
        for (LoopVar n1Index = 0, n1Size = (axis == 1 ? (size_t)1 : dstShape[1]); n1Index < n1Size; ++n1Index) {
            for (LoopVar n2Index = 0, n2Size = (axis == 2 ? (size_t)1 : dstShape[2]); n2Index < n2Size; ++n2Index) {
                for (LoopVar n3Index = 0, n3Size = (axis == 3 ? (size_t)1 : dstShape[3]); n3Index < n3Size; ++n3Index) {
                    DstTileDefine dstTile(dstShape[axis], dstShape[4]);
                    SrcTileDefine srcTile(srcShape[axis], srcShape[4]);
                    TmpTileDefine tmpTile;
                    auto dstOffset = n0Index * dstStride[0] + n1Index * dstStride[1] + n2Index * dstStride[2] +
                                     n3Index * dstStride[3];
                    auto srcOffset = n0Index * srcStride[0] + n1Index * srcStride[1] + n2Index * srcStride[2] +
                                     n3Index * srcStride[3];
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                    if constexpr (op == ReduceOp::SUM) {
                        pto::TCOLSUM(dstTile, srcTile, tmpTile, true);
                    } else if constexpr (op == ReduceOp::ARGMAX) {
                        pto::TCOLARGMAX(dstTile, srcTile, tmpTile);
                    } else if constexpr (op == ReduceOp::ARGMIN) {
                        pto::TCOLARGMIN(dstTile, srcTile, tmpTile);
                    }
                }
            }
        }
    }
}

template <int axis, ReduceOp op, typename T0, typename T1, typename T2>
TILEOP void ColReduceWithTmp(T0 dst, T1 src, T2 tmp)
{
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tmpShapeSize = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, axis + dstShapeSize - 5>();
    constexpr auto
        dstTileW = TileOp::GetAnyAxisMergeResult<axis + dstShapeSize - 3, dstShapeSize, typename T0::TileShape>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, axis + srcShapeSize - 5>();
    constexpr auto
        srcTileW = TileOp::GetAnyAxisMergeResult<axis + srcShapeSize - 3, srcShapeSize, typename T1::TileShape>();
    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T2, tmpShapeSize - 2>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T2, tmpShapeSize - 1>();
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor,
                                    tmpTileH, tmpTileW>;
    ColReduceWithTmpImp<axis, op, DstTileDefine, SrcTileDefine, TmpTileDefine>(dst, src, tmp);
}

#define OP_TILE_OP_ROWSUMLINE TRowSumLine
template <int axis, typename T0, typename T1, typename T2>
TILEOP void TRowSumLine(T0 dst, T1 src, T2 tmp)
{
    ColReduceWithTmp<axis, ReduceOp::SUM, T0, T1, T2>(dst, src, tmp);
}

#define OP_TILE_OP_ROWARGMAXLINE TRowArgMaxLine
template <int axis, typename T0, typename T1, typename T2>
TILEOP void TRowArgMaxLine(T0 dst, T1 src, T2 tmp)
{
    ColReduceWithTmp<axis, ReduceOp::ARGMAX, T0, T1, T2>(dst, src, tmp);
}

template <int axis, ReduceOp op, typename DstValueTileDefine, typename DstIndexTileDefine, typename SrcTileDefine,
          typename TmpTileDefine, typename T0, typename T1, typename T2, typename T3>
TILEOP void ColArgReduceImp(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    constexpr size_t expectSize = 5;
    constexpr auto srcTypeSize = sizeof(typename T2::Type);
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    const auto dstLayout = dstValue.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto tmpLayout = tmp.GetLayout();
    size_t dstShape[] = {static_cast<size_t>(dstLayout.template GetShapeDim<0, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<1, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<2, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<3, expectSize>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<4, expectSize>())};
    size_t dstStride[] = {static_cast<size_t>(dstLayout.template GetStrideDim<0, expectSize>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<1, expectSize>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<2, expectSize>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<3, expectSize>())};
    size_t srcShape[] = {static_cast<size_t>(srcLayout.template GetShapeDim<0, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<1, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<2, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<3, expectSize>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<4, expectSize>())};
    size_t srcStride[] = {static_cast<size_t>(srcLayout.template GetStrideDim<0, expectSize>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<1, expectSize>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<2, expectSize>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<3, expectSize>())};
    for (LoopVar n0Index = 0, n0Size = (axis == 0 ? (size_t)1 : dstShape[0]); n0Index < n0Size; ++n0Index) {
        for (LoopVar n1Index = 0, n1Size = (axis == 1 ? (size_t)1 : dstShape[1]); n1Index < n1Size; ++n1Index) {
            for (LoopVar n2Index = 0, n2Size = (axis == 2 ? (size_t)1 : dstShape[2]); n2Index < n2Size; ++n2Index) {
                for (LoopVar n3Index = 0, n3Size = (axis == 3 ? (size_t)1 : dstShape[3]); n3Index < n3Size; ++n3Index) {
                    DstValueTileDefine dstValueTile(dstShape[axis], dstShape[4]);
                    DstIndexTileDefine dstIndexTile(dstShape[axis], dstShape[4]);
                    SrcTileDefine srcTile(srcShape[axis], srcShape[4]);
                    TmpTileDefine tmpTile;
                    auto dstOffset = n0Index * dstStride[0] + n1Index * dstStride[1] + n2Index * dstStride[2] +
                                     n3Index * dstStride[3];
                    auto srcOffset = n0Index * srcStride[0] + n1Index * srcStride[1] + n2Index * srcStride[2] +
                                     n3Index * srcStride[3];
                    pto::TASSIGN(dstValueTile, (uint64_t)(dstValue.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(dstIndexTile, (uint64_t)(dstIndex.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                    if constexpr (op == ReduceOp::COLARGMAXWITHVALUE) {
                        pto::TCOLARGMAX(dstValueTile, dstIndexTile, srcTile, tmpTile);
                    } else if constexpr (op == ReduceOp::COLARGMINWITHVALUE) {
                        pto::TCOLARGMIN(dstValueTile, dstIndexTile, srcTile, tmpTile);
                    }
                }
            }
        }
    }
}

template <int axis, ReduceOp op, typename T0, typename T1, typename T2, typename T3>
TILEOP void ColArgReduce(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    constexpr auto srcShapeSize = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tmpShapeSize = Std::tuple_size<typename T3::Shape>::value;
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, axis + dstShapeSize - 5>();
    constexpr auto
        dstTileW = TileOp::GetAnyAxisMergeResult<axis + dstShapeSize - 3, dstShapeSize, typename T0::TileShape>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T2, axis + srcShapeSize - 5>();
    constexpr auto
        srcTileW = TileOp::GetAnyAxisMergeResult<axis + srcShapeSize - 3, srcShapeSize, typename T2::TileShape>();
    using DstValueTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW,
                                         pto::BLayout::RowMajor, -1, -1>;
    using DstIndexTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, dstTileH, dstTileW,
                                         pto::BLayout::RowMajor, -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, srcTileH, srcTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T3, tmpShapeSize - 2>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T3, tmpShapeSize - 1>();
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T3::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor,
                                    tmpTileH, tmpTileW>;
    ColArgReduceImp<axis, op, DstValueTileDefine, DstIndexTileDefine, SrcTileDefine, TmpTileDefine>(dstValue, dstIndex,
                                                                                                    src, tmp);
}

#define OP_TILE_OP_ROWARGMAXWITHVALUELINE TRowArgMaxWithValueLine
template <int axis, typename T0, typename T1, typename T2, typename T3>
TILEOP void TRowArgMaxWithValueLine(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    ColArgReduce<axis, ReduceOp::COLARGMAXWITHVALUE, T0, T1, T2, T3>(dstValue, dstIndex, src, tmp);
}

#define OP_TILE_OP_ROWARGMINWITHVALUELINE TRowArgMinWithValueLine
template <int axis, typename T0, typename T1, typename T2, typename T3>
TILEOP void TRowArgMinWithValueLine(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    ColArgReduce<axis, ReduceOp::COLARGMINWITHVALUE, T0, T1, T2, T3>(dstValue, dstIndex, src, tmp);
}
#endif
