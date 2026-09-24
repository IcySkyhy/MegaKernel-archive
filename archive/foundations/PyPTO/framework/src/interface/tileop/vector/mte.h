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
 * \file mte.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_MTE__H
#define TILEOP_TILE_OPERATOR_MTE__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <pto::AtomicType att = pto::AtomicType::AtomicNone>
struct TStoreConfigVec {
    static constexpr pto::AtomicType kAtomicType = att;
};

#define OP_TILE_OP_UB_COPY_IN TLoad
template <typename T, typename U, typename C>
__aicore__ inline void TLoad(T dst, U src, C coordinate)
{
    if constexpr (T::FORMAT == Hardware::UB && U::FORMAT == Hardware::GM) {
        const auto srcLayout = src.GetLayout();
        auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
        auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
        auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

        const auto dstLayout = dst.GetLayout();
        auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
        auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
        auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
        auto gmOffset = srcLayout.template GetGmOffset<C, MAX_DIMS>(coordinate);

        if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0) {
            return;
        }

        if constexpr (TileOp::IsConstContinous<T>() == true) {
            // 对于静态整块场景，将UB合成二维，GM保持五维
            auto srcGlobal = PtoGlobal<U, typename T::Shape, typename U::Stride>(
                                 TileOp::GetPackedGmAddr<typename U::Type>(src.GetAddr(), gmOffset), dst.GetShape(),
                                 src.GetStride())
                                 .Data();
            auto dstTile = PtoTile<T, pto::BLayout::RowMajor, true, void, false>().Data();
            pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
            pto::TLOAD(dstTile, srcGlobal);
            return;
        }

        auto dstTile = PtoTile<T>(dst);
        auto srcGlobal = PtoGlobal<U, typename T::Shape, typename U::Stride, true>(dst.GetShape(), src.GetStride());
        for (LoopVar index0 = 0; index0 < dstShape0; ++index0) {
            for (LoopVar index1 = 0; index1 < dstShape1; ++index1) {
                for (LoopVar index2 = 0; index2 < dstShape2; ++index2) {
                    auto srcOffset = gmOffset + index0 * srcStride0 + index1 * srcStride1 + index2 * srcStride2;
                    srcGlobal.Assign(TileOp::GetPackedGmAddr<typename U::Type>(src.GetAddr(), srcOffset));
                    auto tileOffsets = TileOffset(index0, index1, index2);
                    dstTile.Assign(dst, tileOffsets);
                    pto::TLOAD(dstTile.Data(), srcGlobal.Data());
                }
            }
        }
    }
}

template <typename config, typename T, typename U>
__aicore__ inline void CallStore(T dst, U src)
{
    pto::TSTORE<U, T, config::kAtomicType>(dst, src);
}

#define OP_TILE_OP_UB_COPY_OUT TStoreVec
template <typename config, typename T, typename U, typename C>
__aicore__ inline void TStoreVec(T dst, U src, C coordinate)
{
    if constexpr (U::FORMAT == Hardware::UB && T::FORMAT == Hardware::GM) {
        const auto srcLayout = src.GetLayout();
        auto srcShape0 = srcLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
        auto srcShape1 = srcLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
        auto srcShape2 = srcLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
        const auto dstLayout = dst.GetLayout();
        auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
        auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
        auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
        auto gmOffset = dstLayout.template GetGmOffset<C, MAX_DIMS>(coordinate);

        if (srcShape0 == 0 || srcShape1 == 0 || srcShape2 == 0) {
            return;
        }

        if constexpr (TileOp::IsConstContinous<U>() == true) {
            // 对于静态整块场景，将UB合成二维，GM保持五维
            auto dstGlobal = PtoGlobal<T, typename U::Shape, typename T::Stride>(
                                 TileOp::GetPackedGmAddr<typename T::Type>(dst.GetAddr(), gmOffset), src.GetShape(),
                                 dst.GetStride())
                                 .Data();
            auto srcTile = PtoTile<U, pto::BLayout::RowMajor, true, void, false>().Data();
            pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
            CallStore<config>(dstGlobal, srcTile);
            return;
        }

        auto srcTile = PtoTile<U>(src);
        auto dstGlobal = PtoGlobal<T, typename U::Shape, typename T::Stride, true>(src.GetShape(), dst.GetStride());
        for (LoopVar index0 = 0; index0 < srcShape0; ++index0) {
            for (LoopVar index1 = 0; index1 < srcShape1; ++index1) {
                for (LoopVar index2 = 0; index2 < srcShape2; ++index2) {
                    auto dstOffset = gmOffset + index0 * dstStride0 + index1 * dstStride1 + index2 * dstStride2;
                    dstGlobal.Assign(TileOp::GetPackedGmAddr<typename T::Type>(dst.GetAddr(), dstOffset));
                    auto tileOffsets = TileOffset(index0, index1, index2);
                    srcTile.Assign(src, tileOffsets);
                    CallStore<config>(dstGlobal.Data(), srcTile.Data());
                }
            }
        }
    }
}

#define OP_TILE_OP_RESHAPE_COPY_IN TReshapeLoad
template <typename C, typename S1, typename S2, typename S3, typename S4>
__aicore__ inline size_t GetReshapeGmOffset(C coordinate, S1 gmValidShape1, S2 gmValidShape2, S3 gmValidShape3,
                                            S4 gmValidShape4, Stride5Dim& reshapeStride)
{
    size_t reshapeStride0 = gmValidShape1 * gmValidShape2 * gmValidShape3 * gmValidShape4;
    size_t reshapeStride1 = gmValidShape2 * gmValidShape3 * gmValidShape4;
    size_t reshapeStride2 = gmValidShape3 * gmValidShape4;
    size_t reshapeStride3 = gmValidShape4;
    size_t reshapeStride4 = 1;
    reshapeStride = {reshapeStride0, reshapeStride1, reshapeStride2, reshapeStride3, reshapeStride4};

    auto c0 = TileOp::GetTupleElement<C, DIM_1ST, MAX_DIMS, 0>(coordinate);
    auto c1 = TileOp::GetTupleElement<C, DIM_2ND, MAX_DIMS, 0>(coordinate);
    auto c2 = TileOp::GetTupleElement<C, DIM_3RD, MAX_DIMS, 0>(coordinate);
    auto c3 = TileOp::GetTupleElement<C, DIM_4TH, MAX_DIMS, 0>(coordinate);
    auto c4 = TileOp::GetTupleElement<C, DIM_5TH, MAX_DIMS, 0>(coordinate);

    return c0 * reshapeStride0 + c1 * reshapeStride1 + c2 * reshapeStride2 + c3 * reshapeStride3 + c4 * reshapeStride4;
}

template <typename T, typename U, typename C, typename S0, typename S1, typename S2, typename S3, typename S4>
__aicore__ inline void TReshapeLoad(T dst, U src, C coordinate, S0 gmValidShape0, S1 gmValidShape1, S2 gmValidShape2,
                                    S3 gmValidShape3, S4 gmValidShape4)
{
    (void)gmValidShape0;
    if constexpr (T::FORMAT == Hardware::UB && U::FORMAT == Hardware::GM) {
        // Copy the current GM tile while addressing GM with its valid-shape contiguous strides.
        const auto dstLayout = dst.GetLayout();
        auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
        auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
        auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
        Stride5Dim reshapeStride{};
        auto gmOffset = GetReshapeGmOffset(coordinate, gmValidShape1, gmValidShape2, gmValidShape3, gmValidShape4,
                                           reshapeStride);

        auto reshapeStride0 = TileOp::GetTupleElement<Stride5Dim, DIM_1ST, MAX_DIMS, 0>(reshapeStride);
        auto reshapeStride1 = TileOp::GetTupleElement<Stride5Dim, DIM_2ND, MAX_DIMS, 0>(reshapeStride);
        auto reshapeStride2 = TileOp::GetTupleElement<Stride5Dim, DIM_3RD, MAX_DIMS, 0>(reshapeStride);

        if constexpr (TileOp::IsConstContinous<T>() == true) {
            auto srcGlobal = PtoGlobal<U, typename T::Shape, Stride5Dim>(
                                 TileOp::GetPackedGmAddr<typename U::Type>(src.GetAddr(), gmOffset), dst.GetShape(),
                                 reshapeStride)
                                 .Data();
            auto dstTile = PtoTile<T, pto::BLayout::RowMajor, true, void, false>().Data();
            pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
            pto::TLOAD(dstTile, srcGlobal);
            return;
        }

        auto dstTile = PtoTile<T>(dst);
        auto srcGlobal = PtoGlobal<T, typename T::Shape, Stride5Dim, true>(dst.GetShape(), reshapeStride);
        for (LoopVar index0 = 0; index0 < dstShape0; ++index0) {
            for (LoopVar index1 = 0; index1 < dstShape1; ++index1) {
                for (LoopVar index2 = 0; index2 < dstShape2; ++index2) {
                    auto srcOffset = gmOffset + index0 * reshapeStride0 + index1 * reshapeStride1 +
                                     index2 * reshapeStride2;
                    srcGlobal.Assign(TileOp::GetPackedGmAddr<typename U::Type>(src.GetAddr(), srcOffset));
                    auto tileOffsets = TileOffset(index0, index1, index2);
                    dstTile.Assign(dst, tileOffsets);
                    pto::TLOAD(dstTile.Data(), srcGlobal.Data());
                }
            }
        }
    }
}

#define OP_TILE_OP_RESHAPE_COPY_OUT TReshapeStore
template <typename T, typename U, typename C, typename S0, typename S1, typename S2, typename S3, typename S4>
__aicore__ inline void TReshapeStore(T dst, U src, C coordinate, S0 gmValidShape0, S1 gmValidShape1, S2 gmValidShape2,
                                     S3 gmValidShape3, S4 gmValidShape4)
{
    (void)gmValidShape0;
    if constexpr (U::FORMAT == Hardware::UB && T::FORMAT == Hardware::GM) {
        // Copy the current UB tile while addressing GM with its valid-shape contiguous strides.
        const auto srcLayout = src.GetLayout();
        auto srcShape0 = srcLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
        auto srcShape1 = srcLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
        auto srcShape2 = srcLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
        Stride5Dim reshapeStride{};
        auto gmOffset = GetReshapeGmOffset(coordinate, gmValidShape1, gmValidShape2, gmValidShape3, gmValidShape4,
                                           reshapeStride);
        auto reshapeStride0 = TileOp::GetTupleElement<Stride5Dim, DIM_1ST, MAX_DIMS, 0>(reshapeStride);
        auto reshapeStride1 = TileOp::GetTupleElement<Stride5Dim, DIM_2ND, MAX_DIMS, 0>(reshapeStride);
        auto reshapeStride2 = TileOp::GetTupleElement<Stride5Dim, DIM_3RD, MAX_DIMS, 0>(reshapeStride);

        if constexpr (TileOp::IsConstContinous<U>() == true) {
            auto dstGlobal = PtoGlobal<T, typename U::Shape, Stride5Dim>(
                                 TileOp::GetPackedGmAddr<typename T::Type>(dst.GetAddr(), gmOffset), src.GetShape(),
                                 reshapeStride)
                                 .Data();
            auto srcTile = PtoTile<U, pto::BLayout::RowMajor, true, void, false>().Data();
            pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
            pto::TSTORE(dstGlobal, srcTile);
            return;
        }

        auto srcTile = PtoTile<U>(src);
        auto dstGlobal = PtoGlobal<T, typename U::Shape, Stride5Dim, true>(src.GetShape(), reshapeStride);
        for (LoopVar index0 = 0; index0 < srcShape0; ++index0) {
            for (LoopVar index1 = 0; index1 < srcShape1; ++index1) {
                for (LoopVar index2 = 0; index2 < srcShape2; ++index2) {
                    auto dstOffset = gmOffset + index0 * reshapeStride0 + index1 * reshapeStride1 +
                                     index2 * reshapeStride2;
                    dstGlobal.Assign(TileOp::GetPackedGmAddr<typename T::Type>(dst.GetAddr(), dstOffset));
                    auto tileOffsets = TileOffset(index0, index1, index2);
                    srcTile.Assign(src, tileOffsets);
                    pto::TSTORE(dstGlobal.Data(), srcTile.Data());
                }
            }
        }
    }
}

template <bool copyIn, typename GlobalData, typename TileDefine>
__aicore__ inline void ProcessTransMove(GlobalData globalData, TileDefine ubData)
{
    if constexpr (copyIn) {
        pto::TLOAD(ubData, globalData);
    } else {
        pto::TSTORE(globalData, ubData);
    }
}

template <typename GMType, typename UBType, bool copyIn, int tileW>
__aicore__ inline void DoTransMove(size_t* srcShape, size_t gmShape4, size_t* gmStride, size_t* ubStride,
                                   GMType* gmAddr, size_t gmOffset, uint64_t ubAddr)
{
    using GlobalData = pto::GlobalTensor<GMType, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    for (LoopVar index0 = 0; index0 < srcShape[0]; ++index0) {
        for (LoopVar index1 = 0; index1 < srcShape[1]; ++index1) {
            for (LoopVar index2 = 0; index2 < srcShape[2]; ++index2) {
                for (LoopVar index3 = 0; index3 < srcShape[3]; ++index3) {
                    GlobalData globalData(gmAddr + gmOffset + index0 * gmStride[0] + index1 * gmStride[1] +
                                              index2 * gmStride[2] + index3 * gmStride[3],
                                          pto::Shape(1, 1, 1, 1, gmShape4), pto::Stride(0, 0, 0, 0, 0));
                    using TileDefine = pto::Tile<pto::TileType::Vec, UBType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    TileDefine ubData(1, srcShape[4]);
                    auto ubOffset = index0 * ubStride[0] + index1 * ubStride[1] + index2 * ubStride[2] +
                                    index3 * ubStride[3];
                    pto::TASSIGN(ubData, ubAddr + ubOffset * sizeof(UBType));
                    ProcessTransMove<copyIn>(globalData, ubData);
                }
            }
        }
    }
}

template <unsigned axis0, unsigned axis1, bool copyIn, typename GM, typename UB, typename C>
__aicore__ inline void CallTransMove(GM gm, UB ub, C coordinate)
{
    static_assert(axis0 != 4 && axis1 != 4);
    constexpr auto shapeSize = Std::tuple_size<typename UB::Shape>::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename UB::TileShape>::type::value;
    const auto gmLayout = gm.GetLayout();
    size_t gmShape4 = static_cast<size_t>(gmLayout.template GetShapeDim<4, 5>());
    size_t gmStride[] = {static_cast<size_t>(gmLayout.template GetStrideDim<0, 5>()),
                         static_cast<size_t>(gmLayout.template GetStrideDim<1, 5>()),
                         static_cast<size_t>(gmLayout.template GetStrideDim<2, 5>()),
                         static_cast<size_t>(gmLayout.template GetStrideDim<3, 5>())};
    size_t gmOffset = static_cast<size_t>(gmLayout.template GetGmOffset<C, 5>(coordinate));
    const auto ubLayout = ub.GetLayout();
    size_t srcShape[] = {static_cast<size_t>(ubLayout.template GetShapeDim<0, 5>()),
                         static_cast<size_t>(ubLayout.template GetShapeDim<1, 5>()),
                         static_cast<size_t>(ubLayout.template GetShapeDim<2, 5>()),
                         static_cast<size_t>(ubLayout.template GetShapeDim<3, 5>()),
                         static_cast<size_t>(ubLayout.template GetShapeDim<4, 5>())};
    size_t ubStride[] = {static_cast<size_t>(ubLayout.template GetStrideDim<0, 5>()),
                         static_cast<size_t>(ubLayout.template GetStrideDim<1, 5>()),
                         static_cast<size_t>(ubLayout.template GetStrideDim<2, 5>()),
                         static_cast<size_t>(ubLayout.template GetStrideDim<3, 5>())};
    auto exchangeAxis = [](size_t* arr) {
        auto tmp = arr[axis0];
        arr[axis0] = arr[axis1];
        arr[axis1] = tmp;
    };
    if constexpr (copyIn) {
        exchangeAxis(ubStride);
        exchangeAxis(srcShape);
    } else {
        exchangeAxis(gmStride);
    }
    using GMType = std::conditional_t<std::is_same_v<typename GM::Type, bool>, uint8_t, typename GM::Type>;
    using UBType = std::conditional_t<std::is_same_v<typename UB::Type, bool>, uint8_t, typename UB::Type>;
    DoTransMove<GMType, UBType, copyIn, tileW>(srcShape, gmShape4, gmStride, ubStride, gm.GetAddr(), gmOffset,
                                               ub.GetAddr());
}

#define OP_TILE_OP_TRANSPOSE_MOVEIN TTransMoveIn
template <unsigned axis0, unsigned axis1, typename DST, typename SRC, typename C>
__aicore__ inline void TTransMoveIn(DST dst, SRC src, C coordinate)
{
    static_assert(DST::FORMAT == Hardware::UB && SRC::FORMAT == Hardware::GM);
    CallTransMove<axis0, axis1, true>(src, dst, coordinate);
}

#define OP_TILE_OP_TRANSPOSE_MOVEOUT TTransMoveOut
template <unsigned axis0, unsigned axis1, typename DST, typename SRC, typename C>
__aicore__ inline void TTransMoveOut(DST dst, SRC src, C coordinate)
{
    static_assert(DST::FORMAT == Hardware::GM && SRC::FORMAT == Hardware::UB);
    CallTransMove<axis0, axis1, false>(dst, src, coordinate);
}

template <size_t current, size_t target = 3>
__aicore__ inline size_t IndexPutGetStride(size_t arr[])
{
    size_t result = 1;
    for (size_t i = current; i < target; ++i) {
        result *= arr[i];
    }
    return result;
}

template <size_t dstShapeSize, size_t indicesSize, typename IndicesPtr>
__aicore__ inline uint64_t GetIndexPutDstOffset(size_t dstShapes[], IndicesPtr indicesPtrs[], size_t i)
{
    uint64_t dstOffset = 0;
    if constexpr (indicesSize >= 1) {
        dstOffset += indicesPtrs[0][i] * IndexPutGetStride<4 - dstShapeSize>(dstShapes);
    }
    if constexpr (indicesSize >= 2) {
        dstOffset += indicesPtrs[1][i] * IndexPutGetStride<5 - dstShapeSize>(dstShapes);
    }
    if constexpr (indicesSize >= 3) {
        dstOffset += indicesPtrs[2][i] * IndexPutGetStride<6 - dstShapeSize>(dstShapes);
    }
    if constexpr (indicesSize >= 4) {
        dstOffset += indicesPtrs[3][i] * IndexPutGetStride<7 - dstShapeSize>(dstShapes);
    }
    return dstOffset;
}

template <pto::AtomicType atomicType, size_t dstShapeSize, size_t valuesSize, typename VAL, typename DstType,
          typename IndicesPtr>
__aicore__ inline void DoIndexPut(size_t indicesLength, size_t dstShapes[], size_t valuesStride, size_t valuesShapes[],
                                  uint64_t valuesAddr, DstType* dstAddr, IndicesPtr indicesPtrs[])
{
    using ValuesDtype = std::conditional_t<std::is_same_v<typename VAL::Type, bool>, uint8_t, typename VAL::Type>;
    constexpr auto tileW = Std::tuple_element<valuesSize - 1, typename VAL::TileShape>::type::value;
    using ValuesTileDefine = pto::Tile<pto::TileType::Vec, ValuesDtype, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    using DstData = pto::GlobalTensor<DstType, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    size_t copyShapes[] = {1, 1, valuesShapes[2]};
    size_t copyStrides[] = {0, dstShapes[2]};
    if constexpr (valuesSize == 1) {
        copyShapes[2] = 1;
    } else if constexpr (valuesSize == 3) {
        copyShapes[1] = valuesShapes[1];
    } else if constexpr (valuesSize == 4) {
        copyShapes[0] = valuesShapes[0];
        copyShapes[1] = valuesShapes[1];
        copyStrides[0] = dstShapes[1] * dstShapes[2];
    }
    auto dstDataShape = pto::Shape(1, 1, copyShapes[0], copyShapes[1], copyShapes[2]);
    auto dstDataStride = pto::Stride(0, 0, copyStrides[0], copyStrides[1], 0);
    auto valuesPtr = (__ubuf__ ValuesDtype*)valuesAddr;
    ValuesDtype valuesOrigin;
    if constexpr (valuesSize == 1) {
        valuesOrigin = *valuesPtr;
    }
    for (LoopVar i = 0; i < indicesLength; ++i) {
        uint64_t dstOffset = GetIndexPutDstOffset<dstShapeSize, dstShapeSize - valuesSize + 1>(dstShapes, indicesPtrs,
                                                                                               i);
        if constexpr (valuesSize == 1) {
            valuesPtr[0] = valuesPtr[i];
        }
        ValuesTileDefine valuesData(copyShapes[1], copyShapes[2]);
        DstData dstData(dstAddr + dstOffset, dstDataShape, dstDataStride);
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        pto::TASSIGN(valuesData, (uint64_t)valuesPtr);
        pto::TSTORE<ValuesTileDefine, DstData, atomicType>(dstData, valuesData);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        valuesPtr += valuesStride;
    }
    if constexpr (valuesSize == 1) {
        valuesPtr[0] = valuesOrigin;
    }
}

#define OP_TILE_OP_INDEXPUT TIndexPut
template <bool accumulate, size_t indicesSize, typename DST, typename C, typename VAL, typename IDX>
__aicore__ inline void TIndexPut(DST dst, C coordinate, VAL values, IDX indices0, IDX indices1, IDX indices2,
                                 IDX indices3)
{
    constexpr auto atomicType = accumulate ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone;
    constexpr auto dstShapeSize = Std::tuple_size<typename DST::Shape>::value;
    constexpr auto valuesSize = Std::tuple_size<typename VAL::Shape>::value;
    static_assert(dstShapeSize >= indicesSize && dstShapeSize <= 4 && dstShapeSize == valuesSize + indicesSize - 1);
    using IndicesDtype = typename IDX::Type;
    using IndicesPtr = __ubuf__ IndicesDtype*;
    IndicesPtr indicesPtrs[] = {(IndicesPtr)indices0.GetAddr(), (IndicesPtr)indices1.GetAddr(),
                                (IndicesPtr)indices2.GetAddr(), (IndicesPtr)indices3.GetAddr()};
    const auto indicesLayout = indices0.GetLayout();
    const auto dstLayout = dst.GetLayout();
    const auto valuesLayout = values.GetLayout();
    auto indicesLength = indicesLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    size_t dstShapes[] = {static_cast<size_t>(dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>()),
                          static_cast<size_t>(dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>()),
                          static_cast<size_t>(dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>())};
    size_t valuesShapes[] = {static_cast<size_t>(valuesLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>()),
                             static_cast<size_t>(valuesLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>()),
                             static_cast<size_t>(valuesLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>())};
    size_t valuesStride = static_cast<size_t>(valuesLayout.template GetStrideDim<MAX_DIMS - valuesSize, MAX_DIMS>());
    if constexpr (valuesSize == 1) {
        valuesStride = 0;
    }
    size_t gmOffset = static_cast<size_t>(dstLayout.template GetGmOffset<C, MAX_DIMS>(coordinate));
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    DoIndexPut<atomicType, dstShapeSize, valuesSize, VAL>(indicesLength, dstShapes, valuesStride, valuesShapes,
                                                          values.GetAddr(), dst.GetAddr() + gmOffset, indicesPtrs);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
}

#endif
