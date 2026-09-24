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
 * \file copy_l0c_to_gm_impl.h
 * \brief L0C to DDR Data Movement Interface Implementation (Atlas A3, Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_COPY_L0C_TO_GM_IMPL__H
#define TILEOP_TILE_OPERATOR_COPY_L0C_TO_GM_IMPL__H

#include "cube_utils.h"

template <typename config, typename globalData, typename tileData, typename FpTileData>
INLINE void TStoreExecute(globalData dstGlobal, tileData srcL0C, FpTileData& fixbuf, uint64_t scaleValue)
{
    constexpr bool supportedQuantMode = (std::is_same<typename tileData::DType, int32_t>::value &&
                                         std::is_same<typename globalData::DType, __gm__ half>::value) ||
                                        std::is_same<typename globalData::DType, __gm__ int8_t>::value;
#ifdef __LITE_NPU
    constexpr bool supportedBasicMode = (std::is_same<typename tileData::DType, int32_t>::value &&
                                         std::is_same<typename globalData::DType, __gm__ int32_t>::value) ||
                                        (std::is_same<typename tileData::DType, half>::value &&
                                         std::is_same<typename globalData::DType, __gm__ half>::value);
#else
    constexpr bool supportedBasicMode = (std::is_same<typename tileData::DType, int32_t>::value &&
                                         std::is_same<typename globalData::DType, __gm__ int32_t>::value) ||
                                        (std::is_same<typename tileData::DType, float>::value &&
                                         std::is_same<typename globalData::DType, __gm__ half>::value) ||
                                        (std::is_same<typename tileData::DType, float>::value &&
                                         std::is_same<typename globalData::DType, __gm__ bfloat16_t>::value) ||
                                        (std::is_same<typename tileData::DType, float>::value &&
                                         std::is_same<typename globalData::DType, __gm__ float>::value);
#endif
    if constexpr (supportedQuantMode) {
        // L0C->GM反量化场景
        if (scaleValue != 0) {
            constexpr bool sign = (std::is_same<typename globalData::DType, __gm__ int8_t>::value) ? true : false;
            uint64_t preQuantScalar = (scaleValue & ~(static_cast<uint64_t>(1) << 46)) |
                                      (static_cast<uint64_t>(sign) << 46);
            pto::TSTORE<tileData, globalData, config::kIsAcc ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone,
                        config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstGlobal, srcL0C, preQuantScalar);
        } else {
            constexpr uint64_t shapeSize = Std::tuple_size<typename FpTileData::Shape>::value;
            constexpr auto
                tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename FpTileData::TileShape>::type::value;
            constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename FpTileData::TileShape>::type::value;
            using fpTileData = pto::Tile<pto::TileType::Scaling, uint64_t, tileH, tileW, pto::BLayout::RowMajor, -1,
                                         -1>;
            int64_t fixShape0 = GetShape<0>(fixbuf);
            int64_t fixShape1 = GetShape<1>(fixbuf);
            if (fixShape0 == 0 || fixShape1 == 0) {
                return;
            }
            fpTileData fpData(fixShape0, fixShape1);
            pto::TASSIGN(fpData, static_cast<uint64_t>(fixbuf.GetAddr()));
            pto::TSTORE_FP<tileData, globalData, fpTileData,
                           config::kIsAcc ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone,
                           config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstGlobal, srcL0C, fpData);
        }
    } else if constexpr (supportedBasicMode) {
        // L0C->GM普通场景
        pto::TSTORE<tileData, globalData, config::kIsAcc ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone,
                    config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(dstGlobal,
                                                                                                      srcL0C);
    } else {
        static_assert(supportedQuantMode,
                      "When L0C data type is int32, only fp32->half conversion in CAST_QUANT_PRE mode is supported.");
    }
}

// Copy data from L0C to DDR with NZ -> ND format
template <typename config, typename GlobalData, typename TileData, typename FpTileData>
INLINE void TStoreNZ2ND(GlobalData& dst, TileData& src, FpTileData& fixbuf, const int64_t& offset0,
                        const int64_t& offset1, uint64_t scaleValue = 0)
{
    constexpr auto shapeSize = Std::tuple_size<typename GlobalData::Shape>::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t dstStride0 = GetStride<0>(dst);
    int64_t dstStride1 = GetStride<1>(dst);
    constexpr auto tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename TileData::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    int64_t gmOffset = offset1 + offset0 * dstShape1;
    using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    using tileData = pto::Tile<pto::TileType::Acc, typename TileData::Type, tileH, tileW, pto::BLayout::ColMajor, -1,
                               -1, pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null,
                               pto::CompactMode::Normal>;
    globalData dstGlobal((__gm__ typename GlobalData::Type*)(dst.GetAddr() + gmOffset),
                         pto::Shape<1, 1, 1, -1, -1>(srcShape0, srcShape1),
                         pto::Stride<1, 1, 1, -1, -1>(dstStride0, dstStride1));
    tileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, static_cast<uint64_t>(src.GetAddr()));
    TStoreExecute<config, globalData, tileData, FpTileData>(dstGlobal, srcL0C, fixbuf, scaleValue);
}

// Copy data from L0C to DDR with NZ -> NZ format
template <typename config, typename GlobalData, typename TileData, typename FpTileData>
INLINE void TStoreNZ2NZ(GlobalData& dst, TileData& src, FpTileData& fixbuf, const int64_t& offset0,
                        const int64_t& offset1, const int64_t& curH, const int64_t& curW, uint64_t scaleValue = 0)
{
    constexpr auto shapeSize = Std::tuple_size<typename GlobalData::Shape>::value;
    constexpr int64_t c0Size = std::is_same<typename TileData::Type, int32_t>::value ?
                                   BLOCK_CUBE_M_N :
                                   BLOCK_ALIGN_BYTE / sizeof(typename GlobalData::Type);
    int64_t dstShape0 = curH;
    int64_t dstShape1 = curW;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    constexpr auto tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename TileData::TileShape>::type::value;
    int64_t gmOffset = CalNZOffset(dstShape0, dstShape1, offset0, offset1, c0Size);
    using shapeDim2 = pto::Shape<1, -1, -1, BLOCK_CUBE_M_N, c0Size>;
    using strideDim2 = pto::Stride<-1, -1, -1, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim2, pto::Layout::NZ>;
    globalData dstGlobal((__gm__ typename GlobalData::Type*)(dst.GetAddr() + gmOffset),
                         shapeDim2(dstShape1 / c0Size, dstShape0 / BLOCK_CUBE_M_N),
                         strideDim2(dstShape0 * dstShape1, dstShape0 * c0Size, BLOCK_CUBE_M_N * c0Size));
    using tileData = pto::Tile<pto::TileType::Acc, typename TileData::Type, tileH, tileW, pto::BLayout::ColMajor, -1,
                               -1, pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null,
                               pto::CompactMode::Normal>;
    tileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, static_cast<uint64_t>(src.GetAddr()));
    TStoreExecute<config, globalData, tileData, FpTileData>(dstGlobal, srcL0C, fixbuf, scaleValue);
}

// L1 spill(Only used in deepseek model)
// When L1 space is insufficient, spill to GM. (Supported on A2/A3 only.)
template <typename config, typename Coord, typename GlobalData, typename TileData>
INLINE void TStoreL1SpillImpl(GlobalData& dst, TileData& src, const Coord& coord)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename TileData::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t dstStride0 = GetStride<0>(dst);
    int64_t dstStride1 = GetStride<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    static_assert(config::kMode == CopyOutMode::ND2ND, "Only ND2ND mode is implemented in L1 spill.");
    using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    using tileData = pto::Tile<pto::TileType::Mat, typename TileData::Type, staticL1H, staticL1W,
                               pto::BLayout::RowMajor, -1, -1>;
    globalData dstGlobal((__gm__ typename GlobalData::Type*)(dst.GetAddr()), shapeDim2(srcShape0, srcShape1),
                         strideDim2(dstStride0, dstStride1));
    tileData srcL1(srcShape0, srcShape1);
    pto::TASSIGN(srcL1, static_cast<uint64_t>(src.GetAddr()));
    pto::TSTORE<tileData, globalData>(dstGlobal, srcL1);
}

// Copy data from L0C to DDR with NZ -> ND format before reshape
template <typename config, typename GlobalTensor, typename TileTensor, typename FpTileTensor>
INLINE void TReshapeStoreNZ2ND(GlobalTensor& dst, TileTensor& src, FpTileTensor& fixbuf, const int64_t& offset0,
                               const int64_t& offset1, const int64_t& gShape0, const int64_t& gShape1,
                               uint64_t scaleValue = 0)
{
    constexpr auto shapeSize = Std::tuple_size<typename TileTensor::TileShape>::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    constexpr auto staticL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileTensor::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSize - 1, typename TileTensor::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    int64_t gmOffset = offset1 + offset0 * gShape1;
    using globalData = pto::GlobalTensor<typename GlobalTensor::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    using tileData = pto::Tile<pto::TileType::Acc, typename TileTensor::Type, staticL0CH, staticL0CW,
                               pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, pto::TileConfig::fractalCSize,
                               pto::PadValue::Null, pto::CompactMode::Normal>;
    globalData dstGlobal((__gm__ typename GlobalTensor::Type*)(dst.GetAddr() + gmOffset),
                         shapeDim2(srcShape0, srcShape1), strideDim2(gShape1, 1));
    tileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, static_cast<uint64_t>(src.GetAddr()));
    TStoreExecute<config, globalData, tileData, FpTileTensor>(dstGlobal, srcL0C, fixbuf, scaleValue);
}

#endif // TILEOP_TILE_OPERATOR_COPY_L0C_TO_GM_IMPL__H
