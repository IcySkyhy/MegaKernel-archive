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
 * \file copy_l0c_to_l1_impl.h
 * \brief L0C to L1 Large to Small and Small to Large Data Movement Interface Implementation (Atlas A3, Ascend
 * 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_COPY_L0C_TO_L1_IMPL__H
#define TILEOP_TILE_OPERATOR_COPY_L0C_TO_L1_IMPL__H

#include "cube_utils.h"

// Copy data from L0C to L1(Extract)
template <typename config, typename l1Data, typename l0cData, typename FpTileData>
TILEOP void TExtractL0CToL1(l1Data& dstL1, l0cData& srcL0C, FpTileData& fixbuf, uint16_t l0cOffset0,
                            uint16_t l0cOffset1, uint64_t scaleValue = 0)
{
    constexpr bool supportedQuantMode = std::is_same<typename l0cData::DType, int32_t>::value &&
                                        std::is_same<typename l1Data::DType, half>::value;
    constexpr bool supportedBasicMode = (std::is_same<typename l0cData::DType, float>::value &&
                                         std::is_same<typename l1Data::DType, half>::value) ||
                                        (std::is_same<typename l0cData::DType, float>::value &&
                                         std::is_same<typename l1Data::DType, bfloat16_t>::value);
    if constexpr (supportedQuantMode) {
        // L0C->L1大搬小反量化场景
        if (scaleValue != 0) {
            constexpr pto::ReluPreMode relu_mode = (config::kReluMode == 0) ? pto::ReluPreMode::NoRelu :
                                                                              pto::ReluPreMode::NormalRelu;
            pto::TEXTRACT<l1Data, l0cData, relu_mode>(dstL1, srcL0C, scaleValue, l0cOffset0, l0cOffset1);
        } else {
            auto scaleData = CreateScaleTileData(fixbuf);
            pto::TASSIGN(scaleData, static_cast<uint64_t>(fixbuf.GetAddr()));
            pto::TEXTRACT_FP<l1Data, l0cData, decltype(scaleData),
                             config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstL1, srcL0C, scaleData, l0cOffset0, l0cOffset1);
        }
    } else if constexpr (supportedBasicMode) {
        // L0C->L1大搬小普通场景
        pto::TEXTRACT<l1Data, l0cData,
                      config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
            dstL1, srcL0C, l0cOffset0, l0cOffset1);
    } else {
        static_assert(supportedQuantMode,
                      "When L0C data type is int32, only fp32->half conversion in CAST_QUANT_PRE mode is supported.");
    }
}

// Copy data from L0C to L1(Insert)
template <typename config, typename l1Data, typename l0cData, typename FpTileData>
TILEOP void TInsertL0CToL1(l1Data& dstL1, l0cData& srcL0C, FpTileData& fixbuf, uint16_t l1Offset0, uint16_t l1Offset1,
                           uint64_t scaleValue = 0)
{
    constexpr bool supportedBasicMode = (std::is_same<typename l0cData::DType, float>::value &&
                                         std::is_same<typename l1Data::DType, half>::value) ||
                                        (std::is_same<typename l0cData::DType, float>::value &&
                                         std::is_same<typename l1Data::DType, bfloat16_t>::value);
    constexpr bool supportedQuantMode = std::is_same<typename l0cData::DType, int32_t>::value &&
                                        std::is_same<typename l1Data::DType, half>::value;
    if constexpr (supportedQuantMode) {
        // L0C->L1小搬大反量化场景
        if (scaleValue != 0) {
            constexpr pto::ReluPreMode relu_mode = (config::kReluMode == 0) ? pto::ReluPreMode::NoRelu :
                                                                              pto::ReluPreMode::NormalRelu;
            pto::TINSERT<l1Data, l0cData, relu_mode>(dstL1, srcL0C, scaleValue, l1Offset0, l1Offset1);
        } else {
            auto scaleData = CreateScaleTileData(fixbuf);
            pto::TASSIGN(scaleData, static_cast<uint64_t>(fixbuf.GetAddr()));
            pto::TINSERT_FP<l1Data, l0cData, decltype(scaleData),
                            config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstL1, srcL0C, scaleData, l1Offset0, l1Offset1);
        }
    } else if constexpr (supportedBasicMode) {
        // L0C->L1小搬大普通场景
        pto::TINSERT<l1Data, l0cData, config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
            dstL1, srcL0C, l1Offset0, l1Offset1);
    } else {
        static_assert(supportedQuantMode,
                      "When L0C data type is int32, only fp32->half conversion in CAST_QUANT_PRE mode is supported.");
    }
}

// Copy data from L0C to L1 with quantization ability
template <typename config, typename Coord, typename DstTileData, typename SrcTileData, typename FpTileData>
INLINE void TExtractL0C2L1Impl(DstTileData& dst, SrcTileData& src, FpTileData& fixbuf, const Coord& l1Coord,
                               const Coord& l0cCoord, uint64_t scaleValue = 0)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    int64_t l1Offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(l1Coord);
    int64_t l1Offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(l1Coord);
    int64_t l0cOffset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(l0cCoord);
    int64_t l0cOffset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(l0cCoord);
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename DstTileData::Type);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    constexpr int64_t
        tileL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstTileData::TileShape>::type::value;
    constexpr int64_t tileL1W = Std::tuple_element<shapeSize - 1, typename DstTileData::TileShape>::type::value;
    constexpr int64_t
        tileL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename SrcTileData::TileShape>::type::value;
    constexpr int64_t tileL0CW = Std::tuple_element<shapeSize - 1, typename SrcTileData::TileShape>::type::value;
    using l1TileData = pto::Tile<pto::TileType::Mat, typename DstTileData::Type, tileL1H, tileL1W,
                                 config::kMode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor,
                                 -1, -1,
                                 config::kMode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
    using l0cTileData = pto::Tile<pto::TileType::Acc, typename SrcTileData::Type, tileL0CH, tileL0CW,
                                  pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
    l1TileData dstL1(dstShape0, dstShape1);
    l0cTileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(dstL1, static_cast<uint64_t>(dst.GetAddr()));
    if (dstShape0 < srcShape0 || dstShape1 < srcShape1) {
        TExtractL0CToL1<config, l1TileData, l0cTileData, FpTileData>(dstL1, srcL0C, fixbuf, l0cOffset0, l0cOffset1,
                                                                     scaleValue);
    } else {
        TInsertL0CToL1<config, l1TileData, l0cTileData, FpTileData>(dstL1, srcL0C, fixbuf, l1Offset0, l1Offset1,
                                                                    scaleValue);
    }
}

#endif // TILEOP_TILE_OPERATOR_COPY_L0C_TO_L1_IMPL__H
