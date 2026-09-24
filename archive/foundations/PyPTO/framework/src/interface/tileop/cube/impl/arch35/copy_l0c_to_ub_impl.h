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
 * \file copy_l0c_to_ub_impl.h
 * \brief L0C to UB Data Transfer Interface Implementation (Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_ARCH35_COPY_L0C_TO_UB_IMPL__H
#define TILEOP_TILE_OPERATOR_ARCH35_COPY_L0C_TO_UB_IMPL__H

#include "../cube_utils.h"

template <typename config, typename ubTileData, typename l0cTileData, typename FbTileTensor>
TILEOP void TExtractL0CToUB(ubTileData& ubTile, l0cTileData& l0cTile, FbTileTensor& fixbuf, uint16_t l0cOffset0,
                            uint16_t l0cOffset1, int16_t subblockId, uint64_t scaleValue)
{
    constexpr bool supportedQuantMode = IsSupportedQuantMode<typename l0cTileData::DType, typename ubTileData::DType>();
    constexpr bool supportedBasicMode = IsSupportedBasicMode<typename l0cTileData::DType, typename ubTileData::DType>();
    if constexpr (supportedQuantMode) {
        if (scaleValue != 0) {
            constexpr pto::ReluPreMode relu_mode = (config::kReluMode == 0) ? pto::ReluPreMode::NoRelu :
                                                                              pto::ReluPreMode::NormalRelu;
            if (subblockId == 0) {
                pto::TEXTRACT<ubTileData, l0cTileData, pto::AccToVecMode::SingleModeVec0, relu_mode>(
                    ubTile, l0cTile, scaleValue, l0cOffset0, l0cOffset1);
            } else {
                pto::TEXTRACT<ubTileData, l0cTileData, pto::AccToVecMode::SingleModeVec1, relu_mode>(
                    ubTile, l0cTile, scaleValue, l0cOffset0, l0cOffset1);
            }
        } else {
            auto scaleData = CreateScaleTileData(fixbuf);
            pto::TASSIGN(scaleData, static_cast<uint64_t>(fixbuf.GetAddr()));
            if (subblockId == 0) {
                pto::TEXTRACT<ubTileData, l0cTileData, decltype(scaleData), pto::AccToVecMode::SingleModeVec0,
                              config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                    ubTile, l0cTile, scaleData, l0cOffset0, l0cOffset1);
            } else {
                pto::TEXTRACT<ubTileData, l0cTileData, decltype(scaleData), pto::AccToVecMode::SingleModeVec1,
                              config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                    ubTile, l0cTile, scaleData, l0cOffset0, l0cOffset1);
            }
        }
    } else {
        if (subblockId == 0) {
            pto::TEXTRACT<ubTileData, l0cTileData, pto::AccToVecMode::SingleModeVec0,
                          config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                ubTile, l0cTile, l0cOffset0, l0cOffset1);
        } else {
            pto::TEXTRACT<ubTileData, l0cTileData, pto::AccToVecMode::SingleModeVec1,
                          config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                ubTile, l0cTile, l0cOffset0, l0cOffset1);
        }
    }
}

template <typename config, typename ubTileData, typename l0cTileData, typename FbTileTensor>
TILEOP void TInsertL0CToUB(ubTileData& ubTile, l0cTileData& l0cTile, FbTileTensor& fixbuf, uint16_t ubOffset0,
                           uint16_t ubOffset1, int16_t subblockId, uint64_t scaleValue)
{
    constexpr bool supportedQuantMode = IsSupportedQuantMode<typename l0cTileData::DType, typename ubTileData::DType>();
    constexpr bool supportedBasicMode = IsSupportedBasicMode<typename l0cTileData::DType, typename ubTileData::DType>();
    if constexpr (supportedQuantMode) {
        if (scaleValue != 0) {
            constexpr pto::ReluPreMode relu_mode = (config::kReluMode == 0) ? pto::ReluPreMode::NoRelu :
                                                                              pto::ReluPreMode::NormalRelu;
            if (subblockId == 0) {
                pto::TINSERT<ubTileData, l0cTileData, pto::AccToVecMode::SingleModeVec0, relu_mode>(
                    ubTile, l0cTile, scaleValue, ubOffset0, ubOffset1);
            } else {
                pto::TINSERT<ubTileData, l0cTileData, pto::AccToVecMode::SingleModeVec1, relu_mode>(
                    ubTile, l0cTile, scaleValue, ubOffset0, ubOffset1);
            }
        } else {
            auto scaleData = CreateScaleTileData(fixbuf);
            pto::TASSIGN(scaleData, static_cast<uint64_t>(fixbuf.GetAddr()));
            if (subblockId == 0) {
                pto::TINSERT<ubTileData, l0cTileData, decltype(scaleData), pto::AccToVecMode::SingleModeVec0,
                             config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                    ubTile, l0cTile, scaleData, ubOffset0, ubOffset1);
            } else {
                pto::TINSERT<ubTileData, l0cTileData, decltype(scaleData), pto::AccToVecMode::SingleModeVec1,
                             config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                    ubTile, l0cTile, scaleData, ubOffset0, ubOffset1);
            }
        }
    } else {
        if (subblockId == 0) {
            pto::TINSERT<ubTileData, l0cTileData, pto::AccToVecMode::SingleModeVec0,
                         config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                ubTile, l0cTile, ubOffset0, ubOffset1);
        } else {
            pto::TINSERT<ubTileData, l0cTileData, pto::AccToVecMode::SingleModeVec1,
                         config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                ubTile, l0cTile, ubOffset0, ubOffset1);
        }
    }
}

// Copy data from L0C to UB dualdst
template <DualDstMode dualDstMode, typename ubTileData, typename l0cTileData>
INLINE void TExtractL0CToUBDualDst(ubTileData& ubTile, l0cTileData& l0cTile, uint16_t l0cOffset0, uint16_t l0cOffset1)
{
    constexpr bool supportedQuantMode = IsSupportedQuantMode<typename l0cTileData::DType, typename ubTileData::DType>();
    constexpr bool supportedBasicMode = IsSupportedBasicMode<typename l0cTileData::DType, typename ubTileData::DType>();
    static_assert(
        !supportedQuantMode && !supportedBasicMode,
        "[TCopyL0C2UBDualDst Error]: DualDst mode in L0C2UB does not support fixpipe online quant functionality.");
    if constexpr (dualDstMode == DualDstMode::DUAL_DST_SPLIT_M) {
        pto::TEXTRACT<ubTileData, l0cTileData, pto::AccToVecMode::DualModeSplitM>(ubTile, l0cTile, l0cOffset0,
                                                                                  l0cOffset1);
    } else {
        pto::TEXTRACT<ubTileData, l0cTileData, pto::AccToVecMode::DualModeSplitN>(ubTile, l0cTile, l0cOffset0,
                                                                                  l0cOffset1);
    }
}

// Copy data from L0C to UB
template <typename config, CopyMode mode, DualDstMode dualDstMode, typename Coord, typename DstTileTensor,
          typename SrcTileTensor, typename FbTileTensor>
INLINE void TCopyL0C2UBImpl(DstTileTensor& dst, SrcTileTensor& src, FbTileTensor& fixbuf, const Coord& dstCoord,
                            const Coord& srcCoord, int16_t subblockId, uint64_t scaleValue = 0)
{
    static_assert(mode != CopyMode::UNKNOWN,
                  "[TCopyL0C2UB Error]: Current CopyMode is UNKNOWN. CopyMode only support EXTRACT, INSERT and MOVE");
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileTensor::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename SrcTileTensor::Type);
    int64_t dstOffset0 = TileOp::GetTupleElement<Coord, 0, SHAPE_DIM2, 0>(dstCoord);
    int64_t dstOffset1 = TileOp::GetTupleElement<Coord, 1, SHAPE_DIM2, 0>(dstCoord);
    int64_t srcOffset0 = TileOp::GetTupleElement<Coord, 0, SHAPE_DIM2, 0>(srcCoord);
    int64_t srcOffset1 = TileOp::GetTupleElement<Coord, 1, SHAPE_DIM2, 0>(srcCoord);
    constexpr auto
        staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstTileTensor::TileShape>::type::value;
    constexpr auto staticUBW = Std::tuple_element<shapeSize - 1, typename DstTileTensor::TileShape>::type::value;
    constexpr auto
        staticL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename SrcTileTensor::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSize - 1, typename SrcTileTensor::TileShape>::type::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    using ubTileData = pto::Tile<pto::TileType::Vec, typename DstTileTensor::Type, staticUBH, staticUBW,
                                 config::kMode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor,
                                 -1, -1,
                                 config::kMode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
    using l0cTileData = pto::Tile<pto::TileType::Acc, typename SrcTileTensor::Type, staticL0CH, staticL0CW,
                                  pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, pto::TileConfig::fractalCSize,
                                  pto::PadValue::Null, pto::CompactMode::Normal>;
    ubTileData ubTile(dstShape0, dstShape1);
    l0cTileData l0cTile(srcShape0, srcShape1);
    pto::TASSIGN(ubTile, (uint64_t)dst.GetAddr());
    pto::TASSIGN(l0cTile, (uint64_t)src.GetAddr());
    if constexpr (dualDstMode == DualDstMode::DUAL_DST_DISABLE) {
        if constexpr (mode == CopyMode::EXTRACT || mode == CopyMode::MOVE) {
            TExtractL0CToUB<config, ubTileData, l0cTileData, FbTileTensor>(ubTile, l0cTile, fixbuf, srcOffset0,
                                                                           srcOffset1, subblockId, scaleValue);
        } else if (mode == CopyMode::INSERT) {
            TInsertL0CToUB<config, ubTileData, l0cTileData, FbTileTensor>(ubTile, l0cTile, fixbuf, dstOffset0,
                                                                          dstOffset1, subblockId, scaleValue);
        }
    } else {
        TExtractL0CToUBDualDst<dualDstMode>(ubTile, l0cTile, srcOffset0, srcOffset1);
    }
}
#endif // TILEOP_TILE_OPERATOR_ARCH35_COPY_L0C_TO_UB_IMPL__H
