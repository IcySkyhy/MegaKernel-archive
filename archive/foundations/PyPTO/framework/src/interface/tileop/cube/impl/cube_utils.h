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
 * \file cube_utils.h
 * \brief Utility Functions and Constant Definitions
 */

#ifndef TILEOP_TILE_OPERATOR_CUBE_UTILS__H
#define TILEOP_TILE_OPERATOR_CUBE_UTILS__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

#ifdef __NPU_ARCH__
#if __NPU_ARCH__ == 3510
#define PTO_NPU_ARCH_A5
#endif
#endif

constexpr int16_t SHAPE_DIM2 = 2;
constexpr int16_t SHAPE_DIM3 = 3;
constexpr uint16_t BLOCK_CUBE_M_N = 16;
constexpr uint16_t BLOCK_ALIGN_BYTE = 32;
constexpr int64_t FP4_BLOCK_ALIGN_BYTE = 64;
constexpr int64_t MX_BLOCK_ALIGN_BYTE = 64;

template <CopyOutMode mode, bool isAcc, uint8_t reluMode>
struct TStoreConfig {
    static constexpr CopyOutMode kMode = mode;
    static constexpr bool kIsAcc = isAcc;
    static constexpr uint8_t kReluMode = reluMode;
};

template <int16_t idx, typename U>
INLINE int64_t GetShape(const U& tileTensor)
{
    static_assert(idx < SHAPE_DIM2, "Idx should be less than 2");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetShapeDim<idx>();
}

template <int16_t idx, typename U>
INLINE int64_t GetStride(const U& tileTensor)
{
    static_assert(idx < SHAPE_DIM2, "Idx should be less than 2");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetStrideDim<idx>();
}

INLINE int64_t CalNZOffset(const int64_t& srcShape0, const int64_t& srcShape1, const int64_t& offset0,
                           const int64_t& offset1, const int64_t& c0Size)
{
    int64_t batchSize = srcShape0 * srcShape1;
    int64_t offsetElem = offset1 + offset0 * srcShape1;
    int64_t batchIndex = offsetElem / batchSize;
    int64_t gmOffset = batchIndex * batchSize + (offset1 * srcShape0) + (offset0 - batchIndex * srcShape0) * c0Size;
    return gmOffset;
}

template <typename T>
constexpr INLINE bool CheckIsB4()
{
#if defined PTO_NPU_ARCH_A5
    return std::is_same<typename T::Type, float4_e2m1x2_t>::value ||
           std::is_same<typename T::Type, float4_e1m2x2_t>::value;
#else
    return false;
#endif
}

template <typename T, typename U>
INLINE bool CheckShapeValid(const T& dst, const U& src)
{
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    if (dstShape0 == 0 || dstShape1 == 0 || srcShape0 == 0 || srcShape1 == 0) {
        return false;
    }
    return true;
}

template <typename L0TileData>
INLINE bool CheckBASEMNValid(const L0TileData& l0Tile)
{
    int64_t l0Shape0 = GetShape<0>(l0Tile);
    int64_t l0Shape1 = GetShape<1>(l0Tile);
    if constexpr (L0TileData::FORMAT == Hardware::L0A) {
        return l0Shape0 > 0;
    } else {
        return l0Shape1 > 0;
    }
}

// create Scale Tile Data
template <typename FpTileData>
INLINE auto CreateScaleTileData(FpTileData& fixbuf)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename FpTileData::Shape>::value;
    constexpr int64_t
        scaleTileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename FpTileData::TileShape>::type::value;
    constexpr int64_t scaleTileW = Std::tuple_element<shapeSize - 1, typename FpTileData::TileShape>::type::value;
    int64_t scaleShape0 = GetShape<0>(fixbuf);
    int64_t scaleShape1 = GetShape<1>(fixbuf);
    using scaleTileData = pto::Tile<pto::TileType::Scaling, uint64_t, scaleTileH, scaleTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    return scaleTileData(scaleShape0, scaleShape1);
}

// Type compatibility check for quantization mode
template <typename l0cDType, typename ubDType>
constexpr INLINE bool IsSupportedQuantMode()
{
    return std::is_same<l0cDType, int32_t>::value && std::is_same<ubDType, half>::value;
}

// Type compatibility check for basic mode
template <typename l0cDType, typename ubDType>
constexpr INLINE bool IsSupportedBasicMode()
{
    return (std::is_same<l0cDType, float>::value && std::is_same<ubDType, half>::value) ||
           (std::is_same<l0cDType, float>::value && std::is_same<ubDType, bfloat16_t>::value);
}

// Handle K=0 scenario: fill L1 with zero and return early
// Returns true if early return is needed, false otherwise
template <CopyInMode copyMode, int64_t kIndex, typename TileData>
INLINE bool HandleZeroKScenario(TileData& dst)
{
    if constexpr (kIndex != -1) {
        static_assert(kIndex == 0 || kIndex == 1, "K index should be 0 or 1");
        if constexpr (copyMode != CopyInMode::ND2ND) {
            int64_t validK = GetShape<kIndex>(dst);
            if (validK == 0) {
                TFillPadZeroK(dst);
                return true; // Signal that early return is needed
            }
        }
    }
    return false; // No early return needed
}

#endif // TILEOP_TILE_OPERATOR_CUBE_UTILS__H
