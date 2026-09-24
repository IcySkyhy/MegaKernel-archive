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
 * \file conv_offset_utils.h
 * \brief Calculate load/store GM offset for conv with NCHW/NCDHW format (Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_OFFSET_UTILS__H
#define TILEOP_TILE_OPERATOR_CONV_OFFSET_UTILS__H
#include "../conv_utils.h"

/**
 * Calculate load GM offset for input/weight with NCHW format.
 * shapeInfo: input -> [orgCi, orgHi, orgWi], weight -> [orgCi, kh, kw]
 * shapeInfo: input -> [  0  ,   1  ,   2  ], weight -> [  0  , 1 , 2 ]
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_h_offset, weight -> 0 (conv2d 参数顺序调整：n, c, h, w, 0)
 * offset3: input -> src_w_offset, weight -> 0
 * offset4: input -> 0 (占位), weight -> 0
 * isFmap: true -> input, false -> weight
 */
template <bool isFmap>
INLINE int64_t CalLoadOffsetNCHW(const ShapeInfo& shapeInfo, const OffsetInfo& offsetInfo)
{
    if constexpr (isFmap) {
        int64_t inputOneBatchSize = shapeInfo.shape0 * shapeInfo.shape1 * shapeInfo.shape2;
        int64_t offsetC = offsetInfo.offset1 * shapeInfo.shape1 * shapeInfo.shape2;
        int64_t offsetH = offsetInfo.offset2 < 0 ? 0 : offsetInfo.offset2;
        offsetH = offsetInfo.offset2 > shapeInfo.shape1 ? shapeInfo.shape1 : offsetInfo.offset2;
        int64_t offsetW = offsetInfo.offset3 < 0 ? 0 : offsetInfo.offset3;
        offsetW = offsetInfo.offset3 > shapeInfo.shape2 ? shapeInfo.shape2 : offsetInfo.offset3;
        return offsetInfo.offset0 * inputOneBatchSize + offsetC + offsetH * shapeInfo.shape2 + offsetW;
    } else {
        return offsetInfo.offset0 * shapeInfo.shape0 * shapeInfo.shape1 * shapeInfo.shape2 +
               offsetInfo.offset1 * shapeInfo.shape1 * shapeInfo.shape2;
    }
}

/**
 * Calculate load GM offset for input/weight with NCDHW format.
 * shapeInfo: input -> [orgCi, orgDi, orgHi, orgWi], weight -> [orgCi, kd, kh, kw]
 * shapeInfo: input -> [  0  ,   1  ,   2  ,   3  ], weight -> [  0  , 1 , 2 , 3 ]
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_d_offset, weight -> src_d_offset
 * offset3: input -> src_h_offset, weight -> 0
 * offset4: input -> src_w_offset, weight -> 0
 * isFmap: true -> input, false -> weight
 */
template <bool isFmap>
INLINE int64_t CalLoadOffsetNCDHW(const ShapeInfo& shapeInfo, const OffsetInfo& offsetInfo)
{
    if constexpr (isFmap) {
        int64_t inputOneBatchSize = shapeInfo.shape0 * shapeInfo.shape1 * shapeInfo.shape2 * shapeInfo.shape3;
        int64_t offsetC = offsetInfo.offset1 * shapeInfo.shape1 * shapeInfo.shape2 * shapeInfo.shape3;
        int64_t offsetD = offsetInfo.offset2 * shapeInfo.shape2 * shapeInfo.shape3;
        int64_t offsetH = offsetInfo.offset3 < 0 ? 0 : offsetInfo.offset3;
        offsetH = offsetInfo.offset3 > shapeInfo.shape2 ? shapeInfo.shape2 : offsetInfo.offset3;
        int64_t offsetW = offsetInfo.offset4 < 0 ? 0 : offsetInfo.offset4;
        offsetW = offsetInfo.offset4 > shapeInfo.shape3 ? shapeInfo.shape3 : offsetInfo.offset4;
        return offsetInfo.offset0 * inputOneBatchSize + offsetC + offsetD + offsetH * shapeInfo.shape3 + offsetW;
    } else {
        int64_t khxkw = shapeInfo.shape2 * shapeInfo.shape3;
        int64_t kdxkhxkw = shapeInfo.shape1 * khxkw;
        return offsetInfo.offset0 * shapeInfo.shape0 * kdxkhxkw + offsetInfo.offset1 * kdxkhxkw +
               offsetInfo.offset2 * khxkw;
    }
}

/**
 * Calculate store GM offset with NZ -> NCHW format.
 * shapeInfo: [cout, dout, hout, wout]
 * shapeInfo: [  0 ,  1  ,  2  ,  3  ]
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_d_offset
 * offset3: dst_h_offset
 * offset4: dst_w_offset
 */
INLINE int64_t CalStoreOffsetNCDHW(const ShapeInfo& shapeInfo, const OffsetInfo& offsetInfo, const int64_t& loopH)
{
    int64_t outputOneBatchSize = shapeInfo.shape0 * shapeInfo.shape1 * shapeInfo.shape2 * shapeInfo.shape3;
    int64_t coutOffset = offsetInfo.offset1 * shapeInfo.shape1 * shapeInfo.shape2 * shapeInfo.shape3;
    int64_t doutOffset = offsetInfo.offset2 * shapeInfo.shape2 * shapeInfo.shape3;
    return offsetInfo.offset0 * outputOneBatchSize + coutOffset + doutOffset +
           (offsetInfo.offset3 + loopH) * shapeInfo.shape3 + offsetInfo.offset4;
}

#endif // TILEOP_TILE_OPERATOR_CONV_OFFSET_UTILS__H
