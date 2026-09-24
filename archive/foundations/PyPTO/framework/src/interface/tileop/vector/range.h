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
 * \file range.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_RANGE__H
#define TILEOP_TILE_OPERATOR_VEC_RANGE__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

const int32_t DEFAULT_REPEAT_STRIDE = 8;
const int32_t NUM_EIGHT = 8;
const int32_t ONE_BLK_SIZE = 32;

template <typename T, int Unit>
TILEOP inline void TRangePropagate(__ubuf__ T* base, int32_t loopN, int32_t tailSize, T offset)
{
    if (loopN > 0) {
        using VecTile = pto::Tile<pto::TileType::Vec, T, 1, Unit, pto::BLayout::RowMajor, -1, -1>;
        for (LoopVar i = 0; i < loopN; ++i) {
            VecTile src(1, Unit), dst(1, Unit);
            pto::TASSIGN(src, reinterpret_cast<uint64_t>(base + i * Unit));
            pto::TASSIGN(dst, reinterpret_cast<uint64_t>(base + (i + 1) * Unit));
            pto::TADDS(dst, src, offset);
#ifdef __DAV_V220
            pipe_barrier(PIPE_V);
#else
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
#endif
        }
    }
    if (tailSize > 0) {
        using TailTile = pto::Tile<pto::TileType::Vec, T, 1, Unit, pto::BLayout::RowMajor, -1, -1>;
        TailTile src(1, tailSize), dst(1, tailSize);
        pto::TASSIGN(src, reinterpret_cast<uint64_t>(base + loopN * Unit));
        pto::TASSIGN(dst, reinterpret_cast<uint64_t>(base + (loopN + 1) * Unit));
        pto::TADDS(dst, src, offset);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
    }
}

template <typename TileType>
TILEOP void TRange(TileType dst, unsigned size, typename TileType::Type start, typename TileType::Type step,
                   int64_t tileIdx)
{
    using T = typename TileType::Type;
    const T baseStart = start + step * static_cast<T>(tileIdx);
    constexpr int32_t kBlkElems = ONE_BLK_SIZE / sizeof(T);
    constexpr int32_t kRepElems = (ONE_BLK_SIZE * DEFAULT_REPEAT_STRIDE) / sizeof(T);

    __ubuf__ T* dst_ptr = reinterpret_cast<__ubuf__ T*>(dst.GetAddr());
    unsigned N = size;

    // block 1
    if (N <= static_cast<unsigned>(kBlkElems)) {
        for (int32_t j = 0; j < static_cast<int32_t>(N); ++j) {
            dst_ptr[j] = baseStart + step * static_cast<T>(j);
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        return;
    }

    for (int32_t j = 0; j < kBlkElems; ++j) {
        dst_ptr[j] = baseStart + step * static_cast<T>(j);
    }

    // blocks 2~8
    int32_t loopN = 0, tailSize = 0;
    if (N >= static_cast<unsigned>(kRepElems)) {
        loopN = DEFAULT_REPEAT_STRIDE - 1;
    } else {
        loopN = static_cast<int32_t>(N) / kBlkElems - 1;
        tailSize = static_cast<int32_t>(N) % kBlkElems;
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    TRangePropagate<T, ONE_BLK_SIZE / sizeof(T)>(dst_ptr, loopN, tailSize, step * static_cast<T>(kBlkElems));

    if (N <= static_cast<unsigned>(kRepElems)) {
        return;
    }

    // repeat
    loopN = static_cast<int32_t>(N) / kRepElems - 1;
    tailSize = static_cast<int32_t>(N) % kRepElems;
    TRangePropagate<T, (ONE_BLK_SIZE * DEFAULT_REPEAT_STRIDE) / sizeof(T)>(dst_ptr, loopN, tailSize,
                                                                           step * static_cast<T>(kRepElems));
}
#endif
