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

#ifndef __LOGICALTENSOR_TILEOP_MTE__
#define __LOGICALTENSOR_TILEOP_MTE__

#include "tileop_common.h"

namespace TileOp {

template <typename T, unsigned T0, unsigned T1, unsigned UBS, unsigned GMS>
TILEOP void UBCopyIn(__ubuf__ T* dst, __gm__ T* src)
{
    constexpr uint16_t nBurst = T0;
    constexpr uint32_t lenBurst = T1 * sizeof(T);
    constexpr uint32_t gmGap = (GMS - T1) * sizeof(T);
    constexpr uint32_t blockSize = 32 / sizeof(T);
    constexpr uint32_t ubGap = (UBS - T1) / blockSize;
    static_assert(nBurst < ((1ULL << 12) - 1ULL));
    static_assert(lenBurst < ((1ULL << 21) - 1ULL));
    static_assert(gmGap < ((1ULL << 32) - 1ULL));
    if constexpr (T1 == 0) {
        return;
    }
    if constexpr (sizeof(T) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                 0 /*right padding count*/, gmGap, ubGap);
    } else if (sizeof(T) == 2) {
        copy_gm_to_ubuf_align_b16(dst, src, 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                  0 /*right padding count*/, gmGap, ubGap);
    } else {
        copy_gm_to_ubuf_align_b32(dst, src, 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                  0 /*right padding count*/, gmGap, ubGap);
    }
}

template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4, unsigned UBS1, unsigned UBS2,
          unsigned UBS3, unsigned UBS4, unsigned GMS1, unsigned GMS2, unsigned GMS3, unsigned GMS4>
TILEOP void UBCopyIn(__ubuf__ T* dst, __gm__ T* src)
{
    static_assert((UBS4 * sizeof(T)) % 32 == 0, "UB tile must be 32B aligned!");

    for (int i0 = 0; i0 < T0; i0++) {
        __gm__ T* src0 = src;
        __ubuf__ T* dst0 = dst;
        for (int i1 = 0; i1 < T1; i1++) {
            __gm__ T* src1 = src0;
            __ubuf__ T* dst1 = dst0;
            for (int i2 = 0; i2 < T2; i2++) {
                TileOp::UBCopyIn<T, T3, T4, UBS4, GMS4>(dst1, src1);
                src1 += GMS3 * GMS4;
                dst1 += UBS3 * UBS4;
            }
            src0 += GMS2 * GMS3 * GMS4;
            dst0 += UBS2 * UBS3 * UBS4;
        }
        src += GMS1 * GMS2 * GMS3 * GMS4;
        dst += UBS1 * UBS2 * UBS3 * UBS4;
    }
}

template <typename T, unsigned T0, unsigned T1, unsigned GMS, unsigned UBS>
TILEOP void UBCopyOut(__gm__ T* dst, __ubuf__ T* src)
{
    constexpr uint16_t nBurst = T0;
    constexpr uint32_t lenBurst = T1 * sizeof(T);
    constexpr uint32_t gmGap = (GMS - T1) * sizeof(T);
    constexpr uint32_t blockSize = 32 / sizeof(T);
    constexpr uint32_t ubGap = (UBS - T1) / blockSize;
    static_assert(nBurst < ((1ULL << 12) - 1ULL));
    static_assert(lenBurst < ((1ULL << 21) - 1ULL));
    static_assert(gmGap < ((1ULL << 32) - 1ULL));
    if constexpr (sizeof(T) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                 0 /*right padding count*/, ubGap, gmGap);
    } else if (sizeof(T) == 2) {
        copy_ubuf_to_gm_align_b16(dst, src, 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                  0 /*right padding count*/, ubGap, gmGap);
    } else {
        copy_ubuf_to_gm_align_b32(dst, src, 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                  0 /*right padding count*/, ubGap, gmGap);
    }
}

template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4, unsigned GMS1, unsigned GMS2,
          unsigned GMS3, unsigned GMS4, unsigned UBS1, unsigned UBS2, unsigned UBS3, unsigned UBS4>
TILEOP void UBCopyOut(__gm__ T* dst, __ubuf__ T* src)
{
    static_assert((UBS4 * sizeof(T)) % 32 == 0, "UB tile must be 32B aligned!");

    for (int i0 = 0; i0 < T0; i0++) {
        __gm__ T* dst0 = dst;
        __ubuf__ T* src0 = src;
        for (int i1 = 0; i1 < T1; i1++) {
            __gm__ T* dst1 = dst0;
            __ubuf__ T* src1 = src0;
            for (int i2 = 0; i2 < T2; i2++) {
                TileOp::UBCopyOut<T, T3, T4, GMS4, UBS4>(dst1, src1);
                dst1 += GMS3 * GMS4;
                src1 += UBS3 * UBS4;
            }
            dst0 += GMS2 * GMS3 * GMS4;
            src0 += UBS2 * UBS3 * UBS4;
        }
        dst += GMS1 * GMS2 * GMS3 * GMS4;
        src += UBS1 * UBS2 * UBS3 * UBS4;
    }
}

// NEXTNEXT: delete after pass resolve the problem of redundant copy in
template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4, unsigned UBS1, unsigned UBS2,
          unsigned UBS3, unsigned UBS4, unsigned GMS1, unsigned GMS2, unsigned GMS3, unsigned GMS4, unsigned isNop>
TILEOP void UBCopyIn(__ubuf__ T* dst, __gm__ T* src)
{
    // do nothing
}

template <typename T, typename T2, unsigned src0OriShape1, unsigned src1OriShape1, unsigned GmShape1,
          unsigned src0rawShape1, unsigned cacheMode, unsigned blockSize>
TILEOP void TIndexoutcast(__gm__ T* dst, __ubuf__ T* src0, __ubuf__ T2* src1)
{
    for (auto i = 0; i < src1OriShape1; i++) {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        T2 curValue = *(reinterpret_cast<__ubuf__ T2*>(src1 + i));
        if constexpr (cacheMode == 1) { // PA_NZ
            T2 blockCount = curValue / blockSize;
            T2 index = curValue % blockSize;

            __gm__ T* new_dst = dst + blockCount * blockSize * GmShape1 + index * 32 / sizeof(T);
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            copy_ubuf_to_gm(new_dst, src0 + i * src0OriShape1, 0 /*sid*/, src0OriShape1 / 32 * sizeof(T), 1, 0,
                            blockSize - 1);
        } else {
            __gm__ T* new_dst = dst + curValue * GmShape1;
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            TileOp::UBCopyOut<T, 1, src0OriShape1, GmShape1, src0rawShape1>(new_dst, src0 + i * src0rawShape1);
        }
    }
}

template <typename T, typename T2, unsigned src1OriShape0, unsigned src1OriShape1, unsigned src1rawShape1,
          unsigned src0OriShape3, unsigned src0rawShape1, unsigned src0rawShape3, unsigned cacheMode>
TILEOP void TIndexoutcast(__gm__ T* dst, __ubuf__ T* src, __ubuf__ T2* index)
{
    constexpr unsigned b = src1OriShape0;
    constexpr unsigned s1 = src1OriShape1;
    constexpr unsigned s1_32aligned = src1rawShape1; // 倒数第5个 4
    constexpr unsigned nd = src0OriShape3;
    constexpr unsigned nd_32aligned = src0rawShape3; // 倒数第七个 32

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    __gm__ T* curDst = dst;
    __ubuf__ T2* dstIdx = index;
    __ubuf__ T* curSrc = src;
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < s1; ++j) {
            curDst = dst + *dstIdx * nd; // dst [index[i][j]] [n][d]
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            copy_ubuf_to_gm_align_b32(curDst, curSrc, 0 /*sid*/, 1, nd * sizeof(T), 0, 0,
                                      (nd_32aligned - nd) * sizeof(T) / BLOCK_SIZE, 0);
            curSrc += nd_32aligned;
            dstIdx++;
        }
        curSrc += (src0rawShape1 - s1) * nd_32aligned;
        dstIdx += s1_32aligned - s1;
    }
}

// src1=index [1,2] , src0: [TShape0,TShape1,TShape2,TShape3],  dst [GmShape0,GmShape1,GmShape2,GmShape3]
template <typename T, typename T2, unsigned src0OriShape0, unsigned src0OriShape1, unsigned src0OriShape3,
          unsigned src0rawShape1, unsigned src0rawShape2, unsigned src0rawShape3, unsigned src1OriShape0,
          unsigned src1OriShape1, unsigned src1rawShape3, unsigned GmShape2, unsigned GmShape3, unsigned cacheMode,
          unsigned blockSize>
TILEOP void TIndexoutcast(__gm__ T* dst, __ubuf__ T* src0, __ubuf__ T2* src1)
{
    if (cacheMode == 2) {
        TIndexoutcast<T, T2, src1OriShape0, src1OriShape1, src1rawShape3, src0OriShape3, src0rawShape1, src0rawShape3,
                      cacheMode>(dst, src0, src1);
        return;
    }
    static_assert(src0OriShape1 == 1, "src0OriShape1 now only support 1");
    static_assert(blockSize != 0, "blockSize can not be zero");
    int alignTS2TS3 = src0rawShape2 * src0rawShape3; // ub需要32B对齐
    int alignSrc1 = src1rawShape3;                   // 修改对 src1 的大小计算
    for (int i = 0; i < src0OriShape0; ++i) {
        for (int j = 0; j < src0OriShape1; ++j) {
            TileOp::TIndexoutcast<T, T2, src0OriShape3, src1OriShape1, GmShape3, src0rawShape3, cacheMode, blockSize>(
                dst, src0, src1);
        }
        src0 += alignTS2TS3;
        src1 += alignSrc1;
        dst += GmShape2 * GmShape3;
    }
}
// DMA

template <typename T1, typename T2, int64_t rawShape1>
TILEOP void Load(__ubuf__ T1* dst, __gm__ T1* src, __ubuf__ T2* offsets, int64_t originShape0, int64_t originShape1)
{
    static_assert(std::is_same_v<T2, int32_t> || std::is_same_v<T2, int64_t>);
    pipe_barrier(PIPE_ALL);
    if (rawShape1 == originShape1) {
        int64_t total = originShape0 * originShape1;
        for (int64_t i = 0; i < total; i++) {
            dst[i] = src[offsets[i]];
        }
    } else {
        int64_t idx = 0;
        for (int64_t i = 0; i < originShape0; i++) {
            for (int64_t j = 0; j < originShape1; j++) {
                dst[idx] = src[offsets[idx]];
                idx++;
            }
            idx += rawShape1 - originShape1;
        }
    }
    pipe_barrier(PIPE_ALL);
}

template <typename T1, typename T2, int64_t rawShape1, int64_t rawShape2>
TILEOP void Load(__ubuf__ T1* dst, __gm__ T1* src, __ubuf__ T2* offsets, int64_t originShape0, int64_t originShape1,
                 int64_t originShape2)
{
    static_assert(std::is_same_v<T2, int32_t> || std::is_same_v<T2, int64_t>);
    pipe_barrier(PIPE_ALL);
    if (rawShape1 == originShape1 && rawShape2 == originShape2) {
        int64_t total = originShape0 * originShape1 * originShape2;
        for (int64_t i = 0; i < total; i++) {
            dst[i] = src[offsets[i]];
        }
    } else {
        int64_t idx = 0;
        for (int64_t i = 0; i < originShape0; i++) {
            for (int64_t j = 0; j < originShape1; j++) {
                for (int64_t k = 0; k < originShape2; k++) {
                    dst[idx] = src[offsets[idx]];
                    idx++;
                }
                idx += rawShape2 - originShape2;
            }
            idx += (rawShape1 - originShape1) * rawShape2;
        }
    }
    pipe_barrier(PIPE_ALL);
}

} // namespace TileOp

#endif
