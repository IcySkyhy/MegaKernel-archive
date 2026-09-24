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
 * \file mte_dyn.h
 * \brief
 */

#ifndef TILE_FWK_MTE_DYN_H
#define TILE_FWK_MTE_DYN_H

#include "tileop_common.h"

#include <type_traits>

namespace TileOp {
template <typename T, unsigned T0, unsigned T1, unsigned UBS>
TILEOP void UBCopyInBase(__ubuf__ T* dst, __gm__ T* src, unsigned GMS)
{
    constexpr uint16_t nBurst = T0;
    constexpr uint32_t lenBurst = T1 * sizeof(T);
    uint32_t gmGap = (GMS - T1) * sizeof(T);
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    constexpr uint32_t ubGap = (UBS - T1) / blockSize;
    // NEXTNEXT: Need to generalize large data scene in future
    static_assert(nBurst < ((1ULL << 12) - 1ULL));
    static_assert(lenBurst < ((1ULL << 21) - 1ULL));
    //    static_assert(gmGap < ((1ULL << 32) - 1ULL));
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

// for ub spill out scene
template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4, unsigned UBS1, unsigned UBS2,
          unsigned UBS3, unsigned UBS4>
TILEOP void DynUBCopyIn(__ubuf__ T* dst, __gm__ T* src, unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3,
                        unsigned GMS4)
{
    static_assert((UBS4 * sizeof(T)) % BLOCK_SIZE == 0, "UB tile must be 32B aligned!");

    for (int i0 = 0; i0 < T0; i0++) {
        __gm__ T* src0 = src;
        __ubuf__ T* dst0 = dst;
        for (int i1 = 0; i1 < T1; i1++) {
            __gm__ T* src1 = src0;
            __ubuf__ T* dst1 = dst0;
            for (int i2 = 0; i2 < T2; i2++) {
                TileOp::UBCopyInBase<T, T3, T4, UBS4>(dst1, src1, GMS4);
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

template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4, unsigned UBS1, unsigned UBS2,
          unsigned UBS3, unsigned UBS4>
TILEOP void DynUBCopyIn(__ubuf__ T* dst, __gm__ T* src, unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3,
                        unsigned GMS4, unsigned Offset0, unsigned Offset1, unsigned Offset2, unsigned Offset3,
                        unsigned Offset4)
{
    src += CalcLinearOffset(GMS1, GMS2, GMS3, GMS4, Offset0, Offset1, Offset2, Offset3, Offset4);

    static_assert((UBS4 * sizeof(T)) % BLOCK_SIZE == 0, "UB tile must be 32B aligned!");

    for (int i0 = 0; i0 < T0; i0++) {
        __gm__ T* src0 = src;
        __ubuf__ T* dst0 = dst;
        for (int i1 = 0; i1 < T1; i1++) {
            __gm__ T* src1 = src0;
            __ubuf__ T* dst1 = dst0;
            for (int i2 = 0; i2 < T2; i2++) {
                TileOp::UBCopyInBase<T, T3, T4, UBS4>(dst1, src1, GMS4);
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

template <typename T, unsigned T0, unsigned T1, unsigned UBS>
TILEOP void UBCopyOutBase(__gm__ T* dst, __ubuf__ T* src, unsigned GMS)
{
    constexpr uint16_t nBurst = T0;
    constexpr uint32_t lenBurst = T1 * sizeof(T);
    uint32_t gmGap = (GMS - T1) * sizeof(T);
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    constexpr uint32_t ubGap = (UBS - T1) / blockSize;
    // NEXTNEXT: Need to generalize large data scene in future
    static_assert(nBurst < ((1ULL << 12) - 1ULL));
    static_assert(lenBurst < ((1ULL << 21) - 1ULL));
    //    static_assert(gmGap < ((1ULL << 32) - 1ULL));
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

// for ub spill out scene
template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4, unsigned UBS1, unsigned UBS2,
          unsigned UBS3, unsigned UBS4>
TILEOP void DynUBCopyOut(__gm__ T* dst, __ubuf__ T* src, unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3,
                         unsigned GMS4)
{
    static_assert((UBS4 * sizeof(T)) % BLOCK_SIZE == 0, "UB tile must be 32B aligned!");

    for (int i0 = 0; i0 < T0; i0++) {
        __gm__ T* dst0 = dst;
        __ubuf__ T* src0 = src;
        for (int i1 = 0; i1 < T1; i1++) {
            __gm__ T* dst1 = dst0;
            __ubuf__ T* src1 = src0;
            for (int i2 = 0; i2 < T2; i2++) {
                TileOp::UBCopyOutBase<T, T3, T4, UBS4>(dst1, src1, GMS4);
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

template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4, unsigned UBS1, unsigned UBS2,
          unsigned UBS3, unsigned UBS4>
TILEOP void DynUBCopyOut(__gm__ T* dst, __ubuf__ T* src, unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3,
                         unsigned GMS4, unsigned Offset0, unsigned Offset1, unsigned Offset2, unsigned Offset3,
                         unsigned Offset4)
{
    dst += CalcLinearOffset(GMS1, GMS2, GMS3, GMS4, Offset0, Offset1, Offset2, Offset3, Offset4);

    static_assert((UBS4 * sizeof(T)) % BLOCK_SIZE == 0, "UB tile must be 32B aligned!");

    for (int i0 = 0; i0 < T0; i0++) {
        __gm__ T* dst0 = dst;
        __ubuf__ T* src0 = src;
        for (int i1 = 0; i1 < T1; i1++) {
            __gm__ T* dst1 = dst0;
            __ubuf__ T* src1 = src0;
            for (int i2 = 0; i2 < T2; i2++) {
                TileOp::UBCopyOutBase<T, T3, T4, UBS4>(dst1, src1, GMS4);
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

template <typename T, typename T2, unsigned src0OriShape1, unsigned src1OriShape1, unsigned src0rawShape1,
          unsigned cacheMode, unsigned blockSize>
TILEOP void TIndexoutcastBase(__gm__ T* dst, __ubuf__ T* src0, __ubuf__ T2* src1, unsigned GmShape1)
{
    for (auto i = 0; i < src1OriShape1; i++) {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        T2 curValue = *(reinterpret_cast<__ubuf__ T2*>(src1 + i));
        int64_t idxVal = static_cast<int64_t>(curValue);
        if (idxVal < 0) {
            continue;
        }
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
            TileOp::UBCopyOutBase<T, 1, src0OriShape1, src0rawShape1>(new_dst, src0 + i * src0rawShape1, GmShape1);
        }
    }
}

template <typename T, typename T2>
TILEOP void DynTIndexoutcast(__gm__ T* dst, __ubuf__ T* src, __ubuf__ T2* index, unsigned src1OriShape0,
                             unsigned src1OriShape1, unsigned src1rawShape1, unsigned src0OriShape3,
                             unsigned src0rawShape1, unsigned src0rawShape3)
{
    unsigned b = src1OriShape0;
    unsigned s1 = src1OriShape1;
    unsigned s1_32aligned = src1rawShape1;
    unsigned nd = src0OriShape3;
    unsigned nd_32aligned = src0rawShape3;

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    __gm__ T* curDst = dst;
    __ubuf__ T2* dstIdx = index;
    __ubuf__ T* curSrc = src;
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < s1; ++j) {
            int64_t idxVal = static_cast<int64_t>(*dstIdx);
            if (idxVal >= 0) {
                curDst = dst + *dstIdx * nd; // dst [index[i][j]] [n][d]
                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                copy_ubuf_to_gm_align_b32(curDst, curSrc, 0 /*sid*/, 1, nd * sizeof(T), 0, 0,
                                          (nd_32aligned - nd) * sizeof(T) / BLOCK_SIZE, 0);
                curSrc += nd_32aligned;
                dstIdx++;
            }
        }
        curSrc += (src0rawShape1 - s1) * nd_32aligned;
        dstIdx += s1_32aligned - s1;
    }
}

// src1=index [1,2] , src0: [TShape0,TShape1,TShape2,TShape3],  dst [GmShape0,GmShape1,GmShape2,GmShape3]
template <typename T, typename T2, unsigned src0OriShape0, unsigned src0OriShape1, unsigned src0OriShape3,
          unsigned src0rawShape1, unsigned src0rawShape2, unsigned src0rawShape3, unsigned src1OriShape0,
          unsigned src1OriShape1, unsigned src1rawShape3, unsigned cacheMode, unsigned blockSize>
TILEOP void DynTIndexoutcast(__gm__ T* dst, __ubuf__ T* src0, __ubuf__ T2* src1, unsigned GmShape0, unsigned GmShape1,
                             unsigned GmShape2, unsigned GmShape3, unsigned Offset0, unsigned Offset1, unsigned Offset2,
                             unsigned Offset3)
{
    if (src0OriShape0 == 0 || src0OriShape1 == 0 || src0OriShape3 == 0 || src1OriShape0 == 0 || src1OriShape1 == 0) {
        return;
    }
    if (cacheMode == 2) {
        DynTIndexoutcast<T, T2>(dst, src0, src1, src1OriShape0, src1OriShape1, src1rawShape3, src0OriShape3,
                                src0rawShape1, src0rawShape3);
        return;
    }
    dst += CalcLinearOffset(GmShape1, GmShape2, GmShape3, Offset0, Offset1, Offset2, Offset3);

    static_assert(src0OriShape1 == 1, "src0OriShape1 now only support 1");

    int alignTS2TS3 = src0rawShape2 * src0rawShape3; // ub需要32B对齐
    int alignSrc1 = src1rawShape3;                   // 修改对 src1 的大小计算
    for (int i = 0; i < src0OriShape0; ++i) {
        for (int j = 0; j < src0OriShape1; ++j) {
            TileOp::TIndexoutcastBase<T, T2, src0OriShape3, src1OriShape1, src0rawShape3, cacheMode, blockSize>(
                dst, src0, src1, GmShape3);
        }
        src0 += alignTS2TS3;
        src1 += alignSrc1;
        dst += GmShape2 * GmShape3;
    }
}

/* ------------------------------------- support unaligned scene -------------------------------------*/

template <typename T, unsigned UBS>
TILEOP void UBCopyInBase(__ubuf__ T* dst, __gm__ T* src, unsigned T0, unsigned T1, unsigned GMS)
{
    uint16_t nBurst = T0;
    uint32_t lenBurst = T1 * sizeof(T);
    uint32_t gmGap = (GMS - T1) * sizeof(T);
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    uint32_t ubGap = (UBS - T1) / blockSize;
    // NEXTNEXT: Need to generalize large data scene in future
    //    static_assert(nBurst < ((1ULL << 12) - 1ULL));
    //    static_assert(lenBurst < ((1ULL << 21) - 1ULL));
    //    static_assert(gmGap < ((1ULL << 32) - 1ULL));
    if (T1 == 0) {
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

template <typename T, unsigned UBS>
TILEOP void UBCopyInBase(__ubuf__ T* dst, __gm__ T* src, unsigned T0, unsigned T1, unsigned GMS,
                         unsigned dstStartOffset)
{
    constexpr auto blockSize = BLOCK_SIZE / sizeof(T);
    if (T1 == 0) {
        return;
    }
    if constexpr (UBS % blockSize != 0) {
        pipe_barrier(PIPE_ALL);
        for (unsigned i = 0; i < T0; i++) {
            unsigned startOffset = ((dstStartOffset - 1) / blockSize + 1) * blockSize - dstStartOffset;
            unsigned endOffset = (dstStartOffset + T1) / blockSize * blockSize - dstStartOffset;
            unsigned length = endOffset - startOffset;
            uint32_t lenBurst = length * sizeof(T);
            if (startOffset >= T1) {
                for (unsigned j = 0; j < T1; j++) { // 考虑如何用mask写法替代
                    dcci((__gm__ T*)(src + j), SINGLE_CACHE_LINE);
                    dst[dstStartOffset + j] = src[j];
                }
                dstStartOffset += UBS;
                src += GMS;
                continue;
            }
            for (unsigned j = 0; j < startOffset; j++) { // 考虑如何用mask写法替代
                dcci((__gm__ T*)(src + j), SINGLE_CACHE_LINE);
                dst[dstStartOffset + j] = src[j];
            }
            for (unsigned j = endOffset; j < T1; j++) { // 考虑如何用mask写法替代
                dcci((__gm__ T*)(src + j), SINGLE_CACHE_LINE);
                dst[dstStartOffset + j] = src[j];
            }
            if (length > 0) {
                if constexpr (sizeof(T) == 1) {
                    copy_gm_to_ubuf_align_b8(dst + dstStartOffset + startOffset, src + startOffset, 0 /*sid*/, 1,
                                             lenBurst, 0 /*left padding count*/, 0 /*right padding count*/, 0, 0);
                } else if constexpr (sizeof(T) == 2) {
                    copy_gm_to_ubuf_align_b16(dst + dstStartOffset + startOffset, src + startOffset, 0 /*sid*/, 1,
                                              lenBurst, 0 /*left padding count*/, 0 /*right padding count*/, 0, 0);
                } else {
                    copy_gm_to_ubuf_align_b32(dst + dstStartOffset + startOffset, src + startOffset, 0 /*sid*/, 1,
                                              lenBurst, 0 /*left padding count*/, 0 /*right padding count*/, 0, 0);
                }
            }
            dstStartOffset += UBS;
            src += GMS;
        }
        pipe_barrier(PIPE_ALL);
    } else if ((dstStartOffset % blockSize == 0) && (T1 % blockSize == 0)) { // 头块尾块均对齐到32B
        UBCopyInBase<T, UBS>(dst + dstStartOffset, src, T0, T1, GMS);
    } else {
        pipe_barrier(PIPE_ALL);
        unsigned startOffset = ((dstStartOffset - 1) / blockSize + 1) * blockSize - dstStartOffset;
        unsigned endOffset = (dstStartOffset + T1) / blockSize * blockSize - dstStartOffset;
        unsigned length = endOffset - startOffset;
        uint32_t lenBurst = length * sizeof(T);
        uint32_t gmGap = (GMS - length) * sizeof(T);
        uint32_t ubGap = (UBS - length) / blockSize;
        dst += dstStartOffset;
        if (startOffset >= T1) {
            for (unsigned i = 0; i < T0; i++) {
                for (unsigned j = 0; j < T1; j++) { // 考虑如何用mask写法替代
                    dcci((__gm__ T*)(src + j), SINGLE_CACHE_LINE);
                    dst[j] = src[j];
                }
                dst += UBS;
                src += GMS;
            }
            return;
        }
        if (length > 0) {
            if constexpr (sizeof(T) == 1) {
                copy_gm_to_ubuf_align_b8(dst + startOffset, src + startOffset, 0 /*sid*/, T0, lenBurst,
                                         0 /*left padding count*/, 0 /*right padding count*/, gmGap, ubGap);
            } else if constexpr (sizeof(T) == 2) {
                copy_gm_to_ubuf_align_b16(dst + startOffset, src + startOffset, 0 /*sid*/, T0, lenBurst,
                                          0 /*left padding count*/, 0 /*right padding count*/, gmGap, ubGap);
            } else {
                copy_gm_to_ubuf_align_b32(dst + startOffset, src + startOffset, 0 /*sid*/, T0, lenBurst,
                                          0 /*left padding count*/, 0 /*right padding count*/, gmGap, ubGap);
            }
        }
        for (unsigned i = 0; i < T0; i++) {
            for (unsigned j = 0; j < startOffset; j++) { // 考虑如何用mask写法替代
                dcci((__gm__ T*)(src + j), SINGLE_CACHE_LINE);
                dst[j] = src[j];
            }
            for (unsigned j = endOffset; j < T1; j++) { // 考虑如何用mask写法替代
                dcci((__gm__ T*)(src + j), SINGLE_CACHE_LINE);
                dst[j] = src[j];
            }
            dst += UBS;
            src += GMS;
        }
        pipe_barrier(PIPE_ALL);
    }
}

template <typename T, unsigned UBS1, unsigned UBS2, unsigned UBS3, unsigned UBS4, bool DST_IS_PARTIAL_MEM = false>
TILEOP void DynUBCopyIn(__ubuf__ T* dst, __gm__ T* src, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4,
                        unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3, unsigned GMS4, unsigned Offset0,
                        unsigned Offset1, unsigned Offset2, unsigned Offset3, unsigned Offset4, unsigned dstStartOffset)
{
    constexpr auto blockSize = BLOCK_SIZE / sizeof(T);
    src += CalcLinearOffset(GMS1, GMS2, GMS3, GMS4, Offset0, Offset1, Offset2, Offset3, Offset4);
    for (int i0 = 0; i0 < T0; i0++) {
        __gm__ T* src0 = src;
        for (int i1 = 0; i1 < T1; i1++) {
            __gm__ T* src1 = src0;
            for (int i2 = 0; i2 < T2; i2++) {
                if constexpr (UBS4 % blockSize == 0) {
                    if constexpr (!DST_IS_PARTIAL_MEM) {
                        TileOp::UBCopyInBase<T, UBS4>(dst + dstStartOffset, src1, T3, T4, GMS4);
                    } else if ((dstStartOffset % blockSize == 0) && (T4 % blockSize == 0)) {
                        TileOp::UBCopyInBase<T, UBS4>(dst + dstStartOffset, src1, T3, T4, GMS4);
                    } else {
                        // gm_to_ubuf无法处理尾块非对齐场景
                        TileOp::UBCopyInBase<T, UBS4>(dst, src1, T3, T4, GMS4, dstStartOffset);
                    }
                } else {
                    TileOp::UBCopyInBase<T, UBS4>(dst, src1, T3, T4, GMS4, dstStartOffset);
                }
                src1 += GMS3 * GMS4;
                dstStartOffset += UBS3 * UBS4;
            }
            src0 += GMS2 * GMS3 * GMS4;
            dstStartOffset += (UBS2 - T2) * UBS3 * UBS4;
        }
        src += GMS1 * GMS2 * GMS3 * GMS4;
        dstStartOffset += (UBS1 - T1) * UBS2 * UBS3 * UBS4;
    }
}

template <typename T, unsigned UBS1, unsigned UBS2, unsigned UBS3, unsigned UBS4, bool DST_IS_PARTIAL_MEM = false>
TILEOP void DynUBCopyIn(__ubuf__ T* dst, __gm__ T* src, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4,
                        unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3, unsigned GMS4, unsigned Offset0,
                        unsigned Offset1, unsigned Offset2, unsigned Offset3, unsigned Offset4)
{
    DynUBCopyIn<T, UBS1, UBS2, UBS3, UBS4, DST_IS_PARTIAL_MEM>(dst, src, T0, T1, T2, T3, T4, GMS0, GMS1, GMS2, GMS3,
                                                               GMS4, Offset0, Offset1, Offset2, Offset3, Offset4, 0);
}

template <typename T, unsigned UBS>
TILEOP void UBCopyOutBase(__gm__ T* dst, __ubuf__ T* src, unsigned T0, unsigned T1, unsigned GMS)
{
    uint16_t nBurst = T0;
    uint32_t lenBurst = T1 * sizeof(T);
    uint32_t gmGap = (GMS - T1) * sizeof(T);
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    uint32_t ubGap = (UBS - T1) / blockSize;
    // NEXTNEXT: Need to generalize large data in future
    //    static_assert( nBurst < ((1ULL << 12) - 1ULL));
    //    static_assert(lenBurst < ((1ULL << 21) - 1ULL));
    //    static_assert (gmGap < ((1ULL << 32) - 1ULL));
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

template <typename T, unsigned UBS>
TILEOP void UBCopyOutBase(__gm__ T* dst, __ubuf__ T* src, unsigned T0, unsigned T1, unsigned GMS,
                          unsigned srcStartOffset)
{
    if (T1 == 0) {
        return;
    }
    pipe_barrier(PIPE_ALL);
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    for (unsigned i = 0; i < T0; i++) {
        if ((srcStartOffset % BLOCK_SIZE != 0) || (T1 % BLOCK_SIZE != 0)) {
            for (unsigned j = 0; j < T1; j++) {
                dst[j] = src[srcStartOffset + j];
                dcci((__gm__ T*)(dst + j), SINGLE_CACHE_LINE);
            }
        } else {
            uint32_t lenBurst = T1 * sizeof(T);
            if constexpr (sizeof(T) == 1) {
                copy_ubuf_to_gm_align_b8(dst, src + srcStartOffset, 0 /*sid*/, 1, lenBurst, 0 /*left padding count*/,
                                         0 /*right padding count*/, 0, 0);
            } else if (sizeof(T) == 2) {
                copy_ubuf_to_gm_align_b16(dst, src + srcStartOffset, 0 /*sid*/, 1, lenBurst, 0 /*left padding count*/,
                                          0 /*right padding count*/, 0, 0);
            } else {
                copy_ubuf_to_gm_align_b32(dst, src + srcStartOffset, 0 /*sid*/, 1, lenBurst, 0 /*left padding count*/,
                                          0 /*right padding count*/, 0, 0);
            }
        }
        srcStartOffset += UBS;
        dst += GMS;
    }
    pipe_barrier(PIPE_ALL);
}

template <typename T, unsigned UBS1, unsigned UBS2, unsigned UBS3, unsigned UBS4>
TILEOP void DynUBCopyOut(__gm__ T* dst, __ubuf__ T* src, unsigned T0, unsigned T1, unsigned T2, unsigned T3,
                         unsigned T4, unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3, unsigned GMS4,
                         unsigned Offset0, unsigned Offset1, unsigned Offset2, unsigned Offset3, unsigned Offset4,
                         unsigned srcStartOffset)
{
    dst += CalcLinearOffset(GMS1, GMS2, GMS3, GMS4, Offset0, Offset1, Offset2, Offset3, Offset4);
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    for (int i0 = 0; i0 < T0; i0++) {
        __gm__ T* dst0 = dst;
        for (int i1 = 0; i1 < T1; i1++) {
            __gm__ T* dst1 = dst0;
            for (int i2 = 0; i2 < T2; i2++) {
                if constexpr (UBS4 % blockSize == 0) {
                    if (srcStartOffset % blockSize == 0) {
                        TileOp::UBCopyOutBase<T, UBS4>(dst1, src + srcStartOffset, T3, T4, GMS4);
                    } else {
                        TileOp::UBCopyOutBase<T, UBS4>(dst1, src, T3, T4, GMS4, srcStartOffset);
                    }
                } else {
                    TileOp::UBCopyOutBase<T, UBS4>(dst1, src, T3, T4, GMS4, srcStartOffset);
                }
                dst1 += GMS3 * GMS4;
                srcStartOffset += UBS3 * UBS4;
            }
            dst0 += GMS2 * GMS3 * GMS4;
            srcStartOffset += (UBS2 - T2) * UBS3 * UBS4;
        }
        dst += GMS1 * GMS2 * GMS3 * GMS4;
        srcStartOffset += (UBS1 - T1) * UBS2 * UBS3 * UBS4;
    }
}

template <typename T, unsigned UBS1, unsigned UBS2, unsigned UBS3, unsigned UBS4>
TILEOP void DynUBCopyOut(__gm__ T* dst, __ubuf__ T* src, unsigned T0, unsigned T1, unsigned T2, unsigned T3,
                         unsigned T4, unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3, unsigned GMS4,
                         unsigned Offset0, unsigned Offset1, unsigned Offset2, unsigned Offset3, unsigned Offset4)
{
    DynUBCopyOut<T, UBS1, UBS2, UBS3, UBS4>(dst, src, T0, T1, T2, T3, T4, GMS0, GMS1, GMS2, GMS3, GMS4, Offset0,
                                            Offset1, Offset2, Offset3, Offset4, 0);
}

// for ub spill out scene
template <typename T, unsigned UBS1, unsigned UBS2, unsigned UBS3, unsigned UBS4>
TILEOP void DynUBCopyIn(__ubuf__ T* dst, __gm__ T* src, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4,
                        unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3, unsigned GMS4)
{
    static_assert((UBS4 * sizeof(T)) % BLOCK_SIZE == 0, "UB tile must be 32B aligned!");

    for (int i0 = 0; i0 < T0; i0++) {
        __gm__ T* src0 = src;
        __ubuf__ T* dst0 = dst;
        for (int i1 = 0; i1 < T1; i1++) {
            __gm__ T* src1 = src0;
            __ubuf__ T* dst1 = dst0;
            for (int i2 = 0; i2 < T2; i2++) {
                TileOp::UBCopyInBase<T, UBS4>(dst1, src1, T3, T4, GMS4);
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

// for ub spill out scene
template <typename T, unsigned UBS1, unsigned UBS2, unsigned UBS3, unsigned UBS4>
TILEOP void DynUBCopyIn(__ubuf__ T* dst, __gm__ T* src, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4,
                        unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3, unsigned GMS4,
                        unsigned dstStartOffset)
{
    static_assert((UBS4 * sizeof(T)) % BLOCK_SIZE == 0, "UB tile must be 32B aligned!");
    dst += dstStartOffset;
    for (int i0 = 0; i0 < T0; i0++) {
        __gm__ T* src0 = src;
        __ubuf__ T* dst0 = dst;
        for (int i1 = 0; i1 < T1; i1++) {
            __gm__ T* src1 = src0;
            __ubuf__ T* dst1 = dst0;
            for (int i2 = 0; i2 < T2; i2++) {
                TileOp::UBCopyInBase<T, UBS4>(dst1, src1, T3, T4, GMS4);
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

// for ub spill out scene
template <typename T, unsigned UBS1, unsigned UBS2, unsigned UBS3, unsigned UBS4>
TILEOP void DynUBCopyOut(__gm__ T* dst, __ubuf__ T* src, unsigned T0, unsigned T1, unsigned T2, unsigned T3,
                         unsigned T4, unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3, unsigned GMS4)
{
    static_assert((UBS4 * sizeof(T)) % BLOCK_SIZE == 0, "UB tile must be 32B aligned!");

    for (int i0 = 0; i0 < T0; i0++) {
        __gm__ T* dst0 = dst;
        __ubuf__ T* src0 = src;
        for (int i1 = 0; i1 < T1; i1++) {
            __gm__ T* dst1 = dst0;
            __ubuf__ T* src1 = src0;
            for (int i2 = 0; i2 < T2; i2++) {
                TileOp::UBCopyOutBase<T, UBS4>(dst1, src1, T3, T4, GMS4);
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

template <typename T, typename T2, unsigned src0rawShape1, unsigned cacheMode, unsigned blockSize>
TILEOP void TIndexoutcastBase(__gm__ T* dst, __ubuf__ T* src0, __ubuf__ T2* src1, unsigned src0OriShape1,
                              unsigned src1OriShape1, unsigned GmShape1)
{
    for (auto i = 0; i < src1OriShape1; i++) {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        T2 curValue = *(reinterpret_cast<__ubuf__ T2*>(src1 + i));
        int64_t idxVal = static_cast<int64_t>(curValue);
        if (idxVal < 0) {
            continue;
        }
        if constexpr (cacheMode == 1) { // PA_NZ
            T2 blockCount = curValue / blockSize;
            T2 index = curValue % blockSize;

            __gm__ T* new_dst = dst + blockCount * blockSize * GmShape1 + index * 32 / sizeof(T);
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

            copy_ubuf_to_gm(new_dst, src0 + i * src0OriShape1, 0 /*sid*/, src0OriShape1 / 32 * sizeof(T), 1, 0,
                            blockSize - 1); // ok
        } else {
            __gm__ T* new_dst = dst + curValue * GmShape1;
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            TileOp::UBCopyOutBase<T, src0rawShape1>(new_dst, src0 + i * src0rawShape1, 1, src0OriShape1, GmShape1);
        }
    }
}

// src1=index [1,2] , src0: [TShape0,TShape1,TShape2,TShape3],  dst [GmShape0,GmShape1,GmShape2,GmShape3]
template <typename T, typename T2, unsigned src0rawShape1, unsigned src0rawShape2, unsigned src0rawShape3,
          unsigned src1rawShape3, unsigned cacheMode, unsigned blockSize>
TILEOP void DynTIndexoutcast(__gm__ T* dst, __ubuf__ T* src0, __ubuf__ T2* src1, unsigned src0OriShape0,
                             unsigned src0OriShape1, unsigned src0OriShape3, unsigned src1OriShape0,
                             unsigned src1OriShape1, unsigned GmShape0, unsigned GmShape1, unsigned GmShape2,
                             unsigned GmShape3, unsigned Offset0, unsigned Offset1, unsigned Offset2, unsigned Offset3)
{
    if (src0OriShape0 == 0 || src0OriShape1 == 0 || src0OriShape3 == 0 || src1OriShape0 == 0 || src1OriShape1 == 0) {
        return;
    }
    if (cacheMode == 2) {
        DynTIndexoutcast<T, T2>(dst, src0, src1, src1OriShape0, src1OriShape1, src1rawShape3, src0OriShape3,
                                src0rawShape1, src0rawShape3);
        return;
    }
    dst += CalcLinearOffset(GmShape1, GmShape2, GmShape3, Offset0, Offset1, Offset2, Offset3);

    int alignTS2TS3 = src0rawShape2 * src0rawShape3; // ub需要32B对齐
    int alignSrc1 = src1rawShape3;                   // 修改对 src1 的大小计算
    for (int i = 0; i < src0OriShape0; ++i) {
        for (int j = 0; j < src0OriShape1; ++j) {
            TileOp::TIndexoutcastBase<T, T2, src0rawShape3, cacheMode, blockSize>(dst, src0, src1, src0OriShape3,
                                                                                  src1OriShape1, GmShape3);
        }
        src0 += alignTS2TS3;
        src1 += alignSrc1;
        dst += GmShape2 * GmShape3;
    }
}

template <typename T, unsigned UBS1, unsigned UBS2, unsigned UBS3, unsigned UBS4 /*dst shape*/>
TILEOP void DynReshapeCopyIn(__ubuf__ T* dst, __gm__ T* tmp, unsigned T0, unsigned T1, unsigned T2, unsigned T3,
                             unsigned T4, unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3, unsigned GMS4,
                             unsigned Offset0, unsigned Offset1, unsigned Offset2, unsigned Offset3, unsigned Offset4)
{
    // DynUBCopyIn
    TileOp::DynUBCopyIn<T, UBS1, UBS2, UBS3, UBS4>(dst, tmp, T0, T1, T2, T3, T4, T0, T1, T2, T3, T4, 0, 0, 0, 0, 0);
}

template <typename T, unsigned UBS1, unsigned UBS2, unsigned UBS3, unsigned UBS4> /*src shape*/
TILEOP void DynReshapeCopyOut(__gm__ T* tmp, __ubuf__ T* src, unsigned T0, unsigned T1, unsigned T2, unsigned T3,
                              unsigned T4, unsigned GMS0, unsigned GMS1, unsigned GMS2, unsigned GMS3, unsigned GMS4,
                              unsigned Offset0, unsigned Offset1, unsigned Offset2, unsigned Offset3, unsigned Offset4)
{
    // DynUBCopyOut
    TileOp::DynUBCopyOut<T, UBS1, UBS2, UBS3, UBS4>(tmp, src, T0, T1, T2, T3, T4, T0, T1, T2, T3, T4, 0, 0, 0, 0, 0);
}

} // namespace TileOp
#endif // TILE_FWK_MTE_DYN_H
