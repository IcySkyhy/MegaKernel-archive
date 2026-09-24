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
 * \file vector_dyn.h
 * \brief
 */

#include "tileop_common.h"
#include "vector.h"
#include "mte_dyn.h"
#include <array>
#include <type_traits>

#ifndef TILE_FWK_VECTOR_DYN_H
#define TILE_FWK_VECTOR_DYN_H

namespace TileOp {
// ADD
#define T_BIN DynTadd_
#define T_BIN_PAIR DynTaddpair_
#define V_BIN_FUNC vadd
#define T_BIN_VS DynTadds_
#define V_BIN_FUNC_VS vadds
#include "vector_bin_dyn.h"
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC
// SUB
#define T_BIN DynTsub_
#define T_BIN_PAIR DynTsubpair_
#define V_BIN_FUNC vsub
#define T_BIN_VS DynTsubs_
#define V_BIN_FUNC_VS vadds
#define VS_SUB
#include "vector_bin_dyn.h"
#undef VS_SUB
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC
// MUL
#define T_BIN DynTmul_
#define T_BIN_PAIR DynTmulpair_
#define V_BIN_FUNC vmul
#define T_BIN_VS DynTmuls_
#define V_BIN_FUNC_VS vmuls
#include "vector_bin_dyn.h"
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC
// DIV
#define T_BIN DynTdiv_
#define T_BIN_PAIR DynTdivpair_
#define V_BIN_FUNC vdiv
#define T_BIN_VS DynTdivs_
#define V_BIN_FUNC_VS vmuls
#define VS_DIV
#include "vector_bin_dyn.h"
#undef VS_DIV
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC
// MAX
#define T_BIN DynTmax_
#define T_BIN_PAIR DynTmaxpair_
#define V_BIN_FUNC vmax
#define T_BIN_VS DynTmaxs_
#define V_BIN_FUNC_VS vmaxs
#include "vector_bin_dyn.h"
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
// MIN
#define T_BIN DynTmin_
#define T_BIN_PAIR DynTminpair_
#define V_BIN_FUNC vmin
#define T_BIN_VS DynTmins_
#define V_BIN_FUNC_VS vmins
#include "vector_bin_dyn.h"
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC
#undef T_BIN_VS
#undef V_BIN_FUNC_VS

// ADDWITHBRC
#define T_BIN_BRC DynTaddbrc_
#define V_BIN_BRC_FUNC vadd
#include "vector_bin_brc_dyn.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// SUBWITHBRC
#define T_BIN_BRC DynTsubbrc_
#define V_BIN_BRC_FUNC vsub
#include "vector_bin_brc_dyn.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// MULWITHBRC
#define T_BIN_BRC DynTmulbrc_
#define V_BIN_BRC_FUNC vmul
#include "vector_bin_brc_dyn.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// MAXWITHBRC
#define T_BIN_BRC DynTmaxbrc_
#define V_BIN_BRC_FUNC vmax
#include "vector_bin_brc_dyn.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// MINWITHBRC
#define T_BIN_BRC DynTminbrc_
#define V_BIN_BRC_FUNC vmin
#include "vector_bin_brc_dyn.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// DIVWITHBRC
#define T_BIN_BRC DynTdivbrc_
#define V_BIN_BRC_FUNC vdiv
#include "vector_bin_brc_dyn.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// Unary op
// EXP
#define T_UNA DynTexp_
#define V_UNA_FUNC vexp
#include "vector_una_dyn.h"
#undef T_UNA
#undef V_UNA_FUNC
// RECIPROCAL
#define T_UNA DynTrec_
#define V_UNA_FUNC vrec
#include "vector_una_dyn.h"
#undef T_UNA
#undef V_UNA_FUNC
// RSQRT
#define T_UNA DynTrsqrt_
#define V_UNA_FUNC vrsqrt
#include "vector_una_dyn.h"
#undef T_UNA
#undef V_UNA_FUNC
// SQRT
#define T_UNA DynTsqrt_
#define V_UNA_FUNC vsqrt
#include "vector_una_dyn.h"
#undef T_UNA
#undef V_UNA_FUNC
// ABS
#define T_UNA DynTabs_
#define V_UNA_FUNC vabs
#include "vector_una_dyn.h"
#undef T_UNA
#undef V_UNA_FUNC
// LN
#define T_UNA DynTln_
#define V_UNA_FUNC vln
#include "vector_una_dyn.h"
#undef T_UNA
#undef V_UNA_FUNC

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned srcRawShape1,
          unsigned srcRawShape2, unsigned axis0, unsigned axis1>
TILEOP void DynTtransposeMoveOutBase(__gm__ T* dst, __ubuf__ T* src, unsigned dstShape1, unsigned dstShape2)
{
    if constexpr (axis0 == 0 && axis1 == 1) {
        __gm__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        unsigned nBurst = TShape1;
        unsigned lenBurst = TShape2 * sizeof(T);
        unsigned srcStride = 0;
        unsigned dstStride = (dstShape1 * dstShape2 - TShape2) * sizeof(T);
        for (int b = 0; b < TShape0; b++) {
            copy_ubuf_to_gm_align_b32(dst_, src_, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
            dst_ += dstShape2;
            src_ += srcRawShape1 * srcRawShape2;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned srcRawShape1,
          unsigned srcRawShape2, unsigned srcRawShape3, unsigned axis0, unsigned axis1>
TILEOP void DynTtransposeMoveOut(__gm__ T* dst, __ubuf__ T* src, unsigned dstShape0, unsigned dstShape1,
                                 unsigned dstShape2, unsigned dstShape3, unsigned GmOffset0, unsigned GmOffset1,
                                 unsigned GmOffset2, unsigned GmOffset3)
{
    if constexpr (axis0 == 1 && axis1 == 2) {
        __gm__ T* dst_ = dst +
                         CalcLinearOffset(dstShape1, dstShape2, dstShape3, GmOffset0, GmOffset1, GmOffset2, GmOffset3);
        __ubuf__ T* src_ = src;
        for (int b = 0; b < TShape0; b++) {
            DynTtransposeMoveOutBase<T, TShape1, TShape2, TShape3, srcRawShape2, srcRawShape3, axis0 - 1, axis1 - 1>(
                dst_, src_, dstShape2, dstShape3);
            dst_ += dstShape1 * dstShape2 * dstShape3;
            src_ += srcRawShape1 * srcRawShape2 * srcRawShape3;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

/* ------------------------------------- support unaligned scene -------------------------------------*/

// The src data remains unchanged.
// T: fp32. support: OS0 <= REPEAT_MAX
template <typename T, unsigned DS, unsigned SS, unsigned TBS>
TILEOP void DynTrowsumsingle_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned OS0, unsigned OS1)
{
    //    OS0 <= REPEAT_MAX
    uint64_t srcRepeatPerRow = static_cast<uint64_t>(OS1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint16_t srcRepeatStride = SS * sizeof(T) / BLOCK_SIZE;
    constexpr unsigned nElemPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned remain = OS1 % nElemPerRepeat;
    if (srcRepeatPerRow == 1 && remain == 0) {
        vcadd(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, false);
        return;
    }

    if (OS0 <= REPEAT_MAX && srcRepeatPerRow < 1 && remain > 0) {
        SetContinuousMask(OS1);
        vcadd(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, false);
        set_vector_mask(-1, -1);
        return;
    }

    constexpr uint16_t tmpRepeatStride = TBS * sizeof(T) / BLOCK_SIZE;
    if constexpr (tmpRepeatStride < BLOCK_MAX_PER_REPEAT) {
        // work around for ccec compiling check; If delete will cause compiling error for "copy_ubuf_to_ubuf" after
        // which reminder the range of 7th parameter must be [0, 65535]
        return;
    }

    constexpr bool strideOverFlag = (tmpRepeatStride > REPEAT_STRIDE_MAX) || (srcRepeatStride > REPEAT_STRIDE_MAX);
    unsigned tmpOffset = tmpRepeatStride * BLOCK_SIZE / sizeof(T);
    unsigned srcOffset = srcRepeatStride * BLOCK_SIZE / sizeof(T);
    unsigned curLen = srcRepeatPerRow;
    if constexpr (strideOverFlag) {
        for (unsigned j = 0; j < OS0; j++) {
            for (unsigned i = 0; i < curLen / 2; i++) {
                vadd(tmp + i * nElemPerRepeat + j * tmpOffset, src + i * 2 * nElemPerRepeat + j * srcOffset,
                     src + (i * 2 + 1) * nElemPerRepeat + j * srcOffset, 1, 1, 1, 1, 8, 8, 8);
            }
        }
    } else {
        for (unsigned i = 0; i < curLen / 2; i++) {
            vadd(tmp + i * nElemPerRepeat, src + i * 2 * nElemPerRepeat, src + (i * 2 + 1) * nElemPerRepeat, OS0, 1, 1,
                 1, tmpRepeatStride, srcRepeatStride, srcRepeatStride);
        }
    }
    pipe_barrier(PIPE_V);
    if (curLen == 1 && remain > 0) {
        copy_ubuf_to_ubuf(tmp, src, 0, OS0, BLOCK_MAX_PER_REPEAT, srcRepeatStride - BLOCK_MAX_PER_REPEAT,
                          tmpRepeatStride - BLOCK_MAX_PER_REPEAT);
        pipe_barrier(PIPE_V);
    } else if (curLen % 2 > 0) {
        if constexpr (strideOverFlag) {
            for (unsigned j = 0; j < OS0; j++) {
                vadd(tmp + j * tmpOffset, tmp + j * tmpOffset, src + (curLen - 1) * nElemPerRepeat + j * srcOffset, 1,
                     1, 1, 1, 8, 8, 8);
            }
        } else {
            vadd(tmp, tmp, src + (curLen - 1) * nElemPerRepeat, OS0, 1, 1, 1, tmpRepeatStride, tmpRepeatStride,
                 srcRepeatStride);
        }
        pipe_barrier(PIPE_V);
    }

    if (remain > 0) {
        unsigned repeatOffset = curLen == 1 ? 0 : curLen / 2 - 1;
        SetContinuousMask(remain);
        if constexpr (strideOverFlag) {
            for (unsigned i = 0; i < OS0; i++) {
                vadd(tmp + repeatOffset * nElemPerRepeat + i * tmpOffset, src + curLen * nElemPerRepeat + i * srcOffset,
                     tmp + repeatOffset * nElemPerRepeat + i * tmpOffset, 1, 1, 1, 1, 8, 8, 8);
            }
        } else {
            vadd(tmp + repeatOffset * nElemPerRepeat, src + curLen * nElemPerRepeat,
                 tmp + repeatOffset * nElemPerRepeat, OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride, tmpRepeatStride);
        }
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }

    curLen = curLen / 2;
    bool mergeLast = true;
    while (curLen > 1) {
        if constexpr (strideOverFlag) {
            for (unsigned j = 0; j < OS0; j++) {
                for (unsigned i = 0; i < curLen / 2; i++) {
                    vadd(tmp + i * nElemPerRepeat + j * tmpOffset, tmp + i * 2 * nElemPerRepeat + j * tmpOffset,
                         tmp + (i * 2 + 1) * nElemPerRepeat + j * tmpOffset, 1, 1, 1, 1, 8, 8, 8);
                }
            }
        } else {
            for (unsigned i = 0; i < curLen / 2; i++) {
                vadd(tmp + i * nElemPerRepeat, tmp + i * 2 * nElemPerRepeat, tmp + (i * 2 + 1) * nElemPerRepeat, OS0, 1,
                     1, 1, tmpRepeatStride, tmpRepeatStride, tmpRepeatStride);
            }
        }
        unsigned loopRemain = curLen % 2;
        curLen = curLen / 2;
        if (loopRemain > 0) {
            pipe_barrier(PIPE_V);
            if constexpr (strideOverFlag) {
                for (unsigned i = 0; i < OS0; i++) {
                    vadd(tmp + (curLen - 1) * nElemPerRepeat + i * tmpOffset /*last repeat of new curLen*/,
                         tmp + curLen * 2 * nElemPerRepeat + i * tmpOffset /*remain repeat*/,
                         tmp + (curLen - 1) * nElemPerRepeat + i * tmpOffset, 1, 1, 1, 1, 8, 8, 8);
                }
            } else {
                vadd(tmp + (curLen - 1) * nElemPerRepeat /*last repeat of new curLen*/,
                     tmp + curLen * 2 * nElemPerRepeat /*remain repeat*/, tmp + (curLen - 1) * nElemPerRepeat, OS0, 1,
                     1, 1, tmpRepeatStride, tmpRepeatStride, tmpRepeatStride);
            }
        }
        pipe_barrier(PIPE_V);
    }
    pipe_barrier(PIPE_V);
    vcadd(dst, tmp, OS0, (uint16_t)DS, (uint16_t)1ULL, tmpRepeatStride, false);
}

template <typename T, unsigned DS1, unsigned DS2, unsigned DS3, unsigned SS1, unsigned SS2, unsigned SS3, unsigned TBS3>
TILEOP void DynTrowsumsingle_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned OS0, unsigned OS1,
                              unsigned OS2, unsigned OS3)
{
    static_assert(SS3 * sizeof(T) % BLOCK_SIZE == 0);
    for (int i = 0; i < OS0; ++i) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < OS1; ++j) {
            if (OS2 != 0 && OS3 != 0) {
                TileOp::DynTrowsumsingle_<T, DS3, SS3, TBS3>(dst_, src_, tmp, OS2, OS3);
                dst_ += DS3 * DS2;
                src_ += SS3 * SS2;
                pipe_barrier(PIPE_V);
            }
        }
        dst += DS1 * DS2 * DS3;
        src += SS1 * SS2 * SS3;
    }
}

// The src data remains unchanged.
// T: fp32. support: OS0 <= REPEAT_MAX
template <typename T, unsigned DS, unsigned SS, unsigned TBS>
TILEOP void DynTrowmaxsingle_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned OS0, unsigned OS1)
{
    //    OS0 <= REPEAT_MAX
    uint64_t srcRepeatPerRow = static_cast<uint64_t>(OS1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint16_t srcRepeatStride = SS * sizeof(T) / BLOCK_SIZE;
    constexpr unsigned nElemPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned remain = OS1 % nElemPerRepeat;
    if (srcRepeatPerRow == 1 && OS0 <= REPEAT_MAX && remain == 0) {
        vcmax(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, ONLY_VALUE);
        return;
    }

    if (OS0 <= REPEAT_MAX && srcRepeatPerRow < 1 && remain > 0) {
        SetContinuousMask(OS1);
        vcmax(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, ONLY_VALUE);
        set_vector_mask(-1, -1);
        return;
    }

    constexpr uint16_t tmpRepeatStride = TBS * sizeof(T) / BLOCK_SIZE;
    if ((srcRepeatStride < BLOCK_MAX_PER_REPEAT) || (tmpRepeatStride < BLOCK_MAX_PER_REPEAT)) {
        return;
    }
    if (srcRepeatPerRow == 1 && remain > 0) {
        copy_ubuf_to_ubuf(tmp, src, 0, OS0, BLOCK_MAX_PER_REPEAT, srcRepeatStride - BLOCK_MAX_PER_REPEAT,
                          tmpRepeatStride - BLOCK_MAX_PER_REPEAT);
    } else {
        if constexpr ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
            vmax(tmp, src, src + nElemPerRepeat, OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride, srcRepeatStride);
        } else {
            for (int i = 0; i < OS0; i++) {
                vmax(tmp + i * TBS, src + i * SS, src + i * SS + nElemPerRepeat, 1, 1, 1, 1, 1, 1, 1);
            }
        }
    }
    pipe_barrier(PIPE_V);

    for (int i = 2; i < srcRepeatPerRow; i++) {
        if constexpr ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
            vmax(tmp, src + i * nElemPerRepeat, tmp, OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride, tmpRepeatStride);
        } else {
            for (int j = 0; j < OS0; j++) {
                vmax(tmp + j * TBS, src + i * nElemPerRepeat + j * SS, tmp + j * TBS, 1, 1, 1, 1, 1, 1, 1);
            }
        }
        pipe_barrier(PIPE_V);
    }
    if (remain > 0) {
        SetContinuousMask(remain);
        if constexpr ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
            vmax(tmp, src + srcRepeatPerRow * nElemPerRepeat, tmp, OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride,
                 tmpRepeatStride);
        } else {
            for (int j = 0; j < OS0; j++) {
                vmax(tmp + j * TBS, src + srcRepeatPerRow * nElemPerRepeat + j * SS, tmp + j * TBS, 1, 1, 1, 1, 1, 1,
                     1);
            }
        }
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }
    vcmax(dst, tmp, OS0, (uint16_t)DS, (uint16_t)1ULL, tmpRepeatStride, ONLY_VALUE);
    pipe_barrier(PIPE_V);
}

template <typename T, unsigned DS1, unsigned DS2, unsigned DS3, unsigned SS1, unsigned SS2, unsigned SS3, unsigned TBS3>
TILEOP void DynTrowmaxsingle_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned OS0, unsigned OS1,
                              unsigned OS2, unsigned OS3)
{
    static_assert(SS3 * sizeof(T) % BLOCK_SIZE == 0);
    for (int i = 0; i < OS0; ++i) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < OS1; ++j) {
            if (OS2 != 0 && OS3 != 0) {
                TileOp::DynTrowmaxsingle_<T, DS3, SS3, TBS3>(dst_, src_, tmp, OS2, OS3);
                dst_ += DS3 * DS2;
                src_ += SS3 * SS2;
                pipe_barrier(PIPE_V);
            }
        }
        dst += DS1 * DS2 * DS3;
        src += SS1 * SS2 * SS3;
    }
}

template <typename T, unsigned DS, unsigned SS, unsigned TBS>
TILEOP void DynTrowminsingle_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned OS0, unsigned OS1)
{
    //    OS0 <= REPEAT_MAX
    uint64_t srcRepeatPerRow = static_cast<uint64_t>(OS1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint16_t srcRepeatStride = SS * sizeof(T) / BLOCK_SIZE;
    constexpr unsigned nElemPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned remain = OS1 % nElemPerRepeat;
    if (srcRepeatPerRow == 1 && OS0 <= REPEAT_MAX && remain == 0) {
        vcmin(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, ONLY_VALUE);
        return;
    }

    if (OS0 <= REPEAT_MAX && srcRepeatPerRow < 1 && remain > 0) {
        SetContinuousMask(OS1);
        vcmin(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, ONLY_VALUE);
        set_vector_mask(-1, -1);
        return;
    }

    constexpr uint16_t tmpRepeatStride = TBS * sizeof(T) / BLOCK_SIZE;
    if ((srcRepeatStride < BLOCK_MAX_PER_REPEAT) || (tmpRepeatStride < BLOCK_MAX_PER_REPEAT)) {
        return;
    }
    if (srcRepeatStride >= BLOCK_MAX_PER_REPEAT && srcRepeatPerRow == 1 && remain > 0) {
        copy_ubuf_to_ubuf(tmp, src, 0, OS0, BLOCK_MAX_PER_REPEAT, srcRepeatStride - BLOCK_MAX_PER_REPEAT,
                          tmpRepeatStride - BLOCK_MAX_PER_REPEAT);
    } else {
        if constexpr ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
            vmin(tmp, src, src + nElemPerRepeat, OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride, srcRepeatStride);
        } else {
            for (int i = 0; i < OS0; i++) {
                vmin(tmp + i * TBS, src + i * SS, src + i * SS + nElemPerRepeat, 1, 1, 1, 1, 1, 1, 1);
            }
        }
    }
    pipe_barrier(PIPE_V);

    for (int i = 2; i < srcRepeatPerRow; i++) {
        if constexpr ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
            vmin(tmp, src + i * nElemPerRepeat, tmp, OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride, tmpRepeatStride);
        } else {
            for (int j = 0; j < OS0; j++) {
                vmin(tmp + j * TBS, src + i * nElemPerRepeat + j * SS, tmp + j * TBS, 1, 1, 1, 1, 1, 1, 1);
            }
        }
        pipe_barrier(PIPE_V);
    }
    if (remain > 0) {
        SetContinuousMask(remain);
        if constexpr ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
            vmin(tmp, src + srcRepeatPerRow * nElemPerRepeat, tmp, OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride,
                 tmpRepeatStride);
        } else {
            for (int j = 0; j < OS0; j++) {
                vmin(tmp + j * TBS, src + srcRepeatPerRow * nElemPerRepeat + j * SS, tmp + j * TBS, 1, 1, 1, 1, 1, 1,
                     1);
            }
        }
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }
    vcmin(dst, tmp, OS0, (uint16_t)DS, (uint16_t)1ULL, tmpRepeatStride, ONLY_VALUE);
    pipe_barrier(PIPE_V);
}

template <typename T, unsigned DS1, unsigned DS2, unsigned DS3, unsigned SS1, unsigned SS2, unsigned SS3, unsigned TBS3>
TILEOP void DynTrowminsingle_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned OS0, unsigned OS1,
                              unsigned OS2, unsigned OS3)
{
    static_assert(SS3 * sizeof(T) % BLOCK_SIZE == 0);
    for (int i = 0; i < OS0; ++i) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < OS1; ++j) {
            if (OS2 != 0 && OS3 != 0) {
                TileOp::DynTrowminsingle_<T, DS3, SS3, TBS3>(dst_, src_, tmp, OS2, OS3);
                dst_ += DS3 * DS2;
                src_ += SS3 * SS2;
                pipe_barrier(PIPE_V);
            }
        }
        dst += DS1 * DS2 * DS3;
        src += SS1 * SS2 * SS3;
    }
}

TILEOP uint16_t DupB8ToB16(bool value)
{
    auto u16 = static_cast<uint16_t>(value);
    return u16 + (u16 * 0x100); // 相当于 extended | (extended << 8)
}

TILEOP uint16_t DupB8ToB16(uint8_t value)
{
    auto u16 = static_cast<uint16_t>(value);
    return u16 + (u16 * 0x100); // 相当于 extended | (extended << 8)
}

TILEOP uint16_t DupB8ToB16(int8_t value)
{
    auto ub8 = static_cast<uint8_t>(value);
    return DupB8ToB16(ub8);
}

template <typename T>
struct VdupTrait {
    static constexpr bool isB8 = (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>);

    using DupType = std::conditional_t<isB8, uint16_t, T>;

    TILEOP DupType DupValue(T value)
    {
        if constexpr (isB8) {
            return DupB8ToB16(value);
        } else {
            return value;
        }
    }

    TILEOP uint64_t DupSize(uint64_t size)
    {
        if constexpr (isB8) {
            // UB是32B对齐，这是安全的
            return (size + sizeof(DupType) - 1) / sizeof(DupType);
        } else {
            return size;
        }
    }

    TILEOP constexpr uint64_t DupDstStride(uint64_t stride)
    {
        if constexpr (isB8) {
            return stride / sizeof(DupType);
        } else {
            return stride;
        }
    }
};

template <typename T, unsigned dstStride, unsigned srcStride>
TILEOP void BatchVdup(__ubuf__ T* dst, __ubuf__ T* src, unsigned batchSize, unsigned dupSize)
{
    using DupType = typename VdupTrait<T>::DupType;
    auto dupDst = (__ubuf__ DupType*)dst;
    dupSize = VdupTrait<T>::DupSize(dupSize);
    constexpr unsigned reptEleNum = REPEAT_BYTE / sizeof(DupType);
    constexpr unsigned dupDstStride = VdupTrait<T>::DupDstStride(dstStride);
    unsigned dupRepeats = dupSize / reptEleNum;

    if (dupRepeats < 1) {
        // 16 1 -> 16 16
        SetContinuousMask(dupSize);
        for (int i = 0; i < batchSize; i++) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            T tmp = (T)(*(src + i * srcStride));
            DupType dupValue = VdupTrait<T>::DupValue(tmp);
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
            vector_dup(dupDst + i * dupDstStride, dupValue, 1, 1, 0, 0, 0);
        }
        set_vector_mask(-1, -1);
    } else {
        // 16 1 -> 16 64
        unsigned remainNum = dupSize % reptEleNum;
        unsigned numLoop = dupRepeats / REPEAT_MAX;
        unsigned remainAfterLoop = dupRepeats % REPEAT_MAX;
        for (int i = 0; i < batchSize; i++) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            T tmp = (T)(*(src + i * srcStride));
            DupType dupValue = VdupTrait<T>::DupValue(tmp);
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
            if (numLoop) {
                for (int j = 0; j < numLoop; j++) {
                    vector_dup(dupDst + j * reptEleNum * REPEAT_MAX, dupValue, REPEAT_MAX, 1, 1, 8, 0);
                }
            }
            if (remainAfterLoop) {
                vector_dup(dupDst + numLoop * reptEleNum * REPEAT_MAX, dupValue, remainAfterLoop, 1, 1, 8, 0);
            }
            if (remainNum) {
                SetContinuousMask(remainNum);
                vector_dup(dupDst + dupRepeats * reptEleNum, dupValue, 1, 1, 1, 8, 0);
                set_vector_mask(-1, -1);
            }
            dupDst += dupDstStride;
        }
    }
}
// dim2
template <typename T, unsigned dstRawShape1, unsigned srcRawShape1, unsigned axis>
TILEOP void DynTexpand_(__ubuf__ T* dst, __ubuf__ T* src, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0,
                        unsigned srcShape1)
{
    if (axis == 0) {
        // 1 16 -> 16 16
        if (dstShape1 == 0 || dstShape0 == 0) {
            return;
        }
        uint64_t blockLen = (dstShape1 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int i = 0; i < dstShape0; i++) {
            copy_ubuf_to_ubuf(dst + i * dstRawShape1, src, 0, 1, blockLen, 1, 1);
        }
    } else if (axis == 1) {
        BatchVdup<T, dstRawShape1, srcRawShape1>(dst, src, dstShape0, dstShape1);
    }
}
// dim3
template <typename T, unsigned dstRawShape1, unsigned dstRawShape2, unsigned srcRawShape1, unsigned srcRawShape2,
          unsigned axis>
TILEOP void DynTexpand_(__ubuf__ T* dst, __ubuf__ T* src, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2,
                        unsigned srcShape0, unsigned srcShape1, unsigned srcShape2)
{
    if (axis == 1 || axis == 2) {
        // 16 1 16 -> 16 16 16 or 16 16 1 -> 16 16 16
        for (unsigned i = 0; i < dstShape0; i++) {
            TileOp::DynTexpand_<T, dstRawShape2, srcRawShape2, axis - 1>(dst, src, dstShape1, dstShape2, srcShape1,
                                                                         srcShape2);
            dst += dstRawShape1 * dstRawShape2;
            src += srcRawShape1 * srcRawShape2;
        }
    } else if (axis == 0) {
        // 1 16 16 -> 16 16 16
        if (dstShape2 == 0 || dstShape1 == 0) {
            return;
        }
        uint64_t blockLen = (dstShape2 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t srcGap = (srcRawShape2 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        uint64_t dstGap = (dstRawShape2 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        for (unsigned i = 0; i < dstShape0; i++) {
            copy_ubuf_to_ubuf(dst + i * dstRawShape1 * dstRawShape2, src, 0, dstShape1, blockLen, srcGap, dstGap);
        }
    }
}
// dim4
template <typename T, unsigned dstRawShape1, unsigned dstRawShape2, unsigned dstRawShape3, unsigned srcRawShape1,
          unsigned srcRawShape2, unsigned srcRawShape3, unsigned axis>
TILEOP void DynTexpand_(__ubuf__ T* dst, __ubuf__ T* src, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2,
                        unsigned dstShape3, unsigned srcShape0, unsigned srcShape1, unsigned srcShape2,
                        unsigned srcShape3)
{
    if (axis == 0) {
        // 1 16 16 16 -> 16 16 16 16
        if (dstShape3 == 0 || dstShape2 == 0) {
            return;
        }
        uint64_t blockLen = (dstShape3 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t srcGap = (srcRawShape3 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        uint64_t dstGap = (dstRawShape3 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        for (unsigned i = 0; i < dstShape0; ++i) {
            for (unsigned j = 0; j < dstShape1; j++) {
                copy_ubuf_to_ubuf(
                    dst + i * dstRawShape1 * dstRawShape2 * dstRawShape3 + j * dstRawShape2 * dstRawShape3,
                    src + j * srcRawShape2 * srcRawShape3, 0, dstShape2, blockLen, srcGap, dstGap);
            }
        }
    } else if (axis == 1 || axis == 2 || axis == 3) {
        for (unsigned i = 0; i < dstShape0; ++i) {
            TileOp::DynTexpand_<T, dstRawShape2, dstRawShape3, srcRawShape2, srcRawShape3, axis - 1>(
                dst, src, dstShape1, dstShape2, dstShape3, srcShape1, srcShape2, srcShape3);
            dst += dstRawShape1 * dstRawShape2 * dstRawShape3;
            src += srcRawShape1 * srcRawShape2 * srcRawShape3;
        }
    }
}

template <typename Td, typename Ts, unsigned DS, unsigned SS, unsigned Mode>
TILEOP void DynTcast_(__ubuf__ Td* dst, __ubuf__ Ts* src, unsigned T0, unsigned T1)
{
    // Now support fp32<->fp16, fp32<->bf16, fp32<->int16, fp32<->int32, int32<->fp16, fp16<->int8, fp32->fp32,
    // bf16->int32
    uint64_t repeatWidth = static_cast<uint64_t>(max(sizeof(Td), sizeof(Ts)));

    unsigned elementsPerRepeat = REPEAT_BYTE / repeatWidth;
    unsigned numRepeatPerLine = T1 / elementsPerRepeat;
    unsigned numRemainPerLine = T1 % elementsPerRepeat;
    unsigned dstRepeatStride = repeatWidth == sizeof(Td) ? BLOCK_MAX_PER_REPEAT :
                                                           (BLOCK_MAX_PER_REPEAT / sizeof(Ts) * sizeof(Td));
    unsigned srcRepeatStride = repeatWidth == sizeof(Ts) ? BLOCK_MAX_PER_REPEAT :
                                                           (BLOCK_MAX_PER_REPEAT / sizeof(Td) * sizeof(Ts));
    constexpr unsigned dstNElemPerBlock = BLOCK_SIZE / sizeof(Td);
    constexpr unsigned srcNElemPerBlock = BLOCK_SIZE / sizeof(Ts);

    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < T0; i++) {
            if (numLoop) {
                for (int j = 0; j < numLoop; j++) {
                    GenCastCall<Td, Ts, Mode>(dst + i * DS + j * elementsPerRepeat * REPEAT_MAX,
                                              src + i * SS + j * elementsPerRepeat * REPEAT_MAX, (uint8_t)REPEAT_MAX, 1,
                                              1, (uint16_t)dstRepeatStride, (uint16_t)srcRepeatStride);
                }
            }
            if (remainAfterLoop) {
                GenCastCall<Td, Ts, Mode>(dst + i * DS + numLoop * elementsPerRepeat * REPEAT_MAX,
                                          src + i * SS + numLoop * elementsPerRepeat * REPEAT_MAX,
                                          (uint8_t)remainAfterLoop, 1, 1, (uint16_t)dstRepeatStride,
                                          (uint16_t)srcRepeatStride);
            }
        }
    }

    // shift to deal with tail
    dst += numRepeatPerLine * elementsPerRepeat;
    src += numRepeatPerLine * elementsPerRepeat;

    if (numRemainPerLine) {
        unsigned numLoop = T0 / REPEAT_MAX;
        unsigned remainAfterLoop = T0 % REPEAT_MAX;
        SetContinuousMask(numRemainPerLine);
        if (numLoop) {
            for (int i = 0; i < numLoop; i++) {
                GenCastCall<Td, Ts, Mode>(dst + i * REPEAT_MAX * DS, src + i * REPEAT_MAX * SS, (uint8_t)REPEAT_MAX, 1,
                                          1, (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
            }
        }
        if (remainAfterLoop) {
            GenCastCall<Td, Ts, Mode>(dst + numLoop * REPEAT_MAX * DS, src + numLoop * REPEAT_MAX * SS,
                                      (uint8_t)remainAfterLoop, 1, 1, (uint16_t)DS / dstNElemPerBlock,
                                      (uint16_t)SS / srcNElemPerBlock);
        }
        set_vector_mask(-1, -1);
    }
}

template <typename Td, typename Ts, unsigned DS0, unsigned DS1, unsigned DS2, unsigned SS0, unsigned SS1, unsigned SS2,
          unsigned Mode>
TILEOP void DynTcast_(__ubuf__ Td* dst, __ubuf__ Ts* src, unsigned T0, unsigned T1, unsigned T2, unsigned T3)
{
    static_assert((DS2 * sizeof(Td)) % BLOCK_SIZE == 0);
    static_assert((SS2 * sizeof(Ts)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ Td* dst_ = dst;
        __ubuf__ Ts* src_ = src;
        for (int j = 0; j < T1; j++) {
            DynTcast_<Td, Ts, DS2, SS2, Mode>(dst_, src_, T2, T3);
            dst_ += DS1 * DS2;
            src_ += SS1 * SS2;
        }
        dst += DS0 * DS1 * DS2;
        src += SS0 * SS1 * SS2;
    }
}

template <typename T, unsigned Ds>
TILEOP void DynTduplicate_(__ubuf__ T* dst, T value, unsigned T0, unsigned T1)
{
    constexpr unsigned npr = REPEAT_BYTE / sizeof(T);
    unsigned numRepeatPerLine = T1 / npr;
    unsigned numRemainPerLine = T1 % npr;
    constexpr unsigned blockSizeElem = BLOCK_SIZE / sizeof(T);
    if (numRepeatPerLine > 0) {
        for (unsigned i = 0; i < T0; i++) {
            vector_dup(dst + i * Ds, value, numRepeatPerLine, 1, 1, BLOCK_MAX_PER_REPEAT, (int64_t)0);
        }
    }

    // shift to deal with tail
    dst += numRepeatPerLine * npr;
    if (numRemainPerLine) {
        unsigned numLoop = T0 / REPEAT_MAX;
        unsigned remainAfterLoop = T0 % REPEAT_MAX;
        if (numRemainPerLine >= HALF_MASK) {
            set_vector_mask((((static_cast<uint64_t>(1)) << static_cast<uint32_t>(numRemainPerLine - HALF_MASK)) - 1UL),
                            0xffffffffffffffffUL);
        } else {
            set_vector_mask(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(numRemainPerLine)) - 1UL));
        }
        if (numLoop) {
            for (unsigned i = 0; i < numLoop; i++) {
                vector_dup(dst + i * REPEAT_MAX * Ds, value, REPEAT_MAX, 1, 1, Ds / blockSizeElem, (int64_t)0);
            }
        }
        if (remainAfterLoop) {
            vector_dup(dst + numLoop * REPEAT_MAX * Ds, value, remainAfterLoop, 1, 1, Ds / blockSizeElem, (int64_t)0);
        }
        set_vector_mask(-1, -1);
    }
}

// dim2
template <typename T, unsigned Ds>
TILEOP void DynTduplicate_(__ubuf__ T* dst, T value, unsigned T0, unsigned T1, unsigned dstStartOffset)
{
    constexpr unsigned npr = REPEAT_BYTE / sizeof(T);
    if ((dstStartOffset % npr == 0) && (Ds % npr == 0)) { // 该分支里会处理头块对齐尾块非对齐场景
        DynTduplicate_<T, Ds>(dst + dstStartOffset, value, T0, T1);
    } else {
        pipe_barrier(PIPE_ALL);
        constexpr auto block = BLOCK_SIZE / sizeof(T);
        unsigned curDstStartOffset = dstStartOffset;
        for (unsigned i = 0; i < T0; i++) {
            unsigned startOffset = ((curDstStartOffset - 1) / npr + 1) * npr - curDstStartOffset;
            unsigned endOffset = (curDstStartOffset + T1) / npr * npr - curDstStartOffset;
            if (startOffset >= T1) {
                for (unsigned j = 0; j < T1; j++) { // 可用mask写法替代
                    dst[curDstStartOffset + j] = value;
                }
                curDstStartOffset += Ds;
                continue;
            }
            for (unsigned j = 0; j < startOffset; j++) { // 可用mask写法替代
                dst[curDstStartOffset + j] = value;
            }
            for (unsigned j = endOffset; j < T1; j++) { // 可用mask写法替代
                dst[curDstStartOffset + j] = value;
            }
            unsigned length = endOffset - startOffset;
            unsigned numRepeatPerLine = length / npr;
            unsigned blockPerRepeat = npr / block;
            if (length > 0) {
                vector_dup(dst + curDstStartOffset + startOffset, value, numRepeatPerLine, 1, 1, blockPerRepeat,
                           (int64_t)0);
            }
            curDstStartOffset += Ds;
        }
        pipe_barrier(PIPE_ALL);
    }
}

// dim4
template <typename T, unsigned Ds0, unsigned Ds1, unsigned Ds2>
TILEOP void DynTduplicate_(__ubuf__ T* dst, T value, unsigned T0, unsigned T1, unsigned T2, unsigned T3,
                           unsigned dstStartOffset)
{
    for (unsigned i = 0; i < T0; i++) {
        for (unsigned j = 0; j < T1; j++) {
            DynTduplicate_<T, Ds2>(dst, value, T2, T3, dstStartOffset);
            dstStartOffset += Ds1 * Ds2;
        }
        dstStartOffset += (Ds0 - T1) * Ds1 * Ds2;
    }
}

// dim4
template <typename T, unsigned Ds0, unsigned Ds1, unsigned Ds2>
TILEOP void DynTduplicate_(__ubuf__ T* dst, T value, unsigned T0, unsigned T1, unsigned T2, unsigned T3)
{
    unsigned dstStartOffset = 0;
    for (unsigned i = 0; i < T0; i++) {
        for (unsigned j = 0; j < T1; j++) {
            if constexpr ((Ds2 * sizeof(T)) % BLOCK_SIZE == 0) {
                DynTduplicate_<T, Ds2>(dst + dstStartOffset, value, T2, T3);
            } else {
                DynTduplicate_<T, Ds2>(dst, value, T2, T3, dstStartOffset);
            }
            dstStartOffset += Ds1 * Ds2;
        }
        dstStartOffset += (Ds0 - T1) * Ds1 * Ds2;
    }
}

// dim2
template <typename T, unsigned tmpRawShape1>
TILEOP void DynTrowsumline_(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* tmp, unsigned TShape0, unsigned TShape1,
                            unsigned srcRawShape1)
{
    static_assert(sizeof(T) == 4);
    constexpr unsigned DTypeSize = sizeof(T);
    unsigned lenBurst = (TShape1 * DTypeSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (TShape0 == 1) {
        copy_ubuf_to_ubuf(dst, src0, 0, 1, lenBurst, 0, 0);
        pipe_barrier(PIPE_V);
        return;
    }
    for (uint32_t i = 0; i < TShape0 / 2; i++) {
        set_mask_count();
        set_vector_mask(0, TShape1);
        vadd(tmp + i * tmpRawShape1, src0 + 2 * i * srcRawShape1, src0 + (2 * i + 1) * srcRawShape1, 0, 1, 1, 1, 8, 8,
             8);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }
    if (TShape0 % 2 == 1) {
        set_mask_count();
        set_vector_mask(0, TShape1);
        vadd(tmp, tmp, src0 + (TShape0 - 1) * srcRawShape1, 0, 1, 1, 1, 8, 8, 8);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }

    unsigned cnt = TShape0 / 2;
    while (cnt > 1) {
        for (uint32_t i = 0; i < cnt / 2; i++) {
            set_mask_count();
            set_vector_mask(0, TShape1);
            vadd(tmp + i * tmpRawShape1, tmp + 2 * i * tmpRawShape1, tmp + (2 * i + 1) * tmpRawShape1, 0, 1, 1, 1, 8, 8,
                 8);
            set_mask_norm();
            set_vector_mask(-1, -1);
            pipe_barrier(PIPE_V);
        }
        if (cnt % 2 == 1) {
            set_mask_count();
            set_vector_mask(0, TShape1);
            vadd(tmp, tmp, tmp + (cnt - 1) * tmpRawShape1, 0, 1, 1, 1, 8, 8, 8);
            set_mask_norm();
            set_vector_mask(-1, -1);
            pipe_barrier(PIPE_V);
        }
        cnt /= 2;
    }
    copy_ubuf_to_ubuf(dst, tmp, 0, 1, lenBurst, 0, 0);
    pipe_barrier(PIPE_V);
}

// dim4
template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned srcRawShape3, unsigned dstRawShape1,
          unsigned dstRawShape2, unsigned dstRawShape3, unsigned tmpRawShape1, unsigned tmpRawShape2,
          unsigned tmpRawShape3, unsigned axis>
TILEOP void DynTrowsumline_(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* tmp, unsigned TShape0, unsigned TShape1,
                            unsigned TShape2, unsigned TShape3)
{
    static_assert(sizeof(T) == 4);
    if (TShape0 == 0 || TShape1 == 0 || TShape2 == 0 || TShape3 == 0) {
        return;
    }
    unsigned srcRawShape;
    if constexpr (axis == 1) {
        srcRawShape = srcRawShape2 * srcRawShape3;
    } else if constexpr (axis == 0) {
        srcRawShape = srcRawShape1 * srcRawShape2 * srcRawShape3;
    } else if constexpr (axis == 2) {
        srcRawShape = srcRawShape3;
    }
    unsigned validShape[] = {TShape0, TShape1, TShape2, TShape3};
    for (unsigned n0Index = 0, n0Size = (axis == 0 ? (size_t)1 : TShape0); n0Index < n0Size; ++n0Index) {
        for (unsigned n1Index = 0, n1Size = (axis == 1 ? (size_t)1 : TShape1); n1Index < n1Size; ++n1Index) {
            for (unsigned n2Index = 0, n2Size = (axis == 2 ? (size_t)1 : TShape2); n2Index < n2Size; ++n2Index) {
                auto dstOffset = n0Index * dstRawShape1 * dstRawShape2 * dstRawShape3 +
                                 n1Index * dstRawShape2 * dstRawShape3 + n2Index * dstRawShape3;
                auto srcOffset = n0Index * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                 n1Index * srcRawShape2 * srcRawShape3 + n2Index * srcRawShape3;
                DynTrowsumline_<T, tmpRawShape3>(dst + dstOffset, src0 + srcOffset, tmp, validShape[axis], TShape3,
                                                 srcRawShape);
                pipe_barrier(PIPE_V);
            }
        }
    }
}

// dim2
template <typename T, unsigned srcRawShape1>
TILEOP void DynTrowmaxline_(__ubuf__ T* dst, __ubuf__ T* src0, unsigned TShape0, unsigned TShape1)
{
    static_assert(sizeof(T) == 4 || sizeof(T) == 2);
    uint32_t rptElm = REPEAT_BYTE / sizeof(T);
    uint32_t repeatTime = (TShape1 + rptElm - 1) / rptElm;
    uint32_t remainElm = TShape1 % rptElm;
    if (!remainElm) {
        vcopy((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src0, repeatTime, 1, 1, 8, 8);
    } else {
        if (repeatTime == 1) {
            SetContinuousMask(remainElm);
            vcopy((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src0, 1, 1, 1, 8, 8);
            set_vector_mask(-1, -1);
        } else {
            vcopy((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src0, repeatTime - 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            SetContinuousMask(remainElm);
            vcopy((__ubuf__ uint32_t*)(dst + (repeatTime - 1) * rptElm),
                  (__ubuf__ uint32_t*)(src0 + (repeatTime - 1) * rptElm), 1, 1, 1, 8, 8);
            set_vector_mask(-1, -1);
        }
    }
    pipe_barrier(PIPE_V);

    for (uint32_t j = 1; j < TShape0; j++) {
        if (!remainElm) {
            vmax(dst, dst, src0 + j * srcRawShape1, repeatTime, 1, 1, 1, 8, 8, 8);
        } else {
            if (repeatTime == 1) {
                SetContinuousMask(remainElm);
                vmax(dst, dst, src0 + j * srcRawShape1, 1, 1, 1, 1, 8, 8, 8);
                set_vector_mask(-1, -1);
            } else {
                vmax(dst, dst, src0 + j * srcRawShape1, repeatTime - 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                SetContinuousMask(remainElm);
                vmax(dst + (repeatTime - 1) * rptElm, dst + (repeatTime - 1) * rptElm,
                     src0 + j * srcRawShape1 + (repeatTime - 1) * rptElm, 1, 1, 1, 1, 8, 8, 8);
                set_vector_mask(-1, -1);
            }
        }
        pipe_barrier(PIPE_V);
    }
}

// dim3
template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned dstRawShape1, unsigned dstRawShape2,
          unsigned axis>
TILEOP void DynTrowmaxline_(__ubuf__ T* dst, __ubuf__ T* src0, unsigned TShape0, unsigned TShape1, unsigned TShape2)
{
    static_assert(sizeof(T) == 4 || sizeof(T) == 2);
    if (axis == 0) {
        uint32_t rptElm = REPEAT_BYTE / sizeof(T);
        uint32_t repeatTime = (TShape2 + rptElm - 1) / rptElm;
        uint32_t remainElm = TShape2 % rptElm;
        for (unsigned i = 0; i < TShape1; i++) {
            if (!remainElm) {
                vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2), (__ubuf__ uint32_t*)(src0 + i * srcRawShape2),
                      repeatTime, 1, 1, 8, 8);
            } else {
                if (repeatTime == 1) {
                    SetContinuousMask(remainElm);
                    vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2), (__ubuf__ uint32_t*)(src0 + i * srcRawShape2),
                          1, 1, 1, 8, 8);
                    set_vector_mask(-1, -1);
                } else {
                    vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2), (__ubuf__ uint32_t*)(src0 + i * srcRawShape2),
                          repeatTime - 1, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    SetContinuousMask(remainElm);
                    vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 + (repeatTime - 1) * rptElm),
                          (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 + (repeatTime - 1) * rptElm), 1, 1, 1, 8, 8);
                    set_vector_mask(-1, -1);
                }
            }
            pipe_barrier(PIPE_V);
        }
        pipe_barrier(PIPE_V);
        for (unsigned i = 1; i < TShape0; i++) {
            for (unsigned j = 0; j < TShape1; j++) {
                if (!remainElm) {
                    vmax(dst + j * dstRawShape2, dst + j * dstRawShape2,
                         src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime, 1, 1, 1, 8, 8, 8);
                } else {
                    if (repeatTime == 1) {
                        set_vector_mask(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                        vmax(dst + j * dstRawShape2, dst + j * dstRawShape2,
                             src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime, 1, 1, 1, 8, 8, 8);
                        set_vector_mask(-1, -1);
                    } else {
                        vmax(dst + j * dstRawShape2, dst + j * dstRawShape2,
                             src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime - 1, 1, 1, 1, 8, 8,
                             8);
                        pipe_barrier(PIPE_V);
                        set_vector_mask(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                        vmax(dst + j * dstRawShape2 + (repeatTime - 1) * rptElm,
                             dst + j * dstRawShape2 + (repeatTime - 1) * rptElm,
                             src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2 + (repeatTime - 1) * rptElm, 1,
                             1, 1, 1, 8, 8, 8);
                        set_vector_mask(-1, -1);
                    }
                }
                pipe_barrier(PIPE_V);
            }
        }
    } else if (axis == 1) {
        for (unsigned i = 0; i < TShape0; i++) {
            DynTrowmaxline_<T, srcRawShape2>(dst + i * dstRawShape1 * dstRawShape2,
                                             src0 + i * srcRawShape1 * srcRawShape2, TShape1, TShape2);
        }
    }
}

// dim4
template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned srcRawShape3, unsigned dstRawShape1,
          unsigned dstRawShape2, unsigned dstRawShape3, unsigned axis>
TILEOP void DynTrowmaxline_(__ubuf__ T* dst, __ubuf__ T* src0, unsigned TShape0, unsigned TShape1, unsigned TShape2,
                            unsigned TShape3)
{
    static_assert(sizeof(T) == 4 || sizeof(T) == 2);
    if (TShape0 == 0 || TShape1 == 0 || TShape2 == 0 || TShape3 == 0) {
        return;
    }
    if (axis == 0) {
        uint32_t rptElm = REPEAT_BYTE / sizeof(T);
        uint32_t repeatTime = (TShape3 + rptElm - 1) / rptElm;
        uint32_t remainElm = TShape3 % rptElm;
        for (unsigned i = 0; i < TShape1; i++) {
            for (unsigned j = 0; j < TShape2; j++) {
                if (!remainElm) {
                    vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3),
                          (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3), repeatTime,
                          1, 1, 8, 8);
                } else {
                    if (repeatTime == 1) {
                        SetContinuousMask(remainElm);
                        vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3),
                              (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3), 1, 1, 1,
                              8, 8);
                        set_vector_mask(-1, -1);
                    } else {
                        vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3),
                              (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3),
                              repeatTime - 1, 1, 1, 8, 8);
                        pipe_barrier(PIPE_V);
                        SetContinuousMask(remainElm);
                        vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3 +
                                                   (repeatTime - 1) * rptElm),
                              (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3 +
                                                   (repeatTime - 1) * rptElm),
                              1, 1, 1, 8, 8);
                        set_vector_mask(-1, -1);
                    }
                }
                pipe_barrier(PIPE_V);
            }
        }
        pipe_barrier(PIPE_V);
        for (unsigned i = 1; i < TShape0; i++) {
            for (unsigned j = 0; j < TShape1; j++) {
                for (unsigned k = 0; k < TShape2; k++) {
                    if (!remainElm) {
                        vmax(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                             dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                             src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 + j * srcRawShape2 * srcRawShape3 +
                                 k * srcRawShape3,
                             repeatTime, 1, 1, 1, 8, 8, 8);
                    } else {
                        if (repeatTime == 1) {
                            set_vector_mask(0,
                                            (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                            vmax(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                     j * srcRawShape2 * srcRawShape3 + k * srcRawShape3,
                                 repeatTime, 1, 1, 1, 8, 8, 8);
                            set_vector_mask(-1, -1);
                        } else {
                            vmax(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                     j * srcRawShape2 * srcRawShape3 + k * srcRawShape3,
                                 repeatTime - 1, 1, 1, 1, 8, 8, 8);
                            pipe_barrier(PIPE_V);
                            set_vector_mask(0,
                                            (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                            vmax(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + (repeatTime - 1) * rptElm,
                                 dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + (repeatTime - 1) * rptElm,
                                 src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                     j * srcRawShape2 * srcRawShape3 + k * srcRawShape3 + (repeatTime - 1) * rptElm,
                                 1, 1, 1, 1, 8, 8, 8);
                            set_vector_mask(-1, -1);
                        }
                    }
                    pipe_barrier(PIPE_V);
                }
            }
        }
    } else if (axis == 1 || axis == 2) {
        for (unsigned i = 0; i < TShape0; i++) {
            DynTrowmaxline_<T, srcRawShape2, srcRawShape3, dstRawShape2, dstRawShape3, axis - 1>(
                dst + i * dstRawShape1 * dstRawShape2 * dstRawShape3,
                src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3, TShape1, TShape2, TShape3);
        }
    }
}

// dim2
template <typename T, unsigned srcRawShape1>
TILEOP void DynTrowminline_(__ubuf__ T* dst, __ubuf__ T* src0, unsigned TShape0, unsigned TShape1)
{
    static_assert(sizeof(T) == 4 || sizeof(T) == 2);
    uint32_t rptElm = REPEAT_BYTE / sizeof(T);
    uint32_t repeatTime = (TShape1 + rptElm - 1) / rptElm;
    uint32_t remainElm = TShape1 % rptElm;
    if (!remainElm) {
        vcopy((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src0, repeatTime, 1, 1, 8, 8);
    } else {
        if (repeatTime == 1) {
            SetContinuousMask(remainElm);
            vcopy((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src0, 1, 1, 1, 8, 8);
            set_vector_mask(-1, -1);
        } else {
            vcopy((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src0, repeatTime - 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            SetContinuousMask(remainElm);
            vcopy((__ubuf__ uint32_t*)(dst + (repeatTime - 1) * rptElm),
                  (__ubuf__ uint32_t*)(src0 + (repeatTime - 1) * rptElm), 1, 1, 1, 8, 8);
            set_vector_mask(-1, -1);
        }
    }
    pipe_barrier(PIPE_V);

    for (uint32_t j = 1; j < TShape0; j++) {
        if (!remainElm) {
            vmin(dst, dst, src0 + j * srcRawShape1, repeatTime, 1, 1, 1, 8, 8, 8);
        } else {
            if (repeatTime == 1) {
                SetContinuousMask(remainElm);
                vmin(dst, dst, src0 + j * srcRawShape1, 1, 1, 1, 1, 8, 8, 8);
                set_vector_mask(-1, -1);
            } else {
                vmin(dst, dst, src0 + j * srcRawShape1, repeatTime - 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                SetContinuousMask(remainElm);
                vmin(dst + (repeatTime - 1) * rptElm, dst + (repeatTime - 1) * rptElm,
                     src0 + j * srcRawShape1 + (repeatTime - 1) * rptElm, 1, 1, 1, 1, 8, 8, 8);
                set_vector_mask(-1, -1);
            }
        }
        pipe_barrier(PIPE_V);
    }
}

// dim3
template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned dstRawShape1, unsigned dstRawShape2,
          unsigned axis>
TILEOP void DynTrowminline_(__ubuf__ T* dst, __ubuf__ T* src0, unsigned TShape0, unsigned TShape1, unsigned TShape2)
{
    static_assert(sizeof(T) == 4 || sizeof(T) == 2);
    if (axis == 0) {
        uint32_t rptElm = REPEAT_BYTE / sizeof(T);
        uint32_t repeatTime = (TShape2 + rptElm - 1) / rptElm;
        uint32_t remainElm = TShape2 % rptElm;
        for (unsigned i = 0; i < TShape1; i++) {
            if (!remainElm) {
                vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2), (__ubuf__ uint32_t*)(src0 + i * srcRawShape2),
                      repeatTime, 1, 1, 8, 8);
            } else {
                if (repeatTime == 1) {
                    SetContinuousMask(remainElm);
                    vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2), (__ubuf__ uint32_t*)(src0 + i * srcRawShape2),
                          1, 1, 1, 8, 8);
                    set_vector_mask(-1, -1);
                } else {
                    vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2), (__ubuf__ uint32_t*)(src0 + i * srcRawShape2),
                          repeatTime - 1, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    SetContinuousMask(remainElm);
                    vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 + (repeatTime - 1) * rptElm),
                          (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 + (repeatTime - 1) * rptElm), 1, 1, 1, 8, 8);
                    set_vector_mask(-1, -1);
                }
            }
            pipe_barrier(PIPE_V);
        }
        pipe_barrier(PIPE_V);
        for (unsigned i = 1; i < TShape0; i++) {
            for (unsigned j = 0; j < TShape1; j++) {
                if (!remainElm) {
                    vmin(dst + j * dstRawShape2, dst + j * dstRawShape2,
                         src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime, 1, 1, 1, 8, 8, 8);
                } else {
                    if (repeatTime == 1) {
                        set_vector_mask(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                        vmin(dst + j * dstRawShape2, dst + j * dstRawShape2,
                             src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime, 1, 1, 1, 8, 8, 8);
                        set_vector_mask(-1, -1);
                    } else {
                        vmin(dst + j * dstRawShape2, dst + j * dstRawShape2,
                             src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime - 1, 1, 1, 1, 8, 8,
                             8);
                        pipe_barrier(PIPE_V);
                        set_vector_mask(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                        vmin(dst + j * dstRawShape2 + (repeatTime - 1) * rptElm,
                             dst + j * dstRawShape2 + (repeatTime - 1) * rptElm,
                             src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2 + (repeatTime - 1) * rptElm, 1,
                             1, 1, 1, 8, 8, 8);
                        set_vector_mask(-1, -1);
                    }
                }
                pipe_barrier(PIPE_V);
            }
        }
    } else if (axis == 1) {
        for (unsigned i = 0; i < TShape0; i++) {
            DynTrowminline_<T, srcRawShape2>(dst + i * dstRawShape1 * dstRawShape2,
                                             src0 + i * srcRawShape1 * srcRawShape2, TShape1, TShape2);
        }
    }
}

// dim4
template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned srcRawShape3, unsigned dstRawShape1,
          unsigned dstRawShape2, unsigned dstRawShape3, unsigned axis>
TILEOP void DynTrowminline_(__ubuf__ T* dst, __ubuf__ T* src0, unsigned TShape0, unsigned TShape1, unsigned TShape2,
                            unsigned TShape3)
{
    static_assert(sizeof(T) == 4 || sizeof(T) == 2);
    if (TShape0 == 0 || TShape1 == 0 || TShape2 == 0 || TShape3 == 0) {
        return;
    }
    if (axis == 0) {
        uint32_t rptElm = REPEAT_BYTE / sizeof(T);
        uint32_t repeatTime = (TShape3 + rptElm - 1) / rptElm;
        uint32_t remainElm = TShape3 % rptElm;
        for (unsigned i = 0; i < TShape1; i++) {
            for (unsigned j = 0; j < TShape2; j++) {
                if (!remainElm) {
                    vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3),
                          (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3), repeatTime,
                          1, 1, 8, 8);
                } else {
                    if (repeatTime == 1) {
                        SetContinuousMask(remainElm);
                        vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3),
                              (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3), 1, 1, 1,
                              8, 8);
                        set_vector_mask(-1, -1);
                    } else {
                        vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3),
                              (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3),
                              repeatTime - 1, 1, 1, 8, 8);
                        pipe_barrier(PIPE_V);
                        SetContinuousMask(remainElm);
                        vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3 +
                                                   (repeatTime - 1) * rptElm),
                              (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3 +
                                                   (repeatTime - 1) * rptElm),
                              1, 1, 1, 8, 8);
                        set_vector_mask(-1, -1);
                    }
                }
                pipe_barrier(PIPE_V);
            }
        }
        pipe_barrier(PIPE_V);
        for (unsigned i = 1; i < TShape0; i++) {
            for (unsigned j = 0; j < TShape1; j++) {
                for (unsigned k = 0; k < TShape2; k++) {
                    if (!remainElm) {
                        vmin(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                             dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                             src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 + j * srcRawShape2 * srcRawShape3 +
                                 k * srcRawShape3,
                             repeatTime, 1, 1, 1, 8, 8, 8);
                    } else {
                        if (repeatTime == 1) {
                            set_vector_mask(0,
                                            (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                            vmin(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                     j * srcRawShape2 * srcRawShape3 + k * srcRawShape3,
                                 repeatTime, 1, 1, 1, 8, 8, 8);
                            set_vector_mask(-1, -1);
                        } else {
                            vmin(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                     j * srcRawShape2 * srcRawShape3 + k * srcRawShape3,
                                 repeatTime - 1, 1, 1, 1, 8, 8, 8);
                            pipe_barrier(PIPE_V);
                            set_vector_mask(0,
                                            (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                            vmin(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + (repeatTime - 1) * rptElm,
                                 dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + (repeatTime - 1) * rptElm,
                                 src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                     j * srcRawShape2 * srcRawShape3 + k * srcRawShape3 + (repeatTime - 1) * rptElm,
                                 1, 1, 1, 1, 8, 8, 8);
                            set_vector_mask(-1, -1);
                        }
                    }
                    pipe_barrier(PIPE_V);
                }
            }
        }
    } else if (axis == 1 || axis == 2) {
        for (unsigned i = 0; i < TShape0; i++) {
            DynTrowminline_<T, srcRawShape2, srcRawShape3, dstRawShape2, dstRawShape3, axis - 1>(
                dst + i * dstRawShape1 * dstRawShape2 * dstRawShape3,
                src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3, TShape1, TShape2, TShape3);
        }
    }
}

template <typename T, int64_t cmpOp, int64_t mode>
TILEOP void ProcessCompare(__ubuf__ uint8_t* dst, __ubuf__ T* src0, __ubuf__ T* src1, __ubuf__ uint8_t* tmp,
                           uint64_t countNum, uint64_t repeatNum)
{
    const uint32_t ALIGNMENT = 32;
    const uint32_t vcmpBitsSize = (countNum + 7) / 8;

    __ubuf__ uint8_t* vcmpBitResult = tmp;

    uintptr_t zeroCondAddr = reinterpret_cast<uintptr_t>(vcmpBitResult + vcmpBitsSize);
    zeroCondAddr = (zeroCondAddr + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    __ubuf__ T* zeroCondition = reinterpret_cast<__ubuf__ T*>(zeroCondAddr);

    uintptr_t oneCondAddr = reinterpret_cast<uintptr_t>(zeroCondition + countNum);
    oneCondAddr = (oneCondAddr + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    __ubuf__ T* oneCondition = reinterpret_cast<__ubuf__ T*>(oneCondAddr);

    uintptr_t vselAddr = reinterpret_cast<uintptr_t>(oneCondition + countNum);
    vselAddr = (vselAddr + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    __ubuf__ T* vselResult = reinterpret_cast<__ubuf__ T*>(vselAddr);

    uintptr_t startAddrAddr = reinterpret_cast<uintptr_t>(vselResult + countNum);
    startAddrAddr = (startAddrAddr + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    __ubuf__ uint64_t* startAddrUB = reinterpret_cast<__ubuf__ uint64_t*>(startAddrAddr);

    set_vector_mask(0x0, (uint64_t)countNum);
    auto dst0 = mode == 0 ? vcmpBitResult : dst;
    switch (cmpOp) {
        case 0: // eq
            vcmpv_eq(dst0, src0, src1, repeatNum, 1, 1, 1, 1, 8, 8);
            break;
        case 1: // ne
            vcmpv_ne(dst0, src0, src1, repeatNum, 1, 1, 1, 1, 8, 8);
            break;
        case 2: // lt
            vcmpv_lt(dst0, src0, src1, repeatNum, 1, 1, 1, 1, 8, 8);
            break;
        case 3: // le
            vcmpv_le(dst0, src0, src1, repeatNum, 1, 1, 1, 1, 8, 8);
            break;
        case 4: // gt
            vcmpv_gt(dst0, src0, src1, repeatNum, 1, 1, 1, 1, 8, 8);
            break;
        case 5: // ge
            vcmpv_ge(dst0, src0, src1, repeatNum, 1, 1, 1, 1, 8, 8);
            break;
    }
    if constexpr (mode == 1) {
        return;
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    uint64_t startREG[1] = {0};
    startREG[0] = (uint64_t)((int8_t*)(((uint64_t)((__ubuf__ int8_t*)vcmpBitResult))));
    *(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)startAddrUB) = startREG[0];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    set_cmpmask(((__ubuf__ uint64_t*)startAddrUB));
    vector_dup((__ubuf__ T*)zeroCondition, (T)0.000000e+00f, 1, 1, 1, 8, 0);
    vector_dup((__ubuf__ T*)oneCondition, (T)1.000000e+00f, 1, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    vsel(vselResult, oneCondition, zeroCondition, (uint64_t)571780540399617ULL);
    pipe_barrier(PIPE_V);
    if constexpr (sizeof(T) == 2) {
        vconv_f162u8(dst, vselResult, 1, 1, 1, 4, 8);
    } else if constexpr (sizeof(T) == 4) {
        vconv_f322f16((__ubuf__ half*)vselResult, vselResult, 1, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_f162u8(dst, (__ubuf__ half*)vselResult, 1, 1, 1, 4, 8);
    }
}

template <typename T, int64_t cmpOp, int64_t mode>
TILEOP void DynCompare(__ubuf__ uint8_t* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned T0, __ubuf__ uint8_t* tmp)
{
    constexpr uint64_t COUNT_MAX = 4096 / sizeof(T);
    unsigned numLoop = T0 / COUNT_MAX;
    unsigned remainAfterLoop = T0 % COUNT_MAX;
    uint64_t repeatNum = (COUNT_MAX * sizeof(T) + 255) / 256;
    constexpr int64_t TYPE_REPEAT = 256 / sizeof(T);
    int64_t repeatNumRemain = (remainAfterLoop + TYPE_REPEAT - 1) / TYPE_REPEAT;

    set_mask_count();
    for (int j = 0; j < numLoop; j++) {
        ProcessCompare<T, cmpOp, mode>(dst + j * COUNT_MAX, src0 + j * COUNT_MAX, src1 + j * COUNT_MAX, tmp, COUNT_MAX,
                                       repeatNum);
    }
    if (remainAfterLoop > 0) {
        ProcessCompare<T, cmpOp, mode>(dst + numLoop * COUNT_MAX, src0 + numLoop * COUNT_MAX,
                                       src1 + numLoop * COUNT_MAX, tmp, remainAfterLoop, repeatNumRemain);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename T, unsigned DS0, unsigned DS1, unsigned DS2, unsigned SS0_0, unsigned SS0_1, unsigned SS0_2,
          unsigned SS1_0, unsigned SS1_1, unsigned SS1_2, int64_t cmpOp, int64_t mode>
TILEOP void DynCompare(__ubuf__ uint8_t* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned T0, unsigned T1, unsigned T2,
                       unsigned T3, __ubuf__ uint8_t* tmp)
{
    static_assert((DS2 * sizeof(uint8_t)) % BLOCK_SIZE == 0, "DST dimension 2 not aligned");
    static_assert((SS0_2 * sizeof(T)) % BLOCK_SIZE == 0, "SRC0 dimension 2 not aligned");
    static_assert((SS1_2 * sizeof(T)) % BLOCK_SIZE == 0, "SRC1 dimension 2 not aligned");

    for (int i = 0; i < T0; i++) {
        auto dst_1 = dst;
        auto src0_1 = src0;
        auto src1_1 = src1;
        for (int j = 0; j < T1; j++) {
            auto dst_2 = dst_1;
            auto src0_2 = src0_1;
            auto src1_2 = src1_1;
            for (int k = 0; k < T2; k++) {
                DynCompare<T, cmpOp, mode>(dst_2, src0_2, src1_2, T3, tmp);
                dst_2 += DS2;
                src0_2 += SS0_2;
                src1_2 += SS1_2;
            }
            dst_1 += DS1 * DS2;
            src0_1 += SS0_1 * SS0_2;
            src1_1 += SS1_1 * SS1_2;
        }

        dst += DS0 * DS1 * DS2;
        src0 += SS0_0 * SS0_1 * SS0_2;
        src1 += SS1_0 * SS1_1 * SS1_2;
    }
}

template <typename T, int64_t cmpOp, int64_t mode>
TILEOP void ProcessCmps(__ubuf__ uint8_t* dst, __ubuf__ T* src0, __ubuf__ uint8_t* tmp, uint64_t countNum, T scalarVal,
                        uint64_t repeatNum)
{
    const uint32_t ALIGNMENT = 32;
    const uint32_t vcmpBitsSize = (countNum + 7) / 8;
    __ubuf__ uint8_t* vcmpBitResult = tmp;
    uintptr_t zeroCondAddr = reinterpret_cast<uintptr_t>(vcmpBitResult + vcmpBitsSize);
    zeroCondAddr = (zeroCondAddr + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    __ubuf__ T* zeroCondition = reinterpret_cast<__ubuf__ T*>(zeroCondAddr);

    uintptr_t oneCondAddr = reinterpret_cast<uintptr_t>(zeroCondition + countNum);
    oneCondAddr = (oneCondAddr + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    __ubuf__ T* oneCondition = reinterpret_cast<__ubuf__ T*>(oneCondAddr);

    uintptr_t vselAddr = reinterpret_cast<uintptr_t>(oneCondition + countNum);
    vselAddr = (vselAddr + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    __ubuf__ T* vselResult = reinterpret_cast<__ubuf__ T*>(vselAddr);

    uintptr_t startAddrAddr = reinterpret_cast<uintptr_t>(vselResult + countNum);
    startAddrAddr = (startAddrAddr + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    __ubuf__ uint64_t* startAddrUB = reinterpret_cast<__ubuf__ uint64_t*>(startAddrAddr);

    set_vector_mask(0x0, (uint64_t)countNum);
    auto dst0 = mode == 0 ? vcmpBitResult : dst;

    switch (cmpOp) {
        case 0: // eq
            vcmpvs_eq(dst0, src0, scalarVal, repeatNum, 1, 1, 1, 8);
            break;
        case 1: // ne
            vcmpvs_ne(dst0, src0, scalarVal, repeatNum, 1, 1, 1, 8);
            break;
        case 2: // lt
            vcmpvs_lt(dst0, src0, scalarVal, repeatNum, 1, 1, 1, 8);
            break;
        case 3: // le
            vcmpvs_le(dst0, src0, scalarVal, repeatNum, 1, 1, 1, 8);
            break;
        case 4: // gt
            vcmpvs_gt(dst0, src0, scalarVal, repeatNum, 1, 1, 1, 8);
            break;
        case 5: // ge
            vcmpvs_ge(dst0, src0, scalarVal, repeatNum, 1, 1, 1, 8);
            break;
    }
    if (mode == 1) {
        return;
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    uint64_t startREG[1] = {0};
    startREG[0] = (uint64_t)((int8_t*)(((uint64_t)((__ubuf__ int8_t*)vcmpBitResult))));
    *(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)startAddrUB) = startREG[0];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    set_cmpmask(((__ubuf__ uint64_t*)startAddrUB));
    vector_dup((__ubuf__ T*)zeroCondition, (T)0.000000e+00f, 1, 1, 1, 8, 0);
    vector_dup((__ubuf__ T*)oneCondition, (T)1.000000e+00f, 1, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    vsel(vselResult, oneCondition, zeroCondition, (uint64_t)571780540399617ULL);
    pipe_barrier(PIPE_V);
    if constexpr (sizeof(T) == 2) {
        vconv_f162u8(dst, vselResult, 1, 1, 1, 4, 8);
    } else if constexpr (sizeof(T) == 4) {
        vconv_f322f16((__ubuf__ half*)vselResult, vselResult, 1, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_f162u8(dst, (__ubuf__ half*)vselResult, 1, 1, 1, 4, 8);
    }
}

template <typename T, int64_t cmpOp, int64_t mode>
TILEOP void DynCmps(__ubuf__ uint8_t* dst, __ubuf__ T* src0, unsigned T0, __ubuf__ uint8_t* tmp, T scalarVal)
{
    constexpr uint64_t COUNT_MAX = 4096 / sizeof(T);
    unsigned numLoop = T0 / COUNT_MAX;
    unsigned remainAfterLoop = T0 % COUNT_MAX;
    uint64_t repeatNum = (COUNT_MAX * sizeof(T) + 255) / 256;
    constexpr int64_t TYPE_REPEAT = 256 / sizeof(T);
    int64_t repeatNumRemain = (remainAfterLoop + TYPE_REPEAT - 1) / TYPE_REPEAT;

    set_mask_count();
    for (int j = 0; j < numLoop; j++) {
        ProcessCmps<T, cmpOp, mode>(dst + j * COUNT_MAX, src0 + j * COUNT_MAX, tmp, COUNT_MAX, scalarVal, repeatNum);
    }
    if (remainAfterLoop > 0) {
        ProcessCmps<T, cmpOp, mode>(dst + numLoop * COUNT_MAX, src0 + numLoop * COUNT_MAX, tmp, remainAfterLoop,
                                    scalarVal, repeatNumRemain);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename T, unsigned DS0, unsigned DS1, unsigned DS2, unsigned SS0_0, unsigned SS0_1, unsigned SS0_2,
          int64_t cmpOp, int64_t mode>
TILEOP void DynCmps(__ubuf__ uint8_t* dst, __ubuf__ T* src0, unsigned T0, unsigned T1, unsigned T2, unsigned T3,
                    __ubuf__ uint8_t* tmp, T scalarVal)
{
    static_assert((DS2 * sizeof(uint8_t)) % BLOCK_SIZE == 0, "DST dimension 2 not aligned");
    static_assert((SS0_2 * sizeof(T)) % BLOCK_SIZE == 0, "SRC0 dimension 2 not aligned");

    for (int i = 0; i < T0; i++) {
        auto dst_1 = dst;
        auto src0_1 = src0;
        for (int j = 0; j < T1; j++) {
            auto dst_2 = dst_1;
            auto src0_2 = src0_1;
            for (int k = 0; k < T2; k++) {
                DynCmps<T, cmpOp, mode>(dst_2, src0_2, T3, tmp, scalarVal);
                dst_2 += DS2;
                src0_2 += SS0_2;
            }
            dst_1 += DS1 * DS2;
            src0_1 += SS0_1 * SS0_2;
        }

        dst += DS0 * DS1 * DS2;
        src0 += SS0_0 * SS0_1 * SS0_2;
    }
}

template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned axis0, unsigned axis1>
TILEOP void DynTtransposeMoveOut3dim_(__gm__ T* dst, __ubuf__ T* src, unsigned TShape0, unsigned TShape1,
                                      unsigned TShape2, unsigned dstShape1, unsigned dstShape2)
{
    if (TShape1 == 0 || TShape2 == 0) {
        return;
    }
    if constexpr (axis0 == 0 && axis1 == 1) {
        __gm__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        unsigned nBurst = TShape1;
        unsigned lenBurst = TShape2 * sizeof(T);
        constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
        unsigned srcStride = (srcRawShape2 - TShape2) / blockSize;
        unsigned dstStride = (dstShape1 * dstShape2 - TShape2) * sizeof(T);
        for (int b = 0; b < TShape0; b++) {
            if (sizeof(T) == 2) {
                copy_ubuf_to_gm_align_b16(dst_, src_, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
            } else {
                copy_ubuf_to_gm_align_b32(dst_, src_, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
            }
            dst_ += dstShape2;
            src_ += srcRawShape1 * srcRawShape2;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned srcRawShape3, unsigned axis0,
          unsigned axis1>
TILEOP void DynTtransposeMoveOut4dim_(__gm__ T* dst, __ubuf__ T* src, unsigned TShape0, unsigned TShape1,
                                      unsigned TShape2, unsigned TShape3, unsigned dstShape0, unsigned dstShape1,
                                      unsigned dstShape2, unsigned dstShape3)
{
    if constexpr (axis0 == 1 && axis1 == 2) {
        for (int b = 0; b < TShape0; b++) {
            DynTtransposeMoveOut3dim_<T, srcRawShape2, srcRawShape3, axis0 - 1, axis1 - 1>(
                dst, src, TShape1, TShape2, TShape3, dstShape2, dstShape3);
            dst += dstShape1 * dstShape2 * dstShape3;
            src += srcRawShape1 * srcRawShape2 * srcRawShape3;
        }
    } else if constexpr (axis0 == 0 && axis1 == 2) {
        if (TShape2 == 0 || TShape3 == 0) {
            return;
        }
        for (int i = 0; i < TShape0; i++) {
            __gm__ T* dst0 = dst;
            __ubuf__ T* src0 = src;
            unsigned nBurst = TShape2;
            unsigned lenBurst = TShape3 * sizeof(T);
            constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
            unsigned srcStride = (srcRawShape3 - TShape3) / blockSize;
            unsigned dstStride = (dstShape1 * dstShape2 * dstShape3 - TShape3) * sizeof(T);
            for (int j = 0; j < TShape1; j++) {
                __gm__ T* dst1 = dst0;
                __ubuf__ T* src1 = src0;
                if (sizeof(T) == 2) {
                    copy_ubuf_to_gm_align_b16(dst1, src1, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
                } else {
                    copy_ubuf_to_gm_align_b32(dst1, src1, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
                }
                dst0 += dstShape2 * dstShape3;
                src0 += srcRawShape2 * srcRawShape3;
            }
            dst += dstShape3;
            src += srcRawShape1 * srcRawShape2 * srcRawShape3;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned srcRawShape3, unsigned srcRawShape4,
          unsigned axis0, unsigned axis1>
TILEOP void DynTtransposeMoveOut_(__gm__ T* dst, __ubuf__ T* src, unsigned TShape0, unsigned TShape1, unsigned TShape2,
                                  unsigned TShape3, unsigned TShape4, unsigned dstShape0, unsigned dstShape1,
                                  unsigned dstShape2, unsigned dstShape3, unsigned dstShape4, unsigned GmOffset0,
                                  unsigned GmOffset1, unsigned GmOffset2, unsigned GmOffset3, unsigned GmOffset4)
{
    if constexpr (axis0 == 0 || axis1 == 0) {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
    __gm__ T* dst0 = dst + CalcLinearOffset(dstShape1, dstShape2, dstShape3, dstShape4, GmOffset0, GmOffset1, GmOffset2,
                                            GmOffset3, GmOffset4);
    __ubuf__ T* src0 = src;
    for (int b = 0; b < TShape0; b++) {
        DynTtransposeMoveOut4dim_<T, srcRawShape2, srcRawShape3, srcRawShape4, axis0 - 1, axis1 - 1>(
            dst0, src0, TShape1, TShape2, TShape3, TShape4, dstShape1, dstShape2, dstShape3, dstShape4);
        dst0 += dstShape1 * dstShape2 * dstShape3 * dstShape4;
        src0 += srcRawShape1 * srcRawShape2 * srcRawShape3 * srcRawShape4;
    }
}

template <typename T, unsigned dstRawShape1, unsigned dstRawShape2, unsigned axis0, unsigned axis1>
TILEOP void DynTtransposeMoveIn3dim_(__ubuf__ T* dst, __gm__ T* src, unsigned TShape0, unsigned TShape1,
                                     unsigned TShape2, unsigned srcShape1, unsigned srcShape2)
{
    if (TShape1 == 0 || TShape2 == 0) {
        return;
    }
    if constexpr (axis0 == 0 && axis1 == 1) {
        __ubuf__ T* dst_ = dst;
        __gm__ T* src_ = src;
        unsigned nBurst = TShape1;
        unsigned lenBurst = TShape2 * sizeof(T);
        unsigned srcStride = (srcShape1 * srcShape2 - TShape2) * sizeof(T);
        constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
        unsigned dstStride = (dstRawShape2 - TShape2) / blockSize;
        for (int b = 0; b < TShape0; b++) {
            if (sizeof(T) == 2) {
                copy_gm_to_ubuf_align_b16(dst_, src_, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
            } else {
                copy_gm_to_ubuf_align_b32(dst_, src_, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
            }
            dst_ += dstRawShape1 * dstRawShape2;
            src_ += srcShape2;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

template <typename T, unsigned dstRawShape1, unsigned dstRawShape2, unsigned dstRawShape3, unsigned axis0,
          unsigned axis1>
TILEOP void DynTtransposeMoveIn4dim_(__ubuf__ T* dst, __gm__ T* src, unsigned TShape0, unsigned TShape1,
                                     unsigned TShape2, unsigned TShape3, unsigned srcShape0, unsigned srcShape1,
                                     unsigned srcShape2, unsigned srcShape3)
{
    if constexpr (axis0 == 1 && axis1 == 2) {
        for (int b = 0; b < TShape0; b++) {
            DynTtransposeMoveIn3dim_<T, dstRawShape2, dstRawShape3, axis0 - 1, axis1 - 1>(
                dst, src, TShape1, TShape2, TShape3, srcShape2, srcShape3);
            dst += dstRawShape1 * dstRawShape2 * dstRawShape3;
            src += srcShape1 * srcShape2 * srcShape3;
        }
    } else if constexpr (axis0 == 0 && axis1 == 2) {
        if (TShape2 == 0 || TShape3 == 0) {
            return;
        }
        for (int i = 0; i < TShape0; i++) {
            __ubuf__ T* dst0 = dst;
            __gm__ T* src0 = src;
            unsigned nBurst = TShape2;
            unsigned lenBurst = TShape3 * sizeof(T);
            unsigned srcStride = (srcShape1 * srcShape2 * srcShape3 - TShape3) * sizeof(T);
            constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
            unsigned dstStride = (dstRawShape3 - TShape3) / blockSize;
            for (int j = 0; j < TShape1; j++) {
                __ubuf__ T* dst1 = dst0;
                __gm__ T* src1 = src0;
                if (sizeof(T) == 2) {
                    copy_gm_to_ubuf_align_b16(dst1, src1, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
                } else {
                    copy_gm_to_ubuf_align_b32(dst1, src1, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
                }
                dst0 += dstRawShape2 * dstRawShape3;
                src0 += srcShape2 * srcShape3;
            }
            dst += dstRawShape1 * dstRawShape2 * dstRawShape3;
            src += srcShape3;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

template <typename T>
TILEOP void ProcessLogicalNot(__ubuf__ bool* dst, __ubuf__ T* src, __ubuf__ half* castCondition,
                              __ubuf__ int8_t* vcmpBitResult, __ubuf__ int8_t* compareCondition,
                              __ubuf__ int8_t* oneCondition, __ubuf__ uint64_t* startAddrUB, uint64_t CountNum,
                              int64_t repeatNum)
{
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_mask_count();
    pipe_barrier(PIPE_V);

    set_vector_mask(0x0, (uint64_t)CountNum);
    if constexpr (std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value) {
        vconv_u82f16((__ubuf__ half*)castCondition, (__ubuf__ uint8_t*)src, 1, 1, 1, 8, 4);
    } else if constexpr (std::is_same<T, int8_t>::value) {
        vconv_s82f16((__ubuf__ half*)castCondition, (__ubuf__ int8_t*)src, 1, 1, 1, 8, 4);
    }

    pipe_barrier(PIPE_V);
    if constexpr (std::is_same<T, float>::value) {
        vector_dup((__ubuf__ float*)compareCondition, (float)0.000000e+00f, 1, 1, 1, 8, 0);
        vector_dup((__ubuf__ float*)oneCondition, (float)1.000000e+00f, 1, 1, 1, 8, 0);
    } else {
        vector_dup((__ubuf__ half*)compareCondition, (half)0.000000e+00f, 1, 1, 1, 8, 0);
        vector_dup((__ubuf__ half*)oneCondition, (half)1.000000e+00f, 1, 1, 1, 8, 0);
    }

    set_mask_norm();
    pipe_barrier(PIPE_V);

    if constexpr (std::is_same<T, half>::value || std::is_same<T, float>::value) {
        vcmpv_eq((__ubuf__ uint8_t*)vcmpBitResult, (__ubuf__ T*)src, (__ubuf__ T*)compareCondition, (int64_t)repeatNum,
                 (uint8_t)1ULL, 1, (uint8_t)1ULL, (uint8_t)1ULL, (int64_t)8, (int64_t)8);

    } else if (std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value) {
        vcmpv_eq((__ubuf__ uint8_t*)vcmpBitResult, (__ubuf__ half*)castCondition, (__ubuf__ half*)compareCondition,
                 (int64_t)repeatNum, (uint8_t)1ULL, 1, (uint8_t)1ULL, (uint8_t)1ULL, (int64_t)8, (int64_t)8);
    }

    set_mask_count();
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    uint64_t startREG[1] = {0};
    startREG[0] = (uint64_t)((int8_t*)(((uint64_t)((__ubuf__ int8_t*)vcmpBitResult))));
    *(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)startAddrUB) = startREG[0];

    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

    set_vector_mask(0x0, (uint64_t)CountNum);
    set_cmpmask(((__ubuf__ uint64_t*)startAddrUB));
    pipe_barrier(PIPE_V);

    if (std::is_same<T, float>::value) {
        vsel((__ubuf__ float*)compareCondition, (__ubuf__ float*)oneCondition, (__ubuf__ float*)compareCondition,
             (uint64_t)571780540465409ULL);
        pipe_barrier(PIPE_V);
        vconv_f322f16((__ubuf__ half*)castCondition, (__ubuf__ float*)compareCondition, 1, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_f162s8((__ubuf__ int8_t*)dst, (__ubuf__ half*)castCondition, 1, 1, 1, 4, 8);
    } else {
        vsel((__ubuf__ half*)compareCondition, (__ubuf__ half*)oneCondition, (__ubuf__ half*)compareCondition,
             (uint64_t)571780540465409ULL);
        pipe_barrier(PIPE_V);
        vconv_f162s8((__ubuf__ int8_t*)dst, (__ubuf__ half*)compareCondition, 1, 1, 1, 4, 8);
    }
}

// dim2 & dim1 (T0 = 1 for dim1)
template <typename T, unsigned DS, unsigned SS>
TILEOP void DynTlogicalNot(__ubuf__ bool* dst, __ubuf__ T* src, __ubuf__ int8_t* tmpTensor, unsigned T0, unsigned T1)
{
    constexpr int64_t COUNT_MAX = 2048;
    constexpr int64_t TYPE_SIZE = std::is_same_v<T, float> ? 4 : 2;
    constexpr int64_t TYPE_REPEAT = 256 / TYPE_SIZE;
    constexpr int64_t REPEATNUM = COUNT_MAX / TYPE_REPEAT;

    constexpr uint32_t ALIGN_SIZE = 32;
    uint32_t vcmpBitSize = (COUNT_MAX + 7) / 8;

    __ubuf__ int8_t* vcmpBitResult = reinterpret_cast<__ubuf__ int8_t*>(tmpTensor);

    uintptr_t zeroCondAddr = reinterpret_cast<uintptr_t>(vcmpBitResult + vcmpBitSize);
    zeroCondAddr = (zeroCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ int8_t* compareCondition = reinterpret_cast<__ubuf__ int8_t*>(zeroCondAddr);

    uintptr_t oneCondAddr = reinterpret_cast<uintptr_t>(compareCondition + COUNT_MAX * TYPE_SIZE);
    oneCondAddr = (oneCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ int8_t* oneCondition = reinterpret_cast<__ubuf__ int8_t*>(oneCondAddr);

    uintptr_t castAddr = reinterpret_cast<uintptr_t>(oneCondition + COUNT_MAX * TYPE_SIZE);
    castAddr = (castAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ half* castCondition = reinterpret_cast<__ubuf__ half*>(castAddr);

    uintptr_t startAddrAddr = reinterpret_cast<uintptr_t>(castCondition + COUNT_MAX);
    startAddrAddr = (startAddrAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ uint64_t* startAddrUB = reinterpret_cast<__ubuf__ uint64_t*>(startAddrAddr);

    unsigned numLoop = T1 / COUNT_MAX;
    unsigned remainAfterLoop = T1 % COUNT_MAX;
    int64_t repeatNumRemain = (remainAfterLoop + TYPE_REPEAT - 1) / TYPE_REPEAT;
    for (int i = 0; i < T0; i++) {
        for (int j = 0; j < numLoop; j++) {
            ProcessLogicalNot<T>(dst + i * DS + j * COUNT_MAX, src + i * SS + j * COUNT_MAX, castCondition,
                                 vcmpBitResult, compareCondition, oneCondition, startAddrUB, COUNT_MAX, REPEATNUM);
        }
        if (remainAfterLoop > 0) {
            ProcessLogicalNot<T>(dst + i * DS + numLoop * COUNT_MAX, src + i * SS + numLoop * COUNT_MAX, castCondition,
                                 vcmpBitResult, compareCondition, oneCondition, startAddrUB, remainAfterLoop,
                                 repeatNumRemain);
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

// dim3
template <typename T, unsigned DS0, unsigned DS1, unsigned SS0, unsigned SS1>
TILEOP void DynTlogicalNot(__ubuf__ bool* dst, __ubuf__ T* src, __ubuf__ int8_t* tmpTensor, unsigned T0, unsigned T1,
                           unsigned T2)
{
    static_assert((DS1 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((SS1 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        DynTlogicalNot<T, DS1, SS1>(dst, src, tmpTensor, T1, T2);
        dst += DS0 * DS1;
        src += SS0 * SS1;
    }
}

// dim4
template <typename T, unsigned DS0, unsigned DS1, unsigned DS2, unsigned SS0, unsigned SS1, unsigned SS2>
TILEOP void DynTlogicalNot(__ubuf__ bool* dst, __ubuf__ T* src, __ubuf__ int8_t* tmpTensor, unsigned T0, unsigned T1,
                           unsigned T2, unsigned T3)
{
    static_assert((DS2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((SS2 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ bool* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < T1; j++) {
            DynTlogicalNot<T, DS2, SS2>(dst_, src_, tmpTensor, T2, T3);
            dst_ += DS1 * DS2;
            src_ += SS1 * SS2;
        }
        dst += DS0 * DS1 * DS2;
        src += SS0 * SS1 * SS2;
    }
}
template <typename T_0, typename T_1>
TILEOP void Conv2Float(__ubuf__ bool* dst, __ubuf__ T_0* src0, __ubuf__ T_1* src1, __ubuf__ uint8_t* tmp,
                       uint64_t CountNum)
{
    constexpr uint32_t ALIGN_SIZE = 32;
    const uint64_t vcmpBitsSize = (CountNum + 7) / 8;
    __ubuf__ uint8_t* vcmpBitResult = tmp;

    uintptr_t zeroCondAddr = reinterpret_cast<uintptr_t>(vcmpBitResult + vcmpBitsSize);
    zeroCondAddr = (zeroCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ float* zeroCondition = reinterpret_cast<__ubuf__ float*>(zeroCondAddr);

    uintptr_t oneCondAddr = reinterpret_cast<uintptr_t>(zeroCondition + CountNum);
    oneCondAddr = (oneCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ float* oneCondition = reinterpret_cast<__ubuf__ float*>(oneCondAddr);

    uintptr_t castCondAddr0 = reinterpret_cast<uintptr_t>(oneCondition + CountNum);
    castCondAddr0 = (castCondAddr0 + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ float* castCondition0 = reinterpret_cast<__ubuf__ float*>(castCondAddr0);

    uintptr_t castCondAddr1 = reinterpret_cast<uintptr_t>(castCondition0 + CountNum);
    castCondAddr1 = (castCondAddr1 + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ float* castCondition1 = reinterpret_cast<__ubuf__ float*>(castCondAddr1);

    uintptr_t tmpCondAddr = reinterpret_cast<uintptr_t>(castCondition1 + CountNum);
    tmpCondAddr = (tmpCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ half* tmpCondition = reinterpret_cast<__ubuf__ half*>(tmpCondAddr);

    uintptr_t startAddrAddr = reinterpret_cast<uintptr_t>(tmpCondition + CountNum);
    startAddrAddr = (startAddrAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ uint64_t* startAddrUB = reinterpret_cast<__ubuf__ uint64_t*>(startAddrAddr);

    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_mask_count();
    pipe_barrier(PIPE_V);

    set_vector_mask(0x0, (uint64_t)CountNum);
    // src0 src1转化为float类型
    if constexpr (std::is_same<T_0, bool>::value || std::is_same<T_0, uint8_t>::value) {
        vconv_u82f16(tmpCondition, (__ubuf__ uint8_t*)src0, 1, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vconv_f162f32(castCondition0, tmpCondition, 1, 1, 1, 8, 4);
    } else if constexpr (std::is_same<T_0, int8_t>::value) {
        vconv_s82f16(tmpCondition, (__ubuf__ int8_t*)src0, 1, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vconv_f162f32(castCondition0, tmpCondition, 1, 1, 1, 8, 4);
    } else if constexpr (std::is_same<T_0, half>::value) {
        vconv_f162f32(castCondition0, (__ubuf__ half*)src0, 1, 1, 1, 8, 4);
    }
    pipe_barrier(PIPE_V);
    if constexpr (std::is_same<T_1, bool>::value || std::is_same<T_1, uint8_t>::value) {
        vconv_u82f16(tmpCondition, (__ubuf__ uint8_t*)src1, 1, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vconv_f162f32(castCondition1, tmpCondition, 1, 1, 1, 8, 4);
    } else if constexpr (std::is_same<T_1, int8_t>::value) {
        vconv_s82f16(tmpCondition, (__ubuf__ int8_t*)src1, 1, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vconv_f162f32(castCondition1, tmpCondition, 1, 1, 1, 8, 4);
    } else if constexpr (std::is_same<T_1, half>::value) {
        vconv_f162f32(castCondition1, (__ubuf__ half*)src1, 1, 1, 1, 8, 4);
    }
    pipe_barrier(PIPE_V);
}

template <typename T_0, typename T_1>
TILEOP void ProcessLogicalAnd(__ubuf__ bool* dst, __ubuf__ T_0* src0, __ubuf__ T_1* src1, __ubuf__ uint8_t* tmp,
                              uint64_t CountNum)
{
    constexpr uint32_t ALIGN_SIZE = 32;
    const uint64_t vcmpBitsSize = (CountNum + 7) / 8;
    __ubuf__ uint8_t* vcmpBitResult = tmp;

    uintptr_t zeroCondAddr = reinterpret_cast<uintptr_t>(vcmpBitResult + vcmpBitsSize);
    zeroCondAddr = (zeroCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ float* zeroCondition = reinterpret_cast<__ubuf__ float*>(zeroCondAddr);

    uintptr_t oneCondAddr = reinterpret_cast<uintptr_t>(zeroCondition + CountNum);
    oneCondAddr = (oneCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ float* oneCondition = reinterpret_cast<__ubuf__ float*>(oneCondAddr);

    uintptr_t castCondAddr0 = reinterpret_cast<uintptr_t>(oneCondition + CountNum);
    castCondAddr0 = (castCondAddr0 + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ float* castCondition0 = reinterpret_cast<__ubuf__ float*>(castCondAddr0);

    uintptr_t castCondAddr1 = reinterpret_cast<uintptr_t>(castCondition0 + CountNum);
    castCondAddr1 = (castCondAddr1 + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ float* castCondition1 = reinterpret_cast<__ubuf__ float*>(castCondAddr1);

    uintptr_t tmpCondAddr = reinterpret_cast<uintptr_t>(castCondition1 + CountNum);
    tmpCondAddr = (tmpCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ half* tmpCondition = reinterpret_cast<__ubuf__ half*>(tmpCondAddr);

    uintptr_t startAddrAddr = reinterpret_cast<uintptr_t>(tmpCondition + CountNum);
    startAddrAddr = (startAddrAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ uint64_t* startAddrUB = reinterpret_cast<__ubuf__ uint64_t*>(startAddrAddr);

    uint64_t startREG[1] = {0};
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_mask_count();
    pipe_barrier(PIPE_V);

    set_vector_mask(0x0, (uint64_t)CountNum);
    // 对castcondition0, castcondition1作取绝对值操作，并取其最小值存到castCondition0
    int64_t repeatTime = CountNum * 4 / (32 * 8);
    vector_dup((__ubuf__ float*)zeroCondition, (float)0, 1, 1, 1, 0, 0);
    vector_dup((__ubuf__ float*)oneCondition, (float)1, 1, 1, 1, 0, 0);
    pipe_barrier(PIPE_V);
    if constexpr (std::is_same<T_0, float>::value) {
        vcmpv_eq((__ubuf__ uint8_t*)vcmpBitResult, (__ubuf__ float*)src0, (__ubuf__ float*)zeroCondition, (int64_t)1,
                 (uint8_t)1, 1, (uint8_t)1, (uint8_t)1, (int64_t)8, (int64_t)0);
    } else {
        vcmpv_eq((__ubuf__ uint8_t*)vcmpBitResult, (__ubuf__ float*)castCondition0, (__ubuf__ float*)zeroCondition,
                 (int64_t)1, (uint8_t)1, 1, (uint8_t)1, (uint8_t)1, (int64_t)8, (int64_t)0);
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    startREG[0] = (uint64_t)((int8_t*)(((uint64_t)((__ubuf__ int8_t*)vcmpBitResult))));
    *(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)startAddrUB) = startREG[0];

    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

    set_cmpmask(((__ubuf__ uint64_t*)startAddrUB));
    pipe_barrier(PIPE_V);

    set_mask_count();
    set_vector_mask(0x0, (uint64_t)CountNum);

    vsel(castCondition0, zeroCondition, oneCondition, (uint64_t)571780540465409ULL);
    pipe_barrier(PIPE_V);
    if constexpr (std::is_same<T_1, float>::value) {
        vcmpv_eq((__ubuf__ uint8_t*)vcmpBitResult, (__ubuf__ float*)src1, (__ubuf__ float*)zeroCondition, (int64_t)1,
                 (uint8_t)1, 1, (uint8_t)1, (uint8_t)1, (int64_t)8, (int64_t)0);
    } else {
        vcmpv_eq((__ubuf__ uint8_t*)vcmpBitResult, (__ubuf__ float*)castCondition1, (__ubuf__ float*)zeroCondition,
                 (int64_t)1, (uint8_t)1, 1, (uint8_t)1, (uint8_t)1, (int64_t)8, (int64_t)0);
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    startREG[0] = (uint64_t)((int8_t*)(((uint64_t)((__ubuf__ int8_t*)vcmpBitResult))));
    *(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)startAddrUB) = startREG[0];

    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

    set_cmpmask(((__ubuf__ uint64_t*)startAddrUB));
    pipe_barrier(PIPE_V);

    set_mask_count();
    set_vector_mask(0x0, (uint64_t)CountNum);

    vsel(castCondition1, zeroCondition, oneCondition, (uint64_t)571780540465409ULL);
    pipe_barrier(PIPE_V);
    vmin((__ubuf__ float*)castCondition0, (__ubuf__ float*)castCondition0, (__ubuf__ float*)castCondition1,
         (uint8_t)(repeatTime), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);
    pipe_barrier(PIPE_V);

    vconv_f322f16(tmpCondition, castCondition0, 1, 1, 1, 4, 8);
    pipe_barrier(PIPE_V);
    vconv_f162s8((__ubuf__ int8_t*)dst, tmpCondition, 1, 1, 1, 4, 8);
    pipe_barrier(PIPE_V);
}

// dim2 & dim1 (T0 = 1 for dim1)
template <typename T_0, typename T_1, unsigned DS, unsigned SS0, unsigned SS1>
TILEOP void DynTlogicalAnd(__ubuf__ bool* dst, __ubuf__ T_0* src0, __ubuf__ T_1* src1, __ubuf__ uint8_t* tmp,
                           unsigned T0, unsigned T1)
{
    constexpr uint64_t COUNT_MAX = 64;

    unsigned numLoop = T1 / COUNT_MAX;
    unsigned remainAfterLoop = T1 % COUNT_MAX;
    for (int i = 0; i < T0; i++) {
        for (int j = 0; j < numLoop; j++) {
            Conv2Float<T_0, T_1>(dst + i * DS + j * COUNT_MAX, src0 + i * SS0 + j * COUNT_MAX,
                                 src1 + i * SS1 + j * COUNT_MAX, tmp, COUNT_MAX);
            ProcessLogicalAnd<T_0, T_1>(dst + i * DS + j * COUNT_MAX, src0 + i * SS0 + j * COUNT_MAX,
                                        src1 + i * SS1 + j * COUNT_MAX, tmp, COUNT_MAX);
        }
        if (remainAfterLoop > 0) {
            Conv2Float<T_0, T_1>(dst + i * DS + numLoop * COUNT_MAX, src0 + i * SS0 + numLoop * COUNT_MAX,
                                 src1 + i * SS1 + numLoop * COUNT_MAX, tmp, remainAfterLoop);
            ProcessLogicalAnd<T_0, T_1>(dst + i * DS + numLoop * COUNT_MAX, src0 + i * SS0 + numLoop * COUNT_MAX,
                                        src1 + i * SS1 + numLoop * COUNT_MAX, tmp, remainAfterLoop);
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

// dim3
template <typename T_0, typename T_1, unsigned DS0, unsigned DS1, unsigned SS00, unsigned SS01, unsigned SS10,
          unsigned SS11>
TILEOP void DynTlogicalAnd(__ubuf__ bool* dst, __ubuf__ T_0* src0, __ubuf__ T_1* src1, __ubuf__ uint8_t* tmp,
                           unsigned T0, unsigned T1, unsigned T2)
{
    static_assert((DS1 * sizeof(bool)) % BLOCK_SIZE == 0);
    static_assert((SS01 * sizeof(T_0)) % BLOCK_SIZE == 0);
    static_assert((SS11 * sizeof(T_1)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        DynTlogicalAnd<T_0, T_1, DS1, SS01, SS11>(dst, src0, src1, tmp, T1, T2);
        dst += DS0 * DS1;
        src0 += SS00 * SS01;
        src1 += SS10 * SS11;
    }
}

// dim4
template <typename T_0, typename T_1, unsigned DS0, unsigned DS1, unsigned DS2, unsigned SS00, unsigned SS01,
          unsigned SS02, unsigned SS10, unsigned SS11, unsigned SS12>
TILEOP void DynTlogicalAnd(__ubuf__ bool* dst, __ubuf__ T_0* src0, __ubuf__ T_1* src1, __ubuf__ uint8_t* tmp,
                           unsigned T0, unsigned T1, unsigned T2, unsigned T3)
{
    static_assert((DS2 * sizeof(bool)) % BLOCK_SIZE == 0);
    static_assert((SS02 * sizeof(T_0)) % BLOCK_SIZE == 0);
    static_assert((SS12 * sizeof(T_1)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ bool* dst_ = dst;
        __ubuf__ T_0* src0_ = src0;
        __ubuf__ T_1* src1_ = src1;
        for (int j = 0; j < T1; j++) {
            DynTlogicalAnd<T_0, T_1, DS2, SS02, SS12>(dst_, src0_, src1_, tmp, T2, T3);
            dst_ += DS1 * DS2;
            src0_ += SS01 * SS02;
            src1_ += SS11 * SS12;
        }
        dst += DS0 * DS1 * DS2;
        src0 += SS00 * SS01 * SS02;
        src1 += SS10 * SS11 * SS12;
    }
}

template <typename T, unsigned dstRawShape1, unsigned dstRawShape2, unsigned dstRawShape3, unsigned dstRawShape4,
          unsigned axis0, unsigned axis1>
TILEOP void DynTtransposeMoveIn_(__ubuf__ T* dst, __gm__ T* src, unsigned TShape0, unsigned TShape1, unsigned TShape2,
                                 unsigned TShape3, unsigned TShape4, unsigned srcShape0, unsigned srcShape1,
                                 unsigned srcShape2, unsigned srcShape3, unsigned srcShape4, unsigned GmOffset0,
                                 unsigned GmOffset1, unsigned GmOffset2, unsigned GmOffset3, unsigned GmOffset4)
{
    if constexpr (axis0 == 0 || axis1 == 0) {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
    __ubuf__ T* dst0 = dst;
    __gm__ T* src0 = src + CalcLinearOffset(srcShape1, srcShape2, srcShape3, srcShape4, GmOffset0, GmOffset1, GmOffset2,
                                            GmOffset3, GmOffset4);
    for (int b = 0; b < TShape0; b++) {
        DynTtransposeMoveIn4dim_<T, dstRawShape2, dstRawShape3, dstRawShape4, axis0 - 1, axis1 - 1>(
            dst0, src0, TShape1, TShape2, TShape3, TShape4, srcShape1, srcShape2, srcShape3, srcShape4);
        dst0 += dstRawShape1 * dstRawShape2 * dstRawShape3 * dstRawShape4;
        src0 += srcShape1 * srcShape2 * srcShape3 * srcShape4;
    }
}

// support 2-4 dims
template <typename T, typename T2, unsigned src0RawShape1, unsigned src0RawShape2, unsigned src0RawShape3,
          unsigned src1RawShape1, unsigned src1RawShape2, unsigned src1RawShape3, unsigned dstRawShape1,
          unsigned dstRawShape2, unsigned dstRawShape3, unsigned axis>
TILEOP void DynTgatherElement(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T2* src1, unsigned TShape0, unsigned TShape1,
                              unsigned TShape2, unsigned TShape3)
{
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            for (int k = 0; k < TShape2; ++k) {
                for (int l = 0; l < TShape3; ++l) {
                    T2 index = (T2)(*(src1 + i * src1RawShape1 * src1RawShape2 * src1RawShape3 +
                                      j * src1RawShape2 * src1RawShape3 + k * src1RawShape3 + l)); // indices[i,j,k,l]
                    int src0Offset = 0;
                    int dstOffset = i * dstRawShape1 * dstRawShape2 * dstRawShape3 + j * dstRawShape2 * dstRawShape3 +
                                    k * dstRawShape3 + l;
                    if constexpr (axis == 0) {
                        src0Offset = index * src0RawShape1 * src0RawShape2 * src0RawShape3 +
                                     j * src0RawShape2 * src0RawShape3 + k * src0RawShape3 + l;
                    } else if (axis == 1) {
                        src0Offset = i * src0RawShape1 * src0RawShape2 * src0RawShape3 +
                                     index * src0RawShape2 * src0RawShape3 + k * src0RawShape3 + l;
                    } else if (axis == 2) {
                        src0Offset = i * src0RawShape1 * src0RawShape2 * src0RawShape3 +
                                     j * src0RawShape2 * src0RawShape3 + index * src0RawShape3 + l;
                    } else {
                        src0Offset = i * src0RawShape1 * src0RawShape2 * src0RawShape3 +
                                     j * src0RawShape2 * src0RawShape3 + k * src0RawShape3 + index;
                    }
                    dst[dstOffset] = src0[src0Offset];
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, typename T2>
TILEOP void IndexAddPublicTool(__ubuf__ T* dst, __ubuf__ T* src, T2 alpha, unsigned TShape3, uint64_t dstOffset,
                               uint64_t srcOffset)
{
    uint32_t rptElm = REPEAT_BYTE / sizeof(T);
    uint32_t repeatTime = TShape3 / rptElm;
    uint32_t remainElm = TShape3 % rptElm;
    if (repeatTime) {
        if (abs(static_cast<float>(alpha) - 1) > EPSILON) {
            vmuls(src + srcOffset, src + srcOffset, (T)alpha, repeatTime, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (std::is_same_v<T2, bfloat16_t>) {
                vconv_f322bf16r((__ubuf__ bfloat16_t*)(src + srcOffset), src + srcOffset, repeatTime, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_bf162f32(src + srcOffset, (__ubuf__ bfloat16_t*)(src + srcOffset), repeatTime, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
        }
        vadd(dst + dstOffset, dst + dstOffset, src + srcOffset, repeatTime, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        if constexpr (std::is_same_v<T2, bfloat16_t>) {
            vconv_f322bf16r((__ubuf__ bfloat16_t*)(dst + dstOffset), dst + dstOffset, repeatTime, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vconv_bf162f32(dst + dstOffset, (__ubuf__ bfloat16_t*)(dst + dstOffset), repeatTime, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
    }
    if (remainElm) {
        SetContinuousMask(remainElm);
        if (abs(static_cast<float>(alpha) - 1) > EPSILON) {
            vmuls(src + srcOffset + repeatTime * rptElm, src + srcOffset + repeatTime * rptElm, (T)alpha, 1, 1, 1, 8,
                  8);
            pipe_barrier(PIPE_V);
            if constexpr (std::is_same_v<T2, bfloat16_t>) {
                vconv_f322bf16r((__ubuf__ bfloat16_t*)(src + srcOffset + repeatTime * rptElm),
                                src + srcOffset + repeatTime * rptElm, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_bf162f32(src + srcOffset + repeatTime * rptElm,
                               (__ubuf__ bfloat16_t*)(src + srcOffset + repeatTime * rptElm), 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
        }
        vadd(dst + dstOffset + repeatTime * rptElm, dst + dstOffset + repeatTime * rptElm,
             src + srcOffset + repeatTime * rptElm, 1, 1, 1, 1, 8, 8, 8);
        if constexpr (std::is_same_v<T2, bfloat16_t>) {
            pipe_barrier(PIPE_V);
            vconv_f322bf16r((__ubuf__ bfloat16_t*)(dst + dstOffset + repeatTime * rptElm),
                            dst + dstOffset + repeatTime * rptElm, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vconv_bf162f32(dst + dstOffset + repeatTime * rptElm,
                           (__ubuf__ bfloat16_t*)(dst + dstOffset + repeatTime * rptElm), 1, 1, 1, 8, 8);
        }
        set_vector_mask(-1, -1);
    }
}

template <typename T, typename T1, typename T2>
TILEOP void IndexAddAxis0(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T1* indices, T2 alpha, unsigned TShape0,
                          unsigned TShape1, unsigned TShape2, unsigned TShape3, uint64_t dstBlock1, uint64_t dstBlock2,
                          uint64_t dstBlock3, uint64_t srcBlock1, uint64_t srcBlock2, uint64_t srcBlock3)
{
    uint64_t dstOffset = 0;
    uint64_t srcOffset = 0;
    for (uint32_t idx = 0; idx < TShape0; ++idx) {
        T1 index = *(indices + idx);
        for (uint32_t i = 0; i < TShape1; ++i) {
            for (uint32_t j = 0; j < TShape2; ++j) {
                dstOffset = index * dstBlock1 + i * dstBlock2 + j * dstBlock3;
                srcOffset = idx * srcBlock1 + i * srcBlock2 + j * srcBlock3;
                IndexAddPublicTool<T, T2>(dst, src, alpha, TShape3, dstOffset, srcOffset);
            }
        }
    }
}

template <typename T, typename T1, typename T2>
TILEOP void IndexAddAxis1(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T1* indices, T2 alpha, unsigned TShape0,
                          unsigned TShape1, unsigned TShape2, unsigned TShape3, uint64_t dstBlock1, uint64_t dstBlock2,
                          uint64_t dstBlock3, uint64_t srcBlock1, uint64_t srcBlock2, uint64_t srcBlock3)
{
    uint64_t dstOffset = 0;
    uint64_t srcOffset = 0;
    for (uint32_t i = 0; i < TShape0; ++i) {
        for (uint32_t idx = 0; idx < TShape1; ++idx) {
            T1 index = *(indices + idx);
            for (uint32_t j = 0; j < TShape2; ++j) {
                dstOffset = i * dstBlock1 + index * dstBlock2 + j * dstBlock3;
                srcOffset = i * srcBlock1 + idx * srcBlock2 + j * srcBlock3;
                IndexAddPublicTool<T, T2>(dst, src, alpha, TShape3, dstOffset, srcOffset);
            }
        }
    }
}

template <typename T, typename T1, typename T2>
TILEOP void IndexAddAxis2(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T1* indices, T2 alpha, unsigned TShape0,
                          unsigned TShape1, unsigned TShape2, unsigned TShape3, uint64_t dstBlock1, uint64_t dstBlock2,
                          uint64_t dstBlock3, uint64_t srcBlock1, uint64_t srcBlock2, uint64_t srcBlock3)
{
    uint64_t dstOffset = 0;
    uint64_t srcOffset = 0;
    for (uint32_t i = 0; i < TShape0; ++i) {
        for (uint32_t j = 0; j < TShape1; ++j) {
            for (uint32_t idx = 0; idx < TShape2; ++idx) {
                T1 index = *(indices + idx);
                dstOffset = i * dstBlock1 + j * dstBlock2 + index * dstBlock3;
                srcOffset = i * srcBlock1 + j * srcBlock2 + idx * srcBlock3;
                IndexAddPublicTool<T, T2>(dst, src, alpha, TShape3, dstOffset, srcOffset);
            }
        }
    }
}

template <typename T, typename T1, typename T2>
TILEOP void IndexAddAxis3(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T1* indices, T2 alpha, unsigned TShape0,
                          unsigned TShape1, unsigned TShape2, unsigned TShape3, uint64_t dstBlock1, uint64_t dstBlock2,
                          uint64_t dstBlock3, uint64_t srcBlock1, uint64_t srcBlock2, uint64_t srcBlock3)
{
    uint64_t dstOffset = 0;
    uint64_t srcOffset = 0;
    // 乘法
    if (abs(static_cast<float>(alpha) - 1) > EPSILON) {
        for (uint32_t i = 0; i < TShape0; ++i) {
            for (uint32_t j = 0; j < TShape1; ++j) {
                for (uint32_t k = 0; k < TShape2; ++k) {
                    for (uint32_t idx = 0; idx < TShape3; ++idx) {
                        T1 index = *(indices + idx);
                        dstOffset = i * dstBlock1 + j * dstBlock2 + k * dstBlock3 + index;
                        srcOffset = i * srcBlock1 + j * srcBlock2 + k * srcBlock3 + idx;
                        if constexpr (std::is_same_v<T2, half>) { // half
                            T2 mulsResult = static_cast<float>(src[srcOffset]) * static_cast<float>(alpha);
                            src[srcOffset] = mulsResult;
                        } else if constexpr (std::is_same_v<T2, bfloat16_t>) { // bf16
                            float mulsResult = src[srcOffset] * Bf16ToFp32(alpha);
                            bfloat16_t mulsResBf16 = Fp32ToBf16R(mulsResult);
                            src[srcOffset] = Bf16ToFp32(mulsResBf16);
                        } else { // int8,int16,int32,float32,T2=int8时,T=half
                            T2 mulsResult = static_cast<T2>(src[srcOffset]) * alpha;
                            src[srcOffset] = static_cast<T>(mulsResult);
                        }
                    }
                }
            }
        }
    }
    // 加法
    for (uint32_t i = 0; i < TShape0; ++i) {
        for (uint32_t j = 0; j < TShape1; ++j) {
            for (uint32_t k = 0; k < TShape2; ++k) {
                for (uint32_t idx = 0; idx < TShape3; ++idx) {
                    T1 index = *(indices + idx);
                    dstOffset = i * dstBlock1 + j * dstBlock2 + k * dstBlock3 + index;
                    srcOffset = i * srcBlock1 + j * srcBlock2 + k * srcBlock3 + idx;
                    if constexpr (std::is_same_v<T2, half>) {
                        T2 addResult = static_cast<float>(dst[dstOffset]) + static_cast<float>(src[srcOffset]);
                        dst[dstOffset] = addResult;
                    } else if constexpr (std::is_same_v<T2, bfloat16_t>) {
                        float addResult = dst[dstOffset] + src[srcOffset];
                        bfloat16_t addResBf16 = Fp32ToBf16R(addResult);
                        dst[dstOffset] = Bf16ToFp32(addResBf16);
                    } else { // int8,int16,int32,float32
                        T2 addResult = static_cast<T2>(dst[dstOffset]) + static_cast<T2>(src[srcOffset]);
                        dst[dstOffset] = static_cast<T2>(addResult);
                    }
                }
            }
        }
    }
}

// support 2-4 dim
template <typename T, typename T1, typename T2, unsigned srcRawShape1, unsigned srcRawShape2, unsigned srcRawShape3,
          unsigned dstRawShape1, unsigned dstRawShape2, unsigned dstRawShape3, unsigned axis>
TILEOP void DynTindexAdd(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T1* indices, T2 alpha, unsigned TShape0,
                         unsigned TShape1, unsigned TShape2, unsigned TShape3)
{
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    uint64_t dstBlock1 = dstRawShape1 * dstRawShape2 * dstRawShape3;
    uint64_t dstBlock2 = dstRawShape2 * dstRawShape3;
    uint64_t dstBlock3 = dstRawShape3;
    uint64_t srcBlock1 = srcRawShape1 * srcRawShape2 * srcRawShape3;
    uint64_t srcBlock2 = srcRawShape2 * srcRawShape3;
    uint64_t srcBlock3 = srcRawShape3;

    if constexpr (axis == 0) {
        IndexAddAxis0<T, T1, T2>(dst, src, indices, alpha, TShape0, TShape1, TShape2, TShape3, dstBlock1, dstBlock2,
                                 dstBlock3, srcBlock1, srcBlock2, srcBlock3);
    } else if constexpr (axis == 1) {
        IndexAddAxis1<T, T1, T2>(dst, src, indices, alpha, TShape0, TShape1, TShape2, TShape3, dstBlock1, dstBlock2,
                                 dstBlock3, srcBlock1, srcBlock2, srcBlock3);
    } else if constexpr (axis == 2) {
        IndexAddAxis2<T, T1, T2>(dst, src, indices, alpha, TShape0, TShape1, TShape2, TShape3, dstBlock1, dstBlock2,
                                 dstBlock3, srcBlock1, srcBlock2, srcBlock3);
    } else {
        IndexAddAxis3<T, T1, T2>(dst, src, indices, alpha, TShape0, TShape1, TShape2, TShape3, dstBlock1, dstBlock2,
                                 dstBlock3, srcBlock1, srcBlock2, srcBlock3);
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T>
TILEOP void CumSumPublicTool(__ubuf__ T* dst, __ubuf__ T* input, unsigned TShape3, uint64_t offset, uint32_t idx,
                             uint64_t stride)
{
    uint32_t rptElm = REPEAT_BYTE / sizeof(T);
    uint32_t repeatTime = TShape3 / rptElm;
    uint32_t remainElm = TShape3 % rptElm;

    if (idx == 0) {
        if (repeatTime) {
            vadds(dst + offset, input + offset, 0, repeatTime, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        if (remainElm) {
            SetContinuousMask(remainElm);
            vadds(dst + offset + repeatTime * rptElm, input + offset + repeatTime * rptElm, 0, 1, 1, 1, 8, 8);
            set_vector_mask(-1, -1);
            pipe_barrier(PIPE_V);
        }
    } else {
        if (repeatTime) {
            vadd(dst + offset, input + offset, dst + offset - stride, repeatTime, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        if (remainElm) {
            SetContinuousMask(remainElm);
            vadd(dst + offset + repeatTime * rptElm, input + offset + repeatTime * rptElm,
                 dst + offset + repeatTime * rptElm - stride, 1, 1, 1, 1, 8, 8, 8);
            set_vector_mask(-1, -1);
            pipe_barrier(PIPE_V);
        }
    }
}

template <typename T, unsigned axis>
TILEOP void CumSumAxis0_2(__ubuf__ T* dst, __ubuf__ T* input, unsigned TShape0, unsigned TShape1, unsigned TShape2,
                          unsigned TShape3, uint64_t inputStride1, uint64_t inputStride2, uint64_t inputStride3)
{
    uint64_t offset = 0;
    for (uint32_t i = 0; i < TShape0; ++i) {
        for (uint32_t j = 0; j < TShape1; ++j) {
            for (uint32_t k = 0; k < TShape2; ++k) {
                offset = i * inputStride1 + j * inputStride2 + k * inputStride3;
                if constexpr (axis == 0) {
                    CumSumPublicTool<T>(dst, input, TShape3, offset, i, inputStride1);
                } else if constexpr (axis == 1) {
                    CumSumPublicTool<T>(dst, input, TShape3, offset, j, inputStride2);
                } else if constexpr (axis == 2) {
                    CumSumPublicTool<T>(dst, input, TShape3, offset, k, inputStride3);
                }
            }
        }
    }
}

template <typename T>
TILEOP void CumSumAdd(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, int repeat, int dstBlockStride,
                      int src0BlockStride, int src1BlockStride, int dstRepeatStride, int src0RepeatStride,
                      int src1RepeatStride)
{
    int n = repeat / REPEAT_MAX;
    int rest = repeat % REPEAT_MAX;
    constexpr int elePerRepeat = REPEAT_BYTE / sizeof(T);
    __ubuf__ T* dst_ = dst;
    __ubuf__ T* src0_ = src0;
    __ubuf__ T* src1_ = src1;

    for (size_t i = 0; i < n; i++) {
        vadd(dst_, src0_, src1_, REPEAT_MAX, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
             src0RepeatStride, src1RepeatStride);
        dst_ += REPEAT_MAX * elePerRepeat;
        src0_ += REPEAT_MAX * elePerRepeat;
        src1_ += REPEAT_MAX * elePerRepeat;
    }
    vadd(dst_, src0_, src1_, rest, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
         src1RepeatStride);
}

template <typename T>
TILEOP void CumSumAdds(__ubuf__ T* dst, __ubuf__ T* src0, T src1, int repeat, int dstBlockStride, int src0BlockStride,
                       int dstRepeatStride, int src0RepeatStride)
{
    int n = repeat / REPEAT_MAX;
    int rest = repeat % REPEAT_MAX;
    constexpr int elePerRepeat = REPEAT_BYTE / sizeof(T);

    __ubuf__ T* dst_ = dst;
    __ubuf__ T* src0_ = src0;

    for (size_t i = 0; i < n; i++) {
        vadds(dst_, src0_, src1, REPEAT_MAX, dstBlockStride, src0BlockStride, dstRepeatStride, src0RepeatStride);
        dst_ += REPEAT_MAX * elePerRepeat;
        src0_ += REPEAT_MAX * elePerRepeat;
    }
    vadds(dst_, src0_, src1, rest, dstBlockStride, src0BlockStride, dstRepeatStride, src0RepeatStride);
}

template <typename T, unsigned int RawShape0, unsigned int RawShape1, int Axis>
TILEOP void CumSum2d(__ubuf__ T* dst, __ubuf__ T* src)
{
    if constexpr (Axis == 0) {
        int repeat = RawShape1 * sizeof(T) / REPEAT_BYTE;
        int rest = (RawShape1 * sizeof(T) % REPEAT_BYTE) / sizeof(T);
        constexpr int blockStride = 1;
        constexpr int repeatStride = 8;
        constexpr int elePerRepeat = REPEAT_BYTE / sizeof(T);

        if (repeat > 0) {
            CumSumAdds(dst, src, (T)0, repeat, blockStride, blockStride, repeatStride, repeatStride);
        }
        if (rest > 0) {
            SetContinuousMask(rest);
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
            CumSumAdds(dst + repeat * elePerRepeat, src + repeat * elePerRepeat, (T)0, 1, blockStride, blockStride,
                       repeatStride, repeatStride);
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            set_vector_mask(-1, -1);
        }
        pipe_barrier(PIPE_V);

        int i = 1;
        while (i < RawShape0) {
            repeat = (RawShape0 - i) * RawShape1 * sizeof(T) / REPEAT_BYTE;
            rest = ((RawShape0 - i) * RawShape1 * sizeof(T) % REPEAT_BYTE) / sizeof(T);
            if (repeat > 0) {
                CumSumAdd(dst + i * RawShape1, src, src + i * RawShape1, repeat, blockStride, blockStride, blockStride,
                          repeatStride, repeatStride, repeatStride);
            }
            if (rest > 0) {
                SetContinuousMask(rest);
                set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                CumSumAdd(dst + i * RawShape1 + repeat * elePerRepeat, src + repeat * elePerRepeat,
                          src + i * RawShape1 + repeat * elePerRepeat, 1, blockStride, blockStride, blockStride,
                          repeatStride, repeatStride, repeatStride);
                set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                set_vector_mask(-1, -1);
            }
            pipe_barrier(PIPE_V);
            if (repeat > 0) {
                CumSumAdds(src + i * RawShape1, dst + i * RawShape1, (T)0, repeat, blockStride, blockStride,
                           repeatStride, repeatStride);
            }
            if (rest > 0) {
                SetContinuousMask(rest);
                set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                CumSumAdds(src + i * RawShape1 + repeat * elePerRepeat, dst + i * RawShape1 + repeat * elePerRepeat,
                           (T)0, 1, blockStride, blockStride, repeatStride, repeatStride);
                set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                set_vector_mask(-1, -1);
            }
            pipe_barrier(PIPE_V);
            i *= 2;
        }
    } else {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (size_t i = 0; i < RawShape0; i++) {
            dst_[0] = src_[0];
            for (size_t j = 1; j < RawShape1; j++) {
                if constexpr (std::is_same_v<T, half>) {
                    T tmp = static_cast<float>(src_[j]) + static_cast<float>(dst_[j - 1]);
                    dst_[j] = tmp;
                } else {
                    dst_[j] = dst_[j - 1] + src_[j];
                }
            }
            dst_ += RawShape1;
            src_ += RawShape1;
        }
    }
}

template <int axis, unsigned... RawShapes>
__aicore_host__ constexpr int product()
{
    constexpr size_t rank = sizeof...(RawShapes);
    constexpr std::array<unsigned int, rank> dims = {RawShapes...};
    static_assert(axis >= 0 && axis < rank, "Axis out of bounds");

    int product = 1;
    for (size_t i = axis; i < rank; ++i) {
        product *= dims[i];
    }
    return product;
}

template <typename T, int Axis, unsigned... RawShapes>
TILEOP void CumSum(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr size_t rank = sizeof...(RawShapes);

    if constexpr (Axis == 0) {
        constexpr int dim0 = product<0, RawShapes...>() / product<1, RawShapes...>();
        constexpr int dim1 = product<1, RawShapes...>();
        CumSum2d<T, dim0, dim1, 0>(dst, src);
    } else if constexpr (Axis == (int)rank - 1) {
        constexpr int dim0 = product<0, RawShapes...>() / product<Axis, RawShapes...>();
        constexpr int dim1 = product<Axis, RawShapes...>();
        CumSum2d<T, dim0, dim1, 1>(dst, src);
    } else {
        constexpr int loop = product<0, RawShapes...>() / product<Axis, RawShapes...>();
        constexpr int dim0 = product<Axis, RawShapes...>() / product<Axis + 1, RawShapes...>();
        constexpr int dim1 = product<Axis + 1, RawShapes...>();
        for (size_t i = 0; i < loop; i++) {
            CumSum2d<T, dim0, dim1, 0>(dst + i * dim0 * dim1, src + i * dim0 * dim1);
        }
    }
}

template <typename T, unsigned inputRawShape0, unsigned inputRawShape1, unsigned inputRawShape2,
          unsigned inputRawShape3, unsigned axis, bool flag>
TILEOP void DynTcumSum(__ubuf__ T* dst, __ubuf__ T* input, unsigned TShape0, unsigned TShape1, unsigned TShape2,
                       unsigned TShape3)
{
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    uint64_t inputStride1 = inputRawShape1 * inputRawShape2 * inputRawShape3;
    uint64_t inputStride2 = inputRawShape2 * inputRawShape3;
    uint64_t inputStride3 = inputRawShape3;

    if constexpr (axis != 3) {
        CumSumAxis0_2<T, axis>(dst, input, TShape0, TShape1, TShape2, TShape3, inputStride1, inputStride2,
                               inputStride3);
    } else {
        uint64_t offset = 0;
        for (uint32_t i = 0; i < TShape0; ++i) {
            for (uint32_t j = 0; j < TShape1; ++j) {
                for (uint32_t k = 0; k < TShape2; ++k) {
                    for (uint32_t idx = 0; idx < TShape3; ++idx) {
                        offset = i * inputStride1 + j * inputStride2 + k * inputStride3 + idx;
                        if (idx == 0) {
                            dst[offset] = input[offset];
                        } else {
                            if constexpr (std::is_same_v<T, half>) {
                                int8_t tmp = static_cast<int8_t>(input[offset]) + static_cast<int8_t>(dst[offset - 1]);
                                dst[offset] = tmp;
                            } else {
                                dst[offset] = input[offset] + dst[offset - 1];
                            }
                        }
                    }
                }
            }
        }
    }

    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

constexpr unsigned REDUCE_OP_MAX = 3;
// 2-4dim
template <typename T, typename T1, typename T2, unsigned src1RawShape1, unsigned src1RawShape2, unsigned src1RawShape3,
          unsigned dstRawShape1, unsigned dstRawShape2, unsigned dstRawShape3, unsigned axis, unsigned reduceOp>
TILEOP void DynTscatterElementS(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T1* src1, T2 src2, unsigned src1Shape0,
                                unsigned src1Shape1, unsigned src1Shape2, unsigned src1Shape3)
{
    static_assert(reduceOp < REDUCE_OP_MAX, "Unsupport reduceOp");
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < src1Shape0; ++i) {
        for (int j = 0; j < src1Shape1; ++j) {
            for (int k = 0; k < src1Shape2; ++k) {
                for (int l = 0; l < src1Shape3; ++l) {
                    T1 index = (T1)(*(src1 + i * src1RawShape1 * src1RawShape2 * src1RawShape3 +
                                      j * src1RawShape2 * src1RawShape3 + k * src1RawShape3 + l)); // index[i,j,k,l]
                    int dstOffset = 0;
                    if constexpr (axis == 0) {
                        dstOffset = index * dstRawShape1 * dstRawShape2 * dstRawShape3 +
                                    j * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + l;
                    } else if (axis == 1) {
                        dstOffset = i * dstRawShape1 * dstRawShape2 * dstRawShape3 +
                                    index * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + l;
                    } else if (axis == 2) {
                        dstOffset = i * dstRawShape1 * dstRawShape2 * dstRawShape3 + j * dstRawShape2 * dstRawShape3 +
                                    index * dstRawShape3 + l;
                    } else {
                        dstOffset = i * dstRawShape1 * dstRawShape2 * dstRawShape3 + j * dstRawShape2 * dstRawShape3 +
                                    k * dstRawShape3 + index;
                    }
                    if constexpr (reduceOp == 0) {
                        dst[dstOffset] = src2;
                    } else if constexpr (reduceOp == 1) {
                        dst[dstOffset] = static_cast<T>(static_cast<float>(src2) + static_cast<float>(dst[dstOffset]));
                    } else {
                        dst[dstOffset] = static_cast<T>(static_cast<float>(src2) * static_cast<float>(dst[dstOffset]));
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

// 2-4dim
template <typename T, typename T2, unsigned src1RawShape1, unsigned src1RawShape2, unsigned src1RawShape3,
          unsigned src2RawShape1, unsigned src2RawShape2, unsigned src2RawShape3, unsigned dstRawShape1,
          unsigned dstRawShape2, unsigned dstRawShape3, unsigned axis, unsigned reduceOp>
TILEOP void DynTscatter(__ubuf__ T* dst, __ubuf__ T2* src1, __ubuf__ T* src2, unsigned src1Shape0, unsigned src1Shape1,
                        unsigned src1Shape2, unsigned src1Shape3)
{
    static_assert(reduceOp < REDUCE_OP_MAX, "Unsupport reduceOp");
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < src1Shape0; ++i) {
        for (int j = 0; j < src1Shape1; ++j) {
            for (int k = 0; k < src1Shape2; ++k) {
                for (int l = 0; l < src1Shape3; ++l) {
                    T2 index = (T2)(*(src1 + i * src1RawShape1 * src1RawShape2 * src1RawShape3 +
                                      j * src1RawShape2 * src1RawShape3 + k * src1RawShape3 + l)); // index[i,j,k,l]
                    int src2Offset = i * src2RawShape1 * src2RawShape2 * src2RawShape3 +
                                     j * src2RawShape2 * src2RawShape3 + k * src2RawShape3 + l;
                    int dstOffset = 0;
                    if constexpr (axis == 0) {
                        dstOffset = index * dstRawShape1 * dstRawShape2 * dstRawShape3 +
                                    j * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + l;
                    } else if (axis == 1) {
                        dstOffset = i * dstRawShape1 * dstRawShape2 * dstRawShape3 +
                                    index * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + l;
                    } else if (axis == 2) {
                        dstOffset = i * dstRawShape1 * dstRawShape2 * dstRawShape3 + j * dstRawShape2 * dstRawShape3 +
                                    index * dstRawShape3 + l;
                    } else {
                        dstOffset = i * dstRawShape1 * dstRawShape2 * dstRawShape3 + j * dstRawShape2 * dstRawShape3 +
                                    k * dstRawShape3 + index;
                    }
                    if constexpr (reduceOp == 0) {
                        dst[dstOffset] = src2[src2Offset];
                    } else if constexpr (reduceOp == 1) {
                        dst[dstOffset] = src2[src2Offset] + dst[dstOffset];
                    } else {
                        dst[dstOffset] = src2[src2Offset] * dst[dstOffset];
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, unsigned DS, unsigned SS>
TILEOP void DynTtranspose_vnchwconv_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned T0, unsigned T1,
                                     unsigned TS)
{
    constexpr int block_elem = BLOCK_SIZE / sizeof(T);
    if (((DS % 16) != 0) || ((SS % block_elem) != 0) || ((TS % 16) != 0)) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        for (int i = 0; i < T0; i++) {
            for (int j = 0; j < T1; j++) {
                dst[j * DS + i] = src[i * SS + j];
            }
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        return;
    }
    if (T0 == 0 || T1 == 0) {
        return;
    }
    static_assert(sizeof(T) == 4 || sizeof(T) == 2);
    // go by subtile column, a.k.a. iter in row direction
    int num_subtile_x = (T1 + block_elem - 1) / block_elem;
    int num_subtile_y = T0 / 16;
    if (num_subtile_y) {
        for (int i = 0; i < num_subtile_x; i++) {
            uint64_t srcUb[16] = {0}, tmpUb[16] = {0};
            for (int j = 0; j < 16; j++) {
                srcUb[j] = (uint64_t)(src + i * block_elem + j * SS);
                tmpUb[j] = (sizeof(T) == 2) ? (uint64_t)(tmp + (j + i * block_elem) * TS) :
                                              (uint64_t)(tmp + ((j >> 1) + i * block_elem) * TS + (j & 1) * block_elem);
            }
            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[8]);
            set_va_reg_sb(VA0, tmpUb);
            set_va_reg_sb(VA1, &tmpUb[8]);
            if (sizeof(T) == 2) {
                if (num_subtile_y == 1) {
                    scatter_vnchwconv_b16(VA0, VA2, 1, 0, 0);
                } else {
                    scatter_vnchwconv_b16(VA0, VA2, num_subtile_y, 1, 16 * SS * sizeof(T) / BLOCK_SIZE);
                }
            } else {
                if (num_subtile_y == 1) {
                    scatter_vnchwconv_b32(VA0, VA2, 1, 0, 0);
                } else {
                    scatter_vnchwconv_b32(VA0, VA2, num_subtile_y, 2, 16 * SS * sizeof(T) / BLOCK_SIZE);
                }
            }
        }
    }
    // tail
    int remain_y = T0 % 16;
    if (remain_y) {
        uint64_t srcUb[16] = {0}, tmpUb[16] = {0};
        for (int i = 0; i < remain_y; i++) {
            srcUb[i] = (uint64_t)(src + (num_subtile_y * 16 + i) * SS);
        }
        for (int i = 0; i < 16; i++) {
            tmpUb[i] = (sizeof(T) == 2) ? (uint64_t)(tmp + num_subtile_y * 16 + i * TS) :
                                          (uint64_t)(tmp + num_subtile_y * 16 + (i & 1) * block_elem + (i >> 1) * TS);
        }
        set_va_reg_sb(VA2, srcUb);
        set_va_reg_sb(VA3, &srcUb[8]);
        set_va_reg_sb(VA0, tmpUb);
        set_va_reg_sb(VA1, &tmpUb[8]);
        if (sizeof(T) == 2) {
            if (num_subtile_x == 1) {
                scatter_vnchwconv_b16(VA0, VA2, 1, 0, 0);
            } else {
                scatter_vnchwconv_b16(VA0, VA2, num_subtile_x, block_elem * TS * sizeof(T) / BLOCK_SIZE, 1);
            }
        } else {
            if (num_subtile_x == 1) {
                scatter_vnchwconv_b32(VA0, VA2, 1, 0, 0);
            } else {
                scatter_vnchwconv_b32(VA0, VA2, num_subtile_x, block_elem * TS * sizeof(T) / BLOCK_SIZE, 1);
            }
        }
    }
    // copy to dst
    pipe_barrier(PIPE_V);
    uint16_t lenBurst = (T0 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint16_t srcGap = TS * sizeof(T) / BLOCK_SIZE - lenBurst;
    uint16_t dstGap = DS * sizeof(T) / BLOCK_SIZE - lenBurst;
    copy_ubuf_to_ubuf(dst, tmp, 0, T1, lenBurst, srcGap, dstGap);
}

template <typename T, unsigned DS1, unsigned DS2, unsigned DS3, unsigned DS4, unsigned SS1, unsigned SS2, unsigned SS3,
          unsigned SS4>
TILEOP void DynTtranspose_vnchwconv_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned T0, unsigned T1,
                                     unsigned T2, unsigned T3, unsigned T4)
{
    constexpr int block_elem = BLOCK_SIZE / sizeof(T);
    unsigned TS1 = DS1;
    unsigned TS2 = DS2;
    unsigned TS3 = (T4 + block_elem - 1) / block_elem * block_elem;
    unsigned TS4 = (T3 + 15) / 16 * 16;
    for (unsigned i = 0; i < T0; i++) {
        __ubuf__ T* dst0 = dst;
        __ubuf__ T* src0 = src;
        __ubuf__ T* tmp0 = tmp;
        for (unsigned j = 0; j < T1; j++) {
            __ubuf__ T* dst1 = dst0;
            __ubuf__ T* src1 = src0;
            __ubuf__ T* tmp1 = tmp0;
            for (unsigned k = 0; k < T2; k++) {
                DynTtranspose_vnchwconv_<T, DS4, SS4>(dst1, src1, tmp1, T3, T4, TS4);
                dst1 += DS3 * DS4;
                src1 += SS3 * SS4;
                tmp1 += TS3 * TS4;
            }
            dst0 += DS2 * DS3 * DS4;
            src0 += SS2 * SS3 * SS4;
            tmp0 += TS2 * TS3 * TS4;
        }
        dst += DS1 * DS2 * DS3 * DS4;
        src += SS1 * SS2 * SS3 * SS4;
        tmp += TS1 * TS2 * TS3 * TS4;
    }
}
/**
 * T input 参数类型
 * T2 indices 参数类型
 * input [a,b,c,d],axis=1,index [e,f]
 * result [a,e,f,c,d]
 * before axis轴之前乘积，a
 * after axis轴之后乘积，一次拷贝的长度，cd
 * axis_shape  input在轴上的长度，b
 * UBIndexS*  是index在每个维度的stride，在codegen阶段会扩充到四维，[e,f]->[1,1,e,f]
 * UBOutputS 是output中对应的f，主要是应对 tileshape 不对齐的场景
 *      假设index的维度是[5,5]，经过pass之后，会变成[5,8]，但是output形状是[a,5,5,c,align(d)]
 * TShape* 是index的validshape，用于遍历
 */
template <typename T, typename T2, unsigned before, unsigned after, unsigned axis_shape, unsigned UBIndexS0,
          unsigned UBIndexS1, unsigned UBIndexS2, unsigned UBIndexS3, unsigned UBOutputS>
TILEOP void DynTgatherFromUB_(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T2* src1, unsigned TShape0, unsigned TShape1,
                              unsigned TShape2, unsigned TShape3)
{
    const uint16_t lenBurst = (after * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE; // 一次拷贝的长度
    constexpr uint32_t indexStride34 = UBIndexS3 * UBIndexS2;
    constexpr uint32_t indexStride234 = UBIndexS3 * UBIndexS2 * UBIndexS1;
    constexpr uint32_t indexStride1234 = UBIndexS3 * UBIndexS2 * UBIndexS1 * UBIndexS0;

    constexpr uint32_t oututStride34 = UBOutputS * UBIndexS2;
    constexpr uint32_t oututStride234 = UBOutputS * UBIndexS2 * UBIndexS1;
    constexpr uint32_t oututStride1234 = UBOutputS * UBIndexS2 * UBIndexS1 * UBIndexS0;
    __ubuf__ T2* index = src1;
    __ubuf__ T* output = dst;
    /**
     * gather 操作在 registerInfo 的时候，指定流水为 PIPE_V ，但是当after 为1，也就是 axis
     * 是最后一个维度的时候，会变化成 PIPE_S 操作。
     */
    if constexpr (after == 1) {
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    }

    for (int i = 0; i < before; ++i) {
        for (int j = 0; j < TShape0; ++j) {
            for (int k = 0; k < TShape1; k++) {
                for (int l = 0; l < TShape2; l++) {
                    src1 = index + j * indexStride234 + k * indexStride34 + l * UBIndexS3;
                    dst = output + (j * oututStride234 + k * oututStride34 + l * UBOutputS) * after;
                    for (int m = 0; m < TShape3; m++) {
                        if constexpr (after == 1) {
                            T2 indexInput = (T2)(*(src1 + m));
                            dst[m] = src0[indexInput];
                        } else {
                            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                            T2 indexInput = (T2)(*(src1 + m));
                            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                            // dst, src, sid, nBurst, lenBurst, srcStride, dstStride
                            copy_ubuf_to_ubuf(dst + m * after, src0 + indexInput * after, 0, 1, lenBurst, 1, 1);
                        }
                    }
                }
            }
        }
        src0 += after * axis_shape;
        output += oututStride1234 * after;
    }
    if constexpr (after == 1) {
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    }
}

template <typename T, bool accumulate>
TILEOP void IndexPutCopyOutBase(__gm__ T* dst, __ubuf__ T* src, uint16_t nBurst, uint32_t lenBurst)
{
    if constexpr (accumulate) {
        SetAtomicAddition<T>();
    }
    if constexpr (sizeof(T) == 2) {
        copy_ubuf_to_gm_align_b16(dst, src, 0 /*sid*/, nBurst /*nBurst*/, lenBurst * sizeof(T) /*lenBurst*/,
                                  0 /*left padding count*/, 0 /*right padding count*/, 0 /*ubGap*/, 0 /*gmGap*/);
    } else {
        copy_ubuf_to_gm_align_b32(dst, src, 0 /*sid*/, nBurst /*nBurst*/, lenBurst * sizeof(T) /*lenBurst*/,
                                  0 /*left padding count*/, 0 /*right padding count*/, 0 /*ubGap*/, 0 /*gmGap*/);
    }
    if constexpr (accumulate) {
        set_atomic_none();
    }
}

/*
 * T self/values 类型
 * T2 src2(indices) 类型
 * dst [GmShape0, GmShape1, GmShape2, GmShape3]
 * src1/values: [Tshape0(tile轴), src1RawShape1 src1Rawshape2, src1Rawshape3]
 * src2Dim0: [Tshape0]
 * dstRank: dst/self归一化前的有效秩
 * src1Rank = dstRank - indicesSize + 1
 * indicesSize = 1
 */
template <typename T, typename T2, unsigned dstRank, unsigned src1RawShape1, unsigned src1RawShape2,
          unsigned src1RawShape3, bool accumulate>
TILEOP void DynTIndexPut(__gm__ T* dst, __ubuf__ T* src1, __ubuf__ T2* src2Dim0, unsigned TShape0, unsigned GmShape0,
                         unsigned GmShape1, unsigned GmShape2, unsigned GmShape3)
{
    for (auto i = 0; i < TShape0; i++) {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        T2 indexDim0 = *(reinterpret_cast<__ubuf__ T2*>(src2Dim0 + i));
        if constexpr (dstRank != 1) {
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        }
        uint64_t dstOffset = 0;
        uint64_t src1Offset = 0;
        uint64_t ubNum = 1;
        uint16_t nBurst = 1;
        uint32_t lenBurst = 1;
        if constexpr (dstRank == 1) {
            dstOffset = indexDim0;
            src1[0] = src1[i];
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            TileOp::IndexPutCopyOutBase<T, accumulate>(dst + dstOffset, src1, nBurst, lenBurst);
        } else if constexpr (dstRank == 2) {
            dstOffset = indexDim0 * GmShape3;
            ubNum = src1RawShape3;
            src1Offset = i * ubNum;
            lenBurst = GmShape3;
            TileOp::IndexPutCopyOutBase<T, accumulate>(dst + dstOffset, src1 + src1Offset, nBurst, lenBurst);
        } else if constexpr (dstRank == 3) {
            dstOffset = indexDim0 * GmShape3 * GmShape2;
            ubNum = src1RawShape3 * src1RawShape2;
            src1Offset = i * ubNum;
            nBurst = GmShape2;
            lenBurst = GmShape3;
            TileOp::IndexPutCopyOutBase<T, accumulate>(dst + dstOffset, src1 + src1Offset, nBurst, lenBurst);
        } else if constexpr (dstRank == 4) {
            dstOffset = indexDim0 * GmShape3 * GmShape2 * GmShape1;
            ubNum = src1RawShape3 * src1RawShape2 * src1RawShape1;
            src1Offset = i * ubNum;
            nBurst = GmShape1 * GmShape2;
            lenBurst = GmShape3;
            TileOp::IndexPutCopyOutBase<T, accumulate>(dst + dstOffset, src1 + src1Offset, nBurst, lenBurst);
        }
    }
}

/* indicesSize = 2 */
template <typename T, typename T2, unsigned dstRank, unsigned src1RawShape1, unsigned src1RawShape2,
          unsigned src1RawShape3, bool accumulate>
TILEOP void DynTIndexPut(__gm__ T* dst, __ubuf__ T* src1, __ubuf__ T2* src2Dim0, __ubuf__ T2* src2Dim1,
                         unsigned TShape0, unsigned GmShape0, unsigned GmShape1, unsigned GmShape2, unsigned GmShape3)
{
    for (int i = 0; i < TShape0; i++) {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        T2 indexDim0 = *(reinterpret_cast<__ubuf__ T2*>(src2Dim0 + i));
        T2 indexDim1 = *(reinterpret_cast<__ubuf__ T2*>(src2Dim1 + i));
        if constexpr (dstRank != 2) {
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        }
        uint64_t dstOffset = 0;
        uint64_t src1Offset = 0;
        uint64_t ubNum = 1;
        uint16_t nBurst = 1;
        uint32_t lenBurst = 1;
        if constexpr (dstRank == 2) {
            dstOffset = indexDim0 * GmShape3 + indexDim1;
            src1[0] = src1[i];
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            TileOp::IndexPutCopyOutBase<T, accumulate>(dst + dstOffset, src1, nBurst, lenBurst);
        } else if constexpr (dstRank == 3) {
            dstOffset = indexDim0 * GmShape3 * GmShape2 + indexDim1 * GmShape3;
            ubNum = src1RawShape3;
            src1Offset = i * ubNum;
            lenBurst = GmShape3;
            TileOp::IndexPutCopyOutBase<T, accumulate>(dst + dstOffset, src1 + src1Offset, nBurst, lenBurst);
        } else if constexpr (dstRank == 4) {
            dstOffset = indexDim0 * GmShape3 * GmShape2 * GmShape1 + indexDim1 * GmShape3 * GmShape2;
            ubNum = src1RawShape3 * src1RawShape2;
            src1Offset = i * ubNum;
            nBurst = GmShape2;
            lenBurst = GmShape3;
            TileOp::IndexPutCopyOutBase<T, accumulate>(dst + dstOffset, src1 + src1Offset, nBurst, lenBurst);
        }
    }
}

/* indicesSize = 3 */
template <typename T, typename T2, unsigned dstRank, unsigned src1RawShape1, unsigned src1RawShape2,
          unsigned src1RawShape3, bool accumulate>
TILEOP void DynTIndexPut(__gm__ T* dst, __ubuf__ T* src1, __ubuf__ T2* src2Dim0, __ubuf__ T2* src2Dim1,
                         __ubuf__ T2* src2Dim2, unsigned TShape0, unsigned GmShape0, unsigned GmShape1,
                         unsigned GmShape2, unsigned GmShape3)
{
    for (int i = 0; i < TShape0; i++) {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        T2 indexDim0 = *(reinterpret_cast<__ubuf__ T2*>(src2Dim0 + i));
        T2 indexDim1 = *(reinterpret_cast<__ubuf__ T2*>(src2Dim1 + i));
        T2 indexDim2 = *(reinterpret_cast<__ubuf__ T2*>(src2Dim2 + i));
        if constexpr (dstRank != 3) {
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        }
        uint64_t dstOffset = 0;
        uint64_t src1Offset = 0;
        uint64_t ubNum = 1;
        uint16_t nBurst = 1;
        uint32_t lenBurst = 1;
        if constexpr (dstRank == 3) {
            dstOffset = indexDim0 * GmShape3 * GmShape2 + indexDim1 * GmShape3 + indexDim2;
            src1[0] = src1[i];
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            TileOp::IndexPutCopyOutBase<T, accumulate>(dst + dstOffset, src1, nBurst, lenBurst);
        } else if constexpr (dstRank == 4) {
            dstOffset = indexDim0 * GmShape3 * GmShape2 * GmShape1 + indexDim1 * GmShape3 * GmShape2 +
                        indexDim2 * GmShape3;
            ubNum = src1RawShape3;
            src1Offset = i * ubNum;
            lenBurst = GmShape3;
            TileOp::IndexPutCopyOutBase<T, accumulate>(dst + dstOffset, src1 + src1Offset, nBurst, lenBurst);
        }
    }
}

/* indicesSize = 4 */
template <typename T, typename T2, unsigned dstRank, unsigned src1RawShape1, unsigned src1RawShape2,
          unsigned src1RawShape3, bool accumulate>
TILEOP void DynTIndexPut(__gm__ T* dst, __ubuf__ T* src1, __ubuf__ T2* src2Dim0, __ubuf__ T2* src2Dim1,
                         __ubuf__ T2* src2Dim2, __ubuf__ T2* src2Dim3, unsigned TShape0, unsigned GmShape0,
                         unsigned GmShape1, unsigned GmShape2, unsigned GmShape3)
{
    for (int i = 0; i < TShape0; i++) {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        T2 indexDim0 = *(reinterpret_cast<__ubuf__ T2*>(src2Dim0 + i));
        T2 indexDim1 = *(reinterpret_cast<__ubuf__ T2*>(src2Dim1 + i));
        T2 indexDim2 = *(reinterpret_cast<__ubuf__ T2*>(src2Dim2 + i));
        T2 indexDim3 = *(reinterpret_cast<__ubuf__ T2*>(src2Dim3 + i));
        src1[0] = src1[i];
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        uint64_t dstOffset = indexDim0 * GmShape3 * GmShape2 * GmShape1 + indexDim1 * GmShape3 * GmShape2 +
                             indexDim2 * GmShape3 + indexDim3;
        uint64_t ubNum = 1;
        uint16_t nBurst = 1;
        uint32_t lenBurst = 1;
        TileOp::IndexPutCopyOutBase<T, accumulate>(dst + dstOffset, src1, nBurst, lenBurst);
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
          unsigned reverseOperand>
TILEOP void DynTSadds(__ubuf__ T* dst, __ubuf__ T* src, float scalar, unsigned TShape0, unsigned TShape1)
{
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            T value = (T)(*(src + i * srcShape1 + j));
            int dstOffset = i * dstShape1 + j;
            dst[dstOffset] = scalar + value;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
          unsigned reverseOperand>
TILEOP void DynTSsubs(__ubuf__ T* dst, __ubuf__ T* src, float scalar, unsigned TShape0, unsigned TShape1)
{
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            T value = (T)(*(src + i * srcShape1 + j));
            int dstOffset = i * dstShape1 + j;
            dst[dstOffset] = reverseOperand == 1 ? scalar - value : value - scalar;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
          unsigned reverseOperand>
TILEOP void DynTSmuls(__ubuf__ T* dst, __ubuf__ T* src, float scalar, unsigned TShape0, unsigned TShape1)
{
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            T value = (T)(*(src + i * srcShape1 + j));
            int dstOffset = i * dstShape1 + j;
            dst[dstOffset] = value * scalar;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
          unsigned reverseOperand>
TILEOP void DynTSdivs(__ubuf__ T* dst, __ubuf__ T* src, float scalar, unsigned TShape0, unsigned TShape1)
{
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            int dstOffset = i * dstShape1 + j;
            T value = (T)(*(src + dstOffset));
            dst[dstOffset] = reverseOperand == 1 ? scalar / value : value / scalar;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2, unsigned srcShape0,
          unsigned srcShape1, unsigned srcShape2, unsigned reverseOperand>
TILEOP void DynTSdivs(__ubuf__ T* dst, __ubuf__ T* src, float scalar, unsigned TShape0, unsigned TShape1,
                      unsigned TShape2)
{
    int dstOffset = dstShape1 * dstShape2;
    for (int i = 0; i < TShape0; i++) {
        TileOp::DynTSdivs<T, dstShape0, dstShape1, srcShape0, srcShape1, reverseOperand>(
            dst + i * dstOffset, src + i * dstOffset, scalar, TShape1, TShape2);
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
          unsigned reverseOperand>
TILEOP void DynTSmaxs(__ubuf__ T* dst, __ubuf__ T* src, float scalar, unsigned TShape0, unsigned TShape1)
{
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            T value = (T)(*(src + i * srcShape1 + j));
            int dstOffset = i * dstShape1 + j;
            dst[dstOffset] = value > scalar ? value : scalar;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

const int32_t DEFAULT_REPEAT_STRIDE = 8;
const int32_t NUM_EIGHT = 8;
const int32_t ONE_BLK_SIZE = 32;
template <typename T, int32_t StrideElems>
TILEOP void TRangePropagate(__ubuf__ T* dst, int32_t loopN, int32_t tailSize, T addVal)
{
    if (loopN > 0) {
        set_mask_count();
        set_vector_mask(0, StrideElems);
        for (int32_t i = 0; i < loopN; ++i) {
            vadds(dst + (i + 1) * StrideElems, dst + i * StrideElems, addVal, 1, 1, 1, NUM_EIGHT, NUM_EIGHT);
            pipe_barrier(PIPE_V);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    }

    if (tailSize > 0) {
        set_mask_count();
        set_vector_mask(0, tailSize);
        vadds(dst + (loopN + 1) * StrideElems, dst + loopN * StrideElems, addVal, 1, 1, 1, NUM_EIGHT, NUM_EIGHT);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(-1, -1);
    }
}

template <typename T, unsigned dstShape0>
TILEOP void DynRange(__ubuf__ T* dst, unsigned oriShape0, T baseStart, T step, int64_t tileIdx)
{
    constexpr int32_t kBlkElems = ONE_BLK_SIZE / sizeof(T);
    constexpr int32_t kRepElems = (ONE_BLK_SIZE * DEFAULT_REPEAT_STRIDE) / sizeof(T);

    const unsigned N = oriShape0;
    const T start = baseStart + step * (T)tileIdx;

    if (N <= kBlkElems) {
        for (int32_t j = 0; j < static_cast<int32_t>(N); ++j) {
            dst[j] = start + step * (T)j;
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        return;
    }

    for (int32_t j = 0; j < kBlkElems; ++j) {
        dst[j] = start + step * (T)j;
    }

    int32_t loopN = 0;
    int32_t tailSize = 0;
    if (N >= kRepElems) {
        loopN = DEFAULT_REPEAT_STRIDE - 1;
    } else {
        loopN = static_cast<int32_t>(N) / kBlkElems - 1;
        tailSize = static_cast<int32_t>(N) % kBlkElems;
    }

    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);

    TRangePropagate<T, kBlkElems>(dst, loopN, tailSize, step * (T)kBlkElems);

    if (N <= kRepElems) {
        return;
    }

    loopN = static_cast<int32_t>(N) / kRepElems - 1;
    tailSize = static_cast<int32_t>(N) % kRepElems;
    TRangePropagate<T, kRepElems>(dst, loopN, tailSize, step * (T)kRepElems);
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1, int axis,
          int offset, int isLargest>
TILEOP void DynBitSort(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned oriShape0, unsigned oriShape1)
{
    // 生成index数据,首先创建一个1~8的数组,之后扩展到TShape1,构成0~TShape1的index数组
    // pipe_barrier(PIPE_ALL); // 当前OP无法描述两条流水,UB复用场景存在问题,暂时按照pipe_all规避
    int32_t srcShape1Align = (oriShape1 + 31) / 32 * 32;
    __ubuf__ uint32_t* idx = (__ubuf__ uint32_t*)tmp;
    set_flag(PIPE_V, PIPE_S, EVENT_ID6);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
    for (int32_t j = 0; j < oriShape1; j++) {
        *(idx + j) = (j + offset);
    }
    float FLOAT_MIN = -(0.0 / 0.0);
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);

    // 对于不满足32元素对齐场景,首先将src拷贝到dst的3*srcShape1位置
    if (oriShape1 < 32) {
        uint64_t mask = ~(((static_cast<uint32_t>(1)) << oriShape1) - 1);
        mask = mask & 0xFFFFFFFF;
        for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
            if constexpr (isLargest == 0) {
                set_mask_count();
                set_vector_mask(0, oriShape1);
                vadds((__ubuf__ int32_t*)src + rowIdx * srcShape1, (__ubuf__ int32_t*)src + rowIdx * srcShape1,
                      0x80000000, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                set_mask_norm();
                set_vector_mask(-1, -1);
            }
            set_mask_norm();
            set_vector_mask(0, mask);
            vector_dup(src + rowIdx * srcShape1, FLOAT_MIN, 1, 1, 1, 0, (int64_t)0);
            pipe_barrier(PIPE_V);
            vbitsort((__ubuf__ float*)dst + rowIdx * dstShape1, (__ubuf__ float*)src + rowIdx * srcShape1,
                     (__ubuf__ uint32_t*)idx, 1);
            pipe_barrier(PIPE_V);
            set_vector_mask(-1, -1);
        }
    }

    if (oriShape1 == 32) {
        for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
            // 32个数时，一次完成排序
            __ubuf__ float* srcData = reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1;
            __ubuf__ float* dstData = reinterpret_cast<__ubuf__ float*>(dst) + rowIdx * dstShape1;
            if constexpr (isLargest == 0) {
                set_mask_count();
                set_vector_mask(0, oriShape1);
                vadds((__ubuf__ int32_t*)src + rowIdx * srcShape1, (__ubuf__ int32_t*)src + rowIdx * srcShape1,
                      0x80000000, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                set_mask_norm();
                set_vector_mask(-1, -1);
            }
            vbitsort((__ubuf__ float*)dstData, (__ubuf__ float*)srcData, idx, 1);
            pipe_barrier(PIPE_V);
        }
    }

    if (oriShape1 > 32) {
        int32_t repeat_sort32 = oriShape1 / 32;
        int32_t max_repeat_num = repeat_sort32 / REPEAT_MAX;
        int32_t remain_max_repeat = repeat_sort32 % REPEAT_MAX;
        int32_t tail_sort32 = oriShape1 % 32;
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
            __ubuf__ float* srcData = reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1;
            __ubuf__ float* dstData = reinterpret_cast<__ubuf__ float*>(dst) + rowIdx * dstShape1;
            if constexpr (isLargest == 0) {
                set_mask_count();
                set_vector_mask(0, oriShape1);
                vadds((__ubuf__ int32_t*)src + rowIdx * srcShape1, (__ubuf__ int32_t*)src + rowIdx * srcShape1,
                      0x80000000, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                set_mask_norm();
                set_vector_mask(-1, -1);
            }
            constexpr uint16_t lenBurst = srcShape1 * sizeof(T) / BLOCK_SIZE;
            srcData = reinterpret_cast<__ubuf__ float*>(tmp) + srcShape1Align;
            copy_ubuf_to_ubuf(srcData, src + rowIdx * srcShape1, 0, 1, lenBurst, 0, 0);
            pipe_barrier(PIPE_V);
            if (max_repeat_num > 0) {
                for (int j = 0; j < max_repeat_num; ++j) {
                    vbitsort(dstData + j * REPEAT_MAX * 64, srcData + j * REPEAT_MAX * 32, idx + j * REPEAT_MAX * 32,
                             REPEAT_MAX);
                    pipe_barrier(PIPE_V);
                }
                if (remain_max_repeat) {
                    vbitsort(dstData + max_repeat_num * REPEAT_MAX * 64, srcData + max_repeat_num * REPEAT_MAX * 32,
                             idx + max_repeat_num * REPEAT_MAX * 32, remain_max_repeat);
                    pipe_barrier(PIPE_V);
                }
            } else {
                vbitsort(dstData, srcData, idx, repeat_sort32);
                pipe_barrier(PIPE_V);
            }
            // 首先逐32个数进行排序,需要补齐不对齐的部分
            if (tail_sort32 > 0) {
                // 非整块的时候,首先对尾部补充-nan
                uint64_t mask = ~(((static_cast<uint32_t>(1)) << (tail_sort32)) - 1);
                set_mask_norm();
                set_vector_mask(0, mask);
                vector_dup(srcData + repeat_sort32 * 32, FLOAT_MIN, 1, 1, 1, 8, (int64_t)0);
                pipe_barrier(PIPE_V);
                vbitsort(dstData + repeat_sort32 * 64, srcData + repeat_sort32 * 32, idx + repeat_sort32 * 32, 1);
                pipe_barrier(PIPE_V);
                set_vector_mask(-1, -1);
            }
            pipe_barrier(PIPE_V);
        }
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2, unsigned dstShape3,
          unsigned srcShape0, unsigned srcShape1, unsigned srcShape2, unsigned srcShape3, int axis, int offset,
          int isLargest>
TILEOP void DynBitSort(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned oriShape0, unsigned oriShape1,
                       unsigned oriShape2, unsigned oriShape3)
{
    for (int i = 0; i < oriShape0; ++i) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < oriShape1; ++j) {
            if (oriShape2 != 0 && oriShape3 != 0) {
                TileOp::DynBitSort<T, dstShape2, dstShape3, srcShape2, srcShape3, axis, offset, isLargest>(
                    dst_, src_, tmp, oriShape2, oriShape3);
                dst_ += dstShape2 * dstShape3;
                src_ += srcShape2 * srcShape3;
                pipe_barrier(PIPE_V);
            }
        }
        dst += dstShape1 * dstShape2 * dstShape3;
        src += srcShape1 * srcShape2 * srcShape3;
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1, int axis, int k,
          int mergeSize>
TILEOP void DynMrgSort(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned oriShape0, unsigned oriShape1)
{
    constexpr int32_t kAlign = (k + 7) / 8 * 8; // k需要向32Bytes取整,否则最后搬运出问题
    int32_t totalNum = srcShape1 / 2;
    oriShape1 = oriShape1 / 2;
    for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
        // 每4个合并,计算整块
        int32_t z = mergeSize;
        for (; z * 4 <= oriShape1; z *= 4) {
            __ubuf__ float* srcData = reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1;
            __ubuf__ float* dstData = reinterpret_cast<__ubuf__ float*>(tmp);
            uint64_t config = 0;
            uint32_t repeat_mrg = oriShape1 / (z * 4);
            config |= uint64_t(oriShape1 / (z * 4)); // Xt[7:0]: repeat time
            config |= (uint64_t(0b1111) << 8);       // Xt[11:8]: 4-bit mask signal
            config |= (uint64_t(0b0) << 12);         // Xt[12]: 1-enable input list exhausted suspension

            // 每次计算的数据
            uint64_t src1 = 0;
            src1 |= (uint64_t(z));
            src1 |= (uint64_t(z) << 16);
            src1 |= (uint64_t(z) << 32);
            src1 |= (uint64_t(z) << 48);

            __ubuf__ float* addr_array[4] = {
                (__ubuf__ float*)(srcData + 0 * z * 2), (__ubuf__ float*)(srcData + 1 * z * 2),
                (__ubuf__ float*)(srcData + 2 * z * 2), (__ubuf__ float*)(srcData + 3 * z * 2)};
            pipe_barrier(PIPE_V);
            vmrgsort4(dstData, addr_array, src1, config);
            pipe_barrier(PIPE_V);
            copy_ubuf_to_ubuf((__ubuf__ void*)srcData, (__ubuf__ void*)dstData, 0, 1, z * 4 * repeat_mrg * 2 / 8, 0, 0);
            pipe_barrier(PIPE_V);
        }
        // 合并尾块
        if (z < oriShape1) {
            int32_t arrayCount = 0;
            int32_t mrgArray[15] = {0};
            int32_t tmpInner = oriShape1;
            for (int32_t i = z; i >= 32; i /= 4) {
                int32_t count;
                for (count = 0; count < tmpInner / i; count++) {
                    mrgArray[arrayCount++] = i;
                }
                tmpInner -= count * i;
            }
            if (tmpInner > 0) {
                mrgArray[arrayCount++] = tmpInner;
            }
            uint16_t mrgSortedLen = 0;
            for (int32_t i = 0; i < arrayCount - 1; ++i) {
                __ubuf__ float* srcData = reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1;
                __ubuf__ float* dstData = reinterpret_cast<__ubuf__ float*>(tmp);
                mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
                uint64_t tmpMrgSortedLen = mrgSortedLen;
                uint64_t tmpMrgArray = mrgArray[i + 1];
                if (mrgSortedLen > k) {
                    tmpMrgSortedLen = k;
                }
                if (mrgArray[i + 1] > k) {
                    tmpMrgArray = k;
                }
                uint64_t config = 0;
                config |= uint64_t(1);           // Xt[7:0]: repeat time
                config |= (uint64_t(0b11) << 8); // Xt[11:8]: 4-bit mask signal
                config |= (uint64_t(0b0) << 12); // Xt[12]: 1-enable input list exhausted suspension

                // 每次计算的数据
                uint64_t src1 = 0;
                src1 |= (uint64_t(tmpMrgSortedLen));
                src1 |= (uint64_t(tmpMrgArray) << 16);
                __ubuf__ float* addr_array[4] = {(__ubuf__ float*)(srcData),
                                                 (__ubuf__ float*)(srcData + mrgSortedLen * 2), (__ubuf__ float*)0,
                                                 (__ubuf__ float*)0};
                pipe_barrier(PIPE_V);
                vmrgsort4(dstData, addr_array, src1, config);
                pipe_barrier(PIPE_V);
                copy_ubuf_to_ubuf((__ubuf__ void*)srcData, (__ubuf__ void*)dstData, 0, 1,
                                  ((tmpMrgSortedLen + tmpMrgArray) * 2 + 7) / 8, 0, 0);
                pipe_barrier(PIPE_V);
            }
        }
        copy_ubuf_to_ubuf((__ubuf__ float*)dst + rowIdx * dstShape1, (__ubuf__ float*)src + rowIdx * srcShape1, 0, 1,
                          kAlign / 4, 0, 0);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2, unsigned dstShape3,
          unsigned srcShape0, unsigned srcShape1, unsigned srcShape2, unsigned srcShape3, int axis, int k,
          int mergeSize>
TILEOP void DynMrgSort(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned oriShape0, unsigned oriShape1,
                       unsigned oriShape2, unsigned oriShape3)
{
    for (int i = 0; i < oriShape0; ++i) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < oriShape1; ++j) {
            if (oriShape2 != 0 && oriShape3 != 0) {
                TileOp::DynMrgSort<T, dstShape2, dstShape3, srcShape2, srcShape3, axis, k, mergeSize>(
                    dst_, src_, tmp, oriShape2, oriShape3);
                dst_ += dstShape2 * dstShape3;
                src_ += srcShape2 * srcShape3;
                pipe_barrier(PIPE_V);
            }
        }
        dst += dstShape1 * dstShape2 * dstShape3;
        src += srcShape1 * srcShape2 * srcShape3;
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
          unsigned srcShapeLast, int k>
TILEOP void DynTiledMrgSort(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, __ubuf__ T* src2, __ubuf__ T* src3,
                            __ubuf__ T* tmp, unsigned oriShape0, unsigned oriShape1, unsigned oriShapeLast,
                            int validBit)
{
    constexpr int32_t kAlign = (k + 7) / 8 * 8;
    int32_t kLast = k * 2 > oriShapeLast ? oriShapeLast / 2 : k;
    for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
        if (validBit == 4) {
            __ubuf__ float* src0Data = reinterpret_cast<__ubuf__ float*>(src0) + rowIdx * srcShape1;
            __ubuf__ float* src1Data = reinterpret_cast<__ubuf__ float*>(src1) + rowIdx * srcShape1;
            __ubuf__ float* src2Data = reinterpret_cast<__ubuf__ float*>(src2) + rowIdx * srcShape1;
            __ubuf__ float* src3Data = reinterpret_cast<__ubuf__ float*>(src3) + rowIdx * srcShapeLast;
            uint64_t config = 0;
            config |= uint64_t(1);             // Xt[7:0]: repeat time
            config |= (uint64_t(0b1111) << 8); // Xt[11:8]: 4-bit mask signal
            config |= (uint64_t(0b0) << 12);   // Xt[12]: 1-enable input list exhausted suspension

            // 每次计算的数据
            uint64_t count = 0;
            count |= (uint64_t(k));
            count |= (uint64_t(k) << 16);
            count |= (uint64_t(k) << 32);
            count |= (uint64_t(kLast) << 48);

            __ubuf__ float* addr_array[4] = {(__ubuf__ float*)(src0Data), (__ubuf__ float*)(src1Data),
                                             (__ubuf__ float*)(src2Data), (__ubuf__ float*)(src3Data)};
            pipe_barrier(PIPE_V);
            vmrgsort4(tmp, addr_array, count, config);
            pipe_barrier(PIPE_V);
        }
        if (validBit == 3) {
            __ubuf__ float* src0Data = reinterpret_cast<__ubuf__ float*>(src0) + rowIdx * srcShape1;
            __ubuf__ float* src1Data = reinterpret_cast<__ubuf__ float*>(src1) + rowIdx * srcShape1;
            __ubuf__ float* src2Data = reinterpret_cast<__ubuf__ float*>(src2) + rowIdx * srcShapeLast;
            uint64_t config = 0;
            config |= uint64_t(1);            // Xt[7:0]: repeat time
            config |= (uint64_t(0b111) << 8); // Xt[11:8]: 4-bit mask signal
            config |= (uint64_t(0b0) << 12);  // Xt[12]: 1-enable input list exhausted suspension

            // 每次计算的数据
            uint64_t count = 0;
            count |= (uint64_t(k));
            count |= (uint64_t(k) << 16);
            count |= (uint64_t(kLast) << 32);

            __ubuf__ float* addr_array[4] = {(__ubuf__ float*)(src0Data), (__ubuf__ float*)(src1Data),
                                             (__ubuf__ float*)(src2Data), (__ubuf__ float*)0};
            pipe_barrier(PIPE_V);
            vmrgsort4(tmp, addr_array, count, config);
            pipe_barrier(PIPE_V);
        }
        if (validBit == 2) {
            __ubuf__ float* src0Data = reinterpret_cast<__ubuf__ float*>(src0) + rowIdx * srcShape1;
            __ubuf__ float* src1Data = reinterpret_cast<__ubuf__ float*>(src1) + rowIdx * srcShapeLast;

            uint64_t config = 0;
            config |= uint64_t(1);           // Xt[7:0]: repeat time
            config |= (uint64_t(0b11) << 8); // Xt[11:8]: 4-bit mask signal
            config |= (uint64_t(0b0) << 12); // Xt[12]: 1-enable input list exhausted suspension

            // 每次计算的数据
            uint64_t count = 0;
            count |= (uint64_t(k));
            count |= (uint64_t(kLast) << 16);

            __ubuf__ float* addr_array[4] = {(__ubuf__ float*)(src0Data), (__ubuf__ float*)(src1Data),
                                             (__ubuf__ float*)0, (__ubuf__ float*)0};
            pipe_barrier(PIPE_V);
            vmrgsort4(tmp, addr_array, count, config);
            pipe_barrier(PIPE_V);
        }
        copy_ubuf_to_ubuf((__ubuf__ float*)(dst + rowIdx * dstShape1), (__ubuf__ float*)tmp, 0, 1, kAlign / 4, 0, 0);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2, unsigned dstShape3,
          unsigned srcShape0, unsigned srcShape1, unsigned srcShape2, unsigned srcShape3, unsigned srcShapeLast, int k,
          int validBit>
TILEOP void DynTiledMrgSort(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, __ubuf__ T* src2, __ubuf__ T* src3,
                            __ubuf__ T* tmp, unsigned src0T0, unsigned src0T1, unsigned src0T2, unsigned src0T3,
                            unsigned src1T3, unsigned src2T3, unsigned src3T3)
{
    int validBitNew = validBit;
    if (src0T3 == 0 || src1T3 == 0) {
        return;
    } else if (src2T3 == 0) {
        validBitNew = 2;
        src3T3 = src1T3;
    } else if (src3T3 == 0) {
        validBitNew = 3;
        src3T3 = src2T3;
    }
    for (int i = 0; i < src0T0; ++i) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src0_ = src0;
        __ubuf__ T* src1_ = src1;
        __ubuf__ T* src2_ = src2;
        __ubuf__ T* src3_ = src3;
        for (int j = 0; j < src0T1; ++j) {
            TileOp::DynTiledMrgSort<T, dstShape2, dstShape3, srcShape2, srcShape3, srcShapeLast, k>(
                dst_, src0_, src1_, src2_, src3_, tmp, src0T2, src0T3, src3T3, validBitNew);
            pipe_barrier(PIPE_V);

            dst_ += dstShape2 * dstShape3;
            if (validBitNew == 2) {
                src0_ += srcShape2 * srcShape3;
                src1_ += srcShape2 * srcShapeLast;
            } else if (validBitNew == 3) {
                src0_ += srcShape2 * srcShape3;
                src1_ += srcShape2 * srcShape3;
                src2_ += srcShape2 * srcShapeLast;
            } else if (validBitNew == 4) {
                src0_ += srcShape2 * srcShape3;
                src1_ += srcShape2 * srcShape3;
                src2_ += srcShape2 * srcShape3;
                src3_ += srcShape2 * srcShapeLast;
            }
        }
        dst += dstShape1 * dstShape2 * dstShape3;
        if (validBitNew == 2) {
            src0 += srcShape1 * srcShape2 * srcShape3;
            src1 += srcShape1 * srcShape2 * srcShapeLast;
        } else if (validBitNew == 3) {
            src0 += srcShape1 * srcShape2 * srcShape3;
            src1 += srcShape1 * srcShape2 * srcShape3;
            src2 += srcShape1 * srcShape2 * srcShapeLast;
        } else if (validBitNew == 4) {
            src0 += srcShape1 * srcShape2 * srcShape3;
            src1 += srcShape1 * srcShape2 * srcShape3;
            src2 += srcShape1 * srcShape2 * srcShape3;
            src3 += srcShape1 * srcShape2 * srcShapeLast;
        }
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
          unsigned firstShape>
TILEOP void DynTwoTileMrgSort(__ubuf__ T* dst, __ubuf__ T* src, unsigned oriShape0, unsigned oriShape1)
{
    unsigned oriShape1Align = (oriShape1 + 7) / 8 * 8; // copy_ubuf_to_ubuf 需要32B对齐
    for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
        if (oriShape1 <= firstShape) {
            pipe_barrier(PIPE_V);
            copy_ubuf_to_ubuf((__ubuf__ float*)(dst) + rowIdx * dstShape1,
                              reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1, 0, 1,
                              oriShape1Align * 4 / 32, 0, 0);
            pipe_barrier(PIPE_V);
            continue;
        }

        __ubuf__ float* src0Data = reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1;
        __ubuf__ float* src1Data = reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1 + firstShape;
        __ubuf__ float* addr_array[4] = {(__ubuf__ float*)src0Data, (__ubuf__ float*)src1Data, (__ubuf__ float*)0,
                                         (__ubuf__ float*)0};

        uint64_t config = 0;
        config |= uint64_t(1);
        config |= (uint64_t(0b11) << 8);
        config |= (uint64_t(0b0) << 12);

        uint64_t count = 0;
        count |= (uint64_t(firstShape) / 2);
        count |= ((uint64_t(oriShape1 - firstShape) / 2) << 16);

        pipe_barrier(PIPE_V);
        vmrgsort4((__ubuf__ float*)(dst) + rowIdx * dstShape1, addr_array, count, config);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2, unsigned dstShape3,
          unsigned srcShape0, unsigned srcShape1, unsigned srcShape2, unsigned srcShape3, unsigned firstShape>
TILEOP void DynTwoTileMrgSort(__ubuf__ T* dst, __ubuf__ T* src, unsigned oriShape0, unsigned oriShape1,
                              unsigned oriShape2, unsigned oriShape3)
{
    if (oriShape2 == 0 || oriShape3 == 0) {
        return;
    }
    for (int i = 0; i < oriShape0; i++) {
        __ubuf__ T* src_ = src;
        __ubuf__ T* dst_ = dst;
        for (int j = 0; j < oriShape1; j++) {
            TileOp::DynTwoTileMrgSort<T, dstShape2, dstShape3, srcShape2, srcShape3, firstShape>(dst_, src_, oriShape2,
                                                                                                 oriShape3);
            pipe_barrier(PIPE_V);
            src_ += srcShape2 * srcShape3;
            dst_ += dstShape2 * dstShape3;
        }
        src += srcShape1 * srcShape2 * srcShape3;
        dst += dstShape1 * dstShape2 * dstShape3;
    }
}

template <typename T, typename U, int k, unsigned dstRawShape1, int extractMode, int isLargest>
TILEOP void DynExtract(__ubuf__ T* dst, __ubuf__ U* src, unsigned TShape0)
{
    uint64_t repeat = static_cast<uint64_t>(TShape0 * dstRawShape1 * 2 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t srcBlockStride = 1;
    constexpr uint8_t srcRepeatStride = 8;
    // mode trans, extractMode == 0 取偶数位， extractMode == 1 取奇数位
    int patternMode = 1;
    if constexpr (extractMode == 1) {
        patternMode = 2;
    }
    uint64_t elems = TShape0 * dstRawShape1;
    set_mask_count();
    set_vector_mask(0, elems * 2);
    vreducev2((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src, (__ubuf__ uint32_t*)src, 1, srcBlockStride, patternMode,
              srcRepeatStride, 0);
    set_mask_norm();
    set_vector_mask(-1, -1);
    pipe_barrier(PIPE_V);

    if constexpr (extractMode == 0 && isLargest == 0) {
        // 按照升序排序时,对于value需要乘以-1,恢复原始值
        set_mask_count();
        set_vector_mask(0, TShape0 * dstRawShape1);
        vadds(reinterpret_cast<__ubuf__ int32_t*>(dst), reinterpret_cast<__ubuf__ int32_t*>(dst), 0x80000000, 1, 1, 1,
              8, 8);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, typename U, unsigned dstRawShape1, unsigned dstRawShape2, unsigned dstRawShape3, int k,
          int extractMode, int isLargest>
TILEOP void DynExtract(__ubuf__ T* dst, __ubuf__ U* src, unsigned TShape0, unsigned TShape1, unsigned TShape2)
{
    for (int i = 0; i < TShape0; ++i) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ U* src_ = src;
        for (int j = 0; j < TShape1; ++j) {
            if (TShape2 != 0) {
                TileOp::DynExtract<T, U, k, dstRawShape3, extractMode, isLargest>(dst_, src_, TShape2);
                dst_ += dstRawShape2 * dstRawShape3;
                src_ += dstRawShape2 * dstRawShape3 * 2;
                pipe_barrier(PIPE_V);
            }
        }
        dst += dstRawShape1 * dstRawShape2 * dstRawShape3;
        src += dstRawShape1 * dstRawShape2 * dstRawShape3 * 2;
    }
}

template <typename T, typename U, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
          int extractMode, int isLargest>
TILEOP void DynExtractSingle(__ubuf__ T* dst, __ubuf__ U* src, unsigned oriShape0, unsigned oriShape1)
{
    constexpr uint8_t srcBlockStride = 1;
    constexpr uint8_t srcRepeatStride = 8;
    int patternMode = 1;
    if constexpr (extractMode == 1) {
        patternMode = 2;
    }

    for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
        uint64_t elements = oriShape1 / 2;
        set_mask_count();
        set_vector_mask(0, elements * 2);
        vreducev2((__ubuf__ uint32_t*)dst + rowIdx * dstShape1, (__ubuf__ uint32_t*)src + rowIdx * srcShape1,
                  (__ubuf__ uint32_t*)src + rowIdx * srcShape1, 1, srcBlockStride, patternMode, srcRepeatStride, 0);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);

        if constexpr (extractMode == 0 && isLargest == 0) {
            set_mask_count();
            set_vector_mask(0, elements);
            vadds(reinterpret_cast<__ubuf__ int32_t*>(dst) + rowIdx * dstShape1,
                  reinterpret_cast<__ubuf__ int32_t*>(dst) + rowIdx * dstShape1, 0x80000000, 1, 1, 1, 8, 8);
            set_mask_norm();
            set_vector_mask(-1, -1);
            pipe_barrier(PIPE_V);
        }
    }
}

template <typename T, typename U, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2, unsigned dstShape3,
          unsigned srcShape0, unsigned srcShape1, unsigned srcShape2, unsigned srcShape3, int extractMode,
          int isLargest>
TILEOP void DynExtractSingle(__ubuf__ T* dst, __ubuf__ U* src, unsigned oriShape0, unsigned oriShape1,
                             unsigned oriShape2, unsigned oriShape3)
{
    if (oriShape2 == 0 || oriShape3 == 0) {
        return;
    }
    for (int i = 0; i < oriShape0; i++) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ U* src_ = src;
        for (int j = 0; j < oriShape1; j++) {
            DynExtractSingle<T, U, dstShape2, dstShape3, srcShape2, srcShape3, extractMode, isLargest>(
                dst_, src_, oriShape2, oriShape3);
            pipe_barrier(PIPE_V);
            src_ += srcShape2 * srcShape3;
            dst_ += dstShape2 * dstShape3;
        }
        src += srcShape1 * srcShape2 * srcShape3;
        dst += dstShape1 * dstShape2 * dstShape3;
    }
}

template <typename T, typename idxT, unsigned xShape0, unsigned xShape1, unsigned idxShape0, unsigned idxShape1,
          int descending, int idxStart>
TILEOP void DynSort(__ubuf__ T* y, __ubuf__ idxT* yIdx, __ubuf__ T* tmp, __ubuf__ T* x)
{
    TileOp::Sort<T, idxT, xShape0, xShape1, idxShape0, idxShape1, descending, idxStart>(y, yIdx, tmp, x);
}

template <typename T, typename idxT, unsigned xShape0, unsigned xShape1, unsigned idxShape0, unsigned idxShape1,
          int fullSort, int descending>
TILEOP void DynMerge(__ubuf__ T* y, __ubuf__ idxT* yIdx, __ubuf__ T* tmp, __ubuf__ T* x, __ubuf__ idxT* idx)
{
    TileOp::Merge<T, idxT, xShape0, xShape1, idxShape0, idxShape1, fullSort, descending>(y, yIdx, tmp, x, idx);
}

template <typename T, typename idxT, unsigned xShape0, unsigned xShape1, unsigned idxShape0, unsigned idxShape1,
          int descending>
TILEOP void DynCompareAndSwap(__ubuf__ T* y0, __ubuf__ idxT* yIdx0, __ubuf__ T* y1, __ubuf__ idxT* yIdx1,
                              __ubuf__ T* x0, __ubuf__ idxT* idx0, __ubuf__ T* x1, __ubuf__ idxT* idx1)
{
    TileOp::CompareAndSwap<T, idxT, xShape0, xShape1, idxShape0, idxShape1, descending>(y0, yIdx0, y1, yIdx1, x0, idx0,
                                                                                        x1, idx1);
}

template <typename T, typename U>
TILEOP void ProcessWhere(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, __ubuf__ U* condition,
                         __ubuf__ half* castCondition, __ubuf__ int8_t* vcmpBitResult, __ubuf__ half* compareCondition,
                         __ubuf__ uint64_t* startAddrUB, unsigned repeatNum, unsigned elementsCount)
{
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    pipe_barrier(PIPE_V);
    set_mask_count();
    set_vector_mask(0x0, (uint64_t)elementsCount);
    pipe_barrier(PIPE_V);
    if constexpr (std::is_same_v<U, bool>) {
        vconv_u82f16((__ubuf__ half*)castCondition, (__ubuf__ unsigned char*)condition, repeatNum, 1, 1, 8, 4);
        vector_dup((__ubuf__ half*)compareCondition, (half)1.000000e+00f, repeatNum, 1, 0, 8, 0);
        set_mask_norm();
        pipe_barrier(PIPE_V);
        vcmpv_eq((__ubuf__ unsigned char*)vcmpBitResult, (__ubuf__ half*)castCondition,
                 (__ubuf__ half*)compareCondition, repeatNum, 1, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0x0, (uint64_t)elementsCount);
        pipe_barrier(PIPE_V);
    } else {
        vcmpBitResult = (__ubuf__ int8_t*)condition;
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    uint64_t startREG[1] = {0};
    startREG[0] = (uint64_t)vcmpBitResult;
    *startAddrUB = startREG[0];
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);

    set_cmpmask((__ubuf__ uint64_t*)startAddrUB);
    pipe_barrier(PIPE_V);
    vsel(dst, src0, src1, 16, 1, 1, 1, 8, 8, 8, 2);
}

template <typename T, typename U, unsigned DS, unsigned CS, unsigned SS0>
TILEOP void DynWhere_TT(__ubuf__ T* dst, __ubuf__ uint8_t* temp, __ubuf__ U* condition, __ubuf__ T* src0,
                        __ubuf__ T* src1, unsigned T0, unsigned T1)
{
    unsigned elementsPerCount = 1024;
    unsigned adressUsed = 4;
    unsigned bitsOfByte = 8;
    __ubuf__ half* castCondition = (__ubuf__ half*)temp;
    __ubuf__ half* compareCondition = (__ubuf__ half*)(castCondition + elementsPerCount);
    __ubuf__ int8_t* vcmpBitResult = (__ubuf__ int8_t*)(compareCondition + elementsPerCount);
    __ubuf__ uint64_t* startAddrUB = (__ubuf__ uint64_t*)(vcmpBitResult + elementsPerCount / bitsOfByte);

    unsigned numCountPerLine = T1 / elementsPerCount;
    unsigned elementsRemainPerLine = T1 % elementsPerCount;
    unsigned repeatNum = (elementsPerCount * sizeof(half) + REPEAT_BYTE - 1) / REPEAT_BYTE;
    unsigned repeatNumRemain = (elementsRemainPerLine * sizeof(half) + REPEAT_BYTE - 1) / REPEAT_BYTE;
    if constexpr (std::is_same_v<U, bool>) {
        for (int i = 0; i < T0; i++) {
            for (int j = 0; j < numCountPerLine; j++) {
                ProcessWhere(dst + i * DS + j * elementsPerCount, src0 + i * SS0 + j * elementsPerCount,
                             src1 + i * SS0 + j * elementsPerCount, condition + i * CS + j * elementsPerCount,
                             castCondition, (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNum, elementsPerCount);
            }
            if (elementsRemainPerLine) {
                ProcessWhere(dst + i * DS + elementsPerCount * numCountPerLine,
                             src0 + i * SS0 + elementsPerCount * numCountPerLine,
                             src1 + i * SS0 + elementsPerCount * numCountPerLine,
                             condition + i * CS + elementsPerCount * numCountPerLine, castCondition,
                             (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNumRemain, elementsRemainPerLine);
            }
        }
    } else {
        for (int i = 0; i < T0; i++) {
            for (int j = 0; j < numCountPerLine; j++) {
                ProcessWhere(dst + i * DS + j * elementsPerCount, src0 + i * SS0 + j * elementsPerCount,
                             src1 + i * SS0 + j * elementsPerCount, condition + i * CS + j * elementsPerCount / 8,
                             castCondition, (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNum, elementsPerCount);
            }
            if (elementsRemainPerLine) {
                ProcessWhere(dst + i * DS + elementsPerCount * numCountPerLine,
                             src0 + i * SS0 + elementsPerCount * numCountPerLine,
                             src1 + i * SS0 + elementsPerCount * numCountPerLine,
                             condition + i * CS + elementsPerCount * numCountPerLine / 8, castCondition,
                             (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNumRemain, elementsRemainPerLine);
            }
        }
    }
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
}

// dim4
template <typename T, typename U, unsigned DS0, unsigned DS1, unsigned DS2, unsigned CS0, unsigned CS1, unsigned CS2,
          unsigned S0S0, unsigned S0S1, unsigned S0S2>
TILEOP void DynWhere_TT(__ubuf__ T* dst, __ubuf__ uint8_t* temp, __ubuf__ U* condition, __ubuf__ T* src0,
                        __ubuf__ T* src1, unsigned T0, unsigned T1, unsigned T2, unsigned T3)
{
    static_assert((DS2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((CS2 * sizeof(bool)) % BLOCK_SIZE == 0);
    static_assert((S0S2 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ T* dstLoop = dst;
        __ubuf__ U* conditionLoop = condition;
        __ubuf__ T* src0Loop = src0;
        __ubuf__ T* src1Loop = src1;
        for (int j = 0; j < T1; j++) {
            DynWhere_TT<T, U, DS2, CS2, S0S2>(dstLoop, temp, conditionLoop, src0Loop, src1Loop, T2, T3);
            dstLoop += DS1 * DS2;
            conditionLoop += CS1 * CS2;
            src0Loop += S0S1 * S0S2;
            src1Loop += S0S1 * S0S2;
        }
        dst += DS0 * DS1 * DS2;
        src0 += S0S0 * S0S1 * S0S2;
        src1 += S0S0 * S0S1 * S0S2;
        condition += CS0 * CS1 * CS2;
    }
}

template <typename T, typename U, unsigned DS, unsigned CS, unsigned SS0>
TILEOP void DynWhere_TS(__ubuf__ T* dst, __ubuf__ uint8_t* temp, __ubuf__ U* condition, __ubuf__ T* src0, T src1,
                        unsigned T0, unsigned T1)
{
    if (T0 == 0 || T1 == 0) {
        return;
    }
    unsigned elementsPerCount = 1024;
    unsigned adressUsed = 4;
    unsigned bitsOfByte = 8;
    __ubuf__ half* castCondition = (__ubuf__ half*)temp;
    __ubuf__ half* compareCondition = (__ubuf__ half*)(castCondition + elementsPerCount);
    __ubuf__ int8_t* vcmpBitResult = (__ubuf__ int8_t*)(compareCondition + elementsPerCount);
    __ubuf__ uint64_t* startAddrUB = (__ubuf__ uint64_t*)(vcmpBitResult + elementsPerCount / bitsOfByte);
    __ubuf__ T* otherTempTensor = (__ubuf__ T*)(startAddrUB + adressUsed);

    unsigned numCountPerLine = T1 / elementsPerCount;
    unsigned elementsRemainPerLine = T1 % elementsPerCount;
    unsigned repeatNum = (elementsPerCount * sizeof(half) + REPEAT_BYTE - 1) / REPEAT_BYTE;
    unsigned repeatNumRemain = (elementsRemainPerLine * sizeof(half) + REPEAT_BYTE - 1) / REPEAT_BYTE;
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_mask_count();
    set_vector_mask(0x0, (uint64_t)elementsPerCount);
    pipe_barrier(PIPE_V);
    vector_dup(otherTempTensor, src1, 16, 1, 0, 8, 0);
    pipe_barrier(PIPE_V);
    if constexpr (std::is_same_v<U, bool>) {
        for (int i = 0; i < T0; i++) {
            for (int j = 0; j < numCountPerLine; j++) {
                ProcessWhere(dst + i * DS + j * elementsPerCount, src0 + i * SS0 + j * elementsPerCount,
                             otherTempTensor, condition + i * CS + j * elementsPerCount, castCondition,
                             (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNum, elementsPerCount);
            }
            if (elementsRemainPerLine) {
                ProcessWhere(dst + i * DS + elementsPerCount * numCountPerLine,
                             src0 + i * SS0 + elementsPerCount * numCountPerLine, otherTempTensor,
                             condition + i * CS + elementsPerCount * numCountPerLine, castCondition,
                             (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNumRemain, elementsRemainPerLine);
            }
        }
    } else {
        for (int i = 0; i < T0; i++) {
            for (int j = 0; j < numCountPerLine; j++) {
                ProcessWhere(dst + i * DS + j * elementsPerCount, src0 + i * SS0 + j * elementsPerCount,
                             otherTempTensor, condition + i * CS + j * elementsPerCount / 8, castCondition,
                             (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNum, elementsPerCount);
            }
            if (elementsRemainPerLine) {
                ProcessWhere(dst + i * DS + elementsPerCount * numCountPerLine,
                             src0 + i * SS0 + elementsPerCount * numCountPerLine, otherTempTensor,
                             condition + i * CS + elementsPerCount * numCountPerLine / 8, castCondition,
                             (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNumRemain, elementsRemainPerLine);
            }
        }
    }
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
}

// dim4
template <typename T, typename U, unsigned DS0, unsigned DS1, unsigned DS2, unsigned CS0, unsigned CS1, unsigned CS2,
          unsigned S0S0, unsigned S0S1, unsigned S0S2>
TILEOP void DynWhere_TS(__ubuf__ T* dst, __ubuf__ uint8_t* temp, __ubuf__ U* condition, __ubuf__ T* src0, T src1,
                        unsigned T0, unsigned T1, unsigned T2, unsigned T3)
{
    static_assert((DS2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((CS2 * sizeof(bool)) % BLOCK_SIZE == 0);
    static_assert((S0S2 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ T* dstLoop = dst;
        __ubuf__ U* conditionLoop = condition;
        __ubuf__ T* src0Loop = src0;
        for (int j = 0; j < T1; j++) {
            DynWhere_TS<T, U, DS2, CS2, S0S2>(dstLoop, temp, conditionLoop, src0Loop, src1, T2, T3);
            dstLoop += DS1 * DS2;
            conditionLoop += CS1 * CS2;
            src0Loop += S0S1 * S0S2;
        }
        dst += DS0 * DS1 * DS2;
        src0 += S0S0 * S0S1 * S0S2;
        condition += CS0 * CS1 * CS2;
    }
}

template <typename T, typename U, unsigned DS, unsigned CS, unsigned SS0>
TILEOP void DynWhere_ST(__ubuf__ T* dst, __ubuf__ uint8_t* temp, __ubuf__ U* condition, T src0, __ubuf__ T* src1,
                        unsigned T0, unsigned T1)
{
    if (T0 == 0 || T1 == 0) {
        return;
    }
    unsigned elementsPerCount = 1024;
    unsigned adressUsed = 4;
    unsigned bitsOfByte = 8;
    __ubuf__ half* castCondition = (__ubuf__ half*)temp;
    __ubuf__ half* compareCondition = (__ubuf__ half*)(castCondition + elementsPerCount);
    __ubuf__ int8_t* vcmpBitResult = (__ubuf__ int8_t*)(compareCondition + elementsPerCount);
    __ubuf__ uint64_t* startAddrUB = (__ubuf__ uint64_t*)(vcmpBitResult + elementsPerCount / bitsOfByte);
    __ubuf__ T* inputTempTensor = (__ubuf__ T*)(startAddrUB + adressUsed);
    unsigned numCountPerLine = T1 / elementsPerCount;
    unsigned elementsRemainPerLine = T1 % elementsPerCount;
    unsigned repeatNum = (elementsPerCount * sizeof(half) + REPEAT_BYTE - 1) / REPEAT_BYTE;
    unsigned repeatNumRemain = (elementsRemainPerLine * sizeof(half) + REPEAT_BYTE - 1) / REPEAT_BYTE;
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_mask_count();
    set_vector_mask(0x0, (uint64_t)elementsPerCount);
    pipe_barrier(PIPE_V);
    vector_dup(inputTempTensor, src0, 16, 1, 0, 8, 0);
    pipe_barrier(PIPE_V);
    if constexpr (std::is_same_v<U, bool>) {
        for (int i = 0; i < T0; i++) {
            for (int j = 0; j < numCountPerLine; j++) {
                ProcessWhere(dst + i * DS + j * elementsPerCount, inputTempTensor,
                             src1 + i * SS0 + j * elementsPerCount, condition + i * CS + j * elementsPerCount,
                             castCondition, (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNum, elementsPerCount);
            }
            if (elementsRemainPerLine) {
                ProcessWhere(dst + i * DS + elementsPerCount * numCountPerLine, inputTempTensor,
                             src1 + i * SS0 + elementsPerCount * numCountPerLine,
                             condition + i * CS + elementsPerCount * numCountPerLine, castCondition,
                             (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNumRemain, elementsRemainPerLine);
            }
        }
    } else {
        for (int i = 0; i < T0; i++) {
            for (int j = 0; j < numCountPerLine; j++) {
                ProcessWhere(dst + i * DS + j * elementsPerCount, inputTempTensor,
                             src1 + i * SS0 + j * elementsPerCount, condition + i * CS + j * elementsPerCount / 8,
                             castCondition, (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNum, elementsPerCount);
            }
            if (elementsRemainPerLine) {
                ProcessWhere(dst + i * DS + elementsPerCount * numCountPerLine, inputTempTensor,
                             src1 + i * SS0 + elementsPerCount * numCountPerLine,
                             condition + i * CS + elementsPerCount * numCountPerLine / 8, castCondition,
                             (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNumRemain, elementsRemainPerLine);
            }
        }
    }
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
}

// dim4
template <typename T, typename U, unsigned DS0, unsigned DS1, unsigned DS2, unsigned CS0, unsigned CS1, unsigned CS2,
          unsigned S0S0, unsigned S0S1, unsigned S0S2>
TILEOP void DynWhere_ST(__ubuf__ T* dst, __ubuf__ uint8_t* temp, __ubuf__ U* condition, T src0, __ubuf__ T* src1,
                        unsigned T0, unsigned T1, unsigned T2, unsigned T3)
{
    static_assert((DS2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((CS2 * sizeof(bool)) % BLOCK_SIZE == 0);
    static_assert((S0S2 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ T* dstLoop = dst;
        __ubuf__ U* conditionLoop = condition;
        __ubuf__ T* src1Loop = src1;
        for (int j = 0; j < T1; j++) {
            DynWhere_ST<T, U, DS2, CS2, S0S2>(dstLoop, temp, conditionLoop, src0, src1Loop, T2, T3);
            dstLoop += DS1 * DS2;
            conditionLoop += CS1 * CS2;
            src1Loop += S0S1 * S0S2;
        }
        dst += DS0 * DS1 * DS2;
        src1 += S0S0 * S0S1 * S0S2;
        condition += CS0 * CS1 * CS2;
    }
}

template <typename T, typename U, unsigned DS, unsigned CS, unsigned SS0>
TILEOP void DynWhere_SS(__ubuf__ T* dst, __ubuf__ uint8_t* temp, __ubuf__ U* condition, T src0, T src1, unsigned T0,
                        unsigned T1)
{
    if (T0 == 0 || T1 == 0) {
        return;
    }
    unsigned elementsPerCount = 1024;
    unsigned adressUsed = 4;
    unsigned bitsOfByte = 8;
    __ubuf__ half* castCondition = (__ubuf__ half*)temp;
    __ubuf__ half* compareCondition = (__ubuf__ half*)(castCondition + elementsPerCount);
    __ubuf__ int8_t* vcmpBitResult = (__ubuf__ int8_t*)(compareCondition + elementsPerCount);
    __ubuf__ uint64_t* startAddrUB = (__ubuf__ uint64_t*)(vcmpBitResult + elementsPerCount / bitsOfByte);
    __ubuf__ T* inputTempTensor = (__ubuf__ T*)(startAddrUB + adressUsed);
    __ubuf__ T* otherTempTensor = (__ubuf__ T*)(inputTempTensor + elementsPerCount);
    unsigned numCountPerLine = T1 / elementsPerCount;
    unsigned elementsRemainPerLine = T1 % elementsPerCount;
    unsigned repeatNum = (elementsPerCount * sizeof(half) + REPEAT_BYTE - 1) / REPEAT_BYTE;
    unsigned repeatNumRemain = (elementsRemainPerLine * sizeof(half) + REPEAT_BYTE - 1) / REPEAT_BYTE;
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_mask_count();
    set_vector_mask(0x0, (uint64_t)elementsPerCount);
    pipe_barrier(PIPE_V);
    vector_dup(inputTempTensor, src0, 16, 1, 0, 8, 0);
    vector_dup(otherTempTensor, src1, 16, 1, 0, 8, 0);
    pipe_barrier(PIPE_V);
    if constexpr (std::is_same_v<U, bool>) {
        for (int i = 0; i < T0; i++) {
            for (int j = 0; j < numCountPerLine; j++) {
                ProcessWhere(dst + i * DS + j * elementsPerCount, inputTempTensor, otherTempTensor,
                             condition + i * CS + j * elementsPerCount, castCondition, (__ubuf__ int8_t*)vcmpBitResult,
                             (__ubuf__ half*)compareCondition, (__ubuf__ uint64_t*)startAddrUB, repeatNum,
                             elementsPerCount);
            }
            if (elementsRemainPerLine) {
                ProcessWhere(dst + i * DS + elementsPerCount * numCountPerLine, inputTempTensor, otherTempTensor,
                             condition + i * CS + elementsPerCount * numCountPerLine, castCondition,
                             (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNumRemain, elementsRemainPerLine);
            }
        }
    } else {
        for (int i = 0; i < T0; i++) {
            for (int j = 0; j < numCountPerLine; j++) {
                ProcessWhere(dst + i * DS + j * elementsPerCount, inputTempTensor, otherTempTensor,
                             condition + i * CS + j * elementsPerCount / 8, castCondition,
                             (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNum, elementsPerCount);
            }
            if (elementsRemainPerLine) {
                ProcessWhere(dst + i * DS + elementsPerCount * numCountPerLine, inputTempTensor, otherTempTensor,
                             condition + i * CS + elementsPerCount * numCountPerLine / 8, castCondition,
                             (__ubuf__ int8_t*)vcmpBitResult, (__ubuf__ half*)compareCondition,
                             (__ubuf__ uint64_t*)startAddrUB, repeatNumRemain, elementsRemainPerLine);
            }
        }
    }
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
}

// dim4
template <typename T, typename U, unsigned DS0, unsigned DS1, unsigned DS2, unsigned CS0, unsigned CS1, unsigned CS2,
          unsigned S0S0, unsigned S0S1, unsigned S0S2>
TILEOP void DynWhere_SS(__ubuf__ T* dst, __ubuf__ uint8_t* temp, __ubuf__ U* condition, T src0, T src1, unsigned T0,
                        unsigned T1, unsigned T2, unsigned T3)
{
    static_assert((DS2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((CS2 * sizeof(bool)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ T* dstLoop = dst;
        __ubuf__ U* conditionLoop = condition;
        for (int j = 0; j < T1; j++) {
            DynWhere_SS<T, U, DS2, CS2, S0S2>(dstLoop, temp, conditionLoop, src0, src1, T2, T3);
            dstLoop += DS1 * DS2;
            conditionLoop += CS1 * CS2;
        }
        dst += DS0 * DS1 * DS2;
        condition += CS0 * CS1 * CS2;
    }
}

template <typename T, unsigned xShape0, unsigned xShape1>
TILEOP void DynTopKSort(__ubuf__ T* y, __ubuf__ T* tmp, __ubuf__ T* x, uint32_t idxStart)
{
    // x x 2 = y = tmp == xShape1 x 2
    GenSortIndex<T, T, xShape1>((__ubuf__ T*)y, tmp, idxStart);
    TopKSortWithIndex<T, xShape0, xShape1>(y, tmp, x);
}

template <typename T, unsigned xShape0, unsigned xShape1, int mergeSize>
TILEOP void DynTopKMerge(__ubuf__ T* y, __ubuf__ T* x)
{
    TileOp::TopKMerge<T, xShape0, xShape1, mergeSize>(y, x);
}

template <typename U, typename T, unsigned yShape0, unsigned yShape1, unsigned xShape0, unsigned xShape1, int isIndex,
          int k>
TILEOP void DynTopKExtract(__ubuf__ U* y, __ubuf__ T* x)
{
    TileOp::TopKExtract<U, T, yShape0, yShape1, xShape0, xShape1, isIndex, k>(y, x);
}

// onehot output dim2
template <typename T, unsigned NUM>
TILEOP void DynTonehot_(__ubuf__ int64_t* dst, __ubuf__ T* src, unsigned s0)
{
    constexpr unsigned countPerBlock = BLOCK_SIZE / sizeof(int64_t);
    unsigned blockCountPerLine = NUM / countPerBlock;
    unsigned blockCount = blockCountPerLine * s0;
    constexpr unsigned blockPerRepeat = 8;
    unsigned repeatCount = blockCount / blockPerRepeat;
    unsigned lastBlockCount = blockCount % blockPerRepeat;
    unsigned repeat = repeatCount / REPEAT_MAX;
    unsigned lastRepeatCount = repeatCount % REPEAT_MAX;
    // set all 0
    __ubuf__ int64_t* dst_ = dst;
    for (int i = 0; i < repeat; i++) {
        vector_dup((__ubuf__ int32_t*)dst_, 0, REPEAT_MAX, 1, 0, 8, 0);
        dst_ += REPEAT_MAX * countPerBlock * blockPerRepeat;
    }
    vector_dup((__ubuf__ int32_t*)dst_, 0, lastRepeatCount, 1, 0, 8, 0);
    dst_ += lastRepeatCount * countPerBlock * blockPerRepeat;
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < lastBlockCount * countPerBlock; i++) {
        *dst_ = 0;
        dst_++;
    }
    // set 1
    dst_ = dst;
    for (int i = 0; i < s0; i++) {
        T index = *(src + i);
        *(dst_ + index) = 1ll;
        dst_ += NUM;
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

// onehot output dim3
template <typename T, unsigned DS1, unsigned NUM>
TILEOP void DynTonehot_(__ubuf__ int64_t* dst, __ubuf__ T* src, unsigned s0, unsigned s1)
{
    constexpr unsigned align = BLOCK_SIZE / sizeof(T);
    for (int i = 0; i < s0; i++) {
        DynTonehot_<T, NUM>(dst, src, s1);
        dst += DS1 * NUM;
        src += (DS1 + align - 1) / align * align;
    }
}

// onehot output dim4
template <typename T, unsigned DS1, unsigned DS2, unsigned NUM>
TILEOP void DynTonehot_(__ubuf__ int64_t* dst, __ubuf__ T* src, unsigned s0, unsigned s1, unsigned s2)
{
    constexpr unsigned align = BLOCK_SIZE / sizeof(T);
    for (int i = 0; i < s0; i++) {
        DynTonehot_<T, DS2, NUM>(dst, src, s1, s2);
        dst += DS1 * DS2 * NUM;
        src += DS1 * ((DS2 + align - 1) / align * align);
    }
}
/**
 * 辅助函数，计算 token id 对应物理偏移
 */
template <typename T2, typename T3, unsigned blockSize>
INLINE T2 CalaOffset2PageAttention(__gm__ T3* blockTable, T2 index)
{
    T2 blockID = index / blockSize;            // 这个token对应第几个块，逻辑块
    blockID = blockTable[blockID];             // 页表中存放这个逻辑块到物理块的映射，得到物理块
    T2 blockOffset = index % blockSize;        // token在块内偏移
    index = blockID * blockSize + blockOffset; // 物理块*blockSize 得到物理块的偏移，加上token在块中的偏移
    return index;
}
/**
 * 定制版本，只支持 ds v3.2，使用之前请仔细确认
 * param 2维
 * indices 2维
 * axis -2
 * result 2维 {和标准实现不同}
 * [a,b] [1,c] -2  [c,b]
 *
 * 模板参数
 * T input 参数类型
 * T2 indices 参数类型
 * UBOutputS*  output在ub上的步长
 *
 * 运行时参数
 * dst result，ub上
 * param 输入，在gm上
 * indices 输入索引，在gm上
 * GMParamShape* ,param validshape，用于指导拷贝长度
 * GMParamStride*,param 的步长，用于计算偏移
 * GMParamOffset*,param 的偏移，用于确定分块
 * GMIndicesShape* ,indices 的 validshape ，用于指导循环，
 * GMIndicesStride* ,步长，用于计算偏移
 * blocktable[e,f] e batch的维度  f ceil(maxtoken/blockSize)
 */

template <typename T, typename T2, typename T3, unsigned UBOutputS1, unsigned UBOutputS2, unsigned blockSize>
TILEOP void GatherInUB(__ubuf__ T* dst, __gm__ T* param, __gm__ T2* indices, __gm__ T3* blockTable,
                       unsigned GMParamShape1, unsigned GMParamStride1, unsigned GMParamOffset0,
                       unsigned GMParamOffset1, unsigned GMIndicesShape1, unsigned GMIndicesStride1,
                       unsigned GMIndicesOffset0, unsigned GMIndicesOffset1, unsigned GMBlockTableStride1,
                       unsigned GMBlockTableOffset0, unsigned GMBlockTableOffset1)
{
    // 循环indices，标量流水拿出来索引
    param += GMParamOffset0 * GMParamStride1 + GMParamOffset1;
    indices += GMIndicesOffset0 * GMIndicesStride1 + GMIndicesOffset1;
    blockTable += GMBlockTableOffset0 * GMBlockTableStride1 + GMBlockTableOffset1;

    __gm__ T2* indices0 = indices;
    __gm__ T* param0 = param;
    __ubuf__ T* dst0 = dst;
    for (int j = 0; j < GMIndicesShape1; j++) {
        uint64_t index_1 = indices0[j];
        index_1 = CalaOffset2PageAttention<uint64_t, T3, blockSize>(blockTable, index_1);
        param0 = param + index_1 * GMParamStride1;
        UBCopyInBase<T, UBOutputS2>(dst0, param0, 1, GMParamShape1, GMParamStride1);
        dst0 += UBOutputS2;
    }
}

template <typename T>
TILEOP void DynTbrcb_(__ubuf__ T* dst, __ubuf__ T* src, unsigned T0, unsigned T1)
{
    constexpr unsigned brcPerRepeat = 8;
    if (T0 != 1) {
        unsigned repeatNum = (T0 + brcPerRepeat - 1) / brcPerRepeat;
        vbrcb((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src, 1, 8, repeatNum);
        return;
    }
    unsigned repeatNum = (T1 + brcPerRepeat - 1) / brcPerRepeat;
    vbrcb((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src, 1, 8, repeatNum);
}

// dim4
template <typename T, unsigned DS1, unsigned DS2, unsigned DS3, unsigned SS1, unsigned SS2, unsigned SS3>
TILEOP void DynTbrcb_(__ubuf__ T* dst, __ubuf__ T* src, unsigned T0, unsigned T1, unsigned T2, unsigned T3)
{
    static_assert(DS3 * sizeof(T) == BLOCK_SIZE);
    static_assert(DS2 % BLOCK_NUM_ONE_REPEAT == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < T1; j++) {
            DynTbrcb_<T>(dst_, src_, T2, T3);
            dst_ += DS2 * DS3;
            src_ += SS2 * SS3;
        }
        dst += DS1 * DS2 * DS3;
        src += SS1 * SS2 * SS3;
    }
}
/**
 * T param 类型
 * T2 indices 类型
 * axis 轴，归一化的轴
 * UBOutputS* result ub 步长
 * UBResultShape* ，result 中的 valid shape，指导循环
 * GMParamStride* ，param 中 每个维度的步长
 * GMParamOffset*， param 中 offset ，用于确认本次处理的块的地址
 */
template <typename T, typename T2, unsigned axis, unsigned UBOutputS1, unsigned UBOutputS2, unsigned UBOutputS3,
          unsigned UBOutputS4>
TILEOP void DynTgather(__ubuf__ T* dst, __gm__ T* param, __gm__ T2* indices, unsigned UBResultShape0,
                       unsigned UBResultShape1, unsigned UBResultShape2, unsigned UBResultShape3,
                       unsigned UBResultShape4, unsigned GMParamStride1, unsigned GMParamStride2,
                       unsigned GMParamStride3, unsigned GMParamOffset0, unsigned GMParamOffset1,
                       unsigned GMParamOffset2, unsigned GMParamOffset3, unsigned GMIndicesStride1,
                       unsigned GMIndicesOffset0, unsigned GMIndicesOffset1)
{
    param += GMParamOffset3 + GMParamOffset2 * GMParamStride3 + GMParamOffset1 * (GMParamStride2 * GMParamStride3) +
             GMParamOffset0 * (GMParamStride1 * GMParamStride2 * GMParamStride3);
    indices += GMIndicesOffset0 * GMIndicesStride1 + GMIndicesOffset1;
    if constexpr (axis == 0) {
        /**
         * [a,b,c,d]
         * [e,f]
         * [e,f,b,c,d]
         */
        __gm__ T2* indices0 = indices;
        for (int i = 0; i < UBResultShape0; ++i) { // e
            __gm__ T* param0 = param;
            __ubuf__ T* dst0 = dst;
            for (int j = 0; j < UBResultShape1; ++j) { // f
                __ubuf__ T* dst1 = dst0;
                uint64_t index = indices0[j];
                //[e,f,b,c,d]
                param0 = param + index * GMParamStride1 * GMParamStride2 * GMParamStride3;
                for (int k = 0; k < UBResultShape2; ++k) { // b
                    UBCopyInBase<T, UBOutputS4>(dst1, param0, UBResultShape3, UBResultShape4, GMParamStride3);
                    dst1 += UBOutputS3 * UBOutputS4;
                    param0 += GMParamStride2 * GMParamStride3;
                }
                dst0 += UBOutputS2 * UBOutputS3 * UBOutputS4;
            }
            dst += UBOutputS1 * UBOutputS2 * UBOutputS3 * UBOutputS4;
            indices0 += GMIndicesStride1;
        }
    } else if constexpr (axis == 1) {
        /**
         * [a,b,c,d]
         * [e,f]
         * [a,e,f,c,d]
         */
        for (int i = 0; i < UBResultShape0; ++i) { // a
            __gm__ T* param0 = param;
            __ubuf__ T* dst0 = dst;
            __gm__ T2* indices0 = indices;
            for (int j = 0; j < UBResultShape1; ++j) { // e
                __ubuf__ T* dst1 = dst0;
                for (int k = 0; k < UBResultShape2; ++k) { // f
                    uint64_t index = indices0[k];
                    param0 = param + index * GMParamStride2 * GMParamStride3;
                    UBCopyInBase<T, UBOutputS4>(dst1, param0, UBResultShape3, UBResultShape4, GMParamStride3);
                    dst1 += UBOutputS3 * UBOutputS4;
                }
                dst0 += UBOutputS2 * UBOutputS3 * UBOutputS4;
                indices0 += GMIndicesStride1;
            }
            dst += UBOutputS1 * UBOutputS2 * UBOutputS3 * UBOutputS4;
            param += GMParamStride1 * GMParamStride2 * GMParamStride3;
        }
    } else if constexpr (axis == 2) {
        /**
         * [a,b,c,d]
         * [e,f]
         * [a,b,e,f,d]
         */
        for (int i = 0; i < UBResultShape0; ++i) { // a
            __gm__ T* param0 = param;
            __ubuf__ T* dst0 = dst;
            for (int j = 0; j < UBResultShape1; ++j) { // b
                __gm__ T* param1 = param0;
                __ubuf__ T* dst1 = dst0;
                __gm__ T2* indices0 = indices;
                for (int k = 0; k < UBResultShape2; ++k) { // e
                    __ubuf__ T* dst2 = dst1;
                    for (int l = 0; l < UBResultShape3; l++) { // f
                        uint64_t index = indices0[l];
                        param1 = param0 + index * GMParamStride3;
                        UBCopyInBase<T, UBOutputS4>(dst2, param1, 1, UBResultShape4, GMParamStride3);
                        dst2 += UBOutputS4;
                    }
                    dst1 += UBOutputS3 * UBOutputS4;
                    indices0 += GMIndicesStride1;
                }
                param0 += GMParamStride2 * GMParamStride3;
                dst0 += UBOutputS2 * UBOutputS3 * UBOutputS4;
            }
            dst += UBOutputS1 * UBOutputS2 * UBOutputS3 * UBOutputS4;
            param += GMParamStride1 * GMParamStride2 * GMParamStride3;
        }
    } else if constexpr (axis == 3) {
        /**
         * [a,b,c,d]
         * [e,f]
         * [a,b,c,e,f]
         */
        /**
         * MOV_OUT_TO_UB_ALIGN 要求 UB 32 字节对齐，尾轴情况一次只搬运一个数据，无法满足 32 字节对齐
         * 只能使用标量流水
         *
         * 由于在 registerInfo 注册为 PIPE_MTE2 操作，框架会自动为 gatherinub 添加 PIPE_MTE2 同步指令，但是尾轴情况
         * gatherinub 是一个 S 操作。 方案： 根据同步指令的传递性，建立 PIPE_MTE2 到 PIPE_S 的屏障
         */
        for (int i = 0; i < UBResultShape0; ++i) { // a
            __gm__ T* param0 = param;
            __ubuf__ T* dst0 = dst;
            for (int j = 0; j < UBResultShape1; ++j) { // b
                __gm__ T* param1 = param0;
                __ubuf__ T* dst1 = dst0;
                for (int k = 0; k < UBResultShape2; ++k) { // c
                    __gm__ T* param2 = param1;
                    __ubuf__ T* dst2 = dst1;
                    __gm__ T2* indices0 = indices;
                    for (int l = 0; l < UBResultShape3; ++l) { // e
                        __ubuf__ T* dst3 = dst2;
                        __gm__ T2* indices1 = indices0;
                        for (int p = 0; p < UBResultShape4; ++p) { // f
                            uint64_t index = indices1[p];
                            param2 = param1 + index;
                            // UBCopyInBase<T, UBOutputS4>(dst3, param2, 1, 1, GMParamStride3);
                            *dst3 = *param2;
                            dst3++;
                        }
                        indices0 += GMIndicesStride1;
                        dst2 += UBOutputS4;
                    }
                    dst1 += UBOutputS3 * UBOutputS4;
                    param1 += GMParamStride3;
                }
                param0 += GMParamStride2 * GMParamStride3;
                dst0 += UBOutputS2 * UBOutputS3 * UBOutputS4;
            }
            dst += UBOutputS1 * UBOutputS2 * UBOutputS3 * UBOutputS4;
            param += GMParamStride1 * GMParamStride2 * GMParamStride3;
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    }
}
} // namespace TileOp

#endif // TILE_FWK_VECTOR_DYN_H
