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
 * \file vector.h
 * \brief
 */

#ifndef __LOGICALTENSOR_TILEOP_VECTOR__
#define __LOGICALTENSOR_TILEOP_VECTOR__

#include "tileop_common.h"
#include <type_traits>

namespace TileOp {
constexpr const uint64_t HALF_MASK = 64;
constexpr const uint64_t BLOCK_MAX_PER_REPEAT = 8;

// Binary op
// ADD
#define T_BIN Tadd_
#define T_BIN_PAIR Taddpair_
#define V_BIN_FUNC vadd
#define T_BIN_VS Tadds_
#define V_BIN_FUNC_VS vadds
#include "vector_bin.h"
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC
// SUB
#define T_BIN Tsub_
#define T_BIN_PAIR Tsubpair_
#define V_BIN_FUNC vsub
#define T_BIN_VS Tsubs_
#define V_BIN_FUNC_VS vadds
#define VS_SUB
#include "vector_bin.h"
#undef VS_SUB
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC

// MUL
#define T_BIN Tmul_
#define T_BIN_PAIR Tmulpair_
#define V_BIN_FUNC vmul
#define T_BIN_VS Tmuls_
#define V_BIN_FUNC_VS vmuls
#include "vector_bin.h"
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC

// MAX
#define T_BIN Tmax_
#define T_BIN_PAIR Tmaxpair_
#define V_BIN_FUNC vmax
#define T_BIN_VS Tmaxs_
#include "vector_bin.h"
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC
#undef T_BIN_VS

// MIN
#define T_BIN Tmin_
#define T_BIN_PAIR Tminpair_
#define V_BIN_FUNC vmin
#define T_BIN_VS Tmins_
#include "vector_bin.h"
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC
#undef T_BIN_VS

// DIV
#define T_BIN Tdiv_
#define T_BIN_PAIR Tdivpair_
#define V_BIN_FUNC vdiv
#define T_BIN_VS Tdivs_
#define V_BIN_FUNC_VS vmuls
#define VS_DIV
#include "vector_bin.h"
#undef VS_DIV
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef T_BIN_PAIR
#undef V_BIN_FUNC

// ADDWITHBRC
#define T_BIN_BRC Taddbrc_
#define V_BIN_BRC_FUNC vadd
#include "vector_bin_brc.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// SUBWITHBRC
#define T_BIN_BRC Tsubbrc_
#define V_BIN_BRC_FUNC vsub
#include "vector_bin_brc.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// MULWITHBRC
#define T_BIN_BRC Tmulbrc_
#define V_BIN_BRC_FUNC vmul
#include "vector_bin_brc.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// MAXWITHBRC
#define T_BIN_BRC Tmaxbrc_
#define V_BIN_BRC_FUNC vmax
#include "vector_bin_brc.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// MINWITHBRC
#define T_BIN_BRC Tminbrc_
#define V_BIN_BRC_FUNC vmin
#include "vector_bin_brc.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// DIVWITHBRC
#define T_BIN_BRC Tdivbrc_
#define V_BIN_BRC_FUNC vdiv
#include "vector_bin_brc.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// Unary op
// EXP
#define T_UNA Texp_
#define V_UNA_FUNC vexp
#include "vector_una.h"
#undef T_UNA
#undef V_UNA_FUNC
// RECIPROCAL
#define T_UNA Trec_
#define V_UNA_FUNC vrec
#include "vector_una.h"
#undef T_UNA
#undef V_UNA_FUNC
// SQRT
#define T_UNA Tsqrt_
#define V_UNA_FUNC vsqrt
#include "vector_una.h"
#undef T_UNA
#undef V_UNA_FUNC
// ABS
#define T_UNA Tabs_
#define V_UNA_FUNC vabs
#include "vector_una.h"
#undef T_UNA
#undef V_UNA_FUNC
// LN
#define T_UNA Tln_
#define V_UNA_FUNC vln
#include "vector_una.h"
#undef T_UNA
#undef V_UNA_FUNC

template <typename T, unsigned T0, unsigned T1, unsigned DS, unsigned SS, unsigned TS>
TILEOP void Ttranspose_vnchwconv_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    if constexpr (((DS % 16) != 0) || ((SS % 8) != 0) || ((TS % 16) != 0)) {
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
    static_assert(sizeof(T) == 4);
    // 16 x 32B subtile
    constexpr int block_elem = BLOCK_SIZE / sizeof(T);
    // go by subtile column, a.k.a. iter in row direction
    constexpr int num_subtile_x = (T1 + block_elem - 1) / block_elem;
    constexpr int num_subtile_y = T0 / 16;
    if constexpr (num_subtile_y) {
        for (int i = 0; i < num_subtile_x; i++) {
            uint64_t srcUb[16] = {0}, tmpUb[16] = {0};
            for (int j = 0; j < 16; j++) {
                srcUb[j] = (uint64_t)(src + i * 8 + j * SS);
                tmpUb[j] = (uint64_t)(tmp + ((j >> 1) + i * 8) * TS + (j & 1) * block_elem);
            }
            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[8]);
            set_va_reg_sb(VA0, tmpUb);
            set_va_reg_sb(VA1, &tmpUb[8]);
            if constexpr (num_subtile_y == 1) {
                scatter_vnchwconv_b32(VA0, VA2, 1, 0, 0);
            } else {
                scatter_vnchwconv_b32(VA0, VA2, num_subtile_y, 2, 16 * SS * sizeof(T) / BLOCK_SIZE);
            }
        }
    }
    // tail
    constexpr int remain_y = T0 % 16;
    if constexpr (remain_y) {
        uint64_t srcUb[16] = {0}, tmpUb[16] = {0};
        for (int i = 0; i < remain_y; i++) {
            srcUb[i] = (uint64_t)(src + (num_subtile_y * 16 + i) * SS);
        }
        for (int i = 0; i < 16; i++) {
            tmpUb[i] = (uint64_t)(tmp + num_subtile_y * 16 + (i & 1) * block_elem + (i >> 1) * TS);
        }
        set_va_reg_sb(VA2, srcUb);
        set_va_reg_sb(VA3, &srcUb[8]);
        set_va_reg_sb(VA0, tmpUb);
        set_va_reg_sb(VA1, &tmpUb[8]);
        if constexpr (num_subtile_x == 1) {
            scatter_vnchwconv_b32(VA0, VA2, 1, 0, 0);
        } else {
            scatter_vnchwconv_b32(VA0, VA2, num_subtile_x, 8 * TS * sizeof(T) / BLOCK_SIZE, 1);
        }
    }
    // copy to dst
    pipe_barrier(PIPE_V);
    uint16_t lenBurst = (T0 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint16_t srcGap = TS * sizeof(T) / BLOCK_SIZE - lenBurst;
    uint16_t dstGap = DS * sizeof(T) / BLOCK_SIZE - lenBurst;
    copy_ubuf_to_ubuf(dst, tmp, 0, T1, lenBurst, srcGap, dstGap);
}

template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned T4, unsigned DS1, unsigned DS2,
          unsigned DS3, unsigned DS4, unsigned SS1, unsigned SS2, unsigned SS3, unsigned SS4>
TILEOP void Ttranspose_vnchwconv_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    constexpr unsigned TS1 = DS1;
    constexpr unsigned TS2 = DS2;
    constexpr unsigned TS3 = (T4 + 7) / 8 * 8;
    constexpr unsigned TS4 = (T3 + 15) / 16 * 16;
    for (unsigned i = 0; i < T0; i++) {
        __ubuf__ T* dst0 = dst;
        __ubuf__ T* src0 = src;
        __ubuf__ T* tmp0 = tmp;
        for (unsigned j = 0; j < T1; j++) {
            __ubuf__ T* dst1 = dst0;
            __ubuf__ T* src1 = src0;
            __ubuf__ T* tmp1 = tmp0;
            for (unsigned k = 0; k < T2; k++) {
                Ttranspose_vnchwconv_<T, T3, T4, DS4, SS4, TS4>(dst1, src1, tmp1);
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

// UB TILE Binary Op
// Now assume src and dst shape are equal and 2d tile

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstRawShape0, unsigned dstRawShape1,
          unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void TLn(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 8;

    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    // UB discontinuous
    if ((dstRawShape1 > TShape1 || src0RawShape1 > TShape1) &&
        (TShape1 % blockSize == 0 && dstRawShape1 % blockSize == 0 && src0RawShape1 % blockSize == 0)) {
        srcRepeatStride = src0RawShape1 > TShape1 ? src0RawShape1 / blockSize : TShape1 / blockSize;
        dstRepeatStride = dstRawShape1 > TShape1 ? dstRawShape1 / blockSize : TShape1 / blockSize;

        set_vector_mask(
            static_cast<uint64_t>(
                (TShape1 > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1 - 64)) - 1) : 0),
            static_cast<uint64_t>((TShape1 >= 64) ?
                                      0xffffffffffffffff :
                                      (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1)) - 1)));
        vln(dst, src, TShape0, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_vector_mask(-1, -1);
        return;
    }

    // UB continuous
    if constexpr (repeat < 1) {
        constexpr uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems);
        vln(dst, src, 1, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint8_t repeatLoop = static_cast<uint8_t>(repeat / REPEAT_MAX);
    uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
    for (uint8_t i = 0; i < repeatLoop; i++) {
        vln(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), REPEAT_MAX,
            dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    if (repeatMod != 0) {
        vln(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
            src + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), repeatMod, dstBlockStride, srcBlockStride,
            dstRepeatStride, srcRepeatStride);
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void TLn(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    constexpr uint32_t baseTileSize = TShape2 * TShape3;
    constexpr uint32_t dstRawSize = dstRawShape0 * dstRawShape1;
    constexpr uint32_t src0RawSize = src0RawShape0 * src0RawShape1;
    constexpr uint32_t alignDst = dstRawSize > baseTileSize ? dstRawSize : baseTileSize;
    constexpr uint32_t alignSrc0 = src0RawSize > baseTileSize ? src0RawSize : baseTileSize;
    // ub需要32B对齐
    static_assert(baseTileSize % blockSize == 0);
    static_assert(alignDst % blockSize == 0);
    static_assert(alignSrc0 % blockSize == 0);

    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::TLn<T, TShape2, TShape3, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape0, src0RawShape1>(
                dst, src);
            dst += alignDst;
            src += alignSrc0;
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstRawShape0, unsigned dstRawShape1,
          unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Trsqrt(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 8;

    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    // UB discontinuous
    if ((dstRawShape1 > TShape1 || src0RawShape1 > TShape1) &&
        (TShape1 % blockSize == 0 && dstRawShape1 % blockSize == 0 && src0RawShape1 % blockSize == 0)) {
        srcRepeatStride = src0RawShape1 > TShape1 ? src0RawShape1 / blockSize : TShape1 / blockSize;
        dstRepeatStride = dstRawShape1 > TShape1 ? dstRawShape1 / blockSize : TShape1 / blockSize;

        set_vector_mask(
            static_cast<uint64_t>(
                (TShape1 > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1 - 64)) - 1) : 0),
            static_cast<uint64_t>((TShape1 >= 64) ?
                                      0xffffffffffffffff :
                                      (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1)) - 1)));
        vrsqrt(dst, src, TShape0, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_vector_mask(-1, -1);
        return;
    }

    // UB continuous
    if constexpr (repeat < 1) {
        constexpr uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems);
        vrsqrt(dst, src, 1, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint8_t repeatLoop = static_cast<uint8_t>(repeat / REPEAT_MAX);
    uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
    for (uint8_t i = 0; i < repeatLoop; i++) {
        vrsqrt(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
               REPEAT_MAX, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    if (repeatMod != 0) {
        vrsqrt(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
               src + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), repeatMod, dstBlockStride, srcBlockStride,
               dstRepeatStride, srcRepeatStride);
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Trsqrt(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    constexpr uint32_t baseTileSize = TShape2 * TShape3;
    constexpr uint32_t dstRawSize = dstRawShape0 * dstRawShape1;
    constexpr uint32_t src0RawSize = src0RawShape0 * src0RawShape1;
    constexpr uint32_t alignDst = dstRawSize > baseTileSize ? dstRawSize : baseTileSize;
    constexpr uint32_t alignSrc0 = src0RawSize > baseTileSize ? src0RawSize : baseTileSize;
    // ub需要32B对齐
    static_assert(baseTileSize % blockSize == 0);
    static_assert(alignDst % blockSize == 0);
    static_assert(alignSrc0 % blockSize == 0);

    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::Trsqrt<T, TShape2, TShape3, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src);
            dst += alignDst;
            src += alignSrc0;
        }
    }
}

template <typename T, unsigned T0, unsigned T1, unsigned Ds>
TILEOP void Tduplicate_(__ubuf__ T* dst, T value)
{
    constexpr unsigned npr = REPEAT_BYTE / sizeof(T);
    constexpr unsigned numRepeatPerLine = T1 / npr;
    constexpr unsigned numRemainPerLine = T1 % npr;
    constexpr unsigned blockSizeElem = BLOCK_SIZE / sizeof(T);
    if constexpr (numRepeatPerLine > 0) {
        for (unsigned i = 0; i < T0; i++) {
            vector_dup(dst + i * Ds, value, numRepeatPerLine, 1, 1, BLOCK_MAX_PER_REPEAT, (int64_t)0);
        }
    }

    // shift to deal with tail
    dst += numRepeatPerLine * npr;
    if constexpr (numRemainPerLine) {
        constexpr unsigned numLoop = T0 / REPEAT_MAX;
        constexpr unsigned remainAfterLoop = T0 % REPEAT_MAX;
        if constexpr (numRemainPerLine >= HALF_MASK) {
            set_vector_mask((((static_cast<uint64_t>(1)) << static_cast<uint32_t>(numRemainPerLine - HALF_MASK)) - 1UL),
                            0xffffffffffffffffUL);
        } else {
            set_vector_mask(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(numRemainPerLine)) - 1UL));
        }
        if constexpr (numLoop) {
            for (unsigned i = 0; i < numLoop; i++) {
                vector_dup(dst + i * REPEAT_MAX * Ds, value, REPEAT_MAX, 1, 1, Ds / blockSizeElem, (int64_t)0);
            }
        }
        if constexpr (remainAfterLoop) {
            vector_dup(dst + numLoop * REPEAT_MAX * Ds, value, remainAfterLoop, 1, 1, Ds / blockSizeElem, (int64_t)0);
        }
        set_vector_mask(-1, -1);
    }
}

// dim4
template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned Ds0, unsigned Ds1, unsigned Ds2>
TILEOP void Tduplicate_(__ubuf__ T* dst, T value)
{
    static_assert((Ds2 * sizeof(T)) % BLOCK_SIZE == 0);
    for (unsigned i = 0; i < T0; i++) {
        __ubuf__ T* dst_ = dst;
        for (unsigned j = 0; j < T1; j++) {
            Tduplicate_<T, T2, T3, Ds2>(dst_, value);
            dst_ += Ds1 * Ds2;
        }
        dst += Ds0 * Ds1 * Ds2;
    }
}

// row sum and expand
// TShape0 <= 255, TShape1 * sizeof(T) <= 256B and 32B aligned
template <typename T, unsigned TShape0, unsigned TShape1, unsigned oriShape0, unsigned oriShape1>
TILEOP void Trowsumexpand(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint64_t shape1Repeat = static_cast<uint64_t>(TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    constexpr uint8_t dstRepeatStride = TShape1;
    constexpr uint8_t srcrepeatStride = TShape1 * sizeof(T) / 32;

    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    if constexpr (oriShape1 % blockSize != 0) {              // 尾轴32B非对齐场景 一般是不存在Tadd场景
        vector_dup(dst, 0.0f, TShape0, 1, 1, 8, (int64_t)0); // dst dup m*64
        pipe_barrier(PIPE_V);
        constexpr uint32_t splitFactor = REPEAT_BYTE / sizeof(T); // 64 or 128
        constexpr uint32_t repeatLoop = static_cast<uint32_t>(oriShape1 / splitFactor);
        constexpr uint32_t repeatMod = static_cast<uint32_t>(oriShape1 % splitFactor);

        constexpr uint32_t alignedTShape1 = (oriShape1 + blockSize - 1) / blockSize * blockSize;
        constexpr uint8_t srcRepeatStride = alignedTShape1 / blockSize;

        for (uint32_t i = 0; i < repeatLoop; i++) {
            vadd(dst, src + i * splitFactor, dst, TShape0, 1, 1, 1, 8, srcRepeatStride, 8);
            pipe_barrier(PIPE_V);
        }
        if (repeatMod != 0) {
            set_vector_mask(
                static_cast<uint64_t>(
                    (repeatMod > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(repeatMod - 64)) - 1) : 0),
                static_cast<uint64_t>((repeatMod > 64) ?
                                          0xffffffffffffffff :
                                          (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(repeatMod)) - 1)));

            vadd(dst, src + repeatLoop * splitFactor, dst, TShape0, 1, 1, 1, 8, srcRepeatStride, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(-1, -1);
        }

        vcadd(dst, dst, (uint16_t)TShape0, (uint16_t)splitFactor, (uint16_t)1ULL, (uint16_t)8ULL, false);

        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);

        set_mask_count();
        set_vector_mask(0, TShape1);
        for (int i = TShape0 - 1; i >= 0; i--) {
            vector_dup(dst + i * TShape1, (T)(*(dst + i * 64)), 1, 1, 1, 8, 0);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    if constexpr (shape1Repeat < 1) {
        set_mask_count();
        set_vector_mask(0, TShape1);
        for (int i = 0; i < TShape0; i++) {
            vcadd(dst + i * TShape1, src + i * TShape1, 1, dstRepeatStride, srcBlockStride, srcrepeatStride, false);
        }
        for (int i = 0; i < TShape0; i++) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            T tmp = (T)(*(dst + i * TShape1));
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
            vector_dup(dst + i * TShape1, tmp, 1, 1, 1, srcrepeatStride, 0);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    } else if constexpr (TShape0 % 8 == 0 && TShape1 % 64 == 0) {
        constexpr uint8_t repeatLoop = static_cast<uint8_t>(shape1Repeat / REPEAT_MAX);
        constexpr uint8_t repeatMod = static_cast<uint8_t>(shape1Repeat % REPEAT_MAX);
        vector_dup(dst, 0.0f, TShape0, 1, 1, 8, (int64_t)0); // dst n,64
        pipe_barrier(PIPE_V);
        for (int i = 0; i < TShape0; i++) {
            // vadd + vcadd
            constexpr uint8_t repeatForReduce = static_cast<uint8_t>(TShape1 / 64);
            vadd(dst + i * 64, src + i * TShape1, dst + i * 64, repeatForReduce, 1, 1, 1, (int64_t)0, (int64_t)8,
                 (int64_t)0);
        }

        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(-1, -1);
        // 将reduce add 结果输出到最后部分缓存
        vcadd(dst + TShape1 * TShape0 - TShape0, dst, TShape0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL, 0);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t*)(dst), (__ubuf__ uint32_t*)(dst + TShape1 * TShape0 - TShape0), TShape1 / 8, TShape1,
              TShape0 / 8);
        pipe_barrier(PIPE_V);
        for (int i = 0; i < TShape0; i++) {
            // 首先将最后一个block扩展为8个block
            vcopy((__ubuf__ uint32_t*)(dst + i * TShape1), (__ubuf__ uint32_t*)(dst + i * TShape1), TShape1 / 64, 1, 0,
                  8, 0);
        }
        pipe_barrier(PIPE_V);
    } else {
        constexpr uint8_t repeatLoop = static_cast<uint8_t>(shape1Repeat / REPEAT_MAX);
        constexpr uint8_t repeatMod = static_cast<uint8_t>(shape1Repeat % REPEAT_MAX);
        vector_dup(dst, 0.0f, TShape0, 1, 1, 8, (int64_t)0); // dst n,64
        pipe_barrier(PIPE_V);
        for (int i = 0; i < TShape0; i++) {
            // vadd + vcadd
            constexpr uint8_t repeatForReduce = static_cast<uint8_t>(TShape1 / 64);
            vadd(dst + i * 64, src + i * TShape1, dst + i * 64, repeatForReduce, 1, 1, 1, (int64_t)0, (int64_t)8,
                 (int64_t)0);
            pipe_barrier(PIPE_V);
            vcadd(dst + i * 64, dst + i * 64, (uint16_t)1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL, 0);
            pipe_barrier(PIPE_V);
        }

        for (int i = TShape0 - 1; i >= 0; i--) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            for (uint8_t j = 0; j < repeatLoop; j++) {
                vector_dup(dst + i * TShape1 + j * REPEAT_MAX * REPEAT_BYTE / sizeof(T), (T)(*(dst + i * 64)),
                           REPEAT_MAX, 1, 1, 8, 0);
            }
            if (repeatMod != 0) {
                vector_dup(dst + i * TShape1 + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), (T)(*(dst + i * 64)),
                           repeatMod, 1, 1, 8, 0);
            }
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned oriShape0,
          unsigned oriShape1>
TILEOP void Trowsumexpand(__ubuf__ T* dst, __ubuf__ T* src)
{
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::Trowsumexpand<T, TShape2, TShape3, oriShape0, oriShape1>(dst, src);
            dst += TShape2 * TShape3;
            src += TShape2 * TShape3;
        }
    }
}

// row max and expand
// TShape0 <= 255, TShape1 * sizeof(T) <= 256B and 32B aligned
template <typename T, unsigned TShape0, unsigned TShape1, unsigned oriShape0, unsigned oriShape1>
TILEOP void Trowmaxexpand(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint64_t shape1Repeat = static_cast<uint64_t>(TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    constexpr uint8_t dstRepeatStride = TShape1;
    constexpr uint8_t srcrepeatStride = TShape1 * sizeof(T) / 32;
    union {
        float f;
        unsigned int i;
    } float_neg_inf_num = {.i = 0xFF800000};
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    if constexpr (oriShape1 % blockSize != 0) { // 尾轴32B非对齐场景 一般是不存在Tmax场景
        vector_dup(dst, float_neg_inf_num.f, TShape0, 1, 1, 8, (int64_t)0); // dst dup m*64
        pipe_barrier(PIPE_V);
        constexpr uint32_t splitFactor = REPEAT_BYTE / sizeof(T); // 64 or 128
        constexpr uint32_t repeatLoop = static_cast<uint32_t>(oriShape1 / splitFactor);
        constexpr uint32_t repeatMod = static_cast<uint32_t>(oriShape1 % splitFactor);

        constexpr uint32_t alignedTShape1 = (oriShape1 + blockSize - 1) / blockSize * blockSize;
        constexpr uint8_t srcRepeatStride = alignedTShape1 / blockSize;

        for (uint32_t i = 0; i < repeatLoop; i++) {
            vmax(dst, src + i * splitFactor, dst, TShape0, 1, 1, 1, 8, srcRepeatStride, 8);
            pipe_barrier(PIPE_V);
        }
        if (repeatMod != 0) {
            set_vector_mask(
                static_cast<uint64_t>(
                    (repeatMod > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(repeatMod - 64)) - 1) : 0),
                static_cast<uint64_t>((repeatMod > 64) ?
                                          0xffffffffffffffff :
                                          (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(repeatMod)) - 1)));

            vmax(dst, src + repeatLoop * splitFactor, dst, TShape0, 1, 1, 1, 8, srcRepeatStride, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(-1, -1);
        }

        vcmax(dst, dst, (uint16_t)TShape0, (uint16_t)splitFactor, (uint16_t)1ULL, (uint16_t)8ULL, ONLY_VALUE);

        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);

        set_mask_count();
        set_vector_mask(0, TShape1);
        for (int i = TShape0 - 1; i >= 0; i--) {
            vector_dup(dst + i * TShape1, (T)(*(dst + i * 64)), 1, 1, 1, 8, 0);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    }

    if constexpr (shape1Repeat < 1) {
        set_mask_count();
        set_vector_mask(0, TShape1);
        for (int i = 0; i < TShape0; i++) {
            vcmax(dst + i * TShape1, src + i * TShape1, 1, dstRepeatStride, srcBlockStride, srcrepeatStride,
                  ONLY_VALUE);
        }
        for (int i = 0; i < TShape0; i++) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            T tmp = (T)(*(dst + i * TShape1));
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
            vector_dup(dst + i * TShape1, tmp, 1, 1, 1, srcrepeatStride, 0);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    } else if constexpr (TShape0 % 8 == 0 && TShape1 % 64 == 0) {
        // init dst value, using float_neg_inf_num.f
        constexpr uint8_t repeatLoop = static_cast<uint8_t>(shape1Repeat / REPEAT_MAX);
        constexpr uint8_t repeatMod = static_cast<uint8_t>(shape1Repeat % REPEAT_MAX);
        vector_dup(dst, float_neg_inf_num.f, TShape0, 1, 1, 8, (int64_t)0); // dst n,64
        pipe_barrier(PIPE_V);
        // vmax + vcmax
        for (int i = 0; i < TShape0; i++) {
            constexpr uint8_t repeatForReduce = static_cast<uint8_t>(TShape1 / 64);
            vmax(dst + i * 64, src + i * TShape1, dst + i * 64, repeatForReduce, 1, 1, 1, (int64_t)0, (int64_t)8,
                 (int64_t)0);
        }

        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(-1, -1);
        // 将reduce max结果输出到最后部分缓存
        vcmax(dst + TShape1 * TShape0 - TShape0, dst, TShape0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL,
              ONLY_VALUE);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t*)(dst), (__ubuf__ uint32_t*)(dst + TShape1 * TShape0 - TShape0), TShape1 / 8, TShape1,
              TShape0 / 8);
        pipe_barrier(PIPE_V);
        for (int i = 0; i < TShape0; i++) {
            // 首先将最后一个block扩展为8个block
            vcopy((__ubuf__ uint32_t*)(dst + i * TShape1), (__ubuf__ uint32_t*)(dst + i * TShape1), TShape1 / 64, 1, 0,
                  8, 0);
        }
        pipe_barrier(PIPE_V);
    } else {
        // init dst value, using float_neg_inf_num.f
        union {
            float f;
            unsigned int i;
        } float_neg_inf_num = {.i = 0xFF800000};
        constexpr uint8_t repeatLoop = static_cast<uint8_t>(shape1Repeat / REPEAT_MAX);
        constexpr uint8_t repeatMod = static_cast<uint8_t>(shape1Repeat % REPEAT_MAX);
        vector_dup(dst, float_neg_inf_num.f, TShape0, 1, 1, 8, (int64_t)0); // dst n,64
        pipe_barrier(PIPE_V);
        // vmax + vcmax
        for (int i = 0; i < TShape0; i++) {
            constexpr uint8_t repeatForReduce = static_cast<uint8_t>(TShape1 / 64);
            vmax(dst + i * 64, src + i * TShape1, dst + i * 64, repeatForReduce, 1, 1, 1, (int64_t)0, (int64_t)8,
                 (int64_t)0);
            pipe_barrier(PIPE_V);
            vcmax(dst + i * 64, dst + i * 64, (uint16_t)1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL, ONLY_VALUE);
            pipe_barrier(PIPE_V);
        }

        for (int i = TShape0 - 1; i >= 0; i--) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            for (uint8_t j = 0; j < repeatLoop; j++) {
                vector_dup(dst + i * TShape1 + j * REPEAT_MAX * REPEAT_BYTE / sizeof(T), (T)(*(dst + i * 64)),
                           REPEAT_MAX, 1, 1, 8, 0);
            }
            if (repeatMod != 0) {
                vector_dup(dst + i * TShape1 + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), (T)(*(dst + i * 64)),
                           repeatMod, 1, 1, 8, 0);
            }
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned oriShape0,
          unsigned oriShape1>
TILEOP void Trowmaxexpand(__ubuf__ T* dst, __ubuf__ T* src)
{
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::Trowmaxexpand<T, TShape2, TShape3, oriShape0, oriShape1>(dst, src);
            dst += TShape2 * TShape3;
            src += TShape2 * TShape3;
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstRawShape0, unsigned dstRawShape1,
          unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tadds(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 8;
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    // ub discontinuous
    if constexpr ((dstRawShape1 > TShape1 || src0RawShape1 > TShape1) &&
                  (TShape1 % blockSize == 0 && src0RawShape1 % blockSize == 0)) {
        srcRepeatStride = src0RawShape1 > TShape1 ? src0RawShape1 / blockSize : TShape1 / blockSize;
        dstRepeatStride = dstRawShape1 > TShape1 ? dstRawShape1 / blockSize : TShape1 / blockSize;

        set_vector_mask(
            static_cast<uint64_t>(
                (TShape1 > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1 - 64)) - 1) : 0),
            static_cast<uint64_t>((TShape1 >= 64) ?
                                      0xffffffffffffffff :
                                      (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1)) - 1)));
        vadds(dst, src0, src1, TShape0, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);

        set_vector_mask(-1, -1);
        return;
    }

    // ub continuous
    if constexpr (repeat < 1) {
        constexpr uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems);
        vadds(dst, src0, src1, 1, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint8_t repeatLoop = static_cast<uint8_t>(repeat / REPEAT_MAX);
    uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
    for (uint8_t i = 0; i < repeatLoop; i++) {
        vadds(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src0 + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src1,
              REPEAT_MAX, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    if (repeatMod != 0) {
        vadds(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
              src0 + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src1, repeatMod, dstBlockStride, srcBlockStride,
              dstRepeatStride, srcRepeatStride);
    }
}

// dim1
template <typename T, unsigned TShape0, unsigned dstRawShape0, unsigned src0RawShape0>
TILEOP void Tadds(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    TileOp::Tadds<T, 1, TShape0, 1, dstRawShape0, 1, src0RawShape0>(dst, src0, src1);
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tadds(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    static_assert((TShape2 * TShape3) % blockSize == 0);

    int alignSrc0 = src0RawShape0 * src0RawShape1 > TShape2 * TShape3 ? src0RawShape0 * src0RawShape1 :
                                                                        TShape2 * TShape3;
    int alignDst = dstRawShape0 * dstRawShape1 > TShape2 * TShape3 ? dstRawShape0 * dstRawShape1 : TShape2 * TShape3;
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::Tadds<T, TShape2, TShape3, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src0,
                                                                                                         src1);
            dst += alignDst;
            src0 += alignSrc0;
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tadds(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    int alignSrc0 = src0RawShape0 * src0RawShape1 > TShape1 * TShape2 ? src0RawShape0 * src0RawShape1 :
                                                                        TShape1 * TShape2;
    int alignDst = dstRawShape0 * dstRawShape1 > TShape1 * TShape2 ? dstRawShape0 * dstRawShape1 : TShape1 * TShape2;

    for (int i = 0; i < TShape0; ++i) {
        TileOp::Tadds<T, TShape1, TShape2, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src0, src1);
        dst += alignDst;
        src0 += alignSrc0;
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstRawShape0, unsigned dstRawShape1,
          unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tsubs(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    src1 = src1 * (-1);
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 8;
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    // ub discontinuous
    if constexpr ((dstRawShape1 > TShape1 || src0RawShape1 > TShape1) &&
                  (TShape1 % blockSize == 0 && src0RawShape1 % blockSize == 0)) {
        srcRepeatStride = src0RawShape1 > TShape1 ? src0RawShape1 / blockSize : TShape1 / blockSize;
        dstRepeatStride = dstRawShape1 > TShape1 ? dstRawShape1 / blockSize : TShape1 / blockSize;

        set_vector_mask(
            static_cast<uint64_t>(
                (TShape1 > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1 - 64)) - 1) : 0),
            static_cast<uint64_t>((TShape1 >= 64) ?
                                      0xffffffffffffffff :
                                      (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1)) - 1)));
        vadds(dst, src0, src1, TShape0, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);

        set_vector_mask(-1, -1);
        return;
    }

    // ub continuous
    if constexpr (repeat < 1) {
        constexpr uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems);
        vadds(dst, src0, src1, 1, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint8_t repeatLoop = static_cast<uint8_t>(repeat / REPEAT_MAX);
    uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
    for (uint8_t i = 0; i < repeatLoop; i++) {
        vadds(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src0 + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src1,
              REPEAT_MAX, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    if (repeatMod != 0) {
        vadds(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
              src0 + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src1, repeatMod, dstBlockStride, srcBlockStride,
              dstRepeatStride, srcRepeatStride);
    }
}

// dim1
template <typename T, unsigned TShape0, unsigned dstRawShape0, unsigned src0RawShape0>
TILEOP void Tsubs(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    TileOp::Tsubs<T, 1, TShape0, 1, dstRawShape0, 1, src0RawShape0>(dst, src0, src1);
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tsubs(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    static_assert((TShape2 * TShape3) % blockSize == 0);

    int alignSrc0 = src0RawShape0 * src0RawShape1 > TShape2 * TShape3 ? src0RawShape0 * src0RawShape1 :
                                                                        TShape2 * TShape3;
    int alignDst = dstRawShape0 * dstRawShape1 > TShape2 * TShape3 ? dstRawShape0 * dstRawShape1 : TShape2 * TShape3;
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::Tsubs<T, TShape2, TShape3, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src0,
                                                                                                         src1);
            dst += alignDst;
            src0 += alignSrc0;
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tsubs(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    static_assert((TShape1 * TShape2) % blockSize == 0);

    int alignSrc0 = src0RawShape0 * src0RawShape1 > TShape1 * TShape2 ? src0RawShape0 * src0RawShape1 :
                                                                        TShape1 * TShape2;
    int alignDst = dstRawShape0 * dstRawShape1 > TShape1 * TShape2 ? dstRawShape0 * dstRawShape1 : TShape1 * TShape2;
    for (int i = 0; i < TShape0; ++i) {
        TileOp::Tsubs<T, TShape1, TShape2, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src0, src1);
        dst += alignDst;
        src0 += alignSrc0;
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstRawShape0, unsigned dstRawShape1,
          unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tmuls(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 8;
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    if ((dstRawShape1 > TShape1 || src0RawShape1 > TShape1) &&
        (TShape1 % blockSize == 0 && dstRawShape1 % blockSize == 0 && src0RawShape1 % blockSize == 0)) {
        srcRepeatStride = src0RawShape1 > TShape1 ? src0RawShape1 / blockSize : TShape1 / blockSize;
        dstRepeatStride = dstRawShape1 > TShape1 ? dstRawShape1 / blockSize : TShape1 / blockSize;

        set_vector_mask(
            static_cast<uint64_t>(
                (TShape1 > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1 - 64)) - 1) : 0),
            static_cast<uint64_t>((TShape1 >= 64) ?
                                      0xffffffffffffffff :
                                      (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1)) - 1)));
        vmuls(dst, src0, src1, TShape0, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);

        set_vector_mask(-1, -1);
        return;
    }

    if constexpr (repeat < 1) {
        constexpr uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems);
        vmuls(dst, src0, src1, 1, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint8_t repeatLoop = static_cast<uint8_t>(repeat / REPEAT_MAX);
    uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
    for (uint8_t i = 0; i < repeatLoop; i++) {
        vmuls(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src0 + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src1,
              REPEAT_MAX, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    if (repeatMod != 0) {
        vmuls(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
              src0 + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src1, repeatMod, dstBlockStride, srcBlockStride,
              dstRepeatStride, srcRepeatStride);
    }
}

// dim1
template <typename T, unsigned TShape0, unsigned dstRawShape0, unsigned src0RawShape0>
TILEOP void Tmuls(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    TileOp::Tmuls<T, 1, TShape0, 1, dstRawShape0, 1, src0RawShape0>(dst, src0, src1);
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tmuls(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    static_assert((TShape2 * TShape3) % blockSize == 0);

    int alignSrc0 = src0RawShape0 * src0RawShape1 > TShape2 * TShape3 ? src0RawShape0 * src0RawShape1 :
                                                                        TShape2 * TShape3;
    int alignDst = dstRawShape0 * dstRawShape1 > TShape2 * TShape3 ? dstRawShape0 * dstRawShape1 : TShape2 * TShape3;
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::Tmuls<T, TShape2, TShape3, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src0,
                                                                                                         src1);
            dst += alignDst;
            src0 += alignSrc0;
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tmuls(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    static_assert((TShape1 * TShape2) % blockSize == 0);

    int alignSrc0 = src0RawShape0 * src0RawShape1 > TShape1 * TShape2 ? src0RawShape0 * src0RawShape1 :
                                                                        TShape1 * TShape2;
    int alignDst = dstRawShape0 * dstRawShape1 > TShape1 * TShape2 ? dstRawShape0 * dstRawShape1 : TShape1 * TShape2;
    for (int i = 0; i < TShape0; ++i) {
        TileOp::Tmuls<T, TShape1, TShape2, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src0, src1);
        dst += alignDst;
        src0 += alignSrc0;
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstRawShape0, unsigned dstRawShape1,
          unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tdivs(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    // NEXTNEXT 1/0
    if (src1 != 0) {
        src1 = 1 / src1;
    }
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 8;
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    // ub discontinuous
    if constexpr ((dstRawShape1 > TShape1 || src0RawShape1 > TShape1) &&
                  (TShape1 % blockSize == 0 && src0RawShape1 % blockSize == 0)) {
        srcRepeatStride = src0RawShape1 > TShape1 ? src0RawShape1 / blockSize : TShape1 / blockSize;
        dstRepeatStride = dstRawShape1 > TShape1 ? dstRawShape1 / blockSize : TShape1 / blockSize;

        set_vector_mask(
            static_cast<uint64_t>(
                (TShape1 > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1 - 64)) - 1) : 0),
            static_cast<uint64_t>((TShape1 >= 64) ?
                                      0xffffffffffffffff :
                                      (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1)) - 1)));
        vmuls(dst, src0, src1, TShape0, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);

        set_vector_mask(-1, -1);
        return;
    }

    // ub continuous
    if constexpr (repeat < 1) {
        constexpr uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems);
        vmuls(dst, src0, src1, 1, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint8_t repeatLoop = static_cast<uint8_t>(repeat / REPEAT_MAX);
    uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
    for (uint8_t i = 0; i < repeatLoop; i++) {
        vmuls(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src0 + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src1,
              REPEAT_MAX, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    if (repeatMod != 0) {
        vmuls(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
              src0 + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src1, repeatMod, dstBlockStride, srcBlockStride,
              dstRepeatStride, srcRepeatStride);
    }
}

// dim1
template <typename T, unsigned TShape0, unsigned dstRawShape0, unsigned src0RawShape0>
TILEOP void Tdivs(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    TileOp::Tdivs<T, 1, TShape0, 1, dstRawShape0, 1, src0RawShape0>(dst, src0, src1);
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstRawShape0, unsigned dstRawShape1,
          unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tdivs(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, float src1)
{
    // NEXTNEXT 1/0
    if (src1 != 0) {
        src1 = (float)1.0 / src1;
    }
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 8;
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    // ub discontinuous
    if constexpr ((dstRawShape1 > TShape1 || src0RawShape1 > TShape1) &&
                  (TShape1 % blockSize == 0 && src0RawShape1 % blockSize == 0)) {
        srcRepeatStride = src0RawShape1 > TShape1 ? src0RawShape1 / blockSize : TShape1 / blockSize;
        dstRepeatStride = dstRawShape1 > TShape1 ? dstRawShape1 / blockSize : TShape1 / blockSize;

        set_vector_mask(
            static_cast<uint64_t>(
                (TShape1 > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1 - 64)) - 1) : 0),
            static_cast<uint64_t>((TShape1 >= 64) ?
                                      0xffffffffffffffff :
                                      (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1)) - 1)));

        vconv_s322f32(reinterpret_cast<__ubuf__ float*>(dst), src0, TShape0, dstBlockStride, srcBlockStride,
                      dstRepeatStride, srcRepeatStride);
        pipe_barrier(PIPE_V);
        vmuls(reinterpret_cast<__ubuf__ float*>(dst), reinterpret_cast<__ubuf__ float*>(dst), (float)src1, TShape0,
              dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        pipe_barrier(PIPE_V);
        vconv_f322s32z(dst, reinterpret_cast<__ubuf__ float*>(dst), TShape0, dstBlockStride, srcBlockStride,
                       dstRepeatStride, srcRepeatStride);
        pipe_barrier(PIPE_V);
        set_vector_mask(-1, -1);
        return;
    }

    // ub continuous
    if constexpr (repeat < 1) {
        constexpr uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems);
        vconv_s322f32(reinterpret_cast<__ubuf__ float*>(dst), src0, 1, dstBlockStride, srcBlockStride, dstRepeatStride,
                      srcRepeatStride);
        pipe_barrier(PIPE_V);
        vmuls(reinterpret_cast<__ubuf__ float*>(dst), reinterpret_cast<__ubuf__ float*>(dst), (float)src1, 1,
              dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        pipe_barrier(PIPE_V);
        vconv_f322s32z(dst, reinterpret_cast<__ubuf__ float*>(dst), 1, dstBlockStride, srcBlockStride, dstRepeatStride,
                       srcRepeatStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint8_t repeatLoop = static_cast<uint8_t>(repeat / REPEAT_MAX);
    uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
    for (uint8_t i = 0; i < repeatLoop; i++) {
        vconv_s322f32(reinterpret_cast<__ubuf__ float*>(dst), src0, REPEAT_MAX, dstBlockStride, srcBlockStride,
                      dstRepeatStride, srcRepeatStride);
        pipe_barrier(PIPE_V);
        vmuls(reinterpret_cast<__ubuf__ float*>(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T)),
              reinterpret_cast<__ubuf__ float*>(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T)), (float)src1,
              REPEAT_MAX, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        pipe_barrier(PIPE_V);
        vconv_f322s32z(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
                       reinterpret_cast<__ubuf__ float*>(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T)), REPEAT_MAX,
                       dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    if (repeatMod != 0) {
        vconv_s322f32(reinterpret_cast<__ubuf__ float*>(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T)),
                      src0 + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), repeatMod, dstBlockStride,
                      srcBlockStride, dstRepeatStride, srcRepeatStride);
        pipe_barrier(PIPE_V);
        vmuls(reinterpret_cast<__ubuf__ float*>(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T)),
              reinterpret_cast<__ubuf__ float*>(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T)), src1,
              repeatMod, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        pipe_barrier(PIPE_V);
        vconv_f322s32z(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
                       reinterpret_cast<__ubuf__ float*>(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T)),
                       repeatMod, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
}

template <typename T, unsigned TShape0, unsigned dstRawShape0, unsigned src0RawShape0>
TILEOP void Tdivs(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, int32_t src1)
{
    TileOp::Tdivs<int32_t, 1, TShape0, 1, dstRawShape0, 1, src0RawShape0>(dst, src0, (float)src1);
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tdivs(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    static_assert((TShape1 * TShape2) % blockSize == 0);

    int alignSrc0 = src0RawShape0 * src0RawShape1 > TShape1 * TShape2 ? src0RawShape0 * src0RawShape1 :
                                                                        TShape1 * TShape2;
    int alignDst = dstRawShape0 * dstRawShape1 > TShape1 * TShape2 ? dstRawShape0 * dstRawShape1 : TShape1 * TShape2;
    for (int i = 0; i < TShape0; ++i) {
        TileOp::Tdivs<T, TShape1, TShape2, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src0, src1);
        dst += alignDst;
        src0 += alignSrc0;
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tdivs(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    static_assert((TShape2 * TShape3) % blockSize == 0);

    int alignSrc0 = src0RawShape0 * src0RawShape1 > TShape2 * TShape3 ? src0RawShape0 * src0RawShape1 :
                                                                        TShape2 * TShape3;
    int alignDst = dstRawShape0 * dstRawShape1 > TShape2 * TShape3 ? dstRawShape0 * dstRawShape1 : TShape2 * TShape3;
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::Tdivs<T, TShape2, TShape3, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src0,
                                                                                                         src1);
            dst += alignDst;
            src0 += alignSrc0;
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstRawShape0, unsigned dstRawShape1,
          unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tmins(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 8;
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    // ub discontinuous
    if constexpr ((src0RawShape1 > TShape1) && (TShape1 % blockSize == 0 && src0RawShape1 % blockSize == 0)) {
        srcRepeatStride = src0RawShape1 > TShape1 ? src0RawShape1 / blockSize : TShape1 / blockSize;
        dstRepeatStride = dstRawShape1 > TShape1 ? dstRawShape1 / blockSize : TShape1 / blockSize;

        set_vector_mask(
            static_cast<uint64_t>(
                (TShape1 > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1 - 64)) - 1) : 0),
            static_cast<uint64_t>((TShape1 >= 64) ?
                                      0xffffffffffffffff :
                                      (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1)) - 1)));
        vmins(dst, src0, src1, TShape0, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);

        set_vector_mask(-1, -1);
        return;
    }

    // ub continuous
    if constexpr (repeat < 1) {
        constexpr uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems);
        vmins(dst, src0, src1, 1, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint8_t repeatLoop = static_cast<uint8_t>(repeat / REPEAT_MAX);
    uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
    for (uint8_t i = 0; i < repeatLoop; i++) {
        vmins(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src0 + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src1,
              REPEAT_MAX, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    if (repeatMod != 0) {
        vmins(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
              src0 + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src1, repeatMod, dstBlockStride, srcBlockStride,
              dstRepeatStride, srcRepeatStride);
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tmins(__ubuf__ T* dst, __ubuf__ T* src0, T src1)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    static_assert((TShape2 * TShape3) % blockSize == 0);

    int alignSrc0 = src0RawShape0 * src0RawShape1 > TShape2 * TShape3 ? src0RawShape0 * src0RawShape1 :
                                                                        TShape2 * TShape3;
    int alignDst = dstRawShape0 * dstRawShape1 > TShape2 * TShape3 ? dstRawShape0 * dstRawShape1 : TShape2 * TShape3;
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::Tmins<T, TShape2, TShape3, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src0,
                                                                                                         src1);
            dst += alignDst;
            src0 += alignSrc0;
        }
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1>
TILEOP void Tcompact(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    constexpr uint16_t srcStride = static_cast<uint16_t>(srcShape1 * sizeof(T) / BLOCK_SIZE);
    constexpr uint16_t repeat1 = dstShape0 / 8;
    constexpr uint16_t repeat2 = dstShape0 / 32; // repeat1 / 4
    for (int i = 0; i < dstShape0; i++) {
        copy_ubuf_to_ubuf(tmp + i * 8, src + i * srcShape1, 0, 1, 1, 1, 1);
    }
    pipe_barrier(PIPE_V);
    if constexpr (repeat1 < 1) {
        vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(tmp), reinterpret_cast<__ubuf__ uint32_t*>(tmp), nullptr, 1, 1,
                  3, 0, 0);
    } else {
        vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(tmp), reinterpret_cast<__ubuf__ uint32_t*>(tmp), nullptr,
                  repeat1, 1, 3, 8, 8);
    }

    pipe_barrier(PIPE_V);
    if constexpr (repeat2 < 1) {
        constexpr uint16_t mask_count = dstShape0 * 2;
        set_mask_count();
        set_vector_mask(0, mask_count);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(dst), reinterpret_cast<__ubuf__ uint32_t*>(tmp), nullptr, 1, 1,
                  1, 0, 0);
        set_mask_norm();
        set_vector_mask(-1, -1);
    } else {
        vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(dst), reinterpret_cast<__ubuf__ uint32_t*>(tmp), nullptr,
                  repeat2, 1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2, unsigned dstShape3,
          unsigned srcShape0, unsigned srcShape1, unsigned srcShape2, unsigned srcShape3>
TILEOP void Tcompact(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    for (int i = 0; i < dstShape0; ++i) {
        for (int j = 0; j < dstShape1; ++j) {
            TileOp::Tcompact<T, dstShape2, dstShape3, srcShape2, srcShape3>(dst, src, tmp);
            dst += dstShape2 * dstShape3;
            src += srcShape2 * srcShape3;
        }
    }
}
// dim2
template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
          unsigned dstRawShape1, unsigned srcRawShape1, unsigned axis>
TILEOP void Texpand_(__ubuf__ T* dst, __ubuf__ T* src)
{
    if (axis == 0) {
        // 1 16 -> 16 16
        constexpr uint64_t blockLen = (dstShape1 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int i = 0; i < dstShape0; i++) {
            copy_ubuf_to_ubuf(dst + i * dstRawShape1, src, 0, 1, blockLen, 1, 1);
        }
    } else if (axis == 1) {
        constexpr uint64_t shape1Repeat = static_cast<uint64_t>(dstShape1 * sizeof(T) / REPEAT_BYTE);
        if constexpr (shape1Repeat < 1) {
            // 16 1 -> 16 16
            SetContinuousMask(dstShape1);
            for (int i = 0; i < dstShape0; i++) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                T tmp = (T)(*(src + i * srcRawShape1));
                set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                vector_dup(dst + i * dstRawShape1, tmp, 1, 1, 0, 0, 0);
            }
            set_vector_mask(-1, -1);
        } else {
            // 16 1 -> 16 64
            constexpr unsigned reptEleNum = REPEAT_BYTE / sizeof(T);
            constexpr uint64_t remainNum = static_cast<uint64_t>(dstShape1 % reptEleNum);
            constexpr unsigned numLoop = shape1Repeat / REPEAT_MAX;
            constexpr unsigned remainAfterLoop = shape1Repeat % REPEAT_MAX;
            for (int i = 0; i < dstShape0; i++) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                T tmp = (T)(*(src + i * srcRawShape1));
                set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                if constexpr (numLoop) {
                    for (int j = 0; j < numLoop; j++) {
                        vector_dup(dst + i * dstRawShape1 + j * reptEleNum * REPEAT_MAX, tmp, REPEAT_MAX, 1, 1, 8, 0);
                    }
                }
                if constexpr (remainAfterLoop) {
                    vector_dup(dst + i * dstRawShape1 + numLoop * reptEleNum * REPEAT_MAX, tmp, remainAfterLoop, 1, 1,
                               8, 0);
                }
                if (remainNum) {
                    SetContinuousMask(remainNum);
                    vector_dup(dst + i * dstRawShape1 + shape1Repeat * reptEleNum, tmp, 1, 1, 1, 8, 0);
                    set_vector_mask(-1, -1);
                }
            }
        }
    }
}
// dim3
template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2, unsigned srcShape0,
          unsigned srcShape1, unsigned srcShape2, unsigned dstRawShape1, unsigned dstRawShape2, unsigned srcRawShape1,
          unsigned srcRawShape2, unsigned axis>
TILEOP void Texpand_(__ubuf__ T* dst, __ubuf__ T* src)
{
    if (axis == 1 || axis == 2) {
        // 16 1 16 -> 16 16 16 or 16 16 1 -> 16 16 16
        for (unsigned i = 0; i < dstShape0; i++) {
            TileOp::Texpand_<T, dstShape1, dstShape2, srcShape1, srcShape2, dstRawShape2, srcRawShape2, axis - 1>(dst,
                                                                                                                  src);
            dst += dstRawShape1 * dstRawShape2;
            src += srcRawShape1 * srcRawShape2;
        }
    } else if (axis == 0) {
        // 1 16 16 -> 16 16 16
        constexpr uint64_t blockLen = (dstShape2 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        constexpr uint64_t srcGap = (srcRawShape2 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        constexpr uint64_t dstGap = (dstRawShape2 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        for (unsigned i = 0; i < dstShape0; i++) {
            copy_ubuf_to_ubuf(dst + i * dstRawShape1 * dstRawShape2, src, 0, dstShape1, blockLen, srcGap, dstGap);
        }
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2, unsigned dstShape3,
          unsigned srcShape0, unsigned srcShape1, unsigned srcShape2, unsigned srcShape3, unsigned dstRawShape1,
          unsigned dstRawShape2, unsigned dstRawShape3, unsigned srcRawShape1, unsigned srcRawShape2,
          unsigned srcRawShape3, unsigned axis>
TILEOP void Texpand_(__ubuf__ T* dst, __ubuf__ T* src)
{
    static_assert((dstShape0 != srcShape0) || (dstShape1 != srcShape1) || (dstShape2 != srcShape2) ||
                  (dstShape3 != srcShape3));
    if (axis == 0) {
        // 1 16 16 16 -> 16 16 16 16
        constexpr uint64_t blockLen = (dstShape3 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        constexpr uint64_t srcGap = (srcRawShape3 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        constexpr uint64_t dstGap = (dstRawShape3 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        for (unsigned i = 0; i < dstShape0; ++i) {
            for (unsigned j = 0; j < dstShape1; j++) {
                copy_ubuf_to_ubuf(
                    dst + i * dstRawShape1 * dstRawShape2 * dstRawShape3 + j * dstRawShape2 * dstRawShape3,
                    src + j * srcRawShape2 * srcRawShape3, 0, dstShape2, blockLen, srcGap, dstGap);
            }
        }
    } else if (axis == 1 || axis == 2 || axis == 3) {
        for (unsigned i = 0; i < dstShape0; ++i) {
            TileOp::Texpand_<T, dstShape1, dstShape2, dstShape3, srcShape1, srcShape2, srcShape3, dstRawShape2,
                             dstRawShape3, srcRawShape2, srcRawShape3, axis - 1>(dst, src);
            dst += dstRawShape1 * dstRawShape2 * dstRawShape3;
            src += srcRawShape1 * srcRawShape2 * srcRawShape3;
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstRawShape0, unsigned dstRawShape1,
          unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Trec(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 8;

    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    // UB discontinuous
    if ((dstRawShape1 > TShape1 || src0RawShape1 > TShape1) &&
        (TShape1 % blockSize == 0 && dstRawShape1 % blockSize == 0 && src0RawShape1 % blockSize == 0)) {
        srcRepeatStride = src0RawShape1 > TShape1 ? src0RawShape1 / blockSize : TShape1 / blockSize;
        dstRepeatStride = dstRawShape1 > TShape1 ? dstRawShape1 / blockSize : TShape1 / blockSize;

        set_vector_mask(
            static_cast<uint64_t>(
                (TShape1 > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1 - 64)) - 1) : 0),
            static_cast<uint64_t>((TShape1 >= 64) ?
                                      0xffffffffffffffff :
                                      (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1)) - 1)));
        vrec(dst, src, TShape0, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_vector_mask(-1, -1);
        return;
    }

    // UB continuous
    if constexpr (repeat < 1) {
        constexpr uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems);
        vrec(dst, src, 1, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint8_t repeatLoop = static_cast<uint8_t>(repeat / REPEAT_MAX);
    uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
    for (uint8_t i = 0; i < repeatLoop; i++) {
        vrec(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), REPEAT_MAX,
             dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    if (repeatMod != 0) {
        vrec(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
             src + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), repeatMod, dstBlockStride, srcBlockStride,
             dstRepeatStride, srcRepeatStride);
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Trec(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    constexpr uint32_t baseTileSize = TShape2 * TShape3;
    constexpr uint32_t dstRawSize = dstRawShape0 * dstRawShape1;
    constexpr uint32_t src0RawSize = src0RawShape0 * src0RawShape1;
    constexpr uint32_t alignDst = dstRawSize > baseTileSize ? dstRawSize : baseTileSize;
    constexpr uint32_t alignSrc0 = src0RawSize > baseTileSize ? src0RawSize : baseTileSize;
    // ub需要32B对齐
    static_assert(baseTileSize % blockSize == 0);
    static_assert(alignDst % blockSize == 0);
    static_assert(alignSrc0 % blockSize == 0);

    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::Trec<T, TShape2, TShape3, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src);
            dst += alignDst;
            src += alignSrc0;
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Trec(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    constexpr uint32_t baseTileSize = TShape1 * TShape2;
    constexpr uint32_t dstRawSize = dstRawShape0 * dstRawShape1;
    constexpr uint32_t src0RawSize = src0RawShape0 * src0RawShape1;
    constexpr uint32_t alignDst = dstRawSize > baseTileSize ? dstRawSize : baseTileSize;
    constexpr uint32_t alignSrc0 = src0RawSize > baseTileSize ? src0RawSize : baseTileSize;
    // ub需要32B对齐
    static_assert(baseTileSize % blockSize == 0);
    static_assert(alignDst % blockSize == 0);
    for (int i = 0; i < TShape0; ++i) {
        TileOp::Trec<T, TShape1, TShape2, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src);
        dst += alignDst;
        src += alignSrc0;
    }
}

template <typename Td, typename Ts, unsigned Mode>
TILEOP void GenCastCall(__ubuf__ Td* dst, __ubuf__ Ts* src, uint8_t repeatNum, uint16_t dstBlockStride,
                        uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride)
{
    if constexpr (std::is_same<Td, half>::value && std::is_same<Ts, float>::value) {
        switch (static_cast<CastMode>(Mode)) {
            case CastMode::CAST_RINT:
                vconv_f322f16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_ROUND:
                vconv_f322f16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_FLOOR:
                vconv_f322f16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_CEIL:
                vconv_f322f16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_TRUNC:
                vconv_f322f16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_ODD:
                vconv_f322f16o(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_f322f16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    } else if constexpr (std::is_same<Td, float>::value && std::is_same<Ts, half>::value) {
        vconv_f162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<Td, bfloat16_t>::value && std::is_same<Ts, float>::value) {
        switch (static_cast<CastMode>(Mode)) {
            case CastMode::CAST_RINT:
                vconv_f322bf16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_ROUND:
                vconv_f322bf16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_FLOOR:
                vconv_f322bf16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_CEIL:
                vconv_f322bf16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_TRUNC:
                vconv_f322bf16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_f322bf16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    } else if constexpr (std::is_same<Td, float>::value && std::is_same<Ts, bfloat16_t>::value) {
        vconv_bf162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<Td, int16_t>::value && std::is_same<Ts, float>::value) {
        switch (static_cast<CastMode>(Mode)) {
            case CastMode::CAST_RINT:
                vconv_f322s16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_ROUND:
                vconv_f322s16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_FLOOR:
                vconv_f322s16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_CEIL:
                vconv_f322s16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_TRUNC:
                vconv_f322s16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_f322s16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    } else if constexpr (std::is_same<Td, float>::value && std::is_same<Ts, int16_t>::value) {
        vconv_s162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<Td, int32_t>::value && std::is_same<Ts, float>::value) {
        switch (static_cast<CastMode>(Mode)) {
            case CastMode::CAST_RINT:
                vconv_f322s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_ROUND:
                vconv_f322s32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_FLOOR:
                vconv_f322s32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_CEIL:
                vconv_f322s32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_TRUNC:
                vconv_f322s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_f322s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    } else if constexpr (std::is_same<Td, float>::value && std::is_same<Ts, int32_t>::value) {
        vconv_s322f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<Td, half>::value && std::is_same<Ts, int32_t>::value) {
        set_deqscale(static_cast<half>(1.0));
        pipe_barrier(PIPE_V);
        vconv_deq(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<Td, int32_t>::value && std::is_same<Ts, half>::value) {
        switch (static_cast<CastMode>(Mode)) {
            case CastMode::CAST_RINT:
                vconv_f162s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_ROUND:
                vconv_f162s32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_FLOOR:
                vconv_f162s32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_CEIL:
                vconv_f162s32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_TRUNC:
                vconv_f162s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_f162s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    } else if constexpr (std::is_same<Td, half>::value && std::is_same<Ts, int8_t>::value) {
        vconv_s82f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<Td, int8_t>::value && std::is_same<Ts, half>::value) {
        switch (static_cast<CastMode>(Mode)) {
            case CastMode::CAST_ROUND:
                vconv_f162s8a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_FLOOR:
                vconv_f162s8f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_CEIL:
                vconv_f162s8c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_TRUNC:
                vconv_f162s8z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_f162s8z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    } else if constexpr (std::is_same<Td, float>::value && std::is_same<Ts, float>::value) {
        switch (static_cast<CastMode>(Mode)) {
            case CastMode::CAST_RINT:
                vconv_f322f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_ROUND:
                vconv_f322f32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_FLOOR:
                vconv_f322f32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_CEIL:
                vconv_f322f32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_TRUNC:
                vconv_f322f32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_f322f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    } else if constexpr (std::is_same<Td, int32_t>::value && std::is_same<Ts, bfloat16_t>::value) {
        switch (static_cast<CastMode>(Mode)) {
            case CastMode::CAST_RINT:
                vconv_bf162s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_ROUND:
                vconv_bf162s32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_FLOOR:
                vconv_bf162s32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_CEIL:
                vconv_bf162s32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            case CastMode::CAST_TRUNC:
                vconv_bf162s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_bf162s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    } else {
        static_assert(sizeof(Td) == 0, "Unsupported type conversion");
    }
}

template <typename Td, typename Ts, unsigned T0, unsigned T1, unsigned DS, unsigned SS, unsigned Mode>
TILEOP void Tcast_(__ubuf__ Td* dst, __ubuf__ Ts* src)
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
        constexpr unsigned numLoop = T0 / REPEAT_MAX;
        constexpr unsigned remainAfterLoop = T0 % REPEAT_MAX;
        SetContinuousMask(numRemainPerLine);
        if constexpr (numLoop) {
            for (int i = 0; i < numLoop; i++) {
                GenCastCall<Td, Ts, Mode>(dst + i * REPEAT_MAX * DS, src + i * REPEAT_MAX * SS, (uint8_t)REPEAT_MAX, 1,
                                          1, (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
            }
        }
        if constexpr (remainAfterLoop) {
            GenCastCall<Td, Ts, Mode>(dst + numLoop * REPEAT_MAX * DS, src + numLoop * REPEAT_MAX * SS,
                                      (uint8_t)remainAfterLoop, 1, 1, (uint16_t)DS / dstNElemPerBlock,
                                      (uint16_t)SS / srcNElemPerBlock);
        }
        set_vector_mask(-1, -1);
    }
}

template <typename Td, typename Ts, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned DS0, unsigned DS1,
          unsigned DS2, unsigned SS0, unsigned SS1, unsigned SS2, unsigned Mode>
TILEOP void Tcast_(__ubuf__ Td* dst, __ubuf__ Ts* src)
{
    static_assert((DS2 * sizeof(Td)) % BLOCK_SIZE == 0);
    static_assert((SS2 * sizeof(Ts)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ Td* dst_ = dst;
        __ubuf__ Ts* src_ = src;
        for (int j = 0; j < T1; j++) {
            Tcast_<Td, Ts, T2, T3, DS2, SS2, Mode>(dst_, src_);
            dst_ += DS1 * DS2;
            src_ += SS1 * SS2;
        }
        dst += DS0 * DS1 * DS2;
        src += SS0 * SS1 * SS2;
    }
}

template <typename T, unsigned TShape0, unsigned TShape1>
TILEOP void Treducesum(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint8_t repeat = TShape1 * sizeof(T) / BLOCK_SIZE;
    constexpr uint8_t srcBlockStride = 1;
    constexpr uint8_t dstRptStride = TShape1 * sizeof(T) / BLOCK_SIZE;
    constexpr uint8_t srcRptStride = TShape1 * sizeof(T) / BLOCK_SIZE;

    set_mask_count();
    set_vector_mask(0, TShape1);
    int maxRepeatLen = REPEAT_BYTE / sizeof(T);
    if (TShape1 <= maxRepeatLen) {
        for (int i = 0; i < TShape0; i++) {
            vcadd(dst + i * TShape1, src + i * TShape1, 1, dstRptStride, srcBlockStride, srcRptStride, false);
        }
    } else {
        // TD: 当前只支持TShape1是2的指数倍，即二分累加时，不存在余数
        int leftSize = TShape1;
        int addSize = leftSize / 2;
        if (leftSize > maxRepeatLen) { // 为了避免影响src原值
            set_vector_mask(0, addSize);
            for (int i = 0; i < TShape0; i++) {
                vadd(dst + i * TShape1, src + i * TShape1, src + i * TShape1 + addSize, 1, 1, 1, 1, 8, 8, 8);
            }
        }
        pipe_barrier(PIPE_V);

        // 二分累加
        leftSize = addSize;
        while (leftSize > maxRepeatLen) {
            addSize = leftSize / 2;

            set_vector_mask(0, addSize);
            for (int i = 0; i < TShape0; i++) {
                vadd(dst + i * TShape1, dst + i * TShape1, dst + i * TShape1 + addSize, 1, 1, 1, 1, 8, 8, 8);
            }
            pipe_barrier(PIPE_V);

            leftSize = addSize;
        }

        // reduceSum
        set_vector_mask(0, leftSize);
        for (int i = 0; i < TShape0; i++) {
            vcadd(dst + i * TShape1, dst + i * TShape1, 1, 0, 1, 0, false);
        }
    }
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3>
TILEOP void Treducesum(__ubuf__ T* dst, __ubuf__ T* src)
{
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::Treducesum<T, TShape2, TShape3>(dst, src);
            dst += TShape2 * TShape3;
            src += TShape2 * TShape3;
        }
    }
}

// The src data remains unchanged.
// T: fp32. support: OS0 <= REPEAT_MAX
template <typename T, unsigned OS0, unsigned OS1, unsigned DS, unsigned SS, unsigned TBS>
TILEOP void Trowsumsinglecombine(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    static_assert(OS1 == SS);
    if constexpr (SS == 1024) {
        static_assert(OS0 * 16 <= REPEAT_MAX);
        vcgadd(tmp, src, OS0 * 16, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,1024] -> [m,128]
        pipe_barrier(PIPE_V);
        vadd(tmp, tmp + 64, tmp, OS0, 1, 1, 1, 16, 16, 16); // [m,128] -> [m,64]
        pipe_barrier(PIPE_V);
        vcgadd(tmp, tmp, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)16ULL); // [m,64] -> [m,8]
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgadd(dst, tmp, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 512) {
        static_assert(OS0 * 8 <= REPEAT_MAX);
        vcgadd(tmp, src, OS0 * 8, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,512] -> [m,64]
        pipe_barrier(PIPE_V);
        vcgadd(tmp, tmp, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,64] -> [m,8]
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgadd(dst, tmp, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 256) {
        static_assert(OS0 * 4 <= REPEAT_MAX);
        vcgadd(tmp, src, OS0 * 4, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,256] -> [m,32]
        pipe_barrier(PIPE_V);
        SetContinuousMask(16);
        vadd(tmp, tmp + 16, tmp, OS0, 1, 1, 1, 4, 4, 4); // [m,32] -> [m,16]
        pipe_barrier(PIPE_V);
        SetContinuousMask(8);
        vadd(tmp, tmp + 8, tmp, OS0, 1, 1, 1, 4, 4, 4); // [m,16] -> [m,8]
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        if constexpr (OS0 / 8 != 0) {
            vcgadd(dst, tmp, OS0 / 8, (uint16_t)1ULL, (uint16_t)4ULL, (uint16_t)32ULL); // [m,8] -> [m,1]
        }
        constexpr uint16_t reminder = OS0 % 8;
        if constexpr (reminder != 0) {
            SetContinuousMask(reminder * 8);
            vcgadd(dst + OS0 / 8 * 8, tmp + OS0 / 8 * 8 * 32, 1, (uint16_t)1ULL, (uint16_t)4ULL,
                   (uint16_t)32ULL); // [m,8] -> [m,1]
            set_vector_mask(-1, -1);
        }
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 128) {
        static_assert(OS0 <= REPEAT_MAX);
        vadd(tmp, src + 64, src, OS0, 1, 1, 1, 8, 16, 16); // [m,128] -> [m,64]
        pipe_barrier(PIPE_V);
        vcgadd(tmp, tmp, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,64] -> [m,8]
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgadd(dst, tmp, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 64) {
        static_assert(OS0 <= REPEAT_MAX);
        vcgadd(tmp, src, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,64] -> [m,8]
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgadd(dst, tmp, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 32) {
        static_assert(OS0 <= REPEAT_MAX);
        SetContinuousMask(16);
        vadd(tmp, src + 16, src, OS0, 1, 1, 1, 2, 4, 4); // [m,32] -> [m,16]
        pipe_barrier(PIPE_V);
        SetContinuousMask(8);
        vadd(tmp, tmp + 8, tmp, OS0, 1, 1, 1, 2, 2, 2); // [m,16] -> [m,8]
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        if constexpr (OS0 / 8 != 0) {
            vcgadd(dst, tmp, OS0 / 8, (uint16_t)1ULL, (uint16_t)2ULL, (uint16_t)16ULL); // [m,8] -> [m,1]
        }
        constexpr uint16_t reminder = OS0 % 8;
        if constexpr (reminder != 0) {
            SetContinuousMask(reminder * 8);
            vcgadd(dst + OS0 / 8 * 8, tmp + OS0 / 8 * 8 * 16, 1, (uint16_t)1ULL, (uint16_t)2ULL,
                   (uint16_t)16ULL); // [m,8] -> [m,1]
            set_vector_mask(-1, -1);
        }
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 8) {
        static_assert(OS0 / 8 <= REPEAT_MAX);
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgadd(dst, src, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    } else {
        static_assert(OS0 <= REPEAT_MAX);
        int32_t loop = OS1 / 8 - 1;
        SetContinuousMask(8);
        vadd(tmp, src + 8, src, OS0, 1, 1, 1, 1, OS1 / 8, OS1 / 8);
        pipe_barrier(PIPE_V);
        for (int32_t i = 1; i < loop; i++) {
            vadd(tmp, tmp, src + (i + 1) * 8, OS0, 1, 1, 1, 1, 1, OS1 / 8);
            pipe_barrier(PIPE_V);
        }
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgadd(dst, tmp, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    }
}

template <typename T, unsigned OS0, unsigned OS1, unsigned OS2, unsigned OS3, unsigned DS1, unsigned DS2, unsigned DS3,
          unsigned SS1, unsigned SS2, unsigned SS3, unsigned TBS3>
TILEOP void Trowsumsinglecombine(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    static_assert(SS3 * sizeof(T) % BLOCK_SIZE == 0);
    for (int i = 0; i < OS0; ++i) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < OS1; ++j) {
            TileOp::Trowsumsinglecombine<T, OS2, OS3, DS3, SS3, TBS3>(dst_, src_, tmp);
            dst_ += DS3 * DS2;
            src_ += SS3 * SS2;
            pipe_barrier(PIPE_V);
        }
        dst += DS1 * DS2 * DS3;
        src += SS1 * SS2 * SS3;
    }
}

// The src data remains unchanged.
// T: fp32. support: OS0 <= REPEAT_MAX
template <typename T, unsigned OS0, unsigned OS1, unsigned DS, unsigned SS, unsigned TBS>
TILEOP void Trowsumsingle_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    static_assert(OS0 <= REPEAT_MAX);
    constexpr uint64_t srcRepeatPerRow = static_cast<uint64_t>(OS1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint16_t srcRepeatStride = SS * sizeof(T) / BLOCK_SIZE;
    constexpr unsigned nElemPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned remain = OS1 % nElemPerRepeat;
    if constexpr (srcRepeatPerRow == 1 && remain == 0) {
        vcadd(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, false);
        return;
    }

    if constexpr (OS0 <= REPEAT_MAX && srcRepeatPerRow < 1 && remain > 0) {
        SetContinuousMask(OS1);
        vcadd(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, false);
        set_vector_mask(-1, -1);
        return;
    }

    constexpr uint16_t tmpRepeatStride = TBS * sizeof(T) / BLOCK_SIZE;

    unsigned curLen = srcRepeatPerRow;
    for (unsigned i = 0; i < curLen / 2; i++) {
        vadd(tmp + i * nElemPerRepeat, src + i * 2 * nElemPerRepeat, src + (i * 2 + 1) * nElemPerRepeat, OS0, 1, 1, 1,
             tmpRepeatStride, srcRepeatStride, srcRepeatStride);
    }
    pipe_barrier(PIPE_V);
    if (curLen == 1 && remain > 0) {
        copy_ubuf_to_ubuf(tmp, src, 0, OS0, BLOCK_MAX_PER_REPEAT, srcRepeatStride - BLOCK_MAX_PER_REPEAT,
                          tmpRepeatStride - BLOCK_MAX_PER_REPEAT);
        pipe_barrier(PIPE_V);
    } else if (curLen % 2 > 0) {
        vadd(tmp, tmp, src + (curLen - 1) * nElemPerRepeat, OS0, 1, 1, 1, tmpRepeatStride, tmpRepeatStride,
             srcRepeatStride);
        pipe_barrier(PIPE_V);
    }

    if (remain > 0) {
        unsigned repeatOffset = curLen == 1 ? 0 : curLen / 2 - 1;
        SetContinuousMask(remain);
        vadd(tmp + repeatOffset * nElemPerRepeat, src + curLen * nElemPerRepeat, tmp + repeatOffset * nElemPerRepeat,
             OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride, tmpRepeatStride);
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }

    curLen = curLen / 2;
    bool mergeLast = true;
    while (curLen > 1) {
        for (unsigned i = 0; i < curLen / 2; i++) {
            vadd(tmp + i * nElemPerRepeat, tmp + i * 2 * nElemPerRepeat, tmp + (i * 2 + 1) * nElemPerRepeat, OS0, 1, 1,
                 1, tmpRepeatStride, tmpRepeatStride, tmpRepeatStride);
        }
        unsigned loopRemain = curLen % 2;
        curLen = curLen / 2;
        if (loopRemain > 0) {
            pipe_barrier(PIPE_V);
            vadd(tmp + (curLen - 1) * nElemPerRepeat /*last repeat of new curLen*/,
                 tmp + curLen * 2 * nElemPerRepeat /*remain repeat*/, tmp + (curLen - 1) * nElemPerRepeat, OS0, 1, 1, 1,
                 tmpRepeatStride, tmpRepeatStride, tmpRepeatStride);
        }
        pipe_barrier(PIPE_V);
    }
    pipe_barrier(PIPE_V);
    vcadd(dst, tmp, OS0, (uint16_t)DS, (uint16_t)1ULL, tmpRepeatStride, false);
}

template <typename T, unsigned OS0, unsigned OS1, unsigned OS2, unsigned OS3, unsigned DS1, unsigned DS2, unsigned DS3,
          unsigned SS1, unsigned SS2, unsigned SS3, unsigned TBS3>
TILEOP void Trowsumsingle_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    static_assert(SS3 * sizeof(T) % BLOCK_SIZE == 0);
    for (int i = 0; i < OS0; ++i) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < OS1; ++j) {
            TileOp::Trowsumsingle_<T, OS2, OS3, DS3, SS3, TBS3>(dst_, src_, tmp);
            dst_ += DS3 * DS2;
            src_ += SS3 * SS2;
            pipe_barrier(PIPE_V);
        }
        dst += DS1 * DS2 * DS3;
        src += SS1 * SS2 * SS3;
    }
}

// The src data remains unchanged.
// T: fp32. support: OS0 <= REPEAT_MAX
template <typename T, unsigned OS0, unsigned OS1, unsigned DS, unsigned SS, unsigned TBS>
TILEOP void Trowmaxsinglecombine(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    static_assert(OS1 == SS);
    if constexpr (SS == 1024) {
        static_assert(OS0 * 16 <= REPEAT_MAX);
        vcgmax(tmp, src, OS0 * 16, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,1024] -> [m,128]
        pipe_barrier(PIPE_V);
        vmax(tmp, tmp + 64, tmp, OS0, 1, 1, 1, 16, 16, 16); // [m,128] -> [m,64]
        pipe_barrier(PIPE_V);
        vcgmax(tmp, tmp, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)16ULL); // [m,64] -> [m,8]
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgmax(dst, tmp, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 512) {
        static_assert(OS0 * 8 <= REPEAT_MAX);
        vcgmax(tmp, src, OS0 * 8, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,512] -> [m,64]
        pipe_barrier(PIPE_V);
        vcgmax(tmp, tmp, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,64] -> [m,8]
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgmax(dst, tmp, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 256) {
        static_assert(OS0 * 4 <= REPEAT_MAX);
        vcgmax(tmp, src, OS0 * 4, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,256] -> [m,32]
        pipe_barrier(PIPE_V);
        SetContinuousMask(16);
        vmax(tmp, tmp + 16, tmp, OS0, 1, 1, 1, 4, 4, 4); // [m,32] -> [m,16]
        pipe_barrier(PIPE_V);
        SetContinuousMask(8);
        vmax(tmp, tmp + 8, tmp, OS0, 1, 1, 1, 4, 4, 4); // [m,16] -> [m,8]
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        if constexpr (OS0 / 8 != 0) {
            vcgmax(dst, tmp, OS0 / 8, (uint16_t)1ULL, (uint16_t)4ULL, (uint16_t)32ULL); // [m,8] -> [m,1]
        }
        constexpr uint16_t reminder = OS0 % 8;
        if constexpr (reminder != 0) {
            SetContinuousMask(reminder * 8);
            vcgmax(dst + OS0 / 8 * 8, tmp + OS0 / 8 * 8 * 32, 1, (uint16_t)1ULL, (uint16_t)4ULL,
                   (uint16_t)32ULL); // [m,8] -> [m,1]
            set_vector_mask(-1, -1);
        }
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 128) {
        static_assert(OS0 <= REPEAT_MAX);
        vmax(tmp, src + 64, src, OS0, 1, 1, 1, 8, 16, 16); // [m,128] -> [m,64]
        pipe_barrier(PIPE_V);
        vcgmax(tmp, tmp, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,64] -> [m,8]
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgmax(dst, tmp, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 64) {
        static_assert(OS0 <= REPEAT_MAX);
        vcgmax(tmp, src, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,64] -> [m,8]
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgmax(dst, tmp, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 32) {
        static_assert(OS0 <= REPEAT_MAX);
        SetContinuousMask(16);
        vmax(tmp, src + 16, src, OS0, 1, 1, 1, 2, 4, 4); // [m,32] -> [m,16]
        pipe_barrier(PIPE_V);
        SetContinuousMask(8);
        vmax(tmp, tmp + 8, tmp, OS0, 1, 1, 1, 2, 2, 2); // [m,16] -> [m,8]
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        if constexpr (OS0 / 8 != 0) {
            vcgmax(dst, tmp, OS0 / 8, (uint16_t)1ULL, (uint16_t)2ULL, (uint16_t)16ULL); // [m,8] -> [m,1]
        }
        constexpr uint16_t reminder = OS0 % 8;
        if constexpr (reminder != 0) {
            SetContinuousMask(reminder * 8);
            vcgmax(dst + OS0 / 8 * 8, tmp + OS0 / 8 * 8 * 16, 1, (uint16_t)1ULL, (uint16_t)2ULL,
                   (uint16_t)16ULL); // [m,8] -> [m,1]
            set_vector_mask(-1, -1);
        }
        pipe_barrier(PIPE_V);
        return;
    } else if constexpr (SS == 8) {
        static_assert(OS0 / 8 <= REPEAT_MAX);
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgmax(dst, src, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    } else {
        static_assert(OS0 <= REPEAT_MAX);
        int32_t loop = OS1 / 8 - 1;
        SetContinuousMask(8);
        vmax(tmp, src + 8, src, OS0, 1, 1, 1, 1, OS1 / 8, OS1 / 8);
        pipe_barrier(PIPE_V);
        for (int32_t i = 1; i < loop; i++) {
            vmax(tmp, tmp, src + (i + 1) * 8, OS0, 1, 1, 1, 1, 1, OS1 / 8);
            pipe_barrier(PIPE_V);
        }
        set_mask_count();
        set_vector_mask(0, OS0 * 8);
        vcgmax(dst, tmp, 1, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
        return;
    }
}

template <typename T, unsigned OS0, unsigned OS1, unsigned OS2, unsigned OS3, unsigned DS1, unsigned DS2, unsigned DS3,
          unsigned SS1, unsigned SS2, unsigned SS3, unsigned TBS3>
TILEOP void Trowmaxsinglecombine(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    static_assert(SS3 * sizeof(T) % BLOCK_SIZE == 0);
    for (int i = 0; i < OS0; ++i) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < OS1; ++j) {
            TileOp::Trowmaxsinglecombine<T, OS2, OS3, DS3, SS3, TBS3>(dst_, src_, tmp);
            dst_ += DS3 * DS2;
            src_ += SS3 * SS2;
            pipe_barrier(PIPE_V);
        }
        dst += DS1 * DS2 * DS3;
        src += SS1 * SS2 * SS3;
    }
}

// The src data remains unchanged.
// T: fp32. support: OS0 <= REPEAT_MAX
template <typename T, unsigned OS0, unsigned OS1, unsigned DS, unsigned SS, unsigned TBS>
TILEOP void Trowmaxsingle_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    static_assert(OS0 <= REPEAT_MAX);
    constexpr uint64_t srcRepeatPerRow = static_cast<uint64_t>(OS1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint16_t srcRepeatStride = SS * sizeof(T) / BLOCK_SIZE;
    constexpr unsigned nElemPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned remain = OS1 % nElemPerRepeat;
    if constexpr (srcRepeatPerRow == 1 && OS0 <= REPEAT_MAX && remain == 0) {
        vcmax(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, ONLY_VALUE);
        return;
    }

    if constexpr (OS0 <= REPEAT_MAX && srcRepeatPerRow < 1 && remain > 0) {
        SetContinuousMask(OS1);
        vcmax(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, ONLY_VALUE);
        set_vector_mask(-1, -1);
        return;
    }

    constexpr uint16_t tmpRepeatStride = TBS * sizeof(T) / BLOCK_SIZE;
    if (srcRepeatPerRow == 1 && remain > 0) {
        copy_ubuf_to_ubuf(tmp, src, 0, OS0, BLOCK_MAX_PER_REPEAT, srcRepeatStride - BLOCK_MAX_PER_REPEAT,
                          tmpRepeatStride - BLOCK_MAX_PER_REPEAT);
    } else {
        if ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
            vmax(tmp, src, src + nElemPerRepeat, OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride, srcRepeatStride);
        } else {
            for (int i = 0; i < OS0; i++) {
                vmax(tmp + i * TBS, src + i * SS, src + i * SS + nElemPerRepeat, 1, 1, 1, 1, 1, 1, 1);
            }
        }
    }
    pipe_barrier(PIPE_V);

    for (int i = 2; i < srcRepeatPerRow; i++) {
        if ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
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
        if ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
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

template <typename T, unsigned OS0, unsigned OS1, unsigned OS2, unsigned OS3, unsigned DS1, unsigned DS2, unsigned DS3,
          unsigned SS1, unsigned SS2, unsigned SS3, unsigned TBS3>
TILEOP void Trowmaxsingle_(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    static_assert(SS3 * sizeof(T) % BLOCK_SIZE == 0);
    for (int i = 0; i < OS0; ++i) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < OS1; ++j) {
            TileOp::Trowmaxsingle_<T, OS2, OS3, DS3, SS3, TBS3>(dst_, src_, tmp);
            dst_ += DS3 * DS2;
            src_ += SS3 * SS2;
            pipe_barrier(PIPE_V);
        }
        dst += DS1 * DS2 * DS3;
        src += SS1 * SS2 * SS3;
    }
}

// [case1] params: [src0Shape0,src0Shape1], indices: [TShape0], axis: 0, output: [TShape0,TShape1]
// [case2] params: [src0Shape0,src0Shape1], indices: [TShape0,TShape1], axis: 0, output: [TShape0,TShape1,TShape2]
template <typename T, typename T2, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned src0Shape2,
          unsigned dst0Shape2>
TILEOP void TgatherFromUB_(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T2* src1)
{
    constexpr uint16_t lenBurst = (TShape2 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            T2 index = (T2)(*(src1 + j)); // src1[i,j]
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);

            // dst, src, sid, nBurst, lenBurst, srcStride, dstStride
            copy_ubuf_to_ubuf(dst + j * dst0Shape2, src0 + index * src0Shape2, 0, 1, lenBurst, 1, 1);
        }

        dst += TShape1 * dst0Shape2;
        src1 += TShape1;
    }
}

template <typename T, typename T2, unsigned TShape0, unsigned TShape1, unsigned src0Shape1, unsigned dstShape1,
          unsigned axis>
TILEOP void TgatherElement(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T2* src1)
{
    constexpr uint16_t lenBurst = 1;
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            T2 index = (T2)(*(src1 + i * TShape1 + j)); // src1[i,j]
            int srcOffset = 0;
            if constexpr (axis == 0) {
                srcOffset = index * src0Shape1 + j;
            } else {
                srcOffset = i * src0Shape1 + index;
            }
            dst[i * dstShape1 + j] = src0[srcOffset];
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, typename T2, unsigned src1RawShape1, unsigned dstRawShape1, unsigned src1Shape0,
          unsigned src1Shape1, unsigned axis>
TILEOP void TscatterElementS(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T2* src1, T src2)
{
    for (int i = 0; i < src1Shape0; ++i) {
        for (int j = 0; j < src1Shape1; ++j) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            T2 index = (T2)(*(src1 + i * src1RawShape1 + j)); // src1[i,j]
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
            int dstOffset = 0;
            if constexpr (axis == 0) {
                dstOffset = index * dstRawShape1 + j;
            } else {
                dstOffset = i * dstRawShape1 + index;
            }
            dst[dstOffset] = src2;
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0,
          unsigned srcShape1, unsigned reverseOperand>
TILEOP void TSadds(__ubuf__ T* dst, __ubuf__ T* src, float scalar)
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

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0,
          unsigned srcShape1, unsigned reverseOperand>
TILEOP void TSsubs(__ubuf__ T* dst, __ubuf__ T* src, float scalar)
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

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0,
          unsigned srcShape1, unsigned reverseOperand>
TILEOP void TSmuls(__ubuf__ T* dst, __ubuf__ T* src, float scalar)
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

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0,
          unsigned srcShape1, unsigned reverseOperand>
TILEOP void TSdivs(__ubuf__ T* dst, __ubuf__ T* src, float scalar)
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

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned dstShape0, unsigned dstShape1,
          unsigned dstShape2, unsigned srcShape0, unsigned srcShape1, unsigned srcShape2, unsigned reverseOperand>
TILEOP void TSdivs(__ubuf__ T* dst, __ubuf__ T* src, float scalar)
{
    int dstOffset = dstShape1 * dstShape2;
    for (int i = 0; i < TShape0; i++) {
        TileOp::TSdivs<T, TShape1, TShape2, dstShape0, dstShape1, srcShape0, srcShape1, reverseOperand>(
            dst + i * dstOffset, src + i * dstOffset, scalar);
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0,
          unsigned srcShape1, unsigned reverseOperand>
TILEOP void TSmaxs(__ubuf__ T* dst, __ubuf__ T* src, float scalar)
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

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned dstRawShape1,
          unsigned dstRawShape2, unsigned srcRawShape1, unsigned srcRawShape2, unsigned axis0, unsigned axis1>
TILEOP void TtransposeMoveOut_(__gm__ T* dst, __ubuf__ T* src)
{
    if constexpr (axis0 == 0 && axis1 == 1) {
        __gm__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        unsigned nBurst = TShape1;
        unsigned lenBurst = TShape2 * sizeof(T);
        unsigned srcStride = (srcRawShape2 - TShape2) / BLOCK_NELEM_B32;
        unsigned dstStride = (dstRawShape1 - 1) * dstRawShape2 * sizeof(T);
        for (int b = 0; b < TShape0; b++) {
            copy_ubuf_to_gm_align_b32(dst_, src_, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
            dst_ += dstRawShape2;
            src_ += srcRawShape1 * srcRawShape2;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned dstRawShape1,
          unsigned dstRawShape2, unsigned dstRawShape3, unsigned srcRawShape1, unsigned srcRawShape2,
          unsigned srcRawShape3, unsigned axis0, unsigned axis1>
TILEOP void TtransposeMoveOut_(__gm__ T* dst, __ubuf__ T* src)
{
    if constexpr (axis0 == 1 && axis1 == 2) {
        __gm__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int b = 0; b < TShape0; b++) {
            TtransposeMoveOut_<T, TShape1, TShape2, TShape3, dstRawShape2, dstRawShape3, srcRawShape2, srcRawShape3,
                               axis0 - 1, axis1 - 1>(dst_, src_);
            dst_ += dstRawShape1 * dstRawShape2 * dstRawShape3;
            src_ += srcRawShape1 * srcRawShape2 * srcRawShape3;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

// dim2
template <typename T, unsigned TShape0, unsigned TShape1, unsigned srcRawShape1>
TILEOP void Trowsumline_(__ubuf__ T* dst, __ubuf__ T* src0)
{
    static_assert(sizeof(T) == 4);
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
            SetContinuousMask(remainElm);
            vcopy((__ubuf__ uint32_t*)(dst + (repeatTime - 1) * rptElm),
                  (__ubuf__ uint32_t*)(src0 + (repeatTime - 1) * rptElm), 1, 1, 1, 8, 8);
            set_vector_mask(-1, -1);
        }
    }
    pipe_barrier(PIPE_V);

    for (uint32_t j = 1; j < TShape0; j++) {
        if (!remainElm) {
            vadd(dst, dst, src0 + j * srcRawShape1, repeatTime, 1, 1, 1, 8, 8, 8);
        } else {
            if (repeatTime == 1) {
                SetContinuousMask(remainElm);
                vadd(dst, dst, src0 + j * srcRawShape1, 1, 1, 1, 1, 8, 8, 8);
                set_vector_mask(-1, -1);
            } else {
                vadd(dst, dst, src0 + j * srcRawShape1, repeatTime - 1, 1, 1, 1, 8, 8, 8);
                SetContinuousMask(remainElm);
                vadd(dst + (repeatTime - 1) * rptElm, dst + (repeatTime - 1) * rptElm,
                     src0 + j * srcRawShape1 + (repeatTime - 1) * rptElm, 1, 1, 1, 1, 8, 8, 8);
                set_vector_mask(-1, -1);
            }
        }
        pipe_barrier(PIPE_V);
    }
}

// dim3
template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned srcRawShape1,
          unsigned srcRawShape2, unsigned dstRawShape1, unsigned dstRawShape2, unsigned axis>
TILEOP void Trowsumline_(__ubuf__ T* dst, __ubuf__ T* src0)
{
    static_assert(sizeof(T) == 4);
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
                    SetContinuousMask(remainElm);
                    vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 + (repeatTime - 1) * rptElm),
                          (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 + (repeatTime - 1) * rptElm), 1, 1, 1, 8, 8);
                    set_vector_mask(-1, -1);
                }
            }
        }
        pipe_barrier(PIPE_V);
        for (unsigned i = 1; i < TShape0; i++) {
            for (unsigned j = 0; j < TShape1; j++) {
                if (!remainElm) {
                    vadd(dst + j * dstRawShape2, dst + j * dstRawShape2,
                         src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime, 1, 1, 1, 8, 8, 8);
                } else {
                    if (repeatTime == 1) {
                        set_vector_mask(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                        vadd(dst + j * dstRawShape2, dst + j * dstRawShape2,
                             src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime, 1, 1, 1, 8, 8, 8);
                        set_vector_mask(-1, -1);
                    } else {
                        vadd(dst + j * dstRawShape2, dst + j * dstRawShape2,
                             src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime - 1, 1, 1, 1, 8, 8,
                             8);
                        set_vector_mask(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                        vadd(dst + j * dstRawShape2 + (repeatTime - 1) * rptElm,
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
            Trowsumline_<T, TShape1, TShape2, srcRawShape2>(dst + i * dstRawShape1 * dstRawShape2,
                                                            src0 + i * srcRawShape1 * srcRawShape2);
        }
    }
}

// dim4
template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned srcRawShape1,
          unsigned srcRawShape2, unsigned srcRawShape3, unsigned dstRawShape1, unsigned dstRawShape2,
          unsigned dstRawShape3, unsigned tmpRawShape1, unsigned tmpRawShape2, unsigned tmpRawShape3, unsigned axis>
TILEOP void Trowsumline_(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* tmp)
{
    static_assert(sizeof(T) == 4);
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
                        SetContinuousMask(remainElm);
                        vcopy((__ubuf__ uint32_t*)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3 +
                                                   (repeatTime - 1) * rptElm),
                              (__ubuf__ uint32_t*)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3 +
                                                   (repeatTime - 1) * rptElm),
                              1, 1, 1, 8, 8);
                        set_vector_mask(-1, -1);
                    }
                }
            }
        }
        pipe_barrier(PIPE_V);
        for (unsigned i = 1; i < TShape0; i++) {
            for (unsigned j = 0; j < TShape1; j++) {
                for (unsigned k = 0; k < TShape2; k++) {
                    if (!remainElm) {
                        vadd(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                             dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                             src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 + j * srcRawShape2 * srcRawShape3 +
                                 k * srcRawShape3,
                             repeatTime, 1, 1, 1, 8, 8, 8);
                    } else {
                        if (repeatTime == 1) {
                            set_vector_mask(0,
                                            (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                            vadd(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                     j * srcRawShape2 * srcRawShape3 + k * srcRawShape3,
                                 repeatTime, 1, 1, 1, 8, 8, 8);
                            set_vector_mask(-1, -1);
                        } else {
                            vadd(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                 src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                     j * srcRawShape2 * srcRawShape3 + k * srcRawShape3,
                                 repeatTime - 1, 1, 1, 1, 8, 8, 8);
                            set_vector_mask(0,
                                            (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                            vadd(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + (repeatTime - 1) * rptElm,
                                 dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + (repeatTime - 1) * rptElm,
                                 src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                     j * srcRawShape2 * srcRawShape3 + k * srcRawShape3 + (repeatTime - 1) * rptElm,
                                 1, 1, 1, 1, 8, 8, 8);
                            set_vector_mask(-1, -1);
                        }
                    }
                }
            }
        }
    } else if (axis == 1 || axis == 2) {
        for (unsigned i = 0; i < TShape0; i++) {
            Trowsumline_<T, TShape1, TShape2, TShape3, srcRawShape2, srcRawShape3, dstRawShape2, dstRawShape3,
                         axis - 1>(dst + i * dstRawShape1 * dstRawShape2 * dstRawShape3,
                                   src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3);
        }
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned dstRawShape0, unsigned dstRawShape1,
          unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tvcopy(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 8;

    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    // UB discontinuous
    if constexpr ((dstRawShape1 > TShape1 || src0RawShape1 > TShape1) &&
                  (TShape1 % blockSize == 0 && dstRawShape1 % blockSize == 0 && src0RawShape1 % blockSize == 0)) {
        srcRepeatStride = src0RawShape1 > TShape1 ? src0RawShape1 / blockSize : TShape1 / blockSize;
        dstRepeatStride = dstRawShape1 > TShape1 ? dstRawShape1 / blockSize : TShape1 / blockSize;

        set_vector_mask(
            static_cast<uint64_t>(
                (TShape1 > 64) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1 - 64)) - 1) : 0),
            static_cast<uint64_t>((TShape1 >= 64) ?
                                      0xffffffffffffffff :
                                      (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(TShape1)) - 1)));
        vcopy(dst, src, TShape0, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_vector_mask(-1, -1);
        return;
    }

    // UB continuous
    if constexpr (repeat < 1) {
        constexpr uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems);
        vcopy(dst, src, 1, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint8_t repeatLoop = static_cast<uint8_t>(repeat / REPEAT_MAX);
    uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
    for (uint8_t i = 0; i < repeatLoop; i++) {
        vcopy(dst + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T), src + i * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
              REPEAT_MAX, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    if (repeatMod != 0) {
        vcopy(dst + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T),
              src + repeatLoop * REPEAT_MAX * REPEAT_BYTE / sizeof(T), repeatMod, dstBlockStride, srcBlockStride,
              dstRepeatStride, srcRepeatStride);
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned dstRawShape0,
          unsigned dstRawShape1, unsigned src0RawShape0, unsigned src0RawShape1>
TILEOP void Tvcopy(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    constexpr uint32_t baseTileSize = TShape2 * TShape3;
    constexpr uint32_t dstRawSize = dstRawShape0 * dstRawShape1;
    constexpr uint32_t src0RawSize = src0RawShape0 * src0RawShape1;
    constexpr uint32_t alignDst = dstRawSize > baseTileSize ? dstRawSize : baseTileSize;
    constexpr uint32_t alignSrc0 = src0RawSize > baseTileSize ? src0RawSize : baseTileSize;
    // ub需要32B对齐
    static_assert(baseTileSize % blockSize == 0);
    static_assert(alignDst % blockSize == 0);
    static_assert(alignSrc0 % blockSize == 0);

    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            TileOp::Tvcopy<T, TShape2, TShape3, dstRawShape0, dstRawShape1, src0RawShape0, src0RawShape1>(dst, src);
            dst += alignDst;
            src += alignSrc0;
        }
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
          unsigned oriShape0, unsigned oriShape1, int axis, int offset, int isLargest>
TILEOP void BitSort(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    // 生成index数据,首先创建一个1~8的数组,之后扩展到TShape1,构成0~TShape1的index数组
    // pipe_barrier(PIPE_ALL); // 当前OP无法描述两条流水,UB复用场景存在问题,暂时按照pipe_all规避
    constexpr int32_t srcShape1Align = (oriShape1 + 31) / 32 * 32;
    __ubuf__ uint32_t* idx = (__ubuf__ uint32_t*)tmp;
    for (int32_t j = 0; j < oriShape1; j++) {
        *(idx + j) = j;
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);

    // 对于不满足32元素对齐场景,首先将src拷贝到dst的3*srcShape1位置
    if constexpr (oriShape1 < 32) {
        uint64_t srcShape1_Align_Block_Num = (oriShape1 * sizeof(float) + 31) / 32;
        uint64_t dstShape1_Block_Num = dstShape1 * sizeof(float) / 32;
        copy_ubuf_to_ubuf((__ubuf__ float*)tmp + srcShape1Align, (__ubuf__ void*)src, 0, oriShape0,
                          srcShape1_Align_Block_Num, 0, dstShape1_Block_Num - srcShape1_Align_Block_Num);
        pipe_barrier(PIPE_V);
        if constexpr (isLargest == 0) {
            set_mask_count();
            set_vector_mask(0, oriShape1);
            // 按照升序排列时,需要首先将数据乘以-1,同时不可以污染src
            vmuls((__ubuf__ float*)tmp + srcShape1Align, (__ubuf__ float*)tmp + srcShape1Align, -1.0f, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(-1, -1);
        }
        // 需要将尾块部分置为-inf，之后再排序
        // 计算duplicate的mask
        uint64_t mask = ~(((static_cast<uint64_t>(1)) << oriShape1) - 1);
        mask = mask & 0xFFFFFFFF;
        float FLOAT_MIN = -1.0e38f;
        set_mask_norm();
        set_vector_mask(0, mask);
        vector_dup(tmp + srcShape1Align, FLOAT_MIN, oriShape0, 1, 1, dstShape1 * sizeof(float) / 32, (int64_t)0);
        pipe_barrier(PIPE_V);
        for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
            vbitsort((__ubuf__ float*)dst + rowIdx * dstShape1, (__ubuf__ float*)tmp + srcShape1Align,
                     (__ubuf__ uint32_t*)idx, 1);
        }
        pipe_barrier(PIPE_V);
        set_vector_mask(-1, -1);
    }

    if constexpr (oriShape1 == 32) {
        for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
            // 32个数时，一次完成排序
            __ubuf__ float* srcData = reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1;
            __ubuf__ float* dstData = reinterpret_cast<__ubuf__ float*>(dst) + rowIdx * dstShape1;
            if constexpr (isLargest == 0) {
                set_mask_count();
                set_vector_mask(0, oriShape1);
                // 按照升序排列时,需要首先将数据乘以-1,同时不可以污染src
                srcData = reinterpret_cast<__ubuf__ float*>(tmp) + srcShape1Align;
                vmuls(srcData, reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1, -1.0f, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                set_mask_norm();
                set_vector_mask(-1, -1);
            }
            vbitsort(dstData, srcData, idx, 1);
            pipe_barrier(PIPE_V);
        }
    }

    if constexpr (oriShape1 > 32) {
        constexpr int32_t repeat_sort32 = oriShape1 / 32;
        constexpr int32_t tail_sort32 = oriShape1 % 32;
        for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
            __ubuf__ float* srcData = reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1;
            __ubuf__ float* dstData = reinterpret_cast<__ubuf__ float*>(dst) + rowIdx * dstShape1;
            if constexpr (isLargest == 0) {
                set_mask_count();
                set_vector_mask(0, oriShape1);
                // 按照升序排列时,需要首先将数据乘以-1,同时不可以污染src
                srcData = reinterpret_cast<__ubuf__ float*>(tmp) + srcShape1Align;
                vmuls(srcData, reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1, -1.0f, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                set_mask_norm();
                set_vector_mask(-1, -1);
            }
            // 首先逐32个数进行排序,需要补齐不对齐的部分
            if constexpr (tail_sort32 > 0) {
                // 非整块的时候,首先对尾部补充-inf
                float FLOAT_MIN = -1.0e38f;
                uint64_t mask = ~(((static_cast<uint64_t>(1)) << (64 - tail_sort32)) - 1);
                set_mask_norm();
                set_vector_mask(0, mask);
                vector_dup(srcData + repeat_sort32 * 32, FLOAT_MIN, 1, 1, 1, 8, (int64_t)0);
                pipe_barrier(PIPE_V);
                vbitsort(dstData, srcData, idx, repeat_sort32 + 1);
                pipe_barrier(PIPE_V);
                set_vector_mask(-1, -1);
            } else {
                // 整块时,直接进行逐32元素排序
                vbitsort(dstData, srcData, idx, repeat_sort32);
                pipe_barrier(PIPE_V);
            }
            pipe_barrier(PIPE_V);
        }
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
          unsigned oriShape0, unsigned oriShape1, int axis, int k, int mergeSize>
TILEOP void MrgSort(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp)
{
    constexpr int32_t kAlign = (k + 3) / 4 * 4; // k需要向32Bytes取整,否则最后搬运出问题
    constexpr int32_t totalNum = oriShape1 / 2;
    for (int rowIdx = 0; rowIdx < dstShape0; rowIdx++) {
        // 每4个合并,计算整块
        int32_t z = 32;
        for (; z * 4 <= totalNum; z *= 4) {
            __ubuf__ float* srcData = reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1;
            __ubuf__ float* dstData = reinterpret_cast<__ubuf__ float*>(src);
            uint64_t config = 0;
            uint32_t repeat_mrg = totalNum / (z * 4);
            config |= uint64_t(totalNum / (z * 4)); // Xt[7:0]: repeat time
            config |= (uint64_t(0b1111) << 8);      // Xt[11:8]: 4-bit mask signal
            config |= (uint64_t(0b0) << 12);        // Xt[12]: 1-enable input list exhausted suspension

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
            copy_ubuf_to_ubuf((__ubuf__ void*)srcData, (__ubuf__ void*)dstData, 0, z * 4 * repeat_mrg * 2 / 8, 1, 0, 0);
            pipe_barrier(PIPE_V);
        }
        // 合并尾块
        if (z < totalNum) {
            int32_t arrayCount = 0;
            int32_t mrgArray[15] = {0};
            int32_t tmpInner = totalNum;
            for (int32_t i = z; i >= 32; i /= 4) {
                int32_t count;
                for (count = 0; count < tmpInner / i; count++) {
                    mrgArray[arrayCount++] = i;
                }
                tmpInner -= count * i;
            }
            uint16_t mrgSortedLen = 0;
            for (int32_t i = 0; i < arrayCount - 1; ++i) {
                __ubuf__ float* srcData = reinterpret_cast<__ubuf__ float*>(src) + rowIdx * srcShape1;
                __ubuf__ float* dstData = reinterpret_cast<__ubuf__ float*>(src);
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
                copy_ubuf_to_ubuf((__ubuf__ void*)srcData, (__ubuf__ void*)dstData, 0,
                                  (tmpMrgSortedLen + tmpMrgArray) * 2 / 8, 1, 0, 0);
                pipe_barrier(PIPE_V);
            }
        }
        copy_ubuf_to_ubuf((__ubuf__ float*)dst + rowIdx * dstShape1, (__ubuf__ float*)src + rowIdx * srcShape1, 0,
                          kAlign / 4, 1, 0, 0);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, typename U, unsigned TShape0, unsigned TShape1, int k, int extractMode, int isLargest>
TILEOP void Extract(__ubuf__ T* dst, __ubuf__ U* src)
{
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * 2 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    constexpr uint8_t dstRepeatStride = 8;
    constexpr uint8_t srcRepeatStride = 8;
    // mode trans, extractMode == 0 取奇数位， extractMode == 1 取偶数位
    int patternMode = 1;
    if constexpr (extractMode == 1) {
        patternMode = 2;
    }
    __ubuf__ U* nullsrc1 = REPEAT_BYTE * sizeof(U) + src;
    if constexpr (repeat < 1) {
        constexpr uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems * 2);
        vreducev2((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src, (__ubuf__ uint32_t*)nullsrc1, 1, srcBlockStride,
                  patternMode, srcRepeatStride, 0);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    } else {
        uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
        if (repeatMod != 0) {
            constexpr uint64_t elems = TShape0 * TShape1;
            set_mask_norm();
            set_vector_mask(-1, -1);
            vreducev2((__ubuf__ uint32_t*)(dst), (__ubuf__ uint32_t*)(src), (__ubuf__ uint32_t*)nullsrc1, repeatMod,
                      srcBlockStride, patternMode, srcRepeatStride, 0);
            pipe_barrier(PIPE_V);
        }
    }

    if constexpr (extractMode == 0 && isLargest == 0) {
        // 按照升序排序时,对于value需要乘以-1,恢复原始值
        set_mask_count();
        set_vector_mask(0, TShape0 * TShape1);
        vmuls((__ubuf__ float*)dst, (__ubuf__ float*)dst, -1.0f, 1, 1, 1, 8, 8);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, typename idxT, unsigned xShape0, unsigned xShape1, unsigned idxShape0, unsigned idxShape1,
          int descending>
TILEOP void CompareAndSwap(__ubuf__ T* y0, __ubuf__ idxT* yIdx0, __ubuf__ T* y1, __ubuf__ idxT* yIdx1, __ubuf__ T* x0,
                           __ubuf__ idxT* idx0, __ubuf__ T* x1, __ubuf__ idxT* idx1)
{
    // UB reuse: y0 = x0, yIdx0 = idx0
    constexpr uint32_t oneLength = 256 / sizeof(T);
    constexpr uint32_t repeat = xShape1 / oneLength;
    set_mask_norm();
    set_vector_mask(-1, -1);
    for (uint32_t offset = 0; offset < xShape1; offset += oneLength) {
        if (descending == 1) {
            // src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
            // src1RepeatStride
            vcmp_ge(x0 + offset, x1 + offset, 1, 1, 1, 1, 8, 8, 8);
        } else {
            vcmp_le(x0 + offset, x1 + offset, 1, 1, 1, 1, 8, 8, 8);
        }
        vsel(y1 + offset, x1 + offset, x0 + offset, 1, 1, 1, 1, 8, 8, 8, 0); // mode = 0x0
        vsel((__ubuf__ float*)yIdx1 + offset, (__ubuf__ float*)idx1 + offset, (__ubuf__ float*)idx0 + offset, 1, 1, 1,
             1, 8, 8, 8, 0); // mode = 0x0
        vsel((__ubuf__ float*)yIdx0 + offset, (__ubuf__ float*)idx0 + offset, (__ubuf__ float*)idx1 + offset, 1, 1, 1,
             1, 8, 8, 8, 0);                                                 // mode = 0x0
        vsel(y0 + offset, x0 + offset, x1 + offset, 1, 1, 1, 1, 8, 8, 8, 0); // mode = 0x0
    }
}

template <typename T, typename idxT, unsigned shape>
TILEOP void BitSortAll(__ubuf__ T* y, __ubuf__ idxT* yIdx, __ubuf__ T* tmp, __ubuf__ T* x, __ubuf__ idxT* xIdx)
{
    constexpr uint32_t bitSortLength = 32;
    constexpr uint32_t oneLength = 256 / sizeof(T);
    set_mask_norm();
    set_vector_mask(-1, -1);
    // x == xIdx == y == yIdx == tmp / 4
    constexpr uint32_t repeat = shape / bitSortLength;
    constexpr uint32_t len255 = 255 * bitSortLength;
    if constexpr (shape <= 255 * bitSortLength) {
        vbitsort(tmp, x, (__ubuf__ uint32_t*)xIdx, repeat);
        pipe_barrier(PIPE_V);
        vreducev2((__ubuf__ uint32_t*)y, (__ubuf__ uint32_t*)tmp, (__ubuf__ uint32_t*)tmp, repeat, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vreducev2((__ubuf__ uint32_t*)yIdx, (__ubuf__ uint32_t*)tmp, (__ubuf__ uint32_t*)tmp, repeat, 1, 2, 8, 0);
        pipe_barrier(PIPE_V);
    } else { // shape = 8K, repeat = 256
        vbitsort(tmp, x, (__ubuf__ uint32_t*)xIdx, 255);
        pipe_barrier(PIPE_V);
        vreducev2((__ubuf__ uint32_t*)y, (__ubuf__ uint32_t*)tmp, (__ubuf__ uint32_t*)tmp, 255, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vreducev2((__ubuf__ uint32_t*)yIdx, (__ubuf__ uint32_t*)tmp, (__ubuf__ uint32_t*)tmp, 255, 1, 2, 8, 0);
        pipe_barrier(PIPE_V);
        vbitsort(tmp, x + len255, (__ubuf__ uint32_t*)xIdx + len255, repeat - 255);
        pipe_barrier(PIPE_V);
        vreducev2((__ubuf__ uint32_t*)y + len255, (__ubuf__ uint32_t*)tmp, (__ubuf__ uint32_t*)tmp, repeat - 255, 1, 1,
                  8, 0);
        pipe_barrier(PIPE_V);
        vreducev2((__ubuf__ uint32_t*)yIdx + len255, (__ubuf__ uint32_t*)tmp, (__ubuf__ uint32_t*)tmp, repeat - 255, 1,
                  2, 8, 0);
    }
}

template <typename T, typename idxT, unsigned shape>
TILEOP void CompSwap32(__ubuf__ T* y, __ubuf__ idxT* yIdx, __ubuf__ T* tmp, __ubuf__ T* x, __ubuf__ idxT* xIdx)
{
    constexpr uint32_t mergeLength = 64;
    constexpr uint32_t halfLength = 32;
    constexpr uint32_t repeatOut = shape / mergeLength;
    constexpr uint64_t mask = (static_cast<uint64_t>(1) << 32) - 1;
    set_mask_norm();
    set_vector_mask(0, mask);
    for (uint32_t i = 0; i < repeatOut; i++) {
        uint32_t start = i * mergeLength;
        __ubuf__ T* x0 = x + start;
        __ubuf__ T* x1 = x0 + halfLength;
        __ubuf__ idxT* xIdx0 = xIdx + start;
        __ubuf__ idxT* xIdx1 = xIdx0 + halfLength;
        __ubuf__ T* tmpX = tmp + start / 2;
        __ubuf__ idxT* tmpIdx = (__ubuf__ idxT*)tmpX + shape / 2;
        __ubuf__ T* y0 = y + start;
        __ubuf__ T* y1 = y0 + halfLength;
        __ubuf__ idxT* yIdx0 = yIdx + start;
        __ubuf__ idxT* yIdx1 = yIdx0 + halfLength;
        vcmp_ge(x0, x1, 1, 1, 1, 1, 8, 8, 8);
        vsel(tmpX, x0, x1, 1, 1, 1, 1, 8, 8, 8, 0);
        vsel(y1, x1, x0, 1, 1, 1, 1, 8, 8, 8, 0);
        vsel((__ubuf__ float*)tmpIdx, (__ubuf__ float*)xIdx0, (__ubuf__ float*)xIdx1, 1, 1, 1, 1, 8, 8, 8, 0);
        vsel((__ubuf__ float*)yIdx1, (__ubuf__ float*)xIdx1, (__ubuf__ float*)xIdx0, 1, 1, 1, 1, 8, 8, 8, 0);
    }
    // copy every halfLength (1 burst) from tmp to x0 & xIdx0
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(-1, -1);
    // dst src sid nBurst lenBurst srcStride dstStride
    copy_ubuf_to_ubuf((__ubuf__ void*)x, (__ubuf__ void*)tmp, 0, shape / mergeLength, halfLength / 8, 0,
                      halfLength / 8);
    pipe_barrier(PIPE_V);
    copy_ubuf_to_ubuf((__ubuf__ void*)xIdx, (__ubuf__ void*)(tmp + shape / 2), 0, shape / mergeLength, halfLength / 8,
                      0, halfLength / 8);
    pipe_barrier(PIPE_V);
}

template <typename T, typename idxT, unsigned shape, unsigned mergeLength>
TILEOP void CompSwapCommon(__ubuf__ T* y, __ubuf__ idxT* yIdx, __ubuf__ T* tmp, __ubuf__ T* x, __ubuf__ idxT* xIdx)
{
    constexpr uint32_t oneLength = 256 / sizeof(T);
    constexpr uint32_t halfLength = mergeLength / 2;
    // comp&swap mergeLength each time
    constexpr uint32_t repeatIn = halfLength / oneLength;
    constexpr uint32_t repeatOut = shape / mergeLength;
    set_mask_norm();
    set_vector_mask(-1, -1);
    for (uint32_t i = 0; i < repeatOut; i++) {
        uint32_t start = i * mergeLength;
        __ubuf__ T* x0 = x + start;
        __ubuf__ T* x1 = x0 + halfLength;
        __ubuf__ idxT* xIdx0 = xIdx + start;
        __ubuf__ idxT* xIdx1 = xIdx0 + halfLength;
        __ubuf__ T* tmpX = tmp + start / 2;
        __ubuf__ idxT* tmpIdx = (__ubuf__ idxT*)tmpX + shape / 2;
        __ubuf__ T* y0 = y + start;
        __ubuf__ T* y1 = y0 + halfLength;
        __ubuf__ idxT* yIdx0 = yIdx + start;
        __ubuf__ idxT* yIdx1 = yIdx0 + halfLength;
        // within one mergeLength
        for (uint32_t j = 0; j < repeatIn; j++) {
            uint32_t offset = j * oneLength;
            vcmp_ge(x0 + offset, x1 + offset, 1, 1, 1, 1, 8, 8, 8);
            vsel(tmpX + offset, x0 + offset, x1 + offset, 1, 1, 1, 1, 8, 8, 8, 0);
            vsel(y1 + offset, x1 + offset, x0 + offset, 1, 1, 1, 1, 8, 8, 8, 0);
            vsel((__ubuf__ float*)tmpIdx + offset, (__ubuf__ float*)xIdx0 + offset, (__ubuf__ float*)xIdx1 + offset, 1,
                 1, 1, 1, 8, 8, 8, 0);
            vsel((__ubuf__ float*)yIdx1 + offset, (__ubuf__ float*)xIdx1 + offset, (__ubuf__ float*)xIdx0 + offset, 1,
                 1, 1, 1, 8, 8, 8, 0);
        }
    }
    // copy every halfLength (1 burst) from tmp to x0 & xIdx0
    pipe_barrier(PIPE_V);
    // dst src sid nBurst lenBurst srcStride dstStride
    copy_ubuf_to_ubuf((__ubuf__ void*)x, (__ubuf__ void*)tmp, 0, shape / mergeLength, halfLength / 8, 0,
                      halfLength / 8);
    pipe_barrier(PIPE_V);
    copy_ubuf_to_ubuf((__ubuf__ void*)xIdx, (__ubuf__ void*)(tmp + shape / 2), 0, shape / mergeLength, halfLength / 8,
                      0, halfLength / 8);
    pipe_barrier(PIPE_V);
}

template <typename T, typename idxT, unsigned shape, unsigned mergeLength>
TILEOP void CompSwapSteps(__ubuf__ T* y, __ubuf__ idxT* yIdx, __ubuf__ T* tmp, __ubuf__ T* x, __ubuf__ idxT* xIdx)
{
    constexpr uint32_t halfLength = mergeLength / 2;
    constexpr uint32_t oneLength = 256 / sizeof(T);

    if constexpr (halfLength >= oneLength) {
        CompSwapCommon<T, idxT, shape, mergeLength>(y, yIdx, tmp, x, xIdx);
    } else {
        CompSwap32<T, idxT, shape>(y, yIdx, tmp, x, xIdx);
    }

    if constexpr (halfLength > 32) {
        CompSwapSteps<T, idxT, shape, halfLength>(y, yIdx, tmp, x, xIdx);
    }
}

template <typename T, unsigned shape>
TILEOP void MulsMinusOne(__ubuf__ T* src)
{
    constexpr uint32_t oneLength = 256 / sizeof(T);
    constexpr uint32_t repeat = shape / oneLength;
    constexpr uint32_t len255 = 255 * oneLength;
    set_mask_norm();
    set_vector_mask(-1, -1);
    if constexpr (repeat <= 255) {
        vmuls(src, src, -1.0f, repeat, 1, 1, 8, 8);
    } else { // shape = 16K, repeat = 256
        vmuls(src, src, -1.0f, 255, 1, 1, 8, 8);
        vmuls(src + len255, src + len255, -1.0f, repeat - 255, 1, 1, 8, 8);
    }
}

template <typename T, typename idxT, unsigned xShape0, unsigned xShape1, unsigned idxShape0, unsigned idxShape1,
          int descending>
TILEOP void SortWithIndex(__ubuf__ T* y, __ubuf__ idxT* yIdx, __ubuf__ T* tmp, __ubuf__ T* x, __ubuf__ idxT* xIdx)
{
    // xShape1 <= 8K, y == yIdx == x == xIdx == tmp / 4
    // Step 0: muls -1
    if constexpr (descending == 0) {
        MulsMinusOne<T, xShape1>(x);
        pipe_barrier(PIPE_V);
    }
    // Step 1: vbs
    constexpr uint32_t bitSortLength = 32;
    constexpr uint32_t repeat = xShape1 / bitSortLength;
    constexpr uint32_t len255 = 255 * bitSortLength;
    if constexpr (repeat <= 255) {
        vbitsort(tmp, x, (__ubuf__ uint32_t*)xIdx, repeat);
    } else {
        vbitsort(tmp, x, (__ubuf__ uint32_t*)xIdx, 255);
        vbitsort(tmp + len255 * 2, x + len255, (__ubuf__ uint32_t*)xIdx + len255, repeat - 255);
    }
    pipe_barrier(PIPE_V);
    // Step 2: vms4
    __ubuf__ float* src = tmp + xShape1 * 2;
    __ubuf__ float* dst = tmp;
    uint32_t z = 32;
    for (; z * 4 <= xShape1; z *= 4) {
        __ubuf__ float* swap = src;
        src = dst;
        dst = swap;
        uint64_t config = 0;
        uint32_t repeat_mrg = xShape1 / z / 4;
        config |= uint64_t(repeat_mrg);    // Xt[7:0]: repeat time
        config |= (uint64_t(0b1111) << 8); // Xt[11:8]: 4-bit mask signal
        config |= (uint64_t(0b0) << 12);   // Xt[12]: 1-enable input list exhausted suspension

        // 每次计算的数据
        uint64_t lengthData = 0;
        lengthData |= (uint64_t(z));
        lengthData |= (uint64_t(z) << 16);
        lengthData |= (uint64_t(z) << 32);
        lengthData |= (uint64_t(z) << 48);

        __ubuf__ float* addr[4] = {(__ubuf__ float*)(src), (__ubuf__ float*)(src + z * 2),
                                   (__ubuf__ float*)(src + z * 4), (__ubuf__ float*)(src + z * 6)};
        pipe_barrier(PIPE_V);
        vmrgsort4(dst, addr, lengthData, config);
        pipe_barrier(PIPE_V);
    }
    if (z * 2 == xShape1) {
        __ubuf__ float* swap = src;
        src = dst;
        dst = swap;
        uint64_t config = 0;
        uint32_t repeat_mrg = 1;
        config |= uint64_t(repeat_mrg);  // Xt[7:0]: repeat time
        config |= (uint64_t(0b11) << 8); // Xt[11:8]: 4-bit mask signal
        config |= (uint64_t(0b0) << 12); // Xt[12]: 1-enable input list exhausted suspension

        // 每次计算的数据
        uint64_t lengthData = 0;
        lengthData |= (uint64_t(z));
        lengthData |= (uint64_t(z) << 16);

        __ubuf__ float* addr[4] = {(__ubuf__ float*)(src), (__ubuf__ float*)(src + z * 2), (__ubuf__ float*)(0),
                                   (__ubuf__ float*)(0)};
        pipe_barrier(PIPE_V);
        vmrgsort4(dst, addr, lengthData, config);
        pipe_barrier(PIPE_V);
    }
    // Step 3: extract
    vreducev2((__ubuf__ uint32_t*)y, (__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)dst, xShape1 / 32, 1, 1, 8, 0);
    vreducev2((__ubuf__ uint32_t*)yIdx, (__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)dst, xShape1 / 32, 1, 2, 8, 0);
    pipe_barrier(PIPE_V);
    // Step 4: muls -1
    if constexpr (descending == 0) {
        MulsMinusOne<T, xShape1>(y);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, typename idxT, unsigned xShape1>
TILEOP void GenSortIndex(__ubuf__ idxT* idx, __ubuf__ T* tmp, int idxStart)
{
    __ubuf__ float* tmp1 = (__ubuf__ float*)(tmp + xShape1 / 2); // need 64*4 size

    set_mask_count();
    set_vector_mask(0, 8);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        vector_dup(tmp + i * 8, (float)float(i) * 0.125f, 1, 1, 1, 1, (int64_t)0);
    }
    pipe_barrier(PIPE_V);
    set_vector_mask(0, 64);
    vcgadd((__ubuf__ float*)idx, tmp, 1, 1, 1, 8); // 0-7.0
    pipe_barrier(PIPE_V);
    set_vector_mask(0, 8);
    vmuls(tmp1, (__ubuf__ float*)idx, 8.0f, 1, 1, 1, 8, 8); // 0,8,16,...,56  -- 8 elements
    vmuls(tmp, (__ubuf__ float*)idx, 64.0f, 1, 1, 1, 8, 8); // 0,64,128,...,448 -- 8 elements
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    pipe_barrier(PIPE_V);
    vbrcb((__ubuf__ uint32_t*)(tmp), (__ubuf__ uint32_t*)(tmp), 1, 8,
          1); //[0..0],[64..64],[128..128],...[448..448] -- 64 elements
    pipe_barrier(PIPE_V);
    vadd(tmp1, (__ubuf__ float*)tmp1, tmp, 1, 1, 0, 1, 8, 0, 8); //[0, 8, 16,...504]
    pipe_barrier(PIPE_V);

    vbrcb((__ubuf__ uint32_t*)(tmp), (__ubuf__ uint32_t*)(tmp1), 1, 8,
          8); //[0..0],[8..8], [16..16],...[504..504], -- 64*8 = 512 elements
    pipe_barrier(PIPE_V);
    vadd((__ubuf__ float*)idx, (__ubuf__ float*)idx, tmp, 8, 1, 0, 1, 8, 0, 8);
    pipe_barrier(PIPE_V);

    vconv_f322s32r((__ubuf__ int32_t*)idx, (__ubuf__ float*)idx, 8, 1, 1, 8, 8); //[0....511]
    pipe_barrier(PIPE_V);

    vadds((__ubuf__ int32_t*)idx, (__ubuf__ int32_t*)idx, idxStart * xShape1, 8, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

#pragma unroll
    for (int i = 1; i < xShape1 / 512; i++) {
        vadds((__ubuf__ int32_t*)(idx + 512 * i), (__ubuf__ int32_t*)idx, 512 * i, 8, 1, 1, 8, 8);
    }

    pipe_barrier(PIPE_V);
}

template <typename T, typename idxT, unsigned xShape0, unsigned xShape1, unsigned idxShape0, unsigned idxShape1,
          int descending, int idxStart>
TILEOP void Sort(__ubuf__ T* y, __ubuf__ idxT* yIdx, __ubuf__ T* tmp, __ubuf__ T* x)
{
    // index init
    __ubuf__ idxT* xIdx = yIdx;
    GenSortIndex<T, idxT, xShape1>(xIdx, tmp, idxStart);
    SortWithIndex<T, idxT, xShape0, xShape1, idxShape0, idxShape1, descending>(y, yIdx, tmp, x, xIdx);
}

template <typename T, typename idxT, unsigned xShape0, unsigned xShape1, unsigned idxShape0, unsigned idxShape1,
          int fullSort, int descending>
TILEOP void Merge(__ubuf__ T* y, __ubuf__ idxT* yIdx, __ubuf__ T* tmp, __ubuf__ T* x, __ubuf__ idxT* xIdx)
{
    // ideally, x == xIdx == y == yIdx == tmp
    if constexpr (fullSort == 1) { // sort with index
        SortWithIndex<T, idxT, xShape0, xShape1, idxShape0, idxShape1, descending>(y, yIdx, tmp, x, xIdx);
        return;
    }

    if constexpr (descending == 0) { // ascending
        MulsMinusOne<T, xShape1>(x);
        pipe_barrier(PIPE_V);
    }

    // compswap till each 32 group sorted(intra), then bitsort every group
    CompSwapSteps<T, idxT, xShape1, xShape1>(y, yIdx, tmp, x, xIdx);
    pipe_barrier(PIPE_V);
    BitSortAll<T, idxT, xShape1>(y, yIdx, tmp, x, xIdx);
    pipe_barrier(PIPE_V);

    if constexpr (descending == 0) { // ascending recover
        MulsMinusOne<T, xShape1>(y);
    }
}

template <typename T, unsigned xShape0, unsigned xShape1, int mergeSize>
TILEOP void TopKMerge(__ubuf__ T* y, __ubuf__ T* x)
{
    // x == y == xShape1 x 2
    __ubuf__ float* src = y;
    __ubuf__ float* dst = x;
    uint32_t z = mergeSize;
    for (; z * 4 <= xShape1; z *= 4) {
        __ubuf__ float* swap = src;
        src = dst;
        dst = swap;
        uint64_t config = 0;
        uint32_t repeat_mrg = xShape1 / z / 4;
        config |= uint64_t(repeat_mrg);    // Xt[7:0]: repeat time
        config |= (uint64_t(0b1111) << 8); // Xt[11:8]: 4-bit mask signal
        config |= (uint64_t(0b0) << 12);   // Xt[12]: 1-enable input list exhausted suspension

        // 每次计算的数据
        uint64_t lengthData = 0;
        lengthData |= (uint64_t(z));
        lengthData |= (uint64_t(z) << 16);
        lengthData |= (uint64_t(z) << 32);
        lengthData |= (uint64_t(z) << 48);

        __ubuf__ float* addr[4] = {(__ubuf__ float*)(src), (__ubuf__ float*)(src + z * 2),
                                   (__ubuf__ float*)(src + z * 4), (__ubuf__ float*)(src + z * 6)};
        pipe_barrier(PIPE_V);
        vmrgsort4(dst, addr, lengthData, config);
        pipe_barrier(PIPE_V);
    }
    if (z * 2 == xShape1) {
        __ubuf__ float* swap = src;
        src = dst;
        dst = swap;
        uint64_t config = 0;
        uint32_t repeat_mrg = 1;
        config |= uint64_t(repeat_mrg);  // Xt[7:0]: repeat time
        config |= (uint64_t(0b11) << 8); // Xt[11:8]: 4-bit mask signal
        config |= (uint64_t(0b0) << 12); // Xt[12]: 1-enable input list exhausted suspension

        // 每次计算的数据
        uint64_t lengthData = 0;
        lengthData |= (uint64_t(z));
        lengthData |= (uint64_t(z) << 16);

        __ubuf__ float* addr[4] = {(__ubuf__ float*)(src), (__ubuf__ float*)(src + z * 2), (__ubuf__ float*)(0),
                                   (__ubuf__ float*)(0)};
        pipe_barrier(PIPE_V);
        vmrgsort4(dst, addr, lengthData, config);
        pipe_barrier(PIPE_V);
    }
    if (dst != y) {
        copy_ubuf_to_ubuf((__ubuf__ void*)y, (__ubuf__ void*)dst, 0, xShape1 * 2 / 8, 1, 0, 0);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, unsigned xShape0, unsigned xShape1>
TILEOP void TopKSortWithIndex(__ubuf__ T* y, __ubuf__ T* tmp, __ubuf__ T* x)
{
    // idx stored at y
    constexpr uint32_t bitSortLength = 32;
    constexpr uint32_t repeat = xShape1 / bitSortLength;
    if constexpr (repeat <= 255) {
        vbitsort(tmp, x, (__ubuf__ uint32_t*)y, repeat);
    } else {
        vbitsort(tmp, x, (__ubuf__ uint32_t*)y, repeat / 2);
        vbitsort(tmp + xShape1, x + xShape1 / 2, (__ubuf__ uint32_t*)y + xShape1 / 2, repeat / 2);
    }
    pipe_barrier(PIPE_V);
    // vms4
    TopKMerge<T, xShape0, xShape1, 32>(y, tmp);
}

template <typename T, unsigned xShape0, unsigned xShape1, int idxStart>
TILEOP void TopKSort(__ubuf__ T* y, __ubuf__ T* tmp, __ubuf__ T* x)
{
    // x x 2 = y = tmp == xShape1 x 2
    GenSortIndex<T, T, xShape1>((__ubuf__ T*)y, tmp, idxStart);
    TopKSortWithIndex<T, xShape0, xShape1>(y, tmp, x);
}

template <typename U, typename T, unsigned yShape0, unsigned yShape1, unsigned xShape0, unsigned xShape1, int isIndex,
          int k>
TILEOP void TopKExtract(__ubuf__ U* y, __ubuf__ T* x)
{
    // x = xShape1 x 2, y = yShape1
    if constexpr (isIndex == 0) {
        vreducev2((__ubuf__ uint32_t*)y, (__ubuf__ uint32_t*)x, (__ubuf__ uint32_t*)x, k / 32, 1, 1, 8, 0);
    } else {
        vreducev2((__ubuf__ uint32_t*)y, (__ubuf__ uint32_t*)x, (__ubuf__ uint32_t*)x, k / 32, 1, 2, 8, 0);
    }
    pipe_barrier(PIPE_V);
}

template <typename T, unsigned T0, unsigned T1>
TILEOP void Tbrcb_(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr unsigned brcPerRepeat = 8;
    if constexpr (T0 != 1) {
        constexpr unsigned repeatNumT0 = (T0 + brcPerRepeat - 1) / brcPerRepeat;
        vbrcb((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src, 1, 8, repeatNumT0);
        return;
    }

    if constexpr (T1 != 1) {
        constexpr unsigned repeatNumT1 = (T0 + brcPerRepeat - 1) / brcPerRepeat;
        vbrcb((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src, 1, 8, repeatNumT1);
        return;
    }
}

// dim4
template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned DS1, unsigned DS2, unsigned DS3,
          unsigned SS1, unsigned SS2, unsigned SS3>
TILEOP void Tbrcb_(__ubuf__ T* dst, __ubuf__ T* src)
{
    static_assert(DS3 * sizeof(T) == BLOCK_SIZE);
    static_assert(DS2 % BLOCK_NUM_ONE_REPEAT == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < T1; j++) {
            Tbrcb_<T, T2, T3>(dst_, src_);
            dst_ += DS2 * DS3;
            src_ += SS2 * SS3;
        }
        dst += DS1 * DS2 * DS3;
        src += SS1 * SS2 * SS3;
    }
}
} // namespace TileOp

#endif
