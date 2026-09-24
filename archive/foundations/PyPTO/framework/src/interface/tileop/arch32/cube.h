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
 * \file cube.h
 * \brief
 */

#ifndef __LOGICALTENSOR_TILEOP_CUBE__
#define __LOGICALTENSOR_TILEOP_CUBE__

#include "tileop_common.h"

#include <type_traits>

namespace TileOp {
// cube intrins
// CUBE
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1, unsigned GmShape0, unsigned GmShape1,
          unsigned curH, unsigned curW>
TILEOP void L1CopyInNZ2NZ(__cbuf__ L1T* dst, __gm__ GMT* src, __gm__ GMT* oriSrc, int reserved)
{
    int64_t offsetElem = (int64_t)(src - oriSrc);
    auto inputC0Size = 32 / sizeof(L1T);
    // 计算在那个Batch块中;
    auto batchSize = curH * curW;
    auto batchIndex = offsetElem / batchSize;
    auto inputOffset0 = (offsetElem - batchIndex * batchSize) / curW;
    auto inputOffset1 = (offsetElem - batchIndex * batchSize) % curW;
    // NZ的offset转换
    auto offsetWithNZ = batchIndex * batchSize + (inputOffset1 * curH) + inputOffset0 * inputC0Size;
    int32_t C0 = 32 / sizeof(GMT);
    uint16_t nBurst = TShape1 / C0;
    uint16_t lenBurst = TShape0 * C0 * sizeof(GMT) / 32;
    uint16_t srcStride = (curH - TShape0) * C0 * sizeof(GMT) / 32;
    uint16_t dstStride = 0;
    copy_gm_to_cbuf(dst, oriSrc + offsetWithNZ, 0 /*sid*/, nBurst, lenBurst, srcStride, dstStride, PAD_NONE);
}

template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1, unsigned GmShape0, unsigned GmShape1>
TILEOP void L1CopyIn(__cbuf__ L1T* dst, __gm__ GMT* src, int reserved)
{ // ND2NZ
    constexpr uint16_t ndNum = 1;
    constexpr uint16_t nValue = TShape0;      // n
    constexpr uint16_t dValue = TShape1;      // d
    constexpr uint16_t srcNdMatrixStride = 0; //
    constexpr uint16_t srcDValue = GmShape1;  // D
    auto c0Size = 32 / sizeof(GMT);
    constexpr uint16_t dstNzC0Stride = TShape0; // n
    constexpr uint16_t dstNzNStride = 1;
    constexpr uint16_t dstNzMatrixStride = 1;
    if constexpr (std::is_same<GMT, int8_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ L1T*)dst, (__gm__ GMT*)src, 0 /*sid*/, ndNum, nValue, dValue,
                                       srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, half>::value || std::is_same<GMT, bfloat16_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ L1T*)dst, (__gm__ GMT*)src, 0 /*sid*/, ndNum, nValue, dValue,
                                        srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, float>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ L1T*)dst, (__gm__ GMT*)src, 0 /*sid*/, ndNum, nValue, dValue,
                                         srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
}

template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1, unsigned GmOffset0, unsigned GmOffset1,
          unsigned GmShape0, unsigned GmShape1>
TILEOP void L1CopyIn(__cbuf__ L1T* dst, __gm__ GMT* src, int reserved)
{ // ND2NZ
    constexpr uint16_t ndNum = 1;
    constexpr uint16_t nValue = (GmShape0 - GmOffset0) < TShape0 ? (GmShape0 - GmOffset0) : TShape0; // n
    constexpr uint16_t dValue = (GmShape1 - GmOffset1) < TShape1 ? (GmShape1 - GmOffset1) : TShape1; // n
    constexpr uint16_t srcNdMatrixStride = 0;                                                        //
    constexpr uint16_t srcDValue = GmShape1;                                                         // D
    auto c0Size = 32 / sizeof(GMT);
    constexpr uint16_t dstNzC0Stride = TShape0; // n
    constexpr uint16_t dstNzNStride = 1;
    constexpr uint16_t dstNzMatrixStride = 1;
    if constexpr (std::is_same<GMT, int8_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ L1T*)dst, (__gm__ GMT*)src, 0 /*sid*/, ndNum, nValue, dValue,
                                       srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, half>::value || std::is_same<GMT, bfloat16_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ L1T*)dst, (__gm__ GMT*)src, 0 /*sid*/, ndNum, nValue, dValue,
                                        srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, float>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ L1T*)dst, (__gm__ GMT*)src, 0 /*sid*/, ndNum, nValue, dValue,
                                         srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
}

// L1 spill out scene
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1, unsigned GmShape0, unsigned GmShape1>
TILEOP void L1CopyOutND(__gm__ GMT* dst, __cbuf__ L1T* src, int reserved)
{
    uint16_t nBurst = TShape0;
    uint16_t lenBurst = TShape1 * sizeof(GMT) / 32;
    uint16_t srcStride = 0;
    uint16_t dstStride = (GmShape1 - TShape1) * sizeof(GMT) / 32;

    if (lenBurst == 0) {
        nBurst = 1;
        lenBurst = TShape0 * TShape1 * sizeof(GMT);
        if (lenBurst == 0) {
            lenBurst = 1;
        }
        srcStride = 0;
        dstStride = 0;
    }
    copy_cbuf_to_gm(dst, src, 0 /*sid*/, nBurst, lenBurst, srcStride, dstStride);
}

// Currently 'L1CopyOut' is ONLY used when spilling occurred, and does NOT need data format conversion. the impl
// redirect this function to L1CopyOutND directly.
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1, unsigned GmShape0, unsigned GmShape1>
TILEOP void L1CopyOut(__gm__ GMT* dst, __cbuf__ L1T* src, int reserved)
{
    L1CopyOutND<GMT, L1T, TShape0, TShape1, GmShape0, GmShape1>(dst, src, reserved);
}

// L1 spill out scene
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1, unsigned GmShape0, unsigned GmShape1>
TILEOP void L1CopyInND(__cbuf__ L1T* dst, __gm__ GMT* src, int reserved)
{
    uint16_t nBurst = TShape0;
    uint16_t lenBurst = TShape1 * sizeof(GMT) / 32;
    uint16_t srcStride = 0;
    uint16_t dstStride = (GmShape1 - TShape1) * sizeof(GMT) / 32;

    if (lenBurst == 0) {
        nBurst = 1;
        lenBurst = TShape0 * TShape1 * sizeof(GMT);
        if (lenBurst == 0) {
            lenBurst = 1;
        }
        srcStride = 0;
        dstStride = 0;
    }
    copy_gm_to_cbuf(dst, src, 0 /*sid*/, nBurst, lenBurst, srcStride, dstStride, PAD_NONE);
}

// Nz2Zz
template <typename T, unsigned dstM, unsigned dstK, unsigned Offset0, unsigned Offset1, unsigned srcM, unsigned srcK>
TILEOP void L1ToL0A(__ca__ T* dst, __cbuf__ T* src)
{
    int64_t frac_num = 32 / sizeof(T);
    int64_t m_frac = dstM / 16; //
    uint8_t repeat = dstK / frac_num;
    uint16_t srcStride = srcM / 16; // stride

    uint16_t dstStride = 0; // gap

    for (int64_t m_idx = 0; m_idx < m_frac; ++m_idx) {
        load_cbuf_to_ca(dst + m_idx * 16 * dstK, src + m_idx * 16 * frac_num + (Offset0 * frac_num + Offset1 * srcM), 0,
                        repeat, srcStride, dstStride, 0, 0, inc);
    }
}

// Nz2Zn
template <typename T, unsigned dstK, unsigned dstN, unsigned Offset0, unsigned Offset1, unsigned srcK, unsigned srcN>
TILEOP void L1ToL0B(__cb__ T* dst, __cbuf__ T* src)
{
    auto nBlockSize = 32;
    if constexpr (std::is_same<T, int8_t>::value) {
        for (auto index = 0; index < dstN / nBlockSize; ++index) {
            auto repeatTimes = dstK / (nBlockSize);
            auto srcStride = 1;
            auto dstGap = (nBlockSize * dstN - nBlockSize * 16) / (16 * nBlockSize);
            auto dstFracGap = 0;
            load_cbuf_to_cb_transpose(dst + index * nBlockSize * nBlockSize,
                                      src + Offset0 * nBlockSize + Offset1 * srcK + index * nBlockSize * srcK, 0,
                                      repeatTimes, srcStride, dstGap, inc, dstFracGap);
        }
        return;
    }
    if constexpr (std::is_same<T, float>::value) {
        nBlockSize = 16;
        for (auto index = 0; index < srcK / nBlockSize; ++index) {
            auto repeatTimes = srcN / nBlockSize;
            auto srcStride = (nBlockSize * srcK) / (nBlockSize * nBlockSize);
            auto dstGap = 1;
            auto dstFracGap = 0;
            load_cbuf_to_cb_transpose(dst + index * nBlockSize * srcN, src + index * 8 * nBlockSize, 0, repeatTimes,
                                      srcStride, dstGap, inc, dstFracGap);
        }
        return;
    }
    // L1 n1k1k0no   -> l0b  k1n1n0k0
    int64_t frac_num = 32 / sizeof(T);
    int64_t k_frac = dstK / frac_num; // B32
    uint8_t repeat = dstN / 16;       //
    uint16_t srcStride = srcK / frac_num;
    uint16_t dstStride = 0; // gap;

    for (int64_t k_idx = 0; k_idx < k_frac; ++k_idx) {
        load_cbuf_to_cb(dst + k_idx * frac_num * dstN, src + k_idx * 16 * frac_num + (Offset0 * 16 + Offset1 * srcK), 0,
                        repeat, srcStride, dstStride, 0, 1, inc);
    }
}

// Nz2Zz
template <typename T, unsigned dstK, unsigned dstN, unsigned Offset0, unsigned Offset1, unsigned srcN, unsigned srcK>
TILEOP void L1ToL0Bt(__cb__ T* dst, __cbuf__ T* src)
{
    int64_t frac_num = 32 / sizeof(T);

    if constexpr (dstN == srcN) {
        constexpr uint8_t repeat = dstN / (32 / sizeof(T)) * dstK / 16;
        constexpr uint16_t srcStride = 1;
        constexpr uint16_t dstStride = 0;

        load_cbuf_to_cb(dst, src + (Offset0 * frac_num + Offset1 * srcN), (uint16_t)0, repeat, srcStride, dstStride,
                        (uint8_t)0, (bool)0, (addr_cal_mode_t)0);
    } else {
        int64_t k_frac = dstK / frac_num;
        constexpr uint8_t repeat = dstN / 16;
        constexpr uint16_t srcStride = 1;
        constexpr uint16_t dstStride = 0;

        for (int64_t k_idx = 0; k_idx < k_frac; ++k_idx) {
            load_cbuf_to_cb(dst + k_idx * frac_num * dstN,
                            src + k_idx * srcN * frac_num + (Offset0 * frac_num + Offset1 * srcN), (uint16_t)0, repeat,
                            srcStride, dstStride, (uint8_t)0, (bool)0, (addr_cal_mode_t)0);
        }
    }
}

template <typename Tc, typename Ta, typename Tb, unsigned Offset0, unsigned Offset1, unsigned L0CShape0,
          unsigned L0CShape1>
TILEOP void Tmad(__cc__ Tc* c, __ca__ Ta* a, __cb__ Tb* b, uint16_t m, uint16_t k, uint16_t n, bool zero_C, int uf)
{
    uint8_t unitFlag = uf;
    bool kDirectionAlign = true; // aligned to 8 for fp32
    bool cmatrixSource = false;  // true means bias

    mad((__cc__ Tc*)(c + (Offset0 * 16) + Offset1 * L0CShape0), a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource,
        zero_C);
}

template <typename GMT, typename L0CT, unsigned TShape0, unsigned TShape1, unsigned GmShape0, unsigned GmShape1,
          unsigned GmOffset0, unsigned GmOffset1, unsigned oriTShape0, unsigned oriTShape1>
TILEOP void L0CCopyOut(__gm__ GMT* dst, __cc__ L0CT* src, int uf)
{ // NZ2ND
    uint16_t MSize = oriTShape0 < (GmShape0 - GmOffset0) ? oriTShape0 : (GmShape0 - GmOffset0);
    uint16_t NSize = TShape1 < (GmShape1 - GmOffset1) ? TShape1 : (GmShape1 - GmOffset1);
    uint32_t dstStride_dst_D = GmShape1;
    uint16_t srcStride = TShape0;
    uint64_t ndNum = 1;
    uint64_t src_nd_stride = 0;
    uint64_t dst_nd_stride = 0;

    uint8_t UnitFlagMode = uf;
    uint64_t QuantPRE = NoQuant;
    uint8_t ReLUPRE = 0;
    bool channelSplit = false;
    bool NZ2ND_EN = true;

    uint64_t config = 0, nd_para = 0;
    nd_para = nd_para | (ndNum & 0xffff);
    nd_para = nd_para | ((src_nd_stride & 0xffff) << 16);
    nd_para = nd_para | ((dst_nd_stride & 0xffff) << 32);

    if (std::is_same<L0CT, float>::value) {
        if (std::is_same<GMT, half>::value) {
            QuantPRE = QuantMode_t::F322F16;
        } else if (std::is_same<GMT, bfloat16_t>::value) {
            QuantPRE = QuantMode_t::F322BF16;
        } else {
            QuantPRE = QuantMode_t::NoQuant;
        }
    }
    set_nd_para(nd_para);
    copy_matrix_cc_to_gm((__gm__ GMT*)dst, (__cc__ L0CT*)src, 0, NSize, MSize, dstStride_dst_D, srcStride, UnitFlagMode,
                         QuantPRE, ReLUPRE, channelSplit, NZ2ND_EN);
}

template <typename GMT, typename L0CT, unsigned TShape0, unsigned TShape1, unsigned GmShape0, unsigned GmShape1,
          unsigned GmOffset0, unsigned GmOffset1, unsigned oriTShape0, unsigned oriTShape1, int isAcc>
TILEOP void L0CCopyOut(__gm__ GMT* dst, __cc__ L0CT* src, int uf)
{
    SetAtomicAddition<GMT>();
    L0CCopyOut<GMT, L0CT, TShape0, TShape1, GmShape0, GmShape1, GmOffset0, GmOffset1, oriTShape0, oriTShape1>(dst, src,
                                                                                                              uf);
    if constexpr (isAcc == 1) {
        set_atomic_none();
    }
}
} // namespace TileOp

#endif
