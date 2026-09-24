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
 * \file cube_dyn.h
 * \brief
 */

#ifndef TILE_FWK_CUBE_DYN_H
#define TILE_FWK_CUBE_DYN_H

#include "tileop_common.h"
#include "cube.h"

#include <type_traits>

namespace TileOp {
constexpr uint16_t BLOCK_CUBE_M_N = 16;
constexpr uint16_t BLOCK_ALIGN_BYTE = 32;
constexpr uint16_t MAX_UINT16 = 65535;

template <typename T>
INLINE T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T>
INLINE T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

#define RT_OPERATION_OP_L1_COPY_IN(l1Addr, l1Dtype, l1Shape0, l1Shape1, l1ValidShape0, l1ValidShape1, l1Stride0,     \
                                   l1Stride1, gmAddr, gmDtype, gmShape0, gmShape1, gmOffset0, gmOffset1, copyInMode) \
    do {                                                                                                             \
        constexpr CopyInMode mode = static_cast<CopyInMode>(copyInMode);                                             \
        if (mode == CopyInMode::NZ2NZ) {                                                                             \
            DynL1CopyInNZ2NZ<gmDtype, l1Dtype>(l1Addr, gmAddr, l1ValidShape0, l1ValidShape1, gmShape0, gmShape1,     \
                                               gmOffset0, gmOffset1, gmShape0, gmShape1, 0);                         \
        } else {                                                                                                     \
            DynL1CopyIn<gmDtype, l1Dtype, mode>(l1Addr, gmAddr, l1ValidShape0, l1ValidShape1, gmShape0, gmShape1,    \
                                                gmOffset0, gmOffset1, 0);                                            \
        }                                                                                                            \
    } while (0)

template <typename GMT, typename L1T, CopyInMode mode = CopyInMode::ND2NZ>
TILEOP void DynL1CopyIn(__cbuf__ L1T* dst, __gm__ GMT* src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
                        unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, int reserved)
{
    if (TShape0 == 0 || TShape1 == 0) {
        return;
    }

    src += CalcLinearOffset(GmShape1, GmOffset0, GmOffset1);
    uint16_t nValue = TShape0;
    uint16_t dValue = TShape1;
    uint16_t srcDValue = GmShape1;
    uint16_t dstNzC0Stride = CeilAlign<uint16_t>(TShape0, BLOCK_CUBE_M_N);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t srcNdMatrixStride = 0;
    constexpr uint16_t dstNzNStride = 1;
    constexpr uint16_t dstNzMatrixStride = 1;

    if constexpr (mode == CopyInMode::ND2ND) {
        constexpr bool is_valid_type = std::is_same_v<GMT, half> || std::is_same_v<GMT, float> ||
                                       std::is_same_v<GMT, std::int32_t> || std::is_same_v<GMT, std::uint64_t>;
        static_assert(is_valid_type, "parameter must be one of: fp16, float (fp32), int32_t, uint64_t");
        dstNzC0Stride = 1;
        if constexpr (std::is_same<GMT, uint64_t>::value) {
            // The value 8 means that the input is of type scale_tensor with uint64_t data type, and it needs to be
            // copied in 8 chunks, each 1 byte in size.
            copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ int8_t*)dst, (__gm__ int8_t*)src, 0, ndNum, nValue, dValue * 8,
                                           srcNdMatrixStride, srcDValue * 8, dstNzC0Stride, dstNzNStride,
                                           dstNzMatrixStride);
            return;
        }
    }

    if constexpr (std::is_same<GMT, int8_t>::value) {
        constexpr uint16_t c0Size = BLOCK_ALIGN_BYTE / sizeof(GMT);
        dstNzC0Stride = CeilAlign<uint16_t>(TShape0, c0Size); // int8场景需要按32元素个数对齐
        copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ L1T*)dst, (__gm__ GMT*)src, 0, ndNum, nValue, dValue,
                                       srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, half>::value || std::is_same<GMT, bfloat16_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ L1T*)dst, (__gm__ GMT*)src, 0, ndNum, nValue, dValue,
                                        srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, float>::value || std::is_same<GMT, int32_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ L1T*)dst, (__gm__ GMT*)src, 0, ndNum, nValue, dValue,
                                         srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
}

template <typename GMT, typename L1T>
TILEOP void DynL1CopyInNZ2NZ(__cbuf__ L1T* dst, __gm__ GMT* src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
                             unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, unsigned curH, unsigned curW,
                             int reserved)
{
    if (TShape0 == 0 || TShape1 == 0) {
        return;
    }

    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(GMT);
    int64_t batchSize = curH * curW;
    int64_t offsetElem = GmOffset0 * GmShape1 + GmOffset1;
    int64_t batchIndex = offsetElem / batchSize;
    int64_t srcOffset = batchIndex * batchSize + (GmOffset1 * curH) + (GmOffset0 - batchIndex * curH) * c0Size;
    uint16_t nBurst = TShape1 / c0Size;
    uint16_t lenBurst = TShape0 * c0Size * sizeof(GMT) / BLOCK_ALIGN_BYTE;
    uint16_t srcStride = (curH - TShape0) * c0Size * sizeof(GMT) / BLOCK_ALIGN_BYTE;
    uint16_t dstStride = 0;
    if constexpr (std::is_same<GMT, int8_t>::value) {
        dstStride = CeilAlign<uint16_t>(TShape0, c0Size) - TShape0;
    }
    if (curH - TShape0 > MAX_UINT16) {
        nBurst = 1;
        srcStride = 0;
        for (int32_t nIdx = 0; nIdx < static_cast<int32_t>(TShape1 / c0Size); ++nIdx) {
            int64_t dstOffsetStep = nIdx * c0Size * TShape0;
            int64_t srcOffsetStep = nIdx * c0Size * curH + srcOffset;
            copy_gm_to_cbuf(dst + dstOffsetStep, src + srcOffsetStep, 0, nBurst, lenBurst, srcStride, dstStride,
                            PAD_NONE);
        }
    } else {
        copy_gm_to_cbuf(dst, src + srcOffset, 0, nBurst, lenBurst, srcStride, dstStride, PAD_NONE);
    }
}

#define RT_OPERATION_OP_L1_TO_L0A(l0aAddr, l0aDtype, l0aShape0, l0aShape1, l0aValidShape0, l0aValidShape1, l0aStride0, \
                                  l0aStride1, l1Addr, l1Dtype, l1Shape0, l1Shape1, l1ValidShape0, l1ValidShape1,       \
                                  l1Stride0, l1Stride1, l1Offset0, l1Offset1, reluMode, scaleValue)                    \
    DynL1ToL0A<l1Dtype, l1Offset0, l1Offset1>(l0aAddr, l1Addr, l0aValidShape0, l0aValidShape1, l1ValidShape0,          \
                                              l1ValidShape1);

template <typename T, unsigned Offset0, unsigned Offset1>
TILEOP void DynL1ToL0A(__ca__ T* dst, __cbuf__ T* src, unsigned dstM, unsigned dstK, unsigned srcM, unsigned srcK)
{
    if (dstM == 0 || dstK == 0 || srcM == 0 || srcK == 0) {
        return;
    }

    constexpr uint16_t c0Size = BLOCK_ALIGN_BYTE / sizeof(T);
    dstM = CeilAlign<uint16_t>(dstM, BLOCK_CUBE_M_N);
    dstK = CeilAlign<uint16_t>(dstK, c0Size);
    srcM = CeilAlign<uint16_t>(srcM, BLOCK_CUBE_M_N);
    srcK = CeilAlign<uint16_t>(srcK, c0Size);

    if constexpr (std::is_same<T, float>::value) {
        uint64_t config = srcM | (1 << 16); // 16含义：featureH对应的寄存器偏移
        set_fmatrix(config);
        dstK = CeilAlign<uint16_t>(dstK, BLOCK_CUBE_M_N);
        // LOAD3DV2 param: dstAddr, srcAddr, stepK, stepM, posK, posM, strideW, strideH, Wk, Hk, dilationW, dilationH,
        // filterW, filterH, transpose, fmatrixCtrl, sizeChannel
        img2colv2_cbuf_to_ca(dst, src, dstK, dstM, Offset1, Offset0, 1, 1, 1, 1, 1, 1, false, false, false, false,
                             srcK);
        return;
    }

    if constexpr (std::is_same<T, int8_t>::value) {
        dstM = CeilAlign<uint16_t>(dstM, c0Size);
        srcM = CeilAlign<uint16_t>(srcM, c0Size);
    }

    uint8_t repeat = dstK / c0Size;
    uint16_t srcStride = srcM / BLOCK_CUBE_M_N;
    constexpr uint16_t dstStride = 0;
    int32_t dstOffsetStep = BLOCK_CUBE_M_N * dstK;
    int32_t srcOffset = Offset0 * c0Size + Offset1 * srcM;
    int32_t srcOffsetStep = BLOCK_CUBE_M_N * c0Size;
    int32_t dstOffset = 0;
    for (int32_t mIdx = 0; mIdx < static_cast<int32_t>(dstM / BLOCK_CUBE_M_N); ++mIdx) {
        load_cbuf_to_ca(dst + dstOffset, src + srcOffset, 0, repeat, srcStride, dstStride, 0, 0, inc);
        dstOffset += dstOffsetStep;
        srcOffset += srcOffsetStep;
    }
}

#define RT_OPERATION_OP_L1_TO_L0_AT(l0aAddr, l0aDtype, l0aShape0, l0aShape1, l0aValidShape0, l0aValidShape1,         \
                                    l0aStride0, l0aStride1, l1Addr, l1Dtype, l1Shape0, l1Shape1, l1ValidShape0,      \
                                    l1ValidShape1, l1Stride0, l1Stride1, l1Offset0, l1Offset1, reluMode, scaleValue) \
    DynL1ToL0At<l1Dtype, l1Offset0, l1Offset1>(l0aAddr, l1Addr, l0aValidShape0, l0aValidShape1, l1ValidShape0,       \
                                               l1ValidShape1);

template <typename T, unsigned Offset0, unsigned Offset1>
TILEOP void DynL1ToL0At(__ca__ T* dst, __cbuf__ T* src, unsigned dstM, unsigned dstK, unsigned srcK, unsigned srcM)
{
    if (dstM == 0 || dstK == 0 || srcM == 0 || srcK == 0) {
        return;
    }

    constexpr uint16_t c0Size = BLOCK_ALIGN_BYTE / sizeof(T);
    dstM = CeilAlign<uint16_t>(dstM, c0Size);
    dstK = CeilAlign<uint16_t>(dstK, BLOCK_CUBE_M_N);
    srcM = CeilAlign<uint16_t>(srcM, c0Size);
    srcK = CeilAlign<uint16_t>(srcK, BLOCK_CUBE_M_N);

    if constexpr (std::is_same<T, float>::value) {
        uint64_t config = srcK | (1 << 16); // 16含义：featureH对应的寄存器偏移
        set_fmatrix(config);
        // LOAD3DV2 param: dstAddr, srcAddr, stepK, stepM, posK, posM, strideW, strideH, Wk, Hk, dilationW, dilationH,
        // filterW, filterH, transpose, fmatrixCtrl, sizeChannel
        img2colv2_cbuf_to_ca(dst, src, dstM, dstK, Offset1, Offset0, 1, 1, 1, 1, 1, 1, false, false, true, false, srcM);
        return;
    }

    if constexpr (std::is_same<T, int8_t>::value) {
        dstK = CeilAlign<uint16_t>(dstK, c0Size);
        srcK = CeilAlign<uint16_t>(srcK, c0Size);
        uint8_t repeat = dstK / c0Size;
        uint16_t dstFracStride = dstK / c0Size - 1;
        int32_t dstOffset = 0;
        int32_t srcOffset = Offset0 * c0Size + Offset1 * srcK;
        int32_t dstOffsetStep = dstK * c0Size;
        int32_t srcOffsetStep = srcK * c0Size;
        for (int32_t mIdx = 0; mIdx < static_cast<int32_t>(dstM / c0Size); ++mIdx) {
            load_cbuf_to_ca_transpose(dst + dstOffset, src + srcOffset, 0, repeat, 1, 0, inc, dstFracStride);
            dstOffset += dstOffsetStep;
            srcOffset += srcOffsetStep;
        }
        return;
    }

    if (dstK == srcK) {
        uint8_t repeat = (dstK / BLOCK_CUBE_M_N) * (dstM / c0Size);
        load_cbuf_to_ca(dst, src, 0, repeat, 1, 0, 0, 1, inc);
    } else {
        uint8_t repeat = dstK / BLOCK_CUBE_M_N;
        int32_t dstOffset = 0;
        int32_t dstOffsetStep = BLOCK_CUBE_M_N * dstK;
        int32_t srcOffset = Offset0 * c0Size + Offset1 * srcK;
        int32_t srcOffsetStep = srcK * c0Size;
        for (int32_t mIdx = 0; mIdx < static_cast<int32_t>(dstM / c0Size); ++mIdx) {
            load_cbuf_to_ca(dst + dstOffset, src + srcOffset, 0, repeat, 1, 0, 0, 1, inc);
            dstOffset += dstOffsetStep;
            srcOffset += srcOffsetStep;
        }
    }
}

#define RT_OPERATION_OP_L1_TO_L0_B(l0bAddr, l0bDtype, l0bShape0, l0bShape1, l0bValidShape0, l0bValidShape1,         \
                                   l0bStride0, l0bStride1, l1Addr, l1Dtype, l1Shape0, l1Shape1, l1ValidShape0,      \
                                   l1ValidShape1, l1Stride0, l1Stride1, l1Offset0, l1Offset1, reluMode, scaleValue) \
    DynL1ToL0B<l1Dtype, l1Offset0, l1Offset1>(l0bAddr, l1Addr, l0bValidShape0, l0bValidShape1, l1ValidShape0,       \
                                              l1ValidShape1);

template <typename T, unsigned Offset0, unsigned Offset1>
TILEOP void DynL1ToL0B(__cb__ T* dst, __cbuf__ T* src, unsigned dstK, unsigned dstN, unsigned srcK, unsigned srcN)
{
    if (dstK == 0 || dstN == 0 || srcK == 0 || srcN == 0) {
        return;
    }

    if constexpr (std::is_same<T, float>::value) {
        // need to enable k-alignment in mmad to copy with even numbers align to 8 should actually be odd numbers to 8.
        // load3D automatically to 16
        dstK = CeilAlign<uint16_t>(dstK, BLOCK_CUBE_M_N);
        dstN = CeilAlign<uint16_t>(dstN, BLOCK_CUBE_M_N);
        srcN = CeilAlign<uint16_t>(srcN, BLOCK_ALIGN_BYTE / sizeof(T));
        srcK = CeilAlign<uint16_t>(srcK, BLOCK_CUBE_M_N);
        uint64_t config = srcK | (1 << 16); // 16含义：featureH对应的寄存器偏移
        set_fmatrix_b(config);
        // LOAD3DV2 param: dstAddr, srcAddr, stepK, stepM, posK, posM, strideW, strideH, Wk, Hk, dilationW, dilationH,
        // filterW, filterH, transpose, fmatrixCtrl, sizeChannel
        img2colv2_cbuf_to_cb(dst, src, dstN, dstK, Offset1, Offset0, 1, 1, 1, 1, 1, 1, false, false, false, true, srcN);
        return;
    }

    constexpr uint16_t c0Size = BLOCK_ALIGN_BYTE / sizeof(T);
    dstK = CeilAlign<uint16_t>(dstK, c0Size);
    dstN = CeilAlign<uint16_t>(dstN, c0Size);
    srcN = CeilAlign<uint16_t>(srcN, c0Size);
    srcK = CeilAlign<uint16_t>(srcK, c0Size);

    if constexpr (std::is_same<T, int8_t>::value) {
        uint8_t repeat = dstK / BLOCK_ALIGN_BYTE;
        constexpr uint16_t srcStride = 1;
        uint16_t dstStride = dstN / BLOCK_CUBE_M_N - 1;
        constexpr uint16_t dstFracStride = 0;
        int32_t dstOffset = 0;
        int32_t srcOffset = Offset0 * BLOCK_ALIGN_BYTE + Offset1 * srcK;
        int32_t dstOffsetStep = BLOCK_ALIGN_BYTE * BLOCK_ALIGN_BYTE;
        int32_t srcOffsetStep = srcK * BLOCK_ALIGN_BYTE;
        for (int32_t nIdx = 0; nIdx < static_cast<int32_t>(dstN / BLOCK_ALIGN_BYTE); ++nIdx) {
            load_cbuf_to_cb_transpose(dst + dstOffset, src + srcOffset, 0, repeat, srcStride, dstStride, inc,
                                      dstFracStride);
            dstOffset += dstOffsetStep;
            srcOffset += srcOffsetStep;
        }
        return;
    }

    uint8_t repeat = dstN / BLOCK_CUBE_M_N;
    uint16_t srcStride = srcK / c0Size;
    constexpr uint16_t dstStride = 0;
    for (int32_t kIdx = 0; kIdx < static_cast<int32_t>(dstK / c0Size); ++kIdx) {
        load_cbuf_to_cb(dst + kIdx * c0Size * dstN,
                        src + kIdx * BLOCK_CUBE_M_N * c0Size + (Offset0 * BLOCK_CUBE_M_N + Offset1 * srcK), 0, repeat,
                        srcStride, dstStride, 0, 1, inc);
    }
}

#define RT_OPERATION_OP_L1_TO_L0_BT(l0bAddr, l0bDtype, l0bShape0, l0bShape1, l0bValidShape0, l0bValidShape1,         \
                                    l0bStride0, l0bStride1, l1Addr, l1Dtype, l1Shape0, l1Shape1, l1ValidShape0,      \
                                    l1ValidShape1, l1Stride0, l1Stride1, l1Offset0, l1Offset1, reluMode, scaleValue) \
    DynL1ToL0Bt<l1Dtype, l1Offset0, l1Offset1>(l0bAddr, l1Addr, l0bValidShape0, l0bValidShape1, l1ValidShape0,       \
                                               l1ValidShape1);

template <typename T, unsigned Offset0, unsigned Offset1>
TILEOP void DynL1ToL0Bt(__cb__ T* dst, __cbuf__ T* src, unsigned dstK, unsigned dstN, unsigned srcN, unsigned srcK)
{
    if (dstK == 0 || dstN == 0 || srcK == 0 || srcN == 0) {
        return;
    }

    constexpr uint16_t c0Size = BLOCK_ALIGN_BYTE / sizeof(T);
    dstN = CeilAlign<uint16_t>(dstN, BLOCK_CUBE_M_N);
    dstK = CeilAlign<uint16_t>(dstK, c0Size);
    srcN = CeilAlign<uint16_t>(srcN, BLOCK_CUBE_M_N);
    srcK = CeilAlign<uint16_t>(srcK, c0Size);

    if constexpr (std::is_same<T, int8_t>::value) {
        dstN = CeilAlign<uint16_t>(dstN, c0Size);
        srcN = CeilAlign<uint16_t>(srcN, c0Size);
    }

    if (dstN == srcN) {
        uint8_t repeat = dstN / BLOCK_CUBE_M_N * dstK / c0Size;
        constexpr uint16_t srcStride = 1;
        constexpr uint16_t dstStride = 0;
        load_cbuf_to_cb(dst, src + (Offset0 * c0Size + Offset1 * srcN), (uint16_t)0, repeat, srcStride, dstStride,
                        (uint8_t)0, (bool)0, (addr_cal_mode_t)0);
    } else {
        uint8_t repeat = dstN / BLOCK_CUBE_M_N;
        constexpr uint16_t srcStride = 1;
        constexpr uint16_t dstStride = 0;
        for (int32_t kIdx = 0; kIdx < static_cast<int32_t>(dstK / c0Size); ++kIdx) {
            load_cbuf_to_cb(dst + kIdx * c0Size * dstN,
                            src + kIdx * srcN * c0Size + (Offset0 * c0Size + Offset1 * srcN), (uint16_t)0, repeat,
                            srcStride, dstStride, (uint8_t)0, (bool)0, (addr_cal_mode_t)0);
        }
    }
}

#define RT_OPERATION_OP_L1_TO_BT(biasTableAddr, biasTableDtype, biasTableShape0, biasTableShape1,                \
                                 biasTableValidShape0, biasTableValidShape1, biasTableStride0, biasTableStride1, \
                                 l1Addr, l1Dtype, l1Shape0, l1Shape1, l1ValidShape0, l1ValidShape1, l1Stride0,   \
                                 l1Stride1, l1Offset0, l1Offset1)                                                \
    DynL1ToBT<l1Dtype, biasTableDtype, l1Offset1>(biasTableAddr, l1Addr, biasTableValidShape1);

template <typename L1T, typename BTT, unsigned Offset>
TILEOP void DynL1ToBT(uint64_t dst, __cbuf__ L1T* src, unsigned nSize)
{
    constexpr uint16_t nBurst = 1;
    uint16_t lenBurst = CeilDiv<uint16_t>(nSize * sizeof(L1T), 64); // IN UNIT OF 64B
    constexpr uint16_t sourceGap = 0;
    constexpr uint16_t dstGap = 0;
    constexpr bool convControl = std::is_same<L1T, half>::value;
    copy_cbuf_to_bt(dst, src + Offset, convControl, nBurst, lenBurst, sourceGap, dstGap);
}

#define RT_OPERATION_OP_L1_TO_FIX_QUANT_PRE(fixPipeAddr, fixPipeDtype, fixPipeShape0, fixPipeShape1,                \
                                            fixPipeValidShape0, fixPipeValidShape1, fixPipeStride0, fixPipeStride1, \
                                            l1Addr, l1Dtype, l1Shape0, l1Shape1, l1ValidShape0, l1ValidShape1,      \
                                            l1Stride0, l1Stride1, l1Offset0, l1Offset1)                             \
    DynL1ToFB<l1Dtype, l1Offset1>(fixPipeAddr, l1Addr, fixPipeValidShape1);

template <typename T, unsigned L1Offset>
TILEOP void DynL1ToFB(__fbuf__ T* dst, __cbuf__ T* src, unsigned nSize)
{
    // align to 128B
    uint16_t deqDataSize = CeilDiv<uint16_t>(nSize * sizeof(uint64_t), 128) * 128;
    // l1->fb
    uint16_t fbufBurstLen = deqDataSize / 128; // copy from cbuf to fbuf,burst len uint is 128Bytes
    copy_cbuf_to_fbuf(dst, src + L1Offset, 1, fbufBurstLen, 0, 0);
    // FPC of fixpipe for quant_pre is FPX[15:8],uint is 128Bytes
    // 7 means dst to 8 to set fpc
    uint64_t deqTensorAddr = ((uint64_t)dst >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
}

#define RT_OPERATION_OP_A_MUL_B(l0cAddr, l0cDtype, l0cShape0, l0cShape1, l0cValidShape0, l0cValidShape1, l0cStride0, \
                                l0cStride1, l0aAddr, l0aDtype, l0aShape0, l0aShape1, l0aValidShape0, l0aValidShape1, \
                                l0aStride0, l0aStride1, l0bAddr, l0bDtype, l0bShape0, l0bShape1, l0bValidShape0,     \
                                l0bValidShape1, l0bStride0, l0bStride1, hasbias)                                     \
    DynTmad<l0cDtype, l0aDtype, l0bDtype, 0, 0, hasbias>(l0cAddr, l0aAddr, l0bAddr, l0aValidShape0, l0aValidShape1,  \
                                                         l0bValidShape1, false, 0, l0cValidShape0, l0cValidShape1);

#define RT_OPERATION_OP_A_MULACC_B(l0cAddr, l0cDtype, l0cShape0, l0cShape1, l0cValidShape0, l0cValidShape1,         \
                                   l0cStride0, l0cStride1, l0aAddr, l0aDtype, l0aShape0, l0aShape1, l0aValidShape0, \
                                   l0aValidShape1, l0aStride0, l0aStride1, l0bAddr, l0bDtype, l0bShape0, l0bShape1, \
                                   l0bValidShape0, l0bValidShape1, l0bStride0, l0bStride1, hasbias)                 \
    DynTmad<l0cDtype, l0aDtype, l0bDtype, 0, 0, hasbias>(l0cAddr, l0aAddr, l0bAddr, l0aValidShape0, l0aValidShape1, \
                                                         l0bValidShape1, true, 0, l0cValidShape0, l0cValidShape1);

template <typename Tc, typename Ta, typename Tb, unsigned Offset0, unsigned Offset1, bool HasBias = false>
TILEOP void DynTmad(__cc__ Tc* c, __ca__ Ta* a, __cb__ Tb* b, uint16_t m, uint16_t k, uint16_t n, bool zero_C, int uf,
                    unsigned L0CShape0, unsigned L0CShape1)
{
    if (m == 0 || k == 0 || n == 0) {
        return;
    }
    if constexpr (HasBias) {
        zero_C = false;
    }
    m = CeilAlign<uint16_t>(m, BLOCK_CUBE_M_N);
    if constexpr (std::is_same<Tb, int8_t>::value) {
        n = CeilAlign<uint16_t>(n, 32); // 32含义：int8场景总是保证L0B中32对齐
    }
    constexpr bool kDirectionAlign = true;
    mad((__cc__ Tc*)(c + (Offset0 * BLOCK_CUBE_M_N) + Offset1 * L0CShape0), a, b, m, k, n, 0, kDirectionAlign, HasBias,
        zero_C);
    pipe_barrier(PIPE_M);
}

#define RT_OPERATION_OP_L0C_COPY_OUT(gmAddr, gmDtype, gmShape0, gmShape1, l0cAddr, l0cDtype, l0cShape0, l0cShape1,  \
                                     l0cValidShape0, l0cValidShape1, l0cStride0, l0cStride1, gmOffset0, gmOffset1,  \
                                     reluMode, enableGmAcc, scaleValue)                                             \
    do {                                                                                                            \
        if (enableGmAcc) {                                                                                          \
            DynL0CCopyOut<gmDtype, l0cDtype, 1, true, reluMode>(gmAddr, l0cAddr, l0cValidShape0, l0cValidShape1,    \
                                                                gmShape0, gmShape1, gmOffset0, gmOffset1, gmShape0, \
                                                                gmShape1, 0);                                       \
        } else {                                                                                                    \
            DynL0CCopyOut<gmDtype, l0cDtype, 0, true, reluMode>(gmAddr, l0cAddr, l0cValidShape0, l0cValidShape1,    \
                                                                gmShape0, gmShape1, gmOffset0, gmOffset1, gmShape0, \
                                                                gmShape1, 0, scaleValue);                           \
        }                                                                                                           \
    } while (0)

template <typename GMT, typename L0CT, bool enableNZ2ND, uint8_t reluMode>
TILEOP void DynL0CCopyOut(__gm__ GMT* dst, __cc__ L0CT* src, unsigned oriTShape0, unsigned oriTShape1,
                          unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, unsigned curH,
                          unsigned curW, int uf, uint64_t scaleValue = 0)
{
    if (oriTShape0 == 0 || oriTShape1 == 0) {
        return;
    }

    uint16_t mSize = oriTShape0;
    uint16_t nSize = oriTShape1; // should be multiples of 8 when fp32 & channel split
    uint32_t dstStrideDstD = GmShape1;
    uint16_t srcStride = CeilAlign<uint16_t>(oriTShape0, BLOCK_CUBE_M_N);
    bool channelSplit = false;
    int64_t gmOffset = (GmOffset0 * GmShape1) + GmOffset1;

    if constexpr (!enableNZ2ND) {
        // s32搬出不涉及channel split，因此C0=16
        int64_t c0Size = std::is_same<GMT, int32_t>::value ? BLOCK_CUBE_M_N : BLOCK_ALIGN_BYTE / sizeof(GMT);
        nSize = CeilAlign<uint16_t>(nSize, c0Size);
        int64_t wAlign = CeilAlign<int64_t>(curW, c0Size);
        int64_t elemPerBatch = curH * wAlign;
        int64_t batchIdx = gmOffset / elemPerBatch;
        gmOffset = batchIdx * elemPerBatch + (GmOffset1 * curH) + (GmOffset0 - batchIdx * curH) * c0Size;
        // fp32搬出默认开启channel split
        channelSplit = std::is_same<GMT, float>::value;
        // dst stride between the start addresses of different bursts in unit of 32B
        // 2含义：int32 NZ搬出场景（C0=16），此处dstStride需乘2使得内轴按64B为单位做偏移计算
        dstStrideDstD = std::is_same<GMT, int32_t>::value ? curH * 2 : curH;
    }

    constexpr uint64_t ndNum = 1;
    constexpr uint64_t srcNdStride = 0;
    constexpr uint64_t dstNdStride = 0;
    constexpr uint64_t ndPara = (ndNum & 0xffff) | ((srcNdStride & 0xffff) << 16) | ((dstNdStride & 0xffff) << 32);
    set_nd_para(ndPara);

    uint64_t quantPre = NoQuant;
    if (std::is_same<L0CT, float>::value) {
        if (std::is_same<GMT, half>::value) {
            quantPre = QuantMode_t::F322F16;
        } else if (std::is_same<GMT, bfloat16_t>::value) {
            quantPre = QuantMode_t::F322BF16;
        } else {
            quantPre = QuantMode_t::NoQuant;
        }
    } else if constexpr (std::is_same<L0CT, int32_t>::value && std::is_same<GMT, half>::value) {
        if (scaleValue == 0) {
            quantPre = QuantMode_t::VDEQF16;
        } else {
            set_quant_pre(scaleValue);
            quantPre = QuantMode_t::DEQF16;
        }
    }
    uint8_t unitFlagMode = uf;

    copy_matrix_cc_to_gm((__gm__ GMT*)(dst + gmOffset), (__cc__ L0CT*)src, 0, nSize, mSize, dstStrideDstD, srcStride,
                         unitFlagMode, quantPre, reluMode, channelSplit, enableNZ2ND);
}

template <typename GMT, typename L0CT, int isAcc, bool enableNZ2ND, uint8_t reluMode>
TILEOP void DynL0CCopyOut(__gm__ GMT* dst, __cc__ L0CT* src, unsigned oriTShape0, unsigned oriTShape1,
                          unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, unsigned curH,
                          unsigned curW, int uf)
{
    static_assert(reluMode == 0, "Relu operation is not supported in GM accumulate mode");
    SetAtomicAddition<GMT>();
    DynL0CCopyOut<GMT, L0CT, enableNZ2ND, 0>(dst, src, oriTShape0, oriTShape1, GmShape0, GmShape1, GmOffset0, GmOffset1,
                                             curH, curW, uf);
    if constexpr (isAcc == 1) {
        set_atomic_none();
    }
}

#define RT_OPERATION_OP_L0C_TO_L1(l1Addr, l1Dtype, l1Shape0, l1Shape1, l1ValidShape0, l1ValidShape1, l1Stride0,       \
                                  l1Stride1, l0cAddr, l0cDtype, l0cShape0, l0cShape1, l0cValidShape0, l0cValidShape1, \
                                  l0cStride0, l0cStride1, offset0, offset1, reluMode, scaleValue)                     \
    DynL0CToL1<l1Dtype, l0cDtype, reluMode>(l1Addr, l0cAddr, l0cValidShape0, l0cValidShape1, l1ValidShape0,           \
                                            l1ValidShape1, offset0, offset1, scaleValue);

template <typename L1T, typename L0CT, uint8_t reluMode>
TILEOP void DynL0CToL1(__cbuf__ L1T* dst, __cc__ L0CT* src, unsigned shape0, unsigned shape1, unsigned l1Shape0,
                       unsigned l1Shape1, unsigned l1Offset0, unsigned l1Offset1, unsigned l0cShape0,
                       unsigned l0cShape1, unsigned l0cOffset0, unsigned l0cOffset1, uint64_t scaleValue = 0)
{
    int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(L1T);
    uint16_t mSize = CeilAlign<uint16_t>(shape0, BLOCK_CUBE_M_N);
    uint16_t nSize = CeilAlign<uint16_t>(shape1, c0Size);
    uint32_t dstStrideDstD = CeilAlign<uint16_t>(l1Shape0, BLOCK_CUBE_M_N);
    ;
    uint16_t srcStride = CeilAlign<uint16_t>(l0cShape0, BLOCK_CUBE_M_N);
    uint8_t unitFlagMode = 0;
    uint64_t quantPre = NoQuant;
    uint8_t reluPre = 0;
    if (std::is_same<L0CT, float>::value) {
        if (std::is_same<L1T, half>::value) {
            quantPre = QuantMode_t::F322F16;
        } else if (std::is_same<L1T, bfloat16_t>::value) {
            quantPre = QuantMode_t::F322BF16;
        } else {
            quantPre = QuantMode_t::NoQuant;
        }
    } else if constexpr (std::is_same<L0CT, int32_t>::value && std::is_same<L1T, half>::value) {
        if (scaleValue == 0) {
            quantPre = QuantMode_t::VDEQF16;
        } else {
            set_quant_pre(scaleValue);
            quantPre = QuantMode_t::DEQF16;
        }
    }
    bool channelSplit = std::is_same<L1T, float>::value;
    bool nZ2NDEN = false;
    int64_t l1Offset = l1Offset1 * l1Shape0 + l1Offset0 * c0Size;
    int64_t l0cOffset = l0cOffset1 * l0cShape0 + l0cOffset0 * c0Size;
    copy_matrix_cc_to_cbuf((__cbuf__ L1T*)(dst + l1Offset), (__cc__ L0CT*)(src + l0cOffset), 0, nSize, mSize,
                           dstStrideDstD, srcStride, unitFlagMode, quantPre, reluMode, channelSplit, nZ2NDEN);
}

// Internal: Reserved for custom scenarios.
template <typename T, typename T2, typename T3, int64_t dstRawShape0, int64_t offsetRawShape1,
          int64_t srcColumnStartOffset, int64_t blockSize>
TILEOP void GatherInL1(__cbuf__ T* dst, int64_t dstOriginShape0, int64_t dstOriginShape1, __gm__ T* src,
                       int64_t srcRawShape1, __gm__ T2* offsets, __gm__ T3* blockTable, int64_t offsetsRowStartOffset,
                       int64_t offsetsColumnStartOffset, int64_t GMBlockTableStride1, int64_t GMBlockTableOffset0,
                       int64_t GMBlockTableOffset1)
{
    static_assert(std::is_same_v<T2, int32_t> || std::is_same_v<T2, int64_t>);
    constexpr uint16_t c0Size = BLOCK_SIZE / sizeof(T);
    uint16_t nBurst = dstOriginShape1 / c0Size;
    uint16_t dstStride = CeilAlign<uint16_t>(dstOriginShape0, BLOCK_CUBE_M_N) - 1;
    int64_t offsetsStartOffset = offsetsRowStartOffset * offsetRawShape1 + offsetsColumnStartOffset;
    blockTable += GMBlockTableOffset0 * GMBlockTableStride1 + GMBlockTableOffset1;
    pipe_barrier(PIPE_ALL);
    if (dstOriginShape1 % c0Size > 0) {
        uint16_t dValue = dstOriginShape1;
        uint16_t srcDValue = srcRawShape1;
        uint16_t dstNzC0Stride = CeilAlign<uint16_t>(dstOriginShape0, BLOCK_CUBE_M_N);
        if constexpr (std::is_same<T, int8_t>::value) {
            dstNzC0Stride = CeilAlign<uint16_t>(dstOriginShape0, c0Size); // int8场景需要按32元素个数对齐
            for (int64_t i = 0; i < dstOriginShape0; i++) {
                uint64_t gatherOffset = offsets[i + offsetsStartOffset];
                gatherOffset = CalaOffset2PageAttention<uint64_t, T3, blockSize>(blockTable, gatherOffset);
                copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ T*)dst + i * c0Size,
                                               (__gm__ T*)src + gatherOffset * srcRawShape1 + srcColumnStartOffset, 0,
                                               1, 1, dValue, 0, srcDValue, dstNzC0Stride, 1, 1);
            }
        }
        if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
            for (int64_t i = 0; i < dstOriginShape0; i++) {
                uint64_t gatherOffset = offsets[i + offsetsStartOffset];
                gatherOffset = CalaOffset2PageAttention<uint64_t, T3, blockSize>(blockTable, gatherOffset);
                copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ T*)dst + i * c0Size,
                                                (__gm__ T*)src + gatherOffset * srcRawShape1 + srcColumnStartOffset, 0,
                                                1, 1, dValue, 0, srcDValue, dstNzC0Stride, 1, 1);
            }
        }
        if constexpr (std::is_same<T, float>::value) {
            for (int64_t i = 0; i < dstOriginShape0; i++) {
                uint64_t gatherOffset = offsets[i + offsetsStartOffset];
                gatherOffset = CalaOffset2PageAttention<uint64_t, T3, blockSize>(blockTable, gatherOffset);
                copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ T*)dst + i * c0Size,
                                                 (__gm__ T*)src + gatherOffset * srcRawShape1 + srcColumnStartOffset, 0,
                                                 1, 1, dValue, 0, srcDValue, dstNzC0Stride, 1, 1);
            }
        }
    } else {
        if constexpr (std::is_same<T, int8_t>::value) {
            dstStride = CeilAlign<uint16_t>(dstOriginShape0, c0Size) - 1;
        }
        for (int64_t i = 0; i < dstOriginShape0; i++) {
            uint64_t gatherOffset = offsets[i + offsetsStartOffset];
            gatherOffset = CalaOffset2PageAttention<uint64_t, T3, blockSize>(blockTable, gatherOffset);
            copy_gm_to_cbuf(dst + i * c0Size, src + gatherOffset * srcRawShape1 + srcColumnStartOffset, 0, nBurst, 1, 0,
                            dstStride, PAD_NONE);
        }
    }
    pipe_barrier(PIPE_ALL);
}

// Deprecated: Normal dynamic scene
template <typename T, unsigned dstM, unsigned dstK, unsigned Offset0, unsigned Offset1, unsigned srcM, unsigned srcK>
TILEOP void DynL1ToL0A(__ca__ T* dst, __cbuf__ T* src)
{
    DynL1ToL0A<T, Offset0, Offset1>(dst, src, dstM, dstK, srcM, srcK);
}

template <typename T, unsigned dstM, unsigned dstK, unsigned Offset0, unsigned Offset1, unsigned srcK, unsigned srcM>
TILEOP void DynL1ToL0At(__ca__ T* dst, __cbuf__ T* src)
{
    DynL1ToL0At<T, Offset0, Offset1>(dst, src, dstM, dstK, srcK, srcM);
}

template <typename T, unsigned dstK, unsigned dstN, unsigned Offset0, unsigned Offset1, unsigned srcK, unsigned srcN>
TILEOP void DynL1ToL0B(__cb__ T* dst, __cbuf__ T* src)
{
    DynL1ToL0B<T, Offset0, Offset1>(dst, src, dstK, dstN, srcK, srcN);
}

template <typename T, unsigned dstK, unsigned dstN, unsigned Offset0, unsigned Offset1, unsigned srcN, unsigned srcK>
TILEOP void DynL1ToL0Bt(__cb__ T* dst, __cbuf__ T* src)
{
    DynL1ToL0Bt<T, Offset0, Offset1>(dst, src, dstK, dstN, srcN, srcK);
}

// Deprecated: L1 spill out scene
template <typename GMT, typename L1T>
TILEOP void DynL1CopyOutND(__gm__ GMT* dst, __cbuf__ L1T* src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
                           unsigned GmShape1, int reserved)
{
    uint16_t nBurst = TShape0;
    uint16_t lenBurst = TShape1 * sizeof(GMT) / BLOCK_SIZE;
    uint16_t srcStride = 0;
    uint16_t dstStride = (GmShape1 - TShape1) * sizeof(GMT) / BLOCK_SIZE;

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

// Deprecated: Currently 'L1CopyOut' is ONLY used when spilling occurred, and does NOT need data format conversion. the
// impl redirect this function to L1CopyOutND directly.
template <typename GMT, typename L1T>
TILEOP void DynL1CopyOut(__gm__ GMT* dst, __cbuf__ L1T* src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
                         unsigned GmShape1, int reserved)
{
    TileOp::DynL1CopyOutND<GMT, L1T>(dst, src, TShape0, TShape1, GmShape0, GmShape1, reserved);
}

// Deprecated: L1 spill out scene
template <typename GMT, typename L1T>
TILEOP void DynL1CopyIn(__cbuf__ L1T* dst, __gm__ GMT* src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
                        unsigned GmShape1, int reserved)
{
    uint16_t nBurst = TShape0;
    uint16_t lenBurst = TShape1 * sizeof(GMT) / 32;
    uint16_t srcStride = 0;
    uint16_t dstStride = (GmShape1 - TShape1) * sizeof(GMT) / BLOCK_SIZE;

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

/*
 * Deprecated: The following interfaces are deprecated and will no longer be maintained.
 */
/*
 * brief: dynamic l1 copy in nz2nz functions with batch
 */
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1>
TILEOP void DynL1CopyInNZ2NZ(__cbuf__ L1T* dst, __gm__ GMT* src, unsigned GmShape0, unsigned GmShape1,
                             unsigned GmOffset0, unsigned GmOffset1, unsigned curH, unsigned curW, int reserved)
{
    auto inputC0Size = 32 / sizeof(L1T);
    // 计算在那个Batch块中;
    auto batchSize = curH * curW;
    // 计算当前的偏移点
    auto offsetElem = GmOffset0 * GmShape1 + GmOffset1;
    auto batchIndex = offsetElem / batchSize;
    // auto batchIndex = CalcLinearOffset(GmShape1, GmOffset0, GmOffset1) / batchSize;
    // NZ的offset转换
    auto offsetWithNZ = batchIndex * batchSize + (GmOffset1 * curH) + (GmOffset0 - batchIndex * curH) * inputC0Size;
    int32_t C0 = 32 / sizeof(GMT);
    uint16_t nBurst = TShape1 / C0;
    uint16_t lenBurst = TShape0 * C0 * sizeof(GMT) / 32;
    uint16_t srcStride = (curH - TShape0) * C0 * sizeof(GMT) / 32;
    uint16_t dstStride = 0;
    copy_gm_to_cbuf(dst, src + offsetWithNZ, 0 /*sid*/, nBurst, lenBurst, srcStride, dstStride, PAD_NONE);
}

/*
 * brief: dynamic l1 copy in nd2nz functions
 */
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1>
TILEOP void DynL1CopyIn(__cbuf__ L1T* dst, __gm__ GMT* src, unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0,
                        unsigned GmOffset1, int reserved)
{ // ND2NZ
    src += CalcLinearOffset(GmShape1, GmOffset0, GmOffset1);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t nValue = TShape0;      // n
    constexpr uint16_t dValue = TShape1;      // d
    constexpr uint16_t srcNdMatrixStride = 0; //
    uint16_t srcDValue = GmShape1;            // D
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
TILEOP void DynL1CopyOutND(__gm__ GMT* dst, __cbuf__ L1T* src, int reserved)
{
    uint16_t nBurst = TShape0;
    uint16_t lenBurst = TShape1 * sizeof(GMT) / BLOCK_SIZE;
    uint16_t srcStride = 0;
    uint16_t dstStride = (GmShape1 - TShape1) * sizeof(GMT) / BLOCK_SIZE;

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
TILEOP void DynL1CopyOut(__gm__ GMT* dst, __cbuf__ L1T* src, int reserved)
{
    TileOp::DynL1CopyOutND<GMT, L1T, TShape0, TShape1, GmShape0, GmShape1>(dst, src, reserved);
}

// L1 spill out scene
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1>
TILEOP void DynL1CopyInND(__cbuf__ L1T* dst, __gm__ GMT* src, unsigned GmShape0, unsigned GmShape1, int reserved)
{
    uint16_t nBurst = TShape0;
    uint16_t lenBurst = TShape1 * sizeof(GMT) / 32;
    uint16_t srcStride = 0;
    uint16_t dstStride = (GmShape1 - TShape1) * sizeof(GMT) / BLOCK_SIZE;

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

template <typename GMT, typename L0CT, unsigned TShape0, unsigned TShape1, unsigned oriTShape0, unsigned oriTShape1>
TILEOP void DynL0CCopyOut(__gm__ GMT* dst, __cc__ L0CT* src, unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0,
                          unsigned GmOffset1, int uf)
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
    copy_matrix_cc_to_gm((__gm__ GMT*)(dst + (GmOffset0 * GmShape1) + GmOffset1), (__cc__ L0CT*)src, 0, NSize, MSize,
                         dstStride_dst_D, srcStride, UnitFlagMode, QuantPRE, ReLUPRE, channelSplit, NZ2ND_EN);
}

template <typename GMT, typename L0CT, unsigned TShape0, unsigned TShape1, unsigned oriTShape0, unsigned oriTShape1,
          int isAcc>
TILEOP void DynL0CCopyOut(__gm__ GMT* dst, __cc__ L0CT* src, unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0,
                          unsigned GmOffset1, int uf)
{
    SetAtomicAddition<GMT>();
    DynL0CCopyOut<GMT, L0CT, TShape0, TShape1, oriTShape0, oriTShape1>(dst, src, GmShape0, GmShape1, GmOffset0,
                                                                       GmOffset1, uf);
    if constexpr (isAcc == 1) {
        set_atomic_none();
    }
}
} // namespace TileOp
#endif // TILE_FWK_CUBE_DYN_H
