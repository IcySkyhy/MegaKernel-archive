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
 * \file vector_bin_brc.h
 * \brief
 */

// dim2 & dim1 (T0 = 1 for dim1)
template <typename T, unsigned T0, unsigned T1, unsigned DS, unsigned SS0, unsigned SS1, bool isInputForceCombineAxis>
TILEOP void T_BIN_BRC(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, __ubuf__ T* tmp)
{
    constexpr unsigned elementsPerBlock = BLOCK_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned numRepeatPerLine = T1 / elementsPerRepeat;
    unsigned numRemainPerLine = T1 % elementsPerRepeat;
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t src0BlockStride = 1;

    __ubuf__ T* sourceData = nullptr;
    if (isInputForceCombineAxis) {
        constexpr unsigned brcPerRepeat = 8;
        vbrcb((__ubuf__ uint32_t*)tmp, (__ubuf__ uint32_t*)src1, 1, 8, (T0 + brcPerRepeat - 1) / brcPerRepeat);
        sourceData = tmp;
    } else {
        SetContinuousMask(8);
        for (int i = 0; i < T0; i++) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            T tmpVal = (T)(*(src1 + i * SS1));
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
            vector_dup(src1 + i * SS1, tmpVal, 1, 1, 0, 0, 0);
        }
        set_vector_mask(-1, -1);
        sourceData = src1;
    }
    pipe_barrier(PIPE_V);

    constexpr uint64_t repeatStride = SS0 / elementsPerBlock;

    if (repeatStride < 256 && T0 < 256) {
        for (uint8_t i = 0; i < numRepeatPerLine; i++) {
            V_BIN_BRC_FUNC(dst + i * elementsPerRepeat, src0 + i * elementsPerRepeat, sourceData, T0, dstBlockStride,
                           src0BlockStride, 0, repeatStride, repeatStride, 1);
        }
        if (numRemainPerLine > 0) {
            SetContinuousMask(numRemainPerLine);
            V_BIN_BRC_FUNC(dst + numRepeatPerLine * elementsPerRepeat, src0 + numRepeatPerLine * elementsPerRepeat,
                           sourceData, T0, dstBlockStride, src0BlockStride, 0, repeatStride, repeatStride, 1);
            set_vector_mask(-1, -1);
        }
    } else {
        if (numRepeatPerLine > 0) {
            unsigned numRepeatStride = numRepeatPerLine / REPEAT_MAX;
            unsigned remainStrideAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (int j = 0; j < T0; j++) {
                if (numRepeatStride) {
                    for (int k = 0; k < numRepeatStride; k++) {
                        V_BIN_BRC_FUNC(dst + j * DS + k * elementsPerRepeat * REPEAT_MAX,
                                       src0 + j * SS0 + k * elementsPerRepeat * REPEAT_MAX, sourceData, REPEAT_MAX,
                                       dstBlockStride, src0BlockStride, 0, 8, 8, 1);
                    }
                }
                if (remainStrideAfterLoop) {
                    V_BIN_BRC_FUNC(dst + j * DS + numRepeatStride * elementsPerRepeat * REPEAT_MAX,
                                   src0 + j * SS0 + numRepeatStride * elementsPerRepeat * REPEAT_MAX, sourceData,
                                   remainStrideAfterLoop, dstBlockStride, src0BlockStride, 0, 8, 8, 1);
                }
            }
        }
        if (numRemainPerLine > 0) {
            SetContinuousMask(numRemainPerLine);
            for (int j = 0; j < T0; j++) {
                V_BIN_BRC_FUNC(dst + j * DS + numRepeatPerLine * elementsPerRepeat,
                               src0 + j * SS0 + numRepeatPerLine * elementsPerRepeat, sourceData, 1, dstBlockStride,
                               src0BlockStride, 0, 8, 8, 1);
            }
            set_vector_mask(-1, -1);
        }
    }
}

// dim3
template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned DS0, unsigned DS1, unsigned S0S0, unsigned S0S1,
          unsigned S1S0, unsigned S1S1, bool isInputForceCombineAxis>
TILEOP void T_BIN_BRC(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, __ubuf__ T* tmp)
{
    static_assert((DS1 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S0S1 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S1S1 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        T_BIN_BRC<T, T1, T2, DS1, S0S1, S1S1, isInputForceCombineAxis>(dst, src0, src1, tmp);
        dst += DS0 * DS1;
        src0 += S0S0 * S0S1;
        src1 += S1S0 * S1S1;
    }
}

// dim4
template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned DS0, unsigned DS1, unsigned DS2,
          unsigned S0S0, unsigned S0S1, unsigned S0S2, unsigned S1S0, unsigned S1S1, unsigned S1S2,
          bool isInputForceCombineAxis>
TILEOP void T_BIN_BRC(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, __ubuf__ T* tmp)
{
    static_assert((DS2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S0S2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S1S2 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src0_ = src0;
        __ubuf__ T* src1_ = src1;
        __ubuf__ T* tmp_ = tmp;
        for (int j = 0; j < T1; j++) {
            T_BIN_BRC<T, T2, T3, DS2, S0S2, S1S2, isInputForceCombineAxis>(dst_, src0_, src1_, tmp_);
            dst_ += DS1 * DS2;
            src0_ += S0S1 * S0S2;
            src1_ += S1S1 * S1S2;
        }
        dst += DS0 * DS1 * DS2;
        src0 += S0S0 * S0S1 * S0S2;
        src1 += S1S0 * S1S1 * S1S2;
    }
}
