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
 * \file vector_una.h
 * \brief
 */

// dim2 & dim1 (T0 = 1 for dim1)
template <typename T, unsigned T0, unsigned T1, unsigned DS, unsigned SS>
TILEOP void T_UNA(__ubuf__ T* dst, __ubuf__ T* src)
{
    constexpr unsigned simdw = REPEAT_BYTE / sizeof(T);
    constexpr unsigned numRepeatPerLine = T1 / simdw;
    constexpr unsigned numRemainPerLine = T1 % simdw;
    constexpr unsigned nElemPerBlock = BLOCK_SIZE / sizeof(T);

    if constexpr (numRepeatPerLine > 0) {
        constexpr unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        constexpr unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < T0; i++) {
            if constexpr (numLoop) {
                for (int j = 0; j < numLoop; j++) {
                    V_UNA_FUNC(dst + i * DS + j * simdw * REPEAT_MAX, src + i * SS + j * simdw * REPEAT_MAX, REPEAT_MAX,
                               1, 1, 8, 8);
                }
            }
            if constexpr (remainAfterLoop) {
                V_UNA_FUNC(dst + i * DS + simdw * REPEAT_MAX * numLoop, src + i * SS + simdw * REPEAT_MAX * numLoop,
                           remainAfterLoop, 1, 1, 8, 8);
            }
        }
    }

    // shift to deal with tail
    dst += numRepeatPerLine * simdw;
    src += numRepeatPerLine * simdw;

    if constexpr (numRemainPerLine) {
        constexpr unsigned numLoop = T0 / REPEAT_MAX;
        constexpr unsigned remainAfterLoop = T0 % REPEAT_MAX;
        bool strideOverFlag = (DS / nElemPerBlock > REPEAT_STRIDE_MAX) || (SS / nElemPerBlock > REPEAT_STRIDE_MAX);
        SetContinuousMask(numRemainPerLine);
        if constexpr (numLoop) {
            for (int i = 0; i < numLoop; i++) {
                if (strideOverFlag) {
                    for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                        V_UNA_FUNC(dst + i * REPEAT_MAX * DS + j * DS, src + i * REPEAT_MAX * SS + j * SS, 1, 1, 1, 1,
                                   1);
                    }
                } else {
                    V_UNA_FUNC(dst + i * REPEAT_MAX * DS, src + i * REPEAT_MAX * SS, REPEAT_MAX, 1, 1,
                               DS / nElemPerBlock, SS / nElemPerBlock);
                }
            }
        }
        if constexpr (remainAfterLoop) {
            if (strideOverFlag) {
                for (unsigned j = 0; j < remainAfterLoop; j++) {
                    V_UNA_FUNC(dst + numLoop * REPEAT_MAX * DS + j * DS, src + numLoop * REPEAT_MAX * SS + j * SS, 1, 1,
                               1, 1, 1);
                }
            } else {
                V_UNA_FUNC(dst + numLoop * REPEAT_MAX * DS, src + numLoop * REPEAT_MAX * SS, remainAfterLoop, 1, 1,
                           DS / nElemPerBlock, SS / nElemPerBlock);
            }
        }
        set_vector_mask(-1, -1);
    }
}

// dim3
template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned DS0, unsigned DS1, unsigned SS0, unsigned SS1>
TILEOP void T_UNA(__ubuf__ T* dst, __ubuf__ T* src)
{
    static_assert((DS1 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((SS1 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        T_UNA<T, T1, T2, DS1, SS1>(dst, src);
        dst += DS0 * DS1;
        src += SS0 * SS1;
    }
}

// dim4
template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned DS0, unsigned DS1, unsigned DS2,
          unsigned SS0, unsigned SS1, unsigned SS2>
TILEOP void T_UNA(__ubuf__ T* dst, __ubuf__ T* src)
{
    static_assert((DS2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((SS2 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src_ = src;
        for (int j = 0; j < T1; j++) {
            T_UNA<T, T2, T3, DS2, SS2>(dst_, src_);
            dst_ += DS1 * DS2;
            src_ += SS1 * SS2;
        }
        dst += DS0 * DS1 * DS2;
        src += SS0 * SS1 * SS2;
    }
}
