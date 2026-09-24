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
 * \file vector_bin_dyn.h
 * \brief
 */

#include <float.h>

/* ------------------------------------- support unaligned scene -------------------------------------*/

// dim2 & dim1 (T0 = 1 for dim1)
template <typename T, unsigned DS, unsigned SS0, unsigned SS1>
TILEOP void T_BIN_PAIR(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0T0, unsigned src0T1,
                       unsigned src1T0, unsigned src1T1)
{
    unsigned T0 = src0T0 < src1T0 ? src0T0 : src1T0;
    unsigned T1 = src0T1 < src1T1 ? src0T1 : src1T1;
    if (src0T0 != src1T0) {
        __ubuf__ T* src = src0T0 > src1T0 ? src0 : src1;
        unsigned srcT0 = src0T0 > src1T0 ? src0T0 : src1T0;
        uint16_t lenBurst = (src0T1 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint16_t srcGap = SS0 * sizeof(T) / BLOCK_SIZE - lenBurst;
        uint16_t dstGap = DS * sizeof(T) / BLOCK_SIZE - lenBurst;
        copy_ubuf_to_ubuf(dst, src, 0, srcT0, lenBurst, srcGap, dstGap);
        pipe_barrier(PIPE_V);
    } else if (src0T1 != src1T1) {
        __ubuf__ T* src = src0T1 > src1T1 ? src0 : src1;
        unsigned srcT1 = src0T1 > src1T1 ? src0T1 : src1T1;
        uint16_t lenBurst = (srcT1 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint16_t srcSS = src0T1 > src1T1 ? SS0 : SS1;
        uint16_t srcGap = srcSS * sizeof(T) / BLOCK_SIZE - lenBurst;
        uint16_t dstGap = DS * sizeof(T) / BLOCK_SIZE - lenBurst;
        copy_ubuf_to_ubuf(dst, src, 0, src0T0, lenBurst, srcGap, dstGap);
        pipe_barrier(PIPE_V);
    }
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned numRepeatPerLine = T1 / elementsPerRepeat;
    unsigned numRemainPerLine = T1 % elementsPerRepeat;
    constexpr unsigned blockSizeElem = BLOCK_SIZE / sizeof(T);

    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < T0; i++) {
            if (numLoop) {
                for (int j = 0; j < numLoop; j++) {
                    V_BIN_FUNC(dst + i * DS + j * elementsPerRepeat * REPEAT_MAX,
                               src0 + i * SS0 + j * elementsPerRepeat * REPEAT_MAX,
                               src1 + i * SS1 + j * elementsPerRepeat * REPEAT_MAX, REPEAT_MAX, 1, 1, 1, 8, 8, 8);
                }
            }
            if (remainAfterLoop) {
                V_BIN_FUNC(dst + i * DS + numLoop * elementsPerRepeat * REPEAT_MAX,
                           src0 + i * SS0 + numLoop * elementsPerRepeat * REPEAT_MAX,
                           src1 + i * SS1 + numLoop * elementsPerRepeat * REPEAT_MAX, remainAfterLoop, 1, 1, 1, 8, 8,
                           8);
            }
        }
    }

    // shift to deal with tail
    dst += numRepeatPerLine * elementsPerRepeat;
    src0 += numRepeatPerLine * elementsPerRepeat;
    src1 += numRepeatPerLine * elementsPerRepeat;

    if (numRemainPerLine) {
        unsigned numLoop = T0 / REPEAT_MAX;
        unsigned remainAfterLoop = T0 % REPEAT_MAX;
        constexpr bool strideOverFlag = (DS / blockSizeElem > REPEAT_STRIDE_MAX) ||
                                        (SS0 / blockSizeElem > REPEAT_STRIDE_MAX) ||
                                        (SS1 / blockSizeElem > REPEAT_STRIDE_MAX);
        SetContinuousMask(numRemainPerLine);
        if (numLoop) {
            for (int i = 0; i < numLoop; i++) {
                if constexpr (strideOverFlag) {
                    for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                        V_BIN_FUNC(dst + i * REPEAT_MAX * DS + j * DS, src0 + i * REPEAT_MAX * SS0 + j * SS0,
                                   src1 + i * REPEAT_MAX * SS1 + j * SS1, 1, 1, 1, 1, 1, 1, 1);
                    }
                } else {
                    V_BIN_FUNC(dst + i * REPEAT_MAX * DS, src0 + i * REPEAT_MAX * SS0, src1 + i * REPEAT_MAX * SS1,
                               REPEAT_MAX, 1, 1, 1, DS / blockSizeElem, SS0 / blockSizeElem, SS1 / blockSizeElem);
                }
            }
        }
        if (remainAfterLoop) {
            if constexpr (strideOverFlag) {
                for (unsigned j = 0; j < remainAfterLoop; j++) {
                    V_BIN_FUNC(dst + numLoop * REPEAT_MAX * DS + j * DS, src0 + numLoop * REPEAT_MAX * SS0 + j * SS0,
                               src1 + numLoop * REPEAT_MAX * SS1 + j * SS1, 1, 1, 1, 1, 1, 1, 1);
                }
            } else {
                V_BIN_FUNC(dst + numLoop * REPEAT_MAX * DS, src0 + numLoop * REPEAT_MAX * SS0,
                           src1 + numLoop * REPEAT_MAX * SS1, remainAfterLoop, 1, 1, 1, DS / blockSizeElem,
                           SS0 / blockSizeElem, SS1 / blockSizeElem);
            }
        }
        set_vector_mask(-1, -1);
    }
}

// dim4
template <typename T, unsigned DS0, unsigned DS1, unsigned DS2, unsigned DS3, unsigned S0S0, unsigned S0S1,
          unsigned S0S2, unsigned S0S3, unsigned S1S0, unsigned S1S1, unsigned S1S2, unsigned S1S3>
TILEOP void T_BIN_PAIR(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0T0, unsigned src0T1,
                       unsigned src0T2, unsigned src0T3, unsigned src1T0, unsigned src1T1, unsigned src1T2,
                       unsigned src1T3)
{
    static_assert((DS3 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S0S3 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S1S3 * sizeof(T)) % BLOCK_SIZE == 0);
    if ((src0T0 != 0 && src0T1 != 0 && src0T2 != 0 && src0T3 != 0) &&
        (src1T0 == 0 || src1T1 == 0 || src1T2 == 0 || src1T3 == 0)) {
        uint16_t lenBurst = (S0S1 * S0S2 * S0S3 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        copy_ubuf_to_ubuf(dst, src0, 0, S0S0, lenBurst, 0, 0);
        pipe_barrier(PIPE_V);
        return;
    } else if ((src0T0 == 0 || src0T1 == 0 || src0T2 == 0 || src0T3 == 0) &&
               (src1T0 != 0 && src1T1 != 0 && src1T2 != 0 && src1T3 != 0)) {
        uint16_t lenBurst = (S1S1 * S1S2 * S1S3 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        copy_ubuf_to_ubuf(dst, src1, 0, S1S0, lenBurst, 0, 0);
        pipe_barrier(PIPE_V);
        return;
    } else if ((src0T0 == 0 || src0T1 == 0 || src0T2 == 0 || src0T3 == 0) &&
               (src1T0 == 0 || src1T1 == 0 || src1T2 == 0 || src1T3 == 0)) {
        return;
    }
    if ((S0S0 != 1 || S1S0 != 1) && src0T0 != src1T0) {
        constexpr unsigned MRGDS = DS1 * DS2 * DS3;
        constexpr unsigned MRGS0 = S0S1 * S0S2 * S0S3;
        constexpr unsigned MRGS1 = S1S1 * S1S2 * S1S3;
        unsigned src0Mrg = S0S1 * S0S2 * S0S3;
        unsigned src1Mrg = S1S1 * S1S2 * S1S3;
        T_BIN_PAIR<T, MRGDS, MRGS0, MRGS1>(dst, src0, src1, src0T0, src0Mrg, src1T0, src1Mrg);
        pipe_barrier(PIPE_V);
        return;
    }
    if ((S0S1 != 1 || S1S1 != 1) && src0T1 != src1T1) {
        constexpr unsigned MRGDS = DS2 * DS3;
        constexpr unsigned MRGS0 = S0S2 * S0S3;
        constexpr unsigned MRGS1 = S1S2 * S1S3;
        unsigned src0Mrg = S0S2 * S0S3;
        unsigned src1Mrg = S1S2 * S1S3;
        for (int i = 0; i < src0T0; i++) {
            T_BIN_PAIR<T, MRGDS, MRGS0, MRGS1>(dst, src0, src1, src0T1, src0Mrg, src1T1, src1Mrg);
            dst += DS1 * DS2 * DS3;
            src0 += S0S1 * S0S2 * S0S3;
            src1 += S1S1 * S1S2 * S1S3;
            pipe_barrier(PIPE_V);
        }
        return;
    }
    for (int i = 0; i < src0T0; i++) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src0_ = src0;
        __ubuf__ T* src1_ = src1;
        for (int j = 0; j < src0T1; j++) {
            T_BIN_PAIR<T, DS3, S0S3, S1S3>(dst_, src0_, src1_, src0T2, src0T3, src1T2, src1T3);
            dst_ += DS2 * DS3;
            src0_ += S0S2 * S0S3;
            src1_ += S1S2 * S1S3;
            pipe_barrier(PIPE_V);
        }
        dst += DS1 * DS2 * DS3;
        src0 += S0S1 * S0S2 * S0S3;
        src1 += S1S1 * S1S2 * S1S3;
    }
}

// dim2 & dim1 (T0 = 1 for dim1)
template <typename T, unsigned DS, unsigned S0S0, unsigned S0S1, unsigned S1S0, unsigned S1S1,
          BroadcastOperand OPERAND = BroadcastOperand::NONE>
TILEOP void T_BIN(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0T0, unsigned src0T1,
                  unsigned src1T0, unsigned src1T1)
{
    unsigned T0 = src0T0 < src1T0 ? src1T0 : src0T0;
    unsigned T1 = src0T1 < src1T1 ? src1T1 : src0T1;

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned numRepeatPerLine = T1 / elementsPerRepeat;
    unsigned numRemainPerLine = T1 % elementsPerRepeat;
    constexpr unsigned blockSizeElem = BLOCK_SIZE / sizeof(T);
    constexpr unsigned src0Row = (S0S0 == 1 && S1S0 != 1) ? 0 : S0S1;
    constexpr unsigned src1Row = (S1S0 == 1 && S0S0 != 1) ? 0 : S1S1;
    constexpr bool strideOverFlag = (DS / blockSizeElem > REPEAT_STRIDE_MAX) ||
                                    (src0Row / blockSizeElem > REPEAT_STRIDE_MAX) ||
                                    (src1Row / blockSizeElem > REPEAT_STRIDE_MAX);
    unsigned src0BlockStride = 1;
    unsigned src1BlockStride = 1;
    unsigned src0RepeatStride = 8;
    unsigned src1RepeatStride = 8;
    unsigned src0RowOffset = elementsPerRepeat;
    unsigned src1RowOffset = elementsPerRepeat;

    if constexpr (OPERAND == BroadcastOperand::LEFT_OPERAND) {
        src0BlockStride = 0;
        src0RepeatStride = 0;
        src0RowOffset = 0;
    }

    if constexpr (OPERAND == BroadcastOperand::RIGHT_OPERAND) {
        src1BlockStride = 0;
        src1RepeatStride = 0;
        src1RowOffset = 0;
    }

    if (numRepeatPerLine > 0) {
        if (numRepeatPerLine >= T0 || strideOverFlag) {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (int i = 0; i < T0; i++) {
                if (numLoop) {
                    for (int j = 0; j < numLoop; j++) {
                        V_BIN_FUNC(dst + i * DS + j * elementsPerRepeat * REPEAT_MAX,
                                   src0 + i * src0Row + j * src0RowOffset * REPEAT_MAX,
                                   src1 + i * src1Row + j * src1RowOffset * REPEAT_MAX, REPEAT_MAX, 1, src0BlockStride,
                                   src1BlockStride, 8, src0RepeatStride, src1RepeatStride);
                    }
                }
                if (remainAfterLoop) {
                    V_BIN_FUNC(dst + i * DS + numLoop * elementsPerRepeat * REPEAT_MAX,
                               src0 + i * src0Row + numLoop * src0RowOffset * REPEAT_MAX,
                               src1 + i * src1Row + numLoop * src1RowOffset * REPEAT_MAX, remainAfterLoop, 1,
                               src0BlockStride, src1BlockStride, 8, src0RepeatStride, src1RepeatStride);
                }
            }
        } else {
            if constexpr (!strideOverFlag) {
                // 沿着T0方向开Repeat
                unsigned numLoop = T0 / REPEAT_MAX;
                unsigned remainAfterLoop = T0 % REPEAT_MAX;
                for (int i = 0; i < numRepeatPerLine; i++) {
                    if (numLoop) {
                        for (int j = 0; j < numLoop; j++) {
                            V_BIN_FUNC(dst + i * elementsPerRepeat + j * REPEAT_MAX * DS,
                                       src0 + i * src0RowOffset + j * src0Row * REPEAT_MAX,
                                       src1 + i * src1RowOffset + j * src1Row * REPEAT_MAX, REPEAT_MAX, 1,
                                       src0BlockStride, src1BlockStride, DS / blockSizeElem, src0Row / blockSizeElem,
                                       src1Row / blockSizeElem);
                        }
                    }
                    if (remainAfterLoop) {
                        V_BIN_FUNC(dst + i * elementsPerRepeat + numLoop * REPEAT_MAX * DS,
                                   src0 + i * src0RowOffset + numLoop * src0Row * REPEAT_MAX,
                                   src1 + i * src1RowOffset + numLoop * src1Row * REPEAT_MAX, remainAfterLoop, 1,
                                   src0BlockStride, src1BlockStride, DS / blockSizeElem, src0Row / blockSizeElem,
                                   src1Row / blockSizeElem);
                    }
                }
            }
        }
    }

    // shift to deal with tail
    dst += numRepeatPerLine * elementsPerRepeat;
    src0 += numRepeatPerLine * src0RowOffset;
    src1 += numRepeatPerLine * src1RowOffset;

    if (numRemainPerLine) {
        unsigned numLoop = T0 / REPEAT_MAX;
        unsigned remainAfterLoop = T0 % REPEAT_MAX;
        SetContinuousMask(numRemainPerLine);
        if (numLoop) {
            for (int i = 0; i < numLoop; i++) {
                if constexpr (strideOverFlag) {
                    for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                        V_BIN_FUNC(dst + i * REPEAT_MAX * DS + j * DS, src0 + i * REPEAT_MAX * src0Row + j * src0Row,
                                   src1 + i * REPEAT_MAX * src1Row + j * src1Row, 1, 1, src0BlockStride,
                                   src1BlockStride, 1, 1, 1);
                    }
                } else {
                    V_BIN_FUNC(dst + i * REPEAT_MAX * DS, src0 + i * REPEAT_MAX * src0Row,
                               src1 + i * REPEAT_MAX * src1Row, REPEAT_MAX, 1, src0BlockStride, src1BlockStride,
                               DS / blockSizeElem, src0Row / blockSizeElem, src1Row / blockSizeElem);
                }
            }
        }
        if (remainAfterLoop) {
            if constexpr (strideOverFlag) {
                for (unsigned j = 0; j < remainAfterLoop; j++) {
                    V_BIN_FUNC(dst + numLoop * REPEAT_MAX * DS + j * DS,
                               src0 + numLoop * REPEAT_MAX * src0Row + j * src0Row,
                               src1 + numLoop * REPEAT_MAX * src1Row + j * src1Row, 1, 1, src0BlockStride,
                               src1BlockStride, 1, 1, 1);
                }
            } else {
                V_BIN_FUNC(dst + numLoop * REPEAT_MAX * DS, src0 + numLoop * REPEAT_MAX * src0Row,
                           src1 + numLoop * REPEAT_MAX * src1Row, remainAfterLoop, 1, src0BlockStride, src1BlockStride,
                           DS / blockSizeElem, src0Row / blockSizeElem, src1Row / blockSizeElem);
            }
        }
        set_vector_mask(-1, -1);
    }
}

// dim4
template <typename T, unsigned DS0, unsigned DS1, unsigned DS2, unsigned DS3, unsigned S0S0, unsigned S0S1,
          unsigned S0S2, unsigned S0S3, unsigned S1S0, unsigned S1S1, unsigned S1S2, unsigned S1S3,
          BroadcastOperand OPERAND = BroadcastOperand::NONE>
TILEOP void T_BIN(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0T0, unsigned src0T1,
                  unsigned src0T2, unsigned src0T3, unsigned src1T0, unsigned src1T1, unsigned src1T2, unsigned src1T3)
{
    static_assert((DS3 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S0S3 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S1S3 * sizeof(T)) % BLOCK_SIZE == 0);
    if (src0T0 == 0 || src0T1 == 0 || src0T2 == 0 || src0T3 == 0 || src1T0 == 0 || src1T1 == 0 || src1T2 == 0 ||
        src1T3 == 0) {
        return;
    }

    unsigned T0 = src0T0 < src1T0 ? src1T0 : src0T0;
    unsigned T1 = src0T1 < src1T1 ? src1T1 : src0T1;
    for (int i = 0; i < T0; i++) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src0_ = src0;
        __ubuf__ T* src1_ = src1;
        for (int j = 0; j < T1; j++) {
            T_BIN<T, DS3, S0S2, S0S3, S1S2, S1S3, OPERAND>(dst_, src0_, src1_, src0T2, src0T3, src1T2, src1T3);
            dst_ += DS2 * DS3;
            src0_ += (S0S1 == 1 && S1S1 != 1) ? 0 : S0S2 * S0S3;
            src1_ += (S0S1 != 1 && S1S1 == 1) ? 0 : S1S2 * S1S3;
        }
        dst += DS1 * DS2 * DS3;
        src0 += (S0S0 == 1 && S1S0 != 1) ? 0 : S0S1 * S0S2 * S0S3;
        src1 += (S0S0 != 1 && S1S0 == 1) ? 0 : S1S1 * S1S2 * S1S3;
    }
}

template <typename T, unsigned DS, unsigned SS0>
TILEOP void T_BIN_VS(__ubuf__ T* dst, __ubuf__ T* src0, T src1, unsigned T0, unsigned T1)
{
#ifdef VS_SUB
    src1 = static_cast<float>(src1) * (-1);
#endif
#ifdef VS_DIV
    if (static_cast<float>(src1) != 0) {
        src1 = (float)1.0 / static_cast<float>(src1);
    } else {
        src1 = FLT_MAX;
    }
#endif
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned numRepeatPerLine = T1 / elementsPerRepeat;
    unsigned numRemainPerLine = T1 % elementsPerRepeat;
    constexpr unsigned blockSizeElem = BLOCK_SIZE / sizeof(T);

    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < T0; i++) {
            if (numLoop) {
                for (int j = 0; j < numLoop; j++) {
                    V_BIN_FUNC_VS(dst + i * DS + j * elementsPerRepeat * REPEAT_MAX,
                                  src0 + i * SS0 + j * elementsPerRepeat * REPEAT_MAX, src1, REPEAT_MAX, 1, 1, 8, 8);
                }
            }
            if (remainAfterLoop) {
                V_BIN_FUNC_VS(dst + i * DS + elementsPerRepeat * REPEAT_MAX * numLoop,
                              src0 + i * SS0 + elementsPerRepeat * REPEAT_MAX * numLoop, src1, remainAfterLoop, 1, 1, 8,
                              8);
            }
        }
    }

    // shift to deal with tail
    dst += numRepeatPerLine * elementsPerRepeat;
    src0 += numRepeatPerLine * elementsPerRepeat;

    if (numRemainPerLine) {
        unsigned numLoop = T0 / REPEAT_MAX;
        unsigned remainAfterLoop = T0 % REPEAT_MAX;
        constexpr bool strideOverFlag = (DS / blockSizeElem > REPEAT_STRIDE_MAX) ||
                                        (SS0 / blockSizeElem > REPEAT_STRIDE_MAX);
        SetContinuousMask(numRemainPerLine);
        if (numLoop) {
            for (int i = 0; i < numLoop; i++) {
                if constexpr (strideOverFlag) {
                    for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                        V_BIN_FUNC_VS(dst + i * REPEAT_MAX * DS + j * DS, src0 + i * REPEAT_MAX * SS0 + j * SS0, src1,
                                      1, 1, 1, 1, 1);
                    }
                } else {
                    V_BIN_FUNC_VS(dst + i * REPEAT_MAX * DS, src0 + i * REPEAT_MAX * SS0, src1, REPEAT_MAX, 1, 1,
                                  DS / blockSizeElem, SS0 / blockSizeElem);
                }
            }
        }
        if (remainAfterLoop) {
            if constexpr (strideOverFlag) {
                for (unsigned j = 0; j < remainAfterLoop; j++) {
                    V_BIN_FUNC_VS((__ubuf__ T*)(dst + numLoop * REPEAT_MAX * DS + j * DS),
                                  src0 + numLoop * REPEAT_MAX * SS0 + j * SS0, src1, 1, 1, 1, 1, 1);
                }
            } else {
                V_BIN_FUNC_VS((__ubuf__ T*)(dst + numLoop * REPEAT_MAX * DS), src0 + numLoop * REPEAT_MAX * SS0, src1,
                              remainAfterLoop, 1, 1, DS / blockSizeElem, SS0 / blockSizeElem);
            }
        }
        set_vector_mask(-1, -1);
    }
}

// dim4
template <typename T, unsigned DS0, unsigned DS1, unsigned DS2, unsigned S0S0, unsigned S0S1, unsigned S0S2>
TILEOP void T_BIN_VS(__ubuf__ T* dst, __ubuf__ T* src0, T src1, unsigned T0, unsigned T1, unsigned T2, unsigned T3)
{
    static_assert((DS2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S0S2 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ T* dst_ = dst;
        __ubuf__ T* src0_ = src0;
        for (int j = 0; j < T1; j++) {
            T_BIN_VS<T, DS2, S0S2>(dst_, src0_, src1, T2, T3);
            dst_ += DS1 * DS2;
            src0_ += S0S1 * S0S2;
        }
        dst += DS0 * DS1 * DS2;
        src0 += S0S0 * S0S1 * S0S2;
    }
}
