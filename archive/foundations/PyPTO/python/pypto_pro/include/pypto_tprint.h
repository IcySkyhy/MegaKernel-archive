/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Binary-mode TPRINT implementation using cce::printf (replaces pto-isa's _DEBUG-guarded TPRINT).
#pragma once
#include <cstdint>
// cce::printf is only declared by the toolchain when the print mode is enabled (jit emits
// -DPTOAS_ENABLE_CCE_PRINT=1 only for kernels that actually print). Kernels without dump
// calls keep a no-op macro so the header compiles in plain mode too.
#ifndef __PYPTO_PRINTF
#if defined(PTOAS_ENABLE_CCE_PRINT) && PTOAS_ENABLE_CCE_PRINT
#define __PYPTO_PRINTF(...) cce::printf(__VA_ARGS__)
#else
#define __PYPTO_PRINTF(...) ((void)0)
#endif
#endif
template <typename V>
__aicore__ inline const __gm__ char* __pypto_dtype_name()
{
    return "unknown";
}
template <>
__aicore__ inline const __gm__ char* __pypto_dtype_name<float>()
{
    return "float32";
}
template <>
__aicore__ inline const __gm__ char* __pypto_dtype_name<half>()
{
    return "float16";
}
template <>
__aicore__ inline const __gm__ char* __pypto_dtype_name<int32_t>()
{
    return "int32";
}
template <>
__aicore__ inline const __gm__ char* __pypto_dtype_name<uint32_t>()
{
    return "uint32";
}
template <>
__aicore__ inline const __gm__ char* __pypto_dtype_name<int16_t>()
{
    return "int16";
}
template <>
__aicore__ inline const __gm__ char* __pypto_dtype_name<uint16_t>()
{
    return "uint16";
}
template <>
__aicore__ inline const __gm__ char* __pypto_dtype_name<int8_t>()
{
    return "int8";
}
template <>
__aicore__ inline const __gm__ char* __pypto_dtype_name<uint8_t>()
{
    return "uint8";
}

template <typename V>
__aicore__ inline void __pypto_print_val(V val)
{
    if constexpr (std::is_same_v<V, float>) {
        __PYPTO_PRINTF("%f ", val);
    } else if constexpr (std::is_same_v<V, half>) {
        __PYPTO_PRINTF("%f ", (float)val);
    } else if constexpr (std::is_signed_v<V>) {
        __PYPTO_PRINTF("%d ", (int)val);
    } else if constexpr (std::is_unsigned_v<V>) {
        __PYPTO_PRINTF("%u ", (unsigned int)val);
    } else if constexpr (sizeof(V) == 2) {
        // bf16: upper 16 bits of float32
        uint32_t fbits = (uint32_t)(*(uint16_t*)&val) << 16;
        __PYPTO_PRINTF("%f ", *(float*)&fbits);
    } else if constexpr (sizeof(V) == 1) {
        // fp8: print raw byte value
        __PYPTO_PRINTF("%u ", (unsigned int)(*(uint8_t*)&val));
    } else {
        __PYPTO_PRINTF("? ");
    }
}

template <typename T>
__aicore__ inline void __pypto_tprint(T& src)
{
    pipe_barrier(PIPE_ALL);
    if constexpr (pto::is_global<T>::value) {
        using ElemType = typename T::RawDType;
        int n[5], s[5];
        for (int d = 0; d < 5; d++) {
            n[d] = src.GetShape(d);
            s[d] = src.GetStride(d);
        }
        auto* dataPtr = src.data();
        if constexpr (T::layout == pto::Layout::ND || T::layout == pto::Layout::DN) {
            __PYPTO_PRINTF("=== [dump_tensor] dtype: %s, Layout: %s, shape=[%d,%d,%d,%d,%d] ===\n",
                            __pypto_dtype_name<ElemType>(), T::layout == pto::Layout::ND ? "ND" : "DN", n[0], n[1],
                            n[2], n[3], n[4]);
            for (int i0 = 0; i0 < n[0]; ++i0)
                for (int i1 = 0; i1 < n[1]; ++i1)
                    for (int i2 = 0; i2 < n[2]; ++i2) {
                        __PYPTO_PRINTF("  Batch [%d, %d, %d]:\n", i0, i1, i2);
                        for (int r = 0; r < n[3]; ++r) {
                            for (int c = 0; c < n[4]; ++c) {
                                int64_t off = (int64_t)i0 * s[0] + i1 * s[1] + i2 * s[2] + r * s[3] + c * s[4];
                                __pypto_print_val(dataPtr[off]);
                            }
                            __PYPTO_PRINTF("\n");
                        }
                    }
        } else if constexpr (T::layout == pto::Layout::NZ) {
            int logical_rows = n[2] * n[3];
            int logical_cols = n[1] * n[4];
            __PYPTO_PRINTF("=== [dump_tensor] dtype: %s, Layout: NZ, logical shape=[%d,%d] ===\n",
                            __pypto_dtype_name<ElemType>(), logical_rows, logical_cols);
            for (int r = 0; r < logical_rows; ++r) {
                for (int c = 0; c < logical_cols; ++c) {
                    int br = r / n[3], ir = r % n[3];
                    int bc = c / n[4], ic = c % n[4];
                    int64_t off = (int64_t)br * s[2] + bc * s[1] + ir * s[3] + ic * s[4];
                    __pypto_print_val(dataPtr[off]);
                }
                __PYPTO_PRINTF("\n");
            }
        }
    } else if constexpr (pto::is_tile<T>::value) {
        using DType = typename T::DType;
        int validRows = src.GetValidRow();
        int validCols = src.GetValidCol();
        __PYPTO_PRINTF("=== [dump_tile] dtype: %s, shape=[%d,%d], valid=[%d,%d], Layout: %s ===\n",
                        __pypto_dtype_name<DType>(), T::Rows, T::Cols, validRows, validCols,
                        pto::GetLayoutName(T::BFractal, T::SFractal));
        for (int r = 0; r < T::Rows; ++r) {
            for (int c = 0; c < T::Cols; ++c) {
                int off = (T::BFractal == pto::BLayout::RowMajor) ? r * T::Cols + c : c * T::Rows + r;
                __pypto_print_val(src.GetValue(off));
                if (c == validCols - 1 && validCols < T::Cols)
                    __PYPTO_PRINTF("| ");
            }
            __PYPTO_PRINTF("\n");
            if (r == validRows - 1 && validRows < T::Rows)
                __PYPTO_PRINTF("--------\n");
        }
    }
}
#define TPRINT(x) __pypto_tprint(x)
