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
 * \file aicore_print.h
 * \brief
 */

#pragma once

#include "aicore_print_logger.h"

// ============================================================================
// Internal Helper Functions
// ============================================================================

template <typename T>
INLINE void DispatchPrint(LogContext* ctx, __gm__ const char** fmt, T val)
{
    if constexpr (std::is_integral_v<T>) {
        ctx->PrintInt64(ctx, fmt, static_cast<int64_t>(val));
    } else if constexpr (std::is_floating_point_v<T>) {
        ctx->PrintFp32(ctx, fmt, static_cast<float>(val));
    } else if constexpr (std::is_pointer_v<T>) {
        ctx->PrintInt64(ctx, fmt, reinterpret_cast<int64_t>(val));
#if IS_AICORE
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        ctx->PrintBf16(ctx, fmt, SafeBitCast<uint16_t>(val));
    } else if constexpr (std::is_same_v<T, half>) {
        ctx->PrintFp16(ctx, fmt, SafeBitCast<uint16_t>(val));
#if SUPPORT_FP8_HF8_PRINT
    } else if constexpr (std::is_same_v<T, float8_e4m3_t>) {
        ctx->PrintFp8E4M3(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, float8_e5m2_t>) {
        ctx->PrintFp8E5M2(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, float8_e8m0_t>) {
        ctx->PrintFp8E8M0(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, hifloat8_t>) {
        ctx->PrintHf8(ctx, fmt, SafeBitCast<uint8_t>(val));
#endif
#endif
    }
}

template <typename... Ts>
INLINE void AiCoreLogF(LogContext* ctx, __gm__ const char* fmt, Ts... args)
{
    if (ctx && fmt) {
        (DispatchPrint(ctx, &fmt, args), ...);
        ctx->PrintRaw(ctx, fmt);
        if (ctx->PrintNewLine != nullptr) {
            ctx->PrintNewLine(ctx);
        }
    }
}

template <typename... Ts>
INLINE void AiCoreLogFWithLevel(LogContext* ctx, uint8_t level, __gm__ const char* fmt, Ts... args)
{
    if (ctx && fmt) {
        auto* logger = reinterpret_cast<AicoreLogger*>(ctx);
        logger->EncodeLogLevel(level);
        logger->EncodeTimestamp(get_sys_cnt());
        (DispatchPrint(ctx, &fmt, args), ...);
        ctx->PrintRaw(ctx, fmt);
        if (ctx->PrintNewLine != nullptr) {
            ctx->PrintNewLine(ctx);
        }
    }
}

#define AICORE_LOGD(ctx, fmt, ...)                                                                       \
    do {                                                                                                 \
        if ((ctx) && (ctx)->logLevel <= static_cast<uint8_t>(AicoreLogLevel::DEBUG)) {                   \
            AiCoreLogFWithLevel((ctx), static_cast<uint8_t>(AicoreLogLevel::DEBUG), fmt, ##__VA_ARGS__); \
        }                                                                                                \
    } while (false)

#define AICORE_LOGI(ctx, fmt, ...)                                                                      \
    do {                                                                                                \
        if ((ctx) && (ctx)->logLevel <= static_cast<uint8_t>(AicoreLogLevel::INFO)) {                   \
            AiCoreLogFWithLevel((ctx), static_cast<uint8_t>(AicoreLogLevel::INFO), fmt, ##__VA_ARGS__); \
        }                                                                                               \
    } while (false)

#define AICORE_LOGW(ctx, fmt, ...)                                                                      \
    do {                                                                                                \
        if ((ctx) && (ctx)->logLevel <= static_cast<uint8_t>(AicoreLogLevel::WARN)) {                   \
            AiCoreLogFWithLevel((ctx), static_cast<uint8_t>(AicoreLogLevel::WARN), fmt, ##__VA_ARGS__); \
        }                                                                                               \
    } while (false)

#define AICORE_LOGE(ctx, fmt, ...)                                                                       \
    do {                                                                                                 \
        if ((ctx) && (ctx)->logLevel <= static_cast<uint8_t>(AicoreLogLevel::ERROR)) {                   \
            AiCoreLogFWithLevel((ctx), static_cast<uint8_t>(AicoreLogLevel::ERROR), fmt, ##__VA_ARGS__); \
        }                                                                                                \
    } while (false)

#if defined(__TILE_FWK_AICORE__) && defined(TILEOP_UTILS_TUPLE_H)

template <size_t I, typename ShapeTuple>
INLINE void FillShapeDims(int64_t (&dims)[AicorePrintConst::MAX_SHAPE_DIMS], const ShapeTuple& shape)
{
    constexpr size_t n = Std::tuple_size<ShapeTuple>::value;
    constexpr size_t m = (n < AicorePrintConst::MAX_SHAPE_DIMS) ? n : AicorePrintConst::MAX_SHAPE_DIMS;

    if constexpr (I < m) {
        dims[I] = static_cast<int64_t>(Std::get<I>(shape));
        FillShapeDims<I + 1>(dims, shape);
    }
}

template <size_t N>
INLINE void LogShapeDims(LogContext* ctx, const int64_t (&dims)[AicorePrintConst::MAX_SHAPE_DIMS],
                         __gm__ const char* name = nullptr)
{
    auto* logger = reinterpret_cast<AicoreLogger*>(ctx);
    logger->EncodeLogLevel(ctx->logLevel);
    logger->EncodeTimestamp(get_sys_cnt());
    if (name) {
        logger->PrintRaw(name);
    } else {
        logger->PrintRaw("tensor");
    }

    if constexpr (N == 1) {
        AiCoreLogF(ctx, " shape=[%ld]\n", dims[0]);
    } else if constexpr (N == 2) {
        AiCoreLogF(ctx, " shape=[%ld,%ld]\n", dims[0], dims[1]);
    } else if constexpr (N == 3) {
        AiCoreLogF(ctx, " shape=[%ld,%ld,%ld]\n", dims[0], dims[1], dims[2]);
    } else if constexpr (N == 4) {
        AiCoreLogF(ctx, " shape=[%ld,%ld,%ld,%ld]\n", dims[0], dims[1], dims[2], dims[3]);
    } else if constexpr (N == 5) {
        AiCoreLogF(ctx, " shape=[%ld,%ld,%ld,%ld,%ld]\n", dims[0], dims[1], dims[2], dims[3], dims[4]);
    } else if constexpr (N == 6) {
        AiCoreLogF(ctx, " shape=[%ld,%ld,%ld,%ld,%ld,%ld]\n", dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    }
}

#endif

template <typename T, typename PtrT>
INLINE void PrintTensorImpl(LogContext* ctx, PtrT data, int64_t end, int64_t begin, __gm__ const char* name)
{
    using ElemT = std::remove_cv_t<T>;
    auto* logger = reinterpret_cast<AicoreLogger*>(ctx);
    logger->EncodeLogLevel(ctx->logLevel);
    logger->EncodeTimestamp(get_sys_cnt());
    logger->EncodeTensorHeader(name, begin, end);
    logger->Sync();

    for (int64_t i = begin; i < end; ++i) {
        ElemT tmp = data[i];

        if constexpr (std::is_floating_point_v<ElemT>) {
            float v = static_cast<float>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<float>::Type, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (std::is_integral_v<ElemT>) {
            int64_t v = static_cast<int64_t>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<int64_t>::Type, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (sizeof(ElemT) == 2) {
            auto bits = SafeBitCast<uint16_t>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<ElemT>::Type, i, reinterpret_cast<uint8_t*>(&bits), sizeof(bits));
        } else if constexpr (sizeof(ElemT) == 1) {
            uint8_t bits = SafeBitCast<uint8_t>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<ElemT>::Type, i, reinterpret_cast<uint8_t*>(&bits), sizeof(bits));
        }

        logger->Sync();
    }
    logger->PrintNewLine();
}

// ============================================================================
// Public Interface Functions
// ============================================================================

#if defined(__TILE_FWK_AICORE__) && defined(TILEOP_UTILS_TUPLE_H)
template <typename... Dims>
INLINE void AiCorePrintShape(LogContext* ctx, const TileOp::Shape<Dims...>& shape, __gm__ const char* name = nullptr)
{
    constexpr size_t N = Std::tuple_size<TileOp::Shape<Dims...>>::value;

    if constexpr (N == 0 || N > AicorePrintConst::MAX_SHAPE_DIMS) {
        return;
    }

    int64_t dims[AicorePrintConst::MAX_SHAPE_DIMS]{};
    FillShapeDims<0>(dims, shape);
    LogShapeDims<N>(ctx, dims, name);
}
#endif

template <typename T>
INLINE void AiCorePrintGmTensor(LogContext* ctx, __gm__ const T* data, int64_t end, int64_t begin,
                                __gm__ const char* name)
{
    PrintTensorImpl<T>(ctx, data, end, begin, name);
}

#if IS_AICORE
template <typename T>
INLINE void AiCorePrintUbTensor(LogContext* ctx, __ubuf__ const T* data, int64_t end, int64_t begin,
                                __gm__ const char* name)
{
#ifdef __AIC__
    static_assert(!std::is_same_v<T, T>,
                  "[AIC UB Print Error] AiCorePrintUbTensor is not supported on AIC (Cube) kernel. "
                  "AIC Scalar Processor cannot scalar-load from UB address space. "
                  "Please use AiCorePrintUbTensor in AIV (Vector) kernel instead. ");
#else
    PrintTensorImpl<T>(ctx, data, end, begin, name);
#endif
}

// L0C (accumulator) → GM copy helper using NZ2ND conversion.
// Uses copy_matrix_cc_to_gm packed registers with nz2nd bit set.
// Matches TStoreAccNz2nd (A2/A3) / TStoreAccND (A5) in pto-isa.
template <typename T>
__aicore__ void L0CRawCopyToGM(__gm__ T* dst, __cc__ const T* src, int64_t l0cShape0, int64_t l0cShape1)
{
    if (l0cShape0 <= 0 || l0cShape1 <= 0) {
        return;
    }

    constexpr uint16_t kFractal = 16;
    uint16_t mSize = static_cast<uint16_t>(l0cShape0);
    uint16_t nSize = static_cast<uint16_t>(l0cShape1);
    uint16_t srcStride = (mSize + kFractal - 1) / kFractal * kFractal;
    uint32_t dstStride = nSize;

    uint64_t xmReg = ((uint64_t)(nSize & 0xfff) << 4) | ((uint64_t)(mSize & 0xffff) << 16) |
                     ((uint64_t)(dstStride) << 32);

#if SUPPORT_L1_COPY
    // A2/A3: matches TStoreAccNz2nd in pto-isa.
    constexpr uint16_t ndNum = 1;
    constexpr uint32_t ndPara = ndNum;
    set_nd_para(ndPara);

    constexpr uint8_t nz2ndEn = 1;
    uint64_t xtReg = srcStride | ((uint64_t)(nz2ndEn & 0x1) << 43);
#else
    // A5: matches TStoreAccND in pto-isa.
    constexpr uint8_t unitFlagCtrl = 0;
    constexpr uint64_t quantPre = 0;
    constexpr uint8_t reluPreMode = 0;
    constexpr uint64_t nz2ndEn = 1;

    uint64_t xtReg = srcStride | (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) | (((quantPre >> 5) & 0x1) << 29) |
                     (static_cast<uint64_t>(quantPre & 0x1f) << 34) |
                     ((static_cast<uint64_t>(reluPreMode) & 0x7) << 39) | ((uint64_t)(nz2ndEn & 0x1) << 43);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t srcNdStride = 0;
    constexpr uint32_t dstNdStride = 0;
    uint64_t loop3Config = ndNum | (static_cast<uint64_t>(srcNdStride & 0xffff) << 16) |
                           (static_cast<uint64_t>(dstNdStride & 0xffff) << 32);
    set_loop3_para(loop3Config);
#endif

    copy_matrix_cc_to_gm(dst, const_cast<__cc__ T*>(src), xmReg, xtReg);
}

template <typename T>
INLINE void AiCorePrintL0CTensor(LogContext* ctx, __cc__ const T* data, int64_t end, int64_t begin, int64_t l0cShape0,
                                 int64_t l0cShape1, __gm__ T* staging, __gm__ const char* name)
{
    int64_t count = end - begin;

    if (count <= 0 || l0cShape0 <= 0 || l0cShape1 <= 0) {
        return;
    }

    if (staging == nullptr) {
        AiCoreLogF(ctx, "[WARNING] AiCorePrintL0CTensor: Parameter 7 (L0C staging address) is nullptr. "
                        "Unable to print L0C data. Please provide a valid staging buffer.");
        return;
    }

    uint64_t stagingAddr = reinterpret_cast<uint64_t>(staging);

    if ((stagingAddr & 0x1F) != 0) {
        AiCoreLogF(ctx,
                   "[WARNING] AiCorePrintL0CTensor: Parameter 7 (L0C staging address) is not aligned to 32 bytes. "
                   "Unable to print L0C data. The address must be 32-byte aligned. Please adjust and retry. "
                   "Current L0C staging address: 0x%lx",
                   stagingAddr);
        return;
    }

    pipe_barrier(PIPE_ALL);
    L0CRawCopyToGM<T>(staging, data, l0cShape0, l0cShape1);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    pipe_barrier(PIPE_ALL);

    int64_t totalBytes = l0cShape0 * l0cShape1 * static_cast<int64_t>(sizeof(T));
    int64_t offset = 0;

    while (offset < totalBytes) {
        dcci((__gm__ uint8_t*)staging + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
        offset += CACHE_LINE_SIZE;
    }

    AiCorePrintGmTensor<T>(ctx, staging, end, begin, name);
}
#endif // IS_AICORE

#if IS_AICORE && SUPPORT_L1_COPY
template <typename T>
__aicore__ void L1RawCopyToGM(__gm__ T* dst, __cbuf__ const T* src, int64_t count)
{
    int64_t totalBytes = count * sizeof(T);

    if (totalBytes == 0) {
        return;
    }

    uint16_t nBurst = 1;
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    uint16_t lenBurst = 0;

    if (totalBytes >= 32) {
        lenBurst = static_cast<uint16_t>((totalBytes + 31) / 32);
    } else if (totalBytes > 0) {
        lenBurst = static_cast<uint16_t>(totalBytes);
    } else {
        lenBurst = 1;
    }

    copy_cbuf_to_gm(dst, src, 0, nBurst, lenBurst, srcStride, dstStride);
}

template <typename T>
INLINE void AiCorePrintL1Tensor(LogContext* ctx, __cbuf__ const T* data, int64_t end, int64_t begin, __gm__ T* staging,
                                __gm__ const char* name)
{
    int64_t count = end - begin;

    if (count <= 0) {
        return;
    }

    if (staging == nullptr) {
        AiCoreLogF(ctx, "[WARNING] AiCorePrintL1Tensor: Parameter 5 (L1 staging address) is nullptr. "
                        "Unable to print L1 data. Please provide a valid staging buffer.");
        return;
    }

    uint64_t stagingAddr = reinterpret_cast<uint64_t>(staging);

    if ((stagingAddr & 0x1F) != 0) {
        AiCoreLogF(ctx,
                   "[WARNING] AiCorePrintL1Tensor: Parameter 5 (L1 staging address) is not aligned to 32 bytes. "
                   "Unable to print L1 data. The address must be 32-byte aligned. Please adjust and retry. "
                   "Current L1 staging address: 0x%lx",
                   stagingAddr);
        return;
    }

    pipe_barrier(PIPE_ALL);
    L1RawCopyToGM(staging, data + begin, count);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    pipe_barrier(PIPE_ALL);

    int64_t totalBytes = count * sizeof(T);
    int64_t offset = 0;

    while (offset < totalBytes) {
        dcci((__gm__ uint8_t*)staging + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
        offset += CACHE_LINE_SIZE;
    }

    AiCorePrintGmTensor<T>(ctx, staging, count, 0, name);
}
#endif
