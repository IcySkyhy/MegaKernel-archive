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
 * \file aicore_print_logger.h
 * \brief
 */

#pragma once

#include "aicore_print_host_decode.h"

// ============================================================================
// AicoreLogger Class
// ============================================================================

class AicoreLogger {
public:
    struct RemoteHeader {
        int64_t head_;
        int64_t tail_;
    };

#define AICORE_STATIC_PRINT_WRAPPER(Name, Func, T)                               \
    static __aicore__ void Name(LogContext* ctx, __gm__ const char** fmt, T val) \
    {                                                                            \
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);                       \
        if (self) {                                                              \
            self->Func(fmt, val);                                                \
        }                                                                        \
    }

    AICORE_STATIC_PRINT_WRAPPER(StaticPrintInt64, PrintInt64, int64_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp32, PrintFp32, float)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintBf16, PrintBf16, uint16_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp16, PrintFp16, uint16_t)

    static __aicore__ void StaticPrintRaw(LogContext* ctx, __gm__ const char* fmt)
    {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintRaw(fmt);
        }
    }

    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp8E4M3, PrintFp8E4M3, uint8_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp8E5M2, PrintFp8E5M2, uint8_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp8E8M0, PrintFp8E8M0, uint8_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintHf8, PrintHf8, uint8_t)

#undef AICORE_STATIC_PRINT_WRAPPER

    static __aicore__ void StaticPrintNewLine(LogContext* ctx)
    {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self != nullptr) {
            self->PrintNewLine();
        }
    }

    __aicore__ void Init(__gm__ uint8_t* buf, size_t n, uint8_t level = static_cast<uint8_t>(AicoreLogLevel::NONE))
    {
        if (n < AicorePrintConst::MIN_BUFFER_TOTAL_SIZE) {
            overflowed_ = true;
            size_ = 0;
            remote_ = nullptr;
            data_ = nullptr;
            head_ = 0;
            tail_ = 0;
            ctx_.logLevel = static_cast<uint8_t>(AicoreLogLevel::NONE);
            return;
        }

        remote_ = reinterpret_cast<volatile __gm__ RemoteHeader*>(buf);
        remote_->head_ = 0;
        remote_->tail_ = 0;
        head_ = 0;
        tail_ = 0;
        size_ = n - sizeof(RemoteHeader);
        data_ = buf + sizeof(RemoteHeader);
        overflowed_ = false;

        ctx_.PrintInt64 = StaticPrintInt64;
        ctx_.PrintFp32 = StaticPrintFp32;
        ctx_.PrintBf16 = StaticPrintBf16;
        ctx_.PrintFp16 = StaticPrintFp16;
        ctx_.PrintRaw = StaticPrintRaw;
        ctx_.PrintFp8E4M3 = StaticPrintFp8E4M3;
        ctx_.PrintFp8E5M2 = StaticPrintFp8E5M2;
        ctx_.PrintFp8E8M0 = StaticPrintFp8E8M0;
        ctx_.PrintHf8 = StaticPrintHf8;
        ctx_.PrintNewLine = StaticPrintNewLine;
        ctx_.logLevel = level;
    }

    __aicore__ __gm__ uint8_t* GetBuffer() const { return data_ - sizeof(RemoteHeader); }

    INLINE LogContext* Context() { return &ctx_; }

#ifdef __TILE_FWK_HOST__
    void BindHostBuffer(const uint8_t* hostBuf, size_t n)
    {
        if (n < AicorePrintConst::MIN_BUFFER_TOTAL_SIZE) {
            remote_ = nullptr;
            data_ = nullptr;
            size_ = 0;
            return;
        }
        remote_ = reinterpret_cast<volatile RemoteHeader*>(const_cast<uint8_t*>(hostBuf));
        data_ = const_cast<uint8_t*>(hostBuf) + sizeof(RemoteHeader);
        size_ = static_cast<int64_t>(n - sizeof(RemoteHeader));
        head_ = 0;
        tail_ = 0;
        overflowed_ = false;
        lastTensorName_.clear();
    }
#endif

    __aicore__ void PrintInt64(__gm__ const char** fmt, int64_t val)
    {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);

        if (idx == -1) {
            return;
        }

        switch (curFmt[idx++]) {
            case 's': {
                auto tmp = reinterpret_cast<__gm__ const char*>(val);
                if (tmp == nullptr) {
                    tmp = "<null>";
                }
                EncodeTyped(AicorePrint::DataType::String, reinterpret_cast<__gm__ const uint8_t*>(tmp),
                            StringLength(tmp), *fmt, idx);
                break;
            }
            case 'd':
            case 'i':
            case 'x':
            case 'X':
            case 'o':
            case 'u':
                EncodeTyped(AicorePrint::DataType::Int64, reinterpret_cast<uint8_t*>(&val), sizeof(val), *fmt, idx);
                break;

            case 'p':
                EncodeTyped(AicorePrint::DataType::Pointer, reinterpret_cast<uint8_t*>(&val), sizeof(val), *fmt, idx);
                break;

            case 'c': {
                char c = static_cast<char>(val);
                EncodeTyped(AicorePrint::DataType::Char, reinterpret_cast<uint8_t*>(&c), 1, *fmt, idx);
                break;
            }

            default:
                EncodeTyped(AicorePrint::DataType::Normal, static_cast<uint8_t*>(nullptr), 0, *fmt, idx);
                break;
        }

        *fmt = *fmt + idx;
    }

    __aicore__ void PrintFp32(__gm__ const char** fmt, float val)
    {
        EncodeFloatType(fmt, AicorePrint::DataType::Fp32, reinterpret_cast<uint8_t*>(&val), sizeof(val));
    }

    __aicore__ void PrintBf16(__gm__ const char** fmt, uint16_t rawBits)
    {
        EncodeFloatType(fmt, AicorePrint::DataType::Bf16, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp16(__gm__ const char** fmt, uint16_t rawBits)
    {
        EncodeFloatType(fmt, AicorePrint::DataType::Fp16, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

#define AICORE_ENCODE_FP8_FUNC(Name, DataTypeVal)                                                 \
    __aicore__ void Name(__gm__ const char** fmt, uint8_t rawBits)                                \
    {                                                                                             \
        EncodeFloatType(fmt, DataTypeVal, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits)); \
    }

    AICORE_ENCODE_FP8_FUNC(PrintFp8E4M3, AicorePrint::DataType::Fp8E4M3)
    AICORE_ENCODE_FP8_FUNC(PrintFp8E5M2, AicorePrint::DataType::Fp8E5M2)
    AICORE_ENCODE_FP8_FUNC(PrintFp8E8M0, AicorePrint::DataType::Fp8E8M0)
    AICORE_ENCODE_FP8_FUNC(PrintHf8, AicorePrint::DataType::Hf8)

#undef AICORE_ENCODE_FP8_FUNC

    __aicore__ void PrintRaw(__gm__ const char* str)
    {
        auto n = StringLength(str);

        if (n) {
            EncodeTyped(AicorePrint::DataType::Normal, reinterpret_cast<const __gm__ uint8_t*>(str), n, str, n);
        }

        Sync();
    }

    __aicore__ void PrintNewLine()
    {
        if (!CheckSpaceForRecord(2)) {
            return;
        }
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::Newline));
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
        Sync();
    }

    __aicore__ void EncodeLogLevel(uint8_t level)
    {
        if (!CheckSpaceForRecord(2)) {
            return;
        }
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::LogLevelMarker));
        EncodeByte(level);
    }

    __aicore__ void EncodeTimestamp(uint64_t cycle)
    {
        constexpr int64_t recordSize = 1 + sizeof(uint64_t);
        if (!CheckSpaceForRecord(recordSize)) {
            return;
        }
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::TimestampMarker));
        EncodeValue<uint64_t>(cycle);
    }

    __aicore__ void EncodeTensorHeader(__gm__ const char* name, int64_t begin, int64_t end)
    {
        size_t nameLenRaw = StringLength(name) + 1;

        if (nameLenRaw > static_cast<size_t>(AicorePrintConst::SHORT_MAX_VALUE)) {
            return;
        }

        short nameLen = static_cast<short>(nameLenRaw);
        int64_t recordSize = 1 + sizeof(short) + nameLen + sizeof(int64_t) * 2 + 1;

        if (!CheckSpaceForRecord(recordSize)) {
            return;
        }

        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::TensorHeader));
        EncodeValue<short>(nameLen);

        for (short i = 0; i < nameLen; i++) {
            EncodeByte(static_cast<uint8_t>(name[i]));
        }

        EncodeValue<int64_t>(begin);
        EncodeValue<int64_t>(end);
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    __aicore__ void EncodeIndexed(AicorePrint::DataType ty, int64_t index, const uint8_t* val, short valLen)
    {
        int64_t recordSize = 1 + sizeof(int64_t) + valLen + 1;

        if (!CheckSpaceForRecord(recordSize)) {
            return;
        }

        EncodeByte(static_cast<uint8_t>(ty));
        EncodeValue<int64_t>(index);

        for (short i = 0; i < valLen; i++) {
            EncodeByte(val[i]);
        }

        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    __aicore__ void EncodeOverflowWarning(int64_t bufferSize)
    {
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::OverflowWarning));
        EncodeValue<int64_t>(bufferSize);
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    template <typename PtrT>
    __aicore__ void EncodeTyped(AicorePrint::DataType ty, [[maybe_unused]] PtrT val, short valLen,
                                __gm__ const char* fmt, int fmtLen)
    {
        short paddedFmtLen = fmtLen + 1;
        int64_t recordSize = 1 + sizeof(short) + valLen + sizeof(short) + paddedFmtLen + 1;

        if (!CheckSpaceForRecord(recordSize)) {
            return;
        }

        EncodeByte(static_cast<uint8_t>(ty));
        EncodeValue<short>(valLen);

        if constexpr (!std::is_same_v<std::remove_cv_t<PtrT>, std::nullptr_t>) {
            if (val) {
                for (short i = 0; i < valLen; i++) {
                    EncodeByte(val[i]);
                }
            }
        }

        EncodeValue<short>(paddedFmtLen);

        for (int i = 0; i < fmtLen; i++) {
            EncodeByte(static_cast<uint8_t>(fmt[i]));
        }

        EncodeByte('\0');
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    template <typename PtrT>
    __aicore__ void EncodeFloatType(__gm__ const char** fmt, AicorePrint::DataType ty, PtrT val, short valLen)
    {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);

        if (idx == -1) {
            return;
        }

        if (curFmt[idx] == 'f') {
            EncodeTyped(ty, val, valLen, *fmt, idx + 1);
        } else {
            EncodeTyped(AicorePrint::DataType::Normal, nullptr, 0, *fmt, idx + 1);
        }

        *fmt = *fmt + idx + 1;
    }

    __aicore__ void Sync()
    {
        if (remote_ == nullptr || data_ == nullptr) {
            return;
        }

#ifndef __TILE_FWK_HOST__
        int64_t delta = (int64_t)(&data_[remote_->head_ % size_]) & (CACHE_LINE_SIZE - 1);
        int64_t off = remote_->head_ - delta;

        while (off < head_) {
            dcci(&data_[off % size_], SINGLE_CACHE_LINE, CACHELINE_OUT);
            off += CACHE_LINE_SIZE;
        }

        remote_->head_ = head_;
        remote_->tail_ = tail_;
        dcci(remote_, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
        remote_->head_ = head_;
        remote_->tail_ = tail_;
#endif
    }

#ifdef __TILE_FWK_HOST__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    int Read(char* buf, size_t maxSize, uint8_t* outLevel = nullptr, uint64_t* outTimestamp = nullptr,
             uint32_t maxIterations = 1000)
    {
        if (remote_ == nullptr || data_ == nullptr || size_ == 0) {
            return 0;
        }

        size_t totalWritten = 0;
        head_ = remote_->head_;
        uint8_t lineLevel = static_cast<uint8_t>(AicoreLogLevel::NONE);
        uint64_t lineTimestamp = 0;

        if (tail_ < remote_->tail_) {
            tail_ = remote_->tail_;
        }

        DecodeState state{tail_, head_, size_, data_, lastTensorName_};
        uint32_t iterationCount = 0;

        while (state.tail_ != state.head_ && iterationCount < maxIterations) {
            iterationCount++;
            AicorePrint::DataType type = static_cast<AicorePrint::DataType>(ReadDecodeByte(state, state.tail_++));

            if (type == AicorePrint::DataType::Newline) {
                while (state.tail_ != state.head_ && static_cast<AicorePrint::DataType>(ReadDecodeByte(
                                                         state, state.tail_)) == AicorePrint::DataType::End) {
                    state.tail_++;
                }
                if (totalWritten > 0) {
                    if (outLevel != nullptr) {
                        *outLevel = lineLevel;
                    }
                    if (outTimestamp != nullptr) {
                        *outTimestamp = lineTimestamp;
                    }
                    tail_ = state.tail_;
                    lastTensorName_ = state.lastTensorName_;
                    return static_cast<int>(totalWritten);
                }
                lineLevel = static_cast<uint8_t>(AicoreLogLevel::NONE);
                lineTimestamp = 0;
                continue;
            }

            if (type == AicorePrint::DataType::LogLevelMarker) {
                lineLevel = ReadDecodeValue<uint8_t>(state, state.tail_);
                state.tail_ += sizeof(uint8_t);
                continue;
            }

            if (type == AicorePrint::DataType::TimestampMarker) {
                lineTimestamp = ReadDecodeValue<uint64_t>(state, state.tail_);
                state.tail_ += sizeof(uint64_t);
                continue;
            }

            if (type == AicorePrint::DataType::End) {
                continue;
            }

            if (maxSize == 0) {
                continue;
            }

            int written = DecodeRecordImpl(state, type, buf, maxSize);

            if (written > 0) {
                buf += written;
                totalWritten += written;
                maxSize -= written;
            }
        }

        tail_ = state.tail_;
        lastTensorName_ = state.lastTensorName_;
        return static_cast<int>(totalWritten);
    }
#pragma GCC diagnostic pop
#endif

private:
    __aicore__ bool CheckSpaceForRecord(int64_t recordSize)
    {
        if (overflowed_ || size_ == 0) {
            return false;
        }

        int64_t freeSpace = size_ - (head_ - tail_);
        int64_t requiredSpace = recordSize + AicorePrintConst::WARNING_RESERVE_SPACE;

        if (freeSpace < requiredSpace) {
            EncodeOverflowWarning(size_);
            Sync();
            overflowed_ = true;
            return false;
        }

        return true;
    }

    __aicore__ void EncodeByte(uint8_t val)
    {
        volatile __gm__ uint8_t* p = &data_[head_++ % size_];
        *p = val;
    }

    template <typename T>
    __aicore__ void EncodeValue(T value)
    {
        auto* bytes = reinterpret_cast<uint8_t*>(&value);

        for (size_t i = 0; i < sizeof(T); i++) {
            EncodeByte(bytes[i]);
        }
    }

    INLINE bool IsFormatFlagChar(char c) { return c == '0' || c == '+' || c == '-' || c == ' ' || c == '#'; }

    INLINE int64_t SkipFormatFlags(__gm__ const char* fmt, int64_t idx)
    {
        while (fmt[idx] && IsFormatFlagChar(fmt[idx])) {
            idx++;
        }
        return idx;
    }

    INLINE int64_t SkipLengthModifier(__gm__ const char* fmt, int64_t idx)
    {
        if (fmt[idx] == 'l' || fmt[idx] == 'z' || fmt[idx] == 'h') {
            idx++;
            if (fmt[idx] == 'l') {
                idx++;
            }
        }
        return idx;
    }

    __aicore__ int64_t ParseNextFormat(__gm__ const char* fmt)
    {
        int64_t idx = 0;

        while (fmt[idx]) {
            if (fmt[idx] == '%') {
                if (fmt[idx + 1] == '%') {
                    idx += 2;
                } else {
                    break;
                }
            } else {
                idx++;
            }
        }

        if (!fmt[idx]) {
            return -1;
        }

        idx++;
        idx = SkipFormatFlags(fmt, idx);

        while (IsDigit(fmt[idx])) {
            idx++;
        }

        if (fmt[idx] == '.') {
            idx++;
            while (IsDigit(fmt[idx])) {
                idx++;
            }
        }

        idx = SkipLengthModifier(fmt, idx);

        if (fmt[idx]) {
            return idx;
        } else {
            return -1;
        }
    }

    INLINE size_t StringLength(__gm__ const char* str)
    {
        size_t n = 0;

        while (*str++) {
            n++;
        }

        return n;
    }

    INLINE bool IsDigit(char c) { return c >= '0' && c <= '9'; }

    LogContext ctx_{};
    int64_t head_ = 0;
    int64_t tail_ = 0;
    int64_t size_ = 0;
    volatile __gm__ RemoteHeader* remote_ = nullptr;
    __gm__ uint8_t* data_ = nullptr;
    bool overflowed_ = false;

#ifdef __TILE_FWK_HOST__
    std::string lastTensorName_{};
#endif
};
