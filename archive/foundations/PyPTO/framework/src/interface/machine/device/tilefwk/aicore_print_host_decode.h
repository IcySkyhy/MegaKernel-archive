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
 * \file aicore_print_host_decode.h
 * \brief
 */

#pragma once

#include "aicore_print_base.h"

#ifdef __TILE_FWK_HOST__
#include <inttypes.h>
#include <string>
#include <sstream>
#include <securec.h>

struct DecodeState {
    int64_t tail_;
    int64_t head_;
    int64_t size_;
    __gm__ uint8_t* data_;
    std::string lastTensorName_;
};

inline uint8_t ReadDecodeByte(DecodeState& state, int64_t off) { return state.data_[off % state.size_]; }

template <typename T>
inline T ReadDecodeValue(DecodeState& state, int64_t off)
{
    T val{};
    auto* bytes = reinterpret_cast<uint8_t*>(&val);

    for (size_t i = 0; i < sizeof(T); i++) {
        bytes[i] = ReadDecodeByte(state, off + i);
    }

    return val;
}

inline std::string ReadDecodeString(DecodeState& state, int64_t off)
{
    std::string result;
    result.reserve(64);

    while (off < state.head_) {
        char c = ReadDecodeValue<char>(state, off++);

        if (c == '\0') {
            break;
        }

        result.push_back(c);
    }

    return result;
}

inline int DecodeTensorHeader(DecodeState& state, char* buf, size_t maxSize)
{
    short nameLen = ReadDecodeValue<short>(state, state.tail_);
    state.tail_ += AicorePrintConst::NAMELEN_FIELD_SIZE;

    std::string name = ReadDecodeString(state, state.tail_);
    state.tail_ += nameLen;

    int64_t begin = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;

    int64_t end = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;

    state.lastTensorName_ = name;

    return snprintf_s(buf, maxSize, maxSize - 1, "tensor '%s', range=[%" PRId64 ", %" PRId64 ")\n", name.c_str(), begin,
                      end);
}

template <typename BitsT, typename DecodeFunc>
inline int DecodeIndexedFloat(DecodeState& state, char* buf, size_t maxSize, DecodeFunc decodeFunc)
{
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;

    BitsT bits = ReadDecodeValue<BitsT>(state, state.tail_);
    state.tail_ += sizeof(BitsT);

    float value = decodeFunc(bits);
    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%" PRId64 "] %f\n", state.lastTensorName_.c_str(), index, value);
}

inline int DecodeIndexedFp32(DecodeState& state, char* buf, size_t maxSize)
{
    return DecodeIndexedFloat<float>(state, buf, maxSize, [](float v) { return v; });
}

inline int DecodeIndexedInt64(DecodeState& state, char* buf, size_t maxSize)
{
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;

    int64_t value = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;

    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %" PRId64 "\n", state.lastTensorName_.c_str(), index, value);
}

inline int DecodeIndexedBf16(DecodeState& state, char* buf, size_t maxSize)
{
    return DecodeIndexedFloat<uint16_t>(state, buf, maxSize, DecodeBf16);
}

inline int DecodeIndexedFp16(DecodeState& state, char* buf, size_t maxSize)
{
    return DecodeIndexedFloat<uint16_t>(state, buf, maxSize, DecodeF16);
}

inline int DecodeIndexedFp8E4M3(DecodeState& state, char* buf, size_t maxSize)
{
    return DecodeIndexedFloat<uint8_t>(state, buf, maxSize, DecodeFp8E4M3);
}

inline int DecodeIndexedFp8E5M2(DecodeState& state, char* buf, size_t maxSize)
{
    return DecodeIndexedFloat<uint8_t>(state, buf, maxSize, DecodeFp8E5M2);
}

inline int DecodeIndexedFp8E8M0(DecodeState& state, char* buf, size_t maxSize)
{
    return DecodeIndexedFloat<uint8_t>(state, buf, maxSize, DecodeFp8E8M0);
}

inline int DecodeIndexedHf8(DecodeState& state, char* buf, size_t maxSize)
{
    return DecodeIndexedFloat<uint8_t>(state, buf, maxSize, DecodeHf8);
}

inline int DecodeOverflowWarning(DecodeState& state, char* buf, size_t maxSize)
{
    int64_t bufferSize = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += sizeof(int64_t);

    constexpr int64_t remoteHeaderSize = static_cast<int64_t>(AicorePrintConst::REMOTE_HEADER_SIZE);
    int64_t fullBufferSize = bufferSize + remoteHeaderSize;
    int64_t recommendedSize = fullBufferSize * 2;

    return snprintf_s(buf, maxSize, maxSize - 1,
                      "[WARNING] The PRINT_BUFFER_SIZE (ring buffer) is full! "
                      "Current buffer: %ld bytes (%ld KB). "
                      "Recommend: set PRINT_BUFFER_SIZE >= %ld (%ld KB, double current size) "
                      "in framework/src/interface/machine/device/tilefwk/aicpu_common.h, "
                      "then rebuild and reinstall.\n",
                      fullBufferSize, fullBufferSize / 1024, recommendedSize, recommendedSize / 1024);
}

inline int DecodeLegacyRecord(DecodeState& state, AicorePrint::DataType type, char* buf, size_t maxSize)
{
    auto valOff = state.tail_ + AicorePrintConst::NAMELEN_FIELD_SIZE;
    state.tail_ += ReadDecodeValue<short>(state, state.tail_) + AicorePrintConst::NAMELEN_FIELD_SIZE;

    auto fmtOff = state.tail_ + AicorePrintConst::NAMELEN_FIELD_SIZE;
    std::string fmt = ReadDecodeString(state, fmtOff);
    state.tail_ += ReadDecodeValue<short>(state, state.tail_) + AicorePrintConst::NAMELEN_FIELD_SIZE;

    auto formatValue = [&](auto v) { return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), v); };
    auto formatString = [&](const std::string& s) {
        return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), s.c_str());
    };

    switch (type) {
        case AicorePrint::DataType::Normal:
            return formatValue(0);

        case AicorePrint::DataType::Fp32:
            return formatValue(ReadDecodeValue<float>(state, valOff));

        case AicorePrint::DataType::Int64:
            return formatValue(ReadDecodeValue<int64_t>(state, valOff));

        case AicorePrint::DataType::Char:
            return formatValue(ReadDecodeValue<char>(state, valOff));

        case AicorePrint::DataType::String:
            return formatString(ReadDecodeString(state, valOff));

        case AicorePrint::DataType::Pointer:
            return formatValue(ReadDecodeValue<int64_t>(state, valOff));

        case AicorePrint::DataType::Bf16:
            return formatValue(DecodeBf16(ReadDecodeValue<uint16_t>(state, valOff)));

        case AicorePrint::DataType::Fp16:
            return formatValue(DecodeF16(ReadDecodeValue<uint16_t>(state, valOff)));

        case AicorePrint::DataType::Fp8E4M3:
            return formatValue(DecodeFp8E4M3(ReadDecodeValue<uint8_t>(state, valOff)));

        case AicorePrint::DataType::Fp8E5M2:
            return formatValue(DecodeFp8E5M2(ReadDecodeValue<uint8_t>(state, valOff)));

        case AicorePrint::DataType::Fp8E8M0:
            return formatValue(DecodeFp8E8M0(ReadDecodeValue<uint8_t>(state, valOff)));

        case AicorePrint::DataType::Hf8:
            return formatValue(DecodeHf8(ReadDecodeValue<uint8_t>(state, valOff)));

        default:
            buf[0] = '?';
            return 1;
    }
}

inline int DecodeLogLevelMarker(DecodeState& state, char* buf, size_t maxSize)
{
    uint8_t level = ReadDecodeValue<uint8_t>(state, state.tail_);
    state.tail_ += sizeof(uint8_t);

    const char* tag = "";
    switch (static_cast<AicoreLogLevel>(level)) {
        case AicoreLogLevel::DEBUG:
            tag = "[DEBUG] ";
            break;
        case AicoreLogLevel::INFO:
            tag = "[INFO] ";
            break;
        case AicoreLogLevel::WARN:
            tag = "[WARN] ";
            break;
        case AicoreLogLevel::ERROR:
            tag = "[ERROR] ";
            break;
        default:
            return 0;
    }
    return snprintf_s(buf, maxSize, maxSize - 1, "%s", tag);
}

inline int DecodeRecordImpl(DecodeState& state, AicorePrint::DataType type, char* buf, size_t maxSize)
{
    switch (type) {
        case AicorePrint::DataType::LogLevelMarker:
            return DecodeLogLevelMarker(state, buf, maxSize);

        case AicorePrint::DataType::TensorHeader:
            return DecodeTensorHeader(state, buf, maxSize);

        case AicorePrint::DataType::IndexedFp32:
            return DecodeIndexedFp32(state, buf, maxSize);

        case AicorePrint::DataType::IndexedInt64:
            return DecodeIndexedInt64(state, buf, maxSize);

        case AicorePrint::DataType::IndexedBf16:
            return DecodeIndexedBf16(state, buf, maxSize);

        case AicorePrint::DataType::IndexedFp16:
            return DecodeIndexedFp16(state, buf, maxSize);

        case AicorePrint::DataType::IndexedFp8E4M3:
            return DecodeIndexedFp8E4M3(state, buf, maxSize);

        case AicorePrint::DataType::IndexedFp8E5M2:
            return DecodeIndexedFp8E5M2(state, buf, maxSize);

        case AicorePrint::DataType::IndexedFp8E8M0:
            return DecodeIndexedFp8E8M0(state, buf, maxSize);

        case AicorePrint::DataType::IndexedHf8:
            return DecodeIndexedHf8(state, buf, maxSize);

        case AicorePrint::DataType::OverflowWarning:
            return DecodeOverflowWarning(state, buf, maxSize);

        default:
            return DecodeLegacyRecord(state, type, buf, maxSize);
    }
}

#endif
