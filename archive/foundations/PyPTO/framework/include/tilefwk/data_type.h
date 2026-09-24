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
 * \file data_type.h
 * \brief
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <cstdint>

#ifndef UNUSED
#define UNUSED(n) (void)(n)
#endif

namespace npu::tile_fwk {
const std::unordered_map<size_t, size_t> BLOCK_PADDING_DIM = {{1, 32}, {2, 16}, {4, 8}, {8, 4}};

#define DATA_TYPE_ALL                                            \
    DTYPE_DESC(DT_INT4, 1, 4, false, int4, 29)                   \
    DTYPE_DESC(DT_INT8, 1, 8, false, int8_t, 2)                  \
    DTYPE_DESC(DT_INT16, 2, 16, false, int16_t, 6)               \
    DTYPE_DESC(DT_INT32, 4, 32, false, int32_t, 3)               \
    DTYPE_DESC(DT_INT64, 8, 64, false, int64_t, 9)               \
    DTYPE_DESC(DT_FP8, 1, 8, true, float8_t, 28)                 \
    DTYPE_DESC(DT_FP16, 2, 16, true, half, 1)                    \
    DTYPE_DESC(DT_FP32, 4, 32, true, float, 0)                   \
    DTYPE_DESC(DT_BF16, 2, 16, true, bfloat16_t, 27)             \
    DTYPE_DESC(DT_HF4, 1, 4, true, hfloat4, 28)                  \
    DTYPE_DESC(DT_HF8, 1, 8, true, hifloat8_t, 34)               \
    DTYPE_DESC(DT_UINT8, 1, 8, false, uint8_t, 4)                \
    DTYPE_DESC(DT_UINT16, 2, 16, false, uint16_t, 7)             \
    DTYPE_DESC(DT_UINT32, 4, 32, false, uint32_t, 8)             \
    DTYPE_DESC(DT_UINT64, 8, 64, false, uint64_t, 10)            \
    DTYPE_DESC(DT_BOOL, 1, 8, false, bool, 12)                   \
    DTYPE_DESC(DT_DOUBLE, 8, 64, true, double, 11)               \
    DTYPE_DESC(DT_FP8E4M3, 1, 8, true, float8_e4m3_t, 36)        \
    DTYPE_DESC(DT_FP8E5M2, 1, 8, true, float8_e5m2_t, 35)        \
    DTYPE_DESC(DT_FP8E8M0, 1, 8, true, float8_e8m0_t, 37)        \
    /* cann中没有E2M1X2l类型，使用E2M1的cann_type:40 */ \
    DTYPE_DESC(DT_FP4_E2M1X2, 1, 8, true, float4_e2m1x2_t, 40)   \
    DTYPE_DESC(DT_FP4_E1M2X2, 1, 8, true, float4_e1m2x2_t, 41)   \
    DTYPE_DESC(DT_FP4_E2M1, 1, 4, true, float4_e2m1x2_t, 40)     \
    DTYPE_DESC(DT_FP4_E1M2, 1, 4, true, float4_e1m2x2_t, 41)

enum DataType {
#define DTYPE_DESC(name, byte, bit, is_float, type, cann_type) name,
    DATA_TYPE_ALL
#undef DTYPE_DESC
        DT_BOTTOM
};

enum class NodeType {
    LOCAL = 0,
    INCAST = 1,
    OUTCAST = 2,
};

enum class TileOpFormat {
    TILEOP_ND = 0,
    TILEOP_NZ = 1,
    TILEOP_NC1HWC0 = 2,
    TILEOP_NDC1HWC0 = 3,
    TILEOP_FRACTAL_Z = 4,
    TILEOP_FRACTAL_Z_3D = 5,
    TILEOP_FORMAT_NUM [[maybe_unused]],
};

enum MemoryType {
    MEM_UB = 0,
    MEM_L1 = 1,
    MEM_L0A = 2,
    MEM_L0B = 3,
    MEM_L0C = 4,
    MEM_FIX = 5,
    MEM_FIX_QUANT_PRE = 6,
    MEM_FIX_RELU_PRE = 7,
    MEM_FIX_RELU_POST = 8,
    MEM_FIX_QUANT_POST = 9,
    MEM_FIX_ELT_ANTIQ = 10,
    MEM_FIX_MTE2_ANTIQ = 11,
    MEM_BT = 12,
    MEM_L2 = 13,
    MEM_L3 = 14,
    MEM_DEVICE_DDR = 15,
    MEM_HOST1 = 16,
    MEM_FAR1 = 17,
    MEM_FAR2 = 18,
    MEM_WORKSPACE = 19,
    MEM_VECTOR_REG = 20,
    MEM_L0AMX = 21,
    MEM_L0BMX = 22,
    MEM_UNKNOWN
};

enum CastMode {
    CAST_NONE = 0,
    CAST_RINT = 1,  // round to nearest, tie to even
    CAST_ROUND = 2, // round to nearest, tie away from zero
    CAST_FLOOR = 3, // round to minus infinity
    CAST_CEIL = 4,  // round to positive infinity
    CAST_TRUNC = 5, // round to zero
    CAST_ODD = 6,   // round to odd (Von Neumann rounding)
};

inline std::string NodeType2String(NodeType n)
{
    switch (n) {
        case NodeType::LOCAL:
            return "LOCAL";
        case NodeType::INCAST:
            return "INCAST";
        case NodeType::OUTCAST:
            return "OUTCAST";
        default:
            throw std::invalid_argument("Unknown NodeType");
    }
}

inline bool IsFloat(DataType t)
{
    switch (t) {
#define DTYPE_DESC(name, byte, bit, is_float, type, cann_type) \
    case name:                                                 \
        return is_float;
        DATA_TYPE_ALL
#undef DTYPE_DESC
        default:
            throw std::invalid_argument("Unknown DataType");
    }
    return false;
}

inline bool IsInteger(DataType t)
{
    if ((t == DT_INT4) || (t == DT_INT8) || (t == DT_INT16) || (t == DT_INT32) || (t == DT_INT64)) {
        return true;
    }
    if ((t == DT_UINT8) || (t == DT_UINT16) || (t == DT_UINT32) || (t == DT_UINT64)) {
        return true;
    }
    return t == DT_BOOL;
}

inline const char* DataType2String(DataType t, bool brief = false)
{
    const char* ret = nullptr;
    switch (t) {
#define DTYPE_DESC(name, byte, bit, is_float, type, cann_type) \
    case name:                                                 \
        ret = #name;                                           \
        break;
        DATA_TYPE_ALL
#undef DTYPE_DESC
        default:
            throw std::invalid_argument("Unknown DataType");
    }
    return brief ? (ret + 0x3) : ret; // brief skip "DT_"
}

// Returns the byte width used to represent one element. Do not use it to calculate packed tensor data size.
inline size_t BytesOf(DataType t)
{
    switch (t) {
#define DTYPE_DESC(name, byte, bit, is_float, type, cann_type) \
    case name:                                                 \
        return byte;
        DATA_TYPE_ALL
#undef DTYPE_DESC
        default:
            throw std::invalid_argument("Unknown DataType");
    }
}

inline int64_t BitsOf(DataType t)
{
    switch (t) {
#define DTYPE_DESC(name, byte, bit, is_float, type, cann_type) \
    case name:                                                 \
        return bit;
        DATA_TYPE_ALL
#undef DTYPE_DESC
        default:
            throw std::invalid_argument("Unknown DataType");
    }
}

// The caller must ensure that the total number of bits is byte aligned.
inline int64_t DataSizeOf(int64_t numel, DataType t) { return numel * BitsOf(t) / 0x8; }

inline const char* DataType2CCEStr(DataType t)
{
    switch (t) {
#define DTYPE_DESC(name, byte, bit, is_float, type, cann_type) \
    case name:                                                 \
        return #type;
        DATA_TYPE_ALL
#undef DTYPE_DESC
        default:
            throw std::invalid_argument("Unknown DataType");
    }
}

inline std::string DataType2VectorRegStr(DataType t)
{
    switch (t) {
        case DT_FP16:
            return "vector_f16";
        case DT_FP32:
            return "vector_f32";
        default:
            return "Unknown DataType";
    }
}

const std::unordered_map<std::string, DataType> STR_DATA_TYPE_MAP = {{"int4", DT_INT4},
                                                                     {"int8", DT_INT8},
                                                                     {"int16", DT_INT16},
                                                                     {"int32", DT_INT32},
                                                                     {"int", DT_INT32},
                                                                     {"int64", DT_INT64},
                                                                     {"fp8", DT_FP8},
                                                                     {"float8", DT_FP8},
                                                                     {"fp16", DT_FP16},
                                                                     {"float16", DT_FP16},
                                                                     {"float", DT_FP32},
                                                                     {"float32", DT_FP32},
                                                                     {"bf16", DT_BF16},
                                                                     {"bfloat16", DT_BF16},
                                                                     {"hf4", DT_HF4},
                                                                     {"hf8", DT_HF8},
                                                                     {"uint8", DT_UINT8},
                                                                     {"uint16", DT_UINT16},
                                                                     {"uint32", DT_UINT32},
                                                                     {"uint64", DT_UINT64},
                                                                     {"bool", DT_BOOL},
                                                                     {"double", DT_DOUBLE},
                                                                     {"fp8e4m3", DT_FP8E4M3},
                                                                     {"fp8e5m2", DT_FP8E5M2},
                                                                     {"fp8e8m0", DT_FP8E8M0},
                                                                     {"fp4_e2m1", DT_FP4_E2M1},
                                                                     {"fp4_e2m1x2", DT_FP4_E2M1X2},
                                                                     {"fp4_e1m2", DT_FP4_E1M2},
                                                                     {"fp4_e1m2x2", DT_FP4_E1M2X2}};

inline size_t DataType2CannType(DataType t)
{
    switch (t) {
#define DTYPE_DESC(name, byte, bit, is_float, type, cann_type) \
    case name:                                                 \
        return cann_type;
        DATA_TYPE_ALL
#undef DTYPE_DESC
        default:
            throw std::invalid_argument("Unknown DataType");
    }
}

inline std::string MemoryTypeToString(MemoryType mt)
{
    switch (mt) {
        case MEM_UB:
            return "MEM_UB";
        case MEM_L1:
            return "MEM_L1";
        case MEM_L0A:
            return "MEM_L0A";
        case MEM_L0B:
            return "MEM_L0B";
        case MEM_L0AMX:
            return "MEM_L0AMX";
        case MEM_L0BMX:
            return "MEM_L0BMX";
        case MEM_L0C:
            return "MEM_L0C";
        case MEM_L2:
            return "MEM_L2";
        case MEM_L3:
            return "MEM_L3";
        case MEM_DEVICE_DDR:
            return "MEM_DEVICE_DDR";
        case MEM_HOST1:
            return "MEM_HOST1";
        case MEM_FAR1:
            return "MEM_FAR1";
        case MEM_FAR2:
            return "MEM_FAR2";
        case MEM_BT:
            return "MEM_BT";
        case MEM_VECTOR_REG:
            return "MEM_VECTOR_REG";
        case MEM_FIX:
            return "MEM_FIX";
        case MEM_FIX_QUANT_PRE:
            return "MEM_FIX_QUANT_PRE";
        case MEM_FIX_RELU_PRE:
            return "MEM_FIX_RELU_PRE";
        case MEM_FIX_RELU_POST:
            return "MEM_FIX_RELU_POST";
        case MEM_FIX_QUANT_POST:
            return "MEM_FIX_QUANT_POST";
        case MEM_FIX_ELT_ANTIQ:
            return "MEM_FIX_ELT_ANTIQ";
        case MEM_FIX_MTE2_ANTIQ:
            return "MEM_FIX_MTE2_ANTIQ";
        default:
            return "MEM_UNKNOWN";
    }
#undef CASE
}

[[maybe_unused]] inline std::string BriefMemoryTypeToString(MemoryType mt)
{
    switch (mt) {
        case MEM_UB:
            return "UB";
        case MEM_L1:
            return "L1";
        case MEM_L0A:
            return "L0A";
        case MEM_L0B:
            return "L0B";
        case MEM_L0AMX:
            return "L0AMX";
        case MEM_L0BMX:
            return "L0BMX";
        case MEM_L0C:
            return "L0C";
        case MEM_L2:
            return "L2";
        case MEM_L3:
            return "L3";
        case MEM_DEVICE_DDR:
            return "DDR";
        case MEM_HOST1:
            return "HOST1";
        case MEM_FAR1:
            return "FAR1";
        case MEM_FAR2:
            return "FAR2";
        case MEM_BT:
            return "BT";
        default:
            return "MEM_UNKNOWN";
    }
}

[[maybe_unused]] inline std::string CastMode2String(CastMode mode)
{
    switch (mode) {
        case CastMode::CAST_NONE:
            return "CAST_NONE";
        case CastMode::CAST_RINT:
            return "CAST_RINT";
        case CastMode::CAST_ROUND:
            return "CAST_ROUND";
        case CastMode::CAST_FLOOR:
            return "CAST_FLOOR";
        case CastMode::CAST_CEIL:
            return "CAST_CEIL";
        case CastMode::CAST_TRUNC:
            return "CAST_TRUNC";
        case CastMode::CAST_ODD:
            return "CAST_ODD";
        default:
            throw std::invalid_argument("Unknown CastMode");
    }
}

enum class CachePolicy { PREFETCH, NONE_CACHEABLE, MAX_NUM };
} // namespace npu::tile_fwk

namespace std {
inline std::string to_string(npu::tile_fwk::TileOpFormat format)
{
    switch (format) {
        case npu::tile_fwk::TileOpFormat::TILEOP_ND:
            return "TILEOP_ND";
        case npu::tile_fwk::TileOpFormat::TILEOP_NZ:
            return "TILEOP_NZ";
        case npu::tile_fwk::TileOpFormat::TILEOP_NC1HWC0:
            return "TILEOP_NC1HWC0";
        case npu::tile_fwk::TileOpFormat::TILEOP_NDC1HWC0:
            return "TILEOP_NDC1HWC0";
        case npu::tile_fwk::TileOpFormat::TILEOP_FRACTAL_Z:
            return "TILEOP_FRACTAL_Z";
        case npu::tile_fwk::TileOpFormat::TILEOP_FRACTAL_Z_3D:
            return "TILEOP_FRACTAL_Z_3D";
        default:
            return "UNKNOWN";
    }
}
} // namespace std
