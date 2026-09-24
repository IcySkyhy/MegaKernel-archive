/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dtype.h
 * \brief Data type definitions for PyPTO tensors and operations
 *
 * This file defines the DataType class which represents all supported numeric types
 * in the PyPTO framework, including integers, unsigned integers, floating point,
 * bfloat16, and Hisilicon float formats.
 */

#pragma once
#include <cstdint>
#include <string>

#include "core/bimap.h"

// C-style type strings returned by DataType::ToCTypeString()
#define IR_KW_TYPE_BOOL "bool"
#define IR_KW_TYPE_INT8 "int8_t"
#define IR_KW_TYPE_INT16 "int16_t"
#define IR_KW_TYPE_INT32 "int32_t"
#define IR_KW_TYPE_INT64 "int64_t"
#define IR_KW_TYPE_UINT8 "uint8_t"
#define IR_KW_TYPE_UINT16 "uint16_t"
#define IR_KW_TYPE_UINT32 "uint32_t"
#define IR_KW_TYPE_UINT64 "uint64_t"
#define IR_KW_TYPE_FP16 "half"
#define IR_KW_TYPE_FP32 "float"
#define IR_KW_TYPE_FP64 "double"
#define IR_KW_TYPE_BF16 "bfloat16_t"
#define IR_KW_TYPE_FP8E4M3FN "float8_e4m3_t"
#define IR_KW_TYPE_FP8E5M2 "float8_e5m2_t"
#define IR_KW_TYPE_FP8E8M0 "float8_e8m0_t"
#define IR_KW_TYPE_FP4E2M1 "float4_e2m1x2_t"
#define IR_KW_TYPE_FP4E1M2 "float4_e1m2x2_t"
#define IR_KW_TYPE_HF4 "hifloat4_t"
#define IR_KW_TYPE_HF8 "hifloat8_t"
#define IR_KW_TYPE_UNKNOWN "unknown"

namespace pypto {
namespace ir {

/**
 * \brief Data type representation for PyPTO
 *
 * This class encapsulates all numeric data types supported by PyPTO tensors and operations.
 * It includes:
 * - Signed integers: INT4, INT8, INT16, INT32, INT64
 * - Unsigned integers: UINT4, UINT8, UINT16, UINT32, UINT64
 * - Floating point: FP4, FP4E2M1, FP4E1M2, FP8, FP8E4M3FN, FP8E5M2, FP8E8M0, FP16, FP32
 * - Brain floating point: BF16
 * - Hisilicon float formats: HF4, HF8
 * - Boolean: BOOL
 */
class DataType {
public:
    // Type code constants
    // Organized by category with gaps for future extension

    // Boolean types: 0x00-0x0F
    static constexpr uint8_t kBoolCode = 0x00;

    // Signed integer types: 0x10-0x1F (16 slots reserved)
    static constexpr uint8_t kSignedIntRangeStart = 0x10;
    static constexpr uint8_t kInt4Code = 0x10;
    static constexpr uint8_t kInt8Code = 0x11;
    static constexpr uint8_t kInt16Code = 0x12;
    static constexpr uint8_t kInt32Code = 0x13;
    static constexpr uint8_t kInt64Code = 0x14;
    static constexpr uint8_t kIndexCode = 0x15;
    static constexpr uint8_t kSignedIntRangeEnd = 0x1F;
    // 0x15-0x1F reserved for future signed integer types

    // Unsigned integer types: 0x20-0x2F (16 slots reserved)
    static constexpr uint8_t kUnsignedIntRangeStart = 0x20;
    static constexpr uint8_t kUInt4Code = 0x20;
    static constexpr uint8_t kUInt8Code = 0x21;
    static constexpr uint8_t kUInt16Code = 0x22;
    static constexpr uint8_t kUInt32Code = 0x23;
    static constexpr uint8_t kUInt64Code = 0x24;
    static constexpr uint8_t kUnsignedIntRangeEnd = 0x2F;
    // 0x25-0x2F reserved for future unsigned integer types

    // IEEE floating point types: 0x30-0x3F (16 slots reserved)
    static constexpr uint8_t kIeeeFloatRangeStart = 0x30;
    static constexpr uint8_t kFp4Code = 0x30;
    static constexpr uint8_t kFp8e4m3fnCode = 0x31;
    static constexpr uint8_t kFp8e5m2Code = 0x32;
    static constexpr uint8_t kFp8Code = kFp8e4m3fnCode; // Backward compatibility alias
    static constexpr uint8_t kFp8e8m0Code = 0x36;
    static constexpr uint8_t kFp4e2m1Code = 0x37;
    static constexpr uint8_t kFp4e1m2Code = 0x38;
    static constexpr uint8_t kFp16Code = 0x33;
    static constexpr uint8_t kFp32Code = 0x34;
    static constexpr uint8_t kFp64Code = 0x35; // Reserved for future FP64 support
    static constexpr uint8_t kIeeeFloatRangeEnd = 0x3F;
    // 0x39-0x3F reserved for future IEEE float types

    // Brain/Hisilicon float types: 0x40-0x4F (16 slots reserved)
    static constexpr uint8_t kBrainFloatRangeStart = 0x40;
    static constexpr uint8_t kBf16Code = 0x40;
    static constexpr uint8_t kHf4Code = 0x41;
    static constexpr uint8_t kHf8Code = 0x42;
    static constexpr uint8_t kBrainFloatRangeEnd = 0x4F;
    // 0x43-0x4F reserved for future brain/Hisilicon float types

    // Static constants for all data types
    static const DataType BOOL;      // Boolean (true/false)
    static const DataType INT4;      // 4-bit signed integer
    static const DataType INT8;      // 8-bit signed integer
    static const DataType INT16;     // 16-bit signed integer
    static const DataType INT32;     // 32-bit signed integer
    static const DataType INT64;     // 64-bit signed integer
    static const DataType UINT4;     // 4-bit unsigned integer
    static const DataType UINT8;     // 8-bit unsigned integer
    static const DataType UINT16;    // 16-bit unsigned integer
    static const DataType UINT32;    // 32-bit unsigned integer
    static const DataType UINT64;    // 64-bit unsigned integer
    static const DataType FP4;       // 4-bit floating point
    static const DataType FP8E4M3FN; // 8-bit floating point (IEEE 754 e4m3fn format)
    static const DataType FP8E5M2;   // 8-bit floating point (IEEE 754 e5m2 format)
    static const DataType FP8;       // 8-bit floating point (backward compatibility alias)
    static const DataType FP8E8M0;   // 8-bit floating point (8-bit exponent, 0-bit mantissa)
    static const DataType FP4E2M1;   // 4-bit floating point (2-bit exponent, 1-bit mantissa)
    static const DataType FP4E1M2;   // 4-bit floating point (1-bit exponent, 2-bit mantissa)
    static const DataType FP16;      // 16-bit floating point (IEEE 754 half precision)
    static const DataType FP32;      // 32-bit floating point (IEEE 754 single precision)
    static const DataType FP64;      // 64-bit floating point (IEEE 754 double precision)
    static const DataType BF16;      // 16-bit brain floating point
    static const DataType HF4;       // 4-bit Hisilicon float
    static const DataType HF8;       // 8-bit Hisilicon float
    static const DataType INDEX;     // 32-bit index type

    /**
     * \brief Default constructor, initializes to BOOL type
     */
    constexpr DataType() : code_(kBoolCode) {}

    /**
     * \brief Construct from type code
     * \param code The type code
     */
    constexpr explicit DataType(uint8_t code) : code_(code) {}

    /**
     * \brief Get the size in bits of this data type
     *
     * Returns the storage size in bits for each data type. This accurately
     * represents sub-byte types like INT4, UINT4, FP4, and HF4.
     *
     * \return Size in bits
     */
    [[nodiscard]] size_t GetBit() const
    {
        switch (code_) {
            case kBoolCode:
                return 8; // These dtypes occupy 8 bits.
            case kHf4Code:
            case kFp4Code:
            case kFp4e2m1Code:
            case kFp4e1m2Code:
            case kUInt4Code:
            case kInt4Code:
                return 4; // These dtypes occupy 4 bits.
            case kHf8Code:
            case kFp8e4m3fnCode:
            case kFp8e5m2Code:
            case kFp8e8m0Code:
            case kUInt8Code:
            case kInt8Code:
                return 8; // These dtypes occupy 8 bits.
            case kBf16Code:
            case kFp16Code:
            case kUInt16Code:
            case kInt16Code:
                return 16; // These dtypes occupy 16 bits.
            case kFp32Code:
            case kUInt32Code:
            case kInt32Code:
                return 32; // These dtypes occupy 32 bits.
            case kUInt64Code:
            case kInt64Code:
            case kFp64Code:
            case kIndexCode:
                return 64; // These dtypes occupy 64 bits.
            default:
                return 0;
        }
    }

    /**
     * \brief Get a human-readable string name for this data type
     *
     * \return String representation of the data type
     */
    [[nodiscard]] std::string ToString() const
    {
        switch (code_) {
            case kInt4Code:
                return "int4";
            case kInt8Code:
                return "int8";
            case kInt16Code:
                return "int16";
            case kInt32Code:
                return "int32";
            case kInt64Code:
                return "int64";
            case kIndexCode:
                return "index";
            case kUInt4Code:
                return "uint4";
            case kUInt8Code:
                return "uint8";
            case kUInt16Code:
                return "uint16";
            case kUInt32Code:
                return "uint32";
            case kUInt64Code:
                return "uint64";
            case kFp4Code:
                return "fp4";
            case kFp8e4m3fnCode:
                return "fp8e4m3fn";
            case kFp8e5m2Code:
                return "fp8e5m2";
            case kFp8e8m0Code:
                return "fp8e8m0";
            case kFp4e2m1Code:
                return "fp4e2m1";
            case kFp4e1m2Code:
                return "fp4e1m2";
            case kFp16Code:
                return "fp16";
            case kFp32Code:
                return "fp32";
            case kFp64Code:
                return "fp64";
            case kBf16Code:
                return "bfloat16";
            case kHf4Code:
                return "hf4";
            case kHf8Code:
                return "hf8";
            case kBoolCode:
                return "bool";
            default:
                return "unknown";
        }
    }

    /**
     * \brief Bidirectional map between DataType code (uint8_t) and C-type string.
     *
     * Correspondence follows ToCTypeString(). Forward lookup: code -> string.
     * Reverse lookup: string -> code.
     */
    static inline const npu::tile_fwk::BiMap<uint8_t>& GetDataTypeCTypeStringDict()
    {
        static npu::tile_fwk::BiMap<uint8_t> dict{{
            {kBoolCode, IR_KW_TYPE_BOOL},
            {kInt8Code, IR_KW_TYPE_INT8},
            {kInt16Code, IR_KW_TYPE_INT16},
            {kInt32Code, IR_KW_TYPE_INT32},
            {kInt64Code, IR_KW_TYPE_INT64},
            {kIndexCode, IR_KW_TYPE_INT64},
            {kUInt8Code, IR_KW_TYPE_UINT8},
            {kUInt16Code, IR_KW_TYPE_UINT16},
            {kUInt32Code, IR_KW_TYPE_UINT32},
            {kUInt64Code, IR_KW_TYPE_UINT64},
            {kFp16Code, IR_KW_TYPE_FP16},
            {kFp32Code, IR_KW_TYPE_FP32},
            {kFp64Code, IR_KW_TYPE_FP64},
            {kBf16Code, IR_KW_TYPE_BF16},
            {kFp8e4m3fnCode, IR_KW_TYPE_FP8E4M3FN},
            {kFp8e5m2Code, IR_KW_TYPE_FP8E5M2},
            {kFp8e8m0Code, IR_KW_TYPE_FP8E8M0},
            {kFp4e2m1Code, IR_KW_TYPE_FP4E2M1},
            {kFp4e1m2Code, IR_KW_TYPE_FP4E1M2},
            {kHf4Code, IR_KW_TYPE_UNKNOWN},
            {kHf8Code, IR_KW_TYPE_HF8},
        }};
        return dict;
    }

    /**
     * \brief Get C style type string for code generation
     *
     * Returns the C/C++ type string representation used in code generation.
     * Covers all DataType variants: signed/unsigned integers (incl. INT4/UINT4),
     * IEEE float (FP16, FP32, FP64), FP4/FP8, BF16, HF4/HF8, and BOOL.
     *
     * \return C style type string (e.g. "float", "int32_t", "half", "bfloat16")
     */
    [[nodiscard]] std::string ToCTypeString() const
    {
        return GetDataTypeCTypeStringDict().Find(code_, IR_KW_TYPE_UNKNOWN);
    }

    /**
     * \brief Check if this data type is a floating point type
     *
     * \return true if this is FP4, FP4E2M1, FP4E1M2, FP8, FP8E4M3FN, FP8E5M2, FP8E8M0, FP16, FP32, BF16, HF4, or HF8
     */
    [[nodiscard]] bool IsFloat() const
    {
        // IEEE float types or Brain/Hisilicon float types
        return (code_ >= kIeeeFloatRangeStart && code_ <= kIeeeFloatRangeEnd) ||
               (code_ >= kBrainFloatRangeStart && code_ <= kBrainFloatRangeEnd);
    }

    /**
     * \brief Check if this data type is a signed integer type
     *
     * \return true if this is INT4, INT8, INT16, INT32, or INT64
     */
    [[nodiscard]] bool IsSignedInt() const { return code_ >= kSignedIntRangeStart && code_ <= kSignedIntRangeEnd; }

    /**
     * \brief Check if this data type is an unsigned integer type
     *
     * \return true if this is UINT4, UINT8, UINT16, UINT32, or UINT64
     */
    [[nodiscard]] bool IsUnsignedInt() const
    {
        return code_ >= kUnsignedIntRangeStart && code_ <= kUnsignedIntRangeEnd;
    }

    /**
     * \brief Check if this data type is any integer type (signed or unsigned)
     *
     * \return true if this is any integer type
     */
    [[nodiscard]] bool IsInt() const { return IsSignedInt() || IsUnsignedInt(); }

    /**
     * \brief Equality comparison operator
     *
     * \param other The other DataType to compare with
     * \return true if both types have the same code
     */
    constexpr bool operator==(const DataType& other) const { return code_ == other.code_; }

    /**
     * \brief Inequality comparison operator
     *
     * \param other The other DataType to compare with
     * \return true if types have different codes
     */
    constexpr bool operator!=(const DataType& other) const { return code_ != other.code_; }

    /**
     * \brief Get the underlying type code
     *
     * \return The uint8_t code representing this type
     */
    [[nodiscard]] constexpr uint8_t Code() const { return code_; }

private:
    uint8_t code_; // Internal type code
};

// Static constant definitions
inline constexpr DataType DataType::BOOL = DataType(kBoolCode);
inline constexpr DataType DataType::INT4 = DataType(kInt4Code);
inline constexpr DataType DataType::INT8 = DataType(kInt8Code);
inline constexpr DataType DataType::INT16 = DataType(kInt16Code);
inline constexpr DataType DataType::INT32 = DataType(kInt32Code);
inline constexpr DataType DataType::INT64 = DataType(kInt64Code);
inline constexpr DataType DataType::UINT4 = DataType(kUInt4Code);
inline constexpr DataType DataType::UINT8 = DataType(kUInt8Code);
inline constexpr DataType DataType::UINT16 = DataType(kUInt16Code);
inline constexpr DataType DataType::UINT32 = DataType(kUInt32Code);
inline constexpr DataType DataType::UINT64 = DataType(kUInt64Code);
inline constexpr DataType DataType::FP4 = DataType(kFp4Code);
inline constexpr DataType DataType::FP8E4M3FN = DataType(kFp8e4m3fnCode);
inline constexpr DataType DataType::FP8E5M2 = DataType(kFp8e5m2Code);
inline constexpr DataType DataType::FP8 = DataType(kFp8Code);
inline constexpr DataType DataType::FP8E8M0 = DataType(kFp8e8m0Code);
inline constexpr DataType DataType::FP4E2M1 = DataType(kFp4e2m1Code);
inline constexpr DataType DataType::FP4E1M2 = DataType(kFp4e1m2Code);
inline constexpr DataType DataType::FP16 = DataType(kFp16Code);
inline constexpr DataType DataType::FP32 = DataType(kFp32Code);
inline constexpr DataType DataType::FP64 = DataType(kFp64Code);
inline constexpr DataType DataType::BF16 = DataType(kBf16Code);
inline constexpr DataType DataType::HF4 = DataType(kHf4Code);
inline constexpr DataType DataType::HF8 = DataType(kHf8Code);
inline constexpr DataType DataType::INDEX = DataType(kIndexCode);

/**
 * \brief Convert DataType to its canonical enum name string
 *
 * Returns the uppercase enum-style name for a DataType, suitable for use
 * as a suffix in code generation (e.g., "FP32", "BF16", "INT32").
 *
 * Callers compose the full qualified name:
 *   - Python printer: prefix + "." + DTypeToString(dtype)
 *   - C++ codegen:    "DataType::" + DTypeToString(dtype)
 *
 * \param dtype The data type to convert
 * \return Uppercase enum name string
 */
inline std::string DTypeToString(const DataType& dtype)
{
    if (dtype == DataType::BOOL)
        return "BOOL";
    if (dtype == DataType::INT4)
        return "INT4";
    if (dtype == DataType::INT8)
        return "INT8";
    if (dtype == DataType::INT16)
        return "INT16";
    if (dtype == DataType::INT32)
        return "INT32";
    if (dtype == DataType::INDEX)
        return "INDEX";
    if (dtype == DataType::INT64)
        return "INT64";
    if (dtype == DataType::UINT4)
        return "UINT4";
    if (dtype == DataType::UINT8)
        return "UINT8";
    if (dtype == DataType::UINT16)
        return "UINT16";
    if (dtype == DataType::UINT32)
        return "UINT32";
    if (dtype == DataType::UINT64)
        return "UINT64";
    if (dtype == DataType::FP4)
        return "FP4";
    if (dtype == DataType::FP8E4M3FN)
        return "FP8E4M3FN";
    if (dtype == DataType::FP8E5M2)
        return "FP8E5M2";
    if (dtype == DataType::FP8E8M0)
        return "FP8E8M0";
    if (dtype == DataType::FP4E2M1)
        return "FP4E2M1";
    if (dtype == DataType::FP4E1M2)
        return "FP4E1M2";
    if (dtype == DataType::FP16)
        return "FP16";
    if (dtype == DataType::FP32)
        return "FP32";
    if (dtype == DataType::FP64)
        return "FP64";
    if (dtype == DataType::BF16)
        return "BF16";
    if (dtype == DataType::HF4)
        return "HF4";
    if (dtype == DataType::HF8)
        return "HF8";
    return "UnknownType";
}
} // namespace ir
} // namespace pypto
