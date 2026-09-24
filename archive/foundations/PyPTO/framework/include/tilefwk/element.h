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
 * \file element.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <variant>

#include "tilefwk/data_type.h"
#include "tilefwk/error.h"

namespace npu::tile_fwk {
class Element {
public:
    Element() : type_(DT_BOTTOM) {}

    explicit Element(DataType type, int16_t sData) { Init(type, sData); }
    explicit Element(DataType type, int32_t sData) { Init(type, sData); }
    explicit Element(DataType type, int64_t sData) { Init(type, sData); }
    explicit Element(DataType type, uint16_t uData) { Init(type, uData); }
    explicit Element(DataType type, uint32_t uData) { Init(type, uData); }
    explicit Element(DataType type, uint64_t uData) { Init(type, uData); }
    explicit Element(DataType type, double fData) { Init(type, fData); }

    DataType GetDataType() const { return type_; }
    int64_t GetSignedData() const { return data_.sData; }
    uint64_t GetUnsignedData() const { return data_.uData; }
    double GetFloatData() const { return data_.fData; }
    std::variant<int64_t, uint64_t, double> GetVariantData() const
    {
        if (IsSigned()) {
            return static_cast<int64_t>(data_.sData);
        } else if (IsUnsigned()) {
            return static_cast<uint64_t>(data_.uData);
        } else if (IsFloat()) {
            return static_cast<double>(data_.fData);
        }
        FE_ASSERT(false);
        return int64_t(0);
    }

    bool IsSigned() const
    {
        return type_ == DT_INT4 || type_ == DT_INT8 || type_ == DT_INT16 || type_ == DT_INT32 || type_ == DT_INT64 ||
               type_ == DT_BOOL;
    }
    bool IsUnsigned() const
    {
        return type_ == DT_UINT8 || type_ == DT_UINT16 || type_ == DT_UINT32 || type_ == DT_UINT64;
    }
    bool IsFloat() const
    {
        return type_ == DT_FP8 || type_ == DT_FP16 || type_ == DT_FP32 || type_ == DT_BF16 || type_ == DT_HF4 ||
               type_ == DT_HF8 || type_ == DT_DOUBLE;
    }

    template <typename T>
    T Cast() const;
    Element operator+(const Element& rhs) const;
    Element operator-(const Element& rhs) const;
    Element operator*(const Element& rhs) const;
    Element operator/(const Element& rhs) const;
    Element operator%(const Element& rhs) const;
    bool operator==(const Element& rhs) const;
    bool operator!=(const Element& rhs) const;
    bool operator<(const Element& rhs) const;
    bool operator<=(const Element& rhs) const;
    bool operator>(const Element& rhs) const;
    bool operator>=(const Element& rhs) const;

    uint64_t Abs(uint64_t value1, uint64_t value2) const;
    int64_t Abs(int64_t value1, int64_t value2) const;
    double Abs(double value1, double value2) const;

private:
    template <typename T>
    void Init(DataType type, T value)
    {
        type_ = type;
        if (IsSigned()) {
            data_.sData = static_cast<int64_t>(value);
        } else if (IsUnsigned()) {
            data_.uData = static_cast<uint64_t>(value);
        } else {
            data_.fData = static_cast<double>(value);
        }
    }
    union {
        int64_t sData;
        uint64_t uData;
        double fData;
    } data_;
    DataType type_;
};
} // namespace npu::tile_fwk
