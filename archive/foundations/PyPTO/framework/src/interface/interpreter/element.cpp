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
 * \file element.cpp
 * \brief
 */

#include "interface/inner/element.h"
#include "tilefwk/error.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

constexpr double D_EPSILON = 1e-9;

#define ELEMENT_CAST(ast2Type, type, calcType)                  \
    template <>                                                 \
    type Element::Cast<type>() const                            \
    {                                                           \
        type result{0};                                         \
        if (IsSigned()) {                                       \
            result = static_cast<type>(data_.sData);            \
        } else if (IsUnsigned()) {                              \
            result = static_cast<type>(data_.uData);            \
        } else if (IsFloat()) {                                 \
            result = static_cast<type>(data_.fData);            \
        } else {                                                \
            ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false); \
        }                                                       \
        return result;                                          \
    }

// custom cast of bool type
template <>
bool Element::Cast<bool>() const
{
    if (IsSigned()) {
        return data_.sData != 0;
    } else if (IsUnsigned()) {
        return data_.uData != 0;
    } else if (IsFloat()) {
        return std::abs(data_.fData) > D_EPSILON;
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return false;
}

DISPATCH_DATA_TYPE(ELEMENT_CAST);

#define CALC_ADD(lhs, rhs) ((lhs) + (rhs))
#define CALC_SUB(lhs, rhs) ((lhs) - (rhs))
#define CALC_MUL(lhs, rhs) ((lhs) * (rhs))
#define CALC_DIV(lhs, rhs) ((lhs) / (rhs))
#define CALC_MOD(lhs, rhs) ((lhs) % (rhs))
#define CALC_EQ(lhs, rhs) (Element::Abs((lhs), (rhs)) < D_EPSILON)
#define CALC_NE(lhs, rhs) (Element::Abs((lhs), (rhs)) > D_EPSILON)
#define CALC_LT(lhs, rhs) ((lhs) < (rhs))
#define CALC_LE(lhs, rhs) ((lhs) <= (rhs))
#define CALC_GT(lhs, rhs) ((lhs) > (rhs))
#define CALC_GE(lhs, rhs) ((lhs) >= (rhs))

Json ToJson(const Element& elem)
{
    Json j;
    j["data_type"] = static_cast<int>(elem.GetDataType());

    if (elem.IsSigned()) {
        j["value"] = elem.GetSignedData();
    } else if (elem.IsUnsigned()) {
        j["value"] = elem.GetUnsignedData();
    } else if (elem.IsFloat()) {
        j["value"] = elem.GetFloatData();
    } else {
        j["value"] = nullptr; // 处理未定义类型
    }
    return j;
}

Element parseElement(const Json& j)
{
    if (!j.contains("data_type") || !j.contains("value")) {
        throw std::invalid_argument("Invalid JSON: Missing required fields");
    }
    DataType type = static_cast<DataType>(j["data_type"].get<int>());
    if (j["value"].is_number_integer()) {
        return Element(type, j["value"].get<int64_t>());
    } else if (j["value"].is_number_unsigned()) {
        return Element(type, j["value"].get<uint64_t>());
    } else if (j["value"].is_number_float()) {
        return Element(type, j["value"].get<double>());
    } else if (j["value"].is_null()) {
        return Element(); // 默认构造
    }
    throw std::runtime_error("Unsupported value type in JSON");
}

uint64_t Element::Abs(uint64_t value1, uint64_t value2) const
{
    if (value1 < value2) {
        return value2 - value1;
    }
    return value1 - value2;
}

int64_t Element::Abs(int64_t value1, int64_t value2) const
{
    if (value1 < value2) {
        return value2 - value1;
    }
    return value1 - value2;
}

double Element::Abs(double value1, double value2) const
{
    if (value1 < value2) {
        return value2 - value1;
    }
    return value1 - value2;
}

Element Element::operator+(const Element& rhs) const
{
    ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, GetDataType() == rhs.GetDataType());
    if (IsSigned()) {
        return Element(GetDataType(), CALC_ADD(GetSignedData(), rhs.GetSignedData()));
    } else if (IsUnsigned()) {
        return Element(GetDataType(), CALC_ADD(GetUnsignedData(), rhs.GetUnsignedData()));
    } else if (IsFloat()) {
        return Element(GetDataType(), CALC_ADD(GetFloatData(), rhs.GetFloatData()));
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return Element();
}

Element Element::operator-(const Element& rhs) const
{
    ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, GetDataType() == rhs.GetDataType());
    if (IsSigned()) {
        return Element(GetDataType(), CALC_SUB(GetSignedData(), rhs.GetSignedData()));
    } else if (IsUnsigned()) {
        return Element(GetDataType(), CALC_SUB(GetUnsignedData(), rhs.GetUnsignedData()));
    } else if (IsFloat()) {
        return Element(GetDataType(), CALC_SUB(GetFloatData(), rhs.GetFloatData()));
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return Element();
}

Element Element::operator*(const Element& rhs) const
{
    ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, GetDataType() == rhs.GetDataType());
    if (IsSigned()) {
        return Element(GetDataType(), CALC_MUL(GetSignedData(), rhs.GetSignedData()));
    } else if (IsUnsigned()) {
        return Element(GetDataType(), CALC_MUL(GetUnsignedData(), rhs.GetUnsignedData()));
    } else if (IsFloat()) {
        return Element(GetDataType(), CALC_MUL(GetFloatData(), rhs.GetFloatData()));
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return Element();
}

Element Element::operator/(const Element& rhs) const
{
    ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, GetDataType() == rhs.GetDataType());
    if (IsSigned()) {
        return Element(GetDataType(), CALC_DIV(GetSignedData(), rhs.GetSignedData()));
    } else if (IsUnsigned()) {
        return Element(GetDataType(), CALC_DIV(GetUnsignedData(), rhs.GetUnsignedData()));
    } else if (IsFloat()) {
        return Element(GetDataType(), CALC_DIV(GetFloatData(), rhs.GetFloatData()));
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return Element();
}

Element Element::operator%(const Element& rhs) const
{
    ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, GetDataType() == rhs.GetDataType());
    if (IsSigned()) {
        return Element(GetDataType(), CALC_MOD(GetSignedData(), rhs.GetSignedData()));
    } else if (IsUnsigned()) {
        return Element(GetDataType(), CALC_MOD(GetUnsignedData(), rhs.GetUnsignedData()));
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return Element();
}

bool Element::operator==(const Element& rhs) const
{
    ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, GetDataType() == rhs.GetDataType());
    if (IsSigned()) {
        return CALC_EQ(GetSignedData(), rhs.GetSignedData());
    } else if (IsUnsigned()) {
        return CALC_EQ(GetUnsignedData(), rhs.GetUnsignedData());
    } else if (IsFloat()) {
        return CALC_EQ(GetFloatData(), rhs.GetFloatData());
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return false;
}

bool Element::operator!=(const Element& rhs) const
{
    ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, GetDataType() == rhs.GetDataType());
    if (IsSigned()) {
        return CALC_NE(GetSignedData(), rhs.GetSignedData());
    } else if (IsUnsigned()) {
        return CALC_NE(GetUnsignedData(), rhs.GetUnsignedData());
    } else if (IsFloat()) {
        return CALC_NE(GetFloatData(), rhs.GetFloatData());
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return false;
}

bool Element::operator<(const Element& rhs) const
{
    ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, GetDataType() == rhs.GetDataType());
    if (IsSigned()) {
        return CALC_LT(GetSignedData(), rhs.GetSignedData());
    } else if (IsUnsigned()) {
        return CALC_LT(GetUnsignedData(), rhs.GetUnsignedData());
    } else if (IsFloat()) {
        return CALC_LT(GetFloatData(), rhs.GetFloatData());
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return false;
}

bool Element::operator<=(const Element& rhs) const
{
    ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, GetDataType() == rhs.GetDataType());
    if (IsSigned()) {
        return CALC_LE(GetSignedData(), rhs.GetSignedData());
    } else if (IsUnsigned()) {
        return CALC_LE(GetUnsignedData(), rhs.GetUnsignedData());
    } else if (IsFloat()) {
        return CALC_LE(GetFloatData(), rhs.GetFloatData());
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return false;
}

bool Element::operator>(const Element& rhs) const
{
    ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, GetDataType() == rhs.GetDataType());
    if (IsSigned()) {
        return CALC_GT(GetSignedData(), rhs.GetSignedData());
    } else if (IsUnsigned()) {
        return CALC_GT(GetUnsignedData(), rhs.GetUnsignedData());
    } else if (IsFloat()) {
        return CALC_GT(GetFloatData(), rhs.GetFloatData());
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return false;
}

bool Element::operator>=(const Element& rhs) const
{
    ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, GetDataType() == rhs.GetDataType());
    if (IsSigned()) {
        return CALC_GE(GetSignedData(), rhs.GetSignedData());
    } else if (IsUnsigned()) {
        return CALC_GE(GetUnsignedData(), rhs.GetUnsignedData());
    } else if (IsFloat()) {
        return CALC_GE(GetFloatData(), rhs.GetFloatData());
    } else {
        ASSERT(ElementScene::INVALID_ELEMENT_DTYPE, false);
    }
    return false;
}

std::vector<int> ConvElementVecToIntVec(const std::vector<Element>& input)
{
    std::vector<int> res;
    res.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        res[i] = input[i].Cast<int>();
    }
    return res;
}

std::vector<float> ConvElementVecToFloatVec(const std::vector<Element>& input)
{
    std::vector<float> res(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        res[i] = input[i].Cast<float>();
    }
    return res;
}

} // namespace npu::tile_fwk
