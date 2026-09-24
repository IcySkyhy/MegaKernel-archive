/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_element.cpp
 * \brief Unit tests for Element operations in element.cpp to improve code coverage.
 */

#include <gtest/gtest.h>

#include <vector>

#include "interface/inner/element.h"

namespace npu::tile_fwk {

class ElementTest : public testing::Test {};

// ===== Element::Cast<bool> =====
TEST_F(ElementTest, CastBool_SignedBranch)
{
    EXPECT_TRUE(Element(DT_INT32, int64_t(5)).Cast<bool>());
    EXPECT_FALSE(Element(DT_INT32, int64_t(0)).Cast<bool>());
    EXPECT_TRUE(Element(DT_BOOL, int64_t(1)).Cast<bool>());
    EXPECT_FALSE(Element(DT_BOOL, int64_t(0)).Cast<bool>());
}

TEST_F(ElementTest, CastBool_UnsignedBranch)
{
    EXPECT_TRUE(Element(DT_UINT64, uint64_t(5)).Cast<bool>());
    EXPECT_FALSE(Element(DT_UINT64, uint64_t(0)).Cast<bool>());
}

TEST_F(ElementTest, CastBool_FloatBranch)
{
    EXPECT_TRUE(Element(DT_FP32, 3.14).Cast<bool>());
    EXPECT_FALSE(Element(DT_FP32, 0.0).Cast<bool>());
    EXPECT_FALSE(Element(DT_FP32, 1e-12).Cast<bool>());
}

TEST_F(ElementTest, CastBool_InvalidTypeThrows) { EXPECT_THROW(Element().Cast<bool>(), Error); }

// ===== ToJson =====
TEST_F(ElementTest, ToJson_AllCategories)
{
    {
        Json j = ToJson(Element(DT_INT32, int64_t(-5)));
        EXPECT_EQ(j["data_type"].get<int>(), static_cast<int>(DT_INT32));
        EXPECT_EQ(j["value"].get<int64_t>(), -5);
    }
    {
        Json j = ToJson(Element(DT_UINT64, uint64_t(7)));
        EXPECT_EQ(j["data_type"].get<int>(), static_cast<int>(DT_UINT64));
        EXPECT_EQ(j["value"].get<uint64_t>(), 7ULL);
    }
    {
        Json j = ToJson(Element(DT_FP32, 2.5));
        EXPECT_EQ(j["data_type"].get<int>(), static_cast<int>(DT_FP32));
        EXPECT_DOUBLE_EQ(j["value"].get<double>(), 2.5);
    }
    {
        Json j = ToJson(Element());
        EXPECT_TRUE(j["value"].is_null());
    }
}

// ===== parseElement =====
TEST_F(ElementTest, ParseElement_MissingFieldsThrows)
{
    {
        Json j;
        j["value"] = 5;
        EXPECT_THROW(parseElement(j), std::invalid_argument);
    }
    {
        Json j;
        j["data_type"] = static_cast<int>(DT_INT32);
        EXPECT_THROW(parseElement(j), std::invalid_argument);
    }
}

TEST_F(ElementTest, ParseElement_IntegerAndFloat)
{
    {
        Json j;
        j["data_type"] = static_cast<int>(DT_INT32);
        j["value"] = -42;
        Element e = parseElement(j);
        EXPECT_EQ(e.GetDataType(), DT_INT32);
        EXPECT_EQ(e.GetSignedData(), -42);
    }
    {
        Json j;
        j["data_type"] = static_cast<int>(DT_FP32);
        j["value"] = 1.25;
        Element e = parseElement(j);
        EXPECT_EQ(e.GetDataType(), DT_FP32);
        EXPECT_DOUBLE_EQ(e.GetFloatData(), 1.25);
    }
}

TEST_F(ElementTest, ParseElement_NullValueReturnsDefault)
{
    Json j;
    j["data_type"] = static_cast<int>(DT_INT32);
    j["value"] = nullptr;
    Element e = parseElement(j);
    EXPECT_EQ(e.GetDataType(), DT_BOTTOM);
}

TEST_F(ElementTest, ParseElement_UnsupportedTypeThrows)
{
    {
        Json j;
        j["data_type"] = static_cast<int>(DT_INT32);
        j["value"] = true;
        EXPECT_THROW(parseElement(j), std::runtime_error);
    }
    {
        Json j;
        j["data_type"] = static_cast<int>(DT_INT32);
        j["value"] = "not_a_number";
        EXPECT_THROW(parseElement(j), std::runtime_error);
    }
}

// ===== Element::Abs (value1 < value2 branches) =====
TEST_F(ElementTest, Abs_Uint64_SwappedBranch)
{
    EXPECT_FALSE(Element(DT_UINT64, uint64_t(5)) == Element(DT_UINT64, uint64_t(10)));
    EXPECT_TRUE(Element(DT_UINT64, uint64_t(5)) != Element(DT_UINT64, uint64_t(10)));
}

TEST_F(ElementTest, Abs_Double_SwappedBranch)
{
    EXPECT_FALSE(Element(DT_FP32, 1.5) == Element(DT_FP32, 3.5));
    EXPECT_TRUE(Element(DT_FP32, 1.5) != Element(DT_FP32, 3.5));
}

// ===== Arithmetic operators: unsigned branches =====
TEST_F(ElementTest, Arithmetic_UnsignedBranches)
{
    EXPECT_EQ((Element(DT_UINT64, uint64_t(10)) + Element(DT_UINT64, uint64_t(5))).GetUnsignedData(), 15ULL);
    EXPECT_EQ((Element(DT_UINT64, uint64_t(10)) - Element(DT_UINT64, uint64_t(5))).GetUnsignedData(), 5ULL);
    EXPECT_EQ((Element(DT_UINT64, uint64_t(10)) * Element(DT_UINT64, uint64_t(5))).GetUnsignedData(), 50ULL);
    EXPECT_EQ((Element(DT_UINT64, uint64_t(10)) / Element(DT_UINT64, uint64_t(5))).GetUnsignedData(), 2ULL);
    EXPECT_EQ((Element(DT_UINT64, uint64_t(10)) % Element(DT_UINT64, uint64_t(3))).GetUnsignedData(), 1ULL);
}

// ===== Arithmetic operators: invalid dtype (else) branches throw =====
TEST_F(ElementTest, Arithmetic_InvalidTypeThrows)
{
    EXPECT_THROW(Element() + Element(), Error);
    EXPECT_THROW(Element() - Element(), Error);
    EXPECT_THROW(Element() * Element(), Error);
    EXPECT_THROW(Element() / Element(), Error);
    EXPECT_THROW(Element() % Element(), Error);
}

// ===== Comparison operators: unsigned branches =====
TEST_F(ElementTest, Comparison_UnsignedBranches)
{
    EXPECT_TRUE(Element(DT_UINT64, uint64_t(7)) == Element(DT_UINT64, uint64_t(7)));
    EXPECT_FALSE(Element(DT_UINT64, uint64_t(5)) == Element(DT_UINT64, uint64_t(10)));

    EXPECT_TRUE(Element(DT_UINT64, uint64_t(5)) != Element(DT_UINT64, uint64_t(10)));
    EXPECT_FALSE(Element(DT_UINT64, uint64_t(7)) != Element(DT_UINT64, uint64_t(7)));

    EXPECT_TRUE(Element(DT_UINT64, uint64_t(5)) < Element(DT_UINT64, uint64_t(10)));
    EXPECT_FALSE(Element(DT_UINT64, uint64_t(10)) < Element(DT_UINT64, uint64_t(5)));

    EXPECT_TRUE(Element(DT_UINT64, uint64_t(5)) <= Element(DT_UINT64, uint64_t(10)));
    EXPECT_TRUE(Element(DT_UINT64, uint64_t(7)) <= Element(DT_UINT64, uint64_t(7)));

    EXPECT_TRUE(Element(DT_UINT64, uint64_t(10)) > Element(DT_UINT64, uint64_t(5)));
    EXPECT_FALSE(Element(DT_UINT64, uint64_t(5)) > Element(DT_UINT64, uint64_t(10)));

    EXPECT_TRUE(Element(DT_UINT64, uint64_t(10)) >= Element(DT_UINT64, uint64_t(5)));
    EXPECT_TRUE(Element(DT_UINT64, uint64_t(7)) >= Element(DT_UINT64, uint64_t(7)));
}

// ===== Comparison operators: invalid dtype (else) branches throw =====
TEST_F(ElementTest, Comparison_InvalidTypeThrows)
{
    EXPECT_THROW([] { (void)(Element() == Element()); }(), Error);
    EXPECT_THROW([] { (void)(Element() != Element()); }(), Error);
    EXPECT_THROW([] { (void)(Element() < Element()); }(), Error);
    EXPECT_THROW([] { (void)(Element() <= Element()); }(), Error);
    EXPECT_THROW([] { (void)(Element() > Element()); }(), Error);
    EXPECT_THROW([] { (void)(Element() >= Element()); }(), Error);
}

// ===== ConvElementVecToIntVec / ConvElementVecToFloatVec =====
TEST_F(ElementTest, ConvElementVecToIntVec_BasicAndEmpty)
{
    EXPECT_TRUE(ConvElementVecToIntVec({}).empty());

    std::vector<Element> inputs = {Element(DT_INT32, int64_t(3)), Element(DT_UINT64, uint64_t(7)),
                                   Element(DT_FP32, 2.5)};
    std::vector<int> res = ConvElementVecToIntVec(inputs);
    EXPECT_EQ(res.size(), 3u);
    EXPECT_EQ(res[0], 3);
    EXPECT_EQ(res[1], 7);
    EXPECT_EQ(res[2], 2);
}

TEST_F(ElementTest, ConvElementVecToFloatVec_BasicAndEmpty)
{
    EXPECT_TRUE(ConvElementVecToFloatVec({}).empty());

    std::vector<Element> inputs = {Element(DT_INT32, int64_t(3)), Element(DT_UINT64, uint64_t(7)),
                                   Element(DT_FP32, 2.5)};
    std::vector<float> res = ConvElementVecToFloatVec(inputs);
    EXPECT_EQ(res.size(), 3u);
    EXPECT_FLOAT_EQ(res[0], 3.0f);
    EXPECT_FLOAT_EQ(res[1], 7.0f);
    EXPECT_FLOAT_EQ(res[2], 2.5f);
}

} // namespace npu::tile_fwk
