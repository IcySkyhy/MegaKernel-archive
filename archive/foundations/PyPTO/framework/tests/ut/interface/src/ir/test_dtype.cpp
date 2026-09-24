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
 * \file test_dtype.cpp
 * \brief Unit tests for DataType class
 */

#include "gtest/gtest.h"

#include "core/dtype.h"

namespace pypto {
namespace ir {

TEST(CoreDTypeTest, TestDataTypeConstants)
{
    // Test that all data type constants are defined
    ASSERT_EQ(DataType::BOOL.Code(), DataType::kBoolCode);
    ASSERT_EQ(DataType::INT4.Code(), DataType::kInt4Code);
    ASSERT_EQ(DataType::INT8.Code(), DataType::kInt8Code);
    ASSERT_EQ(DataType::INT16.Code(), DataType::kInt16Code);
    ASSERT_EQ(DataType::INT32.Code(), DataType::kInt32Code);
    ASSERT_EQ(DataType::INT64.Code(), DataType::kInt64Code);
    ASSERT_EQ(DataType::UINT4.Code(), DataType::kUInt4Code);
    ASSERT_EQ(DataType::UINT8.Code(), DataType::kUInt8Code);
    ASSERT_EQ(DataType::UINT16.Code(), DataType::kUInt16Code);
    ASSERT_EQ(DataType::UINT32.Code(), DataType::kUInt32Code);
    ASSERT_EQ(DataType::UINT64.Code(), DataType::kUInt64Code);
    ASSERT_EQ(DataType::FP4.Code(), DataType::kFp4Code);
    ASSERT_EQ(DataType::FP8.Code(), DataType::kFp8Code);
    ASSERT_EQ(DataType::FP16.Code(), DataType::kFp16Code);
    ASSERT_EQ(DataType::FP32.Code(), DataType::kFp32Code);
    ASSERT_EQ(DataType::BF16.Code(), DataType::kBf16Code);
    ASSERT_EQ(DataType::HF4.Code(), DataType::kHf4Code);
    ASSERT_EQ(DataType::HF8.Code(), DataType::kHf8Code);
}

TEST(CoreDTypeTest, TestGetBit)
{
    // Test GetBit() for all data types
    ASSERT_EQ(DataType::BOOL.GetBit(), 8);
    ASSERT_EQ(DataType::INT4.GetBit(), 4);
    ASSERT_EQ(DataType::INT8.GetBit(), 8);
    ASSERT_EQ(DataType::INT16.GetBit(), 16);
    ASSERT_EQ(DataType::INT32.GetBit(), 32);
    ASSERT_EQ(DataType::INT64.GetBit(), 64);
    ASSERT_EQ(DataType::UINT4.GetBit(), 4);
    ASSERT_EQ(DataType::UINT8.GetBit(), 8);
    ASSERT_EQ(DataType::UINT16.GetBit(), 16);
    ASSERT_EQ(DataType::UINT32.GetBit(), 32);
    ASSERT_EQ(DataType::UINT64.GetBit(), 64);
    ASSERT_EQ(DataType::FP4.GetBit(), 4);
    ASSERT_EQ(DataType::FP8.GetBit(), 8);
    ASSERT_EQ(DataType::FP16.GetBit(), 16);
    ASSERT_EQ(DataType::FP32.GetBit(), 32);
    ASSERT_EQ(DataType::BF16.GetBit(), 16);
    ASSERT_EQ(DataType::HF4.GetBit(), 4);
    ASSERT_EQ(DataType::HF8.GetBit(), 8);
}

TEST(CoreDTypeTest, TestToString)
{
    // Test ToString() for all data types
    ASSERT_EQ(DataType::BOOL.ToString(), "bool");
    ASSERT_EQ(DataType::INT4.ToString(), "int4");
    ASSERT_EQ(DataType::INT8.ToString(), "int8");
    ASSERT_EQ(DataType::INT16.ToString(), "int16");
    ASSERT_EQ(DataType::INT32.ToString(), "int32");
    ASSERT_EQ(DataType::INT64.ToString(), "int64");
    ASSERT_EQ(DataType::UINT4.ToString(), "uint4");
    ASSERT_EQ(DataType::UINT8.ToString(), "uint8");
    ASSERT_EQ(DataType::UINT16.ToString(), "uint16");
    ASSERT_EQ(DataType::UINT32.ToString(), "uint32");
    ASSERT_EQ(DataType::UINT64.ToString(), "uint64");
    ASSERT_EQ(DataType::FP4.ToString(), "fp4");
    ASSERT_EQ(DataType::FP8.ToString(), "fp8e4m3fn");
    ASSERT_EQ(DataType::FP16.ToString(), "fp16");
    ASSERT_EQ(DataType::FP32.ToString(), "fp32");
    ASSERT_EQ(DataType::FP64.ToString(), "fp64");
    ASSERT_EQ(DataType::BF16.ToString(), "bfloat16");
    ASSERT_EQ(DataType::HF4.ToString(), "hf4");
    ASSERT_EQ(DataType::HF8.ToString(), "hf8");
}

TEST(CoreDTypeTest, TestIsFloat)
{
    // Test IsFloat()
    ASSERT_TRUE(DataType::FP4.IsFloat());
    ASSERT_TRUE(DataType::FP8.IsFloat());
    ASSERT_TRUE(DataType::FP16.IsFloat());
    ASSERT_TRUE(DataType::FP32.IsFloat());
    ASSERT_TRUE(DataType::BF16.IsFloat());
    ASSERT_TRUE(DataType::HF4.IsFloat());
    ASSERT_TRUE(DataType::HF8.IsFloat());

    ASSERT_FALSE(DataType::BOOL.IsFloat());
    ASSERT_FALSE(DataType::INT4.IsFloat());
    ASSERT_FALSE(DataType::INT8.IsFloat());
    ASSERT_FALSE(DataType::INT32.IsFloat());
    ASSERT_FALSE(DataType::UINT8.IsFloat());
    ASSERT_FALSE(DataType::UINT32.IsFloat());
}

TEST(CoreDTypeTest, TestIsSignedInt)
{
    // Test IsSignedInt()
    ASSERT_TRUE(DataType::INT4.IsSignedInt());
    ASSERT_TRUE(DataType::INT8.IsSignedInt());
    ASSERT_TRUE(DataType::INT16.IsSignedInt());
    ASSERT_TRUE(DataType::INT32.IsSignedInt());
    ASSERT_TRUE(DataType::INT64.IsSignedInt());

    ASSERT_FALSE(DataType::UINT4.IsSignedInt());
    ASSERT_FALSE(DataType::UINT8.IsSignedInt());
    ASSERT_FALSE(DataType::UINT32.IsSignedInt());
    ASSERT_FALSE(DataType::FP32.IsSignedInt());
    ASSERT_FALSE(DataType::BOOL.IsSignedInt());
}

TEST(CoreDTypeTest, TestIsUnsignedInt)
{
    // Test IsUnsignedInt()
    ASSERT_TRUE(DataType::UINT4.IsUnsignedInt());
    ASSERT_TRUE(DataType::UINT8.IsUnsignedInt());
    ASSERT_TRUE(DataType::UINT16.IsUnsignedInt());
    ASSERT_TRUE(DataType::UINT32.IsUnsignedInt());
    ASSERT_TRUE(DataType::UINT64.IsUnsignedInt());

    ASSERT_FALSE(DataType::INT4.IsUnsignedInt());
    ASSERT_FALSE(DataType::INT8.IsUnsignedInt());
    ASSERT_FALSE(DataType::INT32.IsUnsignedInt());
    ASSERT_FALSE(DataType::FP32.IsUnsignedInt());
    ASSERT_FALSE(DataType::BOOL.IsUnsignedInt());
}

TEST(CoreDTypeTest, TestIsInt)
{
    // Test IsInt() - both signed and unsigned
    ASSERT_TRUE(DataType::INT4.IsInt());
    ASSERT_TRUE(DataType::INT8.IsInt());
    ASSERT_TRUE(DataType::INT32.IsInt());
    ASSERT_TRUE(DataType::INT64.IsInt());
    ASSERT_TRUE(DataType::UINT4.IsInt());
    ASSERT_TRUE(DataType::UINT8.IsInt());
    ASSERT_TRUE(DataType::UINT32.IsInt());
    ASSERT_TRUE(DataType::UINT64.IsInt());

    ASSERT_FALSE(DataType::FP32.IsInt());
    ASSERT_FALSE(DataType::FP16.IsInt());
    ASSERT_FALSE(DataType::BF16.IsInt());
    ASSERT_FALSE(DataType::BOOL.IsInt());
}

TEST(CoreDTypeTest, TestEquality)
{
    // Test equality operators
    ASSERT_EQ(DataType::INT32, DataType::INT32);
    ASSERT_EQ(DataType::FP32, DataType::FP32);
    ASSERT_NE(DataType::INT32, DataType::FP32);
    ASSERT_NE(DataType::INT32, DataType::INT64);

    // Test with constructed types
    DataType dt1(DataType::kInt32Code);
    DataType dt2(DataType::kInt32Code);
    ASSERT_EQ(dt1, dt2);
    ASSERT_EQ(dt1, DataType::INT32);
}

TEST(CoreDTypeTest, TestCodeMethod)
{
    // Test Code() method
    ASSERT_EQ(DataType::INT32.Code(), DataType::kInt32Code);
    ASSERT_EQ(DataType::FP32.Code(), DataType::kFp32Code);
    ASSERT_EQ(DataType::BOOL.Code(), DataType::kBoolCode);

    // Test that different types have different codes
    ASSERT_NE(DataType::INT32.Code(), DataType::INT64.Code());
    ASSERT_NE(DataType::INT32.Code(), DataType::FP32.Code());
}

TEST(CoreDTypeTest, TestDefaultConstructor)
{
    // Default constructor should create BOOL type
    DataType dt;
    ASSERT_EQ(dt, DataType::BOOL);
    ASSERT_EQ(dt.Code(), DataType::kBoolCode);
}

} // namespace ir
} // namespace pypto
