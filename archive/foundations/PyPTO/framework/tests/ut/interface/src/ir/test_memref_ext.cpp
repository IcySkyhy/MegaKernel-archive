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
 * \file test_memref_ext.cpp
 * \brief Coverage tests for memref.cpp (MayAlias, SameAllocation, StringToMemorySpace)
 */

#include "gtest/gtest.h"

#include <memory>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/memref.h"
#include "ir/scalar_expr.h"

namespace pypto {
namespace ir {

static MemRefPtr MakeMemRef(MemorySpace space, int64_t offset, uint64_t size)
{
    return std::make_shared<MemRef>(space, std::make_shared<ConstInt>(offset, DataType::INT64, Span::Unknown()), size);
}

// ============================================================================
// StringToMemorySpace
// ============================================================================

class MemRefStringTest : public testing::Test {};

TEST_F(MemRefStringTest, TestAllMemorySpaceStrings)
{
    ASSERT_EQ(StringToMemorySpace("DDR"), MemorySpace::DDR);
    ASSERT_EQ(StringToMemorySpace("Vec"), MemorySpace::Vec);
    ASSERT_EQ(StringToMemorySpace("Mat"), MemorySpace::Mat);
    ASSERT_EQ(StringToMemorySpace("Left"), MemorySpace::Left);
    ASSERT_EQ(StringToMemorySpace("Right"), MemorySpace::Right);
    ASSERT_EQ(StringToMemorySpace("Acc"), MemorySpace::Acc);
    ASSERT_EQ(StringToMemorySpace("Bias"), MemorySpace::Bias);
}

TEST_F(MemRefStringTest, TestUnknownStringThrows) { ASSERT_THROW(StringToMemorySpace("Invalid"), ValueError); }

TEST_F(MemRefStringTest, TestMemorySpaceToStringBias) { ASSERT_EQ(MemorySpaceToString(MemorySpace::Bias), "Bias"); }

TEST_F(MemRefStringTest, TestMemorySpaceToStringDefault)
{
    auto unknown = static_cast<MemorySpace>(99);
    ASSERT_EQ(MemorySpaceToString(unknown), "Unknown");
}

// ============================================================================
// MemRef::MayAlias
// ============================================================================

class MemRefAliasTest : public testing::Test {};

TEST_F(MemRefAliasTest, TestDifferentSpacesNoAlias)
{
    auto a = MakeMemRef(MemorySpace::DDR, 0, 100);
    auto b = MakeMemRef(MemorySpace::Vec, 0, 100);
    ASSERT_FALSE(MemRef::MayAlias(a, b));
}

TEST_F(MemRefAliasTest, TestSameSpaceNoOverlap)
{
    auto a = MakeMemRef(MemorySpace::DDR, 0, 100);
    auto b = MakeMemRef(MemorySpace::DDR, 200, 100);
    ASSERT_FALSE(MemRef::MayAlias(a, b));
}

TEST_F(MemRefAliasTest, TestSameSpaceOverlapping)
{
    auto a = MakeMemRef(MemorySpace::DDR, 0, 100);
    auto b = MakeMemRef(MemorySpace::DDR, 50, 100);
    ASSERT_TRUE(MemRef::MayAlias(a, b));
}

TEST_F(MemRefAliasTest, TestSameSpaceAdjacent)
{
    auto a = MakeMemRef(MemorySpace::DDR, 0, 100);
    auto b = MakeMemRef(MemorySpace::DDR, 100, 100);
    ASSERT_FALSE(MemRef::MayAlias(a, b));
}

TEST_F(MemRefAliasTest, TestSymbolicOffsetAlias)
{
    auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT64), Span::Unknown());
    auto a = std::make_shared<MemRef>(MemorySpace::DDR, var, 100);
    auto b = MakeMemRef(MemorySpace::DDR, 0, 100);
    ASSERT_TRUE(MemRef::MayAlias(a, b));
}

// ============================================================================
// MemRef::SameAllocation
// ============================================================================

class MemRefAllocTest : public testing::Test {};

TEST_F(MemRefAllocTest, TestSameAllocation)
{
    auto a = MakeMemRef(MemorySpace::DDR, 0, 1024);
    auto b = MakeMemRef(MemorySpace::DDR, 0, 1024);
    ASSERT_TRUE(MemRef::SameAllocation(a, b));
}

TEST_F(MemRefAllocTest, TestDifferentOffset)
{
    auto a = MakeMemRef(MemorySpace::DDR, 0, 1024);
    auto b = MakeMemRef(MemorySpace::DDR, 100, 1024);
    ASSERT_FALSE(MemRef::SameAllocation(a, b));
}

TEST_F(MemRefAllocTest, TestDifferentSize)
{
    auto a = MakeMemRef(MemorySpace::DDR, 0, 1024);
    auto b = MakeMemRef(MemorySpace::DDR, 0, 2048);
    ASSERT_FALSE(MemRef::SameAllocation(a, b));
}

TEST_F(MemRefAllocTest, TestDifferentSpace)
{
    auto a = MakeMemRef(MemorySpace::DDR, 0, 1024);
    auto b = MakeMemRef(MemorySpace::Vec, 0, 1024);
    ASSERT_FALSE(MemRef::SameAllocation(a, b));
}

TEST_F(MemRefAllocTest, TestSymbolicOffsetSamePointer)
{
    auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT64), Span::Unknown());
    auto a = std::make_shared<MemRef>(MemorySpace::DDR, var, 1024);
    auto b = std::make_shared<MemRef>(MemorySpace::DDR, var, 1024);
    ASSERT_TRUE(MemRef::SameAllocation(a, b));
}

TEST_F(MemRefAllocTest, TestSymbolicOffsetDifferentPointer)
{
    auto var1 = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT64), Span::Unknown());
    auto var2 = std::make_shared<Var>("y", std::make_shared<ScalarType>(DataType::INT64), Span::Unknown());
    auto a = std::make_shared<MemRef>(MemorySpace::DDR, var1, 1024);
    auto b = std::make_shared<MemRef>(MemorySpace::DDR, var2, 1024);
    ASSERT_FALSE(MemRef::SameAllocation(a, b));
}

} // namespace ir
} // namespace pypto
