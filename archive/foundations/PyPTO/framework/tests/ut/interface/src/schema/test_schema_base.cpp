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
 * \file test_schema_base.cpp
 * \brief Unit tests for schema type system: Dump() methods, UnionType, CoordType, ArrayType, AttributeId, etc.
 */

#include <cstdint>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "interface/schema/schema.h"

using namespace npu::tile_fwk::schema;

class TestSchemaBase : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestSchemaBase, Int32TypeDump)
{
    Int32Type val(42);
    EXPECT_EQ(val.Dump(), "42");
    Int32Type neg(-7);
    EXPECT_EQ(neg.Dump(), "-7");
    Int32Type zero(0);
    EXPECT_EQ(zero.Dump(), "0");
}

TEST_F(TestSchemaBase, Int64TypeDumpSmallValue)
{
    Int64Type val(123);
    EXPECT_EQ(val.Dump(), "123");
    Int64Type neg(-1);
    EXPECT_EQ(neg.Dump(), "-1");
}

TEST_F(TestSchemaBase, UInt32TypeDump)
{
    UInt32Type val(100u);
    EXPECT_EQ(val.Dump(), "100");
}

TEST_F(TestSchemaBase, UInt64TypeDumpSmallValue)
{
    UInt64Type val(999u);
    EXPECT_EQ(val.Dump(), "999");
}

TEST_F(TestSchemaBase, Int64IdTypeRawTensor)
{
    rawTensor rt(5);
    EXPECT_EQ(rt.Dump(), "@5");
}

TEST_F(TestSchemaBase, Int64IdTypeTensor)
{
    tensor t(10);
    EXPECT_EQ(t.Dump(), "%10");
}

TEST_F(TestSchemaBase, Int64IdTypeOperation)
{
    operation op(3);
    EXPECT_EQ(op.Dump(), "!3");
}

TEST_F(TestSchemaBase, AddressTypeDump)
{
    AddressType addr(0x1234ULL);
    std::string dump = addr.Dump();
    EXPECT_NE(dump.find("0x1234"), std::string::npos);
}

TEST_F(TestSchemaBase, StringTypeDump)
{
    StringType s("hello");
    EXPECT_EQ(s.Dump(), "\"hello\"");
    StringType empty("");
    EXPECT_EQ(empty.Dump(), "\"\"");
}

TEST_F(TestSchemaBase, TextTypeDump)
{
    TextType t("raw text");
    EXPECT_EQ(t.Dump(), "raw text");
}

TEST_F(TestSchemaBase, CoordTypeDump)
{
    CoordType c(3, 10L, 20L, 30L);
    std::string dump = c.Dump();
    EXPECT_NE(dump.find("10"), std::string::npos);
    EXPECT_NE(dump.find("20"), std::string::npos);
    EXPECT_NE(dump.find("30"), std::string::npos);
}

TEST_F(TestSchemaBase, ArrayTypeDump)
{
    type::ArrayType<Int64Type> arr;
    arr.dumpIndex_ = false;
    arr.elementList_.push_back(Int64Type(1));
    arr.elementList_.push_back(Int64Type(2));
    arr.elementList_.push_back(Int64Type(3));
    std::string dump = arr.Dump();
    EXPECT_NE(dump.find("1"), std::string::npos);
    EXPECT_NE(dump.find("2"), std::string::npos);
    EXPECT_NE(dump.find("3"), std::string::npos);
}

TEST_F(TestSchemaBase, ArrayTypeWithIndex)
{
    type::ArrayType<Int64Type> arr;
    arr.dumpIndex_ = true;
    arr.elementList_.push_back(Int64Type(100));
    arr.elementList_.push_back(Int64Type(200));
    std::string dump = arr.Dump();
    EXPECT_NE(dump.find("100"), std::string::npos);
}

TEST_F(TestSchemaBase, AttributeIdDumpTopLevel)
{
    type::AttributeId attr("keyword");
    EXPECT_EQ(attr.Dump(true), "#keyword");
    EXPECT_EQ(attr.Dump(false), "keyword");
}

TEST_F(TestSchemaBase, AttributeCall1Dump)
{
    type::AttributeCall_1<Int32Type> call("attr1", Int32Type(42));
    std::string dumpTop = call.Dump(true);
    EXPECT_NE(dumpTop.find("#attr1"), std::string::npos);
    EXPECT_NE(dumpTop.find("42"), std::string::npos);
    std::string dumpNonTop = call.Dump(false);
    EXPECT_NE(dumpNonTop.find("attr1"), std::string::npos);
    EXPECT_EQ(dumpNonTop.find("#attr1"), std::string::npos);
}

TEST_F(TestSchemaBase, AttributeCall2Dump)
{
    type::AttributeCall_2<Int32Type, Int64Type> call("attr2", Int32Type(1), Int64Type(2));
    std::string dump = call.Dump(true);
    EXPECT_NE(dump.find("#attr2"), std::string::npos);
    EXPECT_NE(dump.find("1"), std::string::npos);
    EXPECT_NE(dump.find("2"), std::string::npos);
}

TEST_F(TestSchemaBase, AttributeCall3Dump)
{
    type::AttributeCall_3<Int32Type, Int64Type, UInt32Type> call("attr3", Int32Type(10), Int64Type(20), UInt32Type(30));
    std::string dump = call.Dump(false);
    EXPECT_NE(dump.find("attr3"), std::string::npos);
    EXPECT_NE(dump.find("10"), std::string::npos);
}

TEST_F(TestSchemaBase, UnionTypeConstructFromNone)
{
    none n;
    type::UnionType<none, Int64Type> u(n);
    EXPECT_NE(u.Dump().find("_"), std::string::npos);
}

TEST_F(TestSchemaBase, UnionTypeConstructFromInt64)
{
    type::UnionType<none, Int64Type> u(Int64Type(77));
    std::string dump = u.Dump();
    EXPECT_NE(dump.find("77"), std::string::npos);
}

TEST_F(TestSchemaBase, UnionTypeCopyConstruction)
{
    type::UnionType<none, Int64Type> u1(Int64Type(55));
    type::UnionType<none, Int64Type> u2(u1);
    EXPECT_NE(u2.Dump().find("55"), std::string::npos);
}

TEST_F(TestSchemaBase, UnionTypeCopyAssignment)
{
    type::UnionType<none, Int64Type> u1(Int64Type(99));
    type::UnionType<none, Int64Type> u2{none{}};
    u2 = u1;
    EXPECT_NE(u2.Dump().find("99"), std::string::npos);
}

TEST_F(TestSchemaBase, UnionTypeMultiVariantInt64)
{
    type::UnionType<none, Int64Type, StringType> u(Int64Type(42));
    EXPECT_NE(u.Dump().find("42"), std::string::npos);
}

TEST_F(TestSchemaBase, UnionTypeMultiVariantString)
{
    type::UnionType<none, Int64Type, StringType> u(StringType("test"));
    EXPECT_NE(u.Dump().find("test"), std::string::npos);
}

TEST_F(TestSchemaBase, DumpAttrSingleArg)
{
    Int32Type val(5);
    EXPECT_EQ(DumpAttr(val), "5");
}

TEST_F(TestSchemaBase, DumpAttrMultipleArgs)
{
    Int32Type a(1);
    Int64Type b(2);
    std::string result = DumpAttr(a, b);
    EXPECT_NE(result.find("1"), std::string::npos);
    EXPECT_NE(result.find("2"), std::string::npos);
    EXPECT_NE(result.find(" "), std::string::npos);
}

TEST_F(TestSchemaBase, RangeFactoryUsesAddress)
{
    range r = Range(0x64, 0xC8);
    std::string dump = r.Dump(false);
    EXPECT_NE(dump.find("range"), std::string::npos);
    EXPECT_NE(dump.find("0x"), std::string::npos);
}

TEST_F(TestSchemaBase, shapeListType)
{
    shapeList sl;
    sl.elementList_.push_back(Int64Type(1));
    sl.elementList_.push_back(Int64Type(2));
    sl.elementList_.push_back(Int64Type(3));
    std::string dump = sl.Dump();
    EXPECT_NE(dump.find("1"), std::string::npos);
    EXPECT_NE(dump.find("3"), std::string::npos);
}

TEST_F(TestSchemaBase, AttrTypeFromName)
{
    name n(StringType("myname"));
    AttrType at(n);
    std::string dump = at.Dump();
    EXPECT_NE(dump.find("name"), std::string::npos);
    EXPECT_NE(dump.find("myname"), std::string::npos);
}

TEST_F(TestSchemaBase, AttrTypeFromIncast)
{
    incast ic(Int64Type(42));
    AttrType at(ic);
    std::string dump = at.Dump();
    EXPECT_NE(dump.find("incast"), std::string::npos);
    EXPECT_NE(dump.find("42"), std::string::npos);
}

TEST_F(TestSchemaBase, OperationListType)
{
    OperationList ol;
    ol.elementList_.push_back(operation(1));
    ol.elementList_.push_back(operation(2));
    std::string dump = ol.Dump();
    EXPECT_NE(dump.find("!1"), std::string::npos);
    EXPECT_NE(dump.find("!2"), std::string::npos);
}

TEST_F(TestSchemaBase, AttributeIdDumpCustomKeyword)
{
    type::AttributeId attr("custom_kw");
    EXPECT_EQ(attr.Dump(true), "#custom_kw");
    EXPECT_EQ(attr.Dump(false), "custom_kw");
}

TEST_F(TestSchemaBase, StringTypeEmptyDump)
{
    StringType s;
    EXPECT_EQ(s.Dump(), "\"\"");
}

TEST_F(TestSchemaBase, AddressTypeZero)
{
    AddressType addr(0ULL);
    std::string dump = addr.Dump();
    EXPECT_NE(dump.find("0x0"), std::string::npos);
}

TEST_F(TestSchemaBase, CoordTypeOneDim)
{
    CoordType c(1, 42L);
    std::string dump = c.Dump();
    EXPECT_NE(dump.find("42"), std::string::npos);
}

TEST_F(TestSchemaBase, ArrayTypeEmpty)
{
    type::ArrayType<Int64Type> arr;
    arr.dumpIndex_ = false;
    std::string dump = arr.Dump();
    EXPECT_EQ(dump, "[]");
}

TEST_F(TestSchemaBase, memTypeSmallValue)
{
    memType m(100ULL);
    std::string dump = m.Dump();
    EXPECT_EQ(dump, "100");
}

TEST_F(TestSchemaBase, noneTypeDump)
{
    none n;
    EXPECT_EQ(n.Dump(true), "#_");
    EXPECT_EQ(n.Dump(false), "_");
}
