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
 * \file test_schema_parser.cpp
 * \brief Unit tests for SchemaNode parsing, Dump, and BuildDict
 */

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "gtest/gtest.h"
#include "interface/schema/schema.h"

using namespace npu::tile_fwk::schema;

class TestSchemaParser : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestSchemaParser, ParseKeywordWithArgs)
{
    std::string input = "#trace: #attr{42,99}";
    auto nodes = SchemaNode::ParseSchema(input);
    EXPECT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes[0]->GetName(), "attr");
    EXPECT_EQ(nodes[0]->size(), 2u);
    EXPECT_EQ(nodes[0]->at(0)->GetName(), "42");
    EXPECT_EQ(nodes[0]->at(1)->GetName(), "99");
}

TEST_F(TestSchemaParser, ParseAnonymousArray)
{
    std::string input = "#trace: #data{[1,2,3]}";
    auto nodes = SchemaNode::ParseSchema(input);
    EXPECT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes[0]->GetName(), "data");
    EXPECT_EQ(nodes[0]->size(), 1u);
    EXPECT_EQ(nodes[0]->at(0)->GetName(), "");
    EXPECT_EQ(nodes[0]->at(0)->size(), 3u);
}

TEST_F(TestSchemaParser, ParseNoTracePrefixReturnsEmpty)
{
    std::string input = "no trace here";
    auto nodes = SchemaNode::ParseSchema(input);
    EXPECT_EQ(nodes.size(), 0u);
}

TEST_F(TestSchemaParser, ParseEmptyStringReturnsEmpty)
{
    std::string input = "";
    auto nodes = SchemaNode::ParseSchema(input);
    EXPECT_EQ(nodes.size(), 0u);
}

TEST_F(TestSchemaParser, DumpNamedNodeNoChildren)
{
    SchemaNode node("simple");
    EXPECT_EQ(node.Dump(), "simple");
}

TEST_F(TestSchemaParser, DumpNamedNodeWithChildren)
{
    auto parent = std::make_shared<SchemaNode>("parent");
    auto child1 = std::make_shared<SchemaNode>("c1");
    auto child2 = std::make_shared<SchemaNode>("c2");
    parent->push_back(child1);
    parent->push_back(child2);
    std::string dump = parent->Dump();
    EXPECT_NE(dump.find("parent{c1,c2}"), std::string::npos);
}

TEST_F(TestSchemaParser, DumpAnonymousArray)
{
    auto arr = std::make_shared<SchemaNode>("");
    auto elem1 = std::make_shared<SchemaNode>("x");
    auto elem2 = std::make_shared<SchemaNode>("y");
    arr->push_back(elem1);
    arr->push_back(elem2);
    std::string dump = arr->Dump();
    EXPECT_NE(dump.find("[x,y]"), std::string::npos);
}

TEST_F(TestSchemaParser, GetNameAccessors)
{
    SchemaNode node("testname");
    EXPECT_EQ(node.GetName(), "testname");
    node.GetName() = "changed";
    EXPECT_EQ(node.GetName(), "changed");
}

TEST_F(TestSchemaParser, ParseKeywordWithSingleChild)
{
    std::string input = "#trace: #empty{val}";
    auto nodes = SchemaNode::ParseSchema(input);
    EXPECT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes[0]->GetName(), "empty");
    EXPECT_EQ(nodes[0]->size(), 1u);
    EXPECT_EQ(nodes[0]->at(0)->GetName(), "val");
}

TEST_F(TestSchemaParser, ParseKeywordWithNestedChildren)
{
    std::string input = "#trace: #outer{inner{val}}";
    auto nodes = SchemaNode::ParseSchema(input);
    EXPECT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes[0]->GetName(), "outer");
    EXPECT_EQ(nodes[0]->size(), 1u);
    EXPECT_EQ(nodes[0]->at(0)->GetName(), "inner");
    EXPECT_EQ(nodes[0]->at(0)->size(), 1u);
    EXPECT_EQ(nodes[0]->at(0)->at(0)->GetName(), "val");
}

TEST_F(TestSchemaParser, ParseKeywordWithNestedArray)
{
    std::string input = "#trace: #outer{[1,2,3]}";
    auto nodes = SchemaNode::ParseSchema(input);
    EXPECT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes[0]->GetName(), "outer");
    EXPECT_EQ(nodes[0]->size(), 1u);
    EXPECT_EQ(nodes[0]->at(0)->GetName(), "");
    EXPECT_EQ(nodes[0]->at(0)->size(), 3u);
}

TEST_F(TestSchemaParser, ParseMixedArrayAndKeyword)
{
    std::string input = "#trace: #entry{sub{[a,b]},c}";
    auto nodes = SchemaNode::ParseSchema(input);
    EXPECT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes[0]->GetName(), "entry");
    EXPECT_EQ(nodes[0]->size(), 2u);
    EXPECT_EQ(nodes[0]->at(0)->GetName(), "sub");
    EXPECT_EQ(nodes[0]->at(1)->GetName(), "c");
}

TEST_F(TestSchemaParser, ParseSchemaFromList)
{
    std::vector<std::string> list = {"#trace: #a{x}", "#trace: #b{y}"};
    auto nodes = SchemaNode::ParseSchema(list);
    EXPECT_EQ(nodes.size(), 2u);
    EXPECT_EQ(nodes[0]->GetName(), "a");
    EXPECT_EQ(nodes[1]->GetName(), "b");
}

TEST_F(TestSchemaParser, ParseSchemaListWithEmptyString)
{
    std::vector<std::string> list = {"", "#trace: #a{1}"};
    auto nodes = SchemaNode::ParseSchema(list);
    EXPECT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes[0]->GetName(), "a");
    EXPECT_EQ(nodes[0]->size(), 1u);
}

TEST_F(TestSchemaParser, BuildDictBasic)
{
    std::string input = "#trace: #root{child1,child2}";
    auto nodes = SchemaNode::ParseSchema(input);
    auto dict = SchemaNode::BuildDict(nodes);
    EXPECT_NE(dict.find("root"), dict.end());
    EXPECT_NE(dict.find("child1"), dict.end());
    EXPECT_NE(dict.find("child2"), dict.end());
    EXPECT_EQ(dict["child1"].size(), 1u);
}

TEST_F(TestSchemaParser, BuildDictMultipleOccurrences)
{
    std::string input = "#trace: #outer{inner,inner}";
    auto nodes = SchemaNode::ParseSchema(input);
    auto dict = SchemaNode::BuildDict(nodes);
    EXPECT_EQ(dict["inner"].size(), 2u);
}

TEST_F(TestSchemaParser, DumpRoundTrip)
{
    std::string input = "#trace: #root{x,y}";
    auto nodes = SchemaNode::ParseSchema(input);
    EXPECT_EQ(nodes.size(), 1u);
    std::string dump = nodes[0]->Dump();
    EXPECT_NE(dump.find("root"), std::string::npos);
    EXPECT_NE(dump.find("x"), std::string::npos);
    EXPECT_NE(dump.find("y"), std::string::npos);
}
